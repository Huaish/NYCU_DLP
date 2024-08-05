import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10
from torch.utils.tensorboard import SummaryWriter
import datetime

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.iter = current_epoch
        self.n_iter = args.num_epoch
        self.beta = 1 if args.kl_anneal_type == 'None' else 0
        self.beta_start = 0.
        self.beta_end = 1.
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio
        
    def update(self):
        self.iter += 1
        if self.kl_anneal_type == 'Cyclical':
            self.beta = self.frange_cycle_linear(self.iter, self.n_iter, start=self.beta_start, stop=self.beta_end, n_cycle=self.kl_anneal_cycle, ratio=self.kl_anneal_ratio)
        elif self.kl_anneal_type == 'Monotonic':
            self.beta = self.frange_cycle_linear(self.iter, self.n_iter, start=self.beta_start, stop=self.beta_end, n_cycle=1, ratio=self.kl_anneal_ratio)
        else:
            self.beta = 1
        
    
    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self, iter, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1.):
        """Linearly anneal the value from start to stop in n_cycle period

        Args:
            iter (int): Current iteration
            n_iter (int): Total number of iteration
            start (float, optional): Starting value of the sequence. Defaults to 0.0.
            stop (float, optional): Stopping value of the sequence. Defaults to 1.0.
            n_cycle (int, optional): Number of cycles. Defaults to 1.
            ratio (float, optional): Ratio of increasing phase to total cycle length. Defaults to 1.

        Returns:
            float: Annealed value
        """
        
        cycle_length = n_iter // n_cycle
        step_in_cycle = iter % cycle_length

        if step_in_cycle < cycle_length * ratio:
            return max(1e-6, start + (stop - start) * step_in_cycle / (cycle_length * ratio))
        else:
            return stop


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        # Initialize the tensorboard
        self.writer = SummaryWriter(args.tensorboard_path)

    def forward(self, img, label):
        pass
    
    def training_stage(self):
        for i in range(self.current_epoch, self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            # Logging the loss to tensorboard
            self.writer.add_scalar('Loss/train', loss, self.current_epoch)
            self.writer.add_scalar('TFR', self.tfr, self.current_epoch)
            self.writer.add_scalar('Beta', beta, self.current_epoch)
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            self.writer.add_scalar('Loss/val', loss, self.current_epoch)
    
    def training_one_step(self, batch_images, batch_labels, adapt_TeacherForcing):
        """Training one step of the model

        Args:
            batch_images (torch.Tensor): Input image (B, T, C, H, W) => (2, 16, 3, 256, 256)
            batch_labels (torch.Tensor): Input label (Batch size, Time step, Channel, Height, Width)
            adapt_TeacherForcing (bool): Whether to adapt teacher forcing strategy

        Returns:
            torch.Tensor: Loss value
        """
        
        beta = self.kl_annealing.get_beta()
        total_loss = 0
        
        for (images, labels) in (zip(batch_images, batch_labels)):
            mse_loss, kl_loss = 0., 0.
            
            # Take the first frame as the initial last frame
            last_frame = images[0, :, :, :].unsqueeze(0)
            for i in range(1, self.train_vi_len):
                current_frame = images[i, :, :, :].unsqueeze(0)
                current_label = labels[i, :, :, :].unsqueeze(0)
                
                # Transform the image from RGB-domain to feature-domain
                last_frame_feature = self.frame_transformation(last_frame)
                current_frame_feature = self.frame_transformation(current_frame)
                current_label_feature = self.label_transformation(current_label)
                
                # Conduct Posterior prediction in Encoder
                z, mu, logvar = self.Gaussian_Predictor(current_frame_feature, current_label_feature)

                # Decoder Fusion
                output = self.Decoder_Fusion(last_frame_feature, current_label_feature, z)

                # Generative model
                generated_frame = self.Generator(output)

                # Compute loss
                mse_loss += self.mse_criterion(generated_frame, current_frame)
                kl_loss += kl_criterion(mu, logvar, self.batch_size)

                # Update the last frame with teacher forcing strategy
                if adapt_TeacherForcing:
                    last_frame = current_frame
                else:
                    last_frame = generated_frame

            # Compute one loss of the mini-batch
            loss = mse_loss + beta * kl_loss
            total_loss += loss

            # Backward
            self.optim.zero_grad()
            loss.backward()
            self.optimizer_step()

        return total_loss / len(batch_images)
    
    def val_one_step(self, batch_images, batch_labels):
        """Validation one step of the model

        Args:
            batch_images (torch.Tensor): Input image (B, T, C, H, W)
            batch_labels (torch.Tensor): Input label (B, T, C, H, W)

        Returns:
            torch.Tensor: Avg. loss value
        """

        total_loss = 0.
        psnr = []

        beta = self.kl_annealing.get_beta()
        for images, labels in zip(batch_images, batch_labels):
            mse_loss, kl_loss = 0., 0.
            
            # Take the first frame as the initial last frame
            last_frame = images[0, :, :, :].unsqueeze(0)
            for i in range(1, self.val_vi_len):
                current_frame = images[i, :, :, :].unsqueeze(0)
                current_label = labels[i, :, :].unsqueeze(0)
                
                # Transform the image from RGB-domain to feature-domain
                last_frame_features = self.frame_transformation(last_frame)
                current_frame_features = self.frame_transformation(current_frame)
                current_label_features = self.label_transformation(current_label)
                
                # Conduct Posterior prediction in Encoder
                z, mu, logvar = self.Gaussian_Predictor(current_frame_features, current_label_features)
                
                # Decoder Fusion
                output = self.Decoder_Fusion(last_frame_features, current_label_features, z)
                
                # Generative model
                generated_frame = self.Generator(output)
                
                # Compute loss
                mse_loss += self.mse_criterion(generated_frame, current_frame)
                kl_loss += kl_criterion(mu, logvar, self.batch_size)
                psnr.append(Generate_PSNR(generated_frame, current_frame).cpu())
                
                # Update the last frame
                last_frame = generated_frame
                
                # Logging the generated frame and PSNR to tensorboard
                self.writer.add_image(f'Generated Frame/epoch-{self.current_epoch}', generated_frame.squeeze(0), i, dataformats='CHW')
                self.writer.add_scalar(f'PSNR/epoch-{self.current_epoch}', Generate_PSNR(generated_frame, current_frame), i)
                
            # Compute one loss of the mini-batch
            loss = mse_loss + beta * kl_loss
            total_loss += loss

        # Plot the PSNR per frame
        self.plot_PSNR_per_frame(psnr)
        return total_loss / len(batch_images)
        
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.tfr_sde:
            self.tfr = max(0, self.tfr - self.tfr_d_step)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()

    def plot_PSNR_per_frame(self, psnr_list):
        avg_psnr = np.mean(psnr_list)
        plt.plot(psnr_list, label='Avg_PSNR: {:.3f}'.format(avg_psnr))
        plt.legend()
        plt.xlabel('Frame index')
        plt.ylabel('PSNR')
        plt.title('Per frame Quality(PSNR)')
        plt.savefig('PSNR_per_frame.png')
        plt.close()

def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu", "cuda:1"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=400,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical', choices=['Cyclical', 'Monotonic', 'None'], help="KL annealing strategy")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.5,              help="")
    
    # Tensorboard arguments
    parser.add_argument('--tensorboard',        default=True, action='store_true', help="Use tensorboard to visualize the training process")
    parser.add_argument('--tensorboard_path',   type=str, default=None,        help="The path to save the tensorboard logs")

    args = parser.parse_args()
    
    if args.tensorboard:
        if args.tensorboard_path == None:
            args.tensorboard_path = f"../runs/{args.kl_anneal_type}-tfr_{args.tfr}_{args.tfr_sde}_{args.tfr_d_step}"
    
    main(args)
    
    # Test the kl_annealing class
    # writer = SummaryWriter(f"../runs/kl_annealing_test")
    # kl = kl_annealing(args)
    # for i in range(args.num_epoch):
    #     writer.add_scalar('beta', kl.get_beta(), i)
    #     print(f"{i}: {kl.get_beta()}")
    #     kl.update()
    # writer.close()
