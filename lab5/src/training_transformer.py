import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb

#[x] TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
        if self.args.log:
            # Wandb
            run_id = wandb.util.generate_id()
            self.args.run_id = run_id
            wandb.init(project='MaskGit', config=self.args, id=run_id, resume='allow')

            # Tensorboard
            self.writer = SummaryWriter(f"runs/{run_id}")
            
            os.makedirs(os.path.join("checkpoints", run_id), exist_ok=True)
            os.makedirs(os.path.join("transformer_checkpoints", run_id), exist_ok=True)
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader):
        self.model.train()
        
        total_loss = 0.0
        for i, image in (pbar := tqdm(enumerate(train_loader), total=len(train_loader))):
            x = image.to(self.args.device)
            logits, z_indices = self.model(x)
            
            # compute loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))
            total_loss += loss.item()
            
            # backprop
            loss.backward()
            
            if i % self.args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()
            
            # update progress bar
            pbar.set_description(f"(train) Epoch {epoch} - Loss: {loss.item():.4f}", refresh=False)
            
        if self.args.log:
            self.writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
            wandb.log({"Loss/train": total_loss / len(train_loader)})
        return total_loss / len(train_loader)
            

    @torch.no_grad()
    def eval_one_epoch(self, val_loader):
        self.model.eval()
        
        total_loss = 0.0
        for i, image in (pbar := tqdm(enumerate(val_loader), total=len(val_loader))):
            x = image.to(args.device)
            logits, z_indices = self.model(x)
            
            # compute loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))
            total_loss += loss.item()
            
            # update progress bar
            pbar.set_description(f"(val) Epoch {epoch} - Loss: {loss.item():.4f}", refresh=False)
        
        if self.args.log:
            self.writer.add_scalar("Loss/val", total_loss / len(val_loader), epoch)
            wandb.log({"Loss/val": total_loss / len(val_loader)})
        return total_loss / len(val_loader)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, betas=(0.9, 0.96))
        scheduler = None
        return optimizer,scheduler
    
    def save_checkpoint(self, epoch, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = f"epoch_{epoch}.pt"
            
        # save transformer checkpoint
        torch.save(self.model.transformer.state_dict(), os.path.join("transformer_checkpoints", self.args.run_id, checkpoint_path))
        print(f"Saved transformer checkpoint at {checkpoint_path}")
        
        # save model checkpoint
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'last_epoch': epoch,
            'args': self.args
        }, os.path.join(self.args.checkpoint_path, self.args.run_id, checkpoint_path))
        
        print(f"Saved model checkpoint at {checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #[x] TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=5, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')
    
    # log
    parser.add_argument('--log', action='store_false', help='Use tensorboard for logging')
    parser.add_argument('--run-id', type=str, default="", help='Run ID for wandb')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#[x] TODO2 step1-5:
    best_val_loss = float('inf')
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader)
        val_loss = train_transformer.eval_one_epoch(val_loader)
        
        if epoch % args.save_per_epoch == 0:
            train_transformer.save_checkpoint(epoch)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            train_transformer.save_checkpoint(epoch, "best_model.pt")