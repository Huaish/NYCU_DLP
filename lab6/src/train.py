import os
from tqdm import tqdm
import torch
import torch.nn as nn
from dataloader import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from models.ddpm import ConditionalDDPM
from utils import args_parser, init_logging, save_model_to_wandb
from diffusers import DDPMScheduler

class TrainDDPM:
    def __init__(self, args, DDPM_CONFIGS):
        args.run_id = torch.randint(0, 100000, (1,)).item()
        args.run_name = "DDPM-no-log"
        self.args = args
        self.model = ConditionalDDPM(**DDPM_CONFIGS['model_param']).to(args.device)
        self.noise_schedule = DDPMScheduler(**DDPM_CONFIGS['noise_schedule'])
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.args.current_epoch = 0

        # log
        if args.log:
            import wandb
            self.args, self.writer = init_logging("DDPM", args)
            self.tmp_dir = f"tmp_{self.args.run_name}"
            wandb.watch(self.model)
            
        # create checkpoint directory
        os.makedirs(f"{self.args.ckpt_dir}/{self.args.run_name}-{self.args.run_id}", exist_ok=True)
            
    def train(self, train_loader):
        for epoch in range(self.args.start_from_epoch+1, self.args.epochs+1):
            self.train_one_epoch(train_loader, epoch)
            
            if epoch % self.args.save_per_epoch == 0:
                self.save_checkpoint()
        
        if self.args.log:
            save_model_to_wandb(self.model, self.tmp_dir)
        self.finish_training()

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        
        total_loss = 0.0
        for step, (image, label) in (pbar := tqdm(enumerate(train_loader), total=len(train_loader))):
            image, label = image.to(self.args.device), label.to(self.args.device)
            
            # sample noise
            noise = torch.randn_like(image).to(self.args.device)
            
            # sample timesteps
            timesteps = torch.randint(0, self.noise_schedule.config.num_train_timesteps, (image.shape[0],)).to(self.args.device)
            
            # add noise to image
            noisy_image = self.noise_schedule.add_noise(image, noise, timesteps)
            
            # forward pass
            noise_pred = self.model(noisy_image, timesteps, label)
            
            # compute loss
            loss = self.loss_fn(noise_pred, noise)
            total_loss += loss.item()
            
            # backprop
            loss.backward()
            
            if step % self.args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()
            
            # update progress bar
            pbar.set_description(f"(train) Epoch {epoch} - Loss: {loss.item():.4f}", refresh=False)
            
        if self.args.log:
            import wandb
            self.writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
            wandb.log({"Loss/train": total_loss / len(train_loader)}, step=epoch)
        return total_loss / len(train_loader)
 
    def save_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = f"epoch_{self.args.epoch}.pt"

        # save checkpoint
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'args': self.args
        }, os.path.join(self.args.ckpt_dir, f"{self.args.run_name}-{self.args.run_id}", checkpoint_path))
        
        print(f"Saved model checkpoint at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer'])
        self.args.learning_rate = checkpoint['args'].learning_rate
        self.args.run_id = checkpoint['args'].run_id
        self.args.start_from_epoch = checkpoint['args'].current_epoch
        print(f"{checkpoint['args'].run_id} loaded from {checkpoint_path}")
           
    def finish_training(self):
        if self.args.log:
            import wandb
            self.writer.close()
            wandb.finish()
        os.system(f"rm -r {self.tmp_dir}")


if __name__ == '__main__':
    args = args_parser()
    DDPM_CONFIGS = yaml.safe_load(open(args.config, 'r'))
    
    train_dataset = LoadTrainData(root=args.dr, train_json=args.train_json, object_json=args.object_json)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    train_ddmp = TrainDDPM(args, DDPM_CONFIGS)

    if args.resume_path:
        train_ddmp.load_checkpoint(args.resume_path)
        
    train_ddmp.train(train_loader)