import os
from tqdm import tqdm
import torch
import torch.nn as nn
from dataloader import LoadTestData
import yaml
from torch.utils.data import DataLoader
from models.ddpm import ConditionalDDPM
from utils import args_parser, show_images
from diffusers import DDPMScheduler
from evaluator import evaluation_model

def load_checkpoint(ckpt_path, DDPM_CONFIGS, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ConditionalDDPM(**DDPM_CONFIGS['model_param']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

@torch.no_grad()
def inference(model, test_loader, DDPM_CONFIGS, device):
    model.eval()
    noise_schedule = DDPMScheduler(**DDPM_CONFIGS['noise_schedule'])
    evaluator = evaluation_model()
    total_acc = 0
    results = []
    
    for i, label in (pbar := tqdm(enumerate(test_loader), total=len(test_loader))):
        label = label.to(device)
        
        # sample noisy images
        sample = torch.randn(label.shape[0], 3, 64, 64).to(device)
        denoising_images = []
        for step, timesteps in enumerate(noise_schedule.timesteps):
            noise_pred = model(sample, timesteps, label)
            sample = noise_schedule.step(noise_pred, timesteps, sample).prev_sample
            
            if step % 100 == 0:
                denoising_images.append(sample.squeeze(0))
                
        # compute accuracy
        acc = evaluator.eval(denoising_images[-1].unsqueeze(0), label)
        total_acc += acc
        results.append(denoising_images[-1].squeeze(0))

        # show denoising process
        if i < 5:
            show_images(denoising_images, title=f"Image {i}", save_path=f"images_{i}.png", denoising_process=True)
        
        # update progress bar
        pbar.set_description(f"(test) Accuracy: {acc:.4f}")
        
    show_images(results, title="Final", save_path="images_final.png")
    return total_acc / len(test_loader), results

if __name__ == '__main__':
    args = args_parser()
    
    test_dataset = LoadTestData(root=args.dr, test_json=args.test_json, object_json=args.object_json)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=True)
    print(f"Test dataset loaded with {len(test_dataset)} samples.")

    DDPM_CONFIGS = yaml.safe_load(open(args.config, 'r'))
    model = load_checkpoint(args.ckpt_path, DDPM_CONFIGS, args.device)

    acc, results = inference(model, test_loader, DDPM_CONFIGS, args.device)
    print(f"Accuracy: {acc:.4f}")