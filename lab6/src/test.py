import os
from tqdm import tqdm
import torch
import torch.nn as nn
from dataloader import LoadTestData
import yaml
from torch.utils.data import DataLoader
from models.ddpm import ConditionalDDPM
from utils import args_parser, show_images, set_seed
from diffusers import DDPMScheduler
from evaluator import evaluation_model

def load_checkpoint(ckpt_path, DDPM_CONFIGS, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ConditionalDDPM(**DDPM_CONFIGS['model_param']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

@torch.no_grad()
def inference(model, test_loader, DDPM_CONFIGS, device, test_json=""):
    model.eval()
    noise_schedule = DDPMScheduler(**DDPM_CONFIGS['noise_schedule'])
    evaluator = evaluation_model()
    total_acc = 0
    results = torch.empty(0, 3, 64, 64)

    for i, label in (pbar := tqdm(enumerate(test_loader), total=len(test_loader))):
        label = label.to(device)
        
        # sample noisy images
        sample = torch.randn(label.shape[0], 3, 64, 64).to(device)
        denoising_images = []
        for step, timesteps in enumerate(noise_schedule.timesteps):
            noise_pred = model(sample, timesteps, label)
            sample = noise_schedule.step(noise_pred, timesteps, sample).prev_sample
            
            if (step+1) % 100 == 0:
                denoising_images.append(sample)
                
        # compute accuracy
        acc = evaluator.eval(sample, label)
        total_acc += acc
        results = torch.cat([results, sample.cpu()], dim=0)

        # show denoising process
        if i < 5:
            show_images(denoising_images, title=f"Denoising process image {i+1}", save_path=f"{test_json}-images{i+1}.png", denoising_process=True)
        
        # update progress bar
        pbar.set_description(f"(test) Accuracy: {acc:.4f}")

    # show synthetic images grid
    acc = total_acc / len(test_loader)
    results = torch.cat(results, dim=0)
    show_images(results, title=f"The synthetic image grid on {test_json}.json. (Acc {acc:.4f})", save_path=f"{test_json}-images-grid.png")
    return total_acc / len(test_loader), results

if __name__ == '__main__':
    args = args_parser()
    set_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = LoadTestData(root=args.dr, test_json=args.test_json, object_json=args.object_json)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False)
    print(f"Load {args.test_json} dataset with {len(test_dataset)} labels")

    DDPM_CONFIGS = yaml.safe_load(open(args.config, 'r'))
    model = load_checkpoint(args.ckpt_path, DDPM_CONFIGS, args.device)

    acc, results = inference(model, test_loader, DDPM_CONFIGS, args.device, test_json=os.path.splitext(os.path.basename(args.test_json))[0])
    print(f"Accuracy: {acc:.4f}")