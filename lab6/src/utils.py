
import os
    
def args_parser():
    import argparse
    parser = argparse.ArgumentParser(description="MaskGIT")
    parser.add_argument('--dr', type=str, default="../iclevr/", help='Training Dataset Path')
    parser.add_argument('--train-json', type=str, default="train.json", help='Training JSON file')
    parser.add_argument('--test-json', type=str, default="new_test.json", help='Testing JSON file')
    parser.add_argument('--object-json', type=str, default="objects.json", help='Object JSON file')
    parser.add_argument('--ckpt-dir', type=str, default='./checkpoints/', help='Path to checkpoint folder')
    parser.add_argument('--ckpt-path', type=str, default='./checkpoints/DL_lab6_313551097_鄭淮薰.pth', help='Path to checkpoint')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: all)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=5, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--config', type=str, default='config/DDPM.yml', help='Configurations for DDPM model.')
    
    # log
    parser.add_argument('--log', action='store_true', help='Use wandb and tensorboard for logging')
    parser.add_argument('--resume-path', type=str, default=None, help='Path to resume training')
    
    args = parser.parse_args()
    
    return args

def init_logging(project, config, resume_path=None):
    import wandb
    from torch.utils.tensorboard import SummaryWriter

    if not resume_path:
        run_id = wandb.util.generate_id()
        config.run_id = run_id

    # Wandb
    wandb.init(project=project, config=config, id=config.run_id, resume='allow', sync_tensorboard=True)
    config.run_name = wandb.run.name

    # Tensorboard
    logdir = f"runs/{config.run_name}-{config.run_id}"
    writer = SummaryWriter(logdir)

    # wandb.tensorboard.patch(root_logdir=logdir)
    return config, writer

def save_model_to_wandb(model, tmp_dir):
    try:
        import wandb
        import torch
        run_id = wandb.run.id
        run_name = wandb.run.name
        model_name = f"/{run_name}-{run_id}-{model_name}.pt"
        os.makedirs(f"{tmp_dir}/models", exist_ok=True)
        torch.save(model.state_dict(), f"tmp_{run_name}/models/{model_name}")
        wandb.save(
            os.path.abspath(f"{tmp_dir}/models/{model_name}"),
            base_path=os.path.abspath(tmp_dir)
        )
        print(f"Saved {model_name} to wandb")
    except Exception as e:
        print(f"Failed to save models to wandb: {e}")
        
    return tmp_dir
    
    
def show_images(images, title="", save_path="images.png", nrow=8, denoising_process=False):
    import torch
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    if isinstance(images, list):
        images = torch.stack(images).cpu()
        
    # (-1, 1) -> (0, 1)
    images_tensor = images * 0.5 + 0.5

    # set nrow for denoising process
    if denoising_process:
        nrow = len(images_tensor)

    # generate grid of images
    grid = make_grid(images_tensor, nrow=nrow, padding=2)

    # show grids
    plt.clf()
    plt.figure(figsize=(15, 2) if denoising_process else (15, 10))
    # Convert from CHW to HWC
    plt.imshow(grid.permute(1, 2, 0).clip(0, 1))  
    plt.axis('off')
    plt.margins(0, 0)
    plt.title(title if title else ('Denoising Process (Noisy to Clear)' if denoising_process else 'Image Grid'))
    plt.savefig(save_path)