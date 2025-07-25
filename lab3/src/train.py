import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import termcolor
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils import dice_score

def train(args):
    data_path = args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    print(f"Using device {device}")
    best_score = 0.0
    model_name = args.model
    
    # Load the data
    print(f"Loading data from {data_path}")
    train_dataset = load_dataset(data_path, "train")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = load_dataset(data_path, "valid")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model, loss function and optimizer
    print(f"Using model {model_name}")
    model = None
    if model_name == "UNet":
        model = UNet(in_channels=3, out_channels=2).to(device)
    elif model_name == "ResNet34UNet":
        model = ResNet34_UNet(in_channels=3, out_channels=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize tensorboard
    writer = SummaryWriter(f"../runs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
    # Train the model
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=True) as pbar:
            for i, batch in enumerate(train_dataloader):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                optimizer.zero_grad()
                output = model(images)

                # Compute loss
                loss = criterion(output, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Compute dice score
                train_dice_score = dice_score(output, masks)
                train_dice += train_dice_score

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{train_loss/(i+1):.4f}", "Dice": f"{train_dice_score:.4f}"})

        train_loss /= len(train_dataloader)
        train_dice /= len(train_dataloader)

        # Evaluate the model
        val_dice, val_loss = evaluate(model, val_dataloader, device)
        print(termcolor.colored(f"Train Dice: {train_dice:.4f}, Train Loss: {train_loss:.4f}", "blue"))
        print(termcolor.colored(f"Val Dice: {val_dice:.4f}, Val Loss: {val_loss:.4f}", "green"))

        # Save checkpoint
        torch.save(model.state_dict(), f"../saved_models/{model_name}_latest.pth")
        if val_dice > best_score:
            best_score = val_dice
            torch.save(model.state_dict(), f"../saved_models/{model_name}_best.pth")
    
        # Log the dice score to tensorboard
        writer.add_scalars(f"Dice Score/{model_name}", {"train": train_dice, "val": val_dice}, epoch)
        writer.add_scalars(f"Loss/{model_name}", {"train": train_loss, "val": val_loss}, epoch)
        
        # Early stopping
        if val_dice > 0.99:
            print(termcolor.colored("Dice score > 0.99, stopping training", "yellow"))
            break
    
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet/ResNet34UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet", help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=20, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--model', default='UNet', help='model name')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)
    
# python train.py --model UNet
# python train.py --model ResNet34UNet