import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import termcolor
from oxford_pet import load_dataset
from models.unet import UNet
from evaluate import evaluate

def train(args):
    data_path = args.data_path
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "train"
    
    # Load the data
    print(f"Loading data from {data_path}")
    dataset = load_dataset(data_path, mode)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model, loss function and optimizer
    model = UNet(in_channels=3, out_channels=2).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    model.train()
    train_loss = 0.0
    for epoch in range(epochs):
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False) as pbar:
            for i, batch in enumerate(dataloader):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                pbar.update(1)
            pbar.set_postfix({"Loss": f"{train_loss:.6f}"})
            
        print(f"Loss: {train_loss}")
        
    # Evaluate the model
    mode = "valid"
    dataloader = load_dataset(data_path, mode)
    dice = evaluate(model, dataloader, device)
    print(f"Dice score: {dice}")
    
    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet", help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=10, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)