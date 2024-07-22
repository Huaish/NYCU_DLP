# implement your training script here

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import termcolor
from torch.utils.data import DataLoader
from Dataloader import MIBCI2aDataset
from model.SCCNet import SCCNet
from datetime import datetime
import numpy as np
import random
import argparse
from utils import set_random_seed, get_args


def train(model, device, train_loader, optimizer, criterion, pbar=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()
        
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Update process bar
        pbar.update(1)

    accuracy = 100. * correct / total
    return running_loss / len(train_loader), accuracy

def LOSO_train(args):
    print(termcolor.colored("LOSO Training", "blue"))
    
    # Set Hyper-Parameter
    batch_size = args.batch_size # 300
    learning_rate = args.lr # 0.00055
    epochs = args.epochs # 25
    device = args.device

    # Load Data
    train_dataset = MIBCI2aDataset(mode='train', method='LOSO')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create Mode
    model = SCCNet(numClasses=4, timeSample=438, Nu=22, C=22, Nc=22, Nt=16, dropoutRate=0.5).to(device)

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # training
    train_loss, train_acc = 0., 0.
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=True) as pbar:
            train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, pbar)
            train_losses.append(train_loss)
            pbar.set_postfix({'Loss': f'{train_loss:.6f}', 'Accuracy': f'{train_acc:.2f}%'})

        for data, target in train_loader:
            pass

    # Save Model
    torch.save(model.state_dict(), "LOSO_latest_model.pt")
    print('LOSO Latest Model saved in LOSO_latest_model.pt with train accuracy: {:.2f}%'.format(train_acc))

    return model, train_acc

def LOSO_FT(model, args):
    print(termcolor.colored("Fine-Tuning Training", "blue"))

    # Set Hyper-Parameter
    batch_size = args.batch_size # 300
    learning_rate = args.lr # 0.00055
    epochs = args.ft_epochs # 50
    device = args.device


    # Load Finetune Data
    FT_dataset = MIBCI2aDataset(mode='finetune', method='LOSOFT')
    FT_loader = DataLoader(FT_dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Fine-Tuning
    for epoch in range(epochs):
        with tqdm(total=len(FT_loader), desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=True) as pbar:
            train_loss, train_acc = train(model, device, FT_loader, optimizer, criterion, pbar)
            pbar.set_postfix({'Loss': f'{train_loss:.6f}', 'Accuracy': f'{train_acc:.2f}%'})

    # Save Model
    torch.save(model.state_dict(), "FT_latest_model.pt")
    print('Latest Model saved with accuracy: {:.2f}%'.format(train_acc))

    return model, train_acc

def SD_Train(args):
    print(termcolor.colored("SD Training", "blue"))
    
    # Set Hyper-Parameter
    batch_size = args.batch_size # 300
    learning_rate = args.lr # 0.0005
    epochs = args.epochs # 350
    device = args.device

    # Load Data
    train_dataset = MIBCI2aDataset(mode='train', method='SD')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create Mode
    model = SCCNet(numClasses=4, timeSample=438, Nu=22, C=22, Nc=22, Nt=16, dropoutRate=0.5).to(device)

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # training
    for epoch in range(epochs):
        # LOSO Training
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=True) as pbar:
            train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, pbar)
            pbar.set_postfix({'Loss': f'{train_loss:.6f}', 'Accuracy': f'{train_acc:.2f}%'})

    # Save Model
    torch.save(model.state_dict(), "SD_latest_model.pt")
    print('SD Latest Model saved with train accuracy: {:.2f}%'.format(train_acc))

    return model, train_acc


if __name__ == '__main__':

    args = get_args()
    print(args)
    set_random_seed(args.seed)
    
    if args.method == 'SD':
        model, train_acc = SD_Train(args)

    elif args.method == 'LOSO':
        model, train_acc = LOSO_train(args)

    elif args.method == 'LOSOFT':
        model, train_acc = LOSO_train(args)
        model, train_acc = LOSO_FT(model, args)

    # Save Model
    if model and args.save_model:
        torch.save(model.state_dict(), "latest_model.pt")
        print('Latest Model saved with accuracy: {:.2f}%'.format(train_acc))

# python trainer.py --method "SD" --epochs=350
# python trainer.py --method "LOSO" --epochs=25 --lr=0.00055
# python trainer.py --method "LOSOFT" --epochs=25 --ft_epochs=50 --lr=0.00055