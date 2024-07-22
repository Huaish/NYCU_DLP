# implement your training script here

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import termcolor
from torch.utils.data import DataLoader, Subset
from Dataloader import MIBCI2aDataset
from model.SCCNet import SCCNet
from datetime import datetime
import numpy as np
import random
from tester import test
import matplotlib.pyplot as plt

# Function to set random seed
def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, device, train_loader, optimizer, criterion, pbar):
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
        
        # 使用 torch.max 找到每个样本中预测概率最高的类别
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Update process bar
        pbar.update(1)

    accuracy = 100. * correct / total
    return running_loss / len(train_loader), accuracy


def main(lr=0.001):
    # Set Hyper-Parameter
    batch_size = 16
    learning_rate = lr
    epochs = 1000
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load Data
    train_dataset = MIBCI2aDataset(mode='train', method='SD')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = MIBCI2aDataset(mode='test', method='SD')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Create Mode
    model = SCCNet(numClasses=4, timeSample=438, Nu=22, C=22, Nc=22, Nt=1, dropoutRate=0.5).to(device)

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # training
    train_loss, train_acc = 0., 0.
    epoch = 0
    train_losses = []
    val_losses = []
    while True:
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=True) as pbar:
            train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, pbar)
            train_losses.append(train_loss)
            pbar.set_postfix({'Loss': f'{train_loss:.6f}', 'Accuracy': f'{train_acc:.2f}%'})

        val_loss, val_acc = test(model, device, test_loader, criterion)
        val_losses.append(val_loss)
        
        print(termcolor.colored(f'Test Loss: {val_acc:.4f}, ', "red"), termcolor.colored(f'Accuracy: {val_acc:.2f}%', "green"))
        if val_acc > 60.:
            break

        epoch += 1
        if epoch > epochs:
            print("Best model not found")
            break
        
        # 繪製損失曲線
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('loss_plot.png')


    print(termcolor.colored(f'Train Loss: {train_loss:.4f}, ', "red"), termcolor.colored(f'Accuracy: {train_acc:.2f}%', "green"))

    # Save Model
    torch.save(model.state_dict(), "latest_model.pt")
    print('Latest Model saved with accuracy: {:.2f}%'.format(train_acc))
    
    # # 繪製損失曲線
    # plt.figure()
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Val Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.savefig('loss_plot.png')

    return model, train_acc

if __name__ == '__main__':
    set_random_seed(42)
    for lr in [0.00008]:
        print(f"lr: {lr}")
        model, acc = main(lr)
        # torch.save(model.state_dict(), f"model_{lr}_{acc}.pt")