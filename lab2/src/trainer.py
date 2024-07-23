# implement your training script here

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import termcolor
from torch.utils.data import DataLoader
from Dataloader import MIBCI2aDataset
from model.SCCNet import SCCNet
import numpy as np
from utils import set_random_seed, get_args


def train(model, device, train_loader, optimizer, criterion, epochs):
    history = {'loss': [], 'accuracy': []}
    for epoch in range(epochs):
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False) as pbar:
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for data, target in train_loader:
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
                
                pbar.update(1)

            train_acc = 100. * correct / total
            train_loss = running_loss / len(train_loader)

            pbar.set_postfix({'Loss': f'{train_loss:.6f}', 'Accuracy': f'{train_acc:.2f}%'})
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
        
        for data, target in train_loader:
            pass

    return history


if __name__ == '__main__':

    args = get_args()
    print(args)
    set_random_seed(args.seed)

    # Set Hyper-Parameter
    batch_size = args.batch_size # 300
    learning_rate = args.lr # 0.0005
    epochs = args.epochs # 350
    device = args.device
    method = args.method

    assert method in ['SD', 'LOSO', 'LOSOFT'], "Invalid method. Choose from ['SD', 'LOSO', 'LOSOFT']"

    print(termcolor.colored(f"{method} Training", "blue"))

    # Load Data
    train_dataset = MIBCI2aDataset(mode='train', method=method)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create Mode
    model = SCCNet(numClasses=4, timeSample=438, Nu=22, C=22, Nc=22, Nt=16, dropoutRate=0.5).to(device)

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Training
    history = train(model, device, train_loader, optimizer, criterion, epochs)

    if args.method == 'LOSOFT':
        FT_dataset = MIBCI2aDataset(mode='finetune', method='LOSOFT')
        FT_loader = DataLoader(FT_dataset, batch_size=batch_size, shuffle=True)
        history = train(model, device, FT_loader, optimizer, criterion, args.ft_epochs)

    train_acc = history['accuracy'][-1]
    np.save(f'{method}_history.npy', history)

    # Save Model
    if model and args.save_model:
        torch.save(model.state_dict(), f"{method}_latest_model.pt")
        torch.save(model.state_dict(), "latest_model.pt")
        print('Latest Model saved with accuracy: {:.2f}%'.format(train_acc))


# python trainer.py --method "SD" --epochs=400
# python trainer.py --method "LOSO" --epochs=25
# python trainer.py --method "LOSOFT" --epochs=25 --ft_epochs=50