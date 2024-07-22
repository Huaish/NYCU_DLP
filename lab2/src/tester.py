# implement your testing script here
import torch
import torch.nn as nn
import termcolor
from model.SCCNet import SCCNet
from torch.utils.data import DataLoader
from Dataloader import MIBCI2aDataset

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100. * correct / total
    return test_loss / len(test_loader), accuracy

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Load Data
    test_dataset = MIBCI2aDataset(mode='test', method='LOSOFT')
    test_loader = DataLoader(test_dataset, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    
    model = SCCNet(numClasses=4, timeSample=438, Nu=22, C=22, Nc=22, Nt=1, dropoutRate=0.5).to(device)
    model.load_state_dict(torch.load("latest_model.pt"))
    test_loss, accuracy = test(model, device, test_loader, criterion)
    print(termcolor.colored(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%', "green"))
    