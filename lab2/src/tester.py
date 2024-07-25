# implement your testing script here
import torch
import torch.nn as nn
import termcolor
from model.SCCNet import SCCNet
from torch.utils.data import DataLoader
from Dataloader import MIBCI2aDataset
from utils import get_args



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
    args = get_args()

    # Load Data
    test_dataset = MIBCI2aDataset(mode='test', method=args.method)
    test_loader = DataLoader(test_dataset, shuffle=False)
    
    # Define Loss Function
    criterion = nn.CrossEntropyLoss()
    
    # Load model
    print(termcolor.colored(f"Testing {args.model_path}", "blue"))
    model = SCCNet(numClasses=4, timeSample=438, Nu=22, C=22, Nc=22, Nt=16, dropoutRate=0.5).to(args.device)
    model.load_state_dict(torch.load(args.model_path))
    
    # Show the results
    test_loss, accuracy = test(model, args.device, test_loader, criterion)
    print(termcolor.colored(f'Test Loss: {test_loss:.4f}, \
                            Accuracy: {accuracy:.2f}%', "green"))
    
# python tester.py --method='LOSO'
# python tester.py --method='SD'
# python tester.py --method='LOSOFT'
# python tester.py --method='SD' --model_path="SD_model.pth"
# python tester.py --method='LOSO' --model_path="LOSO_model.pth"
# python tester.py --method='LOSOFT' --model_path="FT_model.pth"