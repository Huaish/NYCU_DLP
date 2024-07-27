import argparse
import torch
from oxford_pet import load_dataset
from models.unet import UNet
from evaluate import evaluate
from torch.utils.data import DataLoader
import termcolor

def test(args):
    data_path = args.data_path
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device {device}")
    
    # Load the data
    print(f"Loading data from {data_path}")
    test_dataset = load_dataset(data_path, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load the model
    model = UNet(in_channels=3, out_channels=2).to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    
    # Evaluate the model
    test_loss, test_dice = evaluate(model, test_dataloader, device)
    print(termcolor.colored(f"Test Loss: {test_loss:.4f}, Test Dice Score: {test_dice:.4f}", "green"))
    
    return test_loss, test_dice
    

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='../saved_models/latest.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet", help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    test(args)