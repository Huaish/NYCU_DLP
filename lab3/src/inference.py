import argparse
import torch
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from torch.utils.data import DataLoader
import termcolor
from utils import dice_score
from tqdm import tqdm

def inference(model, test_dataloader, device, eval=True):
    model.eval().to(device)
    outputs = []
    dice_scores = []
    with torch.no_grad():
        with tqdm(total=len(test_dataloader), desc="Inference", unit="batch", leave=False) as pbar:
            for i, batch in enumerate(test_dataloader):
                images = batch["image"].to(device)
                output = model(images)
                tmp = torch.argmax(output, dim=1).squeeze(0)
                outputs.append(tmp.cpu().numpy())
                pbar.update()

                if eval:
                    masks = batch["mask"].to(device)
                    dice_scores.append(dice_score(output, masks))
                    pbar.set_postfix({"Dice Score": dice_scores[-1]})
                
    if eval:
        print(termcolor.colored(f"Average Dice Score: {sum(dice_scores) / len(dice_scores)}", "green"))

    return outputs, dice_scores
    

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--type', default='UNet', help='model type(UNet or ResNet34UNet)')
    parser.add_argument('--model', default='../saved_models/UNet_latest.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet", help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=20, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    data_path = args.data_path
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the data
    print(f"Loading data from {data_path}")
    test_dataset = load_dataset(data_path, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Loaded {len(test_dataset)} test images")

    # Load the model
    print(f"Loading model from {args.model}")
    model = None
    if args.type == "UNet":
        model = UNet(in_channels=3, out_channels=2).to(device)
    elif args.type == "ResNet34UNet":
        model = ResNet34_UNet(in_channels=3, out_channels=2).to(device)
    model.load_state_dict(torch.load(args.model))
    
    inference(model, test_dataloader, device)