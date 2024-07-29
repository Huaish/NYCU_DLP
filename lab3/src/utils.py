import argparse
import numpy as np
import torch
import termcolor
from PIL import Image
import matplotlib.pyplot as plt
from oxford_pet import SimpleOxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
import albumentations as A

def dice_score(pred_mask, gt_mask, epsilon=1e-6):
    """
    Computes the Dice similarity coefficient between two binary masks.
    
    Args:
        pred_mask (Tensor): predicted  torch.Size([16, 2, 256, 256])
        gt_mask (Tensor): ground truth torch.Size([16, 256, 256])
        epsilon (float): a small number to avoid division by zero.
        
    Returns:
        float: Dice similarity coefficient.
    """
    # Flatten the predicted mask and the ground truth mask
    pred_mask = torch.argmax(pred_mask, dim=1) # torch.Size([16, 256, 256])
    
    # Convert the predicted mask to a binary mask
    pred_mask[pred_mask > 0.5] = torch.tensor(1.0)
    pred_mask[pred_mask <= 0.5] = torch.tensor(0.0)
    
    common_pix = (abs(pred_mask - gt_mask) < epsilon).sum()
    total_pix = pred_mask.reshape(-1).shape[0] + gt_mask.reshape(-1).shape[0]
    dice_score = 2 * common_pix / total_pix
    return dice_score.item()

def dice_score_loss(pred_mask, gt_mask):
    """
    Computes the Dice loss between two binary masks.
    
    Args:
        pred_mask (np.array): predicted binary mask of shape (H, W).
        gt_mask (np.array): ground truth binary mask of shape (H, W).
        
    Returns:
        float: Dice loss.
    """
    return 1 - dice_score(pred_mask, gt_mask)


def visualize_mask(image, mask, alpha=0.5):
    """
    Visualizes the segmentation mask overlaid on the original image.
    
    Args:
        image (np.array): original image of shape (H, W, 3).
        mask (np.array): segmentation mask of shape (H, W).
        alpha (float): transparency level of the mask.
        
    Returns:
        np.array: image with overlaid mask.
    """
    
    # Convert image to RGBA
    image = Image.fromarray(image).convert("RGBA")
    
    # Convert the mask to an RGBA image
    new_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
    
    # Set the RGB channels of the mask
    new_mask[mask == 0] = [68, 1, 84, 0]
    new_mask[mask == 1] = [253, 231, 36, int(255 * alpha)]

    mask = Image.fromarray(new_mask.astype(np.uint8), "RGBA")
    
    # Overlay the mask on the image
    result = Image.alpha_composite(image, mask)
    
    mask = np.array(mask)
    mask[:, :, 3] = 255
    
    return (np.array(image), mask, np.array(result))

def visualize_pred(model_type, model_path, image):
    """
    Visualizes the predicted segmentation mask overlaid on the original image.
    
    Args:
        image (np.array): original image of shape (H, W, 3).
        model (nn.Module): the trained model.
        device (torch.device): the device (GPU or CPU) on which the data and model are located.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)

    # Load the model
    model = None
    if model_type == "UNet":
        model = UNet(in_channels=3, out_channels=2).to(device)
    elif model_type == "ResNet34UNet":
        model = ResNet34_UNet(in_channels=3, out_channels=2).to(device)
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # Perform inference
    model.eval()
    mask = None
    with torch.no_grad():
        mask = model(image)

    # Compute the predicted mask
    mask = torch.argmax(mask, dim=1)

    # Convert the image, mask, and predicted mask to NumPy arrays
    image = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 225).astype(np.uint8)
    mask = mask.squeeze(0).cpu().numpy()

    # Visualize the segmentation mask overlaid on the original image
    result = visualize_mask(image, mask)
    
    return result
    
def visualize(mode, model_type, model_path, data_path, max_images=4):
    assert mode in ['train', 'valid', 'test', 'predict'], "Invalid mode. Choose from 'train', 'valid', 'test', 'predict'"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)

    if mode == 'predict':
        # Load the image and resize it
        image = np.array(Image.open(data_path).convert("RGB"))
        image = np.array(Image.fromarray(image).resize((256, 256), Image.BILINEAR))
        
        # Convert the image to a PyTorch tensor
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image = image.to(device)

        # Visualize predicted mask on the unseen image ( without ground truth mask )
        image, pred_mask, pred_overlay = visualize_pred(model_type, model_path, image)

        # Save the images using subplot
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].imshow(image)
        ax[0].axis("off")
        ax[0].set_title("Original Image")
        
        ax[1].imshow(pred_mask)
        ax[1].axis("off")
        ax[1].set_title("Predicted Mask")
        
        ax[2].imshow(pred_overlay)
        ax[2].axis("off")
        ax[2].set_title("Predicted Mask Overlay")
        
        plt.savefig(f"{model_type}_result.png", bbox_inches='tight')
        
    else:
        # Visualize predicted mask on the train/val/test dataset ( with ground truth mask )
        val_dataset = SimpleOxfordPetDataset(data_path, mode=mode, transform=A.Compose([A.Resize(256, 256),]))
        dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
        
        for i, batch in enumerate(dataloader):
            if i == max_images:
                break
            
            image = batch["image"] / 255.0
            image, pred_mask, pred_overlay = visualize_pred(model_type, model_path, image.to(device))
            _, gt_mask, gt_overlay = visualize_mask(image, batch["mask"].squeeze(0).numpy())

            # Save the images using subplot
            fig, ax = plt.subplots(2, 3, figsize=(15, 10))
            ax[0,0].imshow(image)
            ax[0,0].axis("off")
            ax[0,0].set_title("Original Image")
            
            ax[0,1].imshow(pred_mask)
            ax[0,1].axis("off")
            ax[0,1].set_title("Predicted Mask")
            
            ax[0,2].imshow(pred_overlay)
            ax[0,2].axis("off")
            ax[0,2].set_title("Predicted Mask Overlay")
            
            ax[1,0].imshow(image)
            ax[1,0].axis("off")
            ax[1,0].set_title("Original Image")

            ax[1,1].imshow(gt_mask)
            ax[1,1].axis("off")
            ax[1,1].set_title("Ground Truth Mask")
            
            ax[1,2].imshow(gt_overlay)
            ax[1,2].axis("off")
            ax[1,2].set_title("Ground Truth Mask Overlay")
                    
            
            plt.savefig(f"{model_type}_result_{i}.png", bbox_inches='tight')

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--type', default='ResNet34UNet', help='model type(UNet or ResNet34UNet)')
    parser.add_argument('--mode', default='test', help='mode(train, valid, test, predict)')
    parser.add_argument('--model', default='../saved_models/ResNet34UNet_latest.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet", help='path to the input data')
    parser.add_argument('--max_images', type=int, default=4, help='number of images to visualize')
    parser.add_argument('--seed', type=int, default=16, help='seed for reproducibility')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    return args
    
if __name__ == "__main__":
    args = get_args()
    model_type = args.type
    model_path = args.model
    data_path = args.data_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    mode = args.mode
    
    visualize(mode, model_type, model_path, data_path)
    
# python utils.py --type UNet --mode test --model ../saved_models/DL_Lab3_UNet_313551097_鄭淮薰.pth
# python utils.py --type ResNet34UNet --mode test --model ../saved_models/DL_Lab3_ResNet34_UNet_313551097_鄭淮薰.pth
# python utils.py --type UNet --mode predict --model ../saved_models/DL_Lab3_UNet_313551097_鄭淮薰.pth --data_path ../dataset/oxford-iiit-pet/images/Abyssinian_2.jpg
# python utils.py --type ResNet34UNet --mode predict --model ../saved_models/DL_Lab3_ResNet34_UNet_313551097_鄭淮薰.pth --data_path ../dataset/oxford-iiit-pet/images/Abyssinian_2.jpg