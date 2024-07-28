import argparse
import numpy as np
import torch
import termcolor
from PIL import Image
import matplotlib.pyplot as plt
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

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

# Visualize the segmentation masks overlaid on the original image
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
    new_mask[mask == 0] = [0, 0, 0, 0]  # Black for the background
    new_mask[mask == 1] = [255, 0, 0, int(255 * alpha)]  # Red for the object

    mask = Image.fromarray(new_mask.astype(np.uint8), "RGBA")
    
    # Overlay the mask on the image
    result = Image.alpha_composite(image, mask)
    
    return (np.array(result), np.array(mask), np.array(image))

def visualize_pred(model_type, model_path, image_path):
    """
    Visualizes the predicted segmentation mask overlaid on the original image.
    
    Args:
        image (np.array): original image of shape (H, W, 3).
        model (nn.Module): the trained model.
        device (torch.device): the device (GPU or CPU) on which the data and model are located.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = None
    if model_type == "UNet":
        model = UNet(in_channels=3, out_channels=2).to(device)
    elif model_type == "ResNet34UNet":
        model = ResNet34_UNet(in_channels=3, out_channels=2).to(device)
        
    model.load_state_dict(torch.load(model_path))
    
    # Load the image and resize it
    image = np.array(Image.open(image_path).convert("RGB"))
    image = np.array(Image.fromarray(image).resize((256, 256), Image.BILINEAR))
    
    # Convert the image to a PyTorch tensor
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image = image.to(device)
    
    # Perform inference
    model.eval()
    mask = None
    with torch.no_grad():
        mask = model(image)

    # Compute the predicted mask
    mask = torch.argmax(mask, dim=1)

    # Convert the image, mask, and predicted mask to NumPy arrays
    image = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    mask = mask.squeeze(0).cpu().numpy()

    # Visualize the segmentation mask overlaid on the original image
    result = visualize_mask(image, mask)
    
    return result
    
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--type', default='ResNet34UNet', help='model type(UNet or ResNet34UNet)')
    parser.add_argument('--model', default='../saved_models/ResNet34UNet_latest.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet", help='path to the input data')
    parser.add_argument('--image_path', type=str, default="../dataset/oxford-iiit-pet/images/Abyssinian_2.jpg", help='path to the input image')
    parser.add_argument('--batch_size', '-b', type=int, default=20, help='batch size')
    
    return parser.parse_args()
    
if __name__ == "__main__":
    args = get_args()
    model_type = args.type
    model_path = args.model
    data_path = args.data_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    overlay, mask, image = visualize_pred(args.type, args.model, args.image_path)
    # Save the images using subplot
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(overlay)
    ax[0].axis("off")
    ax[0].set_title("Predicted Mask")
    
    ax[1].imshow(mask)
    ax[1].axis("off")
    ax[1].set_title("Ground Truth Mask")
    
    ax[2].imshow(image)
    ax[2].axis("off")
    ax[2].set_title("Original Image")
    
    
    plt.savefig("result.png", bbox_inches='tight')

    # # Load the model
    # model = None
    # if model_type == "UNet":
    #     model = UNet(in_channels=3, out_channels=2).to(device)
    # elif model_type == "ResNet34UNet":
    #     model = ResNet34_UNet(in_channels=3, out_channels=2).to(device)
        
    # model.load_state_dict(torch.load(model_path))

    # from inference import inference
    # val_dataset = load_dataset(data_path, "valid")
    # dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    # outputs, dice_scores = inference(model, dataloader, device, eval=True)
    
    # image = (val_dataset[0]["image"].permute(1, 2, 0).cpu().numpy())
    # image = (image - image.min()) / (image.max() - image.min()) * 255
    # image = image.astype(np.uint8)
    # pred = outputs[0]
    # mask = val_dataset[0]["mask"]
    

    # # Predict masks overlaid on the original image
    # image_pred = visualize_mask(image, pred)

    # # GT masks overlaid on the original image
    # result_mask = visualize_mask(image, mask)
    
    # # Save the images using subplot
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(image_pred)
    # ax[0].axis("off")
    # ax[0].set_title("Predicted Mask")
    
    # ax[1].imshow(result_mask)
    # ax[1].axis("off")
    # ax[1].set_title("Ground Truth Mask")
    
    # plt.savefig("result.png", bbox_inches='tight')
    
    
    