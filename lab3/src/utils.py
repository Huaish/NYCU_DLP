import numpy as np
import torch
import termcolor

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