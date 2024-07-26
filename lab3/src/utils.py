import numpy as np
import torch
import termcolor

def dice_score(pred_mask, gt_mask, epsilon=1e-6):
    """
    Computes the Dice similarity coefficient between two binary masks.
    
    Args:
        pred_mask (Tensor): predicted  torch.Size([16, 2, 256, 256])
        gt_mask (Tensor): ground truth torch.Size([16, 256, 256])
        
    Returns:
        float: Dice similarity coefficient.
    """
    # Convert predicted mask to class labels
    pred_mask = torch.argmax(pred_mask, dim=1) # torch.Size([16, 256, 256])
    
    # Flatten the tensors
    pred_mask = pred_mask.view(-1).float()
    gt_mask = gt_mask.view(-1).float()
    
    # Calculate intersection and union
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    
    # Compute Dice coefficient
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice.item()


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