import numpy as np

def dice_score(pred_mask, gt_mask):
    """
    Computes the Dice similarity coefficient between two binary masks.
    
    Args:
        pred_mask (np.array): predicted binary mask of shape (H, W).
        gt_mask (np.array): ground truth binary mask of shape (H, W).
        
    Returns:
        float: Dice similarity coefficient.
    """
    pred_mask = pred_mask.astype(np.bool)
    gt_mask = gt_mask.astype(np.bool)
    
    intersection = np.logical_and(pred_mask, gt_mask)
    return 2. * intersection.sum() / (pred_mask.sum() + gt_mask.sum())

