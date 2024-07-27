import torch
import numpy as np
from tqdm import tqdm
from utils import dice_score, dice_score_loss

def evaluate(net, data, device):
    net.eval()
    net.to(device)
    dice_scores = []
    loss = []
    
    with torch.no_grad():
        with tqdm(total=len(data), desc="Evaluation", unit="batch", leave=False) as pbar:
            for i, batch in enumerate(data):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                output = net(images)
                dice_scores.append(dice_score(output, masks))
                loss.append(dice_score_loss(output, masks))
                
                pbar.update()
            
    return np.mean(dice_scores), np.mean(loss)