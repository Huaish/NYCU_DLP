import torch
from utils import dice_score

def evaluate(net, data, device):
    net.eval()
    net.to(device)
    dice = 0.0
    with torch.no_grad():
        for i, batch in enumerate(data):
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            outputs = net(images)
            for j in range(len(outputs)):
                pred_mask = outputs[j].cpu().numpy()
                gt_mask = masks[j].cpu().numpy()
                dice += dice_score(pred_mask, gt_mask)
    return dice / len(data)