import torch

def ranking_loss(pred1, pred2, label, margin=0.1):
    d1 = pred1.mean(dim=(1, 2, 3))
    d2 = pred2.mean(dim=(1, 2, 3))
    return torch.mean(torch.clamp(-label * (d1 - d2) + margin, min=0))
