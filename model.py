import torch
import torch.nn as nn
from texture_utils import compute_texture_heatmap

class TinyDepthCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1)
        )

    def forward(self, x):
        depth = self.net(x)
        texture_map = []

        # Compute texture heatmap for each image in batch
        for i in range(x.shape[0]):
            single_texture = compute_texture_heatmap(x[i])
            texture_map.append(torch.tensor(single_texture))

        # Stack and unsqueeze to match dimensions (batch, 1, H, W)
        texture_tensor = torch.stack(texture_map).unsqueeze(1).to(depth.device)

        return depth, texture_tensor
