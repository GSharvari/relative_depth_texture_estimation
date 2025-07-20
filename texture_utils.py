import torch
import torchvision.transforms.functional as TF

def texture_strength(patch):
    gray = TF.rgb_to_grayscale(patch).unsqueeze(0)
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
    grad_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
    return torch.mean(torch.abs(grad_x) + torch.abs(grad_y)).item()

def compute_texture_heatmap(image_tensor):
    gray = TF.rgb_to_grayscale(image_tensor).unsqueeze(0)
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32, device=gray.device)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32, device=gray.device)
    grad_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
    texture = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze().cpu().numpy()
    return texture
