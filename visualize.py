import torch
import matplotlib.pyplot as plt
import os
from torchvision import transforms 

def visualize_depth_and_texture(model, image, device):
    model.eval()

    # Create the transform and convert image to tensor
    transform = transforms.ToTensor()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_depth, patch_texture = model(input_tensor)

    # Prepare for plotting
    depth_map = predicted_depth.squeeze().cpu().numpy()
    texture_map = patch_texture.squeeze().cpu().numpy()

    # Ensure output directory exists
    os.makedirs('visualizations', exist_ok=True)

    # Plot and save
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title('Predicted Depth Map')
    plt.imshow(depth_map, cmap='inferno')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Patch Texture Map')
    plt.imshow(texture_map, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('visualizations/depth_texture_visualization.png')
    plt.show()
