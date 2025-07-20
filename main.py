import torch
from PIL import Image
from train import train_model
from visualize import visualize_depth_and_texture

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open("depthEstimation/data/jcsmr.jpg").convert("RGB")
    model = train_model(image, device)
    visualize_depth_and_texture(model, image, device)

if __name__ == "__main__":
    main()
