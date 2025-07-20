import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import RelativeDepthDataset
from loss import ranking_loss
from model import TinyDepthCNN

def train_model(image, device, crop_size=64, n_samples=200, batch_size=16, epochs=8):
    dataset = RelativeDepthDataset(image, crop_size, n_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TinyDepthCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total = 0
        for img1, img2, label in tqdm(loader, desc=f"Epoch {epoch+1}"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            pred1 = model(img1)[0]
            pred2 = model(img2)[0]

            loss = ranking_loss(pred1, pred2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1} Loss: {total / len(loader):.4f}")
    return model
