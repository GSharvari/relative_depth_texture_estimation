import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from texture_utils import texture_strength

class RelativeDepthDataset(Dataset):
    def __init__(self, image, crop_size=64, n_samples=200):
        self.image = image
        self.n_samples = n_samples
        self.crop_size = crop_size
        self.to_tensor = T.ToTensor()
        self.crop = T.RandomResizedCrop(crop_size, scale=(0.5, 1.0))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        p1 = self.crop(self.image)
        p2 = self.crop(self.image)
        t1 = self.to_tensor(p1)
        t2 = self.to_tensor(p2)
        strength1 = texture_strength(t1)
        strength2 = texture_strength(t2)
        label = 1.0 if strength1 > strength2 else -1.0
        return t1, t2, torch.tensor(label)
