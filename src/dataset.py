# src/dataset.py
import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class SegmentationDataset(Dataset):
    def __init__(self, data_dir='data/processed', transform=None):
        self.image_dir = os.path.join(data_dir, 'patches')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.transform = transform

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found at {self.image_dir}")
        
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.tif')))
        
        if not self.image_files:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        mask_np = np.array(mask) / 255.0
        mask_np = np.expand_dims(mask_np, axis=0) 

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(mask_np).float()

def get_train_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])