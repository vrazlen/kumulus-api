# src/dataset.py
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class SegmentationDataset(Dataset):
    """
    Dataset for satellite image segmentation.
    Reads images and their corresponding masks.
    """
    def __init__(self, transform=None):
        # --- CORRECTED PATHS ---
        self.image_dir = 'data/processed/patches'
        self.mask_dir = 'data/processed/masks' # This should be the correct path for masks

        self.image_ids = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.tif')])
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Construct mask path from image path
        mask_name = img_name.replace('.tif', '_mask.tif') # Assuming this naming convention
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")

        # For this dataset, we assume masks are single-channel images
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask) / 255.0 # Normalize mask to 0-1
        mask = np.expand_dims(mask, axis=0) # Add channel dimension

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(mask).float()

def get_train_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])