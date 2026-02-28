import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from logger import logger

import torch.nn.functional as F

class SpeckleFilter:
    """
    Applies a very fast spatial smoothing filter (Mean Filter via AvgPool)
    to approximate Speckle noise reduction. This avoids the massive CPU 
    overhead of GaussianBlur inside the DataLoader.
    """
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        # Apply fast mean pooling (shape: C, H, W) -> add fake batch dim
        img = img.unsqueeze(0)
        img = F.avg_pool2d(img, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        img = img.squeeze(0)
        return img

class SARLogTransform:
    """Log-compress SAR intensity values to tame extreme scatterers."""
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        img = torch.clamp(img, min=self.eps)
        img = torch.log1p(img)
        # Normalize to [0, 1] based on current batch max to tame extremes
        img = img / (img.max() + self.eps)
        return img

class EOSARDataset(Dataset):
    """Dataset for paired EO and SAR training images."""
    def __init__(self, sar_root, eo_root, sar_transform=None, eo_transform=None):
        self.sar_root = sar_root
        self.eo_root = eo_root
        self.sar_transform = sar_transform
        self.eo_transform = eo_transform
        
        self.classes = sorted(os.listdir(sar_root))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            sar_cls_path = os.path.join(sar_root, cls_name)
            eo_cls_path = os.path.join(eo_root, cls_name)
            
            for img_name in os.listdir(sar_cls_path):
                sar_img_path = os.path.join(sar_cls_path, img_name)
                eo_img_path = os.path.join(eo_cls_path, img_name)
                
                if os.path.exists(eo_img_path):
                    self.samples.append((sar_img_path, eo_img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sar_path, eo_path, label = self.samples[idx]
        
        sar_img = Image.open(sar_path).convert('L') # SAR is grayscale
        eo_img = Image.open(eo_path).convert('RGB')
        
        if self.sar_transform:
            sar_img = self.sar_transform(sar_img)
        if self.eo_transform:
            eo_img = self.eo_transform(eo_img)
            
        return {'sar': sar_img, 'eo': eo_img, 'label': label}

class SAROnlyDataset(Dataset):
    """Dataset for SAR-only validation/test images."""
    def __init__(self, sar_root, csv_path=None, class_to_idx=None, transform=None):
        self.sar_root = sar_root
        self.transform = transform
        self.class_to_idx = class_to_idx
        
        self.samples = []
        if csv_path and os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img_name = row['image_id']
                if not img_name.endswith('.png'):
                    img_name += '.png'
                img_path = os.path.join(sar_root, img_name)
                
                # In competition, OOD samples might have 'unknown' class
                label = -1
                if 'class' in row and self.class_to_idx:
                    label = self.class_to_idx.get(row['class'], -1)
                
                ood_flag = row.get('OOD_flag', 0)
                self.samples.append((img_path, img_name, label, ood_flag))
        else:
            # If no CSV, just list files
            for img_name in os.listdir(sar_root):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(sar_root, img_name)
                    self.samples.append((img_path, img_name, -1, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, img_name, label, ood_flag = self.samples[idx]
        img = Image.open(img_path).convert('L')
        
        if self.transform:
            img = self.transform(img)
            
        return {'sar': img, 'image_id': img_name, 'label': label, 'ood_flag': ood_flag}

def get_transforms(img_size=224):
    eo_train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    sar_train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        SpeckleFilter(),
        SARLogTransform(),
        transforms.Normalize(mean=[0.5], std=[0.2]) # Approximation, adjust based on stats
    ])
    
    sar_val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        SpeckleFilter(),
        SARLogTransform(),
        transforms.Normalize(mean=[0.5], std=[0.2])
    ])
    
    return eo_train_transform, sar_train_transform, sar_val_transform


def mixup_data(x_sar, x_eo, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_sar.size(0)
    index = torch.randperm(batch_size).to(x_sar.device)

    mixed_sar = lam * x_sar + (1 - lam) * x_sar[index, :]
    mixed_eo = lam * x_eo + (1 - lam) * x_eo[index, :]
    y_a, y_b = y, y[index]
    return mixed_sar, mixed_eo, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
