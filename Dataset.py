import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import random
import numpy as np
import os

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SuperResolutionDataset(Dataset):
    def __init__(self, root_dir, split='train', patch_size=256, scale_factor=4):
        """
        Dataset for super-resolution training/validation
        
        Args:
            root_dir: Root directory containing high-resolution images
            split: 'train' or 'val'
            patch_size: Size of high-resolution training patches (only used during training)
            scale_factor: Downsampling factor for creating low-resolution images
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.lr_patch_size = patch_size // scale_factor
        
        # Get high-resolution images with multiple extensions
        self.hr_dir = self.root_dir
        
        # Common image extensions
        self.extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        
        # Get all high-resolution images
        self.hr_files = []
        for ext in self.extensions:
            self.hr_files.extend(list(self.hr_dir.glob(f'*{ext}')))
            self.hr_files.extend(list(self.hr_dir.glob(f'*{ext.upper()}')))
        
        # Sort the files to ensure deterministic behavior
        self.hr_files = sorted(self.hr_files)
        
        print(f"Found {len(self.hr_files)} high-resolution images in {self.hr_dir}")
        
        # Basic transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Augmentation transforms for training
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
        ]) if split == 'train' else None
        
    def __len__(self):
        return len(self.hr_files)
    
    def get_random_crop_params(self, img):
        """Get random crop parameters for high-resolution image"""
        w, h = img.size
        th, tw = self.patch_size, self.patch_size
        if w == tw and h == th:
            return 0, 0, h, w
        if w < tw or h < th:
            # Handle images smaller than patch size by resizing
            scale = max(tw / w, th / h) * 1.1  # Scale up with a small margin
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            w, h = new_w, new_h
        
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw, img
    
    def create_low_res(self, hr_img):
        """Create low-resolution image by downscaling with bicubic interpolation"""
        w, h = hr_img.size
        lr_w, lr_h = w // self.scale_factor, h // self.scale_factor
        lr_img = hr_img.resize((lr_w, lr_h), Image.BICUBIC)
        return lr_img
    
    def __getitem__(self, idx):
        try:
            # Load high-resolution image
            hr_path = self.hr_files[idx]
            
            # Open image with PIL
            try:
                hr_img = Image.open(hr_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image: {e}")
                # Return a random sample as fallback
                return self.__getitem__(random.randint(0, len(self) - 1))
            
            # Random crop for training
            if self.split == 'train':
                # Handle random cropping with potential resizing
                i, j, h, w, hr_img_resized = self.get_random_crop_params(hr_img)
                if hr_img_resized is not hr_img:  # If image was resized
                    hr_img = hr_img_resized
                
                # Crop high-resolution image
                hr_img = hr_img.crop((j, i, j + w, i + h))
                
                # Apply augmentation
                if random.random() > 0.5 and self.augment:
                    hr_img = self.augment(hr_img)
            
            # Create low-resolution version
            lr_img = self.create_low_res(hr_img)
            
            # Convert to tensors
            hr_tensor = self.transform(hr_img)
            lr_tensor = self.transform(lr_img)
            
            return lr_tensor, hr_tensor
            
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            # Return a random sample as fallback
            return self.__getitem__(random.randint(0, len(self) - 1))


def create_dataloaders(root_dir_train, root_dir_val, batch_size=8, patch_size=256, scale_factor=4, num_workers=4):
    """Create training and validation dataloaders for super-resolution"""
    train_dataset = SuperResolutionDataset(root_dir_train, split='train', patch_size=patch_size, scale_factor=scale_factor)
    val_dataset   = SuperResolutionDataset(root_dir_val, split='train', patch_size=patch_size, scale_factor=scale_factor)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader