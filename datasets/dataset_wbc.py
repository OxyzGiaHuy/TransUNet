import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_segmentation_transforms(image_size=512):
    """
    Define augmentation transforms for segmentation task.
    Sửa lỗi bằng cách chỉ định rõ phương pháp nội suy cho mask.
    """
    train_transforms = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, 
                           interpolation=cv2.INTER_LINEAR,
                           border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0,
                           mask_interpolation=cv2.INTER_NEAREST),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transforms = A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transforms, val_transforms


class BloodCellDataset(Dataset):
    """
    Custom dataset class for blood cell images.
    Handles 3-color segmentation masks and converts them to class indices (0, 1, 2).
    - Class 0: Background (pixel < 50)
    - Class 1: Cytoplasm (50 <= pixel < 200)  
    - Class 2: Nucleus (pixel >= 200)
    """
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        class_mask = np.zeros(mask.shape, dtype=np.uint8)
        
        # Gán nhãn cho Nucleus (lớp 2) - vùng màu trắng
        class_mask[mask > 200] = 2
        
        # Gán nhãn cho Cytoplasm (lớp 1) - vùng màu xám
        class_mask[np.logical_and(mask > 50, mask <= 200)] = 1
        
        if self.transforms:
            augmented = self.transforms(image=image, mask=class_mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            mask = torch.from_numpy(class_mask)
        
        return {'image': image, 'mask': mask.long()}


class WBC_dataset(Dataset):
    def __init__(self, base_dir, split='train', image_size=512, train_ratio=0.8):
        """
        WBC Dataset loader - Updated to use albumentations like in notebook
        Args:
            base_dir: Path to Dataset 1 folder
            split: 'train' or 'test'  
            image_size: Size to resize images to
            train_ratio: ratio of data to use for training
        """
        self.split = split
        self.base_dir = base_dir
        
        # Load file paths and split data like in notebook
        image_paths_bmp = sorted(glob.glob(os.path.join(base_dir, "*.bmp")))
        mask_paths_png = sorted(glob.glob(os.path.join(base_dir, "*.png")))
        
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            image_paths_bmp, mask_paths_png, test_size=(1-train_ratio), random_state=42
        )
        
        if split == 'train':
            self.image_paths = train_imgs
            self.mask_paths = train_masks
            # Get train transforms
            self.transforms, _ = get_segmentation_transforms(image_size=image_size)
        else:
            self.image_paths = val_imgs
            self.mask_paths = val_masks
            # Get validation transforms
            _, self.transforms = get_segmentation_transforms(image_size=image_size)
        
        self.case_names = [os.path.basename(path).split('.')[0] for path in self.image_paths]
            
        print(f"Loading {split} set with {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        class_mask = np.zeros(mask.shape, dtype=np.uint8)
        
        # Gán nhãn cho Nucleus (lớp 2) - vùng màu trắng
        class_mask[mask > 200] = 2
        
        # Gán nhãn cho Cytoplasm (lớp 1) - vùng màu xám
        class_mask[np.logical_and(mask > 50, mask <= 200)] = 1
        
        # Apply albumentations transforms
        if self.transforms:
            augmented = self.transforms(image=image, mask=class_mask)
            image = augmented['image']
            label = augmented['mask']
        else:
            # Fallback if no transforms
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            label = torch.tensor(class_mask, dtype=torch.long)
        
        sample = {
            'image': image, 
            'label': label,
            'case_name': self.case_names[idx]
        }
        return sample
    
