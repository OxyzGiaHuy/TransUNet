import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging # Use logging instead of print for better integration


def gather_masks_and_imgs(data_path):
    """
    Gather images and masks from TNBC dataset
    """
    logging.info("=== Gathering Masks and Images from TNBC dataset ===")
    img_paths = []
    mask_paths = []
    
    # Assuming GT folders match Slide folders by number (GT_01 -> Slide_01)
    slide_folders = sorted([f for f in os.listdir(data_path) if "Slide" in f and os.path.isdir(os.path.join(data_path, f))])
    gt_folders = sorted([f for f in os.listdir(data_path) if "GT" in f and os.path.isdir(os.path.join(data_path, f))])

    if len(slide_folders) != len(gt_folders):
        raise ValueError(f"Mismatch between number of Slide folders ({len(slide_folders)}) and GT folders ({len(gt_folders)})")

    for slide_folder, gt_folder in zip(slide_folders, gt_folders):
        slide_folder_path = os.path.join(data_path, slide_folder)
        gt_folder_path = os.path.join(data_path, gt_folder)
        
        # Find all images and masks, then sort them to ensure they match
        current_imgs = sorted(glob.glob(os.path.join(slide_folder_path, "*.png")))
        current_masks = sorted(glob.glob(os.path.join(gt_folder_path, "*.png")))
        
        img_paths.extend(current_imgs)
        mask_paths.extend(current_masks)

    logging.info(f"Found {len(slide_folders)} patient slides.")
    logging.info(f"Total images found: {len(img_paths)}")
    logging.info(f"Total masks found: {len(mask_paths)}")

    if len(img_paths) != len(mask_paths):
        raise ValueError("Number of images and masks do not match! Please check folder contents and naming.")
    if not img_paths:
        raise ValueError("No images found in the dataset!")

    img_paths = sorted(img_paths)
    mask_paths = sorted(mask_paths)

    return img_paths, mask_paths


def get_tnbc_transformations(img_size):
    """
    Get transformations for TNBC dataset
    """
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5,
                           border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3), # Use a tuple for blur_limit
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, val_transforms


class TNBC_dataset(Dataset):
    """
    TNBC Dataset loader for binary segmentation (background vs. nuclei)
    """
    def __init__(self, base_dir, split='train', image_size=512, train_ratio=0.8, random_state=42):
        self.split = split
        self.base_dir = base_dir
        
        imgs, masks = gather_masks_and_imgs(base_dir)
        
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            imgs, masks, test_size=(1-train_ratio), random_state=random_state
        )
        
        train_transform, val_transform = get_tnbc_transformations(img_size=image_size)
        
        if split == 'train':
            self.image_paths = train_imgs
            self.mask_paths = train_masks
            self.transforms = train_transform
        elif split == 'val':
            self.image_paths = val_imgs
            self.mask_paths = val_masks
            self.transforms = val_transform
        else:
            raise ValueError(f"Invalid split '{split}'. Choose 'train' or 'val'.")
        
        self.case_names = [os.path.basename(path).split('.')[0] for path in self.image_paths]
            
        logging.info(f"Successfully loaded {split} set with {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Convert mask to binary: background (0) and nuclei (1)
        # Assuming nuclei are any non-black pixel
        label = np.where(mask > 0, 1, 0).astype(np.uint8)
        
        if self.transforms:
            augmented = self.transforms(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask'].long()
        else:
            # Fallback (should not happen if using get_tnbc_transformations)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            label = torch.from_numpy(label).long()
        
        sample = {
            'image': image, 
            'label': label,
            'case_name': self.case_names[idx]
        }
        return sample
