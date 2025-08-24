import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from typing import Dict, Any

# Add project root to path
sys.path.append('/home/tanguyen12gb/Desktop/thaigiahuy/test_molex/TransUNet')

from datasets.dataset_colon import ColonDataset
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from utils import SimpleDiceLoss, calculate_pixel_accuracy, calculate_iou, calculate_dice_coefficient, visualize_predictions
from train_tnbc import setup_logging, ModelSaveCallback, VisualizationCallback, create_callbacks, TransUNetLightning


# Lightning DataModule for Colon dataset
class ColonDataModule(L.LightningDataModule):
    """Lightning DataModule for CVC-ColonDB dataset"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.num_workers = config.get('num_workers', 2)
        self.pin_memory = config.get('pin_memory', True)

    def setup(self, stage=None):
        import glob
        import cv2
        from sklearn.model_selection import train_test_split
        import numpy as np
        import logging
        # Gather image paths
        neg_img_paths = sorted([f for f in glob.glob(os.path.join(self.config['root_path'], "tissue-train-neg/*.jpg")) if not f.endswith("mask.jpg")])
        pos_img_paths = sorted([f for f in glob.glob(os.path.join(self.config['root_path'], "tissue-train-pos-v1/*.jpg")) if not f.endswith("mask.jpg")])

        # Generate masks for negative images if they don't exist
        for img_path in neg_img_paths:
            mask_path = img_path.replace(".jpg", "_mask.jpg")
            if not os.path.exists(mask_path):
                img = cv2.imread(img_path)
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.imwrite(mask_path, mask)
                logging.info(f"Created zero mask for {img_path}")

        neg_mask_paths = [p.replace(".jpg", "_mask.jpg") for p in neg_img_paths]
        pos_mask_paths = [p.replace(".jpg", "_mask.jpg") for p in pos_img_paths]

        all_imgs = neg_img_paths + pos_img_paths
        all_masks = neg_mask_paths + pos_mask_paths

        # Split data
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            all_imgs, all_masks, test_size=0.2, random_state=self.config['seed']
        )

        # Get transforms
        from datasets.dataset_colon import get_colon_transformations
        train_ts, val_ts = get_colon_transformations(self.config['img_size'])

        # Create datasets
        self.train_dataset = ColonDataset(train_imgs, train_masks, transform=train_ts)
        self.val_dataset = ColonDataset(val_imgs, val_masks, transform=val_ts)

        logging.info(f"Training dataset size: {len(self.train_dataset)}")
        logging.info(f"Validation dataset size: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )




def train_colon():
    """Main training function for Colon dataset using PyTorch Lightning"""
    print("Starting UNet training on CVC-ColonDB dataset with PyTorch Lightning...")

    config = {
        'dataset': 'ColonDB',
        'vit_name': 'R50-ViT-B_16',
        'batch_size': 4,
        'max_epochs': 50,
        'base_lr': 0.01,
        'img_size': 512,
        'seed': 1234,
        'n_skip': 3,
        'num_classes': 2,  # Binary segmentation: background and nuclei
        'root_path': '/home/tanguyen12gb/Desktop/thaigiahuy/Dataset/CVC_ColonDB',
        'num_workers': 2,
        'pin_memory': True,
        'weight_decay': 0.0001,
        'momentum': 0.9
    }
    
    output_dir = f"./model/Colon_lightning_training_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    
    setup_logging(os.path.join(output_dir, "training.log"))
    
    L.seed_everything(config['seed'])
    
    logging.info("=" * 80)
    logging.info(f"Starting UNet Training on {config['dataset']} with PyTorch Lightning")
    for key, value in config.items():
        logging.info(f"{key.replace('_', ' ').title()}: {value}")
    logging.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logging.info("=" * 80)
    
    data_module = ColonDataModule(config)
    model = TransUNetLightning(config)
    callbacks = create_callbacks(output_dir)
    
    trainer = L.Trainer(
        max_epochs=config['max_epochs'],
        accelerator="auto",
        devices="auto",
        default_root_dir=output_dir,
        callbacks=callbacks,
        log_every_n_steps=20,
        gradient_clip_val=1.0,
    )
    
    logging.info("Starting training...")
    # To resume training, add ckpt_path='path/to/last.ckpt'
    trainer.fit(model, data_module)
    
    logging.info("UNet training on Colon dataset completed successfully!")
    
    return output_dir


if __name__ == "__main__":
    try:
        output_dir = train_colon()
        print(f"Training completed successfully! Output directory: {output_dir}")
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        print(f"Training failed with error: {e}")
        raise