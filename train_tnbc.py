#!/usr/bin/env python3

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

from datasets.dataset_tnbc import TNBC_dataset
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from utils import SimpleDiceLoss, calculate_pixel_accuracy, calculate_iou, calculate_dice_coefficient, visualize_predictions


def setup_logging(log_path):
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )


class ModelSaveCallback(Callback):
    """Custom callback to log model saving events"""
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Called when saving a checkpoint"""
        epoch = trainer.current_epoch
        logging.info(f"Checkpoint saved at epoch {epoch+1}")
    
    def on_train_end(self, trainer, pl_module):
        """Called when training ends"""
        if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
            best_path = trainer.checkpoint_callback.best_model_path
            last_path = trainer.checkpoint_callback.last_model_path
            best_score = trainer.checkpoint_callback.best_model_score
            
            logging.info("=" * 80)
            logging.info("Training completed successfully!")
            logging.info(f"Best model saved at: {best_path}")
            logging.info(f"Last model saved at: {last_path}")
            logging.info(f"Best validation loss: {best_score:.4f}")
            logging.info("=" * 80)


class VisualizationCallback(Callback):
    """Custom callback for generating visualizations"""
    
    def __init__(self, output_dir: str, visualization_frequency: int = 5):
        super().__init__()
        self.output_dir = output_dir
        self.visualization_frequency = visualization_frequency
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Log epoch summary"""
        current_epoch = trainer.current_epoch
        
        # Get current metrics
        train_loss = trainer.callback_metrics.get('train_loss', 0)
        val_loss = trainer.callback_metrics.get('val_loss', 0)
        val_pixel_acc = trainer.callback_metrics.get('val_pixel_acc', 0)
        val_mean_iou = trainer.callback_metrics.get('val_mean_iou', 0)
        val_mean_dice = trainer.callback_metrics.get('val_mean_dice', 0)
        
        # Check if this is the best epoch
        is_best = False
        if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
            current_monitor_val = trainer.callback_metrics.get('val_loss')
            if current_monitor_val is not None:
                best_score = trainer.checkpoint_callback.best_model_score
                if best_score is None or current_monitor_val <= best_score:
                    is_best = True
        
        # Log epoch summary
        best_indicator = "🎉 New best model! " if is_best else ""
        logging.info('=' * 80)
        logging.info(f'Epoch {current_epoch + 1}/{trainer.max_epochs} Summary:')
        logging.info(f'{best_indicator}Training Loss: {train_loss:.4f}')
        logging.info(f'Validation Loss: {val_loss:.4f}')
        logging.info(f'Validation Pixel Accuracy: {val_pixel_acc:.4f} ({val_pixel_acc*100:.2f}%)')
        logging.info(f'Validation Mean IoU: {val_mean_iou:.4f} ({val_mean_iou*100:.2f}%)')
        logging.info(f'Validation Mean Dice: {val_mean_dice:.4f} ({val_mean_dice*100:.2f}%)')
        logging.info('=' * 80)
        
        # Generate visualizations every N epochs
        if (current_epoch + 1) % self.visualization_frequency == 0:
            try:
                self._generate_visualizations(trainer, pl_module, current_epoch)
            except Exception as e:
                logging.warning(f"Failed to generate visualizations: {e}")
    
    def _generate_visualizations(self, trainer, pl_module, epoch):
        """Generate and save visualizations"""
        try:
            # Get validation dataloader
            val_dataloader = trainer.val_dataloaders
            if val_dataloader is not None:
                logging.info("Generating prediction visualizations...")
                # Use pl_module.model since we're in Lightning context
                visualize_predictions(pl_module.model, val_dataloader, pl_module.device, 
                                    self.output_dir, epoch + 1, num_samples=3)
        except Exception as e:
            logging.warning(f"Visualization generation failed: {e}")


class TNBCDataModule(L.LightningDataModule):
    """Lightning DataModule for TNBC dataset"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.num_workers = config.get('num_workers', 2)
        self.pin_memory = config.get('pin_memory', True)
        
    def setup(self, stage=None):
        """Setup train and validation datasets"""
        self.train_dataset = TNBC_dataset(
            base_dir=self.config['root_path'],
            split="train",
            image_size=self.config['img_size'],
            train_ratio=0.8,
            random_state=self.config['seed']
        )
        
        self.val_dataset = TNBC_dataset(
            base_dir=self.config['root_path'],
            split="val",
            image_size=self.config['img_size'],
            train_ratio=0.8,
            random_state=self.config['seed']
        )
        
        logging.info(f"Training dataset size: {len(self.train_dataset)}")
        logging.info(f"Validation dataset size: {len(self.val_dataset)}")
        
    def train_dataloader(self):
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self):
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )


class TransUNetLightning(L.LightningModule):
    """Lightning Module for TransUNet"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # Create model
        config_vit = CONFIGS_ViT_seg[config['vit_name']]
        config_vit.n_classes = config['num_classes']
        config_vit.n_skip = config['n_skip']
        config_vit.patches.grid = (int(config['img_size'] / 16), int(config['img_size'] / 16))
        
        self.model = ViT_seg(config_vit, img_size=config['img_size'], num_classes=config['num_classes'])
        
        # Load pretrained weights
        try:
            self.model.load_from(weights=np.load(config_vit.pretrained_path))
            logging.info("Loaded pretrained weights successfully")
        except Exception as e:
            logging.warning(f"Could not load pretrained weights: {e}")
            logging.info("Training from scratch...")
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = SimpleDiceLoss(config['num_classes'])
        
        # Training parameters
        self.base_lr = config['base_lr']
        self.weight_decay = config.get('weight_decay', 0.0001)
        self.momentum = config.get('momentum', 0.9)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        image, label = batch['image'], batch['label']
        
        # Forward pass
        outputs = self(image)
        
        # Compute losses
        loss_ce = self.ce_loss(outputs, label.long())
        loss_dice = self.dice_loss(outputs, label, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        
        # Calculate metrics
        predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        pixel_acc = calculate_pixel_accuracy(predictions, label)
        _, mean_iou = calculate_iou(predictions, label, self.config['num_classes'])
        _, mean_dice = calculate_dice_coefficient(predictions, label, self.config['num_classes'])
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_ce_loss', loss_ce)
        self.log('train_dice_loss', loss_dice)
        self.log('train_pixel_acc', pixel_acc)
        self.log('train_mean_iou', mean_iou)
        self.log('train_mean_dice', mean_dice)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'])
        
        # Log training progress periodically
        if batch_idx % 50 == 0:
            logging.info(f'Epoch {self.current_epoch+1}, Batch {batch_idx}: '
                        f'Loss={loss.item():.4f}, CE={loss_ce.item():.4f}, '
                        f'Dice={loss_dice.item():.4f}, PixelAcc={pixel_acc:.4f}, '
                        f'MeanIoU={mean_iou:.4f}, MeanDice={mean_dice:.4f}')
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        image, label = batch['image'], batch['label']
        
        # Forward pass
        outputs = self(image)
        
        # Compute losses
        loss_ce = self.ce_loss(outputs, label.long())
        loss_dice = self.dice_loss(outputs, label, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        
        # Calculate metrics
        predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        pixel_acc = calculate_pixel_accuracy(predictions, label)
        _, mean_iou = calculate_iou(predictions, label, self.config['num_classes'])
        _, mean_dice = calculate_dice_coefficient(predictions, label, self.config['num_classes'])
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_ce_loss', loss_ce)
        self.log('val_dice_loss', loss_dice)
        self.log('val_pixel_acc', pixel_acc)
        self.log('val_mean_iou', mean_iou)
        self.log('val_mean_dice', mean_dice)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.base_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1.0 - epoch / self.trainer.max_epochs) ** 0.9
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_callbacks(output_dir: str) -> list:
    """Create training callbacks"""
    # Model checkpoint callback for best model
    best_checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename='best-transunet-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=False,
        verbose=True
    )
    
    # Model checkpoint callback for last model
    last_checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename='last-transunet-{epoch:02d}',
        save_top_k=0,
        save_last=True,
        verbose=True
    )
    
    return [
        best_checkpoint,
        last_checkpoint,
        ModelSaveCallback(),
        VisualizationCallback(output_dir, visualization_frequency=5),
        LearningRateMonitor(logging_interval='epoch'),
    ]


def train_tnbc():
    """Main training function using PyTorch Lightning"""
    print("Starting TransUNet training on TNBC dataset with PyTorch Lightning...")
    
    # Configuration
    config = {
        'dataset': 'TNBC',
        'vit_name': 'R50-ViT-B_16',
        'batch_size': 4,
        'max_epochs': 50,
        'base_lr': 0.01,
        'img_size': 512,
        'seed': 1234,
        'n_skip': 3,
        'num_classes': 2,  # Binary segmentation: background and nuclei
        'root_path': '/home/tanguyen12gb/Desktop/thaigiahuy/Dataset/TNBC_NucleiSegmentation',
        'num_workers': 2,
        'pin_memory': True,
        'weight_decay': 0.0001,
        'momentum': 0.9
    }
    
    # Create output directory
    output_dir = f"./model/TNBC_lightning_training_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(os.path.join(output_dir, "training.log"))
    
    # Set seed for reproducibility
    L.seed_everything(config['seed'])
    
    # Log configuration
    logging.info("=" * 80)
    logging.info("Starting TransUNet Training on TNBC Dataset with PyTorch Lightning")
    logging.info("=" * 80)
    logging.info(f"Dataset: {config['dataset']}")
    logging.info(f"Model: {config['vit_name']}")
    logging.info(f"Classes: {config['num_classes']} (Binary: Background + Nuclei)")
    logging.info(f"Image size: {config['img_size']}")
    logging.info(f"Batch size: {config['batch_size']}")
    logging.info(f"Learning rate: {config['base_lr']}")
    logging.info(f"Max epochs: {config['max_epochs']}")
    logging.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logging.info("=" * 80)
    
    # Create data module
    data_module = TNBCDataModule(config)
    
    # Create model
    model = TransUNetLightning(config)
    
    # Create callbacks
    callbacks = create_callbacks(output_dir)
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=config['max_epochs'],
        accelerator="auto",
        devices="auto",
        default_root_dir=output_dir,
        callbacks=callbacks,
        log_every_n_steps=20,
        gradient_clip_val=1.0,
        deterministic=False,
        enable_model_summary=True,
        enable_progress_bar=True
    )
    
    # Start training
    logging.info("Starting training...")
    trainer.fit(model, data_module)
    
    return output_dir


if __name__ == "__main__":
    try:
        output_dir = train_tnbc()
        print(f"Training completed successfully! Output directory: {output_dir}")
    except Exception as e:
        print(f"Training failed with error: {e}")
        logging.error(f"Training failed: {e}")
        raise
