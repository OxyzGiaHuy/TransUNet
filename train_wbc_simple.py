#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/tanguyen12gb/Desktop/thaigiahuy/test_molex/TransUNet')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import logging
import time
import cv2
import matplotlib.pyplot as plt
import pytorch_lightning as L
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from typing import Dict, Any

from datasets.dataset_wbc import WBC_dataset, BloodCellDataset, get_segmentation_transforms
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from utils import SimpleDiceLoss


def visualize_predictions(model, dataloader, device, output_dir, epoch, num_samples=5):
    """Visualize model predictions and save them"""
    model.eval()
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Color map for 3 classes
    colors = {
        0: [0, 0, 0],       # Background - Black
        1: [128, 128, 128], # Cytoplasm - Gray  
        2: [255, 255, 255]  # Nucleus - White
    }
    
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if i >= num_samples:
                break
                
            image = sample['image'].to(device)
            label = sample['label'].to(device)
            case_name = sample['case_name'][0]
            
            # Get prediction
            output = model(image)
            prediction = torch.argmax(torch.softmax(output, dim=1), dim=1)
            
            # Convert to numpy for visualization
            # For original image, we need to denormalize it back to [0, 255]
            img_np = image[0].permute(1, 2, 0).cpu().numpy()  # CHW to HWC
            
            # Denormalize image from ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean  # Denormalize
            img_np = np.clip(img_np, 0, 1)  # Clip to [0, 1]
            img_np = (img_np * 255).astype(np.uint8)  # Convert to [0, 255]
            
            label_np = label[0].cpu().numpy()
            pred_np = prediction[0].cpu().numpy()
            
            # Create colored masks
            def create_colored_mask(mask):
                colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
                for class_id, color in colors.items():
                    colored[mask == class_id] = color
                return colored
            
            # Create visualizations
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Original image
            axes[0].imshow(img_np)
            axes[0].set_title(f'Input Image\n{case_name}')
            axes[0].axis('off')
            
            # Ground truth
            gt_colored = create_colored_mask(label_np)
            axes[1].imshow(gt_colored)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            pred_colored = create_colored_mask(pred_np)
            axes[2].imshow(pred_colored)
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            # Overlay prediction on original
            img_rgb = img_np.copy()  # img_np is already in RGB format and [0, 255]
            
            # Create overlay using direct blending
            alpha = 0.4
            overlay = img_rgb.copy()
            overlay[pred_np == 1] = [255, 255, 0]  # Yellow for cytoplasm
            overlay[pred_np == 2] = [255, 0, 0]    # Red for nucleus
            
            # Manual blending
            blended = (1-alpha) * img_rgb.astype(np.float32) + alpha * overlay.astype(np.float32)
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            axes[3].imshow(blended)
            axes[3].set_title('Overlay')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'visualizations', f'epoch_{epoch}_sample_{i}_{case_name}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Calculate metrics for this sample
            pixel_acc = calculate_pixel_accuracy(prediction[0:1], label[0:1])
            class_ious, mean_iou = calculate_iou(prediction[0:1], label[0:1], 3)
            
            logging.info(f'Sample {case_name} - Pixel Acc: {pixel_acc:.4f}, Mean IoU: {mean_iou:.4f}')
            logging.info(f'  Class IoUs: Background={class_ious[0]:.3f}, Cytoplasm={class_ious[1]:.3f}, Nucleus={class_ious[2]:.3f}')
    
    model.train()
    logging.info(f'Saved {min(num_samples, len(dataloader))} visualization samples to {output_dir}/visualizations/')


def calculate_pixel_accuracy(pred_tensor, target_tensor):
    """Calculate pixel-wise accuracy using torch operations"""
    pred_flat = pred_tensor.flatten()
    target_flat = target_tensor.flatten()
    correct = torch.sum(pred_flat == target_flat).item()
    total = target_flat.numel()
    return correct / total


def calculate_iou(pred_tensor, target_tensor, num_classes):
    """Calculate IoU for each class and mean IoU using torch operations"""
    pred_flat = pred_tensor.flatten()
    target_flat = target_tensor.flatten()
    
    # Calculate IoU for each class
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_flat == cls)
        target_cls = (target_flat == cls)
        
        intersection = torch.sum(pred_cls & target_cls).item()
        union = torch.sum(pred_cls | target_cls).item()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        ious.append(iou)
    
    return ious, sum(ious) / len(ious)


def calculate_dice_coefficient(pred_tensor, target_tensor, num_classes):
    """Calculate Dice coefficient for each class using torch operations"""
    pred_flat = pred_tensor.flatten()
    target_flat = target_tensor.flatten()
    
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (pred_flat == cls)
        target_cls = (target_flat == cls)
        
        intersection = torch.sum(pred_cls & target_cls).item()
        total = torch.sum(pred_cls).item() + torch.sum(target_cls).item()
        
        if total == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection) / total
        dice_scores.append(dice)
    
    return dice_scores, sum(dice_scores) / len(dice_scores)


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


class WBCDataModule(L.LightningDataModule):
    """Lightning DataModule for WBC dataset"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.num_workers = config.get('num_workers', 2)
        self.pin_memory = config.get('pin_memory', True)
        
    def setup(self, stage=None):
        """Setup train and validation datasets"""
        self.train_dataset = WBC_dataset(
            base_dir=self.config['root_path'],
            split="train",
            image_size=self.config['img_size'],
            train_ratio=0.8
        )
        
        self.val_dataset = WBC_dataset(
            base_dir=self.config['root_path'],
            split="val",
            image_size=self.config['img_size'],
            train_ratio=0.8
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
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
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
    callbacks = []
    
    # Model checkpoint callback for best model
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best-transunet-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=False,
        verbose=True,
        auto_insert_metric_name=False
    )
    callbacks.append(best_checkpoint_callback)
    
    # Model checkpoint callback for last model
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='last-transunet-{epoch:02d}',
        save_top_k=0,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False
    )
    callbacks.append(last_checkpoint_callback)
    
    # Model save logging callback
    model_save_callback = ModelSaveCallback()
    callbacks.append(model_save_callback)
    
    # Visualization callback
    visualization_callback = VisualizationCallback(output_dir, visualization_frequency=5)
    callbacks.append(visualization_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks

def train_wbc():
    """Main training function using PyTorch Lightning"""
    print("Starting TransUNet training on WBC dataset with PyTorch Lightning...")
    
    # Configuration
    config = {
        'dataset': 'WBC',
        'vit_name': 'R50-ViT-B_16',
        'batch_size': 4,
        'max_epochs': 40,
        'base_lr': 0.01,
        'img_size': 224,
        'seed': 1234,
        'n_skip': 3,
        'num_classes': 3,  # 0: background, 1: cytoplasm, 2: nucleus
        'root_path': '/home/tanguyen12gb/Desktop/thaigiahuy/Dataset/segmentation_WBC/Dataset 2',
        'num_workers': 2,
        'pin_memory': True,
        'weight_decay': 0.0001,
        'momentum': 0.9
    }
    
    # Create output directory
    output_dir = f"./model/WBC_2_lightning_training_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(os.path.join(output_dir, "training.log"))
    
    # Set seed for reproducibility
    L.seed_everything(config['seed'])
    
    # Log configuration
    logging.info("=" * 80)
    logging.info("Starting TransUNet Training with PyTorch Lightning")
    logging.info("=" * 80)
    logging.info(f"Dataset: {config['dataset']}")
    logging.info(f"Model: {config['vit_name']}")
    logging.info(f"Classes: {config['num_classes']}")
    logging.info(f"Image size: {config['img_size']}")
    logging.info(f"Batch size: {config['batch_size']}")
    logging.info(f"Learning rate: {config['base_lr']}")
    logging.info(f"Max epochs: {config['max_epochs']}")
    logging.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logging.info("=" * 80)
    
    # Create data module
    data_module = WBCDataModule(config)
    
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
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        deterministic=False,
        enable_model_summary=True,
        enable_progress_bar=True
    )
    
    # Start training
    logging.info("Starting training...")
    trainer.fit(model, data_module)
    
    # Training completed
    logging.info("TransUNet training completed successfully!")
    
    # Get checkpoint paths
    best_model_paths = []
    last_model_paths = []
    
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            if hasattr(callback, 'best_model_path') and callback.best_model_path:
                best_model_paths.append(callback.best_model_path)
            if hasattr(callback, 'last_model_path') and callback.last_model_path:
                last_model_paths.append(callback.last_model_path)
    
    # Log model paths
    if best_model_paths:
        logging.info(f"Best model(s) saved at:")
        for path in best_model_paths:
            logging.info(f"  {path}")
    
    if last_model_paths:
        logging.info(f"Last model(s) saved at:")
        for path in last_model_paths:
            logging.info(f"  {path}")
    
    return output_dir

if __name__ == "__main__":
    try:
        output_dir = train_wbc()
        print(f"Training completed! Models saved in: {output_dir}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
