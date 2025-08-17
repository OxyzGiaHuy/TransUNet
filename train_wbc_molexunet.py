#!/usr/bin/env python3
"""
Training script for MoLEx-UNet on WBC dataset

Dựa trên train_wbc_simple.py nhưng adapted cho MoLEx-UNet:
- Load MoLEx-UNet model từ vit_seg_modeling_molexunet.py
- Thêm expert usage monitoring và logging
- Load balancing loss cho MoLEx router
- Visualization của expert selection patterns
"""

import os
import sys
import time
import logging
import traceback
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to path
sys.path.append('/home/tanguyen12gb/Desktop/thaigiahuy/test_molex/TransUNet')

from datasets.dataset_wbc import WBC_dataset, RandomGenerator
from networks.vit_seg_modeling_molexunet import create_molex_unet, get_molex_config
from simple_utils import SimpleDiceLoss


def visualize_predictions_and_experts(model, dataloader, device, output_dir, epoch, num_samples=3):
    """
    Visualize model predictions và expert usage patterns
    """
    model.eval()
    vis_dir = os.path.join(output_dir, 'visualizations')
    expert_dir = os.path.join(output_dir, 'expert_analysis') 
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(expert_dir, exist_ok=True)
    
    # Color map for 3 classes
    colors = {
        0: [0, 0, 0],       # Background - Black
        1: [128, 128, 128], # Cytoplasm - Gray  
        2: [255, 255, 255]  # Nucleus - White
    }
    
    expert_usage_summary = defaultdict(list)
    
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
            
            # Get expert usage statistics
            expert_stats = model.get_expert_usage_statistics()
            
            # Store expert usage for analysis
            for block_type in ['cnn_blocks', 'transformer_blocks']:
                for block_stat in expert_stats[block_type]:
                    block_idx = block_stat['block_index']
                    stats = block_stat['stats']
                    expert_usage_summary[f'{block_type}_{block_idx}'].append(stats)
            
            # Convert to numpy for visualization
            img_np = image[0, 0].cpu().numpy()  # First channel of first batch
            label_np = label[0].cpu().numpy()
            pred_np = prediction[0].cpu().numpy()
            
            # Create colored masks
            def create_colored_mask(mask):
                colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
                for class_id, color in colors.items():
                    colored[mask == class_id] = color
                return colored
            
            # Create visualizations
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: Segmentation results
            # Original image
            axes[0, 0].imshow(img_np, cmap='gray')
            axes[0, 0].set_title(f'Input Image\\n{case_name}')
            axes[0, 0].axis('off')
            
            # Ground truth
            gt_colored = create_colored_mask(label_np)
            axes[0, 1].imshow(gt_colored)
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')
            
            # Prediction
            pred_colored = create_colored_mask(pred_np)
            axes[0, 2].imshow(pred_colored)
            axes[0, 2].set_title('Prediction')
            axes[0, 2].axis('off')
            
            # Row 2: Expert usage analysis
            # CNN blocks expert usage (may be empty)
            cnn_stats = expert_usage_summary.get('cnn_blocks_0', [])
            if cnn_stats and len(cnn_stats) > 0 and 'expert_usage' in cnn_stats[0]:
                cnn_usage = cnn_stats[0]['expert_usage']
                axes[1, 0].bar(range(len(cnn_usage)), cnn_usage)
                axes[1, 0].set_title('CNN Block Expert Usage')
                axes[1, 0].set_xlabel('Expert Index')
                axes[1, 0].set_ylabel('Usage Count')
            else:
                axes[1, 0].text(0.5, 0.5, 'No CNN MoLEx blocks', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('CNN Block Expert Usage')
            
            # Transformer blocks expert usage  
            transformer_stats = expert_usage_summary.get('transformer_blocks_0', [])
            if transformer_stats and len(transformer_stats) > 0 and 'expert_usage' in transformer_stats[0]:
                transformer_usage = transformer_stats[0]['expert_usage']
                axes[1, 1].bar(range(len(transformer_usage)), transformer_usage)
                axes[1, 1].set_title('Transformer Block Expert Usage')
                axes[1, 1].set_xlabel('Expert Index')
                axes[1, 1].set_ylabel('Usage Count')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Transformer usage data', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Transformer Block Expert Usage')
            
            # Alpha values (mixing weights)
            alpha_values = []
            for block_type in ['cnn_blocks', 'transformer_blocks']:
                for block_stat in expert_stats[block_type]:
                    stats = block_stat['stats']
                    if 'current_alpha' in stats:
                        alpha_values.append(stats['current_alpha'])
            
            if alpha_values:
                axes[1, 2].bar(range(len(alpha_values)), alpha_values)
                axes[1, 2].set_title('Alpha Values (Main vs Expert)')
                axes[1, 2].set_xlabel('Block Index')
                axes[1, 2].set_ylabel('Alpha (Main Weight)')
                axes[1, 2].set_ylim(0, 1)
            else:
                axes[1, 2].text(0.5, 0.5, 'No alpha data', ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Alpha Values (Main vs Expert)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'epoch_{epoch}_sample_{i}_{case_name}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Calculate metrics for this sample
            pixel_acc = calculate_pixel_accuracy(prediction[0:1], label[0:1])
            class_ious, mean_iou = calculate_iou(prediction[0:1], label[0:1], 3)
            
            logging.info(f'Sample {case_name} - Pixel Acc: {pixel_acc:.4f}, Mean IoU: {mean_iou:.4f}')
            logging.info(f'  Class IoUs: Background={class_ious[0]:.3f}, Cytoplasm={class_ious[1]:.3f}, Nucleus={class_ious[2]:.3f}')
    
    # Save expert usage summary
    save_expert_analysis(expert_usage_summary, expert_dir, epoch)
    
    model.train()
    logging.info(f'Saved {min(num_samples, len(dataloader))} visualization samples to {vis_dir}/')


def save_expert_analysis(expert_usage_summary, output_dir, epoch):
    """Save detailed expert usage analysis"""
    
    # Create expert usage visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot expert usage distribution
    all_cnn_usage = []
    all_transformer_usage = []
    
    for key, stats_list in expert_usage_summary.items():
        if 'cnn_blocks' in key and stats_list:
            all_cnn_usage.extend([sum(stats.get('expert_usage', [])) for stats in stats_list])
        elif 'transformer_blocks' in key and stats_list:
            all_transformer_usage.extend([sum(stats.get('expert_usage', [])) for stats in stats_list])
    
    if all_cnn_usage:
        axes[0, 0].hist(all_cnn_usage, bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('CNN Blocks Expert Usage Distribution')
        axes[0, 0].set_xlabel('Total Expert Calls')
        axes[0, 0].set_ylabel('Frequency')
    
    if all_transformer_usage:
        axes[0, 1].hist(all_transformer_usage, bins=20, alpha=0.7, color='red')
        axes[0, 1].set_title('Transformer Blocks Expert Usage Distribution')
        axes[0, 1].set_xlabel('Total Expert Calls')
        axes[0, 1].set_ylabel('Frequency')
    
    # Cross-type preference analysis
    cross_type_preferences = {'cnn_to_transformer': 0, 'transformer_to_cnn': 0}
    
    # Simplified analysis - có thể enhance thêm
    axes[1, 0].bar(cross_type_preferences.keys(), cross_type_preferences.values())
    axes[1, 0].set_title('Cross-Type Preference Pattern')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Success rate analysis
    success_rates = []
    for key, stats_list in expert_usage_summary.items():
        if stats_list:
            for stats in stats_list:
                success_rate = stats.get('success_rate', 0)
                success_rates.append(success_rate)
    
    if success_rates:
        axes[1, 1].hist(success_rates, bins=10, alpha=0.7, color='green')
        axes[1, 1].set_title('Expert Adaptation Success Rate')
        axes[1, 1].set_xlabel('Success Rate')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'expert_analysis_epoch_{epoch}.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()


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


def compute_load_balancing_loss(expert_stats, load_balance_weight=0.01):
    """
    Compute load balancing loss để khuyến khích sử dụng expert đều
    """
    total_loss = 0.0
    num_blocks = 0
    
    for block_type in ['cnn_blocks', 'transformer_blocks']:
        for block_stat in expert_stats[block_type]:
            stats = block_stat['stats']
            expert_usage = stats.get('expert_usage', [])
            
            if expert_usage and len(expert_usage) > 1:
                # Convert to tensor
                usage_tensor = torch.tensor(expert_usage, dtype=torch.float32)
                
                # Normalize to get probabilities
                if usage_tensor.sum() > 0:
                    probs = usage_tensor / usage_tensor.sum()
                    
                    # Compute entropy (higher is better for load balancing)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                    
                    # Convert to loss (lower entropy = higher loss)
                    max_entropy = torch.log(torch.tensor(len(expert_usage), dtype=torch.float32))
                    load_balance_loss = max_entropy - entropy
                    
                    total_loss += load_balance_loss
                    num_blocks += 1
    
    if num_blocks > 0:
        return (total_loss / num_blocks) * load_balance_weight
    return torch.tensor(0.0)


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


def train_wbc_molexunet():
    """Training function for MoLEx-UNet on WBC dataset"""
    print("Starting MoLEx-UNet training on WBC dataset...")
    
    # Configuration
    args = {
        'dataset': 'WBC',
        'model_name': 'MoLEx-UNet',
        'vit_name': 'R50-ViT-B_16',
        'batch_size': 4,
        'max_epochs': 20,
        'base_lr': 0.01,
        'img_size': 224,
        'seed': 1234,
        'n_skip': 3,
        'num_classes': 3,  # 0: background, 1: cytoplasm, 2: nucleus
        'root_path': '/home/tanguyen12gb/Desktop/thaigiahuy/Dataset/segmentation_WBC/Dataset 1',
        'load_balance_weight': 0.01,  # Weight for load balancing loss
        'expert_stats_frequency': 50   # Log expert stats every N iterations
    }
    
    # Create output directory
    output_dir = f"./model/WBC_MoLEx_training_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(os.path.join(output_dir, "training.log"))
    
    # Set seeds
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['seed'])
    
    # Create dataset
    db_train = WBC_dataset(
        base_dir=args['root_path'],
        split="train",
        transform=transforms.Compose([RandomGenerator(output_size=[args['img_size'], args['img_size']])])
    )
    logging.info(f"Training dataset size: {len(db_train)}")
    
    # Create dataloader
    trainloader = DataLoader(
        db_train, 
        batch_size=args['batch_size'], 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    
    # Create MoLEx-UNet model
    try:
        config = get_molex_config()
    except:
        # Fallback to manual config creation
        from networks.vit_seg_configs import get_r50_b16_config
        config = get_r50_b16_config()
        
        # Add MoLEx specific configurations manually
        config.molex = {
            'expert_dropout': 0.1,
            'alpha': 0.7,
            'router_hidden_dim': 256,
            'bias_strength': 2.0,
            'use_residual': True
        }
    
    config.n_classes = args['num_classes']
    config.n_skip = args['n_skip']
    config.patches.grid = (int(args['img_size'] / 16), int(args['img_size'] / 16))
    
    net = create_molex_unet(
        config=config,
        img_size=args['img_size'], 
        num_classes=args['num_classes']
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    net = net.to(device)
    
    # Loss functions and optimizer
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = SimpleDiceLoss(args['num_classes'])
    optimizer = optim.SGD(net.parameters(), lr=args['base_lr'], momentum=0.9, weight_decay=0.0001)
    
    # Training
    net.train()
    iter_num = 0
    max_iterations = args['max_epochs'] * len(trainloader)
    best_loss = float('inf')
    
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")
    logging.info(f"MoLEx Expert Pool Size: {len(net.hybrid_encoder.expert_pool)}")
    
    for epoch in range(args['max_epochs']):
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_dice_loss = 0.0
        epoch_load_balance_loss = 0.0
        
        logging.info(f"Starting epoch {epoch + 1}/{args['max_epochs']}")
        
        for i, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            
            # Forward pass
            outputs = net(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            
            # MoLEx Load Balancing Loss
            expert_stats = net.get_expert_usage_statistics()
            loss_load_balance = compute_load_balancing_loss(expert_stats, args['load_balance_weight'])
            
            # Total loss
            loss = 0.4 * loss_ce + 0.4 * loss_dice + 0.2 * loss_load_balance
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping để stable training
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Learning rate decay
            lr_ = args['base_lr'] * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_ce_loss += loss_ce.item()
            epoch_dice_loss += loss_dice.item()
            epoch_load_balance_loss += loss_load_balance.item() if isinstance(loss_load_balance, torch.Tensor) else 0
            iter_num += 1
            
            # Log expert statistics periodically
            if iter_num % args['expert_stats_frequency'] == 0:
                with torch.no_grad():
                    predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                    
                    # Calculate metrics
                    pixel_acc = calculate_pixel_accuracy(predictions, label_batch)
                    class_ious, mean_iou = calculate_iou(predictions, label_batch, args['num_classes'])
                    class_dice, mean_dice = calculate_dice_coefficient(predictions, label_batch, args['num_classes'])
                    
                    logging.info(f'Epoch {epoch+1}, Iter {iter_num}/{max_iterations}:')
                    logging.info(f'  Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, Dice: {loss_dice.item():.4f}, LB: {loss_load_balance.item() if isinstance(loss_load_balance, torch.Tensor) else 0:.4f})')
                    logging.info(f'  Metrics: Pixel Acc: {pixel_acc:.4f}, Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}')
                    logging.info(f'  LR: {lr_:.6f}')
                    
                    # Expert usage summary
                    total_cnn_calls = sum([len(block['stats'].get('expert_usage', [])) for block in expert_stats.get('cnn_blocks', [])])
                    total_transformer_calls = sum([len(block['stats'].get('expert_usage', [])) for block in expert_stats.get('transformer_blocks', [])])
                    logging.info(f'  Expert Calls: CNN={total_cnn_calls}, Transformer={total_transformer_calls}')
        
        # Calculate epoch-end metrics on full dataset
        with torch.no_grad():
            net.eval()
            total_pixel_acc = 0.0
            total_mean_iou = 0.0
            total_mean_dice = 0.0
            num_batches = 0
            
            for sampled_batch in trainloader:
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                
                outputs = net(image_batch)
                predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                
                pixel_acc = calculate_pixel_accuracy(predictions, label_batch)
                _, mean_iou = calculate_iou(predictions, label_batch, args['num_classes'])
                _, mean_dice = calculate_dice_coefficient(predictions, label_batch, args['num_classes'])
                
                total_pixel_acc += pixel_acc
                total_mean_iou += mean_iou
                total_mean_dice += mean_dice
                num_batches += 1
            
            net.train()  # Switch back to training mode
            
            # Calculate averages
            avg_pixel_acc = total_pixel_acc / num_batches
            avg_mean_iou = total_mean_iou / num_batches
            avg_mean_dice = total_mean_dice / num_batches
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / len(trainloader)
        avg_ce_loss = epoch_ce_loss / len(trainloader)
        avg_dice_loss = epoch_dice_loss / len(trainloader)
        avg_load_balance_loss = epoch_load_balance_loss / len(trainloader)
        
        logging.info('=' * 100)
        logging.info(f'MoLEx-UNet Epoch {epoch + 1}/{args["max_epochs"]} Summary:')
        logging.info(f'  Average Total Loss: {avg_epoch_loss:.4f}')
        logging.info(f'  Average CE Loss: {avg_ce_loss:.4f}')
        logging.info(f'  Average Dice Loss: {avg_dice_loss:.4f}')
        logging.info(f'  Average Load Balance Loss: {avg_load_balance_loss:.4f}')
        logging.info(f'  Average Pixel Accuracy: {avg_pixel_acc:.4f} ({avg_pixel_acc*100:.2f}%)')
        logging.info(f'  Average Mean IoU: {avg_mean_iou:.4f} ({avg_mean_iou*100:.2f}%)')
        logging.info(f'  Average Mean Dice: {avg_mean_dice:.4f} ({avg_mean_dice*100:.2f}%)')
        
        # Expert usage summary for epoch
        final_expert_stats = net.get_expert_usage_statistics()
        logging.info(f'  Total Expert Pool Size: {len(net.hybrid_encoder.expert_pool)}')
        logging.info(f'  CNN Blocks with MoLEx: {len(final_expert_stats.get("cnn_blocks", []))}')
        logging.info(f'  Transformer Blocks with MoLEx: {len(final_expert_stats.get("transformer_blocks", []))}')
        logging.info('=' * 100)
        
        # Save only best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(net.state_dict(), best_model_path)
            logging.info(f"New best MoLEx-UNet model saved: {best_loss:.4f}")
        
        # Generate visualizations every 5 epochs
        if (epoch + 1) % 5 == 0:
            logging.info("Generating prediction and expert analysis visualizations...")
            visualize_predictions_and_experts(net, trainloader, device, output_dir, epoch + 1, num_samples=3)
    
    # Final model save
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save(net.state_dict(), final_model_path)
    logging.info(f"Final MoLEx-UNet model saved to {final_model_path}")
    
    # Final expert statistics
    final_stats = net.get_expert_usage_statistics()
    logging.info("=== FINAL EXPERT USAGE STATISTICS ===")
    logging.info(f"Expert Pool Size: {len(net.hybrid_encoder.expert_pool)}")
    for block_type in ['cnn_blocks', 'transformer_blocks']:
        logging.info(f"\\n{block_type.upper()}:")
        for block_stat in final_stats[block_type]:
            block_idx = block_stat['block_index']
            stats = block_stat['stats']
            logging.info(f"  Block {block_idx}: Success Rate: {stats.get('success_rate', 0):.3f}, "
                        f"Alpha: {stats.get('current_alpha', 0):.3f}, "
                        f"Total Calls: {stats.get('total_calls', 0)}")
    
    logging.info("MoLEx-UNet training completed successfully!")
    
    return output_dir


if __name__ == "__main__":
    try:
        output_dir = train_wbc_molexunet()
        print(f"MoLEx-UNet training completed! Models saved in: {output_dir}")
    except Exception as e:
        print(f"MoLEx-UNet training failed: {e}")
        traceback.print_exc()
