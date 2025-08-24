import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging

class SimpleDiceLoss(nn.Module):
    """Simple Dice Loss implementation without external dependencies"""
    def __init__(self, n_classes, softmax=True):
        super(SimpleDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.softmax = softmax

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = F.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class CombinedLoss(nn.Module):
    """Combined Cross Entropy and Dice Loss"""
    def __init__(self, alpha=0.5, n_classes=2):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.dice = SimpleDiceLoss(n_classes)

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss

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
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_flat == cls)
        target_cls = (target_flat == cls)
        intersection = torch.sum(pred_cls & target_cls).item()
        union = torch.sum(pred_cls | target_cls).item()
        if union == 0:
            iou = 1.0
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
            dice = 1.0
        else:
            dice = (2 * intersection) / total
        dice_scores.append(dice)
    return dice_scores, sum(dice_scores) / len(dice_scores)

def visualize_predictions(model, dataloader, device, output_dir, epoch, num_samples=5):
    """Visualize model predictions and save them"""
    model.eval()
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Color map for 2 classes (binary segmentation)
    colors = {
        0: [0, 0, 0],       # Background - Black
        1: [255, 255, 255]  # Nuclei - White
    }
    
    def colorize(mask):
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            color_mask[mask == class_id] = color
        return color_mask

    def overlay_mask_on_image(image, mask, color, alpha=0.4):
        """
        Overlay a single-channel mask on an RGB image.
        image: HxWx3, float [0,1]
        mask: HxW, int (0 or 1)
        color: (R,G,B) 0-255
        alpha: transparency
        """
        overlay = image.copy()
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * (np.array(color) / 255.0)
        return overlay

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images = sample['image'].to(device)
            masks = sample['label'].to(device)

            # Get predictions
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            # Convert to numpy for visualization
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            predictions_np = predictions.cpu().numpy()

            # Create visualization
            batch_size = images.shape[0]
            fig, axes = plt.subplots(batch_size, 4, figsize=(20, 5*batch_size))

            if batch_size == 1:
                axes = axes.reshape(1, -1)

            for j in range(batch_size):
                # Original image (denormalize)
                img = images_np[j].transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)

                # Ground truth and predicted masks
                gt_mask = colorize(masks_np[j])
                pred_mask = colorize(predictions_np[j])

                # Overlay GT and prediction on image
                overlay_gt = overlay_mask_on_image(img, masks_np[j], color=[0,255,0], alpha=0.4)  # GT: green
                overlay_pred = overlay_mask_on_image(overlay_gt, predictions_np[j], color=[255,0,0], alpha=0.4)  # Pred: red

                # Plot
                axes[j, 0].imshow(img)
                axes[j, 0].set_title('Original Image')
                axes[j, 0].axis('off')

                axes[j, 1].imshow(gt_mask)
                axes[j, 1].set_title('Ground Truth')
                axes[j, 1].axis('off')

                axes[j, 2].imshow(pred_mask)
                axes[j, 2].set_title('Predicted Mask')
                axes[j, 2].axis('off')

                axes[j, 3].imshow(overlay_pred)
                axes[j, 3].set_title('Overlay (GT: green, Pred: red)')
                axes[j, 3].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'visualizations', f'epoch_{epoch}_batch_{i}.png'))
            plt.close()
    
    model.train()
    logging.info(f'Saved {min(num_samples, len(dataloader))} visualization samples to {output_dir}/visualizations/')
