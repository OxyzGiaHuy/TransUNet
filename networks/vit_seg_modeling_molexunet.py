"""
MoLEx-UNet Implementation
Hybrid Sequential Encoder với Dynamic Expert Communication

Dựa theo thiết kế trong ghi_chu_ky_thuat.txt:
- Sequential Hybrid Encoder: CNN Backbone -> Transformer Backbone  
- Comprehensive Expert Pool chứa tất cả blocks
- MoLEx-SA mechanism cho mỗi block trong chuỗi
- Shape-Adapting Hub xử lý format conversion
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)

# Setup paths
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import MoLEx components
from molex_core.molex_layer import MoLExLayer
from molex_core.sa_hub import SAHub

# Import configs and ResNetV2
import vit_seg_configs as configs
from vit_seg_modeling import Encoder, DecoderCup, SegmentationHead
from vit_seg_modeling_resnet_skip import ResNetV2


class MoLExHybridEncoder(nn.Module):
    """
    MoLEx Hybrid Sequential Encoder
    
    Cấu trúc: CNN Backbone -> Transformer Backbone
    Mỗi block được wrap bởi MoLExLayer cho expert communication
    """
    
    def __init__(self, config, img_size=224, vis=False):
        super(MoLExHybridEncoder, self).__init__()
        self.vis = vis
        self.config = config
        
        # === 1. CNN BACKBONE ===
        self.resnet_backbone = ResNetV2(
            block_units=config.resnet.num_layers,
            width_factor=config.resnet.width_factor
        )
        
        # Extract ResNet blocks for MoLEx wrapping
        # ResNetV2 có structure: root -> body[block1, block2, block3]
        self.cnn_root = self.resnet_backbone.root
        self.cnn_block1 = self.resnet_backbone.body.block1  # width*4 channels
        self.cnn_block2 = self.resnet_backbone.body.block2  # width*8 channels  
        self.cnn_block3 = self.resnet_backbone.body.block3  # width*16 channels
        
        # === 2. TRANSFORMER BACKBONE ===  
        # Sử dụng standard ViT encoder
        self.transformer_backbone = Encoder(config, vis)
        
        # === 3. COMPREHENSIVE EXPERT POOL ===
        # Tập hợp tất cả blocks từ cả CNN và Transformer
        self._build_comprehensive_expert_pool()
        
        # === 4. SA HUB ===
        # Shape-Adapting Hub xử lý format conversion
        self.sa_hub = SAHub()
        
        # === 4.5. CHANNEL PROJECTOR ===
        # Project CNN output channels to transformer hidden size
        width = int(64 * self.config.resnet.width_factor)
        cnn_output_channels = width * 16  # 1024 with width_factor=1
        if cnn_output_channels != config.hidden_size:
            self.channel_projector = nn.Linear(cnn_output_channels, config.hidden_size)
        else:
            self.channel_projector = nn.Identity()
        
        # === 5. MOLEX LAYERS ===
        # Wrap các blocks với MoLExLayer
        self._wrap_blocks_with_molex()
        
        logger.info(f"MoLExHybridEncoder initialized with {len(self.expert_pool)} experts")
    
    def _build_comprehensive_expert_pool(self):
        """
        Xây dựng Comprehensive Expert Pool theo ghi_chu_ky_thuat.txt
        Chứa tất cả blocks từ CNN và Transformer backbone
        """
        self.expert_pool = nn.ModuleList()
        self.expert_infos = []
        self.block_type_map = {}
        expert_idx = 0
        
        # === CNN EXPERTS ===
        # Thêm các ResNet blocks vào expert pool
        # ResNetV2 width calculation: width = int(64 * width_factor)
        width = int(64 * self.config.resnet.width_factor)
        
        cnn_blocks = [
            ('cnn_block1', self.cnn_block1, width * 4),   # 256 with width_factor=1
            ('cnn_block2', self.cnn_block2, width * 8),   # 512 with width_factor=1
            ('cnn_block3', self.cnn_block3, width * 16),  # 1024 with width_factor=1
        ]
        
        for block_name, block_module, channels in cnn_blocks:
            self.expert_pool.append(block_module)
            self.expert_infos.append({
                'type': 'cnn',
                'name': block_name,
                'index': expert_idx,
                'channels': channels
            })
            self.block_type_map[expert_idx] = 'cnn'
            expert_idx += 1
            
        # === TRANSFORMER EXPERTS ===
        # Thêm các Transformer blocks vào expert pool
        for i, transformer_block in enumerate(self.transformer_backbone.layer):
            self.expert_pool.append(transformer_block)
            self.expert_infos.append({
                'type': 'transformer', 
                'name': f'transformer_block_{i}',
                'index': expert_idx,
                'channels': self.config.hidden_size
            })
            self.block_type_map[expert_idx] = 'transformer'
            expert_idx += 1
    
    def _wrap_blocks_with_molex(self):
        """
        Wrap các blocks với MoLExLayer theo MoLEx-SA mechanism
        """
        # MoLEx config
        molex_config = {
            'router': {
                'router_hidden_dim': 256,
                'bias_strength': 2.0
            },
            'alpha': 0.5,
            'expert_dropout': 0.1
        }
        
        # === WRAP CNN BLOCKS ===
        wrapped_cnn_blocks = []
        width = int(64 * self.config.resnet.width_factor)
        
        cnn_blocks = [
            ('cnn_block1', self.cnn_block1, width * 4),
            ('cnn_block2', self.cnn_block2, width * 8), 
            ('cnn_block3', self.cnn_block3, width * 16)
        ]
        
        for i, (block_name, block_module, channels) in enumerate(cnn_blocks):
            # Block info cho CNN
            block_info = {
                'type': 'cnn',
                'name': block_name,
                'channels': channels,
            }
            
            # Tạo Router và MoLExLayer
            from molex_core.router import MoLExRouter
            router = MoLExRouter(
                in_channels=channels,
                expert_pool_size=len(self.expert_pool),
                expert_infos=self.expert_infos,
                current_block_info=block_info,
                router_hidden_dim=molex_config['router']['router_hidden_dim'],
                bias_strength=molex_config['router']['bias_strength']
            )
            
            molex_layer = MoLExLayer(
                main_block=block_module,
                expert_pool=list(self.expert_pool),
                router=router,
                sa_hub=self.sa_hub,
                alpha=molex_config['alpha'],
                expert_dropout=molex_config['expert_dropout']
            )
            
            wrapped_cnn_blocks.append(molex_layer)
            # Replace the original block
            setattr(self, block_name, molex_layer)
        
        # === WRAP TRANSFORMER BLOCKS ===
        wrapped_transformer_blocks = []
        for i, transformer_block in enumerate(self.transformer_backbone.layer):
            # Block info cho Transformer
            block_info = {
                'type': 'transformer',
                'name': f'transformer_block_{i}',
                'channels': self.config.hidden_size,
            }
            
            # Tạo Router và MoLExLayer
            from molex_core.router import MoLExRouter
            router = MoLExRouter(
                in_channels=self.config.hidden_size,
                expert_pool_size=len(self.expert_pool),
                expert_infos=self.expert_infos,
                current_block_info=block_info,
                router_hidden_dim=molex_config['router']['router_hidden_dim'],
                bias_strength=molex_config['router']['bias_strength']
            )
            
            molex_layer = MoLExLayer(
                main_block=transformer_block,
                expert_pool=list(self.expert_pool),
                router=router,
                sa_hub=self.sa_hub,
                alpha=molex_config['alpha'],
                expert_dropout=molex_config['expert_dropout']
            )
            
            wrapped_transformer_blocks.append(molex_layer)
        
        # Replace transformer blocks
        self.transformer_backbone.layer = nn.ModuleList(wrapped_transformer_blocks)
        
        self.wrapped_cnn_blocks = wrapped_cnn_blocks
        self.wrapped_transformer_blocks = wrapped_transformer_blocks
    
    def forward(self, x):
        """
        Forward pass theo Sequential Hybrid: CNN -> Transformer
        """
        routing_infos = []
        attentions = []
        features = []
        
        # === CNN BACKBONE PHASE ===
        # Root processing (không wrap với MoLEx)
        x = self.cnn_root(x)  # ResNetV2 root
        features.append(x)
        
        # MaxPool với padding=1 để có output 56x56 thay vì 55x55
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        
        # CNN Blocks với MoLEx (theo ResNetV2 structure)
        x, routing_info = self.cnn_block1(x)
        routing_infos.append(routing_info)
        features.append(x)
        
        x, routing_info = self.cnn_block2(x)
        routing_infos.append(routing_info)
        features.append(x)
        
        x, routing_info = self.cnn_block3(x)
        routing_infos.append(routing_info)
        features.append(x)
        
        # === TRANSFORMER BACKBONE PHASE ===
        # x từ CNN backbone có shape (B, C, H, W) với C = width*16 (1024 with width_factor=1)
        B, C, H, W = x.shape
        
        # Transform cho Transformer: (B, C, H, W) -> (B, HW, hidden_size)
        target_hidden_size = self.config.hidden_size  # Should be 768
        
        if C != target_hidden_size:
            # Reshape to (B, HW, C) then project to (B, HW, hidden_size)
            x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
            x = self.channel_projector(x)     # (B, HW, 768)
        else:
            x = x.flatten(2).transpose(1, 2)  # (B, HW, hidden_size)
        
        # Process through transformer blocks
        for i, layer_block in enumerate(self.transformer_backbone.layer):
            if isinstance(layer_block, MoLExLayer):
                # MoLEx-enabled transformer block
                x, routing_info = layer_block(x)
                routing_infos.append(routing_info)
            else:
                # Standard transformer block từ vit_seg_modeling
                x, weights = layer_block(x)
                if self.vis:
                    attentions.append(weights)
        
        # Encoder norm (trong vit_seg_modeling đang là LayerNorm)
        if hasattr(self.transformer_backbone, 'encoder_norm'):
            encoded = self.transformer_backbone.encoder_norm(x)
        else:
            encoded = x
        
        if self.vis:
            return encoded, attentions, features, routing_infos
        else:
            return encoded, features, routing_infos


class MoLExUNet(nn.Module):
    """
    MoLEx-UNet: Hybrid Sequential Encoder với Dynamic Expert Communication
    
    Main model combining:
    - MoLExHybridEncoder (CNN -> Transformer với MoLEx-SA)
    - Standard UNet Decoder
    """
    
    def __init__(self, config, img_size=224, num_classes=3, vis=False):
        super(MoLExUNet, self).__init__()
        self.num_classes = num_classes
        self.classifier = config.classifier
        self.config = config
        
        # Hybrid Encoder với MoLEx
        self.hybrid_encoder = MoLExHybridEncoder(config, img_size, vis)
        
        # Sử dụng DecoderCup trực tiếp
        self.decoder = DecoderCup(config)
        
        # Segmentation head
        self.segmentation_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        
        logger.info(f"MoLExUNet initialized - Classes: {num_classes}, Image size: {img_size}")
        logger.info(f"Using DecoderCup with {config.n_skip} skip connections")
    
    def forward(self, x):
        # grayscale handling
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        
        # Encoder phase với MoLEx
        encoded, features, routing_infos = self.hybrid_encoder(x)
        
        # xử lý input, skip connection cho decoder
        if features is not None and len(features) >= 3:
            # DecoderCup expects skip connections theo thứ tự từ high-resolution đến low-resolution
            # features[0]: root (112x112, 64 ch)
            # features[1]: block1 (56x56, 256 ch) 
            # features[2]: block2 (28x28, 512 ch)
            # features[3]: block3 (14x14, 1024 ch)
            
            # Reorder để match DecoderCup expectations: [skip0, skip1, skip2]
            # Skip0: 28x28 (features[2]) cho stage đầu tiên sau khi upsample từ 14x14
            # Skip1: 56x56 (features[1]) cho stage thứ hai - Không cần resize nữa!
            # Skip2: 112x112 (features[0]) cho stage thứ ba
            decoder_features = [
                features[2],  # 28x28, 512 channels
                features[1],  # 56x56, 256 channels 
                features[0]   # 112x112, 64 channels
            ]
        else:
            decoder_features = None
        
        # Forward qua decoder
        x = self.decoder(encoded, decoder_features)
        
        # Apply segmentation head
        logits = self.segmentation_head(x)
        
        # Ensure output size matches input image size
        if logits.shape[2:] != (224, 224):
            logits = F.interpolate(
                logits, size=(224, 224), 
                mode='bilinear', align_corners=False
            )
        
        return logits
    
    def get_expert_usage_statistics(self):
        """Lấy thống kê sử dụng expert từ toàn bộ model"""
        stats = {
            'cnn_blocks': [],
            'transformer_blocks': []
        }
        
        # CNN block statistics
        for i, block in enumerate(self.hybrid_encoder.wrapped_cnn_blocks):
            if hasattr(block, 'get_stats'):
                stats['cnn_blocks'].append({
                    'block_index': i,
                    'stats': block.get_stats()
                })
        
        # Transformer block statistics  
        for i, block in enumerate(self.hybrid_encoder.wrapped_transformer_blocks):
            if hasattr(block, 'get_stats'):
                stats['transformer_blocks'].append({
                    'block_index': i,
                    'stats': block.get_stats()
                })
        
        return stats


def get_molex_config():
    """Get default configuration for MoLEx-UNet"""
    config = configs.get_r50_b16_config()
    
    # MoLEx specific configurations
    config.molex = {
        'expert_dropout': 0.1,
        'alpha': 0.5,
        'router_hidden_dim': 256,
        'bias_strength': 2.0,
        'use_residual': True
    }
    
    # Decoder configurations cho DecoderCup
    config.decoder_channels = [256, 128, 64, 16]
    
    # Skip connection channels (DecoderCup expects 4 elements)
    width = int(64 * getattr(config.resnet, 'width_factor', 1))
    config.skip_channels = [
        width * 8,    # 512 - từ block2 (28x28)
        width * 4,    # 256 - từ block1 (resized to 56x56)
        width,        # 64 - từ root (112x112)
        0             # Không sử dụng cho stage cuối
    ]
    
    # Number of skip connections to use
    config.n_skip = 3
    
    return config


def create_molex_unet(config, img_size=224, num_classes=3, vis=False):
    """
    Factory function để tạo MoLExUNet model
    
    Args:
        config: Model configuration
        img_size: Input image size
        num_classes: Number of output classes
        vis: Whether to return attention weights
        
    Returns:
        MoLExUNet model instance
    """
    model = MoLExUNet(
        config=config,
        img_size=img_size, 
        num_classes=num_classes,
        vis=vis
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    config = get_molex_config()
    config.n_classes = 3
    config.n_skip = 3
    
    model = create_molex_unet(config, img_size=224, num_classes=3)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
        print(f"Model output shape: {output.shape}")
        print(f"Expert pool size: {len(model.hybrid_encoder.expert_pool)}")
        
    # Print expert usage statistics
    stats = model.get_expert_usage_statistics()
    print("Expert usage statistics:", stats)
