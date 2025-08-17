"""
SAHub - Shape-Adapting Hub for MoLEx system
Pure shape adaptation component - only handles tensor format/shape transformations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging


class SAHub(nn.Module):
    """
    Shape-Adapting Hub - Pure shape adaptation component
    
    ONLY responsibilities:
    1. adapt_input: Transform input tensor to match expert's expected format
    2. adapt_output_back: Transform expert output back to original format
    """
    
    def __init__(self):
        """
        Simple initialization - no external dependencies
        SAHub should be stateless shape converter
        """
        super().__init__()
        
        # Lightweight adapter cache for efficiency
        self.adapters = nn.ModuleDict()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_block_type(self, tensor: torch.Tensor) -> str:
        """
        Infer block type from tensor shape patterns
        """
        if tensor.dim() == 4:
            # For 4D tensors, distinguish between:
            # - CNN format: (B,C,H,W) where C is typically small and H,W are large
            # - Swin format: (B,H,W,C) where H,W are small and C is large
            
            B, dim1, dim2, dim3 = tensor.shape
            
            # Heuristic: CNN format typically has small channel count but large spatial dims
            # Swin format typically has large feature dim but small spatial dims
            if dim1 <= 2048 and dim2 >= 8 and dim3 >= 8:  # (B,C,H,W) - CNN format
                return 'cnn'
            elif dim1 >= 8 and dim2 >= 8 and dim3 <= 2048:  # (B,H,W,C) - Swin format
                return 'transformer'
            else:
                # Fallback: assume CNN if dimensions are more square-like
                return 'cnn'
                
        elif tensor.dim() == 3:  # (B,seq_len,features) - Transformer format
            return 'transformer'
        else:
            return 'cnn'  # Default fallback
    
    def _infer_expert_input_shape(self, expert_module: nn.Module, sample_input: torch.Tensor) -> Optional[torch.Size]:
        """
        Infer the expected input shape for an expert module
        """
        try:
            expert_type = self._infer_expert_type(expert_module)
            
            # For transformer blocks, typical input is (B, seq_len, hidden_size)
            if expert_type in ['transformer', 'vit']:
                # Try to find the hidden dimension from the module
                if hasattr(expert_module, 'attention') and hasattr(expert_module.attention, 'embed_dim'):
                    hidden_dim = expert_module.attention.embed_dim
                elif hasattr(expert_module, 'norm1') and hasattr(expert_module.norm1, 'normalized_shape'):
                    hidden_dim = expert_module.norm1.normalized_shape[0]
                elif hasattr(expert_module, 'ffn') and hasattr(expert_module.ffn, 'fc1'):
                    hidden_dim = expert_module.ffn.fc1.in_features
                elif hasattr(expert_module, 'mlp') and hasattr(expert_module.mlp, 'fc1'):
                    hidden_dim = expert_module.mlp.fc1.in_features
                else:
                    # Common default for ViT/Transformer
                    hidden_dim = 768
                
                # Estimate sequence length from spatial dimensions if input is CNN format
                if sample_input.dim() == 4:  # (B, C, H, W)
                    B, C, H, W = sample_input.shape
                    seq_len = H * W
                    return torch.Size([B, seq_len, hidden_dim])
                elif sample_input.dim() == 3:  # Already sequence format
                    B, seq_len, _ = sample_input.shape
                    return torch.Size([B, seq_len, hidden_dim])
            
            # For CNN blocks, keep similar spatial structure
            elif expert_type == 'cnn':
                if sample_input.dim() == 4:
                    # Try to infer channel requirements
                    if hasattr(expert_module, 'conv1') and hasattr(expert_module.conv1, 'in_channels'):
                        in_channels = expert_module.conv1.in_channels
                    elif hasattr(expert_module, 'conv') and hasattr(expert_module.conv, 'in_channels'):
                        in_channels = expert_module.conv.in_channels
                    else:
                        # Keep original channels
                        in_channels = sample_input.shape[1]
                    
                    B, _, H, W = sample_input.shape
                    return torch.Size([B, in_channels, H, W])
                    
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to infer expert input shape: {e}")
            return None

    def _infer_expert_type(self, expert_module) -> str:
        """
        Determine expert type from module characteristics
        """
        # Check common CNN indicators
        if isinstance(expert_module, (nn.Conv2d, nn.ConvTranspose2d)):
            return 'cnn'
        if hasattr(expert_module, 'conv') and not hasattr(expert_module, 'attention'):
            return 'cnn'
        
        # Check Transformer/ViT indicators  
        if hasattr(expert_module, 'attention') or hasattr(expert_module, 'self_attn'):
            return 'transformer'  # Use 'transformer' for (B,N,C) format
        if hasattr(expert_module, 'attn'):
            return 'transformer'
        
        # Check class name patterns for better hybrid detection
        module_name = expert_module.__class__.__name__.lower()
        if any(name in module_name for name in ['conv', 'resnet', 'efficient', 'mobile']):
            return 'cnn'
        elif any(name in module_name for name in ['vit', 'vision', 'transformer']):
            return 'transformer'  # Use 'transformer' for (B,N,C) format
        elif any(name in module_name for name in ['swin']):
            return 'swin'  # Swin-style (B,H,W,C)
        elif any(name in module_name for name in ['attention', 'block']):
            # Default transformer blocks to 'transformer' style for hybrid scenarios
            return 'transformer'
        
        return 'cnn'  # Default fallback
    
    def _create_adapter(self, in_features: int, out_features: int, 
                       adapter_type: str = 'conv') -> nn.Module:
        """
        Create simple feature adapter
        """
        if adapter_type == 'conv':
            return nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, bias=False),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True)
            )
        else:  # linear
            return nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.LayerNorm(out_features),
                nn.ReLU(inplace=True)
            )
    
    def _format_cnn_to_transformer(self, x: torch.Tensor, target_format: str = 'standard') -> torch.Tensor:
        """
        Convert CNN (B,C,H,W) to Transformer format
        
        Supports bidirectional conversion:
        - 'transformer' or 'sequence': (B,C,H,W) -> (B,HW,C) for Transformer-style (including ViT)
        - 'swin': (B,C,H,W) -> (B,H,W,C) for Swin transformers  
        - 'standard': Keep (B,C,H,W) for CNN-like transformers
        
        This enables ResNet (B,C,H,W) -> Transformer (B,N,C) conversion
        """
        if target_format in ['transformer', 'sequence']:  # (B,C,H,W) -> (B,HW,C) for Transformer
            B, C, H, W = x.shape
            return x.flatten(2).transpose(1, 2).contiguous()  # (B,C,H*W) -> (B,H*W,C)
        elif target_format == 'swin':  # (B,C,H,W) -> (B,H,W,C) for Swin
            return x.permute(0, 2, 3, 1).contiguous()
        else:  # Keep (B,C,H,W) for standard/CNN-like transformers
            return x
    
    def _format_transformer_to_cnn(self, x: torch.Tensor, source_format: str = 'standard', target_spatial_dims: tuple = None) -> torch.Tensor:
        """
        Convert Transformer format back to CNN (B,C,H,W)
        
        Supports bidirectional conversion:
        - 'transformer' or 'sequence': (B,HW,C) -> (B,C,H,W) from Transformer-style (including ViT)
        - 'swin': (B,H,W,C) -> (B,C,H,W) from Swin
        - 'standard': Already (B,C,H,W) format
        
        This enables Transformer (B,N,C) -> ResNet (B,C,H,W) conversion
        """
        if source_format in ['transformer', 'sequence'] and x.dim() == 3:  # (B,HW,C) -> (B,C,H,W)
            B, HW, C = x.shape
            
            # Use target spatial dimensions if provided
            if target_spatial_dims is not None:
                H, W = target_spatial_dims
                if H * W != HW:
                    # Need to interpolate sequence length to match target spatial dims
                    # Reshape to spatial format first, then interpolate
                    sqrt_HW = int(HW ** 0.5)
                    temp_H = temp_W = sqrt_HW
                    if temp_H * temp_W == HW:
                        # Reshape to (B, C, temp_H, temp_W)
                        temp_spatial = x.transpose(1, 2).view(B, C, temp_H, temp_W).contiguous()
                        # Interpolate to target (H, W)
                        import torch.nn.functional as F
                        adapted = F.interpolate(temp_spatial, size=(H, W), mode='bilinear', align_corners=False)
                        return adapted
                    else:
                        self.logger.warning(f"Cannot interpolate non-square sequence {HW} to {H}x{W}")
                
                return x.transpose(1, 2).view(B, C, H, W).contiguous()
            else:
                # Original logic - use square root approximation
                H = W = int(HW ** 0.5)
                if H * W == HW:
                    return x.transpose(1, 2).view(B, C, H, W).contiguous()  # (B,HW,C) -> (B,C,H,W)
                else:
                    # Handle non-square feature maps
                    # Try common aspect ratios or use provided spatial dimensions
                    self.logger.warning(f"Non-square sequence length {HW}, using sqrt approximation")
                    H = int(HW ** 0.5)
                    W = HW // H
                    return x.transpose(1, 2).view(B, C, H, W).contiguous()
        elif source_format == 'swin' and x.dim() == 4:  # (B,H,W,C) -> (B,C,H,W)
            return x.permute(0, 3, 1, 2).contiguous()
        return x
    
    def _format_adapting(self, input_tensor: torch.Tensor, expert_module: nn.Module) -> Tuple[torch.Tensor, bool]:
        """
        Step 1: Format Adapting - Convert tensor format based on input/expert type mismatch
        
        Handles conversions:
        - CNN (B,C,H,W) -> Transformer (B,N,C) where N = H*W
        - Transformer (B,N,C) -> CNN (B,C,H,W) 
        - CNN (B,C,H,W) -> Swin (B,H,W,C)
        - Swin (B,H,W,C) -> CNN (B,C,H,W)
        
        Args:
            input_tensor: Input tensor
            expert_module: Target expert module
            
        Returns:
            Tuple of (format_adapted_tensor, success_flag)
        """
        try:
            input_type = self._get_block_type(input_tensor)
            expert_type = self._infer_expert_type(expert_module)
            
            # No format conversion needed if types match
            if input_type == expert_type:
                return input_tensor, True
            
            # Format conversion based on type mismatch
            if input_type == 'cnn' and expert_type in ['transformer', 'vit']:
                # CNN (B,C,H,W) -> Transformer (B,N,C)
                adapted = self._format_cnn_to_transformer(input_tensor, 'transformer')
                
            elif input_type == 'cnn' and expert_type == 'swin':
                # CNN (B,C,H,W) -> Swin (B,H,W,C)
                adapted = self._format_cnn_to_transformer(input_tensor, 'swin')
                
            elif input_type == 'transformer' and expert_type == 'cnn':
                # Transformer (B,N,C) -> CNN (B,C,H,W)
                if input_tensor.dim() == 4 and input_tensor.shape[-1] < input_tensor.shape[1]:
                    # Swin (B,H,W,C) -> CNN (B,C,H,W)
                    adapted = self._format_transformer_to_cnn(input_tensor, 'swin')
                elif input_tensor.dim() == 3:
                    # Transformer (B,N,C) -> CNN (B,C,H,W)
                    adapted = self._format_transformer_to_cnn(input_tensor, 'transformer')
                else:
                    adapted = input_tensor
                    
            elif input_type == 'transformer' and expert_type == 'swin':
                # Transformer (B,N,C) -> Swin (B,H,W,C)
                # First convert to CNN, then to Swin
                cnn_tensor = self._format_transformer_to_cnn(input_tensor, 'transformer')
                adapted = self._format_cnn_to_transformer(cnn_tensor, 'swin')
                
            elif input_type == 'swin' and expert_type in ['transformer', 'vit']:
                # Swin (B,H,W,C) -> Transformer (B,N,C)
                # First convert to CNN, then to Transformer
                cnn_tensor = self._format_transformer_to_cnn(input_tensor, 'swin')
                adapted = self._format_cnn_to_transformer(cnn_tensor, 'transformer')
                
            else:
                # No conversion needed or unsupported conversion
                adapted = input_tensor
            
            return adapted, True
            
        except Exception as e:
            self.logger.warning(f"Format adaptation failed: {e}")
            return input_tensor, False
    
    def _resolution_adapting(self, tensor: torch.Tensor, target_shape: torch.Size) -> Tuple[torch.Tensor, bool]:
        """
        Step 2: Resolution Adapting - Adjust tensor dimensions to match target shape
        
        Handles dimension adjustments:
        - Spatial resolution: interpolation for (B,C,H,W) formats
        - Sequence length: projection for (B,N,C) formats  
        - Channel/Feature dimensions: learnable projection
        
        Example:
        (3,1024,64) -> (3,768,64): sequence length 1024→768, keep feature dim 64
        (3,64,32,32) -> (3,128,16,16): channels 64→128, spatial 32x32→16x16
        
        Args:
            tensor: Format-adapted tensor
            target_shape: Target shape to match
            
        Returns:
            Tuple of (resolution_adapted_tensor, success_flag)
        """
        try:
            if tensor.shape == target_shape:
                return tensor, True
            
            adapted = tensor
            
            # Handle different format cases
            if tensor.dim() == 4 and len(target_shape) == 4:
                # CNN format (B,C,H,W) -> (B,C',H',W')
                adapted = self._adapt_cnn_resolution(adapted, target_shape)
                
            elif tensor.dim() == 3 and len(target_shape) == 3:
                # Transformer format (B,N,C) -> (B,N',C')
                adapted = self._adapt_transformer_resolution(adapted, target_shape)
                
            else:
                self.logger.warning(f"Unsupported resolution adaptation: {tensor.shape} -> {target_shape}")
                return tensor, False
            
            return adapted, True
            
        except Exception as e:
            self.logger.warning(f"Resolution adaptation failed: {e}")
            return tensor, False
    
    def _adapt_cnn_resolution(self, tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """
        Adapt CNN format (B,C,H,W) -> (B,C',H',W')
        """
        B, C, H, W = tensor.shape
        target_B, target_C, target_H, target_W = target_shape
        
        adapted = tensor
        
        # 1. Spatial resolution adaptation
        if (H, W) != (target_H, target_W):
            adapted = self._adapt_spatial_resolution(adapted, (target_H, target_W))
        
        # 2. Channel adaptation
        if C != target_C:
            adapted = self._adapt_channels(adapted, target_C)
        
        return adapted
    
    def _adapt_transformer_resolution(self, tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """
        Adapt Transformer format (B,N,C) -> (B,N',C')
        """
        B, N, C = tensor.shape
        target_B, target_N, target_C = target_shape
        
        adapted = tensor
        
        # 1. Sequence length adaptation
        if N != target_N:
            adapted = self._adapt_sequence_length(adapted, target_N)
        
        # 2. Feature dimension adaptation  
        if C != target_C:
            adapted = self._adapt_channels(adapted, target_C)
        
        return adapted
    
    def _adapt_sequence_length(self, tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Adapt sequence length for Transformer format (B,N,C) -> (B,N',C)
        """
        B, N, C = tensor.shape
        
        if N == target_length:
            return tensor
        
        # Use interpolation along sequence dimension
        # Reshape to (B,C,N) for interpolation, then back to (B,N',C)
        tensor_transposed = tensor.transpose(1, 2)  # (B,N,C) -> (B,C,N)
        tensor_expanded = tensor_transposed.unsqueeze(-1)  # (B,C,N,1)
        
        # Interpolate along sequence dimension
        adapted = F.interpolate(
            tensor_expanded, 
            size=(target_length, 1), 
            mode='bilinear', 
            align_corners=False
        )  # (B,C,N',1)
        
        adapted = adapted.squeeze(-1).transpose(1, 2)  # (B,C,N') -> (B,N',C)
        return adapted
    
    def _adapt_spatial_resolution(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Adapt spatial resolution via interpolation
        """
        if x.dim() == 4:  # (B,C,H,W) or (B,H,W,C)
            current_size = x.shape[2:4] if x.shape[1] > x.shape[-1] else x.shape[1:3]
            if current_size != target_size:
                # Ensure (B,C,H,W) format for interpolation
                if x.shape[-1] < x.shape[1]:  # (B,H,W,C) format
                    x = x.permute(0, 3, 1, 2)
                    need_permute_back = True
                else:
                    need_permute_back = False
                
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
                
                if need_permute_back:
                    x = x.permute(0, 2, 3, 1)
        return x
    
    def _adapt_channels(self, x: torch.Tensor, target_channels: int) -> torch.Tensor:
        """
        Adapt channel dimensions with learnable projection
        """
        if x.dim() == 4:  # (B,C,H,W) format
            current_channels = x.shape[1]
            if current_channels != target_channels:
                adapter_key = f"conv_{current_channels}_to_{target_channels}"
                if adapter_key not in self.adapters:
                    self.adapters[adapter_key] = self._create_adapter(
                        current_channels, target_channels, 'conv'
                    ).to(x.device)
                x = self.adapters[adapter_key](x)
        
        elif x.dim() == 3:  # (B,seq_len,features) format
            current_features = x.shape[-1]
            if current_features != target_channels:
                adapter_key = f"linear_{current_features}_to_{target_channels}"
                if adapter_key not in self.adapters:
                    self.adapters[adapter_key] = self._create_adapter(
                        current_features, target_channels, 'linear'
                    ).to(x.device)
                x = self.adapters[adapter_key](x)
        
        return x
    
    def adapt_input(self, input_tensor: torch.Tensor, 
                   expert_module: nn.Module,
                   target_shape: Optional[torch.Size] = None) -> Tuple[torch.Tensor, bool]:
        """
        Two-step shape adaptation: Format Adapting -> Resolution Adapting
        
        Step 1: Format adapting - convert between CNN (B,C,H,W) ↔ Transformer (B,N,C)
        Step 2: Resolution adapting - adjust dimensions to match target shape
        
        Example:
        CNN (3,64,32,32) -> Expert Transformer (3,768,64)
        Step 1: (3,64,32,32) --format--> (3,1024,64)  [32*32=1024]
        Step 2: (3,1024,64) --resolution--> (3,768,64) [adapt sequence length]
        
        Args:
            input_tensor: Input from current block
            expert_module: Target expert module
            target_shape: Optional expected input shape for expert
            
        Returns:
            Tuple of (adapted_tensor, success_flag)
        """
        try:
            # Step 1: Format Adapting
            format_adapted, format_success = self._format_adapting(input_tensor, expert_module)
            if not format_success:
                return input_tensor, False
            
            # Step 2: Resolution Adapting 
            # If target_shape not provided, try to infer it
            if target_shape is None:
                target_shape = self._infer_expert_input_shape(expert_module, input_tensor)
            
            if target_shape is not None:
                resolution_adapted, resolution_success = self._resolution_adapting(format_adapted, target_shape)
                if not resolution_success:
                    return format_adapted, False  # Return format-adapted if resolution fails
                return resolution_adapted, True
            
            return format_adapted, True
            
        except Exception as e:
            self.logger.warning(f"Input adaptation failed: {e}")
            return input_tensor, False
    
    def adapt_output_back(self, expert_output: torch.Tensor,
                         original_shape: torch.Size,
                         original_type: str) -> Tuple[torch.Tensor, bool]:
        """
        Two-step reverse adaptation: Resolution Adapting -> Format Adapting
        
        Step 1: Resolution adapting - adjust dimensions to match original shape constraints
        Step 2: Format adapting - convert back to original format
        
        Args:
            expert_output: Output from expert module
            original_shape: Target shape to match
            original_type: Original block type ('cnn' or 'transformer')
            
        Returns:
            Tuple of (adapted_output, success_flag)
        """
        try:
            adapted = expert_output
            
            # Handle tuple outputs from some modules
            if isinstance(adapted, tuple):
                adapted = adapted[0]
            
            # Step 1: Format Adapting Back (convert to target format first)
            format_adapted, format_success = self._format_adapting_back(adapted, original_type, original_shape)
            if not format_success:
                return expert_output, False
            
            # Step 2: Resolution Adapting (ensure exact shape match)
            if format_adapted.shape != original_shape:
                resolution_adapted, resolution_success = self._resolution_adapting(format_adapted, original_shape)
                if not resolution_success:
                    return format_adapted, False  # Return format-adapted if resolution fails
                return resolution_adapted, True
            
            return format_adapted, True
            
        except Exception as e:
            self.logger.warning(f"Output adaptation failed: {e}")
            return expert_output, False
    
    def _format_adapting_back(self, expert_output: torch.Tensor, 
                             original_type: str, 
                             original_shape: torch.Size) -> Tuple[torch.Tensor, bool]:
        """
        Format adapting back to original type
        
        Args:
            expert_output: Output from expert
            original_type: Target format type
            original_shape: Original shape for format inference
            
        Returns:
            Tuple of (format_adapted_back_tensor, success_flag)
        """
        try:
            expert_output_type = self._get_block_type(expert_output)
            
            # No conversion needed if types already match
            if expert_output_type == original_type:
                return expert_output, True
            
            adapted = expert_output
            
            if original_type == 'cnn' and expert_output_type in ['transformer', 'vit']:
                # Transformer (B,N,C) -> CNN (B,C,H,W) conversion back
                # Extract target spatial dimensions from original_shape
                target_spatial_dims = None
                if len(original_shape) == 4:
                    target_spatial_dims = (original_shape[2], original_shape[3])  # (H, W)
                
                if adapted.dim() == 4 and adapted.shape[-1] < adapted.shape[1]:
                    # Swin (B,H,W,C) -> CNN (B,C,H,W)
                    adapted = self._format_transformer_to_cnn(adapted, 'swin', target_spatial_dims)
                elif adapted.dim() == 3:
                    # Transformer (B,N,C) -> CNN (B,C,H,W)
                    adapted = self._format_transformer_to_cnn(adapted, 'transformer', target_spatial_dims)
                    
            elif original_type == 'transformer' and expert_output_type == 'cnn':
                # CNN (B,C,H,W) -> Transformer (B,N,C) conversion back
                if len(original_shape) == 4 and original_shape[-1] < original_shape[1]:
                    # CNN (B,C,H,W) -> Swin (B,H,W,C)
                    adapted = self._format_cnn_to_transformer(adapted, 'swin')
                elif len(original_shape) == 3:
                    # CNN (B,C,H,W) -> Transformer (B,N,C)
                    adapted = self._format_cnn_to_transformer(adapted, 'transformer')
                    
            elif original_type == 'swin':
                # Convert to Swin format (B,H,W,C)
                if expert_output_type == 'cnn':
                    adapted = self._format_cnn_to_transformer(adapted, 'swin')
                elif expert_output_type in ['transformer', 'vit']:
                    # Transformer -> CNN -> Swin
                    cnn_tensor = self._format_transformer_to_cnn(adapted, 'transformer')
                    adapted = self._format_cnn_to_transformer(cnn_tensor, 'swin')
            
            return adapted, True
            
        except Exception as e:
            self.logger.warning(f"Format adaptation back failed: {e}")
            return expert_output, False


# Factory function
def create_sa_hub() -> SAHub:
    """
    Factory function to create SAHub instance
    
    Returns:
        SAHub instance
    """
    return SAHub()
