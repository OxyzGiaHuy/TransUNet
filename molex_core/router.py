import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
import logging


class MoLExRouter(nn.Module):
    """
    MoLEx Router - Pure expert selection component
    
    ONLY responsibilities:
    1. select_expert: Select the best expert (top-1) from expert pool
    2. get_routing_scores: Get routing probabilities for load balancing
    """
    
    def __init__(self,
                 in_channels: int,
                 expert_pool_size: int,
                 expert_infos: List[Dict[str, Any]],
                 current_block_info: Dict[str, Any],
                 router_hidden_dim: int = 256,
                 bias_strength: float = 2.0,
                 load_balance_factor: float = 0.01):
        """
        Args:
            in_channels: Input channel dimension
            expert_pool_size: Number of experts in the pool
            expert_infos: List of expert information [{'type': 'cnn'/'transformer', 'name': '...'}]
            current_block_info: Current block info {'type': 'cnn'/'transformer', 'name': '...'}
            router_hidden_dim: Hidden dimension for router MLP
            bias_strength: Strength of cross-type preference bias
            load_balance_factor: Factor for load balancing loss
        """
        super().__init__()
        
        self.expert_pool_size = expert_pool_size
        self.current_block_info = current_block_info
        self.expert_infos = expert_infos
        self.in_channels = in_channels
        self.load_balance_factor = load_balance_factor
        
        # Feature aggregation for different input formats
        self.feature_aggregator = nn.AdaptiveAvgPool2d((1, 1))
        
        # Adaptive projection for mismatched input channels
        self.adaptive_projection = None
        
        # Router MLP: Maps aggregated features to expert scores  
        self.router_mlp = nn.Sequential(
            nn.Linear(in_channels, router_hidden_dim),
            nn.LayerNorm(router_hidden_dim),  # Add layer norm for stability
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(router_hidden_dim, router_hidden_dim // 2),
            nn.LayerNorm(router_hidden_dim // 2),  # Add layer norm for stability
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(router_hidden_dim // 2, expert_pool_size)
        )
        
        # Cross-type preference bias - core MoLEx intelligence
        self.expert_type_bias = nn.Parameter(torch.zeros(expert_pool_size))
        self._initialize_cross_type_bias(bias_strength)
        
        # Statistics for monitoring
        self.register_buffer('expert_usage_count', torch.zeros(expert_pool_size))
        self.register_buffer('total_calls', torch.tensor(0))
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _initialize_weights(self):
        """Initialize router MLP weights for numerical stability"""
        for module in self.router_mlp.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization with smaller gain
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _initialize_cross_type_bias(self, bias_strength: float):
        """
        Initialize bias to implement cross-type preference:
        - CNN block -> prefer Transformer expert (for global context)
        - Transformer block -> prefer CNN expert (for local details)
        """
        current_type = self.current_block_info.get('type', 'cnn').lower()
        
        with torch.no_grad():
            for i, expert_info in enumerate(self.expert_infos):
                expert_type = expert_info.get('type', 'cnn').lower()
                
                # Determine expert category for hybrid ResNet->ViT scenarios
                is_transformer_expert = expert_type in ['transformer', 'swin', 'vit', 'deit', 'beit']
                is_cnn_expert = expert_type in ['cnn', 'resnet', 'efficientnet', 'densenet', 'conv']
                
                # Apply cross-type preference bias for hybrid architectures
                if current_type == 'cnn' and is_transformer_expert:
                    # CNN block prefers Transformer/ViT expert (for global context)
                    self.expert_type_bias[i] = bias_strength
                elif current_type in ['transformer', 'swin', 'vit'] and is_cnn_expert:
                    # Transformer/ViT block prefers CNN expert (for detailed anchoring)
                    self.expert_type_bias[i] = bias_strength
                else:
                    # Same type or unknown -> neutral bias
                    self.expert_type_bias[i] = 0.0
    
    def _aggregate_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate input tensor to fixed-size feature vector
        """
        if x.dim() == 4:  # CNN format (B,C,H,W)
            aggregated = self.feature_aggregator(x).squeeze(-1).squeeze(-1)  # (B,C)
        elif x.dim() == 3:  # Transformer format (B,N,D) - take mean over sequence
            aggregated = x.mean(dim=1)  # (B,D)
        else:  # Fallback
            aggregated = x.flatten(1)  # (B, flattened)
        
        return aggregated
    
    def _handle_channel_mismatch(self, features: torch.Tensor) -> torch.Tensor:
        """
        Handle input channel dimension mismatch
        """
        actual_channels = features.shape[1]
        if actual_channels != self.in_channels:
            if self.adaptive_projection is None:
                self.adaptive_projection = nn.Linear(actual_channels, self.in_channels).to(features.device)
            features = self.adaptive_projection(features)
        
        return features
    
    def select_expert(self, input_tensor: torch.Tensor) -> Tuple[int, torch.Tensor, Dict[str, Any]]:
        """
        Select the best expert (top-1) from expert pool
        
        Args:
            input_tensor: Input tensor from current block
            
        Returns:
            Tuple of (expert_idx: int, routing_probabilities: Tensor, routing_info: Dict)
        """
        self.total_calls += 1
        
        try:
            # 1. Aggregate features
            aggregated_features = self._aggregate_features(input_tensor)
            
            # 2. Handle channel mismatch
            aggregated_features = self._handle_channel_mismatch(aggregated_features)
            
            # 3. Compute raw routing scores
            raw_scores = self.router_mlp(aggregated_features)  # (B, expert_pool_size)
            
            # 4. Apply cross-type preference bias
            biased_scores = raw_scores + self.expert_type_bias.unsqueeze(0)
            
            # 5. Add exploration noise during training
            if self.training:
                noise = torch.randn_like(biased_scores) * 0.05
                biased_scores = biased_scores + noise
            
            # 6. Compute expert probabilities
            expert_probs = F.softmax(biased_scores, dim=-1)  # (B, expert_pool_size)
            
            # 7. Select best expert
            if self.training:
                # Training: sample for exploration
                expert_indices = torch.multinomial(expert_probs, 1).squeeze(-1)  # (B,)
            else:
                # Inference: select most probable
                expert_indices = torch.argmax(expert_probs, dim=-1)  # (B,)
            
            # For batch processing, take the first sample's expert
            selected_expert_idx = expert_indices[0].item()
            
            # 8. Update usage statistics
            self.expert_usage_count[selected_expert_idx] += 1
            
            # 9. Prepare routing information
            routing_info = {
                'expert_idx': selected_expert_idx,
                'expert_type': self.expert_infos[selected_expert_idx].get('type', 'unknown'),
                'expert_name': self.expert_infos[selected_expert_idx].get('name', f'expert_{selected_expert_idx}'),
                'routing_prob': expert_probs[0, selected_expert_idx].item(),
                'current_block_type': self.current_block_info.get('type', 'unknown'),
                'is_cross_type_selection': self._is_cross_type_selection(selected_expert_idx),
                'load_balance_loss': self.compute_load_balance_loss(expert_probs)
            }
            
            return selected_expert_idx, expert_probs, routing_info
                
        except Exception as e:
            self.logger.warning(f"Expert selection failed: {e}")
            # Fallback to first expert
            fallback_info = {
                'expert_idx': 0,
                'expert_type': 'fallback',
                'expert_name': 'fallback_expert_0',
                'routing_prob': 1.0,
                'current_block_type': self.current_block_info.get('type', 'unknown'),
                'is_cross_type_selection': False,
                'error': str(e),
                'is_fallback': True
            }
            return 0, torch.ones(1, self.expert_pool_size) / self.expert_pool_size, fallback_info   


    def _is_cross_type_selection(self, expert_idx: int) -> bool:
        """Check if the selection follows cross-type preference"""
        current_type = self.current_block_info.get('type', '').lower()
        expert_type = self.expert_infos[expert_idx].get('type', '').lower()
        
        # Check if it's a cross-type selection for hybrid scenarios
        current_is_cnn = current_type == 'cnn'
        current_is_transformer = current_type in ['transformer', 'swin', 'vit']
        expert_is_transformer = expert_type in ['transformer', 'swin', 'vit', 'deit', 'beit']
        expert_is_cnn = expert_type in ['cnn', 'resnet', 'efficientnet', 'densenet', 'conv']
        
        return (current_is_cnn and expert_is_transformer) or (current_is_transformer and expert_is_cnn)
    
    def compute_load_balance_loss(self, expert_probs: torch.Tensor) -> float:
        """
        Compute load balancing loss based on Switch Transformer formula
        
        Switch Transformer formula:
        load_balance_loss = N * Σ(f_i * P_i)
        where:
        - N = number of experts
        - f_i = fraction of tokens routed to expert i
        - P_i = probability mass assigned to expert i
        
        Args:
            expert_probs: Expert probability distribution (B, expert_pool_size)
            
        Returns:
            Load balance loss value
        """
        if expert_probs.dim() != 2:
            return 0.0
        
        batch_size = expert_probs.size(0)
        
        # P_i: Average probability assigned to expert i across the batch
        P = expert_probs.mean(dim=0)  # (expert_pool_size,)
        
        # f_i: Fraction of tokens that would be routed to expert i 
        # (based on argmax selection for each token)
        expert_assignments = torch.argmax(expert_probs, dim=-1)  # (B,)
        f = torch.zeros_like(P)  # (expert_pool_size,)
        
        for i in range(self.expert_pool_size):
            f[i] = (expert_assignments == i).float().mean()
        
        # Switch Transformer load balance loss: N * Σ(f_i * P_i)
        load_balance_loss = self.expert_pool_size * (f * P).sum()
        
        return self.load_balance_factor * load_balance_loss.item()
    
    def get_routing_scores(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get raw routing scores for all experts (for analysis/debugging)
        
        Args:
            input_tensor: Input tensor
            
        Returns:
            Raw routing scores (B, expert_pool_size)
        """
        try:
            aggregated_features = self._aggregate_features(input_tensor)
            aggregated_features = self._handle_channel_mismatch(aggregated_features)
            raw_scores = self.router_mlp(aggregated_features)
            biased_scores = raw_scores + self.expert_type_bias.unsqueeze(0)
            return F.softmax(biased_scores, dim=-1)
        except Exception as e:
            self.logger.warning(f"Failed to get routing scores: {e}")
            return torch.ones(1, self.expert_pool_size) / self.expert_pool_size
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get expert usage statistics for monitoring"""
        total_calls = self.total_calls.item()
        usage_counts = self.expert_usage_count.cpu().numpy()
        
        if total_calls > 0:
            usage_ratios = usage_counts / total_calls
        else:
            usage_ratios = usage_counts
        
        # Count usage by type for hybrid architectures
        cnn_usage = sum(usage_counts[i] for i, info in enumerate(self.expert_infos) 
                       if info.get('type', '').lower() in ['cnn', 'resnet', 'efficientnet', 'densenet'])
        transformer_usage = sum(usage_counts[i] for i, info in enumerate(self.expert_infos) 
                               if info.get('type', '').lower() in ['transformer', 'swin', 'vit', 'deit', 'beit'])
        
        current_type = self.current_block_info.get('type', 'unknown')
        cross_type_usage = transformer_usage if current_type == 'cnn' else cnn_usage
        
        return {
            'total_calls': total_calls,
            'expert_usage_count': usage_counts.tolist(),
            'expert_usage_ratio': usage_ratios.tolist(),
            'cnn_expert_usage': int(cnn_usage),
            'transformer_expert_usage': int(transformer_usage),
            'current_block_type': current_type,
            'cross_type_selections': int(cross_type_usage),
            'cross_type_ratio': cross_type_usage / total_calls if total_calls > 0 else 0.0
        }


# Factory function
def create_molex_router(in_channels: int,
                       expert_pool_size: int,
                       expert_infos: List[Dict[str, Any]],
                       current_block_info: Dict[str, Any],
                       **kwargs) -> MoLExRouter:
    """
    Factory function to create MoLEx router
    
    Args:
        in_channels: Input channel dimension
        expert_pool_size: Number of experts
        expert_infos: Expert information list
        current_block_info: Current block information
        **kwargs: Additional arguments for MoLExRouter
        
    Returns:
        Configured MoLExRouter instance
    """
    return MoLExRouter(
        in_channels=in_channels,
        expert_pool_size=expert_pool_size,
        expert_infos=expert_infos,
        current_block_info=current_block_info,
        **kwargs
    )
