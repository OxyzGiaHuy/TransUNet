"""
MoLEx Layer - Orchestrates main path and expert path execution

Clean separation of responsibilities:
- MoLExLayer: Coordinates between main block, router, and shape adapter
- Router: Pure expert selection
- SAHub: Pure shape adaptation
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import logging

from .router import MoLExRouter
from .sa_hub import SAHub


class MoLExLayer(nn.Module):
    """
    MoLEx Layer - Orchestrates main path and expert path
    
    ONLY responsibilities:
    1. Execute main path through main_block
    2. Coordinate expert path: Router -> SAHub -> Expert -> SAHub
    3. Combine main and expert outputs with learnable weight
    """
    
    def __init__(self,
                 main_block: nn.Module,
                 expert_pool: List[nn.Module],
                 router: MoLExRouter,
                 sa_hub: SAHub,
                 alpha: float = 0.5,
                 expert_dropout: float = 0.0):
        """
        Args:
            main_block: Main block to wrap
            expert_pool: List of expert modules
            router: Router for expert selection
            sa_hub: Shape adapter for format conversion
            alpha: Initial mixing weight (alpha * main + (1-alpha) * expert)
            expert_dropout: Dropout for expert output
        """
        super().__init__()
        
        self.main_block = main_block
        self.expert_pool = nn.ModuleList(expert_pool)
        self.router = router
        self.sa_hub = sa_hub
        
        # Learnable mixing weight
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
        # Expert dropout for regularization
        self.expert_dropout = nn.Dropout(expert_dropout) if expert_dropout > 0 else nn.Identity()
        
        # Statistics
        self.register_buffer('forward_calls', torch.tensor(0))
        self.register_buffer('expert_successes', torch.tensor(0))
        self.register_buffer('expert_failures', torch.tensor(0))
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass implementing MoLEx dual-path mechanism
        
        Flow:
        1. Main Path: x -> main_block -> main_output
        2. Expert Path: x -> router -> expert -> sa_hub -> expert_output
        3. Combine: α * main_output + (1-α) * expert_output
        
        Returns:
            Tuple of (final_output, routing_info)
        """
        self.forward_calls += 1
        
        # === MAIN PATH ===
        main_output = self._execute_main_path(x)
        
        # === EXPERT PATH ===
        expert_output, routing_info = self._execute_expert_path(x, main_output)
        
        # === COMBINATION ===
        final_output = self._combine_outputs(main_output, expert_output, routing_info)
        
        # Update routing info
        routing_info.update({
            'forward_call_count': self.forward_calls.item(),
            'learned_alpha': torch.sigmoid(self.alpha).item(),
            'expert_success_rate': (self.expert_successes / max(self.forward_calls, 1)).item()
        })
        
        return final_output, routing_info
    
    def _execute_main_path(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute main block processing
        """
        try:
            main_output = self.main_block(x)
            
            # Handle tuple outputs (some blocks return multiple values)
            if isinstance(main_output, tuple):
                main_output = main_output[0]
                
            return main_output
            
        except Exception as e:
            self.logger.warning(f"Main block failed: {e}")
            return x  # Fallback to identity
    
    def _execute_expert_path(self, x: torch.Tensor, main_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Execute expert path: Router -> Adapter -> Expert -> Adapter
        
        Args:
            x: Original input
            main_output: Output from main path (for target shape)
            
        Returns:
            Tuple of (expert_output, routing_info)
        """
        try:
            # Step 1: Router selects expert
            expert_idx, routing_probs, routing_info = self.router.select_expert(x)
            
            # Step 2: Get selected expert
            selected_expert = self.expert_pool[expert_idx]
            
            # Step 3: Adapt input for expert
            adapted_input, input_success = self.sa_hub.adapt_input(
                x, selected_expert, target_shape=None
            )
            
            if not input_success:
                routing_info['error'] = 'Input adaptation failed'
                return None, routing_info
            
            # Step 4: Execute expert
            try:
                expert_raw_output = selected_expert(adapted_input)
                
                # Handle tuple outputs
                if isinstance(expert_raw_output, tuple):
                    expert_raw_output = expert_raw_output[0]
                    
            except Exception as e:
                routing_info['error'] = f'Expert execution failed: {e}'
                return None, routing_info
            
            # Step 5: Adapt expert output back to main format
            original_type = self.sa_hub._get_block_type(x)
            adapted_expert_output, output_success = self.sa_hub.adapt_output_back(
                expert_raw_output, main_output.shape, original_type
            )
            
            if not output_success:
                routing_info['error'] = 'Output adaptation failed'
                return None, routing_info
            
            # Success
            self.expert_successes += 1
            routing_info['adaptation_successful'] = True
            
            return adapted_expert_output, routing_info
            
        except Exception as e:
            # Expert path completely failed
            self.expert_failures += 1
            routing_info = {
                'expert_idx': -1,
                'adaptation_successful': False,
                'error': f'Expert path failed: {e}',
                'fallback_to_main': True
            }
            return None, routing_info
    
    def _combine_outputs(self, main_output: torch.Tensor, 
                        expert_output: Optional[torch.Tensor],
                        routing_info: Dict[str, Any]) -> torch.Tensor:
        """
        Combine main and expert outputs with learnable mixing
        """
        if expert_output is not None:
            # Apply expert dropout
            expert_output = self.expert_dropout(expert_output)
            
            # Learnable weighted combination
            alpha_clamped = torch.sigmoid(self.alpha)  # Ensure [0,1]
            combined = alpha_clamped * main_output + (1 - alpha_clamped) * expert_output
            
            # Update contribution info
            routing_info['main_contribution'] = alpha_clamped.item()
            routing_info['expert_contribution'] = (1 - alpha_clamped).item()
            
            return combined
        else:
            # Expert failed, use only main output
            routing_info['main_contribution'] = 1.0
            routing_info['expert_contribution'] = 0.0
            routing_info['fallback_to_main'] = True
            
            return main_output
    
    def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics"""
        return {
            'forward_calls': self.forward_calls.item(),
            'expert_successes': self.expert_successes.item(),
            'expert_failures': self.expert_failures.item(),
            'success_rate': (self.expert_successes / max(self.forward_calls, 1)).item(),
            'learned_alpha': torch.sigmoid(self.alpha).item(),
            'router_stats': self.router.get_usage_statistics()
        }


# Factory function
def create_molex_layer(main_block: nn.Module,
                      expert_pool: List[nn.Module], 
                      router: MoLExRouter,
                      sa_hub: SAHub,
                      config: Dict[str, Any] = None) -> MoLExLayer:
    """
    Factory function to create MoLExLayer
    
    Args:
        main_block: Main block to wrap
        expert_pool: List of expert modules
        router: Configured router instance
        sa_hub: Configured shape adapter instance
        config: Additional configuration
        
    Returns:
        MoLExLayer instance
    """
    if config is None:
        config = {}
        
    return MoLExLayer(
        main_block=main_block,
        expert_pool=expert_pool,
        router=router,
        sa_hub=sa_hub,
        alpha=config.get('alpha', 0.5),
        expert_dropout=config.get('expert_dropout', 0.0)
    )
