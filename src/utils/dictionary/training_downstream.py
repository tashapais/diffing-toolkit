"""
Enhanced SAE training with downstream loss terms for better model diffing.

This implements the advisor's suggestions:
1. Downstream KL loss to ensure reconstructed differences capture behavioral changes
2. Downstream activation preservation loss to maintain computational effects
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
from omegaconf import DictConfig

def compute_downstream_kl_loss(
    base_model,
    chat_model, 
    base_activations: torch.Tensor,
    chat_activations: torch.Tensor,
    reconstructed_base: torch.Tensor,
    reconstructed_chat: torch.Tensor,
    target_layer: int,
    downstream_layers: List[int],
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute downstream KL divergence loss for model diffing.
    
    This ensures that the reconstructed activation differences actually capture
    behavioral differences between base and chat models.
    
    Args:
        base_model: The base model
        chat_model: The chat model  
        base_activations: Original base model activations [batch, seq, d_model]
        chat_activations: Original chat model activations [batch, seq, d_model]
        reconstructed_base: SAE reconstructed base activations
        reconstructed_chat: SAE reconstructed chat activations
        target_layer: Layer where SAE is applied
        downstream_layers: Layers to measure KL divergence at
        temperature: Temperature for KL computation
        
    Returns:
        KL divergence loss
    """
    kl_losses = []
    
    for layer_idx in downstream_layers:
        if layer_idx <= target_layer:
            continue  # Skip non-downstream layers
            
        with torch.no_grad():
            # Get original outputs at downstream layer
            orig_base_output = run_model_to_layer(base_model, base_activations, target_layer, layer_idx)
            orig_chat_output = run_model_to_layer(chat_model, chat_activations, target_layer, layer_idx)
            
        # Get reconstructed outputs at downstream layer  
        recon_base_output = run_model_to_layer(base_model, reconstructed_base, target_layer, layer_idx)
        recon_chat_output = run_model_to_layer(chat_model, reconstructed_chat, target_layer, layer_idx)
        
        # Compute KL divergences
        # KL between original difference and reconstructed difference
        orig_diff_logits = (orig_chat_output - orig_base_output) / temperature
        recon_diff_logits = (recon_chat_output - recon_base_output) / temperature
        
        # Apply softmax to get probability distributions
        orig_diff_probs = F.softmax(orig_diff_logits, dim=-1)
        recon_diff_log_probs = F.log_softmax(recon_diff_logits, dim=-1)
        
        # KL divergence
        kl_loss = F.kl_div(recon_diff_log_probs, orig_diff_probs, reduction='batchmean')
        kl_losses.append(kl_loss)
    
    return torch.stack(kl_losses).mean()


def compute_downstream_activation_loss(
    model,
    original_activations: torch.Tensor,
    reconstructed_activations: torch.Tensor,
    target_layer: int,
    downstream_layers: Union[str, List[int]],
    aggregation: str = "sum_abs"
) -> torch.Tensor:
    """
    Compute downstream activation preservation loss.
    
    This ensures the SAE preserves computational effects in all downstream layers,
    not just reconstruction quality at the target layer.
    
    Args:
        model: The language model
        original_activations: Original activations at target layer
        reconstructed_activations: SAE reconstructed activations
        target_layer: Layer where SAE is applied
        downstream_layers: "all_downstream" or list of specific layers
        aggregation: How to aggregate differences ("sum_abs", "mse", "max")
        
    Returns:
        Downstream activation preservation loss
    """
    if isinstance(downstream_layers, str) and downstream_layers == "all_downstream":
        # Use all layers after target_layer
        downstream_layers = list(range(target_layer + 1, model.config.num_hidden_layers))
    
    activation_losses = []
    
    for layer_idx in downstream_layers:
        with torch.no_grad():
            # Original downstream activations
            orig_downstream = run_model_to_layer(model, original_activations, target_layer, layer_idx)
            
        # Reconstructed downstream activations
        recon_downstream = run_model_to_layer(model, reconstructed_activations, target_layer, layer_idx)
        
        # Compute difference based on aggregation method
        if aggregation == "sum_abs":
            diff = torch.abs(orig_downstream - recon_downstream).sum()
        elif aggregation == "mse":  
            diff = F.mse_loss(recon_downstream, orig_downstream)
        elif aggregation == "max":
            diff = torch.abs(orig_downstream - recon_downstream).max()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
            
        activation_losses.append(diff)
    
    return torch.stack(activation_losses).mean()


def run_model_to_layer(
    model, 
    activations: torch.Tensor, 
    start_layer: int, 
    end_layer: int
) -> torch.Tensor:
    """
    Run model from start_layer to end_layer with given activations.
    
    Args:
        model: The language model
        activations: Input activations at start_layer
        start_layer: Layer to start from  
        end_layer: Layer to end at
        
    Returns:
        Activations at end_layer
    """
    # This is a simplified version - you'll need to implement this based on your model architecture
    # For transformers, you'd run through the layers from start_layer+1 to end_layer
    
    current_activations = activations
    
    for layer_idx in range(start_layer + 1, end_layer + 1):
        # Apply transformer layer
        layer = model.transformer.h[layer_idx]  # Adjust based on your model structure
        current_activations = layer(current_activations)
        
        # Extract residual stream if needed
        if hasattr(layer, 'ln_2'):  # For GPT-style models
            current_activations = current_activations[0]  # Get hidden states
            
    return current_activations


def enhanced_sae_loss(
    sae_output: torch.Tensor,
    target: torch.Tensor, 
    sae_activations: torch.Tensor,
    config: DictConfig,
    models: Optional[Dict] = None,
    original_activations: Optional[Dict] = None,
    target_layer: int = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Enhanced SAE loss with downstream terms.
    
    Args:
        sae_output: SAE reconstructed activations
        target: Target activations to reconstruct
        sae_activations: SAE latent activations
        config: Training configuration with downstream_loss settings
        models: Dict with 'base' and 'chat' models (for KL loss)
        original_activations: Dict with 'base' and 'chat' original activations
        target_layer: Layer index where SAE is applied
        
    Returns:
        Total loss and loss components dictionary
    """
    # Standard SAE losses (reconstruction + sparsity)
    reconstruction_loss = F.mse_loss(sae_output, target)
    sparsity_loss = sae_activations.norm(p=1, dim=-1).mean()
    
    total_loss = reconstruction_loss + config.training.sparsity_weight * sparsity_loss
    
    loss_components = {
        'reconstruction': reconstruction_loss,
        'sparsity': sparsity_loss
    }
    
    # Add downstream loss terms if enabled
    if config.training.get('downstream_loss', {}).get('enabled', False):
        downstream_cfg = config.training.downstream_loss
        
        # Downstream KL loss for model diffing
        if (downstream_cfg.get('kl_loss', {}).get('weight', 0) > 0 and 
            models is not None and original_activations is not None):
            
            # Split reconstructions for base and chat
            batch_size = sae_output.shape[0] // 2  # Assuming concatenated base+chat
            recon_base = sae_output[:batch_size]
            recon_chat = sae_output[batch_size:]
            
            kl_loss = compute_downstream_kl_loss(
                models['base'], models['chat'],
                original_activations['base'], original_activations['chat'],
                recon_base, recon_chat,
                target_layer,
                downstream_cfg.kl_loss.layers_to_check,
                downstream_cfg.kl_loss.temperature
            )
            
            kl_weight = downstream_cfg.kl_loss.weight
            total_loss += kl_weight * kl_loss
            loss_components['kl_divergence'] = kl_loss
        
        # Downstream activation preservation loss
        if downstream_cfg.get('activation_difference_loss', {}).get('weight', 0) > 0:
            
            # For difference SAEs, we need to be more careful about which model to use
            # This is a simplified version - adjust based on your specific setup
            if models is not None and 'base' in models:
                model = models['base']  # or choose based on config.training.target
                
                act_loss = compute_downstream_activation_loss(
                    model, target, sae_output, target_layer,
                    downstream_cfg.activation_difference_loss.layers_to_check,
                    downstream_cfg.activation_difference_loss.aggregation
                )
                
                act_weight = downstream_cfg.activation_difference_loss.weight
                total_loss += act_weight * act_loss  
                loss_components['downstream_activation'] = act_loss
    
    return total_loss, loss_components


def train_sae_with_downstream_loss(
    sae,
    dataloader, 
    config: DictConfig,
    models: Dict,
    target_layer: int
):
    """
    Training loop with downstream loss terms.
    
    This is a template - integrate with your existing training infrastructure.
    """
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.training.lr)
    
    for batch_idx, batch in enumerate(dataloader):
        # Get activations for base and chat models
        base_activations = batch['base_activations']  
        chat_activations = batch['chat_activations']
        
        # Compute differences based on config.training.target
        if config.training.target == "difference_ftb":
            target_diff = chat_activations - base_activations
        else:
            target_diff = base_activations - chat_activations
            
        # Forward pass through SAE
        sae_output, sae_activations = sae(target_diff)
        
        # Compute enhanced loss
        total_loss, loss_components = enhanced_sae_loss(
            sae_output, target_diff, sae_activations, config,
            models={'base': models['base'], 'chat': models['chat']},
            original_activations={'base': base_activations, 'chat': chat_activations},
            target_layer=target_layer
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Logging
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Total Loss = {total_loss:.4f}")
            for name, loss in loss_components.items():
                print(f"  {name}: {loss:.4f}") 