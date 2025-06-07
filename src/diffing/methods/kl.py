"""
KL Divergence-based model diffing method.

This module computes the KL divergence per token between a base model and a 
finetuned model using cached token sequences.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import torch
from torch.nn import functional as F
from omegaconf import DictConfig
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
import os
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence

from .diffing_method import DiffingMethod
from src.utils.model import load_model_from_config
from src.utils.configs import get_model_configurations, get_dataset_configurations
from src.utils.activations import load_activation_dataset, get_layer_indices
from src.utils.cache import SampleCache
from src.utils.maximum_tracker import MaximumTracker


class KLDivergenceDiffingMethod(DiffingMethod):
    """ 
    Computes KL divergence per token between base and finetuned models.
    
    This method:
    1. Loads base and finetuned models
    2. Uses cached token sequences from SampleCache
    3. Computes per-token KL divergence between model outputs
    4. Tracks max activating examples with full context
    5. Aggregates statistics per dataset
    6. Saves results to disk
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logger.bind(method="KLDivergence")
        
        # Extract model configurations
        self.base_model_cfg, self.finetuned_model_cfg = get_model_configurations(cfg)
        self.datasets = get_dataset_configurations(cfg)
        
        # Method-specific configuration
        self.method_cfg = cfg.diffing.method
        
        # Initialize models and tokenizer placeholders
        self.base_model: Optional[AutoModelForCausalLM] = None
        self.finetuned_model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup results directory
        self.results_dir = Path(cfg.diffing.results_dir) / "kl"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_models(self) -> None:
        """Load base and finetuned models."""
        self.logger.info("Loading models...")
        
        # Load base model
        self.base_model, self.tokenizer = load_model_from_config(self.base_model_cfg)
        self.base_model.eval()
        
        # Load finetuned model (reuse tokenizer from base model)
        self.finetuned_model, _ = load_model_from_config(self.finetuned_model_cfg)
        self.finetuned_model.eval()
        
        self.logger.info(f"Loaded base model: {self.base_model_cfg.model_id}")
        self.logger.info(f"Loaded finetuned model: {self.finetuned_model_cfg.model_id}")
        
    def load_sample_cache(self, dataset_id: str) -> SampleCache:
        """
        Load SampleCache for a specific dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            SampleCache instance
        """
        # Get activation store directory from preprocessing config
        activation_store_dir = Path(self.cfg.preprocessing.activation_store_dir)

        
        # Extract model identifiers for loading cached activations
        base_model_id = self.base_model_cfg.model_id.split('/')[-1]  # Extract model name from path
        finetuned_model_id = self.finetuned_model_cfg.model_id.split('/')[-1]
        
        # As we're not using the activations, any of the cached layers is fine
        layer = get_layer_indices(self.base_model_cfg.model_id, self.cfg.preprocessing.layers)[0]
        
        # Load the paired activation cache
        paired_cache = load_activation_dataset(
            activation_store_dir=activation_store_dir,
            split="train",  # Assume train split
            dataset_name=dataset_id.split("/")[-1],
            base_model=base_model_id,
            finetuned_model=finetuned_model_id,
            layer=layer
        )
        
        # Create SampleCache from the paired activation cache
        sample_cache = SampleCache(paired_cache, bos_token_id=self.tokenizer.bos_token_id)
        
        self.logger.info(f"Loaded sample cache: {dataset_id} ({len(sample_cache)} samples)")
        return sample_cache
        
    def compute_kl_divergence(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-token KL divergence between base and finetuned model outputs.
        
        Args:
            input_ids: Token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            per_token_kl: KL divergence per token [batch_size, seq_len-1]
        """
        batch_size, seq_len = input_ids.shape
        
        with torch.no_grad():
            # Get logits from both models
            base_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            finetuned_outputs = self.finetuned_model(input_ids=input_ids, attention_mask=attention_mask)
            
            base_logits = base_outputs.logits  # [batch_size, seq_len, vocab_size]
            finetuned_logits = finetuned_outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Shape assertions for logits
            vocab_size = base_logits.shape[-1]
            assert base_logits.shape == (batch_size, seq_len, vocab_size), f"Expected: {(batch_size, seq_len, vocab_size)}, got: {base_logits.shape}"
            assert finetuned_logits.shape == (batch_size, seq_len, vocab_size), f"Expected: {(batch_size, seq_len, vocab_size)}, got: {finetuned_logits.shape}"
            assert base_logits.shape == finetuned_logits.shape, f"Expected: {base_logits.shape}, got: {finetuned_logits.shape}"
            
            # Apply temperature
            temperature = self.method_cfg.method_params.temperature
            if temperature != 1.0:
                base_logits = base_logits / temperature
                finetuned_logits = finetuned_logits / temperature
                
            # Convert to log probabilities
            base_log_probs = F.log_softmax(base_logits, dim=-1)  # [batch_size, seq_len, vocab_size]
            finetuned_log_probs = F.log_softmax(finetuned_logits, dim=-1)  # [batch_size, seq_len, vocab_size]

            # Shape assertions for log probabilities
            assert base_log_probs.shape == (batch_size, seq_len, vocab_size), f"Expected: {(batch_size, seq_len, vocab_size)}, got: {base_log_probs.shape}"
            assert finetuned_log_probs.shape == (batch_size, seq_len, vocab_size), f"Expected: {(batch_size, seq_len, vocab_size)}, got: {finetuned_log_probs.shape}"
            
            # Convert to probabilities for KL computation
            base_probs = F.softmax(base_logits, dim=-1)
            finetuned_probs = F.softmax(finetuned_logits, dim=-1)
            
            # Shape assertions for probabilities
            assert base_probs.shape == (batch_size, seq_len, vocab_size), f"Expected: {(batch_size, seq_len, vocab_size)}, got: {base_probs.shape}"
            assert finetuned_probs.shape == (batch_size, seq_len, vocab_size), f"Expected: {(batch_size, seq_len, vocab_size)}, got: {finetuned_probs.shape}"
            
            # Compute KL divergence: KL(finetuned || base) = sum(finetuned * log(finetuned / base))
            # = sum(finetuned * (log_finetuned - log_base))
            kl_div = torch.sum(finetuned_probs * (finetuned_log_probs - base_log_probs), dim=-1)
            
            # Shape assertions for KL divergence
            assert kl_div.shape == (batch_size, seq_len), f"Expected: {(batch_size, seq_len)}, got: {kl_div.shape}"
            
            return kl_div
    
    def update_max_examples_tracker(
        self,
        per_token_kl: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_tracker: MaximumTracker
    ) -> None:
        """
        Update tracker with examples that have high KL divergence tokens.
        
        Args:
            per_token_kl: KL divergence per token [batch_size, seq_len-1]
            input_ids: Token ids [batch_size, seq_len]  
            attention_mask: Attention mask [batch_size, seq_len]
            max_tracker: MaximumTracker instance to update
        """
        # Find max KL for each example in the batch
        max_kl_per_example = []
        for batch_idx in range(per_token_kl.shape[0]):
            valid_mask = attention_mask[batch_idx].bool()
            valid_kl = per_token_kl[batch_idx][valid_mask]
            
            if len(valid_kl) == 0:
                max_kl_per_example.append(0.0)
            else:
                max_kl_per_example.append(torch.max(valid_kl).item())
        
        # Convert to tensor for batch processing
        max_kl_tensor = torch.tensor(max_kl_per_example)
        
        # Use batch processing in MaximumTracker
        max_tracker.add_batch_examples(
            scores_per_example=max_kl_tensor,
            input_ids_batch=input_ids,
            attention_mask_batch=attention_mask,
            scores_per_token_batch=per_token_kl
        )
    
    def compute_statistics(self, all_kl_values: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistical summaries of KL divergence values.
        
        Args:
            all_kl_values: Flattened tensor of all KL values
            
        Returns:
            Dictionary with statistical summaries
        """
        stats = {}
        
        # Convert to float32 first if needed, then to numpy for percentile computation
        if all_kl_values.dtype == torch.bfloat16:
            all_kl_values = all_kl_values.float()
        kl_np = all_kl_values.cpu().numpy()
        
        # Basic statistics
        stats['mean'] = float(torch.mean(all_kl_values).item())
        stats['std'] = float(torch.std(all_kl_values).item())
        stats['median'] = float(torch.median(all_kl_values).item())
        stats['min'] = float(torch.min(all_kl_values).item())
        stats['max'] = float(torch.max(all_kl_values).item())
        
        # Percentiles - read from config
        percentile_values = self.method_cfg.analysis.statistics
        if isinstance(percentile_values, list):
            for item in percentile_values:
                if isinstance(item, dict) and 'percentiles' in item:
                    for p in item['percentiles']:
                        stats[f'percentile_{p}'] = float(np.percentile(kl_np, p))
        return stats
        
    def prepare_batch(self, sequences: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of sequences for processing.
        
        Args:
            sequences: List of token tensors
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        
        # Truncate sequences to max length if specified
        max_len = self.method_cfg.method_params.max_tokens_per_sample
        truncated_sequences = [seq[:max_len] for seq in sequences]
        
        # Pad sequences using torch's built-in function
        input_ids = pad_sequence(
            truncated_sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
        for i, seq in enumerate(truncated_sequences):
            attention_mask[i, :len(seq)] = 1
            
        return input_ids, attention_mask
    
    @torch.no_grad()
    def process_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Process a single dataset and compute KL divergences.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Dictionary containing aggregated statistics and max examples for this dataset
        """
        self.logger.info(f"Processing dataset: {dataset_id}")
        
        # Load sample cache for this dataset
        sample_cache = self.load_sample_cache(dataset_id)
        
        # Get all sequences
        sequences = sample_cache.sequences
        
        # Apply max_samples limit if specified
        if self.method_cfg.method_params.max_samples is not None:
            sequences = sequences[:self.method_cfg.method_params.max_samples]

        # Filter sequences with only one token
        sequences = [seq for seq in sequences if len(seq) > 1]
            
        self.logger.info(f"Processing {len(sequences)} sequences from {dataset_id}")
        
        # Initialize maximum tracker
        num_examples = self.method_cfg.analysis.max_activating_examples.num_examples
        max_tracker = MaximumTracker(num_examples, self.tokenizer)
        
        all_kl_values = []
        batch_size = self.method_cfg.method_params.batch_size
        
        # Process sequences in batches
        for i in trange(0, len(sequences), batch_size, desc=f"Processing batches from {dataset_id}"):
            batch_sequences = sequences[i:i+batch_size]

            # Prepare batch tensors
            input_ids, attention_mask = self.prepare_batch(batch_sequences)
            
            # Move to device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Compute KL divergence
            per_token_kl = self.compute_kl_divergence(input_ids, attention_mask)
            
            # Update max examples tracker
            self.update_max_examples_tracker(per_token_kl, input_ids, attention_mask, max_tracker)
            
            # Collect valid KL values (excluding padding)
            for j in range(per_token_kl.shape[0]):
                valid_mask = attention_mask[j].bool()  
                valid_kl = per_token_kl[j][valid_mask]
                all_kl_values.append(valid_kl)

        
        # Concatenate all KL values
        all_kl_tensor = torch.cat(all_kl_values, dim=0)
        self.logger.info(f"Computed KL divergence for {len(all_kl_tensor)} tokens from {dataset_id}")
        
        # Compute statistics
        statistics = self.compute_statistics(all_kl_tensor)
        
        return {
            'dataset_id': dataset_id,
            'statistics': statistics,
            'max_activating_examples': max_tracker.get_top_examples(),
            'total_tokens_processed': len(all_kl_tensor),
            'total_sequences_processed': len(sequences),
            'metadata': {
                "base_model": self.base_model_cfg.model_id,
                "finetuned_model": self.finetuned_model_cfg.model_id,
            }
        }
    
    def save_results(self, dataset_id: str, results: Dict[str, Any]) -> Path:
        """
        Save results for a dataset to disk.
        
        Args:
            dataset_id: Dataset identifier
            results: Results dictionary
            
        Returns:
            Path to saved file
        """
        # Convert dataset ID to safe filename
        safe_name = dataset_id.split("/")[-1]
        output_file = self.results_dir / f"{safe_name}.json"
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Saved results for {dataset_id} to {output_file}")
        return output_file
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """
        Print results summary if verbose mode is enabled.
        
        Args:
            results: Results dictionary
        """
        if not self.verbose:
            return
            
        dataset_id = results['dataset_id']
        stats = results['statistics']
        
        print(f"\n{'='*50}")
        print(f"RESULTS FOR DATASET: {dataset_id}")
        print(f"{'='*50}")
        
        print(f"Sequences processed: {results['total_sequences_processed']:,}")
        print(f"Tokens processed: {results['total_tokens_processed']:,}")
        print(f"\nKL Divergence Statistics:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")
        print(f"  Median: {stats['median']:.6f}")
        
        # Print percentiles
        for key, value in stats.items():
            if key.startswith('percentile_'):
                print(f"  {key.replace('_', ' ').title()}: {value:.6f}")
        
        # Print max examples info
        examples = results['max_activating_examples']
        print(f"\nMax Activating Examples: {len(examples)}")
        
        if examples:
            print(f"Highest KL divergence: {examples[0]['max_score']:.6f}")
            if 'text' in examples[0]:
                print("Sample text preview:")
                text = examples[0]['text']
                print(f"  '{text[:100]}{'...' if len(text) > 100 else ''}'")
    
    def run(self) -> None:
        """
        Main execution method for KL divergence diffing.
        
        Processes each dataset separately and saves results to disk.
        """
        # Setup models
        self.setup_models()
        
        self.logger.info("Starting KL divergence computation across datasets...")
        
        # Process each dataset separately
        for dataset_cfg in self.datasets:
            # Process dataset
            results = self.process_dataset(dataset_cfg.id.split("/")[-1])
            
            # Save results to disk
            output_file = self.save_results(dataset_cfg.id.split("/")[-1], results)
            
            # Print results if verbose
            self.print_results(results)
        
        self.logger.info("KL divergence computation completed successfully")
        self.logger.info(f"Results saved to: {self.results_dir}")
    
    def visualize(self) -> None:
        """
        Placeholder for visualization method (required by abstract base class).
        """
        self.logger.info("Visualization not implemented yet")
        pass
