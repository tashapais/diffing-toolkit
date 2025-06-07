"""
Norm Difference-based model diffing method.

This module computes the L2 norm difference per token between cached activations
from a base model and a finetuned model, processing each layer separately.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from loguru import logger
import json
import numpy as np
from tqdm import tqdm

from .diffing_method import DiffingMethod
from src.utils.configs import get_model_configurations, get_dataset_configurations
from src.utils.activations import load_activation_dataset, get_layer_indices
from src.utils.model import load_tokenizer_from_config  
from src.utils.cache import SampleCache
from src.utils.maximum_tracker import MaximumTracker


class SampleCacheDataset(Dataset):
    """
    PyTorch Dataset wrapper for SampleCache to enable DataLoader usage.
    
    This allows us to leverage DataLoader's multiprocessing capabilities
    for efficient disk I/O when loading activation samples.
    """
    
    def __init__(self, sample_cache: SampleCache, max_samples: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            sample_cache: SampleCache instance to wrap
            max_samples: Optional limit on number of samples to use
        """
        self.sample_cache = sample_cache
        self.length = len(sample_cache)
        
        if max_samples is not None:
            self.length = min(self.length, max_samples)
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the cache.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (tokens, activations)
        """
        # Skip samples with only one token (no meaningful differences)
        tokens, activations = self.sample_cache[idx]
        
        # Return empty tensors for single-token samples - these will be filtered out
        if len(tokens) <= 1:
            return torch.tensor([]), torch.tensor([])
            
        return tokens, activations


def collate_samples(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Custom collate function that filters out empty samples and returns a list.
    
    We don't want to stack/pad here since each sample can have different lengths.
    Instead, we filter out invalid samples and return a list for individual processing.
    
    Args:
        batch: List of (tokens, activations) tuples
        
    Returns:
        Filtered list of valid (tokens, activations) tuples
    """
    # Filter out empty samples (single-token sequences)
    valid_samples = [(tokens, activations) for tokens, activations in batch if len(tokens) > 1]
    return valid_samples


class NormDiffDiffingMethod(DiffingMethod):
    """
    Computes L2 norm difference per token between base and finetuned model activations.
    
    This method:
    1. Loads paired activation caches (no model loading required)
    2. Processes each configured layer separately
    3. Computes per-token L2 norm differences between base and finetuned activations
    4. Tracks max activating examples with full context
    5. Aggregates statistics per dataset and layer
    6. Saves results to disk in nested structure
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logger.bind(method="NormDiff")
        
        # Extract model configurations
        self.base_model_cfg, self.finetuned_model_cfg = get_model_configurations(cfg)
        self.datasets = get_dataset_configurations(cfg)
        
        # Method-specific configuration
        self.method_cfg = cfg.diffing.method
        
        # Get layers to process
        self.layers = get_layer_indices(self.base_model_cfg.model_id, cfg.preprocessing.layers)
        
        # Setup results directory
        self.results_dir = Path(cfg.diffing.results_dir) / "normdiff"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # DataLoader configuration
        self.batch_size = getattr(self.method_cfg.method_params, 'batch_size', 32)
        self.num_workers = getattr(self.method_cfg.method_params, 'num_workers', 8)
        
        self.logger.info(f"Will process layers: {self.layers}")
        self.logger.info(f"DataLoader config: batch_size={self.batch_size}, num_workers={self.num_workers}")
        
    def load_sample_cache(self, dataset_id: str, layer: int) -> Tuple[SampleCache, Any]:
        """
        Load SampleCache for a specific dataset and layer.
        
        Args:
            dataset_id: Dataset identifier
            layer: Layer index
            
        Returns:
            Tuple of (SampleCache instance, tokenizer from cache)
        """
        # Get activation store directory from preprocessing config
        activation_store_dir = Path(self.cfg.preprocessing.activation_store_dir)
        
        # Extract model identifiers for loading cached activations
        base_model_id = self.base_model_cfg.model_id.split('/')[-1]
        finetuned_model_id = self.finetuned_model_cfg.model_id.split('/')[-1]
        
        # Load the paired activation cache for this specific layer
        paired_cache = load_activation_dataset(
            activation_store_dir=activation_store_dir,
            split="train",  # Assume train split
            dataset_name=dataset_id.split("/")[-1],
            base_model=base_model_id,
            finetuned_model=finetuned_model_id,
            layer=layer
        )
        
        # Get tokenizer from model config
        tokenizer = load_tokenizer_from_config(self.base_model_cfg)
        
        # Create SampleCache from the paired activation cache
        # Note: BOS token ID is typically 2 for most models, but this could be made configurable
        sample_cache = SampleCache(paired_cache, bos_token_id=tokenizer.bos_token_id)
        
        self.logger.info(f"Loaded sample cache: {dataset_id}, layer {layer} ({len(sample_cache)} samples)")
        return sample_cache, tokenizer
        
    def compute_norm_difference(
        self, 
        activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-token L2 norm difference between base and finetuned activations.
        
        Args:
            activations: Stacked activations [seq_len, 2, activation_dim]
                        where index 0 is base, index 1 is finetuned
            
        Returns:
            norm_diffs: L2 norm differences per token [seq_len]
        """
        seq_len, num_models, activation_dim = activations.shape
        
        # Shape assertions
        assert num_models == 2, f"Expected 2 models (base, finetuned), got {num_models}"
        assert seq_len > 0, f"Expected non-empty sequence, got length {seq_len}"
        
        # Extract base and finetuned activations
        base_activations = activations[:, 0, :]  # [seq_len, activation_dim]
        finetuned_activations = activations[:, 1, :]  # [seq_len, activation_dim]
        
        # Shape assertions for extracted activations
        assert base_activations.shape == (seq_len, activation_dim), f"Expected: {(seq_len, activation_dim)}, got: {base_activations.shape}"
        assert finetuned_activations.shape == (seq_len, activation_dim), f"Expected: {(seq_len, activation_dim)}, got: {finetuned_activations.shape}"
        
        # Compute activation differences
        activation_diffs = finetuned_activations - base_activations  # [seq_len, activation_dim]
        
        # Compute L2 norm per token
        norm_diffs = torch.norm(activation_diffs, p=2, dim=-1)  # [seq_len]
        
        # Shape assertion for norm differences
        assert norm_diffs.shape == (seq_len,), f"Expected: {(seq_len,)}, got: {norm_diffs.shape}"
        
        return norm_diffs
    
    def compute_statistics(self, all_norm_values: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistical summaries of norm difference values.
        
        Args:
            all_norm_values: Flattened tensor of all norm values
            
        Returns:
            Dictionary with statistical summaries
        """
        stats = {}
        
        # Convert to float32 first if needed, then to numpy for percentile computation
        if all_norm_values.dtype == torch.bfloat16:
            all_norm_values = all_norm_values.float()
        norm_np = all_norm_values.cpu().numpy()
        
        # Basic statistics
        stats['mean'] = float(torch.mean(all_norm_values).item())
        stats['std'] = float(torch.std(all_norm_values).item())
        stats['median'] = float(torch.median(all_norm_values).item())
        stats['min'] = float(torch.min(all_norm_values).item())
        stats['max'] = float(torch.max(all_norm_values).item())
        
        # Percentiles - read from config
        percentile_values = self.method_cfg.analysis.statistics
        if isinstance(percentile_values, list):
            for item in percentile_values:
                if isinstance(item, dict) and 'percentiles' in item:
                    for p in item['percentiles']:
                        stats[f'percentile_{p}'] = float(np.percentile(norm_np, p))
        return stats
    
    def process_layer(self, dataset_id: str, layer: int) -> Dict[str, Any]:
        """
        Process a single layer for a dataset and compute norm differences.
        
        Args:
            dataset_id: Dataset identifier
            layer: Layer index
            
        Returns:
            Dictionary containing statistics and max examples for this dataset/layer
        """
        self.logger.info(f"Processing dataset: {dataset_id}, layer: {layer}")
        
        # Load sample cache for this dataset and layer
        sample_cache, tokenizer = self.load_sample_cache(dataset_id, layer)
        
        # Initialize maximum tracker
        num_examples = self.method_cfg.analysis.max_activating_examples.num_examples
        max_tracker = MaximumTracker(num_examples, tokenizer)
        
        # Create dataset and dataloader
        dataset = SampleCacheDataset(sample_cache, self.method_cfg.method_params.max_samples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Keep original order for reproducibility
            num_workers=self.num_workers,
            collate_fn=collate_samples,
            pin_memory=False,  # Avoid pinning since we have variable-length sequences
            persistent_workers=self.num_workers > 0,  # Keep workers alive for better performance
        )
        
        self.logger.info(f"Processing {len(dataset)} samples from {dataset_id}, layer {layer}")
        self.logger.info(f"Using DataLoader with {self.num_workers} workers, batch_size={self.batch_size}")
        
        all_norm_values = []
        processed_samples = 0
        
        # Process samples in batches using DataLoader
        for batch in tqdm(dataloader, desc=f"Processing batches for layer {layer}"):
            # Process each sample in the batch
            for tokens, activations in batch:
                # Compute norm differences
                norm_diffs = self.compute_norm_difference(activations)
                
                # Find max norm difference in this sample
                max_norm_value = torch.max(norm_diffs).item()
                
                # Add to maximum tracker
                max_tracker.add_example(
                    score=max_norm_value,
                    input_ids=tokens,
                    scores_per_token=norm_diffs,
                    additional_data={'layer': layer}
                )
                
                # Collect all norm values for statistics
                all_norm_values.append(norm_diffs)
                processed_samples += 1
        
        # Concatenate all norm values
        if all_norm_values:
            all_norm_tensor = torch.cat(all_norm_values, dim=0)
            self.logger.info(f"Computed norm differences for {len(all_norm_tensor)} tokens from {dataset_id}, layer {layer}")
            
            # Compute statistics
            statistics = self.compute_statistics(all_norm_tensor)
            total_tokens = len(all_norm_tensor)
        else:
            self.logger.warning(f"No valid samples found for {dataset_id}, layer {layer}")
            statistics = {}
            total_tokens = 0
        
        return {
            'dataset_id': dataset_id,
            'layer': layer,
            'statistics': statistics,
            'max_activating_examples': max_tracker.get_top_examples(),
            'total_tokens_processed': total_tokens,
            'total_samples_processed': processed_samples,
            'metadata': {
                "base_model": self.base_model_cfg.model_id,
                "finetuned_model": self.finetuned_model_cfg.model_id,
            }
        }
    
    def process_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Process all layers for a single dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Dictionary containing results for all layers of this dataset
        """
        results = {
            'dataset_id': dataset_id,
            'layers': {},
            'metadata': {
                "base_model": self.base_model_cfg.model_id,
                "finetuned_model": self.finetuned_model_cfg.model_id,
                "processed_layers": self.layers
            }
        }
        
        # Process each layer
        for layer in self.layers:
            layer_results = self.process_layer(dataset_id, layer)
            results['layers'][f'layer_{layer}'] = layer_results
            
        return results
    
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
        
        # Create dataset subdirectory
        dataset_dir = self.results_dir / safe_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save overall results
        output_file = dataset_dir / "results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save layer-specific results separately  
        for layer_name, layer_results in results['layers'].items():
            layer_file = dataset_dir / f"{layer_name}.json"
            with open(layer_file, 'w') as f:
                json.dump(layer_results, f, indent=2)
                
        self.logger.info(f"Saved results for {dataset_id} to {dataset_dir}")
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
        
        print(f"\n{'='*60}")
        print(f"RESULTS FOR DATASET: {dataset_id}")
        print(f"{'='*60}")
        
        # Print results for each layer
        for layer_name, layer_results in results['layers'].items():
            layer = layer_results['layer']
            stats = layer_results['statistics']
            
            print(f"\n{'-'*40}")
            print(f"LAYER {layer}")
            print(f"{'-'*40}")
            
            print(f"Samples processed: {layer_results['total_samples_processed']:,}")
            print(f"Tokens processed: {layer_results['total_tokens_processed']:,}")
            print(f"\nNorm Difference Statistics:")
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
            examples = layer_results['max_activating_examples']
            print(f"\nMax Activating Examples: {len(examples)}")
            
            if examples:
                print(f"Highest norm difference: {examples[0]['max_score']:.6f}")
                if 'text' in examples[0]:
                    text = examples[0]['text']
                    print("Sample text preview:")
                    print(f"  '{text[:100]}{'...' if len(text) > 100 else ''}'")
    
    def run(self) -> None:
        """
        Main execution method for norm difference diffing.
        
        Processes each dataset and layer combination, saves results to disk.
        """
        self.logger.info("Starting norm difference computation across datasets and layers...")
        
        # Process each dataset
        for dataset_cfg in self.datasets:
            dataset_id = dataset_cfg.id.split("/")[-1]
            
            # Process all layers for this dataset
            results = self.process_dataset(dataset_id)
            
            # Save results to disk
            output_file = self.save_results(dataset_id, results)
            
            # Print results if verbose
            self.print_results(results)
        
        self.logger.info("Norm difference computation completed successfully")
        self.logger.info(f"Results saved to: {self.results_dir}")
    
    def visualize(self) -> None:
        """
        Placeholder for visualization method (required by abstract base class).
        """
        self.logger.info("Visualization not implemented yet")
        pass
