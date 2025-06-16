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
from transformers import AutoTokenizer
import streamlit as st
from nnsight import LanguageModel

from .diffing_method import DiffingMethod
from src.utils.activations import get_layer_indices, load_activation_dataset_from_config
from src.utils.configs import get_dataset_configurations, DatasetConfig  
from src.utils.cache import SampleCache
from src.utils.maximum_tracker import MaximumTracker
from src.utils.dashboards import AbstractOnlineDiffingDashboard


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
        super().__init__(cfg)
        
        # Method-specific configuration
        self.method_cfg = cfg.diffing.method

        # Get dataset configurations
        self.datasets = get_dataset_configurations(cfg, use_chat_dataset=self.method_cfg.datasets.use_chat_dataset, use_pretraining_dataset=self.method_cfg.datasets.use_pretraining_dataset, use_training_dataset=self.method_cfg.datasets.use_training_dataset)
        
        
        # Get layers to process
        self.layers = get_layer_indices(self.base_model_cfg.model_id, cfg.preprocessing.layers)
        
        # Setup results directory
        self.results_dir = Path(cfg.diffing.results_dir) / "normdiff"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # DataLoader configuration
        self.batch_size = getattr(self.method_cfg.method_params, 'batch_size', 32)
        self.num_workers = getattr(self.method_cfg.method_params, 'num_workers', 8)
        


    def load_sample_cache(self, dataset_cfg: DatasetConfig, layer: int) -> Tuple[SampleCache, Any]:
        """
        Load SampleCache for a specific dataset and layer.
        
        Args:
            dataset_cfg: Dataset configuration
            layer: Layer index
            
        Returns:
            Tuple of (SampleCache instance, tokenizer from cache)
        """


        # Load the paired activation cache for this specific layer
        paired_cache = load_activation_dataset_from_config(
            cfg=self.cfg,
            ds_cfg=dataset_cfg,
            base_model_cfg=self.base_model_cfg,
            finetuned_model_cfg=self.finetuned_model_cfg,
            layer=layer,
            split="train"
        )
        
        
        # Create SampleCache from the paired activation cache
        sample_cache = SampleCache(paired_cache, bos_token_id=self.tokenizer.bos_token_id)
        
        self.logger.info(f"Loaded sample cache: {dataset_cfg.id}, layer {layer} ({len(sample_cache)} samples)")
        return sample_cache, self.tokenizer
        
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
        
        return norm_diffs.cpu()
    
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
    
    def process_layer(self, dataset_cfg: DatasetConfig, layer: int) -> Dict[str, Any]:
        """
        Process a single layer for a dataset and compute norm differences.
        
        Args:
            dataset_cfg: Dataset configuration
            layer: Layer index
            
        Returns:
            Dictionary containing statistics and max examples for this dataset/layer
        """
        self.logger.info(f"Processing dataset: {dataset_cfg.id}, layer: {layer}")
        
        # Load sample cache for this dataset and layer
        sample_cache, tokenizer = self.load_sample_cache(dataset_cfg, layer)
        
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
        
        self.logger.info(f"Processing {len(dataset)} samples from {dataset_cfg.id}, layer {layer}")
        self.logger.info(f"Using DataLoader with {self.num_workers} workers, batch_size={self.batch_size}")
        
        all_norm_values = []
        processed_samples = 0
        
        # Process samples in batches using DataLoader
        for batch in tqdm(dataloader, desc=f"Processing batches for layer {layer}"):
            # Process each sample in the batch
            for tokens, activations in batch:
                # Move activations to GPU for computation
                activations = activations.to(self.device)
                
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
            self.logger.info(f"Computed norm differences for {len(all_norm_tensor)} tokens from {dataset_cfg.id}, layer {layer}")
            
            # Compute statistics
            statistics = self.compute_statistics(all_norm_tensor)
            total_tokens = len(all_norm_tensor)
        else:
            self.logger.warning(f"No valid samples found for {dataset_cfg.id}, layer {layer}")
            statistics = {}
            total_tokens = 0
        
        return {
            'dataset_id': dataset_cfg.id,
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
    
    def process_dataset(self, dataset_cfg: DatasetConfig) -> Dict[str, Any]:
        """
        Process all layers for a single dataset.
        
        Args:
            dataset_cfg: Dataset configuration
            
        Returns:
            Dictionary containing results for all layers of this dataset
        """
        results = {
            'dataset_id': dataset_cfg.id,
            'layers': {},
            'metadata': {
                "base_model": self.base_model_cfg.model_id,
                "finetuned_model": self.finetuned_model_cfg.model_id,
                "processed_layers": self.layers
            }
        }
        
        # Process each layer
        for layer in self.layers:
            layer_results = self.process_layer(dataset_cfg, layer)
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

    def run(self) -> None:
        """
        Main execution method for norm difference diffing.
        
        Processes each dataset and layer combination, saves results to disk.
        """
        self.logger.info("Starting norm difference computation across datasets and layers...")
        
        # Process each dataset
        for dataset_cfg in self.datasets:
            
            # Process all layers for this dataset
            results = self.process_dataset(dataset_cfg)
            
            # Save results to disk
            output_file = self.save_results(dataset_cfg.id, results)

        
        self.logger.info("Norm difference computation completed successfully")
        self.logger.info(f"Results saved to: {self.results_dir}")

    def visualize(self) -> None:
        """
        Create Streamlit visualization for norm difference results with tabs.
        
        Returns:
            Streamlit component displaying dataset statistics and interactive analysis
        """
        from src.utils.visualization import statistic_interactive_tab


        statistic_interactive_tab(self._render_dataset_statistics, lambda: NormDiffOnlineDashboard(self).display(), "Norm Difference Analysis")
    
    def _render_dataset_statistics(self):
        """Render the dataset statistics tab (original visualize functionality)."""
        from src.utils.visualization import (
            convert_max_examples_to_dashboard_format, 
            create_examples_html, 
            render_streamlit_html,
            load_results_file,
            filter_examples_by_search,
            
        )

        # Dataset selector
        dataset_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        if not dataset_dirs:
            st.error(f"No norm difference results found in {self.results_dir}")
            return
        
        dataset_names = [d.name for d in dataset_dirs]
        selected_dataset = st.selectbox("Select Dataset", dataset_names)
        
        if not selected_dataset:
            return
        
        # Load results (cached)
        dataset_dir = self.results_dir / selected_dataset
        results_file = dataset_dir / "results.json"
        
        if not results_file.exists():
            st.error(f"Results file not found: {results_file}")
            return
        
        results = load_results_file(str(results_file))
        
        # Layer selector (local to this method)
        available_layers = list(results['layers'].keys())
        selected_layer = st.selectbox("Select Layer", available_layers)
        
        if not selected_layer:
            return
        
        layer_results = results['layers'][selected_layer]
        
        # Display statistics
        stats = layer_results['statistics']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Norm Diff", f"{stats['mean']:.6f}")
            st.metric("Max Norm Diff", f"{stats['max']:.6f}")
        with col2:
            st.metric("Std Norm Diff", f"{stats['std']:.6f}")
            st.metric("Median Norm Diff", f"{stats['median']:.6f}")
        with col3:
            st.metric("Total Tokens", f"{layer_results['total_tokens_processed']:,}")
            st.metric("Total Samples", f"{layer_results['total_samples_processed']:,}")
        
        # Convert examples to dashboard format (cached)
        dashboard_examples = convert_max_examples_to_dashboard_format(
            layer_results['max_activating_examples'], 
            self.base_model_cfg
        )
        
        # Search functionality
        search_term = st.text_input(
            "🔍 Search in examples", 
            placeholder="Enter text to search for in the examples..."
        )
        
        # Filter examples and show count
        if search_term.strip():
            filtered_examples = filter_examples_by_search(dashboard_examples, search_term)
            st.info(f"Found {len(filtered_examples)} examples containing '{search_term}' out of {len(dashboard_examples)} total examples")
            examples_to_show = filtered_examples
        else:
            examples_to_show = dashboard_examples
            st.info(f"Showing all {len(dashboard_examples)} examples")
        
        if not examples_to_show:
            st.warning("No examples found matching your search.")
            return
        
        # Create HTML visualization
        html_content = create_examples_html(
            examples_to_show,
            self.tokenizer,
            title=f"Norm Difference - {selected_dataset} - {selected_layer}" + (f" - Search: '{search_term}'" if search_term.strip() else ""),
            max_examples=30,
            window_size=50,
            use_absolute_max=False,
        )
        
        # Render in Streamlit
        render_streamlit_html(html_content)
    
    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available norm difference results.
        
        Returns:
            Dict mapping {model: {organism: path_to_results}}
        """
        results = {}
        results_base = results_dir
        
        if not results_base.exists():
            return results
        
        # Scan for normdiff results in the expected structure
        for model_dir in results_base.iterdir():
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name
            
            for organism_dir in model_dir.iterdir():
                if not organism_dir.is_dir():
                    continue
                    
                organism_name = organism_dir.name
                normdiff_dir = organism_dir / "normdiff"
                
                # Check if normdiff results exist (any dataset dirs with results.json)
                if normdiff_dir.exists():
                    has_results = False
                    for dataset_dir in normdiff_dir.iterdir():
                        if dataset_dir.is_dir():
                            results_file = dataset_dir / "results.json"
                            if results_file.exists():
                                has_results = True
                                break
                    
                    if has_results:
                        if model_name not in results:
                            results[model_name] = {}
                        results[model_name][organism_name] = str(normdiff_dir)
        
        return results

    def compute_normdiff_for_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer: int
    ) -> Dict[str, Any]:
        """
        Compute norm difference statistics for given tokens (used by both method and dashboard).
        
        Args:
            input_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len]
            layer: Layer index to analyze
            
        Returns:
            Dictionary with tokens, norm_diff_values, and statistics
        """
   
        # Get base model as LanguageModel
        base_nn_model = LanguageModel(self.base_model, tokenizer=self.tokenizer)
        
        # Get finetuned model as LanguageModel  
        finetuned_nn_model = LanguageModel(self.finetuned_model, tokenizer=self.tokenizer)
        
        # Prepare input batch
        batch = {
            'input_ids': input_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device)
        }
        
        # Get tokens for display (all tokens for norm diff since we don't predict next token)
        token_ids = input_ids[0].cpu().numpy()  # Take first sequence
        tokens = [self.tokenizer.decode([token_id]) for token_id in token_ids]
        
        # Extract activations from both models
        with torch.no_grad():
            # Get base model activations
            with base_nn_model.trace(batch):
                base_activations = base_nn_model.model.layers[layer].output[0].save()
            
            # Get finetuned model activations
            with finetuned_nn_model.trace(batch):
                finetuned_activations = finetuned_nn_model.model.layers[layer].output[0].save()
        
        # Extract the values and move to CPU
        base_acts = base_activations.cpu()  # [batch_size, seq_len, hidden_dim]
        finetuned_acts = finetuned_activations.cpu()  # [batch_size, seq_len, hidden_dim]
        
        # Take first sequence and stack for the compute_norm_difference method
        # Shape: [seq_len, 2, hidden_dim] where index 0 is base, index 1 is finetuned
        seq_len = base_acts.shape[1]
        hidden_dim = base_acts.shape[2]
        
        stacked_activations = torch.stack([
            base_acts[0],  # [seq_len, hidden_dim]
            finetuned_acts[0]  # [seq_len, hidden_dim]
        ], dim=1)  # [seq_len, 2, hidden_dim]
        
        # Shape assertions
        assert stacked_activations.shape == (seq_len, 2, hidden_dim), f"Expected: {(seq_len, 2, hidden_dim)}, got: {stacked_activations.shape}"
        
        # Compute norm differences using existing method
        norm_diff_values = self.compute_norm_difference(stacked_activations)
        
        # Convert to numpy for statistics computation
        norm_diff_np = norm_diff_values.float().cpu().numpy()
        
        # Compute statistics
        statistics = {
            'mean': float(np.mean(norm_diff_np)),
            'std': float(np.std(norm_diff_np)),
            'min': float(np.min(norm_diff_np)),
            'max': float(np.max(norm_diff_np)),
            'median': float(np.median(norm_diff_np)),
        }
        
        return {
            'tokens': tokens,
            'norm_diff_values': norm_diff_np,
            'statistics': statistics,
            'total_tokens': len(tokens),
            'layer': layer
        }


class NormDiffOnlineDashboard(AbstractOnlineDiffingDashboard):
    """
    Online dashboard for interactive norm difference analysis.
    """
    
    def _render_streamlit_method_controls(self) -> Dict[str, Any]:
        """Render NormDiff-specific controls in Streamlit."""
        import streamlit as st
        
        layer = st.selectbox(
            "Select Layer:",
            options=self.method.layers,
            help="Choose which layer to analyze"
        )
        return {"layer": layer}
    
    def compute_statistics_for_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Compute norm difference statistics using the parent method's computation function."""
        layer = kwargs.get("layer", self.method.layers[0])
        results = self.method.compute_normdiff_for_tokens(input_ids, attention_mask, layer)
        
        # Adapt the results format for the abstract dashboard
        return {
            'tokens': results['tokens'],
            'values': results['norm_diff_values'],  # Use 'values' as the standard key
            'statistics': results['statistics'],
            'total_tokens': results['total_tokens']
        }
    
    def get_method_specific_params(self) -> Dict[str, Any]:
        """Get NormDiff-specific parameters."""
        if hasattr(self, 'layer_selector'):
            return {"layer": self.layer_selector.value}
        return {"layer": self.method.layers[0]}
    
    def _get_color_rgb(self) -> tuple:
        """Get red color for norm difference highlighting."""
        return (255, 0, 0)
    
    def _get_title(self) -> str:
        """Get title for norm difference analysis."""
        return "Norm Difference Analysis"
