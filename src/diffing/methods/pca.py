"""
PCA difference-based model diffing method.

This module trains PCA on activation differences between base and finetuned models,
then runs a comprehensive analysis pipeline including evaluation notebooks, component analysis,
and variance explained experiments.

Key assumptions:
- Preprocessing pipeline has generated paired activation caches
- TorchDR library is available for IncrementalPCA
- Same activation difference computation as SAE method
- Sufficient GPU memory and disk space for PCA fitting
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
from loguru import logger
import json
import pickle
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
import streamlit as st
from tqdm import tqdm, trange
from torchdr import IncrementalPCA
import shutil

from .diffing_method import DiffingMethod
from src.utils.activations import get_layer_indices, load_activation_dataset_from_config, load_activation_datasets_from_config
from src.utils.dictionary.training import (
    setup_training_datasets,
    create_training_dataloader,
    setup_sae_cache,
    recompute_normalizer,
)
from src.utils.configs import get_model_configurations, get_dataset_configurations
from src.utils.dashboards import AbstractOnlineDiffingDashboard, MaxActivationDashboardComponent
from src.utils.max_act_store import MaxActStore, ReadOnlyMaxActStore
from src.utils.cache import SampleCache

class PCAMethod(DiffingMethod):
    """
    Trains PCA on activation differences and runs comprehensive analysis.

    This method:
    1. Loads paired activation caches from preprocessing pipeline  
    2. Computes activation differences (finetuned - base or base - finetuned)
    3. Trains IncrementalPCA on normalized differences for specified layers
    4. Saves trained PCA models with configuration and metrics
    5. Runs complete analysis pipeline 
    6. Returns comprehensive results including training metrics and analysis outcomes
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Get layers to process
        layers = self.method_cfg.layers
        if layers is None:
            layers = cfg.preprocessing.layers
        self.layers = get_layer_indices(self.base_model_cfg.model_id, layers)

        # Setup results directory
        self.results_dir = Path(cfg.diffing.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._component_dfs = {}

        self.target_display_names = {
            "difference_ftb": "FT - Base",
            "difference_bft": "Base - FT",
            "base": "Base",
            "ft": "FT"
        }

        self._patch_config()

    def __hash__(self):
        return hash(self.cfg)
    
    def __eq__(self, other):
        return self.cfg == other.cfg

    def _patch_config(self):
        """Patch the config to add not needed parameters that downstream code expects."""
        with open_dict(self.method_cfg):
            self.method_cfg.training.epochs = 1
            self.method_cfg.training.num_validation_samples = 5_000_000 # Not used


    def run(self) -> Dict[str, Any]:
        """
        Main PCA training orchestration with analysis pipeline.

        Trains PCA on differences for each specified layer, then runs the complete
        analysis pipeline for each trained model.

        Returns:
            Dictionary containing training results, model paths, and analysis outcomes
        """
        logger.info(f"Starting PCA difference training for layers: {self.layers}")
        logger.info(f"Training target: {self.method_cfg.training.target}")

        for layer_idx in self.layers:
            logger.info(f"Processing layer {layer_idx}")

            target = self.method_cfg.training.target
            model_results_dir = (
                self.results_dir
                / "pca"
                / f"layer_{layer_idx}"
                / target
            )
            model_results_dir.mkdir(parents=True, exist_ok=True)

            if (
                not (model_results_dir / "pca_model.pkl").exists()
                or self.method_cfg.training.overwrite
            ):
                # Train PCA on differences for this layer
                logger.info(f"Training PCA for layer {layer_idx} with target {target}")
                training_metrics, model_path = self._train_pca_for_layer(
                    layer_idx, target, model_results_dir,
                )
                
                # Save training metrics
                with open(model_results_dir / "training_metrics.json", "w") as f:
                    json.dump(training_metrics, f)

                # Save training config
                OmegaConf.save(self.cfg, model_results_dir / "training_config.yaml")
            else:
                logger.info(f"Found trained PCA model at {model_results_dir / 'pca_model.pkl'}")
                training_metrics = json.load(
                    open(model_results_dir / "training_metrics.json")
                )

            if self.method_cfg.analysis.enabled:
                logger.info(f"Running analysis for layer {layer_idx}")
                self._run_analysis_for_layer(layer_idx, target, model_results_dir)

            logger.info(f"Successfully completed layer {layer_idx}")

        return {"status": "completed", "layers_processed": self.layers}

    def _setup_dataset(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], torch.Tensor, torch.Tensor]:
        """Setup training datasets with difference cache processing."""

        target = self.method_cfg.training.target
        assert target in [
            "difference_bft",
            "difference_ftb",
            "base",
            "ft",
        ], f"Invalid target for PCA: {target}"

        # Setup training datasets with difference cache processing  
        result = setup_training_datasets(  # type: ignore
            self.cfg,
            layer_idx,
            dataset_processing_function=lambda x: setup_sae_cache(
                target=target, paired_cache=x
            ),
            normalizer_function=(lambda x: 
                recompute_normalizer(
                    x,
                    subsample_size=self.method_cfg.datasets.normalization.subsample_size,
                    batch_size=self.method_cfg.datasets.normalization.batch_size,
                    cache_dir=self.method_cfg.datasets.normalization.cache_dir,
                    layer=layer_idx,
                    device=self.device,
                ))
                if self.method_cfg.datasets.normalization.enabled
                else None
        )
        
        # Unpack the result tuple
        return result
    
    def _train_pca_for_layer(
        self, layer_idx: int, target: str, model_results_dir: Path
    ) -> Tuple[Dict[str, Any], Path]:
        """
        Train IncrementalPCA on differences for a specific layer.
        
        Args:
            layer_idx: Layer index to train on
            target: Target direction (difference_ftb or difference_bft or base or ft)
            model_results_dir: Directory to save results
            
        Returns:
            Tuple of (training metrics, model path)
        """
        train_dataset, val_dataset, epoch_idx_per_step, normalize_mean, normalize_std = self._setup_dataset(layer_idx)

        # Get activation dimdatasetirst sample
        sample_activation = train_dataset[0]
        activation_dim = sample_activation.shape[-1]
        assert activation_dim > 0, f"Invalid activation dimension: {activation_dim}"
        logger.info(f"Activation dimension: {activation_dim}")

        # Check whether the PCA model already exists
        if not self.method_cfg.training.overwrite and (model_results_dir / "pca_model.pkl").exists():
            logger.info(f"Found existing PCA model at {model_results_dir / 'pca_model.pkl'}")
            return
        else:
            logger.info(f"No existing PCA model found at {model_results_dir / 'pca_model.pkl'}, creating new one")

        # Create data loaderdataset
        train_dataloader = create_training_dataloader(train_dataset, self.cfg, shuffle=False, drop_last=True)

        # Initialize IncrementalPCA
        n_components = activation_dim
        
        pca = IncrementalPCA(
            n_components=n_components,
            batch_size=self.method_cfg.training.batch_size,
            device=self.device,
            verbose=True,
        )

        logger.info("Fitting IncrementalPCA on activation differences...")
        total_samples = 0
            
        # Fit PCA incrementally using batches from DataLoader
        for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Fitting IncrementalPCA"):
            # batch shape: [batch_size, activation_dim]
            assert batch.ndim == 2, f"Expected 2D batch, got shape {batch.shape}"
            assert batch.shape[1] == activation_dim, f"Expected activation_dim {activation_dim}, got {batch.shape[1]}"
            
            # Move to device and fit
            batch = batch.to(self.device)
            pca.partial_fit(batch)
            
            total_samples += batch.shape[0]

        logger.info(f"PCA fitting completed. Total samples: {total_samples}")

        # Save PCA model
        model_path = model_results_dir / "pca_model.pkl"
        with open(model_path, 'wb') as f:
            # Save PCA state that can be reconstructed
            pca_state = {
                'components_': pca.components_.cpu() if pca.components_ is not None else None,
                'explained_variance_': pca.explained_variance_.cpu(),
                'explained_variance_ratio_': pca.explained_variance_ratio_.cpu(),
                'mean_': pca.mean_.cpu(),
                'var_': pca.var_.cpu(),
                'noise_variance_': pca.noise_variance_.cpu(),
                'n_components': pca.n_components,
                'n_samples_seen_': pca.n_samples_seen_,
                'activation_dim': activation_dim,
                'target': target,
                'config': OmegaConf.to_yaml(self.method_cfg),
            }
            pickle.dump(pca_state, f)

        logger.info(f"PCA model saved to {model_path}")

        # Compute explained variance ratio
        total_variance_explained = float(pca.explained_variance_ratio_.sum())
        logger.info(f"Total variance explained: {total_variance_explained:.4f}")

        # Collect training metrics
        training_metrics = {
            "layer": layer_idx,
            "activation_dim": activation_dim,
            "n_components": n_components,
            "total_samples": total_samples,
            "target": target,
            "total_variance_explained": total_variance_explained,
        }

        return training_metrics, model_path
    
    @torch.no_grad()
    def _collect_maximum_activating_examples(self, pca: IncrementalPCA, layer_idx: int, results_dir: Path, target: str) -> None:
        """Collect maximum activating examples for a trained PCA model."""
            
        dataset_cfgs = get_dataset_configurations(
            self.cfg,
            use_chat_dataset=self.method_cfg.datasets.use_chat_dataset,
            use_pretraining_dataset=self.method_cfg.datasets.use_pretraining_dataset,
            use_training_dataset=self.method_cfg.datasets.use_training_dataset,
        )

        caches = load_activation_datasets_from_config(
            cfg=self.cfg,
            ds_cfgs=dataset_cfgs,
            base_model_cfg=self.base_model_cfg,
            finetuned_model_cfg=self.finetuned_model_cfg,
            layers=[layer_idx],
            split=self.method_cfg.analysis.max_activating_examples.split,
        )  # Dict {dataset_name: {layer: PairedActivationCache, ...}}

        # Now collect maximum activating examples from each dataset separately
        logger.info(f"Computing maximum activating examples for layer {layer_idx} and target {target}...")
        
        # Check whether the max store already exists
        pos_examples_path = results_dir / "positive_examples.db"
        neg_examples_path = results_dir / "negative_examples.db"
        if not self.method_cfg.analysis.max_activating_examples.overwrite and pos_examples_path.exists() and neg_examples_path.exists():
            logger.info(f"Found existing max store at {results_dir}")
            return
        else:
            pos_examples_path.unlink(missing_ok=True)
            neg_examples_path.unlink(missing_ok=True)


       
        # Initialize maximum examples store
        num_examples = self.method_cfg.analysis.max_activating_examples.n_max_activations
        max_store_positive = MaxActStore(
            pos_examples_path,
            max_examples=num_examples,
            storage_format='dense',  # Store full per-token projections,
            per_dataset=True
        )
        max_store_negative = MaxActStore(
            neg_examples_path,
            max_examples=num_examples,
            storage_format='dense',  # Store full per-token projections
            per_dataset=True
        )

        latent_idx = torch.arange(pca.n_components)
        # Use async writers with context managers for efficient background writing
        with max_store_positive.create_async_writer(
            buffer_size=pca.n_components * 10,  # Buffer ~10 batches
            flush_interval=30.0,
            auto_maintain_top_k=True,
            use_memory_db=True,
        ) as positive_writer, \
        max_store_negative.create_async_writer(
            buffer_size=pca.n_components * 10,
            flush_interval=30.0,
            auto_maintain_top_k=True,
            use_memory_db=True,
        ) as negative_writer:
            
            # Process each dataset to find maximum activating examples
            for dataset_idx, (dataset_name, dataset_caches) in enumerate(caches.items()):
                logger.info(f"Processing maximum examples for dataset: {dataset_name}")
                
                paired_cache = dataset_caches[layer_idx]

                # Create processed cache for the target
                processed_cache = setup_sae_cache(target=self.method_cfg.training.target, paired_cache=paired_cache)
                
                # Create SampleCache to get sequences with tokens
                sample_cache = SampleCache(processed_cache, bos_token_id=self.tokenizer.bos_token_id)

                cache = Subset(sample_cache, indices=range(min(len(sample_cache), self.method_cfg.analysis.max_activating_examples.max_num_samples)))
                dataloader = DataLoader(cache, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
                
                for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Processing dataset {dataset_name}"):
                    # Load sample
                    tokens, activations = batch
                    activations = activations[0].to(self.device)
                    tokens = tokens[0]
                    assert tokens.ndim == 1, f"Expected 1D tokens, got shape {tokens.shape}"
                    
                    # PCA transform
                    projections = pca.transform(activations)  # [seq_len, n_components]
                    assert projections.ndim == 2, f"Expected 2D projections, got shape {projections.shape}"
                    assert projections.shape == (len(tokens), pca.n_components)

                    # Compute max/min scores
                    max_positive_score_per_component = torch.max(projections, dim=0)[0]
                    max_negative_score_per_component = torch.min(projections, dim=0)[0]
                    input_ids_batch = tokens.repeat(pca.n_components, 1) # [n_components, seq_len]
                    assert input_ids_batch.shape == (pca.n_components, len(tokens))

                    projections_T = projections.T
                    assert projections_T.shape == (pca.n_components, len(tokens))

                    # Write to stores 
                    # For future information: Writing is the bottleneck here. 
                    positive_writer.add_batch_examples(
                        scores_per_example=max_positive_score_per_component,
                        input_ids_batch=input_ids_batch,
                        scores_per_token_batch=projections_T,
                        latent_idx=latent_idx,
                        dataset_name=dataset_name,
                    )

                    negative_writer.add_batch_examples(
                        scores_per_example=-max_negative_score_per_component, # Negate the scores to get the minimum activating examples
                        input_ids_batch=input_ids_batch,
                        scores_per_token_batch=projections_T,
                        latent_idx=latent_idx,
                        dataset_name=dataset_name,
                    )

    def _run_analysis_for_layer(
        self, layer_idx: int, target: str, model_results_dir: Path
    ) -> None:
        """Run analysis pipeline for a trained PCA model."""
        # Create analysis directory structure
        analysis_dir = model_results_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        plots_dir = model_results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Load PCA model for analysis
        pca = self._load_pca_model(model_results_dir / "pca_model.pkl")
        
        # Save explained variance analysis and create plots
        if pca.explained_variance_ratio_ is not None:
            explained_var_ratio = pca.explained_variance_ratio_
            cumulative_var = explained_var_ratio.cumsum(dim=0)
            variance_analysis = {
                'explained_variance_ratio': explained_var_ratio.tolist(),
                'cumulative_variance_ratio': cumulative_var.tolist(),
                'total_variance_explained': float(explained_var_ratio.sum()),
            }
            
            with open(analysis_dir / "variance_analysis.json", 'w') as f:
                json.dump(variance_analysis, f, indent=2)
            
            # Create explained variance plot
            self._create_explained_variance_plot(
                explained_var_ratio, cumulative_var, plots_dir, layer_idx
            )

        self._collect_maximum_activating_examples(pca, layer_idx, analysis_dir, target)

        

        logger.info(f"Analysis completed for layer {layer_idx}")
    
    


    def _load_pca_model(self, model_path: Path) -> Dict[str, Any]:
        """Load PCA model state from disk."""
        with open(model_path, 'rb') as f:
            pca_state = pickle.load(f)

        pca = IncrementalPCA(
            n_components=pca_state['n_components'],
            batch_size=self.method_cfg.training.batch_size,
            device=self.device,
            verbose=True,
        )
        pca.components_ = pca_state['components_'].to(self.device)
        pca.explained_variance_ = pca_state['explained_variance_'].to(self.device)
        pca.explained_variance_ratio_ = pca_state['explained_variance_ratio_'].to(self.device)
        pca.mean_ = pca_state['mean_'].to(self.device)
        pca.n_samples_seen_ = pca_state['n_samples_seen_']
        if 'var_' in pca_state:
            pca.var_ = pca_state['var_'].to(self.device)
        else:
            logger.warning("No var_ in PCA state, setting to None")
            pca.var_ = None
        if 'noise_variance_' in pca_state:
            pca.noise_variance_ = pca_state['noise_variance_']
        else:
            logger.warning("No noise_variance_ in PCA state, setting to None")
            pca.noise_variance_ = None
        return pca

    def _create_explained_variance_plot(
        self, 
        explained_var_ratio: torch.Tensor, 
        cumulative_var: torch.Tensor, 
        plots_dir: Path, 
        layer_idx: int, 
    ) -> None:
        """Create and save explained variance plot."""
        import matplotlib.pyplot as plt
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Convert to numpy for plotting
        explained_var_np = explained_var_ratio.cpu().numpy()
        cumulative_var_np = cumulative_var.cpu().numpy()
        components = range(1, len(explained_var_np) + 1)
        
        # Plot 1: Individual explained variance
        ax1.bar(components, explained_var_np)
        ax1.set_xlabel('Component Number')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title(f'Individual Component Variance\nLayer {layer_idx}')
        ax1.grid(True, alpha=0.3)
        
        # Add percentage labels on top 5 components
        for i in range(min(5, len(explained_var_np))):
            ax1.text(i + 1, explained_var_np[i] + 0.005, 
                    f'{explained_var_np[i]:.1%}', 
                    ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Cumulative explained variance
        ax2.plot(components, cumulative_var_np, marker='o', linewidth=2, markersize=3)
        ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% threshold')
        ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
        ax2.set_xlabel('Component Number')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title(f'Cumulative Explained Variance\nLayer {layer_idx}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add annotation for 90% threshold
        threshold_90_idx = (cumulative_var_np >= 0.9).nonzero()[0]
        if len(threshold_90_idx) > 0:
            idx_90 = threshold_90_idx[0] + 1  # +1 because components start from 1
            ax2.annotate(f'{idx_90} components\nfor 90% variance', 
                        xy=(idx_90, 0.9), xytext=(idx_90 + 10, 0.8),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=9, ha='left')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"explained_variance_layer_{layer_idx}.png"
        plot_path = plots_dir / plot_filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved explained variance plot to {plot_path}")

    def visualize(self) -> None:
        """
        Create Streamlit visualization for PCA results.
        
        Features:
        - Component Analysis tab: Show explained variance and component statistics
        - Online Inference tab: Real-time PCA projection analysis
        - Plots tab: Display variance explained plots and component visualizations
        """
        st.subheader("PCA Analysis")
        
        # Global PCA selector
        available_pcas = self._get_available_pca_directories()
        if not available_pcas:
            st.error(f"No trained PCA directories found in {self.results_dir / 'pca'}")
            return
        
        # Group PCAs by layer for easier selection
        pcas_by_layer = defaultdict(list)
        for pca_info in available_pcas:
            pcas_by_layer[pca_info['layer']].append(pca_info)

        selected_pca_info = None
        unique_layers = sorted(pcas_by_layer.keys())

        if not unique_layers:
            st.error("No layers with trained PCAs found.")
            return

        # Layer selection
        selected_layer = st.selectbox(
            "Select Layer",
            options=unique_layers,
            help="Choose the layer for which to analyze PCAs"
        )

        pcas_for_selected_layer = pcas_by_layer[selected_layer]
        targets_for_layer = [pca['target'] for pca in pcas_for_selected_layer]
        
        if not targets_for_layer:
            st.warning(f"No trained PCA models found for layer {selected_layer}.")
            return

        # Target selection with human-readable names
        
        
        # Create display options with human-readable names
        display_options = [self.target_display_names.get(target, target) for target in targets_for_layer]
        
        # Default to first option (difference_ftb -> "FT - Base")
        default_index = 0
        if "difference_ftb" in targets_for_layer:
            default_index = targets_for_layer.index("difference_ftb")
        
        selected_display_target = st.selectbox(
            "Select Target Direction",
            options=display_options,
            index=default_index,
            help="Choose which difference direction to analyze"
        )

        # Map back to actual target name
        selected_target = None
        for target, display_name in self.target_display_names.items():
            if display_name == selected_display_target:
                selected_target = target
                break
        
        # Find the complete pca_info dictionary
        for pca_info in pcas_for_selected_layer:
            if pca_info['target'] == selected_target:
                selected_pca_info = pca_info
                break
        
        assert selected_pca_info is not None, "Failed to retrieve selected PCA information"
        
        # Display training metrics if available
        training_metrics_path = selected_pca_info['path'] / "training_metrics.json"
        if training_metrics_path.exists():
            try:
                with open(training_metrics_path, 'r') as f:
                    training_metrics = json.load(f)
                st.markdown(f"**Training Samples:** {training_metrics.get('total_samples', 'N/A')}")
            except Exception as e:
                st.warning(f"Could not load training metrics: {str(e)}")

        # Create tabs for different analyses
        from src.utils.visualization import multi_tab_interface
        multi_tab_interface(
            [
                ("🏆 Component Analysis", lambda: self._render_component_analysis_tab(selected_pca_info)),
                ("📊 MaxAct Examples", lambda: self._render_max_examples_tab(selected_pca_info)),
                ("🔥 Online Inference", lambda: PCAOnlineDashboard(self, selected_pca_info).display()),
                ("🎨 Plots", lambda: self._render_plots_tab(selected_pca_info)),
            ],
            "PCA Analysis",
        )

    def _get_available_pca_directories(self):
        """Get list of available trained PCA directories by target."""
        pca_base_dir = self.results_dir / "pca"
        if not pca_base_dir.exists():
            return []
        
        available_pcas = []
        valid_targets = ["difference_ftb", "difference_bft", "base", "ft"]
        
        # Scan through layer directories
        for layer_dir in pca_base_dir.iterdir():
            if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
                continue
            
            # Extract layer number
            try:
                layer_num = int(layer_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            
            # Scan for target directories in this layer
            for target in valid_targets:
                target_dir = layer_dir / target
                if target_dir.is_dir():
                    # Check if this looks like a valid PCA directory
                    if ((target_dir / "pca_model.pkl").exists() or 
                        (target_dir / "training_config.yaml").exists()):
                        available_pcas.append({
                            'layer': layer_num,
                            'target': target,
                            'path': target_dir,
                            'layer_dir': layer_dir
                        })
        
        # Sort by layer number, then by target
        available_pcas.sort(key=lambda x: (x['layer'], x['target']))
        return available_pcas

    def _render_component_analysis_tab(self, selected_pca_info):
        """Render component analysis tab with variance explained and component statistics."""
        target = selected_pca_info['target']
        layer = selected_pca_info['layer']
        model_results_dir = selected_pca_info['path']
        
        # Human-readable target name
        
        target_display = self.target_display_names.get(target, target)
        
        st.markdown(f"**Selected PCA:** Layer {layer} - {target_display}")
        
        # Load PCA model
        try:
            pca = self._load_pca_model(model_results_dir / "pca_model.pkl")
        except Exception as e:
            st.error(f"Failed to load PCA model: {str(e)}")
            return
        
        # Display variance analysis
        if pca.explained_variance_ratio_ is not None:
            st.markdown("### Explained Variance Analysis")
            
            explained_var_ratio = pca.explained_variance_ratio_
            cumulative_var = explained_var_ratio.cumsum(dim=0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Variance Explained", 
                    f"{float(explained_var_ratio.sum()):.4f}"
                )
                st.metric(
                    "First Component", 
                    f"{float(explained_var_ratio[0]):.4f}"
                )
            with col2:
                st.metric(
                    "Top 10 Components", 
                    f"{float(explained_var_ratio[:10].sum()):.4f}"
                )
                st.metric(
                    "Top 50 Components", 
                    f"{float(explained_var_ratio[:50].sum()):.4f}"
                )
            
            with col3:
                st.metric(
                    "Top 100 Components", 
                    f"{float(explained_var_ratio[:100].sum()):.4f}"
                )
                st.metric(
                    "Components for 90% Variance", 
                    f"{int((cumulative_var < 0.9).sum() + 1)}"
                )
            
            # Plot explained variance
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Individual explained variance
            ax1.bar(range(len(explained_var_ratio)), explained_var_ratio.cpu().numpy())
            ax1.set_xlabel('Component')
            ax1.set_ylabel('Explained Variance Ratio')
            ax1.set_title('Individual Component Variance')
            
            # Cumulative explained variance
            ax2.plot(range(len(cumulative_var)), cumulative_var.cpu().numpy())
            ax2.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
            ax2.set_xlabel('Component')
            ax2.set_ylabel('Cumulative Explained Variance')
            ax2.set_title('Cumulative Explained Variance')
            ax2.legend()
            
            st.pyplot(fig)
            plt.close()
        
        # Display component statistics
        if pca.components_ is not None:
            st.markdown("### Component Statistics")
            
            components = pca.components_  # Shape: [n_components, activation_dim]
            
            st.markdown(f"**Component Matrix Shape:** {list(components.shape)}")
            
            # Component norms
            component_norms = torch.norm(components, dim=1)
            st.markdown(f"**Mean Component Norm:** {float(component_norms.mean()):.4f}")
            st.markdown(f"**Component Norm Range:** [{float(component_norms.min()):.4f}, {float(component_norms.max()):.4f}]")

    def _render_max_examples_tab(self, selected_pca_info):
        """Render maximum activating examples tab using MaxActivationDashboardComponent."""

        target = selected_pca_info['target']
        layer = selected_pca_info['layer']
        model_results_dir = selected_pca_info['path']
        
        # Human-readable target name
        target_display = self.target_display_names.get(target, target)
        
        st.markdown(f"**Selected PCA:** Layer {layer} - {target_display}")
        
        # Look for max examples databases in analysis directory
        analysis_dir = model_results_dir / "analysis"
        pos_examples_path = analysis_dir / "positive_examples.db"
        neg_examples_path = analysis_dir / "negative_examples.db"
        
        # Check if either database exists
        if not pos_examples_path.exists() and not neg_examples_path.exists():
            st.error(f"No maximum examples databases found in {analysis_dir}")
            st.info("Run PCA analysis to generate maximum activating examples.")
            return
        
        # Create selection for positive/negative examples
        available_options = []
        if pos_examples_path.exists():
            available_options.append("Positive Examples")
        if neg_examples_path.exists():
            available_options.append("Negative Examples")
        
        if len(available_options) == 0:
            st.error("No example databases found")
            return
        
        selected_type = st.selectbox(
            "Example Type",
            options=available_options,
            index=0,
            help="Choose between positive and negative activating examples"
        )
        
        # Determine which database to load
        if selected_type == "Positive Examples":
            max_store_path = pos_examples_path
            example_type_display = "Positive"
        else:
            max_store_path = neg_examples_path
            example_type_display = "Negative"
        
        # Create MaxActStore instance
        assert self.tokenizer is not None, "Tokenizer must be available for MaxActStore visualization"
        max_store = ReadOnlyMaxActStore(
            max_store_path, 
            tokenizer=self.tokenizer,
        )

        # Create and display the dashboard component
        component = MaxActivationDashboardComponent(
            max_store, 
            title=f"PCA {example_type_display} Examples - Layer {layer} - {target_display}"
        )
        component.display()

    def _render_plots_tab(self, selected_pca_info):
        """Render plots tab displaying explained variance plots."""
        import streamlit as st
        
        target = selected_pca_info['target']
        layer = selected_pca_info['layer']
        model_results_dir = selected_pca_info['path']
        
        # Human-readable target name
        target_display = self.target_display_names.get(target, target)
        
        st.markdown("### PCA Analysis Plots")
        st.markdown(f"**Selected PCA:** Layer {layer} - {target_display}")
        
        # Look for plots directory
        plots_dir = model_results_dir / "plots"
        if not plots_dir.exists():
            st.warning(f"No plots directory found at {plots_dir}")
            st.info("Run analysis to generate plots.")
            return
        
        # Find explained variance plot
        plot_filename = f"explained_variance_layer_{layer}.png"
        plot_path = plots_dir / plot_filename
        
        if plot_path.exists():
            st.markdown("#### Explained Variance Analysis")
            st.image(str(plot_path), caption=f"Explained Variance - Layer {layer}", use_container_width=True)
            
            # Also display the numerical summary
            variance_analysis_path = model_results_dir / "analysis" / "variance_analysis.json"
            if variance_analysis_path.exists():
                try:
                    with open(variance_analysis_path, 'r') as f:
                        variance_analysis = json.load(f)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Variance Explained", 
                                f"{variance_analysis['total_variance_explained']:.2%}")
                    
                    with col2:
                        # Find components needed for 90% variance
                        cumulative_var = variance_analysis['cumulative_variance_ratio']
                        components_90 = next((i + 1 for i, v in enumerate(cumulative_var) if v >= 0.9), len(cumulative_var))
                        st.metric("Components for 90% Variance", components_90)
                    
                    with col3:
                        # First component variance
                        first_component_var = variance_analysis['explained_variance_ratio'][0]
                        st.metric("First Component Variance", f"{first_component_var:.2%}")
                        
                except Exception as e:
                    st.warning(f"Could not load variance analysis: {str(e)}")
        else:
            st.warning(f"Explained variance plot not found at {plot_path}")
            st.info("Run analysis to generate the explained variance plot.")
        
        # Look for other plots in the directory
        other_plots = [f for f in plots_dir.glob("*.png") if f.name != plot_filename]
        
        if other_plots:
            st.markdown("#### Other Plots")
            for plot_file in other_plots:
                st.markdown(f"**{plot_file.stem.replace('_', ' ').title()}**")
                st.image(str(plot_file), use_container_width=True)

    @torch.no_grad()
    def compute_pca_projections_for_tokens(
        self, pca: IncrementalPCA, target: str, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer: int
    ) -> Dict[str, Any]:
        """
        Compute PCA projections for given tokens (used by online dashboard).
        
        This method:
        1. Extracts activations from both base and finetuned models
        2. Computes activation differences based on training target
        3. Projects differences onto PCA components
        4. Returns tokens, component projections, and statistics
        
        Args:
            target: Target direction (difference_ftb or difference_bft)
            input_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len]  
            layer: Layer index to analyze
            
        Returns:
            Dictionary with tokens, component_projections, and statistics
        """
        from nnsight import LanguageModel
        
        # Shape assertions
        assert input_ids.ndim == 2, f"Expected 2D input_ids, got shape {input_ids.shape}"
        assert attention_mask.ndim == 2, f"Expected 2D attention_mask, got shape {attention_mask.shape}"
        assert input_ids.shape == attention_mask.shape, f"Shape mismatch: input_ids {input_ids.shape} vs attention_mask {attention_mask.shape}"
        
        # Get models
        base_nn_model = LanguageModel(self.base_model, tokenizer=self.tokenizer)  # type: ignore
        finetuned_nn_model = LanguageModel(self.finetuned_model, tokenizer=self.tokenizer)  # type: ignore
        
        # Prepare input batch
        batch = {
            'input_ids': input_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device)
        }
        
        # Get tokens for display
        token_ids = input_ids[0].cpu().numpy()
        tokens = [self.tokenizer.decode([token_id]) for token_id in token_ids]  # type: ignore
        
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
        
        # Shape assertions
        batch_size, seq_len, hidden_dim = base_acts.shape
        assert finetuned_acts.shape == (batch_size, seq_len, hidden_dim), f"Shape mismatch: base {base_acts.shape} vs finetuned {finetuned_acts.shape}"
        
        # Compute activation differences based on training target
        if target == "difference_bft":  # base - finetuned
            activation_diffs = finetuned_acts - base_acts
        elif target == "difference_ftb":  # finetuned - base
            activation_diffs = base_acts - finetuned_acts
        elif target == "base":
            activation_diffs = base_acts
        elif target == "ft":
            activation_diffs = finetuned_acts
            
        # Shape assertion for differences
        assert activation_diffs.shape == (batch_size, seq_len, hidden_dim), f"Expected diff shape {(batch_size, seq_len, hidden_dim)}, got {activation_diffs.shape}"
        
        # Take first sequence for analysis
        diff_sequence = activation_diffs[0]  # [seq_len, hidden_dim]

        component_projections = pca.transform(diff_sequence)  # [seq_len, n_components]
        
        # Shape assertion for projections
        n_components = pca.n_components
        assert component_projections.shape == (seq_len, n_components), f"Expected projection shape {(seq_len, n_components)}, got {component_projections.shape}"
        
        # Convert to numpy for visualization
        component_projections_np = component_projections.detach().numpy()
            
        return {
            'tokens': tokens,
            'component_projections': component_projections_np,
            'total_tokens': len(tokens),
            'layer': layer,
            'n_components': n_components
        }

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available PCA difference results.

        Args:
            results_dir: Base results directory

        Returns:
            Dict mapping {model_pair: {organism: {layer: path_to_results}}}
        """
        results = defaultdict(dict)
        results_base = results_dir

        if not results_base.exists():
            return results

        # Scan for PCA results in the expected structure
        for base_model_dir in results_base.iterdir():
            if not base_model_dir.is_dir():
                continue

            model_name = base_model_dir.name

            for organism_dir in base_model_dir.iterdir():
                if not organism_dir.is_dir():
                    continue

                organism_name = organism_dir.name
                pca_dir = organism_dir / "pca"
                if pca_dir.exists() and any(pca_dir.iterdir()):
                    results[model_name][organism_name] = str(pca_dir)

        return results


class PCAOnlineDashboard(AbstractOnlineDiffingDashboard):
    """
    Online dashboard for interactive PCA difference analysis.
    
    This dashboard allows users to input text and see per-token PCA component projections
    highlighted directly in the text, similar to KL/NormDiff dashboards but for PCA analysis.
    """
    def __init__(self, method_instance, pca_info):
        super().__init__(method_instance)
        self.pca_info = pca_info

        self._pca = None
    
    def _render_streamlit_method_controls(self) -> Dict[str, Any]:
        """Render PCA-specific controls in Streamlit."""
        import streamlit as st

        if self._pca is None:
            pca_model_path = self.pca_info['path'] / "pca_model.pkl"
            self._pca = self.method._load_pca_model(pca_model_path)
        n_components = self._pca.n_components
        
        # Component selection
        selected_component = st.selectbox(
            "Select PCA Component",
            options=list(range(n_components)),
            index=0,
            help=f"Choose which PCA component to visualize (0-{n_components-1})"
        )
        
        return {"selected_component": selected_component}
        
    def compute_statistics_for_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, Any]:
        """Compute PCA projection statistics."""
        layer = self.pca_info['layer']
        target = self.pca_info['target']
        selected_component = kwargs.get('selected_component', 0)
        
        # Get full PCA projections from the parent method
        results = self.method.compute_pca_projections_for_tokens(self._pca, target, input_ids, attention_mask, layer)
        
        # Extract values for the selected component only
        component_projections = results['component_projections']  # [seq_len, n_components]
        assert selected_component < component_projections.shape[1], f"Component {selected_component} not available (max: {component_projections.shape[1]-1})"
        
        values = component_projections[:, selected_component]  # [seq_len]
        analysis_title = f"PCA Component {selected_component} Projection"
        
        # Return adapted results for the abstract dashboard
        return {
            'tokens': results['tokens'],
            'values': values,
            'total_tokens': results['total_tokens'],
            'analysis_title': analysis_title
        }
    
    def get_method_specific_params(self) -> Dict[str, Any]:
        """Get PCA-specific parameters."""
        return {}
    
    def _get_title(self) -> str:
        """Get title for PCA analysis."""
        return "PCA Difference Analysis"
