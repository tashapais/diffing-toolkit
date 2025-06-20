"""
Crosscoder-based model diffing method.

This module trains crosscoders on paired activations from base and finetuned models,
then runs a comprehensive analysis pipeline including evaluation notebooks, scaler computation,
latent statistics, and KL divergence experiments.

Key assumptions:
- Preprocessing pipeline has generated paired activation caches
- dictionary_learning library is available and compatible
- science-of-finetuning repository is available for analysis pipeline
- W&B configuration is available in infrastructure config
- Sufficient GPU memory and disk space for training
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import torch
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import json

from .diffing_method import DiffingMethod
from src.utils.activations import get_layer_indices
from src.utils.dictionary.analysis import build_push_crosscoder_latent_df, make_plots
from src.utils.dictionary.training import train_crosscoder_for_layer
from src.utils.dictionary.latent_scaling.closed_form import compute_scalers_from_config
from src.utils.dictionary.latent_scaling.beta_analysis import make_beta_df
from src.utils.dictionary.latent_activations import (
    collect_dictionary_activations_from_config,
    collect_activating_examples,
    update_latent_df_with_stats,
)
from src.utils.dictionary.utils import load_dictionary_model
from src.utils.dictionary.training import crosscoder_run_name


class CrosscoderDiffingMethod(DiffingMethod):
    """
    Trains crosscoders on paired activations and runs comprehensive analysis.

    This method:
    1. Loads paired activation caches from preprocessing pipeline
    2. Trains crosscoders for specified layers using local shuffling
    3. Saves trained models with configuration and metrics
    4. Optionally uploads models to Hugging Face Hub
    5. Runs complete analysis pipeline from science-of-finetuning
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

    def run(self) -> Dict[str, Any]:
        """
        Main training orchestration with analysis pipeline.

        Trains crosscoders for each specified layer, then runs the complete
        analysis pipeline for each trained model.

        Returns:
            Dictionary containing training results, model paths, and analysis outcomes

        Assumptions:
            - Paired activation caches exist for all specified layers
            - Sufficient resources for training and analysis
        """
        logger.info(f"Starting crosscoder training for layers: {self.layers}")

        for layer_idx in self.layers:
            logger.info(f"Processing layer {layer_idx}")

            logger.info(f"Training crosscoder for layer {layer_idx}")

            dictionary_name = crosscoder_run_name(
                self.cfg, layer_idx, self.base_model_cfg, self.finetuned_model_cfg
            )
            model_results_dir = (
                self.results_dir / "crosscoder" / f"layer_{layer_idx}" / dictionary_name
            )
            model_results_dir.mkdir(parents=True, exist_ok=True)
            if (
                not (model_results_dir / "dictionary_model" / "model.safetensors").exists()
                or self.method_cfg.training.overwrite
            ):  
                # Train crosscoder for this layer
                training_metrics, model_path = train_crosscoder_for_layer(
                    self.cfg, layer_idx, self.device, dictionary_name
                )
                # save training metrics
                with open(model_results_dir / "training_metrics.json", "w") as f:
                    json.dump(training_metrics, f)
                # save model
                dictionary_model = load_dictionary_model(model_path)
                dictionary_model.save_pretrained(model_results_dir / "dictionary_model")

                # save training configs
                OmegaConf.save(self.cfg, model_results_dir / "training_config.yaml")
            else:
                logger.info(
                    f"Found trained model at {model_results_dir / 'dictionary_model'}"
                )
                training_metrics = json.load(
                    open(model_results_dir / "training_metrics.json")
                )

            if self.method_cfg.analysis.enabled:
                logger.info(f"Storing analysis results in {model_results_dir}")
                build_push_crosscoder_latent_df(
                    dictionary_name=dictionary_name,
                    base_layer=0,
                    ft_layer=1,
                )

                if self.method_cfg.analysis.latent_scaling.enabled:
                    compute_scalers_from_config(
                        cfg=self.cfg,
                        layer=layer_idx,
                        dictionary_model=dictionary_name,
                        results_dir=model_results_dir,
                    )
                    make_beta_df(
                        dictionary_name,
                        model_results_dir,
                        num_samples=self.method_cfg.analysis.latent_scaling.num_samples,
                    )

                if self.method_cfg.analysis.latent_activations.enabled:
                    latent_activations_cache = (
                        collect_dictionary_activations_from_config(
                            cfg=self.cfg,
                            layer=layer_idx,
                            dictionary_model_name=dictionary_name,
                            result_dir=model_results_dir,
                        )
                    )
                    collect_activating_examples(
                        dictionary_model_name=dictionary_name,
                        latent_activation_cache=latent_activations_cache,
                        n=self.method_cfg.analysis.latent_activations.n_max_activations,
                        upload_to_hub=self.method_cfg.analysis.latent_activations.upload_to_hub,
                        overwrite=self.method_cfg.analysis.latent_activations.overwrite,
                        save_path=model_results_dir,
                    )
                    update_latent_df_with_stats(
                        dictionary_name=dictionary_name,
                        latent_activation_cache=latent_activations_cache,
                        split_of_cache=self.method_cfg.analysis.latent_activations.split,
                    )
                    
                make_plots(
                    dictionary_name=dictionary_name,
                    plots_dir=model_results_dir,
                )

            logger.info(f"Successfully completed layer {layer_idx}")

    def visualize(self) -> None:
        """
        Create visualizations for crosscoder training and analysis results.

        Note: Primary visualizations are handled by the analysis pipeline.
        This method can be extended for additional custom visualizations.
        """
        logger.info("Crosscoder visualizations are generated by the analysis pipeline")

        # Check if any results exist
        if not any(self.results_dir.glob("*/*/layer_*")):
            logger.warning("No trained crosscoders found for visualization")
            return

        raise NotImplementedError(
            "Crosscoder visualizations are generated by the analysis pipeline"
        )

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Find all available crosscoder results.

        Args:
            results_dir: Base results directory

        Returns:
            Dict mapping {model_pair: {organism: {layer: path_to_results}}}
        """
        crosscoder_results_dir = results_dir / "crosscoder"

        if not crosscoder_results_dir.exists():
            return {}

        results = {}

        # Iterate through model pairs (base-finetuned combinations)
        for model_pair_dir in crosscoder_results_dir.iterdir():
            if not model_pair_dir.is_dir():
                continue

            model_pair = model_pair_dir.name
            results[model_pair] = {}

            # Iterate through organisms
            for organism_dir in model_pair_dir.iterdir():
                if not organism_dir.is_dir():
                    continue

                organism = organism_dir.name
                results[model_pair][organism] = {}

                # Iterate through layers
                for layer_dir in organism_dir.iterdir():
                    if not layer_dir.is_dir() or not layer_dir.name.startswith(
                        "layer_"
                    ):
                        continue

                    # Check if this directory contains a trained model
                    if (layer_dir / "model.pt").exists():
                        layer_name = layer_dir.name
                        results[model_pair][organism][layer_name] = str(layer_dir)

        return results
