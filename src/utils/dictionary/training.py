import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from typing import Dict, Any, Tuple, Callable, List
from omegaconf import DictConfig
from loguru import logger
import wandb
import tempfile
from omegaconf import OmegaConf
from tqdm import tqdm
from dictionary_learning import CrossCoder, BatchTopKCrossCoder
from dictionary_learning.cache import ActivationNormalizer
from dictionary_learning.trainers.crosscoder import (
    CrossCoderTrainer,
    BatchTopKCrossCoderTrainer,
)
from dictionary_learning.trainers import BatchTopKTrainer, BatchTopKSAE
from dictionary_learning.cache import ActivationCache, PairedActivationCache
from dictionary_learning.training import trainSAE

from ..activations import (
    load_activation_datasets_from_config,
    get_local_shuffled_indices,
    calculate_samples_per_dataset,
)
from ..configs import get_model_configurations, get_dataset_configurations
from ..dictionary.utils import push_dictionary_model


class DifferenceCache:
    """
    Cache for computing activation differences between two activation caches.

    This is used for SAE training on difference targets (base-chat or chat-base).
    """

    def __init__(self, cache_1: ActivationCache, cache_2: ActivationCache):
        self.activation_cache_1 = cache_1
        self.activation_cache_2 = cache_2
        assert len(self.activation_cache_1) == len(self.activation_cache_2)

    def __len__(self):
        return len(self.activation_cache_1)

    def __getitem__(self, index: int):
        return self.activation_cache_1[index] - self.activation_cache_2[index]

    @property
    def tokens(self):
        return torch.stack(
            (self.activation_cache_1.tokens, self.activation_cache_2.tokens), dim=0
        )

    @property
    def config(self):
        return self.activation_cache_1.config


def compute_normalizer(caches: List[PairedActivationCache]) -> ActivationNormalizer:
    """
    Compute the normalizer for a dictionary of caches.
    """
    running_stats_1 = None
    running_stats_2 = None
    for cache in caches:
        if running_stats_1 is None:
            running_stats_1 = cache.activation_cache_1.running_stats
            running_stats_2 = cache.activation_cache_2.running_stats
        else:
            running_stats_1.merge(cache.activation_cache_1.running_stats)
            running_stats_2.merge(cache.activation_cache_2.running_stats)

    mean = torch.stack([running_stats_1.mean, running_stats_2.mean], dim=0)
    std = torch.stack([running_stats_1.std(), running_stats_2.std()], dim=0)
    return ActivationNormalizer(mean, std)

def setup_training_datasets(
    cfg: DictConfig,
    layer: int,
    dataset_processing_function: Callable = None,
    overwrite_num_samples: int = None,
    overwrite_num_validation_samples: int = None,
    overwrite_local_shuffling: bool = None,
) -> Tuple[Dataset, Dataset]:
    """
    Set up training and validation datasets using preprocessing configuration.

    Args:
        cfg: Full configuration
        layer: Layer index to load activations for
        dataset_processing_function: Function to process the dataset
        overwrite_num_samples: If provided, use this number of samples instead of the one in the configuration
        overwrite_num_validation_samples: If provided, use this number of validation samples instead of the one in the configuration
        overwrite_local_shuffling: If provided, use this local shuffling setting instead of the one in the configuration    

    Returns:
        Tuple of (training_dataset, validation_dataset)
    """
    base_model_cfg, finetuned_model_cfg = get_model_configurations(cfg)
    dataset_cfgs = get_dataset_configurations(
        cfg,
        use_chat_dataset=cfg.diffing.method.datasets.use_chat_dataset,
        use_pretraining_dataset=cfg.diffing.method.datasets.use_pretraining_dataset,
        use_training_dataset=cfg.diffing.method.datasets.use_training_dataset,
    )

    training_cfg = cfg.diffing.method.training

    caches = load_activation_datasets_from_config(
        cfg=cfg,
        ds_cfgs=dataset_cfgs,
        base_model_cfg=base_model_cfg,
        finetuned_model_cfg=finetuned_model_cfg,
        layers=[layer],
        split="train",
    )  # Dict {dataset_name: {layer: PairedActivationCache, ...}}

    # Collapse layers
    caches = {
        dataset_name: caches[dataset_name][layer] for dataset_name in caches
    }  # Dict {dataset_name: PairedActivationCache}

    normalizer = compute_normalizer(list(caches.values())) if training_cfg.normalize_activations else None

    if dataset_processing_function is not None:
        caches = {
            dataset_name: dataset_processing_function(caches[dataset_name])
            for dataset_name in caches
        }

    # Determine number of samples per dataset
    num_samples_per_dataset = [len(caches[dataset_name]) for dataset_name in caches]

    num_samples_per_dataset = calculate_samples_per_dataset(
        num_samples_per_dataset,
        (
            training_cfg.num_samples
            if overwrite_num_samples is None
            else overwrite_num_samples
        ),
    )

    logger.info(f"Using {sum(num_samples_per_dataset)} samples in total")
    for dataset_name, num_samples in zip(caches.keys(), num_samples_per_dataset):
        logger.info(f"\tUsing {num_samples} samples for {dataset_name}")

    # Create training dataset
    train_dataset = ConcatDataset(
        [
            Subset(caches[dataset_name], torch.arange(0, num_samples))
            for dataset_name, num_samples in zip(caches.keys(), num_samples_per_dataset)
        ]
    )

    # Apply local shuffling if enabled
    if training_cfg.local_shuffling and (
        overwrite_local_shuffling is None or overwrite_local_shuffling
    ):
        logger.info("Applying local shuffling for cache locality optimization")

        # Get shard size from cache configuration
        shard_size = training_cfg.local_shuffling_shard_size

        shuffled_indices, epoch_numbers = get_local_shuffled_indices(
            num_samples_per_dataset, shard_size, training_cfg.epochs
        )

        train_dataset = Subset(train_dataset, shuffled_indices)
        logger.info(
            f"Created shuffled training dataset with {len(train_dataset)} samples"
        )

    # Load validation datasets
    caches_val = load_activation_datasets_from_config(
        cfg=cfg,
        ds_cfgs=dataset_cfgs,
        base_model_cfg=base_model_cfg,
        finetuned_model_cfg=finetuned_model_cfg,
        layers=[layer],
        split="validation",
    )

    # Collapse layers
    caches_val = {
        dataset_name: caches_val[dataset_name][layer] for dataset_name in caches_val
    }  # Dict {dataset_name: PairedActivationCache}

    num_validation_samples = [
        len(caches_val[dataset_name]) for dataset_name in caches_val
    ]
    num_validation_samples = calculate_samples_per_dataset(
        num_validation_samples,
        (
            training_cfg.num_validation_samples
            if overwrite_num_validation_samples is None
            else overwrite_num_validation_samples
        ),
    )

    validation_dataset = ConcatDataset(
        [
            Subset(caches_val[dataset_name], torch.arange(0, num_validation_samples))
            for dataset_name, num_validation_samples in zip(
                caches_val.keys(), num_validation_samples
            )
        ]
    )

    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(validation_dataset)} samples")

    return train_dataset, validation_dataset, epoch_numbers, normalizer


def create_training_dataloader(
    dataset: Dataset, cfg: DictConfig, shuffle: bool = True
) -> DataLoader:
    """
    Create DataLoader with configuration from preprocessing and method settings.

    Args:
        dataset: Dataset to create DataLoader for
        cfg: Full configuration
        shuffle: Whether to shuffle the data (disabled for local shuffling)

    Returns:
        Configured DataLoader
    """
    training_cfg = cfg.diffing.method.training

    return DataLoader(
        dataset,
        batch_size=training_cfg.batch_size,
        shuffle=shuffle
        and not training_cfg.local_shuffling,  # Don't shuffle if using local shuffling
        num_workers=training_cfg.workers,
        pin_memory=True,
    )


def upload_config_to_wandb(cfg: DictConfig) -> None:
    """
    Upload experiment configuration to Weights & Biases as an artifact.

    Args:
        cfg: Full configuration containing all hydra configs
    """
    # Convert OmegaConf to regular dict for YAML serialization
    config_dict = OmegaConf.to_yaml(cfg)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_dict)
        temp_path = f.name

    try:
        # Create and log the artifact
        artifact = wandb.Artifact(
            name="experiment_config",
            type="config",
            description="Full experiment configuration including all hydra configs",
        )
        artifact.add_file(temp_path, name="config.yaml")
        wandb.log_artifact(artifact)
        logger.debug("Successfully uploaded config artifact to W&B")
    finally:
        # Clean up temporary file
        Path(temp_path).unlink()


def create_crosscoder_trainer_config(
    cfg: DictConfig, layer: int, activation_dim: int, device: str, normalizer: ActivationNormalizer
) -> Dict[str, Any]:
    """
    Create trainer configuration from method settings.

    Args:
        cfg: Full configuration
        layer: Layer index being trained
        activation_dim: Dimension of input activations
        device: Training device

    Returns:
        Trainer configuration dictionary
    """
    method_cfg = cfg.diffing.method
    base_model_cfg, finetuned_model_cfg = get_model_configurations(cfg)

    # Extract training parameters
    expansion_factor = method_cfg.training.expansion_factor
    lr = method_cfg.training.lr
    mu = method_cfg.training.mu
    k = method_cfg.training.k

    # Extract model parameters
    model_type = method_cfg.model.type
    code_normalization = method_cfg.model.code_normalization
    same_init_for_all_layers = method_cfg.model.same_init_for_all_layers
    norm_init_scale = method_cfg.model.norm_init_scale
    init_with_transpose = method_cfg.model.init_with_transpose

    # Extract optimization parameters
    resample_steps = method_cfg.optimization.resample_steps
    warmup_steps = method_cfg.optimization.warmup_steps

    dictionary_size = expansion_factor * activation_dim

    # Create run name
    if model_type == "relu":
        run_name = (
            f"{base_model_cfg.model_id.split('/')[-1]}-L{layer}-mu{mu:.1e}-lr{lr:.0e}-x{expansion_factor}"
            + "-local-shuffling"
            + f"-{code_normalization.capitalize()}Loss"
        )
    elif model_type == "batch-top-k":
        run_name = (
            f"{base_model_cfg.model_id.split('/')[-1]}-L{layer}-k{k}-lr{lr:.0e}-x{expansion_factor}"
            + "-local-shuffling"
            + f"-{code_normalization.capitalize()}"
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # Common configuration
    common_config = {
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": lr,
        "resample_steps": resample_steps,
        "device": device,
        "warmup_steps": warmup_steps,
        "layer": layer,
        "lm_name": f"{finetuned_model_cfg.model_id}-{base_model_cfg.model_id}",
        "compile": True,
        "wandb_name": run_name,
        "dict_class_kwargs": {
            "same_init_for_all_layers": same_init_for_all_layers,
            "norm_init_scale": norm_init_scale,
            "init_with_transpose": init_with_transpose,
            "encoder_layers": None,  # Could be made configurable
            "code_normalization": code_normalization,
            "code_normalization_alpha_sae": 1.0,
            "code_normalization_alpha_cc": 0.1,
            "activation_normalizer": normalizer,
        },
    }

    # Type-specific configuration
    if model_type == "relu":
        trainer_config = {
            **common_config,
            "trainer": CrossCoderTrainer,
            "dict_class": CrossCoder,
            "l1_penalty": mu,
            "pretrained_ae": None,
            "use_mse_loss": False,
        }
    elif model_type == "batch-top-k":
        trainer_config = {
            **common_config,
            "trainer": BatchTopKCrossCoderTrainer,
            "dict_class": BatchTopKCrossCoder,
            "k": k,
            "k_max": method_cfg.training.k_max,
            "k_annealing_steps": method_cfg.training.k_annealing_steps,
            "auxk_alpha": method_cfg.training.auxk_alpha,
            "pretrained_ae": None,
        }
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return trainer_config, run_name


def setup_sae_cache(
    target: str, paired_cache: PairedActivationCache
) -> ActivationCache:
    """
    Setup caches for SAE training based on target type.

    Args:
        target: Training target ("base", "ft", "difference_bft", "difference_ftb")
        paired_cache: PairedActivationCache

    Returns:
        Processed cache for SAE training
    """
    if target == "base":
        processed_cache = paired_cache.activation_cache_1
    elif target == "ft":
        processed_cache = paired_cache.activation_cache_2
    elif target == "difference_bft":
        processed_cache = DifferenceCache(
            paired_cache.activation_cache_1, paired_cache.activation_cache_2
        )
    elif target == "difference_ftb":
        processed_cache = DifferenceCache(
            paired_cache.activation_cache_2, paired_cache.activation_cache_1
        )
    else:
        raise ValueError(f"Invalid SAE target: {target}")

    return processed_cache


def create_sae_trainer_config(
    cfg: DictConfig, layer: int, activation_dim: int, device: str
) -> Tuple[Dict[str, Any], str]:
    """
    Create SAE trainer configuration from method settings.

    Args:
        cfg: Full configuration
        layer: Layer index being trained
        activation_dim: Dimension of input activations
        device: Training device

    Returns:
        Tuple of (trainer configuration dictionary, run name)
    """
    method_cfg = cfg.diffing.method
    base_model_cfg, finetuned_model_cfg = get_model_configurations(cfg)

    # Extract SAE-specific parameters
    expansion_factor = method_cfg.training.expansion_factor
    lr = method_cfg.training.lr
    k = method_cfg.training.k
    target = method_cfg.training.target

    # Extract optimization parameters
    warmup_steps = method_cfg.optimization.warmup_steps

    dictionary_size = expansion_factor * activation_dim

    # Create run name for SAE
    run_name = (
        f"SAE-{target}-{base_model_cfg.model_id.split('/')[-1]}-L{layer}-k{k}-lr{lr:.0e}-x{expansion_factor}"
        + ("-local-shuffling" if method_cfg.training.local_shuffling else "")
    )

    # SAE trainer configuration
    trainer_config = {
        "trainer": BatchTopKTrainer,
        "dict_class": BatchTopKSAE,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": lr,
        "device": device,
        "warmup_steps": warmup_steps,
        "layer": layer,
        "lm_name": f"{finetuned_model_cfg.model_id}-{base_model_cfg.model_id}",
        "wandb_name": run_name,
        "k": k,
    }

    return trainer_config, run_name


def train_crosscoder_for_layer(
    cfg: DictConfig,
    layer_idx: int,
    device: str,
) -> Dict[str, Any]:
    """
    Train crosscoder for a specific layer (original implementation).
    """
    logger.info(f"Training crosscoder for layer {layer_idx}")

    # Setup training datasets
    train_dataset, val_dataset, epoch_idx_per_step, normalizer = setup_training_datasets(
        cfg, layer_idx
    )

    # Get activation dimension from first sample
    sample_activation = train_dataset[0]
    activation_dim = sample_activation.shape[-1]

    assert activation_dim > 0, f"Invalid activation dimension: {activation_dim}"
    logger.info(f"Activation dimension: {activation_dim}")

    # Create trainer configuration
    trainer_config, run_name = create_crosscoder_trainer_config(
        cfg, layer_idx, activation_dim, device, normalizer
    )

    # Create data loaders
    train_dataloader = create_training_dataloader(train_dataset, cfg, shuffle=True)
    val_dataloader = create_training_dataloader(val_dataset, cfg, shuffle=False)

    # Calculate max steps if not specified
    max_steps = cfg.diffing.method.training.max_steps
    if max_steps is None:
        max_steps = len(train_dataloader)
        trainer_config["steps"] = max_steps

    validate_every_n_steps = cfg.diffing.method.training.validate_every_n_steps

    logger.info(f"Training configuration: {trainer_config['wandb_name']}")
    logger.info(
        f"Training steps: {max_steps}, validation every: {validate_every_n_steps}"
    )
    save_dir = (
        f"{cfg.infrastructure.storage.checkpoint_dir}/{trainer_config['wandb_name']}"
    )

    # Train the crosscoder
    trainSAE(
        data=train_dataloader,
        trainer_config=trainer_config,
        validate_every_n_steps=validate_every_n_steps,
        validation_data=val_dataloader,
        use_wandb=cfg.wandb.enabled,
        wandb_entity=cfg.wandb.entity,
        wandb_project=cfg.wandb.project,
        log_steps=cfg.wandb.log_steps,
        save_dir=save_dir,
        steps=max_steps,
        save_steps=validate_every_n_steps,
        run_wandb_finish=False,
        epoch_idx_per_step=epoch_idx_per_step,
    )

    wandb_link = None
    hf_repo_id = None

    if cfg.wandb.enabled:
        wandb_link = wandb.run.get_url()
        upload_config_to_wandb(cfg)
        wandb.finish()

    if cfg.diffing.method.upload.model:
        hf_repo_id = push_dictionary_model(Path(save_dir) / "model_final.pt")

    # Collect training metrics
    training_metrics = {
        "layer": layer_idx,
        "activation_dim": activation_dim,
        "dictionary_size": trainer_config["dict_size"],
        "training_steps": max_steps,
        "lr": trainer_config["lr"],
        "wandb_link": wandb_link,
        "model_type": cfg.diffing.method.model.type,
        "run_name": trainer_config["wandb_name"],
        "hf_repo_id": hf_repo_id,
        "training_mode": "crosscoder",
    }

    logger.info(f"Successfully trained crosscoder for layer {layer_idx}")
    return training_metrics


def train_sae_for_layer(
    cfg: DictConfig,
    layer_idx: int,
    device: str,
) -> Dict[str, Any]:
    """
    Train SAE for a specific layer.
    """
    # Finish this function
    logger.info(f"Training SAE for layer {layer_idx}")

    train_dataset, val_dataset, epoch_idx_per_step, normalizer = setup_training_datasets(
        cfg, layer_idx, dataset_processing_function=setup_sae_cache
    )

    # Get activation dimension from first sample
    sample_activation = train_dataset[0]
    activation_dim = sample_activation.shape[-1]

    assert activation_dim > 0, f"Invalid activation dimension: {activation_dim}"
    logger.info(f"Activation dimension: {activation_dim}")

    # Create trainer configuration
    trainer_config, run_name = create_sae_trainer_config(
        cfg, layer_idx, activation_dim, device
    )

    # Create data loaders
    train_dataloader = create_training_dataloader(train_dataset, cfg, shuffle=True)
    val_dataloader = create_training_dataloader(val_dataset, cfg, shuffle=False)

    # Calculate max steps if not specified
    max_steps = cfg.diffing.method.training.max_steps
    if max_steps is None:
        max_steps = len(train_dataloader)

    validate_every_n_steps = cfg.diffing.method.training.validate_every_n_steps

    logger.info(f"Training configuration: {trainer_config['wandb_name']}")
    logger.info(
        f"Training steps: {max_steps}, validation every: {validate_every_n_steps}"
    )
    save_dir = (
        f"{cfg.infrastructure.storage.checkpoint_dir}/{trainer_config['wandb_name']}"
    )

    # Train the SAE
    trainSAE(
        data=train_dataloader,
        trainer_config=trainer_config,
        validate_every_n_steps=validate_every_n_steps,
        validation_data=val_dataloader,
        use_wandb=cfg.wandb.enabled,
        wandb_entity=cfg.wandb.entity,
        wandb_project=cfg.wandb.project,
        log_steps=cfg.wandb.log_steps,
        save_dir=save_dir,
        steps=max_steps,
        save_steps=validate_every_n_steps,
        run_wandb_finish=False,
        epoch_idx_per_step=epoch_idx_per_step,
    )

    wandb_link = None
    hf_repo_id = None

    if cfg.wandb.enabled:
        wandb_link = wandb.run.get_url()
        upload_config_to_wandb(cfg)
        wandb.finish()

    if cfg.diffing.method.upload.model:
        hf_repo_id = push_dictionary_model(Path(save_dir) / "model_final.pt")

    # Collect training metrics
    training_metrics = {
        "layer": layer_idx,
        "activation_dim": activation_dim,
        "dictionary_size": trainer_config["dict_size"],
        "training_steps": max_steps,
        "lr": trainer_config["lr"],
        "wandb_link": wandb_link,
        "model_type": "batch-top-k-sae",
        "run_name": trainer_config["wandb_name"],
        "hf_repo_id": hf_repo_id,
        "training_mode": "sae",
        "target": cfg.diffing.method.training.target,
        "k": trainer_config["k"],
    }

    logger.info(f"Successfully trained SAE for layer {layer_idx}")
    return training_metrics
