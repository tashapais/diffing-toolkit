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
from tqdm import tqdm, trange
import json
from dictionary_learning import CrossCoder, BatchTopKCrossCoder
from dictionary_learning.trainers.crosscoder import (
    CrossCoderTrainer,
    BatchTopKCrossCoderTrainer,
)
import hashlib
from dictionary_learning.trainers import BatchTopKTrainer, BatchTopKSAE
from dictionary_learning.cache import (
    ActivationCache,
    PairedActivationCache,
    RunningStatWelford,
)
from dictionary_learning.training import trainSAE

from ..activations import (
    load_activation_datasets_from_config,
    get_local_shuffled_indices,
    calculate_samples_per_dataset,
)
from ..configs import get_model_configurations, get_dataset_configurations
from ..dictionary.utils import push_dictionary_model
from ..cache import DifferenceCache


def combine_normalizer(
    caches: List[PairedActivationCache],
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    # we want unbiased std for the normalizer
    std = torch.stack([running_stats_1.std(unbiased=False), running_stats_2.std(unbiased=False)], dim=0)
    return mean, std


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


def recompute_diff_normalizer(
    caches: List[PairedActivationCache],
    target: str,
    subsample_size: int,
    layer: int,
    batch_size: int = 4096,
    cache_dir: str = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recompute the normalizer for a dictionary of caches.
    """
    # Compute hash by hashing the configs of all caches
    cache_hash = hashlib.sha256(
        json.dumps(
            [cache.activation_cache_1.config for cache in caches]
            + [cache.activation_cache_2.config for cache in caches]
            + [layer]
        ).encode()
    ).hexdigest()
    if cache_dir is not None:
        cache_dir = Path(cache_dir) / cache_hash
        if (cache_dir / "mean.pt").exists() and (cache_dir / "std.pt").exists():
            logger.info(f"Loading existing normalizer from {cache_dir}")
            mean = torch.load(cache_dir / "mean.pt")
            std = torch.load(cache_dir / "std.pt")
            return mean, std

    logger.info(f"Recomputing normalizer for {len(caches)} caches")
    cache_dir.mkdir(parents=True, exist_ok=True)
    running_stats_1 = None
    running_stats_2 = None
    for cache in caches:
        if running_stats_1 is None:
            running_stats_1 = cache.activation_cache_1.running_stats
            running_stats_2 = cache.activation_cache_2.running_stats
        else:
            running_stats_1.merge(cache.activation_cache_1.running_stats)
            running_stats_2.merge(cache.activation_cache_2.running_stats)

    if target == "difference_bft":
        mean = running_stats_1.mean - running_stats_2.mean
    elif target == "difference_ftb":
        mean = running_stats_2.mean - running_stats_1.mean
    else:
        raise ValueError(f"Invalid Difference target: {target}")

    # Compute std by resampling
    num_resamples = subsample_size
    num_resamples_per_dataset = num_resamples // len(caches)
    running_stats = RunningStatWelford(
        shape=(running_stats_1.mean.shape[0],), device=running_stats_1.mean.device
    )

    for j, cache in enumerate(caches):
        diff_cache = setup_sae_cache(target=target, paired_cache=cache)
        sample_indices = torch.randint(0, len(diff_cache), (num_resamples_per_dataset,))
        # Sort the indices for better cache locality
        sample_indices = sample_indices[torch.argsort(sample_indices)]
        # Process in batches
        for i in trange(
            0,
            len(sample_indices),
            batch_size,
            desc=f"Processing cache {j}/{len(caches)}",
        ):
            running_stats.update(
                torch.stack(
                    [diff_cache[i] for i in sample_indices[i : i + batch_size]], dim=0
                )
            )

    std = running_stats.std(unbiased=False)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(mean, cache_dir / "mean.pt")
        torch.save(std, cache_dir / "std.pt")

    return mean, std


def setup_training_datasets(
    cfg: DictConfig,
    layer: int,
    dataset_processing_function: Callable = None,
    normalizer_function: Callable = None,
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
        normalizer_function: Function to compute the normalizer
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

    if normalizer_function is not None:
        normalize_mean, normalize_std = normalizer_function(list(caches.values()))
    else:
        normalize_mean = None
        normalize_std = None

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
    else:
        epoch_numbers = None
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

    if dataset_processing_function is not None:
        caches_val = {
            dataset_name: dataset_processing_function(caches_val[dataset_name])
            for dataset_name in caches_val
        }

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

    return (
        train_dataset,
        validation_dataset,
        epoch_numbers,
        normalize_mean,
        normalize_std,
    )


def create_training_dataloader(
    dataset: Dataset,
    cfg: DictConfig,
    shuffle: bool = True,
    overwrite_batch_size: int = None,
) -> DataLoader:
    """
    Create DataLoader with configuration from preprocessing and method settings.

    Args:
        dataset: Dataset to create DataLoader for
        cfg: Full configuration
        shuffle: Whether to shuffle the data (disabled for local shuffling)
        overwrite_batch_size: If provided, use this batch size instead of the one in the configuration
    Returns:
        Configured DataLoader
    """
    training_cfg = cfg.diffing.method.training

    return DataLoader(
        dataset,
        batch_size=(
            training_cfg.batch_size
            if overwrite_batch_size is None
            else overwrite_batch_size
        ),
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
    config_dict = OmegaConf.to_yaml(cfg, resolve=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_dict)
        temp_path = f.name

    try:
        # Create and log the artifact
        artifact = wandb.Artifact(
            name="hydra_config",
            type="config",
            description="Full experiment configuration including all hydra configs",
        )
        artifact.add_file(temp_path, name="hydra_config.yaml")
        wandb.log_artifact(artifact)
        logger.debug("Successfully uploaded hydra config artifact to W&B")
    finally:
        # Clean up temporary file
        Path(temp_path).unlink()


### Crosscoder ###
def crosscoder_run_name(
    cfg: DictConfig,
    layer: int,
    base_model_cfg: DictConfig,
    finetuned_model_cfg: DictConfig,
) -> str:
    method_cfg = cfg.diffing.method
    expansion_factor = method_cfg.training.expansion_factor
    lr = method_cfg.training.lr
    mu = method_cfg.training.mu
    k = method_cfg.training.k
    model_type = method_cfg.model.type
    code_normalization = method_cfg.model.code_normalization

    if model_type == "relu":
        run_name = (
            f"{base_model_cfg.name}-{cfg.organism.name}-L{layer}-mu{mu:.1e}-lr{lr:.0e}-x{expansion_factor}"
            + "-local-shuffling"
            + f"-{code_normalization.capitalize()}Loss"
        )
    elif model_type == "batch-top-k":
        run_name = (
            f"{base_model_cfg.name}-{cfg.organism.name}-L{layer}-k{k}-lr{lr:.0e}-x{expansion_factor}"
            + "-local-shuffling"
            + f"-{code_normalization.capitalize()}"
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    if method_cfg.datasets.normalization.enabled:
        run_name += f"-tv{method_cfg.datasets.normalization.target_variance:.1f}"

    return run_name


def sae_difference_run_name(
    cfg: DictConfig,
    layer: int,
    base_model_cfg: DictConfig,
    finetuned_model_cfg: DictConfig,
) -> str:
    method_cfg = cfg.diffing.method
    expansion_factor = method_cfg.training.expansion_factor
    lr = method_cfg.training.lr
    k = method_cfg.training.k

    target = method_cfg.training.target
    target_short = target.split("_")[1]  # "bft" or "ftb"

    run_name = (
        f"SAE-difference_{target_short}-{base_model_cfg.name}-{cfg.organism.name}-L{layer}-k{k}-x{expansion_factor}-lr{lr:.0e}"
        + ("-local-shuffling" if method_cfg.training.local_shuffling else "")
    )
    if not method_cfg.datasets.normalization.enabled:
        run_name += "-no-normalization"

    if method_cfg.datasets.normalization.enabled:
        run_name += f"-tv{method_cfg.datasets.normalization.target_variance:.1f}"

    return run_name


def create_crosscoder_trainer_config(
    cfg: DictConfig,
    layer: int,
    activation_dim: int,
    device: str,
    normalize_mean: torch.Tensor,
    normalize_std: torch.Tensor,
    run_name: str,
) -> Dict[str, Any]:
    """
    Create trainer configuration from method settings.

    Args:
        cfg: Full configuration
        layer: Layer index being trained
        activation_dim: Dimension of input activations
        device: Training device
        normalize_mean: Mean of the activations
        normalize_std: Std of the activations
        run_name: Name of the run

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
    warmup_steps = method_cfg.optimization.warmup_steps
    dictionary_size = expansion_factor * activation_dim

    # Common configuration
    common_config = {
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": lr,
        "device": device,
        "warmup_steps": warmup_steps,
        "layer": layer,
        "lm_name": f"{finetuned_model_cfg.model_id}-{base_model_cfg.model_id}",
        "wandb_name": run_name,
        "dict_class_kwargs": {
            "same_init_for_all_layers": same_init_for_all_layers,
            "norm_init_scale": norm_init_scale,
            "init_with_transpose": init_with_transpose,
            "encoder_layers": None,  # Could be made configurable
            "code_normalization": code_normalization,
            "code_normalization_alpha_sae": 1.0,
            "code_normalization_alpha_cc": 0.1,
        },
        "activation_mean": normalize_mean,
        "activation_std": normalize_std,
        "target_variance": method_cfg.datasets.normalization.target_variance,
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

    return trainer_config


def train_crosscoder_for_layer(
    cfg: DictConfig,
    layer_idx: int,
    device: str,
    run_name: str,
) -> Dict[str, Any]:
    """
    Train crosscoder for a specific layer (original implementation).
    """
    logger.info(f"Training crosscoder for layer {layer_idx}")

    # Setup training datasets
    train_dataset, val_dataset, epoch_idx_per_step, normalize_mean, normalize_std = (
        setup_training_datasets(
            cfg,
            layer_idx,
            normalizer_function=(
                combine_normalizer
                if cfg.diffing.method.datasets.normalization.enabled
                else None
            ),
        )
    )

    # Get activation dimension from first sample
    sample_activation = train_dataset[0]
    activation_dim = sample_activation.shape[-1]

    assert activation_dim > 0, f"Invalid activation dimension: {activation_dim}"
    logger.info(f"Activation dimension: {activation_dim}")

    # Create trainer configuration
    trainer_config = create_crosscoder_trainer_config(
        cfg, layer_idx, activation_dim, device, normalize_mean, normalize_std, run_name
    )

    # Create data loaders
    train_dataloader = create_training_dataloader(train_dataset, cfg, shuffle=True)
    val_dataloader = create_training_dataloader(
        val_dataset,
        cfg,
        shuffle=False,
        overwrite_batch_size=cfg.diffing.method.training.batch_size * 4,
    )

    # Calculate max steps if not specified
    max_steps = cfg.diffing.method.training.max_steps
    if max_steps is None:
        max_steps = len(train_dataloader)
        trainer_config["steps"] = max_steps
    else:
        trainer_config["steps"] = max_steps

    validate_every_n_steps = cfg.diffing.method.training.validate_every_n_steps

    logger.info(f"Training configuration: {trainer_config['wandb_name']}")
    logger.info(
        f"Training steps: {max_steps}, validation every: {validate_every_n_steps}"
    )
    checkpoint_dir = (
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
        wandb_project="Diffing-Game-Crosscoder",
        wandb_group=cfg.organism.name,
        log_steps=50,
        save_dir=checkpoint_dir,
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
        hf_repo_id = push_dictionary_model(Path(checkpoint_dir) / "model_final.pt")

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
    return training_metrics, Path(checkpoint_dir) / "model_final.pt"


### SAE Difference ###


def create_sae_difference_trainer_config(
    cfg: DictConfig,
    layer: int,
    activation_dim: int,
    max_steps: int,
    device: str,
    target: str,
    normalize_mean: torch.Tensor,
    normalize_std: torch.Tensor,
    run_name: str,
) -> Dict[str, Any]:
    """
    Create SAE trainer configuration for difference training.

    Args:
        cfg: Full configuration
        layer: Layer index being trained
        activation_dim: Dimension of input activations
        device: Training device
        target: Training target (difference_bft or difference_ftb)
        normalize_mean: Mean of the activations
        normalize_std: Std of the activations
        run_name: Name of the run
    Returns:
        Tuple of (trainer configuration dictionary, run name)
    """
    method_cfg = cfg.diffing.method
    base_model_cfg, finetuned_model_cfg = get_model_configurations(cfg)

    # Extract SAE-specific parameters
    expansion_factor = method_cfg.training.expansion_factor
    lr = method_cfg.training.lr
    k = method_cfg.training.k

    # Extract optimization parameters
    warmup_steps = method_cfg.optimization.warmup_steps

    dictionary_size = expansion_factor * activation_dim

    # SAE trainer configuration
    trainer_config = {
        "trainer": BatchTopKTrainer,
        "dict_class": BatchTopKSAE,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": lr,
        "steps": max_steps,
        "device": device,
        "warmup_steps": warmup_steps,
        "layer": layer,
        "lm_name": f"{finetuned_model_cfg.model_id}-{base_model_cfg.model_id}",
        "wandb_name": run_name,
        "k": k,
        "activation_mean": normalize_mean,
        "activation_std": normalize_std,
        "target_variance": method_cfg.datasets.normalization.target_variance,
    }

    return trainer_config


def train_sae_difference_for_layer(
    cfg: DictConfig,
    layer_idx: int,
    device: str,
    run_name: str,
) -> Dict[str, Any]:
    """
    Train SAE on differences for a specific layer.

    This function:
    1. Loads paired activation caches
    2. Computes normalization statistics for differences
    3. Sets up difference caches for training
    4. Trains BatchTopK SAE on normalized differences
    5. Saves model and normalization statistics

    Args:
        cfg: Full configuration including SAE difference method settings
        layer_idx: Layer index to train on
        device: Device for training

    Returns:
        Dictionary containing training metrics and model information
    """
    logger.info(f"Training SAE on differences for layer {layer_idx}")

    target = cfg.diffing.method.training.target
    assert target in [
        "difference_bft",
        "difference_ftb",
    ], f"Invalid target for SAE difference: {target}"

    # Load paired activation caches for normalization computation
    base_model_cfg, finetuned_model_cfg = get_model_configurations(cfg)
    dataset_cfgs = get_dataset_configurations(
        cfg,
        use_chat_dataset=cfg.diffing.method.datasets.use_chat_dataset,
        use_pretraining_dataset=cfg.diffing.method.datasets.use_pretraining_dataset,
        use_training_dataset=cfg.diffing.method.datasets.use_training_dataset,
    )

    caches = load_activation_datasets_from_config(
        cfg=cfg,
        ds_cfgs=dataset_cfgs,
        base_model_cfg=base_model_cfg,
        finetuned_model_cfg=finetuned_model_cfg,
        layers=[layer_idx],
        split="train",
    )

    # Collapse layers
    caches = {dataset_name: caches[dataset_name][layer_idx] for dataset_name in caches}

    # Setup training datasets with difference cache processing
    train_dataset, val_dataset, epoch_idx_per_step, normalize_mean, normalize_std = (
        setup_training_datasets(
            cfg,
            layer_idx,
            dataset_processing_function=lambda x: setup_sae_cache(
                target=target, paired_cache=x
            ),
            normalizer_function=(lambda x: (
                recompute_diff_normalizer(
                    x,
                    target=target,
                    subsample_size=cfg.diffing.method.datasets.normalization.subsample_size,
                    batch_size=cfg.diffing.method.datasets.normalization.batch_size,
                    cache_dir=cfg.diffing.method.datasets.normalization.cache_dir,
                    layer=layer_idx,
                )))
                if cfg.diffing.method.datasets.normalization.enabled
                else None
            ,
        )
    )

    # Get activation dimension from first sample
    sample_activation = train_dataset[0]
    activation_dim = sample_activation.shape[-1]

    assert activation_dim > 0, f"Invalid activation dimension: {activation_dim}"
    logger.info(f"Activation dimension: {activation_dim}")

    # Create data loaders
    train_dataloader = create_training_dataloader(train_dataset, cfg, shuffle=True)
    val_dataloader = create_training_dataloader(
        val_dataset,
        cfg,
        shuffle=False,
        overwrite_batch_size=cfg.diffing.method.training.batch_size * 4,
    )

    # Calculate max steps if not specified
    max_steps = cfg.diffing.method.training.max_steps
    if max_steps is None:
        max_steps = len(train_dataloader)

    # Create trainer configuration for SAE difference
    trainer_config = create_sae_difference_trainer_config(
        cfg,
        layer_idx,
        activation_dim,
        max_steps,
        device,
        target,
        normalize_mean,
        normalize_std,
        run_name,
    )

    validate_every_n_steps = cfg.diffing.method.training.validate_every_n_steps

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
        wandb_project="Diffing-Game-DiffSAE",
        wandb_group=cfg.organism.name,
        log_steps=50,
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
    else:
        logger.warning(
            f"Not uploading model to Hugging Face because upload.model is False, which can break downstream code. Only use this if you know what you are doing."
        )
        raise ValueError(
            "Upload model is False, only use for debugging (because downstream code will load from hf only)."
        )

    # Collect training metrics
    training_metrics = {
        "layer": layer_idx,
        "activation_dim": activation_dim,
        "dictionary_size": trainer_config["dict_size"],
        "training_steps": max_steps,
        "lr": trainer_config["lr"],
        "wandb_link": wandb_link,
        "model_type": "batch-top-k-sae-difference",
        "run_name": trainer_config["wandb_name"],
        "hf_repo_id": hf_repo_id,
        "training_mode": "sae_difference",
        "target": target,
        "k": trainer_config["k"],
        "wandb_name": run_name,
    }

    logger.info(f"Successfully trained SAE difference for layer {layer_idx}")
    return training_metrics, Path(save_dir) / "model_final.pt"
