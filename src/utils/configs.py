from dataclasses import dataclass
from typing import Dict, Tuple, List
from omegaconf import DictConfig
from loguru import logger
from hydra import initialize, compose
from pathlib import Path

HF_NAME = "science-of-finetuning"


@dataclass
class ModelConfig:
    """Configuration for a model (base or finetuned)."""

    name: str
    model_id: str
    attn_implementation: str = "eager"
    ignore_first_n_tokens_per_sample: int = 0
    token_level_replacement: dict = None
    text_column: str = "text"
    base_model_id: str = None
    dtype: str = "bfloat16"


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    id: str
    split: str
    is_chat: bool
    text_column: str = None
    messages_column: str = "messages"
    description: str = ""


def create_model_config(
    model_cfg: DictConfig, name_override: str = None
) -> ModelConfig:
    """Create a ModelConfig from configuration object."""
    return ModelConfig(
        name=name_override or model_cfg.name,
        model_id=model_cfg.model_id,
        attn_implementation=model_cfg.get("attn_implementation", "eager"),
        ignore_first_n_tokens_per_sample=model_cfg.get(
            "ignore_first_n_tokens_per_sample", 0
        ),
        token_level_replacement=model_cfg.get("token_level_replacement", None),
        text_column=model_cfg.get("text_column", "text"),
        base_model_id=model_cfg.get("base_model_id", None),
        dtype=model_cfg.get("dtype", "bfloat16"),
    )


def create_dataset_config(
    dataset_cfg: DictConfig, name: str, split: str
) -> DatasetConfig:
    """Create a DatasetConfig from configuration object for a specific split."""
    return DatasetConfig(
        name=name,
        id=dataset_cfg.id,
        split=split,
        is_chat=dataset_cfg.is_chat,
        text_column=dataset_cfg.get("text_column", None),
        messages_column=dataset_cfg.get("messages_column", "messages"),
        description=dataset_cfg.get("description", ""),
    )


def get_model_configurations(cfg: DictConfig) -> Tuple[ModelConfig, ModelConfig]:
    """Extract and prepare base and finetuned model configurations."""
    # Ensure finetuned model is resolved before accessing it
    cfg = ensure_finetuned_model_resolved(cfg)
    
    # Base model configuration
    base_model_cfg = create_model_config(cfg.model)

    # Finetuned model configuration - inherit from base model and override
    organism_cfg = cfg.organism
    finetuned_cfg = organism_cfg.finetuned_model

    # Create finetuned model config with inheritance
    finetuned_model_cfg = ModelConfig(
        name=finetuned_cfg.name,
        model_id=finetuned_cfg.model_id,
        base_model_id=finetuned_cfg.get("base_model_id", None),
        attn_implementation=finetuned_cfg.get(
            "attn_implementation", base_model_cfg.attn_implementation
        ),
        ignore_first_n_tokens_per_sample=finetuned_cfg.get(
            "ignore_first_n_tokens_per_sample",
            base_model_cfg.ignore_first_n_tokens_per_sample,
        ),
        token_level_replacement=finetuned_cfg.get(
            "token_level_replacement", base_model_cfg.token_level_replacement
        ),
        text_column=finetuned_cfg.get("text_column", base_model_cfg.text_column),
        dtype=finetuned_cfg.get("dtype", base_model_cfg.dtype),
    )

    # Apply organism-specific overrides
    organism_overrides = organism_cfg.get("preprocessing_overrides", {})
    if organism_overrides and "ignore_first_n_tokens_per_sample" in organism_overrides:
        finetuned_model_cfg.ignore_first_n_tokens_per_sample = organism_overrides[
            "ignore_first_n_tokens_per_sample"
        ]
        # Also apply to base model for consistency in organism context
        base_model_cfg.ignore_first_n_tokens_per_sample = organism_overrides[
            "ignore_first_n_tokens_per_sample"
        ]

    return base_model_cfg, finetuned_model_cfg


def get_dataset_configurations(
    cfg: DictConfig,
    use_chat_dataset: bool = True,
    use_pretraining_dataset: bool = True,
    use_training_dataset: bool = True,
) -> List[DatasetConfig]:
    """Extract and prepare all dataset configurations."""
    datasets = []

    # General datasets (used for all organisms)
    if hasattr(cfg, "chat_dataset") and use_chat_dataset:
        # Create one DatasetConfig for each split
        for split in cfg.chat_dataset.splits:
            datasets.append(
                create_dataset_config(
                    cfg.chat_dataset, cfg.chat_dataset.id.split("/")[-1], split
                )
            )

    if hasattr(cfg, "pretraining_dataset") and use_pretraining_dataset:
        # Create one DatasetConfig for each split
        for split in cfg.pretraining_dataset.splits:
            datasets.append(
                create_dataset_config(
                    cfg.pretraining_dataset,
                    cfg.pretraining_dataset.id.split("/")[-1],
                    split,
                )
            )

    # Organism-specific datasets
    organism_cfg = cfg.organism

    # Training dataset from finetuned model config (if present)
    if hasattr(organism_cfg, "training_dataset") and use_training_dataset:
        # Create one DatasetConfig for each split
        for split in organism_cfg.training_dataset.splits:
            datasets.append(
                create_dataset_config(
                    organism_cfg.training_dataset,
                    organism_cfg.training_dataset.id.split("/")[-1],
                    split,
                )
            )

    return datasets


def resolve_finetuned_model(cfg: DictConfig) -> DictConfig:
    """
    Resolve the finetuned model from the registry after all overrides are applied.
    
    This function handles the delayed resolution of organism.finetuned_model that
    cannot be done via simple OmegaConf interpolation due to nested variable resolution.
    
    Args:
        cfg: The configuration object after all Hydra overrides are applied
        
    Returns:
        Updated configuration with organism.finetuned_model properly resolved
        
    Raises:
        ValueError: If the model/organism combination is not found in the registry
    """
    try:
        # Extract the model and organism names from the resolved config
        model_name = cfg.model.name
        organism_name = cfg.organism.name
        
        # Look up the finetuned model from the registry
        finetuned_model = cfg.organism_model_registry.mappings[model_name][organism_name]
        
        # Set the resolved finetuned model in the organism config
        cfg.organism.finetuned_model = finetuned_model
        
        logger.debug(f"Resolved finetuned model: {model_name}.{organism_name} -> {finetuned_model.model_id}")
        
    except KeyError as e:
        available_models = list(cfg.organism_model_registry.mappings.keys()) if hasattr(cfg, 'organism_model_registry') else []
        available_organisms = []
        if hasattr(cfg, 'organism_model_registry') and model_name in cfg.organism_model_registry.mappings:
            available_organisms = list(cfg.organism_model_registry.mappings[model_name].keys())
        
        raise ValueError(
            f"No finetuned model found for model='{model_name}' and organism='{organism_name}'. "
            f"Available models: {available_models}. "
            f"Available organisms for {model_name}: {available_organisms}. "
            f"Error: {e}"
        )
    except AttributeError as e:
        raise ValueError(
            f"Configuration missing required fields for model resolution. "
            f"Expected cfg.model.name, cfg.organism.name, and cfg.organism_model_registry. "
            f"Error: {e}"
        )
    
    return cfg


def load_hydra_config(config_path: str, *overrides) -> DictConfig:
    """
    Load a Hydra config from a file and resolve the finetuned model.
    
    This function loads the configuration, applies all overrides, and then
    resolves the organism.finetuned_model from the registry using the final
    model and organism names.
    
    Args:
        config_path: Path to the config file
        *overrides: Hydra override strings (e.g., "model=gemma3_1B", "organism=kansas_abortion")
    
    Returns:
        Fully resolved configuration with organism.finetuned_model set
    """
    with initialize(config_path=str(Path(config_path).parent), version_base=None):
        cfg = compose(config_name=Path(config_path).stem, overrides=overrides)
        
        # Resolve the finetuned model after all overrides are applied
        cfg = resolve_finetuned_model(cfg)
        
    return cfg


def ensure_finetuned_model_resolved(cfg: DictConfig) -> DictConfig:
    """
    Ensure that the finetuned model is resolved in the configuration.
    
    This is a safety function that can be called to ensure the organism.finetuned_model
    is properly set, regardless of how the configuration was loaded.
    
    Args:
        cfg: Configuration that may or may not have organism.finetuned_model resolved
        
    Returns:
        Configuration with organism.finetuned_model guaranteed to be resolved
    """
    # Check if finetuned_model is already resolved
    if hasattr(cfg.organism, 'finetuned_model') and cfg.organism.finetuned_model is not None:
        # Already resolved, just log and return
        logger.debug("Finetuned model already resolved")
        return cfg
    
    # Not resolved, resolve it now
    logger.debug("Finetuned model not found, resolving from registry")
    return resolve_finetuned_model(cfg)
