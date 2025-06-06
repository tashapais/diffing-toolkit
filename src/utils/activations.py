from transformers import AutoConfig
from typing import Union, List
from transformers import PretrainedConfig
from loguru import logger
from dictionary_learning.cache import PairedActivationCache
from pathlib import Path

def get_layer_indices(model: Union[str, object], layers: List[float]) -> List[int]:
    """
    Get the indices of the layers to collect activations from.
    """
    if isinstance(model, str):
        config: PretrainedConfig = AutoConfig.from_pretrained(model)
        try:
            num_layers: int = config.num_hidden_layers
        except:
            logger.warning(f"Using num_hidden_layers from Gemma3TextConfig for {model}")
            num_layers: int = config.text_config.num_hidden_layers
    else:
        num_layers: int = len(model.model.layers)
    return [int(layer * num_layers) for layer in layers]



def load_activation_dataset(
    activation_store_dir: Path,
    split: str,
    dataset_name: str,
    base_model: str = "gemma-2-2b",
    finetuned_model: str = "gemma-2-2b-it",
    layer: int = 13,
):
    """
    Load the saved activations of the base and finetuned models for a given layer and dataset.

    Args:
        activation_store_dir: The directory where the activations are stored
        split: The split to load (e.g., 'train', 'validation', 'test')
        dataset_name: The name of the dataset to load
        base_model: The base model identifier (default: "gemma-2-2b")
        finetuned_model: The finetuned model identifier (default: "gemma-2-2b-it")
        layer: The layer number to load activations fråom (default: 13)

    Returns:
        PairedActivationCache: A cache containing paired activations from both models
    """
    # Load validation datase
    activation_store_dir = Path(activation_store_dir)
    base_model_dir = activation_store_dir / base_model
    instruct_model_dir = activation_store_dir / finetuned_model


    submodule_name = f"layer_{layer}_out"

    # Load validation caches
    base_model_cache = base_model_dir / dataset_name / split
    finetuned_model_cache = instruct_model_dir / dataset_name / split


    cache = PairedActivationCache(
        base_model_cache, finetuned_model_cache, submodule_name
    )

    return cache


def load_activation_datasets(
    activation_store_dir: Path,
    split: str,
    dataset_names: list[str],
    base_model: str = "gemma-2-2b",
    finetuned_model: str = "gemma-2-2b-it",
    layers: list[int] = [13],
):
    """
    Load the saved activations of the base and instruct models for multiple datasets and layers.

    Args:
        activation_store_dir: The directory where the activations are stored
        split: The split to load
        dataset_names: List of dataset names to load
        base_model: The base model to load
        finetuned_model: The finetuned model to load
        layers: List of layers to load

    Returns:
        A dict mapping dataset_name -> {layer: PairedActivationCache, ...}
    """
    result = {}
    
    for dataset_name in dataset_names:
        result[dataset_name] = {}
        
        for layer in layers:
            cache = load_activation_dataset(
                activation_store_dir=activation_store_dir,
                split=split,
                dataset_name=dataset_name,
                base_model=base_model,
                finetuned_model=finetuned_model,
                layer=layer,
            )
            
            result[dataset_name][layer] = cache
    
    return result

