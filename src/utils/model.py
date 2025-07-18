from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Tuple
import torch
from loguru import logger
from pathlib import Path
import inspect

from .configs import ModelConfig

_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}


def has_thinking(cfg: ModelConfig) -> bool:
    return cfg.model.has_enable_thinking


def load_tokenizer(model_name: str) -> AutoTokenizer:
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    return AutoTokenizer.from_pretrained(model_name)


def load_steering_vector(steering_vector: str, layer: int) -> torch.Tensor:
    try:
        from huggingface_hub import hf_hub_download

        file_name = steering_vector.split("/")[-1]
        repo_name = steering_vector.split("/")[0]
        # Download steering vector from Hugging Face repository
        file_path = hf_hub_download(
            repo_id=f"science-of-finetuning/steering-vecs-{repo_name}",
            filename=f"{file_name}_L{layer}.pt",
            repo_type="model",
        )
        return torch.load(file_path, map_location="cpu")
    except Exception as e:
        logger.error(f"Error loading steering vector: {e}")
        raise e


def add_steering_vector(
    model: AutoModelForCausalLM, layer_idx: int, steering_vector: torch.Tensor
):
    # Get the current layer
    current_layer = model.model.layers[layer_idx].mlp.down_proj

    if hasattr(current_layer, "base_layer"):
        # PEFT wrapper
        is_peft = True
        current_layer = current_layer.base_layer
    else:
        is_peft = False

    # Create new linear layer with bias initialized to steering vector
    new_layer = torch.nn.Linear(
        in_features=current_layer.in_features,
        out_features=current_layer.out_features,
        bias=True,
    ).to(current_layer.weight.device, dtype=current_layer.weight.dtype)

    # Copy the original weights
    new_layer.weight.data = current_layer.weight.data.clone()

    # Initialize bias with steering vector
    assert steering_vector.shape == (
        current_layer.out_features,
    ), f"Steering vector shape {steering_vector.shape} doesn't match output features {current_layer.out_features}"
    new_layer.bias.data = steering_vector.to(
        current_layer.weight.device, dtype=current_layer.weight.dtype
    )

    # Replace the layer
    if is_peft:
        model.model.layers[layer_idx].mlp.down_proj.base_layer = new_layer
    else:
        model.model.layers[layer_idx].mlp.down_proj = new_layer

    logger.info(
        f"Bias initialized with steering vector of shape: {new_layer.bias.shape}"
    )
    return model


def load_model(
    model_name: str,
    dtype: torch.dtype,
    attn_implementation: str,
    adapter_id: str = None,
    steering_vector_name: str = None,
    steering_layer_idx: int = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    key = f"{model_name}_{dtype}_{attn_implementation}_{adapter_id}"
    if steering_vector_name is not None and steering_layer_idx is not None:
        key += f"_{steering_vector_name}_{steering_layer_idx}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], _TOKENIZER_CACHE[key]

    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )

    if adapter_id:
        logger.info(f"Loading adapter: {adapter_id}")
        model.load_adapter(adapter_id)

    tokenizer = load_tokenizer(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if steering_vector_name is not None and steering_layer_idx is not None:
        logger.info(f"Adding steering vector to layer {steering_layer_idx}")
        steering_vector = load_steering_vector(steering_vector_name, steering_layer_idx)
        model = add_steering_vector(model, steering_layer_idx, steering_vector)

    _MODEL_CACHE[key] = model
    _TOKENIZER_CACHE[key] = tokenizer

    return model, tokenizer


def get_ft_model_id(model_cfg: ModelConfig) -> str:
    if model_cfg.adapter_id:
        return model_cfg.adapter_id
    return model_cfg.model_id


def load_model_from_config(
    model_cfg: ModelConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    base_model_id = (
        model_cfg.base_model_id
        if model_cfg.base_model_id is not None
        else model_cfg.model_id
    )
    if base_model_id != model_cfg.model_id:
        adapter_id = model_cfg.model_id
    else:
        adapter_id = None
    return load_model(
        base_model_id,
        model_cfg.dtype,
        model_cfg.attn_implementation,
        adapter_id,
        model_cfg.steering_vector,
        model_cfg.steering_layer,
    )


def load_tokenizer_from_config(
    model_cfg: ModelConfig,
) -> AutoTokenizer:
    return load_tokenizer(model_cfg.model_id)
