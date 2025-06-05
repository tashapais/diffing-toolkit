from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Tuple
import torch as th
from loguru import logger
from pathlib import Path

from .configs import ModelConfig

_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}
def load_model(
    model_name: str, dtype: th.dtype, attn_implementation: str, adapter_id: str = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name], _TOKENIZER_CACHE[model_name]
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    
    if adapter_id:
        model = PeftModel.from_pretrained(model, adapter_id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _MODEL_CACHE[model_name] = model
    _TOKENIZER_CACHE[model_name] = tokenizer

    return model, tokenizer

def get_ft_model_id(model_cfg: ModelConfig) -> str:
    if model_cfg.adapter_id:
        return model_cfg.adapter_id
    return model_cfg.model_id

def load_model_from_config(
    model_cfg: ModelConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    base_model_id = model_cfg.base_model_id if model_cfg.base_model_id is not None else model_cfg.model_id
    if base_model_id != model_cfg.model_id:
        adapter_id = model_cfg.model_id
    else:
        adapter_id = None
    return load_model(
        base_model_id, model_cfg.dtype, model_cfg.attn_implementation, adapter_id
    )

