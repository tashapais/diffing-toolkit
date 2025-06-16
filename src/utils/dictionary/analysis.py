"""
Analysis pipeline wrapper for crosscoder evaluation and analysis.

This module provides a wrapper around the comprehensive analysis pipeline from
science-of-finetuning, including evaluation notebooks, scaler computation,
latent statistics, and KL divergence experiments.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import DictConfig
from loguru import logger
from transformers import AutoTokenizer
import torch as th
import pandas as pd
from torch.nn.functional import cosine_similarity
from tqdm.auto import trange

from src.utils.dictionary import load_dictionary_model


def build_push_crosscoder_latent_df(
    dictionary_name: str,
    base_layer: int = 0,
    ft_layer: int = 1,
) -> pd.DataFrame:
    crosscoder = load_dictionary_model(dictionary_name)
    latent_df = {k: {} for k in range(crosscoder.decoder.weight.shape[0])}

    # Norms
    norms = crosscoder.decoder.weight.norm(dim=-1)
    norm_diffs = (
        (norms[base_layer] - norms[ft_layer]) / norms.max(dim=0).values + 1
    ) / 2
    norm_diffs = norm_diffs.cpu()
    for f_idx, (base_norm, instruct_norm, norm_diff) in enumerate(
        zip(norms[base_layer], norms[ft_layer], norm_diffs)
    ):
        latent_df[f_idx]["dec_base_norm"] = base_norm.item()
        latent_df[f_idx]["dec_ft_norm"] = instruct_norm.item()
        latent_df[f_idx]["dec_norm_diff"] = norm_diff.item()

    enc_norms = crosscoder.encoder.weight.norm(dim=1)
    enc_norm_diffs = (
        (enc_norms[base_layer] - enc_norms[ft_layer]) / enc_norms.max(dim=0).values + 1
    ) / 2
    enc_norm_diffs = enc_norm_diffs.cpu()

    for f_idx, (base_norm, instruct_norm, norm_diff) in enumerate(
        zip(enc_norms[base_layer], enc_norms[ft_layer], enc_norm_diffs)
    ):
        latent_df[f_idx]["enc_base_norm"] = base_norm.item()
        latent_df[f_idx]["enc_ft_norm"] = instruct_norm.item()
        latent_df[f_idx]["enc_norm_diff"] = norm_diff.item()

    decoder_cos_sims = cosine_similarity(
        crosscoder.decoder.weight[base_layer],
        crosscoder.decoder.weight[ft_layer],
        dim=1,
    )
    for f_idx, cos_sim in enumerate(decoder_cos_sims):
        latent_df[f_idx]["dec_cos_sim"] = cos_sim.item()

    # Encoder cos sims
    enc_cos_sims = cosine_similarity(
        crosscoder.encoder.weight[base_layer],
        crosscoder.encoder.weight[ft_layer],
        dim=1,
    )
    for f_idx, cos_sim in enumerate(enc_cos_sims):
        latent_df[f_idx]["enc_cos_sim"] = cos_sim.item()

    # Create masks for each category
    # Decoder
    # save Chat only and base only feature index
    treshold = 0.1
    only_it_feature_indices = th.nonzero(norm_diffs < treshold, as_tuple=True)[0]
    only_base_feature_indices = th.nonzero(norm_diffs > 1 - treshold, as_tuple=True)[0]
    shared_feature_indices = th.nonzero(
        (norm_diffs - 0.5).abs() < treshold, as_tuple=True
    )[0]
    is_other_feature = th.ones_like(norm_diffs, dtype=bool)
    is_other_feature[only_it_feature_indices] = False
    is_other_feature[only_base_feature_indices] = False
    is_other_feature[shared_feature_indices] = False

    for f_idx in only_it_feature_indices.tolist():
        latent_df[f_idx]["tag"] = "ft_only"
    for f_idx in only_base_feature_indices.tolist():
        latent_df[f_idx]["tag"] = "base_only"
    for f_idx in shared_feature_indices.tolist():
        latent_df[f_idx]["tag"] = "shared"
    for f_idx in is_other_feature.nonzero(as_tuple=True)[0].tolist():
        latent_df[f_idx]["tag"] = "other"

    latent_df = pd.DataFrame(latent_df)
    return latent_df
