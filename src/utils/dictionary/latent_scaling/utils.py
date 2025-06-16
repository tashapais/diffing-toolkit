import torch as th
from pathlib import Path
import numpy as np
from warnings import warn
from dictionary_learning import CrossCoder
from typing import Literal


def remove_latents(
    activation: th.Tensor, latent_activations: th.Tensor, latent_vectors: th.Tensor
) -> th.Tensor:
    # activation: N x dim_model
    # latent_vectors: num_latent_vectors x dim_model
    # latent_activations: N x num_latent_vectors
    # return: num_latent_vectors x N x dim_model
    num_latent_vectors = latent_vectors.shape[0]
    dim_model = latent_vectors.shape[1]
    N = activation.shape[0]

    assert latent_activations.shape == (N, num_latent_vectors)
    assert activation.shape == (N, dim_model)
    assert latent_vectors.shape == (num_latent_vectors, dim_model)

    # stack the activations num_latent_vectors times -> (num_latent_vectors, N, dim_model)
    activation_stacked = activation.unsqueeze(0).repeat(num_latent_vectors, 1, 1)
    assert activation_stacked.shape == (num_latent_vectors, N, dim_model)

    # remove the latents from the activation
    latent_vectors_reshaped = latent_vectors.unsqueeze(1).repeat(1, N, 1)
    assert latent_vectors_reshaped.shape == (num_latent_vectors, N, dim_model)
    # scale by latent_activations
    latent_activations_reshaped = latent_activations.T.unsqueeze(
        -1
    )  # (num_latent_vectors, N, 1)
    assert latent_activations_reshaped.shape == (num_latent_vectors, N, 1)

    # remove the latents
    activation_stacked = activation_stacked - (
        latent_vectors_reshaped * latent_activations_reshaped
    )
    assert activation_stacked.shape == (num_latent_vectors, N, dim_model)
    return activation_stacked


def identity_fn(x: th.Tensor) -> th.Tensor:
    return x


def load_base_activation(batch, **kwargs):
    return batch[:, 0, :]


def load_ft_activation(batch, **kwargs):
    return batch[:, 1, :]


def load_difference_activation(batch, sae_model: Literal["base", "ft"], **kwargs):
    """Load activation difference (ft - base) or (base - ft) from difference cache"""
    if sae_model == "ft":
        return load_ft_activation(batch) - load_base_activation(batch)
    else:
        return load_base_activation(batch) - load_ft_activation(batch)


def load_base_activation_no_bias(batch, crosscoder: CrossCoder, **kwargs):
    return batch[:, 0, :] - crosscoder.decoder.bias[0, :]


def load_ft_activation_no_bias(batch, crosscoder: CrossCoder, **kwargs):
    return batch[:, 1, :] - crosscoder.decoder.bias[1, :]


def load_base_error(
    batch,
    crosscoder: CrossCoder,
    latent_activations: th.Tensor,
    latent_indices: th.Tensor,
    base_decoder: th.Tensor,
    **kwargs,
):
    reconstruction = crosscoder.decode(latent_activations)
    return batch[:, 0, :] - remove_latents(
        reconstruction[:, 0, :],
        latent_activations[:, latent_indices],
        base_decoder[latent_indices],
    )


def load_ft_error(
    batch,
    crosscoder: CrossCoder,
    latent_activations: th.Tensor,
    latent_indices: th.Tensor,
    latent_vectors: th.Tensor,
    **kwargs,
):
    reconstruction = crosscoder.decode(latent_activations)
    return batch[:, 1, :] - remove_latents(
        reconstruction[:, 1, :], latent_activations[:, latent_indices], latent_vectors
    )


def load_base_reconstruction(
    batch,
    crosscoder: CrossCoder,
    latent_activations: th.Tensor,
    latent_indices: th.Tensor,
    latent_vectors: th.Tensor,
    **kwargs,
):
    reconstruction = crosscoder.decode(latent_activations)
    return reconstruction[:, 0, :]


def load_ft_reconstruction(
    batch,
    crosscoder: CrossCoder,
    latent_activations: th.Tensor,
    latent_indices: th.Tensor,
    latent_vectors: th.Tensor,
    **kwargs,
):
    reconstruction = crosscoder.decode(latent_activations)
    return reconstruction[:, 1, :]


def load_betas(
    betas_dir_path: Path,
    num_samples: int = 50_000_000,
    computation: str = "base_error",
    n_offset: int = 0,
    suffix: str = "",
):
    betas_filename = f"betas_{computation}_N{num_samples}_n_offset{n_offset}{suffix}.pt"

    if not (betas_dir_path / betas_filename).exists():
        raise FileNotFoundError(
            f"Betas file not found: {betas_dir_path / betas_filename}."
        )

    betas = th.load(betas_dir_path / betas_filename, weights_only=True).cpu()
    return betas
