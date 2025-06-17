import torch
from tqdm.auto import trange
from pathlib import Path
from transformers import AutoTokenizer
from omegaconf import DictConfig
from typing import List
from loguru import logger
from huggingface_hub import repo_exists, upload_file, hf_api
import sqlite3
import numpy as np
from tqdm import tqdm
import random
import wandb
from collections import defaultdict
import time
import gc

from dictionary_learning.cache import ActivationCache

from src.utils.cache import LatentActivationCache, SampleCache, DifferenceCache
from src.utils.dictionary import load_dictionary_model
from src.utils.configs import get_model_configurations, get_dataset_configurations
from src.utils.activations import load_activation_datasets_from_config
from src.utils.model import load_tokenizer_from_config
from src.utils.configs import HF_NAME
from src.utils.dictionary.utils import load_latent_df, push_latent_df
from src.utils.dictionary.training import setup_sae_cache

@torch.no_grad()
def get_positive_activations(sample_cache: SampleCache, cc, latent_ids):
    """
    Extract positive activations and their indices from sequences using SampleCache.
    Also compute the maximum activation for each latent feature.

    Args:
        sample_cache: SampleCache containing sequences and activations
        cc: Object with get_activations method
        latent_ids: Tensor of latent indices to extract

    Returns:
        Tuple of:
        - activations tensor: positive activation values
        - indices tensor: in (seq_idx, seq_pos, feature_pos) format
        - max_activations: maximum activation value for each latent feature
    """
    out_activations = []
    out_ids = []
    seq_ranges = [0]        

    # Initialize tensors to track max activations for each latent
    max_activations = torch.zeros(len(latent_ids), device="cpu")

    for seq_idx in trange(len(sample_cache)):
        sample_tokens, sample_activations = sample_cache[seq_idx]
        # sample_activations should be (seq_len, activation_dim)
        assert sample_activations.shape[0] == len(
            sample_tokens
        ), f"Expected {len(sample_tokens)} activations, got {sample_activations.shape[0]}"
        feature_activations = cc.get_activations(sample_activations.cuda().to(cc.dtype))
        feature_activations = feature_activations[:, latent_ids].cpu()
        assert feature_activations.shape == (
            len(sample_activations),
            len(latent_ids),
        ), f"Feature activations shape: {feature_activations.shape}, expected: {(len(sample_activations), len(latent_ids))}"

        # Track maximum activations
        # For each latent feature, find the max activation in this sequence
        seq_max_values, seq_max_positions = feature_activations.max(dim=0)

        # Update global maximums where this sequence has a higher value
        update_mask = seq_max_values > max_activations
        max_activations[update_mask] = seq_max_values[update_mask]

        # Get indices where feature activations are positive
        pos_mask = feature_activations > 0
        pos_indices = torch.nonzero(pos_mask, as_tuple=True)

        # Get the positive activation values
        pos_activations = feature_activations[pos_mask]

        # Create sequence indices tensor matching size of positive indices
        seq_idx_tensor = torch.full_like(pos_indices[0], seq_idx)

        # Stack indices into (seq_idx, seq_pos, feature_pos) format
        pos_ids = torch.stack([seq_idx_tensor, pos_indices[0], pos_indices[1]], dim=1)

        out_activations.append(pos_activations)
        out_ids.append(pos_ids)
        seq_ranges.append(seq_ranges[-1] + len(pos_ids))

    out_activations = torch.cat(out_activations).cpu()
    out_ids = torch.cat(out_ids).cpu()
    return out_activations, out_ids, seq_ranges, max_activations

def add_get_activations_sae(sae):
    """
    Add get_activations method to SAE model.

    Args:
        sae: The SAE model
    """
    def get_activation(x: torch.Tensor, select_features=None, **kwargs):
        # For difference SAEs, x should be the difference already computed by DifferenceCache
        # x shape: (batch_size, activation_dim)
        assert x.ndim == 2, f"Expected 2D tensor for difference SAE, got {x.ndim}D"
        f = sae.encode(x)
        if select_features is not None:
            f = f[:, select_features]
        return f

    sae.get_activations = get_activation
    return sae

def collect_dictionary_activations(
    dictionary_model_name: str,
    activation_caches: list[ActivationCache] | list[DifferenceCache],
    tokenizer: AutoTokenizer,
    dataset_names: list[str] | None = None,
    latent_ids: torch.Tensor | None = None,
    out_dir: Path = Path("latent_activations/"),
    upload_to_hub: bool = False,
    load_from_disk: bool = False,
    is_sae: bool = False,
    is_difference_sae: bool = False,
    difference_target: str = None,
    max_num_samples: int = 10000,
) -> None:
    """
    Compute and save latent activations for a given dictionary model.

    This function processes activations from specified datasets (e.g., FineWeb and LMSYS),
    applies the provided dictionary model to compute latent activations, and saves the results
    to disk. Optionally, it can upload the computed activations to the Hugging Face Hub.

    Args:
        dictionary_model_name (str): Path or identifier for the dictionary (crosscoder) model to use.
        activation_caches (list[ActivationCache]): List of activation caches to process.
        tokenizer (AutoTokenizer): Tokenizer to use for processing sequences.
        dataset_names (list[str] or None, optional): Names of datasets corresponding to each activation cache.
            If None, datasets will be labeled as "dataset_0", "dataset_1", etc. Defaults to None.
        latent_ids (torch.Tensor or None, optional): Tensor of latent indices to compute activations for.
            If None, uses all latents in the dictionary model.
        out_dir (Path, optional): Directory to save computed latent activations.
            Defaults to Path("latent_activations/").
        upload_to_hub (bool, optional): Whether to upload the computed activations to the Hugging Face Hub.
            Defaults to False.
        load_from_disk (bool, optional): If True, load precomputed activations from disk instead of recomputing.
            Defaults to False.
        is_sae (bool, optional): Whether the model is an SAE rather than a crosscoder.
            Defaults to False.
        is_difference_sae (bool, optional): Whether the SAE is trained on activation differences.
            Defaults to False.
        difference_target (str, optional): Target of the difference SAE.
        max_num_samples (int, optional): Maximum number of samples to process per dataset. Defaults to 10000.

    Returns:
        None
    """
    is_sae = is_sae or is_difference_sae
    if is_sae and difference_target is None:
        raise ValueError(
            "difference_target must be provided if is_sae is True. This is the target of the difference SAE."
        )
    
    # Handle dataset names - create default names if not provided
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(activation_caches))]
    elif len(dataset_names) != len(activation_caches):
        raise ValueError(f"Number of dataset_names ({len(dataset_names)}) must match number of activation_caches ({len(activation_caches)})")
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the activation dataset
    if not load_from_disk:

        # For difference SAEs, convert to DifferenceCache
        if is_difference_sae:
            activation_caches = [
                setup_sae_cache(target=difference_target, paired_cache=cache)
                for cache in activation_caches
            ]

        # Load the dictionary model
        dictionary_model = load_dictionary_model(
            dictionary_model_name).to("cuda")
        if is_sae:
            dictionary_model = add_get_activations_sae(dictionary_model)
            
        if latent_ids is None:
            latent_ids = torch.arange(dictionary_model.dict_size)

        # Load the tokenizer
        sample_caches = [   
            SampleCache(cache, tokenizer.bos_token_id, max_num_samples=max_num_samples) for cache in activation_caches
        ]

        out_acts = []
        out_ids = []
        seq_ranges = [0]
        max_activations_list = []
        dataset_ids_list = []  # Track which dataset each sequence comes from
        offset = 0
        
        for dataset_idx, sample_cache in enumerate(sample_caches):
            out_acts_i, out_ids_i, seq_ranges_i, max_activations_i = (
                get_positive_activations(sample_cache, dictionary_model, latent_ids)
            )
            # Adjust sequence indices by offset
            out_ids_i[:, 0] += offset
            offset += len(sample_cache)
            
            out_acts.append(out_acts_i)
            out_ids.append(out_ids_i)
            
            # Track dataset ID for each sequence in this sample cache
            dataset_ids_for_cache = torch.full((len(sample_cache),), dataset_idx, dtype=torch.long)
            dataset_ids_list.append(dataset_ids_for_cache)
            
            # Adjust seq_ranges by adding the current total activation count
            current_act_count = len(out_acts_i) if len(out_acts) == 1 else sum(len(acts) for acts in out_acts[:-1])
            adjusted_ranges = [r + current_act_count for r in seq_ranges_i[1:]]  # Skip first 0
            seq_ranges.extend(adjusted_ranges)
            
            max_activations_list.append(max_activations_i)

        # Concatenate all results
        out_acts = torch.cat(out_acts)
        out_ids = torch.cat(out_ids)
        
        # Concatenate dataset IDs
        dataset_ids = torch.cat(dataset_ids_list)
        
        # Combine max activations by taking element-wise maximum
        combined_max_activations = max_activations_list[0]
        for max_acts in max_activations_list[1:]:
            combined_max_activations = torch.maximum(combined_max_activations, max_acts)

        # Combine sequences from all sample caches  
        sequences_all = []
        for sample_cache in sample_caches:
            sequences_all.extend(sample_cache.sequences)

        # Find max length
        max_len = max(len(s) for s in sequences_all)
        seq_lengths = torch.tensor([len(s) for s in sequences_all])
        # Pad each sequence to max length
        padded_seqs = [
            torch.cat(
                [
                    s,
                    torch.full(
                        (max_len - len(s),), tokenizer.pad_token_id, device=s.device
                    ),
                ]
            )
            for s in sequences_all
        ]
        # Convert to tensor and save
        padded_tensor = torch.stack(padded_seqs)

        # Save tensors
        torch.save(out_acts.cpu(), out_dir / "activations.pt")
        torch.save(out_ids.cpu(), out_dir / "indices.pt")
        torch.save(padded_tensor.cpu(), out_dir / "sequences.pt")
        torch.save(latent_ids.cpu(), out_dir / "latent_ids.pt")
        torch.save(torch.tensor(seq_ranges).cpu(), out_dir / "ranges.pt")
        torch.save(seq_lengths.cpu(), out_dir / "lengths.pt")
        torch.save(combined_max_activations.cpu(), out_dir / "max_activations.pt")
        torch.save(dataset_ids.cpu(), out_dir / "dataset_ids.pt")
        torch.save(dataset_names, out_dir / "dataset_names.pt")

        # Print some stats about max activations
        print("Maximum activation statistics:")
        print(f"  Average: {combined_max_activations.mean().item():.4f}")
        print(f"  Maximum: {combined_max_activations.max().item():.4f}")
        print(f"  Minimum: {combined_max_activations.min().item():.4f}")

    if upload_to_hub:
        # Initialize Hugging Face API
        from huggingface_hub import HfApi

        api = HfApi()

        # Define repository ID for the dataset
        repo_id = f"{HF_NAME}/latent-activations-{dictionary_model_name}"
        # Check if repository exists, create it if it doesn't
        if repo_exists(repo_id=repo_id, repo_type="dataset"):
            print(f"Repository {repo_id} already exists")
        else:
            # Repository doesn't exist, create it
            print(f"Repository {repo_id}, creating it...")
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
                exist_ok=True,
            )
            print(f"Created repository {repo_id}")

        # Upload all tensors to HF Hub directly from saved files
        api.upload_file(
            path_or_fileobj=str(out_dir / "activations.pt"),
            path_in_repo="activations.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )

        api.upload_file(
            path_or_fileobj=str(out_dir / "indices.pt"),
            path_in_repo="indices.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )

        api.upload_file(
            path_or_fileobj=str(out_dir / "sequences.pt"),
            path_in_repo="sequences.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )

        api.upload_file(
            path_or_fileobj=str(out_dir / "latent_ids.pt"),
            path_in_repo="latent_ids.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Upload max activations and indices
        api.upload_file(
            path_or_fileobj=str(out_dir / "max_activations.pt"),
            path_in_repo="max_activations.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )

        api.upload_file(
            path_or_fileobj=str(out_dir / "dataset_ids.pt"),
            path_in_repo="dataset_ids.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )

        api.upload_file(
            path_or_fileobj=str(out_dir / "dataset_names.pt"),
            path_in_repo="dataset_names.pt",
            repo_id=repo_id,
            repo_type="dataset",
        )

        print(f"All files uploaded to Hugging Face Hub at {repo_id}")
    else:
        print("Skipping upload to Hugging Face Hub")
    return LatentActivationCache(out_dir)

def collect_dictionary_activations_from_config(
    cfg: DictConfig,
    layer: int,
    dictionary_model_name: str,
    result_dir: Path,
):  
    latent_activations_cfg = cfg.diffing.method.analysis.latent_activations
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
        layers=[layer],
        split=latent_activations_cfg.split,
    )  # Dict {dataset_name: {layer: PairedActivationCache, ...}}

    activation_caches = [caches[dataset_name][layer] for dataset_name in caches]
    dataset_names = list(caches.keys())  # Extract dataset names from cache keys
    
    tokenizer = load_tokenizer_from_config(base_model_cfg)

    # Check if latent activations already exist
    output_path = Path(result_dir) / "latent_activations"
    if output_path.exists() and (output_path / "activations.pt").exists() and not latent_activations_cfg.overwrite:
        logger.info(f"Found existing latent activations at {output_path}. Skipping computation.")
        return LatentActivationCache(output_path)
    else:
        return collect_dictionary_activations(
            dictionary_model_name=dictionary_model_name,
            activation_caches=activation_caches,
            tokenizer=tokenizer,
            dataset_names=dataset_names,
            latent_ids=None,
            out_dir=output_path,
            upload_to_hub=False,
            load_from_disk=False,
            max_num_samples=latent_activations_cfg.max_num_samples,
            is_difference_sae=cfg.diffing.method.name == "sae_difference",
            difference_target=cfg.diffing.method.training.get("target", None)
        )



def quantile_examples_to_db(
    quantile_examples, all_sequences, activation_details, db_path: Path
):
    """Convert quantile examples to a database with binary blob storage for token IDs.

    Args:
        quantile_examples: Dictionary mapping quantile_idx -> feature_idx -> list of (activation_value, sequence_idx)
        all_sequences: List of all sequences used in the examples
        activation_details: Dictionary mapping feature_idx -> sequence_idx -> list of (position, value) pairs
        db_path: Path to save the database
    """
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS sequences (
                sequence_idx INTEGER PRIMARY KEY,
                token_ids BLOB
            )"""
        )

        cursor.execute(
            """CREATE TABLE IF NOT EXISTS quantile_examples (
                feature_idx INTEGER,
                quantile_idx INTEGER,
                activation REAL,
                sequence_idx INTEGER,
                PRIMARY KEY (feature_idx, sequence_idx),
                FOREIGN KEY (sequence_idx) REFERENCES sequences(sequence_idx)
            )"""
        )

        # First, store all sequences
        for seq_idx, token_ids in tqdm(
            enumerate(all_sequences), desc="Storing sequences"
        ):
            # Convert token IDs to binary blob
            binary_data = np.array(token_ids, dtype=np.int32).tobytes()
            cursor.execute(
                "INSERT INTO sequences VALUES (?, ?)",
                (int(seq_idx), binary_data),
            )

        # Then store the quantile examples with references to sequences
        for q_idx, q_data in tqdm(
            quantile_examples.items(), desc="Storing quantile examples"
        ):
            for feature_idx, examples in q_data.items():
                for activation, sequence_idx in examples:
                    # Get the max position from the original sequence
                    # This assumes we're still tracking max positions somewhere
                    # If not, we'd need to modify the compute_quantile_activating_examples function
                    # to also track positions along with activations

                    cursor.execute(
                        "INSERT INTO quantile_examples VALUES (?, ?, ?, ?)",
                        (
                            int(feature_idx),
                            int(q_idx),
                            float(activation),
                            int(sequence_idx),
                        ),
                    )

        # Create a table for storing activation details
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS activation_details (
                feature_idx INTEGER,
                sequence_idx INTEGER,
                positions BLOB,
                activation_values BLOB,
                PRIMARY KEY (feature_idx, sequence_idx),
                FOREIGN KEY (sequence_idx) REFERENCES sequences(sequence_idx),
                FOREIGN KEY (feature_idx, sequence_idx) REFERENCES quantile_examples(feature_idx, sequence_idx)
            )"""
        )

        # After storing all quantile examples
        # Store activation details
        for feature_idx, sequences in tqdm(
            activation_details.items(), desc="Storing activation details"
        ):
            for sequence_idx, pos_val_pairs in sequences.items():
                if len(pos_val_pairs) == 0:
                    continue

                positions_blob = pos_val_pairs[:, 0].tobytes()
                values_blob = pos_val_pairs[:, 1].tobytes()

                cursor.execute(
                    "INSERT INTO activation_details VALUES (?, ?, ?, ?)",
                    (
                        int(feature_idx),
                        int(sequence_idx),
                        positions_blob,
                        values_blob,
                    ),
                )

        conn.commit()


def fix_activations_details(activation_details):
    """Convert activation details from int32 arrays to tuples of (positions, values) with proper types."""
    converted = {}
    for feat_idx, sequences in activation_details.items():
        converted[feat_idx] = {}
        for seq_idx, arr in sequences.items():
            # arr is a Nx2 array where first column is positions (int) and second column is values (float as int32)
            positions = arr[:, 0].astype(np.int32)
            # Convert back the int32 values to float32
            values = arr[:, 1].view(np.float32)
            converted[feat_idx][seq_idx] = (positions, values)
    return converted


def sort_quantile_examples(quantile_examples):
    """Sort quantile examples by activation value."""
    for q_idx in quantile_examples:
        for feature_idx in quantile_examples[q_idx]:
            quantile_examples[q_idx][feature_idx] = sorted(
                quantile_examples[q_idx][feature_idx],
                key=lambda x: x[0],
                reverse=True,
            )
    return quantile_examples


@torch.no_grad()
def compute_quantile_activating_examples(
    latent_activation_cache,
    quantiles=[0.25, 0.5, 0.75, 0.95],
    n=100,
    save_path=None,
    gc_collect_every=1000,
    use_random_replacement=True,
) -> dict:
    """Compute examples that activate features at different quantile levels.

    Args:
        latent_activation_cache: Pre-computed latent activation cache
        quantiles: List of quantile thresholds (as fractions of max activation)
        n: Number of examples to collect per feature per quantile
        save_path: Path to save results
        gc_collect_every: How often to run garbage collection
        use_random_replacement: Whether to use random replacement for reservoir sampling

    Returns:
        Tuple of (quantile_examples, all_sequences, activation_details) where:
            - quantile_examples: Dictionary mapping quantile_idx -> feature_idx -> list of (activation_value, sequence_idx, position)
            - all_sequences: List of all token sequences used in the examples
            - activation_details: Dictionary mapping feature_idx -> sequence_idx -> (positions, values)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move max_activations and quantiles to GPU
    max_activations = latent_activation_cache.max_activations
    quantiles_tensor = torch.tensor(quantiles, device=device)

    # Calculate quantile thresholds for each feature on GPU
    thresholds = torch.einsum("f,q->fq", max_activations, quantiles_tensor)

    # Initialize collections for each quantile
    quantile_examples = {
        q_idx: {feat_idx: [] for feat_idx in range(len(max_activations))}
        for q_idx in range(len(quantiles) + 1)
    }

    # Keep track of how many examples we've seen for each feature and quantile
    example_counts = {
        q_idx: {feat_idx: 0 for feat_idx in range(len(max_activations))}
        for q_idx in range(len(quantiles) + 1)
    }

    # Store all unique sequences
    sequences_set = set()
    all_sequences = []

    # Dictionary to store feature activation details: {feature_idx: {sequence_idx: [(position, value), ...]}}
    activation_details = defaultdict(dict)

    next_gb = gc_collect_every
    current_seq_idx = 0
    for tokens, (indices, values) in tqdm(latent_activation_cache):
        # GC and device transfer
        next_gb -= 1
        if next_gb <= 0:
            gc.collect()
            next_gb = gc_collect_every

        token_tuple = tuple(tokens.tolist())
        if token_tuple in sequences_set:
            continue
        sequences_set.add(token_tuple)
        all_sequences.append(token_tuple)

        # Core computation
        features, sort_indices = torch.sort(indices[:, 1])
        token_indices = indices[:, 0][sort_indices]
        values = values[sort_indices]
        active_features, inverse_indices, counts = features.unique(
            return_inverse=True, return_counts=True
        )
        max_vals = torch.zeros_like(active_features, dtype=values.dtype)
        max_vals = torch.scatter_reduce(
            max_vals, 0, inverse_indices, values, reduce="amax"
        )
      
        active_thresholds = thresholds[active_features]

        q_idxs = torch.searchsorted(active_thresholds, max_vals.unsqueeze(-1)).squeeze()

        active_features = active_features.tolist()
        counts = counts.tolist()
        max_vals = max_vals.tolist()
        q_idxs = q_idxs.tolist()

        current_idx = 0
        latent_details = (
            torch.stack(
                [token_indices.int(), values.float().view(torch.int32)],
                dim=1,
            )
            .cpu()
            .numpy()
        )

        for feat, count, max_val, q_idx in zip(
            active_features,
            counts,
            max_vals,
            q_idxs,
        ):
            example_counts[q_idx][feat] += 1
            total_count = example_counts[q_idx][feat]

            if total_count <= n:
                quantile_examples[q_idx][feat].append((max_val, current_seq_idx))
                activation_details[feat][current_seq_idx] = latent_details[
                    current_idx : current_idx + count
                ]
            elif use_random_replacement:
                if random.random() < n / total_count:
                    replace_idx = random.randint(0, n - 1)
                    replaced_seq_idx = quantile_examples[q_idx][feat][replace_idx][1]
                    quantile_examples[q_idx][feat][replace_idx] = (
                        max_val,
                        current_seq_idx,
                    )
                    if (
                        feat in activation_details
                        and replaced_seq_idx in activation_details[feat]
                    ):
                        del activation_details[feat][replaced_seq_idx]
                    activation_details[feat][current_seq_idx] = latent_details[
                        current_idx : current_idx + count
                    ]
            current_idx += count

        current_seq_idx += 1

    # Sort and finalize results
    logger.info(f"Sorting {len(quantile_examples)} quantiles")
    quantile_examples = sort_quantile_examples(quantile_examples)
    name = "examples"
    # Save to database
    if save_path is not None:
        logger.info(f"Saving to {save_path / f'{name}.db'}")
        quantile_examples_to_db(
            quantile_examples,
            all_sequences,
            activation_details,
            save_path / f"{name}.db",
        )

    activation_details = fix_activations_details(activation_details)
    if save_path is not None:
        logger.info(f"Saving to {save_path / f'{name}.pt'}")
        torch.save(
            (quantile_examples, all_sequences, activation_details),
            save_path / f"{name}.pt",
        )

    return quantile_examples, all_sequences, activation_details


def collect_activating_examples(
    dictionary_model_name: str,
    latent_activation_cache: LatentActivationCache,
    n: int = 100,
    min_threshold: float = 1e-4,
    quantiles: list[float] = [0.25, 0.5, 0.75, 0.95, 1.0],
    save_path: Path = Path("results"),
    upload_to_hub: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Collect and save examples that activate latent features at different quantiles.

    This function processes latent activations to find examples that activate features
    at specified quantile thresholds. It can optionally save results locally and/or
    upload them to HuggingFace Hub.

    Args:
        crosscoder (str): Name of the crosscoder model to analyze
        latent_activation_cache_path (Path): Path to directory containing latent activation data
        n (int, optional): Number of examples to collect per quantile. Defaults to 100.
        min_threshold (float, optional): Minimum activation threshold. Defaults to 1e-4.
        quantiles (list[float], optional): Quantile thresholds to analyze.
            Defaults to [0.25, 0.5, 0.75, 0.95, 1.0].
        save_path (Path, optional): Directory to save results.
            Defaults to Path("results").
        upload_to_hub (bool, optional): If True, upload results to HuggingFace.
            Defaults to False.
        overwrite (bool, optional): If True, overwrite existing results.
            Defaults to False.  
    Returns:
        None
    """
    save_path = save_path / "latent_activations" 
    # Check if files already exist
    pt_file = save_path / "examples.pt"
    db_file = save_path / "examples.db"
    files_exist = pt_file.exists() or db_file.exists()
    
    if not files_exist or overwrite:
        # Create save directory if it doesn't exist
        save_path.mkdir(parents=True, exist_ok=True)

        # Generate and save quantile examples
        print("Generating quantile examples...")
        compute_quantile_activating_examples(
            latent_activation_cache=latent_activation_cache,
            quantiles=quantiles,
            min_threshold=min_threshold,
            n=n,
            save_path=save_path,
        )

    # Upload to HuggingFace Hub
    repo = f"{HF_NAME}/diffing-stats-" + dictionary_model_name
    if upload_to_hub:
        print(f"Uploading to HuggingFace Hub: {repo}")
        for ftype in ["pt", "db"]:
            name = "examples"
            file_path = save_path / f"{name}.{ftype}"
            print(f"Uploading {file_path} to {repo}")
            if file_path.exists():
                hf_api.upload_file(
                    repo_id=repo,
                    repo_type="dataset",
                    path_or_fileobj=file_path,
                    path_in_repo=f"{name}.{ftype}",
                )



@torch.no_grad()
def compute_latent_stats(
    latent_cache: LatentActivationCache,
    device: torch.device,
):
    """
    Compute maximum activation values and frequencies for each latent feature using a LatentActivationCache.

    Args:
        latent_cache: LatentActivationCache containing the latent activations
        device: Device to perform computations on (e.g. "cuda" or "cpu")

    Returns:
        Tuple containing:
        - Tensor of shape (dict_size,) containing the maximum activation value for each latent feature
        - Tensor of shape (dict_size,) containing the frequency of each feature (non-zero activations / total tokens)
        - int: total number of tokens in the cache
    """
    latent_cache.to(device)
    dict_size = latent_cache.dict_size
    max_activations = torch.zeros(dict_size, device=device)
    nonzero_counts = torch.zeros(dict_size, device=device, dtype=torch.long)
    total_tokens = 0

    # Iterate through all samples in the cache
    pbar = trange(len(latent_cache), desc="Computing max activations")
    for i in pbar:
        # Get latent activations for this sample
        tokens, activations = latent_cache[i]
        total_tokens += len(tokens)
        # If using sparse tensor format
        if isinstance(activations, torch.Tensor) and activations.is_sparse:
            values = activations.values()
            indices = activations.indices()
            latent_indices = indices[1, :]  # Second row contains latent indices
            max_activations.index_reduce_(0, latent_indices, values, "amax")
            # Count non-zero activations per feature
            unique_indices, counts = latent_indices.unique(return_counts=True)
            nonzero_counts.index_add_(0, unique_indices, counts)

        # If using dense format
        elif isinstance(activations, torch.Tensor):
            batch_max = activations.max(dim=0).values
            max_activations = torch.maximum(max_activations, batch_max)
            nonzero_counts += (activations != 0).sum(dim=0)

        # If using sparse tuple format (indices, values)
        else:
            indices, values = activations
            latent_indices = indices[:, 1]
            max_activations.index_reduce_(0, latent_indices, values, "amax")
            # Count non-zero activations
            unique_indices, counts = latent_indices.unique(return_counts=True)
            nonzero_counts.index_add_(0, unique_indices, counts)

    frequencies = nonzero_counts / total_tokens
    return max_activations.cpu(), frequencies.cpu(), total_tokens

def update_latent_df_with_stats(
        dictionary_name: str,
        latent_activation_cache: LatentActivationCache,
        split_of_cache: str,
):
    """
    Update the latent df with the computed max activations and frequencies.
    """
    df = load_latent_df(dictionary_name)
    max_activations, frequencies, total_tokens = compute_latent_stats(
        latent_cache=latent_activation_cache,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    df[f"max_act_{split_of_cache}"] = max_activations.cpu()
    df[f"freq_{split_of_cache}"] = frequencies.cpu() 
    push_latent_df(df, dictionary_name, confirm=False)