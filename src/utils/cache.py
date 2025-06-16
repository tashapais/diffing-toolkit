from dictionary_learning.cache import PairedActivationCache, ActivationCache
import torch
from pathlib import Path
from tqdm.auto import tqdm


class TokenCache:
    """
    A wrapper around an ActivationCache that provides access to a subset of tokens.

    This class enables efficient filtering and selection of tokens from an underlying
    activation cache without duplicating the entire cache. It maintains a mapping of indices
    that reference the original cache, allowing for memory-efficient operations on subsets
    of the data.

    Example:
        from utils.cache import TokenCache
        from utils.activations import load_activation_dataset

        cache = load_activation_dataset(
            activation_store_dir=DATA_ROOT / "activations/",
            split="validation",
            dataset_name="lmsys-chat-1m-chat-formatted",
            base_model="gemma-2-2b",
            finetuned_model="gemma-2-2b-it",
            layer=13,
        )
        token_cache = TokenCache(cache)
        print(token_cache[0])  # Access the first activation (token, activation)
        only_bos_token_cache = token_cache.select_tokens([2])
        print(only_bos_token_cache[0])  # Access the first bos activation (token, activation)
        no_bos_token_cache = token_cache.remove_tokens([2])
        print(no_bos_token_cache[0])  # Access the first bos activation (token, activation)

    Attributes:
        cache: The underlying ActivationCache or PairedActivationCache containing token data and activations.
        indices: List of indices into the original cache that this TokenCache provides access to.
        tokens: The subset of tokens from the original cache corresponding to the selected indices.
    """

    def __init__(
        self, cache: ActivationCache | PairedActivationCache, indices: list[int] = None
    ):
        """
        Initialize a TokenCache with a reference to an activation cache and optional indices.

        Args:
            cache: The underlying ActivationCache or PairedActivationCache to wrap.
            indices: Optional list of indices to include. If None, includes all indices from the original cache.

        Raises:
            ValueError: If the cache type is not supported.
            AssertionError: If using a PairedActivationCache with mismatched tokens.
        """
        self.cache = cache
        if indices is not None:
            assert len(indices) > 0, "Indices must be a non-empty list"
            self.indices = indices
        else:
            self.indices = list(range(len(cache)))

        if isinstance(cache, PairedActivationCache):
            assert torch.all(
                cache.tokens[0] == cache.tokens[1]
            ), "Tokens must be the same for PairedActivationCache"
            self._tokens = cache.tokens[0]
        elif isinstance(cache, ActivationCache):
            self._tokens = cache.tokens
        else:
            raise ValueError(f"Unsupported cache type: {type(cache)}")

    @property
    def tokens(self):
        return self._tokens[self.indices]

    def __len__(self) -> int:
        """
        Return the number of tokens in this cache subset.

        Returns:
            int: The number of tokens in the current subset.
        """
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieve a token and its activation at the specified index.

        Args:
            index: Index into this TokenCache's subset.

        Returns:
            tuple: A pair containing (token, activation) from the underlying cache.
        """
        full_cache_index = self.indices[index]
        return self._tokens[full_cache_index], self.cache[full_cache_index]

    def get_token(self, index: int) -> int:
        """
        Retrieve the token ID at the specified index without its activation.

        This method provides direct access to just the token ID, unlike __getitem__
        which returns both the token and its activation.

        Args:
            index: Index into this TokenCache's subset.

        Returns:
            int: The token ID from the underlying cache at the specified index.
        """
        full_cache_index = self.indices[index]
        return self._tokens[full_cache_index]

    def remove_tokens(self, token_ids: list[int]) -> "TokenCache":
        """
        Create a new TokenCache excluding the specified indices.

        This method filters out tokens at the specified indices from the current subset,
        returning a new TokenCache with the remaining tokens.

        Args:
            token_ids: List of token ids to exclude from the current subset.

        Returns:
            TokenCache: A new TokenCache instance with the filtered subset of tokens.
        """
        indices_to_exclude = set(token_ids)
        tokens = self._tokens.tolist()
        indices = [i for i in self.indices if tokens[i] not in indices_to_exclude]
        return TokenCache(self.cache, indices)

    def select_tokens(self, token_ids: list[int]) -> "TokenCache":
        """
        Create a new TokenCache including only the specified indices.

        This method creates a new subset containing only the tokens at the specified indices,
        allowing for precise selection of tokens of interest.

        Args:
            token_ids: List of token ids to include in the new subset.

        Returns:
            TokenCache: A new TokenCache instance with only the selected tokens.
        """
        indices_set = set(token_ids)
        tokens = self._tokens.tolist()
        indices = [i for i in self.indices if tokens[i] in indices_set]
        return TokenCache(self.cache, indices)


class SampleCache:
    """
    A cache that organizes activations by samples, where each sample starts with a BOS token.

    This class provides a way to access activations and tokens for complete sequences
    (samples) rather than individual tokens. It identifies sample boundaries using
    the beginning-of-sequence (BOS) token.

    Example:
        from utils.cache import SampleCache
        from utils.activations import load_activation_dataset

        cache = load_activation_dataset(
            activation_store_dir="$DATASTORE/activations/",
            split="validation",
            dataset_name="lmsys-chat-1m-chat-formatted",
            base_model="gemma-2-2b",
            finetuned_model="gemma-2-2b-it",
            layer=13,
        )
        sample_cache = SampleCache(cache, tokenizer.bos_token_id)

        sample_cache.sequences # List of sequences of tokens
        sample_cache[0] # (tokens: (len(sequence), ), activations: (len(sequence), num_layers, dim_model))

    Args:
        cache: An ActivationCache or PairedActivationCache containing the tokens and activations.
        bos_token_id: The token ID that marks the beginning of a sequence (default: 2).

    Raises:
        ValueError: If the cache type is not supported.
        AssertionError: If tokens don't match in a PairedActivationCache or if shuffled shards are used.
    """

    def __init__(
        self,
        cache: ActivationCache | PairedActivationCache,
        bos_token_id: int = 2,
        max_num_samples: int = None,
    ):
        self.cache = cache
        self.bos_token_id = bos_token_id
        self.max_num_samples = max_num_samples
        if isinstance(cache, PairedActivationCache):
            assert torch.all(
                cache.tokens[0] == cache.tokens[1]
            ), "Tokens must be the same for PairedActivationCache"
            self._tokens = cache.tokens[0]
            assert (
                not cache.activation_cache_1.config["shuffle_shards"]
                and not cache.activation_cache_2.config["shuffle_shards"]
            ), "Shuffled shards are not supported for SampleCache"
        elif isinstance(cache, ActivationCache):
            self._tokens = cache.tokens
            assert not cache.config[
                "shuffle_shards"
            ], "Shuffled shards are not supported for SampleCache"
        else:
            raise ValueError(f"Unsupported cache type: {type(cache)}")
        tokens = self._tokens.tolist()
        self.sample_start_indices = [
            i for i in range(len(tokens)) if tokens[i] == self.bos_token_id
        ] + [len(tokens)]
        if self.max_num_samples is not None:
            self.sample_start_indices = self.sample_start_indices[
                : self.max_num_samples + 1
            ]
        self._indices_to_seq_pos = None
        self._sequences = None
        self._ranges = None

    def __len__(self):
        """
        Returns the number of samples in the cache.

        Returns:
            int: The number of distinct samples identified by BOS tokens.
        """
        return len(self.sample_start_indices) - 1

    def _compute(self):
        self._ranges = list(
            zip(self.sample_start_indices[:-1], self.sample_start_indices[1:])
        )
        self._sequences = [
            self._tokens[start_index:end_index]
            for start_index, end_index in self._ranges
        ]
        self._indices_to_seq_pos = [
            (i, j)
            for i in range(len(self._ranges))
            for j in range(len(self._ranges[i]))
        ]

    @property
    def sequences(self):
        """
        Returns all token sequences in the cache.

        Returns:
            list: A list of token tensors, where each tensor represents a complete sequence.
        """
        if self._sequences is None:
            self._compute()
        return self._sequences

    @property
    def indices_to_seq_pos(self):
        if self._indices_to_seq_pos is None:
            self._compute()
        return self._indices_to_seq_pos

    @property
    def ranges(self):
        if self._ranges is None:
            self._compute()
        return self._ranges

    def __getitem__(self, index: int):
        """
        Retrieves tokens and activations for a specific sample.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            tuple: A pair containing:
                - The token sequence for the sample
                - A tensor of activations for each token in the sample
        """
        start_index = self.sample_start_indices[index]
        end_index = self.sample_start_indices[index + 1]
        sample_tokens = self._tokens[start_index:end_index]
        sample_activations = torch.stack(
            [self.cache[i] for i in range(start_index, end_index)], dim=0
        )
        return sample_tokens, sample_activations


class LatentActivationCache:
    def __init__(
        self,
        latent_activations_dir: Path,
        expand=True,
        offset=0,
        use_sparse_tensor=False,
        device: torch.device = None,
    ):
        if isinstance(latent_activations_dir, str):
            latent_activations_dir = Path(latent_activations_dir)

        # Create progress bar for 7 files to load
        pbar = tqdm(total=7, desc="Loading cache files")

        pbar.set_postfix_str("Loading out_acts.pt")
        self.acts = torch.load(
            latent_activations_dir / "out_acts.pt", weights_only=True
        )
        pbar.update(1)

        pbar.set_postfix_str("Loading out_ids.pt")
        self.ids = torch.load(latent_activations_dir / "out_ids.pt", weights_only=True)
        pbar.update(1)

        pbar.set_postfix_str("Loading max_activations.pt")
        self.max_activations = torch.load(
            latent_activations_dir / "max_activations.pt", weights_only=True
        )
        pbar.update(1)

        pbar.set_postfix_str("Loading latent_ids.pt")
        self.latent_ids = torch.load(
            latent_activations_dir / "latent_ids.pt", weights_only=True
        )
        pbar.update(1)

        pbar.set_postfix_str("Loading padded_sequences.pt")
        self.padded_sequences = torch.load(
            latent_activations_dir / "padded_sequences.pt", weights_only=True
        )
        pbar.update(1)

        self.dict_size = self.max_activations.shape[0]

        pbar.set_postfix_str("Loading seq_lengths.pt")
        self.sequence_lengths = torch.load(
            latent_activations_dir / "seq_lengths.pt", weights_only=True
        )
        pbar.update(1)

        pbar.set_postfix_str("Loading seq_ranges.pt")
        self.sequence_ranges = torch.load(
            latent_activations_dir / "seq_ranges.pt", weights_only=True
        )
        pbar.update(1)
        pbar.close()

        self.expand = expand
        self.offset = offset
        self.use_sparse_tensor = use_sparse_tensor
        self.device = device
        if device is not None:
            self.to(device)

    def __len__(self):
        return len(self.padded_sequences) - self.offset

    def __getitem__(self, index: int):
        """
        Retrieves tokens and latent activations for a specific sequence.

        Args:
            index (int): The index of the sequence to retrieve.

        Returns:
            tuple: A pair containing:
                - The token sequence for the sample
                - If self.expand is True:
                    - If use_sparse_tensor is True:
                        A sparse tensor of shape (sequence_length, dict_size) containing the latent activations
                    - If use_sparse_tensor is False:
                        A dense tensor of shape (sequence_length, dict_size) containing the latent activations
                - If self.expand is False:
                    A tuple of (indices, values) representing sparse latent activations where:
                    - indices: Tensor of shape (N, 2) containing (token_idx, dict_idx) pairs
                    - values: Tensor of shape (N,) containing activation values
        """
        return self.get_sequence(index), self.get_latent_activations(
            index, expand=self.expand, use_sparse_tensor=self.use_sparse_tensor
        )

    def get_sequence(self, index: int):
        return self.padded_sequences[index + self.offset][
            : self.sequence_lengths[index + self.offset]
        ]

    def get_latent_activations(
        self, index: int, expand: bool = True, use_sparse_tensor: bool = False
    ):
        start_index = self.sequence_ranges[index + self.offset]
        end_index = self.sequence_ranges[index + self.offset + 1]
        seq_indices = self.ids[start_index:end_index]
        assert torch.all(
            seq_indices[:, 0] == index + self.offset
        ), f"Was supposed to find {index + self.offset} but found {seq_indices[:, 0].unique()}"
        seq_indices = seq_indices[:, 1:]  # remove seq_idx column

        if expand:
            if use_sparse_tensor:
                # Create sparse tensor directly
                indices = (
                    seq_indices.t()
                )  # Transpose to get 2xN format required by sparse tensors
                values = self.acts[start_index:end_index]
                sparse_shape = (
                    self.sequence_lengths[index + self.offset],
                    self.dict_size,
                )
                return torch.sparse_coo_tensor(indices, values, sparse_shape)
            else:
                # Create dense tensor as before
                latent_activations = torch.zeros(
                    self.sequence_lengths[index + self.offset],
                    self.dict_size,
                    device=self.acts.device,
                )
                latent_activations[seq_indices[:, 0], seq_indices[:, 1]] = self.acts[
                    start_index:end_index
                ]
                return latent_activations
        else:
            return (seq_indices, self.acts[start_index:end_index])

    def to(self, device: torch.device):
        self.acts = self.acts.to(device)
        self.ids = self.ids.to(device)
        self.max_activations = self.max_activations.to(device)
        self.latent_ids = self.latent_ids.to(device)
        self.padded_sequences = self.padded_sequences.to(device)
        self.device = device
        return self
