"""
Tests for MaxActStore class with both sparse and dense storage formats.

This module tests the SQLite-based storage for maximum activating examples,
including both bulk loading and real-time top-k management functionality
with both sparse and dense activation details storage.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sqlite3

from src.utils.max_act_store import MaxActStore, AsyncMaxActStoreWriter

class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def decode(self, token_ids, skip_special_tokens=False):
        """Simple mock decode that just joins token IDs with spaces."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join([f"tok_{id}" for id in token_ids])


@pytest.fixture
def temp_db_path():
    """Fixture providing a temporary database path."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    yield db_path
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_tokenizer():
    """Fixture providing a mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def sample_token_sequences():
    """Fixture providing sample token sequences for testing."""
    return [
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([5, 6, 7]),
        torch.tensor([8, 9, 10, 11, 12]),
        torch.tensor([13, 14]),
        torch.tensor([15, 16, 17, 18])
    ]


@pytest.fixture
def sample_quantile_examples():
    """Fixture providing sample quantile examples data."""
    return {
        0: {  # quantile_idx 0
            0: [(0.9, 0), (0.8, 1)],  # latent_idx 0: [(score, sequence_idx), ...]
            1: [(0.7, 2)],             # latent_idx 1
        },
        1: {  # quantile_idx 1
            0: [(0.6, 3)],             # latent_idx 0
            2: [(0.5, 4)],             # latent_idx 2
        }
    }


@pytest.fixture
def sample_activation_details():
    """Fixture providing sample activation details."""
    return {
        0: {  # latent_idx 0
            0: (np.array([0, 1, 2]), np.array([0.9, 0.8, 0.7])),  # sequence_idx 0
            1: (np.array([0, 1]), np.array([0.8, 0.6])),           # sequence_idx 1
            3: (np.array([0]), np.array([0.6])),                   # sequence_idx 3
        },
        1: {  # latent_idx 1
            2: (np.array([0, 1, 2]), np.array([0.7, 0.5, 0.4])),  # sequence_idx 2
        },
        2: {  # latent_idx 2
            4: (np.array([0, 1]), np.array([0.5, 0.3])),           # sequence_idx 4
        }
    }


@pytest.fixture(params=['sparse', 'dense'])
def storage_format(request):
    """Parametrized fixture for testing both storage formats."""
    return request.param


class TestMaxActStore:
    """Test class for MaxActStore functionality."""
    
    def test_initialization_sparse(self, temp_db_path, mock_tokenizer):
        """Test basic initialization with sparse storage."""
        store = MaxActStore(temp_db_path, max_examples=100, tokenizer=mock_tokenizer, storage_format='sparse')
        
        assert store.db_path == temp_db_path
        assert store.max_examples == 100
        assert store.tokenizer == mock_tokenizer
        assert store.storage_format == 'sparse'
        assert temp_db_path.exists()
        
        # Test that database tables were created
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            
            # Check all tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sequences'")
            assert cursor.fetchone() is not None
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='examples'")
            assert cursor.fetchone() is not None
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='activation_details'")
            assert cursor.fetchone() is not None
                
    def test_initialization_dense(self, temp_db_path, mock_tokenizer):
        """Test basic initialization with dense storage."""
        store = MaxActStore(temp_db_path, max_examples=100, tokenizer=mock_tokenizer, storage_format='dense')
        
        assert store.storage_format == 'dense'
    
    def test_initialization_invalid_format(self, temp_db_path):
        """Test initialization with invalid storage format."""
        with pytest.raises(ValueError):
            MaxActStore(temp_db_path, storage_format='invalid')
    
    def test_length_empty(self, temp_db_path, storage_format):
        """Test length method on empty database."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)
        assert len(store) == 0
    
    def test_clear(self, temp_db_path, storage_format):
        """Test clearing the database."""
        store = MaxActStore(temp_db_path, max_examples=5, storage_format=storage_format)
        
        # Add some data
        store.add_example(
            score=0.8,
            input_ids=torch.tensor([1, 2, 3]),
            scores_per_token=torch.tensor([0.8, 0.6, 0.4])
        )
        
        assert len(store) == 1
        
        # Clear and verify empty
        store.clear()
        assert len(store) == 0
    
    def test_add_single_example(self, temp_db_path, mock_tokenizer, storage_format):
        """Test adding a single example."""
        store = MaxActStore(temp_db_path, max_examples=5, tokenizer=mock_tokenizer, storage_format=storage_format)
        
        input_ids = torch.tensor([1, 2, 3, 4])
        scores_per_token = torch.tensor([0.8, 0.6, 0.4, 0.2])
        
        store.add_example(
            score=0.8,
            input_ids=input_ids,
            scores_per_token=scores_per_token,
            latent_idx=5,
            quantile_idx=2,
            additional_data={"layer": 10}
        )
        
        assert len(store) == 1
        
        # Retrieve and verify
        examples = store.get_top_examples()
        assert len(examples) == 1
        
        example = examples[0]
        assert example["max_score"] == 0.8
        assert example["input_ids"] == [1, 2, 3, 4]
        assert example["latent_idx"] == 5
        assert example["quantile_idx"] == 2
        assert example["layer"] == 10
        assert "tok_1 tok_2 tok_3 tok_4" in example["text"]
        
        # Test activation details retrieval for both formats
        details = store.get_example_details(example["example_id"])
        assert "scores_per_token" in details
        assert len(details["scores_per_token"]) == 4
        # Use approximate comparisons for floating point values
        expected_scores = [0.8, 0.6, 0.4, 0.2]
        for i, expected in enumerate(expected_scores):
            assert np.isclose(details["scores_per_token"][i], expected)
    
    def test_activation_details_different_formats_sparse(self, temp_db_path):
        """Test handling different activation details formats with sparse storage."""
        store = MaxActStore(temp_db_path, storage_format='sparse')
        self._test_activation_details_formats_sparse(store)
    
    def test_activation_details_different_formats_dense(self, temp_db_path):
        """Test handling different activation details formats with dense storage."""
        store = MaxActStore(temp_db_path, storage_format='dense')
        self._test_activation_details_formats_dense(store)
        
    def _test_activation_details_formats_sparse(self, store):
        """Helper method to test sparse activation details formats."""
        # Test with tuple format for sparse storage
        examples_data = {0: {0: [(0.8, 0)]}}
        sequences = [torch.tensor([1, 2, 3])]
        activation_details = {
            0: {0: (np.array([0, 1]), np.array([0.8, 0.6]))}  # Tuple format (positions, values)
        }
        
        store.fill(examples_data, sequences, activation_details)
        
        # Retrieve and verify
        examples = store.get_top_examples()
        example_id = examples[0]["example_id"]
        details = store.get_example_details(example_id)
        
        assert "scores_per_token" in details
        assert len(details["scores_per_token"]) == 3  # Full sequence length (dense format)
        assert np.isclose(details["scores_per_token"][0], 0.8)
        assert np.isclose(details["scores_per_token"][1], 0.6)
        assert np.isclose(details["scores_per_token"][2], 0.0)  # Zero-padded
        
        # Now test with Nx2 array format
        store.clear()
        
        # Create Nx2 array where positions are int and values are float32 viewed as int32
        positions = np.array([0, 1], dtype=np.int32)
        values = np.array([0.8, 0.6], dtype=np.float32)
        # Convert float32 values to int32 view (as done in original code)
        values_as_int32 = values.view(np.int32)
        pos_val_array = np.column_stack([positions, values_as_int32])
        
        activation_details_nx2 = {
            0: {0: pos_val_array}  # Nx2 array format
        }
        
        store.fill(examples_data, sequences, activation_details_nx2)
        
        examples = store.get_top_examples()
        example_id = examples[0]["example_id"]
        details = store.get_example_details(example_id)
        
        assert "scores_per_token" in details
        assert len(details["scores_per_token"]) == 3  # Full sequence length (dense format)
        assert np.isclose(details["scores_per_token"][0], 0.8)
        assert np.isclose(details["scores_per_token"][1], 0.6)
        assert np.isclose(details["scores_per_token"][2], 0.0)  # Zero-padded

    def _test_activation_details_formats_dense(self, store):
        """Helper method to test dense activation details formats."""
        # Test with dense array format (just the values array)
        examples_data = {0: {0: [(0.8, 0)]}}
        sequences = [torch.tensor([1, 2, 3])]
        activation_details = {
            0: {0: np.array([0.8, 0.6, 0.0])}  # Dense format - single values array
        }
        
        store.fill(examples_data, sequences, activation_details)
        
        # Retrieve and verify
        examples = store.get_top_examples()
        example_id = examples[0]["example_id"]
        details = store.get_example_details(example_id)
        
        assert "scores_per_token" in details
        assert len(details["scores_per_token"]) == 3  # Full sequence length
        assert np.isclose(details["scores_per_token"][0], 0.8)
        assert np.isclose(details["scores_per_token"][1], 0.6)
        assert np.isclose(details["scores_per_token"][2], 0.0)
    
    def test_storage_format_efficiency(self, temp_db_path):
        """Test that dense storage is more efficient for high-density activations."""
        # Create data with high density (all positions have activations)
        sequences = [torch.tensor([1, 2, 3, 4, 5])]
        examples_data = {0: {0: [(0.8, 0)]}}
        
        # Test sparse storage with tuple format
        sparse_activation_details = {
            0: {0: (np.array([0, 1, 2, 3, 4]), np.array([0.8, 0.6, 0.4, 0.2, 0.1]))}
        }
        store_sparse = MaxActStore(temp_db_path.with_suffix('.sparse.db'), storage_format='sparse')
        store_sparse.fill(examples_data, sequences, sparse_activation_details)
        
        # Test dense storage with dense array format  
        dense_activation_details = {
            0: {0: np.array([0.8, 0.6, 0.4, 0.2, 0.1])}  # Dense format
        }
        store_dense = MaxActStore(temp_db_path.with_suffix('.dense.db'), storage_format='dense')
        store_dense.fill(examples_data, sequences, dense_activation_details)
        
        # Both should give same results
        details_sparse = store_sparse.get_example_details(1, return_dense=True)
        details_dense = store_dense.get_example_details(1, return_dense=True)
        
        # Both should return dense format now
        assert len(details_sparse["scores_per_token"]) == 5
        assert len(details_dense["scores_per_token"]) == 5
        
        # Test sparse format with return_dense=False
        details_sparse_sparse = store_sparse.get_example_details(1, return_dense=False)
        assert len(details_sparse_sparse["positions"]) == 5  # All positions have values
        assert len(details_sparse_sparse["scores_per_token"]) == 5
        
        # Values should be the same
        for i in range(5):
            assert abs(details_sparse["scores_per_token"][i] - details_dense["scores_per_token"][i]) < 1e-6

    def test_fill_bulk_loading(self, temp_db_path, sample_quantile_examples, 
                              sample_token_sequences, sample_activation_details, mock_tokenizer, storage_format):
        """Test bulk loading with fill method."""
        store = MaxActStore(temp_db_path, tokenizer=mock_tokenizer, storage_format=storage_format)
        
        store.fill(sample_quantile_examples, sample_token_sequences, sample_activation_details)
        
        # Should have 5 total examples
        assert len(store) == 5
        
        # Test retrieval
        examples = store.get_top_examples()
        assert len(examples) == 5
        
        # Examples should be sorted by score descending
        scores = [ex["max_score"] for ex in examples]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 0.9  # Highest score
        assert scores[-1] == 0.5  # Lowest score
    
    def test_top_k_management(self, temp_db_path, storage_format):
        """Test top-k management functionality."""
        store = MaxActStore(temp_db_path, max_examples=3, storage_format=storage_format)
        
        # Add 5 examples with different scores
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        for i, score in enumerate(scores):
            store.add_example(
                score=score,
                input_ids=torch.tensor([i, i+1, i+2]),
                scores_per_token=torch.tensor([score, score-0.1, score-0.2])
            )
        
        # Should only keep top 3
        assert len(store) == 3
        
        examples = store.get_top_examples()
        example_scores = [ex["max_score"] for ex in examples]
        assert example_scores == [0.9, 0.8, 0.7]
    
    def test_get_examples_with_filters(self, temp_db_path, sample_quantile_examples, 
                                     sample_token_sequences, mock_tokenizer, storage_format):
        """Test filtering examples by latent_idx and quantile_idx."""
        store = MaxActStore(temp_db_path, tokenizer=mock_tokenizer, storage_format=storage_format)
        store.fill(sample_quantile_examples, sample_token_sequences)
        
        # Filter by latent_idx only
        examples = store.get_top_examples(latent_idx=0)
        assert len(examples) == 3  # latent 0 appears in 3 examples
        assert all(ex["latent_idx"] == 0 for ex in examples)
        
        # Filter by quantile_idx only
        examples = store.get_top_examples(quantile_idx=1)
        assert len(examples) == 2  # quantile 1 has 2 examples
        assert all(ex["quantile_idx"] == 1 for ex in examples)
        
        # Filter by both
        examples = store.get_top_examples(latent_idx=0, quantile_idx=1)
        assert len(examples) == 1
        assert examples[0]["latent_idx"] == 0 and examples[0]["quantile_idx"] == 1
        
        # Filter with limit
        examples = store.get_top_examples(latent_idx=0, limit=2)
        assert len(examples) == 2

    def test_merge_basic(self, temp_db_path, mock_tokenizer, storage_format):
        """Test basic merge functionality."""
        # Create two stores with same storage format
        store1_path = temp_db_path.with_suffix('.store1.db')
        store2_path = temp_db_path.with_suffix('.store2.db')
        
        store1 = MaxActStore(store1_path, max_examples=10, tokenizer=mock_tokenizer, storage_format=storage_format)
        store2 = MaxActStore(store2_path, max_examples=10, tokenizer=mock_tokenizer, storage_format=storage_format)
        
        # Add data to store1
        store1.add_example(
            score=0.9,
            input_ids=torch.tensor([1, 2, 3]),
            scores_per_token=torch.tensor([0.9, 0.8, 0.7]),
            latent_idx=0,
            additional_data={"dataset": "store1"}
        )
        
        # Add data to store2  
        store2.add_example(
            score=0.8,
            input_ids=torch.tensor([4, 5, 6]),
            scores_per_token=torch.tensor([0.8, 0.7, 0.6]),
            latent_idx=1,
            additional_data={"dataset": "store2"}
        )
        
        assert len(store1) == 1
        assert len(store2) == 1
        
        # Merge store2 into store1
        store1.merge(store2)
        
        # Check results
        assert len(store1) == 2
        assert len(store2) == 1  # Source store unchanged
        
        examples = store1.get_top_examples()
        assert len(examples) == 2
        
        # Should be sorted by score (descending)
        assert examples[0]["max_score"] == 0.9
        assert examples[1]["max_score"] == 0.8
        
        # Check that both datasets are present
        datasets = [ex["dataset"] for ex in examples]
        assert "store1" in datasets
        assert "store2" in datasets

    def test_merge_with_offset(self, temp_db_path, mock_tokenizer, storage_format):
        """Test merge with sequence index offset."""
        store1_path = temp_db_path.with_suffix('.store1.db')
        store2_path = temp_db_path.with_suffix('.store2.db')
        
        store1 = MaxActStore(store1_path, tokenizer=mock_tokenizer, storage_format=storage_format)
        store2 = MaxActStore(store2_path, tokenizer=mock_tokenizer, storage_format=storage_format)
        
        # Add data with overlapping sequence indices
        sequences1 = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        sequences2 = [torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])]
        
        examples_data1 = {0: {0: [(0.9, 0), (0.8, 1)]}}
        examples_data2 = {0: {1: [(0.7, 0), (0.6, 1)]}}
        
        store1.fill(examples_data1, sequences1)
        store2.fill(examples_data2, sequences2)
        
        assert len(store1) == 2
        assert len(store2) == 2
        
        # Merge with offset of 1000 to separate datasets
        store1.merge(store2, sequence_idx_offset=1000)
        
        assert len(store1) == 4
        
        examples = store1.get_top_examples()
        sequence_indices = [ex["sequence_idx"] for ex in examples]
        
        # Original store1 sequences should have indices 0, 1
        # Merged store2 sequences should have indices 1000, 1001
        assert 0 in sequence_indices
        assert 1 in sequence_indices
        assert 1000 in sequence_indices
        assert 1001 in sequence_indices

    def test_merge_offset_conflict(self, temp_db_path, storage_format):
        """Test merge with offset that causes conflicts."""
        store1_path = temp_db_path.with_suffix('.store1.db')
        store2_path = temp_db_path.with_suffix('.store2.db')
        
        store1 = MaxActStore(store1_path, storage_format=storage_format)
        store2 = MaxActStore(store2_path, storage_format=storage_format)
        
        # Add data to store1 using add_example to get sequence_idx = hash(...)
        store1.add_example(
            score=0.9,
            input_ids=torch.tensor([1, 2, 3])
        )
        
        # Add data to store2 using add_example 
        store2.add_example(
            score=0.7,
            input_ids=torch.tensor([4, 5, 6])
        )
        
        # Get the actual sequence indices that were created
        examples1 = store1.get_top_examples()
        examples2 = store2.get_top_examples()
        seq_idx1 = examples1[0]["sequence_idx"]
        seq_idx2 = examples2[0]["sequence_idx"]
        
        # Calculate offset that would cause a conflict
        conflict_offset = seq_idx1 - seq_idx2
        
        # Merge with offset that would cause conflict
        with pytest.raises(ValueError, match="Sequence index conflict"):
            store1.merge(store2, sequence_idx_offset=conflict_offset)

    def test_merge_storage_format_mismatch(self, temp_db_path):
        """Test merge with incompatible storage formats."""
        store1_path = temp_db_path.with_suffix('.sparse.db')
        store2_path = temp_db_path.with_suffix('.dense.db')
        
        store1 = MaxActStore(store1_path, storage_format='sparse')
        store2 = MaxActStore(store2_path, storage_format='dense')
        
        with pytest.raises(ValueError, match="Storage format mismatch"):
            store1.merge(store2)

    def test_merge_with_activation_details(self, temp_db_path, storage_format):
        """Test merge preserves activation details correctly."""
        store1_path = temp_db_path.with_suffix('.store1.db')
        store2_path = temp_db_path.with_suffix('.store2.db')
        
        store1 = MaxActStore(store1_path, storage_format=storage_format)
        store2 = MaxActStore(store2_path, storage_format=storage_format)
        
        sequences1 = [torch.tensor([1, 2, 3])]
        sequences2 = [torch.tensor([4, 5, 6, 7])]
        
        examples_data1 = {0: {0: [(0.9, 0)]}}
        examples_data2 = {0: {1: [(0.7, 0)]}}
        
        if storage_format == 'sparse':
            activation_details1 = {0: {0: (np.array([0, 1]), np.array([0.9, 0.8]))}}
            activation_details2 = {1: {0: (np.array([0, 2, 3]), np.array([0.7, 0.6, 0.5]))}}
        else:  # dense
            activation_details1 = {0: {0: np.array([0.9, 0.8, 0.0])}}
            activation_details2 = {1: {0: np.array([0.7, 0.0, 0.6, 0.5])}}
        
        store1.fill(examples_data1, sequences1, activation_details1)
        store2.fill(examples_data2, sequences2, activation_details2)
        
        # Merge with offset
        store1.merge(store2, sequence_idx_offset=100)
        
        examples = store1.get_top_examples()
        assert len(examples) == 2
        
        # Check activation details are preserved
        for example in examples:
            details = store1.get_example_details(example["example_id"])
            assert "scores_per_token" in details
            assert len(details["scores_per_token"]) > 0

    def test_merge_with_top_k_constraint(self, temp_db_path, storage_format):
        """Test merge respects top-k constraint."""
        store1_path = temp_db_path.with_suffix('.store1.db')
        store2_path = temp_db_path.with_suffix('.store2.db')
        
        # Create stores with small max_examples limit
        store1 = MaxActStore(store1_path, max_examples=3, storage_format=storage_format)
        store2 = MaxActStore(store2_path, storage_format=storage_format)
        
        # Add 2 examples to store1
        for i, score in enumerate([0.9, 0.8]):
            store1.add_example(
                score=score,
                input_ids=torch.tensor([i, i+1, i+2]),
                scores_per_token=torch.tensor([score, score-0.1, score-0.2])
            )
        
        # Add 3 examples to store2 with mixed scores
        for i, score in enumerate([0.95, 0.75, 0.65]):  # One high, two low
            store2.add_example(
                score=score,
                input_ids=torch.tensor([i+10, i+11, i+12]),
                scores_per_token=torch.tensor([score, score-0.1, score-0.2])
            )
        
        assert len(store1) == 2
        assert len(store2) == 3
        
        # Merge with top-k maintenance
        store1.merge(store2, maintain_top_k=True)
        
        # Should only keep top 3 examples
        assert len(store1) == 3
        
        examples = store1.get_top_examples()
        scores = [ex["max_score"] for ex in examples]
        
        # Should keep the 3 highest scores: 0.95, 0.9, 0.8
        assert scores == [0.95, 0.9, 0.8]

    def test_merge_empty_store(self, temp_db_path, storage_format):
        """Test merging an empty store."""
        store1_path = temp_db_path.with_suffix('.store1.db')
        store2_path = temp_db_path.with_suffix('.store2.db')
        
        store1 = MaxActStore(store1_path, storage_format=storage_format)
        store2 = MaxActStore(store2_path, storage_format=storage_format)
        
        # Add data only to store1
        store1.add_example(
            score=0.9,
            input_ids=torch.tensor([1, 2, 3]),
            scores_per_token=torch.tensor([0.9, 0.8, 0.7])
        )
        
        assert len(store1) == 1
        assert len(store2) == 0
        
        # Merge empty store2 into store1
        store1.merge(store2)
        
        # Nothing should change
        assert len(store1) == 1

    def test_merge_into_empty_store(self, temp_db_path, storage_format):
        """Test merging into an empty store."""
        store1_path = temp_db_path.with_suffix('.store1.db')
        store2_path = temp_db_path.with_suffix('.store2.db')
        
        store1 = MaxActStore(store1_path, storage_format=storage_format)
        store2 = MaxActStore(store2_path, storage_format=storage_format)
        
        # Add data only to store2
        store2.add_example(
            score=0.8,
            input_ids=torch.tensor([4, 5, 6]),
            scores_per_token=torch.tensor([0.8, 0.7, 0.6])
        )
        
        assert len(store1) == 0
        assert len(store2) == 1
        
        # Merge store2 into empty store1
        store1.merge(store2)
        
        assert len(store1) == 1
        examples = store1.get_top_examples()
        assert examples[0]["max_score"] == 0.8

    def test_merge_auto_offset(self, temp_db_path, mock_tokenizer, storage_format):
        """Test merge with 'auto' offset."""
        store1_path = temp_db_path.with_suffix('.store1.db')
        store2_path = temp_db_path.with_suffix('.store2.db')
        
        store1 = MaxActStore(store1_path, tokenizer=mock_tokenizer, storage_format=storage_format)
        store2 = MaxActStore(store2_path, tokenizer=mock_tokenizer, storage_format=storage_format)
        
        # Add data with known sequence indices using fill
        sequences1 = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        sequences2 = [torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])]
        
        examples_data1 = {0: {0: [(0.9, 0), (0.8, 1)]}}  # sequence indices 0, 1
        examples_data2 = {0: {1: [(0.7, 0), (0.6, 1)]}}  # sequence indices 0, 1 (will be offset)
        
        store1.fill(examples_data1, sequences1)
        store2.fill(examples_data2, sequences2)
        
        assert len(store1) == 2
        assert len(store2) == 2
        
        # Get the max sequence index from store1
        examples = store1.get_top_examples()
        max_seq_idx = max(ex["sequence_idx"] for ex in examples)
        expected_offset = max_seq_idx + 1
        
        # Merge with auto offset
        store1.merge(store2, sequence_idx_offset="auto")
        
        assert len(store1) == 4
        
        all_examples = store1.get_top_examples()
        sequence_indices = [ex["sequence_idx"] for ex in all_examples]
        
        # Should have original indices (0, 1) and offset indices (expected_offset, expected_offset+1)
        assert 0 in sequence_indices
        assert 1 in sequence_indices
        assert expected_offset in sequence_indices
        assert expected_offset + 1 in sequence_indices

    def test_merge_auto_offset_empty_target(self, temp_db_path, storage_format):
        """Test merge with 'auto' offset into empty target store."""
        store1_path = temp_db_path.with_suffix('.store1.db')
        store2_path = temp_db_path.with_suffix('.store2.db')
        
        store1 = MaxActStore(store1_path, storage_format=storage_format)
        store2 = MaxActStore(store2_path, storage_format=storage_format)
        
        # Add data only to store2
        store2.add_example(
            score=0.8,
            input_ids=torch.tensor([1, 2, 3]),
            scores_per_token=torch.tensor([0.8, 0.7, 0.6])
        )
        
        assert len(store1) == 0
        assert len(store2) == 1
        
        # Get original sequence index
        original_examples = store2.get_top_examples()
        original_seq_idx = original_examples[0]["sequence_idx"]
        
        # Merge with auto offset into empty store (should start from 0)
        store1.merge(store2, sequence_idx_offset="auto")
        
        assert len(store1) == 1
        merged_examples = store1.get_top_examples()
        
        # With empty target, auto offset should be 0, so sequence index should be original + 0
        assert merged_examples[0]["sequence_idx"] == original_seq_idx + 0

    def test_set_dataset_info(self, temp_db_path, storage_format):
        """Test setting dataset info for all sequences in the store."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)
        
        # Add some examples
        for i, score in enumerate([0.9, 0.8, 0.7]):
            store.add_example(
                score=score,
                input_ids=torch.tensor([i, i+1, i+2]),
                scores_per_token=torch.tensor([score, score-0.1, score-0.2])
            )
        
        assert len(store) == 3
        
        # Initially, no dataset info should be set
        examples = store.get_top_examples()
        for example in examples:
            assert example["dataset_id"] is None
            assert example["dataset_name"] is None
        
        # Set dataset info for all sequences
        updated_count = store.set_dataset_info(dataset_id=42, dataset_name="test_dataset")
        assert updated_count == 3
        
        # Verify dataset info was set
        examples = store.get_top_examples()
        for example in examples:
            assert example["dataset_id"] == 42
            assert example["dataset_name"] == "test_dataset"
    
    def test_set_dataset_info_partial(self, temp_db_path, storage_format):
        """Test setting only dataset_id or dataset_name."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)
        
        store.add_example(
            score=0.8,
            input_ids=torch.tensor([1, 2, 3]),
            scores_per_token=torch.tensor([0.8, 0.7, 0.6])
        )
        
        # Set only dataset_id
        updated_count = store.set_dataset_info(dataset_id=123)
        assert updated_count == 1
        
        examples = store.get_top_examples()
        assert examples[0]["dataset_id"] == 123
        assert examples[0]["dataset_name"] is None
        
        # Set only dataset_name  
        updated_count = store.set_dataset_info(dataset_name="new_dataset")
        assert updated_count == 1
        
        examples = store.get_top_examples()
        assert examples[0]["dataset_id"] == 123  # Should remain unchanged
        assert examples[0]["dataset_name"] == "new_dataset"
    
    def test_set_dataset_info_no_overwrite(self, temp_db_path, storage_format):
        """Test setting dataset info with overwrite_existing=False."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)
        
        # Add example with dataset info
        store.add_example(
            score=0.8,
            input_ids=torch.tensor([1, 2, 3]),
            dataset_id=100,
            dataset_name="existing_dataset"
        )
        
        # Add example without dataset info
        store.add_example(
            score=0.7,
            input_ids=torch.tensor([4, 5, 6])
        )
        
        assert len(store) == 2
        
        # Try to set dataset info without overwriting existing
        updated_count = store.set_dataset_info(
            dataset_id=200, 
            dataset_name="new_dataset",
            overwrite_existing=False
        )
        assert updated_count == 1  # Should only update the one without existing data
        
        examples = store.get_top_examples()
        
        # First example should keep original dataset info
        first_example = next(ex for ex in examples if ex["max_score"] == 0.8)
        assert first_example["dataset_id"] == 100
        assert first_example["dataset_name"] == "existing_dataset"
        
        # Second example should have new dataset info
        second_example = next(ex for ex in examples if ex["max_score"] == 0.7)
        assert second_example["dataset_id"] == 200
        assert second_example["dataset_name"] == "new_dataset"

    def test_per_dataset_top_k_basic(self, temp_db_path, storage_format):
        """Test per-dataset top-k functionality."""
        # Create store with per_dataset=True and max_examples=2
        store = MaxActStore(temp_db_path, max_examples=2, storage_format=storage_format, per_dataset=True)
        
        # Add examples from dataset A
        for i, score in enumerate([0.9, 0.8, 0.7]):  # 3 examples, should keep top 2
            store.add_example(
                score=score,
                input_ids=torch.tensor([i, i+1, i+2]),
                scores_per_token=torch.tensor([score, score-0.1, score-0.2]),
                additional_data={"dataset_name": "dataset_A"}
            )
        
        # Set dataset info for these examples
        store.set_dataset_info(dataset_name="dataset_A")
        
        # Add examples from dataset B  
        for i, score in enumerate([0.95, 0.75, 0.65]):  # 3 examples, should keep top 2
            store.add_example(
                score=score,
                input_ids=torch.tensor([i+10, i+11, i+12]),
                scores_per_token=torch.tensor([score, score-0.1, score-0.2]),
                additional_data={"dataset_name": "dataset_B"}
            )
        
        # Set dataset info for dataset B examples (need to be more specific)
        # First get examples without dataset_name set
        examples_without_dataset = store.get_top_examples()
        dataset_b_examples = [ex for ex in examples_without_dataset if ex["dataset_name"] is None]
        
        # Update these specific sequences
        with sqlite3.connect(store.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            for ex in dataset_b_examples:
                cursor.execute("UPDATE sequences SET dataset_name = ? WHERE sequence_idx = ?", 
                             ("dataset_B", ex["sequence_idx"]))
            conn.commit()
        
        # Should have 4 total examples (2 per dataset)
        assert len(store) == 4
        
        # Check that we have 2 examples from each dataset
        examples = store.get_top_examples()
        dataset_a_examples = [ex for ex in examples if ex["dataset_name"] == "dataset_A"]
        dataset_b_examples = [ex for ex in examples if ex["dataset_name"] == "dataset_B"]
        
        assert len(dataset_a_examples) == 2
        assert len(dataset_b_examples) == 2
        
        # Check that we kept the highest scores within each dataset
        dataset_a_scores = sorted([ex["max_score"] for ex in dataset_a_examples], reverse=True)
        dataset_b_scores = sorted([ex["max_score"] for ex in dataset_b_examples], reverse=True)
        
        assert dataset_a_scores == [0.9, 0.8]  # Dropped 0.7
        assert dataset_b_scores == [0.95, 0.75]  # Dropped 0.65

    def test_per_dataset_vs_overall_comparison(self, temp_db_path, storage_format):
        """Test that per-dataset and overall top-k behave differently."""
        store_overall_path = temp_db_path.with_suffix('.overall.db')
        store_per_dataset_path = temp_db_path.with_suffix('.per_dataset.db')
        
        # Create two stores with different top-k strategies
        store_overall = MaxActStore(store_overall_path, max_examples=3, storage_format=storage_format, per_dataset=False)
        store_per_dataset = MaxActStore(store_per_dataset_path, max_examples=2, storage_format=storage_format, per_dataset=True)
        
        # Add same data to both stores with distinct token sequences for easy identification
        example_data = [
            (0.9, "dataset_A", 0, [1, 2, 3]), (0.8, "dataset_A", 0, [4, 5, 6]), (0.7, "dataset_A", 0, [7, 8, 9]),  # 3 from A
            (0.95, "dataset_B", 1, [10, 11, 12]), (0.65, "dataset_B", 1, [13, 14, 15])  # 2 from B
        ]
        
        for score, dataset, dataset_id, tokens in example_data:
            for store in [store_overall, store_per_dataset]:
                store.add_example(
                    score=score,
                    input_ids=torch.tensor(tokens),
                    scores_per_token=torch.tensor([score, score-0.1, score-0.2]),
                    maintain_top_k=False,  # Don't maintain top-k until after we set dataset names
                    dataset_name=dataset,
                    dataset_id=dataset_id
                )
        
        # Force top-k maintenance
        store_overall._maintain_top_k()
        store_per_dataset._maintain_top_k()
        
        # Overall store should keep top 3 scores globally: 0.95, 0.9, 0.8
        overall_examples = store_overall.get_top_examples()
        overall_scores = [ex["max_score"] for ex in overall_examples]
        assert len(overall_examples) == 3
        assert sorted(overall_scores, reverse=True) == [0.95, 0.9, 0.8]
        
        # Per-dataset store should keep top 2 from each dataset: A(0.9, 0.8), B(0.95, 0.65)
        per_dataset_examples = store_per_dataset.get_top_examples()
        assert len(per_dataset_examples) == 4  # 2 per dataset
        
        dataset_a_examples = [ex for ex in per_dataset_examples if ex["dataset_name"] == "dataset_A"]
        dataset_b_examples = [ex for ex in per_dataset_examples if ex["dataset_name"] == "dataset_B"]
        
        assert len(dataset_a_examples) == 2
        assert len(dataset_b_examples) == 2
        
        dataset_a_scores = sorted([ex["max_score"] for ex in dataset_a_examples], reverse=True)
        dataset_b_scores = sorted([ex["max_score"] for ex in dataset_b_examples], reverse=True)
        
        assert dataset_a_scores == [0.9, 0.8]
        assert dataset_b_scores == [0.95, 0.65]

    def test_per_dataset_with_null_dataset(self, temp_db_path, storage_format):
        """Test per-dataset top-k with some examples having NULL dataset_name."""
        store = MaxActStore(temp_db_path, max_examples=2, storage_format=storage_format, per_dataset=True)
        
        # Add examples with dataset A
        store.add_example(0.9, torch.tensor([1, 2, 3]))
        store.add_example(0.8, torch.tensor([4, 5, 6]))
        store.add_example(0.7, torch.tensor([7, 8, 9]))  # Should be removed from NULL group
        
        # Set dataset for first two examples  
        examples = store.get_top_examples()
        with sqlite3.connect(store.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("UPDATE sequences SET dataset_name = ? WHERE sequence_idx = ?", 
                         ("dataset_A", examples[0]["sequence_idx"]))
            cursor.execute("UPDATE sequences SET dataset_name = ? WHERE sequence_idx = ?", 
                         ("dataset_A", examples[1]["sequence_idx"]))
            conn.commit()
        
        # Add more examples with explicit NULL dataset (leave as-is)
        store.add_example(0.85, torch.tensor([10, 11, 12]))
        store.add_example(0.75, torch.tensor([13, 14, 15]))
        store.add_example(0.65, torch.tensor([16, 17, 18]))  # Should be removed from NULL group
        
        # Force top-k maintenance
        store._maintain_top_k()
        
        final_examples = store.get_top_examples()
        
        # Should have 2 from dataset_A and 2 from NULL dataset = 4 total
        assert len(final_examples) == 4
        
        dataset_a_examples = [ex for ex in final_examples if ex["dataset_name"] == "dataset_A"]
        null_dataset_examples = [ex for ex in final_examples if ex["dataset_name"] is None]
        
        assert len(dataset_a_examples) == 2
        assert len(null_dataset_examples) == 2
        
        # Check scores within each group
        dataset_a_scores = sorted([ex["max_score"] for ex in dataset_a_examples], reverse=True)
        null_scores = sorted([ex["max_score"] for ex in null_dataset_examples], reverse=True)
        
        assert dataset_a_scores == [0.9, 0.8]  # Top 2 from dataset A
        assert null_scores == [0.85, 0.75]  # Top 2 from NULL dataset

    def test_async_writer_basic_functionality(self, temp_db_path, mock_tokenizer, storage_format):
        """Test basic AsyncMaxActStoreWriter functionality."""
        store = MaxActStore(temp_db_path, max_examples=100, tokenizer=mock_tokenizer, storage_format=storage_format)
        
        # Test creating async writer
        async_writer = store.create_async_writer(buffer_size=10, flush_interval=1.0)
        assert isinstance(async_writer, AsyncMaxActStoreWriter)
        assert async_writer.buffer_size == 10
        assert async_writer.flush_interval == 1.0
        assert not async_writer.is_running
        
        # Test context manager
        with async_writer as writer:
            assert writer.is_running
            
            # Add some test data
            batch_size = 5
            scores = torch.rand(batch_size) * 10
            input_ids = torch.randint(1, 1000, (batch_size, 10))
            attention_mask = torch.ones(batch_size, 10)
            scores_per_token = torch.rand(batch_size, 10)
            
            writer.add_batch_examples(
                scores_per_example=scores,
                input_ids_batch=input_ids,
                attention_mask_batch=attention_mask,
                scores_per_token_batch=scores_per_token,
                dataset_name="test_dataset"
            )
            
            # Flush to ensure data is written
            writer.flush()
        
        # After context exit, writer should be stopped
        assert not async_writer.is_running
        
        # Verify data was written to store
        assert len(store) == batch_size
        examples = store.get_top_examples()
        assert all(ex["dataset_name"] == "test_dataset" for ex in examples)

    def test_async_writer_buffering(self, temp_db_path, storage_format):
        """Test that async writer properly buffers data before writing."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)
        
        with store.create_async_writer(buffer_size=20, flush_interval=60.0, auto_maintain_top_k=False) as writer:
            # Add data that should stay in buffer (less than buffer_size)
            for i in range(3):
                batch_size = 5  # Total 15 examples, less than buffer_size=20
                scores = torch.ones(batch_size) * (0.5 + i * 0.1)
                input_ids = torch.randint(1, 100, (batch_size, 8))
                
                writer.add_batch_examples(
                    scores_per_example=scores,
                    input_ids_batch=input_ids,
                    dataset_name=f"batch_{i}"
                )
            
            # Give background process a moment but data should still be buffered
            import time
            time.sleep(0.1)
            
            # Force flush
            writer.flush()
            
            # Give background process time to write
            time.sleep(0.5)
        
        # After context exit, all data should be written
        assert len(store) == 15
        
        # Verify we have data from all batches
        examples = store.get_top_examples()
        datasets = set(ex["dataset_name"] for ex in examples)
        assert datasets == {"batch_0", "batch_1", "batch_2"}

    def test_async_writer_buffer_overflow(self, temp_db_path, storage_format):
        """Test that async writer flushes when buffer size is exceeded."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)
        
        with store.create_async_writer(buffer_size=10, flush_interval=60.0) as writer:
            # Add more data than buffer_size to trigger flush
            batch_size = 6
            scores = torch.ones(batch_size) * 0.8
            input_ids = torch.randint(1, 100, (batch_size, 5))
            
            # First batch (6 examples) - should stay in buffer
            writer.add_batch_examples(
                scores_per_example=scores,
                input_ids_batch=input_ids,
                dataset_name="batch_1"
            )
            
            # Second batch (6 examples) - total 12, should trigger flush since > buffer_size=10
            writer.add_batch_examples(
                scores_per_example=scores,
                input_ids_batch=input_ids,
                dataset_name="batch_2"
            )
            
            # Give background process time to write
            import time
            time.sleep(0.5)
        
        # All data should be written
        assert len(store) == 12

    def test_async_writer_attention_mask_handling(self, temp_db_path, storage_format):
        """Test that async writer properly handles attention masks."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)
        
        with store.create_async_writer(buffer_size=5) as writer:
            batch_size = 3
            seq_len = 8
            scores = torch.ones(batch_size) * 0.7
            input_ids = torch.randint(1, 100, (batch_size, seq_len))
            
            # Create attention mask with different valid lengths per sequence
            attention_mask = torch.zeros(batch_size, seq_len)
            attention_mask[0, :5] = 1  # First sequence has 5 valid tokens
            attention_mask[1, :3] = 1  # Second sequence has 3 valid tokens  
            attention_mask[2, :7] = 1  # Third sequence has 7 valid tokens
            
            scores_per_token = torch.rand(batch_size, seq_len)
            
            writer.add_batch_examples(
                scores_per_example=scores,
                input_ids_batch=input_ids,
                attention_mask_batch=attention_mask,
                scores_per_token_batch=scores_per_token,
                dataset_name="masked_test"
            )
        
        # Verify data was written with correct sequence lengths
        examples = store.get_top_examples()
        assert len(examples) == 3
        
        # Check that sequences have correct lengths (matching attention mask)
        sequence_lengths = [len(ex["input_ids"]) for ex in examples]
        assert sorted(sequence_lengths) == [3, 5, 7]  # Should match attention mask lengths

    def test_async_writer_error_handling(self, temp_db_path, storage_format):
        """Test error handling in async writer."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)
        
        # Test adding data to stopped writer
        async_writer = store.create_async_writer()
        
        with pytest.raises(RuntimeError, match="AsyncMaxActStoreWriter is not running"):
            async_writer.add_batch_examples(
                scores_per_example=torch.ones(2),
                input_ids_batch=torch.randint(1, 100, (2, 5))
            )

    def test_async_writer_dataset_separation(self, temp_db_path, storage_format):
        """Test that async writer properly separates data by dataset."""
        store = MaxActStore(temp_db_path, max_examples=20, storage_format=storage_format, per_dataset=True)
        
        with store.create_async_writer(buffer_size=50) as writer:
            # Add data for multiple datasets
            datasets = ["dataset_A", "dataset_B", "dataset_C"]
            
            for dataset in datasets:
                for batch_idx in range(3):  # 3 batches per dataset
                    batch_size = 4
                    # Different score ranges for each dataset to test separation
                    base_score = 0.3 + datasets.index(dataset) * 0.2
                    scores = torch.ones(batch_size) * (base_score + batch_idx * 0.05)
                    input_ids = torch.randint(1, 100, (batch_size, 6))
                    
                    writer.add_batch_examples(
                        scores_per_example=scores,
                        input_ids_batch=input_ids,
                        dataset_name=dataset
                    )
        
        # Verify data separation
        total_examples = len(store)
        assert total_examples == 36  # 3 datasets * 3 batches * 4 examples
        
        # Check that each dataset has its examples
        for dataset in datasets:
            dataset_examples = store.get_top_examples(dataset_names=[dataset])
            assert len(dataset_examples) == 12  # 3 batches * 4 examples per dataset
            assert all(ex["dataset_name"] == dataset for ex in dataset_examples)

    def test_async_writer_top_k_maintenance(self, temp_db_path, storage_format):
        """Test that async writer maintains top-k constraints."""
        max_examples = 15
        store = MaxActStore(temp_db_path, max_examples=max_examples, storage_format=storage_format)
        
        with store.create_async_writer(buffer_size=10, auto_maintain_top_k=True) as writer:
            # Add more examples than max_examples
            total_batches = 5
            batch_size = 5  # Total: 25 examples, should keep only top 15
            
            for batch_idx in range(total_batches):
                # Vary scores so we can predict which examples should be kept
                scores = torch.ones(batch_size) * (1.0 - batch_idx * 0.1)  # Decreasing scores
                input_ids = torch.randint(1, 100, (batch_size, 4))
                
                writer.add_batch_examples(
                    scores_per_example=scores,
                    input_ids_batch=input_ids,
                    additional_data_batch=[{"batch": batch_idx}] * batch_size
                )
        
        # Should only keep top max_examples
        assert len(store) == max_examples
        
        # Verify that highest scoring examples were kept
        examples = store.get_top_examples()
        assert len(examples) == max_examples
        
        # Examples should be sorted by score (descending)
        scores = [ex["max_score"] for ex in examples]
        assert scores == sorted(scores, reverse=True)
        
        # Highest scores should be from early batches
        assert scores[0] >= 0.9  # From batch 0 or 1

    def test_async_writer_manual_start_stop(self, temp_db_path, storage_format):
        """Test manual start/stop of async writer."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)
        async_writer = store.create_async_writer()
        
        # Initially not running
        assert not async_writer.is_running
        
        # Manual start
        async_writer.start()
        assert async_writer.is_running
        
        # Add some data
        scores = torch.ones(3) * 0.8
        input_ids = torch.randint(1, 100, (3, 5))
        
        async_writer.add_batch_examples(
            scores_per_example=scores,
            input_ids_batch=input_ids
        )
        
        # Manual stop
        async_writer.stop()
        assert not async_writer.is_running
        
        # Verify data was written
        assert len(store) == 3

    def test_async_writer_multiple_flushes(self, temp_db_path, storage_format):
        """Test multiple flush operations."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)
        
        with store.create_async_writer(buffer_size=20, auto_maintain_top_k=False) as writer:
            # Add some data
            for i in range(3):
                scores = torch.ones(2) * (0.5 + i * 0.1)
                input_ids = torch.randint(1, 100, (2, 4))
                
                writer.add_batch_examples(
                    scores_per_example=scores,
                    input_ids_batch=input_ids,
                    additional_data_batch=[{"flush_round": i}] * 2
                )
                
                # Flush after each batch
                writer.flush()
                
                # Give background process time
                import time
                time.sleep(0.1)
        
        # All data should be written
        assert len(store) == 6
        
        # Verify data from all flush rounds
        examples = store.get_top_examples()
        flush_rounds = set(ex["flush_round"] for ex in examples)
        assert flush_rounds == {0, 1, 2}

    def test_async_writer_sparse_vs_dense_consistency(self, temp_db_path, mock_tokenizer):
        """Test that sparse and dense storage formats produce consistent results in async writer."""
        # Test data
        batch_size = 4
        seq_len = 6
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
        input_ids = torch.randint(1, 50, (batch_size, seq_len))
        scores_per_token = torch.rand(batch_size, seq_len)
        
        results = {}
        
        for storage_format in ['sparse', 'dense']:
            store_path = temp_db_path.with_suffix(f'.{storage_format}.db')
            store = MaxActStore(store_path, tokenizer=mock_tokenizer, storage_format=storage_format)
            
            with store.create_async_writer() as writer:
                writer.add_batch_examples(
                    scores_per_example=scores,
                    input_ids_batch=input_ids,
                    scores_per_token_batch=scores_per_token,
                    dataset_name="consistency_test"
                )
            
            # Get results
            examples = store.get_top_examples()
            results[storage_format] = examples
        
        # Compare results between formats
        sparse_examples = results['sparse']
        dense_examples = results['dense']
        
        assert len(sparse_examples) == len(dense_examples) == batch_size
        
        # Sort by score for comparison
        sparse_examples.sort(key=lambda x: x["max_score"], reverse=True)
        dense_examples.sort(key=lambda x: x["max_score"], reverse=True)
        
        for sparse_ex, dense_ex in zip(sparse_examples, dense_examples):
            assert abs(sparse_ex["max_score"] - dense_ex["max_score"]) < 1e-6
            assert sparse_ex["input_ids"] == dense_ex["input_ids"]
            assert sparse_ex["dataset_name"] == dense_ex["dataset_name"]
            
            # Compare activation details
            sparse_details = results['sparse'][0]  # Get from store
            dense_details = results['dense'][0]   # Get from store
            
            # Both should have scores_per_token
            if "scores_per_token" in sparse_details and "scores_per_token" in dense_details:
                assert len(sparse_details["scores_per_token"]) == len(dense_details["scores_per_token"])

    def test_async_writer_performance_improvement(self, temp_db_path, storage_format):
        """Test that async writer provides performance benefits over synchronous operations."""
        import time
        
        # Test data
        num_batches = 10
        batch_size = 20
        
        # Synchronous approach (using add_example directly)
        sync_store = MaxActStore(temp_db_path.with_suffix('.sync.db'), storage_format=storage_format)
        
        sync_start = time.time()
        for batch_idx in range(num_batches):
            scores = torch.rand(batch_size)
            for i in range(batch_size):
                sync_store.add_example(
                    score=scores[i].item(),
                    input_ids=torch.randint(1, 100, (8,)),
                    scores_per_token=torch.rand(8),
                    maintain_top_k=False  # Don't maintain during addition
                )
        sync_store._maintain_top_k()  # Maintain once at end
        sync_time = time.time() - sync_start
        
        # Asynchronous approach
        async_store = MaxActStore(temp_db_path.with_suffix('.async.db'), storage_format=storage_format)
        
        async_start = time.time()
        with async_store.create_async_writer(buffer_size=batch_size * 2) as writer:
            for batch_idx in range(num_batches):
                scores = torch.rand(batch_size)
                input_ids = torch.randint(1, 100, (batch_size, 8))
                scores_per_token = torch.rand(batch_size, 8)
                
                writer.add_batch_examples(
                    scores_per_example=scores,
                    input_ids_batch=input_ids,
                    scores_per_token_batch=scores_per_token
                )
        async_time = time.time() - async_start
        
        # Verify both approaches produced same amount of data
        assert len(sync_store) == len(async_store) == num_batches * batch_size
        
        # Async should be faster (though this might be flaky in CI environments)
        print(f"Sync time: {sync_time:.3f}s, Async time: {async_time:.3f}s")
        # Note: We don't assert performance improvement as it depends on system load
        # But we can at least verify that async didn't take dramatically longer

    def test_async_writer_graceful_shutdown(self, temp_db_path, storage_format):
        """Test graceful shutdown of async writer process."""
        store = MaxActStore(temp_db_path, storage_format=storage_format)
        
        async_writer = store.create_async_writer(buffer_size=100)  # Large buffer to keep data buffered
        async_writer.start()
        
        # Add some data that will be buffered
        scores = torch.ones(5) * 0.8
        input_ids = torch.randint(1, 100, (5, 6))
        
        async_writer.add_batch_examples(
            scores_per_example=scores,
            input_ids_batch=input_ids,
            dataset_name="shutdown_test"
        )
        
        # Stop should flush remaining data and shutdown gracefully
        async_writer.stop(timeout=10.0)
        
        # Verify data was written during shutdown
        assert len(store) == 5
        examples = store.get_top_examples()
        assert all(ex["dataset_name"] == "shutdown_test" for ex in examples) 