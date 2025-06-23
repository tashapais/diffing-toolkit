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

from src.utils.max_act_store import MaxActStore

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
        import sqlite3
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
        with pytest.raises(AssertionError):
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
        assert len(details["scores_per_token"]) == 2  # Only non-zero positions
        assert np.isclose(details["scores_per_token"][0], 0.8)
        assert np.isclose(details["scores_per_token"][1], 0.6)
        
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
        assert len(details["scores_per_token"]) == 2
        assert np.isclose(details["scores_per_token"][0], 0.8)
        assert np.isclose(details["scores_per_token"][1], 0.6)

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
        details_sparse = store_sparse.get_example_details(1)
        details_dense = store_dense.get_example_details(1)
        
        # Sparse format stores only positions and values
        assert len(details_sparse["scores_per_token"]) == 5
        assert len(details_sparse["positions"]) == 5
        
        # Dense format stores full array
        assert len(details_dense["scores_per_token"]) == 5
        assert len(details_dense["positions"]) == 5  # All non-zero positions
        
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