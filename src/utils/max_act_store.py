"""
SQLite-based storage for maximum activating examples.

This module provides persistent storage for maximum activating examples across different
diffing methods, supporting both bulk loading of pre-sorted data and real-time top-k management.
"""

import sqlite3
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from tqdm import tqdm
from loguru import logger
import multiprocessing as mp
import queue
import time
import threading
from dataclasses import dataclass
from enum import Enum


class WriteCommand(Enum):
    ADD_BATCH = "add_batch"
    MAINTAIN_TOP_K = "maintain_top_k"
    SHUTDOWN = "shutdown"
    FLUSH = "flush"

def get_per_example_value(value, example_idx: int, batch_size: int) -> Optional[int]:
    if value is None:
        return None
    elif isinstance(value, (int, float)):
        return int(value)
    elif isinstance(value, (np.ndarray, torch.Tensor)):
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        assert len(value) == batch_size, f"Array length {len(value)} doesn't match batch size {batch_size}"
        return int(value[example_idx])
    else:
        raise TypeError(f"Unsupported type: {type(value)}")

@dataclass
class BatchData:
    """Data structure for batch examples to be written."""
    scores_per_example: np.ndarray
    input_ids_batch: List[List[int]]  # List of token lists (after attention mask applied)
    scores_per_token_batch: Optional[List[Optional[np.ndarray]]]
    additional_data_batch: Optional[List[dict]]
    latent_idx: Optional[Union[int, np.ndarray, torch.Tensor]]  # Single int or array of shape [batch_size]
    quantile_idx: Optional[Union[int, np.ndarray, torch.Tensor]]  # Single int or array of shape [batch_size]
    dataset_name: Optional[str]
    dataset_id: Optional[int]

    def get_per_example_latent_idx(self, example_idx: int, batch_size: int) -> Optional[int]:
        """
        Get latent_idx for a specific example in the batch.
        
        Args:
            example_idx: Index of the example in the batch
            batch_size: Total batch size for validation
            
        Returns:
            Latent index for the specific example, or None if not set
        """
        return get_per_example_value(self.latent_idx, example_idx, batch_size)
    
    def get_per_example_quantile_idx(self, example_idx: int, batch_size: int) -> Optional[int]:
        """
        Get quantile_idx for a specific example in the batch.
        
        Args:
            example_idx: Index of the example in the batch
            batch_size: Total batch size for validation
            
        Returns:
            Quantile index for the specific example, or None if not set
        """
        return get_per_example_value(self.quantile_idx, example_idx, batch_size)
    
    def _get_per_example_value(self, value: Optional[Union[int, np.ndarray, torch.Tensor]], 
                              example_idx: int, batch_size: int) -> Optional[int]:
        """
        Helper method to get per-example values from either scalar or array.
        
        This method supports both:
        1. Single integer values that apply to the entire batch
        2. Arrays/tensors of shape [batch_size] with different values per example
        
        Args:
            value: The value to extract from (single int, array, or None)
            example_idx: Index of the example in the batch
            batch_size: Total batch size for validation
            
        Returns:
            The value for the specific example
        """
        if value is None:
            return None
        elif isinstance(value, (int, float)):
            # Single value applies to all examples in batch
            return int(value)
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            # Array of per-example values
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            
            assert len(value) == batch_size, f"Array length {len(value)} doesn't match batch size {batch_size}"
            assert 0 <= example_idx < batch_size, f"Example index {example_idx} out of bounds for batch size {batch_size}"
            
            return int(value[example_idx])
        else:
            raise TypeError(f"Unsupported type for latent/quantile index: {type(value)}. Expected int, np.ndarray, or torch.Tensor")


@dataclass
class WriteRequest:
    """Request sent to background writer process."""
    command: WriteCommand
    data: Optional[BatchData] = None
    response_queue: Optional[mp.Queue] = None


class AsyncMaxActStoreWriter:
    """
    Asynchronous writer for MaxActStore that handles database writes in a background process.
    
    This dramatically improves performance by:
    1. Buffering multiple batches before writing to database
    2. Performing database I/O in separate process
    3. Deferring top-k maintenance until necessary
    """
    
    def __init__(self, db_path: Path, max_examples: Optional[int] = None, 
                 tokenizer=None, storage_format: str = 'sparse', 
                 per_dataset: bool = False, buffer_size: int = 1000, 
                 flush_interval: float = 30.0, auto_maintain_top_k: bool = True):
        """
        Initialize async writer.
        
        Args:
            db_path: Path to SQLite database file
            max_examples: Maximum number of examples to keep
            tokenizer: Optional tokenizer for text decoding
            storage_format: Storage format ('sparse' or 'dense')
            per_dataset: If True, maintain max_examples per dataset
            buffer_size: Number of examples to buffer before writing
            flush_interval: Time interval (seconds) to force flush
            auto_maintain_top_k: Whether to automatically maintain top-k
        """
        self.db_path = Path(db_path)
        self.max_examples = max_examples
        self.tokenizer = tokenizer
        self.storage_format = storage_format
        self.per_dataset = per_dataset
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.auto_maintain_top_k = auto_maintain_top_k
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Communication with background process
        self.request_queue = mp.Queue()
        self.error_queue = mp.Queue()
        self.writer_process = None
        self.is_running = False
        
        # Buffer for accumulating batches
        self.buffer = []
        self.buffer_count = 0
        self.last_flush_time = time.time()
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def start(self):
        """Start the background writer process."""
        if self.is_running:
            return
            
        self.writer_process = mp.Process(
            target=self._writer_worker,
            args=(
                self.request_queue, self.error_queue, self.db_path,
                self.max_examples, self.storage_format, self.per_dataset,
                self.tokenizer
            )
        )
        self.writer_process.start()
        self.is_running = True
        logger.info(f"Started async writer process for {self.db_path}")
    
    def stop(self, timeout: float = 60.0):
        """Stop the background writer process and wait for completion."""
        if not self.is_running:
            return
            
        try:
            # Flush any remaining data
            self._flush_buffer()
            
            # Send shutdown command
            self.request_queue.put(WriteRequest(WriteCommand.SHUTDOWN))
            
            # Wait for process to finish
            if self.writer_process:
                self.writer_process.join(timeout=timeout)
                if self.writer_process.is_alive():
                    logger.warning("Writer process didn't finish gracefully, terminating...")
                    self.writer_process.terminate()
                    self.writer_process.join(timeout=5.0)
                    if self.writer_process.is_alive():
                        logger.error("Writer process couldn't be terminated, killing...")
                        self.writer_process.kill()
        finally:
            self.is_running = False
            self.writer_process = None
            logger.info("Stopped async writer process")
    
    def add_batch_examples(self, scores_per_example: torch.Tensor,
                          input_ids_batch: torch.Tensor,
                          attention_mask_batch: Optional[torch.Tensor] = None,
                          scores_per_token_batch: Optional[torch.Tensor] = None,
                          additional_data_batch: Optional[List[dict]] = None,
                          latent_idx: Optional[Union[int, torch.Tensor, np.ndarray]] = None,
                          quantile_idx: Optional[Union[int, torch.Tensor, np.ndarray]] = None,
                          dataset_name: Optional[str] = None,
                          dataset_id: Optional[int] = None) -> None:
        """
        Add batch examples to buffer for background writing.
        
        Args:
            scores_per_example: Scores tensor [batch_size]
            input_ids_batch: Token IDs tensor [batch_size, seq_len]
            attention_mask_batch: Attention masks [batch_size, seq_len] (optional)
            scores_per_token_batch: Per-token scores [batch_size, seq_len] (optional)
            additional_data_batch: List of additional metadata dicts (optional)
            latent_idx: Latent feature index (single int or array of shape [batch_size])
            quantile_idx: Quantile index (single int or array of shape [batch_size])
            dataset_name: Dataset name (optional)
            dataset_id: Dataset ID (optional)
        """
        if not self.is_running:
            raise RuntimeError("AsyncMaxActStoreWriter is not running. Call start() first.")
        
        # Check for errors from background process
        self._check_for_errors()
        
        batch_size = scores_per_example.shape[0]
        
        # Efficiently process the batch based on whether attention mask is provided
        if attention_mask_batch is None:
            # No attention mask - process entire batch efficiently
            processed_input_ids = input_ids_batch.cpu().tolist()
            if scores_per_token_batch is not None:
                # Convert entire batch to CPU once, then split into individual arrays
                scores_cpu = scores_per_token_batch.cpu().numpy()
                processed_scores_per_token = [scores_cpu[i] for i in range(batch_size)]
            else:
                processed_scores_per_token = [None] * batch_size
        else:
            # Attention mask provided - need to process each example individually
            processed_input_ids = []
            processed_scores_per_token = []
            
            for i in range(batch_size):
                input_ids = input_ids_batch[i]
                valid_mask = attention_mask_batch[i].bool()
                input_ids = input_ids[valid_mask]
                
                processed_input_ids.append(input_ids.cpu().tolist())
                
                if scores_per_token_batch is not None:
                    scores_per_token = scores_per_token_batch[i][valid_mask]
                    processed_scores_per_token.append(scores_per_token.cpu().numpy())
                else:
                    processed_scores_per_token.append(None)
        
        # Create batch data
        batch_data = BatchData(
            scores_per_example=scores_per_example.cpu().numpy(),
            input_ids_batch=processed_input_ids,
            scores_per_token_batch=processed_scores_per_token if any(x is not None for x in processed_scores_per_token) else None,
            additional_data_batch=additional_data_batch,
            latent_idx=latent_idx,
            quantile_idx=quantile_idx,
            dataset_name=dataset_name,
            dataset_id=dataset_id
        )
        
        with self._lock:
            self.buffer.append(batch_data)
            self.buffer_count += batch_size
            
            # Check if we should flush
            should_flush = (
                self.buffer_count >= self.buffer_size or
                time.time() - self.last_flush_time >= self.flush_interval
            )
            
            if should_flush:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush the current buffer to the background writer (thread-unsafe)."""
        if not self.buffer:
            return
            
        # Send all buffered data to background process
        for batch_data in self.buffer:
            request = WriteRequest(WriteCommand.ADD_BATCH, data=batch_data)
            self.request_queue.put(request)
        
        # Clear buffer
        self.buffer.clear()
        self.buffer_count = 0
        self.last_flush_time = time.time()
        
        # Trigger top-k maintenance if enabled
        if self.auto_maintain_top_k:
            self.request_queue.put(WriteRequest(WriteCommand.MAINTAIN_TOP_K))
    
    def flush(self):
        """Force flush any buffered data."""
        with self._lock:
            self._flush_buffer()
    
    def _check_for_errors(self):
        """Check if background process has reported any errors."""
        try:
            error = self.error_queue.get_nowait()
            raise RuntimeError(f"Background writer error: {error}")
        except queue.Empty:
            pass
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    @staticmethod
    def _writer_worker(request_queue: mp.Queue, error_queue: mp.Queue, 
                      db_path: Path, max_examples: Optional[int], 
                      storage_format: str, per_dataset: bool, tokenizer):
        """Background worker process that handles database writes."""
        try:
            # Create MaxActStore instance in this process
            store = MaxActStore(
                db_path=db_path,
                max_examples=max_examples,
                tokenizer=tokenizer,
                storage_format=storage_format,
                per_dataset=per_dataset
            )
            
            logger.info(f"Background writer started for {db_path}")
            
            while True:
                try:
                    request = request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if request.command == WriteCommand.SHUTDOWN:
                    logger.info("Background writer received shutdown command")
                    break
                elif request.command == WriteCommand.ADD_BATCH:
                    store._process_batch_data(request.data)
                elif request.command == WriteCommand.MAINTAIN_TOP_K:
                    store._maintain_top_k()
                elif request.command == WriteCommand.FLUSH:
                    # Just ensure any pending writes are committed
                    pass
                
        except Exception as e:
            logger.error(f"Background writer error: {e}")
            error_queue.put(str(e))
        finally:
            logger.info("Background writer process finished")


class MaxActStore:
    """
    SQLite-based storage for maximum activating examples.
    
    Supports two main use cases:
    1. Bulk loading of pre-sorted examples (e.g., quantile examples)
    2. Real-time storage with top-k management (e.g., during model diffing)
    """
    
    def __init__(self, db_path: Path, max_examples: Optional[int] = None, tokenizer=None, storage_format: Optional[str] = 'sparse', per_dataset: bool = False):
        """
        Initialize the store.
        
        Args:
            db_path: Path to SQLite database file
            max_examples: Maximum number of examples to keep (None for unlimited)
            tokenizer: Optional tokenizer for text decoding
            storage_format: Storage format for activation details ('sparse', 'dense', or None to read from existing config)
            per_dataset: If True, maintain max_examples per dataset rather than overall
        """
        self.db_path = Path(db_path)
        self.tokenizer = tokenizer
        self.per_dataset = per_dataset
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database and handle config
        self._handle_config(max_examples, storage_format)
        self._init_database()
    
    def _get_connection(self):
        """Get a connection to the database."""
        return sqlite3.connect(self.db_path)
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Create sequences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sequences (
                    sequence_idx INTEGER PRIMARY KEY,
                    token_ids BLOB NOT NULL,
                    text TEXT,
                    sequence_length INTEGER NOT NULL,
                    dataset_id INTEGER DEFAULT NULL,
                    dataset_name TEXT DEFAULT NULL
                )
            """)
            
            # Create examples table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS examples (
                    example_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sequence_idx INTEGER NOT NULL,
                    score REAL NOT NULL,
                    latent_idx INTEGER DEFAULT NULL,
                    quantile_idx INTEGER DEFAULT NULL,
                    metadata TEXT,
                    FOREIGN KEY (sequence_idx) REFERENCES sequences(sequence_idx)
                )
            """)
            
            if self.storage_format == 'sparse':
                # Create activation details table (sparse format)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS activation_details (
                        example_id INTEGER PRIMARY KEY,
                        positions BLOB,
                        activation_values BLOB,
                        FOREIGN KEY (example_id) REFERENCES examples(example_id)
                    )
                """)
            else:
                # Create activation details table (dense format)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS activation_details (
                        example_id INTEGER PRIMARY KEY,
                        activation_values BLOB NOT NULL,
                        FOREIGN KEY (example_id) REFERENCES examples(example_id)
                    )
                """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_examples_score ON examples(score DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_examples_latent ON examples(latent_idx)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_examples_quantile ON examples(quantile_idx)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sequences_dataset ON sequences(dataset_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sequences_dataset_name ON sequences(dataset_name)")
            
            conn.commit()
    
    def _handle_config(self, max_examples: Optional[int], storage_format: Optional[str]):
        """Handle configuration storage and validation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
                   
            # Create config table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            # Try to read existing config
            cursor.execute("SELECT key, value FROM config")
            existing_config = dict(cursor.fetchall())
            
            # If storage_format is None, read from existing config
            if storage_format is None:
                if 'storage_format' not in existing_config:
                    raise ValueError("No existing storage_format found in database and none provided")
                storage_format = existing_config['storage_format']
            
            # Validate storage_format
            if storage_format not in ['sparse', 'dense']:
                raise ValueError(f"storage_format must be 'sparse' or 'dense', got {storage_format}")
            
            # Check for conflicts with existing config
            if 'storage_format' in existing_config and existing_config['storage_format'] != storage_format:
                raise ValueError(f"Storage format conflict: database has '{existing_config['storage_format']}' but '{storage_format}' was provided")
            
            # Set instance attributes
            self._storage_format = storage_format
            self.max_examples = max_examples
            
            # Store/update config in database
            cursor.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", 
                          ('storage_format', storage_format))
            if max_examples is not None:
                cursor.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", 
                              ('max_examples', str(max_examples)))
            
            conn.commit()

    @property
    def storage_format(self):
        return self._storage_format
    
    def clear(self):
        """Clear all data from the database except config."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Enable foreign key constraints for this connection too
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("DELETE FROM activation_details")
            cursor.execute("DELETE FROM examples") 
            cursor.execute("DELETE FROM sequences")
            # Note: We keep the config table intact
            conn.commit()
    
    def __len__(self) -> int:
        """Return the number of examples in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM examples")
            return cursor.fetchone()[0]
    
    def _insert_sequence(self, sequence_idx: int, token_ids: torch.Tensor, 
                        dataset_id: Optional[int] = None, dataset_name: Optional[str] = None) -> None:
        """Insert a single sequence into the database."""
        # Convert token IDs to binary blob
        binary_data = np.array(token_ids.cpu().tolist(), dtype=np.int32).tobytes()
        
        # Get text if tokenizer is available
        text = None
        if self.tokenizer is not None:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Enable foreign key constraints for this connection too
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute(
                "INSERT OR REPLACE INTO sequences VALUES (?, ?, ?, ?, ?, ?)",
                (sequence_idx, binary_data, text, len(token_ids), dataset_id, dataset_name)
            )
            conn.commit()
    
    def _insert_sequences_bulk(self, all_sequences: List, dataset_info: Optional[List[Tuple[int, str]]] = None) -> None:
        """Bulk insert sequences into the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Enable foreign key constraints for this connection too
            cursor.execute("PRAGMA foreign_keys = ON")
            
            for seq_idx, token_ids in tqdm(enumerate(all_sequences), desc="Storing sequences"):
                # Handle both tensor and list inputs
                if isinstance(token_ids, torch.Tensor):
                    token_list = token_ids.cpu().tolist()
                else:
                    token_list = list(token_ids)
                
                # Convert token IDs to binary blob
                binary_data = np.array(token_list, dtype=np.int32).tobytes()
                
                # Get text if tokenizer is available
                text = None
                if self.tokenizer is not None:
                    text = self.tokenizer.decode(token_list, skip_special_tokens=False)
                
                # Get dataset info if provided
                dataset_id, dataset_name = (None, None)
                if dataset_info and seq_idx < len(dataset_info):
                    dataset_id, dataset_name = dataset_info[seq_idx]
                
                cursor.execute(
                    "INSERT OR REPLACE INTO sequences VALUES (?, ?, ?, ?, ?, ?)",
                    (seq_idx, binary_data, text, len(token_list), dataset_id, dataset_name)
                )
            
            conn.commit()
    
    def _insert_example(self, example_data) -> List[int]:
        """
        Insert one or more examples and return their IDs.
        
        Args:
            example_data: List of tuples where each tuple contains:
                - score (float): The activation score for this example
                - sequence_idx (int): Index of the sequence in the sequences table
                - latent_idx (int, optional): Index of the latent dimension
                - quantile_idx (int, optional): Index of the quantile bucket
                - additional_data (dict, optional): Additional metadata to store as JSON
        
        Returns:
            List[int] or int: List of example IDs if multiple examples inserted, 
                            single ID if only one example inserted
        """
        # Prepare data for bulk insert
        bulk_data = []
        for item in example_data:
            # Unpack tuple format: (score, sequence_idx, latent_idx=None, quantile_idx=None, additional_data=None)
            score = item[0]
            sequence_idx = item[1]
            latent_idx = item[2] if len(item) > 2 else None
            quantile_idx = item[3] if len(item) > 3 else None
            additional_data = item[4] if len(item) > 4 else None
    
            metadata = json.dumps(additional_data) if additional_data else None
            bulk_data.append((sequence_idx, float(score), latent_idx, quantile_idx, metadata))
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Enable foreign key constraints for this connection too
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Insert examples one by one to get proper lastrowid behavior
            example_ids = []
            for data in bulk_data:
                cursor.execute(
                    "INSERT INTO examples (sequence_idx, score, latent_idx, quantile_idx, metadata) VALUES (?, ?, ?, ?, ?)",
                    data
                )
                example_ids.append(cursor.lastrowid)
            
            conn.commit()
        
        return example_ids

    def _insert_activation_details_dense(self, example_data: List[Tuple[int, np.ndarray]]):
        """Insert dense activation details for multiple examples."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Enable foreign key constraints for this connection too
            cursor.execute("PRAGMA foreign_keys = ON")
            
            bulk_data = []
            for item in example_data:
                assert len(item) == 2, f"Invalid activation details format: expected 2 items, got {len(item)}"
                # Format: (example_id, values_array) - already dense
                example_id, values = item
                
                
                values_blob = np.array(values, dtype=np.float32).tobytes()
                bulk_data.append((example_id, values_blob))
            
            cursor.executemany("INSERT INTO activation_details VALUES (?, ?)", bulk_data)
            conn.commit()

    def _insert_activation_details_sparse(self, example_data: List[Tuple[int, np.ndarray, np.ndarray]]):
        """Bulk insert sparse activation details for multiple examples."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Enable foreign key constraints for this connection too  
            cursor.execute("PRAGMA foreign_keys = ON")
            
            bulk_data = []
            for item in example_data:
                assert len(item) == 3, f"Invalid activation details format: expected 3 items, got {len(item)}"
                # Format: (example_id, positions_array, values_array) - already sparse
                example_id, positions, values = item
                positions = np.array(positions, dtype=np.int32)
                values = np.array(values, dtype=np.float32)
                
                positions_blob = positions.tobytes()
                values_blob = values.tobytes()
                bulk_data.append((example_id, positions_blob, values_blob))
            
            cursor.executemany("INSERT INTO activation_details VALUES (?, ?, ?)", bulk_data)
            conn.commit()

            
    def _insert_activation_details(self, example_data: List):
        """Insert activation details for multiple examples, automatically selecting the right format."""
        if not example_data:
            return
            
        if self.storage_format == 'dense':
            # Dense format: (example_id, values_array)
            self._insert_activation_details_dense(example_data)
        elif self.storage_format == 'sparse':
            # Sparse format: (example_id, positions_array, values_array)
            self._insert_activation_details_sparse(example_data)
        else:
            raise ValueError(f"Unsupported activation details format. Expected 'dense' or 'sparse', got {self.storage_format}")

    def _process_batch_data(self, batch_data: BatchData) -> None:
        """
        Process batch data in the background worker process.
        
        Args:
            batch_data: BatchData instance containing the examples to add
        """
        batch_size = len(batch_data.input_ids_batch)
        
        for i in range(batch_size):
            score = float(batch_data.scores_per_example[i])
            input_ids = torch.tensor(batch_data.input_ids_batch[i])
            scores_per_token = torch.tensor(batch_data.scores_per_token_batch[i]) if batch_data.scores_per_token_batch and batch_data.scores_per_token_batch[i] is not None else None
            additional_data = batch_data.additional_data_batch[i] if batch_data.additional_data_batch else None
            
            # Get per-example latent_idx and quantile_idx
            latent_idx = batch_data.get_per_example_latent_idx(i, batch_size)
            quantile_idx = batch_data.get_per_example_quantile_idx(i, batch_size)
            
            # Generate a unique sequence index
            sequence_idx = hash(tuple(input_ids.tolist())) % (2**31)
            
            # Insert sequence with dataset info
            self._insert_sequence(sequence_idx, input_ids, 
                                dataset_name=batch_data.dataset_name,
                                dataset_id=batch_data.dataset_id)
            
            # Insert example
            example_ids = self._insert_example([(score, sequence_idx, latent_idx, quantile_idx, additional_data)])
            
            # Insert activation details if provided
            if scores_per_token is not None:
                if self.storage_format == 'sparse':
                    positions = np.arange(len(scores_per_token), dtype=np.int32)
                    values = scores_per_token.float().numpy().astype(np.float32)
                    self._insert_activation_details([(example_ids[0], positions, values)])
                elif self.storage_format == 'dense':
                    values = scores_per_token.numpy().astype(np.float32)
                    self._insert_activation_details([(example_ids[0], values)])

    def _get_grouping_key(self, latent_idx: Optional[int], quantile_idx: Optional[int], 
                         dataset_name: Optional[str]) -> Optional[tuple]:
        """
        Get the grouping key for top-k management based on active dimensions.
        
        Args:
            latent_idx: Latent feature index (optional)
            quantile_idx: Quantile index (optional) 
            dataset_name: Dataset name (optional)
            
        Returns:
            Tuple representing the grouping key, or None for overall grouping
        """
        key = []
        if latent_idx is not None:
            key.append(('latent_idx', latent_idx))
        if quantile_idx is not None:
            key.append(('quantile_idx', quantile_idx))
        if self.per_dataset:
            # Always include dataset_name in key when per_dataset=True, even if None
            key.append(('dataset_name', dataset_name))
        return tuple(key) if key else None  # None means overall grouping

    def _maintain_top_k(self):
        """Remove examples beyond max_examples limit, keeping highest scores per group."""
        if self.max_examples is None:
            return
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Get all unique grouping combinations that exist in the database
            groups = self._get_existing_groups(cursor)
            
            for group_key in groups:
                self._maintain_top_k_for_group(cursor, group_key)
            
            conn.commit()

    def _get_existing_groups(self, cursor) -> List[Optional[tuple]]:
        """Get all existing grouping combinations in the database."""
        cursor.execute("""
            SELECT DISTINCT e.latent_idx, e.quantile_idx, s.dataset_name
            FROM examples e
            JOIN sequences s ON e.sequence_idx = s.sequence_idx
        """)
        
        raw_groups = cursor.fetchall()
        groups = set()
        
        for latent_idx, quantile_idx, dataset_name in raw_groups:
            group_key = self._get_grouping_key(latent_idx, quantile_idx, dataset_name)
            groups.add(group_key)
        
        return list(groups)

    def _maintain_top_k_for_group(self, cursor, group_key: Optional[tuple]):
        """Maintain top-k for a specific grouping."""
        # Build WHERE clause based on group_key
        where_conditions = []
        params = []
        
        if group_key:
            for dim_name, dim_value in group_key:
                if dim_name == 'latent_idx':
                    if dim_value is None:
                        where_conditions.append("e.latent_idx IS NULL")
                    else:
                        where_conditions.append("e.latent_idx = ?")
                        params.append(dim_value)
                elif dim_name == 'quantile_idx':
                    if dim_value is None:
                        where_conditions.append("e.quantile_idx IS NULL")
                    else:
                        where_conditions.append("e.quantile_idx = ?")
                        params.append(dim_value)
                elif dim_name == 'dataset_name':
                    if dim_value is None:
                        where_conditions.append("s.dataset_name IS NULL")
                    else:
                        where_conditions.append("s.dataset_name = ?")
                        params.append(dim_value)
        else:
            # Handle the None group case - examples with no latent_idx, quantile_idx, and (if per_dataset=False) no dataset constraints
            if self.per_dataset:
                # When per_dataset=True, None group means latent_idx=None, quantile_idx=None, dataset_name=None
                where_conditions = ["e.latent_idx IS NULL", "e.quantile_idx IS NULL", "s.dataset_name IS NULL"]
            else:
                # When per_dataset=False, None group means latent_idx=None, quantile_idx=None (dataset_name irrelevant)
                where_conditions = ["e.latent_idx IS NULL", "e.quantile_idx IS NULL"]
        
        where_clause = " AND ".join(where_conditions)
        
        # Get count for this group
        cursor.execute(f"""
            SELECT COUNT(*) FROM examples e
            JOIN sequences s ON e.sequence_idx = s.sequence_idx
            WHERE {where_clause}
        """, params)
        
        current_count = cursor.fetchone()[0]
        
        if current_count > self.max_examples:
            # Get example IDs to delete (lowest scores in this group)
            cursor.execute(f"""
                SELECT e.example_id FROM examples e
                JOIN sequences s ON e.sequence_idx = s.sequence_idx
                WHERE {where_clause}
                ORDER BY e.score ASC 
                LIMIT ?
            """, params + [current_count - self.max_examples])
            
            ids_to_delete = [row[0] for row in cursor.fetchall()]
            self._delete_examples_by_ids(cursor, ids_to_delete)
    
    def _maintain_top_k_overall(self, cursor):
        """Remove examples beyond max_examples limit overall, keeping highest scores."""
        # Get current count
        cursor.execute("SELECT COUNT(*) FROM examples")
        current_count = cursor.fetchone()[0]
        
        if current_count > self.max_examples:
            # Get example IDs to delete (lowest scores)
            cursor.execute("""
                SELECT example_id FROM examples 
                ORDER BY score ASC 
                LIMIT ?
            """, (current_count - self.max_examples,))
            
            ids_to_delete = [row[0] for row in cursor.fetchall()]
            self._delete_examples_by_ids(cursor, ids_to_delete)
    
    def _maintain_top_k_per_dataset(self, cursor):
        """Remove examples beyond max_examples limit per dataset, keeping highest scores within each dataset."""
        # Get all datasets and their example counts
        cursor.execute("""
            SELECT s.dataset_name, COUNT(e.example_id) as count
            FROM examples e
            JOIN sequences s ON e.sequence_idx = s.sequence_idx
            GROUP BY s.dataset_name
        """)
        dataset_counts = cursor.fetchall()
        
        # Process each dataset that exceeds the limit
        for dataset_name, count in dataset_counts:
            if count > self.max_examples:
                # Get example IDs to delete for this dataset (lowest scores)
                if dataset_name is None:
                    # Handle NULL dataset_name case
                    cursor.execute("""
                        SELECT e.example_id FROM examples e
                        JOIN sequences s ON e.sequence_idx = s.sequence_idx
                        WHERE s.dataset_name IS NULL
                        ORDER BY e.score ASC 
                        LIMIT ?
                    """, (count - self.max_examples,))
                else:
                    cursor.execute("""
                        SELECT e.example_id FROM examples e
                        JOIN sequences s ON e.sequence_idx = s.sequence_idx
                        WHERE s.dataset_name = ?
                        ORDER BY e.score ASC 
                        LIMIT ?
                    """, (dataset_name, count - self.max_examples))
                
                ids_to_delete = [row[0] for row in cursor.fetchall()]
                self._delete_examples_by_ids(cursor, ids_to_delete)
    
    def _delete_examples_by_ids(self, cursor, ids_to_delete):
        """Helper method to delete examples and their activation details by IDs."""
        if not ids_to_delete:
            return
            
        # Delete activation details first (foreign key constraint)
        cursor.executemany(
            "DELETE FROM activation_details WHERE example_id = ?",
            [(id_,) for id_ in ids_to_delete]
        )
        
        # Delete examples
        cursor.executemany(
            "DELETE FROM examples WHERE example_id = ?", 
            [(id_,) for id_ in ids_to_delete]
        )
    
    def add_example(self, score: float, input_ids: torch.Tensor, 
                   scores_per_token: Optional[torch.Tensor] = None,
                   latent_idx: Optional[int] = None, quantile_idx: Optional[int] = None,
                   additional_data: Optional[dict] = None,
                   maintain_top_k: bool = True,
                   dataset_id: Optional[int] = None,
                   dataset_name: Optional[str] = None) -> None:
        """
        Add a single example with top-k management.
        
        Args:
            score: Score for this example
            input_ids: Token IDs tensor
            scores_per_token: Per-token scores (optional)
            latent_idx: Latent feature index (optional)
            quantile_idx: Quantile index (optional)
            additional_data: Additional metadata (optional)
            maintain_top_k: Whether to maintain the top-k constraint
            dataset_id: Dataset ID (optional)
            dataset_name: Dataset name (optional)
        """
        # Generate a unique sequence index
        sequence_idx = hash(tuple(input_ids.cpu().tolist())) % (2**31)
        
        # Insert sequence
        self._insert_sequence(sequence_idx, input_ids, dataset_id, dataset_name)
        
        # Insert example
        example_ids = self._insert_example([(score, sequence_idx, latent_idx, quantile_idx, additional_data)])
        
        # Insert activation details if provided
        if scores_per_token is not None:
            if self.storage_format == 'sparse':
                positions = np.arange(len(scores_per_token), dtype=np.int32)
                values = scores_per_token.float().cpu().numpy().astype(np.float32)
                self._insert_activation_details([(example_ids[0], positions, values)])
            elif self.storage_format == 'dense':
                values = scores_per_token.cpu().numpy().astype(np.float32)
                self._insert_activation_details([(example_ids[0], values)])
            else:
                raise ValueError(f"Unsupported activation details format. Expected 'dense' or 'sparse', got {self.storage_format}")
        
        # Maintain top-k constraint
        if maintain_top_k:
            self._maintain_top_k()
            
    def add_batch_examples(self, scores_per_example: torch.Tensor,
                          input_ids_batch: torch.Tensor,
                          attention_mask_batch: Optional[torch.Tensor] = None,
                          scores_per_token_batch: Optional[torch.Tensor] = None,
                          additional_data_batch: Optional[List[dict]] = None,
                          latent_idx: Optional[Union[int, torch.Tensor, np.ndarray]] = None,
                          quantile_idx: Optional[Union[int, torch.Tensor, np.ndarray]] = None,
                          dataset_name: Optional[str] = None) -> None:
        """
        Add multiple examples from a batch with top-k management (synchronous version).
        
        Args:
            scores_per_example: Scores tensor [batch_size]
            input_ids_batch: Token IDs tensor [batch_size, seq_len]
            attention_mask_batch: Attention masks [batch_size, seq_len] (optional)
            scores_per_token_batch: Per-token scores [batch_size, seq_len] (optional)
            additional_data_batch: List of additional metadata dicts (optional)
            latent_idx: Latent feature index (optional)
            quantile_idx: Quantile index (optional)
        """
        batch_size = scores_per_example.shape[0]
        assert input_ids_batch.shape[0] == batch_size, f"Batch size mismatch: scores {batch_size} vs input_ids {input_ids_batch.shape[0]}"
        
        for i in range(batch_size):
            # Extract data for this example
            score = scores_per_example[i].item()
            input_ids = input_ids_batch[i]
            
            # Apply attention mask if provided
            if attention_mask_batch is not None:
                valid_mask = attention_mask_batch[i].bool()
                input_ids = input_ids[valid_mask]
                
                scores_per_token = scores_per_token_batch[i][valid_mask] if scores_per_token_batch is not None else None
            else:
                scores_per_token = scores_per_token_batch[i] if scores_per_token_batch is not None else None
            
            additional_data = additional_data_batch[i] if additional_data_batch is not None else None
    
            
            per_example_latent_idx = get_per_example_value(latent_idx, i, batch_size)
            per_example_quantile_idx = get_per_example_value(quantile_idx, i, batch_size)
            
            # Add this example
            self.add_example(
                score=score,
                input_ids=input_ids,
                scores_per_token=scores_per_token,
                latent_idx=per_example_latent_idx,
                quantile_idx=per_example_quantile_idx,
                additional_data=additional_data,
                maintain_top_k=False
            )
        self._maintain_top_k()

    def create_async_writer(self, buffer_size: int = 1000, 
                           flush_interval: float = 30.0, 
                           auto_maintain_top_k: bool = True) -> AsyncMaxActStoreWriter:
        """
        Create an async writer for this store.
        
        Args:
            buffer_size: Number of examples to buffer before writing
            flush_interval: Time interval (seconds) to force flush
            auto_maintain_top_k: Whether to automatically maintain top-k
            
        Returns:
            AsyncMaxActStoreWriter instance
        """
        return AsyncMaxActStoreWriter(
            db_path=self.db_path,
            max_examples=self.max_examples,
            tokenizer=self.tokenizer,
            storage_format=self.storage_format,
            per_dataset=self.per_dataset,
            buffer_size=buffer_size,
            flush_interval=flush_interval,
            auto_maintain_top_k=auto_maintain_top_k
        )
    
    def fill(self, examples_data: Dict, all_sequences: List, 
             activation_details: Optional[Dict] = None,
             dataset_info: Optional[List[Tuple[int, str]]] = None) -> None:
        """
        Bulk load pre-sorted examples data into the database.
        
        Args:
            examples_data: Dict mapping quantile_idx -> latent_idx -> list of (score, sequence_idx)
            all_sequences: List of all token sequences
            activation_details: Dict mapping latent_idx -> sequence_idx -> (positions, values) for "sparse" or (values) for "dense"
            dataset_info: Optional list of (dataset_id, dataset_name) tuples for each sequence
        """
        logger.info("Bulk loading examples into database...")
        
        # Clear existing data
        self.clear()
        
        # Bulk insert all sequences first
        self._insert_sequences_bulk(all_sequences, dataset_info)
        
        # Bulk insert all examples (already sorted, no filtering needed)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enable foreign key constraints for this connection too
            cursor.execute("PRAGMA foreign_keys = ON")
            
            total_examples = sum(len(examples) for q_data in examples_data.values() for examples in q_data.values())
            example_ids = []
            activation_details_list = []
            
            with tqdm(total=total_examples, desc="Storing examples") as pbar:
                for quantile_idx, latent_data in examples_data.items():
                    for latent_idx, examples in latent_data.items():
                        for score, sequence_idx in examples:
                            cursor.execute(
                                "INSERT INTO examples (sequence_idx, score, latent_idx, quantile_idx) VALUES (?, ?, ?, ?)",
                                (int(sequence_idx), float(score), int(latent_idx), int(quantile_idx))
                            )
                            example_ids.append(cursor.lastrowid)
                            # Only add activation details if they exist
                            if activation_details is not None and latent_idx in activation_details and sequence_idx in activation_details[latent_idx]:
                                detail = activation_details[latent_idx][sequence_idx]
                                
                                if self.storage_format == 'sparse':
                                    # Sparse format expects (positions, values) tuple
                                    if isinstance(detail, tuple):
                                        positions, values = detail
                                        activation_details_list.append((cursor.lastrowid, positions, values))
                                    else:
                                        # Handle Nx2 array format from original latent_activations.py
                                        positions = detail[:, 0].astype(np.int32)
                                        values_as_int32 = detail[:, 1].astype(np.int32)
                                        values = values_as_int32.view(np.float32)
                                        activation_details_list.append((cursor.lastrowid, positions, values))
                                else:  # dense format
                                    # Dense format expects single values array
                                    activation_details_list.append((cursor.lastrowid, detail))
                            pbar.update(1)
                
                conn.commit()
        
        # Bulk insert activation details if provided
        if activation_details_list:
            self._insert_activation_details(activation_details_list)
        
        logger.info(f"Successfully loaded {total_examples} examples into database")
    
    def get_top_examples(self, limit: Optional[int] = None, 
                        latent_idx: Optional[int] = None,
                        quantile_idx: Optional[int] = None,
                        dataset_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get top examples, optionally filtered by latent_idx, quantile_idx, and/or dataset_names.
        
        Args:
            limit: Maximum number of examples to return
            latent_idx: Filter by latent index (optional)
            quantile_idx: Filter by quantile index (optional)
            dataset_names: Filter by dataset names (optional)
            
        Returns:
            List of example dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build query with optional filters
            query = """
                SELECT e.example_id, e.sequence_idx, e.score, e.latent_idx, e.quantile_idx, e.metadata,
                       s.token_ids, s.text, s.sequence_length, s.dataset_id, s.dataset_name
                FROM examples e
                JOIN sequences s ON e.sequence_idx = s.sequence_idx
            """
            
            conditions = []
            params = []
            
            if latent_idx is not None:
                conditions.append("e.latent_idx = ?")
                params.append(latent_idx)
            
            if quantile_idx is not None:
                conditions.append("e.quantile_idx = ?")
                params.append(quantile_idx)
                
            if dataset_names is not None and len(dataset_names) > 0:
                placeholders = ",".join("?" * len(dataset_names))
                conditions.append(f"s.dataset_name IN ({placeholders})")
                params.extend(dataset_names)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY e.score DESC"
            
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            examples = []
            for row in rows:
                example_id, sequence_idx, score, latent_idx, quantile_idx, metadata, token_ids_blob, text, seq_length, dataset_id, dataset_name = row
                
                # Decode token IDs
                token_ids = np.frombuffer(token_ids_blob, dtype=np.int32).tolist()
                
                # Parse metadata
                parsed_metadata = json.loads(metadata) if metadata else {}
                
                example = {
                    "example_id": example_id,
                    "sequence_idx": sequence_idx,
                    "max_score": score,
                    "input_ids": token_ids,
                    "text": text,
                    "sequence_length": seq_length,
                    "latent_idx": latent_idx,
                    "quantile_idx": quantile_idx,
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    **parsed_metadata
                }
                
                examples.append(example)
            
            return examples
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT dataset_name FROM sequences WHERE dataset_name IS NOT NULL ORDER BY dataset_name")
            rows = cursor.fetchall()
            return [row[0] for row in rows]
    
    def get_example_details(self, example_id: int, return_dense: bool = True) -> Dict[str, Any]:
        """
        Get detailed information about a specific example including activation details.
        
        Args:
            example_id: Example ID to retrieve
            return_dense: Whether to return the activation details in dense format if they are sparse
            
        Returns:
            Dictionary with example and activation details
        """
        results = self.get_batch_example_details([example_id], return_dense)
        if not results:
            raise ValueError(f"Example {example_id} not found")
        return results[0]
    
    def get_batch_example_details(self, example_ids: List[int], return_dense: bool = True) -> List[Dict[str, Any]]:
        """
        Get detailed information about multiple examples efficiently in batch.
        
        Args:
            example_ids: List of example IDs to retrieve
            return_dense: Whether to return the activation details in dense format if they are sparse
            
        Returns:
            List of dictionaries with example and activation details
        """
        if not example_ids:
            return []
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build query with IN clause for batch retrieval
            placeholders = ",".join("?" * len(example_ids))
            cursor.execute(f"""
                SELECT e.example_id, e.sequence_idx, e.score, e.latent_idx, e.quantile_idx, e.metadata,
                       s.token_ids, s.text, s.sequence_length
                FROM examples e
                JOIN sequences s ON e.sequence_idx = s.sequence_idx
                WHERE e.example_id IN ({placeholders})
                ORDER BY e.score DESC
            """, example_ids)
            
            example_rows = cursor.fetchall()
            
            # Build results dictionary for efficient lookup
            results_dict = {}
            
            for row in example_rows:
                example_id, sequence_idx, score, latent_idx, quantile_idx, metadata, token_ids_blob, text, seq_length = row
                
                # Decode token IDs
                token_ids = np.frombuffer(token_ids_blob, dtype=np.int32).tolist()
                
                # Parse metadata
                parsed_metadata = json.loads(metadata) if metadata else {}
                
                result = {
                    "example_id": example_id,
                    "sequence_idx": sequence_idx,
                    "max_score": score,
                    "input_ids": token_ids,
                    "text": text,
                    "sequence_length": seq_length,
                    "latent_idx": latent_idx,
                    "quantile_idx": quantile_idx,
                    **parsed_metadata
                }
                
                results_dict[example_id] = result
            
            # Batch retrieve activation details if they exist
            if self.storage_format == 'dense':
                cursor.execute(f"""
                    SELECT example_id, activation_values 
                    FROM activation_details 
                    WHERE example_id IN ({placeholders})
                """, example_ids)
                details_rows = cursor.fetchall()
                
                for example_id, values_blob in details_rows:
                    if example_id in results_dict:
                        values = np.frombuffer(values_blob, dtype=np.float32)
                        results_dict[example_id]["scores_per_token"] = values
                        if not return_dense:
                            # For dense format, positions are just indices where values are non-zero
                            positions = np.where(values != 0)[0]
                            results_dict[example_id]["positions"] = positions
            else:
                cursor.execute(f"""
                    SELECT example_id, positions, activation_values
                    FROM activation_details 
                    WHERE example_id IN ({placeholders})
                """, example_ids)
                details_rows = cursor.fetchall()
                
                for example_id, positions_blob, values_blob in details_rows:
                    if example_id in results_dict:
                        positions = np.frombuffer(positions_blob, dtype=np.int32)
                        values = np.frombuffer(values_blob, dtype=np.float32)
                        
                        if return_dense:
                            # Convert to dense format
                            token_ids = results_dict[example_id]["input_ids"]
                            dense_values = np.zeros(len(token_ids))
                            dense_values[positions] = values
                            results_dict[example_id]["scores_per_token"] = dense_values
                        else:
                            results_dict[example_id]["scores_per_token"] = values
                            results_dict[example_id]["positions"] = positions
            
            # Return results in the same order as requested
            return [results_dict[example_id] for example_id in example_ids if example_id in results_dict]

    def merge(self, source_store: 'MaxActStore', maintain_top_k: bool = True, sequence_idx_offset: Optional[int | str] = None) -> None:
        """
        Merge another MaxActStore into this store.
        
        Args:
            source_store: The source store to merge from
            maintain_top_k: Whether to maintain the top-k constraint after merging
            sequence_idx_offset: Optional offset to add to all source sequence indices. 
                               - If None, automatically avoids conflicts by finding next available indices.
                               - If "auto", uses (max_existing_sequence_idx + 1) as offset.
                               - If integer, uses that value as offset.
                               Useful for keeping datasets separated with non-overlapping index ranges.
        """
        # Validate storage format compatibility
        if self.storage_format != source_store.storage_format:
            raise ValueError(f"Storage format mismatch: target has '{self.storage_format}' but source has '{source_store.storage_format}'")
        
        logger.info(f"Merging store from {source_store.db_path} into {self.db_path}")
        
        # Get existing sequence indices in target to avoid conflicts
        with sqlite3.connect(self.db_path) as target_conn:
            target_cursor = target_conn.cursor()
            target_cursor.execute("PRAGMA foreign_keys = ON")
            target_cursor.execute("SELECT sequence_idx FROM sequences")
            existing_sequence_indices = set(row[0] for row in target_cursor.fetchall())
        
        # Read all data from source store
        with sqlite3.connect(source_store.db_path) as source_conn:
            source_cursor = source_conn.cursor()
            source_cursor.execute("PRAGMA foreign_keys = ON")
            
            # Get all sequences from source
            source_cursor.execute("""
                SELECT sequence_idx, token_ids, text, sequence_length, dataset_id, dataset_name 
                FROM sequences
            """)
            source_sequences = source_cursor.fetchall()
            
            # Get all examples from source  
            source_cursor.execute("""
                SELECT sequence_idx, score, latent_idx, quantile_idx, metadata
                FROM examples
                ORDER BY score DESC
            """)
            source_examples = source_cursor.fetchall()
            
            # Get all activation details from source
            if self.storage_format == 'dense':
                source_cursor.execute("""
                    SELECT e.sequence_idx, ad.activation_values
                    FROM activation_details ad
                    JOIN examples e ON ad.example_id = e.example_id
                """)
                source_activation_details = source_cursor.fetchall()
            else:  # sparse
                source_cursor.execute("""
                    SELECT e.sequence_idx, ad.positions, ad.activation_values
                    FROM activation_details ad
                    JOIN examples e ON ad.example_id = e.example_id
                """)
                source_activation_details = source_cursor.fetchall()
        
        if not source_sequences:
            logger.info("Source store is empty, nothing to merge")
            return
        
        # Create mapping for sequence indices to avoid conflicts
        sequence_idx_mapping = {}
        
        # Handle "auto" offset
        if sequence_idx_offset == "auto":
            if existing_sequence_indices:
                sequence_idx_offset = max(existing_sequence_indices) + 1
            else:
                sequence_idx_offset = 0
            logger.info(f"Auto offset calculated: {sequence_idx_offset}")
        
        if sequence_idx_offset is not None:
            # Use explicit offset for all source indices
            for source_seq_data in source_sequences:
                source_seq_idx = source_seq_data[0]
                new_seq_idx = source_seq_idx + sequence_idx_offset
                if new_seq_idx in existing_sequence_indices:
                    raise ValueError(f"Sequence index conflict: {new_seq_idx} (from source {source_seq_idx} + offset {sequence_idx_offset}) already exists in target store")
                sequence_idx_mapping[source_seq_idx] = new_seq_idx
                existing_sequence_indices.add(new_seq_idx)
        else:
            # Automatically find next available indices
            next_available_idx = max(existing_sequence_indices) + 1 if existing_sequence_indices else 0
            
            for source_seq_data in source_sequences:
                source_seq_idx = source_seq_data[0]
                if source_seq_idx in existing_sequence_indices:
                    # Use next available index
                    while next_available_idx in existing_sequence_indices:
                        next_available_idx += 1
                    sequence_idx_mapping[source_seq_idx] = next_available_idx
                    existing_sequence_indices.add(next_available_idx)
                    next_available_idx += 1
                else:
                    # Can use original index
                    sequence_idx_mapping[source_seq_idx] = source_seq_idx
                    existing_sequence_indices.add(source_seq_idx)
        
        # Insert sequences with new indices
        with sqlite3.connect(self.db_path) as target_conn:
            target_cursor = target_conn.cursor()
            target_cursor.execute("PRAGMA foreign_keys = ON")
            
            sequences_to_insert = []
            for source_seq_data in source_sequences:
                source_seq_idx, token_ids_blob, text, seq_length, dataset_id, dataset_name = source_seq_data
                new_seq_idx = sequence_idx_mapping[source_seq_idx]
                sequences_to_insert.append((new_seq_idx, token_ids_blob, text, seq_length, dataset_id, dataset_name))
            
            target_cursor.executemany(
                "INSERT OR REPLACE INTO sequences VALUES (?, ?, ?, ?, ?, ?)",
                sequences_to_insert
            )
            target_conn.commit()
        
        # Group activation details by sequence index for efficient lookup
        activation_details_by_seq = {}
        if source_activation_details:
            for detail_data in source_activation_details:
                if self.storage_format == 'dense':
                    source_seq_idx, values_blob = detail_data
                    activation_details_by_seq[source_seq_idx] = (values_blob,)
                else:  # sparse
                    source_seq_idx, positions_blob, values_blob = detail_data
                    activation_details_by_seq[source_seq_idx] = (positions_blob, values_blob)
        
        # Insert examples with new sequence indices and collect activation details to insert
        example_data_to_insert = []
        activation_details_to_insert = []
        
        for source_example_data in tqdm(source_examples, desc="Processing examples"):
            source_seq_idx, score, latent_idx, quantile_idx, metadata = source_example_data
            new_seq_idx = sequence_idx_mapping[source_seq_idx]
            example_data_to_insert.append((score, new_seq_idx, latent_idx, quantile_idx, json.loads(metadata) if metadata else None))
        
        # Insert examples in batches and collect their IDs
        with sqlite3.connect(self.db_path) as target_conn:
            target_cursor = target_conn.cursor()
            target_cursor.execute("PRAGMA foreign_keys = ON")
            
            example_ids = self._insert_example(example_data_to_insert)
            
            # Prepare activation details with new example IDs
            for i, source_example_data in enumerate(source_examples):
                source_seq_idx = source_example_data[0]
                new_seq_idx = sequence_idx_mapping[source_seq_idx]
                example_id = example_ids[i]
                
                if source_seq_idx in activation_details_by_seq:
                    if self.storage_format == 'dense':
                        values_blob = activation_details_by_seq[source_seq_idx][0]
                        values = np.frombuffer(values_blob, dtype=np.float32)
                        activation_details_to_insert.append((example_id, values))
                    else:  # sparse
                        positions_blob, values_blob = activation_details_by_seq[source_seq_idx]
                        positions = np.frombuffer(positions_blob, dtype=np.int32)
                        values = np.frombuffer(values_blob, dtype=np.float32)
                        activation_details_to_insert.append((example_id, positions, values))
        
        # Insert activation details
        if activation_details_to_insert:
            self._insert_activation_details(activation_details_to_insert)
        
        # Maintain top-k constraint if requested
        if maintain_top_k:
            self._maintain_top_k()
        
        logger.info(f"Successfully merged {len(source_examples)} examples from source store")

    def set_dataset_info(self, dataset_id: Optional[int] = None, dataset_name: Optional[str] = None, 
                        overwrite_existing: bool = True) -> int:
        """
        Set dataset_id and dataset_name for all sequences in the store.
        
        Args:
            dataset_id: Dataset ID to set (optional)
            dataset_name: Dataset name to set (optional)
            overwrite_existing: Whether to overwrite existing dataset info. If False, only updates
                              sequences where both dataset_id and dataset_name are NULL.
                              
        Returns:
            Number of sequences updated
        """
        if dataset_id is None and dataset_name is None:
            logger.warning("No dataset_id or dataset_name provided, no updates made")
            return 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Build the UPDATE query based on parameters
            set_clauses = []
            params = []
            
            if dataset_id is not None:
                set_clauses.append("dataset_id = ?")
                params.append(dataset_id)
            
            if dataset_name is not None:
                set_clauses.append("dataset_name = ?")
                params.append(dataset_name)
            
            query = f"UPDATE sequences SET {', '.join(set_clauses)}"
            
            if not overwrite_existing:
                query += " WHERE dataset_id IS NULL AND dataset_name IS NULL"
            
            cursor.execute(query, params)
            updated_count = cursor.rowcount
            conn.commit()
        
        logger.info(f"Updated dataset info for {updated_count} sequences")
        return updated_count

    def get_available_latents(self) -> List[int]:
        """Get list of available latent indices from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT latent_idx FROM examples WHERE latent_idx IS NOT NULL ORDER BY latent_idx")
            return [row[0] for row in cursor.fetchall()]

    def get_available_quantiles(self) -> List[int]:
        """Get list of available quantile indices from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT quantile_idx FROM examples WHERE quantile_idx IS NOT NULL ORDER BY quantile_idx")
            return [row[0] for row in cursor.fetchall()]

    def get_lowest_score_for_group(self, latent_idx: Optional[int] = None,
                                   quantile_idx: Optional[int] = None, 
                                   dataset_name: Optional[str] = None) -> Optional[float]:
        """Get the lowest score for a specific grouping."""
        group_key = self._get_grouping_key(latent_idx, quantile_idx, dataset_name)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            where_conditions = []
            params = []
            
            if group_key:
                for dim_name, dim_value in group_key:
                    if dim_name == 'latent_idx':
                        if dim_value is None:
                            where_conditions.append("e.latent_idx IS NULL")
                        else:
                            where_conditions.append("e.latent_idx = ?")
                            params.append(dim_value)
                    elif dim_name == 'quantile_idx':
                        if dim_value is None:
                            where_conditions.append("e.quantile_idx IS NULL")
                        else:
                            where_conditions.append("e.quantile_idx = ?") 
                            params.append(dim_value)
                    elif dim_name == 'dataset_name':
                        if dim_value is None:
                            where_conditions.append("s.dataset_name IS NULL")
                        else:
                            where_conditions.append("s.dataset_name = ?")
                            params.append(dim_value)
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            cursor.execute(f"""
                SELECT MIN(e.score) FROM examples e
                JOIN sequences s ON e.sequence_idx = s.sequence_idx  
                WHERE {where_clause}
            """, params)
            
            result = cursor.fetchone()
            return result[0] if result and result[0] is not None else None

    def get_group_capacity_info(self) -> Dict[tuple, Dict[str, Any]]:
        """Get capacity info for all existing groups."""
        groups = {}
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all existing combinations
            cursor.execute("""
                SELECT e.latent_idx, e.quantile_idx, s.dataset_name, 
                       COUNT(*) as count, MIN(e.score) as min_score
                FROM examples e
                JOIN sequences s ON e.sequence_idx = s.sequence_idx
                GROUP BY e.latent_idx, e.quantile_idx, s.dataset_name
            """)
            
            for latent_idx, quantile_idx, dataset_name, count, min_score in cursor.fetchall():
                group_key = self._get_grouping_key(latent_idx, quantile_idx, dataset_name)
                groups[group_key] = {
                    'count': count,
                    'min_score': min_score,
                    'is_full': count >= self.max_examples if self.max_examples else False
                }
        
        return groups

class ReadOnlyMaxActStore(MaxActStore):
    """
    Read-only version of MaxActStore optimized for concurrent access.
    
    This class inherits all read methods from MaxActStore but:
    1. Opens database in read-only mode to prevent locking issues
    2. Uses WAL mode for better concurrent read access
    3. Disables foreign key constraints for read operations
    4. Blocks all write operations with helpful error messages
    
    Perfect for dashboard/browser applications where multiple users
    need concurrent read access without "Database is locked" errors.
    """
    
    def __init__(self, db_path: Path, tokenizer=None):
        """
        Initialize read-only store.
        
        Args:
            db_path: Path to existing SQLite database file
            tokenizer: Optional tokenizer for text decoding
        """
        self.db_path = Path(db_path)
        self.tokenizer = tokenizer
        
        # Validate database exists
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        # Enable WAL mode for better concurrent access (if not already enabled)
        self._setup_wal_mode()
        
        # Load config from existing database
        self._load_readonly_config()
        
        logger.info(f"Initialized ReadOnlyMaxActStore for {self.db_path}")
    
    def _setup_wal_mode(self):
        """Enable WAL mode for better concurrent read access."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                mode = cursor.fetchone()[0]
                if mode.upper() == 'WAL':
                    logger.debug("WAL mode enabled for better concurrency")
                else:
                    logger.warning(f"Could not enable WAL mode, got: {mode}")
        except Exception as e:
            logger.warning(f"Failed to set WAL mode: {e}")
    
    def _load_readonly_config(self):
        """Load configuration from existing database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Load existing config
            cursor.execute("SELECT key, value FROM config")
            config = dict(cursor.fetchall())
            
            if 'storage_format' not in config:
                raise ValueError("No storage_format found in database config")
            
            self._storage_format = config['storage_format']
            self.max_examples = int(config['max_examples']) if 'max_examples' in config else None
            self.per_dataset = False  # This info isn't stored in config, assume False for read-only
    
    def _get_readonly_connection(self):
        """Get a read-only database connection."""
        # Use URI format with read-only mode for better concurrent access
        readonly_path = f"file:{self.db_path}?mode=ro"
        conn = sqlite3.connect(readonly_path, uri=True)
        
        # Don't enable foreign key constraints for read-only operations
        # This reduces lock contention
        return conn
    

    # Override all write methods to prevent accidents
    def add_example(self, *args, **kwargs):
        """Not supported in read-only mode."""
        raise NotImplementedError(
            "ReadOnlyMaxActStore doesn't support add_example(). "
            "Use the regular MaxActStore for writing operations."
        )
    
    def add_batch_examples(self, *args, **kwargs):
        """Not supported in read-only mode.""" 
        raise NotImplementedError(
            "ReadOnlyMaxActStore doesn't support add_batch_examples(). "
            "Use the regular MaxActStore for writing operations."
        )
    
    def clear(self):
        """Not supported in read-only mode."""
        raise NotImplementedError(
            "ReadOnlyMaxActStore doesn't support clear(). "
            "Use the regular MaxActStore for writing operations."
        )
    
    def fill(self, *args, **kwargs):
        """Not supported in read-only mode."""
        raise NotImplementedError(
            "ReadOnlyMaxActStore doesn't support fill(). "
            "Use the regular MaxActStore for writing operations."
        )
    
    def merge(self, *args, **kwargs):
        """Not supported in read-only mode."""
        raise NotImplementedError(
            "ReadOnlyMaxActStore doesn't support merge(). "
            "Use the regular MaxActStore for writing operations."
        )
    
    def set_dataset_info(self, *args, **kwargs):
        """Not supported in read-only mode."""
        raise NotImplementedError(
            "ReadOnlyMaxActStore doesn't support set_dataset_info(). "
            "Use the regular MaxActStore for writing operations."
        )
    
    def create_async_writer(self, *args, **kwargs):
        """Not supported in read-only mode."""
        raise NotImplementedError(
            "ReadOnlyMaxActStore doesn't support create_async_writer(). "
            "Use the regular MaxActStore for writing operations."
        )
    
    # Override private write methods
    def _insert_sequence(self, *args, **kwargs):
        """Not supported in read-only mode."""
        raise NotImplementedError("Read-only store doesn't support _insert_sequence")
    
    def _insert_example(self, *args, **kwargs):
        """Not supported in read-only mode."""
        raise NotImplementedError("Read-only store doesn't support _insert_example")
    
    def _insert_activation_details(self, *args, **kwargs):
        """Not supported in read-only mode."""
        raise NotImplementedError("Read-only store doesn't support _insert_activation_details")
    
    def _maintain_top_k(self, *args, **kwargs):
        """Not supported in read-only mode."""
        raise NotImplementedError("Read-only store doesn't support _maintain_top_k")
         