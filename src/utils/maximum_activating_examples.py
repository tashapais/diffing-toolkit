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
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from loguru import logger


class MaxActStore:
    """
    SQLite-based storage for maximum activating examples.
    
    Supports two main use cases:
    1. Bulk loading of pre-sorted examples (e.g., quantile examples)
    2. Real-time storage with top-k management (e.g., during model diffing)
    """
    
    def __init__(self, db_path: Path, max_examples: Optional[int] = None, tokenizer=None, storage_format: Optional[str] = 'sparse'):
        """
        Initialize the store.
        
        Args:
            db_path: Path to SQLite database file
            max_examples: Maximum number of examples to keep (None for unlimited)
            tokenizer: Optional tokenizer for text decoding
            storage_format: Storage format for activation details ('sparse', 'dense', or None to read from existing config)
        """
        self.db_path = Path(db_path)
        self.tokenizer = tokenizer
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database and handle config
        self._handle_config(max_examples, storage_format)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Create sequences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sequences (
                    sequence_idx INTEGER PRIMARY KEY,
                    token_ids BLOB NOT NULL,
                    text TEXT,
                    sequence_length INTEGER NOT NULL
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
            
            conn.commit()
    
    def _handle_config(self, max_examples: Optional[int], storage_format: Optional[str]):
        """Handle configuration storage and validation."""
        with sqlite3.connect(self.db_path) as conn:
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
            self.storage_format = storage_format
            self.max_examples = max_examples
            
            # Store/update config in database
            cursor.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", 
                          ('storage_format', storage_format))
            if max_examples is not None:
                cursor.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", 
                              ('max_examples', str(max_examples)))
            
            conn.commit()
    
    def clear(self):
        """Clear all data from the database except config."""
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Enable foreign key constraints for this connection too
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("SELECT COUNT(*) FROM examples")
            return cursor.fetchone()[0]
    
    def _insert_sequence(self, sequence_idx: int, token_ids: torch.Tensor) -> None:
        """Insert a single sequence into the database."""
        # Convert token IDs to binary blob
        binary_data = np.array(token_ids.cpu().tolist(), dtype=np.int32).tobytes()
        
        # Get text if tokenizer is available
        text = None
        if self.tokenizer is not None:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Enable foreign key constraints for this connection too
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute(
                "INSERT OR REPLACE INTO sequences VALUES (?, ?, ?, ?)",
                (sequence_idx, binary_data, text, len(token_ids))
            )
            conn.commit()
    
    def _insert_sequences_bulk(self, all_sequences: List) -> None:
        """Bulk insert sequences into the database."""
        with sqlite3.connect(self.db_path) as conn:
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
                
                cursor.execute(
                    "INSERT OR REPLACE INTO sequences VALUES (?, ?, ?, ?)",
                    (seq_idx, binary_data, text, len(token_list))
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
        
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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

    def _maintain_top_k(self):
        """Remove examples beyond max_examples limit, keeping highest scores."""
        if self.max_examples is None:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enable foreign key constraints for this connection too
            cursor.execute("PRAGMA foreign_keys = ON")
            
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
                
                conn.commit()
    
    def add_example(self, score: float, input_ids: torch.Tensor, 
                   scores_per_token: Optional[torch.Tensor] = None,
                   latent_idx: Optional[int] = None, quantile_idx: Optional[int] = None,
                   additional_data: Optional[dict] = None,
                   maintain_top_k: bool = True) -> None:
        """
        Add a single example with top-k management.
        
        Args:
            score: Score for this example
            input_ids: Token IDs tensor
            scores_per_token: Per-token scores (optional)
            latent_idx: Latent feature index (optional)
            quantile_idx: Quantile index (optional)
            additional_data: Additional metadata (optional)
        """
        # Generate a unique sequence index
        sequence_idx = hash(tuple(input_ids.cpu().tolist())) % (2**31)
        
        # Insert sequence
        self._insert_sequence(sequence_idx, input_ids)
        
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
                          attention_mask_batch: torch.Tensor | None = None,
                          scores_per_token_batch: torch.Tensor | None = None,
                          additional_data_batch: List[dict] | None = None,
                          latent_idx: int | None = None,
                          quantile_idx: int | None = None) -> None:
        """
        Add multiple examples from a batch with top-k management.
        
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
            
            # Add this example
            self.add_example(
                score=score,
                input_ids=input_ids,
                scores_per_token=scores_per_token,
                latent_idx=latent_idx,
                quantile_idx=quantile_idx,
                additional_data=additional_data,
                maintain_top_k=False
            )
        self._maintain_top_k()
    
    def fill(self, examples_data: Dict, all_sequences: List, 
             activation_details: Optional[Dict] = None) -> None:
        """
        Bulk load pre-sorted examples data into the database.
        
        Args:
            examples_data: Dict mapping quantile_idx -> latent_idx -> list of (score, sequence_idx)
            all_sequences: List of all token sequences
            activation_details: Dict mapping latent_idx -> sequence_idx -> (positions, values) for "sparse" or (values) for "dense"
        """
        logger.info("Bulk loading examples into database...")
        
        # Clear existing data
        self.clear()
        
        # Bulk insert all sequences first
        self._insert_sequences_bulk(all_sequences)
        
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
                        quantile_idx: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get top examples, optionally filtered by latent_idx and/or quantile_idx.
        
        Args:
            limit: Maximum number of examples to return
            latent_idx: Filter by latent index (optional)
            quantile_idx: Filter by quantile index (optional)
            
        Returns:
            List of example dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enable foreign key constraints for this connection too
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Build query with optional filters
            query = """
                SELECT e.example_id, e.sequence_idx, e.score, e.latent_idx, e.quantile_idx, e.metadata,
                       s.token_ids, s.text, s.sequence_length
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
                example_id, sequence_idx, score, latent_idx, quantile_idx, metadata, token_ids_blob, text, seq_length = row
                
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
                    **parsed_metadata
                }
                
                examples.append(example)
            
            return examples
    
    def get_example_details(self, example_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific example including activation details.
        
        Args:
            example_id: Example ID to retrieve
            
        Returns:
            Dictionary with example and activation details
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enable foreign key constraints for this connection too
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Get example info
            cursor.execute("""
                SELECT e.example_id, e.sequence_idx, e.score, e.latent_idx, e.quantile_idx, e.metadata,
                       s.token_ids, s.text, s.sequence_length
                FROM examples e
                JOIN sequences s ON e.sequence_idx = s.sequence_idx
                WHERE e.example_id = ?
            """, (example_id,))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Example {example_id} not found")
            
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
            
            # Get activation details if they exist
            if self.storage_format == 'dense':
                cursor.execute("SELECT activation_values FROM activation_details WHERE example_id = ?", (example_id,))
                details_row = cursor.fetchone()
                
                if details_row:
                    values_blob, = details_row
                    values = np.frombuffer(values_blob, dtype=np.float32)
                    result["scores_per_token"] = values.tolist()
                    # For dense format, positions are just indices where values are non-zero
                    positions = np.where(values != 0)[0]
                    result["positions"] = positions.tolist()
            else:
                cursor.execute("SELECT positions, activation_values FROM activation_details WHERE example_id = ?", (example_id,))
                details_row = cursor.fetchone()
                
                if details_row:
                    positions_blob, values_blob = details_row
                    positions = np.frombuffer(positions_blob, dtype=np.int32)
                    values = np.frombuffer(values_blob, dtype=np.float32)
                    result["scores_per_token"] = values.tolist()
                    result["positions"] = positions.tolist()
            
            return result 