"""
Utility class for tracking maximum activating examples across different diffing methods.
"""

from typing import List, Dict, Any, Optional
import torch


class MaximumTracker:
    """
    Tracks the top N examples with highest scores across different diffing methods.
    
    This class maintains a sorted list of examples and efficiently updates it
    as new examples are processed, keeping only the top N examples by score.
    """
    
    def __init__(self, num_examples: int, tokenizer=None):
        """
        Initialize the tracker.
        
        Args:
            num_examples: Maximum number of examples to track
            tokenizer: Optional tokenizer for text decoding
        """
        self.num_examples = num_examples
        self.examples: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        
    def add_example(
        self,
        score: float,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scores_per_token: Optional[torch.Tensor] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new example to the tracker if it qualifies.
        
        Args:
            score: The score for this example (higher is better)
            input_ids: Token ids for this example [seq_len]
            attention_mask: Optional attention mask [seq_len]
            scores_per_token: Optional per-token scores [seq_len]
            additional_data: Optional additional data to store with the example
        """
        # Prepare input ids (use only valid tokens if attention mask provided)
        if attention_mask is not None:
            valid_tokens = input_ids[attention_mask.bool()]
        else:
            valid_tokens = input_ids
            
        # Create example record
        example_record = {
            'max_score': score,
            'input_ids': valid_tokens.cpu().tolist(),
        }
        
        # Add per-token scores if provided
        if scores_per_token is not None:
            if attention_mask is not None:
                valid_scores = scores_per_token[attention_mask.bool()]
            else:
                valid_scores = scores_per_token
            example_record['scores_per_token'] = valid_scores.cpu().tolist()
        
        # Add text if tokenizer is available
        if self.tokenizer is not None:
            example_record['text'] = self.tokenizer.decode(
                valid_tokens, 
                skip_special_tokens=False
            )
        
        # Add any additional data
        if additional_data is not None:
            example_record.update(additional_data)
        
        # Add to examples list
        self.examples.append(example_record)
        
        # Sort by score (descending) and keep only top N
        self.examples.sort(key=lambda x: x['max_score'], reverse=True)
        self.examples = self.examples[:self.num_examples]
    
    def add_batch_examples(
        self,
        scores_per_example: torch.Tensor,
        input_ids_batch: torch.Tensor,
        attention_mask_batch: Optional[torch.Tensor] = None,
        scores_per_token_batch: Optional[torch.Tensor] = None,
        additional_data_batch: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add multiple examples from a batch.
        
        Args:
            scores_per_example: Max scores for each example [batch_size]
            input_ids_batch: Token ids for batch [batch_size, seq_len]
            attention_mask_batch: Optional attention masks [batch_size, seq_len]
            scores_per_token_batch: Optional per-token scores [batch_size, seq_len]
            additional_data_batch: Optional list of additional data dicts
        """
        batch_size = scores_per_example.shape[0]
        
        for i in range(batch_size):
            # Extract data for this example
            score = scores_per_example[i].item()
            input_ids = input_ids_batch[i]
            
            attention_mask = attention_mask_batch[i] if attention_mask_batch is not None else None
            scores_per_token = scores_per_token_batch[i] if scores_per_token_batch is not None else None
            additional_data = additional_data_batch[i] if additional_data_batch is not None else None
            
            self.add_example(
                score=score,
                input_ids=input_ids,
                attention_mask=attention_mask,
                scores_per_token=scores_per_token,
                additional_data=additional_data
            )
    
    def get_top_examples(self) -> List[Dict[str, Any]]:
        """
        Get the current top examples.
        
        Returns:
            List of example dictionaries sorted by score (descending)
        """
        return self.examples.copy()
    
    def __len__(self) -> int:
        """Return the number of examples currently tracked."""
        return len(self.examples) 