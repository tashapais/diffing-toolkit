#!/usr/bin/env python3
"""
Generate validation split for tulu-3-sft-olmo-2-mixture dataset.

This script loads the original dataset, creates a train/validation split,
and uploads the modified dataset to the Hugging Face Hub.

Assumptions:
- Dataset exists and is accessible
- Hugging Face authentication is configured
- Dataset has a 'train' split that can be split further
"""

import argparse
from datasets import load_dataset, DatasetDict
from loguru import logger


def main():
    """Load dataset, create validation split, and upload to Hub."""
    parser = argparse.ArgumentParser(description="Generate validation split for tulu dataset")
    parser.add_argument(
        "--validation-size", 
        type=float, 
        default=0.1, 
        help="Fraction of data to use for validation (default: 0.1)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for splitting (default: 42)"
    )
    parser.add_argument(
        "--max-num-samples",
        type=int,
        default=250_000,
        help="Maximum number of samples to use for validation (default: 250_000)"
    )
    args = parser.parse_args()
    
    logger.info("Loading original dataset...")
    original_dataset = load_dataset("allenai/tulu-3-sft-olmo-2-mixture")
    # Verify dataset structure
    assert "train" in original_dataset, "Dataset must have a 'train' split"
    logger.info(f"Original dataset train split size: {len(original_dataset['train'])}")
    
    # Select a random subset of the train split
    original_dataset["train"] = original_dataset["train"].select(range(args.max_num_samples))
    
    logger.info(f"Selected {args.max_num_samples} samples for train split")
    
    # Create train/validation split
    logger.info(f"Creating train/validation split with test_size={args.validation_size}")
    split_dataset = original_dataset["train"].train_test_split(
        test_size=args.validation_size, 
        seed=args.seed
    )
    
    # Create new dataset dict with proper split names
    new_dataset = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })
    
    logger.info(f"New dataset structure:")
    logger.info(f"  Train: {len(new_dataset['train'])} examples")
    logger.info(f"  Validation: {len(new_dataset['validation'])} examples")
    
    # Upload to Hub
    hub_name = "science-of-finetuning/tulu-3-sft-olmo-2-mixture"
    logger.info(f"Uploading dataset to {hub_name}...")
    
    new_dataset.push_to_hub(hub_name, private=False)
    logger.info("Dataset successfully uploaded to Hub")


if __name__ == "__main__":
    main()
