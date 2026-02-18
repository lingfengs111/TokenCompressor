#!/usr/bin/env python3
"""
Tokenize product items for use with embedding models.
Pre-tokenizes text data to separate CPU-bound tokenization from GPU inference.
"""

import os
import sys
from pathlib import Path
from typing import List

# Disable tokenizers parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.logger import setup_logger

logger = setup_logger("tokenize-items", log_to_file=True)

# Data settings
CATEGORY = "Video_Games"  # Product category to process
NUM_ROWS = None  # Number of rows to process (None = all)
DATA_DIR = Path("data")  # Data directory path

# Model settings
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # HuggingFace model name
BATCH_SIZE = 32  # Batch size for processing
MAX_LENGTH = 2048  # Maximum sequence length


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Add instruction prefix to improve embedding quality."""
    return f"Instruct: {task_description}\nQuery: {query}"


def tokenize_and_save(
    item_contexts: List[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    output_path: Path,
) -> tuple:
    """
    Tokenize all texts and save to disk for later use.

    Args:
        item_contexts: List of item context strings to tokenize
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        output_path: Path to save tokenized data

    Returns:
        Tuple of (input_ids shape, attention_mask shape)
    """
    logger.info("Starting tokenization...")

    all_input_ids = []
    all_attention_masks = []

    task = (
        "Given a product description, generate a semantic embedding that captures its key features and characteristics"
    )

    # Process in batches
    for i in tqdm(range(0, len(item_contexts), batch_size), desc="Tokenizing"):
        batch_texts = item_contexts[i : i + batch_size]
        instructed_texts = [get_detailed_instruct(task, text) for text in batch_texts]

        # Tokenize
        encoded = tokenizer(
            instructed_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Store as numpy arrays
        all_input_ids.append(encoded["input_ids"].numpy())
        all_attention_masks.append(encoded["attention_mask"].numpy())

    # Concatenate all batches
    input_ids = np.vstack(all_input_ids)
    attention_mask = np.vstack(all_attention_masks)

    # Save to disk
    logger.info(f"Saving tokenized data to {output_path}")
    np.savez_compressed(
        output_path,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        n_items=len(item_contexts),
    )

    logger.info(f"Tokenization complete! Shape: {input_ids.shape}")
    return input_ids.shape


def tokenize_items():
    logger.info(f"Model: {MODEL_NAME}, Batch: {BATCH_SIZE}, Max length: {MAX_LENGTH}")

    # Setup paths
    input_path = DATA_DIR / "output" / f"{CATEGORY}_items_updated.parquet"

    # Determine tokenized data path
    suffix = f"_{NUM_ROWS}" if NUM_ROWS else ""
    output_path = DATA_DIR / "output" / f"{CATEGORY}_tokenized{suffix}.npz"

    # Load data
    logger.info(f"Loading data from {input_path}")
    item_df = pl.read_parquet(input_path)
    logger.info(f"Loaded {len(item_df):,} items from {input_path}")
    logger.info("Sample of item_contexts:")
    pl.Config.set_fmt_str_lengths(2000)
    logger.info(item_df["item_context"])

    # Apply row limit if specified
    if NUM_ROWS is not None:
        logger.info(f"Limiting to {NUM_ROWS} rows for testing")
        item_df = item_df.head(NUM_ROWS)

    total_items = len(item_df)
    logger.info(f"Processing {total_items:,} items")

    # Extract item contexts
    item_contexts = item_df["item_context"].to_list()

    # Load tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize and save
    tokenize_and_save(item_contexts, tokenizer, MAX_LENGTH, BATCH_SIZE, output_path)

    logger.info(f"Tokenization saved to {output_path}")


if __name__ == "__main__":
    tokenize_items()
