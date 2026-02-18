"""
Semantic Code Decoder - Complete downstream model for RQ-VAE semantic codes.

This package provides:
- SemanticCodeDecoder: Transformer-based decoder for predicting next semantic codes
- Data loading utilities: Convert item embeddings to semantic codes via refer.py's RQ-VAE
- Training script: Complete training pipeline with validation and checkpointing
"""

from .model import (
    SemanticCodeDecoder,
    TransformerBlock,
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
)

from .data import (
    SemanticCodeSequenceDataset,
    SemanticCodeCollator,
    create_semantic_dataloader,
    create_dummy_semantic_dataloader,
)

from .train import Trainer, create_model

__version__ = "0.1.0"

__all__ = [
    "SemanticCodeDecoder",
    "TransformerBlock",
    "MultiHeadAttention",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SemanticCodeSequenceDataset",
    "SemanticCodeCollator",
    "create_semantic_dataloader",
    "create_dummy_semantic_dataloader",
    "Trainer",
    "create_model",
]
