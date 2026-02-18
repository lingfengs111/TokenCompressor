#!/usr/bin/env python3
"""
Precompute semantic codes for all items using a trained RQ-VAE model.
This script extracts discrete semantic codes from item embeddings and saves them for training.

Once codes are precomputed, RQ-VAE is no longer needed during training.
"""

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.logger import setup_logger
from train_RQVAE import RQVAE, RQVAEConfig

logger = setup_logger("precompute-codes", log_to_file=False)


@dataclass
class PrecomputeConfig:
    """Configuration for precomputing semantic codes."""

    # Paths
    rqvae_checkpoint_path: str = "/home/lingfengs111/checkpoints/rqvae/best_model.pth"
    embeddings_path: str = "/home/lingfengs111/codes/soft_patch_training/data/movielens1m/emb/movielens1m_text_emb_768d.pt"
    output_dir: str = "/home/lingfengs111/codes/soft_patch_training/data/movielens1m/codes"

    # Processing parameters
    batch_size: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0

    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def log_config(self):
        """Log configuration."""
        logger.info("=" * 80)
        logger.info("PRECOMPUTE SEMANTIC CODES - Configuration")
        logger.info("=" * 80)
        logger.info(f"RQ-VAE Checkpoint: {self.rqvae_checkpoint_path}")
        logger.info(f"Embeddings Path: {self.embeddings_path}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 80)


class CodePrecomputer:
    """Precompute semantic codes for all items."""

    def __init__(self, config: PrecomputeConfig):
        """
        Args:
            config: PrecomputeConfig instance
        """
        self.config = config
        self.device = torch.device(config.device)

        # Load RQ-VAE model
        logger.info("Loading RQ-VAE model...")
        self.rqvae = self._load_rqvae()
        self.rqvae.eval()

        # Load embeddings
        logger.info("Loading item embeddings...")
        self.embeddings = self._load_embeddings()
        logger.info(f"Loaded {len(self.embeddings)} item embeddings")

    def _load_rqvae(self) -> RQVAE:
        """Load RQ-VAE model from checkpoint."""
        checkpoint_path = Path(self.config.rqvae_checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"RQ-VAE checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Reconstruct config from checkpoint
        config_dict = checkpoint.get("config", {})
        model_config = RQVAEConfig(**config_dict) if config_dict else RQVAEConfig()

        # Create model
        model = RQVAE(model_config)

        # Load state dict
        state_dict = checkpoint["model_state_dict"]
        
        # Handle torch.compile or DistributedDataParallel wrapped models
        # Remove "_orig_mod." prefix if present
        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            state_dict = {
                key.replace("_orig_mod.", ""): value
                for key, value in state_dict.items()
            }
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)

        logger.info(f"Loaded RQ-VAE model from {checkpoint_path}")
        logger.info(f"  - Item embedding dim: {model_config.item_embedding_dim}")
        logger.info(f"  - Codebook size: {model_config.codebook_size}")
        logger.info(f"  - Quantization levels: {model_config.codebook_quantization_levels}")

        return model

    def _load_embeddings(self) -> torch.Tensor:
        """Load item embeddings."""
        embeddings_path = Path(self.config.embeddings_path)

        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

        embeddings = torch.load(embeddings_path, map_location="cpu")
        logger.info(f"Loaded embeddings with shape: {embeddings.shape}")

        return embeddings

    def precompute(self) -> torch.Tensor:
        """
        Precompute semantic codes for all items.

        Returns:
            Tensor of shape [num_items, num_levels] containing semantic codes
        """
        logger.info("\nPrecomputing semantic codes...")

        # Create dataset and dataloader
        dataset = TensorDataset(self.embeddings)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        all_codes = []

        with torch.no_grad():
            for batch_idx, (batch_embeddings,) in enumerate(
                tqdm(
                    dataloader,
                    desc="Encoding items",
                    total=len(dataloader),
                )
            ):
                batch_embeddings = batch_embeddings.to(self.device)

                # Encode to semantic codes
                # Returns [batch_size, num_levels]
                batch_codes = self.rqvae.encode_to_semantic_ids(batch_embeddings)

                all_codes.append(batch_codes.cpu())

        # Concatenate all codes
        semantic_codes = torch.cat(all_codes, dim=0)
        logger.info(f"Precomputed codes shape: {semantic_codes.shape}")
        logger.info(f"  - Num items: {semantic_codes.shape[0]}")
        logger.info(f"  - Num levels: {semantic_codes.shape[1]}")
        logger.info(f"  - Code range: [{semantic_codes.min().item()}, {semantic_codes.max().item()}]")

        return semantic_codes

    def save_codes(self, semantic_codes: torch.Tensor):
        """Save precomputed codes to file."""
        output_path = Path(self.config.output_dir) / "item_semantic_codes.pt"
        torch.save(semantic_codes, output_path)
        logger.info(f"\nSaved semantic codes to: {output_path}")

        # Also save code statistics
        stats = {
            "num_items": semantic_codes.shape[0],
            "num_levels": semantic_codes.shape[1],
            "min_code": semantic_codes.min().item(),
            "max_code": semantic_codes.max().item(),
            "unique_codes_per_level": [
                torch.unique(semantic_codes[:, i]).numel()
                for i in range(semantic_codes.shape[1])
            ],
        }

        stats_path = Path(self.config.output_dir) / "code_statistics.pt"
        torch.save(stats, stats_path)
        logger.info(f"Saved code statistics to: {stats_path}")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("PRECOMPUTATION COMPLETE - Summary")
        logger.info("=" * 80)
        logger.info(f"Num Items: {stats['num_items']}")
        logger.info(f"Num Levels: {stats['num_levels']}")
        logger.info(f"Code Range: [{stats['min_code']}, {stats['max_code']}]")
        logger.info("Unique Codes per Level:")
        for level, n_unique in enumerate(stats["unique_codes_per_level"]):
            logger.info(f"  Level {level}: {n_unique} unique codes")
        logger.info("=" * 80)


def main():
    """Main execution."""
    config = PrecomputeConfig()
    config.log_config()

    # Create precomputer
    precomputer = CodePrecomputer(config)

    # Precompute codes
    semantic_codes = precomputer.precompute()

    # Save codes
    precomputer.save_codes(semantic_codes)

    logger.info("\n✓ Done! Semantic codes are ready for training.")


if __name__ == "__main__":
    main()
