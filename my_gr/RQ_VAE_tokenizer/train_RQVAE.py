#!/usr/bin/env python3
"""
Train RQ-VAE (Residual Quantized VAE) for hierarchical semantic ID generation.
This model learns to encode item embeddings into discrete hierarchical codes.
repo_URL: https://github.com/eugeneyan/semantic-ids-llm/blob/main/src/train_rqvae.py
"""

import inspect
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import wandb

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.device_manager import DeviceManager
from core.logger import setup_logger

logger = setup_logger("train-rqvae", log_to_file=True)


@dataclass
class RQVAEConfig:
    """Configuration for RQ-VAE training."""

    # Data settings
    category: str = "movielens1m"  # Dataset name / Product category to process
    # data_dir: Path = field(default_factory=lambda: Path("data"))  # Data directory path
    data_dir: str = "/home/lingfengs111/codes/soft_patch_training/data/movielens1m"  # Data directory path
    embeddings_path: Optional[Path] = None  # Path to embeddings file (auto-generated if None)
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints") / "rqvae")

    # Model parameters
    item_embedding_dim: int = 768  # Input embedding dimension (e.g., Qwen3-0.6B)
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])  # Encoder layers
    codebook_embedding_dim: int = 32  # Dimension of codebook vectors
    codebook_quantization_levels: int = 3  # Number of hierarchical levels
    codebook_size: int = 256  # Number of codes per codebook
    commitment_weight: float = 0.25  # Commitment loss weight (beta)
    use_rotation_trick: bool = True  # Use rotation trick for better gradient flow

    # If we want to use EMA (but doesn't work very well)
    use_ema_vq: bool = False  # Use EMA-based vector quantization
    ema_decay: float = 0.99  # Decay factor for exponential moving average
    ema_epsilon: float = 1e-5  # Epsilon for numerical stability in EMA

    # Training parameters
    batch_size: int = 32768  # Batch size for training
    gradient_accumulation_steps: int = 1  # Number of gradient accumulation steps
    num_epochs: int = 20000  # Number of training epochs
    scheduler_type: str = "cosine_with_warmup"  # Learning rate scheduler type ("cosine", "cosine_with_warmup")
    warmup_start_lr: float = 1e-8  # Starting learning rate for warmup (only for cosine_with_warmup)
    warmup_steps: int = 200  # Number of warmup steps (only for cosine_with_warmup)
    max_lr: float = 3e-4  # Maximum learning rate (start of cosine)
    min_lr: float = 1e-6  # Minimum learning rate (end of cosine)
    use_gradient_clipping: bool = True  # Enable gradient clipping
    gradient_clip_norm: float = 1.0  # Maximum gradient norm for clipping
    use_kmeans_init: bool = True  # Use k-means initialization for codebooks
    reset_unused_codes: bool = True  # Reset unused codebook codes during training
    steps_per_codebook_reset: int = 2  # Reset unused codebook codes every N steps (breaks if set to 1)
    codebook_usage_threshold: float = 1.0  # Only reset if usage falls below this proportion (0-1)
    val_split: float = 0.05  # Validation set split ratio

    # Logging and checkpointing
    steps_per_train_log: int = 10  # Log training progress every N steps
    steps_per_val_log: int = 200  # Validate and checkpoint every N steps

    def __post_init__(self):
        """Validate configuration and set computed fields."""
        # Auto-generate embeddings path if not provided
        if self.embeddings_path is None:
            self.embeddings_path = self.data_dir + "/emb/" + f"{self.category}_text_emb_{self.item_embedding_dim}d.pt"

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=== RQ-VAE Configuration ===")

        # Data settings
        logger.info("Data Settings:")
        logger.info(f"  category: {self.category}")
        logger.info(f"  data_dir: {self.data_dir}")
        logger.info(f"  embeddings_path: {self.embeddings_path}")
        logger.info(f"  checkpoint_dir: {self.checkpoint_dir}")

        # Model parameters
        logger.info("Model Parameters:")
        logger.info(f"  item_embedding_dim: {self.item_embedding_dim}")
        logger.info(f"  encoder_hidden_dims: {self.encoder_hidden_dims}")
        logger.info(f"  codebook_embedding_dim: {self.codebook_embedding_dim}")
        logger.info(f"  codebook_quantization_levels: {self.codebook_quantization_levels}")
        logger.info(f"  codebook_size: {self.codebook_size}")
        logger.info(f"  commitment_weight: {self.commitment_weight}")
        logger.info(f"  use_rotation_trick: {self.use_rotation_trick}")

        # EMA settings
        logger.info("EMA Settings:")
        logger.info(f"  use_ema_vq: {self.use_ema_vq}")
        logger.info(f"  ema_decay: {self.ema_decay}")
        logger.info(f"  ema_epsilon: {self.ema_epsilon}")

        # Training parameters
        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        logger.info(f"  effective_batch_size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"  num_epochs: {self.num_epochs}")
        logger.info(f"  scheduler_type: {self.scheduler_type}")
        logger.info(f"  warmup_start_lr: {self.warmup_start_lr}")
        logger.info(f"  warmup_steps: {self.warmup_steps}")
        logger.info(f"  max_lr: {self.max_lr}")
        logger.info(f"  min_lr: {self.min_lr}")
        logger.info(f"  use_gradient_clipping: {self.use_gradient_clipping}")
        logger.info(f"  gradient_clip_norm: {self.gradient_clip_norm}")
        logger.info(f"  use_kmeans_init: {self.use_kmeans_init}")
        logger.info(f"  reset_unused_codes: {self.reset_unused_codes}")
        logger.info(f"  steps_per_codebook_reset: {self.steps_per_codebook_reset}")
        logger.info(f"  codebook_usage_threshold: {self.codebook_usage_threshold}")
        logger.info(f"  val_split: {self.val_split}")

        # Logging and checkpointing
        logger.info("Logging and Checkpointing:")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  steps_per_val_log: {self.steps_per_val_log}")
        logger.info("===========================")

import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    """Dataset adpted for .pt embedding files `

    Expected format:
        torch.save(E, path)
    where E is a FloatTensor of shape [num_items+1, d_model],
    and row 0 is PAD (ignored for training).
    """
    def __init__(self, embeddings_path, limit=None, drop_pad=True):
        E = torch.load(embeddings_path, map_location="cpu")
        if not isinstance(E, torch.Tensor):
            raise ValueError(f"Expected tensor in {embeddings_path}, got {type(E)}")
        
        if drop_pad:
            E = E[1:]
        # limit
        if limit is not None:
            E = E[:limit]

        self.embeddings = E.float()  # [N, d_model]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


class QuantizationOutput(NamedTuple):
    quantized_st: Tensor
    quantized: Tensor
    indices: Tensor
    loss: Tensor
    codebook_loss: Optional[Tensor]  # None for EMA
    commitment_loss: Tensor


class BaseVectorQuantizer(nn.Module):
    """Base class for vector quantization with shared functionality."""

    def __init__(self, config: RQVAEConfig):
        super().__init__()
        self.codebook_embedding_dim = config.codebook_embedding_dim
        self.codebook_size = config.codebook_size
        self.commitment_weight = config.commitment_weight
        self.use_rotation_trick = config.use_rotation_trick

        # Learnable codebook
        self.embedding = nn.Embedding(self.codebook_size, self.codebook_embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.codebook_size, 1 / self.codebook_size)

        # Track codebook usage
        self.register_buffer("usage_count", torch.zeros(self.codebook_size))
        self.register_buffer("update_count", torch.tensor(0))

    @staticmethod
    def l2norm(t: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
        """L2 normalize tensor along specified dimension."""
        return F.normalize(t, p=2, dim=dim, eps=eps)

    @staticmethod
    def safe_div(num: Tensor, den: Tensor, eps: float = 1e-6) -> Tensor:
        """Safe division to avoid numerical issues."""
        return num / den.clamp(min=eps)

    @staticmethod
    def rotation_trick(u: Tensor, q: Tensor, e: Tensor) -> Tensor:
        """
        Efficient rotation trick transform from Eq 4.2 in https://arxiv.org/abs/2410.06424

        Args:
            u: Unit vector from encoder output (normalized x)
            q: Unit vector from quantized output (normalized quantized)
            e: Original encoder output (x)

        Returns:
            Rotated encoder output
        """
        w = BaseVectorQuantizer.l2norm(u + q, dim=-1).detach()

        # Reshape for batch matrix multiplication
        w_col = w.unsqueeze(-1)
        w_row = w.unsqueeze(-2)
        u_col = u.unsqueeze(-1).detach()
        q_row = q.unsqueeze(-2).detach()

        # For 2D input, add temporary batch dimension
        if e.ndim == 2:
            e_expanded = e.unsqueeze(1)  # [B, D] -> [B, 1, D]
            result = e_expanded - 2 * (e_expanded @ w_col @ w_row) + 2 * (e_expanded @ u_col @ q_row)
            return result.squeeze(1)  # [B, 1, D] -> [B, D]
        else:
            return e - 2 * (e @ w_col @ w_row).squeeze(-1) + 2 * (e @ u_col @ q_row).squeeze(-1)

    @staticmethod
    def rotate_to(src: Tensor, tgt: Tensor) -> Tensor:
        """
        Apply rotation trick STE from https://arxiv.org/abs/2410.06424
        to get gradients through VQ layer.

        Args:
            src: Source tensor (encoder output)
            tgt: Target tensor (quantized output)

        Returns:
            Rotated tensor that equals tgt in forward pass but has gradients
        """
        # Flatten to 2D for processing
        orig_shape = src.shape
        src_flat = src.reshape(-1, src.shape[-1])
        tgt_flat = tgt.reshape(-1, tgt.shape[-1])

        # Get norms
        norm_src = src_flat.norm(dim=-1, keepdim=True)
        norm_tgt = tgt_flat.norm(dim=-1, keepdim=True)

        # Apply rotation in normalized space
        rotated_tgt = BaseVectorQuantizer.rotation_trick(
            BaseVectorQuantizer.safe_div(src_flat, norm_src), BaseVectorQuantizer.safe_div(tgt_flat, norm_tgt), src_flat
        )

        # Scale to match target norm
        rotated = rotated_tgt * BaseVectorQuantizer.safe_div(norm_tgt, norm_src).detach()

        # Reshape back
        return rotated.reshape(orig_shape)

    def find_nearest_codes(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Find nearest codebook entries for input tensor."""
        input_shape = x.shape
        flat_x = x.reshape(-1, self.codebook_embedding_dim)

        # Calculate distances to all codebook vectors
        distances = torch.cdist(flat_x, self.embedding.weight)
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(input_shape)

        return indices.view(input_shape[:-1]), quantized

    def apply_gradient_estimator(self, x: Tensor, quantized: Tensor) -> Tensor:
        """Apply rotation trick or straight-through estimator."""
        if self.training and x.requires_grad:
            if self.use_rotation_trick:
                return self.rotate_to(x, quantized)
            else:
                return x + (quantized - x).detach()
        return quantized

    def update_usage(self, indices: Tensor):
        """Update codebook usage statistics."""
        indices_flat = indices.flatten()
        self.usage_count.scatter_add_(0, indices_flat, torch.ones_like(indices_flat, dtype=torch.float))
        self.update_count += 1

    def get_usage_rate(self) -> float:
        """Get proportion of codebook vectors that have been used."""
        if self.update_count == 0:
            return 0.0
        return (self.usage_count > 0).float().mean().item()

    def reset_usage_count(self):
        """Reset usage count (useful for periodic resets)."""
        self.usage_count.zero_()


class VectorQuantizer(BaseVectorQuantizer):
    """Simple vector quantization layer with learnable codebook."""

    def forward(self, x: Tensor) -> QuantizationOutput:
        indices, quantized = self.find_nearest_codes(x)
        quantized_st = self.apply_gradient_estimator(x, quantized)

        # Compute VQ losses
        codebook_loss = F.mse_loss(x.detach(), quantized)
        commitment_loss = F.mse_loss(x, quantized.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        if self.training:
            self.update_usage(indices)

        return QuantizationOutput(quantized_st, quantized, indices, loss, codebook_loss, commitment_loss)

    def reset_unused_codes(self, batch_data: Tensor):
        """Reset unused codebook vectors to random samples from batch."""
        if self.update_count == 0:
            return

        # Find codes with zero usage
        unused_indices = (self.usage_count == 0).nonzero().squeeze(-1)

        if len(unused_indices) > 0 and batch_data.shape[0] >= len(unused_indices):
            # Sample random vectors from batch
            batch_flat = batch_data.reshape(-1, self.codebook_embedding_dim)
            random_indices = torch.randperm(batch_flat.shape[0], device=batch_flat.device)[: len(unused_indices)]
            self.embedding.weight.data[unused_indices] = batch_flat[random_indices].detach()

        # Reset usage count after replacement
        self.reset_usage_count()


class EMAVectorQuantizer(BaseVectorQuantizer):
    """Vector quantization layer with EMA-based codebook updates."""

    def __init__(self, config: RQVAEConfig):
        super().__init__(config)
        self.decay = config.ema_decay
        self.epsilon = config.ema_epsilon

        # EMA parameters (not trainable)
        self.register_buffer("ema_cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    @staticmethod
    def ema_inplace(moving_avg: Tensor, new: Tensor, decay: float) -> None:
        """In-place exponential moving average update."""
        moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)

    @staticmethod
    def laplace_smoothing(x: Tensor, n_categories: int, epsilon: float = 1e-5) -> Tensor:
        """Apply Laplace smoothing for numerical stability."""
        return (x + epsilon) / (x.sum() + n_categories * epsilon) * x.sum()

    def forward(self, x: Tensor) -> QuantizationOutput:
        indices, quantized = self.find_nearest_codes(x)

        # Use EMA to update codebooks instead of gradients
        if self.training:
            # Flatten for EMA update
            flat_x = x.reshape(-1, self.codebook_embedding_dim)
            flat_indices = indices.flatten()

            # Update EMA
            encodings = F.one_hot(flat_indices, self.codebook_size).float()
            self.ema_inplace(self.ema_cluster_size, encodings.sum(0), self.decay)
            dw = encodings.T @ flat_x
            self.ema_inplace(self.ema_w, dw, self.decay)

            # Update embeddings
            cluster_size = self.laplace_smoothing(self.ema_cluster_size, self.codebook_size, self.epsilon)
            self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)

            # Update usage statistics
            self.update_usage(indices)

        # Apply gradient estimator
        quantized_st = self.apply_gradient_estimator(x, quantized)

        # Only commitment loss (no codebook loss with EMA)
        commitment_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_weight * commitment_loss

        return QuantizationOutput(quantized_st, quantized, indices, loss, None, commitment_loss)


class RQVAE(nn.Module):
    """Residual Quantized VAE for generating semantic IDs."""

    def __init__(self, config: RQVAEConfig):
        super().__init__()

        # Store the config
        self.config = config

        # Extract model parameters from config
        self.item_embedding_dim = config.item_embedding_dim
        self.encoder_hidden_dims = config.encoder_hidden_dims
        self.codebook_embedding_dim = config.codebook_embedding_dim
        self.codebook_quantization_levels = config.codebook_quantization_levels
        self.codebook_size = config.codebook_size
        self.use_rotation_trick = config.use_rotation_trick

        # Build encoder: item_embedding_dim -> encoder_hidden_dims -> codebook_embedding_dim
        encoder_layers = []
        dims = [config.item_embedding_dim] + config.encoder_hidden_dims + [config.codebook_embedding_dim]

        for i in range(len(dims) - 2):
            encoder_layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.SiLU()])
        encoder_layers.append(nn.Linear(dims[-2], dims[-1]))

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder: codebook_embedding_dim -> encoder_hidden_dims (reversed) -> item_embedding_dim
        decoder_layers = []
        dims_reversed = [config.codebook_embedding_dim] + config.encoder_hidden_dims[::-1] + [config.item_embedding_dim]

        for i in range(len(dims_reversed) - 2):
            decoder_layers.extend([nn.Linear(dims_reversed[i], dims_reversed[i + 1]), nn.SiLU()])
        decoder_layers.append(nn.Linear(dims_reversed[-2], dims_reversed[-1]))

        self.decoder = nn.Sequential(*decoder_layers)

        # Create quantization layers
        quantizer_class = EMAVectorQuantizer if config.use_ema_vq else VectorQuantizer
        self.vq_layers = nn.ModuleList([quantizer_class(config) for _ in range(config.codebook_quantization_levels)])

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation to output."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor], dict]:
        """Full forward pass through encoder, quantization, and decoder."""
        z = self.encode(x)

        # Residual quantization
        quantized_out = torch.zeros_like(z)
        residual = z

        all_indices = []
        vq_loss = 0
        codebook_losses = []
        commitment_losses = []

        for vq_layer in self.vq_layers:
            vq_output = vq_layer(residual)  # Quantize current residual
            residual = residual - vq_output.quantized.detach()  # Update residual for next level
            quantized_out = quantized_out + vq_output.quantized_st  # Accumulate quantized vectors
            all_indices.append(vq_output.indices)

            vq_loss = vq_loss + vq_output.loss  # Store indices and accumulate loss
            if vq_output.codebook_loss is not None:  # Track individual loss components
                codebook_losses.append(vq_output.codebook_loss)
            commitment_losses.append(vq_output.commitment_loss)

        x_recon = self.decode(quantized_out)  # Decode
        recon_loss = F.mse_loss(x_recon, x)  # Reconstruction loss
        loss = recon_loss + vq_loss  # Total loss

        loss_dict = {
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "codebook_losses": codebook_losses,  # List of losses per level (empty for EMA)
            "commitment_losses": commitment_losses,  # List of losses per level
            "indices": all_indices,  # Store for metric computation
            "residual": residual,  # Store for residual norm calculation
        }

        return x_recon, all_indices, loss_dict

    def encode_to_semantic_ids(self, x: Tensor) -> Tensor:
        """Extract semantic IDs for input batch."""
        with torch.no_grad():
            z = self.encode(x)
            residual = z
            indices_list = []

            for vq_layer in self.vq_layers:
                vq_output = vq_layer(residual)
                indices_list.append(vq_output.indices)
                residual = residual - vq_output.quantized

            # Stack indices from all levels
            semantic_ids = torch.stack(indices_list, dim=-1)
        return semantic_ids

    def decode_from_semantic_ids(self, semantic_ids: Tensor) -> Tensor:
        """Decode from semantic IDs."""
        with torch.no_grad():
            # semantic_ids shape: [batch, codebook_quantization_levels]
            quantized_sum = torch.zeros(semantic_ids.shape[0], self.codebook_embedding_dim, device=semantic_ids.device)

            for level, indices in enumerate(semantic_ids.unbind(dim=-1)):
                codes = self.vq_layers[level].embedding(indices)
                quantized_sum += codes

            return self.decode(quantized_sum)

    def calculate_unique_ids_proportion(self, semantic_ids: Tensor) -> float:
        """Calculate proportion of unique semantic IDs in a batch.

        Args:
            semantic_ids: Tensor of shape [batch_size, codebook_quantization_levels]

        Returns:
            Proportion of items with unique semantic IDs (0 to 1)
        """
        batch_size = semantic_ids.shape[0]
        if batch_size <= 1:
            return 1.0

        # Compare all pairs of semantic IDs
        # Shape: [batch_size, 1, codebook_quantization_levels] == [1, batch_size, codebook_quantization_levels]
        ids_expanded_1 = semantic_ids.unsqueeze(1)  # [B, 1, L]
        ids_expanded_2 = semantic_ids.unsqueeze(0)  # [1, B, L]

        # Check which pairs are identical (all levels match)
        matches = (ids_expanded_1 == ids_expanded_2).all(dim=-1)  # [B, B]

        # Mask upper triangular to avoid self-comparison and double counting
        upper_tri_matches = torch.triu(matches, diagonal=1)  # [B, B]

        # A row has any True if that ID matches any other ID after it
        has_duplicate = upper_tri_matches.any(dim=1)  # [B]

        # Count unique IDs (those that don't have duplicates)
        n_unique = (~has_duplicate).sum().item()

        return n_unique / batch_size

    def calculate_codebook_usage(self) -> List[float]:
        """Get codebook usage rate for each level.

        Returns:
            List of usage percentages for each quantization level
        """
        return [vq_layer.get_usage_rate() for vq_layer in self.vq_layers]

    def calculate_avg_residual_norm(self, residual: Tensor) -> float:
        """Calculate average residual norm after quantization.

        Args:
            residual: Final residual tensor after all quantization levels

        Returns:
            Average L2 norm of the residual
        """
        return residual.norm(dim=-1).mean().item()

    def kmeans_init(self, data_loader, device):
        """Initialize codebooks using k-means on first batch."""
        logger.info("Initializing codebooks with k-means clustering...")
        # Get first batch
        first_batch = next(iter(data_loader))
        if isinstance(first_batch, (list, tuple)):
            first_batch = first_batch[0]
        first_batch = first_batch.to(device)

        # Encode to latent space
        with torch.no_grad():
            z = self.encode(first_batch)

            # Initialize each level's codebook
            residual = z
            for level, vq_layer in enumerate(self.vq_layers):
                residual_np = residual.cpu().numpy().reshape(-1, self.codebook_embedding_dim)  # Flatten for k-means

                kmeans = KMeans(n_clusters=self.codebook_size, n_init=10, random_state=0)
                kmeans.fit(residual_np)  # Run k-means

                vq_layer.embedding.weight.data = torch.from_numpy(kmeans.cluster_centers_).to(device)  # Update codebook
                logger.info(f"  Level {level}: initialized {self.codebook_size} codes")

                if level < self.codebook_quantization_levels - 1:  # Compute next residual
                    vq_output = vq_layer(residual)
                    residual = residual - vq_output.quantized


def train_rqvae(
    model: RQVAE,
    data_loader: torch.utils.data.DataLoader,
    config: RQVAEConfig,
    device: str = "cpu",
    val_loader: Optional[torch.utils.data.DataLoader] = None,
):
    """Train RQVAE model with simplified logging.

    Args:
        model: RQVAE model to train
        data_loader: Training data loader
        config: RQVAEConfig object containing all training parameters
        device: Device to train on
        val_loader: Optional validation data loader
    """
    model = model.to(device)
    if config.use_kmeans_init:
        model.kmeans_init(data_loader, device)

    # Apply torch.compile for faster training (CUDA only, not MPS)
    if device == "cuda":
        logger.info("Compiling model with torch.compile for faster training...")
        model = torch.compile(model)

    # Better optimizer choice for RQ-VAE with fused support
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=0.01, fused=use_fused)

    # Calculate total training steps
    steps_per_epoch = len(data_loader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.num_epochs
    logger.info(f"Total training steps: {total_steps:,} ({steps_per_epoch} steps/epoch x {config.num_epochs} epochs)")

    # Learning rate scheduler
    if config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.min_lr)
        logger.info(f"Cosine annealing: {config.max_lr:.1e} -> {config.min_lr:.1e} for {total_steps:,} steps")
    elif config.scheduler_type == "cosine_with_warmup":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.warmup_start_lr / config.max_lr,
            total_iters=config.warmup_steps,
        )
        logger.info(f"Warmup: {config.warmup_start_lr:.1e} -> {config.max_lr:.1e} for {config.warmup_steps:,} steps")

        cosine_steps = total_steps - config.warmup_steps
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=config.min_lr)
        logger.info(f"Cosine annealing: {config.max_lr:.1e} -> {config.min_lr:.1e} for {cosine_steps:,} steps")

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps]
        )
    else:
        scheduler = None

    # Track best model
    best_loss = float("inf")
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()

        for batch_idx, data in enumerate(data_loader):
            # Only perform optimizer step every gradient_accumulation_steps
            if batch_idx % config.gradient_accumulation_steps == 0:
                t0 = time.time()
                optimizer.zero_grad()
                loss_accum = 0.0

            x_recon, indices, loss_dict = model(data.to(device))

            loss = loss_dict["loss"] / config.gradient_accumulation_steps
            loss_accum += loss_dict["loss"].detach()  # Accumulate unscaled loss for logging

            loss.backward()

            # Only step optimizer after accumulating gradients
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Get gradient norm before clipping
                grad_norm_before = get_gradient_norm(model)

                # Clip gradients to prevent explosion
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
                    grad_norm_after = get_gradient_norm(model)
                else:
                    grad_norm_after = grad_norm_before

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                t1 = time.time()
                batch_time_ms = (t1 - t0) * 1000
                samples_per_second = (data.shape[0] * config.gradient_accumulation_steps) / (t1 - t0)
                global_step += 1

                avg_loss = loss_accum / config.gradient_accumulation_steps

                # Log progress every N steps
                if global_step == 1 or global_step % config.steps_per_train_log == 0:
                    current_lr = optimizer.param_groups[0]["lr"]

                    # Compute codebook usage and unique IDs for current batch
                    codebook_usage = model.calculate_codebook_usage()
                    semantic_ids = torch.stack(loss_dict["indices"], dim=-1)
                    unique_ids_proportion = model.calculate_unique_ids_proportion(semantic_ids)
                    usage_str = "/".join([f"{u:.2f}" for u in codebook_usage])

                    log_str = (
                        f"Step {global_step:05d} | Epoch {epoch + 1:05d} | lr: {current_lr:.2e} | "
                        f"loss: {avg_loss:.2e} | recon: {loss_dict['recon_loss'].item():.2e} | "
                        f"vq: {loss_dict['vq_loss'].item():.2e} | codebook usage: {usage_str} | "
                        f"unique ids: {unique_ids_proportion:.1%} | time: {batch_time_ms:.0f}ms | "
                        f"samples/s: {samples_per_second:,.0f}"
                    )
                    logger.info(log_str)

                    wandb_log_dict = {
                        "loss/total": avg_loss,
                        "loss/reconstruction": loss_dict["recon_loss"].item(),
                        "loss/vq": loss_dict["vq_loss"].item(),
                        "metrics/learning_rate": current_lr,
                        "metrics/gradient_norm": grad_norm_before,
                        "metrics/gradient_norm_clipped": grad_norm_after,
                        "metrics/batch_time_ms": batch_time_ms,
                        "metrics/samples_per_second": samples_per_second,
                        "epoch": epoch + 1,
                        "step": global_step,
                    }

                    # Add per-level losses
                    for level, commitment_loss in enumerate(loss_dict["commitment_losses"]):
                        wandb_log_dict[f"loss/commitment_level_{level}"] = commitment_loss.item()

                    if loss_dict["codebook_losses"]:  # Only for non-EMA
                        for level, codebook_loss in enumerate(loss_dict["codebook_losses"]):
                            wandb_log_dict[f"loss/codebook_level_{level}"] = codebook_loss.item()

                    # Add codebook metrics
                    wandb_log_dict["metrics/unique_ids_proportion_train"] = unique_ids_proportion
                    for level, usage in enumerate(codebook_usage):
                        wandb_log_dict[f"metrics/codebook_usage_train_level_{level}"] = usage

                    wandb.log(wandb_log_dict)

                # Validation and checkpointing
                if global_step % config.steps_per_val_log == 0 and val_loader is not None:
                    # Run validation and compute all metrics
                    metrics = evaluate(model, val_loader, data_loader, device, global_step, epoch + 1)

                    # Save checkpoint and update best model if improved
                    best_loss = save_checkpoint(
                        model, optimizer, scheduler, metrics, config, global_step, epoch, best_loss
                    )

                    model.train()

                # Codebook reset
                if config.reset_unused_codes and global_step % config.steps_per_codebook_reset == 0:
                    if config.scheduler_type == "cosine_with_warmup" and global_step < config.warmup_steps:
                        logger.debug(f"Step {global_step:05d} - Skipping codebook reset during warmup")
                    else:
                        model.eval()

                        # Calculate current codebook usage
                        codebook_usage = model.calculate_codebook_usage()
                        usage_str = "/".join([f"{u:.2f}" for u in codebook_usage])
                        reset_performed = []

                        # Get a sample batch for reset
                        reset_batch = next(iter(data_loader))

                        # Reset unused codes for each VQ layer
                        for level, vq_layer in enumerate(model.vq_layers):
                            if isinstance(vq_layer, VectorQuantizer) and not isinstance(vq_layer, EMAVectorQuantizer):
                                if codebook_usage[level] < config.codebook_usage_threshold:
                                    with torch.no_grad():
                                        z = model.encode(reset_batch.to(device))
                                        residual = z
                                        for i in range(level):
                                            vq_out = model.vq_layers[i](residual)
                                            residual = residual - vq_out.quantized

                                        vq_layer.reset_unused_codes(residual)
                                        reset_performed.append(f"{level}({codebook_usage[level]:.2f})")

                                vq_layer.reset_usage_count()

                        model.train()

        # Handle incomplete gradient accumulation at end of epoch
        if (batch_idx + 1) % config.gradient_accumulation_steps != 0:
            grad_norm = get_gradient_norm(model)

            if config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            global_step += 1
            logger.debug(
                f"Step {global_step:05d} | Epoch {epoch + 1:03d} - Applied remaining gradients at epoch end (grad norm: {grad_norm:.2f})"
            )


def get_gradient_norm(model: nn.Module) -> float:
    """Calculate the L2 norm of gradients across all model parameters."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
    return total_norm.item()


def get_loss(model: RQVAE, val_loader: torch.utils.data.DataLoader, device: str) -> float:
    """Validate model on validation set."""
    model.eval()
    total_loss = 0
    batch_count = 0

    with torch.no_grad():
        for data in val_loader:
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(device)

            _, _, loss_dict = model(data)
            total_loss += loss_dict["loss"].item()
            batch_count += 1

    return total_loss / batch_count


def evaluate(
    model: RQVAE,
    val_loader: DataLoader,
    train_loader: DataLoader,  # For sample batch metrics
    device: str,
    global_step: int,
    epoch: int,
) -> dict:
    """Run validation

    Args:
        model: RQVAE model to validate
        val_loader: Validation data loader
        train_loader: Training data loader (for sample batch metrics)
        device: Device to run on
        global_step: Current training step
        epoch: Current epoch (1-indexed)

    Returns:
        Dictionary containing all metrics including val_loss
    """
    model.eval()

    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        if isinstance(sample_batch, (list, tuple)):
            sample_batch = sample_batch[0]
        sample_batch = sample_batch.to(device)

        _, indices, loss_dict_sample = model(sample_batch)

        # Compute all metrics
        codebook_usage = model.calculate_codebook_usage()
        avg_residual_norm = model.calculate_avg_residual_norm(loss_dict_sample["residual"])
        semantic_ids = torch.stack(indices, dim=-1)
        unique_ids_proportion = model.calculate_unique_ids_proportion(semantic_ids)

    # Compute validation loss
    val_loss = get_loss(model, val_loader, device)
    usage_str = "/".join([f"{u:.2f}" for u in codebook_usage])

    logger.info(
        f"Step {global_step:05d} | Epoch {epoch:05d} | Val loss: {val_loss:.2e} | "
        f"Codebook usage: {usage_str} | Avg residual norm: {avg_residual_norm:.3f} | "
        f"Unique ids: {unique_ids_proportion:.1%}"
    )

    wandb_dict = {
        "loss/validation": val_loss,
        "metrics/avg_residual_norm": avg_residual_norm,
        "metrics/unique_ids_proportion": unique_ids_proportion,
        "epoch": epoch,
        "step": global_step,
    }

    for level, usage in enumerate(codebook_usage):
        wandb_dict[f"metrics/codebook_usage_level_{level}"] = usage

    wandb.log(wandb_dict)

    return {
        "val_loss": val_loss,
        "codebook_usage": codebook_usage,
        "codebook_usage_str": usage_str,
        "avg_residual_norm": avg_residual_norm,
        "unique_ids_proportion": unique_ids_proportion,
    }


def save_checkpoint(
    model: RQVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    metrics: dict,
    config: RQVAEConfig,
    global_step: int,
    epoch: int,
    best_loss: float,
) -> float:
    """Save checkpoint and handle best model tracking.

    Args:
        model: Model to save
        optimizer: Optimizer with state to save
        scheduler: Optional scheduler with state to save
        metrics: Dictionary of metrics from validation
        config: Training configuration
        global_step: Current training step
        epoch: Current epoch (0-indexed for checkpoint compatibility)
        best_loss: Current best validation loss

    Returns:
        Updated best_loss value
    """
    # Ensure checkpoint directory exists
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "epoch": epoch,
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_loss": metrics["val_loss"],
        "config": config.__dict__,
    }

    checkpoint_path = config.checkpoint_dir / f"checkpoint_step_{global_step}.pth"
    torch.save(checkpoint_data, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Check if this is the best model
    if metrics["val_loss"] < best_loss:
        best_loss = metrics["val_loss"]
        best_model_path = config.checkpoint_dir / "best_model.pth"
        torch.save(checkpoint_data, best_model_path)
        logger.info(f"Saved best model with val_loss: {best_loss:.4e}")

        # Save best model
        artifact = wandb.Artifact(
            f"rqvae-best-{config.category}",
            type="model",
            metadata={
                "val_loss": best_loss,
                "codebook_usage": metrics["codebook_usage_str"],
                "avg_residual_norm": metrics["avg_residual_norm"],
                "unique_ids_proportion": metrics["unique_ids_proportion"],
                "step": global_step,
                "epoch": epoch + 1,  # Display as 1-indexed
                "codebook_levels": config.codebook_quantization_levels,
                "codebook_size": config.codebook_size,
                "embedding_dim": config.codebook_embedding_dim,
            },
        )
        artifact.add_file(str(best_model_path))
        wandb.log_artifact(artifact)

    return best_loss


if __name__ == "__main__":
    config = RQVAEConfig()
    device_manager = DeviceManager(logger)
    device = device_manager.device

    run_name = f"rqvae-L{config.codebook_quantization_levels}-C{config.codebook_size}-D{config.codebook_embedding_dim}"
    run = wandb.init(project="rqvae", name=run_name, config=config.__dict__)
    config.log_config()

    dataset = EmbeddingDataset(str(config.embeddings_path))
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Train size: {len(train_dataset):,}, Val size: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=16,  # Increased for better throughput
        pin_memory=device_manager.supports_pin_memory,
        prefetch_factor=8,  # Pre-fetch next batches
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=False,  # Include partial batches to avoid losing data
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,  # Match train loader
        pin_memory=device_manager.supports_pin_memory,
        prefetch_factor=4,
        persistent_workers=True,
    )

    model = RQVAE(config)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_rqvae(model=model, data_loader=train_loader, val_loader=val_loader, config=config, device=device)

    final_path = config.checkpoint_dir / "final_model.pth"
    logger.info(f"Saving final model to {final_path}")
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, final_path)

    logger.info("Training complete!")
