"""Simple device management for training scripts."""

import logging
from typing import Optional

import torch


class DeviceManager:
    """Simple device management for training scripts."""

    def __init__(self, logger: logging.Logger, preferred_device: Optional[str] = None, gpu_id: Optional[int] = None):
        """Initialize device manager.

        Args:
            logger: Logger instance from the main script
            preferred_device: Explicit device string (e.g., "cuda:1", "cpu", "mps")
            gpu_id: CUDA GPU index to use (e.g., 0, 1)
        """
        self.logger = logger
        self.device = self._select_device(preferred_device, gpu_id)
        self._configure_device()

    def _select_device(self, preferred_device: Optional[str], gpu_id: Optional[int]) -> str:
        """Select the best available device, honoring user preference when possible."""
        # User requested an explicit device string
        if preferred_device:
            try:
                requested = torch.device(preferred_device)
                if requested.type == "cuda" and not torch.cuda.is_available():
                    self.logger.warning("Requested CUDA device but CUDA is not available. Falling back.")
                elif requested.type == "mps" and not torch.backends.mps.is_available():
                    self.logger.warning("Requested MPS device but it is not available. Falling back.")
                else:
                    if requested.type == "cuda" and requested.index is not None:
                        torch.cuda.set_device(requested)
                    self.logger.info(f"Using requested device: {requested}")
                    return str(requested)
            except Exception as exc:  # Broad except to avoid crashing on bad input
                self.logger.warning(f"Could not use requested device '{preferred_device}': {exc}. Falling back.")

        # User provided a GPU id without an explicit device string
        if gpu_id is not None and torch.cuda.is_available():
            if 0 <= gpu_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{gpu_id}")
                torch.cuda.set_device(device)
                self.logger.info(f"Using CUDA device cuda:{gpu_id}")
                return str(device)
            self.logger.warning(
                f"Requested gpu_id {gpu_id} is out of range (device count: {torch.cuda.device_count()}). Falling back."
            )

        # Auto-selection fallback
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.logger.info(f"Using device: {device}")
        return device

    def _configure_device(self):
        """Apply device-specific configurations."""
        if self.device.startswith("cuda"):
            # Enable TF32 on Ampere GPUs for faster training
            torch.set_float32_matmul_precision("high")
            self.logger.info("Enabled TF32 precision for matrix multiplications")

    @property
    def is_cuda(self) -> bool:
        """Check if device is CUDA."""
        return self.device.startswith("cuda")

    @property
    def is_mps(self) -> bool:
        """Check if device is MPS (Apple Silicon)."""
        return self.device == "mps"

    @property
    def supports_compile(self) -> bool:
        """Check if device supports torch.compile."""
        return self.device.startswith("cuda")

    @property
    def supports_pin_memory(self) -> bool:
        """Check if device supports pinned memory for DataLoader."""
        return self.device.startswith("cuda")
