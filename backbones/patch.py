"""Patch bank and gating module shared across backbones."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaPatch(nn.Module):
    """Learnable patch bank + gating network flattened into a single eta tensor."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_patches = max(1, config.num_patches)
        self.patch_len = max(0, config.patch_len)
        self.hidden_units = config.hidden_units
        self.use_gating = config.use_gating
        self.gating_hidden_dim = config.gating_hidden_dim

        self.patch_param_size = self.num_patches * self.patch_len * self.hidden_units
        if self.use_gating:
            self.gate_param_size = (
                self.gating_hidden_dim * self.hidden_units
                + self.gating_hidden_dim
                + self.num_patches * self.gating_hidden_dim
                + self.num_patches
            )
        else:
            self.gate_param_size = self.num_patches

        patch_init = torch.randn(self.patch_param_size) * config.patch_init_std
        gate_init = torch.randn(self.gate_param_size) * config.gating_init_std
        self.eta = nn.Parameter(torch.cat([patch_init, gate_init], dim=0))
        self.register_buffer("kmeans_centers", torch.empty(0), persistent=False)
        self.register_buffer("user_to_patch", torch.empty(0, dtype=torch.long), persistent=False)

    def set_kmeans_centers(self, centers: torch.Tensor) -> None:
        if centers.dim() != 2 or centers.size(1) != self.hidden_units:
            raise ValueError("kmeans centers must have shape [num_patches, hidden_units].")
        if centers.size(0) != self.num_patches:
            if centers.size(0) < self.num_patches:
                repeats = self.num_patches - centers.size(0)
                extra = centers[torch.arange(repeats) % centers.size(0)].clone()
                centers = torch.cat([centers, extra], dim=0)
            else:
                centers = centers[: self.num_patches]
        self.kmeans_centers = centers.to(device=self.eta.device, dtype=self.eta.dtype)

    def set_user_table(self, user_to_patch: torch.Tensor) -> None:
        if user_to_patch.dim() != 1:
            raise ValueError("user_to_patch must be a 1D tensor of patch indices.")
        self.user_to_patch = user_to_patch.to(device=self.eta.device, dtype=torch.long)

    def _split_eta(self, eta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_flat = eta[: self.patch_param_size]
        gate_flat = eta[self.patch_param_size :]
        return patch_flat, gate_flat

    def forward(self, seq_emb: torch.Tensor, user_ids: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_with_eta(seq_emb, self.eta, user_ids=user_ids)

    def forward_with_eta(
        self, seq_emb: torch.Tensor, eta: torch.Tensor, user_ids: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.patch_len == 0:
            empty_patch = seq_emb.new_zeros((seq_emb.size(0), 0, self.hidden_units))
            empty_weights = seq_emb.new_zeros((seq_emb.size(0), self.num_patches))
            return empty_patch, empty_weights

        patch_flat, gate_flat = self._split_eta(eta)
        patch_bank = patch_flat.view(self.num_patches, self.patch_len, self.hidden_units)

        routing = getattr(self.config, "patch_routing", "learned")
        if routing != "learned":
            if routing == "single" or self.num_patches == 1:
                weights = seq_emb.new_zeros((seq_emb.size(0), self.num_patches))
                weights[:, 0] = 1.0
            elif routing == "random":
                idx = torch.randint(0, self.num_patches, (seq_emb.size(0),), device=seq_emb.device)
                weights = F.one_hot(idx, num_classes=self.num_patches).to(seq_emb.dtype)
            elif routing == "kmeans":
                if self.kmeans_centers.numel() == 0:
                    raise RuntimeError("KMeans routing requested but centers are not set.")
                centers = self.kmeans_centers
                if centers.device != seq_emb.device:
                    centers = centers.to(seq_emb.device)
                dist = torch.cdist(seq_emb, centers)
                idx = dist.argmin(dim=1)
                weights = F.one_hot(idx, num_classes=self.num_patches).to(seq_emb.dtype)
            elif routing == "user_table":
                if self.user_to_patch.numel() == 0:
                    raise RuntimeError("User-table routing requested but user_to_patch is not set.")
                if user_ids is None:
                    raise RuntimeError("User-table routing requires user_ids.")
                if user_ids.device != self.user_to_patch.device:
                    user_ids = user_ids.to(self.user_to_patch.device)
                idx = self.user_to_patch[user_ids]
                weights = F.one_hot(idx, num_classes=self.num_patches).to(seq_emb.dtype).to(seq_emb.device)
            else:
                raise ValueError(f"Unknown patch_routing: {routing}")
            patch = torch.einsum("bp,plh->blh", weights, patch_bank)
            return patch, weights

        if self.use_gating:
            idx = 0
            w1 = gate_flat[idx : idx + self.gating_hidden_dim * self.hidden_units].view(
                self.gating_hidden_dim, self.hidden_units
            )
            idx += self.gating_hidden_dim * self.hidden_units
            b1 = gate_flat[idx : idx + self.gating_hidden_dim]
            idx += self.gating_hidden_dim
            w2 = gate_flat[idx : idx + self.num_patches * self.gating_hidden_dim].view(
                self.num_patches, self.gating_hidden_dim
            )
            idx += self.num_patches * self.gating_hidden_dim
            b2 = gate_flat[idx : idx + self.num_patches]

            hidden = F.relu(F.linear(seq_emb, w1, b1))
            logits = F.linear(hidden, w2, b2)
        else:
            logits = gate_flat.view(1, -1).expand(seq_emb.size(0), -1)

        temp = float(getattr(self.config, "gating_temperature", 1.0))
        if temp <= 0:
            temp = 1e-6
        logits = logits / temp
        noise_std = float(getattr(self.config, "gating_noise_std", 0.0))
        if noise_std > 0 and self.training:
            logits = logits + noise_std * torch.randn_like(logits)
        weights = torch.softmax(logits, dim=-1)
        patch = torch.einsum("bp,plh->blh", weights, patch_bank)
        return patch, weights

    def zero_eta(self) -> torch.Tensor:
        return torch.zeros_like(self.eta)
