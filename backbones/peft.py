"""PEFT utilities: LoRA linear layers and adapters."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapters (trainable low-rank update)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / float(r)
        self.lora_dropout = nn.Dropout(lora_dropout)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # LoRA init: A random, B zero so initial delta is zero
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> "LoRALinear":
        if not isinstance(linear, nn.Linear):
            raise TypeError("from_linear expects nn.Linear")
        lora = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        lora = lora.to(device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            lora.weight.copy_(linear.weight)
            if linear.bias is not None and lora.bias is not None:
                lora.bias.copy_(linear.bias)
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return result + lora_out * self.scaling


class Adapter(nn.Module):
    """Simple bottleneck adapter with residual connection."""

    def __init__(
        self,
        hidden_size: int,
        bottleneck: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        init: str = "zero",
    ) -> None:
        super().__init__()
        if bottleneck <= 0:
            raise ValueError("Adapter bottleneck must be > 0")
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        if init == "zero":
            nn.init.zeros_(self.up.weight)
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation == "swish":
            return x * torch.sigmoid(x)
        return F.gelu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.down(x)
        z = self._act(z)
        z = self.dropout(z)
        z = self.up(z)
        return x + z


def apply_lora_to_sasrec(model: nn.Module, r: int, alpha: float, dropout: float) -> None:
    """Replace SASRec attention+FFN Linear layers with LoRA-wrapped versions."""
    if not hasattr(model, "blocks"):
        raise ValueError("Model does not look like SASRec (missing blocks)")
    for block in model.blocks:
        if hasattr(block, "attn"):
            block.attn.c_attn = LoRALinear.from_linear(block.attn.c_attn, r=r, lora_alpha=alpha, lora_dropout=dropout)
            block.attn.c_proj = LoRALinear.from_linear(block.attn.c_proj, r=r, lora_alpha=alpha, lora_dropout=dropout)
        if hasattr(block, "ffn"):
            block.ffn.fc1 = LoRALinear.from_linear(block.ffn.fc1, r=r, lora_alpha=alpha, lora_dropout=dropout)
            block.ffn.fc2 = LoRALinear.from_linear(block.ffn.fc2, r=r, lora_alpha=alpha, lora_dropout=dropout)


def apply_lora_to_linrec(model: nn.Module, r: int, alpha: float, dropout: float) -> None:
    """Replace LinRec attention+FFN Linear layers with LoRA-wrapped versions."""
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layer"):
        raise ValueError("Model does not look like LinRec (missing encoder layers)")
    for layer in model.encoder.layer:
        mha = layer.multi_head_attention
        mha.query = LoRALinear.from_linear(mha.query, r=r, lora_alpha=alpha, lora_dropout=dropout)
        mha.key = LoRALinear.from_linear(mha.key, r=r, lora_alpha=alpha, lora_dropout=dropout)
        mha.value = LoRALinear.from_linear(mha.value, r=r, lora_alpha=alpha, lora_dropout=dropout)
        mha.dense = LoRALinear.from_linear(mha.dense, r=r, lora_alpha=alpha, lora_dropout=dropout)
        ff = layer.feed_forward
        ff.dense_1 = LoRALinear.from_linear(ff.dense_1, r=r, lora_alpha=alpha, lora_dropout=dropout)
        ff.dense_2 = LoRALinear.from_linear(ff.dense_2, r=r, lora_alpha=alpha, lora_dropout=dropout)
