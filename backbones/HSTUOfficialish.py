"""Official-inspired dense HSTU variant for backbone and patch experiments.

This module keeps the current project-facing HSTU interface so it can plug into
the existing backbone / patch / A2-A3 training code, but makes a few structural
choices closer to Meta's original HSTU/STU design:

- block norms are always parameterized
- only the content branches (u/v) receive the SiLU activation
- q/k stay linear before attention scoring
- the rel-bias path uses sqrt(attention_dim) scaling instead of seq_len scaling
- PersRec is intentionally not supported in this variant

It is still a dense padded implementation, not a jagged-kernel reproduction of
the official repository.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from backbones.HSTU import HSTU, HSTUBlock


class HSTUOfficialishBlock(HSTUBlock):
    """Dense HSTU block with a structure closer to the official STU layer."""

    def _masked_softmax_attention(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._match_attention_mask(logits, attention_mask)
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        weights = F.softmax(masked_logits, dim=-1)
        valid_row = torch.isfinite(masked_logits).any(dim=-1, keepdim=True)
        weights = torch.where(mask & valid_row, weights, torch.zeros_like(weights))
        return self.attn_dropout(weights)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        normed_x = self._norm_input(x)
        uvqk = torch.matmul(normed_x, self.uvqk)

        u, v, q, k = torch.split(
            uvqk,
            [
                self.linear_dim * self.num_heads,
                self.linear_dim * self.num_heads,
                self.attention_dim * self.num_heads,
                self.attention_dim * self.num_heads,
            ],
            dim=-1,
        )
        if self.linear_activation == "silu":
            u = F.silu(u)
            v = F.silu(v)

        if self.normalization in {"rel_bias", "hstu_rel_bias"}:
            q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.attention_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.linear_dim)
            qk_attn = torch.einsum("bthd,bshd->bhts", q, k)
            if self.rel_pos_bias is not None:
                qk_attn = qk_attn + self.rel_pos_bias(seq_len, qk_attn.dtype).unsqueeze(0).unsqueeze(0)
            qk_attn = F.silu(qk_attn) / math.sqrt(self.attention_dim)
            mask = self._match_attention_mask(qk_attn, attention_mask)
            qk_attn = qk_attn * mask.to(dtype=qk_attn.dtype)
            attn_output = torch.einsum("bhts,bshd->bthd", qk_attn, v).reshape(
                batch_size, seq_len, self.num_heads * self.linear_dim
            )
        elif self.normalization in {"softmax_rel_bias", "softmax1_rel_bias"}:
            q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.attention_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.linear_dim)
            qk_attn = torch.einsum("bthd,bshd->bhts", q, k)
            if self.rel_pos_bias is not None:
                qk_attn = qk_attn + self.rel_pos_bias(seq_len, qk_attn.dtype).unsqueeze(0).unsqueeze(0)
            qk_attn = qk_attn / math.sqrt(self.attention_dim)
            if self.normalization == "softmax1_rel_bias":
                qk_attn = self._softmax1_attention(qk_attn, attention_mask)
            else:
                qk_attn = self._masked_softmax_attention(qk_attn, attention_mask)
            attn_output = torch.einsum("bhts,bshd->bthd", qk_attn, v).reshape(
                batch_size, seq_len, self.num_heads * self.linear_dim
            )
        else:
            raise ValueError(f"Unknown normalization method {self.normalization}")

        if self.concat_ua:
            a = self._norm_attn_output(attn_output)
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * self._norm_attn_output(attn_output)

        return self.o(F.dropout(o_input, p=self.dropout_ratio, training=self.training)) + x


class HSTUOfficialish(HSTU):
    """Project-compatible HSTU variant with a more official-like block structure."""

    def __init__(self, config, item_num: int):
        if bool(getattr(config, "persrec_enable", False)):
            raise ValueError("hstu_officialish does not support persrec_enable.")

        # Official STU layers use parameterized input/output norms. Force that on
        # here so this backbone stays distinct from the compatibility HSTU path.
        if getattr(config, "hstu_attn_dropout", None) is None:
            setattr(config, "hstu_attn_dropout", float(getattr(config, "dropout_rate", 0.0)))
        setattr(config, "hstu_parametric_block_norm", True)
        super().__init__(config, item_num=item_num)

        num_heads = int(getattr(config, "num_heads", 1))
        default_subdim = max(1, self.hidden_units // max(num_heads, 1))
        linear_dim = int(getattr(config, "hstu_linear_dim", default_subdim) or default_subdim)
        attention_dim = int(getattr(config, "hstu_attention_dim", default_subdim) or default_subdim)
        linear_activation = str(getattr(config, "hstu_linear_activation", "silu") or "silu").lower()
        normalization = str(getattr(config, "hstu_normalization", "rel_bias") or "rel_bias").lower()
        concat_ua = bool(getattr(config, "hstu_concat_ua", False))
        epsilon = float(getattr(config, "hstu_epsilon", 1e-6))
        attn_dropout = float(getattr(config, "hstu_attn_dropout", getattr(config, "dropout_rate", 0.0)))
        enable_relative_attention_bias = bool(getattr(config, "hstu_enable_relative_attention_bias", False))

        self.blocks = torch.nn.ModuleList(
            [
                HSTUOfficialishBlock(
                    embedding_dim=self.hidden_units,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    num_heads=num_heads,
                    dropout_ratio=float(getattr(config, "dropout_rate", 0.0)),
                    attn_dropout_ratio=attn_dropout,
                    linear_activation=linear_activation,
                    normalization=normalization,
                    concat_ua=concat_ua,
                    epsilon=epsilon,
                    enable_relative_attention_bias=enable_relative_attention_bias,
                    max_length=self.total_seq_length,
                    parametric_block_norm=True,
                )
                for _ in range(int(getattr(config, "num_blocks", 1)))
            ]
        )
