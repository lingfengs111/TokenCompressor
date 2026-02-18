"""BERT4Rec-style backbone adapted for soft-patch training."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.patch import MetaPatch
from backbones.modules import apply_head as apply_head_fn, head_parameters as head_parameters_fn, sequence_summary


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        if hidden_units % num_heads != 0:
            raise ValueError("hidden_units must be a multiple of num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.hidden_units = hidden_units

        self.qkv = nn.Linear(hidden_units, 3 * hidden_units)
        self.proj = nn.Linear(hidden_units, hidden_units)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.hidden_units, dim=2)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        probs = torch.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        y = torch.matmul(probs, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.attn = MultiHeadSelfAttention(hidden_units, num_heads, dropout_rate)
        self.ln_2 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units * 4, hidden_units),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.ffn(self.ln_2(x))
        return x


class Bert4Rec(nn.Module):
    """BERT4Rec-style encoder with SASRec-compatible interface."""

    def __init__(self, config, item_num: int):
        super().__init__()
        self.config = config
        self.item_num = item_num
        self.hidden_units = config.hidden_units
        self.patch_len = config.patch_len
        self.max_seq_length = config.max_seq_length

        num_layers = getattr(config, "bert_num_layers", getattr(config, "num_blocks", 2))
        num_heads = getattr(config, "bert_num_heads", getattr(config, "num_heads", 1))
        dropout_rate = getattr(config, "bert_dropout", getattr(config, "dropout_rate", 0.2))

        self.item_emb = nn.Embedding(item_num + 2, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_seq_length + self.patch_len + 1, self.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_units, num_heads, dropout_rate) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(self.hidden_units, eps=1e-8)

        # Trainable projection head (only head is adapted in inner loop)
        self.proj_linear = nn.Linear(self.hidden_units, self.hidden_units, bias=True)
        self.proj_ln = nn.LayerNorm(self.hidden_units, eps=1e-8)

        # Meta-patch module (outer loop)
        self.meta_patch = MetaPatch(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        init_range = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=init_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _sequence_summary(self, item_embs: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pool = getattr(self.config, "gating_pool", "last")
        return sequence_summary(item_embs, input_ids, pool=pool)

    def head_parameters(self) -> List[torch.Tensor]:
        return head_parameters_fn(self.config, self.proj_linear, self.proj_ln)

    def apply_head(self, hidden_states: torch.Tensor, head_params: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        return apply_head_fn(hidden_states, self.config, self.proj_linear, self.proj_ln, head_params=head_params)

    def _build_attention_mask(self, input_ids: torch.Tensor, patch_len: int) -> torch.Tensor:
        attention_mask = (input_ids > 0).long()
        if patch_len > 0:
            patch_mask = torch.ones((attention_mask.size(0), patch_len), device=input_ids.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([patch_mask, attention_mask], dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward_features(
        self,
        input_ids: torch.Tensor,
        patch_params: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        batch_size, seq_length = input_ids.size()

        item_embs = self.item_emb(input_ids)

        seq_summary, _ = self._sequence_summary(item_embs, input_ids)
        if use_patch and self.patch_len > 0:
            if patch_params is None:
                patch_emb, gating_weights = self.meta_patch(seq_summary)
            else:
                patch_emb, gating_weights = self.meta_patch.forward_with_eta(seq_summary, patch_params)
        else:
            patch_emb = item_embs.new_zeros((batch_size, 0, self.hidden_units))
            gating_weights = None

        positions = torch.arange(1, seq_length + 1, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        if getattr(self.config, "right_align_positions", True):
            offset = max(self.max_seq_length - seq_length, 0)
            if offset > 0:
                positions = positions + offset
        positions = positions * (input_ids != 0).long()
        pos_embs = self.pos_emb(positions)

        if use_patch and self.patch_len > 0:
            hidden_states = torch.cat([patch_emb, item_embs + pos_embs], dim=1)
        else:
            hidden_states = item_embs + pos_embs

        hidden_states = self.emb_dropout(hidden_states)
        attn_mask = self._build_attention_mask(input_ids, self.patch_len if use_patch else 0)

        for block in self.blocks:
            hidden_states = block(hidden_states, attn_mask)

        hidden_states = self.ln_f(hidden_states)

        if return_gating:
            return hidden_states, gating_weights
        return hidden_states

    def get_gating_weights(self, input_ids: torch.Tensor, patch_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        item_embs = self.item_emb(input_ids)
        seq_summary, _ = self._sequence_summary(item_embs, input_ids)
        if patch_params is None:
            _, weights = self.meta_patch(seq_summary)
        else:
            _, weights = self.meta_patch.forward_with_eta(seq_summary, patch_params)
        return weights

    def forward(
        self,
        input_ids: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        neg_ids: Optional[torch.Tensor] = None,
        patch_params: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ):
        if pos_ids is not None and neg_ids is not None:
            return self.training_step(
                input_ids,
                pos_ids,
                neg_ids,
                patch_params=patch_params,
                return_gating=return_gating,
                use_patch=use_patch,
            )
        return self.forward_features(
            input_ids,
            patch_params=patch_params,
            return_gating=return_gating,
            use_patch=use_patch,
        )

    def predict(
        self,
        input_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        patch_params: Optional[torch.Tensor] = None,
        head_params: Optional[List[torch.Tensor]] = None,
        use_patch: bool = True,
        use_head: bool = True,
    ) -> torch.Tensor:
        hidden_states = self.forward_features(input_ids, patch_params=patch_params, use_patch=use_patch)
        if use_patch and self.patch_len > 0:
            hidden_states = hidden_states[:, self.patch_len :, :]
        final_hidden = hidden_states[:, -1, :]
        if use_head:
            final_hidden = self.apply_head(final_hidden, head_params=head_params)
        candidate_embs = self.item_emb(candidate_ids)
        scores = torch.bmm(candidate_embs, final_hidden.unsqueeze(-1)).squeeze(-1)
        return scores

    def training_step(
        self,
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        patch_params: Optional[torch.Tensor] = None,
        head_params: Optional[List[torch.Tensor]] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        hidden_states, gating_weights = self.forward_features(
            input_ids, patch_params=patch_params, return_gating=True, use_patch=use_patch
        )
        if use_patch and self.patch_len > 0:
            hidden_states = hidden_states[:, self.patch_len :, :]

        projected = self.apply_head(hidden_states, head_params=head_params)
        pos_embs = self.item_emb(pos_ids)
        neg_embs = self.item_emb(neg_ids)

        pos_logits = (projected * pos_embs).sum(dim=-1)
        if neg_embs.dim() == 4:
            neg_logits = (projected.unsqueeze(2) * neg_embs).sum(dim=-1)
        else:
            neg_logits = (projected * neg_embs).sum(dim=-1)

        if return_gating:
            return pos_logits, neg_logits, gating_weights
        return pos_logits, neg_logits
