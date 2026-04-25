"""LONGER-inspired causal backbone for ID-only sequential recommendation.

This is a lightweight adaptation of the LONGER ideas to the existing
leave-two-out next-item pipeline:
- learned global anchor tokens
- token grouping + lightweight InnerTrans within each group
- compressed group-level attention trunk

The design keeps the training interface compatible with the existing
backbone family by returning token-aligned hidden states.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.modules import apply_head as apply_head_fn, head_parameters as head_parameters_fn, sequence_summary
from backbones.patch import MetaPatch


def _causal_mask(query_len: int, key_len: int, allow_prefix: int = 0, device=None) -> torch.Tensor:
    """Return a boolean attention mask where True marks disallowed positions."""
    query_idx = torch.arange(query_len, device=device).unsqueeze(1)
    key_idx = torch.arange(key_len, device=device).unsqueeze(0)
    if allow_prefix > 0:
        merged_key_idx = (key_idx - allow_prefix).clamp_min(0)
        allowed = (key_idx < allow_prefix) | (merged_key_idx <= query_idx)
    else:
        allowed = key_idx <= query_idx
    return ~allowed


class FeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float, expansion: int = 4) -> None:
        super().__init__()
        inner_dim = max(hidden_units, hidden_units * int(expansion))
        self.fc1 = nn.Linear(hidden_units, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SelfAttentionBlock(nn.Module):
    """Pre-LN self-attention block with an additive attention mask."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_units,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.ln_2 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn = FeedForward(hidden_units, dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.ln_1(x)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.ln_2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention block for compressed group queries."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ln_kv = nn.LayerNorm(hidden_units, eps=1e-8)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_units,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.ln_ff = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn = FeedForward(hidden_units, dropout_rate)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.ln_q(query)
        kv = self.ln_kv(context)
        attn_out, _ = self.attn(
            q,
            kv,
            kv,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = query + self.dropout(attn_out)
        x = x + self.ffn(self.ln_ff(x))
        return x


class LONGER(nn.Module):
    """LONGER-inspired backbone with causal group compression and global anchors."""

    def __init__(self, config, item_num: int):
        super().__init__()
        self.config = config
        self.item_num = item_num
        self.max_seq_length = int(config.max_seq_length)
        self.hidden_units = int(config.hidden_units)
        self.patch_len = int(getattr(config, "patch_len", 0) or 0)
        self.global_token_count = max(0, int(getattr(config, "longer_global_tokens", 4) or 0))
        self.merge_size = max(1, int(getattr(config, "longer_merge_size", 4) or 1))
        self.merge_pool = str(getattr(config, "longer_merge_pool", "last") or "last").lower()
        self.inner_num_layers = max(1, int(getattr(config, "longer_inner_num_layers", 1) or 1))

        if self.hidden_units % int(config.num_heads) != 0:
            raise ValueError("LONGER requires hidden_units to be divisible by num_heads.")
        if self.merge_pool not in {"last", "mean"}:
            raise ValueError(f"Unsupported longer_merge_pool: {self.merge_pool}")

        self.item_emb = nn.Embedding(item_num + 2, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_seq_length + 1, self.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(config.dropout_rate)

        if self.global_token_count > 0:
            self.global_tokens = nn.Parameter(torch.empty(self.global_token_count, self.hidden_units))
        else:
            self.register_parameter("global_tokens", None)

        self.inner_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    hidden_units=self.hidden_units,
                    num_heads=int(config.num_heads),
                    dropout_rate=float(config.dropout_rate),
                )
                for _ in range(self.inner_num_layers)
            ]
        )
        self.cross_block = CrossAttentionBlock(
            hidden_units=self.hidden_units,
            num_heads=int(config.num_heads),
            dropout_rate=float(config.dropout_rate),
        )
        trunk_depth = max(0, int(config.num_blocks) - 1)
        self.trunk_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    hidden_units=self.hidden_units,
                    num_heads=int(config.num_heads),
                    dropout_rate=float(config.dropout_rate),
                )
                for _ in range(trunk_depth)
            ]
        )

        self.ln_f = nn.LayerNorm(self.hidden_units, eps=1e-8)
        self.proj_linear = nn.Linear(self.hidden_units, self.hidden_units, bias=True)
        self.proj_ln = nn.LayerNorm(self.hidden_units, eps=1e-8)
        self.meta_patch = MetaPatch(config)

        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, LONGER) and module.global_tokens is not None:
            nn.init.normal_(module.global_tokens, mean=0.0, std=0.02)

    def _freeze_backbone(self) -> None:
        for p in self.item_emb.parameters():
            p.requires_grad = False
        for p in self.pos_emb.parameters():
            p.requires_grad = False
        if self.global_tokens is not None:
            self.global_tokens.requires_grad_(False)
        for block in self.inner_blocks:
            for p in block.parameters():
                p.requires_grad = False
        for p in self.cross_block.parameters():
            p.requires_grad = False
        for block in self.trunk_blocks:
            for p in block.parameters():
                p.requires_grad = False
        for p in self.ln_f.parameters():
            p.requires_grad = False

    def _sequence_summary(self, item_embs: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pool = getattr(self.config, "gating_pool", "last")
        return sequence_summary(item_embs, input_ids, pool=pool)

    def head_parameters(self) -> List[torch.Tensor]:
        return head_parameters_fn(self.config, self.proj_linear, self.proj_ln)

    def apply_head(self, hidden_states: torch.Tensor, head_params: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        return apply_head_fn(hidden_states, self.config, self.proj_linear, self.proj_ln, head_params=head_params)

    def _strip_patch_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.patch_len <= 0:
            return hidden_states
        prefix_len = int(getattr(self.config, "prefix_len", 0) or 0)
        patch_after_prefix = bool(getattr(self.config, "patch_after_prefix", False))
        if patch_after_prefix and prefix_len > 0 and hidden_states.size(1) >= prefix_len + self.patch_len:
            return torch.cat(
                [hidden_states[:, :prefix_len, :], hidden_states[:, prefix_len + self.patch_len :, :]],
                dim=1,
            )
        return hidden_states[:, self.patch_len :, :]

    def strip_patch_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._strip_patch_tokens(hidden_states)

    def _build_grouped_inputs(
        self,
        hidden_states: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        group_count = max(1, math.ceil(seq_len / self.merge_size))
        padded_len = group_count * self.merge_size
        pad_tokens = padded_len - seq_len

        if pad_tokens > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_tokens))
            valid_mask = F.pad(valid_mask, (0, pad_tokens), value=False)

        grouped = hidden_states.view(batch_size, group_count, self.merge_size, hidden_dim)
        grouped_valid = valid_mask.view(batch_size, group_count, self.merge_size)
        return grouped, grouped_valid, group_count, padded_len

    def _run_inner_trans(self, grouped: torch.Tensor, grouped_valid: torch.Tensor) -> torch.Tensor:
        batch_size, group_count, group_size, hidden_dim = grouped.shape
        flat_tokens = grouped.view(batch_size * group_count, group_size, hidden_dim)
        flat_valid = grouped_valid.view(batch_size * group_count, group_size)
        active = flat_valid.any(dim=1)

        flat_out = flat_tokens.new_zeros(flat_tokens.shape)
        if not active.any():
            return flat_out.view(batch_size, group_count, group_size, hidden_dim)

        active_tokens = flat_tokens[active]
        active_valid = flat_valid[active]
        attn_mask = _causal_mask(group_size, group_size, device=grouped.device)
        key_padding_mask = ~active_valid

        for block in self.inner_blocks:
            active_tokens = block(active_tokens, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            active_tokens = active_tokens * active_valid.unsqueeze(-1).to(active_tokens.dtype)

        flat_out[active] = active_tokens
        return flat_out.view(batch_size, group_count, group_size, hidden_dim)

    def _pool_groups(self, local_states: torch.Tensor, grouped_valid: torch.Tensor) -> torch.Tensor:
        if self.merge_pool == "mean":
            denom = grouped_valid.sum(dim=2, keepdim=True).clamp_min(1).to(local_states.dtype)
            pooled = (local_states * grouped_valid.unsqueeze(-1).to(local_states.dtype)).sum(dim=2) / denom
            return pooled

        lengths = grouped_valid.sum(dim=2).clamp_min(1) - 1
        batch_idx = torch.arange(local_states.size(0), device=local_states.device).unsqueeze(1)
        group_idx = torch.arange(local_states.size(1), device=local_states.device).unsqueeze(0)
        pooled = local_states[batch_idx, group_idx, lengths]
        pooled = pooled * grouped_valid.any(dim=2, keepdim=True).to(local_states.dtype)
        return pooled

    def forward_features(
        self,
        input_ids: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
        patch_params: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        batch_size, seq_length = input_ids.size()

        item_embs = self.item_emb(input_ids)
        item_embs = item_embs * (self.hidden_units ** 0.5)

        seq_summary, _ = self._sequence_summary(item_embs, input_ids)
        if use_patch and self.patch_len > 0:
            if patch_params is None:
                patch_emb, gating_weights = self.meta_patch(seq_summary, user_ids=user_ids)
            else:
                patch_emb, gating_weights = self.meta_patch.forward_with_eta(seq_summary, patch_params, user_ids=user_ids)
        else:
            patch_emb = item_embs.new_zeros((batch_size, 0, self.hidden_units))
            gating_weights = None

        positions = torch.arange(1, seq_length + 1, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        prefix_len = int(getattr(self.config, "prefix_len", 0) or 0)
        use_prefix_tail_pos = bool(getattr(self.config, "prefix_tail_positions", False))
        if use_prefix_tail_pos and prefix_len > 0 and seq_length <= self.max_seq_length:
            prefix_len = min(prefix_len, seq_length)
            tail_len = max(seq_length - prefix_len, 0)
            pos_custom = torch.zeros_like(positions)
            if prefix_len > 0:
                pos_custom[:, :prefix_len] = torch.arange(1, prefix_len + 1, dtype=torch.long, device=input_ids.device)
            if tail_len > 0:
                tail_start = max(self.max_seq_length - tail_len + 1, 1)
                pos_custom[:, prefix_len:] = torch.arange(
                    tail_start,
                    tail_start + tail_len,
                    dtype=torch.long,
                    device=input_ids.device,
                )
            positions = pos_custom
        elif getattr(self.config, "right_align_positions", True):
            offset = max(self.max_seq_length - seq_length, 0)
            if offset > 0:
                positions = positions + offset
        positions = positions * (input_ids != 0).long()

        item_with_pos = item_embs + self.pos_emb(positions)
        item_valid = input_ids != 0

        if use_patch and self.patch_len > 0:
            patch_after_prefix = bool(getattr(self.config, "patch_after_prefix", False))
            if patch_after_prefix and prefix_len > 0:
                prefix_len = min(prefix_len, seq_length)
                hidden_states = torch.cat(
                    [item_with_pos[:, :prefix_len, :], patch_emb, item_with_pos[:, prefix_len:, :]],
                    dim=1,
                )
                valid_tokens = torch.cat(
                    [
                        item_valid[:, :prefix_len],
                        torch.ones((batch_size, self.patch_len), device=input_ids.device, dtype=torch.bool),
                        item_valid[:, prefix_len:],
                    ],
                    dim=1,
                )
            else:
                hidden_states = torch.cat([patch_emb, item_with_pos], dim=1)
                valid_tokens = torch.cat(
                    [
                        torch.ones((batch_size, self.patch_len), device=input_ids.device, dtype=torch.bool),
                        item_valid,
                    ],
                    dim=1,
                )
        else:
            hidden_states = item_with_pos
            valid_tokens = item_valid

        hidden_states = self.emb_dropout(hidden_states)

        grouped, grouped_valid, group_count, padded_len = self._build_grouped_inputs(hidden_states, valid_tokens)
        local_states = self._run_inner_trans(grouped, grouped_valid)
        merged = self._pool_groups(local_states, grouped_valid)
        group_valid = grouped_valid.any(dim=2)

        if self.global_token_count > 0:
            global_tokens = self.global_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            context = torch.cat([global_tokens, merged], dim=1)
            context_padding = torch.cat(
                [
                    torch.zeros((batch_size, self.global_token_count), device=input_ids.device, dtype=torch.bool),
                    ~group_valid,
                ],
                dim=1,
            )
            cross_mask = _causal_mask(group_count, self.global_token_count + group_count, allow_prefix=self.global_token_count, device=input_ids.device)
            merged = self.cross_block(
                merged,
                context,
                attn_mask=cross_mask,
                key_padding_mask=context_padding,
            )
            merged = merged * group_valid.unsqueeze(-1).to(merged.dtype)
            trunk = torch.cat([global_tokens, merged], dim=1)
            trunk_mask = _causal_mask(
                self.global_token_count + group_count,
                self.global_token_count + group_count,
                allow_prefix=self.global_token_count,
                device=input_ids.device,
            )
            for block in self.trunk_blocks:
                trunk = block(trunk, attn_mask=trunk_mask, key_padding_mask=context_padding)
            merged_context = trunk[:, self.global_token_count :, :]
        else:
            merged_context = merged
            if self.trunk_blocks:
                trunk_mask = _causal_mask(group_count, group_count, device=input_ids.device)
                key_padding_mask = ~group_valid
                for block in self.trunk_blocks:
                    merged_context = block(merged_context, attn_mask=trunk_mask, key_padding_mask=key_padding_mask)

        merged_context = merged_context * group_valid.unsqueeze(-1).to(merged_context.dtype)
        prev_group_context = torch.zeros_like(merged_context)
        if group_count > 1:
            prev_group_context[:, 1:, :] = merged_context[:, :-1, :]
        local_states = local_states + prev_group_context.unsqueeze(2)
        local_states = local_states * grouped_valid.unsqueeze(-1).to(local_states.dtype)

        output_tokens = local_states.reshape(batch_size, padded_len, self.hidden_units)[:, : hidden_states.size(1), :]
        output_tokens = self.ln_f(output_tokens)

        if return_gating:
            return output_tokens, gating_weights
        return output_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        neg_ids: Optional[torch.Tensor] = None,
        patch_params: Optional[torch.Tensor] = None,
        user_ids: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ):
        if pos_ids is not None and neg_ids is not None:
            return self.training_step(
                input_ids,
                pos_ids,
                neg_ids,
                patch_params=patch_params,
                user_ids=user_ids,
                return_gating=return_gating,
                use_patch=use_patch,
            )
        return self.forward_features(
            input_ids,
            user_ids=user_ids,
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
        user_ids: Optional[torch.Tensor] = None,
        use_patch: bool = True,
        use_head: bool = True,
    ) -> torch.Tensor:
        hidden_states = self.forward_features(
            input_ids,
            user_ids=user_ids,
            patch_params=patch_params,
            use_patch=use_patch,
        )
        if use_patch and self.patch_len > 0:
            hidden_states = self._strip_patch_tokens(hidden_states)
        final_hidden = hidden_states[:, -1, :]
        if use_head:
            final_hidden = self.apply_head(final_hidden, head_params=head_params)
        candidate_embs = self.item_emb(candidate_ids)
        return torch.bmm(candidate_embs, final_hidden.unsqueeze(-1)).squeeze(-1)

    def training_step(
        self,
        input_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        patch_params: Optional[torch.Tensor] = None,
        head_params: Optional[List[torch.Tensor]] = None,
        user_ids: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ):
        hidden_states, gating_weights = self.forward_features(
            input_ids,
            user_ids=user_ids,
            patch_params=patch_params,
            return_gating=True,
            use_patch=use_patch,
        )
        if use_patch and self.patch_len > 0:
            hidden_states = self._strip_patch_tokens(hidden_states)

        projected = self.apply_head(hidden_states, head_params=head_params)
        pos_embs = self.item_emb(pos_ids).to(projected.dtype)
        neg_embs = self.item_emb(neg_ids).to(projected.dtype)

        pos_logits = (projected * pos_embs).sum(dim=-1)
        if neg_embs.dim() == 4:
            neg_logits = (projected.unsqueeze(2) * neg_embs).sum(dim=-1)
        else:
            neg_logits = (projected * neg_embs).sum(dim=-1)

        if return_gating:
            return pos_logits, neg_logits, gating_weights
        return pos_logits, neg_logits
