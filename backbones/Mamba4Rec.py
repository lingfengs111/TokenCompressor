"""Mamba4Rec backbone adapted for soft-patch training."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from backbones.modules import apply_head as apply_head_fn, head_parameters as head_parameters_fn, sequence_summary
from backbones.patch import MetaPatch

try:
    from mamba_ssm import Mamba as _MambaImpl
except Exception as exc:  # pragma: no cover - exercised only when dependency is missing
    _MambaImpl = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


class FeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout: float) -> None:
        super().__init__()
        self.w_1 = nn.Linear(hidden_units, hidden_units * 4)
        self.w_2 = nn.Linear(hidden_units * 4, hidden_units)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-12)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.layer_norm(hidden_states + input_tensor)


class MambaLayer(nn.Module):
    def __init__(
        self,
        hidden_units: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        use_residual: bool,
    ) -> None:
        super().__init__()
        if _MambaImpl is None:
            raise ImportError(
                "mamba_ssm is required to instantiate backbone='mamba4rec'. "
                "Install it with `pip install mamba-ssm causal-conv1d`."
            ) from _MAMBA_IMPORT_ERROR
        self.mamba = _MambaImpl(
            d_model=hidden_units,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-12)
        self.feed_forward = FeedForward(hidden_units, dropout=dropout)
        self.use_residual = use_residual

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.mamba(input_tensor)
        hidden_states = self.dropout(hidden_states)
        if self.use_residual:
            hidden_states = self.layer_norm(hidden_states + input_tensor)
        else:
            hidden_states = self.layer_norm(hidden_states)
        return self.feed_forward(hidden_states)


class Mamba4Rec(nn.Module):
    """Mamba4Rec-style encoder with SASRec-compatible interface."""

    def __init__(self, config, item_num: int):
        super().__init__()
        if _MambaImpl is None:
            raise ImportError(
                "mamba_ssm is required to instantiate backbone='mamba4rec'. "
                "Install it with `pip install mamba-ssm causal-conv1d`."
            ) from _MAMBA_IMPORT_ERROR

        self.config = config
        self.item_num = item_num
        self.hidden_units = int(config.hidden_units)
        self.patch_len = int(getattr(config, "patch_len", 0) or 0)
        self.max_seq_length = int(config.max_seq_length or 0)
        self.total_seq_length = self.max_seq_length + self.patch_len

        num_layers = int(getattr(config, "mamba_num_layers", getattr(config, "num_blocks", 2)) or 2)
        d_state = int(getattr(config, "mamba_d_state", 32) or 32)
        d_conv = int(getattr(config, "mamba_d_conv", 4) or 4)
        expand = int(getattr(config, "mamba_expand", 2) or 2)
        dropout = float(getattr(config, "mamba_dropout", getattr(config, "dropout_rate", 0.2)) or 0.2)

        self.item_emb = nn.Embedding(item_num + 2, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max(self.total_seq_length, 1) + 1, self.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_ln = nn.LayerNorm(self.hidden_units, eps=1e-12)

        self.mamba_layers = nn.ModuleList(
            [
                MambaLayer(
                    self.hidden_units,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    use_residual=(num_layers > 1),
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.hidden_units, eps=1e-8)

        self.proj_linear = nn.Linear(self.hidden_units, self.hidden_units, bias=True)
        self.proj_ln = nn.LayerNorm(self.hidden_units, eps=1e-8)
        self.meta_patch = MetaPatch(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        init_range = float(getattr(self.config, "initializer_range", 0.02) or 0.02)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=init_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _freeze_backbone(self) -> None:
        for p in self.item_emb.parameters():
            p.requires_grad = False
        for p in self.pos_emb.parameters():
            p.requires_grad = False
        for p in self.emb_ln.parameters():
            p.requires_grad = False
        for layer in self.mamba_layers:
            for p in layer.parameters():
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

    def _build_position_ids(self, input_ids: torch.Tensor, patch_tokens: int) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        device = input_ids.device
        valid_mask = input_ids != 0
        positions = valid_mask.cumsum(dim=1)
        if bool(getattr(self.config, "right_align_positions", True)):
            lengths = valid_mask.sum(dim=1, keepdim=True)
            shift = seq_length - lengths
            positions = torch.where(valid_mask, positions + shift, torch.zeros_like(positions))
        if patch_tokens > 0:
            patch_positions = torch.arange(1, patch_tokens + 1, device=device, dtype=torch.long).unsqueeze(0).expand(
                batch_size, -1
            )
            positions = torch.where(valid_mask, positions + patch_tokens, torch.zeros_like(positions))
            positions = torch.cat([patch_positions, positions], dim=1)
        return positions

    def _strip_patch_tokens(self, hidden_states: torch.Tensor, seq_length: Optional[int] = None) -> torch.Tensor:
        if self.patch_len <= 0:
            return hidden_states
        if seq_length is not None and hidden_states.size(1) == seq_length:
            return hidden_states
        return hidden_states[:, self.patch_len :, :]

    def strip_patch_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._strip_patch_tokens(hidden_states)

    def forward_features(
        self,
        input_ids: torch.Tensor,
        patch_params: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        item_embs = self.item_emb(input_ids)
        item_embs = item_embs * (self.hidden_units**0.5)

        seq_summary, _ = self._sequence_summary(item_embs, input_ids)
        if use_patch and self.patch_len > 0:
            if patch_params is None:
                patch_emb, gating_weights = self.meta_patch(seq_summary)
            else:
                patch_emb, gating_weights = self.meta_patch.forward_with_eta(seq_summary, patch_params)
            hidden_states = torch.cat([patch_emb, item_embs], dim=1)
            patch_tokens = self.patch_len
        else:
            gating_weights = None
            hidden_states = item_embs
            patch_tokens = 0

        pos_ids = self._build_position_ids(input_ids, patch_tokens=patch_tokens)
        hidden_states = hidden_states + self.pos_emb(pos_ids)
        hidden_states = self.emb_dropout(hidden_states)
        hidden_states = self.emb_ln(hidden_states)

        for layer in self.mamba_layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.ln_f(hidden_states)

        if return_gating:
            return hidden_states, gating_weights
        return hidden_states

    def get_gating_weights(self, input_ids: torch.Tensor, patch_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        item_embs = self.item_emb(input_ids)
        item_embs = item_embs * (self.hidden_units**0.5)
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
        hidden_states = self._strip_patch_tokens(hidden_states, input_ids.size(1))
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
        return_gating: bool = False,
        use_patch: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        hidden_states, gating_weights = self.forward_features(
            input_ids,
            patch_params=patch_params,
            return_gating=True,
            use_patch=use_patch,
        )
        hidden_states = self._strip_patch_tokens(hidden_states, input_ids.size(1))

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
