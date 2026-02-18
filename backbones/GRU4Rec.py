"""GRU4Rec backbone adapted for soft-patch training."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.patch import MetaPatch
from backbones.modules import apply_head as apply_head_fn, head_parameters as head_parameters_fn, sequence_summary
from backbones.peft import Adapter


class GRU4Rec(nn.Module):
    """GRU4Rec-style encoder with SASRec-compatible interface."""

    def __init__(self, config, item_num: int):
        super().__init__()
        self.config = config
        self.item_num = item_num
        self.hidden_units = config.hidden_units
        self.patch_len = config.patch_len

        embed_size = getattr(config, "gru_embedding_size", self.hidden_units)
        num_layers = getattr(config, "gru_num_layers", 1)
        dropout = getattr(config, "gru_dropout", getattr(config, "dropout_rate", 0.2))

        self.item_emb = nn.Embedding(item_num + 2, embed_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=self.hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bias=False,
        )
        self.dense = nn.Linear(self.hidden_units, self.hidden_units)
        self.ln_f = nn.LayerNorm(self.hidden_units, eps=1e-8)

        # Optional adapter (PEFT baseline)
        self.adapter = None
        if getattr(config, "enable_adapter", False):
            bottleneck = int(getattr(config, "adapter_dim", max(4, self.hidden_units // 8)))
            dropout_p = float(getattr(config, "adapter_dropout", 0.0))
            activation = getattr(config, "adapter_activation", "gelu")
            init = getattr(config, "adapter_init", "zero")
            self.adapter = Adapter(self.hidden_units, bottleneck, dropout=dropout_p, activation=activation, init=init)

        # Trainable projection head (only head is adapted in inner loop)
        self.proj_linear = nn.Linear(self.hidden_units, self.hidden_units, bias=True)
        self.proj_ln = nn.LayerNorm(self.hidden_units, eps=1e-8)

        # Meta-patch module (outer loop)
        self.meta_patch = MetaPatch(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        init_range = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Embedding):
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

    def forward_features(
        self,
        input_ids: torch.Tensor,
        patch_params: Optional[torch.Tensor] = None,
        return_gating: bool = False,
        use_patch: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        batch_size, seq_length = input_ids.size()

        item_embs = self.item_emb(input_ids)
        item_embs = self.emb_dropout(item_embs)

        seq_summary, _ = self._sequence_summary(item_embs, input_ids)
        if use_patch and self.patch_len > 0:
            if patch_params is None:
                patch_emb, gating_weights = self.meta_patch(seq_summary)
            else:
                patch_emb, gating_weights = self.meta_patch.forward_with_eta(seq_summary, patch_params)
            hidden_states = torch.cat([patch_emb, item_embs], dim=1)
        else:
            gating_weights = None
            hidden_states = item_embs

        gru_out, _ = self.gru(hidden_states)
        hidden_states = self.dense(gru_out)
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)
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
