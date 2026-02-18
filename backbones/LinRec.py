"""LinRec backbone (linear attention) adapted for soft-patch training."""

import copy
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.patch import MetaPatch
from backbones.modules import apply_head as apply_head_fn, head_parameters as head_parameters_fn, sequence_summary


def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": gelu,
    "relu": F.relu,
    "swish": swish,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}


class LinRecAttention(nn.Module):
    """Elu-Norm linear attention from LinRec layers.py."""

    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be a multiple of n_heads")

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.elu = nn.ELU()

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # attention_mask is kept for interface compatibility; linear attention ignores it.
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        elu_query = self.elu(query_layer)
        elu_key = self.elu(key_layer)
        query_norm_inverse = 1.0 / torch.norm(elu_query, dim=3, p=2).clamp_min(1e-8)
        key_norm_inverse = 1.0 / torch.norm(elu_key, dim=2, p=2).clamp_min(1e-8)
        normalized_query_layer = torch.einsum("mnij,mni->mnij", elu_query, query_norm_inverse)
        normalized_key_layer = torch.einsum("mnij,mnj->mnij", elu_key, key_norm_inverse)
        context_layer = torch.matmul(
            normalized_query_layer,
            torch.matmul(normalized_key_layer, value_layer),
        ) / self.sqrt_attention_head_size

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = ACT2FN[hidden_act]
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LinRecLayer(nn.Module):
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.multi_head_attention = LinRecAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class LinRecEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        layer = LinRecLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_all_encoded_layers: bool = True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class LinRec(nn.Module):
    """LinRec backbone with SASRec-compatible interface."""

    def __init__(self, config, item_num: int):
        super().__init__()
        self.config = config
        self.item_num = item_num
        self.hidden_units = config.hidden_units
        self.patch_len = config.patch_len
        self.max_seq_length = config.max_seq_length

        n_layers = getattr(config, "linrec_num_layers", getattr(config, "num_blocks", 2))
        n_heads = getattr(config, "linrec_num_heads", getattr(config, "num_heads", 1))
        inner_size = getattr(config, "linrec_inner_size", self.hidden_units * 4)
        hidden_dropout = getattr(config, "linrec_hidden_dropout", getattr(config, "dropout_rate", 0.2))
        attn_dropout = getattr(config, "linrec_attn_dropout", getattr(config, "dropout_rate", 0.2))
        hidden_act = getattr(config, "linrec_hidden_act", "gelu")
        layer_norm_eps = getattr(config, "linrec_layer_norm_eps", 1e-12)

        self.item_emb = nn.Embedding(item_num + 2, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_seq_length + 1, self.hidden_units, padding_idx=0)
        self.layer_norm = nn.LayerNorm(self.hidden_units, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)

        self.encoder = LinRecEncoder(
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=self.hidden_units,
            inner_size=inner_size,
            hidden_dropout_prob=hidden_dropout,
            attn_dropout_prob=attn_dropout,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
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
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape, device=input_ids.device), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1).long()
        extended_attention_mask = extended_attention_mask * subsequent_mask
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

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        attention_mask = self._build_attention_mask(input_ids, self.patch_len if use_patch else 0)
        item_encoded_layers = self.encoder(hidden_states, attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]
        sequence_output = self.ln_f(sequence_output)

        if return_gating:
            return sequence_output, gating_weights
        return sequence_output

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
