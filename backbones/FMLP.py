"""FMLP backbone adapted for soft-patch training."""

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


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    """LayerNorm with epsilon inside sqrt (TF-style)."""

    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_dropout: float, hidden_dropout: float):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be a multiple of num_attention_heads")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FilterLayer(nn.Module):
    def __init__(self, hidden_size: int, max_seq_length: int, hidden_dropout: float):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.complex_weight = nn.Parameter(
            torch.randn(1, max_seq_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02
        )
        self.out_dropout = nn.Dropout(hidden_dropout)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)

    def _match_weight(self, weight: torch.Tensor, target_len: int) -> torch.Tensor:
        if weight.size(1) == target_len:
            return weight
        if weight.size(1) > target_len:
            return weight[:, :target_len, :]
        pad_len = target_len - weight.size(1)
        pad = weight.new_zeros((weight.size(0), pad_len, weight.size(2)))
        return torch.cat([weight, pad], dim=1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        weight = self._match_weight(weight, x.size(1))
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm="ortho")
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, hidden_size: int, hidden_act: str, hidden_dropout: float):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, hidden_size * 4)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

        self.dense_2 = nn.Linear(4 * hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FMLPLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, hidden_dropout: float, attn_dropout: float, hidden_act: str, max_seq_length: int, no_filters: bool):
        super().__init__()
        self.no_filters = no_filters
        if self.no_filters:
            self.attention = SelfAttention(hidden_size, num_heads, attn_dropout, hidden_dropout)
        else:
            self.filterlayer = FilterLayer(hidden_size, max_seq_length=max_seq_length, hidden_dropout=hidden_dropout)
        self.intermediate = Intermediate(hidden_size, hidden_act, hidden_dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.no_filters:
            hidden_states = self.attention(hidden_states, attention_mask)
        else:
            hidden_states = self.filterlayer(hidden_states)
        hidden_states = self.intermediate(hidden_states)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, layer: FMLPLayer, num_layers: int):
        super().__init__()
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_all_encoded_layers: bool = True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class FMLP(nn.Module):
    """Filter-enhanced MLP backbone with SASRec-compatible interface."""

    def __init__(self, config, item_num: int):
        super().__init__()
        self.config = config
        self.item_num = item_num
        self.hidden_units = config.hidden_units
        self.patch_len = config.patch_len
        self.max_seq_length = config.max_seq_length
        self.total_seq_length = self.max_seq_length + self.patch_len

        num_layers = getattr(config, "fmlp_num_layers", getattr(config, "num_blocks", 2))
        num_heads = getattr(config, "fmlp_num_heads", getattr(config, "num_heads", 1))
        hidden_dropout = getattr(config, "fmlp_hidden_dropout", getattr(config, "dropout_rate", 0.2))
        attn_dropout = getattr(config, "fmlp_attn_dropout", getattr(config, "dropout_rate", 0.2))
        hidden_act = getattr(config, "fmlp_hidden_act", "gelu")
        no_filters = getattr(config, "fmlp_no_filters", False)

        # Embeddings
        self.item_emb = nn.Embedding(item_num + 2, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.total_seq_length + 1, self.hidden_units, padding_idx=0)
        self.layer_norm = LayerNorm(self.hidden_units, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout)

        layer = FMLPLayer(
            hidden_size=self.hidden_units,
            num_heads=num_heads,
            hidden_dropout=hidden_dropout,
            attn_dropout=attn_dropout,
            hidden_act=hidden_act,
            max_seq_length=self.total_seq_length,
            no_filters=no_filters,
        )
        self.item_encoder = Encoder(layer, num_layers=num_layers)

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
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
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
        item_encoded_layers = self.item_encoder(hidden_states, attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]

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
