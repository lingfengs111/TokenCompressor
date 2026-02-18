"""
ID-based sequence recommender models with encoder/decoder options.
Supports fixed item embedding tables and soft patch injection.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderOutput(NamedTuple):
    hidden: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None


class DecoderOutput(NamedTuple):
    hidden: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_rot = q * cos[None, None, :, :] + rotate_half(q) * sin[None, None, :, :]
    k_rot = k * cos[None, None, :, :] + rotate_half(k) * sin[None, None, :, :]
    return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional causal masking and RoPE."""

    def __init__(self, dim: int, num_heads: int = 8, attn_dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq_len, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :]
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out_proj(out)


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention from query sequence to encoder memory."""

    def __init__(self, dim: int, num_heads: int = 8, attn_dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, dim = query.shape
        mem_len = memory.size(1)
        q = self.q_proj(query).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(bsz, mem_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(bsz, mem_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if memory_mask is not None:
            mem_mask = memory_mask[:, None, None, :]
            scores = scores.masked_fill(~mem_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, dim)
        return self.out_proj(out)


class EncoderBlock(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask, is_causal=is_causal)
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    """Decoder block with causal self-attn and optional cross-attn."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.norm2 = RMSNorm(dim)
        self.cross_attn = MultiHeadCrossAttention(dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.norm3 = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), attention_mask=attention_mask, is_causal=True)
        if encoder_output is not None:
            x = x + self.cross_attn(self.norm2(x), encoder_output, memory_mask=encoder_mask)
        x = x + self.ffn(self.norm3(x))
        return x


class ItemEmbeddingTable(nn.Module):
    """Fixed item embedding table with optional padding row."""

    def __init__(self, num_items: int, embedding_dim: int, trainable: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        if not trainable:
            for p in self.parameters():
                p.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    @classmethod
    def from_pretrained(
        cls,
        embeddings: torch.Tensor,
        trainable: bool = False,
        pad_zero: bool = True,
    ) -> "ItemEmbeddingTable":
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError("embeddings must be a torch.Tensor")
        if embeddings.dim() != 2:
            raise ValueError("embeddings must be 2D [num_items, dim]")
        if pad_zero:
            pad = torch.zeros((1, embeddings.size(1)), dtype=embeddings.dtype)
            embeddings = torch.cat([pad, embeddings], dim=0)
        table = cls(num_items=embeddings.size(0) - 1, embedding_dim=embeddings.size(1), trainable=trainable)
        with torch.no_grad():
            table.embedding.weight.copy_(embeddings)
        return table


def load_item_embeddings(path: str) -> torch.Tensor:
    if path.endswith(".npy"):
        arr = np.load(path)
        return torch.from_numpy(arr).float()
    if path.endswith(".pt") or path.endswith(".pth"):
        tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Expected tensor in {path}, got {type(tensor)}")
        return tensor.float()
    raise ValueError(f"Unsupported embedding format: {path}")


class IDEncoder(nn.Module):
    """Encoder over item embeddings (non-causal)."""

    def __init__(
        self,
        item_embedding: ItemEmbeddingTable,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.item_embedding = item_embedding
        embed_dim = item_embedding.weight.size(1)
        self.input_proj = nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_emb: Optional[torch.Tensor] = None,
    ) -> EncoderOutput:
        x = self.item_embedding(input_ids)
        x = self.input_proj(x)
        x = self.dropout(x)
        attn = attention_mask
        if patch_emb is not None and patch_emb.numel() > 0:
            if patch_emb.dim() == 2:
                patch = patch_emb.unsqueeze(0).expand(x.size(0), -1, -1)
            elif patch_emb.dim() == 3:
                patch = patch_emb
            else:
                raise ValueError("patch_emb must be [L_soft, H] or [B, L_soft, H]")
            x = torch.cat([patch, x], dim=1)
            patch_mask = torch.ones((x.size(0), patch.size(1)), dtype=torch.bool, device=x.device)
            if attn is None:
                attn = torch.ones((x.size(0), x.size(1) - patch.size(1)), dtype=torch.bool, device=x.device)
            attn = torch.cat([patch_mask, attn], dim=1)
        if attn is None:
            attn = torch.ones((x.size(0), x.size(1)), dtype=torch.bool, device=x.device)
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attn, is_causal=False)
        return EncoderOutput(hidden=x, attention_mask=attn)


class IDDecoder(nn.Module):
    """Decoder over item embeddings (causal) with optional cross-attention."""

    def __init__(
        self,
        item_embedding: ItemEmbeddingTable,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.item_embedding = item_embedding
        embed_dim = item_embedding.weight.size(1)
        self.input_proj = nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        patch_emb: Optional[torch.Tensor] = None,
    ) -> DecoderOutput:
        x = self.item_embedding(input_ids)
        x = self.input_proj(x)
        x = self.dropout(x)
        attn = attention_mask
        if patch_emb is not None and patch_emb.numel() > 0:
            if patch_emb.dim() == 2:
                patch = patch_emb.unsqueeze(0).expand(x.size(0), -1, -1)
            elif patch_emb.dim() == 3:
                patch = patch_emb
            else:
                raise ValueError("patch_emb must be [L_soft, H] or [B, L_soft, H]")
            x = torch.cat([patch, x], dim=1)
            patch_mask = torch.ones((x.size(0), patch.size(1)), dtype=torch.bool, device=x.device)
            if attn is None:
                attn = torch.ones((x.size(0), x.size(1) - patch.size(1)), dtype=torch.bool, device=x.device)
            attn = torch.cat([patch_mask, attn], dim=1)
        if attn is None:
            attn = torch.ones((x.size(0), x.size(1)), dtype=torch.bool, device=x.device)
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attn, encoder_output=encoder_output, encoder_mask=encoder_mask)
        return DecoderOutput(hidden=x, attention_mask=attn)


class IDRecModel(nn.Module):
    """ID-based model with configurable encoder/decoder usage."""

    def __init__(
        self,
        item_table: ItemEmbeddingTable,
        hidden_dim: int = 512,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        mode: str = "encoder_decoder",
    ):
        super().__init__()
        self.item_table = item_table
        self.mode = mode

        self.encoder: Optional[IDEncoder] = None
        self.decoder: Optional[IDDecoder] = None

        if mode == "encoder_decoder":
            self.encoder = IDEncoder(
                item_embedding=item_table,
                hidden_dim=hidden_dim,
                num_layers=encoder_layers,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
        if mode in ("decoder_only", "encoder_decoder"):
            self.decoder = IDDecoder(
                item_embedding=item_table,
                hidden_dim=hidden_dim,
                num_layers=decoder_layers,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "decoder_only":
            if self.decoder is None:
                raise ValueError("decoder is not initialized")
            dec_out = self.decoder(input_ids, attention_mask=attention_mask, patch_emb=patch_emb)
            return dec_out.hidden, dec_out.attention_mask

        if self.mode == "encoder_decoder":
            if self.encoder is None or self.decoder is None:
                raise ValueError("encoder/decoder are not initialized")
            enc_out = self.encoder(input_ids, attention_mask=attention_mask, patch_emb=patch_emb)
            dec_out = self.decoder(
                input_ids,
                attention_mask=attention_mask,
                encoder_output=enc_out.hidden,
                encoder_mask=enc_out.attention_mask,
            )
            return dec_out.hidden, dec_out.attention_mask

        raise ValueError(f"Unknown mode: {self.mode}")
