"""
Encoder-Decoder model for predicting next semantic codes.
Architecture based on GR_Pipeline's EncoderDecoderRetrievalModel pattern.

Components:
  - Encoder: Processes input semantic code sequences → contextualized representations
  - Decoder: Takes encoder context + generates logits for next codes

Takes sequences of semantic codes and predicts the next code(s).
"""

from typing import Optional, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderOutput(NamedTuple):
    """Output from encoder."""
    encoded: torch.Tensor  # [batch, seq_len, hidden_dim]
    attention_mask: Optional[torch.Tensor] = None  # [batch, seq_len]


class DecoderOutput(NamedTuple):
    """Output from decoder."""
    logits: torch.Tensor  # [batch, num_levels, codebook_size]
    hidden: torch.Tensor  # [batch, hidden_dim]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) for better position encoding."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Pre-compute frequency matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos and sin matrices for RoPE.
        
        Returns:
            Tuple of (cos_cached, sin_cached) with shape [seq_len, dim]
        """
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_cached = emb.cos()
        sin_cached = emb.sin()
        
        return cos_cached, sin_cached


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key."""
    # q, k: [batch, heads, seq_len, head_dim]
    # cos, sin: [seq_len, dim]
    
    def rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_rot = q * cos[None, None, :, :] + rotate_half(q) * sin[None, None, :, :]
    k_rot = k * cos[None, None, :, :] + rotate_half(k) * sin[None, None, :, :]
    
    return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE."""
    
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
        """
        Args:
            x: [batch, seq_len, dim]
            attention_mask: [batch, seq_len] bool tensor (True = attend, False = mask)
            is_causal: Whether to apply causal mask
        
        Returns:
            output: [batch, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # Project to q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [batch, num_heads, seq_len, head_dim]
        
        # Apply RoPE
        cos, sin = self.rope(seq_len, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # scores: [batch, num_heads, seq_len, seq_len]
        
        # Apply causal mask
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))
        
        # Apply attention mask
        if attention_mask is not None:
            # Convert [batch, seq_len] to [batch, 1, 1, seq_len]
            attention_mask = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: [batch, num_heads, seq_len, head_dim]
        
        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        return output


class EncoderBlock(nn.Module):
    """Single transformer block with attention and FFN."""
    
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
        """Pre-norm residual connections."""
        # Attention with pre-norm
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask, is_causal=is_causal)
        
        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        
        return x


class Encoder(nn.Module):
    """
    Encoder that processes semantic code sequences.
    Transforms input codes into contextualized representations.
    """
    
    def __init__(
        self,
        code_embeddings: nn.ModuleList,
        num_levels: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        
        # Shared code embeddings across encoder/decoder
        self.code_embeddings = code_embeddings
        self.embed_dim = self.code_embeddings[0].embedding_dim
        
        # Project concatenated embeddings to hidden_dim
        self.embed_proj = nn.Linear(self.embed_dim * num_levels, hidden_dim)
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            EncoderBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_emb: Optional[torch.Tensor] = None,
    ) -> EncoderOutput:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len, num_levels] semantic codes
                      or [batch, seq_len*num_levels] flattened codes
            attention_mask: [batch, seq_len] or [batch, seq_len*num_levels] bool mask
        
        Returns:
            EncoderOutput with encoded representations
        """
        # Handle flattened codes: [batch, seq_len*num_levels] -> [batch, seq_len, num_levels]
        if input_ids.dim() == 2:
            batch_size, flattened_len = input_ids.shape
            if flattened_len % self.num_levels != 0:
                raise ValueError(
                    f"Flattened input length {flattened_len} is not divisible by num_levels={self.num_levels}"
                )
            seq_len = flattened_len // self.num_levels
            input_ids = input_ids.view(batch_size, seq_len, self.num_levels)
            
            # Also reshape attention_mask if provided
            if attention_mask is not None and attention_mask.dim() == 2 and attention_mask.shape[1] == flattened_len:
                attention_mask = attention_mask.view(batch_size, seq_len * self.num_levels)
                # Average pool mask across num_levels to get seq_len mask
                attention_mask = attention_mask.view(batch_size, seq_len, self.num_levels).any(dim=-1)
        
        batch_size, seq_len, num_levels = input_ids.shape
        assert num_levels == self.num_levels, f"Expected {self.num_levels} levels, got {num_levels}"
        
        # Embed each code level separately and concatenate
        embeddings = []
        for level in range(num_levels):
            codes_at_level = input_ids[:, :, level]  # [batch, seq_len]
            emb = self.code_embeddings[level](codes_at_level)  # [batch, seq_len, hidden_dim]
            embeddings.append(emb)
        
        # Concatenate embeddings from all levels
        x = torch.cat(embeddings, dim=-1)  # [batch, seq_len, hidden_dim]
        x = self.embed_proj(x)
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
            dtype = attn.dtype if attn is not None else torch.bool
            patch_mask = torch.ones((x.size(0), patch.size(1)), dtype=dtype, device=x.device)
            if attn is None:
                attn = torch.ones((x.size(0), x.size(1) - patch.size(1)), dtype=dtype, device=x.device)
            attn = torch.cat([patch_mask, attn], dim=1)

        # Apply transformer blocks (non-causal, encoder can see all positions)
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attn, is_causal=False)
        
        return EncoderOutput(encoded=x, attention_mask=attn)


class CausalEncoder(nn.Module):
    """
    Causal variant of the encoder that only attends to previous positions.
    Useful for decoder-only style models that rely on autoregressive history.
    """
    
    def __init__(
        self,
        code_embeddings: nn.ModuleList,
        num_levels: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.code_embeddings = code_embeddings
        self.embed_dim = self.code_embeddings[0].embedding_dim
        
        self.embed_proj = nn.Linear(self.embed_dim * num_levels, hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            EncoderBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_emb: Optional[torch.Tensor] = None,
    ) -> EncoderOutput:
        # Handle flattened codes: [batch, seq_len*num_levels] -> [batch, seq_len, num_levels]
        if input_ids.dim() == 2:
            batch_size, flattened_len = input_ids.shape
            if flattened_len % self.num_levels != 0:
                raise ValueError(
                    f"Flattened input length {flattened_len} is not divisible by num_levels={self.num_levels}"
                )
            seq_len = flattened_len // self.num_levels
            input_ids = input_ids.view(batch_size, seq_len, self.num_levels)
            
            if attention_mask is not None and attention_mask.dim() == 2 and attention_mask.shape[1] == flattened_len:
                attention_mask = attention_mask.view(batch_size, seq_len * self.num_levels)
                attention_mask = attention_mask.view(batch_size, seq_len, self.num_levels).any(dim=-1)
        
        batch_size, seq_len, num_levels = input_ids.shape
        assert num_levels == self.num_levels, f"Expected {self.num_levels} levels, got {num_levels}"
        
        embeddings = []
        for level in range(num_levels):
            codes_at_level = input_ids[:, :, level]
            emb = self.code_embeddings[level](codes_at_level)
            embeddings.append(emb)
        
        x = torch.cat(embeddings, dim=-1)
        x = self.embed_proj(x)
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
            dtype = attn.dtype if attn is not None else torch.bool
            patch_mask = torch.ones((x.size(0), patch.size(1)), dtype=dtype, device=x.device)
            if attn is None:
                attn = torch.ones((x.size(0), x.size(1) - patch.size(1)), dtype=dtype, device=x.device)
            attn = torch.cat([patch_mask, attn], dim=1)

        # Causal transformer blocks (decoder-only style)
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attn, is_causal=True)
        
        return EncoderOutput(encoded=x, attention_mask=attn)


class CrossAttention(nn.Module):
    """Cross-attention: query from decoder, key/value from encoder."""
    
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
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, hidden_dim] - decoder query (single token)
            encoder_output: [batch, seq_len, hidden_dim] - encoder output
            encoder_mask: [batch, seq_len] - encoder attention mask
        
        Returns:
            [batch, hidden_dim]
        """
        batch_size, hidden_dim = query.shape
        seq_len = encoder_output.shape[1]
        
        # Project query
        q = self.q_proj(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        # q: [batch, num_heads, 1, head_dim]
        
        # Project key/value from encoder
        k = self.k_proj(encoder_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # k, v: [batch, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # scores: [batch, num_heads, 1, seq_len]
        
        # Apply encoder mask
        if encoder_mask is not None:
            encoder_mask = encoder_mask[:, None, None, :]  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(~encoder_mask, float("-inf"))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: [batch, num_heads, 1, head_dim]
        
        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, hidden_dim)
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        return output


class DecoderBlock(nn.Module):
    """Decoder block with cross-attention to encoder and FFN."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        
        # Cross-attention to encoder
        self.norm1 = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, attn_dropout=attn_dropout)
        
        # FFN
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
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            x: [batch, hidden_dim] - decoder input (single token per level)
            encoder_output: [batch, seq_len, hidden_dim] - encoder output
            encoder_mask: [batch, seq_len] - attention mask for encoder
        
        Returns:
            [batch, hidden_dim]
        """
        # Cross-attention to encoder with pre-norm
        x = x + self.cross_attn(self.norm1(x), encoder_output=encoder_output, encoder_mask=encoder_mask)
        
        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        
        return x


class Decoder(nn.Module):
    """
    Decoder that generates next semantic codes level by level given encoder context.
    Uses cross-attention to attend to encoder outputs.
    Each level shares the same decoder blocks but generates independently.
    """
    
    def __init__(
        self,
        code_embeddings: nn.ModuleList,
        codebook_size: int = 256,  # vocab size including PAD
        num_levels: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        carry_decoder_state: bool = False,
    ):
        super().__init__()
        
        self.codebook_size = codebook_size  # vocab size including PAD
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.carry_decoder_state = carry_decoder_state
        
        # Shared code embeddings for all levels
        self.code_embeddings = code_embeddings
        
        # Shared decoder blocks for all levels
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection for each level (separate)
        self.output_projs = nn.ModuleList([
            nn.Linear(hidden_dim, codebook_size)
            for _ in range(num_levels)
        ])

    def _next_state(self, x_level: torch.Tensor, code_emb: torch.Tensor) -> torch.Tensor:
        if self.carry_decoder_state:
            return x_level + code_emb
        return code_emb
    
    def forward(
        self,
        context: torch.Tensor,
        encoder_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        max_level: Optional[int] = None,
    ) -> DecoderOutput:
        """
        Forward pass for training (teacher forcing).
        
        Args:
            context: [batch, hidden_dim] - pooled encoder output (optional, can use zeros)
            encoder_output: [batch, seq_len, hidden_dim] - full encoder output for cross-attention
            attention_mask: [batch, seq_len] - encoder mask for cross-attention
            decoder_input_ids: [batch, num_levels] - previous level codes for teacher forcing (optional)
            max_level: if set, only decode levels < max_level (used for incremental decoding/beam search)
        
        Returns:
            DecoderOutput with logits [batch, num_levels, codebook_size]
        """
        batch_size = encoder_output.shape[0]
        
        # Start with context
        x = context  # [batch, hidden_dim]
        
        # Collect logits for all levels
        all_logits = []
        
        # Generate each level sequentially
        num_levels_to_decode = self.num_levels if max_level is None else min(self.num_levels, max_level)
        for level in range(num_levels_to_decode):
            # If we have previous codes (teacher forcing), embed them as context
            if decoder_input_ids is not None and level > 0 and level - 1 < decoder_input_ids.size(1):
                prev_code = decoder_input_ids[:, level - 1]  # [batch]
                prev_emb = self.code_embeddings[level - 1](prev_code)  # [batch, hidden_dim]
                x = x + prev_emb  # Incorporate previous level's code
            
            # Apply decoder blocks with cross-attention to encoder
            x_level = x  # [batch, hidden_dim]
            for block in self.decoder_blocks:
                x_level = block(x_level, encoder_output=encoder_output, encoder_mask=attention_mask)
            
            # Project to logits for this level
            level_logits = self.output_projs[level](x_level)  # [batch, codebook_size]
            # Never predict PAD token
            level_logits[:, 0] = -1e9
            all_logits.append(level_logits)
            
            # Update x with embedding of predicted (or teacher-forced) code for next level
            if decoder_input_ids is not None and level < decoder_input_ids.size(1):
                code = decoder_input_ids[:, level]  # [batch]
                code_emb = self.code_embeddings[level](code)
                x = self._next_state(x_level, code_emb)
            else:
                code = torch.argmax(level_logits, dim=-1)  # [batch]
                code_emb = self.code_embeddings[level](code)
                x = self._next_state(x_level, code_emb)
        
        # Stack logits: [batch, num_levels, codebook_size]
        logits = torch.stack(all_logits, dim=1)
        
        return DecoderOutput(logits=logits, hidden=x)


class SemanticCodePredictor(nn.Module):
    """
    Complete Encoder-Decoder model for semantic code prediction.
    
    Architecture:
    - Encoder: Processes input code sequences → contextualized representations
    - Decoder: Takes decoder input (BOS) + encoder context → predicts next codes
    """
    
    def __init__(
        self,
        codebook_size: int = 256,  # number of real codes per level (excludes PAD)
        num_levels: int = 3,
        hidden_dim: int = 512,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        max_seq_len: int = 512,
        carry_decoder_state: bool = False,
    ):
        super().__init__()
        
        self.codebook_size = codebook_size  # real codes
        self.vocab_size = codebook_size + 1  # +1 for PAD=0
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.carry_decoder_state = carry_decoder_state
        
        # Shared code embeddings (one per level)
        self.code_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, hidden_dim, padding_idx=0)
            for _ in range(num_levels)
        ])
        
        # Encoder
        self.encoder = Encoder(
            code_embeddings=self.code_embeddings,
            num_levels=num_levels,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        
        # Decoder
        self.decoder = Decoder(
            code_embeddings=self.code_embeddings,
            codebook_size=self.vocab_size,
            num_levels=num_levels,
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        
        # BOS token embedding (beginning of sequence for decoder)
        self.bos_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        max_level: Optional[int] = None,
        patch_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training and generation.
        """
        encoder_output = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            patch_emb=patch_emb,
        )
        encoded = encoder_output.encoded  # [batch, seq_len, hidden_dim]
        enc_mask = encoder_output.attention_mask
        
        if enc_mask is not None:
            last_valid_pos = enc_mask.long().sum(dim=1) - 1
            batch_size = encoded.shape[0]
            context = encoded[torch.arange(batch_size), last_valid_pos, :]
        else:
            context = encoded[:, -1, :]  # [batch, hidden_dim]
        
        context = context + self.bos_embedding
        
        decoder_output = self.decoder(
            context=context,
            encoder_output=encoded,
            attention_mask=enc_mask,
            decoder_input_ids=decoder_input_ids,
            max_level=max_level,
        )
        
        return decoder_output.logits, decoder_output.hidden
    
    def get_next_code_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits, _ = self.forward(input_ids, attention_mask=attention_mask, patch_emb=patch_emb)
        return F.softmax(logits, dim=-1)
    
    def sample_next_code(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        patch_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits, _ = self.forward(input_ids, attention_mask=attention_mask, patch_emb=patch_emb)
        logits = logits / temperature
        
        probs = F.softmax(logits, dim=-1)
        batch_size = probs.shape[0]
        probs_flat = probs.reshape(batch_size * self.num_levels, self.vocab_size)
        next_codes_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        next_codes = next_codes_flat.view(batch_size, self.num_levels)
        
        return next_codes
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        return_scores: bool = False,
        patch_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if num_beams == 1:
            logits, _ = self.forward(input_ids, attention_mask=attention_mask, patch_emb=patch_emb)
            next_codes = torch.argmax(logits, dim=-1)  # [batch, num_levels]
            return next_codes.unsqueeze(1)  # [batch, 1, num_levels]
        else:
            return self.beam_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                return_scores=return_scores,
                patch_emb=patch_emb,
            )
    
    def beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 5,
        num_return_sequences: int = 1,
        return_scores: bool = False,
        patch_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Safety: cannot return more sequences than beams
        num_return_sequences = min(num_return_sequences, num_beams)

        batch_size = input_ids.shape[0]
        device = input_ids.device

        with torch.no_grad():
            encoder_output = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                patch_emb=patch_emb,
            )
            encoded = encoder_output.encoded
            enc_mask = encoder_output.attention_mask if encoder_output.attention_mask is not None else attention_mask

            if enc_mask is not None:
                last_valid_pos = enc_mask.long().sum(dim=1) - 1
                context = encoded[torch.arange(batch_size, device=device), last_valid_pos, :]
            else:
                context = encoded[:, -1, :]

            context = context + self.bos_embedding

            # Expand encoder outputs and masks for beams
            encoded = encoded.repeat_interleave(num_beams, dim=0)
            enc_mask = enc_mask.repeat_interleave(num_beams, dim=0) if enc_mask is not None else None
            beam_hidden = context.repeat_interleave(num_beams, dim=0)

        beam_codes = torch.zeros(
            (batch_size * num_beams, self.num_levels),
            dtype=torch.long,
            device=device,
        )
        beam_scores = torch.zeros((batch_size, num_beams), device=device)
        beam_scores[:, 1:] = float("-inf")
        beam_scores = beam_scores.view(-1)

        beam_offset = (torch.arange(batch_size, device=device) * num_beams).unsqueeze(1)

        for level in range(self.num_levels):
            with torch.no_grad():
                decoder_out = self.decoder(
                    context=beam_hidden,
                    encoder_output=encoded,
                    attention_mask=enc_mask,
                    decoder_input_ids=beam_codes if level > 0 else None,
                    max_level=level + 1,
                )

            level_logits = decoder_out.logits[:, level, :]
            level_scores = torch.log_softmax(level_logits, dim=-1)
            level_scores = level_scores + beam_scores[:, None]
            level_scores = level_scores.view(batch_size, num_beams * self.vocab_size)

            top_scores, top_indices = torch.topk(
                level_scores,
                min(2 * num_beams, num_beams * self.vocab_size),
                dim=1,
                largest=True,
                sorted=True,
            )

            beam_indices = torch.div(top_indices, self.vocab_size, rounding_mode="floor")
            level_codes = top_indices % self.vocab_size

            top_scores = top_scores[:, :num_beams]
            beam_indices = beam_indices[:, :num_beams]
            level_codes = level_codes[:, :num_beams]

            beam_scores = top_scores.reshape(-1)

            gather_index = (beam_indices + beam_offset).reshape(-1)
            old_beam_codes = beam_codes[gather_index, :]
            new_level_codes = level_codes.reshape(-1, 1)
            beam_codes = torch.cat([old_beam_codes[:, :level], new_level_codes], dim=1)

            # Carry decoder hidden state for selected beams
            beam_hidden = decoder_out.hidden[gather_index, :]

        selection_mask = torch.zeros(batch_size, num_beams, dtype=bool, device=device)
        selection_mask[:, :num_return_sequences] = True

        selected_codes = beam_codes[selection_mask.view(-1), :]
        selected_codes = selected_codes.view(batch_size, num_return_sequences, self.num_levels)

        if return_scores:
            selected_scores = beam_scores[selection_mask.view(-1)]
            selected_scores = selected_scores / self.num_levels
            return selected_codes, selected_scores.view(batch_size, num_return_sequences)

        return selected_codes


class SemanticCodeDecoderWrapper(nn.Module):
    """
    Thin wrapper to keep old `SemanticCodeDecoder`/`SemanticCodeDecoderModel` imports working.
    Uses the full encoder-decoder `SemanticCodePredictor` under the hood.
    """
    
    def __init__(self, num_layers=None, encoder_layers=None, decoder_layers=None, **kwargs):
        super().__init__()
        
        # Handle backward compatibility: num_layers -> split into encoder/decoder
        if num_layers is not None and encoder_layers is None and decoder_layers is None:
            encoder_layers = num_layers // 2
            decoder_layers = num_layers // 2
        elif encoder_layers is None:
            encoder_layers = 3
        if decoder_layers is None:
            decoder_layers = 3
        
        self.model = SemanticCodePredictor(
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            **kwargs
        )
        # Copy attributes for backward compatibility
        self.codebook_size = self.model.codebook_size
        self.num_levels = self.model.num_levels
        self.hidden_dim = self.model.hidden_dim
    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, patch_emb=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            patch_emb=patch_emb,
        )
    
    def get_next_code_probs(self, input_ids, attention_mask=None, patch_emb=None):
        return self.model.get_next_code_probs(
            input_ids,
            attention_mask=attention_mask,
            patch_emb=patch_emb,
        )
    
    def sample_next_code(self, input_ids, attention_mask=None, temperature=1.0, patch_emb=None):
        return self.model.sample_next_code(
            input_ids,
            attention_mask=attention_mask,
            temperature=temperature,
            patch_emb=patch_emb,
        )

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def beam_search(self, *args, **kwargs):
        return self.model.beam_search(*args, **kwargs)


# Backward compatibility aliases
SemanticCodeDecoderModel = SemanticCodePredictor
SemanticCodeDecoderLegacy = SemanticCodeDecoderWrapper
SemanticCodeDecoder = SemanticCodeDecoderWrapper


class SemanticCodeDecoderOnly(nn.Module):
    """
    Decoder-only variant: uses a causal transformer over the history to build context,
    then predicts the next item's semantic codes level by level without cross-attention.
    """
    
    def __init__(
        self,
        codebook_size: int = 256,
        num_levels: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        max_seq_len: int = 512,
        carry_decoder_state: bool = False,
    ):
        super().__init__()
        
        self.codebook_size = codebook_size
        self.vocab_size = codebook_size + 1
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.carry_decoder_state = carry_decoder_state
        
        self.code_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, hidden_dim, padding_idx=0)
            for _ in range(num_levels)
        ])
        
        self.causal_encoder = CausalEncoder(
            code_embeddings=self.code_embeddings,
            num_levels=num_levels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        
        self.level_mlps = nn.ModuleList([
            nn.Sequential(
                RMSNorm(hidden_dim),
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_levels)
        ])
        
        self.output_projs = nn.ModuleList([
            nn.Linear(hidden_dim, self.vocab_size)
            for _ in range(num_levels)
        ])
        
        self.bos_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)
    
    def _next_state(self, x_level: torch.Tensor, code_emb: torch.Tensor) -> torch.Tensor:
        if self.carry_decoder_state:
            return x_level + code_emb
        return code_emb
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        max_level: Optional[int] = None,
        patch_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.causal_encoder(
            input_ids,
            attention_mask=attention_mask,
            patch_emb=patch_emb,
        )
        encoded = encoder_output.encoded
        enc_mask = encoder_output.attention_mask
        
        if enc_mask is not None:
            last_valid_pos = enc_mask.long().sum(dim=1) - 1
            batch_size = encoded.shape[0]
            context = encoded[torch.arange(batch_size), last_valid_pos, :]
        else:
            context = encoded[:, -1, :]
        
        x = context + self.bos_embedding
        
        all_logits = []
        num_levels_to_decode = self.num_levels if max_level is None else min(self.num_levels, max_level)
        
        for level in range(num_levels_to_decode):
            x_level = self.level_mlps[level](x)
            level_logits = self.output_projs[level](x_level)
            level_logits[:, 0] = -1e9
            all_logits.append(level_logits)
            
            if decoder_input_ids is not None and level < decoder_input_ids.size(1):
                code = decoder_input_ids[:, level]
            else:
                code = torch.argmax(level_logits, dim=-1)
            code_emb = self.code_embeddings[level](code)
            x = self._next_state(x_level, code_emb)
        
        logits = torch.stack(all_logits, dim=1)
        return logits, x
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        return_scores: bool = False,
        patch_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if num_beams == 1:
            logits, _ = self.forward(
                input_ids,
                attention_mask=attention_mask,
                patch_emb=patch_emb,
            )
            next_codes = torch.argmax(logits, dim=-1)
            return next_codes.unsqueeze(1)
        else:
            return self.beam_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                return_scores=return_scores,
                patch_emb=patch_emb,
            )
    
    def beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 5,
        num_return_sequences: int = 1,
        return_scores: bool = False,
        patch_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_return_sequences = min(num_return_sequences, num_beams)
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        with torch.no_grad():
            encoder_output = self.causal_encoder(
                input_ids,
                attention_mask=attention_mask,
                patch_emb=patch_emb,
            )
            encoded = encoder_output.encoded
            enc_mask = encoder_output.attention_mask if encoder_output.attention_mask is not None else attention_mask
            
            if enc_mask is not None:
                last_valid_pos = enc_mask.long().sum(dim=1) - 1
                context = encoded[torch.arange(batch_size, device=device), last_valid_pos, :]
            else:
                context = encoded[:, -1, :]
            
            context = context + self.bos_embedding
            beam_hidden = context.repeat_interleave(num_beams, dim=0)
        
        beam_codes = torch.zeros(
            (batch_size * num_beams, self.num_levels),
            dtype=torch.long,
            device=device,
        )
        beam_scores = torch.zeros((batch_size, num_beams), device=device)
        beam_scores[:, 1:] = float("-inf")
        beam_scores = beam_scores.view(-1)
        beam_offset = (torch.arange(batch_size, device=device) * num_beams).unsqueeze(1)
        
        for level in range(self.num_levels):
            with torch.no_grad():
                x_level = self.level_mlps[level](beam_hidden)
                level_logits = self.output_projs[level](x_level)
                level_logits[:, 0] = -1e9
            
            level_scores = torch.log_softmax(level_logits, dim=-1)
            level_scores = level_scores + beam_scores[:, None]
            level_scores = level_scores.view(batch_size, num_beams * self.vocab_size)
            
            top_scores, top_indices = torch.topk(
                level_scores,
                min(2 * num_beams, num_beams * self.vocab_size),
                dim=1,
                largest=True,
                sorted=True,
            )
            
            beam_indices = torch.div(top_indices, self.vocab_size, rounding_mode="floor")
            level_codes = top_indices % self.vocab_size
            
            top_scores = top_scores[:, :num_beams]
            beam_indices = beam_indices[:, :num_beams]
            level_codes = level_codes[:, :num_beams]
            
            beam_scores = top_scores.reshape(-1)
            gather_index = (beam_indices + beam_offset).reshape(-1)
            
            old_beam_codes = beam_codes[gather_index, :]
            new_level_codes = level_codes.reshape(-1, 1)
            beam_codes = torch.cat([old_beam_codes[:, :level], new_level_codes], dim=1)
            
            code_emb = self.code_embeddings[level](beam_codes[:, level])
            beam_hidden = self._next_state(x_level, code_emb)
        
        selection_mask = torch.zeros(batch_size, num_beams, dtype=bool, device=device)
        selection_mask[:, :num_return_sequences] = True
        
        selected_codes = beam_codes[selection_mask.view(-1), :]
        selected_codes = selected_codes.view(batch_size, num_return_sequences, self.num_levels)
        
        if return_scores:
            selected_scores = beam_scores[selection_mask.view(-1)]
            selected_scores = selected_scores / self.num_levels
            return selected_codes, selected_scores.view(batch_size, num_return_sequences)
        
        return selected_codes
