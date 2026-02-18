from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration

class ItemTable(nn.Module):
    def __init__(self, num_items: int, d_model: int, trainable: bool = False, init: str = "normal"):
        super().__init__()
        # item_table: reserve PAD=0
        self.emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        if init == "zeros":
            nn.init.zeros_(self.emb.weight)
        else:
            nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, ids: torch.LongTensor) -> torch.FloatTensor:
        return self.emb(ids)
    @property
    def table(self) -> torch.Tensor: return self.emb.weight

class GlobalSoftPatch(nn.Module):
    def __init__(self, L_soft: int, d_model: int, device: Optional[str] = None):
        super().__init__()
        self.phi = nn.Parameter(torch.randn(L_soft, d_model) * 0.02)
        if device:
            self.to(device)

    def forward(self, B: int) -> torch.Tensor:
        return self.phi.unsqueeze(0).expand(B, -1, -1)

def build_student(t5_name: str, device: str, grad_ckpt: bool = False) -> T5ForConditionalGeneration:
    model = T5ForConditionalGeneration.from_pretrained(t5_name)
    model.to(device)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    return model

def logits_from_ids(student,
                    item_table,          # ItemTable
                    eta_tensor,          # None | [L_soft, d] | [B, L_soft, d]
                    recent_ids,          # [B, Lr]
                    mask_recent,         # [B, Lr] 0/1
                    L_soft: int,         # 仅作兼容，不再直接使用；以实际补丁长度为准
                    pool: str = 'last',
                    use_cosine: bool | None = None,
                    temperature: float | None = None):
    device = next(student.parameters()).device
    recent_ids  = recent_ids.to(device)
    mask_recent = mask_recent.to(device).long()

    B, Lr = recent_ids.size()
    emb_recent = item_table(recent_ids)                    # [B, Lr, d]
    d = emb_recent.size(-1)

    # ---- patch 归一化开关（从 E 上读取；默认 True, 'l2'）----
    normalize_patch = bool(getattr(item_table, "normalize_patch_default", True))
    patch_norm_kind = getattr(item_table, "patch_norm_kind", "l2")  # 'l2' | 'ln' | 'none'

    # ---- 组 patch：None / [L,d] / [B,L,d] ----
    if (eta_tensor is None) or (eta_tensor.numel() == 0):
        patch = emb_recent.new_zeros((B, 0, d))            # [B,0,d]
    else:
        eta = eta_tensor.to(device)
        if normalize_patch and eta.numel() > 0:
            if patch_norm_kind == "l2":
                eta = F.normalize(eta, dim=-1, eps=1e-6)
            elif patch_norm_kind == "ln":
                eta = (eta - eta.mean(dim=-1, keepdim=True)) / (eta.std(dim=-1, keepdim=True) + 1e-6)
            # 'none' 不处理
        if eta.dim() == 2:                                 # [L_soft, d]
            patch = eta.unsqueeze(0).expand(B, -1, -1)     # [B, Ls, d]
        elif eta.dim() == 3:                               # [B, L_soft, d]
            assert eta.size(0) == B, "eta_tensor batch mismatch"
            patch = eta
        else:
            raise ValueError("eta_tensor must be [L_soft,d] or [B,L_soft,d]")

    Ls = patch.size(1)                                     # ← 实际补丁长度
    inputs = torch.cat([patch, emb_recent], dim=1)         # [B, Ls+Lr, d]

    # ---- attention mask 用 Ls，而不是传入的 L_soft ----
    attn_patch  = torch.ones((B, Ls), dtype=torch.long, device=device)
    attn        = torch.cat([attn_patch, mask_recent], dim=1)  # [B, Ls+Lr]

    # ---- encoder 前向 ----
    enc_out = student.encoder(inputs_embeds=inputs, attention_mask=attn, return_dict=True)
    H = enc_out.last_hidden_state                           # [B, Ls+Lr, d]

    # ---- pool: last / mean —— 用 Ls 对齐 ----
    if pool == 'last':
        lengths = mask_recent.sum(dim=1)                    # [B]
        idx_last = (lengths - 1).clamp_min(0)               # [B]
        pos = Ls + idx_last                                 # [B]
        u = H[torch.arange(B, device=device), pos, :]       # [B, d]
    elif pool == 'mean':
        denom = mask_recent.sum(dim=1).clamp_min(1).unsqueeze(1).float()
        recent_H = H[:, Ls:, :]                             # [B, Lr, d]
        u = (recent_H * mask_recent.unsqueeze(-1).float()).sum(dim=1) / denom
    else:
        raise ValueError("pool must be 'last' or 'mean'")

    # ---- 打分口径：默认从 E 读取 ----
    if use_cosine is None:
        use_cosine = bool(getattr(item_table, "use_cosine_default", True))

    if temperature is None:
        if hasattr(item_table, "logit_scale"):              # 对数温度
            t = getattr(item_table, "logit_scale")
            temperature = float(torch.exp(t).item()) if isinstance(t, torch.Tensor) \
                          else float(torch.exp(torch.tensor(t)).item())
        elif hasattr(item_table, "temperature"):
            temperature = float(getattr(item_table, "temperature"))
        else:
            temperature = 1.0

    if use_cosine:
        u_n   = F.normalize(u, dim=-1, eps=1e-6)
        tab_n = F.normalize(item_table.table, dim=-1, eps=1e-6)
        logits = (u_n @ tab_n[1:].T).float() * float(temperature)   # [B, |I|]
    else:
        logits = (u @ item_table.table[1:].T).float()

    return logits, u


