from __future__ import annotations
from typing import List, Tuple, Optional
import math, torch, torch.nn as nn, torch.nn.functional as F
from transformers import T5ForConditionalGeneration

class ItemTable(nn.Module):
    def __init__(self, num_items:int, d_model:int, trainable:bool=False, init:str="normal"):
        super().__init__()
        # item_table: reserve PAD=0
        self.emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)   # +1, 0用于PAD
        print(f"[Init] ItemTable: num_items={num_items}, d_model={d_model}, trainable={trainable}")
        if init=="zeros": nn.init.zeros_(self.emb.weight)
        else: nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        for p in self.parameters(): p.requires_grad=trainable
    def forward(self, ids: torch.LongTensor) -> torch.FloatTensor: return self.emb(ids)
    @property
    def table(self) -> torch.Tensor: return self.emb.weight

class GlobalSoftPatch(nn.Module):
    def __init__(self, L_soft:int, d_model:int, device:Optional[str]=None):
        super().__init__(); self.phi = nn.Parameter(torch.randn(L_soft, d_model)*0.02)
        if device: self.to(device)
    def forward(self, B:int) -> torch.Tensor: return self.phi.unsqueeze(0).expand(B,-1,-1)

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r:int, alpha:float, dropout:float=0.0):
        super().__init__(); self.r=r; self.scaling=alpha/max(r,1)
        self.weight=base.weight; self.bias=base.bias
        self.weight.requires_grad=False
        if self.bias is not None: self.bias.requires_grad=False
        if r>0:
            self.lora_A=nn.Linear(base.in_features,r,bias=False)
            self.lora_B=nn.Linear(r,base.out_features,bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)); nn.init.zeros_(self.lora_B.weight)
            self.drop=nn.Dropout(dropout)
        else: self.lora_A=None; self.lora_B=None; self.drop=nn.Identity()
    def forward(self,x):
        base_out=F.linear(x,self.weight,self.bias)
        if self.r>0: return base_out + self.lora_B(self.lora_A(self.drop(x)))*self.scaling
        return base_out

def apply_lora_qv(student: T5ForConditionalGeneration, r:int, alpha:float, target_blocks:int, dropout:float=0.0) -> int:
    enc=student.encoder; blocks=enc.block[:target_blocks]; k=0
    for blk in blocks:
        attn=blk.layer[0].SelfAttention
        for name in ["q","v"]:
            base=getattr(attn,name); lora=LoRALinear(base,r,alpha,dropout).to(base.weight.device)
            setattr(attn,name,lora); k+=1
    return k

def build_student(t5_name:str, device:str, grad_ckpt:bool=False) -> T5ForConditionalGeneration:
    model=T5ForConditionalGeneration.from_pretrained(t5_name); model.to(device) # 这个居然是用了pretrained啊
    if grad_ckpt: model.gradient_checkpointing_enable()
    return model

# class EncoderOnlyScorer(nn.Module):
#     def __init__(self, d_model:int, pool:str="first"):
#         super().__init__(); assert pool in ("first","mean"); self.pool=pool
#     def forward(self, enc_last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
#         if self.pool=="first": return enc_last_hidden[:,0,:]
#         denom = attn_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
#         return (enc_last_hidden * attn_mask.unsqueeze(-1).float()).sum(dim=1) / denom

def logits_from_ids(student,
                    item_table,          # ItemTable
                    eta_tensor,          # torch.Tensor, shape [L_soft, d]
                    recent_ids,          # [B, Lr]
                    mask_recent,         # [B, Lr] 0/1
                    L_soft: int,
                    pool: str = 'last'):
    device = next(student.parameters()).device
    B, Lr = recent_ids.size()

    emb_recent = item_table(recent_ids.to(device))          # [B, Lr, d]
    patch = eta_tensor.unsqueeze(0).expand(B, -1, -1)       # [B, L_soft, d]  <-- 用 eta_tensor
    inputs = torch.cat([patch, emb_recent], dim=1)          # [B, Ls+Lr, d]

    attn_patch = torch.ones((B, L_soft), dtype=torch.long, device=device)
    attn_recent = mask_recent.to(device).long()
    attn = torch.cat([attn_patch, attn_recent], dim=1)      # [B, Ls+Lr]

    enc_out = student.encoder(inputs_embeds=inputs, attention_mask=attn, return_dict=True)
    H = enc_out.last_hidden_state                           # [B, Ls+Lr, d]

    if pool == 'last':
        lengths = mask_recent.sum(dim=1)                    # [B]
        idx_last_recent = (lengths - 1).clamp_min(0)        # [B]
        pos = L_soft + idx_last_recent                      # [B]
        u = H[torch.arange(B, device=device), pos, :]       # [B, d]
    elif pool == 'mean':
        denom = mask_recent.sum(dim=1).clamp_min(1).unsqueeze(1).to(device).float()
        u = (H[:, L_soft:, :] * mask_recent.unsqueeze(-1).to(device).float()).sum(dim=1) / denom
    else:
        raise ValueError("pool must be 'last' or 'mean'")

    logits = u @ item_table.table[1:].T                     # [B, |I|]  去掉 PAD 列
    return logits, u
