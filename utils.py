import torch, random, numpy as np

def set_seed(seed:int=1234):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class AmpAutocast:
    def __init__(self, amp_dtype:str='bf16'):
        self.amp_dtype = amp_dtype; self.ctx=None
    def __enter__(self):
        if self.amp_dtype in ('bf16','fp16') and torch.cuda.is_available():
            dtype = torch.bfloat16 if self.amp_dtype=='bf16' else torch.float16
            self.ctx = torch.amp.autocast('cuda', dtype=dtype); self.ctx.__enter__()
        return self
    def __exit__(self, et, ex, tb):
        if self.ctx: self.ctx.__exit__(et, ex, tb)

def max_cuda_gb():
    return (torch.cuda.max_memory_allocated()/(1024**3)) if torch.cuda.is_available() else 0.0


def pack_theta_state(student, theta_param_names):
    state = {}
    sd = student.state_dict()
    for n in theta_param_names:
        state[n] = sd[n].clone()
    return state


@torch.no_grad()
def run_eval_online(student, E, phi, dl_val, cfg):
    student.eval()
    ks = tuple(cfg["train"]["metrics_topk"])
    max_batches = cfg["train"]["max_eval_batches"]

    tot = 0
    hit = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}

    L_soft = cfg["compressor"]["L_soft"]

    done = 0
    for recent_ids, targets, mask_recent in dl_val:
        B = targets.size(0)
        # 全 soft
        inputs = phi.phi.unsqueeze(0).expand(B, -1, -1).to(student.device)
        attn   = torch.ones((B, L_soft), dtype=torch.long, device=student.device)

        out = student.encoder(inputs_embeds=inputs, attention_mask=attn, return_dict=True)
        u = out.last_hidden_state[:, -1, :]                               # [B, d]
        scores = (u @ E.table[1:].T).float()                              # [B, |I|]
        tgt = (targets - 1).to(scores.device)                             # [B]

        maxk = max(ks)
        _, topk_idx = torch.topk(scores, k=maxk, dim=1)                   # [B, maxk]
        hits = (topk_idx == tgt.unsqueeze(1))                             # [B, maxk]
        ranks = torch.argmax(hits.float(), dim=1)                          # 0..maxk-1（未命中时值0，但乘命中掩码）
        for k in ks:
            h = hits[:, :k].any(dim=1).float()
            hit[k]  += h.sum().item()
            ndcg[k] += (h * (1.0 / torch.log2(ranks.float() + 2.0))).sum().item()

        tot += B
        done += 1
        if max_batches is not None and done >= max_batches:
            break

    student.train()
    return {f"HR@{k}": hit[k]/tot for k in ks} | {f"NDCG@{k}": ndcg[k]/tot for k in ks}
