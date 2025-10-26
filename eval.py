# eval.py
import os, json, yaml, torch, numpy as np
from typing import List, Dict
from utils import AmpAutocast
from data import make_dataloaders_from_txt
from model_id_t5 import ItemTable, build_student, GlobalSoftPatch, logits_from_ids

def _clip_right_pad(batch_ids: torch.Tensor, batch_mask: torch.Tensor, L_real_eval: int):
    """
    保留每条序列最后 L_real_eval 个非 PAD 元素；再右填充到 batch 内最大长度。
    输入:
      batch_ids:  [B, Lr]
      batch_mask: [B, Lr] 0/1
    输出:
      out_ids:  [B, L_eval]
      out_mask: [B, L_eval]
    """
    B, Lr = batch_ids.size()
    kept_list = []
    for i in range(B):
        ids = batch_ids[i]
        msk = batch_mask[i]
        length = int(msk.sum().item())
        keep = min(length, L_real_eval)
        if keep > 0:
            kept = ids[:length][-keep:]   # 仅保留非 PAD 段的最后 keep 个
        else:
            kept = ids[:0]                # 空
        kept_list.append(kept)

    max_len = max((x.size(0) for x in kept_list), default=0)
    device = batch_ids.device
    out_ids  = torch.zeros((B, max_len), dtype=torch.long, device=device)
    out_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, kept in enumerate(kept_list):
        L = kept.size(0)
        if L > 0:
            out_ids[i, :L]  = kept
            out_mask[i, :L] = 1
    return out_ids, out_mask

@torch.no_grad()
def _topk_metrics(scores: torch.Tensor, targets: torch.Tensor, ks=(10,20,50)):
    """
    scores: [B, |I|]  (可能是 bf16/fp16/fp32)
    targets: [B]      (1..|I|)  注意：如果你的 scores 对应的是 E.table[1:]，要先减 1
    ks: 迭代的 K
    返回: dict
    """
    device = scores.device
    # 统一到 fp32，避免 bf16 → numpy 报错；且更安全
    scores = scores.float()

    # 取得 topk 索引（一次性拿到最大K）
    maxk = max(ks)
    topk_scores, topk_idx = torch.topk(scores, k=maxk, dim=1)  # [B, maxk]

    # 如果你的 scores 是对 E.table[1:] 的打分，则 item 索引是 [0..|I|-1]
    # 而 targets ∈ [1..|I|]，需要减 1 对齐：
    tgt = targets.to(device) - 1  # [B]

    # 命中矩阵: [B, maxk]
    hits = (topk_idx == tgt.unsqueeze(1))

    metrics = {}
    B = scores.size(0)
    for k in ks:
        h_at_k = hits[:, :k].any(dim=1).float()                    # [B]
        # HitRate@K
        hr = h_at_k.mean().item()
        # NDCG@K: 1/log2(rank+2) if hit else 0
        # rank：若命中，则是第一次出现的位置
        ranks = torch.argmax(hits[:, :k].float(), dim=1)           # [B]  命中位置；未命中则0，但下面会乘以 h_at_k
        ndcg = (1.0 / torch.log2(ranks.float() + 2.0) * h_at_k).mean().item()

        metrics[f"HR@{k}"] = hr
        metrics[f"NDCG@{k}"] = ndcg

    return metrics


def evaluate(cfg: dict):
    file_path = os.path.dirname(__file__)
    device  = cfg["system"]["device"]
    dsname  = cfg["data"]["name"]
    proc_dir = os.path.join(file_path, "data", dsname, "proc")

    # dataloader（测试集）
    _, _, dl_te = make_dataloaders_from_txt(proc_dir, cfg["data"]["L_real"], cfg["train"]["batch_size"])

    # 载入训练产物
    ckpt = torch.load(os.path.join(file_path, "artifacts","ckpt3.pt"), map_location=device)
    
    with open(os.path.join(proc_dir, "item2idx.json")) as f:
        num_items = len(json.load(f))
    E = ItemTable(num_items, cfg["items"]["d_model"], trainable=False).to(device)
    E.load_state_dict(ckpt["item_table"])

    student = build_student(cfg["student"]["t5_name"], device, cfg["student"]["grad_ckpt"])
    for p in student.parameters(): p.requires_grad = False

    phi_state = ckpt["phi"]  # 只有 patch 模式会用

    ks = cfg["eval"]["k_list"]
    modes = cfg["eval"]["modes"]  # 列表: {name, L_soft, L_real}

    results = {}
    with torch.no_grad():
        for mode in modes:
            name         = mode["name"]
            L_soft_eval  = int(mode["L_soft"])
            L_real_eval  = int(mode["L_real"])

            # 准备 eta_eval（tensor）
            if L_soft_eval > 0:
                d_model = E.table.size(1)
                phi = GlobalSoftPatch(L_soft_eval, d_model, device=device).to(device)
                phi.load_state_dict(phi_state)
                eta_eval = phi.phi  # [L_soft, d]
            else:
                # 空补丁（长度为0）
                d_model = E.table.size(1)
                eta_eval = torch.empty(0, d_model, device=device)

            all_scores, all_targets = [], []

            for recent_ids, targets, mask_recent in dl_te:
                recent_ids  = recent_ids.to(device)
                mask_recent = mask_recent.to(device)
                targets     = targets.to(device)

                # 裁剪到最后 L_real_eval 个
                ids_eval, mask_eval = _clip_right_pad(recent_ids, mask_recent, L_real_eval)

                with AmpAutocast(cfg["system"]["amp"]):
                    logits, _ = logits_from_ids(
                        student,
                        E,
                        eta_eval,
                        ids_eval,
                        mask_eval,
                        L_soft_eval,
                        pool='last'
                    )
                all_scores.append(logits)
                all_targets.append(targets)

            scores  = torch.cat(all_scores,  dim=0)
            targets = torch.cat(all_targets, dim=0)
            metrics = _topk_metrics(scores, targets, ks)
            results[name] = metrics
            print(f"[eval:{name}] " + " ".join(f"{k}={v:.4f}" for k,v in metrics.items()))

    return results

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    cfg = yaml.safe_load(open(path))
    evaluate(cfg)
