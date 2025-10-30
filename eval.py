# eval.py  —— 合并共享逻辑 + 离线CRE + 在线CRE
import os, json, yaml, torch, torch.nn.functional as F
from typing import Optional, Tuple, Iterable
from contextlib import nullcontext

from data import make_dataloaders_from_txt
from model_id_t5 import ItemTable, build_student, GlobalSoftPatch, logits_from_ids


# =========================
# 公共：Top-K 指标（与你原来 _topk_metrics 口径一致）
# =========================
@torch.no_grad()
def compute_topk_metrics(scores: torch.Tensor, targets: torch.Tensor, ks: Iterable[int], one_based_labels: bool = True):
    """
    scores: [N, |I|]  —— 对 E.table[1:] 的打分
    targets: [N]      —— 如果 one_based_labels=True，则为 1..|I|，内部会 -1 对齐
    """
    if scores.numel() == 0:
        return {f"HR@{k}": 0.0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks}

    scores = scores.float()
    device = scores.device
    tgt = targets.to(device)
    if one_based_labels:
        tgt = tgt - 1

    maxk = max(ks)
    _, topk_idx = torch.topk(scores, k=maxk, dim=1)  # [N, maxk]

    out = {}
    for k in ks:
        hits = (topk_idx[:, :k] == tgt.unsqueeze(1))   # [N, k]
        h = hits.any(dim=1).float()
        hr = h.mean().item()
        pos = torch.argmax(hits.float(), dim=1)        # 未命中时为0，但乘h遮罩
        ndcg = (h * (1.0 / torch.log2(pos.float() + 2.0))).mean().item()
        out[f"HR@{k}"] = hr
        out[f"NDCG@{k}"] = ndcg
    return out


# =========================
# 公共：裁剪到最后 L_real 非PAD 并右填充
# =========================
def clip_right_pad(batch_ids: torch.Tensor, batch_mask: torch.Tensor, L_real_eval: int):
    """
    取每条序列最后 L_real_eval 个非 PAD；右填充到 batch 内最大长度。
    """
    B, Lr = batch_ids.size()
    kept_list = []
    for i in range(B):
        ids = batch_ids[i]
        msk = batch_mask[i]
        length = int(msk.sum().item())
        keep = min(length, L_real_eval)
        kept = ids[:length][-keep:] if keep > 0 else ids[:0]
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


# =========================
# AMP 小工具（避免循环依赖 utils.AmpAutocast）
# =========================
def _parse_amp(amp_setting):
    if isinstance(amp_setting, str):
        s = amp_setting.lower()
        if "bf16" in s or "bfloat16" in s: return torch.bfloat16, False
        if "fp16" in s or "float16" in s or "half" in s: return torch.float16, True
        return None, False
    if isinstance(amp_setting, bool) and amp_setting: return torch.float16, True
    return None, False

def _amp_ctx(amp_setting):
    dtype, _ = _parse_amp(amp_setting)
    return torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()

def _make_scaler(amp_setting):
    _, use_scaler = _parse_amp(amp_setting)
    return torch.amp.GradScaler("cuda", enabled=use_scaler)


# =========================
# 公共：按 (η ⊕ recent) 形态收集 scores/targets
# =========================
@torch.no_grad()
def collect_scores_targets(
    student,
    item_table,
    eta_tensor: torch.Tensor,                # [L_soft, d] 或空张量 [0, d]
    loader,
    L_soft_eval: int,
    L_real_eval: int,
    pool: str,
    amp_setting,
    clip_last: bool = True,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    student.eval()
    device = next(student.parameters()).device
    all_scores, all_targets = [], []

    for bidx, (recent_ids, targets, mask_recent) in enumerate(loader):
        if (max_batches is not None) and (bidx >= max_batches):
            break
        recent_ids  = recent_ids.to(device)
        mask_recent = mask_recent.to(device)
        targets     = targets.to(device)

        if clip_last:
            ids_eval, mask_eval = clip_right_pad(recent_ids, mask_recent, L_real_eval)
        else:
            ids_eval  = recent_ids[:, -L_real_eval:]
            mask_eval = mask_recent[:, -L_real_eval:]

        with _amp_ctx(amp_setting):
            scores, _ = logits_from_ids(
                student=student,
                item_table=item_table,
                eta_tensor=eta_tensor.to(device),
                recent_ids=ids_eval,
                mask_recent=mask_eval,
                L_soft=L_soft_eval,
                pool=pool
            )
            scores = scores.float()

        all_scores.append(scores.cpu())
        all_targets.append(targets.cpu())

    scores  = torch.cat(all_scores,  dim=0) if len(all_scores)  else torch.empty(0, item_table.table.size(0)-1)
    targets = torch.cat(all_targets, dim=0) if len(all_targets) else torch.empty(0, dtype=torch.long)
    student.train()
    return scores, targets


# =========================
# 公共：CRE 重训到收敛（返回 best_state, best_val）
# =========================
def cre_train_one_mode(
    student, item_table, eta_tensor,
    dl_tr, dl_va, L_soft, L_real, pool, ks, amp_setting,
    max_epochs, lr, weight_decay, patience, metric_key,
    grad_clip=0.0, max_train_batches=None, max_val_batches=None,
    metrics_fn=compute_topk_metrics
):
    for p in student.parameters(): p.requires_grad = True
    device = next(student.parameters()).device
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = _make_scaler(amp_setting)

    best_metric = -1.0
    best_state  = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    wait = 0

    for epoch in range(1, max_epochs + 1):
        student.train()

        for bidx, (recent_ids, targets, mask_recent) in enumerate(dl_tr):
            if (max_train_batches is not None) and (bidx >= max_train_batches):
                break
            recent_ids  = recent_ids.to(device)
            mask_recent = mask_recent.to(device)
            targets     = targets.to(device)

            ids_eval, mask_eval = clip_right_pad(recent_ids, mask_recent, L_real)

            with _amp_ctx(amp_setting):
                logits, _ = logits_from_ids(
                    student=student,
                    item_table=item_table,
                    eta_tensor=eta_tensor,          # [L_soft, d] 或空
                    recent_ids=ids_eval,
                    mask_recent=mask_eval,
                    L_soft=L_soft,
                    pool=pool
                )
                loss = F.cross_entropy(logits, targets - 1)  # 你当前实现：labels 为 1-based

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                opt.step()

        # 验证（同一 mode）→ 统一 metrics 口径做早停
        scores, targets = collect_scores_targets(
            student=student, item_table=item_table, eta_tensor=eta_tensor,
            loader=dl_va, L_soft_eval=L_soft, L_real_eval=L_real,
            pool=pool, amp_setting=amp_setting, clip_last=True, max_batches=max_val_batches
        )
        metrics = metrics_fn(scores, targets, ks) if scores.numel() else {metric_key: 0.0}
        val_score = float(metrics.get(metric_key, 0.0))

        if val_score > best_metric:
            best_metric = val_score
            best_state  = {k: v.detach().cpu() for k, v in student.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return best_state, best_metric


# =========================
# 离线 CRE 评测（与之前 evaluate(cfg) 兼容）
# =========================
def evaluate(cfg: dict):
    """
    对 cfg['eval']['modes'] 中的每个 mode（指定 L_soft, L_real）：
      1) 从同一起点初始化一个新的 student；
      2) 用 [eta ⊕ recent(L_real)] 在 train/val 上重训到早停；
      3) 在 test 上评 Metrics（HR/NDCG）。
    """
    file_path = os.path.dirname(__file__)
    device   = cfg["system"]["device"]
    dsname   = cfg["data"]["name"]
    proc_dir = os.path.join(file_path, "data", dsname, "proc")

    # ===== 1) 基准（短序列）loader：用于 recent / patch / 默认 =====
    base_dl_tr, base_dl_va, base_dl_te = make_dataloaders_from_txt(
        proc_dir,
        cfg["data"]["L_real"],                  # 例如 128
        cfg["train"]["batch_size"]
    )

    # ===== 2) full 专用（长序列）loader：只在需要 full 时才建 =====
    modes = cfg["eval"]["modes"]
    need_full = any(str(m["name"]).lower() == "full" for m in modes)
    if need_full:
        L_full = max(int(m["L_real"]) for m in modes if str(m["name"]).lower() == "full")
        bs_full = int(cfg["eval"].get("batch_size_full", max(8, cfg["train"]["batch_size"] // 8)))
        dl_tr_full, dl_va_full, dl_te_full = make_dataloaders_from_txt(
            proc_dir,
            L_full,                               # 例如 1000
            bs_full
        )
    else:
        dl_tr_full = dl_va_full = dl_te_full = None
    
    # # 看 loader 里实际能提供的长度分布
    # lens = []
    # for recent_ids, _, mask_recent in dl_te:
    #     lens.append(mask_recent.sum(dim=1).cpu())
    # print("loader lengths (min/median/max):",
    #     torch.cat(lens).min().item(),
    #     torch.quantile(torch.cat(lens).float(), torch.tensor(0.5)).item(),
    #     torch.cat(lens).max().item())


    # 载入 ckpt（ItemTable + φ）
    ckpt_path = (
        cfg.get("cre", {}).get("ckpt_path")
        or cfg.get("eval", {}).get("ckpt_path")
        or os.path.join(file_path, "artifacts", "ckpt3.pt")
    )
    ckpt = torch.load(ckpt_path, map_location=device)

    with open(os.path.join(proc_dir, "item2idx.json")) as f:
        num_items = len(json.load(f))

    # Item table（冻结）
    E = ItemTable(num_items, cfg["items"]["d_model"], trainable=False).to(device)
    E.load_state_dict(ckpt["item_table"])
    print(f"[Init] ItemTable: num_items={num_items}, d_model={cfg['items']['d_model']}, trainable=False")

    phi_state = ckpt["phi"]

    # 评测配置
    ks    = cfg["eval"]["k_list"]
    modes = cfg["eval"]["modes"]
    pool  = cfg["head"]["pool"]
    amp   = cfg["system"]["amp"]

    # CRE 训练超参
    cre_cfg = cfg.get("cre", {})
    max_epochs  = int(cre_cfg.get("epochs", 20))
    lr          = float(cre_cfg.get("lr", 1e-4))
    weight_decay= float(cre_cfg.get("wd", 0.0))
    patience_es = int(cre_cfg.get("early_stop", 3))
    metric_key  = cre_cfg.get("metric", "NDCG@20")
    grad_clip   = float(cre_cfg.get("grad_clip", 0.0))
    max_train_batches = cre_cfg.get("max_train_batches", None)
    max_val_batches   = cre_cfg.get("max_val_batches", None)

    # 一个小函数：从 phi_state 推断 φ 的长度
    def _infer_phi_len(phi_state_dict):
        for k, v in phi_state_dict.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                return v.shape[0]
        return 0

    phi_len_ckpt = _infer_phi_len(phi_state)

    results = {}
    for mode in modes:

        name        = str(mode["name"]).lower()
        L_soft_eval = int(mode["L_soft"])
        L_real_eval = int(mode["L_real"])

        # 给 full 模式用专门的 loader
        if name == "full" and dl_tr_full is not None:
            dl_tr_mode, dl_va_mode, dl_te_mode = dl_tr_full, dl_va_full, dl_te_full
        else:
            dl_tr_mode, dl_va_mode, dl_te_mode = base_dl_tr, base_dl_va, base_dl_te
        
        # 构造 eta（空补丁或从 ckpt 还原）
        if L_soft_eval > 0:
            if phi_len_ckpt != L_soft_eval:
                raise RuntimeError(
                    f"[eval] Soft-patch length mismatch: ckpt phi_len={phi_len_ckpt}, "
                    f"but eval.modes[{name}].L_soft={L_soft_eval}. "
                    f"→ 请确保 ckpt 与配置一致或调整配置/重导出 ckpt。\nckpt_path={ckpt_path}"
                )
            d_model = E.table.size(1)
            phi = GlobalSoftPatch(L_soft_eval, d_model, device=device).to(device)
            phi.load_state_dict(phi_state)
            eta_eval = phi.phi  # [L_soft, d]
        else:
            d_model = E.table.size(1)
            eta_eval = torch.empty(0, d_model, device=device)

        # 从同一起点初始化新的 student
        student = build_student(cfg["student"]["t5_name"], device, cfg["student"]["grad_ckpt"])

        # CRE 重训（train/val）
        best_state, best_val = cre_train_one_mode(
            student=student, item_table=E, eta_tensor=eta_eval,
            dl_tr=dl_tr_mode, dl_va=dl_va_mode,
            L_soft=L_soft_eval, L_real=L_real_eval, pool=pool, ks=ks, amp_setting=amp,
            max_epochs=max_epochs, lr=lr, weight_decay=weight_decay,
            patience=patience_es, metric_key=metric_key, grad_clip=grad_clip,
            max_train_batches=max_train_batches, max_val_batches=max_val_batches,
            metrics_fn=compute_topk_metrics
        )

        # Test：载入最佳权重 → 评测
        student.load_state_dict(best_state)
        scores, targets = collect_scores_targets(
            student=student, item_table=E, eta_tensor=eta_eval,
            loader=dl_te_mode, L_soft_eval=L_soft_eval, L_real_eval=L_real_eval,
            pool=pool, amp_setting=amp, clip_last=True, max_batches=None
        )
        metrics = compute_topk_metrics(scores, targets, ks)
        results[name] = metrics
        print(f"[CRE|{name}] val_best={best_val:.4f}  " +
              " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    return results


# =========================
# 在线 CRE（给训练循环调用）
# =========================
def run_eval_online_cre(E, phi, dl_tr, dl_va, cfg):
    """
    在线 CRE：使用“当前时刻的 soft-patch φ（eta）⊕ recent”，
    从同一起点初始化一个新 student，重训到早停，再在验证集上评测。
    返回 metrics dict（HR@K / NDCG@K）。
    """
    device   = cfg["system"]["device"]
    amp      = cfg["system"]["amp"]
    ks       = tuple(cfg["train"]["metrics_topk"])
    pool     = cfg["head"]["pool"]
    clip_last= bool(cfg["eval"].get("clip_last", True))

    # 选取一个评测 mode（通常是 cfg['eval']['best_mode']）
    best_mode = cfg["eval"].get("best_mode", "patch")
    modes     = cfg["eval"]["modes"]
    mcfg      = next((m for m in modes if m["name"] == best_mode), None)
    if mcfg is None:
        raise ValueError(f"[run_eval_online_cre] best_mode='{best_mode}' 不在 eval.modes 中。")
    L_soft = int(mcfg["L_soft"])
    L_real = int(mcfg["L_real"])

    # 构造 eta（在线：直接用当前 φ.phi）
    if L_soft > 0:
        eta = phi.phi.to(device)
        if eta.size(0) != L_soft:
            raise AssertionError(f"[online_cre] phi length {eta.size(0)} != L_soft {L_soft}")
    else:
        d_model = E.table.size(1)
        eta = torch.empty(0, d_model, device=device)

    # CRE 在线评测用的超参（如未配置，则回退到 cre.*）
    cre_on = cfg.get("cre_online", {})
    cre    = cfg.get("cre", {})
    max_epochs  = int(cre_on.get("epochs",      cre.get("epochs", 10)))
    lr          = float(cre_on.get("lr",        cre.get("lr", 1e-4)))
    weight_decay= float(cre_on.get("wd",        cre.get("wd", 0.0)))
    patience    = int(cre_on.get("early_stop",  cre.get("early_stop", 2)))
    metric_key  =      cre_on.get("metric",     cre.get("metric", "NDCG@20"))
    grad_clip   = float(cre_on.get("grad_clip", cre.get("grad_clip", 0.0)))
    max_train_batches = cre_on.get("max_train_batches", None)
    max_val_batches   = cre_on.get("max_val_batches",   cfg["train"]["max_eval_batches"])

    # 从同一起点初始化新 student
    student = build_student(cfg["student"]["t5_name"], device, cfg["student"]["grad_ckpt"])

    # CRE 重训（train/val）
    best_state, _ = cre_train_one_mode(
        student=student, item_table=E, eta_tensor=eta,
        dl_tr=dl_tr, dl_va=dl_va,
        L_soft=L_soft, L_real=L_real, pool=pool, ks=ks, amp_setting=amp,
        max_epochs=max_epochs, lr=lr, weight_decay=weight_decay,
        patience=patience, metric_key=metric_key, grad_clip=grad_clip,
        max_train_batches=max_train_batches, max_val_batches=max_val_batches,
        metrics_fn=compute_topk_metrics
    )

    # val 集上评测
    student.load_state_dict(best_state)
    scores, targets = collect_scores_targets(
        student=student, item_table=E, eta_tensor=eta,
        loader=dl_va, L_soft_eval=L_soft, L_real_eval=L_real,
        pool=pool, amp_setting=amp, clip_last=clip_last, max_batches=max_val_batches
    )
    return compute_topk_metrics(scores, targets, ks)


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    evaluate(cfg)
