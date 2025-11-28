import os, json, yaml, torch
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from utils import set_seed, AmpAutocast, cuda_mem_gb
from model_id_t5 import ItemTable, GlobalSoftPatch, build_student, apply_lora_qv
from loss import ce_full_softmax
from mixflow import get_fwdrev_grad_fn_eta, MomentumInner
from eval import run_eval_online_cre


def train(cfg: dict):
    path = os.path.dirname(__file__)
    set_seed(cfg["train"]["seed"])
    device = cfg["system"]["device"]

    # -----------------------------
    # Data
    # -----------------------------
    dsname = cfg["data"]["name"]
    proc_dir = os.path.join(path, "data", dsname, "proc")
    print(f"\n[Init] Loading {dsname} dataset from {proc_dir}...", flush=True)

    num_items = len(json.load(open(os.path.join(proc_dir, "item2idx.json"))))
    if cfg["items"]["num_items"] is None:
        cfg["items"]["num_items"] = num_items
    print(f"[Init] num_items={num_items:,}, d_model={cfg['items']['d_model']}", flush=True)

    from data import make_dataloaders_from_txt

    # 训练/在线 CRE：用较长的训练 loader（你已配置 L_real_train）
    print(f"\n[Step 1/2] Creating train/val dataloaders (L_real={cfg['data'].get('L_real_train', cfg['data']['L_real'])})...", flush=True)
    L_real_train = cfg["data"].get("L_real_train", cfg["data"]["L_real"])
    dl_tr, dl_va, _ = make_dataloaders_from_txt(proc_dir, L_real_train, cfg["train"]["batch_size"])

    # ★ 新增：给 outer 用的"真实/full 序列" loader（对标 TD3 的真实数据监督）
    print(f"\n[Step 2/2] Creating full dataloaders for outer loop (L_full={cfg['data'].get('L_full_outer', cfg['data']['L_real'])})...", flush=True)
    L_full_outer = cfg["data"].get("L_full_outer", cfg["data"]["L_real"])
    dl_tr_full, _, _ = make_dataloaders_from_txt(proc_dir, L_full_outer, cfg["train"]["batch_size"])
    iter_full = iter(dl_tr_full)
    print(f"\n[Ready] All dataloaders created, starting training...\n", flush=True)

    # -----------------------------
    # Modules
    # -----------------------------
    E = ItemTable(cfg["items"]["num_items"], cfg["items"]["d_model"], trainable=cfg["items"]["trainable"]).to(device)

    if cfg["items"]["init_from_text"]:
        txt_path = os.path.join(path, "data", dsname, cfg["items"]["init_path"])
        E.table.data.copy_(torch.load(txt_path, map_location="cpu"))
        print("[Init] E from text:", txt_path)
    for p in E.parameters():
        p.requires_grad = cfg["items"]["trainable"]  # false -> 冻结

    student = build_student(cfg["student"]["t5_name"], device, cfg["student"]["grad_ckpt"])
    student.config.use_cache = False

    # 尽量使用非重入的 gradient checkpointing（兼容 functorch）
    try:
        student.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        print("[Warn] transformers 版本较旧，fallback 到默认 reentrant checkpointing")
        student.encoder.gradient_checkpointing_enable()

    # 冻结除 LoRA/全量微调部分之外的参数
    for p in student.parameters():
        p.requires_grad = False

    theta = []
    if not cfg["mixflow"]["full_ft"]:
        rep = apply_lora_qv(
            student,
            cfg["mixflow"]["lora_rank"],
            cfg["mixflow"]["lora_alpha"],
            cfg["mixflow"]["target_blocks"],
            cfg["mixflow"]["dropout"],
        )
        for n, p in student.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                p.requires_grad = True
                theta.append(p)
        print(f"[Init] LoRA q/v replaced: {rep} | theta params: {sum(p.numel() for p in theta):,}")
    else:
        for n, p in student.encoder.named_parameters():
            p.requires_grad = True
            theta.append(p)
        print(f"[Init] Full-FT encoder params: {sum(p.numel() for p in theta):,}")

    d_model = student.config.d_model
    L_soft_cfg = int(cfg["compressor"]["L_soft"])

    import torch.nn.functional as F

    logit_scale = torch.nn.Parameter(torch.tensor(0.0, device=device))  # 温度 = exp(logit_scale)
    
    # 创建 PATCH 参数 - GlobalSoftPatch
    patch = GlobalSoftPatch(L_soft_cfg, d_model, device=device).to(device)
    outer_params = list(patch.parameters())
    outer_params.append(logit_scale)  # ← 别漏
    
    opt_eta = torch.optim.AdamW(outer_params, lr=float(cfg["outer"]["lr"]))

    # 让所有 logits_from_ids 默认走“余弦+温度”，不用每次传参
    E.use_cosine_default = True
    E.logit_scale = logit_scale.data   # 挂一个 Tensor 引用即可
    E.normalize_patch_default = True   # ← 新增：评测端也会默认归一化补丁
    E.patch_norm_kind = cfg["compressor"].get("patch_norm", "l2")

    inner_opt = MomentumInner(theta, lr=cfg["mixflow"]["inner_lr"], momentum=cfg["mixflow"]["momentum"])

    from torch.func import functional_call

    # ---- 仅 encoder 的基线参数与 buffer（去掉 'encoder.' 前缀）----
    BASE_PARAMS_ENC, BASE_BUFFERS_ENC = {}, {}
    for n, p in student.named_parameters():
        if n.startswith("encoder."):
            BASE_PARAMS_ENC[n[len("encoder.") :]] = p
    for n, b in student.named_buffers():
        if n.startswith("encoder."):
            BASE_BUFFERS_ENC[n[len("encoder.") :]] = b

    theta_names_enc = []
    for n, p in student.named_parameters():
        if n.startswith("encoder.") and p.requires_grad:
            theta_names_enc.append(n[len("encoder.") :])

    # ---- 获取 PATCH 嵌入的函数 ----
    def get_patch_emb(recent_ids: torch.Tensor, mask_recent: torch.Tensor):
        """返回 PATCH 嵌入 [L_soft, d]，需要时归一化"""
        patch_emb = patch.phi  # [L_soft, d]
        # === 补丁归一化（建议开启）===
        norm_kind = cfg["compressor"].get("patch_norm", "l2")   # 'l2' | 'ln' | 'none'
        if norm_kind == "l2":
            patch_emb = F.normalize(patch_emb, dim=-1, eps=1e-6)
        elif norm_kind == "ln":
            patch_emb = (patch_emb - patch_emb.mean(dim=-1, keepdim=True)) / (patch_emb.std(dim=-1, keepdim=True) + 1e-6)
        return patch_emb

    # ---- inner_loss: 支持 patch=None（用于外循环验证） ----
    def inner_loss(theta_list, patch_emb, recent_ids, targets, mask_recent):
        """
        参数:
          patch_emb: None | [L_soft, d]
                    None 表示不使用 PATCH（外循环场景）
                    [L_soft, d] 表示使用 PATCH（内循环场景）
        """
        override = {n: t for n, t in zip(theta_names_enc, theta_list)}
        param_and_buffers = {**BASE_PARAMS_ENC, **override, **BASE_BUFFERS_ENC}

        B = recent_ids.size(0)
        emb_recent = E(recent_ids.to(device))  # [B, Lr, d]

        # 处理 PATCH（可能为 None）
        if patch_emb is None or patch_emb.numel() == 0:
            patch = emb_recent.new_zeros((B, 0, emb_recent.size(-1)))
        else:
            if patch_emb.dim() == 2:
                patch = patch_emb.unsqueeze(0).expand(B, -1, -1)  # [B, L_soft, d]
            elif patch_emb.dim() == 3:
                assert patch_emb.size(0) == B, "patch batch size mismatch"
                patch = patch_emb
            else:
                raise ValueError("patch_emb must be [L_soft,d] or [B,L_soft,d]")

        L_soft = patch.size(1)
        inputs = torch.cat([patch, emb_recent], dim=1)  # [B, L_soft+Lr, d]
        attn = torch.cat(
            [
                torch.ones((B, L_soft), dtype=torch.long, device=device),
                mask_recent.to(device).long(),
            ],
            dim=1,
        )

        # 临时关闭 checkpoint，以兼容 functorch.functional_call
        was_gc = getattr(student.encoder, "gradient_checkpointing", False)
        if was_gc:
            student.encoder.gradient_checkpointing_disable()

        try:
            with AmpAutocast(cfg["system"]["amp"]):
                enc_out = functional_call(
                    student.encoder,
                    param_and_buffers,
                    args=(),
                    kwargs=dict(inputs_embeds=inputs, attention_mask=attn, return_dict=True),
                )
                H = enc_out.last_hidden_state  # [B, L_soft+Lr, d]

                pool = cfg["head"]["pool"]  # 'last' or 'mean'
                
                if pool == 'last':
                    lengths = mask_recent.sum(dim=1)
                    idx_last = (lengths - 1).clamp_min(0)
                    pos = L_soft + idx_last
                    u = H[torch.arange(B, device=device), pos, :]
                elif pool == 'mean':
                    recent = H[:, L_soft:, :]  # [B, Lr, d]
                    denom = mask_recent.sum(dim=1).clamp_min(1).unsqueeze(1).to(device).float()
                    u = (recent * mask_recent.unsqueeze(-1).float().to(device)).sum(dim=1) / denom
                else:
                    raise ValueError("head.pool must be 'last' or 'mean'")
                
                # 计算 loss（cosine + 温度）
                u = F.normalize(u, dim=-1)
                tab = F.normalize(E.table, dim=-1)
                logits = (u @ tab[1:].T).float() * torch.exp(logit_scale)   # 温度 > 0
                loss = F.cross_entropy(logits, (targets.to(device) - 1))
            return loss
        finally:
            try:
                student.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                student.encoder.gradient_checkpointing_enable()

    grad_fn = get_fwdrev_grad_fn_eta(inner_loss)

    theta_param_names = [n for n, p in student.named_parameters() if p.requires_grad]
    best_ndcg = -1.0
    it = 0

    for epoch in range(cfg["train"]["epochs"]):
        for recent_ids, targets, mask_recent in dl_tr:
            it += 1

            #######################################
            if it == 1:  # 只跑一次，调试用
                print('临时检查')
                lens = mask_recent.sum(dim=1).cpu()
                print(f"[dbg lengths] min={lens.min().item()} median={lens.median().item()} max={lens.max().item()}")
                with torch.no_grad(), AmpAutocast(cfg["system"]["amp"]):
                    patch_emb_dbg = get_patch_emb(recent_ids, mask_recent)
                    B = recent_ids.size(0)
                    emb_recent = E(recent_ids.to(device))
                    patch_tensor = patch_emb_dbg.unsqueeze(0).expand(B, -1, -1)
                    inputs = torch.cat([patch_tensor, emb_recent], dim=1)
                    attn = torch.cat(
                        [torch.ones((B, patch_tensor.size(1)), dtype=torch.long, device=device),
                         mask_recent.to(device).long()],
                        dim=1,
                    )
                    out = student.encoder(inputs_embeds=inputs, attention_mask=attn, return_dict=True)
                    H = out.last_hidden_state
                    lengths = mask_recent.sum(dim=1)
                    pos = patch_tensor.size(1) + (lengths - 1).clamp_min(0)
                    u = H[torch.arange(B, device=device), pos, :]

                    from loss import ce_full_softmax
                    ce1, _ = ce_full_softmax(u, E.table, targets.to(device))

                    import torch.nn.functional as F
                    logits_dbg = (u @ E.table[1:].T).float()
                    ce2 = F.cross_entropy(logits_dbg, (targets - 1).to(device))

                    print(f"[dbg CE align] ce_full_softmax={ce1.item():.4f} | explicit_ce={ce2.item():.4f}")
            #######################################

            # ===== Inner K steps + Outer update =====
            w_state, m_state = inner_opt.snapshot()

            # 内循环：用 PATCH + 最近数据优化 encoder（K 步）
            patch_emb = get_patch_emb(recent_ids, mask_recent)
            for k in range(cfg["mixflow"]["inner_steps"]):
                gflat = grad_fn(theta, patch_emb, recent_ids, targets, mask_recent)
                inner_opt.step(gflat)

            # 外循环：用完整数据，但不用 PATCH（迫使 PATCH 学到长历史信息）
            try:
                full_ids, full_tgts, full_mask = next(iter_full)
            except StopIteration:
                iter_full = iter(dl_tr_full)
                full_ids, full_tgts, full_mask = next(iter_full)

            # ★ 关键改进：外循环传 None（无 PATCH），迫使 PATCH 补充缺失信息
            opt_eta.zero_grad(set_to_none=True)
            loss_outer = inner_loss(theta, None, full_ids, full_tgts, full_mask)
            loss_outer.backward()
            torch.nn.utils.clip_grad_norm_(outer_params, 1.0)
            opt_eta.step()

            inner_opt.restore(w_state, m_state)

            if it % cfg["train"]["log_every"] == 0:
                peak = cuda_mem_gb(model=student, device_str=cfg["system"]["device"], kind="alloc")
                print(
                    f"[it {it:06d}] loss_outer={loss_outer.item():.4f} "
                    f"| temp={torch.exp(logit_scale).item():.2f} "
                    f"| max CUDA(GB)={peak:.3f}"
                )

            # ===== 在线 CRE（不更新 PATCH，只重训 student）=====
            from utils import pack_theta_state
            if cfg["train"]["eval_every_steps"] and (it % cfg["train"]["eval_every_steps"] == 0):
                metrics = run_eval_online_cre(E, patch, dl_tr, dl_va, cfg)
                ndcg20 = metrics.get("NDCG@20", 0.0)
                print(f"[eval it {it}] " + " ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

                # 保存 best
                if cfg["train"]["save_best"] and ndcg20 > best_ndcg:
                    best_ndcg = ndcg20
                    os.makedirs("artifacts", exist_ok=True)
                    state = {
                        "theta": pack_theta_state(student, theta_param_names),
                        "item_table": E.state_dict(),
                        "cfg": cfg,
                        "it": it,
                        "metrics": metrics,
                        "logit_scale": logit_scale.detach().cpu(),
                        "patch": patch.state_dict(),
                    }
                    torch.save(state, os.path.join("artifacts", "best.pt"))
                    print(f"[save] best checkpoint at it={it} (NDCG@20={ndcg20:.4f})")

    # ----- save final -----
    dsname = cfg["data"]["name"]
    artifacts_dir = os.path.join("artifacts", dsname)
    os.makedirs(artifacts_dir, exist_ok=True)
    final_state = {
        "item_table": E.state_dict(),
        "cfg": cfg,
        "logit_scale": logit_scale.detach().cpu(),
        "patch": patch.state_dict(),
    }
    num = cfg["system"].get("ckpt_num", 0)
    ckpt = os.path.join(artifacts_dir, f"checkpoint_{num}.pt")
    torch.save(final_state, ckpt)
    print(f"Saved {ckpt}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(os.path.dirname(__file__), cfg_path)
    
    cfg = yaml.safe_load(open(cfg_path))
    train(cfg)
