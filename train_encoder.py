import os, json, yaml, torch
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from utils import set_seed, AmpAutocast, cuda_mem_gb
from model_id_t5 import ItemTable, GlobalSoftPatch, build_student, apply_lora_qv
from loss import ce_full_softmax
from mixflow import get_fwdrev_grad_fn_eta, MomentumInner
from eval import run_eval_online_cre

def train(cfg:dict):
    path = os.path.dirname(__file__)
    set_seed(cfg["train"]["seed"]); device=cfg["system"]["device"]
    
    # data
    dsname = cfg["data"]["name"]
    proc_dir = os.path.join(path, "data", dsname, "proc")

    # 直接从 item2idx.json 读物品数
    num_items = len(json.load(open(os.path.join(proc_dir, 'item2idx.json'))))
    if cfg["items"]["num_items"] is None:
        cfg["items"]["num_items"] = num_items

    # 用 train.txt 的 dataloader
    from data import make_dataloaders_from_txt

    # dl_tr, dl_va, dl_te = make_dataloaders_from_txt(proc_dir, cfg["data"]["L_real"], cfg["train"]["batch_size"])
    # 训练/在线CRE：用更长的训练loader
    L_real_train = cfg["data"].get("L_real_train", cfg["data"]["L_real"])
    dl_tr, dl_va, _ = make_dataloaders_from_txt(proc_dir, L_real_train, cfg["train"]["batch_size"])

    # modules
    E = ItemTable(cfg["items"]["num_items"], cfg["items"]["d_model"], trainable=cfg["items"]["trainable"]).to(device)

    # 可选：用文本初始化 E
    if cfg["items"]["init_from_text"]:
        txt_path = os.path.join(path, 'data', dsname, cfg["items"]["init_path"])
        E.table.data.copy_(torch.load(txt_path, map_location="cpu"))
        print("[Init] E from text:", txt_path)
        for p in E.parameters():
            p.requires_grad = cfg["items"]["trainable"]  # false -> 冻结
    

    student = build_student(cfg["student"]["t5_name"], device, cfg["student"]["grad_ckpt"])
   
    student.config.use_cache = False                # 训练/二阶必须关 cache
    # 只给 encoder 开启非重入 checkpoint（兼容 functorch）
    try:
        student.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        # 你的 transformers 版本太老不支持 kwargs；见“备选方案 B”
        print("[Warn] transformers太旧，不支持use_reentrant=False，将采用备选方案B")
    

    d_model = student.config.d_model
    # scorer = EncoderOnlyScorer(d_model, pool=cfg["head"]["pool"]).to(device)
    for p in student.parameters(): p.requires_grad=False

    theta=[]
    if not cfg["mixflow"]["full_ft"]:
        rep = apply_lora_qv(student, cfg["mixflow"]["lora_rank"], cfg["mixflow"]["lora_alpha"], cfg["mixflow"]["target_blocks"], cfg["mixflow"]["dropout"])
        for n,p in student.named_parameters():
            if "lora_A" in n or "lora_B" in n: p.requires_grad=True; theta.append(p)
        print(f"[Init] LoRA q/v replaced: {rep} | theta params: {sum(p.numel() for p in theta):,}")
    else:
        for n,p in student.encoder.named_parameters(): p.requires_grad=True; theta.append(p)
        print(f"[Init] Full-FT encoder params: {sum(p.numel() for p in theta):,}")

    phi = GlobalSoftPatch(cfg["compressor"]["L_soft"], d_model, device=device).to(device)
    opt_eta = torch.optim.AdamW(phi.parameters(), lr=cfg["outer"]["lr"])
    inner_opt = MomentumInner(theta, lr=cfg["mixflow"]["inner_lr"], momentum=cfg["mixflow"]["momentum"])

    from torch.func import functional_call

    # ==== 构建“仅 encoder 用”的基线参数与 buffer（去掉 'encoder.' 前缀；过滤 shared.*） ====
    BASE_PARAMS_ENC  = {}
    BASE_BUFFERS_ENC = {}
    for n, p in student.named_parameters():
        if n.startswith("encoder."):
            BASE_PARAMS_ENC[n[len("encoder."):]] = p
    for n, b in student.named_buffers():
        if n.startswith("encoder."):
            BASE_BUFFERS_ENC[n[len("encoder."):]] = b

    # 只收集 encoder 里可训练（LoRA）那部分 θ 的名字，且去掉 'encoder.' 前缀
    theta_names_enc = []
    for n, p in student.named_parameters():
        if n.startswith("encoder.") and p.requires_grad:
            theta_names_enc.append(n[len("encoder."):])

    def inner_loss(theta_list, eta_tensor, recent_ids, targets, mask_recent):
        # 覆盖 LoRA θ：名字已是“encoder 相对路径”
        override = {n: t for n, t in zip(theta_names_enc, theta_list)}
        param_and_buffers = {**BASE_PARAMS_ENC, **override, **BASE_BUFFERS_ENC}

        B = recent_ids.size(0)
        L_soft = cfg["compressor"]["L_soft"]
        emb_recent = E(recent_ids.to(device))                    # [B, Lr, d]
        patch = eta_tensor.unsqueeze(0).expand(B, -1, -1)        # [B, L_soft, d]
        inputs = torch.cat([patch, emb_recent], dim=1)           # [B, L_soft+Lr, d]
        attn = torch.cat([
            torch.ones((B, L_soft), dtype=torch.long, device=device),
            mask_recent.to(device).long()
        ], dim=1)
        

        # === 2) 关键：临时关闭 encoder 的 gradient checkpointing ===
        was_gc = getattr(student.encoder, "gradient_checkpointing", False)
        if was_gc:
            student.encoder.gradient_checkpointing_disable()

        try:
            with AmpAutocast(cfg["system"]["amp"]):
                enc_out = functional_call(
                    student.encoder,            # ✅ 只跑 encoder
                    param_and_buffers,          # ✅ 仅 encoder 的 param/buffer
                    args=(),
                    kwargs=dict(inputs_embeds=inputs, attention_mask=attn, return_dict=True)
                )
                H = enc_out.last_hidden_state   # [B, L_soft+Lr, d]

                # pool='last'
                lengths = mask_recent.sum(dim=1)
                idx_last = (lengths - 1).clamp_min(0)
                pos = L_soft + idx_last
                u = H[torch.arange(B, device=device), pos, :]

                # 全库 CE（E 冻结没问题）
                loss, _ = ce_full_softmax(u, E.table, targets.to(device))
        
            return loss 
        finally:
            # 若你之前用的是非重入：
            try:
                student.encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                # 老版本 transformers 没有 kwargs，就启用默认重入版本
                student.encoder.gradient_checkpointing_enable()


    grad_fn = get_fwdrev_grad_fn_eta(inner_loss)


    theta_param_names = [n for n,p in student.named_parameters() if p.requires_grad]
    best_ndcg = -1.0
    it=0
    for epoch in range(cfg["train"]["epochs"]):
        for recent_ids, targets, mask_recent in dl_tr:
            it += 1
            # === 内层 K 步 + 外层更新（你的原逻辑）===
            w_state, m_state = inner_opt.snapshot()
            for k in range(cfg["mixflow"]["inner_steps"]):
                gflat = grad_fn(theta, phi.phi, recent_ids, targets, mask_recent)
                inner_opt.step(gflat)
            opt_eta.zero_grad(set_to_none=True)
            loss_outer = inner_loss(theta, phi.phi, recent_ids, targets, mask_recent)
            loss_outer.backward()
            torch.nn.utils.clip_grad_norm_(phi.parameters(), 1.0)
            opt_eta.step()
            inner_opt.restore(w_state, m_state)
            if it % cfg["train"]["log_every"]==0:
                peak = cuda_mem_gb(model=student, device_str=cfg["system"]["device"], kind="alloc")
                print(f"[it {it:06d}] loss_outer={loss_outer.item():.4f} | max CUDA(GB)={peak:.3f}")


            # === 周期性评测 ===
            from utils import pack_theta_state
            if cfg["train"]["eval_every_steps"] and (it % cfg["train"]["eval_every_steps"] == 0):
                metrics = run_eval_online_cre(E, phi, dl_tr, dl_va, cfg)
                ndcg20 = metrics.get("NDCG@20", 0.0)
                print(f"[eval it {it}] " + " ".join([f"{k}={v:.4f}" for k,v in metrics.items()]))

                # 保存 best
                if cfg["train"]["save_best"] and ndcg20 > best_ndcg:
                    best_ndcg = ndcg20
                    os.makedirs("artifacts", exist_ok=True)
                    torch.save({
                        "phi": phi.state_dict(),
                        "theta": pack_theta_state(student, theta_param_names),  # 见下
                        "item_table": E.state_dict(),
                        "cfg": cfg,
                        "it": it,
                        "metrics": metrics,
                    }, os.path.join("artifacts", "best.pt"))
                    print(f"[save] best checkpoint at it={it} (NDCG@20={ndcg20:.4f})")
    # save
    os.makedirs("artifacts", exist_ok=True)
    torch.save({"phi":phi.state_dict(),"item_table":E.state_dict(),"cfg":cfg}, os.path.join(path, "artifacts","ckpt4.pt"))
    print("Saved artifacts/ckpt4.pt")


if __name__=='__main__':
    # cfg = DEFAULT_CFG
    path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if os.path.exists(path):
        cfg = yaml.safe_load(open(path))
    train(cfg)
