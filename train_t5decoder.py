import os, json, yaml, torch
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from utils import set_seed, AmpAutocast, max_cuda_gb
from model_id_t5 import ItemTable, GlobalSoftPatch, build_student, apply_lora_qv, logits_from_ids
from loss import ce_full_softmax
from mixflow import get_fwdrev_grad_fn_eta, MomentumInner


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
    dl_tr, dl_va, dl_te = make_dataloaders_from_txt(proc_dir, cfg["data"]["L_real"], cfg["train"]["batch_size"])


    # modules
    E = ItemTable(cfg["items"]["num_items"], cfg["items"]["d_model"], trainable=cfg["items"]["trainable"]).to(device)
    student = build_student(cfg["student"]["t5_name"], device, cfg["student"]["grad_ckpt"])
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

    # 1) 记录“基线参数/缓冲”
    BASE_PARAMS  = dict(student.named_parameters())
    BASE_BUFFERS = dict(student.named_buffers())

    # 2) 记录 θ 的名字顺序（必须与打包/解包 theta_list 的顺序完全一致）
    #    你在替换 LoRA 时收集这些名字；示例：
    theta_names = []
    for n, p in student.named_parameters():
        if p.requires_grad:   # 这里恰好是你设为可训的那些 LoRA 权重
            theta_names.append(n)
            
    from torch.func import functional_call

    def inner_loss(theta_list, eta_tensor, recent_ids, targets, mask_recent):
        override = {n: t for n, t in zip(theta_names, theta_list)}
        param_and_buffers = {**BASE_PARAMS, **override, **BASE_BUFFERS}

        B = recent_ids.size(0)
        L_soft = cfg["compressor"]["L_soft"]
        emb_recent = E(recent_ids.to(device))
        patch = eta_tensor.unsqueeze(0).expand(B, -1, -1)
        inputs = torch.cat([patch, emb_recent], dim=1)
        attn = torch.cat([
            torch.ones((B, L_soft), dtype=torch.long, device=device),
            mask_recent.to(device).long()
        ], dim=1)

        # 造一个假的 decoder input（防止 T5 报错）
        dec_ids = torch.zeros((B, 1), dtype=torch.long, device=device)

        with AmpAutocast(cfg["system"]["amp"]):
            out = functional_call(
                student,
                param_and_buffers,
                args=(),
                kwargs=dict(
                    inputs_embeds=inputs,
                    attention_mask=attn,
                    decoder_input_ids=dec_ids,
                    return_dict=True
                ),
                tie_weights=True,
                strict=False
            )
            H = out.encoder_last_hidden_state
            lengths = mask_recent.sum(dim=1)
            idx_last = (lengths - 1).clamp_min(0)
            pos = L_soft + idx_last
            u = H[torch.arange(B, device=device), pos, :]
            loss, _ = ce_full_softmax(u, E.table, targets.to(device))
        return loss

    grad_fn = get_fwdrev_grad_fn_eta(inner_loss)

    it=0
    for epoch in range(cfg["train"]["epochs"]):
        for recent_ids, targets, mask_recent in dl_tr:
            it += 1
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
                print(f"[it {it:06d}] loss_outer={loss_outer.item():.4f} | max CUDA(GB)={max_cuda_gb():.3f}")
    
    # save
    os.makedirs("artifacts", exist_ok=True)
    torch.save({"phi":phi.state_dict(),"item_table":E.state_dict(),"cfg":cfg}, os.path.join(path, "artifacts","ckpt.pt"))
    print("Saved artifacts/ckpt.pt")


if __name__=='__main__':
    # cfg = DEFAULT_CFG
    path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if os.path.exists(path):
        cfg = yaml.safe_load(open(path))
    train(cfg)
