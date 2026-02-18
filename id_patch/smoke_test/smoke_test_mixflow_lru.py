"""MixFlow 2nd-order gradient smoke test for LRU backbone."""

from dataclasses import dataclass
from typing import List, Tuple
import os
import sys
try:
    import torch
    import torch.nn.functional as F
    from torch.func import functional_call
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch not found. Activate your training environment first, e.g.:\n"
        "  conda activate <your_env>\n"
        "Then re-run this script."
    ) from exc
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from core.mixflow import get_fwdrev_grad_fn_eta, MomentumInner
from backbones.LRU import LRU


@dataclass
class TinyConfig:
    # backbone
    hidden_units: int = 16
    num_blocks: int = 2
    dropout_rate: float = 0.1
    lru_num_blocks: int = 2
    lru_dropout: float = 0.1
    lru_attn_dropout: float = 0.1

    # patch/gating
    num_patches: int = 2
    patch_len: int = 2
    use_gating: bool = True
    gating_hidden_dim: int = 8
    patch_init_std: float = 0.01
    gating_init_std: float = 0.01
    gating_pool: str = "last"
    gating_temperature: float = 1.0
    gating_noise_std: float = 0.0
    patch_routing: str = "learned"

    # head
    enable_projection_head: bool = True
    head_use_gelu: bool = False
    head_use_ln: bool = True
    head_residual: bool = True

    # misc
    max_seq_length: int = 8


def _collect_params_and_buffers(module: torch.nn.Module):
    params = {n: p for n, p in module.named_parameters()}
    buffers = {n: b for n, b in module.named_buffers()}
    return params, buffers


def _select_theta_names(
    model: torch.nn.Module,
    enable_bias_ln: bool = True,
    enable_head: bool = True,
    exclude_complex: bool = True,
) -> List[str]:
    trainable = set()
    if enable_bias_ln:
        for name, p in model.named_parameters():
            if name.endswith(".bias") and (not exclude_complex or not p.is_complex()):
                trainable.add(name)
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                for param_name, p in module.named_parameters(recurse=False):
                    if exclude_complex and p.is_complex():
                        continue
                    full = f"{module_name}.{param_name}" if module_name else param_name
                    trainable.add(full)
    if enable_head and hasattr(model, "proj_linear"):
        for name, p in model.proj_linear.named_parameters(prefix="proj_linear"):
            if exclude_complex and p.is_complex():
                continue
            trainable.add(name)
    if enable_head and hasattr(model, "proj_ln"):
        for name, p in model.proj_ln.named_parameters(prefix="proj_ln"):
            if exclude_complex and p.is_complex():
                continue
            trainable.add(name)
    ordered = [name for name, _ in model.named_parameters() if name in trainable]
    return ordered


def _make_batch(batch_size: int, seq_len: int, item_num: int, device: torch.device):
    # random ids in [2, item_num+1], reserve 0 for padding
    input_ids = torch.randint(2, item_num + 2, (batch_size, seq_len), device=device)
    # random padding
    pad_mask = torch.rand(batch_size, seq_len, device=device) < 0.2
    input_ids = input_ids.masked_fill(pad_mask, 0)

    # next-item targets (simple shift), pad at end
    pos_ids = torch.roll(input_ids, shifts=-1, dims=1)
    pos_ids[:, -1] = 0

    # negatives sampled uniformly
    neg_ids = torch.randint(2, item_num + 2, (batch_size, seq_len), device=device)
    neg_ids = neg_ids.masked_fill(pos_ids == 0, 0)

    # ensure at least one valid position
    if (pos_ids != 0).sum().item() == 0:
        pos_ids[0, 0] = 2
        neg_ids[0, 0] = 3
    return input_ids, pos_ids, neg_ids


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")

    config = TinyConfig()
    item_num = 50
    model = LRU(config, item_num=item_num).to(device)

    base_params, base_buffers = _collect_params_and_buffers(model)
    theta_names = _select_theta_names(model, enable_bias_ln=True, enable_head=True)
    theta_list = [base_params[n].detach().clone().requires_grad_(True) for n in theta_names]

    eta = model.meta_patch.eta
    eta.requires_grad_(True)

    input_ids, pos_ids, neg_ids = _make_batch(batch_size=4, seq_len=6, item_num=item_num, device=device)

    def inner_loss(theta_list_local, eta_local, input_ids_local, pos_ids_local, neg_ids_local):
        override = {n: t for n, t in zip(theta_names, theta_list_local)}
        params = {**base_params, **override, **base_buffers}
        pos_logits, neg_logits = functional_call(
            model,
            params,
            args=(),
            kwargs={
                "input_ids": input_ids_local,
                "pos_ids": pos_ids_local,
                "neg_ids": neg_ids_local,
                "patch_params": eta_local,
                "use_patch": True,
            },
        )
        valid = pos_ids_local != 0
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits), reduction="none")
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits), reduction="none")
        loss = pos_loss + neg_loss
        if valid.any():
            return loss[valid].mean()
        return loss.mean()

    fwdrev = get_fwdrev_grad_fn_eta(inner_loss)

    grad_flat = fwdrev(theta_list, eta, input_ids, pos_ids, neg_ids)
    inner_opt = MomentumInner(theta_list, lr=0.1, momentum=0.0)
    inner_opt.step(grad_flat)

    outer_loss = inner_loss(theta_list, eta, input_ids, pos_ids, neg_ids)
    outer_loss.backward()

    print("Smoke test OK")
    print("outer_loss:", outer_loss.detach().item())
    print("eta grad norm:", eta.grad.norm().detach().item())


if __name__ == "__main__":
    main()
