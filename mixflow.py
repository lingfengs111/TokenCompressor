
"""
mixflow.py
----------
Core primitives for MixFlow with eta-aware meta-gradients,
including:
  - get_fwdrev_grad_fn_eta(inner_loss_fn): builds a custom autograd op that
    returns grad_theta L and, on backward, propagates both:
      * H_{θθ} · v   via JVP on grad (efficient HVP)
      * H_{θη} · v   via VJP of <gradθ, v> onto eta (efficient MVP)
  - MomentumInner: simple in-graph SGD + momentum state for inner loop.

Usage sketch:
-------------
# Define inner loss: loss(theta_list, eta, *rest) -> scalar
def inner_loss(theta_list, eta, batch):
    ...
    return loss

fwdrev = get_fwdrev_grad_fn_eta(inner_loss)
theta = [p1, p2, ...]  # tensors requiring grad (LoRA or full FT)
eta   = nn.Parameter(...)  # outer variable (soft patch)
grad_flat = fwdrev(theta, eta, batch)  # returns flattened grad wrt theta
# update theta with MomentumInner; then do outer backward on eta.
"""

from __future__ import annotations
from typing import List, Tuple
import torch

def _flatten(params: List[torch.Tensor]):
    flat = torch.cat([p.reshape(-1) for p in params])
    shapes = [p.shape for p in params]
    return flat, shapes

def _unflatten(flat: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
    out = []; idx = 0
    for s in shapes:
        n = 1
        for d in s: n *= d
        out.append(flat[idx:idx+n].view(s))
        idx += n
    return out

def get_fwdrev_grad_fn_eta(inner_loss_fn):
    """
    Build a function G(theta_list, eta, *rest) -> flat_grad_theta
    whose backward implements:
      - grad wrt theta_flat input: H_{θθ} · v  (efficient HVP via JVP)
      - grad wrt eta:             H_{θη} · v  (efficient MVP via VJP)
    Notes:
      * Only tensors are saved via save_for_backward; python objects go to ctx.rest.
      * inner_loss_fn must be pure in inputs and return a scalar loss.
      * Before calling, we set inner_loss_fn.theta_template for shape bookkeeping.
    """
    class FwdRev(torch.autograd.Function):
        @staticmethod
        def forward(ctx, theta_flat: torch.Tensor, eta: torch.Tensor, *rest_tensors: torch.Tensor):
            # save only tensors; keep pythonic 'rest' on ctx (already tensors here)
            ctx.save_for_backward(theta_flat, eta, *rest_tensors)
            # unflatten theta by template stored on the function
            shapes = FwdRev.theta_shapes
            theta = _unflatten(theta_flat, shapes)
            # compute grad wrt theta (structured list)
            grads = torch.func.grad(inner_loss_fn, argnums=0)(theta, eta, *rest_tensors)
            # stash meta for backward
            ctx.theta_shapes = shapes
            ctx.n_rest = len(rest_tensors)
            return torch.cat([g.reshape(-1) for g in grads])

        @staticmethod
        def backward(ctx, ct: torch.Tensor):
            saved = ctx.saved_tensors
            theta_flat, eta = saved[0], saved[1]
            rest_tensors = saved[2:2+ctx.n_rest]
            shapes = ctx.theta_shapes
            theta = _unflatten(theta_flat, shapes)

            # --- efficient HVP: JVP on grad_theta (wrt theta) ---
            def grad_theta_only(th_list):
                return torch.func.grad(inner_loss_fn, argnums=0)(th_list, eta, *rest_tensors)

            # split ct to match theta
            split = _unflatten(ct, shapes)
            _, hvp_theta_list = torch.func.jvp(grad_theta_only, (tuple(theta),), (tuple(split),))
            flat_hvp_theta = torch.cat([h.reshape(-1) for h in hvp_theta_list])

            # --- efficient MVP wrt eta: VJP of <grad_theta, v> onto eta ---
            with torch.enable_grad():
                theta_req = [t.detach().requires_grad_(True) for t in theta]
                grads = torch.func.grad(inner_loss_fn, argnums=0)(theta_req, eta, *rest_tensors)
                grad_theta_flat = torch.cat([g.reshape(-1) for g in grads])
                hvp_eta = torch.autograd.grad(grad_theta_flat, eta, grad_outputs=ct, retain_graph=False, allow_unused=False)[0]

            grad_rest = tuple([None for _ in rest_tensors])
            # Return grads for inputs of forward in order:
            # (theta_flat, eta, *rest_tensors)
            return flat_hvp_theta, hvp_eta, *grad_rest

    def wrapped(theta_list: List[torch.Tensor], eta: torch.Tensor, *rest_tensors: torch.Tensor):
        flat, shapes = _flatten(theta_list)
        # stash shapes on class (no autograd content)
        FwdRev.theta_shapes = shapes
        return FwdRev.apply(flat, eta, *rest_tensors)

    return wrapped

class MomentumInner:
    """
    Simple SGD+momentum optimizer for the inner loop (stateful).
    Keeps velocity buffers aligned with params list.
    """
    def __init__(self, params: List[torch.Tensor], lr: float, momentum: float):
        self.params = params
        self.lr = lr
        self.mom = momentum
        self.m = [torch.zeros_like(p) for p in params]

    def step(self, grad_flat: torch.Tensor):
        idx = 0
        for i, p in enumerate(self.params):
            n = p.numel()
            g = grad_flat[idx:idx+n].view_as(p)
            self.m[i] = self.mom * self.m[i] + g
            with torch.no_grad():
                p.add_(-self.lr * self.m[i])
            idx += n

    def snapshot(self):
        w = [p.detach().clone() for p in self.params]
        m = [mi.detach().clone() for mi in self.m]
        return w, m

    def restore(self, w_state, m_state):
        for p, s in zip(self.params, w_state):
            p.data.copy_(s)
        for i in range(len(self.m)):
            self.m[i].data.copy_(m_state[i])
