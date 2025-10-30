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

def cuda_mem_gb(model=None, device_str=None, kind="alloc", reset=False):
    if not torch.cuda.is_available(): return 0.0
    device = torch.device(device_str) if device_str else (
        next(model.parameters()).device if model is not None
        else torch.device(f"cuda:{torch.cuda.current_device()}")
    )
    if reset: torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    if kind == "alloc":
        bytes_ = torch.cuda.max_memory_allocated(device)
    elif kind == "reserved":
        bytes_ = torch.cuda.max_memory_reserved(device)
    else:
        bytes_ = torch.cuda.memory_allocated(device)
    return bytes_ / (1024**3)

def pack_theta_state(student, theta_param_names):
    state = {}
    sd = student.state_dict()
    for n in theta_param_names:
        state[n] = sd[n].clone()
    return state