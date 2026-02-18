from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import yaml


def setup_logger(
    name: str = "id-only",
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_dir: str = "logs",
) -> logging.Logger:
    try:
        import torch.multiprocessing as mp

        is_main_process = mp.current_process().name == "MainProcess"
    except ImportError:
        is_main_process = True

    if not is_main_process and log_to_file:
        log_to_file = False

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = Path(log_dir) / f"log-{timestamp}.txt"
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a dict, got {type(cfg)}")
    return cfg


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AmpAutocast:
    def __init__(self, amp_dtype: str = "bf16"):
        self.amp_dtype = amp_dtype
        self.ctx = None

    def __enter__(self):
        if self.amp_dtype in ("bf16", "fp16") and torch.cuda.is_available():
            dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
            self.ctx = torch.amp.autocast("cuda", dtype=dtype)
            self.ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.ctx:
            self.ctx.__exit__(exc_type, exc_value, traceback)


def cuda_mem_gb(model=None, device_str: str | None = None, kind: str = "alloc", reset: bool = False) -> float:
    if not torch.cuda.is_available():
        return 0.0
    device = torch.device(device_str) if device_str else (
        next(model.parameters()).device if model is not None else torch.device(f"cuda:{torch.cuda.current_device()}")
    )
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    if kind == "alloc":
        bytes_ = torch.cuda.max_memory_allocated(device)
    elif kind == "reserved":
        bytes_ = torch.cuda.max_memory_reserved(device)
    else:
        bytes_ = torch.cuda.memory_allocated(device)
    return bytes_ / (1024 ** 3)


def pack_theta_state(student, theta_param_names: Iterable[str]) -> Dict[str, torch.Tensor]:
    state = {}
    sd = student.state_dict()
    for name in theta_param_names:
        state[name] = sd[name].clone()
    return state
