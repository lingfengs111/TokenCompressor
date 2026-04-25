"""Protocol-aware sequential dataset utilities for ID-only training."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from core.streaming_eval import resolve_train_cutoff


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class LooDatasetSpec:
    name: str
    root: Path

    @property
    def data_txt(self) -> Path:
        return self.root / "data.txt"

    @property
    def stats_json(self) -> Path:
        return self.root / "stats.json"

    @property
    def item2idx_json(self) -> Path:
        return self.root / "item2idx.json"

    @property
    def user2idx_json(self) -> Path:
        return self.root / "user2idx.json"


LOO_DATASETS = {
    "taobao_loo202": LooDatasetSpec(
        name="taobao_loo202",
        root=REPO_ROOT / "data" / "taobao" / "loo_202",
    ),
    "ml10m_loo202": LooDatasetSpec(
        name="ml10m_loo202",
        root=REPO_ROOT / "data" / "movielens" / "ml-10m" / "loo_202",
    ),
    "xlong_loo402": LooDatasetSpec(
        name="xlong_loo402",
        root=REPO_ROOT / "data" / "xlong2018" / "loo_402",
    ),
}

LOO_DATASET_ALIASES = {
    "taobao": "taobao_loo202",
    "ml-10m": "ml10m_loo202",
    "ml10m": "ml10m_loo202",
    "movielens-10m": "ml10m_loo202",
    "xlong": "xlong_loo402",
    "xlong2018": "xlong_loo402",
    "xlong-2018": "xlong_loo402",
}


def resolve_loo_dataset(dataset: str, data_dir: Optional[str] = None) -> LooDatasetSpec:
    if data_dir:
        root = Path(data_dir).expanduser()
        name = dataset or root.name
        spec = LooDatasetSpec(name=name, root=root)
    else:
        key = str(dataset).strip()
        key_lower = key.lower()
        if key in LOO_DATASETS:
            spec = LOO_DATASETS[key]
        elif key_lower in LOO_DATASETS:
            spec = LOO_DATASETS[key_lower]
        else:
            alias_key = LOO_DATASET_ALIASES.get(key_lower)
            if alias_key and alias_key in LOO_DATASETS:
                spec = LOO_DATASETS[alias_key]
            else:
                path = Path(key).expanduser()
                if path.exists():
                    spec = LooDatasetSpec(name=path.name, root=path)
                else:
                    available = ", ".join(sorted(LOO_DATASETS.keys()))
                    raise ValueError(
                        f"Unknown dataset '{dataset}'. Available: {available} "
                        "or pass data_dir/path to a LOO dataset folder."
                    )

    if not spec.data_txt.exists():
        raise FileNotFoundError(f"Missing data.txt in {spec.root}")
    return spec


def infer_loo_min_len(spec: LooDatasetSpec) -> Optional[int]:
    candidates = [spec.name, spec.root.name]
    for text in candidates:
        match = re.search(r"loo[_-]?(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    if spec.stats_json.exists():
        try:
            with spec.stats_json.open("r", encoding="utf-8") as f:
                stats = json.load(f)
            min_len = stats.get("min_len")
            if isinstance(min_len, int) and min_len > 0:
                return min_len
        except Exception:
            return None
    return None


class LooSequenceDataset(Dataset):
    """Sequence dataset from data.txt with protocol-aware train/test holdout."""

    def __init__(
        self,
        data_txt_path: str | Path,
        config,
        min_items: int = 2,
        id_offset: int = 1,
        logger=None,
    ):
        self.config = config
        self.max_seq_length = getattr(config, "max_seq_length", 0)
        self.drop_unseen_items = bool(getattr(config, "drop_unseen_items", True))
        self.id_offset = int(id_offset)
        self.min_item_id = 1 + self.id_offset
        self.unk_idx = 1 if self.id_offset > 0 else 0
        self.neg_item_by_user = {}

        self.user_seq = {}
        self.users = []
        self.internal_to_user_id = {}

        internal_id = 0
        max_train_len = 0
        max_item_raw = 0
        dropped_short = 0
        total_lines = 0

        data_txt_path = Path(data_txt_path)

        with data_txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    dropped_short += 1
                    continue
                user_id = int(parts[0])
                items_raw = [int(x) for x in parts[1:]]
                if len(items_raw) < min_items:
                    dropped_short += 1
                    continue
                if items_raw:
                    max_item_raw = max(max_item_raw, max(items_raw))
                if self.id_offset:
                    items = [x + self.id_offset for x in items_raw]
                else:
                    items = items_raw

                self.user_seq[internal_id] = items
                self.users.append(internal_id)
                self.internal_to_user_id[internal_id] = user_id
                train_len = resolve_train_cutoff(
                    len(items),
                    eval_protocol=getattr(config, "eval_protocol", "legacy_loo"),
                    last_k_eval_test=int(getattr(config, "last_k_eval_test", 0) or 0),
                )
                if train_len > max_train_len:
                    max_train_len = train_len
                internal_id += 1

        self.num_users = len(self.users)
        self.num_items = max_item_raw
        self.max_item = (self.num_items + self.id_offset) if self.num_items > 0 else 0
        self.max_train_seq_len = max_train_len

        if logger is not None:
            avg_seq_len = (
                np.mean([len(self.user_seq[u]) for u in self.users]) if self.users else 0.0
            )
            logger.info(
                "Loaded %s sequences (%s raw lines). Dropped %s short sequences.",
                f"{self.num_users:,}",
                f"{total_lines:,}",
                f"{dropped_short:,}",
            )
            logger.info("Average sequence length: %.2f", avg_seq_len)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        user = self.users[idx]
        seq = self.user_seq[user]
        return user, seq
