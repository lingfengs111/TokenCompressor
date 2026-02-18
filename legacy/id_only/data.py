
"""Text-based sequence dataset for ID-only training."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from utils import setup_logger

logger = setup_logger("id-only-data")

class TxtSeqDataset(Dataset):
    """
    Read data/<name>/proc/data.txt.

    Each line: user item1 item2 ... itemN
    Rules:
      - Only keep sequences with >= 2 items.
      - train: recent = seq[:-2], target = seq[-2]
      - val/test: recent = seq[:-1], target = seq[-1]
    """

    def __init__(self, txt_path: str | Path, stage: str, L_real: int):
        self.stage = stage
        self.L_real = L_real
        self.rows: List[List[int]] = []

        txt_path = Path(txt_path)
        logger.info("Loading %s dataset from %s", stage, txt_path)

        with txt_path.open("r") as f:
            for line_idx, line in enumerate(f, start=1):
                if line_idx % 10000 == 0:
                    logger.info("Loaded %s lines...", line_idx)

                parts = [int(x) for x in line.strip().split()]
                if len(parts) < 3:
                    continue
                items = parts[1:]
                if len(items) < 2:
                    continue
                self.rows.append(items)

        logger.info("Loaded %s dataset: %s sequences", stage, len(self.rows))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.rows[idx]
        if self.stage == "train":
            recent = seq[:-2]
            target = seq[-2]
        else:
            recent = seq[:-1]
            target = seq[-1]
        recent = recent[-self.L_real :]
        return torch.tensor(recent, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def make_dataloaders_from_txt(
    proc_dir: str | Path,
    L_real: int,
    batch_size: int,
    num_workers: int = 0,
):
    txt_path = Path(proc_dir) / "data.txt"
    ds_tr = TxtSeqDataset(txt_path, "train", L_real)
    ds_va = TxtSeqDataset(txt_path, "val", L_real)
    ds_te = TxtSeqDataset(txt_path, "test", L_real)

    def collate(batch):
        recents = [b[0] for b in batch]
        targets = torch.stack([b[1] for b in batch])
        recents_padded = pad_sequence(recents, batch_first=True, padding_value=0)
        mask_recent = (recents_padded != 0).long()
        return recents_padded, targets, mask_recent

    return (
        DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers, drop_last=False),
        DataLoader(ds_va, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers, drop_last=False),
        DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers, drop_last=False),
    )
