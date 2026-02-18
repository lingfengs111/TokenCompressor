"""Data loading for ID-based MixFlow training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataConfig:
    data_path: str
    max_seq_len: int = 100
    batch_size: int = 128
    num_workers: int = 0
    item_id_offset: int = 0


def _resolve_split_key(npz: np.lib.npyio.NpzFile, candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in npz:
            return key
    return None


def _read_npz_sequences(npz_path: str, split: str) -> List[List[int]]:
    npz = np.load(npz_path, allow_pickle=True)
    split_candidates: Dict[str, Sequence[str]] = {
        "train": ("train", "train_seqs", "train_sequences", "train_seq"),
        "val": ("val", "valid", "validation", "val_seqs", "valid_seqs"),
        "test": ("test", "test_seqs", "test_sequences", "test_seq"),
    }
    key = _resolve_split_key(npz, split_candidates[split])
    if key is None:
        raise ValueError(f"Missing {split} split in {npz_path}. Keys={list(npz.keys())}")
    arr = npz[key]
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [list(seq) for seq in arr]
    return [list(seq) for seq in arr.tolist()]


def _histories_to_sequences(history_arr: np.ndarray) -> List[List[int]]:
    """Convert a 2D padded history array to variable-length sequences."""
    sequences: List[List[int]] = []
    for row in history_arr:
        seq = [int(x) for x in row.tolist() if int(x) != 0]
        sequences.append(seq)
    return sequences


def _read_txt_sequences(txt_path: str) -> List[List[int]]:
    rows: List[List[int]] = []
    with Path(txt_path).open("r") as f:
        for line in f:
            parts = [int(x) for x in line.strip().split()]
            if len(parts) < 3:
                continue
            items = parts[1:]
            if len(items) < 2:
                continue
            rows.append(items)
    return rows


class SeqDataset(Dataset):
    """Sequence dataset for train/val/test splits."""

    def __init__(
        self,
        sequences: List[List[int]],
        stage: str,
        max_seq_len: int,
        item_id_offset: int = 0,
        train_stride: int = 5,
        user_ids: Optional[List[int]] = None,
    ):
        self.sequences = sequences
        self.stage = stage
        self.max_seq_len = max_seq_len
        self.item_id_offset = item_id_offset
        self.train_stride = train_stride
        if user_ids is None:
            user_ids = list(range(len(sequences)))
        if len(user_ids) != len(sequences):
            raise ValueError("user_ids length must match sequences length")
        self.user_ids = user_ids
        self.samples = self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_idx, target_pos = self.samples[idx]
        seq = self.sequences[seq_idx]
        user_id = self.user_ids[seq_idx]
        recent = seq[:target_pos]
        target = seq[target_pos]
        if self.item_id_offset != 0:
            recent = [x + self.item_id_offset for x in recent]
            target = target + self.item_id_offset
        if self.max_seq_len > 0:
            recent = recent[-self.max_seq_len :]
        return (
            torch.tensor(recent, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(user_id, dtype=torch.long),
        )

    def _build_samples(self) -> List[Tuple[int, int]]:
        if self.stage == "train":
            offsets = None
        elif self.stage == "val":
            offsets = [2]
        else:
            offsets = [1]
        samples: List[Tuple[int, int]] = []
        for seq_idx, seq in enumerate(self.sequences):
            seq_len = len(seq)
            if self.stage == "train":
                stride = max(self.train_stride, 1)
                for offset in range(3, seq_len, stride):
                    target_pos = seq_len - offset
                    if target_pos <= 0:
                        continue
                    samples.append((seq_idx, target_pos))
            else:
                for offset in offsets:
                    if seq_len < offset:
                        continue
                    target_pos = seq_len - offset
                    if target_pos <= 0:
                        continue
                    samples.append((seq_idx, target_pos))
        return samples


class TablePairDataset(Dataset):
    """Dataset for table-style npz with history/target/label."""

    def __init__(
        self,
        histories: np.ndarray,
        targets: np.ndarray,
        labels: np.ndarray,
        max_seq_len: int,
        item_id_offset: int = 0,
        user_ids: Optional[np.ndarray] = None,
    ):
        self.histories = histories
        self.targets = targets
        self.labels = labels
        self.max_seq_len = max_seq_len
        self.item_id_offset = item_id_offset
        if user_ids is None:
            user_ids = np.arange(len(targets))
        self.user_ids = user_ids

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history = np.asarray(self.histories[idx])
        target = int(self.targets[idx])
        label = float(self.labels[idx])
        if self.item_id_offset != 0:
            history = history + self.item_id_offset
            target = target + self.item_id_offset
        if self.max_seq_len > 0:
            history = history[-self.max_seq_len :]
        return (
            torch.tensor(history, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(self.user_ids[idx], dtype=torch.long),
        )


def _collate(batch):
    recents = [b[0] for b in batch]
    targets = torch.stack([b[1] for b in batch])
    recents_padded = pad_sequence(recents, batch_first=True, padding_value=0)
    attention_mask = (recents_padded != 0)
    out = {
        "input_ids": recents_padded,
        "attention_mask": attention_mask,
        "target_ids": targets,
    }
    labels = None
    user_ids = None
    if len(batch[0]) == 4:
        labels = torch.stack([b[2] for b in batch])
        user_ids = torch.stack([b[3] for b in batch])
    elif len(batch[0]) == 3:
        third = batch[0][2]
        if torch.is_floating_point(third):
            labels = torch.stack([b[2] for b in batch])
        else:
            user_ids = torch.stack([b[2] for b in batch])
    if labels is not None:
        out["labels"] = labels
    if user_ids is not None:
        out["user_ids"] = user_ids
    return out


def create_dataloaders_from_path(
    data_path: str,
    max_seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    item_id_offset: int = 0,
    train_stride: int = 5,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if data_path.endswith(".npz"):
        npz = np.load(data_path, allow_pickle=True)
        if "history_item_index" in npz and "target_item_index" not in npz:
            sequences = _histories_to_sequences(npz["history_item_index"])
            train_sequences = sequences
            val_sequences = sequences
            test_sequences = sequences
        elif "history_item_index" in npz and "target_item_index" in npz and "label" in npz:
            return create_dataloaders_from_table_npz(
                data_path,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                num_workers=num_workers,
                item_id_offset=item_id_offset,
            )
        else:
            train_sequences = _read_npz_sequences(data_path, "train")
            val_sequences = _read_npz_sequences(data_path, "val")
            test_sequences = _read_npz_sequences(data_path, "test")
    else:
        sequences = _read_txt_sequences(data_path)
        train_sequences = sequences
        val_sequences = sequences
        test_sequences = sequences

    ds_tr = SeqDataset(
        train_sequences,
        "train",
        max_seq_len,
        item_id_offset=item_id_offset,
        train_stride=train_stride,
    )
    ds_va = SeqDataset(val_sequences, "val", max_seq_len, item_id_offset=item_id_offset)
    ds_te = SeqDataset(test_sequences, "test", max_seq_len, item_id_offset=item_id_offset)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
    )
    return dl_tr, dl_va, dl_te


def load_item_embeddings_from_npz(npz_path: str) -> Optional[np.ndarray]:
    npz = np.load(npz_path, allow_pickle=True)
    for key in ("item_embeddings", "item_emb", "embeddings", "item_embedding"):
        if key in npz:
            return npz[key]
    return None


def load_item_embeddings_from_indexed_npz(npz_path: str) -> Optional[np.ndarray]:
    npz = np.load(npz_path, allow_pickle=True)
    index_key = "item_index" if "item_index" in npz else "history_item_index" if "history_item_index" in npz else None
    if index_key is None or "embedding" not in npz:
        return None
    item_index = npz[index_key].astype(np.int64)
    embedding = npz["embedding"]
    max_idx = int(item_index.max())
    table = np.zeros((max_idx + 1, embedding.shape[1]), dtype=embedding.dtype)
    table[item_index] = embedding
    return table


def create_dataloaders_from_table_npz(
    data_path: str,
    max_seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    item_id_offset: int = 0,
    seed: int = 1234,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    npz = np.load(data_path, allow_pickle=True)
    histories = npz["history_item_index"]
    targets = npz["target_item_index"]
    labels = npz["label"]
    user_ids = npz["user_id"] if "user_id" in npz else None

    num_rows = len(targets)
    # Keep full table as train; leave val/test empty to avoid accidental random splits.
    ds_tr = TablePairDataset(histories, targets, labels, max_seq_len, item_id_offset, user_ids=user_ids)
    ds_va = TablePairDataset(
        histories[:0],
        targets[:0],
        labels[:0],
        max_seq_len,
        item_id_offset,
        user_ids=user_ids[:0] if user_ids is not None else None,
    )
    ds_te = TablePairDataset(
        histories[:0],
        targets[:0],
        labels[:0],
        max_seq_len,
        item_id_offset,
        user_ids=user_ids[:0] if user_ids is not None else None,
    )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
    )
    return dl_tr, dl_va, dl_te


# === XLong text format (train/test files with pos/neg items) ===
def load_xlong_samples(data_path: str) -> List[Tuple[int, int, List[int], int]]:
    """Load xlong rows and append pos_item to the end of each sequence."""
    samples: List[Tuple[int, int, List[int], int]] = []
    path = Path(data_path)
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 7:
                raise ValueError(f"{path} line {line_num}: expected 7 fields, got {len(parts)}")
            idx = int(parts[0])
            user_id = int(parts[1])
            item_seq = [int(x) for x in parts[2].split(",") if x]
            pos_item = int(parts[3])
            neg_item = int(parts[4])
            full_seq = item_seq + [pos_item]
            samples.append((idx, user_id, full_seq, neg_item))
    return samples


def _build_xlong_item_map(train_samples: List[Tuple[int, int, List[int], int]]) -> Dict[int, int]:
    item_set = set()
    for _, _, seq, neg_item in train_samples:
        item_set.update(seq)
        item_set.add(neg_item)
    item_list = sorted(item_set)
    return {item_id: i + 1 for i, item_id in enumerate(item_list)}


def _map_xlong_sequences(
    samples: List[Tuple[int, int, List[int], int]],
    item_to_id: Dict[int, int],
) -> Tuple[List[List[int]], List[int], int]:
    sequences: List[List[int]] = []
    user_ids: List[int] = []
    dropped = 0
    for _, user_id, seq, neg_item in samples:
        if any(x not in item_to_id for x in seq) or neg_item not in item_to_id:
            dropped += 1
            continue
        mapped = [item_to_id[x] for x in seq]
        sequences.append(mapped)
        user_ids.append(user_id)
    return sequences, user_ids, dropped


def create_xlong_dataloaders(
    train_path: str,
    test_path: str,
    max_seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    train_stride: int = 5,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Build dataloaders for xlong text files.

    Returns train/val loaders from the train file and a test loader from the test file.
    The returned metadata includes the item vocabulary size and dropped-row counts.
    """
    train_samples = load_xlong_samples(train_path)
    item_to_id = _build_xlong_item_map(train_samples)

    train_sequences, train_user_ids, dropped_train = _map_xlong_sequences(train_samples, item_to_id)
    ds_tr = SeqDataset(
        train_sequences,
        "train",
        max_seq_len,
        item_id_offset=0,
        train_stride=train_stride,
        user_ids=train_user_ids,
    )
    # Validation uses the latest interaction from the train file (same as SASRec setup).
    ds_va = SeqDataset(train_sequences, "test", max_seq_len, item_id_offset=0, user_ids=train_user_ids)

    test_samples = load_xlong_samples(test_path) if test_path else []
    test_sequences, test_user_ids, dropped_test = _map_xlong_sequences(test_samples, item_to_id) if test_samples else ([], [], 0)
    ds_te = SeqDataset(test_sequences, "test", max_seq_len, item_id_offset=0, user_ids=test_user_ids)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=False,
    )

    meta = {
        "num_items": len(item_to_id),
        "dropped_train": dropped_train,
        "dropped_test": dropped_test,
    }
    return dl_tr, dl_va, dl_te, meta
