"""
Data loading module for semantic code sequences.
Loads pre-computed semantic codes (no need for RQ-VAE during training).
"""

from typing import Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, BatchSampler
from einops import rearrange


class SemanticCodeSequenceDataset(Dataset):
    """
    Dataset that loads item index sequences and their pre-computed semantic codes.
    
    Expected structure:
        - semantic_codes: [num_items, num_levels] pre-computed codes
        - sequences: [num_sequences, seq_len] item indices
        - targets: [num_sequences, num_levels] next item codes (optional)
    """
    
    def __init__(
        self,
        semantic_codes: torch.Tensor,  # [num_items, num_levels] pre-computed codes
        sequences: torch.Tensor,       # [num_sequences, seq_len] item indices
        device: str = "cpu",
        targets: Optional[torch.Tensor] = None,  # [num_sequences, num_levels] next item codes
    ):
        """
        Args:
            semantic_codes: Pre-computed semantic codes [num_items, num_levels]
            sequences: Item index sequences [num_sequences, seq_len]
            device: Device to use
            targets: Optional target codes for supervised learning
        """
        self.semantic_codes = semantic_codes.to(device)
        self.sequences = sequences
        self.device = device
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            seq_codes: [seq_len, num_levels] semantic codes for the sequence
            target: [num_levels] next item codes (if provided)
        """
        item_indices = self.sequences[idx]  # [seq_len]
        seq_codes = self.semantic_codes[item_indices]  # [seq_len, num_levels]
        
        if self.targets is not None:
            target = self.targets[idx]  # [num_levels]
            return seq_codes, target
        else:
            return seq_codes, None


class SemanticCodeCollator:
    """
    Collate function for semantic code sequences.
    Handles variable length sequences and flattening codes.
    """
    
    def __init__(self, flatten_codes: bool = False, pad_token_id: int = 0):
        """
        Args:
            flatten_codes: If True, flatten [seq_len, num_levels] -> [seq_len*num_levels].
                           Prefer keeping 3D shape for clarity.
            pad_token_id: Token ID for padding masked positions (should be 0 for code indices)
        """
        self.flatten_codes = flatten_codes
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        """
        Collate batch of variable-length sequences.
        
        Args:
            batch: List of (seq_codes, target) tuples
        
        Returns:
            Dict with:
                - input_ids: [B, seq_len] or [B, seq_len*num_levels] if flattened
                - attention_mask: [B, seq_len] or [B, seq_len*num_levels]
                - target_ids: [B, num_levels] or None
        """
        seq_codes_list = [item[0] for item in batch]
        targets_list = [item[1] for item in batch]
        
        # Handle variable lengths - pad to max in batch
        max_seq_len = max(codes.shape[0] for codes in seq_codes_list)
        num_levels = seq_codes_list[0].shape[1]
        batch_size = len(batch)
        
        # Pad sequences
        padded_codes = torch.full(
            (batch_size, max_seq_len, num_levels),
            self.pad_token_id,
            dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        
        for i, codes in enumerate(seq_codes_list):
            seq_len = codes.shape[0]
            padded_codes[i, :seq_len] = codes
            attention_mask[i, :seq_len] = True
        
        # Optionally flatten [B, seq_len, num_levels] -> [B, seq_len*num_levels]
        if self.flatten_codes:
            # Flatten seq_len and num_levels dimensions
            padded_codes = rearrange(padded_codes, "b s l -> b (s l)")
            # Expand attention mask to match flattened shape: each position gets repeated num_levels times
            attention_mask = rearrange(attention_mask, "b s -> b s 1")  # [batch, seq_len, 1]
            attention_mask = rearrange(attention_mask, "b s l -> b (s l)")  # [batch, seq_len*num_levels]
        
        result = {
            "input_ids": padded_codes,
            "attention_mask": attention_mask,
        }
        
        # Add targets if provided
        if targets_list[0] is not None:
            targets = torch.stack(targets_list, dim=0)  # [B, num_levels]
            result["target_ids"] = targets
        
        return result

# # Legacy helper, not used in current training path.
# def create_semantic_dataloader(
#     semantic_codes_path: str,
#     sequences_path: str,
#     batch_size: int = 32,
#     num_workers: int = 0,
#     device: str = "cpu",
#     targets_path: Optional[str] = None,
#     flatten_codes: bool = False,
#     shuffle: bool = True,
# ) -> DataLoader:
#     """
#     Create a DataLoader for semantic code sequences. (Legacy helper, not used in current training path.)
    
#     Args:
#         semantic_codes_path: Path to pre-computed semantic codes (.pt file)
#         sequences_path: Path to sequence indices (.pt file)
#         batch_size: Batch size for DataLoader
#         num_workers: Number of workers (0 for no multiprocessing)
#         device: Device to use
#         targets_path: Optional path to target codes
#         flatten_codes: Whether to flatten [seq_len, num_levels] to [seq_len*num_levels]
#         shuffle: Whether to shuffle the dataset
    
#     Returns:
#         DataLoader instance
#     """
#     # Load data
#     semantic_codes = torch.load(semantic_codes_path).long() + 1  # shift to reserve 0 as PAD
#     sequences = torch.load(sequences_path)
#     targets = torch.load(targets_path) if targets_path else None
    
#     # Create dataset
#     dataset = SemanticCodeSequenceDataset(
#         semantic_codes=semantic_codes,
#         sequences=sequences,
#         device=device,
#         targets=targets,
#     )
    
#     # Create collator and dataloader
#     collator = SemanticCodeCollator(flatten_codes=flatten_codes)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         collate_fn=collator,
#         num_workers=num_workers,
#     )
    
#     return dataloader


def create_dataloaders_from_txt(
    data_txt_path: str,
    semantic_codes_path: str,
    batch_size: int = 32,
    num_levels: int = 3,
    flatten_codes: bool = False,
    max_seq_len: int = 50,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders from data.txt and pre-computed semantic codes.
    
    Data format in each sequence:
    - train: input = seq[:-2], target = seq[-2]   (predict 2nd to last)
    - val:   input = seq[:-1], target = seq[-1]   (predict last)
    - test:  input = seq[:-1], target = seq[-1]   (predict last, same as val)
    
    Args:
        data_txt_path: Path to data.txt (format: item1 item2 ... itemN per line)
        semantic_codes_path: Path to item_semantic_codes.pt [num_items, num_levels]
        batch_size: Batch size
        num_levels: Number of RQ-VAE levels
        flatten_codes: Whether to flatten codes
        max_seq_len: Maximum sequence length
    
    Returns:
        (train_dataloader, val_dataloader, test_dataloader)
    """
    # Load semantic codes
    semantic_codes = torch.load(semantic_codes_path).long() + 1  # shift to reserve 0 as PAD

    # Load sequences from data.txt
    sequences = []
    with open(data_txt_path, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            # First element is user_id, rest are items
            # Note: items in data.txt are 1-indexed (from item2idx), 
            # but we need 0-indexed for semantic_codes
            items = [x - 1 for x in parts[1:]] if len(parts) > 1 else []
            # Keep only the most recent interactions (tail)
            if items:
                items = items[-max_seq_len:]
            # Need at least 2 items: input + target
            if len(items) >= 2:
                sequences.append(items)
    
    print(f"[Data] Loaded {len(sequences)} sequences from {data_txt_path}")


    
    # Create datasets with targets
    def create_dataset(seqs, split_type: str):
        """
        Convert sequences to (input_seq, target) pairs.
        
        Args:
            seqs: List of sequences
            split_type: 'train', 'val', or 'test'
                - train: input=seq[:-2], target=seq[-2]  (leave out last 2)
                - val:   input=seq[:-1], target=seq[-1]  (leave out last 1)
                - test:  input=seq[:-1], target=seq[-1]  (leave out last 1, same as val)
        """
        seq_codes_list = []
        target_codes_list = []
        
        for seq in seqs:
            if split_type == 'train':
                # Need at least 3 items: reserve last two for val/test
                if len(seq) < 3:
                    continue
                input_seq = seq[:-2]
                target_idx = seq[-2]
            else:  # 'val' or 'test'
                if len(seq) < 2:
                    continue
                input_seq = seq[:-1]
                target_idx = seq[-1]
            
            # Get codes
            input_codes = semantic_codes[input_seq]  # [input_len, num_levels]
            target_codes = semantic_codes[target_idx]  # [num_levels]
            
            seq_codes_list.append(input_codes)
            target_codes_list.append(target_codes)
        
        return seq_codes_list, target_codes_list
    
    # Use ALL sequences for all splits (each sequence contributes to all 3)
    train_codes, train_targets = create_dataset(sequences, 'train')
    val_codes, val_targets = create_dataset(sequences, 'val')
    test_codes, test_targets = create_dataset(sequences, 'test')
    
    print(f"[Data] Train: {len(train_codes)} samples, Val: {len(val_codes)} samples, Test: {len(test_codes)} samples")
    
    # Create dataloaders
    class CodeDataset(Dataset):
        def __init__(self, codes_list, targets_list):
            self.codes_list = codes_list
            self.targets_list = targets_list
        
        def __len__(self):
            return len(self.codes_list)
        
        def __getitem__(self, idx):
            return self.codes_list[idx], self.targets_list[idx]
    
    collator = SemanticCodeCollator(flatten_codes=flatten_codes)
    
    train_dl = DataLoader(
        CodeDataset(train_codes, train_targets),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    
    val_dl = DataLoader(
        CodeDataset(val_codes, val_targets),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    
    test_dl = DataLoader(
        CodeDataset(test_codes, test_targets),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    
    return train_dl, val_dl, test_dl
