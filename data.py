
# === NEW: 用 train.txt 的数据管道（右填充 + mask + last-position 目标） ===
import os, json, torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TxtSeqDataset(Dataset):
    """
    读取 data/<name>/proc/train.txt
    每行: user item1 item2 ... itemN
    规则:
      - 仅保留长度 >= 3 的序列 (保证 train/val/test 构造)
      - train:  recent = seq[:-2], target = seq[-2]
      - val:    recent = seq[:-1], target = seq[-1]
      - test:   recent = seq[:-1], target = seq[-1]  (与 val 一样，常见设置)
    """
    def __init__(self, txt_path: str, stage: str, L_real: int):
        self.stage = stage
        self.L_real = L_real
        self.rows = []
        with open(txt_path, "r") as f:
            for line in f:
                parts = [int(x) for x in line.strip().split()]
                if len(parts) < 3:  # user + >=2 items → len >= 3
                    continue
                u, items = parts[0], parts[1:]
                if len(items) < 2:  # 至少两个 item 才能构建 val/test
                    continue
                self.rows.append(items)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        seq = self.rows[i]
        if self.stage == 'train':
            recent = seq[:-2]
            target = seq[-2]
        else:  # 'val' or 'test'
            recent = seq[:-1]
            target = seq[-1]
        recent = recent[-self.L_real:]  # 只取最近窗口
        return torch.tensor(recent, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def make_dataloaders_from_txt(proc_dir: str, L_real: int, batch_size: int, num_workers: int = 0):
    txt_path = os.path.join(proc_dir, "data.txt")
    ds_tr = TxtSeqDataset(txt_path, 'train', L_real)
    ds_va = TxtSeqDataset(txt_path, 'val',   L_real)
    ds_te = TxtSeqDataset(txt_path, 'test',  L_real)

    def collate(batch):
        recents = [b[0] for b in batch]             # 每个 1D LongTensor，元素∈[1..|I|]；右填充用 0（PAD）
        targets = torch.stack([b[1] for b in batch])
        recents_padded = pad_sequence(recents, batch_first=True, padding_value=0)  # 右填充
        mask_recent = (recents_padded != 0).long()
        return recents_padded, targets, mask_recent

    return (
        DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  collate_fn=collate, num_workers=num_workers, drop_last=False),
        DataLoader(ds_va, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers, drop_last=False),
        DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers, drop_last=False),
    )
