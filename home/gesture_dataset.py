"""
Step 3b: Dataset + DataLoader for cached landmark sequences produced by
preprocess_dataset.py.

Cache layout expected:
    cache/labels.json                 label_name -> label_index
    cache/<label_name>/video1.npz     x: (T,126), c: (T,), label: str
    cache/<label_name>/video2.npz
    ...

Handles variable-length sequences via padding + an explicit `lengths` tensor
(NOT the same thing as the occlusion-driven c_t=0 signal - padding is
"this sequence doesn't exist here", not "the hand is occluded here", so we
track it separately and use it to pick out each sample's true final hidden
state after running the model).
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GestureDataset(Dataset):
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        with open(self.cache_dir / "labels.json") as f:
            self.label_to_idx = json.load(f)
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        self.samples = []  # list of (npz_path, label_idx)
        for label_name, label_idx in self.label_to_idx.items():
            label_dir = self.cache_dir / label_name
            if not label_dir.exists():
                continue
            for npz_path in sorted(label_dir.glob("*.npz")):
                self.samples.append((npz_path, label_idx))

        if not self.samples:
            raise RuntimeError(f"No cached samples found under {self.cache_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, label_idx = self.samples[idx]
        data = np.load(npz_path)
        x = torch.from_numpy(data["x"]).float()   # (T, 126)
        c = torch.from_numpy(data["c"]).float()   # (T,)
        return x, c, label_idx


def collate_fn(batch):
    """
    Pads a list of (x, c, label_idx) tuples of varying T to the max T in the
    batch. Returns:
        x_padded: (B, T_max, 126)
        c_padded: (B, T_max)     - padded frames get c=0 (treated as occlusion,
                                    which is harmless since we discard those
                                    timesteps' output anyway)
        labels:   (B,)
        lengths:  (B,)           - true (unpadded) length of each sequence
    """
    xs, cs, labels = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    T_max = int(lengths.max().item())
    B = len(xs)
    input_size = xs[0].shape[1]

    x_padded = torch.zeros(B, T_max, input_size, dtype=torch.float32)
    c_padded = torch.zeros(B, T_max, dtype=torch.float32)

    for i, (x, c) in enumerate(zip(xs, cs)):
        T = x.shape[0]
        x_padded[i, :T, :] = x
        c_padded[i, :T] = c

    labels = torch.tensor(labels, dtype=torch.long)
    return x_padded, c_padded, labels, lengths


def make_dataloaders(cache_dir: str, batch_size: int = 8, val_split: float = 0.2,
                      seed: int = 0, num_workers: int = 0):
    """
    Splits the dataset into train/val using a STRATIFIED split (each class
    split proportionally) rather than a plain random split.

    This matters a lot here: with class sizes ranging from 6 to 40 samples
    (6.7x imbalance), a random split can easily leave a tiny class like
    "6_help me" (6 total samples) with 0 or 1 val samples, making its
    reported val accuracy nearly meaningless. Stratifying ensures every
    class contributes its own val_split fraction (rounded, minimum 1 sample
    if the class has more than 1 sample).
    """
    dataset = GestureDataset(cache_dir)

    # Group sample indices by label
    label_to_sample_indices = {}
    for i, (_, label_idx) in enumerate(dataset.samples):
        label_to_sample_indices.setdefault(label_idx, []).append(i)

    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []

    for label_idx, indices in label_to_sample_indices.items():
        indices = list(indices)
        rng.shuffle(indices)
        n = len(indices)
        n_val = max(1, round(n * val_split)) if n > 1 else 0
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers)
    return train_loader, val_loader, dataset


if __name__ == "__main__":
    # Quick smoke test against a real cache directory, if one exists.
    import sys
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "cache"
    train_loader, val_loader, dataset = make_dataloaders(cache_dir, batch_size=4)
    print(f"Dataset size: {len(dataset)}, labels: {dataset.label_to_idx}")
    x, c, labels, lengths = next(iter(train_loader))
    print("batch x:", x.shape, "c:", c.shape, "labels:", labels, "lengths:", lengths)
    
    
from collections import Counter

dataset = GestureDataset("cache")
labels = [label_idx for _, label_idx in dataset.samples]

counts = Counter(labels)
for label_idx, count in sorted(counts.items()):
    label_name = dataset.idx_to_label[label_idx]
    print(f"{label_name:20s}: {count} samples")

print(f"\nTotal: {len(dataset)} samples")
print(f"Min class size: {min(counts.values())}, Max class size: {max(counts.values())}")    