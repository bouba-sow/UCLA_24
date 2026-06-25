"""Dataset for concept decoding from the cached clusterless tensor.

The clusterless cache (see preprocessing.py) holds:
    X        : (n_windows, 2, n_channels, 50)   already z-scored per bundle
    channels : (n_channels,)  channel names
    bundles  : (n_channels,)  bundle/region label per channel

Labels are the 1 Hz, 8-concept annotations; each second maps to four identical
250 ms windows (paper convention).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

CONCEPTS = ["WhiteHouse", "CTU", "Hostage", "Handcuff",
            "J.Bauer", "B.Buchanan", "A.Fayed", "A.Amar"]

WINDOWS_PER_SEC = 4
MIN_COMBO_SECONDS = 50.0     # paper: exclude label combinations < 50 s screen time


def load_concept_labels(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    return (df[CONCEPTS].fillna(0).values > 0.5).astype(np.float32)   # (T_sec, 8)


def load_clusterless(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    return d["X"], list(d["channels"]), list(d["bundles"])


def build_region_layout(bundles: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (gather_idx (R, Ne_max), elec_mask (R, Ne_max), region_names).

    gather_idx points into the channel axis; padded slots point to a zero channel
    (index = n_channels) added by ConceptDataset.
    """
    order, seen = [], {}
    for b in bundles:
        if b not in seen:
            seen[b] = len(order)
            order.append(b)
    groups = {b: [] for b in order}
    for c, b in enumerate(bundles):
        groups[b].append(c)
    ne_max = max(len(v) for v in groups.values())
    n_ch = len(bundles)
    gather = np.full((len(order), ne_max), n_ch, dtype=np.int64)   # pad -> zero channel
    mask = np.zeros((len(order), ne_max), dtype=bool)
    for r, b in enumerate(order):
        for j, c in enumerate(groups[b]):
            gather[r, j] = c
            mask[r, j] = True
    return gather, mask, order


class ConceptDataset(Dataset):
    def __init__(self, X: np.ndarray, labels: np.ndarray, gather_idx: np.ndarray,
                 elec_mask: np.ndarray, indices: np.ndarray):
        n_ch = X.shape[2]
        # pad a zero channel so padded electrode slots gather zeros
        zero = np.zeros((X.shape[0], X.shape[1], 1, X.shape[3]), dtype=X.dtype)
        self.Xpad = np.concatenate([X, zero], axis=2)     # (n_windows, 2, n_ch+1, 50)
        self.labels = labels
        self.gather = gather_idx
        self.elec_mask = torch.from_numpy(elec_mask)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        w = int(self.indices[i])
        xf = self.Xpad[w]                                 # (2, n_ch+1, 50)
        regionized = xf[:, self.gather, :]                # (2, R, Ne_max, 50)
        x = np.transpose(regionized, (1, 0, 2, 3))        # (R, 2, Ne_max, 50)
        y = self.labels[w // WINDOWS_PER_SEC]
        return (torch.from_numpy(np.ascontiguousarray(x)).float(),
                self.elec_mask,
                torch.from_numpy(y).float())


# ---------------------------------------------------------------------------
# Splits and sampling
# ---------------------------------------------------------------------------

def total_samples(X: np.ndarray, labels: np.ndarray) -> int:
    return int(min(X.shape[0], labels.shape[0] * WINDOWS_PER_SEC))


def _combo_key(labels: np.ndarray, w: int) -> tuple:
    return tuple(int(v) for v in labels[w // WINDOWS_PER_SEC])


def fold_indices(n_samples: int, n_folds: int, fold: int, buffer: int = 2):
    """Contiguous temporal-block fold (Zhang-style), with a ±buffer to avoid leakage."""
    fold_size = n_samples // n_folds
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
    val_idx = np.arange(val_start, val_end)
    blocked = set(range(max(0, val_start - buffer), min(n_samples, val_end + buffer)))
    train_idx = np.array([i for i in range(n_samples) if i not in blocked])
    return train_idx, val_idx


def filter_small_combos(train_idx: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Drop training windows whose label-combination has < MIN_COMBO_SECONDS of time."""
    keys = [_combo_key(labels, w) for w in train_idx]
    counts: dict[tuple, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    min_windows = MIN_COMBO_SECONDS * WINDOWS_PER_SEC
    return np.array([w for w, k in zip(train_idx, keys) if counts[k] >= min_windows])


def make_stratified_sampler(train_idx: np.ndarray, labels: np.ndarray) -> WeightedRandomSampler:
    """Proportional representation of each unique label-combination (paper)."""
    keys = [_combo_key(labels, w) for w in train_idx]
    counts: dict[tuple, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    weights = np.array([1.0 / counts[k] for k in keys], dtype=np.float64)
    return WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)
