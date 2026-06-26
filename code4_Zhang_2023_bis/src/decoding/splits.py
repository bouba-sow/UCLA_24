"""Leakage-free CV splits for sliding-window decoding on continuous movies."""
from __future__ import annotations

import numpy as np


def temporal_block_splits(
    sample_frames: np.ndarray,
    n_folds: int,
    half_win: int,
    val_fraction: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Contiguous temporal blocks with boundary purge.

    Samples are ordered in time (``sample_frames`` non-decreasing). Each fold
    holds one contiguous block out for test. Training/validation centers must be
    at least ``2 * half_win`` frames away from any test center so ±half_win
    windows do not share spike bins.

    Returns list of (train_i, val_i, test_i) index arrays into the sample axis.
    """
    n_samples = len(sample_frames)
    if n_samples < n_folds * 3:
        raise ValueError(f"Too few samples ({n_samples}) for {n_folds} folds")

    fold_size = n_samples // n_folds
    blocks = [
        np.arange(i * fold_size, (i + 1) * fold_size if i < n_folds - 1 else n_samples)
        for i in range(n_folds)
    ]

    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fold_idx, test_i in enumerate(blocks):
        test_lo = int(sample_frames[test_i[0]])
        test_hi = int(sample_frames[test_i[-1]])

        all_i = np.arange(n_samples)
        pool = all_i[~np.isin(all_i, test_i)]
        purge = 2 * half_win
        pool = pool[
            (sample_frames[pool] < test_lo - purge)
            | (sample_frames[pool] > test_hi + purge)
        ]

        order = np.argsort(sample_frames[pool])
        pool = pool[order]
        n_val = max(1, int(len(pool) * val_fraction))
        val_i = pool[:n_val]
        train_i = pool[n_val:]
        splits.append((train_i, val_i, test_i))

    return splits


def assert_no_window_overlap(
    sample_frames: np.ndarray,
    train_i: np.ndarray,
    test_i: np.ndarray,
    half_win: int,
) -> None:
    """Raise if any train/test window shares a spike frame bin."""
    gap = 2 * half_win
    for ti in test_i:
        ft = int(sample_frames[ti])
        for tr in train_i:
            fr = int(sample_frames[tr])
            if abs(ft - fr) < gap:
                raise AssertionError(
                    f"Window overlap: test center {ft} vs train center {fr} (need gap>={gap})"
                )
