"""BIDS data loading for character decoding.

All inputs come from:
  - data/bids/sub-{sub}/ses-{ses}/ieeg/*_events.tsv        (frame labels)
  - data/bids/derivatives/spike-sorted/sub-{sub}/ses-{ses}/ieeg/*_spikedata.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Top-4 characters by screen time in the 22 March 2024 session
TOP4_CHARS = ["char_j_bauer", "char_b_buchanan", "char_c_obrian", "char_a_fayed"]

# Frames on either side of a label transition are marked Don't-Know
DNK_MARGIN = 5

# Label codes
NO, YES, DNK = 0, 1, 2


# ---------------------------------------------------------------------------
# Firing-rate matrix
# ---------------------------------------------------------------------------

def load_firing_rates(
    spike_dir: Path,
    events_tsv: Path,
    unit_threshold: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """Build (N_frames, N_channels) spike-count matrix from BIDS NPZ files.

    Spikes are binned to the exact frame grid defined by stimulus_time in
    events_tsv. Only spikes with cluster_id >= unit_threshold are counted
    (cluster_id=0 is noise).

    Returns
    -------
    rates : float32 array of shape (N_frames, N_channels)
    channels : list of channel name strings
    """
    ev = pd.read_csv(events_tsv, sep="\t")
    stim = ev["stimulus_time"].to_numpy(dtype=np.float64)
    n_frames = len(stim)

    # Build bin edges centred on each frame time
    dt = np.diff(stim)
    mid = (stim[:-1] + stim[1:]) / 2.0
    bin_edges = np.empty(n_frames + 1)
    bin_edges[0] = stim[0] - dt[0] / 2.0
    bin_edges[1:-1] = mid
    bin_edges[-1] = stim[-1] + dt[-1] / 2.0

    npz_files = sorted(spike_dir.glob("*_spikedata.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No *_spikedata.npz files found in {spike_dir}")

    channels: list[str] = []
    rate_cols: list[np.ndarray] = []

    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        mask = d["cluster_id"] >= unit_threshold
        spikes = d["spike_times_movie"][mask]
        counts, _ = np.histogram(spikes, bins=bin_edges)
        rate_cols.append(counts.astype(np.float32))
        channels.append(str(d["channel"]))

    return np.stack(rate_cols, axis=1), channels  # (N_frames, N_ch)


# ---------------------------------------------------------------------------
# Label matrix
# ---------------------------------------------------------------------------

def load_labels(
    events_tsv: Path,
    char_cols: list[str] = TOP4_CHARS,
    dnk_margin: int = DNK_MARGIN,
) -> np.ndarray:
    """Build (N_frames, N_chars) label array.

    Values: NO=0, YES=1, DNK=2.
    Frames within dnk_margin of a character label transition are marked DNK
    to handle visual-presence ambiguity at scene cuts (Zhang et al. 2023).
    """
    ev = pd.read_csv(events_tsv, sep="\t")
    raw = ev[char_cols].to_numpy(dtype=np.int8)
    labels = raw.copy()

    n = raw.shape[0]
    for c in range(raw.shape[1]):
        transitions = np.where(np.diff(raw[:, c]) != 0)[0]
        for t in transitions:
            lo = max(0, t - dnk_margin + 1)
            hi = min(n, t + dnk_margin + 1)
            labels[lo:hi, c] = DNK

    return labels  # (N_frames, N_chars)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CharacterDataset(Dataset):
    """Windowed firing-rate dataset for character presence decoding.

    Each sample is a (T, N_ch) window centred on a frame, paired with a
    (N_chars,) label vector (NO/YES/DNK per character).

    Parameters
    ----------
    rates : (N_frames, N_ch) float32 array — pre-normalised firing rates
    labels : (N_frames, N_chars) int8 array
    frame_indices : 1-D array of valid frame indices (must satisfy half_win
        <= index < N_frames - half_win)
    half_win : half-window size in frames (default 30 → ±1 s at 30 fps)
    """

    def __init__(
        self,
        rates: np.ndarray,
        labels: np.ndarray,
        frame_indices: np.ndarray,
        half_win: int = 30,
    ) -> None:
        self.rates = rates
        self.labels = labels
        self.frame_indices = frame_indices
        self.half_win = half_win

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.frame_indices[i]
        hw = self.half_win
        x = torch.from_numpy(self.rates[f - hw : f + hw].copy())   # (T, N_ch)
        y = torch.from_numpy(self.labels[f].astype(np.int64))       # (N_chars,)
        return x, y


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def fit_normalizer(rates: np.ndarray, frame_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std on the given frames (training set)."""
    subset = rates[frame_indices]
    mu = subset.mean(axis=0)
    sigma = subset.std(axis=0)
    sigma[sigma == 0] = 1.0  # avoid division by zero for silent channels
    return mu, sigma


def apply_normalizer(
    rates: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Return z-scored copy of rates."""
    return ((rates - mu) / sigma).astype(np.float32)
