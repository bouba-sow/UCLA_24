"""Data loading — Zhang et al. 2023 (Sci. Rep. 13:651)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from zhang2023_constants import DNK_MARGIN, FRAME_SUBSAMPLE, HALF_WINDOW_FRAMES

TOP4_CHARS = [
    "char_j_bauer",
    "char_b_buchanan",
    "char_c_obrian",
    "char_a_fayed",
]

NO, YES, DNK = 0, 1, 2


def char_col_to_csv_name(col: str) -> str:
    body = col.removeprefix("char_")
    initial, last = body.split("_", 1)
    if last == "obrian":
        last = "OBrian"
    else:
        last = last.capitalize()
    return f"{initial.upper()}.{last}"


def zhang_frame_indices(n_frames: int, half_win: int = HALF_WINDOW_FRAMES) -> np.ndarray:
    """Every 4th frame with valid ±1 s window (~18,900 samples)."""
    return np.arange(half_win, n_frames - half_win, FRAME_SUBSAMPLE)


def apply_dnk_full_timeline(raw: np.ndarray, dnk_margin: int = DNK_MARGIN) -> np.ndarray:
    """Mark ±dnk_margin frames around each label transition (full 30 Hz timeline)."""
    labels = raw.copy()
    n = raw.shape[0]
    for c in range(raw.shape[1]):
        for t in np.where(np.diff(raw[:, c]) != 0)[0]:
            lo = max(0, t - dnk_margin + 1)
            hi = min(n, t + dnk_margin + 1)
            labels[lo:hi, c] = DNK
    return labels


def load_labels_from_csv(
    csv_path: Path,
    char_cols: list[str] = TOP4_CHARS,
    n_frames: int | None = None,
    dnk_margin: int = DNK_MARGIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Load labels on Zhang subsampled frames. Returns (labels, frame_indices)."""
    ev = pd.read_csv(csv_path)
    csv_names = [char_col_to_csv_name(c) for c in char_cols]
    missing = [n for n in csv_names if n not in ev.columns]
    if missing:
        raise ValueError(f"CSV missing character columns: {missing}")

    if n_frames is None:
        n_frames = int(ev["Frame"].max()) + 1 if "Frame" in ev.columns else len(ev) * FRAME_SUBSAMPLE

    sample_idx = zhang_frame_indices(n_frames)
    full = np.zeros((n_frames, len(char_cols)), dtype=np.int8)

    if "Frame" in ev.columns:
        frames = ev["Frame"].to_numpy(dtype=np.int64)
        vals = ev[csv_names].to_numpy(dtype=np.int8)
        ok = (frames >= 0) & (frames < n_frames)
        full[frames[ok]] = vals[ok]
    else:
        dense = ev[csv_names].to_numpy(dtype=np.int8)
        for i, f in enumerate(range(0, min(len(dense) * FRAME_SUBSAMPLE, n_frames), FRAME_SUBSAMPLE)):
            if i < len(dense):
                full[f] = dense[i]

    labelled = apply_dnk_full_timeline(full, dnk_margin)
    return labelled[sample_idx], sample_idx


def load_labels(
    events_tsv: Path,
    char_cols: list[str] = TOP4_CHARS,
    dnk_margin: int = DNK_MARGIN,
) -> tuple[np.ndarray, np.ndarray]:
    ev = pd.read_csv(events_tsv, sep="\t")
    n_frames = len(ev)
    raw_full = ev[char_cols].to_numpy(dtype=np.int8)
    sample_idx = zhang_frame_indices(n_frames)
    labelled = apply_dnk_full_timeline(raw_full, dnk_margin)
    return labelled[sample_idx], sample_idx


def load_firing_rates(
    spike_dir: Path,
    events_tsv: Path,
    unit_threshold: int = 1,
    min_rate_hz: float = 0.05,
) -> tuple[np.ndarray, list[str]]:
    """30 Hz spike counts per sorted unit (one column per cluster).

    Zhang et al. use individually sorted neurons, not microwire-channel pools.
    Units with mean rate < min_rate_hz over the movie are dropped.
    """
    ev = pd.read_csv(events_tsv, sep="\t")
    stim = ev["stimulus_time"].to_numpy(dtype=np.float64)
    n_frames = len(stim)

    dt = np.diff(stim)
    mid = (stim[:-1] + stim[1:]) / 2.0
    bin_edges = np.empty(n_frames + 1)
    bin_edges[0] = stim[0] - dt[0] / 2.0
    bin_edges[1:-1] = mid
    bin_edges[-1] = stim[-1] + dt[-1] / 2.0
    duration = stim[-1] - stim[0] if len(stim) > 1 else 1.0

    npz_files = sorted(spike_dir.glob("*_spikedata.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No *_spikedata.npz in {spike_dir}")

    unit_names: list[str] = []
    cols: list[np.ndarray] = []
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        channel = str(d["channel"])
        cluster_ids = d["cluster_id"]
        spike_times = d["spike_times_movie"]
        for cluster in sorted(set(cluster_ids)):
            cid = int(cluster)
            if cid < unit_threshold:
                continue
            mask = cluster_ids == cid
            spikes = spike_times[mask]
            counts, _ = np.histogram(spikes, bins=bin_edges)
            rate_hz = counts.sum() / max(duration, 1e-6)
            if rate_hz < min_rate_hz:
                continue
            cols.append(counts.astype(np.float32))
            unit_names.append(f"{channel}_c{cid}")

    if not cols:
        raise RuntimeError("No units passed min firing rate filter")
    return np.stack(cols, axis=1), unit_names


class CharacterDataset(Dataset):
    def __init__(
        self,
        rates: np.ndarray,
        labels: np.ndarray,
        frame_indices: np.ndarray,
        half_win: int = HALF_WINDOW_FRAMES,
    ) -> None:
        self.rates = rates
        self.labels = labels
        self.frame_indices = frame_indices
        self.half_win = half_win

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        f = int(self.frame_indices[i])
        hw = self.half_win
        x = torch.from_numpy(self.rates[f - hw : f + hw].copy())
        y = torch.from_numpy(self.labels[i].astype(np.int64))
        return x, y


def fit_normalizer(rates: np.ndarray, frame_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    subset = rates[frame_indices]
    mu = subset.mean(axis=0)
    sigma = subset.std(axis=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def apply_normalizer(rates: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((rates - mu) / sigma).astype(np.float32)
