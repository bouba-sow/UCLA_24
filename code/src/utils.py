"""
Shared utilities for signal alignment, SRT parsing, and text normalization.
Used across feature extraction scripts and notebooks.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def srt_time_to_seconds(ts: str) -> float:
    hh, mm, rest = ts.split(':')
    ss, ms = rest.split(',')
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_srt(file_path: Path) -> pd.DataFrame:
    txt = file_path.read_text(encoding='utf-8')
    blocks = [b.strip() for b in txt.split('\n\n') if b.strip()]
    rows = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3 or '-->' not in lines[1]:
            continue
        start_txt, end_txt = [x.strip() for x in lines[1].split('-->')]
        rows.append({
            'start': srt_time_to_seconds(start_txt),
            'end': srt_time_to_seconds(end_txt),
            'text': ' '.join(lines[2:]).strip(),
        })
    return pd.DataFrame(rows).sort_values('start').reset_index(drop=True)


def normalize_token(x: str) -> str:
    return re.sub(r"[^\w']+", '', str(x).lower())


def minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    out = np.zeros_like(x)
    if finite.any():
        lo, hi = np.nanmin(x[finite]), np.nanmax(x[finite])
        out[finite] = (x[finite] - lo) / (hi - lo + 1e-8)
    return out


def onset_and_duration(
    series: pd.Series,
    is_valid,
    frame_time: np.ndarray,
    frame_dt: float,
) -> tuple[pd.Series, pd.Series]:
    """Return (onset_label, onset_duration) Series for contiguous valid runs."""
    onset_label = pd.Series('', index=series.index, dtype=object)
    onset_duration = pd.Series(0.0, index=series.index, dtype=float)
    n = len(series)
    i = 0
    while i < n:
        label = series.iat[i]
        if not is_valid(label):
            i += 1
            continue
        j = i
        while j + 1 < n and series.iat[j + 1] == label:
            j += 1
        onset_label.iat[i] = str(label)
        onset_duration.iat[i] = float((frame_time[j] + frame_dt) - frame_time[i])
        i = j + 1
    return onset_label, onset_duration


def align(source_values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Bin-average source_values onto target_times; interpolate empty bins."""
    source_values = np.asarray(source_values, dtype=float)
    source_times = np.asarray(source_times, dtype=float)
    target_times = np.asarray(target_times, dtype=float)

    if source_values.ndim != 1 or source_times.ndim != 1 or target_times.ndim != 1:
        raise ValueError('align expects 1-D arrays')
    if len(source_values) != len(source_times):
        raise ValueError('source_values and source_times must have the same length')
    if len(source_values) == 0 or len(target_times) == 0:
        return np.zeros(len(target_times), dtype=float)
    if np.any(np.diff(source_times) < 0) or np.any(np.diff(target_times) < 0):
        raise ValueError('source_times and target_times must be monotonic non-decreasing')

    n_target = len(target_times)
    out = np.zeros(n_target, dtype=float)
    frame_dt_target = float(np.median(np.diff(target_times))) if n_target > 1 else 0.0
    edges = np.empty(n_target + 1)
    if n_target > 1:
        edges[1:-1] = 0.5 * (target_times[:-1] + target_times[1:])
        edges[0] = target_times[0] - 0.5 * frame_dt_target
        edges[-1] = target_times[-1] + 0.5 * frame_dt_target
    else:
        edges[0] = target_times[0] - 0.5
        edges[-1] = target_times[0] + 0.5

    bin_idx = np.searchsorted(edges, source_times, side='right') - 1
    counts = np.zeros(n_target, dtype=float)
    for src_i, b in enumerate(bin_idx):
        if 0 <= b < n_target:
            out[b] += source_values[src_i]
            counts[b] += 1.0

    empty = counts == 0
    if empty.any():
        out[empty] = np.interp(
            target_times[empty], source_times, source_values,
            left=float(source_values[0]), right=float(source_values[-1]),
        )
        counts[empty] = 1.0

    out /= counts
    return out
