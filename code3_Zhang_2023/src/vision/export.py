"""Export frame-wise labels at Zhang subsampling (every 4th frame)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zhang2023_constants import FRAME_SUBSAMPLE

from .io import probe_video


def export_frame_labels(
    stage4_npz: Path,
    video_path: Path,
    characters: list[str],
    out_csv: Path,
    prob_thresh: float = 0.5,
    frame_step: int = FRAME_SUBSAMPLE,
    auxiliary: list[str] | None = None,
) -> Path:
    data = np.load(stage4_npz, allow_pickle=True)
    probs = data["frame_probs"]
    thresh = float(np.asarray(data["prob_thresh"]).item()) if "prob_thresh" in data else prob_thresh
    meta = probe_video(video_path)

    frame_indices = np.arange(0, meta.n_frames, frame_step)
    rows = []
    for f in frame_indices:
        if f >= probs.shape[0]:
            break
        row = {"Frame": int(f)}
        p = probs[f]
        binary = (p[: len(characters)] >= thresh).astype(np.int8)
        for i, name in enumerate(characters):
            row[name] = int(binary[i])
        rows.append(row)

    df = pd.DataFrame(rows)
    aux = auxiliary or ["Face", "Person", "No Characters"]
    if "Face" in aux:
        df["Face"] = (df[characters].sum(axis=1) > 0).astype(np.int8)
    if "Person" in aux:
        df["Person"] = 0
    if "No Characters" in aux:
        df["No Characters"] = (df[characters].sum(axis=1) == 0).astype(np.int8)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv
