"""Stage 3 — map visual clusters to character identities (human supervision)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .stage1_detect_track import load_stage1_tracks


def bootstrap_assignments_from_reference(
    tracks: list[dict],
    cluster_info: dict,
    reference_csv: Path,
    characters: list[str],
) -> dict[int, str]:
    """Heuristic: vote reference labels on frames where each cluster appears."""
    import pandas as pd

    ref = pd.read_csv(reference_csv)
    track_to_cluster = {int(k): int(v) for k, v in cluster_info["track_to_cluster"].items()}
    cluster_votes: dict[int, dict[str, int]] = {}

    for ti, tr in enumerate(tracks):
        if ti not in track_to_cluster:
            continue
        cid = track_to_cluster[ti]
        votes = cluster_votes.setdefault(cid, {c: 0 for c in characters})
        for f in tr["frames"]:
            if f >= len(ref):
                continue
            row = ref.iloc[int(f)]
            for c in characters:
                if c in row and int(row[c]) == 1:
                    votes[c] += 1

    assignments: dict[int, str] = {}
    for cid, votes in cluster_votes.items():
        best = max(votes, key=votes.get)
        if votes[best] > 0:
            assignments[cid] = best
    return assignments


def export_cluster_montages(
    tracks: list[dict],
    cluster_info: dict,
    out_dir: Path,
    n_samples: int = 9,
) -> None:
    """Save montage JPEGs per cluster for manual inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters = cluster_info["clusters"]
    for cid_str, members in clusters.items():
        cid = int(cid_str)
        paths: list[str] = []
        for ti in members:
            paths.extend([p for p in tracks[int(ti)].get("crop_paths", []) if p])
        paths = paths[:: max(1, len(paths) // n_samples)][:n_samples]
        tiles = []
        for p in paths:
            img = cv2.imread(p)
            if img is not None and img.size:
                tiles.append(cv2.resize(img, (128, 128)))
        if not tiles:
            continue
        row = cv2.hconcat(tiles)
        cv2.imwrite(str(out_dir / f"cluster_{cid:03d}.jpg"), row)


def load_assignments(path: Path) -> dict[int, str]:
    with open(path) as fh:
        raw = json.load(fh)
    return {int(k): v for k, v in raw.items() if not str(k).startswith("_")}


def run_stage3(
    tracks_npz: Path,
    stage2_json: Path,
    assignments_path: Path,
    work_dir: Path,
    reference_csv: Path | None = None,
    characters: list[str] | None = None,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "stage3_assignments.json"

    tracks = load_stage1_tracks(tracks_npz)
    with open(stage2_json) as fh:
        cluster_info = json.load(fh)

    export_cluster_montages(tracks, cluster_info, work_dir / "cluster_montages")

    if assignments_path.exists():
        assignments = load_assignments(assignments_path)
    elif reference_csv is not None and characters:
        assignments = bootstrap_assignments_from_reference(
            tracks, cluster_info, reference_csv, characters
        )
        with open(work_dir / "stage3_assignments_bootstrapped.json", "w") as fh:
            json.dump({str(k): v for k, v in assignments.items()}, fh, indent=2)
    else:
        raise FileNotFoundError(
            "Stage 3 (human supervision): create cluster_assignments.json by inspecting "
            f"{work_dir / 'cluster_montages'}/ and mapping cluster IDs to character names. "
            "See config/cluster_assignments.example.json"
        )

    track_to_char: dict[int, str] = {}
    t2c = {int(k): int(v) for k, v in cluster_info["track_to_cluster"].items()}
    for ti, cid in t2c.items():
        if cid in assignments:
            track_to_char[ti] = assignments[cid]

    payload = {
        "cluster_assignments": {str(k): v for k, v in assignments.items()},
        "track_to_character": {str(k): v for k, v in track_to_char.items()},
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return out_path
