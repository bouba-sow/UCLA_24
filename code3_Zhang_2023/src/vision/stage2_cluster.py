"""Stage 2 — FaceNet + iterative k-means / KNN (Zhang stage 2)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.cluster import KMeans

from zhang2023_constants import (
    CLUSTER_ITERATIONS,
    FACE_DROP_CLUSTER_FRAC,
    KMEANS_K,
    KNN_DISTANCE_THRESH,
    KNN_K,
    SUPERNODE_GROWTH_THRESH,
)


@dataclass
class Stage2Config:
    n_clusters: int = KMEANS_K
    knn_k: int = KNN_K
    knn_thresh: float = KNN_DISTANCE_THRESH
    max_iters: int = CLUSTER_ITERATIONS
    device: str = "cpu"
    seed: int = 42


def _require_facenet(device: str) -> tuple[MTCNN, InceptionResnetV1]:
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn, embedder


@torch.inference_mode()
def _embed_cluster_crops(
    paths: list[str],
    mtcnn: MTCNN,
    embedder: InceptionResnetV1,
    device: str,
) -> np.ndarray | None:
    embs: list[np.ndarray] = []
    n_fail = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            n_fail += 1
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)
        if face is None:
            n_fail += 1
            continue
        vec = embedder(face.unsqueeze(0).to(device)).cpu().numpy().ravel()
        embs.append(vec)
    if not paths or n_fail / len(paths) > FACE_DROP_CLUSTER_FRAC:
        return None
    if not embs:
        return None
    return np.mean(embs, axis=0)


def _cluster_median_distance(a_embs: list[np.ndarray], b_embs: list[np.ndarray], k: int) -> float:
    dists = []
    for ea in a_embs:
        d = [np.linalg.norm(ea - eb) for eb in b_embs]
        d.sort()
        dists.append(float(np.median(d[: min(k, len(d))])))
    return float(min(dists)) if dists else np.inf


def _distortion(feats: np.ndarray, labels: np.ndarray, cid: int) -> float:
    members = feats[labels == cid]
    if len(members) < 2:
        return 0.0
    centroid = members.mean(axis=0)
    return float(np.mean(np.linalg.norm(members - centroid, axis=1)))


def run_stage2(tracks: list[dict], work_dir: Path, cfg: Stage2Config) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "stage2_clusters.json"
    if out_path.exists():
        return out_path

    mtcnn, embedder = _require_facenet(cfg.device)
    track_feats: list[np.ndarray] = []
    valid_tracks: list[int] = []
    track_paths: dict[int, list[str]] = {}

    for ti, tr in enumerate(tracks):
        paths = [p for p in tr.get("crop_paths", []) if p]
        if not paths:
            continue
        feat = _embed_cluster_crops(paths, mtcnn, embedder, cfg.device)
        if feat is None:
            continue
        track_feats.append(feat)
        valid_tracks.append(ti)
        track_paths[ti] = paths

    if not track_feats:
        raise RuntimeError("Stage 2: no FaceNet embeddings — check YOLO/SORT stage 1 output")

    ti_to_row = {ti: i for i, ti in enumerate(valid_tracks)}
    X = np.stack(track_feats)
    k = min(cfg.n_clusters, len(X))
    labels = KMeans(n_clusters=k, random_state=cfg.seed, n_init=10).fit_predict(X)

    # Good supernodes: low distortion (below median)
    distortions = [_distortion(X, labels, c) for c in range(k)]
    thresh_d = float(np.median(distortions))
    supernodes: dict[int, list[int]] = {}
    ccc: list[int] = []
    for ti, lb in zip(valid_tracks, labels):
        if distortions[int(lb)] <= thresh_d:
            supernodes.setdefault(int(lb), []).append(int(ti))
        else:
            ccc.append(int(ti))

    for _ in range(cfg.max_iters):
        if not ccc:
            break
        sizes_before = {sid: len(v) for sid, v in supernodes.items()}
        for cand in list(ccc):
            if cand not in track_paths:
                continue
            cand_emb = X[ti_to_row[cand]]
            best_sid, best_d = None, np.inf
            for sid, members in supernodes.items():
                member_embs = np.stack([X[ti_to_row[m]] for m in members])
                d = float(np.min(np.linalg.norm(member_embs - cand_emb, axis=1)))
                if d < best_d:
                    best_d, best_sid = d, sid
            if best_sid is not None and best_d < cfg.knn_thresh:
                supernodes[best_sid].append(cand)
                ccc.remove(cand)
        for sid, members in supernodes.items():
            growth = (len(members) - sizes_before.get(sid, 0)) / max(sizes_before.get(sid, 1), 1)
            if growth < SUPERNODE_GROWTH_THRESH:
                continue

    track_to_cluster = {}
    clusters: dict[str, list[int]] = {}
    for sid, members in supernodes.items():
        clusters[str(sid)] = members
        for m in members:
            track_to_cluster[int(m)] = int(sid)
    for cand in ccc:
        track_to_cluster[int(cand)] = int(cand) + 10000
        clusters[str(cand + 10000)] = [int(cand)]

    payload = {
        "n_clusters": len(clusters),
        "track_to_cluster": {str(k): int(v) for k, v in track_to_cluster.items()},
        "clusters": clusters,
        "embedder": "FaceNet-vggface2",
        "n_valid_tracks": len(valid_tracks),
        "n_ccc_isolated": len(ccc),
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return out_path
