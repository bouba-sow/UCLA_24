"""Stage 1 — PySceneDetect + YOLOv3 + SORT (Zhang Supplementary Methods stage 1)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from zhang2023_constants import FRAME_SUBSAMPLE, SCENE_DETECT_THRESHOLD

from .io import crop_bbox, iter_frames, probe_video
from .sort_tracker import SORTTracker
from .yolo_detect import YOLOv3Detector


@dataclass
class Stage1Config:
    frame_step: int = FRAME_SUBSAMPLE
    max_frames: int | None = None
    device: str = "cpu"
    save_crops: bool = True
    scene_thresh: float = SCENE_DETECT_THRESHOLD
    yolo_weights: str | Path | None = None


def detect_scene_cuts(video_path: Path, thresh: float) -> list[tuple[int, int]]:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=thresh))
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()
    if not scenes:
        meta = probe_video(video_path)
        return [(0, meta.n_frames)]
    return [(int(s[0].get_frames()), int(s[1].get_frames())) for s in scenes]


def run_stage1(video_path: Path, work_dir: Path, cfg: Stage1Config, *, force: bool = False) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_npz = work_dir / "stage1_tracks.npz"
    if out_npz.exists() and not force:
        return out_npz

    meta = probe_video(video_path)
    stop = cfg.max_frames if cfg.max_frames is not None else meta.n_frames
    cuts = detect_scene_cuts(video_path, cfg.scene_thresh)
    crop_dir = work_dir / "crops"
    if cfg.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLOv3Detector(device=cfg.device, weights=cfg.yolo_weights)
    all_records: list[dict] = []
    global_id = 0

    for cut_i, (cut_start, cut_end) in enumerate(cuts):
        cut_stop = min(cut_end, stop)
        if cut_start >= cut_stop:
            continue
        tracker = SORTTracker()
        cut_tracks: dict[int, dict] = {}

        for frame_idx, frame in iter_frames(
            video_path, start=cut_start, stop=cut_stop, step=cfg.frame_step
        ):
            dets = detector.detect(frame)
            active = tracker.update(dets, frame_idx)
            seen_ids = set()
            for tr in active:
                if tr.id in seen_ids:
                    continue
                seen_ids.add(tr.id)
                if tr.id not in cut_tracks:
                    cut_tracks[tr.id] = {
                        "cut": cut_i,
                        "sort_id": int(tr.id),
                        "frames": [],
                        "boxes": [],
                        "crop_paths": [],
                    }
                rec = cut_tracks[tr.id]
                box = tr.get_state()[0].astype(np.float32)
                rec["frames"].append(int(frame_idx))
                rec["boxes"].append(box)
                if cfg.save_crops:
                    crop = crop_bbox(frame, tuple(box.astype(int)))
                    if crop.size == 0:
                        rec["crop_paths"].append("")
                    else:
                        cp = crop_dir / f"cut{cut_i:04d}_id{tr.id:04d}_f{frame_idx:06d}.jpg"
                        cv2.imwrite(str(cp), crop)
                        rec["crop_paths"].append(str(cp))

        for rec in cut_tracks.values():
            if len(rec["frames"]) >= 1:
                rec["global_track"] = global_id
                global_id += 1
                all_records.append(rec)

    with open(work_dir / "stage1_meta.json", "w") as fh:
        json.dump({
            "video": str(video_path),
            "fps": meta.fps,
            "n_frames": meta.n_frames,
            "frame_step": cfg.frame_step,
            "n_tracks": len(all_records),
            "cuts": cuts,
            "detector": "YOLOv3",
            "tracker": "SORT",
        }, fh, indent=2)

    np.savez_compressed(
        out_npz,
        n_tracks=len(all_records),
        tracks_json=np.array([json.dumps(r) for r in all_records], dtype=object),
    )
    return out_npz


def load_stage1_tracks(path: Path) -> list[dict]:
    data = np.load(path, allow_pickle=True)
    return [json.loads(str(s)) for s in data["tracks_json"]]
