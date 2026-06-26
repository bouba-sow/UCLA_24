"""Video frame I/O at native FPS."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: Path
    fps: float
    n_frames: int
    width: int
    height: int


def probe_video(path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoMeta(path=path, fps=fps, n_frames=n_frames, width=width, height=height)


def iter_frames(
    path: Path,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
):
    """Yield (frame_idx, BGR image)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    meta = probe_video(path)
    if stop is None:
        stop = meta.n_frames
    idx = start
    while idx < stop:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += step
    cap.release()


def crop_bbox(frame: np.ndarray, box: tuple[int, int, int, int], pad: float = 0.05) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw))
    y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw))
    y2 = min(h, int(y2 + pad * bh))
    return frame[y1:y2, x1:x2].copy()
