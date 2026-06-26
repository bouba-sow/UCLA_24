"""YOLOv3 person detection with original Darknet weights (Redmon & Farhadi 2018).

Uses vendored PyTorch-YOLOv3 (eriklindernoren, MIT) + pjreddie yolov3.weights.
This matches Zhang et al. 2023 Supplementary Methods stage 1b(i).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from zhang2023_constants import (
    MAX_ASPECT_RATIO,
    MIN_BOX_AREA,
    YOLO_CONF_THRESH,
    YOLO_IMG_SIZE,
    YOLO_IOU_THRESH,
    YOLO_PERSON_CLASS,
)

_VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "pytorch_yolov3"
_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
_DEFAULT_CFG = _VENDOR / "config" / "yolov3.cfg"
_DEFAULT_WEIGHTS = _WEIGHTS_DIR / "yolov3.weights"


def default_weights_path() -> Path:
    return _DEFAULT_WEIGHTS


def ensure_yolov3_weights(path: Path | None = None) -> Path:
    path = path or _DEFAULT_WEIGHTS
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Darknet YOLOv3 weights not found at {path}.\n"
        "Run: bash code3_Zhang_2023/scripts/download_yolov3_weights.sh"
    )


def _prep_tensor(frame_bgr: np.ndarray, img_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad to square + resize to img_size (Darknet letterbox-style)."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    dim = max(h, w)
    canvas = np.full((dim, dim, 3), 128, dtype=np.uint8)
    y0, x0 = (dim - h) // 2, (dim - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = img
    resized = cv2.resize(canvas, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0)
    return tensor, (h, w)


class YOLOv3Detector:
    """Original Darknet YOLOv3 — not Ultralytics."""

    def __init__(
        self,
        device: str = "cpu",
        weights: str | Path | None = None,
        cfg: str | Path | None = None,
        img_size: int = YOLO_IMG_SIZE,
    ) -> None:
        if str(_VENDOR) not in sys.path:
            sys.path.insert(0, str(_VENDOR))

        from pytorchyolo.models import Darknet
        from pytorchyolo.utils.utils import non_max_suppression, rescale_boxes

        self._non_max_suppression = non_max_suppression
        self._rescale_boxes = rescale_boxes
        self.device = torch.device(device)
        self.img_size = img_size

        cfg_path = Path(cfg) if cfg else _DEFAULT_CFG
        weights_path = ensure_yolov3_weights(Path(weights) if weights else None)

        self.model = Darknet(str(cfg_path))
        self.model.load_darknet_weights(str(weights_path))
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return Nx5 [x1, y1, x2, y2, score] for person (COCO class 0)."""
        tensor, orig_shape = _prep_tensor(frame_bgr, self.img_size)
        tensor = tensor.unsqueeze(0).to(self.device)

        raw = self.model(tensor)
        detections = self._non_max_suppression(
            raw,
            conf_thres=YOLO_CONF_THRESH,
            iou_thres=YOLO_IOU_THRESH,
            classes=[YOLO_PERSON_CLASS],
        )[0]

        if detections is None or len(detections) == 0:
            return np.empty((0, 5), dtype=np.float32)

        detections = self._rescale_boxes(detections, self.img_size, orig_shape)
        out = detections[:, :5].cpu().numpy().astype(np.float32)
        keep = [i for i, d in enumerate(out) if _valid_box(d[:4])]
        return out[keep] if keep else np.empty((0, 5), dtype=np.float32)


def _valid_box(box: np.ndarray) -> bool:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w * h < MIN_BOX_AREA:
        return False
    ar = max(w / (h + 1e-6), h / (w + 1e-6))
    return ar <= MAX_ASPECT_RATIO
