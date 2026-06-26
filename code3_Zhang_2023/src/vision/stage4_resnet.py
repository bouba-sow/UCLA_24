"""Stage 4 — 10-class ResNet on all subsampled frames (Zhang stage 4)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from zhang2023_constants import (
    FRAME_SUBSAMPLE,
    RESNET_EPOCHS,
    RESNET_NUM_CLASSES,
    RESNET_PROB_THRESH,
)

from .io import crop_bbox, iter_frames, probe_video
from .stage1_detect_track import load_stage1_tracks
from .yolo_detect import YOLOv3Detector


@dataclass
class Stage4Config:
    epochs: int = RESNET_EPOCHS
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "cpu"
    prob_thresh: float = RESNET_PROB_THRESH
    frame_step: int = FRAME_SUBSAMPLE
    yolo_weights: str | Path | None = None


class CropCharDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], tfm):
        self.samples = samples
        self.tfm = tfm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Unreadable crop: {path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tfm(rgb), label


def _build_samples(tracks: list[dict], track_to_char: dict[int, str], class_names: list[str]):
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    other_idx = name_to_idx["Other"]
    samples: list[tuple[str, int]] = []
    for ti, tr in enumerate(tracks):
        label = name_to_idx.get(track_to_char.get(ti, ""), other_idx)
        for p in tr.get("crop_paths", []):
            if p:
                samples.append((p, label))
    return samples


def _train_resnet(samples, n_classes: int, cfg: Stage4Config):
    train_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(
        CropCharDataset(samples, train_tfm),
        batch_size=min(cfg.batch_size, len(samples)),
        shuffle=True,
        num_workers=0,
    )
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model = model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(cfg.epochs):
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
    model.eval()
    return model, eval_tfm


@torch.inference_mode()
def _predict_crop(model, tfm, crop_bgr, device):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = tfm(rgb).unsqueeze(0).to(device)
    return torch.softmax(model(x), dim=1).cpu().numpy().ravel()


def run_stage4(
    video_path: Path,
    tracks_npz: Path,
    stage3_json: Path,
    class_names: list[str],
    work_dir: Path,
    cfg: Stage4Config,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_npz = work_dir / "stage4_frame_char_probs.npz"
    if out_npz.exists():
        return out_npz

    tracks = load_stage1_tracks(tracks_npz)
    with open(stage3_json) as fh:
        s3 = json.load(fh)
    track_to_char = {int(k): v for k, v in s3["track_to_character"].items()}

    samples = _build_samples(tracks, track_to_char, class_names)
    if len(samples) < 2:
        raise RuntimeError("Stage 4: insufficient labeled crops from stage 3")

    model, eval_tfm = _train_resnet(samples, len(class_names), cfg)
    detector = YOLOv3Detector(device=cfg.device, weights=cfg.yolo_weights)

    meta = probe_video(video_path)
    n_frames = meta.n_frames
    n_classes = len(class_names)
    frame_probs = np.zeros((n_frames, n_classes), dtype=np.float32)

    char_only = [c for c in class_names if c != "Other"]
    char_idx = [class_names.index(c) for c in char_only]

    for frame_idx, frame in iter_frames(video_path, start=0, stop=n_frames, step=cfg.frame_step):
        dets = detector.detect(frame)
        if dets.size == 0:
            continue
        probs_acc = np.zeros(n_classes, dtype=np.float32)
        n_det = 0
        for det in dets:
            crop = crop_bbox(frame, tuple(det[:4].astype(int)))
            if crop.size == 0:
                continue
            probs_acc += _predict_crop(model, eval_tfm, crop, cfg.device)
            n_det += 1
        if n_det:
            frame_probs[frame_idx] = probs_acc / n_det

    torch.save(model.state_dict(), work_dir / "stage4_resnet18_10class.pt")
    np.savez_compressed(
        out_npz,
        characters=np.array(class_names, dtype=object),
        frame_probs=frame_probs,
        prob_thresh=cfg.prob_thresh,
        frame_step=cfg.frame_step,
    )
    return out_npz
