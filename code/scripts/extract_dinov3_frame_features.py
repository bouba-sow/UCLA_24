#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


TRUE_FPS_FROM_README = 29.97002997002997


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_model(repo: str, model_name: str, source: str, weights: str, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load(repo, model_name, source=source, weights=weights)
    model.eval()
    model.to(device)
    return model


def resolve_embedding(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        out = model_output
    elif isinstance(model_output, dict):
        for key in ("x_norm_clstoken", "cls_token", "x_norm_patchtokens", "features"):
            if key in model_output and isinstance(model_output[key], torch.Tensor):
                out = model_output[key]
                break
        else:
            tensor_values = [v for v in model_output.values() if isinstance(v, torch.Tensor)]
            if not tensor_values:
                raise RuntimeError("Model output dict did not contain a tensor feature.")
            out = tensor_values[0]
    elif isinstance(model_output, (list, tuple)) and model_output and isinstance(model_output[0], torch.Tensor):
        out = model_output[0]
    else:
        raise RuntimeError(f"Unsupported model output type: {type(model_output)}")

    if out.ndim == 3:
        out = out.mean(dim=1)
    if out.ndim != 2:
        raise RuntimeError(f"Expected 2D embedding tensor, got shape {tuple(out.shape)}")
    return out


def write_metadata(
    metadata_path: Path,
    video_path: Path,
    n_frames: int,
    video_fps: float,
    output_dim: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "video_path": str(video_path),
        "n_frames": int(n_frames),
        "video_fps_opencv": float(video_fps),
        "true_fps_from_readme": float(args.true_fps),
        "model_repo": args.model_repo,
        "model_name": args.model_name,
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "layer_index": int(args.layer_index),
        "total_layers": int(args.total_layers),
        "feature_dim": int(output_dim),
        "device": args.device,
    }
    metadata_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-frame DINOv3 features and save one .npy per frame.")
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("data/40m_act_24_S06E01_30fps.m4v"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features/dinov3_vithplus_distilled/s06e01/frame_npy"),
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("data/features/dinov3_vithplus_distilled/s06e01/frame_npy_paths.csv"),
    )
    parser.add_argument(
        "--manifest-txt",
        type=Path,
        default=Path("data/features/dinov3_vithplus_distilled/s06e01/frame_npy_paths.txt"),
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=Path("data/features/dinov3_vithplus_distilled/s06e01/extraction_metadata.json"),
    )
    parser.add_argument("--model-repo", type=str, default="dinov3")
    parser.add_argument("--model-name", type=str, default="dinov3_vith16plus")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument(
        "--layer-index",
        type=int,
        default=None,
        help="Transformer block index to extract (0-based). Defaults to middle layer.",
    )
    parser.add_argument("--true-fps", type=float, default=TRUE_FPS_FROM_README)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.video_path.exists():
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_txt.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    transform = build_transform(args.image_size)
    source = "local"
    weights= "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
    model = load_model(args.model_repo, args.model_name, source, weights, device)
    if not hasattr(model, "get_intermediate_layers") or not hasattr(model, "blocks"):
        raise RuntimeError("Loaded model does not support intermediate layer extraction.")
    total_layers = len(model.blocks)
    args.total_layers = total_layers
    if args.layer_index is None:
        args.layer_index = total_layers // 2
    if args.layer_index < 0 or args.layer_index >= total_layers:
        raise ValueError(f"--layer-index must be in [0, {total_layers - 1}], got {args.layer_index}")
    print(f"Using intermediate layer {args.layer_index} of {total_layers} total layers.")

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps > 0 and abs(video_fps - args.true_fps) > 0.05:
        print(f"Warning: OpenCV fps={video_fps:.6f} differs from README fps={args.true_fps:.6f}")

    manifest_rows: list[dict[str, Any]] = []
    batch_tensors: list[torch.Tensor] = []
    batch_frame_indices: list[int] = []
    frame_idx = 0
    output_dim = -1

    def flush_batch() -> None:
        nonlocal output_dim
        if not batch_tensors:
            return
        with torch.inference_mode():
            inputs = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
            layer_outputs = model.get_intermediate_layers(
                inputs,
                n=[args.layer_index],
                return_class_token=True,
                norm=True,
            )
            if not layer_outputs:
                raise RuntimeError("Model returned no intermediate layer outputs.")
            _, cls_token = layer_outputs[0]
            embeddings = cls_token.detach().cpu().numpy().astype(np.float32)
        output_dim = int(embeddings.shape[1])

        for local_i, idx in enumerate(batch_frame_indices):
            out_path = args.output_dir / f"frame_{idx:06d}.npy"
            np.save(out_path, embeddings[local_i], allow_pickle=False)
            manifest_rows.append(
                {
                    "frame_idx": idx,
                    "time_sec_video_fps": (idx / video_fps) if video_fps > 0 else np.nan,
                    "time_sec_true_fps": idx / args.true_fps,
                    "npy_path": str(out_path),
                }
            )
        batch_tensors.clear()
        batch_frame_indices.clear()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        batch_tensors.append(transform(image))
        batch_frame_indices.append(frame_idx)
        frame_idx += 1

        if len(batch_tensors) >= args.batch_size:
            flush_batch()
            if frame_idx % 1000 == 0:
                print(f"Processed {frame_idx} frames...")

    flush_batch()
    cap.release()

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(args.manifest_csv, index=False)
    args.manifest_txt.write_text("\n".join(manifest_df["npy_path"].tolist()) + "\n")
    write_metadata(args.metadata_json, args.video_path, frame_idx, video_fps, output_dim, args)

    print(f"Done. Total frames processed: {frame_idx}")
    print(f"Per-frame .npy directory: {args.output_dir}")
    print(f"Manifest CSV: {args.manifest_csv}")
    print(f"Manifest TXT: {args.manifest_txt}")
    print(f"Metadata JSON: {args.metadata_json}")


if __name__ == "__main__":
    main()
