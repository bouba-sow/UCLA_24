"""Run Zhang 2023 vision pipeline (strict: YOLOv3 + SORT + FaceNet + ResNet)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vision.pipeline import VisionPipelineConfig, run_vision_pipeline
from vision.stage1_detect_track import Stage1Config
from vision.stage2_cluster import Stage2Config
from vision.stage4_resnet import Stage4Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default="data/40m_act_24_S06E01_30fps.m4v", type=Path)
    p.add_argument("--work-dir", default="results/code3_Zhang_2023/vision", type=Path)
    p.add_argument("--output-csv", default=None, type=Path)
    p.add_argument("--cluster-assignments", required=True, type=Path,
                   help="Stage 3: cluster_id → character name JSON (human supervision)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-frames", default=None, type=int)
    p.add_argument("--bootstrap-from-reference", default=None, type=Path,
                   help="NON-PAPER: auto-map clusters from reference CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VisionPipelineConfig(
        video=args.video,
        work_dir=args.work_dir,
        cluster_assignments=args.cluster_assignments,
        output_csv=args.output_csv,
        bootstrap_from_reference=args.bootstrap_from_reference,
        stage1=Stage1Config(device=args.device, max_frames=args.max_frames),
        stage2=Stage2Config(device=args.device),
        stage4=Stage4Config(device=args.device),
    )
    out = run_vision_pipeline(cfg)
    print(f"Labels → {out}")


if __name__ == "__main__":
    main()
