"""Full Zhang 2023 pipeline: strict vision → iEEG decoding."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from decoding.data import TOP4_CHARS, load_firing_rates, load_labels_from_csv
from decoding.train import TrainConfig, compute_shuffle_baseline, run_cross_validation
from vision.pipeline import VisionPipelineConfig, run_vision_pipeline
from vision.stage1_detect_track import Stage1Config
from vision.stage2_cluster import Stage2Config
from vision.stage4_resnet import Stage4Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default="data/40m_act_24_S06E01_30fps.m4v", type=Path)
    p.add_argument("--work-dir", default="results/code3_Zhang_2023", type=Path)
    p.add_argument("--cluster-assignments", default=None, type=Path)
    p.add_argument("--skip-vision", action="store_true")
    p.add_argument("--labels-csv", default=None, type=Path)
    p.add_argument("--bids-dir", default="data/bids", type=Path)
    p.add_argument("--sub", default="572")
    p.add_argument("--ses", default="01")
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--n-folds", default=5, type=int)
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def _resolve_bids(args):
    bids_dir = Path(args.bids_dir)
    ieeg_dir = bids_dir / f"sub-{args.sub}" / f"ses-{args.ses}" / "ieeg"
    events_tsv = next(
        ieeg_dir.glob(f"sub-{args.sub}_ses-{args.ses}_task-movie24presleep_acq-micro_run-01_events.tsv"),
        None,
    )
    if events_tsv is None:
        raise FileNotFoundError(f"events.tsv not found in {ieeg_dir}")
    spike_dir = bids_dir / "derivatives/spike-sorted" / f"sub-{args.sub}" / f"ses-{args.ses}" / "ieeg"
    return events_tsv, spike_dir


def main() -> None:
    args = parse_args()
    work = Path(args.work_dir)
    vision_dir = work / "vision"
    labels_csv = args.labels_csv

    if not args.skip_vision and labels_csv is None:
        if args.cluster_assignments is None:
            raise SystemExit("--cluster-assignments required for vision (stage 3 human supervision)")
        labels_csv = run_vision_pipeline(VisionPipelineConfig(
            video=args.video,
            work_dir=vision_dir,
            cluster_assignments=args.cluster_assignments,
            output_csv=vision_dir / "characters_30fps.csv",
            stage1=Stage1Config(device=args.device),
            stage2=Stage2Config(device=args.device),
            stage4=Stage4Config(device=args.device),
        ))
    elif labels_csv is None:
        labels_csv = vision_dir / "characters_30fps.csv"
        if not labels_csv.exists():
            raise FileNotFoundError("No labels CSV — run vision first or pass --labels-csv")

    events_tsv, spike_dir = _resolve_bids(args)
    rates, channels = load_firing_rates(spike_dir, events_tsv)
    labels, sample_idx = load_labels_from_csv(labels_csv, TOP4_CHARS, n_frames=rates.shape[0])

    rng = np.random.default_rng(args.seed)
    shuffle_f1 = compute_shuffle_baseline(labels, TOP4_CHARS, n_repeats=20, rng=rng)
    cfg = TrainConfig(
        n_folds=args.n_folds, epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, device=args.device, char_cols=TOP4_CHARS,
    )
    summary = run_cross_validation(rates, labels, sample_idx, cfg)
    summary["labels_csv"] = str(labels_csv)
    summary["shuffle_baseline_f1"] = shuffle_f1
    summary["channels"] = channels

    print(f"\nMacro F1: {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
    out_path = work / f"sub-{args.sub}_ses-{args.ses}_full_pipeline_results.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
