"""Character decoding — Zhang protocol, temporal CV without window leakage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from decoding.data import TOP4_CHARS, load_firing_rates, load_labels, load_labels_from_csv
from decoding.train import TrainConfig, compute_shuffle_baseline, run_cross_validation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bids-dir", default="data/bids", type=Path)
    p.add_argument("--sub", default="572")
    p.add_argument("--ses", default="01")
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--n-folds", default=5, type=int)
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--output-dir", default="results/code4_Zhang_2023_bis", type=Path)
    p.add_argument("--labels-csv", default=None, type=Path)
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
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
    print("=" * 60)
    print("code4_Zhang_2023_bis — temporal block CV (no window leakage)")
    print("=" * 60)

    events_tsv, spike_dir = _resolve_paths(args)
    rates, units = load_firing_rates(spike_dir, events_tsv)
    print(f"Spikes: {rates.shape}  ({len(units)} units)")

    if args.labels_csv is not None:
        labels, sample_frames = load_labels_from_csv(args.labels_csv, TOP4_CHARS, n_frames=rates.shape[0])
        print(f"Labels: {args.labels_csv}")
    else:
        labels, sample_frames = load_labels(events_tsv, TOP4_CHARS)
        print(f"Labels: {events_tsv}")
    print(f"n_samples={len(labels)}")

    rng = np.random.default_rng(args.seed)
    shuffle_f1 = compute_shuffle_baseline(labels, TOP4_CHARS, n_repeats=20, rng=rng)

    cfg = TrainConfig(
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        char_cols=TOP4_CHARS,
    )
    summary = run_cross_validation(rates, labels, sample_frames, cfg)
    summary["shuffle_baseline_f1"] = shuffle_f1
    summary["units"] = units
    summary["char_cols"] = TOP4_CHARS

    print(f"\nMacro AUC: {summary['macro_auc_mean']:.4f} ± {summary['macro_auc_std']:.4f}  (chance 0.5)")
    print(f"Macro F1 : {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
    for c in TOP4_CHARS:
        print(f"  {c}: AUC={summary['per_char_auc_mean'][c]:.4f}  F1={summary['per_char_f1_mean'][c]:.4f}  (shuffle F1 {shuffle_f1[c]:.4f})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sub-{args.sub}_ses-{args.ses}_no_leakage_results.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
