"""Character presence decoding — Zhang et al. 2023 (Sci. Rep. 13:651, 2023).

Reimplementation from scratch:
  - Sorted spike firing rates, ±1 s windows (60 frames)
  - 2-layer LSTM (128 units), KLD loss, DNK masked
  - Randomized 5-fold CV (70 / 10 / 20), F1 metric

Usage
-----
python code3_Zhang_2023/scripts/decode_characters.py \\
    --bids-dir data/bids --sub 572 --ses 01 \\
    --device cuda --epochs 100

Quick smoke test:
python code3_Zhang_2023/scripts/decode_characters.py --epochs 3 --device cpu
"""
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-dir", default="data/bids", type=Path)
    p.add_argument("--sub", default="572")
    p.add_argument("--ses", default="01")
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--n-folds", default=5, type=int)
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--hidden", default=128, type=int)
    p.add_argument("--half-win", default=30, type=int,
                   help="Half-window in frames (30 → ±1 s, T=60)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--output-dir", default="results/code3_Zhang_2023", type=Path)
    p.add_argument("--labels-csv", default=None, type=Path,
                   help="Vision pipeline CSV (default: char_* from BIDS events.tsv)")
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
    spike_dir = (
        bids_dir / "derivatives/spike-sorted" / f"sub-{args.sub}" / f"ses-{args.ses}" / "ieeg"
    )
    return events_tsv, spike_dir


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("code3_Zhang_2023 — character decoding (randomized 5-fold CV)")
    print("=" * 60)

    events_tsv, spike_dir = _resolve_paths(args)
    print(f"Events : {events_tsv}")
    print(f"Spikes : {spike_dir}")

    print("\n[1/4] Firing rates …")
    rates, channels = load_firing_rates(spike_dir, events_tsv)
    print(f"  shape {rates.shape}  ({len(channels)} channels)")

    print("[2/4] Labels (every 4th frame, ~18,900) …")
    if args.labels_csv is not None:
        labels, sample_idx = load_labels_from_csv(args.labels_csv, TOP4_CHARS, n_frames=rates.shape[0])
        print(f"  source: {args.labels_csv}")
    else:
        labels, sample_idx = load_labels(events_tsv, char_cols=TOP4_CHARS)
        print(f"  source: {events_tsv} (char_* subsampled)")
    print(f"  n_samples={len(labels)}")
    for k, c in enumerate(TOP4_CHARS):
        col = labels[:, k]
        print(f"  {c:<22} Yes={(col == 1).sum():6d}  No={(col == 0).sum():6d}  DNK={(col == 2).sum():6d}")

    print("\n[3/4] Shuffle baseline …")
    rng = np.random.default_rng(args.seed)
    shuffle_f1 = compute_shuffle_baseline(labels, TOP4_CHARS, n_repeats=20, rng=rng)
    for name, f1 in shuffle_f1.items():
        print(f"  {name:<22} {f1:.4f}")

    print(f"\n[4/4] {args.n_folds}-fold CV …")
    cfg = TrainConfig(
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden,
        half_win=args.half_win,
        seed=args.seed,
        device=args.device,
        char_cols=TOP4_CHARS,
    )
    summary = run_cross_validation(rates, labels, sample_idx, cfg)
    summary["shuffle_baseline_f1"] = shuffle_f1
    summary["channels"] = channels
    summary["char_cols"] = TOP4_CHARS
    summary["n_samples"] = len(labels)
    summary["window_frames"] = 2 * args.half_win

    print("\n" + "=" * 60)
    print(f"Macro F1: {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
    for c in TOP4_CHARS:
        mu = summary["per_char_f1_mean"][c]
        sd = summary["per_char_f1_std"][c]
        sh = shuffle_f1[c]
        print(f"  {c:<22} {mu:.4f} ± {sd:.4f}  (shuffle {sh:.4f})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sub-{args.sub}_ses-{args.ses}_zhang2023_results.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
