"""Character presence decoding from iEEG spikes — Zhang et al. 2023 pipeline.

Trains a 2-layer LSTM on windowed per-frame firing rates and evaluates with
5-fold cross-validation on the movie viewing session.

Usage
-----
python code/scripts/decode_characters.py \\
    --bids-dir data/bids \\
    --sub 572 --ses 01 \\
    --epochs 100 --n-folds 5 \\
    --output-dir results/decoding

Add --device cuda  to run on GPU.
Add --epochs 5     for a quick smoke-test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow imports from code/src without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decoding.data import TOP4_CHARS, load_firing_rates, load_labels
from decoding.train import TrainConfig, compute_shuffle_baseline, run_cross_validation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-dir",   default="data/bids", type=Path)
    p.add_argument("--sub",        default="572")
    p.add_argument("--ses",        default="01")
    p.add_argument("--epochs",     default=100, type=int)
    p.add_argument("--n-folds",    default=5,   type=int)
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--lr",         default=1e-3, type=float)
    p.add_argument("--hidden",     default=128, type=int)
    p.add_argument("--half-win",   default=30,  type=int,
                   help="Half-window in frames (default 30 → ±1 s at 30 fps)")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--seed",       default=42,  type=int)
    p.add_argument("--output-dir", default="results/decoding", type=Path)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        bids_dir
        / "derivatives/spike-sorted"
        / f"sub-{args.sub}"
        / f"ses-{args.ses}"
        / "ieeg"
    )
    return events_tsv, spike_dir


def _print_data_summary(
    rates: np.ndarray,
    labels: np.ndarray,
    channels: list[str],
    char_cols: list[str],
) -> None:
    n_frames, n_ch = rates.shape
    print(f"Loaded {n_ch} channels, {n_frames} frames")
    print(f"Mean firing rate per channel: "
          f"{rates.mean(axis=0).mean():.4f} spikes/bin  "
          f"(range {rates.mean(axis=0).min():.4f}–{rates.mean(axis=0).max():.4f})")
    print(f"\nCharacter label distribution:")
    for k, c in enumerate(char_cols):
        col = labels[:, k]
        n_yes = (col == 1).sum()
        n_dnk = (col == 2).sum()
        n_no  = (col == 0).sum()
        print(f"  {c:<22} Yes={n_yes:6d} ({100*n_yes/n_frames:.1f}%)  "
              f"No={n_no:6d}  DNK={n_dnk}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Character decoding  —  Zhang et al. 2023 pipeline")
    print("=" * 60)

    events_tsv, spike_dir = _resolve_paths(args)
    print(f"Events TSV : {events_tsv}")
    print(f"Spike dir  : {spike_dir}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/4] Loading firing rates …")
    rates, channels = load_firing_rates(spike_dir, events_tsv)

    print("[2/4] Building character labels …")
    labels = load_labels(events_tsv, char_cols=TOP4_CHARS)

    _print_data_summary(rates, labels, channels, TOP4_CHARS)

    # ── Shuffle baseline ─────────────────────────────────────────────────────
    print("\n[3/4] Computing shuffle baseline …")
    rng = np.random.default_rng(args.seed)
    shuffle_f1 = compute_shuffle_baseline(labels, TOP4_CHARS, n_repeats=20, rng=rng)
    print("  Shuffle baseline F1 per character:")
    for name, f1 in shuffle_f1.items():
        print(f"    {name:<22} {f1:.4f}")

    # ── Cross-validation ─────────────────────────────────────────────────────
    print(f"\n[4/4] Running {args.n_folds}-fold cross-validation …")
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
    summary = run_cross_validation(rates, labels, cfg)
    summary["shuffle_baseline_f1"] = shuffle_f1
    summary["channels"] = channels
    summary["char_cols"] = TOP4_CHARS

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Results summary")
    print("=" * 60)
    print(f"Macro F1 across folds: {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
    print("\nPer-character F1 (mean ± std across folds):")
    for c in TOP4_CHARS:
        mu = summary["per_char_f1_mean"][c]
        sd = summary["per_char_f1_std"][c]
        sh = shuffle_f1[c]
        print(f"  {c:<22} {mu:.4f} ± {sd:.4f}  (shuffle: {sh:.4f})")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"sub-{args.sub}_ses-{args.ses}_decoding_results.json"
    with open(results_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
