"""Single-unit selectivity test: does any neuron fire differently when a
character is present vs absent?

For each (unit, character):
  - feature = summed spike count in a ±half_win window centred on each
    subsampled frame (matches the decoder's ±1 s input).
  - label   = YES / NO (DNK frames excluded).
  - statistic = ROC AUC of firing rate discriminating present vs absent.
  - null = circular shifts of the label series (preserves temporal
    autocorrelation of both spikes and labels — a plain shuffle would be
    anti-conservative).
  - two-sided p from |AUC - 0.5|; Benjamini-Hochberg FDR per character.

This says whether decoding is even possible before blaming the decoder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from constants import HALF_WINDOW_FRAMES
from decoding.data import (
    NO,
    YES,
    TOP4_CHARS,
    load_firing_rates,
    load_labels,
    load_labels_from_csv,
)


def auc_fast(scores: np.ndarray, pos_mask: np.ndarray) -> float:
    """ROC AUC via rank-sum (Mann-Whitney). scores: (N,), pos_mask: bool (N,)."""
    n_pos = int(pos_mask.sum())
    n_neg = int((~pos_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores)
    sum_pos = ranks[pos_mask].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def window_sum(rates: np.ndarray, frames: np.ndarray, half_win: int) -> np.ndarray:
    """Sum counts in [f-half_win, f+half_win) per unit. Returns (n_samples, n_units)."""
    csum = np.cumsum(rates, axis=0)
    csum = np.vstack([np.zeros((1, rates.shape[1])), csum])  # prepend 0 row
    lo = np.clip(frames - half_win, 0, rates.shape[0])
    hi = np.clip(frames + half_win, 0, rates.shape[0])
    return csum[hi] - csum[lo]


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: return boolean mask of significant tests."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.zeros_like(p, dtype=bool)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out
    pv = p[idx]
    order = np.argsort(pv)
    m = pv.size
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = pv[order] <= thresh
    if not passed.any():
        return out
    kmax = np.where(passed)[0].max()
    sig_sorted = np.zeros(m, dtype=bool)
    sig_sorted[order[: kmax + 1]] = True
    out[idx] = sig_sorted
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bids-dir", default="data/bids", type=Path)
    p.add_argument("--sub", default="572")
    p.add_argument("--ses", default="01")
    p.add_argument("--labels-csv", default="data/40m_act_24_S06E01_30fps_characters.csv", type=Path)
    p.add_argument("--half-win", default=HALF_WINDOW_FRAMES, type=int)
    p.add_argument("--n-perm", default=1000, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--output", default="results/code4_Zhang_2023_bis/sub-572_unit_selectivity.json", type=Path)
    args = p.parse_args()

    ieeg_dir = args.bids_dir / f"sub-{args.sub}" / f"ses-{args.ses}" / "ieeg"
    events_tsv = next(ieeg_dir.glob(f"sub-{args.sub}_ses-{args.ses}_task-movie24presleep_acq-micro_run-01_events.tsv"))
    spike_dir = args.bids_dir / "derivatives/spike-sorted" / f"sub-{args.sub}" / f"ses-{args.ses}" / "ieeg"

    rates, units = load_firing_rates(spike_dir, events_tsv)
    print(f"rates {rates.shape}  ({len(units)} units)")

    if args.labels_csv and Path(args.labels_csv).exists():
        labels, frames = load_labels_from_csv(args.labels_csv, TOP4_CHARS, n_frames=rates.shape[0])
        src = str(args.labels_csv)
    else:
        labels, frames = load_labels(events_tsv, TOP4_CHARS)
        src = str(events_tsv)
    print(f"labels {labels.shape} from {src}")

    feats = window_sum(rates, frames, args.half_win)  # (n_samples, n_units)
    n_units = feats.shape[1]
    rng = np.random.default_rng(args.seed)

    summary: dict = {"sub": args.sub, "labels": src, "half_win": args.half_win,
                     "n_perm": args.n_perm, "units": units, "characters": {}}

    for k, char in enumerate(TOP4_CHARS):
        col = labels[:, k]
        keep = col != 2  # drop DNK
        y = (col[keep] == YES)
        F = feats[keep]
        n_pos = int(y.sum())
        print(f"\n{char}: n_pos={n_pos}  n_neg={int((~y).sum())}")

        obs = np.array([auc_fast(F[:, u], y) for u in range(n_units)])

        # circular-shift null (shift label vector, keep feature order)
        null_max_dev = np.zeros((args.n_perm, n_units))
        N = len(y)
        for pi in range(args.n_perm):
            shift = int(rng.integers(1, N))
            y_shift = np.roll(y, shift)
            null_max_dev[pi] = [auc_fast(F[:, u], y_shift) for u in range(n_units)]

        dev_obs = np.abs(obs - 0.5)
        dev_null = np.abs(null_max_dev - 0.5)
        pvals = (1.0 + (dev_null >= dev_obs[None, :]).sum(axis=0)) / (args.n_perm + 1.0)

        sig = bh_fdr(pvals, alpha=0.05)
        order = np.argsort(-dev_obs)
        top = [
            {"unit": units[u], "auc": float(obs[u]), "p": float(pvals[u]), "fdr_sig": bool(sig[u])}
            for u in order[:10]
        ]
        n_sig = int(sig.sum())
        print(f"  significant units (FDR<0.05): {n_sig}/{n_units}")
        print(f"  best |AUC-0.5|: {units[order[0]]} AUC={obs[order[0]]:.3f} p={pvals[order[0]]:.4f}")

        summary["characters"][char] = {
            "n_pos": n_pos,
            "n_neg": int((~y).sum()),
            "n_sig_fdr": n_sig,
            "max_abs_auc_dev": float(dev_obs[order[0]]),
            "top_units": top,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved → {args.output}")

    print("\n=== SUMMARY ===")
    for char in TOP4_CHARS:
        c = summary["characters"][char]
        print(f"  {char:<20} sig={c['n_sig_fdr']:3d}/{n_units}  best|AUC-.5|={c['max_abs_auc_dev']:.3f}")


if __name__ == "__main__":
    main()
