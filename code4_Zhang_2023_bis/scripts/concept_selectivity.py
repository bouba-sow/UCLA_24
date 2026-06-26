"""Single-unit selectivity for the 8 Ding concepts during MOVIE VIEWING.

If concept cells exist (Ding et al.), some units should fire differently when
a concept is on screen — even during viewing, before any recall data.

Vectorised circular-shift permutation test (fast: 10k perms in seconds):
  - feature  = summed spike count in a ±win_sec window per second per unit
  - label    = concept present / absent (1 Hz, 8 concepts)
  - stat     = ROC AUC (rank-sum)
  - null     = circular shifts of the label vector (preserves autocorrelation)
  - FDR (Benjamini-Hochberg) across units, per concept
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from decoding.data import load_firing_rates

CONCEPTS = ["WhiteHouse", "CTU", "Hostage", "Handcuff",
            "J.Bauer", "B.Buchanan", "A.Fayed", "A.Amar"]
FPS = 29.97002997002997


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals, float)
    out = np.zeros_like(p, dtype=bool)
    order = np.argsort(p)
    m = p.size
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    if passed.any():
        kmax = np.where(passed)[0].max()
        out[order[: kmax + 1]] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bids-dir", default="data/bids", type=Path)
    ap.add_argument("--sub", default="572")
    ap.add_argument("--ses", default="01")
    ap.add_argument("--concepts-csv", default="data/24_S06E01_8concepts_merged.csv", type=Path)
    ap.add_argument("--win-sec", default=1.0, type=float)
    ap.add_argument("--n-perm", default=10000, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--output", default="results/code4_Zhang_2023_bis/sub-572_concept_selectivity.json", type=Path)
    args = ap.parse_args()

    ieeg_dir = args.bids_dir / f"sub-{args.sub}" / f"ses-{args.ses}" / "ieeg"
    events_tsv = next(ieeg_dir.glob(f"sub-{args.sub}_ses-{args.ses}_task-movie24presleep_acq-micro_run-01_events.tsv"))
    spike_dir = args.bids_dir / "derivatives/spike-sorted" / f"sub-{args.sub}" / f"ses-{args.ses}" / "ieeg"

    rates, units = load_firing_rates(spike_dir, events_tsv)  # (n_frames, n_units) @ 30 Hz
    n_frames, n_units = rates.shape
    print(f"rates {rates.shape}  ({n_units} units)")

    df = pd.read_csv(args.concepts_csv)
    labels = (df[CONCEPTS].fillna(0).values > 0.5)  # (T_sec, 8)
    n_sec = labels.shape[0]
    print(f"concepts {labels.shape}")

    # window-sum spikes per second
    csum = np.vstack([np.zeros((1, n_units)), np.cumsum(rates, axis=0)])
    hw = int(round(args.win_sec * FPS / 2))
    centers = np.round(np.arange(n_sec) * FPS).astype(int)
    lo = np.clip(centers - hw, 0, n_frames)
    hi = np.clip(centers + hw, 0, n_frames)
    feats = csum[hi] - csum[lo]  # (n_sec, n_units)

    # precompute ranks per unit once (shared across permutations)
    ranks = np.column_stack([rankdata(feats[:, u]) for u in range(n_units)])  # (n_sec, n_units)

    rng = np.random.default_rng(args.seed)
    shifts = rng.integers(1, n_sec, size=args.n_perm)

    summary = {"sub": args.sub, "win_sec": args.win_sec, "n_perm": args.n_perm,
               "n_sec": n_sec, "units": units, "concepts": {}}

    for k, concept in enumerate(CONCEPTS):
        y = labels[:, k]
        n_pos, n_neg = int(y.sum()), int((~y).sum())
        pos_idx = np.where(y)[0]

        # observed AUC per unit
        sum_pos_obs = ranks[pos_idx].sum(axis=0)  # (n_units,)
        auc_obs = (sum_pos_obs - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

        # null: circular shifts of label → shifted positive indices
        shifted = (pos_idx[None, :] + shifts[:, None]) % n_sec  # (P, n_pos)
        # sum ranks over shifted positives, per unit
        # gather then sum: (P, n_pos, n_units) is big; loop units to bound memory
        dev_obs = np.abs(auc_obs - 0.5)
        pvals = np.empty(n_units)
        for u in range(n_units):
            sp = ranks[shifted, u].sum(axis=1)  # (P,)
            auc_p = (sp - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            dev_p = np.abs(auc_p - 0.5)
            pvals[u] = (1.0 + (dev_p >= dev_obs[u]).sum()) / (args.n_perm + 1.0)

        sig = bh_fdr(pvals, 0.05)
        order = np.argsort(-dev_obs)
        top = [{"unit": units[u], "auc": float(auc_obs[u]), "p": float(pvals[u]),
                "fdr_sig": bool(sig[u])} for u in order[:8]]
        print(f"{concept:<12} pos={n_pos:4d}  sig(FDR)={int(sig.sum()):3d}/{n_units}"
              f"  best {units[order[0]]} AUC={auc_obs[order[0]]:.3f} p={pvals[order[0]]:.4f}")
        summary["concepts"][concept] = {
            "n_pos": n_pos, "n_sig_fdr": int(sig.sum()),
            "best_auc": float(auc_obs[order[0]]), "top_units": top,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
