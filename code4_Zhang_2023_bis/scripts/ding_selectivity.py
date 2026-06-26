"""Replicate Ding et al. 2025 character-selectivity test on sub-572.

Ding Methods ('Character selectivity analysis'):
  - A character appearance is included only if the character appeared ALONE
    onscreen for >= 1 second.
  - Per unit: paired t-test of firing rate in a 1 s pre-onset window vs a
    1 s post-onset window, across appearances.
  - Surrogate: randomly flip the sign of each appearance's pre->post
    difference (n=1000) to build a t-statistic null.
  - A unit is SELECTIVE to a character iff:
      (1) it fires >= 1 spike in > 30% of post-onset windows,
      (2) observed t exceeds 99% of the permuted t (p < 0.01, one-tailed increase),
      (3) it is significant for exactly ONE character.
  - RESPONSIVE = (1)+(2) but may respond to >1 character.

Ding reports 37/626 ~ 6% selective units (single-character) across participants.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from decoding.data import load_firing_rates

# named characters in the 40m_act CSV (exclude Face/Person/No Characters)
NAMED = ["A.Amar", "A.Fayed", "B.Buchanan", "C.Manning", "C.OBrian", "J.Bauer",
         "J.Wallace", "K.Hayes", "M.OBrian", "M.Pressman", "N.Yassir",
         "R.Wallace", "S.Wallace", "T.Lennox", "W.Palmer"]
FPS = 29.97002997002997


def alone_onsets(present_alone: np.ndarray, win: int) -> list[int]:
    """Onsets of runs where the character is alone, lasting >= win frames,
    with a full pre-window available and the character not alone just before."""
    onsets = []
    n = len(present_alone)
    in_run = False
    run_start = 0
    for i in range(n):
        if present_alone[i] and not in_run:
            in_run, run_start = True, i
        elif not present_alone[i] and in_run:
            in_run = False
            if i - run_start >= win and run_start - win >= 0 and run_start + win <= n:
                onsets.append(run_start)
    if in_run and n - run_start >= win and run_start - win >= 0 and run_start + win <= n:
        onsets.append(run_start)
    return onsets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bids-dir", default="data/bids", type=Path)
    ap.add_argument("--sub", default="572")
    ap.add_argument("--ses", default="01")
    ap.add_argument("--char-csv", default="data/40m_act_24_S06E01_30fps_characters.csv", type=Path)
    ap.add_argument("--win-frames", default=30, type=int, help="1 s at 30 fps")
    ap.add_argument("--n-perm", default=1000, type=int)
    ap.add_argument("--alpha", default=0.01, type=float)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--output", default="results/code4_Zhang_2023_bis/sub-572_ding_selectivity.json", type=Path)
    args = ap.parse_args()

    ieeg_dir = args.bids_dir / f"sub-{args.sub}" / f"ses-{args.ses}" / "ieeg"
    events_tsv = next(ieeg_dir.glob(f"sub-{args.sub}_ses-{args.ses}_task-movie24presleep_acq-micro_run-01_events.tsv"))
    spike_dir = args.bids_dir / "derivatives/spike-sorted" / f"sub-{args.sub}" / f"ses-{args.ses}" / "ieeg"

    rates, units = load_firing_rates(spike_dir, events_tsv)  # (n_frames, n_units) @ 30 Hz
    n_frames, n_units = rates.shape
    print(f"rates {rates.shape}  ({n_units} units)")

    df = pd.read_csv(args.char_csv)
    named = [c for c in NAMED if c in df.columns]
    pres = {c: (pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy() > 0.5) for c in named}
    any_named = np.zeros(n_frames, dtype=int)
    for c in named:
        any_named[: len(pres[c])] += pres[c][:n_frames].astype(int)

    win = args.win_frames
    csum = np.vstack([np.zeros((1, n_units)), np.cumsum(rates, axis=0)])
    rng = np.random.default_rng(args.seed)

    # sig[char] = boolean (n_units,) ; collect for criteria 1&2
    sig_by_char: dict[str, np.ndarray] = {}
    detail: dict = {}

    for c in named:
        p = pres[c][:n_frames]
        alone = p & (any_named == 1)  # this char present and no other named char
        onsets = alone_onsets(alone, win)
        n_app = len(onsets)
        if n_app < 5:
            sig_by_char[c] = np.zeros(n_units, dtype=bool)
            detail[c] = {"n_appearances": n_app, "note": "too few alone appearances", "n_sig": 0}
            continue

        on = np.array(onsets)
        post = csum[on + win] - csum[on]      # (n_app, n_units)
        pre = csum[on] - csum[on - win]       # (n_app, n_units)
        diff = post - pre                     # (n_app, n_units)

        # one-sample t across appearances
        mean = diff.mean(axis=0)
        sd = diff.std(axis=0, ddof=1)
        sd[sd == 0] = np.inf
        t_obs = mean / (sd / np.sqrt(n_app))

        # sign-flip null
        signs = rng.choice([-1.0, 1.0], size=(args.n_perm, n_app))
        # t_perm[p,u]
        dperm_mean = (signs @ diff) / n_app                      # (n_perm, n_units)
        # std of sign-flipped diffs == std of diffs (sign flip preserves |diff|), but compute per perm
        # var of s*diff = mean((s*diff)^2) - mean(s*diff)^2
        sq = (diff ** 2)
        m2 = (signs ** 2 @ sq) / n_app                            # = mean(diff^2) (signs^2=1)
        var_perm = (m2 - dperm_mean ** 2) * n_app / (n_app - 1)
        sd_perm = np.sqrt(np.maximum(var_perm, 1e-12))
        t_perm = dperm_mean / (sd_perm / np.sqrt(n_app))

        # one-tailed (increase): p = fraction perms with t_perm >= t_obs
        p_one = (1.0 + (t_perm >= t_obs[None, :]).sum(axis=0)) / (args.n_perm + 1.0)

        # criterion (1): >=1 spike in >30% of post windows
        frac_active = (post >= 1).mean(axis=0)
        crit1 = frac_active > 0.30
        crit2 = (t_obs > 0) & (p_one < args.alpha)
        sig = crit1 & crit2
        sig_by_char[c] = sig

        order = np.argsort(p_one)
        detail[c] = {
            "n_appearances": n_app,
            "n_sig": int(sig.sum()),
            "top_units": [
                {"unit": units[u], "t": float(t_obs[u]), "p": float(p_one[u]),
                 "frac_post_active": float(frac_active[u]), "sig": bool(sig[u])}
                for u in order[:5]
            ],
        }
        print(f"{c:<12} appearances={n_app:4d}  sig(crit1&2)={int(sig.sum()):3d}/{n_units}")

    # responses per unit across characters
    sig_matrix = np.column_stack([sig_by_char[c] for c in named])  # (n_units, n_chars)
    n_resp_per_unit = sig_matrix.sum(axis=1)
    responsive = n_resp_per_unit >= 1
    selective = n_resp_per_unit == 1  # criterion (3): only one character

    n_selective = int(selective.sum())
    n_responsive = int(responsive.sum())
    print("\n=== DING REPLICATION (sub-572) ===")
    print(f"  SELECTIVE units (exactly 1 char, p<{args.alpha}): {n_selective}/{n_units} "
          f"({100*n_selective/n_units:.1f}%)   [Ding: ~6%]")
    print(f"  RESPONSIVE units (>=1 char):                      {n_responsive}/{n_units}")
    sel_units = [units[u] for u in np.where(selective)[0]]
    if sel_units:
        print("  selective units:", sel_units)

    out = {
        "sub": args.sub, "method": "Ding2025_selectivity_pre_post_onset_signflip",
        "win_frames": win, "n_perm": args.n_perm, "alpha": args.alpha, "n_units": n_units,
        "n_selective_single_char": n_selective,
        "pct_selective": 100.0 * n_selective / n_units,
        "n_responsive": n_responsive,
        "selective_units": sel_units,
        "per_char": detail,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
