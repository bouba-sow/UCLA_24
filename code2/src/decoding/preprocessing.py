"""Faithful clusterless voltage-spike extraction (Ding et al. 2025, Methods).

Reads the broadband micro iEEG (BrainVision, 32 kHz) from the BIDS dataset and
produces the model input tensor:

    X : float32  (n_windows, 2, n_channels, 50)
        dim 1 = spike polarity (0 = negative-going, 1 = positive-going)
        dim 3 = 50 time bins of 5 ms inside each 250 ms window

Pipeline (per the paper):
  1. notch filter (4th-order Butterworth, 300..3000 Hz step 60, bw 4) + 300 Hz
     high-pass (4th-order Butterworth), zero-phase.
  2. per-channel threshold at +/- 3 * SD (SD computed on the movie epoch).
  3. detect negative-going (minimum) and positive-going (maximum) excursions.
  4. remove coincident events (>= 6 of 8 channels in a bundle within 4 ms).
  5. discretize amplitudes (in SD units) into 0.5-wide bins from 3.5 to 30.5.
  6. sum discretized amplitudes into 5 ms bins -> (2, n_channels, n_total_bins).
  7. z-score within each bundle, separately per polarity.

The result is cached to an .npz so it only needs to be computed once.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

# --- Fixed constants from the paper ---------------------------------------
NOTCH_FREQS = np.arange(300, 3001, 60, dtype=float)  # 300..3000 step 60
NOTCH_BW_HZ = 4.0
HIGHPASS_HZ = 300.0
THRESHOLD_SD = 3.0
COINCIDENCE_CH = 6           # >= 6 of 8 channels
COINCIDENCE_MS = 4.0
AMP_BIN_LO = 3.5
AMP_BIN_HI = 30.5
AMP_BIN_STEP = 0.5
WINDOW_MS = 250.0
BIN_MS = 5.0
BINS_PER_WINDOW = int(WINDOW_MS / BIN_MS)  # 50

NEG, POS = 0, 1


def bundle_of(channel: str) -> str:
    """GA1-RA3 -> 'RA' (microwires on one shaft share a bundle)."""
    part = channel.split("-", 1)[1] if "-" in channel else channel
    return part.rstrip("0123456789")


def build_sos(fs: float) -> np.ndarray:
    """Combined zero-phase SOS: 60 Hz-harmonic notches + 300 Hz high-pass."""
    sections = []
    for f0 in NOTCH_FREQS:
        if f0 >= fs / 2:
            continue
        b, a = iirnotch(w0=f0 / (fs / 2), Q=f0 / NOTCH_BW_HZ)
        sections.append(tf2sos(b, a))
    hp = butter(4, HIGHPASS_HZ / (fs / 2), btype="highpass", output="sos")
    sections.append(hp)
    return np.vstack(sections)


def _detect_excursions(sig: np.ndarray, thr: float, polarity: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (sample_index, |amplitude|) of one extremum per threshold excursion."""
    if polarity == NEG:
        mask = sig < -thr
    else:
        mask = sig > thr
    if not mask.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    # contiguous runs of True
    edges = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]

    idx = np.empty(starts.size, dtype=np.int64)
    amp = np.empty(starts.size, dtype=np.float32)
    for k, (s, e) in enumerate(zip(starts, ends)):
        seg = sig[s:e]
        j = int(np.argmin(seg)) if polarity == NEG else int(np.argmax(seg))
        idx[k] = s + j
        amp[k] = abs(seg[j])
    return idx, amp


def _process_channel(
    vhdr: str,
    ch_index: int,
    start_sample: int,
    n_samples: int,
    sos: np.ndarray,
) -> dict:
    """Filter one channel over the movie epoch and detect spikes (in SD units)."""
    import mne

    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    sig = raw.get_data(picks=[ch_index], start=start_sample,
                       stop=start_sample + n_samples)[0].astype(np.float64)
    sig = sosfiltfilt(sos, sig)

    sd = float(np.std(sig))
    if sd <= 0 or not np.isfinite(sd):
        return {"neg": (np.empty(0, np.int64), np.empty(0, np.float32)),
                "pos": (np.empty(0, np.int64), np.empty(0, np.float32)), "sd": sd}

    thr = THRESHOLD_SD * sd
    neg_idx, neg_amp = _detect_excursions(sig, thr, NEG)
    pos_idx, pos_amp = _detect_excursions(sig, thr, POS)
    # amplitudes in SD units
    return {
        "neg": (neg_idx, (neg_amp / sd).astype(np.float32)),
        "pos": (pos_idx, (pos_amp / sd).astype(np.float32)),
        "sd": sd,
    }


def _discretize_sd(amp_sd: np.ndarray) -> np.ndarray:
    """Clip to [3.5, 30.5) then floor to the 0.5-wide bin's left edge."""
    a = np.clip(amp_sd, AMP_BIN_LO, AMP_BIN_HI - 1e-6)
    return (np.floor((a - AMP_BIN_LO) / AMP_BIN_STEP) * AMP_BIN_STEP + AMP_BIN_LO).astype(np.float32)


def extract_clusterless(
    bids_root: Path,
    subject: str,
    session: str,
    task: str,
    run: str,
    movie_start_rel: float,
    drift: float,
    movie_duration_video: float,
    n_jobs: int = 8,
    max_seconds: float | None = None,
) -> dict:
    import mne

    ieeg = (bids_root / f"sub-{subject}" / f"ses-{session}" / "ieeg")
    vhdr = str(ieeg / f"sub-{subject}_ses-{session}_task-{task}_acq-micro_run-{run}_ieeg.vhdr")

    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)
    n_channels = len(ch_names)

    dur = movie_duration_video if max_seconds is None else min(movie_duration_video, max_seconds)
    n_windows = int(np.floor(dur / (WINDOW_MS / 1000.0)))
    n_total_bins = n_windows * BINS_PER_WINDOW

    bin_width_samp = (BIN_MS / 1000.0) * drift * fs   # ~160 samples
    movie_start_samp = int(round(movie_start_rel * fs))
    span = int(np.ceil(n_total_bins * bin_width_samp)) + 1

    sos = build_sos(fs)

    print(f"fs={fs} n_ch={n_channels} n_windows={n_windows} "
          f"n_total_bins={n_total_bins} span_samples={span}")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_channel)(vhdr, c, movie_start_samp, span, sos)
        for c in range(n_channels)
    )

    bundles = [bundle_of(c) for c in ch_names]

    # --- coincidence removal (per bundle, 4 ms window) ---------------------
    coinc_w = COINCIDENCE_MS / 1000.0 * fs
    bad_coinc: dict[str, set[int]] = {}
    by_bundle: dict[str, list[int]] = {}
    for c, b in enumerate(bundles):
        by_bundle.setdefault(b, []).append(c)

    for b, chans in by_bundle.items():
        counts: dict[int, set[int]] = {}
        for c in chans:
            for pol in ("neg", "pos"):
                idx = results[c][pol][0]
                for s in (idx / coinc_w).astype(np.int64):
                    counts.setdefault(int(s), set()).add(c)
        bad_coinc[b] = {cb for cb, chs in counts.items() if len(chs) >= COINCIDENCE_CH}

    # --- bin discretized amplitudes into 5 ms bins ------------------------
    X = np.zeros((2, n_channels, n_total_bins), dtype=np.float32)
    for c, b in enumerate(bundles):
        bad = bad_coinc[b]
        for pol_name, pol in (("neg", NEG), ("pos", POS)):
            idx, amp_sd = results[c][pol_name]
            if idx.size == 0:
                continue
            bin_idx = (idx / bin_width_samp).astype(np.int64)
            keep = bin_idx < n_total_bins
            if bad:
                coinc_id = (idx / coinc_w).astype(np.int64)
                keep &= ~np.isin(coinc_id, list(bad))
            bin_idx = bin_idx[keep]
            disc = _discretize_sd(amp_sd[keep])
            np.add.at(X[pol, c], bin_idx, disc)

    X = X.reshape(2, n_channels, n_windows, BINS_PER_WINDOW)
    X = np.transpose(X, (2, 0, 1, 3)).copy()   # (n_windows, 2, n_channels, 50)

    # --- z-score within each bundle, per polarity -------------------------
    for b, chans in by_bundle.items():
        for pol in (NEG, POS):
            block = X[:, pol, chans, :]
            mu = block.mean()
            sd = block.std() + 1e-8
            X[:, pol, chans, :] = (block - mu) / sd

    return {
        "X": X,
        "channels": np.array(ch_names),
        "bundles": np.array(bundles),
        "fs": np.float64(fs),
        "movie_start_rel": np.float64(movie_start_rel),
        "drift": np.float64(drift),
        "n_windows": np.int64(n_windows),
    }


def _read_alignment(npz_dir: Path) -> tuple[float, float, float]:
    import glob
    files = sorted(glob.glob(str(npz_dir / "sub-*/ses-*/ieeg/*_spikedata.npz")))
    if not files:
        raise FileNotFoundError(f"No spikedata.npz under {npz_dir}")
    d = np.load(files[0], allow_pickle=True)
    return (float(d["movie_start_rel"]), float(d["drift_correction_multiplier"]),
            float(d["movie_duration_sec"]))


def main() -> None:
    p = argparse.ArgumentParser(description="Clusterless voltage-spike extraction (faithful).")
    p.add_argument("--bids-root", type=Path, default=Path("data/bids"))
    p.add_argument("--deriv-dir", type=Path, default=Path("data/bids/derivatives/spike-sorted"),
                   help="Used only to read movie alignment scalars.")
    p.add_argument("--subject", default="572")
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="movie24presleep")
    p.add_argument("--run", default="01")
    p.add_argument("--out", type=Path, default=Path("data/clusterless/sub-572_ses-01_clusterless.npz"))
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=None,
                   help="Limit movie duration (smoke test).")
    args = p.parse_args()

    movie_start_rel, drift, movie_dur_video = _read_alignment(args.deriv_dir)
    print(f"movie_start_rel={movie_start_rel} drift={drift} movie_duration_video={movie_dur_video}")

    out = extract_clusterless(
        args.bids_root, args.subject, args.session, args.task, args.run,
        movie_start_rel, drift, movie_dur_video,
        n_jobs=args.n_jobs, max_seconds=args.max_seconds,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"Saved clusterless tensor {out['X'].shape} -> {args.out}")


if __name__ == "__main__":
    main()
