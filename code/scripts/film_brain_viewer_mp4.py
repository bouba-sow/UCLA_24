#!/usr/bin/env python3
"""
Brain-surface MP4 renderer for iEEG (macro) and single-unit (microwire) data.

Select a mode with --mode:

  macro       Macro iEEG activity (high-gamma envelope or RMS) projected onto the
              fsaverage cortical surface.  Reads a BrainVision .vhdr file produced
              by ucla2bids.py, computes a per-frame activity z-score for each macro
              electrode, and maps it to a diverging blue→neutral→yellow colormap.

  singleunit  Binary spike activity from sorted microwire units.  Reads
              *_spikedata.npz bundles produced by ucla2bids.py.  Each electrode is
              black (silent) or yellow (fired in the current ~33 ms bin).  Cluster 0
              (noise) is excluded.  No normalization — the signal is purely binary.

Both modes produce a composited MP4:
  - Top panel  : optional embedded movie clip (with audio, seeked to film_time_start)
  - Bottom panel: 3 (or 1) rotating brain views rendered off-screen via PyVista/VTK

The compositing is done with ffmpeg filter_complex (not PIL), which is much faster
than per-frame image stacking.

Speed tips
----------
  --n-workers N      Render N chunks in parallel (N PyVista subprocesses).
                     Use 1 on GPU nodes (hardware OpenGL is already fast),
                     4–8 on CPU nodes.
  --render-scale S   Render at S× resolution, ffmpeg upscales in composite step
                     (0.5 → 4× fewer pixels → ~4× faster rendering).
  --out-fps 15       Produce fewer output frames.

Usage examples
--------------
  # Macro iEEG high-gamma, 140 s starting at t=70 s
  python film_brain_viewer_mp4.py --mode macro \\
      --film-time-start 70 --duration-sec 140 \\
      --out-mp4 outputs/brain_macro.mp4

  # Single-unit spikes, GPU node
  python film_brain_viewer_mp4.py --mode singleunit \\
      --n-workers 1 --render-scale 1.0 \\
      --out-mp4 outputs/brain_singleunit.mp4

  # Dry-run: preview layout without full render
  python film_brain_viewer_mp4.py --mode macro --duration-sec 2 --out-fps 2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from nilearn import datasets, surface


# ─────────────────────────────────────────────────────────────────────────────
# Project-level defaults (sub-572, Experiment-9)
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path("/store/scratch/bsow/Documents/UCLA_24")
_BIDS = _ROOT / "data/bids"
_IEEG_DIR = _BIDS / "sub-572/ses-01/ieeg"
_NPZ_DIR = _BIDS / "derivatives/spike-sorted/sub-572/ses-01/ieeg"

_DEFAULTS: dict[str, dict[str, Path | str | float | None]] = {
    "macro": {
        "vhdr_path": _IEEG_DIR / "sub-572_ses-01_task-movie24presleep_acq-macro_run-01_ieeg.vhdr",
        "electrodes_tsv": _IEEG_DIR / "sub-572_ses-01_electrodes.tsv",
        "align_json": (
            _ROOT
            / "data/ucla_data/572/Experiment-9/Audio"
            / "572_exp_09_preSleep_movie_24_audio_movie_start_time.json"
        ),
        "out_mp4": _ROOT / "outputs/brain_macro.mp4",
    },
    "singleunit": {
        "npz_dir": _NPZ_DIR,
        "electrodes_tsv": _IEEG_DIR / "sub-572_ses-01_electrodes.tsv",
        "out_mp4": _ROOT / "outputs/brain_singleunit.mp4",
    },
}
_EMBED_DEFAULT = _ROOT / "data/40m_act_24_S06E01_30fps.m4v"


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decimate_faces(
    coords: np.ndarray, faces: np.ndarray, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin the mesh by keeping every `step`-th face.  step=1 returns unchanged."""
    step = max(1, int(step))
    if step == 1:
        return coords, faces, np.arange(coords.shape[0], dtype=np.int64)
    faces_d = faces[::step]
    if faces_d.size == 0:
        return coords, faces, np.arange(coords.shape[0], dtype=np.int64)
    used = np.unique(faces_d.ravel())
    idx_map = np.full(coords.shape[0], -1, dtype=np.int64)
    idx_map[used] = np.arange(used.size, dtype=np.int64)
    return coords[used], idx_map[faces_d].astype(np.int32), used


def _to_pv_faces(faces: np.ndarray) -> np.ndarray:
    """Prepend face count (3) to each triangle row for PyVista."""
    return np.c_[np.full(faces.shape[0], 3, dtype=np.int32), faces].ravel()


def _normalize_sulc(vals: np.ndarray, contrast: float) -> np.ndarray:
    """Map sulcal depth values to [0, 1] with optional gamma contrast."""
    lo, hi = np.percentile(vals, [5, 95])
    x = np.clip((vals - lo) / max(1e-9, hi - lo), 0.0, 1.0)
    return np.clip(x ** max(0.2, float(contrast)), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Camera helpers
# ─────────────────────────────────────────────────────────────────────────────

def _camera_position(
    center: np.ndarray,
    radius: float,
    azim_deg: float,
    elev_deg: float,
    distance_factor: float,
) -> tuple[tuple, tuple, tuple]:
    """Return (position, focal_point, up_vector) for a spherical camera."""
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return tuple(center + d * distance_factor * radius), tuple(center), (0.0, 0.0, 1.0)


def _parse_offsets(spec: str | None, n: int) -> list[float]:
    """
    Parse comma-separated per-view angle offsets.

    A single value is broadcast to all views.
    Examples: "0,270,270" → [0.0, 270.0, 270.0] | "10" → [10.0, 10.0, 10.0]
    """
    if not spec or spec.strip() == "":
        return [0.0] * n
    vals = [float(p) for p in spec.split(",") if p.strip()]
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"Expected 1 or {n} offset values, got {len(vals)} from '{spec}'.")
    return vals


def _parse_spin_mask(spec: str | None, n: int) -> list[bool]:
    """
    Parse per-view spin enable flags from comma-separated 0/1 values.

    Default for 3 views: [True, False, True] (middle panel fixed).
    Example: "1,0,1" → [True, False, True]
    """
    if not spec or spec.strip() == "":
        return ([True, False, True] if n == 3 else [True] * n)
    vals = [p in ("1", "true", "True", "yes") for p in spec.split(",") if p.strip()]
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"Expected 1 or {n} spin-mask values, got {len(vals)} from '{spec}'.")
    return vals


# ─────────────────────────────────────────────────────────────────────────────
# Colormap builders
# ─────────────────────────────────────────────────────────────────────────────

def _diverging_cmap(neutral_color: str) -> LinearSegmentedColormap:
    """
    Diverging colormap for macro iEEG z-scores: deep blue → neutral → bright yellow.

    The neutral color sits at the center (z ≈ 0) and is visible on a black
    background without appearing white or washed out.
    """
    from matplotlib.colors import to_rgb
    stops = [
        (0.00, "#1f4e79"),    # strong negative: deep blue
        (0.35, "#2c7fb8"),    # moderate negative
        (0.50, neutral_color),# near-zero / below-threshold
        (0.65, "#ffd54a"),    # moderate positive
        (1.00, "#fff200"),    # strong positive: bright yellow
    ]
    positions = [s[0] for s in stops]
    rgb_vals = [to_rgb(s[1]) for s in stops]
    cmap_data = {
        ch: [(pos, col[i], col[i]) for pos, col in zip(positions, rgb_vals)]
        for i, ch in enumerate(("red", "green", "blue"))
    }
    return LinearSegmentedColormap("macro_diverging", cmap_data, N=256)


def _binary_cmap() -> LinearSegmentedColormap:
    """
    Binary colormap for single-unit spikes: black (silent) → yellow (fired).

    The transition is near the midpoint so any value > 0.5 snaps to yellow.
    This ensures every active electrode shows the same yellow regardless of
    how many spikes happened to fall in the bin.
    """
    from matplotlib.colors import to_rgb
    stops = [
        (0.00, "#000000"),   # 0 = silent: black
        (0.49, "#000000"),   # just below 0.5: still black
        (0.51, "#ffee00"),   # just above 0.5: yellow
        (1.00, "#ffee00"),   # 1 = active: yellow
    ]
    positions = [s[0] for s in stops]
    rgb_vals = [to_rgb(s[1]) for s in stops]
    cmap_data = {
        ch: [(pos, col[i], col[i]) for pos, col in zip(positions, rgb_vals)]
        for i, ch in enumerate(("red", "green", "blue"))
    }
    return LinearSegmentedColormap("singleunit_binary", cmap_data, N=256)


def _build_cmap_from_spec(spec: dict) -> LinearSegmentedColormap:
    """Reconstruct a colormap inside a subprocess from a serializable spec dict."""
    if spec["kind"] == "diverging":
        return _diverging_cmap(spec["neutral_color"])
    if spec["kind"] == "binary":
        return _binary_cmap()
    raise ValueError(f"Unknown colormap kind: {spec['kind']}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders — macro iEEG
# ─────────────────────────────────────────────────────────────────────────────

def _read_align_json(path: Path | None) -> tuple[float, float]:
    """
    Read audio-alignment JSON produced by the Fried Lab pipeline.

    Returns (movie_start_rel_rec, drift_correction_multiplier).
    Falls back to (0.0, 1.0) if the file is missing or keys are absent.
    """
    if path is None or not path.is_file():
        return 0.0, 1.0
    data = json.loads(path.read_text())
    t0 = float(data.get("start_rel_rec", [0.0])[0])
    drift = float(data.get("drift_correction_multiplier", [1.0])[0])
    return t0, drift


def _load_macro_electrodes(electrodes_tsv: Path, channel_names: list[str]) -> pd.DataFrame:
    """Return macro electrodes with valid MNI coordinates that appear in channel_names."""
    el = pd.read_csv(electrodes_tsv, sep="\t")
    macro = el[el["group"].astype(str).str.lower() == "macro"].copy()
    macro = macro[macro["name"].isin(channel_names)].copy()
    for c in ("x", "y", "z"):
        macro[c] = pd.to_numeric(macro[c], errors="coerce")
    macro = macro.dropna(subset=["x", "y", "z"])
    if macro.empty:
        raise ValueError("No macro channels with valid MNI coordinates found in electrodes TSV.")
    return macro


def _safe_n_frames(
    n_samp: int,
    sfreq: float,
    crop_start_rec: float,
    movie_t0_rec: float,
    drift: float,
    film_time_start: float,
    out_fps: float,
    n_want: int,
) -> int:
    """Reduce n_frames until the last frame's iEEG window fits inside the cropped data."""
    win = max(1, int(round(sfreq / out_fps)))
    for n in range(n_want, 0, -1):
        t_mov = film_time_start + (n - 0.5) / out_fps
        t_in = (movie_t0_rec + t_mov * drift) - crop_start_rec
        c0 = int(np.clip(round(t_in * sfreq - win / 2), 0, n_samp - win))
        if c0 + win <= n_samp:
            return n
    return 1


def _compute_macro_activity(
    data: np.ndarray,
    sfreq: float,
    crop_start_rec: float,
    movie_t0_rec: float,
    drift: float,
    film_time_start: float,
    out_fps: float,
    n_frames: int,
    metric: str,
    hg_low_hz: float,
    hg_high_hz: float,
    hg_use_hilbert: bool,
) -> np.ndarray:
    """
    Build (n_frames, n_channels) z-score matrix for macro iEEG activity.

    Normalisation: per-channel median subtraction + MAD scaling → clipped to ±4.
    This is robust to outlier frames and preserves between-channel amplitude
    differences via a shared (channel-wise) baseline.

    Parameters
    ----------
    metric : {"rms", "mean_abs", "high_gamma"}
        Activity metric per frame window.  high_gamma bandpasses to
        [hg_low_hz, hg_high_hz] Hz and computes the Hilbert envelope
        (or squared signal if hg_use_hilbert=False).
    """
    import mne
    from scipy.signal import hilbert as _hilbert

    n_ch, n_samp = data.shape
    win = max(1, int(round(sfreq / out_fps)))
    metric_key = metric.strip().lower()

    if metric_key == "high_gamma":
        data_m = mne.filter.filter_data(
            data.copy(), sfreq=sfreq,
            l_freq=float(hg_low_hz), h_freq=float(hg_high_hz),
            method="fir", verbose="ERROR",
        )
        data_m = np.abs(_hilbert(data_m, axis=1)) if hg_use_hilbert else data_m ** 2
    else:
        data_m = data

    values = np.empty((n_frames, n_ch), dtype=np.float64)
    for k in range(n_frames):
        t_mov = film_time_start + (k + 0.5) / out_fps
        t_in = (movie_t0_rec + t_mov * drift) - crop_start_rec
        c0 = int(np.clip(round(t_in * sfreq - win / 2), 0, n_samp - win))
        sl = data_m[:, c0 : c0 + win]
        if metric_key == "rms":
            values[k] = np.sqrt(np.mean(sl ** 2, axis=1))
        elif metric_key == "mean_abs":
            values[k] = np.mean(np.abs(sl), axis=1)
        else:
            values[k] = np.mean(sl, axis=1)

    med = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - med), axis=0, keepdims=True)
    z = (values - med) / np.maximum(1e-9, 1.4826 * mad)
    return np.clip(z, -4.0, 4.0)


def load_macro_data(
    vhdr_path: Path,
    electrodes_tsv: Path,
    align_json: Path | None,
    film_time_start: float,
    duration_sec: float,
    out_fps: float,
    movie_t0_override: float | None,
    drift_override: float | None,
    crop_start: float | None,
    notch_hz: float | None,
    highpass_hz: float | None,
    metric: str,
    hg_low_hz: float,
    hg_high_hz: float,
    hg_use_hilbert: bool,
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """
    Load macro iEEG and compute the per-frame activity matrix.

    Returns
    -------
    pts : (n_ch, 3) MNI coordinates.
    color_matrix : (n_frames, n_ch) z-score values clipped to ±4.
    n_frames : actual number of frames after bounds check.
    meta : dict with cmax, clim, scalar_bar_title.
    """
    import mne

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
    tmax_rec = float(raw.times[-1])

    t_align, drift = _read_align_json(align_json)
    if movie_t0_override is not None:
        t_align = float(movie_t0_override)
    if drift_override is not None:
        drift = float(drift_override)

    t_rec_start = t_align + film_time_start * drift
    t_rec_end   = t_align + (film_time_start + duration_sec) * drift
    rec_lo = float(max(0.0, crop_start if crop_start is not None else t_rec_start - 0.5))
    rec_hi = float(min(tmax_rec, t_rec_end + 0.5))
    if rec_lo >= rec_hi:
        raise ValueError(f"Crop window is empty: rec_lo={rec_lo:.2f} rec_hi={rec_hi:.2f}.")

    raw.crop(tmin=rec_lo, tmax=rec_hi)
    raw.load_data()
    if notch_hz is not None:
        raw.notch_filter(freqs=[notch_hz], verbose="ERROR")
    if highpass_hz is not None:
        raw.filter(l_freq=highpass_hz, h_freq=None, verbose="ERROR")

    coords = _load_macro_electrodes(electrodes_tsv, list(raw.ch_names))
    raw.pick(list(coords["name"]))
    data = raw.get_data() * 1e6  # V → µV
    sfreq = float(raw.info["sfreq"])

    n_want = max(1, int(np.floor(duration_sec * out_fps)))
    n_frames = _safe_n_frames(
        n_samp=data.shape[1], sfreq=sfreq, crop_start_rec=rec_lo,
        movie_t0_rec=t_align, drift=drift,
        film_time_start=film_time_start, out_fps=out_fps, n_want=n_want,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = _compute_macro_activity(
            data=data, sfreq=sfreq, crop_start_rec=rec_lo,
            movie_t0_rec=t_align, drift=drift,
            film_time_start=film_time_start, out_fps=out_fps, n_frames=n_frames,
            metric=metric, hg_low_hz=hg_low_hz,
            hg_high_hz=hg_high_hz, hg_use_hilbert=hg_use_hilbert,
        )

    pts = np.c_[coords["x"].to_numpy(), coords["y"].to_numpy(), coords["z"].to_numpy()]
    cmax = max(1.0, float(np.nanpercentile(np.abs(z), 98)))
    return pts, z, n_frames, {
        "cmax": cmax,
        "clim": (-cmax, cmax),
        "scalar_bar_title": f"z({metric})",
        "show_scalar_bar": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders — single-unit microwire
# ─────────────────────────────────────────────────────────────────────────────

def _load_singleunit_electrodes(
    npz_dir: Path, electrodes_tsv: Path
) -> tuple[np.ndarray, list[str], list[Path]]:
    """
    Match *_spikedata.npz files to micro electrode coordinates from the TSV.

    Returns (pts, channel_names, npz_paths) for all channels that have both
    a valid MNI coordinate and a NPZ file.
    """
    el = pd.read_csv(electrodes_tsv, sep="\t")
    micro = el[el["group"].astype(str).str.lower() == "micro"].copy()
    for c in ("x", "y", "z"):
        micro[c] = pd.to_numeric(micro[c], errors="coerce")
    micro = micro.dropna(subset=["x", "y", "z"]).reset_index(drop=True)
    if micro.empty:
        raise ValueError("No micro channels with valid MNI coordinates in electrodes TSV.")

    def _norm(s: str) -> str:
        return s.replace("-", "").replace("_", "").upper()

    npz_map: dict[str, Path] = {}
    for p in npz_dir.glob("*spikedata.npz"):
        for part in p.stem.split("_"):
            if part.startswith("desc-"):
                npz_map[_norm(part[5:])] = p
                break

    pts_list, names_list, paths_list = [], [], []
    for _, row in micro.iterrows():
        key = _norm(str(row["name"]))
        if key in npz_map:
            pts_list.append([row["x"], row["y"], row["z"]])
            names_list.append(str(row["name"]))
            paths_list.append(npz_map[key])

    if not names_list:
        raise ValueError(
            f"No micro channels matched between electrodes TSV and NPZ files in {npz_dir}."
        )
    return np.array(pts_list, dtype=np.float64), names_list, paths_list


def _build_binary_activity(
    npz_paths: list[Path],
    film_time_start: float,
    duration_sec: float,
    out_fps: float,
) -> np.ndarray:
    """
    Build a (n_frames, n_channels) binary matrix in {0, 1}.

    Value is 1.0 on every output frame whose ~33 ms bin contains at least one
    spike (cluster_id >= 1; cluster 0 / noise is excluded).  Value is 0.0
    otherwise.  No normalization or smoothing — purely binary.
    """
    d0 = np.load(npz_paths[0], allow_pickle=True)
    bin_hz        = float(d0["firing_rate_hz"])
    bin_edges     = d0["firing_rate_bin_edges"]
    n_bins        = len(bin_edges) - 1
    movie_duration = float(d0["movie_duration_sec"])
    n_frames       = max(1, int(np.floor(duration_sec * out_fps)))

    binary = np.zeros((n_bins, len(npz_paths)), dtype=np.float64)
    for j, p in enumerate(npz_paths):
        d = np.load(p, allow_pickle=True)
        times = d["spike_times_movie"].astype(np.float64)
        valid = times[d["cluster_id"] > 0]
        counts, _ = np.histogram(valid, bins=n_bins, range=(0.0, movie_duration))
        binary[:, j] = (counts > 0).astype(np.float64)

    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    t_end = min(film_time_start + duration_sec, movie_duration)
    frame_times = np.clip(
        film_time_start + (np.arange(n_frames) + 0.5) / out_fps, 0.0, t_end
    )
    bin_idx = np.clip(
        np.searchsorted(bin_centres, frame_times, side="right") - 1, 0, n_bins - 1
    )
    return binary[bin_idx, :]   # (n_frames, n_ch) ∈ {0, 1}


def load_singleunit_data(
    npz_dir: Path,
    electrodes_tsv: Path,
    film_time_start: float,
    duration_sec: float,
    out_fps: float,
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """
    Load single-unit NPZ bundles and build the binary activity matrix.

    Returns
    -------
    pts : (n_ch, 3) MNI coordinates.
    color_matrix : (n_frames, n_ch) binary values in {0, 1}.
    n_frames : number of output frames.
    meta : dict with clim, show_scalar_bar.
    """
    pts, _, npz_paths = _load_singleunit_electrodes(npz_dir, electrodes_tsv)
    print(f"  {len(npz_paths)} micro channels matched to NPZ files.")
    color_matrix = _build_binary_activity(npz_paths, film_time_start, duration_sec, out_fps)
    n_frames = color_matrix.shape[0]
    return pts, color_matrix, n_frames, {"clim": (0.0, 1.0), "show_scalar_bar": False}


# ─────────────────────────────────────────────────────────────────────────────
# Threshold helpers (macro only)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_threshold(
    z: np.ndarray,
    mode: str,
    fixed_z: float,
    quantile: float,
) -> float:
    """Return the scalar threshold value for the given mode."""
    mode = mode.strip().lower()
    if mode == "fixed":
        return float(fixed_z)
    if mode == "quantile":
        return float(np.quantile(np.abs(z), float(quantile)))
    return 0.0  # "none"


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk render worker (subprocess-safe — called via multiprocessing.spawn)
# ─────────────────────────────────────────────────────────────────────────────

def _render_chunk_worker(kwargs: dict) -> str:
    """
    Render a contiguous slice of frames to a temp MP4 file.

    All PyVista state is created fresh inside this subprocess so it is safe to
    run under multiprocessing.spawn without shared memory issues.

    Parameters (passed via `kwargs` dict for pickling compatibility)
    ------------------------------------------------------------------
    color_chunk     : (n_frames_chunk, n_ch) activity values.
    frame_offset    : absolute frame index of chunk[0] (for spin calculation).
    pts             : (n_ch, 3) MNI electrode coordinates.
    cmap_spec       : {"kind": "diverging", "neutral_color": ...} or {"kind": "binary"}.
    clim            : (vmin, vmax) colormap limits.
    threshold_mode  : "none" | "fixed" | "quantile" (macro only).
    thr_val         : scalar threshold value (used when threshold_mode != "none").
    show_gray_bg    : show a neutral-color electrode layer under sub-threshold dots.
    show_scalar_bar : display a colormap legend (True for macro, False for singleunit).
    scalar_bar_title: label for the scalar bar.
    neutral_color   : hex color for the background electrode layer.
    temp_path       : output MP4 path for this chunk.
    out_fps, render_w, render_h: video and resolution settings.
    brain_views     : 1 or 3.
    view_angles     : list of (azimuth_deg, elevation_deg) per view.
    spin_mask       : list of bool, per-view spin enable.
    spin_deg_per_sec: camera rotation speed.
    camera_dist     : distance_factor for _camera_position.
    surface_decim   : mesh decimation step (1 = full detail).
    sulc_contrast   : gamma contrast for sulcal depth shading.
    brain_alpha     : surface opacity.
    brain_cmap      : matplotlib colormap name for sulcal shading.
    brain_clim_low/high: colormap limits for sulcal shading.
    brain_solid_color: if set, use uniform cortex color (disables sulcal map).
    electrode_size  : point size in pixels.
    parallel_proj   : use parallel (orthographic) projection.
    parallel_scale  : scale for parallel projection (controls zoom).
    show_text       : display title text in upper-left corner.
    mode            : "macro" | "singleunit" (for title text).
    """
    # ── environment setup ────────────────────────────────────────────────────
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import warnings
    import numpy as np
    import pyvista as pv
    import imageio.v2 as iio
    from pathlib import Path
    from nilearn import datasets, surface

    pv.OFF_SCREEN = True
    try:
        pv.start_xvfb()
    except Exception:
        pass

    # ── unpack kwargs ────────────────────────────────────────────────────────
    color_chunk      = kwargs["color_chunk"]
    frame_offset     = int(kwargs["frame_offset"])
    pts              = kwargs["pts"]
    cmap_spec        = kwargs["cmap_spec"]
    clim             = kwargs["clim"]
    threshold_mode   = kwargs["threshold_mode"]
    thr_val          = float(kwargs["thr_val"])
    show_gray_bg     = bool(kwargs["show_gray_bg"])
    show_scalar_bar  = bool(kwargs["show_scalar_bar"])
    scalar_bar_title = str(kwargs["scalar_bar_title"])
    neutral_color    = str(kwargs["neutral_color"])
    temp_path        = Path(kwargs["temp_path"])
    out_fps          = float(kwargs["out_fps"])
    render_w         = int(kwargs["render_w"])
    render_h         = int(kwargs["render_h"])
    brain_views      = int(kwargs["brain_views"])
    view_angles      = kwargs["view_angles"]
    spin_mask        = kwargs["spin_mask"]
    spin_deg_per_sec = float(kwargs["spin_deg_per_sec"])
    camera_dist      = float(kwargs["camera_dist"])
    surface_decim    = int(kwargs["surface_decim"])
    sulc_contrast    = float(kwargs["sulc_contrast"])
    brain_alpha      = float(kwargs["brain_alpha"])
    brain_cmap       = str(kwargs["brain_cmap"])
    brain_clim_low   = float(kwargs["brain_clim_low"])
    brain_clim_high  = float(kwargs["brain_clim_high"])
    brain_solid_color = kwargs["brain_solid_color"]
    electrode_size   = float(kwargs["electrode_size"])
    parallel_proj    = bool(kwargs["parallel_proj"])
    parallel_scale   = float(kwargs["parallel_scale"])
    show_text        = bool(kwargs["show_text"])
    mode             = str(kwargs["mode"])

    # ── build fsaverage brain surface ────────────────────────────────────────
    fsaverage = datasets.fetch_surf_fsaverage()
    lh_c, lh_f = surface.load_surf_data(fsaverage.pial_left)
    rh_c, rh_f = surface.load_surf_data(fsaverage.pial_right)
    lh_sulc = surface.load_surf_data(fsaverage.sulc_left).astype(np.float64)
    rh_sulc = surface.load_surf_data(fsaverage.sulc_right).astype(np.float64)

    def _decim(coords, faces, step):
        step = max(1, int(step))
        if step == 1:
            return coords, faces, np.arange(coords.shape[0], dtype=np.int64)
        fd = faces[::step]
        used = np.unique(fd.ravel())
        im = np.full(coords.shape[0], -1, dtype=np.int64)
        im[used] = np.arange(used.size)
        return coords[used], im[fd].astype(np.int32), used

    def _norm_sulc(v, c):
        lo, hi = np.percentile(v, [5, 95])
        x = np.clip((v - lo) / max(1e-9, hi - lo), 0.0, 1.0)
        return np.clip(x ** max(0.2, c), 0.0, 1.0)

    lh_c, lh_f, lh_u = _decim(lh_c.astype(np.float64), lh_f.astype(np.int32), surface_decim)
    rh_c, rh_f, rh_u = _decim(rh_c.astype(np.float64), rh_f.astype(np.int32), surface_decim)
    lh_poly = pv.PolyData(lh_c, np.c_[np.full(lh_f.shape[0], 3, np.int32), lh_f].ravel())
    rh_poly = pv.PolyData(rh_c, np.c_[np.full(rh_f.shape[0], 3, np.int32), rh_f].ravel())
    lh_poly.point_data["sulc"] = _norm_sulc(lh_sulc[lh_u], sulc_contrast)
    rh_poly.point_data["sulc"] = _norm_sulc(rh_sulc[rh_u], sulc_contrast)
    brain_poly = lh_poly.merge(rh_poly)

    b = brain_poly.bounds
    center   = np.array([(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2])
    span_max = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
    radius   = span_max * 0.55

    # ── build colormap ───────────────────────────────────────────────────────
    elec_cmap = _build_cmap_from_spec(cmap_spec)

    # ── create plotter ───────────────────────────────────────────────────────
    shape = (1, 1) if brain_views == 1 else (1, 3)
    plotter = pv.Plotter(off_screen=True, shape=shape, window_size=(render_w, render_h))
    elec_meshes: list = []

    def _cam(az, el):
        az_r, el_r = np.deg2rad(az), np.deg2rad(el)
        d = np.array([np.cos(el_r)*np.cos(az_r), np.cos(el_r)*np.sin(az_r), np.sin(el_r)])
        return tuple(center + d * camera_dist * radius), tuple(center), (0., 0., 1.)

    for i, (az, el) in enumerate(view_angles):
        plotter.subplot(0, i)
        plotter.set_background("black")
        plotter.add_mesh(
            brain_poly,
            scalars=None if brain_solid_color else "sulc",
            color=brain_solid_color or None,
            cmap=brain_cmap if not brain_solid_color else None,
            clim=[brain_clim_low, brain_clim_high] if not brain_solid_color else None,
            opacity=brain_alpha, smooth_shading=True,
            specular=0.05, specular_power=12.0, ambient=0.20, diffuse=0.45,
            show_scalar_bar=False,
        )
        # Neutral-color background layer (visible when threshold_style="dim")
        bg = pv.PolyData(pts.copy())
        bg.point_data["c"] = np.ones(pts.shape[0])
        bg_actor = plotter.add_mesh(
            bg, color=neutral_color, render_points_as_spheres=True,
            point_size=electrode_size, ambient=0.35, diffuse=0.90,
            specular=0.10, show_scalar_bar=False,
        )
        bg_actor.SetVisibility(show_gray_bg)

        em = pv.PolyData(pts.copy())
        em.point_data["activity"] = color_chunk[0].astype(np.float64)
        plotter.add_mesh(
            em, scalars="activity", cmap=elec_cmap, clim=clim,
            render_points_as_spheres=True, point_size=electrode_size,
            ambient=0.35, diffuse=0.90, specular=0.15,
            nan_color="#000000", show_scalar_bar=False,
        )
        plotter.camera_position = _cam(az, el)
        plotter.camera.parallel_projection = parallel_proj
        if parallel_proj:
            plotter.camera.parallel_scale = parallel_scale
        plotter.enable_anti_aliasing("msaa")
        plotter.add_light(
            pv.Light(position=(300, 200, 300), focal_point=tuple(center), intensity=0.9)
        )
        if show_text and i == 0:
            label = (
                f"Macro iEEG — {scalar_bar_title}"
                if mode == "macro"
                else "Single-unit spikes"
            )
            plotter.add_text(label, position="upper_left", color="white", font_size=11)
        elec_meshes.append(em)

    if show_scalar_bar:
        plotter.subplot(0, len(view_angles) - 1)
        plotter.add_scalar_bar(
            title=scalar_bar_title, n_labels=5, color="white",
            title_font_size=14, label_font_size=11, fmt="%.1f",
        )

    # ── render frames ────────────────────────────────────────────────────────
    writer = iio.get_writer(
        str(temp_path), fps=out_fps, codec="libx264",
        pixelformat="yuv420p", ffmpeg_log_level="error", macro_block_size=1,
    )
    for k, frame_vals in enumerate(color_chunk):
        t_sec = (frame_offset + k) / out_fps
        for i, (az0, el0) in enumerate(view_angles):
            plotter.subplot(0, i)
            fv = frame_vals.copy().astype(np.float64)
            if threshold_mode != "none":
                fv[np.abs(fv) < thr_val] = np.nan
            elec_meshes[i]["activity"] = fv
            elec_meshes[i].set_active_scalars("activity")
            elec_meshes[i].Modified()
            if spin_deg_per_sec != 0.0 and spin_mask[i]:
                plotter.camera_position = _cam(az0 + spin_deg_per_sec * t_sec, el0)
        plotter.render()
        writer.append_data(plotter.screenshot(return_img=True))

    writer.close()
    plotter.close()
    return str(temp_path)


# ─────────────────────────────────────────────────────────────────────────────
# ffmpeg compositing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ffmpeg_composite(
    brain_video: Path,
    embed_video_path: Path | None,
    film_time_start: float,
    duration_sec: float,
    out_mp4: Path,
    width: int,
    height: int,
    top_h: int,
    gap_h: int,
    brain_h: int,
    embed_width_frac: float,
    embed_x_align: str,
    brain_panel_y_shift_px: int,
    ffmpeg_bin: str,
) -> None:
    """
    Stack [movie panel (top)] + [brain render (bottom)] via ffmpeg filter_complex
    and mux audio from the embedded video, trimmed to [film_time_start, +duration_sec].

    If no embed video is provided, the brain video is simply copied to out_mp4.
    """
    if embed_video_path is None or not embed_video_path.is_file():
        shutil.copy2(brain_video, out_mp4)
        return

    panel_w = max(1, int(round(width * float(np.clip(embed_width_frac, 0.1, 1.0)))))
    x_off = {"left": 0, "right": width - panel_w}.get(
        embed_x_align.strip().lower(), (width - panel_w) // 2
    )
    shift = max(0, int(brain_panel_y_shift_px))

    top_filter = (
        f"[0:v]trim=start={film_time_start}:duration={duration_sec},"
        f"setpts=PTS-STARTPTS,"
        f"scale={panel_w}:{top_h}:force_original_aspect_ratio=decrease,"
        f"pad={panel_w}:{top_h}:(ow-iw)/2:(oh-ih)/2,"
        f"pad={width}:{top_h}:{x_off}:0:black,setsar=1[top]"
    )
    bot_filter = (
        f"[1:v]scale={width}:{brain_h}:flags=lanczos,"
        f"pad={width}:{brain_h + shift}:0:{shift}:black,setsar=1[bot]"
    )
    audio_filter = (
        f"[0:a]atrim=start={film_time_start}:duration={duration_sec},"
        f"asetpts=PTS-STARTPTS[aud]"
    )
    stack_filter = (
        (f"color=black:{width}x{gap_h}:r=25[gap];[top][gap][bot]vstack=inputs=3,setsar=1[out]")
        if gap_h > 0
        else f"[top][bot]vstack=inputs=2,setsar=1[out]"
    )

    filter_complex = ";".join([top_filter, bot_filter, audio_filter, stack_filter])
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(embed_video_path),
        "-i", str(brain_video),
        "-filter_complex", filter_complex,
        "-map", "[out]", "-map", "[aud]",
        "-t", str(duration_sec),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-shortest",
        str(out_mp4),
    ]
    print("Running ffmpeg composite …")
    subprocess.run(cmd, check=True)


def _ffmpeg_concat(temp_paths: list[Path], out_path: Path, ffmpeg_bin: str) -> None:
    """Concatenate multiple MP4 chunks into one file using ffmpeg concat demuxer."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_file = f.name
        for p in temp_paths:
            f.write(f"file '{p.resolve()}'\n")
    try:
        subprocess.run(
            [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", str(out_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    finally:
        os.unlink(list_file)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run(
    # ── Mode ──────────────────────────────────────────────────────────────────
    mode: str,

    # ── Output ────────────────────────────────────────────────────────────────
    out_mp4: Path,
    film_time_start: float,
    duration_sec: float,
    out_fps: float,

    # ── Macro-specific inputs ─────────────────────────────────────────────────
    vhdr_path: Path | None,
    electrodes_tsv: Path,
    align_json: Path | None,
    movie_t0_override: float | None,
    drift_override: float | None,
    crop_start: float | None,
    notch_hz: float | None,
    highpass_hz: float | None,
    metric: str,
    hg_low_hz: float,
    hg_high_hz: float,
    hg_use_hilbert: bool,
    threshold_mode: str,
    threshold_z: float,
    threshold_quantile: float,
    threshold_style: str,

    # ── Single-unit-specific inputs ───────────────────────────────────────────
    npz_dir: Path | None,

    # ── Layout ────────────────────────────────────────────────────────────────
    width: int,
    height: int,
    brain_views: int,
    embed_video_path: Path | None,
    embed_top_frac: float,
    embed_gap_px: int,
    embed_width_frac: float,
    embed_x_align: str,
    brain_panel_y_shift_px: int,

    # ── Rendering ────────────────────────────────────────────────────────────
    n_workers: int,
    render_scale: float,

    # ── Brain surface ─────────────────────────────────────────────────────────
    surface_decim: int,
    sulc_contrast: float,
    brain_alpha: float,
    brain_cmap: str,
    brain_clim_low: float,
    brain_clim_high: float,
    brain_solid_color: str | None,

    # ── Electrodes ────────────────────────────────────────────────────────────
    electrode_size: float,
    neutral_color: str,

    # ── Camera ────────────────────────────────────────────────────────────────
    camera_distance_factor: float,
    parallel_projection: bool,
    parallel_scale_factor: float,
    spin_deg_per_sec: float,
    spin_view_mask: str | None,
    azim_ccw_deg: float,
    view_azim_offsets: str | None,
    view_elev_offsets: str | None,

    # ── Misc ──────────────────────────────────────────────────────────────────
    show_text: bool,
    ffmpeg_exe: str | None,
) -> None:

    mode = mode.strip().lower()
    if mode not in ("macro", "singleunit"):
        raise ValueError("--mode must be 'macro' or 'singleunit'.")

    # ── 1. Load data and build activity matrix ─────────────────────────────
    print(f"[{mode}] Loading data …")
    if mode == "macro":
        pts, color_matrix, n_frames, meta = load_macro_data(
            vhdr_path=vhdr_path,
            electrodes_tsv=electrodes_tsv,
            align_json=align_json,
            film_time_start=film_time_start,
            duration_sec=duration_sec,
            out_fps=out_fps,
            movie_t0_override=movie_t0_override,
            drift_override=drift_override,
            crop_start=crop_start,
            notch_hz=notch_hz,
            highpass_hz=highpass_hz,
            metric=metric,
            hg_low_hz=hg_low_hz,
            hg_high_hz=hg_high_hz,
            hg_use_hilbert=hg_use_hilbert,
        )
        threshold_mode = threshold_mode.strip().lower()
        thr_val = _resolve_threshold(color_matrix, threshold_mode, threshold_z, threshold_quantile)
        show_gray_bg = (threshold_mode != "none" and threshold_style.strip().lower() == "dim")
        cmap_spec: dict = {"kind": "diverging", "neutral_color": neutral_color}
    else:
        pts, color_matrix, n_frames, meta = load_singleunit_data(
            npz_dir=npz_dir,
            electrodes_tsv=electrodes_tsv,
            film_time_start=film_time_start,
            duration_sec=duration_sec,
            out_fps=out_fps,
        )
        threshold_mode = "none"
        thr_val = 0.0
        show_gray_bg = False
        cmap_spec = {"kind": "binary"}

    clim              = meta["clim"]
    show_scalar_bar   = meta.get("show_scalar_bar", False)
    scalar_bar_title  = meta.get("scalar_bar_title", "")
    print(f"  {n_frames} frames × {pts.shape[0]} electrodes  |  clim={clim}")

    # ── 2. Resolve layout ──────────────────────────────────────────────────
    brain_views = int(brain_views)
    if brain_views not in (1, 3):
        raise ValueError("--brain-views must be 1 or 3.")

    use_embed = embed_video_path is not None and Path(embed_video_path).is_file()
    if embed_video_path is not None and not use_embed:
        print(f"Warning: embed video not found: {embed_video_path}")

    top_frac = float(np.clip(embed_top_frac, 0.0, 0.8)) if use_embed else 0.0
    top_h    = int(round(height * top_frac))
    gap_h    = max(0, int(embed_gap_px)) if use_embed else 0
    brain_h  = max(200, height - top_h - gap_h)

    render_scale = float(np.clip(render_scale, 0.1, 1.0))
    render_w = max(64, int(round(width  * render_scale))); render_w += render_w % 2
    render_h = max(64, int(round(brain_h * render_scale))); render_h += render_h % 2
    if render_scale < 1.0:
        print(f"  Rendering at {render_w}×{render_h} (scale={render_scale}); ffmpeg upscales.")

    base_angles = [(78.0, 14.0)] if brain_views == 1 else [(90.0, 0.0), (0.0, 89.0), (0.0, 0.0)]
    az_off = _parse_offsets(view_azim_offsets, len(base_angles))
    el_off = _parse_offsets(view_elev_offsets, len(base_angles))
    view_angles = [
        (az + azim_ccw_deg + az_off[i], el + el_off[i])
        for i, (az, el) in enumerate(base_angles)
    ]
    spin_mask = _parse_spin_mask(spin_view_mask, len(view_angles))

    # Parallel scale: correct for brain panel being smaller than full height
    from nilearn import datasets as _ds, surface as _surf
    _fs = _ds.fetch_surf_fsaverage()
    _lc, _ = _surf.load_surf_data(_fs.pial_left)
    _rc, _ = _surf.load_surf_data(_fs.pial_right)
    _all = np.vstack([_lc, _rc])
    _span_max = float(np.max(_all.max(axis=0) - _all.min(axis=0)))
    parallel_scale = _span_max * parallel_scale_factor * (brain_h / height)

    ffmpeg_bin = ffmpeg_exe or shutil.which("ffmpeg") or "ffmpeg"

    # ── 3. Build shared kwargs for all workers ─────────────────────────────
    base_kwargs: dict = dict(
        pts=pts, cmap_spec=cmap_spec, clim=clim,
        threshold_mode=threshold_mode, thr_val=thr_val,
        show_gray_bg=show_gray_bg, show_scalar_bar=show_scalar_bar,
        scalar_bar_title=scalar_bar_title, neutral_color=neutral_color,
        out_fps=out_fps, render_w=render_w, render_h=render_h,
        brain_views=brain_views, view_angles=view_angles,
        spin_mask=spin_mask, spin_deg_per_sec=spin_deg_per_sec,
        camera_dist=camera_distance_factor,
        surface_decim=surface_decim, sulc_contrast=sulc_contrast,
        brain_alpha=brain_alpha, brain_cmap=brain_cmap,
        brain_clim_low=brain_clim_low, brain_clim_high=brain_clim_high,
        brain_solid_color=brain_solid_color, electrode_size=electrode_size,
        parallel_proj=parallel_projection, parallel_scale=parallel_scale,
        show_text=show_text, mode=mode,
    )

    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # ── 4. Render (single worker or parallel) ─────────────────────────────
    n_workers = max(1, int(n_workers))
    with tempfile.TemporaryDirectory(prefix="brain_viewer_") as tmpdir:
        tmp = Path(tmpdir)

        if n_workers == 1:
            brain_silent = tmp / "brain_silent.mp4"
            _render_chunk_worker({
                **base_kwargs,
                "color_chunk": color_matrix,
                "frame_offset": 0,
                "temp_path": str(brain_silent),
            })
        else:
            chunks   = np.array_split(color_matrix, n_workers, axis=0)
            offsets  = [0] + list(np.cumsum([len(c) for c in chunks[:-1]]))
            tmp_chunks = [tmp / f"chunk_{i:03d}.mp4" for i in range(len(chunks))]
            worker_args = [
                {**base_kwargs,
                 "color_chunk": chunks[i],
                 "frame_offset": int(offsets[i]),
                 "temp_path": str(tmp_chunks[i])}
                for i in range(len(chunks))
            ]
            print(f"Rendering {n_frames} frames across {n_workers} workers …")
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(n_workers) as pool:
                pool.map(_render_chunk_worker, worker_args)
            brain_silent = tmp / "brain_silent.mp4"
            _ffmpeg_concat(tmp_chunks, brain_silent, ffmpeg_bin)

        # ── 5. Composite with embedded video + audio ───────────────────────
        print("Compositing …")
        _ffmpeg_composite(
            brain_video=brain_silent,
            embed_video_path=embed_video_path if use_embed else None,
            film_time_start=film_time_start,
            duration_sec=duration_sec,
            out_mp4=out_mp4,
            width=width, height=height,
            top_h=top_h, gap_h=gap_h, brain_h=brain_h,
            embed_width_frac=embed_width_frac,
            embed_x_align=embed_x_align,
            brain_panel_y_shift_px=brain_panel_y_shift_px,
            ffmpeg_bin=ffmpeg_bin,
        )

    print(f"Done → {out_mp4}  ({n_frames} frames @ {out_fps} fps)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _add_arg(p: argparse.ArgumentParser, *args, **kwargs) -> None:
    p.add_argument(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Brain-surface MP4 renderer for iEEG (macro) and single-unit (microwire) data.\n"
            "Select a visualization mode with --mode macro | singleunit."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Mode (required) ───────────────────────────────────────────────────────
    p.add_argument(
        "--mode", required=True, choices=["macro", "singleunit"],
        help="Visualization mode. 'macro': iEEG high-gamma z-score on macro electrodes. "
             "'singleunit': binary spike activity on microwire electrodes.",
    )

    # ── Output / timing ───────────────────────────────────────────────────────
    g = p.add_argument_group("Output / timing")
    g.add_argument("--out-mp4", type=Path, default=None,
        help="Output MP4 path. Defaults to outputs/brain_macro.mp4 or brain_singleunit.mp4.")
    g.add_argument("--film-time-start", type=float, default=70.0,
        help="Start time within the movie (seconds). Both panels seek to this point.")
    g.add_argument("--duration-sec", type=float, default=140.0,
        help="Duration of the output clip (seconds).")
    g.add_argument("--out-fps", type=float, default=20.0,
        help="Output frame rate. Lower values (e.g. 15) render faster.")

    # ── Macro-specific inputs ─────────────────────────────────────────────────
    g = p.add_argument_group("Macro iEEG inputs [--mode macro]")
    g.add_argument("--vhdr-path", type=Path,
        default=_DEFAULTS["macro"]["vhdr_path"],
        help="BrainVision .vhdr file produced by ucla2bids.py.")
    g.add_argument("--align-json", type=Path,
        default=_DEFAULTS["macro"]["align_json"],
        help="Audio-alignment JSON providing movie_start_rel_rec and drift_correction_multiplier.")
    g.add_argument("--no-align-file", action="store_true",
        help="Ignore --align-json and use drift=1.0 / t0=0.0.")
    g.add_argument("--movie-t0-sec", type=float, default=None,
        help="Override movie start relative to recording (seconds). Overrides align_json.")
    g.add_argument("--drift", type=float, default=None,
        help="Override drift correction multiplier. Overrides align_json.")
    g.add_argument("--crop-start", type=float, default=None,
        help="Force iEEG crop start (seconds). Default: auto (film_time_start − 0.5 s).")
    g.add_argument("--notch-hz", type=float, default=60.0,
        help="Notch filter frequency (Hz). Set negative to disable.")
    g.add_argument("--highpass-hz", type=float, default=1.0,
        help="High-pass filter cutoff (Hz). Set negative to disable.")
    g.add_argument("--activity-metric", choices=["rms", "mean_abs", "high_gamma"],
        default="high_gamma",
        help="Per-frame activity metric. 'high_gamma' bandpasses to [hg-low-hz, hg-high-hz] Hz "
             "and computes the Hilbert envelope.")
    g.add_argument("--hg-low-hz", type=float, default=70.0,
        help="Low cutoff for high-gamma bandpass (Hz).")
    g.add_argument("--hg-high-hz", type=float, default=150.0,
        help="High cutoff for high-gamma bandpass (Hz).")
    g.add_argument("--hg-no-hilbert", action="store_true",
        help="Use squared bandpassed signal instead of Hilbert envelope.")

    # ── Macro threshold ───────────────────────────────────────────────────────
    g = p.add_argument_group("Macro iEEG threshold [--mode macro]")
    g.add_argument("--threshold-mode", choices=["none", "fixed", "quantile"],
        default="quantile",
        help="Sub-threshold electrode display: none (show all), fixed (|z| < threshold-z), "
             "quantile (|z| < quantile of |z| distribution).")
    g.add_argument("--threshold-z", type=float, default=2.0,
        help="Fixed |z| threshold when --threshold-mode fixed.")
    g.add_argument("--threshold-quantile", type=float, default=0.95,
        help="Quantile of |z| used as threshold when --threshold-mode quantile.")
    g.add_argument("--threshold-style", choices=["dim", "hide"], default="dim",
        help="How to display sub-threshold electrodes: dim (neutral color) or hide (NaN/invisible).")

    # ── Single-unit inputs ────────────────────────────────────────────────────
    g = p.add_argument_group("Single-unit inputs [--mode singleunit]")
    g.add_argument("--npz-dir", type=Path, default=_DEFAULTS["singleunit"]["npz_dir"],
        help="Directory containing *_spikedata.npz files produced by ucla2bids.py.")

    # ── Shared electrode TSV ──────────────────────────────────────────────────
    p.add_argument("--electrodes-tsv", type=Path,
        default=_DEFAULTS["macro"]["electrodes_tsv"],
        help="BIDS electrodes.tsv with MNI coordinates (used by both modes).")

    # ── Embedded movie panel ──────────────────────────────────────────────────
    g = p.add_argument_group("Embedded movie panel (top)")
    g.add_argument("--embed-video-path", type=Path, default=_EMBED_DEFAULT,
        help="Movie file to embed as the top panel. Audio is muxed into the output. "
             "Set to empty string to disable.")
    g.add_argument("--embed-top-frac", type=float, default=0.45,
        help="Fraction of output height reserved for the embedded movie panel (0–0.8).")
    g.add_argument("--embed-gap-px", type=int, default=0,
        help="Vertical gap in pixels between the movie panel and brain panel.")
    g.add_argument("--embed-width-frac", type=float, default=0.4,
        help="Width of the movie panel as a fraction of output width (0.1–1.0).")
    g.add_argument("--embed-x-align", choices=["left", "center", "right"], default="center",
        help="Horizontal alignment of the movie panel within the full output width.")
    g.add_argument("--brain-panel-y-shift-px", type=int, default=14,
        help="Vertical shift (px) applied to the brain panel. Positive moves it down.")

    # ── Output dimensions ─────────────────────────────────────────────────────
    g = p.add_argument_group("Output dimensions")
    g.add_argument("--width", type=int, default=1280, help="Output video width (pixels).")
    g.add_argument("--height", type=int, default=720, help="Output video height (pixels).")
    g.add_argument("--brain-views", type=int, default=3,
        help="Number of brain views: 1 (single lateral) or 3 (lateral L / medial / lateral R).")

    # ── Rendering performance ─────────────────────────────────────────────────
    g = p.add_argument_group("Rendering performance")
    g.add_argument("--n-workers", type=int, default=4,
        help="Number of parallel PyVista render workers. Use 1 on GPU nodes "
             "(hardware OpenGL is faster than multiple CPU processes).")
    g.add_argument("--render-scale", type=float, default=1.0,
        help="Render resolution multiplier (0.1–1.0). 0.5 renders at half size "
             "(4× fewer pixels → ~4× faster); ffmpeg upscales in the composite step.")

    # ── Brain surface ─────────────────────────────────────────────────────────
    g = p.add_argument_group("Brain surface")
    g.add_argument("--brain-alpha", type=float, default=0.6,
        help="Cortical surface opacity (0=transparent, 1=opaque).")
    g.add_argument("--surface-decim", type=int, default=1,
        help="Mesh decimation step. 1 = full detail (163k vertices). "
             "Higher values reduce triangle count for faster rendering.")
    g.add_argument("--sulc-contrast", type=float, default=1.0,
        help="Gamma contrast for sulcal depth shading. >1 deepens sulci, <1 flattens.")
    g.add_argument("--brain-cmap", type=str, default="gray",
        help="Matplotlib colormap for sulcal depth shading. Examples: gray, bone, cividis.")
    g.add_argument("--brain-clim-low", type=float, default=0.05,
        help="Lower bound for sulcal colormap (0–1). Slightly above 0 avoids pure black sulci.")
    g.add_argument("--brain-clim-high", type=float, default=0.95,
        help="Upper bound for sulcal colormap (0–1). Slightly below 1 avoids pure white gyri.")
    g.add_argument("--brain-solid-color", type=str, default="#6f6f6f",
        help="If set, use a uniform cortex color instead of sulcal depth shading. "
             "Empty string to disable and show sulcal map.")

    # ── Electrodes ────────────────────────────────────────────────────────────
    g = p.add_argument_group("Electrodes")
    g.add_argument("--electrode-size", type=float, default=14.0,
        help="Electrode sphere size in render pixels.")
    g.add_argument("--neutral-color", type=str, default="#8a8a8a",
        help="Color for sub-threshold (macro) or background electrode dots.")

    # ── Camera ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Camera")
    g.add_argument("--camera-distance-factor", type=float, default=5.8,
        help="Camera distance as a multiple of brain radius. Increase to zoom out.")
    g.add_argument("--parallel-scale-factor", type=float, default=1.0,
        help="Scale multiplier for parallel (orthographic) projection zoom. "
             "Increase to zoom out. Automatically corrected for brain panel height.")
    g.add_argument("--perspective", action="store_true",
        help="Use perspective projection instead of the default parallel (orthographic).")
    g.add_argument("--spin-deg-per-sec", type=float, default=3.0,
        help="Camera rotation speed in degrees/second. 0 = fixed camera.")
    g.add_argument("--spin-view-mask", type=str, default=None,
        help="Comma-separated 0/1 flags enabling spin per view. "
             "Default for 3 views: '1,0,1' (lateral views spin, medial fixed).")
    g.add_argument("--azim-ccw-deg", type=float, default=0.0,
        help="Counter-clockwise azimuth offset applied to all views (degrees).")
    g.add_argument("--view-azim-offsets", type=str, default="0,270,270",
        help="Per-view azimuth offsets in degrees (comma-separated). "
             "A single value is broadcast to all views.")
    g.add_argument("--view-elev-offsets", type=str, default="0,180,0",
        help="Per-view elevation offsets in degrees (comma-separated). "
             "A single value is broadcast to all views.")

    # ── Misc ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Miscellaneous")
    g.add_argument("--no-text", action="store_true",
        help="Suppress the title text overlay in the brain panel.")
    g.add_argument("--ffmpeg-exe", type=str, default=None,
        help="Explicit path to the ffmpeg executable. Auto-detected if not set.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve mode-specific output path default
    if args.out_mp4 is None:
        args.out_mp4 = _DEFAULTS[args.mode]["out_mp4"]

    # Disable embed if empty string was passed
    embed = args.embed_video_path
    if embed is not None and str(embed).strip() == "":
        embed = None

    run(
        mode=args.mode,
        out_mp4=args.out_mp4,
        film_time_start=args.film_time_start,
        duration_sec=args.duration_sec,
        out_fps=args.out_fps,
        # macro
        vhdr_path=args.vhdr_path,
        electrodes_tsv=args.electrodes_tsv,
        align_json=None if args.no_align_file else args.align_json,
        movie_t0_override=args.movie_t0_sec,
        drift_override=args.drift,
        crop_start=args.crop_start,
        notch_hz=None if args.notch_hz < 0 else args.notch_hz,
        highpass_hz=None if args.highpass_hz < 0 else args.highpass_hz,
        metric=args.activity_metric,
        hg_low_hz=args.hg_low_hz,
        hg_high_hz=args.hg_high_hz,
        hg_use_hilbert=not args.hg_no_hilbert,
        threshold_mode=args.threshold_mode,
        threshold_z=args.threshold_z,
        threshold_quantile=args.threshold_quantile,
        threshold_style=args.threshold_style,
        # singleunit
        npz_dir=args.npz_dir,
        # layout
        width=args.width,
        height=args.height,
        brain_views=args.brain_views,
        embed_video_path=embed,
        embed_top_frac=args.embed_top_frac,
        embed_gap_px=args.embed_gap_px,
        embed_width_frac=args.embed_width_frac,
        embed_x_align=args.embed_x_align,
        brain_panel_y_shift_px=args.brain_panel_y_shift_px,
        # rendering
        n_workers=args.n_workers,
        render_scale=args.render_scale,
        # brain surface
        surface_decim=args.surface_decim,
        sulc_contrast=args.sulc_contrast,
        brain_alpha=args.brain_alpha,
        brain_cmap=args.brain_cmap,
        brain_clim_low=args.brain_clim_low,
        brain_clim_high=args.brain_clim_high,
        brain_solid_color=args.brain_solid_color if args.brain_solid_color else None,
        # electrodes
        electrode_size=args.electrode_size,
        neutral_color=args.neutral_color,
        # camera
        camera_distance_factor=args.camera_distance_factor,
        parallel_projection=not args.perspective,
        parallel_scale_factor=args.parallel_scale_factor,
        spin_deg_per_sec=args.spin_deg_per_sec,
        spin_view_mask=args.spin_view_mask,
        azim_ccw_deg=args.azim_ccw_deg,
        view_azim_offsets=args.view_azim_offsets,
        view_elev_offsets=args.view_elev_offsets,
        # misc
        show_text=not args.no_text,
        ffmpeg_exe=args.ffmpeg_exe,
    )


if __name__ == "__main__":
    main()
