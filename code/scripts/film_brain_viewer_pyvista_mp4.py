#!/usr/bin/env python3
"""
High-detail brain-only MP4 for macro iEEG activity using PyVista/VTK.

Compared with matplotlib 3D, this backend provides much better cortical
surface detail (gyri/sulci) and lighting control for publication-style videos.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import warnings
from pathlib import Path

import imageio.v2 as iio
import mne
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.signal import hilbert
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from nilearn import datasets, surface
from PIL import Image


def _read_align(path: Path | None) -> tuple[float, float]:
    if path is None or not path.is_file():
        return 0.0, 1.0
    data = json.loads(path.read_text())
    t0 = float(data.get("start_rel_rec", [0.0])[0])
    drift = float(data.get("drift_correction_multiplier", [1.0])[0])
    return t0, drift


def _resize_rgb(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    im = Image.fromarray(arr)
    im = im.resize((w, h), Image.Resampling.LANCZOS)
    return np.asarray(im)


def _fit_rgb(
    arr: np.ndarray,
    box_w: int,
    box_h: int,
    mode: str,
) -> np.ndarray:
    """Fit frame into box with stretch/contain/cover behavior."""
    box_w = max(1, int(box_w))
    box_h = max(1, int(box_h))
    mode_k = mode.strip().lower()
    if mode_k == "stretch":
        return _resize_rgb(arr, box_w, box_h)

    h0, w0 = arr.shape[:2]
    if h0 <= 0 or w0 <= 0:
        return np.zeros((box_h, box_w, 3), dtype=np.uint8)

    if mode_k == "contain":
        s = min(box_w / w0, box_h / h0)
        w1 = max(1, int(round(w0 * s)))
        h1 = max(1, int(round(h0 * s)))
        out = np.zeros((box_h, box_w, 3), dtype=np.uint8)
        fit = _resize_rgb(arr, w1, h1)
        x0 = (box_w - w1) // 2
        y0 = (box_h - h1) // 2
        out[y0 : y0 + h1, x0 : x0 + w1] = fit
        return out

    if mode_k == "cover":
        s = max(box_w / w0, box_h / h0)
        w1 = max(1, int(round(w0 * s)))
        h1 = max(1, int(round(h0 * s)))
        fit = _resize_rgb(arr, w1, h1)
        x0 = (w1 - box_w) // 2
        y0 = (h1 - box_h) // 2
        return fit[y0 : y0 + box_h, x0 : x0 + box_w]

    raise ValueError(f"Unknown embed fit mode '{mode}'. Use stretch, contain, or cover.")


class _VideoFrames:
    """Read frames by time from a movie file."""

    def __init__(self, path: Path):
        self._reader = None
        self._cap = None
        self._kind = ""
        self.fps = 30.0
        self.nframes = 10**9
        try:
            self._reader = iio.get_reader(str(path), "ffmpeg")
            meta = self._reader.get_meta_data()
            self.fps = float(meta.get("fps", 30) or 30)
            try:
                self.nframes = int(self._reader.count_frames())
            except Exception:
                self.nframes = 10**9
            self._kind = "imageio"
            return
        except Exception as img_err:
            try:
                import cv2

                self._cap = cv2.VideoCapture(str(path))
                if not self._cap.isOpened():
                    raise RuntimeError("cv2.VideoCapture failed to open file") from img_err
                self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
                if self.fps <= 0:
                    self.fps = 30.0
                n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.nframes = n if n > 0 else 10**9
                self._kind = "cv2"
                return
            except Exception as cv_err:
                raise RuntimeError(
                    "Could not open embedded video. Tried imageio[ffmpeg] and OpenCV.\n"
                    f"  imageio error: {img_err}\n"
                    f"  cv2 error: {cv_err}"
                ) from cv_err

    def frame_rgb(self, t_sec: float, w: int, h: int) -> np.ndarray:
        idx = int(np.clip(round(float(t_sec) * self.fps), 0, max(0, self.nframes - 1)))
        if self._kind == "imageio":
            arr = self._reader.get_data(idx)
        else:
            import cv2

            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = self._cap.read()
            if not ok:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bgr = self._cap.read()
            if not ok:
                raise RuntimeError("cv2 failed to read frame")
            arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return _resize_rgb(arr, w, h)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def _load_macro_coords(electrodes_tsv: Path, channels: list[str]) -> pd.DataFrame:
    el = pd.read_csv(electrodes_tsv, sep="\t")
    macro = el[el["group"].astype(str).str.lower() == "macro"].copy()
    macro = macro[macro["name"].isin(channels)].copy()
    for c in ("x", "y", "z"):
        macro[c] = pd.to_numeric(macro[c], errors="coerce")
    macro = macro.dropna(subset=["x", "y", "z"])
    if macro.empty:
        raise ValueError("No macro channels with valid coordinates.")
    return macro


def _n_frames_fits(
    n_samp: int,
    sfreq: float,
    crop_start_rec: float,
    ieeg_movie_t0: float,
    drift: float,
    film_time_start: float,
    out_fps: float,
    n_want: int,
) -> int:
    win = max(1, int(round(sfreq / out_fps)))
    for n in range(int(n_want), 0, -1):
        t_mov = film_time_start + (n - 0.5) / out_fps
        t_rec = ieeg_movie_t0 + t_mov * drift
        t_in = t_rec - crop_start_rec
        c0 = int(np.round(t_in * sfreq - win / 2))
        c0 = int(np.clip(c0, 0, n_samp - win))
        if c0 + win <= n_samp:
            return n
    return 1


def _precompute_metric_z(
    data: np.ndarray,
    sfreq: float,
    crop_start_rec: float,
    ieeg_movie_t0: float,
    drift: float,
    film_time_start: float,
    out_fps: float,
    n_frames: int,
    metric: str,
    hg_low_hz: float,
    hg_high_hz: float,
    hg_use_hilbert: bool,
) -> np.ndarray:
    n_ch, n_samp = data.shape
    win = max(1, int(round(sfreq / out_fps)))
    values = np.empty((n_frames, n_ch), dtype=np.float64)

    metric_key = metric.strip().lower()
    if metric_key == "high_gamma":
        data_m = mne.filter.filter_data(
            data.copy(),
            sfreq=sfreq,
            l_freq=float(hg_low_hz),
            h_freq=float(hg_high_hz),
            method="fir",
            verbose="ERROR",
        )
        if hg_use_hilbert:
            data_m = np.abs(hilbert(data_m, axis=1))
        else:
            data_m = data_m**2
    else:
        data_m = data

    for k in range(n_frames):
        t_mov = film_time_start + (k + 0.5) / out_fps
        t_rec = ieeg_movie_t0 + t_mov * drift
        t_in_segment = t_rec - crop_start_rec
        c0 = int(np.round(t_in_segment * sfreq - win / 2))
        c0 = int(np.clip(c0, 0, n_samp - win))
        sl = data_m[:, c0 : c0 + win]
        if metric_key == "rms":
            values[k] = np.sqrt(np.mean(sl**2, axis=1))
        elif metric_key == "mean_abs":
            values[k] = np.mean(np.abs(sl), axis=1)
        elif metric_key == "high_gamma":
            values[k] = np.mean(sl, axis=1)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use rms, mean_abs, or high_gamma.")

    med = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - med), axis=0, keepdims=True)
    scale = np.maximum(1e-9, 1.4826 * mad)
    z = (values - med) / scale
    return np.clip(z, -4.0, 4.0)


def _decimate_faces(coords: np.ndarray, faces: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step_i = max(1, int(step))
    if step_i == 1:
        used = np.arange(coords.shape[0], dtype=np.int64)
        return coords, faces, used
    faces_d = faces[::step_i]
    if faces_d.size == 0:
        used = np.arange(coords.shape[0], dtype=np.int64)
        return coords, faces, used
    used = np.unique(faces_d.ravel())
    idx_map = np.full(coords.shape[0], -1, dtype=np.int64)
    idx_map[used] = np.arange(used.size, dtype=np.int64)
    coords_d = coords[used]
    faces_d = idx_map[faces_d]
    return coords_d, faces_d.astype(np.int32), used


def _to_pv_faces(faces: np.ndarray) -> np.ndarray:
    n = faces.shape[0]
    return np.c_[np.full(n, 3, dtype=np.int32), faces].ravel()


def _normalize_sulc(vals: np.ndarray, contrast: float) -> np.ndarray:
    lo, hi = np.percentile(vals, [5, 95])
    den = max(1e-9, float(hi - lo))
    x = np.clip((vals - lo) / den, 0.0, 1.0)
    # Gamma-like control for fold contrast.
    g = max(0.2, float(contrast))
    return np.clip(x**g, 0.0, 1.0)


def _make_fsaverage_brain_polydata(surface_decim: int, sulc_contrast: float) -> pv.PolyData:
    fsaverage = datasets.fetch_surf_fsaverage()
    lh_coords, lh_faces = surface.load_surf_data(fsaverage.pial_left)
    rh_coords, rh_faces = surface.load_surf_data(fsaverage.pial_right)
    lh_sulc = surface.load_surf_data(fsaverage.sulc_left).astype(np.float64)
    rh_sulc = surface.load_surf_data(fsaverage.sulc_right).astype(np.float64)

    lh_coords, lh_faces, lh_used = _decimate_faces(
        lh_coords.astype(np.float64), lh_faces.astype(np.int32), surface_decim
    )
    rh_coords, rh_faces, rh_used = _decimate_faces(
        rh_coords.astype(np.float64), rh_faces.astype(np.int32), surface_decim
    )

    lh_poly = pv.PolyData(lh_coords, _to_pv_faces(lh_faces))
    rh_poly = pv.PolyData(rh_coords, _to_pv_faces(rh_faces))
    lh_poly.point_data["sulc"] = _normalize_sulc(lh_sulc[lh_used], sulc_contrast)
    rh_poly.point_data["sulc"] = _normalize_sulc(rh_sulc[rh_used], sulc_contrast)
    return lh_poly.merge(rh_poly)


def _camera_from_angles(
    center: np.ndarray, radius: float, azim_deg: float, elev_deg: float, distance_factor: float
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pos = center + direction * (float(distance_factor) * radius)
    return tuple(pos), tuple(center), (0.0, 0.0, 1.0)


def _electrode_cmap(neutral_color: str) -> LinearSegmentedColormap:
    """
    Diverging colormap with dark neutral center (not white),
    so electrodes remain readable on black background.
    """
    colors = [
        "#1f4e79",  # strong negative: deep blue
        "#2c7fb8",
        neutral_color,  # near-zero and below-threshold dim color
        "#ffd54a",
        "#fff200",  # strong positive: bright yellow
    ]
    return LinearSegmentedColormap.from_list("dark_diverging", colors, N=256)


def _parse_angle_offsets(spec: str | None, n: int) -> list[float]:
    """
    Parse comma-separated per-view angle offsets.

    Examples:
        "0,0,15" -> [0.0, 0.0, 15.0]
        "10" -> [10.0, 10.0, 10.0]
    """
    if spec is None or str(spec).strip() == "":
        return [0.0] * n
    parts = [p.strip() for p in str(spec).split(",") if p.strip() != ""]
    vals = [float(p) for p in parts]
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"Expected 1 or {n} offsets, got {len(vals)} from '{spec}'.")
    return vals


def _parse_spin_mask(spec: str | None, n: int) -> list[bool]:
    """
    Parse per-view spin mask from comma-separated 0/1 values.

    Examples:
        "1,0,1" -> [True, False, True]
        "1" -> [True, True, True]
    """
    if spec is None or str(spec).strip() == "":
        # Default for 3 views: keep center fixed.
        if n == 3:
            return [True, False, True]
        return [True] * n
    parts = [p.strip() for p in str(spec).split(",") if p.strip() != ""]
    vals = [p in ("1", "true", "True", "yes", "y") for p in parts]
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"Expected 1 or {n} spin-mask values, got {len(vals)} from '{spec}'.")
    return vals


def run(
    vhdr_path: Path,
    electrodes_tsv: Path,
    out_mp4: Path,
    align_json: Path | None,
    film_time_start: float,
    duration_sec: float,
    out_fps: float,
    notch_hz: float | None,
    highpass_hz: float | None,
    crop_start: float | None,
    movie_t0_in_recording: float | None,
    drift: float | None,
    width: int,
    height: int,
    brain_views: int,
    electrode_size: float,
    brain_alpha: float,
    surface_decim: int,
    sulc_contrast: float,
    show_text: bool,
    camera_distance_factor: float,
    spin_deg_per_sec: float,
    parallel_projection: bool,
    parallel_scale_factor: float,
    brain_cmap: str,
    brain_clim_low: float,
    brain_clim_high: float,
    brain_solid_color: str | None,
    neutral_color: str,
    activity_metric: str,
    hg_low_hz: float,
    hg_high_hz: float,
    hg_use_hilbert: bool,
    threshold_mode: str,
    threshold_z: float,
    threshold_quantile: float,
    threshold_style: str,
    embed_video_path: Path | None,
    embed_top_frac: float,
    embed_gap_px: int,
    embed_width_frac: float,
    embed_x_align: str,
    embed_fit: str,
    brain_panel_y_shift_px: int,
    azim_ccw_deg: float,
    view_azim_offsets: str | None,
    view_elev_offsets: str | None,
    spin_view_mask: str | None,
    ffmpeg_exe: str | None,
) -> None:
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
    tmax_rec = float(raw.times[-1])
    t_align, drift_d = _read_align(align_json)
    if movie_t0_in_recording is not None:
        t_align = float(movie_t0_in_recording)
    if drift is not None:
        drift_d = float(drift)

    t_mov_end = film_time_start + duration_sec
    t_rec_end = t_align + t_mov_end * drift_d
    t_rec_start = t_align + film_time_start * drift_d

    if crop_start is not None:
        rec_lo = max(0.0, float(crop_start))
    else:
        rec_lo = max(0.0, t_rec_start - 0.5)
    rec_hi = min(tmax_rec, t_rec_end + 0.5)
    if rec_lo >= rec_hi:
        raise ValueError(f"Invalid crop: rec_lo={rec_lo} rec_hi={rec_hi}.")

    raw.crop(tmin=rec_lo, tmax=rec_hi)
    raw.load_data()
    if notch_hz is not None:
        raw.notch_filter(freqs=[notch_hz], verbose="ERROR")
    if highpass_hz is not None:
        raw.filter(l_freq=highpass_hz, h_freq=None, verbose="ERROR")

    coords = _load_macro_coords(electrodes_tsv, list(raw.ch_names))
    ch_names = list(coords["name"])
    raw.pick(ch_names)
    data = raw.get_data() * 1e6
    sfreq = float(raw.info["sfreq"])
    n_want = max(1, int(np.floor(duration_sec * out_fps)))
    n_frames = _n_frames_fits(
        n_samp=data.shape[1],
        sfreq=sfreq,
        crop_start_rec=rec_lo,
        ieeg_movie_t0=t_align,
        drift=drift_d,
        film_time_start=film_time_start,
        out_fps=out_fps,
        n_want=n_want,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = _precompute_metric_z(
            data=data,
            sfreq=sfreq,
            crop_start_rec=rec_lo,
            ieeg_movie_t0=t_align,
            drift=drift_d,
            film_time_start=film_time_start,
            out_fps=out_fps,
            n_frames=n_frames,
            metric=activity_metric,
            hg_low_hz=hg_low_hz,
            hg_high_hz=hg_high_hz,
            hg_use_hilbert=hg_use_hilbert,
        )
    cmax = float(np.nanpercentile(np.abs(z), 98))
    cmax = max(cmax, 1.0)
    clim = (-cmax, cmax)

    brain_views = int(brain_views)
    if brain_views not in (1, 3):
        raise ValueError("--brain-views must be 1 or 3.")

    pts = np.c_[coords["x"].to_numpy(), coords["y"].to_numpy(), coords["z"].to_numpy()]

    if brain_views == 1:
        view_angles = [(78.0, 14.0)]
        shape = (1, 1)
    else:
        # sagittal / axial / coronal-like panel views
        view_angles = [(90.0, 0.0), (0.0, 89.0), (0.0, 0.0)]
        shape = (1, 3)
    az_off = _parse_angle_offsets(view_azim_offsets, len(view_angles))
    el_off = _parse_angle_offsets(view_elev_offsets, len(view_angles))
    spin_mask = _parse_spin_mask(spin_view_mask, len(view_angles))
    view_angles = [
        (az + float(azim_ccw_deg) + az_off[i], el + el_off[i])
        for i, (az, el) in enumerate(view_angles)
    ]

    use_embed = embed_video_path is not None and Path(embed_video_path).is_file()
    top_frac = float(np.clip(embed_top_frac, 0.0, 0.8)) if use_embed else 0.0
    top_h = int(round(height * top_frac))
    gap_h = max(0, int(embed_gap_px)) if use_embed else 0
    brain_h = max(200, int(height - top_h - gap_h))
    brain_w = int(width)
    video_reader: _VideoFrames | None = _VideoFrames(Path(embed_video_path)) if use_embed else None

    pv.OFF_SCREEN = True
    try:
        pv.start_xvfb()
    except Exception:
        pass

    plotter = pv.Plotter(off_screen=True, shape=shape, window_size=(brain_w, brain_h))
    brain_poly = _make_fsaverage_brain_polydata(surface_decim=surface_decim, sulc_contrast=sulc_contrast)
    b = brain_poly.bounds
    center = np.array(
        [(b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5, (b[4] + b[5]) * 0.5],
        dtype=np.float64,
    )
    brain_span = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]], dtype=np.float64)
    radius = float(np.max(brain_span) * 0.55)
    elec_cmap = _electrode_cmap(neutral_color=neutral_color)
    elec_meshes: list[pv.PolyData] = []
    base_view_angles: list[tuple[float, float]] = []
    threshold_mode = threshold_mode.strip().lower()
    threshold_style = threshold_style.strip().lower()
    if threshold_mode not in ("none", "fixed", "quantile"):
        raise ValueError("--threshold-mode must be one of: none, fixed, quantile.")
    if threshold_style not in ("dim", "hide"):
        raise ValueError("--threshold-style must be one of: dim, hide.")
    if threshold_mode == "fixed":
        thr_val = float(threshold_z)
    elif threshold_mode == "quantile":
        thr_val = float(np.quantile(np.abs(z), float(threshold_quantile)))
    else:
        thr_val = 0.0
    show_gray_background = threshold_mode != "none" and threshold_style == "dim"

    for i, (az, el) in enumerate(view_angles):
        plotter.subplot(0, i)
        plotter.set_background("black")
        plotter.add_mesh(
            brain_poly,
            scalars=None if brain_solid_color else "sulc",
            color=brain_solid_color if brain_solid_color else None,
            cmap=brain_cmap if not brain_solid_color else None,
            clim=[float(brain_clim_low), float(brain_clim_high)] if not brain_solid_color else None,
            opacity=float(brain_alpha),
            smooth_shading=True,
            specular=0.05,
            specular_power=12.0,
            ambient=0.20,
            diffuse=0.45,
            show_scalar_bar=False,
        )
        # Constant background electrode layer (single neutral color).
        bg = pv.PolyData(pts.copy())
        bg.point_data["const"] = np.ones(pts.shape[0], dtype=np.float64)
        bg_actor = plotter.add_mesh(
            bg,
            scalars="const",
            color=neutral_color,
            render_points_as_spheres=True,
            point_size=float(electrode_size),
            ambient=0.35,
            diffuse=0.90,
            specular=0.10,
            show_scalar_bar=False,
        )
        bg_actor.SetVisibility(bool(show_gray_background))

        em = pv.PolyData(pts.copy())
        em.point_data["activity"] = z[0].astype(np.float64)
        act = plotter.add_mesh(
            em,
            scalars="activity",
            cmap=elec_cmap,
            clim=clim,
            render_points_as_spheres=True,
            point_size=float(electrode_size),
            ambient=0.35,
            diffuse=0.90,
            specular=0.15,
            nan_color="#000000",
            show_scalar_bar=False,
        )
        cam = _camera_from_angles(
            center,
            radius,
            azim_deg=az,
            elev_deg=el,
            distance_factor=camera_distance_factor,
        )
        plotter.camera_position = cam
        plotter.camera.parallel_projection = bool(parallel_projection)
        if parallel_projection:
            # In parallel projection, zoom is controlled by parallel_scale (not camera distance).
            pscale = float(np.max(brain_span) * parallel_scale_factor)
            # Preserve pre-embed apparent brain size when the brain panel height is reduced.
            if use_embed and height > 0:
                pscale *= float(brain_h / float(height))
            plotter.camera.parallel_scale = pscale
        plotter.enable_anti_aliasing("msaa")
        plotter.add_light(pv.Light(position=(300, 200, 300), focal_point=tuple(center), intensity=0.9))
        if show_text and i == 0:
            plotter.add_text(
                "Macro iEEG activity on cortical surface",
                position="upper_left",
                color="white",
                font_size=12,
            )
        elec_meshes.append(em)
        base_view_angles.append((az, el))

    # scalar bar only on the last subplot
    plotter.subplot(0, len(view_angles) - 1)
    plotter.add_scalar_bar(
        title=f"z({activity_metric})",
        n_labels=5,
        color="white",
        title_font_size=14,
        label_font_size=11,
        fmt="%.1f",
    )

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    do_audio_mux = bool(use_embed)
    silent_out = (
        out_mp4.with_name(f"{out_mp4.stem}.silent_tmp{out_mp4.suffix}") if do_audio_mux else out_mp4
    )
    writer = iio.get_writer(
        str(silent_out),
        fps=out_fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_log_level="error",
    )
    for k in range(n_frames):
        t_sec = k / out_fps
        for i in range(len(view_angles)):
            plotter.subplot(0, i)
            frame_vals = z[k].astype(np.float64).copy()
            if threshold_mode != "none":
                mask = np.abs(frame_vals) < thr_val
                if threshold_style == "dim":
                    # Keep exact one-color background for sub-threshold electrodes.
                    frame_vals[mask] = np.nan
                else:
                    frame_vals[mask] = np.nan
            elec_meshes[i]["activity"] = frame_vals
            elec_meshes[i].set_active_scalars("activity")
            elec_meshes[i].Modified()
            if spin_deg_per_sec != 0.0 and spin_mask[i]:
                az0, el0 = base_view_angles[i]
                az_t = az0 + spin_deg_per_sec * t_sec
                plotter.camera_position = _camera_from_angles(
                    center,
                    radius,
                    azim_deg=az_t,
                    elev_deg=el0,
                    distance_factor=camera_distance_factor,
                )
        plotter.render()
        brain_frame = plotter.screenshot(return_img=True)
        if use_embed and video_reader is not None:
            full = np.zeros((height, width, 3), dtype=np.uint8)
            t_movie = film_time_start + t_sec
            top_src = video_reader.frame_rgb(t_movie, width, top_h)
            panel_w = max(1, int(round(width * float(np.clip(embed_width_frac, 0.1, 1.0)))))
            panel_h = max(1, top_h)
            top_panel = _fit_rgb(top_src, panel_w, panel_h, mode=embed_fit)
            align_k = embed_x_align.strip().lower()
            if align_k == "left":
                x0 = 0
            elif align_k == "right":
                x0 = width - panel_w
            else:
                x0 = (width - panel_w) // 2
            full[:panel_h, x0 : x0 + panel_w, :] = top_panel
            # Optional vertical shift for the brain panel placement.
            y0 = top_h + gap_h + int(brain_panel_y_shift_px)
            y1 = y0 + brain_h
            dst_y0 = max(0, y0)
            dst_y1 = min(height, y1)
            if dst_y1 > dst_y0:
                src_y0 = max(0, -y0)
                src_y1 = src_y0 + (dst_y1 - dst_y0)
                full[dst_y0:dst_y1, :, :] = brain_frame[src_y0:src_y1, :, :]
            writer.append_data(full)
        else:
            writer.append_data(brain_frame)
    writer.close()
    if video_reader is not None:
        video_reader.close()
    plotter.close()
    if do_audio_mux:
        ffmpeg_bin = ffmpeg_exe if ffmpeg_exe else shutil.which("ffmpeg")
        if ffmpeg_bin:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-ss",
                str(float(film_time_start)),
                "-t",
                str(float(duration_sec)),
                "-i",
                str(embed_video_path),
                "-i",
                str(silent_out),
                "-map",
                "1:v:0",
                "-map",
                "0:a:0?",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(out_mp4),
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    silent_out.unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception:
                # Keep silent video if audio mux fails for any reason.
                try:
                    if out_mp4.exists():
                        out_mp4.unlink()
                    silent_out.rename(out_mp4)
                except Exception:
                    pass
                print("Warning: audio mux failed; wrote video without audio.")
        else:
            try:
                if out_mp4.exists():
                    out_mp4.unlink()
                silent_out.rename(out_mp4)
            except Exception:
                pass
            print("Warning: ffmpeg not found; wrote video without audio.")
    print(f"Wrote {out_mp4}  ({n_frames} frames @ {out_fps} fps)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="High-detail PyVista brain-only MP4 for macro iEEG.")
    p.add_argument(
        "--vhdr-path",
        type=Path,
        default=Path(
            "/store/scratch/bsow/Documents/UCLA_24/data/bids/sub-572/ses-01/ieeg/"
            "sub-572_ses-01_task-movie24presleep_acq-macro_run-01_ieeg.vhdr"
        ),
    )
    p.add_argument(
        "--electrodes-tsv",
        type=Path,
        default=Path("/store/scratch/bsow/Documents/UCLA_24/data/bids/sub-572/ses-01/ieeg/sub-572_ses-01_electrodes.tsv"),
    )
    p.add_argument(
        "--out-mp4",
        type=Path,
        default=Path("/store/scratch/bsow/Documents/UCLA_24/outputs/brain_pyvista.mp4"),
    )
    p.add_argument(
        "--align-json",
        type=Path,
        default=Path(
            "/store/scratch/bsow/Documents/UCLA_24/data/ucla_data/572/Experiment-9/Audio/"
            "572_exp_09_preSleep_movie_24_audio_movie_start_time.json"
        ),
    )
    p.add_argument("--no-align-file", action="store_true")
    p.add_argument("--movie-t0-sec", type=float, default=None)
    p.add_argument("--drift", type=float, default=None)
    p.add_argument("--film-time-start", type=float, default=70.0)
    p.add_argument("--duration-sec", type=float, default=140.0)
    p.add_argument(
        "--embed-video-path",
        type=Path,
        default=Path("/store/scratch/bsow/Documents/UCLA_24/data/40m_act_24_S06E01_30fps.m4v"),
        help="Optional movie file to embed as a top 'TV' panel.",
    )
    p.add_argument(
        "--embed-top-frac",
        type=float,
        default=0.45,
        help="Fraction of output height reserved for embedded top video (0-0.8).",
    )
    p.add_argument(
        "--embed-gap-px",
        type=int,
        default=0,
        help="Vertical gap in pixels between top embedded video and brains.",
    )
    p.add_argument(
        "--embed-width-frac",
        type=float,
        default=0.4,
        help="Width fraction of top embedded movie panel (0.1-1.0).",
    )
    p.add_argument(
        "--embed-x-align",
        type=str,
        default="center",
        choices=["left", "center", "right"],
        help="Horizontal alignment for the top movie panel.",
    )
    p.add_argument(
        "--embed-fit",
        type=str,
        default="contain",
        choices=["stretch", "contain", "cover"],
        help="How to fit embedded movie inside top panel (contain avoids stretching).",
    )
    p.add_argument(
        "--brain-panel-y-shift-px",
        type=int,
        default=14,
        help="Vertical shift (px) for brain panel placement when embedding movie; positive moves down.",
    )
    p.add_argument(
        "--ffmpeg-exe",
        type=str,
        default=None,
        help="Optional explicit ffmpeg executable path for audio mux.",
    )
    p.add_argument("--out-fps", type=float, default=20.0)
    p.add_argument("--notch-hz", type=float, default=60.0, help="Set <0 to disable.")
    p.add_argument("--highpass-hz", type=float, default=1.0, help="Set <0 to disable.")
    p.add_argument("--crop-start", type=float, default=None)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--brain-views", type=int, default=3, help="1 or 3.")
    p.add_argument("--electrode-size", type=float, default=12.0, help="Point size in pixels.")
    p.add_argument("--brain-alpha", type=float, default=0.6)
    p.add_argument("--surface-decim", type=int, default=1, help="1 keeps highest mesh detail.")
    p.add_argument("--sulc-contrast", type=float, default=1.0, help="Lower values increase fold contrast.")
    p.add_argument(
        "--activity-metric",
        type=str,
        default="high_gamma",
        choices=["rms", "mean_abs", "high_gamma"],
        help="Electrode activity metric per frame.",
    )
    p.add_argument("--hg-low-hz", type=float, default=70.0, help="Low cutoff for high-gamma band.")
    p.add_argument("--hg-high-hz", type=float, default=150.0, help="High cutoff for high-gamma band.")
    p.add_argument(
        "--hg-no-hilbert",
        action="store_true",
        help="Use squared bandpassed signal instead of Hilbert envelope for high-gamma.",
    )
    p.add_argument(
        "--threshold-mode",
        type=str,
        default="quantile",
        choices=["none", "fixed", "quantile"],
        help="Thresholding mode for electrode display.",
    )
    p.add_argument("--threshold-z", type=float, default=2.0, help="Fixed |z| threshold when mode=fixed.")
    p.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
        help="Quantile of |z| when mode=quantile (e.g. 0.90).",
    )
    p.add_argument(
        "--threshold-style",
        type=str,
        default="dim",
        choices=["dim", "hide"],
        help="Below-threshold style: dim to neutral or hide via NaN.",
    )
    p.add_argument(
        "--camera-distance-factor",
        type=float,
        default=5.8,
        help="Larger values zoom out (default 5.8).",
    )
    p.add_argument(
        "--parallel-scale-factor",
        type=float,
        default=1.0,
        help="Parallel projection zoom control; larger values zoom out more.",
    )
    p.add_argument(
        "--brain-cmap",
        type=str,
        default="gray",
        help=(
            "Cortical colormap for sulcal shading. "
            "Examples: gray, bone, cividis, viridis, magma, coolwarm, twilight."
        ),
    )
    p.add_argument(
        "--brain-clim-low",
        type=float,
        default=0.05,
        help="Lower sulcal intensity bound (0-1).",
    )
    p.add_argument(
        "--brain-clim-high",
        type=float,
        default=0.95,
        help="Upper sulcal intensity bound (0-1).",
    )
    p.add_argument(
        "--neutral-color",
        type=str,
        default="#8a8a8a",
        help="Neutral color for near-zero and dimmed-below-threshold electrodes.",
    )
    p.add_argument(
        "--brain-solid-color",
        type=str,
        default="#6f6f6f",
        help="If set (e.g. '#6f6f6f'), use uniform cortex color and disable sulcal colormap shading.",
    )
    p.add_argument(
        "--perspective",
        action="store_true",
        help="Use perspective projection instead of parallel projection.",
    )
    p.add_argument(
        "--spin-deg-per-sec",
        type=float,
        default=3.0,
        help="Optional camera spin speed in degrees/sec (0 = fixed views).",
    )
    p.add_argument(
        "--spin-view-mask",
        type=str,
        default=None,
        help="Per-view spin mask (comma-separated 0/1). Default for 3 views is 1,0,1 (middle fixed).",
    )
    p.add_argument(
        "--azim-ccw-deg",
        type=float,
        default=0.0,
        help="Counter-clockwise azimuth offset (degrees) for all brain views, e.g. 90.",
    )
    p.add_argument(
        "--view-azim-offsets",
        type=str,
        default="0,270,270",
        help="Per-view azimuth offsets in degrees (comma-separated). 1 value applies to all views.",
    )
    p.add_argument(
        "--view-elev-offsets",
        type=str,
        default="0,180,0",
        help="Per-view elevation offsets in degrees (comma-separated). 1 value applies to all views.",
    )
    p.add_argument("--no-text", action="store_true", help="Disable title text overlay.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    align = None if args.no_align_file else args.align_json
    notch = None if args.notch_hz < 0 else args.notch_hz
    hp = None if args.highpass_hz < 0 else args.highpass_hz
    run(
        vhdr_path=args.vhdr_path,
        electrodes_tsv=args.electrodes_tsv,
        out_mp4=args.out_mp4,
        align_json=align,
        film_time_start=args.film_time_start,
        duration_sec=args.duration_sec,
        out_fps=args.out_fps,
        notch_hz=notch,
        highpass_hz=hp,
        crop_start=args.crop_start,
        movie_t0_in_recording=args.movie_t0_sec,
        drift=args.drift,
        width=args.width,
        height=args.height,
        brain_views=args.brain_views,
        electrode_size=args.electrode_size,
        brain_alpha=args.brain_alpha,
        surface_decim=args.surface_decim,
        sulc_contrast=args.sulc_contrast,
        show_text=not args.no_text,
        camera_distance_factor=args.camera_distance_factor,
        spin_deg_per_sec=args.spin_deg_per_sec,
        parallel_projection=not args.perspective,
        parallel_scale_factor=args.parallel_scale_factor,
        brain_cmap=args.brain_cmap,
        brain_clim_low=args.brain_clim_low,
        brain_clim_high=args.brain_clim_high,
        brain_solid_color=args.brain_solid_color,
        neutral_color=args.neutral_color,
        activity_metric=args.activity_metric,
        hg_low_hz=args.hg_low_hz,
        hg_high_hz=args.hg_high_hz,
        hg_use_hilbert=not args.hg_no_hilbert,
        threshold_mode=args.threshold_mode,
        threshold_z=args.threshold_z,
        threshold_quantile=args.threshold_quantile,
        threshold_style=args.threshold_style,
        embed_video_path=args.embed_video_path,
        embed_top_frac=args.embed_top_frac,
        embed_gap_px=args.embed_gap_px,
        embed_width_frac=args.embed_width_frac,
        embed_x_align=args.embed_x_align,
        embed_fit=args.embed_fit,
        brain_panel_y_shift_px=args.brain_panel_y_shift_px,
        azim_ccw_deg=args.azim_ccw_deg,
        view_azim_offsets=args.view_azim_offsets,
        view_elev_offsets=args.view_elev_offsets,
        spin_view_mask=args.spin_view_mask,
        ffmpeg_exe=args.ffmpeg_exe,
    )


if __name__ == "__main__":
    main()

