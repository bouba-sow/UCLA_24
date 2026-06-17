#!/usr/bin/env python3
"""
Brain-surface MP4 for single-unit (microwire) spike activity using PyVista/VTK.

Each micro channel is shown as a dot on the fsaverage cortical surface.
Colour is BINARY: black when silent, yellow when a spike occurred within
the past --hold-frames output frames (default 5).
Data are read directly from the pre-computed *_spikedata.npz bundles written
by ucla2bids.py — no raw .mat files are needed.

Speed tips
----------
* --n-workers 8     : render frame chunks in parallel (N PyVista workers)
* --render-scale 0.5: render at half resolution → ffmpeg upscales (4x fewer pixels)
* --out-fps 15      : fewer frames
* On GPU node       : submit job_singleunit_brain.sh (hardware OpenGL is much faster)

Compositing pipeline
--------------------
Brain frames are rendered brain-only to a temp file.
ffmpeg then stacks the movie panel (top) + brain panel (bottom) and muxes audio.
This is far faster than PIL compositing per frame.

Usage example
-------------
python film_brain_viewer_singleunit_mp4.py \\
    --film-time-start 0 --duration-sec 60 \\
    --n-workers 8 --render-scale 0.5 \\
    --embed-video-path data/40m_act_24_S06E01_30fps_subtitled_marked.mp4 \\
    --out-mp4 outputs/brain_singleunit.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from nilearn import datasets, surface


# ---------------------------------------------------------------------------
# Geometry / render helpers (same as macro viewer)
# ---------------------------------------------------------------------------

def _decimate_faces(
    coords: np.ndarray, faces: np.ndarray, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step_i = max(1, int(step))
    if step_i == 1:
        return coords, faces, np.arange(coords.shape[0], dtype=np.int64)
    faces_d = faces[::step_i]
    if faces_d.size == 0:
        return coords, faces, np.arange(coords.shape[0], dtype=np.int64)
    used = np.unique(faces_d.ravel())
    idx_map = np.full(coords.shape[0], -1, dtype=np.int64)
    idx_map[used] = np.arange(used.size, dtype=np.int64)
    return coords[used], idx_map[faces_d].astype(np.int32), used


def _to_pv_faces(faces: np.ndarray) -> np.ndarray:
    return np.c_[np.full(faces.shape[0], 3, dtype=np.int32), faces].ravel()


def _normalize_sulc(vals: np.ndarray, contrast: float) -> np.ndarray:
    lo, hi = np.percentile(vals, [5, 95])
    x = np.clip((vals - lo) / max(1e-9, hi - lo), 0.0, 1.0)
    return np.clip(x ** max(0.2, float(contrast)), 0.0, 1.0)


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
    center: np.ndarray, radius: float, azim_deg: float, elev_deg: float, dist: float
) -> tuple:
    az, el = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return tuple(center + d * dist * radius), tuple(center), (0.0, 0.0, 1.0)


def _electrode_cmap(neutral_color: str) -> LinearSegmentedColormap:
    # Used by macro viewer (diverging); kept for potential reuse.
    colors = ["#1f4e79", "#2c7fb8", neutral_color, "#ffd54a", "#fff200"]
    return LinearSegmentedColormap.from_list("dark_diverging", colors, N=256)


def _spike_binary_cmap() -> LinearSegmentedColormap:
    """
    Strictly binary colormap: black (0) → yellow (1).
    A very narrow transition at 0.5 so any value > 0 snaps to yellow.
    """
    from matplotlib.colors import to_rgb
    colors = [(0.00, "#000000"), (0.49, "#000000"), (0.51, "#ffee00"), (1.00, "#ffee00")]
    positions = [c[0] for c in colors]
    rgb_vals  = [to_rgb(c[1]) for c in colors]
    cmap_data = {
        ch: [(pos, col[i], col[i]) for pos, col in zip(positions, rgb_vals)]
        for i, ch in enumerate(("red", "green", "blue"))
    }
    return LinearSegmentedColormap("spike_binary", cmap_data, N=256)


def _parse_offsets(spec: str | None, n: int) -> list[float]:
    if not spec or spec.strip() == "":
        return [0.0] * n
    vals = [float(p) for p in spec.split(",") if p.strip()]
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"Expected 1 or {n} values, got {len(vals)}.")
    return vals


def _parse_spin_mask(spec: str | None, n: int) -> list[bool]:
    if not spec or spec.strip() == "":
        return ([True, False, True] if n == 3 else [True] * n)
    vals = [p in ("1", "true", "True", "yes") for p in spec.split(",") if p.strip()]
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"Expected 1 or {n} spin-mask values.")
    return vals


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_spike_channels(
    npz_dir: Path,
    electrodes_tsv: Path,
) -> tuple[np.ndarray, list[str], list[Path]]:
    """Return (pts, ch_names, npz_paths) for all micro channels that have an NPZ."""
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
        if key not in npz_map:
            continue
        pts_list.append([row["x"], row["y"], row["z"]])
        names_list.append(str(row["name"]))
        paths_list.append(npz_map[key])

    if not names_list:
        raise ValueError(
            "No micro channels matched between electrodes TSV and NPZ files.\n"
            f"  NPZ dir: {npz_dir}\n"
            f"  Example TSV channels: {list(micro['name'])[:5]}"
        )
    return np.array(pts_list, dtype=np.float64), names_list, paths_list


def _build_binary_activity_matrix(
    npz_paths: list[Path],
    film_time_start: float,
    duration_sec: float,
    out_fps: float,
) -> np.ndarray:
    """
    Build a (n_output_frames, n_channels) binary matrix: 1.0 (yellow) or 0.0 (black).
    An electrode is yellow on exactly the frame(s) whose bin contains a spike.
    """
    n_frames = max(1, int(np.floor(duration_sec * out_fps)))
    d0 = np.load(npz_paths[0], allow_pickle=True)
    bin_hz = float(d0["firing_rate_hz"])
    bin_edges = d0["firing_rate_bin_edges"]
    n_bins = len(bin_edges) - 1
    movie_duration = float(d0["movie_duration_sec"])

    binary_matrix = np.zeros((n_bins, len(npz_paths)), dtype=np.float64)
    for j, p in enumerate(npz_paths):
        d = np.load(p, allow_pickle=True)
        spike_times = d["spike_times_movie"].astype(np.float64)
        cluster_ids = d["cluster_id"]
        valid = spike_times[cluster_ids > 0]
        counts, _ = np.histogram(valid, bins=n_bins, range=(0.0, movie_duration))
        binary_matrix[:, j] = (counts > 0).astype(np.float64)

    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    t_end = min(film_time_start + duration_sec, movie_duration)
    frame_times = np.clip(
        film_time_start + (np.arange(n_frames) + 0.5) / out_fps, 0.0, t_end
    )
    bin_idx = np.clip(
        np.searchsorted(bin_centres, frame_times, side="right") - 1, 0, n_bins - 1
    )
    return binary_matrix[bin_idx, :]   # (n_frames, n_ch), values in {0, 1}


# ---------------------------------------------------------------------------
# Per-chunk render worker  (called in a subprocess via multiprocessing)
# ---------------------------------------------------------------------------

def _render_chunk_worker(kwargs: dict) -> str:
    """
    Render a contiguous slice of frames to kwargs['temp_path'].
    All PyVista state is created fresh in this subprocess — safe to spawn.
    Returns the path string of the written temp file.
    """
    # Import inside worker so parent process doesn't need all deps at import time.
    import os
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import warnings
    import numpy as np
    import pyvista as pv
    import imageio.v2 as iio
    from pathlib import Path
    from matplotlib.colors import LinearSegmentedColormap
    from nilearn import datasets, surface

    pv.OFF_SCREEN = True
    try:
        pv.start_xvfb()
    except Exception:
        pass

    # Unpack kwargs
    z_chunk         = kwargs["z_chunk"]           # (n_frames_chunk, n_ch)
    frame_offset    = kwargs["frame_offset"]
    pts             = kwargs["pts"]
    temp_path       = Path(kwargs["temp_path"])
    out_fps         = float(kwargs["out_fps"])
    render_w        = int(kwargs["render_w"])
    render_h        = int(kwargs["render_h"])
    brain_views     = int(kwargs["brain_views"])
    view_angles     = kwargs["view_angles"]       # list of (az, el)
    spin_mask       = kwargs["spin_mask"]
    spin_deg_per_sec = float(kwargs["spin_deg_per_sec"])
    camera_dist     = float(kwargs["camera_dist"])
    clim            = kwargs["clim"]
    threshold_mode  = kwargs["threshold_mode"]
    thr_val         = float(kwargs["thr_val"])
    surface_decim   = int(kwargs["surface_decim"])
    sulc_contrast   = float(kwargs["sulc_contrast"])
    brain_alpha     = float(kwargs["brain_alpha"])
    brain_cmap      = kwargs["brain_cmap"]
    brain_clim_low  = float(kwargs["brain_clim_low"])
    brain_clim_high = float(kwargs["brain_clim_high"])
    brain_solid_color = kwargs["brain_solid_color"]
    neutral_color   = kwargs["neutral_color"]
    electrode_size  = float(kwargs["electrode_size"])
    parallel_proj   = bool(kwargs["parallel_proj"])
    parallel_scale  = float(kwargs["parallel_scale"])
    brain_h_frac    = float(kwargs["brain_h_frac"])
    show_text       = bool(kwargs["show_text"])
    show_gray_bg    = bool(kwargs["show_gray_bg"])

    # -- Build brain surface (nilearn caches after first download) ----------------
    fsaverage = datasets.fetch_surf_fsaverage()
    lh_coords, lh_faces = surface.load_surf_data(fsaverage.pial_left)
    rh_coords, rh_faces = surface.load_surf_data(fsaverage.pial_right)
    lh_sulc = surface.load_surf_data(fsaverage.sulc_left).astype(np.float64)
    rh_sulc = surface.load_surf_data(fsaverage.sulc_right).astype(np.float64)

    def _norm_sulc(v, c):
        lo, hi = np.percentile(v, [5, 95])
        x = np.clip((v - lo) / max(1e-9, hi - lo), 0.0, 1.0)
        return np.clip(x ** max(0.2, float(c)), 0.0, 1.0)

    def _faces_pv(f):
        return np.c_[np.full(f.shape[0], 3, dtype=np.int32), f].ravel()

    def _decim(coords, faces, step):
        step = max(1, int(step))
        if step == 1:
            return coords, faces, np.arange(coords.shape[0], dtype=np.int64)
        fd = faces[::step]
        used = np.unique(fd.ravel())
        im = np.full(coords.shape[0], -1, dtype=np.int64)
        im[used] = np.arange(used.size)
        return coords[used], im[fd].astype(np.int32), used

    lh_c, lh_f, lh_u = _decim(lh_coords.astype(np.float64), lh_faces.astype(np.int32), surface_decim)
    rh_c, rh_f, rh_u = _decim(rh_coords.astype(np.float64), rh_faces.astype(np.int32), surface_decim)
    lh_poly = pv.PolyData(lh_c, _faces_pv(lh_f))
    rh_poly = pv.PolyData(rh_c, _faces_pv(rh_f))
    lh_poly.point_data["sulc"] = _norm_sulc(lh_sulc[lh_u], sulc_contrast)
    rh_poly.point_data["sulc"] = _norm_sulc(rh_sulc[rh_u], sulc_contrast)
    brain_poly = lh_poly.merge(rh_poly)

    b = brain_poly.bounds
    center = np.array([(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2])
    span_max = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
    radius = span_max * 0.55

    shape = (1, 1) if brain_views == 1 else (1, 3)
    plotter = pv.Plotter(off_screen=True, shape=shape, window_size=(render_w, render_h))

    from matplotlib.colors import to_rgb
    _pos = [0.00, 0.49, 0.51, 1.00]
    _hex = ["#000000", "#000000", "#ffee00", "#ffee00"]
    _rgb = [to_rgb(h) for h in _hex]
    _cd  = {ch: [(_pos[k], _rgb[k][i], _rgb[k][i]) for k in range(len(_pos))]
             for i, ch in enumerate(("red", "green", "blue"))}
    elec_cmap = LinearSegmentedColormap("spike_binary", _cd, N=256)
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
            color=brain_solid_color if brain_solid_color else None,
            cmap=brain_cmap if not brain_solid_color else None,
            clim=[brain_clim_low, brain_clim_high] if not brain_solid_color else None,
            opacity=brain_alpha, smooth_shading=True,
            specular=0.05, specular_power=12.0, ambient=0.20, diffuse=0.45,
            show_scalar_bar=False,
        )
        bg = pv.PolyData(pts.copy())
        bg.point_data["c"] = np.ones(pts.shape[0])
        bg_a = plotter.add_mesh(
            bg, color=neutral_color, render_points_as_spheres=True,
            point_size=electrode_size, ambient=0.35, diffuse=0.90,
            specular=0.10, show_scalar_bar=False,
        )
        bg_a.SetVisibility(show_gray_bg)

        em = pv.PolyData(pts.copy())
        em.point_data["spike"] = z_chunk[0].astype(np.float64)
        plotter.add_mesh(
            em, scalars="spike", cmap=elec_cmap, clim=clim,
            render_points_as_spheres=True, point_size=electrode_size,
            ambient=0.35, diffuse=0.90, specular=0.15, nan_color="#000000",
            show_scalar_bar=False,
        )
        plotter.camera_position = _cam(az, el)
        plotter.camera.parallel_projection = parallel_proj
        if parallel_proj:
            pscale = span_max * parallel_scale * brain_h_frac
            plotter.camera.parallel_scale = pscale
        plotter.enable_anti_aliasing("msaa")
        plotter.add_light(pv.Light(position=(300, 200, 300), focal_point=tuple(center), intensity=0.9))
        if show_text and i == 0:
            plotter.add_text(
                "Single-unit spikes",
                position="upper_left", color="white", font_size=11,
            )
        elec_meshes.append(em)

    writer = iio.get_writer(
        str(temp_path), fps=out_fps, codec="libx264",
        pixelformat="yuv420p", ffmpeg_log_level="error",
        macro_block_size=1,
    )
    for k in range(len(z_chunk)):
        abs_k = frame_offset + k
        t_sec = abs_k / out_fps
        for i in range(len(view_angles)):
            plotter.subplot(0, i)
            fv = z_chunk[k].copy()
            elec_meshes[i]["spike"] = fv
            elec_meshes[i].set_active_scalars("spike")
            elec_meshes[i].Modified()
            if spin_deg_per_sec != 0.0 and spin_mask[i]:
                az0, el0 = view_angles[i]
                plotter.camera_position = _cam(az0 + spin_deg_per_sec * t_sec, el0)
        plotter.render()
        writer.append_data(plotter.screenshot(return_img=True))
    writer.close()
    plotter.close()
    return str(temp_path)


# ---------------------------------------------------------------------------
# ffmpeg compositing helpers
# ---------------------------------------------------------------------------

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
    Stack movie (top) + brain animation (bottom) with ffmpeg and mux audio.
    Supports width fraction, alignment, and brain panel vertical shift.
    If no embed video, just copy brain_video → out_mp4.
    """
    if embed_video_path is None or not embed_video_path.is_file():
        shutil.copy2(brain_video, out_mp4)
        return

    # Movie panel: scale to panel_w × top_h, then pad to full width at x_off
    panel_w = max(1, int(round(width * float(np.clip(embed_width_frac, 0.1, 1.0)))))
    align_k = embed_x_align.strip().lower()
    if align_k == "left":
        x_off = 0
    elif align_k == "right":
        x_off = width - panel_w
    else:
        x_off = (width - panel_w) // 2

    # Brain panel: add top padding for y-shift, then scale to full width
    shift = max(0, int(brain_panel_y_shift_px))
    brain_h_padded = brain_h + shift

    top_filter = (
        f"[0:v]trim=start={film_time_start}:duration={duration_sec},"
        f"setpts=PTS-STARTPTS,"
        f"scale={panel_w}:{top_h}:force_original_aspect_ratio=decrease,"
        f"pad={panel_w}:{top_h}:(ow-iw)/2:(oh-ih)/2,"
        f"pad={width}:{top_h}:{x_off}:0:black,"
        f"setsar=1[top]"
    )
    bot_filter = (
        f"[1:v]scale={width}:{brain_h}:flags=lanczos,"
        f"pad={width}:{brain_h_padded}:0:{shift}:black,"
        f"setsar=1[bot]"
    )

    if gap_h > 0:
        stack_filter = (
            f"color=black:{width}x{gap_h}:r=25[gap];"
            f"[top][gap][bot]vstack=inputs=3,setsar=1[out]"
        )
    else:
        stack_filter = f"[top][bot]vstack=inputs=2,setsar=1[out]"

    audio_filter = (
        f"[0:a]atrim=start={film_time_start}:duration={duration_sec},"
        f"asetpts=PTS-STARTPTS[aud]"
    )
    filter_complex = f"{top_filter};{bot_filter};{stack_filter};{audio_filter}"

    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(embed_video_path),
        "-i", str(brain_video),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "[aud]",
        "-t", str(duration_sec),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-shortest",
        str(out_mp4),
    ]
    print("Running ffmpeg composite …")
    subprocess.run(cmd, check=True)


def _ffmpeg_concat(temp_paths: list[Path], out_path: Path, ffmpeg_bin: str) -> None:
    """Concatenate temp mp4 chunks into one file using ffmpeg concat demuxer."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_file = f.name
        for p in temp_paths:
            f.write(f"file '{p.resolve()}'\n")
    try:
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        os.unlink(list_file)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run(
    npz_dir: Path,
    electrodes_tsv: Path,
    out_mp4: Path,
    film_time_start: float,
    duration_sec: float,
    out_fps: float,
    n_workers: int,
    render_scale: float,
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
    threshold_mode: str,
    threshold_z: float,
    threshold_quantile: float,
    threshold_style: str,
    embed_video_path: Path | None,
    embed_top_frac: float,
    embed_gap_px: int,
    embed_width_frac: float,
    embed_x_align: str,
    brain_panel_y_shift_px: int,
    azim_ccw_deg: float,
    view_azim_offsets: str | None,
    view_elev_offsets: str | None,
    spin_view_mask: str | None,
    ffmpeg_exe: str | None,
    show_electrode_positions: bool = True,
) -> None:
    # ------------------------------------------------------------------ data --
    pts, ch_names, npz_paths = _load_spike_channels(npz_dir, electrodes_tsv)
    print(f"Loaded {len(ch_names)} micro channels with coordinates and NPZ files.")

    print(f"Building binary spike activity matrix ({out_fps} fps) …")
    z = _build_binary_activity_matrix(npz_paths, film_time_start, duration_sec, out_fps)
    n_frames = z.shape[0]
    clim = (0.0, 1.0)
    print(f"  {n_frames} output frames,  {len(npz_paths)} channels")

    # ---------------------------------------------------------- threshold ----
    threshold_mode = "none"   # binary plot: no threshold needed
    thr_val = 0.0
    threshold_style = "dim"
    show_gray_bg = show_electrode_positions

    # --------------------------------------------------------- layout  -------
    brain_views = int(brain_views)
    if brain_views not in (1, 3):
        raise ValueError("--brain-views must be 1 or 3.")

    use_embed = embed_video_path is not None and Path(embed_video_path).is_file()
    if embed_video_path is not None and not use_embed:
        print(f"Warning: embed video not found: {embed_video_path}")

    top_frac = float(np.clip(embed_top_frac, 0.0, 0.8)) if use_embed else 0.0
    top_h = int(round(height * top_frac))
    gap_h = max(0, int(embed_gap_px)) if use_embed else 0
    brain_h = max(200, height - top_h - gap_h)

    render_scale = float(np.clip(render_scale, 0.1, 1.0))
    render_w = max(64, int(round(width * render_scale)))
    render_h = max(64, int(round(brain_h * render_scale)))
    # make dimensions even (required by some codecs)
    render_w += render_w % 2
    render_h += render_h % 2
    if render_scale < 1.0:
        print(f"  Rendering at {render_w}×{render_h} (scale={render_scale}); ffmpeg will upscale.")

    view_angles_base = [(78.0, 14.0)] if brain_views == 1 else [(90.0, 0.0), (0.0, 89.0), (0.0, 0.0)]
    az_off = _parse_offsets(view_azim_offsets, len(view_angles_base))
    el_off = _parse_offsets(view_elev_offsets, len(view_angles_base))
    view_angles = [
        (az + azim_ccw_deg + az_off[i], el + el_off[i])
        for i, (az, el) in enumerate(view_angles_base)
    ]
    spin_mask = _parse_spin_mask(spin_view_mask, len(view_angles))

    ffmpeg_bin = ffmpeg_exe or shutil.which("ffmpeg") or "ffmpeg"

    # ---------------------------------------- build shared worker kwargs dict --
    base_kwargs = dict(
        pts=pts,
        out_fps=out_fps,
        render_w=render_w,
        render_h=render_h,
        brain_views=brain_views,
        view_angles=view_angles,
        spin_mask=spin_mask,
        spin_deg_per_sec=spin_deg_per_sec,
        camera_dist=camera_distance_factor,
        clim=clim,
        threshold_mode=threshold_mode,
        thr_val=thr_val,
        surface_decim=surface_decim,
        sulc_contrast=sulc_contrast,
        brain_alpha=brain_alpha,
        brain_cmap=brain_cmap,
        brain_clim_low=brain_clim_low,
        brain_clim_high=brain_clim_high,
        brain_solid_color=brain_solid_color,
        neutral_color=neutral_color,
        electrode_size=electrode_size,
        parallel_proj=parallel_projection,
        parallel_scale=parallel_scale_factor,
        brain_h_frac=brain_h / height,
        show_text=show_text,
        show_gray_bg=show_gray_bg,
    )

    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------- render (single / multi) -
    n_workers = max(1, int(n_workers))
    with tempfile.TemporaryDirectory(prefix="singleunit_brain_") as tmpdir:
        tmp = Path(tmpdir)

        if n_workers == 1:
            brain_silent = tmp / "brain_silent.mp4"
            chunk_kwargs = {**base_kwargs, "z_chunk": z, "frame_offset": 0,
                            "temp_path": str(brain_silent)}
            print("Rendering frames (1 worker) …")
            _render_chunk_worker(chunk_kwargs)

        else:
            # Split z into n_workers even chunks
            chunk_sizes = [len(r) for r in np.array_split(np.arange(n_frames), n_workers)]
            offsets = [0] + list(np.cumsum(chunk_sizes[:-1]))
            chunks = np.array_split(z, n_workers, axis=0)

            temp_chunks = [tmp / f"chunk_{i:03d}.mp4" for i in range(n_workers)]
            worker_args = [
                {**base_kwargs,
                 "z_chunk": chunks[i],
                 "frame_offset": int(offsets[i]),
                 "temp_path": str(temp_chunks[i])}
                for i in range(len(chunks))
            ]
            print(f"Rendering {n_frames} frames across {n_workers} workers …")

            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(n_workers) as pool:
                pool.map(_render_chunk_worker, worker_args)

            brain_silent = tmp / "brain_silent.mp4"
            _ffmpeg_concat(temp_chunks, brain_silent, ffmpeg_bin)
            print("Concatenated chunk videos.")

        # ------------------------------------------------ composite + audio ----
        print("Compositing brain + movie + audio via ffmpeg …")
        _ffmpeg_composite(
            brain_video=brain_silent,
            embed_video_path=embed_video_path if use_embed else None,
            film_time_start=film_time_start,
            duration_sec=duration_sec,
            out_mp4=out_mp4,
            width=width,
            height=height,
            top_h=top_h,
            gap_h=gap_h,
            brain_h=brain_h,
            embed_width_frac=embed_width_frac,
            embed_x_align=embed_x_align,
            brain_panel_y_shift_px=brain_panel_y_shift_px,
            ffmpeg_bin=ffmpeg_bin,
        )

    print(f"Done → {out_mp4}  ({n_frames} frames @ {out_fps} fps)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PyVista brain-surface MP4 for single-unit microwire spike activity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _DATA = Path("/store/scratch/bsow/Documents/UCLA_24")
    p.add_argument("--npz-dir", type=Path,
        default=_DATA / "data/bids/derivatives/spike-sorted/sub-572/ses-01/ieeg",
        help="Directory containing *_spikedata.npz files.")
    p.add_argument("--electrodes-tsv", type=Path,
        default=_DATA / "data/bids/sub-572/ses-01/ieeg/sub-572_ses-01_electrodes.tsv")
    p.add_argument("--out-mp4", type=Path,
        default=_DATA / "outputs/brain_singleunit.mp4")
    p.add_argument("--film-time-start", type=float, default=70.0,
        help="Movie start time in seconds.")
    p.add_argument("--duration-sec", type=float, default=40.0)
    p.add_argument("--out-fps", type=float, default=20.0)

    p.add_argument("--n-workers", type=int, default=30,
        help="Parallel render workers.  Each spawns an independent PyVista process. "
             "Try --n-workers 8 on CPU nodes, 1 on GPU nodes.")
    p.add_argument("--render-scale", type=float, default=1.0,
        help="Render resolution multiplier (0.5 = half size → ~4x faster, ffmpeg upscales).")

    # Embed video
    p.add_argument("--embed-video-path", type=Path,
        default=_DATA / "data/40m_act_24_S06E01_30fps.m4v",
        help="Movie file to embed as top panel (with audio).")
    p.add_argument("--embed-top-frac", type=float, default=0.45,
        help="Fraction of output height reserved for embedded movie (0–0.8).")
    p.add_argument("--embed-gap-px", type=int, default=0,
        help="Vertical gap (px) between movie panel and brain panel.")
    p.add_argument("--embed-width-frac", type=float, default=0.4,
        help="Width fraction of the top movie panel (0.1–1.0).")
    p.add_argument("--embed-x-align", type=str, default="center",
        choices=["left", "center", "right"],
        help="Horizontal alignment of the movie panel.")
    p.add_argument("--embed-fit", type=str, default="contain",
        choices=["stretch", "contain", "cover"],
        help="How to fit the movie inside the top panel (informational; ffmpeg uses contain).")
    p.add_argument("--brain-panel-y-shift-px", type=int, default=14,
        help="Vertical offset (px) added above brain panel (positive = moves down).")

    # Layout / output
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--brain-views", type=int, default=3, help="1 or 3.")
    p.add_argument("--ffmpeg-exe", type=str, default=None)

    # Brain surface
    p.add_argument("--electrode-size", type=float, default=16.0)
    p.add_argument("--brain-alpha", type=float, default=0.4)
    p.add_argument("--surface-decim", type=int, default=1,
        help="1 = full mesh detail.  Higher values decimate (faster but lower quality).")
    p.add_argument("--sulc-contrast", type=float, default=1.0)
    p.add_argument("--brain-cmap", type=str, default="gray")
    p.add_argument("--brain-clim-low", type=float, default=0.05)
    p.add_argument("--brain-clim-high", type=float, default=0.95)
    p.add_argument("--neutral-color", type=str, default="#8a8a8a")
    p.add_argument("--brain-solid-color", type=str, default="#6f6f6f",
        help="Uniform cortex colour; disables sulcal shading.")

    # Camera / animation
    p.add_argument("--camera-distance-factor", type=float, default=5.8)
    p.add_argument("--parallel-scale-factor", type=float, default=1.0)
    p.add_argument("--perspective", action="store_true",
        help="Use perspective projection (default: parallel).")
    p.add_argument("--spin-deg-per-sec", type=float, default=3.0,
        help="Camera rotation speed in deg/s (0 = fixed).")
    p.add_argument("--spin-view-mask", type=str, default=None,
        help="Per-view spin enable (comma-separated 0/1). Default for 3 views: 1,0,1.")
    p.add_argument("--azim-ccw-deg", type=float, default=0.0,
        help="Global counter-clockwise azimuth offset for all views.")
    p.add_argument("--view-azim-offsets", type=str, default="0,270,270",
        help="Per-view azimuth offsets (comma-separated).")
    p.add_argument("--view-elev-offsets", type=str, default="0,180,0",
        help="Per-view elevation offsets (comma-separated).")

    p.add_argument("--no-text", action="store_true")
    p.add_argument("--hide-electrode-positions", action="store_true",
        help="Hide silent electrode dots (only yellow spikes shown). Default: always show positions.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        npz_dir=args.npz_dir,
        electrodes_tsv=args.electrodes_tsv,
        out_mp4=args.out_mp4,
        film_time_start=args.film_time_start,
        duration_sec=args.duration_sec,
        out_fps=args.out_fps,
        n_workers=args.n_workers,
        render_scale=args.render_scale,
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
        threshold_mode="none",
        threshold_z=0.0,
        threshold_quantile=0.0,
        threshold_style="dim",
        embed_video_path=args.embed_video_path,
        embed_top_frac=args.embed_top_frac,
        embed_gap_px=args.embed_gap_px,
        embed_width_frac=args.embed_width_frac,
        embed_x_align=args.embed_x_align,
        brain_panel_y_shift_px=args.brain_panel_y_shift_px,
        azim_ccw_deg=args.azim_ccw_deg,
        view_azim_offsets=args.view_azim_offsets,
        view_elev_offsets=args.view_elev_offsets,
        spin_view_mask=args.spin_view_mask,
        ffmpeg_exe=args.ffmpeg_exe,
        show_electrode_positions=not args.hide_electrode_positions,
    )


if __name__ == "__main__":
    main()
