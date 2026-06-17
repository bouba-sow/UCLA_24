#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def read_channel_names(experiment_dir: Path) -> tuple[list[str], list[str]]:
    micro_csv = experiment_dir / "CSC_micro" / "outFileNames.csv"
    macro_csv = experiment_dir / "CSC_macro" / "outFileNames.csv"

    micro_names: list[str] = []
    for line in micro_csv.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        stem = Path(line).name.replace("_001.mat", "")
        micro_names.append(stem)

    macro_names: list[str] = []
    for line in macro_csv.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        stem = Path(line).name.replace("_001.mat", "")
        macro_names.append(stem)

    return micro_names, macro_names


def read_acq_channel_files(experiment_dir: Path, acq: str) -> list[tuple[str, Path]]:
    if acq not in {"macro", "micro"}:
        raise ValueError(f"Unsupported acquisition: {acq}")
    acq_dir = experiment_dir / f"CSC_{acq}"
    out_csv = acq_dir / "outFileNames.csv"
    pairs: list[tuple[str, Path]] = []
    for line in out_csv.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        fname = Path(line).name
        local_path = acq_dir / fname
        if not local_path.exists():
            raise FileNotFoundError(f"Missing channel file: {local_path}")
        ch_name = fname.replace("_001.mat", "")
        pairs.append((ch_name, local_path))
    return pairs


def read_mat_channel_info(mat_path: Path) -> tuple[int, float]:
    with h5py.File(mat_path, "r") as f:
        if "data" not in f:
            raise ValueError(f"{mat_path} does not contain 'data'")
        n_samples = int(f["data"].shape[1])
        if "samplingIntervalSeconds" in f:
            dt = float(f["samplingIntervalSeconds"][0, 0])
            sfreq = 1.0 / dt if dt > 0 else np.nan
        else:
            sfreq = np.nan
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"Invalid sampling frequency in {mat_path}")
    return n_samples, float(sfreq)


def write_brainvision_ieeg(
    experiment_dir: Path,
    bids_root: Path,
    subject: str,
    session: str,
    task: str,
    acq: str,
    run: str,
) -> None:
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    channel_files = read_acq_channel_files(experiment_dir, acq)
    if not channel_files:
        raise ValueError(f"No channels found for acquisition {acq}")

    first_samples, sfreq = read_mat_channel_info(channel_files[0][1])
    n_channels = len(channel_files)
    n_samples = first_samples

    with tempfile.TemporaryDirectory(prefix=f"ucla2bids_{acq}_") as tmpdir:
        mmap_path = Path(tmpdir) / f"{acq}_data_float64.mmap"
        data = np.memmap(mmap_path, dtype="float64", mode="w+", shape=(n_channels, n_samples))

        for idx, (ch_name, mat_path) in enumerate(channel_files):
            with h5py.File(mat_path, "r") as f:
                ch_data = f["data"][0, :]
                ad_bit_volts = float(f["ADBitVolts"][0, 0]) if "ADBitVolts" in f else 1.0
                if ch_data.shape[0] != n_samples:
                    raise ValueError(
                        f"Sample length mismatch in {mat_path}: got {ch_data.shape[0]}, expected {n_samples}"
                    )
                data[idx, :] = ch_data.astype(np.float64) * ad_bit_volts

        info = mne.create_info(
            ch_names=[name for name, _ in channel_files],
            sfreq=sfreq,
            ch_types=["seeg"] * n_channels,
        )
        info["line_freq"] = 60.0
        raw = mne.io.RawArray(data, info, copy="auto", verbose="ERROR")

        bids_path = BIDSPath(
            root=str(bids_root),
            subject=subject,
            session=session,
            task=task,
            acquisition=acq,
            run=run,
            datatype="ieeg",
        )
        write_raw_bids(
            raw,
            bids_path=bids_path,
            format="BrainVision",
            overwrite=True,
            allow_preload=True,
            verbose=False,
        )


def load_localization_sheet(localization_xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(localization_xlsx, sheet_name="Sheet1")
    required = {"electrode", "MNI_x", "MNI_y", "MNI_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in Sheet1: {sorted(missing)}")

    df = df.copy()
    df["electrode"] = df["electrode"].astype(str).str.replace("-", "", regex=False)
    df["lookup"] = df["electrode"].map(normalize_name)
    return df


def micro_lookup_key(channel: str) -> str:
    # GA1-RA3 -> RA_micro1: all microwires on a shaft share the tip position
    if "-" in channel:
        channel = channel.split("-", 1)[1]
    shaft = channel.rstrip("0123456789")
    return normalize_name(f"{shaft}_micro1")


def macro_lookup_key(channel: str) -> str:
    return normalize_name(channel)


def build_electrodes_table(
    micro_channels: list[str],
    macro_channels: list[str],
    loc_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    loc_map = {row["lookup"]: row for _, row in loc_df.iterrows()}
    rows = []
    unmatched: list[str] = []

    def append_row(name: str, group: str, key: str) -> None:
        loc_row = loc_map.get(key)
        if loc_row is None:
            unmatched.append(name)
            rows.append(
                {
                    "name": name,
                    "group": group,
                    "x": "n/a",
                    "y": "n/a",
                    "z": "n/a",
                    "size": "n/a",
                    "manufacturer": "n/a",
                    "group_note": "unmatched_localization",
                }
            )
            return

        rows.append(
            {
                "name": name,
                "group": group,
                "x": loc_row["MNI_x"] if pd.notna(loc_row["MNI_x"]) else "n/a",
                "y": loc_row["MNI_y"] if pd.notna(loc_row["MNI_y"]) else "n/a",
                "z": loc_row["MNI_z"] if pd.notna(loc_row["MNI_z"]) else "n/a",
                "size": "n/a",
                "manufacturer": "Neuralynx",
                "group_note": str(loc_row["region"]) if "region" in loc_row and pd.notna(loc_row["region"]) else "n/a",
            }
        )

    for ch in sorted(micro_channels):
        append_row(ch, "micro", micro_lookup_key(ch))
    for ch in sorted(macro_channels):
        append_row(ch, "macro", macro_lookup_key(ch))

    out = pd.DataFrame(rows)
    return out, sorted(set(unmatched))


def write_ieeg_metadata(
    bids_root: Path,
    subject: str,
    session: str,
    electrodes_df: pd.DataFrame,
) -> None:
    ieeg_dir = bids_root / f"sub-{subject}" / f"ses-{session}" / "ieeg"
    ieeg_dir.mkdir(parents=True, exist_ok=True)

    electrodes_tsv = ieeg_dir / f"sub-{subject}_ses-{session}_electrodes.tsv"
    electrodes_json = ieeg_dir / f"sub-{subject}_ses-{session}_electrodes.json"
    coordsystem_json = ieeg_dir / f"sub-{subject}_ses-{session}_coordsystem.json"

    electrodes_df.to_csv(electrodes_tsv, sep="\t", index=False)

    with open(electrodes_json, "w") as f:
        json.dump(
            {
                "x": {"Description": "MNI x coordinate", "Units": "mm"},
                "y": {"Description": "MNI y coordinate", "Units": "mm"},
                "z": {"Description": "MNI z coordinate", "Units": "mm"},
                "group": {"Description": "Acquisition group (macro or micro)"},
                "group_note": {"Description": "Localization region or matching note"},
                "size": {"Description": "Contact size", "Units": "mm^2"},
                "manufacturer": {"Description": "Hardware manufacturer"},
            },
            f,
            indent=2,
        )

    with open(coordsystem_json, "w") as f:
        json.dump(
            {
                "iEEGCoordinateSystem": "MNI152NLin6ASym",
                "iEEGCoordinateUnits": "mm",
                "iEEGCoordinateProcessingDescription": "Coordinates loaded from sub-572_localizations.xlsx (Sheet1). Unmatched electrodes set to n/a.",
            },
            f,
            indent=2,
        )


def _first_numeric(value: object, default: float) -> float:
    if isinstance(value, list):
        if not value:
            return default
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_movie_alignment_seconds(audio_align_json: Path) -> tuple[float, float]:
    payload = json.loads(audio_align_json.read_text())
    movie_start_rel = _first_numeric(payload.get("start_rel_rec"), 0.0)
    drift_multiplier = _first_numeric(payload.get("drift_correction_multiplier"), 1.0)
    return movie_start_rel, drift_multiplier


def slugify_label(value: object) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).lower())).strip("_")


VIDEO_NOMINAL_FPS = 29.97
CONCEPT_ANNOTATION_HZ = 1.0
CONCEPT_TRAINING_BIN_HZ = 4.0

# Paper eight-concept schema (movie on-screen annotations only; source CSV column names).
PAPER_CONCEPT_SOURCE_MAP: dict[str, list[str]] = {
    "white_house": ["WhiteHouse"],
    "cia": ["CTU"],
    "sacrifice": ["Hostage"],
    "handcuff": ["Handcuff"],
    "j_bauer": ["J.Bauer"],
    "b_buchanan": ["B.Buchanan"],
    "a_fayed": ["A.Fayed"],
    "a_amar": ["A.Amar"],
}

# Frame-aligned audio / phonological regressors (must match enriched feature CSV).
AUDIO_PHONOLOGICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "env",
    "env_peak_rate",
    "pitch_hz",
    "pitch_norm",
    "pitch_up",
    "pitch_down",
    "pause_duration_ms",
    "word_char_len",
    *(f"mel_{i:02d}" for i in range(16)),
)


def frame_duration_seconds(stimulus_times: np.ndarray) -> float:
    diffs = np.diff(np.asarray(stimulus_times, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0 / VIDEO_NOMINAL_FPS
    return float(np.median(diffs))


def align_paper_concepts(concept_df: pd.DataFrame, stimulus_times: np.ndarray) -> dict[str, np.ndarray]:
    """Expand 1 Hz on-screen concept CSV to frame rows using the paper concept schema."""
    sec_idx = np.floor(stimulus_times).astype(int)
    n_sec = len(concept_df)
    valid = (sec_idx >= 0) & (sec_idx < n_sec)
    out: dict[str, np.ndarray] = {}
    for paper_name, source_cols in PAPER_CONCEPT_SOURCE_MAP.items():
        merged_sec = np.zeros(n_sec, dtype=float)
        for col in source_cols:
            if col not in concept_df.columns:
                raise ValueError(
                    f"Concept CSV missing column {col!r} required for concept_{paper_name}"
                )
            merged_sec = np.maximum(
                merged_sec,
                pd.to_numeric(concept_df[col], errors="coerce").fillna(0.0).to_numpy(),
            )
        vals = np.zeros(len(stimulus_times), dtype=float)
        vals[valid] = merged_sec[sec_idx[valid]]
        out[f"concept_{paper_name}"] = vals
    return out


def build_events_table(
    feature_csv: Path,
    audio_align_json: Path,
    phoneme_csv: Path,
    characters_csv: Path,
    concepts_csv: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    feat = pd.read_csv(feature_csv)
    phon = pd.read_csv(phoneme_csv)
    movie_start_rel, drift_multiplier = get_movie_alignment_seconds(audio_align_json)

    for col in ("vowel_onset", "word_onset"):
        if col not in feat.columns:
            raise ValueError(f"Missing required column in features CSV: {col}")
        feat[col] = feat[col].fillna("")

    if "word_frequency" not in feat.columns:
        feat["word_frequency"] = 0.0

    if "vowel_duration" not in feat.columns or "word_duration" not in feat.columns:
        raise ValueError("Features CSV must include vowel_duration and word_duration.")

    missing_audio = [c for c in AUDIO_PHONOLOGICAL_FEATURE_COLUMNS if c not in feat.columns]
    if missing_audio:
        raise ValueError(
            "Features CSV is missing audio-phonological columns "
            f"(expected enriched 24_S06E01_events_vowel_word_features): {missing_audio}"
        )

    # Keep one events row per frame (full frame-wise table).
    ev = feat.copy()
    if len(phon) != len(ev):
        raise ValueError(
            f"Phoneme CSV row count ({len(phon)}) does not match feature CSV row count ({len(ev)})."
        )

    stimulus_times = ev["time"].astype(float).to_numpy()
    frame_dt = frame_duration_seconds(stimulus_times)
    ev["stimulus_time"] = stimulus_times
    ev["onset"] = movie_start_rel + stimulus_times * drift_multiplier
    ev["duration"] = frame_dt

    for col in AUDIO_PHONOLOGICAL_FEATURE_COLUMNS:
        ev[col] = pd.to_numeric(ev[col], errors="coerce").fillna(0.0)

    # Frame-wise character one-hot columns (30 fps labels).
    char_df = pd.read_csv(characters_csv)
    if "Frame" not in char_df.columns:
        raise ValueError(f"Character CSV must include 'Frame' column: {characters_csv}")
    char_cols = [c for c in char_df.columns if c != "Frame"]
    char_aligned = (
        char_df.set_index("Frame")[char_cols]
        .reindex(ev["frame"].astype(int).to_numpy(), fill_value=0)
        .reset_index(drop=True)
    )
    char_name_map = {c: f"char_{slugify_label(c)}" for c in char_cols}
    for src, dst in char_name_map.items():
        ev[dst] = pd.to_numeric(char_aligned[src], errors="coerce").fillna(0.0)

    concept_df = pd.read_csv(concepts_csv)
    concept_cols = align_paper_concepts(concept_df, stimulus_times)
    for dst, vals in concept_cols.items():
        ev[dst] = vals

    keep_cols = [
        "onset",
        "duration",
        "stimulus_time",
        "frame",
        "vowel_onset",
        "word_onset",
        "vowel_duration",
        "word_duration",
        "word_frequency",
        *AUDIO_PHONOLOGICAL_FEATURE_COLUMNS,
        *char_name_map.values(),
        *concept_cols.keys(),
    ]
    ev = ev[keep_cols].sort_values("onset").reset_index(drop=True)

    timing_meta = {
        "StimulusTiming": {
            "Description": "Alignment between movie stimulus clock and iEEG recording clock.",
            "Formula": "onset = start_rel_rec + stimulus_time * drift_correction_multiplier",
            "start_rel_rec": movie_start_rel,
            "drift_correction_multiplier": drift_multiplier,
            "stimulus_time_reference": (
                "Seconds from audio-aligned movie start (pre-drift video clock; matches feature CSV time)."
            ),
            "video_nominal_fps": VIDEO_NOMINAL_FPS,
            "frame_duration_sec": frame_dt,
        },
        "ConceptLabels": {
            "Description": (
                "Eight paper decoder-target concepts from on-screen movie annotations only "
                "(no free-recall linguistic mentions)."
            ),
            "Source": "movie_viewing_on_screen",
            "AnnotationRateHz": CONCEPT_ANNOTATION_HZ,
            "TrainingBinHz": CONCEPT_TRAINING_BIN_HZ,
            "FrameAssignment": (
                "Each 1 s annotation applies to all frames with floor(stimulus_time) equal to that "
                "second index (equivalent to four identical 250 ms training bins per second)."
            ),
            "SourceColumnMerges": {
                "concept_cia": "CTU",
                "concept_sacrifice": "Hostage",
            },
        },
    }
    return ev, timing_meta


EVENT_COLUMN_DESCRIPTIONS: dict[str, dict[str, object]] = {
    "onset": {
        "Description": (
            "Frame onset in seconds relative to the start of the iEEG recording "
            "(drift-corrected movie clock)."
        ),
    },
    "duration": {
        "Description": "Frame duration in seconds (median inter-frame stimulus_time step).",
    },
    "stimulus_time": {
        "Description": (
            "Time in seconds from audio-aligned movie start in the pre-drift stimulus clock "
            "(feature CSV time axis)."
        ),
    },
    "frame": {"Description": "Frame index in the movie-derived linguistic feature table."},
    "vowel_onset": {"Description": "IPA vowel token at onset row, else empty."},
    "word_onset": {"Description": "Word token at onset row, else empty."},
    "vowel_duration": {"Description": "Duration in seconds of vowel segment, written only at onset rows."},
    "word_duration": {"Description": "Duration in seconds of word segment, written only at onset rows."},
    "word_frequency": {"Description": "Zipf frequency from wordfreq package (language='en') at word onset rows."},
    "env": {"Description": "Short-time loudness / envelope feature aligned to the movie audio track (frame-wise)."},
    "env_peak_rate": {"Description": "Rate of envelope peaks in the analysis window (frame-wise)."},
    "pitch_hz": {"Description": "F0 estimate in Hz from the movie audio (frame-wise)."},
    "pitch_norm": {"Description": "Speaker-normalized pitch (min-max on voiced frames) in arbitrary units (frame-wise)."},
    "pitch_up": {"Description": "Magnitude of upward pitch movement in the frame (frame-wise)."},
    "pitch_down": {"Description": "Magnitude of downward pitch movement in the frame (frame-wise)."},
    "pause_duration_ms": {
        "Description": "Duration of preceding silence / pause in milliseconds when defined at word onsets; else 0.",
        "Units": "ms",
    },
    "word_char_len": {"Description": "Character length of the word token at word-onset rows; 0 otherwise."},
    **{
        f"mel_{i:02d}": {
            "Description": f"Log-mel spectrum bin {i} (normalized), frame-aligned to movie audio.",
        }
        for i in range(16)
    },
    **{
        f"concept_{name}": {
            "Description": (
                f"Paper concept '{name.replace('_', ' ')}' presence (1/0) from on-screen movie "
                f"annotations at {CONCEPT_ANNOTATION_HZ:g} Hz."
            ),
        }
        for name in PAPER_CONCEPT_SOURCE_MAP
    },
}


def write_events_files(
    bids_root: Path,
    subject: str,
    session: str,
    task: str,
    acqs: list[str],
    run: str,
    events_df: pd.DataFrame,
    timing_meta: dict[str, object] | None = None,
) -> None:
    ieeg_dir = bids_root / f"sub-{subject}" / f"ses-{session}" / "ieeg"
    ieeg_dir.mkdir(parents=True, exist_ok=True)

    for acq in acqs:
        prefix = f"sub-{subject}_ses-{session}_task-{task}_acq-{acq}_run-{run}_events"
        tsv_path = ieeg_dir / f"{prefix}.tsv"
        json_path = ieeg_dir / f"{prefix}.json"

        events_df.to_csv(tsv_path, sep="\t", index=False, na_rep="n/a")

        metadata: dict[str, object] = {}
        if timing_meta:
            metadata.update(timing_meta)
        for col in events_df.columns:
            metadata[col] = EVENT_COLUMN_DESCRIPTIONS.get(
                col,
                {"Description": f"{col} (task-specific event column)."},
            )
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)


SPIKE_PIPELINE_NAME = "spike-sorted"
UNIT_CLASS_LABELS = {1: "single_unit", 2: "multiunit", 3: "noise"}


def channel_to_desc(channel: str) -> str:
    desc = re.sub(r"[^a-zA-Z0-9]", "", channel)
    if not desc:
        raise ValueError(f"Cannot build BIDS desc label from channel name: {channel}")
    return desc


def find_continuous_micro_mat(experiment_dir: Path, channel: str) -> Path | None:
    micro_dir = experiment_dir / "CSC_micro"
    if not micro_dir.is_dir():
        return None
    for suffix in ("_001.mat", "_002.mat"):
        candidate = micro_dir / f"{channel}{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(micro_dir.glob(f"{channel}_*.mat"))
    return matches[0] if matches else None


def load_spikes_pipeline_mat(
    mat_path: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load wave_clus pipeline GA1-RA1_spikes.mat (detection waveforms + times)."""
    with h5py.File(mat_path, "r") as f:
        if "spikeTimestamps" not in f or "spikes" not in f:
            raise ValueError(f"{mat_path} missing spikeTimestamps and/or spikes")
        timestamps = np.asarray(f["spikeTimestamps"][()], dtype=np.float64).ravel()
        waveforms = np.asarray(f["spikes"][()], dtype=np.float32)
        if "param/sr" in f:
            sr = float(f["param/sr"][0, 0])
        else:
            sr = np.nan
    if waveforms.ndim != 2:
        raise ValueError(f"{mat_path}: spikes must be 2-D, got {waveforms.shape}")
    if not np.isfinite(sr) or sr <= 0:
        raise ValueError(f"{mat_path}: invalid param/sr")
    return timestamps, waveforms, sr


def match_detection_indices(
    series_times: np.ndarray,
    detection_times: np.ndarray,
    atol: float = 1e-4,
) -> np.ndarray:
    """Map each series spike time to an index in detection_times (pipeline spikes.mat)."""
    order = np.argsort(detection_times)
    sorted_times = detection_times[order]
    positions = np.searchsorted(sorted_times, series_times)
    positions = np.clip(positions, 0, len(detection_times) - 1)
    left = np.maximum(positions - 1, 0)
    choose_right = np.abs(sorted_times[positions] - series_times) < np.abs(
        sorted_times[left] - series_times
    )
    idx_sorted = np.where(choose_right, positions, left)
    indices = order[idx_sorted]
    if np.any(np.abs(detection_times[indices] - series_times) > atol):
        bad = np.abs(detection_times[indices] - series_times) > atol
        raise ValueError(
            "Could not match manual spike times to pipeline spikeTimestamps; "
            f"{bad.sum()} spikes exceed atol={atol}"
        )
    return indices.astype(int)


def channel_from_times_path(mat_path: Path) -> str:
    stem = mat_path.stem
    if stem.startswith("times_manual_"):
        return stem.replace("times_manual_", "", 1)
    if stem.startswith("times_"):
        return stem.replace("times_", "", 1)
    raise ValueError(f"Unrecognized times file name: {mat_path.name}")


def resolve_times_mat_files(
    times_source: str,
    spike_pipeline_dir: Path,
    spike_manual_dir: Path,
) -> list[Path]:
    if times_source == "pipeline":
        files = sorted(
            p
            for p in spike_pipeline_dir.glob("times_*.mat")
            if not p.name.startswith("times_manual_")
        )
        if not files:
            raise FileNotFoundError(f"No times_*.mat files in {spike_pipeline_dir}")
        return files
    if times_source == "manual":
        files = sorted(spike_manual_dir.glob("times_manual_*.mat"))
        if not files:
            raise FileNotFoundError(f"No times_manual_*.mat files in {spike_manual_dir}")
        return files
    raise ValueError(f"Unsupported times_source: {times_source}")


def load_times_mat(mat_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load times_*.mat or times_manual_*.mat (v7.3 HDF5).

    Returns cluster_id, spike_times_series (s), unit_class, timestampsStart (Unix).
    """
    with h5py.File(mat_path, "r") as f:
        if "cluster_class" not in f:
            raise ValueError(f"{mat_path} missing cluster_class")
        cc = np.asarray(f["cluster_class"][()], dtype=np.float64)
        if "timestampsStart" not in f:
            raise ValueError(f"{mat_path} missing timestampsStart")
        ts_start = float(f["timestampsStart"][0, 0])

    if cc.ndim != 2:
        raise ValueError(f"{mat_path}: cluster_class must be 2-D, got shape {cc.shape}")

    if cc.shape[0] == 3:
        cluster_id = cc[0, :].astype(int)
        spike_times_series = cc[1, :]
        unit_class = cc[2, :].astype(int)
    elif cc.shape[1] == 3:
        cluster_id = cc[:, 0].astype(int)
        spike_times_series = cc[:, 1]
        unit_class = cc[:, 2].astype(int)
    elif cc.shape[0] == 2:
        cluster_id = cc[0, :].astype(int)
        spike_times_series = cc[1, :]
        unit_class = np.ones(cluster_id.shape[0], dtype=int)
    elif cc.shape[1] == 2:
        cluster_id = cc[:, 0].astype(int)
        spike_times_series = cc[:, 1]
        unit_class = np.ones(cluster_id.shape[0], dtype=int)
    else:
        raise ValueError(
            f"{mat_path}: expected cluster_class (2|3, n) or (n, 2|3), got {cc.shape}"
        )

    return cluster_id, spike_times_series, unit_class, ts_start


def get_exp9_series_anchor(audio_align_json: Path, timestamps_start: float) -> tuple[float, float]:
    """Return (rec_t0_series, movie_start_rel) in seconds for alignment."""
    payload = json.loads(audio_align_json.read_text())
    rec_t0_unix = _first_numeric(payload.get("rec_t0_unix"), np.nan)
    if not np.isfinite(rec_t0_unix):
        raise ValueError(f"Missing rec_t0_unix in {audio_align_json}")
    rec_t0_series = rec_t0_unix - timestamps_start
    movie_start_rel = _first_numeric(payload.get("start_rel_rec"), 0.0)
    return rec_t0_series, movie_start_rel


def read_bids_recording_duration(
    bids_root: Path,
    subject: str,
    session: str,
    task: str,
    run: str,
    acqs: tuple[str, ...] = ("micro", "macro"),
) -> float:
    for acq in acqs:
        ieeg_json = (
            bids_root
            / f"sub-{subject}"
            / f"ses-{session}"
            / "ieeg"
            / f"sub-{subject}_ses-{session}_task-{task}_acq-{acq}_run-{run}_ieeg.json"
        )
        if not ieeg_json.exists():
            continue
        payload = json.loads(ieeg_json.read_text())
        duration = _first_numeric(payload.get("RecordingDuration"), np.nan)
        if np.isfinite(duration) and duration > 0:
            return float(duration)
    raise FileNotFoundError(
        f"No ieeg.json with RecordingDuration found under {bids_root}/sub-{subject}/ses-{session}/ieeg "
        f"for task={task}, run={run}, acqs={acqs}"
    )


def _movie_duration_video(feature_events_csv: Path) -> float:
    """Return movie duration in stimulus_time seconds (max time from feature CSV)."""
    feat = pd.read_csv(feature_events_csv, usecols=["time"])
    if feat.empty:
        raise ValueError(f"No rows in feature events CSV: {feature_events_csv}")
    return float(feat["time"].max())


def get_movie_start_series(audio_align_json: Path, timestamps_start: float) -> float:
    payload = json.loads(audio_align_json.read_text())
    start_unix = _first_numeric(payload.get("start_unix"), np.nan)
    if not np.isfinite(start_unix):
        raise ValueError(f"Missing start_unix in {audio_align_json}")
    return float(start_unix) - timestamps_start


def build_firing_rate_bins(
    spike_times_movie: np.ndarray,
    movie_duration: float,
    bin_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin spike counts for the full movie (0 .. movie_duration)."""
    n_bins = max(1, int(np.ceil(movie_duration * bin_hz)))
    counts, edges = np.histogram(spike_times_movie, bins=n_bins, range=(0.0, movie_duration))
    return counts.astype(np.float32), edges.astype(np.float64)


def load_movie_micro_downsampled(
    micro_mat: Path,
    rec_start_sample: int,
    n_samples: int,
    in_sfreq: float,
    out_sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load continuous micro for the movie segment, decimated to out_sfreq (volts)."""
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float64)
    with h5py.File(micro_mat, "r") as f:
        scale = float(f["ADBitVolts"][0, 0]) if "ADBitVolts" in f else 1.0
        stop = rec_start_sample + n_samples
        raw = np.asarray(f["data"][0, rec_start_sample:stop], dtype=np.float32)
    volts = raw * np.float32(scale)
    factor = max(1, int(round(in_sfreq / out_sfreq)))
    if factor > 1:
        volts = volts[::factor]
    times = (np.arange(volts.size, dtype=np.float64) / out_sfreq).astype(np.float64)
    return volts, times


def build_spike_events_for_channel(
    mat_path: Path,
    audio_align_json: Path,
    recording_duration: float,
    rec_t0_series: float,
    movie_start_rel: float,
    movie_start_series: float,
    *,
    spike_pipeline_dir: Path | None = None,
    experiment_dir: Path | None = None,
    single_units_only: bool = True,
    time_window: str = "recording",
    movie_duration: float | None = None,
    drift_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray | None, dict[str, object]]:
    """Build spike table and optional (n_spikes, n_samples) waveform matrix from pipeline mat."""
    cluster_id, spike_times_series, unit_class, ts_start = load_times_mat(mat_path)
    channel = channel_from_times_path(mat_path)
    rec_t0_file, _ = get_exp9_series_anchor(audio_align_json, ts_start)
    if not np.isclose(rec_t0_file, rec_t0_series, rtol=0, atol=1e-3):
        raise ValueError(
            f"Series anchor mismatch for {mat_path.name}: "
            f"rec_t0 from audio json={rec_t0_series:.6f}, from file timestampsStart={rec_t0_file:.6f}"
        )

    t_ieeg = spike_times_series - rec_t0_series
    mask = (t_ieeg >= 0.0) & (t_ieeg < recording_duration)
    if time_window == "movie":
        if movie_duration is None:
            raise ValueError("movie_duration is required when time_window='movie'")
        mask &= (t_ieeg >= movie_start_rel) & (t_ieeg < movie_start_rel + movie_duration)
    elif time_window != "recording":
        raise ValueError(f"Unsupported time_window: {time_window}")

    if single_units_only:
        mask &= unit_class == 1

    empty_cols = [
        "onset",
        "movie_onset",
        "duration",
        "channel",
        "cluster_id",
        "unit_class",
        "unit_class_label",
        "series_onset",
        "detection_index",
        "micro_sample_index",
    ]
    if not np.any(mask):
        return pd.DataFrame(columns=empty_cols), None, {}

    series_onset = spike_times_series[mask]
    t_ieeg_masked = t_ieeg[mask]

    detection_index = np.full(series_onset.shape[0], -1, dtype=int)
    micro_sample_index = np.full(series_onset.shape[0], -1, dtype=int)
    waveforms_out: np.ndarray | None = None
    aux_meta: dict[str, object] = {"channel": channel}

    if spike_pipeline_dir is not None:
        pipeline_mat = spike_pipeline_dir / f"{channel}_spikes.mat"
        if pipeline_mat.exists():
            det_times, det_waveforms, sr = load_spikes_pipeline_mat(pipeline_mat)
            detection_index = match_detection_indices(series_onset, det_times)
            micro_sample_index = np.rint(t_ieeg_masked * sr).astype(int)
            waveforms_out = det_waveforms[:, detection_index].T.copy()
            aux_meta.update(
                {
                    "pipeline_spikes_mat": str(pipeline_mat.resolve()),
                    "sampling_frequency_hz": sr,
                    "waveform_n_samples": int(det_waveforms.shape[0]),
                }
            )
        else:
            aux_meta["pipeline_spikes_mat"] = None

    if experiment_dir is not None:
        micro_mat = find_continuous_micro_mat(experiment_dir, channel)
        aux_meta["continuous_micro_mat"] = str(micro_mat.resolve()) if micro_mat else None

    # Divide by drift_multiplier to convert from neural clock to video frame time,
    # consistent with the 'stimulus_time' column in companion events.tsv.
    t_movie_masked = (series_onset - movie_start_series) / drift_multiplier
    rows = pd.DataFrame(
        {
            "onset": t_ieeg_masked,
            "movie_onset": t_movie_masked,
            "duration": 0.0,
            "channel": channel,
            "cluster_id": cluster_id[mask],
            "unit_class": unit_class[mask],
            "series_onset": series_onset,
            "detection_index": detection_index,
            "micro_sample_index": micro_sample_index,
        }
    )
    rows["unit_class_label"] = rows["unit_class"].map(UNIT_CLASS_LABELS).fillna("unknown")
    sort_order = np.argsort(rows["onset"].to_numpy())
    rows = rows.iloc[sort_order].reset_index(drop=True)
    if waveforms_out is not None:
        waveforms_out = waveforms_out[sort_order]
    return rows, waveforms_out, aux_meta


SPIKE_COLUMN_DESCRIPTIONS: dict[str, dict[str, object]] = {
    "onset": {
        "Description": (
            "Spike time in seconds relative to the start of the BIDS iEEG recording "
            "(same reference as sub-*_task-*_run-*_events.tsv in the raw dataset)."
        ),
        "Units": "s",
    },
    "movie_onset": {
        "Description": (
            "Spike time in seconds from audio-aligned movie start in video frame time "
            "(equivalent to frame_number / video_fps; matches the 'stimulus_time' column in companion events.tsv)."
        ),
        "Units": "s",
    },
    "duration": {"Description": "Spike event duration; always 0 for point events.", "Units": "s"},
    "channel": {"Description": "Microwire channel name (e.g. GA1-RA1)."},
    "cluster_id": {
        "Description": "Sorting cluster ID from times_* or times_manual_* cluster_class.",
    },
    "unit_class": {
        "Description": "Fried Lab cluster class code: 1=single unit, 2=multiunit, 3=noise.",
        "Levels": {"1": "single_unit", "2": "multiunit", "3": "noise"},
    },
    "unit_class_label": {"Description": "Human-readable label for unit_class."},
    "series_onset": {
        "Description": (
            "Spike time in the experimental-series reference frame (seconds since "
            "timestampsStart / first recording of the series). Matches spikeTimestamps in "
            "pipeline *_spikes.mat."
        ),
        "Units": "s",
    },
    "detection_index": {
        "Description": (
            "0-based column index into pipeline {channel}_spikes.mat variables spikes and "
            "spikeTimestamps for this event. Use to load the aligned waveform row in the "
            "companion *_spikewaveforms.npy."
        ),
    },
    "micro_sample_index": {
        "Description": (
            "0-based sample index into the Experiment continuous micro file "
            "(CSC_micro/{channel}_001.mat, data[0, :]) at the spike peak, for this "
            "BIDS recording. Equals round(onset * sampling_frequency)."
        ),
    },
}

SPIKE_DATA_NPZ_SCHEMA: dict[str, dict[str, object]] = {
    "channel": {"Description": "Microwire channel name (e.g. GA1-RA1)."},
    "times_source": {"Description": "Source of sorted times: 'pipeline' or 'manual'."},
    "spike_times_movie": {"Description": "Spike time (s) from audio-aligned movie start.", "Units": "s"},
    "spike_times_recording": {"Description": "Spike time (s) from start of BIDS iEEG recording.", "Units": "s"},
    "spike_times_series": {"Description": "Spike time (s) in series reference (timestampsStart = 0).", "Units": "s"},
    "cluster_id": {"Description": "Sorting cluster ID (0=noise, 1+=isolated neuron)."},
    "detection_index": {"Description": "Index into pipeline {channel}_spikes.mat (spikes[:, index])."},
    "micro_sample_index": {"Description": "Sample index into Exp-9 CSC_micro continuous file at spike peak."},
    "waveforms": {"Description": "float32 array (n_spikes, n_waveform_samples). Row i matches events row i."},
    "firing_rate_counts": {"Description": "Spike count per bin over the movie; bin width = 1/firing_rate_hz. Includes all clusters (cluster 0+). For single-unit analyses recompute from spike_times_movie[cluster_id >= 1]."},
    "firing_rate_bin_edges": {"Description": "Bin edges in movie seconds (length = len(counts)+1).", "Units": "s"},
    "firing_rate_hz": {"Description": "Firing-rate binning frequency (e.g. 30 for video FPS).", "Units": "Hz"},
    "micro_movie_volts": {"Description": "Continuous micro voltage for the movie segment, downsampled.", "Units": "V"},
    "micro_movie_times": {"Description": "Time axis for micro_movie_volts (s from movie start).", "Units": "s"},
    "micro_movie_downsample_hz": {"Description": "Sample rate of micro_movie_volts.", "Units": "Hz"},
    "sampling_frequency_hz": {"Description": "Native micro sampling rate used for waveforms / sample indices.", "Units": "Hz"},
    "movie_start_rel": {"Description": "Movie start relative to iEEG recording onset.", "Units": "s"},
    "movie_start_series": {"Description": "Movie start in series time.", "Units": "s"},
    "movie_duration_sec": {"Description": "Exported movie duration.", "Units": "s"},
    "drift_correction_multiplier": {"Description": "Multiply movie_onset by this to align with recording clock."},
    "timestampsStart": {"Description": "Unix time of series reference.", "Units": "s"},
}


# ---------------------------------------------------------------------------
# Alignment helpers (post-hoc rename of legacy *_spikes.tsv → *_events.tsv)
# ---------------------------------------------------------------------------

def _parse_desc_from_name(name: str) -> str | None:
    for pattern in (
        r"_desc-([^_]+(?:_[^_]+)*?)_(?:spikes|events)(?:\.|$)",
        r"_desc-([^_]+)_spikedata\.npz$",
        r"_desc-([^_]+)_spikewaveforms\.npy$",
    ):
        m = re.search(pattern, name)
        if m:
            return m.group(1)
    return None


def _spikes_tsv_to_events_tsv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "cluster_id" not in out.columns:
        raise ValueError("Expected column 'cluster_id' in spikes TSV")
    out["trial_type"] = "spike"
    out["value"] = out["cluster_id"].astype(int)
    lead = ["onset", "duration", "trial_type", "value"]
    return out[lead + [c for c in out.columns if c not in lead]]


def _build_events_json(old_json: dict | None, new_prefix: str, desc: str) -> dict:
    meta: dict[str, object] = {}
    if old_json:
        meta.update(old_json)
    meta.update(SPIKE_COLUMN_DESCRIPTIONS)
    meta["trial_type"] = {
        "Description": "Type of event.",
        "Levels": {"spike": "Sorted microwire spike (point event)."},
    }
    meta["value"] = {"Description": "Sorting cluster ID for this spike."}
    meta["SpikeDataNPZ"] = {
        "Description": (
            "Companion NumPy archive with waveforms, movie-aligned times, firing-rate bins, "
            "and downsampled continuous micro."
        ),
        "Filename": f"{new_prefix}_spikedata.npz",
        "Arrays": SPIKE_DATA_NPZ_SCHEMA,
    }
    meta["SpikeWaveformsNPY"] = {
        "Description": "float32 (n_spikes, n_waveform_samples); duplicate of waveforms in SpikeDataNPZ.",
        "Filename": f"{new_prefix}_spikewaveforms.npy",
    }
    meta["desc"] = {"Description": "Channel label embedded in filename.", "Value": desc}
    return meta


def _collect_channel_descs(ieeg_dir: Path) -> set[str]:
    descs: set[str] = set()
    for p in ieeg_dir.iterdir():
        if p.is_file():
            d = _parse_desc_from_name(p.name)
            if d:
                descs.add(d)
    return descs


def align_spike_derivatives(
    deriv_root: Path,
    subject: str,
    session: str,
    task: str,
    acq: str,
    run: str,
    dry_run: bool = False,
    delete_old: bool = False,
) -> int:
    """
    Post-hoc rename of legacy *_spikes.tsv → *_events.tsv with proper BIDS entities.
    Returns number of channels aligned.
    """
    ieeg_dir = deriv_root / f"sub-{subject}" / f"ses-{session}" / "ieeg"
    if not ieeg_dir.is_dir():
        raise FileNotFoundError(f"Missing derivatives ieeg dir: {ieeg_dir}")

    descs = sorted(_collect_channel_descs(ieeg_dir))
    if not descs:
        print(f"No desc-* channel files found under {ieeg_dir}")
        return 0

    print(f"Found {len(descs)} channels under {ieeg_dir}")
    n_ok = 0
    for desc in descs:
        new_prefix = f"sub-{subject}_ses-{session}_task-{task}_acq-{acq}_run-{run}_desc-{desc}"
        legacy_tsv  = list(ieeg_dir.glob(f"*_desc-{desc}_spikes.tsv"))
        legacy_json = list(ieeg_dir.glob(f"*_desc-{desc}_spikes.json"))
        legacy_wf   = list(ieeg_dir.glob(f"*_desc-{desc}_spikewaveforms.npy"))
        legacy_npz  = list(ieeg_dir.glob(f"*_desc-{desc}_spikedata.npz"))

        new_tsv  = ieeg_dir / f"{new_prefix}_events.tsv"
        if new_tsv.exists() and not legacy_tsv:
            continue
        if not legacy_tsv:
            continue

        old_tsv       = legacy_tsv[0]
        old_json_path = legacy_json[0] if legacy_json else None
        old_wf        = legacy_wf[0]   if legacy_wf   else None
        old_npz       = legacy_npz[0]  if legacy_npz  else None
        new_json = ieeg_dir / f"{new_prefix}_events.json"
        new_wf   = ieeg_dir / f"{new_prefix}_spikewaveforms.npy"
        new_npz  = ieeg_dir / f"{new_prefix}_spikedata.npz"

        if dry_run:
            print(f"  [dry-run] {desc}: {old_tsv.name} -> {new_tsv.name}")
            if old_wf:
                print(f"            {old_wf.name} -> {new_wf.name}")
            if old_npz:
                print(f"            {old_npz.name} -> {new_npz.name}")
            n_ok += 1
            continue

        df = pd.read_csv(old_tsv, sep="\t")
        _spikes_tsv_to_events_tsv(df).to_csv(new_tsv, sep="\t", index=False, float_format="%.6f")

        old_json = json.loads(old_json_path.read_text()) if old_json_path and old_json_path.is_file() else None
        with open(new_json, "w") as f:
            json.dump(_build_events_json(old_json, new_prefix, desc), f, indent=2)

        for old, new in ((old_wf, new_wf), (old_npz, new_npz)):
            if old and old != new:
                if new.exists():
                    new.unlink()
                shutil.move(str(old), str(new))

        if new_npz.is_file():
            z = np.load(new_npz, allow_pickle=True)
            kw = {k: z[k] for k in z.files}
            z.close()
            kw["channel"] = np.array(desc)
            np.savez_compressed(new_npz, **kw)

        if delete_old:
            for p in (old_tsv, old_json_path):
                if p and p.is_file() and p != new_tsv and p != new_json:
                    p.unlink()
            for legacy in ieeg_dir.glob(f"*_desc-{desc}_spikes.*"):
                if legacy.is_file():
                    legacy.unlink()

        print(f"  aligned {desc} -> {new_tsv.name}")
        n_ok += 1

    # Update dataset_description.json
    desc_path = deriv_root / "dataset_description.json"
    payload = {
        "Name": "Sorted microwire spikes (movie window, pipeline times_*)",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "ucla2bids", "Version": "0.2.0",
            "Description": (
                "Exports sorted spikes from times_*.mat for the full audio-aligned movie. "
                "Each channel has BIDS-style events.tsv (trial_type=spike), JSON sidecars, "
                "spikewaveforms.npy, and spikedata.npz."
            )}],
        "SourceDatasets": json.loads(desc_path.read_text()).get("SourceDatasets", []) if desc_path.exists() else [],
    }
    if dry_run:
        print(f"[dry-run] Would update {desc_path}")
    else:
        with open(desc_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Updated {desc_path}")

    print(f"Done. Aligned {n_ok} channel(s).")
    if dry_run:
        print("Re-run without --align-dry-run to apply changes.")
    return n_ok


def write_spike_derivatives_dataset_description(
    deriv_root: Path,
    source_bids_root: Path,
    subject: str,
    session: str,
) -> None:
    deriv_root.mkdir(parents=True, exist_ok=True)
    desc_path = deriv_root / "dataset_description.json"
    if desc_path.exists():
        return

    source_dataset = source_bids_root / "dataset_description.json"
    source_name = "UCLA Movie Paradigm BIDS dataset"
    if source_dataset.exists():
        source_name = json.loads(source_dataset.read_text()).get("Name", source_name)

    with open(desc_path, "w") as f:
        json.dump(
            {
                "Name": "Sorted spike neural data (waveforms, movie micro, firing rates)",
                "BIDSVersion": "1.8.0",
                "DatasetType": "derivative",
                "GeneratedBy": [
                    {
                        "Name": "ucla2bids",
                        "Version": "0.1.0",
                        "Description": (
                            "Exports sorted spikes from times_*.mat (pipeline) or "
                            "times_manual_*.mat with pipeline waveforms, full-movie "
                            "downsampled micro voltage, and binned firing rates."
                        ),
                    }
                ],
                "SourceDatasets": [
                    {
                        "URL": str(source_bids_root.resolve()),
                        "Version": "n/a",
                        "Name": source_name,
                    }
                ],
            },
            f,
            indent=2,
        )


def write_spike_derivatives(
    bids_root: Path,
    audio_align_json: Path,
    subject: str,
    session: str,
    task: str,
    run: str,
    *,
    times_source: str = "pipeline",
    spike_pipeline_dir: Path,
    spike_manual_dir: Path,
    experiment_dir: Path | None = None,
    feature_events_csv: Path | None = None,
    single_units_only: bool = True,
    time_window: str = "movie",
    raster_hz: float = 30.0,
    micro_downsample_hz: float = 1000.0,
    export_movie_micro: bool = True,
) -> tuple[int, int]:
    """Write per-channel spike tables and neural NPZ bundles under derivatives/spike-sorted/."""
    times_files = resolve_times_mat_files(times_source, spike_pipeline_dir, spike_manual_dir)

    _, _, _, timestamps_start = load_times_mat(times_files[0])
    rec_t0_series, movie_start_rel = get_exp9_series_anchor(audio_align_json, timestamps_start)
    movie_start_series = get_movie_start_series(audio_align_json, timestamps_start)
    recording_duration = read_bids_recording_duration(
        bids_root, subject, session, task, run, acqs=("micro", "macro")
    )

    movie_duration: float | None = None       # neural-clock duration (for time-window filter + micro extraction)
    movie_duration_video: float | None = None  # video-frame-time duration (for firing-rate bins + NPZ)
    drift_multiplier = 1.0
    if time_window == "movie":
        if feature_events_csv is None:
            raise ValueError("feature_events_csv is required when time_window='movie'")
        _, drift_multiplier = get_movie_alignment_seconds(audio_align_json)
        movie_duration_video = _movie_duration_video(feature_events_csv)
        movie_duration = movie_duration_video * drift_multiplier
    elif time_window != "recording":
        raise ValueError(f"Unsupported time_window: {time_window}")

    deriv_ieeg_dir = (
        bids_root
        / "derivatives"
        / SPIKE_PIPELINE_NAME
        / f"sub-{subject}"
        / f"ses-{session}"
        / "ieeg"
    )
    deriv_ieeg_dir.mkdir(parents=True, exist_ok=True)
    write_spike_derivatives_dataset_description(
        bids_root / "derivatives" / SPIKE_PIPELINE_NAME,
        bids_root,
        subject,
        session,
    )

    source_ieeg = (
        f"sub-{subject}/ses-{session}/ieeg/"
        f"sub-{subject}_ses-{session}_task-{task}_acq-micro_run-{run}_ieeg.vhdr"
    )
    n_written = 0
    n_skipped = 0

    for mat_path in times_files:
        channel = channel_from_times_path(mat_path)
        spikes, waveforms, aux_meta = build_spike_events_for_channel(
            mat_path,
            audio_align_json,
            recording_duration,
            rec_t0_series,
            movie_start_rel,
            movie_start_series,
            spike_pipeline_dir=spike_pipeline_dir,
            experiment_dir=experiment_dir,
            single_units_only=single_units_only,
            time_window=time_window,
            movie_duration=movie_duration,
            drift_multiplier=drift_multiplier,
        )
        if spikes.empty:
            n_skipped += 1
            continue

        desc = channel_to_desc(channel)
        prefix = f"sub-{subject}_ses-{session}_task-{task}_acq-micro_run-{run}_desc-{desc}"
        tsv_path      = deriv_ieeg_dir / f"{prefix}_events.tsv"
        json_path     = deriv_ieeg_dir / f"{prefix}_events.json"
        wave_npy_path = deriv_ieeg_dir / f"{prefix}_spikewaveforms.npy"
        npz_path      = deriv_ieeg_dir / f"{prefix}_spikedata.npz"

        events_df = spikes.copy()
        events_df["trial_type"] = "spike"
        events_df["value"] = events_df["cluster_id"].astype(int)
        lead = ["onset", "duration", "trial_type", "value"]
        events_df = events_df[lead + [c for c in events_df.columns if c not in lead]]
        events_df.to_csv(tsv_path, sep="\t", index=False, float_format="%.6f")
        if waveforms is not None:
            np.save(wave_npy_path, waveforms)

        sr = float(aux_meta.get("sampling_frequency_hz", 32000.0))
        # movie_duration_video: stimulus_time seconds (matches events.tsv stimulus_time column)
        # movie_duration: neural-clock seconds (used for micro sample extraction)
        mov_dur_video = float(
            movie_duration_video if movie_duration_video is not None else recording_duration
        )
        mov_dur_neural = float(movie_duration if movie_duration is not None else recording_duration)
        t_movie = spikes["movie_onset"].to_numpy(dtype=np.float64)
        rate_counts, rate_edges = build_firing_rate_bins(t_movie, mov_dur_video, raster_hz)

        micro_volts = np.zeros(0, dtype=np.float32)
        micro_times = np.zeros(0, dtype=np.float64)
        if export_movie_micro and time_window == "movie" and experiment_dir is not None:
            micro_mat = find_continuous_micro_mat(experiment_dir, channel)
            if micro_mat is not None:
                rec_start = int(round(movie_start_rel * sr))
                n_samp = int(round(mov_dur_neural * sr))
                micro_volts, micro_times = load_movie_micro_downsampled(
                    micro_mat,
                    rec_start,
                    n_samp,
                    in_sfreq=sr,
                    out_sfreq=micro_downsample_hz,
                )

        np.savez_compressed(
            npz_path,
            channel=np.array(channel),
            times_source=np.array(times_source),
            spike_times_movie=t_movie.astype(np.float64),
            spike_times_recording=spikes["onset"].to_numpy(dtype=np.float64),
            spike_times_series=spikes["series_onset"].to_numpy(dtype=np.float64),
            cluster_id=spikes["cluster_id"].to_numpy(dtype=np.int32),
            detection_index=spikes["detection_index"].to_numpy(dtype=np.int32),
            micro_sample_index=spikes["micro_sample_index"].to_numpy(dtype=np.int64),
            waveforms=waveforms if waveforms is not None else np.zeros((0, 0), dtype=np.float32),
            firing_rate_counts=rate_counts,
            firing_rate_bin_edges=rate_edges,
            firing_rate_hz=np.float64(raster_hz),
            micro_movie_volts=micro_volts,
            micro_movie_times=micro_times,
            micro_movie_downsample_hz=np.float64(micro_downsample_hz),
            sampling_frequency_hz=np.float64(sr),
            movie_start_rel=np.float64(movie_start_rel),
            movie_start_series=np.float64(movie_start_series),
            movie_duration_sec=np.float64(mov_dur_video),
            drift_correction_multiplier=np.float64(drift_multiplier),
            timestampsStart=np.float64(timestamps_start),
        )

        metadata = {
            **SPIKE_COLUMN_DESCRIPTIONS,
            "trial_type": {
                "Description": "Type of event.",
                "Levels": {"spike": "Sorted microwire spike (point event)."},
            },
            "value": {"Description": "Sorting cluster ID for this spike."},
            "SpikeDataNPZ": {
                "Description": (
                    "Companion NumPy archive with waveforms, movie-aligned times, firing-rate bins, "
                    "and downsampled continuous micro."
                ),
                "Filename": npz_path.name,
                "Arrays": SPIKE_DATA_NPZ_SCHEMA,
            },
            "Sources": {
                "Description": "Raw iEEG recording and manual spike sorting inputs.",
                "References": [
                    source_ieeg,
                    str(mat_path.resolve()),
                ],
            },
            "time_reference": {
                "Description": "onset is relative to the start of the BIDS iEEG recording for this task/run.",
            },
            "series_timestampsStart": {
                "Description": "Unix time of the experimental-series reference (column 2 of times_manual).",
                "Units": "s",
                "Value": timestamps_start,
            },
            "experiment_rec_t0_series": {
                "Description": "Experiment recording start in series time (rec_t0_unix - timestampsStart).",
                "Units": "s",
                "Value": rec_t0_series,
            },
            "times_source": {
                "Description": "Sorted spike times read from pipeline times_*.mat or manual times_manual_*.mat.",
                "Value": times_source,
            },
            "time_window": {
                "Description": "Which subset of series spikes was exported.",
                "Value": time_window,
            },
            "spikedata_npz": {
                "Description": (
                    "Self-contained neural bundle: waveforms (n_spikes x n_samples), spike_times_movie "
                    "(0..movie_duration), firing_rate_counts at firing_rate_hz, and micro_movie_volts "
                    "(continuous micro for the movie segment, downsampled)."
                ),
                "Filename": npz_path.name,
            },
            "NeuralDataExtraction": {
                "Description": (
                    "Primary bundle: load spikedata.npz. Row i waveforms[i]; "
                    "spike_times_movie[i] is in stimulus_time (matches events.tsv stimulus_time column). "
                    "To get neural-clock time: spike_times_movie * drift_correction_multiplier."
                ),
            },
        }
        if waveforms is not None:
            metadata["spikewaveforms_npy"] = {
                "Description": (
                    "float32 array shape (n_spikes, n_waveform_samples). Row order matches this TSV. "
                    "Same waveforms as pipeline spikes.mat[:, detection_index]."
                ),
                "Filename": wave_npy_path.name,
                "SamplingFrequency": aux_meta.get("sampling_frequency_hz"),
                "WaveformSamples": aux_meta.get("waveform_n_samples"),
            }
        if aux_meta.get("pipeline_spikes_mat"):
            metadata["Sources"]["References"].append(str(aux_meta["pipeline_spikes_mat"]))
        if aux_meta.get("continuous_micro_mat"):
            metadata["continuous_micro_mat"] = {
                "Description": "Unpacked Neuralynx micro channel for this experiment (int16, use ADBitVolts for volts).",
                "Path": aux_meta["continuous_micro_mat"],
            }
            metadata["Sources"]["References"].append(str(aux_meta["continuous_micro_mat"]))
        if time_window == "movie":
            metadata["movie_start_rel"] = {
                "Description": "Movie start relative to recording onset (matches raw events.tsv).",
                "Units": "s",
                "Value": movie_start_rel,
            }
            metadata["movie_duration"] = {
                "Description": "Movie duration in stimulus_time seconds (max time from feature CSV).",
                "Units": "s",
                "Value": movie_duration_video,
            }

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        n_written += 1

    return n_written, n_skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UCLA iEEG BIDS metadata (electrodes + events).")
    parser.add_argument("--experiment-dir", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/ucla_data/572/Experiment-9"))
    parser.add_argument("--localization-xlsx", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/ucla_data/572/sub-572_localizations.xlsx"))
    parser.add_argument("--bids-root", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/bids"))
    parser.add_argument("--feature-events-csv", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_events_vowel_word_features.csv"))
    parser.add_argument("--phoneme-csv", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_phonemes.csv"))
    parser.add_argument("--characters-csv", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/40m_act_24_S06E01_30fps_characters.csv"))
    parser.add_argument("--concepts-csv", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_8concepts_merged.csv"))
    parser.add_argument("--audio-align-json", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/ucla_data/572/Experiment-9/Audio/572_exp_09_preSleep_movie_24_audio_movie_start_time.json"))
    parser.add_argument("--subject", type=str, default="572")
    parser.add_argument("--session", type=str, default="01")
    parser.add_argument("--task", type=str, default="movie24presleep")
    parser.add_argument("--acqs", nargs="+", default=["macro", "micro"])
    parser.add_argument("--run", type=str, default="01")
    parser.add_argument("--skip-signal-export", action="store_true")
    parser.add_argument(
        "--spike-manual-dir",
        type=Path,
        default=Path(
            "/store/scratch/bsow/Documents/UCLA_24/data/ucla_data/572/"
            "Experiment-8-9-10-11/CSC_micro_spikes_CAR"
        ),
        help="Directory with times_manual_*.mat (full-series manual sort).",
    )
    parser.add_argument(
        "--spike-pipeline-dir",
        type=Path,
        default=Path(
            "/store/scratch/bsow/Documents/UCLA_24/data/ucla_data/572/"
            "Experiment-8-9-10-11/CSC_micro_spikes_removePLI-0_CAR-1_rejectNoiseSpikes-1"
        ),
        help="Directory with {channel}_spikes.mat (waveforms + detection times).",
    )
    parser.add_argument(
        "--skip-spike-derivatives",
        action="store_true",
        help="Do not write derivatives/spike-sorted/ spike tables.",
    )
    parser.add_argument(
        "--spike-times-source",
        choices=("pipeline", "manual"),
        default="pipeline",
        help="Use times_*.mat from --spike-pipeline-dir or times_manual_*.mat from --spike-manual-dir.",
    )
    parser.add_argument(
        "--spike-time-window",
        choices=("recording", "movie"),
        default="movie",
        help="Export spikes for full experiment recording, or full movie segment only (default).",
    )
    parser.add_argument(
        "--spike-raster-hz",
        type=float,
        default=30.0,
        help="Bin width for firing_rate_counts in spikedata.npz (e.g. 30 for video FPS).",
    )
    parser.add_argument(
        "--spike-micro-downsample-hz",
        type=float,
        default=1000.0,
        help="Sample rate for micro_movie_volts in spikedata.npz.",
    )
    parser.add_argument(
        "--no-export-movie-micro",
        action="store_true",
        help="Skip embedding downsampled continuous micro in spikedata.npz.",
    )
    parser.add_argument(
        "--spike-include-multiunit",
        action="store_true",
        help="Include multiunit/noise (manual only); pipeline times_* exports all sorted spikes.",
    )

    # Post-hoc alignment of legacy *_spikes.tsv → *_events.tsv
    parser.add_argument(
        "--align-derivatives",
        action="store_true",
        help=(
            "Rename legacy *_spikes.tsv → *_events.tsv under derivatives/spike-sorted/, "
            "adding acq/run BIDS entities and trial_type/value columns. "
            "When passed, only alignment runs (no BIDS build)."
        ),
    )
    parser.add_argument("--align-dry-run", action="store_true",
        help="Preview alignment renames without writing files (use with --align-derivatives).")
    parser.add_argument("--align-delete-old", action="store_true",
        help="Delete legacy *_spikes.tsv/json after successful alignment.")
    args = parser.parse_args()

    if args.align_derivatives:
        align_spike_derivatives(
            deriv_root=args.bids_root / "derivatives" / SPIKE_PIPELINE_NAME,
            subject=args.subject,
            session=args.session,
            task=args.task,
            acq="micro",
            run=args.run,
            dry_run=args.align_dry_run,
            delete_old=args.align_delete_old,
        )
        return

    if args.spike_include_multiunit and args.spike_times_source == "pipeline":
        print(
            "Warning: --spike-include-multiunit has no effect with --spike-times-source pipeline. "
            "Pipeline times_*.mat files do not store unit_class; all pipeline spikes are assigned "
            "class 1 (single unit). Use --spike-times-source manual to filter by unit class."
        )

    micro_channels, macro_channels = read_channel_names(args.experiment_dir)
    loc_df = load_localization_sheet(args.localization_xlsx)
    electrodes_df, unmatched = build_electrodes_table(micro_channels, macro_channels, loc_df)

    if not args.skip_signal_export:
        for acq in args.acqs:
            write_brainvision_ieeg(
                experiment_dir=args.experiment_dir,
                bids_root=args.bids_root,
                subject=args.subject,
                session=args.session,
                task=args.task,
                acq=acq,
                run=args.run,
            )

    write_ieeg_metadata(args.bids_root, args.subject, args.session, electrodes_df)
    events_df, timing_meta = build_events_table(
        args.feature_events_csv,
        args.audio_align_json,
        args.phoneme_csv,
        args.characters_csv,
        args.concepts_csv,
    )
    write_events_files(
        args.bids_root,
        args.subject,
        args.session,
        args.task,
        args.acqs,
        args.run,
        events_df,
        timing_meta=timing_meta,
    )

    print(f"Wrote electrodes for sub-{args.subject}, ses-{args.session}")
    print(f"Total channels: {len(electrodes_df)}")
    print(f"Unmatched channels (written as n/a): {len(unmatched)}")
    for ch in unmatched:
        print(f" - {ch}")
    print(f"Wrote events rows: {len(events_df)}")
    print(f"Acquisitions with event files: {', '.join(args.acqs)}")

    if not args.skip_spike_derivatives:
        n_written, n_skipped = write_spike_derivatives(
            bids_root=args.bids_root,
            audio_align_json=args.audio_align_json,
            subject=args.subject,
            session=args.session,
            task=args.task,
            run=args.run,
            times_source=args.spike_times_source,
            spike_pipeline_dir=args.spike_pipeline_dir,
            spike_manual_dir=args.spike_manual_dir,
            experiment_dir=args.experiment_dir,
            feature_events_csv=args.feature_events_csv,
            single_units_only=not args.spike_include_multiunit,
            time_window=args.spike_time_window,
            raster_hz=args.spike_raster_hz,
            micro_downsample_hz=args.spike_micro_downsample_hz,
            export_movie_micro=not args.no_export_movie_micro,
        )
        deriv_dir = args.bids_root / "derivatives" / SPIKE_PIPELINE_NAME
        print(f"Wrote spike derivatives to {deriv_dir}")
        print(f"  Channel bundles (tsv + json + spikewaveforms.npy + spikedata.npz): {n_written}")
        print(f"  Channels skipped (no spikes in window): {n_skipped}")
        print(f"  Times source: {args.spike_times_source}  |  Time window: {args.spike_time_window}")


if __name__ == "__main__":
    main()
