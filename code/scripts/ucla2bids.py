#!/usr/bin/env python3

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import re
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
    # GA1-RA1 -> RA1 (localization uses shaft/contact naming)
    if "-" in channel:
        channel = channel.split("-", 1)[1]
    return normalize_name(channel)


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


def normalize_event_token(value: object) -> str:
    return re.sub(r"[^\w']+", "", str(value).lower())


def srt_time_to_seconds(ts: str) -> float:
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def load_srt_sentence_starts(srt_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    text = srt_path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    starts: list[float] = []
    ends: list[float] = []
    first_tokens: list[str] = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_txt, end_txt = [x.strip() for x in lines[1].split("-->")]
        sent_text = " ".join(lines[2:]).strip()
        first_raw = sent_text.split()[0] if sent_text else ""
        starts.append(srt_time_to_seconds(start_txt))
        ends.append(srt_time_to_seconds(end_txt))
        first_tokens.append(normalize_event_token(first_raw))
    return np.asarray(starts, dtype=float), np.asarray(ends, dtype=float), first_tokens


def slugify_label(value: object) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).lower())).strip("_")


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
    "biphone_surprisal",
    *(f"mel_{i:02d}" for i in range(16)),
)


def build_events_table(
    feature_csv: Path,
    audio_align_json: Path,
    phoneme_csv: Path,
    subtitle_srt: Path,
    characters_csv: Path,
    concepts_csv: Path,
) -> pd.DataFrame:
    feat = pd.read_csv(feature_csv)
    phon = pd.read_csv(phoneme_csv)
    movie_start_rel, drift_multiplier = get_movie_alignment_seconds(audio_align_json)

    for col in ("vowel_onset", "word_onset", "word_ner"):
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

    ev["onset"] = movie_start_rel + ev["time"].astype(float) * drift_multiplier
    ev["duration"] = ev[["vowel_duration", "word_duration"]].fillna(0.0).max(axis=1)
    ev["Event"] = np.where(ev["word_onset"].astype(str).str.strip().ne(""), "word_onset", "no_event")
    sub_starts, sub_ends, sub_first_tokens = load_srt_sentence_starts(subtitle_srt)
    word_onset_idx = np.flatnonzero(ev["Event"].eq("word_onset").to_numpy())
    for idx in word_onset_idx:
        word_t = float(ev.iloc[idx]["time"])
        token = normalize_event_token(ev.iloc[idx]["word_onset"])
        if not token or sub_starts.size == 0:
            continue
        sub_i = bisect_right(sub_starts, word_t) - 1
        if sub_i < 0 or sub_i >= len(sub_first_tokens):
            continue
        if not (sub_starts[sub_i] <= word_t <= sub_ends[sub_i]):
            continue
        if token == sub_first_tokens[sub_i]:
            ev.iloc[idx, ev.columns.get_loc("Event")] = "first_word_onset"

    ev["word_ner"] = ev["word_ner"].replace("", "n/a")

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

    # Event override: mark only Jack Bauer on-screen onsets (0->1 transitions).
    j_bauer_col = next((c for c in char_cols if normalize_event_token(c) == "jbauer"), None)
    if j_bauer_col is not None:
        jb = pd.to_numeric(char_aligned[j_bauer_col], errors="coerce").fillna(0.0).to_numpy()
        jb_present = jb > 0
        jb_onset = jb_present & np.concatenate(([True], ~jb_present[:-1]))
        ev.loc[jb_onset, "Event"] = "j_bauer"

    # 1 Hz concept labels expanded to frame rows by floor(Time).
    concept_df = pd.read_csv(concepts_csv)
    concept_cols = list(concept_df.columns)
    concept_name_map = {c: f"concept_{slugify_label(c)}" for c in concept_cols}
    sec_idx = np.floor(ev["time"].astype(float).to_numpy()).astype(int)
    n_concepts = len(concept_df)
    valid = (sec_idx >= 0) & (sec_idx < n_concepts)
    for src, dst in concept_name_map.items():
        vals = np.zeros(len(ev), dtype=float)
        arr = pd.to_numeric(concept_df[src], errors="coerce").fillna(0.0).to_numpy()
        vals[valid] = arr[sec_idx[valid]]
        ev[dst] = vals

    keep_cols = [
        "onset",
        "duration",
        "Event",
        "frame",
        "Time",
        "vowel_onset",
        "word_onset",
        "vowel_duration",
        "word_duration",
        "word_frequency",
        "word_ner",
        *AUDIO_PHONOLOGICAL_FEATURE_COLUMNS,
        *char_name_map.values(),
        *concept_name_map.values(),
    ]
    ev = ev.rename(columns={"time": "Time"})
    ev = ev[keep_cols].sort_values("onset").reset_index(drop=True)
    return ev


EVENT_COLUMN_DESCRIPTIONS: dict[str, dict[str, object]] = {
    "onset": {"Description": "Onset of the event in seconds relative to the start of the iEEG recording."},
    "duration": {"Description": "Event duration in seconds. Uses max(word_duration, vowel_duration) per frame row."},
    "Event": {
        "Description": (
            "Frame-level event label: 'first_word_onset' for sentence-initial words "
            "(matched against subtitle sentence-initial token), 'word_onset' for other word onsets, "
            "'j_bauer' for Jack Bauer on-screen onset frames, "
            "otherwise 'no_event'."
        ),
        "Levels": {
            "first_word_onset": "Frame contains a sentence-initial word onset based on subtitle sentence starts.",
            "word_onset": "Frame contains a word onset token.",
            "j_bauer": "Frame is an onset where Jack Bauer appears on screen (0->1 transition).",
            "no_event": "Frame does not contain a word onset token.",
        },
    },
    "frame": {"Description": "Frame index in the movie-derived linguistic feature table."},
    "Time": {"Description": "Time in seconds from movie start in the feature table."},
    "vowel_onset": {"Description": "IPA vowel token at onset row, else empty."},
    "word_onset": {"Description": "Word token at onset row, else empty."},
    "vowel_duration": {"Description": "Duration in seconds of vowel segment, written only at onset rows."},
    "word_duration": {"Description": "Duration in seconds of word segment, written only at onset rows."},
    "word_frequency": {"Description": "Zipf frequency from wordfreq package (language='en') at word onset rows."},
    "word_ner": {"Description": "Named entity label at word onset rows derived from subtitle-context NER; n/a if none."},
    "env": {"Description": "Short-time loudness / envelope feature aligned to the movie audio track (frame-wise)."},
    "env_peak_rate": {"Description": "Rate of envelope peaks in the analysis window (frame-wise)."},
    "pitch_hz": {"Description": "F0 estimate in Hz from the movie audio (frame-wise)."},
    "pitch_norm": {"Description": "Speaker-normalized pitch (e.g. z-scored F0) in arbitrary units (frame-wise)."},
    "pitch_up": {"Description": "Magnitude of upward pitch movement in the frame (frame-wise)."},
    "pitch_down": {"Description": "Magnitude of downward pitch movement in the frame (frame-wise)."},
    "pause_duration_ms": {
        "Description": "Duration of preceding silence / pause in milliseconds when defined at word onsets; else 0.",
        "Units": "ms",
    },
    "word_char_len": {"Description": "Character length of the word token at word-onset rows; 0 otherwise."},
    "biphone_surprisal": {"Description": "Phoneme biphone surprisal (negative log probability) at relevant frames; 0 if n/a."},
    **{
        f"mel_{i:02d}": {
            "Description": f"Log-mel spectrum bin {i} (normalized), frame-aligned to movie audio.",
        }
        for i in range(16)
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
) -> None:
    ieeg_dir = bids_root / f"sub-{subject}" / f"ses-{session}" / "ieeg"
    ieeg_dir.mkdir(parents=True, exist_ok=True)

    for acq in acqs:
        prefix = f"sub-{subject}_ses-{session}_task-{task}_acq-{acq}_run-{run}_events"
        tsv_path = ieeg_dir / f"{prefix}.tsv"
        json_path = ieeg_dir / f"{prefix}.json"

        events_df.to_csv(tsv_path, sep="\t", index=False, na_rep="n/a")

        metadata = {}
        for col in events_df.columns:
            metadata[col] = EVENT_COLUMN_DESCRIPTIONS.get(
                col,
                {"Description": f"{col} (task-specific event column)."},
            )
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UCLA iEEG BIDS metadata (electrodes + events).")
    parser.add_argument("--experiment-dir", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/ucla_data/572/Experiment-9"))
    parser.add_argument("--localization-xlsx", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/ucla_data/572/sub-572_localizations.xlsx"))
    parser.add_argument("--bids-root", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/bids"))
    parser.add_argument("--feature-events-csv", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_events_vowel_word_features.csv"))
    parser.add_argument("--phoneme-csv", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_phonemes.csv"))
    parser.add_argument("--subtitle-srt", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01.srt"))
    parser.add_argument("--characters-csv", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/40m_act_24_S06E01_30fps_characters.csv"))
    parser.add_argument("--concepts-csv", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/24_S06E01_8concepts_merged.csv"))
    parser.add_argument("--audio-align-json", type=Path, default=Path("/store/scratch/bsow/Documents/UCLA_24/data/ucla_data/572/Experiment-9/Audio/572_exp_09_preSleep_movie_24_audio_movie_start_time.json"))
    parser.add_argument("--subject", type=str, default="572")
    parser.add_argument("--session", type=str, default="01")
    parser.add_argument("--task", type=str, default="movie24presleep")
    parser.add_argument("--acqs", nargs="+", default=["macro", "micro"])
    parser.add_argument("--run", type=str, default="01")
    parser.add_argument("--skip-signal-export", action="store_true")
    args = parser.parse_args()

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
    events_df = build_events_table(
        args.feature_events_csv,
        args.audio_align_json,
        args.phoneme_csv,
        args.subtitle_srt,
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
    )

    print(f"Wrote electrodes for sub-{args.subject}, ses-{args.session}")
    print(f"Total channels: {len(electrodes_df)}")
    print(f"Unmatched channels (written as n/a): {len(unmatched)}")
    for ch in unmatched:
        print(f" - {ch}")
    print(f"Wrote events rows: {len(events_df)}")
    print(f"Acquisitions with event files: {', '.join(args.acqs)}")


if __name__ == "__main__":
    main()
