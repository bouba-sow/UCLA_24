#!/usr/bin/env python3
"""
Re-align spike-sorted derivatives to a stricter BIDS-style layout (metadata only).

Changes per channel:
  - *_spikes.tsv  -> *_events.tsv with trial_type='spike' and value=cluster_id
  - Filenames gain acq-<acq>_run-<run> entities to match raw iEEG
  - *_spikes.json -> *_events.json with SpikeDataNPZ block for companion .npz
  - Companion *_spikewaveforms.npy and *_spikedata.npz renamed to match

Updates derivatives/spike-sorted/dataset_description.json.

Example (run on a compute node, not login node):
  python code/scripts/align_spike_derivatives_bids.py \\
    --deriv-root /store/scratch/bsow/Documents/UCLA_24/data/bids/derivatives/spike-sorted \\
    --subject 572 --session 01 --task movie24presleep --acq micro --run 01

  # Preview only:
  python code/scripts/align_spike_derivatives_bids.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SPIKE_DATA_NPZ_SCHEMA: dict[str, dict[str, object]] = {
    "channel": {"Description": "Microwire channel name (e.g. GA1-RA1)."},
    "times_source": {
        "Description": "Source of sorted times: 'pipeline' (times_*.mat) or 'manual' (times_manual_*.mat)."
    },
    "spike_times_movie": {
        "Description": "Spike time (s) from audio-aligned movie start (0 = first frame).",
        "Units": "s",
    },
    "spike_times_recording": {
        "Description": "Spike time (s) from start of the BIDS iEEG recording (Exp 9).",
        "Units": "s",
    },
    "spike_times_series": {
        "Description": "Spike time (s) in experimental-series reference (timestampsStart = 0).",
        "Units": "s",
    },
    "cluster_id": {"Description": "Sorting cluster ID (duplicate of value for convenience)."},
    "detection_index": {
        "Description": "Index into pipeline {channel}_spikes.mat (spikes[:, index])."
    },
    "micro_sample_index": {
        "Description": "Sample index into Exp-9 CSC_micro continuous file at spike peak."
    },
    "waveforms": {
        "Description": "float32 array (n_spikes, n_waveform_samples). Row i matches events row i.",
    },
    "firing_rate_counts": {
        "Description": "Spike count per bin over the movie; bin width = 1/firing_rate_hz.",
    },
    "firing_rate_bin_edges": {
        "Description": "Bin edges in movie seconds (length = len(counts)+1).",
        "Units": "s",
    },
    "firing_rate_hz": {"Description": "Firing-rate binning frequency (e.g. 30 for video FPS).", "Units": "Hz"},
    "micro_movie_volts": {
        "Description": "Continuous micro voltage for the movie segment, downsampled.",
        "Units": "V",
    },
    "micro_movie_times": {
        "Description": "Time axis for micro_movie_volts (s from movie start).",
        "Units": "s",
    },
    "micro_movie_downsample_hz": {
        "Description": "Sample rate of micro_movie_volts.",
        "Units": "Hz",
    },
    "sampling_frequency_hz": {
        "Description": "Native micro sampling rate used for waveforms / sample indices.",
        "Units": "Hz",
    },
    "movie_start_rel": {
        "Description": "Movie start relative to iEEG recording onset (matches raw events.tsv).",
        "Units": "s",
    },
    "movie_start_series": {"Description": "Movie start in series time.", "Units": "s"},
    "movie_duration_sec": {"Description": "Exported movie duration.", "Units": "s"},
    "drift_correction_multiplier": {
        "Description": "Multiply movie_onset by this to align with recording clock."
    },
    "timestampsStart": {"Description": "Unix time of series reference.", "Units": "s"},
}

EVENTS_COLUMN_DESCRIPTIONS: dict[str, dict[str, object]] = {
    "onset": {
        "Description": (
            "Event onset in seconds relative to the start of the matching raw iEEG recording "
            "(sub-*_task-*_acq-micro_run-*_ieeg)."
        ),
        "Units": "s",
    },
    "duration": {"Description": "Event duration in seconds; 0 for point-like spikes.", "Units": "s"},
    "trial_type": {
        "Description": "Type of event.",
        "Levels": {"spike": "Sorted microwire spike (point event)."},
    },
    "value": {"Description": "Sorting cluster ID for this spike."},
    "movie_onset": {
        "Description": "Spike time (s) from audio-aligned movie start (0 = first frame).",
        "Units": "s",
    },
    "channel": {"Description": "Microwire channel name (e.g. GA1-RA1)."},
    "unit_class": {
        "Description": "Fried Lab class when available: 1=single unit, 2=multiunit, 3=noise.",
    },
    "unit_class_label": {"Description": "Human-readable unit_class label."},
    "series_onset": {
        "Description": "Spike time in series reference frame (seconds since timestampsStart).",
        "Units": "s",
    },
    "detection_index": {"Description": "Index into pipeline *_spikes.mat for this event."},
    "micro_sample_index": {
        "Description": "Sample index into continuous micro at spike peak (recording-relative)."
    },
}


def _parse_desc_from_name(name: str) -> str | None:
    m = re.search(r"_desc-([^_]+(?:_[^_]+)*?)_(?:spikes|events)(?:\.|$)", name)
    if m:
        return m.group(1)
    m = re.search(r"_desc-([^_]+)_spikedata\.npz$", name)
    if m:
        return m.group(1)
    m = re.search(r"_desc-([^_]+)_spikewaveforms\.npy$", name)
    if m:
        return m.group(1)
    return None


def build_bids_prefix(
    subject: str,
    session: str,
    task: str,
    acq: str,
    run: str,
    desc: str,
) -> str:
    return (
        f"sub-{subject}_ses-{session}_task-{task}_acq-{acq}_run-{run}_desc-{desc}"
    )


def spikes_tsv_to_events_tsv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "cluster_id" not in out.columns:
        raise ValueError("Expected column 'cluster_id' in spikes TSV")
    out["trial_type"] = "spike"
    out["value"] = out["cluster_id"].astype(int)
    lead = ["onset", "duration", "trial_type", "value"]
    extras = [c for c in out.columns if c not in lead]
    return out[lead + extras]


def build_events_json(old_json: dict | None, new_prefix: str, desc: str) -> dict:
    meta: dict[str, object] = {}
    if old_json:
        meta.update(old_json)
    for col, spec in EVENTS_COLUMN_DESCRIPTIONS.items():
        meta[col] = spec
    meta["trial_type"] = EVENTS_COLUMN_DESCRIPTIONS["trial_type"]
    meta["value"] = EVENTS_COLUMN_DESCRIPTIONS["value"]
    meta["SpikeDataNPZ"] = {
        "Description": (
            "Companion NumPy archive with waveforms, movie-aligned times, firing-rate bins, "
            "and downsampled continuous micro. Fried Lab target format is NWB Units; "
            "this NPZ is a practical stand-in until NWB export is available."
        ),
        "Filename": f"{new_prefix}_spikedata.npz",
        "Arrays": SPIKE_DATA_NPZ_SCHEMA,
    }
    meta["SpikeWaveformsNPY"] = {
        "Description": "float32 (n_spikes, n_waveform_samples); duplicate of waveforms in SpikeDataNPZ.",
        "Filename": f"{new_prefix}_spikewaveforms.npy",
    }
    meta["BIDSAlignmentNote"] = {
        "Description": (
            "Events file follows BIDS events.tsv conventions (onset, duration, trial_type). "
            "Raw iEEG: sub-{subject}/ses-{session}/ieeg/{prefix}_ieeg.vhdr without 'desc'."
        ).format(subject="*", session="*", prefix=new_prefix.rsplit("_desc-", 1)[0]),
    }
    meta["desc"] = {"Description": "Channel label embedded in filename.", "Value": desc}
    return meta


def update_dataset_description(deriv_root: Path, dry_run: bool) -> None:
    path = deriv_root / "dataset_description.json"
    payload = {
        "Name": "Sorted microwire spikes (movie window, pipeline times_*)",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "ucla2bids + align_spike_derivatives_bids",
                "Version": "0.2.0",
                "Description": (
                    "Exports sorted spikes from pipeline times_*.mat (or times_manual_*.mat) "
                    "for the full audio-aligned movie. Each channel has BIDS-style events.tsv "
                    "(trial_type=spike), JSON sidecars, spikewaveforms.npy, and spikedata.npz "
                    "(waveforms, movie micro LFP, firing-rate bins). Long-term target: NWB Units."
                ),
            }
        ],
        "SourceDatasets": [],
    }
    if path.exists():
        existing = json.loads(path.read_text())
        payload["SourceDatasets"] = existing.get("SourceDatasets", [])
    if dry_run:
        print(f"[dry-run] Would write {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Updated {path}")


def collect_channel_descs(ieeg_dir: Path) -> set[str]:
    descs: set[str] = set()
    for path in ieeg_dir.iterdir():
        if not path.is_file():
            continue
        d = _parse_desc_from_name(path.name)
        if d:
            descs.add(d)
    return descs


def align_channel(
    ieeg_dir: Path,
    desc: str,
    subject: str,
    session: str,
    task: str,
    acq: str,
    run: str,
    dry_run: bool,
    delete_old: bool,
) -> bool:
    prefix = build_bids_prefix(subject, session, task, acq, run, desc)

    # Locate legacy files (any of these patterns)
    legacy_events = list(ieeg_dir.glob(f"*_desc-{desc}_spikes.tsv"))
    legacy_json = list(ieeg_dir.glob(f"*_desc-{desc}_spikes.json"))
    legacy_wf = list(ieeg_dir.glob(f"*_desc-{desc}_spikewaveforms.npy"))
    legacy_npz = list(ieeg_dir.glob(f"*_desc-{desc}_spikedata.npz"))

    # Already aligned?
    new_events = ieeg_dir / f"{prefix}_events.tsv"
    if new_events.exists() and not legacy_events:
        return False

    if not legacy_events:
        print(f"  skip {desc}: no *_spikes.tsv found")
        return False

    old_tsv = legacy_events[0]
    old_json_path = legacy_json[0] if legacy_json else None
    old_wf = legacy_wf[0] if legacy_wf else None
    old_npz = legacy_npz[0] if legacy_npz else None

    new_tsv = ieeg_dir / f"{prefix}_events.tsv"
    new_json = ieeg_dir / f"{prefix}_events.json"
    new_wf = ieeg_dir / f"{prefix}_spikewaveforms.npy"
    new_npz = ieeg_dir / f"{prefix}_spikedata.npz"

    df = pd.read_csv(old_tsv, sep="\t")
    events_df = spikes_tsv_to_events_tsv(df)

    old_json: dict | None = None
    if old_json_path and old_json_path.is_file():
        old_json = json.loads(old_json_path.read_text())
    events_meta = build_events_json(old_json, prefix, desc)

    if dry_run:
        print(f"  [dry-run] {desc}: {old_tsv.name} -> {new_tsv.name} ({len(events_df)} rows)")
        if old_wf:
            print(f"            {old_wf.name} -> {new_wf.name}")
        if old_npz:
            print(f"            {old_npz.name} -> {new_npz.name}")
        return True

    events_df.to_csv(new_tsv, sep="\t", index=False, float_format="%.6f")
    with open(new_json, "w") as f:
        json.dump(events_meta, f, indent=2)

    if old_wf and old_wf != new_wf:
        if new_wf.exists():
            new_wf.unlink()
        shutil.move(str(old_wf), str(new_wf))
    elif old_wf is None and new_wf.exists():
        pass

    if old_npz and old_npz != new_npz:
        if new_npz.exists():
            new_npz.unlink()
        shutil.move(str(old_npz), str(new_npz))
    elif old_npz is None and new_npz.exists():
        pass

    # Patch NPZ with channel string if present (cheap)
    if new_npz.is_file():
        z = np.load(new_npz, allow_pickle=True)
        save_kw = {k: z[k] for k in z.files}
        z.close()
        save_kw["channel"] = np.array(desc)
        np.savez_compressed(new_npz, **save_kw)

    if delete_old:
        for p in (old_tsv, old_json_path):
            if p and p.is_file() and p != new_tsv and p != new_json:
                p.unlink()
        # Remove old-prefix wf/npz if rename did not happen (same path)
        for legacy in ieeg_dir.glob(f"*_desc-{desc}_spikes.*"):
            if legacy.is_file():
                legacy.unlink()

    print(f"  aligned {desc} -> {new_tsv.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align spike-sorted derivatives to BIDS-style events filenames and metadata."
    )
    parser.add_argument(
        "--deriv-root",
        type=Path,
        default=Path(
            "/store/scratch/bsow/Documents/UCLA_24/data/bids/derivatives/spike-sorted"
        ),
    )
    parser.add_argument("--subject", default="572")
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="movie24presleep")
    parser.add_argument("--acq", default="micro")
    parser.add_argument("--run", default="01")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned renames only; do not write files.",
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help="Remove legacy *_spikes.tsv/json after successful alignment.",
    )
    args = parser.parse_args()

    ieeg_dir = args.deriv_root / f"sub-{args.subject}" / f"ses-{args.session}" / "ieeg"
    if not ieeg_dir.is_dir():
        raise FileNotFoundError(f"Missing derivatives ieeg dir: {ieeg_dir}")

    descs = sorted(collect_channel_descs(ieeg_dir))
    if not descs:
        raise FileNotFoundError(f"No desc-* channel files found under {ieeg_dir}")

    print(f"Found {len(descs)} channels under {ieeg_dir}")
    n_ok = 0
    for desc in descs:
        if align_channel(
            ieeg_dir,
            desc,
            args.subject,
            args.session,
            args.task,
            args.acq,
            args.run,
            dry_run=args.dry_run,
            delete_old=args.delete_old,
        ):
            n_ok += 1

    update_dataset_description(args.deriv_root, dry_run=args.dry_run)
    print(f"Done. Aligned {n_ok} channel(s).")
    if args.dry_run:
        print("Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
