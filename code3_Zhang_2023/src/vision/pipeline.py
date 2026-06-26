"""End-to-end Zhang 2023 vision pipeline (strict)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zhang2023_constants import FRAME_SUBSAMPLE

from .config import load_characters_config
from .export import export_frame_labels
from .stage1_detect_track import Stage1Config, load_stage1_tracks, run_stage1
from .stage2_cluster import Stage2Config, run_stage2
from .stage3_assign import run_stage3
from .stage4_resnet import Stage4Config, run_stage4


@dataclass
class VisionPipelineConfig:
    video: Path
    work_dir: Path
    characters_yaml: Path | None = None
    cluster_assignments: Path | None = None
    output_csv: Path | None = None
    stage1: Stage1Config | None = None
    stage2: Stage2Config | None = None
    stage4: Stage4Config | None = None
    bootstrap_from_reference: Path | None = None  # non-paper fallback only


def run_vision_pipeline(cfg: VisionPipelineConfig) -> Path:
    char_cfg = load_characters_config(cfg.characters_yaml)
    major = list(char_cfg["major_characters"])
    vision_classes = major + ["Other"]
    work = cfg.work_dir
    work.mkdir(parents=True, exist_ok=True)

    stage1_cfg = cfg.stage1 or Stage1Config()
    stage1_cfg.frame_step = FRAME_SUBSAMPLE
    tracks_npz = run_stage1(cfg.video, work, stage1_cfg)
    tracks = load_stage1_tracks(tracks_npz)
    if not tracks:
        raise RuntimeError(
            "Stage 1 produced zero tracks. Check YOLOv3 weights and video path."
        )

    stage2_cfg = cfg.stage2 or Stage2Config(device=stage1_cfg.device)
    stage2_json = run_stage2(tracks, work, stage2_cfg)

    assign_path = cfg.cluster_assignments or (work / "cluster_assignments.json")
    stage3_json = run_stage3(
        tracks_npz,
        stage2_json,
        assign_path,
        work,
        reference_csv=cfg.bootstrap_from_reference,
        characters=major,
    )

    out_csv = cfg.output_csv or (work / "characters_30fps.csv")
    stage4_cfg = cfg.stage4 or Stage4Config(device=stage1_cfg.device)
    stage4_npz = run_stage4(
        cfg.video, tracks_npz, stage3_json, vision_classes, work, stage4_cfg
    )
    return export_frame_labels(
        stage4_npz, cfg.video, major, out_csv, stage4_cfg.prob_thresh, frame_step=FRAME_SUBSAMPLE
    )
