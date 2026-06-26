"""Shared config and name mappings for the Zhang 2023 vision pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_characters_config(path: Path | None = None) -> dict:
    path = path or (CONFIG_DIR / "characters.yaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def csv_name_to_char_col(name: str) -> str:
    """A.Fayed → char_a_fayed ; C.OBrian → char_c_obrian."""
    initial, last = name.split(".", 1)
    return f"char_{initial.lower()}_{last.lower()}"


def char_col_to_csv_name(col: str) -> str:
    """char_a_fayed → A.Fayed."""
    body = col.removeprefix("char_")
    initial, last = body.split("_", 1)
    if last == "obrian":
        last = "OBrian"
    else:
        last = last.capitalize()
    return f"{initial.upper()}.{last}"


def load_cluster_assignments(path: Path) -> dict[int, str]:
    with open(path) as fh:
        raw = json.load(fh)
    return {int(k): v for k, v in raw.items() if not str(k).startswith("_")}
