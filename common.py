# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(r"D:\Positioning")
CONFIG_FILE = ROOT / "config.yaml"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_config() -> dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def find_image(directory: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def class_to_id_map(cfg: dict) -> dict[str, int]:
    return {name: i for i, name in enumerate(cfg["classes"])}
