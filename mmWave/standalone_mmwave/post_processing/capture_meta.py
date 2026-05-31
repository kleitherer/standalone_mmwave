"""Shared capture metadata helpers for post-processing."""

from __future__ import annotations

import json
from pathlib import Path


def roi_from_metadata(meta: dict) -> tuple[float | None, float | None]:
    proc = meta.get("processing") or {}
    settings = meta.get("settings_file")
    if proc.get("roi_min_m") is None and settings:
        try:
            cfg = json.loads(Path(settings).read_text())
            proc = cfg.get("processing") or proc
        except (OSError, json.JSONDecodeError):
            pass
    lo = proc.get("roi_min_m")
    hi = proc.get("roi_max_m")
    return (float(lo) if lo is not None else None, float(hi) if hi is not None else None)
