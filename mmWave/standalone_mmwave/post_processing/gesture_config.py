"""Load gesture / heatmap processing settings (JSON, not hardcoded range)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

_DEFAULT = Path(__file__).resolve().parents[1] / "config" / "gesture_processing.json"


@dataclass
class BackgroundConfig:
    capture: Optional[Path]
    use_start_frames: int
    profile_drop_db: float
    margin_m: float
    min_above_median_db: float


@dataclass
class GestureProcessingConfig:
    snr_threshold_db: float
    clutter_window_frames: int
    range_min_m: float
    range_max_m: Optional[float]
    range_max_m_cap: Optional[float]
    background: BackgroundConfig

    @classmethod
    def load(cls, path: Path | None = None) -> "GestureProcessingConfig":
        path = Path(path) if path else _DEFAULT
        data = json.loads(path.read_text())
        bg = data.get("background", {})
        rg = data.get("range_gate", {})
        return cls(
            snr_threshold_db=float(data.get("snr_threshold_db", 8.0)),
            clutter_window_frames=int(data.get("clutter_window_frames", 16)),
            range_min_m=float(rg.get("min_m", 0.5)),
            range_max_m=_opt_float(rg.get("max_m")),
            range_max_m_cap=_opt_float(rg.get("max_m_cap")),
            background=BackgroundConfig(
                capture=Path(bg["capture"]) if bg.get("capture") else None,
                use_start_frames=int(bg.get("use_start_frames", 45)),
                profile_drop_db=float(bg.get("profile_drop_db", 12.0)),
                margin_m=float(bg.get("margin_m", 0.3)),
                min_above_median_db=float(bg.get("min_above_median_db", 6.0)),
            ),
        )

    def save_applied(self, out_path: Path, applied: dict) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_source": str(_DEFAULT),
            "applied": applied,
        }
        out_path.write_text(json.dumps(payload, indent=2))


def _opt_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    return float(v)


def resolve_range_gate(
    cfg: GestureProcessingConfig,
    *,
    estimated_max_m: float,
) -> Tuple[float, float]:
    """Return (min_m, max_m) after manual override, auto estimate, and optional cap."""
    r_min = cfg.range_min_m
    if cfg.range_max_m is not None:
        r_max = cfg.range_max_m
    else:
        r_max = estimated_max_m
    if cfg.range_max_m_cap is not None:
        r_max = min(r_max, cfg.range_max_m_cap)
    if r_max <= r_min:
        r_max = r_min + 0.5
    return r_min, r_max
