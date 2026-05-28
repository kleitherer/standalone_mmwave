"""Load heatmap processing settings with live config as default."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

_DEFAULT = Path(__file__).resolve().parents[1] / "config" / "live_radar_to_max.json"


@dataclass
class BackgroundConfig:
    capture: Optional[Path]
    max_frames: int
    use_start_frames: int
    profile_drop_db: float
    margin_m: float
    min_above_median_db: float


@dataclass
class AngleEstimationConfig:
    method: str  # "fft" or "music"
    fft_bins: int
    fov_deg: float


@dataclass
class ProcessingConfig:
    snr_threshold_db: float
    calibration_frames: int
    range_min_m: float
    range_max_m: Optional[float]
    range_max_m_cap: Optional[float]
    angle: AngleEstimationConfig
    background: BackgroundConfig

    @classmethod
    def load(cls, path: Path | None = None) -> "ProcessingConfig":
        path = Path(path) if path else _DEFAULT
        data = json.loads(path.read_text())
        # Prefer live_radar_to_max.json schema; keep legacy gesture_processing fallback.
        rg = data.get("range_gate", {})
        ang = data.get("angle_estimation", data.get("angle", {}))
        if "processing" in data:
            p = data.get("processing", {})
            bg = data.get("background", {})
            range_min_m = float(p.get("roi_min_m", 0.5))
            range_max_m = _opt_float(p.get("roi_max_m"))
            calibration_frames = int(p.get("calibration_frames", p.get("clutter_window", 45)))
            snr_threshold_db = float(p.get("presence_threshold_db", 12.0))
            if not ang and "angle_method" in p:
                ang = {
                    "method": p.get("angle_method", "fft"),
                    "fft_bins": p.get("angle_fft_bins", 128),
                    "fov_deg": p.get("angle_fov_deg", 90),
                }
        else:
            bg = data.get("background", {})
            rg = data.get("range_gate", {})
            range_min_m = float(rg.get("min_m", 0.5))
            range_max_m = _opt_float(rg.get("max_m"))
            calibration_frames = int(data.get("calibration_frames", data.get("clutter_window_frames", 16)))
            snr_threshold_db = float(data.get("snr_threshold_db", 8.0))

        method = str(ang.get("method", "fft")).lower()
        if method not in {"fft", "music"}:
            raise ValueError(f"angle_estimation.method must be 'fft' or 'music', got {method!r}")

        return cls(
            snr_threshold_db=snr_threshold_db,
            calibration_frames=calibration_frames,
            range_min_m=range_min_m,
            range_max_m=range_max_m,
            range_max_m_cap=_opt_float(rg.get("max_m_cap")),
            angle=AngleEstimationConfig(
                method=method,
                fft_bins=int(ang.get("fft_bins", ang.get("angle_fft_bins", 128))),
                fov_deg=float(ang.get("fov_deg", ang.get("angle_fov_deg", 90))),
            ),
            background=BackgroundConfig(
                capture=Path(bg["capture"]) if bg.get("capture") else None,
                max_frames=int(bg.get("max_frames", 0)),
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
    cfg: ProcessingConfig,
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
