"""Estimate max range from background clutter (per room / setup)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from post_processing.gesture_config import BackgroundConfig, GestureProcessingConfig
from post_processing.rd_maps import build_range_time_volume


def range_profile_from_volume(snr_rt: np.ndarray) -> np.ndarray:
    """Median SNR vs range over time: shape (n_range,)."""
    return np.median(snr_rt, axis=0).astype(np.float64)


def estimate_max_range_m(
    range_m: np.ndarray,
    profile: np.ndarray,
    bg: BackgroundConfig,
) -> Tuple[float, dict]:
    """
    Furthest range bin that belongs to the static scene, minus margin.

    Uses median range profile from background data. The outer edge of strong
  static returns (walls, furniture) sets the limit; gesture processing ignores
    ranges beyond that (not a hardcoded 6 m).
    """
    baseline = float(np.median(profile))
    active = profile >= (baseline + bg.min_above_median_db)
    info = {
        "baseline_db": baseline,
        "range_m_full": range_m.tolist(),
        "profile_median_db": profile.tolist(),
    }

    if not np.any(active):
        r_max = float(range_m[-1])
        info["method"] = "fallback_full_span"
        info["range_max_m"] = r_max - bg.margin_m
        return max(range_m[0], r_max - bg.margin_m), info

    idx = np.where(active)[0]
    sub_p = profile[idx]
    sub_r = range_m[idx]
    peak = float(np.max(sub_p))
    thresh = peak - bg.profile_drop_db
    above = sub_p >= thresh
    if not np.any(above):
        r_edge = float(sub_r[-1])
    else:
        r_edge = float(sub_r[np.where(above)[0][-1]])

    r_max = max(float(range_m[0]), r_edge - bg.margin_m)
    info.update(
        {
            "method": "background_profile_edge",
            "profile_peak_db": peak,
            "profile_thresh_db": thresh,
            "range_edge_m": r_edge,
            "margin_m": bg.margin_m,
            "range_max_m": r_max,
        }
    )
    return r_max, info


def estimate_range_gate_for_capture(
    gesture_capture: Path,
    proc_cfg: GestureProcessingConfig,
    *,
    radar_max_range_m: float,
    show_progress: bool = False,
) -> Tuple[float, float, dict]:
    """
    Build background profile and return (range_min_m, range_max_m, debug_info).
    """
    bg = proc_cfg.background
    bg_path = bg.capture if bg.capture else gesture_capture

    n_bg = bg.use_start_frames
    if bg.capture:
        print(f"  background from: {bg_path}")
    else:
        print(f"  background from first {n_bg} frames of gesture capture")

    vol = build_range_time_volume(
        bg_path,
        range_gate_m=(proc_cfg.range_min_m, radar_max_range_m),
        clutter_window=proc_cfg.clutter_window_frames,
        snr_threshold_db=proc_cfg.snr_threshold_db,
        max_frames=n_bg if not bg.capture else 0,
        show_progress=show_progress,
    )

    profile = range_profile_from_volume(vol.snr_db)
    est_max, est_info = estimate_max_range_m(vol.range_m, profile, bg)
    from post_processing.gesture_config import resolve_range_gate

    r_min, r_max = resolve_range_gate(proc_cfg, estimated_max_m=est_max)

    info = {
        "background_source": str(Path(bg_path).resolve()),
        "background_frames_used": vol.snr_db.shape[0],
        "estimate": est_info,
        "range_gate_m": [r_min, r_max],
    }
    return r_min, r_max, info
