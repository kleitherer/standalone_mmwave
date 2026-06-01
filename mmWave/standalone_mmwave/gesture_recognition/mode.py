"""Gesture mode selection: config1 (2-ID tracking) vs config2 (simple velocity)."""

from __future__ import annotations

from typing import Any, Literal, Union

import numpy as np

from gesture_recognition.simple import SimpleGestureConfig, SimpleGestureProcessor
from gesture_recognition.tracker import RangePeakTracker, TrackingConfig, TrackingVolume

GestureModeName = Literal["config1", "config2"]
GestureProcessor = Union[RangePeakTracker, SimpleGestureProcessor]


def gesture_mode_from_settings(settings: dict[str, Any]) -> GestureModeName:
    gesture = settings.get("gesture", {})
    mode = str(gesture.get("mode", "config1")).strip().lower()
    if mode in ("config2", "simple", "velocity"):
        return "config2"
    if mode in ("config1", "tracking", "dual"):
        return "config1"
    # Legacy: no mode key but only tracking block → config1
    if "config1" in gesture or "tracking" in gesture:
        return "config1"
    return "config1"


def config1_enabled(settings: dict[str, Any]) -> bool:
    """When false, live/replay/plot use config2 even if mode is config1."""
    gesture = settings.get("gesture", {})
    block = gesture.get("config1", gesture.get("tracking", {}))
    if not isinstance(block, dict):
        return True
    return bool(block.get("enabled", True))


def active_gesture_mode(settings: dict[str, Any]) -> GestureModeName:
    mode = gesture_mode_from_settings(settings)
    if mode == "config1" and not config1_enabled(settings):
        return "config2"
    return mode


def make_gesture_processor(settings: dict[str, Any]) -> GestureProcessor:
    if active_gesture_mode(settings) == "config2":
        return SimpleGestureProcessor(SimpleGestureConfig.from_settings(settings))
    return RangePeakTracker(TrackingConfig.from_settings(settings))


def make_osc_gesture_limiter(settings: dict[str, Any]):
    from gesture_recognition.gesture import OscGestureRateLimiter

    return OscGestureRateLimiter.from_settings(settings)


def gesture_mode_label(settings: dict[str, Any]) -> str:
    return active_gesture_mode(settings)


def describe_gesture_mode(settings: dict[str, Any]) -> str:
    mode = active_gesture_mode(settings)
    if mode == "config2":
        cfg = SimpleGestureConfig.from_settings(settings)
        return (
            f"Gesture config2: closest-range peak, 2-point velocity, "
            f"threshold {cfg.min_velocity_mps:.2f} m/s, "
            f"confirm {cfg.gesture_confirm_frames} frames"
        )
    cfg = TrackingConfig.from_settings(settings)
    return (
        f"Gesture config1: NN gate {cfg.associate_gate_m:.2f} m, "
        f"jump ≤ {cfg.max_range_jump_m:.2f} m/frame, "
        f"≤{cfg.max_tracks} tracks, history {cfg.history_frames} frames"
    )


def simulate_gesture_volume(
    snr_db,
    range_m,
    peak_cfg,
    settings: dict[str, Any],
    dt_s: float,
):
    """Replay active gesture mode over a range–time SNR volume."""
    from gesture_recognition.simple import simulate_simple_volume
    from gesture_recognition.tracker import simulate_tracking_volume

    if active_gesture_mode(settings) == "config2":
        vol = simulate_simple_volume(
            snr_db,
            range_m,
            peak_cfg,
            SimpleGestureConfig.from_settings(settings),
            dt_s,
        )
    else:
        track_cfg = TrackingConfig.from_settings(settings)
        vol = simulate_tracking_volume(
            snr_db,
            range_m,
            peak_cfg,
            track_cfg,
            dt_s,
        )
    from gesture_recognition.gesture import apply_osc_gesture_rate_limit

    body_v = getattr(vol, "track1_velocity_mps", None)
    body_max_v = 0.0
    if active_gesture_mode(settings) == "config1":
        body_max_v = TrackingConfig.from_settings(settings).body_max_velocity_for_gesture_mps

    osc_gest = apply_osc_gesture_rate_limit(
        vol.gesture,
        dt_s,
        body_velocity_mps=body_v,
        body_max_velocity_for_gesture_mps=body_max_v,
    )
    osc_vel = np.zeros_like(vol.velocity_mps)
    for fi, g in enumerate(osc_gest):
        osc_vel[fi] = vol.velocity_mps[fi] if g != "none" else 0.0
    return TrackingVolume(
        track1_range_m=vol.track1_range_m,
        track1_snr_db=vol.track1_snr_db,
        track1_gesture=vol.track1_gesture,
        track1_velocity_mps=getattr(vol, "track1_velocity_mps", np.zeros_like(vol.velocity_mps)),
        track2_range_m=vol.track2_range_m,
        track2_snr_db=vol.track2_snr_db,
        track2_gesture=vol.track2_gesture,
        gesture=osc_gest,
        raw_gesture=vol.raw_gesture,
        velocity_mps=osc_vel,
    )
