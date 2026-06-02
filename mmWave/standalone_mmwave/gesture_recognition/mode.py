"""Gesture mode selection: config1/config2/config3/config4/config5/config6 or off."""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING, Union

import numpy as np

from gesture_recognition.simple import SimpleGestureConfig, SimpleGestureProcessor
from gesture_recognition.tracker import RangePeakTracker, TrackingConfig, TrackingVolume

if TYPE_CHECKING:
    from gesture_recognition.rd_peak import RdKalmanProcessor, RdPeakProcessor

GestureModeName = Literal["config1", "config2", "config3", "config4", "config5", "config6", "off"]
GestureProcessor = Union[
    RangePeakTracker,
    SimpleGestureProcessor,
    "RdPeakProcessor",
    "RdKalmanProcessor",
    None,
]


def _raw_mode_string(settings: dict[str, Any]) -> str:
    return str(settings.get("gesture", {}).get("mode", "config1")).strip().lower()


def gesture_enabled(settings: dict[str, Any]) -> bool:
    """Top-level ``gesture.enabled`` (default true)."""
    return bool(settings.get("gesture", {}).get("enabled", True))


def config1_enabled(settings: dict[str, Any]) -> bool:
    """When false and mode is config1, fall back to config2."""
    gesture = settings.get("gesture", {})
    block = gesture.get("config1", gesture.get("tracking", {}))
    if not isinstance(block, dict):
        return True
    return bool(block.get("enabled", True))


def active_gesture_mode(settings: dict[str, Any]) -> GestureModeName:
    """
    Resolved mode for live / replay / plots.

    - ``off``: no tracking, no gestures (strongest peak → range / SNR / presence only)
    - ``config1``: 2-ID NN tracker + hand gestures
    - ``config2``: single closest peak + velocity gestures
    - ``config3``: RD-map peak → range_m + doppler_mps (no tracking / gestures)
    - ``config4``: config3 + continuity rule (limit range jumps)
    - ``config5``: Kalman-smoothed RD peaks (range + doppler + angle)
    - ``config6``: config4 + Doppler continuity (limit velocity jumps)
    """
    mode = _raw_mode_string(settings)
    if mode in ("config3", "rd", "rd_peak"):
        return "config3"
    if mode in ("config4", "rd_continuity", "rd_jump"):
        return "config4"
    if mode in ("config5", "rd_kalman", "kalman"):
        return "config5"
    if mode in ("config6", "rd_continuity_v", "rd_jump_v"):
        return "config6"

    if not gesture_enabled(settings):
        return "off"

    if mode in ("off", "none", "disabled", "false"):
        return "off"
    if mode in ("config2", "simple", "velocity"):
        return "config2"
    if mode in ("config1", "tracking", "dual"):
        if not config1_enabled(settings):
            return "config2"
        return "config1"
    if "config1" in settings.get("gesture", {}) or "tracking" in settings.get("gesture", {}):
        if not config1_enabled(settings):
            return "config2"
        return "config1"
    return "config1"


def make_gesture_processor(settings: dict[str, Any]) -> GestureProcessor:
    mode = active_gesture_mode(settings)
    if mode == "off":
        return None
    if mode == "config3":
        from gesture_recognition.rd_peak import RdPeakConfig, RdPeakProcessor

        return RdPeakProcessor(RdPeakConfig.from_settings(settings))
    if mode == "config4":
        from gesture_recognition.rd_peak import RdPeakConfig, RdPeakProcessor

        cfg = RdPeakConfig.from_settings(settings)
        block = settings.get("gesture", {}).get("config4", {})
        if isinstance(block, dict):
            cfg = RdPeakConfig(
                min_range_m=float(block.get("min_range_m", cfg.min_range_m)),
                max_range_m=float(block.get("max_range_m", cfg.max_range_m)),
                power_threshold_db=float(block.get("power_threshold_db", cfg.power_threshold_db)),
                max_abs_doppler_mps=float(block.get("max_abs_doppler_mps", cfg.max_abs_doppler_mps)),
                max_abs_angle_deg=float(block.get("max_abs_angle_deg", cfg.max_abs_angle_deg)),
                max_range_jump_m=float(block.get("max_range_jump_m", 1.0)),
                max_candidates=max(1, int(block.get("max_candidates", cfg.max_candidates))),
                declutter=bool(block.get("declutter", cfg.declutter)),
                window=bool(block.get("window", cfg.window)),
                n_angle_fft=max(2, int(block.get("n_angle_fft", cfg.n_angle_fft))),
            )
        return RdPeakProcessor(cfg)
    if mode == "config5":
        from gesture_recognition.rd_peak import RdKalmanConfig, RdKalmanProcessor

        return RdKalmanProcessor(RdKalmanConfig.from_settings(settings))
    if mode == "config6":
        from gesture_recognition.rd_peak import RdPeakConfig, RdPeakProcessor

        cfg = RdPeakConfig.from_settings(settings)
        block = settings.get("gesture", {}).get("config6", {})
        if isinstance(block, dict):
            cfg = RdPeakConfig(
                min_range_m=float(block.get("min_range_m", cfg.min_range_m)),
                max_range_m=float(block.get("max_range_m", cfg.max_range_m)),
                power_threshold_db=float(block.get("power_threshold_db", cfg.power_threshold_db)),
                max_abs_doppler_mps=float(block.get("max_abs_doppler_mps", cfg.max_abs_doppler_mps)),
                max_doppler_jump_mps=float(block.get("max_doppler_jump_mps", 0.5)),
                max_abs_angle_deg=float(block.get("max_abs_angle_deg", cfg.max_abs_angle_deg)),
                max_range_jump_m=float(block.get("max_range_jump_m", 0.3)),
                max_candidates=max(1, int(block.get("max_candidates", cfg.max_candidates))),
                declutter=bool(block.get("declutter", cfg.declutter)),
                window=bool(block.get("window", cfg.window)),
                n_angle_fft=max(2, int(block.get("n_angle_fft", cfg.n_angle_fft))),
            )
        return RdPeakProcessor(cfg)
    if mode == "config2":
        return SimpleGestureProcessor(SimpleGestureConfig.from_settings(settings))
    return RangePeakTracker(TrackingConfig.from_settings(settings))


def make_osc_gesture_limiter(settings: dict[str, Any]):
    from gesture_recognition.gesture import OscGestureRateLimiter

    return OscGestureRateLimiter.from_settings(settings)


def gesture_mode_label(settings: dict[str, Any]) -> str:
    return active_gesture_mode(settings)


def describe_gesture_mode(settings: dict[str, Any]) -> str:
    mode = active_gesture_mode(settings)
    if mode == "off":
        return "Gesture off: strongest range peak → range_m / snr_db / presence (no tracks, no gesture OSC)"
    if mode == "config3":
        from gesture_recognition.rd_peak import RdPeakConfig

        cfg = RdPeakConfig.from_settings(settings)
        return (
            f"Gesture config3: RD-map peak → range_m + doppler_mps + angle "
            f"(ULA FFT n={cfg.n_angle_fft}, |angle|<={cfg.max_abs_angle_deg:.0f}°, r <= {cfg.max_range_m:.2f} m, "
            f"power thr {cfg.power_threshold_db:.1f} dB)"
        )
    if mode == "config4":
        from gesture_recognition.rd_peak import RdPeakConfig

        base = RdPeakConfig.from_settings(settings)
        c4 = settings.get("gesture", {}).get("config4", {})
        max_jump = float(c4.get("max_range_jump_m", 1.0)) if isinstance(c4, dict) else 1.0
        return (
            f"Gesture config4: RD-map peak continuity (jump<={max_jump:.2f}m), "
            f"|angle|<={base.max_abs_angle_deg:.0f}°, |doppler|<={base.max_abs_doppler_mps:.1f}m/s"
        )
    if mode == "config5":
        return "Gesture config5: Kalman filter on RD peaks (range, doppler, angle)"
    if mode == "config6":
        from gesture_recognition.rd_peak import RdPeakConfig

        base = RdPeakConfig.from_settings(settings)
        c6 = settings.get("gesture", {}).get("config6", {})
        range_jump = float(c6.get("max_range_jump_m", 0.3)) if isinstance(c6, dict) else 0.3
        dop_jump = float(c6.get("max_doppler_jump_mps", 0.5)) if isinstance(c6, dict) else 0.5
        return (
            f"Gesture config6: RD continuity (Δr<={range_jump:.2f}m, "
            f"Δv<={dop_jump:.2f}m/s), |angle|<={base.max_abs_angle_deg:.0f}°"
        )
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
    if active_gesture_mode(settings) == "off":
        return None
    if active_gesture_mode(settings) in ("config3", "config4", "config5", "config6"):
        return None

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
