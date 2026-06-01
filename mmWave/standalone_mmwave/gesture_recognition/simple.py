"""Config 2: single-peak velocity gesture (no multi-ID tracking)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gesture_recognition.gesture import (
    GESTURE_NONE,
    classify_gesture_from_velocity,
    confirm_gesture_label,
)
from gesture_recognition.peaks import RangePeakDetectionConfig
from gesture_recognition.tracker import TrackingVolume


@dataclass(frozen=True)
class SimpleGestureConfig:
    """``gesture.config2`` in config/live_radar_to_max.json."""

    min_velocity_mps: float = 0.15
    gesture_confirm_frames: int = 3

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> SimpleGestureConfig:
        gesture = settings.get("gesture", {})
        block = gesture.get("config2", {})
        if not isinstance(block, dict):
            block = {}
        return cls(
            min_velocity_mps=float(
                block.get(
                    "min_velocity_mps",
                    gesture.get("min_velocity_mps", 0.15),
                )
            ),
            gesture_confirm_frames=max(
                2, int(block.get("gesture_confirm_frames", 3))
            ),
        )


@dataclass
class SimpleGestureProcessor:
    """
    Closest range peak per frame; velocity from the last two samples.

    Peaks are SNR-sorted in the peak list — config2 picks **minimum range**
    (nearest reflector) so hand motion dominates push/pull, not the torso.
    """

    cfg: SimpleGestureConfig
    _last_range_m: float | None = field(default=None, init=False)
    _range_m: float = field(default=0.0, init=False)
    _snr_db: float = field(default=0.0, init=False)
    _velocity_mps: float = field(default=0.0, init=False)
    _gesture: str = field(default=GESTURE_NONE, init=False)
    _raw_gesture: str = field(default=GESTURE_NONE, init=False)
    _gesture_pending: str = field(default=GESTURE_NONE, init=False)
    _gesture_streak: int = field(default=0, init=False)

    def reset(self) -> None:
        self._last_range_m = None
        self._range_m = 0.0
        self._snr_db = 0.0
        self._velocity_mps = 0.0
        self._gesture = GESTURE_NONE
        self._raw_gesture = GESTURE_NONE
        self._gesture_pending = GESTURE_NONE
        self._gesture_streak = 0

    def update(
        self,
        peaks: list[tuple[float, float]],
        dt_s: float,
        peak_cfg: RangePeakDetectionConfig,
    ) -> None:
        dt_s = max(float(dt_s), 1e-6)
        target = _closest_range_peak(peaks, peak_cfg)

        if target is None:
            self._velocity_mps = 0.0
            self._gesture = GESTURE_NONE
            self._raw_gesture = GESTURE_NONE
            self._gesture_pending = GESTURE_NONE
            self._gesture_streak = 0
            return

        r_meas, snr_meas = target
        velocity = 0.0
        if self._last_range_m is not None:
            velocity = (float(r_meas) - float(self._last_range_m)) / dt_s

        self._last_range_m = float(r_meas)
        self._range_m = float(r_meas)
        self._snr_db = float(snr_meas)
        self._velocity_mps = velocity

        raw = classify_gesture_from_velocity(
            velocity,
            min_velocity_mps=self.cfg.min_velocity_mps,
        )
        self._raw_gesture = raw
        confirmed, self._gesture_pending, self._gesture_streak = confirm_gesture_label(
            raw,
            self._gesture_pending,
            self._gesture_streak,
            self.cfg.gesture_confirm_frames,
        )
        self._gesture = confirmed

    @property
    def range_m(self) -> float:
        return self._range_m

    @property
    def snr_db(self) -> float:
        return self._snr_db

    @property
    def velocity_mps(self) -> float:
        if self._gesture == GESTURE_NONE:
            return 0.0
        return self._velocity_mps

    @property
    def raw_gesture(self) -> str:
        return self._raw_gesture

    @property
    def gesture(self) -> str:
        return self._gesture


def _closest_range_peak(
    peaks: list[tuple[float, float]],
    peak_cfg: RangePeakDetectionConfig,
) -> tuple[float, float] | None:
    """Nearest reflector above SNR threshold (smallest range_m)."""
    best: tuple[float, float] | None = None
    for r_meas, snr_meas in peaks:
        if float(snr_meas) < peak_cfg.snr_threshold_db:
            continue
        r = float(r_meas)
        if best is None or r < best[0]:
            best = (r, float(snr_meas))
    return best


def simulate_simple_volume(
    snr_db: np.ndarray,
    range_m: np.ndarray,
    peak_cfg: RangePeakDetectionConfig,
    simple_cfg: SimpleGestureConfig,
    dt_s: float,
) -> TrackingVolume:
    """Replay config-2 processor over a range–time SNR volume."""
    from gesture_recognition.peaks import peaks_from_profile

    n_frames = int(snr_db.shape[0])
    t1r = np.full(n_frames, np.nan, dtype=np.float64)
    t1s = np.full(n_frames, np.nan, dtype=np.float64)
    t1g = np.full(n_frames, GESTURE_NONE, dtype=object)
    t2r = np.full(n_frames, np.nan, dtype=np.float64)
    t2s = np.full(n_frames, np.nan, dtype=np.float64)
    t2g = np.full(n_frames, GESTURE_NONE, dtype=object)
    gest = np.full(n_frames, GESTURE_NONE, dtype=object)
    raw = np.full(n_frames, GESTURE_NONE, dtype=object)
    vel = np.zeros(n_frames, dtype=np.float64)

    proc = SimpleGestureProcessor(simple_cfg)
    dt_s = max(float(dt_s), 1e-6)
    for fi in range(n_frames):
        peaks = peaks_from_profile(snr_db[fi], range_m, peak_cfg)
        proc.update(peaks, dt_s, peak_cfg)
        if proc.snr_db >= peak_cfg.snr_threshold_db:
            t1r[fi] = proc.range_m
            t1s[fi] = proc.snr_db
            t1g[fi] = proc.gesture
        gest[fi] = proc.gesture
        raw[fi] = proc.raw_gesture
        vel[fi] = proc.velocity_mps

    return TrackingVolume(
        track1_range_m=t1r,
        track1_snr_db=t1s,
        track1_gesture=t1g,
        track1_velocity_mps=np.zeros(n_frames, dtype=np.float64),
        track2_range_m=t2r,
        track2_snr_db=t2s,
        track2_gesture=t2g,
        gesture=gest,
        raw_gesture=raw,
        velocity_mps=vel,
    )
