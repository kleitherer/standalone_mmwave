"""Lightweight nearest-neighbor range peak tracker (1–2 tracks, real-time)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gesture_recognition.gesture import (
    GESTURE_NONE,
    classify_gesture_from_velocity,
    confirm_gesture_label,
)
from gesture_recognition.peaks import RangePeakDetectionConfig


@dataclass(frozen=True)
class TrackingConfig:
    """``gesture.config1`` in config/live_radar_to_max.json."""

    max_tracks: int = 2
    associate_gate_m: float = 0.20
    max_range_jump_m: float = 0.50
    min_hand_ahead_m: float = 0.20
    max_hand_ahead_m: float = 0.70
    history_frames: int = 8
    max_missed_frames: int = 3
    min_velocity_mps: float = 0.15
    gesture_confirm_frames: int = 3
    body_slower_than_hand: bool = True
    body_hand_velocity_margin_mps: float = 0.05
    body_calibration_s: float = 5.0
    body_calibration_max_velocity_mps: float = 0.10
    body_anchor_blend: float = 0.5
    body_snr_margin_db: float = 6.0
    body_max_velocity_for_gesture_mps: float = 0.10
    hand_snr_threshold_db: float = 10.0

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> TrackingConfig:
        gesture = settings.get("gesture", {})
        block = gesture.get("config1", gesture.get("tracking", {}))
        if not isinstance(block, dict):
            block = {}
        return cls(
            max_tracks=max(1, int(block.get("max_tracks", 2))),
            associate_gate_m=float(block.get("associate_gate_m", 0.20)),
            max_range_jump_m=float(block.get("max_range_jump_m", 0.50)),
            min_hand_ahead_m=float(block.get("min_hand_ahead_m", 0.20)),
            max_hand_ahead_m=float(block.get("max_hand_ahead_m", 0.70)),
            history_frames=max(2, int(block.get("history_frames", 8))),
            max_missed_frames=max(1, int(block.get("max_missed_frames", 3))),
            min_velocity_mps=float(
                block.get("min_velocity_mps", gesture.get("min_velocity_mps", 0.15))
            ),
            gesture_confirm_frames=max(
                2, int(block.get("gesture_confirm_frames", 3))
            ),
            body_slower_than_hand=bool(block.get("body_slower_than_hand", True)),
            body_hand_velocity_margin_mps=float(
                block.get("body_hand_velocity_margin_mps", 0.05)
            ),
            body_calibration_s=float(block.get("body_calibration_s", 5.0)),
            body_calibration_max_velocity_mps=float(
                block.get("body_calibration_max_velocity_mps", 0.10)
            ),
            body_anchor_blend=float(block.get("body_anchor_blend", 0.5)),
            body_snr_margin_db=float(block.get("body_snr_margin_db", 6.0)),
            body_max_velocity_for_gesture_mps=float(
                block.get("body_max_velocity_for_gesture_mps", 0.10)
            ),
            hand_snr_threshold_db=float(block.get("hand_snr_threshold_db", 10.0)),
        )


@dataclass(frozen=True)
class BodyProfile:
    """Learned body signature from standing still at startup."""

    range_m: float
    snr_db: float
    range_spread_m: float
    snr_spread_db: float


@dataclass
class TrackedTarget:
    track_id: int
    range_m: float
    snr_db: float
    velocity_mps: float = 0.0
    gesture: str = GESTURE_NONE
    raw_gesture: str = GESTURE_NONE
    missed: int = 0
    persistent: bool = False
    _history: deque[float] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        if not self._history:
            self._history = deque([self.range_m], maxlen=256)


@dataclass
class RangePeakTracker:
    """
    Greedy nearest-neighbor association with temporal continuity.

    **Track id=1 (body)** is anchored by an optional still-standing calibration
    (``body_calibration_s``). Without calibration it bootstraps on the first
    valid peak. Id=1 never drops; it coasts when unmatched and does not emit
    push/pull.
    **Track id=2 (hands)** spawns on the first peak in the hand band in front
    of the body. Hands associate before the body; id=1 never exceeds id=2 speed.
    OSC push/pull requires ``gesture_confirm_frames`` consecutive frames.
    """

    cfg: TrackingConfig
    _tracks: list[TrackedTarget] = field(default_factory=list, init=False)
    _gesture_pending: str = field(default=GESTURE_NONE, init=False)
    _gesture_streak: int = field(default=0, init=False)
    _body_profile: BodyProfile | None = field(default=None, init=False)
    _calib_ranges: list[float] = field(default_factory=list, init=False)
    _calib_snrs: list[float] = field(default_factory=list, init=False)
    _calib_still_time_s: float = field(default=0.0, init=False)
    _last_calib_range: float | None = field(default=None, init=False)

    @property
    def body_calibration_active(self) -> bool:
        return self.cfg.body_calibration_s > 0 and self._body_profile is None

    @property
    def body_calibrated(self) -> bool:
        return self.cfg.body_calibration_s <= 0 or self._body_profile is not None

    @property
    def body_profile(self) -> BodyProfile | None:
        return self._body_profile

    @property
    def body_calib_still_s(self) -> float:
        return float(self._calib_still_time_s)

    def reset(self) -> None:
        self._tracks.clear()
        self._gesture_pending = GESTURE_NONE
        self._gesture_streak = 0
        self._body_profile = None
        self._calib_ranges.clear()
        self._calib_snrs.clear()
        self._calib_still_time_s = 0.0
        self._last_calib_range = None

    def _spawn_body_track(
        self,
        range_m: float,
        snr_db: float,
    ) -> None:
        if _track_by_id(self._tracks, 1) is not None:
            return
        hist: deque[float] = deque(maxlen=self.cfg.history_frames)
        for _ in range(min(2, self.cfg.history_frames)):
            hist.append(float(range_m))
        self._tracks.append(
            TrackedTarget(
                track_id=1,
                range_m=float(range_m),
                snr_db=float(snr_db),
                velocity_mps=0.0,
                gesture=GESTURE_NONE,
                persistent=True,
                _history=hist,
            )
        )

    def _ensure_body_track(
        self,
        measurements: list[tuple[float, float]],
        peak_cfg: RangePeakDetectionConfig,
    ) -> None:
        if _track_by_id(self._tracks, 1) is not None:
            return
        if self.body_calibration_active:
            return
        for r_meas, snr_meas in measurements:
            if float(snr_meas) < peak_cfg.body_snr_threshold_db:
                continue
            self._spawn_body_track(float(r_meas), float(snr_meas))
            return

    def _dominant_peak(
        self,
        measurements: list[tuple[float, float]],
        peak_cfg: RangePeakDetectionConfig,
    ) -> tuple[float, float] | None:
        for r_meas, snr_meas in measurements:
            if float(snr_meas) >= peak_cfg.body_snr_threshold_db:
                return float(r_meas), float(snr_meas)
        return None

    def _reset_body_calibration(self) -> None:
        self._calib_ranges.clear()
        self._calib_snrs.clear()
        self._calib_still_time_s = 0.0

    def _update_body_calibration(
        self,
        measurements: list[tuple[float, float]],
        dt_s: float,
        peak_cfg: RangePeakDetectionConfig,
    ) -> bool:
        """Collect still-standing samples; spawn id=1 when ``body_calibration_s`` reached."""
        if not self.body_calibration_active:
            return True

        peak = self._dominant_peak(measurements, peak_cfg)
        if peak is None:
            return False

        r_meas, snr_meas = peak
        if self._last_calib_range is not None:
            v = abs(r_meas - self._last_calib_range) / dt_s
            if v > self.cfg.body_calibration_max_velocity_mps:
                self._reset_body_calibration()
                self._last_calib_range = r_meas
                return False

        self._last_calib_range = r_meas
        self._calib_ranges.append(r_meas)
        self._calib_snrs.append(snr_meas)
        self._calib_still_time_s += dt_s

        if self._calib_still_time_s < self.cfg.body_calibration_s:
            return False

        ranges = np.asarray(self._calib_ranges, dtype=np.float64)
        snrs = np.asarray(self._calib_snrs, dtype=np.float64)
        self._body_profile = BodyProfile(
            range_m=float(np.median(ranges)),
            snr_db=float(np.median(snrs)),
            range_spread_m=float(np.std(ranges)),
            snr_spread_db=float(np.std(snrs)),
        )
        self._spawn_body_track(self._body_profile.range_m, self._body_profile.snr_db)
        return True

    def update(
        self,
        measurements: list[tuple[float, float]],
        dt_s: float,
        peak_cfg: RangePeakDetectionConfig,
        *,
        track_cfg: TrackingConfig | None = None,
    ) -> list[TrackedTarget]:
        """
        Associate peaks ``[(range_m, snr_db), ...]`` (SNR-sorted) to tracks.

        Id=1 bootstraps after optional still calibration (or first valid peak).
        Id=2 is optional and carries push/pull; id=1 always coasts with
        gesture=none.
        """
        dt_s = max(float(dt_s), 1e-6)
        hand_snr_thr = (
            float(track_cfg.hand_snr_threshold_db)
            if track_cfg is not None
            else peak_cfg.body_snr_threshold_db
        )
        if not self._update_body_calibration(measurements, dt_s, peak_cfg):
            return []
        self._ensure_body_track(measurements, peak_cfg)

        matched_meas: set[int] = set()
        matched_track_ids: set[int] = set()
        body = _track_by_id(self._tracks, 1)
        body_anchor = self._body_profile.range_m if self._body_profile else None
        body_min_snr = (
            self._body_profile.snr_db - self.cfg.body_snr_margin_db
            if self._body_profile
            else None
        )

        for track in sorted(self._tracks, key=lambda t: t.track_id, reverse=True):
            body_range = body.range_m if body is not None else None
            hand = _track_by_id(self._tracks, 2)
            hand_v_ceiling: float | None = None
            if (
                track.track_id == 1
                and hand is not None
                and self.cfg.body_slower_than_hand
            ):
                hand_v_ceiling = abs(float(hand.velocity_mps)) + float(
                    self.cfg.body_hand_velocity_margin_mps
                )
            best_j = _best_measurement_for_track(
                track,
                measurements,
                matched_meas,
                dt_s,
                self.cfg.associate_gate_m,
                self.cfg.max_range_jump_m,
                body_range_m=body_range,
                min_hand_ahead_m=self.cfg.min_hand_ahead_m,
                max_abs_velocity_mps=hand_v_ceiling,
                body_anchor_m=body_anchor if track.track_id == 1 else None,
                body_anchor_blend=self.cfg.body_anchor_blend,
                min_snr_db=(
                    body_min_snr
                    if track.track_id == 1 and body_min_snr is not None
                    else peak_cfg.body_snr_threshold_db
                    if track.track_id == 1
                    else hand_snr_thr
                ),
            )
            if best_j is None:
                continue
            r_meas, snr_meas = measurements[best_j]
            if track.track_id == 2 and body is not None:
                if not _hand_range_valid(
                    float(r_meas), body.range_m, self.cfg.min_hand_ahead_m
                ):
                    continue
            matched_meas.add(best_j)
            matched_track_ids.add(track.track_id)
            track._history.append(float(r_meas))
            while len(track._history) > self.cfg.history_frames:
                track._history.popleft()
            track.range_m = float(r_meas)
            track.snr_db = float(snr_meas)
            track.missed = 0
            track.velocity_mps = _velocity_from_history(track._history, dt_s)
            if track.track_id == 1:
                track.gesture = GESTURE_NONE
                track.raw_gesture = GESTURE_NONE
            else:
                raw = classify_gesture_from_velocity(
                    track.velocity_mps,
                    min_velocity_mps=self.cfg.min_velocity_mps,
                )
                track.raw_gesture = raw
                confirmed, self._gesture_pending, self._gesture_streak = (
                    confirm_gesture_label(
                        raw,
                        self._gesture_pending,
                        self._gesture_streak,
                        self.cfg.gesture_confirm_frames,
                    )
                )
                track.gesture = confirmed

        for track in self._tracks:
            if track.track_id not in matched_track_ids:
                track.missed += 1
                if track.track_id == 1:
                    track.gesture = GESTURE_NONE
                    track.raw_gesture = GESTURE_NONE
                elif track.track_id == 2:
                    track.gesture = GESTURE_NONE
                    track.raw_gesture = GESTURE_NONE
                    self._gesture_pending = GESTURE_NONE
                    self._gesture_streak = 0

        self._tracks = [
            t
            for t in self._tracks
            if t.persistent or t.missed <= self.cfg.max_missed_frames
        ]
        if _track_by_id(self._tracks, 2) is None:
            self._gesture_pending = GESTURE_NONE
            self._gesture_streak = 0

        body = _track_by_id(self._tracks, 1)
        if body is not None:
            hand = _track_by_id(self._tracks, 2)
            if hand is not None and not _hand_range_valid(
                hand.range_m, body.range_m, self.cfg.min_hand_ahead_m
            ):
                self._tracks = [t for t in self._tracks if t.track_id != 2]
                self._gesture_pending = GESTURE_NONE
                self._gesture_streak = 0

        if _track_by_id(self._tracks, 2) is None and body is not None:
            for j, (r_meas, snr_meas) in enumerate(measurements):
                if j in matched_meas:
                    continue
                if float(snr_meas) < hand_snr_thr:
                    continue
                r = float(r_meas)
                if not _hand_range_valid(r, body.range_m, self.cfg.min_hand_ahead_m):
                    continue
                if not _may_spawn_hand_track(
                    r,
                    body.range_m,
                    self.cfg.min_hand_ahead_m,
                    self.cfg.max_hand_ahead_m,
                ):
                    continue
                hist = deque(maxlen=self.cfg.history_frames)
                hist.append(float(r_meas))
                self._tracks.append(
                    TrackedTarget(
                        track_id=2,
                        range_m=float(r_meas),
                        snr_db=float(snr_meas),
                        velocity_mps=0.0,
                        gesture=GESTURE_NONE,
                        _history=hist,
                    )
                )
                break

        return sorted(self._tracks, key=lambda t: t.track_id)


def _last_range(track: TrackedTarget) -> float:
    return float(track._history[-1])


def _predicted_range(track: TrackedTarget, dt_s: float) -> float:
    return _last_range(track) + float(track.velocity_mps) * dt_s


def _hand_range_valid(
    r_hand: float,
    r_body: float,
    min_hand_ahead_m: float,
) -> bool:
    """Hands must be closer to the radar than the body (never behind id=1)."""
    gap = float(r_body) - float(r_hand)
    return gap >= float(min_hand_ahead_m)


def _velocity_from_history(history: deque[float], dt_s: float) -> float:
    if len(history) < 2:
        return 0.0
    dr = float(history[-1]) - float(history[0])
    dt = (len(history) - 1) * dt_s
    return dr / dt if dt > 0 else 0.0


def _velocity_if_updated(
    history: deque[float],
    r_meas: float,
    dt_s: float,
    history_frames: int,
) -> float:
    """Velocity after appending ``r_meas`` (same window as tracker update)."""
    test = deque(history, maxlen=history.maxlen or history_frames)
    test.append(float(r_meas))
    while len(test) > history_frames:
        test.popleft()
    return _velocity_from_history(test, dt_s)


def _best_measurement_for_track(
    track: TrackedTarget,
    measurements: list[tuple[float, float]],
    skip: set[int],
    dt_s: float,
    associate_gate_m: float,
    max_range_jump_m: float,
    *,
    body_range_m: float | None = None,
    min_hand_ahead_m: float = 0.20,
    max_abs_velocity_mps: float | None = None,
    body_anchor_m: float | None = None,
    body_anchor_blend: float = 0.5,
    min_snr_db: float | None = None,
) -> int | None:
    """Closest peak to predicted range within gate and jump limits."""
    last_r = _last_range(track)
    pred_r = _predicted_range(track, dt_s)
    if track.track_id == 1 and body_anchor_m is not None:
        blend = min(max(float(body_anchor_blend), 0.0), 1.0)
        pred_r = (1.0 - blend) * pred_r + blend * float(body_anchor_m)
    best_j: int | None = None
    best_cost = float(associate_gate_m)
    history_frames = len(track._history) if track._history.maxlen is None else track._history.maxlen
    for j, (r_meas, snr_meas) in enumerate(measurements):
        if j in skip:
            continue
        r = float(r_meas)
        if min_snr_db is not None and float(snr_meas) < float(min_snr_db):
            continue
        if track.track_id == 2 and body_range_m is not None:
            if not _hand_range_valid(r, body_range_m, min_hand_ahead_m):
                continue
        if abs(r - last_r) > max_range_jump_m:
            continue
        if max_abs_velocity_mps is not None:
            v = _velocity_if_updated(
                track._history, r, dt_s, int(history_frames)
            )
            if abs(v) > float(max_abs_velocity_mps):
                continue
        cost = abs(r - pred_r)
        if cost >= best_cost:
            continue
        best_cost = cost
        best_j = j
    return best_j


def _may_spawn_hand_track(
    r_hand: float,
    r_body: float,
    min_hand_ahead_m: float,
    max_hand_ahead_m: float,
) -> bool:
    """Hand peak must sit in [min_hand_ahead_m, max_hand_ahead_m] in front of body."""
    gap = float(r_body) - float(r_hand)
    return float(min_hand_ahead_m) <= gap <= float(max_hand_ahead_m)


def _track_by_id(tracks: list[TrackedTarget], track_id: int) -> TrackedTarget | None:
    return next((t for t in tracks if t.track_id == track_id), None)


def gesture_from_tracks(tracks: list[TrackedTarget]) -> str:
    """Push/pull only from hand track (id=2). Body (id=1) never gestures."""
    hand = _track_by_id(tracks, 2)
    if hand is None:
        return GESTURE_NONE
    return hand.gesture


def velocity_from_tracks(tracks: list[TrackedTarget]) -> float:
    """OSC velocity from hand track when a confirmed push/pull is active."""
    hand = _track_by_id(tracks, 2)
    if hand is None or hand.gesture == GESTURE_NONE:
        return 0.0
    return hand.velocity_mps


@dataclass(frozen=True)
class TrackingVolume:
    """Per-frame tracker output (same path as live/replay OSC)."""

    track1_range_m: np.ndarray
    track1_snr_db: np.ndarray
    track1_gesture: np.ndarray
    track1_velocity_mps: np.ndarray
    track2_range_m: np.ndarray
    track2_snr_db: np.ndarray
    track2_gesture: np.ndarray
    gesture: np.ndarray
    raw_gesture: np.ndarray
    velocity_mps: np.ndarray


def simulate_tracking_volume(
    snr_db: np.ndarray,
    range_m: np.ndarray,
    peak_cfg: RangePeakDetectionConfig,
    track_cfg: TrackingConfig,
    dt_s: float,
) -> TrackingVolume:
    """Replay ``RangePeakTracker`` over a range–time SNR volume."""
    from gesture_recognition.peaks import peaks_from_profile

    n_frames = int(snr_db.shape[0])
    t1r = np.full(n_frames, np.nan, dtype=np.float64)
    t1s = np.full(n_frames, np.nan, dtype=np.float64)
    t1v = np.zeros(n_frames, dtype=np.float64)
    t1g = np.full(n_frames, GESTURE_NONE, dtype=object)
    t2r = np.full(n_frames, np.nan, dtype=np.float64)
    t2s = np.full(n_frames, np.nan, dtype=np.float64)
    t2g = np.full(n_frames, GESTURE_NONE, dtype=object)
    gest = np.full(n_frames, GESTURE_NONE, dtype=object)
    raw = np.full(n_frames, GESTURE_NONE, dtype=object)
    vel = np.zeros(n_frames, dtype=np.float64)

    tracker = RangePeakTracker(track_cfg)
    dt_s = max(float(dt_s), 1e-6)
    for fi in range(n_frames):
        peaks = peaks_from_profile(
            snr_db[fi], range_m, peak_cfg, for_tracking=True, track_cfg=track_cfg
        )
        tracks = tracker.update(peaks, dt_s, peak_cfg, track_cfg=track_cfg)
        t1 = _track_by_id(tracks, 1)
        t2 = _track_by_id(tracks, 2)
        if t1 is not None:
            t1r[fi] = t1.range_m
            t1s[fi] = t1.snr_db
            t1v[fi] = t1.velocity_mps
            t1g[fi] = GESTURE_NONE
        if t2 is not None:
            t2r[fi] = t2.range_m
            t2s[fi] = t2.snr_db
            t2g[fi] = t2.gesture
            raw[fi] = t2.raw_gesture
        gest[fi] = gesture_from_tracks(tracks)
        vel[fi] = velocity_from_tracks(tracks)

    return TrackingVolume(
        track1_range_m=t1r,
        track1_snr_db=t1s,
        track1_gesture=t1g,
        track1_velocity_mps=t1v,
        track2_range_m=t2r,
        track2_snr_db=t2s,
        track2_gesture=t2g,
        gesture=gest,
        raw_gesture=raw,
        velocity_mps=vel,
    )
