"""Push/pull and presence gesture labels (from range peaks or velocity)."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

GESTURE_NONE = "none"
GESTURE_SINGLE = "single"
GESTURE_PUSH = "push"
GESTURE_PULL = "pull"
GESTURE_PUSH_PULL = "push_pull"


def rd_roi_snr_map(rd_roi: np.ndarray) -> np.ndarray:
    noise_floor = float(np.median(rd_roi))
    return rd_roi.astype(np.float64) - noise_floor


def _range_profile_peaks(snr_along_range: np.ndarray, thresh: float) -> list[int]:
    peaks: list[int] = []
    n = int(snr_along_range.size)
    for i in range(n):
        v = float(snr_along_range[i])
        if v < thresh:
            continue
        left = float(snr_along_range[i - 1]) if i > 0 else -np.inf
        right = float(snr_along_range[i + 1]) if i + 1 < n else -np.inf
        if v >= left and v >= right:
            peaks.append(i)
    return peaks


def classify_gesture_from_rd(
    rd_roi: np.ndarray,
    range_bins_m: np.ndarray,
    *,
    snr_within_db_of_max: float = 4.0,
    min_peak_snr_db: float = 5.0,
    min_range_sep_m: float = 0.2,
    max_range_sep_m: float = 1.5,
) -> str:
    snr_map = rd_roi_snr_map(rd_roi)
    max_snr = float(np.max(snr_map))
    if max_snr < float(min_peak_snr_db):
        return GESTURE_NONE

    snr_along_range = np.max(snr_map, axis=0)
    thresh = max(max_snr - float(snr_within_db_of_max), float(min_peak_snr_db))
    peaks = _range_profile_peaks(snr_along_range, thresh)
    if len(peaks) < 2:
        return GESTURE_SINGLE if peaks else GESTURE_NONE

    ranges = range_bins_m[np.asarray(peaks, dtype=np.intp)]
    min_sep = float(min_range_sep_m)
    max_sep = float(max_range_sep_m)
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            sep = abs(float(ranges[i]) - float(ranges[j]))
            if min_sep <= sep <= max_sep:
                return GESTURE_PUSH_PULL
    return GESTURE_SINGLE


def classify_gesture_from_velocity(
    velocity_mps: float,
    *,
    min_velocity_mps: float = 0.15,
) -> str:
    """
    ``velocity_mps`` is ``d(range)/dt`` (m/s).

    Range decreasing (approaching radar) → push; increasing → pull.
    """
    v = float(velocity_mps)
    thr = float(min_velocity_mps)
    if v < -thr:
        return GESTURE_PUSH
    if v > thr:
        return GESTURE_PULL
    return GESTURE_NONE


def confirm_gesture_label(
    raw: str,
    pending: str,
    streak: int,
    confirm_frames: int,
) -> tuple[str, str, int]:
    """
    Require ``confirm_frames`` consecutive raw push/pull labels before OSC.

    Returns ``(confirmed_label, new_pending, new_streak)``.
    """
    if raw not in (GESTURE_PUSH, GESTURE_PULL):
        return GESTURE_NONE, GESTURE_NONE, 0
    if raw == pending:
        streak += 1
    else:
        pending = raw
        streak = 1
    confirmed = raw if streak >= int(confirm_frames) else GESTURE_NONE
    return confirmed, pending, streak


def gesture_for_osc(
    confirmed: str,
    body_velocity_mps: float | None,
    *,
    body_max_velocity_for_gesture_mps: float,
) -> str:
    """
    Gate OSC push/pull when track id=1 is moving.

    Returns ``none`` when ``|body_velocity|`` exceeds the threshold; otherwise
    returns the confirmed label unchanged.
    """
    if confirmed not in (GESTURE_PUSH, GESTURE_PULL):
        return GESTURE_NONE
    if float(body_max_velocity_for_gesture_mps) <= 0:
        return confirmed
    if body_velocity_mps is None:
        return confirmed
    if abs(float(body_velocity_mps)) > float(body_max_velocity_for_gesture_mps):
        return GESTURE_NONE
    return confirmed


@dataclass
class OscGestureRateLimiter:
    """
    OSC push/pull: emit only when the confirmed label changes.

    Returns ``None`` when unchanged (caller should skip ``/radar/gesture``).
    Returns ``push``, ``pull``, or ``none`` on transitions (including release).
    """

    _last_emitted: str = field(default=GESTURE_NONE, init=False)

    @classmethod
    def from_settings(cls, settings: dict) -> OscGestureRateLimiter:
        return cls()

    def reset(self) -> None:
        self._last_emitted = GESTURE_NONE

    def update(self, confirmed: str, dt_s: float = 0.0) -> str | None:
        """Map frame-confirmed push/pull to change-only OSC gesture."""
        label = (
            confirmed
            if confirmed in (GESTURE_PUSH, GESTURE_PULL)
            else GESTURE_NONE
        )
        if label == self._last_emitted:
            return None
        self._last_emitted = label
        return label


def apply_osc_gesture_rate_limit(
    confirmed: np.ndarray,
    dt_s: float,
    min_interval_s: float = 0.0,
    *,
    body_velocity_mps: np.ndarray | None = None,
    body_max_velocity_for_gesture_mps: float = 0.0,
) -> np.ndarray:
    """Replay change-only OSC gesture output over a per-frame confirmed series."""
    limiter = OscGestureRateLimiter()
    out = np.full(confirmed.shape, GESTURE_NONE, dtype=object)
    for fi, label in enumerate(confirmed):
        body_v = (
            float(body_velocity_mps[fi])
            if body_velocity_mps is not None
            else None
        )
        gated = gesture_for_osc(
            str(label),
            body_v,
            body_max_velocity_for_gesture_mps=body_max_velocity_for_gesture_mps,
        )
        emitted = limiter.update(gated, dt_s)
        if emitted is not None:
            out[fi] = emitted
    return out


def gesture_settings_kwargs(settings: dict) -> dict[str, float]:
    gesture_cfg = settings.get("gesture", {})
    return {
        "gesture_min_peak_snr_db": float(gesture_cfg.get("min_peak_snr_db", 5.0)),
        "gesture_min_range_sep_m": float(gesture_cfg.get("min_range_sep_m", 0.2)),
        "gesture_max_range_sep_m": float(gesture_cfg.get("max_range_sep_m", 1.5)),
        "gesture_min_velocity_mps": float(gesture_cfg.get("min_velocity_mps", 0.15)),
    }
