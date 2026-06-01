"""Single code path: range peak list + gesture processor → OSC."""

from __future__ import annotations

from gesture_recognition.gesture import (
    GESTURE_NONE,
    GESTURE_PULL,
    GESTURE_PUSH,
    OscGestureRateLimiter,
    gesture_for_osc,
)
from gesture_recognition.mode import GestureProcessor
from gesture_recognition.osc import OscPublisher
from gesture_recognition.peaks import RangePeakDetectionConfig
from gesture_recognition.simple import SimpleGestureProcessor
from gesture_recognition.status import StreamFrameResult
from gesture_recognition.tracker import (
    RangePeakTracker,
    _track_by_id,
    gesture_from_tracks,
)


def publish_radar_frame(
    publisher: OscPublisher,
    peak_cfg: RangePeakDetectionConfig,
    peaks: list[tuple[float, float]] | None,
    processor: GestureProcessor,
    dt_s: float,
    osc_gesture_limiter: OscGestureRateLimiter,
) -> StreamFrameResult | None:
    """
    Publish one frame to Max from range peaks + gesture processor.

    ``processor`` is either :class:`RangePeakTracker` (config1) or
    :class:`SimpleGestureProcessor` (config2).
    """
    if peaks is None:
        return None

    if isinstance(processor, SimpleGestureProcessor):
        return _publish_simple_frame(
            publisher, peak_cfg, peaks, processor, dt_s, osc_gesture_limiter
        )
    return _publish_tracking_frame(
        publisher, peak_cfg, peaks, processor, dt_s, osc_gesture_limiter
    )


def _osc_gesture_and_velocity(
    confirmed: str,
    velocity_mps: float,
    dt_s: float,
    limiter: OscGestureRateLimiter,
) -> tuple[str | None, float, str]:
    """Returns (osc_gesture_or_none, doppler_mps, confirmed_label)."""
    osc_gest = limiter.update(confirmed, dt_s)
    vel = (
        float(velocity_mps)
        if osc_gest in (GESTURE_PUSH, GESTURE_PULL)
        else 0.0
    )
    return osc_gest, vel, confirmed


def _publish_simple_frame(
    publisher: OscPublisher,
    peak_cfg: RangePeakDetectionConfig,
    peaks: list[tuple[float, float]],
    processor: SimpleGestureProcessor,
    dt_s: float,
    osc_gesture_limiter: OscGestureRateLimiter,
) -> StreamFrameResult:
    processor.update(peaks, dt_s, peak_cfg)
    present = processor.snr_db >= peak_cfg.snr_threshold_db
    gest_osc, vel, confirmed = _osc_gesture_and_velocity(
        processor.gesture,
        processor.velocity_mps,
        dt_s,
        osc_gesture_limiter,
    )

    if not present:
        result = StreamFrameResult(
            published=False,
            present=False,
            range_m=0.0,
            snr_db=0.0,
            gesture=GESTURE_NONE,
        )
        if not publisher.emit_below_threshold:
            publisher.send_frame(
                range_m=0.0,
                snr_db=0.0,
                present=False,
                gesture=GESTURE_NONE,
            )
        return result

    published = publisher.send_frame(
        range_m=processor.range_m,
        snr_db=processor.snr_db,
        present=True,
        doppler_mps=vel,
        gesture=gest_osc,
    )
    return StreamFrameResult(
        published=published,
        present=True,
        range_m=processor.range_m,
        snr_db=processor.snr_db,
        track1_id=1,
        doppler_mps=vel,
        gesture=confirmed,
    )


def _publish_tracking_frame(
    publisher: OscPublisher,
    peak_cfg: RangePeakDetectionConfig,
    peaks: list[tuple[float, float]],
    tracker: RangePeakTracker,
    dt_s: float,
    osc_gesture_limiter: OscGestureRateLimiter,
) -> StreamFrameResult | None:
    tracks = tracker.update(peaks, dt_s, peak_cfg)
    body = _track_by_id(tracks, 1)
    hand = _track_by_id(tracks, 2)
    if body is None:
        if tracker.body_calibration_active:
            return StreamFrameResult(
                published=False,
                present=False,
                range_m=0.0,
                snr_db=0.0,
                gesture=GESTURE_NONE,
                body_calibrating=True,
                body_calib_progress_s=tracker.body_calib_still_s,
                body_calib_target_s=tracker.cfg.body_calibration_s,
            )
        result = StreamFrameResult(
            published=False,
            present=False,
            range_m=0.0,
            snr_db=0.0,
            gesture=GESTURE_NONE,
        )
        if not publisher.emit_below_threshold:
            publisher.send_frame(
                range_m=0.0,
                snr_db=0.0,
                present=False,
                gesture=GESTURE_NONE,
            )
        return result

    present = body.snr_db >= peak_cfg.snr_threshold_db
    confirmed = gesture_from_tracks(tracks)
    osc_confirmed = gesture_for_osc(
        confirmed,
        body.velocity_mps,
        body_max_velocity_for_gesture_mps=tracker.cfg.body_max_velocity_for_gesture_mps,
    )
    hand_vel = (
        float(hand.velocity_mps)
        if hand is not None and osc_confirmed != GESTURE_NONE
        else 0.0
    )
    gest_osc, vel, _ = _osc_gesture_and_velocity(
        osc_confirmed, hand_vel, dt_s, osc_gesture_limiter
    )

    r2 = hand.range_m if hand is not None else None
    snr2 = hand.snr_db if hand is not None else None
    id2 = hand.track_id if hand is not None else None

    published = publisher.send_frame(
        range_m=body.range_m,
        snr_db=body.snr_db,
        present=present,
        range2_m=r2,
        snr2_db=snr2,
        doppler_mps=vel,
        gesture=gest_osc,
    )

    return StreamFrameResult(
        published=published,
        present=present,
        range_m=body.range_m,
        snr_db=body.snr_db,
        range2_m=r2,
        snr2_db=snr2,
        track1_id=body.track_id,
        track2_id=id2,
        doppler_mps=vel,
        gesture=confirmed,
    )
