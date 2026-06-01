"""Unified terminal status line for live and replay."""

from __future__ import annotations

from dataclasses import dataclass

from gesture_recognition.gesture import GESTURE_NONE


@dataclass(frozen=True)
class StreamFrameResult:
    """One published (or skipped) radar frame."""

    published: bool
    present: bool
    range_m: float
    snr_db: float
    range2_m: float | None = None
    snr2_db: float | None = None
    track1_id: int | None = None
    track2_id: int | None = None
    doppler_mps: float | None = None
    angle_deg: float | None = None
    track1_doppler_mps: float | None = None
    gesture: str = GESTURE_NONE
    body_calibrating: bool = False
    body_calib_progress_s: float | None = None
    body_calib_target_s: float | None = None
    body_calib_range_m: float | None = None
    body_calib_snr_db: float | None = None


def format_status_line(
    result: StreamFrameResult,
    *,
    elapsed_s: float,
    osc_sent: int,
    osc_rate_hz: float,
    frame_label: str | None = None,
    gesture_mode: str | None = None,
) -> str:
    """
    Same layout for live and replay::

        12.3s  frame 42/835  osc=380 (45.0/s)
        T1[id=1]: R=1.45m SNR=18.2dB  |  T2[id=2]: R=1.12m SNR=14.1dB
        V=-0.08m/s  gesture=push
    """
    head = f"  {elapsed_s:5.1f}s"
    if gesture_mode:
        head += f"  {gesture_mode}"
    if frame_label:
        head += f"  {frame_label}"
    head += f"  osc={osc_sent} ({osc_rate_hz:.1f}/s)"

    if result.body_calibrating:
        prog = result.body_calib_progress_s if result.body_calib_progress_s is not None else 0.0
        target = result.body_calib_target_s if result.body_calib_target_s is not None else 0.0
        return f"{head}  body calib: {prog:.1f}/{target:.1f}s — stand still in front of radar"

    id1 = result.track1_id if result.track1_id is not None else 1
    t2 = (
        f"T2[id={result.track2_id}]: R={result.range2_m:.2f}m SNR={result.snr2_db:.1f}dB"
        if result.range2_m is not None
        and result.snr2_db is not None
        and result.track2_id is not None
        else "T2: —"
    )
    line1 = (
        f"{head}  T1[id={id1}]: R={result.range_m:.2f}m SNR={result.snr_db:.1f}dB  |  {t2}"
    )

    dop = result.doppler_mps if result.doppler_mps is not None else 0.0
    gest = result.gesture or GESTURE_NONE
    line2 = f"           V={dop:+.2f}m/s  gesture={gest}"
    if result.angle_deg is not None:
        rd = (
            f"  RD={result.track1_doppler_mps:+.2f}m/s"
            if result.track1_doppler_mps is not None
            else ""
        )
        line2 += f"  angle={result.angle_deg:+.1f}°{rd}"
    return f"{line1}\n{line2}"
