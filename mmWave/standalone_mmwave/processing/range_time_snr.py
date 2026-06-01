"""Range–time SNR targets — same pipeline as ``post_processing.plot_heatmap`` / ``range_time_snr.npz``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np

from background_model import (
    estimate_rd_background_from_capture,
    estimate_rda_background_from_capture,
)
from processing.angle_estimate import angle_at_track_range
from processing.cube import frame_to_radar_cube
from processing.rd_map import frame_to_rd_power_db
from processing.rda import compute_rda, range_doppler_axes
from processing.range_peak_detection import (
    RangePeakDetectionConfig,
    osc_targets_from_profile,
    peaks_from_profile,
)


def rd_to_snr_db(rd_roi: np.ndarray) -> np.ndarray:
    """SNR (dB) relative to per-frame ROI mean (matches ``rd_maps.rd_to_snr_db``)."""
    noise_db = float(np.mean(rd_roi))
    return rd_roi.astype(np.float64) - noise_db


def range_time_snr_along_range(rd_roi_declutter: np.ndarray) -> np.ndarray:
    """1D range profile: decluttered RD → median SNR → max over Doppler."""
    return np.max(rd_to_snr_db(rd_roi_declutter), axis=0)


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


def select_range_time_targets(
    snr_along_range: np.ndarray,
    range_bins_m: np.ndarray,
    *,
    min_secondary_snr_db: float = -np.inf,
    secondary_min_sep_m: float = 0.3,
    secondary_max_sep_m: float = 2.0,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """
    Primary = argmax on the range–time SNR row; secondary = strongest local peak
    ``secondary_min_sep_m`` < |Δrange| < ``secondary_max_sep_m`` from primary.
    """
    if snr_along_range.size == 0:
        return None, None

    primary_idx = int(np.argmax(snr_along_range))
    r1 = float(range_bins_m[primary_idx])
    snr1 = float(snr_along_range[primary_idx])
    primary = (r1, snr1)

    peaks = _range_profile_peaks(snr_along_range, -np.inf)
    min_sep = float(secondary_min_sep_m)
    max_sep = float(secondary_max_sep_m)
    min_snr2 = float(min_secondary_snr_db)
    secondary: tuple[float, float] | None = None
    best_snr2 = -np.inf
    for idx in peaks:
        if idx == primary_idx:
            continue
        snr2 = float(snr_along_range[idx])
        if snr2 < min_snr2:
            continue
        r2 = float(range_bins_m[idx])
        sep = abs(r2 - r1)
        if min_sep < sep < max_sep and snr2 > best_snr2:
            best_snr2 = snr2
            secondary = (r2, snr2)

    return primary, secondary


def targets_list_from_pair(
    primary: tuple[float, float] | None,
    secondary: tuple[float, float] | None,
) -> list[tuple[float, float]]:
    if primary is None:
        return []
    out = [primary]
    if secondary is not None:
        out.append(secondary)
    return out


@dataclass
class RangeTimeSnrNpz:
    """Precomputed ``analysis/range_time_snr.npz`` from ``plot_heatmap``."""

    snr_db: np.ndarray  # (n_frames, n_range)
    range_m: np.ndarray
    snr_threshold_db: float

    @classmethod
    def load(cls, path: Path) -> Optional[RangeTimeSnrNpz]:
        if not path.is_file():
            return None
        data = np.load(path)
        if "snr_db" not in data or "range_m" not in data:
            return None
        thr = float(data["snr_threshold_db"]) if "snr_threshold_db" in data else 0.0
        return cls(
            snr_db=np.asarray(data["snr_db"], dtype=np.float64),
            range_m=np.asarray(data["range_m"], dtype=np.float64),
            snr_threshold_db=thr,
        )

    def peaks_for_frame(
        self,
        frame_idx: int,
        peak_cfg: RangePeakDetectionConfig,
        *,
        track_cfg: Any = None,
        for_tracking: bool = False,
    ) -> list[tuple[float, float]]:
        if frame_idx < 0 or frame_idx >= self.snr_db.shape[0]:
            return []
        return peaks_from_profile(
            self.snr_db[frame_idx],
            self.range_m,
            peak_cfg,
            for_tracking=for_tracking,
            track_cfg=track_cfg,
        )

    def targets_for_frame(
        self,
        frame_idx: int,
        peak_cfg: RangePeakDetectionConfig,
    ) -> list[tuple[float, float]]:
        if frame_idx < 0 or frame_idx >= self.snr_db.shape[0]:
            return []
        return osc_targets_from_profile(self.snr_db[frame_idx], self.range_m, peak_cfg)


class RangeTimeSnrProcessor:
    """
    Per-frame range–time SNR targets using the post-processing RD path
    (``frame_to_rd_power_db`` + optional limiter + background subtraction).

    Angle at a tracked range uses the same split as ``post_processing.rd_maps``:
    decluttered RD power for Doppler selection, decluttered complex RDa for AoA.
    """

    def __init__(
        self,
        params: Dict,
        *,
        range_gate_m: Tuple[float, float],
        background_rd_mean: np.ndarray | None = None,
        background_rda_mean: np.ndarray | None = None,
        limiter: bool = False,
        inline_calib_frames: int = 0,
    ) -> None:
        self.params = params
        self.range_gate_m = range_gate_m
        self._bg_rd: np.ndarray | None = (
            np.asarray(background_rd_mean, dtype=np.float64) if background_rd_mean is not None else None
        )
        self._bg_rda: np.ndarray | None = (
            np.asarray(background_rda_mean, dtype=np.complex128)
            if background_rda_mean is not None
            else None
        )
        self._limiter = bool(limiter)
        self._inline_calib_frames = max(0, int(inline_calib_frames))
        self._rd_sum: np.ndarray | None = None
        self._rda_sum: np.ndarray | None = None
        self._frames_seen = 0
        range_axis, doppler_axis = range_doppler_axes(params)
        self._range_axis = range_axis
        self._doppler_axis = doppler_axis
        r_lo, r_hi = range_gate_m
        self._r_mask = (range_axis >= r_lo) & (range_axis <= r_hi)
        self._range_bins_m = range_axis[self._r_mask]

    @classmethod
    def from_capture(
        cls,
        capture_path: Path,
        params: Dict,
        *,
        range_gate_m: Tuple[float, float],
        declutter_mean_frames: int,
        limiter: bool = False,
        background_capture: Path | None = None,
        background_max_frames: int = 0,
    ) -> RangeTimeSnrProcessor:
        bg_src = background_capture if background_capture is not None else capture_path
        max_f = background_max_frames if background_capture is not None else declutter_mean_frames
        bg_rd = estimate_rd_background_from_capture(
            Path(bg_src),
            params,
            max_frames=max(0, int(max_f)),
            limiter=limiter,
        )
        bg_rda = estimate_rda_background_from_capture(
            Path(bg_src),
            params,
            max_frames=max(0, int(max_f)),
        )
        return cls(
            params,
            range_gate_m=range_gate_m,
            background_rd_mean=bg_rd,
            background_rda_mean=bg_rda,
            limiter=limiter,
        )

    @property
    def ready(self) -> bool:
        return self._bg_rd is not None

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    def _frame_rd_rda_from_int16(
        self,
        frame_int16: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Decluttered RD power (dB) + decluttered complex RDa — matches ``rd_maps.py``.

        RD uses ``frame_to_rd_power_db`` (optional limiter for range–time display).
        RDa uses ``compute_rda(frame_to_radar_cube(...))`` without limiter.

        Returns ``(rda_declutter, rd_declutter)`` or ``None`` during inline warmup.
        """
        rd_raw = frame_to_rd_power_db(
            frame_int16,
            self.params,
            limiter=self._limiter,
        ).astype(np.float64)
        rda = compute_rda(frame_to_radar_cube(frame_int16, self.params))

        if self._bg_rd is None and self._inline_calib_frames > 0:
            if self._rd_sum is None:
                self._rd_sum = np.zeros_like(rd_raw, dtype=np.float64)
                self._rda_sum = np.zeros_like(rda, dtype=np.complex128)
            self._rd_sum += rd_raw
            self._rda_sum += rda.astype(np.complex128)
            self._frames_seen += 1
            if self._frames_seen < self._inline_calib_frames:
                return None
            self._bg_rd = self._rd_sum / float(self._frames_seen)
            self._bg_rda = self._rda_sum / float(self._frames_seen)
            self._rd_sum = None
            self._rda_sum = None

        if self._bg_rd is None:
            self._bg_rd = np.zeros_like(rd_raw, dtype=np.float64)
        if self._bg_rda is None:
            self._bg_rda = np.zeros_like(rda, dtype=np.complex128)

        return rda - self._bg_rda, rd_raw - self._bg_rd

    def _profile_from_frame(
        self,
        frame_int16: np.ndarray,
    ) -> np.ndarray | None:
        prep = self._frame_rd_rda_from_int16(frame_int16)
        if prep is None:
            return None
        _, rd_declutter = prep
        if not np.any(self._r_mask):
            return np.array([], dtype=np.float64)
        rd_roi = rd_declutter[:, self._r_mask]
        return range_time_snr_along_range(rd_roi)

    def angle_at_range_m(
        self,
        frame_int16: np.ndarray,
        range_m: float,
        *,
        n_bins: int = 128,
        fov_deg: float = 90.0,
    ) -> tuple[float, float] | None:
        """
        Track-id-1 angle at a known range.

        Doppler from the decluttered RD heatmap column at ``range_m``
        (``doppler_idx_per_range``, same as range–azimuth plots). Antenna FFT
        on the decluttered complex vector ``RDa[d, :, r]``.

        Returns ``(angle_deg, doppler_mps)`` or ``None`` during warmup.
        """
        prep = self._frame_rd_rda_from_int16(frame_int16)
        if prep is None:
            return None
        rda_declutter, rd_declutter = prep
        if not np.any(self._r_mask):
            return None

        angle_deg, doppler_mps, _, _ = angle_at_track_range(
            rda_declutter,
            rd_declutter,
            range_m,
            self._range_bins_m,
            self._r_mask,
            self._doppler_axis,
            n_bins=int(n_bins),
            fov_deg=float(fov_deg),
        )
        return angle_deg, doppler_mps

    def peaks_from_frame(
        self,
        frame_int16: np.ndarray,
        peak_cfg: RangePeakDetectionConfig,
        track_cfg: Any | None = None,
    ) -> list[tuple[float, float]] | None:
        profile = self._profile_from_frame(frame_int16)
        if profile is None:
            return None
        if profile.size == 0:
            return []
        if track_cfg is not None:
            return peaks_from_profile(
                profile,
                self._range_bins_m,
                peak_cfg,
                for_tracking=True,
                track_cfg=track_cfg,
            )
        return peaks_from_profile(profile, self._range_bins_m, peak_cfg)

    def targets_from_frame(
        self,
        frame_int16: np.ndarray,
        peak_cfg: RangePeakDetectionConfig,
    ) -> list[tuple[float, float]] | None:
        profile = self._profile_from_frame(frame_int16)
        if profile is None:
            return None
        if profile.size == 0:
            return []
        return osc_targets_from_profile(profile, self._range_bins_m, peak_cfg)
