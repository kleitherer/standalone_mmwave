"""Range–time SNR targets — same pipeline as ``post_processing.plot_heatmap`` / ``range_time_snr.npz``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from background_model import estimate_rd_background_from_capture, frame_rd_power_db
from processing.rda import range_doppler_axes


def rd_to_snr_db(rd_roi: np.ndarray) -> np.ndarray:
    """SNR (dB) relative to per-frame ROI median (matches ``rd_maps.rd_to_snr_db``)."""
    noise_db = float(np.median(rd_roi))
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

    def targets_for_frame(
        self,
        frame_idx: int,
        *,
        min_secondary_snr_db: float,
        secondary_min_sep_m: float,
        secondary_max_sep_m: float,
    ) -> list[tuple[float, float]]:
        if frame_idx < 0 or frame_idx >= self.snr_db.shape[0]:
            return []
        primary, secondary = select_range_time_targets(
            self.snr_db[frame_idx],
            self.range_m,
            min_secondary_snr_db=min_secondary_snr_db,
            secondary_min_sep_m=secondary_min_sep_m,
            secondary_max_sep_m=secondary_max_sep_m,
        )
        return targets_list_from_pair(primary, secondary)


class RangeTimeSnrProcessor:
    """
    Per-frame range–time SNR targets using the post-processing RD path
    (``frame_rd_power_db`` + optional limiter + background subtraction).
    """

    def __init__(
        self,
        params: Dict,
        *,
        range_gate_m: Tuple[float, float],
        background_rd_mean: np.ndarray | None = None,
        limiter: bool = False,
        inline_calib_frames: int = 0,
    ) -> None:
        self.params = params
        self.range_gate_m = range_gate_m
        self._bg: np.ndarray | None = (
            np.asarray(background_rd_mean, dtype=np.float64) if background_rd_mean is not None else None
        )
        self._limiter = bool(limiter)
        self._inline_calib_frames = max(0, int(inline_calib_frames))
        self._rd_sum: np.ndarray | None = None
        self._frames_seen = 0
        range_axis, _ = range_doppler_axes(params)
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
        bg = estimate_rd_background_from_capture(
            Path(bg_src),
            params,
            max_frames=max(0, int(max_f)),
            limiter=limiter,
        )
        return cls(
            params,
            range_gate_m=range_gate_m,
            background_rd_mean=bg,
            limiter=limiter,
        )

    @property
    def ready(self) -> bool:
        return self._bg is not None

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    def targets_from_frame(
        self,
        frame_int16: np.ndarray,
        *,
        min_secondary_snr_db: float,
        secondary_min_sep_m: float,
        secondary_max_sep_m: float,
    ) -> list[tuple[float, float]] | None:
        rd_raw = frame_rd_power_db(
            frame_int16,
            self.params,
            limiter=self._limiter,
        ).astype(np.float64)

        if self._bg is None and self._inline_calib_frames > 0:
            if self._rd_sum is None:
                self._rd_sum = np.zeros_like(rd_raw, dtype=np.float64)
            self._rd_sum += rd_raw
            self._frames_seen += 1
            if self._frames_seen < self._inline_calib_frames:
                return None
            self._bg = self._rd_sum / float(self._frames_seen)
            self._rd_sum = None

        if self._bg is None:
            self._bg = np.zeros_like(rd_raw, dtype=np.float64)

        rd_declutter = rd_raw - self._bg
        if not np.any(self._r_mask):
            return []
        rd_roi = rd_declutter[:, self._r_mask]
        profile = range_time_snr_along_range(rd_roi)
        primary, secondary = select_range_time_targets(
            profile,
            self._range_bins_m,
            min_secondary_snr_db=min_secondary_snr_db,
            secondary_min_sep_m=secondary_min_sep_m,
            secondary_max_sep_m=secondary_max_sep_m,
        )
        return targets_list_from_pair(primary, secondary)
