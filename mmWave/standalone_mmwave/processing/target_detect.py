"""Per-frame mmWave target: range, Doppler, angle, SNR from radar cube / RDA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from processing.cube import frame_to_radar_cube
from processing.rda import compute_rda, range_doppler_axes, rda_power_db


def pick_rd_peak(
    rd_roi: np.ndarray,
    range_bins_m: np.ndarray,
    *,
    push_pull_mode: bool = False,
    push_pull_snr_within_db: float = 3.0,
) -> tuple[int, int, float, float]:
    """
    Choose a range–Doppler cell in the ROI.

    Default: global argmax SNR (peak − median).
    push_pull_mode: among cells within ``push_pull_snr_within_db`` of max SNR,
    pick the closest range (favors moving arms over a stationary torso).
    """
    noise_floor = float(np.median(rd_roi))
    snr_map = rd_roi.astype(np.float64) - noise_floor
    max_snr = float(np.max(snr_map))

    if push_pull_mode:
        thresh = max_snr - float(push_pull_snr_within_db)
        mask = snr_map >= thresh
        if np.any(mask):
            d_idxs, r_idxs = np.where(mask)
            closest = int(np.argmin(range_bins_m[r_idxs]))
            d_idx = int(d_idxs[closest])
            r_local = int(r_idxs[closest])
        else:
            d_idx, r_local = np.unravel_index(int(np.argmax(snr_map)), snr_map.shape)
            d_idx, r_local = int(d_idx), int(r_local)
    else:
        d_idx, r_local = np.unravel_index(int(np.argmax(snr_map)), snr_map.shape)
        d_idx, r_local = int(d_idx), int(r_local)

    peak_db = float(rd_roi[d_idx, r_local])
    snr_db = float(snr_map[d_idx, r_local])
    return d_idx, r_local, peak_db, snr_db


@dataclass
class LiveRadarTarget:
    range_m: float
    doppler_mps: float
    angle_deg: float
    snr_db: float
    peak_db: float
    energy: float
    presence: float


class LiveRadarTargetProcessor:
    """
    Strongest peak in range–Doppler (ROI) + FFT angle on virtual antennas.

    Angle uses FFT across antennas at the peak cell (approximate azimuth).
    """

    def __init__(
        self,
        params: Dict,
        *,
        range_gate_m: Tuple[float, float] = (0.5, 12.0),
        doppler_gate_mps: Tuple[float, float] | None = None,
        clutter_window: int = 45,
        background_rd_mean: np.ndarray | None = None,
        smooth_alpha: float = 0.2,
        angle_fft_bins: int = 128,
        presence_threshold_db: float = 12.0,
        push_pull_mode: bool = False,
        push_pull_snr_within_db: float = 3.0,
    ):
        self.params = params
        self.range_gate_m = range_gate_m
        self.doppler_gate_mps = doppler_gate_mps
        self._calibration_frames = max(1, int(clutter_window))
        self._smooth_alpha = float(smooth_alpha)
        self._angle_fft_bins = int(angle_fft_bins)
        self._presence_threshold_db = float(presence_threshold_db)
        self._push_pull_mode = bool(push_pull_mode)
        self._push_pull_snr_within_db = float(push_pull_snr_within_db)
        self._rd_sum: Optional[np.ndarray] = None
        self._rd_mean: Optional[np.ndarray] = (
            np.asarray(background_rd_mean, dtype=np.float64) if background_rd_mean is not None else None
        )
        self._frames_seen = 0
        self._smooth: Optional[Tuple[float, float, float]] = None  # range, doppler, angle

        self._range_axis, self._doppler_axis = range_doppler_axes(params)

    def _angle_deg(self, rda: np.ndarray, d_idx: int, r_idx: int) -> float:
        snap = rda[d_idx, :, r_idx]
        if snap.size < 2:
            return 0.0
        spec = np.fft.fftshift(np.fft.fft(snap, n=self._angle_fft_bins))
        a_idx = int(np.argmax(np.abs(spec)))
        angles = np.linspace(-90.0, 90.0, self._angle_fft_bins)
        return float(angles[a_idx])

    def update(self, frame_int16: np.ndarray) -> Optional[LiveRadarTarget]:
        cube = frame_to_radar_cube(frame_int16, self.params)
        rda = compute_rda(cube)
        rd_db = rda_power_db(rda)

        if self._rd_mean is None:
            if self._rd_sum is None:
                self._rd_sum = np.zeros_like(rd_db, dtype=np.float64)
            self._rd_sum += rd_db.astype(np.float64)
            self._frames_seen += 1
            if self._frames_seen < self._calibration_frames:
                return None
            self._rd_mean = self._rd_sum / float(self._frames_seen)
            self._rd_sum = None

        rd_db = rd_db - self._rd_mean

        r_lo, r_hi = self.range_gate_m
        r_mask = (self._range_axis >= r_lo) & (self._range_axis <= r_hi)
        if not np.any(r_mask):
            return None

        rd_roi = rd_db[:, r_mask]
        d_axis = self._doppler_axis
        if self.doppler_gate_mps is not None:
            v_lo, v_hi = self.doppler_gate_mps
            d_mask = (d_axis >= v_lo) & (d_axis <= v_hi)
            if np.any(d_mask):
                rd_roi = rd_roi[d_mask, :]
                d_axis = d_axis[d_mask]

        if rd_roi.size == 0:
            return None

        range_bins_m = self._range_axis[r_mask]
        d_idx, r_local, peak_db, snr_db = pick_rd_peak(
            rd_roi,
            range_bins_m,
            push_pull_mode=self._push_pull_mode,
            push_pull_snr_within_db=self._push_pull_snr_within_db,
        )
        r_idx = int(np.flatnonzero(r_mask)[r_local])
        if self.doppler_gate_mps is not None:
            d_idx_full = int(np.flatnonzero(d_mask)[d_idx])
        else:
            d_idx_full = d_idx

        range_m = float(self._range_axis[r_idx])
        doppler_mps = float(self._doppler_axis[d_idx_full])
        angle_deg = self._angle_deg(rda, d_idx_full, r_idx)

        if self._smooth_alpha > 0.0:
            if self._smooth is None:
                self._smooth = (range_m, doppler_mps, angle_deg)
            else:
                a = self._smooth_alpha
                sr, sd, sa = self._smooth
                self._smooth = (
                    a * range_m + (1 - a) * sr,
                    a * doppler_mps + (1 - a) * sd,
                    a * angle_deg + (1 - a) * sa,
                )
            range_m, doppler_mps, angle_deg = self._smooth

        r_lo, r_hi = self.range_gate_m
        r_mask = (self._range_axis >= r_lo) & (self._range_axis <= r_hi)
        roi = rd_db[:, r_mask]
        energy = float(np.sum(10.0 ** (roi.astype(np.float64) / 10.0)))

        presence = 1.0 if snr_db >= self._presence_threshold_db else 0.0

        return LiveRadarTarget(
            range_m=range_m,
            doppler_mps=doppler_mps,
            angle_deg=angle_deg,
            snr_db=snr_db,
            peak_db=peak_db,
            energy=energy,
            presence=presence,
        )
