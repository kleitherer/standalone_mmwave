"""Angle estimation shared by live target_detect and post-processing plots."""

from __future__ import annotations

import numpy as np


def angle_axis_deg(n_bins: int, fov_deg: float = 90.0) -> np.ndarray:
    """Azimuth bin centers (0° = boresight, − left, + right)."""
    half = float(fov_deg) / 2.0
    return np.linspace(-half, half, int(n_bins), dtype=np.float64)


def angle_spectrum_fft(
    antenna_snap: np.ndarray,
    n_bins: int,
    fov_deg: float = 90.0,
) -> np.ndarray:
    """
    |FFT|^2 across virtual antennas (same FFT as ``LiveRadarTargetProcessor``).

    Returns power length ``n_bins`` (only ``fov_deg`` span is meaningful if n_bins > fov mapping).
    """
    snap = np.asarray(antenna_snap).ravel()
    if snap.size < 2:
        return np.zeros(int(n_bins), dtype=np.float64)
    spec = np.fft.fftshift(np.fft.fft(snap, n=int(n_bins)))
    return (np.abs(spec) ** 2).astype(np.float64)


def angle_deg_at_rd_cell(
    rda: np.ndarray,
    d_idx: int,
    r_idx: int,
    *,
    n_bins: int = 128,
    fov_deg: float = 90.0,
) -> float:
    """Argmax on antenna FFT at one range–Doppler cell (live/replay target angle)."""
    snap = rda[int(d_idx), :, int(r_idx)]
    if snap.size < 2:
        return 0.0
    spec = np.fft.fftshift(np.fft.fft(snap, n=int(n_bins)))
    a_idx = int(np.argmax(np.abs(spec)))
    return float(angle_axis_deg(n_bins, fov_deg)[a_idx])
