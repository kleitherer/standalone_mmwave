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


def angle_deg_ula_fft_at_rd_cell(
    rda: np.ndarray,
    d_idx: int,
    r_idx: int,
    *,
    n_angle_fft: int = 128,
) -> tuple[float, int]:
    """
    ULA spatial FFT (Bartlett-like) on the complex antenna vector at one RD cell.

    FFT along virtual antennas → pick strongest bin ``k_max`` (fftshift layout).
    Signed bin ``k = k_max - n_angle_fft // 2`` maps to
    ``θ = arcsin(2 / n_angle_fft · k)`` (radians internally, returns degrees).
    """
    snap = np.asarray(rda[int(d_idx), :, int(r_idx)]).ravel()
    n_fft = max(2, int(n_angle_fft))
    if snap.size < 2:
        return 0.0, 0

    spec = np.fft.fftshift(np.fft.fft(snap, n=n_fft))
    k_max = int(np.argmax(np.abs(spec) ** 2))
    k_signed = k_max - (n_fft // 2)
    sin_theta = float(np.clip(2.0 * k_signed / float(n_fft), -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_theta))), k_signed


def range_angle_to_cartesian_m(range_m: float, angle_deg: float) -> tuple[float, float]:
    """
    Polar (range, azimuth) → Cartesian in the radar ground plane.

    ``x`` = lateral (sin), ``y`` = downrange / boresight (cos); 0° = boresight.
    """
    theta = np.deg2rad(float(angle_deg))
    r = float(range_m)
    return r * float(np.sin(theta)), r * float(np.cos(theta))


def angle_at_track_range(
    rda_declutter: np.ndarray,
    rd_declutter: np.ndarray,
    range_m: float,
    range_bins_m: np.ndarray,
    r_mask: np.ndarray,
    doppler_axis: np.ndarray,
    *,
    n_bins: int = 128,
    fov_deg: float = 90.0,
) -> tuple[float, float, int, int]:
    """
    Angle at a known range using the RD-heatmap / range–azimuth pipeline.

    1. Decluttered RD map (same as plotted heatmap) → per-range Doppler peak
    2. Complex antenna vector from decluttered ``RDa`` at that (Doppler, range)
    3. FFT across virtual antennas → peak angle

    Returns ``(angle_deg, doppler_mps, d_idx, r_idx)``.
    """
    from processing.range_azimuth import doppler_idx_per_range

    rd_roi = rd_declutter[:, r_mask]
    r_local = int(np.argmin(np.abs(range_bins_m - float(range_m))))
    d_idx = int(doppler_idx_per_range(rd_roi)[r_local])
    r_idx = int(np.flatnonzero(r_mask)[r_local])
    angle_deg = angle_deg_at_rd_cell(
        rda_declutter,
        d_idx,
        r_idx,
        n_bins=int(n_bins),
        fov_deg=float(fov_deg),
    )
    doppler_mps = float(doppler_axis[d_idx])
    return angle_deg, doppler_mps, d_idx, r_idx
