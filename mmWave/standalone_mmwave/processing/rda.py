"""Range–Doppler FFT (simplified from pipeline_utils.RD / apply_fft)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def range_doppler_axes(params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Range (m) and Doppler (m/s) bin centers from capture metadata."""
    n_samples = int(params["n_samples"])
    n_chirps = int(params["n_chirps"])
    n_tx = int(params["n_tx"])
    range_res = float(params["range_res"])
    velocity_res = float(params["velocity_res"])
    n_slow = n_chirps // n_tx

    range_axis = np.arange(n_samples) * range_res
    doppler_axis = (np.arange(n_slow) - n_slow // 2) * velocity_res
    return range_axis, doppler_axis


def compute_rda(
    radar_cube: np.ndarray,
    *,
    declutter: bool = True,
    window: bool = True,
) -> np.ndarray:
    """
    Range–Doppler map per virtual antenna.

    Parameters
    ----------
    radar_cube : (n_slow, n_ant, n_samples) complex

    Returns
    -------
    RDa : (n_slow, n_ant, n_range) complex — Doppler fftshift along axis 0,
          range FFT along axis 2.
    """
    rda = np.asarray(radar_cube, dtype=np.complex64)
    if declutter:
        rda = rda - rda.mean(axis=0, keepdims=True)

    if window:
        w_r = np.hanning(rda.shape[-1]).astype(np.float32)
        w_d = np.hanning(rda.shape[0]).astype(np.float32)
        rda = rda * w_d[:, None, None] * w_r[None, None, :]

    rda = np.fft.fft(rda, axis=-1)
    rda = np.fft.fftshift(np.fft.fft(rda, axis=0), axes=0)
    return rda


def rda_power_db(RDa: np.ndarray, axis_ant: int = 1) -> np.ndarray:
    """Log power summed over virtual antennas: (n_slow, n_range)."""
    p = np.mean(np.abs(RDa) ** 2, axis=axis_ant)
    return 10.0 * np.log10(p + 1e-12)
