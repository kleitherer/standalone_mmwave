"""Range–Doppler FFT (simplified from pipeline_utils.RD / apply_fft)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _native_slow(params: Dict) -> int:
    return int(params.get("n_slow", int(params["n_chirps"]) // int(params["n_tx"])))


def effective_fft_size(native: int, requested: Optional[int]) -> int:
    """Use ``requested`` when it exceeds native size; otherwise native (no upsampling)."""
    if requested is None or requested <= 0:
        return native
    return max(native, int(requested))


def range_doppler_axes(
    params: Dict,
    *,
    n_doppler_fft: Optional[int] = None,
    n_range_fft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Range (m) and Doppler (m/s) bin centers.

    When ``n_doppler_fft`` / ``n_range_fft`` exceed native sizes, axes reflect
    zero-padded FFT bin spacing (display interpolation — same physical span,
    finer visual resolution).
    """
    n_samples_native = int(params["n_samples"])
    n_slow_native = _native_slow(params)
    n_samples = effective_fft_size(n_samples_native, n_range_fft)
    n_slow = effective_fft_size(n_slow_native, n_doppler_fft)

    range_max = float(params["range_max"])
    velocity_max = float(params["velocity_max"])
    range_axis = np.arange(n_samples) * (range_max / float(n_samples))
    velocity_res = (2.0 * velocity_max) / float(n_slow)
    doppler_axis = (np.arange(n_slow) - n_slow // 2) * velocity_res
    return range_axis, doppler_axis


def compute_rda(
    radar_cube: np.ndarray,
    *,
    declutter: bool = True,
    window: bool = True,
    n_doppler_fft: Optional[int] = None,
    n_range_fft: Optional[int] = None,
) -> np.ndarray:
    """
    Range–Doppler map per virtual antenna.

    Parameters
    ----------
    radar_cube : (n_slow, n_ant, n_samples) complex

    n_doppler_fft, n_range_fft
        Optional zero-padded FFT lengths for display. Values ≤ native size are
        ignored. Detection / peak picking should omit these (native 32 Doppler bins).

    Returns
    -------
    RDa : (n_doppler, n_ant, n_range) complex — Doppler fftshift along axis 0,
          range FFT along axis 2.
    """
    rda = np.asarray(radar_cube, dtype=np.complex64)
    n_slow_native = rda.shape[0]
    n_samples_native = rda.shape[-1]
    n_doppler = effective_fft_size(n_slow_native, n_doppler_fft)
    n_range = effective_fft_size(n_samples_native, n_range_fft)

    if declutter:
        rda = rda - rda.mean(axis=0, keepdims=True)

    if window:
        w_r = np.hanning(n_samples_native).astype(np.float32)
        w_d = np.hanning(n_slow_native).astype(np.float32)
        rda = rda * w_d[:, None, None] * w_r[None, None, :]

    rda = np.fft.fft(rda, n=n_range, axis=-1)
    rda = np.fft.fftshift(np.fft.fft(rda, n=n_doppler, axis=0), axes=0)
    return rda


def rda_power_db(RDa: np.ndarray, axis_ant: int = 1) -> np.ndarray:
    """Log power summed over virtual antennas: (n_slow, n_range)."""
    p = np.mean(np.abs(RDa) ** 2, axis=axis_ant)
    return 10.0 * np.log10(p + 1e-12)


def pw2db(x: np.ndarray, scale: float = 10.0) -> np.ndarray:
    """mmw-tracking ``pipeline_utils.pw2db`` — dB from linear power (or magnitude)."""
    return scale * np.log10(np.abs(x) + 1e-9)


def rd_power_mmw_db(RDa: np.ndarray, axis_ant: int = 1) -> np.ndarray:
    """
    mmw ``rd_heatmap.py`` display: mean |RDa|² over antennas, then ``pw2db``.

    Expects ``RDa`` from ``compute_rda(..., declutter=True)`` (chirp-mean DC removal).
    """
    return pw2db(np.mean(np.abs(RDa) ** 2, axis=axis_ant))
