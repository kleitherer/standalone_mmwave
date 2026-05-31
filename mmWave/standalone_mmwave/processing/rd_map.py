"""
Range–Doppler power maps — single entry point for live and post-processing.

``frame_to_rd_power_db`` is the only function callers need for per-frame RD (dB).
"""

from __future__ import annotations

from typing import Any, Dict, Literal

import numpy as np

from processing.adc_cube import frame_to_adc_cube
from processing.rda import compute_rda, rd_power_mmw_db

Pipeline = Literal["standalone", "mmw"]


def adc_to_virtual_cube(adc: np.ndarray, n_tx: int, n_rx: int) -> np.ndarray:
    """
    TDM slow-time × virtual antenna cube.

    Virtual index ``v = tx * n_rx + rx`` uses slow-time series
    ``adc[tx::n_tx, rx, :]`` (same TX only — never FFT across interleaved chirps).
    """
    n_chirps, _, n_samples = adc.shape
    n_slow = n_chirps // n_tx
    virt = np.empty((n_slow, n_tx * n_rx, n_samples), dtype=np.complex64)
    for tx in range(n_tx):
        for rx in range(n_rx):
            virt[:, tx * n_rx + rx, :] = adc[tx::n_tx, rx, :]
    return virt


def frame_to_rd_power_db(
    frame: np.ndarray,
    params: Dict[str, Any],
    *,
    pipeline: Pipeline = "standalone",
    declutter: bool = True,
    window: bool = True,
    n_doppler_fft: int | None = None,
    n_range_fft: int | None = None,
    limiter: bool = False,
) -> np.ndarray:
    """
    One int16 frame → range–Doppler power map in dB, antennas non-coherently combined.

    Parameters
    ----------
    pipeline
        ``standalone`` — TDM virtual cube + ``compute_rda`` (default; supports display
        zero-pad via ``n_doppler_fft`` / ``n_range_fft``).
        ``mmw`` — mmw-tracking ``mmw_rd`` chain (native 32×256, Doppler FFT first).
    limiter
        1-bit IQ limiter before FFT (range–time display only; not for RD analysis).
    """
    if pipeline == "mmw":
        from processing.mmw_rd import frame_to_mmw_rd_power_db

        return frame_to_mmw_rd_power_db(
            frame,
            params,
            declutter=declutter,
            window=window,
        )

    adc = frame_to_adc_cube(frame, params, limiter=limiter)
    n_tx = int(params["n_tx"])
    n_rx = int(params["n_rx"])
    virt = adc_to_virtual_cube(adc, n_tx, n_rx)
    return rd_power_mmw_db(
        compute_rda(
            virt,
            declutter=declutter,
            window=window,
            n_doppler_fft=n_doppler_fft,
            n_range_fft=n_range_fft,
        )
    )
