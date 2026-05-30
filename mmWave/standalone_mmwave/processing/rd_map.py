"""Range–Doppler power map with explicit TDM virtual-antenna organization."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from processing.adc_cube import frame_to_adc_cube
from processing.rda import compute_rda, rd_power_mmw_db


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


def compute_rd_power_db_from_adc(
    adc: np.ndarray,
    params: Dict[str, Any],
    *,
    declutter: bool = True,
    window: bool = True,
    n_doppler_fft: int | None = None,
    n_range_fft: int | None = None,
) -> np.ndarray:  # noqa: D401 - adc may already be limited by the caller

    """
    RD power (dB): per-virtual-antenna range then Doppler FFT, then mean power
    over antennas (non-coherent). Equivalent to ``compute_rda`` + ``rda_power_db``.
    """
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


def frame_to_rd_power_db(
    frame: np.ndarray,
    params: Dict[str, Any],
    *,
    declutter: bool = True,
    window: bool = True,
    n_doppler_fft: int | None = None,
    n_range_fft: int | None = None,
    limiter: bool = False,
) -> np.ndarray:
    adc = frame_to_adc_cube(frame, params, limiter=limiter)
    return compute_rd_power_db_from_adc(
        adc,
        params,
        declutter=declutter,
        window=window,
        n_doppler_fft=n_doppler_fft,
        n_range_fft=n_range_fft,
    )
