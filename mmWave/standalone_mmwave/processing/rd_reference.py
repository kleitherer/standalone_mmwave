"""Reference range–Doppler FFT (single TX/RX, explicit TDM decimation)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from processing.adc_cube import frame_to_adc_cube


def tdm_doppler_params(params: Dict[str, Any]) -> dict[str, float]:
    """
    Doppler axis for slow-time FFT within one frame (TDM-MIMO).

    Same-TX chirp spacing = n_tx × (idle + ramp).
    """
    n_chirps = int(params["n_chirps"])
    n_tx = int(params["n_tx"])
    n_slow = n_chirps // n_tx
    operating_freq_ghz = float(params.get("operating_freq_ghz", 60.0))
    chirp_us = float(params.get("chirp_time_us", params.get("chirp_time", 139.0)))
    if "chirp_period_same_tx_s" in params:
        chirp_period_same_tx_s = float(params["chirp_period_same_tx_s"])
    else:
        chirp_period_same_tx_s = n_tx * chirp_us * 1e-6

    wavelength = 3e8 / (operating_freq_ghz * 1e9)
    velocity_max = wavelength / (4.0 * chirp_period_same_tx_s)
    velocity_res = (2.0 * velocity_max) / float(n_slow)
    doppler_freq = np.fft.fftshift(np.fft.fftfreq(n_slow, d=chirp_period_same_tx_s))
    velocity_axis = doppler_freq * wavelength / 2.0
    return {
        "n_chirps_total": float(n_chirps),
        "n_chirps_per_tx": float(n_slow),
        "n_tx": float(n_tx),
        "chirp_period_same_tx_s": chirp_period_same_tx_s,
        "wavelength_m": wavelength,
        "velocity_max_mps": velocity_max,
        "velocity_res_mps": velocity_res,
        "velocity_axis": velocity_axis,
    }


def compute_rd_single_tx_rx(
    adc: np.ndarray,
    params: Dict[str, Any],
    *,
    tx_id: int,
    rx_id: int,
    declutter: bool = False,
    window: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Minimal RD pipeline on ``adc[tx_id::n_tx, rx_id, :]``.

    Returns
    -------
    rd_db : (n_doppler, n_range) float — 20·log10|RD|
    range_m, doppler_mps
    """
    x = np.asarray(adc[tx_id:: int(params["n_tx"]), rx_id, :], dtype=np.complex64)
    n_slow, n_samples = x.shape

    if declutter:
        x = x - np.mean(x, axis=0, keepdims=True)

    if window:
        x = x * np.hanning(n_slow)[:, None].astype(np.float32)
        x = x * np.hanning(n_samples)[None, :].astype(np.float32)

    range_fft = np.fft.fft(x, axis=1)
    rd = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)

    range_m = np.arange(n_samples) * float(params["range_res"])
    dp = tdm_doppler_params(params)
    doppler_mps = dp["velocity_axis"]
    rd_db = 20.0 * np.log10(np.abs(rd) + 1e-12)
    return rd_db.astype(np.float32), range_m.astype(np.float64), doppler_mps.astype(np.float64)


def compute_rd_from_frame(
    frame: np.ndarray,
    params: Dict[str, Any],
    *,
    tx_id: int = 0,
    rx_id: int = 0,
    declutter: bool = False,
    window: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    adc = frame_to_adc_cube(frame, params)
    return compute_rd_single_tx_rx(
        adc,
        params,
        tx_id=tx_id,
        rx_id=rx_id,
        declutter=declutter,
        window=window,
    )
