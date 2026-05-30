"""
mmw-tracking ``pipeline_utils.RD`` + ``rd_heatmap.py`` pipeline (ported verbatim).

Input cube layout matches ``radar_capture_utils.decode_data``:
``(n_frames, n_slow, n_tx, n_rx, n_samples)`` e.g. ``(1, 32, 3, 4, 256)``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from processing.adc_cube import frame_to_adc_cube


def pw2db(x: np.ndarray, scale: float = 10.0) -> np.ndarray:
    return scale * np.log10(np.abs(x) + 1e-9)


def frame_to_mmw_cube(
    frame: np.ndarray,
    params: Dict[str, Any],
    *,
    apply_rx_phase_bias: bool = False,
) -> np.ndarray:
    """
    One frame → ``(1, n_slow, n_tx, n_rx, n_samples)`` (mmw ``decode_data`` layout).

    Does **not** apply ``compRangeBiasAndRxChanPhase`` by default — mmw
    ``radarDataLoader.load_data()`` uses ``decode_data`` only (no dsp bias).
    """
    adc = frame_to_adc_cube(frame, params)
    n_tx = int(params["n_tx"])
    n_slow = adc.shape[0] // n_tx
    n_rx = adc.shape[1]
    cube = adc.reshape(n_slow, n_tx, n_rx, adc.shape[2])[None, ...]

    if apply_rx_phase_bias:
        from processing.cube import _rx_phase_bias_complex

        bias = _rx_phase_bias_complex(params.get("rx_phase_bias", []))
        if bias.size >= n_tx * n_rx:
            for t in range(n_tx):
                for r in range(n_rx):
                    cube[:, :, t, r, :] *= bias[t * n_rx + r]

    return cube.astype(np.complex64, copy=False)


def mmw_rd(
    radar_cube: np.ndarray,
    *,
    declutter: bool = True,
    window: bool = True,
) -> np.ndarray:
    """
    Range + Doppler FFT (mmw ``RD()`` else branch).

    Parameters
    ----------
    radar_cube : (n_frames, n_slow, n_tx, n_rx, n_samples)

    Returns
    -------
    RDa : (n_frames, n_slow, n_tx * n_rx, n_samples) complex
    """
    rda = np.asarray(radar_cube, dtype=np.complex64)
    if declutter:
        rda = rda - np.expand_dims(rda.mean(axis=1), axis=1)

    if window:
        w_range = np.hanning(rda.shape[-1]).astype(np.float32)
        w_doppler = np.hanning(rda.shape[1]).astype(np.float32)
        rda = rda * w_range[None, None, None, None, :]
        rda = rda * w_doppler[None, :, None, None, None]

    RDa = np.fft.fftshift(np.fft.fft(rda, axis=1), axes=1)
    RDa = np.fft.fft(RDa, axis=4)
    return RDa.reshape(
        RDa.shape[0],
        RDa.shape[1],
        RDa.shape[2] * RDa.shape[3],
        RDa.shape[4],
    )


def mmw_rd_power(
    RDa: np.ndarray,
    *,
    frame_index: int = 0,
    antenna_axis: int = 1,
) -> np.ndarray:
    """Linear power: ``mean(|RDa|²)`` over virtual antennas (``rd_heatmap.py``)."""
    frame = RDa[int(frame_index)]
    return np.mean(np.abs(frame) ** 2, axis=antenna_axis)


def mmw_rd_power_db(
    RDa: np.ndarray,
    *,
    frame_index: int = 0,
) -> np.ndarray:
    return pw2db(mmw_rd_power(RDa, frame_index=frame_index))


def mmw_range_doppler_axes(params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Axes from mmw ``radarDataLoader.load_data()`` + ``rd_heatmap.py`` extent.

    Uses ``velocity_res = 2*velocity_max / n_chirps`` (96), not ``n_slow`` (32).
    Doppler extent endpoints match a 32-bin RD row count when passed to ``imshow``.
    """
    n_samples = int(params["n_samples"])
    n_chirps = int(params["n_chirps"])
    n_tx = int(params["n_tx"])
    range_res = float(params["range_res"])
    operating_freq_ghz = float(params.get("operating_freq_ghz", 60.0))
    chirp_us = float(params.get("chirp_time_us", params.get("chirp_time", 139.0)))
    chirp_time_s = chirp_us * 1e-6 * n_tx
    velocity_max = (3e8 / (operating_freq_ghz * 1e9)) / (4.0 * chirp_time_s)
    velocity_res = (2.0 * velocity_max) / float(n_chirps)

    range_axis = np.arange(n_samples) * range_res
    doppler_axis = np.arange(-n_chirps // 2, n_chirps // 2) * velocity_res
    return range_axis, doppler_axis


def frame_to_mmw_rd_power_db(
    frame: np.ndarray,
    params: Dict[str, Any],
    *,
    declutter: bool = True,
    window: bool = True,
    apply_rx_phase_bias: bool = False,
) -> np.ndarray:
    """Single-frame RD power map (dB), mmw ``rd_heatmap.py`` chain."""
    cube = frame_to_mmw_cube(frame, params, apply_rx_phase_bias=apply_rx_phase_bias)
    RDa = mmw_rd(cube, declutter=declutter, window=window)
    return mmw_rd_power_db(RDa, frame_index=0)
