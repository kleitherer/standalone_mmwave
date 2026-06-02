"""RD power maps — same chain as ``utils/rd_heatmap_new`` / ``pipeline_utils.RD``."""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.pipeline_utils import RD, pw2db


def cube3_to_radar_cube_5d(cube3: np.ndarray, n_tx: int, n_rx: int) -> np.ndarray:
    """
    Expand live ``frame_to_radar_cube`` layout to ``(1, n_chirps, n_tx, n_rx, n_samples)``.

    ``cube3[slow, tx*n_rx + rx, sample]`` maps to chirp ``slow * n_tx + tx``.
    """
    n_slow, n_virt, n_samples = cube3.shape
    if n_virt != n_tx * n_rx:
        raise ValueError(f"expected {n_tx * n_rx} virtual antennas, got {n_virt}")

    cube4 = np.zeros((n_slow, n_tx, n_rx, n_samples), dtype=cube3.dtype)
    for t in range(n_tx):
        cube4[:, t, :, :] = cube3[:, t * n_rx : (t + 1) * n_rx, :]

    n_chirps = n_slow * n_tx
    cube5 = np.zeros((1, n_chirps, n_tx, n_rx, n_samples), dtype=cube3.dtype)
    for s in range(n_slow):
        for t in range(n_tx):
            c = s * n_tx + t
            cube5[0, c, t, :, :] = cube4[s, t, :, :]
    return cube5


def rd_axes_from_params(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    range_res = float(params["range_res"])
    velocity_res = float(params["velocity_res"])
    n_samples = int(params["n_samples"])
    n_chirps = int(params["n_chirps"])
    r_axis = np.arange(n_samples) * range_res
    d_axis = np.arange(-n_chirps // 2, n_chirps // 2) * velocity_res
    return r_axis, d_axis


def apply_range_gate_rd(
    rd_pw: np.ndarray,
    r_axis: np.ndarray,
    max_range_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep range columns with r <= max_range_m (after RD FFT)."""
    keep = r_axis <= max_range_m
    if not np.any(keep):
        raise ValueError(
            f"No range bins <= {max_range_m} m (r_axis spans {r_axis[0]:.3f} .. {r_axis[-1]:.3f})"
        )
    return rd_pw[..., keep], r_axis[keep]


def rd_power_from_radar_cube_5d(
    radar_cube: np.ndarray,
    params: dict[str, Any],
    *,
    declutter: bool = True,
    window: bool = True,
    max_range_m: float | None = None,
    return_rda: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Antenna-averaged linear RD power per frame.

    Returns ``(rd_pw, r_axis, d_axis)`` with ``rd_pw`` shape ``(n_frames, n_doppler, n_range)``.
    When ``return_rda=True``, also returns complex ``RDa`` with the same range gate applied.
    """
    RDa, _, _ = RD(radar_cube, declutter=declutter, window=window, quiet=True)
    rd_pw = (np.abs(RDa) ** 2).mean(axis=2).astype(np.float32)
    r_axis, d_axis = rd_axes_from_params(params)
    if max_range_m is not None:
        keep = r_axis <= max_range_m
        rd_pw = rd_pw[..., keep]
        RDa = RDa[..., keep]
        r_axis = r_axis[keep]
    if return_rda:
        return rd_pw, RDa, r_axis, d_axis
    return rd_pw, r_axis, d_axis


def rd_map_from_frame_int16(
    frame_int16: np.ndarray,
    params: dict[str, Any],
    *,
    declutter: bool = True,
    window: bool = True,
    max_range_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single-frame RD power + complex RDa ``(n_doppler, n_ant, n_range)``.

    Same ``pipeline_utils.RD`` chain as ``rd_heatmap_new``.
    """
    from processing.cube import frame_to_radar_cube

    n_tx = int(params["n_tx"])
    n_rx = int(params["n_rx"])
    cube3 = frame_to_radar_cube(frame_int16, params)
    cube5 = cube3_to_radar_cube_5d(cube3, n_tx, n_rx)
    rd_pw, RDa, r_axis, d_axis = rd_power_from_radar_cube_5d(
        cube5,
        params,
        declutter=declutter,
        window=window,
        max_range_m=max_range_m,
        return_rda=True,
    )
    return rd_pw[0], RDa[0], r_axis, d_axis


def rd_power_from_frame_int16(
    frame_int16: np.ndarray,
    params: dict[str, Any],
    *,
    declutter: bool = True,
    window: bool = True,
    max_range_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single-frame RD power map ``(n_doppler, n_range)`` — live ADC path.

    Uses the same ``pipeline_utils.RD`` chain as ``rd_heatmap_new`` on packed NPZ data.
    """
    rd_pw, _, r_axis, d_axis = rd_map_from_frame_int16(
        frame_int16,
        params,
        declutter=declutter,
        window=window,
        max_range_m=max_range_m,
    )
    return rd_pw, r_axis, d_axis


def peak_from_rd_power(
    rd_pw: np.ndarray,
    r_axis: np.ndarray,
    d_axis: np.ndarray,
    *,
    power_threshold_db: float = 0.0,
    max_abs_doppler_mps: float | None = None,
) -> tuple[int, int, float, float, float]:
    """
    Global max on antenna-averaged RD power (linear domain).

    Returns ``(d_idx, r_idx, range_m, doppler_mps, power_db)``.
    """
    if rd_pw.size == 0:
        raise ValueError("empty RD power map")

    rd_db = pw2db(rd_pw)
    search = rd_pw.astype(np.float64, copy=True)
    if power_threshold_db > 0.0:
        search[rd_db < power_threshold_db] = 0.0
    if max_abs_doppler_mps is not None and max_abs_doppler_mps > 0.0:
        keep_d = np.abs(d_axis) <= float(max_abs_doppler_mps)
        if np.any(keep_d):
            search[~keep_d, :] = 0.0

    d_idx, r_idx = np.unravel_index(int(np.argmax(search)), search.shape)
    return (
        int(d_idx),
        int(r_idx),
        float(r_axis[r_idx]),
        float(d_axis[d_idx]),
        float(rd_db[d_idx, r_idx]),
    )


def peak_candidates_from_rd_power(
    rd_pw: np.ndarray,
    r_axis: np.ndarray,
    d_axis: np.ndarray,
    *,
    power_threshold_db: float = 0.0,
    max_abs_doppler_mps: float | None = None,
    max_candidates: int = 8,
) -> list[tuple[int, int, float, float, float]]:
    """
    Sorted RD peak candidates (highest power first).

    Returns a list of ``(d_idx, r_idx, range_m, doppler_mps, power_db)``.
    """
    if rd_pw.size == 0:
        return []

    rd_db = pw2db(rd_pw)
    search = rd_pw.astype(np.float64, copy=True)
    if power_threshold_db > 0.0:
        search[rd_db < power_threshold_db] = 0.0
    if max_abs_doppler_mps is not None and max_abs_doppler_mps > 0.0:
        keep_d = np.abs(d_axis) <= float(max_abs_doppler_mps)
        if np.any(keep_d):
            search[~keep_d, :] = 0.0

    flat = search.ravel()
    if not np.any(flat > 0):
        d_idx, r_idx = np.unravel_index(int(np.argmax(search)), search.shape)
        return [
            (
                int(d_idx),
                int(r_idx),
                float(r_axis[r_idx]),
                float(d_axis[d_idx]),
                float(rd_db[d_idx, r_idx]),
            )
        ]

    n = max(1, int(max_candidates))
    top_idx = np.argpartition(flat, -n)[-n:]
    top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
    out: list[tuple[int, int, float, float, float]] = []
    for idx in top_idx:
        d_idx, r_idx = np.unravel_index(int(idx), search.shape)
        out.append(
            (
                int(d_idx),
                int(r_idx),
                float(r_axis[r_idx]),
                float(d_axis[d_idx]),
                float(rd_db[d_idx, r_idx]),
            )
        )
    return out
