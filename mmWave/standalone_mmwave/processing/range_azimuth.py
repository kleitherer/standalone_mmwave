"""Range–azimuth maps: per-range Doppler peak + antenna FFT (same FFT as ``target_detect``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from capture_store import CaptureSession
from processing.angle_estimate import angle_axis_deg, angle_spectrum_fft
from processing.cube import frame_to_radar_cube
from processing.rda import compute_rda, range_doppler_axes


def doppler_idx_per_range(rd_roi: np.ndarray) -> np.ndarray:
    """
    Per range column: Doppler bin with largest SNR (decluttered dB − median).

    Returns shape (n_range,) indices into the Doppler axis of ``rd_roi``.
    """
    noise_floor = float(np.median(rd_roi))
    snr_map = rd_roi.astype(np.float64) - noise_floor
    return np.argmax(snr_map, axis=0).astype(np.intp)


def range_azimuth_power_frame(
    rda: np.ndarray,
    rd_db: np.ndarray,
    range_m: np.ndarray,
    r_mask: np.ndarray,
    r_idx: np.ndarray,
    *,
    angle_bins: int,
    fov_deg: float = 90.0,
) -> np.ndarray:
    """
    One frame: range × azimuth power (linear).

    At each range bin, pick the strongest Doppler column (per-range SNR peak), then
    run the same antenna FFT as ``target_detect`` on that (Doppler, range) cell.

    Returns (n_range, n_angle).
    """
    del range_m  # axis labels only; indexing uses r_mask / r_idx
    rd_roi = rd_db[:, r_mask]
    d_idxs = doppler_idx_per_range(rd_roi)
    n_range = len(r_idx)
    power = np.zeros((n_range, int(angle_bins)), dtype=np.float64)
    for j, ridx in enumerate(r_idx):
        d_idx = int(d_idxs[j])
        power[j, :] = angle_spectrum_fft(rda[d_idx, :, ridx], angle_bins, fov_deg)
    return power


def range_azimuth_power_db_frame(
    rda: np.ndarray,
    rd_db: np.ndarray,
    range_m: np.ndarray,
    r_mask: np.ndarray,
    r_idx: np.ndarray,
    *,
    angle_bins: int,
    fov_deg: float = 90.0,
) -> np.ndarray:
    """``range_azimuth_power_frame`` converted to dB."""
    power = range_azimuth_power_frame(
        rda, rd_db, range_m, r_mask, r_idx, angle_bins=angle_bins, fov_deg=fov_deg
    )
    return (10.0 * np.log10(power + 1e-12)).astype(np.float64)


def collect_range_azimuth_frames(
    capture_path: Path,
    *,
    range_gate_m: tuple[float, float],
    angle_bins: int = 128,
    fov_deg: float = 90.0,
    background_capture: Path | None = None,
    background_max_frames: int = 0,
    calibration_frames: int = 45,
    max_frames: int = 0,
    show_progress: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-frame range–azimuth maps for video export.

    Returns
    -------
    frames_db : list of (n_range, n_angle) dB maps
    range_m, angle_deg, time_s
    """
    from background_model import (
        estimate_rd_background_from_capture,
        estimate_rda_background_from_capture,
    )
    from processing.rd_map import frame_to_rd_power_db
    from post_processing.rd_maps import _frame_times

    session = CaptureSession.open(Path(capture_path))
    params = session.radar_params()
    paths = session.frame_paths()
    if max_frames > 0:
        paths = paths[:max_frames]
    if not paths:
        raise RuntimeError(f"No frames in {capture_path}")

    range_axis, _ = range_doppler_axes(params)
    r_lo, r_hi = range_gate_m
    r_mask = (range_axis >= r_lo) & (range_axis <= r_hi)
    r_idx = np.flatnonzero(r_mask)
    range_m = range_axis[r_mask]
    angle_deg = angle_axis_deg(angle_bins, fov_deg)

    rd_raw_list = []
    rda_list = []
    for i, p in enumerate(paths):
        raw = np.load(p)
        rd_raw_list.append(frame_to_rd_power_db(raw, params))
        rda_list.append(compute_rda(frame_to_radar_cube(raw, params)))
        if show_progress and (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{len(paths)} frames…", flush=True)

    if background_capture is not None:
        bg = estimate_rd_background_from_capture(
            Path(background_capture), params, max_frames=background_max_frames
        )
        bg_c = estimate_rda_background_from_capture(
            Path(background_capture), params, max_frames=background_max_frames
        )
    else:
        n_cal = min(len(rd_raw_list), max(1, int(calibration_frames)))
        bg = np.mean(np.stack(rd_raw_list[:n_cal], axis=0), axis=0)
        bg_c = np.mean(np.stack(rda_list[:n_cal], axis=0), axis=0)

    bg_ra_db = None
    if background_capture is not None:
        bg_session = CaptureSession.open(Path(background_capture))
        bg_paths = bg_session.frame_paths()
        if background_max_frames > 0:
            bg_paths = bg_paths[:background_max_frames]
        if bg_paths:
            bg_ra_acc = np.zeros((range_m.size, angle_bins), dtype=np.float64)
            for p in bg_paths:
                raw = np.load(p)
                rda_b = compute_rda(frame_to_radar_cube(raw, params)) - bg_c
                rd_b = frame_to_rd_power_db(raw, params) - bg
                bg_ra_acc += range_azimuth_power_db_frame(
                    rda_b,
                    rd_b,
                    range_m,
                    r_mask,
                    r_idx,
                    angle_bins=angle_bins,
                    fov_deg=fov_deg,
                )
            bg_ra_db = bg_ra_acc / len(bg_paths)

    frames_db: list[np.ndarray] = []
    for rd_raw, rda in zip(rd_raw_list, rda_list):
        rd_db = rd_raw - bg
        rda_d = rda - bg_c
        ra_db = range_azimuth_power_db_frame(
            rda_d,
            rd_db,
            range_m,
            r_mask,
            r_idx,
            angle_bins=angle_bins,
            fov_deg=fov_deg,
        )
        if bg_ra_db is not None:
            ra_db = ra_db - bg_ra_db
        frames_db.append(ra_db)

    time_s = _frame_times(session, len(frames_db))
    return frames_db, range_m, angle_deg, time_s


def build_range_azimuth_map(
    capture_path: Path,
    *,
    range_gate_m: tuple[float, float],
    angle_bins: int = 128,
    fov_deg: float = 90.0,
    background_capture: Path | None = None,
    background_max_frames: int = 0,
    calibration_frames: int = 45,
    max_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean range–azimuth power (dB), averaged over frames.

    Per frame and per range: strongest Doppler at that range → antenna FFT (live style).
    """
    from background_model import (
        estimate_rd_background_from_capture,
        estimate_rda_background_from_capture,
    )
    from processing.rd_map import frame_to_rd_power_db

    session = CaptureSession.open(Path(capture_path))
    params = session.radar_params()
    paths = session.frame_paths()
    if max_frames > 0:
        paths = paths[:max_frames]
    if not paths:
        raise RuntimeError(f"No frames in {capture_path}")

    range_axis, _ = range_doppler_axes(params)
    r_lo, r_hi = range_gate_m
    r_mask = (range_axis >= r_lo) & (range_axis <= r_hi)
    r_idx = np.flatnonzero(r_mask)
    range_m = range_axis[r_mask]
    if range_m.size == 0:
        raise RuntimeError("Range gate produced no bins for range-azimuth map")

    angle_deg = angle_axis_deg(angle_bins, fov_deg)
    n_range = range_m.size

    def _process(paths_in: list[Path]) -> np.ndarray:
        rd_raw_list = []
        rda_list = []
        for p in paths_in:
            raw = np.load(p)
            rd_raw_list.append(frame_to_rd_power_db(raw, params))
            rda_list.append(compute_rda(frame_to_radar_cube(raw, params)))

        if background_capture is not None:
            bg = estimate_rd_background_from_capture(
                Path(background_capture), params, max_frames=background_max_frames
            )
            bg_c = estimate_rda_background_from_capture(
                Path(background_capture), params, max_frames=background_max_frames
            )
        else:
            n_cal = min(len(rd_raw_list), max(1, int(calibration_frames)))
            bg = np.mean(np.stack(rd_raw_list[:n_cal], axis=0), axis=0)
            bg_c = np.mean(np.stack(rda_list[:n_cal], axis=0), axis=0)

        acc = np.zeros((n_range, angle_bins), dtype=np.float64)
        for rd_raw, rda in zip(rd_raw_list, rda_list):
            rd_db = rd_raw - bg
            rda_d = rda - bg_c
            acc += range_azimuth_power_frame(
                rda_d,
                rd_db,
                range_m,
                r_mask,
                r_idx,
                angle_bins=angle_bins,
                fov_deg=fov_deg,
            )
        return acc / max(len(paths_in), 1)

    power = _process(paths)
    if background_capture is not None:
        bg_session = CaptureSession.open(Path(background_capture))
        bg_paths = bg_session.frame_paths()
        if background_max_frames > 0:
            bg_paths = bg_paths[:background_max_frames]
        if bg_paths:
            bg_power = _process(bg_paths)
            power = np.maximum(power - bg_power, 1e-12)

    power_db = 10.0 * np.log10(power + 1e-12)
    return range_m, angle_deg, power_db
