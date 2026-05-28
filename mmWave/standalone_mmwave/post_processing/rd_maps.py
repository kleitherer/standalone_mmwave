"""Build range–time SNR maps (no single-target peak tracking)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from capture_store import CaptureSession
from background_model import (
    estimate_rd_background_from_capture,
    estimate_rda_background_from_capture,
    frame_rd_power_db,
)
from processing.cube import frame_to_radar_cube
from processing.rda import compute_rda, range_doppler_axes
from processing.angle_estimate import (
    angle_axis_deg,
    angle_deg_at_rd_cell,
    angle_spectrum_fft,
)
from processing.target_detect import pick_rd_peak


@dataclass
class RangeTimeVolume:
    """SNR (dB) vs range and time."""

    snr_db: np.ndarray  # (n_frames, n_range) — max over Doppler per frame
    raw_db: np.ndarray  # (n_frames, n_range) — max over Doppler, no declutter
    declutter_db: np.ndarray  # (n_frames, n_range) — max over Doppler, decluttered
    time_s: np.ndarray
    range_m: np.ndarray
    doppler_mps: np.ndarray
    rd_stack: np.ndarray  # (n_frames, n_doppler, n_range) full SNR cube
    rd_raw_stack: np.ndarray  # (n_frames, n_doppler, n_range) raw RD power (dB)
    rd_declutter_stack: np.ndarray  # (n_frames, n_doppler, n_range) decluttered RD power (dB)
    doppler_time_snr: np.ndarray  # (n_frames, n_doppler) — max over range
    azimuth_time_snr: np.ndarray  # (n_frames, n_angle) — antenna FFT at RD peak cell
    angle_time_deg: np.ndarray  # (n_frames,) — peak angle (same as live target)
    angle_deg: np.ndarray
    session_id: str
    snr_threshold_db: float

    def mask(self) -> np.ndarray:
        return self.snr_db >= self.snr_threshold_db


def _frame_times(session: CaptureSession, n: int) -> np.ndarray:
    index_path = session.root / "index.csv"
    fps = 1000.0 / float(session.radar_params().get("frame_time", 22.22))
    if index_path.is_file():
        rows = list(csv.DictReader(index_path.open()))
        if len(rows) >= n:
            t0 = float(rows[0]["timestamp_unix"])
            return np.array([float(r["timestamp_unix"]) - t0 for r in rows[:n]])
    return (np.arange(n, dtype=np.float64)) / fps


def rd_to_snr_db(rd_db: np.ndarray, noise_db: float | None = None) -> np.ndarray:
    """SNR in dB relative to noise floor (default: per-frame median)."""
    if noise_db is None:
        noise_db = float(np.median(rd_db))
    return rd_db - noise_db


def build_range_time_volume(
    capture_path: Path,
    *,
    range_gate_m: Tuple[float, float] = (0.5, 12.0),
    clutter_window: int = 16,
    background_capture: Path | None = None,
    background_max_frames: int = 0,
    angle_bins: int = 128,
    angle_fov_deg: float = 90.0,
    push_pull_mode: bool = False,
    push_pull_snr_within_db: float = 3.0,
    snr_threshold_db: float = 8.0,
    max_frames: int = 0,
    show_progress: bool = True,
) -> RangeTimeVolume:
    """
    Stack per-frame range–Doppler SNR into a range–time image.

    Each time slice: SNR(d,r) = RD_power(d,r) - noise, then max over Doppler → SNR(r).
    """
    session = CaptureSession.open(Path(capture_path))
    params = session.radar_params()
    range_axis, doppler_axis = range_doppler_axes(params)

    r_lo, r_hi = range_gate_m
    r_mask = (range_axis >= r_lo) & (range_axis <= r_hi)
    range_m = range_axis[r_mask]
    doppler_mps = doppler_axis

    paths = session.frame_paths()
    if max_frames > 0:
        paths = paths[:max_frames]

    rd_raw_list = []
    rda_complex_list = []
    rd_list = []
    rd_raw_roi_list = []
    rd_declutter_roi_list = []
    rt_list = []
    raw_rt_list = []
    declutter_rt_list = []
    azimuth_time_list = []
    angle_time_list = []

    angle_deg = angle_axis_deg(angle_bins, angle_fov_deg)

    for i, path in enumerate(paths):
        raw = np.load(path)
        rd_raw = frame_rd_power_db(raw, params)
        rd_raw_list.append(rd_raw)
        rda_complex_list.append(compute_rda(frame_to_radar_cube(raw, params)))

        if show_progress and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(paths)} frames…", flush=True)

    if not rd_raw_list:
        raise RuntimeError(f"No frames in {capture_path}")

    if background_capture is not None:
        bg = estimate_rd_background_from_capture(
            Path(background_capture), params, max_frames=background_max_frames
        )
        bg_c = estimate_rda_background_from_capture(
            Path(background_capture), params, max_frames=background_max_frames
        )
    else:
        n_calib = min(len(rd_raw_list), max(1, int(clutter_window)))
        bg = np.mean(np.stack(rd_raw_list[:n_calib], axis=0), axis=0)
        bg_c = np.mean(np.stack(rda_complex_list[:n_calib], axis=0), axis=0)

    rd_declutter_list = [rd_raw - bg for rd_raw in rd_raw_list]
    rda_declutter_list = [rda - bg_c for rda in rda_complex_list]

    for rd_raw, rd_declutter, rda_d in zip(rd_raw_list, rd_declutter_list, rda_declutter_list):
        rd_roi_raw = rd_raw[:, r_mask]
        rd_roi_declutter = rd_declutter[:, r_mask]
        snr = rd_to_snr_db(rd_roi_declutter)
        rt_list.append(np.max(snr, axis=0))
        raw_rt_list.append(np.max(rd_roi_raw, axis=0))
        declutter_rt_list.append(np.max(rd_roi_declutter, axis=0))
        rd_raw_roi_list.append(rd_roi_raw)
        rd_declutter_roi_list.append(rd_roi_declutter)
        rd_list.append(snr)

        d_idx, r_local, _, _ = pick_rd_peak(
            rd_roi_declutter,
            range_m,
            push_pull_mode=push_pull_mode,
            push_pull_snr_within_db=push_pull_snr_within_db,
        )
        r_idx_peak = int(np.flatnonzero(r_mask)[r_local])
        snap = rda_d[d_idx, :, r_idx_peak]
        spec = angle_spectrum_fft(snap, angle_bins, angle_fov_deg)
        spec_db = 10.0 * np.log10(spec + 1e-12)
        azimuth_time_list.append(spec_db - float(np.median(spec_db)))
        angle_time_list.append(
            angle_deg_at_rd_cell(
                rda_d, d_idx, r_idx_peak, n_bins=angle_bins, fov_deg=angle_fov_deg
            )
        )

    snr_rt = np.stack(rt_list, axis=0)
    raw_rt = np.stack(raw_rt_list, axis=0)
    declutter_rt = np.stack(declutter_rt_list, axis=0)
    rd_stack = np.stack(rd_list, axis=0)
    rd_raw_stack = np.stack(rd_raw_roi_list, axis=0)
    rd_declutter_stack = np.stack(rd_declutter_roi_list, axis=0)
    doppler_time_snr = np.max(rd_stack, axis=2)
    azimuth_time_snr = np.stack(azimuth_time_list, axis=0)
    angle_time_deg = np.asarray(angle_time_list, dtype=np.float64)
    time_s = _frame_times(session, snr_rt.shape[0])

    session_id = session.root.name
    return RangeTimeVolume(
        snr_db=snr_rt,
        raw_db=raw_rt,
        declutter_db=declutter_rt,
        time_s=time_s,
        range_m=range_m,
        doppler_mps=doppler_mps,
        rd_stack=rd_stack,
        rd_raw_stack=rd_raw_stack,
        rd_declutter_stack=rd_declutter_stack,
        doppler_time_snr=doppler_time_snr,
        azimuth_time_snr=azimuth_time_snr,
        angle_time_deg=angle_time_deg,
        angle_deg=angle_deg,
        session_id=session_id,
        snr_threshold_db=snr_threshold_db,
    )
