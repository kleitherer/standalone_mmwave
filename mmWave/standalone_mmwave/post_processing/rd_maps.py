"""Build range–time SNR maps (no single-target peak tracking)."""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from capture_store import CaptureSession
from processing.cube import frame_to_radar_cube
from processing.rda import compute_rda, range_doppler_axes, rda_power_db


@dataclass
class RangeTimeVolume:
    """SNR (dB) vs range and time."""

    snr_db: np.ndarray  # (n_frames, n_range) — max over Doppler per frame
    time_s: np.ndarray
    range_m: np.ndarray
    doppler_mps: np.ndarray
    rd_stack: np.ndarray  # (n_frames, n_doppler, n_range) full SNR cube
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


def frame_rd_power_db(
    frame_int16: np.ndarray,
    params: Dict,
    *,
    declutter_time: deque[np.ndarray] | None = None,
) -> np.ndarray:
    """Range–Doppler power (dB), antennas averaged: shape (n_doppler, n_range)."""
    cube = frame_to_radar_cube(frame_int16, params)
    rd = rda_power_db(compute_rda(cube))
    if declutter_time is not None and len(declutter_time) >= 2:
        bg = np.median(np.stack(list(declutter_time), axis=0), axis=0)
        rd = rd - bg
    return rd


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

    history: deque[np.ndarray] = deque(maxlen=max(1, clutter_window))
    rd_list = []
    rt_list = []

    for i, path in enumerate(paths):
        raw = np.load(path)
        rd = frame_rd_power_db(raw, params, declutter_time=history)
        history.append(rd.copy())

        rd_roi = rd[:, r_mask]
        snr = rd_to_snr_db(rd_roi)
        rt_list.append(np.max(snr, axis=0))
        rd_list.append(snr)

        if show_progress and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(paths)} frames…", flush=True)

    if not rt_list:
        raise RuntimeError(f"No frames in {capture_path}")

    snr_rt = np.stack(rt_list, axis=0)
    rd_stack = np.stack(rd_list, axis=0)
    time_s = _frame_times(session, snr_rt.shape[0])

    session_id = session.root.name
    return RangeTimeVolume(
        snr_db=snr_rt,
        time_s=time_s,
        range_m=range_m,
        doppler_mps=doppler_mps,
        rd_stack=rd_stack,
        session_id=session_id,
        snr_threshold_db=snr_threshold_db,
    )
