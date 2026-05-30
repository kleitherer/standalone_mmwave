"""Shared background model utilities for live and post-processing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from capture_store import CaptureSession
from processing.cube import frame_to_radar_cube
from processing.rda import compute_rda
from processing.rd_map import frame_to_rd_power_db


def frame_rd_power_db(
    frame_int16: np.ndarray,
    params: Dict,
    *,
    n_doppler_fft: int | None = None,
    n_range_fft: int | None = None,
    limiter: bool = False,
) -> np.ndarray:
    return frame_to_rd_power_db(
        frame_int16,
        params,
        declutter=True,
        window=True,
        n_doppler_fft=n_doppler_fft,
        n_range_fft=n_range_fft,
        limiter=limiter,
    )


def estimate_rd_background_mean(
    frame_iter: Iterable[np.ndarray],
    params: Dict,
    *,
    max_frames: int = 0,
    n_doppler_fft: int | None = None,
    n_range_fft: int | None = None,
    limiter: bool = False,
) -> np.ndarray:
    acc = None
    n = 0
    for frame in frame_iter:
        rd = frame_rd_power_db(
            frame,
            params,
            n_doppler_fft=n_doppler_fft,
            n_range_fft=n_range_fft,
            limiter=limiter,
        ).astype(np.float64)
        if acc is None:
            acc = np.zeros_like(rd, dtype=np.float64)
        acc += rd
        n += 1
        if max_frames > 0 and n >= max_frames:
            break
    if acc is None or n == 0:
        raise RuntimeError("No frames available for background estimation")
    return acc / float(n)


def estimate_rd_background_from_capture(
    capture_path: Path,
    params: Dict,
    *,
    max_frames: int = 0,
    n_doppler_fft: int | None = None,
    n_range_fft: int | None = None,
    limiter: bool = False,
) -> np.ndarray:
    session = CaptureSession.open(Path(capture_path))
    paths = session.frame_paths()
    if max_frames > 0:
        paths = paths[:max_frames]
    if not paths:
        raise RuntimeError(f"No frames found in background capture: {capture_path}")
    frames = (np.load(p) for p in paths)
    return estimate_rd_background_mean(
        frames,
        params,
        max_frames=0,
        n_doppler_fft=n_doppler_fft,
        n_range_fft=n_range_fft,
        limiter=limiter,
    )


def estimate_rda_background_from_capture(
    capture_path: Path,
    params: Dict,
    *,
    max_frames: int = 0,
) -> np.ndarray:
    """Mean complex range–Doppler–antenna cube from a background capture."""
    session = CaptureSession.open(Path(capture_path))
    paths = session.frame_paths()
    if max_frames > 0:
        paths = paths[:max_frames]
    if not paths:
        raise RuntimeError(f"No frames found in background capture: {capture_path}")

    acc = None
    n = 0
    for p in paths:
        rda = compute_rda(frame_to_radar_cube(np.load(p), params))
        if acc is None:
            acc = np.zeros_like(rda, dtype=np.complex128)
        acc += rda.astype(np.complex128)
        n += 1
    return acc / float(n)
