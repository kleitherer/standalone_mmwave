"""Range–azimuth maps via FFT or MUSIC on virtual-antenna snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Literal

import numpy as np

from capture_store import CaptureSession
from processing.cube import frame_to_radar_cube
from processing.rda import compute_rda, range_doppler_axes

AngleMethod = Literal["fft", "music"]


def angle_axis_deg(n_bins: int, fov_deg: float = 90.0) -> np.ndarray:
    """Azimuth grid: 0° at center, negative left, positive right."""
    half = float(fov_deg) / 2.0
    return np.linspace(-half, half, n_bins, dtype=np.float64)


def _steering_vector(n_ant: int, theta_deg: float) -> np.ndarray:
    """ULA steering vector (λ/2 spacing), broadside = 0°."""
    theta = np.deg2rad(theta_deg)
    k = np.arange(n_ant, dtype=np.float64)
    return np.exp(1j * np.pi * k * np.sin(theta))


def fft_azimuth_spectrum(snap: np.ndarray, n_bins: int, fov_deg: float = 90.0) -> np.ndarray:
    """|FFT|^2 spectrum over azimuth for one antenna snapshot."""
    spec = np.fft.fftshift(np.fft.fft(snap, n=n_bins))
    return (np.abs(spec) ** 2).astype(np.float64)


def music_azimuth_spectrum(
    snap: np.ndarray,
    *,
    n_bins: int,
    fov_deg: float = 90.0,
    n_sources: int = 1,
) -> np.ndarray:
    """
    Basic MUSIC pseudo-spectrum (ULA, λ/2).

    n_sources: number of assumed coherent sources (rest → noise subspace).
    """
    x = np.asarray(snap, dtype=np.complex128).ravel()
    n_ant = x.size
    if n_ant < 2:
        return np.zeros(n_bins, dtype=np.float64)

    r = np.outer(x, x.conj())
    eigvals, eigvecs = np.linalg.eigh(r)
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]
    n_noise = max(1, n_ant - int(n_sources))
    en = eigvecs[:, :n_noise]

    angles = angle_axis_deg(n_bins, fov_deg)
    out = np.zeros(n_bins, dtype=np.float64)
    for i, theta in enumerate(angles):
        a = _steering_vector(n_ant, theta)
        denom = np.linalg.norm(en.conj().T @ a) ** 2
        out[i] = 1.0 / (denom + 1e-12)
    return out


def azimuth_spectrum(
    snap: np.ndarray,
    *,
    method: AngleMethod,
    n_bins: int,
    fov_deg: float = 90.0,
) -> np.ndarray:
    if method == "music":
        return music_azimuth_spectrum(snap, n_bins=n_bins, fov_deg=fov_deg)
    return fft_azimuth_spectrum(snap, n_bins=n_bins, fov_deg=fov_deg)


def build_range_azimuth_map(
    capture_path: Path,
    *,
    range_gate_m: tuple[float, float],
    method: AngleMethod = "fft",
    angle_bins: int = 128,
    fov_deg: float = 90.0,
    background_capture: Path | None = None,
    background_max_frames: int = 0,
    max_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean range–azimuth power map (dB).

    Returns
    -------
    range_m : (n_range,)
    angle_deg : (n_angle,)  — 0° center, − left, + right
    power_db : (n_range, n_angle)
    """
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

    def _accumulate(paths_in: list[Path]) -> np.ndarray:
        acc = np.zeros((n_range, angle_bins), dtype=np.float64)
        for p in paths_in:
            raw = np.load(p)
            cube = frame_to_radar_cube(raw, params)
            rda = compute_rda(cube)
            for j, ridx in enumerate(r_idx):
                ant_vec = np.mean(rda[:, :, ridx], axis=0)
                spec = azimuth_spectrum(
                    ant_vec, method=method, n_bins=angle_bins, fov_deg=fov_deg
                )
                acc[j, :] += spec
        return acc / max(len(paths_in), 1)

    power = _accumulate(paths)

    if background_capture is not None:
        bg_session = CaptureSession.open(Path(background_capture))
        bg_paths = bg_session.frame_paths()
        if background_max_frames > 0:
            bg_paths = bg_paths[:background_max_frames]
        if bg_paths:
            bg_power = _accumulate(bg_paths)
            power = np.maximum(power - bg_power, 1e-12)

    power_db = 10.0 * np.log10(power + 1e-12)
    return range_m, angle_deg, power_db
