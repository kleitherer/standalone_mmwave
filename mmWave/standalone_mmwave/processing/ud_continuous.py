"""Micro-Doppler from a full radar cube."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, filtfilt, stft

# Tuned on captures/potential walk (per_frame uD + strong slow-time high-pass).
WALK_UD_PRESET: dict = {
    "declutter": "per_frame",
    "highpass_hz": 20.0,
    "zero_doppler_notch_bins": 10,
    "zero_doppler_guard_mps": 0.02,
    "antenna": "tx0_rx0",
    "stft_mode": "continuous",
    "normalize_columns": True,
}
DeclutterMode = Literal["ema", "running", "per_frame", "smooth_frame", "global", "none"]
AntennaMode = Literal["mean", "tx0_rx0"]
StftMode = Literal["continuous", "per_frame"]


def _collapse_antenna(rda: np.ndarray, antenna: AntennaMode) -> np.ndarray:
    if antenna == "mean":
        return rda.mean(axis=(2, 3))
    return rda[:, :, 0, 0, :]


def _range_bin_mask(n_samples: int, range_res: float, r_lo: float | None, r_hi: float | None) -> np.ndarray:
    r_axis = np.arange(n_samples) * range_res
    lo = r_lo if r_lo is not None else r_axis[0]
    hi = r_hi if r_hi is not None else r_axis[-1]
    return (r_axis >= lo) & (r_axis <= hi)


def _subtract_running_clutter(x: np.ndarray, window_chirps: int) -> np.ndarray:
    w = max(3, int(window_chirps))
    if w % 2 == 0:
        w += 1
    re = uniform_filter1d(x.real, size=w, axis=0, mode="nearest")
    im = uniform_filter1d(x.imag, size=w, axis=0, mode="nearest")
    return x - (re + 1j * im)


def _ema_clutter(x: np.ndarray, alpha: float) -> np.ndarray:
    """Chirp-by-chirp exponential clutter map (continuous, adapts over time)."""
    out = np.empty_like(x)
    clutter = x[0].copy()
    for k in range(x.shape[0]):
        out[k] = x[k] - clutter
        clutter = alpha * clutter + (1.0 - alpha) * x[k]
    return out


def _highpass_slow_time(x: np.ndarray, fs_hz: float, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0 or x.shape[0] < 16:
        return x
    wn = min(0.99, cutoff_hz / (fs_hz / 2.0))
    if wn <= 0:
        return x
    b, a = butter(2, wn, btype="high")
    out = np.empty_like(x)
    for col in range(x.shape[1]):
        out[:, col] = filtfilt(b, a, x[:, col])
    return out


def _suppress_zero_doppler_ridge(uD: np.ndarray, half_width_bins: int) -> np.ndarray:
    """Remove residual static clutter ridge at 0 m/s (per time column)."""
    if half_width_bins <= 0:
        return uD
    c = uD.shape[0] // 2
    lo = max(0, c - half_width_bins)
    hi = min(uD.shape[0], c + half_width_bins + 1)
    ridge = np.median(uD[lo:hi], axis=0, keepdims=True)
    return uD - ridge


def _guard_zero_doppler(
    uD: np.ndarray,
    *,
    n_uD_fft: int,
    velocity_max: float,
    guard_mps: float,
) -> np.ndarray:
    """Replace |Doppler| < guard with off-center median (kills 0 m/s pulses)."""
    if guard_mps <= 0:
        return uD
    axis = np.abs(uD_axis_mps(n_uD_fft, velocity_max))
    moving = axis >= guard_mps
    if not np.any(moving):
        return uD
    fill = np.median(uD[moving], axis=0, keepdims=True)
    out = uD.copy()
    out[~moving] = fill
    return out


def _mti_slow_time(rd: np.ndarray) -> np.ndarray:
    """Chirp-to-chirp cancelation along slow-time (within continuous stream)."""
    n_frames, n_chirps, n_samples = rd.shape
    x = rd.reshape(n_frames * n_chirps, n_samples)
    out = np.empty_like(x)
    out[0] = 0.0
    out[1:] = x[1:] - x[:-1]
    return out.reshape(n_frames, n_chirps, n_samples)


def _declutter_slow_time(
    rd: np.ndarray,
    mode: DeclutterMode,
    *,
    running_window_chirps: int,
    ema_alpha: float,
    chirp_rate_hz: float,
    highpass_hz: float,
) -> np.ndarray:
    if mode == "per_frame":
        rd = rd - np.expand_dims(rd.mean(axis=1), axis=1)
    elif mode == "smooth_frame":
        frame_mean = rd.mean(axis=1)
        smooth = gaussian_filter1d(frame_mean, sigma=1.0, axis=0, mode="nearest")
        rd = rd - smooth[:, None, :]
    elif mode == "global":
        rd = rd - rd.mean(axis=(0, 1), keepdims=True)
    elif mode in ("running", "ema"):
        n_frames, n_chirps, n_samples = rd.shape
        x = rd.reshape(n_frames * n_chirps, n_samples)
        if mode == "running":
            x = _subtract_running_clutter(x, running_window_chirps)
        else:
            x = _ema_clutter(x, ema_alpha)
        rd = x.reshape(n_frames, n_chirps, n_samples)

    if highpass_hz > 0:
        n_frames, n_chirps, n_samples = rd.shape
        x = rd.reshape(n_frames * n_chirps, n_samples)
        x = _highpass_slow_time(x, chirp_rate_hz, highpass_hz)
        rd = x.reshape(n_frames, n_chirps, n_samples)

    return rd


def _stft_uD(
    rd_bbox: np.ndarray,
    *,
    n_uD_fft: int,
    overlap_ratio: float,
    uD_window: str,
    stft_mode: StftMode,
    n_slow: int,
) -> np.ndarray:
    w_doppler = np.hanning(rd_bbox.shape[1]).astype(np.float32)
    w_range = np.hanning(rd_bbox.shape[2]).astype(np.float32)
    rd_bbox = rd_bbox * w_doppler[None, :, None] * w_range[None, None, :]

    if stft_mode == "continuous":
        x = rd_bbox.reshape(-1, rd_bbox.shape[-1])
        _, _, zxx = stft(
            x,
            nfft=n_uD_fft,
            nperseg=n_uD_fft,
            noverlap=int(overlap_ratio * n_uD_fft),
            window=uD_window,
            return_onesided=False,
            axis=0,
        )
        zxx = np.fft.fftshift(zxx, 0).transpose(0, 2, 1)
        return (20 * np.log(np.mean(np.abs(zxx), -1) + 1e-9)).astype(np.float32)

    # STFT inside each radar frame only — no chirps concatenated across frames.
    n_frames = rd_bbox.shape[0]
    nperseg = n_slow
    noverlap = int(overlap_ratio * nperseg)
    cols: list[np.ndarray] = []
    for f in range(n_frames):
        x = rd_bbox[f]
        if x.shape[0] < nperseg:
            continue
        _, _, zxx = stft(
            x,
            nfft=n_uD_fft,
            nperseg=nperseg,
            noverlap=noverlap,
            window=uD_window,
            return_onesided=False,
            axis=0,
        )
        zxx = np.fft.fftshift(zxx, 0).transpose(0, 2, 1)
        cols.append(20 * np.log(np.mean(np.abs(zxx), -1) + 1e-9))
    if not cols:
        raise ValueError("per_frame STFT produced no columns")
    return np.concatenate(cols, axis=1).astype(np.float32)


def micro_doppler_from_cube(
    radar_cube: np.ndarray,
    *,
    n_uD_fft: int = 128,
    overlap_ratio: float = 0.875,
    uD_window: str = "hann",
    declutter: DeclutterMode = "smooth_frame",
    antenna: AntennaMode = "tx0_rx0",
    stft_mode: StftMode = "continuous",
    range_res: float = 0.075,
    range_gate_m: tuple[float | None, float | None] = (None, None),
    running_clutter_sec: float = 1.0,
    ema_tau_sec: float = 0.5,
    n_slow: int = 32,
    fps: float = 45.0,
    velocity_max: float = 3.0,
    highpass_hz: float = 0.3,
    zero_doppler_notch_bins: int = 0,
    zero_doppler_guard_mps: float = 0.0,
    mti: bool = False,
    normalize_columns: bool = True,
) -> np.ndarray:
    rda = np.asarray(radar_cube, dtype=np.complex64)
    chirp_rate_hz = float(n_slow * fps)
    running_window_chirps = max(3, int(running_clutter_sec * chirp_rate_hz))
    ema_alpha = float(np.exp(-1.0 / (max(ema_tau_sec, 1e-3) * chirp_rate_hz)))

    rd_bbox = _collapse_antenna(rda, antenna)
    rd_bbox = _declutter_slow_time(
        rd_bbox,
        declutter,
        running_window_chirps=running_window_chirps,
        ema_alpha=ema_alpha,
        chirp_rate_hz=chirp_rate_hz,
        highpass_hz=highpass_hz,
    )

    if mti:
        rd_bbox = _mti_slow_time(rd_bbox)

    r_lo, r_hi = range_gate_m
    r_mask = _range_bin_mask(rd_bbox.shape[-1], range_res, r_lo, r_hi)
    rd_bbox = rd_bbox[..., r_mask]
    if rd_bbox.shape[-1] == 0:
        raise ValueError(f"Empty range gate {range_gate_m}")

    uD = _stft_uD(
        rd_bbox,
        n_uD_fft=n_uD_fft,
        overlap_ratio=overlap_ratio,
        uD_window=uD_window,
        stft_mode=stft_mode,
        n_slow=n_slow,
    )

    if zero_doppler_notch_bins > 0:
        uD = _suppress_zero_doppler_ridge(uD, zero_doppler_notch_bins)

    if normalize_columns:
        uD = uD - np.median(uD, axis=0, keepdims=True)

    if zero_doppler_guard_mps > 0:
        uD = _guard_zero_doppler(
            uD,
            n_uD_fft=n_uD_fft,
            velocity_max=velocity_max,
            guard_mps=zero_doppler_guard_mps,
        )

    return uD.astype(np.float32)


def uD_axis_mps(n_uD_fft: int, velocity_max: float) -> np.ndarray:
    return (
        np.arange(-n_uD_fft // 2, n_uD_fft // 2)
        * 2
        * velocity_max
        / n_uD_fft
    )


def uD_bins_per_second(
    n_chirps: int,
    n_uD_fft: int,
    overlap_ratio: float,
    fps: float,
    *,
    stft_mode: StftMode = "continuous",
) -> float:
    if stft_mode == "per_frame":
        hop = max(1, n_chirps - int(overlap_ratio * n_chirps))
        return fps * (n_chirps - int(overlap_ratio * n_chirps)) / hop
    return float(int(n_chirps / (n_uD_fft * (1 - overlap_ratio)) * fps))
