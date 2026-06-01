"""Range peak list — body vs hands via multi-peak SNR profile (OSC + analysis)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RangePeakDetectionConfig:
    """``range_peak_detection`` — analysis plots and body (id=1) SNR gate only."""

    snr_threshold_db: float = 10.0
    min_peak_separation_m: float = 0.12
    max_peaks_per_frame: int = 8
    require_local_maximum: bool = True
    build_if_missing: bool = True
    max_frames: int = 0
    save_peaks_npz: bool = True
    write_profile_panels: bool = True
    dpi: int = 150
    figsize_range_time: tuple[float, float] = (12.0, 5.0)
    figsize_profiles: tuple[float, float] = (12.0, 8.0)
    profile_count: int = 6
    profile_frame_indices: tuple[int, ...] = ()
    heatmap_cmap: str = "viridis"
    peak_marker_color: str = "white"
    peak_marker_size: float = 10.0
    peak_marker_edgecolor: str = "crimson"
    peak_marker_linewidth: float = 0.8
    threshold_line_color: str = "crimson"

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> RangePeakDetectionConfig:
        block = settings.get("range_peak_detection", {})
        if not isinstance(block, dict):
            block = {}
        analysis = block.get("analysis", {})
        plot = block.get("plot", {})

        def _pair(key: str, default: tuple[float, float]) -> tuple[float, float]:
            v = plot.get(key, list(default))
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                return default
            return float(v[0]), float(v[1])

        indices = plot.get("profile_frame_indices", [])
        if indices is None:
            indices = []

        return cls(
            snr_threshold_db=float(block.get("snr_threshold_db", 10.0)),
            min_peak_separation_m=float(block.get("min_peak_separation_m", 0.12)),
            max_peaks_per_frame=max(1, int(block.get("max_peaks_per_frame", 8))),
            require_local_maximum=bool(block.get("require_local_maximum", True)),
            build_if_missing=bool(analysis.get("build_if_missing", True)),
            max_frames=int(analysis.get("max_frames", 0)),
            save_peaks_npz=bool(analysis.get("save_peaks_npz", True)),
            write_profile_panels=bool(analysis.get("write_profile_panels", True)),
            dpi=int(plot.get("dpi", 150)),
            figsize_range_time=_pair("figsize_range_time", (12.0, 5.0)),
            figsize_profiles=_pair("figsize_profiles", (12.0, 8.0)),
            profile_count=max(1, int(plot.get("profile_count", 6))),
            profile_frame_indices=tuple(int(i) for i in indices),
            heatmap_cmap=str(plot.get("heatmap_cmap", "viridis")),
            peak_marker_color=str(plot.get("peak_marker_color", "white")),
            peak_marker_size=float(plot.get("peak_marker_size", 10.0)),
            peak_marker_edgecolor=str(plot.get("peak_marker_edgecolor", "crimson")),
            peak_marker_linewidth=float(plot.get("peak_marker_linewidth", 0.8)),
            threshold_line_color=str(plot.get("threshold_line_color", "crimson")),
        )

    @property
    def body_snr_threshold_db(self) -> float:
        """Track id=1 (body) minimum SNR — same as ``snr_threshold_db``."""
        return float(self.snr_threshold_db)


def _local_maxima_indices(profile: np.ndarray, threshold_db: float) -> list[int]:
    peaks: list[int] = []
    n = int(profile.size)
    for i in range(n):
        v = float(profile[i])
        if v < threshold_db:
            continue
        left = float(profile[i - 1]) if i > 0 else -np.inf
        right = float(profile[i + 1]) if i + 1 < n else -np.inf
        if v >= left and v >= right:
            peaks.append(i)
    return peaks


def detect_range_peaks_frame(
    profile: np.ndarray,
    range_m: np.ndarray,
    cfg: RangePeakDetectionConfig,
    *,
    snr_threshold_db: float | None = None,
    min_peak_separation_m: float | None = None,
) -> list[tuple[int, float, float]]:
    """All peaks in one range profile, sorted by SNR (highest first)."""
    if profile.size == 0:
        return []

    thr = float(cfg.snr_threshold_db if snr_threshold_db is None else snr_threshold_db)
    if cfg.require_local_maximum:
        candidates = _local_maxima_indices(profile, thr)
    else:
        candidates = [int(i) for i in np.flatnonzero(profile >= thr)]

    if not candidates:
        return []

    candidates.sort(key=lambda i: float(profile[i]), reverse=True)
    min_sep = float(
        cfg.min_peak_separation_m if min_peak_separation_m is None else min_peak_separation_m
    )
    picked: list[tuple[int, float, float]] = []
    for idx in candidates:
        r = float(range_m[idx])
        if any(abs(r - pr) < min_sep for _, pr, _ in picked):
            continue
        picked.append((idx, r, float(profile[idx])))
        if len(picked) >= cfg.max_peaks_per_frame:
            break
    return picked


def detect_range_peaks_for_tracking(
    profile: np.ndarray,
    range_m: np.ndarray,
    peak_cfg: RangePeakDetectionConfig,
    track_cfg: Any,
) -> list[tuple[int, float, float]]:
    """
    Peak list for config1 tracker.

    Uses ``track_cfg.hand_snr_threshold_db`` for candidates (not body threshold).
    Keeps ``peak_cfg.min_peak_separation_m`` unchanged. Adds a hand-band peak
    in front of the strongest peak when one exists above the hand threshold.
    """
    hand_thr = float(track_cfg.hand_snr_threshold_db)
    peaks = detect_range_peaks_frame(
        profile,
        range_m,
        peak_cfg,
        snr_threshold_db=hand_thr,
    )
    if profile.size == 0 or not peaks:
        return peaks

    _i1, r_body, _ = peaks[0]
    band_min = float(track_cfg.min_hand_ahead_m)
    band_max = float(track_cfg.max_hand_ahead_m)

    best_hand: tuple[int, float, float] | None = None
    best_hand_snr = -np.inf
    if peak_cfg.require_local_maximum:
        hand_candidates = _local_maxima_indices(profile, hand_thr)
    else:
        hand_candidates = [int(i) for i in np.flatnonzero(profile >= hand_thr)]

    for idx in hand_candidates:
        r = float(range_m[idx])
        gap = float(r_body) - r
        if band_min <= gap <= band_max:
            snr = float(profile[idx])
            if snr > best_hand_snr:
                best_hand_snr = snr
                best_hand = (idx, r, snr)

    if best_hand is None:
        return peaks

    dup_tol = 0.05
    if any(abs(best_hand[1] - pr) < dup_tol for _, pr, _ in peaks):
        return peaks

    merged = list(peaks)
    merged.append(best_hand)
    merged.sort(key=lambda item: item[2], reverse=True)
    return merged[: peak_cfg.max_peaks_per_frame]


def peaks_from_profile(
    profile: np.ndarray,
    range_m: np.ndarray,
    peak_cfg: RangePeakDetectionConfig,
    *,
    snr_threshold_db: float | None = None,
    for_tracking: bool = False,
    track_cfg: Any = None,
) -> list[tuple[float, float]]:
    """
    Peak list as ``[(range_m, snr_db), ...]`` sorted by SNR (highest first).

    Analysis / plots: body ``snr_threshold_db`` only.
    ``for_tracking=True`` requires ``track_cfg`` and uses ``hand_snr_threshold_db``.
    """
    if for_tracking:
        if track_cfg is None:
            raise ValueError("track_cfg required when for_tracking=True")
        picked = detect_range_peaks_for_tracking(profile, range_m, peak_cfg, track_cfg)
    else:
        picked = detect_range_peaks_frame(
            profile, range_m, peak_cfg, snr_threshold_db=snr_threshold_db
        )
    return [(r, snr) for _idx, r, snr in picked]


def osc_targets_from_profile(
    profile: np.ndarray,
    range_m: np.ndarray,
    peak_cfg: RangePeakDetectionConfig,
    *,
    track_cfg: Any = None,
) -> list[tuple[float, float]]:
    """
    Legacy two-target list: strongest peak + best in hand band (config1 geometry).
    """
    peaks = detect_range_peaks_frame(profile, range_m, peak_cfg)
    if not peaks:
        return []

    _i1, r1, snr1 = peaks[0]
    out: list[tuple[float, float]] = [(r1, snr1)]
    if track_cfg is None:
        return out

    min_sep = float(track_cfg.min_hand_ahead_m)
    max_sep = float(track_cfg.max_hand_ahead_m)
    hand_thr = float(track_cfg.hand_snr_threshold_db)
    best: tuple[float, float] | None = None
    best_snr = -np.inf
    for _idx, r2, snr2 in peaks[1:]:
        if snr2 < hand_thr:
            continue
        gap = float(r1) - float(r2)
        if min_sep <= gap <= max_sep and snr2 > best_snr:
            best_snr = snr2
            best = (r2, snr2)
    if best is not None:
        out.append(best)
    return out


def detect_peaks_volume(
    snr_db: np.ndarray,
    range_m: np.ndarray,
    cfg: RangePeakDetectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_frames, _ = snr_db.shape
    max_p = cfg.max_peaks_per_frame
    peak_range = np.full((n_frames, max_p), np.nan, dtype=np.float64)
    peak_snr = np.full((n_frames, max_p), np.nan, dtype=np.float64)
    peak_count = np.zeros(n_frames, dtype=np.int32)

    for fi in range(n_frames):
        peaks = detect_range_peaks_frame(snr_db[fi], range_m, cfg)
        peak_count[fi] = len(peaks)
        for j, (_idx, r, s) in enumerate(peaks):
            peak_range[fi, j] = r
            peak_snr[fi, j] = s

    return peak_range, peak_snr, peak_count
