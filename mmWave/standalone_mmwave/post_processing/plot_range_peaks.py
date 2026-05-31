#!/usr/bin/env python3
"""
Static-threshold range peak detection on ``range_time_snr`` data (FMCW-style).

Settings: ``range_peak_detection`` block in config/live_radar_to_max.json
(same parameters used by live/replay OSC).

Usage (from standalone_mmwave/):
  python3 -m post_processing.plot_range_peaks --capture no_presence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from post_processing.processing_config import ProcessingConfig, resolve_capture_range_gate
from processing.range_peak_detection import RangePeakDetectionConfig, detect_peaks_volume

_DEFAULT_CONFIG = _ROOT / "config" / "live_radar_to_max.json"


def _resolve_capture_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    direct = p.resolve()
    if direct.exists():
        return direct
    under = (_ROOT / "captures" / p).resolve()
    if under.exists():
        return under
    return direct


def _load_or_build_snr_volume(
    capture: Path,
    npz_path: Path,
    settings_path: Path,
    peak_cfg: RangePeakDetectionConfig,
) -> dict[str, Any]:
    if npz_path.is_file():
        data = np.load(npz_path)
        return {
            "snr_db": np.asarray(data["snr_db"], dtype=np.float64),
            "time_s": np.asarray(data["time_s"], dtype=np.float64),
            "range_m": np.asarray(data["range_m"], dtype=np.float64),
            "source": str(npz_path),
        }

    if not peak_cfg.build_if_missing:
        raise FileNotFoundError(
            f"Missing {npz_path}. Set range_peak_detection.analysis.build_if_missing=true "
            "or run plot_heatmap first."
        )

    from capture_store import CaptureSession
    from post_processing.rd_maps import build_range_time_volume

    proc = ProcessingConfig.load(settings_path if settings_path.is_file() else None)
    pp = proc.post_processing
    push_pull_cfg = json.loads(proc.config_path.read_text()).get("push_pull", {})
    params = CaptureSession.open(capture).radar_params()
    radar_max = float(params["range_max"])
    r_min, r_max, _ = resolve_capture_range_gate(capture, proc, radar_max_m=radar_max)

    print(f"Building range_time_snr.npz via {proc.config_path.name} …")
    vol = build_range_time_volume(
        capture,
        range_gate_m=(r_min, r_max),
        clutter_window=proc.declutter_mean_frames,
        background_capture=proc.background.capture,
        background_max_frames=proc.background.max_frames,
        angle_bins=proc.angle.fft_bins,
        angle_fov_deg=proc.angle.fov_deg,
        push_pull_mode=bool(push_pull_cfg.get("enabled", False)),
        push_pull_snr_within_db=float(push_pull_cfg.get("snr_within_db_of_max", 3.0)),
        snr_threshold_db=proc.snr_threshold_db,
        max_frames=peak_cfg.max_frames,
        limiter=pp.range_time_limiter,
        show_progress=True,
    )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        snr_db=vol.snr_db,
        time_s=vol.time_s,
        range_m=vol.range_m,
        snr_threshold_db=vol.snr_threshold_db,
        range_min_m=r_min,
        range_max_m=r_max,
    )
    print(f"  wrote {npz_path}")
    return {
        "snr_db": np.asarray(vol.snr_db, dtype=np.float64),
        "time_s": np.asarray(vol.time_s, dtype=np.float64),
        "range_m": np.asarray(vol.range_m, dtype=np.float64),
        "source": f"built → {npz_path}",
    }


def _profile_frame_indices(n_frames: int, cfg: RangePeakDetectionConfig) -> list[int]:
    if cfg.profile_frame_indices:
        return [i for i in cfg.profile_frame_indices if 0 <= i < n_frames]
    n = min(cfg.profile_count, n_frames)
    if n <= 1:
        return [0]
    return [int(round(i * (n_frames - 1) / (n - 1))) for i in range(n)]


def _plot_range_time_peaks(
    *,
    snr_db: np.ndarray,
    time_s: np.ndarray,
    range_m: np.ndarray,
    peak_range: np.ndarray,
    session_id: str,
    cfg: RangePeakDetectionConfig,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [time_s[0], time_s[-1] if len(time_s) > 1 else time_s[0] + 1, range_m[0], range_m[-1]]
    thr = cfg.snr_threshold_db
    vmax = float(np.percentile(snr_db, 99))
    vmin = max(thr - 3.0, float(np.percentile(snr_db, 5)))

    fig, ax = plt.subplots(figsize=cfg.figsize_range_time)
    im = ax.imshow(
        snr_db.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cfg.heatmap_cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    t_pts: list[float] = []
    r_pts: list[float] = []
    for fi in range(snr_db.shape[0]):
        t = float(time_s[fi])
        for j in range(cfg.max_peaks_per_frame):
            r = peak_range[fi, j]
            if np.isnan(r):
                break
            t_pts.append(t)
            r_pts.append(float(r))

    if t_pts:
        ax.scatter(
            t_pts,
            r_pts,
            s=cfg.peak_marker_size,
            c=cfg.peak_marker_color,
            edgecolors=cfg.peak_marker_edgecolor,
            linewidths=cfg.peak_marker_linewidth,
            marker="o",
            zorder=3,
            label=f"peaks ≥ {thr:.1f} dB",
        )
        ax.legend(loc="upper right", fontsize=9)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("range (m)")
    ax.set_title(
        f"{session_id} — static range peaks (≥ {thr:.1f} dB SNR, "
        f"sep ≥ {cfg.min_peak_separation_m:.2f} m)"
    )
    cb = fig.colorbar(im, ax=ax, label="SNR (dB)")
    cb.ax.axhline(thr, color=cfg.threshold_line_color, linewidth=1.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_range_peak_profiles(
    *,
    snr_db: np.ndarray,
    time_s: np.ndarray,
    range_m: np.ndarray,
    peak_range: np.ndarray,
    peak_snr: np.ndarray,
    frame_indices: list[int],
    session_id: str,
    cfg: RangePeakDetectionConfig,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(frame_indices)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=cfg.figsize_profiles, squeeze=False)
    thr = cfg.snr_threshold_db

    for k, fi in enumerate(frame_indices):
        ax = axes[k // ncols][k % ncols]
        profile = snr_db[fi]
        ax.plot(range_m, profile, color="0.35", linewidth=1.0, label="SNR profile")
        ax.axhline(thr, color=cfg.threshold_line_color, linestyle="--", linewidth=1.2, label="threshold")

        pr: list[float] = []
        ps: list[float] = []
        for j in range(cfg.max_peaks_per_frame):
            r = peak_range[fi, j]
            s = peak_snr[fi, j]
            if np.isnan(r):
                break
            pr.append(float(r))
            ps.append(float(s))
        if pr:
            ax.scatter(pr, ps, s=cfg.peak_marker_size * 1.5, c=cfg.peak_marker_edgecolor, zorder=3)

        ax.set_xlim(range_m[0], range_m[-1])
        ax.set_xlabel("range (m)")
        ax.set_ylabel("SNR (dB)")
        ax.set_title(f"frame {fi}  t={time_s[fi]:.2f} s")
        ax.grid(True, alpha=0.25)
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle(f"{session_id} — range profiles + static peak detection", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Static-threshold range peak plots (config/live_radar_to_max.json)"
    )
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Settings JSON (default: config/live_radar_to_max.json)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings_path = args.config.resolve()
    settings = json.loads(settings_path.read_text())
    peak_cfg = RangePeakDetectionConfig.from_settings(settings)

    capture = _resolve_capture_path(args.capture).resolve()
    if not capture.is_dir():
        print(f"ERROR: capture not found: {capture}")
        return 1

    out_dir = capture / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "range_time_snr.npz"

    print(f"Config: {settings_path}")
    print(f"  peak threshold: {peak_cfg.snr_threshold_db:.1f} dB")
    print(f"  min separation: {peak_cfg.min_peak_separation_m:.2f} m")
    print(f"  capture: {capture}")

    vol = _load_or_build_snr_volume(capture, npz_path, settings_path, peak_cfg)
    snr_db = vol["snr_db"]
    time_s = vol["time_s"]
    range_m = vol["range_m"]
    print(f"  SNR volume: {vol['source']}")

    peak_range, peak_snr, peak_count = detect_peaks_volume(snr_db, range_m, peak_cfg)
    n_det = int(np.sum(peak_count))
    print(f"  detected {n_det} peaks in {int(np.sum(peak_count > 0))}/{snr_db.shape[0]} frames")

    session_id = capture.name
    peaks_png = out_dir / "range_time_peaks.png"
    _plot_range_time_peaks(
        snr_db=snr_db,
        time_s=time_s,
        range_m=range_m,
        peak_range=peak_range,
        session_id=session_id,
        cfg=peak_cfg,
        out_path=peaks_png,
    )
    print(f"  {peaks_png}")

    if peak_cfg.write_profile_panels:
        prof_png = out_dir / "range_peak_profiles.png"
        _plot_range_peak_profiles(
            snr_db=snr_db,
            time_s=time_s,
            range_m=range_m,
            peak_range=peak_range,
            peak_snr=peak_snr,
            frame_indices=_profile_frame_indices(snr_db.shape[0], peak_cfg),
            session_id=session_id,
            cfg=peak_cfg,
            out_path=prof_png,
        )
        print(f"  {prof_png}")

    if peak_cfg.save_peaks_npz:
        peaks_npz = out_dir / "range_time_peaks.npz"
        np.savez_compressed(
            peaks_npz,
            peak_range_m=peak_range,
            peak_snr_db=peak_snr,
            peak_count=peak_count,
            time_s=time_s,
            range_m=range_m,
            snr_threshold_db=peak_cfg.snr_threshold_db,
            min_peak_separation_m=peak_cfg.min_peak_separation_m,
            max_peaks_per_frame=peak_cfg.max_peaks_per_frame,
            config_path=str(settings_path),
        )
        print(f"  {peaks_npz}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
