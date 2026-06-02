#!/usr/bin/env python3
"""
Plot config3 OSC outputs (range_m, doppler_mps) from a capture or packed NPZ.

Uses the same pipeline as live/replay config3 (``RdPeakProcessor``).

Usage (from standalone_mmwave/):
  # Time-series PNG + RD snapshot at one frame
  python3 utils/plot_config3_osc.py --capture testbonk --frame 164

  # Movie: RD heatmap with moving peak marker (+ trace inset)
  python3 utils/plot_config3_osc.py --capture testbonk

  python3 utils/plot_config3_osc.py --capture testbonk --format gif --max-frames 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from capture_store import CaptureSession
from gesture_recognition.mode import active_gesture_mode
from gesture_recognition.mode import make_gesture_processor
from gesture_recognition.peaks import RangePeakDetectionConfig
from gesture_recognition.rd_peak import (
    Config3OscVolume,
    RdKalmanProcessor,
    RdPeakProcessor,
    process_config3_frames,
)
from live_radar_to_max import _load_settings, _resolve_capture_path
from utils.pipeline_utils import pw2db
from utils.rd_power import rd_map_from_frame_int16

DEFAULT_VMIN_DB = 50.0
DEFAULT_VMAX_DB = 95.0
DEFAULT_DOPPLER_PLOT_LIMIT_MPS = 3.0


def _load_frames_from_capture(capture_path: Path, max_frames: int) -> list[np.ndarray]:
    session = CaptureSession.open(capture_path)
    frames = [frame for _, frame in session.iter_frames()]
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def _build_volume(
    frames: list[np.ndarray],
    params: dict,
    settings: dict,
) -> tuple[Config3OscVolume, RdPeakProcessor | RdKalmanProcessor, RangePeakDetectionConfig]:
    peak_cfg = RangePeakDetectionConfig.from_settings(settings)
    processor = make_gesture_processor(settings)
    if not isinstance(processor, (RdPeakProcessor, RdKalmanProcessor)):
        raise ValueError(
            "This script requires gesture.mode=config3/config4/config5 "
            "(RD-peak processor modes)."
        )
    frame_time_ms = float(params.get("frame_time", 22.22))
    vol = process_config3_frames(
        frames,
        params,
        processor,
        peak_cfg_snr_threshold_db=peak_cfg.body_snr_threshold_db,
        frame_time_ms=frame_time_ms,
    )
    return vol, processor, peak_cfg


def _mask_present(vol: Config3OscVolume) -> tuple[np.ndarray, np.ndarray]:
    """Return (range, doppler) with NaN where presence is false."""
    r = vol.range_m.copy()
    d = vol.doppler_mps.copy()
    r[~vol.present] = np.nan
    d[~vol.present] = np.nan
    return r, d


def save_tracks_npz(out_path: Path, vol: Config3OscVolume) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        time_s=vol.time_s,
        range_m=vol.range_m,
        doppler_mps=vol.doppler_mps,
        snr_db=vol.snr_db,
        present=vol.present,
        angle_deg=vol.angle_deg,
        x_m=vol.x_m,
        y_m=vol.y_m,
        doppler_idx=vol.doppler_idx,
        range_idx=vol.range_idx,
    )


def plot_frame_timeseries(
    vol: Config3OscVolume,
    frames: list[np.ndarray],
    params: dict,
    processor: RdPeakProcessor,
    *,
    frame_idx: int,
    out_path: Path,
    vmin_db: float,
    vmax_db: float,
) -> None:
    """Range & Doppler vs time, with RD snapshot at ``frame_idx``."""
    n = len(vol.time_s)
    if frame_idx < 0 or frame_idx >= n:
        raise ValueError(f"--frame must be in [0, {n - 1}]")

    r_pres, d_pres = _mask_present(vol)
    t = vol.time_s
    hi = frame_idx

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.4], hspace=0.32)

    ax_r = fig.add_subplot(gs[0])
    ax_d = fig.add_subplot(gs[1])
    ax_rd = fig.add_subplot(gs[2])

    ax_r.plot(t[: hi + 1], vol.range_m[: hi + 1], color="0.35", lw=0.8, alpha=0.5)
    ax_r.plot(t[: hi + 1], r_pres[: hi + 1], "C0", lw=1.5, marker=".", ms=3)
    ax_r.axvline(t[hi], color="crimson", ls="--", lw=1)
    ax_r.scatter([t[hi]], [vol.range_m[hi]], c="crimson", s=40, zorder=5)
    ax_r.set_ylabel("range_m (OSC)")
    ax_r.set_xlim(t[0], t[-1] if n > 1 else t[0] + 1e-3)
    ax_r.grid(True, alpha=0.3)

    ax_d.plot(t[: hi + 1], vol.doppler_mps[: hi + 1], color="0.35", lw=0.8, alpha=0.5)
    ax_d.plot(t[: hi + 1], d_pres[: hi + 1], "C1", lw=1.5, marker=".", ms=3)
    ax_d.axvline(t[hi], color="crimson", ls="--", lw=1)
    ax_d.scatter([t[hi]], [vol.doppler_mps[hi]], c="crimson", s=40, zorder=5)
    ax_d.set_ylabel("doppler_mps (OSC)")
    ax_d.set_xlim(t[0], t[-1] if n > 1 else t[0] + 1e-3)
    ax_d.grid(True, alpha=0.3)

    frame = frames[hi]
    rd_pw, _, r_axis, d_axis = rd_map_from_frame_int16(
        frame,
        params,
        declutter=processor.cfg.declutter,
        window=processor.cfg.window,
        max_range_m=processor.cfg.max_range_m,
    )
    rd_db = pw2db(rd_pw)
    extent = [float(r_axis[0]), float(r_axis[-1]), float(d_axis[-1]), float(d_axis[0])]
    im = ax_rd.imshow(
        rd_db,
        aspect="auto",
        cmap="jet",
        origin="upper",
        extent=extent,
        vmin=vmin_db,
        vmax=vmax_db,
    )
    di = int(vol.doppler_idx[hi])
    ri = int(vol.range_idx[hi])
    if di >= 0 and ri >= 0:
        ax_rd.scatter(
            [vol.range_m[hi]],
            [vol.doppler_mps[hi]],
            s=80,
            facecolors="none",
            edgecolors="white",
            linewidths=2,
            zorder=5,
        )
        ax_rd.scatter(
            [vol.range_m[hi]],
            [vol.doppler_mps[hi]],
            s=20,
            c="crimson",
            zorder=6,
        )
    ax_rd.set_xlabel("Range (m)")
    ax_rd.set_ylabel("Doppler (m/s)")
    ax_rd.set_title(
        f"RD @ frame {hi}  "
        f"R={vol.range_m[hi]:.2f}m  V={vol.doppler_mps[hi]:+.2f}m/s  "
        f"SNR={vol.snr_db[hi]:.1f}dB  present={bool(vol.present[hi])}"
    )
    fig.colorbar(im, ax=ax_rd, label="Power (dB)", fraction=0.046)

    fig.suptitle(
        f"Config3 OSC tracks through t={t[hi]:.2f}s (frame {hi}/{n - 1})",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_osc_movie(
    vol: Config3OscVolume,
    frames: list[np.ndarray],
    params: dict,
    processor: RdPeakProcessor,
    *,
    out_path: Path,
    fps: float,
    fmt: str,
    dpi: int,
    vmin_db: float,
    vmax_db: float,
) -> None:
    """RD movie with config3 peak marker; inset traces range & doppler over time."""
    n = len(frames)
    r_pres, d_pres = _mask_present(vol)
    t = vol.time_s

    # Precompute RD dB stack (needed for smooth animation)
    rd_stack: list[np.ndarray] = []
    r_axis = d_axis = None
    for frame in frames:
        rd_pw, _, r_axis, d_axis = rd_map_from_frame_int16(
            frame,
            params,
            declutter=processor.cfg.declutter,
            window=processor.cfg.window,
            max_range_m=processor.cfg.max_range_m,
        )
        rd_stack.append(pw2db(rd_pw))
    assert r_axis is not None and d_axis is not None
    d_plot_lim = float(DEFAULT_DOPPLER_PLOT_LIMIT_MPS)
    d_keep = np.abs(d_axis) <= d_plot_lim
    d_axis = d_axis[d_keep]
    rd_stack = [rd[d_keep, :] for rd in rd_stack]
    extent = [float(r_axis[0]), float(r_axis[-1]), float(d_axis[-1]), float(d_axis[0])]

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], width_ratios=[1.6, 1], hspace=0.35, wspace=0.28)
    ax_rd = fig.add_subplot(gs[0, :])
    ax_r = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    im = ax_rd.imshow(
        rd_stack[0],
        aspect="auto",
        cmap="jet",
        origin="upper",
        extent=extent,
        vmin=vmin_db,
        vmax=vmax_db,
    )
    peak_sc = ax_rd.scatter([], [], s=80, facecolors="none", edgecolors="white", linewidths=2)
    peak_dot = ax_rd.scatter([], [], s=25, c="crimson")
    peak_angle = ax_rd.annotate(
        "",
        (0.0, 0.0),
        xytext=(6, 6),
        textcoords="offset points",
        color="white",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6),
    )
    ax_rd.set_xlabel("Range (m)")
    ax_rd.set_ylabel("Doppler (m/s)")
    title = ax_rd.set_title("")
    fig.colorbar(im, ax=ax_rd, label="Power (dB)", fraction=0.02)

    line_r, = ax_r.plot([], [], "C0", lw=1.5)
    cur_r, = ax_r.plot([], [], "C0", marker="o", ms=4)
    ax_r.set_xlim(t[0], t[-1] if n > 1 else 1.0)
    yr = vol.range_m[np.isfinite(vol.range_m)]
    ax_r.set_ylim(max(0, float(np.nanmin(yr)) - 0.2), float(np.nanmax(yr)) + 0.2 if yr.size else 4)
    ax_r.set_ylabel("range_m")
    ax_r.set_xlabel("Time (s)")
    ax_r.grid(True, alpha=0.3)

    line_d, = ax_d.plot([], [], "C1", lw=1.5)
    cur_d, = ax_d.plot([], [], "C1", marker="o", ms=4)
    ax_d.set_xlim(t[0], t[-1] if n > 1 else 1.0)
    ax_d.set_ylim(-d_plot_lim, d_plot_lim)
    ax_d.set_ylabel("doppler_mps")
    ax_d.set_xlabel("Time (s)")
    ax_d.grid(True, alpha=0.3)

    def _update(i: int):
        im.set_data(rd_stack[i])
        rm, dm = vol.range_m[i], vol.doppler_mps[i]
        ang = vol.angle_deg[i]
        if np.isfinite(rm) and np.isfinite(dm):
            peak_sc.set_offsets(np.c_[[rm], [dm]])
            peak_dot.set_offsets(np.c_[[rm], [dm]])
            if np.isfinite(ang):
                peak_angle.xy = (float(rm), float(dm))
                peak_angle.set_text(f"{float(ang):+.1f}°")
                peak_angle.set_visible(True)
            else:
                peak_angle.set_visible(False)
        else:
            peak_sc.set_offsets(np.empty((0, 2)))
            peak_dot.set_offsets(np.empty((0, 2)))
            peak_angle.set_visible(False)
        sl = slice(0, i + 1)
        line_r.set_data(t[sl], r_pres[sl])
        cur_r.set_data(t[i : i + 1], r_pres[i : i + 1])
        line_d.set_data(t[sl], d_pres[sl])
        cur_d.set_data(t[i : i + 1], d_pres[i : i + 1])
        title.set_text(
            f"frame {i + 1}/{n}  t={t[i]:.2f}s  "
            f"R={rm:.2f}m  V={dm:+.2f}m/s  A={float(ang):+.1f}°  SNR={vol.snr_db[i]:.1f}dB"
            if np.isfinite(rm) and np.isfinite(ang)
            else f"frame {i + 1}/{n}  t={t[i]:.2f}s  "
            f"R={rm:.2f}m  V={dm:+.2f}m/s  SNR={vol.snr_db[i]:.1f}dB"
            if np.isfinite(rm)
            else f"frame {i + 1}/{n}  t={t[i]:.2f}s"
        )
        return im, peak_sc, peak_dot, peak_angle, line_r, cur_r, line_d, cur_d, title

    interval_ms = 1000.0 / max(fps, 0.1)
    anim = FuncAnimation(fig, _update, frames=n, interval=interval_ms, blit=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "gif":
        writer = PillowWriter(fps=fps)
    else:
        try:
            writer = FFMpegWriter(fps=fps, bitrate=1800)
        except Exception as e:
            raise RuntimeError("MP4 needs ffmpeg — use --format gif") from e

    print(f"Writing movie {out_path} ({n} frames @ {fps:.1f} fps)…")
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot config3 OSC range/doppler from capture (frame plot or movie)"
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=_ROOT / "config/live_radar_to_max.json",
    )
    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="Capture folder (e.g. testbonk or captures/testbonk)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Frame index: save range/doppler time-series PNG through this frame",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output PNG or movie path")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames (0=all)")
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--fps", type=float, default=0, help="Movie FPS (0=from frame_time)")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--vmin", type=float, default=DEFAULT_VMIN_DB)
    parser.add_argument("--vmax", type=float, default=DEFAULT_VMAX_DB)
    args = parser.parse_args()

    settings = _load_settings(args.settings)
    capture_path = _resolve_capture_path(args.capture)
    if not (capture_path / "metadata.json").is_file() and not (capture_path / "session.json").is_file():
        raise FileNotFoundError(f"Not a capture folder: {capture_path}")

    session = CaptureSession.open(capture_path)
    params = dict(session.radar_params())
    out_dir = capture_path / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = active_gesture_mode(settings)
    if mode not in ("config3", "config4", "config5", "config6"):
        raise ValueError(
            f"Unsupported gesture.mode for this plot: {mode!r}. "
            "Use config3/config4/config5/config6."
        )

    print(f"Capture: {capture_path.name}")
    frames = _load_frames_from_capture(capture_path, args.max_frames)
    print(f"Processing {len(frames)} frames ({mode} OSC pipeline)…")
    vol, processor, _peak_cfg = _build_volume(frames, params, settings)

    npz_path = out_dir / f"{mode}_osc_tracks.npz"
    save_tracks_npz(npz_path, vol)
    print(f"Saved tracks: {npz_path}")

    frame_time_ms = float(params.get("frame_time", 22.22))
    fps = args.fps if args.fps > 0 else 1000.0 / frame_time_ms

    if args.frame is not None:
        out_path = args.out or out_dir / f"{mode}_osc_frame{args.frame}.png"
        plot_frame_timeseries(
            vol,
            frames,
            params,
            processor,
            frame_idx=int(args.frame),
            out_path=Path(out_path),
            vmin_db=args.vmin,
            vmax_db=args.vmax,
        )
        print(f"Saved: {out_path}")
        return

    ext = args.format
    out_path = args.out or out_dir / f"{mode}_osc_movie.{ext}"
    render_osc_movie(
        vol,
        frames,
        params,
        processor,
        out_path=Path(out_path),
        fps=fps,
        fmt=ext,
        dpi=args.dpi,
        vmin_db=args.vmin,
        vmax_db=args.vmax,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
