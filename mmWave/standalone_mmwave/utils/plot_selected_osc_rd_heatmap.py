#!/usr/bin/env python3
"""
Plot selected OSC outputs on a Range/Doppler axis (no heatmap).

- If ``--frame`` is provided: save one annotated PNG for that frame.
- If ``--frame`` is omitted: save an annotated GIF over all frames.

Input OSC tracks: ``config3_osc_tracks.npz`` from ``utils/plot_config3_osc.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from capture_store import CaptureSession
from live_radar_to_max import _resolve_capture_path
from utils.rd_power import rd_axes_from_params


def main() -> None:
    p = argparse.ArgumentParser(description="Range/Doppler scatter from selected OSC values")
    p.add_argument("--capture", type=Path, required=True, help="Capture folder or name")
    p.add_argument(
        "--tracks-npz",
        type=Path,
        default=None,
        help="Path to config3_osc_tracks.npz (default: <capture>/analysis/config3_osc_tracks.npz)",
    )
    p.add_argument("--frame", type=int, default=None, help="Frame index for single PNG output")
    p.add_argument("--out", type=Path, default=None, help="Output PNG/GIF path")
    p.add_argument("--fps", type=float, default=12.0, help="GIF FPS when --frame is not set")
    p.add_argument("--max-frames", type=int, default=0, help="Limit number of frames (0=all)")
    p.add_argument("--max-range", type=float, default=3.4, help="Range gate for display (m)")
    p.add_argument(
        "--max-abs-doppler",
        type=float,
        default=3.0,
        help="Doppler gate for display and accumulation (m/s)",
    )
    p.add_argument(
        "--use-presence-only",
        action="store_true",
        help="Use only frames where present=true in tracks NPZ",
    )
    args = p.parse_args()

    capture = _resolve_capture_path(args.capture)
    if not capture.exists():
        raise FileNotFoundError(f"Capture not found: {capture}")

    if args.tracks_npz is not None:
        tracks_path = args.tracks_npz.expanduser().resolve()
    else:
        tracks_path = (capture / "analysis" / "config3_osc_tracks.npz").resolve()
    if not tracks_path.is_file():
        raise FileNotFoundError(f"Tracks NPZ not found: {tracks_path}")

    session = CaptureSession.open(capture)
    params = session.radar_params()

    z = np.load(tracks_path)
    range_track = np.asarray(z["range_m"], dtype=np.float64)
    doppler_track = np.asarray(z["doppler_mps"], dtype=np.float64)
    angle_track = np.asarray(z["angle_deg"], dtype=np.float64) if "angle_deg" in z.files else np.full_like(range_track, np.nan)
    present = np.asarray(z["present"], dtype=bool) if "present" in z.files else np.ones_like(range_track, dtype=bool)

    n = min(len(range_track), len(doppler_track))
    if args.max_frames > 0:
        n = min(n, int(args.max_frames))
    if n <= 0:
        raise RuntimeError("No points in tracks NPZ")
    range_track = range_track[:n]
    doppler_track = doppler_track[:n]
    angle_track = angle_track[:n]
    present = present[:n]

    def _valid_point(i: int) -> bool:
        ok = np.isfinite(range_track[i]) and np.isfinite(doppler_track[i])
        if args.use_presence_only:
            ok = ok and bool(present[i])
        if ok and args.max_abs_doppler > 0:
            ok = abs(float(doppler_track[i])) <= float(args.max_abs_doppler)
        return ok

    r_axis, d_axis = rd_axes_from_params(params)
    r_max = min(float(args.max_range), float(r_axis[-1]))
    d_lim = float(args.max_abs_doppler) if args.max_abs_doppler > 0 else float(np.max(np.abs(d_axis)))

    if args.frame is not None:
        i = int(args.frame)
        if i < 0 or i >= n:
            raise ValueError(f"--frame must be in [0, {n - 1}]")
        out = args.out or (capture / "analysis" / f"config3_selected_rd_frame{i}.png")
        out = out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Doppler (m/s)")
        ax.set_xlim(0.0, r_max)
        ax.set_ylim(-d_lim, d_lim)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Selected OSC point at frame {i}")
        if _valid_point(i):
            rr = float(range_track[i])
            dd = float(doppler_track[i])
            aa = float(angle_track[i]) if np.isfinite(angle_track[i]) else np.nan
            ax.scatter([rr], [dd], s=90, c="crimson", edgecolors="black", linewidths=0.7, zorder=5)
            label = f"{aa:+.1f} deg" if np.isfinite(aa) else "angle n/a"
            ax.annotate(
                label,
                (rr, dd),
                xytext=(6, 6),
                textcoords="offset points",
                color="black",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
            )
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")
        return

    out = args.out or (capture / "analysis" / "config3_selected_rd_movie.gif")
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Doppler (m/s)")
    ax.set_xlim(0.0, r_max)
    ax.set_ylim(-d_lim, d_lim)
    ax.grid(True, alpha=0.3)
    marker = ax.scatter([], [], s=90, c="crimson", edgecolors="black", linewidths=0.7, zorder=5)
    ann = ax.annotate(
        "",
        (0, 0),
        xytext=(6, 6),
        textcoords="offset points",
        color="black",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
    )
    title = ax.set_title("")

    def _update(i: int):
        title.set_text(f"Selected OSC point frame {i + 1}/{n}")
        if _valid_point(i):
            rr = float(range_track[i])
            dd = float(doppler_track[i])
            aa = float(angle_track[i]) if np.isfinite(angle_track[i]) else np.nan
            marker.set_offsets(np.array([[rr, dd]], dtype=np.float64))
            ann.xy = (rr, dd)
            ann.set_text(f"{aa:+.1f} deg" if np.isfinite(aa) else "angle n/a")
            ann.set_visible(True)
        else:
            marker.set_offsets(np.empty((0, 2), dtype=np.float64))
            ann.set_visible(False)
        return marker, ann, title

    anim = FuncAnimation(fig, _update, frames=n, interval=1000.0 / max(float(args.fps), 0.1), blit=False)
    anim.save(str(out), writer=PillowWriter(fps=float(args.fps)), dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

