#!/usr/bin/env python3
"""
Range-azimuth heatmap from capture frames.

If --frame is provided: saves one PNG for that frame.
If --frame is omitted: saves a GIF over all frames.
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

from live_radar_to_max import _load_settings, _resolve_capture_path
from processing.range_azimuth import collect_range_azimuth_frames


def _plot_one(
    frame_db: np.ndarray,
    range_m: np.ndarray,
    angle_deg: np.ndarray,
    t_s: float,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        frame_db,
        aspect="auto",
        origin="upper",
        cmap="jet",
        extent=[float(angle_deg[0]), float(angle_deg[-1]), float(range_m[-1]), float(range_m[0])],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_title(f"Range-Azimuth @ t={t_s:.2f}s")
    plt.colorbar(im, ax=ax, label="Power (dB)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _movie(
    frames_db: list[np.ndarray],
    range_m: np.ndarray,
    angle_deg: np.ndarray,
    time_s: np.ndarray,
    out_path: Path,
    fps: float,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        frames_db[0],
        aspect="auto",
        origin="upper",
        cmap="jet",
        extent=[float(angle_deg[0]), float(angle_deg[-1]), float(range_m[-1]), float(range_m[0])],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    title = ax.set_title("")
    plt.colorbar(im, ax=ax, label="Power (dB)")

    def _update(i: int):
        im.set_data(frames_db[i])
        title.set_text(f"Range-Azimuth frame {i + 1}/{len(frames_db)}  t={float(time_s[i]):.2f}s")
        return im, title

    anim = FuncAnimation(fig, _update, frames=len(frames_db), interval=1000.0 / max(fps, 0.1), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Range-azimuth frame plot or GIF from capture")
    p.add_argument("--settings", type=Path, default=_ROOT / "config/live_radar_to_max.json")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--frame", type=int, default=None, help="If set, output only this frame as PNG")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--angle-bins", type=int, default=128)
    args = p.parse_args()

    settings = _load_settings(args.settings)
    proc = settings.get("processing", {})
    ang = settings.get("angle_estimation", {})
    bg = settings.get("background", {})

    capture = _resolve_capture_path(args.capture)
    range_gate = (float(proc.get("roi_min_m", 0.3)), float(proc.get("roi_max_m", 4.0)))
    fov = float(ang.get("fov_deg", 60.0))

    background_capture = bg.get("capture")
    bg_path = _resolve_capture_path(Path(background_capture)) if background_capture else None
    bg_max = int(bg.get("max_frames", 0))
    cal_frames = int(proc.get("declutter_mean_frames", 45))

    frames_db, range_m, angle_deg, time_s = collect_range_azimuth_frames(
        capture,
        range_gate_m=range_gate,
        angle_bins=int(args.angle_bins),
        fov_deg=fov,
        background_capture=bg_path,
        background_max_frames=bg_max,
        calibration_frames=cal_frames,
        max_frames=int(args.max_frames),
        show_progress=True,
    )

    if args.frame is not None:
        i = int(args.frame)
        if i < 0 or i >= len(frames_db):
            raise ValueError(f"--frame must be in [0, {len(frames_db)-1}]")
        out = args.out or (Path(capture) / "analysis" / f"range_azimuth_frame{i}.png")
        _plot_one(frames_db[i], range_m, angle_deg, float(time_s[i]), Path(out), vmin=args.vmin, vmax=args.vmax)
        print(f"Saved: {out}")
        return

    out = args.out or (Path(capture) / "analysis" / "range_azimuth_movie.gif")
    _movie(frames_db, range_m, angle_deg, time_s, Path(out), fps=float(args.fps), vmin=args.vmin, vmax=args.vmax)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

