#!/usr/bin/env python3
"""
Build an RD heatmap movie from a packed NPZ (same pipeline as rd_heatmap_new.py).

Usage (from standalone_mmwave/):
    python3 utils/rd_heatmap_movie_new.py \\
        --npz captures/testbonk/analysis/testbonk_fixed.npz
"""

from __future__ import annotations

import argparse
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

from utils.rd_heatmap_new import load_rd_power_stack


DEFAULT_VMIN_DB = 50.0
DEFAULT_VMAX_DB = 95.0


def _color_limits(vmin: float | None, vmax: float | None) -> tuple[float, float]:
    return (
        float(DEFAULT_VMIN_DB if vmin is None else vmin),
        float(DEFAULT_VMAX_DB if vmax is None else vmax),
    )


def _frame_times(npz_path: Path, n_frames: int, frame_time_ms: float) -> np.ndarray:
    z = np.load(npz_path)
    if "radar_time" in z.files and len(z["radar_time"]) >= n_frames:
        t = np.asarray(z["radar_time"][:n_frames], dtype=np.float64)
        if t[-1] > 1e12:
            t = (t - t[0]) * 1e-9
        elif t[-1] > 1e6:
            t = (t - t[0]) * 1e-6
        else:
            t = t - t[0]
        return t.astype(np.float64)
    fps = 1000.0 / frame_time_ms
    return (np.arange(n_frames, dtype=np.float64) / fps)


def render_movie(
    rd_db: np.ndarray,
    r_axis: np.ndarray,
    d_axis: np.ndarray,
    time_s: np.ndarray,
    out_path: Path,
    *,
    fps: float,
    fmt: str = "mp4",
    title_prefix: str = "",
    dpi: int = 120,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    vmin, vmax = _color_limits(vmin, vmax)
    n_frames = rd_db.shape[0]
    extent = [float(r_axis[0]), float(r_axis[-1]), float(d_axis[-1]), float(d_axis[0])]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        rd_db[0],
        aspect="auto",
        cmap="jet",
        origin="upper",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Doppler (m/s)")
    title = ax.set_title("")
    fig.colorbar(im, ax=ax, label="Power (dB)")

    def _update(i: int):
        im.set_data(rd_db[i])
        t = float(time_s[i]) if i < len(time_s) else float(i)
        title.set_text(
            f"{title_prefix}frame {i + 1}/{n_frames}  t={t:.2f}s"
            if title_prefix
            else f"frame {i + 1}/{n_frames}  t={t:.2f}s"
        )
        return (im, title)

    interval_ms = 1000.0 / max(fps, 0.1)
    anim = FuncAnimation(
        fig, _update, frames=n_frames, interval=interval_ms, blit=False
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "gif":
        writer = PillowWriter(fps=fps)
    else:
        try:
            writer = FFMpegWriter(fps=fps, bitrate=1800)
        except Exception as e:
            raise RuntimeError(
                "MP4 export needs ffmpeg. Install ffmpeg or use --format gif"
            ) from e

    print(f"Writing {out_path} ({n_frames} frames @ {fps:.1f} fps, vmin={vmin:.1f} vmax={vmax:.1f})…")
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RD heatmap movie from packed NPZ (rd_heatmap_new pipeline)"
    )
    parser.add_argument("--npz", type=str, required=True, help="Packed radar NPZ")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: <npz_dir>/rd_heatmap_movie_new.mp4)",
    )
    parser.add_argument("--max-range", type=float, default=3.4, help="Range gate (m)")
    parser.add_argument(
        "--fps",
        type=float,
        default=0,
        help="Movie FPS (0 = from radar frame_time in cfg)",
    )
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--vmin", type=float, default=DEFAULT_VMIN_DB, help="Color scale min (dB)"
    )
    parser.add_argument(
        "--vmax", type=float, default=DEFAULT_VMAX_DB, help="Color scale max (dB)"
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames (0=all)")
    args = parser.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    print(f"Loading: {npz_path}")
    _, rd_db, r_axis, d_axis, params = load_rd_power_stack(
        npz_path, max_range_m=args.max_range
    )
    if args.max_frames > 0:
        rd_db = rd_db[: args.max_frames]

    frame_time_ms = float(params.get("frame_time", 22.22))
    fps = args.fps if args.fps > 0 else 1000.0 / frame_time_ms
    time_s = _frame_times(npz_path, rd_db.shape[0], frame_time_ms)

    if args.out is not None:
        out_path = Path(args.out).expanduser().resolve()
    else:
        ext = args.format
        out_path = npz_path.parent / f"rd_heatmap_movie_new.{ext}"

    render_movie(
        rd_db,
        r_axis,
        d_axis,
        time_s,
        out_path,
        fps=fps,
        fmt=args.format,
        title_prefix=f"{npz_path.name} — ",
        dpi=args.dpi,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
