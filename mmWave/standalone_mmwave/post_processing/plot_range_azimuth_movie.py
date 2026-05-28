#!/usr/bin/env python3
"""
Per-frame range–azimuth heatmap → movie (MP4 or GIF).

Per range bin: strongest Doppler at that range, then antenna FFT (same as static ``range_azimuth.png``).

  python3 -m post_processing.plot_range_azimuth_movie --capture captures/push_pull
  python3 -m post_processing.plot_range_azimuth_movie --capture push_pull --format gif

Output: <capture>/analysis/range_azimuth_movie.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from capture_store import CaptureSession
from post_processing.processing_config import ProcessingConfig
from post_processing.range_gate import estimate_range_gate_for_capture
from processing.range_azimuth import collect_range_azimuth_frames


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


def render_range_azimuth_movie(
    ra_frames: list[np.ndarray],
    range_m: np.ndarray,
    angle_deg: np.ndarray,
    time_s: np.ndarray,
    out_path: Path,
    *,
    fps: float,
    fmt: str,
    session_id: str,
    dpi: int = 100,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    stack = np.stack(ra_frames, axis=0)
    vmin = float(np.percentile(stack, 5))
    vmax = float(np.percentile(stack, 99))

    extent = [
        float(angle_deg[0]),
        float(angle_deg[-1]),
        float(range_m[0]),
        float(range_m[-1]),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        ra_frames[0],
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.axvline(0.0, color="white", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel("azimuth (deg)  [0° = center, − left, + right]")
    ax.set_ylabel("range (m)")
    title = ax.set_title("")
    fig.colorbar(im, ax=ax, label="power (dB)")

    def _update(i: int):
        im.set_data(ra_frames[i])
        t = float(time_s[i]) if i < len(time_s) else float(i)
        title.set_text(
            f"{session_id} — range–azimuth  frame {i + 1}/{len(ra_frames)}  t={t:.2f}s"
        )
        return (im, title)

    interval_ms = 1000.0 / max(fps, 0.1)
    anim = FuncAnimation(
        fig, _update, frames=len(ra_frames), interval=interval_ms, blit=False
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "gif":
        writer = PillowWriter(fps=fps)
    else:
        try:
            writer = FFMpegWriter(fps=fps, bitrate=4000)
        except Exception as e:
            raise RuntimeError(
                "MP4 export needs ffmpeg. Install ffmpeg or use --format gif"
            ) from e

    print(f"Writing {out_path} ({len(ra_frames)} frames @ {fps:.1f} fps)…", flush=True)
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Range–azimuth heatmap movie from capture")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--background-capture", type=Path, default=None)
    p.add_argument("--roi-min", type=float, default=None)
    p.add_argument("--roi-max", type=float, default=None)
    p.add_argument("--calibration-frames", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--angle-bins", type=int, default=None)
    p.add_argument("--angle-fov-deg", type=float, default=None)
    p.add_argument(
        "--format",
        choices=["mp4", "gif"],
        default="mp4",
        help="Output video format (mp4 requires ffmpeg)",
    )
    p.add_argument("--fps", type=float, default=0.0, help="0 = capture frame rate from metadata")
    p.add_argument("--dpi", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    capture = _resolve_capture_path(args.capture).resolve()
    out_dir = args.out_dir or (capture / "analysis")

    proc = ProcessingConfig.load(args.config)
    if args.background_capture:
        proc.background.capture = args.background_capture.resolve()
    if args.roi_min is not None:
        proc.range_min_m = args.roi_min
    if args.roi_max is not None:
        proc.range_max_m = args.roi_max
    if args.calibration_frames is not None:
        proc.calibration_frames = args.calibration_frames

    angle_bins = int(
        args.angle_bins if args.angle_bins is not None else proc.angle.fft_bins
    )
    angle_fov = float(
        args.angle_fov_deg if args.angle_fov_deg is not None else proc.angle.fov_deg
    )

    cfg_path = args.config or (_ROOT / "config" / "live_radar_to_max.json")
    session = CaptureSession.open(capture)
    params = session.radar_params()
    radar_max = float(params["range_max"])

    print(f"Range gate for {capture.name}…")
    r_min, r_max, _ = estimate_range_gate_for_capture(capture, proc, radar_max_range_m=radar_max)
    print(f"  range gate: {r_min:.2f} – {r_max:.2f} m")
    print(f"  angle FFT bins: {angle_bins}  FOV: ±{angle_fov/2:.0f}°")

    frame_time_ms = float(params.get("frame_time", 22.22))
    fps = args.fps if args.fps > 0 else 1000.0 / frame_time_ms

    print("Building per-frame range–azimuth maps…")
    ra_frames, range_m, angle_deg, time_s = collect_range_azimuth_frames(
        capture,
        range_gate_m=(r_min, r_max),
        angle_bins=angle_bins,
        fov_deg=angle_fov,
        background_capture=proc.background.capture,
        background_max_frames=proc.background.max_frames,
        calibration_frames=proc.calibration_frames,
        max_frames=args.max_frames,
    )

    ext = "gif" if args.format == "gif" else "mp4"
    out_path = out_dir / f"range_azimuth_movie.{ext}"
    render_range_azimuth_movie(
        ra_frames,
        range_m,
        angle_deg,
        time_s,
        out_path,
        fps=fps,
        fmt=args.format,
        session_id=capture.name,
        dpi=args.dpi,
    )
    print(f"Done: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
