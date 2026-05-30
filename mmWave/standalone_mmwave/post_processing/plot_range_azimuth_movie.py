#!/usr/bin/env python3
"""
Per-frame range–azimuth heatmap → movie (MP4 or GIF).

Settings: config/live_radar_to_max.json (``processing.roi_*``, ``post_processing.movie_*``).

  python3 -m post_processing.plot_range_azimuth_movie --capture newccrma

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
from post_processing.processing_config import ProcessingConfig, resolve_capture_range_gate
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
    p = argparse.ArgumentParser(
        description="Range–azimuth movie (settings: config/live_radar_to_max.json)"
    )
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    capture = _resolve_capture_path(args.capture).resolve()
    out_dir = args.out_dir or (capture / "analysis")

    proc = ProcessingConfig.load(args.config)
    pp = proc.post_processing
    angle_bins = proc.angle.fft_bins
    angle_fov = proc.angle.fov_deg

    session = CaptureSession.open(capture)
    params = session.radar_params()
    radar_max = float(params["range_max"])

    r_min, r_max, gate_info = resolve_capture_range_gate(
        capture, proc, radar_max_m=radar_max
    )
    print(f"Config: {proc.config_path}")
    print(f"  ROI: {r_min:.2f} – {r_max:.2f} m")
    if gate_info is not None:
        proc.save_applied(out_dir / "range_gate.json", gate_info)
    print(f"  angle FFT bins: {angle_bins}  FOV: ±{angle_fov/2:.0f}°")

    frame_time_ms = float(params.get("frame_time", 22.22))
    fps = pp.movie_fps if pp.movie_fps > 0 else 1000.0 / frame_time_ms

    print("Building per-frame range–azimuth maps…")
    ra_frames, range_m, angle_deg, time_s = collect_range_azimuth_frames(
        capture,
        range_gate_m=(r_min, r_max),
        angle_bins=angle_bins,
        fov_deg=angle_fov,
        background_capture=proc.background.capture,
        background_max_frames=proc.background.max_frames,
        calibration_frames=proc.declutter_mean_frames,
        max_frames=args.max_frames,
    )

    fmt = pp.movie_format if pp.movie_format in ("mp4", "gif") else "mp4"
    out_path = out_dir / f"range_azimuth_movie.{fmt}"
    render_range_azimuth_movie(
        ra_frames,
        range_m,
        angle_deg,
        time_s,
        out_path,
        fps=fps,
        fmt=fmt,
        session_id=capture.name,
        dpi=pp.movie_dpi,
    )
    print(f"Done: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
