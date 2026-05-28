#!/usr/bin/env python3
"""
Range–Doppler heatmap per frame → movie (MP4 or GIF).

  python3 -m post_processing.plot_rd_movie --capture captures/push_pull
  python3 -m post_processing.plot_rd_movie --capture push_pull --format gif --fps 15

Output: <capture>/analysis/rd_heatmap_movie.mp4 (or .gif)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from background_model import estimate_rd_background_from_capture, frame_rd_power_db
from capture_store import CaptureSession
from post_processing.processing_config import ProcessingConfig
from post_processing.rd_maps import _frame_times, rd_to_snr_db
from post_processing.range_gate import estimate_range_gate_for_capture
from processing.rda import range_doppler_axes


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


def collect_decluttered_rd(
    capture_path: Path,
    *,
    range_gate_m: tuple[float, float],
    calibration_frames: int,
    background_capture: Path | None,
    background_max_frames: int,
    max_frames: int = 0,
    use_snr: bool = True,
    show_progress: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Return per-frame RD maps (doppler × range), axes, and time_s."""
    session = CaptureSession.open(capture_path)
    params = session.radar_params()
    range_axis, doppler_axis = range_doppler_axes(params)

    r_lo, r_hi = range_gate_m
    r_mask = (range_axis >= r_lo) & (range_axis <= r_hi)
    range_m = range_axis[r_mask]
    doppler_mps = doppler_axis

    paths = session.frame_paths()
    if max_frames > 0:
        paths = paths[:max_frames]
    if not paths:
        raise RuntimeError(f"No frames in {capture_path}")

    rd_raw_list = []
    for i, path in enumerate(paths):
        rd_raw_list.append(frame_rd_power_db(np.load(path), params))
        if show_progress and (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{len(paths)} frames…", flush=True)

    if background_capture is not None:
        bg = estimate_rd_background_from_capture(
            Path(background_capture), params, max_frames=background_max_frames
        )
    else:
        n_calib = min(len(rd_raw_list), max(1, int(calibration_frames)))
        bg = np.mean(np.stack(rd_raw_list[:n_calib], axis=0), axis=0)

    frames_out: list[np.ndarray] = []
    for rd_raw in rd_raw_list:
        rd = rd_raw - bg
        rd_roi = rd[:, r_mask]
        if use_snr:
            rd_roi = rd_to_snr_db(rd_roi)
        frames_out.append(rd_roi.astype(np.float32))

    time_s = _frame_times(session, len(frames_out))
    return frames_out, range_m, doppler_mps, time_s


def render_rd_movie(
    rd_frames: list[np.ndarray],
    range_m: np.ndarray,
    doppler_mps: np.ndarray,
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

    stack = np.stack(rd_frames, axis=0)
    vmin = float(np.percentile(stack, 5))
    vmax = float(np.percentile(stack, 99))

    extent_rd = [float(doppler_mps[0]), float(doppler_mps[-1]), float(range_m[0]), float(range_m[-1])]
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        rd_frames[0].T,
        aspect="auto",
        origin="lower",
        extent=extent_rd,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("Doppler (m/s)")
    ax.set_ylabel("range (m)")
    title = ax.set_title("")
    fig.colorbar(im, ax=ax, label="SNR (dB)" if vmax > 20 else "power (dB)")

    def _update(i: int):
        im.set_data(rd_frames[i].T)
        t = float(time_s[i]) if i < len(time_s) else float(i)
        title.set_text(f"{session_id} — frame {i + 1}/{len(rd_frames)}  t={t:.2f}s")
        return (im, title)

    interval_ms = 1000.0 / max(fps, 0.1)
    anim = FuncAnimation(fig, _update, frames=len(rd_frames), interval=interval_ms, blit=False)

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

    print(f"Writing {out_path} ({len(rd_frames)} frames @ {fps:.1f} fps)…", flush=True)
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Range–Doppler heatmap movie from capture")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--background-capture", type=Path, default=None)
    p.add_argument("--roi-min", type=float, default=None)
    p.add_argument("--roi-max", type=float, default=None)
    p.add_argument("--calibration-frames", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument(
        "--format",
        choices=["mp4", "gif"],
        default="mp4",
        help="Output video format (mp4 requires ffmpeg)",
    )
    p.add_argument("--fps", type=float, default=0.0, help="0 = use capture frame rate from metadata")
    p.add_argument(
        "--power-db",
        action="store_true",
        help="Plot decluttered power (dB) instead of per-frame SNR",
    )
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

    session = CaptureSession.open(capture)
    params = session.radar_params()
    radar_max = float(params["range_max"])

    print(f"Range gate for {capture.name}…")
    r_min, r_max, _ = estimate_range_gate_for_capture(capture, proc, radar_max_range_m=radar_max)
    print(f"  range gate: {r_min:.2f} – {r_max:.2f} m")

    frame_time_ms = float(params.get("frame_time", 22.22))
    fps = args.fps if args.fps > 0 else 1000.0 / frame_time_ms

    print("Building per-frame range–Doppler maps…")
    rd_frames, range_m, doppler_mps, time_s = collect_decluttered_rd(
        capture,
        range_gate_m=(r_min, r_max),
        calibration_frames=proc.calibration_frames,
        background_capture=proc.background.capture,
        background_max_frames=proc.background.max_frames,
        max_frames=args.max_frames,
        use_snr=not args.power_db,
    )

    ext = "gif" if args.format == "gif" else "mp4"
    out_path = out_dir / f"rd_heatmap_movie.{ext}"
    render_rd_movie(
        rd_frames,
        range_m,
        doppler_mps,
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
