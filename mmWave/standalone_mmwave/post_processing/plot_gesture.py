#!/usr/bin/env python3
"""
Plot gesture metrics vs. time from any raw capture session.

  python3 -m post_processing.plot_gesture --capture captures/20260526_115737_two_step
  python3 -m post_processing.plot_gesture --capture captures/push_pull --save-only

Outputs (under <capture>/analysis/):
  gesture_timeseries.csv
  gesture_plots.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from post_processing.analyze_capture import analyze_capture, load_timeseries


def _plot(ts, out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = ts.time_s
    fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(title, fontsize=12)

    axes[0].plot(t, ts.range_m, color="#2563eb", linewidth=1.2)
    axes[0].set_ylabel("range (m)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, ts.angle_deg, color="#7c3aed", linewidth=1.2)
    axes[1].set_ylabel("angle (deg)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, ts.doppler_mps, color="#059669", linewidth=1.2)
    axes[2].set_ylabel("Doppler (m/s)")
    axes[2].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, ts.energy, color="#d97706", linewidth=1.2)
    axes[3].set_ylabel("energy")
    axes[3].set_yscale("log")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(t, ts.presence, color="#dc2626", linewidth=1.5, drawstyle="steps-post")
    axes[4].set_ylabel("presence")
    axes[4].set_ylim(-0.1, 1.1)
    axes[4].set_xlabel("time (s)")
    axes[4].grid(True, alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Plot gesture time series from a capture")
    p.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="Capture session folder (captures/YYYYMMDD_name)",
    )
    p.add_argument(
        "--csv-only",
        action="store_true",
        help="Only write CSV (skip plots)",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Plot from existing analysis/gesture_timeseries.csv",
    )
    p.add_argument("--save-only", action="store_true", help="Save files, do not open plot window")
    p.add_argument("--roi-min", type=float, default=0.5)
    p.add_argument("--roi-max", type=float, default=12.0)
    p.add_argument("--smooth-alpha", type=float, default=0.15)
    p.add_argument("--presence-threshold-db", type=float, default=12.0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <capture>/analysis/",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    capture = Path(args.capture).resolve()
    if not capture.is_dir():
        print(f"Not a directory: {capture}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or (capture / "analysis")
    csv_path = out_dir / "gesture_timeseries.csv"
    png_path = out_dir / "gesture_plots.png"

    if args.plot_only:
        if not csv_path.is_file():
            print(f"Missing {csv_path}; run without --plot-only first.", file=sys.stderr)
            return 1
        ts = load_timeseries(csv_path)
        print(f"Loaded {len(ts.time_s)} frames from {csv_path}")
    else:
        from capture_store import CaptureSession

        n_raw = len(CaptureSession.open(capture).frame_paths())
        print(f"Analyzing {capture.name} ({n_raw} raw frames)…")
        ts = analyze_capture(
            capture,
            range_gate_m=(args.roi_min, args.roi_max),
            smooth_alpha=args.smooth_alpha,
            presence_threshold_db=args.presence_threshold_db,
            max_frames=args.max_frames,
        )
        ts.to_csv(csv_path)
        print(f"Wrote {csv_path}  ({len(ts.time_s)} frames)")

    if not args.csv_only:
        title = f"{ts.session_id}  ({len(ts.time_s)} frames)"
        _plot(ts, png_path, title)
        print(f"Wrote {png_path}")
        if not args.save_only:
            print("Open gesture_plots.png in the capture/analysis/ folder.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
