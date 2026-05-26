#!/usr/bin/env python3
"""
Range–time SNR heatmap for a capture (no global peak picker).

  python3 -m post_processing.plot_heatmap --capture captures/20260526_115659_right_left
  python3 -m post_processing.plot_heatmap --capture captures/... --snr-threshold 10

Outputs in <capture>/analysis/:
  range_time_snr.png       — SNR (dB) vs range & time
  range_time_mask.png      — pixels above threshold only
  range_time_snr.npz       — arrays for custom plots
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
from post_processing.gesture_config import GestureProcessingConfig
from post_processing.range_gate import estimate_range_gate_for_capture
from post_processing.rd_maps import RangeTimeVolume, build_range_time_volume


def _plot_volume(vol: RangeTimeVolume, out_dir: Path, range_max_m: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    t = vol.time_s
    r = vol.range_m
    snr = vol.snr_db
    thr = vol.snr_threshold_db

    extent = [t[0], t[-1] if len(t) > 1 else t[0] + 1, r[0], r[-1]]

    # Main heatmap
    fig, ax = plt.subplots(figsize=(12, 5))
    vmax = np.percentile(snr, 99)
    vmin = max(thr - 3, np.percentile(snr, 5))
    im = ax.imshow(
        snr.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("range (m)")
    title = f"{vol.session_id} — SNR (dB), max over Doppler"
    if range_max_m is not None:
        title += f"  [range ≤ {range_max_m:.2f} m]"
        ax.axhline(range_max_m, color="red", linewidth=1.2, linestyle="--", alpha=0.9)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, label="SNR (dB)")
    cb.ax.axhline(thr, color="red", linewidth=1.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "range_time_snr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Threshold mask only
    masked = np.where(snr >= thr, snr, np.nan)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        masked.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="hot",
        vmin=thr,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("range (m)")
    ax.set_title(f"SNR ≥ {thr:.0f} dB (targets above threshold)")
    fig.colorbar(im, ax=ax, label="SNR (dB)")
    fig.tight_layout()
    fig.savefig(out_dir / "range_time_mask.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Mean range–Doppler over capture (scene summary)
    rd_mean = np.mean(vol.rd_stack, axis=0)
    d = vol.doppler_mps
    fig, ax = plt.subplots(figsize=(10, 5))
    extent_rd = [d[0], d[-1], r[0], r[-1]]
    im = ax.imshow(
        rd_mean.T,
        aspect="auto",
        origin="lower",
        extent=extent_rd,
        cmap="viridis",
        vmin=np.percentile(rd_mean, 5),
        vmax=np.percentile(rd_mean, 99),
    )
    ax.set_xlabel("Doppler (m/s)")
    ax.set_ylabel("range (m)")
    ax.set_title("Mean SNR map (avg over time)")
    fig.colorbar(im, ax=ax, label="SNR (dB)")
    fig.tight_layout()
    fig.savefig(out_dir / "range_doppler_mean.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Range–time SNR heatmap from raw capture")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config (default: config/gesture_processing.json)",
    )
    p.add_argument(
        "--background-capture",
        type=Path,
        default=None,
        help="Empty-room capture for range limit (overrides config background.capture)",
    )
    p.add_argument("--roi-min", type=float, default=None, help="Override config range_gate.min_m")
    p.add_argument("--roi-max", type=float, default=None, help="Override config range_gate.max_m")
    p.add_argument("--snr-threshold", type=float, default=None)
    p.add_argument("--clutter-window", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    capture = Path(args.capture).resolve()
    out_dir = args.out_dir or (capture / "analysis")

    proc = GestureProcessingConfig.load(args.config)
    if args.background_capture:
        proc.background.capture = args.background_capture.resolve()
    if args.roi_min is not None:
        proc.range_min_m = args.roi_min
    if args.roi_max is not None:
        proc.range_max_m = args.roi_max
    if args.snr_threshold is not None:
        proc.snr_threshold_db = args.snr_threshold
    if args.clutter_window is not None:
        proc.clutter_window_frames = args.clutter_window

    params = CaptureSession.open(capture).radar_params()
    radar_max = float(params["range_max"])

    print(f"Estimating range gate from background for {capture.name}…")
    r_min, r_max, gate_info = estimate_range_gate_for_capture(capture, proc, radar_max_range_m=radar_max)
    print(f"  range gate: {r_min:.2f} – {r_max:.2f} m  (nothing beyond {r_max:.2f} m processed)")

    proc.save_applied(out_dir / "range_gate.json", gate_info)

    print(f"Building range–time SNR volume…")
    vol = build_range_time_volume(
        capture,
        range_gate_m=(r_min, r_max),
        clutter_window=proc.clutter_window_frames,
        snr_threshold_db=proc.snr_threshold_db,
        max_frames=args.max_frames,
    )

    np.savez_compressed(
        out_dir / "range_time_snr.npz",
        snr_db=vol.snr_db,
        time_s=vol.time_s,
        range_m=vol.range_m,
        doppler_mps=vol.doppler_mps,
        rd_stack=vol.rd_stack,
        snr_threshold_db=vol.snr_threshold_db,
        range_min_m=r_min,
        range_max_m=r_max,
    )

    _plot_volume(vol, out_dir, range_max_m=r_max)
    n_above = int(np.sum(vol.mask()))
    print(f"Done: {vol.snr_db.shape[0]} frames, {vol.snr_db.shape[1]} range bins")
    print(f"  pixels ≥ {proc.snr_threshold_db} dB: {n_above}")
    print(f"  {out_dir.resolve()}/range_time_snr.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
