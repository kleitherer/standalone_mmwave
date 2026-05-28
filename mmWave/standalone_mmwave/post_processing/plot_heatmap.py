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
from processing.range_azimuth import AngleMethod, build_range_azimuth_map
from post_processing.processing_config import ProcessingConfig
from post_processing.range_gate import estimate_range_gate_for_capture
from post_processing.rd_maps import RangeTimeVolume, build_range_time_volume


def _plot_time_snr_2d(
    time_s: np.ndarray,
    axis_values: np.ndarray,
    snr: np.ndarray,
    out_path: Path,
    *,
    x_label: str,
    y_label: str,
    title: str,
    threshold_db: float,
    center_line: float | None = None,
) -> None:
    """Generic time vs axis SNR heatmap (snr shape: n_frames × n_axis)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [
        float(time_s[0]),
        float(time_s[-1] if len(time_s) > 1 else time_s[0] + 1),
        float(axis_values[0]),
        float(axis_values[-1]),
    ]
    vmax = np.percentile(snr, 99)
    vmin = max(threshold_db - 3, np.percentile(snr, 5))

    fig, ax = plt.subplots(figsize=(12, 5))
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
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if center_line is not None:
        ax.axhline(center_line, color="white", linewidth=1.0, linestyle="--", alpha=0.8)
    cb = fig.colorbar(im, ax=ax, label="SNR (dB)")
    cb.ax.axhline(threshold_db, color="red", linewidth=1.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_range_azimuth(
    range_m: np.ndarray,
    angle_deg: np.ndarray,
    power_db: np.ndarray,
    out_path: Path,
    *,
    method: str,
    range_max_m: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [float(angle_deg[0]), float(angle_deg[-1]), float(range_m[0]), float(range_m[-1])]
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        power_db,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=np.percentile(power_db, 5),
        vmax=np.percentile(power_db, 99),
        interpolation="nearest",
    )
    ax.axvline(0.0, color="white", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel("azimuth (deg)  [0° = center, − left, + right]")
    ax.set_ylabel("range (m)")
    title = f"Range–azimuth ({method.upper()})"
    if range_max_m is not None:
        title += f"  [≤ {range_max_m:.2f} m]"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="power (dB)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    rd_peak = np.max(vol.rd_stack, axis=0)
    rd_raw_mean = np.mean(vol.rd_raw_stack, axis=0)
    rd_dec_mean = np.mean(vol.rd_declutter_stack, axis=0)
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

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        rd_peak.T,
        aspect="auto",
        origin="lower",
        extent=extent_rd,
        cmap="viridis",
        vmin=np.percentile(rd_peak, 5),
        vmax=np.percentile(rd_peak, 99),
    )
    ax.set_xlabel("Doppler (m/s)")
    ax.set_ylabel("range (m)")
    ax.set_title("Peak SNR map (max over time)")
    fig.colorbar(im, ax=ax, label="SNR (dB)")
    fig.tight_layout()
    fig.savefig(out_dir / "range_doppler_peak.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    im0 = axes[0].imshow(
        rd_raw_mean.T,
        aspect="auto",
        origin="lower",
        extent=extent_rd,
        cmap="jet",
        vmin=np.percentile(rd_raw_mean, 5),
        vmax=np.percentile(rd_raw_mean, 99),
    )
    axes[0].set_xlabel("Doppler (m/s)")
    axes[0].set_ylabel("range (m)")
    axes[0].set_title("Raw RD mean power (dB)")
    fig.colorbar(im0, ax=axes[0], label="Power (dB)")

    im1 = axes[1].imshow(
        rd_dec_mean.T,
        aspect="auto",
        origin="lower",
        extent=extent_rd,
        cmap="jet",
        vmin=np.percentile(rd_dec_mean, 5),
        vmax=np.percentile(rd_dec_mean, 99),
    )
    axes[1].set_xlabel("Doppler (m/s)")
    axes[1].set_title("Decluttered RD mean power (dB)")
    fig.colorbar(im1, ax=axes[1], label="Power (dB)")
    fig.tight_layout()
    fig.savefig(out_dir / "range_doppler_raw_vs_decluttered.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Raw vs decluttered range-time (max over Doppler, both in dB power)
    raw_db = vol.raw_db
    dec_db = vol.declutter_db
    raw_vmin = np.percentile(raw_db, 5)
    raw_vmax = np.percentile(raw_db, 99)
    dec_vmin = np.percentile(dec_db, 5)
    dec_vmax = np.percentile(dec_db, 99)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    im0 = axes[0].imshow(
        raw_db.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="jet",
        vmin=raw_vmin,
        vmax=raw_vmax,
        interpolation="nearest",
    )
    axes[0].set_title("Raw range-time power (dB, max over Doppler)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("range (m)")
    fig.colorbar(im0, ax=axes[0], label="Power (dB)")

    im1 = axes[1].imshow(
        dec_db.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="jet",
        vmin=dec_vmin,
        vmax=dec_vmax,
        interpolation="nearest",
    )
    axes[1].set_title("Decluttered range-time power (dB, max over Doppler)")
    axes[1].set_xlabel("time (s)")
    fig.colorbar(im1, ax=axes[1], label="Power (dB)")

    fig.tight_layout()
    fig.savefig(out_dir / "range_time_raw_vs_decluttered.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Range–time SNR heatmap from raw capture")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config (default: config/live_radar_to_max.json)",
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
    p.add_argument("--calibration-frames", type=int, default=None)
    p.add_argument("--clutter-window", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--declutter-method",
        choices=["global_mean"],
        default="global_mean",
        help="Clutter removal method (global_mean only, matches live).",
    )
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument(
        "--angle-method",
        choices=["fft", "music"],
        default=None,
        help="Azimuth estimator: fft or music (default from config angle_estimation.method)",
    )
    p.add_argument("--angle-bins", type=int, default=None, help="Override config angle_estimation.fft_bins")
    p.add_argument("--angle-fov-deg", type=float, default=None, help="Override config angle_estimation.fov_deg")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    capture = Path(args.capture).resolve()
    out_dir = args.out_dir or (capture / "analysis")

    proc = ProcessingConfig.load(args.config)
    if args.background_capture:
        proc.background.capture = args.background_capture.resolve()
    if args.roi_min is not None:
        proc.range_min_m = args.roi_min
    if args.roi_max is not None:
        proc.range_max_m = args.roi_max
    if args.snr_threshold is not None:
        proc.snr_threshold_db = args.snr_threshold
    if args.calibration_frames is not None:
        proc.calibration_frames = args.calibration_frames
    if args.clutter_window is not None:
        proc.calibration_frames = args.clibration_frames
    angle_method: AngleMethod = (
        args.angle_method if args.angle_method else proc.angle.method  # type: ignore[assignment]
    )
    angle_bins = int(args.angle_bins if args.angle_bins is not None else proc.angle.fft_bins)
    angle_fov = float(args.angle_fov_deg if args.angle_fov_deg is not None else proc.angle.fov_deg)

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
        clutter_window=proc.calibration_frames,
        background_capture=proc.background.capture,
        background_max_frames=proc.background.max_frames,
        angle_method=angle_method,
        angle_bins=angle_bins,
        angle_fov_deg=angle_fov,
        snr_threshold_db=proc.snr_threshold_db,
        max_frames=args.max_frames,
    )

    np.savez_compressed(
        out_dir / "range_time_snr.npz",
        snr_db=vol.snr_db,
        raw_db=vol.raw_db,
        declutter_db=vol.declutter_db,
        time_s=vol.time_s,
        range_m=vol.range_m,
        doppler_mps=vol.doppler_mps,
        rd_stack=vol.rd_stack,
        rd_raw_stack=vol.rd_raw_stack,
        rd_declutter_stack=vol.rd_declutter_stack,
        snr_threshold_db=vol.snr_threshold_db,
        declutter_method=args.declutter_method,
        range_min_m=r_min,
        range_max_m=r_max,
        doppler_time_snr=vol.doppler_time_snr,
        azimuth_time_snr=vol.azimuth_time_snr,
        angle_deg=vol.angle_deg,
        angle_method=angle_method,
    )

    _plot_volume(vol, out_dir, range_max_m=r_max)

    thr = vol.snr_threshold_db
    _plot_time_snr_2d(
        vol.time_s,
        vol.doppler_mps,
        vol.doppler_time_snr,
        out_dir / "doppler_time_snr.png",
        x_label="time (s)",
        y_label="Doppler (m/s)",
        title=f"{vol.session_id} — SNR (dB), max over range",
        threshold_db=thr,
    )
    _plot_time_snr_2d(
        vol.time_s,
        vol.angle_deg,
        vol.azimuth_time_snr,
        out_dir / f"azimuth_time_snr_{angle_method}.png",
        x_label="time (s)",
        y_label="azimuth (deg)  [0° = center, − left, + right]",
        title=f"{vol.session_id} — SNR (dB), max over range ({angle_method.upper()})",
        threshold_db=thr,
        center_line=0.0,
    )

    print(f"Building range–azimuth map ({angle_method.upper()}, {angle_bins} bins, FOV ±{angle_fov/2:.0f}°)…")
    range_m_ra, angle_deg_ra, ra_db = build_range_azimuth_map(
        capture,
        range_gate_m=(r_min, r_max),
        method=angle_method,
        angle_bins=angle_bins,
        fov_deg=angle_fov,
        background_capture=proc.background.capture,
        background_max_frames=proc.background.max_frames,
        max_frames=args.max_frames,
    )
    ra_png = out_dir / f"range_azimuth_{angle_method}.png"
    _plot_range_azimuth(
        range_m_ra, angle_deg_ra, ra_db, ra_png, method=angle_method, range_max_m=r_max
    )
    np.savez_compressed(
        out_dir / f"range_azimuth_{angle_method}.npz",
        range_m=range_m_ra,
        angle_deg=angle_deg_ra,
        power_db=ra_db,
        angle_method=angle_method,
        range_min_m=r_min,
        range_max_m=r_max,
    )

    n_above = int(np.sum(vol.mask()))
    print(f"Done: {vol.snr_db.shape[0]} frames, {vol.snr_db.shape[1]} range bins")
    print(f"  pixels ≥ {proc.snr_threshold_db} dB: {n_above}")
    print(f"  {out_dir.resolve()}/range_time_snr.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
