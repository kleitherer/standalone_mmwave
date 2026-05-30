#!/usr/bin/env python3
"""
Capture analysis plots.

Settings: config/live_radar_to_max.json
  processing.roi_min_m / roi_max_m  — range gate for all outputs
  post_processing.write_range_time_snr — range vs time overview
  post_processing.write_rd_movie       — per-frame RD movie (ROI-gated)
  post_processing.rd_snapshot_frame    — optional single RD png (frame index)

  python3 -m post_processing.plot_heatmap --capture newccrma
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from capture_store import CaptureSession
from post_processing.processing_config import ProcessingConfig, resolve_capture_range_gate
from post_processing.rd_maps import RangeTimeVolume, build_range_time_volume
from post_processing.rd_plot_mmw import write_rd_outputs


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


def _plot_range_time_snr(vol: RangeTimeVolume, out_dir: Path, range_max_m: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = vol.time_s
    r = vol.range_m
    snr = vol.snr_db
    thr = vol.snr_threshold_db
    extent = [t[0], t[-1] if len(t) > 1 else t[0] + 1, r[0], r[-1]]

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
    ax.set_title(f"{vol.session_id} — SNR (dB), max over Doppler  [{r[0]:.1f}–{range_max_m:.1f} m]")
    cb = fig.colorbar(im, ax=ax, label="SNR (dB)")
    cb.ax.axhline(thr, color="red", linewidth=1.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "range_time_snr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Capture analysis (settings: config/live_radar_to_max.json)"
    )
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    capture = _resolve_capture_path(args.capture).resolve()
    out_dir = args.out_dir or (capture / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = ProcessingConfig.load(args.config)
    pp = proc.post_processing

    push_pull_cfg = json.loads(proc.config_path.read_text()).get("push_pull", {})
    push_pull_mode = bool(push_pull_cfg.get("enabled", False))
    push_pull_snr_within_db = float(push_pull_cfg.get("snr_within_db_of_max", 3.0))

    params = CaptureSession.open(capture).radar_params()
    radar_max = float(params["range_max"])
    r_min, r_max, gate_info = resolve_capture_range_gate(
        capture, proc, radar_max_m=radar_max
    )
    print(f"Config: {proc.config_path}")
    print(f"  ROI: {r_min:.2f} – {r_max:.2f} m")
    if gate_info is not None:
        proc.save_applied(out_dir / "range_gate.json", gate_info)

    if not pp.write_range_time_snr and not pp.write_rd_movie and pp.rd_snapshot_frame is None:
        print("Nothing to write — enable write_range_time_snr or write_rd_movie in config")
        return 0

    if pp.write_range_time_snr:
        print("Building range–time SNR…")
        vol = build_range_time_volume(
            capture,
            range_gate_m=(r_min, r_max),
            clutter_window=proc.declutter_mean_frames,
            background_capture=proc.background.capture,
            background_max_frames=proc.background.max_frames,
            angle_bins=proc.angle.fft_bins,
            angle_fov_deg=proc.angle.fov_deg,
            push_pull_mode=push_pull_mode,
            push_pull_snr_within_db=push_pull_snr_within_db,
            snr_threshold_db=proc.snr_threshold_db,
            max_frames=args.max_frames,
        )
        _plot_range_time_snr(vol, out_dir, range_max_m=r_max)
        np.savez_compressed(
            out_dir / "range_time_snr.npz",
            snr_db=vol.snr_db,
            time_s=vol.time_s,
            range_m=vol.range_m,
            snr_threshold_db=vol.snr_threshold_db,
            range_min_m=r_min,
            range_max_m=r_max,
        )
        n_above = int(np.sum(vol.mask()))
        print(f"  {vol.snr_db.shape[0]} frames × {vol.snr_db.shape[1]} range bins")
        print(f"  pixels ≥ {proc.snr_threshold_db} dB: {n_above}")
        print(f"  {out_dir.resolve()}/range_time_snr.png")

    if pp.write_rd_movie or pp.rd_snapshot_frame is not None:
        print("Building RD movie / snapshot…")
        movie_path = write_rd_outputs(capture, proc, out_dir, max_frames=args.max_frames)
        if movie_path is not None:
            print(f"  {movie_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
