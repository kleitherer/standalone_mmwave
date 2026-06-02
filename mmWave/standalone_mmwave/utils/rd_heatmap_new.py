#!/usr/bin/env python3
"""
Plot an RD (range–Doppler) heat map from a packed radar NPZ (radar_data + decode_data).

Usage (from standalone_mmwave/):
    python3 utils/rd_heatmap_new.py --npz captures/testbonk/analysis/testbonk_fixed.npz --frame 270
    python3 utils/rd_heatmap_new.py --npz path/to/file.npz --frame 270

Movies (--format, --max-frames, --fps): use rd_heatmap_movie_new.py instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.pipeline_utils import RD, pw2db, radarDataLoader
from utils.radar_config_new import RadarConfig

_MMW_ROOT = _ROOT.parent / "mmw-tracking-versions"
RADAR_CONFIG_PATH = (
    _MMW_ROOT / "waveform_configs" / "xwr68xx_3Tx_wfv0.5_RDAhigh_19.2m.cfg"
).resolve()


def apply_range_gate(rd_pw, r_axis, max_range_m):
    """Keep range columns with r <= max_range_m (applied after RD FFT)."""
    keep = r_axis <= max_range_m
    if not np.any(keep):
        raise ValueError(
            f"No range bins <= {max_range_m} m (r_axis spans {r_axis[0]:.3f} .. {r_axis[-1]:.3f})"
        )
    return rd_pw[..., keep], r_axis[keep]


def load_rd_power_stack(
    npz_path: str | Path,
    *,
    max_range_m: float = 3.4,
    radar_config_path: Path | None = None,
):
    """
    Load packed NPZ, decode, RD FFT → linear power maps.

    Returns
    -------
    rd_pw : (n_frames, n_doppler, n_range) float32
    rd_db : pw2db(rd_pw)
    r_axis, d_axis : 1D axes (m, m/s)
    """
    npz_path = Path(npz_path).expanduser().resolve()
    cfg_path = radar_config_path or RADAR_CONFIG_PATH
    with open(cfg_path, "r") as f:
        radar_params = RadarConfig(f.readlines()).get_params()

    loader = radarDataLoader(str(npz_path), radar_params)
    radar_cube, _, _, _, r_axis, d_axis = loader.load_data()
    RDa, _, _ = RD(radar_cube, declutter=True, window=True)
    rd_pw = (np.abs(RDa) ** 2).mean(axis=2).astype(np.float32)
    rd_pw, r_axis = apply_range_gate(rd_pw, r_axis, max_range_m)
    return rd_pw, pw2db(rd_pw), r_axis, d_axis, radar_params


def main():
    parser = argparse.ArgumentParser(description="Plot RD heat map from packed radar NPZ")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (default 0)")
    parser.add_argument(
        "--mean", action="store_true", help="Plot mean power over all frames"
    )
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    parser.add_argument(
        "--npz",
        type=str,
        required=True,
        help="Path to packed NPZ (radar_data + optional radar_time)",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=3.4,
        help="Keep range bins with r <= this (m)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=50.0,
        help="Color scale min (dB)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=95.0,
        help="Color scale max (dB)",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    print(f"Loading: {npz_path}")
    RD_pw, _, r_axis, d_axis, _ = load_rd_power_stack(
        npz_path, max_range_m=args.max_range
    )
    n_f = RD_pw.shape[0]
    print(f"RD stack: {RD_pw.shape}")

    if args.mean:
        frame_data = pw2db(np.mean(RD_pw, axis=0))
        title_suffix = f" (mean over {n_f} frames, r <= {args.max_range} m)"
    else:
        if args.frame < 0 or args.frame >= RD_pw.shape[0]:
            raise ValueError(f"--frame must be in [0, {RD_pw.shape[0] - 1}]")
        frame_data = pw2db(RD_pw[args.frame])
        title_suffix = f" (frame {args.frame}, r <= {args.max_range} m)"

    fig, ax = plt.subplots()
    im = ax.imshow(
        frame_data,
        aspect="auto",
        cmap="jet",
        extent=[r_axis[0], r_axis[-1], d_axis[-1], d_axis[0]],
        origin="upper",
        vmin=args.vmin,
        vmax=args.vmax,
    )
    plt.colorbar(im, ax=ax, label="Power (dB)")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Doppler (m/s)")
    ax.set_title(f"RD heat map — {npz_path.name}{title_suffix}")

    if args.out is not None:
        out_path = Path(args.out).expanduser().resolve()
    else:
        stem = "rd_heatmap_mean" if args.mean else f"rd_heatmap_frame{args.frame}"
        out_path = npz_path.parent / f"{stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    print(f"Range bins kept: {len(r_axis)} (0 .. {r_axis[-1]:.3f} m)")


if __name__ == "__main__":
    main()
