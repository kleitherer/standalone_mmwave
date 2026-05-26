#!/usr/bin/env python3
"""
Process saved captures → radar cube → range–Doppler.

Usage (from standalone_mmwave/):
  python3 -m processing.process_capture --capture captures/20250526_120000
  python3 -m processing.process_capture --frame captures/.../frame_000001.npy --metadata captures/.../metadata.json --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Parent package: radar_config for .cfg without metadata.json
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processing.cube import (
    frame_to_radar_cube,
    iter_capture_frames,
    load_capture_metadata,
    load_frame,
)
from processing.rda import compute_rda, rda_power_db, range_doppler_axes


def _load_params(args) -> dict:
    if args.metadata:
        meta = json.loads(Path(args.metadata).read_text())
        return meta["radar_params"]
    if args.capture:
        cap = Path(args.capture)
        try:
            from capture_store import CaptureSession

            return CaptureSession.open(cap).radar_params()
        except Exception:
            return load_capture_metadata(cap)["radar_params"]
    if args.cfg:
        from radar_config import RadarConfig

        lines = Path(args.cfg).read_text().splitlines(keepends=True)
        return dict(RadarConfig(lines).get_params())
    raise SystemExit("Need --capture, --metadata, or --cfg")


def main() -> int:
    p = argparse.ArgumentParser(description="Process radar .npy captures to cube + RDA")
    p.add_argument("--capture", type=Path, help="Directory with frame_*.npy + metadata.json")
    p.add_argument("--frame", type=Path, help="Single frame .npy")
    p.add_argument("--metadata", type=Path, help="metadata.json (with --frame)")
    p.add_argument("--cfg", type=Path, help="mmWave .cfg if no metadata")
    p.add_argument("--out", type=Path, help="Save processed npz here")
    p.add_argument("--plot", action="store_true", help="Show range–Doppler (needs matplotlib)")
    p.add_argument("--max-frames", type=int, default=0, help="Limit frames (0=all)")
    args = p.parse_args()

    params = _load_params(args)
    print("Radar params:")
    for k in (
        "n_chirps",
        "n_tx",
        "n_rx",
        "n_samples",
        "range_res",
        "range_max",
        "velocity_res",
        "velocity_max",
        "frame_time",
    ):
        if k in params:
            print(f"  {k}: {params[k]}")

    cubes = []
    rd_maps = []

    if args.frame:
        frames = [(args.frame, load_frame(args.frame))]
    elif args.capture:
        frames = list(iter_capture_frames(args.capture))
    else:
        raise SystemExit("Provide --capture or --frame")

    if args.max_frames:
        frames = frames[: args.max_frames]

    for i, (path, raw) in enumerate(frames):
        cube = frame_to_radar_cube(raw, params)
        rda = compute_rda(cube)
        rd_db = rda_power_db(rda)
        cubes.append(cube)
        rd_maps.append(rd_db)
        print(f"  {path.name}: cube {cube.shape}, RD power {rd_db.shape}")

    cubes = np.stack(cubes, axis=0)
    rd_maps = np.stack(rd_maps, axis=0)
    r_axis, d_axis = range_doppler_axes(params)
    print(f"Stacked cubes {cubes.shape}, RD maps {rd_maps.shape}")

    out = args.out
    if out is None and args.capture:
        out = Path(args.capture) / "processed.npz"
    if out:
        np.savez_compressed(
            out,
            radar_cubes=cubes,
            rd_power_db=rd_maps,
            range_axis=r_axis,
            doppler_axis=d_axis,
            radar_params=json.dumps(params, default=str),
        )
        print(f"Wrote {out.resolve()}")

    if args.plot and len(rd_maps):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.imshow(
            rd_maps[0],
            aspect="auto",
            origin="lower",
            extent=[r_axis[0], r_axis[-1], d_axis[0], d_axis[-1]],
        )
        plt.xlabel("range (m)")
        plt.ylabel("velocity (m/s)")
        plt.title("Range–Doppler (frame 0)")
        plt.colorbar(label="dB")
        plt.tight_layout()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
