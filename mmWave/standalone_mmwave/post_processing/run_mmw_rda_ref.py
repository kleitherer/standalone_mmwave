#!/usr/bin/env python3
"""
RD + RA visualizations using mmw-tracking reference pipeline (unchanged generate_uD_RDA).

Packs raw ``frame_*.npy`` → mmw ``.npz``, then calls ``plot_radar_RA`` / ``plot_radar_RD``
from ``mmw-tracking-versions/utils/pipeline_utils.py``.

  python3 -m post_processing.run_mmw_rda_ref --capture walk
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_MMW = _ROOT.parent / "mmw-tracking-versions"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_MMW) not in sys.path:
    sys.path.insert(0, str(_MMW))

from capture_store import CaptureSession

def _resolve_capture(p: Path) -> Path:
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    for candidate in (p.resolve(), (_ROOT / "captures" / p).resolve()):
        if candidate.exists():
            return candidate
    return p.resolve()


def _load_radar_config(capture_dir: Path, cfg_override: Path | None) -> Path:
    if cfg_override is not None:
        return cfg_override.resolve()
    meta = json.loads((capture_dir / "metadata.json").read_text())
    cfg = Path(meta["radar_cfg"])
    if not cfg.is_file():
        cfg = _MMW / "waveform_configs" / Path(meta["radar_cfg"]).name
    if not cfg.is_file():
        raise FileNotFoundError(f"Radar cfg not found: {meta['radar_cfg']}")
    return cfg.resolve()


def run_rda(
    capture_dir: Path,
    *,
    out_dir: Path,
    radar_config_path: Path,
    capture_time: float | None = None,
    max_frames: int | None = None,
) -> None:
    from utils.pipeline_utils import plot_radar_RA, plot_radar_RD
    from utils.xwr_raw.radar_config import RadarConfig

    from capture_store import CaptureSession
    from processing.mmw_rd import frame_to_mmw_cube

    capture_dir = _resolve_capture(capture_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = CaptureSession.open(capture_dir)
    paths = session.frame_paths()
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No frames under {capture_dir}")

    with open(radar_config_path) as f:
        radar_params = RadarConfig(f.readlines()).get_params()

    n_samples = int(radar_params["n_samples"])
    range_res = float(radar_params["range_res"])
    velocity_res = float(radar_params["velocity_res"])
    n_chirps = int(radar_params["n_chirps"])
    r_axis = np.arange(n_samples) * range_res
    d_axis = np.arange(-n_chirps // 2, n_chirps // 2) * velocity_res

    meta_params = session.radar_params()
    cubes = [frame_to_mmw_cube(np.load(p), meta_params) for p in paths]
    radar_cube = np.concatenate(cubes, axis=0)
    radar_file_name = capture_dir.name

    # Match generate_uD_RDA.yaml defaults
    args = SimpleNamespace(
        output_dir=str(out_dir),
        n_angle_fft=128,
        n_uD_fft=128,
        overlap_ratio=0.875,
        fov_deg=60,
        fc=60e9,
        c=299_792_458.0,
        range_window_size=20,
        box_size=1.5,
        plot_scale=50,
        use_uD_vmin_vmax=False,
        separate_tracks=False,
    )
    args.radar_fps = radar_params["fps"]
    args.range_res = radar_params["range_res"]

    if capture_time is None:
        capture_time = radar_cube.shape[0] / float(radar_params["fps"])
    args.capture_time = capture_time

    n_keep = min(radar_cube.shape[0], int(capture_time * radar_params["fps"]) + 1)
    radar_cube = radar_cube[:n_keep, ...]
    print(f"Loaded {len(paths)} frames → cube {radar_cube.shape} ({n_keep} used @ {radar_params['fps']:.2f} fps)")

    if radar_params["n_tx"] == 3:
        radar_cube = radar_cube[:, :, :8, :]

    print("Generating RA heatmap movie …")
    plot_radar_RA(radar_cube, args, radar_file_name, r_axis)

    print("Generating RD heatmap movie …")
    plot_radar_RD(radar_cube, args, radar_file_name, r_axis, d_axis)

    ra_mp4 = out_dir / f"RA_heatmaps/{radar_file_name}_RA.mp4"
    rd_mp4 = out_dir / f"RD_heatmaps/{radar_file_name}_RD.mp4"
    print(f"\nDone ({n_keep} frames)")
    if ra_mp4.is_file():
        print(f"  RA: {ra_mp4}")
    if rd_mp4.is_file():
        print(f"  RD: {rd_mp4}")


def main() -> int:
    p = argparse.ArgumentParser(description="RD/RA via mmw reference pipeline (no uD)")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--radar-config", type=Path, default=None)
    p.add_argument(
        "--capture-time",
        type=float,
        default=None,
        help="Seconds to process (default: full capture)",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Limit frames when packing")
    args = p.parse_args()

    capture = _resolve_capture(args.capture)
    out_dir = args.out_dir or (capture / "analysis" / "debug")
    cfg = _load_radar_config(capture, args.radar_config)
    run_rda(
        capture,
        out_dir=out_dir,
        radar_config_path=cfg,
        capture_time=args.capture_time,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
