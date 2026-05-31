#!/usr/bin/env python3
"""
Run mmw-tracking ``generate_uD_RDA``-style outputs on standalone captures.

- **uD**: ``plot_ud_continuous`` / ``WALK_UD_PRESET`` (correct cube + ROI + filters)
- **RD**: mmw FFT chain via ``rd_plot_mmw`` (ROI + percentile scaling, not ``plot_radar_RD`` vmax=80)
- **RA**: unchanged mmw ``plot_radar_RA`` (generate_uD_RDA path)

  python3 -m post_processing.run_mmw_rda_ref --capture potential --skip-ra
  python3 -m post_processing.run_mmw_rda_ref --capture potential --with-uD --skip-ra
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


def _render_walk_uD(
    capture_dir: Path,
    *,
    out_dir: Path,
    radar_file_name: str,
    meta_params: dict,
    radar_params: dict,
) -> Path:
    from post_processing.plot_ud_continuous import save_uD_plot
    from processing.mmw_rd import frame_to_mmw_cube
    from processing.ud_continuous import (
        WALK_UD_PRESET,
        micro_doppler_from_cube,
        uD_axis_mps,
        uD_bins_per_second,
    )
    from post_processing.capture_meta import roi_from_metadata

    meta = json.loads((capture_dir / "metadata.json").read_text())
    session = CaptureSession.open(capture_dir)
    paths = session.frame_paths()
    cubes = [frame_to_mmw_cube(np.load(p), meta_params) for p in paths]
    cube = np.concatenate(cubes, axis=0)

    n_uD_fft = 128
    overlap_ratio = 0.875
    fps = float(radar_params["fps"])
    n_slow = int(meta_params.get("n_slow", meta_params["n_chirps"] // meta_params["n_tx"]))
    velocity_max = float(meta_params["velocity_max"])
    range_res = float(meta_params["range_res"])
    range_gate = roi_from_metadata(meta)

    preset = dict(WALK_UD_PRESET)
    uD = micro_doppler_from_cube(
        cube,
        n_uD_fft=n_uD_fft,
        overlap_ratio=overlap_ratio,
        range_res=range_res,
        range_gate_m=range_gate,
        n_slow=n_slow,
        fps=fps,
        velocity_max=velocity_max,
        **preset,
    )
    uD_axis = uD_axis_mps(n_uD_fft, velocity_max)
    bins_ps = int(
        round(
            uD_bins_per_second(
                n_slow, n_uD_fft, overlap_ratio, fps, stft_mode=preset["stft_mode"]
            )
        )
    )

    uD_dir = out_dir / "tracks_uD_figs"
    uD_dir.mkdir(parents=True, exist_ok=True)
    out_png = uD_dir / f"{radar_file_name}_track_None.png"
    save_uD_plot(
        uD,
        out_png=out_png,
        title=f"{radar_file_name} uD (walk preset, {range_gate[0]:.1f}–{range_gate[1]:.1f} m)",
        n_uD_fft=n_uD_fft,
        uD_bins_ps=bins_ps,
        uD_axis=uD_axis,
    )
    print(f"  uD: {out_png}")
    return out_png


def _render_rd_movie(
    capture_dir: Path,
    *,
    out_dir: Path,
    radar_file_name: str,
) -> Path:
    """mmw RD chain + ROI + percentile colors (matches ``rd_heatmap_movie`` quality)."""
    from post_processing.capture_meta import roi_from_metadata
    from post_processing.rd_plot_mmw import collect_rd_frames, color_limits, render_rd_movie

    meta = json.loads((capture_dir / "metadata.json").read_text())
    session = CaptureSession.open(capture_dir)
    params = session.radar_params()
    range_gate = roi_from_metadata(meta)
    frame_time_ms = float(params.get("frame_time", 22.22))
    fps = 1000.0 / frame_time_ms

    print(f"Generating RD heatmap movie (mmw chain, ROI {range_gate[0]:.1f}–{range_gate[1]:.1f} m) …")
    rd_frames, range_m, doppler_mps, time_s = collect_rd_frames(
        capture_dir,
        range_gate_m=range_gate,
        rd_pipeline="mmw",
        rd_full_range=False,
    )
    stack = np.stack(rd_frames, axis=0)
    vmin, vmax = color_limits(stack, use_percentile=True, vmin_db=45.0, vmax_db=140.0)

    rd_dir = out_dir / "RD_heatmaps"
    rd_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = rd_dir / f"{radar_file_name}_RD.mp4"
    render_rd_movie(
        rd_frames,
        range_m,
        doppler_mps,
        time_s,
        out_mp4,
        fps=fps,
        fmt="mp4",
        session_id=radar_file_name,
        dpi=120,
        vmin=vmin,
        vmax=vmax,
        colorbar_label="Power (dB)",
        upsample=1,
        interpolation="nearest",
    )
    print(f"  RD: {out_mp4}  ({len(rd_frames)} frames, vmin={vmin:.1f} vmax={vmax:.1f} dB)")
    return out_mp4


def run_rda(
    capture_dir: Path,
    *,
    out_dir: Path,
    radar_config_path: Path,
    capture_time: float | None = None,
    max_frames: int | None = None,
    with_uD: bool = False,
    skip_ra: bool = False,
    skip_rd: bool = False,
) -> None:
    from utils.pipeline_utils import plot_radar_RA
    from utils.xwr_raw.radar_config import RadarConfig

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
    n_chirps = int(radar_params["n_chirps"])
    r_axis = np.arange(n_samples) * range_res

    meta_params = session.radar_params()
    cubes = [frame_to_mmw_cube(np.load(p), meta_params) for p in paths]
    radar_cube = np.concatenate(cubes, axis=0)
    radar_file_name = capture_dir.name

    args = SimpleNamespace(
        output_dir=str(out_dir),
        n_angle_fft=128,
        fov_deg=60,
        fc=60e9,
        c=299_792_458.0,
    )
    args.radar_fps = radar_params["fps"]
    args.range_res = radar_params["range_res"]

    if capture_time is None:
        capture_time = radar_cube.shape[0] / float(radar_params["fps"])

    n_keep = min(radar_cube.shape[0], int(capture_time * radar_params["fps"]) + 1)
    radar_cube = radar_cube[:n_keep, ...]
    print(f"Loaded {len(paths)} frames → cube {radar_cube.shape} ({n_keep} used @ {radar_params['fps']:.2f} fps)")

    if radar_params["n_tx"] == 3:
        radar_cube = radar_cube[:, :, :8, :]

    if with_uD:
        print("Generating micro-Doppler (walk preset) …")
        _render_walk_uD(
            capture_dir,
            out_dir=out_dir,
            radar_file_name=radar_file_name,
            meta_params=meta_params,
            radar_params=radar_params,
        )

    if not skip_ra:
        print("Generating RA heatmap movie …")
        plot_radar_RA(radar_cube, args, radar_file_name, r_axis)

    if not skip_rd:
        _render_rd_movie(capture_dir, out_dir=out_dir, radar_file_name=radar_file_name)

    ra_mp4 = out_dir / f"RA_heatmaps/{radar_file_name}_RA.mp4"
    rd_mp4 = out_dir / f"RD_heatmaps/{radar_file_name}_RD.mp4"
    print(f"\nDone ({n_keep} frames)")
    if ra_mp4.is_file():
        print(f"  RA: {ra_mp4}")
    if rd_mp4.is_file():
        print(f"  RD: {rd_mp4}")


def main() -> int:
    p = argparse.ArgumentParser(description="mmw-style RDA on standalone captures")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--radar-config", type=Path, default=None)
    p.add_argument(
        "--capture-time",
        type=float,
        default=None,
        help="Seconds to process (default: full capture)",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--with-uD",
        action="store_true",
        help="Regenerate tracks_uD_figs with walk preset",
    )
    p.add_argument("--skip-ra", action="store_true")
    p.add_argument("--skip-rd", action="store_true")
    args = p.parse_args()

    capture = _resolve_capture(args.capture)
    out_dir = args.out_dir or (capture / "analysis" / "mmw_uD_RDA")
    cfg = _load_radar_config(capture, args.radar_config)
    run_rda(
        capture,
        out_dir=out_dir,
        radar_config_path=cfg,
        capture_time=args.capture_time,
        max_frames=args.max_frames,
        with_uD=args.with_uD,
        skip_ra=args.skip_ra,
        skip_rd=args.skip_rd,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
