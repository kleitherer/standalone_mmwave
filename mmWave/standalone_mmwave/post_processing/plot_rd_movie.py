#!/usr/bin/env python3
"""
Range–Doppler heatmap movie (ROI-gated).

Settings: config/live_radar_to_max.json

  python3 -m post_processing.plot_rd_movie --capture newccrma
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from post_processing.processing_config import ProcessingConfig, resolve_capture_range_gate
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Range–Doppler movie (settings: config/live_radar_to_max.json)"
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

    from capture_store import CaptureSession

    params = CaptureSession.open(capture).radar_params()
    r_min, r_max, gate_info = resolve_capture_range_gate(
        capture, proc, radar_max_m=float(params["range_max"])
    )
    print(f"Config: {proc.config_path}")
    print(f"  ROI: {r_min:.2f} – {r_max:.2f} m")
    if gate_info is not None:
        proc.save_applied(out_dir / "range_gate.json", gate_info)

    if not pp.write_rd_movie and pp.rd_snapshot_frame is None:
        print("Nothing to write (set write_rd_movie or rd_snapshot_frame in config)")
        return 0

    print(f"  rd_display: {pp.rd_display}  pipeline: {pp.rd_pipeline}")
    if pp.rd_pipeline != "mmw":
        n_doppler_fft, n_range_fft = pp.rd_fft_sizes(params)
        n_slow = int(params.get("n_slow", params["n_chirps"] // params["n_tx"]))
        if n_doppler_fft or n_range_fft:
            print(
                f"  RD display FFT: Doppler {n_doppler_fft or n_slow} bins "
                f"(native {n_slow}), range {n_range_fft or int(params['n_samples'])} bins"
            )
    out_path = write_rd_outputs(capture, proc, out_dir, max_frames=args.max_frames)
    if out_path is not None:
        print(f"Done: {out_path.resolve()}")
    else:
        print(f"Done: snapshot in {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
