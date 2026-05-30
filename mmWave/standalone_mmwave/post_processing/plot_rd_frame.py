#!/usr/bin/env python3
"""
Single-frame RD heatmap for debugging (compare raw / decluttered / SNR).

  python3 -m post_processing.plot_rd_frame --capture newccrma --time 11
  python3 -m post_processing.plot_rd_frame --capture newccrma --frame 495

Writes comparison PNGs under <capture>/analysis/debug/
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
from post_processing.processing_config import ProcessingConfig, resolve_capture_range_gate
from post_processing.rd_maps import _frame_times, rd_to_snr_db
from post_processing.rd_plot_mmw import color_limits, plot_rd_frame, prepare_rd_for_display
from processing.rd_map import frame_to_rd_power_db
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


def _pick_frame(session: CaptureSession, *, frame: int | None, time_s: float | None) -> tuple[int, float]:
    paths = session.frame_paths()
    n = len(paths)
    if n == 0:
        raise RuntimeError("No frames in capture")
    times = _frame_times(session, n)
    if frame is not None:
        idx = int(frame)
        if idx < 0 or idx >= n:
            raise ValueError(f"--frame must be in [0, {n - 1}]")
        return idx, float(times[idx])
    if time_s is not None:
        idx = int(np.argmin(np.abs(times - float(time_s))))
        return idx, float(times[idx])
    raise ValueError("Specify --frame or --time")


def _debug_shapes(frame_int16: np.ndarray, params: dict, fft_kw: dict) -> None:
    """Walk the RD pipeline for one frame, printing shapes + the channel axis.

    Verifies (1) the saved frame size matches what the .cfg implies, and
    (2) channels are combined over the antenna axis (axis=1), not range/Doppler.
    """
    from processing.adc_cube import frame_to_adc_cube
    from processing.rd_map import adc_to_virtual_cube
    from processing.rda import compute_rda

    n_chirps = int(params["n_chirps"])
    n_tx = int(params["n_tx"])
    n_rx = int(params["n_rx"])
    n_samples = int(params["n_samples"])
    fmt = int(params.get("adc_output_fmt", 1))
    expected = n_chirps * n_rx * n_samples * (2 if fmt > 0 else 1)

    print("── shape / axis debug ────────────────────────────────────")
    ok = "OK" if frame_int16.size == expected else "MISMATCH (config != data!)"
    print(f"  raw int16 frame   : {frame_int16.shape}  expected {expected}  -> {ok}")

    adc = frame_to_adc_cube(frame_int16, params)
    print(f"  adc cube          : {adc.shape}  (n_chirps, n_rx, n_samples)")

    virt = adc_to_virtual_cube(adc, n_tx, n_rx)
    print(f"  radar_cube (virt) : {virt.shape}  (n_slow, n_ant, n_samples)")
    print(f"    channel axis    : axis=1  -> n_ant = n_tx*n_rx = {n_tx}*{n_rx} = {n_tx * n_rx}")

    print("  FFT axes on radar_cube (n_slow, n_ant, n_samples):")
    print("    range   FFT -> axis=2 (samples)")
    print("    Doppler FFT -> axis=0 (slow-time) + fftshift")
    print("    antenna combine -> axis=1")
    rda = compute_rda(virt, **fft_kw)
    print(f"  RDa per antenna   : {rda.shape}  (n_doppler, n_ant, n_range)")

    sum_rx = np.mean(np.abs(rda) ** 2, axis=1)  # combine channels over antenna axis
    print(f"  sum_rx (combined) : {sum_rx.shape}  (n_doppler, n_range)  [mean |RD|^2 over axis=1]")
    print("──────────────────────────────────────────────────────────")


def parse_args():
    p = argparse.ArgumentParser(description="Debug RD map at one frame")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--time", type=float, default=None, help="Pick nearest frame to this time (s)")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--debug-shapes", action="store_true", help="Print pipeline shapes + channel axis for the chosen frame")
    p.add_argument("--no-pad", action="store_true", help="Disable zero-padding; plot native FFT sizes (32 Doppler x 256 range)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    capture = _resolve_capture_path(args.capture).resolve()
    out_dir = args.out_dir or (capture / "analysis" / "debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = ProcessingConfig.load(args.config)
    session = CaptureSession.open(capture)
    params = session.radar_params()
    paths = session.frame_paths()

    frame_idx, t = _pick_frame(session, frame=args.frame, time_s=args.time)
    r_min, r_max, _ = resolve_capture_range_gate(
        capture, proc, radar_max_m=float(params["range_max"])
    )
    if args.no_pad:
        n_doppler_fft, n_range_fft = None, None  # native: 32 Doppler x 256 range
    else:
        n_doppler_fft, n_range_fft = proc.post_processing.rd_fft_sizes(params)
    range_axis, doppler_axis = range_doppler_axes(
        params, n_doppler_fft=n_doppler_fft, n_range_fft=n_range_fft
    )
    r_mask = (range_axis >= r_min) & (range_axis <= r_max)
    range_m = range_axis[r_mask]
    doppler_mps = doppler_axis

    pp = proc.post_processing
    fft_kw = {"n_doppler_fft": n_doppler_fft, "n_range_fft": n_range_fft}
    # RD map computed on-the-fly from the raw int16 I/Q frame (decode -> FFTs -> dB).
    frame0 = np.load(paths[frame_idx])
    if args.debug_shapes:
        _debug_shapes(frame0, params, fft_kw)
    raw = frame_to_rd_power_db(frame0, params, **fft_kw)
    # Static-clutter background = mean of the first n_cal frames' RD maps, also live.
    n_cal = min(len(paths), max(1, proc.declutter_mean_frames))
    bg = np.mean(
        np.stack(
            [frame_to_rd_power_db(np.load(p), params, **fft_kw) for p in paths[:n_cal]],
            axis=0,
        ),
        axis=0,
    )

    raw_roi = raw[:, r_mask]
    dec_roi = raw_roi - bg[:, r_mask]
    snr_roi = rd_to_snr_db(dec_roi)

    n_slow = int(params.get("n_slow", params["n_chirps"] // params["n_tx"]))
    print(f"Config: {proc.config_path}")
    print(
        f"  frame {frame_idx}  t={t:.3f} s  ROI {r_min:.2f}–{r_max:.2f} m  "
        f"Doppler {raw.shape[0]} bins (native {n_slow})"
    )

    for name, data in [("raw_power", raw_roi), ("decluttered", dec_roi), ("snr", snr_roi)]:
        vmin, vmax = color_limits(
            data,
            use_percentile=pp.percentile_color_scale,
            vmin_db=pp.rd_vmin_db,
            vmax_db=pp.rd_vmax_db,
        )
        out = out_dir / f"rd_frame{frame_idx}_t{t:.1f}s_{name}.png"
        plot_rd_frame(
            data,
            range_m,
            doppler_mps,
            out,
            title=f"{capture.name} frame {frame_idx} (t={t:.2f}s) — {name}",
            vmin=vmin,
            vmax=vmax,
            upsample=pp.rd_display_upsample,
            interpolation=pp.rd_interpolation,
        )
        peak = np.unravel_index(int(np.argmax(data)), data.shape)
        print(
            f"  {name:12s} → {out.name}  "
            f"peak {data[peak]:.1f} dB @ {range_m[peak[1]]:.2f} m, {doppler_mps[peak[0]]:+.2f} m/s"
        )

    print(f"Done: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
