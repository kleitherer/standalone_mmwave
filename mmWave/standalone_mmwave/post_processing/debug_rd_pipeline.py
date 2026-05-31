#!/usr/bin/env python3
"""
Debug range–Doppler pipeline: dimensions, per-TX/RX maps, range profile.

  python3 -m post_processing.debug_rd_pipeline --capture newccrma --time 11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from background_model import estimate_rd_background_from_capture
from capture_store import CaptureSession
from post_processing.processing_config import ProcessingConfig, resolve_capture_range_gate
from post_processing.rd_maps import _frame_times
from post_processing.rd_plot_mmw import color_limits, plot_rd_frame
from processing.adc_cube import frame_to_adc_cube
from processing.cube import frame_to_radar_cube
from processing.rda import compute_rda, range_doppler_axes, rda_power_db
from processing.rd_map import frame_to_rd_power_db
from processing.rd_reference import compute_rd_from_frame, tdm_doppler_params


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


def _pick_frame(session, *, frame: int | None, time_s: float | None) -> tuple[int, float]:
    paths = session.frame_paths()
    times = _frame_times(session, len(paths))
    if frame is not None:
        idx = int(frame)
    else:
        idx = int(np.argmin(np.abs(times - float(time_s))))
    return idx, float(times[idx])


def _print_dims(params: dict, adc: np.ndarray) -> None:
    n_tx = int(params["n_tx"])
    n_rx = int(params["n_rx"])
    n_chirps = int(params["n_chirps"])
    n_slow = int(params.get("n_slow", n_chirps // n_tx))
    dp = tdm_doppler_params(params)

    print("=== Array dimensions ===")
    print(f"  adc.shape              = {adc.shape}  (chirps, rx, samples)")
    print(f"  n_chirps_total         = {n_chirps}")
    print(f"  n_chirps_per_tx        = {n_slow}")
    print(f"  n_tx / n_rx            = {n_tx} / {n_rx}")
    print(f"  chirp_time_us          = {params.get('chirp_time_us', params.get('chirp_time'))}")
    print(f"  chirp_period_same_tx_s = {params.get('chirp_period_same_tx_s', dp['chirp_period_same_tx_s']):.6f}")
    print(f"  frame_period_s         = {float(params['frame_time']) * 1e-3:.4f}")
    print(f"  carrier_frequency_hz   = {float(params.get('operating_freq_ghz', 60)) * 1e9:.3e}")
    print(f"  wavelength_m           = {dp['wavelength_m']:.6f}")
    print(f"  velocity_max_mps       = {dp['velocity_max_mps']:.4f}  (metadata {params['velocity_max']:.4f})")
    print(f"  velocity_res_mps       = {dp['velocity_res_mps']:.4f}  (metadata {params['velocity_res']:.4f})")
    print(f"  range_res_m            = {params['range_res']:.6f}")


def parse_args():
    p = argparse.ArgumentParser(description="Debug RD pipeline")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--time", type=float, default=11.0)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    capture = _resolve_capture_path(args.capture).resolve()
    out_dir = args.out_dir or (capture / "analysis" / "debug_rd")
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = ProcessingConfig.load(args.config)
    session = CaptureSession.open(capture)
    params = dict(session.radar_params())
    # Re-parse cfg for corrected TDM velocity if metadata is stale.
    cfg_path = proc.config_path.parent.parent / "mmw-tracking-versions/waveform_configs/xwr68xx_3Tx_wfv0.5_RDAhigh_19.2m.cfg"
    if not cfg_path.is_file():
        cfg_path = _ROOT.parent / "mmw-tracking-versions/waveform_configs/xwr68xx_3Tx_wfv0.5_RDAhigh_19.2m.cfg"
    if cfg_path.is_file():
        from radar_config import RadarConfig

        params.update(RadarConfig(cfg_path.read_text().splitlines()).get_params())

    frame_idx, t = _pick_frame(session, frame=args.frame, time_s=args.time)
    raw_int16 = np.load(session.frame_paths()[frame_idx])
    adc = frame_to_adc_cube(raw_int16, params)
    _print_dims(params, adc)

    r_min, r_max, _ = resolve_capture_range_gate(
        capture, proc, radar_max_m=float(params["range_max"])
    )
    range_axis, doppler_axis = range_doppler_axes(params)
    r_mask = (range_axis >= r_min) & (range_axis <= r_max)
    range_m = range_axis[r_mask]

    n_tx = int(params["n_tx"])
    n_rx = int(params["n_rx"])

    # --- Per TX/RX reference maps (raw power, no clutter removal) ---
    print("\n=== Single TX/RX RD (reference pipeline) ===")
    for tx in range(n_tx):
        for rx in range(n_rx):
            rd_db, _, d_mps = compute_rd_from_frame(
                raw_int16, params, tx_id=tx, rx_id=rx, declutter=False, window=True
            )
            rd_roi = rd_db[:, r_mask]
            vmin, vmax = color_limits(rd_roi, use_percentile=True)
            plot_rd_frame(
                rd_roi,
                range_m,
                d_mps,
                out_dir / f"rd_ref_tx{tx}_rx{rx}_frame{frame_idx}.png",
                title=f"ref TX{tx} RX{rx} frame {frame_idx} (t={t:.2f}s) raw 20log10|RD|",
                vmin=vmin,
                vmax=vmax,
            )
            peak = np.unravel_index(int(np.argmax(rd_roi)), rd_roi.shape)
            print(
                f"  TX{tx} RX{rx}: peak {rd_roi[peak]:.1f} dB @ "
                f"{range_m[peak[1]]:.2f} m, {d_mps[peak[0]]:+.2f} m/s"
            )

    # --- Production RD path (explicit virt-ant / per-TX decimation) ---
    rd_prod = frame_to_rd_power_db(raw_int16, params)[:, r_mask]
    vmin, vmax = color_limits(rd_prod, use_percentile=True)
    plot_rd_frame(
        rd_prod,
        range_m,
        doppler_axis,
        out_dir / f"rd_production_frame{frame_idx}.png",
        title=f"production frame_to_rd_power_db (per-TX virt ant, declutter) frame {frame_idx}",
        vmin=vmin,
        vmax=vmax,
    )

    # --- Current pipeline (virtual antenna mean) ---
    cube = frame_to_radar_cube(raw_int16, params)
    print(f"\n=== Current virtual-antenna cube ===")
    print(f"  radar_cube.shape = {cube.shape}  (slow, virt_ant={n_rx * n_tx}, samples)")
    rda = compute_rda(cube, declutter=False, window=True)
    rd_current = rda_power_db(rda)[:, r_mask]
    # rda_power_db uses 10*log10; reference uses 20*log10 — compare shape not absolute level
    vmin, vmax = color_limits(rd_current, use_percentile=True)
    plot_rd_frame(
        rd_current,
        range_m,
        doppler_axis,
        out_dir / f"rd_current_virtAntMean_frame{frame_idx}.png",
        title=f"current pipeline (virt ant mean, 10log10) frame {frame_idx}",
        vmin=vmin,
        vmax=vmax,
    )

    # --- Same frame, current pipeline with declutter (explains dark zero-Doppler) ---
    rda_dc = compute_rda(cube, declutter=True, window=True)
    rd_dc = rda_power_db(rda_dc)[:, r_mask]
    vmin, vmax = color_limits(rd_dc, use_percentile=True)
    plot_rd_frame(
        rd_dc,
        range_m,
        doppler_axis,
        out_dir / f"rd_current_declutter_frame{frame_idx}.png",
        title=f"current + DC clutter removal (expect dark 0-Doppler row)",
        vmin=vmin,
        vmax=vmax,
    )

    # --- Range profile (TX0 RX0, range FFT only) ---
    x0 = adc[0::n_tx, 0, :]
    range_fft = np.fft.fft(x0 * np.hanning(x0.shape[1]), axis=1)
    profile = 20.0 * np.log10(np.abs(range_fft).mean(axis=0) + 1e-12)
    prof_roi = profile[r_mask]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range_m, prof_roi)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("20·log10|range FFT| (dB)")
    ax.set_title(f"TX0 RX0 mean range profile — frame {frame_idx} (t={t:.2f}s)")
    ax.axvline(0.9, color="red", linestyle="--", alpha=0.6, label="~person range")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"range_profile_tx0_rx0_frame{frame_idx}.png", dpi=150)
    plt.close(fig)
    pk = int(np.argmax(prof_roi))
    pk75 = int(np.argmin(np.abs(range_m - 0.75)))
    print(f"\n=== Range profile TX0 RX0 ===")
    print(f"  peak @ {range_m[pk]:.2f} m ({prof_roi[pk]:.1f} dB)")
    print(f"  @ 0.75 m: {prof_roi[pk75]:.1f} dB")

    # Background-subtracted comparison (matches range-time SNR path)
    n_cal = min(45, frame_idx)
    if n_cal > 0:
        bg_frames = [np.load(p) for p in session.frame_paths()[:n_cal]]
        bg = np.mean(
            np.stack([frame_to_rd_power_db(f, params) for f in bg_frames], axis=0), axis=0
        )
        rd_bg = (frame_to_rd_power_db(raw_int16, params) - bg)[:, r_mask]
        vmin, vmax = color_limits(rd_bg, use_percentile=True)
        plot_rd_frame(
            rd_bg,
            range_m,
            doppler_axis,
            out_dir / f"rd_production_bgSub_frame{frame_idx}.png",
            title=f"production + temporal bg subtract (first {n_cal} frames)",
            vmin=vmin,
            vmax=vmax,
        )
        peak = np.unravel_index(int(np.argmax(rd_bg)), rd_bg.shape)
        print(f"\n=== After temporal background subtract ===")
        print(
            f"  peak {rd_bg[peak]:.1f} dB @ {range_m[peak[1]]:.2f} m, "
            f"{doppler_axis[peak[0]]:+.2f} m/s"
        )
        pk75 = int(np.argmin(np.abs(range_m - 0.75)))
        print(f"  max @ 0.75 m: {rd_bg[:, pk75].max():.1f} dB")

    print(f"\nWrote debug plots: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
