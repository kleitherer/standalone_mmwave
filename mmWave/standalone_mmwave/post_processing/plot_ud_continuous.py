#!/usr/bin/env python3
"""
Micro-Doppler from a full capture cube (``.npz`` or ``frame_*.npy``).

Default: smooth_frame declutter + walk ROI + TX0/RX0 + column normalization.

  python3 -m post_processing.plot_ud_continuous \\
    --npz captures/potential/analysis/mmw_input/potential.npz

If motion is clearest with per_frame:
  --declutter per_frame
  # defaults: highpass 0.2 Hz + narrow 0 m/s notch (guard off)
  # more 0 m/s cleanup: --zero-doppler-guard-mps 0.25
  # less filtering: --highpass-hz 0 --zero-doppler-notch 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_MMW = _ROOT.parent / "mmw-tracking-versions"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_MMW) not in sys.path:
    sys.path.insert(0, str(_MMW))

from capture_store import CaptureSession
from processing.mmw_rd import frame_to_mmw_cube
from processing.ud_continuous import (
    micro_doppler_from_cube,
    uD_axis_mps,
    uD_bins_per_second,
)


def _capture_from_npz(npz_path: Path) -> Path | None:
    name = npz_path.stem
    candidate = npz_path.parents[2] if npz_path.parent.name == "mmw_input" else None
    if candidate is not None and candidate.name == name and (candidate / "metadata.json").is_file():
        return candidate
    cap = _ROOT / "captures" / name
    return cap if (cap / "metadata.json").is_file() else None


def load_radar_cube_npz(npz_path: Path, capture_dir: Path | None = None) -> np.ndarray:
    npz_path = Path(npz_path)
    with np.load(npz_path) as z:
        if "radar_cube" in z:
            return z["radar_cube"]

    capture_dir = capture_dir or _capture_from_npz(npz_path)
    if capture_dir is None:
        raise FileNotFoundError(
            f"{npz_path} has no radar_cube array; pass --capture or repack with --with-cube"
        )
    return load_radar_cube_capture(capture_dir)


def load_radar_cube_capture(capture_dir: Path) -> np.ndarray:
    session = CaptureSession.open(capture_dir)
    params = session.radar_params()
    paths = session.frame_paths()
    if not paths:
        raise FileNotFoundError(f"No frames under {capture_dir}")
    cubes = [frame_to_mmw_cube(np.load(p), params) for p in paths]
    return np.concatenate(cubes, axis=0)


from post_processing.capture_meta import roi_from_metadata
    uD: np.ndarray,
    *,
    out_png: Path,
    title: str,
    n_uD_fft: int,
    uD_bins_ps: int,
    uD_axis: np.ndarray,
    plot_scale: float = 50.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(6, uD.shape[1] / plot_scale)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    im = ax.imshow(uD, aspect="auto", interpolation="none", cmap="jet")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Doppler (m/s)")
    ticks = np.linspace(0, uD.shape[1], 11)
    ax.set_xticks(ticks, np.round(ticks / uD_bins_ps, 2))
    ax.set_yticks(
        np.linspace(0, n_uD_fft, 11),
        np.round(np.linspace(uD_axis[0], uD_axis[-1], 11), 2),
    )
    ax.invert_yaxis()
    fig.tight_layout(pad=0.5)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Micro-Doppler from full radar cube")
    p.add_argument("--npz", type=Path, help="Packed .npz (radar_cube or linked capture)")
    p.add_argument("--capture", type=Path, help="Capture dir with raw/frame_*.npy")
    p.add_argument("--out", type=Path, default=None, help="Output PNG")
    p.add_argument(
        "--declutter",
        choices=("ema", "running", "smooth_frame", "per_frame", "global", "none"),
        default="smooth_frame",
        help="Clutter removal (default: smooth_frame = per-frame mean, Gaussian-smoothed over time)",
    )
    p.add_argument(
        "--stft-mode",
        choices=("per_frame", "continuous"),
        default="continuous",
        help="per_frame = STFT within each CPI (avoids cross-frame stripes)",
    )
    p.add_argument(
        "--clutter-sec",
        type=float,
        default=1.0,
        help="Running clutter window in seconds (running mode)",
    )
    p.add_argument(
        "--ema-tau-sec",
        type=float,
        default=0.5,
        help="EMA clutter time constant in seconds (ema mode)",
    )
    p.add_argument(
        "--highpass-hz",
        type=float,
        default=None,
        help="High-pass slow-time after declutter (default: 0.2 for per_frame, else 0.3; 0 = off)",
    )
    p.add_argument(
        "--zero-doppler-notch",
        type=int,
        default=None,
        metavar="BINS",
        help="Half-width (bins) of 0 m/s ridge to subtract (default: 5 for per_frame, else 0)",
    )
    p.add_argument(
        "--zero-doppler-guard-mps",
        type=float,
        default=None,
        help="Blank |Doppler| below this speed (default: off; try 0.2–0.3 if 0 m/s pulses remain)",
    )
    p.add_argument(
        "--mti",
        action="store_true",
        help="Chirp-difference MTI before STFT (extra 0 m/s suppression)",
    )
    p.add_argument(
        "--antenna",
        choices=("mean", "tx0_rx0"),
        default="tx0_rx0",
        help="Antenna combine before STFT",
    )
    p.add_argument("--range-min-m", type=float, default=None, help="Range gate min (default: capture ROI)")
    p.add_argument("--range-max-m", type=float, default=None, help="Range gate max (default: capture ROI)")
    p.add_argument("--full-range", action="store_true", help="Use all range bins (ignore ROI)")
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip per-time-column median subtraction (display only)",
    )
    args = p.parse_args()

    if args.npz is None and args.capture is None:
        p.error("Provide --npz and/or --capture")

    capture_dir = Path(args.capture).resolve() if args.capture else None
    if args.npz is not None:
        cube = load_radar_cube_npz(Path(args.npz), capture_dir)
        if capture_dir is None:
            capture_dir = _capture_from_npz(Path(args.npz))
    else:
        capture_dir = capture_dir or Path(args.capture).resolve()
        cube = load_radar_cube_capture(capture_dir)

    assert capture_dir is not None
    meta = json.loads((capture_dir / "metadata.json").read_text())
    params = meta["radar_params"]
    name = capture_dir.name

    n_uD_fft = 128
    overlap_ratio = 0.875
    fps = float(params.get("fps") or round(1000.0 / float(params["frame_time"])))
    n_slow = int(params.get("n_slow", params["n_chirps"] // params["n_tx"]))
    velocity_max = float(params["velocity_max"])
    range_res = float(params["range_res"])

    if args.full_range:
        range_gate = (None, None)
    else:
        roi_lo, roi_hi = roi_from_metadata(meta)
        range_gate = (
            args.range_min_m if args.range_min_m is not None else roi_lo,
            args.range_max_m if args.range_max_m is not None else roi_hi,
        )

    highpass_hz = args.highpass_hz
    if highpass_hz is None:
        highpass_hz = 0.2 if args.declutter == "per_frame" else 0.3

    zero_notch = args.zero_doppler_notch
    if zero_notch is None:
        zero_notch = 5 if args.declutter == "per_frame" else 0

    guard_mps = args.zero_doppler_guard_mps if args.zero_doppler_guard_mps is not None else 0.0

    uD = micro_doppler_from_cube(
        cube,
        n_uD_fft=n_uD_fft,
        overlap_ratio=overlap_ratio,
        declutter=args.declutter,
        antenna=args.antenna,
        stft_mode=args.stft_mode,
        range_res=range_res,
        range_gate_m=range_gate,
        running_clutter_sec=args.clutter_sec,
        ema_tau_sec=args.ema_tau_sec,
        n_slow=n_slow,
        fps=fps,
        velocity_max=velocity_max,
        highpass_hz=highpass_hz,
        zero_doppler_notch_bins=zero_notch,
        zero_doppler_guard_mps=guard_mps,
        mti=args.mti,
        normalize_columns=not args.no_normalize,
    )
    uD_axis = uD_axis_mps(n_uD_fft, velocity_max)
    bins_ps = uD_bins_per_second(
        n_slow, n_uD_fft, overlap_ratio, fps, stft_mode=args.stft_mode
    )

    out = args.out or (
        capture_dir / "analysis" / "mmw_uD_RDA" / "tracks_uD_figs" / f"{name}_track_None.png"
    )
    out = Path(out)

    gate_str = "full range" if args.full_range else f"range {range_gate[0]:.1f}–{range_gate[1]:.1f} m"
    save_uD_plot(
        uD,
        out_png=out,
        title=f"{name} uD ({args.declutter}, {args.stft_mode}, {args.antenna}, {gate_str})",
        n_uD_fft=n_uD_fft,
        uD_bins_ps=int(round(bins_ps)),
        uD_axis=uD_axis,
    )
    print(f"Cube {cube.shape} → uD {uD.shape} @ ~{bins_ps:.0f} bins/s")
    print(
        f"  declutter={args.declutter} stft={args.stft_mode} "
        f"antenna={args.antenna} gate={range_gate} "
        f"highpass={highpass_hz}Hz notch={zero_notch}bins"
        + (f" guard={guard_mps}m/s" if guard_mps > 0 else "")
        + (" mti=1" if args.mti else "")
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
