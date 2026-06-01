#!/usr/bin/env python3
"""
Range–time power heatmap (mmw ``rd_heatmap.py`` conventions).

Per frame (same chain as range–Doppler movies, collapsed over Doppler):

  1. Complex ADC cube
  2. ``RD(..., declutter=True, window=True)`` — chirp-mean DC removal + Hann window,
     then Doppler FFT then range FFT
  3. Power = ``mean_antennas(|RDa|²)`` (non-coherent combine)
  4. Range gate (``processing.roi_min_m`` / ``roi_max_m`` from config)
  5. ``pw2db`` → ``10·log10(power + 1e-9)``
  6. Max over Doppler → one power value per range bin → range–time image

Not SNR (no per-frame median noise floor). Not multi-frame background subtraction.

Input (``--capture``):

- **Capture directory** — reads ``raw/frame_*.npy`` (int16 per-frame ADC), one file per frame.
- **Packed ``.npz``** — mmw-style ``radar_data`` + ``radar_time`` (e.g. ``20250528124759.npz``);
  decoded via ``mmw-tracking`` ``decode_data`` (same as ``pack_capture_npz`` output).

Output: ``<capture>/analysis/range_time_power.png`` and ``range_time_power.npz``
(``power_db``, ``time_s``, ``range_m``). That output NPZ is processed metrics — not raw ADC.

  python3 -m post_processing.plot_range_time_power --capture testing_gesture_osc
  python3 -m post_processing.plot_range_time_power --capture captures/radar/20250528124759.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_MMW = _ROOT.parent / "mmw-tracking-versions"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from capture_store import CaptureSession
from post_processing.processing_config import (
    ProcessingConfig,
    resolve_capture_range_gate,
    resolve_range_gate,
)
from post_processing.rd_maps import _frame_times
from processing.mmw_rd import mmw_range_doppler_axes, mmw_rd, mmw_rd_power_db
from processing.rd_map import frame_to_rd_power_db
from radar_config import RadarConfig


@dataclass
class RangeTimePowerVolume:
    power_db: np.ndarray  # (n_frames, n_range) — max over Doppler per frame
    time_s: np.ndarray
    range_m: np.ndarray
    session_id: str


def _load_mmw_decode_data():
    path = _MMW / "utils" / "radar_capture_utils.py"
    if not path.is_file():
        raise FileNotFoundError(f"mmw decode_data not found: {path}")
    spec = importlib.util.spec_from_file_location("mmw_radar_capture_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to import mmw radar_capture_utils")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DCA1000.decode_data


def _load_params_from_config(proc: ProcessingConfig) -> dict[str, Any]:
    raw_cfg = json.loads(proc.config_path.read_text())["device"]["radar_cfg"]
    cfg_path = Path(raw_cfg)
    if not cfg_path.is_absolute():
        candidates = [
            (_ROOT / cfg_path).resolve(),
            (proc.config_path.parent / cfg_path).resolve(),
            (_MMW / "waveform_configs" / cfg_path.name).resolve(),
        ]
        cfg_path = next((p for p in candidates if p.is_file()), candidates[0])
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Radar cfg not found: {raw_cfg}")
    lines = cfg_path.read_text().splitlines()
    return dict(RadarConfig(lines).get_params())


def _range_mask(params: dict[str, Any], range_gate_m: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    range_axis, _ = mmw_range_doppler_axes(params)
    r_lo, r_hi = range_gate_m
    r_mask = (range_axis >= r_lo) & (range_axis <= r_hi)
    return range_axis[r_mask], r_mask


def _rd_to_range_time_db(rd_db: np.ndarray, r_mask: np.ndarray) -> np.ndarray:
    return np.max(rd_db[:, r_mask], axis=0)


def _time_s_from_packed_npz(radar_time: np.ndarray, n_frames: int) -> np.ndarray:
    t = np.asarray(radar_time[:n_frames], dtype=np.float64)
    if t.size >= 2 and t[-1] > 1e12:
        t = (t - t[0]) * 1e-9
    elif t.size >= 2 and t[-1] > 1e6:
        t = (t - t[0]) * 1e-3
    elif t.size >= 2:
        t = t - t[0]
    if t.size != n_frames:
        fps = 1000.0 / 22.22
        t = np.arange(n_frames, dtype=np.float64) / fps
    return t


def build_range_time_power_from_npz(
    npz_path: Path,
    *,
    params: dict[str, Any],
    range_gate_m: tuple[float, float],
    max_frames: int = 0,
    show_progress: bool = True,
) -> RangeTimePowerVolume:
    """Decode mmw ``radar_data`` NPZ and stack range–time power (dB)."""
    decode_data = _load_mmw_decode_data()
    with np.load(npz_path) as z:
        if "radar_data" not in z:
            raise ValueError(f"{npz_path} has no radar_data array (expected packed mmw NPZ)")
        radar_data = z["radar_data"]
        radar_time = z["radar_time"] if "radar_time" in z else None

    decoded, _, _info, _num_frames = decode_data(
        radar_data.tobytes(),
        num_Tx=int(params["n_tx"]),
        num_Rx=int(params["n_rx"]),
    )
    n_total = decoded.shape[0]
    if max_frames > 0:
        decoded = decoded[:max_frames]
    if decoded.shape[0] == 0:
        raise RuntimeError(f"decode_data returned no frames from {npz_path}")

    range_m, r_mask = _range_mask(params, range_gate_m)
    rt_list: list[np.ndarray] = []
    for i in range(decoded.shape[0]):
        rd_db = mmw_rd_power_db(
            mmw_rd(decoded[i : i + 1], declutter=True, window=True),
            frame_index=0,
        )
        rt_list.append(_rd_to_range_time_db(rd_db, r_mask))
        if show_progress and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{decoded.shape[0]} frames…", flush=True)

    power_rt = np.stack(rt_list, axis=0)
    if radar_time is not None and radar_time.size >= power_rt.shape[0]:
        time_s = _time_s_from_packed_npz(radar_time, power_rt.shape[0])
    else:
        fps = 1000.0 / float(params.get("frame_time", 22.22))
        time_s = np.arange(power_rt.shape[0], dtype=np.float64) / fps

    if n_total != (radar_time.size if radar_time is not None else n_total):
        print(
            f"  Note: decode_data got {n_total} frames"
            + (f" from {radar_time.size} radar_time stamps" if radar_time is not None else "")
        )
    return RangeTimePowerVolume(
        power_db=power_rt,
        time_s=time_s,
        range_m=range_m,
        session_id=npz_path.stem,
    )


def build_range_time_power(
    capture_path: Path,
    *,
    range_gate_m: tuple[float, float],
    max_frames: int = 0,
    show_progress: bool = True,
) -> RangeTimePowerVolume:
    """
    Stack per-frame range–time power (dB).

    Uses ``processing.mmw_rd`` / ``frame_to_rd_power_db(..., pipeline='mmw')`` —
    matches ``rd_heatmap.py`` (chirp-mean declutter only, no clutter map).
    """
    session = CaptureSession.open(Path(capture_path))
    params = session.radar_params()
    range_m, r_mask = _range_mask(params, range_gate_m)

    paths = session.frame_paths()
    if max_frames > 0:
        paths = paths[:max_frames]
    if not paths:
        raise RuntimeError(f"No frames in {capture_path}")

    rt_list: list[np.ndarray] = []
    for i, path in enumerate(paths):
        rd_db = frame_to_rd_power_db(
            np.load(path),
            params,
            pipeline="mmw",
            declutter=True,
            window=True,
        )
        rt_list.append(_rd_to_range_time_db(rd_db, r_mask))

        if show_progress and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(paths)} frames…", flush=True)

    power_rt = np.stack(rt_list, axis=0)
    time_s = _frame_times(session, power_rt.shape[0])
    return RangeTimePowerVolume(
        power_db=power_rt,
        time_s=time_s,
        range_m=range_m,
        session_id=session.root.name,
    )


def _plot_range_time_power(
    vol: RangeTimePowerVolume,
    out_dir: Path,
    *,
    range_max_m: float,
    vmin_db: float,
    vmax_db: float,
    use_percentile: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = vol.time_s
    r = vol.range_m
    power = vol.power_db
    extent = [t[0], t[-1] if len(t) > 1 else t[0] + 1, r[0], r[-1]]

    if use_percentile:
        vmin = float(np.percentile(power, 5))
        vmax = float(np.percentile(power, 99))
    else:
        vmin, vmax = vmin_db, vmax_db

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        power.T,
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
    ax.set_title(
        f"{vol.session_id} — Power (dB), max over Doppler  [{r[0]:.1f}–{range_max_m:.1f} m]"
    )
    fig.colorbar(im, ax=ax, label="Power (dB)")
    fig.tight_layout()
    fig.savefig(out_dir / "range_time_power.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _resolve_input_path(p: Path) -> Path:
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
        description="Range–time power (mmw rd_heatmap chain; settings: live_radar_to_max.json)"
    )
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument(
        "--range-min-m",
        type=float,
        default=None,
        help="Override config ROI min (m); default: processing.roi_min_m",
    )
    p.add_argument(
        "--range-max-m",
        type=float,
        default=None,
        help="Override config ROI max (m); default: processing.roi_max_m",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = _resolve_input_path(args.capture).resolve()
    use_npz = source.suffix.lower() == ".npz"
    if use_npz:
        out_dir = args.out_dir or (source.parent / source.stem / "analysis")
    else:
        out_dir = args.out_dir or (source / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = ProcessingConfig.load(args.config)
    pp = proc.post_processing

    if use_npz:
        params = _load_params_from_config(proc)
        radar_max = float(params["range_max"])
        r_min, r_max = resolve_range_gate(proc, estimated_max_m=radar_max)
        gate_info = None
        input_label = f"packed NPZ ({source.name})"
    else:
        params = CaptureSession.open(source).radar_params()
        radar_max = float(params["range_max"])
        r_min, r_max, gate_info = resolve_capture_range_gate(
            source, proc, radar_max_m=radar_max
        )
        input_label = f"capture dir (raw/frame_*.npy)"

    if args.range_min_m is not None:
        r_min = float(args.range_min_m)
    if args.range_max_m is not None:
        r_max = min(float(args.range_max_m), radar_max)
    if r_max <= r_min:
        r_max = min(radar_max, r_min + 0.5)

    print(f"Config: {proc.config_path}")
    print(f"  Input: {input_label}")
    print(f"  ROI: {r_min:.2f} – {r_max:.2f} m")
    print("  Pipeline: mmw (chirp-mean declutter + Hann, pw2db, no BG/SNR)")
    if gate_info is not None:
        proc.save_applied(out_dir / "range_gate.json", gate_info)

    print("Building range–time power…")
    if use_npz:
        vol = build_range_time_power_from_npz(
            source,
            params=params,
            range_gate_m=(r_min, r_max),
            max_frames=args.max_frames,
        )
    else:
        vol = build_range_time_power(
            source,
            range_gate_m=(r_min, r_max),
            max_frames=args.max_frames,
        )
    _plot_range_time_power(
        vol,
        out_dir,
        range_max_m=r_max,
        vmin_db=pp.rd_vmin_db,
        vmax_db=pp.rd_vmax_db,
        use_percentile=pp.percentile_color_scale,
    )
    np.savez_compressed(
        out_dir / "range_time_power.npz",
        power_db=vol.power_db,
        time_s=vol.time_s,
        range_m=vol.range_m,
        range_min_m=r_min,
        range_max_m=r_max,
    )
    print(f"  {vol.power_db.shape[0]} frames × {vol.power_db.shape[1]} range bins")
    print(f"  {out_dir.resolve()}/range_time_power.png")
    print(f"  {out_dir.resolve()}/range_time_power.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
