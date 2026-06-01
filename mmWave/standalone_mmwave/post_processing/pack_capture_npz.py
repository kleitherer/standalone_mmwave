#!/usr/bin/env python3
"""
Pack standalone captures into mmw-style NPZ (``radar_data`` + ``radar_time``).

Output layout for ``decode_data``:

  **Preferred (new captures):** one ROS-sized record per frame — wire (399360 B) plus
  the 256-byte frame trailer from UDP (``ros_frame_size`` = 399616 B, 199808 int16).
  Matches ``rosbag_decode_2cam.py`` / ``20250528124759.npz``.

  **Legacy wire-only (existing captures):** wire only (399360 B, 199680 int16), no
  padding. Still decodes cleanly; do **not** zero-pad to 399616 — that merges 256 B
  into the last chirp (4152→4408) and ``decode_data`` drops every frame.

Requires ``raw/wire_frame_NNNNNN.npy`` from capture (see ``radar_receiver.read_frame``).
Without wire files, packing falls back to ``attach_lvds_chirp_headers()`` which does
not round-trip through ``decode_data``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from processing.lvds_frame import (
    attach_lvds_chirp_headers,
    ros_frame_byte_size,
    wire_frame_byte_size,
)

# Reference ROS NPZ (6843 3Tx v0.5): 199808 int16 = wire + real frame trailer.
REF_FRAME_INT16 = 199808
REF_FRAME_BYTES = REF_FRAME_INT16 * 2


def _wire_path_for_adc(adc_path: Path) -> Path:
    return adc_path.parent / adc_path.name.replace("frame_", "wire_frame_", 1)


def load_wire_frame(adc_path: Path, params: dict) -> tuple[np.ndarray, str]:
    """
    Load captured wire / ROS frame int16 for one frame.

    Returns (data, source) where source is ``"capture"`` or ``"reconstructed"``.
    Accepts ``wire_frame_size`` (legacy) or ``ros_frame_size`` (preferred) on disk.
    """
    wire_path = _wire_path_for_adc(adc_path)
    wire_int16 = wire_frame_byte_size(params) // 2
    ros_int16 = ros_frame_byte_size(params) // 2
    if wire_path.is_file():
        data = np.load(wire_path).astype(np.int16).ravel()
        if data.size == ros_int16:
            return data, "capture"
        if data.size == wire_int16:
            return data, "capture"
        raise ValueError(
            f"{wire_path.name}: expected {wire_int16} or {ros_int16} int16, got {data.size}"
        )

    adc = np.load(adc_path)
    wire = attach_lvds_chirp_headers(adc, params)
    return wire, "reconstructed"


def pack_radar_data_int16(
    frame_paths: list[Path],
    params: dict,
) -> tuple[np.ndarray, dict]:
    """
    Concatenate per-frame wire/ROS records with **no** synthetic tail padding.
    """
    wire_int16 = wire_frame_byte_size(params) // 2
    ros_int16 = ros_frame_byte_size(params) // 2
    frames: list[np.ndarray] = []
    n_capture = 0
    n_reconstructed = 0
    n_ros_sized = 0
    n_wire_sized = 0
    for p in frame_paths:
        data, src = load_wire_frame(p, params)
        if src == "capture":
            n_capture += 1
        else:
            n_reconstructed += 1
        if data.size == ros_int16:
            n_ros_sized += 1
        elif data.size == wire_int16:
            n_wire_sized += 1
        else:
            raise ValueError(f"{p.name}: unexpected int16 count {data.size}")
        frames.append(data)
    meta = {
        "n_frames": len(frame_paths),
        "wire_from_capture": n_capture,
        "wire_reconstructed": n_reconstructed,
        "ros_sized_frames": n_ros_sized,
        "wire_sized_frames": n_wire_sized,
        "bytes_per_frame": frames[0].nbytes if frames else 0,
        "int16_per_frame": frames[0].size if frames else 0,
    }
    if n_reconstructed:
        warnings.warn(
            f"{n_reconstructed}/{len(frame_paths)} frames use reconstructed wire "
            "(attach_lvds_chirp_headers). decode_data round-trip is NOT reliable — "
            "re-capture with wire_frame_*.npy saved.",
            stacklevel=2,
        )
    if n_wire_sized and not n_ros_sized:
        warnings.warn(
            f"All {n_wire_sized} frames are wire-only ({wire_int16} int16). "
            "decode_data works on the concatenated stream, but bytes/frame differs "
            f"from ROS reference ({ros_int16} int16). Re-capture with updated "
            "radar_receiver for full rosbag-compatible NPZ.",
            stacklevel=2,
        )
    return np.concatenate(frames), meta


def pack_radar_time(
    capture_dir: Path,
    n_frames: int,
    params: dict,
) -> np.ndarray:
    index_csv = capture_dir / "index.csv"
    if index_csv.is_file():
        with index_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))[:n_frames]
        if rows:
            return np.array(
                [int(float(r["timestamp_unix"]) * 1e9) for r in rows],
                dtype=np.int64,
            )
    fps = 1000.0 / float(params.get("frame_time", 22.22))
    t = np.arange(n_frames, dtype=np.int64)
    return (t * (1e9 / fps)).astype(np.int64)


def pack_capture(
    capture_dir: Path,
    out_npz: Path,
    *,
    max_frames: int | None = None,
    with_cube: bool = False,
    with_time: bool = True,
) -> int:
    capture_dir = Path(capture_dir)
    meta = json.loads((capture_dir / "metadata.json").read_text())
    params = meta["radar_params"]
    raw = capture_dir / "raw"
    paths = sorted(raw.glob("frame_*.npy"))
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No frame_*.npy under {raw}")

    radar_data, pack_meta = pack_radar_data_int16(paths, params)
    payload: dict[str, np.ndarray] = {"radar_data": radar_data}
    if with_time:
        payload["radar_time"] = pack_radar_time(capture_dir, len(paths), params)
    if with_cube:
        from processing.mmw_rd import frame_to_mmw_cube

        cubes = [frame_to_mmw_cube(np.load(p), params) for p in paths]
        payload["radar_cube"] = np.concatenate(cubes, axis=0)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **payload)

    n = len(paths)
    mb = radar_data.nbytes / 1e6
    extra = f", cube {payload['radar_cube'].shape}" if with_cube else ""
    print(
        f"Packed {n} frames → {out_npz} ({mb:.1f} MB, "
        f"{pack_meta['bytes_per_frame']} B/frame, "
        f"ros={pack_meta['ros_sized_frames']} wire-only={pack_meta['wire_sized_frames']}"
        f"{extra})"
    )
    return n


def main() -> int:
    p = argparse.ArgumentParser(description="Pack capture to decode_data-compatible NPZ")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--with-cube",
        action="store_true",
        help="Also store radar_cube (offline validation only)",
    )
    args = p.parse_args()
    pack_capture(
        args.capture,
        args.out,
        max_frames=args.max_frames,
        with_cube=args.with_cube,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
