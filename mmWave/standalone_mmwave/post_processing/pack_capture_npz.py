#!/usr/bin/env python3
"""Pack standalone ``raw/frame_*.npy`` captures into mmw-style ``radar_data`` NPZ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from post_processing.compare_mmw_capture import _synthetic_mmw_stream

HSI2 = b"\xc9\x0c\xcc\x09\xc9\x0c\xcc\x09"


def pack_capture(
    capture_dir: Path,
    out_npz: Path,
    *,
    max_frames: int | None = None,
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

    chunks: list[bytes] = []
    for i, p in enumerate(paths):
        frame = np.load(p)
        stream = _synthetic_mmw_stream(frame, params, iq_mode="standalone_lvds")
        if i > 0:
            chunks.append(HSI2)
        chunks.append(stream)

    radar_data = np.frombuffer(b"".join(chunks), dtype=np.uint8)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, radar_data=radar_data)
    print(f"Packed {len(paths)} frames → {out_npz} ({radar_data.nbytes / 1e6:.1f} MB)")
    return len(paths)


def main() -> int:
    p = argparse.ArgumentParser(description="Pack .npy capture to mmw radar_data .npz")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()
    pack_capture(args.capture, args.out, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
