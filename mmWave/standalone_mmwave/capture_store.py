#!/usr/bin/env python3
"""
Archive format for mmWave DCA raw captures (reprocessable later).

Layout
------
captures/<session_id>/
  session.json      # manifest (format, counts, layout)
  metadata.json     # radar_params + hardware (for older tools)
  profile.cfg       # copy of mmWave profile used at capture time
  index.csv         # per-frame index
  raw/
    frame_000001.npy   # int16, flat ADC (same bytes as UDP reassembly)

Always store **raw** frames. Run cube / RDA / Max / tracking in processing later.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

FORMAT_VERSION = 1
RAW_SUBDIR = "raw"


@dataclass
class FrameRecord:
    index: int
    path: Path
    timestamp_unix: float
    n_samples: int


class CaptureSession:
    """Create or open a raw capture session directory."""

    def __init__(self, root: Path, *, write: bool = False):
        self.root = Path(root)
        self.raw_dir = self.root / RAW_SUBDIR
        self._write = write
        self._frame_count = 0
        self._index_rows: list[dict] = []
        self._session: Dict[str, Any] = {}

    @classmethod
    def create(
        cls,
        base_dir: Path,
        *,
        cfg_path: Path,
        radar_params: Dict[str, Any],
        label: str = "",
        cmd_tty: Optional[str] = None,
        dca_ip: str = "192.168.33.180",
        host_data_port: int = 4098,
        notes: str = "",
    ) -> "CaptureSession":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label.strip())
        session_id = f"{stamp}_{safe}" if safe else stamp
        root = Path(base_dir) / session_id
        root.mkdir(parents=True, exist_ok=False)
        (root / RAW_SUBDIR).mkdir()

        session = cls(root, write=True)
        session._session = {
            "format_version": FORMAT_VERSION,
            "session_id": session_id,
            "label": label,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
            "data": {
                "dtype": "int16",
                "layout": "dca_lvds_flat",
                "description": (
                    "Flat int16 ADC frame from DCA1000 UDP reassembly (FrameBuffer). "
                    "De-interleave to complex cube with processing.frame_to_radar_cube()."
                ),
                "frame_int16_count": int(radar_params["frame_size"] // 2),
                "frame_byte_count": int(radar_params["frame_size"]),
                "reshape": {
                    "n_chirps": int(radar_params["n_chirps"]),
                    "n_tx": int(radar_params["n_tx"]),
                    "n_rx": int(radar_params["n_rx"]),
                    "n_samples": int(radar_params["n_samples"]),
                    "adc_output_fmt": int(radar_params.get("adc_output_fmt", 1)),
                },
            },
            "hardware": {
                "cmd_tty": cmd_tty,
                "dca_ip": dca_ip,
                "host_data_port": host_data_port,
            },
            "source_cfg": str(cfg_path.resolve()),
            "n_frames": 0,
            "target_duration_sec": None,
        }

        shutil.copy2(cfg_path, root / "profile.cfg")

        meta = {
            "cfg": str(cfg_path.resolve()),
            "profile_cfg": "profile.cfg",
            "cmd_tty": cmd_tty,
            "dca_ip": dca_ip,
            "host_data_port": host_data_port,
            "radar_params": _json_safe(radar_params),
            "session_id": session_id,
        }
        (root / "metadata.json").write_text(json.dumps(meta, indent=2))

        readme = root / "README.txt"
        readme.write_text(
            f"mmWave raw capture — {session_id}\n\n"
            f"Frames: raw/frame_NNNNNN.npy (int16, {meta['radar_params']['frame_size']//2} samples)\n"
            f"Profile: profile.cfg\n"
            f"Params: metadata.json / session.json\n\n"
            "Reprocess:\n"
            "  python3 -m processing.process_capture --capture .\n"
            "  python3 -m processing.target_detect  (import CaptureSession)\n"
        )
        return session

    def write_frame(self, frame_int16: np.ndarray, *, timestamp: Optional[float] = None) -> FrameRecord:
        if not self._write:
            raise RuntimeError("Session not opened for writing")
        self._frame_count += 1
        ts = time.time() if timestamp is None else timestamp
        name = f"frame_{self._frame_count:06d}.npy"
        path = self.raw_dir / name
        arr = np.asarray(frame_int16, dtype=np.int16).ravel()
        np.save(path, arr)
        rec = FrameRecord(
            index=self._frame_count,
            path=path,
            timestamp_unix=ts,
            n_samples=int(arr.size),
        )
        self._index_rows.append(
            {
                "frame_index": rec.index,
                "filename": name,
                "timestamp_unix": f"{rec.timestamp_unix:.6f}",
                "n_samples": rec.n_samples,
            }
        )
        return rec

    def finalize(self) -> Path:
        if not self._write:
            return self.root
        self._session["n_frames"] = self._frame_count
        (self.root / "session.json").write_text(json.dumps(self._session, indent=2))

        with (self.root / "index.csv").open("w", newline="") as f:
            if self._index_rows:
                w = csv.DictWriter(f, fieldnames=self._index_rows[0].keys())
                w.writeheader()
                w.writerows(self._index_rows)

        return self.root

    @classmethod
    def open(cls, path: Path) -> "CaptureSession":
        root = Path(path)
        if not (root / "session.json").is_file() and (root / "metadata.json").is_file():
            return cls._open_legacy(root)
        session = cls(root, write=False)
        session._session = json.loads((root / "session.json").read_text())
        session._frame_count = int(session._session.get("n_frames", 0))
        return session

    @classmethod
    def _open_legacy(cls, root: Path) -> "CaptureSession":
        """Frames at session root (frame_*.npy) without raw/ subdir."""
        session = cls(root, write=False)
        session._session = {
            "format_version": 0,
            "legacy": True,
            "n_frames": len(list(root.glob("frame_*.npy"))),
        }
        return session

    def radar_params(self) -> Dict[str, Any]:
        meta = json.loads((self.root / "metadata.json").read_text())
        return meta["radar_params"]

    def profile_cfg_path(self) -> Path:
        p = self.root / "profile.cfg"
        if p.is_file():
            return p
        meta = json.loads((self.root / "metadata.json").read_text())
        return Path(meta["cfg"])

    def iter_frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        raw = self.raw_dir if self.raw_dir.is_dir() else self.root
        paths = sorted(raw.glob("frame_*.npy"))
        for i, p in enumerate(paths, start=1):
            yield i, np.load(p)

    def frame_paths(self) -> list[Path]:
        raw = self.raw_dir if self.raw_dir.is_dir() else self.root
        return sorted(raw.glob("frame_*.npy"))


def _json_safe(params: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in params.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out
