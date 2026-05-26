"""Extract per-frame gesture features from a raw capture session."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from capture_store import CaptureSession
from processing.target_detect import LiveRadarTargetProcessor


@dataclass
class GestureTimeSeries:
    """Per-frame metrics for plotting."""

    time_s: np.ndarray
    frame_index: np.ndarray
    range_m: np.ndarray
    angle_deg: np.ndarray
    doppler_mps: np.ndarray
    energy: np.ndarray
    presence: np.ndarray
    snr_db: np.ndarray
    session_id: str
    n_frames_total: int

    def to_csv(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "frame_index",
                    "time_s",
                    "range_m",
                    "angle_deg",
                    "doppler_mps",
                    "energy",
                    "presence",
                    "snr_db",
                ]
            )
            for i in range(len(self.time_s)):
                w.writerow(
                    [
                        int(self.frame_index[i]),
                        f"{self.time_s[i]:.6f}",
                        f"{self.range_m[i]:.6f}",
                        f"{self.angle_deg[i]:.6f}",
                        f"{self.doppler_mps[i]:.6f}",
                        f"{self.energy[i]:.6f}",
                        f"{self.presence[i]:.0f}",
                        f"{self.snr_db[i]:.6f}",
                    ]
                )


def _frame_times(session: CaptureSession, n_valid: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame_index, time_s) aligned to processed frames."""
    index_path = session.root / "index.csv"
    frame_time_ms = float(session.radar_params().get("frame_time", 22.22))
    fps = 1000.0 / frame_time_ms

    if index_path.is_file():
        rows = list(csv.DictReader(index_path.open()))
        if len(rows) >= n_valid:
            t0 = float(rows[0]["timestamp_unix"])
            idx = np.array([int(r["frame_index"]) for r in rows[:n_valid]])
            t = np.array([float(r["timestamp_unix"]) - t0 for r in rows[:n_valid]])
            return idx, t

    idx = np.arange(1, n_valid + 1, dtype=np.int64)
    t = (idx - 1) / fps
    return idx, t


def analyze_capture(
    capture_path: Path,
    *,
    range_gate_m: tuple[float, float] = (0.5, 12.0),
    clutter_window: int = 8,
    smooth_alpha: float = 0.15,
    presence_threshold_db: float = 12.0,
    max_frames: int = 0,
    show_progress: bool = True,
) -> GestureTimeSeries:
    """
    Process every raw frame in a capture directory.

    Parameters
    ----------
    capture_path : folder with raw/*.npy and metadata.json
    """
    session = CaptureSession.open(Path(capture_path))
    params = session.radar_params()
    processor = LiveRadarTargetProcessor(
        params,
        range_gate_m=range_gate_m,
        clutter_window=clutter_window,
        smooth_alpha=smooth_alpha,
        presence_threshold_db=presence_threshold_db,
    )

    ranges, angles, dopplers, energies, presences, snrs = [], [], [], [], [], []
    frame_indices = []

    paths = session.frame_paths()
    if max_frames > 0:
        paths = paths[:max_frames]
    n_total = len(session.frame_paths())

    for path in paths:
        raw = np.load(path)
        est = processor.update(raw)
        if est is None:
            continue

        frame_indices.append(int(path.stem.split("_")[-1]))
        ranges.append(est.range_m)
        angles.append(est.angle_deg)
        dopplers.append(est.doppler_mps)
        energies.append(est.energy)
        snrs.append(est.snr_db)
        presences.append(est.presence)

        if show_progress and len(frame_indices) % 50 == 0:
            print(f"  processed {len(frame_indices)} frames…", flush=True)

    if not frame_indices:
        raise RuntimeError(f"No valid frames in {capture_path}")

    n = len(frame_indices)
    idx_arr, time_s = _frame_times(session, n)
    if len(idx_arr) != n:
        idx_arr = np.array(frame_indices, dtype=np.int64)
        frame_time_ms = float(params.get("frame_time", 22.22))
        time_s = (idx_arr - idx_arr[0]) * (frame_time_ms / 1000.0)

    session_id = session.root.name
    if (session.root / "session.json").is_file():
        session_id = json.loads((session.root / "session.json").read_text()).get(
            "session_id", session_id
        )

    return GestureTimeSeries(
        time_s=time_s,
        frame_index=idx_arr,
        range_m=np.array(ranges, dtype=np.float64),
        angle_deg=np.array(angles, dtype=np.float64),
        doppler_mps=np.array(dopplers, dtype=np.float64),
        energy=np.array(energies, dtype=np.float64),
        presence=np.array(presences, dtype=np.float64),
        snr_db=np.array(snrs, dtype=np.float64),
        session_id=session_id,
        n_frames_total=n_total,
    )


def load_timeseries(csv_path: Path) -> GestureTimeSeries:
    """Reload a previously saved gesture_timeseries.csv."""
    rows = list(csv.DictReader(Path(csv_path).open()))
    return GestureTimeSeries(
        time_s=np.array([float(r["time_s"]) for r in rows]),
        frame_index=np.array([int(r["frame_index"]) for r in rows]),
        range_m=np.array([float(r["range_m"]) for r in rows]),
        angle_deg=np.array([float(r["angle_deg"]) for r in rows]),
        doppler_mps=np.array([float(r["doppler_mps"]) for r in rows]),
        energy=np.array([float(r["energy"]) for r in rows]),
        presence=np.array([float(r["presence"]) for r in rows]),
        snr_db=np.array([float(r["snr_db"]) for r in rows]),
        session_id=Path(csv_path).parent.name,
        n_frames_total=len(rows),
    )
