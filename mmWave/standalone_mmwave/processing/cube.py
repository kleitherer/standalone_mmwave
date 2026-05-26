"""Build complex radar cube from saved int16 frames (DCA UDP / FrameBuffer path)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from processing.lvds import fix_byte_order


def load_capture_metadata(capture_dir: Path) -> Dict[str, Any]:
    meta_path = Path(capture_dir) / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"No metadata.json in {capture_dir}")
    return json.loads(meta_path.read_text())


def load_frame(path: Path) -> np.ndarray:
    return np.load(path)


def _rx_phase_bias_complex(rx_phase_bias) -> np.ndarray:
    """Pairs of floats from compRangeBiasAndRxChanPhase → complex per virtual RX."""
    arr = np.asarray(rx_phase_bias, dtype=np.float64)
    if arr.size % 2 != 0:
        arr = arr[1:]  # skip range bias scalar if present
    return arr[0::2] + 1j * arr[1::2]


def frame_to_radar_cube(
    frame: np.ndarray,
    params: Dict[str, Any],
    *,
    flip_aop_phase: bool = False,
    platform: str | None = None,
) -> np.ndarray:
    """
    Convert one flat int16 frame to radar cube.

    Matches multimodal-ros ``sensors/dsp.py`` ``_reshape_frame`` for complex LVDS
    (``adc_output_fmt > 0``), which is what ``xwr68xx_3Tx_...cfg`` uses.

    Returns
    -------
    radar_cube : np.ndarray, complex64
        Shape ``(n_chirps // n_tx, n_rx * n_tx, n_samples)`` — slow-time × virtual
        antennas × range samples. For 3 TX / 4 RX / 96 chirps / 256 samples:
        ``(32, 12, 256)``.
    """
    data = np.asarray(frame, dtype=np.int16).ravel()
    n_chirps = int(params["n_chirps"])
    n_rx = int(params["n_rx"])
    n_tx = int(params["n_tx"])
    n_samples = int(params["n_samples"])
    adc_output_fmt = int(params.get("adc_output_fmt", 1))
    rx = params.get("rx", [1, 1, 1, 1])
    tx = params.get("tx", [1, 1, 1, 0])
    platform = platform or str(params.get("platform", "xWR68xx"))

    if adc_output_fmt <= 0:
        return data.reshape(n_chirps // n_tx, n_rx * n_tx, n_samples).astype(np.complex64)

    # LVDS uint16 reorder then IQ de-interleave (same as dsp.py)
    u16 = fix_byte_order(data)
    iq = np.zeros(u16.size // 2, dtype=np.complex64)
    iq[0::2] = 1j * u16[0::4] + u16[2::4]
    iq[1::2] = 1j * u16[1::4] + u16[3::4]
    cube = iq.reshape(n_chirps, n_rx, n_samples)

    if "xWR68xx" in platform and flip_aop_phase:
        for i_rx in (0, 2):
            if i_rx < n_rx:
                cube[:, i_rx, :] *= -1

    cube = cube.reshape(n_chirps // n_tx, n_rx * n_tx, n_samples)

    bias = _rx_phase_bias_complex(params.get("rx_phase_bias", []))
    if bias.size >= n_rx * n_tx:
        for c in range(n_rx * n_tx):
            cube[:, c, :] *= bias[c]

    return cube.astype(np.complex64, copy=False)


def iter_capture_frames(capture_dir: Path):
    """Yield (path, frame int16 array) sorted by frame index."""
    capture_dir = Path(capture_dir)
    raw = capture_dir / "raw"
    search = raw if raw.is_dir() else capture_dir
    paths = sorted(search.glob("frame_*.npy"))
    for p in paths:
        yield p, load_frame(p)
