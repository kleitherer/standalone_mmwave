"""LVDS per-chirp headers (``lvdsStreamCfg`` enableHeader=1)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

# TI mmWave LVDS stream: 16 complex padding/header samples per chirp when enableHeader=1.
LVDS_COMPLEX_HEADER_PER_CHIRP = 16
DEFAULT_HEADER_POSITION = "back"  # skip-last-16 validated on walk/kmstest captures


def lvds_enable_header(params: Dict[str, Any]) -> bool:
    return bool(int(params.get("lvds_enable_header", 0)))


def adc_frame_byte_size(params: Dict[str, Any]) -> int:
    return int(params.get("adc_frame_size", params["frame_size"]))


def wire_frame_byte_size(params: Dict[str, Any]) -> int:
    return int(params.get("wire_frame_size", params["frame_size"]))


def ros_frame_byte_size(params: Dict[str, Any]) -> int:
    """ROS ``recver.cpp`` / rosbag NPZ record size (wire + 256 B frame trailer)."""
    if "ros_frame_size" in params:
        return int(params["ros_frame_size"])
    return wire_frame_byte_size(params) + int(params.get("ros_frame_trailer_bytes", 256))


def strip_lvds_chirp_headers(
    wire_int16: np.ndarray,
    params: Dict[str, Any],
    *,
    position: str = DEFAULT_HEADER_POSITION,
) -> np.ndarray:
    """
    Remove per-chirp LVDS header samples from a wire-aligned int16 frame.

    Wire layout per chirp (complex fmt): ``n_rx * n_samples`` ADC complex samples
    plus ``lvds_header_complex_per_chirp`` header complex samples (front or back).
    """
    if not lvds_enable_header(params):
        return np.asarray(wire_int16, dtype=np.int16).ravel()

    data = np.asarray(wire_int16, dtype=np.int16).ravel()
    n_chirps = int(params["n_chirps"])
    n_rx = int(params["n_rx"])
    n_samples = int(params["n_samples"])
    hdr = int(params.get("lvds_header_complex_per_chirp", LVDS_COMPLEX_HEADER_PER_CHIRP))
    adc_int16 = n_rx * n_samples * 2
    stride_int16 = (n_rx * n_samples + hdr) * 2

    expected = n_chirps * stride_int16
    if data.size < expected:
        raise ValueError(
            f"Wire frame too short: got {data.size} int16, expected {expected} "
            f"(wire_frame_size={wire_frame_byte_size(params)} bytes)"
        )

    parts: list[np.ndarray] = []
    for c in range(n_chirps):
        seg = data[c * stride_int16 : (c + 1) * stride_int16]
        if position == "back":
            parts.append(seg[:adc_int16])
        elif position == "front":
            parts.append(seg[hdr * 2 : hdr * 2 + adc_int16])
        else:
            raise ValueError(f"unsupported header position: {position!r}")
    return np.concatenate(parts)


def attach_lvds_chirp_headers(
    adc_int16: np.ndarray,
    params: Dict[str, Any],
    *,
    position: str = DEFAULT_HEADER_POSITION,
) -> np.ndarray:
    """Pad stripped ADC frames back to wire layout (zeros in header slots)."""
    if not lvds_enable_header(params):
        return np.asarray(adc_int16, dtype=np.int16).ravel()

    data = np.asarray(adc_int16, dtype=np.int16).ravel()
    n_chirps = int(params["n_chirps"])
    n_rx = int(params["n_rx"])
    n_samples = int(params["n_samples"])
    hdr = int(params.get("lvds_header_complex_per_chirp", LVDS_COMPLEX_HEADER_PER_CHIRP))
    adc_int16 = n_rx * n_samples * 2
    stride_int16 = (n_rx * n_samples + hdr) * 2
    pad = np.zeros(hdr * 2, dtype=np.int16)

    expected_adc = n_chirps * adc_int16
    if data.size < expected_adc:
        raise ValueError(f"ADC frame too short: {data.size} int16, expected {expected_adc}")

    out = np.empty(n_chirps * stride_int16, dtype=np.int16)
    for c in range(n_chirps):
        body = data[c * adc_int16 : (c + 1) * adc_int16]
        off = c * stride_int16
        if position == "back":
            out[off : off + adc_int16] = body
            out[off + adc_int16 : off + stride_int16] = pad
        elif position == "front":
            out[off : off + hdr * 2] = pad
            out[off + hdr * 2 : off + stride_int16] = body
        else:
            raise ValueError(f"unsupported header position: {position!r}")
    return out
