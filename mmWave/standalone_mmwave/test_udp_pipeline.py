#!/usr/bin/env python3
"""
Validate standalone mmWave UDP capture pipeline and mmw NPZ compatibility.

Runs four pipeline checks plus NPY↔NPZ equivalence and optional baseline comparison:

  1. UDP payload size validation (10-byte header strip, consistent payload sizes)
  2. Frame delimiter detection (HSI_HEADER_ID1/ID2 in mmw-packed stream)
  3. Frame size verification (wire / ADC / packed mmw sizes)
  4. LVDS de-interleaving validation (IQ magnitude, NaN/Inf, Re/Im balance)
  5. NPY concatenation vs packed .npz round-trip (mmw decode_data)
  6. Optional reference .npz baseline comparison

Usage
-----
  python3 test_udp_pipeline.py --capture testing_gesture_osc
  python3 test_udp_pipeline.py --capture testing_gesture_osc \\
      --reference-npz 20250528124759.npz --write-npz --max-frames 50
  python3 test_udp_pipeline.py --capture testing_gesture_osc --udp-dump capture.udp
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from capture_store import CaptureSession
from frame_buffer import FrameBuffer
from post_processing.pack_capture_npz import REF_FRAME_BYTES, REF_FRAME_INT16, pack_capture
from processing.adc_cube import frame_to_adc_cube
from processing.lvds import fix_byte_order
from processing.lvds_frame import (
    attach_lvds_chirp_headers,
    strip_lvds_chirp_headers,
    wire_frame_byte_size,
    ros_frame_byte_size,
    adc_frame_byte_size,
)

# mmw HSI delimiters (radar_capture_utils.decode_data)
HSI_HEADER_ID1 = b"\xdc\x0a\xda\x0c\xdc\x0a\xda\x0c"
HSI_HEADER_ID2 = b"\xc9\x0c\xcc\x09\xc9\x0c\xcc\x09"
CHIRP_HEADER_SIZE = 56
UDP_HEADER_SIZE = 10
# DCA1000 CONFIG_PACKET_DATA: total packet size 0x05c0 = 1472 bytes
DCA_PACKET_SIZE = 1472
DEFAULT_PAYLOAD_SIZE = DCA_PACKET_SIZE - UDP_HEADER_SIZE


@dataclass
class TestResult:
    name: str
    passed: bool
    measurements: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_capture(p: Path) -> Path:
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    for candidate in (p.resolve(), (_ROOT / "captures" / p).resolve()):
        if candidate.exists():
            return candidate
    return p.resolve()


def _load_mmw_decode_data():
    mmw_root = _ROOT.parent / "mmw-tracking-versions"
    path = mmw_root / "utils" / "radar_capture_utils.py"
    if not path.is_file():
        return None, f"not found: {path}"
    spec = importlib.util.spec_from_file_location("mmw_radar_capture_utils", path)
    if spec is None or spec.loader is None:
        return None, "import spec failed"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DCA1000.decode_data, None


def _parse_udp_packets(raw: bytes) -> list[tuple[int, int, bytes]]:
    """Parse concatenated UDP messages (10-byte header + payload)."""
    packets: list[tuple[int, int, bytes]] = []
    offset = 0
    while offset + UDP_HEADER_SIZE <= len(raw):
        seqn, bytec = struct.unpack("<IIxx", raw[offset : offset + UDP_HEADER_SIZE])
        # Infer payload length: next packet start or remainder of file.
        next_off = offset + DCA_PACKET_SIZE
        if next_off > len(raw):
            payload = raw[offset + UDP_HEADER_SIZE :]
        else:
            payload = raw[offset + UDP_HEADER_SIZE : next_off]
        packets.append((seqn, bytec, payload))
        offset += UDP_HEADER_SIZE + len(payload)
        if len(payload) == 0:
            break
    return packets


def _wire_to_udp_packets(wire_bytes: bytes, payload_size: int) -> list[tuple[int, int, bytes]]:
    """Simulate DCA1000 UDP packets from a wire-aligned byte stream."""
    packets: list[tuple[int, int, bytes]] = []
    seq = 1
    bytec = 0
    off = 0
    while off < len(wire_bytes):
        chunk = wire_bytes[off : off + payload_size]
        bytec += len(chunk)
        packets.append((seq, bytec, chunk))
        seq += 1
        off += len(chunk)
    return packets


def _reassemble_wire_from_packets(
    packets: list[tuple[int, int, bytes]], wire_frame_size: int
) -> list[np.ndarray]:
    """Reassemble wire frames via FrameBuffer (mirrors radar_receiver.read_frame)."""
    buf = FrameBuffer(2 * wire_frame_size, wire_frame_size)
    frames: list[np.ndarray] = []
    for seqn, _bytec, payload in packets:
        frame_view, new_frame = buf.add_msg(seqn, payload)
        if new_frame:
            frames.append(np.asarray(frame_view, dtype=np.int16).ravel().copy())
    return frames


def test_udp_payload_sizes(
    packets: list[tuple[int, int, bytes]],
    *,
    expected_payload: int = DEFAULT_PAYLOAD_SIZE,
    wire_frame_size: int | None = None,
) -> TestResult:
    """Test 1: strip 10-byte UDP header and verify payload sizes."""
    result = TestResult(name="udp_payload_size", passed=True)

    if not packets:
        result.passed = False
        result.issues.append("No UDP packets to analyze")
        return result

    sizes = [len(p[2]) for p in packets]
    result.measurements["n_packets"] = len(packets)
    result.measurements["payload_bytes_min"] = int(min(sizes))
    result.measurements["payload_bytes_max"] = int(max(sizes))
    result.measurements["payload_bytes_median"] = int(np.median(sizes))
    result.measurements["expected_payload_bytes"] = expected_payload

    # Last packet of a wire frame is often shorter; allow one trailing partial.
    full_size_count = sum(1 for s in sizes if s == expected_payload)
    short_packets = [(i, s) for i, s in enumerate(sizes) if s != expected_payload]
    result.measurements["n_full_size_payloads"] = full_size_count
    result.measurements["n_nonstandard_payloads"] = len(short_packets)
    if short_packets:
        result.measurements["nonstandard_packets"] = [
            {"index": i, "bytes": s, "byte_offset": int(i * (UDP_HEADER_SIZE + expected_payload))}
            for i, s in short_packets[:20]
        ]

    seq_gaps = []
    for i in range(1, len(packets)):
        prev_seq, curr_seq = packets[i - 1][0], packets[i][0]
        if curr_seq != prev_seq + 1:
            seq_gaps.append(
                {"index": i, "prev_seq": int(prev_seq), "curr_seq": int(curr_seq)}
            )
    result.measurements["sequence_gaps"] = seq_gaps[:20]
    result.measurements["n_sequence_gaps"] = len(seq_gaps)

    unique_sizes = sorted(set(sizes))
    result.measurements["unique_payload_sizes"] = unique_sizes

    # Fail only if non-full packets look like truncation (not just the final chunk).
    expected_tail = None
    if wire_frame_size is not None and expected_payload > 0:
        expected_tail = wire_frame_size % expected_payload
        if expected_tail == 0:
            expected_tail = expected_payload
    interior_short = []
    for i, s in short_packets:
        is_last = i == len(sizes) - 1
        if is_last and expected_tail is not None and s == expected_tail:
            continue
        if i < len(sizes) - 1:
            interior_short.append((i, s))
        elif expected_tail is None and s < expected_payload * 0.25:
            interior_short.append((i, s))
        elif expected_tail is not None and s != expected_tail:
            interior_short.append((i, s))
    if len(unique_sizes) > 2 or interior_short:
        result.passed = False
        result.issues.append(
            f"Inconsistent payload sizes: {unique_sizes} (expected ~{expected_payload})"
        )
    elif interior_short and interior_short[0][1] < expected_payload * 0.5:
        result.passed = False
        result.issues.append(
            f"Suspiciously small payload at index {interior_short[0][0]} "
            f"({interior_short[0][1]} B, offset ~{interior_short[0][0] * (UDP_HEADER_SIZE + expected_payload)})"
        )

    if seq_gaps:
        result.issues.append(
            f"{len(seq_gaps)} sequence gap(s) — possible packet loss at indices "
            f"{[g['index'] for g in seq_gaps[:5]]}"
        )

    return result


def _find_marker_offsets(data: bytes, marker: bytes) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        idx = data.find(marker, start)
        if idx < 0:
            break
        offsets.append(idx)
        start = idx + len(marker)
    return offsets


def _stream_after_fix_byte_order(stream_bytes: bytes) -> bytes:
    """Apply mmw ``fix_byte_order`` (decode_data step 1) before HSI marker search."""
    u16 = np.frombuffer(stream_bytes, dtype=np.uint16)
    return fix_byte_order(u16).tobytes()


def test_hsi_delimiters(
    stream_bytes: bytes,
    params: dict[str, Any],
    *,
    label: str = "mmw_packed",
) -> TestResult:
    """Test 2: search for HSI frame/chirp delimiters after fix_byte_order."""
    result = TestResult(name=f"hsi_delimiters_{label}", passed=True)
    n_chirps = int(params["n_chirps"])
    n_slow = int(params.get("n_slow", n_chirps // int(params["n_tx"])))

    fixed_bytes = _stream_after_fix_byte_order(stream_bytes)
    result.measurements["searched_after_fix_byte_order"] = True

    id2_offsets = _find_marker_offsets(fixed_bytes, HSI_HEADER_ID2)
    id1_offsets = _find_marker_offsets(fixed_bytes, HSI_HEADER_ID1)

    result.measurements["HSI2_count"] = len(id2_offsets)
    result.measurements["HSI1_count"] = len(id1_offsets)
    result.measurements["HSI2_offsets_sample"] = id2_offsets[:10]
    result.measurements["HSI1_offsets_sample"] = id1_offsets[:10]
    result.measurements["stream_bytes"] = len(stream_bytes)
    result.measurements["fixed_stream_bytes"] = len(fixed_bytes)

    if not id2_offsets and not id1_offsets:
        result.passed = False
        result.issues.append(
            "No HSI markers after fix_byte_order — stream is not mmw Ethernet format. "
            "Use --write-npz to build mmw-compatible radar_data from .npy frames."
        )
        return result

    # Split frames on HSI2 (mmw decode_data convention)
    frames_raw = fixed_bytes.split(HSI_HEADER_ID2)
    n_frames = max(0, len(frames_raw) - 1) if id2_offsets else 1
    result.measurements["n_frames_from_HSI2_split"] = n_frames

    id1_per_frame: list[int] = []
    id1_spacings: list[int] = []
    for frame_idx, frame in enumerate(frames_raw[1:] if id2_offsets else frames_raw):
        offs = _find_marker_offsets(frame, HSI_HEADER_ID1)
        id1_per_frame.append(len(offs))
        if len(offs) >= 2:
            sp = np.diff(offs).astype(int).tolist()
            id1_spacings.extend(sp)

    if id1_per_frame:
        result.measurements["HSI1_per_frame_min"] = int(min(id1_per_frame))
        result.measurements["HSI1_per_frame_max"] = int(max(id1_per_frame))
        result.measurements["HSI1_per_frame_median"] = float(np.median(id1_per_frame))
        result.measurements["expected_HSI1_per_frame"] = n_chirps

    if id1_spacings:
        result.measurements["HSI1_spacing_bytes_min"] = int(min(id1_spacings))
        result.measurements["HSI1_spacing_bytes_max"] = int(max(id1_spacings))
        result.measurements["HSI1_spacing_bytes_median"] = float(np.median(id1_spacings))
        spacing_std = float(np.std(id1_spacings))
        result.measurements["HSI1_spacing_bytes_std"] = spacing_std
        if spacing_std > 1.0:
            result.passed = False
            result.issues.append(
                f"Irregular HSI1 spacing (std={spacing_std:.1f} B) — possible corruption"
            )

    if id1_per_frame:
        bad = [i for i, c in enumerate(id1_per_frame) if c != n_chirps]
        result.measurements["frames_with_wrong_HSI1_count"] = len(bad)
        if bad:
            result.passed = False
            result.issues.append(
                f"{len(bad)} frame(s) with HSI1 count ≠ {n_chirps} "
                f"(e.g. frame indices {bad[:5]})"
            )

    # HSI2: expect one per frame boundary (between frames)
    if id2_offsets and n_frames > 0:
        expected_hsi2 = n_frames - 1
        if len(id2_offsets) not in (expected_hsi2, n_frames):
            result.issues.append(
                f"HSI2 count {len(id2_offsets)} — expected ~{expected_hsi2} "
                f"between {n_frames} frames"
            )

    result.measurements["note"] = (
        f"Config: n_chirps={n_chirps} (fast), n_slow={n_slow} (TDM slow-time)"
    )
    return result


def test_frame_sizes(
    frame_paths: list[Path],
    params: dict[str, Any],
    wire_frames: list[np.ndarray] | None = None,
) -> TestResult:
    """Test 3: verify ADC, wire, and per-chirp sizes."""
    result = TestResult(name="frame_size_verification", passed=True)

    adc_bytes = adc_frame_byte_size(params)
    wire_bytes = wire_frame_byte_size(params)
    n_chirps = int(params["n_chirps"])
    n_rx = int(params["n_rx"])
    n_samples = int(params["n_samples"])
    hdr = int(params.get("lvds_header_complex_per_chirp", 16))
    adc_int16 = n_rx * n_samples * 2
    stride_int16 = (n_rx * n_samples + hdr) * 2

    result.measurements["expected_adc_bytes"] = adc_bytes
    result.measurements["expected_wire_bytes"] = wire_bytes
    result.measurements["expected_adc_int16"] = adc_bytes // 2
    result.measurements["expected_wire_int16"] = wire_bytes // 2
    result.measurements["chirp_stride_int16"] = stride_int16
    result.measurements["lvds_header_complex_per_chirp"] = hdr

    # Per-chirp mmw HSI payload (56-byte header + IQ int16 pairs)
    mmw_chirp_payload = CHIRP_HEADER_SIZE + n_rx * n_samples * 4
    result.measurements["mmw_chirp_block_bytes"] = mmw_chirp_payload
    result.measurements["mmw_frame_adc_bytes"] = n_chirps * n_rx * n_samples * 4
    result.measurements["mmw_frame_with_headers_bytes"] = (
        CHIRP_HEADER_SIZE + n_chirps * mmw_chirp_payload
    )

    npy_sizes = []
    bad_npy = []
    for i, p in enumerate(frame_paths):
        arr = np.load(p, mmap_mode="r")
        nbytes = arr.size * arr.dtype.itemsize
        npy_sizes.append(nbytes)
        if nbytes != adc_bytes:
            bad_npy.append({"frame": p.name, "bytes": nbytes, "offset_index": i})

    result.measurements["n_npy_frames"] = len(frame_paths)
    result.measurements["npy_bytes_min"] = int(min(npy_sizes)) if npy_sizes else 0
    result.measurements["npy_bytes_max"] = int(max(npy_sizes)) if npy_sizes else 0

    if bad_npy:
        result.passed = False
        result.issues.append(
            f"{len(bad_npy)} .npy frame(s) wrong size (expected {adc_bytes} B): "
            f"{bad_npy[:3]}"
        )
        result.measurements["bad_npy_frames"] = bad_npy[:20]

    if wire_frames:
        wire_sizes = [f.nbytes for f in wire_frames]
        bad_wire = [
            {"index": i, "bytes": s}
            for i, s in enumerate(wire_sizes)
            if s != wire_bytes
        ]
        result.measurements["n_wire_frames"] = len(wire_frames)
        result.measurements["wire_bytes_min"] = int(min(wire_sizes)) if wire_sizes else 0
        result.measurements["wire_bytes_max"] = int(max(wire_sizes)) if wire_sizes else 0
        if bad_wire:
            result.passed = False
            result.issues.append(
                f"{len(bad_wire)} reassembled wire frame(s) wrong size "
                f"(expected {wire_bytes} B)"
            )
            result.measurements["bad_wire_frames"] = bad_wire[:20]

    return result


def test_lvds_deinterleave(
    frame_paths: list[Path],
    params: dict[str, Any],
    *,
    max_frames: int = 10,
) -> TestResult:
    """Test 4: IQ de-interleave quality on saved .npy frames."""
    result = TestResult(name="lvds_deinterleaving", passed=True)
    sample_paths = frame_paths[:max_frames]

    mags: list[float] = []
    re_stds: list[float] = []
    im_stds: list[float] = []
    nan_frames: list[str] = []
    zero_mag_frames: list[str] = []

    for p in sample_paths:
        frame = np.load(p)
        adc = frame_to_adc_cube(frame, params)
        mag = np.abs(adc)
        mags.append(float(np.median(mag)))
        re_stds.append(float(np.std(adc.real)))
        im_stds.append(float(np.std(adc.imag)))

        if not np.isfinite(adc).all():
            nan_frames.append(p.name)
        if float(np.max(mag)) < 1e-6:
            zero_mag_frames.append(p.name)

    result.measurements["frames_sampled"] = len(sample_paths)
    result.measurements["magnitude_median"] = float(np.median(mags)) if mags else 0.0
    result.measurements["magnitude_median_min"] = float(min(mags)) if mags else 0.0
    result.measurements["magnitude_median_max"] = float(max(mags)) if mags else 0.0
    result.measurements["real_std_median"] = float(np.median(re_stds)) if re_stds else 0.0
    result.measurements["imag_std_median"] = float(np.median(im_stds)) if im_stds else 0.0
    if re_stds and im_stds:
        ratio = float(np.median(re_stds) / (np.median(im_stds) + 1e-12))
        result.measurements["real_imag_std_ratio"] = ratio

    if nan_frames:
        result.passed = False
        result.issues.append(f"NaN/Inf in frames: {nan_frames}")
    if zero_mag_frames:
        result.passed = False
        result.issues.append(
            f"Zero magnitude (wrong I/Q order?) in: {zero_mag_frames}"
        )
    if re_stds and im_stds:
        ratio = result.measurements.get("real_imag_std_ratio", 1.0)
        if ratio < 0.01 or ratio > 100:
            result.passed = False
            result.issues.append(
                f"Real/Imag std ratio {ratio:.3g} — likely byte-order or I/Q swap issue"
            )

    # Cross-check: fix_byte_order should NOT be applied to standalone .npy
    if sample_paths:
        frame = np.load(sample_paths[0])
        raw_u16 = np.asarray(frame, dtype=np.int16).view(np.uint16)
        reordered = fix_byte_order(raw_u16)
        adc_wrong = frame_to_adc_cube(reordered.view(np.int16), params)
        adc_correct = frame_to_adc_cube(frame, params)
        diff = float(np.max(np.abs(adc_correct - adc_wrong)))
        result.measurements["fix_byte_order_max_adc_diff"] = diff
        result.measurements["fix_byte_order_should_not_match"] = diff > 1.0
        if diff < 1.0:
            result.issues.append(
                "fix_byte_order unexpectedly matches standalone decode — "
                "check whether data is mmw Ethernet format"
            )

    return result


def test_npy_npz_equivalence(
    frame_paths: list[Path],
    params: dict[str, Any],
    packed_npz: Path | None,
) -> TestResult:
    """Test 5: packed NPZ decodes via ``decode_data``; bytes/frame is wire or ROS size."""
    result = TestResult(name="npy_npz_equivalence", passed=True)

    if not frame_paths:
        result.passed = False
        result.issues.append("No .npy frames")
        return result

    n_tx = int(params["n_tx"])
    n_rx = int(params["n_rx"])
    wire_bpf = wire_frame_byte_size(params)
    ros_bpf = ros_frame_byte_size(params)
    result.measurements["n_frames"] = len(frame_paths)
    result.measurements["ros_bytes_per_frame"] = ros_bpf
    result.measurements["wire_bytes_per_frame"] = wire_bpf

    wire_paths = [
        p.parent / p.name.replace("frame_", "wire_frame_", 1) for p in frame_paths
    ]
    n_wire = sum(1 for w in wire_paths if w.is_file())
    result.measurements["n_wire_frames_on_disk"] = n_wire
    if n_wire < len(frame_paths):
        result.issues.append(
            f"Only {n_wire}/{len(frame_paths)} wire_frame_*.npy files — "
            "re-capture with updated radar_receiver for reliable decode_data round-trip"
        )

    if packed_npz is None or not packed_npz.is_file():
        result.issues.append("No packed NPZ — skipped decode round-trip")
        return result

    npz = np.load(packed_npz)
    result.measurements["packed_npz_keys"] = list(npz.keys())
    rd = npz["radar_data"]
    result.measurements["packed_radar_data_dtype"] = str(rd.dtype)
    result.measurements["packed_radar_data_bytes"] = int(rd.nbytes)
    bpf = rd.nbytes // len(frame_paths)
    result.measurements["bytes_per_frame"] = bpf
    if bpf not in (wire_bpf, ros_bpf):
        result.passed = False
        result.issues.append(
            f"bytes/frame {bpf} not wire ({wire_bpf}) or ROS ({ros_bpf})"
        )
    elif bpf == wire_bpf:
        result.measurements["note"] = (
            "wire-only NPZ (legacy capture); re-capture for rosbag-sized records"
        )

    decode_data, err = _load_mmw_decode_data()
    if decode_data is None:
        result.passed = False
        result.issues.append(f"mmw decode_data unavailable: {err}")
        return result

    try:
        decoded, _pc, info, num_frames = decode_data(
            rd.tobytes(), num_Tx=n_tx, num_Rx=n_rx
        )
        result.measurements["decode_data_shape"] = list(decoded.shape)
        result.measurements["decode_num_frames"] = dict(num_frames)
        result.measurements["decode_ok"] = True
        result.measurements["varying_chirp_frames"] = num_frames.get(
            "varying_chirp_frames", []
        )
        if decoded.shape[-1] != int(params["n_samples"]):
            result.passed = False
            result.issues.append(f"unexpected n_samples: {decoded.shape[-1]}")
        if num_frames.get("varying_chirp_frames"):
            result.passed = False
            result.issues.append(
                f"varying chirp sizes in frames: {num_frames['varying_chirp_frames'][:10]}"
            )
        # Expect most frames to decode (some may drop like reference capture).
        if decoded.shape[0] < max(1, len(frame_paths) - 5):
            result.passed = False
            result.issues.append(
                f"decode returned too few frames: {decoded.shape[0]} vs {len(frame_paths)}"
            )
    except Exception as exc:
        result.passed = False
        result.measurements["decode_ok"] = False
        result.issues.append(f"decode_data failed: {exc}")

    return result


def test_udp_roundtrip(
    frame_paths: list[Path],
    params: dict[str, Any],
    *,
    payload_size: int = DEFAULT_PAYLOAD_SIZE,
    max_frames: int = 5,
) -> TestResult:
    """Simulate UDP capture round-trip: .npy → wire → UDP → FrameBuffer → strip → compare."""
    result = TestResult(name="udp_roundtrip_simulation", passed=True)
    wire_size = wire_frame_byte_size(params)
    sample = frame_paths[:max_frames]

    max_diffs: list[float] = []
    for p in sample:
        original = np.load(p)
        wire = attach_lvds_chirp_headers(original, params)
        wire_bytes = wire.astype(np.int16).tobytes()
        packets = _wire_to_udp_packets(wire_bytes, payload_size)
        wire_frames = _reassemble_wire_from_packets(packets, wire_size)
        if not wire_frames:
            result.passed = False
            result.issues.append(f"No wire frame reassembled for {p.name}")
            continue
        recovered = strip_lvds_chirp_headers(wire_frames[-1], params)
        diff = float(np.max(np.abs(original.astype(np.int16) - recovered)))
        max_diffs.append(diff)

    result.measurements["frames_tested"] = len(sample)
    result.measurements["max_abs_diff"] = float(max(max_diffs)) if max_diffs else None
    result.measurements["payload_size_used"] = payload_size
    if max_diffs and max(max_diffs) > 0:
        result.passed = False
        result.issues.append(
            f"UDP round-trip mismatch (max diff={max(max_diffs)}) — "
            "check FrameBuffer or header strip logic"
        )
    return result


def test_reference_baseline(
    reference_npz: Path,
    params: dict[str, Any],
    packed_npz: Path | None,
) -> TestResult:
    """Compare against a reference .npz from another capture setup."""
    result = TestResult(name="reference_baseline", passed=True)
    ref = np.load(reference_npz)
    result.measurements["reference_path"] = str(reference_npz)
    result.measurements["reference_keys"] = list(ref.keys())

    rd = ref["radar_data"]
    result.measurements["reference_radar_data_dtype"] = str(rd.dtype)
    result.measurements["reference_radar_data_bytes"] = int(rd.nbytes)

    n_frames_time = int(ref["radar_time"].shape[0]) if "radar_time" in ref else None
    if n_frames_time:
        result.measurements["reference_n_frames_from_time"] = n_frames_time
        result.measurements["reference_bytes_per_frame"] = int(rd.nbytes // n_frames_time)

    ref_bytes = ref["radar_data"].tobytes()
    has_hsi = HSI_HEADER_ID1 in ref_bytes or HSI_HEADER_ID2 in ref_bytes
    result.measurements["reference_has_hsi_markers"] = has_hsi

    wire_expected = wire_frame_byte_size(params)
    adc_expected = adc_frame_byte_size(params)
    bpf = result.measurements.get("reference_bytes_per_frame")
    if bpf:
        result.measurements["expected_wire_bytes"] = wire_expected
        result.measurements["expected_adc_bytes"] = adc_expected
        if abs(bpf - wire_expected) > 512 and abs(bpf - adc_expected) > 512:
            result.issues.append(
                f"Reference frame size {bpf} B ≠ wire ({wire_expected}) or "
                f"ADC ({adc_expected}) — may use different header layout"
            )

    if not has_hsi:
        result.issues.append(
            "Reference raw bytes have no HSI markers (normal — markers appear after "
            "fix_byte_order inside decode_data, same as packed radar_data)."
        )

    if packed_npz and Path(packed_npz).is_file():
        packed = np.load(packed_npz)
        result.measurements["packed_keys"] = list(packed.keys())
        result.measurements["packed_radar_data_bytes"] = int(packed["radar_data"].nbytes)
        result.measurements["packed_radar_data_dtype"] = str(packed["radar_data"].dtype)
        fixed_packed = _stream_after_fix_byte_order(packed["radar_data"].tobytes())
        packed_has_hsi = (
            HSI_HEADER_ID1 in fixed_packed or HSI_HEADER_ID2 in fixed_packed
        )
        result.measurements["packed_has_hsi_markers"] = packed_has_hsi
        if not packed_has_hsi:
            result.passed = False
            result.issues.append("Packed NPZ missing HSI markers after fix_byte_order")

    return result


def _print_result(r: TestResult) -> None:
    status = "PASS" if r.passed else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"[{status}] {r.name}")
    print(f"{'=' * 60}")
    for k, v in r.measurements.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6g}")
        else:
            print(f"  {k}: {v}")
    if r.issues:
        print("  Issues:")
        for issue in r.issues:
            print(f"    • {issue}")


def run_all_tests(args: argparse.Namespace) -> dict[str, Any]:
    capture = _resolve_capture(args.capture)
    session = CaptureSession.open(capture)
    params = session.radar_params()
    frame_paths = session.frame_paths()
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]

    out_dir = args.out_dir or (capture / "analysis" / "udp_pipeline_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[TestResult] = []

    # --- Test 1: UDP payload sizes ---
    packets: list[tuple[int, int, bytes]] = []
    if args.udp_dump and Path(args.udp_dump).is_file():
        raw_udp = Path(args.udp_dump).read_bytes()
        packets = _parse_udp_packets(raw_udp)
        results.append(
            test_udp_payload_sizes(
                packets,
                expected_payload=args.payload_size,
                wire_frame_size=wire_frame_byte_size(params),
            )
        )
    elif frame_paths:
        wire = attach_lvds_chirp_headers(np.load(frame_paths[0]), params)
        packets = _wire_to_udp_packets(wire.tobytes(), args.payload_size)
        r = test_udp_payload_sizes(
            packets,
            expected_payload=args.payload_size,
            wire_frame_size=wire_frame_byte_size(params),
        )
        r.measurements["source"] = "simulated_from_npy_wire"
        results.append(r)

    # --- UDP round-trip ---
    if frame_paths:
        results.append(
            test_udp_roundtrip(
                frame_paths, params, payload_size=args.payload_size, max_frames=min(5, len(frame_paths))
            )
        )

    # --- Wire reassembly for frame size test ---
    wire_frames: list[np.ndarray] | None = None
    if packets and frame_paths:
        wire_size = wire_frame_byte_size(params)
        n_wire_frames = min(3, len(frame_paths))
        all_packets: list[tuple[int, int, bytes]] = []
        seq_base = 1
        for p in frame_paths[:n_wire_frames]:
            wire = attach_lvds_chirp_headers(np.load(p), params)
            pkts = _wire_to_udp_packets(wire.tobytes(), args.payload_size)
            for seq, bc, payload in pkts:
                all_packets.append((seq_base, bc, payload))
                seq_base += 1
        wire_frames = _reassemble_wire_from_packets(all_packets, wire_size)

    # --- Test 3: Frame sizes ---
    results.append(test_frame_sizes(frame_paths, params, wire_frames))

    # --- Test 4: LVDS de-interleave ---
    results.append(
        test_lvds_deinterleave(frame_paths, params, max_frames=min(20, len(frame_paths)))
    )

    # --- Pack NPZ if requested ---
    packed_npz = args.packed_npz
    if args.write_npz:
        packed_npz = out_dir / f"{capture.name}_packed.npz"
        pack_capture(
            capture,
            packed_npz,
            max_frames=args.max_frames,
            with_cube=args.with_cube,
            with_time=True,
        )
        print(f"\nWrote packed NPZ: {packed_npz}")

    # --- Test 2: LVDS / fix_byte_order on wire (reference NPZ path) ---
    if packed_npz and Path(packed_npz).is_file():
        stream = np.load(packed_npz)["radar_data"].tobytes()
        results.append(test_hsi_delimiters(stream, params, label="packed_npz"))
    elif frame_paths:
        wire = attach_lvds_chirp_headers(np.load(frame_paths[0]), params)
        results.append(test_hsi_delimiters(wire.tobytes(), params, label="wire_frame"))

    # --- Test 5: NPY vs NPZ equivalence ---
    results.append(test_npy_npz_equivalence(frame_paths, params, packed_npz))

    # --- Reference baseline ---
    if args.reference_npz:
        ref_path = Path(args.reference_npz)
        if not ref_path.is_file():
            ref_path = _ROOT / args.reference_npz
        if ref_path.is_file():
            results.append(test_reference_baseline(ref_path, params, packed_npz))
        else:
            r = TestResult("reference_baseline", False, issues=[f"Not found: {args.reference_npz}"])
            results.append(r)

    # --- Write report ---
    report = {
        "capture": str(capture),
        "n_frames_tested": len(frame_paths),
        "radar_params_summary": {
            "n_chirps": params["n_chirps"],
            "n_tx": params["n_tx"],
            "n_rx": params["n_rx"],
            "n_samples": params["n_samples"],
            "adc_frame_size": params.get("adc_frame_size", params["frame_size"]),
            "wire_frame_size": params.get("wire_frame_size", params["frame_size"]),
        },
        "tests": [r.to_dict() for r in results],
        "all_passed": all(r.passed for r in results),
    }

    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report: {report_path}")

    baseline_path = out_dir / "baseline.json"
    if args.write_baseline:
        baseline_path.write_text(json.dumps(report, indent=2))
        print(f"Wrote baseline: {baseline_path}")
    elif args.compare_baseline and baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())
        print(f"\nBaseline comparison ({baseline_path}):")
        print(f"  Baseline all_passed: {baseline.get('all_passed')}")
        print(f"  Current all_passed: {report['all_passed']}")

    for r in results:
        _print_result(r)

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {'PASS' if report['all_passed'] else 'FAIL'} ({len(results)} tests)")
    print(f"{'=' * 60}")

    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate standalone mmWave UDP pipeline and mmw NPZ compatibility"
    )
    p.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="Capture directory (captures/<name> or path)",
    )
    p.add_argument(
        "--udp-dump",
        type=Path,
        default=None,
        help="Raw concatenated UDP packets (.bin). If omitted, simulates from .npy wire frames.",
    )
    p.add_argument(
        "--reference-npz",
        type=Path,
        default=None,
        help="Reference .npz from ROS/mmw setup for baseline comparison",
    )
    p.add_argument(
        "--packed-npz",
        type=Path,
        default=None,
        help="Existing packed mmw-style NPZ (skip --write-npz)",
    )
    p.add_argument(
        "--write-npz",
        action="store_true",
        help="Pack capture to mmw-compatible radar_data NPZ",
    )
    p.add_argument(
        "--with-cube",
        action="store_true",
        help="Include radar_cube in packed NPZ",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Report/output directory")
    p.add_argument("--max-frames", type=int, default=None, help="Limit frames for speed")
    p.add_argument(
        "--payload-size",
        type=int,
        default=DEFAULT_PAYLOAD_SIZE,
        help=f"Expected UDP payload bytes after 10-byte header (default {DEFAULT_PAYLOAD_SIZE})",
    )
    p.add_argument(
        "--write-baseline",
        action="store_true",
        help="Save validation_report.json as baseline.json for future comparison",
    )
    p.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare against existing baseline.json in out-dir",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = run_all_tests(args)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
