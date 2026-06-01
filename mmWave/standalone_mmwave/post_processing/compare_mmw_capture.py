#!/usr/bin/env python3
"""
Compare standalone capture frames vs mmw-tracking ``decode_data`` / ``RD()``.

Standalone saves flat DCA FrameBuffer int16 (no per-chirp HSI headers).
mmw NPZ stores the full Ethernet stream; ``decode_data`` strips headers then
de-interleaves IQ differently from standalone LVDS.

Usage:
  python3 -m post_processing.compare_mmw_capture --capture newccrma --frame 495
  python3 -m post_processing.compare_mmw_capture --capture newccrma --frame 495 --write-npz
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from capture_store import CaptureSession
from processing.adc_cube import frame_to_adc_cube
from processing.cube import frame_to_radar_cube
from processing.lvds import fix_byte_order
from processing.mmw_rd import (
    frame_to_mmw_cube,
    frame_to_mmw_rd_power_db,
    mmw_rd,
    mmw_rd_power_db,
)
from processing.rda import compute_rda, rd_power_mmw_db


def _resolve_capture(p: Path) -> Path:
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    for candidate in (p.resolve(), (_ROOT / "captures" / p).resolve()):
        if candidate.exists():
            return candidate
    return p.resolve()


def _mmw_fix_byte_order(frame_int16: np.ndarray) -> np.ndarray:
    """Copy of mmw ``radar_capture_utils.fix_byte_order``."""
    buf = np.asarray(frame_int16, dtype=np.int16).ravel()
    u16 = buf.view(np.uint16)
    tmp = np.empty_like(u16)
    tmp[0::4] = u16[0::4]
    tmp[1:-1:4] = u16[2::4]
    tmp[2::4] = u16[1::4]
    tmp[3::4] = u16[3::4]
    return tmp


def adc_mmw_pair_deinterleave(frame_int16: np.ndarray, params: dict) -> np.ndarray:
    """
    IQ from mmw ``decode_data`` formula on flat payload (no headers).

    ``chirp_data[::2]*1j + chirp_data[1::2]`` after ``fix_byte_order``.
    """
    n_chirps = int(params["n_chirps"])
    n_rx = int(params["n_rx"])
    n_samples = int(params["n_samples"])
    u16 = _mmw_fix_byte_order(frame_int16)
    cd = u16.view(np.int16)
    iq = cd[0::2].astype(np.float32) * 1j + cd[1::2].astype(np.float32)
    return iq.astype(np.complex64).reshape(n_chirps, n_rx, n_samples)


def _load_mmw_decode_data():
    """Import mmw ``DCA1000.decode_data`` without pulling in ``pipeline_utils``."""
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


def _inverse_fix_byte_order(u16: np.ndarray) -> np.ndarray:
    """Invert mmw ``fix_byte_order`` (for building wire-format uint16 streams)."""
    out = np.empty_like(u16)
    out[0::4] = u16[0::4]
    out[1::4] = u16[2::4]
    out[2::4] = u16[1:-1:4]
    out[3::4] = u16[3::4]
    return out


# Minimal TLV block from a native mmw capture (zero detected objects, header_size=64).
_MMW_TLV_TEMPLATE = bytes.fromhex(
    "f00000000000000044074000040200020005000000000000000000000000040030000"
    "c0000000000000000000f0f0f0f0f0f0f0f0f0f0f0f"
) + bytes(248 - 56)


def _synthetic_mmw_stream(
    frame_int16: np.ndarray,
    params: dict,
    *,
    iq_mode: str = "standalone_lvds",
    prefix: bytes = b"",
) -> bytes:
    """
    Wrap one standalone frame in mmw HSI header structure for ``decode_data``.

    Payload per chirp: int16 Re,Im pairs so ``decode_data``'s ``[::2]*1j+[1::2]``
    recovers the complex chirp (matches standalone LVDS when iq_mode=standalone_lvds).

    ``prefix`` (typically ``HSI_HEADER_ID2``) must be supplied **before**
    ``_inverse_fix_byte_order`` so frame boundaries survive ``fix_byte_order``
    inside ``decode_data``.
    """
    header_size = 56
    hsi1 = b"\xdc\x0a\xda\x0c\xdc\x0a\xda\x0c"

    n_chirps = int(params["n_chirps"])

    if iq_mode == "standalone_lvds":
        adc = frame_to_adc_cube(frame_int16, params)
    else:
        adc = adc_mmw_pair_deinterleave(frame_int16, params)

    chirp_header = bytes(header_size)
    chirp_payloads = []
    for c in range(n_chirps):
        flat = adc[c].reshape(-1).astype(np.complex64)
        reim = np.empty(flat.size * 2, dtype=np.int16)
        reim[0::2] = flat.real.astype(np.int16)
        reim[1::2] = flat.imag.astype(np.int16)
        chirp_payloads.append(chirp_header + reim.tobytes())

    tlv_block = _MMW_TLV_TEMPLATE
    frame_body = prefix + tlv_block + hsi1 + hsi1.join(chirp_payloads)
    u16 = np.frombuffer(frame_body, dtype=np.uint16)
    wire = _inverse_fix_byte_order(u16)
    return wire.tobytes()


def compare_frame(
    frame_int16: np.ndarray,
    params: dict,
    *,
    run_mmw_decode: bool = True,
) -> dict:
    """Return comparison metrics for one frame."""
    n_tx = int(params["n_tx"])
    n_slow = int(params["n_chirps"]) // n_tx

    adc_standalone = frame_to_adc_cube(frame_int16, params)
    adc_mmw_pairs = adc_mmw_pair_deinterleave(frame_int16, params)

    cube_sa_virt = frame_to_radar_cube(frame_int16, params)
    cube_sa_mmw5d = frame_to_mmw_cube(frame_int16, params, apply_rx_phase_bias=False)
    cube_mmw5d_from_pairs = adc_mmw_pairs.reshape(n_slow, n_tx, params["n_rx"], -1)[None]

    rd_sa = rd_power_mmw_db(compute_rda(cube_sa_virt))
    rd_mmw_port = frame_to_mmw_rd_power_db(frame_int16, params)
    rd_pairs = mmw_rd_power_db(mmw_rd(cube_mmw5d_from_pairs))

    out = {
        "adc_standalone_vs_mmw_pair_iq": {
            "max_abs": float(np.max(np.abs(adc_standalone - adc_mmw_pairs))),
            "mean_abs": float(np.mean(np.abs(adc_standalone - adc_mmw_pairs))),
            "match": bool(np.allclose(adc_standalone, adc_mmw_pairs, rtol=0, atol=1e-3)),
        },
        "cube_standalone_virt_vs_mmw5d": {
            "max_abs": float(
                np.max(np.abs(cube_sa_virt - cube_sa_mmw5d.reshape(n_slow, n_tx * params["n_rx"], -1)))
            ),
            "match": bool(
                np.allclose(
                    cube_sa_virt,
                    cube_sa_mmw5d.reshape(n_slow, n_tx * params["n_rx"], -1),
                    rtol=0,
                    atol=1e-3,
                )
            ),
        },
        "rd_standalone_vs_mmw_port": {
            "max_abs_db": float(np.max(np.abs(rd_sa - rd_mmw_port))),
            "match": bool(np.allclose(rd_sa, rd_mmw_port, rtol=0, atol=1e-2)),
        },
        "rd_mmw_port_vs_mmw_pair_iq": {
            "max_abs_db": float(np.max(np.abs(rd_mmw_port - rd_pairs))),
        },
    }

    if run_mmw_decode:
        decode_data, err = _load_mmw_decode_data()
        if decode_data is None:
            out["mmw_decode_data"] = {"error": err}
        else:
            for mode in ("standalone_lvds", "mmw_pair"):
                try:
                    stream = _synthetic_mmw_stream(frame_int16, params, iq_mode=mode)
                    # decode_data expects uint16 wire buffer before fix_byte_order
                    dat, _, _, _ = decode_data(stream, num_Tx=n_tx, num_Rx=int(params["n_rx"]))
                    # dat: (n_frames, n_slow, n_tx, n_rx, n_samples)
                    decoded = dat[0]
                    if mode == "standalone_lvds":
                        ref = adc_standalone.reshape(n_slow, n_tx, params["n_rx"], -1)
                    else:
                        ref = adc_mmw_pairs.reshape(n_slow, n_tx, params["n_rx"], -1)
                    out[f"decode_data_{mode}"] = {
                        "shape": list(dat.shape),
                        "max_abs_vs_ref": float(np.max(np.abs(decoded - ref))),
                        "match": bool(np.allclose(decoded, ref, rtol=0, atol=1e-2)),
                    }
                    rd_dec = mmw_rd_power_db(mmw_rd(dat))
                    out[f"decode_data_{mode}_rd"] = {
                        "max_abs_db_vs_mmw_port": float(np.max(np.abs(rd_dec - rd_mmw_port))),
                    }
                except Exception as exc:
                    out[f"decode_data_{mode}"] = {"error": str(exc)}

    return out


def _print_report(results: dict, frame_idx: int) -> None:
    print(f"\n=== Frame {frame_idx} comparison ===")
    for key, val in results.items():
        print(f"\n{key}:")
        if isinstance(val, dict):
            for k, v in val.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {val}")

    print("\n--- Interpretation ---")
    iq = results["adc_standalone_vs_mmw_pair_iq"]
    if not iq["match"]:
        print(
            "• Standalone LVDS IQ ≠ mmw decode_data pair IQ (::2*1j+::2) on the same int16 buffer."
        )
        print("  Flat FrameBuffer captures use standalone deinterleave (multimodal-ros dsp.py).")
    rd = results["rd_standalone_vs_mmw_port"]
    if rd["match"]:
        print("• RD maps match when mmw RD() uses standalone ADC reshaped to (1,32,3,4,256).")
    dec = results.get("decode_data_standalone_lvds")
    if dec and dec.get("match"):
        print("• Synthetic NPZ round-trips through mmw decode_data with standalone IQ payload.")
    elif dec and "error" not in dec:
        print(
            f"• decode_data round-trip: max ADC diff {dec.get('max_abs_vs_ref', '?'):.2g} "
            f"(match={dec.get('match')})"
        )
    else:
        err = (dec or {}).get("error", "skipped")
        print(
            "• Full decode_data NPZ round-trip needs a native mmw .npz capture "
            f"(synthetic wrapper: {err})."
        )
        print("  IQ + RD comparisons above use the same standalone ADC data mmw RD() should use.")


def parse_args():
    p = argparse.ArgumentParser(description="Compare standalone capture vs mmw decode/RD")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--frame", type=int, default=495)
    p.add_argument("--write-npz", action="store_true", help="Write synthetic mmw-style .npz for frame")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-decode", action="store_true", help="Skip mmw decode_data import test")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    capture = _resolve_capture(args.capture)
    out_dir = args.out_dir or (capture / "analysis" / "mmw_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    session = CaptureSession.open(capture)
    params = dict(session.radar_params())
    paths = session.frame_paths()
    if args.frame < 0 or args.frame >= len(paths):
        raise SystemExit(f"--frame must be in [0, {len(paths) - 1}]")

    frame = np.load(paths[args.frame])
    results = compare_frame(frame, params, run_mmw_decode=not args.no_decode)
    _print_report(results, args.frame)

    if args.write_npz:
        stream = _synthetic_mmw_stream(frame, params, iq_mode="standalone_lvds")
        npz_path = out_dir / f"frame_{args.frame:06d}_synthetic_mmw.npz"
        np.savez(npz_path, radar_data=np.frombuffer(stream, dtype=np.uint8))
        print(f"\nWrote synthetic mmw NPZ: {npz_path}")
        print("  Load in mmw with: radarDataLoader(str(npz_path), radar_params).load_data()")

    report_path = out_dir / f"compare_frame{args.frame}.json"
    import json

    report_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
