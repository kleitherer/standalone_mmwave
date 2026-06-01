#!/usr/bin/env python3
"""
Record raw mmWave frames for offline algorithm development.

Stores reprocessable int16 ADC frames + profile + metadata (see capture_store.py).

Usage:
  cd standalone_mmwave
  python3 capture_radar.py --name hallway_walk --frames 300 \\
    --cmd-tty /dev/cu.usbserial-00D832110

  # Ctrl+C stops early; session is still finalized on disk
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from capture_store import CaptureSession
from radar_config import RadarConfig
from radar_receiver import RadarReceiver


def parse_args():
    p = argparse.ArgumentParser(description="Record raw mmWave captures for later processing")
    p.add_argument(
        "--cfg",
        type=Path,
        default=_ROOT.parent
        / "multimodal-ros/xwr_raw_ros/configs/6843isk/xwr68xx_3Tx_wfv0.5_RDAhigh_19.2m.cfg",
    )
    p.add_argument("--name", default="", help="Label appended to session folder name")
    p.add_argument("--notes", default="", help="Free text stored in session.json")
    p.add_argument("--out", type=Path, default=Path("captures"), help="Base captures directory")
    p.add_argument("--frames", type=int, default=0, help="Stop after N frames (0 = until Ctrl+C)")
    p.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Record for this many seconds (uses frame_time from .cfg to count frames)",
    )
    p.add_argument("--cmd-tty", default="/dev/cu.usbserial-00D832110")
    p.add_argument("--dca-ip", default="192.168.33.180")
    p.add_argument("--host-data-port", type=int, default=4098)
    p.add_argument("--dca-only", action="store_true")
    p.add_argument("--packet-timeout", type=float, default=5.0)
    p.add_argument("--verbose-uart", action="store_true")
    p.add_argument("--print-every", type=int, default=30)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.cfg.is_file():
        print(f"cfg not found: {args.cfg}", file=sys.stderr)
        return 1

    cfg_lines = args.cfg.read_text().splitlines(keepends=True)
    params = dict(RadarConfig(cfg_lines).get_params())

    frame_time_ms = float(params["frame_time"])
    fps = 1000.0 / frame_time_ms
    if args.duration_sec > 0:
        if args.frames > 0:
            raise SystemExit("Use only one of --frames or --duration-sec")
        args.frames = max(1, int(round(args.duration_sec * fps)))
        print(f"Duration {args.duration_sec}s @ {fps:.2f} fps → {args.frames} frames")

    notes = args.notes
    if args.duration_sec > 0 and not notes:
        notes = f"duration_sec={args.duration_sec}"

    session = CaptureSession.create(
        args.out,
        cfg_path=args.cfg,
        radar_params=params,
        label=args.name,
        cmd_tty=args.cmd_tty if not args.dca_only else None,
        dca_ip=args.dca_ip,
        host_data_port=args.host_data_port,
        notes=notes,
    )
    if args.duration_sec > 0:
        session._session["target_duration_sec"] = args.duration_sec
    print(f"Recording → {session.root.resolve()}")
    print(f"  raw int16/frame: {params['frame_size']//2} samples")
    print(f"  shape (chirps, rx, samples): ({params['n_chirps']}, {params['n_rx']}, {params['n_samples']})")

    receiver = RadarReceiver(
        cfg_lines,
        cmd_tty=args.cmd_tty,
        dca_ip=args.dca_ip,
        host_data_port=args.host_data_port,
        use_radar_cli=not args.dca_only,
        verbose_uart=args.verbose_uart,
    )

    stop = False

    def on_signal(_s, _f):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if receiver.radar_cli is not None:
        print("Configuring mmWave…")
        receiver.radar_cli.configure(receiver.config)
    receiver.dca1000.configure()
    receiver.dca1000.start_capture()
    if receiver.radar_cli is not None:
        time.sleep(1)
        receiver.radar_cli.start()

    print("Capturing raw frames. Ctrl+C to stop.\n")
    n = 0
    packet_timeout = args.packet_timeout if args.packet_timeout > 0 else 0.0

    try:
        while not stop:
            try:
                frame, wire, _ = receiver.read_frame(packet_timeout)
            except TimeoutError:
                print("  (waiting for UDP data…)")
                continue

            n += 1
            rec = session.write_frame(frame, wire=wire)
            if n == 1 or n <= 3 or (args.print_every and n % args.print_every == 0):
                print(f"  [{n}] {rec.path.name}  ({rec.n_samples} int16)")

            if args.frames and n >= args.frames:
                break
    finally:
        receiver.close()
        session.finalize()
        print(f"\nDone: {n} raw frames in {session.root.resolve()}")
        print("  profile.cfg, metadata.json, session.json, index.csv, raw/*.npy")

    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
