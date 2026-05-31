#!/usr/bin/env python3
"""
Standalone mmWave capture: DCA1000 UDP + optional UART radar control.

Extracted from multimodal-ros sensors/radar_pub.py (no ROS).

Data path:
  DCA1000 --UDP--> dca1000.recv_data() --> frame_buffer.add_msg() --> int16 frame
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from dca1000 import DCA1000
from frame_buffer import FrameBuffer
from radar_config import RadarConfig


class RadarReceiver:
    """DCA UDP capture with optional UART config/start (same role as RadarPub)."""

    def __init__(
        self,
        cfg: List[str],
        cmd_tty: str = "/dev/ttyUSB0",
        dca_ip: str = "192.168.33.180",
        dca_cmd_port: int = 4096,
        host_cmd_port: int = 4096,
        host_data_port: int = 4098,
        use_radar_cli: bool = True,
        verbose_uart: bool = False,
    ):
        self.config = RadarConfig(cfg)
        self.params = self.config.get_params()
        self.use_radar_cli = use_radar_cli

        self.radar_cli = None
        if use_radar_cli:
            from radar_cli import RadarCLI

            self.radar_cli = RadarCLI(cmd_tty, verbose=verbose_uart)

        self.dca1000 = DCA1000(
            dca_ip=dca_ip,
            dca_cmd_port=dca_cmd_port,
            host_cmd_port=host_cmd_port,
            host_data_port=host_data_port,
        )

        if hasattr(self.dca1000, "data_socket"):
            self.dca1000.data_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, 131071 * 5
            )

        self.wire_frame_size = int(
            self.params.get("wire_frame_size", self.params["frame_size"])
        )
        self.frame_buffer = FrameBuffer(2 * self.wire_frame_size, self.wire_frame_size)

    def configure(self) -> None:
        if self.radar_cli is not None:
            self.radar_cli.configure(self.config)
        self.dca1000.configure()

    def start_capture(self) -> None:
        self.dca1000.start_capture()
        if self.radar_cli is not None:
            time.sleep(1)
            self.radar_cli.start()

    def stop_capture(self) -> None:
        if self.radar_cli is not None:
            self.radar_cli.stop()
        self.dca1000.stop_capture()

    def read_packet(self):
        """One UDP packet: (seqn, byte_count, payload)."""
        return self.dca1000.recv_data()

    def read_packet_timed(self, timeout_sec: float):
        """One UDP packet; raises TimeoutError if none arrives in time."""
        old = self.dca1000.data_socket.gettimeout()
        try:
            self.dca1000.data_socket.settimeout(timeout_sec)
            return self.read_packet()
        finally:
            self.dca1000.data_socket.settimeout(old)

    def read_frame(self, packet_timeout_sec: float = 0.0):
        """Block until a complete wire frame is assembled; return header-stripped ADC int16."""
        from processing.lvds_frame import strip_lvds_chirp_headers

        while True:
            if packet_timeout_sec > 0:
                seqn, _bytec, msg = self.read_packet_timed(packet_timeout_sec)
            else:
                seqn, _bytec, msg = self.read_packet()
            frame_data, new_frame = self.frame_buffer.add_msg(seqn, msg)
            if new_frame:
                wire = np.asarray(frame_data, dtype=np.int16).ravel()
                adc = strip_lvds_chirp_headers(wire, self.params)
                return adc, True

    def close(self) -> None:
        self.stop_capture()
        if self.radar_cli is not None:
            self.radar_cli.close()
        self.dca1000.close()


def parse_args():
    p = argparse.ArgumentParser(description="Standalone TI mmWave UDP capture")
    p.add_argument(
        "--cfg",
        type=Path,
        required=True,
        help="Path to mmWave .cfg file (sets frame size via RadarConfig)",
    )
    p.add_argument(
        "--cmd-tty",
        default=None,
        help="mmWave command UART (macOS: /dev/cu.usbserial-... from --list-ports)",
    )
    p.add_argument(
        "--list-ports",
        action="store_true",
        help="Print /dev/cu.* serial ports and exit",
    )
    p.add_argument("--dca-ip", default="192.168.33.180")
    p.add_argument("--dca-cmd-port", type=int, default=4096)
    p.add_argument("--host-cmd-port", type=int, default=4096)
    p.add_argument("--host-data-port", type=int, default=4098)
    p.add_argument(
        "--dca-only",
        action="store_true",
        help="Only talk to DCA1000 over UDP (radar already started elsewhere)",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop after N complete frames (0 = run until Ctrl+C)",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save each frame as .npy under this directory",
    )
    p.add_argument(
        "--capture",
        action="store_true",
        help="Save frames under captures/YYYYMMDD_HHMMSS/ (same as --save with timestamp)",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Log frame count every N frames",
    )
    p.add_argument(
        "--packet-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait per UDP packet before retrying (0 = block forever)",
    )
    p.add_argument(
        "--verbose-uart",
        action="store_true",
        help="Print each UART command and warn on empty responses",
    )
    return p.parse_args()


def _load_cfg(path: Path) -> List[str]:
    if not str(path) or path == Path("."):
        raise SystemExit(
            "No .cfg file given. Set CFG to a file path, e.g.\n"
            "  CFG=../multimodal-ros/xwr_raw_ros/configs/6843isk/"
            "xwr68xx_3Tx_wfv0.5_RDAhigh_19.2m.cfg\n"
            "  python3 radar_receiver.py --cfg \"$CFG\" --dca-only --frames 10"
        )
    if not path.is_file():
        raise SystemExit(f"Not a file: {path.resolve()}")
    return path.read_text().splitlines(keepends=True)


def main() -> int:
    args = parse_args()
    if args.list_ports:
        from radar_cli import list_serial_ports

        ports = list_serial_ports()
        if ports:
            print("Serial ports (use --cmd-tty with a cu.* device):")
            for p in ports:
                print(f"  {p}")
        else:
            print("No /dev/cu.* ports found. Connect mmWave USB and try again.")
        return 0

    cfg_lines = _load_cfg(args.cfg)
    if args.capture:
        if args.save is not None:
            raise SystemExit("Use only one of --capture or --save")
        args.save = Path("captures") / datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Capture directory: {args.save.resolve()}")

    cmd_tty = args.cmd_tty
    if not args.dca_only and cmd_tty is None:
        from radar_cli import list_serial_ports

        ports = list_serial_ports()
        raise SystemExit(
            "Without --dca-only you must set --cmd-tty to the mmWave USB port.\n"
            "  python3 radar_receiver.py --list-ports\n"
            f"  Example: --cmd-tty {ports[0] if ports else '/dev/cu.usbserial-1410'}"
        )
    if cmd_tty is None:
        cmd_tty = "/dev/ttyUSB0"  # unused when --dca-only

    receiver = RadarReceiver(
        cfg_lines,
        cmd_tty=cmd_tty,
        dca_ip=args.dca_ip,
        dca_cmd_port=args.dca_cmd_port,
        host_cmd_port=args.host_cmd_port,
        host_data_port=args.host_data_port,
        use_radar_cli=not args.dca_only,
        verbose_uart=args.verbose_uart,
    )

    stop = False

    def on_signal(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    params = receiver.params
    print(
        f"adc_frame={params.get('adc_frame_size', params['frame_size'])} B, "
        f"wire_frame={params.get('wire_frame_size', params['frame_size'])} B, "
        f"shape=({params['n_chirps']}, {params['n_rx']}, {params['n_samples']}) "
        f"complex={'yes' if params['adc_output_fmt'] > 0 else 'no'}"
    )
    print(f"UDP data port {args.host_data_port}, DCA at {args.dca_ip}")

    # UART first so mmWave is configured before DCA starts streaming
    if receiver.radar_cli is not None:
        print("Configuring mmWave over UART...")
        receiver.radar_cli.configure(receiver.config)
    receiver.dca1000.configure()
    receiver.dca1000.start_capture()
    if receiver.radar_cli is not None:
        time.sleep(1)
        print("Starting sensor (sensorStart)...")
        receiver.radar_cli.start()

    print(
        f"\nDCA RECORD_START OK — listening for UDP data on port {args.host_data_port}..."
    )
    if args.dca_only:
        print(
            "With --dca-only the mmWave must already be streaming (mmWave Studio "
            "Sensor Start, or re-run without --dca-only and --cmd-tty /dev/cu.usb...).\n"
            "Also check: ribbon mmWave↔DCA, DCA powered, radar chirping.\n"
        )
    else:
        print("UART will start the sensor; first frames may take a few seconds.\n")

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        meta = {
            "cfg": str(args.cfg.resolve()),
            "cmd_tty": cmd_tty if not args.dca_only else None,
            "dca_ip": args.dca_ip,
            "host_data_port": args.host_data_port,
            "radar_params": {k: (v if not hasattr(v, "tolist") else v) for k, v in receiver.params.items()},
        }
        (args.save / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    frame_count = 0
    packet_timeout = args.packet_timeout if args.packet_timeout > 0 else 0.0
    try:
        while not stop:
            try:
                frame_data, _ = receiver.read_frame(packet_timeout)
            except TimeoutError:
                if stop:
                    break
                print(
                    "No UDP data yet (timed out waiting on port "
                    f"{args.host_data_port}). Still listening..."
                )
                continue
            frame_count += 1
            if frame_count == 1:
                print(f"First frame received ({frame_data.size} int16 samples).")

            if args.save:
                out = args.save / f"frame_{frame_count:06d}.npy"
                np.save(out, frame_data)
                if frame_count <= 3 or frame_count == args.frames:
                    print(f"  saved {out.name}")

            if args.print_every and frame_count % args.print_every == 0:
                print(f"frame {frame_count}, samples={frame_data.size}")

            if args.frames and frame_count >= args.frames:
                break
    finally:
        receiver.close()
        if args.save and frame_count:
            print(f"stopped after {frame_count} frames → {args.save.resolve()}")
        else:
            print(f"stopped after {frame_count} frames")

    return 0


if __name__ == "__main__":
    sys.exit(main())
