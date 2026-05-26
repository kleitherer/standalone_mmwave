#!/usr/bin/env python3
"""
Live mmWave (DCA + UART) → processed range / Doppler / angle / SNR → OSC to Max.

Mirrors standalone_uwb/live_radar_to_max.py (OSC over UDP, no ROS).

Max patch:
  [udpreceive 9000]
  → [OSC-route /radar]
  → [route range_m doppler_mps angle_deg snr_db presence]

Usage (from standalone_mmwave/):
  python3 live_radar_to_max.py \\
    --cfg ../multimodal-ros/xwr_raw_ros/configs/6843isk/xwr68xx_3Tx_wfv0.5_RDAhigh_19.2m.cfg \\
    --cmd-tty /dev/cu.usbserial-00D832110

  python3 live_radar_to_max.py --help
"""

from __future__ import annotations

import argparse
import signal
import socket
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from processing.osc_utils import build_osc_message
from processing.target_detect import LiveRadarTargetProcessor
from radar_config import RadarConfig
from radar_receiver import RadarReceiver


class OscSender:
    def __init__(self, host: str, port: int) -> None:
        self._dest = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, address: str, value: float) -> None:
        self._sock.sendto(build_osc_message(address, value), self._dest)

    def send_bundle(self, address: str, *values: float) -> None:
        self._sock.sendto(build_osc_message(address, *values), self._dest)

    def close(self) -> None:
        self._sock.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live mmWave → Max OSC (no ROS)")
    p.add_argument(
        "--cfg",
        type=Path,
        default=_ROOT.parent
        / "multimodal-ros/xwr_raw_ros/configs/6843isk/xwr68xx_3Tx_wfv0.5_RDAhigh_19.2m.cfg",
    )
    p.add_argument("--cmd-tty", default="/dev/cu.usbserial-00D832110")
    p.add_argument("--dca-ip", default="192.168.33.180")
    p.add_argument("--host-data-port", type=int, default=4098)
    p.add_argument("--dca-only", action="store_true")
    p.add_argument("--osc-host", default="127.0.0.1")
    p.add_argument("--osc-port", type=int, default=9000)
    p.add_argument("--roi-min", type=float, default=0.5, help="Range gate min (m)")
    p.add_argument("--roi-max", type=float, default=12.0, help="Range gate max (m)")
    p.add_argument("--smooth-alpha", type=float, default=0.2, help="EMA on outputs (0=off)")
    p.add_argument("--clutter-window", type=int, default=8, help="RD clutter frames")
    p.add_argument("--packet-timeout", type=float, default=5.0)
    p.add_argument("--range-address", default="/radar/range_m")
    p.add_argument("--doppler-address", default="/radar/doppler_mps")
    p.add_argument("--angle-address", default="/radar/angle_deg")
    p.add_argument("--snr-address", default="/radar/snr_db")
    p.add_argument("--presence-address", default="/radar/presence")
    p.add_argument(
        "--bundle-address",
        default="",
        help="If set, also send one OSC message with (range, doppler, angle, snr)",
    )
    p.add_argument("--presence-threshold-db", type=float, default=12.0)
    p.add_argument("--no-doppler", action="store_true")
    p.add_argument("--no-angle", action="store_true")
    p.add_argument("--no-snr", action="store_true")
    p.add_argument("--status-interval", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    log = lambda msg: print(msg, flush=True)

    if not args.cfg.is_file():
        log(f"ERROR: cfg not found: {args.cfg}")
        return 1

    cfg_lines = args.cfg.read_text().splitlines(keepends=True)
    params = dict(RadarConfig(cfg_lines).get_params())

    log("=== Live mmWave → Max (no ROS) ===")
    log(f"  cfg:     {args.cfg}")
    log(f"  UART:    {args.cmd_tty if not args.dca_only else '(dca-only)'}")
    log(f"  DCA:     {args.dca_ip}  UDP data {args.host_data_port}")
    log(f"  OSC:     {args.osc_host}:{args.osc_port}")
    log(f"  ROI:     {args.roi_min:.2f}–{args.roi_max:.2f} m")
    log("  Max: [udpreceive %d] → [OSC-route /radar]" % args.osc_port)
    log("")

    processor = LiveRadarTargetProcessor(
        params,
        range_gate_m=(args.roi_min, args.roi_max),
        clutter_window=args.clutter_window,
        smooth_alpha=args.smooth_alpha,
    )
    osc = OscSender(args.osc_host, args.osc_port)

    receiver = RadarReceiver(
        cfg_lines,
        cmd_tty=args.cmd_tty,
        dca_ip=args.dca_ip,
        host_data_port=args.host_data_port,
        use_radar_cli=not args.dca_only,
    )

    stop = False

    def on_signal(_s, _f):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    sent = 0
    t0 = time.monotonic()
    last_status = t0
    packet_timeout = args.packet_timeout if args.packet_timeout > 0 else 0.0

    try:
        if receiver.radar_cli is not None:
            log("Configuring mmWave over UART...")
            receiver.radar_cli.configure(receiver.config)
        receiver.dca1000.configure()
        receiver.dca1000.start_capture()
        if receiver.radar_cli is not None:
            time.sleep(1)
            log("Starting sensor (sensorStart)...")
            receiver.radar_cli.start()
        log("Streaming… Ctrl+C to stop.")

        while not stop:
            try:
                frame, _ = receiver.read_frame(packet_timeout)
            except TimeoutError:
                continue

            est = processor.update(frame)
            if est is None:
                continue

            osc.send(args.range_address, est.range_m)
            if not args.no_doppler:
                osc.send(args.doppler_address, est.doppler_mps)
            if not args.no_angle:
                osc.send(args.angle_address, est.angle_deg)
            if not args.no_snr:
                osc.send(args.snr_address, est.snr_db)
            present = 1.0 if est.snr_db >= args.presence_threshold_db else 0.0
            osc.send(args.presence_address, present)

            if args.bundle_address:
                osc.send_bundle(
                    args.bundle_address,
                    est.range_m,
                    est.doppler_mps,
                    est.angle_deg,
                    est.snr_db,
                )

            sent += 1

            now = time.monotonic()
            if now - last_status >= args.status_interval:
                elapsed = now - t0
                rate = sent / max(elapsed, 1e-6)
                log(
                    f"  {elapsed:5.1f}s  osc={sent} ({rate:.1f}/s)  "
                    f"R={est.range_m:.2f}m  V={est.doppler_mps:+.2f}m/s  "
                    f"A={est.angle_deg:+.1f}°  SNR={est.snr_db:.1f}dB"
                )
                last_status = now

    except KeyboardInterrupt:
        log("\nStopped.")
    finally:
        receiver.close()
        osc.close()

    log(f"Sent {sent} OSC updates.")
    return 0 if sent > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
