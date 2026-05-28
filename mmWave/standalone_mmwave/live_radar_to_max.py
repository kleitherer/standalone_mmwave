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
import json
import signal
import socket
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from background_model import estimate_rd_background_from_capture
from processing.osc_utils import build_osc_message
from processing.target_detect import GESTURE_NONE, LiveRadarTargetProcessor
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

    def send_text(self, address: str, text: str) -> None:
        """OSC string argument for Max [route] / [text] labels."""
        self._sock.sendto(build_osc_message(address, text), self._dest)

    def close(self) -> None:
        self._sock.close()


class CaptureWriter:
    def __init__(self, root: Path, save_raw_frames: bool) -> None:
        self.root = root
        self.save_raw_frames = save_raw_frames
        self._raw_dir = self.root / "raw"
        self._n = 0
        self.root.mkdir(parents=True, exist_ok=True)
        if self.save_raw_frames:
            self._raw_dir.mkdir(parents=True, exist_ok=True)

    def write_metadata(self, metadata: dict) -> None:
        (self.root / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    def write_frame(self, frame) -> None:
        if not self.save_raw_frames:
            return
        import numpy as np

        self._n += 1
        np.save(self._raw_dir / f"frame_{self._n:06d}.npy", frame)


def _load_settings(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Settings file must contain a JSON object: {path}")
    return data


def _resolve_capture_path(p: Path) -> Path:
    """
    Resolve capture path from config/CLI.

    Accepts:
    - absolute paths
    - relative paths from CWD
    - bare capture names (resolved under ./captures/<name>)
    """
    if p.is_absolute():
        return p
    direct = p.resolve()
    if direct.exists():
        return direct
    under_captures = (_ROOT / "captures" / p).resolve()
    if under_captures.exists():
        return under_captures
    return direct


def _parse_args() -> argparse.Namespace:
    settings_default = _ROOT / "config/live_radar_to_max.json"
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--settings", type=Path, default=settings_default)
    bootstrap_args, _ = bootstrap.parse_known_args()
    settings = _load_settings(bootstrap_args.settings)

    proc_cfg = settings.get("processing", {})
    ang_cfg = settings.get("angle_estimation", {})
    push_pull_cfg = settings.get("push_pull", {})
    gesture_cfg = settings.get("gesture", {})
    net_cfg = settings.get("network", {})
    osc_cfg = settings.get("osc", {})
    device_cfg = settings.get("device", {})
    cap_cfg = settings.get("capture", {})
    run_cfg = settings.get("run", {})
    bg_cfg = settings.get("background", {})

    p = argparse.ArgumentParser(description="Live mmWave → Max OSC (no ROS)")
    p.add_argument(
        "--settings",
        type=Path,
        default=bootstrap_args.settings,
        help="JSON settings file for environment-specific defaults",
    )
    p.add_argument(
        "--cfg",
        type=Path,
        default=Path(
            device_cfg.get(
                "radar_cfg",
                _ROOT.parent
                / "multimodal-ros/xwr_raw_ros/configs/6843isk/xwr68xx_3Tx_wfv0.5_RDAhigh_19.2m.cfg",
            )
        ),
    )
    p.add_argument("--cmd-tty", default=device_cfg.get("cmd_tty", "/dev/cu.usbserial-00D832110"))
    p.add_argument("--dca-ip", default=net_cfg.get("dca_ip", "192.168.33.180"))
    p.add_argument("--host-data-port", type=int, default=int(net_cfg.get("host_data_port", 4098)))
    p.add_argument("--dca-only", action="store_true")
    p.add_argument("--osc-host", default=net_cfg.get("osc_host", "127.0.0.1"))
    p.add_argument("--osc-port", type=int, default=int(net_cfg.get("osc_port", 9000)))
    p.add_argument("--roi-min", type=float, default=float(proc_cfg.get("roi_min_m", 0.5)), help="Range gate min (m)")
    p.add_argument("--roi-max", type=float, default=float(proc_cfg.get("roi_max_m", 5.0)), help="Range gate max (m)")
    p.add_argument("--smooth-alpha", type=float, default=float(proc_cfg.get("smooth_alpha", 0.2)), help="EMA on outputs (0=off)")
    default_calibration_frames = int(proc_cfg.get("calibration_frames", proc_cfg.get("clutter_window", 45)))
    p.add_argument(
        "--calibration-frames",
        type=int,
        default=default_calibration_frames,
        help="Global-mean calibration frame count (stand clear during this warmup)",
    )
    p.add_argument(
        "--clutter-window",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--frame-average-count",
        type=int,
        default=int(proc_cfg.get("frame_average_count", 1)),
        help="Sliding average count over raw frames before estimation (1=off)",
    )
    p.add_argument("--packet-timeout", type=float, default=float(net_cfg.get("packet_timeout_sec", 5.0)))
    p.add_argument("--range-address", default=osc_cfg.get("range_address", "/radar/range_m"))
    p.add_argument("--doppler-address", default=osc_cfg.get("doppler_address", "/radar/doppler_mps"))
    p.add_argument("--angle-address", default=osc_cfg.get("angle_address", "/radar/angle_deg"))
    p.add_argument("--snr-address", default=osc_cfg.get("snr_address", "/radar/snr_db"))
    p.add_argument("--presence-address", default=osc_cfg.get("presence_address", "/radar/presence"))
    p.add_argument(
        "--gesture-address",
        default=osc_cfg.get("gesture_address", "/radar/gesture"),
        help="OSC string label: none | push | pull (or single when push/pull off)",
    )
    p.add_argument("--no-gesture", action="store_true", help="Do not send /radar/gesture")
    p.add_argument("--mode-address", default=osc_cfg.get("mode_address", "/radar/mode"))
    p.add_argument("--live-mode-text", default=osc_cfg.get("live_mode_text", "Recording Live Data"))
    p.add_argument(
        "--bundle-address",
        default="",
        help="If set, also send one OSC message with (range, doppler, angle, snr)",
    )
    p.add_argument("--presence-threshold-db", type=float, default=float(proc_cfg.get("presence_threshold_db", 12.0)))
    p.add_argument(
        "--push-pull",
        action=argparse.BooleanOptionalAction,
        default=bool(push_pull_cfg.get("enabled", False)),
        help="Prefer nearer-range peak among SNR candidates near global max (arms vs torso)",
    )
    p.add_argument(
        "--push-pull-snr-within-db",
        type=float,
        default=float(push_pull_cfg.get("snr_within_db_of_max", 3.0)),
        help="With --push-pull, only consider peaks within this many dB of max SNR",
    )
    p.add_argument(
        "--push-pull-range-derivative",
        action=argparse.BooleanOptionalAction,
        default=bool(push_pull_cfg.get("use_range_derivative_for_doppler", False)),
        help="TEMP: send d(range)/dt on /radar/doppler_mps when push/pull peak rule is active",
    )
    p.add_argument(
        "--gesture-min-peak-snr-db",
        type=float,
        default=float(gesture_cfg.get("min_peak_snr_db", 5.0)),
        help="Minimum global SNR to classify gesture",
    )
    p.add_argument(
        "--gesture-min-range-sep-m",
        type=float,
        default=float(gesture_cfg.get("min_range_sep_m", 0.2)),
        help="Push/pull: min separation between two range peaks (m)",
    )
    p.add_argument(
        "--gesture-max-range-sep-m",
        type=float,
        default=float(gesture_cfg.get("max_range_sep_m", 1.5)),
        help="Push/pull: max separation between two range peaks (m)",
    )
    p.add_argument(
        "--gesture-min-velocity-mps",
        type=float,
        default=float(gesture_cfg.get("min_velocity_mps", 0.15)),
        help="Push/pull: |d(range)/dt| must exceed this for push or pull label",
    )
    p.add_argument(
        "--emit-below-threshold",
        action="store_true",
        help="If set, keep publishing range/doppler/angle/snr even when SNR < threshold",
    )
    p.add_argument("--no-doppler", action="store_true")
    p.add_argument("--no-angle", action="store_true")
    p.add_argument("--no-snr", action="store_true")
    p.add_argument("--status-interval", type=float, default=1.0)
    p.add_argument(
        "--background-capture",
        type=Path,
        default=Path(bg_cfg["capture"]) if bg_cfg.get("capture") else None,
        help="Optional capture folder used as fixed background subtraction model",
    )
    p.add_argument(
        "--background-max-frames",
        type=int,
        default=int(bg_cfg.get("max_frames", 0)),
        help="Max frames from background capture to use (0 = all)",
    )
    p.add_argument(
        "--capture-enabled",
        action="store_true",
        default=bool(cap_cfg.get("enabled", False)),
        help="Save live run under captures/ with metadata and optional raw frames",
    )
    p.add_argument(
        "--capture-base-dir",
        type=Path,
        default=Path(cap_cfg.get("base_dir", "captures")),
        help="Base directory for saved captures",
    )
    p.add_argument(
        "--capture-name",
        default=str(cap_cfg.get("name", "live_radar")),
        help="Name suffix for capture folder",
    )
    p.add_argument(
        "--capture-save-raw",
        action="store_true",
        default=bool(cap_cfg.get("save_raw_frames", True)),
        help="Save raw frame_*.npy files (if capture is enabled)",
    )
    _dur = run_cfg.get("duration_sec")
    p.add_argument(
        "--duration-sec",
        type=float,
        default=float(_dur) if _dur is not None else 0.0,
        help="Stop after N seconds (0 or omit in JSON null = run until Ctrl+C)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    log = lambda msg: print(msg, flush=True)
    ang_cfg = _load_settings(args.settings).get("angle_estimation", {})
    if args.background_capture:
        args.background_capture = _resolve_capture_path(args.background_capture)

    if not args.cfg.is_file():
        log(f"ERROR: cfg not found: {args.cfg}")
        return 1

    cfg_lines = args.cfg.read_text().splitlines(keepends=True)
    params = dict(RadarConfig(cfg_lines).get_params())

    log("=== Live mmWave → Max (no ROS) ===")
    log(f"  settings:{args.settings}")
    log(f"  cfg:     {args.cfg}")
    log(f"  UART:    {args.cmd_tty if not args.dca_only else '(dca-only)'}")
    log(f"  DCA:     {args.dca_ip}  UDP data {args.host_data_port}")
    log(f"  OSC:     {args.osc_host}:{args.osc_port}")
    log(f"  ROI:     {args.roi_min:.2f}–{args.roi_max:.2f} m")
    log(f"  SNR min: {args.presence_threshold_db:.1f} dB")
    if args.push_pull:
        log(
            f"  Push/pull: ON — nearest range among peaks within "
            f"{args.push_pull_snr_within_db:.1f} dB of max SNR"
        )
        if args.push_pull_range_derivative:
            log("  Push/pull: d(range)/dt → /radar/doppler_mps (temporary)")
    log(f"  Avg:     {max(1, int(args.frame_average_count))} frame(s)")
    if args.duration_sec and args.duration_sec > 0:
        log(f"  Duration:{args.duration_sec:.1f}s (auto-stop)")
    else:
        log("  Duration: unlimited (Ctrl+C to stop)")
    if args.clutter_window is not None:
        args.calibration_frames = int(args.clutter_window)
    if args.background_capture:
        log(f"  BG src:  {args.background_capture} (fixed background model)")
    else:
        log(f"  Calib:   global-mean over first {args.calibration_frames} frames")
    log("  Max: [udpreceive %d] → [OSC-route /radar]" % args.osc_port)
    log("")

    background_rd_mean = None
    if args.background_capture:
        session_json = args.background_capture / "session.json"
        metadata_json = args.background_capture / "metadata.json"
        if not (session_json.is_file() or metadata_json.is_file()):
            log(
                "ERROR: background capture not found or invalid: "
                f"{args.background_capture}\n"
                "Expected a capture folder with session.json (or metadata.json for legacy).\n"
                "Tip: set background.capture to either an absolute path or a folder under captures/."
            )
            return 1
        background_rd_mean = estimate_rd_background_from_capture(
            args.background_capture,
            params,
            max_frames=max(0, int(args.background_max_frames)),
        )

    processor = LiveRadarTargetProcessor(
        params,
        range_gate_m=(args.roi_min, args.roi_max),
        clutter_window=args.calibration_frames,
        background_rd_mean=background_rd_mean,
        smooth_alpha=args.smooth_alpha,
        angle_fft_bins=int(ang_cfg.get("fft_bins", 128)),
        angle_fov_deg=float(ang_cfg.get("fov_deg", 90.0)),
        presence_threshold_db=args.presence_threshold_db,
        push_pull_mode=args.push_pull,
        push_pull_snr_within_db=args.push_pull_snr_within_db,
        push_pull_use_range_derivative_for_doppler=args.push_pull_range_derivative,
        gesture_min_peak_snr_db=args.gesture_min_peak_snr_db,
        gesture_min_range_sep_m=args.gesture_min_range_sep_m,
        gesture_max_range_sep_m=args.gesture_max_range_sep_m,
        gesture_min_velocity_mps=args.gesture_min_velocity_mps,
    )
    osc = OscSender(args.osc_host, args.osc_port)
    osc.send_text(args.mode_address, args.live_mode_text)
    log(f"  Mode → Max: {args.live_mode_text!r}  ({args.mode_address})")

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
    cap_writer = None
    frame_avg_count = max(1, int(args.frame_average_count))
    frame_hist: deque = deque(maxlen=frame_avg_count)
    frame_acc = None

    if args.capture_enabled:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cap_root = (args.capture_base_dir / f"{stamp}_{args.capture_name}").resolve()
        cap_writer = CaptureWriter(cap_root, save_raw_frames=args.capture_save_raw)
        cap_writer.write_metadata(
            {
                "created_at": stamp,
                "settings_file": str(args.settings.resolve()),
                "radar_cfg": str(args.cfg.resolve()),
                "cmd_tty": None if args.dca_only else args.cmd_tty,
                "dca_ip": args.dca_ip,
                "host_data_port": args.host_data_port,
                "osc_host": args.osc_host,
                "osc_port": args.osc_port,
                "processing": {
                    "roi_min_m": args.roi_min,
                    "roi_max_m": args.roi_max,
                    "presence_threshold_db": args.presence_threshold_db,
                    "calibration_frames": args.calibration_frames,
                    "frame_average_count": frame_avg_count,
                    "smooth_alpha": args.smooth_alpha,
                    "push_pull": {
                        "enabled": args.push_pull,
                        "snr_within_db_of_max": args.push_pull_snr_within_db,
                    },
                },
                "background": {
                    "capture": str(args.background_capture.resolve()) if args.background_capture else None,
                    "max_frames": args.background_max_frames,
                },
                "capture": {
                    "enabled": True,
                    "save_raw_frames": args.capture_save_raw,
                },
                "run": {
                    "duration_sec": args.duration_sec if args.duration_sec > 0 else None,
                },
                "radar_params": params,
            }
        )
        log(f"  Capture: {cap_root}")

    duration_limit = float(args.duration_sec) if args.duration_sec and args.duration_sec > 0 else 0.0

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
        if args.background_capture:
            log("Streaming… (using fixed background capture model) Ctrl+C to stop.")
        else:
            log("Streaming… (calibrating clutter first; stay out during warmup) Ctrl+C to stop.")
        stream_t0 = time.monotonic()

        while not stop:
            if duration_limit > 0 and (time.monotonic() - stream_t0) >= duration_limit:
                log(f"\nReached duration limit ({duration_limit:.1f}s).")
                break
            try:
                frame, _ = receiver.read_frame(packet_timeout)
            except TimeoutError:
                continue

            if frame_avg_count > 1:
                import numpy as np

                frame_f = frame.astype(np.float64, copy=False)
                if frame_acc is None:
                    frame_acc = np.zeros_like(frame_f, dtype=np.float64)
                if len(frame_hist) == frame_hist.maxlen:
                    frame_acc -= frame_hist[0]
                frame_hist.append(frame_f.copy())
                frame_acc += frame_hist[-1]
                frame = np.rint(frame_acc / len(frame_hist)).astype(frame.dtype, copy=False)

            est = processor.update(frame)
            if est is None:
                continue
            if cap_writer is not None:
                cap_writer.write_frame(frame)

            if (not args.emit_below_threshold) and est.snr_db < args.presence_threshold_db:
                osc.send(args.presence_address, 0.0)
                if not args.no_gesture and args.gesture_address:
                    osc.send_text(args.gesture_address, GESTURE_NONE)
                continue

            osc.send(args.range_address, est.range_m)
            if not args.no_doppler:
                osc.send(args.doppler_address, est.doppler_mps)
            if not args.no_angle:
                osc.send(args.angle_address, est.angle_deg)
            if not args.no_snr:
                osc.send(args.snr_address, est.snr_db)
            osc.send(args.presence_address, est.presence)
            if not args.no_gesture and args.gesture_address:
                osc.send_text(args.gesture_address, est.gesture)

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
                    f"A={est.angle_deg:+.1f}°  SNR={est.snr_db:.1f}dB  "
                    f"gesture={est.gesture}"
                )
                last_status = now

    except KeyboardInterrupt:
        log("\nStopped (Ctrl+C).")
    finally:
        receiver.close()
        osc.close()

    log(f"Sent {sent} OSC updates.")
    return 0 if sent > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
