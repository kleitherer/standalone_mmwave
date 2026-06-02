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
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from background_model import (
    estimate_rd_background_from_capture,
    estimate_rda_background_from_capture,
)
from gesture_recognition.osc import OscSender, make_publisher
from gesture_recognition.peaks import RangePeakDetectionConfig
from gesture_recognition.mode import (
    active_gesture_mode,
    describe_gesture_mode,
    make_gesture_processor,
    make_osc_gesture_limiter,
)
from gesture_recognition.tracker import TrackingConfig
from gesture_recognition.publisher import publish_radar_frame
from gesture_recognition.status import format_status_line
from processing.range_time_snr import RangeTimeSnrProcessor
from radar_config import RadarConfig
from radar_receiver import RadarReceiver


# Re-export for replay_capture_to_max.py
__all__ = ["OscSender", "_load_settings", "_resolve_capture_path"]


class CaptureWriter:
    """
    Persist per-frame ADC + wire from :meth:`RadarReceiver.read_frame`.

    - ``frame_*.npy`` — header-stripped ADC (live processing / post_processing)
    - ``wire_frame_*.npy`` — full UDP/ROS record (``ros_frame_size`` int16) for
      ``pack_capture_npz`` → ``radar_data`` without reconstructing wire
    """

    def __init__(self, root: Path, save_raw_frames: bool) -> None:
        self.root = root
        self.save_raw_frames = save_raw_frames
        self._raw_dir = self.root / "raw"
        self._n = 0
        self._logged_first_save = False
        self.root.mkdir(parents=True, exist_ok=True)
        if self.save_raw_frames:
            self._raw_dir.mkdir(parents=True, exist_ok=True)

    def write_metadata(self, metadata: dict) -> None:
        (self.root / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    def write_frame(self, frame, *, wire=None, log=None) -> None:
        if not self.save_raw_frames:
            return
        import numpy as np

        self._n += 1
        np.save(self._raw_dir / f"frame_{self._n:06d}.npy", frame)
        if wire is not None:
            np.save(self._raw_dir / f"wire_frame_{self._n:06d}.npy", wire)
        if log is not None and not self._logged_first_save:
            wire_note = f", wire_frame ({wire.size} int16)" if wire is not None else ""
            log(
                f"  Saving raw/: frame_*.npy ({np.asarray(frame).size} int16){wire_note} "
                f"— pack with post_processing.pack_capture_npz"
            )
            self._logged_first_save = True


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
    default_calibration_frames = int(proc_cfg.get("calibration_frames", proc_cfg.get("clutter_window", 0)))
    p.add_argument(
        "--calibration-frames",
        type=int,
        default=default_calibration_frames,
        help="Live-only: inline RD warmup frames (0=off; use declutter_mean_frames offline)",
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
    p.add_argument("--range2-address", default=osc_cfg.get("range2_address", "/radar/range2_m"),
                   help="Second target range (only if 0.3–2 m from primary)")
    p.add_argument("--snr2-address", default=osc_cfg.get("snr2_address", "/radar/snr2_db"),
                   help="Second target SNR (only sent with range2_m)")
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
        help="If set, keep publishing range/snr even when SNR < range_peak_detection.snr_threshold_db",
    )
    p.add_argument(
        "--range-time-limiter",
        action=argparse.BooleanOptionalAction,
        default=bool(settings.get("post_processing", {}).get("range_time_limiter", True)),
        help="1-bit limiter on RD (must match post_processing.range_time_limiter)",
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
        help="Save raw/frame_*.npy and raw/wire_frame_*.npy (if capture is enabled)",
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
    settings = _load_settings(args.settings)
    proc_cfg = settings.get("processing", {})
    ang_cfg = settings.get("angle_estimation", {})
    osc_cfg = settings.get("osc", {})
    send_track1_angle = (
        bool(osc_cfg.get("send_angle", False))
        and not bool(getattr(args, "no_angle", False))
    )
    peak_cfg = RangePeakDetectionConfig.from_settings(settings)
    gesture_mode = active_gesture_mode(settings)
    gesture_processor = make_gesture_processor(settings)
    osc_gesture_limiter = make_osc_gesture_limiter(settings)
    track_cfg = (
        TrackingConfig.from_settings(settings) if gesture_mode == "config1" else None
    )
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
    if track_cfg is not None:
        log(
            f"  Peak SNR threshold: body={peak_cfg.body_snr_threshold_db:.1f} dB, "
            f"hand={track_cfg.hand_snr_threshold_db:.1f} dB"
        )
    else:
        log(f"  Peak SNR threshold: body={peak_cfg.body_snr_threshold_db:.1f} dB")
    if send_track1_angle:
        log(
            f"  Angle: track id=1, FFT bins={int(ang_cfg.get('fft_bins', 128))}, "
            f"FoV={float(ang_cfg.get('fov_deg', 90.0)):.0f}°"
        )
    log(f"  Gesture: {gesture_mode} — {describe_gesture_mode(settings)}")
    if gesture_mode == "config1" and track_cfg is not None:
        if track_cfg.body_calibration_s > 0:
            log(
                f"  Body calib: stand still {track_cfg.body_calibration_s:.0f}s "
                f"(id=1 learns your range + SNR)"
            )
    log(f"  Avg:     {max(1, int(args.frame_average_count))} frame(s)")
    if args.duration_sec and args.duration_sec > 0:
        log(f"  Duration:{args.duration_sec:.1f}s (auto-stop)")
    else:
        log("  Duration: unlimited (Ctrl+C to stop)")
    if args.clutter_window is not None:
        args.calibration_frames = int(args.clutter_window)
    if args.background_capture:
        log(f"  BG src:  {args.background_capture} (fixed background model)")
    elif args.calibration_frames > 0:
        log(f"  Live BG: global-mean over first {args.calibration_frames} frames (no OSC until done)")
    else:
        log("  Live BG: none (raw RD for OSC; offline declutter via plot_heatmap / replay)")
    log("  Max: [udpreceive %d] → [OSC-route /radar]" % args.osc_port)
    log("")

    background_rd_mean = None
    background_rda_mean = None
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
            limiter=args.range_time_limiter,
        )
        background_rda_mean = estimate_rda_background_from_capture(
            args.background_capture,
            params,
            max_frames=max(0, int(args.background_max_frames)),
        )

    range_time_processor: RangeTimeSnrProcessor | None = None
    if background_rd_mean is not None:
        range_time_processor = RangeTimeSnrProcessor(
            params,
            range_gate_m=(args.roi_min, args.roi_max),
            background_rd_mean=background_rd_mean,
            background_rda_mean=background_rda_mean,
            limiter=args.range_time_limiter,
        )
    else:
        inline_calib = args.calibration_frames
        if inline_calib <= 0:
            inline_calib = int(proc_cfg.get("declutter_mean_frames", 0))
        range_time_processor = RangeTimeSnrProcessor(
            params,
            range_gate_m=(args.roi_min, args.roi_max),
            limiter=args.range_time_limiter,
            inline_calib_frames=inline_calib,
        )
        if inline_calib > 0:
            log(f"  Range SNR: inline calib {inline_calib} frames (limiter={'on' if args.range_time_limiter else 'off'})")
        else:
            log(f"  Range SNR: no declutter (limiter={'on' if args.range_time_limiter else 'off'})")

    frame_dt_s = max(float(params.get("frame_time", 22.22)) / 1000.0, 1e-6)

    publisher = make_publisher(args, settings)
    publisher.send_mode(args.live_mode_text)
    log(f"  Mode → Max: {args.live_mode_text!r}  ({publisher.addresses.mode})")

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
    frames_rx = 0
    udp_timeouts = 0
    t0 = time.monotonic()
    last_status = t0
    last_diag = t0
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
        if args.capture_save_raw:
            ros_b = int(params.get("ros_frame_size", params.get("wire_frame_size", 0) + 256))
            wire_b = int(params.get("wire_frame_size", params.get("frame_size", 0)))
            log(
                f"  Raw save: frame (ADC {params['adc_frame_size']//2} int16) + "
                f"wire_frame (ROS/UDP {ros_b//2} int16, wire-only {wire_b//2})"
            )

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
            log("Streaming… (fixed background capture) Ctrl+C to stop.")
        elif args.calibration_frames > 0:
            log("Streaming… (inline calib: stay clear until warmup done) Ctrl+C to stop.")
        else:
            log("Streaming… Ctrl+C to stop.")
        stream_t0 = time.monotonic()

        while not stop:
            if duration_limit > 0 and (time.monotonic() - stream_t0) >= duration_limit:
                log(f"\nReached duration limit ({duration_limit:.1f}s).")
                break
            try:
                adc, wire, _ = receiver.read_frame(packet_timeout)
            except TimeoutError:
                udp_timeouts += 1
                now_diag = time.monotonic()
                if now_diag - last_diag >= 2.0:
                    calib_n = range_time_processor.frames_seen
                    calib_total = range_time_processor._inline_calib_frames
                    log(
                        f"  [diag] no UDP packet in {packet_timeout:.0f}s — "
                        f"frames={frames_rx} timeouts={udp_timeouts} "
                        f"calib={'done' if range_time_processor.ready else f'{calib_n}/{calib_total}'}"
                    )
                    last_diag = now_diag
                continue

            frames_rx += 1

            # adc: header-stripped for processing; wire: full FrameBuffer for NPZ export.
            capture_frame = adc
            if frame_avg_count > 1:
                import numpy as np

                frame_f = capture_frame.astype(np.float64, copy=False)
                if frame_acc is None:
                    frame_acc = np.zeros_like(frame_f, dtype=np.float64)
                if len(frame_hist) == frame_hist.maxlen:
                    frame_acc -= frame_hist[0]
                frame_hist.append(frame_f.copy())
                frame_acc += frame_hist[-1]
                capture_frame = np.rint(frame_acc / len(frame_hist)).astype(
                    capture_frame.dtype, copy=False
                )

            if cap_writer is not None:
                cap_writer.write_frame(capture_frame, wire=wire, log=log)

            if gesture_mode == "config3":
                result = publish_radar_frame(
                    publisher,
                    peak_cfg,
                    None,
                    gesture_processor,
                    frame_dt_s,
                    osc_gesture_limiter,
                    frame_int16=capture_frame,
                    radar_params=params,
                )
            else:
                peaks = range_time_processor.peaks_from_frame(
                    capture_frame, peak_cfg, track_cfg
                )
                if peaks is None:
                    now_diag = time.monotonic()
                    if now_diag - last_diag >= 2.0:
                        calib_n = range_time_processor.frames_seen
                        calib_total = range_time_processor._inline_calib_frames
                        log(
                            f"  [diag] UDP ok — frames={frames_rx} "
                            f"calibrating {calib_n}/{calib_total} (no OSC yet)"
                        )
                        last_diag = now_diag
                    continue

                result = publish_radar_frame(
                    publisher,
                    peak_cfg,
                    peaks,
                    gesture_processor,
                    frame_dt_s,
                    osc_gesture_limiter,
                    frame_int16=capture_frame if send_track1_angle else None,
                    range_time_processor=range_time_processor if send_track1_angle else None,
                    angle_fft_bins=int(ang_cfg.get("fft_bins", 128)),
                    angle_fov_deg=float(ang_cfg.get("fov_deg", 90.0)),
                    radar_params=params,
                )
            if result is None:
                continue
            if result.published:
                sent += 1

            now = time.monotonic()
            if now - last_status >= args.status_interval:
                elapsed = now - t0
                rate = sent / max(elapsed, 1e-6)
                log(
                    format_status_line(
                        result,
                        elapsed_s=elapsed,
                        osc_sent=sent,
                        osc_rate_hz=rate,
                        gesture_mode=gesture_mode,
                    )
                )
                last_status = now

    except KeyboardInterrupt:
        log("\nStopped (Ctrl+C).")
    finally:
        receiver.close()
        publisher.sender.close()

    log(
        f"Sent {sent} OSC updates. "
        f"(UDP frames assembled: {frames_rx}, packet timeouts: {udp_timeouts})"
    )
    if frames_rx == 0:
        log(
            "ERROR: zero frames received — radar UART/DCA config ran, but no LVDS data on UDP :4098.\n"
            "  Check: ping 192.168.33.180, Mac IP 192.168.33.30, DCA powered, Ethernet (not Wi‑Fi),\n"
            "  close mmWave Studio if it holds the stream, power-cycle EVM+DCA, then:\n"
            "  python3 radar_receiver.py --cfg <same.cfg> --cmd-tty <your.port> --frames 10"
        )
    elif sent == 0 and frames_rx > 0:
        log(
            f"ERROR: got {frames_rx} frames but 0 OSC — "
            f"wait for range–time SNR calibration, or check processing."
        )
    return 0 if sent > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
