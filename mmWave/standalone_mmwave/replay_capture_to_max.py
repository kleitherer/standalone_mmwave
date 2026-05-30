#!/usr/bin/env python3
"""
Replay a saved capture to Max over OSC.

Range / SNR / range2_m come from the same pipeline as ``range_time_snr.png``
(precomputed ``analysis/range_time_snr.npz`` when present, otherwise computed
on the fly with ``post_processing.range_time_limiter``). Doppler, angle, and
gesture still use ``LiveRadarTargetProcessor.update()``.

Usage (from standalone_mmwave/):
  python3 replay_capture_to_max.py --capture captures/push_pull
  python3 replay_capture_to_max.py --capture push_pull --speed 1.0
  python3 replay_capture_to_max.py --capture push_pull --loop
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from background_model import estimate_rd_background_from_capture
from capture_store import CaptureSession
from live_radar_to_max import OscSender, _load_settings, _resolve_capture_path
from processing.range_time_snr import RangeTimeSnrNpz, RangeTimeSnrProcessor
from processing.target_detect import GESTURE_NONE, LiveRadarTargetProcessor


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
    run_cfg = settings.get("run", {})
    bg_cfg = settings.get("background", {})

    p = argparse.ArgumentParser(description="Replay mmWave capture → Max OSC")
    p.add_argument("--settings", type=Path, default=bootstrap_args.settings)
    p.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="Capture folder (e.g. captures/push_pull or push_pull)",
    )
    p.add_argument("--osc-host", default=net_cfg.get("osc_host", "127.0.0.1"))
    p.add_argument("--osc-port", type=int, default=int(net_cfg.get("osc_port", 9000)))
    p.add_argument("--roi-min", type=float, default=float(proc_cfg.get("roi_min_m", 0.5)))
    p.add_argument("--roi-max", type=float, default=float(proc_cfg.get("roi_max_m", 5.0)))
    p.add_argument("--smooth-alpha", type=float, default=float(proc_cfg.get("smooth_alpha", 0.2)))
    p.add_argument(
        "--calibration-frames",
        type=int,
        default=int(proc_cfg.get("calibration_frames", 0)),
        help="Live-style inline warmup (0=off); prefer --declutter-mean-frames for replay",
    )
    p.add_argument(
        "--declutter-mean-frames",
        type=int,
        default=int(proc_cfg.get("declutter_mean_frames", 45)),
        help="Global-mean RD background from first N frames of this capture",
    )
    p.add_argument(
        "--frame-average-count",
        type=int,
        default=int(proc_cfg.get("frame_average_count", 1)),
    )
    p.add_argument("--presence-threshold-db", type=float, default=float(proc_cfg.get("presence_threshold_db", 12.0)))
    p.add_argument(
        "--push-pull",
        action=argparse.BooleanOptionalAction,
        default=bool(push_pull_cfg.get("enabled", False)),
        help="Prefer nearer-range peak among SNR candidates near global max",
    )
    p.add_argument(
        "--push-pull-snr-within-db",
        type=float,
        default=float(push_pull_cfg.get("snr_within_db_of_max", 3.0)),
    )
    p.add_argument(
        "--push-pull-range-derivative",
        action=argparse.BooleanOptionalAction,
        default=bool(push_pull_cfg.get("use_range_derivative_for_doppler", False)),
        help="TEMP: d(range)/dt on /radar/doppler_mps when push/pull peak rule is active",
    )
    p.add_argument(
        "--gesture-min-peak-snr-db",
        type=float,
        default=float(gesture_cfg.get("min_peak_snr_db", 5.0)),
    )
    p.add_argument(
        "--gesture-min-range-sep-m",
        type=float,
        default=float(gesture_cfg.get("min_range_sep_m", 0.2)),
    )
    p.add_argument(
        "--gesture-max-range-sep-m",
        type=float,
        default=float(gesture_cfg.get("max_range_sep_m", 1.5)),
    )
    p.add_argument(
        "--gesture-min-velocity-mps",
        type=float,
        default=float(gesture_cfg.get("min_velocity_mps", 0.15)),
    )
    p.add_argument(
        "--secondary-min-range-sep-m",
        type=float,
        default=float(osc_cfg.get("secondary_min_range_sep_m", 0.3)),
        help="range2_m: min separation from primary (m)",
    )
    p.add_argument(
        "--secondary-max-range-sep-m",
        type=float,
        default=float(osc_cfg.get("secondary_max_range_sep_m", 2.0)),
        help="range2_m: max separation from primary (m)",
    )
    p.add_argument(
        "--range-time-limiter",
        action=argparse.BooleanOptionalAction,
        default=bool(settings.get("post_processing", {}).get("range_time_limiter", True)),
        help="1-bit limiter on RD (must match post_processing.range_time_limiter / npz)",
    )
    p.add_argument(
        "--emit-below-threshold",
        action="store_true",
        help="Send range/doppler/angle/snr even when SNR < threshold",
    )
    p.add_argument("--no-doppler", action="store_true")
    p.add_argument("--no-angle", action="store_true")
    p.add_argument("--no-snr", action="store_true")
    p.add_argument("--bundle-address", default="")
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
    p.add_argument(
        "--replay-mode-text",
        default=osc_cfg.get("replay_mode_text", "Re-running Previous Capture"),
    )
    p.add_argument(
        "--background-capture",
        type=Path,
        default=Path(bg_cfg["capture"]) if bg_cfg.get("capture") else None,
    )
    p.add_argument("--background-max-frames", type=int, default=int(bg_cfg.get("max_frames", 0)))
    _dur = run_cfg.get("duration_sec")
    p.add_argument(
        "--duration-sec",
        type=float,
        default=float(_dur) if _dur is not None else 0.0,
        help="Stop after N seconds of replay (0 = full capture)",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed (1.0 = real-time frame rate from capture metadata)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Send as fast as possible (ignore frame timing)",
    )
    p.add_argument("--loop", action="store_true", help="Loop capture until Ctrl+C or duration")
    p.add_argument("--max-frames", type=int, default=0, help="Limit frames replayed (0 = all)")
    p.add_argument("--status-interval", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    log = lambda msg: print(msg, flush=True)
    ang_cfg = _load_settings(args.settings).get("angle_estimation", {})

    capture_path = _resolve_capture_path(args.capture)
    if not (capture_path / "metadata.json").is_file() and not (capture_path / "session.json").is_file():
        log(f"ERROR: not a capture folder: {capture_path}")
        return 1

    if args.background_capture:
        args.background_capture = _resolve_capture_path(args.background_capture)

    session = CaptureSession.open(capture_path)
    params = dict(session.radar_params())
    paths = session.frame_paths()
    if args.max_frames > 0:
        paths = paths[: args.max_frames]
    if not paths:
        log(f"ERROR: no frames in {capture_path}")
        return 1

    frame_time_ms = float(params.get("frame_time", 22.22))
    fps = 1000.0 / frame_time_ms
    frame_period_s = frame_time_ms / 1000.0
    if args.speed > 0 and not args.fast:
        frame_period_s /= args.speed

    log("=== Replay capture → Max (OSC) ===")
    log(f"  settings: {args.settings}")
    log(f"  capture:  {capture_path}")
    log(f"  frames:   {len(paths)} @ {fps:.2f} fps")
    log(f"  OSC:      {args.osc_host}:{args.osc_port}")
    log(f"  ROI:      {args.roi_min:.2f}–{args.roi_max:.2f} m")
    log(f"  SNR min:  {args.presence_threshold_db:.1f} dB")
    if args.push_pull:
        log(
            f"  Push/pull: ON — nearest range within "
            f"{args.push_pull_snr_within_db:.1f} dB of max SNR"
        )
        if args.push_pull_range_derivative:
            log("  Push/pull: d(range)/dt → /radar/doppler_mps (temporary)")
    log(f"  Avg:      {max(1, int(args.frame_average_count))} frame(s)")
    if args.fast:
        log("  Timing:   as fast as possible")
    else:
        log(f"  Timing:   {args.speed:.2f}x real-time ({1.0/frame_period_s:.1f} Hz)")
    if args.duration_sec and args.duration_sec > 0:
        log(f"  Duration: {args.duration_sec:.1f}s")
    elif args.loop:
        log("  Duration: loop until Ctrl+C")
    else:
        log(f"  Duration: full capture (~{len(paths) / fps:.1f}s)")
    if args.background_capture:
        log(f"  BG src:   {args.background_capture}")
    elif args.calibration_frames > 0:
        log(f"  Live calib: first {args.calibration_frames} replay frames (no OSC until done)")
    elif args.declutter_mean_frames > 0:
        log(f"  Declutter: global mean of first {args.declutter_mean_frames} frames of this capture")
    else:
        log("  Declutter: none (raw RD)")
    npz_path = capture_path / "analysis" / "range_time_snr.npz"
    range_time_npz = RangeTimeSnrNpz.load(npz_path)
    if range_time_npz is not None:
        log(f"  Range SNR: precomputed {npz_path.name} ({range_time_npz.snr_db.shape[0]} frames)")
    else:
        log(f"  Range SNR: computed on the fly (limiter={'on' if args.range_time_limiter else 'off'})")
    log(f"  Max: [udpreceive {args.osc_port}] → [OSC-route /radar]")
    log("")

    background_rd_mean = None
    if args.background_capture:
        bg_meta = args.background_capture / "metadata.json"
        bg_sess = args.background_capture / "session.json"
        if not (bg_meta.is_file() or bg_sess.is_file()):
            log(f"ERROR: invalid background capture: {args.background_capture}")
            return 1
        background_rd_mean = estimate_rd_background_from_capture(
            args.background_capture,
            params,
            max_frames=max(0, int(args.background_max_frames)),
        )
    elif args.calibration_frames <= 0 and args.declutter_mean_frames > 0:
        background_rd_mean = estimate_rd_background_from_capture(
            capture_path,
            params,
            max_frames=int(args.declutter_mean_frames),
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
    range_time_processor = None
    if range_time_npz is None:
        range_time_processor = RangeTimeSnrProcessor.from_capture(
            capture_path,
            params,
            range_gate_m=(args.roi_min, args.roi_max),
            declutter_mean_frames=args.declutter_mean_frames,
            limiter=args.range_time_limiter,
            background_capture=args.background_capture,
            background_max_frames=args.background_max_frames,
        )
    osc = OscSender(args.osc_host, args.osc_port)
    mode_text = f"{args.replay_mode_text} ({capture_path.name})"
    osc.send_text(args.mode_address, mode_text)
    log(f"  Mode → Max: {mode_text!r}  ({args.mode_address})")

    stop = False

    def on_signal(_s, _f):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    sent = 0
    t0 = time.monotonic()
    last_status = t0
    duration_limit = float(args.duration_sec) if args.duration_sec and args.duration_sec > 0 else 0.0
    frame_avg_count = max(1, int(args.frame_average_count))
    frame_hist: deque = deque(maxlen=frame_avg_count)
    frame_acc = None

    def _range_targets(frame_idx: int, frame: np.ndarray) -> list[tuple[float, float]]:
        kw = dict(
            min_secondary_snr_db=args.presence_threshold_db,
            secondary_min_sep_m=args.secondary_min_range_sep_m,
            secondary_max_sep_m=args.secondary_max_range_sep_m,
        )
        if range_time_npz is not None:
            return range_time_npz.targets_for_frame(frame_idx, **kw)
        assert range_time_processor is not None
        return range_time_processor.targets_from_frame(frame, **kw)

    def _send_frame(frame_idx: int, frame: np.ndarray, est) -> None:
        nonlocal sent
        targets = _range_targets(frame_idx, frame)
        if not targets:
            return
        t1 = targets[0]
        t2 = targets[1] if len(targets) > 1 else None
        present = t1[1] >= args.presence_threshold_db

        if (not args.emit_below_threshold) and not present:
            osc.send(args.presence_address, 0.0)
            if not args.no_gesture and args.gesture_address:
                osc.send_text(args.gesture_address, GESTURE_NONE)
            return

        r1, snr1 = t1
        osc.send(args.range_address, r1)
        if not args.no_snr:
            osc.send(args.snr_address, snr1)
        if t2 is not None:
            osc.send(args.range2_address, t2[0])
            if not args.no_snr:
                osc.send(args.snr2_address, t2[1])
        if est is not None:
            if not args.no_doppler:
                osc.send(args.doppler_address, est.doppler_mps)
            if not args.no_angle:
                osc.send(args.angle_address, est.angle_deg)
        osc.send(args.presence_address, 1.0 if present else 0.0)
        if est is not None and not args.no_gesture and args.gesture_address:
            osc.send_text(args.gesture_address, est.gesture)
        if args.bundle_address:
            r2, snr2 = t2 if t2 is not None else (0.0, 0.0)
            if est is not None:
                osc.send_bundle(
                    args.bundle_address,
                    r1,
                    est.doppler_mps,
                    est.angle_deg,
                    snr1,
                )
            else:
                osc.send_bundle(args.bundle_address, r1, snr1, r2[0], r2[1])
        sent += 1
        return r1, snr1, t2

    try:
        run_start = time.monotonic()
        pass_idx = 0
        while not stop:
            pass_idx += 1
            pass_start = time.monotonic()
            if pass_idx > 1:
                log(f"--- loop pass {pass_idx} ---")

            for i, path in enumerate(paths):
                if stop:
                    break
                if duration_limit > 0 and (time.monotonic() - run_start) >= duration_limit:
                    log(f"\nReached duration limit ({duration_limit:.1f}s).")
                    stop = True
                    break

                if not args.fast:
                    target_t = pass_start + i * frame_period_s
                    wait = target_t - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)

                frame = np.load(path)
                if frame_avg_count > 1:
                    frame_f = frame.astype(np.float64, copy=False)
                    if frame_acc is None:
                        frame_acc = np.zeros_like(frame_f, dtype=np.float64)
                    if len(frame_hist) == frame_hist.maxlen:
                        frame_acc -= frame_hist[0]
                    frame_hist.append(frame_f.copy())
                    frame_acc += frame_hist[-1]
                    frame = np.rint(frame_acc / len(frame_hist)).astype(frame.dtype, copy=False)

                est = processor.update(frame)
                result = _send_frame(i, frame, est)
                if result is None:
                    continue
                r1, snr1, t2 = result

                now = time.monotonic()
                if now - last_status >= args.status_interval:
                    elapsed = now - t0
                    rate = sent / max(elapsed, 1e-6)
                    t2_str = (
                        f"R2={t2[0]:.2f}m SNR={t2[1]:.1f}dB"
                        if t2 is not None
                        else "R2=—"
                    )
                    dop = est.doppler_mps if est is not None else 0.0
                    ang = est.angle_deg if est is not None else 0.0
                    gest = est.gesture if est is not None else GESTURE_NONE
                    log(
                        f"  {elapsed:5.1f}s  frame {i+1}/{len(paths)}  osc={sent} ({rate:.1f}/s)  "
                        f"R={r1:.2f}m SNR={snr1:.1f}dB  {t2_str}  "
                        f"V={dop:+.2f}m/s  A={ang:+.1f}°  gesture={gest}"
                    )
                    last_status = now

            if not args.loop:
                break
            if duration_limit > 0 and (time.monotonic() - run_start) >= duration_limit:
                break
            # Reset processor calibration only if we used inline calib (no fixed bg)
            if background_rd_mean is None:
                processor = LiveRadarTargetProcessor(
                    params,
                    range_gate_m=(args.roi_min, args.roi_max),
                    clutter_window=args.calibration_frames,
                    background_rd_mean=None,
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
                if range_time_npz is None:
                    range_time_processor = RangeTimeSnrProcessor.from_capture(
                        capture_path,
                        params,
                        range_gate_m=(args.roi_min, args.roi_max),
                        declutter_mean_frames=args.declutter_mean_frames,
                        limiter=args.range_time_limiter,
                        background_capture=args.background_capture,
                        background_max_frames=args.background_max_frames,
                    )
                frame_hist.clear()
                frame_acc = None

    except KeyboardInterrupt:
        log("\nStopped (Ctrl+C).")
    finally:
        osc.close()

    log(f"Sent {sent} OSC updates.")
    return 0 if sent > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
