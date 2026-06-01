# Gesture recognition — peak list, tracking, OSC, live/replay streaming

## What lives here

| Module | Role |
|--------|------|
| `peaks.py` | Range peak list from range–time SNR profile |
| `tracker.py` | Nearest-neighbor tracking (1–2 IDs), velocity history |
| `gesture.py` | push / pull / none from `d(range)/dt` |
| `osc.py` | OSC addresses + `OscPublisher` (shared by live & replay) |
| `mode.py` | Select config1 vs config2; build processor |
| `simple.py` | Config2: strongest peak + 2-point velocity |
| `publisher.py` | One function: peaks + processor → OSC |
| `status.py` | One terminal status formatter |

Live (`live_radar_to_max.py`) and replay (`replay_capture_to_max.py`) both call
`publish_radar_frame()` and `format_status_line()`.

## Gesture modes

Set `gesture.mode` in `config/live_radar_to_max.json`:

| Mode | Description |
|------|-------------|
| **`config1`** | 2-ID NN tracker (body + hands). Toggle with `config1.enabled`. |
| **`config2`** | Single strongest peak; velocity from **last two frames** only. |

Switch to config2:

```json
"gesture": { "mode": "config2" }
```

Or disable config1 (falls back to config2):

```json
"gesture": { "mode": "config1", "config1": { "enabled": false, ... } }
```

## Config1 — tracking (body vs hands)

Each frame:

1. `detect_range_peaks_frame` → SNR-sorted peak list
2. `RangePeakTracker.update` associates peaks to existing tracks
3. **Track id=1 (body)** learns your range and SNR from **5 s standing still**
   at startup (or bootstraps on the first peak if calibration is disabled)
4. **Track id=2 (hands)** spawns in the hand band in front of id=1

### Config1 keys (`gesture.config1`)

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | true | When false, use config2 even if `mode` is config1 |
| `associate_gate_m` | 0.20 | NN gate from predicted range (m) |
| `max_range_jump_m` | 0.50 | Max per-frame step on same ID (m) |
| `min_hand_ahead_m` | 0.20 | id=2 at least this much closer than id=1 (m) |
| `max_hand_ahead_m` | 0.70 | id=2 at most this much closer than id=1 (m) |
| `history_frames` | 8 | Velocity window for id=2 |
| `max_missed_frames` | 5 | Drop **id=2** after N misses |
| `min_velocity_mps` | 0.15 | Push/pull speed threshold |
| `gesture_confirm_frames` | 3 | Frame debounce before raw label counts |
| `body_slower_than_hand` | true | id=1 may not move faster than id=2 |
| `body_hand_velocity_margin_mps` | 0.05 | Tolerance on the speed rule (m/s) |
| `body_calibration_s` | 5.0 | Stand still this long to learn id=1 (0 = off) |
| `body_calibration_max_velocity_mps` | 0.10 | Max motion during calib before timer resets |
| `body_anchor_blend` | 0.5 | Pull id=1 association toward calibrated range |
| `body_snr_margin_db` | 6.0 | Reject id=1 peaks this far below calibrated SNR |
| `body_max_velocity_for_gesture_mps` | 0.10 | Block OSC push/pull when id=1 moves faster (0 = off) |

### Body calibration (config1)

At startup, **stand in front of the radar without moving** for
`body_calibration_s` seconds (default 5). The tracker records the dominant
peak each frame and learns:

- **Range** — median distance to your torso
- **SNR** — median strength of that reflection

That profile becomes **track id=1**. If you move during calibration, the timer
resets. After calibration, id=1 prefers peaks near your learned range and
rejects weak clutter below `body_snr_margin_db` of your body SNR.

Set `body_calibration_s: 0` to restore instant bootstrap on the first peak.

Hands (id=2) associate **before** the body. Gestures come **only from id=2**.
OSC push/pull is suppressed while **id=1 is moving** (whole-body motion).

## Config2 — simple velocity

1. Pick the **closest** peak above SNR threshold (minimum range)
2. `velocity = (range_now − range_prev) / dt`
3. Push/pull when `|velocity| > min_velocity_mps` for `gesture_confirm_frames`

### Config2 keys (`gesture.config2`)

| Key | Default | Meaning |
|-----|---------|---------|
| `min_velocity_mps` | 0.15 | Push/pull threshold (m/s) |
| `gesture_confirm_frames` | 3 | Frame debounce before raw label counts |

No `/radar/range2_m` or `/radar/snr2_db` in config2.

## Gesture (push / pull)

Negative velocity → **push** (approaching); positive → **pull**.

Both configs debounce frame-level labels with **`gesture_confirm_frames`**. OSC
sends **`/radar/gesture` only when the confirmed label changes** (push, pull, or
``none`` on release). Unchanged frames skip the gesture message; range/SNR still
stream every frame.

## Per-frame peaks (`range_peak_detection`)

| Key | Role |
|-----|------|
| `snr_threshold_db` | Peak cutoff + presence |
| `min_peak_separation_m` | Merge peaks within one frame |
| `max_peaks_per_frame` | Peak list length |

## OSC map (defaults)

| Address | Type | Source |
|---------|------|--------|
| `/radar/mode` | string | live vs replay label |
| `/radar/range_m` | float | T1 (body) range |
| `/radar/snr_db` | float | T1 SNR |
| `/radar/range2_m` | float | T2 (hands) range |
| `/radar/snr2_db` | float | T2 SNR |
| `/radar/doppler_mps` | float | hand track `d(range)/dt` (id=2) |
| `/radar/presence` | float | 0/1 |
| `/radar/gesture` | string | none / push / pull |

Angle is **off** by default (`osc.send_angle: false`).

Analysis plots: `python3 -m post_processing.plot_range_peaks --capture …`

The range–time peak PNG overlays the same NN tracker used for OSC:

- **Blue circles** = track id=1 (body, persistent)
- **Orange** = track id=2 (hands); **^** push / **v** pull on id=2 only

Tracker arrays are saved in `analysis/range_time_peaks.npz`.
