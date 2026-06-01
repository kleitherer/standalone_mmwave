# Config: `live_radar_to_max.json`

Single settings file for live OSC, replay, and analysis plots.

## Who reads what

| Block | Used by | Purpose |
|-------|---------|---------|
| `range_peak_detection` | Peak list per frame, plots | **Static** detection on one range profile |
| `gesture.tracking` | Live/replay OSC, tracker overlay | **Temporal** id=1/id=2 + push/pull |
| `processing` | RD declutter, ROI | Background mean, range gate |
| `osc` | Max UDP addresses | `/radar/range_m`, gesture, etc. |

## `range_peak_detection` — per-frame peaks (no memory)

| Key | Meaning |
|-----|---------|
| `snr_threshold_db` | Ignore peaks below this SNR; also `/radar/presence` gate on body |
| `min_peak_separation_m` | Merge peaks closer than this **within the same frame** |
| `max_peaks_per_frame` | Max peaks listed per frame (analysis + tracker input) |
| `require_local_maximum` | Peak must be a local max on the range profile |

Does **not** assign track IDs or gestures.

## `gesture.tracking` — body/hands tracker (OSC path)

| Key | Meaning |
|-----|---------|
| `associate_gate_m` | NN match to predicted range (m) |
| `max_range_jump_m` | Max per-frame step on same ID (m) |
| `min_hand_ahead_m` | id=2 must be **at least** this much closer than id=1 (m) |
| `max_hand_ahead_m` | id=2 must be **at most** this much closer than id=1 (m) |
| `history_frames` | Window for hand velocity → push/pull |
| `max_missed_frames` | Drop id=2 after N misses (id=1 never drops) |
| `min_velocity_mps` | \|d(range)/dt\| threshold for push or pull |

**id=1 (body):** persistent, range only, no gesture.  
**id=2 (hands):** optional, must stay in `[min_hand_ahead_m, max_hand_ahead_m]` in front of body.

## Removed / legacy (do not duplicate)

These used to appear in multiple blocks; **do not re-add**:

| Old key | Was duplicated in | Now use |
|---------|-------------------|---------|
| `secondary_min_range_sep_m` | `range_peak_detection` | `gesture.tracking.min_hand_ahead_m` |
| `secondary_max_range_sep_m` | `range_peak_detection` | `gesture.tracking.max_hand_ahead_m` |
| `gesture.min_range_sep_m` | `gesture` | `gesture.tracking.min/max_hand_ahead_m` |
| `gesture.min_peak_snr_db` | `gesture` | `range_peak_detection.snr_threshold_db` |
| `push_pull` | top-level | Tracker + `min_velocity_mps` (RD push/pull path unused for OSC) |

`gesture_processing.json` is unused — edit this file only.
