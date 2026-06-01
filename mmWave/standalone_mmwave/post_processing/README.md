# Post-processing plots

All scripts read `config/live_radar_to_max.json` unless you pass `--config`.  
Input is always **`captures/<session>/raw/frame_*.npy`** (header-stripped ADC int16).

## Main entry: `plot_heatmap`

```bash
python3 -m post_processing.plot_heatmap --capture testing_gesture_osc
```

Writes under `<capture>/analysis/` (controlled by `post_processing.*` flags):

| Output | What it is |
|--------|------------|
| **`range_time_snr.png`** | **SNR (dB)** vs time × range. Primary tracking view. |
| **`range_time_declutter_power.png`** | Same processing chain as SNR, but **power (dB)** after background subtract (no per-frame noise step). |
| **`range_time_snr.npz`** | Arrays: `snr_db`, `time_s`, `range_m`, … |
| **`range_time_declutter_power.npz`** | Arrays: `power_db`, `time_s`, `range_m` |
| **`rd_heatmap_movie.mp4`** | Per-frame range–Doppler movies (if `write_rd_movie`) |
| **`rd_heatmap_frame<N>.png`** | Single RD snapshot (if `rd_snapshot_frame` set) |

### Processing chain (range–time plots)

Shared steps for SNR and declutter-power:

1. **Chirp-mean declutter** — subtract mean over slow-time chirps before FFT (per frame).
2. **Optional limiter** — `post_processing.range_time_limiter` (1-bit IQ; display only).
3. **RD FFT** — standalone pipeline: Hann window, range + Doppler FFT, `mean(|RDa|²)` → dB.
4. **Background subtract** — subtract a global RD clutter map:
   - `background_session_mean: true` → mean over **all frames** in the capture.
   - `background_session_mean: false` → mean of first `declutter_mean_frames` only.
   - Or separate empty-room capture via `background.capture` in config.
5. **Range gate** — `processing.roi_min_m` … `roi_max_m`.
6. **Collapse Doppler** — `max` over Doppler → one value per range bin per frame.

**SNR only (step 7):** subtract the **per-frame mean** of the background-subtracted RD ROI (all Doppler×range cells in gate), then max over Doppler:

```text
SNR(d, r) = RD_declutter(d, r) − mean(RD_declutter in ROI this frame)
range–time cell = max_d SNR(d, r)
```

Uses **mean** (not median) so the noise reference tracks the average cell level; bright targets pull the reference up slightly, which you found works better than median here.

**Declutter-power** stops after step 6 — no step 7. Expect a higher floor and more “smear” around targets.

### Other plot scripts

| Script | Output | Notes |
|--------|--------|--------|
| `plot_range_time_power` | `range_time_power.png` | **Different pipeline**: mmw `rd_heatmap` style, raw power, **no** background subtract. For reference NPZ / mmw parity. |
| `plot_range_peaks` | `range_time_peaks.png` | Peak tracks on range–time SNR. |
| `plot_rd_movie` | RD movie | Same RD display settings as heatmap movie path. |
| `plot_range_azimuth_movie` | `range_azimuth_movie.mp4` | Range × azimuth over time. |
| `plot_ud_continuous` | Micro-Doppler PNG | Slow-time / STFT uD from cube or NPZ. |
| `plot_rd_frame` | Debug RD png | Raw vs decluttered vs SNR for one frame. |

---

## Why does the background look louder when someone is in the frame?

Same room, but the **map is per-frame** and a person is a **distributed strong reflector**. You see higher “background” when a target is present mainly because of **processing**, not because the empty room got noisier:

1. **FFT sidelobes + Hann window** — A strong return spreads energy into neighboring Doppler and range bins. **`max over Doppler`** then reports that leakage at many range cells → vertical smear and elevated power around the target on **`range_time_declutter_power`**.

2. **Session-mean background** — With `background_session_mean: true`, the clutter map includes frames **with** the person. That helps subtract static walls, but the template is wrong for frames where the person is at a **different** range/Doppler, and it never fully removes **moving** energy. Residual + sidelobes raise the floor on target frames.

3. **Per-frame mean SNR step** — On target frames, many bins are bright, so the **mean** noise reference is **higher** than on empty frames. After subtraction, empty-looking range bins can still show sidelobe peaks above the reference; on **`declutter_power`** (no mean step) the whole frame looks brighter when the target is on.

4. **Physical spread** — Torso, arms, and multipath fill multiple range/Doppler cells; that is real energy, not just one bin.

**SNR** (`range_time_snr.png`) hides much of this via step 7 (per-frame mean). **Declutter-power** shows it raw — use SNR for tracking, power for comparing to mmw reference or debugging clutter.

---

## Config cheat sheet (`processing` + `post_processing`)

```json
"processing": {
  "roi_min_m": 0.3,
  "roi_max_m": 3.4,
  "declutter_mean_frames": 45,
  "background_session_mean": true,
  "snr_per_frame_median": true
},
"post_processing": {
  "write_range_time_snr": true,
  "write_range_time_declutter_power": true,
  "range_time_limiter": true,
  "write_rd_movie": true
}
```

`snr_per_frame_median: true` → per-frame mean/median via `rd_to_snr_db` (mean in code today).  
`false` → fixed noise from first N frames (legacy; usually worse for gesture clips).
