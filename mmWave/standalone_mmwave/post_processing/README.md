# Post-processing

## Recommended: range–time SNR heatmap

Shows the **whole scene** over time (no global peak picker):

```bash
python3 -m post_processing.plot_heatmap \
  --capture captures/20260526_115526_push_pull
```

Uses `config/gesture_processing.json` — **max range is learned from background**, not fixed at 6 m.

```bash
# Optional: dedicated empty-room background capture
python3 -m post_processing.plot_heatmap \
  --capture captures/20260526_115526_push_pull \
  --background-capture captures/your_empty_room
```

Outputs in `<capture>/analysis/`:

| File | Description |
|------|-------------|
| `range_time_snr.png` | SNR (dB) vs range & time |
| `range_time_mask.png` | Only pixels ≥ threshold |
| `range_doppler_mean.png` | Average RD map over the clip |
| `range_time_snr.npz` | `snr_db`, `time_s`, `range_m`, … |

**SNR** = range–Doppler power (dB) minus per-frame median noise, then **max over Doppler** per range bin.

Tune `--snr-threshold` (default 8 dB) to define “target” regions in the mask plot.

See `HOW_METRICS_WORKED.md` for why the old peak-picker line plots looked bad.

---

## Legacy: peak-picker line plots

Analyze any raw capture and plot **range, angle, Doppler, energy, presence** vs. time (single global peak — often misleading).

## Quick start

```bash
cd standalone_mmwave

# Your two_step capture
python3 -m post_processing.plot_gesture \
  --capture captures/20260526_115737_two_step \
  --save-only

# push_pull (when recorded)
python3 -m post_processing.plot_gesture \
  --capture captures/YYYYMMDD_HHMMSS_push_pull \
  --save-only
```

## Outputs

Under `<capture>/analysis/`:

| File | Content |
|------|---------|
| `gesture_timeseries.csv` | Per-frame columns for replotting / ML |
| `gesture_plots.png` | 5 stacked plots vs. time |

## Plots

1. **range (m)** — distance to strongest target in ROI  
2. **angle (deg)** — FFT azimuth on virtual antennas at peak cell  
3. **Doppler (m/s)** — radial velocity at peak  
4. **energy** — sum of linear power in range–Doppler ROI (log y-axis)  
5. **presence** — 1 if SNR ≥ threshold (default 12 dB)

## Options

```bash
python3 -m post_processing.plot_gesture --capture captures/... \
  --roi-min 0.5 --roi-max 10 \
  --smooth-alpha 0.15 \
  --presence-threshold-db 12 \
  --max-frames 0

# Replot from CSV only
python3 -m post_processing.plot_gesture --capture captures/... --plot-only --save-only
```

## Python API

```python
from post_processing.analyze_capture import analyze_capture

ts = analyze_capture("captures/20260526_115737_two_step")
print(ts.range_m, ts.time_s)
ts.to_csv("out.csv")
```
