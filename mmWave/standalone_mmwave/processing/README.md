# Radar processing

Turn **DCA UDP captures** (`.npy` frames) into a **radar cube**, **range–Doppler** maps, and (via `target_detect.py`) live **range / velocity / angle / gesture**.

**New here?** Read the full pipeline roadmap: **[../docs/PROCESSING_ROADMAP.md](../docs/PROCESSING_ROADMAP.md)** (where FFTs run, raw → OSC, file map).

## FFT summary

| FFT | File | Purpose |
|-----|------|---------|
| Range + Doppler | `rda.py` → `compute_rda()` | fast-time → range; slow-time → velocity |
| Azimuth | `angle_estimate.py` | virtual antennas → angle at RD peak |

## Data path

```text
frame_*.npy (int16) → cube.py → radar_cube
  → rda.py (2× FFT) → RD dB → background subtract → target_detect.py
  → angle FFT → range_m, doppler_mps, angle_deg, gesture
```

Matches **`multimodal-ros/sensors/dsp.py`** for DCA LVDS captures (not Desktop `pipeline_utils` UART `.npz` format).

## Usage

```bash
cd standalone_mmwave

# After --capture run:
python3 -m processing.process_capture --capture captures/YYYYMMDD_HHMMSS --plot

# One frame:
python3 -m processing.process_capture \
  --frame captures/.../frame_000001.npy \
  --metadata captures/.../metadata.json \
  --plot

# From .cfg only (no metadata.json):
python3 -m processing.process_capture \
  --frame frame_000001.npy \
  --cfg ../multimodal-ros/xwr_raw_ros/configs/6843isk/xwr68xx_3Tx_wfv0.5_RDAhigh_19.2m.cfg \
  --plot
```

Output: `processed.npz` with `radar_cubes`, `rd_power_db`, `range_axis`, `doppler_axis`.

## Cube shape (6843 3Tx profile)

| Axis | Size | Meaning |
|------|------|---------|
| 0 | 32 | Slow time (96 chirps / 3 TX) |
| 1 | 12 | Virtual antennas (4 RX × 3 TX) |
| 2 | 256 | Range samples |

## Modules

| File | Role |
|------|------|
| `lvds.py` | Byte order fix for DCA payload |
| `cube.py` | int16 frame → complex radar cube |
| `rda.py` | **Range + Doppler FFT**, `rda_power_db` |
| `angle_estimate.py` | **Antenna FFT** → azimuth |
| `target_detect.py` | Peak pick, SNR, gesture, live `LiveRadarTargetProcessor` |
| `range_azimuth.py` | Offline range–azimuth maps (per-range Doppler + antenna FFT) |
| `osc_utils.py` | OSC message bytes for Max |
| `process_capture.py` | Batch CLI → `processed.npz` |

Live/replay OSC: `../live_radar_to_max.py`, `../replay_capture_to_max.py` — see [../docs/push_pull.md](../docs/push_pull.md).
