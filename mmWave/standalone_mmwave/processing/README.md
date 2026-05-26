# Radar processing

Turn **DCA UDP captures** (`.npy` frames from `radar_receiver.py`) into a **radar cube** and **range–Doppler** maps.

## Data path (your setup)

```text
UDP packets → FrameBuffer → frame_*.npy (int16)
         → fix_byte_order + IQ de-interleave (lvds.py, cube.py)
         → radar_cube (n_slow, n_ant, n_samples) complex
         → FFT (rda.py) → RD power (dB)
```

This matches **`multimodal-ros/sensors/sensors/dsp.py`** for raw LVDS streaming.

## Not the same as `pipeline_utils.radarDataLoader`

`pipeline_utils.py` on your Desktop uses `DCA1000.decode_data()` on **HSI-header UART streams** in `.npz` files. Your Mac captures are **raw DCA Ethernet** frames — use this `processing/` package instead.

You can still reuse ideas from `pipeline_utils.RD()` / `apply_fft()` once you have `radar_cube` with shape `(n_frames, n_slow, n_ant, n_range)`.

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
| `lvds.py` | `fix_byte_order` from `radar_capture_utils` |
| `cube.py` | int16 frame → complex cube (`dsp._reshape_frame`) |
| `rda.py` | Range + Doppler FFT, axes from `radar_params` |
| `process_capture.py` | CLI |

Optional: `pip install matplotlib` for `--plot`.

## Live stream to Max

Same OSC pattern as `bosch_UWB/standalone_uwb/live_radar_to_max.py`:

```bash
python3 live_radar_to_max.py \
  --cfg ../multimodal-ros/xwr_raw_ros/configs/6843isk/xwr68xx_3Tx_wfv0.5_RDAhigh_19.2m.cfg \
  --cmd-tty /dev/cu.usbserial-00D832110 \
  --roi-min 0.5 --roi-max 10
```

OSC messages (float each):

| Address | Value |
|---------|--------|
| `/radar/range_m` | meters |
| `/radar/doppler_mps` | m/s |
| `/radar/angle_deg` | degrees (FFT on virtual antennas) |
| `/radar/snr_db` | peak − median in RD ROI |
| `/radar/presence` | 1 if SNR above threshold |

Max: `[udpreceive 9000]` → `[OSC-route /radar]` → `[route range_m doppler_mps angle_deg snr_db presence]`
