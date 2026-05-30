# Post-processing

This repo now keeps only the heatmap path for offline analysis.
By default it reads processing values from `config/live_radar_to_max.json`.

## Range–azimuth movie (per frame)

```bash
python3 -m post_processing.plot_range_azimuth_movie --capture captures/push_pull
python3 -m post_processing.plot_range_azimuth_movie --capture push_pull --angle-bins 1024 --format mp4
```

Output: `<capture>/analysis/range_azimuth_movie.mp4` (or `.gif` without ffmpeg).

Same processing as the static `range_azimuth.png`: per range bin, pick strongest Doppler then antenna FFT (live-style).

## Range-time SNR heatmap

```bash
python3 -m post_processing.plot_heatmap \
  --capture captures/20260526_115526_push_pull
```

Optional dedicated background capture:

```bash
python3 -m post_processing.plot_heatmap \
  --capture captures/20260526_115526_push_pull \
  --background-capture captures/your_empty_room
```

Outputs in `<capture>/analysis/`:

- `range_time_snr.png`: SNR (dB) vs range and time
- `doppler_time_snr.png`: SNR (dB) vs Doppler and time (max over range)
- `azimuth_time_snr_fft.png` (or `_music`): SNR (dB) vs azimuth and time (max over range)
- `range_time_mask.png`: pixels above threshold only
- `range_doppler_mean.png`: average range-Doppler map
- `range_doppler_peak.png`: peak range-Doppler SNR over time
- `range_doppler_raw_vs_decluttered.png`: raw vs decluttered RD means
- `range_azimuth_fft.png` or `range_azimuth_music.png`: range–azimuth heatmap (x = angle, y = range)
- `range_azimuth_<method>.npz`: arrays for custom plots
- `range_time_raw_vs_decluttered.png`: side-by-side raw vs clutter-removed
- `range_time_snr.npz`: arrays (`snr_db`, `raw_db`, `declutter_db`, `time_s`, `range_m`, ...)

## Processing alignment with live

- Declutter method is global-mean only.
- Background is the global mean RD map from the first `declutter_mean_frames` frames of the capture (or a separate empty-room capture). Live streaming uses `calibration_frames: 0` so recording starts immediately with no warmup.
- SNR is computed from decluttered RD power relative to per-frame median noise floor.
- Range limits use `processing.roi_min_m` / `processing.roi_max_m` from `config/live_radar_to_max.json`.
- Azimuth uses `angle_estimation.method`: `fft` (default) or `music` (super-resolution).

```json
"angle_estimation": {
  "method": "fft",
  "fft_bins": 128,
  "fov_deg": 90
}
```

```bash
python3 -m post_processing.plot_heatmap --capture captures/your_capture --angle-method music
```

## Range–Doppler movie (frame-by-frame)

```bash
python3 -m post_processing.plot_rd_movie --capture captures/push_pull
python3 -m post_processing.plot_rd_movie --capture push_pull --format gif --fps 15
```

Output: `<capture>/analysis/rd_heatmap_movie.mp4` (or `.gif`). Uses same config/declutter/ROI as other post-processing. MP4 needs `ffmpeg` installed.
