# How the old plots worked (and why they looked bad)

## Range

1. FFT radar cube → **range–Doppler** power map `RD(doppler, range)` in dB  
2. Subtract rolling median clutter over past frames  
3. **`argmax`** over the whole ROI → one bin wins  
4. **Range** = range_axis[winning_range_bin]

So range was **whatever single cell was brightest**, not a stable person/track. Clutter spikes, wall reflections, or sidelobes often won → jumps like 3 m → 6 m → 3 m in your CSV.

## Angle

1. At the **same** winning (doppler, range) cell  
2. Take complex samples across **12 virtual antennas**  
3. FFT across antennas → pick strongest angle bin in ±90°

Angle is only meaningful if the **correct** range–Doppler cell is selected. Wrong peak → angle is meaningless (hence noisy ±20° swings).

## Doppler / energy / presence

- **Doppler**: velocity at the global peak cell  
- **Energy**: sum of linear power in the whole ROI (not comparable frame-to-frame in plots)  
- **Presence**: SNR at peak vs 12 dB threshold  

All tied to the **one global peak** — not recommended for gestures.

## Better approach: range–time SNR heatmap

See `plot_heatmap.py`:

- Per frame: full **range–Doppler** map, SNR = power − median noise (dB)  
- **Max over Doppler** → range profile vs time  
- Plot **all ranges** at once; targets = regions above `--snr-threshold`  

No single peak tracker.
