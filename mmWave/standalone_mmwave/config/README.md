# Gesture processing config

Edit `gesture_processing.json` for your **room / radar setup** — not hardcoded to 6 m.

## Range limit from background

By default `range_gate.max_m` is `null` → **auto** from background:

1. Build median SNR vs range from background data  
2. Find outer edge of static clutter (walls)  
3. Set `range_max = edge − margin_m`  
4. Drop all range bins beyond that in heatmaps / processing  

**Background source** (pick one):

| Setting | Use when |
|---------|----------|
| `background.capture: null` | Use **first `use_start_frames`** of each gesture (stay still ~1 s at start) |
| `background.capture: "captures/empty_room"` | Dedicated empty-room recording (best) |

## Example: your lab after measuring

If background profile shows clutter out to ~6.2 m, either let auto pick it, or lock manually:

```json
"range_gate": {
  "min_m": 0.5,
  "max_m": 6.0,
  "max_m_cap": null
}
```

## Tunables

| Field | Meaning |
|-------|---------|
| `profile_drop_db` | How far below peak profile counts as “end of scene” (default 12 dB) |
| `margin_m` | Pull max range inward from edge (default 0.3 m) |
| `min_above_median_db` | Bin must exceed median profile by this much to be “scene” |
| `max_m_cap` | Optional hard ceiling (e.g. 10) even if auto estimates higher |

Applied gate is saved per run: `<capture>/analysis/range_gate.json`.
