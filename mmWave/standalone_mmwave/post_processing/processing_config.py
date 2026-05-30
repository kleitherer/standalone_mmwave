"""Load live + post-processing settings from config/live_radar_to_max.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

_DEFAULT = Path(__file__).resolve().parents[1] / "config" / "live_radar_to_max.json"

_RD_DISPLAY_CHOICES = frozenset({"raw_power", "decluttered", "snr"})
_RD_PIPELINE_CHOICES = frozenset({"mmw", "standalone"})


@dataclass
class BackgroundConfig:
    capture: Optional[Path]
    max_frames: int
    use_start_frames: int
    profile_drop_db: float
    margin_m: float
    min_above_median_db: float


@dataclass
class AngleEstimationConfig:
    method: str  # "fft" or "music"
    fft_bins: int
    fov_deg: float


@dataclass
class PostProcessingConfig:
    """
    Plot / analysis defaults (``post_processing`` block in live_radar_to_max.json).

    Range limits always come from ``processing.roi_min_m`` / ``processing.roi_max_m``
    unless ``auto_range_gate`` is true.
    """

    rd_display: str  # raw_power | decluttered | snr
    subtract_background: bool
    auto_range_gate: bool
    write_range_time_snr: bool
    write_rd_movie: bool
    rd_snapshot_frame: Optional[int]
    movie_format: str
    movie_fps: float
    percentile_color_scale: bool
    movie_dpi: int
    rd_doppler_fft_bins: int  # 0 = native; e.g. 256 for display interpolation
    rd_range_fft_bins: int  # 0 = native (256 samples); e.g. 512 for finer range display
    rd_vmin_db: float  # fixed color scale (mmw default 25 * 1.8 = 45)
    rd_vmax_db: float  # fixed color scale (mmw default 40 * 2 = 80)
    rd_display_upsample: int  # bilinear upsample of RD grid before imshow (1 = off)
    rd_interpolation: str  # matplotlib imshow interpolation (bilinear recommended)
    rd_pipeline: str  # mmw | standalone — mmw matches rd_heatmap.py exactly
    rd_full_range: bool  # mmw style: plot 0..range_max (ignore ROI for RD heatmap)
    range_time_limiter: bool  # 1-bit limiter for range-time SNR display only (clean single-target track)

    @property
    def use_snr(self) -> bool:
        return self.rd_display == "snr"

    def rd_fft_sizes(self, params: dict[str, Any]) -> tuple[int | None, int | None]:
        """Display-only zero-pad lengths (None = native, no upsampling)."""
        if self.rd_pipeline == "mmw":
            return None, None
        n_slow = int(params.get("n_slow", int(params["n_chirps"]) // int(params["n_tx"])))
        n_samples = int(params["n_samples"])
        n_doppler = self.rd_doppler_fft_bins if self.rd_doppler_fft_bins > n_slow else None
        n_range = self.rd_range_fft_bins if self.rd_range_fft_bins > n_samples else None
        return n_doppler, n_range

    def effective_subtract_background(self, *, background_capture: Path | None) -> bool:
        if background_capture is not None:
            return True
        if self.subtract_background:
            return True
        return self.rd_display in ("decluttered", "snr")


@dataclass
class ProcessingConfig:
    config_path: Path
    snr_threshold_db: float
    calibration_frames: int
    declutter_mean_frames: int
    range_min_m: float
    range_max_m: Optional[float]
    range_max_m_cap: Optional[float]
    angle: AngleEstimationConfig
    background: BackgroundConfig
    post_processing: PostProcessingConfig

    @classmethod
    def load(cls, path: Path | None = None) -> "ProcessingConfig":
        path = Path(path) if path else _DEFAULT
        data = json.loads(path.read_text())
        rg = data.get("range_gate", {})
        ang = data.get("angle_estimation", data.get("angle", {}))
        pp = data.get("post_processing", {})
        if "processing" in data:
            p = data.get("processing", {})
            bg = data.get("background", {})
            range_min_m = float(p.get("roi_min_m", 0.5))
            range_max_m = _opt_float(p.get("roi_max_m"))
            calibration_frames = int(p.get("calibration_frames", p.get("clutter_window", 0)))
            declutter_mean_frames = int(
                p.get(
                    "declutter_mean_frames",
                    calibration_frames if calibration_frames > 0 else 45,
                )
            )
            snr_threshold_db = float(p.get("presence_threshold_db", 12.0))
            if not ang and "angle_method" in p:
                ang = {
                    "method": p.get("angle_method", "fft"),
                    "fft_bins": p.get("angle_fft_bins", 128),
                    "fov_deg": p.get("angle_fov_deg", 90),
                }
        else:
            bg = data.get("background", {})
            rg = data.get("range_gate", {})
            range_min_m = float(rg.get("min_m", 0.5))
            range_max_m = _opt_float(rg.get("max_m"))
            calibration_frames = int(data.get("calibration_frames", data.get("clutter_window_frames", 0)))
            declutter_mean_frames = int(
                data.get(
                    "declutter_mean_frames",
                    calibration_frames if calibration_frames > 0 else 16,
                )
            )
            snr_threshold_db = float(data.get("snr_threshold_db", 8.0))

        method = str(ang.get("method", "fft")).lower()
        if method not in {"fft", "music"}:
            raise ValueError(f"angle_estimation.method must be 'fft' or 'music', got {method!r}")

        rd_display = str(pp.get("rd_display", "raw_power")).lower()
        if rd_display not in _RD_DISPLAY_CHOICES:
            raise ValueError(
                f"post_processing.rd_display must be one of {sorted(_RD_DISPLAY_CHOICES)}, got {rd_display!r}"
            )
        rd_pipeline = str(pp.get("rd_pipeline", "mmw")).lower()
        if rd_pipeline not in _RD_PIPELINE_CHOICES:
            raise ValueError(
                f"post_processing.rd_pipeline must be one of {sorted(_RD_PIPELINE_CHOICES)}, got {rd_pipeline!r}"
            )

        return cls(
            config_path=path.resolve(),
            snr_threshold_db=snr_threshold_db,
            calibration_frames=calibration_frames,
            declutter_mean_frames=declutter_mean_frames,
            range_min_m=range_min_m,
            range_max_m=range_max_m,
            range_max_m_cap=_opt_float(rg.get("max_m_cap")),
            angle=AngleEstimationConfig(
                method=method,
                fft_bins=int(ang.get("fft_bins", ang.get("angle_fft_bins", 128))),
                fov_deg=float(ang.get("fov_deg", ang.get("angle_fov_deg", 90))),
            ),
            background=BackgroundConfig(
                capture=Path(bg["capture"]) if bg.get("capture") else None,
                max_frames=int(bg.get("max_frames", 0)),
                use_start_frames=int(bg.get("use_start_frames", 45)),
                profile_drop_db=float(bg.get("profile_drop_db", 12.0)),
                margin_m=float(bg.get("margin_m", 0.3)),
                min_above_median_db=float(bg.get("min_above_median_db", 6.0)),
            ),
            post_processing=PostProcessingConfig(
                rd_display=rd_display,
                subtract_background=bool(pp.get("subtract_background", False)),
                auto_range_gate=bool(pp.get("auto_range_gate", False)),
                write_range_time_snr=bool(
                    pp.get("write_range_time_snr", pp.get("snr_analysis", True))
                ),
                write_rd_movie=bool(pp.get("write_rd_movie", True)),
                rd_snapshot_frame=_opt_int(pp.get("rd_snapshot_frame")),
                movie_format=str(pp.get("movie_format", "mp4")),
                movie_fps=float(pp.get("movie_fps", 0.0)),
                percentile_color_scale=bool(pp.get("percentile_color_scale", True)),
                movie_dpi=int(pp.get("movie_dpi", 100)),
                rd_doppler_fft_bins=int(pp.get("rd_doppler_fft_bins", 256)),
                rd_range_fft_bins=int(pp.get("rd_range_fft_bins", 0)),
                rd_vmin_db=float(pp.get("rd_vmin_db", 25.0 * 1.8)),
                rd_vmax_db=float(pp.get("rd_vmax_db", 40.0 * 2.0)),
                rd_display_upsample=max(1, int(pp.get("rd_display_upsample", 1))),
                rd_interpolation=str(pp.get("rd_interpolation", "bilinear")),
                rd_pipeline=rd_pipeline,
                rd_full_range=bool(pp.get("rd_full_range", True)),
                range_time_limiter=bool(pp.get("range_time_limiter", False)),
            ),
        )

    def save_applied(self, out_path: Path, applied: dict) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_source": str(self.config_path),
            "applied": applied,
        }
        out_path.write_text(json.dumps(payload, indent=2))


def _opt_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    return float(v)


def _opt_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    return int(v)


def resolve_range_gate(
    cfg: ProcessingConfig,
    *,
    estimated_max_m: float,
) -> Tuple[float, float]:
    """Return (min_m, max_m) from ``processing.roi_min_m`` / ``roi_max_m``."""
    r_min = cfg.range_min_m
    if cfg.range_max_m is not None:
        r_max = cfg.range_max_m
    else:
        r_max = estimated_max_m
    if cfg.range_max_m_cap is not None:
        r_max = min(r_max, cfg.range_max_m_cap)
    if r_max <= r_min:
        r_max = r_min + 0.5
    return r_min, r_max


def resolve_capture_range_gate(
    capture: Path,
    cfg: ProcessingConfig,
    *,
    radar_max_m: float,
) -> Tuple[float, float, dict | None]:
    """
    Range gate for post-processing.

    Uses ``processing.roi_min_m`` / ``roi_max_m`` from live_radar_to_max.json.
    When ``post_processing.auto_range_gate`` is true, ``roi_max_m`` is ignored and
    the max range is estimated from a background profile.
    """
    if cfg.post_processing.auto_range_gate:
        from post_processing.range_gate import estimate_range_gate_for_capture

        r_min, r_max, info = estimate_range_gate_for_capture(
            capture, cfg, radar_max_range_m=radar_max_m
        )
        return r_min, r_max, info
    r_min, r_max = resolve_range_gate(cfg, estimated_max_m=radar_max_m)
    return r_min, r_max, None
