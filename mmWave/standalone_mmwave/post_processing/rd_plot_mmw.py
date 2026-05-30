"""Range–Doppler plotting aligned with mmw-tracking-versions/rd_heatmap.py."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from background_model import estimate_rd_background_from_capture, frame_rd_power_db
from capture_store import CaptureSession
from post_processing.rd_maps import _frame_times, rd_to_snr_db
from processing.rda import range_doppler_axes

if TYPE_CHECKING:
    from post_processing.processing_config import ProcessingConfig

# mmw-tracking rd_heatmap.py / process_frame_RD_CLEAN defaults
RD_CMAP = "jet"


def full_range_gate(params: dict[str, Any]) -> tuple[float, float]:
    """Use the full configured range axis (matches rd_heatmap.py extent)."""
    return 0.0, float(params["range_max"])


def collect_rd_frames(
    capture_path: Path,
    *,
    range_gate_m: tuple[float, float] | None = None,
    background_capture: Path | None = None,
    background_max_frames: int = 0,
    calibration_frames: int = 0,
    max_frames: int = 0,
    subtract_background: bool = False,
    use_snr: bool = False,
    show_progress: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-frame RD maps for plotting.

    Default (mmw-tracking style): raw log-power (dB), full range, DC declutter only
    (inside ``frame_rd_power_db``), no temporal background subtract, no SNR median.

    Returns
    -------
    frames : list of (n_doppler, n_range) float32
    range_m, doppler_mps, time_s
    """
    session = CaptureSession.open(capture_path)
    params = session.radar_params()
    range_axis, doppler_axis = range_doppler_axes(params)

    if range_gate_m is None:
        range_gate_m = full_range_gate(params)
    r_lo, r_hi = range_gate_m
    r_mask = (range_axis >= r_lo) & (range_axis <= r_hi)
    range_m = range_axis[r_mask]
    doppler_mps = doppler_axis

    paths = session.frame_paths()
    if max_frames > 0:
        paths = paths[:max_frames]
    if not paths:
        raise RuntimeError(f"No frames in {capture_path}")

    rd_raw_list: list[np.ndarray] = []
    for i, path in enumerate(paths):
        rd_raw_list.append(frame_rd_power_db(np.load(path), params))
        if show_progress and (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{len(paths)} frames…", flush=True)

    if subtract_background:
        if background_capture is not None:
            bg = estimate_rd_background_from_capture(
                Path(background_capture), params, max_frames=background_max_frames
            )
        else:
            n_calib = min(len(rd_raw_list), max(1, int(calibration_frames)))
            bg = np.mean(np.stack(rd_raw_list[:n_calib], axis=0), axis=0)
    else:
        bg = 0.0

    frames_out: list[np.ndarray] = []
    for rd_raw in rd_raw_list:
        rd = rd_raw - bg
        rd_roi = rd[:, r_mask]
        if use_snr:
            rd_roi = rd_to_snr_db(rd_roi)
        frames_out.append(rd_roi.astype(np.float32))

    time_s = _frame_times(session, len(frames_out))
    return frames_out, range_m, doppler_mps, time_s


def rd_imshow_kwargs(
    range_m: np.ndarray,
    doppler_mps: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> dict[str, Any]:
    """imshow kwargs matching mmw-tracking rd_heatmap.py (range on x, Doppler on y)."""
    return {
        "aspect": "auto",
        "cmap": RD_CMAP,
        "extent": [
            float(range_m[0]),
            float(range_m[-1]),
            float(doppler_mps[-1]),
            float(doppler_mps[0]),
        ],
        "origin": "upper",
        "vmin": vmin,
        "vmax": vmax,
        "interpolation": "nearest",
    }


def color_limits(stack: np.ndarray, *, use_percentile: bool) -> tuple[float | None, float | None]:
    if not use_percentile:
        return None, None
    return float(np.percentile(stack, 5)), float(np.percentile(stack, 99))


def plot_rd_frame(
    rd_db: np.ndarray,
    range_m: np.ndarray,
    doppler_mps: np.ndarray,
    out_path: Path,
    *,
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 150,
) -> None:
    """Static RD heatmap — same layout as mmw-tracking ``rd_heatmap.py``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(rd_db, **rd_imshow_kwargs(range_m, doppler_mps, vmin=vmin, vmax=vmax))
    fig.colorbar(im, ax=ax, label="SNR (dB)" if title.lower().find("snr") >= 0 else "Power (dB)")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Doppler (m/s)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_rd_movie(
    rd_frames: list[np.ndarray],
    range_m: np.ndarray,
    doppler_mps: np.ndarray,
    time_s: np.ndarray,
    out_path: Path,
    *,
    fps: float,
    fmt: str,
    session_id: str,
    dpi: int = 100,
    use_percentile_scale: bool = True,
    colorbar_label: str = "Power (dB)",
) -> None:
    """Animated RD heatmap with mmw-tracking axis layout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    stack = np.stack(rd_frames, axis=0)
    vmin, vmax = color_limits(stack, use_percentile=use_percentile_scale)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        rd_frames[0],
        **rd_imshow_kwargs(range_m, doppler_mps, vmin=vmin, vmax=vmax),
    )
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Doppler (m/s)")
    title = ax.set_title("")
    fig.colorbar(im, ax=ax, label=colorbar_label)

    def _update(i: int):
        im.set_data(rd_frames[i])
        t = float(time_s[i]) if i < len(time_s) else float(i)
        title.set_text(f"{session_id} — frame {i + 1}/{len(rd_frames)}  t={t:.2f}s")
        return (im, title)

    interval_ms = 1000.0 / max(fps, 0.1)
    anim = FuncAnimation(fig, _update, frames=len(rd_frames), interval=interval_ms, blit=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "gif":
        writer = PillowWriter(fps=fps)
    else:
        try:
            writer = FFMpegWriter(fps=fps, bitrate=1800)
        except Exception as e:
            raise RuntimeError(
                "MP4 export needs ffmpeg. Install ffmpeg or use --format gif"
            ) from e

    print(f"Writing {out_path} ({len(rd_frames)} frames @ {fps:.1f} fps)…", flush=True)
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)


def write_rd_outputs(
    capture: Path,
    proc: ProcessingConfig,
    out_dir: Path,
    *,
    max_frames: int = 0,
) -> Path | None:
    """
    ROI-gated RD movie and optional single-frame snapshot.

    Snapshot is written only when ``post_processing.rd_snapshot_frame`` is set in config.
    Returns path to movie, or None if ``write_rd_movie`` is false.
    """
    from post_processing.processing_config import resolve_capture_range_gate

    pp = proc.post_processing
    session = CaptureSession.open(capture)
    params = session.radar_params()
    radar_max = float(params["range_max"])
    r_min, r_max, _ = resolve_capture_range_gate(capture, proc, radar_max_m=radar_max)

    subtract_bg = pp.effective_subtract_background(background_capture=proc.background.capture)
    frame_time_ms = float(params.get("frame_time", 22.22))
    fps = pp.movie_fps if pp.movie_fps > 0 else 1000.0 / frame_time_ms

    rd_frames, range_m, doppler_mps, time_s = collect_rd_frames(
        capture,
        range_gate_m=(r_min, r_max),
        calibration_frames=proc.declutter_mean_frames,
        background_capture=proc.background.capture,
        background_max_frames=proc.background.max_frames,
        max_frames=max_frames,
        subtract_background=subtract_bg,
        use_snr=pp.use_snr,
    )
    if not rd_frames:
        raise RuntimeError(f"No RD frames for {capture}")

    stack = np.stack(rd_frames, axis=0)
    vmin, vmax = color_limits(stack, use_percentile=pp.percentile_color_scale)
    colorbar_label = "SNR (dB)" if pp.use_snr else "Power (dB)"

    if pp.rd_snapshot_frame is not None:
        idx = int(pp.rd_snapshot_frame)
        if idx < 0 or idx >= len(rd_frames):
            raise ValueError(
                f"post_processing.rd_snapshot_frame={idx} out of range [0, {len(rd_frames) - 1}]"
            )
        plot_rd_frame(
            rd_frames[idx],
            range_m,
            doppler_mps,
            out_dir / f"rd_heatmap_frame{idx}.png",
            title=f"{capture.name} — RD heat map (frame {idx})",
            vmin=vmin,
            vmax=vmax,
        )

    if not pp.write_rd_movie:
        return None

    fmt = pp.movie_format if pp.movie_format in ("mp4", "gif") else "mp4"
    out_path = out_dir / f"rd_heatmap_movie.{fmt}"
    render_rd_movie(
        rd_frames,
        range_m,
        doppler_mps,
        time_s,
        out_path,
        fps=fps,
        fmt=fmt,
        session_id=capture.name,
        dpi=pp.movie_dpi,
        use_percentile_scale=pp.percentile_color_scale,
        colorbar_label=colorbar_label,
    )
    return out_path
