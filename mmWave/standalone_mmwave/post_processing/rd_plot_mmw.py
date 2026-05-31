"""Range–Doppler plotting aligned with mmw-tracking-versions/rd_heatmap.py."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from background_model import estimate_rd_background_from_capture
from capture_store import CaptureSession
from post_processing.rd_maps import _frame_times, rd_to_snr_db
from processing.mmw_rd import mmw_range_doppler_axes
from processing.rd_map import frame_to_rd_power_db
from processing.rda import range_doppler_axes

if TYPE_CHECKING:
    from post_processing.processing_config import ProcessingConfig

# mmw-tracking rd_heatmap.py / process_frame_RD_CLEAN defaults
RD_CMAP = "jet"


def upsample_2d_bilinear(arr: np.ndarray, factor: int) -> np.ndarray:
    """Display-only bilinear upsample (same physical extent, finer pixel grid)."""
    if factor <= 1:
        return arr
    fy, fx = arr.shape
    ny, nx = fy * factor, fx * factor
    yi = np.linspace(0.0, fy - 1.0, ny)
    xi = np.linspace(0.0, fx - 1.0, nx)
    row_interp = np.vstack([np.interp(xi, np.arange(fx), arr[i]) for i in range(fy)])
    out = np.empty((ny, nx), dtype=arr.dtype)
    for j in range(nx):
        out[:, j] = np.interp(yi, np.arange(fy), row_interp[:, j])
    return out


def prepare_rd_for_display(rd_db: np.ndarray, *, upsample: int = 1) -> np.ndarray:
    return upsample_2d_bilinear(np.asarray(rd_db, dtype=np.float32), upsample)


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
    n_doppler_fft: int | None = None,
    n_range_fft: int | None = None,
    rd_pipeline: str = "mmw",
    rd_full_range: bool = True,
    show_progress: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-frame RD maps for plotting.

    Default (mmw-tracking style): ``rd_pipeline='mmw'`` uses ``processing.mmw_rd``
    (same as ``rd_heatmap.py``): DC declutter, Hann window, Doppler then range FFT,
    ``pw2db(mean(|RDa|²))``, full range axis, no zero-pad.

    Returns
    -------
    frames : list of (n_doppler, n_range) float32
    range_m, doppler_mps, time_s
    """
    session = CaptureSession.open(capture_path)
    params = session.radar_params()
    use_mmw = rd_pipeline == "mmw"

    if use_mmw:
        range_axis, doppler_axis = mmw_range_doppler_axes(params)
    else:
        range_axis, doppler_axis = range_doppler_axes(
            params, n_doppler_fft=n_doppler_fft, n_range_fft=n_range_fft
        )

    if range_gate_m is None or (use_mmw and rd_full_range):
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

    pipeline: str = "mmw" if use_mmw else "standalone"
    rd_raw_list: list[np.ndarray] = []
    for i, path in enumerate(paths):
        rd_raw_list.append(
            frame_to_rd_power_db(
                np.load(path),
                params,
                pipeline=pipeline,  # type: ignore[arg-type]
                n_doppler_fft=n_doppler_fft,
                n_range_fft=n_range_fft,
            )
        )
        if show_progress and (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{len(paths)} frames…", flush=True)

    if subtract_background:
        if background_capture is not None:
            bg = estimate_rd_background_from_capture(
                Path(background_capture),
                params,
                max_frames=background_max_frames,
                n_doppler_fft=n_doppler_fft,
                n_range_fft=n_range_fft,
                pipeline=pipeline,
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
    interpolation: str = "bilinear",
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
        "interpolation": interpolation,
        "filterrad": 4.0,
    }


def color_limits(
    stack: np.ndarray,
    *,
    use_percentile: bool,
    vmin_db: float = 45.0,
    vmax_db: float = 80.0,
) -> tuple[float, float]:
    if use_percentile:
        return float(np.percentile(stack, 5)), float(np.percentile(stack, 99))
    return float(vmin_db), float(vmax_db)


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
    upsample: int = 1,
    interpolation: str = "bilinear",
) -> None:
    """Static RD heatmap — same layout as mmw-tracking ``rd_heatmap.py``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rd_show = prepare_rd_for_display(rd_db, upsample=upsample)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        rd_show,
        **rd_imshow_kwargs(
            range_m, doppler_mps, vmin=vmin, vmax=vmax, interpolation=interpolation
        ),
    )
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
    vmin: float = 45.0,
    vmax: float = 80.0,
    colorbar_label: str = "Power (dB)",
    upsample: int = 1,
    interpolation: str = "bilinear",
) -> None:
    """Animated RD heatmap with mmw-tracking axis layout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    rd0 = prepare_rd_for_display(rd_frames[0], upsample=upsample)
    imshow_kw = rd_imshow_kwargs(
        range_m, doppler_mps, vmin=vmin, vmax=vmax, interpolation=interpolation
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(rd0, **imshow_kw)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Doppler (m/s)")
    title = ax.set_title("")
    fig.colorbar(im, ax=ax, label=colorbar_label)

    def _update(i: int):
        im.set_data(prepare_rd_for_display(rd_frames[i], upsample=upsample))
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
    n_doppler_fft, n_range_fft = pp.rd_fft_sizes(params)

    rd_frames, range_m, doppler_mps, time_s = collect_rd_frames(
        capture,
        range_gate_m=(r_min, r_max),
        calibration_frames=proc.declutter_mean_frames,
        background_capture=proc.background.capture,
        background_max_frames=proc.background.max_frames,
        max_frames=max_frames,
        subtract_background=subtract_bg,
        use_snr=pp.use_snr,
        n_doppler_fft=n_doppler_fft,
        n_range_fft=n_range_fft,
        rd_pipeline=pp.rd_pipeline,
        rd_full_range=pp.rd_full_range,
    )
    if not rd_frames:
        raise RuntimeError(f"No RD frames for {capture}")

    f0 = rd_frames[0]
    n_slow = int(params.get("n_slow", int(params["n_chirps"]) // int(params["n_tx"])))
    print(
        f"  RD pipeline: {pp.rd_pipeline}  grid {f0.shape[0]}×{f0.shape[1]} "
        f"(native {n_slow} Doppler × {int(params['n_samples'])} range)"
        + ("  full range" if pp.rd_full_range and pp.rd_pipeline == "mmw" else ""),
        flush=True,
    )

    stack = np.stack(rd_frames, axis=0)
    vmin, vmax = color_limits(
        stack,
        use_percentile=pp.percentile_color_scale,
        vmin_db=pp.rd_vmin_db,
        vmax_db=pp.rd_vmax_db,
    )
    colorbar_label = "Power (dB)"

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
            upsample=1 if pp.rd_pipeline == "mmw" else pp.rd_display_upsample,
            interpolation="nearest" if pp.rd_pipeline == "mmw" else pp.rd_interpolation,
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
        vmin=vmin,
        vmax=vmax,
        colorbar_label=colorbar_label,
        upsample=1 if pp.rd_pipeline == "mmw" else pp.rd_display_upsample,
        interpolation="nearest" if pp.rd_pipeline == "mmw" else pp.rd_interpolation,
    )
    return out_path
