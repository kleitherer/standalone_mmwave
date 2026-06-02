"""Config3: RD-map peak detection (range + Doppler from antenna-averaged RD power)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from processing.angle_estimate import angle_deg_ula_fft_at_rd_cell, range_angle_to_cartesian_m
from utils.rd_power import peak_candidates_from_rd_power, rd_map_from_frame_int16


@dataclass(frozen=True)
class RdPeakTarget:
    range_m: float
    doppler_mps: float
    snr_db: float
    range_idx: int
    doppler_idx: int
    angle_deg: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    angle_bin: int = 0


@dataclass(frozen=True)
class RdPeakConfig:
    """``gesture.config3`` in config/live_radar_to_max.json."""

    min_range_m: float = 0.0
    max_range_m: float = 3.4
    power_threshold_db: float = 0.0
    max_abs_doppler_mps: float = 3.0
    max_doppler_jump_mps: float = 0.0
    max_abs_angle_deg: float = 30.0
    max_range_jump_m: float = 0.0
    max_candidates: int = 8
    declutter: bool = True
    window: bool = True
    n_angle_fft: int = 128

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> RdPeakConfig:
        gesture = settings.get("gesture", {})
        block = gesture.get("config3", {})
        if not isinstance(block, dict):
            block = {}
        proc = settings.get("processing", {})
        ang = settings.get("angle_estimation", {})
        return cls(
            min_range_m=float(block.get("min_range_m", proc.get("roi_min_m", 0.0))),
            max_range_m=float(block.get("max_range_m", proc.get("roi_max_m", 3.4))),
            power_threshold_db=float(block.get("power_threshold_db", 0.0)),
            max_abs_doppler_mps=float(block.get("max_abs_doppler_mps", 3.0)),
            max_doppler_jump_mps=float(block.get("max_doppler_jump_mps", 0.0)),
            max_abs_angle_deg=float(block.get("max_abs_angle_deg", 30.0)),
            max_range_jump_m=float(block.get("max_range_jump_m", 0.0)),
            max_candidates=max(1, int(block.get("max_candidates", 8))),
            declutter=bool(block.get("declutter", True)),
            window=bool(block.get("window", True)),
            n_angle_fft=max(2, int(block.get("n_angle_fft", ang.get("fft_bins", 128)))),
        )


@dataclass
class RdPeakProcessor:
    """Detect strongest cell on the per-frame RD power map."""

    cfg: RdPeakConfig
    _last_range_m: float | None = None
    _last_doppler_mps: float | None = None
    _last_target: RdPeakTarget | None = None

    def reset(self) -> None:
        self._last_range_m = None
        self._last_doppler_mps = None
        self._last_target = None

    def detect(
        self,
        frame_int16: np.ndarray,
        params: dict[str, Any],
    ) -> RdPeakTarget | None:
        rd_pw, rda, r_axis, d_axis = rd_map_from_frame_int16(
            frame_int16,
            params,
            declutter=self.cfg.declutter,
            window=self.cfg.window,
            max_range_m=self.cfg.max_range_m,
        )
        if rd_pw.size == 0:
            return None

        candidates = peak_candidates_from_rd_power(
            rd_pw,
            r_axis,
            d_axis,
            power_threshold_db=self.cfg.power_threshold_db,
            max_abs_doppler_mps=self.cfg.max_abs_doppler_mps,
            max_candidates=self.cfg.max_candidates,
        )
        if self.cfg.min_range_m > 0.0:
            candidates = [c for c in candidates if float(c[2]) >= float(self.cfg.min_range_m)]
        if not candidates:
            return None

        chosen = candidates[0]
        found_within_jump = True
        has_range_gate = self._last_range_m is not None and self.cfg.max_range_jump_m > 0.0
        has_doppler_gate = (
            self._last_doppler_mps is not None and self.cfg.max_doppler_jump_mps > 0.0
        )
        if has_range_gate or has_doppler_gate:
            jump_lim = float(self.cfg.max_range_jump_m)
            dop_jump_lim = float(self.cfg.max_doppler_jump_mps)
            for cand in candidates:
                _d, _r, cand_range, cand_dopp, _p = cand
                ok_range = (
                    abs(float(cand_range) - float(self._last_range_m)) <= jump_lim
                    if has_range_gate
                    else True
                )
                ok_doppler = (
                    abs(float(cand_dopp) - float(self._last_doppler_mps)) <= dop_jump_lim
                    if has_doppler_gate
                    else True
                )
                if ok_range and ok_doppler:
                    chosen = cand
                    found_within_jump = True
                    break
            if not found_within_jump and self._last_target is not None:
                # Hard continuity for config4-style use: if no candidate satisfies
                # the jump rule, keep previous OSC target instead of jumping.
                return self._last_target
        d_idx, r_idx, range_m, doppler_mps, power_db = chosen
        self._last_range_m = float(range_m)
        self._last_doppler_mps = float(doppler_mps)
        angle_deg, angle_bin = angle_deg_ula_fft_at_rd_cell(
            rda,
            d_idx,
            r_idx,
            n_angle_fft=self.cfg.n_angle_fft,
        )
        if self.cfg.max_abs_angle_deg > 0:
            angle_deg = float(
                np.clip(
                    angle_deg,
                    -float(self.cfg.max_abs_angle_deg),
                    float(self.cfg.max_abs_angle_deg),
                )
            )
        x_m, y_m = range_angle_to_cartesian_m(range_m, angle_deg)

        target = RdPeakTarget(
            range_m=range_m,
            doppler_mps=doppler_mps,
            snr_db=power_db,
            range_idx=r_idx,
            doppler_idx=d_idx,
            angle_deg=angle_deg,
            x_m=x_m,
            y_m=y_m,
            angle_bin=angle_bin,
        )
        self._last_target = target
        return target


@dataclass(frozen=True)
class RdKalmanConfig:
    """``gesture.config5`` in config/live_radar_to_max.json."""

    min_range_m: float = 0.0
    max_range_m: float = 3.4
    power_threshold_db: float = 0.0
    max_abs_doppler_mps: float = 3.0
    max_abs_angle_deg: float = 30.0
    max_candidates: int = 8
    declutter: bool = True
    window: bool = True
    n_angle_fft: int = 128
    process_var_range: float = 0.05
    process_var_doppler: float = 0.08
    process_var_angle: float = 2.0
    meas_var_range: float = 0.12
    meas_var_doppler: float = 0.15
    meas_var_angle: float = 5.0

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "RdKalmanConfig":
        gesture = settings.get("gesture", {})
        block = gesture.get("config5", {})
        if not isinstance(block, dict):
            block = {}
        proc = settings.get("processing", {})
        ang = settings.get("angle_estimation", {})
        return cls(
            min_range_m=float(block.get("min_range_m", proc.get("roi_min_m", 0.0))),
            max_range_m=float(block.get("max_range_m", proc.get("roi_max_m", 3.4))),
            power_threshold_db=float(block.get("power_threshold_db", 0.0)),
            max_abs_doppler_mps=float(block.get("max_abs_doppler_mps", 3.0)),
            max_abs_angle_deg=float(block.get("max_abs_angle_deg", 30.0)),
            max_candidates=max(1, int(block.get("max_candidates", 8))),
            declutter=bool(block.get("declutter", True)),
            window=bool(block.get("window", True)),
            n_angle_fft=max(2, int(block.get("n_angle_fft", ang.get("fft_bins", 128)))),
            process_var_range=float(block.get("process_var_range", 0.05)),
            process_var_doppler=float(block.get("process_var_doppler", 0.08)),
            process_var_angle=float(block.get("process_var_angle", 2.0)),
            meas_var_range=float(block.get("meas_var_range", 0.12)),
            meas_var_doppler=float(block.get("meas_var_doppler", 0.15)),
            meas_var_angle=float(block.get("meas_var_angle", 5.0)),
        )


@dataclass
class RdKalmanProcessor:
    """
    Config5: Kalman smoothing on RD-peak measurements (range, doppler, angle).

    State is ``x = [range_m, doppler_mps, angle_deg]`` with simple constant-velocity
    kinematics for range: ``range += doppler * dt``.
    """

    cfg: RdKalmanConfig
    _x: np.ndarray | None = None
    _P: np.ndarray | None = None
    _last_target: RdPeakTarget | None = None

    def reset(self) -> None:
        self._x = None
        self._P = None
        self._last_target = None

    def _measurement_from_frame(
        self,
        frame_int16: np.ndarray,
        params: dict[str, Any],
    ) -> RdPeakTarget | None:
        rd_pw, rda, r_axis, d_axis = rd_map_from_frame_int16(
            frame_int16,
            params,
            declutter=self.cfg.declutter,
            window=self.cfg.window,
            max_range_m=self.cfg.max_range_m,
        )
        if rd_pw.size == 0:
            return None
        candidates = peak_candidates_from_rd_power(
            rd_pw,
            r_axis,
            d_axis,
            power_threshold_db=self.cfg.power_threshold_db,
            max_abs_doppler_mps=self.cfg.max_abs_doppler_mps,
            max_candidates=self.cfg.max_candidates,
        )
        if self.cfg.min_range_m > 0.0:
            candidates = [c for c in candidates if float(c[2]) >= float(self.cfg.min_range_m)]
        if not candidates:
            return None
        d_idx, r_idx, range_m, doppler_mps, power_db = candidates[0]
        angle_deg, angle_bin = angle_deg_ula_fft_at_rd_cell(
            rda,
            d_idx,
            r_idx,
            n_angle_fft=self.cfg.n_angle_fft,
        )
        if self.cfg.max_abs_angle_deg > 0.0:
            angle_deg = float(
                np.clip(
                    angle_deg,
                    -float(self.cfg.max_abs_angle_deg),
                    float(self.cfg.max_abs_angle_deg),
                )
            )
        x_m, y_m = range_angle_to_cartesian_m(range_m, angle_deg)
        return RdPeakTarget(
            range_m=float(range_m),
            doppler_mps=float(doppler_mps),
            snr_db=float(power_db),
            range_idx=int(r_idx),
            doppler_idx=int(d_idx),
            angle_deg=float(angle_deg),
            x_m=float(x_m),
            y_m=float(y_m),
            angle_bin=int(angle_bin),
        )

    def detect(
        self,
        frame_int16: np.ndarray,
        params: dict[str, Any],
    ) -> RdPeakTarget | None:
        meas = self._measurement_from_frame(frame_int16, params)
        if meas is None:
            return self._last_target

        dt = max(float(params.get("frame_time", 22.22)) / 1000.0, 1e-3)
        A = np.array(
            [
                [1.0, dt, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        H = np.eye(3, dtype=np.float64)
        Q = np.diag(
            [
                float(self.cfg.process_var_range),
                float(self.cfg.process_var_doppler),
                float(self.cfg.process_var_angle),
            ]
        )
        R = np.diag(
            [
                float(self.cfg.meas_var_range),
                float(self.cfg.meas_var_doppler),
                float(self.cfg.meas_var_angle),
            ]
        )

        z = np.array([meas.range_m, meas.doppler_mps, meas.angle_deg], dtype=np.float64)
        if self._x is None or self._P is None:
            self._x = z.copy()
            self._P = np.diag([0.4, 0.4, 8.0]).astype(np.float64)
        else:
            # predict
            x_pred = A @ self._x
            P_pred = A @ self._P @ A.T + Q
            # update
            y = z - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            self._x = x_pred + K @ y
            self._P = (np.eye(3) - K @ H) @ P_pred

        range_f = float(self._x[0])
        doppler_f = float(self._x[1])
        angle_f = float(self._x[2])
        if self.cfg.max_abs_doppler_mps > 0:
            doppler_f = float(
                np.clip(doppler_f, -float(self.cfg.max_abs_doppler_mps), float(self.cfg.max_abs_doppler_mps))
            )
        if self.cfg.max_abs_angle_deg > 0:
            angle_f = float(
                np.clip(angle_f, -float(self.cfg.max_abs_angle_deg), float(self.cfg.max_abs_angle_deg))
            )
        range_f = float(np.clip(range_f, float(self.cfg.min_range_m), float(self.cfg.max_range_m)))
        x_m, y_m = range_angle_to_cartesian_m(range_f, angle_f)
        out = RdPeakTarget(
            range_m=range_f,
            doppler_mps=doppler_f,
            snr_db=meas.snr_db,
            range_idx=meas.range_idx,
            doppler_idx=meas.doppler_idx,
            angle_deg=angle_f,
            x_m=float(x_m),
            y_m=float(y_m),
            angle_bin=meas.angle_bin,
        )
        self._last_target = out
        return out


@dataclass
class Config3OscVolume:
    """Per-frame config3 outputs (same values sent on OSC)."""

    range_m: np.ndarray
    doppler_mps: np.ndarray
    snr_db: np.ndarray
    present: np.ndarray
    angle_deg: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    doppler_idx: np.ndarray
    range_idx: np.ndarray
    time_s: np.ndarray


def process_config3_frames(
    frames: list[np.ndarray],
    params: dict[str, Any],
    processor: RdPeakProcessor,
    *,
    peak_cfg_snr_threshold_db: float,
    frame_time_ms: float,
) -> Config3OscVolume:
    """Run config3 detection on a list of raw int16 frames."""
    n = len(frames)
    range_m = np.full(n, np.nan, dtype=np.float64)
    doppler_mps = np.full(n, np.nan, dtype=np.float64)
    snr_db = np.full(n, np.nan, dtype=np.float64)
    present = np.zeros(n, dtype=bool)
    angle_deg = np.full(n, np.nan, dtype=np.float64)
    x_m = np.full(n, np.nan, dtype=np.float64)
    y_m = np.full(n, np.nan, dtype=np.float64)
    d_idx = np.full(n, -1, dtype=np.int32)
    r_idx = np.full(n, -1, dtype=np.int32)

    for fi, frame in enumerate(frames):
        target = processor.detect(frame, params)
        if target is None:
            continue
        range_m[fi] = target.range_m
        doppler_mps[fi] = target.doppler_mps
        snr_db[fi] = target.snr_db
        angle_deg[fi] = target.angle_deg
        x_m[fi] = target.x_m
        y_m[fi] = target.y_m
        d_idx[fi] = target.doppler_idx
        r_idx[fi] = target.range_idx
        present[fi] = float(target.snr_db) >= float(peak_cfg_snr_threshold_db)

    dt = max(float(frame_time_ms) / 1000.0, 1e-6)
    time_s = np.arange(n, dtype=np.float64) * dt
    return Config3OscVolume(
        range_m=range_m,
        doppler_mps=doppler_mps,
        snr_db=snr_db,
        present=present,
        angle_deg=angle_deg,
        x_m=x_m,
        y_m=y_m,
        doppler_idx=d_idx,
        range_idx=r_idx,
        time_s=time_s,
    )
