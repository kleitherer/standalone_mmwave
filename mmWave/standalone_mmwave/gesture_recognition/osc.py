"""OSC transport and address layout for Max [udpreceive]."""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Any

from processing.osc_utils import build_osc_message


def osc_pad4(data: bytes) -> bytes:
    pad = (4 - (len(data) % 4)) % 4
    return data + (b"\x00" * pad)


class OscSender:
    def __init__(self, host: str, port: int) -> None:
        self._dest = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, address: str, value: float) -> None:
        self._sock.sendto(build_osc_message(address, value), self._dest)

    def send_bundle(self, address: str, *values: float) -> None:
        self._sock.sendto(build_osc_message(address, *values), self._dest)

    def send_text(self, address: str, text: str) -> None:
        self._sock.sendto(build_osc_message(address, text), self._dest)

    def close(self) -> None:
        self._sock.close()


@dataclass(frozen=True)
class OscAddresses:
    """``osc`` block in config/live_radar_to_max.json."""

    mode: str = "/radar/mode"
    range_m: str = "/radar/range_m"
    range2_m: str = "/radar/range2_m"
    snr_db: str = "/radar/snr_db"
    snr2_db: str = "/radar/snr2_db"
    doppler_mps: str = "/radar/doppler_mps"
    angle_deg: str = "/radar/angle_deg"
    x_m: str | None = None
    y_m: str | None = None
    presence: str = "/radar/presence"
    gesture: str = "/radar/gesture"
    bundle: str | None = None
    live_mode_text: str = "Recording Live Data"
    replay_mode_text: str = "Re-running Previous Capture"

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> OscAddresses:
        osc = settings.get("osc", {})
        bundle = osc.get("bundle_address")
        return cls(
            mode=str(osc.get("mode_address", "/radar/mode")),
            range_m=str(osc.get("range_address", "/radar/range_m")),
            range2_m=str(osc.get("range2_address", "/radar/range2_m")),
            snr_db=str(osc.get("snr_address", "/radar/snr_db")),
            snr2_db=str(osc.get("snr2_address", "/radar/snr2_db")),
            doppler_mps=str(osc.get("doppler_address", "/radar/doppler_mps")),
            angle_deg=str(osc.get("angle_address", "/radar/angle_deg")),
            x_m=str(osc["x_address"]) if osc.get("x_address") else None,
            y_m=str(osc["y_address"]) if osc.get("y_address") else None,
            presence=str(osc.get("presence_address", "/radar/presence")),
            gesture=str(osc.get("gesture_address", "/radar/gesture")),
            bundle=str(bundle) if bundle else None,
            live_mode_text=str(osc.get("live_mode_text", "Recording Live Data")),
            replay_mode_text=str(
                osc.get("replay_mode_text", "Re-running Previous Capture")
            ),
        )

    @classmethod
    def from_args(cls, args: Any, settings: dict[str, Any]) -> OscAddresses:
        """CLI ``*_address`` flags override ``osc`` block in settings."""
        base = cls.from_settings(settings)
        bundle = getattr(args, "bundle_address", None) or base.bundle
        return cls(
            mode=getattr(args, "mode_address", base.mode),
            range_m=getattr(args, "range_address", base.range_m),
            range2_m=getattr(args, "range2_address", base.range2_m),
            snr_db=getattr(args, "snr_address", base.snr_db),
            snr2_db=getattr(args, "snr2_address", base.snr2_db),
            doppler_mps=getattr(args, "doppler_address", base.doppler_mps),
            angle_deg=getattr(args, "angle_address", base.angle_deg),
            x_m=base.x_m,
            y_m=base.y_m,
            presence=getattr(args, "presence_address", base.presence),
            gesture=getattr(args, "gesture_address", base.gesture),
            bundle=str(bundle) if bundle else None,
            live_mode_text=getattr(args, "live_mode_text", base.live_mode_text),
            replay_mode_text=getattr(args, "replay_mode_text", base.replay_mode_text),
        )


def make_publisher(args: Any, settings: dict[str, Any]) -> OscPublisher:
    """Build publisher from network + OSC CLI flags."""
    host = getattr(args, "osc_host", settings.get("network", {}).get("osc_host", "127.0.0.1"))
    port = int(getattr(args, "osc_port", settings.get("network", {}).get("osc_port", 9000)))
    addrs = OscAddresses.from_args(args, settings)
    osc_cfg = settings.get("osc", {})
    send_angle = bool(osc_cfg.get("send_angle", False))
    return OscPublisher(
        sender=OscSender(str(host), port),
        addresses=addrs,
        no_snr=bool(getattr(args, "no_snr", False)),
        no_doppler=bool(getattr(args, "no_doppler", False)),
        no_angle=bool(getattr(args, "no_angle", False)) or not send_angle,
        no_gesture=bool(getattr(args, "no_gesture", False)),
        emit_below_threshold=bool(getattr(args, "emit_below_threshold", False)),
    )


@dataclass
class OscPublisher:
    """Send one frame's radar outputs to Max."""

    sender: OscSender
    addresses: OscAddresses
    no_snr: bool = False
    no_doppler: bool = False
    no_angle: bool = False
    no_gesture: bool = False
    emit_below_threshold: bool = False

    def send_mode(self, text: str) -> None:
        self.sender.send_text(self.addresses.mode, text)

    def send_frame(
        self,
        *,
        range_m: float,
        snr_db: float,
        present: bool,
        range2_m: float | None = None,
        snr2_db: float | None = None,
        doppler_mps: float | None = None,
        angle_deg: float | None = None,
        x_m: float | None = None,
        y_m: float | None = None,
        gesture: str | None = None,
    ) -> bool:
        """
        Publish OSC for one frame. Returns True if a full update was sent.

        When ``present`` is false and ``emit_below_threshold`` is false, only
        presence (and gesture=none) are sent.
        """
        from gesture_recognition.gesture import GESTURE_NONE

        if not present and not self.emit_below_threshold:
            self.sender.send(self.addresses.presence, 0.0)
            if not self.no_gesture and self.addresses.gesture:
                self.sender.send_text(self.addresses.gesture, GESTURE_NONE)
            return False

        self.sender.send(self.addresses.range_m, range_m)
        if not self.no_snr:
            self.sender.send(self.addresses.snr_db, snr_db)
        if range2_m is not None:
            self.sender.send(self.addresses.range2_m, range2_m)
            if not self.no_snr and snr2_db is not None:
                self.sender.send(self.addresses.snr2_db, snr2_db)
        if doppler_mps is not None and not self.no_doppler:
            self.sender.send(self.addresses.doppler_mps, doppler_mps)
        if angle_deg is not None and not self.no_angle:
            self.sender.send(self.addresses.angle_deg, angle_deg)
        if x_m is not None and self.addresses.x_m:
            self.sender.send(self.addresses.x_m, x_m)
        if y_m is not None and self.addresses.y_m:
            self.sender.send(self.addresses.y_m, y_m)
        self.sender.send(self.addresses.presence, 1.0 if present else 0.0)
        if gesture is not None and not self.no_gesture and self.addresses.gesture:
            self.sender.send_text(self.addresses.gesture, gesture)

        if self.addresses.bundle:
            r2 = range2_m if range2_m is not None else 0.0
            snr2 = snr2_db if snr2_db is not None else 0.0
            if doppler_mps is not None and angle_deg is not None:
                self.sender.send_bundle(
                    self.addresses.bundle,
                    range_m,
                    doppler_mps,
                    angle_deg,
                    snr_db,
                )
            else:
                self.sender.send_bundle(self.addresses.bundle, range_m, snr_db, r2, snr2)
        return True
