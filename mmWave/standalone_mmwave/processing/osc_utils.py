"""Minimal OSC 1.0 message builder for Max [udpreceive]."""

from __future__ import annotations

import struct


def osc_pad4(data: bytes) -> bytes:
    pad = (4 - (len(data) % 4)) % 4
    return data + (b"\x00" * pad)


def _osc_string_arg(s: str) -> bytes:
    return osc_pad4(s.encode("utf-8") + b"\x00")


def build_osc_message(address: str, *values: float | str) -> bytes:
    """Build OSC message with float (f) and/or string (s) arguments."""
    if not address.startswith("/"):
        raise ValueError(f"OSC address must start with '/': {address!r}")
    addr = osc_pad4(address.encode("ascii") + b"\x00")
    if not values:
        types = b",\x00"
        args = b""
    else:
        type_chars = []
        args = b""
        for v in values:
            if isinstance(v, str):
                type_chars.append("s")
                args += _osc_string_arg(v)
            else:
                type_chars.append("f")
                args += struct.pack(">f", float(v))
        types = osc_pad4(b"," + "".join(type_chars).encode("ascii") + b"\x00")
    return addr + types + args
