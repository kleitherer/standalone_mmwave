"""Minimal OSC 1.0 message builder for Max [udpreceive]."""

from __future__ import annotations

import struct


def osc_pad4(data: bytes) -> bytes:
    pad = (4 - (len(data) % 4)) % 4
    return data + (b"\x00" * pad)


def build_osc_message(address: str, *values: float) -> bytes:
    if not address.startswith("/"):
        raise ValueError(f"OSC address must start with '/': {address!r}")
    addr = osc_pad4(address.encode("ascii") + b"\x00")
    if not values:
        types = b",\x00"
    else:
        types = b"," + (b"f" * len(values)) + b"\x00"
    types = osc_pad4(types)
    args = b"".join(struct.pack(">f", float(v)) for v in values)
    return addr + types + args
