"""LVDS / DCA1000 raw ADC byte-order helpers (from radar_capture_utils)."""

import numpy as np


def fix_byte_order(data: np.ndarray) -> np.ndarray:
    """
    Reorder uint16 words for 2-lane LVDS capture.

    Lane order on wire: real(s0), real(s1), imag(s0), imag(s1) per pair of samples.
    """
    if data.dtype != np.uint16:
        buf = np.asarray(data, dtype=np.int16).view(np.uint16)
    else:
        buf = np.asarray(data, dtype=np.uint16)

    tmp = np.empty_like(buf)
    tmp[0::4] = buf[0::4]
    tmp[1:-1:4] = buf[2::4]
    tmp[2::4] = buf[1::4]
    tmp[3::4] = buf[3::4]
    return tmp
