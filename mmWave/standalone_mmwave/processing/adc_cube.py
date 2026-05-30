"""Raw ADC cube (chirps × RX × samples) before virtual-antenna combine."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def hard_limiter(adc: np.ndarray) -> np.ndarray:
    """
    1-bit hard limiter on I and Q: ``sign(real) + 1j*sign(imag)``.

    Display-only enhancement for single-target range tracking. It turns each
    beat sinusoid into a square wave at the same frequency, so the strongest
    target's range bin is hugely boosted relative to clutter/noise — giving a
    very clean range–time track. It is NOT physically faithful (true amplitudes,
    Doppler magnitude, phase and weak/multiple targets are lost), so never use it
    for RD maps, Doppler, or angle estimation. This reproduces, on correctly
    decoded data, the high-contrast track that the old buggy ``uint16`` decode
    produced as a side effect of sign-slicing.
    """
    return (np.sign(adc.real) + 1j * np.sign(adc.imag)).astype(np.complex64)


def frame_to_adc_cube(
    frame: np.ndarray, params: Dict[str, Any], *, limiter: bool = False
) -> np.ndarray:
    """
    De-interleave one capture frame to complex ADC cube.

    Decoding matches the trusted raw-DCA path (multimodal-ros
    ``sensors/dsp.py::_reshape_frame`` and the C++ ``recver.cpp``): the DCA1000
    UDP payload is **signed int16** in 2-lane LVDS order and is consumed with no
    word reorder. ``fix_byte_order`` is only for the mmw HSI-header Ethernet
    stream, NOT these header-less FrameBuffer captures — applying it here (and
    reading the samples as ``uint16``) corrupts the IQ.

    Parameters
    ----------
    limiter : bool
        Apply :func:`hard_limiter` after decoding (display-only; see its
        docstring). Default ``False`` — keep off for RD/Doppler/angle.

    Returns
    -------
    adc : complex64, shape ``(n_chirps, n_rx, n_samples)``
        Chirp order matches DCA stream (TDM interleaved: TX0, TX1, TX2, …).
    """
    data = np.asarray(frame, dtype=np.int16).ravel()
    n_chirps = int(params["n_chirps"])
    n_rx = int(params["n_rx"])
    n_samples = int(params["n_samples"])
    adc_output_fmt = int(params.get("adc_output_fmt", 1))

    if adc_output_fmt <= 0:
        adc = data.reshape(n_chirps, n_rx, n_samples).astype(np.complex64)
        return hard_limiter(adc) if limiter else adc

    sig = data.astype(np.float32)
    iq = np.zeros(sig.size // 2, dtype=np.complex64)
    iq[0::2] = 1j * sig[0::4] + sig[2::4]
    iq[1::2] = 1j * sig[1::4] + sig[3::4]
    adc = iq.reshape(n_chirps, n_rx, n_samples).astype(np.complex64, copy=False)
    return hard_limiter(adc) if limiter else adc


def slow_time_cube(adc: np.ndarray, n_tx: int) -> np.ndarray:
    """
    TDM slow-time cube for one logical RX column across TX.

    ``adc[tx::n_tx, rx, :]`` → shape ``(n_chirps // n_tx, n_samples)``.
    """
    return adc[0::n_tx, :, :]
