#!/usr/bin/env python3
"""
Debug raw per-chirp framing / HSI header for DCA1000 captures (read-only).

The saved frames are a continuous int16 stream sliced into equal byte chunks.
If the LVDS stream carries a per-chirp header (lvdsStreamCfg enableHeader=1),
the real samples-per-chirp is larger than n_rx*n_samples, and decoding at the
assumed stride scrambles the slow-time (Doppler) phase while leaving range
mostly intact.

This tool lets you decode at an arbitrary stride / header and inspect the
result, or run --compare to score candidate header layouts by Doppler
coherence.

Examples
--------
  # Compare skip-first vs skip-last (and per-rx) header layouts:
  python3 -m post_processing.debug_framing --capture walk --time 9 --compare

  # Decode with an explicit framing and render diagnostics:
  python3 -m post_processing.debug_framing --capture walk --time 9 \
      --adc-samples-per-chirp 1024 --stride-complex-per-chirp 1040 \
      --strip-complex-header 16 --strip-position front
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from capture_store import CaptureSession  # noqa: E402


def iq(int16: np.ndarray) -> np.ndarray:
    """DCA 2-lane de-interleave: pairs (0,2)(1,3) -> complex (matches frame_to_adc_cube)."""
    d = int16.astype(np.float32)
    a = np.zeros(d.size // 2, np.complex64)
    a[0::2] = 1j * d[0::4] + d[2::4]
    a[1::2] = 1j * d[1::4] + d[3::4]
    return a


def decode_chirp(seg: np.ndarray, nrx: int, ns: int, header: int, position: str) -> np.ndarray:
    """Extract one chirp's (nrx, ns) ADC block from a `stride`-long complex segment."""
    if position == "front":          # 16-complex header at chirp start
        body = seg[header : header + nrx * ns]
        return body.reshape(nrx, ns)
    if position == "back":           # 16-complex padding/header at chirp end
        body = seg[: nrx * ns]
        return body.reshape(nrx, ns)
    if position == "rx_back":        # (ns + header/nrx) per RX, drop trailing per RX
        per = nrx * ns + header
        extra = header // nrx
        return seg[:per].reshape(nrx, ns + extra)[:, :ns]
    if position == "rx_front":       # drop leading per RX
        extra = header // nrx
        return seg[: nrx * (ns + extra)].reshape(nrx, ns + extra)[:, extra:]
    raise ValueError(position)


def decode_true_frame(stream, f_true, nc, nrx, ns, stride, header, position) -> np.ndarray:
    adc = np.empty((nc, nrx, ns), np.complex64)
    base0 = f_true * nc * stride
    for c in range(nc):
        b = base0 + c * stride
        adc[c] = decode_chirp(stream[b : b + stride], nrx, ns, header, position)
    return adc


def doppler_diag(adc, ntx, ns, vmax):
    """tx0/rx0: declutter slow-time, range+Doppler FFT, best *moving* range bin."""
    n_slow = adc.shape[0] // ntx
    x = adc[0::ntx, 0, :].astype(np.complex64)          # (n_slow, ns)
    x = x - x.mean(axis=0, keepdims=True)
    xr = np.fft.fft(x * np.hanning(ns)[None, :], axis=1)
    rd = np.fft.fftshift(np.fft.fft(xr * np.hanning(n_slow)[:, None], axis=0), axes=0)
    # exclude the 3 central Doppler bins so we pick a MOVING target, not residual DC
    c0 = n_slow // 2
    moving = np.abs(rd) ** 2
    moving[c0 - 1 : c0 + 2, :] = 0.0
    rbin = int(np.argmax(np.sum(moving, axis=0)[2 : ns // 2]) + 2)
    spec = np.abs(rd[:, rbin]) ** 2
    concentration = float(spec.max() / (spec.sum() + 1e-12))      # 1 bin / all bins
    spec_db = 10 * np.log10(spec + 1e-12)
    peak_med_db = float(spec_db.max() - np.median(spec_db))
    return concentration, peak_med_db, rbin, x[:, rbin], rd


def range_sharpness(adc, ns):
    """Fast-time quality: mean |range FFT| over tx0 chirps, peak/median (excl DC).

    Including header samples in the 256-pt window injects a fast-time glitch that
    spreads energy and lowers this ratio, so it discriminates skip-first vs skip-last.
    """
    x = adc[:, 0, :].astype(np.complex64)
    rp = np.mean(np.abs(np.fft.fft(x * np.hanning(ns)[None, :], axis=1)), axis=0)
    return float(rp[4 : ns // 2].max() / (np.median(rp) + 1e-12))


def combined_decluttered_rd(adc, ntx, nrx, ns):
    """12-antenna decluttered RD power (dB): (n_doppler, n_range)."""
    n_slow = adc.shape[0] // ntx
    virt = np.empty((n_slow, ntx * nrx, ns), np.complex64)
    for tx in range(ntx):
        for rx in range(nrx):
            virt[:, tx * nrx + rx, :] = adc[tx::ntx, rx, :]
    virt = virt - virt.mean(axis=0, keepdims=True)
    virt = virt * np.hanning(n_slow)[:, None, None] * np.hanning(ns)[None, None, :]
    V = np.fft.fft(virt, axis=2)
    V = np.fft.fftshift(np.fft.fft(V, axis=0), axes=0)
    return 10 * np.log10(np.mean(np.abs(V) ** 2, axis=1) + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug per-chirp framing / HSI header")
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--frame", type=int, default=None, help="saved-frame index")
    ap.add_argument("--time", type=float, default=None, help="time (s) -> nearest frame")
    ap.add_argument("--adc-samples-per-chirp", type=int, default=None,
                    help="complex ADC samples per chirp (default n_rx*n_samples)")
    ap.add_argument("--stride-complex-per-chirp", type=int, default=None,
                    help="complex samples per chirp on the wire (default = adc samples)")
    ap.add_argument("--strip-complex-header", type=int, default=0,
                    help="complex header/padding samples per chirp")
    ap.add_argument("--strip-position", choices=["front", "back", "rx_front", "rx_back"],
                    default="front")
    ap.add_argument("--compare", action="store_true",
                    help="score candidate header layouts across several frames")
    ap.add_argument("--inspect-header", action="store_true",
                    help="show per-position mean |sample| over a chirp block to locate the header")
    ap.add_argument("--n-frames", type=int, default=6, help="frames to average in --compare")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    cap = args.capture
    if not cap.is_absolute():
        cap = (_ROOT / "captures" / cap) if (_ROOT / "captures" / cap).exists() else cap.resolve()
    session = CaptureSession.open(cap)
    p = dict(session.radar_params())
    nc, nrx, ns, ntx = int(p["n_chirps"]), int(p["n_rx"]), int(p["n_samples"]), int(p["n_tx"])
    vmax = float(p["velocity_max"]); range_res = float(p["range_res"])
    fps = 1000.0 / float(p.get("frame_time", 22.22))
    paths = session.frame_paths()
    out_dir = args.out_dir or (cap / "analysis" / "debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_idx = (args.frame if args.frame is not None
                 else int(round((args.time or 0.0) * fps)))
    adc_per = args.adc_samples_per_chirp or (nrx * ns)
    stride = args.stride_complex_per_chirp or adc_per

    # Build a continuous complex stream covering enough chirps to scan a frame phase.
    need_frames = 10
    lo = max(0, saved_idx)
    blk = [np.load(paths[i]) for i in range(lo, min(len(paths), lo + need_frames))]
    stream = iq(np.concatenate(blk))

    if args.inspect_header:
        # align to a frame boundary, then average |sample| at each position in the
        # 1040-complex chirp block to see whether the 16 header samples lead or trail.
        strd = nrx * ns + 16
        best_k, best_c = 0, -1.0
        for k in range(0, min(nc + 4, stream.size // strd - nc - 1)):
            adc = np.empty((nc, nrx, ns), np.complex64)
            for c in range(nc):
                b = (k + c) * strd
                adc[c] = decode_chirp(stream[b : b + strd], nrx, ns, 16, "front")
            conc, *_ = doppler_diag(adc, ntx, ns, vmax)
            if conc > best_c:
                best_c, best_k = conc, k
        blocks = np.stack([np.abs(stream[(best_k + c) * strd : (best_k + c) * strd + strd])
                           for c in range(nc)])
        prof = blocks.mean(axis=0)
        body_med = np.median(prof[16:1024])
        print(f"frame boundary at chirp offset k={best_k} (conc={best_c:.3f}); stride={strd}")
        print(f"mean |sample| -- first 20 positions : "
              f"{np.array2string(prof[:20], precision=0)}")
        print(f"mean |sample| -- last  20 positions : "
              f"{np.array2string(prof[-20:], precision=0)}")
        print(f"median over body[16:1024]           : {body_med:.0f}")
        lead = prof[:16].mean(); trail = prof[1024:].mean()
        print(f"mean |sample| leading 16  : {lead:.0f}  ({lead/body_med:.1f}x body)")
        print(f"mean |sample| trailing 16 : {trail:.0f}  ({trail/body_med:.1f}x body)")
        if lead > trail * 1.5:
            print("-> header is at the FRONT  => --strip-position front (skip-first-16)")
        elif trail > lead * 1.5:
            print("-> header is at the BACK   => --strip-position back  (skip-last-16)")
        else:
            print("-> no strong magnitude marker; front/back are signal-equivalent")
        return 0

    if args.compare:
        candidates = [
            ("old 1024 (no header)", nrx * ns, 0, "front"),
            ("1040 skip-first-16", nrx * ns + 16, 16, "front"),
            ("1040 skip-last-16", nrx * ns + 16, 16, "back"),
            ("1040 per-rx drop-last", nrx * ns + 16, 16, "rx_back"),
            ("1040 per-rx drop-first", nrx * ns + 16, 16, "rx_front"),
        ]
        # Frame-boundary phase is unknown, so scan the starting chirp offset and
        # keep the best (= the true radar-frame boundary). Doppler conc confirms the
        # STRIDE; range sharpness at that offset picks the HEADER POSITION.
        k_max = min(nc + 4, (stream.size // (nrx * ns + 16)) - nc - 1)
        print(f"capture={cap.name}  saved_idx={saved_idx}  scanning {k_max} chirp offsets, "
              f"nc={nc} per window")
        print(f"{'layout':26s} {'bestDopConc':>11} {'pk-med dB':>9} "
              f"{'rngSharp':>9} {'rbin(m)':>8} {'k*':>4}  frame_bytes")
        rows = []
        for name, strd, hdr, pos in candidates:
            best = (-1.0, 0.0, 0.0, 0, 0)
            for k in range(0, k_max):
                base = k * strd
                if base + nc * strd > stream.size:
                    break
                adc = np.empty((nc, nrx, ns), np.complex64)
                for c in range(nc):
                    b = base + c * strd
                    adc[c] = decode_chirp(stream[b : b + strd], nrx, ns, hdr, pos)
                conc, pm, rb, _, _ = doppler_diag(adc, ntx, ns, vmax)
                if conc > best[0]:
                    best = (conc, pm, range_sharpness(adc, ns), rb, k)
            conc, pm, rsh, rb, kbest = best
            fbytes = nc * strd * 2 * 2
            print(f"{name:26s} {conc:11.3f} {pm:9.1f} {rsh:9.1f} "
                  f"{rb * range_res:8.2f} {kbest:4d}  {fbytes}")
            rows.append((conc, rsh, name))
        # decision: stride by Doppler conc; among the top-conc stride, header by sharpness
        top_conc = max(r[0] for r in rows)
        stride_winners = [r for r in rows if r[0] >= top_conc - 0.02]
        hdr_winner = max(stride_winners, key=lambda r: r[1])
        print(f"\nstride decided by Doppler coherence; header decided by range sharpness")
        print(f"BEST: {hdr_winner[2]}  (conc={hdr_winner[0]:.3f}, rngSharp={hdr_winner[1]:.1f})")
        return 0
    # single decode + render: auto-align to the radar-frame boundary (best Doppler conc).
    print(f"decode: adc/chirp={adc_per} stride={stride} header={args.strip_complex_header} "
          f"pos={args.strip_position}")
    best_k, best_conc, adc = 0, -1.0, None
    for k in range(0, min(nc + 4, stream.size // stride - nc - 1)):
        cand = np.empty((nc, nrx, ns), np.complex64)
        for c in range(nc):
            b = (k + c) * stride
            cand[c] = decode_chirp(stream[b : b + stride], nrx, ns,
                                   args.strip_complex_header, args.strip_position)
        cc, *_ = doppler_diag(cand, ntx, ns, vmax)
        if cc > best_conc:
            best_conc, best_k, adc = cc, k, cand
    print(f"  aligned at chirp offset k={best_k}")
    conc, pm, rb, st, rd = doppler_diag(adc, ntx, ns, vmax)
    rd_comb = combined_decluttered_rd(adc, ntx, nrx, ns)
    print(f"  Doppler concentration={conc:.3f}  peak-median={pm:.1f} dB  "
          f"best moving range bin {rb} = {rb * range_res:.2f} m")
    # point 5: persistent 1.42 m / 0 m/s artifact level
    rbin_142 = int(round(1.42 / range_res))
    c0 = (nc // ntx) // 2
    print(f"  power @1.42 m, 0 m/s (decluttered): {rd_comb[c0, rbin_142]:.1f} dB "
          f"(map max {rd_comb.max():.1f} dB)")
    print(f"  implied frame: {nc} chirps * {nrx} rx * {ns} samp + header "
          f"-> {nc * stride} complex/frame = {nc * stride * 2 * 2} bytes")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d_axis = (np.arange(nc // ntx) - (nc // ntx) // 2) * (2 * vmax / (nc // ntx))
    r_axis = np.arange(ns) * range_res
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    ax[0].plot(st.real, "o-", ms=3, label="I"); ax[0].plot(st.imag, "o-", ms=3, label="Q")
    ax[0].set_title("slow-time phasor (tx0/rx0)"); ax[0].set_xlabel("chirp"); ax[0].legend()
    ax[1].plot(d_axis, 10 * np.log10(np.abs(rd[:, rb]) ** 2 + 1e-9), "o-", ms=3)
    ax[1].set_title(f"Doppler spectrum @ {rb * range_res:.2f} m (pk-med {pm:.1f} dB)")
    ax[1].set_xlabel("Doppler (m/s)")
    im = ax[2].imshow(rd_comb, aspect="auto", cmap="jet", origin="lower",
                      extent=[r_axis[0], r_axis[-1], d_axis[0], d_axis[-1]])
    ax[2].set_title("combined decluttered RD"); ax[2].set_xlabel("range (m)"); ax[2].set_ylabel("Doppler (m/s)")
    fig.colorbar(im, ax=ax[2])
    fig.suptitle(f"{cap.name} f{saved_idx}  stride={stride} hdr={args.strip_complex_header} pos={args.strip_position}")
    fig.tight_layout()
    out = out_dir / f"framing_{args.strip_position}_s{stride}_h{args.strip_complex_header}.png"
    fig.savefig(out, dpi=120)
    print("wrote", out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
