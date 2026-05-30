#!/usr/bin/env python3
"""Parse TI mmWave .cfg files and derive frame size / chirp parameters."""

from collections import OrderedDict


class RadarConfig(OrderedDict):
    headers = ["Created for SDK", "Platform"]

    cmds = [
        "dfeDataOutputMode",
        "channelCfg",
        "adcCfg",
        "adcbufCfg",
        "lowPower",
        "frameCfg",
        "guiMonitor",
        "multiObjBeamForming",
        "calibDcRangeSig",
        "clutterRemoval",
        "aoaFovCfg",
        "compRangeBiasAndRxChanPhase",
        "measureRangeBiasAndRxChanPhase",
        "extendedMaxVelocity",
        "bpmCfg",
        "analogMonitor",
        "lvdsStreamCfg",
        "calibData",
    ]

    multi_cmds = [
        "profileCfg",
        "chirpCfg",
        "cfarCfg",
        "cfarFovCfg",
        "CQRxSatMonitor",
        "CQSigImgMonitor",
    ]

    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, list):
            self.from_cfg(cfg)
        elif isinstance(cfg, dict):
            for key, value in cfg.items():
                self[key] = value

    def from_cfg(self, cfg):
        for line in cfg:
            for hdr in RadarConfig.headers:
                if hdr in line:
                    self[hdr] = line.split(":")[1].strip()
                    break

            for cmd in RadarConfig.cmds:
                if cmd in line:
                    params = [
                        float(x) if "." in x else int(x) for x in line.split()[1:]
                    ]
                    self[cmd] = params
                    break

            for cmd in RadarConfig.multi_cmds:
                if cmd in line:
                    params = [
                        float(x) if "." in x else int(x) for x in line.split()[1:]
                    ]
                    if cmd not in self:
                        self[cmd] = [params]
                    else:
                        self[cmd].append(params)
                    break

    def to_cfg(self):
        out = []
        for cmd, params in self.items():
            if isinstance(params[0], list):
                for param in params:
                    out.append(
                        " ".join(
                            [cmd]
                            + [
                                f"{x:.2f}" if isinstance(x, float) else f"{x}"
                                for x in param
                            ]
                        )
                    )
            else:
                out.append(
                    " ".join(
                        [cmd]
                        + [
                            f"{x:.2f}" if isinstance(x, float) else f"{x}"
                            for x in params
                        ]
                    )
                )
        return out

    def get_params(self):
        adc_output_fmt = int(self["adcCfg"][1])

        n_samples = int(self["profileCfg"][0][9])

        rx_bin = bin(int(self["channelCfg"][0]))[2:]
        rx = [int(x) for x in reversed(rx_bin)]

        tx_bin = bin(int(self["channelCfg"][1]))[2:]
        tx = [int(x) for x in reversed(tx_bin)]

        n_chirps = (
            int(self["frameCfg"][1]) - int(self["frameCfg"][0]) + 1
        ) * self["frameCfg"][2]

        n_tx = sum(tx)
        n_rx = sum(rx)

        frame_size = n_samples * n_rx * n_chirps * 2 * (2 if adc_output_fmt > 0 else 1)
        frame_time = self["frameCfg"][4]

        range_bias = self["compRangeBiasAndRxChanPhase"][0]
        rx_phase_bias = self["compRangeBiasAndRxChanPhase"][1:]

        operating_freq = self["profileCfg"][0][1]
        chirp_time_us = self["profileCfg"][0][2] + self["profileCfg"][0][4]
        n_slow = n_chirps // n_tx
        # TDM: consecutive chirps from the same TX are n_tx chirp periods apart.
        chirp_period_same_tx_s = n_tx * chirp_time_us * 1e-6
        wavelength = 3e8 / (operating_freq * 1e9)
        velocity_max = wavelength / (4.0 * chirp_period_same_tx_s)
        velocity_res = (2.0 * velocity_max) / float(n_slow)

        chirp_slope = self["profileCfg"][0][7] * 1e12
        sample_rate = self["profileCfg"][0][10] * 1e3
        range_max = (sample_rate * 3e8) / (2 * chirp_slope)
        range_res = range_max / n_samples

        return OrderedDict(
            [
                ("sdk", self["Created for SDK"]),
                ("platform", self["Platform"]),
                ("adc_output_fmt", adc_output_fmt),
                ("range_bias", range_bias),
                ("rx_phase_bias", rx_phase_bias),
                ("n_chirps", n_chirps),
                ("n_slow", n_slow),
                ("rx", rx),
                ("n_rx", n_rx),
                ("tx", tx),
                ("n_tx", n_tx),
                ("n_samples", n_samples),
                ("frame_size", frame_size),
                ("frame_time", frame_time),
                ("operating_freq_ghz", operating_freq),
                ("chirp_time_us", chirp_time_us),
                ("chirp_time", chirp_time_us),
                ("chirp_period_same_tx_s", chirp_period_same_tx_s),
                ("chirp_slope", chirp_slope),
                ("sample_rate", sample_rate),
                ("velocity_max", velocity_max),
                ("velocity_res", velocity_res),
                ("range_max", range_max),
                ("range_res", range_res),
            ]
        )
