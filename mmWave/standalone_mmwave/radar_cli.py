#!/usr/bin/env python3
"""UART control for TI mmWave EVM (sensorStart / sensorStop / cfg upload)."""

import serial


def list_serial_ports() -> list[str]:
    """macOS/Linux cu.* ports (call before opening RadarCLI)."""
    from pathlib import Path

    ports = sorted(Path("/dev").glob("cu.*"))
    return [str(p) for p in ports if "Bluetooth" not in p.name]


class RadarCLI:
    cmds = ["sensorStop", "flushCfg", "sensorStart"]

    def __init__(self, cmd_tty: str = "/dev/ttyUSB0", verbose: bool = False):
        self.verbose = verbose
        if "XXXX" in cmd_tty or cmd_tty in ("/dev/ttyUSB0", "/dev/ttyUSB1"):
            hint = list_serial_ports()
            raise SystemExit(
                f"Invalid or placeholder serial port: {cmd_tty!r}\n"
                "Plug USB from the mmWave board into the Mac, then run:\n"
                "  ls /dev/cu.*\n"
                "  python3 radar_receiver.py --list-ports\n"
                "Use the cu.* device (not tty.*), e.g. --cmd-tty /dev/cu.usbserial-1410\n"
                f"Ports visible now: {hint or '(none — is mmWave USB connected?)'}"
            )
        self.cmd_serial = serial.Serial(
            port=cmd_tty,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=2.0,
        )
        self.started = False
        if self.verbose:
            print(f"[UART] Opened {cmd_tty} @ 115200")

    def _recv(self, wait_sec: float = 0.3) -> str:
        """Read CLI response (mmWave replies can take 100–500 ms)."""
        self.cmd_serial.timeout = wait_sec
        data = self.cmd_serial.read(size=32768)
        return data.decode(errors="replace")

    def _send(self, cmd: str) -> None:
        self.cmd_serial.reset_input_buffer()
        if self.verbose:
            print(f"[UART] >> {cmd}")
        self.cmd_serial.write(cmd.encode("utf-8"))
        self.cmd_serial.write("\r".encode())

    def configure(self, cfg) -> None:
        for cmd in RadarCLI.cmds[:2]:
            self._send(cmd)
            response = self._recv()
            if response.strip():
                print(response)
            elif self.verbose:
                print(f"[UART] (no response to {cmd})")

        for cmd in cfg.to_cfg():
            self._send(cmd)
            response = self._recv()
            if response.strip():
                print(response)

    def start(self) -> None:
        if not self.started:
            self._send(RadarCLI.cmds[2])
            response = self._recv(wait_sec=1.0)
            if response.strip():
                print(response)
            elif self.verbose:
                print("[UART] (no response to sensorStart — wrong port?)")
            self.started = True

    def stop(self) -> None:
        if self.started:
            self._send(RadarCLI.cmds[0])
            self._recv()
            self.started = False

    def close(self) -> None:
        self.cmd_serial.close()
