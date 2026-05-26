#!/usr/bin/env python3
"""DCA1000EVM socket client for radar ADC data over UDP."""

import socket
import struct


class DCA1000:
    cmds = {
        "RESET_FPGA_CMD_CODE": b"",
        "RESET_AR_DEV_CMD_CODE": b"",
        "CONFIG_FPGA_GEN_CMD_CODE": b"\x5a\xa5\x03\x00\x06\x00\x01\x02\x01\x02\x03\x1e\xaa\xee",
        "CONFIG_EEPROM_CMD_CODE": b"",
        "RECORD_START_CMD_CODE": b"\x5a\xa5\x05\x00\x00\x00\xaa\xee",
        "RECORD_STOP_CMD_CODE": b"\x5a\xa5\x06\x00\x00\x00\xaa\xee",
        "PLAYBACK_START_CMD_CODE": b"",
        "PLAYBACK_STOP_CMD_CODE": b"",
        "SYSTEM_CONNECT_CMD_CODE": b"\x5a\xa5\x09\x00\x00\x00\xaa\xee",
        "SYSTEM_ERROR_CMD_CODE": b"\x5a\xa5\x0a\x00\x01\x00\xaa\xee",
        "CONFIG_PACKET_DATA_CMD_CODE": b"\x5a\xa5\x0b\x00\x06\x00\xc0\x05\x35\x0c\x00\x00\xaa\xee",
        "CONFIG_DATA_MODE_AR_DEV_CMD_CODE": b"",
        "INIT_FPGA_PLAYBACK_CMD_CODE": b"",
        "READ_FPGA_VERSION_CMD_CODE": b"\x5a\xa5\x0e\x00\x00\x00\xaa\xee",
    }

    def __init__(
        self,
        dca_ip: str = "192.168.33.180",
        dca_cmd_port: int = 4096,
        host_ip: str = "192.168.33.30",
        host_cmd_port: int = 4096,
        host_data_port: int = 4098,
    ):
        self.dca_cmd_addr = (dca_ip, dca_cmd_port)
        self.host_data_port = host_data_port

        self.cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_socket.bind(("0.0.0.0", host_cmd_port))
        self.cmd_socket.settimeout(10)

        if host_data_port:
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.data_socket.bind(("0.0.0.0", host_data_port))
            self.data_socket.setblocking(True)

        self.capturing = False

    def _recv_cmd(self) -> str:
        msg, _ = self.cmd_socket.recvfrom(2048)
        return " ".join(msg.hex()[i : i + 2] for i in range(0, len(msg.hex()), 2))

    def _send_cmd(self, cmd: bytes) -> None:
        self.cmd_socket.sendto(cmd, self.dca_cmd_addr)
        print("DCA1000:/>", " ".join(cmd.hex()[i : i + 2] for i in range(0, len(cmd.hex()), 2)))

    def configure(self) -> None:
        for name in (
            "SYSTEM_CONNECT_CMD_CODE",
            "READ_FPGA_VERSION_CMD_CODE",
            "CONFIG_FPGA_GEN_CMD_CODE",
            "CONFIG_PACKET_DATA_CMD_CODE",
        ):
            self._send_cmd(DCA1000.cmds[name])
            try:
                print(self._recv_cmd())
            except TimeoutError as exc:
                raise TimeoutError(
                    f"DCA1000 did not reply to {name}. "
                    "Check: Ethernet cable to DCA, host IP 192.168.33.30/24, "
                    "ping 192.168.33.180, radar+DCA powered, Mac firewall off for Python."
                ) from exc

    def start_capture(self) -> None:
        if not self.capturing:
            self._send_cmd(DCA1000.cmds["RECORD_START_CMD_CODE"])
            print(self._recv_cmd())
            self.capturing = True

    def stop_capture(self) -> None:
        if self.capturing:
            self._send_cmd(DCA1000.cmds["RECORD_STOP_CMD_CODE"])
            old_timeout = self.cmd_socket.gettimeout()
            try:
                self.cmd_socket.settimeout(1.0)
                print(self._recv_cmd())
            except socket.timeout:
                pass
            finally:
                self.cmd_socket.settimeout(old_timeout)
            self.capturing = False

    def recv_data(self):
        """Return (sequence_number, byte_count, adc_payload_bytes)."""
        msg, _ = self.data_socket.recvfrom(2048)
        seqn, bytec = struct.unpack("<IIxx", msg[:10])
        return seqn, bytec, msg[10:]

    def close(self) -> None:
        self.cmd_socket.close()
        if hasattr(self, "data_socket"):
            self.data_socket.close()
