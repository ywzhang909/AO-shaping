"""Micro DM (R50Power) driver for 50-channel deformable mirror.

This module provides a Python driver for the R50Power 50-channel micro deformable mirror
using TCP socket communication. Based on the MATLAB R50Power class protocol.

Reference:
    - docs/micro deformable mirror/R50Power.m
    - docs/micro deformable mirror/Demo.m

Example:
    >>> with MicroDM() as dm:
    ...     dm.open()
    ...     dm.set_relay_state(True)
    ...     voltages = np.zeros(50)
    ...     dm.set_all_voltage_by_arr(voltages)
    ...     dm.set_channel_voltage(0, 2.5)
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from loguru import logger

from ao_shaping.drivers.device_base import Device, DeviceState, DeviceType
from ao_shaping.drivers.dm.base import DM


class MicroDMError(Exception):
    """Base exception for MicroDM errors."""

    pass


class MicroDMConnectionError(MicroDMError):
    """Raised when connection to MicroDM fails."""

    pass


class MicroDMVoltageError(MicroDMError):
    """Raised when voltage value is out of range."""

    pass


class RelayState(Enum):
    """Relay state enumeration."""

    OFF = auto()
    ON = auto()


# Protocol constants
HEADER_START = bytes([0xAA, 0xBB])
FOOTER_END = bytes([0xCC, 0xDD])

# Command codes
CMD_SET_CHANNEL_VOLTAGE = 0x04
CMD_SET_ALL_CHANNEL_VOLTAGE = 0x08
CMD_SET_ALL_VOLTAGE_BY_ARR = 0x09
CMD_SET_RELAY_ON = 0x06
CMD_SET_RELAY_OFF = 0x07
CMD_SET_IP = 0x06


@dataclass
class MicroDMMetadata:
    """MicroDM device metadata."""

    device_ip: str = "192.168.0.101"
    device_port: int = 10101
    channel_count: int = 50
    voltage_min: float = -1.0
    voltage_max: float = 6.5
    firmware_version: str = ""
    serial_number: str = ""


class MicroDM(DM):
    """Micro DM (R50Power) 50-channel deformable mirror driver.

    Attributes:
        DM_Num: Number of actuators (50).
        V_Min: Minimum voltage (-1.0V).
        V_Max: Maximum voltage (6.5V).

    Example:
        >>> dm = MicroDM(ip="192.168.0.101", port=10101)
        >>> dm.open()
        >>> dm.set_relay_state(True)
        >>> dm.set_all_voltage_by_arr(np.zeros(50))
        >>> dm.close()
    """

    # Class-level constants
    DM_Num: int = 50
    V_Min: float = -1.0
    V_Max: float = 6.5

    # Device type classification
    device_type = DeviceType.DM
    manufacturer = "R50Power"
    model = "MicroDM-50"

    def __init__(
        self,
        ip: str = "192.168.0.101",
        port: int = 10101,
        timeout: float = 10.0,
        device_id: str = "",
    ):
        """Initialize MicroDM driver.

        Args:
            ip: Device IP address. Default: 192.168.0.101
            port: Device TCP port. Default: 10101
            timeout: Socket timeout in seconds. Default: 10.0
            device_id: Unique device identifier. If empty, auto-generated.
        """
        super().__init__(device_id)

        self._ip = ip
        self._port = port
        self._timeout = timeout

        # Socket connection
        self._sock: socket.socket | None = None

        # Internal state tracking
        self._last_voltages: np.ndarray = np.zeros(self.DM_Num)
        self._relay_state = RelayState.OFF

        # Metadata
        self._metadata = MicroDMMetadata(
            device_ip=ip,
            device_port=port,
            channel_count=self.DM_Num,
            voltage_min=self.V_Min,
            voltage_max=self.V_Max,
        )

        # Register device parameters
        self._register_parameters()

        logger.debug(f"MicroDM ({ip}:{port}) initialized")

    def _register_parameters(self) -> None:
        """Register device parameters for parameter management system."""
        self.register_parameter(
            "voltage_min",
            self.V_Min,
            description="Minimum allowed voltage (V)",
        )
        self.register_parameter(
            "voltage_max",
            self.V_Max,
            description="Maximum allowed voltage (V)",
        )
        self.register_parameter(
            "channel_count",
            self.DM_Num,
            description="Number of DM channels",
        )

    # ==================== Device Base Interface ====================

    def open(self) -> None:
        """Open TCP connection to MicroDM.

        Raises:
            MicroDMConnectionError: If connection fails.
        """
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(self._timeout)
            self._sock.connect((self._ip, self._port))
            self._set_state(DeviceState.READY)
            logger.info(f"MicroDM connected to {self._ip}:{self._port}")
        except socket.error as e:
            self._set_state(DeviceState.ERROR, str(e))
            raise MicroDMConnectionError(f"Failed to connect to {self._ip}:{self._port}") from e

    def close(self) -> None:
        """Close TCP connection and release resources."""
        if self._sock is not None:
            try:
                # Optionally reset all voltages before closing
                self.set_all_channel_voltage(0.0)
                self._sock.close()
            except Exception as e:
                logger.warning(f"Error closing MicroDM connection: {e}")
            finally:
                self._sock = None
                self._set_state(DeviceState.DISCONNECTED)
                logger.info("MicroDM connection closed")

    def is_connected(self) -> bool:
        """Check if device is connected and ready.

        Returns:
            True if connected, False otherwise.
        """
        return self._sock is not None and self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """Get hardware-specific information.

        Returns:
            Dictionary containing device info.
        """
        return {
            "manufacturer": self.manufacturer,
            "model": self.model,
            "ip_address": self._ip,
            "port": self._port,
            "channel_count": self.DM_Num,
            "voltage_range": [self.V_Min, self.V_Max],
            "relay_state": self._relay_state.name,
            "firmware_version": self._metadata.firmware_version,
            "serial_number": self._metadata.serial_number,
        }

    def get_actuator_positions(self) -> np.ndarray:
        """Get current actuator voltages.

        Returns:
            Array of current voltages for all 50 channels.
        """
        return self._last_voltages.copy()

    # ==================== DM Base Interface ====================

    def transform(self, cmd: np.ndarray) -> np.ndarray:
        """Transform normalized command to voltage range.

        Args:
            cmd: Normalized command array (-1 to 1).

        Returns:
            Voltage array in device range (V_Min to V_Max).
        """
        cmd = np.clip(cmd, -1, 1)
        return (cmd + 1) * (self.V_Max - self.V_Min) / 2 + self.V_Min

    def send(self, cmd: np.ndarray | float) -> np.ndarray:
        """Send command to DM.

        Args:
            cmd: Voltage array or single voltage value.

        Returns:
            Applied voltages array.

        Raises:
            MicroDMVoltageError: If voltage values are out of range.
        """
        if isinstance(cmd, np.ndarray):
            return self.send_voltages(cmd)
        elif isinstance(cmd, (int, float)):
            return self.set_all_channel_voltage(float(cmd))
        raise MicroDMVoltageError(f"Unsupported command type: {type(cmd)}")

    def send_voltages(self, vs: np.ndarray, wait_time_s: float = 0.001) -> np.ndarray:
        """Send voltage array to all channels.

        Args:
            vs: Voltage array for all 50 channels.
            wait_time_s: Wait time after sending.

        Returns:
            Applied voltages array.

        Raises:
            MicroDMVoltageError: If voltage array length is incorrect or values out of range.
        """
        vs = np.array(vs, dtype=np.float64)
        if vs.shape != (self.DM_Num,):
            raise MicroDMVoltageError(
                f"Expected {self.DM_Num} channels, got {vs.shape[0]}"
            )

        vs = np.clip(vs, self.V_Min, self.V_Max)
        self.set_all_voltage_by_arr(vs)
        self._last_voltages = vs
        time.sleep(wait_time_s)
        return self._last_voltages

    # ==================== Voltage Conversion ====================

    def _voltage_to_bytes(self, voltage: float) -> tuple[int, int]:
        """Convert voltage to high and low bytes.

        Based on MATLAB formula:
            value = (voltage + 1) / 20 / 3.4 / 3.3 * 65535.0
            highByte = floor(value / 255)
            lowByte = floor(mod(value, 255))

        Args:
            voltage: Voltage value in volts.

        Returns:
            Tuple of (high_byte, low_byte).

        Note:
            For SetAllVoltageByArr, voltage is offset by +20 before conversion.
        """
        voltage = np.clip(voltage, self.V_Min, self.V_Max)
        value = (voltage + 1) / 20 / 3.4 / 3.3 * 65535.0
        high_byte = int(value // 255)
        low_byte = int(value % 255)
        return high_byte, low_byte

    def _voltage_to_bytes_offset(self, voltage: float) -> tuple[int, int]:
        """Convert voltage to bytes with +20V offset (for SetAllVoltageByArr).

        Args:
            voltage: Voltage value in volts.

        Returns:
            Tuple of (high_byte, low_byte).
        """
        voltage = np.clip(voltage, self.V_Min, self.V_Max)
        value = (voltage + 20) / 20 / 3.4 / 3.3 * 65535.0
        high_byte = int(value // 255)
        low_byte = int(value % 255)
        return high_byte, low_byte

    # ==================== Protocol Commands ====================

    def _send_command(self, command: bytes) -> None:
        """Send raw command to device.

        Args:
            command: Complete command bytes (header + data + footer).

        Raises:
            MicroDMConnectionError: If send fails.
        """
        if self._sock is None:
            raise MicroDMConnectionError("Not connected to MicroDM")

        try:
            self._sock.sendall(command)
            logger.debug(f"Sent command: {command.hex()}")
        except socket.error as e:
            raise MicroDMConnectionError(f"Failed to send command: {e}") from e

    def set_channel_voltage(self, channel: int, voltage: float) -> None:
        """Set voltage for a single channel.

        Args:
            channel: Channel number (0-49).
            voltage: Voltage value in volts.

        Raises:
            MicroDMVoltageError: If channel is out of range or voltage is invalid.
        """
        if not 0 <= channel < self.DM_Num:
            raise MicroDMVoltageError(f"Channel must be 0-{self.DM_Num - 1}, got {channel}")

        voltage = float(voltage)
        if not self.V_Min <= voltage <= self.V_Max:
            raise MicroDMVoltageError(
                f"Voltage must be {self.V_Min}-{self.V_Max}V, got {voltage}"
            )

        hv, lv = self._voltage_to_bytes(voltage)
        command = HEADER_START + bytes([CMD_SET_CHANNEL_VOLTAGE, channel, hv, lv]) + FOOTER_END
        self._send_command(command)
        self._last_voltages[channel] = voltage
        logger.debug(f"Set channel {channel} to {voltage}V")

    def set_all_voltage_by_arr(self, voltage_arr: np.ndarray) -> None:
        """Set all channels using voltage array.

        Note: Uses +20V offset in conversion per R50Power protocol.

        Args:
            voltage_arr: Array of 50 voltage values.

        Raises:
            MicroDMVoltageError: If array length is not 50.
        """
        if len(voltage_arr) != self.DM_Num:
            raise MicroDMVoltageError(
                f"Expected {self.DM_Num} voltages, got {len(voltage_arr)}"
            )

        command = HEADER_START + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR])
        for voltage in voltage_arr:
            hv, lv = self._voltage_to_bytes_offset(voltage)
            command += bytes([hv, lv])
        command += FOOTER_END

        self._send_command(command)
        self._last_voltages = np.array(voltage_arr)
        logger.debug(f"Set all voltages by array (shape: {voltage_arr.shape})")

    def set_all_channel_voltage(self, voltage: float) -> np.ndarray:
        """Set all channels to the same voltage.

        Args:
            voltage: Voltage value in volts for all channels.

        Returns:
            Array of applied voltages (50 channels, all same value).

        Raises:
            MicroDMVoltageError: If voltage is out of range.
        """
        voltage = float(voltage)
        if not self.V_Min <= voltage <= self.V_Max:
            raise MicroDMVoltageError(
                f"Voltage must be {self.V_Min}-{self.V_Max}V, got {voltage}"
            )

        hv, lv = self._voltage_to_bytes(voltage)
        command = HEADER_START + bytes([CMD_SET_ALL_CHANNEL_VOLTAGE, hv, lv]) + FOOTER_END
        self._send_command(command)

        vs = np.full(self.DM_Num, voltage)
        self._last_voltages = vs
        logger.debug(f"Set all channels to {voltage}V")
        return self._last_voltages

    def set_relay_state(self, state: bool) -> None:
        """Set relay state (open/close).

        Args:
            state: True to open relay, False to close.
        """
        if state:
            command = HEADER_START + bytes([CMD_SET_RELAY_ON]) + FOOTER_END
            self._relay_state = RelayState.ON
        else:
            command = HEADER_START + bytes([CMD_SET_RELAY_OFF]) + FOOTER_END
            self._relay_state = RelayState.OFF

        self._send_command(command)
        logger.info(f"Relay {'opened' if state else 'closed'}")

    def set_ip_address(self, ip_address: str) -> None:
        """Set device IP address.

        Args:
            ip_address: IP address string (e.g., "192.168.0.101").

        Raises:
            MicroDMConnectionError: If IP format is invalid.
        """
        try:
            ip_parts = [int(x) for x in ip_address.split(".")]
            if len(ip_parts) != 4:
                raise ValueError("IP must have 4 parts")
        except (ValueError, AttributeError) as e:
            raise MicroDMConnectionError(f"Invalid IP address format: {ip_address}") from e

        command = HEADER_START + bytes([CMD_SET_IP] + ip_parts) + FOOTER_END
        self._send_command(command)
        logger.info(f"Set device IP to {ip_address}")

    # ==================== Utility Methods ====================

    def reset_all(self) -> None:
        """Reset all channels to 0V."""
        self.set_all_channel_voltage(0.0)
        logger.info("MicroDM reset to 0V")

    def __repr__(self) -> str:
        return (
            f"MicroDM("
            f"ip={self._ip}, "
            f"channels={self.DM_Num}, "
            f"voltage_range=[{self.V_Min}, {self.V_Max}], "
            f"state={self._state.name}"
            f")"
        )


class SimMicroDM(DM):
    """Simulation mode for MicroDM without hardware.

    Use this class for testing and development without actual hardware.

    Attributes:
        DM_Num: Number of actuators (50).
        V_Min: Minimum voltage (-1.0V).
        V_Max: Maximum voltage (6.5V).
    """

    DM_Num: int = 50
    V_Min: float = -1.0
    V_Max: float = 6.5

    device_type = DeviceType.DM
    manufacturer = "R50Power"
    model = "MicroDM-50-Sim"

    def __init__(self, device_id: str = ""):
        """Initialize simulated MicroDM.

        Args:
            device_id: Unique device identifier.
        """
        super().__init__(device_id)

        self._last_voltages: np.ndarray = np.zeros(self.DM_Num)
        self._relay_state = RelayState.OFF

        self._register_parameters()
        logger.debug("SimMicroDM initialized")

    def _register_parameters(self) -> None:
        """Register device parameters."""
        self.register_parameter("voltage_min", self.V_Min, description="Minimum voltage (V)")
        self.register_parameter("voltage_max", self.V_Max, description="Maximum voltage (V)")
        self.register_parameter("channel_count", self.DM_Num, description="Number of channels")

    def open(self) -> None:
        """Open simulated connection."""
        self._set_state(DeviceState.READY)
        logger.info("SimMicroDM connection opened")

    def close(self) -> None:
        """Close simulated connection."""
        self._set_state(DeviceState.DISCONNECTED)
        logger.info("SimMicroDM connection closed")

    def is_connected(self) -> bool:
        """Check if simulated connection is active."""
        return self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """Get simulated hardware info."""
        return {
            "manufacturer": self.manufacturer,
            "model": self.model,
            "channel_count": self.DM_Num,
            "voltage_range": [self.V_Min, self.V_Max],
            "relay_state": self._relay_state.name,
            "simulation": True,
        }

    def transform(self, cmd: np.ndarray) -> np.ndarray:
        """Transform normalized command to voltage range."""
        cmd = np.clip(cmd, -1, 1)
        return (cmd + 1) * (self.V_Max - self.V_Min) / 2 + self.V_Min

    def send(self, cmd: np.ndarray | float) -> np.ndarray:
        """Send command to simulated DM."""
        if isinstance(cmd, np.ndarray):
            return self.send_voltages(cmd)
        elif isinstance(cmd, (int, float)):
            return self.set_all_channel_voltage(float(cmd))
        raise MicroDMVoltageError(f"Unsupported command type: {type(cmd)}")

    def send_voltages(self, vs: np.ndarray, wait_time_s: float = 0.0) -> np.ndarray:
        """Send simulated voltage array."""
        vs = np.clip(vs, self.V_Min, self.V_Max)
        self._last_voltages = vs
        return self._last_voltages.copy()

    def set_channel_voltage(self, channel: int, voltage: float) -> None:
        """Set simulated channel voltage."""
        voltage = np.clip(voltage, self.V_Min, self.V_Max)
        self._last_voltages[channel] = voltage

    def set_all_voltage_by_arr(self, voltage_arr: np.ndarray) -> None:
        """Set all simulated channels by array."""
        voltage_arr = np.clip(voltage_arr, self.V_Min, self.V_Max)
        self._last_voltages = voltage_arr.copy()

    def set_all_channel_voltage(self, voltage: float) -> np.ndarray:
        """Set all simulated channels to same voltage."""
        voltage = np.clip(voltage, self.V_Min, self.V_Max)
        self._last_voltages = np.full(self.DM_Num, voltage)
        return self._last_voltages.copy()

    def set_relay_state(self, state: bool) -> None:
        """Set simulated relay state."""
        self._relay_state = RelayState.ON if state else RelayState.OFF
        logger.info(f"SimMicroDM relay {'opened' if state else 'closed'}")

    def get_actuator_positions(self) -> np.ndarray:
        """Get simulated actuator positions."""
        return self._last_voltages.copy()

    def reset_all(self) -> None:
        """Reset simulated DM to zero voltages."""
        self._last_voltages = np.zeros(self.DM_Num)

    def __repr__(self) -> str:
        return f"SimMicroDM(channels={self.DM_Num}, state={self._state.name})"