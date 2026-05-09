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

from ao_shaping.drivers.device_base import DeviceState, DeviceType
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
        >>> with MicroDM() as dm:
        ...     dm.send_voltages(np.zeros(50), 0.1)
    """

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
        self._device_id = device_id

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


# =============================================================================
# Joint DM Controller - Multiple DMs with Lookup Tables
# =============================================================================

class JointDMError(Exception):
    """Base exception for JointDM errors."""

    pass


class JointDMMappingError(JointDMError):
    """Raised when mapping is invalid."""

    pass


@dataclass
class ChannelMapping:
    """Single channel mapping from joint index to child DM."""

    child_index: int  #: Index of child DM in the list
    child_channel: int  #: Channel index in the child DM


class JointDMLookup:
    """Lookup tables for JointDM channel mapping.

    Provides two lookup modes:
    - 1D: logical_index → (child_index, child_channel)
    - 2D: (x, y) position → (child_index, child_channel)

    Example:
        >>> lookup = JointDMLookup(n_child_dms=2, channels_per_dm=50)
        >>>
        >>> # Add 1D mapping (joint index 0 → DM0 channel 10)
        >>> lookup.add_1d_mapping(0, 0, 10)
        >>>
        >>> # Add 2D mapping (position (100, 100) → DM1 channel 5)
        >>> lookup.add_2d_mapping(100, 100, 1, 5)
        >>>
        >>> # Query 1D
        >>> mapping = lookup.get_1d(0)  # Returns ChannelMapping(0, 10)
        >>>
        >>> # Query 2D
        >>> mapping = lookup.get_2d(100, 100)  # Returns ChannelMapping(1, 5)
    """

    def __init__(
        self,
        n_child_dms: int = 2,
        channels_per_dm: int = 50,
    ):
        """Initialize lookup tables.

        Args:
            n_child_dms: Number of child DMs
            channels_per_dm: Number of channels per child DM
        """
        self.n_child_dms = n_child_dms
        self.channels_per_dm = channels_per_dm

        # 1D lookup: joint_index → (child_index, child_channel)
        self._lookup_1d: dict[int, ChannelMapping] = {}

        # 2D lookup: (x, y) → (child_index, child_channel)
        self._lookup_2d: dict[tuple[int, int], ChannelMapping] = {}

        # Total number of joint channels
        self._n_joint_channels: int = 0

    def add_1d_mapping(
        self,
        joint_index: int,
        child_index: int,
        child_channel: int,
    ) -> None:
        """Add 1D mapping from joint index to child channel.

        Args:
            joint_index: Joint/logical channel index
            child_index: Index of child DM
            child_channel: Channel in the child DM
        """
        if not 0 <= child_index < self.n_child_dms:
            raise JointDMMappingError(f"Invalid child_index: {child_index}")
        if not 0 <= child_channel < self.channels_per_dm:
            raise JointDMMappingError(f"Invalid child_channel: {child_channel}")

        self._lookup_1d[joint_index] = ChannelMapping(
            child_index=child_index,
            child_channel=child_channel,
        )
        self._n_joint_channels = max(self._n_joint_channels, joint_index + 1)

    def add_2d_mapping(
        self,
        x: int,
        y: int,
        child_index: int,
        child_channel: int,
    ) -> None:
        """Add 2D mapping from position to child channel.

        Args:
            x: X position (e.g., pixel coordinate)
            y: Y position (e.g., pixel coordinate)
            child_index: Index of child DM
            child_channel: Channel in the child DM
        """
        if not 0 <= child_index < self.n_child_dms:
            raise JointDMMappingError(f"Invalid child_index: {child_index}")
        if not 0 <= child_channel < self.channels_per_dm:
            raise JointDMMappingError(f"Invalid child_channel: {child_channel}")

        self._lookup_2d[(x, y)] = ChannelMapping(
            child_index=child_index,
            child_channel=child_channel,
        )

    def get_1d(self, joint_index: int) -> ChannelMapping | None:
        """Get 1D mapping by joint index."""
        return self._lookup_1d.get(joint_index)

    def get_2d(self, x: int, y: int) -> ChannelMapping | None:
        """Get 2D mapping by position."""
        return self._lookup_2d.get((x, y))

    @property
    def n_joint_channels(self) -> int:
        """Total number of joint channels."""
        return self._n_joint_channels

    def get_joint_voltages(self, child_voltages: list[np.ndarray]) -> np.ndarray:
        """Extract joint voltages from child DM voltages.

        Args:
            child_voltages: List of voltage arrays, one per child DM

        Returns:
            Joint voltage array (padded with zeros for unmapped positions)
        """
        joint = np.zeros(self._n_joint_channels)
        for joint_idx, mapping in self._lookup_1d.items():
            joint[joint_idx] = child_voltages[mapping.child_index][mapping.child_channel]
        return joint

    def distribute_joint_voltages(
        self,
        joint_voltages: np.ndarray,
    ) -> list[np.ndarray]:
        """Distribute joint voltages to child DM voltage arrays.

        Args:
            joint_voltages: Joint voltage array

        Returns:
            List of voltage arrays for each child DM
        """
        child_voltages = [
            np.zeros(self.channels_per_dm)
            for _ in range(self.n_child_dms)
        ]
        for joint_idx, mapping in self._lookup_1d.items():
            if joint_idx < len(joint_voltages):
                child_voltages[mapping.child_index][mapping.child_channel] = joint_voltages[joint_idx]
        return child_voltages

    def save_to_csv(self, filepath: str) -> None:
        """Save lookup table to CSV file.

        Args:
            filepath: Output CSV path
        """
        with open(filepath, "w") as f:
            f.write("joint_index,x,y,child_dm,child_channel\n")
            # Write 1D mappings with x=y=-1 (placeholder)
            for j_idx, mapping in self._lookup_1d.items():
                f.write(f"{j_idx},-1,-1,{mapping.child_index},{mapping.child_channel}\n")
            # Write 2D mappings
            for (x, y), mapping in self._lookup_2d.items():
                j_idx = -1  # 2D-only entries
                f.write(f"{j_idx},{x},{y},{mapping.child_index},{mapping.child_channel}\n")

    @classmethod
    def load_from_csv(cls, filepath: str) -> JointDMLookup:
        """Load lookup table from CSV file.

        Args:
            filepath: Input CSV path

        Returns:
            Loaded JointDMLookup instance
        """
        lookup = cls()
        with open(filepath) as f:
            lines = f.readlines()
        # Skip header
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            joint_idx = int(parts[0])
            x = int(parts[1])
            y = int(parts[2])
            child_idx = int(parts[3])
            child_ch = int(parts[4])

            if joint_idx >= 0:
                lookup.add_1d_mapping(joint_idx, child_idx, child_ch)
            if x >= 0 and y >= 0:
                lookup.add_2d_mapping(x, y, child_idx, child_ch)

        return lookup


class JointDM(DM):
    """Joint DM controller managing multiple child DMs.

    This class combines multiple child DM instances into a single logical DM,
    with lookup tables for 1D and 2D channel mapping.

    Attributes:
        child_dms: List of child DM instances
        DM_Num: Total number of joint channels (from lookup)
        V_Min: Minimum voltage (from children)
        V_Max: Maximum voltage (from children)

    Example:
        >>> # Create two MicroDMs as children
        >>> dm1 = MicroDM(ip="192.168.0.101", port=10101)
        >>> dm2 = MicroDM(ip="192.168.0.102", port=10101)
        >>>
        >>> # Create JointDM with lookup
        >>> lookup = JointDMLookup(n_child_dms=2, channels_per_dm=50)
        >>> lookup.add_1d_mapping(0, 0, 10)   # joint 0 → DM0 channel 10
        >>> lookup.add_1d_mapping(1, 0, 11)   # joint 1 → DM0 channel 11
        >>> lookup.add_1d_mapping(2, 1, 5)    # joint 2 → DM1 channel 5
        >>>
        >>> joint = JointDM([dm1, dm2], lookup)
        >>> joint.open()
        >>> joint.set_channel_voltage(0, 2.5)  # Sets DM0 channel 10 to 2.5V
        >>> joint.set_all_voltage_by_arr(np.zeros(3))  # Sets joint voltages
        >>> joint.close()
    """

    def __init__(
        self,
        child_dms: list[DM],
        lookup: JointDMLookup,
    ):
        """Initialize JointDM.

        Args:
            child_dms: List of child DM instances
            lookup: Lookup table for channel mapping
        """
        self._child_dms = child_dms
        self._lookup = lookup

        # Total joint channels from lookup
        self.DM_Num = lookup.n_joint_channels

        # Voltage range from first child (assumed same for all)
        if child_dms:
            self.V_Min = getattr(child_dms[0], "V_Min", -1.0)
            self.V_Max = getattr(child_dms[0], "V_Max", 6.5)
        else:
            self.V_Min = -1.0
            self.V_Max = 6.5

        # Current joint voltages
        self._joint_voltages: np.ndarray = np.zeros(self.DM_Num)

    # ==================== DM Interface ====================

    def transform(self, cmd: np.ndarray) -> np.ndarray:
        """Transform normalized command to voltage range.

        Args:
            cmd: Normalized command array (-1 to 1)

        Returns:
            Voltage array in device range
        """
        cmd = np.clip(cmd, -1, 1)
        return (cmd + 1) * (self.V_Max - self.V_Min) / 2 + self.V_Min

    def send(self, cmd: np.ndarray | float) -> np.ndarray:
        """Send command to joint DM.

        Args:
            cmd: Voltage array or single value

        Returns:
            Applied joint voltages
        """
        if isinstance(cmd, np.ndarray):
            return self.send_voltages(cmd)
        elif isinstance(cmd, (int, float)):
            vs = np.full(self.DM_Num, float(cmd))
            return self.send_voltages(vs)
        raise JointDMError(f"Unsupported command type: {type(cmd)}")

    def open(self) -> None:
        """Open all child DM connections."""
        for dm in self._child_dms:
            dm.open()
        logger.info(f"JointDM opened with {len(self._child_dms)} child DMs")

    def close(self) -> None:
        """Close all child DM connections."""
        for dm in self._child_dms:
            dm.close()
        logger.info("JointDM closed")

    def get_actuator_positions(self) -> np.ndarray:
        """Get joint actuator positions.

        Returns:
            Array of joint voltages
        """
        # Collect voltages from all children
        child_voltages = [dm.get_actuator_positions() for dm in self._child_dms]
        return self._lookup.get_joint_voltages(child_voltages)

    # ==================== Control Methods ====================

    def set_channel_voltage(self, joint_index: int, voltage: float) -> None:
        """Set voltage for a single joint channel.

        Args:
            joint_index: Joint channel index
            voltage: Voltage value
        """
        mapping = self._lookup.get_1d(joint_index)
        if mapping is None:
            raise JointDMMappingError(f"No mapping for joint index {joint_index}")

        voltage = np.clip(voltage, self.V_Min, self.V_Max)
        self._child_dms[mapping.child_index].set_channel_voltage(
            mapping.child_channel, voltage
        )
        self._joint_voltages[joint_index] = voltage
        logger.debug(f"Set joint[{joint_index}] = {voltage}V → DM[{mapping.child_index}].ch[{mapping.child_channel}]")

    def set_all_voltage_by_arr(self, joint_voltages: np.ndarray) -> None:
        """Set all joint channels by array.

        Args:
            joint_voltages: Joint voltage array
        """
        child_voltages = self._lookup.distribute_joint_voltages(joint_voltages)

        for i, (dm, cv) in enumerate(zip(self._child_dms, child_voltages)):
            if len(cv) > 0:
                dm.set_all_voltage_by_arr(cv)

        self._joint_voltages = np.array(joint_voltages)
        logger.debug(f"Set all joint voltages: shape={joint_voltages.shape}")

    def set_all_channel_voltage(self, voltage: float) -> np.ndarray:
        """Set all joint channels to same voltage.

        Args:
            voltage: Voltage value for all channels

        Returns:
            Applied joint voltages
        """
        voltage = np.clip(voltage, self.V_Min, self.V_Max)
        vs = np.full(self.DM_Num, voltage)
        return self.send_voltages(vs)

    def send_voltages(self, joint_voltages: np.ndarray, wait_time_s: float = 0.001) -> np.ndarray:
        """Send joint voltage array.

        Args:
            joint_voltages: Joint voltage array
            wait_time_s: Wait time after sending

        Returns:
            Applied voltages
        """
        joint_voltages = np.clip(joint_voltages, self.V_Min, self.V_Max)
        self.set_all_voltage_by_arr(joint_voltages)
        import time
        time.sleep(wait_time_s)
        return self._joint_voltages.copy()

    # ==================== Position-based Methods ====================

    def set_position_voltage(self, x: int, y: int, voltage: float) -> None:
        """Set voltage by 2D position.

        Args:
            x: X position
            y: Y position
            voltage: Voltage value
        """
        mapping = self._lookup.get_2d(x, y)
        if mapping is None:
            raise JointDMMappingError(f"No mapping for position ({x}, {y})")

        voltage = np.clip(voltage, self.V_Min, self.V_Max)
        self._child_dms[mapping.child_index].set_channel_voltage(
            mapping.child_channel, voltage
        )
        logger.debug(f"Set position ({x},{y}) = {voltage}V → DM[{mapping.child_index}].ch[{mapping.child_channel}]")

    def get_position_map(self) -> np.ndarray:
        """Get 2D voltage map.

        Returns:
            2D array of voltages at mapped positions
        """
        result = {}
        for (x, y), mapping in self._lookup._lookup_2d.items():
            child_v = self._child_dms[mapping.child_index].get_actuator_positions()
            result[(x, y)] = child_v[mapping.child_channel]
        return result

    # ==================== Utility Methods ====================

    def reset_all(self) -> None:
        """Reset all channels to zero."""
        self.set_all_channel_voltage(0.0)

    def get_hardware_info(self) -> dict[str, Any]:
        """Get hardware info."""
        return {
            "type": "JointDM",
            "n_child_dms": len(self._child_dms),
            "n_joint_channels": self.DM_Num,
            "voltage_range": [self.V_Min, self.V_Max],
            "child_models": [
                getattr(dm, "model", "Unknown") for dm in self._child_dms
            ],
        }

    def is_connected(self) -> bool:
        """Check if all child DMs are connected."""
        return all(dm.is_connected() for dm in self._child_dms)

    def __repr__(self) -> str:
        return (
            f"JointDM("
            f"children={len(self._child_dms)}, "
            f"joint_channels={self.DM_Num}, "
            f"voltage_range=[{self.V_Min}, {self.V_Max}]"
            f")"
        )
