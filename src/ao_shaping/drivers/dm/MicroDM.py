"""Micro DM (R50Power) driver for deformable mirror control via async TCP.

Controls one or more R50Power controllers via TCP/IP using asyncio.
Each controller manages 50 channels in the range -20V to 120V.
Supports up to 26 controllers (1296 actuators) with IP-based addressing.

Wiring Map:
    Controller IPs and channel mappings are loaded from
    ``libs/micro_drive1300/wiring_map.json`` by default.
    The wiring map defines the relationship between:
    - Physical positions in the 39×39 actuator array
    - Controller IP addresses and payload positions (1-50)
    - Needle IDs and physical labels

    Disable with ``use_wiring_map=False`` to use default IPs.

Protocol:
    Header: 0xAA 0xBB, Footer: 0xCC 0xDD
    Commands:
        0x04 ch hv lv  - Set single channel voltage
        0x08 hv lv     - Set all channels to the same voltage
        0x09 +50*(hv,lv) - Set all 50 channels by array
        0x06           - Open relay
        0x07           - Close relay
    Voltage conversion (from MATLAB reference R50PowerV1.m):
        value = (voltage + 20) / 20 / 3.4 / 3.3 * 65535.0
        high_byte = floor(value / 255)
        low_byte = floor(mod(value, 256))

Reference:
    - libs/micro_drive1300/py/dm_control.py  (Python native reference)
    - libs/micro_drive1300/c/dm_control.c     (C reference)
    - libs/micro_drive1300/docs/R50PowerV1.m  (MATLAB reference)
    - docs/micro deformable mirror/R50Power(1).m

Example:
    >>> dm = MicroDM()
    >>> dm.open()
    >>> dm.send_voltages(np.zeros(50))
    >>> dm.set_relay_state(True)
    >>> print(dm.get_actuator_positions())
    >>> dm.close()

Channel Lookup:
    >>> dm = MicroDM()
    >>> info = dm.get_channel_by_xy(x=1, y=3)  # 39x39 array coordinates
    >>> print(info.ip_address, info.payload_position)
    >>> info = dm.get_channel_by_ip_position(ip_suffix=101, payload_position=13)
    >>> print(info.physical_label, info.physical_position)
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import Device, DeviceState, DeviceType
from ao_shaping.drivers.dm.base import DM
from ao_shaping.drivers.dm._registry import register_dm
from ao_shaping.utils.device_config import ConfigHandler, DeviceParam, param
from ao_shaping.utils.file import ROOT_DIR

# =============================================================================
# Protocol Constants
# =============================================================================

HEADER = bytes([0xAA, 0xBB])
FOOTER = bytes([0xCC, 0xDD])

CMD_SET_CHANNEL_VOLTAGE = 0x04
CMD_SET_ALL_CHANNEL_VOLTAGE = 0x08
CMD_SET_ALL_VOLTAGE_BY_ARR = 0x09
CMD_RELAY_ON = 0x06
CMD_RELAY_OFF = 0x07

# Hardware limits (from R50Power datasheet)
VOLTAGE_MIN = -20.0
VOLTAGE_MAX = 120.0
MAX_CONTROLLERS = 26
MAX_CHANNELS = 50
MAX_ACTUATORS = 1296
DEFAULT_TIMEOUT = 10.0

# Default IPs: 192.168.0.101 .. 192.168.0.126
DEFAULT_IPS = [f"192.168.0.{100 + i}" for i in range(1, MAX_CONTROLLERS + 1)]

# Wiring map path (relative to project root)
WIRING_MAP_PATH = ROOT_DIR / "libs" / "micro_drive1300" / "wiring_map.json"

# ── MicroDM 配置参数 ──────────────────────────────────────

_MICRO_DM_CONFIG_DIR = Path(
    os.environ.get("MICRO_DM_CONFIG_DIR", ROOT_DIR / "data" / "micro_dm_configs")
)


@dataclass
class MicroDMParams(DeviceParam):
    """MicroDM 配置参数（可持久化的标量参数）。"""
    timeout: float = param(default=DEFAULT_TIMEOUT, cast=float)
    use_wiring_map: bool = param(default=True, cast=bool)
    safety_mode: bool = param(default=True, cast=bool)


# 模块级单例，所有 MicroDM 实例共用
MICRO_DM_CONFIG = ConfigHandler(_MICRO_DM_CONFIG_DIR, "micro_dm", MicroDMParams)


# =============================================================================
# Wiring Map Data Classes (type-safe JSON schema definition)
# =============================================================================


@dataclass(frozen=True)
class SourceFiles:
    """Source Excel files used to generate the wiring map."""

    wiring_table: str
    device_mapping: str


@dataclass(frozen=True)
class Metadata:
    """Wiring map metadata."""

    description: str
    generated_at: str
    source_files: SourceFiles
    array_size: str
    total_channels: int


@dataclass(frozen=True)
class ChannelSchemaDoc:
    """Documentation for channel fields (from wiring_map.json schema.channel)."""

    needle_id: str
    physical_label: str
    mapping_row: str
    physical_position: str
    ip_suffix: str
    payload_position: str
    port: str


@dataclass(frozen=True)
class SchemaDoc:
    """Schema documentation section."""

    channel: ChannelSchemaDoc


@dataclass(frozen=True)
class ChannelEntry:
    """Single channel mapping entry from the wiring map.

    Represents one needle pin (277-330) and its mapping to:
    - Physical position in the 39×39 actuator array
    - Controller IP address (via ip_suffix)
    - Payload byte position within the controller's 50-channel frame
    - TCP port (10000 + ip_suffix)
    """

    needle_id: int | None
    physical_label: str | None
    mapping_row: int | None
    physical_position: int | None
    ip_suffix: int | None
    payload_position: int | None
    port: int | None = None

    @property
    def is_valid(self) -> bool:
        """Check if this entry has meaningful data (not all null)."""
        return self.physical_position is not None

    @property
    def ip_address(self) -> str | None:
        """Full IP address (assumes 192.168.0.x subnet)."""
        if self.ip_suffix is not None:
            return f"192.168.0.{self.ip_suffix}"
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChannelEntry:
        """Parse a ChannelEntry from a JSON dict."""
        return cls(
            needle_id=data.get("needle_id"),
            physical_label=data.get("physical_label"),
            mapping_row=data.get("mapping_row"),
            physical_position=data.get("physical_position"),
            ip_suffix=data.get("ip_suffix"),
            payload_position=data.get("payload_position"),
            port=data.get("port"),
        )


@dataclass(frozen=True)
class Group:
    """A group of channels (e.g., "一组", "二组")."""

    name: str
    channel_count: int
    channels: list[ChannelEntry] = field(default_factory=list)

    @classmethod
    def from_dict(cls, key: str, data: dict[str, Any]) -> Group:
        """Parse a Group from a JSON dict.

        Args:
            key: Group key from JSON (e.g., "group_1").
            data: Group data dict.
        """
        channels = [ChannelEntry.from_dict(ch) for ch in data.get("channels", [])]
        return cls(
            name=data.get("name", key),
            channel_count=data.get("channel_count", len(channels)),
            channels=channels,
        )


@dataclass(frozen=True)
class RangeInfo:
    """Min/max range for a numeric field."""

    min: int
    max: int


@dataclass(frozen=True)
class Summary:
    """Wiring map summary statistics."""

    unique_ip_suffixes: list[int]
    needle_id_range: RangeInfo
    physical_position_range: RangeInfo

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Summary:
        """Parse a Summary from a JSON dict."""
        nid = data.get("needle_id_range", {})
        ppr = data.get("physical_position_range", {})
        return cls(
            unique_ip_suffixes=data.get("unique_ip_suffixes", []),
            needle_id_range=RangeInfo(min=nid.get("min", 0), max=nid.get("max", 0)),
            physical_position_range=RangeInfo(
                min=ppr.get("min", 0), max=ppr.get("max", 0)
            ),
        )


@dataclass(frozen=True)
class WiringMap:
    """Complete wiring map for the 1300-channel DM cabinet.

    Parsed from ``libs/micro_drive1300/wiring_map.json``.
    Maps each needle pin (277-330) across multiple groups to:
    - Physical positions in a 39×39 actuator array
    - Controller IPs (via ip_suffix)
    - Payload byte positions (1-50) within each controller's frame
    """

    schema_version: str
    metadata: Metadata
    schema: SchemaDoc
    groups: dict[str, Group]
    summary: Summary

    @property
    def all_channels(self) -> list[ChannelEntry]:
        """Flattened list of all valid channel entries across all groups."""
        return [
            ch for group in self.groups.values() for ch in group.channels if ch.is_valid
        ]

    @property
    def unique_ips(self) -> list[str]:
        """Sorted list of unique controller IP addresses."""
        return [f"192.168.0.{s}" for s in sorted(self.summary.unique_ip_suffixes)]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WiringMap:
        """Parse a complete WiringMap from a JSON dict."""
        meta_data = data.get("metadata", {})
        sf_data = meta_data.get("source_files", {})
        meta = Metadata(
            description=meta_data.get("description", ""),
            generated_at=meta_data.get("generated_at", ""),
            source_files=SourceFiles(
                wiring_table=sf_data.get("wiring_table", ""),
                device_mapping=sf_data.get("device_mapping", ""),
            ),
            array_size=meta_data.get("array_size", ""),
            total_channels=meta_data.get("total_channels", 0),
        )

        ch_schema = data.get("schema", {}).get("channel", {})
        schema = SchemaDoc(
            channel=ChannelSchemaDoc(
                needle_id=ch_schema.get("needle_id", ""),
                physical_label=ch_schema.get("physical_label", ""),
                mapping_row=ch_schema.get("mapping_row", ""),
                physical_position=ch_schema.get("physical_position", ""),
                ip_suffix=ch_schema.get("ip_suffix", ""),
                payload_position=ch_schema.get("payload_position", ""),
                port=ch_schema.get("port", ""),
            ),
        )

        groups = {
            key: Group.from_dict(key, val)
            for key, val in data.get("groups", {}).items()
        }

        summary = Summary.from_dict(data.get("summary", {}))

        return cls(
            schema_version=data.get("$schema", ""),
            metadata=meta,
            schema=schema,
            groups=groups,
            summary=summary,
        )

    @classmethod
    def from_file(cls, path: Path) -> WiringMap | None:
        """Load a WiringMap from a JSON file.

        Args:
            path: Path to the wiring_map.json file.

        Returns:
            Parsed WiringMap, or None if file not found or invalid.
        """
        if not path.exists():
            logger.warning(f"Wiring map not found: {path}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            result = cls.from_dict(data)
            logger.info(
                f"Loaded wiring map: {result.metadata.description} "
                f"({len(result.groups)} groups, {result.metadata.total_channels} channels)"
            )
            return result
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning(f"Failed to load wiring map: {exc}")
            return None


# =============================================================================
# Runtime Channel Index (built from WiringMap for fast lookup)
# =============================================================================


@dataclass(frozen=True)
class ChannelInfo:
    """Runtime lookup view of a channel, built from WiringMap.

    Includes group context and pre-computed indices for fast lookup.
    """

    needle_id: int | None
    physical_label: str | None
    mapping_row: int | None
    physical_position: int | None
    ip_suffix: int | None
    payload_position: int | None
    port: int | None = None
    group_name: str | None = None
    group_key: str | None = None

    @property
    def ip_address(self) -> str | None:
        """Full IP address (assumes 192.168.0.x subnet)."""
        if self.ip_suffix is not None:
            return f"192.168.0.{self.ip_suffix}"
        return None

    @classmethod
    def from_entry(
        cls, entry: ChannelEntry, group_name: str, group_key: str
    ) -> ChannelInfo:
        """Create a ChannelInfo from a ChannelEntry with group context."""
        return cls(
            needle_id=entry.needle_id,
            physical_label=entry.physical_label,
            mapping_row=entry.mapping_row,
            physical_position=entry.physical_position,
            ip_suffix=entry.ip_suffix,
            payload_position=entry.payload_position,
            port=entry.port,
            group_name=group_name,
            group_key=group_key,
        )


# =============================================================================
# Voltage Conversion
# =============================================================================


def voltages_to_payload(voltages: np.ndarray | list[float] | float) -> bytes:
    """Convert voltage(s) to the 0x09 command payload.

    Supports single float (returns 2 bytes) or array (returns 2*N bytes).
    Vectorized numpy version — clips, scales, and interleaves
    high/low bytes in one pass.

    Protocol reference (from R50PowerV1.m MATLAB):
        value = (voltage + 20) / 20 / 3.4 / 3.3 * 65535.0
        highByte = floor(value / 255)
        lowByte = floor(mod(value, 256))

    # NOTO
    THEORETICAL CORRECT IMPLEMENTATION (consistent byte extraction):
        The MATLAB implementation has an inconsistency: it uses 255 for high byte
        division but 256 for low byte (via mod). Theoretically correct would be:
            raw = round(value)  # proper rounding to nearest integer
            high = raw // 256   # consistent with low = raw % 256
            low = raw % 256
        This ensures high * 256 + low == raw for the full value range.
        However, the MATLAB behavior is preserved for hardware compatibility.

    Args:
        voltages: Single voltage (float) or array of voltages (list/np.ndarray).

    Returns:
        Interleaved high/low bytes as bytes object.
    """
    v = np.atleast_1d(np.asarray(voltages, dtype=np.float64))
    np.clip(v, VOLTAGE_MIN, VOLTAGE_MAX, out=v)
    value = (v + 20.0) / 20.0 / 3.4 / 3.3 * 65535.0
    raw = np.round(value).astype(np.int32)
    high = (raw >> 8).astype(np.uint8)
    low = (raw & 0xFF).astype(np.uint8)
    interleaved = np.empty(2 * len(v), dtype=np.uint8)
    interleaved[0::2] = high
    interleaved[1::2] = low
    return interleaved.tobytes()


# =============================================================================
# Exceptions
# =============================================================================


class MicroDMError(Exception):
    """Base exception for MicroDM errors."""


class MicroDMConnectionError(MicroDMError):
    """Raised when connection to a controller fails."""


class MicroDMVoltageError(MicroDMError):
    """Raised when a voltage value is out of range."""


# =============================================================================
# Relay State
# =============================================================================


class RelayState(IntEnum):
    """Relay open/close state."""

    OFF = 0
    ON = 1


@dataclass(frozen=True)
class ControllerStatus:
    """Connection and reachability status for a single controller."""

    controller_id: int
    ip: str
    port: int
    ping_reachable: bool
    tcp_connected: bool

    @property
    def is_available(self) -> bool:
        """Controller is reachable via ping and has an active TCP connection."""
        return self.ping_reachable and self.tcp_connected


# =============================================================================
# Low-Level Async R50 Controller
# =============================================================================


class R50Controller:
    """Async TCP client for a single R50Power controller (50 channels).

    Low-level helper used internally by MicroDM. Each instance manages a
    persistent TCP connection to one physical power supply unit.

    Implements the same method names as :class:`DM` where applicable
    (``open``/``close``/``is_connected``) so it composes naturally
    with the DM interface.

    Attributes:
        controller_id: 1-based controller identifier.
        ip: IP address string.
        port: TCP port number.
    """

    def __init__(
        self,
        controller_id: int,
        ip: str,
        port: int,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.controller_id = controller_id
        self.ip = ip
        self.port = port
        self._timeout = timeout

        self._socket: socket.socket | None = None

    # ---- Properties ---------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Check if the TCP connection is established."""
        return self._socket is not None

    # ---- Context Manager ---------------------------------------------------

    def __enter__(self) -> R50Controller:
        """Context manager entry — opens the TCP connection.

        Usage::

            with R50Controller(1, "192.168.0.101", 10101) as ctrl:
                ctrl.set_all_channel_voltage(0.0)
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit — closes the TCP connection."""
        self.close()

    # ---- Connection Management ----------------------------------------------

    def open(self) -> bool:
        """Open a TCP connection to the controller.

        Returns:
            True on success, False on failure.
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self._timeout)
            sock.connect((self.ip, self.port))
            self._socket = sock
            logger.debug(
                f"R50Controller[{self.controller_id}] connected to {self.ip}:{self.port}"
            )
            return True
        except (socket.timeout, OSError, ConnectionError) as exc:
            logger.warning(f"R50Controller[{self.controller_id}] connect failed: {exc}")
            self._socket = None
            return False

    def close(self) -> None:
        """Close the TCP connection."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
            logger.debug(f"R50Controller[{self.controller_id}] disconnected")

    # ---- Command Sending ----------------------------------------------------

    def send(self, data: bytes) -> bool:
        """Send raw command bytes (DM-compatible alias for :meth:`send_command`).

        Args:
            data: Complete command packet (header + payload + footer).

        Returns:
            True on success.
        """
        return self.send_command(data)

    def send_command(self, data: bytes) -> bool:
        """Send raw command bytes to the controller.

        Args:
            data: Complete command packet (header + payload + footer).

        Returns:
            True on success. On failure, marks the controller as disconnected.
        """
        if self._socket is None:
            return False
        try:
            self._socket.sendall(data)
            return True
        except (OSError, ConnectionError) as exc:
            logger.warning(f"R50Controller[{self.controller_id}] send error: {exc}")
            self._socket = None
            return False

    def set_all_channel_voltage(self, voltage: float) -> bool:
        """Set all 50 channels to the same voltage (command 0x08).

        Args:
            voltage: Voltage in volts (clipped to [-20, 120]).

        Returns:
            True on success.
        """
        payload = voltages_to_payload(voltage)
        hv, lv = payload[0], payload[1]
        cmd = HEADER + bytes([CMD_SET_ALL_CHANNEL_VOLTAGE, hv, lv]) + FOOTER
        return self.send_command(cmd)

    def set_channel_voltage(self, channel: int, voltage: float) -> bool:
        """Set a single channel voltage (command 0x04).

        Args:
            channel: Channel index (0-49).
            voltage: Voltage in volts.

        Returns:
            True on success, False if channel is out of range.
        """
        if not 0 <= channel < MAX_CHANNELS:
            logger.warning(
                f"R50Controller[{self.controller_id}] invalid channel: {channel}"
            )
            return False
        payload = voltages_to_payload(voltage)
        hv, lv = payload[0], payload[1]
        cmd = HEADER + bytes([CMD_SET_CHANNEL_VOLTAGE, channel, hv, lv]) + FOOTER
        return self.send_command(cmd)

    def set_all_voltage_array(self, voltages: list[float]) -> bool:
        """Set all 50 channels by array (command 0x09, fastest method).

        Args:
            voltages: List of exactly 50 voltage values.

        Returns:
            True on success, False if array length is not 50.
        """
        if len(voltages) != MAX_CHANNELS:
            logger.warning(
                f"R50Controller[{self.controller_id}] expected {MAX_CHANNELS} voltages, "
                f"got {len(voltages)}"
            )
            return False

        cmd = (
            HEADER
            + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR])
            + voltages_to_payload(voltages)
            + FOOTER
        )
        return self.send_command(cmd)

    async def set_all_voltage_array_async(
        self, voltages: np.ndarray | list[float]
    ) -> bool:
        """Non-blocking version of :meth:`set_all_voltage_array`.

        Builds the command packet using the vectorised
        :func:`voltages_to_payload` helper, then offloads the TCP send
        to a thread-pool executor.

        Accepts both ``list[float]`` and ``np.ndarray`` — callers can
        pass data directly without converting.

        Args:
            voltages: Exactly 50 voltage values, as a list or array.

        Returns:
            True on success, False if array length is not 50.
        """
        if len(voltages) != MAX_CHANNELS:
            logger.warning(
                f"R50Controller[{self.controller_id}] expected {MAX_CHANNELS} voltages, "
                f"got {len(voltages)}"
            )
            return False

        cmd = (
            HEADER
            + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR])
            + voltages_to_payload(voltages)
            + FOOTER
        )
        return await self.send_command_async(cmd)

    def set_relay(self, state: bool) -> bool:
        """Open (True) or close (False) the relay.

        Command 0x06 = open, 0x07 = close.
        """
        cmd = HEADER + bytes([CMD_RELAY_ON if state else CMD_RELAY_OFF]) + FOOTER
        return self.send_command(cmd)

    async def send_command_async(self, data: bytes) -> bool:
        """Non-blocking version of send_command, runs the TCP send in a thread pool.

        Unlike the sync :meth:`send_command`, this method does not block the
        calling coroutine.  It offloads the blocking ``socket.sendall()`` to a
        default thread-pool executor so that multiple sends (e.g. to different
        controllers) can run concurrently via ``asyncio.gather``.

        Args:
            data: Complete command packet (header + payload + footer).

        Returns:
            True on success. On failure, marks the controller as disconnected.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.send_command, data)


# =============================================================================
# Main MicroDM Driver
# =============================================================================


@register_dm("micro")
class MicroDM(DM, Device):
    """Micro DM (R50Power) deformable mirror driver.

    Controls one or more R50Power controllers via async TCP.
    Defaults to a single 50-channel controller at 192.168.0.101:10101.

    When multiple IPs are supplied, channels are assigned sequentially::

        controller 0  →  channels   0-49
        controller 1  →  channels  50-99
        ...

    All TCP communication to all controllers happens in parallel via asyncio.

    Attributes:
        DM_Num: Total logical channel count (50 per controller).
        V_Min: Minimum voltage (-20.0 V).
        V_Max: Maximum voltage (120.0 V).

    Example:
        >>> dm = MicroDM()
        >>> dm.open()
        >>> dm.send_voltages(np.zeros(50))
        >>> dm.set_relay_state(True)
        >>> print(dm.get_actuator_positions())
        >>> dm.close()
    """

    DM_Num: int = 39 * 39
    DM_NUM: int = 39 * 39  # Alias for base class compatibility
    V_Min: float = VOLTAGE_MIN
    V_Max: float = VOLTAGE_MAX
    max_neibor_diff: float = float("inf")  # No neighbor constraint

    device_type = DeviceType.DM
    manufacturer = "R50Power"
    model = "MicroDM"

    @classmethod
    def is_reachable(cls) -> bool:
        for suffix in range(101, 127):
            ip = f"192.168.0.{suffix}"
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex((ip, 10000 + (suffix - 100)))
                sock.close()
                if result == 0:
                    return True
            except OSError:
                continue
        return False

    @property
    def default_dm_unit_mask(self) -> np.ndarray:
        return np.ones(self.DM_NUM, dtype=bool)

    def __init__(
        self,
        ips: list[str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        device_id: str = "",
        use_wiring_map: bool = True,
        exclude_ips: list[str] | None = None,
        exclude_ids: list[int] | None = None,
        safety_mode: bool = True,
    ):
        """Initialize the MicroDM driver.

        Args:
            ips: IP addresses of R50Power controllers.
                Default: loaded from wiring map if ``use_wiring_map=True``,
                otherwise ``["192.168.0.101"]`` (single controller).
                Pass multiple IPs for multi-controller setups.
            timeout: TCP connection/send timeout in seconds.
            device_id: Unique device identifier (auto-generated if empty).
            use_wiring_map: If True (default), load controller IPs from
                ``libs/micro_drive1300/wiring_map.json``.
            exclude_ips: IP addresses to skip during initialization.
                Controllers with these IPs will not be created.
            exclude_ids: Controller IDs (1-based) to skip during initialization.
                Controllers with these IDs will not be created.
            safety_mode: If True (default), send_voltages ramps from current
                state to target in steps bounded by max_neibor_diff.
        """
        self._init_values = {
            "timeout": timeout,
            "use_wiring_map": use_wiring_map,
            "safety_mode": safety_mode,
        }
        # 使用 defaults + __init__ 参数解析可持久化的标量参数
        params = MICRO_DM_CONFIG.resolve_from_config({}, init_values=self._init_values)

        DM.__init__(self, safety_mode=params.safety_mode)
        Device.__init__(self, device_id)

        # Load wiring map if enabled
        self._wiring_map: WiringMap | None = None
        self._channel_by_position: dict[
            int, ChannelInfo
        ] = {}  # physical_position → info
        self._channel_by_ip_payload: dict[
            tuple[int, int], ChannelInfo
        ] = {}  # (ip_suffix, payload_pos) → info
        self._channel_by_xy: dict[
            tuple[int, int], ChannelInfo
        ] = {}  # (x, y) in 39x39 → info

        if params.use_wiring_map:
            self._wiring_map = WiringMap.from_file(WIRING_MAP_PATH)
            if self._wiring_map is not None:
                self._build_channel_indices(self._wiring_map)

        # Determine IPs from wiring map or use defaults
        if ips is not None:
            self._ips = ips
        elif self._wiring_map is not None:
            self._ips = self._wiring_map.unique_ips
        else:
            self._ips = [DEFAULT_IPS[0]]

        # Filter out excluded IPs
        exclude_ip_set = set(exclude_ips or [])
        self._ips = [ip for ip in self._ips if ip not in exclude_ip_set]

        if exclude_ip_set:
            excluded_found = exclude_ip_set & set(self._ips)
            for ip in excluded_found:
                logger.warning(f"Excluded controller IP: {ip}")

        self._timeout = params.timeout

        self._relay_state = RelayState.OFF

        # Build async controllers, skipping excluded IDs
        exclude_id_set = set(exclude_ids or [])
        self._controllers: list[R50Controller] = [
            R50Controller(
                controller_id=i,
                ip=ip_str,
                port=10000 + int(ip_str.split(".")[-1]),  # 10000 + ip_suffix
                timeout=timeout,
            )
            for i, ip_str in enumerate(self._ips, start=1)
            if (i + 1) not in exclude_id_set
        ]

        if exclude_id_set:
            for cid in exclude_id_set:
                logger.warning(f"Excluded controller ID: {cid}")

        # Async event loop running in a background thread
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

        # Register device parameters
        self._register_parameters()

        logger.debug(
            f"MicroDM initialized: {len(self._controllers)} controller(s), "
            f"{self.DM_Num} channels, "
            f"voltage range [{self.V_Min}, {self.V_Max}] V"
        )

    # ---- Parameter Registration ---------------------------------------------

    def _register_parameters(self) -> None:
        """Register device parameters for the parameter management system."""
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
            description="Total logical channel count",
        )
        self.register_parameter(
            "n_controllers",
            len(self._controllers),
            description="Number of physical R50Power controllers",
        )

    # ---- Wiring Map Methods -------------------------------------------------

    def _build_channel_indices(self, wiring_map: WiringMap) -> None:
        """Build lookup indices from a parsed WiringMap.

        Creates three indices for O(1) channel lookup:
        - _channel_by_position: physical_position → ChannelInfo
        - _channel_by_ip_payload: (ip_suffix, payload_position) → ChannelInfo
        - _channel_by_xy: (x, y) in 39x39 grid → ChannelInfo
        """
        for group_key, group in wiring_map.groups.items():
            for entry in group.channels:
                if not entry.is_valid:
                    continue

                info = ChannelInfo.from_entry(entry, group.name, group_key)

                # Index by physical position (safe: is_valid guarantees non-None)
                assert entry.physical_position is not None
                self._channel_by_position[entry.physical_position] = info

                # Index by (ip_suffix, payload_position)
                if entry.ip_suffix is not None and entry.payload_position is not None:
                    self._channel_by_ip_payload[
                        (entry.ip_suffix, entry.payload_position)
                    ] = info

                # Index by (x, y) from physical_label (format: "group-row-col")
                if entry.physical_label is not None:
                    parts = entry.physical_label.split("-")
                    if len(parts) == 3:
                        try:
                            row = int(parts[1])  # y coordinate (1-based)
                            col = int(parts[2])  # x coordinate (1-based)
                            self._channel_by_xy[(col, row)] = info
                        except ValueError:
                            pass

        logger.debug(
            f"Built channel indices: {len(self._channel_by_position)} by position, "
            f"{len(self._channel_by_ip_payload)} by ip/payload, "
            f"{len(self._channel_by_xy)} by xy"
        )

    @property
    def wiring_map(self) -> WiringMap | None:
        """Access the loaded wiring map (read-only)."""
        return self._wiring_map

    def get_channel_by_xy(self, x: int, y: int) -> ChannelInfo | None:
        """Get channel info by x, y coordinates in the 39×39 array.

        Args:
            x: Column index (1-based, 1-39).
            y: Row index (1-based, 1-39).

        Returns:
            ChannelInfo if found, None otherwise.
        """
        return self._channel_by_xy.get((x, y))

    def get_channel_by_ip_position(
        self, ip_suffix: int, payload_position: int
    ) -> ChannelInfo | None:
        """Get channel info by controller IP suffix and payload position.

        Args:
            ip_suffix: IP address suffix (e.g., 101 for 192.168.0.101).
            payload_position: Channel position within the controller (1-50).

        Returns:
            ChannelInfo if found, None otherwise.
        """
        return self._channel_by_ip_payload.get((ip_suffix, payload_position))

    # ---- Async Infrastructure -----------------------------------------------

    def _run_async(self, coro: Any) -> Any:
        """Schedule a coroutine on the background event loop and wait for it.

        Args:
            coro: The coroutine to execute.

        Returns:
            The return value of the coroutine.

        Raises:
            RuntimeError: If the event loop is not running.
            TimeoutError: If the operation exceeds the configured timeout.
        """
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("MicroDM event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self._timeout)

    def _run_loop(self) -> None:
        """Target for the background thread: run the asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()
            self._loop = None

    # ---- Device Interface ---------------------------------------------------

    def open(self) -> None:
        """Open connections to all R50Power controllers.

        Starts the async event loop in a background thread, then connects
        to every controller in parallel. Individual connection failures
        are logged as warnings but do not prevent other controllers from
        connecting.

        Raises:
            MicroDMConnectionError: If no controller can be reached.
        """
        # Start the background event loop
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        # Wait until the loop is actually running
        while self._loop is None or not self._loop.is_running():
            time.sleep(0.001)

        # Connect all controllers in parallel, allowing partial failures
        connected, failed = self._run_async(self._connect_all())

        for ctrl_id, ip in failed:
            logger.warning(f"Controller[{ctrl_id}] {ip} connection failed")

        if connected == 0:
            self._set_state(DeviceState.ERROR, "No controllers connected")
            raise MicroDMConnectionError(
                f"Failed to connect to any of {len(self._controllers)} controller(s)"
            )

        self._set_state(DeviceState.READY)
        logger.info(
            f"MicroDM ready: {connected}/{len(self._controllers)} controllers connected"
        )

    def close(self) -> None:
        """Close all controller connections and stop the event loop."""
        if self._controllers:
            try:
                self._run_async(self._disconnect_all())
            except Exception as exc:
                logger.warning(f"Error disconnecting MicroDM: {exc}")

        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)

        self._set_state(DeviceState.DISCONNECTED)
        logger.info("MicroDM disconnected")

    def is_connected(self) -> bool:
        """Check whether at least one controller is connected and ready.

        Returns:
            True if at least one controller has an active TCP connection.
        """
        return (
            self._state == DeviceState.READY
            and self._loop is not None
            and any(ctrl.is_connected for ctrl in self._controllers)
        )

    # ---- Per-Controller Connection Management --------------------------------

    @staticmethod
    def _ping_host(ip: str, timeout: float = 2.0) -> bool:
        """Check if a host is reachable via ICMP ping.

        Args:
            ip: IP address or hostname to ping.
            timeout: Timeout in seconds.

        Returns:
            True if the host responds to ping, False otherwise.
        """
        param = "-n" if subprocess.os.name == "nt" else "-c"
        timeout_arg = (
            str(int(timeout * 1000))
            if subprocess.os.name == "nt"
            else str(int(timeout))
        )
        try:
            result = subprocess.run(
                ["ping", param, "1", "-W", timeout_arg, ip],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout + 1,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def get_connection_status(self) -> list[ControllerStatus]:
        """Get connection and reachability status for all controllers.

        Checks both ICMP ping reachability and TCP connection state.

        Returns:
            List of ControllerStatus, one per controller.
        """
        statuses: list[ControllerStatus] = []
        for ctrl in self._controllers:
            ping_ok = self._ping_host(ctrl.ip, timeout=self._timeout)
            tcp_ok = ctrl.is_connected
            statuses.append(
                ControllerStatus(
                    controller_id=ctrl.controller_id,
                    ip=ctrl.ip,
                    port=ctrl.port,
                    ping_reachable=ping_ok,
                    tcp_connected=tcp_ok,
                )
            )
        return statuses

    def connect_controller(self, controller_id: int) -> bool:
        """Connect a single controller by its ID.

        Args:
            controller_id: 1-based controller identifier.

        Returns:
            True if connection succeeded, False otherwise.

        Raises:
            MicroDMError: If controller ID not found.
        """
        ctrl = self._find_controller(controller_id)
        if ctrl.is_connected:
            logger.debug(f"Controller[{controller_id}] already connected")
            return True
        result = ctrl.open()
        if result:
            logger.info(f"Controller[{controller_id}] {ctrl.ip} connected")
        else:
            logger.warning(f"Controller[{controller_id}] {ctrl.ip} connection failed")
        return result

    def disconnect_controller(self, controller_id: int) -> None:
        """Disconnect a single controller by its ID.

        Args:
            controller_id: 1-based controller identifier.

        Raises:
            MicroDMError: If controller ID not found.
        """
        ctrl = self._find_controller(controller_id)
        ctrl.close()
        logger.info(f"Controller[{controller_id}] {ctrl.ip} disconnected")

    def reconnect_controller(self, controller_id: int) -> bool:
        """Reconnect a single controller by its ID.

        Disconnects first if already connected, then reconnects.

        Args:
            controller_id: 1-based controller identifier.

        Returns:
            True if reconnection succeeded, False otherwise.

        Raises:
            MicroDMError: If controller ID not found.
        """
        ctrl = self._find_controller(controller_id)
        ctrl.close()
        result = ctrl.open()
        if result:
            logger.info(f"Controller[{controller_id}] {ctrl.ip} reconnected")
        else:
            logger.warning(f"Controller[{controller_id}] {ctrl.ip} reconnection failed")
        return result

    def _find_controller(self, controller_id: int) -> R50Controller:
        """Find a controller by its ID.

        Args:
            controller_id: 1-based controller identifier.

        Returns:
            The matching R50Controller.

        Raises:
            MicroDMError: If controller ID not found.
        """
        for ctrl in self._controllers:
            if ctrl.controller_id == controller_id:
                return ctrl
        raise MicroDMError(f"Controller ID {controller_id} not found")

    def get_hardware_info(self) -> dict[str, Any]:
        """Get hardware-specific information.

        Returns:
            Dictionary with device info, connection status, and ranges.
        """
        connected_count = sum(1 for c in self._controllers if c.is_connected)
        return {
            "manufacturer": self.manufacturer,
            "model": self.model,
            "n_controllers": len(self._controllers),
            "connected_controllers": connected_count,
            "channels_per_controller": MAX_CHANNELS,
            "total_channels": self.DM_Num,
            "voltage_range": [self.V_Min, self.V_Max],
            "relay_state": self._relay_state.name,
            "controller_ips": [ctrl.ip for ctrl in self._controllers],
            "controller_ports": [ctrl.port for ctrl in self._controllers],
        }

    def get_actuator_positions(self) -> np.ndarray:
        """Get the current actuator voltages.

        Returns:
            Copy of the current voltage array (DM_Num elements).
        """
        return self._last_voltages.copy()

    # ---- DM Interface -------------------------------------------------------

    def transform(self, cmd: np.ndarray) -> np.ndarray:
        """Transform a normalized command ``[-1, 1]`` to device voltage range.

        Maps [-1, 1] linearly to [V_Min, V_Max].

        Args:
            cmd: Normalized command array.

        Returns:
            Voltage array in the device range.
        """
        cmd = np.clip(cmd, -1.0, 1.0)
        return (cmd + 1.0) * (self.V_Max - self.V_Min) / 2.0 + self.V_Min

    def send(self, cmd: np.ndarray | float) -> np.ndarray:
        """Send a voltage command to all channels.

        Args:
            cmd: Voltage array (DM_Num elements) or a single float for all channels.

        Returns:
            The applied voltage array.

        Raises:
            MicroDMVoltageError: If the command type is unsupported.
        """
        if isinstance(cmd, np.ndarray):
            return self.send_voltages(cmd)
        if isinstance(cmd, (int, float)):
            return self.set_all_channel_voltage(float(cmd))
        raise MicroDMVoltageError(f"Unsupported command type: {type(cmd)}")

    def _apply_voltages(self, vs: np.ndarray) -> np.ndarray:
        """Low-level voltage application via async TCP to all controllers."""
        vs = np.clip(vs, self.V_Min, self.V_Max)
        self._run_async(self._send_voltages_async(vs))
        self._last_voltages = vs.copy()
        return self._last_voltages

    def send_voltages(self, vs: np.ndarray, wait_time_s: float = 0.001) -> np.ndarray:
        """Send a voltage array to all channels using async parallel TCP.

        Voltages are distributed across R50Power controllers automatically.
        When safety_mode is True (default), voltages are ramped from current
        state to target in steps bounded by max_neibor_diff.

        Args:
            vs: Voltage array for all logical channels.
            wait_time_s: Sleep after sending (hardware settling time).

        Returns:
            The applied voltage array.

        Raises:
            MicroDMVoltageError: If the array length does not match DM_Num.
        """
        vs = np.asarray(vs, dtype=np.float64)
        if vs.shape != (self.DM_Num,):
            raise MicroDMVoltageError(
                f"Expected {self.DM_Num} voltages, got {vs.shape}"
            )
        result = super().send_voltages(vs, wait_time_s=wait_time_s)
        return result

    # ---- Async Internal Methods ---------------------------------------------

    async def _connect_all(self) -> tuple[int, list[tuple[int, str]]]:
        """Connect to all controllers in parallel.

        Returns:
            Tuple of (connected_count, list_of_(controller_id, ip) for failures).
        """
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, ctrl.open) for ctrl in self._controllers]
        results = await asyncio.gather(*tasks)
        connected = sum(1 for r in results if r)
        failed = [
            (ctrl.controller_id, ctrl.ip)
            for ctrl, ok in zip(self._controllers, results)
            if not ok
        ]
        return connected, failed

    async def _disconnect_all(self) -> None:
        """Disconnect all controllers in parallel."""
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, ctrl.close) for ctrl in self._controllers]
        await asyncio.gather(*tasks)

    async def _send_voltages_async(self, vs: np.ndarray) -> None:
        """Send voltage array to all controllers in parallel.

        Channels are distributed round-robin across controllers:

            controller 0  →  vs[0:50]
            controller 1  →  vs[50:100]
            ...
        """
        tasks: list[Any] = []
        for ctrl_idx, ctrl in enumerate(self._controllers):
            start = ctrl_idx * MAX_CHANNELS
            end = start + MAX_CHANNELS
            if start >= len(vs):
                break
            chunk = vs[start:end]
            # Pad with zeros if the last controller has fewer than 50
            if len(chunk) < MAX_CHANNELS:
                chunk = np.pad(
                    chunk, (0, MAX_CHANNELS - len(chunk)), constant_values=0.0
                )

            cmd = (
                HEADER
                + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR])
                + voltages_to_payload(chunk)
                + FOOTER
            )
            tasks.append(ctrl.send_command_async(cmd))

        if tasks:
            await asyncio.gather(*tasks)

    async def _set_relay_async(self, state: bool) -> None:
        """Send relay command to all connected controllers in parallel."""
        cmd = HEADER + bytes([CMD_RELAY_ON if state else CMD_RELAY_OFF]) + FOOTER
        tasks = [
            ctrl.send_command_async(cmd)
            for ctrl in self._controllers
            if ctrl.is_connected
        ]
        if tasks:
            await asyncio.gather(*tasks)

    # ---- Protocol Commands --------------------------------------------------

    def set_channel_voltage(self, channel: int, voltage: float) -> None:
        """Set voltage for a single logical channel.

        Automatically routes to the correct physical controller.

        Args:
            channel: Logical channel index (0 to DM_Num - 1).
            voltage: Voltage in volts.

        Raises:
            MicroDMVoltageError: If channel is out of range.
        """
        if not 0 <= channel < self.DM_Num:
            raise MicroDMVoltageError(
                f"Channel must be 0-{self.DM_Num - 1}, got {channel}"
            )

        ctrl_idx = channel // MAX_CHANNELS
        ch_idx = channel % MAX_CHANNELS

        if ctrl_idx < len(self._controllers):
            ctrl = self._controllers[ctrl_idx]
            voltage_clipped = max(self.V_Min, min(self.V_Max, float(voltage)))
            ctrl.set_channel_voltage(ch_idx, voltage_clipped)
            self._last_voltages[channel] = voltage_clipped

    def set_all_voltage_by_arr(self, voltages: np.ndarray) -> None:
        """Set all logical channels by array.

        Delegates to :meth:`send_voltages`.

        Args:
            voltages: Voltage array for all channels.
        """
        self.send_voltages(voltages)

    def set_all_channel_voltage(self, voltage: float) -> np.ndarray:
        """Set all logical channels to the same voltage.

        Sends to all controllers in parallel.

        Args:
            voltage: Voltage in volts for all channels.

        Returns:
            The applied voltage array.
        """
        voltage = max(self.V_Min, min(self.V_Max, float(voltage)))
        vs = np.full(self.DM_Num, voltage)
        self.send_voltages(vs)
        logger.debug(f"Set all channels to {voltage} V")
        return self._last_voltages.copy()

    def set_relay_state(self, state: bool) -> None:
        """Set relay state on all connected controllers in parallel.

        Args:
            state: True to open relay, False to close.
        """
        self._run_async(self._set_relay_async(state))
        self._relay_state = RelayState.ON if state else RelayState.OFF
        logger.info(f"Relay {'opened' if state else 'closed'} on all controllers")

    def reset_all(self) -> None:
        """Reset all channels to 0 V."""
        vs = np.zeros(self.DM_Num)
        self.send_voltages(vs)
        logger.info("MicroDM reset to 0 V")

    def __repr__(self) -> str:
        connected = sum(1 for c in self._controllers if c.is_connected)
        return (
            f"MicroDM("
            f"controllers={len(self._controllers)}, "
            f"connected={connected}, "
            f"channels={self.DM_Num}, "
            f"voltage=[{self.V_Min}, {self.V_Max}] V, "
            f"state={self._state.name}"
            f")"
        )

if __name__ == '__main__':
    CONTROLLER_IP = '192.168.0.101'
    CONTROLLER_PORT = 10101
    controller = R50Controller(controller_id=1, ip=CONTROLLER_IP, port=CONTROLLER_PORT)
    if controller.open():
        controller.set_relay(True)
        # controller.set_all_channel_voltage(5.0)
        volts = [0.0 for i in range(50)]
        volts[0] = 15.0
        controller.set_all_voltage_array(volts)

        volts[0] = 10.0
        controller.set_all_voltage_array(volts)

        volts[0] = 5.0
        controller.set_all_voltage_array(volts)

        controller.set_relay(False)
        controller.close()