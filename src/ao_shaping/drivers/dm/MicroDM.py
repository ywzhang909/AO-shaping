"""Micro DM (R50Power) driver for deformable mirror control via async TCP.

Controls one or more R50Power controllers via TCP/IP using asyncio.
Each controller manages 50 channels in the range -20V to 120V.
Supports up to 26 controllers (1296 actuators) with IP-based addressing.

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
    >>> dm.close()
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from enum import IntEnum
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import Device, DeviceState, DeviceType
from ao_shaping.drivers.dm.base import DM

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


# =============================================================================
# Voltage Conversion
# =============================================================================

def voltage_to_bytes(voltage: float) -> tuple[int, int]:
    """Convert voltage to high/low bytes per the R50Power protocol.

    Formula (from MATLAB R50PowerV1.m):
        value = (voltage + 20) / 20 / 3.4 / 3.3 * 65535.0
        high_byte = floor(value / 255)
        low_byte = floor(mod(value, 256))

    The C/Python reference rounds: raw = int(value + 0.5), then raw // 256 / % 256.

    Args:
        voltage: Voltage in volts (-20 to 120).

    Returns:
        Tuple of (high_byte, low_byte).
    """
    value = (voltage + 20.0) / 20.0 / 3.4 / 3.3 * 65535.0
    raw = int(value + 0.5)  # Round, matching C/Python reference
    return raw // 256, raw % 256


def voltage_to_bytes_clipped(voltage: float) -> tuple[int, int]:
    """Convert voltage with hard clipping to [-20, 120] range."""
    return voltage_to_bytes(max(VOLTAGE_MIN, min(VOLTAGE_MAX, voltage)))


def voltages_to_payload(voltages: np.ndarray | list[float]) -> bytes:
    """Convert a voltage array to the 0x09 command payload (100 bytes).

    Vectorized numpy version — clips, scales, rounds, and interleaves
    high/low bytes in one pass.  Accepts both ``list[float]`` and
    ``np.ndarray``, so callers (sync and async) can pass data in whatever
    form they already have without an extra conversion step.

    Compared to a per-element loop::

        for v in voltages:                             # 50 Python calls
            hv, lv = voltage_to_bytes_clipped(v)       # 50 clip + convert
            cmd += bytes([hv, lv])                     # 50 small allocations

    this function replaces all 150+ intermediate operations with a single
    numpy vectorised pass.
    """
    v = np.asarray(voltages, dtype=np.float64)
    np.clip(v, VOLTAGE_MIN, VOLTAGE_MAX, out=v)  # in-place, no copy
    value = (v + 20.0) / (20.0 * 3.4 * 3.3) * 65535.0
    raw = np.round(value).astype(np.int32)
    high = (raw // 256).astype(np.uint8)
    low = (raw % 256).astype(np.uint8)
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
        hv, lv = voltage_to_bytes_clipped(voltage)
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
        hv, lv = voltage_to_bytes_clipped(voltage)
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

        cmd = HEADER + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR])
        for v in voltages:
            hv, lv = voltage_to_bytes_clipped(v)
            cmd += bytes([hv, lv])
        cmd += FOOTER
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
    V_Min: float = VOLTAGE_MIN
    V_Max: float = VOLTAGE_MAX

    device_type = DeviceType.DM
    manufacturer = "R50Power"
    model = "MicroDM"

    def __init__(
        self,
        ips: list[str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        device_id: str = "",
    ):
        """Initialize the MicroDM driver.

        Args:
            ips: IP addresses of R50Power controllers.
                Default: ``["192.168.0.101"]`` (single controller).
                Pass multiple IPs for multi-controller setups.
            timeout: TCP connection/send timeout in seconds.
            device_id: Unique device identifier (auto-generated if empty).
        """
        Device.__init__(self, device_id)

        self._ips = ips or [DEFAULT_IPS[0]]
        self._timeout = timeout

        # Current voltage state (logical actuator voltages)
        self._voltages: np.ndarray = np.zeros(self.DM_Num)
        self._relay_state = RelayState.OFF

        # Build async controllers
        self._controllers: list[R50Controller] = [
            R50Controller(
                controller_id=i + 1,
                ip=ip_str,
                port=10000 + i + 1,  # 10101, 10102, ...
                timeout=timeout,
            )
            for i, ip_str in enumerate(self._ips)
        ]

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
            "voltage_min", self.V_Min, description="Minimum allowed voltage (V)",
        )
        self.register_parameter(
            "voltage_max", self.V_Max, description="Maximum allowed voltage (V)",
        )
        self.register_parameter(
            "channel_count", self.DM_Num,
            description="Total logical channel count",
        )
        self.register_parameter(
            "n_controllers", len(self._controllers),
            description="Number of physical R50Power controllers",
        )

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
        to every controller in parallel.

        Raises:
            MicroDMConnectionError: If no controller can be reached.
        """
        # Start the background event loop
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        # Wait until the loop is actually running
        while self._loop is None or not self._loop.is_running():
            time.sleep(0.001)

        # Connect all controllers in parallel
        connected = self._run_async(self._connect_all())

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
        return self._voltages.copy()

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

    def send_voltages(self, vs: np.ndarray, wait_time_s: float = 0.001) -> np.ndarray:
        """Send a voltage array to all channels using async parallel TCP.

        Voltages are distributed across R50Power controllers automatically.

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

        vs = np.clip(vs, self.V_Min, self.V_Max)
        self._run_async(self._send_voltages_async(vs))
        self._voltages = vs.copy()
        time.sleep(wait_time_s)
        return self._voltages.copy()

    # ---- Async Internal Methods ---------------------------------------------

    async def _connect_all(self) -> int:
        """Connect to all controllers in parallel.

        Returns:
            Number of successfully connected controllers.
        """
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(None, ctrl.open)
            for ctrl in self._controllers
        ]
        results = await asyncio.gather(*tasks)
        return sum(1 for r in results if r)

    async def _disconnect_all(self) -> None:
        """Disconnect all controllers in parallel."""
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(None, ctrl.close)
            for ctrl in self._controllers
        ]
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
            chunk = vs[start:end].tolist()
            # Pad with zeros if the last controller has fewer than 50
            if len(chunk) < MAX_CHANNELS:
                chunk.extend([0.0] * (MAX_CHANNELS - len(chunk)))

            # Build full 0x09 command bytes synchronously, then async send
            cmd = HEADER + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR])
            for v in chunk:
                hv, lv = voltage_to_bytes_clipped(v)
                cmd += bytes([hv, lv])
            cmd += FOOTER
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
            self._voltages[channel] = voltage_clipped

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
        self._run_async(self._send_voltages_async(vs))
        self._voltages = vs.copy()
        logger.debug(f"Set all channels to {voltage} V")
        return self._voltages.copy()

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

