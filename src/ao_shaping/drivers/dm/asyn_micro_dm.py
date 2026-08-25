"""Asynchronous Micro DM (R50Power) driver using asyncio TCP.

Non-blocking alternative to the synchronous MicroDM driver.  Uses
``asyncio.open_connection`` for TCP I/O so that the event loop is never
blocked during voltage transmission.  A thin synchronous wrapper bridges
the async API to callers that expect a blocking interface (``open``/``close``).

Architecture:
    VoltageConverter  – pre-computed LUT for O(1) voltage → payload bytes
    SendResult        – frozen outcome of a single controller send
    AsyncR50Controller – per-controller async TCP client
    AsyncMicroDM      – multi-controller DM driver (subclass of DM)

Protocol constants and voltage conversion match MicroDM.py exactly.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from ao_shaping.drivers.dm._registry import register_dm
from ao_shaping.drivers.dm.base import DM
from ao_shaping.drivers.dm.MicroDM import (
    CMD_RELAY_OFF,
    CMD_RELAY_ON,
    CMD_SET_ALL_VOLTAGE_BY_ARR,
    FOOTER,
    HEADER,
    MAX_CHANNELS,
    VOLTAGE_MAX,
    VOLTAGE_MIN,
    WiringMap,
    voltages_to_payload,
)

# Re-export WiringMap so callers can import from asyn_micro_dm directly.
__all__ = [
    "AsyncMicroDM",
    "AsyncR50Controller",
    "SendResult",
    "VoltageConverter",
    "WiringMap",
]

DEFAULT_PORT = 10101
_DEFAULT_CONTROLLER_TIMEOUT = 10.0  # seconds

# =============================================================================
# Voltage Converter (LUT)
# =============================================================================


class VoltageConverter:
    """Pre-computed lookup table for voltage → payload byte conversion.

    Builds a 65536-entry table mapping every possible rounded raw value to
    its (high, low) byte pair using the same formula as
    :func:`voltages_to_payload`.  ``fill_buffer`` then converts a 50-element
    voltage array into a 100-byte payload in O(50) with no per-element
    float math.
    """

    def __init__(self) -> None:
        scale = 65535.0 / (20.0 * 3.4 * 3.3)  # ≈ 292.741
        offset = 20.0 * scale
        self._lut: list[tuple[int, int]] = []
        for raw in range(65536):
            high = raw // 256
            low = raw % 256
            self._lut.append((high, low))

        # Pre-compute constants for fill_buffer
        self._scale = scale
        self._offset = offset

    def fill_buffer(self, voltages: np.ndarray, buf: bytearray) -> None:
        """Convert 50 voltages into 100 interleaved bytes.

        Args:
            voltages: Array of exactly 50 float voltages.
            buf: Pre-allocated bytearray of at least 100 bytes (modified in place).
        """
        v = np.asarray(voltages, dtype=np.float32).ravel()
        np.clip(v, VOLTAGE_MIN, VOLTAGE_MAX, out=v)
        v *= self._scale
        v += self._offset
        raw = np.round(v).astype(np.uint16)
        for i in range(len(raw)):
            high, low = self._lut[int(raw[i])]
            buf[i * 2] = high
            buf[i * 2 + 1] = low

    def convert_single(self, voltage: float) -> tuple[int, int]:
        """Convert a single voltage to (high, low) bytes via LUT."""
        v = float(np.clip(voltage, VOLTAGE_MIN, VOLTAGE_MAX))
        raw = int(round(v * self._scale + self._offset))
        raw = max(0, min(65535, raw))
        return self._lut[raw]


# =============================================================================
# Send Result
# =============================================================================


@dataclass(frozen=True, slots=True)
class SendResult:
    """Outcome of a single ``AsyncR50Controller.send_voltages`` call.

    Attributes:
        success: Whether the data was written to the TCP stream.
        error: Human-readable error string (``None`` on success).
        latency_us: Wall-clock round-trip latency in microseconds.
    """

    success: bool
    error: str | None = None
    latency_us: float = 0.0


# =============================================================================
# Async R50 Controller
# =============================================================================


class AsyncR50Controller:
    """Async TCP client for a single R50Power controller (50 channels).

    Manages a persistent ``asyncio.StreamReader`` / ``StreamWriter`` pair.
    All network methods are coroutines.
    """

    def __init__(
        self,
        controller_id: int,
        ip: str,
        port: int = DEFAULT_PORT,
        timeout: float = _DEFAULT_CONTROLLER_TIMEOUT,
    ) -> None:
        self.controller_id = controller_id
        self.ip = ip
        self.port = port
        self._timeout = timeout

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._converter = VoltageConverter()

    @property
    def is_connected(self) -> bool:
        """Whether the TCP connection is active."""
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self) -> bool:
        """Open an async TCP connection to the controller.

        Returns:
            True on success, False on failure.
        """
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.ip, self.port),
                timeout=self._timeout,
            )
            logger.debug(
                f"AsyncR50Controller[{self.controller_id}] connected to "
                f"{self.ip}:{self.port}"
            )
            return True
        except (OSError, asyncio.TimeoutError) as exc:
            logger.warning(
                f"AsyncR50Controller[{self.controller_id}] connect failed: {exc}"
            )
            self._reader = None
            self._writer = None
            return False

    async def disconnect(self) -> None:
        """Close the TCP connection."""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._reader = None
            self._writer = None
            logger.debug(
                f"AsyncR50Controller[{self.controller_id}] disconnected"
            )

    async def send_voltages(
        self, voltages: np.ndarray, timeout: float | None = None
    ) -> SendResult:
        """Send a 50-channel voltage frame via the 0x09 command.

        Args:
            voltages: Array of 50 float voltages.
            timeout: Optional per-send timeout (seconds). Falls back to
                ``self._timeout`` when *None*.

        Returns:
            A :class:`SendResult` describing the outcome.
        """
        if self._writer is None:
            return SendResult(success=False, error="not_connected")

        buf = bytearray(MAX_CHANNELS * 2)
        self._converter.fill_buffer(voltages, buf)
        cmd = HEADER + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR]) + bytes(buf) + FOOTER

        t0 = time.perf_counter()
        try:
            self._writer.write(cmd)
            await asyncio.wait_for(
                self._writer.drain(),
                timeout=timeout or self._timeout,
            )
            latency = (time.perf_counter() - t0) * 1e6
            return SendResult(success=True, latency_us=latency)
        except asyncio.TimeoutError:
            return SendResult(success=False, error="drain_timeout")
        except (OSError, ConnectionError) as exc:
            return SendResult(success=False, error=str(exc))

    async def send_relay(self, state: bool) -> SendResult:
        """Open (True) or close (False) the relay."""
        if self._writer is None:
            return SendResult(success=False, error="not_connected")
        cmd = HEADER + bytes([CMD_RELAY_ON if state else CMD_RELAY_OFF]) + FOOTER
        try:
            self._writer.write(cmd)
            await asyncio.wait_for(self._writer.drain(), timeout=self._timeout)
            return SendResult(success=True)
        except (OSError, asyncio.TimeoutError) as exc:
            return SendResult(success=False, error=str(exc))


# =============================================================================
# Async MicroDM Driver
# =============================================================================


@register_dm("asyn_micro")
class AsyncMicroDM(DM):
    """Asynchronous multi-controller R50Power deformable mirror driver.

    Wraps multiple :class:`AsyncR50Controller` instances and presents the
    standard :class:`DM` interface.  The sync methods ``open()`` / ``close()``
    bridge to the async internals via ``_run_async``.

    Attributes:
        DM_Num: Total logical channel count (39×39 = 1521).
        V_Min: Minimum voltage (-20.0 V).
        V_Max: Maximum voltage (120.0 V).
    """

    DM_Num: int = 39 * 39
    V_Min: float = VOLTAGE_MIN
    V_Max: float = VOLTAGE_MAX

    def __init__(
        self,
        ips: list[str] | None = None,
        timeout: float = _DEFAULT_CONTROLLER_TIMEOUT,
        safety_mode: bool = True,
    ) -> None:
        super().__init__(safety_mode=safety_mode)

        self._ips: list[str] = ips or ["192.168.0.101"]
        self._timeout = timeout

        self._controllers: list[AsyncR50Controller] = [
            AsyncR50Controller(
                controller_id=i,
                ip=ip_str,
                port=10000 + int(ip_str.split(".")[-1]),
                timeout=timeout,
            )
            for i, ip_str in enumerate(self._ips, start=1)
        ]

        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._open = False

        logger.debug(
            f"AsyncMicroDM initialized: {len(self._controllers)} controller(s), "
            f"{self.DM_Num} channels"
        )

    # ---- Sync ↔ Async Bridge ------------------------------------------------

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Return the running event loop, or create one if needed."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._event_loop = loop
            return loop

    def _run_async(self, coro) -> object:  # noqa: ANN401
        """Run an async coroutine from sync code.

        If an event loop is already running, schedules via ``asyncio.ensure_future``.
        Otherwise runs to completion with ``loop.run_until_complete``.
        """
        loop = self._get_or_create_loop()
        if loop.is_running():
            # We're inside an async context — schedule and return the task
            return asyncio.ensure_future(coro)
        return loop.run_until_complete(coro)

    # ---- DM Interface (sync) ------------------------------------------------

    @classmethod
    def is_reachable(cls) -> bool:
        """Async driver is always considered reachable (connect at open time)."""
        return True

    @property
    def default_dm_unit_mask(self) -> np.ndarray:
        return np.ones(self.DM_NUM, dtype=bool)

    def open(self) -> None:
        """Connect to all controllers (sync bridge)."""
        self._run_async(self._async_open())

    async def _async_open(self) -> None:
        results = await self.connect_all()
        n_ok = sum(1 for v in results.values() if v)
        if n_ok == 0:
            raise ConnectionError(
                f"AsyncMicroDM: failed to connect any of "
                f"{len(self._controllers)} controller(s)"
            )
        self._open = True
        logger.info(
            f"AsyncMicroDM ready: {n_ok}/{len(self._controllers)} connected"
        )

    def close(self) -> None:
        """Disconnect all controllers (sync bridge)."""
        self._run_async(self._async_close())

    async def _async_close(self) -> None:
        for ctrl in self._controllers:
            await ctrl.disconnect()
        self._open = False
        logger.info("AsyncMicroDM disconnected")

    def is_connected(self) -> bool:
        return self._open and any(c.is_connected for c in self._controllers)

    def get_actuator_positions(self) -> np.ndarray:
        return self._last_voltages.copy()

    def transform(self, cmd: np.ndarray) -> np.ndarray:
        """Map [-1, 1] → [V_Min, V_Max]."""
        return self.transform_voltage(cmd)

    def _apply_voltages(self, vs: np.ndarray) -> np.ndarray:
        """Clip and store voltages. Actual TCP send happens in ``send_frame``."""
        vs = np.clip(vs, self.V_Min, self.V_Max)
        self._last_voltages = vs.copy()
        return self._last_voltages

    # ---- Async-specific methods ---------------------------------------------

    async def connect_all(self) -> dict[int, bool]:
        """Connect all controllers concurrently.

        Returns:
            Dict mapping controller_id (1-based) → success.
        """
        tasks = {
            ctrl.controller_id: ctrl.connect() for ctrl in self._controllers
        }
        results: dict[int, bool] = {}
        for cid, coro in tasks.items():
            results[cid] = await coro
        return results

    async def send_frame(self, voltages: np.ndarray | None = None) -> list[SendResult]:
        """Send a voltage frame to all controllers concurrently.

        Args:
            voltages: If *None*, sends ``self._last_voltages``.

        Returns:
            List of :class:`SendResult`, one per controller.
        """
        if voltages is not None:
            vs = np.clip(np.asarray(voltages, dtype=np.float64), self.V_Min, self.V_Max)
            self._last_voltages = vs.copy()
        else:
            vs = self._last_voltages

        chunk_size = MAX_CHANNELS
        tasks = []
        for idx, ctrl in enumerate(self._controllers):
            start = idx * chunk_size
            end = start + chunk_size
            chunk = vs[start:end]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), constant_values=0.0)
            tasks.append(ctrl.send_voltages(chunk))

        return list(await asyncio.gather(*tasks))

    async def shutdown(self, home_voltage: float = 0.0) -> None:
        """Safe shutdown: home voltages, relay off, disconnect all.

        Args:
            home_voltage: Voltage to set on all channels before disconnecting.
        """
        # Set home voltage on all controllers
        home_vs = np.full(self.DM_Num, home_voltage)
        self._last_voltages = home_vs.copy()

        for ctrl in self._controllers:
            if ctrl.is_connected:
                await ctrl.send_voltages(np.full(MAX_CHANNELS, home_voltage))
                await ctrl.send_relay(False)

        await self._async_close()
        logger.info("AsyncMicroDM shut down")


# =============================================================================
# Backwards compatibility alias
# =============================================================================

# Some callers may import the class under a different name
MicroDMAsync = AsyncMicroDM
