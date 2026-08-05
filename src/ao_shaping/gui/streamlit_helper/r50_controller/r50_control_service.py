"""R50ControlService — hardware-owning child process for the R50 controller UI.

Implements the dual-process architecture: this module runs in a separate
process (spawned by the Streamlit UI) and owns ALL hardware IO — TCP to the
R50Power controllers, waveform loops, relay control and the joint 36x36 matrix.

The Streamlit process never touches hardware: it sends :class:`ServiceCommand`
objects through a :class:`multiprocessing.Queue` and polls :class:`ServiceStatus`
objects from a second queue. No ``streamlit`` import is allowed here so the
module stays importable under the ``spawn`` start method.

Safety guarantees:
- ``close()`` powers the relay off before closing TCP connections.
- Waveform stop always sends 0V to the waveform targets (safety cleanup).
- Command watchdog: if no command arrives for ``watchdog_timeout`` seconds
  while a waveform is running, the waveform is stopped and relays powered off.
- Parent watchdog: if the UI process dies, the service powers everything off
  and exits (no orphan process, no hardware left energized).
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import queue
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

# The service runs in a ``spawn``-context child process whose sys.path is not
# inherited from the Streamlit parent, so re-add the project root and src dir
# before any ao_shaping import (idempotent when already present).
_SERVICE_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _SERVICE_FILE.parents[5]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from loguru import logger
from multiprocessing.context import SpawnProcess

from ao_shaping.drivers.dm.MicroDM import MicroDM
from ao_shaping.gui.streamlit_helper.r50_controller.r50_channel_select import (
    CFG,
    GRID_SIZE,
    SINGLE_CHANNELS,
    build_groups,
    jc_build_ip_index,
    jc_build_wiring_index,
    jc_matrix_to_flat,
    load_csv,
)
from ao_shaping.gui.streamlit_helper.r50_controller.r50_connection import (
    SimulatedMicroDM,
    create_controller,
    power_off_and_close,
    set_relay,
)
from ao_shaping.gui.streamlit_helper.r50_controller.r50_voltage_send import clip_voltage

STATUS_PUSH_INTERVAL = 0.5
WATCHDOG_CHECK_INTERVAL = 1.0
PARENT_CHECK_INTERVAL = 2.0
WAVEFORM_DEFAULT_DT = 0.05


# =============================================================================
# Wire dataclasses (UI <-> Service)
# =============================================================================


class WaveformType(Enum):
    """Waveform kinds supported by the service waveform engine."""

    DC = auto()
    SINE = auto()
    SQUARE = auto()
    ALT = auto()
    HOLD = auto()


@dataclass
class WaveformConfig:
    """Parameters for a waveform run. ``targets`` are ``(ip_suffix, payload_position)`` 1-based."""

    type: WaveformType = WaveformType.DC
    targets: list[tuple[int, int]] = field(default_factory=list)
    voltage: float = 0.0
    amp: float = 0.0
    offset: float = 0.0
    freq: float = 1.0
    voltage_a: float = 0.0
    voltage_b: float = 0.0
    dt: float = WAVEFORM_DEFAULT_DT
    vmin: float = CFG.HW_VOLTAGE_MIN
    vmax: float = CFG.HW_VOLTAGE_MAX


@dataclass
class ServiceCommand:
    """A command from the UI to the service."""

    action: str
    ip: str = ""
    port: int = CFG.DEFAULT_PORT
    simulate: bool = False
    controller_id: int = 1
    group_name: str = ""
    waveform: WaveformConfig | None = None
    matrix: np.ndarray | None = None
    relay_on: bool = False
    voltage: float = 0.0
    targets: list[tuple[int, int]] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceStatus:
    """Status snapshot pushed from the service to the UI every 500ms."""

    controllers: dict[int, dict] = field(default_factory=dict)
    joint_connected: bool = False
    joint_relay_on: bool = False
    joint_controller_count: int = 0
    group_connected: bool = False
    group_relay_on: bool = False
    group_name: str = ""
    waveform_running: bool = False
    waveform_type: str | None = None
    current_voltages: dict[int, list[float]] = field(default_factory=dict)
    joint_matrix: list[list[float]] | None = None
    last_error: str = ""
    timestamp: float = 0.0


# =============================================================================
# Waveform math (pure, no IO)
# =============================================================================


class WaveformEngine:
    """Pure math for waveform computation and clipping (unit-testable)."""

    @staticmethod
    def compute(cfg: WaveformConfig, t: float) -> float:
        """Voltage at time ``t`` for the given waveform config (clipped)."""
        if cfg.type in (WaveformType.DC, WaveformType.HOLD):
            v = cfg.voltage
        elif cfg.type == WaveformType.SINE:
            v = cfg.offset + cfg.amp * np.sin(2.0 * np.pi * cfg.freq * t)
        elif cfg.type == WaveformType.SQUARE:
            v = cfg.voltage_a if int(t * 2.0 * cfg.freq) % 2 == 0 else cfg.voltage_b
        elif cfg.type == WaveformType.ALT:
            v = cfg.voltage if int(t * 2.0 * cfg.freq) % 2 == 0 else 0.0
        else:
            v = cfg.voltage
        return WaveformEngine.clip_all(float(v), cfg)

    @staticmethod
    def clip_all(v: float, cfg: WaveformConfig) -> float:
        """Clip a voltage into the config's safe range."""
        return float(np.clip(float(v), cfg.vmin, cfg.vmax))


# =============================================================================
# Controller adapters (blocking driver IO via executor, per-controller locks)
# =============================================================================


class ControllerAdapter(ABC):
    """Interface every adapter implements; adapters differ by connection mode."""

    @abstractmethod
    async def set_relay(self, on: bool) -> bool:
        """Power relay on/off for all controllers this adapter owns."""

    @abstractmethod
    async def send_voltages(self, voltages_by_ip: dict[int, np.ndarray]) -> dict[int, Exception]:
        """Send 50-channel arrays keyed by ip_suffix; returns per-ip errors."""

    @abstractmethod
    async def read_voltages(self) -> dict[int, np.ndarray]:
        """Read current voltages from hardware (or local copy in simulation)."""

    @abstractmethod
    async def close(self) -> None:
        """Power off relays, then close all connections."""

    @abstractmethod
    def get_status(self) -> dict:
        """Status dict for ServiceStatus.controllers."""


class SingleControllerAdapter(ControllerAdapter):
    """Owns exactly one R50Power controller (single-controller mode)."""

    def __init__(self, ip_suffix: int, ctrl: Any, simulate: bool = False) -> None:
        self.ip_suffix = int(ip_suffix)
        self.ctrl = ctrl
        self.simulate = simulate
        self.relay_on = False
        self._voltages = np.zeros(SINGLE_CHANNELS, dtype=np.float64)
        self._lock = asyncio.Lock()

    async def set_relay(self, on: bool) -> bool:
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(None, set_relay, self.ctrl, on)
        if ok:
            self.relay_on = bool(on)
        return ok

    async def send_voltages(self, voltages_by_ip: dict[int, np.ndarray]) -> dict[int, Exception]:
        arr = voltages_by_ip.get(self.ip_suffix)
        if arr is None:
            return {}
        errors: dict[int, Exception] = {}
        async with self._lock:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._send_blocking, np.asarray(arr, dtype=np.float64))
                self._voltages[:] = np.asarray(arr, dtype=np.float64)
            except Exception as exc:  # noqa: BLE001 — hardware failures must be isolated per adapter
                errors[self.ip_suffix] = exc
        return errors

    def _send_blocking(self, arr: np.ndarray) -> None:
        if hasattr(self.ctrl, "set_all_voltage_array"):
            if not self.ctrl.set_all_voltage_array(arr.tolist()):
                raise ConnectionError(f"send failed for 192.168.0.{self.ip_suffix}")
            return
        if not self.ctrl.set_all_channel_voltage(float(arr[0] if arr.size else 0.0)):
            raise ConnectionError(f"send failed for 192.168.0.{self.ip_suffix}")

    async def read_voltages(self) -> dict[int, np.ndarray]:
        return {self.ip_suffix: self._voltages.copy()}

    async def close(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, power_off_and_close, self.ctrl)

    def get_status(self) -> dict:
        return {
            "connected": True,
            "relay_on": self.relay_on,
            "simulate": self.simulate,
            "mode": "single",
        }


class JointControllerAdapter(ControllerAdapter):
    """Owns the full MicroDM (joint mode): one flat array over all controllers."""

    def __init__(
        self,
        dm: Any,
        ip_suffixes: list[int],
        ip_to_ctrl_idx: dict[int, int],
        dm_num: int,
        simulate: bool = False,
    ) -> None:
        self.dm = dm
        self.ip_suffixes = ip_suffixes
        self.ip_to_ctrl_idx = ip_to_ctrl_idx
        self.dm_num = dm_num
        self.simulate = simulate
        self.relay_on = False
        self._flat = np.zeros(dm_num, dtype=np.float64)
        self._lock = asyncio.Lock()

    async def set_relay(self, on: bool) -> bool:
        loop = asyncio.get_running_loop()

        def _blocking() -> bool:
            try:
                self.dm.set_relay_state(on)
                return True
            except Exception:  # noqa: BLE001
                return False

        ok = await loop.run_in_executor(None, _blocking)
        if ok:
            self.relay_on = bool(on)
        return ok

    async def send_voltages(self, voltages_by_ip: dict[int, np.ndarray]) -> dict[int, Exception]:
        flat = self._flat.copy()
        for ip_suffix, arr in voltages_by_ip.items():
            ctrl_idx = self.ip_to_ctrl_idx.get(int(ip_suffix))
            if ctrl_idx is None:
                continue
            start = ctrl_idx * SINGLE_CHANNELS
            if start >= flat.size:
                continue
            end = min(start + SINGLE_CHANNELS, flat.size)
            flat[start:end] = np.asarray(arr, dtype=np.float64)[: end - start]
        if not flat.any() and not voltages_by_ip:
            return {}
        errors: dict[int, Exception] = {}
        async with self._lock:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.dm.send_voltages, flat.copy(), 0.001)
                self._flat[:] = flat
            except Exception as exc:  # noqa: BLE001
                errors[0] = exc
        return errors

    async def send_flat(self, flat: np.ndarray) -> dict[int, Exception]:
        errors: dict[int, Exception] = {}
        async with self._lock:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.dm.send_voltages, np.asarray(flat, dtype=np.float64).copy(), 0.001)
                self._flat[:] = np.asarray(flat, dtype=np.float64)
            except Exception as exc:  # noqa: BLE001
                errors[0] = exc
        return errors

    async def read_voltages(self) -> dict[int, np.ndarray]:
        if not self.simulate:
            loop = asyncio.get_running_loop()
            try:
                flat = await loop.run_in_executor(None, self.dm.get_actuator_positions)
                self._flat[:] = np.asarray(flat, dtype=np.float64)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Joint readback failed: {exc}")
        return {ip: self._flat[i * SINGLE_CHANNELS : (i + 1) * SINGLE_CHANNELS].copy() for i, ip in enumerate(self.ip_suffixes)}

    async def close(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            try:
                if self.relay_on:
                    await loop.run_in_executor(None, self.dm.set_relay_state, False)
            except Exception:  # noqa: BLE001
                pass
            try:
                await loop.run_in_executor(None, self.dm.close)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Joint close failed: {exc}")

    def get_status(self) -> dict:
        return {
            "connected": True,
            "relay_on": self.relay_on,
            "simulate": self.simulate,
            "mode": "joint",
        }


class GroupControllerAdapter(ControllerAdapter):
    """Owns the controllers of one CSV-defined group (group mode)."""

    def __init__(self, group_name: str, controllers: dict[int, Any], simulate: bool = False) -> None:
        self.group_name = group_name
        self.controllers = controllers
        self.simulate = simulate
        self.relay_on = False
        self._voltages: dict[int, np.ndarray] = {
            int(ip): np.zeros(SINGLE_CHANNELS, dtype=np.float64) for ip in controllers
        }
        self._lock = asyncio.Lock()

    async def set_relay(self, on: bool) -> bool:
        async with self._lock:
            loop = asyncio.get_running_loop()
            results = await asyncio.gather(
                *(loop.run_in_executor(None, set_relay, ctrl, on) for ctrl in self.controllers.values())
            )
        if all(results):
            self.relay_on = bool(on)
        return all(results)

    async def send_voltages(self, voltages_by_ip: dict[int, np.ndarray]) -> dict[int, Exception]:
        errors: dict[int, Exception] = {}
        for ip_suffix, arr in voltages_by_ip.items():
            ctrl = self.controllers.get(int(ip_suffix))
            if ctrl is None:
                continue
            async with self._lock:
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None, self._send_one_blocking, ctrl, np.asarray(arr, dtype=np.float64)
                    )
                    self._voltages[int(ip_suffix)][:] = np.asarray(arr, dtype=np.float64)
                except Exception as exc:  # noqa: BLE001
                    errors[int(ip_suffix)] = exc
        return errors

    def _send_one_blocking(self, ctrl: Any, arr: np.ndarray) -> None:
        if not ctrl.set_all_voltage_array(arr.tolist()):
            raise ConnectionError("send failed")

    async def read_voltages(self) -> dict[int, np.ndarray]:
        return {int(ip): arr.copy() for ip, arr in self._voltages.items()}

    async def close(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            for ip_suffix, ctrl in self.controllers.items():
                await loop.run_in_executor(None, power_off_and_close, ctrl)

    def get_status(self) -> dict:
        return {
            "connected": True,
            "relay_on": self.relay_on,
            "simulate": self.simulate,
            "mode": "group",
            "group_name": self.group_name,
        }


# =============================================================================
# Service
# =============================================================================


class R50ControlService:
    """Command dispatcher + waveform engine + watchdog (runs in child process)."""

    def __init__(
        self,
        cmd_queue: Any,
        status_queue: Any,
        watchdog_timeout: float = 30.0,
    ) -> None:
        self.cmd_queue = cmd_queue
        self.status_queue = status_queue
        self.watchdog_timeout = watchdog_timeout
        self.single: SingleControllerAdapter | None = None
        self.joint: JointControllerAdapter | None = None
        self.group: GroupControllerAdapter | None = None
        self._waveform_task: asyncio.Task | None = None
        self._waveform_cfg: WaveformConfig | None = None
        self._last_command = time.time()
        self._running = True
        self._parent_pid = os.getppid()
        self._last_error = ""
        self._joint_matrix: np.ndarray | None = None
        self._pos_to_hw: dict[int, tuple[int, int]] = {}
        self._ip_to_ctrl_idx: dict[int, int] = {}
        self._joint_ip_suffixes: list[int] = []
        self._joint_dm_num = 0
        self._cv: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ run

    async def run(self) -> None:
        status_task = asyncio.create_task(self._status_pusher())
        watchdog_task = asyncio.create_task(self._watchdog_loop())
        parent_task = asyncio.create_task(self._parent_watchdog_loop())
        try:
            while self._running:
                try:
                    cmd: ServiceCommand = self.cmd_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.001)
                    continue
                self._last_command = time.time()
                await self._handle_command(cmd)
        finally:
            for task in (status_task, watchdog_task, parent_task):
                task.cancel()
            await self._shutdown()

    async def _handle_command(self, cmd: ServiceCommand) -> None:
        logger.debug(f"Command received: {cmd.action}")
        try:
            match cmd.action:
                case "connect_single":
                    await self._connect_single(cmd)
                case "disconnect_single":
                    await self._disconnect_single()
                case "connect_joint":
                    await self._connect_joint(cmd)
                case "disconnect_joint":
                    await self._disconnect_joint()
                case "connect_group":
                    await self._connect_group(cmd)
                case "disconnect_group":
                    await self._disconnect_group()
                case "set_relay":
                    await self._set_relay(cmd)
                case "waveform_start":
                    await self._waveform_start(cmd)
                case "waveform_stop":
                    await self._waveform_stop()
                case "set_voltage_direct":
                    await self._set_voltage_direct(cmd)
                case "set_joint_matrix":
                    await self._set_joint_matrix(cmd)
                case "refresh_from_hardware":
                    await self._refresh_from_hardware()
                case "ping_test":
                    self._ping_test(cmd)
                case "stop_service":
                    self._running = False
                case _:
                    logger.warning(f"Unknown command: {cmd.action}")
        except Exception as exc:  # noqa: BLE001 — service never crashes on a bad command
            self._last_error = str(exc)
            logger.exception(f"Command {cmd.action} failed: {exc}")

    # ------------------------------------------------------- connect/disconnect

    async def _connect_single(self, cmd: ServiceCommand) -> None:
        if self.single is not None:
            await self.single.close()
        loop = asyncio.get_running_loop()

        def _blocking() -> Any:
            return create_controller(
                controller_id=cmd.controller_id, ip=cmd.ip, port=int(cmd.port), simulate=cmd.simulate
            )

        ctrl = await loop.run_in_executor(None, _blocking)
        suffix = int(cmd.ip.rsplit(".", 1)[-1])
        self.single = SingleControllerAdapter(suffix, ctrl, simulate=cmd.simulate)
        logger.info(f"Connected single controller: {cmd.ip}:{cmd.port} simulate={cmd.simulate}")

    async def _disconnect_single(self) -> None:
        if self.single is not None:
            await self.single.close()
            self.single = None
        logger.info("Disconnected single controller")

    async def _connect_joint(self, cmd: ServiceCommand) -> None:
        if self.joint is not None:
            await self.joint.close()
        df = load_csv()
        if df.empty:
            self._last_error = "1300-5-enriched.csv 加载失败"
            raise RuntimeError(self._last_error)
        ip_suffixes = sorted(int(ip) for ip in df["IP组"].unique())
        loop = asyncio.get_running_loop()
        if cmd.simulate:
            dm: Any = await loop.run_in_executor(None, SimulatedMicroDM, ip_suffixes)
            await loop.run_in_executor(None, dm.open)
        else:
            dm = MicroDM(use_wiring_map=True)
            await loop.run_in_executor(None, dm.open)
        pos_to_hw = jc_build_wiring_index(df)
        ip_to_ctrl = jc_build_ip_index(df)
        dm_num = getattr(dm, "DM_Num", len(ip_suffixes) * SINGLE_CHANNELS)
        self.joint = JointControllerAdapter(dm, ip_suffixes, ip_to_ctrl, dm_num, simulate=cmd.simulate)
        self._pos_to_hw = pos_to_hw
        self._ip_to_ctrl_idx = ip_to_ctrl
        self._joint_ip_suffixes = ip_suffixes
        self._joint_dm_num = dm_num
        matrix = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        if not cmd.simulate:
            try:
                flat = await loop.run_in_executor(None, dm.get_actuator_positions)
                for physical_pos, (ip_s, pp) in pos_to_hw.items():
                    row = (physical_pos - 1) // GRID_SIZE
                    col = (physical_pos - 1) % GRID_SIZE
                    ctrl_idx = ip_to_ctrl.get(ip_s)
                    if ctrl_idx is not None:
                        idx = ctrl_idx * SINGLE_CHANNELS + (pp - 1)
                        if idx < flat.size:
                            matrix[row, col] = flat[idx]
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Joint initial readback failed: {exc}")
        self._joint_matrix = matrix
        logger.info(f"Connected MicroDM: {len(ip_suffixes)} controllers simulate={cmd.simulate}")

    async def _disconnect_joint(self) -> None:
        if self.joint is not None:
            await self.joint.close()
            self.joint = None
        self._joint_matrix = None
        self._joint_ip_suffixes = []
        logger.info("Disconnected MicroDM")

    async def _connect_group(self, cmd: ServiceCommand) -> None:
        if self.group is not None:
            await self.group.close()
        groups = build_groups()
        group_def = groups.get(cmd.group_name)
        if group_def is None:
            self._last_error = f"组别不存在: {cmd.group_name}"
            raise RuntimeError(self._last_error)
        loop = asyncio.get_running_loop()
        controllers: dict[int, Any] = {}
        for ip_suffix in sorted(group_def.channels_by_ip.keys()):
            ip = f"192.168.0.{ip_suffix}"

            def _blocking(_suffix: int = ip_suffix, _ip: str = ip) -> Any:
                return create_controller(controller_id=_suffix, ip=_ip, port=CFG.DEFAULT_PORT, simulate=cmd.simulate)

            try:
                controllers[int(ip_suffix)] = await loop.run_in_executor(None, _blocking)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Group connect failed for {ip}: {exc}")
        if not controllers:
            self._last_error = "组内所有控制器连接失败"
            raise RuntimeError(self._last_error)
        self.group = GroupControllerAdapter(cmd.group_name, controllers, simulate=cmd.simulate)
        logger.info(f"Connected group '{cmd.group_name}': {len(controllers)} controllers")

    async def _disconnect_group(self) -> None:
        if self.group is not None:
            await self.group.close()
            self.group = None
        logger.info("Disconnected group")

    # ------------------------------------------------------------ relay

    async def _set_relay(self, cmd: ServiceCommand) -> None:
        mode = cmd.payload.get("mode", "all")
        adapters: list[ControllerAdapter] = []
        if mode in ("single", "all") and self.single is not None:
            adapters.append(self.single)
        if mode in ("joint", "all") and self.joint is not None:
            adapters.append(self.joint)
        if mode in ("group", "all") and self.group is not None:
            adapters.append(self.group)
        if not adapters:
            self._last_error = "没有已连接的控制器可操作继电器"
            return
        for adapter in adapters:
            await adapter.set_relay(cmd.relay_on)
        logger.info(f"Relay {'ON' if cmd.relay_on else 'OFF'} for {mode}")

    # ------------------------------------------------------------ waveform

    async def _waveform_start(self, cmd: ServiceCommand) -> None:
        if cmd.waveform is None:
            self._last_error = "waveform_start 缺少 waveform 配置"
            return
        if self._waveform_task is not None and not self._waveform_task.done():
            await self._waveform_stop()
        cfg = cmd.waveform
        self._waveform_cfg = cfg
        self._waveform_task = asyncio.create_task(self._waveform_loop(cfg))
        logger.info(f"Waveform started: {cfg.type.name} targets={len(cfg.targets)}")

    async def _waveform_stop(self) -> None:
        if self._waveform_task is None or self._waveform_task.done():
            self._waveform_task = None
            return
        task = self._waveform_task
        cfg = self._waveform_cfg
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._waveform_task = None
        self._waveform_cfg = None
        if cfg is not None and cfg.targets:
            await self._send_targets_zero(cfg.targets)
        logger.info("Waveform stopped, 0V sent to targets")

    async def _waveform_loop(self, cfg: WaveformConfig) -> None:
        t0 = time.time()
        n = 0
        try:
            while True:
                t = time.time() - t0
                voltage = WaveformEngine.compute(cfg, t)
                await self._send_targets_voltage(cfg.targets, voltage)
                n += 1
                next_tick = t0 + n * cfg.dt
                delay = next_tick - time.time()
                if delay > 0:
                    await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"波形异常: {exc}"
            logger.exception(f"Waveform loop crashed: {exc}")

    async def _send_targets_voltage(self, targets: list[tuple[int, int]], voltage: float) -> None:
        by_ip: dict[int, list[int]] = {}
        for ip_suffix, pp in targets:
            by_ip.setdefault(int(ip_suffix), []).append(int(pp))
        sends: dict[int, np.ndarray] = {}
        for ip_suffix, positions in by_ip.items():
            arr = np.zeros(SINGLE_CHANNELS, dtype=np.float64)
            base = self._current_array(ip_suffix)
            arr[:] = base
            for pp in positions:
                if 1 <= pp <= SINGLE_CHANNELS:
                    arr[pp - 1] = voltage
            sends[int(ip_suffix)] = arr
        if not sends:
            return
        await self._dispatch_sends(sends)
        for ip_suffix, arr in sends.items():
            self._set_current_array(int(ip_suffix), arr)

    async def _send_targets_zero(self, targets: list[tuple[int, int]]) -> None:
        await self._send_targets_voltage(targets, 0.0)

    def _current_array(self, ip_suffix: int) -> np.ndarray:
        return self._cv.get(int(ip_suffix), np.zeros(SINGLE_CHANNELS, dtype=np.float64))

    def _set_current_array(self, ip_suffix: int, arr: np.ndarray) -> None:
        self._cv[int(ip_suffix)] = np.asarray(arr, dtype=np.float64)

    async def _dispatch_sends(self, sends: dict[int, np.ndarray]) -> None:
        if self.single is not None:
            own = {k: v for k, v in sends.items() if k == self.single.ip_suffix}
            if own:
                await self.single.send_voltages(own)
        if self.group is not None:
            own = {k: v for k, v in sends.items() if k in self.group.controllers}
            if own:
                await self.group.send_voltages(own)
        if self.joint is not None:
            own = {k: v for k, v in sends.items() if k in self._ip_to_ctrl_idx}
            if own:
                await self.joint.send_voltages(own)

    # ------------------------------------------------------------ direct sends

    async def _set_voltage_direct(self, cmd: ServiceCommand) -> None:
        if self._waveform_task is not None and not self._waveform_task.done():
            self._last_error = "波形运行中, 禁止直流下发 (先停止波形)"
            raise RuntimeError(self._last_error)
        if not cmd.targets:
            self._last_error = "未指定目标单元"
            raise RuntimeError(self._last_error)
        sends: dict[int, np.ndarray] = {}
        v = clip_voltage(cmd.voltage)
        for ip_suffix, pp in cmd.targets:
            arr = sends.setdefault(int(ip_suffix), self._current_array(int(ip_suffix)).copy())
            if 1 <= pp <= SINGLE_CHANNELS:
                arr[pp - 1] = v
        await self._dispatch_sends(sends)
        for ip_suffix, arr in sends.items():
            self._set_current_array(int(ip_suffix), arr)
        logger.info(f"Direct voltage {cmd.voltage}V sent to {len(cmd.targets)} targets")

    async def _set_joint_matrix(self, cmd: ServiceCommand) -> None:
        if self.joint is None:
            self._last_error = "联合控制未连接"
            raise RuntimeError(self._last_error)
        matrix = np.asarray(cmd.matrix, dtype=np.float64)
        if matrix.shape != (GRID_SIZE, GRID_SIZE):
            self._last_error = f"矩阵形状错误: {matrix.shape}"
            raise RuntimeError(self._last_error)
        self._joint_matrix = matrix.copy()
        flat = jc_matrix_to_flat(matrix, self._pos_to_hw, self._ip_to_ctrl_idx, self._joint_dm_num)
        await self.joint.send_flat(flat)
        for i, ip in enumerate(self._joint_ip_suffixes):
            start = i * SINGLE_CHANNELS
            self._set_current_array(int(ip), flat[start : start + SINGLE_CHANNELS])
        logger.info("Joint matrix applied")

    async def _refresh_from_hardware(self) -> None:
        if self.joint is not None and not self.joint.simulate:
            voltages = await self.joint.read_voltages()
            for ip, arr in voltages.items():
                self._set_current_array(int(ip), arr)
            self._last_error = "已从硬件刷新"
        elif self.single is not None and not self.single.simulate:
            voltages = await self.single.read_voltages()
            for ip, arr in voltages.items():
                self._set_current_array(int(ip), arr)

    def _ping_test(self, cmd: ServiceCommand) -> None:
        from ao_shaping.gui.streamlit_helper.r50_controller.r50_connection import ping_reachable

        self._last_error = f"ping {cmd.ip} -> {'可达' if ping_reachable(cmd.ip, timeout=1.0) else '不可达'}"

    # ------------------------------------------------------------ watchdogs

    async def _watchdog_loop(self) -> None:
        while self._running:
            await asyncio.sleep(WATCHDOG_CHECK_INTERVAL)
            if self._waveform_task is None or self._waveform_task.done():
                continue
            idle = time.time() - self._last_command
            if idle > self.watchdog_timeout:
                logger.warning(f"Watchdog triggered after {idle:.0f}s idle, stopping waveform")
                self._last_error = "看门狗: 长时间无命令, 已停止波形并下电"
                await self._waveform_stop()
                await self._set_relay(ServiceCommand(action="set_relay", relay_on=False, payload={"mode": "all"}))

    async def _parent_watchdog_loop(self) -> None:
        while self._running:
            await asyncio.sleep(PARENT_CHECK_INTERVAL)
            if os.getppid() != self._parent_pid:
                logger.warning("Parent process died, shutting down service safely")
                self._last_error = "UI 进程退出, 服务安全关闭"
                self._running = False

    # ------------------------------------------------------------ status

    async def _status_pusher(self) -> None:
        while self._running:
            await asyncio.sleep(STATUS_PUSH_INTERVAL)
            status = self._build_status()
            try:
                self.status_queue.put(status, block=False)
            except queue.Full:
                try:
                    self.status_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.status_queue.put(status, block=False)
                except queue.Full:
                    pass

    def _build_status(self) -> ServiceStatus:
        controllers: dict[int, dict] = {}
        current: dict[int, list[float]] = {}
        if self.single is not None:
            controllers[self.single.ip_suffix] = self.single.get_status()
            current[self.single.ip_suffix] = self.single._voltages.tolist()
        if self.joint is not None:
            for ip in self.joint.ip_suffixes:
                controllers[int(ip)] = self.joint.get_status()
            for i, ip in enumerate(self.joint.ip_suffixes):
                start = i * SINGLE_CHANNELS
                current[int(ip)] = self.joint._flat[start : start + SINGLE_CHANNELS].tolist()
        if self.group is not None:
            for ip in self.group.controllers:
                controllers[int(ip)] = self.group.get_status()
                current[int(ip)] = self.group._voltages[int(ip)].tolist()
        matrix = None
        if self._joint_matrix is not None:
            matrix = self._joint_matrix.tolist()
        running = self._waveform_task is not None and not self._waveform_task.done()
        return ServiceStatus(
            controllers=controllers,
            joint_connected=self.joint is not None,
            joint_relay_on=bool(self.joint and self.joint.relay_on),
            joint_controller_count=len(self._joint_ip_suffixes),
            group_connected=self.group is not None,
            group_relay_on=bool(self.group and self.group.relay_on),
            group_name=self.group.group_name if self.group else "",
            waveform_running=running,
            waveform_type=self._waveform_cfg.type.name if self._waveform_cfg and running else None,
            current_voltages=current,
            joint_matrix=matrix,
            last_error=self._last_error,
            timestamp=time.time(),
        )

    # ------------------------------------------------------------ shutdown

    async def _shutdown(self) -> None:
        if self._waveform_task is not None and not self._waveform_task.done():
            await self._waveform_stop()
        for adapter in (self.single, self.joint, self.group):
            if adapter is not None:
                try:
                    await adapter.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Shutdown close failed: {exc}")
        logger.info("R50ControlService shutdown complete")


# =============================================================================
# Process entry point
# =============================================================================


def start_service(watchdog_timeout: float = 30.0) -> tuple[Any, Any, SpawnProcess]:
    """Spawn the control service child process; returns (cmd_queue, status_queue, proc)."""
    ctx = mp.get_context("spawn")
    cmd_q = ctx.Queue()
    status_q = ctx.Queue(maxsize=4)
    proc = ctx.Process(
        target=_service_main,
        args=(cmd_q, status_q, watchdog_timeout),
        daemon=True,
        name="r50-control-service",
    )
    proc.start()
    logger.info(f"R50ControlService spawned, PID={proc.pid}")
    return cmd_q, status_q, proc


def _service_main(cmd_q: Any, status_q: Any, watchdog_timeout: float) -> None:
    logger.info(f"R50ControlService started, PID={os.getpid()}")
    service = R50ControlService(cmd_q, status_q, watchdog_timeout=watchdog_timeout)
    try:
        asyncio.run(service.run())
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"R50ControlService crashed: {exc}")
    logger.info("R50ControlService stopped")
