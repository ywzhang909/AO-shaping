"""R50 controller connection layer: simulated controllers, factory, power safety.

No top-level streamlit import. Hardware mode uses the real R50Controller /
MicroDM drivers; simulation mode provides twin classes with an identical send
surface plus readback() for tests and scripted self-verification.
"""

from __future__ import annotations

import socket
import subprocess
import sys
from typing import Any

import numpy as np

from loguru import logger

from ao_shaping.drivers.dm.MicroDM import DEFAULT_TIMEOUT, R50Controller
from ao_shaping.gui.streamlit_helper.r50_controller.r50_channel_select import CFG, SINGLE_CHANNELS


# =============================================================================
# Simulated Controllers
# =============================================================================


class SimulatedR50Controller:
    """In-memory R50Controller twin with the same send surface + readback().

    Logs every state change through loguru (so simulation activity is visible
    in the service console) and can expose a JSON HTTP API on localhost via
    :meth:`start_http_server` for external verification.
    """

    def __init__(self, controller_id: int, ip: str, port: int) -> None:
        self.controller_id = controller_id
        self.ip = ip
        self.port = port
        self._opened = False
        self._relay_on = False
        self._voltages = np.zeros(SINGLE_CHANNELS, dtype=np.float64)
        self._http: Any = None

    def open(self) -> bool:
        self._opened = True
        logger.info(f"[SIM {self.ip}] opened")
        return True

    def close(self) -> None:
        self.stop_http_server()
        self._opened = False
        logger.info(f"[SIM {self.ip}] closed")

    def is_connected(self) -> bool:
        return self._opened

    def set_relay(self, on: bool) -> bool:
        if not self._opened:
            return False
        self._relay_on = bool(on)
        logger.info(f"[SIM {self.ip}] relay -> {'ON' if on else 'OFF'}")
        return True

    def set_channel_voltage(self, channel: int, voltage: float) -> bool:
        if not self._opened or not (0 <= int(channel) < SINGLE_CHANNELS):
            return False
        self._voltages[int(channel)] = float(voltage)
        logger.debug(f"[SIM {self.ip}] ch{int(channel) + 1} -> {float(voltage):.2f}V")
        return True

    def set_all_channel_voltage(self, voltage: float) -> bool:
        if not self._opened:
            return False
        self._voltages[:] = float(voltage)
        logger.debug(f"[SIM {self.ip}] all channels -> {float(voltage):.2f}V")
        return True

    def set_all_voltage_array(self, voltages: list[float]) -> bool:
        if not self._opened:
            return False
        arr = np.asarray(voltages, dtype=np.float64)
        if arr.size != SINGLE_CHANNELS:
            return False
        self._voltages[:] = arr
        logger.debug(
            f"[SIM {self.ip}] array sent: min={arr.min():.2f} max={arr.max():.2f} mean={arr.mean():.2f}V"
        )
        return True

    def readback(self) -> np.ndarray:
        """Copy of current per-channel voltages (sim inspection / tests)."""
        return self._voltages.copy()

    def start_http_server(self, port: int | None = None) -> int:
        """Expose the simulation on a localhost JSON HTTP API; returns the port.

        Default port is ``18000 + ip_suffix`` (e.g. 192.168.0.101 -> 18101);
        falls back to an ephemeral port when the derived one is taken.
        """
        from ao_shaping.gui.streamlit_helper.r50_controller.r50_sim_http import SimHttpServer

        if self._http is not None:
            return self._http.port
        derived = 18000 + int(self.ip.rsplit(".", 1)[-1])
        self._http = SimHttpServer(self, port if port is not None else derived)
        return self._http.start()

    def stop_http_server(self) -> None:
        if self._http is not None:
            self._http.stop()
            self._http = None

    @property
    def http_port(self) -> int | None:
        """Bound HTTP API port, or None when the API is not running."""
        return self._http.port if self._http is not None else None


class SimulatedMicroDM:
    """In-memory MicroDM twin: flat array distributed 50 per controller."""

    def __init__(self, ips: list[int], timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self._relay_on = False
        self._controllers: dict[int, SimulatedR50Controller] = {
            int(ip): SimulatedR50Controller(int(ip), f"192.168.0.{ip}", CFG.DEFAULT_PORT)
            for ip in ips
        }

    def open(self) -> None:
        for ctrl in self._controllers.values():
            ctrl.open()

    def close(self) -> None:
        for ctrl in self._controllers.values():
            ctrl.close()

    def is_connected(self) -> bool:
        return all(ctrl.is_connected() for ctrl in self._controllers.values())

    def set_relay_state(self, state: bool) -> None:
        self._relay_on = bool(state)

    def send_voltages(self, vs: np.ndarray, wait_time_s: float = 0.001) -> np.ndarray:
        """Distribute a flat voltage array to per-controller 50-channel slots."""
        arr = np.asarray(vs, dtype=np.float64)
        for i, ctrl in enumerate(self._controllers.values()):
            start = i * SINGLE_CHANNELS
            if start >= arr.size:
                break
            chunk = arr[start : start + SINGLE_CHANNELS]
            if chunk.size < SINGLE_CHANNELS:
                padded = np.zeros(SINGLE_CHANNELS, dtype=np.float64)
                padded[: chunk.size] = chunk
                chunk = padded
            ctrl._voltages[:] = chunk
        return arr

    def readback(self, ip_suffix: int) -> np.ndarray:
        return self._controllers[int(ip_suffix)].readback()

    def start_http_servers(self) -> dict[int, int]:
        """Expose every inner controller on its own localhost JSON HTTP API.

        Returns ``{ip_suffix: port}``; ports derive from each controller's IP
        (18000 + suffix) with ephemeral fallback when taken.
        """
        return {int(ip): ctrl.start_http_server() for ip, ctrl in self._controllers.items()}

    def stop_http_servers(self) -> None:
        for ctrl in self._controllers.values():
            ctrl.stop_http_server()

    @property
    def http_ports(self) -> dict[int, int]:
        """Per-IP HTTP API ports, or an empty dict when not running."""
        return {int(ip): ctrl.http_port for ip, ctrl in self._controllers.items() if ctrl.http_port is not None}

    @property
    def ip_suffixes(self) -> list[int]:
        return sorted(self._controllers)


# =============================================================================
# Factory & Power Safety
# =============================================================================


def create_controller(
    controller_id: int,
    ip: str,
    port: int,
    simulate: bool,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    """Create a controller for the given mode.

    simulate=True -> opened SimulatedR50Controller.
    simulate=False -> real R50Controller; raises ConnectionError if open() fails.
    """
    if simulate:
        ctrl: Any = SimulatedR50Controller(controller_id, ip, port)
        ctrl.open()
        return ctrl
    ctrl = R50Controller(controller_id, ip, port, timeout=timeout)
    if not ctrl.open():
        raise ConnectionError(f"无法连接 R50 控制器 {ip}:{port}")
    return ctrl


def set_relay(ctrl: Any | None, on: bool) -> bool:
    """Set relay state; returns success (None-safe).

    R50Controller exposes ``set_relay``; MicroDM and SimulatedMicroDM expose
    ``set_relay_state`` (which returns None on success). Falls back so the
    power-off safety path works on both surfaces.
    """
    if ctrl is None:
        return False
    try:
        fn = getattr(ctrl, "set_relay", None)
        if fn is None:
            fn = getattr(ctrl, "set_relay_state", None)
        if fn is None:
            return False
        result = fn(on)
        return True if result is None else bool(result)
    except Exception as e:
        logger.warning(f"set_relay({on}) failed: {e}")
        return False


def power_off_and_close(ctrl: Any | None) -> None:
    """Relay OFF first, then close controller. None-safe, exception-safe."""
    if ctrl is None:
        return
    try:
        set_relay(ctrl, False)
    except Exception:
        pass
    try:
        ctrl.close()
    except Exception as e:
        logger.warning(f"close failed: {e}")


# =============================================================================
# Connectivity Probes
# =============================================================================


def tcp_reachable(ip: str, port: int, timeout: float = 1.0) -> bool:
    """Probe a TCP port; returns reachable or not."""
    try:
        with socket.create_connection((ip, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def ping_reachable(ip: str, timeout: float = 2.0) -> bool:
    """Ping a host via system ping; used by the connectivity test panel."""
    try:
        if sys.platform.startswith("win"):
            cmd = ["ping", "-n", "1", "-w", str(int(timeout * 1000)), ip]
        else:
            cmd = ["ping", "-c", "1", "-W", str(int(timeout)), ip]
        result = subprocess.run(cmd, capture_output=True, timeout=timeout + 2.0)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False
