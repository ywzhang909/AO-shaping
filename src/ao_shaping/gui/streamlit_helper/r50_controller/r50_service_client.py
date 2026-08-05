"""R50ServiceClient — Streamlit-side wrapper over the control service queues.

Sends :class:`ServiceCommand` objects to the service process and polls
:class:`ServiceStatus` snapshots. Pure thin client: no hardware access, no
threading — the service process owns all IO.
"""

from __future__ import annotations

import queue

from loguru import logger

from ao_shaping.gui.streamlit_helper.r50_controller.r50_channel_select import CFG
from ao_shaping.gui.streamlit_helper.r50_controller.r50_control_service import (
    ServiceCommand,
    ServiceStatus,
    WaveformConfig,
)


class R50ServiceClient:
    """Command sender + status poller for one service process."""

    def __init__(self, cmd_queue: queue.Queue, status_queue: queue.Queue) -> None:
        self.cmd_queue = cmd_queue
        self.status_queue = status_queue

    def _send(self, action: str, **kwargs: object) -> None:
        self.cmd_queue.put(ServiceCommand(action=action, **kwargs))

    def connect_single(self, ip: str, port: int = CFG.DEFAULT_PORT, simulate: bool = False, controller_id: int = 1) -> None:
        self._send("connect_single", ip=ip, port=int(port), simulate=simulate, controller_id=controller_id)

    def disconnect_single(self) -> None:
        self._send("disconnect_single")

    def connect_joint(self, simulate: bool = False) -> None:
        self._send("connect_joint", simulate=simulate)

    def disconnect_joint(self) -> None:
        self._send("disconnect_joint")

    def connect_group(self, group_name: str, simulate: bool = False) -> None:
        self._send("connect_group", group_name=group_name, simulate=simulate)

    def disconnect_group(self) -> None:
        self._send("disconnect_group")

    def set_relay(self, on: bool, mode: str = "all") -> None:
        self._send("set_relay", relay_on=on, payload={"mode": mode})

    def start_waveform(self, cfg: WaveformConfig) -> None:
        self._send("waveform_start", waveform=cfg)

    def stop_waveform(self) -> None:
        self._send("waveform_stop")

    def set_voltage_direct(self, voltage: float, targets: list[tuple[int, int]]) -> None:
        self._send("set_voltage_direct", voltage=float(voltage), targets=list(targets))

    def set_joint_matrix(self, matrix: object) -> None:
        self._send("set_joint_matrix", matrix=matrix)

    def refresh(self) -> None:
        self._send("refresh_from_hardware")

    def ping_test(self, ip: str) -> None:
        self._send("ping_test", ip=ip)

    def stop_service(self) -> None:
        self._send("stop_service")

    def poll_status(self) -> ServiceStatus | None:
        """Consume all pending status messages and return the newest one.

        Drops stale messages when more than two have accumulated so a slow UI
        never falls arbitrarily far behind the live hardware state.
        """
        last: ServiceStatus | None = None
        count = 0
        while True:
            try:
                last = self.status_queue.get_nowait()
                count += 1
            except queue.Empty:
                break
        if count > 2 and last is not None:
            logger.debug(f"poll_status dropped {count - 1} stale messages")
        return last
