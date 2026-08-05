"""End-to-end integration test: real spawned service process.

Starts the actual control service via start_service() (spawn context),
drives it through the client and verifies status flow and safe shutdown.
All hardware is simulated.
"""

from __future__ import annotations

import time

import pytest

from ao_shaping.gui.streamlit_helper.r50_controller.r50_control_service import (
    ServiceStatus,
    WaveformConfig,
    WaveformType,
    start_service,
)
from ao_shaping.gui.streamlit_helper.r50_controller.r50_service_client import R50ServiceClient


def _wait_status(client: R50ServiceClient, predicate, timeout: float = 10.0) -> ServiceStatus:
    deadline = time.time() + timeout
    last: ServiceStatus | None = None
    while time.time() < deadline:
        status = client.poll_status()
        if status is not None:
            last = status
            if predicate(status):
                return status
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for status; last={last}")


def test_service_lifecycle_and_simulated_control() -> None:
    cmd_q, status_q, proc = start_service()
    try:
        assert proc.is_alive()
        client = R50ServiceClient(cmd_q, status_q)

        client.connect_single("192.168.0.101", simulate=True)
        status = _wait_status(client, lambda s: 101 in s.controllers)
        assert status.controllers[101]["simulate"] is True

        client.set_voltage_direct(12.0, [(101, 3)])
        status = _wait_status(
            client, lambda s: s.current_voltages.get(101, [0])[2] == pytest.approx(12.0)
        )
        assert status.current_voltages[101][2] == pytest.approx(12.0)

        client.start_waveform(
            WaveformConfig(type=WaveformType.DC, targets=[(101, 3)], voltage=20.0, dt=0.01)
        )
        status = _wait_status(client, lambda s: s.waveform_running)
        assert status.waveform_running is True

        client.stop_waveform()
        status = _wait_status(client, lambda s: not s.waveform_running)
        assert status.waveform_running is False
        assert status.current_voltages[101][2] == pytest.approx(0.0)

        client.stop_service()
        proc.join(timeout=10.0)
        assert not proc.is_alive()
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)


def test_respawn_after_process_death() -> None:
    cmd_q, status_q, proc = start_service()
    client = R50ServiceClient(cmd_q, status_q)
    client.stop_service()
    proc.join(timeout=10.0)
    assert not proc.is_alive()

    cmd_q2, status_q2, proc2 = start_service()
    try:
        client2 = R50ServiceClient(cmd_q2, status_q2)
        assert proc2.is_alive()
        client2.connect_single("192.168.0.102", simulate=True)
        _wait_status(client2, lambda s: 102 in s.controllers)
    finally:
        client2.stop_service()
        proc2.join(timeout=10.0)
