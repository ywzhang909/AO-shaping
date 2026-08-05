"""R50ControlService command-handler tests (in-process, simulated hardware).

Drives the async service directly with ServiceCommand objects and asserts
adapter state, voltage tracking and safety behavior. No mp processes here —
process-level behavior is covered by test_integration.py.
"""

from __future__ import annotations

import asyncio
import queue
import time

import numpy as np
import pytest

from ao_shaping.gui.streamlit_helper.r50_controller.r50_channel_select import (
    CFG,
    SINGLE_CHANNELS,
)
from ao_shaping.gui.streamlit_helper.r50_controller.r50_control_service import (
    R50ControlService,
    ServiceCommand,
    WaveformConfig,
    WaveformType,
)


def _make_service() -> R50ControlService:
    return R50ControlService(queue.Queue(), queue.Queue())


async def _cmd(service: R50ControlService, action: str, **kw) -> None:
    await service._handle_command(ServiceCommand(action=action, **kw))


class TestSingleMode:
    def test_connect_disconnect_single(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "connect_single", ip="192.168.0.101", simulate=True)
            assert service.single is not None
            assert service.single.ip_suffix == 101
            await _cmd(service, "disconnect_single")
            assert service.single is None

        asyncio.run(_run())

    def test_direct_voltage_updates_tracking(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "connect_single", ip="192.168.0.101", simulate=True)
            await _cmd(service, "set_voltage_direct", voltage=12.0, targets=[(101, 3)])
            arr = service._current_array(101)
            assert arr[2] == pytest.approx(12.0)
            assert service.single is not None
            assert service.single._voltages[2] == pytest.approx(12.0)

        asyncio.run(_run())

    def test_direct_voltage_clips_to_hardware_range(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "connect_single", ip="192.168.0.101", simulate=True)
            await _cmd(service, "set_voltage_direct", voltage=999.0, targets=[(101, 1)])
            assert service._current_array(101)[0] == pytest.approx(CFG.HW_VOLTAGE_MAX)

        asyncio.run(_run())


class TestJointMode:
    def test_connect_joint_simulated(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "connect_joint", simulate=True)
            assert service.joint is not None
            assert service.joint.ip_suffixes
            assert service._joint_matrix is not None
            await _cmd(service, "disconnect_joint")
            assert service.joint is None

        asyncio.run(_run())

    def test_set_joint_matrix_applies_flat(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "connect_joint", simulate=True)
            matrix = np.zeros((36, 36), dtype=np.float64)
            matrix[0, 0] = 5.0
            await _cmd(service, "set_joint_matrix", matrix=matrix)
            assert service._joint_matrix is not None
            assert service._joint_matrix[0, 0] == pytest.approx(5.0)
            # flat array of first controller carries the value
            assert service.joint is not None
            assert service._current_array(service.joint.ip_suffixes[0]).max() >= 0.0
            await _cmd(service, "disconnect_joint")

        asyncio.run(_run())


class TestGroupMode:
    def test_connect_group_from_csv(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "connect_group", group_name="一组", simulate=True)
            assert service.group is not None
            assert service.group.group_name == "一组"
            assert service.group.controllers
            await _cmd(service, "disconnect_group")
            assert service.group is None

        asyncio.run(_run())

    def test_unknown_group_reports_error(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "connect_group", group_name="不存在", simulate=True)
            assert service.group is None
            assert service._last_error

        asyncio.run(_run())


class TestWaveform:
    def test_dc_waveform_ticks_and_stop_zeroes_targets(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "connect_single", ip="192.168.0.101", simulate=True)
            cfg = WaveformConfig(
                type=WaveformType.DC,
                targets=[(101, 5)],
                voltage=20.0,
                dt=0.01,
            )
            await _cmd(service, "waveform_start", waveform=cfg)
            assert service._waveform_task is not None
            await asyncio.sleep(0.08)
            assert not service._waveform_task.done()
            assert service._current_array(101)[4] == pytest.approx(20.0)
            await _cmd(service, "waveform_stop")
            assert service._waveform_task is None
            assert service._current_array(101)[4] == pytest.approx(0.0)

        asyncio.run(_run())

    def test_direct_send_blocked_while_waveform_running(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "connect_single", ip="192.168.0.101", simulate=True)
            cfg = WaveformConfig(
                type=WaveformType.SINE,
                targets=[(101, 1)],
                amp=5.0,
                offset=0.0,
                freq=1.0,
                dt=0.01,
            )
            await _cmd(service, "waveform_start", waveform=cfg)
            await _cmd(service, "set_voltage_direct", voltage=1.0, targets=[(101, 1)])
            assert "波形运行中" in service._last_error
            assert service._current_array(101)[0] != pytest.approx(1.0)
            await _cmd(service, "waveform_stop")

        asyncio.run(_run())


class TestWaveformDuration:
    def test_waveform_auto_stops_after_duration_and_zeroes_targets(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "connect_single", ip="192.168.0.101", simulate=True)
            cfg = WaveformConfig(
                type=WaveformType.DC,
                targets=[(101, 5)],
                voltage=20.0,
                dt=0.01,
                duration=0.15,
            )
            await _cmd(service, "waveform_start", waveform=cfg)
            assert service._waveform_task is not None
            await asyncio.sleep(0.05)
            assert service._waveform_task is not None and not service._waveform_task.done()
            assert service._current_array(101)[4] == pytest.approx(20.0)
            await asyncio.sleep(0.35)
            assert service._waveform_task is None
            assert service._waveform_cfg is None
            assert service._current_array(101)[4] == pytest.approx(0.0)

        asyncio.run(_run())

    def test_duration_zero_runs_until_stopped(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "connect_single", ip="192.168.0.101", simulate=True)
            cfg = WaveformConfig(
                type=WaveformType.DC,
                targets=[(101, 5)],
                voltage=20.0,
                dt=0.01,
                duration=0.0,
            )
            await _cmd(service, "waveform_start", waveform=cfg)
            await asyncio.sleep(0.25)
            assert service._waveform_task is not None and not service._waveform_task.done()
            await _cmd(service, "waveform_stop")
            assert service._waveform_task is None
            assert service._current_array(101)[4] == pytest.approx(0.0)

        asyncio.run(_run())

    def test_status_reports_duration_and_remaining(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "connect_single", ip="192.168.0.101", simulate=True)
            cfg = WaveformConfig(
                type=WaveformType.DC,
                targets=[(101, 1)],
                voltage=10.0,
                dt=0.02,
                duration=5.0,
            )
            await _cmd(service, "waveform_start", waveform=cfg)
            status = service._build_status()
            assert status.waveform_running is True
            assert status.waveform_duration == pytest.approx(5.0)
            assert 0.0 < status.waveform_remaining <= 5.0
            await _cmd(service, "waveform_stop")

        asyncio.run(_run())


class TestRelayAndLifecycle:
    def test_relay_all_missing_controllers_reports_error(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "set_relay", relay_on=True, payload={"mode": "all"})
            assert service._last_error

        asyncio.run(_run())

    def test_stop_service_flag(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "stop_service")
            assert service._running is False

        asyncio.run(_run())

    def test_unknown_action_does_not_crash(self) -> None:
        service = _make_service()

        async def _run() -> None:
            await _cmd(service, "no_such_action")

        asyncio.run(_run())
