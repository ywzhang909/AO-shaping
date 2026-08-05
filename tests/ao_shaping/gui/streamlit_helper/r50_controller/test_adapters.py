"""Controller adapter tests against the simulated hardware twins.

Adapters must work identically on SimulatedR50Controller / SimulatedMicroDM
(their send surface matches the real drivers), so no hardware is needed.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from ao_shaping.gui.streamlit_helper.r50_controller.r50_channel_select import (
    CFG,
    SINGLE_CHANNELS,
    jc_build_ip_index,
)
from ao_shaping.gui.streamlit_helper.r50_controller.r50_connection import (
    SimulatedMicroDM,
    SimulatedR50Controller,
)
from ao_shaping.gui.streamlit_helper.r50_controller.r50_control_service import (
    GroupControllerAdapter,
    JointControllerAdapter,
    SingleControllerAdapter,
)


def _single_adapter() -> tuple[SingleControllerAdapter, SimulatedR50Controller]:
    ctrl = SimulatedR50Controller(101, "192.168.0.101", CFG.DEFAULT_PORT)
    ctrl.open()
    return SingleControllerAdapter(101, ctrl, simulate=True), ctrl


def test_single_send_voltages_updates_hardware_copy() -> None:
    adapter, ctrl = _single_adapter()

    async def _run() -> None:
        errs = await adapter.send_voltages({101: np.linspace(0, 50, 50)})
        assert errs == {}
        assert ctrl.readback().tolist() == pytest.approx(list(np.linspace(0, 50, 50)))

    asyncio.run(_run())


def test_single_relay_toggle() -> None:
    adapter, _ = _single_adapter()

    async def _run() -> None:
        assert await adapter.set_relay(True) is True
        assert adapter.relay_on is True
        assert await adapter.set_relay(False) is True
        assert adapter.relay_on is False

    asyncio.run(_run())


def test_single_close_powers_off_relay() -> None:
    adapter, ctrl = _single_adapter()

    async def _run() -> None:
        await adapter.set_relay(True)
        await adapter.close()
        assert ctrl._relay_on is False
        assert ctrl.is_connected() is False

    asyncio.run(_run())


def _joint_adapter() -> tuple[JointControllerAdapter, SimulatedMicroDM, dict[int, int]]:
    ips = [101, 102, 103]
    dm = SimulatedMicroDM(ips)
    dm.open()
    ip_to_ctrl = jc_build_ip_index()
    adapter = JointControllerAdapter(dm, ips, ip_to_ctrl, len(ips) * SINGLE_CHANNELS, simulate=True)
    return adapter, dm, ip_to_ctrl


def test_joint_send_voltages_distributes_by_controller_index() -> None:
    adapter, dm, ip_to_ctrl = _joint_adapter()
    arr = np.full(SINGLE_CHANNELS, 3.0)

    async def _run() -> None:
        errs = await adapter.send_voltages({102: arr})
        assert errs == {}
        idx = ip_to_ctrl[102]
        assert dm._controllers[102].readback()[0] == pytest.approx(3.0)
        assert dm._controllers[101].readback()[0] == pytest.approx(0.0)
        assert (adapter._flat[idx * SINGLE_CHANNELS] == 3.0)

    asyncio.run(_run())


def test_joint_read_voltages_slices_per_ip() -> None:
    adapter, dm, _ = _joint_adapter()
    flat = np.arange(len(adapter.ip_suffixes) * SINGLE_CHANNELS, dtype=np.float64)

    async def _run() -> None:
        await adapter.send_flat(flat)
        vols = await adapter.read_voltages()
        assert vols[102][0] == pytest.approx(flat[1 * SINGLE_CHANNELS])
        assert vols[103][-1] == pytest.approx(flat[2 * SINGLE_CHANNELS + SINGLE_CHANNELS - 1])

    asyncio.run(_run())


def _group_adapter() -> tuple[GroupControllerAdapter, dict[int, SimulatedR50Controller]]:
    ctrls = {101: SimulatedR50Controller(101, "192.168.0.101", CFG.DEFAULT_PORT),
             102: SimulatedR50Controller(102, "192.168.0.102", CFG.DEFAULT_PORT)}
    for c in ctrls.values():
        c.open()
    adapter = GroupControllerAdapter("一组", ctrls, simulate=True)
    return adapter, ctrls


def test_group_send_voltages_targets_only_requested_ips() -> None:
    adapter, ctrls = _group_adapter()

    async def _run() -> None:
        errs = await adapter.send_voltages({101: np.full(SINGLE_CHANNELS, 9.0)})
        assert errs == {}
        assert ctrls[101].readback()[10] == pytest.approx(9.0)
        assert ctrls[102].readback()[10] == pytest.approx(0.0)

    asyncio.run(_run())


def test_group_relay_all_controllers() -> None:
    adapter, ctrls = _group_adapter()

    async def _run() -> None:
        assert await adapter.set_relay(True) is True
        assert all(c._relay_on for c in ctrls.values())
        assert adapter.relay_on is True

    asyncio.run(_run())
