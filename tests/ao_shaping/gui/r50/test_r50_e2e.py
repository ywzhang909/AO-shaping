"""W3-T9 e2e regression for the R50 controller rewrite.

Locks in the legacy bug fix: "多个控制器发送电压不能一次成功，要点击下发好几次才能全部发送到"
(one click must reach ALL controllers). Uses simulated controllers and verifies
delivery end-to-end via readback() across the connect -> relay -> single-unit ->
single-controller -> loops -> group -> joint -> disconnect flow.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from ao_shaping.gui.r50 import r50_channel_select as cs
from ao_shaping.gui.r50 import r50_connection as conn
from ao_shaping.gui.r50 import r50_voltage_send as vs


def _sim_controllers(ips: list[int]) -> dict[int, object]:
    return {
        ip: conn.create_controller(ip, f"192.168.0.{ip}", cs.CFG.DEFAULT_PORT, simulate=True)
        for ip in ips
    }


class TestFixtureSetup:
    def test_csv_loads_units_and_groups(self):
        df = cs.load_csv()
        units = cs.build_all_units(df)
        groups = cs.build_groups(df)
        assert len(df) > 0
        assert len(units) > 0
        assert len(units) <= cs.SINGLE_CHANNELS * len({u.ip_suffix for u in units})
        assert len(groups) >= 1
        assert set(groups) <= {"一组", "二组", "三组", "四组", "五组"}

    def test_joint_mapping_built(self):
        df = cs.load_csv()
        pos_to_hw = cs.jc_build_wiring_index(df)
        ip_to_ctrl = cs.jc_build_ip_index(df)
        assert len(pos_to_hw) > 0
        assert len(ip_to_ctrl) > 0


class TestConnectAndRelay:
    def test_connect_relay_disconnect(self):
        df = cs.load_csv()
        ips = sorted({u.ip_suffix for u in cs.build_all_units(df)})
        controllers = _sim_controllers(ips)
        try:
            assert len(controllers) == len(ips)
            assert all(conn.set_relay(c, True) for c in controllers.values())
            assert all(c.is_connected() for c in controllers.values())
        finally:
            for c in controllers.values():
                conn.power_off_and_close(c)
        assert all(not c.is_connected() for c in controllers.values())


class TestSingleUnitTab:
    def test_apply_units_via_controller_delivers(self):
        df = cs.load_csv()
        units = cs.build_all_units(df)
        sel = [units[0], units[3]]
        ctrl = conn.create_controller(sel[0].ip_suffix, f"192.168.0.{sel[0].ip_suffix}", cs.CFG.DEFAULT_PORT, simulate=True)
        try:
            result = vs.apply_units_via_controller(ctrl, np.zeros(cs.SINGLE_CHANNELS), sel, 30.0)
            assert result.ok >= 1
            assert result.fail == 0
            rb = ctrl.readback()
            for u in sel:
                assert abs(rb[u.payload_position - 1] - 30.0) < 1e-9
        finally:
            conn.power_off_and_close(ctrl)


class TestSingleControllerTab:
    def test_once_send_all_50_channels(self):
        df = cs.load_csv()
        ips = sorted({u.ip_suffix for u in cs.build_all_units(df)})
        ctrl = conn.create_controller(ips[0], f"192.168.0.{ips[0]}", cs.CFG.DEFAULT_PORT, simulate=True)
        try:
            sel_all = cs.ChannelSelection(all_mode=True, channels=[])
            _, result = vs.apply_single_controller(ctrl, np.zeros(cs.SINGLE_CHANNELS), sel_all, 40.0)
            assert result.ok == 1
            assert result.fail == 0
            assert np.allclose(ctrl.readback(), 40.0)
        finally:
            conn.power_off_and_close(ctrl)


class TestLoops:
    def _run_ticks(self, tick_fn, params, ticks: int = 4) -> threading.Thread:
        ctrl = conn.create_controller(101, "192.168.0.101", cs.CFG.DEFAULT_PORT, simulate=True)
        stop = threading.Event()
        q: queue.Queue = queue.Queue()
        th = vs.start_loop(tick_fn, ctrl, np.zeros(cs.SINGLE_CHANNELS), params, stop, q)
        time.sleep(0.05 * (ticks + 2))
        stop.set()
        th.join(timeout=2.0)
        return th, ctrl

    def test_hold_loop(self):
        sel_all = cs.ChannelSelection(all_mode=True, channels=[])
        th, ctrl = self._run_ticks(
            vs.hold_tick,
            {"selection": sel_all, "voltage": 50.0, "vmin": cs.HW_VOLTAGE_MIN, "vmax": cs.HW_VOLTAGE_MAX},
        )
        try:
            assert not th.is_alive()
            assert np.allclose(ctrl.readback(), 50.0)
        finally:
            conn.power_off_and_close(ctrl)

    def test_sine_loop(self):
        sel_all = cs.ChannelSelection(all_mode=True, channels=[])
        th, ctrl = self._run_ticks(
            vs.sine_tick,
            {"selection": sel_all, "offset": 40.0, "amp": 10.0, "freq": 2.0, "t0": time.time(), "vmin": cs.HW_VOLTAGE_MIN, "vmax": cs.HW_VOLTAGE_MAX},
        )
        try:
            assert not th.is_alive()
            rb = ctrl.readback()
            assert 0 < rb.max() <= 50.0
        finally:
            conn.power_off_and_close(ctrl)

    def test_alt_loop(self):
        sel_all = cs.ChannelSelection(all_mode=True, channels=[])
        th, ctrl = self._run_ticks(
            vs.alt_tick,
            {"selection": sel_all, "voltage": 60.0, "freq": 2.0, "t0": time.time(), "vmin": cs.HW_VOLTAGE_MIN, "vmax": cs.HW_VOLTAGE_MAX},
        )
        try:
            assert not th.is_alive()
            rb = ctrl.readback()
            assert all(abs(v - 0.0) < 1e-9 or abs(v - 60.0) < 1e-9 for v in rb)
        finally:
            conn.power_off_and_close(ctrl)


class TestGroupTabOneClickDelivery:
    def test_one_call_reaches_every_affected_controller(self):
        df = cs.load_csv()
        groups = cs.build_groups(df)
        ips = sorted({u.ip_suffix for u in cs.build_all_units(df)})
        controllers = _sim_controllers(ips)
        try:
            gname = sorted(groups)[0]
            gdef = groups[gname]
            all_payloads = sorted(gdef.all_payload_positions)
            result = vs.apply_group_controllers(controllers, gdef, all_payloads, 55.0, current_map={})
            affected = [ip for ip, chs in gdef.channels_by_ip.items() if any(c.payload_position in set(all_payloads) for c in chs)]
            assert result.fail == 0, result.failed_targets
            assert result.ok == len(affected)
            for ip in affected:
                rb = controllers[ip].readback()
                for ci in gdef.channels_by_ip[ip]:
                    assert abs(rb[ci.payload_position - 1] - 55.0) < 1e-9
        finally:
            for c in controllers.values():
                conn.power_off_and_close(c)


class TestJointTab:
    def test_joint_flat_send_across_controllers(self):
        df = cs.load_csv()
        units = cs.build_all_units(df)
        ips = sorted({u.ip_suffix for u in units})
        pos_to_hw = cs.jc_build_wiring_index(df)
        ip_to_ctrl = cs.jc_build_ip_index(df)
        dm = conn.SimulatedMicroDM(ips)
        dm.open()
        try:
            joint_units = [units[i] for i in range(0, len(units), max(1, len(units) // 6))][:6]
            matrix = np.zeros((cs.GRID_SIZE, cs.GRID_SIZE), dtype=np.float64)
            flat = cs.jc_matrix_to_flat(matrix, pos_to_hw, ip_to_ctrl, cs.CFG.DM_NUM_ACTUATORS)
            flat, result = vs.apply_joint(dm, flat, joint_units, 45.0, pos_to_hw=pos_to_hw, ip_to_ctrl=ip_to_ctrl)
            assert result.fail == 0
            assert result.ok == len(joint_units)
            for u in joint_units:
                rb = dm.readback(u.ip_suffix)
                payload_pos = pos_to_hw[u.physical_position][1]
                assert abs(rb[payload_pos - 1] - 45.0) < 1e-9
        finally:
            dm.close()
        assert not dm.is_connected()
