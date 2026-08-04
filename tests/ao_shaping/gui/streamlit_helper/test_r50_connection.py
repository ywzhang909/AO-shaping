"""RED tests for r50_channel_select + r50_connection (pure logic, no streamlit).

Contract covered (Wave 1):
- S1 basis: ChannelInfo / ChannelSelection / CSV wiring index semantics
- Simulated controller readback (needed by every later sim e2e check)
- create_controller factory (sim + real-unreachable raises)
- power_off_and_close safety ordering
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ao_shaping.gui.streamlit_helper.r50_channel_select import (
    CFG,
    ChannelInfo,
    ChannelSelection,
    GroupDef,
    build_csv_index,
    build_groups,
    get_channel_info,
    jc_build_ip_index,
    jc_build_wiring_index,
    jc_matrix_to_flat,
    load_csv,
    row_to_channel_info,
)
from ao_shaping.gui.streamlit_helper.r50_connection import (
    SimulatedMicroDM,
    SimulatedR50Controller,
    create_controller,
    power_off_and_close,
    tcp_reachable,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_df() -> pd.DataFrame:
    """6-row synthetic CSV mirroring 1300-5-enriched.csv column layout."""
    return pd.DataFrame(
        {
            "位置序号": [109, 110, 74, 111, 112, 76],
            "36×36行": [3, 3, 2, 3, 3, 2],
            "36×36列": [0, 1, 1, 2, 3, 3],
            "IP组": [124, 111, 101, 101, 101, 101],
            "序号": [26, 16, 12, 3, 4, 5],
            "组": ["一组", "一组", "一组", "一组", "一组", "一组"],
            "引脚编号": [1, 2, 3, 4, 5, 6],
            "连接器": ["1-1-1", "1-1-3", "1-1-5", "1-1-7", "1-1-9", "1-1-11"],
        }
    )


# ---------------------------------------------------------------------------
# ChannelInfo / CSV index (r50_channel_select)
# ---------------------------------------------------------------------------

class TestChannelInfo:
    def test_row_to_channel_info_fields(self):
        ci = row_to_channel_info(_sample_df().iloc[0])
        assert ci.ip_suffix == 124
        assert ci.payload_position == 27          # 序号 26 (0-based) + 1
        assert ci.physical_position == 109        # 3*36 + 0 + 1
        assert ci.group == "一组"
        assert ci.needle_id == 1
        assert ci.physical_label == "1-1-1"

    def test_ip_property(self):
        ci = row_to_channel_info(_sample_df().iloc[0])
        assert ci.ip == "192.168.0.124"

    def test_short_info(self):
        ci = row_to_channel_info(_sample_df().iloc[0])
        assert ci.short_info() == "一组 针脚#1 (1-1-1)"

    def test_build_csv_index_keys(self):
        idx = build_csv_index(_sample_df())
        assert len(idx) == 6
        ci = idx[(101, 4)]                        # IP 101, 序号 3 → payload 4
        assert ci.physical_position == 111        # 3*36 + 2 + 1
        assert idx[(124, 27)].physical_label == "1-1-1"

    def test_build_csv_index_empty_df(self):
        assert build_csv_index(pd.DataFrame()) == {}

    def test_get_channel_info(self):
        idx = get_channel_info(101, 3)            # 0-based channel 3
        assert idx is not None
        assert idx.payload_position == 4

    def test_get_channel_info_missing(self):
        assert get_channel_info(999, 0) is None

    def test_load_csv_real_file(self):
        df = load_csv()
        assert not df.empty
        assert len(df) == 1296                   # 36x36 grid
        assert set(df["IP组"].unique()) == set(range(101, 127))


class TestChannelSelection:
    def test_empty(self):
        sel = ChannelSelection()
        assert sel.is_empty
        assert sel.normalized(50) == []

    def test_all_mode(self):
        sel = ChannelSelection(all_mode=True)
        assert not sel.is_empty
        assert sel.normalized(50) == list(range(50))

    def test_channels_dedup_sorted(self):
        sel = ChannelSelection(channels=[5, 1, 5])
        assert sel.normalized(50) == [1, 5]

    def test_select_all_keeps_all_mode_flag(self):
        sel = ChannelSelection()
        sel.select_all(50)
        assert not sel.all_mode
        assert sel.normalized(50) == list(range(50))

    def test_invert(self):
        sel = ChannelSelection(channels=[1, 2])
        sel.invert(5)
        assert sel.normalized(5) == [0, 3, 4]

    def test_invert_empty_becomes_all(self):
        sel = ChannelSelection()
        sel.invert(5)
        assert sel.normalized(5) == [0, 1, 2, 3, 4]


class TestGroups:
    def test_build_groups(self):
        df = _sample_df().copy()
        df["组"] = ["一组", "一组", "二组", "二组", "二组", "二组"]
        groups = build_groups(df)
        assert set(groups) == {"一组", "二组"}
        g1 = groups["一组"]
        assert isinstance(g1, GroupDef)
        assert g1.total_channels == 2
        assert set(g1.channels_by_ip) == {124, 111}
        assert g1.all_payload_positions == [17, 27]

    def test_build_groups_empty(self):
        assert build_groups(pd.DataFrame()) == {}


class TestJointMapping:
    def test_jc_build_wiring_index(self):
        pos_to_hw = jc_build_wiring_index(_sample_df())
        assert pos_to_hw[109] == (124, 27)
        assert pos_to_hw[111] == (101, 4)
        assert pos_to_hw[112] == (101, 5)

    def test_jc_build_ip_index_sorted(self):
        ip_idx = jc_build_ip_index(_sample_df())
        assert ip_idx == {101: 0, 111: 1, 124: 2}

    def test_jc_matrix_to_flat(self):
        matrix = np.arange(4, dtype=np.float64).reshape(2, 2) + 1
        pos_to_hw = {1: (101, 1), 2: (101, 2), 3: (102, 1), 4: (102, 2)}
        ip_to_ctrl = {101: 0, 102: 1}
        flat = jc_matrix_to_flat(matrix, pos_to_hw, ip_to_ctrl, dm_num=100)
        assert flat[0] == 1.0
        assert flat[1] == 2.0
        assert flat[50] == 3.0
        assert flat[51] == 4.0
        assert flat[99] == 0.0

    def test_jc_matrix_to_flat_clips_out_of_range(self):
        matrix = np.full((2, 2), 999.0)
        pos_to_hw = {1: (101, 1)}
        ip_to_ctrl = {101: 0}
        flat = jc_matrix_to_flat(matrix, pos_to_hw, ip_to_ctrl, dm_num=100)
        assert flat[0] == 999.0                    # raw value passes through (clip at send)


# ---------------------------------------------------------------------------
# Simulated controllers (r50_connection)
# ---------------------------------------------------------------------------

class TestSimulatedR50Controller:
    def test_open_close(self):
        ctrl = SimulatedR50Controller(1, "192.168.0.101", 10101)
        assert ctrl.open()
        assert ctrl.is_connected()
        ctrl.close()
        assert not ctrl.is_connected()

    def test_set_channel_voltage(self):
        ctrl = SimulatedR50Controller(1, "192.168.0.101", 10101)
        ctrl.open()
        assert ctrl.set_channel_voltage(3, 5.5)
        rb = ctrl.readback()
        assert rb[3] == 5.5
        assert rb.sum() == 5.5

    def test_set_all_channel_voltage(self):
        ctrl = SimulatedR50Controller(1, "192.168.0.101", 10101)
        ctrl.open()
        assert ctrl.set_all_channel_voltage(7.5)
        assert np.all(ctrl.readback() == 7.5)

    def test_set_all_voltage_array(self):
        ctrl = SimulatedR50Controller(1, "192.168.0.101", 10101)
        ctrl.open()
        arr = np.linspace(0, 10, 50)
        assert ctrl.set_all_voltage_array(arr.tolist())
        assert np.allclose(ctrl.readback(), arr)

    def test_readback_is_copy(self):
        ctrl = SimulatedR50Controller(1, "192.168.0.101", 10101)
        ctrl.open()
        ctrl.set_all_channel_voltage(3.0)
        rb = ctrl.readback()
        rb[:] = 999.0
        assert np.all(ctrl.readback() == 3.0)

    def test_set_relay(self):
        ctrl = SimulatedR50Controller(1, "192.168.0.101", 10101)
        ctrl.open()
        assert ctrl.set_relay(True)
        assert ctrl.set_relay(False)

    def test_closed_controller_rejects_ops(self):
        ctrl = SimulatedR50Controller(1, "192.168.0.101", 10101)
        assert not ctrl.set_channel_voltage(0, 1.0)
        assert not ctrl.set_all_voltage_array([0.0] * 50)


class TestSimulatedMicroDM:
    def test_open_close_relay(self):
        dm = SimulatedMicroDM(ips=[101, 102])
        dm.open()
        assert dm.is_connected()
        dm.set_relay_state(True)
        dm.close()
        assert not dm.is_connected()

    def test_send_voltages_distributes(self):
        dm = SimulatedMicroDM(ips=[101, 102])
        dm.open()
        vs = np.linspace(0, 99, 100)
        dm.send_voltages(vs)
        assert np.allclose(dm.readback(101), vs[:50])
        assert np.allclose(dm.readback(102), vs[50:])

    def test_send_voltages_rounds_partial_last(self):
        dm = SimulatedMicroDM(ips=[101, 102, 103])
        dm.open()
        vs = np.linspace(0, 129, 130)
        dm.send_voltages(vs)
        assert np.allclose(dm.readback(101), vs[:50])
        assert np.allclose(dm.readback(102), vs[50:100])
        rb = dm.readback(103)
        assert np.allclose(rb[:30], vs[100:])
        assert np.all(rb[30:] == 0.0)


class TestCreateController:
    def test_sim_mode(self):
        ctrl = create_controller(1, "192.168.0.101", 10101, simulate=True)
        assert isinstance(ctrl, SimulatedR50Controller)
        assert ctrl.is_connected()

    def test_real_unreachable_raises(self):
        with pytest.raises(ConnectionError):
            create_controller(1, "127.0.0.1", 1, simulate=False, timeout=1.0)


class TestPowerOffAndClose:
    def test_relay_off_then_close(self):
        ctrl = SimulatedR50Controller(1, "192.168.0.101", 10101)
        ctrl.open()
        ctrl.set_relay(True)
        ctrl.set_all_channel_voltage(10.0)
        power_off_and_close(ctrl)
        assert not ctrl.is_connected()
        assert ctrl._relay_on is False           # relay powered off before close
        ctrl.open()
        assert ctrl.is_connected()

    def test_none_safe(self):
        power_off_and_close(None)  # must not raise


class TestTcpReachable:
    def test_closed_port_false(self):
        assert tcp_reachable("127.0.0.1", 1, timeout=1.0) is False


class TestConfig:
    def test_constants(self):
        assert CFG.SINGLE_CHANNELS == 50
        assert CFG.GRID_SIZE == 36
        assert CFG.DM_NUM_ACTUATORS == 1296
        assert CFG.HW_VOLTAGE_MIN == -20.0
        assert CFG.HW_VOLTAGE_MAX == 120.0
        assert CFG.DEFAULT_PORT == 10101
