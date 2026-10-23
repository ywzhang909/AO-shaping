"""V2.0 非理想物理封装 (NonIdealModule) 的单元测试 + 比特级回归。

覆盖报告 ``report/shape_trials/sim_vs_hw/sim_vs_hw_20261022.md`` 第 7 节列出的器件级
非理想效应，以及「默认 = identity = 与改动前逐位相同」的纪律：

* :mod:`ao_shaping.drivers.sim.nonideal` 的纯模型单测
  (``SlmLutModel`` / ``ZeroOrderModel`` / ``CameraNonIdealModel`` / ``build_nonideal_bundle``)。
* 这些模型接入 :class:`SimPibSystem` 后的行为
  (零级晕、LUT 往返量化、PRNU/暗场 FPN 在量化前的固定模式、远场各向异性补零)。
* 默认/identity 旋钮 ⇒ 远场与未接入非理想模型时逐位相同 (既有调用方不被破坏)。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ao_shaping.drivers.sim.nonideal import (
    SLM_TWO_PI_GRAY,
    CameraNonIdealModel,
    SlmLutModel,
    ZeroOrderModel,
    build_nonideal_bundle,
)
from ao_shaping.drivers.sim.slm_pib_sim import (
    FAR_FIELD_PADDING,
    SimNoiseSpec,
    SimPibSystem,
    SimSLMPib,
)

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# 小面板形状 —— 让 FFT 在单测里足够快 (远场节距按公式用同样的形状反算, 与尺寸无关)
# ---------------------------------------------------------------------------
PANEL = (48, 48)


def _random_phase(shape, seed=0) -> np.ndarray:
    """确定性的随机相位 (不取 flat, 否则零级晕会被峰值归一掩盖)。"""
    return np.random.default_rng(seed).random(shape) * TWO_PI


def _center_fraction(ff: np.ndarray) -> float:
    """远场中心像素能量 / 总能量 (零级晕的直接度量)。"""
    ff = np.asarray(ff, dtype=np.float64)
    cy, cx = ff.shape[0] // 2, ff.shape[1] // 2
    return float(ff[cy, cx] / ff.sum())


# ===========================================================================
# A. nonideal.py 纯模型单测
# ===========================================================================


class TestSlmLutModel:
    def test_identity_default_is_linear_no_table(self):
        m = SlmLutModel()
        assert m.gamma == 1.0
        assert m.lut_table is None
        assert m.is_identity() is True

    def test_gamma_one_quantizes_exact_phase(self):
        """γ=1 无表: 仍走 2π/993 量化, 不在灰阶上的点被 round 到最近灰阶。"""
        m = SlmLutModel(gamma=1.0)
        # 一个不在灰阶上的相位 → 量化后偏离原值
        off = m.actual_phase(np.array([0.003]))
        assert off[0] != pytest.approx(0.003)
        # 精确落在灰阶 g=100 上的相位 → 往返后回到原值
        at_gray = TWO_PI * 100.0 / SLM_TWO_PI_GRAY
        back = m.actual_phase(np.array([at_gray]))
        assert back[0] == pytest.approx(at_gray, abs=1e-9)

    def test_gamma_two_is_nonlinear(self):
        m = SlmLutModel(gamma=2.0)
        assert m.is_identity() is False
        phase = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, TWO_PI])
        out = m.actual_phase(phase)
        gmax = float(SLM_TWO_PI_GRAY)
        for p, o in zip(phase, out):
            gray = int(np.round(np.mod(p, TWO_PI) / TWO_PI * gmax))
            frac = gray / gmax
            assert o == pytest.approx(TWO_PI * (frac**2), abs=1e-9)

    def test_lut_table_lookup_path(self):
        # 表[i] = 2π·i/993 ⇒ 在灰阶上等价于理想线性; 验证走查表而非 gamma 幂律
        table = np.linspace(0.0, TWO_PI, SLM_TWO_PI_GRAY + 1)
        m = SlmLutModel(lut_table=table)
        assert m.is_identity() is False  # 有表 ⇒ 不再是 identity
        phase = np.array([0.0, TWO_PI * 37.0 / SLM_TWO_PI_GRAY, TWO_PI * 0.5, TWO_PI])
        out = m.actual_phase(phase)
        gmax = SLM_TWO_PI_GRAY
        for p, o in zip(phase, out):
            gray = int(np.clip(np.round(np.mod(p, TWO_PI) / TWO_PI * gmax), 0, gmax))
            assert o == pytest.approx(table[gray], abs=1e-9)

    def test_lut_table_wrong_length_raises(self):
        m = SlmLutModel(lut_table=np.zeros(10))
        with pytest.raises(ValueError):
            m.actual_phase(np.array([0.0]))

    def test_build_nonideal_bundle_builds_lut_from_gamma(self):
        b = build_nonideal_bundle(slm_lut_gamma=2.0)
        assert isinstance(b["slm_lut"], SlmLutModel)
        assert b["slm_lut"].gamma == 2.0
        assert b["zero_order_eta"] == 0.0

    def test_build_nonideal_bundle_builds_lut_from_table(self):
        table = np.linspace(0.0, TWO_PI, SLM_TWO_PI_GRAY + 1)
        b = build_nonideal_bundle(slm_lut_table=table)
        assert b["slm_lut"] is not None
        assert np.array_equal(b["slm_lut"].lut_table, table)


class TestZeroOrderModel:
    def test_identity_at_zero_eta(self):
        z = ZeroOrderModel()
        assert z.eta == 0.0
        assert z.is_identity() is True

    def test_pupil_weights_math(self):
        z = ZeroOrderModel(eta=0.1)
        w_mod, w_zero = z.pupil_weights()
        assert w_mod == pytest.approx(math.sqrt(0.9))
        assert w_zero == pytest.approx(math.sqrt(0.1))
        # 能量守恒: 两分量权重平方和为 1
        assert w_mod**2 + w_zero**2 == pytest.approx(1.0)
        # 负 eta 视为 identity
        assert ZeroOrderModel(eta=-0.01).is_identity() is True


class TestCameraNonIdealModel:
    def test_identity_returns_ones_and_base_dark(self):
        c = CameraNonIdealModel()
        assert c.is_identity() is True
        shape = (16, 16)
        assert np.all(c.prnu_map(shape) == 1.0)
        assert np.all(c.dark_map(shape, base_dark=7.0) == 7.0)

    def test_prnu_map_is_deterministic_and_centered(self):
        c = CameraNonIdealModel(prnu_sigma=0.02, seed=3)
        # 足够大的样本使均值/标准差的抽样误差远小于容差 (SE_mean ≈ 0.02/√N)
        shape = (128, 128)
        m1 = c.prnu_map(shape)
        m2 = c.prnu_map(shape)
        assert np.array_equal(m1, m2)  # 固定模式 (帧间恒定)
        assert m1.shape == shape
        assert m1.mean() == pytest.approx(1.0, abs=2e-3)
        assert m1.std() == pytest.approx(0.02, abs=2e-3)

    def test_dark_map_is_deterministic(self):
        c = CameraNonIdealModel(dark_fpn_sigma=1.5, seed=3)
        shape = (128, 128)
        d1 = c.dark_map(shape, base_dark=5.0)
        d2 = c.dark_map(shape, base_dark=5.0)
        assert np.array_equal(d1, d2)
        assert d1.mean() == pytest.approx(5.0, abs=5e-2)
        assert d1.std() == pytest.approx(1.5, abs=5e-2)

    def test_build_nonideal_bundle_identity_camera_is_none(self):
        b = build_nonideal_bundle()  # 全默认
        assert b["nonideal_camera"] is None
        # 给了非零参数 ⇒ 不再是 None
        b2 = build_nonideal_bundle(prnu_sigma=0.02)
        assert isinstance(b2["nonideal_camera"], CameraNonIdealModel)


# ===========================================================================
# B. SimPibSystem 接入后的行为
# ===========================================================================


class TestZeroOrderWiring:
    def test_zero_order_adds_central_pedestal(self):
        phase = _random_phase(PANEL, seed=11)
        sys0 = SimPibSystem(slm_shape=PANEL, zero_order_eta=0.0, seed=0)
        sys1 = SimPibSystem(slm_shape=PANEL, zero_order_eta=0.1, seed=0)
        sys0.set_phase_rad(phase)
        sys1.set_phase_rad(phase)
        f0 = sys0.far_field()
        f1 = sys1.far_field()
        c0 = _center_fraction(f0)
        c1 = _center_fraction(f1)
        # 双分量瞳面的未调制常数项在远场 0 级形成明显晕 ⇒ 中心占比显著上升
        # (单像素中心占比随面板尺寸/照明盘变化, 但相对 η=0 的比值恒为强信号)
        assert c1 > 10.0 * c0
        assert c1 > 1e-3


class TestLutRoundTripWiring:
    def test_lut_gamma_one_quantizes_via_create_phase(self):
        """显示给远场的是 LUT 往返后的相位, 即使 γ=1 也量化。"""
        cmd = np.full(PANEL, 0.123)  # 不在灰阶上
        sys_none = SimPibSystem(slm_shape=PANEL, slm_lut=None, seed=0)
        sys_lut = SimPibSystem(
            slm_shape=PANEL, slm_lut=SlmLutModel(gamma=1.0), seed=0
        )
        slm_none = SimSLMPib(system=sys_none)
        slm_lut = SimSLMPib(system=sys_lut)
        slm_none.create_phase_from_array(cmd)
        slm_lut.create_phase_from_array(cmd)
        # 经 LUT 往返后存下的相位 ≠ 命令相位 (量化误差)
        assert not np.array_equal(
            slm_none.last_applied_phase_rad, slm_lut.last_applied_phase_rad
        )
        # 且与 SlmLutModel.actual_phase 一致
        expected = SlmLutModel(gamma=1.0).actual_phase(cmd)
        assert np.allclose(slm_lut.last_applied_phase_rad, expected)

    def test_lut_none_is_bit_for_bit_command(self):
        cmd = _random_phase(PANEL, seed=5)
        sys_none = SimPibSystem(slm_shape=PANEL, slm_lut=None, seed=0)
        slm_none = SimSLMPib(system=sys_none)
        slm_none.create_phase_from_array(cmd)
        # 不经 LUT ⇒ 上膛相位 == 命令相位 (逐位)
        assert np.array_equal(slm_none.last_applied_phase_rad, cmd)

    def test_lut_gamma_two_departs_from_command(self):
        cmd = np.linspace(0.0, TWO_PI, PANEL[0] * PANEL[1]).reshape(PANEL)
        sys_lut = SimPibSystem(
            slm_shape=PANEL, slm_lut=SlmLutModel(gamma=2.0), seed=0
        )
        slm_lut = SimSLMPib(system=sys_lut)
        slm_lut.create_phase_from_array(cmd)
        assert not np.allclose(slm_lut.last_applied_phase_rad, cmd, atol=1e-3)
        expected = SlmLutModel(gamma=2.0).actual_phase(cmd)
        assert np.allclose(slm_lut.last_applied_phase_rad, expected)


class TestCameraNonIdealWiring:
    def test_prnu_is_fixed_pattern_before_quantization(self):
        """PRNU 是固定乘性图, 在量化前乘上去; 帧间恒定 (非白噪声)。"""
        spec = SimNoiseSpec(
            shot_k=0.0, read_sigma_counts=0.0, dark_level_counts=0.0, quantize_bits=None
        )
        sys_prnu = SimPibSystem(
            slm_shape=(32, 32),
            peak_counts_per_ms=100.0,
            noise_spec=spec,
            nonideal_camera=CameraNonIdealModel(prnu_sigma=0.02, seed=1),
            seed=0,
        )
        sys_id = SimPibSystem(
            slm_shape=(32, 32),
            peak_counts_per_ms=100.0,
            noise_spec=spec,
            nonideal_camera=None,
            seed=0,
        )
        flat = np.zeros((32, 32))
        sys_prnu.set_phase_rad(flat)
        sys_id.set_phase_rad(flat)
        c_prnu_1 = sys_prnu.counts_frame(1.0)
        c_prnu_2 = sys_prnu.counts_frame(1.0)
        c_id = sys_id.counts_frame(1.0)
        # 无随机噪声 ⇒ PRNU 帧逐位可复现
        assert np.array_equal(c_prnu_1, c_prnu_2)
        # 与理想相机不同 (PRNU 已乘上)
        assert not np.array_equal(c_prnu_1, c_id)
        # 在信号>0 处, c_prnu / c_id ≈ PRNU 图
        mask = c_id > 0
        prnu_ref = sys_prnu._prnu_map(c_id.shape)
        ratio = c_prnu_1[mask] / c_id[mask]
        assert np.corrcoef(ratio, prnu_ref[mask])[0, 1] > 0.99

    def test_dark_fpn_is_fixed_additive_pattern(self):
        """暗电平 FPN 是逐像元固定偏置, 在量化前加进去。"""
        spec = SimNoiseSpec(
            shot_k=0.0, read_sigma_counts=0.0, dark_level_counts=5.0, quantize_bits=None
        )
        sys_dark = SimPibSystem(
            slm_shape=(32, 32),
            peak_counts_per_ms=100.0,
            noise_spec=spec,
            nonideal_camera=CameraNonIdealModel(dark_fpn_sigma=1.5, seed=1),
            seed=0,
        )
        sys_id = SimPibSystem(
            slm_shape=(32, 32),
            peak_counts_per_ms=100.0,
            noise_spec=spec,
            nonideal_camera=None,
            seed=0,
        )
        flat = np.zeros((32, 32))
        sys_dark.set_phase_rad(flat)
        sys_id.set_phase_rad(flat)
        c_dark = sys_dark.counts_frame(1.0)
        c_id = sys_id.counts_frame(1.0)
        # 两者共享同一份 signal + 同一份全局 base_dark, 唯一差别是逐像元 FPN
        fpn_recovered = c_dark - c_id
        assert fpn_recovered.std() == pytest.approx(1.5, abs=0.4)
        assert fpn_recovered.std() > 0.5  # 确实注入了固定模式噪声


class TestFarFieldAnisotropy:
    def test_tuple_padding_normalizes_per_axis(self):
        sys = SimPibSystem(slm_shape=(120, 80), far_field_padding=(2, 4), seed=0)
        assert sys._pad_rows == 2
        assert sys._pad_cols == 4
        # property getter 返回原始 (未归一化的) 元组
        assert sys.far_field_padding == (2, 4)

    def test_anisotropic_pitch_matches_closed_form(self):
        h, w = 120, 80
        sys = SimPibSystem(slm_shape=(h, w), far_field_padding=(2, 4), seed=0)
        row_pitch, col_pitch = sys.far_field_pitch_um
        lam = sys.wavelength_nm * 1e-9
        pitch = sys.slm_pixel_um * 1e-6
        expected_row = lam * sys.focal_length_m / (h * 2 * pitch) * 1e6
        expected_col = lam * sys.focal_length_m / (w * 4 * pitch) * 1e6
        assert row_pitch == pytest.approx(expected_row, rel=1e-9)
        assert col_pitch == pytest.approx(expected_col, rel=1e-9)
        # 两轴节距不同 (各向异性)
        assert row_pitch != pytest.approx(col_pitch)

    def test_direct_assignment_renormalizes(self):
        sys = SimPibSystem(slm_shape=(64, 64), seed=0)
        sys.far_field_padding = 8  # 外部采集脚本的直接赋值形态
        assert sys._pad_rows == 8
        assert sys._pad_cols == 8

    def test_default_padding_is_module_constant(self):
        sys = SimPibSystem(slm_shape=(64, 64), seed=0)
        assert sys.far_field_padding == FAR_FIELD_PADDING


# ===========================================================================
# C. 默认/identity 纪律 —— 与改动前逐位相同
# ===========================================================================


class TestIdentityParity:
    def test_default_equals_explicit_identity(self):
        phase = _random_phase(PANEL, seed=7)
        sys_a = SimPibSystem(slm_shape=PANEL, seed=0)
        sys_b = SimPibSystem(
            slm_shape=PANEL,
            seed=0,
            zero_order_eta=0.0,
            slm_lut=None,
            nonideal_camera=None,
        )
        sys_a.set_phase_rad(phase)
        sys_b.set_phase_rad(phase)
        assert np.array_equal(sys_a.far_field(), sys_b.far_field())

    def test_build_nonideal_default_bundle_is_bit_for_bit(self):
        phase = _random_phase(PANEL, seed=9)
        bundle = build_nonideal_bundle()  # 全默认 ⇒ 全部 identity
        sys_a = SimPibSystem(slm_shape=PANEL, seed=0)
        sys_b = SimPibSystem(slm_shape=PANEL, seed=0, **bundle)
        sys_a.set_phase_rad(phase)
        sys_b.set_phase_rad(phase)
        assert np.array_equal(sys_a.far_field(), sys_b.far_field())

    def test_identity_far_field_is_deterministic(self):
        phase = _random_phase(PANEL, seed=13)
        sys = SimPibSystem(slm_shape=PANEL, seed=0)
        sys.set_phase_rad(phase)
        f1 = sys.far_field()
        f2 = sys.far_field()
        assert np.array_equal(f1, f2)
