"""V2.0 非理想物理封装 (NonIdealModule) 单元测试。

覆盖:
* ``nonideal.py`` 三个默认 identity 的器件级模型;
* 它们接入 ``SimPibSystem`` / ``SimSLMPib`` 的行为
  (零级晕, LUT 量化往返, PRNU/暗 FPN 固定图, 各向异性补零);
* 默认/identity ⇒ 与改动前逐位相同的纪律。

为速度, 测试用 64×64 的小面板 (真实面板 1920×1200); 物理与节距公式与尺寸无关。
"""
from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.sim.nonideal import (
    SlmLutModel,
    ZeroOrderModel,
    CameraNonIdealModel,
    build_nonideal_bundle,
)
from ao_shaping.drivers.sim.slm_pib_sim import (
    SimPibSystem,
    SimSLMPib,
    reset_system,
)

PANEL = (64, 64)


def _make_system(**kw):
    """构造一个小面板 system (快), 其余口径用对照实验的典型值。"""
    base = dict(
        slm_shape=PANEL,
        seed=0,
        far_field_padding=4,
        peak_counts_per_ms=133.5,
        camera_defocus_m=0.010084735,
        counts_reference="flat",
        beam_w0=20.0,
    )
    base.update(kw)
    return SimPibSystem(**base)


# --------------------------------------------------------------------------
# nonideal.py 纯模型
# --------------------------------------------------------------------------

def test_slm_lut_gamma1_still_quantizes():
    lut = SlmLutModel(gamma=1.0)
    phase = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    displayed = lut.actual_phase(phase)
    # γ=1 仍是量化往返: 不在 2π/993 格点上的相位会被 round 到格点。
    gray = np.round(np.mod(phase, 2 * np.pi) / (2 * np.pi) * lut.two_pi_gray).astype(int)
    expected = (2 * np.pi) * (gray / lut.two_pi_gray)
    assert np.allclose(displayed, expected, atol=1e-9)


def test_slm_lut_gamma2_nonlinear():
    lut = SlmLutModel(gamma=2.0)
    phase = np.full((8, 8), 1.0)
    flat = lut.actual_phase(phase)
    # γ=2 幂律: 同一相位, 不同 γ 结果不同。
    assert not np.allclose(flat, SlmLutModel(gamma=1.0).actual_phase(phase))


def test_slm_lut_table_path_and_length_check():
    gmax = 993
    table = (2 * np.pi) * (np.arange(gmax + 1) / gmax)
    lut = SlmLutModel(lut_table=table)
    assert lut.is_identity() is False
    out = lut.actual_phase(np.array([0.0, np.pi]))
    assert out.shape == (2,)
    # 长度不符应抛 ValueError
    with pytest.raises(ValueError):
        SlmLutModel(lut_table=table[:10]).actual_phase(np.array([0.0]))


def test_zero_order_weights_energy_conserving():
    zo = ZeroOrderModel(eta=0.1)
    w_mod, w_zero = zo.pupil_weights()
    assert abs(w_mod**2 + w_zero**2 - 1.0) < 1e-12
    assert abs(w_zero - np.sqrt(0.1)) < 1e-12


def test_zero_order_identity_at_zero():
    assert ZeroOrderModel(eta=0.0).is_identity()
    assert not ZeroOrderModel(eta=0.1).is_identity()


def test_prnu_map_deterministic_centered():
    cam = CameraNonIdealModel(prnu_sigma=0.02, seed=3)
    shape = (24, 24)
    m1 = cam.prnu_map(shape)
    m2 = cam.prnu_map(shape)
    assert np.array_equal(m1, m2)  # 固定模式 (帧间恒定)
    assert m1.shape == shape
    # 576 样本下, 样本均值/标准差的涨落分别约 0.0008 / 0.0006
    assert m1.mean() == pytest.approx(1.0, abs=3e-3)
    assert m1.std() == pytest.approx(0.02, abs=4e-3)


def test_dark_map_deterministic():
    cam = CameraNonIdealModel(dark_fpn_sigma=1.5, seed=3)
    shape = (24, 24)
    d1 = cam.dark_map(shape, base_dark=5.0)
    d2 = cam.dark_map(shape, base_dark=5.0)
    assert np.array_equal(d1, d2)
    assert d1.mean() == pytest.approx(5.0, abs=1e-1)
    assert d1.std() == pytest.approx(1.5, abs=1e-1)


def test_camera_identity_no_map():
    cam = CameraNonIdealModel()
    assert cam.prnu_map((10, 10)).mean() == 1.0
    assert cam.dark_map((10, 10), 7.0).mean() == 7.0


def test_build_nonideal_bundle_default_camera_none():
    b = build_nonideal_bundle()
    assert b["zero_order_eta"] == 0.0
    assert b["slm_lut"] is None
    assert b["nonideal_camera"] is None


# --------------------------------------------------------------------------
# 接入 SimPibSystem
# --------------------------------------------------------------------------

def test_zero_order_adds_central_pedestal():
    # 棋盘相位 ⇒ 调制项 ``exp(iφ)`` 在瞳面交替 ±1, 其远场 0 级 (中心) DC ≈ 0。
    # 零级常数项是相干的、居中, 在远场中心灌入一片弥散晕 ⇒ 中心强度显著上升。
    h, w = PANEL
    yy, xx = np.mgrid[0:h, 0:w]
    phase = np.pi * ((xx + yy) % 2)  # 交替 ±1 ⇒ 调制项 DC ≈ 0

    base = _make_system()
    base.set_phase_rad(phase)
    base.far_field()  # 触发 _far_field_raw 缓存
    raw0 = base._far_field_raw

    zo = _make_system(zero_order_eta=0.1)
    zo.set_phase_rad(phase)
    zo.far_field()
    raw1 = zo._far_field_raw

    cy, cx = raw0.shape[0] // 2, raw0.shape[1] // 2
    # 零级相干 DC ⇒ 中心上升 (改动前调制项 DC≈0)
    assert raw1[cy, cx] > raw0[cy, cx]
    assert raw1[cy, cx] > 0.0


def test_lut_roundtrip_quantizes_via_create_phase_from_array():
    phase = np.linspace(0, 2 * np.pi, PANEL[0] * PANEL[1]).reshape(PANEL)
    none_sys = _make_system()
    pib_none = SimSLMPib(system=none_sys)
    pib_none.create_phase_from_array(phase)
    ff_none = none_sys.far_field()

    lut_sys = _make_system(slm_lut=SlmLutModel(gamma=1.0))
    pib_lut = SimSLMPib(system=lut_sys)
    pib_lut.create_phase_from_array(phase)
    ff_lut = lut_sys.far_field()

    # LUT γ=1 也量化 ⇒ 远场不再是逐位相同 (但形状大致保留)
    assert not np.array_equal(ff_none, ff_lut)
    corr = np.corrcoef(ff_none.ravel(), ff_lut.ravel())[0, 1]
    assert corr > 0.9


def test_prnu_deterministic_fixed_pattern_before_quantize():
    cam = CameraNonIdealModel(prnu_sigma=0.05, dark_fpn_sigma=0.0, seed=7)
    sys1 = _make_system(nonideal_camera=cam)
    sys2 = _make_system(nonideal_camera=cam)
    f1 = sys1.counts_frame(0.2)
    f2 = sys2.counts_frame(0.2)
    # 固定图 ⇒ 两帧高度相关 (不是独立白噪声)
    corr = np.corrcoef(f1.ravel(), f2.ravel())[0, 1]
    assert corr > 0.99


def test_dark_fpn_recoverable_sigma():
    cam = CameraNonIdealModel(prnu_sigma=0.0, dark_fpn_sigma=1.5, seed=9)
    sys1 = _make_system(nonideal_camera=cam)
    f = sys1.counts_frame(0.0)
    # 暗场固定图使零光帧仍有 ~dark_fpn_sigma 的空间标准差
    assert f.std() > 1.0


def test_anisotropic_padding_per_axis_pitch():
    sys_iso = _make_system(far_field_padding=4)
    r_iso, c_iso = sys_iso.far_field_pitch_um
    sys_aniso = _make_system(far_field_padding=(2, 4))
    r_an, c_an = sys_aniso.far_field_pitch_um
    assert r_an != c_an
    # 列方向补零同为 4 ⇒ 列节距与各项同性一致; 行补零减半 ⇒ 行节距加倍
    assert abs(c_an - c_iso) < 1e-9
    assert abs(r_an - r_iso * 2) < 1e-6


def test_direct_assignment_renormalizes_padding():
    sys_a = _make_system(far_field_padding=8)
    sys_a.far_field_padding = 4
    assert sys_a._pad_rows == 4 and sys_a._pad_cols == 4
    sys_a.far_field_padding = (3, 5)
    assert sys_a._pad_rows == 3 and sys_a._pad_cols == 5


def test_default_is_bit_identical_to_no_nonideal():
    # 默认旋钮 ⇒ 与未接入非理想模型逐位相同
    p = np.linspace(0, 2 * np.pi, PANEL[0] * PANEL[1]).reshape(PANEL)
    a = _make_system()  # 全默认
    a.set_phase_rad(p)
    fa = a.far_field()
    b = _make_system(zero_order_eta=0.0, slm_lut=None, nonideal_camera=None)
    b.set_phase_rad(p)
    fb = b.far_field()
    assert np.array_equal(fa, fb)


def test_reset_system_forwards_knobs():
    cam = CameraNonIdealModel(prnu_sigma=0.01, seed=1)
    s = reset_system(
        seed=0,
        far_field_padding=4,
        zero_order_eta=0.1,
        slm_lut=SlmLutModel(gamma=1.0),
        nonideal_camera=cam if not cam.is_identity() else None,
    )
    assert abs(s._zero_order.eta - 0.1) < 1e-12
    assert s.slm_lut is not None
    assert s._nonideal_camera is not None
