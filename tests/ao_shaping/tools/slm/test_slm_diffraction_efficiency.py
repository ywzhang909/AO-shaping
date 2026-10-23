"""闪耀光栅衍射效率工具的测试 (纯算法层 + 数字孪生端到端)。

覆盖:

1. **纯算法层** (:mod:`...diffraction_efficiency.core`) —— 图案生成、阶次
   识别、理论效率、拟合反演、线性化查找表。
2. **数字孪生端到端** —— 仿真帧走**与硬件完全相同**的
   :func:`core.run_sweep` → :func:`core.measure_orders` → 拟合链路, 检验
   拟合能把隐藏的真值 (相位深度 / γ) 反演回来。
3. **CLI 冒烟** —— ``simulate`` 子命令能跑完并写出产物。

不碰硬件: 硬件后端的检查只验证"缺光路标定时明确报错"这一条契约。
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.slm.calibration.diffraction_efficiency.core import (
    BlazeSpec,
    OrderDetectionError,
    PhaseResponseFit,
    SweepPoint,
    blaze_gray,
    blaze_phase,
    build_linearization_lut,
    diagnose,
    discrete_order_efficiencies,
    evaluate_fwhm_gate,
    fit_phase_response,
    measure_orders,
    model_ratios,
    negative_order_efficiency,
    order_efficiencies,
    run_sweep,
)


# ---------------------------------------------------------------------------
# 图案生成
# ---------------------------------------------------------------------------


class TestBlazePatterns:
    def test_phase_is_a_sawtooth_reaching_the_requested_depth(self) -> None:
        spec = BlazeSpec(period_px=8, axis="x", depth_rad=2.0)
        phase = blaze_phase(spec, (3, 32))
        assert phase.shape == (3, 32)
        row = phase[1]
        # 采样约定: 周期内第 j 个像元 (j=1..P) 取 depth·j/P —— 最后一个
        # 像元恰好落在 depth 上 (否则命令深度会被系统性地缩小 P/(P-1) 倍)。
        assert row[0] == pytest.approx(2.0 * 1 / 8)
        assert row[4] == pytest.approx(2.0 * 5 / 8)
        assert row[7] == pytest.approx(2.0)
        # 周期复位
        assert row[8] == pytest.approx(2.0 * 1 / 8)
        assert row.max() == pytest.approx(2.0)

    def test_phase_is_not_wrapped(self) -> None:
        """raw-only 契约: 生成器不得自行 mod 2π。"""
        spec = BlazeSpec(period_px=4, axis="x", depth_rad=3.0 * math.pi)
        phase = blaze_phase(spec, (2, 16))
        assert phase.max() > 2.0 * math.pi

    def test_y_axis_ramps_along_the_rows(self) -> None:
        spec = BlazeSpec(period_px=8, axis="y", depth_rad=1.0)
        phase = blaze_phase(spec, (16, 5))
        col = phase[:, 0]
        assert col[0] == pytest.approx(1.0 / 8)
        assert col[4] == pytest.approx(5.0 / 8)
        # 沿列方向恒定
        assert np.allclose(phase[3, :], phase[3, 0])

    def test_pupil_mask_zeroes_the_outside(self) -> None:
        spec = BlazeSpec(
            period_px=8, axis="x", depth_rad=1.0, pupil_center=(10, 5), pupil_radius=3
        )
        phase = blaze_phase(spec, (11, 21))
        assert phase[5, 10] > 0.0
        assert phase[0, 0] == 0.0

    def test_gray_ramp_max_equals_the_command(self) -> None:
        spec = BlazeSpec(period_px=8, axis="x", depth_rad=2.0 * math.pi)
        gray = blaze_gray(spec, (2, 32), max_gray=993)
        assert gray.dtype == np.uint16
        # 采样约定: 周期内第 j 个像元 (j=1..P) 落在 depth·j/P 上, 所以
        # 斜坡的最大灰度**恰好**是命令值。
        assert int(gray.max()) == 993
        assert int(gray.min()) == 993 // 8

    def test_zero_depth_is_a_flat_zero_gray(self) -> None:
        spec = BlazeSpec(period_px=8, axis="x", depth_rad=0.0)
        gray = blaze_gray(spec, (2, 32), max_gray=993)
        assert int(gray.max()) == 0

    def test_invalid_period_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            BlazeSpec(period_px=1)


# ---------------------------------------------------------------------------
# 理论模型
# ---------------------------------------------------------------------------


class TestTheory:
    def test_efficiencies_at_the_landmarks(self) -> None:
        e0, e1 = order_efficiencies(0.0)
        assert e0 == pytest.approx(1.0)
        assert e1 == pytest.approx(0.0, abs=1e-12)
        e0, e1 = order_efficiencies(2.0 * math.pi)
        assert e0 == pytest.approx(0.0, abs=1e-12)
        assert e1 == pytest.approx(1.0)

    def test_efficiency_is_symmetric_around_half_a_wave(self) -> None:
        e0, e1 = order_efficiencies(math.pi)
        assert e0 == pytest.approx((2.0 / math.pi) ** 2)
        assert e1 == pytest.approx((2.0 / math.pi) ** 2)

    def test_the_two_first_orders_are_not_equal(self) -> None:
        """锯齿光栅不对称: η₋₁/η₊₁ = ((Δφ+2π)/(Δφ-2π))²。

        把两侧当对称会让拟合出的相位深度系统性偏小 (数字孪生上实测
        11.5%) —— 这条断言就是钉住这个事实。
        """
        e0, e_plus = order_efficiencies(math.pi)
        e_minus = negative_order_efficiency(math.pi)
        assert e_plus == pytest.approx((2.0 / math.pi) ** 2)
        assert e_minus == pytest.approx((1.0 / (1.5 * math.pi)) ** 2)
        assert e_plus > 8.0 * e_minus
        # Δφ = 2π 时 −1 级完全消失, 能量全在 +1 级
        e0_at_2pi, e_plus_at_2pi = order_efficiencies(2.0 * math.pi)
        assert negative_order_efficiency(2.0 * math.pi) == pytest.approx(0.0, abs=1e-12)
        assert e0_at_2pi == pytest.approx(0.0, abs=1e-12)
        assert e_plus_at_2pi == pytest.approx(1.0)

    def test_model_ratios_sum_to_one(self) -> None:
        _, r1 = model_ratios(np.linspace(0.0, 2.0 * math.pi, 17), 0.1)
        r0, _ = model_ratios(np.linspace(0.0, 2.0 * math.pi, 17), 0.1)
        assert np.allclose(r0 + r1, 1.0)

    def test_leakage_lowers_the_peak(self) -> None:
        _, r1_clean = model_ratios(2.0 * math.pi, 0.0)
        _, r1_leaky = model_ratios(2.0 * math.pi, 0.5)
        assert r1_leaky < r1_clean


# ---------------------------------------------------------------------------
# 阶次识别
# ---------------------------------------------------------------------------


def _three_spot_frame(
    center: tuple[float, float] = (128.0, 128.0),
    separation: float = 64.0,
    sigma: float = 3.0,
    size: int = 256,
) -> np.ndarray:
    """合成帧: 零级 + 两个一级衍射斑 (沿行方向)。"""
    cy, cx = center
    yy, xx = np.mgrid[0:size, 0:size]
    frame = np.zeros((size, size), dtype=np.float64)
    for off, amp in ((0.0, 100.0), (-separation, 40.0), (separation, 40.0)):
        frame += amp * np.exp(
            -((yy - cy) ** 2 + (xx - (cx + off)) ** 2) / (2.0 * sigma**2)
        )
    return frame


class TestOrderDetection:
    def test_finds_the_three_orders_and_the_ratios(self) -> None:
        frame = _three_spot_frame()
        measurement = measure_orders(
            frame, expected_center=(128.0, 128.0), predicted_separation_px=64.0
        )
        assert measurement.zero is not None
        assert measurement.zero.cy == pytest.approx(128.0, abs=1.0)
        assert measurement.eta_zero == pytest.approx(100.0 / 180.0, rel=0.05)
        assert measurement.eta_plus_total == pytest.approx(80.0 / 180.0, rel=0.05)
        assert measurement.separation_px == pytest.approx(64.0, abs=2.0)
        assert measurement.residual_px == pytest.approx(0.0, abs=2.0)

    def test_extinguished_zero_order_is_reported_as_missing_energy(self) -> None:
        """零级被"熄灭"时 (Δφ=2π) 仍要给出 ±1 级的能量。"""
        frame = _three_spot_frame()
        frame[100:160, 100:160] = 0.0  # 抹掉零级
        measurement = measure_orders(
            frame, expected_center=(128.0, 128.0), predicted_separation_px=64.0
        )
        assert measurement.e_zero == pytest.approx(0.0, abs=1e-6)
        assert measurement.eta_plus_total == pytest.approx(1.0, rel=0.05)

    def test_empty_frame_raises(self) -> None:
        with pytest.raises(OrderDetectionError):
            measure_orders(np.zeros((64, 64)))

    def test_orders_are_found_without_a_calibrated_center(self) -> None:
        frame = _three_spot_frame()
        measurement = measure_orders(frame, predicted_separation_px=64.0)
        assert measurement.eta_plus_total == pytest.approx(80.0 / 180.0, rel=0.1)

    def test_the_airy_ring_around_the_zero_order_is_not_mistaken_for_an_order(self) -> None:
        """回归: 零级光斑自带一圈 Airy 旁瓣环 (实测 ≈ 零级峰值的 8%), 离零级
        只有 ~9px。低调制深度处一级衍射本身也只占几个百分点 —— 不按几何
        窗口挑, 旁瓣环会被当成一级衍射, 效率直接算成"零级几乎没有、
        一级占 2/3"这种胡说 (数字孪生上真实发生过)。
        """
        size = 512
        center = (256.0, 256.0)
        separation = 200.0
        yy, xx = np.mgrid[0:size, 0:size]
        frame = np.zeros((size, size), dtype=np.float64)
        # 零级 + 四个旁瓣 (位于 ±9px 的环上, 峰值 8%)
        frame += 100.0 * np.exp(
            -((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * 3.0**2)
        )
        for dy, dx in ((9, 0), (-9, 0), (0, 9), (0, -9)):
            frame += 8.0 * np.exp(
                -((yy - (center[0] + dy)) ** 2 + (xx - (center[1] + dx)) ** 2)
                / (2 * 2.0**2)
            )
        # 真正的一级衍射 (±200px, 弱)
        for off in (-separation, separation):
            frame += 20.0 * np.exp(
                -((yy - center[0]) ** 2 + (xx - (center[1] + off)) ** 2)
                / (2 * 3.0**2)
            )
        measurement = measure_orders(
            frame,
            expected_center=center,
            predicted_separation_px=separation,
        )
        assert measurement.separation_px == pytest.approx(separation, rel=0.05)
        # 一级总能量 40, 零级 100 → r1 ≈ 0.29 (不是"旁瓣"给出的 0.66)
        assert measurement.eta_plus_total == pytest.approx(40.0 / 140.0, rel=0.15)

    def test_one_side_only_and_a_self_split_zero_order_still_measures(self) -> None:
        """回归 (2026-10-23 实验系统实测): 闪耀光栅只有 **+1** 级, 且零级自己会分成
        几个局部极大。此前的两处判据会同时失效:

        1. "锚点 → 最亮旁斑"取方向时, 最亮的"旁斑"其实是**零级自己的次极大**
           (实测距中心 9px), 于是方向指向零级旁瓣方向, 真正的 +1 级
           (在 −行 235px) 投影到 ~0 而被整条丢掉;
        2. 窗口内按"最接近预测间距"挑, 会挑到同一条 +1 光斑里最弱的那个次峰。

        现象: ``η0`` 恒等于 1.000 (低调制端整段作废)。修法: 锚点自身光斑内
        (``ANCHOR_CORE_FRACTION``) 的峰不参与方向判定; 窗口内取**最亮**者。
        """
        size = 560
        center = (300.0, 280.0)
        separation = 235.0
        yy, xx = np.mgrid[0:size, 0:size]
        frame = np.zeros((size, size), dtype=np.float64)
        # 零级: 一个宽斑, 内部两个相距 12px 的极大 (实测就是这样)
        frame += 100.0 * np.exp(
            -((yy - center[0]) ** 2 + (xx - (center[1] - 6.0)) ** 2) / (2 * 5.0**2)
        )
        frame += 95.0 * np.exp(
            -((yy - center[0]) ** 2 + (xx - (center[1] + 6.0)) ** 2) / (2 * 5.0**2)
        )
        # 只有 +1 级 (锯齿光栅在 Δφ→2π 时 −1 级消失), 沿**行**负方向
        frame += 35.0 * np.exp(
            -((yy - (center[0] - separation)) ** 2 + (xx - center[1]) ** 2)
            / (2 * 5.0**2)
        )
        measurement = measure_orders(
            frame,
            expected_center=center,
            predicted_separation_px=separation,
        )
        assert measurement.plus_one is not None
        assert measurement.minus_one is None
        assert measurement.separation_px == pytest.approx(separation, rel=0.05)
        assert measurement.eta_plus_total is not None
        assert measurement.eta_plus_total > 0.0
        # 零级不再被当成"全部能量" (此前 η0 恒为 1.000, 低调制端整段作废)。
        assert 0.0 < measurement.eta_zero < 1.0
        # 归一化是"三者和", 所以两者必须互补。
        assert measurement.eta_zero + measurement.eta_plus_total == pytest.approx(1.0)

    def test_a_point_without_visible_orders_reports_zero_efficiency(self) -> None:
        """深度太浅时一级衍射不可见 —— 本点一级效率应记 0 (正确读数, 不是失败)。"""
        size = 512
        yy, xx = np.mgrid[0:size, 0:size]
        frame = 100.0 * np.exp(-((yy - 256.0) ** 2 + (xx - 256.0) ** 2) / 18.0)
        measurement = measure_orders(
            frame, expected_center=(256.0, 256.0), predicted_separation_px=200.0
        )
        assert measurement.eta_plus_total == pytest.approx(0.0, abs=1e-9)
        assert measurement.eta_zero == pytest.approx(1.0, abs=1e-6)

    def test_a_broad_zero_order_is_reported_as_unresolvable(self) -> None:
        """回归: 光斑比阶次间距还宽时, 读数不可信 —— 必须**说出来**。

        能量积分 ROI 只有 0.08×间距, 在它里面测 FWHM 会被截顶, 于是
        "光斑已经糊到和阶次同宽"这种致命状态永远看不出来 (离焦时永远报一个
        看起来正常的小值)。FWHM 必须在**大窗口**里测, 并且要点名。
        """
        size = 512
        separation = 200.0
        yy, xx = np.mgrid[0:size, 0:size]
        frame = np.zeros((size, size), dtype=np.float64)
        # 零级故意糊成 FWHM≈60px (σ≈25.5)
        frame += 100.0 * np.exp(
            -((yy - 256.0) ** 2 + (xx - 256.0) ** 2) / (2 * 25.5**2)
        )
        for off in (-separation, separation):
            frame += 40.0 * np.exp(
                -((yy - 256.0) ** 2 + (xx - (256.0 + off)) ** 2) / (2 * 25.5**2)
            )
        measurement = measure_orders(
            frame, expected_center=(256.0, 256.0), predicted_separation_px=separation
        )
        assert measurement.zero is not None
        # FWHM 必须如实报出宽光斑 (而不是被 ROI 截顶成 ~2×ROI=31px)
        assert measurement.zero.fwhm_px > 45.0
        assert any("过宽" in note for note in measurement.notes)
        assert any("不可信" in note for note in measurement.notes)

    def test_a_sharp_zero_order_reports_its_true_fwhm(self) -> None:
        """对照: 在焦时 FWHM 应接近真实值, 且不报"过宽"。"""
        size = 512
        separation = 200.0
        yy, xx = np.mgrid[0:size, 0:size]
        frame = np.zeros((size, size), dtype=np.float64)
        frame += 100.0 * np.exp(
            -((yy - 256.0) ** 2 + (xx - 256.0) ** 2) / (2 * 3.0**2)
        )
        for off in (-separation, separation):
            frame += 40.0 * np.exp(
                -((yy - 256.0) ** 2 + (xx - (256.0 + off)) ** 2) / (2 * 3.0**2)
            )
        measurement = measure_orders(
            frame, expected_center=(256.0, 256.0), predicted_separation_px=separation
        )
        assert measurement.zero is not None
        assert measurement.zero.fwhm_px == pytest.approx(7.0, abs=3.0)
        assert not any("过宽" in note for note in measurement.notes)


# ---------------------------------------------------------------------------
# 拟合
# ---------------------------------------------------------------------------


def _synthetic_curve(
    depth_max: float,
    gamma: float,
    leakage: float,
    n: int = 25,
    period: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """按**离散器件模型**生成一条扫描曲线 (u, r1)。"""
    u = np.linspace(0.0, 1.0, n)
    _, r1 = discrete_order_efficiencies(u, depth_max, gamma, period, leakage)
    return u, r1


class TestDiscreteModel:
    def test_converges_to_the_ideal_formula_for_a_long_period(self) -> None:
        """离散模型在 P→∞ 时必须回到理想 sinc² 公式。"""
        u = np.linspace(0.0, 1.0, 9)
        _, r1_discrete = discrete_order_efficiencies(u, 2.0 * math.pi, 1.0, 4096, 0.0)
        _, r1_ideal = model_ratios(2.0 * math.pi * u, 0.0)
        assert np.allclose(r1_discrete, r1_ideal, atol=2e-3)

    def test_ideal_formula_would_misjudge_a_nonlinear_device(self) -> None:
        """γ≠1 时理想公式与离散模型差别显著 —— 这正是必须用离散模型的原因。"""
        u = np.linspace(0.0, 1.0, 9)
        _, r1_discrete = discrete_order_efficiencies(u, 2.0 * math.pi, 1.8, 16, 0.0)
        _, r1_ideal = model_ratios(2.0 * math.pi * u, 0.0)
        assert np.max(np.abs(r1_discrete - r1_ideal)) > 0.05

    def test_ratios_sum_to_one(self) -> None:
        r0, r1 = discrete_order_efficiencies(
            np.linspace(0.0, 1.0, 11), 2.0 * math.pi, 1.0, 16, 0.1
        )
        assert np.allclose(r0 + r1, 1.0)


class TestFit:
    def test_recovers_depth_gamma_and_leakage(self) -> None:
        u, r1 = _synthetic_curve(2.0 * math.pi, 1.4, 0.05)
        fit = fit_phase_response(
            u * 1023.0, r1, command_max=1023.0, period_px=16
        )
        assert fit.depth_max_rad == pytest.approx(2.0 * math.pi, rel=0.05)
        assert fit.gamma == pytest.approx(1.4, rel=0.15)
        assert fit.leakage == pytest.approx(0.05, abs=0.05)
        assert fit.residual_rms < 1e-3

    def test_detects_over_modulation(self) -> None:
        """真值深度 > 2π 时曲线非单调, 拟合必须把它认出来。"""
        u, r1 = _synthetic_curve(4.0 * math.pi, 1.0, 0.0)
        fit = fit_phase_response(
            u * 1023.0, r1, command_max=1023.0, period_px=16
        )
        assert fit.depth_max_rad == pytest.approx(4.0 * math.pi, rel=0.1)
        diagnosis = diagnose([], fit)
        assert diagnosis["modulation"] == "over"

    def test_under_modulation_is_flagged(self) -> None:
        u, r1 = _synthetic_curve(1.4 * math.pi, 1.0, 0.0)
        fit = fit_phase_response(
            u * 1023.0, r1, command_max=1023.0, period_px=16
        )
        diagnosis = diagnose([], fit)
        assert diagnosis["modulation"] == "under"

    def test_too_few_points_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            fit_phase_response(
                [0.0, 1.0], [0.0, 0.1], command_max=1023.0, period_px=16
            )


class TestLinearizationLut:
    def test_lut_inverts_the_power_law(self) -> None:
        fit = PhaseResponseFit(
            depth_max_rad=2.0 * math.pi,
            gamma=1.5,
            leakage=0.0,
            residual_rms=0.0,
            n_points=3,
            converged=True,
            message="",
            commands=np.zeros(3),
            model_depth_rad=np.zeros(3),
            model_eta_plus_total=np.zeros(3),
            measured_eta_plus_total=np.zeros(3),
            outliers=(),
            stderr={},
        )
        lut = build_linearization_lut(fit, size=256, gray_max=1023)
        assert lut.dtype == np.uint16
        for index in (0, 64, 128, 255):
            u = lut[index] / 1023.0
            assert 2.0 * math.pi * u**1.5 == pytest.approx(
                2.0 * math.pi * index / 255.0, rel=2e-3
            )


# ---------------------------------------------------------------------------
# 扫描协议
# ---------------------------------------------------------------------------


class TestSweepProtocol:
    def test_run_sweep_uses_the_displayed_pattern(self) -> None:
        """display 与 capture 必须看到同一份灰度 —— 协议错位会静默产出假数据。"""
        seen: list[np.ndarray] = []
        frames = [_three_spot_frame() for _ in range(3)]

        def pattern_for(command: float) -> np.ndarray:
            return np.full((2, 2), int(command), dtype=np.uint16)

        def display(gray: np.ndarray) -> None:
            seen.append(gray)

        def capture() -> np.ndarray:
            return frames[len(seen) - 1]

        points = run_sweep(
            pattern_for, display, capture, [10.0, 20.0, 30.0], predicted_separation_px=64.0
        )
        assert len(points) == 3
        assert [p.command for p in points] == [10.0, 20.0, 30.0]
        assert int(seen[1][0, 0]) == 20
        assert all(p.eta_plus_total is not None for p in points)


# ---------------------------------------------------------------------------
# 数字孪生端到端
# ---------------------------------------------------------------------------


def _sim_module():
    return importlib.import_module(
        "ao_shaping.tools.slm.calibration.diffraction_efficiency.simulate"
    )


class TestSimulatedEndToEnd:
    def test_fit_recovers_the_hidden_device_truth(self) -> None:
        sim = _sim_module()
        truth_depth = 2.0 * math.pi
        truth_gamma = 1.0
        config = sim.SimConfig(
            region=240,
            padding=8,
            pupil_radius_px=480,
            depth_max_rad=truth_depth,
            gamma=truth_gamma,
            seed=7,
        )
        bench = sim.SimulatedBench(config)
        gray_levels = int(config.gray_levels)
        commands = np.linspace(0.0, float(gray_levels), 15)

        def pattern_for(command: float) -> np.ndarray:
            spec = BlazeSpec(
                period_px=32,
                axis="x",
                depth_rad=2.0 * math.pi * command / gray_levels,
                pupil_center=(0, 0),
                pupil_radius=0,
            )
            return blaze_gray(spec, bench.panel_shape, gray_levels)

        issued: list[np.ndarray] = [np.zeros((1, 1), dtype=np.uint16)]

        def display(gray: np.ndarray) -> None:
            issued[0] = gray

        def capture() -> np.ndarray:
            return bench.capture(issued[0])

        frame_center = (bench.frame_shape[0] / 2.0, bench.frame_shape[1] / 2.0)
        points = run_sweep(
            pattern_for,
            display,
            capture,
            commands,
            expected_center=frame_center,
            predicted_separation_px=config.kx / 32.0,
        )
        assert all(p.eta_plus_total is not None for p in points)
        fit = fit_phase_response(
            [p.command for p in points],
            [p.eta_plus_total for p in points],
            command_max=float(gray_levels),
            period_px=32,
        )
        # 真值是隐藏的, 拟合必须把它反演回来 (离散器件模型 + 不对称两侧)
        assert fit.depth_max_rad == pytest.approx(truth_depth, rel=0.05)
        assert fit.gamma == pytest.approx(truth_gamma, rel=0.1)
        # 几何: 测得的 ±1 级间距必须与 K/P 相符 (光轴交叉也不能错)
        separations = [p.separation_px for p in points if p.separation_px]
        assert separations
        assert float(np.median(separations)) == pytest.approx(config.kx / 32.0, rel=0.01)

    def test_a_nonlinear_device_is_not_mistaken_for_a_linear_one(self) -> None:
        """γ≠1 的器件: 用理想公式拟合会把 Δφ_max 与 γ **同时**偏掉。

        这条断言守住"必须用离散器件模型"这个决定。
        """
        sim = _sim_module()
        truth_gamma = 1.6
        config = sim.SimConfig(
            region=240,
            padding=8,
            pupil_radius_px=480,
            depth_max_rad=2.0 * math.pi,
            gamma=truth_gamma,
            seed=3,
        )
        bench = sim.SimulatedBench(config)
        gray_levels = int(config.gray_levels)
        commands = np.linspace(0.0, float(gray_levels), 15)

        def pattern_for(command: float) -> np.ndarray:
            spec = BlazeSpec(
                period_px=32,
                axis="x",
                depth_rad=2.0 * math.pi * command / gray_levels,
                pupil_center=(0, 0),
                pupil_radius=0,
            )
            return blaze_gray(spec, bench.panel_shape, gray_levels)

        issued: list[np.ndarray] = [np.zeros((1, 1), dtype=np.uint16)]

        def display(gray: np.ndarray) -> None:
            issued[0] = gray

        def capture() -> np.ndarray:
            return bench.capture(issued[0])

        frame_center = (bench.frame_shape[0] / 2.0, bench.frame_shape[1] / 2.0)
        points = run_sweep(
            pattern_for,
            display,
            capture,
            commands,
            expected_center=frame_center,
            predicted_separation_px=config.kx / 32.0,
        )
        fit = fit_phase_response(
            [p.command for p in points],
            [p.eta_plus_total for p in points],
            command_max=float(gray_levels),
            period_px=32,
        )
        assert fit.gamma == pytest.approx(truth_gamma, rel=0.15)


# ---------------------------------------------------------------------------
# 光斑宽度门禁 (相对在焦参照值的倍数)
# ---------------------------------------------------------------------------


def _gate_point(
    command: float,
    fwhm: float | None,
    eta_zero: float | None = 0.5,
) -> SweepPoint:
    """构造一个只带门禁所需字段的扫描点。"""
    return SweepPoint(
        command=command,
        e_zero=1.0,
        e_plus=1.0,
        e_minus=1.0,
        eta_zero=eta_zero,
        eta_plus_total=0.5,
        separation_px=None,
        residual_px=None,
        peak_adu=1000.0,
        fwhm_px=fwhm,
        ellipticity=None,
        zero_center=None,
    )


class TestFwhmGate:
    def test_the_reference_is_the_flat_point(self) -> None:
        """参照值 = 命令最小那一点的 FWHM (平相位)。"""
        points = [
            _gate_point(0.0, 9.0),
            _gate_point(500.0, 10.0),
            _gate_point(1000.0, 12.0),
        ]
        gate = evaluate_fwhm_gate(points, gate_ratio=3.0)
        assert gate.reference_fwhm_px == pytest.approx(9.0)
        assert gate.limit_px == pytest.approx(27.0)
        assert gate.ok
        assert gate.evaluated_points == 3

    def test_blurred_points_are_flagged(self) -> None:
        points = [
            _gate_point(0.0, 9.0),
            _gate_point(500.0, 10.0),
            _gate_point(1000.0, 60.0),  # 扫描过程中光斑变糊
        ]
        gate = evaluate_fwhm_gate(points, gate_ratio=3.0)
        assert not gate.ok
        assert gate.violations == (2,)
        assert gate.max_ratio == pytest.approx(60.0 / 9.0)

    def test_extinguished_zero_order_points_are_skipped_not_killed(self) -> None:
        """零级被熄灭的点, 宽度读数是噪声底上的读数 —— 不能拿它当门禁判据。"""
        points = [
            _gate_point(0.0, 9.0, eta_zero=1.0),
            _gate_point(900.0, 400.0, eta_zero=0.001),  # 零级已熄灭
        ]
        gate = evaluate_fwhm_gate(points, gate_ratio=3.0)
        assert gate.ok
        assert gate.skipped_points == 1
        assert gate.evaluated_points == 1

    def test_an_absolute_reference_can_be_supplied(self) -> None:
        """跨次运行可比: 传入一条已知在焦的光路量出来的绝对参照值。"""
        points = [_gate_point(0.0, 40.0), _gate_point(500.0, 42.0)]
        gate = evaluate_fwhm_gate(
            points, reference_fwhm_px=10.0, gate_ratio=3.0
        )
        assert gate.reference_fwhm_px == pytest.approx(10.0)
        assert not gate.ok  # 40px 相对 10px 的参照 = 4 倍 → 越限
        assert gate.violations == (0, 1)

    def test_the_fit_drops_violating_points_and_the_verdict_is_marked(self) -> None:
        """门禁越限的点必须从拟合里剔除 —— 留着它们会把相位深度拉偏。"""
        points = [
            _gate_point(0.0, 9.0),
            _gate_point(340.0, 10.0),
            _gate_point(680.0, 11.0),
            _gate_point(1023.0, 12.0),
        ]
        gate = evaluate_fwhm_gate(points, gate_ratio=3.0)
        fit = fit_phase_response(
            [p.command for p in points],
            [p.eta_plus_total for p in points],
            command_max=1023.0,
            period_px=16,
            gate=gate,
        )
        assert fit.gate_excluded == 0
        assert fit.n_points == 4
        diagnosis = diagnose(points, fit, gate=gate)
        assert diagnosis["modulation_verdict_trustworthy"] is True
        assert diagnosis["fwhm_gate"]["ok"] is True
        assert diagnosis["fwhm_gate"]["reference_fwhm_px"] == pytest.approx(9.0)

    def test_a_blurred_run_is_excluded_and_flagged(self) -> None:
        """扫描过程中变糊 ⇒ 那些点被剔除, 且判定被标为不可信。"""
        points = [
            _gate_point(0.0, 9.0),
            _gate_point(200.0, 10.0),
            _gate_point(400.0, 10.5),
            _gate_point(680.0, 60.0),  # 变糊
            _gate_point(850.0, 65.0),  # 变糊
            _gate_point(1023.0, 70.0),  # 变糊
        ]
        gate = evaluate_fwhm_gate(points, gate_ratio=3.0)
        assert gate.violations == (3, 4, 5)
        fit = fit_phase_response(
            [p.command for p in points],
            [p.eta_plus_total for p in points],
            command_max=1023.0,
            period_px=16,
            gate=gate,
        )
        assert fit.gate_excluded == 3
        assert fit.n_points == 3
        diagnosis = diagnose(points, fit, gate=gate)
        assert diagnosis["modulation_verdict_trustworthy"] is False
        assert diagnosis["fwhm_gate"]["ok"] is False
        assert any("门禁" in note for note in diagnosis["notes"])

    def test_extinguished_points_cannot_sneak_into_a_violated_fit(self) -> None:
        """回归: 门禁触发时, "零级被熄灭而跳过评估"的点必须**一并**被剔除。

        否则拟合会靠这几个最不可信的点凑出一个看着正常其实荒唐的深度
        (数字孪生实测: 剔除 17/21 个点后, 拟合靠剩下 4 个被跳过的点给出
        1.9775π) —— 那正是这个门禁要消灭的东西。
        """
        points = [
            _gate_point(0.0, 9.0, eta_zero=0.5),
            _gate_point(340.0, 10.0, eta_zero=0.5),
            _gate_point(800.0, 60.0, eta_zero=0.5),  # 变糊 → 违例
            _gate_point(900.0, 65.0, eta_zero=0.001),  # 零级熄灭 → 跳过评估
            _gate_point(1023.0, 70.0, eta_zero=0.001),  # 同上
        ]
        gate = evaluate_fwhm_gate(points, gate_ratio=3.0)
        assert gate.violations == (2,)
        assert gate.skipped_indices == (3, 4)
        # 只剩 2 个可用点 → 拟合必须**显式失败**, 而不是给出一个荒唐的深度
        with pytest.raises(ValueError, match="门禁"):
            fit_phase_response(
                [p.command for p in points],
                [p.eta_plus_total for p in points],
                command_max=1023.0,
                period_px=16,
                gate=gate,
            )

    def test_extinguished_points_still_fit_when_the_gate_is_clean(self) -> None:
        """门禁没触发时, 零级被熄灭的点**照常参与拟合** (它们信息量最大)。"""
        points = [
            _gate_point(0.0, 9.0, eta_zero=1.0),
            _gate_point(340.0, 10.0, eta_zero=0.5),
            _gate_point(680.0, 10.5, eta_zero=0.5),
            _gate_point(1023.0, 400.0, eta_zero=0.001),  # 零级熄灭, 宽度读数无意义
        ]
        gate = evaluate_fwhm_gate(points, gate_ratio=3.0)
        assert gate.ok
        assert gate.skipped_indices == (3,)
        fit = fit_phase_response(
            [p.command for p in points],
            [p.eta_plus_total for p in points],
            command_max=1023.0,
            period_px=16,
            gate=gate,
        )
        assert fit.n_points == 4
        assert fit.gate_excluded == 0


# ---------------------------------------------------------------------------
# 数字孪生的器件模型 (填充因子 / LUT / 面板相差)
# ---------------------------------------------------------------------------


class TestSimulatedDeviceModels:
    """锁住数字孪生里"器件非理想"这几项的物理行为。

    这些不是装饰: 报告里的每张表都由它们生成, 模型错了结论就错。
    """

    def _bench(self, **kwargs):
        sim = _sim_module()
        base = dict(region=240, padding=8, pupil_radius_px=480, seed=5)
        base.update(kwargs)
        return sim.SimulatedBench(sim.SimConfig(**base))

    @staticmethod
    def _capture_full_depth(bench, period: int = 32) -> np.ndarray:
        """下发满深度闪耀光栅, 采一帧。"""
        gray_levels = int(bench.config.gray_levels)
        spec = BlazeSpec(
            period_px=period,
            axis="x",
            depth_rad=2.0 * math.pi,
            pupil_center=(0, 0),
            pupil_radius=0,
        )
        gray = blaze_gray(spec, bench.panel_shape, gray_levels)
        frame = bench.capture(gray)
        center = (bench.frame_shape[0] / 2.0, bench.frame_shape[1] / 2.0)
        return measure_orders(
            frame,
            expected_center=center,
            predicted_separation_px=7546.4 / period,
        )

    def test_fill_factor_scales_the_absolute_first_order_energy_by_ff_squared(self) -> None:
        """填充因子 ff: ±1 级的**绝对**能量按 ff² 缩放 (死区只把光漏进零级)。"""
        full = self._capture_full_depth(self._bench(fill_factor=1.0))
        partial = self._capture_full_depth(self._bench(fill_factor=0.8))
        ratio = (partial.e_plus + partial.e_minus) / (full.e_plus + full.e_minus)
        assert ratio == pytest.approx(0.8**2, rel=0.05)

    def test_relative_efficiency_is_almost_insensitive_to_fill_factor(self) -> None:
        """相对效率对填充因子几乎不敏感 —— 所以**相对**效率测不出填充因子。

        绝对能量按 ff² 掉 (0.8² = 0.64), 但相对效率只掉几个百分点 ——
        因为死区漏掉的那部分光进了零级, 而零级在归一化的分母里。这条断言
        是报告结论的根: 填充因子打的是绝对通量, 而工具报的是相对值。
        """
        full = self._capture_full_depth(self._bench(fill_factor=1.0))
        partial = self._capture_full_depth(self._bench(fill_factor=0.8))
        absolute_ratio = (partial.e_plus + partial.e_minus) / (full.e_plus + full.e_minus)
        relative_ratio = partial.eta_plus_total / full.eta_plus_total
        assert absolute_ratio == pytest.approx(0.8**2, rel=0.05)
        assert relative_ratio > 0.9

    def test_lut_table_overrides_the_power_law(self) -> None:
        gray_levels = 1023
        table = tuple(2.0 * math.pi * (i / gray_levels) ** 3 for i in range(gray_levels + 1))
        bench = self._bench(lut_table=table)
        phase = bench.gray_to_phase(np.array([0, 512, 1023]))
        assert phase[0] == pytest.approx(0.0)
        assert phase[2] == pytest.approx(2.0 * math.pi)
        assert phase[1] == pytest.approx(table[512])
        # 幂律模型被查找表取代: γ=1 但实际相位律是三次方
        assert bench.gray_to_phase(np.array([512]))[0] != pytest.approx(
            2.0 * math.pi * 0.5
        )

    def test_panel_aberration_leaves_the_efficiency_intact(self) -> None:
        """面板相差 (倾斜) 不改变效率 —— 它只移动光斑。

        所以"效率读数"对相差是稳健的, 相差要靠**光斑形状**诊断 (椭圆率)。
        """
        reference = self._capture_full_depth(self._bench())
        tilted = self._capture_full_depth(self._bench(panel_zernike={2: 0.5}))
        assert tilted.eta_plus_total == pytest.approx(reference.eta_plus_total, rel=0.02)


# ---------------------------------------------------------------------------
# CLI 冒烟
# ---------------------------------------------------------------------------


class TestCli:
    def test_simulate_command_writes_the_artifacts(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        module = importlib.import_module(
            "ao_shaping.tools.slm.calibration.diffraction_efficiency.__main__"
        )
        out_dir = tmp_path / "sim"
        runner = CliRunner()
        result = runner.invoke(
            module.main,
            [
                "simulate",
                "--out-dir",
                str(out_dir),
                "--n-steps",
                "5",
                "--period-px",
                "32",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "measurements.csv").is_file()
        assert (out_dir / "summary.json").is_file()


# ---------------------------------------------------------------------------
# 硬件后端的契约 (不碰硬件)
# ---------------------------------------------------------------------------


class TestHardwareContract:
    def test_missing_geometry_calibration_fails_loudly(self, tmp_path: Path) -> None:
        """没有光路标定就预测不了 ±1 级落点 —— 必须明确报错, 不能猜。"""
        module = importlib.import_module(
            "ao_shaping.tools.slm.calibration.diffraction_efficiency.measure"
        )
        config = module.MeasurementConfig(
            calibration_path=tmp_path / "does_not_exist.npz",
            out_dir=tmp_path / "out",
        )
        with pytest.raises((RuntimeError, FileNotFoundError)):
            module.run_measurement(config)


# ---------------------------------------------------------------------------
# 前置焦面判定: 先判在焦, 再测量
# ---------------------------------------------------------------------------


def _focus_point(*, diopter: float, fwhm: float, status: str = "ok"):
    from ao_shaping.tools.slm.panel.focus_probe import FocusPoint

    focal_mm = float("inf") if abs(diopter) < 1e-12 else 1000.0 / float(diopter)
    return FocusPoint(
        diopter_m1=float(diopter),
        focal_mm=focal_mm,
        exposure_ms=1.0,
        peak=100.0,
        peak_fraction=0.5,
        peak_per_ms=100.0,
        fwhm_px=float(fwhm),
        hollowness=0.1,
        status=status,
    )


def _focus_result(points, best=None, *, blocked: bool = False, reason: str = ""):
    from ao_shaping.tools.slm.panel.focus_probe import FocusProbeResult

    return FocusProbeResult(
        points=tuple(points),
        best=best,
        best_focal_mm=(best.focal_mm if best is not None else None),
        best_diopter_m1=(best.diopter_m1 if best is not None else None),
        gain_vs_flat=None,
        linearity_ratio=None,
        range_m1=(-6.0, 6.0),
        exposure_bounds_ms=(0.02, 20.0),
        notes=(),
        path_blocked=blocked,
        blockage_reason=reason,
    )


class TestFocusPreflight:
    """``judge_focus`` 是"拒绝测量"的唯一判据, 必须离线可测。"""

    def test_a_narrow_best_point_authorises_the_measurement(self) -> None:
        from ao_shaping.tools.slm.panel.focus_probe import judge_focus

        best = _focus_point(diopter=0.62, fwhm=15.2)
        verdict = judge_focus(_focus_result([best], best), fwhm_max_px=40.0)
        assert verdict.ok
        assert verdict.kind == "ok"
        assert verdict.best_fwhm_px == pytest.approx(15.2)
        assert verdict.best_focal_mm == pytest.approx(1000.0 / 0.62)
        assert "在焦" in verdict.verdict

    def test_a_blurred_best_point_refuses(self) -> None:
        from ao_shaping.tools.slm.panel.focus_probe import judge_focus

        best = _focus_point(diopter=0.0, fwhm=71.3)
        verdict = judge_focus(_focus_result([best], best), fwhm_max_px=40.0)
        assert not verdict.ok
        assert verdict.kind == "not_in_focus"
        assert "拒绝测量" in verdict.verdict
        assert "71.3" in verdict.reason

    def test_the_threshold_is_inclusive(self) -> None:
        from ao_shaping.tools.slm.panel.focus_probe import judge_focus

        best = _focus_point(diopter=0.6, fwhm=40.0)
        assert judge_focus(_focus_result([best], best), fwhm_max_px=40.0).ok

    def test_no_usable_point_refuses(self) -> None:
        from ao_shaping.tools.slm.panel.focus_probe import judge_focus

        verdict = judge_focus(_focus_result([], None), fwhm_max_px=40.0)
        assert verdict.kind == "not_in_focus"
        assert verdict.best_fwhm_px is None
        assert "没有可用测点" in verdict.reason

    def test_a_blocked_path_is_a_separate_fault(self) -> None:
        from ao_shaping.tools.slm.panel.focus_probe import judge_focus

        verdict = judge_focus(
            _focus_result([], None, blocked=True, reason="光没到相机"),
            fwhm_max_px=40.0,
        )
        assert verdict.kind == "blocked"
        assert "光路受阻" in verdict.verdict
        assert verdict.reason == "光没到相机"


class TestRawGrayConversion:
    """原始灰度路径的包裹点: 叠加离焦补偿后必须绕回, 不能被削平。"""

    def test_phase_to_gray_wraps_instead_of_clipping(self) -> None:
        from ao_shaping.tools.slm.calibration.diffraction_efficiency.core import (
            phase_to_gray,
        )

        gray = phase_to_gray(np.array([[2.0 * math.pi + 1.0]]), 993)
        # 1.0 rad 折成灰度 ≈ 993/(2π) ≈ 158; 不包裹的实现会夹成 993。
        assert int(gray.ravel()[0]) == pytest.approx(round(993 / (2 * math.pi)), abs=1)

    def test_phase_to_gray_matches_blaze_gray_for_a_plain_ramp(self) -> None:
        from ao_shaping.tools.slm.calibration.diffraction_efficiency.core import (
            phase_to_gray,
        )

        spec = BlazeSpec(period_px=16, axis="x", depth_rad=math.pi)
        phase = blaze_phase(spec, (8, 16))
        assert np.array_equal(phase_to_gray(phase, 993), blaze_gray(spec, (8, 16), 993))

    def test_extra_phase_wraps_and_keeps_the_ramp_depth(self) -> None:
        spec = BlazeSpec(period_px=16, axis="x", depth_rad=2.0 * math.pi)
        extra = np.full((8, 16), 1.0)
        gray_extra = blaze_gray(spec, (8, 16), 993, extra_phase_rad=extra)
        gray_plain = blaze_gray(spec, (8, 16), 993)

        # 期望值 = 斜坡相位 + 1.0 rad 后**按 2π 绕回**再标度。
        # 斜坡约定: 第 j 个像元 (1..P) 取 depth·j/P, 所以最后一格正好等于 2π。
        j = np.arange(1, 17, dtype=np.float64) / 16.0
        expected = np.rint(
            np.mod(j * 2.0 * math.pi + 1.0, 2.0 * math.pi) / (2.0 * math.pi) * 993.0
        ).astype(np.uint16)
        assert np.array_equal(gray_extra[0], expected)

        # 关键判据: 不包裹的实现会把接缝夹成 993 的平顶, 有效深度被削掉一截。
        assert not np.array_equal(gray_extra, gray_plain)
        assert int(gray_extra.max()) <= 993
        assert not np.any(gray_extra == 993)
        # 斜坡的 16 级台阶必须全部保留 (削平会少掉若干级)。
        assert len(np.unique(gray_extra)) == 16

    def test_extra_phase_shape_mismatch_is_rejected(self) -> None:
        spec = BlazeSpec(period_px=16, axis="x", depth_rad=math.pi)
        with pytest.raises(ValueError, match="extra_phase_rad"):
            blaze_gray(spec, (8, 16), 993, extra_phase_rad=np.zeros((4, 4)))


class TestPreflightConfig:
    def _measure_module(self):
        return importlib.import_module(
            "ao_shaping.tools.slm.calibration.diffraction_efficiency.measure"
        )

    def test_the_gate_is_on_by_default(self, tmp_path: Path) -> None:
        module = self._measure_module()
        config = module.MeasurementConfig(
            calibration_path=tmp_path / "calib.npz", out_dir=tmp_path / "out"
        )
        assert config.focus_gate is True
        assert config.focus_auto_defocus is True
        assert config.focus_fwhm_max_px is None
        assert module.DEFAULT_FOCUS_FWHM_MAX_PX == pytest.approx(40.0)

    def test_too_few_focus_points_are_rejected(self, tmp_path: Path) -> None:
        module = self._measure_module()
        with pytest.raises(ValueError, match="focus_points"):
            module.MeasurementConfig(
                calibration_path=tmp_path / "calib.npz",
                out_dir=tmp_path / "out",
                focus_points=2,
            )

    def test_a_non_positive_fwhm_threshold_is_rejected(self, tmp_path: Path) -> None:
        module = self._measure_module()
        with pytest.raises(ValueError, match="focus_fwhm_max_px"):
            module.MeasurementConfig(
                calibration_path=tmp_path / "calib.npz",
                out_dir=tmp_path / "out",
                focus_fwhm_max_px=0.0,
            )

    def test_focus_only_does_not_need_the_geometry_calibration(
        self, tmp_path: Path
    ) -> None:
        """焦面扫描不用 Kx/Ky, 所以重新标定前也能先判在焦。"""
        module = self._measure_module()
        missing = tmp_path / "no_such_calib.npz"
        focus_only = module.MeasurementConfig(
            calibration_path=missing, out_dir=tmp_path / "a", focus_only=True
        )
        assert module.calibration_is_required(focus_only) is False
        # 正式扫描 (需要 ±1 级落点预测) 仍然必须已有标定。
        full_run = module.MeasurementConfig(
            calibration_path=missing, out_dir=tmp_path / "b"
        )
        assert module.calibration_is_required(full_run) is True


class TestMeasureCliSurface:
    def test_help_lists_the_preflight_flags(self) -> None:
        from click.testing import CliRunner

        module = importlib.import_module(
            "ao_shaping.tools.slm.calibration.diffraction_efficiency.__main__"
        )
        result = CliRunner().invoke(module.main, ["measure", "--help"])
        assert result.exit_code == 0, result.output
        for flag in (
            "--focus-gate",
            "--focus-fwhm-max-px",
            "--focus-diopter-range",
            "--focus-points",
            "--focus-refine",
            "--focus-auto-defocus",
            "--focus-only",
        ):
            assert flag in result.output

    def test_measure_command_parses_the_pupil_center_option(
        self, tmp_path: Path
    ) -> None:
        """``--pupil-center`` 必须是字符串类型 + ``parse_tuple`` 回调。

        历史缺陷: 注解写成 ``tuple[float, float]`` 时 click 把类型推成 FLOAT,
        于是拿默认值 ``(960.0, 600.0)`` 去做 ``float(...)``, 命令在**解析参数**
        阶段就 TypeError 崩掉 —— 也就是说这个命令以前从未真正从 CLI 跑过。
        """
        from click.testing import CliRunner

        module = importlib.import_module(
            "ao_shaping.tools.slm.calibration.diffraction_efficiency.__main__"
        )
        result = CliRunner().invoke(
            module.main,
            [
                "measure",
                "--calibration",
                str(tmp_path / "no_such_calib.npz"),
                "--out-dir",
                str(tmp_path / "out"),
                "--pupil-center",
                "960,600",
            ],
        )
        # 缺标定必须**按契约**失败 (RuntimeError/FileNotFoundError/ClickException),
        # 而不是参数解析阶段的 TypeError。注意**不能**加 ``--focus-only``: 那条路
        # 故意绕过标定前置条件, 会真的去开设备。
        assert result.exit_code != 0
        assert "float() argument" not in result.output

