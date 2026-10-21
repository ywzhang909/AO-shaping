"""数字孪生参数反演 (``bench_twin_calibrate``) 的物理断言。

被测对象: ``ao_shaping/tools/slm/bench_twin_calibrate.py``。本文件只断言
**物理口径**, 不锁死实现细节:

* **合成真值可复原** —— 造一台注入了已知误差 (LUT Gamma / 相机离焦 / 面板
  Zernike / 光路倾斜) 的孪生, 用它生成 (命令相位, 画面) 配对, 再用
  ``fit_twin`` 反演, 断言恢复出的参数落在容差内;
* **损失的性质** —— 实测画面被平移几个像素 ⇒ 损失**显著变大** (证明损失对
  平移敏感, 没有被「逐帧按 argmax 重新居中」藏掉); 实测整体缩放 20% ⇒
  **能量项**变大 (证明能量项用的是未归一化的绝对积分);
* **边界** —— 初值越界时, 反演结果仍落在 bounds 内;
* **确定性** —— 同 seed 两次 ``fit_twin`` 的结果逐位一致;
* **两阶段优于单阶段** —— Stage1+Stage2 的最终损失 ≤ 只跑 Stage 1 的损失;
* **存读往返** —— ``save_fit`` → ``load_fit`` 后配置一致, 且用读回的配置算出的
  ``far_field`` 与拟合时逐位相同;
* **预补偿** —— LUT Gamma 误差被补偿后, 实际相位与命令相位的残差 ≤ 一个量化级,
  且明显小于未补偿时。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.drivers.sim.bench_twin import (
    BenchTwin,
    BenchTwinConfig,
    TwinErrorModel,
)
from ao_shaping.tools.slm.bench_twin_calibrate import (
    DEFAULT_BOUNDS,
    TwinSample,
    _ssim_numpy,
    apply_parameter_vector,
    compensated_phase,
    fit_twin,
    load_fit,
    parameter_vector,
    save_fit,
    simulate_sample,
    ssim_backend,
    twin_loss,
)

# ---------------------------------------------------------------------------
# 合成真值 (几何刻意取小, 让整套反演在秒级完成)
# ---------------------------------------------------------------------------

REGION = 32
FAR_FIELD_PADDING = 4
REGION_SPAN_PX = 900.0
BEAM_W0_PX = 250.0
PEAK_ADU = 100.0

#: 注入的已知误差 (真值)。
TRUTH_GAMMA = 1.35
TRUTH_DEFOCUS_M = 3.0e-4
TRUTH_SLM_ZERNIKE = {4: 0.15, 5: -0.1}
TRUTH_TILT_X_RAD = 2.0e-4

#: 复原容差 (题目给定的口径)。
TOL_GAMMA = 0.15
TOL_DEFOCUS_M = 1.5e-4
TOL_ZERNIKE_RAD = 0.1
TOL_TILT_RAD = 1.0e-4


def _base_config(**overrides) -> BenchTwinConfig:
    payload = dict(
        region=REGION,
        region_span_px=REGION_SPAN_PX,
        far_field_padding=FAR_FIELD_PADDING,
        beam_w0_px=BEAM_W0_PX,
        peak_adu=PEAK_ADU,
        noise_adu=0.0,
        seed=0,
    )
    payload.update(overrides)
    return BenchTwinConfig(**payload)


def _command_phases(region: int = REGION) -> list[np.ndarray]:
    """4 张命令相位: 平场 / 倾斜 / 离焦型 / 倾斜+离焦混合。

    平场用来暴露面板静态相差与装调误差; 其余三张覆盖不同灰度区间, 让 LUT
    的 Gamma 非线性可辨识 (平场命令在 LUT 往返后仍是 0 ⇒ 不含 Gamma 信息)。
    """
    idx = np.arange(region, dtype=np.float64) - (region - 1) / 2.0
    x = np.broadcast_to(idx[None, :], (region, region))
    y = np.broadcast_to(idx[:, None], (region, region))
    r2 = x**2 + y**2
    half = region / 2.0
    return [
        np.zeros((region, region)),
        0.9 * math.pi * (x / half),
        1.8 * math.pi * (r2 / half**2 - 0.5),
        1.2 * math.pi * (x / half) + 0.6 * math.pi * (y / half),
    ]


def _truth_config() -> BenchTwinConfig:
    return _base_config(
        errors=TwinErrorModel(
            lut_gamma=TRUTH_GAMMA,
            defocus_m=TRUTH_DEFOCUS_M,
            slm_zernike=dict(TRUTH_SLM_ZERNIKE),
            slm_zernike_n_max=6,
            tilt_x_rad=TRUTH_TILT_X_RAD,
        )
    )


def _initial_config() -> BenchTwinConfig:
    """反演的起点: 几何与真值相同, 误差全 0 (但要声明「要反演哪些模式」)。"""
    return _base_config(
        errors=TwinErrorModel(
            slm_zernike={4: 0.0, 5: 0.0},
            slm_zernike_n_max=6,
        )
    )


def _truth_samples(config: BenchTwinConfig | None = None) -> list[TwinSample]:
    twin = BenchTwin(config or _truth_config())
    return [
        simulate_sample(twin, phase, label=f"s{index}")
        for index, phase in enumerate(_command_phases())
    ]


# ---------------------------------------------------------------------------
# 1. 合成真值可复原
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def truth_fit():
    """一次完整的两阶段反演 (模块级, 供多个断言复用)。"""
    samples = _truth_samples()
    result = fit_twin(
        samples,
        _initial_config(),
        maxiter=6,
        popsize=6,
        maxiter_fine=100,
        seed=0,
    )
    return result, samples


class TestSyntheticTruthRecovery:
    def test_recovered_parameters_are_within_tolerance(self, truth_fit) -> None:
        result, _samples = truth_fit
        errors = result.config.errors

        assert abs(errors.lut_gamma - TRUTH_GAMMA) <= TOL_GAMMA
        assert abs(errors.defocus_m - TRUTH_DEFOCUS_M) <= TOL_DEFOCUS_M
        assert abs(errors.tilt_x_rad - TRUTH_TILT_X_RAD) <= TOL_TILT_RAD
        for noll, value in TRUTH_SLM_ZERNIKE.items():
            assert abs(errors.slm_zernike[noll] - value) <= TOL_ZERNIKE_RAD

    def test_fit_beats_the_initial_configuration(self, truth_fit) -> None:
        result, samples = truth_fit
        initial_loss, _ = twin_loss(BenchTwin(_initial_config()), samples)
        assert result.loss < 0.01 * initial_loss

    def test_fit_is_deterministic_in_its_own_evaluation(self, truth_fit) -> None:
        """拟合出的配置重算损失, 必须与拟合报告的损失一致。"""
        result, samples = truth_fit
        loss, details = twin_loss(BenchTwin(result.config), samples)
        assert loss == pytest.approx(result.loss, rel=1e-9)
        assert len(details) == len(samples)
        assert all(entry["ssim"] > 0.99 for entry in details)

    def test_parameter_vector_round_trip(self) -> None:
        """参数向量 ↔ 配置 的映射必须互逆 (这是反演的核心契约)。"""
        config = _initial_config()
        for which in ("coarse", "fine"):
            vector = parameter_vector(config, which)
            assert vector.ndim == 1
            restored = apply_parameter_vector(config, vector, which)
            np.testing.assert_allclose(
                parameter_vector(restored, which), vector, rtol=0, atol=0
            )

    def test_measured_lut_table_is_not_fitted(self) -> None:
        """实测 LUT 表是**外部测量量**, 不参与反演 (拟合后必须逐位保留)。"""
        levels = 1023
        table = 2.0 * math.pi * (np.arange(levels + 1) / levels) ** 0.8
        config = _base_config(
            errors=TwinErrorModel(
                lut_table=table.copy(),
                slm_zernike={4: 0.0, 5: 0.0},
                slm_zernike_n_max=6,
            )
        )
        samples = _truth_samples()
        result = fit_twin(
            samples,
            config,
            maxiter=2,
            popsize=3,
            maxiter_fine=3,
            seed=0,
        )
        fitted = result.config.errors.lut_table
        assert fitted is not None
        np.testing.assert_array_equal(fitted, table)
        # LUT 表存在时正向模型只看表 (不再看 Gamma) ⇒ Gamma 是**不可辨识**方向。
        phase = _command_phases()[1]
        with_table = BenchTwin(config).far_field(phase)
        other_gamma = _base_config(
            errors=TwinErrorModel(
                lut_gamma=2.0,
                lut_table=table.copy(),
                slm_zernike={4: 0.0, 5: 0.0},
                slm_zernike_n_max=6,
            )
        )
        assert np.array_equal(with_table, BenchTwin(other_gamma).far_field(phase))


# ---------------------------------------------------------------------------
# 2. 损失的性质
# ---------------------------------------------------------------------------


class TestLossProperties:
    def test_shifting_the_measurement_increases_the_loss(self) -> None:
        """平移必须被惩罚 —— 若损失里含「按 argmax 重新居中」, 这里会失效。"""
        twin = BenchTwin(_truth_config())
        samples = _truth_samples()
        base_loss, base_details = twin_loss(twin, samples)

        shifted = [
            TwinSample(
                phase_rad=sample.phase_rad,
                image=np.roll(sample.image, shift=(3, -2), axis=(0, 1)),
                label=sample.label,
            )
            for sample in samples
        ]
        shifted_loss, shifted_details = twin_loss(twin, shifted)

        assert shifted_loss > base_loss + 0.1
        # 空间项 (SSIM / 相关) 必须明显变差, 能量项基本不变 (平移不改变总积分)。
        assert min(entry["ssim"] for entry in shifted_details) < min(
            entry["ssim"] for entry in base_details
        ) - 0.05
        assert min(entry["corr"] for entry in shifted_details) < min(
            entry["corr"] for entry in base_details
        ) - 0.05
        base_energy = max(entry["energy"] for entry in base_details)
        shifted_energy = max(entry["energy"] for entry in shifted_details)
        assert shifted_energy < base_energy + 1e-3

    def test_scaling_the_measurement_increases_the_energy_term(self) -> None:
        """能量项用**未归一化**的积分 ⇒ 实测整体缩放 20% ⇒ 能量项变大。"""
        twin = BenchTwin(_truth_config())
        samples = _truth_samples()
        base_loss, base_details = twin_loss(twin, samples)

        scaled = [
            TwinSample(
                phase_rad=sample.phase_rad,
                image=sample.image * 0.8,
                label=sample.label,
            )
            for sample in samples
        ]
        scaled_loss, scaled_details = twin_loss(twin, scaled)

        for base_entry, scaled_entry in zip(base_details, scaled_details):
            assert scaled_entry["energy"] > base_entry["energy"] * 10.0
        assert scaled_loss > base_loss

    def test_energy_term_uses_absolute_integrals(self) -> None:
        """能量项的定义就是 (Σsim − Σmeas)/Σmeas 的平方。"""
        twin = BenchTwin(_truth_config())
        samples = _truth_samples()
        _loss, details = twin_loss(twin, samples)
        for sample, entry in zip(samples, details):
            sim = twin.far_field(sample.phase_rad)
            expected = (float(sim.sum()) - float(sample.image.sum())) / float(
                sample.image.sum()
            )
            assert entry["energy_rel"] == pytest.approx(expected, rel=1e-12)
            assert entry["energy"] == pytest.approx(expected**2, rel=1e-12)

    def test_loss_of_a_perfect_match_is_zero(self) -> None:
        twin = BenchTwin(_truth_config())
        samples = _truth_samples()
        loss, details = twin_loss(twin, samples)
        assert loss == pytest.approx(0.0, abs=1e-12)
        assert all(entry["ssim"] == pytest.approx(1.0, abs=1e-12) for entry in details)

    def test_ssim_backend_is_reported(self) -> None:
        assert ssim_backend() in {"skimage", "numpy"}

    def test_weights_can_be_overridden(self) -> None:
        """权重可覆盖; 全零权重 ⇒ 联合损失为 0 (但明细仍是同一份度量)。"""
        twin = BenchTwin(_truth_config())
        samples = _truth_samples()
        zeroed, details = twin_loss(
            twin, samples, weights={"ssim": 0.0, "energy": 0.0, "corr": 0.0}
        )
        assert zeroed == pytest.approx(0.0, abs=1e-15)
        assert all(entry["loss"] == pytest.approx(0.0, abs=1e-15) for entry in details)

        _default_loss, default_details = twin_loss(twin, samples)
        for zero_entry, default_entry in zip(details, default_details):
            assert zero_entry["ssim"] == pytest.approx(default_entry["ssim"])
            assert zero_entry["energy"] == pytest.approx(default_entry["energy"])

    def test_numpy_ssim_fallback_is_a_valid_similarity(self) -> None:
        """自实现的 numpy 高斯窗 SSIM (skimage 不可用时的回落路径)。"""
        twin = BenchTwin(_truth_config())
        image = twin.far_field(_command_phases()[1])
        identical = _ssim_numpy(image, image)
        shifted = _ssim_numpy(image, np.roll(image, shift=4, axis=1))
        assert identical == pytest.approx(1.0, abs=1e-9)
        assert shifted < identical - 0.1
        assert -1.0 <= shifted <= 1.0


# ---------------------------------------------------------------------------
# 3. 边界
# ---------------------------------------------------------------------------


class TestBounds:
    def test_out_of_range_initial_values_stay_inside_the_bounds(self) -> None:
        """越界的初值不得让反演结果跑到边界外。"""
        out_of_range = _base_config(
            errors=TwinErrorModel(
                lut_gamma=5.0,  # 上界 2.5
                lut_phase_max_rad=9.0,  # 上界 2.5π
                defocus_m=2.0e-3,  # 上界 5e-4
                slm_zernike={4: 3.0, 5: -2.5},  # 上界 ±1
                slm_zernike_n_max=6,
                tilt_x_rad=0.4,  # 上界 0.05
                tilt_y_rad=-0.3,
            )
        )
        samples = _truth_samples()
        result = fit_twin(
            samples,
            out_of_range,
            maxiter=3,
            popsize=4,
            maxiter_fine=4,
            seed=0,
        )
        errors = result.config.errors
        assert DEFAULT_BOUNDS["lut_gamma"][0] <= errors.lut_gamma
        assert errors.lut_gamma <= DEFAULT_BOUNDS["lut_gamma"][1]
        assert DEFAULT_BOUNDS["defocus_m"][0] <= errors.defocus_m
        assert errors.defocus_m <= DEFAULT_BOUNDS["defocus_m"][1]
        assert DEFAULT_BOUNDS["tilt"][0] <= errors.tilt_x_rad
        assert errors.tilt_x_rad <= DEFAULT_BOUNDS["tilt"][1]
        for noll in (4, 5):
            assert DEFAULT_BOUNDS["slm_zernike"][0] <= errors.slm_zernike[noll]
            assert errors.slm_zernike[noll] <= DEFAULT_BOUNDS["slm_zernike"][1]

    def test_custom_bounds_are_respected(self) -> None:
        samples = _truth_samples()
        result = fit_twin(
            samples,
            _initial_config(),
            bounds={
                "coarse": {"lut_gamma": (0.9, 1.1)},
                "fine": {"tilt": (-0.001, 0.001)},
            },
            maxiter=2,
            popsize=3,
            maxiter_fine=3,
            seed=0,
        )
        errors = result.config.errors
        assert 0.9 <= errors.lut_gamma <= 1.1
        assert -0.001 <= errors.tilt_x_rad <= 0.001
        assert -0.001 <= errors.tilt_y_rad <= 0.001


# ---------------------------------------------------------------------------
# 4. 确定性
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_gives_bitwise_identical_results(self) -> None:
        samples = _truth_samples()
        kwargs = dict(maxiter=3, popsize=4, maxiter_fine=4, seed=7)
        first = fit_twin(samples, _initial_config(), **kwargs)
        second = fit_twin(samples, _initial_config(), **kwargs)

        assert first.loss == second.loss
        assert first.n_evaluations == second.n_evaluations
        assert parameter_vector(first.config, "coarse").tolist() == parameter_vector(
            second.config, "coarse"
        ).tolist()
        assert parameter_vector(first.config, "fine").tolist() == parameter_vector(
            second.config, "fine"
        ).tolist()


# ---------------------------------------------------------------------------
# 5. 两阶段优于单阶段
# ---------------------------------------------------------------------------


class TestTwoStages:
    def test_two_stages_are_not_worse_than_stage_one_alone(self, truth_fit) -> None:
        result, samples = truth_fit
        stage_one = fit_twin(
            samples,
            _initial_config(),
            stage1=True,
            stage2=False,
            maxiter=6,
            popsize=6,
            seed=0,
        )
        assert result.loss <= stage_one.loss
        assert result.stages["stage2_loss"] <= stage_one.stages["stage1_loss"]

    def test_stage_one_alone_reports_its_own_loss(self) -> None:
        samples = _truth_samples()
        stage_one = fit_twin(
            samples,
            _initial_config(),
            stage1=True,
            stage2=False,
            maxiter=3,
            popsize=4,
            seed=0,
        )
        assert "stage1_loss" in stage_one.stages
        assert "stage2_loss" not in stage_one.stages
        assert stage_one.loss == pytest.approx(
            stage_one.stages["stage1_loss"], rel=1e-12
        )


# ---------------------------------------------------------------------------
# 6. 存读往返
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_load_round_trip_preserves_the_configuration(self, truth_fit, tmp_path) -> None:
        result, samples = truth_fit
        path = save_fit(result, tmp_path / "fit.json")
        assert path.exists()

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "config" in payload and "loss" in payload and "stages" in payload

        restored = load_fit(path)
        assert restored.loss == pytest.approx(result.loss, rel=1e-12)
        assert restored.n_evaluations == result.n_evaluations
        assert parameter_vector(restored.config, "coarse").tolist() == parameter_vector(
            result.config, "coarse"
        ).tolist()
        assert parameter_vector(restored.config, "fine").tolist() == parameter_vector(
            result.config, "fine"
        ).tolist()

        phase = _command_phases()[2]
        original = BenchTwin(result.config).far_field(phase)
        reloaded = BenchTwin(restored.config).far_field(phase)
        np.testing.assert_array_equal(original, reloaded)
        assert len(restored.per_sample) == len(samples)


# ---------------------------------------------------------------------------
# 7. 预补偿
# ---------------------------------------------------------------------------


class TestCompensation:
    @pytest.fixture(scope="class")
    def gamma_only_fit(self):
        """只注入 LUT Gamma 误差的合成真值 (补偿残差的定量判据)。"""
        truth = _base_config(errors=TwinErrorModel(lut_gamma=TRUTH_GAMMA))
        twin = BenchTwin(truth)
        samples = [
            simulate_sample(twin, phase, label=f"g{index}")
            for index, phase in enumerate(_command_phases())
        ]
        result = fit_twin(
            samples,
            _base_config(),
            maxiter=6,
            popsize=6,
            maxiter_fine=30,
            seed=0,
        )
        return truth, samples, result

    def test_compensation_removes_the_gamma_error(self, gamma_only_fit) -> None:
        truth_config, _samples, result = gamma_only_fit
        truth = BenchTwin(truth_config)
        levels = int(truth.errors.gray_levels)

        # 命令相位取遍 [0, 2π) (SLM 是 2π 周期器件)。灰度值刻意取在量化边界
        # 的中间 (小数部分 0.5), 这样量化残差稳定地是「半级」而不是「贴着
        # 整数边界」—— 后者会被 float32↔float64 的最后一位抖动掀翻一级。
        grays = np.arange(REGION, dtype=np.float64) + 0.5
        grays = 0.05 * levels + (0.9 * levels - 0.05 * levels) * (
            grays / (REGION - 1.0)
        )
        cmd = 2.0 * math.pi * (grays / levels) ** TRUTH_GAMMA
        cmd = np.broadcast_to(cmd[None, :], (REGION, REGION)).copy()

        compensated = compensated_phase(result, cmd)
        actual = truth.apply_lut(truth.apply_panel_aberration(compensated))
        residual = np.abs(np.mod(actual - cmd + math.pi, 2.0 * math.pi) - math.pi)

        uncompensated = truth.apply_lut(truth.apply_panel_aberration(cmd))
        raw_residual = np.abs(
            np.mod(uncompensated - cmd + math.pi, 2.0 * math.pi) - math.pi
        )

        quantum = TRUTH_GAMMA * 2.0 * math.pi / levels
        assert float(residual.max()) <= quantum
        # 未补偿时的残差要大一个数量级以上。
        assert float(raw_residual.max()) > 10.0 * float(residual.max())

    def test_compensation_does_not_touch_the_truth_when_there_is_no_error(
        self, gamma_only_fit
    ) -> None:
        _truth_config, _samples, _result = gamma_only_fit
        ideal = BenchTwin(_base_config())
        cmd = 0.7 * math.pi * np.ones((REGION, REGION))
        # 理想孪生下补偿是恒等 (只差灰度量化)。
        result = fit_twin(
            [simulate_sample(ideal, cmd)],
            _base_config(),
            maxiter=2,
            popsize=4,
            maxiter_fine=4,
            seed=0,
        )
        compensated = compensated_phase(result, cmd)
        actual = ideal.apply_lut(ideal.apply_panel_aberration(compensated))
        residual = np.abs(np.mod(actual - cmd + math.pi, 2.0 * math.pi) - math.pi)
        assert float(residual.max()) <= 2.0 * math.pi / ideal.errors.gray_levels
