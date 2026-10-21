"""SLM 灰度→振幅耦合 (``TwinErrorModel.slm_amplitude_coupling``) 的物理断言。

被测对象: ``ao_shaping/drivers/sim/bench_twin.py`` 新增的灰度域一次谐波模型::

    I(g) = 1 + c·sin(2π·g/gray_levels + ψ),  eta(g) = sqrt(max(I, 0))

``eta`` 乘在瞳面场幅度上 (LCOS 处, 透镜像差与 FFT 之前); ``c = 0`` 必须逐位恒等于
旧模型。模型形式的依据是仓库自己的测量代码
(``tools/slm/calibration/gray_response.py`` 的灰度域正弦拟合与
``tools/slm/panel/diagnose.py`` 的 ``rel_spread > 0.15`` 调制判据)。

⚠️ 仓库的 ``addopts`` 把 ``tests/ao_shaping/drivers`` 排除在默认 discovery 之外
(硬件测试会阻塞), 本文件只在**显式给出该路径**时运行, 例如
``./run_pytest.sh tests/ao_shaping/drivers/sim`` —— 这是预期行为, 不是遗漏。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ao_shaping.drivers.sim.bench_twin import (
    BenchTwin,
    BenchTwinConfig,
    TwinErrorModel,
)

GOLDEN_PATH = Path(__file__).with_name("golden_far_field_c0_pre_coupling.npz")

# 演示用的耦合幅度 (面板真值仓库未记录, 这里只取一个可观测的量级)。
C_TEST = 0.15


def _twin(**errors: object) -> BenchTwin:
    return BenchTwin(BenchTwinConfig(errors=TwinErrorModel(**errors)))  # type: ignore[arg-type]


def _varying_phase(region: int, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-np.pi, np.pi, (region, region))


# ---------------------------------------------------------------------------
# 1. 默认关闭
# ---------------------------------------------------------------------------


def test_default_error_model_has_no_coupling() -> None:
    errors = TwinErrorModel()
    assert errors.slm_amplitude_coupling == 0.0
    assert errors.slm_amplitude_coupling_phase_rad == 0.0


def test_coupling_factor_is_exactly_one_when_coupling_is_zero() -> None:
    """c=0 时因子必须**精确**为 1.0 (array_equal, 不是 allclose) —— 后面
    bit-identity 断言的前提就是这一点。"""
    twin = _twin()
    gray = np.arange(twin.region * twin.region, dtype=np.uint16).reshape(
        twin.region, twin.region
    )
    eta = twin.apply_amplitude_coupling(gray)
    assert np.array_equal(eta, np.ones_like(eta))


def test_far_field_is_bit_identical_to_the_pre_change_model_when_coupling_is_zero() -> None:
    """c=0 时远场必须与**改动前**的实现逐位相同。

    为什么用 bit-identity 而不是容差: 耦合在 c=0 必须是纯粹的 no-op ——
    仓库里已经记录的所有远场数字 (报告、golden、标定) 都要继续有效;
    若只要求 "足够接近", 那些数字会在无人察觉时发生漂移。

    golden 的出处 (fixture 随仓库提交): ``golden_far_field_c0_pre_coupling.npz``
    于 2026-10-22 从**改动前**的实现捕获 ——
    ``git show HEAD:src/ao_shaping/drivers/sim/bench_twin.py``
    (HEAD a3828f95, 捕获时该文件在工作区未改动), 在 ao-shaping-test:gpu
    容器里以 ``BenchTwin(BenchTwinConfig())`` (默认配置, 无任何误差族) 对
    ``np.random.default_rng(20261022).uniform(-pi, pi, (128, 128))`` 跑一次
    ``far_field``。npz 里同时存了该孪生的 ``to_dict`` 配置, 重建走
    ``from_dict``, 不依赖默认值。
    """
    fixture = np.load(GOLDEN_PATH)
    phase = fixture["phase"]
    golden = fixture["far_field"]
    twin = BenchTwin.from_dict(json.loads(str(fixture["config_json"])))
    assert twin.errors.slm_amplitude_coupling == 0.0
    assert twin.errors.gray_levels == 1023
    assert np.array_equal(twin.far_field(phase), golden)


# ---------------------------------------------------------------------------
# 2. 灰度域一次谐波的结构
# ---------------------------------------------------------------------------


def test_coupling_factor_has_one_2pi_period_in_grey() -> None:
    """周期 = gray_levels (一个 2π 灰度), 这正是仿真要把 ``gray_levels``
    设成器件 2π 灰度数 (1064 nm 下 993) 的原因。"""
    levels = 993
    twin = _twin(
        gray_levels=levels,
        slm_amplitude_coupling=C_TEST,
        slm_amplitude_coupling_phase_rad=0.7,
    )
    gray = np.linspace(0.0, levels, 257)
    one_period = twin.apply_amplitude_coupling(gray)
    shifted = twin.apply_amplitude_coupling(gray + levels)
    np.testing.assert_allclose(one_period, shifted, rtol=0.0, atol=1e-12)
    # 非平凡性: 一个周期内因子必须真的起伏 (否则上面的断言是空话)。
    assert float(one_period.max() - one_period.min()) > 0.1


def test_coupling_factor_dark_and_bright_states() -> None:
    """ψ=0: g=0 ⇒ 正弦过零 ⇒ 因子精确 1.0; g=3·levels/4 ⇒ 幅角 3π/2 (暗态)
    ⇒ sqrt(1-c); g=levels/4 ⇒ π/2 (亮态) ⇒ sqrt(1+c)。"""
    levels = 1000
    c = 0.3
    twin = _twin(
        gray_levels=levels, slm_amplitude_coupling=c, slm_amplitude_coupling_phase_rad=0.0
    )
    gray = np.array([0.0, levels / 4.0, 3.0 * levels / 4.0])
    eta = twin.apply_amplitude_coupling(gray)
    assert eta[0] == 1.0
    np.testing.assert_allclose(eta[1], math.sqrt(1.0 + c), rtol=1e-12)
    np.testing.assert_allclose(eta[2], math.sqrt(1.0 - c), rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. 可观测性边界
# ---------------------------------------------------------------------------


def test_flat_phase_coupling_is_invisible_after_peak_normalisation() -> None:
    """均匀命令相位 ⇒ 灰度均匀 ⇒ eta 是全局标度 ⇒ 峰值归一后远场不变。

    这就是可观测性边界: 孪生的输出约定 (峰值归一到 ``peak_adu``) 吸收全局幅度
    因子, 只有**空间变化**的耦合分量可见。仓库因此用灰度扫描 0 级桶
    (diagnose/gray_response) 测这个效应, 而不是用归一化远场。
    浮点路径不同 (标量乘进 FFT 再归一 vs 直接归一), 断言用紧容差而非 array_equal。
    """
    phase = np.full((64, 64), 1.1, dtype=np.float64)
    without = BenchTwin(
        BenchTwinConfig(region=64, errors=TwinErrorModel())
    ).far_field(phase)
    with_coupling = BenchTwin(
        BenchTwinConfig(
            region=64,
            errors=TwinErrorModel(
                slm_amplitude_coupling=C_TEST,
                slm_amplitude_coupling_phase_rad=0.9,
            ),
        )
    ).far_field(phase)
    np.testing.assert_allclose(without, with_coupling, rtol=1e-12, atol=1e-10)


def test_spatially_varying_phase_is_affected_by_the_coupling() -> None:
    """非均匀相位 ⇒ 灰度逐点不同 ⇒ eta 是**空间**调制 ⇒ 整形后的远场可测量地改变。"""
    phase = _varying_phase(64)
    without = BenchTwin(
        BenchTwinConfig(region=64, errors=TwinErrorModel())
    ).far_field(phase)
    with_coupling = BenchTwin(
        BenchTwinConfig(region=64, errors=TwinErrorModel(slm_amplitude_coupling=C_TEST))
    ).far_field(phase)
    assert not np.array_equal(without, with_coupling)
    assert float(np.max(np.abs(without - with_coupling))) > 1.0  # 峰值标度 100 ADU
    assert float(np.corrcoef(without.ravel(), with_coupling.ravel())[0, 1]) < 0.999


# ---------------------------------------------------------------------------
# 4. 序列化
# ---------------------------------------------------------------------------


def test_dict_roundtrip_preserves_the_new_fields() -> None:
    twin = _twin(
        slm_amplitude_coupling=C_TEST,
        slm_amplitude_coupling_phase_rad=1.234,
    )
    payload = twin.to_dict()
    assert payload["errors"]["slm_amplitude_coupling"] == C_TEST
    assert payload["errors"]["slm_amplitude_coupling_phase_rad"] == 1.234

    restored = BenchTwin.from_dict(json.loads(json.dumps(payload)))
    assert restored.to_dict() == payload
    phase = _varying_phase(twin.region, seed=13)
    assert np.array_equal(restored.far_field(phase), twin.far_field(phase))
