"""合成数据测试：逐轮 SLM 相位单调性诊断脚本（离线，不依赖 data / cuda / 真实优化器）。

按路径 importlib 加载脚本（它位于 ``scripts/slm``），用合成的相位栈覆盖纯函数：
增量形状与数值、逐像素单调性（单调栈 / 随机符号栈 / active mask）、逐轮方向
一致性、lr 建议三分支，以及面板映射（mock spec + mock main_mod）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "slm" / "replay_diff_phase_monotonicity.py"


def _load_script():
    """从文件路径加载诊断脚本（scripts/slm 不是稳定的包导入路径）。"""
    spec = importlib.util.spec_from_file_location("replay_diff_phase_monotonicity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


diag = _load_script()


# ---------------------------------------------------------------------------
# phase_delta_stack
# ---------------------------------------------------------------------------


def test_phase_delta_stack_shape_and_values() -> None:
    # (N=3, H=1, W=2)：只有一列像素，逐帧为 [0,1] → [2,4] → [5,8]。
    phases = np.array([[[0.0, 1.0]], [[2.0, 4.0]], [[5.0, 8.0]]])
    deltas = diag.phase_delta_stack(phases)
    assert deltas.shape == (2, 1, 2)
    np.testing.assert_allclose(deltas, [[[2.0, 3.0]], [[3.0, 4.0]]])


def test_phase_delta_stack_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        diag.phase_delta_stack(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        diag.phase_delta_stack(np.zeros((1, 3, 3)))


# ---------------------------------------------------------------------------
# pixel_monotonicity
# ---------------------------------------------------------------------------


def test_pixel_monotonicity_perfectly_monotonic_stack() -> None:
    # phases[t] = t*base, base 各元素非零 → 每像素增量符号全一致。
    base = np.array([[1.0, -2.0], [0.5, 3.0]])
    phases = np.stack([t * base for t in range(6)])
    metrics = diag.pixel_monotonicity(diag.phase_delta_stack(phases))
    assert metrics["frac_pixels_monotonic"] == pytest.approx(1.0)
    assert metrics["mean_sign_consistency"] == pytest.approx(1.0)
    assert metrics["frac_active_pixels"] == pytest.approx(1.0)
    np.testing.assert_array_equal(metrics["active_mask"], np.ones((2, 2), dtype=bool))
    np.testing.assert_allclose(metrics["sign_consistency_map"], np.ones((2, 2)))


def test_pixel_monotonicity_random_signs_low() -> None:
    rng = np.random.default_rng(42)
    deltas = rng.normal(size=(40, 8, 8))
    metrics = diag.pixel_monotonicity(deltas)
    # 随机符号的一致性约 0.5 + 微小样本噪声，远低于单调阈值 0.9。
    assert metrics["frac_pixels_monotonic"] < 0.2
    assert 0.5 <= metrics["mean_sign_consistency"] < 0.62


def test_pixel_monotonicity_active_mask_excludes_static_pixels() -> None:
    # [step, H, W]。W=0 列恒 +1；[H0,W1] 恒 +1；[H1,W1] 先 +1 后 -1 净变化为 0 → 非 active。
    deltas = np.zeros((4, 2, 2))
    deltas[:, :, 0] = 1.0
    deltas[:, 0, 1] = 1.0
    deltas[:2, 1, 1] = 1.0
    deltas[2:, 1, 1] = -1.0
    metrics = diag.pixel_monotonicity(deltas, min_total_change=1e-3)
    expected = np.array([[True, True], [True, False]])
    np.testing.assert_array_equal(metrics["active_mask"], expected)
    assert metrics["frac_active_pixels"] == pytest.approx(0.75)
    # 三个 active 像素的总变化均为 4（W=0 两行 +4，H0W1 +4）。
    assert metrics["mean_total_change_active"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# per_iteration_monotonicity
# ---------------------------------------------------------------------------


def test_per_iteration_monotonicity_all_consistent() -> None:
    deltas = np.stack([np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))])
    active = np.ones((2, 2), dtype=bool)
    result = diag.per_iteration_monotonicity(deltas, active)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, [1.0, 1.0])


def test_per_iteration_monotonicity_flip_and_zero() -> None:
    # 第 1→2 步：两像素都翻转（0 一致）；第 2→3 步：一步零（0 保护），一步继续为负（一致）。
    d0 = np.array([[1.0, 1.0], [1.0, 1.0]])
    d1 = np.array([[-1.0, -1.0], [-1.0, -1.0]])
    d2 = np.array([[0.0, -1.0], [0.0, -1.0]])
    result = diag.per_iteration_monotonicity(np.stack([d0, d1, d2]), np.ones((2, 2), bool))
    np.testing.assert_allclose(result, [0.0, 0.5])


def test_per_iteration_monotonicity_short_stack() -> None:
    assert diag.per_iteration_monotonicity(np.ones((1, 2, 2)), np.ones((2, 2), bool)).size == 0


# ---------------------------------------------------------------------------
# lr_recommendation（三分支）
# ---------------------------------------------------------------------------


def test_lr_recommendation_three_branches() -> None:
    high = diag.lr_recommendation(
        {"frac_pixels_monotonic": 0.85, "mean_total_change_active": 1.0}
    )
    mid = diag.lr_recommendation(
        {"frac_pixels_monotonic": 0.55, "mean_total_change_active": 1.0}
    )
    low = diag.lr_recommendation(
        {"frac_pixels_monotonic": 0.2, "mean_total_change_active": 1.0}
    )
    assert "高度单调" in high and "放大 lr" in high
    assert "中等" in mid and "A/B" in mid
    assert "不建议放大 lr" in low


def test_lr_recommendation_saturated_high_fraction_falls_back() -> None:
    # 单调比例高但相位变化已饱和（≥π）→ 不应再建议放大 lr。
    saturated = diag.lr_recommendation(
        {"frac_pixels_monotonic": 0.9, "mean_total_change_active": 4.0}
    )
    assert "不建议放大 lr" in saturated


# ---------------------------------------------------------------------------
# map_phases_to_panel（合成 spec + mock main_mod）
# ---------------------------------------------------------------------------


def test_map_phases_to_panel_shape_with_mock() -> None:
    # 伪造 spec（属性与 phase_model_to_panel 用到的字段一致）。
    spec = SimpleNamespace(
        pupil_center=(6.0, 6.0), n_pupil=4, dp=2.0, m_grid=12, block_centre=6.5
    )

    def fake_phase_model_to_panel(phase_model: np.ndarray, spec_: object) -> np.ndarray:
        assert spec_ is spec
        out = np.zeros((14, 16), dtype=np.float64)
        out[2:6, 2:6] = phase_model[0:4, 0:4]
        return out

    main_mod = SimpleNamespace(phase_model_to_panel=fake_phase_model_to_panel)
    # +1 保证没有 0 值，方便 count_nonzero 断言。
    phases = np.arange(2 * 12 * 12, dtype=np.float64).reshape(2, 12, 12) + 1.0
    mapped = diag.map_phases_to_panel(phases, spec, indices=[0, 1], main_mod=main_mod)
    assert mapped.shape == (2, 14, 16)
    np.testing.assert_allclose(mapped[0][2:6, 2:6], phases[0][0:4, 0:4])
    np.testing.assert_allclose(mapped[1][2:6, 2:6], phases[1][0:4, 0:4])
    assert np.count_nonzero(mapped[0]) == 16
