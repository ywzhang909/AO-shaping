"""``beam_simulation.zernike_polynomial`` 的数值特征测试 (R-50 / #66 残留)。

钉住的理由 (不是风格问题):
``beam_simulation.zernike_polynomial`` 现在把径向部分委托给权威
:func:`ao_shaping.utils.wavefront.zernike_calc.zernike_radial` (aotools ``RZern``)。
在此之前它是仓库里的第三套手写实现 (scipy ``factorial`` 直接展开, 未归一化)。
2026-10-14 的 ``531fc1fb`` 以「全仓零调用点」为由删掉了它 —— 该声称**是错的**
(:func:`ao_shaping.optimizer.rl.envs.StaticAberrationAOEnv._sample_disturbance`
一直在调用), 于是 RL 环境的 ``reset`` 直接 ``AttributeError``, 直到 ``6d5e0f6d``
把委托版补回。换实现等于换归一化约定 ⇒ **强化学习的相位量已被静默改动过, 且当时
没有任何测试拦住它**。本文件补的就是这一层: 委托关系、归一化约定、逐模式数值。

数值口径: 全部离线、无硬件、无随机数; 期望值由容器内实测写入 (见各用例的
``rtol``/``atol``)。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ao_shaping.drivers.sim.beam_simulation import normalize_rho, zernike_polynomial
from ao_shaping.utils.wavefront.zernike_calc import noll_to_nm, zernike_radial

#: Noll 1..20 (n<=6), 覆盖 ``envs.py`` 默认 ``n_zernike_modes=10`` 的取样区间有余。
NOLL_MODES: list[tuple[int, int, int]] = [(j,) + noll_to_nm(j) for j in range(1, 21)]

RHO_SAMPLE = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
THETA_SAMPLE = np.array([0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 3])


def test_radial_part_is_delegated_not_reimplemented() -> None:
    """委托契约: 角度之外必须与权威径向**逐位相同**。

    ``rtol=0, atol=0`` 是刻意的 —— 这条用例防的是「又长出第二份径向实现」，
    那种实现往往只差一个尺度因子, 用宽松容差恰好漏过。
    """
    for j, n, m in NOLL_MODES:
        radial = zernike_radial(n, abs(m), RHO_SAMPLE)
        expected = (
            radial * np.cos(abs(m) * THETA_SAMPLE)
            if m >= 0
            else radial * np.sin(abs(m) * THETA_SAMPLE)
        )
        np.testing.assert_allclose(
            zernike_polynomial(n, m, RHO_SAMPLE, THETA_SAMPLE), expected, rtol=0, atol=0
        )


def test_endpoint_matches_aotools_normalization_convention() -> None:
    """瞳边缘幅值是 aotools 归一化的**指纹**: ``|R_n^m(1)| = sqrt(n+1)`` (m=0) / ``sqrt(2(n+1))`` (m!=0)。

    被删掉的手写教科书径向在 ``rho=1`` 处恒为 1, 因此一旦有人把委托换回未归一化
    实现 (或换到另一种归一化), 这条就会炸 —— 这正是 2026-10-17 那次静默换算所缺的守卫。
    """
    one = np.array([1.0])
    for _j, n, m in NOLL_MODES:
        expected = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))
        assert abs(float(zernike_radial(n, abs(m), one)[0])) == pytest.approx(
            expected, rel=1e-12
        )


def test_radial_values_at_half_pupil_are_pinned() -> None:
    """``rho = 0.5`` 处 20 个模式的径向值逐一钉死 (实测期望)。"""
    expected = {
        (0, 0): 1.0,
        (1, 1): 1.0,
        (2, 0): -0.8660254037844386,
        (2, 2): 0.6123724356957945,
        (3, 1): -1.7677669529663689,
        (3, 3): 0.3535533905932738,
        (4, 0): -0.2795084971874737,
        (4, 2): -1.5811388300841898,
        (4, 4): 0.19764235376052372,
        (5, 1): 1.0825317547305482,
        (5, 3): -1.190784930203603,
        (5, 5): 0.10825317547305482,
        (6, 0): 1.1575161985907585,
    }
    half = np.array([0.5])
    for (n, m), want in expected.items():
        assert float(zernike_radial(n, m, half)[0]) == pytest.approx(want, rel=1e-12)


def test_radial_is_the_textbook_polynomial_times_the_convention_scale() -> None:
    """径向 = **教科书多项式 × 约定尺度** (`sqrt(n+1)` / `sqrt(2(n+1))`)。

    这是比端点更强的守卫: 端点只能抓住归一化换掉, 这条同时钉住**多项式本身**。
    仓库里被删掉的手写实现 (`531fc1fb` 之前, ``scipy.special.factorial`` 直接展开)
    就是教科书未归一化版本 —— 用它当独立参考, 就能把「委托换实现」与「换归一化约定」
    两种漂移分开暴露。参考式在此**有意**重复一遍: 它是测试里的参照物, 不是第二份生产
    实现 (同 `test_bench_twin.py` 里手算 `_reference_far_field` 的做法)。
    """
    rho = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    for n, m in [(0, 0), (1, 1), (2, 0), (2, 2), (3, 1), (3, 3), (4, 0), (4, 2), (5, 1), (5, 5), (6, 0), (6, 2)]:
        textbook = np.zeros_like(rho)
        for k in range((n - m) // 2 + 1):
            coeff = (
                (-1) ** k
                * math.factorial(n - k)
                / (
                    math.factorial(k)
                    * math.factorial((n + m) // 2 - k)
                    * math.factorial((n - m) // 2 - k)
                )
            )
            textbook = textbook + coeff * rho ** (n - 2 * k)
        scale = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))
        np.testing.assert_allclose(
            zernike_radial(n, m, rho), textbook * scale, rtol=1e-12, atol=1e-12
        )


def test_signed_m_selects_the_sine_component() -> None:
    """``m >= 0`` 用 cos, ``m < 0`` 用 sin —— 与 ``ZernikeGenerator`` 同一约定。"""
    for j, n, m in NOLL_MODES:
        if m == 0:
            continue
        pos = zernike_polynomial(n, abs(m), RHO_SAMPLE, THETA_SAMPLE)
        neg = zernike_polynomial(n, -abs(m), RHO_SAMPLE, THETA_SAMPLE)
        radial = zernike_radial(n, abs(m), RHO_SAMPLE)
        np.testing.assert_allclose(pos, radial * np.cos(abs(m) * THETA_SAMPLE), rtol=0, atol=0)
        np.testing.assert_allclose(neg, radial * np.sin(abs(m) * THETA_SAMPLE), rtol=0, atol=0)
        assert not np.array_equal(pos, neg)


def test_normalize_rho_divides_by_the_grid_max_not_the_pupil_radius() -> None:
    """特征测试 (钉现状, **不代表认可**): ``rho`` 归一到**网格最远点** = 1。

    网格角点到中心的距离最大, 所以圆形瞳边缘上的 ``rho`` 小于 1。改动它会让
    ``envs.py`` 取样的 Zernike 相位整体换尺度, 因此先钉住现状; 若要改成按瞳半径
    归一, 必须显式更新本用例并连带复核 RL 训练量纲。
    """
    x = np.array([[-1.0, 0.0], [0.0, 1.0]])
    y = np.array([[0.0, 1.0], [-1.0, 0.0]])
    np.testing.assert_allclose(normalize_rho(x, y), np.ones((2, 2)), rtol=0, atol=0)

    grid_x, grid_y = np.meshgrid(np.linspace(-1.0, 1.0, 3), np.linspace(-1.0, 1.0, 3))
    rho = normalize_rho(grid_x, grid_y)
    assert float(rho.max()) == pytest.approx(1.0)
    assert float(rho[1, 2]) == pytest.approx(1.0 / math.sqrt(2.0))
