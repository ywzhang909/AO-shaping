from __future__ import annotations

import numpy as np

from ao_shaping.utils.pattern_helper import PatternHelper


class TestPatternHelperZernikeCaching:
    """Zernike generation caching + aperture masking (regression for D1/D2)."""

    def test_repeat_generation_identical_and_cached(self):
        """同参数重复生成必须逐元素一致, 且复用缓存的 ZernikeGenerator."""
        ph = PatternHelper((400, 300), bits=10)
        coeffs = {(0, 0): 1.0, (2, 0): 0.5}
        a = ph.generate_zernike_polynomial(coefficients=coeffs, radius=150.0)
        b = ph.generate_zernike_polynomial(coefficients=coeffs, radius=150.0)
        np.testing.assert_array_equal(a, b)
        # 同一 (radius, n_orders) 只构建一次 generator
        assert len(ph._zernike_generators) == 1

    def test_radius_controls_aperture_and_masks_outside(self):
        """『孔径半径』必须真正改变孔径, 且孔径外置 0 (回归: 之前 radius 被忽略)."""
        ph = PatternHelper((200, 200), bits=10)
        coeffs = {(0, 0): 1.0, (2, 0): 0.5}
        small = ph.generate_zernike_polynomial(coefficients=coeffs, radius=50.0)
        large = ph.generate_zernike_polynomial(coefficients=coeffs, radius=100.0)
        assert not np.array_equal(small, large)
        # 角落 (距中心 ~141px) 在两个孔径外 → 必须为 0
        assert small[0, 0] == 0
        assert large[0, 0] == 0
        # 中心在两个孔径内 → 非零
        assert small[100, 100] > 0
        assert large[100, 100] > 0

    def test_piston_only_constant_inside_aperture(self):
        """纯 piston: 孔径内为常数灰度, 孔径外为 0."""
        ph = PatternHelper((200, 200), bits=10)
        img = ph.generate_zernike_polynomial(coefficients={(0, 0): 1.0}, radius=100.0)
        inside = img[img > 0]
        assert inside.size > 0
        assert np.all(inside == inside[0])
        assert img[0, 0] == 0  # 孔径外为 0

    def test_n_max_kwarg_accepted(self):
        """n_max 关键字必须被接受 (slm_zernike_pib / phase_capture 依赖)."""
        ph = PatternHelper((200, 200), bits=10)
        img = ph.generate_zernike_polynomial(
            n_max=4,
            coefficients={(0, 0): 1.0, (2, 0): 0.5},
            radius=100.0,
        )
        assert img.shape == (200, 200)
        assert img.dtype == np.float64  # 2026-09-15: 只返回原始弧度相位
        assert img[100, 100] > 0

    def test_generate_zernike_single_masked(self):
        """generate_zernike 单模式同样受孔径约束并掩模孔径外."""
        ph = PatternHelper((200, 200), bits=10)
        img = ph.generate_zernike(2, 0, amplitude=1.0, radius=50.0)
        assert img.shape == (200, 200)
        assert img.dtype == np.float64  # 2026-09-15: 只返回原始弧度相位
        assert img[0, 0] == 0  # 孔径外为 0
        # 孔径内 (距中心 20px < 半径 50) 非零 (原始未包裹离焦在此处为负值)
        assert img[100, 120] != 0

    def test_radius_change_clears_generators(self):
        """半径变化 → 旧 generator 全部释放，字典只含新半径的条目 (完全重建)."""
        ph = PatternHelper((200, 200), bits=10)
        coeffs = {(0, 0): 1.0, (2, 0): 0.5}
        ph.generate_zernike_polynomial(coefficients=coeffs, radius=100.0)
        ph.generate_zernike_polynomial(coefficients=coeffs, radius=100.0, n_max=4)
        assert len(ph._zernike_generators) == 2  # (100,6) + (100,4)

        # 切换半径 → 旧的全部清空，只留新半径
        ph.generate_zernike_polynomial(coefficients=coeffs, radius=150.0)
        assert len(ph._zernike_generators) == 1
        assert (150.0, 6) in ph._zernike_generators

    def test_same_radius_reuses_generators(self):
        """同半径重复生成 → 复用缓存，字典不增长."""
        ph = PatternHelper((200, 200), bits=10)
        coeffs = {(0, 0): 1.0}
        a = ph.generate_zernike_polynomial(coefficients=coeffs, radius=80.0)
        b = ph.generate_zernike_polynomial(coefficients=coeffs, radius=80.0)
        c = ph.generate_zernike_polynomial(coefficients=coeffs, radius=80.0)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(b, c)
        assert len(ph._zernike_generators) == 1  # 始终只有 1 个