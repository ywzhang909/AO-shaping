from __future__ import annotations

import numpy as np

from ao_shaping.utils.wavefront.pattern_helper import PatternHelper
from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator


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


class TestZernikeRoundTrip:
    """Zernike 相位生成 → 灰度化 → 反解 → 拟合 完整回归测试.

    验证 PatternHelper 生成的 Zernike 相位经过 SLM 灰度化管线后,
    反解回弧度相位再拟合 Zernike 系数, 结果与原始系数一致.
    使用 ``phase_to_slm_grayscale(slm=None)`` 纯数学后端, 无需真实硬件.

    注意: SLM 只能显示 [0, 2π) 相位, ``phase_to_slm_grayscale`` 内部会做
    ``np.mod(phase, 2π)``. 这会破坏 Zernike 多项式的零均值结构, 导致拟合系数
    与原始系数不同. 测试验证的是:
    1. 灰度化数学正确性: phase → gray → phase_recovered ≡ phase (mod 2π)
    2. 孔径外为 0 保持不变
    3. 大幅度相位正确取模

    量化误差: 10-bit 灰度 (1024 级) 对应相位量化步长 ≈ 2π/1024 ≈ 0.0061 rad.
    测试容差设为略大于量化步长以容忍截断误差.
    """

    # 10-bit 灰度量化导致的相位误差上界 (使用 1023 而非 1024 更保守)
    QUANTIZATION_TOL = 2 * np.pi / 1023 + 1e-10  # ≈ 0.00614 rad

    def _round_trip_gray(self, phase_rad, max_grayscale=1023):
        """灰度化往返: phase → gray → phase_recovered."""
        gray = phase_to_slm_grayscale(phase_rad, max_grayscale=max_grayscale, slm=None)
        phase_recovered = (gray.astype(np.float64) / max_grayscale) * (2 * np.pi)
        return phase_recovered, gray

    def _get_mask(self, resolution, radius):
        """获取与 generate_zernike_polynomial 一致的孔径掩模."""
        gen = ZernikeGenerator(resolution=resolution, radius=radius, n_orders=6)
        return gen.mask.astype(bool)

    def test_grayscale_round_trip_math_correctness(self):
        """验证弧度 → 灰度 → 弧度数学等价性 (模 2π)."""
        resolution = (200, 200)
        radius = 80.0
        ph = PatternHelper(resolution, bits=10)
        coeffs = {(2, 0): 1.0, (2, -2): 0.5, (3, 1): -0.3}
        phase_rad = ph.generate_zernike_polynomial(
            coefficients=coeffs, radius=radius, n_max=6
        )
        mask = self._get_mask(resolution, radius)

        phase_recovered, gray = self._round_trip_gray(phase_rad)

        # 反解相位应等于原相位模 2π (在孔径内), 容忍量化误差
        expected = np.mod(phase_rad[mask], 2 * np.pi)
        actual = phase_recovered[mask]
        np.testing.assert_allclose(actual, expected, atol=self.QUANTIZATION_TOL)

        # 灰度值应在有效范围内
        assert np.all(gray[mask] >= 0)
        assert np.all(gray[mask] <= 1023)

    def test_aperture_mask_preserved(self):
        """孔径外区域保持为 0 (灰度 0)."""
        ph = PatternHelper((200, 200), bits=10)
        coeffs = {(2, 0): 1.0}
        phase_rad = ph.generate_zernike_polynomial(
            coefficients=coeffs, radius=50.0, n_max=6
        )
        mask = self._get_mask((200, 200), 50.0)

        phase_recovered, gray = self._round_trip_gray(phase_rad)

        # 孔径外原相位为 0
        assert np.all(phase_rad[~mask] == 0)
        # 灰度化后孔径外也为 0
        assert np.all(gray[~mask] == 0)
        # 反解后孔径外也为 0
        assert np.all(phase_recovered[~mask] == 0)

    def test_piston_modulo_preserved(self):
        """Piston (常数相位) 正确取模."""
        resolution = (200, 200)
        radius = 80.0
        ph = PatternHelper(resolution, bits=10)
        # 3 rad piston
        phase_rad = ph.generate_zernike_polynomial(
            coefficients={(0, 0): 3.0}, radius=radius, n_max=6
        )
        mask = self._get_mask(resolution, radius)

        phase_recovered, _ = self._round_trip_gray(phase_rad)
        # 3 rad mod 2π = 3 rad
        expected = 3.0 % (2 * np.pi)
        np.testing.assert_allclose(
            phase_recovered[mask], expected, atol=self.QUANTIZATION_TOL
        )

    def test_defocus_modulo_preserved(self):
        """Defocus 相位正确取模 (原相位有正负, 模 2π 后全为正)."""
        resolution = (200, 200)
        radius = 80.0
        ph = PatternHelper(resolution, bits=10)
        phase_rad = ph.generate_zernike_polynomial(
            coefficients={(2, 0): 1.0}, radius=radius, n_max=6
        )
        mask = self._get_mask(resolution, radius)

        phase_recovered, _ = self._round_trip_gray(phase_rad)

        # 验证模 2π 等价性, 容忍量化误差
        expected = np.mod(phase_rad[mask], 2 * np.pi)
        np.testing.assert_allclose(
            phase_recovered[mask], expected, atol=self.QUANTIZATION_TOL
        )
        # 反解相位全在 [0, 2π) 内
        assert np.all(phase_recovered[mask] >= 0)
        assert np.all(phase_recovered[mask] < 2 * np.pi)

    def test_large_amplitude_wrapping(self):
        """大幅度相位 (多圈 2π) 正确取模."""
        resolution = (200, 200)
        radius = 80.0
        ph = PatternHelper(resolution, bits=10)
        # 5 rad ≈ 0.8 圈, 10 rad ≈ 1.6 圈
        for amp in [5.0, 10.0, 20.0]:
            phase_rad = ph.generate_zernike_polynomial(
                coefficients={(2, 0): amp}, radius=radius, n_max=6
            )
            mask = self._get_mask(resolution, radius)
            phase_recovered, _ = self._round_trip_gray(phase_rad)
            expected = np.mod(phase_rad[mask], 2 * np.pi)
            np.testing.assert_allclose(
                phase_recovered[mask], expected, atol=self.QUANTIZATION_TOL
            )
            # 全在 [0, 2π) 内
            assert np.all(phase_recovered[mask] >= 0)
            assert np.all(phase_recovered[mask] < 2 * np.pi)

    def test_generate_zernike_single_mode_grayscale(self):
        """generate_zernike 单模式灰度化数学正确性."""
        resolution = (200, 200)
        radius = 80.0
        ph = PatternHelper(resolution, bits=10)
        phase_rad = ph.generate_zernike(2, 0, amplitude=1.0, radius=radius)
        mask = self._get_mask(resolution, radius)

        phase_recovered, gray = self._round_trip_gray(phase_rad)

        expected = np.mod(phase_rad[mask], 2 * np.pi)
        np.testing.assert_allclose(
            phase_recovered[mask], expected, atol=self.QUANTIZATION_TOL
        )
        assert np.all(gray[mask] >= 0)
        assert np.all(gray[mask] <= 1023)

    def test_slm_grayscale_exact_values(self):
        """验证特定相位值的灰度转换精确值."""
        # 0 rad → 0
        assert (
            phase_to_slm_grayscale(np.array([[0.0]]), max_grayscale=1023, slm=None)[
                0, 0
            ]
            == 0
        )
        # π rad → 511 (1023/2 = 511.5, truncated)
        assert (
            phase_to_slm_grayscale(np.array([[np.pi]]), max_grayscale=1023, slm=None)[
                0, 0
            ]
            == 511
        )
        # π/2 rad → 255 (1023/4 = 255.75, truncated)
        assert (
            phase_to_slm_grayscale(
                np.array([[np.pi / 2]]), max_grayscale=1023, slm=None
            )[0, 0]
            == 255
        )
        # 3π/2 rad → 767 (1023*3/4 = 767.25, truncated)
        assert (
            phase_to_slm_grayscale(
                np.array([[3 * np.pi / 2]]), max_grayscale=1023, slm=None
            )[0, 0]
            == 767
        )
        # 2π rad → 0 (wrapped)
        assert (
            phase_to_slm_grayscale(
                np.array([[2 * np.pi]]), max_grayscale=1023, slm=None
            )[0, 0]
            == 0
        )
        # -π/2 rad → 767 (wrapped to 3π/2)
        assert (
            phase_to_slm_grayscale(
                np.array([[-np.pi / 2]]), max_grayscale=1023, slm=None
            )[0, 0]
            == 767
        )
