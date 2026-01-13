"""
Zernike多项式测试

测试统一的Zernike类和相关功能
"""

import numpy as np
from ao_shaping.utils.zernike import Zernike, zernike_radial_py as zernike_radial, zernike_polynomial, normalize_zernike


class TestZernikeClass:
    """测试Zernike类"""

    def test_zernike_initialization(self):
        """测试Zernike类初始化"""
        z = Zernike(n_max=5, N=64)
        assert z.n_max == 5
        assert z.N == 64
        assert z.num_modes > 0

    def test_zernike_mode_generation(self):
        """测试Zernike模式生成"""
        z = Zernike(n_max=3, N=32)

        # 测试获取模式
        mode1 = z.get_mode(1)  # Piston
        assert mode1.shape == (32, 32)

        # 测试根据n,m获取模式
        mode_tilt_x = z.get_mode_by_nm(1, -1)
        assert mode_tilt_x.shape == (32, 32)

    def test_zernike_reconstruction(self):
        """测试波前重建"""
        z = Zernike(n_max=5, N=32)

        # 创建测试系数
        coeffs = np.random.randn(z.num_modes) * 0.1

        # 重建波前
        wavefront = z.reconstruct(coeffs)
        assert wavefront.shape == (32, 32)

    def test_zernike_fit(self):
        """测试Zernike拟合"""
        z = Zernike(n_max=5, N=32)

        # 创建测试波前
        test_coeffs = np.array([0.1, 0.05, 0.02, 0.01, 0.005])
        test_wavefront = z.reconstruct(test_coeffs)

        # 拟合
        fitted_coeffs = z.fit(test_wavefront)

        # 检查拟合精度
        assert len(fitted_coeffs) >= len(test_coeffs)
        np.testing.assert_allclose(fitted_coeffs[:len(test_coeffs)], test_coeffs, rtol=1e-3)

    def test_piston_tilt_basis(self):
        """测试piston-tilt基函数"""
        z = Zernike(n_max=3, N=32)
        basis = z.piston_tilt_basis()

        assert basis.shape == (3, 32, 32)

        # 检查所有基函数都有合理的范数
        for i in range(3):
            norm = np.sqrt(np.sum(basis[i]**2))
            assert norm > 0.1, f"基函数 {i} 的范数 {norm} 太小"

        # 检查正交性（x-tilt 和 y-tilt 应该正交）
        dot_xt_yt = np.sum(basis[1] * basis[2])  # x-tilt 和 y-tilt
        assert abs(dot_xt_yt) < 1e-6, f"x-tilt 和 y-tilt 的点积为 {dot_xt_yt}"

        # 检查piston与tilt的正交性（近似）
        dot_p_xt = np.sum(basis[0] * basis[1])  # piston 和 x-tilt
        dot_p_yt = np.sum(basis[0] * basis[2])  # piston 和 y-tilt
        # piston与tilt不一定完全正交，但应该接近0
        assert abs(dot_p_xt) < 0.1, f"piston 和 x-tilt 的点积为 {dot_p_xt}"
        assert abs(dot_p_yt) < 0.1, f"piston 和 y-tilt 的点积为 {dot_p_yt}"

    def test_zernike_rms(self):
        """测试RMS计算"""
        z = Zernike(n_max=5, N=32)
        coeffs = np.array([0.1, 0.05, 0.02])

        rms = z.compute_rms(coeffs)
        expected_rms = np.sqrt(np.sum(coeffs**2))
        assert abs(rms - expected_rms) < 1e-10

    def test_zernike_variance(self):
        """测试方差计算"""
        z = Zernike(n_max=5, N=32)

        variance = z.compute_variance(n=2, r0=0.1, D=1.0)
        assert variance > 0

    def test_zernike_name_mapping(self):
        """测试模式名称映射"""
        assert Zernike.get_name(0, 0) == "Piston"
        assert Zernike.get_name(1, -1) == "Tilt X"
        assert Zernike.get_name(1, 1) == "Tilt Y"
        assert Zernike.get_name(2, 0) == "Defocus"
        assert Zernike.get_name(999, 999) == "Z(999,999)"


class TestZernikeFunctions:
    """测试Zernike相关函数"""

    def test_zernike_radial(self):
        """测试径向多项式"""
        rho = np.linspace(0, 1, 10)

        # 测试基本情况
        R = zernike_radial(0, 0, rho)
        np.testing.assert_allclose(R, np.ones_like(rho))

        # 测试n=2, m=0 (defocus)
        R = zernike_radial(2, 0, rho)
        expected = 2 * rho**2 - 1
        np.testing.assert_allclose(R, expected)

    def test_zernike_polynomial(self):
        """测试Zernike多项式"""
        rho = np.array([0.5, 0.8])
        theta = np.array([0.0, np.pi/4])

        # 测试piston
        Z = zernike_polynomial(0, 0, rho, theta)
        np.testing.assert_allclose(Z, np.ones_like(rho))

        # 测试tilt x
        Z = zernike_polynomial(1, -1, rho, theta)
        expected = rho * np.sin(theta)
        np.testing.assert_allclose(Z, expected)

        # 测试tilt y
        Z = zernike_polynomial(1, 1, rho, theta)
        expected = rho * np.cos(theta)
        np.testing.assert_allclose(Z, expected)

    def test_normalize_zernike(self):
        """测试归一化因子"""
        assert normalize_zernike(0, 0) == np.sqrt(1)  # piston
        assert normalize_zernike(1, 1) == np.sqrt(4)  # tilt: sqrt(2*(1+1)) = sqrt(4) = 2
        assert normalize_zernike(2, 0) == np.sqrt(3)  # defocus


class TestZernikeIntegration:
    """测试Zernike与其他组件的集成"""

    def test_zernike_with_wavefront_reconstruction(self):
        """测试与波前重建的集成"""
        from ao_shaping.algorithm.wavefront import zernike_piston_tilt

        # 测试zernike_piston_tilt函数
        N = 32
        basis = zernike_piston_tilt(N)
        assert basis.shape == (3, (N+1)**2)

        # 检查正交性
        for i in range(3):
            for j in range(3):
                dot_product = np.dot(basis[i], basis[j])
                if i == j:
                    assert abs(dot_product - 1.0) < 1e-6
                else:
                    assert abs(dot_product) < 1e-6

    def test_zernike_with_centroid_calculator(self):
        """测试与质心计算器的集成"""
        from ao_shaping.utils.wavefront_calc import ZernikeCentroidCalculator

        # 创建计算器
        calculator = ZernikeCentroidCalculator(n_max=5)

        # 创建测试系数
        zernike_coef = np.array([0.1, 0.05, 0.02])

        # 计算质心
        (cx, cy), wavefront = calculator.get_centroid(zernike_coef)

        assert isinstance(cx, (int, float))
        assert isinstance(cy, (int, float))
        assert wavefront.shape == (360, 360)