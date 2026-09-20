from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.wavefront.zernike_calc import (
    ZERNIKE_NAMES,
    ZernikeGenerator,
    calc_n_zernike_terms,
    fit_zernike,
    get_zernike_name,
    noll_to_nm,
    zernike_modes,
    zernike_radial,
)


class TestZernikeGeneratorSquare:
    """Tests for square mode in ZernikeGenerator."""

    def test_square_wider_than_tall(self):
        """Test square mode with width > height (landscape)."""
        width, height = 1920, 1080
        gen = ZernikeGenerator((width, height), square=True)
        gen.set_bits(10)

        img = gen.generate(2, 0, amplitude=1.0)

        # Output must match requested resolution
        assert img.shape == (height, width), (
            f"Expected ({height}, {width}), got {img.shape}"
        )

    def test_square_taller_than_wide(self):
        """Test square mode with height > width (portrait)."""
        width, height = 1080, 1920
        gen = ZernikeGenerator((width, height), square=True)
        gen.set_bits(10)

        img = gen.generate(2, 0, amplitude=1.0)

        # Output must match requested resolution
        assert img.shape == (height, width), (
            f"Expected ({height}, {width}), got {img.shape}"
        )

    def test_square_already_square(self):
        """Test square mode when already square (no cropping needed)."""
        width, height = 1000, 1000
        gen = ZernikeGenerator((width, height), square=True)
        gen.set_bits(10)

        img = gen.generate(2, 0, amplitude=1.0)

        # Output must match requested resolution
        assert img.shape == (height, width), (
            f"Expected ({height}, {width}), got {img.shape}"
        )

    def test_square_false_preserves_aspect(self):
        """Test square=False preserves original non-square shape."""
        width, height = 1920, 1080
        gen = ZernikeGenerator((width, height), square=False)
        gen.set_bits(10)

        img = gen.generate(2, 0, amplitude=1.0)

        # Output unchanged when square=False
        assert img.shape == (height, width), (
            f"Expected ({height}, {width}), got {img.shape}"
        )

    def test_square_generate_noll(self):
        """Test square mode with generate_noll."""
        width, height = 1920, 1080
        gen = ZernikeGenerator((width, height), square=True)
        gen.set_bits(10)

        coeffs = np.ones(10)
        img = gen.generate_noll(coeffs)

        # Output must match requested resolution
        assert img.shape == (height, width), (
            f"Expected ({height}, {width}), got {img.shape}"
        )

    def test_square_generate_polynomial(self):
        """Test square mode with generate_polynomial."""
        width, height = 1920, 1080
        gen = ZernikeGenerator((width, height), square=True)
        gen.set_bits(10)

        coeffs = {(0, 0): 1.0, (1, -1): 0.3, (2, 0): 0.2}
        img = gen.generate_polynomial(coeffs)

        assert img.shape == (height, width), (
            f"Expected ({height}, {width}), got {img.shape}"
        )

    def test_square_empty_coefficients(self):
        """Test square mode with empty coefficients."""
        width, height = 1920, 1080
        gen = ZernikeGenerator((width, height), square=True)
        gen.set_bits(10)

        coeffs = {}
        img = gen.generate_polynomial(coeffs)

        assert img.shape == (height, width), (
            f"Expected ({height}, {width}), got {img.shape}"
        )


class TestZernikeGeneratorBasic:
    """Basic tests for ZernikeGenerator."""

    def test_init_default_radius(self):
        gen = ZernikeGenerator((800, 600))
        assert gen.resolution == (800, 600)
        assert gen.radius == 300.0

    def test_init_custom_radius(self):
        gen = ZernikeGenerator((800, 600), radius=250.0)
        assert gen.radius == 250.0

    def test_set_bits(self):
        gen = ZernikeGenerator((800, 600))
        gen.set_bits(8)
        assert gen._max_val == 255

    def test_generate_without_set_bits_raises(self):
        gen = ZernikeGenerator((800, 600))
        with pytest.raises(ValueError, match="Call set_bits"):
            gen.generate(2, 0)

    def test_generate_piston(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        gen.set_bits(10)
        img = gen.generate(0, 0, amplitude=1.0)
        assert img.shape == (100, 100)
        assert isinstance(img, np.ndarray)

    def test_generate_tilt(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        gen.set_bits(10)
        img = gen.generate(1, 1, amplitude=0.5)
        assert img.shape == (100, 100)
        assert not np.all(np.isnan(img))

    def test_generate_defocus(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        gen.set_bits(10)
        img = gen.generate(2, 0, amplitude=0.5)
        assert img.shape == (100, 100)

    def test_generate_negative_m(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        gen.set_bits(10)
        img = gen.generate(1, -1, amplitude=0.5)
        assert img.shape == (100, 100)

    def test_generate_polynomial(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        gen.set_bits(10)
        coeffs = {(0, 0): 1.0, (1, -1): 0.3, (2, 0): 0.2}
        img = gen.generate_polynomial(coeffs)
        assert img.shape == (100, 100)

    def test_generate_polynomial_with_zero_coeffs(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        gen.set_bits(10)
        coeffs = {(0, 0): 0.0, (1, 1): 0.0}
        img = gen.generate_polynomial(coeffs)
        assert img.shape == (100, 100)

    def test_mask_property(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        mask = gen.mask
        assert mask.shape == (100, 100)

    def test_R_property(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        R = gen.R
        assert R.shape == (100, 100)
        assert R.max() > 1.0

    def test_Theta_property(self):
        gen = ZernikeGenerator((100, 100), radius=50.0)
        Theta = gen.Theta
        assert Theta.shape == (100, 100)

    def test_radius_controls_aperture_size(self):
        """『孔径半径』必须真正约束圆形孔径 (回归: 之前 radius 被忽略)."""
        gen_small = ZernikeGenerator((100, 100), radius=25.0)
        gen_small.set_bits(10)
        gen_large = ZernikeGenerator((100, 100), radius=50.0)
        gen_large.set_bits(10)
        coeffs = {(0, 0): 1.0, (2, 0): 0.5}

        small = gen_small.generate_polynomial(coeffs)
        large = gen_large.generate_polynomial(coeffs)
        assert small.shape == large.shape == (100, 100)
        # 孔径区域 = 非 NaN 的像素; 半径 50 的孔径应约为半径 25 的 4x (π r²)
        small_aperture = (~np.isnan(small)).sum()
        large_aperture = (~np.isnan(large)).sum()
        assert large_aperture > 3.5 * small_aperture
        # 孔径中心区域应有非平凡相位 (非纯 flat)
        assert not np.allclose(small[50, 50], small[0, 0])

    def test_grid_cache_reused_across_instances(self):
        """同参数重复生成应复用缓存 (回归: 之前每次生成重建 RZern 网格, ~2.7s)."""
        import time

        gen1 = ZernikeGenerator((100, 100), radius=50.0)
        gen1.set_bits(10)
        gen2 = ZernikeGenerator((100, 100), radius=50.0)
        gen2.set_bits(10)
        coeffs = {(0, 0): 1.0, (2, 0): 0.5}

        # 首次 (冷) 与第二次 (缓存命中) 在第二次构造时共享 _build_cached_grid
        a = gen1.generate_polynomial(coeffs)
        b = gen2.generate_polynomial(coeffs)
        # NaN (孔径外) 与数值 (孔径内) 都应逐元素一致
        np.testing.assert_array_equal(a, b)
        # 同一 (分辨率, radius, n_orders) 共享同一个 RZern cart + 网格缓存
        assert gen1._cart is gen2._cart
        assert np.array_equal(gen1.xv, gen2.xv)

    def test_radius_change_rebuilds_cart(self):
        """半径变化 → 完全重建 (新 cart), 不再复用旧 radius 的基底."""
        gen_small = ZernikeGenerator((100, 100), radius=25.0)
        gen_large = ZernikeGenerator((100, 100), radius=50.0)
        # 不同 radius → 不同 cart (单活动槽: 后者替换前者, 旧基底释放)
        assert gen_small._cart is not gen_large._cart
        # 坐标网格必须随半径不同 (归一化单位是 radius 像素)
        assert not np.array_equal(gen_small.xv, gen_large.xv)

    def test_radius_change_releases_old_cart(self):
        """半径变化替换缓存槽后, 旧 cart 必须被释放 (无泄漏, 内存恒定单份)."""
        import gc
        import weakref

        gen = ZernikeGenerator((100, 100), radius=33.0)
        ref_old_cart = weakref.ref(gen._cart)
        ref_old_xv = weakref.ref(gen.xv)
        del gen
        gc.collect()
        # 实例释放后旧 cart 仍由模块级单活动槽持有
        assert ref_old_cart() is not None

        # 换半径 → 单槽替换 → 旧 cart / 网格不再被任何对象引用
        ZernikeGenerator((100, 100), radius=77.0)
        gc.collect()
        assert ref_old_cart() is None
        assert ref_old_xv() is None

    def test_n_orders_change_rebuilds_cart(self):
        """n_orders 变化 → 重新构建 (列数不同), 但同 (radius, n_orders) 仍复用."""
        gen4 = ZernikeGenerator((100, 100), radius=50.0, n_orders=4)
        gen6 = ZernikeGenerator((100, 100), radius=50.0, n_orders=6)
        assert gen4._cart is not gen6._cart
        # 绕回 radius=50, n_orders=6: 与 gen6 是同一活动槽 → 同一 cart
        gen6_again = ZernikeGenerator((100, 100), radius=50.0, n_orders=6)
        assert gen6_again._cart is gen6._cart


class TestFitZernike:
    def test_fit_zernike_default(self):
        # API now returns a NumPy array of coefficients
        wavefront = np.random.rand(100, 100)
        coeffs = fit_zernike(wavefront, n_max=2)
        assert isinstance(coeffs, np.ndarray)
        assert coeffs.size > 0

    def test_fit_zernike_custom_radius(self):
        # Radius keyword is not part of current API; ensure function accepts n_max and returns ndarray
        wavefront = np.random.rand(100, 100)
        coeffs = fit_zernike(wavefront, n_max=2)
        assert isinstance(coeffs, np.ndarray)
        assert coeffs.size > 0

    def test_fit_zernike_single_mode(self):
        # n_max=1 should return 1 + 2 = 3 coefficients
        wavefront = np.random.rand(50, 50)
        coeffs = fit_zernike(wavefront, n_max=1)
        assert isinstance(coeffs, np.ndarray)
        assert coeffs.size == 3

    def test_fit_zernike_contains_expected_modes(self):
        # With n_max=2, the number of Zernike terms is 6; ensure correct length is returned
        wavefront = np.random.rand(60, 60)
        coeffs = fit_zernike(wavefront, n_max=2)
        assert isinstance(coeffs, np.ndarray)
        assert coeffs.size == 6


class TestZernikeRadial:
    def test_zernike_radial_piston(self):
        r = np.linspace(0, 1, 10)
        result = zernike_radial(0, 0, r)
        assert result.shape == r.shape

    def test_zernike_radial_tilt(self):
        r = np.linspace(0, 1, 10)
        result = zernike_radial(1, 1, r)
        assert result.shape == r.shape

    def test_zernike_radial_defocus(self):
        r = np.linspace(0, 1, 10)
        result = zernike_radial(2, 0, r)
        assert result.shape == r.shape


class TestZernikeNames:
    """ZERNIKE_NAMES lookup table (Noll's scheme)."""

    def test_n7_all_modes_present(self):
        """n=7 的 8 个模式 (m=-7..7 step 2) 必须全部命名."""
        for m in (-7, -5, -3, -1, 1, 3, 5, 7):
            assert (7, m) in ZERNIKE_NAMES, f"missing n=7, m={m}"
        assert len([k for k in ZERNIKE_NAMES if k[0] == 7]) == 8

    def test_n7_noll_mapping(self):
        """Noll 29..36 必须映射到 n=7 的 8 个模式, 且都有名称."""
        expected = [(7, -1), (7, 1), (7, -3), (7, 3), (7, -5), (7, 5), (7, -7), (7, 7)]
        actual = [noll_to_nm(j) for j in range(29, 37)]
        assert actual == expected
        for nm in expected:
            assert get_zernike_name(*nm) == ZERNIKE_NAMES[nm]
            assert "n=" not in get_zernike_name(*nm)  # 有命名, 非回退格式

    def test_n7_names_use_x_y_convention(self):
        """m<0 → Y (sin), m>0 → X (cos), 与既有 (3,±3)/(4,±4) 风格一致.

        名称现为 "English / 中文" 双语格式, 故只校验英文段后缀。
        (注意: n=1 的 Tip/Tilt 与 Astig 45°/0° 系列不遵循该 X/Y 规律,
        此处只对 n=7 的 coma/trefoil/foil 系列断言。)
        """

        def english(n: int, m: int) -> str:
            return ZERNIKE_NAMES[(n, m)].split("/")[0].strip()

        assert english(7, -7).endswith("Y")
        assert english(7, 7).endswith("X")
        assert english(7, -1).endswith("Y")
        assert english(7, 1).endswith("X")
        assert english(7, -3).endswith("Y")
        assert english(7, 3).endswith("X")

    def test_table_orders_complete(self):
        """每个已收录的 n 阶必须含 n+1 个模式, 且 m 与 n 同奇偶、|m|<=n.

        结构性不变量 (对未来扩展 n>7 同样成立): 阶数连续无跳阶。
        """
        orders = sorted({n for (n, _) in ZERNIKE_NAMES})
        assert orders == list(range(orders[0], orders[-1] + 1)), f"跳阶: {orders}"
        for n in orders:
            modes = sorted(m for (nn, m) in ZERNIKE_NAMES if nn == n)
            assert len(modes) == n + 1, f"n={n} 模式数应为 {n + 1}, 实为 {modes}"
            assert all((m - n) % 2 == 0 and abs(m) <= n for m in modes), (
                f"n={n}: {modes}"
            )

    def test_unknown_mode_falls_back(self):
        """未收录模式回退为 n=,m= 格式."""
        assert get_zernike_name(99, 1) == "n=99,m=1"


class TestZernikeModes:
    """Tests for :func:`ao_shaping.utils.wavefront.zernike_calc.zernike_modes`.

    This function was extracted from ``ZernikeControl._modes`` in the GUI
    layer (2026-09-18) — it is pure Zernike math (the parity rule
    ``n - |m| == even``) and belongs in ``utils.zernike_calc`` alongside
    :func:`get_zernike_name` and :func:`calc_n_zernike_terms`.
    """

    def test_modes_count_matches_calc_n_zernike_terms(self) -> None:
        from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms

        for n_max in (0, 1, 2, 5, 10):
            assert len(zernike_modes(n_max)) == calc_n_zernike_terms(n_max), (
                f"n_max={n_max}: 模式数应为 (n+1)(n+2)//2 = "
                f"{calc_n_zernike_terms(n_max)}"
            )

    def test_modes_parity_rule(self) -> None:
        """n - |m| must be even for every returned (n, m)."""
        for n_max in (1, 5, 10):
            for n, m in zernike_modes(n_max):
                assert (n - abs(m)) % 2 == 0, f"({n},{m}) 违反奇偶规则"
                assert abs(m) <= n, f"({n},{m}) 超出 |m|<=n"

    def test_modes_contain_canonical_low_order(self) -> None:
        modes = zernike_modes(5)
        assert (0, 0) in modes  # piston
        assert (2, 0) in modes  # defocus
        assert (2, -2) in modes and (2, 2) in modes  # astigmatism
        assert (4, 0) in modes  # spherical
        assert (2, 1) not in modes  # parity violation excluded

    def test_modes_n_max_zero(self) -> None:
        assert zernike_modes(0) == [(0, 0)]

    def test_modes_is_pure_function_no_side_effects(self) -> None:
        a = zernike_modes(3)
        b = zernike_modes(3)
        assert a == b
        assert a is not b  # new list each call, no caching at this layer

    def test_modes_coerces_int(self) -> None:
        """n_max may arrive as a string from a widget; coerce to int."""
        assert zernike_modes("4") == zernike_modes(4)
