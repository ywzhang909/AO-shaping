"""多形状目标的接线测试: 掩膜、GS 目标、CLI 参数。

三条不变量 (违反任一条都会**静默**产出看似合理的错结果):

1. **方斑逐位不变。** ``--target-shape`` 的默认值必须让旧运行一个字节都不变。
   这里用 ``square_mask`` 的逐位对拍 + 一个改动前取下的 GS 相位 sha1 金标钉住。
2. **掩膜与 GS 目标必须是同一个形状。** 形状同时喂两处: GS 的目标和 PIB/CV 的
   评分掩膜。若只有一处切换, 优化器会去追一个它并不在评分的形状 (例如用方框给
   环形打分, 框内大片空白被算进 CV 与 PIB 的分母)。
3. **``annular`` ↔ ``annulus`` 只能有一处映射。** CLI 拼写与算法层拼写不同,
   两张表各写一份就会漂移。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from ao_shaping.algorithm.signal_processing.square_shaping import (
    SquareShapingConfig,
    composite_from_pib_cv,
    gs_shape,
    make_square_target,
    power_in_bucket,
    uniformity_cv,
)
from ao_shaping.drivers.sim.slm_shaping_bench import ShapingBenchConfig
from ao_shaping.optimizer.wfless.slm_gs_refine import (
    _GAUSSIAN_SUPPORT_LEVEL,
    _GS_NATIVE_TARGET_KINDS,
    SlmGsRefineConfig,
    _gs_target_config,
    _rasterise_target,
    _target_mask,
    target_geometry,
)
from ao_shaping.runners.runner_common import (
    SlmGsRefineParams,
    build_gs_refine_config,
)
from ao_shaping.utils.image.target.metrics import TARGET_SHAPE_CHOICES
from ao_shaping.utils.image.targets import square_mask

#: ROI 窗口尺寸: 与 CLI 默认的 ``--cam-size 300`` 同量级, 且容得下 2:1 矩形。
FRAME = (256, 256)
CENTRE = (128, 128)
SIDE = 100

#: Numerical golden for the default square GS phase / target. See
#: :meth:`TestSquareIsUnchanged.test_default_gs_phase_matches_the_pinned_golden`
#: for why this is a tolerance comparison, not the sha1 pin it replaced.
GOLDEN_PATH = Path(__file__).with_name("gs_refine_target_shapes_golden.npz")
GS_GOLDEN_RTOL = 1e-4
GS_GOLDEN_ATOL = 1e-6


def _load_gs_golden() -> dict[str, np.ndarray]:
    """Load (and, under ``AO_GS_REFINE_GOLDEN_UPDATE=1``, regenerate) the golden."""
    if os.environ.get("AO_GS_REFINE_GOLDEN_UPDATE") == "1":
        cfg = ShapingBenchConfig(
            n_grid=32,
            aperture_size=32 * 8e-6,
            wavelength=1064e-9,
            focal_length=0.1224,
            far_field_padding=4,
            target_side_px=12,
            seed=0,
        )
        np.savez_compressed(
            GOLDEN_PATH,
            phase=np.ascontiguousarray(gs_shape(cfg, n_iters=20, seed=0).phase),
            target=np.ascontiguousarray(make_square_target(cfg)),
        )
    return {k: v for k, v in np.load(GOLDEN_PATH).items()}


def _mask(shape: str, *, side: int = SIDE, frame: tuple[int, int] = FRAME, **kw):
    config = SlmGsRefineConfig(target_shape=shape, target_side=side, **kw)
    return _target_mask(np.zeros(frame), CENTRE[0], CENTRE[1], side, config)


# ---------------------------------------------------------------------------
# 1. 方斑逐位不变
# ---------------------------------------------------------------------------


class TestSquareIsUnchanged:
    @pytest.mark.parametrize(
        "frame,cy,cx,side",
        [
            ((100, 100), 50, 50, 20),
            ((64, 64), 32, 32, 7),  # 奇数边长 (历史 round() 口径)
            ((50, 50), 0, 0, 20),  # 贴边裁剪
            ((120, 160), 70, 90, 33),  # 非方形画幅 + 偏中心
        ],
    )
    def test_square_mask_matches_the_legacy_helper_bit_for_bit(
        self, frame, cy, cx, side
    ):
        """形状掩膜走工厂, 但方斑必须逐位等于历史 ``square_mask``。

        两者差一个像素是**结构性的**: ``square_mask`` 用 ``round(cy ± side/2)``
        切片 (端点不含), 工厂用 ``|x-cx| <= half`` (端点含), 整数 half 时多一列。
        方斑是既有运行的口径, 所以方斑这一支显式委托, 由本测试锁住。
        """
        config = SlmGsRefineConfig(target_shape="square", target_side=side)
        got = _target_mask(np.zeros(frame), cy, cx, side, config)
        assert np.array_equal(got, square_mask(frame, cy, cx, side))

    def test_default_gs_phase_matches_the_pinned_golden(self):
        """默认方斑的 GS 相位必须逐位等于钉住的基线 (数值 golden, 标定容差)。

        GS 是确定性交替投影, 默认参数一处改动就会改变结果 —— 这正是「默认行为
        不变」的可执行定义. 但相位是 float64 远场, 逐位 (sha1) 比较对 numpy/BLAS
        构建差异极其敏感: 同一配置在作者机器上是 ``ebd0b18e…``, 在本仓库的其它
        机器上是 ``c94410b9…`` (后者也是 6a5e1067 起各提交的实际值), 差 1 个 bit
        哈希即全变 —— 这是 #167 differentiable_shaping golden 同类的「机器相关
        golden」.

        故改为数值容差比较: 形状 / dtype 仍精确相等, 逐元素用标定容差. 容差按
        本 golden 要抓的变异标定: ``n_iters`` 20→19 让相位移动 ~3.3e-3 相对, 而
        跨构建漂移是 float64 FFT 的 ~1e-6 量级; 因此 rtol=1e-4 / atol=1e-6 位于
        漂移之上 ~100 倍、变异之下 ~33 倍.

        golden 数组由 ``AO_GS_REFINE_GOLDEN_UPDATE=1`` 重生 (``_load_gs_golden``).
        """
        cfg = ShapingBenchConfig(
            n_grid=32,
            aperture_size=32 * 8e-6,
            wavelength=1064e-9,
            focal_length=0.1224,
            far_field_padding=4,
            target_side_px=12,
            seed=0,
        )
        phase = np.ascontiguousarray(gs_shape(cfg, n_iters=20, seed=0).phase)
        target = np.ascontiguousarray(make_square_target(cfg))
        golden = _load_gs_golden()
        # shape + dtype are exact invariants; a real geometry/default change moves them.
        assert phase.shape == golden["phase"].shape and phase.dtype == golden["phase"].dtype
        assert (
            target.shape == golden["target"].shape
            and target.dtype == golden["target"].dtype
        )
        # cross-build drift is ~1e-6; a real default change moves >= 3.3e-3.
        assert np.allclose(phase, golden["phase"], rtol=GS_GOLDEN_RTOL, atol=GS_GOLDEN_ATOL)
        # target is build-independent geometry -> exact.
        assert np.array_equal(target, golden["target"])

    @pytest.mark.parametrize("kind", ["square", "circle", "gaussian", "annulus"])
    def test_native_targets_keep_their_historical_geometry(self, kind):
        """三个新字段留 ``None`` 时, 内建目标必须等于历史公式。"""
        cfg = SquareShapingConfig(
            n_grid=16, far_field_padding=2, target_side_px=12, target_kind=kind
        )
        n = cfg.far_field_size
        y, x = np.ogrid[:n, :n]
        cx = cy = n // 2
        side = 12
        expected = np.zeros((n, n))
        if kind == "square":
            expected[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2] = 1.0
        elif kind == "circle":
            expected[(x - cx) ** 2 + (y - cy) ** 2 <= (side / 2) ** 2] = 1.0
        elif kind == "gaussian":
            s = side / 4
            expected = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2))
        else:  # annulus
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            expected[(r >= side / 2 - side / 4) & (r <= side / 2 + side / 4)] = 1.0
        expected = expected / expected.sum()
        assert np.array_equal(make_square_target(cfg), expected)


# ---------------------------------------------------------------------------
# 2. 八个形状的掩膜
# ---------------------------------------------------------------------------


class TestShapeMasks:
    @pytest.mark.parametrize("shape", TARGET_SHAPE_CHOICES)
    def test_mask_is_non_empty_sized_and_centred(self, shape):
        mask = _mask(shape)
        assert mask.dtype == bool
        assert mask.shape == FRAME
        assert mask.any(), f"{shape} mask is empty"
        frac = float(mask.mean())
        assert 1e-3 < frac < 0.6, f"{shape} covers {frac:.4%} of the frame"
        ys, xs = np.nonzero(mask)
        assert abs(ys.mean() - CENTRE[0]) <= 1.0, f"{shape} row centre off"
        assert abs(xs.mean() - CENTRE[1]) <= 1.0, f"{shape} col centre off"

    @pytest.mark.parametrize("shape", ["circle", "rectangle", "annular", "pentagon"])
    def test_square_box_and_shape_differ_where_the_shape_is_not_a_box(self, shape):
        """方框口径与形状口径**不是**同一个掩膜 (方斑除外)。"""
        assert not np.array_equal(_mask(shape), square_mask(FRAME, *CENTRE, SIDE))

    def test_gaussian_support_is_the_one_over_e_squared_contour(self):
        """高斯目标永远不为零, 所以支撑域取 1/e² 等强线 (r = 2 sigma)。"""
        config = SlmGsRefineConfig(target_shape="gaussian", target_side=SIDE)
        geom = target_geometry(config, SIDE)
        pattern = _rasterise_target("gaussian", FRAME, CENTRE[1], CENTRE[0], geom)
        assert pattern.max() > 0.0
        # 严格大于: 恰好落在等强线上的像素 (float32 舍入后有 20 个) 属于边界外,
        # 与 ``_target_mask`` 同一判据。
        expected = pattern > _GAUSSIAN_SUPPORT_LEVEL
        assert np.array_equal(_mask("gaussian"), expected)
        # 面积应接近 π(2σ)² / 画幅 (2σ = 50px -> 7854 / 65536 = 12.0%)
        assert float(_mask("gaussian").mean()) == pytest.approx(0.12, abs=0.01)

    def test_grid_and_cross_are_rasterised_inside_their_bounding_size(self):
        """``--target-side`` 是 grid/cross 的**包围尺寸**, 不是整幅画面。"""
        for shape in ("grid", "cross"):
            mask = _mask(shape)
            ys, xs = np.nonzero(mask)
            assert ys.max() - ys.min() <= SIDE, f"{shape} exceeds its bounding box"
            assert xs.max() - xs.min() <= SIDE, f"{shape} exceeds its bounding box"

    def test_degenerate_mask_names_the_shape_and_both_frames(self):
        config = SlmGsRefineConfig(target_shape="cross", target_side=2)
        with pytest.raises(ValueError) as excinfo:
            _target_mask(np.zeros(FRAME), CENTRE[0], CENTRE[1], 2, config)
        message = str(excinfo.value)
        assert "cross" in message
        assert "WINDOW-LOCAL" in message
        assert "FULL-FRAME" in message
        assert f"{FRAME[0]}x{FRAME[1]}" in message


# ---------------------------------------------------------------------------
# 3. annular <-> annulus
# ---------------------------------------------------------------------------


class TestAnnularMapping:
    def test_cli_spelling_maps_onto_the_algorithm_spelling(self):
        assert _GS_NATIVE_TARGET_KINDS["annular"] == "annulus"
        assert _GS_NATIVE_TARGET_KINDS["square"] == "square"
        assert _GS_NATIVE_TARGET_KINDS["circle"] == "circle"
        assert _GS_NATIVE_TARGET_KINDS["gaussian"] == "gaussian"
        # 只有这四个能走内建 kind; 其余必须走注入
        assert set(_GS_NATIVE_TARGET_KINDS) == {"square", "circle", "gaussian", "annular"}

    def test_annular_gs_target_uses_the_same_ring_as_the_mask(self):
        """内建 ``annulus`` 的默认环是 side/4..3side/4 (比例 1:3), 与
        ``--target-side`` 的「外径 = side, 内径 = ratio×外径」不是同一个环;
        所以两个半径必须显式传给 GS, 否则目标与掩膜落在两个环上。"""
        config = SlmGsRefineConfig(target_shape="annular", target_side=SIDE)
        probe = ShapingBenchConfig(n_grid=32, far_field_padding=4, target_side_px=40)
        aimed = _gs_target_config(config, probe, bench_side=40)
        assert aimed.target_kind == "annulus"
        geom = target_geometry(config, 40)
        assert aimed.target_inner_radius_px == pytest.approx(geom.inner_radius)
        assert aimed.target_outer_radius_px == pytest.approx(geom.outer_radius)
        assert geom.outer_radius == pytest.approx(20.0)
        assert geom.inner_radius == pytest.approx(0.6 * 20.0)

        # 目标支撑域的面积必须是一个 12..20 px 的环, 而不是内建默认的
        # 10..30 px 环 (那会大 2.9 倍)。
        support = int((make_square_target(aimed) > 0).sum())
        expected = np.pi * (20.0**2 - 12.0**2)
        assert support == pytest.approx(expected, rel=0.05)

    def test_gaussian_gs_target_uses_the_configured_sigma(self):
        config = SlmGsRefineConfig(
            target_shape="gaussian", target_side=SIDE, target_sigma_ratio=0.1
        )
        probe = ShapingBenchConfig(n_grid=16, far_field_padding=2, target_side_px=40)
        aimed = _gs_target_config(config, probe, bench_side=40)
        assert aimed.target_kind == "gaussian"
        assert aimed.target_gaussian_sigma_px == pytest.approx(4.0)

    def test_square_returns_the_probe_untouched(self):
        config = SlmGsRefineConfig(target_shape="square", target_side=SIDE)
        probe = ShapingBenchConfig(n_grid=16, far_field_padding=2, target_side_px=40)
        assert _gs_target_config(config, probe, bench_side=40) is probe


# ---------------------------------------------------------------------------
# 4. CV 只在掩膜内算
# ---------------------------------------------------------------------------


class TestCvOnlyInsideTheMask:
    @pytest.mark.parametrize("shape", ["cross", "grid"])
    def test_uniform_inside_the_shape_scores_zero_cv(self, shape):
        """掩膜内均匀、框内大片空白的合成帧: 形状口径 CV≈0, 方框口径明显偏高。

        这就是方框切片对 cross/grid 的具体错误 —— 空像素被算进 CV 的分母, 于是
        一个**已经完美**的十字会被报成不均匀, 优化器去追框里那些填不满的角落。
        """
        mask = _mask(shape)
        frame = mask.astype(np.float64)
        assert uniformity_cv(frame, mask) == pytest.approx(0.0, abs=1e-12)
        box = square_mask(FRAME, *CENTRE, SIDE)
        cv_box = uniformity_cv(frame, box)
        assert cv_box > 0.5, f"{shape}: box CV {cv_box:.3f} should be far above zero"

        pib_mask = power_in_bucket(frame, mask)
        assert pib_mask == pytest.approx(1.0)
        score_shape = composite_from_pib_cv(pib_mask, uniformity_cv(frame, mask))
        score_box = composite_from_pib_cv(power_in_bucket(frame, box), cv_box)
        assert score_shape > score_box


# ---------------------------------------------------------------------------
# 5. GS 目标注入
# ---------------------------------------------------------------------------


class TestGsTargetInjection:
    @staticmethod
    def _bench(n_grid: int = 64, padding: int = 4, side: int = 50) -> ShapingBenchConfig:
        return ShapingBenchConfig(
            n_grid=n_grid,
            aperture_size=n_grid * 8e-6,
            wavelength=1064e-9,
            focal_length=0.1224,
            far_field_padding=padding,
            target_side_px=side,
            seed=0,
        )

    @staticmethod
    def _pib_cv(phase: np.ndarray | None, cfg: ShapingBenchConfig, mask: np.ndarray):
        from ao_shaping.algorithm.signal_processing.square_shaping import (
            _fraunhofer_intensity_from_field,
        )

        pupil = cfg.make_gaussian_pupil().astype(np.complex128)
        p = np.zeros((cfg.n_grid, cfg.n_grid)) if phase is None else np.asarray(phase)
        intensity = _fraunhofer_intensity_from_field(
            pupil * np.exp(1j * p), cfg.far_field_size
        )
        intensity = intensity / intensity.sum()
        return power_in_bucket(intensity, mask), uniformity_cv(intensity, mask)

    @pytest.mark.parametrize("shape", ["circle", "pentagon"])
    def test_injected_target_beats_the_flat_phase_inside_its_own_mask(self, shape):
        """注入目标的 GS 必须真的把能量摊进该形状。

        ⚠️ 判据是**综合评分与 CV**, 不是 PIB: 平场相位是一个落在 0 级的小点,
        它本来就在圆/五边形掩膜内部, 所以 ``PIB_flat = 1.0`` 已经是上限, 不可能
        再被超过。能区分「小点」与「铺满形状」的是 CV 与综合评分。
        """
        cfg = self._bench()
        mask = _target_mask(
            np.zeros((cfg.far_field_size,) * 2),
            cfg.far_field_size // 2,
            cfg.far_field_size // 2,
            50,
            SlmGsRefineConfig(target_shape=shape, target_side=50),
        )
        phase = gs_shape(cfg, n_iters=150, seed=0).phase
        pib_gs, cv_gs = self._pib_cv(phase, cfg, mask)
        pib_flat, cv_flat = self._pib_cv(None, cfg, mask)

        assert cv_gs < 0.5 * cv_flat, f"{shape}: CV {cv_gs:.3f} vs flat {cv_flat:.3f}"
        # 0.05 而不是 0.1: 这个网格 (64 格 = 10 个 Airy 半径跨过目标) 只够证明
        # 「能量被摊开了」, 不足以证明收敛到工程可用值 —— 那是 128 格离线检查
        # (logs/check_gs_target_shapes.py) 与硬件实测的事。
        assert composite_from_pib_cv(pib_gs, cv_gs) > (
            composite_from_pib_cv(pib_flat, cv_flat) + 0.05
        )

    def test_injected_amplitude_is_the_square_root_of_the_rasterised_target(self):
        cfg = self._bench()
        config = SlmGsRefineConfig(target_shape="pentagon", target_side=50)
        aimed = _gs_target_config(config, cfg, bench_side=50)
        assert aimed.target_kind == "square"  # untouched default kind
        assert aimed.target_amplitude is not None
        geom = target_geometry(config, 50)
        expected = np.sqrt(
            _rasterise_target(
                "pentagon",
                (cfg.far_field_size, cfg.far_field_size),
                cfg.far_field_size // 2,
                cfg.far_field_size // 2,
                geom,
            ).astype(np.float64)
        )
        assert np.allclose(aimed.target_amplitude, expected)

    def test_a_bad_injected_amplitude_is_rejected(self):
        with pytest.raises(ValueError, match="target_amplitude"):
            SquareShapingConfig(target_amplitude=np.zeros((4, 4)))
        with pytest.raises(ValueError, match="2-D"):
            SquareShapingConfig(target_amplitude=np.ones(4))


# ---------------------------------------------------------------------------
# 6. CLI 层
# ---------------------------------------------------------------------------


class TestCliParams:
    @pytest.mark.parametrize(
        ("shape", "knob", "value"),
        [
            ("rectangle", "target_aspect_ratio", 3.0),
            ("annular", "target_inner_ratio", 0.4),
            ("gaussian", "target_sigma_ratio", 0.3),
            ("grid", "target_line_width", 0.04),
            ("grid", "target_grid_nx", 3),
            ("grid", "target_grid_ny", 7),
            ("cross", "target_thickness", 0.09),
        ],
    )
    def test_shape_knob_reaches_the_config_for_the_shape_that_reads_it(
        self, shape: str, knob: str, value: float
    ):
        """Each knob is wired field by field — for the shape that reads it.

        It used to be checked on ``circle`` with all seven knobs at once, which
        is exactly the combination ``build_gs_refine_config`` now refuses: a knob
        the shape does not read must not silently land in the config.
        """
        params = SlmGsRefineParams(target_shape=shape, target_side=60, **{knob: value})
        config = build_gs_refine_config(params, debug=False)
        assert config.target_shape == shape
        assert config.target_side == 60
        assert getattr(config, knob) == value

    def test_knobs_the_shape_does_not_read_are_refused(self):
        params = SlmGsRefineParams(
            target_shape="circle",
            target_side=60,
            target_aspect_ratio=3.0,
            target_inner_ratio=0.4,
            target_sigma_ratio=0.3,
            target_line_width=0.04,
            target_grid_nx=3,
            target_grid_ny=7,
            target_thickness=0.09,
        )
        with pytest.raises(ValueError, match="circle"):
            build_gs_refine_config(params, debug=False)

    def test_defaults_are_square(self):
        config = build_gs_refine_config(SlmGsRefineParams(), debug=False)
        assert config.target_shape == "square"
        assert config.target_aspect_ratio == 2.0
        assert config.target_inner_ratio == 0.6
        assert config.target_sigma_ratio == 0.25
        assert config.target_line_width == 0.02
        assert (config.target_grid_nx, config.target_grid_ny) == (5, 5)
        assert config.target_thickness == 0.05

    def test_unknown_shape_is_rejected_by_click(self):
        from ao_shaping.runners.slm.gs_refine_runner import run

        result = CliRunner().invoke(run, ["--target-shape", "banana"])
        assert result.exit_code == 2, result.output
        assert "banana" in result.output
        assert "Invalid value" in result.output

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"target_shape": "banana"},
            {"target_aspect_ratio": 0.5},
            {"target_inner_ratio": 1.0},
            {"target_sigma_ratio": 0.0},
            {"target_line_width": 0.0},
            {"target_thickness": 0.0},
            {"target_grid_nx": 0},
            {"target_grid_ny": 0},
        ],
    )
    def test_config_rejects_out_of_range_shape_parameters(self, kwargs):
        with pytest.raises(ValueError):
            SlmGsRefineConfig(**kwargs)
