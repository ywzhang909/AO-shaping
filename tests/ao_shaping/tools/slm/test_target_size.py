"""``ao_shaping.tools.slm.panel.target_size`` 的单元测试.

纯 numpy + 标准库模块 (无 torch / 驱动 / 设备)。覆盖:

* ``OpticalLimits`` 解析界限 (λ=1064 nm / f=0.125 m / N=1920 / p=8 µm 的三个
  验收数值);
* ``SizeMetrics`` 防退化口径 (效率可 >1、零级盘比值 ∈[0,1]、``valid`` /
  ``mean_core`` / ``fill`` 字段、序列化往返);
* ``is_valid`` (mean_core ≥ k×噪声地板, 2026-10-23 实测 k=3);
* ``degenerate_metrics`` (efficiency 随尺寸单调 / CV 极差 / 零级极差);
* ``select_best_window`` (先按 valid 过滤、退化指标剔除、无有效点 / 全退化
  显式返回 ``None`` 窗口 + 说明);
* ``fit_optimum`` (单峰顶点 / 不呈单峰显式 ``None``);
* ``TargetSizeCalibration`` (save/load 往返、签名不匹配显式拒绝、
  ``recommend(constraint=...)`` 的 ``largest_valid`` / ``best_balance``)。

2026-10-23 实测数据回归: 细扫 40–240 px 中 40/60/80/100 有效、120 起失效,
``recommend`` 默认应给出最大有效尺寸 100 px (而不是旧口径下被空盒点拖到
1555 px 的行为)。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.slm.panel.target_size import (
    DEFAULT_SYSTEM,
    OpticalLimits,
    SizeMetrics,
    TargetSizeCalibration,
    TargetSizeCalibrationError,
    composite_scores,
    degenerate_metrics,
    default_fractions,
    fit_optimum,
    is_valid,
    select_best_window,
)

# --------------------------------------------------------------------------- #
# 测试用默认光路 (λ=1064 nm, f=0.125 m, 1200×1920, p=8 µm, 2.2 µm 相机像元)
# --------------------------------------------------------------------------- #
LAMBDA_M = 1064e-9
FOCAL_M = 0.125
S = OpticalLimits(
    wavelength_m=LAMBDA_M,
    focal_m=FOCAL_M,
    slm_pixels=(1200, 1920),
    pixel_pitch_m=8e-6,
    camera_pixel_m=2.2e-6,
    camera_shape=(1944, 2592),
)


def _point(size, eff, cv, zero=0.1, vs_base=None, mean_core=1.0, fill=0.5, valid=True):
    return SizeMetrics(
        size_px=float(size),
        efficiency=float(eff),
        uniformity=float(cv),
        zero_order_ratio=float(zero),
        efficiency_vs_baseline=float(vs_base if vs_base is not None else eff),
        mean_core=float(mean_core),
        fill=float(fill),
        valid=bool(valid),
    )


# --------------------------------------------------------------------------- #
# OpticalLimits 解析界限
# --------------------------------------------------------------------------- #
class TestOpticalLimits:
    def test_far_field_sample_m(self):
        # λf/(N_x·p) = 1064e-9 × 0.125 / (1920 × 8e-6) ≈ 8.6597e-6 m
        expected = LAMBDA_M * FOCAL_M / (1920 * 8e-6)
        assert S.far_field_sample_m == pytest.approx(expected, rel=1e-12)
        assert S.far_field_sample_m == pytest.approx(8.6597e-6, rel=1e-3)

    def test_slm_fov_m(self):
        # λf/p = 1064e-9 × 0.125 / 8e-6 = 0.016625 m
        assert S.slm_fov_m == pytest.approx(0.016625, abs=1e-12)

    def test_practical_max_px(self):
        # min(FOV/像素, 传感器短边 1944) = 1944 (传感器短边先到达)
        assert S.practical_max_px == pytest.approx(1944.0, abs=1e-9)

    def test_candidate_sizes_default(self):
        sizes = S.candidate_sizes_px()
        assert len(sizes) == 16
        assert sizes[0] == pytest.approx(97.2, abs=1e-6)
        assert sizes[-1] == pytest.approx(1555.2, abs=1e-6)
        # 单调递增
        assert all(b > a for a, b in zip(sizes, sizes[1:]))

    def test_candidate_sizes_custom_fractions(self):
        sizes = S.candidate_sizes_px((0.05, 0.10))
        assert sizes == pytest.approx((97.2, 194.4), abs=1e-6)

    def test_candidate_sizes_rejects_degenerate(self):
        # 去重后不足 2 个
        with pytest.raises(ValueError):
            S.candidate_sizes_px((0.05, 0.05))

    def test_signature_stable_and_sensitive(self):
        assert S.signature() == S.signature()
        other = OpticalLimits(
            wavelength_m=1550e-9,
            focal_m=FOCAL_M,
            slm_pixels=(1200, 1920),
            pixel_pitch_m=8e-6,
            camera_pixel_m=2.2e-6,
            camera_shape=(1944, 2592),
        )
        assert other.signature() != S.signature()
        assert len(S.signature()) == 16

    def test_invalid_parameters_rejected(self):
        with pytest.raises(ValueError):
            OpticalLimits(wavelength_m=-1.0, focal_m=FOCAL_M)
        with pytest.raises(ValueError):
            OpticalLimits(wavelength_m=LAMBDA_M, focal_m=FOCAL_M, slm_pixels=(0, 1920))

    def test_default_system_matches_repo_bench(self):
        assert DEFAULT_SYSTEM == S


# --------------------------------------------------------------------------- #
# SizeMetrics 口径与序列化
# --------------------------------------------------------------------------- #
class TestSizeMetrics:
    def test_efficiency_may_exceed_one(self):
        # 盒比固定参考窗大 → 效率是能量比, 允许 >1 (防退化修订后的新口径)
        p = _point(1000.0, 1.7, 0.9)
        assert p.efficiency == pytest.approx(1.7)

    def test_zero_order_ratio_must_be_fraction(self):
        with pytest.raises(ValueError):
            _point(100.0, 0.5, 0.5, zero=1.5)

    def test_negative_values_rejected(self):
        with pytest.raises(ValueError):
            SizeMetrics(size_px=-1.0, efficiency=0.5, uniformity=0.5, zero_order_ratio=0.1)
        with pytest.raises(ValueError):
            _point(100.0, 0.5, -0.1)

    def test_fill_must_be_fraction(self):
        with pytest.raises(ValueError):
            _point(100.0, 0.5, 0.5, fill=1.4)

    def test_fidelity_range(self):
        with pytest.raises(ValueError):
            SizeMetrics(
                size_px=100.0,
                efficiency=0.5,
                uniformity=0.5,
                zero_order_ratio=0.1,
                fidelity=1.4,
            )

    def test_to_dict_from_dict_roundtrip(self):
        p = _point(100.0, 1.2, 0.8, zero=0.25, vs_base=0.3, mean_core=4.0, fill=0.7)
        p = SizeMetrics(
            size_px=100.0,
            efficiency=1.2,
            uniformity=0.8,
            zero_order_ratio=0.25,
            fidelity=0.93,
            target_type="square",
            efficiency_baseline=0.9,
            efficiency_vs_baseline=0.3,
            zero_abs=55.0,
            mean_core=4.0,
            fill=0.7,
            valid=True,
        )
        d = p.to_dict()
        q = SizeMetrics.from_dict(d)
        assert q == p
        # 旧存档 (没有新字段) 向后兼容: 缺省有效
        legacy = {
            "size_px": 100.0,
            "efficiency": 0.5,
            "uniformity": 0.6,
            "zero_order_ratio": 0.1,
            "fidelity": None,
        }
        old = SizeMetrics.from_dict(legacy)
        assert old.valid is True
        assert old.fill == 0.0


# --------------------------------------------------------------------------- #
# 有效性门 (2026-10-23 实测: k=3, floor=0.377)
# --------------------------------------------------------------------------- #
class TestIsValid:
    def test_measured_fine_scan_thresholds(self):
        floor = 0.377
        # 40 px: mean 6.631 有效; 100 px: mean 1.500 有效; 120 px: mean 1.031 失效
        assert is_valid(6.631, floor) is True
        assert is_valid(1.500, floor) is True
        assert is_valid(1.031, floor) is False

    def test_custom_k(self):
        assert is_valid(0.5, 0.1, k=10.0) is False
        assert is_valid(0.5, 0.1, k=4.0) is True


# --------------------------------------------------------------------------- #
# 退化指标检测
# --------------------------------------------------------------------------- #
class TestDegenerateMetrics:
    def test_monotonic_efficiency_flagged(self):
        # efficiency 随尺寸单调 (旧口径 0.214→0.700 的形态) ⇒ Spearman ≈ 1
        pts = [_point(100 * (i + 1), 0.2 + 0.05 * i, 0.8) for i in range(6)]
        findings = degenerate_metrics(pts)
        names = [n for n, _ in findings]
        assert "efficiency" in names

    def test_constant_zero_order_flagged(self):
        # 零级占比全程不动 (0.1215) ⇒ 极差 < 1e-3
        pts = [_point(100 * (i + 1), 0.3 + 0.01 * i, 0.8 + 0.05 * i, zero=0.1215) for i in range(5)]
        findings = degenerate_metrics(pts)
        names = [n for n, _ in findings]
        assert "zero_order_ratio" in names

    def test_flat_uniformity_flagged(self):
        # CV 极差 < 0.02
        pts = [_point(100 * (i + 1), 0.3 + 0.01 * i, 0.5 + 0.001 * i, zero=0.1) for i in range(5)]
        findings = degenerate_metrics(pts)
        names = [n for n, _ in findings]
        assert "uniformity" in names

    def test_non_degenerate_points(self):
        # 三项都有内部结构: 无退化
        pts = [
            _point(40.0, 0.5, 0.7, zero=0.12, vs_base=0.05),
            _point(80.0, 0.6, 1.1, zero=0.15, vs_base=0.12),
            _point(120.0, 0.55, 1.5, zero=0.11, vs_base=0.10),
            _point(160.0, 0.5, 1.9, zero=0.14, vs_base=0.07),
        ]
        assert degenerate_metrics(pts) == []

    def test_only_valid_points_checked(self):
        # 无效点不参与判据 (全无效 ⇒ 直接无判据)
        pts = [_point(100 * (i + 1), 0.2 + 0.05 * i, 0.8, valid=False) for i in range(6)]
        assert degenerate_metrics(pts) == []


# --------------------------------------------------------------------------- #
# 峰值窗口
# --------------------------------------------------------------------------- #
class TestSelectBestWindow:
    def test_single_peak_window(self):
        # 分数在 80 px 附近单峰 (efficiency_vs_baseline 内部峰 + CV 单调升)
        pts = [
            _point(40.0, 0.5, 0.7, vs_base=0.05, zero=0.12),
            _point(60.0, 0.55, 0.9, vs_base=0.10, zero=0.13),
            _point(80.0, 0.6, 1.2, vs_base=0.16, zero=0.12),
            _point(120.0, 0.55, 1.6, vs_base=0.10, zero=0.14),
            _point(160.0, 0.5, 2.0, vs_base=0.04, zero=0.13),
        ]
        lo, hi, best, degenerate, message = select_best_window(pts)
        assert degenerate == []
        assert message == ""
        assert best == pytest.approx(80.0)
        assert lo <= best <= hi
        assert hi - lo < 80.0  # 平台不是全区间

    def test_invalid_points_excluded(self):
        # 大尺寸点全部无效 ⇒ 窗口只落在有效区
        pts = [
            _point(40.0, 0.5, 0.7, vs_base=0.15, mean_core=6.6, valid=True),
            _point(60.0, 0.5, 0.9, vs_base=0.14, mean_core=3.9, valid=True),
            _point(200.0, 0.7, 0.4, vs_base=0.30, mean_core=0.02, valid=False),
        ]
        lo, hi, best, _deg, _msg = select_best_window(pts)
        assert best <= 60.0
        assert hi <= 60.0

    def test_no_valid_points_returns_none_window(self):
        pts = [_point(100 * (i + 1), 0.3, 8.0, mean_core=0.01, valid=False) for i in range(4)]
        lo, hi, best, _deg, message = select_best_window(pts)
        assert (lo, hi, best) == (None, None, None)
        assert "没有可填满的目标" in message

    def test_all_degenerate_returns_none_window(self):
        # efficiency 单调 + CV 恒定 ⇒ 两项参与排序的指标全部退化
        pts = [_point(100 * (i + 1), 0.2 + 0.05 * i, 0.5, zero=0.1) for i in range(5)]
        # zero 恒定也会被判退化, 但 zero 不参与排序 —— 排序只剩 efficiency(单调)+CV(恒定)
        lo, hi, best, degenerate, message = select_best_window(pts)
        assert (lo, hi, best) == (None, None, None)
        names = {n for n, _ in degenerate}
        assert "efficiency" in names and "uniformity" in names
        assert "全部退化" in message

    def test_two_valid_points_single_point_window(self):
        pts = [
            _point(40.0, 0.5, 0.7, vs_base=0.1, valid=True),
            _point(200.0, 0.6, 0.8, vs_base=0.1, valid=False),
            _point(80.0, 0.55, 0.9, vs_base=0.12, valid=True),
        ]
        lo, hi, best, _deg, message = select_best_window(pts)
        assert lo == hi == best  # 窗口退化为单点
        assert "单点" in message

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            select_best_window([_point(40.0, 0.5, 0.7), _point(80.0, 0.6, 0.9)])


# --------------------------------------------------------------------------- #
# 单峰拟合
# --------------------------------------------------------------------------- #
class TestFitOptimum:
    def test_synthetic_single_peak(self):
        # s(u) = -(u-u0)²+1 的形态 (efficiency_vs_baseline 单峰, CV 恒定 ⇒ 合成 = 效率项)
        sizes = np.array([40.0, 60.0, 100.0, 160.0, 250.0, 400.0])
        u0 = math.log(100.0)
        pts = []
        for s in sizes:
            u = math.log(s)
            v = max(0.05, -(u - u0) ** 2 * 0.5 + 1.0)
            pts.append(_point(s, v, 0.5, vs_base=v, zero=0.1))
        size_star, sigma, r2 = fit_optimum(pts)
        assert size_star is not None
        assert size_star == pytest.approx(100.0, rel=0.05)
        assert sigma is not None and sigma > 0
        assert r2 > 0.9

    def test_monotonic_returns_none_vertex(self):
        # 效率单调 (退化形态) + CV 单调 ⇒ 合成分数无内部峰 ⇒ 顶点显式 None
        pts = [_point(100 * (i + 1), 0.2 + 0.05 * i, 0.5 + 0.1 * i, vs_base=0.01 + 0.02 * i, zero=0.1) for i in range(6)]
        size_star, sigma, r2 = fit_optimum(pts)
        assert size_star is None
        assert sigma is None
        assert 0.0 <= r2 <= 1.0

    def test_too_few_valid_raises(self):
        pts = [
            _point(40.0, 0.5, 0.7, valid=True),
            _point(80.0, 0.6, 0.9, valid=False),
            _point(120.0, 0.55, 1.1, valid=False),
        ]
        with pytest.raises(ValueError):
            fit_optimum(pts)


# --------------------------------------------------------------------------- #
# 合成分数
# --------------------------------------------------------------------------- #
class TestCompositeScores:
    def test_uses_efficiency_vs_baseline(self):
        a = _point(40.0, 0.9, 0.5, vs_base=0.01)
        b = _point(80.0, 0.5, 0.5, vs_base=0.5)
        # 原始 efficiency a 更高, 但打分用 vs_base ⇒ b 的分数更高
        scores = composite_scores([a, b], 0.0, 1.0)
        assert scores[1] > scores[0]

    def test_uniformity_inverted(self):
        a = _point(40.0, 0.5, 2.0)
        b = _point(80.0, 0.5, 0.2)
        scores = composite_scores([a, b], 1.0, 0.0)
        assert scores[1] > scores[0]

    def test_zero_weights_rejected(self):
        with pytest.raises(ValueError):
            composite_scores([_point(40.0, 0.5, 0.5)], 0.0, 0.0)


# --------------------------------------------------------------------------- #
# 查找表: 序列化 / 签名 / recommend 约束
# --------------------------------------------------------------------------- #
class TestTargetSizeCalibration:
    def _cal(self, points, window=(40.0, 100.0), best_size=None):
        return TargetSizeCalibration(
            signature=S.signature(),
            limits=S,
            points=points,
            window=window,
            windows={"square": window},
            best_size=best_size,
            method="test",
            notes="unit test",
            recorded_utc="2026-10-23T00:00:00+00:00",
        )

    def test_save_load_roundtrip(self, tmp_path: Path):
        pts = [
            _point(40.0, 0.564, 0.735, vs_base=0.064, mean_core=6.631, fill=0.344, valid=True),
            _point(100.0, 0.794, 2.136, vs_base=0.09, mean_core=1.5, fill=0.077, valid=True),
            _point(120.0, 0.802, 2.646, vs_base=0.095, mean_core=1.031, fill=0.053, valid=False),
        ]
        cal = self._cal(pts)
        path = cal.save(tmp_path / "sub" / "cal.json")
        loaded = TargetSizeCalibration.load(path, limits=S)
        assert loaded.points == pts
        assert loaded.window == (40.0, 100.0)
        assert loaded.signature == S.signature()

    def test_signature_mismatch_rejected(self, tmp_path: Path):
        cal = self._cal([_point(40.0, 0.5, 0.7)])
        path = cal.save(tmp_path / "cal.json")
        other = OpticalLimits(
            wavelength_m=1550e-9,
            focal_m=FOCAL_M,
            slm_pixels=(1200, 1920),
            pixel_pitch_m=8e-6,
            camera_pixel_m=2.2e-6,
            camera_shape=(1944, 2592),
        )
        with pytest.raises(TargetSizeCalibrationError, match="签名"):
            TargetSizeCalibration.load(path, limits=other)

    def test_missing_file_rejected(self, tmp_path: Path):
        with pytest.raises(TargetSizeCalibrationError):
            TargetSizeCalibration.load(tmp_path / "nope.json", limits=S)

    def test_recommend_largest_valid(self):
        # 2026-10-23 实测细扫形态: 40–100 有效, 120 起失效 ⇒ 默认推荐 100
        pts = [
            _point(40.0, 0.564, 0.735, vs_base=0.064, valid=True),
            _point(60.0, 0.729, 1.168, vs_base=0.029, valid=True),
            _point(80.0, 0.781, 1.622, vs_base=0.031, valid=True),
            _point(100.0, 0.794, 2.136, vs_base=0.034, valid=True),
            _point(120.0, 0.802, 2.646, vs_base=0.036, mean_core=1.031, valid=False),
        ]
        cal = self._cal(pts)
        assert cal.recommend("square") == pytest.approx(100.0)  # 默认 largest_valid
        assert cal.recommend("square", constraint="largest_valid") == pytest.approx(100.0)

    def test_recommend_best_balance(self):
        pts = [
            _point(40.0, 0.564, 0.735, vs_base=0.064, valid=True),
            _point(60.0, 0.729, 1.168, vs_base=0.029, valid=True),
            _point(80.0, 0.781, 1.622, vs_base=0.031, valid=True),
            _point(100.0, 0.794, 2.136, vs_base=0.034, valid=True),
        ]
        cal = self._cal(pts)
        balanced = cal.recommend("square", constraint="best_balance")
        # (1 − minmax(CV)) × minmax(eff) 的折中点介于最小与最大有效之间
        assert 40.0 < balanced < 100.0

    def test_recommend_unknown_constraint(self):
        cal = self._cal([_point(40.0, 0.5, 0.7)])
        with pytest.raises(ValueError):
            cal.recommend("square", constraint="smallest")

    def test_recommend_no_valid_points_falls_back_to_window(self):
        cal = self._cal([_point(200.0, 0.8, 5.0, valid=False)])
        # 无有效点 ⇒ 回退窗口中心
        assert cal.recommend("square") == pytest.approx(70.0)

    def test_recommend_nothing_at_all_raises(self):
        cal = TargetSizeCalibration(
            signature=S.signature(),
            limits=S,
            points=[_point(200.0, 0.8, 5.0, valid=False)],
            window=None,
            best_size=None,
            method="test",
            notes="x",
            recorded_utc="2026-10-23T00:00:00+00:00",
        )
        with pytest.raises(TargetSizeCalibrationError):
            cal.recommend("square")

    def test_add_and_windows(self):
        cal = self._cal([_point(40.0, 0.5, 0.7)])
        cal.add(_point(80.0, 0.6, 0.9))
        assert len(cal.points) == 2
        with pytest.raises(TypeError):
            cal.add("not a point")

    def test_reversed_window_rejected(self):
        with pytest.raises(ValueError):
            TargetSizeCalibration(
                signature=S.signature(),
                limits=S,
                window=(100.0, 40.0),
                recorded_utc="2026-10-23T00:00:00+00:00",
            )


# --------------------------------------------------------------------------- #
# 2026-10-23 实测细扫回归 (40–240 px, 步长 20)
# --------------------------------------------------------------------------- #
class TestMeasuredFineScanRegression:
    @staticmethod
    def _pts():
        # size: eff / cv / mean_core / fill / valid (用户实测数据)
        raw = [
            (40.0, 0.564, 0.735, 6.631, 0.344, True),
            (60.0, 0.729, 1.168, 3.909, 0.201, True),
            (80.0, 0.781, 1.622, 2.356, 0.121, True),
            (100.0, 0.794, 2.136, 1.500, 0.077, True),
            (120.0, 0.802, 2.646, 1.031, 0.053, False),
            (140.0, 0.81, 3.0, 0.6, 0.03, False),
            (160.0, 0.82, 3.4, 0.4, 0.02, False),
            (180.0, 0.83, 3.9, 0.28, 0.015, False),
            (200.0, 0.84, 4.4, 0.2, 0.012, False),
            (220.0, 0.85, 4.9, 0.15, 0.01, False),
            (240.0, 0.86, 5.3, 0.13, 0.009, False),
        ]
        pts = []
        for size, eff, cv, mean, fill, valid in raw:
            pts.append(
                SizeMetrics(
                    size_px=size,
                    efficiency=eff,
                    uniformity=cv,
                    zero_order_ratio=0.1215,
                    efficiency_vs_baseline=eff - 0.5,
                    mean_core=mean,
                    fill=fill,
                    valid=valid,
                )
            )
        return pts

    def test_valid_partition(self):
        pts = self._pts()
        valid = [p for p in pts if p.valid]
        sizes = [p.size_px for p in valid]
        assert sizes == [40.0, 60.0, 80.0, 100.0]

    def test_window_and_recommendation(self):
        pts = self._pts()
        lo, hi, best, degenerate, message = select_best_window(pts)
        # efficiency 随尺寸单调 (Spearman=1) ⇒ 被判退化, 从分数中剔除
        names = {n for n, _ in degenerate}
        assert "efficiency" in names
        assert "zero_order_ratio" in names
        # 推荐默认 = 最大有效尺寸 100 px (而不是旧口径下的 240 px)
        cal = TargetSizeCalibration(
            signature=S.signature(),
            limits=S,
            points=pts,
            window=(lo, hi) if lo is not None else None,
            windows={"square": (lo, hi)} if lo is not None else {},
            best_size=None,
            method="regression",
            notes="",
            recorded_utc="2026-10-23T00:00:00+00:00",
        )
        assert cal.recommend("square") == pytest.approx(100.0)
        assert cal.recommend("square", constraint="largest_valid") == pytest.approx(100.0)
