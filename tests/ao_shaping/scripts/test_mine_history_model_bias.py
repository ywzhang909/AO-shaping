"""``scripts/slm/mine_history_model_bias.py`` 的测试 (只用合成 / 极小样本)。

覆盖:

* 配对抽取的筛选逻辑 —— 口径不兼容的样本被排除且原因可查;
* 曝光归一化的仿射拟合 —— 已知偏置 + 增益能复原;
* 训练 / 验证按**会话**分组的切分 —— 同一会话不出现在两侧;
* 四类分解在合成数据上的正确性;
* 阈值判定函数。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.slm.mine_history_model_bias import (  # noqa: E402
    CENTROID_SHIFT_LIMIT_CELLS,
    CV_ABS_LIMIT,
    WIDTH_RATIO_TOLERANCE,
    Calibration,
    Exclusion,
    HistoryPair,
    Session,
    _median,
    _verdict,
    decompose_pair,
    fit_calibration,
    fit_exposure_response,
    judge_thresholds,
    normalize_exposure,
    split_sessions_by_session,
)

# ---------------------------------------------------------------------------
# 合成工具
# ---------------------------------------------------------------------------


def _gaussian_spot(size: int, sigma: float, peak: float = 100.0) -> np.ndarray:
    """合成一个高斯光斑 (像素坐标, 中心在正中)。"""
    axis = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    ys, xs = np.meshgrid(axis, axis, indexing="ij")
    return peak * np.exp(-(xs**2 + ys**2) / (2.0 * sigma**2))


def _make_pair(
    pair_id: str,
    session_id: str,
    *,
    sigma_pred: float = 3.0,
    sigma_meas: float = 3.0,
    shift: float = 0.0,
    gain: float = 1.0,
    size: int = 64,
) -> HistoryPair:
    """造一对合成的 (预测, 实测)。"""
    predicted = _gaussian_spot(size, sigma_pred)
    measured = _gaussian_spot(size, sigma_meas, peak=100.0 * gain)
    if shift:
        from scipy.ndimage import shift as _shift

        measured = _shift(measured, shift=shift, order=1, mode="constant", cval=0.0)
    return HistoryPair(
        pair_id=pair_id,
        session_id=session_id,
        campaign="synthetic",
        stamp="synthetic",
        variant="synthetic",
        probe_split="train",
        round_index=1,
        file=f"{pair_id}.npz",
        exposure_ms=0.4,
        region=size,
        far_field_size=size,
        camera_px_per_model_px=1.0,
        rotation_quarters=1,
        phase=np.zeros((size, size)),
        predicted_crop=predicted,
        measured_crop=measured,
        exposure_fit=fit_exposure_response([(0.4, 40.0)], "synthetic"),
    )


def _make_session(session_id: str, pairs: list[HistoryPair]) -> Session:
    """造一个合成会话。"""
    return Session(
        session_id=session_id,
        campaign="synthetic",
        stamp=session_id,
        variant="synthetic",
        region=64,
        far_field_size=64,
        camera_px_per_model_px=1.0,
        rotation_quarters=1,
        focal_length_m=0.1224,
        wavelength_m=1064e-9,
        slm_pixel_um=8.0,
        ccd_pixel_um=2.2,
        panel_span_px=900.0,
        exposure_fit=fit_exposure_response([(0.4, 40.0)], "synthetic"),
        pairs=pairs,
    )


# ---------------------------------------------------------------------------
# 曝光归一化
# ---------------------------------------------------------------------------


class TestExposureResponse:
    """曝光响应曲线 ``counts = bias + gain · exposure_ms``。"""

    def test_known_bias_and_gain_are_recovered(self) -> None:
        """已知偏置 + 增益能被复原。"""
        truth_bias, truth_gain = 2.5, 3.25
        points = [
            (exposure, truth_bias + truth_gain * exposure)
            for exposure in (0.05, 0.1, 0.2, 0.4, 0.8)
        ]
        fit = fit_exposure_response(points, "synthetic")
        assert fit.bias == pytest.approx(truth_bias, abs=1e-9)
        assert fit.gain == pytest.approx(truth_gain, abs=1e-9)
        assert fit.max_abs_residual == pytest.approx(0.0, abs=1e-9)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-12)

    def test_normalization_recovers_the_underlying_signal(self) -> None:
        """归一化后的辐照度与真值一致 (偏置 + 增益都被除掉)。"""
        truth_bias, truth_gain = 1.5, 2.0
        exposure = 0.25
        raw = truth_bias + truth_gain * exposure * np.full((8, 8), 3.0)
        fit = fit_exposure_response(
            [(0.1, truth_bias + truth_gain * 0.1), (0.5, truth_bias + truth_gain * 0.5)],
            "synthetic",
        )
        normalized = normalize_exposure(raw, exposure, fit)
        assert normalized.mean() == pytest.approx(3.0, abs=1e-9)

    def test_single_exposure_degrades_to_identity(self) -> None:
        """曝光单一时退化为 gain=1 / bias=0, 并在来源里写明。"""
        fit = fit_exposure_response([(0.4, 40.0)], "synthetic")
        assert fit.gain == 1.0
        assert fit.bias == 0.0
        assert "退化" in fit.source

    def test_saturated_points_are_left_to_the_caller(self) -> None:
        """饱和点由调用方剔除; 本函数只做拟合, 不擅自剔除。"""
        truth_bias, truth_gain = 2.0, 3.0
        points = [
            (0.1, truth_bias + truth_gain * 0.1),
            (0.5, truth_bias + truth_gain * 0.5),
            (1.0, 255.0),  # 饱和
        ]
        fit = fit_exposure_response(points, "synthetic")
        assert fit.n_points == 3
        assert fit.max_abs_residual is not None and fit.max_abs_residual > 1.0


# ---------------------------------------------------------------------------
# 四类分解
# ---------------------------------------------------------------------------


class TestDecomposition:
    """四类分解 (能量 / 宽度 / 位置 / 形状) 在合成数据上的正确性。"""

    def test_identical_images_give_unit_ratios(self) -> None:
        """预测 == 实测 ⇒ 能量比 1、宽度比 1、质心偏移 0、形状相关 1。"""
        image = _gaussian_spot(64, 4.0)
        result = decompose_pair(image, image.copy(), 64)
        assert result["energy_ratio"] == pytest.approx(1.0, abs=1e-9)
        assert result["width_ratio"] == pytest.approx(1.0, abs=1e-9)
        assert result["centroid_shift_cells"] == pytest.approx(0.0, abs=1e-6)
        assert result["shape_correlation"] == pytest.approx(1.0, abs=1e-9)
        assert result["correlation_raw"] == pytest.approx(1.0, abs=1e-9)

    def test_width_ratio_tracks_the_synthetic_width(self) -> None:
        """宽度比 = 预测宽度 / 实测宽度。"""
        narrow = _gaussian_spot(64, 3.0)
        wide = _gaussian_spot(64, 4.0)
        result = decompose_pair(narrow, wide, 64)
        assert result["width_ratio"] == pytest.approx(3.0 / 4.0, rel=1e-6)

    def test_centroid_shift_tracks_the_synthetic_shift(self) -> None:
        """质心偏移 = 预测质心与实测质心的距离 (格)。

        符号约定与审计一致: ``dx = 预测质心 - 实测质心``。
        """
        from scipy.ndimage import shift as _shift

        image = _gaussian_spot(64, 4.0)
        moved = _shift(image, shift=(0.0, 3.0), order=1, mode="constant", cval=0.0)
        result = decompose_pair(image, moved, 64)
        assert result["centroid_shift_cells"] == pytest.approx(3.0, rel=0.05)
        assert result["centroid_shift_dx_cells"] == pytest.approx(-3.0, rel=0.05)

    def test_energy_ratio_tracks_the_synthetic_gain(self) -> None:
        """能量比 = 预测总能量 / 实测总能量。"""
        predicted = _gaussian_spot(64, 4.0)
        measured = _gaussian_spot(64, 4.0, peak=50.0)
        result = decompose_pair(predicted, measured, 64)
        assert result["energy_ratio"] == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 会话切分
# ---------------------------------------------------------------------------


class TestSessionSplit:
    """按会话分组的切分。"""

    def test_no_session_appears_on_both_sides(self) -> None:
        """同一会话不出现在训练 / 验证两侧。"""
        sessions = []
        for index in range(10):
            pairs = [_make_pair(f"s{index}p{j}", f"session-{index}") for j in range(2)]
            sessions.append(_make_session(f"session-{index}", pairs))
        train, held = split_sessions_by_session(sessions, fraction=0.7, seed=7)
        train_ids = {s.session_id for s in train}
        held_ids = {s.session_id for s in held}
        assert train_ids and held_ids
        assert not (train_ids & held_ids)
        assert train_ids | held_ids == {s.session_id for s in sessions}

    def test_every_pair_stays_with_its_session(self) -> None:
        """配对不与自己的会话分家。"""
        sessions = []
        for index in range(6):
            pairs = [_make_pair(f"s{index}p{j}", f"session-{index}") for j in range(3)]
            sessions.append(_make_session(f"session-{index}", pairs))
        train, held = split_sessions_by_session(sessions, fraction=0.5, seed=3)
        for group in (train, held):
            for session in group:
                for pair in session.pairs:
                    assert pair.session_id == session.session_id


# ---------------------------------------------------------------------------
# 标定拟合
# ---------------------------------------------------------------------------


class TestCalibration:
    """系统误差拟合 (焦距尺度 / 瞳面中心 / 增益)。"""

    def test_scale_and_gain_are_recovered(self) -> None:
        """已知「模型偏宽 + 偏暗」能被拟合出来。"""
        pairs = [
            _make_pair(
                f"p{index}",
                "session-0",
                sigma_pred=4.0,
                sigma_meas=3.0,
                gain=2.0,
            )
            for index in range(5)
        ]
        calibration = fit_calibration(pairs)
        # 模型比实测宽 4/3 ⇒ 缩放因子应为 3/4。
        assert calibration.scale == pytest.approx(0.75, rel=0.05)
        # 实测总能量是模型的两倍 ⇒ 增益应为 2。
        assert calibration.gain == pytest.approx(2.0, rel=0.05)
        assert abs(calibration.shift_dx) < 0.2
        assert abs(calibration.shift_dy) < 0.2

    def test_shift_is_recovered(self) -> None:
        """已知的图样平移能被拟合成中心偏移。"""
        pairs = [
            _make_pair(f"p{index}", "session-0", shift=2.0) for index in range(4)
        ]
        calibration = fit_calibration(pairs)
        assert calibration.shift_dx == pytest.approx(2.0, abs=0.3)

    def test_empty_input_is_identity(self) -> None:
        """没有配对时返回恒等标定。"""
        calibration = fit_calibration([])
        assert calibration.scale == 1.0
        assert calibration.gain == 1.0
        assert calibration.n_pairs == 0


# ---------------------------------------------------------------------------
# 筛选逻辑
# ---------------------------------------------------------------------------


class TestExclusionLogic:
    """配对抽取的筛选逻辑: 口径不兼容的样本被排除且原因可查。"""

    def test_exclusion_records_are_serialisable_and_readable(self) -> None:
        """排除记录带原因, 且能写进 JSON。"""
        exclusions = [
            Exclusion("session", "20261019_103208", "rolled_back_coefficients", "回滚"),
            Exclusion("session", "x", "simulation", "cam_type=sim"),
            Exclusion("probe", "y", "degenerate_frame", "帧已饱和"),
        ]
        payload = json.dumps([item.to_dict() for item in exclusions], ensure_ascii=False)
        restored = json.loads(payload)
        assert {item["reason"] for item in restored} == {
            "rolled_back_coefficients",
            "simulation",
            "degenerate_frame",
        }
        assert all(item["identifier"] for item in restored)

    def test_rolled_back_stamp_is_denied(self) -> None:
        """审计判定为回滚的会话戳被列入排除名单。"""
        from scripts.slm.mine_history_model_bias import ROLLED_BACK_STAMPS

        assert "20261019_103208" in ROLLED_BACK_STAMPS

    def test_incompatible_geometry_is_rejected(self) -> None:
        """缺几何 (无法做相机 ↔ 模型网格换算) 的会话被排除。"""
        from scripts.slm.mine_history_model_bias import _footprint_cells

        assert _footprint_cells(300, 1.0467) == 287
        assert _footprint_cells(300, 2.7955) == 107
        # 尺度非法时退到一个最小视场, 不抛异常。
        assert _footprint_cells(300, 0.0) == 8


# ---------------------------------------------------------------------------
# 阈值判定
# ---------------------------------------------------------------------------


class TestThresholdJudgement:
    """阶段 5 的阈值判定。"""

    def test_width_ratio_band(self) -> None:
        """宽度比进 1.0 ± 0.10 判为达标。"""
        assert _verdict("宽度比", 1.05, 1.05, WIDTH_RATIO_TOLERANCE, "band") == (
            "标定前后均达标"
        )
        assert _verdict("宽度比", 1.20, 1.05, WIDTH_RATIO_TOLERANCE, "band") == (
            "标定后达标 (标定前不达标)"
        )
        assert _verdict("宽度比", 1.20, 1.18, WIDTH_RATIO_TOLERANCE, "band") == (
            "标定后仍未达标"
        )

    def test_centroid_and_cv_limits(self) -> None:
        """质心偏移与 CV 绝对误差的上限判定。"""
        assert (
            _verdict("质心", 0.9, 0.4, CENTROID_SHIFT_LIMIT_CELLS, "max")
            == "标定后达标 (标定前不达标)"
        )
        assert (
            _verdict("CV", 0.2, 0.04, CV_ABS_LIMIT, "max")
            == "标定后达标 (标定前不达标)"
        )

    def test_judge_thresholds_builds_a_verdict_block(self) -> None:
        """judge_thresholds 给出三条判据的通过与否。"""
        block = judge_thresholds(
            {
                "width_ratio_median": 1.05,
                "centroid_shift_median": 0.4,
                "cv_abs_error_median": 0.04,
            }
        )
        assert block["width_ratio"]["passed"] is True
        assert block["centroid_shift"]["passed"] is True
        assert block["cv_abs_error"]["passed"] is True

    def test_missing_values_are_marked_not_zero(self) -> None:
        """缺数据时写「未记录」, 不填 0。"""
        block = judge_thresholds({})
        assert block["width_ratio"]["available"] is False
        assert "未记录" in str(block["width_ratio"]["reason"])


def test_median_helper() -> None:
    """_median 对空序列返回 0, 对非空返回中位数。"""
    assert _median([]) == 0.0
    assert _median([3.0, 1.0, 2.0]) == 2.0
