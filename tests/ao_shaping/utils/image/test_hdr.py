"""多曝光 HDR 融合的离线验证（合成场景，不碰设备）。"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.hdr import HdrFuseError, box_metrics, hdr_fuse, robust_mean


def _scene(radiance: np.ndarray, exposure_ms: float, *, full_scale: float = 255.0) -> np.ndarray:
    return np.clip(radiance * exposure_ms, 0, full_scale)


def test_recovers_dim_and_bright_regions_together():
    """亮区在长曝光饱和、暗区在短曝光贴底 ⇒ 融合必须同时还原两者。"""
    radiance = np.array([[50.0, 0.5]])          # counts/ms
    short = _scene(radiance, 4.0)                # 短曝光: 200, 2
    long = _scene(radiance, 40.0)                # 长曝光: 2000→255 饱和, 20
    result = hdr_fuse([short, long], [4.0, 40.0], full_scale=255.0, reference_exposure_ms=4.0)
    assert np.allclose(result.radiance, radiance, rtol=1e-6)
    assert np.allclose(result.fused, radiance * 4.0, rtol=1e-6)
    # 亮区取短曝光（第 0 帧）、暗区取长曝光（第 1 帧）
    assert result.source_index[0, 0] == 0
    assert result.source_index[0, 1] == 1
    assert result.valid.all()
    assert result.coverage == 1.0


def test_all_saturated_pixels_are_invalid():
    bright = np.full((2, 2), 250.0)
    result = hdr_fuse([bright, bright], [4.0, 40.0], full_scale=255.0)
    assert not result.valid.any()
    assert result.invalid_pixels == 4
    assert result.saturated_pixels == 4


def test_radiance_is_invariant_to_common_exposure_scaling():
    """所有曝光同乘一个常数，radiance（每毫秒）必须不变。"""
    radiance = np.array([[3.0, 0.2], [30.0, 0.05]])
    a = hdr_fuse([_scene(radiance, 2.0), _scene(radiance, 20.0)], [2.0, 20.0])
    b = hdr_fuse([_scene(radiance, 4.0), _scene(radiance, 40.0)], [4.0, 40.0])
    assert np.allclose(a.radiance, b.radiance, rtol=1e-3, atol=1e-3)


def test_prefers_longest_unsaturated_exposure():
    radiance = np.array([[0.5]])                 # 短曝光 2 counts、长曝光 20 counts，都未饱和
    result = hdr_fuse([radiance * 4.0, radiance * 40.0], [4.0, 40.0])
    assert result.source_index[0, 0] == 1        # 取长曝光（信噪比更好）


def test_dark_subtraction_and_min_counts_gate():
    frames = [np.array([[1.0, 0.5]]), np.array([[10.0, 5.0]])]
    result = hdr_fuse(frames, [4.0, 40.0], dark=1.0, min_counts=0.0)
    assert np.isclose(result.radiance[0, 0], (10.0 - 1.0) / 40.0)   # 0 计数被 min_counts 挡住 ⇒ 用长曝光
    assert result.valid[0, 1]


def test_rejects_bad_inputs():
    with pytest.raises(HdrFuseError):
        hdr_fuse([], [])
    with pytest.raises(HdrFuseError):
        hdr_fuse([np.zeros((2, 2))], [4.0, 40.0])
    with pytest.raises(HdrFuseError):
        hdr_fuse([np.zeros((2, 2))], [-1.0])
    with pytest.raises(HdrFuseError):
        hdr_fuse([np.zeros((2, 2)), np.zeros((3, 2))], [4.0, 40.0])


def test_robust_mean_clips_outliers():
    image = np.array([[1.0, 1.0, 1.0, 1.0, 100.0]])
    mask = np.ones((1, 5), dtype=bool)
    assert robust_mean(image, mask) == pytest.approx(1.0, abs=1e-9)
    assert robust_mean(image, mask, clip_sigma=None) == pytest.approx(20.8, rel=1e-6)


def test_box_metrics_reports_fill_and_validity():
    image = np.zeros((101, 101))
    image[45:56, 45:56] = 20.0       # 11x11 的亮块（围绕中心）
    image[2:5, 2:5] = 5.0            # 远处另一团光，保证 ``ee_box`` 有区分度
    small = box_metrics(image, (50, 50), box_px=11, step_px=1.0)
    large = box_metrics(image, (50, 50), box_px=39, step_px=1.0)
    assert small["mean_core"] > large["mean_core"]
    assert 0 < large["fill"] < small["fill"] <= 1
    assert small["ee_box"] >= large["ee_box"]         # 能量都在两块内部时二者相等（不因框大而虚高）
    gated = box_metrics(image, (50, 50), box_px=39, step_px=1.0, signal_floor=5.0)
    assert gated["valid"] is False                     # 大框均值远低于门槛
    assert box_metrics(image, (50, 50), box_px=11, step_px=1.0, signal_floor=5.0)["valid"] is True


def test_box_metrics_handles_all_nan():
    image = np.full((21, 21), np.nan)
    result = box_metrics(image, (10, 10), box_px=10)
    assert not result["valid"]
    assert np.isnan(result["mean_core"])
    assert result["coverage"] == 0.0





def test_read_noise_bias_is_corrected_at_long_exposure():
    """相机把负噪声截到 0 ⇒ 暗区在长曝光下系统性正偏；给 read_noise_counts 后应消掉。"""
    rng = np.random.default_rng(7)
    truth = np.full((64, 64), 0.07)                       # counts/ms
    exposures = [1.0, 4.0, 16.0]
    frames = [np.clip(truth * e + rng.normal(0, 1.2, truth.shape), 0, 255) for e in exposures]
    naive = hdr_fuse(frames, exposures, full_scale=255.0, reference_exposure_ms=4.0,
                     selection="longest")
    # 0.07 counts/ms 在 16 ms 下只有 1.1 counts ≈ 0.9σ ⇒ 本就不该被"算出来"
    gated = hdr_fuse(frames, exposures, full_scale=255.0, reference_exposure_ms=4.0,
                     read_noise_counts=1.2, min_snr=3.0)
    assert naive.coverage > 0.5                       # 旧行为：硬算，覆盖率虚高
    assert gated.coverage < 0.05                      # 新行为：信号不足 ⇒ 判不可测
    # 给到 64 ms（≈ 4.5 counts ≈ 3.7σ）后，同一区域应变为可测且接近真值
    long_frames = [np.clip(truth * e + rng.normal(0, 1.2, truth.shape), 0, 255) for e in (4.0, 16.0, 64.0)]
    measured = hdr_fuse(long_frames, [4.0, 16.0, 64.0], full_scale=255.0, reference_exposure_ms=4.0,
                        read_noise_counts=1.2, min_snr=3.0)
    assert measured.coverage > 0.5
    error = abs(float(np.nanmean(measured.radiance)) - 0.07) / 0.07
    assert error < 0.25


def test_selection_modes_agree_on_clean_data():
    radiance = np.array([[0.3, 3.0]])
    frames = [radiance * 2.0, radiance * 20.0]
    a = hdr_fuse(frames, [2.0, 20.0], selection="longest")
    b = hdr_fuse(frames, [2.0, 20.0], selection="best_snr")
    assert np.allclose(a.radiance, b.radiance, rtol=1e-9)


def test_unknown_selection_rejected():
    with pytest.raises(HdrFuseError):
        hdr_fuse([np.zeros((2, 2))], [4.0], selection="median")



def test_box_metrics_rejects_low_core_coverage():
    """HDR 阶梯没覆盖暗区时，核心区大部分像素无效 ⇒ 该框必须判不可测（不能拿亮像素充数）。"""
    image = np.full((41, 41), np.nan)
    image[18:23, 18:23] = 5.0                    # 只有极少数像素有效
    result = box_metrics(image, (20, 20), box_px=20, min_coverage=0.9)
    assert result["core_coverage"] < 0.3
    assert result["valid"] is False
    full = np.full((41, 41), 5.0)
    assert box_metrics(full, (20, 20), box_px=20, min_coverage=0.9)["valid"] is True
