"""多曝光 HDR 融合的离线验证（合成场景，不碰设备）。"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.hdr import HdrFuseError, hdr_fuse, robust_mean


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
