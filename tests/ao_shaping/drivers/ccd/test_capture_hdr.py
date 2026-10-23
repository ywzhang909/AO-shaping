"""`BaseCamera.capture_hdr` 的离线验证（假相机，不碰硬件）。

覆盖三件事：
1. 通用采集契约对任意后端可用（只依赖 ``reset_exposure_time`` / ``get_numpy_image``）；
2. 输出是**线性 radiance**（counts/ms），可直接喂 ``box_metrics``；
3. 重构后 ``capture_fused``（显示融合）行为不变，且曝光在退出时恢复。
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.ccd.base import BaseCamera
from ao_shaping.utils.image.hdr import box_metrics


class FakeCamera(BaseCamera):
    """按 radiance 模型出图的假相机；可选钳位与噪声。"""

    def __init__(self, radiance: np.ndarray, *, noise: float = 0.0, full_scale: float = 255.0,
                 clamp_max_ms: float | None = None, seed: int = 0) -> None:
        super().__init__(cam_id=0, exposure_time_ms=4.0, skip_sampling=True)
        self.radiance = np.asarray(radiance, dtype=np.float64)
        self.noise = float(noise)
        self.full_scale = float(full_scale)
        self.clamp_max_ms = clamp_max_ms
        self._rng = np.random.default_rng(seed)
        self.exposure_time_ms = 4.0

    # ---- 抽象接口的最小实现 ----
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def initialize(self) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def reset_exposure_time(self, time_ms: float) -> float:
        value = float(time_ms)
        if self.clamp_max_ms is not None:
            value = min(value, float(self.clamp_max_ms))
        self.exposure_time_ms = max(value, 1e-6)
        return self.exposure_time_ms

    def reset_window(self, *args, **kwargs):
        return None

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = False) -> np.ndarray:
        frame = self.radiance * float(self.exposure_time_ms)
        if self.noise > 0:
            frame = frame + self._rng.normal(0.0, self.noise, frame.shape)
        return np.clip(frame, 0, self.full_scale)

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        return False

    def get_auto_exposure_state(self) -> dict:
        return {"exposure_time_ms": self.exposure_time_ms}

    def set_auto_exposure_range(self, *args, **kwargs):
        return None

    @staticmethod
    def get_cam_list():
        return []


def _scene() -> np.ndarray:
    yy, xx = np.mgrid[:64, :64]
    spot = 40.0 * np.exp(-(((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 3.0 ** 2)))
    halo = 0.05 * np.exp(-(((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 20.0 ** 2)))
    return spot + halo + 0.002


def test_capture_hdr_recovers_radiance():
    camera = FakeCamera(_scene())
    camera.exposure_time_ms = 4.0
    result = camera.capture_hdr([1.0, 4.0, 16.0], reference_exposure_ms=4.0)
    assert result.full_scale == pytest.approx(255.0)
    assert result.applied_exposures_ms == (1.0, 4.0, 16.0)
    truth = _scene()
    for size in (16, 64):
        got = box_metrics(result.radiance, (32, 32), size)
        want = box_metrics(truth, (32, 32), size)
        assert got["mean_core"] == pytest.approx(want["mean_core"], rel=0.05)
    # 融合图 = radiance × 参考曝光
    assert np.allclose(np.nan_to_num(result.fused), np.nan_to_num(result.radiance) * 4.0)


def test_capture_hdr_restores_exposure():
    camera = FakeCamera(_scene())
    camera.exposure_time_ms = 2.5
    camera.capture_hdr([1.0, 4.0])
    assert camera.exposure_time_ms == pytest.approx(2.5)


def test_capture_hdr_uses_clamped_exposure():
    """后端把 16 ms 钳到 8 ms ⇒ 归一化必须用实际值，否则 radiance 会整体偏小。"""
    camera = FakeCamera(_scene(), clamp_max_ms=8.0)
    result = camera.capture_hdr([1.0, 4.0, 16.0], reference_exposure_ms=4.0)
    assert result.applied_exposures_ms == (1.0, 4.0, 8.0)
    got = box_metrics(result.radiance, (32, 32), 16)
    want = box_metrics(_scene(), (32, 32), 16)
    assert got["mean_core"] == pytest.approx(want["mean_core"], rel=0.05)


def test_capture_hdr_flags_unmeasurable_dark_region():
    """阶梯够不到暗区 ⇒ 该区域必须判不可测，而不是给一个噪声整流值。"""
    camera = FakeCamera(_scene(), noise=1.2, seed=3)
    result = camera.capture_hdr([1.0, 4.0, 16.0], read_noise_counts=1.2, min_snr=3.0,
                                reference_exposure_ms=4.0)
    outer = box_metrics(result.radiance, (32, 32), 60)
    assert outer["core_coverage"] < 0.5
    assert outer["valid"] is False


def test_capture_fused_still_works_after_refactor():
    camera = FakeCamera(_scene())
    fused = camera.capture_fused([1.0, 4.0, 16.0])
    assert fused.shape == _scene().shape
    assert float(fused.full_scale) == pytest.approx(255.0)
    assert np.isfinite(fused).all()


def test_invalid_ladders_rejected():
    camera = FakeCamera(_scene())
    with pytest.raises(ValueError):
        camera.capture_hdr([4.0])
    with pytest.raises(ValueError):
        camera.capture_hdr([4.0, -1.0])
    with pytest.raises(ValueError):
        camera.capture_hdr([4.0, 8.0], n_sample=0)

