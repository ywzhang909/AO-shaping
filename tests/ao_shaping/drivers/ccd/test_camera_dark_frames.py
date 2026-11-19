"""BaseCamera 的按序列号暗帧加载入口 (离线, 无 SDK / 无硬件)。

锁定 :meth:`BaseCamera.serial_number` / :meth:`load_dark_frames` /
:meth:`get_dark_frame` 的行为: 无序列号必须报错而不是静默落进 "unknown" 目录,
空集是合法状态, 曝光默认严格匹配 (不隐式跨曝光套用)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.drivers.ccd.dark_frames import save_dark_capture


class _StubCamera(BaseCamera):
    """只满足 BaseCamera 抽象类要求的假相机; 不接任何设备。"""

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def initialize(self) -> None:
        self.cam = object()

    def open(self) -> None:
        self.cam = object()

    def close(self) -> None:
        self.cam = None

    def reset_exposure_time(self, time_ms: float) -> float:
        return float(time_ms)

    def reset_window(self, center, size):
        return None

    def enable_auto_exposure(self, enable: bool = True, mode: str = "once") -> None:
        pass

    def get_auto_exposure_state(self) -> dict:
        return {}

    def set_auto_exposure_range(self, min_ms: float, max_ms: float) -> None:
        pass

    @staticmethod
    def get_cam_list() -> list:
        return []

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        return np.zeros((4, 4), dtype=np.uint8)


@pytest.fixture()
def isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """把 CCD 配置根指到 tmp_path, 避免碰真实 data/ccd_configs。"""
    monkeypatch.setenv("CCD_CONFIG_DIR", str(tmp_path))
    return tmp_path


def test_serial_number_reads_backing_sn() -> None:
    cam = _StubCamera()
    assert cam.serial_number is None
    cam._sn = "MER2-TEST-0001"
    assert cam.serial_number == "MER2-TEST-0001"


def test_load_dark_frames_without_serial_raises(isolated_config: Path) -> None:
    cam = _StubCamera()
    with pytest.raises(CameraError, match="序列号"):
        cam.load_dark_frames()


def test_load_dark_frames_empty_when_never_captured(isolated_config: Path) -> None:
    cam = _StubCamera()
    cam._sn = "NO-DARKS"
    darks = cam.load_dark_frames()
    assert not darks
    assert len(darks) == 0


def test_load_and_get_dark_frame_by_exposure(isolated_config: Path) -> None:
    cam = _StubCamera()
    cam._sn = "CAM-WITH-DARKS"
    root = isolated_config / "ccd"  # ccd_config_dir() = CCD_CONFIG_DIR / "ccd"
    mean = np.full((4, 4), 3.0, dtype=np.float32)
    save_dark_capture(cam._sn, exposure_ms=0.043, mean=mean, frames=8, root=root)
    save_dark_capture(cam._sn, exposure_ms=0.047, mean=mean * 2, frames=8, root=root)

    darks = cam.load_dark_frames()
    assert len(darks) == 2
    assert set(darks.exposures_ms) == {0.043, 0.047}

    # 严格匹配: 同曝光拿到, 不存在的曝光默认返回 None (不跨曝光套用)。
    frame = cam.get_dark_frame(0.043)
    assert frame is not None
    assert frame.exposure_ms == pytest.approx(0.043)
    assert frame.mean.shape == (4, 4)
    assert cam.get_dark_frame(0.020) is None

    # 显式允许就近匹配才退到最近档。
    nearest = cam.get_dark_frame(0.020, allow_nearest=True)
    assert nearest is not None
    assert nearest.exposure_ms == pytest.approx(0.043)


def test_get_dark_frame_without_serial_returns_none() -> None:
    cam = _StubCamera()
    assert cam.get_dark_frame(0.043) is None
