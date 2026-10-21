"""离线测试: 大恒 ``reset_exposure_time`` 之后第一帧就是新曝光。

背景坑 (2026-10-21 实测, 相机 FJB24112232): gxipy 在取流进行中直接写
``ExposureTime``, **改动之前**就已经开始积分的帧仍留在 SDK 流水线里。实测改
曝光后紧接着取到的 1~2 帧还是**上一档曝光**的画面 (``get_numpy_image`` 自带的
``skip_first`` 只丢 1 帧, 不够)。这会让任何「改曝光 -> 立刻量一帧」的判据
(曝光上限阶梯就是其中之一) 配错档位。

修复落在驱动层: ``reset_exposure_time`` 在曝光真的改变后主动消费
``EXPOSURE_CHANGE_DISCARD_FRAMES`` 帧, 把「改曝光后的第一帧就是新曝光」变成
驱动层不变量。本测试用假 SDK 复现「改曝光后前 N 帧返回旧画面」的行为, 并
锁住这条不变量。
"""

from unittest.mock import MagicMock, patch

import numpy as np

SUCCESS = 0


class _FakeFrame:
    def __init__(self, payload: int) -> None:
        self._payload = payload

    def get_status(self) -> int:
        return SUCCESS

    def get_numpy_array(self) -> np.ndarray:
        # 峰值即「这一帧是用哪一档曝光积分出来的」的标记
        return np.full((4, 4), self._payload, dtype=np.uint8)


class _FakeExposureFeature:
    def __init__(self, pipeline: "_FakePipeline") -> None:
        self._pipeline = pipeline
        self._us = 20_000

    def set(self, value: int) -> None:
        self._pipeline.mark_exposure_change(int(value))
        self._us = int(value)

    def get(self) -> int:
        return self._us


class _FakePipeline:
    """改曝光后仍吐出 N 帧旧曝光的 SDK 取流。"""

    def __init__(self, stale_frames: int = 2) -> None:
        self.stale_frames = int(stale_frames)
        self.exposure = _FakeExposureFeature(self)
        self._stale_left = 0
        self._pending_us = self.exposure.get()

    def mark_exposure_change(self, us: int) -> None:
        """SDK 写 ExposureTime: 改动之前就开始积分的帧仍留在流水线里。"""
        if us != self.exposure.get():
            self._pending_us = self.exposure.get()
            self._stale_left = self.stale_frames
        self.exposure._us = int(us)

    def next_payload(self) -> int:
        if self._stale_left > 0:
            self._stale_left -= 1
            us = self._pending_us
        else:
            us = self.exposure.get()
        return int(us / 1000.0)  # 曝光 ms 当作峰值标记


def _make_camera(monkeypatch):
    """构造一个只挂了假 SDK 的 DahengCamera (不碰真硬件)。"""
    from ao_shaping.drivers.ccd.daheng import driver as daheng_driver

    cam = daheng_driver.DahengCamera(cam_id=0, exposure_time_ms=20.0)
    pipeline = _FakePipeline(stale_frames=2)
    mock_cam = MagicMock()
    mock_cam.ExposureTime = pipeline.exposure
    stream = MagicMock()
    stream.get_image.side_effect = lambda: _FakeFrame(pipeline.next_payload())
    mock_cam.data_stream = [stream]
    cam.cam = mock_cam
    monkeypatch.setattr(daheng_driver, "gx", MagicMock(), raising=False)
    monkeypatch.setattr(
        daheng_driver.gx, "GxFrameStatusList", MagicMock(SUCCESS=SUCCESS), raising=False
    )
    return cam, pipeline


def _peak(cam) -> int:
    return int(np.max(np.asarray(cam.get_numpy_image(n_sample=1))))


def test_the_first_frame_after_an_exposure_change_is_the_new_exposure(monkeypatch) -> None:
    from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

    cam, _pipeline = _make_camera(monkeypatch)
    for _ in range(5):  # 让流水线在 20 ms 上稳定
        cam.get_numpy_image(n_sample=1)
    assert _peak(cam) == 20

    cam.reset_exposure_time(2.56)
    # 不变量: 改曝光后第一帧就是新曝光 (2.56 ms), 不是上一档 (20 ms)
    assert _peak(cam) == 2


def test_the_invariant_depends_on_the_driver_consuming_the_stale_frames(monkeypatch) -> None:
    """把驱动层的消费关掉, 陈旧帧立刻回到调用方手里 —— 证明这条测试会咬人。"""
    from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

    cam, _pipeline = _make_camera(monkeypatch)
    for _ in range(5):
        cam.get_numpy_image(n_sample=1)

    monkeypatch.setattr(DahengCamera, "EXPOSURE_CHANGE_DISCARD_FRAMES", 0)
    cam.reset_exposure_time(2.56)
    assert _peak(cam) == 20  # 旧曝光的画面又回来了


def test_an_unchanged_exposure_does_not_disturb_the_pipeline(monkeypatch) -> None:
    cam, _pipeline = _make_camera(monkeypatch)
    for _ in range(5):
        cam.get_numpy_image(n_sample=1)
    cam.reset_exposure_time(20.0)  # 与当前相同 -> 不该有陈旧帧
    assert _peak(cam) == 20
