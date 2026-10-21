"""离线测试: 大恒改曝光后, 第一帧必须已经是新曝光。

背景 (2026-10-21 实测, 相机 FJB24112232):

大恒 SDK 在**取流进行中**直接写 ``ExposureTime``, 改动之前就已经开始积分的帧仍留在
SDK 流水线里 —— 改完曝光后紧接着取到的 1~2 帧还是**上一档曝光**的画面:

- 2 -> 20 ms: 第 1 帧 mean=0.343/max=5、第 2 帧 0.320/5 (都还是 2 ms 的画面),
  第 3 帧才出现 20 ms 的 0.559/42;
- 20 -> 2 ms: 前两帧 0.596/43、0.596/41 (还是 20 ms), 第 3 帧 0.326/5;
- 1 -> 50 / 10 -> 100 ms: 滞后 1 帧; 100 -> 10 ms 那次没滞后 (间歇, 所以不能靠
  「发现不一致再补救」, 必须无条件消费)。

``get_numpy_image`` 自带的 ``skip_first`` 只丢 1 帧, **不够**。后果是实打实的:
2026-10-20 录下的测光阶梯 (``data/debug/light_level_probe/20261020_092550``)
把 5 ms 档的峰值记成了 2 ms 档的值 (17 == 17), 而第二帧与第一帧之差大到 ≈ 峰值本身
(50 ms 档 delta=156 vs max=158) —— 那一档的读数是旧曝光的画面。

修复: ``DahengCamera.reset_exposure_time`` 在曝光真的改变后消费掉
``EXPOSURE_CHANGE_DISCARD_FRAMES`` 帧, 把「改曝光后的第一帧就是新曝光」变成驱动层
的不变量。

本文件用假相机离线锁住这条不变量: 假相机在 ``ExposureTime.set`` 时把「旧曝光下
已经排队」的帧放进队列, 取帧时先出队 —— 于是「驱动是否消费掉了旧帧」直接表现为
「返回的帧带的是哪一档曝光」。
"""

from unittest.mock import patch

import numpy as np


class _FakeExposureFeature:
    """GenICam 的 ``ExposureTime`` 特征 (整数微秒)。"""

    def __init__(self, cam: "_FakeCam", value_us: int) -> None:
        self._cam = cam
        self._value_us = int(value_us)

    def set(self, value_us: int) -> None:
        self._value_us = int(value_us)
        self._cam._on_exposure_set(self._value_us)

    def get(self) -> int:
        return self._value_us


class _FakeFrame:
    """一帧; 像素值编码「采集这一帧时的曝光档位」(ms)。"""

    def __init__(self, exposure_ms: float, success_token) -> None:
        self._exposure_ms = float(exposure_ms)
        self._success = success_token

    def get_status(self):
        return self._success

    def get_numpy_array(self) -> np.ndarray:
        return np.full((2, 2), self._exposure_ms, dtype=np.uint8)


class _FakeStream:
    def __init__(self, cam: "_FakeCam") -> None:
        self._cam = cam

    def get_image(self) -> _FakeFrame:
        return self._cam._pop_frame()


class _FakeCam:
    """模拟 SDK 的排队行为: 改曝光时, 已经在流水线里的旧帧**不会**被丢掉。"""

    def __init__(self, stale_frames: int, success_token, exposure_ms: float = 20.0):
        self.ExposureTime = _FakeExposureFeature(self, int(exposure_ms * 1000))
        self.data_stream = [_FakeStream(self)]
        self._stale_frames = int(stale_frames)
        self._success = success_token
        self._current_ms = float(exposure_ms)
        self._pending: list[_FakeFrame] = []
        self.grabs = 0

    # --- SDK 侧行为 ---------------------------------------------------------
    def _on_exposure_set(self, value_us: int) -> None:
        # 曝光改动时, 旧曝光下已经在积分/排队的帧仍然留在队列里。
        for _ in range(self._stale_frames):
            self._pending.append(self._make_frame(self._current_ms))
        self._current_ms = value_us / 1000.0

    def _make_frame(self, exposure_ms: float) -> _FakeFrame:
        return _FakeFrame(exposure_ms, self._success)

    def _pop_frame(self) -> _FakeFrame:
        self.grabs += 1
        if self._pending:
            return self._pending.pop(0)
        return self._make_frame(self._current_ms)


def _make_cam(monkeypatch, stale_frames: int = 2, exposure_ms: float = 20.0):
    """构造接上假相机的 DahengCamera (不碰真实 SDK)。"""
    from ao_shaping.drivers.ccd.daheng import driver as daheng_driver

    cam = daheng_driver.DahengCamera(cam_id=0, exposure_time_ms=exposure_ms)
    mock_cam = _FakeCam(
        stale_frames=stale_frames,
        success_token=daheng_driver.gx.GxFrameStatusList.SUCCESS,
        exposure_ms=exposure_ms,
    )
    cam.cam = mock_cam
    return cam, mock_cam


class TestExposureChangeConsumesStaleFrames:
    def test_first_frame_after_the_change_is_the_new_exposure(self, monkeypatch):
        """改曝光后的第一帧必须是新曝光的画面 (核心不变量)。"""
        cam, _ = _make_cam(monkeypatch, stale_frames=2)

        cam.reset_exposure_time(2.0)
        frame = cam.get_numpy_image(n_sample=1)

        assert frame[0, 0] == 2.0, (
            "改曝光后的第一帧仍是旧曝光的画面 —— 驱动没有消费掉 SDK 流水线里的旧帧"
        )

    def test_the_stale_frames_are_actually_consumed(self, monkeypatch):
        """驱动必须真的去取那些帧 (而不是靠调用方的时序碰运气)。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        cam, mock_cam = _make_cam(monkeypatch, stale_frames=2)

        cam.reset_exposure_time(2.0)

        assert mock_cam.grabs == DahengCamera.EXPOSURE_CHANGE_DISCARD_FRAMES

    def test_two_stale_frames_are_covered_by_the_default_budget(self, monkeypatch):
        """实测滞后上限是 2 帧, 默认预算必须覆盖它。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        assert DahengCamera.EXPOSURE_CHANGE_DISCARD_FRAMES >= 2

    def test_unchanged_exposure_consumes_nothing(self, monkeypatch):
        """曝光没变就不该白丢帧 (否则每次设置都多花 3 帧采集时间)。"""
        cam, mock_cam = _make_cam(monkeypatch, stale_frames=2, exposure_ms=20.0)

        cam.reset_exposure_time(20.0)

        assert mock_cam.grabs == 0

    def test_the_readback_is_still_returned(self, monkeypatch):
        """既有的契约不变: 返回设备真正生效的曝光 (经 µs 量化)。"""
        cam, _ = _make_cam(monkeypatch, stale_frames=2)

        effective = cam.reset_exposure_time(2.0)

        assert effective == 2.0
        assert cam.get_exposure_time_ms() == 2.0

    def test_the_budget_is_tunable(self, monkeypatch):
        """把预算设成 0 就退回旧行为 (便于对照)。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        monkeypatch.setattr(DahengCamera, "EXPOSURE_CHANGE_DISCARD_FRAMES", 0)
        cam, mock_cam = _make_cam(monkeypatch, stale_frames=2)

        cam.reset_exposure_time(2.0)
        grabs_after_reset = mock_cam.grabs
        frame = cam.get_numpy_image(n_sample=1)

        assert grabs_after_reset == 0, "预算为 0 时驱动不该消费任何帧"
        assert frame[0, 0] == 20.0, "预算为 0 时应复现旧行为 (第一帧仍是旧曝光)"


@patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
def test_module_level_sdk_stub_is_not_required(_mock_gx):
    """本文件用 monkeypatch 注入假相机; 这里只保证 SDK 缺失时模块仍可导入。"""
    from ao_shaping.drivers.ccd.daheng import driver as daheng_driver

    assert hasattr(daheng_driver, "DahengCamera")
