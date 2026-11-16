r"""离线测试: 大恒驱动审计修复 H1/H2/H3/M1/M2/M3 的回归锁。

每条锁的都是「静默失败」型缺陷 —— 它们不会立刻报错, 而是让调用方拿着一个错误结论
继续跑, 或让采集线程无声挂死:

* **H1** 未开流取帧死等。``data_stream[0].get_image()`` 在 ``stream_off`` 之后
  **永不返回**, 而 ``__take_one_shot`` 外面还套着 ``while True`` —— 停流窗口期
  (``reset_window`` 配窗口、``initialize`` 开流前) 取一帧就是永久挂死, 无超时无日志。
* **H2** ``close()`` 的 ``stream_off`` 一抛错就跳过 ``close_device`` ⇒ 设备句柄泄漏,
  下一次 ``open_device_by_sn`` 直接撞 ``REPEAT_OPENED``。
* **H3** ``initialize()`` 打开设备后、配好之前失败 ⇒ 留下「cam 非空但没配好」的半开
  设备: ``_ensure_open`` 会放行, 句柄同样泄漏。
* **M1** ``enable_auto_exposure`` / ``set_auto_exposure_target`` 写完不回读。gxipy 对
  不可写特征只 print 噪声并**静默保留旧值** (与 ``PixelFormat``、``Width`` 同源),
  于是「没生效」被报成「成功」。
* **M2** ``get_exposure_range`` 返回 ``initialize`` 时采过一次的缓存; 缓存与设备脱节后,
  ``auto_exposure`` 会拿错误边界裁剪曝光。
* **M3** ``reset_exposure_time`` 的排空判定用**本地缓存**比, 不用设备回读比。缓存陈旧时
  「看着没变」会跳过排空, 把上一档曝光的帧交给调用方 (2026-10-21 测光阶梯配错档位的
  那个事故的同一条路径)。
"""

import ast
import textwrap
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ao_shaping.drivers.ccd.base import CameraError
from ao_shaping.drivers.ccd.daheng import constants

W_RANGE = {"min": 4, "max": 1680, "inc": 4}
H_RANGE = {"min": 2, "max": 1138, "inc": 2}
OX_RANGE = {"min": 0, "max": 800, "inc": 4}
OY_RANGE = {"min": 0, "max": 400, "inc": 2}


def _fake_gx() -> types.SimpleNamespace:
    """假 SDK 片段: 取帧路径只读 ``gx.GxFrameStatusList.SUCCESS``。"""
    return types.SimpleNamespace(
        GxFrameStatusList=types.SimpleNamespace(SUCCESS="fake-frame-success")
    )


# ══════════════════════════════════════════════════════════════════════
# 假相机: 帧上编码「采集这一帧时的曝光档位」, 且改曝光时旧帧**不会**被丢掉
# ══════════════════════════════════════════════════════════════════════


class _FakeFrame:
    def __init__(self, exposure_ms: float, success_token) -> None:
        self._exposure_ms = float(exposure_ms)
        self._success = success_token

    def get_status(self):
        return self._success

    def get_numpy_array(self) -> np.ndarray:
        return np.full((2, 2), self._exposure_ms, dtype=np.uint8)


class _FakeExposureFeature:
    """GenICam ``ExposureTime`` 特征 (整数 µs)。"""

    def __init__(self, cam: "_FakeCam", value_us: int) -> None:
        self._cam = cam
        self._value_us = int(value_us)

    def set(self, value_us: int) -> None:
        self._value_us = int(value_us)
        self._cam._on_exposure_set(self._value_us)

    def get(self) -> int:
        return self._value_us

    def get_range(self):
        return self._cam.exposure_range


class _FakeCam:
    """模拟 SDK 的排队行为: 改曝光时, 已经在流水线里的旧帧仍留在队列里。"""

    def __init__(
        self,
        success_token,
        exposure_ms: float = 20.0,
        stale_frames: int = 2,
        exposure_range=None,
    ) -> None:
        self.ExposureTime = _FakeExposureFeature(self, int(exposure_ms * 1000))
        self.data_stream = [types.SimpleNamespace(get_image=self._pop_frame)]
        self.exposure_range = exposure_range
        self._stale_frames = int(stale_frames)
        self._success = success_token
        self._current_ms = float(exposure_ms)
        self._pending: list[_FakeFrame] = []
        self.grabs = 0
        self.stream_on_calls = 0
        self.stream_off_calls = 0
        self.close_device_calls = 0

    def _on_exposure_set(self, value_us: int) -> None:
        for _ in range(self._stale_frames):
            self._pending.append(_FakeFrame(self._current_ms, self._success))
        self._current_ms = value_us / 1000.0

    def _pop_frame(self) -> _FakeFrame:
        self.grabs += 1
        if self._pending:
            return self._pending.pop(0)
        return _FakeFrame(self._current_ms, self._success)

    def stream_on(self) -> None:
        self.stream_on_calls += 1

    def stream_off(self) -> None:
        self.stream_off_calls += 1

    def close_device(self) -> None:
        self.close_device_calls += 1


def _make_cam(fake: _FakeCam, *, exposure_ms: float = 20.0, streaming: bool = True):
    """把假设备挂到一个 DahengCamera 上 (不跑 initialize)。"""
    from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

    cam = DahengCamera(cam_id=0, exposure_time_ms=exposure_ms)
    cam.cam = fake
    cam._acquisition_on = streaming
    return cam


def _window_mock() -> MagicMock:
    """一个能跑通 reset_window 全幅路径的假设备。"""
    mock_cam = MagicMock()
    mock_cam.WidthMax.get.return_value = 1680
    mock_cam.HeightMax.get.return_value = 1140
    mock_cam.Width.get.return_value = 1680
    mock_cam.Height.get.return_value = 1140
    mock_cam.OffsetX.get_range.return_value = OX_RANGE
    mock_cam.OffsetY.get_range.return_value = OY_RANGE
    mock_cam.OffsetX.get.return_value = 0
    mock_cam.OffsetY.get.return_value = 0
    return mock_cam


# ══════════════════════════════════════════════════════════════════════
# H1 — 未开流不许取帧; reset_window 的停流窗口期必须落标记并保证复流
# ══════════════════════════════════════════════════════════════════════


class TestTakeOneShotRefusesToBlockWhenNotStreaming:
    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_raises_instead_of_hanging(self, _mock_gx):
        """停流状态下取帧必须**立刻抛错**: get_image 永不返回, while True 会死等。"""
        fake = _FakeCam("fake-frame-success")
        cam = _make_cam(fake, streaming=False)

        with pytest.raises(CameraError, match="取流未开启"):
            cam._DahengCamera__take_one_shot()

        assert fake.grabs == 0, "拒绝取帧 ⇒ 一次 get_image 都不许发出去"

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_get_numpy_image_propagates_the_same_error(self, _mock_gx):
        """公开取帧入口同样不裸等 (skip_first 的那一帧也会撞上守卫)。"""
        cam = _make_cam(_FakeCam("fake-frame-success"), streaming=False)

        with pytest.raises(CameraError):
            cam.get_numpy_image(n_sample=1)

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", _fake_gx())
    def test_still_returns_a_frame_while_streaming(self):
        """守卫不能误伤正常路径。"""
        cam = _make_cam(_FakeCam("fake-frame-success", exposure_ms=7.0), streaming=True)

        img = cam._DahengCamera__take_one_shot()

        assert img.max() == 7


class TestResetWindowNeverLeavesTheCameraStopped:
    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_flag_is_down_during_configuration_and_up_after(self, _mock_gx):
        """停流窗口期 ``_acquisition_on`` 必须为假 —— 那正是取帧会死等的那段时间。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = _window_mock()
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam
        cam._acquisition_on = True
        seen = {}

        real = DahengCamera._configure_window

        def spy(self, center, size):
            seen["during"] = self._acquisition_on
            return real(self, center, size)

        with patch.object(DahengCamera, "_configure_window", spy):
            cam.reset_window(center=(840, 570), size=(0, 0))

        assert seen["during"] is False
        assert cam._acquisition_on is True

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_a_take_during_the_stopped_window_raises_not_hangs(self, _mock_gx):
        """停流窗口期里任何取帧 (回调 / 并发线程) 都抛错, 不会挂死采集线程。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = _window_mock()
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam
        cam._acquisition_on = True
        errors = []

        real = DahengCamera._configure_window

        def spy(self, center, size):
            try:
                self.get_numpy_image(n_sample=1)
            except CameraError as e:
                errors.append(e)
            return real(self, center, size)

        with patch.object(DahengCamera, "_configure_window", spy):
            cam.reset_window(center=(840, 570), size=(0, 0))

        assert len(errors) == 1
        assert cam._acquisition_on is True

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_failure_in_configuration_still_restores_the_stream(self, _mock_gx):
        """配置中途抛错也不许把相机留在停流状态 (finally 复流)。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = _window_mock()
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam
        cam._acquisition_on = True

        def boom(self, center, size):
            raise RuntimeError("window config exploded")

        with patch.object(DahengCamera, "_configure_window", boom):
            with pytest.raises(RuntimeError, match="exploded"):
                cam.reset_window(center=(840, 570), size=(0, 0))

        assert mock_cam.stream_off.call_count == 1
        assert mock_cam.stream_on.call_count == 1, "抛错也必须复流"
        assert cam._acquisition_on is True

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_stream_on_is_sent_exactly_once_on_success(self, _mock_gx):
        """复流点唯一: 对已开流的设备再发一次 ACQUISITION_START, SDK 会直接报错。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = _window_mock()
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam
        cam._acquisition_on = True

        cam.reset_window(center=(840, 570), size=(0, 0))

        assert mock_cam.stream_off.call_count == 1
        assert mock_cam.stream_on.call_count == 1

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_configure_window_does_not_stream_by_itself(self, _mock_gx):
        """实现体只碰 GenICam 特征; 开/停流是 reset_window 的责任。"""
        import inspect

        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        src = inspect.getsource(DahengCamera._configure_window)
        tree = ast.parse(textwrap.dedent(src))
        fn = tree.body[0]
        # 去掉 docstring, 只看实现体 (docstring 里的 stream_on 属文档措辞, 不是调用)
        if (
            fn.body
            and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)
        ):
            fn.body = fn.body[1:]
        code = ast.unparse(fn)
        assert "stream_on" not in code
        assert "stream_off" not in code


# ══════════════════════════════════════════════════════════════════════
# H2 — close() 分层兜底
# ══════════════════════════════════════════════════════════════════════


class TestCloseAlwaysReleasesTheHandle:
    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_stream_off_failure_does_not_skip_close_device(self, _mock_gx):
        """句柄泄漏 = 下次 open_device_by_sn 撞 REPEAT_OPENED。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        mock_cam.stream_off.side_effect = RuntimeError("stream_off blew up")
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam
        cam._sn = "TEST001"
        cam._acquisition_on = True

        cam.close()  # 不许把二次异常抛给调用方

        mock_cam.close_device.assert_called_once()
        assert cam.cam is None
        assert cam.sn is None
        assert cam._acquisition_on is False
        assert (cam.cam_width, cam.cam_height) == (0, 0)

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_close_device_failure_does_not_leak_a_dead_handle(self, _mock_gx):
        """还句柄本身失败也要清空状态: 留死句柄比报不出二次异常危险得多。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        mock_cam.close_device.side_effect = RuntimeError("close_device blew up")
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam
        cam._sn = "TEST001"
        cam._acquisition_on = True

        cam.close()

        assert cam.cam is None
        assert cam.sn is None
        assert cam._acquisition_on is False

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_flag_drops_before_stream_off_so_a_failure_cannot_resurrect_it(self, _mock_gx):
        """先落标记再停流: stream_off 抛错后不能留下「标记说在取流、设备已停」。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam
        cam._acquisition_on = True

        def check_flag_off(*_a, **_k):
            assert cam._acquisition_on is False
            raise RuntimeError("stream_off blew up")

        mock_cam.stream_off.side_effect = check_flag_off
        cam.close()

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_happy_path_stops_stream_then_closes_device(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam
        cam._acquisition_on = True

        cam.close()

        mock_cam.stream_off.assert_called_once()
        mock_cam.close_device.assert_called_once()
        assert cam.cam is None

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_idempotent_when_not_open(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        cam = DahengCamera(cam_id=0)
        assert cam.cam is None

        cam.close()
        cam.close()

        assert cam.cam is None
        assert cam._acquisition_on is False


# ══════════════════════════════════════════════════════════════════════
# H3 — initialize() 失败不许留半开设备
# ══════════════════════════════════════════════════════════════════════


def _initialize_mocks(mock_gx, mock_cam, sn="TEST001"):
    mock_cam.ExposureTime.get_range.return_value = {"min": 20.0, "max": 100_000.0}
    mock_cam.WidthMax.get.return_value = 1920
    mock_cam.HeightMax.get.return_value = 1080
    mock_cam.Width.get.return_value = 1920
    mock_cam.Height.get.return_value = 1080
    mock_cam.PixelFormat.get.return_value = (
        constants.GxPixelFormatEntry.MONO8.value,
        "Mono8",
    )
    mock_cam.data_stream = [MagicMock()]
    mock_cam.data_stream[0].StreamBufferHandlingMode.get.return_value = (
        constants.GxDSStreamBufferHandlingModeEntry.NEWEST_ONLY.value,
        "NewestOnly",
    )
    mock_gx.GxPixelFormatEntry.MONO8 = 0x01080001
    manager = MagicMock()
    mock_gx.DeviceManager.return_value = manager
    manager.update_device_list.return_value = (1, [{"sn": sn}])
    manager.open_device_by_sn.return_value = mock_cam
    return manager


class TestInitializeReleasesTheDeviceOnFailure:
    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_config_failure_closes_the_device_and_reraises(self, mock_gx):
        """半开态最坏: ``_ensure_open`` 会放行, 调用方拿着没配好的设备继续跑。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        _initialize_mocks(mock_gx, mock_cam)
        mock_cam.Gain.set.side_effect = RuntimeError("gain write failed")

        cam = DahengCamera(cam_id=0, exposure_time_ms=10.0)
        with pytest.raises(RuntimeError, match="gain write failed"):
            cam.open()

        mock_cam.close_device.assert_called_once()
        assert cam.cam is None
        assert cam.sn is None
        assert cam._acquisition_on is False

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_close_device_failure_does_not_mask_the_original_error(self, mock_gx):
        """释放路径上的二次异常只记 warning, 不许盖掉真正的原始错误。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        _initialize_mocks(mock_gx, mock_cam)
        mock_cam.Gain.set.side_effect = RuntimeError("gain write failed")
        mock_cam.close_device.side_effect = RuntimeError("close_device blew up")

        cam = DahengCamera(cam_id=0, exposure_time_ms=10.0)
        with pytest.raises(RuntimeError, match="gain write failed"):
            cam.open()

        assert cam.cam is None

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_stream_buffer_mode_failure_also_releases(self, mock_gx):
        """NewestOnly 没切成功是显式 raise 的一条路径, 同样不许留半开设备。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        _initialize_mocks(mock_gx, mock_cam)
        mock_cam.data_stream[0].StreamBufferHandlingMode.get.return_value = (999, "OldestFirst")

        cam = DahengCamera(cam_id=0, exposure_time_ms=10.0)
        with pytest.raises(RuntimeError, match="NewestOnly"):
            cam.open()

        mock_cam.close_device.assert_called_once()
        assert cam.cam is None
        assert cam._acquisition_on is False

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_success_path_still_streams_and_marks_acquisition_on(self, mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        _initialize_mocks(mock_gx, mock_cam)

        cam = DahengCamera(cam_id=0, exposure_time_ms=10.0)
        cam.open()

        mock_cam.stream_on.assert_called_once()
        assert cam._acquisition_on is True
        assert cam.sn == "TEST001"
        mock_cam.close_device.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
# M1 — 自动曝光特征写完必须回读校验
# ══════════════════════════════════════════════════════════════════════


class TestIntFromReadback:
    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_normalizes_every_shape_gxipy_uses(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        f = DahengCamera._int_from_readback
        assert f(1) == 1
        assert f(0) == 0
        assert f(True) == 1
        assert f(2.0) == 2
        assert f((1, "Continuous")) == 1  # gxipy EnumFeature.get() 的形态
        assert f(constants.GxAutoEntry.ONCE) == 2  # 枚举成员
        assert f(types.SimpleNamespace(value=7)) == 7

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_unparseable_readback_is_a_failure_not_a_pass(self, _mock_gx):
        """裸 MagicMock 绝不能被当成「设备已确认」, 否则这道校验形同虚设。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        assert DahengCamera._int_from_readback(MagicMock()) is None
        assert DahengCamera._int_from_readback(None) is None
        assert DahengCamera._int_from_readback("Continuous") is None
        assert DahengCamera._int_from_readback(()) is None


class TestEnableAutoExposureVerifiesTheDevice:
    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_true_only_when_the_device_confirms(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        mock_cam.ExposureAuto.get.return_value = (
            constants.GxAutoEntry.CONTINUOUS.value,
            "Continuous",
        )
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam

        assert cam.enable_auto_exposure(enable=True) is True
        mock_cam.ExposureAuto.set.assert_called_once_with(
            constants.GxAutoEntry.CONTINUOUS.value
        )
        mock_cam.ExposureAuto.get.assert_called_once()

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_silent_refusal_is_an_error_not_a_success(self, _mock_gx):
        """gxipy 对不可写特征只 print 噪声并保留旧值 —— 那必须报成失败。

        否则 ``auto_exposure`` 阶段 1 会拿**手动曝光**的帧当「SDK 已收敛」的结果,
        把曝光一路推到边界 (与 12-bit 满量程误判同型的静默错误)。
        """
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        mock_cam.ExposureAuto.get.return_value = (
            constants.GxAutoEntry.OFF.value,
            "Off",
        )
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam

        with pytest.raises(CameraError, match="ExposureAuto readback mismatch"):
            cam.enable_auto_exposure(enable=True)

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_unparseable_readback_is_an_error(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()  # ExposureAuto.get() → 裸 MagicMock
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam

        with pytest.raises(CameraError, match="ExposureAuto readback mismatch"):
            cam.enable_auto_exposure(enable=True)

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_missing_feature_still_reports_unsupported(self, _mock_gx):
        """「不支持」与「写了没生效」必须分开: 前者调用方可以退回比例迭代。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        mock_cam.ExposureAuto.set.side_effect = AttributeError("no such feature")
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam

        assert cam.enable_auto_exposure(enable=True) is False

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_disable_verifies_off(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        mock_cam.ExposureAuto.get.return_value = (
            constants.GxAutoEntry.OFF.value,
            "Off",
        )
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam

        assert cam.enable_auto_exposure(enable=False) is True
        mock_cam.ExposureAuto.set.assert_called_once_with(
            constants.GxAutoEntry.OFF.value
        )


class TestSetAutoExposureTargetVerifiesTheDevice:
    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_returns_the_effective_value_after_readback(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        mock_cam.ExpectedGrayValue.get.return_value = (180, "180")
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam

        assert cam.set_auto_exposure_target(180) == 180
        mock_cam.ExpectedGrayValue.set.assert_called_once_with(180)

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_clamped_request_returns_the_clamped_value(self, _mock_gx):
        """契约: 返回**钳制后的生效值**, 不是请求值。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        mock_cam.ExpectedGrayValue.get.return_value = (255, "255")
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam

        assert cam.set_auto_exposure_target(500) == 255
        mock_cam.ExpectedGrayValue.set.assert_called_once_with(255)

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_silent_refusal_is_an_error(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        mock_cam.ExpectedGrayValue.get.return_value = (120, "120")
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam

        with pytest.raises(CameraError, match="ExpectedGrayValue readback mismatch"):
            cam.set_auto_exposure_target(180)

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_missing_feature_still_returns_the_clamped_target(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        mock_cam.ExpectedGrayValue.set.side_effect = AttributeError("no such feature")
        cam = DahengCamera(cam_id=0)
        cam.cam = mock_cam

        assert cam.set_auto_exposure_target(-5) == 0


# ══════════════════════════════════════════════════════════════════════
# M2 — get_exposure_range 每次回读设备, 不返回陈旧缓存
# ══════════════════════════════════════════════════════════════════════


class TestParseExposureRange:
    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_converts_microseconds_to_milliseconds(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        assert DahengCamera._parse_exposure_range({"min": 20.0, "max": 100_000.0}) == (
            0.02,
            100.0,
        )

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_unusable_readback_is_none_not_a_bogus_number(self, _mock_gx):
        """裸 MagicMock 的 ``float()`` 会安静给出 1.0 ⇒ 必须用 isinstance 挡在门外。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        f = DahengCamera._parse_exposure_range
        assert f(MagicMock()) is None
        assert f(None) is None
        assert f({}) is None
        assert f({"min": 1000.0}) is None
        assert f({"min": "x", "max": "y"}) is None
        assert f({"min": 1000.0, "max": 10.0}) is None  # max <= min
        assert f({"min": -5.0, "max": 10.0}) is None


class TestGetExposureRangeReadsTheDevice:
    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_reflects_a_range_the_device_changed_since_initialize(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        mock_cam.ExposureTime.get_range.side_effect = [
            {"min": 20_000.0, "max": 100_000_000.0},
            {"min": 11_000.0, "max": 50_000_000.0},
        ]
        cam = DahengCamera(cam_id=0, exposure_time_ms=30.0)
        cam.cam = mock_cam

        assert cam.get_exposure_range() == (20.0, 100_000.0)
        assert cam.get_exposure_range() == (11.0, 50_000.0), "第二次必须回读, 不能吃缓存"

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", _fake_gx())
    def test_refreshed_range_is_what_reset_exposure_time_clamps_against(self):
        """缓存与设备脱节时, 一次合法的改曝光不该在写缓存时炸掉。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        fake = _FakeCam(
            "fake-frame-success",
            exposure_ms=20.0,
            exposure_range={"min": 11_000.0, "max": 50_000_000.0},
        )
        cam = _make_cam(fake, exposure_ms=20.0)

        assert cam.get_exposure_range() == (11.0, 50_000.0)
        assert cam.reset_exposure_time(45_000.0) == pytest.approx(45_000.0)

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_unreadable_range_falls_back_to_the_cache(self, _mock_gx):
        """读不到量程是降级不是致命错误: auto_exposure 每轮都调这个入口。"""
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()  # get_range() → 裸 MagicMock
        cam = DahengCamera(cam_id=0, exposure_time_ms=10.0)
        cam.cam = mock_cam

        lo, hi = cam.get_exposure_range()
        assert (lo, hi) == (0.0, 100_000.0)  # ExposureTime dataclass 的默认量程

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_requires_an_open_device(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        cam = DahengCamera(cam_id=0)
        assert cam.cam is None

        with pytest.raises(CameraError):
            cam.get_exposure_range()


# ══════════════════════════════════════════════════════════════════════
# M3 — 排空判定用设备回读的前后差, 不用本地缓存
# ══════════════════════════════════════════════════════════════════════


class TestDrainDecisionUsesTheDeviceReadback:
    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", _fake_gx())
    def test_a_stale_cache_can_no_longer_skip_the_drain(self):
        """本条就是 M3 的 bug 现场。

        设备真值 5 ms (例如 SDK 原生自动曝光刚把它推下去), 而本地缓存还停在 20 ms。
        请求 20 ms:
        * 旧判定 ``|请求 - 缓存| == 0`` ⇒ **不排空** ⇒ 调用方拿到的是 5 ms 的旧帧;
        * 新判定 ``|回读后 - 回读前| == 15 ms`` ⇒ 排空 3 帧 ⇒ 第一帧就是 20 ms。
        """
        fake = _FakeCam("fake-frame-success", exposure_ms=5.0, stale_frames=2)
        cam = _make_cam(fake, exposure_ms=20.0)  # 缓存 20 ms, 设备 5 ms
        assert cam.get_exposure_time_ms() == pytest.approx(5.0)

        assert cam.reset_exposure_time(20.0) == pytest.approx(20.0)

        assert fake.grabs == cam.EXPOSURE_CHANGE_DISCARD_FRAMES, "缓存没变也必须排空"
        img = cam.get_numpy_image(n_sample=1)
        assert img.max() == 20, "第一帧必须已经是新曝光"

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", _fake_gx())
    def test_a_real_no_op_still_consumes_nothing(self):
        """设备真值也没动 ⇒ 不排空 (别把 M3 修成"每次都丢 3 帧")。"""
        fake = _FakeCam("fake-frame-success", exposure_ms=20.0, stale_frames=3)
        cam = _make_cam(fake, exposure_ms=20.0)

        assert cam.reset_exposure_time(20.0) == pytest.approx(20.0)

        assert fake.grabs == 0

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", _fake_gx())
    def test_a_normal_change_drains_the_pipeline(self):
        fake = _FakeCam("fake-frame-success", exposure_ms=20.0, stale_frames=2)
        cam = _make_cam(fake, exposure_ms=20.0)

        assert cam.reset_exposure_time(2.0) == pytest.approx(2.0)

        assert fake.grabs == cam.EXPOSURE_CHANGE_DISCARD_FRAMES
        assert cam.get_numpy_image(n_sample=1).max() == 2

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", _fake_gx())
    def test_sub_microsecond_jitter_is_not_a_change(self):
        """阈值 0.5 µs = 设备整数 µs 网格的一半; 亚 µs 抖动不该触发 3 帧的代价。"""
        fake = _FakeCam("fake-frame-success", exposure_ms=20.0, stale_frames=2)
        cam = _make_cam(fake, exposure_ms=20.0)

        # 设备把 20.0004 ms 落成整数 µs = 20000 µs, 回读仍是 20.0
        assert cam.reset_exposure_time(20.0004) == pytest.approx(20.0)

        assert fake.grabs == 0

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_source_compares_the_device_not_the_cache(self, _mock_gx):
        """口径锁: 判定式里不许再出现「拿本地缓存 current 比请求值」那一套。"""
        import inspect

        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        src = inspect.getsource(DahengCamera.reset_exposure_time)
        assert "get_exposure_time_ms()" in src
        assert "__exposure_time_ms.current" not in src
        assert "EXPOSURE_CHANGE_DISCARD_FRAMES" in src  # 采集契约要求的字面量


# ══════════════════════════════════════════════════════════════════════
# M2b — 阶段 1 SDK 自动曝光: OFF 必须走 finally, 不许留在设备上
# ══════════════════════════════════════════════════════════════════════


class _FakeAutoFeature:
    """记录每次 ``set`` 的值; ``raises_on`` 命中时对该值写入抛错。"""

    def __init__(self, *, raises_on=None) -> None:
        self.values: list = []
        self._raises_on = raises_on

    def set(self, value) -> None:
        self.values.append(value)
        if self._raises_on is not None and value == self._raises_on:
            raise RuntimeError("SDK auto-exposure feature rejected the write")


def _sdk_auto_cam(*, raises_on=None):
    """挂上 ExposureAuto / ExpectedGrayValue 的假相机, 走通阶段 1 → 阶段 2。"""
    fake = _FakeCam(
        "fake-frame-success",
        exposure_ms=20.0,
        exposure_range={"min": 4_000.0, "max": 1_680_000.0},
    )
    exposure_auto = _FakeAutoFeature(raises_on=raises_on)
    fake.ExposureAuto = exposure_auto
    fake.ExpectedGrayValue = _FakeAutoFeature()
    return _make_cam(fake), fake, exposure_auto


class TestSdkAutoExposureDisableIsNeverSkipped:
    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", _fake_gx())
    def test_success_disables_once_exactly_once(self):
        """阶段 1 成功: ONCE → OFF, OFF 恰一次且在 ONCE 之后。"""
        cam, _fake, exposure_auto = _sdk_auto_cam()

        img = cam.auto_exposure(target_max=20.0, tolerance=5.0, max_iterations=3)

        once = constants.GxAutoEntry.ONCE.value
        off = constants.GxAutoEntry.OFF.value
        assert exposure_auto.values == [once, off]
        assert exposure_auto.values.count(off) == 1
        assert img.max() == 20  # 阶段 2 正常收尾

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", _fake_gx())
    def test_settle_failure_still_disables(self):
        """收敛循环中途抛错: 原异常照常外传 (阶段 2 再撞上), 但 OFF 已由 finally 下发。"""
        cam, _fake, exposure_auto = _sdk_auto_cam()

        with patch.object(
            cam,
            "_DahengCamera__take_one_shot",
            side_effect=RuntimeError("settle boom"),
        ):
            with pytest.raises(RuntimeError, match="settle boom"):
                cam.auto_exposure(target_max=20.0, tolerance=5.0, max_iterations=3)

        once = constants.GxAutoEntry.ONCE.value
        off = constants.GxAutoEntry.OFF.value
        assert exposure_auto.values == [once, off]
        assert exposure_auto.values.count(off) == 1

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", _fake_gx())
    def test_failed_enable_never_sends_off(self):
        """ONCE 本身就没开成功 ⇒ 不再发 OFF, 且阶段 2 照常返回图像。"""
        cam, _fake, exposure_auto = _sdk_auto_cam(
            raises_on=constants.GxAutoEntry.ONCE.value
        )

        img = cam.auto_exposure(target_max=20.0, tolerance=5.0, max_iterations=3)

        once = constants.GxAutoEntry.ONCE.value
        off = constants.GxAutoEntry.OFF.value
        assert once in exposure_auto.values
        assert off not in exposure_auto.values
        assert img.max() == 20
