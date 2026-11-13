"""离线测试: 相机位深契约 ``bit_depth`` / ``supported_bit_depths`` / ``set_bit_depth``。

位深**不是**「设一次就永远成立」的参数: ``PixelFormat`` 是设备状态, 与 ROI、传输块大小
(见 ``test_frame_averaging.py``) 成对变化; 而 gxipy 的枚举特征 ``set`` 可能**静默保留旧值**
(同 ``IntFeature.set`` 的坑, 见 ccd/AGENTS.md §已知坑)。所以契约是

    写 PixelFormat → 回读设备真值 → 以回读为准 → 换档后旧格式帧不许流到调用方

与「改曝光后的第一帧就是新曝光」(``daheng/test_reset_exposure_time.py``) 同一条思路。
全程假 SDK, **不接相机**。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.drivers.ccd.daheng import constants
from ao_shaping.drivers.ccd.daheng import driver as daheng_driver
from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

# ── 假 SDK ───────────────────────────────────────────────


class _Feature:
    """GenICam 特征。``tuple_get=True`` 时 ``get()`` 返回 ``(值, 名称)`` (枚举特征的形态)。"""

    def __init__(self, value, *, name: str = "", tuple_get: bool = False):
        self.value = value
        self.name = name
        self.tuple_get = tuple_get
        self.set_calls: list[object] = []
        #: 设了 ``lies`` 就让 ``get()`` 撒谎: 返回它而不是刚写进去的值 (模拟 SDK 静默保留旧值)
        self.lies = None

    def set(self, value) -> None:
        self.set_calls.append(value)
        if self.lies is None:
            self.value = value

    def get(self):
        if self.lies is not None:
            return self.lies
        return (self.value, self.name) if self.tuple_get else self.value

    def get_range(self):
        return {"min": 0.0, "max": 1000.0}

    def __call__(self, *args, **kwargs):  # cam.stream_off() 这类无参方法
        return self.value


class _Stream:
    """``cam.data_stream[0]``: 一帧一帧地往外给, 并统计取了几帧。"""

    def __init__(self, success_token, buffer_handling_mode, acquisition_mode):
        self.AcquisitionMode = _Feature(
            acquisition_mode, name="SingleFrame", tuple_get=True
        )
        self.StreamBufferHandlingMode = _Feature(
            buffer_handling_mode, name="NewestOnly", tuple_get=True
        )
        self._success = success_token
        self.grabs = 0
        self.start_calls = 0
        self.stop_calls = 0

    def get_image(self):
        self.grabs += 1
        return _Frame(self._success)

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


class _Frame:
    def __init__(self, success_token) -> None:
        self._success = success_token

    def get_status(self):
        return self._success

    def get_numpy_array(self):
        return np.zeros((4, 4), dtype=np.uint8)

    def close(self) -> None:  # pragma: no cover - 驱动不调用, 留着防资源告警
        pass


class _FakeCam:
    """够驱动 ``initialize()`` 与 ``set_bit_depth()`` 跑完的最小大恒设备。"""

    _NEWEST = constants.GxDSStreamBufferHandlingModeEntry.NEWEST_ONLY.value
    _SINGLE = constants.GxAcquisitionModeEntry.SINGLE_FRAME.value
    _SUCCESS = constants.GxFrameStatusList.SUCCESS

    def __init__(self, pixel_format: str = "MONO8", *, exposure_ms: float = 20.0) -> None:
        self.sn = "FJB24112232"
        self.model = "FAKE-MERGE8"
        self.ExposureTime = _Feature(int(exposure_ms * 1000), name="ExposureTime")
        self.PixelFormat = _Feature(
            getattr(constants.GxPixelFormatEntry, pixel_format),
            name=pixel_format,
            tuple_get=True,
        )
        self.Width = _Feature(4, name="Width")
        self.Height = _Feature(4, name="Height")
        self.WidthMax = _Feature(4, name="WidthMax")
        self.HeightMax = _Feature(4, name="HeightMax")
        self.DeviceModelName = _Feature(self.model, name="DeviceModelName")
        self.SensorName = _Feature(self.model, name="SensorName")
        self.data_stream = [_Stream(self._SUCCESS, self._NEWEST, self._SINGLE)]

    def __getattr__(self, name: str) -> _Feature:
        # 初始化还会写 Gain / Binning / Offset / TriggerMode / BlackLevel 等一大堆特征;
        # 位深测试不关心它们的语义, 统一给一个可记 set 的桩, 免得每碰一次 SDK 就崩一次。
        return _Feature(0, name=name)


def _fake_gx():
    return SimpleNamespace(
        GxFrameStatusList=constants.GxFrameStatusList,
        GxPixelFormatEntry=constants.GxPixelFormatEntry,
        GxAcquisitionModeEntry=constants.GxAcquisitionModeEntry,
        GxDSStreamBufferHandlingModeEntry=constants.GxDSStreamBufferHandlingModeEntry,
        GxAutoEntry=constants.GxAutoEntry,
    )


def _make_camera(fake: _FakeCam, monkeypatch, *, bit_depth: int = 8) -> DahengCamera:
    """造一台**未 open** 的大恒相机: SDK 换成假设备, 不碰磁盘/配置。"""
    monkeypatch.setattr(daheng_driver, "gx", _fake_gx())
    monkeypatch.setattr(DahengCamera, "load_config", lambda self: {})
    monkeypatch.setattr(
        type(daheng_driver.CCD_CONFIG),
        "apply_from_config",
        lambda self, instance, config, init_values=None: None,
    )
    cam = DahengCamera(bit_depth=bit_depth)
    cam.cam = fake
    cam.device_manager = SimpleNamespace(close_device=lambda device: None)
    return cam


# ── 基类契约 ─────────────────────────────────────────────


class _NoBitDepthCamera(BaseCamera):
    """没有登记 :attr:`BaseCamera.supported_bit_depths` 的后端 (ffmpeg / image_folder 那一类)。

    基类必须替它把「换位深」判成**失败**, 而不是静默当成已经换了 —— 调用方拿到的
    ``bit_depth`` 才不会是虚报值。方法集照 ``test_ensure_open._NullCamera`` 给最小实现。
    """

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def initialize(self) -> None:
        pass

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def reset_exposure_time(self, time_ms: float) -> float:
        return float(time_ms)

    def reset_window(self, center, size):
        return (0, 0), (0, 0)

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True):
        raise NotImplementedError

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        return False

    def get_auto_exposure_state(self) -> dict:
        return {}

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        return False

    @staticmethod
    def get_cam_list():
        return []


# ── 基类契约 ─────────────────────────────────────────────


def test_base_camera_declares_bit_depth_contract():
    cam = _NoBitDepthCamera()
    assert cam.supported_bit_depths == ()
    assert cam.bit_depth is None

    with pytest.raises(CameraError, match="does not support bit depth selection"):
        cam.set_bit_depth(12)


def test_daheng_supported_bit_depths_matches_format_map():
    # 档位表与 PixelFormat 映射同源: 加一档只改 _BIT_DEPTH_PIXEL_FORMATS 就够
    assert DahengCamera.supported_bit_depths == tuple(DahengCamera._BIT_DEPTH_PIXEL_FORMATS)
    assert DahengCamera.supported_bit_depths == (8, 12, 16)
    assert DahengCamera._BIT_DEPTH_PIXEL_FORMATS == {8: "MONO8", 12: "MONO12", 16: "MONO16"}


# ── set_bit_depth: 写 + 回读 ─────────────────────────────


@pytest.mark.parametrize("bits", [8, 12, 16])
def test_daheng_set_bit_depth_writes_expected_pixel_format(monkeypatch, bits):
    fake = _FakeCam()
    cam = _make_camera(fake, monkeypatch)
    cam._acquisition_on = True

    assert cam.set_bit_depth(bits) == bits
    assert fake.PixelFormat.set_calls == [getattr(constants.GxPixelFormatEntry, f"MONO{bits}")]
    assert cam.bit_depth == bits
    # 换档 (8 → bits) 后旧格式帧必须被消费掉
    expected_grabs = 0 if bits == 8 else DahengCamera.EXPOSURE_CHANGE_DISCARD_FRAMES
    assert fake.data_stream[0].grabs == expected_grabs


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("MONO12", 12),
        ("Mono12", 12),
        (constants.GxPixelFormatEntry.MONO12.value, 12),
        ((constants.GxPixelFormatEntry.MONO12, "Mono12"), 12),
        (constants.GxPixelFormatEntry.MONO12, 12),
        ((1, "BayerGB12"), None),
        ("Mono", None),
        (None, None),
    ],
)
def test_bit_depth_of_format_is_tolerant_to_readback_shape(raw, expected):
    # SDK 的枚举 get() 形态不统一 (串 / PFNC 码 / (码, 名) / 枚举成员), 回读都要能认
    assert DahengCamera._bit_depth_of_format(raw) == expected


def test_daheng_unsupported_bit_depth_raises_without_touching_sdk(monkeypatch):
    fake = _FakeCam()
    cam = _make_camera(fake, monkeypatch)
    cam._acquisition_on = True

    with pytest.raises(ValueError, match="8, 12 or 16"):
        cam.set_bit_depth(10)

    assert cam.bit_depth == 8  # 请求被拒, 缓存不动
    assert fake.PixelFormat.set_calls == []
    assert fake.data_stream[0].grabs == 0


def test_daheng_bit_depth_readback_mismatch_raises(monkeypatch):
    """SDK 静默保留旧格式时**必须抛错** (同 PixelFormat 坑), 而不是骗调用方换好了。"""
    fake = _FakeCam()
    cam = _make_camera(fake, monkeypatch)
    cam.bit_depth = 12
    fake.PixelFormat.lies = (constants.GxPixelFormatEntry.MONO8, "Mono8")

    with pytest.raises(CameraError, match="PixelFormat readback mismatch"):
        cam.set_bit_depth(12)


def test_daheng_stores_device_truth_even_when_readback_mismatches(monkeypatch):
    """抛错之前先记下**设备真值**: 缓存里绝不留一个设备并未使用的档位。"""
    fake = _FakeCam()
    cam = _make_camera(fake, monkeypatch)
    cam.bit_depth = 12
    fake.PixelFormat.lies = (constants.GxPixelFormatEntry.MONO8, "Mono8")

    with pytest.raises(CameraError):
        cam.set_bit_depth(12)

    assert cam.bit_depth == 8


# ── 换档后不许流到调用方旧格式帧 ────────────────────────


def test_daheng_discards_frames_after_real_bit_depth_change(monkeypatch):
    fake = _FakeCam(pixel_format="MONO8")
    cam = _make_camera(fake, monkeypatch)
    cam.bit_depth = 8
    cam._acquisition_on = True

    assert cam.set_bit_depth(16) == 16

    assert fake.data_stream[0].grabs == DahengCamera.EXPOSURE_CHANGE_DISCARD_FRAMES


def test_daheng_does_not_discard_frames_when_bit_depth_unchanged(monkeypatch):
    """没真的换档就别白花 3 帧采集时间 (与 reset_exposure_time 同一口径)。"""
    fake = _FakeCam(pixel_format="MONO12")
    cam = _make_camera(fake, monkeypatch)
    cam.bit_depth = 12
    cam._acquisition_on = True

    assert cam.set_bit_depth(12) == 12

    assert fake.data_stream[0].grabs == 0


def test_daheng_never_fetches_frames_while_acquisition_off(monkeypatch):
    """停流时 ``get_image`` 永不返回 ⇒ 死等。开流前换位深只写格式, 不取帧。"""
    fake = _FakeCam(pixel_format="MONO8")
    cam = _make_camera(fake, monkeypatch)
    cam.bit_depth = 8
    cam._acquisition_on = False

    assert cam.set_bit_depth(12) == 12
    assert fake.data_stream[0].grabs == 0

    cam._acquisition_on = True
    assert cam.set_bit_depth(16) == 16
    assert fake.data_stream[0].grabs == DahengCamera.EXPOSURE_CHANGE_DISCARD_FRAMES


# ── 与 open()/initialize() 的衔接 ────────────────────────


def test_daheng_initialize_applies_configured_bit_depth_before_stream_on(monkeypatch):
    fake = _FakeCam(pixel_format="MONO8")
    monkeypatch.setattr(daheng_driver, "gx", _fake_gx())
    monkeypatch.setattr(DahengCamera, "load_config", lambda self: {})
    monkeypatch.setattr(
        type(daheng_driver.CCD_CONFIG),
        "apply_from_config",
        lambda self, instance, config, init_values=None: None,
    )

    device_manager = SimpleNamespace(
        update_device_list=lambda: (1, [{"sn": fake.sn, "model": fake.model}]),
        open_device_by_sn=lambda sn: fake,
        close_device=lambda device: None,
    )

    cam = DahengCamera(bit_depth=12)
    assert cam.bit_depth == 12  # 构造期记的是请求值, 尚未下发
    cam.device_manager = device_manager
    fake.PixelFormat.set_calls.clear()

    cam.open()

    # 配置档位经 set_bit_depth 落到 PixelFormat, 而不是另开一条写格式的路
    assert fake.PixelFormat.set_calls == [constants.GxPixelFormatEntry.MONO12]
    assert cam.bit_depth == 12
    assert cam._acquisition_on is True  # stream_on 在 initialize 末尾, 位深必须更早已写好
    # stream_on 之前取流未开 ⇒ 应用配置位深时一帧都不许拿 (未开流取帧就是死等)
    assert fake.data_stream[0].grabs == 0

    fake.PixelFormat.set_calls.clear()
    assert cam.set_bit_depth(8) == 8
    assert fake.data_stream[0].grabs == DahengCamera.EXPOSURE_CHANGE_DISCARD_FRAMES

    cam.close()
    assert cam._acquisition_on is False


def test_daheng_close_clears_acquisition_state(monkeypatch):
    """关流后 ``_acquisition_on`` 必须跟着收敛, 且设备已释放 —— 换位深只能报错, 不能死等取帧。"""
    fake = _FakeCam(pixel_format="MONO12")
    cam = _make_camera(fake, monkeypatch)
    cam._acquisition_on = True
    cam.bit_depth = 12

    cam.close()

    assert cam._acquisition_on is False
    with pytest.raises(CameraError):
        cam.set_bit_depth(8)
    assert fake.data_stream[0].grabs == 0
