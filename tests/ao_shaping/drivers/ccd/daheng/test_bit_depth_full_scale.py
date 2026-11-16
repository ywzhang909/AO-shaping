"""离线测试: 大恒高位置深 (12/16 bit) 的满量程折算必须按**生效位深**, 不按 dtype。

背景 (2026-11-16 实测):

大恒 MONO12 帧的 ``get_numpy_array()`` 返回 dtype 为 ``uint16`` 的数组, 但真实
满量程是 4095 而非 ``np.iinfo(uint16).max = 65535``。``common.full_scale`` 按
dtype 推导会得到 65535, ``normalize_peak_to_gray`` 把饱和帧峰值 4095 折成
``4095 * 255 / 65535 = 15.9`` 灰度 —— 自动曝光把饱和帧误判成「过暗」,
一路把曝光推到 1000 ms 上限, 实测日志 ``target brightness 180 unreachable
(peak 15.9); exposure forced to max`` 与该折算完全吻合。

修复: ``_full_scale`` / ``_auto_exposure_peak`` 已上移到 ``BaseCamera``, 大恒只负责
声明**生效位深** (``bit_depth`` 契约属性); ``get_numpy_image`` 给 ``AveragedFrame``
标注同量满量程。本文件用假相机离线锁住大恒侧这两条口径 (基类优先级与各后端映射的
完整契约见 ``tests/ao_shaping/drivers/ccd/test_full_scale_contract.py``)。
"""

import types

import numpy as np


def _fake_gx() -> types.SimpleNamespace:
    """假 SDK 片段: 驱动取帧路径只读 ``gx.GxFrameStatusList.SUCCESS``。"""
    return types.SimpleNamespace(
        GxFrameStatusList=types.SimpleNamespace(SUCCESS="fake-frame-success")
    )


class _FakeFrame:
    """一帧; 像素值直接给死, 用来模拟 12-bit 饱和帧 (峰值 4095)。"""

    def __init__(self, pixels: np.ndarray, success_token) -> None:
        self._pixels = pixels
        self._success = success_token

    def get_status(self):
        return self._success

    def get_numpy_array(self) -> np.ndarray:
        return self._pixels


class _FakeStream:
    def __init__(self, cam: "_FakeCam") -> None:
        self._cam = cam

    def get_image(self) -> _FakeFrame:
        return self._cam._pop_frame()


class _FakeCam:
    """固定像素的假相机: 每次取帧都返回同一帧。"""

    def __init__(self, pixels: np.ndarray, success_token) -> None:
        self.data_stream = [_FakeStream(self)]
        self._pixels = pixels
        self._success = success_token
        #: 最小 ExposureTime 特征: reset_exposure_time 会写它, get_exposure_time_ms 会读它。
        self.ExposureTime = _FakeExposureFeature(self, 20_000)

    def _pop_frame(self) -> _FakeFrame:
        return _FakeFrame(self._pixels, self._success)


class _FakeExposureFeature:
    """GenICam 的 ``ExposureTime`` 特征 (整数微秒)。"""

    def __init__(self, cam: "_FakeCam", value_us: int) -> None:
        self._cam = cam
        self._value_us = int(value_us)

    def set(self, value_us: int) -> None:
        self._value_us = int(value_us)

    def get(self) -> int:
        return self._value_us


def _make_cam(monkeypatch, pixels: np.ndarray, bit_depth: int = 12):
    """构造接上假相机的 DahengCamera, 并把生效位深设为 ``bit_depth``。"""
    from ao_shaping.drivers.ccd.daheng import driver as daheng_driver

    fake_gx = _fake_gx()
    monkeypatch.setattr(daheng_driver, "gx", fake_gx)
    cam = daheng_driver.DahengCamera(cam_id=0, exposure_time_ms=20.0)
    mock_cam = _FakeCam(pixels, fake_gx.GxFrameStatusList.SUCCESS)
    cam.cam = mock_cam
    # 假相机 = 已开流状态: __take_one_shot 的「停流不许取帧」把关以此标记为准
    # (停流时 get_image 永不返回 = 死等), 直接装 cam 的测试须显式声明流已开。
    cam._acquisition_on = True
    # 契约: bit_depth 只放设备回读真值; 这里直接模拟 open 后的生效位深。
    cam.bit_depth = bit_depth
    return cam, mock_cam


class TestBitDepthFullScale:
    def test_twelve_bit_saturated_frame_reads_full_gray(self, monkeypatch):
        """12-bit 饱和帧 (峰值 4095) 必须折成 255 灰度, 而不是 15.9。"""
        pixels = np.full((2, 2), 4095, dtype=np.uint16)
        cam, _ = _make_cam(monkeypatch, pixels, bit_depth=12)

        peak = cam._auto_exposure_peak(pixels)

        assert peak == 255.0, (
            f"12-bit 饱和帧折算错误: 期望 255 灰度, 实际 {peak} "
            "—— 按 dtype(uint16) 折算会把 4095 当成 65535 满量程"
        )

    def test_twelve_bit_mid_frame_rescales_correctly(self, monkeypatch):
        """12-bit 中等亮度帧 (峰值 2048) 折算: 2048 * 255 / 4095 ≈ 127.5。"""
        pixels = np.full((2, 2), 2048, dtype=np.uint16)
        cam, _ = _make_cam(monkeypatch, pixels, bit_depth=12)

        peak = cam._auto_exposure_peak(pixels)

        assert peak == 2048.0 * 255.0 / 4095.0

    def test_eight_bit_frame_untouched(self, monkeypatch):
        """8-bit 帧 (uint8) 满量程 255, 折算必须原样返回。"""
        pixels = np.full((2, 2), 200, dtype=np.uint8)
        cam, _ = _make_cam(monkeypatch, pixels, bit_depth=8)

        peak = cam._auto_exposure_peak(pixels)

        assert peak == 200.0

    def test_sixteen_bit_frame_uses_65535(self, monkeypatch):
        """16-bit 帧满量程 65535: 峰值 32768 → 128 灰度 (与 dtype 推导一致)。"""
        pixels = np.full((2, 2), 32768, dtype=np.uint16)
        cam, _ = _make_cam(monkeypatch, pixels, bit_depth=16)

        peak = cam._auto_exposure_peak(pixels)

        assert peak == 32768.0 * 255.0 / 65535.0

    def test_averaged_frame_carries_bit_depth_full_scale(self, monkeypatch):
        """多帧平均的 AveragedFrame 必须带生效位深的满量程, 而非 dtype 满量程。"""
        pixels = np.full((2, 2), 4095, dtype=np.uint16)
        cam, _ = _make_cam(monkeypatch, pixels, bit_depth=12)

        frame = cam.get_numpy_image(n_sample=3, skip_first=False)

        assert hasattr(frame, "full_scale")
        assert frame.full_scale == 4095.0, (
            f"AveragedFrame 满量程错误: 期望 4095 (12-bit), 实际 {frame.full_scale} "
            "—— 按 dtype(uint16) 会得到 65535, 饱和保护与峰值折算都会错量纲"
        )
        # 平均帧上的峰值折算与原始帧同口径。
        assert cam._auto_exposure_peak(frame) == 255.0

    def test_auto_exposure_converges_on_saturated_twelve_bit_frame(self, monkeypatch):
        """12-bit 饱和帧下, 自动曝光不得把曝光推到上限 (回归实测场景)。

        假相机恒定返回饱和帧: 修复前 peak=15.9 → 比例环判定「过暗」→
        曝光被推到 max; 修复后 peak=255 → 直接命中目标带, 曝光不动。
        """
        pixels = np.full((2, 2), 4095, dtype=np.uint16)
        cam, _ = _make_cam(monkeypatch, pixels, bit_depth=12)
        cam.get_exposure_range = lambda: (0.05, 1000.0)  # 类型: ignore[method-assign]

        img = cam.auto_exposure(
            target_max=180.0,
            tolerance=5.0,
            twice_valid=False,
            max_iterations=5,
            use_sdk_auto=False,
        )

        # 饱和帧折算成 255 灰度, 落在目标带 [175, 185] 之外但不再被误判「过暗」:
        # 比例环只做单步比例修正, 且曝光不会因 peak=15.9 的误判而冲向上限。
        assert img.max() == 4095
        assert cam.get_exposure_time_ms() <= 1000.0
