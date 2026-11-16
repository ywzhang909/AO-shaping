"""离线测试: 满量程契约 —— 绑**生效位深**, 不绑帧 dtype (issue #246)。

背景 (2026-11-16 实测 FJB24112232):

大恒 MONO12 帧的 ``get_numpy_array()`` 返回 dtype 为 ``uint16`` 的数组, 但真实满量程
是 4095 而非 ``np.iinfo(uint16).max = 65535``; MiiCam 高位深模式同样是「12/14 bit 装进
uint16 容器」。按 dtype 推导满量程于是得到 65535, 饱和帧峰值 4095 被
``normalize_peak_to_gray`` 折成 ``4095 * 255 / 65535 = 15.9`` 灰度 —— 自动曝光把饱和帧
误判成「过暗」, 一路把曝光推到 1000 ms 上限。同源的还有饱和保护 (``is_saturated``)、
多帧融合的 ``detector_full_scale``、HDR 阶梯。

本文件锁住**取值优先级**这一条口径 (权威实现 ``BaseCamera._full_scale`` /
``effective_full_scale``):

    1. 帧上的显式标注 ``AveragedFrame.full_scale`` (采集时刻写下的实测饱和值);
    2. 相机实例的生效位深 ``bit_depth`` (设备回读真值) → ``(1 << bits) - 1``;
    3. ``source_dtype`` 的整数上限 —— **仅**在位深未知时兜底 (ffmpeg / image_folder);
    4. ``common.DETECTOR_FULL_SCALE`` (255.0) —— 非整数帧且无相机信息时。

以及各后端「谁产帧谁标注」的映射, 和 ``capture_hdr`` / ``capture_fused`` 两条多帧
融合路径都经 ``effective_full_scale`` 取满量程。

全程假 SDK / 假相机, **不接硬件**。物理锁定见 ``test_full_scale_hardware.py``,
大恒 12-bit 回归另见 ``daheng/test_bit_depth_full_scale.py``。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ao_shaping.drivers.ccd import common
from ao_shaping.drivers.ccd.base import AveragedFrame, BaseCamera

# ── 假相机 ───────────────────────────────────────────────


class _StubCamera(BaseCamera):
    """最小后端桩: 只实现抽象方法, 位深由测试设定。

    方法集照 ``test_capture_hdr.FakeCamera`` / ``test_camera_bit_depth._NoBitDepthCamera``
    给最小实现, 因此 ``capture_hdr`` / ``capture_fused`` 也能直接跑。
    """

    def __init__(self, *, bit_depth: int | None = None, exposure_time_ms: float = 4.0) -> None:
        super().__init__(cam_id=0, exposure_time_ms=exposure_time_ms)
        #: 契约属性: 设备回读的**生效**位深, None = 未声明 (ffmpeg / image_folder 那一类)。
        self.bit_depth = bit_depth

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def initialize(self) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def reset_exposure_time(self, time_ms: float) -> float:
        self.exposure_time_ms = max(float(time_ms), 1e-6)
        return self.exposure_time_ms

    def reset_window(self, *args, **kwargs):
        return None

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        return np.zeros((4, 4), dtype=np.uint8)

    def get_exposure_time_ms(self) -> float:
        return float(self.exposure_time_ms)

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        return False

    def get_auto_exposure_state(self) -> dict:
        return {"exposure_time_ms": self.exposure_time_ms}

    def set_auto_exposure_range(self, *args, **kwargs):
        return None

    @staticmethod
    def get_cam_list():
        return []


def _saturated_twelve_bit_frame(peak: int = 4095) -> np.ndarray:
    """一帧「12-bit 计数装进 uint16 容器」的画面 —— #246 的缺陷入口。"""
    return np.full((8, 8), peak, dtype=np.uint16)


# ── 1. _full_scale 的取值优先级 ──────────────────────────


@pytest.mark.parametrize(
    "bits,expected",
    [(8, 255.0), (10, 1023.0), (12, 4095.0), (14, 16383.0), (16, 65535.0)],
)
def test_full_scale_follows_the_declared_bit_depth(bits: int, expected: float) -> None:
    """第 2 档: 位深已知时 ``(1 << bits) - 1`` 就是唯一答案。"""
    assert _StubCamera(bit_depth=bits)._full_scale() == expected


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.dtype(np.uint16)])
def test_bit_depth_beats_the_frame_dtype(dtype) -> None:
    """第 2 档压过第 3 档: MONO12 帧的 uint16 容器不许把满量程抬到 65535。"""
    assert _StubCamera(bit_depth=12)._full_scale(dtype) == 4095.0


@pytest.mark.parametrize("dtype,expected", [(np.uint8, 255.0), (np.uint16, 65535.0)])
def test_dtype_is_only_a_fallback_while_the_bit_depth_is_unknown(dtype, expected: float) -> None:
    """第 3 档: 回放后端没有位深真值, 才允许按 dtype 兜底。"""
    assert _StubCamera(bit_depth=None)._full_scale(dtype) == expected


def test_non_integer_dtype_falls_through_to_the_detector_default() -> None:
    """第 4 档: 多帧均值恒为 float32, ``np.iinfo`` 抛 ValueError 时必须落到 255 而不是崩。"""
    cam = _StubCamera(bit_depth=None)
    assert cam._full_scale(np.float32) == float(common.DETECTOR_FULL_SCALE)
    assert cam._full_scale(np.float64) == float(common.DETECTOR_FULL_SCALE)
    # 位深已知时同样不看 dtype: 12-bit 后端的 float32 均值帧仍是 4095。
    assert _StubCamera(bit_depth=12)._full_scale(np.float32) == 4095.0


def test_full_scale_is_a_plain_float() -> None:
    """满量程参与 ``* 255 / scale`` 折算, 返回 numpy 标量会让量纲检查静默失效。"""
    value = _StubCamera(bit_depth=12)._full_scale(np.uint16)
    assert type(value) is float


# ── 2. effective_full_scale: 帧标注优先 ──────────────────


def test_averaged_frame_defaults_to_the_source_dtype_when_unannotated() -> None:
    """不传 ``full_scale`` 时按 source dtype 推导 —— 这是**兜底**, 后端必须显式标注。"""
    assert AveragedFrame(np.zeros((2, 2)), np.dtype(np.uint16)).full_scale == 65535.0
    assert AveragedFrame(np.zeros((2, 2)), np.dtype(np.uint8)).full_scale == 255.0


def test_averaged_frame_keeps_an_explicit_annotation() -> None:
    frame = AveragedFrame(np.zeros((2, 2)), np.dtype(np.uint16), full_scale=4095.0)
    assert frame.full_scale == 4095.0
    assert frame.dtype == np.float32


def test_frame_annotation_wins_over_the_camera_bit_depth() -> None:
    """第 1 档压过第 2 档: 标注是采集时刻写下的实测饱和值, 比位深推算更可信。

    换位深 / 多档混合的融合图无法用单一位深表达, 这条优先级是它唯一能自证量纲的途径。
    """
    cam = _StubCamera(bit_depth=12)
    frame = AveragedFrame(np.zeros((2, 2)), np.dtype(np.uint16), full_scale=1023.0)
    assert cam.effective_full_scale(frame) == 1023.0


def test_bare_frame_falls_back_to_the_camera_bit_depth_not_its_dtype() -> None:
    """单帧路径没有 ``AveragedFrame`` 标注, 判据必须经**相机实例**取满量程。"""
    cam = _StubCamera(bit_depth=12)
    assert cam.effective_full_scale(_saturated_twelve_bit_frame()) == 4095.0


def test_annotation_survives_slicing_and_arithmetic() -> None:
    """``__array_finalize__`` 必须把标注带过 view / ufunc, 否则 ROI 裁剪后量纲丢失。"""
    frame = AveragedFrame(np.zeros((4, 4)), np.dtype(np.uint16), full_scale=4095.0)
    assert frame[1:3, 1:3].full_scale == 4095.0
    assert (frame * 2).full_scale == 4095.0


# ── 3. _auto_exposure_peak 的 12-bit 回归 ────────────────


def test_auto_exposure_peak_reads_a_clipped_twelve_bit_frame_as_bright() -> None:
    """#246 的正例: 峰值 4095 的 12-bit 饱和帧折算到 0-255 必须是 **255**, 不是 15.9。"""
    cam = _StubCamera(bit_depth=12)
    assert cam._auto_exposure_peak(_saturated_twelve_bit_frame()) == pytest.approx(255.0)


def test_auto_exposure_peak_control_the_dtype_guess_reports_15_9() -> None:
    """对照: 按 dtype 猜 (65535) 得到 15.9 灰度 —— 自动曝光据此判「过暗」并把曝光推到上限。

    这一条**故意**断言错误值, 用来钉住回归的量级: 修好之后两者差 16 倍。
    """
    img = _saturated_twelve_bit_frame()
    wrong = common.normalize_peak_to_gray(img, full_scale=65535.0)
    assert wrong == pytest.approx(15.9, abs=0.05)

    fixed = _StubCamera(bit_depth=12)._auto_exposure_peak(img)
    assert fixed == pytest.approx(255.0)
    assert fixed / wrong == pytest.approx(65535.0 / 4095.0, rel=1e-6)


@pytest.mark.parametrize(
    "bits,counts,expected",
    [
        (8, 200, 200.0),                      # 8-bit: 满量程 255 不折算, 原样透传
        (12, 2048, 2048 * 255 / 4095),        # 12-bit 中灰 → 8-bit 中灰
        (14, 8192, 8192 * 255 / 16383),
        (16, 32768, 32768 * 255 / 65535),
    ],
)
def test_auto_exposure_peak_maps_every_bit_depth_onto_the_same_gray_scale(
    bits: int, counts: int, expected: float
) -> None:
    """``target_max=40`` 在 8/12/14/16-bit 后端上必须是同一个亮度 (0-255 灰度)。"""
    dtype = np.uint8 if bits == 8 else np.uint16
    cam = _StubCamera(bit_depth=bits)
    assert cam._auto_exposure_peak(np.full((4, 4), counts, dtype)) == pytest.approx(
        expected, rel=1e-6
    )


def test_auto_exposure_peak_uses_the_frame_annotation_when_present() -> None:
    """多帧平均帧 (float32) 靠自身标注定标, 不看 dtype 也不看相机位深。"""
    cam = _StubCamera(bit_depth=None)
    frame = AveragedFrame(np.full((4, 4), 4095.0), np.dtype(np.uint16), full_scale=4095.0)
    assert cam._auto_exposure_peak(frame) == pytest.approx(255.0)


# ── 4. common.full_scale / is_saturated 的 camera 参数 ───


def test_common_full_scale_prefers_the_frame_annotation() -> None:
    cam = _StubCamera(bit_depth=12)
    frame = AveragedFrame(np.zeros((4, 4)), np.dtype(np.uint16), full_scale=65535.0)
    assert common.full_scale(frame, cam) == 65535.0
    assert common.full_scale(frame) == 65535.0


def test_common_full_scale_uses_the_camera_bit_depth_for_bare_arrays() -> None:
    cam = _StubCamera(bit_depth=12)
    assert common.full_scale(_saturated_twelve_bit_frame(), cam) == 4095.0


def test_common_full_scale_without_a_camera_can_only_guess_from_the_dtype() -> None:
    """没有相机上下文时只能落到 dtype 兜底 —— 这正是调用方**必须传 camera** 的理由。"""
    assert common.full_scale(_saturated_twelve_bit_frame()) == 65535.0


def test_is_saturated_sees_twelve_bit_clipping_only_with_the_camera() -> None:
    """回归: 峰值 4095 的 12-bit 饱和帧, 不传 camera 时被误判成「未饱和」。"""
    img = _saturated_twelve_bit_frame()
    assert common.is_saturated(img, _StubCamera(bit_depth=12)) is True
    # ← #246 的缺陷形态: 饱和保护静默失效, 削平帧照常进入下游统计。
    assert common.is_saturated(img) is False


def test_is_saturated_is_unchanged_for_eight_bit_frames() -> None:
    """8-bit 路径不受本次改动影响: 传不传 camera 结论一致。"""
    cam = _StubCamera(bit_depth=8)
    clipped = np.full((4, 4), 255, np.uint8)
    bright = np.full((4, 4), 250, np.uint8)
    assert common.is_saturated(clipped, cam) is True
    assert common.is_saturated(clipped) is True
    assert common.is_saturated(bright, cam) is False
    assert common.is_saturated(bright) is False


# ── 5. resample_on_saturation 的触发 ─────────────────────


class _RecExposure:
    """记下被调了几次、拿到什么参数, 并返回一张可辨认的「重采帧」。"""

    def __init__(self) -> None:
        self.calls: list[tuple[object, float, int]] = []

    def __call__(self, cam, target, n_sample: int = 1) -> np.ndarray:
        self.calls.append((cam, float(target), int(n_sample)))
        return np.zeros((8, 8), dtype=np.uint16)


def test_resample_on_saturation_triggers_on_a_clipped_twelve_bit_frame() -> None:
    """自动曝光模式 (``exposure_time_ms == 0``) 下, 峰值 4095 的 12-bit 帧必须被判成饱和。"""
    cam = _StubCamera(bit_depth=12)
    rec = _RecExposure()
    out = common.resample_on_saturation(
        _saturated_twelve_bit_frame(), cam, auto_exposure_fn=rec
    )
    assert len(rec.calls) == 1
    np.testing.assert_array_equal(out, np.zeros((8, 8), dtype=np.uint16))


def test_resample_on_saturation_threshold_defaults_to_the_camera_bit_depth() -> None:
    """缺省阈值经 ``full_scale(img, cam)`` 取 —— 传了 cam 才拿得到 4095 而不是 65535。"""
    cam = _StubCamera(bit_depth=12)
    rec = _RecExposure()
    common.resample_on_saturation(_saturated_twelve_bit_frame(), cam, auto_exposure_fn=rec)
    assert rec.calls[0][0] is cam
    assert rec.calls[0][1] == pytest.approx(220.0)  # target_max_brightness=0 → 220


def test_resample_on_saturation_leaves_an_unclipped_twelve_bit_frame_alone() -> None:
    """4000 < 4095: 接近满量程但未削平, 不许触发重采 (否则每次采集都白跑一遍自动曝光)。"""
    cam = _StubCamera(bit_depth=12)
    rec = _RecExposure()
    img = np.full((4, 4), 4000, np.uint16)
    out = common.resample_on_saturation(img, cam, auto_exposure_fn=rec)
    assert rec.calls == []
    np.testing.assert_array_equal(out, img)


def test_resample_on_saturation_needs_the_camera_to_see_twelve_bit_clipping() -> None:
    """同一帧交给没有位深真值的相机 (回放后端形态) 就漏判 —— 守卫的强度取决于 camera。"""
    rec = _RecExposure()
    img = _saturated_twelve_bit_frame()
    out = common.resample_on_saturation(img, _StubCamera(bit_depth=None), auto_exposure_fn=rec)
    assert rec.calls == []  # 4095 < 65535, 按 dtype 兜底就看不见削平
    np.testing.assert_array_equal(out, img)


def test_resample_on_saturation_skips_a_fixed_exposure() -> None:
    """``exposure_time_ms > 0`` = 调用方钉死了曝光, 守卫直接放行, 不重采。"""
    cam = _StubCamera(bit_depth=12)
    rec = _RecExposure()
    img = _saturated_twelve_bit_frame()
    out = common.resample_on_saturation(img, cam, exposure_time_ms=5.0, auto_exposure_fn=rec)
    assert rec.calls == []
    np.testing.assert_array_equal(out, img)


def test_resample_on_saturation_honours_an_explicit_threshold_and_target() -> None:
    """显式阈值/目标仍然可用 (例如想提前触发); n_sample 透传给自动曝光。"""
    cam = _StubCamera(bit_depth=12)
    rec = _RecExposure()
    common.resample_on_saturation(
        np.full((4, 4), 3000, np.uint16),
        cam,
        target_max_brightness=120.0,
        auto_exposure_fn=rec,
        saturation_threshold=2500.0,
        n_sample=4,
    )
    assert rec.calls == [(cam, 120.0, 4)]


# ── 6. 各后端的位深 → 满量程映射 ─────────────────────────


def test_daheng_declares_the_tiers_it_can_switch_to() -> None:
    from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

    assert DahengCamera.supported_bit_depths == (8, 12, 16)


def test_daheng_inherits_the_full_scale_helpers_from_the_base_class() -> None:
    """``_full_scale`` / ``_auto_exposure_peak`` 已上移到基类: 大恒只负责声明位深。

    钉住这一点是为了防止后端再各自复刻一份折算 —— 那份复刻正是 #246 的温床。
    """
    from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

    assert "_full_scale" not in DahengCamera.__dict__
    assert "_auto_exposure_peak" not in DahengCamera.__dict__
    assert DahengCamera._full_scale is BaseCamera._full_scale
    assert DahengCamera._auto_exposure_peak is BaseCamera._auto_exposure_peak


@pytest.mark.parametrize("bits,pixel_max", [(8, 255), (12, 4095), (16, 65535)])
def test_daheng_annotates_its_average_frame_with_the_effective_full_scale(
    monkeypatch: pytest.MonkeyPatch, bits: int, pixel_max: int
) -> None:
    """谁产帧谁标注: 12-bit 帧的 dtype 是 uint16, 标注必须是 4095 而不是 65535。"""
    from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

    cam = object.__new__(DahengCamera)
    cam.bit_depth = bits
    dtype = np.dtype(np.uint8 if bits == 8 else np.uint16)
    pixels = np.full((4, 4), pixel_max, dtype)
    monkeypatch.setattr(
        DahengCamera, "_DahengCamera__take_one_shot", lambda self: pixels.copy()
    )

    frame = cam.get_numpy_image(n_sample=3, skip_first=False)
    assert isinstance(frame, AveragedFrame)
    assert frame.full_scale == float(pixel_max)
    assert cam.effective_full_scale(frame) == float(pixel_max)
    # 饱和帧经标注后判据自洽: is_saturated 不需要 camera 也能看出来。
    assert common.is_saturated(frame) is True


@pytest.mark.parametrize(
    "requested,max_reported,expected",
    [
        (8, 12, 8),    # 请求 8-bit → 低位深输出 → 8
        (16, 12, 12),  # 高位深输出 → 取 MaxBitDepth() 的**回读真值**
        (16, 14, 14),  # 本机 MiiCam 实测报 MONO16 而传感器是 14-bit 的同一形态
        (16, 8, 8),    # 设备不支持高位深 → 回落 8, 不留虚报档位
        (16, 32, 16),  # 回读值不在 9..16 → 退回请求值
    ],
)
def test_miicam_bit_depth_is_the_device_readback(
    requested: int, max_reported: int, expected: int
) -> None:
    """``_init_bit_depth`` 必须把回读真值写进契约属性 ``bit_depth``, 否则满量程只能按 dtype 猜。"""
    from ao_shaping.drivers.ccd.miicam import driver as miicam_driver

    if miicam_driver.miicam is None:  # pragma: no cover - SDK 缺失的环境
        pytest.skip("MiiCam SDK 不可用")

    options: list[tuple[object, object]] = []
    cam = object.__new__(miicam_driver.MIICamera)
    cam._bit_depth = requested
    cam.cam = SimpleNamespace(
        MaxBitDepth=lambda: max_reported,
        put_Option=lambda key, value: options.append((key, value)),
    )

    cam._init_bit_depth()

    assert cam.bit_depth == expected
    assert cam._full_scale(np.uint16) == float((1 << expected) - 1)


@pytest.mark.parametrize(
    "bits,pixel_max", [(8, 255), (12, 4095), (14, 16383), (16, 65535)
]
)
def test_miicam_annotates_its_average_frame_with_the_effective_full_scale(
    monkeypatch: pytest.MonkeyPatch, bits: int, pixel_max: int
) -> None:
    from ao_shaping.drivers.ccd.miicam.driver import MIICamera

    cam = object.__new__(MIICamera)
    cam.bit_depth = bits
    dtype = np.dtype(np.uint8 if bits == 8 else np.uint16)
    pixels = np.full((4, 4), pixel_max, dtype)
    monkeypatch.setattr(MIICamera, "_MIICamera__take_one_shot", lambda self: pixels.copy())

    frame = cam.get_numpy_image(n_sample=2, skip_first=False)
    assert isinstance(frame, AveragedFrame)
    assert frame.full_scale == float(pixel_max)
    assert cam.effective_full_scale(frame) == float(pixel_max)


@pytest.mark.parametrize(
    "backend",
    ["ffmpeg", "image_folder"],
)
def test_playback_backends_declare_no_bit_depth_truth(backend: str) -> None:
    """回放源没有位深真值: ``bit_depth`` 恒 None、不支持切档, 满量程只能按 dtype 兜底。"""
    from ao_shaping.drivers.ccd.ffmpeg.driver import FFmpegCamera, ImageFolderCamera

    cls = FFmpegCamera if backend == "ffmpeg" else ImageFolderCamera
    assert cls.supported_bit_depths == ()

    cam = object.__new__(cls)
    assert cam.bit_depth is None
    assert cam._full_scale(np.uint8) == 255.0
    assert cam._full_scale(np.uint16) == 65535.0
    assert cam._full_scale(np.float32) == float(common.DETECTOR_FULL_SCALE)


def test_ffmpeg_still_annotates_its_average_frame_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """位深未知的后端也要**显式**标注 —— 让「按 dtype」成为被声明的选择而非隐式默认。"""
    from ao_shaping.drivers.ccd.ffmpeg.driver import FFmpegCamera

    # get_numpy_image 内部 `import cv2`; 桩掉以免测试依赖视频后端是否装上。
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())
    monkeypatch.setattr(FFmpegCamera, "_ensure_open", lambda self, detail="": None)

    pixels = np.full((4, 4), 255, np.uint8)
    cam = object.__new__(FFmpegCamera)
    cam.bit_depth = None
    cam._cap = SimpleNamespace(read=lambda: (True, pixels.copy()))
    cam._frame_count = 0
    cam._is_file = False
    cam._total_frames = 0
    cam.exposure_time_ms = 0.0
    cam.skip_sampling = False

    frame = cam.get_numpy_image(n_sample=2, skip_first=False)
    assert isinstance(frame, AveragedFrame)
    assert frame.full_scale == 255.0
    assert common.is_saturated(frame) is True


# ── 7. 多帧融合路径 ──────────────────────────────────────


class _TwelveBitStub(_StubCamera):
    """12-bit 假相机: 帧装在 uint16 容器里, 计数上限 4095。"""

    def __init__(self, radiance: np.ndarray, *, bit_depth: int | None = 12) -> None:
        super().__init__(bit_depth=bit_depth, exposure_time_ms=4.0)
        self.radiance = np.asarray(radiance, dtype=np.float64)

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = True) -> np.ndarray:
        ceiling = self._full_scale(np.uint16)
        counts = np.clip(self.radiance * float(self.exposure_time_ms), 0.0, ceiling)
        if n_sample == 1:
            # 单帧路径: 裸 uint16 数组, **没有** AveragedFrame 标注。
            return counts.astype(np.uint16)
        return AveragedFrame(counts, np.dtype(np.uint16), full_scale=ceiling)


def _ramp() -> np.ndarray:
    return np.linspace(0.0, 900.0, 64).reshape(8, 8)


def test_capture_hdr_takes_its_full_scale_from_the_effective_bit_depth() -> None:
    """HDR 阶梯的 ``detector_full_scale`` 必须是 4095; 按 uint16 容器猜会得到 65535。"""
    cam = _TwelveBitStub(_ramp())
    result = cam.capture_hdr([1.0, 4.0, 16.0], n_sample=1)
    assert result.full_scale == pytest.approx(4095.0)
    assert result.applied_exposures_ms == (1.0, 4.0, 16.0)


def test_capture_hdr_with_averaged_frames_keeps_the_same_full_scale() -> None:
    """``n_sample > 1`` 时帧是 float32 的 AveragedFrame, 融合仍按生效位深定标。"""
    cam = _TwelveBitStub(_ramp())
    result = cam.capture_hdr([1.0, 4.0, 16.0], n_sample=3)
    assert result.full_scale == pytest.approx(4095.0)


def test_capture_fused_takes_its_full_scale_from_the_effective_bit_depth() -> None:
    cam = _TwelveBitStub(_ramp())
    fused = cam.capture_fused([1.0, 4.0, 16.0], n_sample=1)
    assert isinstance(fused, AveragedFrame)
    assert fused.full_scale == pytest.approx(4095.0)
    assert float(np.nanmax(fused)) <= 4095.0 + 1e-3


def test_fusion_paths_follow_the_bit_depth_when_it_is_unknown() -> None:
    """对照: 位深未知时两条融合路径都退回 uint16 容器的 65535 —— 量纲由后端声明决定。"""
    cam = _TwelveBitStub(_ramp(), bit_depth=None)
    assert cam.capture_hdr([1.0, 4.0], n_sample=1).full_scale == pytest.approx(65535.0)
    assert cam.capture_fused([1.0, 4.0], n_sample=1).full_scale == pytest.approx(65535.0)
