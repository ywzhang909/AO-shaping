"""离线锁定 ``BaseCamera`` 的曝光契约与三个自动曝光实现。

不碰硬件、不加载 SDK: 后端侧只用 ``MagicMock`` 顶替设备句柄。覆盖点
(见 ``drivers/ccd/AGENTS.md`` §自动曝光 / §曝光兼容助手 / §BaseCamera 接口):

- 契约默认值的**唯一来源**是 ``BaseCamera.AUTO_EXPOSURE_*`` 常量, 后端覆写引用它们;
- :meth:`BaseCamera.auto_exposure` 的通用比例环: 收敛判据、``twice_valid``、单步
  放大上限 ``AUTO_EXPOSURE_MAX_RATIO``、以及**按生效值**判边界;
- 🔴 基类环**绝不**回调 ``common.auto_exposure`` —— 后者的职责是「有原生实现就委托给
  后端」, 反向调用即无限递归;
- 峰值一律折算到 0-255 灰度 (16-bit 帧不得被当成「过亮」而把曝光压到下限);
- MiiCam 的驱动缓存必须等于下发给 SDK 的**整数微秒** (issue #32);
- ``denoise`` 不属于采集契约; 没有目标亮度能力的后端要**抛错**而不是静默接受。
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ao_shaping.drivers.ccd.base import AveragedFrame, BaseCamera, CameraError
from ao_shaping.drivers.ccd.common import (
    auto_exposure as common_auto_exposure,
    normalize_peak_to_gray,
    set_camera_exposure_ms,
)

GRAY_FULL_SCALE = 255.0


def _import_backend(name: str) -> Any:
    """按后端取类 (大恒/MiiCam 的模块 import 本身离线可用)。

    返回 ``Any``: 这里是按名字挑不同后端类的测试夹具, 各后端的公开面并不相同
    (例如 ``_quantize_exposure_us`` 只存在于 MiiCam); 声明成联合类型只会让断言
    被逐个后端重述一遍, 反而盖住真正要验证的契约。
    """
    if name == "daheng":
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        return DahengCamera
    if name == "miicam":
        from ao_shaping.drivers.ccd.miicam.driver import MIICamera

        return MIICamera
    if name == "ffmpeg":
        from ao_shaping.drivers.ccd.ffmpeg.driver import FFmpegCamera

        return FFmpegCamera
    if name == "image_folder":
        from ao_shaping.drivers.ccd.ffmpeg.driver import ImageFolderCamera

        return ImageFolderCamera
    if name == "sim_pib":
        from ao_shaping.drivers.sim.slm_pib_sim import SimPibCCD

        return SimPibCCD
    if name == "sim_ccd":
        from ao_shaping.drivers.sim.ccd.simulated_ccd import SimulatedCCD

        return SimulatedCCD
    raise KeyError(name)


class _ContractCamera(BaseCamera):
    """线性响应的最小后端: ``灰度峰值 = gain × 曝光(ms)``。

    参数:
        gain: 每毫秒曝光产生的 **0-255 灰度**峰值。
        exposure_ms: 初始曝光。
        dtype: 帧的像素类型;  counts 按满量程 (255 / 65535) 换算, 用来验证峰值折算。
        declared_range: 驱动**声明**的量程 (``get_exposure_range()`` 报出的值)。
        device_range: 设备**真实**量程; 与之不同时即可检验「生效值 ≠ 请求值」。
    """

    def __init__(
        self,
        gain: float = 10.0,
        exposure_ms: float = 20.0,
        dtype: type = np.uint8,
        declared_range: tuple[float, float] | None = None,
        device_range: tuple[float, float] | None = None,
    ):
        self.cam: object | None = "stub"  # 基类 _device_open() 读它
        self.cam_id = 0
        self.exposure_time_ms = float(exposure_ms)
        self._gain = float(gain)
        self._dtype = dtype
        self._full_scale = float(np.iinfo(dtype).max)
        declared = tuple(declared_range or device_range or (0.011, 10_000.0))
        device = tuple(device_range or declared)
        # 通用 get_exposure_range() 直接读这两个属性
        self.min_exposure_ms, self.max_exposure_ms = float(declared[0]), float(declared[1])
        self._dev_lo, self._dev_hi = float(device[0]), float(device[1])
        self.requests: list[float] = []  # 比例环请求的值
        self.applied: list[float] = []  # 设备真正用上的值

    # ------------------------------------------------------------ 采集
    def get_numpy_image(self, n_sample=1, skip_first=True):
        counts = round(self.exposure_time_ms * self._gain * self._full_scale / GRAY_FULL_SCALE)
        return np.full((4, 4), int(min(counts, self._full_scale)), dtype=self._dtype)

    # ------------------------------------------------------------ 曝光
    def reset_exposure_time(self, time_ms: float) -> float:
        """钳制到设备**真实**量程并返回生效值 (可以不等于请求)。"""
        self.requests.append(float(time_ms))
        self.exposure_time_ms = float(
            min(max(float(time_ms), self._dev_lo), self._dev_hi)
        )
        self.applied.append(self.exposure_time_ms)
        return self.exposure_time_ms

    # ------------------------------------------------------------ 其余抽象方法
    def initialize(self) -> None:
        pass

    def open(self) -> None:
        pass

    def close(self) -> None:
        self.cam = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def reset_window(
        self,
        center: tuple[int, int] | tuple[np.intp, ...],
        size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """契约测试桩: 按基类签名返回回读后的 ((w, h), (cx, cy))。"""
        return size, (int(center[0]), int(center[1]))

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        return False

    def get_auto_exposure_state(self) -> dict:
        return {"enabled": False, "target": self._auto_exposure_target}

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        return False

    @staticmethod
    def get_cam_list() -> list[Any]:
        return []


# ---------------------------------------------------------------- 契约默认值


class TestContractDefaults:
    def test_base_constants_are_the_single_source_of_truth(self):
        assert BaseCamera.AUTO_EXPOSURE_TARGET_MAX == pytest.approx(40.0)
        assert BaseCamera.AUTO_EXPOSURE_TOLERANCE == pytest.approx(5.0)
        assert BaseCamera.AUTO_EXPOSURE_MAX_ITERATIONS == 20
        assert BaseCamera.AUTO_EXPOSURE_N_SAMPLE == 1
        # 单步放大上限: 三个实现必须一致, 否则同一个 target 在不同后端收敛到不同曝光
        assert BaseCamera.AUTO_EXPOSURE_MAX_RATIO == pytest.approx(3.0)
        assert BaseCamera.DEFAULT_EXPOSURE_RANGE_MS == (0.011, 10_000.0)

    def test_contract_constants_are_class_attributes_not_instance_state(self):
        for name in (
            "AUTO_EXPOSURE_TARGET_MAX",
            "AUTO_EXPOSURE_TOLERANCE",
            "AUTO_EXPOSURE_MAX_ITERATIONS",
            "AUTO_EXPOSURE_N_SAMPLE",
            "AUTO_EXPOSURE_MAX_RATIO",
            "DEFAULT_EXPOSURE_RANGE_MS",
        ):
            assert name in vars(BaseCamera), name

    @pytest.mark.parametrize("backend", ["daheng", "miicam", "sim_pib"])
    def test_backend_signatures_match_the_base_contract(self, backend: str):
        """后端覆写的参数名/默认值与基类契约逐字一致 (调用方不必按后端分支)。"""
        cls = _import_backend(backend)
        signature = inspect.signature(cls.auto_exposure)
        contract = ["target_max", "tolerance", "twice_valid", "max_iterations", "n_sample"]
        assert list(inspect.signature(BaseCamera.auto_exposure).parameters)[1:] == contract
        assert list(signature.parameters)[1 : len(contract) + 1] == contract
        expected = {
            "target_max": BaseCamera.AUTO_EXPOSURE_TARGET_MAX,
            "tolerance": BaseCamera.AUTO_EXPOSURE_TOLERANCE,
            "max_iterations": BaseCamera.AUTO_EXPOSURE_MAX_ITERATIONS,
            "n_sample": BaseCamera.AUTO_EXPOSURE_N_SAMPLE,
        }
        for name, value in expected.items():
            assert signature.parameters[name].default == value, f"{backend}.{name}"

    @pytest.mark.parametrize("backend", ["daheng", "miicam", "sim_pib"])
    def test_backend_defaults_reference_class_vars_not_literals(self, backend: str):
        """默认值写的是 ``BaseCamera.AUTO_EXPOSURE_*``, 环里用的是共享常量与折算入口。"""
        src = inspect.getsource(_import_backend(backend).auto_exposure)
        assert "BaseCamera.AUTO_EXPOSURE_TARGET_MAX" in src, backend
        assert "BaseCamera.AUTO_EXPOSURE_TOLERANCE" in src, backend
        # SimPibCCD scales simulated gain directly; it has no exposure-control loop.
        if backend != "sim_pib":
            assert "AUTO_EXPOSURE_MAX_RATIO" in src, backend
            assert "_auto_exposure_peak" in src, backend

    @pytest.mark.parametrize("backend", ["daheng", "miicam", "sim_pib", "ffmpeg", "sim_ccd"])
    def test_inherited_loop_defaults_are_the_contract(self, backend: str):
        """直接继承基类环的后端, 默认值就是契约档位。"""
        cls = _import_backend(backend)
        signature = inspect.signature(cls.auto_exposure)
        assert signature.parameters["target_max"].default == pytest.approx(40.0)
        assert signature.parameters["twice_valid"].default is True

    def test_generic_loop_never_delegates_back_to_common_helper(self):
        """🔴 基类环里不得出现对 ``auto_exposure`` 的调用 (那是委托入口, 反向即递归)。"""
        callees = set()
        for node in ast.walk(_function_ast(BaseCamera.auto_exposure)):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                callees.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                callees.add(node.func.id)
        assert "auto_exposure" not in callees
        assert "_auto_exposure_peak" in callees  # 折算峰值确实走的是共享入口

    def test_helper_only_delegates_to_an_override(self):
        """``common.auto_exposure`` 只在后端**覆写**了实现时才委托 (否则自递归)。"""
        from ao_shaping.drivers.ccd.common import _call_native_auto_exposure

        assert callable(_call_native_auto_exposure)
        src = inspect.getsource(
            __import__(
                "ao_shaping.drivers.ccd.common", fromlist=["auto_exposure"]
            ).auto_exposure
        )
        assert "BaseCamera.auto_exposure" in src


def _function_ast(func) -> ast.Module:
    """把 (可能缩进的) 函数源码解析成 AST。"""
    return ast.parse(textwrap.dedent(inspect.getsource(func)))


# ---------------------------------------------------------------- 通用比例环


class TestGenericLoop:
    def test_converges_to_target_band_on_a_base_subclass(self):
        cam = _ContractCamera(gain=10.0, exposure_ms=20.0)  # 灰度峰值 = 曝光 × 10

        img = cam.auto_exposure(target_max=40.0, tolerance=5.0)

        assert isinstance(img, np.ndarray)
        assert cam.exposure_time_ms == pytest.approx(4.0)  # 40 / 10
        assert cam.requests == [pytest.approx(4.0)]  # 一次修改即到位

    def test_defaults_reach_the_contract_band(self):
        """不传参时按 ``BaseCamera.AUTO_EXPOSURE_*`` 收敛到 40±5 灰度。"""
        cam = _ContractCamera(gain=10.0, exposure_ms=20.0)

        cam.auto_exposure()

        assert 35.0 <= normalize_peak_to_gray(cam.get_numpy_image()) <= 45.0

    def test_per_step_increase_is_capped_by_max_ratio(self):
        """很暗的画面一步最多放大 3× (AUTO_EXPOSURE_MAX_RATIO), 不会一次冲顶。"""
        cam = _ContractCamera(gain=0.01, exposure_ms=20.0)

        cam.auto_exposure(target_max=40.0, tolerance=5.0, max_iterations=1)

        assert cam.requests == [pytest.approx(60.0)]  # 20 × 3, 而不是 20 × 4000

    def test_twice_valid_needs_two_consecutive_in_band_frames(self):
        cam = _ContractCamera(gain=2.0, exposure_ms=20.0)  # 峰值已在带内 (40)
        frames = _count_frames(cam)
        cam.auto_exposure(target_max=40.0, tolerance=5.0, twice_valid=True)
        assert len(frames) == 2  # 命中两次才认收敛
        assert cam.requests == []  # 达标就不动曝光

        cam = _ContractCamera(gain=2.0, exposure_ms=20.0)
        frames = _count_frames(cam)
        cam.auto_exposure(target_max=40.0, tolerance=5.0, twice_valid=False)
        assert len(frames) == 1
        assert cam.requests == []

    def test_n_sample_is_passed_to_capture(self):
        cam = _ContractCamera(gain=10.0, exposure_ms=20.0)
        seen: list[int] = []
        original = cam.get_numpy_image

        def spy(n_sample=1, skip_first=True):
            seen.append(int(n_sample))
            return original(n_sample=n_sample, skip_first=skip_first)

        cam.get_numpy_image = spy  # type: ignore[method-assign]

        cam.auto_exposure(target_max=40.0, tolerance=5.0, n_sample=5)

        assert seen and set(seen) == {5}

    def test_tolerance_must_be_positive(self):
        with pytest.raises(ValueError, match="tolerance"):
            _ContractCamera().auto_exposure(target_max=40.0, tolerance=0.0)

    def test_boundary_exit_uses_the_applied_value(self):
        """过亮时后端回报「已到下限」⇒ 一次修改即退出, 不空转到 max_iterations。"""
        cam = _ContractCamera(gain=1000.0, exposure_ms=20.0, declared_range=(10.0, 100.0))

        cam.auto_exposure(target_max=40.0, tolerance=5.0)

        assert cam.exposure_time_ms == pytest.approx(10.0)
        assert cam.applied == [pytest.approx(10.0)]

    def test_tighter_device_clamp_cannot_pretend_convergence(self):
        """生效值停在设备下限 (高于声明下限) 时不得谎报收敛, 但必须有界退出。

        这条是「比较生效值而不是请求值」的另一半: 后端回报 25 ms 而声明量程是
        10 ms ⇒ 不满足边界退出条件, 环一路跑到 ``max_iterations`` 而不是无限循环
        (MiiCam 的台阶响应正是这种有界不收敛)。
        """
        cam = _ContractCamera(
            gain=1000.0,
            exposure_ms=20.0,
            declared_range=(10.0, 100.0),
            device_range=(25.0, 100.0),
        )

        img = cam.auto_exposure(target_max=40.0, tolerance=5.0, max_iterations=4)

        assert cam.exposure_time_ms == pytest.approx(25.0)
        assert len(cam.applied) == 4  # 有界: 恰好 max_iterations 次修改
        assert normalize_peak_to_gray(img) > 45.0  # 仍未达标, 没被当成收敛

    def test_common_helper_does_not_recurse_into_base_loop(self):
        """``common.auto_exposure`` 对「只继承基类实现」的相机必须自己跑环。"""
        cam = _ContractCamera(gain=10.0, exposure_ms=20.0)

        img = common_auto_exposure(cam, 40.0, 5.0)

        assert isinstance(img, np.ndarray)
        assert cam.exposure_time_ms == pytest.approx(4.0)


def _count_frames(cam: _ContractCamera) -> list[int]:
    """把 ``get_numpy_image`` 包一层计数器。"""
    counter: list[int] = []
    original = cam.get_numpy_image

    def counted(n_sample=1, skip_first=True):
        counter.append(1)
        return original(n_sample=n_sample, skip_first=skip_first)

    cam.get_numpy_image = counted  # type: ignore[method-assign]
    return counter


# ---------------------------------------------------------------- 峰值折算


class TestPeakNormalization:
    def test_uint8_peak_is_untouched(self):
        assert normalize_peak_to_gray(np.full((4, 4), 200, dtype=np.dtype(np.uint8))) == 200.0

    def test_uint16_peak_is_rescaled_to_gray(self):
        peak = normalize_peak_to_gray(np.full((4, 4), 32768, dtype=np.dtype(np.uint16)))
        assert peak == pytest.approx(32768.0 * 255.0 / 65535.0)  # = 128 灰度

    def test_averaged_frame_uses_full_scale_not_dtype(self):
        """多帧平均后 dtype 恒为 float32, 满量程只能取 ``AveragedFrame.full_scale``。"""
        frame16 = AveragedFrame(np.full((4, 4), 2048.0, dtype=np.float32), np.dtype(np.uint16))
        assert normalize_peak_to_gray(frame16) == pytest.approx(2048.0 * 255.0 / 65535.0)
        frame8 = AveragedFrame(np.full((4, 4), 200.0, dtype=np.float32), np.dtype(np.uint8))
        assert normalize_peak_to_gray(frame8) == pytest.approx(200.0)

    def test_base_peak_entry_delegates_to_the_helper(self):
        cam = _ContractCamera()
        frame = np.full((4, 4), 32768, dtype=np.uint16)
        assert cam._auto_exposure_peak(frame) == pytest.approx(
            normalize_peak_to_gray(frame)
        )
        assert cam._auto_exposure_peak(frame) == pytest.approx(32768.0 * 255.0 / 65535.0)

    def test_16bit_backend_converges_to_the_gray_target(self):
        """16-bit 帧必须按 0-255 灰度判定, 否则会被当成过亮而把曝光压到下限。"""
        cam = _ContractCamera(gain=10.0, dtype=np.uint16)

        img = cam.auto_exposure(target_max=40.0, tolerance=5.0)

        assert img.dtype == np.uint16
        assert cam.exposure_time_ms == pytest.approx(4.0)
        assert normalize_peak_to_gray(img) == pytest.approx(40.0)


# ---------------------------------------------------------------- 助手返回值


class TestHelperReturnContract:
    def test_set_camera_exposure_ms_returns_the_applied_value(self):
        """后端回报的生效值才是返回值, 不是请求值 (调用方拿它判边界)。"""

        class _ClampingCamera(_ContractCamera):
            def reset_exposure_time(self, time_ms: float) -> float:
                self.exposure_time_ms = 12.0  # 设备自作主张钳到 12 ms
                return self.exposure_time_ms

        cam = _ClampingCamera(exposure_ms=1.0)
        assert set_camera_exposure_ms(cam, 1.0) == pytest.approx(12.0)

    def test_set_camera_exposure_ms_falls_back_to_readback_for_void_setters(self):
        """老式 setter 不回报生效值时, 必须回读一次, 而不是把 None 交给调用方。

        这里用**鸭子类型**的旧后端 (不继承 ``BaseCamera``): 基类契约要求
        ``reset_exposure_time`` 返回 float, 若在这里覆写成 None 就是在测试里
        违反自己定的契约。
        """

        class _LegacyVoidSetter:
            exposure_time_ms = 1.0

            def reset_exposure_time(self, time_ms: float):
                self.exposure_time_ms = float(time_ms) + 0.5
                return None  # 旧后端: 不回报生效值

        cam = _LegacyVoidSetter()
        assert set_camera_exposure_ms(cam, 4.0) == pytest.approx(4.5)

    def test_get_exposure_time_ms_reads_the_cache_and_raises_without_one(self):
        cam = _ContractCamera(exposure_ms=7.5)
        assert cam.get_exposure_time_ms() == pytest.approx(7.5)

        del cam.exposure_time_ms
        with pytest.raises(CameraError, match="get_exposure_time_ms"):
            cam.get_exposure_time_ms()

    def test_get_exposure_range_prefers_attrs_then_default(self):
        assert _ContractCamera(declared_range=(0.2, 350.0)).get_exposure_range() == (
            0.2,
            350.0,
        )

        plain = _ContractCamera()
        del plain.min_exposure_ms
        del plain.max_exposure_ms
        assert plain.get_exposure_range() == BaseCamera.DEFAULT_EXPOSURE_RANGE_MS

    def test_set_auto_exposure_target_records_the_effective_value(self):
        cam = _ContractCamera()
        assert cam.set_auto_exposure_target(42) == 42
        assert cam._auto_exposure_target == 42
        assert cam.get_auto_exposure_state()["target"] == 42
        # 0-255 灰度: 越界被钳制, 回报的是生效值而不是请求值
        assert cam.set_auto_exposure_target(300) == 255
        assert cam.set_auto_exposure_target(-5) == 0

    def test_target_record_is_safe_without_super_init(self):
        """``_auto_exposure_target`` 是类属性: 没调用 ``super().__init__`` 也安全。"""
        cam = _ContractCamera()
        assert 0 <= cam._auto_exposure_target <= 255


# ---------------------------------------------------------------- MiiCam 量化


def _miicam(exposure_ms: float = 10.0):
    MIICamera = _import_backend("miicam")
    cam = MIICamera(cam_id=0, exposure_time_ms=exposure_ms)
    cam.cam = MagicMock()
    return cam


class TestMiiCamQuantization:
    def test_quantizes_to_integer_microseconds_by_rounding(self):
        MIICamera = _import_backend("miicam")
        assert MIICamera._quantize_exposure_us(2.0) == 2000
        # 截断会得到 1000 (且对已量化值不闭合), 四舍五入才是设备用的那个整数
        assert MIICamera._quantize_exposure_us(1.001) == 1001
        assert int(1.001 * 1000) == 1000
        assert MIICamera._quantize_exposure_us(0.0117) == 12

    def test_clamps_to_the_device_range(self):
        MIICamera = _import_backend("miicam")
        lo = round(MIICamera.MIN_EXPOSURE_MS * 1000)
        assert MIICamera._quantize_exposure_us(0.001) == lo
        assert MIICamera._quantize_exposure_us(20_000.0) == 10_000_000

    def test_quantization_is_idempotent(self):
        """回读值再量化必须原样不动, 否则每读一次掉 1 µs。"""
        MIICamera = _import_backend("miicam")
        for ms in (0.0117, 1.001, 2.9999, 5.5555, 100.0004, 0.001, 99_999.0):
            once = MIICamera._quantize_exposure_us(ms)
            assert MIICamera._quantize_exposure_us(once / 1000.0) == once

    def test_clamp_writes_the_quantized_cache(self):
        cam = _miicam()
        applied = cam._clamp_exposure_ms(0.0117)
        assert applied == pytest.approx(0.012)
        assert cam.exposure_time_ms == pytest.approx(0.012)
        assert cam.get_exposure_time_ms() == pytest.approx(0.012)

    def test_reset_exposure_time_sends_integer_microseconds(self):
        cam = _miicam(exposure_ms=1.0)
        with patch("ao_shaping.drivers.ccd.miicam.driver.time.sleep"):
            applied = cam.reset_exposure_time(2.3456)

        assert applied == pytest.approx(2.346)
        sent = [call.args[0] for call in cam.cam.put_ExpoTime.call_args_list]
        assert sent == [2346]
        assert all(isinstance(value, int) for value in sent)
        assert cam.cam.put_AutoExpoEnable.call_args.args == (0,)

        # 缓存是权威: 与下发给 SDK 的值一致, 重复设置不再漂移
        assert cam.get_exposure_time_ms() == pytest.approx(2.346)
        with patch("ao_shaping.drivers.ccd.miicam.driver.time.sleep"):
            again = cam.reset_exposure_time(applied)
        assert again == pytest.approx(2.346)
        assert cam.cam.put_ExpoTime.call_args.args == (2346,)

    def test_exposure_range_is_the_device_constant_range(self):
        cam = _miicam()
        assert cam.get_exposure_range() == (
            cam.MIN_EXPOSURE_MS,
            cam.MAX_EXPOSURE_MS,
        )

    def test_exposure_range_matches_the_instance_method_contract(self):
        cls = _import_backend("miicam")
        descriptor = inspect.getattr_static(cls, "get_exposure_range")
        assert not isinstance(descriptor, (staticmethod, classmethod))
        assert inspect.signature(cls.get_exposure_range) == inspect.signature(
            BaseCamera.get_exposure_range
        )


# ---------------------------------------------------------------- 后端表面


class TestBackendSurface:
    @pytest.mark.parametrize(
        "backend", ["daheng", "miicam", "ffmpeg", "image_folder", "sim_pib", "sim_ccd"]
    )
    def test_denoise_is_not_part_of_the_capture_contract(self, backend: str):
        """降噪不属于采集契约: 任何后端的 ``get_numpy_image`` 都不该有该参数。"""
        cls = _import_backend(backend)
        params = list(inspect.signature(cls.get_numpy_image).parameters)
        assert "denoise" not in params, backend
        assert params[:3] == ["self", "n_sample", "skip_first"], backend

    @pytest.mark.parametrize("backend", ["daheng", "miicam"])
    def test_readback_methods_are_overridden_by_hardware_backends(self, backend: str):
        """基类只能读缓存 ⇒ 有设备侧真值的后端必须覆写 (issue #32)。"""
        cls = _import_backend(backend)
        for name in ("get_exposure_time_ms", "get_exposure_range", "set_auto_exposure_target"):
            assert name in vars(cls), f"{backend}.{name}"

    @pytest.mark.parametrize(
        "backend", ["daheng", "miicam", "ffmpeg", "image_folder", "sim_pib", "sim_ccd"]
    )
    def test_reset_window_matches_the_base_contract(self, backend: str):
        """``reset_window`` 是实例方法, 参数注解与基类一致 (不得收窄 ``center``)。"""
        cls = _import_backend(backend)
        descriptor = inspect.getattr_static(cls, "reset_window")
        assert not isinstance(descriptor, (staticmethod, classmethod)), backend
        base_params = inspect.signature(BaseCamera.reset_window).parameters
        sub_params = inspect.signature(cls.reset_window).parameters
        for name, base_p in base_params.items():
            assert name in sub_params, f"{backend}.reset_window missing {name}"
            assert sub_params[name].annotation == base_p.annotation, (
                f"{backend}.reset_window {name}"
            )

    @pytest.mark.parametrize("backend", ["ffmpeg", "image_folder"])
    def test_replay_backends_raise_on_unsupported_target(self, backend: str):
        cls = _import_backend(backend)
        cam = object.__new__(cls)  # 不构造: 无需视频源 / 文件夹
        with pytest.raises(NotImplementedError):
            cam.set_auto_exposure_target(40)

    def test_daheng_discards_stale_frames_after_an_exposure_change(self):
        DahengCamera = _import_backend("daheng")
        assert DahengCamera.EXPOSURE_CHANGE_DISCARD_FRAMES == 3
        assert "EXPOSURE_CHANGE_DISCARD_FRAMES" in inspect.getsource(
            DahengCamera.reset_exposure_time
        )

    @pytest.mark.parametrize("backend", ["daheng", "miicam"])
    def test_auto_exposure_reads_range_and_exposure_through_the_contract(self, backend: str):
        """后端环自己不得再翻 SDK 私有字段: 量程/回读走契约方法。"""
        cls = _import_backend(backend)
        src = inspect.getsource(cls.auto_exposure)
        assert "self.get_exposure_range()" in src, backend
        assert "self.get_exposure_time_ms()" in src, backend
