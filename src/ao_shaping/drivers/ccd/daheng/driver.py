from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import numpy.typing as npt

from ao_shaping.drivers.ccd import BaseCamera
from ao_shaping.drivers.ccd.base import AveragedFrame, CameraError
from ao_shaping.drivers.ccd.common import ExposureTime
from ao_shaping.drivers.ccd.daheng import constants
from ao_shaping.drivers.ccd.pixel_size import (
    ensure_model_pixel_size_um,
    resolve_model_pixel_size_um,
    store_model_pixel_size_um,
)
from ao_shaping.utils.io import paths
from ao_shaping.utils.io.device_config import ConfigHandler, DeviceParam, param
from ao_shaping.utils.io.file import ROOT_DIR as PROJECT_ROOT
from ao_shaping.utils.io.file import logger

_SDK_IMPORT_ERROR: str | None = None
try:
    import gxipy as gx  # type: ignore[import-untyped]
except (ImportError, OSError) as e:
    gx = None  # type: ignore[assignment]
    _SDK_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    logger.error(f"Daheng SDK import failed: {e}")

# ── CCD 配置参数 ─────────────────────────────────────────


_CCD_CONFIG_DIR = PROJECT_ROOT / "data" / "ccd_configs"
_DEFAULT_CCD_CONFIG_DIR = _CCD_CONFIG_DIR


def _config_dir() -> Path:
    """CCD 配置目录: 模块常量 (测试可覆盖) 优先, 否则走统一解析器。

    解析器优先级: 环境变量 ``CCD_CONFIG_DIR`` → 配置文件 ``[paths] ccd_config_dir``
    → 默认 ``<锚定目录>/data/ccd_configs``。import 期不读盘, 调用时才解析。
    """
    if _CCD_CONFIG_DIR is not _DEFAULT_CCD_CONFIG_DIR:
        return _CCD_CONFIG_DIR
    return paths.device_config_dir("ccd")


@dataclass
class CCDParams(DeviceParam):
    """Daheng 相机配置参数。"""

    cam_id: int = param(default=0, cast=int)
    exposure_time_ms: float = param(default=0.0, cast=float)
    skip_sampling: bool = param(default=False, cast=bool)
    bit_depth: int = param(default=8, cast=int)


# 模块级单例，所有 DahengCamera 实例共用
CCD_CONFIG = ConfigHandler(_config_dir, "ccd", CCDParams)


class DahengCamera(BaseCamera):
    #: 位深 → GenICam ``PixelFormat`` 名。12 bit 必须落到 MONO12, **不允许静默回落**
    #: 成 MONO8: 满量程从 4095 变成 255, 一切按满量程折算的判据 (饱和保护、线性度、
    #: 峰值统计) 都会建立在错量纲上。
    _BIT_DEPTH_PIXEL_FORMATS: ClassVar[dict[int, str]] = {
        8: "MONO8",
        12: "MONO12",
        16: "MONO16",
    }

    #: 大恒可切换的档位 = 上面真的映射到 PixelFormat 的那些 (单一来源, 别再抄一遍)。
    supported_bit_depths: ClassVar[tuple[int, ...]] = tuple(_BIT_DEPTH_PIXEL_FORMATS)

    @classmethod
    def from_params(cls, params: Any, **overrides: Any) -> Self:
        """由驱动参数对象构造一个大恒相机。"""
        kwargs = {
            "cam_id": getattr(params, "cam_id", 0),
            "exposure_time_ms": getattr(params, "exposure_time_ms", 0.0),
            "skip_sampling": getattr(params, "skip_sampling", False),
            "bit_depth": getattr(params, "bit_depth", 8),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def __init__(
        self,
        cam_id: int = 0,
        exposure_time_ms: float = 0.0,
        skip_sampling: bool = False,
        bit_depth: int = 8,
    ):
        self._init_values = {
            "cam_id": cam_id,
            "exposure_time_ms": exposure_time_ms,
            "skip_sampling": skip_sampling,
            "bit_depth": bit_depth,
        }
        # 使用 defaults + __init__ 参数解析（尚未连接，无序列号）
        params = CCD_CONFIG.resolve_from_config({}, init_values=self._init_values)

        # device_manager 延迟到连接路径 (initialize) 才创建,
        # 使无 SDK 环境 (Linux 离线) 下构造实例不抛 NameError。
        self._device_manager = None
        self.cam_id = params.cam_id
        self.__exposure_time_ms = ExposureTime(params.exposure_time_ms)
        self.skip_sampling = params.skip_sampling
        # 位深只有一处存储: 公开的 `bit_depth` (契约见 BaseCamera)。这里先记**请求值**,
        # open() 经 set_bit_depth 用设备回读真值覆盖它 —— 不再另设私有缓存。
        self.bit_depth = params.bit_depth

        self.cam = None
        self._sn: str | None = None
        self.cam_width, self.cam_height = 0, 0
        #: 相机型号 (SDK 的 DeviceModelName), initialize() 时确定。
        self._model: str | None = None
        #: 像元尺寸显式覆盖 (µm); None = 按型号解析。
        self._pixel_size_um: float | None = None
        #: SDK 取流是否在跑 (initialize 末尾 ``stream_on`` 后置真, close 置假)。
        #: 换设置后「消费旧流水线帧」必须先开流 —— 未开流时 ``get_image`` 永不返回,
        #: 会死等。reset_window 期间的短暂停流不改本标记: 那段时间没有位深切换入口,
        #: 且停流后抛异常时保持为假 = 不取帧 (失败即安全)。
        self._acquisition_on = False

    @property
    def device_manager(self):
        """惰性创建并缓存 gxipy DeviceManager (仅连接路径使用, 无 SDK 时 ``gx`` 未定义)。"""
        if self._device_manager is None:
            self._device_manager = gx.DeviceManager()  # type: ignore[name-defined]
        return self._device_manager

    @device_manager.setter
    def device_manager(self, value) -> None:
        self._device_manager = value

    @property
    def cam_type(self) -> str:
        return "daheng"

    def open(self) -> None:
        """打开相机设备 (initialize 的别名)。

        契约 (与 :class:`~ao_shaping.drivers.ccd.base.BaseCamera` 一致):
        成功返回 ``None``; 失败抛异常并携带具体信息
        (``ConnectionAbortedError`` 未找到设备 / ``CameraError`` 初始化失败)。
        """
        self.initialize()

    def close(self) -> None:
        """关闭相机设备并释放资源。"""
        if self.cam:
            self.cam_width, self.cam_height = 0, 0
            self.cam.stream_off()
            self._acquisition_on = False
            self.cam.close_device()
            self.cam = None
            self._sn = None

    @property
    def sn(self) -> str | None:
        """获取相机序列号。"""
        return self._sn

    @property
    def model_name(self) -> str | None:
        """相机型号 (SDK ``DeviceModelName``), 未打开时为 ``None``。"""
        return self._model

    @property
    def pixel_size_um(self) -> float:
        """像元边长 (µm): 型号配置 → 已知型号表 → 抛 :class:`CameraError`。

        大恒可接入不同型号, 像元尺寸因此**不写死在驱动里**: 按型号从
        ``<CCD_CONFIG_DIR>/ccd/models.json`` 载入 (见 :mod:`ao_shaping.drivers.ccd.pixel_size`)。
        型号未登记在这里报错, 不回落"差不多"的数 —— 它决定全部远场物理标度。

        Raises:
            CameraError: 相机未打开 (型号未知), 或该型号未登记像元尺寸。
        """
        if self._pixel_size_um is not None:
            return float(self._pixel_size_um)
        if not self._model:
            raise CameraError(
                "相机尚未打开, 型号未知 ⇒ 像元尺寸无法确定; "
                "请先 initialize()/open() 再读取 pixel_size_um"
            )
        return resolve_model_pixel_size_um(self._model)

    def ensure_pixel_size_um(self, *, prompt=None) -> float:
        """确保当前型号的像元尺寸已登记; 未登记则询问用户并写入配置。

        脚本侧入口 (驱动本身不交互); 录入一次后同型号后续运行直接命中 ``models.json``。

        Args:
            prompt: 提问函数, 默认 ``input``。
        """
        if not self._model:
            raise CameraError("相机尚未打开, 型号未知 ⇒ 无法登记像元尺寸")
        value = ensure_model_pixel_size_um(self._model, prompt=prompt)
        self._pixel_size_um = value
        return value

    def store_pixel_size_um(self, pixel_size_um: float) -> None:
        """把当前型号的像元尺寸写入型号配置, 并作为本实例的显式值。"""
        if not self._model:
            raise CameraError("相机尚未打开, 型号未知 ⇒ 无法登记像元尺寸")
        store_model_pixel_size_um(self._model, pixel_size_um)
        self._pixel_size_um = float(pixel_size_um)

    def is_connected(self) -> bool:
        """检查相机是否已连接且就绪。"""
        return self.cam is not None and self._sn is not None

    def _resolve_model(self, dev_info: dict) -> str | None:
        """确定相机型号名: 设备列表项 → GenICam ``DeviceModelName``。

        像元尺寸按型号登记, 所以型号必须在 ``initialize`` 时就定下来; 拿不到就留
        ``None``, 之后读 ``pixel_size_um`` 会明确报错, 而不是猜。
        """
        raw: Any = dev_info.get("model_name", "") or ""
        if isinstance(raw, tuple):
            raw = raw[0] if raw else ""
        name = str(raw).strip()
        if name:
            return name
        feature = getattr(self.cam, "DeviceModelName", None)
        if feature is None:
            return None
        raw = feature.get()
        return str(raw).strip() or None

    def load_config(self) -> dict:
        """加载当前设备的配置文件。

        Returns:
            配置字典；无序列号或文件不存在时返回空字典。
        """
        if not self._sn:
            return {}
        return CCD_CONFIG._manager.load_config(self._sn)

    def save_config(self) -> None:
        """将当前参数保存到 JSON 配置文件。"""
        if not self._sn:
            logger.warning("未获取到序列号，跳过配置保存")
            return
        config = CCD_CONFIG.collect(self)
        CCD_CONFIG._manager.save_config(self._sn, config)
        config_file = CCD_CONFIG._manager._get_config_file(self._sn)
        logger.info(f"相机配置已保存: {config_file}")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _require_sdk(self) -> None:
        """SDK 不可用时给出可行动的错误, 而不是 `NameError`。

        模块顶层的 `import gxipy` 失败时 `gx` 被绑成 ``None``, 因此这里必须
        **先**拦住: 否则后续对 ``gx.`` 的取值会炸成 ``AttributeError``/``NameError``。
        """
        if gx is None:
            raise CameraError(
                f"gxipy SDK 不可用 ({_SDK_IMPORT_ERROR or 'import failed'}); "
                "大恒相机需要 libs/gxipy 在 PYTHONPATH 上"
            )

    def initialize(self):
        """初始化相机设备。

        依次执行: 关闭旧设备 → 更新设备列表并检查数量 → 打开设备 (处理 REPEAT_OPENED)
        → 读曝光范围 (µs→ms) 并把请求曝光钳到 [min,max] → 关 SDK 自动曝光 → 设增益/
        位深/合并/偏移/宽高 → 记序列号与型号 → 按序列号加载并应用配置并重写曝光 →
        更新属性 → 设流缓冲 NewestOnly → 开流。找不到设备记错误并抛 ConnectionAbortedError。
        """
        self._require_sdk()
        # 关闭之前打开的相机设备（如果有）
        self.__exit__(None, None, None)

        # 更新设备列表并获取设备信息列表
        _, dev_info_list = self.device_manager.update_device_list()
        # 检查设备列表长度是否小于等于指定的相机ID
        if not dev_info_list or len(dev_info_list) <= self.cam_id:
            error_info = f"Camera ID {self.cam_id} not found. "
            if dev_info_list:
                error_info += (
                    f" Available cameras: {[_.get('sn') for _ in dev_info_list]}."
                )
            logger.error(error_info)
            raise ConnectionAbortedError(error_info)

        sn = dev_info_list[self.cam_id].get("sn")
        try:
            self.cam = self.device_manager.open_device_by_sn(sn)
        except gx.gxiapi.InvalidAccess as e:  # type: ignore[attr-defined]
            if "REPEAT_OPENED" in str(e) or "device has been open" in str(e):
                logger.warning(
                    f"Device {sn} already opened, attempting to reinitialize..."
                )
                self.device_manager = gx.DeviceManager()
                self.cam = self.device_manager.open_device_by_sn(sn)
            else:
                raise
        self._ensure_open("camera not found")
        # 设置相机的曝光时间
        float_range = self.cam.ExposureTime.get_range()
        if float_range:
            # gxipy SDK 的 ExposureTime 单位是微秒 (μs)，
            # 转换为毫秒 (ms) 存入内部 ExposureTime，保证全驱动统一用 ms。
            self.__exposure_time_ms.min = float_range["min"] / 1000.0
            self.__exposure_time_ms.max = float_range["max"] / 1000.0
            logger.info(
                "Exposure time range: {:.1f}~{:.1f} µs ({:.3f}~{:.3f} ms)",
                float_range["min"],
                float_range["max"],
                self.__exposure_time_ms.min,
                self.__exposure_time_ms.max,
            )
        else:
            logger.warning(
                f"Exposure time range not found for camera {sn}. Using default value."
            )
        # 写入前把请求的曝光钳到设备范围内。
        # 越界的值 (尤其是 0 ms, 即约定俗成的"自动"哨兵值) 会让 SDK 的 `set`
        # 越界失败, 相机则静默沿用它上一次/默认的 (往往很长的) 曝光 → 饱和帧。
        _req = self.__exposure_time_ms.ms
        _lo, _hi = self.__exposure_time_ms.min, self.__exposure_time_ms.max
        if not (_lo <= _req <= _hi):
            _clamped = min(max(_req, _lo), _hi)
            logger.warning(
                "Exposure {:.3f}ms out of range [{:.3f}, {:.3f}]ms; clamped to {:.3f}ms",
                _req,
                _lo,
                _hi,
                _clamped,
            )
            self.__exposure_time_ms.ms = _clamped
        # 写入 SDK 时再转回 µs
        self.cam.ExposureTime.set(int(self.__exposure_time_ms.ms * 1000))
        # 关闭 SDK 原生自动曝光，确保手动曝光控制
        self.cam.ExposureAuto.set(constants.GxAutoEntry.OFF.value)

        # 设置相机的增益
        self.cam.Gain.set(0.0)
        # 请求的位深必须对应实际格式，不允许 12 bit 静默回落为 Mono8。
        # 走公共入口 set_bit_depth: 写 PixelFormat → 回读设备真值 → 不符即报错。
        # 此刻取流还没开 (stream_on 在 initialize 末尾), set_bit_depth 内部因此
        # **不消费帧** —— 未开流时取帧会死等, 且位深本来就得在开流之前写好。
        if self.bit_depth is not None:
            self.set_bit_depth(self.bit_depth)
        if self.skip_sampling:
            # 设置相机的合并因子为2
            self.cam.BinningHorizontal.set(2)
            self.cam.BinningVertical.set(2)

        # 设置相机的水平偏移量为0
        self.cam.OffsetX.set(0)
        self.cam.OffsetY.set(0)
        # 设置相机的宽度为最大宽度
        self.cam.Width.set(self.cam.WidthMax.get())
        self.cam.Height.set(self.cam.HeightMax.get())

        self._sn = sn
        self._model = self._resolve_model(dev_info_list[self.cam_id])

        # 按序列号加载并应用配置文件（不覆盖 __init__ 显式参数）
        config = self.load_config()
        CCD_CONFIG.apply_from_config(self, config, init_values=self._init_values)
        # 重新应用曝光时间（配置可能覆盖了 exposure_time_ms）
        self.cam.ExposureTime.set(int(self.__exposure_time_ms.ms * 1000))

        self.__update_properties()
        stream_mode = self.cam.data_stream[0].StreamBufferHandlingMode
        stream_mode.set(constants.GxDSStreamBufferHandlingModeEntry.NEWEST_ONLY.value)
        if stream_mode.get()[0] != constants.GxDSStreamBufferHandlingModeEntry.NEWEST_ONLY.value:
            raise RuntimeError("Daheng stream buffer mode did not switch to NewestOnly")
        self.cam.stream_on()
        self._acquisition_on = True

    #: 改曝光后必须消费掉的帧数 (2026-10-21 实测, 见 ``reset_exposure_time``)。
    EXPOSURE_CHANGE_DISCARD_FRAMES: int = 3
    """改曝光后主动丢掉的帧数 (大恒 SDK 改曝光后 1~2 帧仍在上次曝光, 留 1 帧余量;
    设 0 退回「不消费」的旧行为)。"""

    def reset_exposure_time(self, time_ms: float) -> float:
        """重设曝光时间, 并让**改曝光后的第一帧就是新曝光**。

        大恒 SDK 在取流进行中直接写 ``ExposureTime``, 改动前已开积分的 1~2 帧仍留在
        SDK 流水线, 仍是上一档曝光的画面 (2026-10-21 实测 FJB24112232: 2→20ms 第 3 帧
        才出现新值)。``get_numpy_image`` 的 ``skip_first`` 只丢 1 帧**不够** —— 2026-10-20
        的测光阶梯因此把 5ms 档峰值记成 2ms 档。故这里在曝光真的改变后主动消费
        ``EXPOSURE_CHANGE_DISCARD_FRAMES`` 帧, 把「第一帧即新曝光」变成**驱动层不变量**
        (代价 3 帧采集时间, 100ms 曝光约 300ms)。

        参数:
            time_ms: 新的曝光时间, 单位毫秒。

        返回:
            float: 实际设定的曝光时间, 单位毫秒。
        """
        self._ensure_open()
        time_ms = float(time_ms)
        if time_ms < self.__exposure_time_ms.min:
            v = self.__exposure_time_ms.min
            logger.warning(
                f"exposure time must >= {self.__exposure_time_ms.min}ms. set to {self.__exposure_time_ms.min}ms."
            )
        elif time_ms > self.__exposure_time_ms.max:
            v = self.__exposure_time_ms.max
            logger.warning(
                f"exposure time must <= {self.__exposure_time_ms.max}ms. set to {self.__exposure_time_ms.max}ms."
            )
        else:
            v = time_ms
        previous = float(self.__exposure_time_ms.current)
        self.cam.ExposureTime.set(int(v * 1000))
        self.__exposure_time_ms.ms = self.exposure_time
        if abs(float(v) - previous) > 1e-6:
            self._consume_stale_exposure_frames()

        return self.exposure_time

    def _consume_stale_exposure_frames(self) -> int:
        """消费掉改曝光之前就在 SDK 流水线里的那些帧。

        返回:
            int: 实际消费的帧数。
        """
        for _ in range(self.EXPOSURE_CHANGE_DISCARD_FRAMES):
            self.__take_one_shot()
        return self.EXPOSURE_CHANGE_DISCARD_FRAMES

    @classmethod
    def _bit_depth_of_format(cls, value: Any) -> int | None:
        """把设备回读的 ``PixelFormat`` 归一化成位深 (bit); 认不出来返回 ``None``。

        gxipy 枚举特征的 ``get()`` 形态并不统一: 可能是名称串 (``"Mono12"``)、PFNC
        代码 (``0x01100005``)、带 ``.name`` 的枚举成员, 也可能是 ``(代码, 名称)``
        元组 (本文件 ``StreamBufferHandlingMode.get()[0]`` 就是这种)。位深**只能以
        回读为准** —— SDK 写 ``PixelFormat`` 可能被静默保留旧值 (同 ``IntFeature.set``
        的坑), 所以这里对四种形态都容错。
        """
        candidates = list(value) if isinstance(value, (tuple, list)) else [value]
        for candidate in candidates:
            member_name = getattr(candidate, "name", None)
            if isinstance(member_name, str):  # Enum 成员 → 用它的名字
                candidate = member_name
            if isinstance(candidate, str):
                text = candidate.strip().upper()
                if text.startswith("MONO") and text[4:].isdigit():
                    return int(text[4:])
                continue
            code = getattr(candidate, "value", candidate)
            try:
                code_int = int(code)
            except (TypeError, ValueError):
                continue
            for bits, fmt_name in cls._BIT_DEPTH_PIXEL_FORMATS.items():
                entry = getattr(constants.GxPixelFormatEntry, fmt_name, None)
                if entry is not None and int(getattr(entry, "value", entry)) == code_int:
                    return bits
        return None

    def set_bit_depth(self, bits: int) -> int:
        """切换输出位深 (8 / 12 / 16 bit), 返回**设备回读**的生效位深。

        落实 :meth:`BaseCamera.set_bit_depth` 的四条契约:

        * 档位不在 :attr:`supported_bit_depths` 里 ⇒ :class:`ValueError`, **不回落**
          (12 bit 静默退回 MONO8 会把满量程从 4095 变成 255, 饱和判据与线性度全错);
        * 写完必须回读 ``PixelFormat``; 回读与请求不符 ⇒ :class:`CameraError`,
          但 :attr:`bit_depth` 仍先记下**设备真值** —— 调用方读它不会拿到虚报档位;
        * 位深真的变了、且 SDK 正在拉流 ⇒ 消费掉 ``EXPOSURE_CHANGE_DISCARD_FRAMES``
          帧: 改格式之前就已进入流水线的帧仍是旧格式画面 (与「改曝光后 1~2 帧仍是
          旧曝光」同一根因, 故共用同一帧数旋钮), 使「换位深后的第一帧就是新格式」;
        * 未开流**绝不取帧** (`get_image` 在停流状态下永不返回 ⇒ 死等)。所以
          ``initialize()`` 里应用配置位深时 (``stream_on`` 还没执行) 只写格式不消费帧,
          由 ``_acquisition_on`` 把关。

        参数:
            bits: 请求位深 (bit)。

        返回:
            int: 设备实际生效的位深。

        抛出:
            ValueError: 请求档位不被支持。
            CameraError: 设备未打开, 或回读位深与请求不符。
        """
        bits = int(bits)
        if bits not in self._BIT_DEPTH_PIXEL_FORMATS:
            raise ValueError("Daheng bit_depth must be 8, 12 or 16")
        self._ensure_open("camera not open")

        requested_format = self._BIT_DEPTH_PIXEL_FORMATS[bits]
        previous = self.bit_depth
        self.cam.PixelFormat.set(getattr(gx.GxPixelFormatEntry, requested_format))
        raw = self.cam.PixelFormat.get()
        effective = self._bit_depth_of_format(raw)
        #: 设备真值优先: 即使下面判为不符要抛错, 缓存也不能留一个设备并未使用的档位
        self.bit_depth = effective
        if effective != bits:
            raise CameraError(
                f"Daheng PixelFormat readback mismatch: requested {bits}-bit "
                f"({requested_format}), device reports {raw!r} "
                f"(parsed bit_depth={effective})"
            )
        if previous != effective and self._acquisition_on:
            for _ in range(self.EXPOSURE_CHANGE_DISCARD_FRAMES):
                self.get_numpy_image(1, skip_first=False)
        return effective

    @staticmethod
    def _clamp_int(value: int, int_range: dict | None) -> int:
        """把整数值钳制/对齐到 GenICam int 特征的 [min, max, inc] 网格。

        gxipy ``IntFeature.set`` 对越界值**不抛异常**——只 print 噪声后**静默保留旧值**
        (窗口不生效, 调用方无从知晓), 故必须在 set 前对齐到合法网格。

        Args:
            value: 请求值。
            int_range: ``feature.get_range()`` 的 {"min","max","inc"} (None 原样返回)。

        Returns:
            int: 对齐后的值, 满足 ``min <= v <= max`` 且 ``(v - min) % inc == 0``。
        """
        if int_range is None:
            return int(value)
        lo = int(int_range["min"])
        hi = int(int_range["max"])
        inc = max(int(int_range["inc"]), 1)
        if value < lo:
            return lo
        if value > hi:
            return hi
        # 向下对齐到增量网格：lo + (value - lo) // inc * inc
        return lo + (int(value) - lo) // inc * inc

    def reset_window(
        self,
        center: tuple[int, int] | tuple[np.intp, ...] = (0, 0),
        size: tuple[int, int] = (0, 0),
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """重置相机窗口大小与位置, 使图像中心位于指定位置。

        参数:
        size (Tuple[int]): 期望窗口大小 (宽, 高)。
        center (Tuple[int]): 期望窗口中心 (x, y)。

        返回:
        ((实际生效宽高), (实际生效中心坐标))。请求值越出相机 GenICam [min,max,inc]
        网格时会被钳制并记 warning (SDK 对越界 set 只 print 噪声并静默保留旧窗口,
        绝不传非法值给 SDK); 返回 set 后 readback 校验的真实值。
        """
        # 中心坐标大于0
        self._ensure_open()
        center = int(center[0]), int(center[1])
        self.cam.stream_off()
        # 如果未指定窗口大小，则使用相机的最大宽度和高度
        if size == (0, 0):
            width, height = (self.cam.WidthMax.get(), self.cam.HeightMax.get())
            if not (width and height):
                raise ValueError("camera width and height must be greater than 0")
            x_offset, y_offset = 0, 0
        else:
            width, height = int(size[0]), int(size[1])
            # GenICam 的 Width.range 是**随当前 OffsetX 收缩**的 (max = WidthMax -
            # OffsetX)，Height 同理。上一次的 ROI 若留下非零偏移，直接读范围会把
            # "请求全幅" 误钳成小窗口 (实测残留 OffsetX=568 时 2592 被钳到 2024，
            # 再叠 Height 约束后落到 1452×1122)，且 readback 看起来完全合法。
            # 因此先把偏移归零，让 range 反映传感器全幅。
            self.cam.OffsetX.set(0)
            self.cam.OffsetY.set(0)
            range_w, range_h = self.cam.Width.get_range(), self.cam.Height.get_range()
            if not (range_w and range_h):
                raise ValueError("camera width and height range not found")
            # gxipy IntFeature.set 对越界值不抛异常，而是 print 噪声 + 静默
            # 保留旧窗口 —— 先钳制到相机 [min, max, inc] 网格，set 永不越界。
            width = self._clamp_int(width, range_w)
            height = self._clamp_int(height, range_h)
            if (width, height) != (int(size[0]), int(size[1])):
                logger.warning(
                    "Window size ({}, {}) out of camera grid, clamped to ({}, {}) "
                    "(Width.range=[{}, {}, {}], Height.range=[{}, {}, {}])",
                    int(size[0]),
                    int(size[1]),
                    width,
                    height,
                    range_w["min"],
                    range_w["max"],
                    range_w["inc"],
                    range_h["min"],
                    range_h["max"],
                    range_h["inc"],
                )
            # 先设置尺寸再算偏移 —— OffsetX/OffsetY 的有效范围依赖当前
            # 已设置的 Width/Height（例如 OffsetX.max = WidthMax - Width）。
            self.cam.Width.set(width)
            self.cam.Height.set(height)
            # 计算窗口的偏移量，确保中心位置在指定位置
            x_offset = self._clamp_int(
                center[0] - (width // 2), self.cam.OffsetX.get_range()
            )
            y_offset = self._clamp_int(
                center[1] - (height // 2), self.cam.OffsetY.get_range()
            )
            if (x_offset, y_offset) != (
                center[0] - (width // 2),
                center[1] - (height // 2),
            ):
                logger.warning(
                    "Window offset ({}, {}) (center {}, size ({}, {})) out of "
                    "camera range, clamped to ({}, {})",
                    center[0] - (width // 2),
                    center[1] - (height // 2),
                    center,
                    width,
                    height,
                    x_offset,
                    y_offset,
                )
        self.cam.OffsetX.set(x_offset)
        self.cam.OffsetY.set(y_offset)
        logger.info("ROI Window offset: ({}, {})", x_offset, y_offset)
        self.__update_properties()
        self.cam.stream_on()

        # 读回校验：SDK 可能静默保留旧窗口/旧偏移（越界 set 不报错），
        # 返回实际生效值并 warning，避免调用方把请求值当生效值使用。
        eff_w, eff_h = self.cam_width, self.cam_height
        if (eff_w, eff_h) != (width, height):
            logger.warning(
                "Camera kept the previous window: requested ({}, {}), read back "
                "({}, {})",
                width,
                height,
                eff_w,
                eff_h,
            )
        eff_x = int(self.cam.OffsetX.get() or x_offset)
        eff_y = int(self.cam.OffsetY.get() or y_offset)
        if (eff_x, eff_y) != (x_offset, y_offset):
            logger.warning(
                "Camera kept the previous offsets: requested ({}, {}), read back "
                "({}, {})",
                x_offset,
                y_offset,
                eff_x,
                eff_y,
            )

        # 返回实际生效的窗口：((宽, 高), (实际中心x, 实际中心y))
        return (eff_w, eff_h), (eff_x + eff_w // 2, eff_y + eff_h // 2)

    def __take_one_shot(self) -> npt.NDArray[np.uint8]:
        """拍摄一张相机图像 (uint8)。

        返回:
            np.ndarray: 拍摄到的图像数据, 数据类型 uint8。
        """
        self._ensure_open()
        while True:
            raw_image = self.cam.data_stream[0].get_image()
            if raw_image and raw_image.get_status() == gx.GxFrameStatusList.SUCCESS:
                return raw_image.get_numpy_array()

    def get_numpy_image(
        self, n_sample: int = 1, skip_first: bool = True
    ) -> npt.NDArray[np.uint16] | npt.NDArray[np.uint8] | npt.NDArray[np.float32]:
        """获取相机图像数据并做平均处理 (n_sample 必须 > 0, 多帧平均为 float32)。"""
        if n_sample <= 0:
            raise ValueError("Sample count must be > 0")
        if skip_first:
            self.__take_one_shot()
        first_img = self.__take_one_shot()
        if n_sample == 1:
            return first_img
        total = np.asarray(first_img, dtype=np.float64).copy()
        for _ in range(n_sample - 1):
            total += self.__take_one_shot()
        avg_img = total / n_sample
        return AveragedFrame(avg_img, first_img.dtype)

    def auto_exposure(
        self,
        target_max: float = BaseCamera.AUTO_EXPOSURE_TARGET_MAX,
        tolerance: float = BaseCamera.AUTO_EXPOSURE_TOLERANCE,
        twice_valid: bool = True,
        max_iterations: int = BaseCamera.AUTO_EXPOSURE_MAX_ITERATIONS,
        n_sample: int = BaseCamera.AUTO_EXPOSURE_N_SAMPLE,
        use_sdk_auto: bool = True,
        sdk_settle_frames: int = 3,
    ) -> npt.NDArray[np.uint8]:
        """自动曝光调整 - 优先大恒 SDK 原生自动曝光, 再比例迭代修正。

        ``target_max``/``tolerance`` 均为 **0-255 灰度** (与 SDK ``ExpectedGrayValue``
        一致)。阶段 1 (use_sdk_auto): ExposureAuto="Once"+ExpectedGrayValue=target →
        采 sdk_settle_frames 帧让 SDK 收敛 → 读回曝光作初值 → 切手动。阶段 2 比例迭代:
        拍图取峰值, 落 ``target±tolerance`` (twice_valid 需连续两次) 即停; 否则
        ``new_exp = exp*target/peak`` (单步放大上限 3×) 裁剪到 [min,max], 处理已到
        边界仍不达标, 重复至收敛或 max_iterations。

        Args:
            target_max: 目标最大亮度 (0-255, 默认40)
            tolerance: 峰值容差 (0-255, 默认5)
            twice_valid: True 需连续两次落入容差才收敛 (默认True)
            max_iterations: 第二阶段最大迭代次数 (默认20)
            n_sample: 每次估计峰值的采样帧数 (默认1)
            use_sdk_auto: 是否先用 SDK 原生自动曝光 (默认True)
            sdk_settle_frames: SDK 收敛等待帧数 (默认3)

        Returns:
            np.ndarray: 调整后采集到的图像 (uint8)。
        """
        self._ensure_open()
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")

        target_val = float(target_max)
        low = int(max(target_val - tolerance, 10))
        high = int(min(target_val + tolerance, 254))
        min_exp, max_exp = self.get_exposure_range()

        # ── 阶段 1: SDK 原生自动曝光 ──
        if use_sdk_auto:
            try:
                sdk_target = int(max(0, min(255, target_val)))
                self.cam.ExposureAuto.set(constants.GxAutoEntry.ONCE.value)
                self.cam.ExpectedGrayValue.set(sdk_target)
                logger.info(
                    "[SDK auto-exposure] enabled, target={}, waiting {} frames "
                    "for convergence...",
                    sdk_target,
                    sdk_settle_frames,
                )
                # 采集若干帧让 SDK 内部自动曝光收敛
                for _ in range(sdk_settle_frames):
                    self.__take_one_shot()
                logger.info(
                    "[SDK auto-exposure] converged: exp={:.2f}ms",
                    self.get_exposure_time_ms(),
                )
                # 切换到手动模式，后续用比例迭代修正
                self.cam.ExposureAuto.set(constants.GxAutoEntry.OFF.value)
            except Exception as e:  # 保留宽泛捕获: gxipy 第三方 Feature 访问, 不支持时抛出的类型不可枚举 (UnexpectedError/FeatureTypeError/ParameterTypeError 各自独立继承 Exception, 无公共基类)
                logger.warning(
                    "[SDK auto-exposure] not available, falling back to manual: {}",
                    e,
                )

        logger.info(
            "Auto exposure start: target={:.0f}±{:.0f} (range=[{}, {}]ms, max_iter={})",
            target_val,
            tolerance,
            min_exp,
            max_exp,
            max_iterations,
        )

        # ── 阶段 2: 比例迭代修正 ──
        twice_ok = False
        img = self.get_numpy_image(n_sample, skip_first=True)

        for i in range(max(1, int(max_iterations))):
            # 0-255 灰度量纲 (与基类同一入口): 12/16-bit 帧满量程不是 255
            peak = self._auto_exposure_peak(img)

            if low <= peak <= high:
                if twice_ok or not twice_valid:
                    logger.info(
                        "Auto exposure converged at iter {}: exp={:.3f}ms, max={:.1f}",
                        i + 1,
                        self.get_exposure_time_ms(),
                        peak,
                    )
                    return img
                twice_ok = True
            else:
                twice_ok = False
                current = self.get_exposure_time_ms()
                if current <= 0:
                    break
                ratio = min(target_val / max(peak, 1.0), self.AUTO_EXPOSURE_MAX_RATIO)
                new_exp = float(np.clip(current * ratio, min_exp, max_exp))
                if abs(new_exp - current) < 1e-9:
                    break
                # 用**生效值**判边界: reset_exposure_time 会把请求钳到设备量程,
                # 拿请求值比较会在「设备拒绝该值」时一直转到 max_iterations。
                applied = float(self.reset_exposure_time(new_exp))
                if applied <= min_exp and peak > high:
                    logger.warning(
                        "target brightness {:.0f} unreachable (peak {:.1f}); "
                        "exposure forced to min",
                        target_val,
                        peak,
                    )
                    break
                if applied >= max_exp and peak < low:
                    logger.warning(
                        "target brightness {:.0f} unreachable (peak {:.1f}); "
                        "exposure forced to max",
                        target_val,
                        peak,
                    )
                    break

            img = self.get_numpy_image(n_sample, skip_first=True)

        logger.info(
            "Auto exposure finished: exp={:.3f}ms, max={:.1f}",
            self.get_exposure_time_ms(),
            self._auto_exposure_peak(img),
        )
        return img

    def __update_properties(self):
        self._ensure_open()
        width_val = self.cam.Width.get()
        height_val = self.cam.Height.get()
        assert width_val is not None and height_val is not None, (
            "camera size not available"
        )
        self.cam_width = int(width_val)
        self.cam_height = int(height_val)
        logger.info(
            f"Open cam {self._sn} success. width={self.cam_width}, height={self.cam_height}"
        )
        self.xv, self.yv = self.__get_grid(self.cam_width, self.cam_height)

    @property
    def exposure_time(self) -> float:
        self._ensure_open()
        _exp_time = self.cam.ExposureTime.get()
        return _exp_time / 1000.0 if _exp_time else 0.0

    @exposure_time.setter
    def exposure_time(self, time_ms: float):
        self._ensure_open()
        self.reset_exposure_time(time_ms)

    @staticmethod
    def __get_grid(width, height):
        x = np.arange(0, width)
        y = np.arange(0, height)
        xv, yv = np.meshgrid(x, y)
        return xv, yv

    @staticmethod
    def get_cam_list() -> list[Any]:
        """获取可用相机列表（大恒返回 SDK 的设备信息对象，非纯字符串）。"""
        device_manager = gx.DeviceManager()
        _, dev_info_list = device_manager.update_device_list()
        return dev_info_list if dev_info_list else list()

    def get_exposure_range(self) -> tuple[float, float]:
        """获取相机支持的曝光时间范围。

        返回:
            (min_exposure_ms, max_exposure_ms)
        """
        self._ensure_open()
        return float(self.__exposure_time_ms.min), float(self.__exposure_time_ms.max)

    def get_exposure_time_ms(self) -> float:
        """从设备**回读**当前生效的曝光时间 (ms)。

        直接读 SDK 的 ``ExposureTime`` 特征 (设备实际在用的值), 而非本地
        ``__exposure_time_ms`` 缓存 —— 后者在 ``reset_exposure_time`` 里经钳制
        + 量化后下发, 与原始请求可能差 1 µs, 且缓存从不随设备回读更新
        (issue #32)。
        """
        return float(self.exposure_time)

    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """启用或关闭自动曝光 (大恒 SDK ``ExposureAuto`` "On"/"Off")。

        启用时相机自动调整曝光与增益, 以达到 ``ExpectedGrayValue`` 设定的目标亮度。

        参数:
            enable: True 启用, False 关闭。
            mode: 自动曝光模式 (0=禁用, 1=连续, 2=单次); 只在 enable=True 时有意义,
                大恒 SDK 无单独模式枚举, 始终用连续模式。

        返回:
            bool: 成功 True, 不支持 False。
        """
        self._ensure_open()
        try:
            if enable:
                self.cam.ExposureAuto.set(constants.GxAutoEntry.CONTINUOUS.value)
                logger.info("Auto exposure enabled (continuous)")
            else:
                self.cam.ExposureAuto.set(constants.GxAutoEntry.OFF.value)
                logger.info("Auto exposure disabled")
            return True
        except Exception as e:  # 同上: gxipy Feature 访问, 不支持时抛出的类型不可枚举
            logger.warning(f"Auto exposure not supported on Daheng camera: {e}")
            return False

    def set_auto_exposure_target(self, target: int) -> int:
        """设置自动曝光的目标亮度。

        对应大恒 SDK ``ExpectedGrayValue`` (IntFeature, 0-255, 默认 120), 仅在自动
        曝光启用时生效。

        参数:
            target: 目标亮度值, 0-255, 默认 120。

        返回:
            实际设定的目标值。
        """
        self._ensure_open()
        target = max(0, min(255, target))
        try:
            self.cam.ExpectedGrayValue.set(target)
            logger.info(f"Auto exposure target brightness set to {target}")
        except Exception as e:  # 同上: gxipy Feature 访问, 不支持时抛出的类型不可枚举
            logger.warning(f"Auto exposure target not supported: {e}")
        return target

    def get_auto_exposure_state(self) -> dict:
        """获取当前自动曝光状态 (从大恒 SDK 读 ``ExposureAuto`` 与 ``ExpectedGrayValue``)。

        返回:
            字典: enabled (bool) / mode (int, 1=连续 0=禁用) / target (int) /
            exposure_time_ms (float, ms)。
        """
        self._ensure_open()
        state: dict = {
            "enabled": False,
            "mode": 0,
            "target": 120,
            "exposure_time_ms": 0.0,
        }
        try:
            _val, auto_key = self.cam.ExposureAuto.get()
            state["enabled"] = auto_key == "On"
            state["mode"] = 1 if auto_key == "On" else 0
            try:
                state["target"] = self.cam.ExpectedGrayValue.get()
            # policy: the auto-exposure target is an optional register; a state snapshot records only what this firmware exposes
            except Exception:  # 同上: gxipy Feature 访问, 不支持时抛出的类型不可枚举
                pass
            state["exposure_time_ms"] = self.exposure_time
        except Exception as e:  # 同上: gxipy Feature 访问, 不支持时抛出的类型不可枚举
            logger.warning(f"Auto exposure state not supported: {e}")
        return state

    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """设置自动曝光的时间与增益范围。

        对应大恒 SDK ``AutoExposureTimeMin/Max`` (FloatFeature, µs) 与 ``AutoGainMin/Max``,
        限定自动曝光/增益的搜索空间。

        参数:
            max_time_ms: 最大曝光时间 (ms); min_time_ms: 最小曝光时间 (ms);
                max_gain: 最大增益; min_gain: 最小增益。

        返回:
            bool: 成功 True, 不支持 False。
        """
        self._ensure_open()
        try:
            self.cam.AutoExposureTimeMin.set(min_time_ms * 1000)
            self.cam.AutoExposureTimeMax.set(max_time_ms * 1000)
            self.cam.AutoGainMin.set(min_gain)
            self.cam.AutoGainMax.set(max_gain)
            logger.info(
                f"Auto exposure range set: time=[{min_time_ms}, {max_time_ms}]ms, "
                f"gain=[{min_gain}, {max_gain}]"
            )
            return True
        except Exception as e:  # 同上: gxipy Feature 访问, 不支持时抛出的类型不可枚举
            logger.warning(f"Auto exposure range not supported: {e}")
            return False

    @property
    def auto_exposure_value(self) -> float:
        """当前自动曝光的数值 (曝光时间, ms); 启用时返回 SDK 调到的值, 关闭时返回手动值。"""
        return self.exposure_time
