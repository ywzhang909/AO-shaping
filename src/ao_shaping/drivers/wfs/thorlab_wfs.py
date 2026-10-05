# WFS Python SDK 驱动编写指南 (官方手册 API 全量清单 / ctypes 绑定约定 / 状态位与错误表 / 坑位清单):
#   docs/thorlab-wfs/agent.md
#
# 参考文档 : https://github.com/nvladimus/WFS/blob/master/python/Thorlabs-WFS-read-average-wavefront.ipynb

from __future__ import annotations

import ctypes
import os
import shutil
from copy import deepcopy
from ctypes import (
    byref,
    c_bool,
    c_double,
    c_float,
    c_int32,
    c_uint8,
    c_ulong,
    create_string_buffer,
)
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from functools import wraps
from pathlib import Path
from typing import Any, Self

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import Device, DeviceState, DeviceType
from ao_shaping.drivers.wfs._registry import register_wfs
from ao_shaping.drivers.wfs.base import BaseWFS
from ao_shaping.drivers.wfs._thorlab_wfs import (
    MAX_SPOTS,
    VI_NULL,
    ViInt32,
    ViStatus,
    WfsError,
    load_dll,
    np2c,
)
from ao_shaping.utils.io.device_config import ConfigHandler, DeviceParam, param
from ao_shaping.utils.io.file import ROOT_DIR as PROJECT_ROOT

WFS_DEBUG_MODE = os.environ.get("WFS_DEBUG", "0") == "1"

EXP_TIME_LOW = 0.002
EXP_TIME_HIGH = 86
MAX_AUTOEXPOSE_ATTEMPTS = 10
MAX_MLA_INDICES = 16


def require_take_image(func):
    """装饰器: 保证被包装的方法之前已经调用过 take_image()。

    如果尚未采集图像, 就自动先调用一次 take_image()。

    **文件变换**: 无。它只保证依赖点场的函数执行前, DLL 里已有当前的点场图像
    与质心数据。

    注意:
        自动调用 take_image() 之后会把 ``_image_captured = False`` 置上, 好让函数
        自己在完成图像采集后把它设回 ``True``。
    """

    @wraps(func)
    def wrapper(self: ThorlabWFS, *args, **kwargs):
        if not self._image_captured:
            logger.debug(
                f"{func.__name__} requires take_image() to be called first. "
                f"Automatically calling take_image() now."
            )
            self.take_image()
            self._image_captured = False
        return func(self, *args, **kwargs)

    return wrapper


# define  CAM_RES_1280                  (0) // 1280x1024
# define  CAM_RES_1024                  (1) // 1024x1024
# define  CAM_RES_768                   (2) // 768x768
# define  CAM_RES_512                   (3) // 512x512
# define  CAM_RES_320                   (4) // 320x320 smallest!
class MlaRes(IntEnum):
    Res1280 = 0
    Res1024 = 1
    Res768 = 2
    Res512 = 3
    Res320 = 4

    @classmethod
    def from_str(
        cls, value: str | int | MlaRes, default: MlaRes | None = None
    ) -> MlaRes:
        """把字符串或整数转成 MlaRes 枚举成员。

        参数:
            value: 分辨率字符串 ('512', '768', '1024', '1280', '320')、整数
                   (512, 768, 1024, 1280, 320), 或 MlaRes 实例
            default: 输入非法时返回的默认 MlaRes (默认 None, 即非法时抛异常)

        返回:
            MlaRes 枚举成员

        异常:
            ValueError: value 不是受支持的分辨率, 且未提供 default
        """
        # 若已经是 MlaRes 实例, 原样返回
        if isinstance(value, cls):
            return value

        if isinstance(value, int):
            mapping = {
                1280: cls.Res1280,
                1024: cls.Res1024,
                768: cls.Res768,
                512: cls.Res512,
                320: cls.Res320,
            }
            if value in mapping:
                return mapping[value]
            if default is not None:
                return default
            raise ValueError(
                f"Invalid resolution {value}. Must be one of: 320, 512, 768, 1024, 1280"
            )

        if not isinstance(value, str):
            raise ValueError(
                f"Value must be str, int, or MlaRes, got {type(value).__name__}"
            )

        mapping = {
            "1280": cls.Res1280,
            "1024": cls.Res1024,
            "768": cls.Res768,
            "512": cls.Res512,
            "320": cls.Res320,
        }
        if value in mapping:
            return mapping[value]
        if default is not None:
            return default
        raise ValueError(
            f"Invalid resolution '{value}'. Must be one of: '320', '512', '768', '1024', '1280'"
        )


Mla_pix = {
    MlaRes.Res1280: (1280, 1024),
    MlaRes.Res1024: (1024, 1024),
    MlaRes.Res768: (768, 768),
    MlaRes.Res512: (512, 512),
    MlaRes.Res320: (320, 320),
}


# ── WFS 配置参数 dataclass ──────────────────────────────


def _to_mla_res(v: Any) -> MlaRes:
    """把 int/str/MlaRes 取值转成 MlaRes 枚举。"""
    if isinstance(v, MlaRes):
        return v
    if isinstance(v, str):
        return MlaRes.from_str(v)
    return MlaRes(int(v))


@dataclass
class WFSParams(DeviceParam):
    """WFS 设备参数 dataclass。

    字段定义同时充当:
      - :attr:`config_key` — JSON 配置文件的 key
      - :attr:`attr` — 设备实例的属性名
      - :attr:`cast` — 从 JSON 读入时的类型转换

    Note:
        ``pupil_center`` 和 ``pupil_diameter`` 是复合字段
        (config→tuple, instance→scalar)，在 ``open()`` 中手动处理。
    """
    mla_index: MlaRes = param(default=MlaRes.Res768, cast=_to_mla_res)
    exposure_time: float = param(default=0.0, cast=float, attr="_explosure_time")
    high_speed: bool = param(default=False, cast=bool, attr="enable_high_speed")
    use_custom_ref: bool = param(default=False, cast=bool)


# 模块级单例，所有 ThorlabWFS 实例共用
_WFS_CONFIG_DIR = PROJECT_ROOT / "data" / "wfs_configs"
WFS_CONFIG = ConfigHandler(_WFS_CONFIG_DIR, "wfs", WFSParams)


@register_wfs("thorlab")
class ThorlabWFS(BaseWFS):
    """Thorlabs 波前传感器 (WFS) 设备驱动。

    为 Thorlabs WFS 硬件提供完整控制, 覆盖点场图像采集、波前重建、Zernike 拟合
    以及参考管理。

    属性:
        device_type: DeviceType.WFS
        manufacturer: "Thorlabs"
        model: "WFS"
        version: "1.0.0"
    """

    device_type = DeviceType.WFS
    manufacturer = "Thorlabs"
    model = "WFS"
    version = "1.0.0"

    @classmethod
    def from_params(cls, params: Any, **overrides: Any) -> Self:
        """由驱动参数对象构造一个 Thorlabs WFS。"""
        kwargs = {
            "mla_index": getattr(params, "mla_index", None),
            "exposure_time": getattr(params, "exposure_time", None),
            "high_speed": getattr(params, "high_speed", None),
            "use_custom_ref": getattr(params, "use_custom_ref", None),
            "pupil_diameter": getattr(params, "pupil_diameter", None),
            "pupil_center": getattr(params, "pupil_center", None),
            "stable_sample_enable": getattr(params, "stable_sample_enable", False),
            "stable_sample_n": getattr(params, "stable_sample_n", 5),
            "stable_variance_threshold": getattr(
                params, "stable_variance_threshold", 0.1
            ),
            "stable_max_attempts": getattr(params, "stable_max_attempts", 50),
            "device_id": getattr(params, "device_id", ""),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def __init__(
        self,
        mla_index: MlaRes | str | None = None,
        exposure_time: float | None = None,
        high_speed: bool | None = None,
        use_custom_ref: bool | None = None,
        pupil_diameter: float | None = None,
        pupil_center: tuple | None = None,
        stable_sample_enable: bool = False,
        stable_sample_n: int = 5,
        stable_variance_threshold: float = 0.1,
        stable_max_attempts: int = 50,
        device_id: str = "",
    ):
        """初始化 Thorlabs WFS 驱动。

        参数:
            mla_index: MLA 分辨率 (为 None 时从配置读取)。
                位置参数, 为兼容旧版 WFSManager。
            exposure_time: 曝光时间, 单位 ms; 0 表示自动 (为 None 时从配置读取)。
            high_speed: 启用高速模式 (为 None 时从配置读取)。
            use_custom_ref: 使用自定义参考文件 (为 None 时从配置读取)。
            pupil_diameter: 光瞳直径, 单位 mm (为 None 时从配置读取)。
            pupil_center: (cx, cy) 中心位置 (为 None 时从配置读取)。
            stable_sample_enable: 启用自动稳定采样过滤。
            stable_sample_n: 需要采集的稳定采样数。
            stable_variance_threshold: 判定稳定性的方差阈值。
            stable_max_attempts: 放弃前的最大尝试次数。
            device_id: 设备唯一标识 (传给 Device 基类)。
        """
        super().__init__(device_id)

        # 暂存初始化参数 (推迟到 open())
        if isinstance(mla_index, str):
            mla_index = MlaRes.from_str(mla_index)
        # 用于 ConfigHandler.apply_from_config() 的 init_values
        self._init_values: dict[str, Any] = {
            "mla_index": mla_index,
            "_explosure_time": exposure_time,
            "enable_high_speed": high_speed,
            "use_custom_ref": use_custom_ref,
            "_pupil_center": pupil_center,   # 保留原始 tuple，手动处理
            "_pupil_diameter": pupil_diameter,  # 保留原始 float，手动处理
        }
        self._init_stable_sample_enable: bool = stable_sample_enable
        self._init_stable_sample_n: int = stable_sample_n
        self._init_stable_variance_threshold: float = stable_variance_threshold
        self._init_stable_max_attempts: int = stable_max_attempts

        # 原生 DLL 以惰性方式解析 (见 ``_lib``); 构造驱动不得要求厂商运行时已安装。
        self._wfs_instrument_index = c_int32()
        self.device_name = ""
        self.serial_num = ""
        self._instrument_handle = c_ulong(0)

        # 实例属性 (将在 open() 中更新)
        self.use_custom_ref: bool = False
        self.mla_index: MlaRes = MlaRes.Res768
        self.image_pix = Mla_pix[self.mla_index]
        self.num_spots_x, self.num_spots_y = 0, 0

        self.c_x: float = 0.0
        self.c_y: float = 0.0
        self.d_x: float = 2.0
        self.d_y: float = 2.0

        self._explosure_time: float = 0.0
        self._gain = 1.0
        self.enable_high_speed: bool = False
        self._image_captured = False

        # 稳定采样参数
        self.stable_sample_enable: bool = False
        self.stable_sample_n: int = 5
        self.stable_variance_threshold: float = 0.1
        self.stable_max_attempts: int = 50

        # 注册参数与能力
        self._register_parameters()
        self._register_capabilities()

    @property
    def _lib(self):
        """``WFS_64.dll`` 句柄, 首次使用时加载并缓存。

        此前是在 ``__init__`` 里急切绑定的, 那使得在没有 Thorlabs / VISA 运行时的
        任何地方构造 ``ThorlabWFS`` 都会抛 ``OSError``, 挡住离线工具与驱动自省。
        所有调用点本来就都在读 ``self._lib.<fn>``, 因此把加载推迟到这里既不用改动
        它们, 又把该依赖推迟到了 ``open()``。
        """
        return self._ensure_sdk()

    @staticmethod
    def _load_sdk():
        return load_dll()

    def _register_parameters(self) -> None:
        """注册 WFS 专有参数。"""
        self.register_parameter(
            "exposure_time_ms",
            default_value=0.0,
            min_value=EXP_TIME_LOW,
            max_value=EXP_TIME_HIGH,
            unit="ms",
            description="Camera exposure time in milliseconds",
        )
        self.register_parameter(
            "master_gain",
            default_value=1.0,
            min_value=1.0,
            max_value=24.0,
            unit="",
            description="Master gain for the WFS camera",
        )
        self.register_parameter(
            "high_speed_mode",
            default_value=False,
            unit="",
            description="Enable high speed camera mode",
        )
        self.register_parameter(
            "use_custom_ref",
            default_value=False,
            unit="",
            description="Use custom user reference file",
        )
        self.register_parameter(
            "mla_resolution",
            default_value=MlaRes.Res768,
            unit="",
            description="Microlens array resolution setting",
        )
        self.register_parameter(
            "black_level",
            default_value=c_int32(100),
            min_value=0,
            max_value=255,
            unit="",
            description="Black level offset for camera",
        )
        self.register_parameter(
            "trigger_mode",
            default_value=c_int32(0),
            min_value=0,
            max_value=3,
            unit="",
            description="Camera trigger mode (0=internal, 1-3=external)",
        )

    def _register_capabilities(self) -> None:
        """注册 WFS 能力项。"""
        self.register_capability(
            "measure_wavefront",
            description="Measure wavefront from current spotfield image",
            return_type=np.ndarray,
        )
        self.register_capability(
            "fit_zernike",
            description="Fit Zernike polynomials to measured wavefront",
            return_type=np.ndarray,
        )
        self.register_capability(
            "get_spot_image",
            description="Get current spotfield image",
            return_type=np.ndarray,
        )
        self.register_capability(
            "get_spot_deviations",
            description="Get spot deviations from reference positions",
            return_type=tuple,
        )
        self.register_capability(
            "save_reference",
            description="Save user reference file",
            return_type=Path,
        )
        self.register_capability(
            "load_reference",
            description="Load user reference file",
            return_type=bool,
        )

    # ==================== Device 基类覆写 ====================

    def open(self) -> None:
        """打开 WFS 设备连接并完成初始化。

        异常:
            ConnectionError: WFS 已被占用, 或初始化失败。
        """
        self._set_state(DeviceState.CONNECTING)

        device_in_use = ViInt32()
        device_name = create_string_buffer(256)
        serial_number = create_string_buffer(256)
        resource_name = create_string_buffer(256)
        res = self._lib.WFS_GetInstrumentListInfo(
            VI_NULL(),
            ViInt32(0),
            byref(self._wfs_instrument_index),
            byref(device_in_use),
            device_name,
            serial_number,
            resource_name,
        )
        if res:
            self.handle_error(res)
            return

        # 检查 WFS 是否被占用, 未被占用则连接设备
        assert not device_in_use, (
            "Wavefront sensor currently in use.... closing program"
        )

        self._lib.WFS_init(
            resource_name, c_bool(False), c_bool(True), byref(self._instrument_handle)
        )
        self.device_name = str(device_name.value, encoding="utf8")
        self.serial_num = str(serial_number.value, encoding="utf8")
        logger.info(
            f"Connected to {self.device_name} with Serial Number {self.serial_num}"
        )

        # 加载配置 (init 参数优先, 其次取配置值)
        config = self.load_config()

        # 通过 ConfigHandler 应用简单字段 (mla_index, exposure_time, high_speed, use_custom_ref)
        WFS_CONFIG.apply_from_config(self, config, init_values=self._init_values)

        # 校验曝光时间
        if not (
            self._explosure_time == 0.0
            or EXP_TIME_LOW <= self._explosure_time <= EXP_TIME_HIGH
        ):
            logger.warning(
                f"exp_time {self._explosure_time} out of range, resetting to auto"
            )
            self._explosure_time = 0.0

        # pupil center (复合字段: config→tuple(cx,cy), instance→scalars)
        init_pupil_center = self._init_values.get("_pupil_center")
        if init_pupil_center is not None:
            self.c_x, self.c_y = init_pupil_center
        elif "pupil_center" in config:
            self.c_x, self.c_y = config["pupil_center"]
        else:
            self.c_x, self.c_y = 0.0, 0.0

        # pupil diameter (复合字段: config→tuple(dx,dy), instance→scalars)
        init_pupil_diameter = self._init_values.get("_pupil_diameter")
        if init_pupil_diameter is not None:
            self.d_x, self.d_y = init_pupil_diameter, init_pupil_diameter
        elif "pupil_diameter" in config:
            self.d_x, self.d_y = config["pupil_diameter"]
        else:
            self.d_x, self.d_y = 2.0, 2.0

        # 稳定采样参数
        self.stable_sample_enable = self._init_stable_sample_enable
        self.stable_sample_n = self._init_stable_sample_n
        self.stable_variance_threshold = self._init_stable_variance_threshold
        self.stable_max_attempts = self._init_stable_max_attempts
        if self.stable_sample_enable:
            logger.info(
                f"Stable sample enabled: n={self.stable_sample_n}, "
                f"threshold={self.stable_variance_threshold}, "
                f"max_attempts={self.stable_max_attempts}"
            )

        if self.enable_high_speed:
            logger.info("high speed mode can only use auto exposure time!")

        self.select_mla(self.mla_index)
        self.set_ref_plane(self.use_custom_ref)
        if self._explosure_time <= 0:
            self._explosure_time, _ = self.optimize_exposure_time_and_gain()
        self.exposure_time = self._explosure_time
        self.high_speed = self.enable_high_speed
        self.pupil = (
            (self.c_x, self.c_y, self.d_x, self.d_y)
            if (self.d_x > 0 and self.d_y > 0)
            else self.optimize_pupil()
        )

        self._set_state(DeviceState.READY)
        logger.info(f"WFS device {self.device_id} ready")

    def close(self) -> None:
        """关闭 WFS 设备连接并释放资源。"""
        if self._instrument_handle.value > 0:
            # 保存配置
            self.save_config()
            self.enable_high_speed = False

            self._lib.WFS_close(self._instrument_handle)
            self._instrument_handle = c_ulong(0)

        self._set_state(DeviceState.DISCONNECTED)
        logger.info(f"WFS device {self.device_id} closed")

    def is_connected(self) -> bool:
        """检查 WFS 设备是否已连接且就绪。

        返回:
            仪器句柄有效且状态为 READY 时返回 True。
        """
        return self._instrument_handle.value > 0 and self._state == DeviceState.READY

    def get_hardware_info(self) -> dict:
        """获取硬件相关信息。

        返回:
            含 serial_number、device_name、manufacturer、model 以及固件版本的字典。
        """
        return {
            "serial_number": self.serial_num,
            "device_name": self.device_name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "firmware_version": "N/A",
        }

    # ==================== 配置管理 ====================

    def load_config(self) -> dict:
        """按序列号加载 JSON 配置文件。

        返回:
            配置字典; 无序列号或文件缺失时返回空字典。
        """
        if not self.serial_num:
            logger.warning("WFS serial number not available, skipping config load")
            return {}
        return WFS_CONFIG._manager.load_config(self.serial_num)

    def save_config(self) -> None:
        """把当前参数保存到 JSON 配置文件。

        配置项包括: serial_number、mla_index、exposure_time、high_speed、
        pupil_center、pupil_diameter、use_custom_ref。
        """
        if not self.serial_num:
            logger.warning("WFS serial number not available, skipping config save")
            return

        # 从 WFS_CONFIG 收集已注册的参数字段
        config = WFS_CONFIG.collect(self)
        # 附加复合字段（WFSParams 不包含的 scalar→tuple 转换）
        config["pupil_center"] = (self.c_x, self.c_y)
        config["pupil_diameter"] = (self.d_x, self.d_y)

        WFS_CONFIG._manager.save_config(self.serial_num, config)
        config_file = WFS_CONFIG._manager._get_config_file(self.serial_num)
        logger.info(f"WFS configuration saved: {config_file}")

    # ==================== 错误处理 ====================

    def handle_error(self, err: ViStatus, no_raise: bool = False) -> None:
        """取出并记录错误消息以处理 WFS 错误。

        参数:
            err: WFS 库返回的错误码。
            no_raise: 为 True 时只记录错误, 不抛异常。
        """
        info = create_string_buffer(256)
        self._lib.WFS_error_message(self._instrument_handle, err, byref(info))
        logger.error(f"error: {info.value.decode('utf-8')}")
        if not no_raise:
            raise WfsError(info.value.decode('utf-8'))

    # ==================== MLA 配置 ====================

    def select_mla(self, mla_index: MlaRes) -> None:
        """选择并配置 MLA (微透镜阵列) 全息元件。

        参数:
            mla_index: MLA 分辨率枚举值。

        注意:
            这会重置相机配置并更新 num_spots_x/y。
        """
        self._lib.WFS_SelectMla(self._instrument_handle, 0)
        num_spots_x = c_int32()
        num_spots_y = c_int32()
        self._lib.WFS_ConfigureCam(
            self._instrument_handle,
            c_int32(0),
            c_int32(mla_index.value),
            byref(num_spots_x),
            byref(num_spots_y),
        )
        self.mla_index = mla_index
        self.image_pix = Mla_pix[mla_index]
        self.num_spots_x, self.num_spots_y = num_spots_x.value, num_spots_y.value
        logger.info(
            f"Number of detectable spots in X: {num_spots_x.value} \n"
            + f"Number of detectable spots in Y: {num_spots_y.value}"
        )

    def set_ref_plane(self, custom: bool) -> None:
        """设置波前测量的参考平面。

        **DLL 调用** (``custom=True`` 时的顺序):
        1. ``WFS_SetReferencePlane(handle, 1)`` → 切到自定义参考平面
        2. ``WFS_LoadUserRefFile(handle)`` → 从 DLL 管理的目录加载 ``.ref`` 文件

        **DLL 调用** (``custom=False`` 时):
        1. ``WFS_SetReferencePlane(handle, 0)`` → 切回出厂默认

        **文件变换**:
        - ``custom=True`` 时:
          DLL 读取 ``<Ref_Dir>/<filename>.ref`` (由 ``save_user_ref()`` 创建)。
          不写入也不修改任何文件。
        - ``custom=False`` 时:
          无文件操作, 只改变 DLL 内部状态。

        **回退行为**:
        若 ``WFS_LoadUserRefFile`` 失败 (尚无 ``.ref`` 文件), 则带警告回退到默认参考,
        而不是抛异常。

        参数:
            custom: 为 True 时加载自定义用户参考文件, 否则使用默认参考。
        """
        _select = 1 if custom else 0
        if err := self._lib.WFS_SetReferencePlane(
            self._instrument_handle, c_int32(_select)
        ):
            self.handle_error(err, no_raise=True)
            logger.warning(
                "WFS_SetReferencePlane failed, falling back to default reference"
            )
            self.use_custom_ref = False
            return

        if custom:
            if err := self._lib.WFS_LoadUserRefFile(self._instrument_handle):
                self.handle_error(err, no_raise=True)
                logger.warning(
                    "No user reference file available (use save_user_ref() first). "
                    "Falling back to default reference."
                )
                # 回退到默认参考
                self._lib.WFS_SetReferencePlane(self._instrument_handle, c_int32(0))
                self.use_custom_ref = False
            else:
                self.use_custom_ref = True
        else:
            self.use_custom_ref = False

    @require_take_image
    def create_default_user_ref(self) -> bool:
        """由当前点场图像创建默认用户参考。

        **DLL 调用**:
        1. ``WFS_CreateDefaultUserReference(handle)`` —— 由最近一次采集的点场图像
           计算参考斑点位置。

        **文件变换**: 无。
        参考只写入 DLL 内部内存。若要持久化到磁盘, 需随后调用 ``save_user_ref()``
        (它会写出 ``.ref`` 文件)。

        **典型流程**:
        ``take_image()`` → ``create_default_user_ref()`` → ``save_user_ref()``

        返回:
            成功返回 True, 失败返回 False。
        """
        if err := self._lib.WFS_CreateDefaultUserReference(self._instrument_handle):
            self.handle_error(err, no_raise=True)
            logger.error("Failed to create default user reference")
            return False
        logger.info("Default user reference created from current spotfield image")
        return True

    def get_mla_name(self) -> str:
        """获取当前选中的 MLA 名称 (如 'MLA150M-5C')。

        调用 ``WFS_GetMlaData(handle, 0, ...)`` 取回微透镜阵列的名称, 它对应初始化
        期间由 ``WFS_SelectMla(handle, 0)`` 选中的那个阵列。

        .. important::

           ``WFS_GetMlaData`` 的 ``MLAIndex`` 参数**不是**相机分辨率代码
           (0=1280, 1=1024, 2=768, 3=512, 4=320), 而是 MLA 的**选择索引**
           (0 到 ``MLACount-1``), 其取值与传给 ``WFS_SelectMla`` 的相同。
           由于驱动始终用 ``WFS_SelectMla(handle, 0)`` 选中 MLA 0, 这里传索引 0。

        返回:
            MLA 名称字符串 (如 ``"MLA150M-5C"``); DLL 调用失败时返回空字符串。
        """
        mla_name_buffer = create_string_buffer(256)
        err = self._lib.WFS_GetMlaData(
            self._instrument_handle,
            c_int32(0),  # MLA index = 0 (选中的 MLA)
            mla_name_buffer,
            byref(c_double()),
            byref(c_double()),
            byref(c_double()),
            byref(c_double()),
            byref(c_double()),
            byref(c_double()),
            byref(c_double()),
        )
        if err != 0:
            self.handle_error(err, no_raise=True)
            name = self._try_get_mla_name_fallback()
            if name:
                logger.warning(
                    f"WFS_GetMlaData failed for index 0, "
                    f"falling back to alternative index: '{name}'"
                )
            return name
        name = mla_name_buffer.value
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        return name.strip("\x00").strip()

    def _try_get_mla_name_fallback(self) -> str:
        """尝试用其他 MLA 索引 (1, 2, ... 直到 15) 取得名称。

        若主 MLA 索引 (0) 不可用, 这里作为兜底向上搜索更多索引。WFS 通常只有
        1-2 个已标定的 MLA。

        返回:
            MLA 名称字符串; 全部尝试失败时返回空字符串。
        """
        for alt_idx in range(1, MAX_MLA_INDICES):  # 尝试索引 1..MAX_MLA_INDICES-1
            buf = create_string_buffer(256)
            err = self._lib.WFS_GetMlaData(
                self._instrument_handle,
                c_int32(alt_idx),
                buf,
                byref(c_double()),
                byref(c_double()),
                byref(c_double()),
                byref(c_double()),
                byref(c_double()),
                byref(c_double()),
                byref(c_double()),
            )
            if err == 0:
                name = buf.value
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                name = name.strip("\x00").strip()
                if name:
                    logger.warning(
                        f"_try_get_mla_name_fallback found name '{name}' at index {alt_idx}"
                    )
                    return name
        return ""

    @staticmethod
    def _get_ref_default_dir() -> Path:
        """获取 WFS 默认参考文件目录。

        Thorlabs WFS DLL 把参考文件保存/加载到:
            C:\\Users\\<user>\\Documents\\Thorlabs\\Wavefront Sensor\\Reference

        在 Linux 上回退到 ~/.local/share/Thorlabs/WFS/Ref 或项目 data 目录。

        返回:
            参考目录的 Path 对象。
        """
        import platform

        system = platform.system()

        if system == "Windows":
            # 用 Path.home(), 它能可靠地解析出 C:\Users\<username>
            # home() 失败时回退到 USERPROFILE
            home = Path.home()
            if not home or not home.exists():
                user_profile = os.environ.get("USERPROFILE")
                if user_profile:
                    home = Path(user_profile)
                else:
                    logger.warning("Cannot determine user home directory")
                    home = Path("C:/Users/Public")
            return home / "Documents" / "Thorlabs" / "Wavefront Sensor" / "Reference"
        else:
            # Linux/macOS 回退: 用 XDG_DATA_HOME 或项目 data 目录
            xdg_data = os.environ.get("XDG_DATA_HOME", "")
            if xdg_data:
                base = Path(xdg_data)
            else:
                base = Path.home() / ".local" / "share"

            ref_dir = base / "Thorlabs" / "WFS" / "Ref"
            logger.debug(f"[WFS _get_ref_default_dir] Linux fallback: {ref_dir}")

            # 若不可写, 回退到项目 data/calibration
            project_fallback = Path("data") / "calibration" / "wfs_ref"
            if not ref_dir.parent.exists():
                logger.debug(
                    f"[WFS _get_ref_default_dir] XDG path not accessible, using project fallback: {project_fallback}"
                )
                return project_fallback

            return ref_dir

    def _get_ref_filename(self, fallback_if_empty: bool = True) -> str:
        """构造 WFS DLL 期望的 ``.ref`` 文件名。

        **文件名格式** (来自 Thorlabs SDK):
            ``WFS_<serial_number>_<mla_name>_<cam_resol_idx>.ref``

        该文件名同时被 ``save_user_ref()`` (用于知道 DLL 把文件写到了哪里) 与
        ``load_user_ref()`` (用于定位要复制或读取的文件) 使用, 与 DLL 内部期望一致。

        **无文件变换** —— 纯字符串拼接。

        参数:
            fallback_if_empty: 为 True 且 ``mla_name`` 为空时, 用 ``"unknown"``
                作占位符, 以保证文件名仍然合法。

        返回:
            形如 ``WFS_M00224955_MLA150M-5C_0.ref`` 的文件名字符串;
            MLA 名称不可得时为 ``WFS_M00224955_unknown_3.ref``。
        """
        mla_name = self.get_mla_name()
        if not mla_name and fallback_if_empty:
            mla_name = "unknown"
            logger.warning(
                f"Cannot determine MLA name (WFS_GetMlaData failed), "
                f"using fallback filename with 'unknown': "
                f"WFS_{self.serial_num}_unknown_{int(self.mla_index)}.ref"
            )
        # cam_resol_idx 就是 MLA 分辨率索引 (self.mla_index 的值)
        cam_resol_idx = int(self.mla_index)
        filename = f"WFS_{self.serial_num}_{mla_name}_{cam_resol_idx}.ref"
        return filename

    @require_take_image
    def save_user_ref(self, backup_dir: str | Path | None = None) -> Path | None:
        """保存用户参考文件并创建带时间戳的备份。

        **DLL 调用** (按顺序):
        1. ``WFS_SetSpotsToUserReference(handle)`` —— 把当前斑点质点提升为参考
           (只存在于 DLL 内部内存)。
        2. ``WFS_SaveUserRefFile(handle)`` —— 把 ``.ref`` 文件写到 DLL 管理的磁盘位置。

        **文件变换** (逐步):
        ::

            ┌─ DLL 写入:  <Ref_Dir>/<filename>.ref        (新建)
            └─ shutil.copy2 →  <backup_dir>/<filename>_<ts>.ref (新建)

        - **Ref_Dir** = ``%USERPROFILE%/Documents/Thorlabs/Wavefront Sensor/Reference``
        - **filename** = ``WFS_{serial}_{mla_name}_{res_index}``
        - 若无法确定 MLA 名称 (``WFS_GetMlaData`` 失败), 文件名中用 ``"unknown"`` 兜底。
        - 若 DLL 保存后在预期路径下找不到文件名, 就在参考目录里搜索任意
          ``WFS_{serial}_*.ref`` 文件作为兜底。
        - DLL 管理的那份副本在下一次保存时可能被静默覆盖; 带时间戳的备份保留历史。

        参数:
            backup_dir: 存放备份的目录。默认 ``data/calibration/``。

        返回:
            带时间戳的备份文件路径; 失败时返回 None。
        """
        if backup_dir is None:
            backup_dir = Path("data/calibration")
        else:
            backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        # 保存前显式把当前斑点位置设为参考。
        # 少了这次调用, WFS_SaveUserRefFile 可能无内容可存, 导致随后加载时报
        # "No User Reference available!" 错误。
        if err := self._lib.WFS_SetSpotsToUserReference(self._instrument_handle):
            self.handle_error(err, no_raise=True)
            return None

        # 调用 DLL 把参考文件保存到它的默认位置
        if err := self._lib.WFS_SaveUserRefFile(self._instrument_handle):
            self.handle_error(err, no_raise=True)
            return None

        # 取源文件路径 (DLL 保存的位置)
        ref_dir = self._get_ref_default_dir()
        ref_filename = self._get_ref_filename()
        src_path = ref_dir / ref_filename

        if not src_path.exists():
            # 兜底: 在 ref_dir 中搜索与序列号匹配的任何 .ref 文件。
            # MLA 名称不可得时 DLL 可能选了别的文件名。
            logger.warning(
                f"Reference file not found at expected path: {src_path}\n"
                f"Searching {ref_dir} for files matching serial {self.serial_num}..."
            )
            candidates = []
            if ref_dir.exists():
                pattern = f"WFS_{self.serial_num}_*.ref"
                candidates = sorted(ref_dir.glob(pattern))
            if candidates:
                src_path = candidates[0]
                logger.info(f"Found alternative reference file: {src_path}")
            else:
                logger.error(
                    f"No .ref file found for serial {self.serial_num} in {ref_dir}. "
                    f"The DLL (WFS_SaveUserRefFile) may have saved to a different location."
                )
                return None

        # 生成带时间戳的备份文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{src_path.stem}_{timestamp}{src_path.suffix}"
        backup_path = backup_dir / backup_filename

        # 复制到备份位置
        try:
            shutil.copy2(src_path, backup_path)
            logger.info(f"User reference saved to DLL location: {src_path}")
            logger.info(f"Backup created at: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None

    def load_user_ref(self, backup_path: str | Path | None = None) -> bool:
        """从备份或默认位置加载用户参考文件。

        **DLL 调用** (按顺序):
        1. ``WFS_LoadUserRefFile(handle)`` —— 从 DLL 目录加载 ``.ref`` 文件
        2. ``WFS_SetReferencePlane(handle, 1)`` —— 切到自定义参考平面

        **文件变换** (逐步):

        **提供了 ``backup_path``** 时::
        ::

            shutil.copy2(backup_path → <Ref_Dir>/<filename>.ref)  (已复制)
            DLL 读取 ←  <Ref_Dir>/<filename>.ref                  (读取)

        **未提供 ``backup_path``** 时::
        ::

            DLL 读取 ←  <Ref_Dir>/<filename>.ref                  (读取)

        - **Ref_Dir** = ``%USERPROFILE%/Documents/Thorlabs/Wavefront Sensor/Reference``
        - 若给了 ``backup_path``, 会先把文件复制到 DLL 期望的位置, 好让
          ``WFS_LoadUserRefFile`` 能找到它。
        - 若预期路径下没有该文件, DLL 会返回错误 (此前是 "No User Reference
          available!") —— 见 ``save_user_ref()`` 如何创建它。

        参数:
            backup_path: 要加载的备份 ``.ref`` 文件路径, 可选。
                若提供, 会先复制到 DLL 管理的目录, 再调用 ``WFS_LoadUserRefFile``。

        返回:
            成功返回 True, 失败返回 False。
        """
        # 调试: 打印当前系统信息
        import platform

        logger.debug(f"[WFS load_user_ref] Platform: {platform.system()}")
        logger.debug(
            f"[WFS load_user_ref] USERPROFILE: {os.environ.get('USERPROFILE', 'NOT_SET')}"
        )

        if backup_path is not None:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            # 把备份复制到 DLL 期望的位置
            ref_dir = self._get_ref_default_dir()
            logger.debug(f"[WFS load_user_ref] ref_dir: {ref_dir}")
            logger.debug(f"[WFS load_user_ref] ref_dir exists: {ref_dir.exists()}")
            logger.debug(
                f"[WFS load_user_ref] ref_dir parent exists: {ref_dir.parent.exists() if ref_dir.parent != ref_dir else 'N/A'}"
            )

            ref_dir.mkdir(parents=True, exist_ok=True)
            ref_filename = self._get_ref_filename()
            dst_path = ref_dir / ref_filename
            logger.debug(f"[WFS load_user_ref] dst_path: {dst_path}")

            try:
                shutil.copy2(backup_path, dst_path)
                logger.info(f"Copied backup {backup_path} to {dst_path}")
            except Exception as e:
                logger.error(f"Failed to copy backup file: {e}")
                return False

        # 调用 DLL 从它的默认位置加载参考文件
        logger.debug("[WFS load_user_ref] Calling WFS_LoadUserRefFile...")
        if err := self._lib.WFS_LoadUserRefFile(self._instrument_handle):
            self.handle_error(err, no_raise=True)
            logger.error("Failed to load user reference file via WFS_LoadUserRefFile")
            # 附加调试: 列出 DLL 期望的参考文件路径
            ref_dir = self._get_ref_default_dir()
            ref_filename = self._get_ref_filename()
            expected_path = ref_dir / ref_filename
            logger.debug(f"[WFS load_user_ref] Expected path: {expected_path}")
            logger.debug(
                f"[WFS load_user_ref] Expected path exists: {expected_path.exists()}"
            )
            if not expected_path.exists():
                # 列出目录内容以帮助调试
                try:
                    files = list(ref_dir.glob("*")) if ref_dir.exists() else []
                    logger.debug(f"[WFS load_user_ref] Files in ref_dir: {files}")
                except Exception as list_err:
                    logger.debug(f"[WFS load_user_ref] Cannot list ref_dir: {list_err}")
            return False

        # 同时更新参考平面设置
        logger.debug("[WFS load_user_ref] Setting reference plane to custom...")
        if err := self._lib.WFS_SetReferencePlane(self._instrument_handle, c_int32(1)):
            self.handle_error(err, no_raise=True)
            logger.warning("WFS_SetReferencePlane returned error but continuing")

        self.use_custom_ref = True
        logger.info("User reference file loaded successfully")
        return True

    # ==================== 光瞳 / 图像采集 ====================

    def optimize_pupil(self) -> tuple[float, float, float, float]:
        """优化光瞳检测。

        本函数由当前点场数据计算光束质心与直径。

        注意:
            **只计算并返回, 不写回设备** —— 调用方必须显式
            ``wfs.pupil = wfs.optimize_pupil()`` 才会调用 ``WFS_SetPupil`` 生效。
            硬编码/未写回的 pupil 与真实光束不符时, 边界无效子孔径会污染
            ``WFS_ZernikeLsf`` 全孔径 LSF 拟合 → 巨大的假 tip/tilt (2026-09 实测
            (0,0,8mm) 硬编码产生 |z|=4.6~12.8λ, 自动 pupil ≈(±0.15, ±3.7)mm 后
            恢复 0.006~0.217λ 并线性度 R²=0.9603)。

        返回:
            tuple[float, float, float, float]: 光束质心 x、光束质心 y、
                光束直径 x、光束直径 y
        """
        assert not self.enable_high_speed, "turn off high speed mode first"
        self._lib.WFS_CalcSpotsCentrDiaIntens(
            self._instrument_handle, c_int32(1), c_int32(1)
        )
        beam_centroid_x = c_double()
        beam_centroid_y = c_double()
        beam_diameter_x = c_double()
        beam_diameter_y = c_double()
        self._lib.WFS_CalcBeamCentroidDia(
            self._instrument_handle,
            byref(beam_centroid_x),
            byref(beam_centroid_y),
            byref(beam_diameter_x),
            byref(beam_diameter_y),
        )
        cx_v, cy_v, dx_v, dy_v = (
            beam_centroid_x.value,
            beam_centroid_y.value,
            beam_diameter_x.value,
            beam_diameter_y.value,
        )
        # 防御性检查 (2026-09 实测固化): 光斑场不可用/未对准时 DLL 可能返回
        # 非有限值或 0 直径; 直接写回会污染后续 WFS_ZernikeLsf 拟合. 只报警, 不改语义.
        if (
            not np.isfinite([cx_v, cy_v, dx_v, dy_v]).all()
            or dx_v <= 0
            or dy_v <= 0
            or dx_v > 12.0
            or dy_v > 12.0
        ):
            logger.warning(
                "optimize_pupil() returned suspicious beam "
                "center=({:.3f},{:.3f})mm diameter=({:.3f},{:.3f})mm — "
                "请确认曝光/对准后重跑; 勿将欠佳 pupil 写回设备",
                cx_v, cy_v, dx_v, dy_v,
            )
        return (cx_v, cy_v, dx_v, dy_v)

    def take_image(self, n_sample: int = 10, dynamicNoiseCut: bool = True) -> None:
        """采集点场图像并计算斑点质心/直径/强度。

        参数:
            n_sample: 自动曝光模式下采集的采样张数 (exposure_time <= 0 时使用)。
            dynamicNoiseCut: 是否为斑点计算启用动态噪声底截止。

        注意:
            采集成功后把 self._image_captured 标志置为 True。
            固定曝光时只拍一张; 自动曝光时迭代 n_sample 次。
        """
        if self._explosure_time > 0:
            if err := self._lib.WFS_TakeSpotfieldImage(self._instrument_handle):
                self.handle_error(err)
            else:
                self._image_captured = True
        else:
            # 无固定曝光时间, 用多次采样的自动曝光
            actual_exposure = c_double()
            actual_gain = c_double()
            for _ in range(n_sample):
                self._lib.WFS_TakeSpotfieldImageAutoExpos(
                    self._instrument_handle, byref(actual_exposure), byref(actual_gain)
                )
            self._image_captured = True

        # 计算斑点质心、直径与强度
        if res := self._lib.WFS_CalcSpotsCentrDiaIntens(
            self._instrument_handle, c_int32(1 if dynamicNoiseCut else 0), c_int32(0)
        ):
            self.handle_error(res)

    @require_take_image
    def get_spotfiled_image(self) -> np.ndarray:
        """从 WFS 设备取出已采集的点场图像。

        返回:
            np.ndarray: 二维 uint8 图像数组 (行 × 列)。

        异常:
            RuntimeError: WFS_GetSpotfieldImageCopy 失败。
        """
        # 按手册给出的最大图像尺寸分配缓冲区
        # (CAM_MAXPIX_X * CAM_MAXPIX_Y; 1280×1024 是最大的 MLA)
        max_h, max_w = 1024, 1280
        image_buf = np.empty((max_h, max_w), dtype=np.uint8)
        rows = c_int32()
        cols = c_int32()
        if err := self._lib.WFS_GetSpotfieldImageCopy(
            self._instrument_handle,
            image_buf.ctypes.data_as(ctypes.POINTER(c_uint8)),
            byref(rows),
            byref(cols),
        ):
            raise RuntimeError(self.handle_error(err))
        # 只返回有效区域 (行 × 列)
        return image_buf[: rows.value, : cols.value]

    @require_take_image
    def get_spots_statics(self) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """获取斑点强度与质心位置。

        返回:
            含以下内容的 tuple:
                - np.ndarray: 形状为 (num_spots_x, num_spots_y) 的斑点强度数组
                - tuple[np.ndarray, np.ndarray]: (centroid_x, centroid_y) 数组,
                  每个形状都是 (num_spots_x, num_spots_y)

        注意:
            要求已关闭高速模式。
            活动光瞳区域之外的斑点值为 NaN。
        """
        assert not self.enable_high_speed, "turn off high speed mode first"
        spots_intensities = np.empty(MAX_SPOTS, dtype=np.float32)
        spots_center_x = np.empty(MAX_SPOTS, dtype=np.float32)
        spots_center_y = np.empty(MAX_SPOTS, dtype=np.float32)
        if err := self._lib.WFS_CalcSpotsCentrDiaIntens(
            self._instrument_handle, ViInt32(0), ViInt32(1)
        ):
            self.handle_error(err)
        else:
            # spots_diameter_x, spots_diameter_y = spots_intensities.copy(), spots_intensities.copy()
            self._lib.WFS_GetSpotIntensities(self._instrument_handle, spots_intensities)
            self._lib.WFS_GetSpotCentroids(
                self._instrument_handle, spots_center_x, spots_center_y
            )
            # self._lib.WFS_GetSpotDiameters(self._instrument_handle,
            #     np2c(spots_diameter_x), np2c(spots_diameter_y))
        return spots_intensities[: self.num_spots_x, : self.num_spots_y], (
            spots_center_x[: self.num_spots_x, : self.num_spots_y],
            spots_center_y[: self.num_spots_x, : self.num_spots_y],
        )

    def build_subaperture_mask(
        self,
        n_avg: int = 30,
        threshold_ratio: float = 0.3,
        edge_clip: int = 1,
        plot: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """为 WFS40-5C 构建有效子孔径掩码。

        参数:
            n_avg: 为抑制噪声而平均的帧数。
            threshold_ratio: 以最大强度为基准的强度阈值比例 (典型 0.2-0.4)。
            edge_clip: 从边缘裁掉的透镜行/列数 (推荐 1-2)。
            plot: 为 True 时显示可视化。

        返回:
            tuple: (mask_bool, valid_indices_flat)
                - mask_bool: 形状 (num_spots_x, num_spots_y) 的二维布尔数组
                - valid_indices_flat: 有效子孔径展平索引构成的一维数组
        """
        from scipy import ndimage

        intensities = []
        for _ in range(n_avg):
            self.take_image(n_sample=1, dynamicNoiseCut=True)
            spots_intensities = np.empty(MAX_SPOTS, dtype=np.float32)
            self._lib.WFS_GetSpotIntensities(self._instrument_handle, spots_intensities)
            int_mat = spots_intensities[: self.num_spots_x, : self.num_spots_y]
            intensities.append(int_mat)

        int_mean = np.mean(intensities, axis=0)
        int_max = np.max(int_mean)
        threshold = int_max * threshold_ratio

        mask = int_mean > threshold

        if edge_clip > 0:
            mask[:edge_clip, :] = False
            mask[-edge_clip:, :] = False
            mask[:, :edge_clip] = False
            mask[:, -edge_clip:] = False

        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))

        valid_flat = np.where(mask.flatten())[0]

        logger.info(
            f"Valid subapertures: {np.sum(mask)}/{mask.size} "
            f"({np.sum(mask) / mask.size * 100:.1f}%), "
            f"threshold={threshold:.1f} (max={int_max:.1f})"
        )

        if plot:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(int_mean, cmap="hot")
            axes[0].set_title("Mean Intensity")
            axes[1].imshow(mask, cmap="gray")
            axes[1].set_title("Valid Mask")
            overlay = np.dstack([int_mean / int_max] * 3)
            overlay[np.logical_not(mask)] = [0, 0, 1]
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay (invalid=blue)")
            plt.tight_layout()
            plt.show()

        return mask, valid_flat

    def _get_stable_samples(
        self,
        n_samples: int,
        variance_threshold: float,
        max_attempts: int,
    ) -> list[np.ndarray]:
        """按方差阈值采集稳定样本。

        参数:
            n_samples: 要采集的稳定样本数。
            variance_threshold: 判定样本稳定所允许的最大方差。
            max_attempts: 放弃前的最大尝试次数。

        返回:
            稳定样本数组的列表。
        """
        stable_samples = []
        attempts = 0

        while len(stable_samples) < n_samples and attempts < max_attempts:
            self.take_image(n_sample=1, dynamicNoiseCut=True)
            self._lib.WFS_CalcSpotToReferenceDeviations(
                self._instrument_handle, c_int32(0)
            )

            wavefront = np.empty(MAX_SPOTS, dtype=c_float)
            self._lib.WFS_CalcWavefront(
                self._instrument_handle,
                ViInt32(0),
                ViInt32(0),
                wavefront,
            )
            wf_slice = wavefront[: self.num_spots_x, : self.num_spots_y]

            variance = float(np.var(wf_slice))
            if variance < variance_threshold:
                stable_samples.append(wf_slice)
                logger.debug(
                    f"Stable sample {len(stable_samples)}/{n_samples}: "
                    f"variance={variance:.6f} < {variance_threshold}"
                )
            else:
                logger.debug(
                    f"Unstable sample rejected: variance={variance:.6f} >= {variance_threshold}"
                )

            attempts += 1

        if len(stable_samples) < n_samples:
            logger.warning(
                f"Only collected {len(stable_samples)}/{n_samples} stable samples "
                f"after {attempts} attempts"
            )

        return stable_samples

    # ==================== 波前测量 ====================

    @require_take_image
    def get_wavefront(self, cancel_tile: bool = False) -> tuple[np.ndarray, dict]:
        """由斑点偏差计算波前。

        参数:
            cancel_tile: 为 True 时从波前测量中去掉 tip/tilt。

        返回:
            tuple[np.ndarray, dict]: (波前数组, 统计量字典)
                - wavefront: 波前值二维数组, 单位 waves
                - statistics: 键为 'min'、'max'、'diff'、'mean'、'rms'、'wighted_rms' 的 dict

        注意:
            使用实测波前 (type=0); 若光瞳已定义, 则带自适应光瞳补偿。
            stable_sample_enable 为 True 时采集多个稳定样本并返回其均值。
        """
        # 若启用了稳定采样则使用它
        if self.stable_sample_enable:
            stable_wfs = self._get_stable_samples(
                self.stable_sample_n,
                self.stable_variance_threshold,
                self.stable_max_attempts,
            )
            if stable_wfs:
                wavefront = np.mean(stable_wfs, axis=0)
            else:
                return np.zeros(
                    (self.num_spots_x, self.num_spots_y), dtype=np.float32
                ), {
                    "min": np.nan,
                    "max": np.nan,
                    "diff": np.nan,
                    "mean": np.nan,
                    "rms": np.nan,
                    "wighted_rms": np.nan,
                }
        else:
            if res := self._lib.WFS_CalcSpotToReferenceDeviations(
                self._instrument_handle, c_int32(1 if cancel_tile else 0)
            ):
                self.handle_error(res)
            adaptive_pupil = 0 if (self.d_x and self.d_y) else 1
            wavefront = np.empty(MAX_SPOTS, dtype=c_float)
            wavefront_type = 0
            if err := self._lib.WFS_CalcWavefront(
                self._instrument_handle,
                ViInt32(wavefront_type),
                ViInt32(adaptive_pupil),
                wavefront,
            ):
                self.handle_error(err)
                wavefront = np.zeros(
                    (self.num_spots_x, self.num_spots_y), dtype=np.float32
                )
            else:
                wavefront = deepcopy(wavefront)[: self.num_spots_x, : self.num_spots_y]

        # 由波前计算统计量 (无论波前是怎么得到的)
        min_val, max_val, diff_val, mean_val = (
            c_double(),
            c_double(),
            c_double(),
            c_double(),
        )
        rms_val, wighted_rms_val = c_double(), c_double()
        self._lib.WFS_CalcWavefrontStatistics(
            self._instrument_handle,
            byref(min_val),
            byref(max_val),
            byref(diff_val),
            byref(mean_val),
            byref(rms_val),
            byref(wighted_rms_val),
        )

        if np.all(wavefront == 0):
            logger.warning(
                "WFS_CalcWavefront returned zero-filled buffer — DLL may not have written data"
            )

        if WFS_DEBUG_MODE:
            wf_variance = np.var(wavefront)
            wf_std = np.std(wavefront)
            logger.debug(
                f"WFS wavefront stats: var={wf_variance:.6f}, std={wf_std:.6f}, shape={wavefront.shape}"
            )

        return wavefront, {
            "min": min_val.value
            if not self.stable_sample_enable
            else float(np.min(wavefront)),
            "max": max_val.value
            if not self.stable_sample_enable
            else float(np.max(wavefront)),
            "diff": diff_val.value
            if not self.stable_sample_enable
            else float(np.max(wavefront) - np.min(wavefront)),
            "mean": mean_val.value
            if not self.stable_sample_enable
            else float(np.mean(wavefront)),
            "rms": rms_val.value
            if not self.stable_sample_enable
            else float(np.std(wavefront)),
            "wighted_rms": wighted_rms_val.value
            if not self.stable_sample_enable
            else float(np.std(wavefront)),
        }

    def _remove_tilt(self, wavefront: np.ndarray) -> np.ndarray:
        """通过拟合平面并减去它来去除波前中的倾斜 (tip/tilt)。

        参数:
            wavefront: 二维波前数组。

        返回:
            去掉倾斜后的波前。
        """
        # 建立坐标网格 (归一化到 [-1, 1] 以保证数值稳定性)
        ny, nx = wavefront.shape
        y, x = np.meshgrid(
            np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="ij"
        )

        # 展平以便拟合
        z = wavefront.flatten()

        # 构造平面的设计矩阵: z = a*x + b*y + c
        A = np.column_stack([x.flatten(), y.flatten(), np.ones_like(x.flatten())])

        # 用最小二乘解出系数
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
            a, b, c = coeffs

            # 构造倾斜平面
            tilt_plane = a * x + b * y + c

            # 减去倾斜
            wavefront_no_tilt = wavefront - tilt_plane
        except Exception:
            # 拟合失败就原样返回
            logger.warning("Failed to fit tilt plane, returning original wavefront")
            return wavefront

        return wavefront_no_tilt

    @staticmethod
    def calc_n_zernike_terms(n: int) -> int:
        """计算给定阶数下的 Zernike 项数。

        参数:
            n: Zernike 阶数。

        返回:
            Zernike 项数。
        """
        return (n + 1) * (n + 2) // 2 + 1

    @require_take_image
    def get_zernike(self, zernike_order: int = 10) -> np.ndarray:
        """由斑点偏差计算 Zernike 多项式系数。

        参数:
            zernike_order: Zernike 阶数 (最大 10, 从 0 起索引)。

        返回:
            np.ndarray: Zernike 系数数组。

        异常:
            AssertionError: zernike_order 超过 10。

        注意:
            - 结果依赖 pupil 正确性: 调用前必须 ``wfs.pupil = wfs.optimize_pupil()``。
              硬编码 pupil 与光束不符时, 边界无效子孔径会污染 LSF 拟合, 产生巨大的
              假 tip/tilt 系数 (2026-09 实测: |z|=4.6~12.8λ → pupil 修正后 ≤0.22λ)。
            - 系数为 Noll 1976 约定 (索引 1 起, 前 66 项), 输出单位 µm (µm/0.532 = λ @532nm),
              RoC = coeff[5]。
        """
        assert zernike_order <= 10, (
            f"zernike order must be less than or equal to 10, got {zernike_order}"
        )
        roc_mm = c_double()
        coeff_num = self.calc_n_zernike_terms(zernike_order)
        zernike_order_c = c_int32(zernike_order)
        zernike_um = np.empty((coeff_num,), c_float)
        zernike_orders_rms_um = np.empty((zernike_order + 1,), c_float)

        if res := self._lib.WFS_CalcSpotToReferenceDeviations(
            self._instrument_handle, c_int32(0)
        ):
            self.handle_error(res)

        if err := self._lib.WFS_ZernikeLsf(
            self._instrument_handle,
            byref(zernike_order_c),
            zernike_um.ctypes.data_as(ctypes.POINTER(c_float)),
            zernike_orders_rms_um.ctypes.data_as(ctypes.POINTER(c_float)),
            byref(roc_mm),
        ):
            self.handle_error(err)
            return np.empty_like(zernike_um)
        return zernike_um

    # ==================== 斑点偏差 ====================

    @require_take_image
    def get_spot_deviation(
        self, cancel_tile: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """获取斑点相对参考位置的偏差。

        参数:
            cancel_tile: 为 True 时从偏差中去掉 tip/tilt。

        返回:
            tuple[np.ndarray, np.ndarray]: (deviation_x, deviation_y) 数组,
               每个形状都是 (num_spots_x, num_spots_y)。

        注意:
            stable_sample_enable 为 True 时采集多个稳定样本并返回其均值。
        """
        # 若启用了稳定采样则使用它
        if self.stable_sample_enable:
            stable_x_list = []
            stable_y_list = []
            attempts = 0

            while (
                len(stable_x_list) < self.stable_sample_n
                and attempts < self.stable_max_attempts
            ):
                self.take_image(n_sample=1, dynamicNoiseCut=True)

                _spots_dev_x = np.empty(MAX_SPOTS, dtype=np.float32)
                _spots_dev_y = np.empty(MAX_SPOTS, dtype=np.float32)

                if (
                    res := self._lib.WFS_CalcSpotToReferenceDeviations(
                        self._instrument_handle, c_int32(1 if cancel_tile else 0)
                    )
                ) == 0:
                    if err := self._lib.WFS_GetSpotDeviations(
                        self._instrument_handle, _spots_dev_x, _spots_dev_y
                    ):
                        self.handle_error(err)
                else:
                    self.handle_error(res)
                    continue

                dev_x = _spots_dev_x[: self.num_spots_x, : self.num_spots_y]
                dev_y = _spots_dev_y[: self.num_spots_x, : self.num_spots_y]

                # 以方差作为稳定性度量
                variance = float(np.var(dev_x) + np.var(dev_y))
                if variance < self.stable_variance_threshold:
                    stable_x_list.append(dev_x)
                    stable_y_list.append(dev_y)
                    logger.debug(
                        f"Stable deviation sample {len(stable_x_list)}/{self.stable_sample_n}: "
                        f"variance={variance:.6f} < {self.stable_variance_threshold}"
                    )
                else:
                    logger.debug(
                        f"Unstable deviation sample rejected: variance={variance:.6f} >= {self.stable_variance_threshold}"
                    )

                attempts += 1

            if len(stable_x_list) < self.stable_sample_n:
                logger.warning(
                    f"Only collected {len(stable_x_list)}/{self.stable_sample_n} stable deviation samples "
                    f"after {attempts} attempts"
                )

            if stable_x_list:
                x = np.mean(stable_x_list, axis=0)
                y = np.mean(stable_y_list, axis=0)
            else:
                x = np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32)
                y = np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32)
        else:
            _spots_deviation_x = np.empty(MAX_SPOTS, dtype=np.float32)
            _spots_deviation_y = np.empty(MAX_SPOTS, dtype=np.float32)

            if (
                res := self._lib.WFS_CalcSpotToReferenceDeviations(
                    self._instrument_handle, c_int32(1 if cancel_tile else 0)
                )
            ) == 0:
                if err := self._lib.WFS_GetSpotDeviations(
                    self._instrument_handle, _spots_deviation_x, _spots_deviation_y
                ):
                    self.handle_error(err)
            else:
                self.handle_error(res)
            x = _spots_deviation_x[: self.num_spots_x, : self.num_spots_y]
            y = _spots_deviation_y[: self.num_spots_x, : self.num_spots_y]

        if np.all(x == 0) and np.all(y == 0):
            logger.warning(
                "WFS_GetSpotDeviations returned zero-filled buffers — DLL may not have written data"
            )

        return x, y

    @require_take_image
    def get_stable_spot_deviation(
        self, intensity_threshold: float = 0.0, cancel_tile: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """获取斑点偏差, 并把低强度子孔径的偏差置零。

        本函数:
        1. 经 WFS_CalcSpotsCentrDiaIntens 计算斑点强度与质心
        2. 经 WFS_CalcSpotToReferenceDeviations 计算斑点偏差
        3. 把强度 < intensity_threshold 的子孔径的偏差置零

        参数:
            intensity_threshold: 最小强度阈值。强度低于该值的子孔径偏差会被置 0。
                默认 0.0 表示不过滤 (纳入全部子孔径)。
            cancel_tile: 为 True 时从偏差中去掉 tip/tilt。

        返回:
            tuple[np.ndarray, np.ndarray]: (deviation_x, deviation_y)
                形状为 (num_spots_x, num_spots_y) 的数组。
                强度 < 阈值的子孔径偏差 = 0。
        """
        # 第 1 步: 计算强度与质心
        spots_intensities = np.empty(MAX_SPOTS, dtype=np.float32)
        if res := self._lib.WFS_CalcSpotsCentrDiaIntens(
            self._instrument_handle, c_int32(1), c_int32(0)
        ):
            self.handle_error(res)
            return (
                np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32),
                np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32),
            )

        self._lib.WFS_GetSpotIntensities(self._instrument_handle, spots_intensities)
        intensities = spots_intensities[: self.num_spots_x, : self.num_spots_y]

        # 第 2 步: 计算偏差
        _spots_deviation_x = np.empty(MAX_SPOTS, dtype=np.float32)
        _spots_deviation_y = np.empty(MAX_SPOTS, dtype=np.float32)

        if (
            res := self._lib.WFS_CalcSpotToReferenceDeviations(
                self._instrument_handle, c_int32(1 if cancel_tile else 0)
            )
        ) != 0:
            self.handle_error(res)
            return (
                np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32),
                np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32),
            )

        if err := self._lib.WFS_GetSpotDeviations(
            self._instrument_handle, _spots_deviation_x, _spots_deviation_y
        ):
            self.handle_error(err)
            return (
                np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32),
                np.zeros((self.num_spots_x, self.num_spots_y), dtype=np.float32),
            )

        x = _spots_deviation_x[: self.num_spots_x, : self.num_spots_y]
        y = _spots_deviation_y[: self.num_spots_x, : self.num_spots_y]

        # 第 3 步: 把低强度子孔径的偏差置零
        if intensity_threshold > 0.0:
            low_intensity_mask = intensities < intensity_threshold
            x[low_intensity_mask] = 0.0
            y[low_intensity_mask] = 0.0
            zeroed_count = np.sum(low_intensity_mask)
            if zeroed_count > 0:
                logger.info(
                    f"Zeroed deviations for {zeroed_count} subapertures "
                    f"with intensity < {intensity_threshold}"
                )

        return x, y

    # ==================== 曝光 / 增益管理 ====================

    def optimize_exposure_time_and_gain(self) -> tuple[float, float]:
        """自动优化曝光时间与增益, 以获得清晰的点场图像。

        最多拍 10 张测试图, 每张之后检查设备状态, 找出既不饱和 (光功率过高)
        也不过暗 (光功率过低) 的可用曝光。

        返回:
            tuple[float, float]: (optimal_exposure_time_ms, optimal_gain)

        注意:
            不会改动设备设置; 调用方应自行应用返回值。
        """
        lib, instrument_handle = self._lib, self._instrument_handle
        # 连续拍图直到有一张可用; 每拍一张后检查设备状态以判定是否可用
        actual_exposure = c_double()
        actual_gain = c_double()
        device_status = c_int32()
        for i in range(MAX_AUTOEXPOSE_ATTEMPTS):
            lib.WFS_TakeSpotfieldImageAutoExpos(
                instrument_handle, byref(actual_exposure), byref(actual_gain)
            )
            lib.WFS_GetStatus(instrument_handle, byref(device_status))
            if device_status.value & 0x00000002:
                logger.warning("Power too high")
            elif device_status.value & 0x00000004:
                logger.warning("Power too low")
            elif device_status.value & 0x00000008:
                logger.warning("High ambient light")
            else:
                logger.info(
                    f"Image is usable at {actual_exposure.value} ms.... breaking loop"
                )
                break
        return actual_exposure.value, actual_gain.value

    def get_exposure_time_range(self) -> tuple[float, float, float]:
        """获取硬件支持的曝光时间范围。

        返回:
            tuple[float, float, float]: (min_exposure_ms, max_exposure_ms, increment_ms)
                所有值均为秒。

        注意:
            若此前已取过则返回缓存值; 首次调用时才查询硬件。
        """
        if hasattr(self, "_exposure_time_range"):
            return self._exposure_time_range

        min_exp = c_double()
        max_exp = c_double()
        increment = c_double()
        err = self._lib.WFS_GetExposureTimeRange(
            self._instrument_handle,
            byref(min_exp),
            byref(max_exp),
            byref(increment),
        )
        if err != 0:
            self.handle_error(err, no_raise=True)
            # 硬件查询失败时回退到常量
            logger.warning("Failed to get exposure range from hardware, using defaults")
            return EXP_TIME_LOW, EXP_TIME_HIGH, 0.001

        # 从微秒换算为毫秒
        self._exposure_time_range = (
            min_exp.value / 1000.0,
            max_exp.value / 1000.0,
            increment.value / 1000.0 if increment.value > 0 else 0.001,
        )
        logger.info(
            f"Exposure time range: {self._exposure_time_range[0]:.3f} ~ "
            f"{self._exposure_time_range[1]:.3f} ms (step: {self._exposure_time_range[2]:.3f} ms)"
        )
        return self._exposure_time_range

    # ==================== 属性 ====================

    @property
    def exposure_time(self) -> float:
        """获取当前曝光时间, 单位毫秒。"""
        actual_exposure = c_double()
        self._lib.WFS_GetExposureTime(self._instrument_handle, byref(actual_exposure))
        return actual_exposure.value

    @exposure_time.setter
    def exposure_time(self, value: float) -> None:
        """设置曝光时间。

        参数:
            value: 曝光时间, 单位毫秒。

        异常:
            AssertionError: value 超出合法范围 [0.002, 86] ms。
        """
        assert EXP_TIME_LOW <= value <= EXP_TIME_HIGH, (
            f"exposure time must be in range [{EXP_TIME_LOW}, {EXP_TIME_HIGH}] ms"
        )
        actual_exposure = c_double()
        self._lib.WFS_SetExposureTime(
            self._instrument_handle, c_double(value), byref(actual_exposure)
        )
        logger.info(f"actual exposure time is {actual_exposure.value} ms.")

    @property
    def pupil(self) -> tuple[float, float, float, float]:
        """获取当前光瞳配置。

        返回:
            tuple[float, float, float, float]: (centroid_x, centroid_y, diameter_x, diameter_y), 单位 mm。
        """
        beam_centroid_x = c_double()
        beam_centroid_y = c_double()
        beam_diameter_x = c_double()
        beam_diameter_y = c_double()
        self._lib.WFS_GetPupil(
            self._instrument_handle,
            byref(beam_centroid_x),
            byref(beam_centroid_y),
            byref(beam_diameter_x),
            byref(beam_diameter_y),
        )
        self.c_x, self.c_y = beam_centroid_x.value, beam_centroid_y.value
        self.d_x, self.d_y = beam_diameter_x.value, beam_diameter_y.value
        return self.c_x, self.c_y, self.d_x, self.d_y

    @pupil.setter
    def pupil(self, center_and_diameter: tuple[float, float, float, float]) -> None:
        """设置光瞳配置。

        参数:
            center_and_diameter: (centroid_x, centroid_y, diameter_x, diameter_y), 单位 mm。

        注意:
            pupil 应来自 ``optimize_pupil()`` 且必须显式写回:
            ``wfs.pupil = wfs.optimize_pupil()`` (optimize_pupil 只计算不设置,
            本 setter 也不会因直径 ≤ 0 自动调用它 — 直径 ≤ 0 / 中心 (0,0) 按
            SDK 语义原样下发, 可能是 adaptive/pupil 自动模式)。
            硬编码 pupil 与真实光束不符时, 边界无效子孔径会污染 WFS_ZernikeLsf
            全孔径 LSF 拟合 → 巨大的假 tip/tilt (2026-09 实测)。
        """
        c_x, c_y, d_x, d_y = center_and_diameter
        # 防御性检查 (2026-09 实测固化): SDK 约束 中心 ±5.0mm / 直径 0.1~10.0mm;
        # 越界 pupil 只能拟合出垃圾 Zernike 系数. 只报警, 不改行为.
        if (
            not np.isfinite([c_x, c_y, d_x, d_y]).all()
            or abs(c_x) > 5.0
            or abs(c_y) > 5.0
            or d_x > 10.0
            or d_y > 10.0
        ):
            logger.warning(
                "pupil out of SDK range (中心 ±5.0mm / 直径 0.1~10.0mm): "
                "center=({:.3f},{:.3f})mm diameter=({:.3f},{:.3f})mm — "
                "use wfs.pupil = wfs.optimize_pupil()",
                c_x, c_y, d_x, d_y,
            )
        self._lib.WFS_SetPupil(
            self._instrument_handle,
            c_double(c_x),
            c_double(c_y),
            c_double(d_x),
            c_double(d_y),
        )
        logger.info(f"pupil is {c_x=}, {c_y=}, {d_x=}, {d_y=}")
        self.c_x, self.c_y, self.d_x, self.d_y = c_x, c_y, d_x, d_y

    @property
    def high_speed(self) -> bool:
        """检查是否已启用高速模式。

        返回:
            bool: 高速模式生效时为 True。
        """
        enable_high_speed = self._lib.WFS_CheckHighspeedCentroids(
            self._instrument_handle
        ).value
        return enable_high_speed

    @high_speed.setter
    def high_speed(self, enable: bool) -> None:
        """启用或关闭高速模式。

        参数:
            enable: 为 True 启用高速模式, 为 False 关闭。

        注意:
            高速模式只支持 512x512 分辨率, 且要求自动曝光。
            启用后会自动重新优化光瞳。
        """
        if self.device_name.upper() == "WFS40-5C":
            logger.warning(f"{self.device_name} not support high speed mode!")
            return

        def __set_high_speed():
            return self._lib.WFS_SetHighspeedMode(
                self._instrument_handle,
                c_int32(1 if enable else 0),
                c_int32(1),
                c_int32(1),
                c_int32(1),
            )

        self.enable_high_speed = False
        if enable:
            self.optimize_exposure_time_and_gain()
            self._lib.WFS_CalcSpotsCentrDiaIntens(
                self._instrument_handle, c_int32(1), c_int32(1)
            )

        if res := __set_high_speed():  # res == 0 表示成功
            self.handle_error(res, True)
            self.pupil = self.optimize_pupil()
            if res := __set_high_speed():  # 用自动光瞳设置再试一次
                self.handle_error(res, True)
        else:
            self.enable_high_speed = enable
            if self.enable_high_speed:
                windowCountX = ViInt32()
                windowCountY = ViInt32()
                windowSizeX = ViInt32()
                windowSizeY = ViInt32()
                windowStartposX = np.zeros(
                    self.num_spots_x, dtype=np.int32
                )  # 此参数返回一维数组, 含 X 方向上各斑点窗口的起始像素位置。
                windowStartposY = np.zeros(
                    self.num_spots_y, dtype=np.int32
                )  # 此参数返回一维数组, 含 Y 方向上各斑点窗口的起始像素位置。

                self._lib.WFS_GetHighspeedWindows(
                    self._instrument_handle,
                    byref(windowCountX),
                    byref(windowCountY),
                    byref(windowSizeX),
                    byref(windowSizeY),
                    np2c(windowStartposX),
                    np2c(windowStartposY),
                )

                self.hs_window_count_x = windowCountX.value
                self.hs_window_count_y = windowCountY.value
                self.hs_window_size_x = windowSizeX.value
                self.hs_window_size_y = windowSizeY.value
                self.hs_window_startpos_x = windowStartposX
                self.hs_window_startpos_y = windowStartposY

        logger.info("high speed mode is " + "on" if self.enable_high_speed else "off")
