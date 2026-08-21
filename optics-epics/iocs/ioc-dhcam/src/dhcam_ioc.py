"""ioc-dhcam: 大恒(Daheng)GigE 相机 caproto 软 IOC。

PV 一览(前缀 DH-CAM-01:,见 config/ioc.yaml):
    Acquire       写 1 采集一帧(阻塞采集在线程池执行,完成后刷新 Image 等)
    ExposureTime  曝光时间(ms,写后钳位到驱动范围)
    CenterX/Y     ROI 中心(存储;在下一次 Width/Height 写入时生效)
    Width/Height  ROI 尺寸(0=全幅;写入即触发 reset_window)
    AutoExposure  写 1 触发驱动 auto_exposure,刷新 ExposureTime/MeanIntensity
    Connected     连接状态(只读)
    SerialNumber  相机序列号(只读)
    CamType       设备类型(只读)
    Image         最新一帧图像(int32 波形,只读)
    ImageWidth/H  实际帧尺寸(只读)
    FrameCounter  采集帧计数(只读)
    MeanIntensity 最近一帧平均亮度(只读)

caproto 1.3 语义(与 ioc-slm/ioc-dm 同,实测):
    - 只读 PV 值更新必须 await instance.write(...);startup 钩子返回值被丢弃。
    - putter 抛 ValueError 会以 CA 错误返回客户端(离线写被拒)。
    - 阻塞相机调用(采图/曝光/开窗)一律经 loop.run_in_executor 在线程池执行,
      避免卡住 caproto 事件循环。
    - 跨 PV 写入:putter/钩子内 await self.<pv>.write(value, verify_value=False)。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from caproto import ChannelType
from caproto.server import PVGroup, pvproperty

logger = logging.getLogger("ioc.dhcam")

CAM_TYPE = "daheng"
# Image 波形上限:1920x1200 全幅 int32(与 ioc-slm PhasePattern 同级容量,已验证可传)
MAX_SENSOR_PIXELS = 1920 * 1200
# Acquire 侧默认接受的帧像素数上限(512x512=1MB;ioc.yaml max_pixels 可调)
DEFAULT_MAX_PIXELS = 512 * 512


class DHCamIoc(PVGroup):
    """Daheng GigE 相机软 IOC。

    设备访问约定(runner 契约,与 SLMIoc/DMIoc 相同):
        - 类方法 create_device(spec):按 ioc.yaml device.type 构造
          DahengCamManager;gxipy 缺失/无相机/构造失败返回 None(离线模式)。
        - 实例方法 startup():打开设备;shutdown():关闭设备。
    """

    # ---- 控制 PV ----
    acquire = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="Acquire",
        doc="写 1 采集一帧",
    )
    exposure_time = pvproperty(
        dtype=ChannelType.DOUBLE,
        value=0.0,
        name="ExposureTime",
        doc="曝光时间(ms)",
    )
    center_x = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="CenterX",
        doc="ROI 中心 x(存储,Width/Height 写入时生效)",
    )
    center_y = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="CenterY",
        doc="ROI 中心 y(存储,Width/Height 写入时生效)",
    )
    width = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="Width",
        doc="ROI 宽度(0=全幅)",
    )
    height = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="Height",
        doc="ROI 高度(0=全幅)",
    )
    auto_exposure = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="AutoExposure",
        doc="写 1 触发自动曝光",
    )

    # ---- 状态 PV(只读) ----
    connected = pvproperty(
        dtype=ChannelType.ENUM,
        enum_strings=["Disconnected", "Connected"],
        value=0,
        name="Connected",
        doc="设备连接状态",
        read_only=True,
    )
    serial_number = pvproperty(
        dtype=ChannelType.STRING,
        value="",
        name="SerialNumber",
        doc="相机序列号",
        read_only=True,
    )
    cam_type = pvproperty(
        dtype=ChannelType.STRING,
        value="",
        name="CamType",
        doc="设备类型",
        read_only=True,
    )
    image = pvproperty(
        dtype=ChannelType.INT,
        value=np.zeros(MAX_SENSOR_PIXELS, dtype=np.int32),
        max_length=MAX_SENSOR_PIXELS,
        name="Image",
        doc="最新一帧图像(int32)",
        read_only=True,
    )
    image_width = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="ImageWidth",
        doc="图像宽度(px)",
        read_only=True,
    )
    image_height = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="ImageHeight",
        doc="图像高度(px)",
        read_only=True,
    )
    frame_counter = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="FrameCounter",
        doc="采集帧计数",
        read_only=True,
    )
    mean_intensity = pvproperty(
        dtype=ChannelType.DOUBLE,
        value=0.0,
        name="MeanIntensity",
        doc="最近一帧平均亮度",
        read_only=True,
    )

    def __init__(
        self,
        *args: Any,
        device: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._cam = device
        spec = getattr(type(self), "ioc_spec", None)
        self._cam_id: int = 0
        self._exp_ms: float = 0.0
        self._max_pixels: int = DEFAULT_MAX_PIXELS
        if spec is not None and spec.devices:
            params = spec.devices[0].params
            self._cam_id = int(params.get("cam_id", self._cam_id))
            self._exp_ms = float(params.get("exposure_time_ms", self._exp_ms))
            self._max_pixels = int(params.get("max_pixels", self._max_pixels))
        self._sn = ""
        self._connected = False
        self._frame_counter = 0
        self._roi_center = (0, 0)
        self._roi_size = (0, 0)

    # ------------------------------------------------------------------
    # 工厂:由 runner 调用,按 ioc.yaml 构造硬件实例
    # ------------------------------------------------------------------
    @classmethod
    def create_device(cls, spec) -> Any:
        """按 device.type 构造 DahengCamManager;无相机/不可导入返回 None。"""
        if not spec.devices:
            logger.warning("ioc.yaml 未声明 devices,以离线模式启动")
            return None
        cam_type = str(spec.devices[0].type).lower()
        if cam_type != CAM_TYPE:
            logger.warning("未知相机类型 %r,以离线模式启动", cam_type)
            return None
        params = spec.devices[0].params
        try:
            from ao_shaping.drivers.ccd.daheng.driver import DahengCamManager
        except ImportError:
            logger.warning("无法导入 Daheng 驱动,以离线模式启动")
            return None
        try:
            # 硬件探测:枚举不到相机即离线(与 ioc-dm 的 is_reachable 同思路)
            count, _ = DahengCamManager.get_cam_list()
            if not count:
                logger.warning("未发现 Daheng 相机,以离线模式启动")
                return None
            cam = DahengCamManager(
                cam_id=int(params.get("cam_id", 0)),
                exposure_time_ms=float(params.get("exposure_time_ms", 0.0)),
            )
            logger.info("DahengCamManager 构造完成(cam_id=%d)", cam.cam_id)
            return cam
        except Exception as exc:  # noqa: BLE001 - 驱动/硬件缺失降级离线
            logger.error("Daheng 相机构造失败: %s", exc)
            return None

    # ------------------------------------------------------------------
    # 启动/停止钩子(runner 显式调用;同步,事件循环外)
    # ------------------------------------------------------------------
    def startup(self) -> None:
        """打开相机并抓取状态。"""
        if self._cam is None:
            logger.warning("相机设备未注入,运行在离线模式(仅注册 PV)")
            self._connected = False
            return
        try:
            self._cam.open()
            self._sn = self._cam.sn or ""
            try:
                self._exp_ms = float(self._cam.exposure_time)
            except Exception:  # noqa: BLE001 - 读曝光失败保持默认
                pass
            self._connected = True
            logger.info(
                "相机已打开 (sn=%s, %dx%d)",
                self._sn,
                self._cam.cam_width,
                self._cam.cam_height,
            )
            if self._cam.cam_width * self._cam.cam_height > self._max_pixels:
                logger.warning(
                    "全幅 %dx%d 超过 Image PV 上限 %d 像素;请调大 ioc.yaml max_pixels "
                    "或先写 Width/Height 缩小 ROI 再 Acquire",
                    self._cam.cam_width,
                    self._cam.cam_height,
                    self._max_pixels,
                )
        except Exception as exc:  # noqa: BLE001 - 硬件未连接时降级离线
            logger.error("相机 open 失败: %s", exc)
            self._connected = False

    def shutdown(self) -> None:
        """关闭相机。"""
        if self._cam is not None:
            try:
                self._cam.close()
                logger.info("相机已关闭")
            except Exception as exc:  # noqa: BLE001
                logger.error("相机 close 失败: %s", exc)
        self._connected = False

    # ------------------------------------------------------------------
    # 只读 PV 初始值(startup 钩子,事件循环内;须显式 write)
    # ------------------------------------------------------------------
    @connected.startup
    async def connected(self, instance, async_lib):
        await instance.write(1 if self._connected else 0, verify_value=False)

    @serial_number.startup
    async def serial_number(self, instance, async_lib):
        await instance.write(self._sn, verify_value=False)

    @cam_type.startup
    async def cam_type(self, instance, async_lib):
        await instance.write(CAM_TYPE if self._connected else "offline", verify_value=False)

    @image.startup
    async def image(self, instance, async_lib):
        await instance.write(np.zeros(MAX_SENSOR_PIXELS, dtype=np.int32), verify_value=False)

    @image_width.startup
    async def image_width(self, instance, async_lib):
        await instance.write(0, verify_value=False)

    @image_height.startup
    async def image_height(self, instance, async_lib):
        await instance.write(0, verify_value=False)

    @frame_counter.startup
    async def frame_counter(self, instance, async_lib):
        await instance.write(self._frame_counter, verify_value=False)

    @mean_intensity.startup
    async def mean_intensity(self, instance, async_lib):
        await instance.write(0.0, verify_value=False)

    @exposure_time.startup
    async def exposure_time(self, instance, async_lib):
        await instance.write(self._exp_ms, verify_value=False)

    # ------------------------------------------------------------------
    # 写回调(putter 返回值作为新值存储)
    # ------------------------------------------------------------------
    @acquire.putter
    async def acquire(self, instance, value):
        """写 1 采集一帧,完成后刷新图像相关 PV。"""
        v = int(value)
        if v != 1:
            return v
        if not self._connected or self._cam is None:
            raise ValueError("相机未连接")
        loop = asyncio.get_running_loop()
        try:
            img = await loop.run_in_executor(None, self._cam.get_numpy_image, 1, True, False)
        except Exception as exc:  # noqa: BLE001
            logger.error("采集失败: %s", exc)
            raise ValueError(f"采集失败: {exc}") from exc
        if img.size > self._max_pixels:
            raise ValueError(
                f"图像 {img.shape} 超过 Image PV 上限 {self._max_pixels} 像素;"
                "请调大 ioc.yaml max_pixels 或先写 Width/Height 缩小 ROI"
            )
        flat = np.ascontiguousarray(img.ravel(), dtype=np.int32)
        self._frame_counter += 1
        await self.image.write(flat, verify_value=False)
        await self.image_width.write(int(img.shape[1]), verify_value=False)
        await self.image_height.write(int(img.shape[0]), verify_value=False)
        await self.mean_intensity.write(float(np.mean(img)), verify_value=False)
        await self.frame_counter.write(self._frame_counter, verify_value=False)
        logger.info("采集完成: %s, mean=%.1f", img.shape, float(np.mean(img)))
        return v

    @exposure_time.putter
    async def exposure_time(self, instance, value):
        """写曝光时间(ms),钳位到驱动有效范围。"""
        if not self._connected or self._cam is None:
            raise ValueError("相机未连接")
        loop = asyncio.get_running_loop()
        try:
            actual = await loop.run_in_executor(None, self._cam.reset_exposure_time, float(value))
        except Exception as exc:  # noqa: BLE001
            logger.error("set exposure 失败: %s", exc)
            raise ValueError(f"set exposure 失败: {exc}") from exc
        self._exp_ms = float(actual)
        return self._exp_ms

    @width.putter
    async def width(self, instance, value):
        """写 ROI 宽度(0=全幅);触发 reset_window。"""
        if not self._connected or self._cam is None:
            raise ValueError("相机未连接")
        self._roi_size = (int(value), self._roi_size[1])
        await self._apply_window()
        return self._roi_size[0]

    @height.putter
    async def height(self, instance, value):
        """写 ROI 高度(0=全幅);触发 reset_window。"""
        if not self._connected or self._cam is None:
            raise ValueError("相机未连接")
        self._roi_size = (self._roi_size[0], int(value))
        await self._apply_window()
        return self._roi_size[1]

    @center_x.putter
    async def center_x(self, instance, value):
        """写 ROI 中心 x(存储,Width/Height 写入时生效)。"""
        if not self._connected or self._cam is None:
            raise ValueError("相机未连接")
        self._roi_center = (int(value), self._roi_center[1])
        return self._roi_center[0]

    @center_y.putter
    async def center_y(self, instance, value):
        """写 ROI 中心 y(存储,Width/Height 写入时生效)。"""
        if not self._connected or self._cam is None:
            raise ValueError("相机未连接")
        self._roi_center = (self._roi_center[0], int(value))
        return self._roi_center[1]

    @auto_exposure.putter
    async def auto_exposure(self, instance, value):
        """写 1 触发驱动 auto_exposure,刷新 ExposureTime/MeanIntensity。"""
        v = int(value)
        if v != 1:
            return v
        if not self._connected or self._cam is None:
            raise ValueError("相机未连接")
        loop = asyncio.get_running_loop()
        try:
            final_exp, final_mean = await loop.run_in_executor(
                None, self._cam.auto_exposure
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("auto exposure 失败: %s", exc)
            raise ValueError(f"auto exposure 失败: {exc}") from exc
        self._exp_ms = float(final_exp)
        await self.exposure_time.write(self._exp_ms, verify_value=False)
        # 驱动 auto_exposure 返回的 mean 已是 0-255 尺度(np.mean(uint8 img))
        await self.mean_intensity.write(float(final_mean), verify_value=False)
        logger.info("自动曝光完成: exp=%.1fms, mean=%.1f", final_exp, final_mean)
        return v

    async def _apply_window(self) -> None:
        """应用 ROI 窗口(驱动内 stream_off/set/stream_on,线程池执行)。"""
        loop = asyncio.get_running_loop()
        try:
            (w, h), (cx, cy) = await loop.run_in_executor(
                None, self._cam.reset_window, self._roi_center, self._roi_size
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("reset_window 失败: %s", exc)
            raise ValueError(f"reset_window 失败: {exc}") from exc
        self._roi_size, self._roi_center = (int(w), int(h)), (int(cx), int(cy))
        # 回写驱动量化后的实际值(verify_value=False 避免递归)
        await self.width.write(self._roi_size[0], verify_value=False)
        await self.height.write(self._roi_size[1], verify_value=False)
        await self.center_x.write(self._roi_center[0], verify_value=False)
        await self.center_y.write(self._roi_center[1], verify_value=False)
        await self.image_width.write(self._roi_size[0], verify_value=False)
        await self.image_height.write(self._roi_size[1], verify_value=False)
        logger.info("ROI 已应用: size=%s, center=%s", self._roi_size, self._roi_center)


__all__ = ["DHCamIoc"]
