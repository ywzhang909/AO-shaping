"""ioc-mii: MiiCam 高速相机 caproto 软 IOC。

PV 一览(前缀 MII-CAM-01:,见 config/ioc.yaml):
    Acquire            写 1 采集一帧(阻塞采集在线程池执行,完成后刷新 Image 等)
    ExposureTime       曝光时间(ms,写后钳位到驱动范围 0.011~10000)
    CenterX/Y          ROI 中心(存储;在下一次 Width/Height 写入时生效)
    Width/Height       ROI 尺寸(0=全幅;写入即触发 reset_window,回写实际值)
    AutoExposure       写 0/1/2:0=关,1=连续,2=单次(驱动 put_AutoExpoEnable)
    AutoExposureTarget 自动曝光目标亮度(16~220,驱动钳位)
    Connected          连接状态(只读)
    SerialNumber       相机序列号(只读)
    CamType            设备类型(只读)
    Image              最新一帧图像(int32 波形,只读)
    ImageWidth/H       实际帧尺寸(只读)
    FrameCounter       采集帧计数(只读)
    MeanIntensity      最近一帧平均亮度(只读)

与 ioc-dhcam 的差异(驱动 API 实测):
    - get_numpy_image(n_sample, skip_first) 无 denoise 参数
    - reset_window 返回的 center 是相对窗口的 (w//2, h//2),故不回写中心,
      保留用户请求的 CenterX/Y
    - AutoExposure 为模式写(0/1/2)而非触发;ExposureTime 由驱动自动更新
    - bit_depth=8(MONO8)/16(MONO16),来自 ioc.yaml params
    - 序列号无公共属性,用 _sn
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from caproto import ChannelType
from caproto.server import PVGroup, pvproperty

logger = logging.getLogger("ioc.miicam")

CAM_TYPE = "miicam"
# Image 波形上限:1920x1200 全幅 int32(与 ioc-dhcam 同级容量,已验证可传)
MAX_SENSOR_PIXELS = 1920 * 1200
# Acquire 侧默认接受的帧像素数上限(512x512=1MB;ioc.yaml max_pixels 可调)
DEFAULT_MAX_PIXELS = 512 * 512


class MiiCamIoc(PVGroup):
    """MiiCam 高速相机软 IOC。

    设备访问约定(runner 契约,与 DHCamIoc 相同):
        - 类方法 create_device(spec):按 ioc.yaml device.type 构造
          CameraStreamManager;SDK 缺失/无相机/构造失败返回 None(离线模式)。
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
        doc="自动曝光模式:0=关,1=连续,2=单次",
    )
    auto_exposure_target = pvproperty(
        dtype=ChannelType.INT,
        value=120,
        name="AutoExposureTarget",
        doc="自动曝光目标亮度(16~220)",
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
        self._exp_ms: float = 20.0
        self._bit_depth: int = 8
        self._max_pixels: int = DEFAULT_MAX_PIXELS
        if spec is not None and spec.devices:
            params = spec.devices[0].params
            self._cam_id = int(params.get("cam_id", self._cam_id))
            self._exp_ms = float(params.get("exposure_time_ms", self._exp_ms))
            self._bit_depth = 16 if int(params.get("bit_depth", 8)) == 16 else 8
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
        """按 device.type 构造 CameraStreamManager;SDK 缺失/无相机返回 None。"""
        if not spec.devices:
            logger.warning("ioc.yaml 未声明 devices,以离线模式启动")
            return None
        cam_type = str(spec.devices[0].type).lower()
        if cam_type != CAM_TYPE:
            logger.warning("未知相机类型 %r,以离线模式启动", cam_type)
            return None
        params = spec.devices[0].params
        try:
            from ao_shaping.drivers.ccd.miicam.driver import CameraStreamManager
        except ImportError:
            logger.warning("无法导入 MiiCam 驱动,以离线模式启动")
            return None
        try:
            # 硬件探测:枚举不到相机即离线(SDK 缺失时 get_cam_list 抛 AttributeError)
            cam_list = CameraStreamManager.get_cam_list()
            if not cam_list:
                logger.warning("未发现 MiiCam 相机,以离线模式启动")
                return None
            cam = CameraStreamManager(
                cam_id=int(params.get("cam_id", 0)),
                exposure_time_ms=float(params.get("exposure_time_ms", 20.0)),
                skip_sampling=bool(params.get("skip_sampling", False)),
                bit_depth=16 if int(params.get("bit_depth", 8)) == 16 else 8,
            )
            logger.info("CameraStreamManager 构造完成(cam_id=%d)", cam.cam_id)
            return cam
        except Exception as exc:  # noqa: BLE001 - 驱动/硬件缺失降级离线
            logger.error("MiiCam 相机构造失败: %s", exc)
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
            self._sn = getattr(self._cam, "_sn", None) or ""
            try:
                self._exp_ms = float(getattr(self._cam, "exposure_time_ms", self._exp_ms))
            except Exception:  # noqa: BLE001 - 读曝光失败保持默认
                pass
            self._connected = True
            logger.info(
                "相机已打开 (sn=%s, %dx%d, bit_depth=%d)",
                self._sn,
                self._cam.cam_width,
                self._cam.cam_height,
                self._bit_depth,
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

    @auto_exposure.startup
    async def auto_exposure(self, instance, async_lib):
        await instance.write(0, verify_value=False)

    @auto_exposure_target.startup
    async def auto_exposure_target(self, instance, async_lib):
        await instance.write(120, verify_value=False)

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
            img = await loop.run_in_executor(None, self._cam.get_numpy_image, 1, True)
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
        """写曝光时间(ms),钳位到驱动有效范围(0.011~10000)。"""
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
        """写自动曝光模式 0/1/2;回读实际生效模式。"""
        v = int(value)
        if v not in (0, 1, 2):
            raise ValueError("AutoExposure 只接受 0(关)/1(连续)/2(单次)")
        if not self._connected or self._cam is None:
            raise ValueError("相机未连接")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._cam.enable_auto_exposure, v > 0, v)
            state = await loop.run_in_executor(None, self._cam.get_auto_exposure_state)
        except Exception as exc:  # noqa: BLE001
            logger.error("auto exposure 设置失败: %s", exc)
            raise ValueError(f"auto exposure 设置失败: {exc}") from exc
        mode = int(state.get("mode", v))
        logger.info("自动曝光模式已设置: 请求=%d, 实际=%d", v, mode)
        return mode

    @auto_exposure_target.putter
    async def auto_exposure_target(self, instance, value):
        """写自动曝光目标亮度(驱动钳位 16~220)。"""
        if not self._connected or self._cam is None:
            raise ValueError("相机未连接")
        loop = asyncio.get_running_loop()
        try:
            target = await loop.run_in_executor(
                None, self._cam.set_auto_exposure_target, int(value)
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("set auto exposure target 失败: %s", exc)
            raise ValueError(f"set auto exposure target 失败: {exc}") from exc
        return int(target)

    async def _apply_window(self) -> None:
        """应用 ROI 窗口(线程池执行)。

        MiiCam 驱动返回的 center 是相对窗口的 (w//2, h//2),不具绝对意义,
        故仅回写实际 width/height,保留用户请求的 CenterX/Y。
        """
        loop = asyncio.get_running_loop()
        try:
            (w, h), _ = await loop.run_in_executor(
                None, self._cam.reset_window, self._roi_center, self._roi_size
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("reset_window 失败: %s", exc)
            raise ValueError(f"reset_window 失败: {exc}") from exc
        self._roi_size = (int(w), int(h))
        # 回写驱动量化后的实际值(verify_value=False 避免递归)
        await self.width.write(self._roi_size[0], verify_value=False)
        await self.height.write(self._roi_size[1], verify_value=False)
        await self.image_width.write(self._roi_size[0], verify_value=False)
        await self.image_height.write(self._roi_size[1], verify_value=False)
        logger.info("ROI 已应用: size=%s, center=%s", self._roi_size, self._roi_center)


__all__ = ["MiiCamIoc"]
