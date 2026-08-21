"""ioc-wfs: Thorlabs 波前传感器(WFS)caproto 软 IOC。

PV 一览(前缀 WFS-01:,见 config/ioc.yaml):
    Acquire       写 1 采集一帧:take_image + spotfield + spots 统计,
                 刷新 Image/Mean/Max/SpotMax/NumSpots/FrameCounter
    ExposureTime  曝光时间(ms,驱动属性 setter,钳位 0.002~86)
    HighSpeed     高速模式 enum [off,on](驱动属性 setter)
    MlaIndex      MLA 分辨率(320/512/768/1024/1280;写入即 select_mla)
    PupilX/Y      瞳面中心(mm);PupilDiameter 瞳面直径(mm)
                  (任一写入即整体应用 4 元组到驱动 pupil setter)
    AutoExpose    写 1 触发驱动 optimize_exposure_time_and_gain,
                 刷新 ExposureTime/Gain
    Connected     连接状态(只读)
    SerialNumber  设备序列号(只读)
    DeviceName    设备名(只读)
    Image         最近一帧 spotfield(int32 波形,只读,上限 512x512)
    ImageWidth/H  实际 spotfield 尺寸(只读)
    NumSpotsX/Y   有效子孔径数(只读)
    MeanIntensity 帧平均亮度(只读)
    MaxIntensity  帧峰值亮度(只读)
    SpotMaxIntensity 最强子孔径强度(只读;高速模式或未捕获时为 0)
    Gain          最近一次自动曝光得到的增益(只读)
    FrameCounter  采集帧计数(只读)

驱动 API 要点(源码确认):
    - ThorlabWFS.__init__ 即 load_dll()(硬编码
      C:\\Program Files\\IVI Foundation\\VISA\\Win64\\Bin\\WFS_64.dll,
      缺失抛 OSError)→ create_device 捕获后离线
    - open() 极重:枚举/初始化/MLA/参考面/自动曝光/瞳面(可达数秒),
      且设备被占用时抛 ConnectionError → startup 捕获后离线
    - get_spots_statics() 要求非高速模式(assert)
    - exposure_time 属性 setter 断言 [EXP_TIME_LOW, EXP_TIME_HIGH]
    - master_gain 仅注册表参数,无硬件 setter → Gain 只读,由 AutoExpose 更新
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from caproto import ChannelType
from caproto.server import PVGroup, pvproperty

logger = logging.getLogger("ioc.wfs")

CAM_TYPE = "thorlab"
# spotfield 波形上限:SDK 绑定固定 (512, 512) uint8 缓冲
MAX_SPOTFIELD_PIXELS = 512 * 512
DEFAULT_N_SAMPLE = 5


class WfsIoc(PVGroup):
    """Thorlabs WFS 软 IOC。

    设备访问约定(runner 契约,与 SLMIoc/DMIoc/DHCamIoc 相同):
        - 类方法 create_device(spec):构造 ThorlabWFS(SDK DLL 缺失返回 None)。
        - 实例方法 startup():open()(失败降级离线);shutdown():close()。
        - 阻塞 WFS SDK 调用一律经 loop.run_in_executor 线程池执行。
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
    high_speed = pvproperty(
        dtype=ChannelType.ENUM,
        enum_strings=["off", "on"],
        value=0,
        name="HighSpeed",
        doc="高速模式",
    )
    mla_index = pvproperty(
        dtype=ChannelType.INT,
        value=768,
        name="MlaIndex",
        doc="MLA 分辨率(320/512/768/1024/1280)",
    )
    pupil_x = pvproperty(
        dtype=ChannelType.DOUBLE,
        value=0.0,
        name="PupilX",
        doc="瞳面中心 x(mm)",
    )
    pupil_y = pvproperty(
        dtype=ChannelType.DOUBLE,
        value=0.0,
        name="PupilY",
        doc="瞳面中心 y(mm)",
    )
    pupil_diameter = pvproperty(
        dtype=ChannelType.DOUBLE,
        value=2.0,
        name="PupilDiameter",
        doc="瞳面直径(mm)",
    )
    auto_expose = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="AutoExpose",
        doc="写 1 触发自动曝光,刷新 ExposureTime/Gain",
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
        doc="设备序列号",
        read_only=True,
    )
    device_name = pvproperty(
        dtype=ChannelType.STRING,
        value="",
        name="DeviceName",
        doc="设备名",
        read_only=True,
    )
    image = pvproperty(
        dtype=ChannelType.INT,
        value=np.zeros(MAX_SPOTFIELD_PIXELS, dtype=np.int32),
        max_length=MAX_SPOTFIELD_PIXELS,
        name="Image",
        doc="最近一帧 spotfield 图像(int32)",
        read_only=True,
    )
    image_width = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="ImageWidth",
        doc="spotfield 宽度(px)",
        read_only=True,
    )
    image_height = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="ImageHeight",
        doc="spotfield 高度(px)",
        read_only=True,
    )
    num_spots_x = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="NumSpotsX",
        doc="有效子孔径数 x",
        read_only=True,
    )
    num_spots_y = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="NumSpotsY",
        doc="有效子孔径数 y",
        read_only=True,
    )
    mean_intensity = pvproperty(
        dtype=ChannelType.DOUBLE,
        value=0.0,
        name="MeanIntensity",
        doc="帧平均亮度",
        read_only=True,
    )
    max_intensity = pvproperty(
        dtype=ChannelType.DOUBLE,
        value=0.0,
        name="MaxIntensity",
        doc="帧峰值亮度",
        read_only=True,
    )
    spot_max_intensity = pvproperty(
        dtype=ChannelType.DOUBLE,
        value=0.0,
        name="SpotMaxIntensity",
        doc="最强子孔径强度",
        read_only=True,
    )
    gain = pvproperty(
        dtype=ChannelType.DOUBLE,
        value=0.0,
        name="Gain",
        doc="最近一次自动曝光的增益",
        read_only=True,
    )
    frame_counter = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="FrameCounter",
        doc="采集帧计数",
        read_only=True,
    )

    def __init__(
        self,
        *args: Any,
        device: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._wfs = device
        spec = getattr(type(self), "ioc_spec", None)
        self._n_sample: int = DEFAULT_N_SAMPLE
        if spec is not None and spec.devices:
            params = spec.devices[0].params
            self._n_sample = max(1, int(params.get("n_sample", self._n_sample)))
        self._sn = ""
        self._device_name = ""
        self._connected = False
        self._frame_counter = 0
        self._gain: float = 0.0
        self._mla_res: int = 768
        self._pupil = (0.0, 0.0, 2.0, 2.0)

    # ------------------------------------------------------------------
    # 工厂:由 runner 调用,按 ioc.yaml 构造硬件实例
    # ------------------------------------------------------------------
    @classmethod
    def create_device(cls, spec) -> Any:
        """按 device.type 构造 ThorlabWFS;SDK DLL 缺失/构造失败返回 None。"""
        if not spec.devices:
            logger.warning("ioc.yaml 未声明 devices,以离线模式启动")
            return None
        cam_type = str(spec.devices[0].type).lower()
        if cam_type != CAM_TYPE:
            logger.warning("未知 WFS 类型 %r,以离线模式启动", cam_type)
            return None
        params = spec.devices[0].params
        try:
            from ao_shaping.drivers.wfs.thorlab.driver import ThorlabWFS
        except ImportError:
            logger.warning("无法导入 Thorlab WFS 驱动,以离线模式启动")
            return None
        try:
            # 构造即 load_dll():本机未装 Thorlabs SDK 时抛 OSError
            wfs = ThorlabWFS(
                mla_index=params.get("mla_index"),
                exposure_time=params.get("exposure_time"),
                high_speed=params.get("high_speed"),
                use_custom_ref=params.get("use_custom_ref"),
                pupil_diameter=params.get("pupil_diameter"),
                pupil_center=params.get("pupil_center"),
            )
            logger.info("ThorlabWFS 构造完成(SDK 已加载)")
            return wfs
        except Exception as exc:  # noqa: BLE001 - SDK/依赖缺失降级离线
            logger.error("ThorlabWFS 构造失败: %s", exc)
            return None

    # ------------------------------------------------------------------
    # 启动/停止钩子(runner 显式调用;同步,事件循环外)
    # ------------------------------------------------------------------
    def startup(self) -> None:
        """打开 WFS(open 较重,可达数秒)。"""
        if self._wfs is None:
            logger.warning("WFS 设备未注入,运行在离线模式(仅注册 PV)")
            self._connected = False
            return
        try:
            self._wfs.open()
            self._sn = str(self._wfs.serial_num)
            self._device_name = str(self._wfs.device_name)
            self._mla_res = int(self._wfs.mla_index.name[3:])  # Res768 -> 768
            self._pupil = tuple(float(v) for v in self._wfs.pupil)
            try:
                self._gain = float(self._wfs._gain)  # noqa: SLF001 - 驱动无公共 getter
            except Exception:  # noqa: BLE001
                pass
            self._connected = True
            logger.info(
                "WFS 已打开 (sn=%s, device=%s, mla=%d, pupil=%s)",
                self._sn,
                self._device_name,
                self._mla_res,
                self._pupil,
            )
        except Exception as exc:  # noqa: BLE001 - 硬件未连接/被占用时降级离线
            logger.error("WFS open 失败: %s", exc)
            self._connected = False

    def shutdown(self) -> None:
        """关闭 WFS。"""
        if self._wfs is not None:
            try:
                self._wfs.close()
                logger.info("WFS 已关闭")
            except Exception as exc:  # noqa: BLE001
                logger.error("WFS close 失败: %s", exc)
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

    @device_name.startup
    async def device_name(self, instance, async_lib):
        await instance.write(self._device_name, verify_value=False)

    @image.startup
    async def image(self, instance, async_lib):
        await instance.write(np.zeros(MAX_SPOTFIELD_PIXELS, dtype=np.int32), verify_value=False)

    @image_width.startup
    async def image_width(self, instance, async_lib):
        await instance.write(0, verify_value=False)

    @image_height.startup
    async def image_height(self, instance, async_lib):
        await instance.write(0, verify_value=False)

    @num_spots_x.startup
    async def num_spots_x(self, instance, async_lib):
        await instance.write(0, verify_value=False)

    @num_spots_y.startup
    async def num_spots_y(self, instance, async_lib):
        await instance.write(0, verify_value=False)

    @mean_intensity.startup
    async def mean_intensity(self, instance, async_lib):
        await instance.write(0.0, verify_value=False)

    @max_intensity.startup
    async def max_intensity(self, instance, async_lib):
        await instance.write(0.0, verify_value=False)

    @spot_max_intensity.startup
    async def spot_max_intensity(self, instance, async_lib):
        await instance.write(0.0, verify_value=False)

    @gain.startup
    async def gain(self, instance, async_lib):
        await instance.write(self._gain, verify_value=False)

    @frame_counter.startup
    async def frame_counter(self, instance, async_lib):
        await instance.write(self._frame_counter, verify_value=False)

    @exposure_time.startup
    async def exposure_time(self, instance, async_lib):
        await instance.write(0.0, verify_value=False)

    @high_speed.startup
    async def high_speed(self, instance, async_lib):
        await instance.write(0, verify_value=False)

    @mla_index.startup
    async def mla_index(self, instance, async_lib):
        await instance.write(self._mla_res, verify_value=False)

    @pupil_x.startup
    async def pupil_x(self, instance, async_lib):
        await instance.write(self._pupil[0], verify_value=False)

    @pupil_y.startup
    async def pupil_y(self, instance, async_lib):
        await instance.write(self._pupil[1], verify_value=False)

    @pupil_diameter.startup
    async def pupil_diameter(self, instance, async_lib):
        await instance.write(self._pupil[2], verify_value=False)

    # ------------------------------------------------------------------
    # 写回调(putter 返回值作为新值存储)
    # ------------------------------------------------------------------
    @acquire.putter
    async def acquire(self, instance, value):
        """写 1 采集一帧,刷新图像与统计 PV。"""
        v = int(value)
        if v != 1:
            return v
        if not self._connected or self._wfs is None:
            raise ValueError("WFS 未连接")
        loop = asyncio.get_running_loop()

        def _capture():
            self._wfs.take_image(self._n_sample)
            img = self._wfs.get_spotfiled_image()
            stats = None
            try:
                stats = self._wfs.get_spots_statics()  # 高速模式时 assert 拒绝
            except (AssertionError, RuntimeError) as exc:
                logger.warning("spots 统计不可用(高速模式?): %s", exc)
            return img, stats

        try:
            img, stats = await loop.run_in_executor(None, _capture)
        except Exception as exc:  # noqa: BLE001
            logger.error("采集失败: %s", exc)
            raise ValueError(f"采集失败: {exc}") from exc
        flat = np.ascontiguousarray(img.ravel(), dtype=np.int32)
        self._frame_counter += 1
        await self.image.write(flat, verify_value=False)
        await self.image_width.write(int(img.shape[1]), verify_value=False)
        await self.image_height.write(int(img.shape[0]), verify_value=False)
        await self.mean_intensity.write(float(np.mean(img)), verify_value=False)
        await self.max_intensity.write(float(np.max(img)), verify_value=False)
        if stats is not None:
            intensities, _ = stats
            await self.num_spots_x.write(int(self._wfs.num_spots_x), verify_value=False)
            await self.num_spots_y.write(int(self._wfs.num_spots_y), verify_value=False)
            await self.spot_max_intensity.write(float(np.nanmax(intensities)), verify_value=False)
        else:
            await self.spot_max_intensity.write(0.0, verify_value=False)
        await self.frame_counter.write(self._frame_counter, verify_value=False)
        logger.info("WFS 采集完成: %s, mean=%.1f max=%.1f", img.shape, float(np.mean(img)), float(np.max(img)))
        return v

    @exposure_time.putter
    async def exposure_time(self, instance, value):
        """写曝光时间(ms),驱动属性 setter 断言 [EXP_TIME_LOW, EXP_TIME_HIGH]。"""
        if not self._connected or self._wfs is None:
            raise ValueError("WFS 未连接")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, setattr, self._wfs, "exposure_time", float(value))
        except Exception as exc:  # noqa: BLE001
            logger.error("set exposure 失败: %s", exc)
            raise ValueError(f"set exposure 失败: {exc}") from exc
        try:
            actual = float((await loop.run_in_executor(None, lambda: self._wfs.exposure_time)).value)
        except Exception:  # noqa: BLE001
            actual = float(value)
        return actual

    @high_speed.putter
    async def high_speed(self, instance, value):
        """写高速模式 off/on(驱动属性 setter)。"""
        if not self._connected or self._wfs is None:
            raise ValueError("WFS 未连接")
        v = bool(int(value))
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, setattr, self._wfs, "high_speed", v)
        except Exception as exc:  # noqa: BLE001
            logger.error("set high_speed 失败: %s", exc)
            raise ValueError(f"set high_speed 失败: {exc}") from exc
        return 1 if v else 0

    @mla_index.putter
    async def mla_index(self, instance, value):
        """写 MLA 分辨率(320/512/768/1024/1280)→ select_mla。"""
        if not self._connected or self._wfs is None:
            raise ValueError("WFS 未连接")
        res = int(value)
        loop = asyncio.get_running_loop()
        try:
            from ao_shaping.drivers.wfs.thorlab.driver import MlaRes

            await loop.run_in_executor(None, self._wfs.select_mla, MlaRes.from_str(res))
        except Exception as exc:  # noqa: BLE001
            logger.error("select_mla 失败: %s", exc)
            raise ValueError(f"select_mla 失败: {exc}") from exc
        self._mla_res = int(self._wfs.mla_index.name[3:])
        return self._mla_res

    @pupil_x.putter
    async def pupil_x(self, instance, value):
        if not self._connected or self._wfs is None:
            raise ValueError("WFS 未连接")
        self._pupil = (float(value), self._pupil[1], self._pupil[2], self._pupil[3])
        await self._apply_pupil()
        return self._pupil[0]

    @pupil_y.putter
    async def pupil_y(self, instance, value):
        if not self._connected or self._wfs is None:
            raise ValueError("WFS 未连接")
        self._pupil = (self._pupil[0], float(value), self._pupil[2], self._pupil[3])
        await self._apply_pupil()
        return self._pupil[1]

    @pupil_diameter.putter
    async def pupil_diameter(self, instance, value):
        if not self._connected or self._wfs is None:
            raise ValueError("WFS 未连接")
        self._pupil = (self._pupil[0], self._pupil[1], float(value), float(value))
        await self._apply_pupil()
        return self._pupil[2]

    @auto_expose.putter
    async def auto_expose(self, instance, value):
        """写 1 触发驱动自动曝光,刷新 ExposureTime/Gain。"""
        v = int(value)
        if v != 1:
            return v
        if not self._connected or self._wfs is None:
            raise ValueError("WFS 未连接")
        loop = asyncio.get_running_loop()
        try:
            exp_ms, gain = await loop.run_in_executor(
                None, self._wfs.optimize_exposure_time_and_gain
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("自动曝光失败: %s", exc)
            raise ValueError(f"自动曝光失败: {exc}") from exc
        self._gain = float(gain)
        await self.exposure_time.write(float(exp_ms), verify_value=False)
        await self.gain.write(self._gain, verify_value=False)
        logger.info("自动曝光完成: exp=%.3fms, gain=%.2f", exp_ms, gain)
        return v

    async def _apply_pupil(self) -> None:
        """应用瞳面 4 元组到驱动 pupil setter(线程池执行)。"""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, setattr, self._wfs, "pupil", self._pupil)
        except Exception as exc:  # noqa: BLE001
            logger.error("set pupil 失败: %s", exc)
            raise ValueError(f"set pupil 失败: {exc}") from exc
        # 回读实际值
        try:
            actual = tuple(
                float(v) for v in await loop.run_in_executor(None, lambda: self._wfs.pupil)
            )
            self._pupil = actual
        except Exception:  # noqa: BLE001
            pass
        await self.pupil_x.write(self._pupil[0], verify_value=False)
        await self.pupil_y.write(self._pupil[1], verify_value=False)
        await self.pupil_diameter.write(self._pupil[2], verify_value=False)
        logger.info("瞳面已应用: %s", self._pupil)


__all__ = ["WfsIoc"]
