"""ioc-slm:Santec SLM-200 caproto 软 IOC。

PV 一览(前缀 SLM-01:):
    Wavelength      - 工作波长(nm),写后 set_wavelength
    PhasePattern    - 相位灰度图(1920x1200 int32),写后 write_phase+display_memory
    FlatGray        - 平坦相位灰度值(0-1023),生成 uint16 全灰图下发
    DisplaySlot     - 最近使用的内存槽位(只读)
    MemorySlotCycle - 轮换槽位总数(只读)
    SerialNumber    - SLM 序列号(只读)
    Connected       - 连接状态(只读)
    DisplayedPhase  - 当前显示内容(只读)
    Temperature     - LCOS 温度(只读,每 5s 轮询)

安全规则(AGENTS.md):
    1) 灰度 RAW:平坦相位/灰度图案用 np.full(..., dtype=np.uint16),禁止弧度转换。
    2) 内存槽轮换:连续写相位必须换槽(默认 cycle([3,4,5]))。

caproto 1.3 语义(实测):
    - pvproperty 的 name= 参数决定注册 PV 名(prefix + name)。
    - 只读 PV 的值更新必须 await instance.write(...);
      startup/scan 钩子的返回值会被丢弃。
    - putter 的返回值作为新值存储;抛 ValueError 会以 CA 错误返回给客户端。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from caproto import ChannelType
from caproto.server import PVGroup, pvproperty

from ao_epics_common import (
    DEFAULT_SLOTS,
    GRAYSCALE_MAX,
    GRAYSCALE_MIN,
    SLM_PANEL_HEIGHT,
    SLM_PANEL_WIDTH,
    MemorySlotRotator,
    SlmRuleError,
    flat_phase_grayscale,
    validate_grayscale,
    validate_phase_array,
)

logger = logging.getLogger("ioc.slm")


class SLMIoc(PVGroup):
    """Santec SLM-200 soft IOC。

    设备访问约定(runner 契约):
        - 类方法 create_device(spec):根据 ioc.yaml 构造 SantecSLM200 实例,
          硬件缺失时返回 None(离线模式,仅注册 PV)。
        - 实例方法 startup():打开设备;shutdown():关闭设备。
    """

    # ---- 控制 PV ----
    wavelength = pvproperty(
        dtype=ChannelType.INT,
        value=1064,
        name="Wavelength",
        doc="工作波长(nm)",
    )
    phase_pattern = pvproperty(
        dtype=ChannelType.INT,
        value=np.zeros(SLM_PANEL_HEIGHT * SLM_PANEL_WIDTH, dtype=np.int32),
        max_length=SLM_PANEL_HEIGHT * SLM_PANEL_WIDTH,
        name="PhasePattern",
        doc="相位灰度图(1920x1200)",
    )
    flat_gray = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="FlatGray",
        doc="平坦相位灰度值(0-1023)",
    )

    # ---- 状态 PV(只读) ----
    display_slot = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="DisplaySlot",
        doc="最近使用的内存槽位",
        read_only=True,
    )
    memory_slot_cycle = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="MemorySlotCycle",
        doc="轮换槽位总数",
        read_only=True,
    )
    serial_number = pvproperty(
        dtype=ChannelType.STRING,
        value="",
        name="SerialNumber",
        doc="SLM 序列号",
        read_only=True,
    )
    connected = pvproperty(
        dtype=ChannelType.ENUM,
        enum_strings=["Disconnected", "Connected"],
        value=0,
        name="Connected",
        doc="设备连接状态",
        read_only=True,
    )
    displayed_phase = pvproperty(
        dtype=ChannelType.STRING,
        value="",
        name="DisplayedPhase",
        doc="当前显示内容",
        read_only=True,
    )
    temperature = pvproperty(
        dtype=ChannelType.DOUBLE,
        value=0.0,
        name="Temperature",
        doc="LCOS 温度(℃)",
        read_only=True,
    )

    def __init__(
        self,
        *args: Any,
        device: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._slm = device
        # 槽位/面板参数优先取 ioc.yaml(spec 由 runner 挂在子类上)
        spec = getattr(type(self), "ioc_spec", None)
        slots: tuple[int, ...] = DEFAULT_SLOTS
        panel_w: int = SLM_PANEL_WIDTH
        panel_h: int = SLM_PANEL_HEIGHT
        if spec is not None and spec.devices:
            params = spec.devices[0].params
            raw_slots = params.get("memory_slots", DEFAULT_SLOTS)
            slots = tuple(int(s) for s in raw_slots)
            panel_w = int(params.get("panel_width", panel_w))
            panel_h = int(params.get("panel_height", panel_h))
        self._rotator = MemorySlotRotator(slots)
        # 面板尺寸:设备优先,回退 ioc.yaml/常量
        self._panel_h = panel_h
        self._panel_w = panel_w
        if self._slm is not None:
            try:
                res = self._slm.Panel_Res  # (width, height)
                self._panel_w = int(res[0])
                self._panel_h = int(res[1])
            except Exception:  # noqa: BLE001 - 离线模式保持默认
                pass
        self._slots = self._rotator.slots
        self._connected = False

    # ------------------------------------------------------------------
    # 工厂:由 runner 调用,按 ioc.yaml 构造硬件实例
    # ------------------------------------------------------------------
    @classmethod
    def create_device(cls, spec) -> Any:
        """根据 ioc.yaml 构造 SantecSLM200;硬件缺失/不可导入返回 None。"""
        if not spec.devices:
            logger.warning("ioc.yaml 未声明 devices,以离线模式启动")
            return None
        params = spec.devices[0].params
        try:
            from ao_shaping.drivers.slm.santec.driver import SantecSLM200
        except ImportError:
            logger.warning("无法导入 SantecSLM200 驱动,以离线模式启动")
            return None
        try:
            slm = SantecSLM200(
                slm_number=int(params.get("slm_number", 1)),
                wavelength=int(params.get("wavelength")) if params.get("wavelength") else None,
            )
            logger.info("SantecSLM200 构造完成(slm_number=%d)", slm.slm_number)
            return slm
        except Exception as exc:  # noqa: BLE001 - 构造失败降级离线
            logger.error("SantecSLM200 构造失败: %s", exc)
            return None

    # ------------------------------------------------------------------
    # 启动/停止钩子(runner 显式调用;同步,事件循环外)
    # ------------------------------------------------------------------
    def startup(self) -> None:
        """打开设备并更新连接状态 PV。"""
        if self._slm is None:
            logger.warning("SLM 设备未注入,运行在离线模式(仅注册 PV)")
            self._connected = False
            return
        try:
            self._slm.open()
            self._connected = True
            logger.info("SLM 已打开 (slm_number=%d)", self._slm.slm_number)
        except Exception as exc:  # noqa: BLE001 - 硬件未连接时降级离线
            logger.error("SLM open 失败: %s", exc)
            self._connected = False

    def shutdown(self) -> None:
        """关闭设备。"""
        if self._slm is not None:
            try:
                self._slm.close()
                logger.info("SLM 已关闭")
            except Exception as exc:  # noqa: BLE001
                logger.error("SLM close 失败: %s", exc)
        self._connected = False

    # ------------------------------------------------------------------
    # 只读 PV 初始值(startup 钩子,事件循环内;返回值被丢弃,须显式 write)
    # ------------------------------------------------------------------
    @serial_number.startup
    async def serial_number(self, instance, async_lib):
        sn = ""
        if self._connected and self._slm is not None:
            try:
                sn = self._slm.get_serial_number() or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("读取序列号失败: %s", exc)
        # verify_value=False:内部状态更新,不经 putter(启动期无硬件交互)
        await instance.write(sn, verify_value=False)

    @wavelength.startup
    async def wavelength(self, instance, async_lib):
        wl = 1064
        if self._connected and self._slm is not None:
            try:
                wl, _ = self._slm.get_wavelength_info()
            except Exception as exc:  # noqa: BLE001
                logger.warning("读取波长失败: %s", exc)
        await instance.write(int(wl), verify_value=False)

    @connected.startup
    async def connected(self, instance, async_lib):
        await instance.write(1 if self._connected else 0, verify_value=False)

    @displayed_phase.startup
    async def displayed_phase(self, instance, async_lib):
        await instance.write("idle", verify_value=False)

    @memory_slot_cycle.startup
    async def memory_slot_cycle(self, instance, async_lib):
        await instance.write(len(self._slots), verify_value=False)

    # ------------------------------------------------------------------
    # 温度轮询(scan 钩子;返回值被丢弃,须显式 write)
    # ------------------------------------------------------------------
    @temperature.scan(period=5)
    async def temperature(self, instance, async_lib):
        temp = 0.0
        if self._connected and self._slm is not None:
            try:
                drive_temp, _ = self._slm.temperature
                temp = float(drive_temp)
            except Exception as exc:  # noqa: BLE001
                logger.debug("读取温度失败: %s", exc)
        await instance.write(temp, verify_value=False)

    # ------------------------------------------------------------------
    # 写回调(putter 返回值作为新值存储)
    # ------------------------------------------------------------------
    @wavelength.putter
    async def wavelength(self, instance, value):
        """设置工作波长(写后调用 set_wavelength)。"""
        wl = int(value)
        if not self._connected or self._slm is None:
            raise ValueError("SLM 设备未连接")
        try:
            self._slm.set_wavelength(wl)
        except Exception as exc:  # noqa: BLE001
            logger.error("set_wavelength(%d) 失败: %s", wl, exc)
            raise ValueError(f"set_wavelength 失败: {exc}") from exc
        logger.info("波长已设为 %d nm", wl)
        return wl

    @phase_pattern.putter
    async def phase_pattern(self, instance, value):
        """写入相位灰度图(规则 1+2:先校验再下发,槽位轮换)。

        规则 1:输入先按原类型校验(浮点=弧度直接拒绝),
        确认合法后才转 uint16——顺序不可颠倒。
        """
        if not self._connected or self._slm is None:
            raise ValueError("SLM 设备未连接")
        # caproto 波形数据按一维送达:先整形为 (h, w),再按原始 dtype 校验
        # (拒绝浮点/越界),最后转 uint16——校验顺序不可颠倒
        arr = np.asarray(value)
        if arr.ndim == 1:
            expected = self._panel_h * self._panel_w
            if arr.size != expected:
                raise ValueError(
                    f"相位数组长度 {arr.size} 不匹配 {self._panel_h}x{self._panel_w}"
                )
            arr = arr.reshape(self._panel_h, self._panel_w)
        try:
            arr = validate_phase_array(arr)
        except SlmRuleError as exc:
            logger.error("相位校验失败: %s", exc)
            raise ValueError(str(exc)) from exc

        # 规则 2:轮换槽位,保证与上次不同
        slot = self._rotator.next()
        try:
            self._slm.write_phase(arr, memory_number=slot)
            self._slm.display_memory(slot)
        except Exception as exc:  # noqa: BLE001
            logger.error("write_phase(槽位 %d) 失败: %s", slot, exc)
            raise ValueError(f"write_phase 失败: {exc}") from exc

        await self.display_slot.write(slot)
        await self.displayed_phase.write(
            f"memory slot {slot} ({arr.shape[1]}x{arr.shape[0]})"
        )
        logger.info("相位已下发: 槽位 %d, %dx%d", slot, arr.shape[1], arr.shape[0])
        return arr.ravel()

    @flat_gray.putter
    async def flat_gray(self, instance, value):
        """平坦相位灰度(规则 1:uint16 全灰图,不经弧度转换)。"""
        if not self._connected or self._slm is None:
            raise ValueError("SLM 设备未连接")
        try:
            gray = validate_grayscale(value)
        except SlmRuleError as exc:
            logger.error("灰度校验失败: %s", exc)
            raise ValueError(str(exc)) from exc
        # 规则 1:np.full uint16,禁止经 create_phase_from_array
        arr = flat_phase_grayscale(
            height=self._panel_h,
            width=self._panel_w,
            gray=gray,
        )
        slot = self._rotator.next()
        try:
            self._slm.write_phase(arr, memory_number=slot)
            self._slm.display_memory(slot)
        except Exception as exc:  # noqa: BLE001
            logger.error("平坦相位下发(槽位 %d)失败: %s", slot, exc)
            raise ValueError(f"平坦相位下发失败: {exc}") from exc
        await self.display_slot.write(slot)
        await self.displayed_phase.write(f"flat gray={gray} (slot {slot})")
        logger.info("平坦相位已下发: gray=%d, 槽位 %d", gray, slot)
        return gray


__all__ = ["SLMIoc"]
