"""ioc-dm:NLight / MicroDM 变形镜 caproto 软 IOC。

PV 一览(前缀 NLight-DM: 或 MicroDM:):
    Voltages      - 全部通道电压数组(float64,长度须等于 ActuatorCount),写后下发
    Zero          - 写 1 触发复位(reset_all 到 0V)
    Relay         - 继电器(0=OFF, 1=ON;仅 MicroDM 支持)
    HV            - 高压开关(0=OFF, 1=ON;仅 NLight 支持)
    Connected     - 连接状态(只读)
    ActuatorCount - 通道数(只读)
    VMin          - 最小电压 V(只读)
    VMax          - 最大电压 V(只读)
    Type          - 设备类型(只读)

caproto 1.3 语义(与 ioc-slm 同,实测):
    - 只读 PV 值更新必须 await instance.write(...);startup 钩子返回值被丢弃。
    - putter 抛 ValueError 会以 CA 错误返回客户端(离线写被拒)。
    - 钩子方法名必须与 pvproperty 属性名一致(否则 CaprotoRuntimeError)。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from caproto import ChannelType
from caproto.server import PVGroup, pvproperty

logger = logging.getLogger("ioc.dm")

# Voltages 波形的最大长度:取两类 DM 的最大通道数(MicroDM 39x39=1521)
MAX_ACTUATORS = 1521
DEFAULT_V_MIN = -300.0
DEFAULT_V_MAX = 499.0


class DMIoc(PVGroup):
    """NLight / MicroDM soft IOC。

    设备访问约定(runner 契约,与 SLMIoc 相同):
        - 类方法 create_device(spec):按 ioc.yaml device.type 构造驱动实例
          (nlight → NLight,micro → MicroDM),硬件缺失时返回 None(离线模式)。
        - 实例方法 startup():打开设备;shutdown():关闭设备。
    """

    # ---- 控制 PV ----
    voltages = pvproperty(
        dtype=ChannelType.DOUBLE,
        value=np.zeros(MAX_ACTUATORS, dtype=np.float64),
        max_length=MAX_ACTUATORS,
        name="Voltages",
        doc="全部通道电压数组(V)",
    )
    zero = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="Zero",
        doc="写 1 复位全部通道到 0V",
    )
    relay = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="Relay",
        doc="继电器(0=OFF, 1=ON;仅 MicroDM)",
    )
    hv = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="HV",
        doc="高压开关(0=OFF, 1=ON;仅 NLight)",
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
    actuator_count = pvproperty(
        dtype=ChannelType.INT,
        value=0,
        name="ActuatorCount",
        doc="通道数",
        read_only=True,
    )
    v_min = pvproperty(
        dtype=ChannelType.DOUBLE,
        value=DEFAULT_V_MIN,
        name="VMin",
        doc="最小电压(V)",
        read_only=True,
    )
    v_max = pvproperty(
        dtype=ChannelType.DOUBLE,
        value=DEFAULT_V_MAX,
        name="VMax",
        doc="最大电压(V)",
        read_only=True,
    )
    dm_type = pvproperty(
        dtype=ChannelType.STRING,
        value="",
        name="Type",
        doc="设备类型",
        read_only=True,
    )

    def __init__(
        self,
        *args: Any,
        device: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._dm = device
        # 离线回退值优先取 ioc.yaml(spec 由 runner 挂在子类上)
        spec = getattr(type(self), "ioc_spec", None)
        self._n_actuators: int = MAX_ACTUATORS
        self._v_min: float = DEFAULT_V_MIN
        self._v_max: float = DEFAULT_V_MAX
        if spec is not None and spec.devices:
            params = spec.devices[0].params
            self._n_actuators = int(params.get("actuator_count", self._n_actuators))
            self._v_min = float(params.get("vmin", self._v_min))
            self._v_max = float(params.get("vmax", self._v_max))
        # 设备优先:驱动类常量是权威值
        if self._dm is not None:
            self._n_actuators = int(
                getattr(self._dm, "DM_NUM", getattr(self._dm, "DM_Num", self._n_actuators))
            )
            self._v_min = float(getattr(self._dm, "V_Min", self._v_min))
            self._v_max = float(getattr(self._dm, "V_Max", self._v_max))
        self._connected = False

    # ------------------------------------------------------------------
    # 工厂:由 runner 调用,按 ioc.yaml 构造硬件实例
    # ------------------------------------------------------------------
    @classmethod
    def create_device(cls, spec) -> Any:
        """按 device.type 构造 DM 驱动;硬件缺失/不可导入返回 None。"""
        if not spec.devices:
            logger.warning("ioc.yaml 未声明 devices,以离线模式启动")
            return None
        dm_type = str(spec.devices[0].type).lower()
        try:
            if dm_type == "nlight":
                from ao_shaping.drivers.dm.nlight.driver import NLight

                # NLight.open() 仅做 UDP 发送(无硬件也"成功"),必须先用 TCP
                # 探测 192.168.6.10:1001,不可达即离线,避免误报 Connected=True。
                if not NLight.is_reachable():
                    logger.warning(
                        "NLight 硬件不可达(%s:%s),以离线模式启动",
                        NLight._IP,
                        NLight._PORT,
                    )
                    return None
                dm = NLight()
                logger.info("NLight DM 构造完成 (%d 通道)", dm.DM_NUM)
                return dm
            if dm_type == "micro":
                from ao_shaping.drivers.dm.micro.driver import MicroDM

                dm = MicroDM()
                logger.info("MicroDM 构造完成 (%d 通道, %d 控制器)", dm.DM_Num, len(dm._controllers))
                return dm
            logger.warning("未知 DM 类型 %r,以离线模式启动", dm_type)
            return None
        except Exception as exc:  # noqa: BLE001 - 驱动/硬件缺失降级离线
            logger.error("DM 驱动构造失败(%s): %s", dm_type, exc)
            return None

    # ------------------------------------------------------------------
    # 启动/停止钩子(runner 显式调用;同步,事件循环外)
    # ------------------------------------------------------------------
    def startup(self) -> None:
        """打开设备并更新连接状态。"""
        if self._dm is None:
            logger.warning("DM 设备未注入,运行在离线模式(仅注册 PV)")
            self._connected = False
            return
        try:
            self._dm.open()
            self._connected = True
            logger.info(
                "DM 已打开 (%s, %d 通道)", type(self._dm).__name__, self._n_actuators
            )
        except Exception as exc:  # noqa: BLE001 - 硬件未连接时降级离线
            logger.error("DM open 失败: %s", exc)
            self._connected = False

    def shutdown(self) -> None:
        """关闭设备。"""
        if self._dm is not None:
            try:
                self._dm.close()
                logger.info("DM 已关闭")
            except Exception as exc:  # noqa: BLE001
                logger.error("DM close 失败: %s", exc)
        self._connected = False

    # ------------------------------------------------------------------
    # 只读 PV 初始值(startup 钩子,事件循环内;返回值被丢弃,须显式 write)
    # ------------------------------------------------------------------
    @connected.startup
    async def connected(self, instance, async_lib):
        await instance.write(1 if self._connected else 0, verify_value=False)

    @actuator_count.startup
    async def actuator_count(self, instance, async_lib):
        await instance.write(self._n_actuators, verify_value=False)

    @v_min.startup
    async def v_min(self, instance, async_lib):
        await instance.write(self._v_min, verify_value=False)

    @v_max.startup
    async def v_max(self, instance, async_lib):
        await instance.write(self._v_max, verify_value=False)

    @dm_type.startup
    async def dm_type(self, instance, async_lib):
        if self._connected and self._dm is not None:
            name = type(self._dm).__name__
        else:
            name = "offline"
        await instance.write(name, verify_value=False)

    @voltages.startup
    async def voltages(self, instance, async_lib):
        n = self._n_actuators if self._connected else MAX_ACTUATORS
        await instance.write(np.zeros(n, dtype=np.float64), verify_value=False)

    # ------------------------------------------------------------------
    # 写回调(putter 返回值作为新值存储)
    # ------------------------------------------------------------------
    @voltages.putter
    async def voltages(self, instance, value):
        """写入全部通道电压。

        直接调用驱动 _apply_voltages:
        - NLight:其内部带 max_iter_diff 分步斜坡安全逻辑;
        - MicroDM:其基类 _ramp_voltages 在 max_neibor_diff=inf 时存在
          一拍滞后 bug(首轮即置零,目标电压未下发),直接应用更可靠。
        """
        if not self._connected or self._dm is None:
            raise ValueError("DM 设备未连接")
        arr = np.asarray(value, dtype=np.float64)
        n = self._n_actuators
        if arr.shape != (n,):
            raise ValueError(f"Voltages 长度须为 {n},收到 {arr.shape}")
        try:
            self._dm._apply_voltages(arr)  # noqa: SLF001 - 低层应用接口(见注释)
        except Exception as exc:  # noqa: BLE001
            logger.error("send_voltages 失败: %s", exc)
            raise ValueError(f"send_voltages 失败: {exc}") from exc
        logger.info("电压已下发: %d 通道", n)
        return arr

    @zero.putter
    async def zero(self, instance, value):
        """写 1 触发全部通道复位到 0V。"""
        if not self._connected or self._dm is None:
            raise ValueError("DM 设备未连接")
        v = int(value)
        if v != 0:
            try:
                self._dm.reset_all()
            except Exception as exc:  # noqa: BLE001
                logger.error("reset_all 失败: %s", exc)
                raise ValueError(f"reset_all 失败: {exc}") from exc
            logger.info("DM 已复位到 0V")
        return v

    @relay.putter
    async def relay(self, instance, value):
        """继电器控制(仅 MicroDM)。"""
        if not self._connected or self._dm is None:
            raise ValueError("DM 设备未连接")
        if not hasattr(self._dm, "set_relay_state"):
            raise ValueError("当前 DM 不支持 Relay(仅 MicroDM)")
        v = int(value)
        try:
            self._dm.set_relay_state(v != 0)
        except Exception as exc:  # noqa: BLE001
            logger.error("set_relay_state(%d) 失败: %s", v, exc)
            raise ValueError(f"set_relay_state 失败: {exc}") from exc
        logger.info("Relay %s", "ON" if v else "OFF")
        return v

    @hv.putter
    async def hv(self, instance, value):
        """高压开关(仅 NLight)。"""
        if not self._connected or self._dm is None:
            raise ValueError("DM 设备未连接")
        if not hasattr(self._dm, "set_hv"):
            raise ValueError("当前 DM 不支持 HV(仅 NLight)")
        v = int(value)
        try:
            self._dm.set_hv(v != 0)
        except Exception as exc:  # noqa: BLE001
            logger.error("set_hv(%d) 失败: %s", v, exc)
            raise ValueError(f"set_hv 失败: {exc}") from exc
        logger.info("HV %s", "ON" if v else "OFF")
        return v


__all__ = ["DMIoc"]
