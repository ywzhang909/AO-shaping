"""Micro DM (R50Power) 异步驱动 (asyncio TCP).

同步版 :class:`~ao_shaping.drivers.dm.micro.driver.MicroDM` 的非阻塞替代：
用 ``asyncio.open_connection`` 做 TCP I/O，下发电压时事件循环不被阻塞。
``open()`` / ``close()`` 是同步包装，把 async API 桥接给期望阻塞接口的调用方。

架构:
    VoltageConverter  – 预计算 LUT，O(1) 电压 → 载荷字节
    SendResult        – 单次控制器发送的冻结结果
    AsyncR50Controller – 单控制器 async TCP 客户端
    AsyncMicroDM      – 多控制器 DM 驱动 (DM 子类)

组帧协议与电压换算与同步驱动逐字节一致 (共用
:mod:`ao_shaping.drivers.dm.micro.constants`)。
"""

from __future__ import annotations

import asyncio
import socket
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from ao_shaping.drivers.dm._registry import register_dm
from ao_shaping.drivers.dm.base import DM
from ao_shaping.drivers.dm.micro.constants import (
    CMD_RELAY_OFF,
    CMD_RELAY_ON,
    CMD_SET_ALL_VOLTAGE_BY_ARR,
    FOOTER,
    HEADER,
    PAYLOAD_OFFSET,
    PAYLOAD_SCALE,
)
from ao_shaping.drivers.dm.micro.micro_constants import (
    CHANNELS_PER_CONTROLLER,
    DEFAULT_TIMEOUT,
    DM_NUM,
    PORT_BASE,
    VOLTAGE_MAX,
    VOLTAGE_MIN,
)
from ao_shaping.drivers.dm.micro.wiring_map import WiringMap

#: 控制器默认端口 (10101 → 192.168.0.101)。
DEFAULT_PORT: int = PORT_BASE + 101
#: 单控制器 TCP 超时 (s)。
_DEFAULT_CONTROLLER_TIMEOUT: float = DEFAULT_TIMEOUT

MAX_CHANNELS: int = CHANNELS_PER_CONTROLLER

# 从本模块直接 re-export WiringMap, 方便调用方导入。
__all__ = [
    "AsyncMicroDM",
    "AsyncR50Controller",
    "MicroDMAsync",
    "SendResult",
    "VoltageConverter",
    "WiringMap",
]


# =============================================================================
# 电压换算器 (LUT)
# =============================================================================


class VoltageConverter:
    """电压 → 载荷字节转换的预计算查找表。

    构建一张 65536 项的表, 用与
    :func:`~ao_shaping.drivers.dm.micro.driver.voltages_to_payload`
    相同的公式把每个可能的取整后原始值映射到其 (high, low) 字节对。
    随后 ``fill_buffer`` 可在 O(50) 时间内把一个 50 元素电压数组转成
    100 字节载荷, 且不做逐元素的浮点运算。
    """

    def __init__(self) -> None:
        self._lut: list[tuple[int, int]] = []
        for raw in range(65536):
            high = raw // 256
            low = raw % 256
            self._lut.append((high, low))

        # 为 fill_buffer 预计算常量
        self._scale = PAYLOAD_SCALE
        self._offset = PAYLOAD_OFFSET

    def fill_buffer(self, voltages: npt.NDArray[np.floating], buf: bytearray) -> None:
        """把 50 个电压转换为 100 个交织字节。

        Args:
            voltages: 恰好 50 个浮点电压构成的数组。
            buf: 至少 100 字节的预分配 bytearray (原地修改)。
        """
        v = np.asarray(voltages, dtype=np.float32).ravel()
        np.clip(v, VOLTAGE_MIN, VOLTAGE_MAX, out=v)
        v *= self._scale
        v += self._offset
        raw = np.round(v).astype(np.uint16)
        for i in range(len(raw)):
            high, low = self._lut[int(raw[i])]
            buf[i * 2] = high
            buf[i * 2 + 1] = low

    def convert_single(self, voltage: float) -> tuple[int, int]:
        """经 LUT 把单个电压转换为 (high, low) 字节。"""
        v = float(np.clip(voltage, VOLTAGE_MIN, VOLTAGE_MAX))
        raw = int(round(v * self._scale + self._offset))
        raw = max(0, min(65535, raw))
        return self._lut[raw]


# =============================================================================
# 发送结果
# =============================================================================


@dataclass(frozen=True, slots=True)
class SendResult:
    """单次 ``AsyncR50Controller.send_voltages`` 调用的结果。

    Attributes:
        success: 数据是否已写入 TCP 流。
        error: 面向人阅读的错误字符串 (成功时为 ``None``)。
        latency_us: 往返墙钟延迟, 单位微秒。
        controller_id: 该结果所属的控制器 ID (从 1 开始)
            (无控制器上下文的调用方构造时为 ``None``)。
        ip: 该结果所属的控制器 IP (回退语义同上)。
    """

    success: bool
    error: str | None = None
    latency_us: float = 0.0
    controller_id: int | None = None
    ip: str | None = None


# =============================================================================
# 异步 R50 控制器
# =============================================================================


class AsyncR50Controller:
    """单台 R50Power 控制器 (50 通道) 的异步 TCP 客户端。

    管理一对持久的 ``asyncio.StreamReader`` / ``StreamWriter``。
    所有网络方法均为协程。
    """

    def __init__(
        self,
        controller_id: int,
        ip: str,
        port: int = DEFAULT_PORT,
        timeout: float = _DEFAULT_CONTROLLER_TIMEOUT,
    ) -> None:
        self.controller_id = controller_id
        self.ip = ip
        self.port = port
        self._timeout = timeout

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._converter = VoltageConverter()

    @property
    def is_connected(self) -> bool:
        """TCP 连接是否处于活动状态。"""
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self) -> bool:
        """打开到控制器的异步 TCP 连接。

        Returns:
            成功返回 True, 失败返回 False。
        """
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.ip, self.port),
                timeout=self._timeout,
            )
            # 禁用 Nagle: 帧很小且一次性写入, 延迟 ACK / Nagle 缓冲
            # 会给每帧增加几十毫秒。
            sock = self._writer.get_extra_info("socket")
            if sock is not None:
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except OSError:
                    pass
            logger.debug(
                f"AsyncR50Controller[{self.controller_id}] connected to "
                f"{self.ip}:{self.port}"
            )
            return True
        except (OSError, asyncio.TimeoutError) as exc:
            logger.warning(
                f"AsyncR50Controller[{self.controller_id}] connect failed: {exc}"
            )
            self._reader = None
            self._writer = None
            return False

    async def disconnect(self) -> None:
        """关闭 TCP 连接。"""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._reader = None
            self._writer = None
            logger.debug(f"AsyncR50Controller[{self.controller_id}] disconnected")

    def prebuild_command(self, voltages: npt.NDArray[np.floating]) -> bytes:
        """预计算 50 通道帧的完整 0x09 命令字节。

        编码方式与 :meth:`send_voltages` 完全一致, 但不做网络 I/O,
        因此调用方可以缓存结果, 并在热循环里经 :meth:`send_bytes`
        零成本地重放。

        Args:
            voltages: 50 个浮点电压构成的数组。

        Returns:
            完整的命令字节串 (帧头 + 操作码 + 载荷 + 帧尾)。
        """
        buf = bytearray(MAX_CHANNELS * 2)
        self._converter.fill_buffer(voltages, buf)
        return bytes(HEADER + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR]) + buf + FOOTER)

    async def send_bytes(self, cmd: bytes, timeout: float | None = None) -> SendResult:
        """写入预构建的命令字节并 drain, 同时测量延迟。

        Args:
            cmd: 原始命令字节串 (例如来自 :meth:`prebuild_command`)。
            timeout: 可选的单次发送超时 (秒)。为 *None* 时回退到
                ``self._timeout``。

        Returns:
            描述结果的 :class:`SendResult`。
        """
        if self._writer is None:
            return SendResult(
                success=False,
                error="not_connected",
                controller_id=self.controller_id,
                ip=self.ip,
            )

        t0 = time.perf_counter()
        try:
            self._writer.write(cmd)
            await asyncio.wait_for(
                self._writer.drain(),
                timeout=timeout or self._timeout,
            )
            latency = (time.perf_counter() - t0) * 1e6
            return SendResult(
                success=True,
                latency_us=latency,
                controller_id=self.controller_id,
                ip=self.ip,
            )
        except asyncio.TimeoutError:
            return SendResult(
                success=False,
                error="drain_timeout",
                controller_id=self.controller_id,
                ip=self.ip,
            )
        except (OSError, ConnectionError) as exc:
            return SendResult(
                success=False,
                error=str(exc),
                controller_id=self.controller_id,
                ip=self.ip,
            )

    async def send_voltages(
        self, voltages: npt.NDArray[np.floating], timeout: float | None = None
    ) -> SendResult:
        """经 0x09 命令发送 50 通道电压帧。

        Args:
            voltages: 50 个浮点电压构成的数组。
            timeout: 可选的单次发送超时 (秒)。为 *None* 时回退到
                ``self._timeout``。

        Returns:
            描述结果的 :class:`SendResult`。
        """
        if self._writer is None:
            return SendResult(success=False, error="not_connected")
        return await self.send_bytes(self.prebuild_command(voltages), timeout=timeout)

    async def send_relay(self, state: bool) -> SendResult:
        """打开 (True) 或关闭 (False) 继电器。"""
        if self._writer is None:
            return SendResult(
                success=False,
                error="not_connected",
                controller_id=self.controller_id,
                ip=self.ip,
            )
        cmd = HEADER + bytes([CMD_RELAY_ON if state else CMD_RELAY_OFF]) + FOOTER
        try:
            self._writer.write(cmd)
            await asyncio.wait_for(self._writer.drain(), timeout=self._timeout)
            return SendResult(
                success=True,
                controller_id=self.controller_id,
                ip=self.ip,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            return SendResult(
                success=False,
                error=str(exc),
                controller_id=self.controller_id,
                ip=self.ip,
            )


# =============================================================================
# 异步 MicroDM 驱动
# =============================================================================

# AsyncMicroDM 刻意没有 from_params 工厂: 驱动层的任何参数类
# 都没有携带它所需的控制器 IP 列表。


@register_dm("asyn_micro")
class AsyncMicroDM(DM):
    """异步的多控制器 R50Power 变形镜驱动。

    包装多个 :class:`AsyncR50Controller` 实例, 并对外呈现标准的
    :class:`DM` 接口。同步方法 ``open()`` / ``close()``
    通过 ``_run_async`` 桥接到异步内部实现。

    Attributes:
        DM_Num: 逻辑通道总数 (39×39 = 1521)。
        V_Min: 最小电压 (-20.0 V)。
        V_Max: 最大电压 (120.0 V)。
    """

    DM_Num: int = DM_NUM
    V_Min: float = VOLTAGE_MIN
    V_Max: float = VOLTAGE_MAX

    def __init__(
        self,
        ips: list[str] | None = None,
        timeout: float = _DEFAULT_CONTROLLER_TIMEOUT,
        safety_mode: bool = True,
    ) -> None:
        super().__init__(safety_mode=safety_mode)

        self._ips: list[str] = ips or ["192.168.0.101"]
        self._timeout = timeout

        self._controllers: list[AsyncR50Controller] = [
            AsyncR50Controller(
                controller_id=i,
                ip=ip_str,
                port=PORT_BASE + int(ip_str.split(".")[-1]),
                timeout=timeout,
            )
            for i, ip_str in enumerate(self._ips, start=1)
        ]

        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._open = False

        logger.debug(
            f"AsyncMicroDM initialized: {len(self._controllers)} controller(s), "
            f"{self.DM_Num} channels, endpoints="
            f"{[(c.ip, c.port) for c in self._controllers]}"
        )

    # ---- 同步 ↔ 异步 桥接 ------------------------------------------------

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """返回正在运行的事件循环, 需要时则新建一个。"""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._event_loop = loop
            return loop

    def _run_async(self, coro: Any) -> object:
        """从同步代码运行一个异步协程。

        若已有事件循环在运行, 则经 ``asyncio.ensure_future`` 调度。
        否则用 ``loop.run_until_complete`` 一直运行到完成。
        """
        loop = self._get_or_create_loop()
        if loop.is_running():
            # 我们处在异步上下文中 —— 调度并返回该 task
            return asyncio.ensure_future(coro)
        return loop.run_until_complete(coro)

    # ---- DM 接口 (同步) ------------------------------------------------

    @classmethod
    def is_reachable(cls) -> bool:
        """异步驱动始终视为可达 (在 open 时才连接)。"""
        return True

    @property
    def default_dm_unit_mask(self) -> npt.NDArray[np.bool_]:
        return np.ones(self.DM_NUM, dtype=bool)

    def open(self) -> None:
        """连接到所有控制器 (同步桥接)。"""
        self._run_async(self._async_open())

    async def _async_open(self) -> None:
        results = await self.connect_all()
        n_ok = sum(1 for v in results.values() if v)
        if n_ok == 0:
            raise ConnectionError(
                f"AsyncMicroDM: failed to connect any of "
                f"{len(self._controllers)} controller(s)"
            )
        self._open = True
        logger.info(f"AsyncMicroDM ready: {n_ok}/{len(self._controllers)} connected")

    def close(self) -> None:
        """断开所有控制器 (同步桥接)。"""
        self._run_async(self._async_close())

    async def _async_close(self) -> None:
        for ctrl in self._controllers:
            await ctrl.disconnect()
        self._open = False
        logger.info("AsyncMicroDM disconnected")

    def is_connected(self) -> bool:
        return self._open and any(c.is_connected for c in self._controllers)

    def get_actuator_positions(self) -> npt.NDArray[np.floating]:
        return self._last_voltages.copy()

    def transform(self, cmd: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """把 [-1, 1] 映射到 [V_Min, V_Max]。"""
        return self.transform_voltage(cmd)

    def _apply_voltages(self, vs: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """钳位并保存电压。实际 TCP 发送在 ``send_frame`` 中进行。"""
        vs = np.clip(vs, self.V_Min, self.V_Max)
        self._last_voltages = vs.copy()
        return self._last_voltages

    # ---- 异步专属方法 -------------------------------------------------

    @property
    def controller_info(self) -> list[tuple[int, str, int, bool]]:
        """逐控制器的自省信息: ``(controller_id, ip, port, is_connected)``。

        便于诊断 —— 让调用方准确报告尝试过哪些端点、当前哪些已连接。
        """
        return [
            (c.controller_id, c.ip, c.port, c.is_connected) for c in self._controllers
        ]

    async def connect_all(self) -> dict[int, bool]:
        """并发连接所有控制器。

        Returns:
            controller_id (从 1 开始) -> 是否成功的字典。
        """
        results = await asyncio.gather(*(ctrl.connect() for ctrl in self._controllers))
        return {ctrl.controller_id: ok for ctrl, ok in zip(self._controllers, results)}

    async def send_frame(
        self, voltages: npt.NDArray[np.floating] | None = None
    ) -> list[SendResult]:
        """并发向所有控制器发送电压帧。

        Args:
            voltages: 为 *None* 时发送 ``self._last_voltages``。

        Returns:
            :class:`SendResult` 列表, 每台控制器一项。
        """
        if voltages is not None:
            vs = np.clip(np.asarray(voltages, dtype=np.float64), self.V_Min, self.V_Max)
            self._last_voltages = vs.copy()
        else:
            vs = self._last_voltages

        chunk_size = MAX_CHANNELS
        tasks = []
        for idx, ctrl in enumerate(self._controllers):
            start = idx * chunk_size
            end = start + chunk_size
            chunk = vs[start:end]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), constant_values=0.0)
            tasks.append(ctrl.send_voltages(chunk))

        return list(await asyncio.gather(*tasks))

    def build_frame_commands(self, voltages: npt.NDArray[np.floating]) -> list[bytes]:
        """为完整电压帧预计算各控制器的命令字节。

        完全按 :meth:`send_frame` 的方式对逻辑通道数组分块,
        并为每台控制器返回一条完整编码的命令字节串, 可经
        :meth:`send_frame_commands` 零分配地重放。

        Args:
            voltages: 所有逻辑通道的电压数组。

        Returns:
            预构建的命令字节串列表, 与 ``self._controllers`` 一一对应。
        """
        vs = np.clip(np.asarray(voltages, dtype=np.float64), self.V_Min, self.V_Max)
        commands = []
        for idx, ctrl in enumerate(self._controllers):
            start = idx * MAX_CHANNELS
            end = start + MAX_CHANNELS
            chunk = vs[start:end]
            if len(chunk) < MAX_CHANNELS:
                chunk = np.pad(
                    chunk, (0, MAX_CHANNELS - len(chunk)), constant_values=0.0
                )
            commands.append(ctrl.prebuild_command(chunk))
        return commands

    async def send_frame_commands(
        self,
        commands: list[bytes],
        voltages: npt.NDArray[np.floating] | None = None,
    ) -> list[SendResult]:
        """并发发送预构建的各控制器命令字节。

        :meth:`send_frame` 的热循环快路径: 不做电压编码、不分块、
        不分配内存 —— 只在缓存的字节上对每台控制器做一次 ``write``+``drain``。

        Args:
            commands: 每台控制器一条预构建的命令字节串, 形式如
                :meth:`build_frame_commands` 的返回值。
            voltages: 若提供, 则记录为 ``self._last_voltages``。

        Returns:
            :class:`SendResult` 列表, 每台控制器一项。
        """
        if voltages is not None:
            self._last_voltages = np.clip(
                np.asarray(voltages, dtype=np.float64), self.V_Min, self.V_Max
            ).copy()
        return list(
            await asyncio.gather(
                *[
                    ctrl.send_bytes(cmd)
                    for ctrl, cmd in zip(self._controllers, commands)
                ]
            )
        )

    async def set_relay(self, state: bool) -> dict[int, SendResult]:
        """并发打开 (True) 或关闭 (False) 所有控制器上的继电器。

        Args:
            state: True 给继电器上电, False 断电。

        Returns:
            controller_id (从 1 开始) -> SendResult 的字典。
        """
        results = await asyncio.gather(
            *[ctrl.send_relay(state) for ctrl in self._controllers]
        )
        return {
            ctrl.controller_id: result
            for ctrl, result in zip(self._controllers, results)
        }

    async def shutdown(self, home_voltage: float = 0.0) -> None:
        """安全关机: 电压归零, 继电器断开, 断开所有连接。

        Args:
            home_voltage: 断开前要设置到所有通道上的电压。
        """
        # 在所有控制器上设置原点电压
        home_vs = np.full(self.DM_Num, home_voltage)
        self._last_voltages = home_vs.copy()

        for ctrl in self._controllers:
            if ctrl.is_connected:
                await ctrl.send_voltages(np.full(MAX_CHANNELS, home_voltage))
                await ctrl.send_relay(False)

        await self._async_close()
        logger.info("AsyncMicroDM shut down")


# =============================================================================
# 向后兼容别名
# =============================================================================

# 部分调用方可能用另一个名字导入该类
MicroDMAsync = AsyncMicroDM
