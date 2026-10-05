"""Micro DM (R50Power) 驱动 —— 同步 TCP 路径.

通过 TCP/IP (阻塞 socket) 控制一台或多台 R50Power 控制器，每台 50 通道，
电压范围 -20V..120V，最多 26 台 (1296 通道)。

组帧协议 (0xAA 0xBB 头 / 0xCC 0xDD 脚) 与电压编码见
:mod:`ao_shaping.drivers.dm.micro.constants`；接线表解析见
:mod:`ao_shaping.drivers.dm.micro.wiring_map`。

Wiring Map:
    控制器 IP 与通道映射默认从 ``libs/micro_drive1300/wiring_map.json``
    加载 (``use_wiring_map=True``)。``use_wiring_map=False`` 时退回默认 IP。

示例:
    >>> dm = MicroDM()
    >>> dm.open()
    >>> dm.send_voltages(np.zeros(50))
    >>> dm.set_relay_state(True)
    >>> print(dm.get_actuator_positions())
    >>> dm.close()

Channel Lookup:
    >>> dm = MicroDM()
    >>> info = dm.get_channel_by_xy(x=1, y=3)  # 39x39 array coordinates
    >>> print(info.ip_address, info.payload_position)
    >>> info = dm.get_channel_by_ip_position(ip_suffix=101, payload_position=13)
    >>> print(info.physical_label, info.physical_position)
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from ao_shaping.drivers.device_base import Device, DeviceState, DeviceType
from ao_shaping.drivers.dm._registry import register_dm
from ao_shaping.drivers.dm.base import DM
from ao_shaping.drivers.dm.micro.constants import (
    CMD_RELAY_OFF,
    CMD_RELAY_ON,
    CMD_SET_ALL_CHANNEL_VOLTAGE,
    CMD_SET_ALL_VOLTAGE_BY_ARR,
    CMD_SET_CHANNEL_VOLTAGE,
    FOOTER,
    HEADER,
    PAYLOAD_OFFSET,
    PAYLOAD_SCALE,
    WIRING_MAP_PATH,
)
from ao_shaping.drivers.dm.micro.micro_constants import (
    CHANNELS_PER_CONTROLLER,
    DEFAULT_IPS,
    DEFAULT_TIMEOUT,
    DM_NUM,
    IP_SUFFIX_MAX,
    IP_SUFFIX_MIN,
    MAX_ACTUATORS,
    MAX_CONTROLLERS,
    PORT_BASE,
    VOLTAGE_MAX,
    VOLTAGE_MIN,
)
from ao_shaping.drivers.dm.micro.wiring_map import ChannelInfo, WiringMap
from ao_shaping.model.quantities import DmCommands
from ao_shaping.utils.io.device_config import ConfigHandler, DeviceParam, param
from ao_shaping.utils.io.file import ROOT_DIR

# ── MicroDM 配置参数 ──────────────────────────────────────

_MICRO_DM_CONFIG_DIR = Path(
    os.environ.get("MICRO_DM_CONFIG_DIR", ROOT_DIR / "data" / "micro_dm_configs")
)


@dataclass
class MicroDMParams(DeviceParam):
    """MicroDM 配置参数（可持久化的标量参数）。"""

    timeout: float = param(default=DEFAULT_TIMEOUT, cast=float)
    use_wiring_map: bool = param(default=True, cast=bool)
    safety_mode: bool = param(default=True, cast=bool)


# MicroDM 刻意没有 from_params 工厂: MicroDMParams 只涵盖
# 可持久化的标量设置, 不含控制器 IP、设备 ID 或排除列表。


# 模块级单例，所有 MicroDM 实例共用
MICRO_DM_CONFIG = ConfigHandler(_MICRO_DM_CONFIG_DIR, "micro_dm", MicroDMParams)

#: 历史别名 —— 外部以 ``MAX_CHANNELS`` 指单台控制器的通道数。
MAX_CHANNELS: int = CHANNELS_PER_CONTROLLER


# =============================================================================
# 电压换算
# =============================================================================
def voltages_to_payload(
    voltages: npt.NDArray[np.floating] | list[float] | float,
) -> bytes:
    """把电压转换为 0x09 命令的载荷。

    支持单个 float (返回 2 字节) 或数组 (返回 2*N 字节)。
    向量化 numpy 版本 —— 一次遍历完成钳位、缩放与高/低字节交织。

    协议参考 (来自 R50PowerV1.m MATLAB):
        value = (voltage + 20) / 20 / 3.4 / 3.3 * 65535.0
        highByte = floor(value / 255)
        lowByte = floor(mod(value, 256))

    # 注意
    理论上一致的实现 (统一的字节提取):
        MATLAB 实现存在不一致: 高字节除法用 255, 而低字节 (经 mod) 用 256。
        理论上正确的做法应为:
            raw = round(value)  # 正确地四舍五入到整数
            high = raw // 256   # 与 low = raw % 256 保持一致
            low = raw % 256
        这可保证在整个取值范围内 high * 256 + low == raw。
        但为兼容硬件, 这里保留 MATLAB 的行为。

    Args:
        voltages: 单个电压 (float) 或电压数组 (list/np.ndarray)。

    Returns:
        以 bytes 对象返回的高/低字节交织结果。
    """
    v = np.asarray(voltages, dtype=np.float32)
    if v.ndim == 0:
        v = v.reshape(1)
    elif not v.flags["C_CONTIGUOUS"]:
        v = np.ascontiguousarray(v, dtype=np.float32)
    np.clip(v, VOLTAGE_MIN, VOLTAGE_MAX, out=v)
    v *= PAYLOAD_SCALE
    v += PAYLOAD_OFFSET
    raw = np.round(v).astype(np.uint16)
    # 转换为 big-endian 字节序，使 uint16 内存布局为 [high_byte, low_byte]
    raw_be = raw.byteswap().view(np.uint8)
    return raw_be.tobytes()


# =============================================================================
# 异常
# =============================================================================


class MicroDMError(Exception):
    """MicroDM 错误基类异常。"""


class MicroDMConnectionError(MicroDMError):
    """连接控制器失败时抛出。"""


class MicroDMVoltageError(MicroDMError):
    """电压取值超出范围时抛出。"""


# =============================================================================
# 继电器状态
# =============================================================================


class RelayState(IntEnum):
    """继电器开/关状态。"""

    OFF = 0
    ON = 1


# =============================================================================
# 底层同步 R50 控制器
# =============================================================================


class R50Controller:
    """单台 R50Power 控制器 (50 通道) 的同步 TCP 客户端。

    MicroDM 内部使用的底层辅助类。每个实例管理到一台物理电源单元的
    持久 TCP 连接。

    在适用处实现与 :class:`DM` 相同的方法名
    (``open``/``close``/``is_connected``), 以便与 DM 接口自然组合。

    Attributes:
        controller_id: 从 1 开始的控制器标识。
        ip: IP 地址字符串。
        port: TCP 端口号。
    """

    def __init__(
        self,
        controller_id: int,
        ip: str,
        port: int,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.controller_id = controller_id
        self.ip = ip
        self.port = port
        self._timeout = timeout

        self._socket: socket.socket | None = None

    # ---- 属性 ------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """检查 TCP 连接是否已建立。"""
        return self._socket is not None

    # ---- 上下文管理器 -----------------------------------------------------

    def __enter__(self) -> R50Controller:
        """上下文管理器入口 —— 打开 TCP 连接。

        用法::

            with R50Controller(1, "192.168.0.101", 10101) as ctrl:
                ctrl.set_all_channel_voltage(0.0)
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口 —— 关闭 TCP 连接。"""
        self.close()

    # ---- 连接管理 --------------------------------------------------------

    def open(self) -> bool:
        """打开到控制器的 TCP 连接。

        Returns:
            成功返回 True, 失败返回 False。
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self._timeout)
            sock.connect((self.ip, self.port))
            # 禁用 Nagle算法（减少小包延迟，适合低延迟控制）
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._socket = sock
            logger.debug(
                f"R50Controller[{self.controller_id}] connected to {self.ip}:{self.port}"
            )
            return True
        except (socket.timeout, OSError, ConnectionError) as exc:
            logger.warning(f"R50Controller[{self.controller_id}] connect failed: {exc}")
            self._socket = None
            return False

    def close(self) -> None:
        """关闭 TCP 连接。"""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
            logger.debug(f"R50Controller[{self.controller_id}] disconnected")

    # ---- 命令发送 --------------------------------------------------------

    def send(self, data: bytes) -> bool:
        """向控制器发送原始命令字节。

        Args:
            data: 完整的命令包 (帧头 + 载荷 + 帧尾)。

        Returns:
            成功返回 True。失败时会把该控制器标记为已断开。
        """
        if self._socket is None:
            return False
        try:
            self._socket.sendall(data)
            return True
        except (OSError, ConnectionError) as exc:
            logger.warning(f"R50Controller[{self.controller_id}] send error: {exc}")
            self._socket = None
            return False

    def set_all_channel_voltage(self, voltage: float) -> bool:
        """把所有 50 个通道设为同一电压 (命令 0x08)。

        Args:
            voltage: 电压, 单位伏特 (钳位到 [-20, 120])。

        Returns:
            成功返回 True。
        """
        payload = voltages_to_payload(voltage)
        hv, lv = payload[0], payload[1]
        cmd = HEADER + bytes([CMD_SET_ALL_CHANNEL_VOLTAGE, hv, lv]) + FOOTER
        return self.send(cmd)

    def set_channel_voltage(self, channel: int, voltage: float) -> bool:
        """设置单个通道的电压 (命令 0x04)。

        Args:
            channel: 通道索引 (0-49)。
            voltage: 电压, 单位伏特。

        Returns:
            成功返回 True, 通道越界返回 False。
        """
        if not 0 <= channel < MAX_CHANNELS:
            logger.warning(
                f"R50Controller[{self.controller_id}] invalid channel: {channel}"
            )
            return False
        payload = voltages_to_payload(voltage)
        hv, lv = payload[0], payload[1]
        cmd = HEADER + bytes([CMD_SET_CHANNEL_VOLTAGE, channel, hv, lv]) + FOOTER
        return self.send(cmd)

    def set_all_voltage_array(self, voltages: list[float]) -> bool:
        """按数组设置全部 50 个通道 (命令 0x09, 最快的方式)。

        Args:
            voltages: 恰好 50 个电压值构成的列表。

        Returns:
            成功返回 True, 数组长度不是 50 返回 False。
        """
        if len(voltages) != MAX_CHANNELS:
            logger.warning(
                f"R50Controller[{self.controller_id}] expected {MAX_CHANNELS} voltages, "
                f"got {len(voltages)}"
            )
            return False

        cmd = (
            HEADER
            + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR])
            + voltages_to_payload(voltages)
            + FOOTER
        )
        return self.send(cmd)

    def set_relay(self, state: bool) -> bool:
        """打开 (True) 或关闭 (False) 继电器。

        命令 0x06 = 打开, 0x07 = 关闭。
        """
        cmd = HEADER + bytes([CMD_RELAY_ON if state else CMD_RELAY_OFF]) + FOOTER
        return self.send(cmd)

    def power_off_and_close(self, home_voltage: float = 0.0) -> bool:
        """安全关机: 先把所有通道归零, 继电器断开, 再关闭连接。

        在切断继电器电源之前先把输出电压归零, 使镜面在控制器仍带电时
        回到参考位置。CLI 工具与 GUI 共用这一调用。

        Returns:
            电压命令与继电器命令都成功时返回 True。
        """
        ok1 = self.set_all_channel_voltage(home_voltage)
        ok2 = self.set_relay(False)
        self.close()
        return bool(ok1 and ok2)


# =============================================================================
# 主 MicroDM 驱动
# =============================================================================


@register_dm("micro")
class MicroDM(DM, Device):
    """Micro DM (R50Power) 变形镜驱动。

    通过同步 TCP 控制一台或多台 R50Power 控制器。默认使用单台 50 通道
    控制器，地址为 192.168.0.101:10101。

    提供多个 IP 时, 通道按顺序分配::

        controller 0  →  channels   0-49
        controller 1  →  channels  50-99
        ...

    与所有控制器的全部 TCP 通信均为同步进行,
    不使用异步事件循环。

    Attributes:
        DM_Num: 逻辑通道总数 (每台控制器 50 个)。
        V_Min: 最小电压 (-20.0 V)。
        V_Max: 最大电压 (120.0 V)。

    示例:
        >>> dm = MicroDM()
        >>> dm.open()
        >>> dm.send_voltages(np.zeros(50))
        >>> dm.set_relay_state(True)
        >>> print(dm.get_actuator_positions())
        >>> dm.close()
    """

    DM_Num: int = DM_NUM
    DM_NUM: int = DM_NUM  # 与基类兼容的别名
    V_Min: float = VOLTAGE_MIN
    V_Max: float = VOLTAGE_MAX
    max_neibor_diff: float = float("inf")  # 无邻居约束

    device_type = DeviceType.DM
    manufacturer = "R50Power"
    model = "MicroDM"

    @classmethod
    def is_reachable(cls) -> bool:
        """检查是否至少有一台 R50Power 控制器在 TCP 上可达。"""
        for suffix in range(IP_SUFFIX_MIN, IP_SUFFIX_MAX + 1):
            ip = f"192.168.0.{suffix}"
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex((ip, PORT_BASE + suffix))
                sock.close()
                if result == 0:
                    return True
            except OSError:
                continue
        return False

    @property
    def default_dm_unit_mask(self) -> npt.NDArray[np.bool_]:
        return np.ones(self.DM_NUM, dtype=bool)

    def __init__(
        self,
        ips: list[str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        device_id: str = "",
        use_wiring_map: bool = True,
        exclude_ips: list[str] | None = None,
        exclude_ids: list[int] | None = None,
        safety_mode: bool = True,
    ):
        """初始化 MicroDM 驱动。

        Args:
            ips: R50Power 控制器的 IP 地址。
                默认: 若 ``use_wiring_map=True`` 则从接线表加载,
                否则为 ``["192.168.0.101"]`` (单台控制器)。
                多控制器 setups 请传入多个 IP。
            timeout: TCP 连接/发送超时, 单位秒。
            device_id: 唯一设备标识 (为空时自动生成)。
            use_wiring_map: 为 True (默认) 时从
                ``libs/micro_drive1300/wiring_map.json`` 加载控制器 IP。
            exclude_ips: 初始化时要跳过的 IP 地址。
                这些 IP 对应的控制器不会被创建。
            exclude_ids: 初始化时要跳过的控制器 ID (从 1 开始)。
                这些 ID 对应的控制器不会被创建。
            safety_mode: 为 True (默认) 时, send_voltages 会从当前状态
                斜坡到目标值, 每步变化量不超过 max_neibor_diff。
        """
        self._init_values = {
            "timeout": timeout,
            "use_wiring_map": use_wiring_map,
            "safety_mode": safety_mode,
        }
        # 使用 defaults + __init__ 参数解析可持久化的标量参数
        params = MICRO_DM_CONFIG.resolve_from_config({}, init_values=self._init_values)

        DM.__init__(self, safety_mode=params.safety_mode)
        Device.__init__(self, device_id)

        # 启用则加载接线表
        self._wiring_map: WiringMap | None = None
        self._channel_by_position: dict[
            int, ChannelInfo
        ] = {}  # physical_position → info
        self._channel_by_ip_payload: dict[
            tuple[int, int], ChannelInfo
        ] = {}  # (ip_suffix, payload_pos) → info
        self._channel_by_xy: dict[
            tuple[int, int], ChannelInfo
        ] = {}  # (x, y) in 39x39 → info

        if params.use_wiring_map:
            self._wiring_map = WiringMap.from_file(WIRING_MAP_PATH)
            if self._wiring_map is not None:
                self._build_channel_indices(self._wiring_map)

        # 从接线表确定 IP, 否则用默认值
        if ips is not None:
            self._ips = ips
        elif self._wiring_map is not None:
            self._ips = self._wiring_map.unique_ips
        else:
            self._ips = [DEFAULT_IPS[0]]

        # 过滤掉被排除的 IP
        exclude_ip_set = set(exclude_ips or [])
        self._ips = [ip for ip in self._ips if ip not in exclude_ip_set]

        if exclude_ip_set:
            excluded_found = exclude_ip_set & set(self._ips)
            for ip in excluded_found:
                logger.warning(f"Excluded controller IP: {ip}")

        self._timeout = params.timeout

        self._relay_state = RelayState.OFF

        # 构建同步控制器, 跳过被排除的 ID
        exclude_id_set = set(exclude_ids or [])
        self._controllers: list[R50Controller] = [
            R50Controller(
                controller_id=i,
                ip=ip_str,
                port=PORT_BASE + int(ip_str.split(".")[-1]),
                timeout=params.timeout,
            )
            for i, ip_str in enumerate(self._ips, start=1)
            if (i + 1) not in exclude_id_set
        ]

        if exclude_id_set:
            for cid in exclude_id_set:
                logger.warning(f"Excluded controller ID: {cid}")

        # 注册设备参数
        self._register_parameters()

        logger.debug(
            f"MicroDM initialized: {len(self._controllers)} controller(s), "
            f"{self.DM_Num} channels, "
            f"voltage range [{self.V_Min}, {self.V_Max}] V"
        )

    # ---- 参数注册 --------------------------------------------------------

    def _register_parameters(self) -> None:
        """为参数管理系统注册设备参数。"""
        self.register_parameter(
            "voltage_min",
            self.V_Min,
            description="Minimum allowed voltage (V)",
        )
        self.register_parameter(
            "voltage_max",
            self.V_Max,
            description="Maximum allowed voltage (V)",
        )
        self.register_parameter(
            "channel_count",
            self.DM_Num,
            description="Total logical channel count",
        )
        self.register_parameter(
            "n_controllers",
            len(self._controllers),
            description="Number of physical R50Power controllers",
        )

    # ---- 接线表方法 ------------------------------------------------------

    def _build_channel_indices(self, wiring_map: WiringMap) -> None:
        """从解析出的 WiringMap 构建查找索引。

        创建三个用于 O(1) 通道查找的索引:
        - _channel_by_position: physical_position → ChannelInfo
        - _channel_by_ip_payload: (ip_suffix, payload_position) → ChannelInfo
        - _channel_by_xy: 39x39 网格中的 (x, y) → ChannelInfo
        """
        for group_key, group in wiring_map.groups.items():
            for entry in group.channels:
                if not entry.is_valid:
                    continue

                info = ChannelInfo.from_entry(entry, group.name, group_key)

                # 按物理位置索引 (安全: is_valid 保证非 None)
                assert entry.physical_position is not None
                self._channel_by_position[entry.physical_position] = info

                # 按 (ip_suffix, payload_position) 索引
                if entry.ip_suffix is not None and entry.payload_position is not None:
                    self._channel_by_ip_payload[
                        (entry.ip_suffix, entry.payload_position)
                    ] = info

                # 从 physical_label 解析出 (x, y) 并索引 (格式: "group-row-col")
                if entry.physical_label is not None:
                    parts = entry.physical_label.split("-")
                    if len(parts) == 3:
                        try:
                            row = int(parts[1])  # y 坐标 (从 1 开始)
                            col = int(parts[2])  # x 坐标 (从 1 开始)
                            self._channel_by_xy[(col, row)] = info
                        except ValueError:
                            pass

        logger.debug(
            f"Built channel indices: {len(self._channel_by_position)} by position, "
            f"{len(self._channel_by_ip_payload)} by ip/payload, "
            f"{len(self._channel_by_xy)} by xy"
        )

    @property
    def wiring_map(self) -> WiringMap | None:
        """访问已加载的接线表 (只读)。"""
        return self._wiring_map

    def get_channel_by_xy(self, x: int, y: int) -> ChannelInfo | None:
        """按 39×39 阵列中的 x, y 坐标获取通道信息。

        Args:
            x: 列索引 (从 1 开始, 1-39)。
            y: 行索引 (从 1 开始, 1-39)。

        Returns:
            找到则返回 ChannelInfo, 否则为 None。
        """
        return self._channel_by_xy.get((x, y))

    def get_channel_by_ip_position(
        self, ip_suffix: int, payload_position: int
    ) -> ChannelInfo | None:
        """按控制器 IP 后缀与载荷位置获取通道信息。

        Args:
            ip_suffix: IP 地址后缀 (例如 101 对应 192.168.0.101)。
            payload_position: 控制器内的通道位置 (1-50)。

        Returns:
            找到则返回 ChannelInfo, 否则为 None。
        """
        return self._channel_by_ip_payload.get((ip_suffix, payload_position))

    # ---- 设备接口 --------------------------------------------------------

    def open(self) -> None:
        """打开到所有 R50Power 控制器的连接。

        同步连接到每台控制器。单台连接失败会记为警告日志,
        但不影响其他控制器继续连接。

        Raises:
            MicroDMConnectionError: 没有任何控制器能连上时。
        """
        self._set_state(DeviceState.CONNECTING)

        connected, failed = 0, []
        for ctrl in self._controllers:
            if ctrl.open():
                connected += 1
            else:
                failed.append((ctrl.controller_id, ctrl.ip))

        for ctrl_id, ip in failed:
            logger.warning(f"Controller[{ctrl_id}] {ip} connection failed")

        if connected == 0:
            self._set_state(DeviceState.ERROR, "No controllers connected")
            raise MicroDMConnectionError(
                f"Failed to connect to any of {len(self._controllers)} controller(s)"
            )

        self._set_state(DeviceState.READY)
        logger.info(
            f"MicroDM ready: {connected}/{len(self._controllers)} controllers connected"
        )

    def close(self) -> None:
        """关闭所有控制器连接。"""
        for ctrl in self._controllers:
            try:
                ctrl.close()
            except Exception as exc:
                logger.warning(f"Error closing controller: {exc}")

        self._set_state(DeviceState.DISCONNECTED)
        logger.info("MicroDM disconnected")

    def is_connected(self) -> bool:
        """检查是否至少有一台控制器已连接并就绪。

        Returns:
            至少一台控制器有活动 TCP 连接时返回 True。
        """
        return self._state == DeviceState.READY and any(
            ctrl.is_connected for ctrl in self._controllers
        )

    def get_hardware_info(self) -> dict[str, Any]:
        """获取硬件专属信息。

        Returns:
            含设备信息、连接状态与取值范围的字典。
        """
        connected_count = sum(1 for c in self._controllers if c.is_connected)
        return {
            "manufacturer": self.manufacturer,
            "model": self.model,
            "n_controllers": len(self._controllers),
            "connected_controllers": connected_count,
            "channels_per_controller": MAX_CHANNELS,
            "total_channels": self.DM_Num,
            "voltage_range": [self.V_Min, self.V_Max],
            "relay_state": self._relay_state.name,
            "controller_ips": [ctrl.ip for ctrl in self._controllers],
            "controller_ports": [ctrl.port for ctrl in self._controllers],
        }

    def get_actuator_positions(self) -> npt.NDArray[np.floating]:
        """获取当前的致动器电压。

        Returns:
            当前电压数组的副本 (DM_Num 个元素)。
        """
        return self._last_voltages.copy()

    # ---- DM 接口 ---------------------------------------------------------

    def transform(self, cmd: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """把归一化命令 ``[-1, 1]`` 转换为设备电压范围。

        将 [-1, 1] 线性映射到 [V_Min, V_Max]。

        Args:
            cmd: 归一化命令数组。

        Returns:
            设备范围内的电压数组。
        """
        return self.transform_voltage(cmd)

    def send(self, cmd: npt.NDArray[np.floating] | float) -> npt.NDArray[np.floating]:
        """向所有通道发送电压命令。

        Args:
            cmd: 电压数组 (DM_Num 个元素), 或一个作用于所有通道的标量 float。

        Returns:
            施加后的电压数组。

        Raises:
            MicroDMVoltageError: 命令类型不受支持时。
        """
        if isinstance(cmd, np.ndarray):
            return self.send_voltages(cmd)
        if isinstance(cmd, (int, float)):
            return self.set_all_channel_voltage(float(cmd))
        raise MicroDMVoltageError(f"Unsupported command type: {type(cmd)}")

    def _apply_voltages(self, vs: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """经同步 TCP 向所有控制器施加电压的底层实现。

        通道按顺序轮转分配到各控制器:

            controller 0  →  vs[0:50]
            controller 1  →  vs[50:100]
            ...
        """
        vs = np.clip(vs, self.V_Min, self.V_Max)
        failed: list[str] = []
        for ctrl_idx, ctrl in enumerate(self._controllers):
            start = ctrl_idx * MAX_CHANNELS
            end = start + MAX_CHANNELS
            if start >= len(vs):
                break
            chunk = vs[start:end]
            if len(chunk) < MAX_CHANNELS:
                chunk = np.pad(
                    chunk, (0, MAX_CHANNELS - len(chunk)), constant_values=0.0
                )

            cmd = (
                HEADER
                + bytes([CMD_SET_ALL_VOLTAGE_BY_ARR])
                + voltages_to_payload(chunk)
                + FOOTER
            )
            ok = ctrl.send(cmd)
            if not ok:
                failed.append(ctrl.ip)

        if failed:
            raise MicroDMConnectionError(
                f"电压下发失败 (控制器未响应): {', '.join(failed)}"
            )

        self._last_voltages = vs.copy()
        return self._last_voltages

    def send_voltages(
        self, vs: npt.NDArray[np.floating], wait_time_s: float = 0.001
    ) -> npt.NDArray[np.floating]:
        """用同步并行 TCP 把电压数组发送到所有通道。

        电压会自动分配到各台 R50Power 控制器。
        safety_mode 为 True (默认) 时, 电压会从当前状态斜坡到目标值,
        每步变化量不超过 max_neibor_diff。

        Args:
            vs: 所有逻辑通道的电压数组。
            wait_time_s: 发送后休眠时间 (硬件稳定时间)。

        Returns:
            施加后的电压数组。

        Raises:
            MicroDMVoltageError: 数组长度与 DM_Num 不符时。
        """
        vs = np.asarray(vs, dtype=np.float64)
        if vs.shape != (self.DM_Num,):
            raise MicroDMVoltageError(
                f"Expected {self.DM_Num} voltages, got {vs.shape}"
            )
        result = super().send_voltages(vs, wait_time_s=wait_time_s)
        # 基类会回显其输入类型, 但上面的 ``vs`` 已被强制转为
        # 普通 ndarray (``DmCommands`` 没有数组协议, 因此本来就会
        # 被前面的形状检查拦下)。
        assert isinstance(result, np.ndarray)
        return result

    # ---- 协议命令 --------------------------------------------------------

    def set_channel_voltage(self, channel: int, voltage: float) -> None:
        """设置单个逻辑通道的电压。

        自动路由到正确的物理控制器。

        Args:
            channel: 逻辑通道索引 (0 到 DM_Num - 1)。
            voltage: 电压, 单位伏特。

        Raises:
            MicroDMVoltageError: 通道越界时。
        """
        if not 0 <= channel < self.DM_Num:
            raise MicroDMVoltageError(
                f"Channel must be 0-{self.DM_Num - 1}, got {channel}"
            )

        ctrl_idx = channel // MAX_CHANNELS
        ch_idx = channel % MAX_CHANNELS

        if ctrl_idx < len(self._controllers):
            ctrl = self._controllers[ctrl_idx]
            voltage_clipped = max(self.V_Min, min(self.V_Max, float(voltage)))
            ctrl.set_channel_voltage(ch_idx, voltage_clipped)
            self._last_voltages[channel] = voltage_clipped

    def set_all_channel_voltage(self, voltage: float) -> npt.NDArray[np.floating]:
        """把所有逻辑通道设为同一电压。

        发送到所有控制器。

        Args:
            voltage: 所有通道的电压, 单位伏特。

        Returns:
            施加后的电压数组。
        """
        voltage = max(self.V_Min, min(self.V_Max, float(voltage)))
        vs = np.full(self.DM_Num, voltage)
        self.send_voltages(vs)
        logger.debug(f"Set all channels to {voltage} V")
        return self._last_voltages.copy()

    def set_relay_state(self, state: bool) -> None:
        """设置所有已连接控制器上的继电器状态。

        Args:
            state: True 打开继电器, False 关闭。
        """
        cmd = HEADER + bytes([CMD_RELAY_ON if state else CMD_RELAY_OFF]) + FOOTER
        for ctrl in self._controllers:
            if ctrl.is_connected:
                ctrl.send(cmd)
        self._relay_state = RelayState.ON if state else RelayState.OFF
        logger.info(f"Relay {'opened' if state else 'closed'} on all controllers")

    def reset_all(self) -> None:
        """把所有通道复位到 0 V。"""
        vs = np.zeros(self.DM_Num)
        self.send_voltages(vs)
        logger.info("MicroDM reset to 0 V")

    def __repr__(self) -> str:
        connected = sum(1 for c in self._controllers if c.is_connected)
        return (
            f"MicroDM("
            f"controllers={len(self._controllers)}, "
            f"connected={connected}, "
            f"channels={self.DM_Num}, "
            f"voltage=[{self.V_Min}, {self.V_Max}] V, "
            f"state={self._state.name}"
            f")"
        )


__all__ = [
    "MAX_ACTUATORS",
    "MAX_CHANNELS",
    "MAX_CONTROLLERS",
    "MICRO_DM_CONFIG",
    "MicroDM",
    "MicroDMConnectionError",
    "MicroDMError",
    "MicroDMParams",
    "MicroDMVoltageError",
    "R50Controller",
    "RelayState",
    "voltages_to_payload",
]
