"""数字孪生管理的设备基类。

本模块为所有硬件设备提供统一接口, 便于设备注册、元数据管理与数字孪生支持。
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, ClassVar, Self
from collections.abc import Callable

from loguru import logger


class DeviceType(Enum):
    """设备类型分类。"""

    CAMERA = auto()
    SLM = auto()
    DM = auto()  # 变形镜
    WFS = auto()  # 波前传感器
    POWER_METER = auto()  # 光功率计 (例如 Thorlabs PM100)
    STAGE = auto()  # 运动台
    LASER = auto()
    FILTER = auto()
    OTHER = auto()


class DeviceState(Enum):
    """设备运行状态。"""

    UNKNOWN = auto()
    DISCONNECTED = auto()
    CONNECTING = auto()
    READY = auto()
    BUSY = auto()
    ERROR = auto()
    CALIBRATING = auto()


@dataclass
class DeviceParameter:
    """设备参数元数据。

    Attributes:
        name: 参数名。
        value: 当前值。
        value_type: 参数的数据类型。
        min_value: 允许的最小值 (可选)。
        max_value: 允许的最大值 (可选)。
        unit: 物理单位 (例如 "ms"、"nm"、"V")。
        description: 面向人阅读的描述。
        writable: 该参数是否可设置。
    """

    name: str
    value: Any
    value_type: type = float
    min_value: float | None = None
    max_value: float | None = None
    unit: str = ""
    description: str = ""
    writable: bool = True

    def validate(self, value: Any) -> bool:
        """校验 value 是否落在允许范围内。"""
        if not isinstance(value, self.value_type):
            try:
                value = self.value_type(value)
            except (TypeError, ValueError):
                return False

        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


@dataclass
class DeviceCapability:
    """设备能力元数据。

    Attributes:
        name: 能力名。
        description: 面向人阅读的描述。
        parameters: 该能力所需的参数。
        return_type: 期望的返回类型。
    """

    name: str
    description: str = ""
    parameters: list[str] = field(default_factory=list)
    return_type: type | None = None


@dataclass
class DeviceMetadata:
    """用于数字孪生注册的设备元数据。

    Attributes:
        device_id: 唯一设备标识 (UUID)。
        device_type: 设备分类。
        manufacturer: 设备制造商。
        model: 设备型号名。
        serial_number: 硬件序列号。
        firmware_version: 固件版本。
        hardware_version: 硬件版本。
        connection_info: 连接参数 (地址、端口等)。
        registration_time: 设备注册时间。
        last_seen: 最近一次通信的时间戳。
    """

    device_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    device_type: DeviceType = DeviceType.OTHER
    manufacturer: str = "Unknown"
    model: str = "Unknown"
    serial_number: str = ""
    firmware_version: str = ""
    hardware_version: str = ""
    connection_info: dict[str, Any] = field(default_factory=dict)
    registration_time: datetime = field(default_factory=datetime.now)
    last_seen: datetime | None = None


class Device(ABC):
    """所有硬件设备的抽象基类。

    本类为设备管理提供统一接口, 使数字孪生功能与标准化设备控制成为可能。

    所有设备驱动都必须继承本类并实现其抽象方法。

    示例:
        >>> class MyCamera(Device):
        ...     device_type = DeviceType.CAMERA
        ...     manufacturer = "MyCam"
        ...     model = "MC-100"
        ...
        ...     def __init__(self, device_id: str = ""):
        ...         super().__init__(device_id)
        ...         self._register_parameters()
        ...
        ...     def _register_parameters(self) -> None:
        ...         self.register_parameter(
        ...             "exposure_time",
        ...             20.0,
        ...             min_value=1.0,
        ...             max_value=1000.0,
        ...             unit="ms",
        ...             description="Exposure time in milliseconds"
        ...         )
    """

    # 类级设备标识
    device_type: ClassVar[DeviceType] = DeviceType.OTHER
    manufacturer: ClassVar[str] = "Unknown"
    model: ClassVar[str] = "Unknown"
    version: ClassVar[str] = "1.0.0"

    def __init__(self, device_id: str = ""):
        """初始化设备基类。

        Args:
            device_id: 唯一设备标识。为空时自动生成。
        """
        # 设备标识
        self._device_id = device_id or str(uuid.uuid4())
        self._state = DeviceState.DISCONNECTED
        self._error_message: str | None = None

        # 参数注册表: name -> DeviceParameter
        self._parameters: dict[str, DeviceParameter] = {}

        # 能力注册表: name -> DeviceCapability
        self._capabilities: dict[str, DeviceCapability] = {}

        # 流式采集的数据回调
        self._data_callbacks: list[Callable[[str, Any], None]] = []

        # 初始化元数据
        self._metadata = DeviceMetadata(
            device_id=self._device_id,
            device_type=self.device_type,
            manufacturer=self.manufacturer,
            model=self.model,
        )

        # 注册标准能力
        self._register_standard_capabilities()

        logger.debug(f"Device {self._device_id} ({self.model}) initialized")

    # ==================== 原生 SDK (惰性解析) ====================

    _sdk: Any = None

    @staticmethod
    def _load_sdk() -> Any:
        """解析本驱动所需的原生 SDK 句柄。

        默认返回 ``None``: 纯软件设备与仿真设备没有原生句柄。基于 SDK 的驱动覆写本方法。
        它刻意与 :meth:`_ensure_sdk` 分开, 是为了让解析能被显式触发 (由 ``open()``),
        而不是由属性访问触发。
        """
        return None

    def _ensure_sdk(self) -> Any:
        """返回原生 SDK 句柄, 首次使用时加载并缓存。

        构造过程不得触碰 SDK。那些在 ``__init__`` 里就绑定原生库的驱动, 在没有厂商运行时的
        机器上根本无法构造、注册或自省 —— 这会阻断离线工具, 也让硬件/仿真替换极易出错。
        改在这里解析意味着 SDK 只由 ``open()`` 索取, 而且仅由 ``open()`` 索取。
        """
        if self._sdk is None:
            self._sdk = self._load_sdk()
        return self._sdk

    # ==================== 抽象方法 ====================

    @abstractmethod
    def open(self) -> None:
        """打开与设备的连接。

        Raises:
            ConnectionError: 连接失败时。
            DeviceError: 设备初始化失败时。
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭连接并释放资源。"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """检查设备是否已连接并就绪。"""
        pass

    @abstractmethod
    def get_hardware_info(self) -> dict[str, Any]:
        """获取硬件专属信息。

        Returns:
            包含硬件信息 (序列号、固件等) 的字典。
        """
        pass

    # ==================== 上下文管理器 ====================

    def __enter__(self) -> Self:
        """上下文管理器入口。

        Returns:
            Self (实际的子类实例, 而非基类 ``Device``) —— 让类型检查器能在
            ``with SomeDevice(...) as dev:`` 上解析出子类专属属性
            (例如 ``wfs.optimize_pupil()``)。
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口。"""
        self.close()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self._device_id[:8]}, "
            f"model={self.manufacturer}_{self.model}, "
            f"state={self._state.name}"
            f")"
        )

    # ==================== 设备注册 ====================

    @property
    def device_id(self) -> str:
        """唯一设备标识。"""
        return self._device_id

    @property
    def metadata(self) -> DeviceMetadata:
        """用于注册的设备元数据。"""
        # 更新时间戳
        if self.is_connected():
            self._metadata.last_seen = datetime.now()
        return self._metadata

    @property
    def state(self) -> DeviceState:
        """当前设备状态。"""
        return self._state

    def _set_state(self, state: DeviceState, error_msg: str | None = None) -> None:
        """更新设备状态。"""
        old_state = self._state
        self._state = state
        self._error_message = error_msg

        if state == DeviceState.ERROR and error_msg:
            logger.error(f"Device {self._device_id} error: {error_msg}")
        elif old_state != state:
            logger.debug(f"Device {self._device_id} state: {old_state.name} -> {state.name}")

    # ==================== 参数管理 ====================

    def register_parameter(
        self,
        name: str,
        default_value: Any,
        min_value: float | None = None,
        max_value: float | None = None,
        unit: str = "",
        description: str = "",
        writable: bool = True,
    ) -> None:
        """注册一个设备参数。

        Args:
            name: 参数名。
            default_value: 默认值。
            min_value: 允许的最小值。
            max_value: 允许的最大值。
            unit: 物理单位。
            description: 面向人阅读的描述。
            writable: 该参数是否可修改。
        """
        self._parameters[name] = DeviceParameter(
            name=name,
            value=default_value,
            value_type=type(default_value),
            min_value=min_value,
            max_value=max_value,
            unit=unit,
            description=description,
            writable=writable,
        )

    def get_parameter(self, name: str) -> DeviceParameter | None:
        """获取参数元数据。"""
        return self._parameters.get(name)

    def get_parameter_value(self, name: str) -> Any:
        """获取参数的当前值。"""
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")
        return self._parameters[name].value

    def set_parameter_value(self, name: str, value: Any) -> bool:
        """带校验地设置参数值。

        Args:
            name: 参数名。
            value: 新值。

        Returns:
            成功返回 True, 校验失败返回 False。

        Raises:
            KeyError: 参数不存在时。
            PermissionError: 参数为只读时。
        """
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")

        param = self._parameters[name]
        if not param.writable:
            raise PermissionError(f"Parameter '{name}' is read-only")

        if not param.validate(value):
            logger.warning(f"Invalid value {value} for parameter '{name}'")
            return False

        # 更新值
        old_value = param.value
        param.value = value

        # 调用参数变更钩子
        self._on_parameter_changed(name, old_value, value)

        logger.debug(f"Parameter '{name}': {old_value} -> {value}")
        return True

    def list_parameters(self) -> list[str]:
        """列出所有已注册的参数名。"""
        return list(self._parameters.keys())

    def get_all_parameters(self) -> dict[str, DeviceParameter]:
        """获取所有已注册的参数。"""
        return self._parameters.copy()

    def _on_parameter_changed(self, name: str, old_value: Any, new_value: Any) -> None:
        """参数变化时被调用的钩子。

        在子类中覆写以实现设备专属逻辑。
        """
        pass

    # ==================== 能力管理 ====================

    def register_capability(
        self,
        name: str,
        description: str = "",
        parameters: list[str] | None = None,
        return_type: type | None = None,
    ) -> None:
        """注册一项设备能力。

        Args:
            name: 能力名。
            description: 面向人阅读的描述。
            parameters: 所需的参数名。
            return_type: 期望的返回类型。
        """
        self._capabilities[name] = DeviceCapability(
            name=name,
            description=description,
            parameters=parameters or [],
            return_type=return_type,
        )

    def has_capability(self, name: str) -> bool:
        """检查设备是否拥有某项能力。"""
        return name in self._capabilities

    def get_capability(self, name: str) -> DeviceCapability | None:
        """获取能力元数据。"""
        return self._capabilities.get(name)

    def list_capabilities(self) -> list[str]:
        """列出所有已注册的能力名。"""
        return list(self._capabilities.keys())

    def _register_standard_capabilities(self) -> None:
        """注册所有设备共有的标准能力。"""
        self.register_capability(
            "connect",
            description="Connect to the device",
        )
        self.register_capability(
            "disconnect",
            description="Disconnect from the device",
        )
        self.register_capability(
            "get_status",
            description="Get device status information",
            return_type=dict,
        )

    # ==================== 数据采集 ====================

    def register_data_callback(self, callback: Callable[[str, Any], None]) -> None:
        """注册一个数据流回调。

        Args:
            callback: 新数据到来时以 (data_type, data) 调用的函数。
        """
        self._data_callbacks.append(callback)

    def unregister_data_callback(self, callback: Callable[[str, Any], None]) -> None:
        """注销一个数据回调。"""
        if callback in self._data_callbacks:
            self._data_callbacks.remove(callback)

    def _emit_data(self, data_type: str, data: Any) -> None:
        """把数据发给所有已注册的回调。"""
        for callback in self._data_callbacks:
            try:
                callback(data_type, data)
            except Exception as e:
                logger.warning(f"Data callback error: {e}")

    # ==================== 数字孪生支持 ====================

    def get_twin_state(self) -> dict[str, Any]:
        """获取用于数字孪生同步的当前状态。

        Returns:
            包含设备状态、参数与元数据的字典。
        """
        return {
            "device_id": self._device_id,
            "device_type": self.device_type.name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "state": self._state.name,
            "parameters": {
                name: {
                    "value": param.value,
                    "unit": param.unit,
                }
                for name, param in self._parameters.items()
            },
            "capabilities": list(self._capabilities.keys()),
            "metadata": {
                "serial_number": self._metadata.serial_number,
                "firmware_version": self._metadata.firmware_version,
                "hardware_version": self._metadata.hardware_version,
            },
        }

    def sync_from_twin(self, twin_state: dict[str, Any]) -> None:
        """从数字孪生同步设备状态。

        Args:
            twin_state: 来自数字孪生的状态字典。
        """
        # 更新参数
        if "parameters" in twin_state:
            for name, param_data in twin_state["parameters"].items():
                if name in self._parameters:
                    try:
                        self.set_parameter_value(name, param_data["value"])
                    except Exception as e:
                        logger.warning(f"Failed to sync parameter '{name}': {e}")

    # ==================== 工具方法 ====================

    def get_status(self) -> dict[str, Any]:
        """获取设备的综合状态。"""
        return {
            "device_id": self._device_id,
            "model": f"{self.manufacturer}_{self.model}",
            "state": self._state.name,
            "error": self._error_message,
            "connected": self.is_connected(),
            "parameter_count": len(self._parameters),
            "capability_count": len(self._capabilities),
        }

    def health_check(self) -> tuple[bool, str]:
        """对设备执行健康检查。

        Returns:
            (是否健康, 消息) 二元组。
        """
        if not self.is_connected():
            return False, "Device not connected"

        if self._state == DeviceState.ERROR:
            return False, self._error_message or "Device in error state"

        return True, "OK"


class DeviceError(Exception):
    """设备错误基类异常。"""

    pass


class DeviceNotFoundError(DeviceError):
    """未找到设备时抛出。"""

    pass


class DeviceBusyError(DeviceError):
    """设备忙时抛出。"""

    pass
