"""用于数字孪生管理的设备注册表。

为构建数字孪生系统提供集中的设备注册、发现与管理能力。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

from loguru import logger

from ao_shaping.drivers.device_base import (
    Device,
    DeviceError,
    DeviceState,
    DeviceType,
)


@dataclass
class RegisteredDevice:
    """已注册设备的包装, 附带额外管理信息。"""

    device: Device
    alias: str = ""  # 面向用户的名字
    tags: list[str] = field(default_factory=list)  # 用于分组/过滤
    priority: int = 0  # 设备优先级 (越大越重要)
    auto_connect: bool = False  # 注册表加载时自动连接
    twin_sync_enabled: bool = True  # 启用数字孪生同步


class DeviceRegistry:
    """管理所有硬件设备的中央注册表。

    本类提供:
    - 设备注册与发现
    - 对所有设备的统一访问
    - 数字孪生状态管理
    - 对设备组的批量操作

    示例:
        >>> registry = DeviceRegistry()
        >>>
        >>> # Register devices
        >>> registry.register(camera, alias="main_camera", tags=["imaging"])
        >>> registry.register(slm, alias="phase_modulator", tags=["modulation"])
        >>>
        >>> # Find devices
        >>> cameras = registry.find_by_type(DeviceType.CAMERA)
        >>> imaging_devices = registry.find_by_tag("imaging")
        >>>
        >>> # Batch operations
        >>> registry.connect_all()
        >>> states = registry.get_all_twin_states()
    """

    def __init__(self):
        """初始化空的设备注册表。"""
        # device_id -> RegisteredDevice
        self._devices: dict[str, RegisteredDevice] = {}

        # alias -> device_id (供快速查找)
        self._aliases: dict[str, str] = {}

        # 注册表事件的回调
        self._on_device_registered: list[Callable[[str, Device], None]] = []
        self._on_device_unregistered: list[Callable[[str, Device], None]] = []
        self._on_state_change: list[Callable[[str, DeviceState, DeviceState], None]] = []

        logger.info("Device registry initialized")

    # ==================== 注册 ====================

    def register(
        self,
        device: Device,
        alias: str = "",
        tags: list[str] | None = None,
        priority: int = 0,
        auto_connect: bool = False,
        twin_sync_enabled: bool = True,
    ) -> str:
        """把设备注册到注册表中。

        Args:
            device: 要注册的设备实例。
            alias: 设备的面向用户名字。
            tags: 用于对设备分组/过滤的标签。
            priority: 设备优先级 (越大越重要)。
            auto_connect: 注册表加载时是否自动连接。
            twin_sync_enabled: 是否启用数字孪生同步。

        Returns:
            设备 ID。

        Raises:
            ValueError: alias 已被占用时。
        """
        device_id = device.device_id

        # 检查 alias 唯一性
        if alias and alias in self._aliases:
            raise ValueError(f"Alias '{alias}' is already registered")

        # 创建注册记录
        reg = RegisteredDevice(
            device=device,
            alias=alias or device_id[:8],
            tags=tags or [],
            priority=priority,
            auto_connect=auto_connect,
            twin_sync_enabled=twin_sync_enabled,
        )

        self._devices[device_id] = reg
        if alias:
            self._aliases[alias] = device_id

        logger.info(f"Registered device: {alias or device_id[:8]} ({device.model})")

        # 通知回调
        for callback in self._on_device_registered:
            try:
                callback(device_id, device)
            except Exception as e:
                logger.warning(f"Registration callback error: {e}")

        return device_id

    def unregister(self, device_id: str) -> bool:
        """注销一个设备。

        Args:
            device_id: 设备 ID 或 alias。

        Returns:
            设备被找到并移除时返回 True。
        """
        # 必要时解析 alias
        device_id = self._resolve_id(device_id)

        if device_id not in self._devices:
            return False

        reg = self._devices.pop(device_id)

        # 移除 alias 映射
        if reg.alias in self._aliases:
            del self._aliases[reg.alias]

        # 已连接则关闭设备
        try:
            reg.device.close()
        except (DeviceError, ConnectionError, RuntimeError) as e:
            logger.warning(f"Error closing device during unregister: {e}")

        logger.info(f"Unregistered device: {reg.alias}")

        # 通知回调
        for callback in self._on_device_unregistered:
            try:
                callback(device_id, reg.device)
            except Exception as e:
                logger.warning(f"Unregistration callback error: {e}")

        return True

    def _resolve_id(self, device_id_or_alias: str) -> str:
        """必要时把 alias 解析为设备 ID。"""
        if device_id_or_alias in self._aliases:
            return self._aliases[device_id_or_alias]
        return device_id_or_alias

    # ==================== 设备访问 ====================

    def get(self, device_id: str) -> Device | None:
        """按 ID 或 alias 获取设备。"""
        device_id = self._resolve_id(device_id)
        reg = self._devices.get(device_id)
        return reg.device if reg else None

    def __getitem__(self, device_id: str) -> Device:
        """按 ID 或 alias 获取设备 (类字典访问)。"""
        device = self.get(device_id)
        if device is None:
            raise KeyError(f"Device '{device_id}' not found")
        return device

    def __contains__(self, device_id: str) -> bool:
        """检查设备是否已注册。"""
        device_id = self._resolve_id(device_id)
        return device_id in self._devices

    def list_devices(self) -> list[str]:
        """列出所有已注册的设备 ID。"""
        return list(self._devices.keys())

    def list_aliases(self) -> list[str]:
        """列出所有已注册的 alias。"""
        return list(self._aliases.keys())

    def get_registration_info(self, device_id: str) -> RegisteredDevice | None:
        """获取某个设备的完整注册信息。"""
        device_id = self._resolve_id(device_id)
        return self._devices.get(device_id)

    # ==================== 查找 ====================

    def find_by_type(self, device_type: DeviceType) -> list[Device]:
        """查找某一类型的全部设备。"""
        return [
            reg.device
            for reg in self._devices.values()
            if reg.device.device_type == device_type
        ]

    def find_by_tag(self, tag: str) -> list[Device]:
        """查找带有某个标签的全部设备。"""
        return [
            reg.device
            for reg in self._devices.values()
            if tag in reg.tags
        ]

    def find_by_manufacturer(self, manufacturer: str) -> list[Device]:
        """查找来自某个制造商的全部设备。"""
        return [
            reg.device
            for reg in self._devices.values()
            if reg.device.manufacturer == manufacturer
        ]

    def find_by_model(self, model: str) -> list[Device]:
        """查找某个型号的全部设备。"""
        return [
            reg.device
            for reg in self._devices.values()
            if reg.device.model == model
        ]

    def find_by_state(self, state: DeviceState) -> list[Device]:
        """查找处于某个状态的全部设备。"""
        return [
            reg.device
            for reg in self._devices.values()
            if reg.device.state == state
        ]

    # ==================== 批量操作 ====================

    def connect_all(self, device_type: DeviceType | None = None) -> dict[str, bool]:
        """连接所有设备 (可按类型过滤)。

        Returns:
            device_id -> 是否成功的字典。
        """
        results = {}
        for device_id, reg in self._devices.items():
            if device_type and reg.device.device_type != device_type:
                continue
            try:
                reg.device.open()
                results[device_id] = True
            except Exception as e:
                logger.error(f"Failed to connect {reg.alias}: {e}")
                results[device_id] = False
        return results

    def disconnect_all(self, device_type: DeviceType | None = None) -> dict[str, bool]:
        """断开所有设备。"""
        results = {}
        for device_id, reg in self._devices.items():
            if device_type and reg.device.device_type != device_type:
                continue
            try:
                reg.device.close()
                results[device_id] = True
            except Exception as e:
                logger.error(f"Failed to disconnect {reg.alias}: {e}")
                results[device_id] = False
        return results

    def health_check_all(self) -> dict[str, tuple[bool, str]]:
        """对所有设备执行健康检查。

        Returns:
            device_id -> (是否健康, 消息) 的字典。
        """
        return {
            device_id: reg.device.health_check()
            for device_id, reg in self._devices.items()
        }

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """获取所有设备的状态。"""
        return {
            device_id: reg.device.get_status()
            for device_id, reg in self._devices.items()
        }

    # ==================== 数字孪生 ====================

    def get_all_twin_states(self) -> dict[str, dict[str, Any]]:
        """获取所有已注册设备的孪生状态。

        Returns:
            device_id -> 孪生状态的字典。
        """
        states = {}
        for device_id, reg in self._devices.items():
            if reg.twin_sync_enabled:
                try:
                    states[device_id] = reg.device.get_twin_state()
                except Exception as e:
                    logger.warning(f"Failed to get twin state for {reg.alias}: {e}")
        return states

    def sync_from_twin_states(self, twin_states: dict[str, dict[str, Any]]) -> dict[str, bool]:
        """从孪生状态同步设备。

        Args:
            twin_states: device_id -> 孪生状态的字典。

        Returns:
            device_id -> 是否成功的字典。
        """
        results = {}
        for device_id, state in twin_states.items():
            reg = self._devices.get(device_id)
            if not reg or not reg.twin_sync_enabled:
                results[device_id] = False
                continue

            try:
                reg.device.sync_from_twin(state)
                results[device_id] = True
            except Exception as e:
                logger.error(f"Failed to sync {reg.alias}: {e}")
                results[device_id] = False

        return results

    def get_twin_snapshot(self) -> dict[str, Any]:
        """获取用于数字孪生初始化的完整快照。

        Returns:
            包含所有设备状态与元数据的字典。
        """
        return {
            "registry_info": {
                "device_count": len(self._devices),
                "device_types": list(set(
                    reg.device.device_type.name
                    for reg in self._devices.values()
                )),
            },
            "devices": self.get_all_twin_states(),
            "aliases": {
                alias: device_id
                for alias, device_id in self._aliases.items()
            },
        }

    # ==================== 事件回调 ====================

    def on_device_registered(self, callback: Callable[[str, Device], None]) -> None:
        """注册设备注册事件的回调。"""
        self._on_device_registered.append(callback)

    def on_device_unregistered(self, callback: Callable[[str, Device], None]) -> None:
        """注册设备注销事件的回调。"""
        self._on_device_unregistered.append(callback)

    # 注: on_state_change 回调为将来使用而预留,
    # 待自动状态变更通知实现后再启用。

    # ==================== 导入/导出 ====================

    def export_config(self) -> dict[str, Any]:
        """导出注册表配置。

        Returns:
            用于持久化的配置字典。
        """
        return {
            "devices": [
                {
                    "device_id": device_id,
                    "alias": reg.alias,
                    "tags": reg.tags,
                    "priority": reg.priority,
                    "auto_connect": reg.auto_connect,
                    "twin_sync_enabled": reg.twin_sync_enabled,
                    "device_type": reg.device.device_type.name,
                    "manufacturer": reg.device.manufacturer,
                    "model": reg.device.model,
                    "parameters": {
                        name: param.value
                        for name, param in reg.device.get_all_parameters().items()
                    },
                }
                for device_id, reg in self._devices.items()
            ]
        }

    def __len__(self) -> int:
        """已注册设备的数量。"""
        return len(self._devices)

    def __iter__(self):
        """迭代已注册的设备。"""
        return iter(self._devices.values())

    def __repr__(self) -> str:
        return f"DeviceRegistry(devices={len(self._devices)}, aliases={len(self._aliases)})"


# 供便捷使用的全局注册表实例
_global_registry: DeviceRegistry | None = None


def get_global_registry() -> DeviceRegistry:
    """获取或创建全局设备注册表。"""
    global _global_registry
    if _global_registry is None:
        _global_registry = DeviceRegistry()
    return _global_registry
