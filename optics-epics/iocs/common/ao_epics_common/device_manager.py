"""设备访问门面:统一 IOC 内设备的打开/关闭/状态追踪。

各 IOC 驱动(SantecSLM200、DahengCamManager、NLight、ThorlabWFS 等)
均实现 open()/close()/is_connected();此门面提供:
- 启动时按序 open 全部设备
- 退出/信号时按序 close
- 连接状态汇总
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger("ao_epics")


class OpenCloseDevice(Protocol):
    def open(self) -> None: ...

    def close(self) -> None: ...

    def is_connected(self) -> bool: ...


class DeviceManager:
    """管理 IOC 内设备生命周期。"""

    def __init__(self) -> None:
        self._devices: dict[str, OpenCloseDevice] = {}

    def add(self, name: str, device: Any) -> None:
        """注册设备(不立即 open)。"""
        self._devices[name] = device

    def open_all(self) -> None:
        """按注册顺序打开全部设备。单个设备失败不阻断其余设备。"""
        for name, dev in self._devices.items():
            try:
                dev.open()
                logger.info("Device %s opened", name)
            except Exception as exc:  # noqa: BLE001
                logger.error("Device %s open failed: %s", name, exc)

    def close_all(self) -> None:
        """按逆序关闭全部设备。"""
        for name, dev in reversed(list(self._devices.items())):
            try:
                dev.close()
                logger.info("Device %s closed", name)
            except Exception as exc:  # noqa: BLE001
                logger.error("Device %s close failed: %s", name, exc)

    def get(self, name: str) -> Any:
        return self._devices[name]

    def connection_status(self) -> dict[str, bool]:
        return {name: dev.is_connected() for name, dev in self._devices.items()}

    def __enter__(self) -> "DeviceManager":
        self.open_all()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close_all()
