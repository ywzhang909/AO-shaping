"""optics-epics IOC 公共框架。

设计原则:
- 每个 IOC 是一个 caproto PVGroup 子类,由 config/ioc.yaml 声明式描述。
- ioc.yaml 采用 ibek 风格:描述 IOC 元信息、PV 前缀、CA 端口、设备参数。
- 运行器启动 caproto CA 服务端,注册全部 PV。

架构:
    ioc.yaml ──> ioc_config.load() ──> IocSpec(dataclass)
                    │
                    ▼
    ioc_main.py ──> 构造设备驱动实例 ──> 构造 PVGroup 子类 ──> caproto 运行
"""
from __future__ import annotations

from .ao_epics_common import (  # noqa: F401
    DeviceSpec,
    IocSpec,
    load_ioc_config,
    run_ioc,
    validate,
)

__all__ = ["DeviceSpec", "IocSpec", "load_ioc_config", "validate", "run_ioc"]
