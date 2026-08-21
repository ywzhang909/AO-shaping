"""caproto 软 IOC 运行器。

提供:
- `run_ioc()`:加载 ioc.yaml,实例化 PVGroup 并启动 caproto CA 服务端。
- 每个具体 IOC 的 main 脚本:

    from ao_epics_common import run_ioc
    from slm_ioc import SLMIoc
    run_ioc("config/ioc.yaml", SLMIoc)

环境变量:
    EPICS_CA_PORT  - 覆盖 ioc.yaml 的 ca_port(写入 EPICS_CA_SERVER_PORT)
    EPICS_CAS_INTF_ADDR_LIST - 监听地址(默认全部接口,caproto 原生支持)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Type

from caproto.server import PVGroup, ioc_arg_parser, pvproperty, run

from .ioc_config import IocSpec, load_ioc_config

logger = logging.getLogger("ao_epics")


def build_pv_group(
    group_class: Type[PVGroup],
    spec: IocSpec,
) -> Type[PVGroup]:
    """根据 ioc.yaml 前缀构造 PVGroup 工厂(保留原类,仅在类上记录 spec)。"""
    prefix = spec.prefix.rstrip(":")

    class _PrefixedGroup(group_class):  # type: ignore[misc]
        """带前缀的 IOC PVGroup。"""

        __module__ = group_class.__module__

    _PrefixedGroup.ioc_spec = spec  # type: ignore[attr-defined]
    _PrefixedGroup.prefix = prefix  # type: ignore[attr-defined]
    return _PrefixedGroup


def run_ioc(
    ioc_yaml: str | Path,
    group_class: Type[PVGroup],
    argv: list[str] | None = None,
) -> None:
    """加载配置并启动 IOC(阻塞)。

    Args:
        ioc_yaml: config/ioc.yaml 路径(相对 IOC 根目录)。
        group_class: 具体 IOC 的 PVGroup 子类(内含 pvproperty 声明)。
        argv: 传给 ioc_arg_parser 的剩余参数(默认 sys.argv[1:]),
            供 serve.py 这类先消费位置参数再转发 CLI 的入口使用。
    """
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    spec = load_ioc_config(ioc_yaml)
    logger.info(
        "Loading IOC %s (prefix=%r, ca_port=%d, devices=%d)",
        spec.name,
        spec.prefix,
        spec.ca_port,
        len(spec.devices),
    )

    # caproto Context 从 EPICS_CA_SERVER_PORT 读取端口;允许环境变量覆盖
    ca_port = int(os.environ.get("EPICS_CA_PORT", spec.ca_port))
    os.environ["EPICS_CA_SERVER_PORT"] = str(ca_port)

    # caproto 1.3:args={'prefix','macros'} 给 PVGroup 构造器;
    #            kwargs={'module_name','log_pv_names','interfaces'} 给 run()
    args, kwargs = ioc_arg_parser(
        default_prefix=spec.prefix,
        desc=spec.description or f"{spec.name} soft IOC",
        argv=argv,
    )

    group_class_ready = build_pv_group(group_class, spec)

    # 设备注入:若 IOC 类实现了 create_device(spec),则用于构造硬件实例
    device = None
    if hasattr(group_class_ready, "create_device"):
        try:
            device = group_class_ready.create_device(spec)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - 硬件缺失时降级为离线模式
            logger.warning("create_device 失败(%s),以离线模式启动(仅注册 PV)", exc)
            device = None

    group = group_class_ready(**args, device=device)

    if hasattr(group, "startup"):
        group.startup()  # type: ignore[attr-defined]

    logger.info(
        "Starting %s on port %d ... (prefix=%s)",
        spec.name,
        ca_port,
        spec.prefix,
    )
    try:
        run(group.pvdb, **kwargs)
    finally:
        if hasattr(group, "shutdown"):
            group.shutdown()  # type: ignore[attr-defined]


__all__ = [
    "PVGroup",
    "Any",
    "IocSpec",
    "load_ioc_config",
    "run_ioc",
]
