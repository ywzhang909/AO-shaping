"""大恒相机在线检查的统一枚举漏斗 (零导入叶子)。

**为什么需要这一层**: 仓库里曾有 6 处各自复刻的 ``gx.DeviceManager()
.update_device_list()`` 枚举点, 且只有 ``transfer.hardware.prepare_camera_sdk``
会在 ``gxipy`` 不在 ``PYTHONPATH`` 时把随包 ``libs`` 注入 ``sys.path``; 其余
直接 ``import gxipy`` + try/except, SDK 不在路径上时一律被误判成「相机离线」。
本模块把两件事收口成一处:

1. **SDK 引导** (:func:`bootstrap_sdk`): 先直接 ``import gxipy``; 失败再把
   ``AO_SHAPING_SDK_PATH`` (或仓库 ``libs``) 注入 ``sys.path`` 重试一次。
2. **枚举** (:func:`enumerate_devices`): 只调官方 SDK 的
   ``update_device_list()``, **绝不 ping / TCP 探测固定 IP** —— 大恒在线与否
   只由子网枚举结果决定, 与 IP 无关。

对外区分三态, 供 GUI / CLI / 自检分别表达:

- ``sdk_ready=False`` —— SDK 不可导入 (没装 / 路径错), **不是**相机离线;
- ``sdk_ready=True, online=False`` —— SDK 在, 但子网里没有大恒相机 (离线 /
  网线未接 / 被同网段枚举不到);
- ``sdk_ready=True, online=True`` —— 枚举到至少一台, 返回原始设备信息列表。

本模块是零导入叶子: 模块作用域不 import ``gxipy``、不做磁盘 I/O
(``import ao_shaping.drivers.ccd.daheng.discovery`` 不触发 SDK 加载),
因此离线可测 (见 ``tests/ao_shaping/drivers/ccd/daheng/test_discovery.py``)。
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

#: ``gxipy`` 随包绑定的父目录 (``<repo>/libs``, 内含 ``gxipy/`` 包)。
#: 与 ``tools/slm/transfer/hardware.py::prepare_camera_sdk`` 同源: 本模块位于
#: ``src/ao_shaping/drivers/ccd/daheng/``, 上溯 5 层即仓库根。
_LIBS_DIR = Path(__file__).resolve().parents[5] / "libs"


@dataclass(frozen=True)
class DahengDiscovery:
    """一次大恒在线检查的完整结果 (纯数据, 界面/CLI 可直接消费)。"""

    sdk_ready: bool
    online: bool
    count: int
    devices: list[Any] = field(default_factory=list)
    error: str | None = None

    @property
    def summary(self) -> str:
        """一行中文结论 (与 :func:`summary_text` 同口径)。"""
        return summary_text(self)


def bootstrap_sdk() -> Any | None:
    """把随包 ``gxipy`` 暴露给解释器, 返回 SDK 模块; 不可用返回 ``None``。

    先直接 ``import gxipy`` (已装在 ``PYTHONPATH`` 的情况); 失败再尝试把
    ``AO_SHAPING_SDK_PATH`` 或仓库 ``libs`` 注入 ``sys.path`` 后重导一次。
    不抛异常 —— 调用方通过 :func:`enumerate_devices` 的 ``sdk_ready`` 区分
    「SDK 未装」与「相机离线」。
    """
    module, _ = _load_gxipy()
    if module is not None:
        return module
    if _inject_sdk_path():
        module, _ = _load_gxipy()
        return module
    return None


def _load_gxipy() -> tuple[Any | None, str | None]:
    """尝试导入 ``gxipy``, 返回 ``(模块, 错误摘要)``; 成功时错误为 ``None``。"""
    try:
        return importlib.import_module("gxipy"), None
    except (ImportError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _inject_sdk_path() -> bool:
    """把 ``gxipy`` 的父目录注入 ``sys.path``; 找不到合法绑定返回 ``False``。"""
    raw = os.environ.get("AO_SHAPING_SDK_PATH")
    libs = Path(raw) if raw else _LIBS_DIR
    if not (libs / "gxipy" / "__init__.py").is_file():
        return False
    if str(libs) not in sys.path:
        sys.path.insert(0, str(libs))
    return True


def enumerate_devices() -> DahengDiscovery:
    """用官方 SDK 子网枚举大恒相机, **不 ping、不按固定 IP 探测**。

    返回 :class:`DahengDiscovery`; 任何枚举异常 (GigE 驱动缺失、设备被别的
    进程独占、网卡异常) 都被归一成 ``online=False`` 并带 ``error`` 摘要 ——
    但**只在 SDK 真的可导入时**才归一, SDK 缺失单独记 ``sdk_ready=False``。
    """
    module, sdk_error = _load_gxipy()
    if module is None and _inject_sdk_path():
        module, sdk_error = _load_gxipy()
    if module is None:
        return DahengDiscovery(
            sdk_ready=False,
            online=False,
            count=0,
            error=f"gxipy 不可用 ({sdk_error})",
        )

    # policy: gxipy 的异常类型只能在 SDK 导入后才知道 (本模块是零导入叶子),
    # 枚举失败的后果对调用方就是「探不到相机」—— 归一成 online=False 并留日志,
    # 否则这句归一就是静默的。
    try:
        _count, devices = module.DeviceManager().update_device_list()
    except Exception as exc:  # policy: 枚举异常类型不可穷举 (SDK 内部错误), 归一为离线并留日志
        logger.warning(
            f"大恒相机枚举失败 ({type(exc).__name__}: {exc}); 按没有可连接相机处理"
        )
        return DahengDiscovery(
            sdk_ready=True,
            online=False,
            count=0,
            error=f"枚举失败 ({type(exc).__name__}: {exc})",
        )

    materialised = list(devices) if devices else []
    return DahengDiscovery(
        sdk_ready=True,
        online=bool(materialised),
        count=len(materialised),
        devices=materialised,
    )


def device_manager() -> Any | None:
    """返回一个可用的 ``gx.DeviceManager`` 实例 (用于需要 open/close 的场景)。

    仅在 SDK 可导入时返回; 否则 ``None``。**不**在这里枚举 —— 枚举统一走
    :func:`enumerate_devices`, 本函数只负责让调用方拿到设备管理器句柄。
    """
    module = bootstrap_sdk()
    if module is None:
        return None
    try:
        return module.DeviceManager()
    except Exception as exc:  # policy: 驱动初始化失败与 SDK 缺失对调用方同义 —— 都拿不到句柄
        logger.warning(f"大恒 DeviceManager 初始化失败 ({type(exc).__name__}: {exc})")
        return None


def summary_text(disc: DahengDiscovery | None = None) -> str:
    """一行中文结论, 用于界面 caption / CLI 日志。"""
    result = disc if disc is not None else enumerate_devices()
    if not result.sdk_ready:
        return f"大恒 SDK 不可用: {result.error or 'gxipy 未导入'}"
    if not result.online:
        detail = f" ({result.error})" if result.error else ""
        return f"大恒相机离线: 子网未枚举到设备{detail}"
    return f"大恒相机在线: 枚举到 {result.count} 台"
