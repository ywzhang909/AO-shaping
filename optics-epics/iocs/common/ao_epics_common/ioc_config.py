"""ioc.yaml 声明式配置加载与校验。

ioc.yaml 格式(ibek 风格):

    ioc:
      name: ioc-slm            # IOC 名称
      description: ...         # 可选
      prefix: "SLM-01:"        # PV 前缀
      ca_port: 5065            # EPICS_CA_PORT(独立端口,避免冲突)
      devices:                 # 设备描述列表(传给驱动构造)
        - name: slm
          type: santec_slm200
          params:
            wavelength: 1064
      pvs:                     # PV 声明(供文档/OPI 生成,运行时由驱动类注册)
        - {name: Wavelength, type: int, desc: ...}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class IocConfigError(ValueError):
    """ioc.yaml 配置无效。"""


@dataclass
class DeviceSpec:
    """单个设备描述。"""

    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class IocSpec:
    """解析后的 IOC 声明。"""

    name: str
    prefix: str
    ca_port: int
    description: str = ""
    devices: list[DeviceSpec] = field(default_factory=list)
    pvs: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def pv_names(self) -> list[str]:
        """返回全限定 PV 名(带前缀)。"""
        return [f"{self.prefix}{pv['name']}" for pv in self.pvs]


def _parse_ioc_section(section: dict[str, Any]) -> dict[str, Any]:
    required = {"name", "prefix", "ca_port"}
    missing = required - set(section)
    if missing:
        raise IocConfigError(f"ioc.yaml 缺少字段: {sorted(missing)}")
    ca_port = section["ca_port"]
    if not (1 <= ca_port <= 65535):
        raise IocConfigError(f"ca_port 越界: {ca_port}")
    return {
        "name": str(section["name"]),
        "prefix": str(section["prefix"]),
        "ca_port": int(ca_port),
        "description": str(section.get("description", "")),
    }


def _parse_devices(raw_devices: Any) -> list[DeviceSpec]:
    if raw_devices is None:
        return []
    if not isinstance(raw_devices, list):
        raise IocConfigError("devices 必须是列表")
    devices: list[DeviceSpec] = []
    for item in raw_devices:
        if not isinstance(item, dict) or "name" not in item or "type" not in item:
            raise IocConfigError(f"设备描述非法(需含 name/type): {item}")
        devices.append(
            DeviceSpec(
                name=str(item["name"]),
                type=str(item["type"]),
                params=dict(item.get("params", {})),
            )
        )
    return devices


def _parse_pvs(raw_pvs: Any) -> list[dict[str, Any]]:
    if raw_pvs is None:
        return []
    if not isinstance(raw_pvs, list):
        raise IocConfigError("pvs 必须是列表")
    pvs: list[dict[str, Any]] = []
    for item in raw_pvs:
        if not isinstance(item, dict) or "name" not in item:
            raise IocConfigError(f"PV 声明非法(需含 name): {item}")
        pvs.append(dict(item))
    return pvs


def load_ioc_config(path: str | Path) -> IocSpec:
    """从文件加载并校验 ioc.yaml。"""
    p = Path(path)
    if not p.exists():
        raise IocConfigError(f"ioc.yaml 不存在: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise IocConfigError(f"ioc.yaml 解析失败: {exc}") from exc
    if not isinstance(raw, dict) or "ioc" not in raw:
        raise IocConfigError("ioc.yaml 顶层必须含 ioc 节")
    return validate(raw)


def validate(raw: dict[str, Any]) -> IocSpec:
    """从原始 dict 构造并校验 IocSpec。"""
    section_raw = raw.get("ioc", {})
    section = _parse_ioc_section(section_raw)
    devices = _parse_devices(section_raw.get("devices"))
    pvs = _parse_pvs(section_raw.get("pvs"))
    # 检查 PV 名不重复(前缀内)
    names = [pv["name"] for pv in pvs]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise IocConfigError(f"PV 名重复: {sorted(dupes)}")
    return IocSpec(
        name=section["name"],
        prefix=section["prefix"],
        ca_port=section["ca_port"],
        description=section["description"],
        devices=devices,
        pvs=pvs,
        raw=raw,
    )
