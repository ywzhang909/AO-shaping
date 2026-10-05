"""R50Power 接线表 (wiring map) 的类型化 JSON schema.

接线表把针脚 (277-330) 映射到 39×39 阵列物理位置、控制器 IP 后缀与
载荷字节位置。本模块只做 **解析 + 索引**, 不含任何硬件 I/O；
:class:`~ao_shaping.drivers.dm.micro.driver.MicroDM` 在 ``__init__`` 时
读一次并建好三个 O(1) 索引。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class SourceFiles:
    """用于生成接线表的源 Excel 文件。"""

    wiring_table: str
    device_mapping: str


@dataclass(frozen=True)
class Metadata:
    """接线表元数据。"""

    description: str
    generated_at: str
    source_files: SourceFiles
    array_size: str
    total_channels: int


@dataclass(frozen=True)
class ChannelSchemaDoc:
    """通道字段的文档说明 (来自 wiring_map.json 的 schema.channel)。"""

    needle_id: str
    physical_label: str
    mapping_row: str
    physical_position: str
    ip_suffix: str
    payload_position: str
    port: str


@dataclass(frozen=True)
class SchemaDoc:
    """schema 文档小节。"""

    channel: ChannelSchemaDoc


@dataclass(frozen=True)
class ChannelEntry:
    """接线表中的单条通道映射记录。

    表示一根针脚 (277-330) 及其到以下各项的映射:
    - 39×39 致动器阵列中的物理位置
    - 控制器 IP 地址 (经由 ip_suffix)
    - 该控制器 50 通道帧内的载荷字节位置
    - TCP 端口 (10000 + ip_suffix)
    """

    needle_id: int | None
    physical_label: str | None
    mapping_row: int | None
    physical_position: int | None
    ip_suffix: int | None
    payload_position: int | None
    port: int | None = None

    @property
    def is_valid(self) -> bool:
        """检查该记录是否含有有效数据 (而非全为 null)。"""
        return self.physical_position is not None

    @property
    def ip_address(self) -> str | None:
        """完整 IP 地址 (假定为 192.168.0.x 子网)。"""
        if self.ip_suffix is not None:
            return f"192.168.0.{self.ip_suffix}"
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChannelEntry:
        """从 JSON 字典解析出 ChannelEntry。"""
        return cls(
            needle_id=data.get("needle_id"),
            physical_label=data.get("physical_label"),
            mapping_row=data.get("mapping_row"),
            physical_position=data.get("physical_position"),
            ip_suffix=data.get("ip_suffix"),
            payload_position=data.get("payload_position"),
            port=data.get("port"),
        )


@dataclass(frozen=True)
class Group:
    """一组通道 (例如 "一组"、"二组")。"""

    name: str
    channel_count: int
    channels: list[ChannelEntry] = field(default_factory=list)

    @classmethod
    def from_dict(cls, key: str, data: dict[str, Any]) -> Group:
        """从 JSON 字典解析出 Group。

        Args:
            key: JSON 中的组键 (例如 "group_1")。
            data: 组数据字典。
        """
        channels = [ChannelEntry.from_dict(ch) for ch in data.get("channels", [])]
        return cls(
            name=data.get("name", key),
            channel_count=data.get("channel_count", len(channels)),
            channels=channels,
        )


@dataclass(frozen=True)
class RangeInfo:
    """某个数值字段的最小/最大范围。"""

    min: int
    max: int


@dataclass(frozen=True)
class Summary:
    """接线表的汇总统计。"""

    unique_ip_suffixes: list[int]
    needle_id_range: RangeInfo
    physical_position_range: RangeInfo

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Summary:
        """从 JSON 字典解析出 Summary。"""
        nid = data.get("needle_id_range", {})
        ppr = data.get("physical_position_range", {})
        return cls(
            unique_ip_suffixes=data.get("unique_ip_suffixes", []),
            needle_id_range=RangeInfo(min=nid.get("min", 0), max=nid.get("max", 0)),
            physical_position_range=RangeInfo(
                min=ppr.get("min", 0), max=ppr.get("max", 0)
            ),
        )


@dataclass(frozen=True)
class WiringMap:
    """1300 通道 DM 柜的完整接线表。

    解析自 ``libs/micro_drive1300/wiring_map.json``。
    把跨多个组的每根针脚 (277-330) 映射到:
    - 39×39 致动器阵列中的物理位置
    - 控制器 IP (经由 ip_suffix)
    - 各控制器帧内的载荷字节位置 (1-50)
    """

    schema_version: str
    metadata: Metadata
    schema: SchemaDoc
    groups: dict[str, Group]
    summary: Summary

    @property
    def all_channels(self) -> list[ChannelEntry]:
        """跨所有组展开的全部有效通道记录。"""
        return [
            ch for group in self.groups.values() for ch in group.channels if ch.is_valid
        ]

    @property
    def unique_ips(self) -> list[str]:
        """去重后排序的控制器 IP 地址列表。

        IP 由各组中的实际通道数据推导而来, 若 JSON 中存在 summary 字段
        则在无实际数据时回退到它。这确保即使 JSON 缺失可选的 ``summary``
        小节, 接线表依然可用。
        """
        suffixes = {
            ch.ip_suffix
            for group in self.groups.values()
            for ch in group.channels
            if ch.is_valid and ch.ip_suffix is not None
        }
        if not suffixes:
            suffixes = set(self.summary.unique_ip_suffixes)
        return [f"192.168.0.{s}" for s in sorted(suffixes)]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WiringMap:
        """从 JSON 字典解析出完整的 WiringMap。"""
        meta_data = data.get("metadata", {})
        sf_data = meta_data.get("source_files", {})
        meta = Metadata(
            description=meta_data.get("description", ""),
            generated_at=meta_data.get("generated_at", ""),
            source_files=SourceFiles(
                wiring_table=sf_data.get("wiring_table", ""),
                device_mapping=sf_data.get("device_mapping", ""),
            ),
            array_size=meta_data.get("array_size", ""),
            total_channels=meta_data.get("total_channels", 0),
        )

        ch_schema = data.get("schema", {}).get("channel", {})
        schema = SchemaDoc(
            channel=ChannelSchemaDoc(
                needle_id=ch_schema.get("needle_id", ""),
                physical_label=ch_schema.get("physical_label", ""),
                mapping_row=ch_schema.get("mapping_row", ""),
                physical_position=ch_schema.get("physical_position", ""),
                ip_suffix=ch_schema.get("ip_suffix", ""),
                payload_position=ch_schema.get("payload_position", ""),
                port=ch_schema.get("port", ""),
            ),
        )

        groups = {
            key: Group.from_dict(key, val)
            for key, val in data.get("groups", {}).items()
        }

        summary = Summary.from_dict(data.get("summary", {}))

        return cls(
            schema_version=data.get("$schema", ""),
            metadata=meta,
            schema=schema,
            groups=groups,
            summary=summary,
        )

    @classmethod
    def from_file(cls, path: Path) -> WiringMap | None:
        """从 JSON 文件加载 WiringMap。

        Args:
            path: wiring_map.json 文件的路径。

        Returns:
            解析出的 WiringMap, 文件不存在或无效时为 None。
        """
        if not path.exists():
            logger.warning(f"Wiring map not found: {path}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            result = cls.from_dict(data)
            logger.info(
                f"Loaded wiring map: {result.metadata.description} "
                f"({len(result.groups)} groups, {result.metadata.total_channels} channels)"
            )
            return result
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning(f"Failed to load wiring map: {exc}")
            return None


@dataclass(frozen=True)
class ChannelInfo:
    """由 WiringMap 构建的通道运行时查找视图。

    包含组上下文与预先算好的索引, 便于快速查找。
    """

    needle_id: int | None
    physical_label: str | None
    mapping_row: int | None
    physical_position: int | None
    ip_suffix: int | None
    payload_position: int | None
    port: int | None = None
    group_name: str | None = None
    group_key: str | None = None

    @property
    def ip_address(self) -> str | None:
        """完整 IP 地址 (假定为 192.168.0.x 子网)。"""
        if self.ip_suffix is not None:
            return f"192.168.0.{self.ip_suffix}"
        return None

    @classmethod
    def from_entry(
        cls, entry: ChannelEntry, group_name: str, group_key: str
    ) -> ChannelInfo:
        """由带组上下文的 ChannelEntry 创建 ChannelInfo。"""
        return cls(
            needle_id=entry.needle_id,
            physical_label=entry.physical_label,
            mapping_row=entry.mapping_row,
            physical_position=entry.physical_position,
            ip_suffix=entry.ip_suffix,
            payload_position=entry.payload_position,
            port=entry.port,
            group_name=group_name,
            group_key=group_key,
        )


__all__ = [
    "ChannelEntry",
    "ChannelInfo",
    "ChannelSchemaDoc",
    "Group",
    "Metadata",
    "RangeInfo",
    "SchemaDoc",
    "SourceFiles",
    "Summary",
    "WiringMap",
]
