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
    """Source Excel files used to generate the wiring map."""

    wiring_table: str
    device_mapping: str


@dataclass(frozen=True)
class Metadata:
    """Wiring map metadata."""

    description: str
    generated_at: str
    source_files: SourceFiles
    array_size: str
    total_channels: int


@dataclass(frozen=True)
class ChannelSchemaDoc:
    """Documentation for channel fields (from wiring_map.json schema.channel)."""

    needle_id: str
    physical_label: str
    mapping_row: str
    physical_position: str
    ip_suffix: str
    payload_position: str
    port: str


@dataclass(frozen=True)
class SchemaDoc:
    """Schema documentation section."""

    channel: ChannelSchemaDoc


@dataclass(frozen=True)
class ChannelEntry:
    """Single channel mapping entry from the wiring map.

    Represents one needle pin (277-330) and its mapping to:
    - Physical position in the 39×39 actuator array
    - Controller IP address (via ip_suffix)
    - Payload byte position within the controller's 50-channel frame
    - TCP port (10000 + ip_suffix)
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
        """Check if this entry has meaningful data (not all null)."""
        return self.physical_position is not None

    @property
    def ip_address(self) -> str | None:
        """Full IP address (assumes 192.168.0.x subnet)."""
        if self.ip_suffix is not None:
            return f"192.168.0.{self.ip_suffix}"
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChannelEntry:
        """Parse a ChannelEntry from a JSON dict."""
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
    """A group of channels (e.g., "一组", "二组")."""

    name: str
    channel_count: int
    channels: list[ChannelEntry] = field(default_factory=list)

    @classmethod
    def from_dict(cls, key: str, data: dict[str, Any]) -> Group:
        """Parse a Group from a JSON dict.

        Args:
            key: Group key from JSON (e.g., "group_1").
            data: Group data dict.
        """
        channels = [ChannelEntry.from_dict(ch) for ch in data.get("channels", [])]
        return cls(
            name=data.get("name", key),
            channel_count=data.get("channel_count", len(channels)),
            channels=channels,
        )


@dataclass(frozen=True)
class RangeInfo:
    """Min/max range for a numeric field."""

    min: int
    max: int


@dataclass(frozen=True)
class Summary:
    """Wiring map summary statistics."""

    unique_ip_suffixes: list[int]
    needle_id_range: RangeInfo
    physical_position_range: RangeInfo

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Summary:
        """Parse a Summary from a JSON dict."""
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
    """Complete wiring map for the 1300-channel DM cabinet.

    Parsed from ``libs/micro_drive1300/wiring_map.json``.
    Maps each needle pin (277-330) across multiple groups to:
    - Physical positions in a 39×39 actuator array
    - Controller IPs (via ip_suffix)
    - Payload byte positions (1-50) within each controller's frame
    """

    schema_version: str
    metadata: Metadata
    schema: SchemaDoc
    groups: dict[str, Group]
    summary: Summary

    @property
    def all_channels(self) -> list[ChannelEntry]:
        """Flattened list of all valid channel entries across all groups."""
        return [
            ch for group in self.groups.values() for ch in group.channels if ch.is_valid
        ]

    @property
    def unique_ips(self) -> list[str]:
        """Sorted list of unique controller IP addresses.

        Derives IPs from the actual channel data in the groups, falling back
        to the summary field if present. This ensures the wiring map works
        even when the optional ``summary`` section is missing from the JSON.
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
        """Parse a complete WiringMap from a JSON dict."""
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
        """Load a WiringMap from a JSON file.

        Args:
            path: Path to the wiring_map.json file.

        Returns:
            Parsed WiringMap, or None if file not found or invalid.
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
    """Runtime lookup view of a channel, built from WiringMap.

    Includes group context and pre-computed indices for fast lookup.
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
        """Full IP address (assumes 192.168.0.x subnet)."""
        if self.ip_suffix is not None:
            return f"192.168.0.{self.ip_suffix}"
        return None

    @classmethod
    def from_entry(
        cls, entry: ChannelEntry, group_name: str, group_key: str
    ) -> ChannelInfo:
        """Create a ChannelInfo from a ChannelEntry with group context."""
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
