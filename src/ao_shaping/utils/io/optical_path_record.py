"""标准光路记录 (``ao-optical-path-record/v1``) 的纯组装函数。

为什么需要它
------------
``libs/skills/ao-optical-path-calibration/SKILL.md`` §4 规定: 每次实验都要在
``data/runs/<run_id>/`` 下写一份 ``optical_path_record.json``, 字段固定。字段
固定是这份记录唯一的价值来源 —— 跨实验比较时, 只有两边的 ``config_id`` /
``fresnel_compensation_m_inv`` / ``two_pi_grayscale`` / ``exposure_used_ms`` /
``geometry_calib_mtime`` 都在, 才能把"算法效果"和"光路 / 设备漂移"分开。

因此这里的三条原则:

* **不省键**: schema 里的每个键都出现; 拿不到的值写 ``null``。
* **null 带原因**: 原因记在 ``missing`` (点分路径 → 人话)。原因**不**塞进
  数值字段 —— 那会把 ``float | None`` 变成 ``float | None | str``, 跨实验比较
  时每次都要先判类型。
* **不碰硬件**: 输入全是调用方已经采到的普通 dict, 所以这个模块可以在没有
  SLM / 相机的机器上被 import 和测试。

``missing`` 是 schema 之外唯一新增的键: 它承载"为什么这里是 null", 不影响
schema 内任何字段的类型。

用法::

    >>> record = build_optical_path_record(
    ...     run_id="20261116_093000_probe",
    ...     date="2026-11-16",
    ...     devices={"slm": {"slm_number": 1}, "camera": {"serial": "SN1"}},
    ...     optical_path={"probe_status": "ok"},
    ...     exposure_check={"peak_counts_at_used_exposure": 150.0},
    ...     occupancy={"acquired_at": "2026-11-16T09:30:00+00:00"},
    ...     git_commit="1a2b3c4",
    ... )
    >>> record["schema"]
    'ao-optical-path-record/v1'
    >>> record["optical_path"]["aperture"] is None
    True
    >>> "optical_path.aperture" in record["missing"]
    True
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA",
    "build_optical_path_record",
]

#: 记录格式标识 (SKILL.md §4)。跨实验比较靠字符串完全相等。
SCHEMA = "ao-optical-path-record/v1"

#: ``devices.slm`` 的字段 (顺序照 SKILL.md §4 的 JSON 例子)。
SLM_FIELDS: tuple[str, ...] = (
    "model",
    "drive_board_id",
    "option_board_id",
    "slm_number",
    "wavelength_nm",
    "two_pi_grayscale",
    "panel_res",
    "readsu_status_at_open",
    "open_mode",
    "power_cycled_this_session",
)

#: ``devices.camera`` 的字段。
CAMERA_FIELDS: tuple[str, ...] = (
    "model",
    "serial",
    "bit_depth",
    "full_scale",
    "exposure_range_ms",
    "exposure_used_ms",
    "gain",
    "averaging_frames",
    "discard_frames_after_exposure_change",
    "pixel_size_um",
)

#: ``optical_path`` 的字段。
OPTICAL_PATH_FIELDS: tuple[str, ...] = (
    "config_id",
    "aperture",
    "fresnel_compensation_m_inv",
    "geometry_calib_file",
    "geometry_calib_mtime",
    "probe_status",
    "probe_report",
)

#: ``exposure_check`` 的字段。
EXPOSURE_CHECK_FIELDS: tuple[str, ...] = (
    "peak_counts_at_used_exposure",
    "saturation_threshold_counts",
    "saturated",
    "settle_timeout",
)

#: ``occupancy`` 的字段。
OCCUPANCY_FIELDS: tuple[str, ...] = (
    "acquired_at",
    "released_at",
    "closed_normally",
)

#: 本台设备上未接入的器件 —— 探测从不使用 WFS / DM, 所以缺省值就是事实描述。
_SECTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "devices.slm": {},
    "devices.camera": {},
    "optical_path": {},
    "exposure_check": {},
    "occupancy": {},
    "devices": {"wfs": "not used", "dm": "not used"},
}


def _json_safe(value: Any) -> Any:
    """把调用方给的值整理成 JSON 能直接序列化的形状。

    tuple → list (JSON 没有 tuple), ``Path`` → str, numpy 标量 → Python 标量
    (``.item()``), 其余未知类型退化成 ``str``: 记录宁可少一点类型信息, 也不能
    因为一个没预料到的对象在写盘时炸掉。
    """
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return str(value)


def _section(
    prefix: str,
    fields: tuple[str, ...],
    source: Mapping[str, Any] | None,
    missing: dict[str, str],
    reasons: Mapping[str, str],
) -> dict[str, Any]:
    """填一段 (``prefix``) 的固定字段; 缺的写 ``null`` 并在 ``missing`` 里记原因。"""
    section_missing = None
    if source is None:
        section_missing = f"{prefix} 整段未提供"
    elif not isinstance(source, Mapping):
        section_missing = (
            f"{prefix} 不是映射 (收到 {type(source).__name__}), 按未提供处理"
        )
        source = None

    filled: dict[str, Any] = {}
    for field in fields:
        path = f"{prefix}.{field}"
        value = None if source is None else source.get(field)
        if value is None and prefix in _SECTION_DEFAULTS:
            value = _SECTION_DEFAULTS[prefix].get(field)
        if value is None:
            filled[field] = None
            missing[path] = (
                reasons.get(path) or section_missing or "调用方未提供该字段"
            )
        else:
            filled[field] = _json_safe(value)
    return filled


def build_optical_path_record(
    *,
    run_id: str | None = None,
    date: str | None = None,
    operator_note: str = "",
    devices: Mapping[str, Any] | None = None,
    optical_path: Mapping[str, Any] | None = None,
    exposure_check: Mapping[str, Any] | None = None,
    occupancy: Mapping[str, Any] | None = None,
    git_commit: str | None = None,
    probe_status: str | None = None,
    reasons: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """组装一份 ``ao-optical-path-record/v1`` 记录 (纯函数, 不碰硬件)。

    Args:
        run_id: ``<YYYYMMDD>_<HHMMSS>_<short_tag>``。
        date: ``<YYYY-MM-DD>``。
        operator_note: 人工备注; 缺省空串 (不是 ``null``)。
        devices: ``{"slm": {...}, "camera": {...}, "wfs": ..., "dm": ...}``。
            段内缺的键写 ``null``, 整段缺也一样。
        optical_path: ``config_id`` / ``aperture`` / ``fresnel_compensation_m_inv``
            / ``geometry_calib_file`` / ``geometry_calib_mtime`` /
            ``probe_status`` / ``probe_report``。
        exposure_check: ``peak_counts_at_used_exposure`` /
            ``saturation_threshold_counts`` / ``saturated`` / ``settle_timeout``。
        occupancy: ``acquired_at`` / ``released_at`` / ``closed_normally``。
        git_commit: 短 sha。
        probe_status: 探针判定; 给出时**覆盖** ``optical_path["probe_status"]``
            (判定是跑出来的, 不该让调用方在两处维护)。
        reasons: 点分路径 → 原因, 覆盖缺省原因文案 (例如"探针不控制光阑")。

    Returns:
        字段齐全的记录 dict; ``record["missing"]`` 是 ``{点分路径: 原因}``,
        只列出值为 ``null`` 的字段。
    """
    reason_map = dict(reasons or {})
    missing: dict[str, str] = {}

    def scalar(path: str, value: Any) -> Any:
        if value is None:
            missing[path] = reason_map.get(path) or "调用方未提供该字段"
            return None
        return _json_safe(value)

    devices_source = devices if isinstance(devices, Mapping) else None
    slm_source = (
        devices_source.get("slm") if devices_source is not None else None
    )
    camera_source = (
        devices_source.get("camera") if devices_source is not None else None
    )

    optical_path_source = dict(optical_path) if isinstance(optical_path, Mapping) else None
    if probe_status is not None:
        optical_path_source = dict(optical_path_source or {})
        optical_path_source["probe_status"] = probe_status

    optical_path_section = _section(
        "optical_path",
        OPTICAL_PATH_FIELDS,
        optical_path_source,
        missing,
        reason_map,
    )

    devices_section: dict[str, Any] = {
        "slm": _section("devices.slm", SLM_FIELDS, slm_source, missing, reason_map),
        "camera": _section(
            "devices.camera", CAMERA_FIELDS, camera_source, missing, reason_map
        ),
    }
    for key in ("wfs", "dm"):
        value = devices_source.get(key) if devices_source is not None else None
        if value is None:
            value = _SECTION_DEFAULTS["devices"][key]
        devices_section[key] = _json_safe(value)

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": scalar("run_id", run_id),
        "date": scalar("date", date),
        "operator_note": "" if operator_note is None else str(operator_note),
        "devices": devices_section,
        "optical_path": optical_path_section,
        "exposure_check": _section(
            "exposure_check",
            EXPOSURE_CHECK_FIELDS,
            exposure_check,
            missing,
            reason_map,
        ),
        "occupancy": _section(
            "occupancy", OCCUPANCY_FIELDS, occupancy, missing, reason_map
        ),
        "git_commit": scalar("git_commit", git_commit),
        "missing": missing,
    }
    return record
