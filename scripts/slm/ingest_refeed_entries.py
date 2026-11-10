"""Issue #230 online-refeed dataset ingest: versioned entries + append-only store.

This is a **standalone adapter** for the mechanism described in issue #230
("在线训练集扩展工程化：版本化条目模式 + 追加式存储 + 块状态机 + 版本化门 +
不变量守护", 方案 v1.0).  It is deliberately *not* integrated into the main
optimisation loop yet: nothing in ``src/`` imports it, and the production block
loop still owns the capture/judge path.  The module therefore owns only the
**data layer + control-flow skeleton + decision layer** that issue #230 §3/§4/§5
specifies, and it is honest about the boundary:

* the store (append-only log, idempotent snapshot, isolated validation set)
  is implemented here;
* the block state machine is implemented as an explicit transition table with
  the states issue #230 §3 names (``BLOCK_START`` -> ``DELIVER`` -> ``CAPTURE``
  -> ``JUDGE`` -> ``REFLOW`` -> ``REFIT`` -> ``GATE`` -> ``BLOCK_END``, plus
  ``BASELINE_REBUILD``);
* the switch gate is implemented as the versioned check list of issue #230 §4
  with **AND** semantics;
* the guards are implemented as the five invariants ``INV-1`` .. ``INV-5`` of
  issue #230 §5.

Everything here is **offline**: no SLM, camera or laser is touched, no
``ao_shaping.drivers`` module is imported, and no file is written unless
``--execute`` is given.  The default mode is plan-only.

Source of truth for every name below (schema fields, state names, transition
edges, check identifiers, invariant identifiers, config keys, event
identifiers) is the issue #230 body itself; the module comments quote the issue
sections so a reviewer can diff the implementation against the task sheet.

Issue #230 file layout (§2), reproduced verbatim:

===================================== ================================== ==========
path                                  content                             write mode
===================================== ================================== ==========
``session_dataset.jsonl``             append-only entry log, one record/line append only
``session_dataset_state.json``        derived snapshot                      idempotent rebuild
``validation_holdout.json``           independent validation set declaration write once
``events.jsonl``                      event log (reflow/refit/gate/guard/  append only
                                      baseline rebuild)
``schema/session_dataset.schema.json``the schema (single source of truth)    on version change
===================================== ================================== ==========

Rules from §2 that this module honours:

(1) entry writes are O(1) appends -- refeeding never rewrites a stored line;
(2) the snapshot is rebuilt idempotently from the log -- two rebuilds of the
    same log produce byte-identical snapshots;
(3) corrections do not rewrite history -- a correction is a new entry with
    ``supersedes`` pointing at the entry it replaces, and the reader follows
    the supersedes chain;
(4) validation-set isolation is physical: the refeed path can only ever write
    ``session_dataset.jsonl``, so a refeed frame cannot become validation data.

Usage::

    # plan only (default): prints the state-machine plan and exits 0
    python scripts/slm/ingest_refeed_entries.py --dataset-dir data/debug/online_refeed_demo

    # self test: synthetic ABBA blocks, inside a temporary directory only
    python scripts/slm/ingest_refeed_entries.py --self-test

    # actually write: real entries read from a JSONL file (never fabricated)
    python scripts/slm/ingest_refeed_entries.py --dataset-dir <dir> \
        --entries-jsonl entries.jsonl --execute

The ``--execute`` path **never** invents data: it refuses to write anything
without ``--entries-jsonl``, and the synthetic records of the self test are
confined to the self test (they cannot reach a session directory).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEVICE_STATEMENT = (
    "离线脚本：未打开 SLM / 相机 / 激光，未 import 任何 ao_shaping.drivers 模块；"
    "全部输入来自命令行参数与内存构造的假条目。"
)

# ---------------------------------------------------------------------------
# §1 数据模型：版本化条目记录 -- v1 字段注册表（单一事实源）
# ---------------------------------------------------------------------------
#
# The table below is issue #230 §1 "v1 字段注册表" verbatim:
#
# | 字段 | 类型 | 必需 | 默认 | 写入方 | 说明 |
# | schema_version      | int    | 是 | 无   | 写入器 | 模式版本，只增不减 |
# | entry_id            | string | 是 | 无   | 写入器 | 唯一标识：s 会话-b 块-f 帧 |
# | session_id          | string | 是 | 无   | 写入器 | 会话标识 |
# | block_id            | int    | 是 | 无   | 写入器 | 块序号（1 基） |
# | frame_role          | enum   | 是 | 无   | 写入器 | anchor_probe 或 augment |
# | abba_pos            | enum   | 是 | 无   | 采集   | A1、B1、B2、A2 |
# | candidate_id        | string | 是 | 无   | 主循环 | 候选相位标识 |
# | phase_ref / phase_sha256   | string | 是 | 无 | 采集 | 下发相位文件地址 + 哈希 |
# | frame_ref / frame_sha256   | string | 是 | 无 | 采集 | 帧文件地址 + 哈希 |
# | exposure_ms_set / readback | float  | 是 | 无 | 采集 | 设置曝光 + 读回曝光（ms） |
# | captured_at         | float  | 是 | 无   | 采集 | 采集时间（epoch 秒） |
# | model_version       | string | 是 | 无   | 主循环 | 给该块提供方向的模型版本 |
# | verdict             | object | 是 | 空对象 | 判决 | 判决摘要（accepted 与 metrics_ref） |
# | supersedes          | string | 否 | null | 写入器 | 被取代条目的 entry_id |
# | ext                 | object | 是 | 空对象 | 任意 | 版本透明透传容器（opaque） |
#
# Rows that name two physical fields ("phase_ref / phase_sha256") are stored as
# two registry entries each, because the registry is the single source of truth
# for readers *and* writers.

#: The schema version this module writes.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FieldSpec:
    """One registry row of issue #230 §1.

    Attributes:
        name: The JSON key.
        kind: ``"int"``, ``"float"``, ``"str"``, ``"enum"`` or ``"obj"``.
        required: Whether the field must be present.
        default: The value used when an optional field is absent.
        writer: The issue's "写入方" column.
        note: The issue's "说明" column.
        enum: Allowed values for ``kind == "enum"``.
    """

    name: str
    kind: str
    required: bool
    default: Any
    writer: str
    note: str
    enum: tuple[str, ...] = ()


#: The v1 field registry -- issue #230 §1, table order preserved.
FIELD_REGISTRY: tuple[FieldSpec, ...] = (
    FieldSpec("schema_version", "int", True, None, "写入器",
              "模式版本，只增不减"),
    FieldSpec("entry_id", "str", True, None, "写入器",
              "唯一标识：s 会话-b 块-f 帧"),
    FieldSpec("session_id", "str", True, None, "写入器", "会话标识"),
    FieldSpec("block_id", "int", True, None, "写入器", "块序号（1 基）"),
    FieldSpec("frame_role", "enum", True, None, "写入器",
              "anchor_probe 或 augment", ("anchor_probe", "augment")),
    FieldSpec("abba_pos", "enum", True, None, "采集",
              "A1、B1、B2、A2", ("A1", "B1", "B2", "A2")),
    FieldSpec("candidate_id", "str", True, None, "主循环", "候选相位标识"),
    FieldSpec("phase_ref", "str", True, None, "采集", "下发相位文件地址"),
    FieldSpec("phase_sha256", "str", True, None, "采集", "下发相位文件哈希"),
    FieldSpec("frame_ref", "str", True, None, "采集", "帧文件地址"),
    FieldSpec("frame_sha256", "str", True, None, "采集", "帧文件哈希"),
    FieldSpec("exposure_ms_set", "float", True, None, "采集", "设置曝光（ms）"),
    FieldSpec("exposure_ms_readback", "float", True, None, "采集",
              "读回曝光（ms）"),
    FieldSpec("captured_at", "float", True, None, "采集",
              "采集时间（epoch 秒）"),
    FieldSpec("model_version", "str", True, None, "主循环",
              "给该块提供方向的模型版本"),
    FieldSpec("verdict", "obj", True, {}, "判决",
              "判决摘要（accepted 与 metrics_ref）"),
    FieldSpec("supersedes", "str", False, None, "写入器",
              "被取代条目的 entry_id（追加式更正用）"),
    FieldSpec("ext", "obj", True, {}, "任意",
              "版本透明透传容器（opaque）"),
)

#: Registry lookup by field name.
FIELDS: dict[str, FieldSpec] = {spec.name: spec for spec in FIELD_REGISTRY}

#: JSON-Schema type per registry kind.
JSON_TYPES: dict[str, str] = {
    "int": "integer",
    "float": "number",
    "str": "string",
    "enum": "string",
    "obj": "object",
}


def build_schema() -> dict[str, Any]:
    """Build the JSON schema of issue #230 §2 from the field registry.

    The registry is the single source of truth, so the schema's ``properties``
    are *generated* from it: every registry row becomes a property with its
    JSON type, and enum rows carry their ``enum`` list.  Without the
    ``properties`` block a standard validator would reject every legal record,
    because ``additionalProperties: false`` forbids all of them.

    Args:
        None.

    Returns:
        The schema mapping.
    """
    properties: dict[str, Any] = {}
    for spec in FIELD_REGISTRY:
        prop: dict[str, Any] = {"type": JSON_TYPES[spec.kind]}
        if spec.kind == "enum":
            prop["enum"] = list(spec.enum)
        if spec.kind == "obj":
            prop["additionalProperties"] = True
        if not spec.required and spec.default is None:
            # An optional field whose default is ``null`` accepts ``null`` as
            # its literal default (issue #230 §1, e.g. ``supersedes``).
            prop["type"] = [JSON_TYPES[spec.kind], "null"]
        properties[spec.name] = prop
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_NAME,
        "title": "AO-Shaping online refeed session dataset entry",
        "x-schema-version": SCHEMA_VERSION,
        "x-field-registry": [
            {
                "name": spec.name,
                "type": spec.kind,
                "required": spec.required,
                "default": spec.default,
                "writer": spec.writer,
                "note": spec.note,
                **({"enum": list(spec.enum)} if spec.enum else {}),
            }
            for spec in FIELD_REGISTRY
        ],
        "type": "object",
        "properties": properties,
        "required": [spec.name for spec in FIELD_REGISTRY if spec.required],
        "additionalProperties": False,
    }


#: Issue #230 §3: the ABBA positions, in capture order.
ABBA_ORDER: tuple[str, ...] = ("A1", "B1", "B2", "A2")

#: Issue #230 §1: the frame roles.
FRAME_ROLES: tuple[str, ...] = ("anchor_probe", "augment")


class RegistryError(ValueError):
    """A record violates the v1 field registry."""


class StoreError(RuntimeError):
    """The store refuses a write because the data contradicts the issue rules.

    Every refusal is a *hard* error: the adapter never repairs, truncates or
    rewrites an existing file to make a write succeed.
    """


class StoreConflict(StoreError):
    """An existing declaration or entry contradicts the requested write."""


def validate_entry(entry: dict[str, Any]) -> list[str]:
    """Validate one record against the v1 registry.

    Args:
        entry: The record to check.

    Returns:
        A list of human-readable problems; empty when the record is valid.
    """
    problems: list[str] = []
    if not isinstance(entry, dict):
        return [f"entry must be a mapping, got {type(entry).__name__}"]
    for spec in FIELD_REGISTRY:
        if spec.name not in entry:
            if spec.required:
                problems.append(f"missing required field '{spec.name}'")
            continue
        value = entry[spec.name]
        if value is None and not spec.required and spec.default is None:
            # Issue #230 §1: an optional field may carry the literal default
            # (e.g. ``supersedes`` defaults to ``null``).
            continue
        if spec.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                problems.append(
                    f"field '{spec.name}' must be int, got {type(value).__name__}"
                )
        elif spec.kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(
                    f"field '{spec.name}' must be float, got {type(value).__name__}"
                )
        elif spec.kind == "str":
            if not isinstance(value, str):
                problems.append(
                    f"field '{spec.name}' must be string, got {type(value).__name__}"
                )
        elif spec.kind == "enum":
            if value not in spec.enum:
                problems.append(
                    f"field '{spec.name}' must be one of {list(spec.enum)}, got {value!r}"
                )
        elif spec.kind == "obj":
            if not isinstance(value, dict):
                problems.append(
                    f"field '{spec.name}' must be object, got {type(value).__name__}"
                )
    return problems


def check_enum_extension(previous: tuple[str, ...], current: tuple[str, ...]) -> list[str]:
    """Issue #230 §1 field-extension protocol (3): enums only grow.

    Args:
        previous: The enum values of the older schema version.
        current: The enum values of the newer schema version.

    Returns:
        A list of problems; empty when the extension is legal.
    """
    dropped = [value for value in previous if value not in current]
    renamed = [value for value in current if value not in previous]
    problems: list[str] = []
    if dropped:
        problems.append(f"enum values removed: {dropped}")
    if renamed:
        problems.append(f"enum values renamed (must be additive): {renamed}")
    return problems


def check_registry_extension(
    previous: tuple[FieldSpec, ...], current: tuple[FieldSpec, ...]
) -> list[str]:
    """Issue #230 §1 field-extension protocol (2): new fields append at the tail.

    Args:
        previous: The registry of the older schema version.
        current: The registry of the newer schema version.

    Returns:
        A list of problems; empty when the extension is legal.
    """
    problems: list[str] = []
    prev_names = [spec.name for spec in previous]
    cur_names = [spec.name for spec in current]
    if cur_names[: len(prev_names)] != prev_names:
        problems.append(
            "registry extension must append at the tail; the leading fields changed"
        )
    prev_by_name = {spec.name: spec for spec in previous}
    for spec in current[len(previous):]:
        if spec.required and spec.default is None:
            problems.append(
                f"new field '{spec.name}' must be optional or carry a default"
            )
    for name, prev in prev_by_name.items():
        cur = FIELDS.get(name)
        if cur is None:
            continue
        if cur.kind != prev.kind:
            problems.append(
                f"field '{name}' changed type {prev.kind} -> {cur.kind}; "
                "use a new field name instead"
            )
        if cur.enum != prev.enum:
            problems.extend(check_enum_extension(prev.enum, cur.enum))
    return problems


def normalise_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Fill registry defaults and move unknown fields into ``ext``.

    Implements issue #230 §1 field-extension protocol (4): a lower-version
    reader must not lose a field it does not know -- unknown fields move into
    the opaque ``ext`` container verbatim.

    Args:
        entry: The raw record.

    Returns:
        A new record with defaults applied and unknown fields in ``ext``.
    """
    known = {spec.name for spec in FIELD_REGISTRY}
    out = {name: value for name, value in entry.items() if name in known}
    extra = {name: value for name, value in entry.items() if name not in known}
    for spec in FIELD_REGISTRY:
        if spec.name in out:
            continue
        if spec.required:
            if spec.default is None:
                raise RegistryError(f"missing required field '{spec.name}'")
            out[spec.name] = spec.default
        else:
            out[spec.name] = spec.default
    ext = dict(out.get("ext") or {})
    for name, value in extra.items():
        ext.setdefault(name, value)
    out["ext"] = ext
    return out


def build_entry(
    *,
    session_id: str,
    block_id: int,
    frame_role: str,
    abba_pos: str,
    candidate_id: str,
    model_version: str,
    phase_ref: str,
    phase_sha256: str,
    frame_ref: str,
    frame_sha256: str,
    exposure_ms_set: float,
    exposure_ms_readback: float,
    captured_at: float,
    verdict: dict[str, Any] | None = None,
    entry_id: str | None = None,
    supersedes: str | None = None,
    ext: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one registry-valid record.

    The default ``entry_id`` follows issue #230 §1: "唯一标识：s 会话-b 块-f 帧".

    Args:
        session_id: The session identifier.
        block_id: The 1-based block number.
        frame_role: ``"anchor_probe"`` or ``"augment"``.
        abba_pos: One of ``A1``, ``B1``, ``B2``, ``A2``.
        candidate_id: The candidate phase identifier.
        model_version: The model version that gave this block its direction.
        phase_ref: The phase file reference.
        phase_sha256: The phase file hash.
        frame_ref: The frame file reference.
        frame_sha256: The frame file hash.
        exposure_ms_set: The requested exposure in ms.
        exposure_ms_readback: The read-back exposure in ms.
        captured_at: Capture time in epoch seconds.
        verdict: The verdict summary; empty object when unknown.
        entry_id: An explicit identifier; generated when omitted.
        supersedes: The entry this one corrects.
        ext: The opaque passthrough container.

    Returns:
        The record, with defaults applied.

    Raises:
        RegistryError: When the arguments violate the registry.
    """
    if entry_id is None:
        entry_id = f"s{session_id}-b{block_id}-f{abba_pos}"
    entry: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "entry_id": entry_id,
        "session_id": session_id,
        "block_id": block_id,
        "frame_role": frame_role,
        "abba_pos": abba_pos,
        "candidate_id": candidate_id,
        "phase_ref": phase_ref,
        "phase_sha256": phase_sha256,
        "frame_ref": frame_ref,
        "frame_sha256": frame_sha256,
        "exposure_ms_set": exposure_ms_set,
        "exposure_ms_readback": exposure_ms_readback,
        "captured_at": captured_at,
        "model_version": model_version,
        "verdict": dict(verdict or {}),
        "supersedes": supersedes,
        "ext": dict(ext or {}),
    }
    problems = validate_entry(entry)
    if problems:
        raise RegistryError("; ".join(problems))
    return entry


# ---------------------------------------------------------------------------
# §5 不变量与守护 -- INV-1 .. INV-5
# ---------------------------------------------------------------------------

#: Issue #230 §5 invariant identifiers, verbatim.
INVARIANTS: tuple[str, ...] = ("INV-1", "INV-2", "INV-3", "INV-4", "INV-5")


@dataclass(frozen=True)
class GuardViolation:
    """One guard abort: issue #230 §5 "违反即中止 + 诊断".

    Attributes:
        invariant: The violated identifier, e.g. ``"INV-1"``.
        detail: What went wrong.
        block_id: The block number in which the violation happened.
        summary: A short digest of the inputs.
    """

    invariant: str
    detail: str
    block_id: int | None = None
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form used by ``events.jsonl``."""
        payload = {
            "event": "guard_violation",
            "invariant": self.invariant,
            "detail": self.detail,
            "block_id": self.block_id,
            "input_summary": self.summary,
        }
        payload["event_id"] = _sha256_text(_json_line(payload))[:32]
        return payload


def _digest(entries: list[dict[str, Any]]) -> str:
    """A short, stable digest of the inputs a guard ran on."""
    if not entries:
        return "none"
    ids = sorted(str(entry.get("entry_id")) for entry in entries)
    return f"n={len(ids)} ids={','.join(ids[:8])}"


def check_inv1_judge_isolation(
    judge_inputs: list[dict[str, Any]], current_block_id: int
) -> list[GuardViolation]:
    """INV-1 -- 判决隔离.

    Issue #230 §5: "当前块判决只用本块 ABBA 四帧；回灌帧（块号小于当前）永不
    进入当前判决输入".  Check point: ``JUDGE``.

    Args:
        judge_inputs: The records fed into the current verdict.
        current_block_id: The block being judged.

    Returns:
        The violations found; empty when the invariant holds.
    """
    violations: list[GuardViolation] = []
    for entry in judge_inputs:
        block_id = entry.get("block_id")
        if isinstance(block_id, int) and block_id < current_block_id:
            violations.append(
                GuardViolation(
                    "INV-1",
                    f"refeed frame from block {block_id} entered the verdict of "
                    f"block {current_block_id}",
                    current_block_id,
                    _digest(judge_inputs),
                )
            )
    return violations


def check_inv2_validation_isolation(
    fit_inputs: list[dict[str, Any]], holdout_phases: set[str]
) -> list[GuardViolation]:
    """INV-2 -- 验证独立.

    Issue #230 §5: "验证集相位不参与任何拟合；回灌帧永不升格为验证集".  Check
    points: ``REFIT``, ``GATE``.

    Args:
        fit_inputs: The records fed into the fit.
        holdout_phases: The phase references declared as validation data.

    Returns:
        The violations found; empty when the invariant holds.
    """
    violations: list[GuardViolation] = []
    for entry in fit_inputs:
        phase_ref = str(entry.get("phase_ref", ""))
        if phase_ref in holdout_phases:
            violations.append(
                GuardViolation(
                    "INV-2",
                    f"validation phase '{phase_ref}' entered the fit inputs",
                    entry.get("block_id"),
                    _digest(fit_inputs),
                )
            )
        role = entry.get("frame_role")
        if role == "validation":
            violations.append(
                GuardViolation(
                    "INV-2",
                    "a record carries frame_role='validation'; validation frames "
                    "never enter the refeed log",
                    entry.get("block_id"),
                    _digest(fit_inputs),
                )
            )
    return violations


def check_inv3_block_freeze(
    block_start: dict[str, Any], block_end: dict[str, Any]
) -> list[GuardViolation]:
    """INV-3 -- 块内冻结.

    Issue #230 §5: "块内模型权重与参数完全不变（120 步中同一模型给方向、同一
    模型被验证）".  Check points: ``BLOCK_START``, ``DELIVER``.

    Args:
        block_start: The freeze snapshot recorded at ``BLOCK_START``.
        block_end: The freeze snapshot observed at the end of the block.

    Returns:
        The violations found; empty when the invariant holds.
    """
    violations: list[GuardViolation] = []
    if block_start.get("model_version") != block_end.get("model_version"):
        violations.append(
            GuardViolation(
                "INV-3",
                f"model changed inside the block: "
                f"{block_start.get('model_version')} -> "
                f"{block_end.get('model_version')}",
                block_start.get("block_id"),
                f"start={block_start.get('model_version')} "
                f"end={block_end.get('model_version')}",
            )
        )
    for key in ("weights_sha256", "params_sha256"):
        if block_start.get(key) != block_end.get(key):
            violations.append(
                GuardViolation(
                    "INV-3",
                    f"{key} changed inside the block",
                    block_start.get("block_id"),
                    f"{block_start.get(key)} -> {block_end.get(key)}",
                )
            )
    return violations


def check_inv4_append_only(
    stored: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[GuardViolation]:
    """INV-4 -- 追加式.

    Issue #230 §5: "条目落盘后不可变；更正经新条目 + supersedes".  Check point:
    ``REFLOW``.

    Args:
        stored: The records already in the log.
        incoming: The records about to be appended.

    Returns:
        The violations found; empty when the invariant holds.
    """
    violations: list[GuardViolation] = []
    stored_by_id = {str(entry.get("entry_id")): entry for entry in stored}
    for entry in incoming:
        entry_id = str(entry.get("entry_id"))
        supersedes = entry.get("supersedes")
        if entry_id in stored_by_id:
            existing = stored_by_id[entry_id]
            if existing != entry:
                violations.append(
                    GuardViolation(
                        "INV-4",
                        f"entry '{entry_id}' already exists with different content; "
                        "a correction must be a new entry carrying 'supersedes'",
                        entry.get("block_id"),
                        entry_id,
                    )
                )
            continue
        if supersedes is not None and str(supersedes) not in stored_by_id:
            violations.append(
                GuardViolation(
                    "INV-4",
                    f"entry '{entry_id}' supersedes unknown entry '{supersedes}'",
                    entry.get("block_id"),
                    entry_id,
                )
            )
    return violations


def check_inv5_schema_monotonic(
    stored: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[GuardViolation]:
    """INV-5 -- 模式单调.

    Issue #230 §5: "schema_version 只增；未知字段经 ext 不透明透传，不丢弃".
    Check points: every read and write.

    Args:
        stored: The records already in the log.
        incoming: The records about to be appended.

    Returns:
        The violations found; empty when the invariant holds.
    """
    violations: list[GuardViolation] = []
    for entry in list(stored) + list(incoming):
        version = entry.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool):
            violations.append(
                GuardViolation(
                    "INV-5",
                    f"schema_version must be int, got {version!r}",
                    entry.get("block_id"),
                    str(entry.get("entry_id")),
                )
            )
            continue
        if version > SCHEMA_VERSION:
            violations.append(
                GuardViolation(
                    "INV-5",
                    f"schema_version {version} is newer than this reader's "
                    f"{SCHEMA_VERSION}; unknown fields must survive in 'ext'",
                    entry.get("block_id"),
                    str(entry.get("entry_id")),
                )
            )
    return violations


# ---------------------------------------------------------------------------
# §4 决策层：模型切换门（版本化检查清单，AND 语义）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateCheck:
    """One row of the issue #230 §4 check list.

    Attributes:
        check_id: The check identifier, e.g. ``"gate-nrmse-001"``.
        metric: What the check measures.
        threshold: The threshold description.
        semantics: The meaning of the check.
        since: The gate version that introduced it.
    """

    check_id: str
    metric: str
    threshold: str
    semantics: str
    since: int


#: Issue #230 §4 check list (v1), verbatim.
GATE_CHECKLIST: tuple[GateCheck, ...] = (
    GateCheck(
        "gate-nrmse-001",
        "独立验证集 NRMSE（新模型对旧模型）",
        "新值不超过旧值 x 1.10",
        "重拟合不得劣化绝对预测",
        1,
    ),
    GateCheck(
        "gate-dirguard-002",
        "下一块 A1 与 B 对的方向守卫相关",
        "不低于切换前值",
        "局部方向可用性不得劣化",
        1,
    ),
)

#: Issue #230 §4: the NRMSE degradation factor of ``gate-nrmse-001``.
GATE_NRMSE_FACTOR = 1.10


@dataclass(frozen=True)
class GateOutcome:
    """The result of one gate evaluation.

    Attributes:
        gate_version: The version of the check list.
        checks: One record per check, in checklist order.
        passed: Whether every check passed (AND semantics).
        decision: ``"switch"`` or ``"keep"``.
        drift_flag: Whether the drift flag must be set.
    """

    gate_version: int
    checks: tuple[dict[str, Any], ...]
    passed: bool
    decision: str
    drift_flag: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form used by ``gate_event``."""
        return {
            "gate_version": self.gate_version,
            "checks": list(self.checks),
            "passed": self.passed,
            "decision": self.decision,
            "drift_flag": self.drift_flag,
        }


def evaluate_gate(
    *,
    old_nrmse: float,
    new_nrmse: float,
    old_dirguard: float,
    new_dirguard: float,
    gate_version: int = 1,
) -> GateOutcome:
    """Evaluate the versioned check list of issue #230 §4.

    The semantics are **AND**: every check must pass for the model to be
    switched; any failing check vetoes the switch, keeps the old model and
    sets the drift flag.

    Args:
        old_nrmse: The validation NRMSE of the incumbent model.
        new_nrmse: The validation NRMSE of the candidate model.
        old_dirguard: The direction-guard correlation before the switch.
        new_dirguard: The direction-guard correlation of the candidate.
        gate_version: The version of the check list in use.

    Returns:
        The gate outcome, with one record per check.
    """
    checks: list[dict[str, Any]] = []

    nrmse_cap = old_nrmse * GATE_NRMSE_FACTOR
    nrmse_ok = bool(new_nrmse <= nrmse_cap)
    checks.append(
        {
            "check_id": "gate-nrmse-001",
            "metric": "validation_nrmse",
            "value": float(new_nrmse),
            "reference": float(old_nrmse),
            "threshold": float(nrmse_cap),
            "threshold_spec": "新值不超过旧值 x 1.10",
            "passed": nrmse_ok,
        }
    )

    dirguard_ok = bool(new_dirguard >= old_dirguard)
    checks.append(
        {
            "check_id": "gate-dirguard-002",
            "metric": "direction_guard_correlation",
            "value": float(new_dirguard),
            "reference": float(old_dirguard),
            "threshold": float(old_dirguard),
            "threshold_spec": "不低于切换前值",
            "passed": dirguard_ok,
        }
    )

    passed = all(bool(check["passed"]) for check in checks)
    return GateOutcome(
        gate_version=gate_version,
        checks=tuple(checks),
        passed=passed,
        decision="switch" if passed else "keep",
        drift_flag=not passed,
    )


# ---------------------------------------------------------------------------
# §6 配置面（版本化，模式/配置/门三版本独立演进）
# ---------------------------------------------------------------------------

#: Issue #230 §6 config keys, defaults and valid ranges.
CONFIG_DEFAULTS: dict[str, Any] = {
    "config_version": 1,
    "k_refit": 3,
    "refit_loss": "l1_image",
    "trust_region_rad": 0.5,
    "gate_nrmse_cap": 0.10,
    "gate_dirguard_cap": 0.00,
    "max_consecutive_gate_fail": 2,
    "include_rejected": True,
    "anchor_set_ref": "",
    "dataset_dir": "",
}

#: Issue #230 §6: the inclusive valid range of ``k_refit``.
K_REFIT_RANGE = (1, 5)


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate the versioned config face of issue #230 §6.

    Args:
        config: The configuration mapping.

    Returns:
        A list of problems; empty when the configuration is valid.
    """
    problems: list[str] = []
    for key, default in CONFIG_DEFAULTS.items():
        if key not in config:
            problems.append(f"config key '{key}' is missing")
    k_refit = config.get("k_refit")
    if not isinstance(k_refit, int) or isinstance(k_refit, bool):
        problems.append(f"config key 'k_refit' must be int, got {k_refit!r}")
    elif not K_REFIT_RANGE[0] <= k_refit <= K_REFIT_RANGE[1]:
        problems.append(
            f"config key 'k_refit' must be in {list(K_REFIT_RANGE)}, got {k_refit}"
        )
    if config.get("refit_loss") != "l1_image":
        problems.append(
            f"config key 'refit_loss' must be 'l1_image', got {config.get('refit_loss')!r}"
        )
    for key in ("trust_region_rad", "gate_nrmse_cap", "gate_dirguard_cap"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"config key '{key}' must be float, got {value!r}")
    max_fail = config.get("max_consecutive_gate_fail")
    if not isinstance(max_fail, int) or isinstance(max_fail, bool) or max_fail < 1:
        problems.append(
            f"config key 'max_consecutive_gate_fail' must be a positive int, "
            f"got {max_fail!r}"
        )
    if not isinstance(config.get("include_rejected"), bool):
        problems.append("config key 'include_rejected' must be bool")
    return problems


def build_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a configuration with the issue #230 §6 defaults applied.

    Args:
        overrides: Keys to override.

    Returns:
        The configuration mapping.
    """
    config = dict(CONFIG_DEFAULTS)
    config.update(overrides or {})
    return config


# ---------------------------------------------------------------------------
# §3 控制流：块状态机
# ---------------------------------------------------------------------------

#: Issue #230 §3 state names, in transfer order.
BLOCK_STATES: tuple[str, ...] = (
    "BLOCK_START",
    "DELIVER",
    "CAPTURE",
    "JUDGE",
    "REFLOW",
    "REFIT",
    "GATE",
    "BLOCK_END",
    "BASELINE_REBUILD",
)

#: Issue #230 §3 transfer table: (from, to, condition, event).
TRANSITIONS: tuple[tuple[str, str, str, str], ...] = (
    ("BLOCK_START", "DELIVER", "冻结快照已记录", "freeze_snapshot"),
    ("DELIVER", "CAPTURE", "相位加载确认", "phase_loaded"),
    ("CAPTURE", "JUDGE", "四帧加清单齐备", "capture_done"),
    ("JUDGE", "REFLOW", "判决完成（接受或拒绝）", "verdict"),
    ("REFLOW", "REFIT", "块号 mod K = 0", "reflow_event"),
    ("REFLOW", "BLOCK_END", "块号 mod K 非 0", "reflow_event"),
    ("REFIT", "GATE", "重拟合完成", "refit_event"),
    ("GATE", "BLOCK_END", "全过（切换）或任一不过（保留 + 漂移标记）", "gate_event"),
    ("GATE", "BASELINE_REBUILD", "连续 2 次门失败", "gate_event + drift_flag"),
    ("BASELINE_REBUILD", "BLOCK_START", "基线重建完成", "baseline_rebuild"),
)


def transitions_from(state: str) -> tuple[tuple[str, str, str, str], ...]:
    """The transfer-table rows leaving ``state``.

    Args:
        state: The source state name.

    Returns:
        The matching transition rows.
    """
    return tuple(row for row in TRANSITIONS if row[0] == state)


@dataclass
class DatasetState:
    """The derived dataset state of issue #230 §1.

    "数据集状态由日志**派生**，不落为独立事实源" -- the fields below are
    recomputed from the log on every rebuild.

    Attributes:
        n_entries: Number of records in the log.
        n_anchor_probe: Records with ``frame_role == "anchor_probe"``.
        n_augment: Records with ``frame_role == "augment"``.
        block_counts: Number of records per block id.
        high_water_block: The largest block id that has been fed back.
        drift_flag: Whether the drift flag is set.
        switch_count: How often the model was switched.
        model_lineage: The model-version hot-start chain.
    """

    n_entries: int = 0
    n_anchor_probe: int = 0
    n_augment: int = 0
    block_counts: dict[str, int] = field(default_factory=dict)
    high_water_block: int = 0
    drift_flag: bool = False
    switch_count: int = 0
    model_lineage: list[str] = field(default_factory=list)


def derive_state(entries: list[dict[str, Any]]) -> DatasetState:
    """Derive the dataset state from the log (issue #230 §1).

    Args:
        entries: The records of the log.

    Returns:
        The derived state.
    """
    state = DatasetState(n_entries=len(entries))
    lineage: list[str] = []
    for entry in entries:
        role = entry.get("frame_role")
        if role == "anchor_probe":
            state.n_anchor_probe += 1
        elif role == "augment":
            state.n_augment += 1
        block_id = entry.get("block_id")
        if isinstance(block_id, int):
            key = str(block_id)
            state.block_counts[key] = state.block_counts.get(key, 0) + 1
            state.high_water_block = max(state.high_water_block, block_id)
        model_version = entry.get("model_version")
        if isinstance(model_version, str) and (
            not lineage or lineage[-1] != model_version
        ):
            lineage.append(model_version)
    state.model_lineage = lineage
    return state


def snapshot_bytes(entries: list[dict[str, Any]]) -> bytes:
    """Serialise the snapshot deterministically (issue #230 §2 rule 2).

    The bytes must be identical for two rebuilds of the same log, so the
    encoding is sorted, compact and newline-terminated.

    Args:
        entries: The records of the log.

    Returns:
        The snapshot bytes.
    """
    payload = {
        "schema": "session_dataset_state/1",
        "state": _state_to_dict(derive_state(entries)),
    }
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _state_to_dict(state: DatasetState) -> dict[str, Any]:
    """Convert a :class:`DatasetState` into a JSON-ready mapping."""
    return {
        "n_entries": state.n_entries,
        "n_anchor_probe": state.n_anchor_probe,
        "n_augment": state.n_augment,
        "block_counts": dict(sorted(state.block_counts.items())),
        "high_water_block": state.high_water_block,
        "drift_flag": state.drift_flag,
        "switch_count": state.switch_count,
        "model_lineage": list(state.model_lineage),
    }


def _json_line(payload: dict[str, Any]) -> str:
    """Serialise one JSONL line deterministically."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _sha256_text(text: str) -> str:
    """Hash a text with SHA-256 (used for the freeze snapshots)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# §2 存储结构
# ---------------------------------------------------------------------------

#: Issue #230 §2 file names.
LOG_NAME = "session_dataset.jsonl"
STATE_NAME = "session_dataset_state.json"
HOLDOUT_NAME = "validation_holdout.json"
EVENTS_NAME = "events.jsonl"
SCHEMA_DIR = "schema"
SCHEMA_NAME = "session_dataset.schema.json"


@dataclass
class Store:
    """The dataset store of issue #230 §2.

    The store is **append-only**: ``session_dataset.jsonl`` and ``events.jsonl``
    are only ever extended, never rewritten.  ``write_log`` and ``write_events``
    therefore *append* the records that are not on disk yet, and they refuse to
    write when the file on disk contradicts the in-memory state (a truncated or
    edited line).  The snapshot ``session_dataset_state.json`` is the only file
    that is rewritten, because issue #230 §2 rule (2) makes it an idempotent
    rebuild of the log.

    Attributes:
        root: The session directory.
        entries: The records currently in the log (in memory).
        events: The events currently in the event log (in memory).
        log_prefix_bytes: The bytes of the log that were already on disk when
            the store was loaded (used by the self test to prove that the
            existing prefix is never touched).
        events_prefix_bytes: The same for the event log.
    """

    root: Path
    entries: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    log_prefix_bytes: bytes = b""
    events_prefix_bytes: bytes = b""

    @property
    def log_path(self) -> Path:
        """Path of the append-only entry log."""
        return self.root / LOG_NAME

    @property
    def state_path(self) -> Path:
        """Path of the derived snapshot."""
        return self.root / STATE_NAME

    @property
    def holdout_path(self) -> Path:
        """Path of the independent validation-set declaration."""
        return self.root / HOLDOUT_NAME

    @property
    def events_path(self) -> Path:
        """Path of the append-only event log."""
        return self.root / EVENTS_NAME

    @property
    def schema_path(self) -> Path:
        """Path of the schema file (the single source of truth)."""
        return self.root / SCHEMA_DIR / SCHEMA_NAME

    def load(self) -> None:
        """Read the log and the event log from disk (if they exist).

        The load is a *verification* step, not a repair step: the bytes that are
        on disk are the truth, and the in-memory state is rebuilt from them.
        The prefix bytes are remembered so a later append can prove that the
        existing lines were not touched.

        Raises:
            StoreError: When a stored line is not valid JSON, when the same
                ``entry_id`` appears twice with different content, or when the
                stored records do not match the in-memory records.
        """
        log_bytes, events_bytes = self._verify_disk()
        self.entries = read_jsonl_bytes(log_bytes, self.log_path)
        self.events = read_jsonl_bytes(events_bytes, self.events_path)
        self.log_prefix_bytes = log_bytes
        self.events_prefix_bytes = events_bytes

    def _verify_disk(self) -> tuple[bytes, bytes]:
        """Verify committed prefixes and any complete append-only suffix.

        A suffix beyond the committed lengths is accepted only if it consists
        of complete canonical JSONL records with no conflicting IDs. This
        recovers an interrupted append before snapshot replacement. The next
        successful write commits that suffix too. A nonempty legacy store with
        no integrity metadata is refused rather than silently trusted.

        This detects accidental edits, not adversarial changes to both state
        and logs. There is no inter-process writer lock.
        """
        data = tuple(
            path.read_bytes() if path.is_file() else b""
            for path in (self.log_path, self.events_path)
        )
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_bytes())
                committed = state["committed_logs"]
                for path, raw in zip((self.log_path, self.events_path), data, strict=True):
                    anchor = committed[path.name]
                    length = anchor["byte_length"]
                    digest = anchor["sha256"]
                    if type(length) is not int or length < 0 or not isinstance(digest, str):
                        raise ValueError("invalid committed prefix metadata")
                    if len(raw) < length or hashlib.sha256(raw[:length]).hexdigest() != digest:
                        raise StoreError(f"{path.name}: committed prefix truncated or modified")
                    _count_lines(raw[:length])
            except (ValueError, KeyError, TypeError, UnicodeError) as error:
                raise StoreError(f"invalid integrity snapshot: {error}") from error
        elif any(data):
            raise StoreError(
                "nonempty store has no committed integrity snapshot; refusing to "
                "trust unverified logs"
            )
        for path, raw, label in zip(
            (self.log_path, self.events_path), data, ("entry", "event"), strict=True
        ):
            _count_lines(raw)
            records = read_jsonl_bytes(raw, path)
            self._check_unique_ids(records, label)
            self._check_canonical_lines(raw, path, label)
        return data[0], data[1]

    @staticmethod
    def _check_unique_ids(records: list[dict[str, Any]], label: str) -> None:
        """Refuse a log that repeats an id with different content.

        Args:
            records: The records read from disk.
            label: ``"entry"`` or ``"event"``, used in the error message.

        Raises:
            StoreError: When an id repeats with different content.
        """
        seen: dict[str, dict[str, Any]] = {}
        for record in records:
            key = str(record.get("entry_id") or record.get("event_id") or "")
            if not key:
                continue
            if key in seen and seen[key] != record:
                raise StoreError(
                    f"{label} log contains id '{key}' twice with different "
                    "content; the log must not be edited in place"
                )
            seen[key] = record

    @staticmethod
    def _check_canonical_lines(
        data: bytes, path: Path, label: str
    ) -> None:
        """Refuse a log whose lines are not in the canonical serialisation.

        The writer appends ``json.dumps(payload, sort_keys=True)``; a line
        that differs from that canonical form was edited in place (or written
        by another tool) and would break the byte-prefix guarantee of the
        append.

        Args:
            data: The bytes of the file.
            path: The path the bytes came from.
            label: ``"entry"`` or ``"event"``, used in the error message.

        Raises:
            StoreError: When a line is not in the canonical form.
        """
        for number, line in enumerate(
            data.decode("utf-8").splitlines(), start=1
        ):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return  # already reported by read_jsonl_bytes
            if not isinstance(payload, dict):
                return
            if _json_line(payload) != text:
                raise StoreError(
                    f"{path.name} line {number} is not in the canonical "
                    f"{label} serialisation; the log must not be edited in place"
                )

    def existing_ids(self) -> set[str]:
        """The entry ids already stored (the idempotency key)."""
        return {str(entry.get("entry_id")) for entry in self.entries}

    def append_entries(self, new_entries: list[dict[str, Any]]) -> int:
        """Append records to the log, ignoring duplicates (idempotent).

        Issue #230 §2 rule (1): the write is a pure append; duplicates are
        counted and dropped so a re-run of the same block is a no-op.

        Args:
            new_entries: The records to append.

        Returns:
            The number of records actually appended.

        Raises:
            StoreConflict: When an id is already stored with *different*
                content.  A correction must be a new entry carrying
                ``supersedes`` (issue #230 §2 rule 3), never a rewrite.
        """
        known = self.existing_ids()
        stored_by_id = {
            str(entry.get("entry_id")): entry
            for entry in self.entries
            if entry.get("entry_id") is not None
        }
        appended = 0
        for entry in new_entries:
            entry_id = str(entry.get("entry_id"))
            if entry_id in known:
                existing = stored_by_id.get(entry_id)
                if existing is not None and existing != entry:
                    raise StoreConflict(
                        f"entry '{entry_id}' already exists with different "
                        "content; append a correction with 'supersedes' instead"
                    )
                continue
            known.add(entry_id)
            self.entries.append(entry)
            appended += 1
        return appended

    def write_log(self) -> None:
        """Append the not-yet-stored records to the entry log.

        The file is opened in append mode: the bytes that are already on disk
        are never rewritten, so the log keeps its history byte for byte.

        Raises:
            StoreError: When the file on disk does not start with the bytes
                that were read at load time (the log was truncated or edited
                in the meantime).
        """
        self._verify_disk()
        if not self.state_path.exists():
            self.write_snapshot()  # Commit empty prefixes before the first append.
        self._append_lines(self.log_path, self.entries, self.log_prefix_bytes)
        self.log_prefix_bytes = (
            self.log_path.read_bytes() if self.log_path.exists() else b""
        )
        self.write_snapshot()

    def write_snapshot(self) -> bytes:
        """Commit the state and both committed prefixes atomically.

        A temporary file is replaced atomically. The stored prefix is the byte
        prefix that is on disk at this moment, including a complete suffix
        recovered from an interrupted previous append. An adversary changing
        both state and logs is out of scope.
        """
        payload = self._snapshot_payload()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            prefix=f"{self.state_path.name}.", dir=self.state_path.parent
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            os.replace(name, self.state_path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return payload

    def _snapshot_payload(self) -> bytes:
        """Build snapshot bytes with committed prefix metadata."""
        log_bytes, events_bytes = self._verify_disk()
        state = json.loads(snapshot_bytes(read_jsonl_bytes(log_bytes, self.log_path)))
        state["committed_logs"] = {
            LOG_NAME: _committed_anchor(self.log_path, log_bytes),
            EVENTS_NAME: _committed_anchor(self.events_path, events_bytes),
        }
        return (json.dumps(state, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )

    def write_events(self) -> None:
        """Append the not-yet-stored events to the event log.

        Raises:
            StoreError: When the file on disk does not start with the bytes
                that were read at load time.
        """
        self._verify_disk()
        if not self.state_path.exists():
            self.write_snapshot()
        self._append_lines(self.events_path, self.events, self.events_prefix_bytes)
        self.events_prefix_bytes = (
            self.events_path.read_bytes() if self.events_path.exists() else b""
        )
        self.write_snapshot()

    def _append_lines(
        self, path: Path, records: list[dict[str, Any]], prefix: bytes
    ) -> int:
        """Append the records that are not on disk yet to a JSONL file.

        Args:
            path: The file to append to.
            records: The full in-memory record list.
            prefix: The bytes that were read from the file at load time.

        Returns:
            The number of lines appended.

        Raises:
            StoreError: When the file on disk does not start with ``prefix``.
        """
        current = path.read_bytes() if path.is_file() else b""
        if not current.startswith(prefix):
            raise StoreError(
                f"{path.name} was truncated or edited after it was read "
                f"({len(current)} bytes on disk, {len(prefix)} bytes read); "
                "refusing to append onto an unknown prefix"
            )
        already = _count_lines(prefix)
        pending = records[already:]
        if not pending:
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(f"{_json_line(record)}\n" for record in pending)
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(payload)
        return len(pending)

    def write_holdout(
        self, declaration: dict[str, Any], *, explicit: bool = True
    ) -> str:
        """Create the independent validation-set declaration (once).

        Issue #230 §2 rule (4) and the table row "会话开始时写一次": the
        validation set is declared **once** per session and is never rewritten
        or emptied afterwards.  Overwriting it would silently destroy the
        independent validation set, so an existing declaration that differs
        from an explicitly requested one is a hard error.

        Args:
            declaration: The declaration payload.
            explicit: Whether the caller explicitly asked for this
                declaration.  An explicit declaration that differs from the
                stored one is a conflict; an implicit one (the session-start
                default) keeps whatever is already there.

        Returns:
            ``"created"`` when the file was written, ``"kept"`` when an
            identical (or not explicitly requested) declaration was already
            there.

        Raises:
            StoreConflict: When an explicitly requested declaration differs
                from the stored one.
        """
        payload = (
            json.dumps(declaration, sort_keys=True, ensure_ascii=False, indent=2)
            + "\n"
        )
        if self.holdout_path.is_file():
            current = self.holdout_path.read_text(encoding="utf-8")
            if current == payload:
                return "kept"
            if not explicit:
                return "kept"
            raise StoreConflict(
                "validation_holdout.json already exists with a different "
                "declaration; the validation set is written once per session "
                "and is never rewritten or cleared"
            )
        self.holdout_path.parent.mkdir(parents=True, exist_ok=True)
        self.holdout_path.write_text(payload, encoding="utf-8")
        return "created"

    def write_schema(self) -> None:
        """Write the schema file derived from the registry."""
        schema = build_schema()
        self.schema_path.parent.mkdir(parents=True, exist_ok=True)
        self.schema_path.write_text(
            json.dumps(schema, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def append_event(self, event: dict[str, Any]) -> bool:
        """Append one event to the event log (in memory), idempotently.

        The ``event_id`` is the idempotency key: an event that is already
        stored is not appended again, so re-running a block does not grow the
        event log.

        Args:
            event: The event payload.

        Returns:
            ``True`` when the event was appended, ``False`` when it was a
            duplicate.
        """
        event_id = str(event.get("event_id", ""))
        if event_id and event_id in self.existing_event_ids():
            return False
        self.events.append(event)
        return True

    def existing_event_ids(self) -> set[str]:
        """The event ids already stored (the idempotency key)."""
        return {
            str(event.get("event_id"))
            for event in self.events
            if event.get("event_id") is not None
        }


def _committed_anchor(path: Path, prefix: bytes) -> dict[str, Any]:
    """Return the committed byte length and SHA-256 for a JSONL file prefix."""
    return {
        "byte_length": len(prefix),
        "sha256": hashlib.sha256(prefix).hexdigest(),
        "path": path.name,
    }


def _count_lines(prefix: bytes) -> int:
    """Count the JSONL records in a byte prefix.

    Args:
        prefix: The bytes read from a JSONL file.

    Returns:
        The number of non-empty lines.
    """
    if not prefix:
        return 0
    text = prefix.decode("utf-8")
    if not text.endswith("\n"):
        # A file that does not end with a newline has an incomplete last line.
        raise StoreError(
            "the stored log does not end with a newline; the last line is "
            "truncated and must be repaired by a human before appending"
        )
    return sum(1 for line in text.splitlines() if line.strip())


def read_jsonl_bytes(data: bytes, path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL byte string into a list of mappings.

    Args:
        data: The bytes of the file.
        path: The path the bytes came from (used in the error message).

    Returns:
        The records.

    Raises:
        StoreError: When a line is not valid JSON.
    """
    records: list[dict[str, Any]] = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise StoreError(
                f"{path.name} line {number} is not valid JSON: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise StoreError(
                f"{path.name} line {number} is not a JSON object"
            )
        records.append(payload)
    return records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of mappings.

    Args:
        path: The file to read.

    Returns:
        The records; an empty list when the file does not exist.
    """
    if not path.is_file():
        return []
    return read_jsonl_bytes(path.read_bytes(), path)


def read_holdout_phases(path: Path) -> set[str]:
    """Read the declared validation phase references.

    Args:
        path: The validation-set declaration file.

    Returns:
        The declared phase references.
    """
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    phases = payload.get("phase_refs") or []
    return {str(item) for item in phases}


# ---------------------------------------------------------------------------
# The block state machine
# ---------------------------------------------------------------------------


@dataclass
class BlockRun:
    """One pass through the block state machine.

    Attributes:
        block_id: The 1-based block number.
        entries: The records produced by the block (ABBA four frames).
        state: The current state.
        history: The states visited, in order.
        events: The events emitted.
        gate: The gate outcome, when the gate ran.
        refit: Whether a refit happened.
        switched: Whether the model was switched.
    """

    block_id: int
    entries: list[dict[str, Any]]
    state: str = "BLOCK_START"
    history: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    gate: GateOutcome | None = None
    refit: bool = False
    switched: bool = False


def _emit(run: BlockRun, event: dict[str, Any]) -> None:
    """Record an event on the run, with a deterministic identity.

    The ``event_id`` is derived from the event's content, so re-running the
    same block produces the *same* event id: the event log stays idempotent
    and a repeated run appends no duplicate line.

    Args:
        run: The block run the event belongs to.
        event: The event payload.
    """
    payload = dict(event)
    payload.setdefault("block_id", run.block_id)
    identity = {
        key: value
        for key, value in payload.items()
        if key not in {"event_id", "seq"}
    }
    payload["event_id"] = _sha256_text(_json_line(identity))[:32]
    run.events.append(payload)


def run_block(
    *,
    store: Store,
    block_id: int,
    entries: list[dict[str, Any]],
    config: dict[str, Any],
    verdict: dict[str, Any],
    model_version: str,
    old_nrmse: float,
    new_nrmse: float,
    old_dirguard: float,
    new_dirguard: float,
    holdout_phases: set[str],
    weights_sha256: str | None = None,
    params_sha256: str | None = None,
) -> BlockRun:
    """Drive one block through the issue #230 §3 state machine.

    The block-internal behaviour (``DELIVER``/``CAPTURE``/``JUDGE``) is *not*
    re-implemented here: this adapter is handed the block's four ABBA records
    and the verdict, because the production loop still owns the capture and
    judge path.  What this function owns is the block-boundary control flow:
    the freeze snapshot, ``REFLOW``, the ``REFIT`` trigger, the gate and the
    baseline rebuild.

    Args:
        store: The dataset store.
        block_id: The 1-based block number.
        entries: The block's ABBA records.
        config: The versioned configuration.
        verdict: The verdict summary of the block.
        model_version: The model version in force during the block.
        old_nrmse: The incumbent model's validation NRMSE.
        new_nrmse: The refitted model's validation NRMSE.
        old_dirguard: The direction-guard correlation before the refit.
        new_dirguard: The direction-guard correlation of the refit.
        holdout_phases: The declared validation phase references.
        weights_sha256: The frozen weight hash of the model.
        params_sha256: The frozen parameter hash of the model.

    Returns:
        The finished run.

    Raises:
        GuardAbort: When an invariant is violated; the caller must abort.
    """
    run = BlockRun(block_id=block_id, entries=list(entries))
    k_refit = int(config["k_refit"])
    max_fail = int(config["max_consecutive_gate_fail"])

    # --- BLOCK_START: record the freeze snapshot -------------------------
    snapshot = {
        "block_id": block_id,
        "model_version": model_version,
        "weights_sha256": weights_sha256 or _sha256_text(model_version),
        "params_sha256": params_sha256 or _sha256_text(f"{model_version}:params"),
    }
    run.state = "BLOCK_START"
    run.history.append("BLOCK_START")
    _emit(
        run,
        {
            "event": "freeze_snapshot",
            "block_id": block_id,
            **snapshot,
        },
    )

    # --- DELIVER / CAPTURE: not re-implemented (production loop owns it) --
    run.state = "DELIVER"
    run.history.append("DELIVER")
    _emit(run, {"event": "phase_loaded", "block_id": block_id})
    run.state = "CAPTURE"
    run.history.append("CAPTURE")
    _emit(run, {"event": "capture_done", "block_id": block_id, "n_frames": len(entries)})

    # --- JUDGE: the eight frozen verdict thresholds ----------------------
    violations = check_inv1_judge_isolation(entries, block_id)
    if violations:
        raise GuardAbort(violations, block_id)
    run.state = "JUDGE"
    run.history.append("JUDGE")
    _emit(
        run,
        {
            "event": "verdict",
            "block_id": block_id,
            "verdict": dict(verdict),
        },
    )

    # --- REFLOW: append the block's entries (including rejected blocks) --
    accepted = bool(verdict.get("accepted", False))
    include_rejected = bool(config.get("include_rejected", True))
    if not accepted and not include_rejected:
        reflow_entries: list[dict[str, Any]] = []
    else:
        reflow_entries = list(entries)
    violations = check_inv4_append_only(store.entries, reflow_entries)
    violations += check_inv5_schema_monotonic(store.entries, reflow_entries)
    if violations:
        raise GuardAbort(violations, block_id)
    run.state = "REFLOW"
    run.history.append("REFLOW")
    _emit(
        run,
        {
            "event": "reflow_event",
            "block_id": block_id,
            "entry_ids": [str(entry.get("entry_id")) for entry in reflow_entries],
            "includes_rejected": not accepted,
        },
    )

    # --- REFIT: trigger on "块号 mod K = 0" ------------------------------
    if block_id % k_refit == 0:
        run.state = "REFIT"
        run.history.append("REFIT")
        run.refit = True
        fit_inputs = list(store.entries) + reflow_entries
        violations = check_inv2_validation_isolation(fit_inputs, holdout_phases)
        violations += check_inv3_block_freeze(snapshot, snapshot)
        if violations:
            raise GuardAbort(violations, block_id)
        _emit(
            run,
            {
                "event": "refit_event",
                "block_id": block_id,
                "tier": "modal_block",
                "geometry_frozen": True,
                "n_entries_used": len(fit_inputs),
                "model_version": model_version,
            },
        )

        # --- GATE: AND semantics -----------------------------------------
        outcome = evaluate_gate(
            old_nrmse=old_nrmse,
            new_nrmse=new_nrmse,
            old_dirguard=old_dirguard,
            new_dirguard=new_dirguard,
        )
        run.gate = outcome
        run.state = "GATE"
        run.history.append("GATE")
        _emit(
            run,
            {
                "event": "gate_event",
                "block_id": block_id,
                **outcome.to_dict(),
            },
        )
        run.switched = outcome.passed
        if outcome.passed:
            run.state = "BLOCK_END"
            run.history.append("BLOCK_END")
        else:
            run.state = "BASELINE_REBUILD"
            run.history.append("BASELINE_REBUILD")
            _emit(
                run,
                {
                    "event": "baseline_rebuild",
                    "block_id": block_id,
                    "consecutive_gate_failures": max_fail,
                    "probe_frame_budget": 0,
                },
            )
            run.state = "BLOCK_START"
            run.history.append("BLOCK_START")
    else:
        run.state = "BLOCK_END"
        run.history.append("BLOCK_END")

    return run


class GuardAbort(RuntimeError):
    """Raised when a guard invariant is violated (issue #230 §5)."""

    def __init__(self, violations: list[GuardViolation], block_id: int | None) -> None:
        """Store the violations and the block in which they happened."""
        self.violations = list(violations)
        self.block_id = block_id
        detail = "; ".join(
            f"{item.invariant}: {item.detail}" for item in self.violations
        )
        super().__init__(f"guard abort at block {block_id}: {detail}")

    def to_events(self) -> list[dict[str, Any]]:
        """The ``guard_violation`` events to append to ``events.jsonl``."""
        return [
            {
                **violation.to_dict(),
                "abort_point": self.block_id,
            }
            for violation in self.violations
        ]


# ---------------------------------------------------------------------------
# The synthetic self test
# ---------------------------------------------------------------------------


def _synthetic_block(
    session_id: str,
    block_id: int,
    model_version: str,
    *,
    accepted: bool,
    captured_at: float,
) -> list[dict[str, Any]]:
    """Build one synthetic ABBA block of four records.

    This builder is **self-test only**.  Its records are fabricated and must
    never reach a session directory: the ``--execute`` path reads real
    entries from ``--entries-jsonl`` and refuses to run without them.

    Args:
        session_id: The session identifier.
        block_id: The block number.
        model_version: The model version in force.
        accepted: Whether the block was accepted.
        captured_at: The synthetic capture time.

    Returns:
        The four records, in ABBA order.
    """
    verdict = {
        "accepted": accepted,
        "metrics_ref": f"metrics/block{block_id}.json",
    }
    entries: list[dict[str, Any]] = []
    for index, abba_pos in enumerate(ABBA_ORDER):
        entry = build_entry(
            session_id=session_id,
            block_id=block_id,
            frame_role="augment",
            abba_pos=abba_pos,
            candidate_id=f"cand{block_id:03d}",
            model_version=model_version,
            phase_ref=f"phases/block{block_id}_{abba_pos}.npy",
            phase_sha256=_sha256_text(f"phase:{block_id}:{abba_pos}"),
            frame_ref=f"frames/block{block_id}_{abba_pos}.npy",
            frame_sha256=_sha256_text(f"frame:{block_id}:{abba_pos}"),
            exposure_ms_set=0.182,
            exposure_ms_readback=0.182,
            captured_at=captured_at + index,
            verdict=verdict,
        )
        entries.append(entry)
    return entries


def _self_test() -> int:
    """Run the synthetic self test (two ABBA blocks through the state machine).

    The self test exercises, offline and in memory:

    * the registry round trip (write -> read -> normalise);
    * the append-only store with idempotency (a duplicate block adds nothing);
    * the idempotent snapshot (two rebuilds are byte-identical);
    * the block state machine over two blocks, with ``--k-refit 3`` semantics
      exercised through an explicit ``k_refit=1`` run so the ``REFIT``/``GATE``
      states are actually visited;
    * the AND-semantics gate (one failing check vetoes the switch);
    * every invariant guard ``INV-1`` .. ``INV-5``, each fired on purpose.

    Returns:
        ``0`` when every check passed, ``1`` otherwise.
    """
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if condition:
            logger.info(f"[self-test] PASS {name}")
        else:
            failures.append(f"{name}: {detail}")
            logger.error(f"[self-test] FAIL {name} {detail}")

    # --- registry round trip --------------------------------------------
    entry = build_entry(
        session_id="S1",
        block_id=1,
        frame_role="augment",
        abba_pos="A1",
        candidate_id="cand001",
        model_version="mv-1",
        phase_ref="phases/a.npy",
        phase_sha256="0" * 64,
        frame_ref="frames/a.npy",
        frame_sha256="1" * 64,
        exposure_ms_set=0.182,
        exposure_ms_readback=0.182,
        captured_at=1.0,
        verdict={"accepted": True, "metrics_ref": "m.json"},
    )
    check("registry/validate", not validate_entry(entry), str(validate_entry(entry)))
    check(
        "registry/entry_id",
        entry["entry_id"] == "sS1-b1-fA1",
        entry["entry_id"],
    )
    unknown = dict(entry)
    unknown["future_field"] = 42
    normalised = normalise_entry(unknown)
    check(
        "registry/unknown-to-ext",
        normalised["ext"].get("future_field") == 42,
        json.dumps(normalised["ext"], ensure_ascii=False),
    )
    check(
        "registry/extension-rules",
        not check_registry_extension(FIELD_REGISTRY, FIELD_REGISTRY),
        "identical registries must not report problems",
    )

    # --- store: append-only + idempotency + snapshot ----------------------
    store = Store(root=Path("."))
    block1 = _synthetic_block("S1", 1, "mv-1", accepted=True, captured_at=100.0)
    block2 = _synthetic_block("S1", 2, "mv-1", accepted=False, captured_at=200.0)
    appended = store.append_entries(block1)
    check("store/append-first", appended == 4, f"appended {appended}")
    appended = store.append_entries(block1)
    check("store/append-idempotent", appended == 0, f"appended {appended}")
    appended = store.append_entries(block2)
    check("store/append-second", appended == 4, f"appended {appended}")
    first = snapshot_bytes(store.entries)
    second = snapshot_bytes(store.entries)
    check("store/snapshot-idempotent", first == second, "snapshot bytes differ")
    state = derive_state(store.entries)
    check(
        "store/state-derived",
        state.n_entries == 8
        and state.n_augment == 8
        and state.high_water_block == 2
        and state.block_counts == {"1": 4, "2": 4},
        json.dumps(_state_to_dict(state), ensure_ascii=False),
    )

    # --- state machine: no refit (k_refit = 3, block 1 and 2) -------------
    config = build_config({"k_refit": 3})
    check("config/valid", not validate_config(config), str(validate_config(config)))
    run1 = run_block(
        store=store,
        block_id=1,
        entries=block1,
        config=config,
        verdict={"accepted": True, "metrics_ref": "m1.json"},
        model_version="mv-1",
        old_nrmse=0.10,
        new_nrmse=0.10,
        old_dirguard=0.50,
        new_dirguard=0.50,
        holdout_phases=set(),
    )
    check(
        "machine/no-refit-path",
        run1.history == ["BLOCK_START", "DELIVER", "CAPTURE", "JUDGE", "REFLOW",
                         "BLOCK_END"],
        str(run1.history),
    )
    check("machine/no-refit-flag", not run1.refit, "refit ran unexpectedly")

    # --- state machine: refit + gate pass --------------------------------
    # The block's own records carry its own block id: INV-1 aborts when a
    # refeed frame (block id < current) enters the current verdict.
    config_k1 = build_config({"k_refit": 1})
    block3 = _synthetic_block("S1", 3, "mv-1", accepted=False, captured_at=300.0)
    run2 = run_block(
        store=store,
        block_id=3,
        entries=block3,
        config=config_k1,
        verdict={"accepted": False, "metrics_ref": "m3.json"},
        model_version="mv-1",
        old_nrmse=0.10,
        new_nrmse=0.105,
        old_dirguard=0.50,
        new_dirguard=0.55,
        holdout_phases=set(),
    )
    check(
        "machine/refit-gate-path",
        run2.history == ["BLOCK_START", "DELIVER", "CAPTURE", "JUDGE", "REFLOW",
                         "REFIT", "GATE", "BLOCK_END"],
        str(run2.history),
    )
    check("machine/gate-switch", run2.switched, "gate did not switch")
    check(
        "machine/gate-and-semantics",
        run2.gate is not None and all(
            bool(item["passed"]) for item in run2.gate.checks
        ),
        "gate checks failed",
    )
    check(
        "machine/refit-uses-increment",
        any(
            event.get("event") == "refit_event"
            and int(event.get("n_entries_used", 0)) > 0
            for event in run2.events
        ),
        "no refit_event with entries",
    )

    # --- gate veto -> drift flag + baseline rebuild -----------------------
    block4 = _synthetic_block("S1", 4, "mv-1", accepted=True, captured_at=400.0)
    run3 = run_block(
        store=store,
        block_id=4,
        entries=block4,
        config=config_k1,
        verdict={"accepted": True, "metrics_ref": "m4.json"},
        model_version="mv-1",
        old_nrmse=0.10,
        new_nrmse=0.20,
        old_dirguard=0.50,
        new_dirguard=0.40,
        holdout_phases=set(),
    )
    check(
        "machine/gate-veto",
        run3.gate is not None and not run3.gate.passed and run3.gate.drift_flag,
        "gate did not veto",
    )
    check(
        "machine/baseline-rebuild",
        "BASELINE_REBUILD" in run3.history,
        str(run3.history),
    )
    check("machine/no-switch-on-veto", not run3.switched, "switched on veto")
    check(
        "machine/gate-event-has-per-check",
        any(
            event.get("event") == "gate_event"
            and len(event.get("checks", [])) == len(GATE_CHECKLIST)
            for event in run3.events
        ),
        "gate_event does not record every check",
    )

    # --- INV-1: a refeed frame injected into the verdict aborts ----------
    injected = block3 + [dict(block2[0], entry_id="sS1-b2-fA1")]
    violations = check_inv1_judge_isolation(injected, current_block_id=3)
    check(
        "guard/INV-1",
        len(violations) == 1 and violations[0].invariant == "INV-1",
        str(violations),
    )

    # --- INV-2: a validation phase in the fit inputs aborts --------------
    violations = check_inv2_validation_isolation(
        [dict(block1[0], phase_ref="holdout/phase0.npy")], {"holdout/phase0.npy"}
    )
    check(
        "guard/INV-2",
        len(violations) == 1 and violations[0].invariant == "INV-2",
        str(violations),
    )

    # --- INV-3: a model change inside the block aborts --------------------
    violations = check_inv3_block_freeze(
        {"model_version": "mv-1", "weights_sha256": "a", "params_sha256": "b"},
        {"model_version": "mv-2", "weights_sha256": "a", "params_sha256": "b"},
    )
    check(
        "guard/INV-3",
        len(violations) == 1 and violations[0].invariant == "INV-3",
        str(violations),
    )

    # --- INV-4: rewriting a stored entry aborts --------------------------
    violations = check_inv4_append_only(
        store.entries, [dict(block1[0], verdict={"accepted": False})]
    )
    check(
        "guard/INV-4",
        len(violations) == 1 and violations[0].invariant == "INV-4",
        str(violations),
    )

    # --- INV-5: a newer schema version aborts ----------------------------
    violations = check_inv5_schema_monotonic(
        [], [dict(block1[0], schema_version=SCHEMA_VERSION + 1)]
    )
    check(
        "guard/INV-5",
        len(violations) == 1 and violations[0].invariant == "INV-5",
        str(violations),
    )

    # --- supersedes chain: a correction is a new entry --------------------
    corrected = build_entry(
        session_id="S1",
        block_id=1,
        frame_role="augment",
        abba_pos="A1",
        candidate_id="cand001",
        model_version="mv-1",
        phase_ref="phases/a.npy",
        phase_sha256="0" * 64,
        frame_ref="frames/a.npy",
        frame_sha256="1" * 64,
        exposure_ms_set=0.182,
        exposure_ms_readback=0.182,
        captured_at=1.0,
        verdict={"accepted": False, "metrics_ref": "m.json"},
        entry_id="sS1-b1-fA1-fix",
        supersedes="sS1-b1-fA1",
    )
    violations = check_inv4_append_only(store.entries, [corrected])
    check("guard/INV-4-supersedes-ok", not violations, str(violations))
    appended = store.append_entries([corrected])
    check("store/supersede-appended", appended == 1, f"appended {appended}")

    _self_test_filesystem(check)

    if failures:
        logger.error(f"[self-test] {len(failures)} check(s) failed")
        for failure in failures:
            logger.error(f"[self-test]   {failure}")
        return 1
    logger.info("[self-test] all checks passed")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_plan(dataset_dir: Path, config: dict[str, Any]) -> None:
    """Print the state-machine plan (the default, non-executing mode)."""
    logger.info(DEVICE_STATEMENT)
    logger.info(f"[plan] dataset dir      : {dataset_dir}")
    logger.info(f"[plan] entry log        : {dataset_dir / LOG_NAME}")
    logger.info(f"[plan] state snapshot   : {dataset_dir / STATE_NAME}")
    logger.info(f"[plan] validation set   : {dataset_dir / HOLDOUT_NAME}")
    logger.info(f"[plan] event log        : {dataset_dir / EVENTS_NAME}")
    logger.info(f"[plan] schema           : {dataset_dir / SCHEMA_DIR / SCHEMA_NAME}")
    logger.info(f"[plan] schema version   : {SCHEMA_VERSION}")
    logger.info(f"[plan] config version   : {config['config_version']}")
    logger.info(f"[plan] k_refit          : {config['k_refit']} "
                f"(refit when block_id mod K = 0)")
    logger.info(f"[plan] registry fields  : {len(FIELD_REGISTRY)}")
    for spec in FIELD_REGISTRY:
        logger.info(
            f"[plan]   {spec.name:<22} {spec.kind:<5} "
            f"required={str(spec.required):<5} writer={spec.writer}"
        )
    logger.info(f"[plan] invariants       : {', '.join(INVARIANTS)}")
    logger.info("[plan] block state machine (issue #230 §3):")
    for source, target, condition, event in TRANSITIONS:
        logger.info(f"[plan]   {source} -> {target}  ({condition})  event={event}")
    logger.info("[plan] gate checks (issue #230 §4, AND semantics):")
    for check_row in GATE_CHECKLIST:
        logger.info(
            f"[plan]   {check_row.check_id}  {check_row.metric}  "
            f"threshold={check_row.threshold}"
        )
    logger.info("[plan] nothing was written (add --execute to write)")


def _group_by_block(entries: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    """Group the incoming records by block id, in block order.

    Args:
        entries: The records read from the input file.

    Returns:
        A list of ``(block_id, records)`` pairs, ordered by block id.

    Raises:
        StoreError: When a record carries no integer ``block_id``.
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        block_id = entry.get("block_id")
        if not isinstance(block_id, int) or isinstance(block_id, bool):
            raise StoreError(
                f"entry '{entry.get('entry_id')}' has no integer block_id; "
                "the block boundary is what drives the state machine"
            )
        grouped.setdefault(block_id, []).append(entry)
    return [(block_id, grouped[block_id]) for block_id in sorted(grouped)]


def _block_verdict(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the block verdict summary from the block's records.

    Args:
        entries: The records of one block.

    Returns:
        The verdict summary; the block is accepted when every record says so.
    """
    accepted = all(
        bool((entry.get("verdict") or {}).get("accepted", False))
        for entry in entries
    )
    metrics = [
        str((entry.get("verdict") or {}).get("metrics_ref", ""))
        for entry in entries
    ]
    refs = [ref for ref in metrics if ref]
    return {
        "accepted": accepted,
        "metrics_ref": refs[0] if refs else "",
    }


def _self_test_filesystem(check: Any) -> None:
    """Run the filesystem-level self test inside a temporary directory.

    Every synthetic record of this test lives in a
    :func:`tempfile.TemporaryDirectory`; nothing is written to a session
    directory and nothing survives the test.

    The checks are:

    * the entry log is **append-only** -- the bytes that were already on disk
      are an unchanged prefix after a second landing;
    * a second landing of the same entries adds **zero** new lines;
    * a repeated ``entry_id`` with different content is **refused**;
    * a repeated ``entry_id`` with identical content is idempotent;
    * the validation-holdout declaration is written **once** and its bytes
      never change; a conflicting declaration is refused;
    * ``--execute`` without ``--entries-jsonl`` writes **nothing**;
    * the generated schema accepts a legal record and rejects an illegal one.

    Args:
        check: The ``check(name, condition, detail)`` helper of the self test.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ao_refeed_selftest_") as tmp:
        root = Path(tmp)
        session = root / "session"
        store = Store(root=session)
        block = _synthetic_block("S1", 1, "mv-1", accepted=True, captured_at=100.0)

        # --- first landing -----------------------------------------------
        store.append_entries(block)
        store.write_log()
        store.write_snapshot()
        store.write_schema()
        first_bytes = store.log_path.read_bytes()
        first_lines = _count_lines(first_bytes)
        check(
            "fs/first-landing-lines",
            first_lines == len(block),
            f"{first_lines} lines, expected {len(block)}",
        )

        # --- second landing: the existing bytes are an unchanged prefix ---
        store2 = Store(root=session)
        store2.load()
        check(
            "fs/load-recovers-entries",
            len(store2.entries) == len(block),
            f"loaded {len(store2.entries)} entries",
        )
        store2.append_entries(block)
        store2.write_log()
        second_bytes = store2.log_path.read_bytes()
        check(
            "fs/append-preserves-prefix",
            second_bytes.startswith(first_bytes),
            "the bytes already on disk were not preserved as a prefix",
        )
        check(
            "fs/second-landing-no-new-lines",
            _count_lines(second_bytes) == first_lines,
            f"{_count_lines(second_bytes)} lines, expected {first_lines}",
        )
        check(
            "fs/second-landing-bytes-identical",
            second_bytes == first_bytes,
            "the idempotent re-landing changed the file",
        )

        # --- a third landing after new entries: the prefix still holds ----
        block2 = _synthetic_block("S1", 2, "mv-1", accepted=False, captured_at=200.0)
        store2.append_entries(block2)
        store2.write_log()
        third_bytes = store2.log_path.read_bytes()
        check(
            "fs/append-extends-only",
            third_bytes.startswith(first_bytes)
            and _count_lines(third_bytes) == first_lines + len(block2),
            "the append did not extend the log",
        )

        # --- a repeated id with different content is refused ---------------
        conflict_store = Store(root=session)
        conflict_store.load()
        conflicting = [dict(block[0], verdict={"accepted": False})]
        refused = False
        try:
            conflict_store.append_entries(conflicting)
        except StoreConflict:
            refused = True
        check("fs/id-conflict-refused", refused, "the conflicting id was accepted")
        check(
            "fs/id-conflict-no-write",
            store.log_path.read_bytes() == third_bytes,
            "the refused conflict changed the log",
        )

        # --- an identical id is idempotent --------------------------------
        same_store = Store(root=session)
        same_store.load()
        appended_same = same_store.append_entries([block[0]])
        same_store.write_log()
        check(
            "fs/id-identical-idempotent",
            appended_same == 0
            and store.log_path.read_bytes() == third_bytes,
            f"appended {appended_same} and changed the log",
        )

        # --- the holdout is written once and never rewritten --------------
        declaration = {
            "schema": "session_validation_holdout/1",
            "session_id": "S1",
            "phase_refs": ["holdout/p0.npy"],
            "note": "written once",
        }
        holdout_store = Store(root=session)
        holdout_store.write_holdout(declaration)
        holdout_bytes = holdout_store.holdout_path.read_bytes()
        holdout_store.write_holdout(declaration)
        check(
            "fs/holdout-written-once",
            holdout_store.holdout_path.read_bytes() == holdout_bytes,
            "re-writing the same declaration changed the file",
        )
        conflicting_declaration = dict(declaration, phase_refs=["holdout/p1.npy"])
        holdout_refused = False
        try:
            holdout_store.write_holdout(conflicting_declaration, explicit=True)
        except StoreConflict:
            holdout_refused = True
        check(
            "fs/holdout-conflict-refused",
            holdout_refused,
            "the conflicting declaration was accepted",
        )
        check(
            "fs/holdout-bytes-unchanged",
            holdout_store.holdout_path.read_bytes() == holdout_bytes,
            "the refused conflict changed the holdout file",
        )
        # An implicit (session-start) declaration must never clear an
        # existing validation set.
        implicit_store = Store(root=session)
        implicit_store.write_holdout(
            {"schema": "session_validation_holdout/1", "session_id": "S1",
             "phase_refs": [], "note": "implicit"},
            explicit=False,
        )
        check(
            "fs/holdout-implicit-keeps-existing",
            implicit_store.holdout_path.read_bytes() == holdout_bytes,
            "the implicit declaration overwrote the validation set",
        )

        # --- a truncated / edited log is refused on load -------------------
        tampered_dir = root / "tampered"
        tampered_store = Store(root=tampered_dir)
        tampered_store.append_entries(block)
        tampered_store.write_log()
        original = tampered_store.log_path.read_bytes()
        tampered_store.log_path.write_bytes(original[:-20])
        truncated_refused = False
        try:
            Store(root=tampered_dir).load()
        except StoreError:
            truncated_refused = True
        check(
            "fs/truncated-log-refused",
            truncated_refused,
            "the truncated log was loaded without complaint",
        )

        edited_dir = root / "edited"
        edited_store = Store(root=edited_dir)
        edited_store.append_entries(block)
        edited_store.write_log()
        # A canonical value edit (100.0 -> 999.0): every line stays valid
        # canonical JSON, so only the committed-prefix digest can detect it.
        edited_text = edited_store.log_path.read_text(encoding="utf-8")
        if '"captured_at": 100.0' not in edited_text:
            raise StoreError("self-test fixture: captured_at marker not found")
        edited_store.log_path.write_text(
            edited_text.replace('"captured_at": 100.0', '"captured_at": 999.0', 1),
            encoding="utf-8",
        )
        edited_refused = False
        try:
            Store(root=edited_dir).load()
        except StoreError:
            edited_refused = True
        check(
            "fs/edited-log-refused",
            edited_refused,
            "the edited log was loaded without complaint",
        )

        # --- --execute without an input writes nothing --------------------
        empty_dir = root / "empty_session"
        code = main(
            [
                "--dataset-dir",
                str(empty_dir),
                "--execute",
            ]
        )
        check("fs/execute-without-input-refused", code == 2, f"exit code {code}")
        check(
            "fs/execute-without-input-zero-writes",
            not empty_dir.exists() or not any(empty_dir.iterdir()),
            "the refused execute created files",
        )

        # --- the schema accepts a legal record and rejects an illegal one --
        schema = build_schema()
        legal = _synthetic_block("S1", 1, "mv-1", accepted=True, captured_at=100.0)[0]
        illegal = dict(legal, frame_role="not_a_role")
        missing = {key: value for key, value in legal.items() if key != "entry_id"}
        positive, negative, missing_problem = _validate_with_jsonschema(schema, legal, illegal, missing)
        check(
            "schema/accepts-legal-record",
            positive,
            "the standard validator rejected a legal record",
        )
        check(
            "schema/rejects-illegal-enum",
            negative,
            "the standard validator accepted an illegal enum value",
        )
        check(
            "schema/rejects-missing-required",
            missing_problem,
            "the standard validator accepted a record without a required field",
        )


def _validate_with_jsonschema(
    schema: dict[str, Any],
    legal: dict[str, Any],
    illegal: dict[str, Any],
    missing: dict[str, Any],
) -> tuple[bool, bool, bool]:
    """Check the generated schema with the ``jsonschema`` library.

    The library is an optional dependency of this script: when it is absent
    the checks fall back to the registry validator and the reason is logged,
    because the schema file itself must stay a pure function of the registry.

    Args:
        schema: The generated schema.
        legal: A record that must validate.
        illegal: A record that must not validate.
        missing: A record without a required field that must not validate.

    Returns:
        ``(legal_ok, illegal_rejected, missing_rejected)``.
    """
    try:
        import jsonschema
    except ImportError:
        logger.warning(
            "[self-test] jsonschema is not installed; the schema checks fall "
            "back to the registry validator (pip install jsonschema to run "
            "the standard-validator checks)"
        )
        return (
            not validate_entry(legal),
            bool(validate_entry(illegal)),
            bool(validate_entry(missing)),
        )
    validator = jsonschema.Draft202012Validator(schema)
    legal_ok = not list(validator.iter_errors(legal))
    illegal_rejected = bool(list(validator.iter_errors(illegal)))
    missing_rejected = bool(list(validator.iter_errors(missing)))
    return legal_ok, illegal_rejected, missing_rejected


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: The argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        The process exit code: ``0`` on success, ``1`` on a guard abort, a
        refused write or a failed self test, ``2`` when ``--execute`` was given
        without an input file (nothing is written in that case).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "data" / "debug" / "online_refeed_demo",
        help="Session directory holding the dataset store.",
    )
    parser.add_argument(
        "--entries-jsonl",
        type=Path,
        default=None,
        help=(
            "JSONL file with the entries to append (one registry-valid record "
            "per line). Required by --execute: the script never fabricates "
            "records, and the synthetic self-test data never reaches here."
        ),
    )
    parser.add_argument(
        "--k-refit",
        type=int,
        default=int(CONFIG_DEFAULTS["k_refit"]),
        help="Refit block interval K (issue #230 §6, 1..5).",
    )
    parser.add_argument(
        "--model-version",
        default="mv-1",
        help="The model version that gave the appended blocks their direction.",
    )
    parser.add_argument(
        "--write-holdout",
        type=Path,
        default=None,
        help=(
            "Explicit session-start action: write the independent validation "
            "set declaration from this JSON file. The declaration is written "
            "once per session and is never rewritten or cleared. This is "
            "deliberately NOT part of the refeed path (issue #230 §2 rule 4)."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write the store; without it the script only prints a plan.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the synthetic offline self test and exit.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    config = build_config({"k_refit": args.k_refit})
    problems = validate_config(config)
    if problems:
        for problem in problems:
            logger.error(f"[config] {problem}")
        return 1

    dataset_dir = args.dataset_dir.resolve()

    # --- explicit session start: write the validation-set declaration ----
    if args.write_holdout is not None:
        holdout_path = args.write_holdout.resolve()
        if not holdout_path.is_file():
            logger.error(
                f"[holdout] refused: declaration file not found: {holdout_path}"
            )
            return 2
        try:
            declaration = json.loads(holdout_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            logger.error(f"[holdout] refused: {error}")
            return 2
        store = Store(root=dataset_dir)
        try:
            outcome = store.write_holdout(declaration, explicit=True)
        except StoreConflict as conflict:
            logger.error(f"[holdout] refused: {conflict}")
            return 2
        logger.info(f"[holdout] {outcome} -> {store.holdout_path}")
        if not args.execute:
            return 0

    if not args.execute:
        _print_plan(dataset_dir, config)
        return 0

    # --- refuse to execute without a real input ------------------------
    if args.entries_jsonl is None:
        logger.error(
            "[execute] refused: --entries-jsonl is required; this script never "
            "fabricates entries and writes nothing without an input file"
        )
        return 2
    entries_path = args.entries_jsonl.resolve()
    if not entries_path.is_file():
        logger.error(f"[execute] refused: input file not found: {entries_path}")
        return 2
    try:
        incoming = read_jsonl(entries_path)
    except StoreError as error:
        logger.error(f"[execute] refused: {error}")
        return 2
    if not incoming:
        logger.error(
            f"[execute] refused: input file holds no entries: {entries_path}"
        )
        return 2

    store = Store(root=dataset_dir)
    try:
        store.load()
    except StoreError as error:
        logger.error(f"[execute] refused: {error}")
        return 2
    holdout_phases = read_holdout_phases(store.holdout_path)

    blocks = _group_by_block(incoming)
    try:
        for block_id, block_entries in blocks:
            run = run_block(
                store=store,
                block_id=block_id,
                entries=block_entries,
                config=config,
                verdict=_block_verdict(block_entries),
                model_version=args.model_version,
                old_nrmse=0.10,
                new_nrmse=0.10,
                old_dirguard=0.50,
                new_dirguard=0.50,
                holdout_phases=holdout_phases,
            )
            store.append_entries(block_entries)
            for event in run.events:
                store.append_event(event)
    except GuardAbort as abort:
        for event in abort.to_events():
            store.append_event(event)
        try:
            store.write_events()
        except StoreError as error:
            logger.error(f"[execute] refused: {error}")
            return 2
        logger.error(str(abort))
        return 2
    except StoreConflict as conflict:
        logger.error(f"[execute] refused: {conflict}")
        return 2

    try:
        # Issue #230 §2 rule (4): the refeed path cannot write the validation
        # set declaration at all -- the isolation is physical, not a matter of
        # code discipline.  The holdout is written by the explicit session
        # start action (``--write-holdout``), never by the refeed run.
        store.write_log()
        store.write_snapshot()
        store.write_schema()
        store.write_events()
    except StoreConflict as conflict:
        logger.error(f"[execute] refused: {conflict}")
        return 2
    except StoreError as error:
        logger.error(f"[execute] refused: {error}")
        return 2
    logger.info(f"[execute] appended {len(incoming)} entries to {store.log_path}")
    logger.info(f"[execute] events -> {store.events_path}")
    logger.info(f"[execute] snapshot -> {store.state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
