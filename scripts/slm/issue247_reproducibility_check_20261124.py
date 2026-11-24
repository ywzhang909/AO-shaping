"""Issue #247 同窗口重复性离线检查（2026-11-24）。

只读一个 3 轮采集目录（``r1/r2/r3``，各 23 图样 × 4 帧 = 276 帧），按
``docs/dev/hw_window_20261124_issue247_reproducibility_prereg.md`` 校验冻结项，
并用 ``issue247_waveform_probe_solve_20261122.py`` 的同一分析器逐轮重算。
输入不合格一律拒绝（退出码 3），不修补、不丢样本；不写采集目录、不碰设备。

接受的 ``plan.json`` 轮次结构（三选一）
---------------------------------------
1. ``rounds`` 是列表：每项形如
   ``{"id": "r1", "patterns": [{...}, ...]}``；
2. ``rounds`` 是映射：``{"r1": [...], "r2": [...], "r3": [...]}``；
3. ``patterns`` 平铺：每项带 ``round``/``round_id``，或 id 以 ``r1__``、
   ``r1_``、``r1-`` 等轮次前缀开头，或采用采集通道实际发出的
   ``<base_pattern_id>__rN`` 后缀 + 1 基 ``repeat_index``
   （顶层 ``reproduction.rounds`` 为 ``[r1, r2, r3]``）。

每个图样必须有轮内唯一的 ``id``；``base_id``/``base_pattern_id`` 可显式给出，
也可由轮次前缀或 ``__rN`` 后缀剥离。后缀、``repeat_index``、显式 round 字段、
显式 base id 两两交叉校验，任何不一致都拒绝。23 个 base id 必须与
attribution 冻结集合完全一致。相位摘要
``phase_sha256`` 必须逐轮相同。缺帧、重复帧、错误曝光/帧数/settle、
饱和、flat 漂移超 2.0 px、设备未正常释放、基线分析产物缺失，均拒绝。

证据分工
--------
- 采集目录的 ``optical_path_record.json`` 只提供设备身份（SLM 型号/#1/波长、
  相机型号/序列号/位深/满量程/曝光）与 ``exposure_check``（saturated /
  settle_timeout）；它不含六步预检 verdict，这是采集侧设计。
- 六步预检 verdict 只来自 ``--probe-dir`` 指向的 run 目录内
  ``optical_path_probe.json``：要求 ``verdict.state == 'ok'``、exit_code 0，
  并从 ``verdict.reasons`` 保守解析可用曝光窗口，冻结曝光 0.04 ms 必须在窗口内；
  解析不出就拒绝，不假定曝光可用。
- ``capture_settings.settle_s``（若存在）与其它 settle 记录一样按 0.8 s 严判；
  缺失 settle 一律拒绝，没有容忍开关。

输出只写到 ``--out-dir``，默认不覆盖已有产物。JSON 保留逐帧/逐轮质心、
候选峰、``axis_angle_deg``、峰数与峰/背景比；Markdown 只做可读摘要。
方向重复性门只用 ``axis_angle_deg``（模 180°），不退回 principal angle；
缺峰保留为缺测，不调阈值。

用法::

    python scripts/slm/issue247_reproducibility_check_20261124.py \
        --dir data/debug/hw_window_20261124_issue247_reproducibility \
        --baseline-dir data/debug/hw_window_20261123_issue247_waveform_attribution \
        --probe-dir data/debug/optical_path_probe/<session>/<run_ts> \
        --out-dir report/shape_trials/issue247_reproducibility_20261124
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import fmt_general
from scripts._common.provenance import provenance_block_for

ANALYZER_PATH = Path(__file__).resolve().parent / "issue247_waveform_probe_solve_20261122.py"
PROBE_FILE_NAME = "optical_path_probe.json"
DEFAULT_JSON_NAME = "issue247_reproducibility_20261124.json"
DEFAULT_MD_NAME = "issue247_reproducibility_20261124.md"
#: 报告溯源块的注册表 key：`report/shape_trials/<系列_日期>/<系列_日期>.md`
REPORT_KEY = "report/shape_trials/issue247_reproducibility_20261124/issue247_reproducibility_20261124.md"

ROUND_IDS = ("r1", "r2", "r3")
EXPECTED_FRAMES_PER_PATTERN = 4
EXPECTED_PATTERNS_PER_ROUND = 23
EXPECTED_TOTAL_FRAMES = 276
EXPECTED_EXPOSURE_MS = 0.04
EXPECTED_SETTLE_S = 0.8
SATURATION_FRACTION = 0.9
SATURATION_PLATEAU_PIXELS = 9
DRIFT_MAX_PX = 2.0
REPEATABILITY_MAX_DEG = 2.0
#: principal 峰心落入任一边界 8 px 带内 → 视为被边界截断的极大值（缺测）。
EDGE_BAND_PX = 8.0

#: attribution 窗口的 23 个冻结 base pattern，顺序即 r1/r3 的冻结顺序。
EXPECTED_BASE_IDS: tuple[str, ...] = (
    "flat_pre",
    "grating_x_p128",
    "tilt_x_plus",
    "tilt_x_minus",
    "patch_x_plus_p128",
    "patch_x_minus_p128",
    "grating_y_p128",
    "tilt_y_plus",
    "tilt_y_minus",
    "patch_y_plus_p128",
    "patch_y_minus_p128",
    "flat_mid",
    "grating_x_p192",
    "tilt_x_plus_p192",
    "tilt_x_minus_p192",
    "patch_x_plus_p192",
    "patch_x_minus_p192",
    "grating_y_p192",
    "tilt_y_plus_p192",
    "tilt_y_minus_p192",
    "patch_y_plus_p192",
    "patch_y_minus_p192",
    "flat_post",
)
FLAT_BASE_IDS = frozenset({"flat_pre", "flat_mid", "flat_post"})
EVALUABLE_BASE_IDS = tuple(
    base for base in EXPECTED_BASE_IDS if base not in FLAT_BASE_IDS
)
GATE_KEYS = (
    "G1_waveform_dependency",
    "G2_single_family_anomaly",
    "G3_axis_separation",
    "G4_sign_consistency",
)

_ANALYZER: Any | None = None


class CheckError(ValueError):
    """冻结项不满足；调用方应拒绝该采集，而不是修补。"""


def _analyzer() -> Any:
    """按文件路径加载分析器，避免 scripts 包布局差异影响 CLII。"""
    global _ANALYZER
    if _ANALYZER is None:
        spec = importlib.util.spec_from_file_location(
            "issue247_waveform_probe_solve_20261122_cached", ANALYZER_PATH
        )
        if spec is None or spec.loader is None:
            raise CheckError(f"无法加载分析器 {ANALYZER_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _ANALYZER = module
    return _ANALYZER


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"缺少 {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckError(f"{path} 不是合法 JSON：{exc}") from exc
    _require(isinstance(payload, dict), f"{path} 不是 JSON 对象")
    return payload


def _sha256_array(array: np.ndarray) -> str:
    """与采集通道 ``digest(array)`` 相同的规范化：连续化后的数组字节 SHA-256。

    采集侧摘要按内存数组字节计算（不是 .npy 文件字节），因此这里必须用同一
    口径核对；两者不同会把完好的采集误判为篡改。
    """
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _round_name(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    compact = text.replace(" ", "").replace("-", "").replace("_", "")
    aliases = {
        "r1": "r1",
        "round1": "r1",
        "1": "r1",
        "r2": "r2",
        "round2": "r2",
        "2": "r2",
        "r3": "r3",
        "round3": "r3",
        "3": "r3",
    }
    return aliases.get(compact)


def _split_round_suffix(pattern_id: str) -> tuple[str, str] | None:
    """把 ``<base>__rN`` 拆成 ``(base, rN)``；无后缀返回 ``None``。"""
    match = re.fullmatch(r"(.+?)__r([123])", pattern_id.strip(), flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1), f"r{match.group(2)}"


def _strip_round_prefix(pattern_id: str, round_id: str) -> str | None:
    """剥离 ``r1__`` / ``round_r1_`` / ``r1-`` 等前缀，返回 base id。"""
    tokens = [round_id]
    if round_id.startswith("r"):
        tokens.append(round_id[1:])
    for token in tokens:
        prefixes = (
            token,
            f"round_{token}",
            f"round{token}",
            f"round-{token}",
            f"round {token}",
        )
        for prefix in prefixes:
            if not pattern_id.lower().startswith(prefix.lower()):
                continue
            rest = pattern_id[len(prefix) :]
            if rest and rest[0] in "_-./:| ":
                candidate = rest.lstrip("_-./:| ")
                if candidate:
                    return candidate
    return None


@dataclass(frozen=True)
class PatternRecord:
    round_id: str
    pattern_id: str
    base_id: str
    entry: dict[str, Any]
    order_index: int


@dataclass
class RoundRecord:
    round_id: str
    patterns: list[PatternRecord]

    @property
    def base_ids(self) -> list[str]:
        return [item.base_id for item in self.patterns]


@dataclass
class FrameEvidence:
    path: Path
    sha256: str
    max_counts: float
    pixels_at_or_above_threshold: int


def _normalise_round_entries(
    round_id: str, raw_patterns: list[Any]
) -> list[PatternRecord]:
    records: list[PatternRecord] = []
    seen_ids: set[str] = set()
    seen_bases: set[str] = set()
    for index, raw in enumerate(raw_patterns):
        _require(isinstance(raw, dict), f"{round_id} 第 {index} 个 pattern 不是对象")
        entry = dict(raw)
        pattern_id = str(
            entry.get("id") or entry.get("pattern_id") or ""
        ).strip()
        _require(bool(pattern_id), f"{round_id} 第 {index} 个 pattern 缺少 id")
        _require(
            pattern_id not in seen_ids,
            f"{round_id} 内 pattern id 重复：{pattern_id}",
        )
        seen_ids.add(pattern_id)

        explicit_round = entry.get("round") or entry.get("round_id")
        if explicit_round is not None:
            _require(
                _round_name(explicit_round) == round_id,
                f"{pattern_id} 的 round={explicit_round!r} 与轮次 {round_id} 不一致",
            )

        suffix = _split_round_suffix(pattern_id)
        if suffix is not None:
            _require(
                suffix[1] == round_id,
                f"{pattern_id} 的轮次后缀 {suffix[1]!r} 与轮次 {round_id} 不一致",
            )

        repeat_index = entry.get("repeat_index")
        if repeat_index is not None:
            _require(
                _is_number(repeat_index)
                and not isinstance(repeat_index, bool)
                and int(repeat_index) == float(repeat_index),
                f"{pattern_id} 的 repeat_index={repeat_index!r} 不是整数",
            )
            repeat_round = _round_name(f"r{int(repeat_index)}")
            _require(
                repeat_round == round_id,
                f"{pattern_id} 的 repeat_index={repeat_index} 与轮次 {round_id} 不一致",
            )
            if suffix is not None:
                _require(
                    suffix[1] == repeat_round,
                    f"{pattern_id} 的轮次后缀 {suffix[1]!r} 与 "
                    f"repeat_index={repeat_index} 不一致",
                )

        explicit_base = (
            entry.get("base_id")
            or entry.get("base_pattern_id")
            or entry.get("base")
        )
        if explicit_base is not None:
            base_id = str(explicit_base).strip()
            if suffix is not None:
                _require(
                    base_id == suffix[0],
                    f"{pattern_id} 的 base_pattern_id={base_id!r} 与 "
                    f"轮次后缀蕴含的 {suffix[0]!r} 不一致",
                )
        elif suffix is not None:
            base_id = suffix[0]
        else:
            stripped = _strip_round_prefix(pattern_id, round_id)
            base_id = stripped if stripped is not None else pattern_id
        _require(bool(base_id), f"{pattern_id} 无法解析 base id")
        _require(
            base_id in EXPECTED_BASE_IDS,
            f"{round_id}/{pattern_id} 的 base id={base_id!r} 不在 23 个冻结图样内",
        )
        _require(
            base_id not in seen_bases,
            f"{round_id} 内 base pattern 重复：{base_id}",
        )
        seen_bases.add(base_id)
        entry["id"] = pattern_id
        entry["base_id"] = base_id
        records.append(
            PatternRecord(
                round_id=round_id,
                pattern_id=pattern_id,
                base_id=base_id,
                entry=entry,
                order_index=index,
            )
        )
    _require(
        len(records) == EXPECTED_PATTERNS_PER_ROUND,
        f"{round_id} 有 {len(records)} 个图样，预注册要求 {EXPECTED_PATTERNS_PER_ROUND} 个",
    )
    _require(
        set(seen_bases) == set(EXPECTED_BASE_IDS),
        f"{round_id} 的 base pattern 集合与冻结 23 图样不一致",
    )
    return records


def _extract_rounds(plan: dict[str, Any]) -> dict[str, RoundRecord]:
    raw_rounds = plan.get("rounds")
    grouped: dict[str, list[Any]] = {}

    if isinstance(raw_rounds, list):
        _require(
            len(raw_rounds) == 3,
            f"plan.rounds 有 {len(raw_rounds)} 轮，预注册要求 3 轮",
        )
        for item in raw_rounds:
            _require(isinstance(item, dict), "plan.rounds 的每一项必须是对象")
            round_id = _round_name(
                item.get("id") or item.get("round") or item.get("round_id")
                or item.get("name")
            )
            _require(round_id is not None, f"无法识别轮次名称：{item!r}")
            _require(round_id not in grouped, f"轮次重复：{round_id}")
            patterns = item.get("patterns")
            _require(
                isinstance(patterns, list) and patterns,
                f"{round_id} 缺少非空 patterns 列表",
            )
            grouped[round_id] = patterns
    elif isinstance(raw_rounds, dict):
        _require(
            len(raw_rounds) == 3,
            f"plan.rounds 有 {len(raw_rounds)} 轮，预注册要求 3 轮",
        )
        for key, value in raw_rounds.items():
            round_id = _round_name(key)
            _require(round_id is not None, f"无法识别轮次名称：{key!r}")
            _require(round_id not in grouped, f"轮次重复：{round_id}")
            if isinstance(value, dict):
                value = value.get("patterns")
            _require(
                isinstance(value, list) and value,
                f"{round_id} 缺少非空 patterns 列表",
            )
            grouped[round_id] = value
    else:
        flat = plan.get("patterns")
        _require(
            isinstance(flat, list) and flat,
            "plan.json 必须提供 rounds，或带轮次信息的非空 patterns 列表",
        )
        for raw in flat:
            _require(isinstance(raw, dict), "plan.patterns 的每一项必须是对象")
            round_id = _round_name(raw.get("round") or raw.get("round_id"))
            pattern_id = str(raw.get("id") or raw.get("pattern_id") or "").strip()
            suffix = _split_round_suffix(pattern_id)
            if suffix is not None:
                if round_id is not None:
                    _require(
                        round_id == suffix[1],
                        f"平铺 pattern {pattern_id!r} 的轮次字段与后缀 "
                        f"{suffix[1]!r} 不一致",
                    )
                round_id = suffix[1]
            if round_id is None and raw.get("repeat_index") is not None:
                repeat_index = raw["repeat_index"]
                _require(
                    _is_number(repeat_index)
                    and int(repeat_index) == float(repeat_index),
                    f"平铺 pattern {pattern_id!r} 的 repeat_index={repeat_index!r} "
                    "不是整数",
                )
                round_id = _round_name(f"r{int(repeat_index)}")
            if round_id is None:
                for candidate in ROUND_IDS:
                    if _strip_round_prefix(pattern_id, candidate) is not None:
                        round_id = candidate
                        break
            _require(
                round_id is not None,
                f"平铺 pattern {raw.get('id')!r} 无法确定轮次",
            )
            grouped.setdefault(round_id, []).append(raw)

    _require(
        set(grouped) == set(ROUND_IDS),
        f"轮次集合 {sorted(grouped)} 不等于 {list(ROUND_IDS)}",
    )
    rounds = {
        round_id: RoundRecord(
            round_id=round_id,
            patterns=_normalise_round_entries(round_id, grouped[round_id]),
        )
        for round_id in ROUND_IDS
    }

    all_pattern_ids = [
        record.pattern_id for round_record in rounds.values()
        for record in round_record.patterns
    ]
    _require(
        len(all_pattern_ids) == 69,
        f"pattern id 总数 {len(all_pattern_ids)}，预注册要求 69 个唯一 ID",
    )
    _require(
        len(set(all_pattern_ids)) == 69,
        "69 个轮次 pattern id 重复；轮次 ID 必须唯一",
    )
    return rounds


def _check_order(
    plan: dict[str, Any],
    rounds: dict[str, RoundRecord],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    r1 = rounds["r1"].base_ids
    r2 = rounds["r2"].base_ids
    r3 = rounds["r3"].base_ids
    _require(r2 == list(reversed(r1)), "r2 顺序不是 r1 的严格反序")
    _require(r3 == r1, "r3 顺序与 r1 不一致（预注册要求 r1=r3）")

    frozen = (
        plan.get("frozen_order")
        or plan.get("base_pattern_order")
        or plan.get("attribution_order")
    )
    source = "plan.frozen_order"
    if not isinstance(frozen, list) or not frozen:
        frozen = list(EXPECTED_BASE_IDS)
        source = "frozen_issue247_attribution_order"
    frozen_ids = [str(item) for item in frozen]
    _require(
        frozen_ids == r1,
        f"r1 顺序与冻结顺序（{source}）不一致",
    )
    return {"frozen_order_source": source, "r1": r1, "r2": r2, "r3": r3}


def _check_phase_hashes(
    rounds: dict[str, RoundRecord],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    hashes: dict[str, dict[str, str]] = {}
    for base_id in EXPECTED_BASE_IDS:
        per_round: dict[str, str] = {}
        for round_id in ROUND_IDS:
            record = next(
                item for item in rounds[round_id].patterns if item.base_id == base_id
            )
            value = record.entry.get("phase_sha256")
            _require(
                isinstance(value, str) and bool(value.strip()),
                f"{round_id}/{base_id} 缺少 phase_sha256",
            )
            text = value.strip().lower()
            _require(
                re.fullmatch(r"[0-9a-f]{64}", text) is not None,
                f"{round_id}/{base_id} 的 phase_sha256 不是 64 位十六进制",
            )
            per_round[round_id] = text
        _require(
            len(set(per_round.values())) == 1,
            f"{base_id} 的相位摘要三轮不一致：{per_round}",
        )
        hashes[base_id] = per_round

    baseline_checked = 0
    if baseline is not None:
        baseline_hashes = baseline["phase_sha256"]
        for base_id in EXPECTED_BASE_IDS:
            expected = baseline_hashes.get(base_id)
            if not expected:
                continue
            _require(
                hashes[base_id]["r1"] == expected,
                f"{base_id} 的相位摘要与基线 attribution 不一致",
            )
            baseline_checked += 1
    return {
        "all_equal_across_rounds": True,
        "baseline_checked": baseline_checked,
        "base_patterns": len(hashes),
    }


def _collect_numbers(value: object) -> list[float]:
    found: list[float] = []
    if _is_number(value):
        found.append(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_collect_numbers(item))
    return found


def _collect_settle_values(
    plan: dict[str, Any],
    summary: dict[str, Any],
    rounds: dict[str, RoundRecord],
) -> list[tuple[float, str]]:
    values: list[tuple[float, str]] = []
    keys_s = ("settle_time_s", "settle_s", "settle_seconds", "settle_time_seconds")
    keys_ms = ("settle_ms", "settle_time_ms")
    for name, mapping in (("plan", plan), ("summary", summary)):
        for key in keys_s:
            if key in mapping:
                values.extend(
                    (number, f"{name}.{key}") for number in _collect_numbers(mapping[key])
                )
        for key in keys_ms:
            if key in mapping:
                values.extend(
                    (number / 1000.0, f"{name}.{key}")
                    for number in _collect_numbers(mapping[key])
                )
        settle = mapping.get("settle")
        if settle is not None:
            if _is_number(settle):
                values.append((float(settle), f"{name}.settle"))
            elif isinstance(settle, dict):
                for key in keys_s:
                    if key in settle:
                        values.extend(
                            (number, f"{name}.settle.{key}")
                            for number in _collect_numbers(settle[key])
                        )
                for key in keys_ms:
                    if key in settle:
                        values.extend(
                            (number / 1000.0, f"{name}.settle.{key}")
                            for number in _collect_numbers(settle[key])
                        )
        settings = mapping.get("capture_settings")
        if isinstance(settings, dict):
            for key in keys_s:
                if key in settings:
                    values.extend(
                        (number, f"{name}.capture_settings.{key}")
                        for number in _collect_numbers(settings[key])
                    )
            for key in keys_ms:
                if key in settings:
                    values.extend(
                        (number / 1000.0, f"{name}.capture_settings.{key}")
                        for number in _collect_numbers(settings[key])
                    )
    for round_record in rounds.values():
        for record in round_record.patterns:
            for key in keys_s:
                if key in record.entry:
                    values.extend(
                        (number, f"{record.pattern_id}.{key}")
                        for number in _collect_numbers(record.entry[key])
                    )
            for key in keys_ms:
                if key in record.entry:
                    values.extend(
                        (number / 1000.0, f"{record.pattern_id}.{key}")
                        for number in _collect_numbers(record.entry[key])
                    )
    return values


def _check_fixed_parameters(
    plan: dict[str, Any],
    summary: dict[str, Any],
    rounds: dict[str, RoundRecord],
) -> dict[str, Any]:
    exposure_values: list[tuple[float, str]] = []
    for name, mapping in (("summary", summary), ("plan", plan)):
        for key in ("exposure_ms", "requested_exposure_ms", "exposure_time_ms"):
            if _is_number(mapping.get(key)):
                exposure_values.append((float(mapping[key]), f"{name}.{key}"))
        declared = mapping.get("declared_constants")
        if isinstance(declared, dict):
            for key in ("exposure_ms", "exposure_used_ms"):
                if _is_number(declared.get(key)):
                    exposure_values.append(
                        (float(declared[key]), f"{name}.declared_constants.{key}")
                    )
        settings = mapping.get("capture_settings")
        if isinstance(settings, dict):
            for key in ("exposure_ms", "exposure_used_ms"):
                if _is_number(settings.get(key)):
                    exposure_values.append(
                        (float(settings[key]), f"{name}.capture_settings.{key}")
                    )
    _require(exposure_values, "缺少曝光记录（exposure_ms / requested_exposure_ms）")
    for value, source in exposure_values:
        _require(
            abs(value - EXPECTED_EXPOSURE_MS) <= 1e-9,
            f"{source}={value} ms，预注册固定曝光为 {EXPECTED_EXPOSURE_MS} ms",
        )

    settle_values = _collect_settle_values(plan, summary, rounds)
    _require(settle_values, "缺少 settle 记录（settle_time_s / settle_s）")
    for value, source in settle_values:
        _require(
            abs(value - EXPECTED_SETTLE_S) <= 1e-9,
            f"{source}={value} s，预注册固定 settle 为 {EXPECTED_SETTLE_S} s",
        )

    frames_per_pattern = plan.get("frames_per_pattern")
    frames_by_pattern: dict[str, int] = {}
    for round_record in rounds.values():
        for record in round_record.patterns:
            value: object = record.entry.get("frames")
            if value is None and isinstance(frames_per_pattern, dict):
                value = frames_per_pattern.get(record.pattern_id)
                if value is None:
                    value = frames_per_pattern.get(record.base_id)
            elif value is None and _is_number(frames_per_pattern):
                value = frames_per_pattern
            _require(
                _is_number(value),
                f"{record.round_id}/{record.pattern_id} 缺少每图样帧数",
            )
            frames = int(float(value))
            _require(
                frames == EXPECTED_FRAMES_PER_PATTERN,
                f"{record.round_id}/{record.pattern_id} 帧数 {frames}，"
                f"预注册固定 {EXPECTED_FRAMES_PER_PATTERN} 帧",
            )
            frames_by_pattern[record.pattern_id] = frames
            summary_pattern = summary.get("per_pattern")
            if isinstance(summary_pattern, dict) and record.pattern_id in summary_pattern:
                recorded = summary_pattern[record.pattern_id].get("frames")
                if _is_number(recorded):
                    _require(
                        int(float(recorded)) == EXPECTED_FRAMES_PER_PATTERN,
                        f"summary 中 {record.pattern_id} 的 frames={recorded}",
                    )
    return {
        "exposure_ms": EXPECTED_EXPOSURE_MS,
        "settle_s": EXPECTED_SETTLE_S,
        "frames_per_pattern": EXPECTED_FRAMES_PER_PATTERN,
        "exposure_sources": [source for _, source in exposure_values],
        "settle_sources": [source for _, source in settle_values],
        "patterns_with_frame_count": len(frames_by_pattern),
    }


def _check_summary(summary: dict[str, Any]) -> dict[str, Any]:
    _require(summary.get("status") == "ok", "summary.status 不是 ok")
    _require(summary.get("error") in (None, ""), "summary.error 非空")
    _require(
        _is_number(summary.get("n_frames")),
        "summary.n_frames 缺失或不是数值",
    )
    _require(
        int(float(summary["n_frames"])) == EXPECTED_TOTAL_FRAMES,
        f"summary.n_frames={summary['n_frames']}，预注册要求 {EXPECTED_TOTAL_FRAMES}",
    )
    _require(
        _is_number(summary.get("saturation_count")),
        "summary.saturation_count 缺失或不是数值",
    )
    _require(
        int(float(summary["saturation_count"])) == 0,
        f"summary.saturation_count={summary['saturation_count']}，必须为 0",
    )
    return {
        "status": summary.get("status"),
        "n_frames": summary.get("n_frames"),
        "saturation_count": summary.get("saturation_count", 0),
        "exposure_ms": summary.get("exposure_ms"),
    }


def _check_release(released: dict[str, Any], label: str) -> dict[str, Any]:
    _require(released.get("status") == "ok", f"{label} released.status 不是 ok")
    _require(
        released.get("closed_normally") is True,
        f"{label} 未正常关闭（closed_normally != true）",
    )
    for key in ("slm_closed", "camera_closed", "lock_released"):
        _require(released.get(key) is True, f"{label} released.{key} 不是 true")
    _require(released.get("error") in (None, ""), f"{label} released.error 非空")
    return {
        "status": released.get("status"),
        "closed_normally": released.get("closed_normally"),
        "slm_closed": released.get("slm_closed"),
        "camera_closed": released.get("camera_closed"),
        "lock_released": released.get("lock_released"),
    }


def _check_supplemental_record(
    capture_dir: Path, full_scale: float
) -> dict[str, Any]:
    path = capture_dir / "optical_path_record.json"
    _require(path.is_file(), "缺少 optical_path_record.json；无法核实六步预检")
    payload = _read_json(path)
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        _require(passed, f"optical_path_record.{name}: {detail}")
        checks.append({"name": name, "pass": True, "detail": detail})

    # 采集侧记录只携带设备身份与曝光自检；六步预检 verdict 由 --probe-dir 提供。
    devices = payload.get("devices") if isinstance(payload.get("devices"), dict) else {}
    slm = devices.get("slm") if isinstance(devices.get("slm"), dict) else {}
    camera = devices.get("camera") if isinstance(devices.get("camera"), dict) else {}
    slm_model = str(slm.get("model", ""))
    add(
        "slm_model",
        "Santec" in slm_model and "SLM-200" in slm_model,
        f"slm.model={slm.get('model')!r}（预注册为 Santec SLM-200）",
    )
    add(
        "slm_number",
        _is_number(slm.get("slm_number")) and int(slm["slm_number"]) == 1,
        f"slm_number={slm.get('slm_number')!r}（预注册为 #1）",
    )
    add(
        "wavelength_nm",
        _is_number(slm.get("wavelength_nm"))
        and abs(float(slm["wavelength_nm"]) - 1064.0) <= 1e-9,
        f"wavelength_nm={slm.get('wavelength_nm')!r}（预注册为 1064 nm）",
    )
    add(
        "camera_model",
        "MER2-507" in str(camera.get("model", "")),
        f"camera.model={camera.get('model')!r}",
    )
    serial = camera.get("serial")
    add(
        "camera_serial",
        isinstance(serial, str) and bool(serial.strip()),
        f"camera.serial={serial!r}（必须记录序列号以固定设备身份）",
    )
    add(
        "camera_bit_depth",
        _is_number(camera.get("bit_depth")) and int(camera["bit_depth"]) == 12,
        f"camera.bit_depth={camera.get('bit_depth')!r}（预注册为 12 bit）",
    )
    add(
        "full_scale",
        _is_number(camera.get("full_scale"))
        and abs(float(camera["full_scale"]) - full_scale) <= 1e-9,
        f"camera.full_scale={camera.get('full_scale')!r}（plan full_scale={full_scale}）",
    )
    add(
        "exposure_used_ms",
        _is_number(camera.get("exposure_used_ms"))
        and abs(float(camera["exposure_used_ms"]) - EXPECTED_EXPOSURE_MS) <= 1e-9,
        f"camera.exposure_used_ms={camera.get('exposure_used_ms')!r}",
    )
    exposure_check = payload.get("exposure_check")
    _require(isinstance(exposure_check, dict), "缺少 optical_path_record.exposure_check")
    add(
        "exposure_saturated",
        exposure_check.get("saturated") is False,
        f"exposure_check.saturated={exposure_check.get('saturated')!r}",
    )
    add(
        "settle_timeout",
        exposure_check.get("settle_timeout") is False,
        f"exposure_check.settle_timeout={exposure_check.get('settle_timeout')!r}",
    )
    occupancy = payload.get("occupancy")
    if isinstance(occupancy, dict) and occupancy.get("closed_normally") is not None:
        add(
            "closed_normally",
            occupancy["closed_normally"] is True,
            f"occupancy.closed_normally={occupancy['closed_normally']}",
        )
    return {
        "present": True,
        "path": _rel(path),
        "evidence": "采集侧设备身份（SLM 型号/#1/波长、相机型号/序列号/位深/"
        "满量程/曝光）与曝光自检（saturated / settle_timeout）；"
        "六步预检 verdict 不在此文件中，由 precheck_probe 提供",
        "checks": checks,
        "notes": [],
    }


def _parse_usable_exposure_window(reasons: object) -> list[float]:
    """保守解析 ``verdict.reasons`` 里的可用曝光窗口（ms）；解析不出就拒绝。"""
    _require(
        isinstance(reasons, list) and bool(reasons),
        "probe verdict.reasons 缺失或为空，无法确认可用曝光窗口",
    )
    pattern = re.compile(r"可用曝光窗口\s*([0-9][0-9.,、\s]*?)\s*ms")
    values: list[float] = []
    matched = False
    for reason in reasons:
        if not isinstance(reason, str):
            continue
        match = pattern.search(reason)
        if match is None:
            continue
        matched = True
        for token in re.split(r"[,、]", match.group(1)):
            token = token.strip()
            if not token:
                continue
            _require(
                re.fullmatch(r"\d+(?:\.\d+)?", token) is not None,
                f"无法保守解析预检曝光窗口数值 {token!r}",
            )
            values.append(float(token))
    _require(
        matched and bool(values),
        "probe verdict.reasons 中找不到可解析的『可用曝光窗口 … ms』；"
        "拒绝假定曝光可用",
    )
    return values


def _check_probe(probe_dir: Path, exposure_ms: float) -> dict[str, Any]:
    """校验独立六步光路预检 run 目录（``optical_path_probe.json``）。"""
    probe_dir = Path(probe_dir)
    _require(probe_dir.is_dir(), f"六步光路预检目录不存在：{probe_dir}")
    path = probe_dir / PROBE_FILE_NAME
    _require(path.is_file(), f"六步光路预检缺少文件：{path}")
    payload = _read_json(path)
    verdict = payload.get("verdict")
    _require(isinstance(verdict, dict), "probe 缺少 verdict 对象")
    state = verdict.get("state")
    _require(state == "ok", f"六步预检 verdict.state={state!r}（要求 'ok'）")
    exit_code = verdict.get("exit_code", payload.get("exit_code"))
    _require(
        _is_number(exit_code) and int(float(exit_code)) == 0,
        f"六步预检 exit_code={exit_code!r}（要求 0）",
    )
    window = _parse_usable_exposure_window(verdict.get("reasons"))
    _require(
        any(abs(value - exposure_ms) <= 1e-9 for value in window),
        f"冻结曝光 {exposure_ms} ms 不在预检可用曝光窗口 {window} ms 内",
    )
    return {
        "dir": _rel(probe_dir),
        "file": _rel(path),
        "verdict_state": state,
        "exit_code": int(float(exit_code)),
        "usable_exposure_window_ms": window,
        "capture_exposure_ms": float(exposure_ms),
        "exposure_in_window": True,
        "evidence": "六步光路预检 verdict.state / exit_code / 可用曝光窗口",
    }


def _frame_path_matches_pattern(path: Path, pattern_id: str) -> bool:
    stem = path.stem
    escaped = re.escape(pattern_id)
    return bool(
        stem == pattern_id
        or re.fullmatch(rf"{escaped}[_-](?:frame)?\d+", stem)
        or re.fullmatch(rf"(?:frame)?\d+_{escaped}", stem)
    )


def _validate_frames(
    capture_dir: Path,
    rounds: dict[str, RoundRecord],
    summary: dict[str, Any],
    full_scale: float,
) -> tuple[dict[tuple[str, str], list[Path]], dict[str, Any]]:
    analyzer = _analyzer()
    frames_dir = capture_dir / "frames"
    _require(frames_dir.is_dir(), f"{frames_dir} 不存在")

    all_npy = sorted(
        path.resolve() for path in frames_dir.rglob("*.npy") if path.is_file()
    )
    _require(
        len(all_npy) == EXPECTED_TOTAL_FRAMES,
        f"frames/ 下找到 {len(all_npy)} 个 .npy，预注册要求 {EXPECTED_TOTAL_FRAMES}",
    )
    used: dict[Path, tuple[str, str]] = {}
    files_by_pattern: dict[tuple[str, str], list[Path]] = {}
    evidence: dict[tuple[str, str], list[FrameEvidence]] = {}
    frame_shape: dict[str, int] | None = None
    threshold = SATURATION_FRACTION * full_scale

    for round_record in rounds.values():
        for record in round_record.patterns:
            files = [
                path.resolve()
                for path in analyzer.discover_frames(frames_dir, record.pattern_id)
                if _frame_path_matches_pattern(path, record.pattern_id)
            ]
            pattern_subdir = frames_dir / record.pattern_id
            if pattern_subdir.is_dir():
                files.extend(
                    path.resolve()
                    for path in pattern_subdir.glob("*.npy")
                    if path.is_file()
                )
            files = sorted(set(files))
            _require(
                len(files) == EXPECTED_FRAMES_PER_PATTERN,
                f"{record.round_id}/{record.pattern_id} 找到 {len(files)} 帧，"
                f"要求 {EXPECTED_FRAMES_PER_PATTERN}",
            )
            _require(
                len(set(files)) == len(files),
                f"{record.round_id}/{record.pattern_id} 帧路径重复",
            )
            for path in files:
                _require(
                    path not in used,
                    f"帧 {path.name} 同时匹配 {used.get(path)} 与 "
                    f"{record.round_id}/{record.pattern_id}",
                )
                used[path] = (record.round_id, record.pattern_id)

            frame_items: list[FrameEvidence] = []
            digests: list[str] = []
            for path in files:
                raw = np.load(path)
                _require(
                    raw.ndim == 2 and raw.size > 0,
                    f"{path.name} 不是非空二维帧",
                )
                shape = {"rows": int(raw.shape[0]), "cols": int(raw.shape[1])}
                if frame_shape is None:
                    frame_shape = shape
                else:
                    _require(
                        shape == frame_shape,
                        f"{path.name} 帧 shape={shape} 与其它帧 {frame_shape} 不一致",
                    )
                digest = _sha256_array(np.asarray(raw))
                digests.append(digest)
                max_counts = float(np.asarray(raw).max())
                count = int(np.count_nonzero(np.asarray(raw) >= threshold))
                _require(
                    count < SATURATION_PLATEAU_PIXELS,
                    f"{record.round_id}/{record.pattern_id} 的 {path.name} "
                    f"有 {count} 个像素 >= {threshold:.1f}"
                    f"（{SATURATION_FRACTION} × full_scale），判定饱和",
                )
                frame_items.append(
                    FrameEvidence(
                        path=path,
                        sha256=digest,
                        max_counts=max_counts,
                        pixels_at_or_above_threshold=count,
                    )
                )
            files_by_pattern[(record.round_id, record.pattern_id)] = files
            evidence[(record.round_id, record.pattern_id)] = frame_items

    orphan = set(all_npy) - set(used)
    _require(not orphan, f"frames/ 有未归属帧：{sorted(path.name for path in orphan)}")

    summary_pattern = summary.get("per_pattern")
    if isinstance(summary_pattern, dict):
        for (round_id, pattern_id), items in evidence.items():
            entry = summary_pattern.get(pattern_id)
            _require(
                isinstance(entry, dict),
                f"summary.per_pattern 缺少 {round_id}/{pattern_id}",
            )
            digests = entry.get("digests")
            if isinstance(digests, list):
                _require(
                    len(digests) == EXPECTED_FRAMES_PER_PATTERN,
                    f"summary.{pattern_id}.digests 长度不是 4",
                )
                _require(
                    list(digests) == [item.sha256 for item in items],
                    f"summary.{pattern_id}.digests 与实际帧不一致",
                )
        all_summary_digests = summary.get("frame_digests")
        if isinstance(all_summary_digests, list):
            actual = {
                item.sha256
                for items in evidence.values()
                for item in items
            }
            _require(
                set(all_summary_digests) == actual,
                "summary.frame_digests 与 frames/ 实际摘要集合不一致",
            )
    else:
        _require(
            summary.get("per_pattern") is None,
            "summary.per_pattern 不是对象",
        )

    return files_by_pattern, {
        "n_files": len(all_npy),
        "n_patterns": len(files_by_pattern),
        "n_frames": sum(len(items) for items in evidence.values()),
        "saturation_threshold_counts": threshold,
        "summary_per_pattern_verified": isinstance(summary_pattern, dict),
        "frame_shape": frame_shape,
    }


def _round_plan(plan: dict[str, Any], round_record: RoundRecord) -> dict[str, Any]:
    patterns: list[dict[str, Any]] = []
    for record in round_record.patterns:
        entry = dict(record.entry)
        entry["id"] = record.base_id
        patterns.append(entry)
    round_plan = dict(plan)
    round_plan["patterns"] = patterns
    return round_plan


def _build_round_capture(
    capture_dir: Path,
    round_plan: dict[str, Any],
    round_record: RoundRecord,
    files_by_pattern: dict[tuple[str, str], list[Path]],
    summary: dict[str, Any],
    full_scale: float,
) -> Any:
    analyzer = _analyzer()
    specs = analyzer.build_specs(round_plan)
    by_base = {record.base_id: record for record in round_record.patterns}
    threshold = SATURATION_FRACTION * full_scale
    means: dict[str, np.ndarray] = {}
    frame_stats: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for spec in specs:
        record = by_base[spec.pattern_id]
        files = files_by_pattern[(round_record.round_id, record.pattern_id)]
        accumulator: np.ndarray | None = None
        maxima: list[float] = []
        counts: list[int] = []
        for path in files:
            raw = np.load(path).astype(np.float64)
            accumulator = raw if accumulator is None else accumulator + raw
            maxima.append(float(raw.max()))
            counts.append(int(np.count_nonzero(raw >= threshold)))
        if accumulator is None:
            missing.append(spec.pattern_id)
            continue
        means[spec.pattern_id] = accumulator / float(len(files))
        frame_stats[spec.pattern_id] = {
            "n_frames": len(files),
            "frames": [path.name for path in files],
            "frame_max_counts": maxima,
            "max_counts": max(maxima),
            "pixels_at_or_above_threshold": max(counts),
            "saturated_frame_indices": [],
            "saturated": False,
            "saturation_reliable": True,
        }
    _require(not missing, f"{round_record.round_id} 缺少图样帧：{missing}")
    return analyzer.Capture(
        plan=round_plan,
        catalog=specs,
        means=means,
        frame_stats=frame_stats,
        full_scale=float(full_scale),
        assumed_full_scale=False,
        full_scale_source="declared",
        summary=summary,
        missing=[],
    )


def _peak_centroid(
    zero_order: object, principal: object
) -> dict[str, float] | None:
    if not isinstance(zero_order, dict) or not isinstance(principal, dict):
        return None
    if not all(_is_number(principal.get(key)) for key in ("dr", "dc")):
        return None
    if not all(_is_number(zero_order.get(key)) for key in ("row", "col")):
        return None
    return {
        "row": float(zero_order["row"]) + float(principal["dr"]),
        "col": float(zero_order["col"]) + float(principal["dc"]),
    }


def _edge_clip(
    centroid: dict[str, float] | None,
    frame_shape: dict[str, int] | None,
    *,
    peak: object = None,
    background: object = None,
) -> dict[str, Any] | None:
    """principal 峰心是否落在帧边界 8 px 带内（被截断的边界极大值）。

    只报事实：峰心绝对坐标、到最近边界的距离、峰/背景计数。返回 ``None`` 表示
    不在边界带内或无法判定（缺峰心/缺帧 shape），不改变任何门禁。
    """
    if centroid is None or frame_shape is None:
        return None
    rows = int(frame_shape["rows"])
    cols = int(frame_shape["cols"])
    row = float(centroid["row"])
    col = float(centroid["col"])
    distance = min(row, rows - 1 - row, col, cols - 1 - col)
    if distance >= EDGE_BAND_PX:
        return None
    peak_text = "n/a" if not _is_number(peak) else f"{float(peak):.1f}"
    background_text = (
        "n/a" if not _is_number(background) else f"{float(background):.1f}"
    )
    return {
        "row": row,
        "col": col,
        "frame_rows": rows,
        "frame_cols": cols,
        "distance_to_edge_px": float(distance),
        "peak_counts": None if not _is_number(peak) else float(peak),
        "background_counts": None if not _is_number(background) else float(background),
        "reason": (
            "principal peak on frame edge (clipped border maximum, "
            f"peak={peak_text} counts / background={background_text} counts, "
            f"row={row:.1f}, col={col:.1f})"
        ),
    }


def _frame_status(record: dict[str, Any]) -> tuple[str | None, float | None]:
    angle = record.get("axis_angle_deg")
    if angle is not None:
        return "assessed", float(angle)
    if record.get("saturated"):
        return "saturated", None
    if int(record.get("peak_count") or 0) == 0:
        return "missing_peak", None
    return "no_axis_angle", None


def _frame_metrics(
    round_record: RoundRecord,
    round_plan: dict[str, Any],
    analysis: dict[str, Any],
    files_by_pattern: dict[tuple[str, str], list[Path]],
    full_scale: float,
) -> dict[str, Any]:
    analyzer = _analyzer()
    specs = {spec.pattern_id: spec for spec in analyzer.build_specs(round_plan)}
    sigma = float(analysis["flat_reference"]["sigma_est_px"])
    threshold = SATURATION_FRACTION * full_scale
    result: dict[str, Any] = {}
    for record in round_record.patterns:
        mean_record = analysis["patterns"][record.base_id]
        files = files_by_pattern[(round_record.round_id, record.pattern_id)]
        item: dict[str, Any] = {
            "base_id": record.base_id,
            "round_pattern_id": record.pattern_id,
            "family": mean_record.get("family"),
            "order_index": record.order_index,
            "zero_order": mean_record.get("zero_order"),
            "mean": {
                "axis_angle_deg": mean_record.get("axis_angle_deg"),
                "zero_order": mean_record.get("zero_order"),
                "principal_peak_centroid": _peak_centroid(
                    mean_record.get("zero_order"), mean_record.get("principal")
                ),
                "principal_angle_deg": (
                    mean_record.get("principal") or {}
                ).get("angle_deg"),
                "peak_count": mean_record.get("peak_count"),
                "peaks": mean_record.get("peaks"),
                "principal": mean_record.get("principal"),
                "multi_peak": mean_record.get("multi_peak"),
                "counter_lobe_missing": mean_record.get("counter_lobe_missing"),
                "counter_lobe_weak": mean_record.get("counter_lobe_weak"),
                "background_counts": mean_record.get("background_counts"),
                "max_counts": mean_record.get("max_counts"),
            },
            "frames": [],
        }
        if record.base_id in FLAT_BASE_IDS:
            item["frames"] = _flat_frame_metrics(files, mean_record)
            item["status"] = "flat_reference"
            result[record.base_id] = item
            continue

        spec = specs[record.base_id]
        reference = (
            float(mean_record["zero_order"]["row"]),
            float(mean_record["zero_order"]["col"]),
        )
        frame_angles: list[float | None] = []
        for index, path in enumerate(files):
            raw = np.load(path).astype(np.float64)
            count = int(np.count_nonzero(raw >= threshold))
            stats = {
                "n_frames": 1,
                "saturated": count >= SATURATION_PLATEAU_PIXELS,
                "max_counts": float(raw.max()),
            }
            measured = analyzer._measure_pattern(spec, raw, reference, sigma, stats)
            status, angle = _frame_status(measured)
            frame_angles.append(angle)
            principal = measured.get("principal")
            peak = (principal or {}).get("peak")
            background = measured.get("background_counts")
            ratio = (
                float(peak) / float(background)
                if _is_number(peak) and _is_number(background) and float(background) > 0
                else None
            )
            item["frames"].append(
                {
                    "index": index,
                    "file": path.name,
                    "status": status,
                    "axis_angle_deg": angle,
                    "zero_order": {"row": reference[0], "col": reference[1]},
                    "principal_peak_centroid": _peak_centroid(
                        {"row": reference[0], "col": reference[1]}, principal
                    ),
                    "principal_angle_deg": (
                        principal or {}
                    ).get("angle_deg"),
                    "peak_count": measured.get("peak_count"),
                    "peaks": measured.get("peaks"),
                    "principal": principal,
                    "lobes": measured.get("lobes"),
                    "multi_peak": measured.get("multi_peak"),
                    "counter_lobe_missing": measured.get(
                        "counter_lobe_missing"
                    ),
                    "counter_lobe_weak": measured.get("counter_lobe_weak"),
                    "max_counts": float(raw.max()),
                    "background_counts": background,
                    "peak_to_background_ratio": ratio,
                    "pixels_at_or_above_threshold": count,
                }
            )
        assessable = [angle for angle in frame_angles if angle is not None]
        item["frame_angles_deg"] = frame_angles
        item["frame_spread_deg"] = _axial_spread(assessable)
        item["missing_frames"] = [
            index for index, angle in enumerate(frame_angles) if angle is None
        ]
        result[record.base_id] = item
    return result


def _flat_frame_metrics(
    files: list[Path], mean_record: dict[str, Any]
) -> list[dict[str, Any]]:
    analyzer = _analyzer()
    frames: list[dict[str, Any]] = []
    for index, path in enumerate(files):
        raw = np.load(path).astype(np.float64)
        coarse = analyzer._argmax_center(raw)
        sigma = analyzer._core_sigma(raw, coarse)
        inner = analyzer._ring_inner_radius(raw.shape, sigma)
        background, _, _ = analyzer._ring_background(raw, coarse, inner)
        centroid = analyzer._flat_centroid(
            np.maximum(raw - background, 0.0), coarse
        )
        frames.append(
            {
                "index": index,
                "file": path.name,
                "coarse_max": {"row": coarse[0], "col": coarse[1]},
                "centroid": {"row": centroid[0], "col": centroid[1]},
                "sigma_est_px": sigma,
                "background_counts": background,
                "max_counts": float(raw.max()),
            }
        )
    return frames


def _axial_spread(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    analyzer = _analyzer()
    return max(
        float(analyzer.axial_diff_deg(first, second))
        for index, first in enumerate(values)
        for second in values[index + 1 :]
    )


def _pairwise_diff(values: dict[str, float]) -> dict[str, float]:
    analyzer = _analyzer()
    keys = sorted(values)
    result: dict[str, float] = {}
    for index, first in enumerate(keys):
        for second in keys[index + 1 :]:
            diff = analyzer.axial_diff_deg(values[first], values[second])
            result[f"{first}-{second}"] = None if diff is None else float(diff)
    return result


def _round_analysis_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": analysis.get("verdict"),
        "failed_gates": analysis.get("failed_gates"),
        "gates": analysis.get("gates"),
        "attribution_gates": analysis.get("attribution_gates"),
        "held_out_validation": analysis.get("held_out_validation"),
        "flat_reference": analysis.get("flat_reference"),
        "notes": analysis.get("notes"),
        "patterns": analysis.get("patterns"),
    }


def _plan_pattern_roles(rounds: dict[str, RoundRecord]) -> dict[str, str | None]:
    """按 ``base_pattern_id`` 汇总 plan 条目自带的 ``role``（仅展示，不参与门禁）。

    三轮取值一致时采用该值；plan 未给出 ``role`` 时保持 ``None``；三轮互相矛盾时
    也返回 ``None``，不替数据挑一个。
    """
    roles: dict[str, str | None] = {}
    for base_id in EXPECTED_BASE_IDS:
        values = {
            str(record.entry["role"])
            for round_id in ROUND_IDS
            for record in rounds[round_id].patterns
            if record.base_id == base_id and record.entry.get("role") is not None
        }
        roles[base_id] = next(iter(values)) if len(values) == 1 else None
    return roles


def _repeatability(
    round_analyses: dict[str, dict[str, Any]],
    frame_metrics: dict[str, dict[str, Any]],
    baseline_angles: dict[str, float | None] | None,
    plan_roles: dict[str, str | None] | None = None,
    frame_shape: dict[str, int] | None = None,
) -> dict[str, Any]:
    analyzer = _analyzer()
    patterns: list[dict[str, Any]] = []
    counts = {
        "reproduced": 0,
        "not_reproduced": 0,
        "not_assessable": 0,
        "edge_clipped_rounds": 0,
        "assessable_rounds_total": 0,
        "insufficient_evidence": 0,
    }
    for base_id in EVALUABLE_BASE_IDS:
        per_round: list[dict[str, Any]] = []
        angles: dict[str, float] = {}
        missing: dict[str, str] = {}
        for round_id in ROUND_IDS:
            record = round_analyses[round_id]["patterns"][base_id]
            angle = record.get("axis_angle_deg")
            centroid = _peak_centroid(
                record.get("zero_order"), record.get("principal")
            )
            edge = _edge_clip(
                centroid,
                frame_shape,
                peak=(record.get("principal") or {}).get("peak"),
                background=record.get("background_counts"),
            )
            if angle is not None and edge is not None:
                # 预注册：缺峰保留为缺测，禁止调阈值救回；边界极大值不是波瓣。
                # 原始轴角保留在记录里（审计用），但该轮不进重复性门。
                status = "not_assessable"
                value = float(angle)
                missing[round_id] = edge["reason"]
                counts["edge_clipped_rounds"] += 1
            elif angle is None:
                reason = (
                    "saturated"
                    if record.get("saturated")
                    else (
                        "missing_peak"
                        if int(record.get("peak_count") or 0) == 0
                        else "no_axis_angle"
                    )
                )
                missing[round_id] = reason
                status = "not_assessable"
                value = None
            else:
                value = float(angle)
                angles[round_id] = value
                status = "assessed"
            frame_item = frame_metrics[round_id].get(base_id, {})
            per_round.append(
                {
                    "round": round_id,
                    "status": status,
                    "axis_angle_deg": value,
                    "gate_axis_angle_deg": (
                        float(angle) if round_id in angles else None
                    ),
                    "principal_angle_deg": (
                        record.get("principal") or {}
                    ).get("angle_deg"),
                    "principal": record.get("principal"),
                    "zero_order": record.get("zero_order"),
                    "principal_peak_centroid": centroid,
                    "frame_shape": frame_shape,
                    "edge_clipped": edge,
                    "core_exclude_px": record.get("core_exclude_px"),
                    "lobe_radius_px": record.get("lobe_radius_px"),
                    "frame_angles_deg": frame_item.get("frame_angles_deg"),
                    "frame_spread_deg": frame_item.get("frame_spread_deg"),
                    "missing_frames": frame_item.get("missing_frames"),
                    "peak_count": record.get("peak_count"),
                    "background_counts": record.get("background_counts"),
                    "max_counts": record.get("max_counts"),
                    "multi_peak": record.get("multi_peak"),
                    "counter_lobe_missing": record.get(
                        "counter_lobe_missing"
                    ),
                    "counter_lobe_weak": record.get("counter_lobe_weak"),
                }
            )

        pairwise = _pairwise_diff(angles)
        max_diff = max(pairwise.values()) if pairwise else None
        counts["assessable_rounds_total"] += len(angles)
        hard_missing = [
            row["round"]
            for row in per_round
            if row["status"] == "not_assessable" and row["edge_clipped"] is None
        ]
        if len(angles) < 2:
            status = "not_assessable"
            if hard_missing:
                counts["insufficient_evidence"] += 1
                reason = (
                    "fewer_than_two_assessable_rounds"
                    if len(angles) == 1
                    else "no_assessable_rounds"
                )
            else:
                # 全部轮次只被边界带排除：视为缺测，不计入重复性门。
                reason = "edge_clipped_rounds_excluded"
        elif max_diff is not None and max_diff <= REPEATABILITY_MAX_DEG:
            status = "reproduced"
            reason = "round_axis_diff_le_2deg"
        else:
            status = "not_reproduced"
            reason = "round_axis_diff_gt_2deg"
        counts[status] += 1

        baseline_item: dict[str, Any] | None = None
        if baseline_angles is not None:
            baseline_angle = baseline_angles.get(base_id)
            diffs = {
                round_id: (
                    None
                    if baseline_angle is None
                    else analyzer.axial_diff_deg(angle, float(baseline_angle))
                )
                for round_id, angle in angles.items()
            }
            baseline_item = {
                "axis_angle_deg": baseline_angle,
                "round_vs_baseline_diff_deg": {
                    key: None if value is None else float(value)
                    for key, value in diffs.items()
                },
            }

        patterns.append(
            {
                "base_id": base_id,
                "family": (
                    round_analyses["r1"]["patterns"][base_id].get("family")
                ),
                "axis": round_analyses["r1"]["patterns"][base_id].get("axis"),
                "role": None if plan_roles is None else plan_roles.get(base_id),
                "gate_basis": "axis_angle_deg (mod 180)",
                "per_round": per_round,
                "assessable_rounds": sorted(angles),
                "missing_rounds": missing,
                "edge_clipped_rounds": sorted(
                    row["round"]
                    for row in per_round
                    if row.get("edge_clipped") is not None
                ),
                "pairwise_round_diff_deg": pairwise,
                "round_max_diff_deg": max_diff,
                "status": status,
                "reason": reason,
                "baseline": baseline_item,
            }
        )

    overall_pass = (
        counts["reproduced"] > 0
        and counts["not_reproduced"] == 0
        and counts["insufficient_evidence"] == 0
    )
    return {
        "threshold_deg": REPEATABILITY_MAX_DEG,
        "axis_metric": "axis_angle_deg",
        "gate_rule": (
            "每 pattern 用 ≥2 个可评估轮次的 axis_angle_deg 作两两差，最大差 ≤2° 即 "
            "reproduced；principal 峰心落在帧边界 8 px 带内的轮次视为缺测，"
            "不计入可评估轮次；因硬缺测导致可评估轮次不足 2 的 pattern 视为证据不足，"
            "门不通过。"
        ),
        "patterns": patterns,
        "counts": counts,
        "n_evaluable_patterns": len(EVALUABLE_BASE_IDS),
        "overall_pass": overall_pass,
    }


def _check_thresholds(plan: dict[str, Any]) -> dict[str, float]:
    analyzer = _analyzer()
    _require(
        not plan.get("analysis_thresholds"),
        "plan.analysis_thresholds 非空：预注册禁止事后调检测阈值",
    )
    _require(
        not plan.get("waveform_probe_thresholds"),
        "plan.waveform_probe_thresholds 非空：预注册禁止事后调检测阈值",
    )
    fraction = plan.get("saturation_fraction")
    if _is_number(fraction):
        _require(
            abs(float(fraction) - SATURATION_FRACTION) <= 1e-9,
            f"plan.saturation_fraction={fraction}，冻结值为 {SATURATION_FRACTION}",
        )
    thresholds = dict(analyzer.DEFAULT_THRESHOLDS)
    _require(
        abs(thresholds["saturation_fraction"] - SATURATION_FRACTION) <= 1e-9,
        "分析器默认 saturation_fraction 与预注册不一致",
    )
    _require(
        abs(thresholds["drift_max_px"] - DRIFT_MAX_PX) <= 1e-9,
        "分析器默认 drift_max_px 与预注册不一致",
    )
    return thresholds


def _baseline(baseline_dir: Path) -> dict[str, Any]:
    for name in (
        "plan.json",
        "summary.json",
        "released.json",
        "waveform_probe_analysis.json",
    ):
        _require(
            (baseline_dir / name).is_file(),
            f"基线目录缺少 {name}；预注册要求基线分析产物存在",
        )
    plan = _read_json(baseline_dir / "plan.json")
    summary = _read_json(baseline_dir / "summary.json")
    released = _read_json(baseline_dir / "released.json")
    analysis = _read_json(baseline_dir / "waveform_probe_analysis.json")

    release = _check_release(released, "baseline")
    _require(summary.get("status") == "ok", "baseline summary.status 不是 ok")
    if _is_number(summary.get("n_frames")):
        _require(
            int(float(summary["n_frames"])) == 92,
            f"baseline summary.n_frames={summary['n_frames']}，应为 92",
        )
    if _is_number(summary.get("saturation_count")):
        _require(
            int(float(summary["saturation_count"])) == 0,
            "baseline summary.saturation_count 非 0",
        )

    patterns = analysis.get("patterns")
    _require(isinstance(patterns, dict), "baseline 分析缺少 patterns 对象")
    _require(
        set(patterns) == set(EXPECTED_BASE_IDS),
        "baseline 分析的 23 个 pattern 与冻结集合不一致",
    )
    gates = analysis.get("gates")
    _require(isinstance(gates, dict), "baseline 分析缺少 gates")
    for key in ("drift_gate", "saturation_gate"):
        _require(
            isinstance(gates.get(key), dict)
            and gates[key].get("pass") is True,
            f"baseline {key} 未通过，基线数据无效",
        )
    attribution = analysis.get("attribution_gates")
    _require(
        isinstance(attribution, dict),
        "baseline 分析缺少 attribution_gates（G1–G4）",
    )
    for key in GATE_KEYS:
        _require(key in attribution, f"baseline attribution_gates 缺少 {key}")
    _require(
        isinstance(analysis.get("held_out_validation"), dict),
        "baseline 分析缺少 held_out_validation",
    )
    analyzer = _analyzer()
    _require(
        analysis.get("verdict") in analyzer.VERDICTS,
        f"baseline verdict={analysis.get('verdict')!r} 非法",
    )

    baseline_patterns = plan.get("patterns")
    _require(
        isinstance(baseline_patterns, list)
        and len(baseline_patterns) == EXPECTED_PATTERNS_PER_ROUND,
        "baseline plan.patterns 不是 23 项",
    )
    order = [str(item.get("id")) for item in baseline_patterns]
    _require(
        set(order) == set(EXPECTED_BASE_IDS),
        "baseline plan 的 pattern 集合与冻结集合不一致",
    )
    phase = {
        str(item.get("id")): item.get("phase_sha256")
        for item in baseline_patterns
        if isinstance(item, dict)
    }
    checked = sum(
        isinstance(phase.get(base_id), str) and bool(phase[base_id].strip())
        for base_id in EXPECTED_BASE_IDS
    )
    baseline_angles = {
        base_id: (
            None
            if patterns[base_id].get("axis_angle_deg") is None
            else float(patterns[base_id]["axis_angle_deg"])
        )
        for base_id in EVALUABLE_BASE_IDS
    }
    return {
        "supplied": True,
        "dir": _rel(baseline_dir),
        "pattern_order": order,
        "phase_sha256": phase,
        "phase_checked": checked,
        "release": release,
        "summary": {
            "status": summary.get("status"),
            "n_frames": summary.get("n_frames"),
            "saturation_count": summary.get("saturation_count", 0),
        },
        "verdict": analysis.get("verdict"),
        "gates": {
            "drift": gates.get("drift_gate"),
            "saturation": gates.get("saturation_gate"),
        },
        "attribution_gates": attribution,
        "held_out_validation": analysis.get("held_out_validation"),
        "angles_deg": baseline_angles,
        "principal_facts": {
            base_id: (patterns[base_id].get("principal") or {})
            for base_id in EXPECTED_BASE_IDS
            if isinstance(patterns.get(base_id), dict)
        },
    }


def check_capture(
    capture_dir: Path,
    baseline_dir: Path | None = None,
    probe_dir: Path | None = None,
    *,
    env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """校验一个 3 轮采集目录并返回可序列化结果；不合格直接抛 CheckError。"""
    del env  # 预留注入点；当前检查不依赖进程环境。
    capture_dir = Path(capture_dir)
    _require(capture_dir.is_dir(), f"采集目录不存在：{capture_dir}")
    _require(
        probe_dir is not None,
        "完整门禁运行需要 --probe-dir：指向六步光路预检 run 目录，"
        "采集目录自身的 optical_path_record.json 不含预检 verdict",
    )
    plan = _read_json(capture_dir / "plan.json")
    summary = _read_json(capture_dir / "summary.json")
    released = _read_json(capture_dir / "released.json")
    analyzer = _analyzer()

    baseline = _baseline(Path(baseline_dir)) if baseline_dir is not None else None
    rounds = _extract_rounds(plan)
    order = _check_order(plan, rounds, baseline)
    phase = _check_phase_hashes(rounds, baseline)
    fixed = _check_fixed_parameters(plan, summary, rounds)
    summary_info = _check_summary(summary)
    release = _check_release(released, "capture")

    full_scale, full_scale_source = analyzer.resolve_full_scale(plan, summary)
    _require(
        full_scale is not None,
        "缺少满量程（full_scale/full_scale_required），无法执行 0.9×full_scale 饱和门",
    )
    _require(
        full_scale_source != "",
        "满量程来源为空，无法确认饱和判据",
    )
    thresholds = _check_thresholds(plan)
    files_by_pattern, frames_info = _validate_frames(
        capture_dir, rounds, summary, float(full_scale)
    )
    supplemental = _check_supplemental_record(capture_dir, float(full_scale))
    probe = _check_probe(Path(probe_dir), EXPECTED_EXPOSURE_MS)

    round_analyses: dict[str, dict[str, Any]] = {}
    frame_metrics: dict[str, dict[str, Any]] = {}
    for round_id in ROUND_IDS:
        round_record = rounds[round_id]
        round_plan = _round_plan(plan, round_record)
        try:
            capture = _build_round_capture(
                capture_dir,
                round_plan,
                round_record,
                files_by_pattern,
                summary,
                float(full_scale),
            )
            analysis = analyzer.analyze(capture, thresholds)
        except CheckError:
            raise
        except Exception as exc:
            raise CheckError(f"{round_id} 分析失败：{exc}") from exc
        _require(
            analysis.get("gates", {}).get("saturation_gate", {}).get("pass")
            is True,
            f"{round_id} saturation_gate 未通过",
        )
        _require(
            analysis.get("gates", {}).get("drift_gate", {}).get("pass")
            is True,
            f"{round_id} drift_gate 未通过",
        )
        attribution = analysis.get("attribution_gates")
        _require(isinstance(attribution, dict), f"{round_id} 缺少 attribution_gates")
        for gate_key in GATE_KEYS:
            _require(
                isinstance(attribution.get(gate_key), dict)
                and isinstance(attribution[gate_key].get("pass"), bool),
                f"{round_id} 缺少有效 {gate_key}",
            )
        round_analyses[round_id] = _round_analysis_summary(analysis)
        frame_metrics[round_id] = _frame_metrics(
            round_record,
            round_plan,
            analysis,
            files_by_pattern,
            float(full_scale),
        )

    flat_global = _flat_global_drift(round_analyses)
    _require(
        flat_global["max_drift_px"] <= DRIFT_MAX_PX,
        f"跨轮 flat 质心最大漂移 {flat_global['max_drift_px']:.3f} px"
        f" > {DRIFT_MAX_PX} px",
    )

    baseline_angles = None if baseline is None else baseline["angles_deg"]
    repeatability = _repeatability(
        round_analyses,
        frame_metrics,
        baseline_angles,
        _plan_pattern_roles(rounds),
        frames_info.get("frame_shape"),
    )
    baseline_out = (
        {"supplied": False}
        if baseline is None
        else {
            "supplied": True,
            "dir": baseline["dir"],
            "phase_checked": baseline["phase_checked"],
            "summary": baseline["summary"],
            "release": baseline["release"],
            "verdict": baseline["verdict"],
            "gates": baseline["gates"],
            "attribution_gates": baseline["attribution_gates"],
            "held_out_validation": baseline["held_out_validation"],
            "principal_facts": baseline["principal_facts"],
        }
    )

    round_rows = [
        {
            "round": round_id,
            "order": rounds[round_id].base_ids,
            "n_patterns": len(rounds[round_id].patterns),
            "n_frames": sum(
                len(files_by_pattern[(round_id, item.pattern_id)])
                for item in rounds[round_id].patterns
            ),
            "analysis": round_analyses[round_id],
        }
        for round_id in ROUND_IDS
    ]

    limitations: list[str] = []
    if baseline is None:
        limitations.append(
            "未提供 --baseline-dir；冻结顺序只能校验 r2 反序、r3=r1，"
            "不能对照 attribution 原始顺序与相位摘要。"
        )
    if not supplemental["present"]:
        limitations.append("采集目录缺少 optical_path_record.json；设备/曝光预检未复核。")
    if not frames_info["summary_per_pattern_verified"]:
        limitations.append("summary.per_pattern 缺失；只按实际帧数/摘要自检。")
    limitations.append(
        "本检查只判断同一硬件打开周期内的重复性；"
        "不证明根因、不冻结 panel→CCD 映射、不代表跨会话独立复现。"
    )

    payload = {
        "schema": "issue247-reproducibility-check/v1",
        "script": _rel(Path(__file__)),
        "analyzer": _rel(ANALYZER_PATH),
        "prereg": "docs/dev/hw_window_20261124_issue247_reproducibility_prereg.md",
        "generated_at_utc": _utc_now(),
        "capture_dir": _rel(capture_dir),
        "baseline_dir": None if baseline_dir is None else _rel(Path(baseline_dir)),
        "probe_dir": _rel(Path(probe_dir)),
        "evidence_sources": {
            "capture_plan": "plan.json：轮次/顺序/相位摘要/帧数/"
            "capture_settings.settle_s/exposure_ms",
            "capture_summary": "summary.json：状态/总帧数/饱和计数/逐图样帧数",
            "capture_released": "released.json：SLM/相机/锁正常释放",
            "capture_optical_record": "optical_path_record.json：设备身份与"
            "exposure_check（六步预检 verdict 不在其中）",
            "probe_run": f"{PROBE_FILE_NAME}：六步预检 verdict.state、"
            "exit_code、可用曝光窗口",
            "baseline": "waveform_probe_analysis.json / plan.json / summary.json / "
            "released.json：attribution 基线",
        },
        "invariants": {
            "rounds": list(ROUND_IDS),
            "rounds_count": len(rounds),
            "patterns_per_round": EXPECTED_PATTERNS_PER_ROUND,
            "unique_pattern_ids": 69,
            "total_frames": frames_info["n_frames"],
            "order": order,
            "phase_hashes": phase,
            "fixed_parameters": fixed,
            "summary": summary_info,
            "release": release,
            "full_scale": float(full_scale),
            "full_scale_source": full_scale_source,
            "saturation_threshold_counts": frames_info[
                "saturation_threshold_counts"
            ],
            "frame_shape": frames_info["frame_shape"],
            "flat_global_drift": flat_global,
            "supplemental_optical_record": supplemental,
            "precheck_probe": probe,
            "thresholds": thresholds,
            "all_pass": True,
        },
        "rounds": round_rows,
        "repeatability": repeatability,
        "baseline": baseline_out,
        "limitations": limitations,
        "conclusion_boundary": [
            "重复性门通过只能说该异常在本次连续硬件会话内可重复；"
            "不等同于跨会话独立复现。",
            "若未通过，结论为本窗口信号不稳定/证据不足，不等同于 issue 无效。",
            "panel-y 轴路由、实际 LUT/相位写回、匹配支持域及未见方向验证"
            "仍为未解决证据缺口。",
        ],
    }
    return _json_safe(payload)


def _flat_global_drift(round_analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    points: list[tuple[str, float, float]] = []
    for round_id in ROUND_IDS:
        flat = round_analyses[round_id].get("flat_reference") or {}
        centroids = flat.get("centroids") or {}
        for base_id, point in centroids.items():
            if isinstance(point, dict):
                points.append(
                    (
                        f"{round_id}/{base_id}",
                        float(point["row"]),
                        float(point["col"]),
                    )
                )
    pairwise: dict[str, float] = {}
    for index, (name_a, row_a, col_a) in enumerate(points):
        for name_b, row_b, col_b in points[index + 1 :]:
            pairwise[f"{name_a}->{name_b}"] = float(
                math.hypot(row_b - row_a, col_b - col_a)
            )
    max_drift = max(pairwise.values()) if pairwise else 0.0
    return {
        "max_drift_px": max_drift,
        "pairwise_px": pairwise,
        "n_points": len(points),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating,)):
        return _json_safe(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _fmt_cell(value: Any, digits: int = 3) -> str:
    return fmt_general(value, f".{digits}f", missing_none="—", bool_labels=("pass", "fail"))


def _gate_cell(gate: dict[str, Any]) -> str:
    status = _fmt_cell(gate.get("pass"))
    details: list[str] = []
    if "measured_spread_deg" in gate:
        details.append(f"spread={_fmt_cell(gate.get('measured_spread_deg'), 2)}°")
    axes = gate.get("axes")
    if isinstance(axes, dict):
        for axis in ("x", "y"):
            item = axes.get(axis) or {}
            if "measured_spread_deg" in item:
                details.append(
                    f"{axis}: {_fmt_cell(item.get('measured_spread_deg'), 2)}°"
                )
    if "anomalous_family" in gate:
        details.append(f"anomalous={gate.get('anomalous_family') or 'none'}")
    families = gate.get("families")
    if isinstance(families, dict):
        for name, item in families.items():
            if isinstance(item, dict) and "measured_separation_deg" in item:
                details.append(
                    f"{name}: {_fmt_cell(item.get('measured_separation_deg'), 2)}°"
                )
    if gate.get("axis_routing_failed_families"):
        details.append(
            "routing_failed=" + ",".join(gate["axis_routing_failed_families"])
        )
    return status + (" (" + "; ".join(details) + ")" if details else "")


def _short_missing(reason: object) -> str:
    text = str(reason)
    if text.startswith("principal peak on frame edge"):
        return "edge_clipped"
    return text


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Issue #247 同窗口重复性复核（离线检查）")
    lines.append("")
    block = provenance_block_for(REPORT_KEY)
    if block:
        lines.append(block)
        lines.append("")
    lines.append(f"- **数据**: `{payload['capture_dir']}`")
    if payload.get("baseline_dir"):
        lines.append(f"- **基线**: `{payload['baseline_dir']}`")
    if payload.get("probe_dir"):
        lines.append(f"- **预检 run**: `{payload['probe_dir']}`")
    lines.append(f"- **分析器**: `{payload['analyzer']}`")
    lines.append(f"- **预注册**: `{payload['prereg']}`")
    lines.append(f"- **生成时间**: {payload['generated_at_utc']}")
    lines.append("")

    rep = payload["repeatability"]
    counts = rep["counts"]
    overall = "pass" if rep["overall_pass"] else "fail"
    lines.append("## 结论先行")
    lines.append("")
    lines.append(
        f"- 输入冻结项：全部通过（3 轮 × 23 图样 × 4 帧 = "
        f"{payload['invariants']['total_frames']} 帧）。"
    )
    lines.append(
        f"- 方向重复性门（任意两轮轴角差 ≤ {rep['threshold_deg']}°，"
        f"只用 `axis_angle_deg`）：**{overall}**；"
        f"reproduced={counts['reproduced']}，"
        f"not_reproduced={counts['not_reproduced']}，"
        f"not_assessable={counts['not_assessable']}"
        f"（其中边界极大值排除 {counts['edge_clipped_rounds']} 轮）；"
        f"可评估轮次合计 {counts['assessable_rounds_total']}。"
    )
    gate_bits: list[str] = []
    for round_row in payload["rounds"]:
        gates = round_row["analysis"]["attribution_gates"]
        gate_bits.append(
            f"{round_row['round']}: "
            + ", ".join(
                f"{key.split('_')[0]}={_fmt_cell(gates[key].get('pass'))}"
                for key in GATE_KEYS
            )
        )
    lines.append("- 逐轮 G1–G4：" + "；".join(gate_bits) + "。")
    lines.append(
        "- 边界：本报告只说明同一硬件打开周期内是否重复；不证明根因、"
        "不冻结映射、不代表跨会话独立复现。"
    )
    lines.append("")

    lines.append("## 1. 输入与冻结项校验")
    lines.append("")
    inv = payload["invariants"]
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 轮次 | {', '.join(inv['rounds'])} |")
    lines.append(f"| 每轮图样数 | {inv['patterns_per_round']} |")
    lines.append(f"| 唯一 pattern ID | {inv['unique_pattern_ids']} |")
    lines.append(f"| 实际帧数 | {inv['total_frames']} |")
    lines.append(f"| 曝光 | {_fmt_cell(inv['fixed_parameters']['exposure_ms'], 3)} ms |")
    lines.append(f"| settle | {_fmt_cell(inv['fixed_parameters']['settle_s'], 3)} s |")
    lines.append(
        f"| 每图样帧数 | {inv['fixed_parameters']['frames_per_pattern']} |"
    )
    lines.append(f"| full scale | {_fmt_cell(inv['full_scale'], 1)} |")
    shape = inv.get("frame_shape") or {}
    lines.append(
        f"| 帧 shape | {shape.get('rows', '—')} × {shape.get('cols', '—')} px |"
    )
    lines.append(
        f"| 饱和门 | {_fmt_cell(inv['saturation_threshold_counts'], 1)} counts |"
    )
    lines.append(
        f"| flat 跨轮最大漂移 | {_fmt_cell(inv['flat_global_drift']['max_drift_px'])} px "
        f"(≤ {DRIFT_MAX_PX}) |"
    )
    lines.append(
        f"| 冻结顺序来源 | {inv['order']['frozen_order_source']} |"
    )
    lines.append(
        f"| 相位摘要 | {inv['phase_hashes']['base_patterns']} 个 base pattern "
        "三轮一致"
        + (
            f"，基线核对 {inv['phase_hashes']['baseline_checked']} 个"
            if inv["phase_hashes"]["baseline_checked"]
            else ""
        )
        + " |"
    )
    lines.append(
        f"| 设备释放 | status={_fmt_cell(inv['release']['status'])}，"
        f"closed_normally={_fmt_cell(inv['release']['closed_normally'])} |"
    )
    supp = inv["supplemental_optical_record"]
    lines.append(
        f"| 采集光路记录 | {'已复核 ' + str(len(supp['checks'])) + ' 项' if supp['present'] else '缺失（见限制）'} |"
    )
    probe = inv["precheck_probe"]
    lines.append(
        f"| 六步预检 | verdict.state={probe['verdict_state']}，"
        f"exit_code={probe['exit_code']} |"
    )
    lines.append(
        "| 预检可用曝光窗口 | "
        + ", ".join(_fmt_cell(value, 3) for value in probe["usable_exposure_window_ms"])
        + f" ms（冻结 {_fmt_cell(probe['capture_exposure_ms'], 3)} ms 在内） |"
    )
    lines.append("")
    lines.append("**证据来源**")
    lines.append("")
    for key, text in payload["evidence_sources"].items():
        lines.append(f"- `{key}`：{text}")
    lines.append("")

    lines.append("## 2. 每轮 G1–G4（逐轮单独计算）")
    lines.append("")
    lines.append("| 门槛 | r1 | r2 | r3 |")
    lines.append("|---|---|---|---|")
    for key in GATE_KEYS:
        cells = [
            _gate_cell(
                payload["rounds"][index]["analysis"]["attribution_gates"][key]
            )
            for index in range(3)
        ]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| 轮次 | verdict | 失败 gates | held-out 轴向准确率 |")
    lines.append("|---|---|---|---|")
    for round_row in payload["rounds"]:
        analysis = round_row["analysis"]
        held = analysis.get("held_out_validation") or {}
        lines.append(
            f"| {round_row['round']} | {analysis.get('verdict')} | "
            f"{', '.join(analysis.get('failed_gates') or []) or '—'} | "
            f"{_fmt_cell(held.get('axis_accuracy'), 3)} |"
        )
    lines.append("")

    lines.append("## 3. 逐 pattern 方向重复性（axis_angle_deg，模 180°）")
    lines.append("")
    show_role = any(item.get("role") is not None for item in rep["patterns"])
    if show_role:
        lines.append(
            "| base pattern | 族 | 角色 | r1 | r2 | r3 | 帧内极差 r1/r2/r3 | "
            "轮间最大差 | 状态 | 缺测 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
    else:
        lines.append(
            "| base pattern | 族 | r1 | r2 | r3 | 帧内极差 r1/r2/r3 | "
            "轮间最大差 | 状态 | 缺测 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
    for item in rep["patterns"]:
        angles = {
            row["round"]: (
                _fmt_cell(row["axis_angle_deg"])
                + (
                    "†"
                    if row.get("edge_clipped") is not None
                    and row["axis_angle_deg"] is not None
                    else ""
                )
            )
            for row in item["per_round"]
        }
        spreads = {
            row["round"]: row.get("frame_spread_deg")
            for row in item["per_round"]
        }
        missing = ", ".join(
            f"{key}:{_short_missing(value)}"
            for key, value in item["missing_rounds"].items()
        )
        role_cell = f" {item.get('role') or 'null'} |" if show_role else ""
        lines.append(
            f"| {item['base_id']} | {item['family']} |{role_cell} "
            f"{_fmt_cell(angles.get('r1'))} | {_fmt_cell(angles.get('r2'))} | "
            f"{_fmt_cell(angles.get('r3'))} | "
            f"{_fmt_cell(spreads.get('r1'), 2)} / {_fmt_cell(spreads.get('r2'), 2)} / "
            f"{_fmt_cell(spreads.get('r3'), 2)} | "
            f"{_fmt_cell(item['round_max_diff_deg'], 2)}° | {item['status']} | "
            f"{missing or '—'} |"
        )
    lines.append("")
    lines.append(
        "缺峰保留为缺测，不调阈值；完整逐帧质心、候选峰、峰数与"
        "峰/背景比见 JSON 的 `rounds[].analysis.patterns` 与 "
        "`repeatability.patterns[].per_round[]`。"
    )
    lines.append("")

    edge_items = [
        item for item in rep["patterns"] if item.get("edge_clipped_rounds")
    ]
    if edge_items:
        shape = (payload["invariants"].get("frame_shape") or {})
        rows_px = shape.get("rows")
        cols_px = shape.get("cols")
        lines.append("### 3.1 判读边界条件（重复性门的排除项）")
        lines.append("")
        lines.append(
            f"- 规则：principal 峰心落在帧任一边界 {EDGE_BAND_PX:.0f} px 带内，"
            "即视为被边界截断的极大值（不是波瓣）；该轮记为缺测，"
            f"不进入 ≤{rep['threshold_deg']}° 重复性门。"
            "原始 axis_angle_deg、principal、peak、background、峰心仍完整保留在 JSON。"
        )
        lines.append(
            f"- 本次帧 shape：{rows_px} × {cols_px} px；被排除的图样 "
            f"{len(edge_items)} 个、轮次 {rep['counts']['edge_clipped_rounds']} 个："
        )
        for item in edge_items:
            for row in item["per_round"]:
                edge = row.get("edge_clipped")
                if edge is None:
                    continue
                lines.append(
                    f"  - `{item['base_id']}` {row['round']}："
                    f"principal peak={_fmt_cell(edge.get('peak_counts'), 1)} counts / "
                    f"background={_fmt_cell(edge.get('background_counts'), 1)} counts，"
                    f"峰心 row={edge['row']:.1f}, col={edge['col']:.1f}"
                    f"（距最近边界 {edge['distance_to_edge_px']:.1f} px），"
                    f"原始 axis_angle_deg="
                    f"{_fmt_cell(row['axis_angle_deg'], 2)}° 已排除。"
                )
        facts = (payload.get("baseline") or {}).get("principal_facts") or {}
        if (payload.get("baseline") or {}).get("supplied") and facts:
            lines.append("- 基线对照（基线 principal 峰位/峰值）：")
            for item in edge_items:
                base_id = item["base_id"]
                fact = facts.get(base_id)
                core_exclude = next(
                    (
                        row.get("core_exclude_px")
                        for row in item["per_round"]
                        if _is_number(row.get("core_exclude_px"))
                    ),
                    None,
                )
                if not isinstance(fact, dict) or not fact:
                    lines.append(
                        f"  - `{base_id}`：基线 principal 为空"
                        "（基线口径下该图样即缺测）。"
                    )
                elif _is_number(fact.get("magnitude")):
                    magnitude = float(fact["magnitude"])
                    peak = fact.get("peak")
                    if _is_number(core_exclude) and magnitude < float(core_exclude):
                        lines.append(
                            f"  - `{base_id}`：基线 principal "
                            f"magnitude={_fmt_cell(magnitude, 1)} px、"
                            f"peak={_fmt_cell(peak, 1)} counts，"
                            f"即真实波瓣位于 0 级约 {magnitude:.0f} px 处；"
                            f"本次逐轮记录 core_exclude_px="
                            f"{_fmt_cell(float(core_exclude), 1)} px，大于该距离，"
                            "因此本次口径下该波瓣被 0 级核心排除。"
                        )
                    else:
                        lines.append(
                            f"  - `{base_id}`：基线 principal "
                            f"magnitude={_fmt_cell(magnitude, 1)} px、"
                            f"peak={_fmt_cell(peak, 1)} counts。"
                        )
        gate_states = [
            round_row["analysis"]["attribution_gates"][key].get("pass")
            for round_row in payload["rounds"]
            for key in GATE_KEYS
        ]
        if gate_states and all(state is False for state in gate_states):
            verdict_text = "三轮 G1–G4 仍全 fail，结论不依赖这些伪值"
        else:
            verdict_text = "G1–G4 的通过/失败判定见上文 §2 表，本节只列事实"
        lines.append(
            f"- G1–G4 的输入里这 {len(edge_items)} 个图样的轴角是边界极大值伪值；"
            f"{verdict_text}。`rounds[*].analysis.gates/verdict` 仍是分析器原值，"
            "本节排除只作用于重复性门。"
        )
        lines.append("")

    if payload.get("baseline", {}).get("supplied"):
        lines.append("## 4. 与归因基线对照（仅报告，不作重复性门）")
        lines.append("")
        lines.append(
            f"- 基线 verdict：`{payload['baseline'].get('verdict')}`"
        )
        lines.append(
            f"- 基线相位核对：{payload['baseline'].get('phase_checked')} 个 base pattern"
        )
        lines.append(
            "- 逐 pattern 的基线与各轮轴角差见 "
            "`repeatability.patterns[].baseline`；基线不同会话，"
            "不参与 ≤2° 重复性判定。"
        )
        lines.append("")

    lines.append("## 5. 限制与结论边界")
    lines.append("")
    for item in payload["limitations"]:
        lines.append(f"- {item}")
    for item in payload["conclusion_boundary"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Issue #247 同窗口重复性离线检查：校验 3 轮采集的冻结项，"
            "逐轮重跑分析器并用 axis_angle_deg 判 ≤2° 重复性。"
        )
    )
    parser.add_argument(
        "--dir",
        dest="capture_dir",
        type=Path,
        required=True,
        help="3 轮采集目录（含 plan.json / summary.json / released.json / frames/）",
    )
    parser.add_argument(
        "--baseline-dir",
        dest="baseline_dir",
        type=Path,
        default=None,
        help="可选的 2026-11-23 attribution 采集目录（含基线分析 JSON）",
    )
    parser.add_argument(
        "--probe-dir",
        dest="probe_dir",
        type=Path,
        required=True,
        help="六步光路预检 run 目录（含 optical_path_probe.json，"
        "提供 verdict.state / exit_code / 可用曝光窗口）",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=Path,
        required=True,
        help="只写这里；不得与采集/基线目录相同",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖 out-dir 中已有的同名产物（默认拒绝）",
    )
    parser.add_argument(
        "--json-name",
        default=DEFAULT_JSON_NAME,
        help=f"JSON 文件名（默认 {DEFAULT_JSON_NAME}）",
    )
    parser.add_argument(
        "--md-name",
        default=DEFAULT_MD_NAME,
        help=f"Markdown 文件名（默认 {DEFAULT_MD_NAME}）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    capture_dir = Path(args.capture_dir)
    baseline_dir = None if args.baseline_dir is None else Path(args.baseline_dir)
    probe_dir = Path(args.probe_dir)
    out_dir = Path(args.out_dir)

    for label, value in (("json", args.json_name), ("md", args.md_name)):
        _require(
            bool(value) and Path(value).name == value,
            f"--{label}-name 不允许包含路径分隔符：{value!r}",
        )
    for forbidden in (capture_dir, baseline_dir, probe_dir):
        if forbidden is not None and out_dir.resolve() == forbidden.resolve():
            print(
                "!! --out-dir 不得与采集/基线/预检目录相同；请指定独立输出目录",
                file=sys.stderr,
            )
            return 3

    json_path = out_dir / args.json_name
    md_path = out_dir / args.md_name
    if not args.overwrite and (json_path.exists() or md_path.exists()):
        print(
            f"!! 输出已存在（{json_path} / {md_path}）；"
            "默认拒绝覆盖，确认请加 --overwrite",
            file=sys.stderr,
        )
        return 4

    try:
        payload = check_capture(capture_dir, baseline_dir, probe_dir)
    except CheckError as exc:
        print(f"!! 输入被拒绝：{exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"!! 输入被拒绝（解析失败）：{exc}", file=sys.stderr)
        return 3

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with json_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        markdown = render_markdown(payload)
        with md_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(markdown)
            handle.write("\n")
    except OSError as exc:
        print(f"!! 写入输出失败：{exc}", file=sys.stderr)
        return 5

    rep = payload["repeatability"]
    print(
        "重复性检查完成："
        f"reproduced={rep['counts']['reproduced']}/"
        f"{rep['n_evaluable_patterns']}，"
        f"not_reproduced={rep['counts']['not_reproduced']}，"
        f"not_assessable={rep['counts']['not_assessable']}；"
        f"overall={'pass' if rep['overall_pass'] else 'fail'}"
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0 if rep["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
