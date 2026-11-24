"""Issue #247 跨会话独立复现离线检查（2026-11-24）。

只读一个跨会话采集根目录（``--root``）与其 ``cross_session_manifest.json``，
按 ``docs/dev/hw_window_20261124_issue247_cross_session_prereg.md`` 回答两个问题：

1. 主判据（冻结口径）：11-24 同窗口“15 个可评估图样方向稳定、5 个 p192 缺测”
   的现象，能否在三个真正独立的 SLM 开关会话中重复出现；
2. 附加审计（仅补充证据）：被 ``core_exclude_px=41.0`` 吞掉的 ``35–45 px`` 环带内，
   p192 是否存在显著高于局部背景、方向跨会话稳定的候选峰。

证据来源全部复用现有契约，不复制判据实现
------------------------------------------
* 清单 schema 与 ``plan_digest``：``issue247_cross_session_capture_20261124.py``；
* 逐会话冻结项校验与 ``axis_angle_deg`` 重算：
  ``issue247_reproducibility_check_20261124.py::check_capture``（``CheckError``）；
* p192 环带逐帧候选：``issue247_core_blind_audit_20261124.py::audit_capture``
  （``AuditError``）。

判定口径（冻结，不得回改）
--------------------------
* 主判据：15 个冻结主判据 id（另有 5 个 p192 在冻结主判据下不可评估，仅由补审
  覆盖）。图样范围按预注册 §3.1 的 ``PINNED_MAIN_BASE_IDS`` 显式写死，
  不按本轮结果事后挑选；每会话取非空 ``per_round[].gate_axis_angle_deg``；
  每会话至少 2 个内部轮次可评估才产生会话级轴角（``axial_mean_deg``，模 180°）。
  三个会话都必须可评估，任意两会话轴向差 ``axial_diff_deg <= 2.0`` 才算
  ``reproduced``。缺一个会话即 ``not_assessable``，**不得用另两个会话替代**；
  角度超门即 ``not_reproduced``。
* 主判据之外的那 5 个 p192（预注册 §3.1 列为 11-24 因 ``core_exclude_px=41.0``
  缺测的图样）**不计入主判据通过数、也不当作失败**：本轮仍在 ``main.patterns[]``
  逐条如实记录，另列 ``main.non_pinned_patterns`` 与 ``main.p192_not_assessable``。
  若本轮它们反而可评估且方向不一致，只记为 ``contradiction_outside_gate``
  （“与 11-24 相反的新证据”），不改写 15/15 的结论口径。
* 补审：每会话对 ``strong_candidate and within_core_exclude`` 的帧，用该帧候选峰
  相对 ``zero_order`` 的位移算轴角；每会话至少 3 帧才可评估，三会话两两轴向差
  ``<= 2.0`` 才 ``reproduced``。补审判定与主判据 p192 缺测记录**分开存放，永不合并**。

输入不合格一律拒绝（退出码 3），不重算、不写任何产物：schema 不符、
``dry_run is not False``、``stopped_early is not False``、``completed_sessions != 3``、
会话不是严格有序的 ``s1/s2/s3``、``status != ok``、``exit_code != 0``、
``artifact_problems`` 非空、三会话 ``plan_sha256`` 不一致或不等于清单顶层值、
``out``/``probe_dir`` 缺失或彼此不互异、捕获参数不是冻结的 ``0.04 ms`` / ``0.8 s``、
``check_capture``/``audit_capture`` 拒绝、单会话 ``plan_digest`` 与清单不一致。

边界（必须写清）
----------------
* 本脚本不定位根因，不证明 panel-y 路由、LUT 实际相位、支持域匹配；
* 不证明任何校正算法收益，不冻结映射，不调阈值，不关闭 issue；
* 即使全部门通过，也只能写“在本预注册口径下跨会话独立可重复”；
* 不 import 任何设备驱动，不连接设备；采集目录与预检目录只读；
* 输出只写到 ``--out-dir``（必须全新），默认拒绝覆盖已有产物（退出码 4）。

用法::

    python scripts/slm/issue247_cross_session_check_20261124.py \
        --root data/debug/issue247_cross_session_20261124 \
        --out-dir data/debug/issue247_cross_session_check_20261124 \
        [--baseline-dir data/debug/attribution_20261123]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import fmt_general
from scripts._common.provenance import provenance_block_for
from scripts.slm import issue247_reproducibility_check_20261124 as repeatability_check

# core_blind_audit / cross_session_capture 仅在硬件机工作区, 尚未入库 (任何 git ref 均无):
# 缺依赖时降级为可导入的跳过态, 硬件机上 import 成功、行为与原版逐字节一致。
try:
    from scripts.slm import (
        issue247_core_blind_audit_20261124 as core_audit,
        issue247_cross_session_capture_20261124 as capture_manifest,
    )

    DEPS_MISSING: str | None = None
except ImportError as exc:
    DEPS_MISSING = str(exc)
    core_audit = None
    capture_manifest = None

ANALYZER_PATH = (
    Path(__file__).resolve().parent / "issue247_waveform_probe_solve_20261122.py"
)
PREREG_REL = "docs/dev/hw_window_20261124_issue247_cross_session_prereg.md"
DEFAULT_JSON_NAME = "issue247_cross_session_check_20261124.json"
DEFAULT_MD_NAME = "issue247_cross_session_check_20261124.md"
#: 报告溯源块的注册表 key；未注册时 ``provenance_block_for`` 返回空串，不伪造。
REPORT_KEY = "issue247_cross_session_check_20261124"

CHECK_SCHEMA = "issue247-cross-session-check/v1"
MANIFEST_NAME = capture_manifest.MANIFEST_NAME if capture_manifest else ""
MANIFEST_SCHEMA = capture_manifest.MANIFEST_SCHEMA if capture_manifest else ""
SESSION_IDS: tuple[str, ...] = (
    tuple(capture_manifest.SESSIONS) if capture_manifest else ()
)

#: 预注册冻结采集参数。
FROZEN_EXPOSURE_MS = 0.04
FROZEN_SETTLE_S = 0.8
PARAM_TOLERANCE = 1e-9

#: 主判据：会话级轴向差门与每会话最少可评估内部轮次。
GATE_AXIAL_DIFF_DEG = 2.0
MIN_ASSESSABLE_ROUNDS = 2
#: 补审：每会话最少可用于判定的 strong_candidate 帧数。
MIN_SUPPLEMENTAL_FRAMES = 3

EVALUABLE_BASE_IDS: tuple[str, ...] = tuple(repeatability_check.EVALUABLE_BASE_IDS)

#: 预注册 §3.1 冻结的主判据图样范围（顺序照抄预注册，不事后挑选）。
PINNED_MAIN_BASE_IDS: tuple[str, ...] = (
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
    "grating_x_p192",
    "tilt_x_plus_p192",
    "tilt_x_minus_p192",
    "patch_x_plus_p192",
    "grating_y_p192",
)
#: 预注册 §3.1 明确排除在主判据通过数之外的 5 个 p192（11-24 因
#: ``core_exclude_px=41.0`` 缺测）。它们不计入主判据通过数、也不当作失败。
EXCLUDED_P192_BASE_IDS: tuple[str, ...] = (
    "patch_x_minus_p192",
    "tilt_y_plus_p192",
    "tilt_y_minus_p192",
    "patch_y_plus_p192",
    "patch_y_minus_p192",
)
_NON_PINNED_BASE_IDS: tuple[str, ...] = tuple(
    base for base in EVALUABLE_BASE_IDS if base not in PINNED_MAIN_BASE_IDS
)


def _assert_pinned_scope_matches_registry() -> None:
    """防止冻结清单与既有注册表静默漂移；不一致即拒绝运行，不自行修补。

    实际调用点在 ``_require`` 定义之后（模块末尾），因为本函数依赖它。
    """
    pinned = set(PINNED_MAIN_BASE_IDS)
    _require(
        len(pinned) == len(PINNED_MAIN_BASE_IDS),
        "PINNED_MAIN_BASE_IDS 有重复 id",
    )
    _require(
        pinned < set(EVALUABLE_BASE_IDS),
        "PINNED_MAIN_BASE_IDS 必须是 EVALUABLE_BASE_IDS 的真子集",
    )
    _require(
        set(_NON_PINNED_BASE_IDS) == set(EXCLUDED_P192_BASE_IDS),
        "EVALUABLE_BASE_IDS - PINNED_MAIN_BASE_IDS 必须正好等于预注册排除的 5 个 "
        f"p192；实际={sorted(_NON_PINNED_BASE_IDS)}",
    )


#: 预注册 §3.2 冻结的是“所有 ``_p192`` 图样”，不是一个 5 项子集：11-24 窗口里
#: 恰好有 5 个 p192 缺测，那是**结果**，不能反向写成冻结清单。补审因此覆盖全部
#: p192 图样；主判据中实际缺测的那部分另行列在 ``main.p192_not_assessable``。
SUPPLEMENTAL_BASE_IDS: tuple[str, ...] = (
    tuple(base for base in EVALUABLE_BASE_IDS if core_audit.P192_TOKEN in base)
    if core_audit
    else ()
)

VERDICT_REPRODUCED = "reproduced"
VERDICT_NOT_REPRODUCED = "not_reproduced"
VERDICT_NOT_ASSESSABLE = "not_assessable"
VERDICTS = frozenset(
    {VERDICT_REPRODUCED, VERDICT_NOT_REPRODUCED, VERDICT_NOT_ASSESSABLE}
)

FROZEN_MAIN_VERDICT_SCOPE = (
    "冻结主判据（预注册 §3.1 的 pinned_15）：15 个冻结 base id；逐会话非空 "
    "per_round[].gate_axis_angle_deg；每会话至少 2 个内部轮次可评估；"
    "三会话齐全且任意两两轴向差 <= 2.0 才 reproduced；"
    "另有 5 个 p192 在旧 core_exclude_px=41.0 下记 not_assessable，"
    "不计入主判据通过数、也不当作失败，只由补审覆盖，不因补审发现峰而改写。"
)
SUPPLEMENTAL_AUDIT_SCOPE = (
    "附加审计（仅补充证据）：35–45 px 环带内 strong_candidate 且 within_core_exclude "
    "的帧，按候选峰相对 zero_order 的位移算轴角；每会话至少 3 帧才可评估；"
    "三会话两两轴向差 <= 2.0 才 reproduced；与主判据 p192 缺测记录分开存放，永不合并。"
)

CONCLUSION_BOUNDARIES: tuple[str, ...] = (
    "不定位根因：不区分 panel-y 路由、LUT 实际相位、支持域匹配等假设。",
    "不证明任何校正算法收益，也不比较算法优劣。",
    "不冻结任何映射（灰阶/相位/方向），不回改图样、曝光、环带或阈值。",
    "即使补审发现候选峰，也不能写成“波瓣在物理上消失/不存在”，只能写“当前主判据不可评估”。",
    "不关闭 issue，不 push；跨会话一致只在本预注册口径内成立。",
    "补审判定永不并入主判据计数或结论。",
)

EXIT_POLICY = (
    "退出码由冻结主判据（pinned_15）与补审共同决定："
    "0 = 15 个冻结主判据图样全部 reproduced 且补审 reproduced；"
    "1 = 输入合法但未全通过（含主判据只有缺测、或补审不足/未复现）；"
    "3 = 清单或输入被拒绝（不重算、不写产物）；"
    "4 = 输出位置被拒绝（重叠只读输入或已有同名产物）；5 = 写盘 OSError。"
    "补审判定本身不并入主判据 verdict，两者在报告中分别列出。"
)


class RejectedError(ValueError):
    """输入未通过冻结项校验（退出码 3）；不修补、不丢样本、不出结论。"""


class OutputRefusedError(ValueError):
    """输出位置被拒绝（重叠只读输入或已有同名产物，退出码 4）。"""


#: 注入点：测试可替换为纯合成实现，签名与 ``check_capture`` 一致。
_RUN_CHECK: Callable[[Path, Path | None, Path], dict[str, Any]] = (
    repeatability_check.check_capture
)
def _audit_unavailable(_path: Path) -> dict[str, Any]:
    raise RuntimeError(f"blind audit 依赖模块未入库, 不可运行: {DEPS_MISSING}")


#: 注入点：测试可替换为纯合成实现，签名与 ``audit_capture`` 一致。
_RUN_AUDIT: Callable[[Path], dict[str, Any]] = (
    core_audit.audit_capture if core_audit else _audit_unavailable
)


def _analyzer() -> Any:
    """复用同窗口检查器加载的冻结分析器（同一份 ``axis_angle_deg`` 口径）。"""
    return repeatability_check._analyzer()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RejectedError(message)


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RejectedError(f"无法读取 JSON {path}：{exc}") from exc
    _require(isinstance(payload, dict), f"{path} 顶层不是 JSON 对象")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


# ---------------------------------------------------------------------------
# 清单校验（任何不合格都在重算与写盘之前拒绝）
# ---------------------------------------------------------------------------
def _session_plan_sha256(session: dict[str, Any], session_id: str) -> str:
    value = session.get("plan_sha256")
    _require(
        isinstance(value, str) and bool(value.strip()),
        f"{session_id} 缺少非空 plan_sha256，无法核对会话计划摘要",
    )
    return value


def _validate_session_entries(
    sessions: list[Any],
) -> list[dict[str, Any]]:
    _require(
        len(sessions) == len(SESSION_IDS),
        f"清单 sessions 应为 {len(SESSION_IDS)} 项，实际 {len(sessions)}",
    )
    records: list[dict[str, Any]] = []
    for index, session_id in enumerate(SESSION_IDS):
        session = sessions[index]
        _require(
            isinstance(session, dict),
            f"清单 sessions[{index}] 不是对象",
        )
        _require(
            session.get("session_id") == session_id,
            f"清单会话顺序异常：第 {index + 1} 项应为 {session_id}，"
            f"实际 {session.get('session_id')!r}",
        )
        _require(
            session.get("status") == "ok",
            f"{session_id} status 不是 ok：{session.get('status')!r}",
        )
        exit_code = session.get("exit_code")
        _require(
            isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
            and exit_code == 0,
            f"{session_id} exit_code 不是 0：{exit_code!r}",
        )
        problems = session.get("artifact_problems")
        _require(
            isinstance(problems, list) and not problems,
            f"{session_id} artifact_problems 非空：{problems!r}",
        )
        out = session.get("out")
        probe_dir = session.get("probe_dir")
        _require(
            isinstance(out, str) and bool(out.strip()),
            f"{session_id} 缺少 out",
        )
        _require(
            isinstance(probe_dir, str) and bool(probe_dir.strip()),
            f"{session_id} 缺少 probe_dir",
        )
        out_path = Path(out).resolve()
        probe_path = Path(probe_dir).resolve()
        _require(out_path.is_dir(), f"{session_id} out 不存在：{out_path}")
        _require(
            probe_path.is_dir(),
            f"{session_id} probe_dir 不存在：{probe_path}",
        )
        records.append(
            {
                "session_id": session_id,
                "out": out_path,
                "probe_dir": probe_path,
                "plan_sha256": _session_plan_sha256(session, session_id),
                "status": session.get("status"),
                "exit_code": exit_code,
                "artifact_problems": problems,
            }
        )
    return records


def _check_entry_distinctness(
    records: list[dict[str, Any]], root: Path
) -> None:
    seen: dict[Path, str] = {}
    for record in records:
        for field in ("out", "probe_dir"):
            path = record[field]
            label = f"{record['session_id']}.{field}"
            _require(
                path != root,
                f"{label} 与 --root 相同：{path}",
            )
            previous = seen.get(path)
            _require(
                previous is None,
                f"{label} 与 {previous} 指向同一目录：{path}；"
                "各会话输出与预检目录必须互异",
            )
            seen[path] = label


def _check_plan_digests(
    records: list[dict[str, Any]], top_level: object
) -> None:
    digests = [record["plan_sha256"] for record in records]
    _require(
        len(set(digests)) == 1,
        "三会话 plan_sha256 不一致：" + "，".join(
            f"{record['session_id']}={record['plan_sha256']}"
            for record in records
        ),
    )
    if top_level is not None:
        _require(
            isinstance(top_level, str),
            "清单顶层 plan_sha256 不是字符串",
        )
        _require(
            top_level == digests[0],
            "清单顶层 plan_sha256 与各会话不一致："
            f"顶层={top_level}，会话={digests[0]}",
        )


def _check_capture_settings(manifest: dict[str, Any]) -> dict[str, Any]:
    settings = manifest.get("capture_settings")
    _require(
        isinstance(settings, dict),
        "清单缺少 capture_settings，无法核对冻结曝光/settle",
    )
    exposure = settings.get("exposure_ms")
    settle = settings.get("settle_s")
    _require(
        _is_number(exposure),
        f"capture_settings.exposure_ms 不是数值：{exposure!r}",
    )
    _require(
        _is_number(settle),
        f"capture_settings.settle_s 不是数值：{settle!r}",
    )
    _require(
        abs(float(exposure) - FROZEN_EXPOSURE_MS) <= PARAM_TOLERANCE,
        f"capture_settings.exposure_ms 不是冻结值 {FROZEN_EXPOSURE_MS}："
        f"{exposure!r}",
    )
    _require(
        abs(float(settle) - FROZEN_SETTLE_S) <= PARAM_TOLERANCE,
        f"capture_settings.settle_s 不是冻结值 {FROZEN_SETTLE_S}：{settle!r}",
    )
    return {
        "exposure_ms": float(exposure),
        "settle_s": float(settle),
        "frozen_exposure_ms": FROZEN_EXPOSURE_MS,
        "frozen_settle_s": FROZEN_SETTLE_S,
    }


def validate_manifest(root: Path) -> dict[str, Any]:
    """严格校验跨会话清单；任何不合格抛 ``RejectedError``（退出码 3）。"""
    root = Path(root).resolve()
    _require(root.is_dir(), f"--root 不存在或不是目录：{root}")
    manifest_path = root / MANIFEST_NAME
    _require(
        manifest_path.is_file(),
        f"--root 下缺少清单 {MANIFEST_NAME}：{manifest_path}",
    )
    manifest = _read_json(manifest_path)

    _require(
        manifest.get("schema") == MANIFEST_SCHEMA,
        f"清单 schema 不是 {MANIFEST_SCHEMA}：{manifest.get('schema')!r}",
    )
    _require(
        manifest.get("dry_run") is False,
        f"清单 dry_run 不是 false（试运行或缺失）：{manifest.get('dry_run')!r}",
    )
    _require(
        manifest.get("stopped_early") is False,
        "清单 stopped_early 不是 false（采集提前停止）："
        f"{manifest.get('stopped_early')!r}",
    )
    completed = manifest.get("completed_sessions")
    _require(
        isinstance(completed, int)
        and not isinstance(completed, bool)
        and completed == len(SESSION_IDS),
        f"清单 completed_sessions 应为 {len(SESSION_IDS)}：{completed!r}",
    )
    sessions = manifest.get("sessions")
    _require(isinstance(sessions, list), "清单 sessions 不是列表")
    records = _validate_session_entries(sessions)
    _check_entry_distinctness(records, root)
    _check_plan_digests(records, manifest.get("plan_sha256"))
    settings = _check_capture_settings(manifest)

    return {
        "path": manifest_path,
        "sha256": _sha256_file(manifest_path),
        "schema": MANIFEST_SCHEMA,
        "root": root,
        "manifest": manifest,
        "sessions": records,
        "capture_settings": settings,
        "dry_run": manifest.get("dry_run"),
        "stopped_early": manifest.get("stopped_early"),
        "completed_sessions": completed,
    }


# ---------------------------------------------------------------------------
# 逐会话重算（复用现有 check_capture / audit_capture）
# ---------------------------------------------------------------------------
def _invoke_check(
    capture_dir: Path, baseline_dir: Path | None, probe_dir: Path
) -> dict[str, Any]:
    try:
        return _RUN_CHECK(capture_dir, baseline_dir, probe_dir)
    except repeatability_check.CheckError as exc:
        raise RejectedError(
            f"{capture_dir} 未通过冻结采集证据校验：{exc}"
        ) from exc
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # 与既有检查器 CLI 同一口径：解析失败也算输入被拒绝，不得 traceback。
        raise RejectedError(
            f"{capture_dir} 采集证据解析失败：{exc}"
        ) from exc


def _invoke_audit(capture_dir: Path) -> dict[str, Any]:
    try:
        return _RUN_AUDIT(capture_dir)
    except core_audit.AuditError as exc:
        raise RejectedError(f"{capture_dir} 补审读取失败：{exc}") from exc
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RejectedError(f"{capture_dir} 补审解析失败：{exc}") from exc


def _repeatability_items(payload: dict[str, Any], label: str) -> dict[str, dict]:
    repeatability = payload.get("repeatability")
    _require(
        isinstance(repeatability, dict),
        f"{label} 的 check_capture 结果缺 repeatability",
    )
    patterns = repeatability.get("patterns")
    _require(
        isinstance(patterns, list),
        f"{label} 的 repeatability.patterns 不是列表",
    )
    items: dict[str, dict] = {}
    for item in patterns:
        if not isinstance(item, dict):
            continue
        base_id = item.get("base_id")
        if isinstance(base_id, str):
            items[base_id] = item
    missing = [base for base in EVALUABLE_BASE_IDS if base not in items]
    _require(
        not missing,
        f"{label} 的 repeatability 缺可评估图样：{missing}",
    )
    return items


def _supplemental_items(payload: dict[str, Any], label: str) -> dict[str, dict]:
    audit = payload.get("supplemental_audit")
    _require(
        isinstance(audit, dict),
        f"{label} 的 audit_capture 结果缺 supplemental_audit",
    )
    patterns = audit.get("patterns")
    _require(
        isinstance(patterns, list),
        f"{label} 的 supplemental_audit.patterns 不是列表",
    )
    items: dict[str, dict] = {}
    for item in patterns:
        if not isinstance(item, dict):
            continue
        pattern_id = item.get("pattern_id")
        if isinstance(pattern_id, str):
            items[pattern_id] = item
    missing = [base for base in SUPPLEMENTAL_BASE_IDS if base not in items]
    _require(
        not missing,
        f"{label} 的补审缺 p192 图样：{missing}",
    )
    return items


def _run_sessions(
    validated: dict[str, Any], baseline_dir: Path | None
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for record in validated["sessions"]:
        session_id = record["session_id"]
        out = record["out"]
        payload = _invoke_check(out, baseline_dir, record["probe_dir"])
        plan = _read_json(out / "plan.json")
        digest = capture_manifest.plan_digest(plan)
        _require(
            digest == record["plan_sha256"],
            f"{session_id} plan_digest 与清单不一致："
            f"重算={digest}，清单={record['plan_sha256']}",
        )
        main_items = _repeatability_items(payload, session_id)
        audit_payload = _invoke_audit(out)
        supplemental_items = _supplemental_items(audit_payload, session_id)
        repeatability = payload["repeatability"]
        reports.append(
            {
                "session_id": session_id,
                "capture_dir": out,
                "probe_dir": record["probe_dir"],
                "plan_digest": digest,
                "repeatability_counts": repeatability.get("counts"),
                "repeatability_overall_pass": repeatability.get("overall_pass"),
                "main_items": main_items,
                "supplemental_items": supplemental_items,
            }
        )
    return reports


# ---------------------------------------------------------------------------
# 主判据
# ---------------------------------------------------------------------------
def _gate_angles(pattern_item: dict[str, Any]) -> list[float]:
    rows = pattern_item.get("per_round")
    if not isinstance(rows, list):
        return []
    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        angle = row.get("gate_axis_angle_deg")
        if _is_number(angle):
            values.append(float(angle))
    return values


def _session_axial(pattern_item: dict[str, Any]) -> dict[str, Any]:
    angles = _gate_angles(pattern_item)
    mean = (
        _analyzer().axial_mean_deg(angles)
        if len(angles) >= MIN_ASSESSABLE_ROUNDS
        else None
    )
    if mean is None:
        status = VERDICT_NOT_ASSESSABLE
        reason = (
            f"可评估内部轮次 {len(angles)} < {MIN_ASSESSABLE_ROUNDS}"
            if len(angles) < MIN_ASSESSABLE_ROUNDS
            else "轴向平均退化，无法定轴"
        )
    else:
        status = "assessed"
        reason = None
    return {
        "n_assessable_rounds": len(angles),
        "gate_angles_deg": angles,
        "axial_mean_deg": None if mean is None else float(mean),
        "status": status,
        "reason": reason,
    }


def _combine_session_means(
    means: dict[str, float | None],
) -> tuple[str, dict[str, float], float | None, str | None]:
    missing = [
        session_id for session_id in SESSION_IDS if means.get(session_id) is None
    ]
    if missing:
        return (
            VERDICT_NOT_ASSESSABLE,
            {},
            None,
            "缺会话级轴角（不得用另两个会话替代）：" + ",".join(missing),
        )
    analyzer = _analyzer()
    pairwise: dict[str, float] = {}
    for index, first in enumerate(SESSION_IDS):
        for second in SESSION_IDS[index + 1 :]:
            diff = analyzer.axial_diff_deg(means[first], means[second])
            _require(diff is not None, f"{first}-{second} 轴向差无法计算")
            pairwise[f"{first}-{second}"] = float(diff)
    max_diff = max(pairwise.values())
    if max_diff <= GATE_AXIAL_DIFF_DEG:
        return VERDICT_REPRODUCED, pairwise, max_diff, None
    return (
        VERDICT_NOT_REPRODUCED,
        pairwise,
        max_diff,
        f"最大两两会话轴向差 {max_diff:.3f}° > {GATE_AXIAL_DIFF_DEG}°",
    )


def main_pattern_verdict(
    items_by_session: dict[str, dict[str, dict]], base_id: str
) -> dict[str, Any]:
    sessions = {
        session_id: _session_axial(items_by_session[session_id][base_id])
        for session_id in SESSION_IDS
    }
    verdict, pairwise, max_diff, reason = _combine_session_means(
        {
            session_id: sessions[session_id]["axial_mean_deg"]
            for session_id in SESSION_IDS
        }
    )
    return {
        "base_id": base_id,
        "sessions": sessions,
        "pairwise_diff_deg": pairwise,
        "max_pairwise_diff_deg": max_diff,
        "verdict": verdict,
        "reason": reason,
    }


def main_criterion(
    reports: list[dict[str, Any]]
) -> dict[str, Any]:
    items_by_session = {
        report["session_id"]: report["main_items"] for report in reports
    }
    patterns = [
        main_pattern_verdict(items_by_session, base_id)
        for base_id in EVALUABLE_BASE_IDS
    ]
    counts = {
        verdict: sum(1 for item in patterns if item["verdict"] == verdict)
        for verdict in sorted(VERDICTS)
    }
    by_id = {item["base_id"]: item for item in patterns}
    pinned = [by_id[base_id] for base_id in PINNED_MAIN_BASE_IDS]
    pinned_counts = {
        verdict: sum(1 for item in pinned if item["verdict"] == verdict)
        for verdict in sorted(VERDICTS)
    }
    # 预注册 §3.1：只有 pinned_15 参与 overall_pass。其余 5 个 p192 不计入通过数、
    # 也不当作失败，因此不会把 core_exclude_px=41.0 的结构性盲区伪装成失败。
    overall_pass = pinned_counts[VERDICT_REPRODUCED] == len(PINNED_MAIN_BASE_IDS)
    if overall_pass:
        overall_verdict = VERDICT_REPRODUCED
    elif pinned_counts[VERDICT_NOT_REPRODUCED]:
        overall_verdict = VERDICT_NOT_REPRODUCED
    else:
        # 只有缺测、没有任何角度超门：结论是证据不足，不能写成“未复现”。
        overall_verdict = VERDICT_NOT_ASSESSABLE
    # 11-24 同窗口的“5 个 p192 缺测”是**结果**，不是冻结清单；这里按主判据
    # 实际判定结果派生，供报告与既有结论对照，不参与任何判定。
    p192_not_assessable = [
        item["base_id"]
        for item in patterns
        if core_audit.P192_TOKEN in item["base_id"]
        and item["verdict"] == VERDICT_NOT_ASSESSABLE
    ]
    non_pinned_patterns = [
        {
            "base_id": item["base_id"],
            "status": item["verdict"],
            "reason": item["reason"],
            "gate_scope_excludes": (
                "预注册 §3.1：该 p192 不计入 pinned_15 主判据通过数，也不当作失败；"
                "此处只如实记录本轮状态，方向稳定性由补审覆盖。"
            ),
        }
        for item in patterns
        if item["base_id"] not in PINNED_MAIN_BASE_IDS
    ]
    # 非 pinned 图样若本轮反而可评估且方向超门，属于“与 11-24 相反的新证据”，
    # 单独标出供报告明写，但不改写 15/15 的主判据口径。
    contradiction_ids = [
        item["base_id"]
        for item in non_pinned_patterns
        if item["status"] == VERDICT_NOT_REPRODUCED
    ]
    return {
        "gate": {
            "scope": "pinned_15",
            "n_patterns": len(PINNED_MAIN_BASE_IDS),
            "n_required": len(PINNED_MAIN_BASE_IDS),
            "n_reported_patterns": len(EVALUABLE_BASE_IDS),
            "pinned_base_ids": list(PINNED_MAIN_BASE_IDS),
            "excluded_p192": list(EXCLUDED_P192_BASE_IDS),
            "max_axial_diff_deg": GATE_AXIAL_DIFF_DEG,
            "min_assessable_rounds_per_session": MIN_ASSESSABLE_ROUNDS,
            "sessions_required": list(SESSION_IDS),
        },
        "patterns": patterns,
        "counts": counts,
        "pinned_counts": pinned_counts,
        "p192_not_assessable": p192_not_assessable,
        "non_pinned_patterns": non_pinned_patterns,
        "contradiction_outside_gate": bool(contradiction_ids),
        "contradiction_outside_gate_ids": contradiction_ids,
        "overall_pass": overall_pass,
        "verdict": overall_verdict,
    }


# ---------------------------------------------------------------------------
# p192 附加审计（仅补充证据）
# ---------------------------------------------------------------------------
def _axis_angle_deg(delta_row: float, delta_col: float) -> float:
    analyzer = _analyzer()
    angle = analyzer.axial_mean_deg([analyzer.angle_deg(delta_row, delta_col)])
    _require(angle is not None, "候选峰位移退化，无法定轴")
    return float(angle)


def _frame_axis_angles(pattern_item: dict[str, Any]) -> list[dict[str, Any]]:
    zero = pattern_item.get("zero_order") or {}
    zero_row = zero.get("row") if isinstance(zero, dict) else None
    zero_col = zero.get("col") if isinstance(zero, dict) else None
    _require(
        _is_number(zero_row) and _is_number(zero_col),
        f"{pattern_item.get('pattern_id')} 缺 zero_order.row/col，无法定轴",
    )
    frames = pattern_item.get("frames")
    if not isinstance(frames, list):
        return []
    kept: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        if not (frame.get("strong_candidate") and frame.get("within_core_exclude")):
            continue
        row = frame.get("row")
        col = frame.get("col")
        if not (_is_number(row) and _is_number(col)):
            continue
        kept.append(
            {
                "frame": frame.get("frame"),
                "row": float(row),
                "col": float(col),
                "radius_px": frame.get("radius_px"),
                "peak_minus_background": frame.get("peak_minus_background"),
                "axis_angle_deg": _axis_angle_deg(
                    float(row) - float(zero_row), float(col) - float(zero_col)
                ),
            }
        )
    return kept


def _supplemental_session(pattern_item: dict[str, Any]) -> dict[str, Any]:
    frames = _frame_axis_angles(pattern_item)
    mean = (
        _analyzer().axial_mean_deg([frame["axis_angle_deg"] for frame in frames])
        if len(frames) >= MIN_SUPPLEMENTAL_FRAMES
        else None
    )
    if mean is None:
        status = VERDICT_NOT_ASSESSABLE
        reason = (
            f"strong_candidate 且 within_core_exclude 的帧 {len(frames)} < "
            f"{MIN_SUPPLEMENTAL_FRAMES}"
            if len(frames) < MIN_SUPPLEMENTAL_FRAMES
            else "候选峰轴向平均退化，无法定轴"
        )
    else:
        status = "assessed"
        reason = None
    return {
        "n_frames": pattern_item.get("n_frames"),
        "n_strong_frames_within_core": len(frames),
        "frames": frames,
        "axial_mean_deg": None if mean is None else float(mean),
        "status": status,
        "reason": reason,
    }


def supplemental_pattern_verdict(
    items_by_session: dict[str, dict[str, dict]], base_id: str
) -> dict[str, Any]:
    sessions = {
        session_id: _supplemental_session(items_by_session[session_id][base_id])
        for session_id in SESSION_IDS
    }
    verdict, pairwise, max_diff, reason = _combine_session_means(
        {
            session_id: sessions[session_id]["axial_mean_deg"]
            for session_id in SESSION_IDS
        }
    )
    return {
        "base_id": base_id,
        "sessions": sessions,
        "pairwise_diff_deg": pairwise,
        "max_pairwise_diff_deg": max_diff,
        "verdict": verdict,
        "reason": reason,
    }


def supplemental_criterion(
    reports: list[dict[str, Any]]
) -> dict[str, Any]:
    items_by_session = {
        report["session_id"]: report["supplemental_items"]
        for report in reports
    }
    patterns = [
        supplemental_pattern_verdict(items_by_session, base_id)
        for base_id in SUPPLEMENTAL_BASE_IDS
    ]
    counts = {
        verdict: sum(1 for item in patterns if item["verdict"] == verdict)
        for verdict in sorted(VERDICTS)
    }
    if counts[VERDICT_NOT_REPRODUCED]:
        overall = VERDICT_NOT_REPRODUCED
    elif counts[VERDICT_REPRODUCED] == len(SUPPLEMENTAL_BASE_IDS):
        overall = VERDICT_REPRODUCED
    else:
        overall = VERDICT_NOT_ASSESSABLE
    return {
        "gate": {
            "max_axial_diff_deg": GATE_AXIAL_DIFF_DEG,
            "min_strong_frames_per_session": MIN_SUPPLEMENTAL_FRAMES,
            "n_supplemental_patterns": len(SUPPLEMENTAL_BASE_IDS),
        },
        "patterns": patterns,
        "counts": counts,
        "verdict": overall,
        "note": (
            "补审只回答“环带内是否存在方向稳定的候选峰”，不改变主判据 p192 的"
            "not_assessable 记录，也不等于波瓣物理存在或消失。"
        ),
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_payload(
    validated: dict[str, Any],
    reports: list[dict[str, Any]],
    baseline_dir: Path | None,
) -> dict[str, Any]:
    main = main_criterion(reports)
    supplemental = supplemental_criterion(reports)
    payload = {
        "schema": CHECK_SCHEMA,
        "generated_at_utc": _utc_now(),
        "prereg": PREREG_REL,
        "analyzer": _rel(ANALYZER_PATH),
        "root": _rel(validated["root"]),
        "baseline_dir": None if baseline_dir is None else _rel(baseline_dir),
        "manifest": {
            "name": MANIFEST_NAME,
            "path": _rel(validated["path"]),
            "sha256": validated["sha256"],
            "schema": validated["schema"],
        },
        "manifest_validation": {
            "dry_run": validated["dry_run"],
            "stopped_early": validated["stopped_early"],
            "completed_sessions": validated["completed_sessions"],
            "capture_settings": validated["capture_settings"],
            "sessions": [
                {
                    "session_id": record["session_id"],
                    "status": record["status"],
                    "exit_code": record["exit_code"],
                    "artifact_problems": record["artifact_problems"],
                    "plan_sha256": record["plan_sha256"],
                    "out": _rel(record["out"]),
                    "probe_dir": _rel(record["probe_dir"]),
                }
                for record in validated["sessions"]
            ],
        },
        "frozen_main_verdict_scope": FROZEN_MAIN_VERDICT_SCOPE,
        "supplemental_audit_scope": SUPPLEMENTAL_AUDIT_SCOPE,
        "main": main,
        "supplemental": supplemental,
        "sessions": [
            {
                "session_id": report["session_id"],
                "capture_dir": _rel(report["capture_dir"]),
                "probe_dir": _rel(report["probe_dir"]),
                "plan_digest": report["plan_digest"],
                "repeatability_counts": report["repeatability_counts"],
                "repeatability_overall_pass": report[
                    "repeatability_overall_pass"
                ],
            }
            for report in reports
        ],
        "exit_policy": EXIT_POLICY,
        "conclusion_boundaries": list(CONCLUSION_BOUNDARIES),
    }
    return _json_safe(payload)


def _fmt_cell(value: object, digits: int = 2) -> str:
    return fmt_general(value, f".{digits}f", missing_none="—", bool_labels=("是", "否"))


def _cell(session: dict[str, Any]) -> str:
    mean = _fmt_cell(session.get("axial_mean_deg"))
    rounds = session.get("n_assessable_rounds")
    return f"{mean}° ({rounds} 轮)"


def render_markdown(payload: dict[str, Any]) -> str:
    main = payload["main"]
    supplemental = payload["supplemental"]
    non_pinned_counts = {
        verdict: sum(
            1
            for item in main["non_pinned_patterns"]
            if item["status"] == verdict
        )
        for verdict in sorted(VERDICTS)
    }
    lines: list[str] = []
    lines.append("# Issue #247 跨会话独立复现离线检查")
    lines.append("")
    block = provenance_block_for(REPORT_KEY)
    if block:
        lines.append(block)
        lines.append("")
    lines.append(f"- **清单**: `{payload['manifest']['path']}`")
    lines.append(f"- **清单 SHA-256**: `{payload['manifest']['sha256']}`")
    lines.append(f"- **采集根目录**: `{payload['root']}`")
    if payload.get("baseline_dir"):
        lines.append(f"- **基线**: `{payload['baseline_dir']}`")
    lines.append(f"- **分析器**: `{payload['analyzer']}`")
    lines.append(f"- **预注册**: `{payload['prereg']}`")
    lines.append(f"- **生成时间**: {payload['generated_at_utc']}")
    lines.append("")

    lines.append("## 结论先行")
    lines.append("")
    lines.append(
        "- 清单冻结项：schema、`dry_run=false`、`stopped_early=false`、"
        "`completed_sessions=3`、会话顺序 `s1/s2/s3`、`status=ok`、`exit_code=0`、"
        "`artifact_problems` 为空、三会话 `plan_sha256` 一致、"
        f"曝光 `{payload['manifest_validation']['capture_settings']['exposure_ms']}` ms / "
        f"settle `{payload['manifest_validation']['capture_settings']['settle_s']}` s：全部通过。"
    )
    lines.append(
        f"- 冻结主判据（`{main['gate']['scope']}`，{main['gate']['n_patterns']} 个"
        f"冻结图样、要求 {main['gate']['n_required']} 个全部 reproduced，"
        f"任意两会话轴向差 ≤ {main['gate']['max_axial_diff_deg']}°）："
        f"**{'pass' if main['overall_pass'] else 'fail'}**"
        f"（`{main['verdict']}`）；"
        f"reproduced={main['pinned_counts'][VERDICT_REPRODUCED]}/"
        f"{main['gate']['n_patterns']}，"
        f"not_reproduced={main['pinned_counts'][VERDICT_NOT_REPRODUCED]}，"
        f"not_assessable={main['pinned_counts'][VERDICT_NOT_ASSESSABLE]}。"
    )
    lines.append(
        f"- 冻结主判据排除在外的 5 个 p192（`excluded_p192`，不计入通过数、"
        f"不当作失败）：`{'`、`'.join(main['gate']['excluded_p192'])}`；"
        f"本轮逐条状态 reproduced={non_pinned_counts[VERDICT_REPRODUCED]}、"
        f"not_reproduced={non_pinned_counts[VERDICT_NOT_REPRODUCED]}、"
        f"not_assessable={non_pinned_counts[VERDICT_NOT_ASSESSABLE]}。"
    )
    lines.append(
        f"- p192 附加审计（仅补充证据，{supplemental['gate']['n_supplemental_patterns']} "
        f"个 p192 图样，每会话至少 "
        f"{supplemental['gate']['min_strong_frames_per_session']} 帧）："
        f"**{supplemental['verdict']}**；"
        f"reproduced={supplemental['counts'][VERDICT_REPRODUCED]}，"
        f"not_reproduced={supplemental['counts'][VERDICT_NOT_REPRODUCED]}，"
        f"not_assessable={supplemental['counts'][VERDICT_NOT_ASSESSABLE]}。"
    )
    lines.append(f"- 退出码策略：{payload['exit_policy']}")
    missing_p192 = main.get("p192_not_assessable") or []
    if missing_p192:
        lines.append(
            "- 主判据下仍缺测的 p192（按实际判定派生，不参与判定）："
            + "、".join(f"`{base}`" for base in missing_p192)
            + "；缺测只表示“当前冻结口径不可评估”，不等于波瓣物理消失。"
        )
    else:
        lines.append(
            "- 主判据下 p192 全部可评估；11-24 同窗口的“5 个 p192 缺测”"
            "在本轮未重现（这是新增证据，不改冻结口径）。"
        )
    lines.append("")
    if main["contradiction_outside_gate"]:
        lines.append(
            "- **与 11-24 相反的新证据**：以下非 pinned p192 本轮可评估但方向"
            "跨会话超门："
            + "、".join(
                f"`{base}`" for base in main["contradiction_outside_gate_ids"]
            )
            + "。按预注册 §3.1，这不改写 pinned_15 主判据结论，仅如实记录。"
        )
        lines.append("")

    lines.append("## 主判据：非 pinned 的 5 个 p192（不计入通过数、不当作失败）")
    lines.append("")
    lines.append("| base_id | 本轮状态 | 说明 |")
    lines.append("|---|---|---|")
    for item in main["non_pinned_patterns"]:
        lines.append(
            "| {base} | {status} | {note} |".format(
                base=item["base_id"],
                status=item["status"],
                note=item["gate_scope_excludes"],
            )
        )
    lines.append("")

    lines.append("## 主判据（冻结口径）")
    lines.append("")
    lines.append(f"> {payload['frozen_main_verdict_scope']}")
    lines.append("")
    lines.append("| 图样 | s1 | s2 | s3 | 最大两两差 | 判定 |")
    lines.append("| 图样 | s1 | s2 | s3 | 最大两两差 | 判定 | 计入 pinned_15 |")
    lines.append("|---|---|---|---|---|---|---|")
    for item in main["patterns"]:
        sessions = item["sessions"]
        lines.append(
            "| {base} | {s1} | {s2} | {s3} | {diff}° | {verdict} | {pinned} |".format(
                base=item["base_id"],
                s1=_cell(sessions["s1"]),
                s2=_cell(sessions["s2"]),
                s3=_cell(sessions["s3"]),
                diff=_fmt_cell(item["max_pairwise_diff_deg"], 3),
                verdict=item["verdict"],
                pinned=(
                    "是" if item["base_id"] in PINNED_MAIN_BASE_IDS else "否（仅记录）"
                ),
            )
        )
    lines.append("")

    lines.append("## p192 附加审计（仅补充证据，不并入主判据）")
    lines.append("")
    lines.append(f"> {payload['supplemental_audit_scope']}")
    lines.append("")
    lines.append("| 图样 | 会话 | strong 帧 | 轴向均值 | 判定 |")
    lines.append("|---|---|---|---|---|")
    for item in supplemental["patterns"]:
        for session_id in SESSION_IDS:
            session = item["sessions"][session_id]
            lines.append(
                "| {base} | {sid} | {n} | {mean}° | {status} |".format(
                    base=item["base_id"],
                    sid=session_id,
                    n=session["n_strong_frames_within_core"],
                    mean=_fmt_cell(session["axial_mean_deg"]),
                    status=session["status"],
                )
            )
        lines.append(
            "| {base} | **合计** | — | — | **{verdict}** |".format(
                base=item["base_id"], verdict=item["verdict"]
            )
        )
    lines.append("")
    lines.append(f"> {supplemental['note']}")
    lines.append("")

    lines.append("## 逐会话重算摘要")
    lines.append("")
    lines.append("| 会话 | 采集目录 | plan_digest | 同窗口 overall_pass |")
    lines.append("|---|---|---|---|")
    for session in payload["sessions"]:
        lines.append(
            "| {sid} | `{path}` | `{digest}` | {flag} |".format(
                sid=session["session_id"],
                path=session["capture_dir"],
                digest=session["plan_digest"],
                flag=_fmt_cell(session["repeatability_overall_pass"]),
            )
        )
    lines.append("")

    lines.append("## 结论边界")
    lines.append("")
    for item in payload["conclusion_boundaries"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Issue #247 跨会话独立复现离线检查：只读跨会话清单与三个会话目录，"
            "复用现有检查器与补审脚本，按冻结主判据输出 reproduced / "
            "not_reproduced / not_assessable。"
        )
    )
    parser.add_argument(
        "--root",
        dest="root",
        type=Path,
        required=True,
        help=f"跨会话采集根目录（含 {MANIFEST_NAME} 与 s1/s2/s3 输出）",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=Path,
        required=True,
        help="全新输出目录；只写这里，不得与任何输入目录重叠",
    )
    parser.add_argument(
        "--baseline-dir",
        dest="baseline_dir",
        type=Path,
        default=None,
        help="可选的 2026-11-23 attribution 采集目录（含基线分析 JSON）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖 out-dir 中已有的同名产物（默认拒绝，退出码 4）",
    )
    return parser.parse_args(argv)


def _check_out_dir(
    out_dir: Path, validated: dict[str, Any], overwrite: bool
) -> tuple[Path, Path]:
    out_dir = Path(out_dir).resolve()
    inputs = [validated["root"]] + [
        path
        for record in validated["sessions"]
        for path in (record["out"], record["probe_dir"])
    ]
    inputs.append(validated["path"])
    if out_dir.exists() and not out_dir.is_dir():
        raise OutputRefusedError(f"--out-dir 已存在且不是目录：{out_dir}")
    if out_dir.is_dir():
        for path in inputs:
            if path.is_relative_to(out_dir):
                raise OutputRefusedError(
                    f"--out-dir 包含只读输入：{path} 落在 {out_dir} 内"
                )
    for path in inputs:
        if path.is_dir() and (
            out_dir == path or out_dir.is_relative_to(path)
        ):
            raise OutputRefusedError(
                f"--out-dir 与只读输入目录重叠：{out_dir} 落在 {path} 内"
            )
        if out_dir == path:
            raise OutputRefusedError(
                f"--out-dir 与只读输入文件相同：{out_dir}"
            )
    json_path = out_dir / DEFAULT_JSON_NAME
    md_path = out_dir / DEFAULT_MD_NAME
    if not overwrite:
        existing = [path for path in (json_path, md_path) if path.exists()]
        if existing:
            raise OutputRefusedError(
                "输出已存在，默认拒绝覆盖（如需覆盖请显式 --overwrite）："
                + "，".join(str(path) for path in existing)
            )
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    if DEPS_MISSING:
        print(f"!! 依赖模块未入库 (仅存在于硬件机), 无法运行：{DEPS_MISSING}", file=sys.stderr)
        return 3
    args = _parse_args(argv)
    try:
        _assert_pinned_scope_matches_registry()
        validated = validate_manifest(args.root)
    except RejectedError as exc:
        print(f"!! 拒绝：{exc}", file=sys.stderr)
        return 3
    try:
        json_path, md_path = _check_out_dir(
            args.out_dir, validated, bool(args.overwrite)
        )
    except OutputRefusedError as exc:
        print(f"!! 拒绝写出：{exc}", file=sys.stderr)
        return 4
    try:
        baseline_dir = (
            None if args.baseline_dir is None else Path(args.baseline_dir)
        )
        if baseline_dir is not None and not baseline_dir.is_dir():
            raise RejectedError(f"--baseline-dir 不存在：{baseline_dir}")
        reports = _run_sessions(validated, baseline_dir)
        payload = build_payload(validated, reports, baseline_dir)
    except RejectedError as exc:
        print(f"!! 拒绝：{exc}", file=sys.stderr)
        return 3

    json_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with json_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        with md_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(render_markdown(payload))
            handle.write("\n")
    except OSError as exc:
        print(f"!! 写入输出失败：{exc}", file=sys.stderr)
        return 5

    main_block = payload["main"]
    counts = main_block["counts"]
    pinned_counts = main_block["pinned_counts"]
    print(
        "跨会话检查完成："
        f"pinned_15 reproduced={pinned_counts[VERDICT_REPRODUCED]}/"
        f"{len(PINNED_MAIN_BASE_IDS)}，"
        f"not_reproduced={pinned_counts[VERDICT_NOT_REPRODUCED]}，"
        f"not_assessable={pinned_counts[VERDICT_NOT_ASSESSABLE]}；"
        f"非 pinned 逐条（不计入判定）：reproduced="
        f"{counts[VERDICT_REPRODUCED] - pinned_counts[VERDICT_REPRODUCED]}，"
        f"not_reproduced="
        f"{counts[VERDICT_NOT_REPRODUCED] - pinned_counts[VERDICT_NOT_REPRODUCED]}，"
        f"not_assessable="
        f"{counts[VERDICT_NOT_ASSESSABLE] - pinned_counts[VERDICT_NOT_ASSESSABLE]}；"
        f"main={'pass' if main_block['overall_pass'] else 'fail'}"
        f"（{main_block['verdict']}）；"
        f"supplemental={payload['supplemental']['verdict']}"
    )
    if main_block["contradiction_outside_gate"]:
        print(
            "!! 非 pinned 图样出现与 11-24 相反的新证据（方向超门）："
            + "，".join(main_block["contradiction_outside_gate_ids"])
        )
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    # 退出码 0 要求 pinned_15 全部 reproduced 且补审 reproduced；
    # 补审判定不并入 main.verdict，只参与最终通过与否。
    passed = (
        main_block["overall_pass"]
        and payload["supplemental"]["verdict"] == VERDICT_REPRODUCED
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
