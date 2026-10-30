"""逐段实测成本：把「这一段跑了几秒、几块」落进段目录，并把历史折算成估算。

为什么按段落在段目录里
----------------------
成本是**段**的属性（块数、曝光、模型分辨率都在段内固定），而段目录本来就是这段
全部产物的家。所以一份 ``cost.json`` 写在 ``shape/<形状>/<阶段>/`` 里，既不会和
journal 抢同一个文件（journal 是 append-only 的，成本是覆盖写的），也天然带着
"哪一段算不出来就只是缺这一行"的粒度。:func:`read_campaign_costs` 反向扫回来，
按 ``<形状>/<阶段>`` 归并成一张表。

估算口径
--------
只用**每块秒**（``seconds / blocks``）做外推：段长的绝对秒数没有可比性（段与段的
块数不同），而"每块多少秒"是同一条光路上最稳定的量。低/中/高取已测每块秒的
min / 中位 / max，乘上计划块数——这不是置信区间，只是"历史最快到历史最慢"的
包络。没有任何历史时返回全 0，并如实给出 ``basis_seconds_per_block`` 为空，
让页面能说「首次运行，无历史成本」而不是印一张看起来免费的 0 表。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

__all__ = [
    "COST_FILE",
    "estimate_campaign",
    "read_campaign_costs",
    "record_stage_cost",
]

#: 逐段成本文件名（落在段目录根部，与 ``manifest.json`` / ``summary.json`` 并列）。
COST_FILE = "cost.json"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_seconds(seconds: Any) -> float:
    """段耗时必须是有限非负数；否则宁可不写，也不写一个假数。"""
    number = _number(seconds)
    if number is None or number < 0:
        raise ValueError(f"stage seconds must be a finite non-negative number, got {seconds!r}")
    return number


def record_stage_cost(stage_dir: str | Path, *, stage: str, shape: str, seconds: float,
                      blocks: int = 0, source: str = "measured") -> Path:
    """把一段的实测耗时写进 ``stage_dir``，返回 ``cost.json`` 的路径。

    Args:
        stage_dir: 段目录（``out/shape/<形状>/<阶段>``）。
        stage: 阶段名（与 :data:`~ao_shaping.tools.slm.transfer.stages.STAGE_RECIPES`
            的 ``name`` 同一口径）。
        shape: 形状键（``square`` / ``rectangle`` ...）。
        seconds: 实测墙钟秒数。
        blocks: 这段真正测过的块数（由段 manifest 的行数给出）。0 表示"只有段首
            那一行"，此时 ``seconds_per_block`` 记 ``None``。
        source: ``measured``（runner 实测）或别的来源标记；估算只认有块数的行。

    成本文件是**覆盖写**：同一次续跑重跑同一段时，后一次的结果就是唯一事实。
    """
    duration = _positive_seconds(seconds)
    block_count = max(0, int(blocks or 0))
    payload: dict[str, Any] = {
        "stage": str(stage),
        "shape": str(shape),
        "seconds": duration,
        "blocks": block_count,
        "seconds_per_block": duration / block_count if block_count else None,
        "source": str(source),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    path = Path(stage_dir) / COST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True,
                               default=str), encoding="utf-8")
    return path


def _stage_key(payload: Mapping[str, Any], fallback: str) -> str:
    stage, shape = payload.get("stage"), payload.get("shape")
    if stage and shape:
        return f"{shape}/{stage}"
    return str(stage or fallback)


def read_campaign_costs(campaign_dir: str | Path) -> dict[str, dict[str, Any]]:
    """扫回 ``campaign_dir`` 下所有段的 ``cost.json``，按 ``<形状>/<阶段>`` 归并。

    认不出的文件（半截写、坏了、不是 JSON 对象）跳过，不抛异常——成本面板不该
    因为一行坏数据整页消失。
    """
    base = Path(campaign_dir)
    rows: dict[str, dict[str, Any]] = {}
    if not base.is_dir():
        return rows
    for path in sorted(base.rglob(COST_FILE)):
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            continue
        fallback = "/".join(path.parent.relative_to(base).parts)
        row = dict(payload)
        row["path"] = str(path)
        rows[_stage_key(row, fallback)] = row
    return rows


def _rows(rows: Any) -> list[Mapping[str, Any]]:
    """成本表 → 行的列表（映射与序列两种形状都认）。"""
    if isinstance(rows, Mapping):
        return [row for row in rows.values() if isinstance(row, Mapping)]
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        return [row for row in rows if isinstance(row, Mapping)]
    return []


def _seconds_per_block(row: Mapping[str, Any]) -> float | None:
    """一行的每块秒；优先用落盘值，其次由秒数与块数现算。"""
    per_block = _number(row.get("seconds_per_block", row.get("per_block_s")))
    if per_block is not None and per_block > 0:
        return per_block
    blocks = _number(row.get("blocks", row.get("n_blocks")))
    seconds = _number(row.get("seconds", row.get("elapsed_s")))
    if blocks is not None and blocks > 0 and seconds is not None and seconds >= 0:
        return seconds / blocks
    return None


def estimate_campaign(rows: Any, planned_blocks: int | None = None) -> dict[str, Any]:
    """按历史每块秒把 ``planned_blocks`` 折算成低/中/高三个总秒数。

    Args:
        rows: :func:`read_campaign_costs` 的返回（映射），或同形的行序列。
        planned_blocks: 计划块数；``None`` / 0 时总秒数全 0，只给出历史基准。

    Returns:
        ``{"low", "mid", "high"}`` 为总秒数（含 ``seconds_low`` 等同义键），
        ``basis_blocks`` 是这次外推用的计划块数，``basis_seconds_per_block``
        是每块秒的 ``{"low", "mid", "high", "rows"}`` 历史基准。没有任何可用的
        每块秒时，基准里的数全为 ``None``、总秒数全 0。
    """
    rates = sorted(rate for rate in (_seconds_per_block(row) for row in _rows(rows))
                   if rate is not None and rate > 0)
    planned = max(0, int(planned_blocks or 0))
    if rates:
        basis = {"low": rates[0], "mid": float(median(rates)), "high": rates[-1],
                 "rows": len(rates)}
    else:
        basis = {"low": None, "mid": None, "high": None, "rows": 0}
    totals = ({
        "low": basis["low"] * planned,
        "mid": basis["mid"] * planned,
        "high": basis["high"] * planned,
    } if rates else {"low": 0.0, "mid": 0.0, "high": 0.0})
    return {
        "low": float(totals["low"]),
        "mid": float(totals["mid"]),
        "high": float(totals["high"]),
        "seconds_low": float(totals["low"]),
        "seconds_mid": float(totals["mid"]),
        "seconds_high": float(totals["high"]),
        "basis_blocks": planned,
        "basis_seconds_per_block": basis,
    }
