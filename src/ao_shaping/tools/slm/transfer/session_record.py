"""会话收尾与段级记账：指纹落盘、检查点登记、段成本、段间暂停。

为什么单独一层
--------------
这些动作只读**已经落在会话目录里的产物**：收尾指纹来自 ``journal.json`` /
``config.json`` / ``manifest.json`` / 拟合产物，段成本来自 journal 里已经写下的
``running`` / ``complete`` 两条时间戳，段间暂停只轮询目录里的一个请求文件。
所以它们不需要设备、不需要 click，也不需要知道 ``validate`` 是怎么跑完一段的
—— 段内测量留在 runner，段外记账留在这里，两边共用同一份目录，而不是各算一套。

容错口径
--------
写侧两件事都必须容忍残缺目录 —— 一次跑完的会话不该因为"指纹算不全"而算失败；
目录不完整时也照写一条缺字段的指纹，供历史浏览器逐格显示「未记录」。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ao_shaping.tools.slm.transfer import campaign_control, cost_table
from ao_shaping.tools.slm.transfer.checkpoint_registry import register_session
from ao_shaping.tools.slm.transfer.optical_session import (
    compute_optical_fingerprint,
    load_fingerprint,
    save_fingerprint,
)

__all__ = [
    "finish_segment",
    "measured_blocks",
    "measured_peak_rates",
    "optical_session_field",
    "pause_requested",
    "record_optical_session",
    "record_stage_cost_of",
]


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------ session closure


def optical_session_field(out: Path) -> dict[str, Any]:
    """收尾时附带的光路指纹（会话目录里已经有一份才带，否则一个字段都不加）。

    指纹由 :func:`record_optical_session` 在本函数读之前写落盘；这里只读不造，
    读不到就一个字段都不加。
    """
    fingerprint = load_fingerprint(out)
    return {"optical_session": fingerprint} if fingerprint else {}


def record_optical_session(out: Path) -> None:
    """会话收尾的写侧：用已有产物算指纹落到 ``fingerprint.json``，再登记检查点索引。

    两件事都必须容忍残缺目录 —— 一次跑完的会话不该因为"指纹算不全"而算失败；
    目录不完整时也照写一条缺字段的指纹，供历史浏览器逐格显示「未记录」。
    """
    try:
        save_fingerprint(compute_optical_fingerprint(out), out)
    except Exception as exc:  # 写侧失败不得让跑完的会话失败
        logger.warning("fingerprint write skipped: {}", exc)
    try:
        register_session(out)
    except Exception as exc:  # 索引写不进去同理
        logger.warning("checkpoint index write skipped: {}", exc)


# ------------------------------------------------------- per-segment ledger


def _elapsed_seconds(started: Mapping[str, Any] | None,
                     finished: Mapping[str, Any] | None) -> float | None:
    """两条 journal 记录之间的墙钟秒数；单调时钟优先，ISO 时间戳兜底。

    成本只从 journal 已有的时间戳算，**不**另立一套计时器：这样 ``--resume`` 续跑、
    断点复盘时，成本与阶段记录永远是同一个时间轴的产物。
    """
    if not isinstance(started, Mapping) or not isinstance(finished, Mapping):
        return None
    start_ns, end_ns = started.get("monotonic_ns"), finished.get("monotonic_ns")
    if isinstance(start_ns, int) and isinstance(end_ns, int) and end_ns >= start_ns:
        return (end_ns - start_ns) / 1e9
    try:
        start = datetime.fromisoformat(str(started.get("timestamp_utc")))
        end = datetime.fromisoformat(str(finished.get("timestamp_utc")))
    except (TypeError, ValueError):
        return None
    return max(0.0, (end - start).total_seconds())


def _running_entry(journal, key: str) -> Mapping[str, Any] | None:
    """该段最近一条 ``running`` 记录（段耗时从这里起算）。"""
    for entry in reversed(getattr(journal, "entries", [])):
        if entry.get("stage") == key and entry.get("status") == "running":
            return entry
    return None


def measured_blocks(stage_dir: Path, fallback_blocks: int) -> int:
    """这段真正测过的块数：段 manifest 的行数减掉段首那一行（seed 重测）。

    manifest 缺失（或半截写）时退回配方块数 ``fallback_blocks``：宁可给一个计划值，
    也不把一段的耗时摊到 0 块上。
    """
    manifest_path = Path(stage_dir) / "manifest.json"
    if manifest_path.is_file():
        try:
            rows = _read_json(manifest_path)
        except (OSError, ValueError):
            rows = None
        if isinstance(rows, list) and rows:
            return max(0, len(rows) - 1)
    return int(fallback_blocks)


def measured_peak_rates(runs: list[Path]) -> list[tuple[float, float, float]]:
    """``(core_peak, full_scale, exposure_ms)`` for every block already measured.

    Read from the stage manifests of *this* lane, so the replay ladder is sized
    by the same bench and the same shaped spot it will replay. A missing manifest
    (``--resume`` into a half-done lane) simply contributes nothing.
    """
    observations: list[tuple[float, float, float]] = []
    for run in runs:
        manifest_path = Path(run) / "manifest.json"
        if not manifest_path.is_file():
            continue
        for row in _read_json(manifest_path):
            observations.append((row.get("core_peak"), row.get("full_scale"),
                                 row.get("exposure_ms")))
    return observations


def record_stage_cost_of(journal, key: str, stage_dir: Path, *, stage: str,
                         shape: str, blocks: int) -> float | None:
    """把刚跑完这一段（journal 里已是 ``complete``）的实测耗时写进段目录。

    返回写入的秒数（算不出来时什么也不写并返回 ``None``）。
    """
    finished = journal.entries[-1] if getattr(journal, "entries", None) else None
    seconds = _elapsed_seconds(_running_entry(journal, key), finished)
    if seconds is None:
        return None
    cost_table.record_stage_cost(stage_dir, stage=stage, shape=shape,
                                 seconds=seconds, blocks=blocks)
    return seconds


def pause_requested(journal, key: str, out: Path) -> bool:
    """段边界的协作式暂停：请求写进目录，换个页面/重开浏览器都还看得见。

    本段已经完整落盘（manifest + cost），所以标 ``paused`` 而不是 ``stopped``，
    清掉请求后照常返回 —— ``--resume`` 会把这一段当 ``reused`` 跳过。返回 ``True``
    表示调用方应当停在这一段之后（不再起 replay / 下一段）。
    """
    if not campaign_control.stage_stop_requested(out):
        return False
    journal.record(key, "paused", reason="segment-boundary pause requested")
    campaign_control.clear_stage_stop(out)
    logger.info("pause requested; stopping after {}", key)
    return True


def finish_segment(journal, key: str, stage_dir: Path, *, stage: str, shape: str,
                   fallback_blocks: int, out: Path) -> bool:
    """一段测完之后的全部记账：成本落盘 + 段边界暂停判定。

    调用方在段前写了 ``running``、段后写了 ``complete``，所以耗时与块数都从盘上的
    journal / manifest 读回（见 :func:`record_stage_cost_of`），不另立计时器。
    返回 ``True`` 表示操作员在段边界按了暂停，调用方应当停在这一段之后。
    """
    blocks = measured_blocks(stage_dir, fallback_blocks)
    seconds = record_stage_cost_of(journal, key, stage_dir, stage=stage, shape=shape,
                                   blocks=blocks)
    if seconds is not None:
        logger.info("{} done in {:.1f} s ({} blocks)", key, seconds, blocks)
    return pause_requested(journal, key, out)
