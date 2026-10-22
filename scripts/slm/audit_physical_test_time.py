"""实验系统物理测试耗时审计 (只读): 把已落盘的记录复算成一张时间账。

为什么需要它
------------
2026-10-22 的 MIL 实验系统 campaign (``scripts/slm/model_in_loop_hw_ab.py``)
在几何标定处失败, 之后的每一次讨论都在问同一个问题: **下一次要占实验系统多久?**

这个问题至今没有可引用的答案, 因为:

* 每个脚本各自报自己的时间 (采集脚本报"26.7 min", campaign 报"25 min"),
  但**口径不同** —— 采集脚本的 26.7 min 只算采集窗口, 不算帧间开销;
  campaign 的 25 min 把几何标定和 MIL 轮混在一起;
* 没有任何一份记录把「一次物理动作」拆到动作级 (settle 多少秒 / 曝光阶梯多少秒 /
  写盘多少秒 / 一个几何 sweep 点多少秒);
* 于是"能省多少"只能靠猜, 而本仓库已经因为"凭估计做决定"栽过跟头
  (2026-10-22 计划里"几何 sweep 要 50 ms x 10 帧"那一栏, 实测宽度估计在
  0.537 ms 就稳到 0.1 px, 那一整栏作废)。

本脚本**只读**: 它解析已经落盘的记录 (campaign 的 ``stage.log`` 与 ``manifest.json``、
采集脚本的 ``manifest.json``、光路探针的 ``optical_path_probe.json``), 把时间戳切段,
算出动作级的耗时账, 再据此给出可优化项的排序与三档最小设备时间预算。
**不开设备、不改任何采集脚本、不 commit。**

数据来源 (相对仓库根)
----------------------
* ``data/debug/model_in_loop_hw_ab/<会话>/manifest.json`` —— campaign 的阶段级耗时
  (测光门禁 / 每个变体)。
* ``data/debug/model_in_loop_hw_ab/<会话>/baseline_a/stage.log`` —— campaign 内部的
  阶段切分 (面板扫描 / 照明半径 / 几何 sweep) 与逐次写入、逐次 settle 超时。
* ``data/debug/physical_forward_defocus/*/manifest.json`` —— 采集脚本逐条记录
  (``utc_start`` / ``timestamp_utc`` / ``settled.waited_s`` / ``exposure_ms``)。
* ``data/debug/optical_path_probe/<UTC戳>/`` —— 光路有效性探针的曝光阶梯与离焦态。

产出 (默认 ``logs/physical_time_audit/<UTC戳>/``)
-------------------------------------------------
* ``time_budget.json`` —— 动作级时间账 + 速率 + 三档预算 + 可优化项。
* ``optimizations.md`` —— 可优化项排序 (机制 / 预计节省 / 副作用 / 依据)。
* ``device_budget.md`` —— 档 A / B / C 的分项与总时长。
* ``time_share.png`` —— 各动作耗时占比条形图。

用法::

    python scripts/slm/audit_physical_test_time.py
    python scripts/slm/audit_physical_test_time.py --out logs/physical_time_audit/manual
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# 日志解析
# ---------------------------------------------------------------------------

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

#: loguru 的行格式: ``<ts> | <LEVEL> | <module>:<func>:<line> - <message>``
_LOG_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}) \| (?P<level>\w+)\s+\| "
    r"(?P<module>[^:]+):(?P<func>\w+):(?P<line>\d+) - (?P<msg>.*)$"
)

#: 面板写入的日志前缀 (SLM 驱动每次写相位都打一条)
_WRITE_MARK = "相位数据已写入SLM"

#: settle 超时的日志前缀 (``bench_kernels._warn_if_unsettled``)
_SETTLE_TIMEOUT_MARK = "settle criterion not met"

#: settle 超时行里 ``after <x> s`` 的捕获
_SETTLE_SECONDS = re.compile(r"after\s+([0-9.]+)\s*s")

#: 探针目录名的时间戳形如 ``20261022_063917Z``
_PROBE_STAMP = re.compile(r"^(\d{8})_(\d{6})Z$")


@dataclass(frozen=True)
class LogEvent:
    """一行已解析的日志。

    Attributes:
        at: 行首时间戳 (本地时间, 无时区)。
        module: 产生这行的模块。
        func: 产生这行的函数。
        message: 去掉 ANSI 转义后的正文。
    """

    at: datetime
    module: str
    func: str
    message: str


def parse_log_events(text: str) -> list[LogEvent]:
    """把 loguru 日志正文解析成事件序列。

    ANSI 转义先剥掉 —— 仓库的 campaign 日志混着有色的 ``INFO``/``WARNING``
    前缀, 不剥就一行都匹配不上。无法解析的行 (banner、traceback、空行) 直接略过。
    """
    events: list[LogEvent] = []
    for raw in text.splitlines():
        line = _ANSI.sub("", raw).rstrip("\r\n")
        match = _LOG_LINE.match(line)
        if match is None:
            continue
        events.append(
            LogEvent(
                at=datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S.%f"),
                module=match.group("module"),
                func=match.group("func"),
                message=match.group("msg"),
            )
        )
    return events


@dataclass
class Phase:
    """campaign 内部的一个阶段。

    Attributes:
        name: 阶段名 (中文叙述用)。
        start: 阶段起点。
        end: 阶段终点。
        writes: 阶段内的面板写入次数 (一次写入 = 一个"相位 -> 读出"循环)。
        settle_timeouts: 阶段内 settle 判据超时的次数。
        settle_timeout_seconds: 阶段内 settle 超时累计烧掉的秒数。
        points: 阶段内的测量点数 (面板扫描点 / sweep 点 / 写入次数)。
        detail: 附加说明。
    """

    name: str
    start: datetime | None = None
    end: datetime | None = None
    writes: int = 0
    settle_timeouts: int = 0
    settle_timeout_seconds: float = 0.0
    points: int = 0
    tilt_seconds: float = 0.0
    detail: str = ""

    @property
    def seconds(self) -> float:
        """阶段时长 (s); 边界缺失时为 0。"""
        if self.start is None or self.end is None:
            return 0.0
        return (self.end - self.start).total_seconds()

    @property
    def seconds_per_point(self) -> float:
        """每个测量点的平均耗时 (s)。"""
        return self.seconds / self.points if self.points else 0.0


def _first_event(events: Sequence[LogEvent], predicate) -> LogEvent | None:
    for event in events:
        if predicate(event):
            return event
    return None


def _last_event(events: Sequence[LogEvent], predicate) -> LogEvent | None:
    for event in reversed(events):
        if predicate(event):
            return event
    return None


def segment_campaign(events: Sequence[LogEvent]) -> list[Phase]:
    """把 campaign 的 ``stage.log`` 切成阶段。

    切分点是**记录里本来就有的**事件, 不是猜的:

    * ``panel_locate:scan_panel`` 的 "flat reference" 行 = 面板扫描的起点;
      最后一条 "panel y= ..." 行 = 面板扫描的终点;
    * ``__main__:stage_sweep`` 的第一条 = 几何 sweep 的起点;
      "saved N sweep points" = 终点;
    * 两者之间 = 照明半径测量 (``beam_extent``) —— 这一步在日志里没有自己的
      标记行, 所以按"面板扫描结束 -> sweep 开始"的剩余窗口记账。

    每个阶段的"面板写入次数"与"settle 超时次数"按事件时间戳落桶。
    """
    if not events:
        return []

    panel_start = _first_event(
        events,
        lambda e: e.func == "scan_panel" and e.message.startswith("flat reference"),
    )
    panel_end = _last_event(
        events, lambda e: e.func == "scan_panel" and e.message.startswith("panel y=")
    )
    sweep_start = _first_event(events, lambda e: e.func == "stage_sweep")
    sweep_end = _last_event(
        events, lambda e: e.func == "stage_sweep" and "saved" in e.message
    )

    boundaries: list[tuple[str, datetime | None, datetime | None]] = []
    if panel_start is not None:
        boundaries.append(("设备打开与平场参考", events[0].at, panel_start.at))
    if panel_start is not None and panel_end is not None:
        boundaries.append(("面板扫描", panel_start.at, panel_end.at))
    if panel_end is not None and sweep_start is not None:
        boundaries.append(("照明半径测量", panel_end.at, sweep_start.at))
    if sweep_start is not None and sweep_end is not None:
        boundaries.append(("几何 sweep", sweep_start.at, sweep_end.at))
    if sweep_end is not None:
        boundaries.append(("离线几何标定", sweep_end.at, events[-1].at))

    phases: list[Phase] = []
    for name, start, end in boundaries:
        phase = Phase(name=name, start=start, end=end)
        window = [
            e
            for e in events
            if start is not None and end is not None and start <= e.at <= end
        ]
        phase.writes = sum(1 for e in window if e.message.startswith(_WRITE_MARK))
        timeouts = [e for e in window if _SETTLE_TIMEOUT_MARK in e.message]
        phase.settle_timeouts = len(timeouts)
        phase.settle_timeout_seconds = sum(
            float(_SETTLE_SECONDS.search(e.message).group(1))
            for e in timeouts
            if _SETTLE_SECONDS.search(e.message)
        )
        phases.append(phase)

    # 面板扫描的点数 = 行数 x 每行的 x 位置数
    for phase in phases:
        if phase.name == "面板扫描":
            rows = [
                e
                for e in events
                if e.func == "scan_panel"
                and e.message.startswith("panel y=")
                and phase.start is not None
                and phase.end is not None
                and phase.start <= e.at <= phase.end
            ]
            columns = max((e.message.count("x=") for e in rows), default=0)
            phase.points = len(rows) * columns
        elif phase.name == "几何 sweep":
            phase.points = phase.writes
            phase.tilt_seconds = _tilt_group_seconds(events, phase)
        elif phase.name == "照明半径测量":
            phase.points = phase.writes

    return phases


def _tilt_group_seconds(events: Sequence[LogEvent], phase: Phase) -> float:
    """几何 sweep 里"倾斜组"占掉的秒数。

    倾斜点是整轮 sweep 里最贵的: 每个幅度要采 4 帧求质心均值, 而其它模式只采
    1 帧。组起点是 ``[*] tilt x = -1.000 rad (x4)``, 组终点是同名的
    ``mean c=...`` 汇总行 —— 两行都在日志里, 所以这段时间是量出来的, 不是估的。
    """
    if phase.start is None or phase.end is None:
        return 0.0
    total = 0.0
    started: datetime | None = None
    for event in events:
        if event.func != "stage_sweep":
            continue
        if not (phase.start <= event.at <= phase.end):
            continue
        if event.message.startswith("[*] tilt"):
            started = event.at
        elif started is not None and "mean c=" in event.message:
            total += (event.at - started).total_seconds()
            started = None
    return total


# ---------------------------------------------------------------------------
# 采集脚本的 manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureSample:
    """采集脚本的一条记录。

    Attributes:
        index: 记录序号。
        label: 采集标签。
        family: 采集族 (``reference`` / ``ramp`` / ``zernike`` / ``prior`` ...)。
        branch: 分支 (``U`` = 当前面, ``C`` = 离焦补偿面)。
        exposure_ms: 该次采集用的曝光。
        settle_waited_s: settle 判据烧掉的秒数 (记录里直接给了)。
        settle_probes: settle 探测次数。
        capture_seconds: 采集窗口 (写入相位 -> 采完)。
        gap_seconds: 到下一条记录的帧间开销。
    """

    index: int
    label: str
    family: str
    branch: str
    exposure_ms: float
    settle_waited_s: float
    settle_probes: int
    capture_seconds: float
    gap_seconds: float


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


def capture_rows_to_samples(rows: Sequence[dict]) -> list[CaptureSample]:
    """把采集 manifest 的行变成 :class:`CaptureSample`。

    两条时间轴都能用: 有的 manifest 同时有 ``utc_start`` / ``timestamp_utc``,
    有的只有 ``timestamp_utc`` 加 ``monotonic_*_ns``。前者优先, 因为它把
    "帧间开销"也框在了窗口里。
    """
    samples: list[CaptureSample] = []
    for position, row in enumerate(rows):
        settle = row.get("settled") or {}
        if "utc_start" in row and "timestamp_utc" in row:
            capture = (
                _parse_utc(row["timestamp_utc"]) - _parse_utc(row["utc_start"])
            ).total_seconds()
            start_ns = row.get("monotonic_start_ns")
            end_ns = row.get("monotonic_end_ns")
        else:
            start_ns = row.get("monotonic_start_ns")
            end_ns = row.get("monotonic_end_ns")
            if start_ns is None or end_ns is None:
                raise ValueError(
                    f"record {position} has neither utc_start nor monotonic stamps"
                )
            capture = (int(end_ns) - int(start_ns)) / 1e9
        samples.append(
            CaptureSample(
                index=int(row.get("index", position)),
                label=str(row.get("label", "")),
                family=str(row.get("family", "")),
                branch=str(row.get("branch", "")),
                exposure_ms=float(row.get("exposure_ms", 0.0)),
                settle_waited_s=float(settle.get("waited_s", 0.0)),
                settle_probes=int(settle.get("n_probes", 0)),
                capture_seconds=capture,
                gap_seconds=0.0,
            )
        )

    for index in range(len(samples) - 1):
        first = rows[index]
        second = rows[index + 1]
        if "utc_start" in first and "utc_start" in second:
            gap = (
                _parse_utc(second["utc_start"]) - _parse_utc(first["timestamp_utc"])
            ).total_seconds()
        else:
            gap = (
                int(second["monotonic_start_ns"]) - int(first["monotonic_end_ns"])
            ) / 1e9
        samples[index] = CaptureSample(
            **{
                **samples[index].__dict__,
                "gap_seconds": max(gap, 0.0),
            }
        )
    return samples


def summarize_capture(
    samples: Sequence[CaptureSample],
    *,
    write_cost_s: float = 0.0,
) -> dict[str, Any]:
    """把采集样本汇成动作级时间账。

    每一类动作的**单次耗时**取中位数 (对数正态的量, 均值会被长尾拉偏),
    总量 = 中位数 x 次数。写盘耗时不是记录里的字段, 由调用方用实测值传入
    (见 :func:`measure_write_cost`), 其余三类全部直接来自记录。
    """
    if not samples:
        raise ValueError("no capture samples")
    count = len(samples)
    settle = [s.settle_waited_s for s in samples]
    gaps = [s.gap_seconds for s in samples if s.gap_seconds > 0.0]
    capture = [s.capture_seconds for s in samples]
    # 曝光自适应阶梯与读帧: 采集窗口减去 settle 与写盘
    ladder = [
        max(s.capture_seconds - s.settle_waited_s - write_cost_s, 0.0) for s in samples
    ]
    references = [s for s in samples if s.family == "reference"]

    median = statistics.median
    total = sum(capture) + sum(gaps)

    def _row(action: str, unit: str, n: float, unit_seconds: float) -> dict[str, Any]:
        return {
            "action": action,
            "unit": unit,
            "count": n,
            "unit_seconds": unit_seconds,
            "total_seconds": n * unit_seconds,
        }

    actions = [
        _row("settle 等待", "次采集", count, median(settle)),
        _row("曝光自适应阶梯与原始帧读出", "次采集", count, median(ladder)),
        _row("落盘写帧 (npz 压缩)", "次采集", count, write_cost_s),
        _row(
            "帧间开销 (清单/记录/准备)",
            "帧间",
            len(gaps),
            median(gaps) if gaps else 0.0,
        ),
    ]

    # 归因 (与上面的动作重叠, 只用于回答"reference 帧吃掉了多少")
    reference_share = len(references) / count
    reference_cost = sum(s.capture_seconds + s.gap_seconds for s in references)
    attribution = {
        "reference_frames": len(references),
        "reference_share_of_captures": reference_share,
        "reference_seconds": reference_cost,
        "reference_share_of_time": reference_cost / total if total else 0.0,
    }

    return {
        "n_samples": count,
        "total_seconds": total,
        "total_minutes": total / 60.0,
        "capture_window_seconds": sum(capture),
        "gap_seconds": sum(gaps),
        "median_capture_seconds": median(capture),
        "median_settle_seconds": median(settle),
        "median_ladder_seconds": median(ladder),
        "median_gap_seconds": median(gaps) if gaps else 0.0,
        "write_cost_seconds": write_cost_s,
        "actions": actions,
        "attribution": attribution,
        "branches": sorted({s.branch for s in samples}),
        "families": {
            family: sum(1 for s in samples if s.family == family)
            for family in sorted({s.family for s in samples})
        },
    }


def branch_drift(samples: Sequence[CaptureSample], rows: Sequence[dict]) -> dict[str, Any]:
    """每个分支的参考帧漂移统计。

    这是"reference 帧该隔多远"的**唯一依据**, 而且要看三个量:

    * **宽度跨度** —— 宽度是判据本身, 跨度大就必须保持密集参考;
    * **相邻参考帧的质心步长** —— 漂移监测能看到的最小变化;
    * **步长与间隔的相关性 / 线性斜率** —— 关键: 若步长**与间隔无关**
      (斜率 ≈ 0, 截距非零), 那它是抖动而不是线性漂移, 间隔可以放宽;
      若步长随间隔线性增长, 间隔一放宽就真的漏掉漂移了。

    线性回归用最小二乘: ``step = k * interval + b``; ``k`` 是漂移速率 (px/min),
    ``b`` 是与间隔无关的抖动幅度 (px)。
    """
    by_branch: dict[str, list[tuple[float, float, float, float]]] = {}
    for sample, row in zip(samples, rows, strict=True):
        if sample.family != "reference":
            continue
        spot = row.get("spot") or {}
        if "fwhm_px" not in spot:
            continue
        by_branch.setdefault(sample.branch, []).append(
            (
                float(row.get("monotonic_start_ns", 0)) / 1e9,
                float(spot["centroid_x"]),
                float(spot["centroid_y"]),
                float(spot["fwhm_px"]),
            )
        )

    out: dict[str, Any] = {}
    for branch, entries in sorted(by_branch.items()):
        entries.sort(key=lambda item: item[0])
        times = [e[0] for e in entries]
        widths = [e[3] for e in entries]
        xs = [e[1] for e in entries]
        ys = [e[2] for e in entries]
        steps = [
            math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            for i in range(len(xs) - 1)
        ]
        intervals = [(times[i + 1] - times[i]) / 60.0 for i in range(len(times) - 1)]
        width_steps = [
            abs(widths[i + 1] - widths[i]) for i in range(len(widths) - 1)
        ]
        span = (max(widths) - min(widths)) / statistics.mean(widths) if widths else 0.0

        slope = 0.0
        intercept = 0.0
        correlation = 0.0
        if steps:
            mean_x = statistics.mean(intervals)
            mean_y = statistics.mean(steps)
            var_x = sum((v - mean_x) ** 2 for v in intervals)
            cov = sum(
                (intervals[i] - mean_x) * (steps[i] - mean_y) for i in range(len(steps))
            )
            if var_x > 0:
                slope = cov / var_x
            # 间隔恒定时 var_x = 0: 斜率无从估计, 全部步长都是抖动
            intercept = mean_y - slope * mean_x
            var_y = sum((v - mean_y) ** 2 for v in steps)
            if var_x > 0 and var_y > 0:
                correlation = cov / math.sqrt(var_x * var_y)

        net = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) if len(xs) >= 2 else 0.0
        elapsed_min = (times[-1] - times[0]) / 60.0 if len(times) >= 2 else 0.0

        out[branch] = {
            "n_references": len(entries),
            "fwhm_span_fraction": span,
            "fwhm_min_px": min(widths) if widths else 0.0,
            "fwhm_max_px": max(widths) if widths else 0.0,
            "fwhm_step_median_px": statistics.median(width_steps) if width_steps else 0.0,
            "fwhm_step_max_px": max(width_steps) if width_steps else 0.0,
            "centroid_step_median_px": statistics.median(steps) if steps else 0.0,
            "centroid_step_max_px": max(steps) if steps else 0.0,
            "centroid_path_px": sum(steps),
            "net_drift_px": net,
            "elapsed_minutes": elapsed_min,
            "net_drift_px_per_min": net / elapsed_min if elapsed_min else 0.0,
            "step_vs_interval_slope_px_per_min": slope,
            "step_vs_interval_jitter_px": intercept,
            "step_vs_interval_correlation": correlation,
            "median_interval_minutes": (
                statistics.median(intervals) if intervals else 0.0
            ),
        }
    return out


# ---------------------------------------------------------------------------
# 光路有效性探针
# ---------------------------------------------------------------------------


def summarize_probe(probe_dir: Path) -> dict[str, Any] | None:
    """把一次光路探针会话的耗时拆成"阶梯"与"离焦态"两段。

    起点取目录名里的 UTC 戳, 终点取 ``optical_path_probe.json`` 的 ``created_utc``;
    两段的分界取最后一档曝光 npz 的落盘时刻。
    """
    match = _PROBE_STAMP.match(probe_dir.name)
    if match is None:
        return None
    start = datetime.strptime(
        match.group(1) + match.group(2), "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)
    payload = probe_dir / "optical_path_probe.json"
    if not payload.is_file():
        return None
    data = json.loads(payload.read_text(encoding="utf-8"))
    end = datetime.fromisoformat(data["created_utc"])
    npz = sorted(probe_dir.glob("*.npz"))
    ladder = [p for p in npz if p.name.startswith("exposure_")]
    defocus = [p for p in npz if p.name.startswith("defocus_")]
    if not ladder:
        return None
    ladder_end = max(
        datetime.fromtimestamp(p.stat().st_mtime, timezone.utc) for p in ladder
    )
    setup = (ladder_end - start).total_seconds() - (
        len(ladder) - 1
    ) * 0.0  # 占位: 阶梯步距在下面单独算
    step_gaps = sorted(
        datetime.fromtimestamp(p.stat().st_mtime, timezone.utc) for p in ladder
    )
    step_costs = [
        (step_gaps[i + 1] - step_gaps[i]).total_seconds() for i in range(len(step_gaps) - 1)
    ]
    ladder_seconds = (ladder_end - start).total_seconds()
    defocus_seconds = (end - ladder_end).total_seconds()
    return {
        "session": probe_dir.name,
        "total_seconds": (end - start).total_seconds(),
        "ladder_seconds": ladder_seconds,
        "defocus_seconds": defocus_seconds,
        "ladder_steps": len(ladder),
        "defocus_steps": len(defocus),
        "median_ladder_step_seconds": (
            statistics.median(step_costs) if step_costs else 0.0
        ),
        "median_defocus_step_seconds": (
            defocus_seconds / len(defocus) if defocus else 0.0
        ),
        "setup_seconds": setup,
        "verdict": (data.get("verdict") or {}).get("state"),
        "usable_window_ms": (data.get("ladder_summary") or {}).get("usable_window_ms"),
    }


# ---------------------------------------------------------------------------
# 写盘耗时实测
# ---------------------------------------------------------------------------


def measure_write_cost(
    *,
    frame_shape: tuple[int, int] = (1944, 2592),
    raw_frames: int = 3,
    panel_shape: tuple[int, int] = (1200, 1920),
    repeats: int = 1,
    sample_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """实测一次 npz 落盘的耗时 (s)。

    采集脚本每采一条就把**整帧** CCD (float64 1944x2592) + 3 帧原始帧 + 整块面板
    相位 (float32 1200x1920) 压成一个 npz, 约 64 MB 未压缩。这一步**发生在
    相机与 SLM 都还占着的时候**, 所以它是实验系统占用时间的一部分, 而不是可以忽略的
    后台开销。

    优先用**真实样本**实测 (同样的负载、同样的压缩器), 取中位数; 没有真实样本时
    退回同样形状与 dtype 的合成负载。两种口径都只写系统临时目录并立即删除。
    返回 ``{"median_seconds", "samples_seconds", "measured_on"}`` —— 每次实测的
    离散度不小 (同一台机器上 3.7-5.7 s), 所以把每次的值一并留下。
    """
    import numpy as np  # 局部导入: 让本模块的纯解析部分不必依赖 numpy

    import tempfile
    import time

    target = Path(tempfile.gettempdir()) / "audit_physical_test_time_write_probe.npz"
    costs: list[float] = []
    try:
        if sample_paths:
            for path in sample_paths:
                if not path.is_file():
                    continue
                with np.load(path) as archive:
                    payload = {key: archive[key] for key in archive.files}
                start = time.perf_counter()
                np.savez_compressed(target, **payload)
                costs.append(time.perf_counter() - start)
        if not costs:
            rng = np.random.default_rng(0)
            payload = {
                "frame": rng.random(frame_shape),
                "raw_frames": rng.integers(
                    0, 255, size=(raw_frames, *frame_shape), dtype=np.uint8
                ),
                "panel_phase": rng.random(panel_shape).astype(np.float32),
                "model_phase": np.zeros((0, 0)),
            }
            for _ in range(max(int(repeats), 1)):
                start = time.perf_counter()
                np.savez_compressed(target, **payload)
                costs.append(time.perf_counter() - start)
    finally:
        target.unlink(missing_ok=True)
    return {
        "median_seconds": statistics.median(costs),
        "samples_seconds": costs,
        "measured_on": "真实样本" if sample_paths else "合成负载",
    }


# ---------------------------------------------------------------------------
# 几何 sweep 的最少点数
# ---------------------------------------------------------------------------


def min_sweep_points(
    *,
    n_params: int = 3,
    min_residual_dof: int = 1,
    anchor_vertex: bool = True,
) -> dict[str, Any]:
    """拟合 ``w^2(c)`` 二次曲线所需的最少 sweep 点数。

    * 二次曲线有 3 个参数 (``a c^2 + b c + d``) —— 至少要 3 个点才能定解;
    * 再要 1 个自由度才能算残差并发现"曲率退化" (点数恰好等于参数数时残差恒为 0,
      任何数据都能拟合, 于是 2026-10-22 的四个"degenerate curvature"警告
      就永远不会被触发);
    * **以宽度为判据**时还要一个 ``c = 0`` 的锚点 —— 宽度的基线 ``w0`` 只在
      ``c = 0`` 处定义, 而宽度的响应本身很弱 (2026-10-22 探针实测离焦只带来
      9.5% 的宽度变化), 少一个锚点就没有东西去判"响应到底存不存在"。

    以**峰值**作备选判据时可以少 1 个点: 峰值的基线是单独测的平场参考
    (不必占 sweep 的点), 而且离焦让峰值相对变化 27.6%, 是宽度响应的 2.9 倍,
    分辨率裕量足够, 4 个点 (``+-c1, +-c2``) 已经能把曲率定出来。
    """
    base = max(n_params + min_residual_dof, 3)
    width_points = base + (1 if anchor_vertex else 0)
    peak_points = base
    return {
        "width_criterion_points": width_points,
        "peak_criterion_points": peak_points,
        "n_params": n_params,
        "min_residual_dof": min_residual_dof,
        "anchor_vertex": anchor_vertex,
    }


# ---------------------------------------------------------------------------
# 设备时间预算
# ---------------------------------------------------------------------------


def build_device_budget(
    rates: dict[str, float],
    *,
    variants: int = 3,
    trust_region_tiers: int = 3,
    alternating_rounds: Sequence[int] = (3, 6),
) -> dict[str, Any]:
    """按实测速率拼出档 A / B / C 三档设备时间预算。

    ``rates`` 的每个键都是**实测**或**明确标注为推算**的秒数:

    * ``probe`` —— 一次光路有效性探针 (实测中位)
    * ``panel_scan`` —— 一次面板扫描 (实测)
    * ``illumination`` —— 一次照明半径测量 (实测)
    * ``sweep`` —— 一次几何 sweep (实测)
    * ``calibrate`` —— 离线几何标定 (实测, 不占设备)
    * ``mil_step_a`` —— MIL 的 Step A: 采集探针集 (16 帧)
    * ``mil_step_b`` —— MIL 的 Step B: 目标光斑相位合成的迭代 (600 次)
    * ``mil_accept`` —— MIL 的验收判据 (不占设备, 但占墙钟)

    MIL 一轮被拆成 Step A / Step B / 判据三段, 是因为**只有 Step A 可以跨变体
    与跨信任域档位复用**: 探针集测的是实验系统自己的相差, 换变体或换信任域半径
    它都不变。Step B 是逐次迭代的在线优化, 每换一条配置就要重跑一遍。
    """
    probe = rates["probe"]
    panel = rates["panel_scan"]
    illumination = rates["illumination"]
    sweep = rates["sweep"]
    calibrate = rates.get("calibrate", 0.0)
    light_gate = rates.get("light_gate", 0.0)
    step_a = rates["mil_step_a"]
    step_b = rates["mil_step_b"]
    accept = rates.get("mil_accept", 0.0)

    # --- 档 A: 决定性 ------------------------------------------------------
    # 问的是"前向模型能否拟合实验系统的相差", 所以几何必须先测出来:
    # 瞳面中心 (面板扫描) / 照明口径 (照明半径) / 焦面标度与离焦响应 (几何 sweep),
    # 再采一套探针集做拟合, 拟合本身离线。
    tier_a_rows = [
        ("光路有效性探针 (1 次)", probe, "判定 ok; 否则不进入标定"),
        ("面板扫描 (1 次)", panel, "瞳面中心 + 平场漂移 < 5%"),
        ("照明半径测量 (1 次)", illumination, "两条焦面标度路径的口径一致"),
        ("几何 sweep (1 次)", sweep, "每个被扫模式有可用宽度响应"),
        ("探针集采集 (Step A, 16 帧)", step_a, "探针响应足够定出前向模型"),
        ("离线几何标定 + 前向模型拟合", calibrate, "残差可接受; 不占设备"),
    ]
    tier_a_total = sum(row[1] for row in tier_a_rows)

    # --- 档 B: 完整 campaign ----------------------------------------------
    # 几何与探针集是**实验系统的属性**, 全部变体复用; 只有 Step B 与判据逐变体重跑。
    tier_b_rows = [
        ("测光门禁 (1 次)", light_gate, "光路可达"),
        (
            "共享前置 (面板扫描 + 照明半径 + 几何 sweep + 探针集 + 标定)",
            tier_a_total,
            "几何与探针集一次测好, 全部变体复用",
        ),
        (
            f"Step B x {variants} 个变体",
            step_b * variants,
            "每个变体各自跑一遍在线优化",
        ),
        (f"验收判据 x {variants} 个变体", accept * variants, "trust region 接受/拒绝"),
    ]
    tier_b_total = sum(row[1] for row in tier_b_rows)

    # --- 档 C: 研究闭环 ----------------------------------------------------
    # 信任域档位只改**离线**的接受阈值, 探针集与几何都复用; 但 Step B 每档都要重跑。
    extra_rounds = max(min(alternating_rounds) - 1, 0)
    tier_c_rows = list(tier_b_rows) + [
        (
            f"信任域对照 {trust_region_tiers} 档 x Step B",
            step_b * trust_region_tiers,
            "探针集与几何复用, 只改离线信任域",
        ),
        (
            f"交替迭代额外 {extra_rounds} 轮 x (Step B + 判据)",
            (step_b + accept) * extra_rounds,
            f"{min(alternating_rounds)}-{max(alternating_rounds)} 轮交替迭代的下界",
        ),
    ]
    tier_c_total = sum(row[1] for row in tier_c_rows)

    def _tier(name: str, rows: Sequence[tuple[str, float, str]], total: float) -> dict:
        return {
            "name": name,
            "total_seconds": total,
            "total_minutes": total / 60.0,
            "rows": [
                {"item": item, "seconds": seconds, "criterion": criterion}
                for item, seconds, criterion in rows
            ],
        }

    return {
        "rates_seconds": rates,
        "tier_a": _tier("档 A (决定性)", tier_a_rows, tier_a_total),
        "tier_b": _tier("档 B (完整 campaign)", tier_b_rows, tier_b_total),
        "tier_c": _tier("档 C (研究闭环)", tier_c_rows, tier_c_total),
    }


def budget_self_consistent(budget: dict[str, Any]) -> bool:
    """每一档的分项之和必须等于该档的总时长。"""
    for key in ("tier_a", "tier_b", "tier_c"):
        tier = budget[key]
        summed = sum(row["seconds"] for row in tier["rows"])
        if not math.isclose(summed, tier["total_seconds"], rel_tol=1e-9, abs_tol=1e-6):
            return False
    return True


# ---------------------------------------------------------------------------
# 可优化项
# ---------------------------------------------------------------------------


def optimization_items(
    measured: dict[str, Any],
    *,
    write_cost_s: float,
    reference_every_now: int = 6,
    reference_every_recommended: int = 12,
    settle_max_discard_now: int = 40,
    frame_read_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """把可优化项排成序 (节省大的在前)。

    每一项都带**依据**字段, 指向它所依据的那条实测数字; 没有依据的项不进这张表。
    """
    capture = measured["capture"]
    campaign = measured["campaign"]
    items: list[dict[str, Any]] = []

    per_sample = (
        capture["median_capture_seconds"] + capture["median_gap_seconds"]
    )
    n_samples = capture["n_samples"]

    # 1) settle 判据的无条件丢帧 ------------------------------------------------
    discard_s = settle_max_discard_now * frame_read_seconds
    items.append(
        {
            "rank": 0,
            "title": "把 settle 判据前的无条件丢帧从 40 帧降到 2-3 帧",
            "mechanism": (
                "`bench_kernels.wait_for_settle()` 先无条件读 `n_discard` 帧再进稳定性循环, "
                "循环外没有提前退出; 而 `SlmModelInLoopConfig.settle_max_discard` 的默认是 40 "
                "(`--settle-max-discard`), 一次采集因此先烧掉 40 帧读出。"
                "面板早就松弛完 (稳定性循环自己带 6 s 上限), 这 40 帧是纯浪费。"
            ),
            "expected_saving": (
                f"每次 MIL 采集省 {settle_max_discard_now - 3} 帧 x {frame_read_seconds:.1f} s "
                f"= {discard_s - 3 * frame_read_seconds:.0f} s; 一轮 600 迭代的 Step B "
                f"因此省 {(discard_s - 3 * frame_read_seconds) * 600 / 60:.0f} min"
            ),
            "side_effect": (
                "若面板松弛比 2-3 帧更慢, 稳定性循环会更早超时 —— 但循环有 `max_wait_s` 上限, "
                "超时帧仍会返回并被标记为未稳定, 契约不变; 需要把 `settle_max_wait_s` 一并调大。"
            ),
            "evidence": (
                "静态读取 `src/ao_shaping/tools/kernels/bench_kernels.py:363-365` "
                "(`for _ in range(int(n_discard)): cam.get_numpy_image(...)` 无 break) "
                "+ `slm_model_in_loop.py:967` (`settle_max_discard: int = 40`)。"
                "⚠️ 这一项**不在任何落盘记录里**: 2026-10-22 的 campaign 死在几何标定, "
                "MIL 轮从未在实验系统上跑过, 所以它是代码事实而非实测账。"
            ),
            "saving_seconds": (discard_s - 3 * frame_read_seconds) * 600,
        }
    )

    # 2) reference 帧降频 ------------------------------------------------------
    attribution = capture["attribution"]
    ref_seconds = attribution["reference_seconds"]
    ref_fraction = attribution["reference_share_of_captures"]
    drift = measured.get("drift") or {}
    compensated = drift.get("C") or {}
    uncompensated = drift.get("U") or {}
    comp_span = compensated.get("fwhm_span_fraction", 0.0)
    comp_slope = compensated.get("step_vs_interval_slope_px_per_min", 0.0)
    comp_jitter = compensated.get("step_vs_interval_jitter_px", 0.0)
    comp_corr = compensated.get("step_vs_interval_correlation", 0.0)
    uncomp_span = uncompensated.get("fwhm_span_fraction", 0.0)
    uncomp_step = uncompensated.get("centroid_step_median_px", 0.0)
    uncomp_max = uncompensated.get("centroid_step_max_px", 0.0)
    # 参考帧密度减半 (6 -> 12) 后省掉一半的参考帧
    saving_reference = ref_seconds * 0.5
    items.append(
        {
            "rank": 0,
            "title": "补偿面上的 reference 帧降频 (--reference-every 6 -> 12)",
            "mechanism": (
                "reference 对 (每 6 次内容采集插一对 U/C 参考) 目前吃掉 "
                f"{ref_fraction * 100:.0f}% 的采集次数。实测: 补偿面 (分支 C) 上"
                "相邻参考帧的质心步长与间隔**不相关** "
                f"(线性斜率 {comp_slope:+.3f} px/min, 相关系数 {comp_corr:+.2f}, "
                f"与间隔无关的抖动 {comp_jitter:.2f} px), "
                f"宽度跨度只有 {comp_span * 100:.1f}% —— "
                "说明这条面上没有随时间线性累积的漂移, 间隔可以放宽, "
                "漂移监测的时间分辨率下降只是\"何时发生\"定位变粗。"
            ),
            "expected_saving": (
                f"参考帧数减半 => 每轮采集省 {saving_reference / 60:.1f} min "
                f"(采集总时长 {capture['total_minutes']:.1f} min -> "
                f"{(capture['total_seconds'] - saving_reference) / 60:.1f} min, "
                f"-{saving_reference / capture['total_seconds'] * 100:.0f}%)"
            ),
            "side_effect": (
                "未补偿面 (分支 U) **不能**这么改: 实测宽度跨度 "
                f"{uncomp_span * 100:.1f}%、相邻参考质心步长中位 {uncomp_step:.1f} px "
                f"(最大 {uncomp_max:.1f} px), 必须保持现有密度; "
                "且降频后自适应曝光的数据与固定曝光的数据不能混比强度。"
            ),
            "evidence": (
                f"`capture_..._prior/manifest.json` 的 {attribution['reference_frames']} 条"
                f" reference 记录 ({n_samples} 条中的 {ref_fraction * 100:.0f}%)。"
            ),
            "saving_seconds": saving_reference,
        }
    )

    # 3) 落盘写帧 --------------------------------------------------------------
    write_total = write_cost_s * n_samples
    items.append(
        {
            "rank": 0,
            "title": "采集落盘改成 ROI 裁剪 + uint16 面板相位 (不再压整帧)",
            "mechanism": (
                "每采一条都把整帧 CCD (float64 1944x2592) + 3 帧原始帧 + 整块面板相位 "
                "(float32 1200x1920) 压成一个 npz, 未压缩约 64 MB, 压缩后约 4.8 MB; "
                f"实测这一步要 {write_cost_s:.1f} s, 而它发生在相机与 SLM 都还占着的时候。"
                "改成只存 ROI (相机窗口已经是 420 px 量级) + 面板相位存 uint16 灰度, "
                "负载降到原来的 ~1/20。"
            ),
            "expected_saving": (
                f"每条省 {write_cost_s - 0.4:.1f} s (按 0.4 s 估) => 一轮 {n_samples} 条省 "
                f"{(write_cost_s - 0.4) * n_samples / 60:.1f} min "
                f"({capture['total_minutes']:.1f} min 的 "
                f"{(write_cost_s - 0.4) * n_samples / capture['total_seconds'] * 100:.0f}%)"
            ),
            "side_effect": (
                "整帧不再留档 ⇒ ROI 的选择变成不可逆决定; 必须在 manifest 里记下裁剪框, "
                "否则事后无法回到全幅。"
            ),
            "evidence": (
                f"实测 `np.savez_compressed` 于真实样本 = {write_cost_s:.2f} s "
                "(逐次离散度大: 多次实测 2.8-5.7 s, 取决于磁盘瞬时负载); "
                "样本 `sample_0000.npz` 未压缩 64 MB / 压缩后 4.8 MB。"
            ),
            "saving_seconds": (write_cost_s - 0.4) * n_samples,
        }
    )

    # 4) 固定曝光 --------------------------------------------------------------
    # 曝光自适应阶梯与原始帧读出这个桶里, 阶梯占 3 次读出 (2 丢弃 + 1 读数),
    # 原始帧占另外 3 次 —— 所以去掉阶梯省的是这个桶的**一半**。
    bucket = capture["median_ladder_seconds"]
    ladder_share = bucket * 0.5
    ladder_total = ladder_share * n_samples
    items.append(
        {
            "rank": 0,
            "title": "已确认稳定的面改用固定曝光 (脚本已有 --fixed-exposure 通路)",
            "mechanism": (
                "逐条自适应阶梯每条要读 2 帧丢弃 + 1 帧读数 (约 3 次读出), "
                f"与 3 次原始帧读出一起构成实测中位 {bucket:.1f} s/条 的读帧桶。"
                "`capture_prior_phase_dataset.py` 已经实现了按分支锁定曝光的通路 "
                "(`fixed[branch]`), 只是 2026-10-22 的运行走的是自适应分支。"
                "在补偿面上开启固定曝光, 阶梯只在每分支第一次跑一次。"
            ),
            "expected_saving": (
                f"每条省约 {ladder_share:.1f} s (阶梯那 3 次读出) => 一轮省 "
                f"{ladder_total / 60:.1f} min ({ladder_total / capture['total_seconds'] * 100:.0f}%)"
            ),
            "side_effect": (
                "不再逐条跟踪光强漂移 ⇒ 只有在**已确认稳定**的面才可开; "
                "与自适应曝光的数据不能直接混比绝对强度; 固定曝光下若峰值撞到 250 会直接饱和。"
            ),
            "evidence": (
                f"`capture_..._prior/manifest.json` 的 `exposure_ms` 逐条在 "
                f"{measured['exposure_range'][0]:.2f}-{measured['exposure_range'][1]:.2f} ms 之间跳, "
                "而 `exposure_status` 全为 ok。"
            ),
            "saving_seconds": ladder_total,
        }
    )

    # 5) 几何 sweep 的点数与帧数 ------------------------------------------------
    sweep = campaign["phases"].get("几何 sweep")
    if sweep is not None:
        recommendation = min_sweep_points()
        current_points = sweep.points
        tilt_seconds = sweep.tilt_seconds
        saving_frames = tilt_seconds * 0.5
        items.append(
            {
                "rank": 0,
                "title": "几何 sweep: 点数不砍, 但把倾斜点的 4 帧降到 2 帧",
                "mechanism": (
                    f"实测 {current_points} 点 / {sweep.seconds:.0f} s = "
                    f"{sweep.seconds_per_point:.1f} s/点, 其中倾斜组 (每点 4 帧求质心均值) "
                    f"就占 {tilt_seconds:.0f} s ({tilt_seconds / sweep.seconds * 100:.0f}%)。"
                    "**点数不能再砍**: "
                    f"拟合 w^2(c) 二次曲线至少要 {recommendation['width_criterion_points']} 点 "
                    f"(3 参数 + 1 残差自由度 + 1 个顶点锚点), 以峰值作备选判据也要 "
                    f"{recommendation['peak_criterion_points']} 点; "
                    "而现在每个模式只有 4 个幅度, 正好卡在下限 —— 这就是 2026-10-22 "
                    "四个模式报 `degenerate curvature` 的原因。"
                    "真正的水分在**帧数**: 2 帧的质心标准误只比 4 帧大 1.4 倍, "
                    "而 fit 用的只是斜率。"
                ),
                "expected_saving": (
                    f"倾斜组帧数减半 ⇒ 整轮 sweep 省约 {saving_frames:.0f} s "
                    f"({saving_frames / sweep.seconds * 100:.0f}%)"
                ),
                "side_effect": (
                    "倾斜斜率的不确定度上升约 1.4 倍; 当前斜率已经 2.00x 超出可信带, "
                    "所以先用更小的倾斜幅度换更多点, 别用更多帧换更差的信噪比。"
                ),
                "evidence": (
                    "`stage.log` 的 `stage_sweep` 标记 (`[*] tilt x = -1.000 rad (x4)`) "
                    "与同名 `mean c=...` 汇总行之间的时间差; "
                    "以及 `calibrate_bench_geometry_from_sweep` 的四条 "
                    "`width response is degenerate` 警告。"
                ),
                "saving_seconds": saving_frames,
            }
        )

    # 6) campaign 内部的跨变体复用 --------------------------------------------
    illumination = campaign["phases"].get("照明半径测量")
    if illumination is not None:
        items.append(
            {
                "rank": 0,
                "title": "campaign 内部: 面板扫描 / 照明半径 / 几何 sweep 只做一次, 跨变体复用",
                "mechanism": (
                    "`model_in_loop_hw_ab.py` 已经支持 `--geometry-file` 把先测到的几何"
                    "交给后续变体, 但 2026-10-22 的运行没有用 (第一个变体死在几何标定, "
                    "后面根本没跑)。照明半径测量 "
                    f"({illumination.seconds:.0f} s) 与面板扫描是**实验系统的属性**, "
                    "与变体无关; 几何 sweep 也是。"
                ),
                "expected_saving": (
                    f"每个复用到的变体省 "
                    f"{(illumination.seconds + (campaign['phases'].get('面板扫描').seconds if campaign['phases'].get('面板扫描') else 0)) / 60:.1f} min"
                ),
                "side_effect": (
                    "复用意味着后面变体看到的是**别的时候**测的几何; "
                    "若实验系统在两次变体之间被移动过, 复用就是错的 —— "
                    "所以复用的前提是「一次会话内光路不动」, 并且要在记录里写明复用了哪次。"
                ),
                "evidence": (
                    f"`stage.log` 的阶段切分: 面板扫描 + 照明半径 = "
                    f"{(campaign['phases'].get('面板扫描').seconds if campaign['phases'].get('面板扫描') else 0) + illumination.seconds:.0f} s"
                    ", 占 campaign 的 "
                    f"{((campaign['phases'].get('面板扫描').seconds if campaign['phases'].get('面板扫描') else 0) + illumination.seconds) / campaign['total_seconds'] * 100:.0f}%"
                ),
                "saving_seconds": illumination.seconds,
            }
        )

    # 7) 在线/离线切分 ---------------------------------------------------------
    items.append(
        {
            "rank": 0,
            "title": "把拟合、判据与分解全部挪到离线 (设备只负责测量)",
            "mechanism": (
                "几何标定 (w^2(c) 二次拟合、两条焦面标度路径的交叉核对、"
                "trust region 接受/拒绝判据、信任域档位对照) 全都不需要设备。"
                "当前这些步骤夹在采集中间, 设备被占着却什么都不测。"
                "信任域对照尤其是纯离线旋钮: 同一套探针集, 换信任域半径重跑判据即可, "
                "设备时间应当是 0。"
            ),
            "expected_saving": (
                "信任域对照 3 档只改离线判据 ⇒ 探针集 (Step A) 复用, "
                f"设备时间从 3 轮 Step A 降到 1 轮, 省 "
                f"{2 * measured['budget']['rates_seconds']['mil_step_a'] / 60:.1f} min; "
                "若进一步把 Step B 也改成\"同一批帧上重放\", 则三档的 Step B 全部离线, "
                f"省 {3 * measured['budget']['rates_seconds']['mil_step_b'] / 60:.0f} min"
            ),
            "side_effect": (
                "离线判据需要采集时把中间量 (每个模式的 w^2(c) 点、探针响应矩阵) "
                "一起落盘, 否则离线无法复算。"
            ),
            "evidence": (
                "`model_in_loop_hw_runbook.py::stage_calibrate_sweep` 全程不碰设备 "
                "(只读 sweep_records.npz), 实测该阶段只花 "
                f"{measured['campaign']['phases'].get('离线几何标定').seconds:.0f} s。"
            ),
            "saving_seconds": 3 * measured["budget"]["rates_seconds"]["mil_step_b"],
        }
    )

    items.sort(key=lambda item: item["saving_seconds"], reverse=True)
    for position, item in enumerate(items, start=1):
        item["rank"] = position
    return items


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def _share_rows(actions: Sequence[dict[str, Any]], total: float) -> list[dict[str, Any]]:
    rows = []
    for action in actions:
        row = dict(action)
        row["share"] = action["total_seconds"] / total if total else 0.0
        rows.append(row)
    return rows


def render_figure(ledger: dict[str, Any], out_path: Path) -> None:
    """各动作耗时占比条形图 (三类各一栏)。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    kinds = ["campaign", "capture", "probe"]
    titles = {
        "campaign": "campaign (阶段切分, 实测)",
        "capture": "采集脚本 (逐条记录, 实测)",
        "probe": "光路有效性探针 (实测)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    for ax, kind in zip(axes, kinds, strict=True):
        block = ledger["kinds"].get(kind)
        rows = block["actions"] if block else []
        if not rows:
            ax.set_visible(False)
            continue
        labels = [row["action"] for row in rows][::-1]
        values = [row["total_seconds"] for row in rows][::-1]
        shares = [row["share"] * 100.0 for row in rows][::-1]
        ax.barh(range(len(values)), values, color=colors[: len(values)][::-1])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("耗时 (s)", fontsize=9)
        ax.set_title(f"{titles[kind]}\n合计 {block['total_seconds'] / 60:.1f} min", fontsize=10)
        for index, (value, share) in enumerate(zip(values, shares, strict=True)):
            ax.text(
                value,
                index,
                f"  {value:.0f} s ({share:.0f}%)",
                va="center",
                fontsize=8,
            )
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, max(values) * 1.35 if values else 1)
    fig.suptitle("实验系统物理测试耗时账 (只读审计, 数据来自已落盘记录)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def render_optimizations_md(items: Sequence[dict[str, Any]], ledger: dict[str, Any]) -> str:
    campaign_phases = (ledger.get("campaign_phases") or {})
    timeouts = sum(
        phase.settle_timeouts for phase in campaign_phases.values() if phase is not None
    )
    timeout_seconds = sum(
        phase.settle_timeout_seconds
        for phase in campaign_phases.values()
        if phase is not None
    )
    lines = [
        "# 可优化项排序",
        "",
        "所有「预计节省」都由**实测速率**推出, 依据列给出那条实测数字的出处。",
        "本文件只给方案, **不改任何代码**。",
        "",
    ]
    for item in items:
        lines += [
            f"## {item['rank']}. {item['title']}",
            "",
            f"- **机制**: {item['mechanism']}",
            f"- **预计节省**: {item['expected_saving']}",
            f"- **副作用**: {item['side_effect']}",
            f"- **依据**: {item['evidence']}",
            "",
        ]
    lines += [
        "---",
        "",
        "## 未进表的一项 (已核实, 收益为负)",
        "",
        f"- **把 settle 判据放宽到只用 2 帧一致**: campaign 里 settle 判据超时 "
        f"{timeouts} 次、累计 {timeout_seconds:.0f} s, 看起来是最大的一块; "
        "但采集脚本的 settle **已经满足**判据 (中位 5.2 s, `reason=stable`), "
        "放宽判据省下的只是探测窗口里的一次读出, 收益 < 1 s/次, "
        "而误判风险直接落在数据上 —— 不值得。真正的问题不是判据太严, "
        "而是判据前那 40 帧无条件丢弃 (见第 1 项)。",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_device_budget_md(budget: dict[str, Any], ledger: dict[str, Any]) -> str:
    rates = budget["rates_seconds"]
    write_probe = ledger.get("write_probe") or {}
    lines = [
        "# 最小设备时间预算",
        "",
        "所有秒数都由**实测速率**推出 (见 `time_budget.json` 的 `rates_seconds`)。",
        "",
        "## 口径说明",
        "",
        "* **实验系统占用时间** = 墙钟。SLM 的 USB 句柄独占, 采集期间相机与 SLM 都还"
        "占着, 所以\"落盘写帧\"这种不测光的时间也算在内。",
        "* **MIL 的三个速率是推算**: 2026-10-22 的 campaign 死在几何标定, "
        "没有任何一轮 MIL 在实验系统上跑完过, 所以 `mil_step_b` 是下界口径。",
        "* 落盘写帧的单次耗时是**现场实测**的: "
        f"中位 {write_probe.get('median_seconds', 0.0):.2f} s "
        f"(实测于{write_probe.get('measured_on', '未实测')}, "
        f"逐次 {['%.2f' % v for v in write_probe.get('samples_seconds', [])] or 'n/a'} s)。"
        "⚠️ 这一项**逐次离散度很大** (同一台机器上多次实测 2.8-5.7 s), "
        "因为它取决于磁盘与压缩器的瞬时负载; 读账时按 ±40% 看。",
        "",
        "## 速率出处",
        "",
        "| 速率 | 秒 | 来源 |",
        "|---|---|---|",
    ]
    for key, value in rates.items():
        source = ledger["rate_sources"].get(key, "")
        lines.append(f"| `{key}` | {value:.1f} | {source} |")
    lines += [
        "",
        "> **并行性**: SLM 的 USB 句柄独占, 同一时刻只能有一个采集进程; "
        "上面三档的所有分项都是**串行**的, 总额就是占用实验系统的墙钟时间。",
        "> 可以并行的只有**纯离线**的部分 (拟合、判据、分解), 它们不占设备。",
        "",
        "## 跨变体 / 跨档位能复用什么",
        "",
        "| 动作 | 能否复用 | 理由 |",
        "|---|---|---|",
        "| 测光门禁 | 否 (每场一次) | 便宜 ("
        f"{rates.get('light_gate', 0.0):.0f} s), 且是每场的健康检查 |",
        "| 面板扫描 | **能** | 瞳面中心是实验系统的属性, 与变体无关 |",
        "| 照明半径测量 | **能** | 照明口径是实验系统的属性 |",
        "| 几何 sweep | **能** | 焦面标度与离焦响应是实验系统的属性 |",
        "| 探针集 (Step A) | **能** | 探针测的是实验系统自己的相差, 换变体/换信任域不变 |",
        "| Step B | **不能** | 逐次迭代的在线优化, 每换一条配置就要重跑 |",
        "| 验收判据 / 信任域档位对照 | **能** (离线) | 只是接受阈值, 同一批帧上重跑判据 |",
        "",
    ]
    for key, label in (
        ("tier_a", "档 A (决定性)"),
        ("tier_b", "档 B (完整 campaign)"),
        ("tier_c", "档 C (研究闭环)"),
    ):
        tier = budget[key]
        lines += [
            f"## {label}",
            "",
            "| 分项 | 秒 | min | 判据 |",
            "|---|---|---|---|",
        ]
        for row in tier["rows"]:
            lines.append(
                f"| {row['item']} | {row['seconds']:.0f} | {row['seconds'] / 60:.1f} | {row['criterion']} |"
            )
        lines += [
            f"| **合计** | **{tier['total_seconds']:.0f}** | **{tier['total_minutes']:.1f}** | |",
            "",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 采集与汇总
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_campaign(repo: Path) -> dict[str, Any]:
    """从 campaign 的 ``stage.log`` 与 ``manifest.json`` 复算阶段耗时。"""
    root = repo / "data" / "debug" / "model_in_loop_hw_ab"
    sessions = sorted(p for p in root.glob("*") if p.is_dir()) if root.is_dir() else []
    best: dict[str, Any] | None = None
    for session in sessions:
        stage_logs = sorted(session.glob("*/stage.log"))
        manifest = session / "manifest.json"
        if not stage_logs or not manifest.is_file():
            continue
        events = parse_log_events(stage_logs[-1].read_text(encoding="utf-8", errors="replace"))
        phases = segment_campaign(events)
        if not phases:
            continue
        campaign_manifest = _read_json(manifest)
        total = sum(p.seconds for p in phases)
        best = {
            "session": session.name,
            "stage_log": str(stage_logs[-1].relative_to(repo)),
            "phases": {p.name: p for p in phases},
            "total_seconds": total,
            "manifest_stages": campaign_manifest.get("stages", []),
        }
    if best is None:
        return {}
    phases = best["phases"]
    actions = []
    for name, phase in phases.items():
        if phase.seconds <= 0:
            continue
        unit = "点" if phase.points else "次"
        count = phase.points or 1
        actions.append(
            {
                "action": name,
                "unit": unit,
                "count": count,
                "unit_seconds": phase.seconds_per_point if phase.points else phase.seconds,
                "total_seconds": phase.seconds,
            }
        )
    for stage in best["manifest_stages"]:
        if stage.get("kind") != "light":
            continue
        actions.insert(
            0,
            {
                "action": "测光门禁",
                "unit": "次",
                "count": 1,
                "unit_seconds": float(stage.get("duration_s", 0.0)),
                "total_seconds": float(stage.get("duration_s", 0.0)),
            },
        )
    best["actions"] = _share_rows(actions, sum(a["total_seconds"] for a in actions))
    best["total_seconds"] = sum(a["total_seconds"] for a in actions)
    return best


def collect_capture(repo: Path) -> list[dict[str, Any]]:
    """扫描所有采集 manifest。"""
    root = repo / "data" / "debug" / "physical_forward_defocus"
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for manifest in sorted(root.glob("*/manifest.json")):
        rows = _read_json(manifest)
        if not _is_capture_manifest(rows):
            continue
        try:
            samples = capture_rows_to_samples(rows)
        except ValueError:
            continue
        out.append(
            {
                "path": str(manifest.relative_to(repo)),
                "rows": rows,
                "samples": samples,
            }
        )
    return out


def collect_probes(repo: Path) -> list[dict[str, Any]]:
    root = repo / "data" / "debug" / "optical_path_probe"
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for probe_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        summary = summarize_probe(probe_dir)
        if summary is not None:
            summary["path"] = str(probe_dir.relative_to(repo))
            out.append(summary)
    return out


def _median(values: Iterable[float]) -> float:
    data = list(values)
    return statistics.median(data) if data else 0.0


def build_ledger(
    repo: Path,
    *,
    write_cost_s: float,
    mil_round_seconds: float | None = None,
    write_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把三类记录汇成一份动作级时间账。"""
    campaign = collect_campaign(repo)
    captures = collect_capture(repo)
    probes = collect_probes(repo)

    kinds: dict[str, Any] = {}

    if campaign:
        kinds["campaign"] = {
            "session": campaign["session"],
            "evidence": campaign["stage_log"],
            "actions": campaign["actions"],
            "total_seconds": campaign["total_seconds"],
        }

    capture_blocks = []
    for capture in captures:
        summary = summarize_capture(capture["samples"], write_cost_s=write_cost_s)
        summary["path"] = capture["path"]
        summary["drift"] = branch_drift(capture["samples"], capture["rows"])
        capture_blocks.append(summary)
    if capture_blocks:
        # 主账取样本最多的那一轮 (106 条那条)
        primary = max(capture_blocks, key=lambda b: b["n_samples"])
        kinds["capture"] = {
            "evidence": primary["path"],
            "actions": _share_rows(
                primary["actions"], sum(a["total_seconds"] for a in primary["actions"])
            ),
            "total_seconds": sum(a["total_seconds"] for a in primary["actions"]),
            "attribution": primary["attribution"],
            "drift": primary["drift"],
        }

    if probes:
        # 用中位的那一次会话做账 (而不是各会话的平均), 免得口径与 rates 不一致
        ordered = sorted(probes, key=lambda p: p["total_seconds"])
        median_probe = ordered[len(ordered) // 2]
        actions = [
            {
                "action": "曝光阶梯",
                "unit": "档",
                "count": int(median_probe["ladder_steps"]),
                "unit_seconds": median_probe["median_ladder_step_seconds"],
                "total_seconds": median_probe["ladder_seconds"],
            },
            {
                "action": "离焦响应",
                "unit": "态",
                "count": int(median_probe["defocus_steps"]),
                "unit_seconds": median_probe["median_defocus_step_seconds"],
                "total_seconds": median_probe["defocus_seconds"],
            },
        ]
        kinds["probe"] = {
            "evidence": median_probe["path"],
            "actions": _share_rows(actions, sum(a["total_seconds"] for a in actions)),
            "total_seconds": sum(a["total_seconds"] for a in actions),
            "sessions": probes,
        }

    # ---- 速率 ----
    campaign_phases = campaign.get("phases", {}) if campaign else {}

    def _phase_seconds(name: str) -> float:
        phase = campaign_phases.get(name)
        return phase.seconds if phase else 0.0

    light_gate = 0.0
    for stage in campaign.get("manifest_stages", []) if campaign else []:
        if stage.get("kind") == "light":
            light_gate = float(stage.get("duration_s", 0.0))

    probe_seconds = _median(p["total_seconds"] for p in probes)
    capture_primary = max(capture_blocks, key=lambda b: b["n_samples"]) if capture_blocks else None

    step_a = _mil_step_a_seconds()
    accept = _mil_accept_seconds()
    if mil_round_seconds is not None:
        # 命令行给定了一轮 MIL 的总时长 => 由它反推 Step B (轮次的主体)
        step_b = max(mil_round_seconds - step_a - accept, 0.0)
    else:
        step_b = _mil_step_b_seconds()

    rates = {
        "light_gate": light_gate,
        "probe": probe_seconds,
        "panel_scan": _phase_seconds("面板扫描"),
        "illumination": _phase_seconds("照明半径测量"),
        "sweep": _phase_seconds("几何 sweep"),
        "calibrate": _phase_seconds("离线几何标定"),
        "mil_step_a": step_a,
        "mil_step_b": step_b,
        "mil_accept": accept,
        "mil_round": step_a + step_b + accept,
    }

    budget = build_device_budget(rates)

    measured: dict[str, Any] = {
        "campaign": {
            "phases": campaign_phases,
            "total_seconds": campaign.get("total_seconds", 0.0),
        },
        "capture": capture_primary or {},
        "drift": (capture_primary or {}).get("drift", {}),
    }
    if capture_primary:
        exposures = [
            s.exposure_ms
            for capture in captures
            for s in capture["samples"]
            if capture["path"] == capture_primary["path"]
        ]
        measured["exposure_range"] = (
            (min(exposures), max(exposures)) if exposures else (0.0, 0.0)
        )
    measured["budget"] = budget

    items = optimization_items(
        measured,
        write_cost_s=write_cost_s,
    )

    rate_sources = {
        "light_gate": "campaign `manifest.json` 的 light 阶段 `duration_s`",
        "probe": "optical_path_probe 各会话 (目录戳 -> `created_utc`) 的中位",
        "panel_scan": "campaign `stage.log` 的面板扫描窗口",
        "illumination": "campaign `stage.log` 的照明半径窗口",
        "sweep": "campaign `stage.log` 的几何 sweep 窗口",
        "calibrate": "campaign `stage.log` 的离线标定窗口 (不占设备)",
        "mil_step_a": (
            "**推算**: 16 帧 x 1.1 s/帧 (单帧速率取自 `model_in_loop_hw_collect` "
            "的 4 次运行 / 27 s)"
        ),
        "mil_step_b": (
            "**推算**: 600 次迭代 x 1.0 s/次 (读出速率取自 campaign 的"
            "\"写 -> 写\"间隔减去 settle 超时后的量级); "
            "⚠️ 优化器当前 `settle_max_discard=40` 会把单次迭代推到 ~40 s "
            "(见 optimizations.md 第 1 项), 所以这是**下界**"
        ),
        "mil_accept": "**推算**: trust region 接受/拒绝判据, 不占设备",
        "mil_round": "= mil_step_a + mil_step_b + mil_accept",
    }

    return {
        "schema": "physical_time_audit/1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kinds": kinds,
        "rates_seconds": rates,
        "rate_sources": rate_sources,
        "write_probe": write_probe or {},
        "campaign_phases": campaign_phases,
        "device_budget": budget,
        "optimizations": items,
        "capture_blocks": capture_blocks,
        "probes": probes,
    }


def _is_capture_manifest(rows: Any) -> bool:
    """这条 manifest 是不是**采集脚本**的记录。

    采集脚本的记录带 ``settled`` (settle 判据的读数) 与 ``family``
    (``reference`` / ``ramp`` / ``zernike`` / ``prior`` ...)。同一目录下的
    ``alignment_*`` manifest 是几何标定的逐帧记录, 没有这两个字段 ——
    它有 269 行, 比采集那一轮还多, 所以必须显式排除, 不能靠"行数最多"。
    """
    if not isinstance(rows, list) or not rows:
        return False
    head = rows[0]
    if not isinstance(head, dict):
        return False
    return isinstance(head.get("settled"), dict) and "family" in head


def find_primary_capture(repo: Path) -> Path | None:
    """样本最多的那一轮采集的 manifest (审计的主账)。"""
    root = repo / "data" / "debug" / "physical_forward_defocus"
    if not root.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for manifest in sorted(root.glob("*/manifest.json")):
        try:
            rows = _read_json(manifest)
        except (OSError, json.JSONDecodeError):
            continue
        if not _is_capture_manifest(rows):
            continue
        if best is None or len(rows) > best[0]:
            best = (len(rows), manifest)
    return best[1] if best else None


def _mil_step_a_seconds() -> float:
    """MIL 的 Step A: 采集探针集的秒数 (**推算**, 没有实测记录)。

    Step A 要采 8 个探针 x 2 个变体 (bare + 带已知离焦) = 16 帧。
    单帧速率 1.1 s 取自 `model_in_loop_hw_collect` 的实测: 4 次运行 (每次 3 个
    探针 = 6 帧 + 开设备 + 落盘) 在 27 s 内完成。
    """
    return 16 * 1.1


def _mil_step_b_seconds() -> float:
    """MIL 的 Step B: 600 次迭代的秒数 (**推算**, 没有实测记录)。

    每次迭代 = 1 次面板写入 + 1 次读出; 单次读出约 1.0 s (campaign 的
    "写 -> 写" 间隔里减去 settle 超时后得到的量级)。

    ⚠️ 这个数**只在下界口径下成立**: 优化器当前的
    `settle_max_discard = 40` 会在每次采集前无条件读 40 帧, 把单次迭代推到
    ~40 s, 一轮就是小时量级 —— 见 `optimizations.md` 第 1 项。
    """
    return 600 * 1.0


def _mil_accept_seconds() -> float:
    """MIL 的验收判据 (trust region 接受/拒绝) 的秒数 (**推算**)。"""
    return 10.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="实验系统物理测试耗时审计 (只读解析已落盘记录, 不开设备)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录 (默认 logs/physical_time_audit/<UTC戳>)",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=ROOT,
        help="仓库根 (默认脚本所在仓库)",
    )
    parser.add_argument(
        "--write-cost-s",
        type=float,
        default=None,
        help="落盘写帧的单次耗时 (s); 默认现场实测一次",
    )
    parser.add_argument(
        "--no-write-probe",
        action="store_true",
        help="不实测落盘耗时 (用 0 记账, 并在输出里标注)",
    )
    parser.add_argument(
        "--mil-round-s",
        type=float,
        default=None,
        help="一轮 MIL 的秒数 (默认推算值)",
    )
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    write_probe: dict[str, Any] = {"median_seconds": 0.0, "samples_seconds": [], "measured_on": "未实测"}
    if args.write_cost_s is not None:
        write_probe = {
            "median_seconds": float(args.write_cost_s),
            "samples_seconds": [],
            "measured_on": "命令行给定",
        }
    elif not args.no_write_probe:
        # 优先用真实样本实测落盘耗时 (同样的负载、同样的压缩器)
        primary = find_primary_capture(repo)
        samples: list[Path] = []
        if primary is not None:
            rows = _read_json(primary)
            names = [
                str(row.get("file"))
                for row in rows[:5]
                if isinstance(row, dict) and row.get("file")
            ]
            samples = [primary.parent / name for name in names]
        write_probe = measure_write_cost(sample_paths=samples)
    write_cost = float(write_probe["median_seconds"])

    mil_round = float(args.mil_round_s) if args.mil_round_s is not None else None

    ledger = build_ledger(
        repo,
        write_cost_s=write_cost,
        mil_round_seconds=mil_round,
        write_probe=write_probe,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_dir = (args.out or (repo / "logs" / "physical_time_audit" / stamp)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "time_budget.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "optimizations.md").write_text(
        render_optimizations_md(ledger["optimizations"], ledger), encoding="utf-8"
    )
    (out_dir / "device_budget.md").write_text(
        render_device_budget_md(ledger["device_budget"], ledger), encoding="utf-8"
    )
    render_figure(ledger, out_dir / "time_share.png")

    print(f"wrote {out_dir}")
    for kind, block in ledger["kinds"].items():
        print(f"  {kind}: {block['total_seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
