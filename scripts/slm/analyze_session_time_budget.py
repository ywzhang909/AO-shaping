"""把一次 transfer 会话的墙钟时间拆成可归因的几段 (设备无关, 只读已落盘记录)。

用途: 判断"提速提案值不值得做"时, 需要的不是猜测而是**这一段的占比**。
本脚本从三个阶段各自已写盘的记录里取时间戳:

* ``journal.json`` —— 各阶段的 ``running`` / ``complete`` 单调时钟;
* ``shape/<shape>/<stage>/manifest.json`` —— 每一帧的 ``monotonic_ns`` /
  ``end_monotonic_ns`` / ``exposure_ms``, 于是"相机读出"与"两次读出之间的
  间隔"可以分开统计;
* ``<shape>_exposure_reads.json`` —— 曝光调节期间每一张试拍的落盘时刻。

归因口径 (逐项互斥):

``camera``     ``end_monotonic_ns - monotonic_ns`` 之和: 相机积分+读出+传输。
``gap``        相邻两帧之间的间隔 (减上一帧的相机时间): SLM 显示 + 液晶稳定
               ``time.sleep`` + 优化器计算 + 落盘。
``tuning``     曝光调节区间 (第一张试拍到 `adapt_exposure` 收敛), 单列, 因为它
               是"反馈式调节"成本的直接度量。
``stage_other`` 阶段总时长减去上面三项: 目标构造、模型前向、manifest 写入等。

用法::

    python scripts/slm/analyze_session_time_budget.py --session <会话目录>
    python scripts/slm/analyze_session_time_budget.py --session <会话目录> --json out.json

会话目录 = ``--out`` 给的那个目录 (含 ``journal.json``)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

#: 帧间隔里必然包含一次 SLM 显示 + 液晶稳定; 低于它的间隔只能说明这一帧没有
#: 重新下发相位 (例如 ``discard`` 帧)。仅用于在报告里解释 gap 的构成。
SETTLE_FLOOR_S = 0.6


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _ns_to_s(ns: float) -> float:
    return float(ns) / 1e9


def stage_windows(journal: list[dict]) -> list[dict]:
    """Pair each ``running`` entry with its ``complete`` entry (same stage name)."""
    open_at: dict[str, dict] = {}
    windows = []
    for entry in journal:
        stage, status = entry.get("stage"), entry.get("status")
        if status == "running":
            open_at[stage] = entry
        elif status in ("complete", "reused", "skipped"):
            started = open_at.pop(stage, None)
            if started is None:
                continue
            windows.append(dict(
                stage=stage, status=status,
                start_ns=started["monotonic_ns"], end_ns=entry["monotonic_ns"],
                seconds=_ns_to_s(entry["monotonic_ns"] - started["monotonic_ns"]),
            ))
    return windows


def capture_cost(manifest_path: Path) -> dict:
    """Camera time vs inter-frame gap, summed over every captured frame."""
    if not manifest_path.is_file():
        return dict(frames=0, camera_s=0.0, gap_s=0.0, exposure_ms=[], gap_median_s=0.0)
    rows = _load(manifest_path)
    frames, camera, gaps = [], 0.0, []
    previous_end = None
    for row in rows:
        for frame in row.get("frames", []):
            start, end = frame.get("monotonic_ns"), frame.get("end_monotonic_ns")
            if start is None or end is None:
                continue
            frames.append(frame)
            camera += _ns_to_s(end - start)
            if previous_end is not None and start >= previous_end:
                gaps.append(_ns_to_s(start - previous_end))
            previous_end = end
    gaps_sorted = sorted(gaps)
    return dict(
        frames=len(frames),
        camera_s=round(camera, 3),
        gap_s=round(sum(gaps), 3),
        gap_median_s=round(gaps_sorted[len(gaps_sorted) // 2], 3) if gaps_sorted else 0.0,
        exposure_ms=sorted({round(float(f.get("exposure_ms", 0.0)), 3) for f in frames}),
    )


def tuning_cost(stage_dir: Path, shape: str) -> dict:
    """How long the exposure search took, and how many trial frames it burned."""
    stamps_path = stage_dir / f"{shape}_exposure_reads.json"
    if not stamps_path.is_file():
        return dict(frames=0, seconds=0.0, settled_ms=None)
    stamps = _load(stamps_path)
    if not stamps:
        return dict(frames=0, seconds=0.0, settled_ms=None)
    span = _ns_to_s(stamps[-1]["monotonic_ns"] - stamps[0]["monotonic_ns"])
    settled = stage_dir / f"{shape}_exposure.json"
    settled_ms = None
    if settled.is_file():
        payload = _load(settled)
        if payload.get("frames"):
            settled_ms = payload["frames"][-1].get("exposure_ms")
    return dict(frames=len(stamps), seconds=round(span, 3), settled_ms=settled_ms)


def session_budget(session: Path) -> dict:
    journal = _load(session / "journal.json")
    windows = stage_windows(journal)
    total = None
    plan = next((e for e in journal if e.get("stage") == "plan"), None)
    if plan is not None and journal:
        total = _ns_to_s(journal[-1]["monotonic_ns"] - plan["monotonic_ns"])

    stages = []
    for window in windows:
        name = window["stage"]
        parts = name.split("/")
        if len(parts) == 2:
            shape, recipe = parts
            stage_dir = session / "shape" / shape / recipe
        else:
            shape, stage_dir = None, session / name
        captures = capture_cost(stage_dir / "manifest.json") if stage_dir.is_dir() else dict(
            frames=0, camera_s=0.0, gap_s=0.0, gap_median_s=0.0, exposure_ms=[])
        tuning = tuning_cost(stage_dir, shape) if (shape and stage_dir.is_dir()) else dict(
            frames=0, seconds=0.0, settled_ms=None)
        stages.append(dict(
            stage=name, seconds=round(window["seconds"], 3), **captures, tuning=tuning,
            stage_other_s=round(window["seconds"] - captures["camera_s"] - captures["gap_s"]
                                - tuning["seconds"], 3),
        ))

    return dict(session=str(session), total_s=round(total, 1) if total else None,
                stages=stages)


def _print(report: dict) -> None:
    print(f"session: {report['session']}")
    print(f"total:   {report['total_s']} s")
    header = (f"{'stage':28} {'sec':>8} {'cam':>7} {'gap':>7} {'tune':>7} "
              f"{'other':>7} {'frames':>6} {'gap_med':>7}")
    print(header)
    print("-" * len(header))
    for stage in report["stages"]:
        print(f"{stage['stage']:28} {stage['seconds']:8.1f} {stage['camera_s']:7.1f} "
              f"{stage['gap_s']:7.1f} {stage['tuning']['seconds']:7.1f} "
              f"{stage['stage_other_s']:7.1f} {stage['frames']:6d} "
              f"{stage['gap_median_s']:7.2f}")
    camera = sum(s["camera_s"] for s in report["stages"])
    gap = sum(s["gap_s"] for s in report["stages"])
    tuning = sum(s["tuning"]["seconds"] for s in report["stages"])
    frames = sum(s["frames"] for s in report["stages"])
    total = report["total_s"] or 0.0
    print("-" * len(header))
    print(f"{'sum of stages':28} {sum(s['seconds'] for s in report['stages']):8.1f} "
          f"{camera:7.1f} {gap:7.1f} {tuning:7.1f} "
          f"{sum(s['stage_other_s'] for s in report['stages']):7.1f} {frames:6d}")
    if total:
        print(f"\nshares of the whole session ({total:.0f} s): "
              f"camera {camera / total:.1%}, settle/other-gap {gap / total:.1%}, "
              f"exposure tuning {tuning / total:.1%}, stage overhead "
              f"{sum(s['stage_other_s'] for s in report['stages']) / total:.1%}")
        print(f"per captured frame: {camera / frames:.2f} s camera + "
              f"{gap / frames:.2f} s gap" if frames else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, type=Path,
                        help="transfer 会话目录 (含 journal.json)")
    parser.add_argument("--json", type=Path, help="把拆解写成 JSON")
    args = parser.parse_args(argv)
    report = session_budget(args.session)
    _print(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
