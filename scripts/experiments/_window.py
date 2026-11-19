"""设备窗口信封: 一次占用 / 多段串行 / 段间冷却 / 证据落盘。

为什么需要这一层 (2026-11-19)
-----------------------------
合并计划 (`docs/dev/hw_plan_consolidated_20261118.md` §1) 要求「把确需设备的段合并
成一次窗口, 段间冷却, 每段独立 ``--out``, 采完即释放, 占用期不做任何分析」, 但仓库里
没有任何一层实现这条纪律:

* ``python -m ao_shaping.tools.slm.transfer run`` **自己不拿 bench DeviceLock**, 也
  没有 ``--wait-until-idle`` (只有 ``open_loop`` 显示段内部拿锁)。两个会话同时开
  SLM 会触发驱动板假死 (``ReadSU=-1000`` / ``Open=-10002``), 需要用户手动重启;
* 背对背 SLM open/close 同样会假死, 所以段与段之间必须插入冷却 (合并版给 ≥200 s);
* ``run_session`` 用 ``out.mkdir(parents=True, exist_ok=False)``, 所以**不能**预先
  创建子段目录 —— 本模块因此把证据写在窗口目录下的 ``segments/``, 而不是子段 ``--out``。

协议 (与 ``scripts/slm/replay_transfer_shapes_hardware.py`` 的外壳语义一致)
--------------------------------------------------------------------
1. 没有 ``--execute``: 只打印计划 (段名 / argv / 落盘目录 / 冷却 / 锁归属模式),
   **不创建任何目录、不导入任何设备模块**;
2. 有 ``--execute``: 先拒绝复用已有窗口目录与已有子段目录 (证据要有落脚点, 且
   ``exist_ok=False`` 的子进程不能被空目录挡住);
3. 设备锁有**两种归属模式**, 由调用方用 ``lock_mode`` 显式选择, 绝不混用:

   * ``"window"`` (默认, transfer 窗口用): 窗口自己 ``wait_until_idle`` + 持锁到
     窗口结束 (``finally`` 释放), 子段**不带**自己的等待 (它们拿不到锁就必须立刻
     失败, 否则会父子互锁);
   * ``"segment"`` (issue #238 周期扫描用): 窗口**完全不拿锁**, 每个子段脚本自己
     ``acquire`` / ``release``, 真实等待值随 argv 转发; 窗口只负责串行、段间冷却,
     并在每段结束后**只读**回读锁状态、结束时记录最终锁证据。窗口**从不释放**不是
     自己拿到的锁, 也从不替别的进程解锁。

4. 每段: 子进程 stdout/stderr 落 ``segments/<name>.log``, 记录 argv / 起止 UTC /
   墙钟 / 退出码 / 子段产物是否出现; 任一段非零退出即**停止整个窗口** (不自动续跑:
   非零退出通常是驱动板假死, 继续只会扩大故障);
5. ``segment`` 模式下每段结束回读一次锁: 仍被别人占着就记进 ``segments/<name>.json``
   的 ``lock_after`` 并停止窗口 (下一段只会干等, 徒增故障面), 但仍**不释放**它;
6. 段间冷却按 ``--cooldown-s`` 分段睡眠, 并记录冷却期间锁持有者; ``window`` 模式应
   看到自身进程, ``segment`` 模式应为空闲 (或明确记下外来持有者, 不掩盖);
7. 结束写 ``released.json`` (锁释放结果 / 最终状态) 与 ``window.json`` (总表)。

``--wait-until-idle 0`` **不是**「跳过锁」: 探针脚本无论如何都会 ``acquire``, 0 只是
把等待上限设为 0, 被占就立刻 ``exit 2`` (证据见 2026-11-19 ``run3_period_sweep``
``period32.log``)。所以窗口要么自己持锁并要求子段 0 (= ``window`` 模式), 要么自己
完全不持锁并把真实等待值交给子段 (= ``segment`` 模式)。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: 新采集的统一落盘根 (用户 2026-11-19 指令); 历史已采数据留在 ``data/debug/`` 不动。
DATA_ROOT = ROOT / "data" / "experiments"

#: 冷却期间每段的睡眠切片 (s): 便于在日志里留下进度, 也便于 Ctrl-C 响应。
COOLDOWN_SLICE_S = 20.0

#: 设备锁归属模式: 窗口持锁 (默认, transfer 窗口) / 子段自取自放 (issue #238)。
#: 两者不能混用 —— 同时持锁的子段在窗口持锁时只会立刻失败 (见模块 docstring)。
LOCK_MODE_WINDOW = "window"
LOCK_MODE_SEGMENT = "segment"
LOCK_MODES: tuple[str, ...] = (LOCK_MODE_WINDOW, LOCK_MODE_SEGMENT)

#: 计划表的锁归属说明 (dry-run 与 window_plan.json 共用)。
LOCK_MODE_NOTES: dict[str, str] = {
    LOCK_MODE_WINDOW: "窗口持锁到结束; 子段以 --wait-until-idle 0 立即失败而非再拿锁",
    LOCK_MODE_SEGMENT: "窗口不持锁; 每段脚本自取自放, 真实 --wait-until-idle 转发给子段",
}


def utc_now() -> str:
    """ISO 8601 UTC 时间戳 (秒精度), 证据文件统一用它。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    """写 JSON (先写临时文件再替换, 避免半截文件)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, default=str, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


@dataclass(frozen=True)
class Segment:
    """窗口里的一个子段: 一条完整的子进程 argv + 它独占的输出目录。"""

    name: str
    """短名 (证据文件名 / 目录名都用它)。"""

    argv: tuple[str, ...]
    """完整子进程 argv (含 ``--execute`` 等子进程自己的开关)。"""

    out: Path
    """子段独占输出目录; 必须不存在 (子进程多数用 ``exist_ok=False``)。"""

    purpose: str
    """一行说明: 这一段回答哪个计划问题。"""

    requires: tuple[str, ...] = field(default_factory=tuple)
    """子段依赖的窗口级输入名 (如 ``dataset`` / ``checkpoint``); 缺失即拒绝。"""


def describe_segments(segments: Sequence[Segment]) -> list[dict[str, Any]]:
    """计划表: dry-run 打印与 ``window_plan.json`` 共用。"""
    return [
        {
            "name": seg.name,
            "purpose": seg.purpose,
            "out": str(seg.out),
            "argv": list(seg.argv),
            "requires": list(seg.requires),
        }
        for seg in segments
    ]


def print_plan(
    segments: Sequence[Segment],
    *,
    window: Path,
    cooldown_s: float,
    wait_s: float,
    lock_mode: str = LOCK_MODE_WINDOW,
) -> None:
    """dry-run: 只打印, 不建目录、不导入设备模块。"""
    print("=" * 78)
    print("设备窗口计划 (dry-run, 未占用任何设备)")
    print("=" * 78)
    print(f"窗口目录 : {window}")
    print(f"设备锁   : {lock_mode} — {LOCK_MODE_NOTES.get(lock_mode, '未知模式')}")
    print(f"设备等待 : --wait-until-idle {wait_s:g} s")
    print(f"段间冷却 : {cooldown_s:g} s")
    print(f"段数     : {len(segments)}")
    for index, seg in enumerate(segments, start=1):
        print(f"\n[{index}/{len(segments)}] {seg.name} — {seg.purpose}")
        print(f"  落盘: {seg.out}")
        print("  argv: " + " ".join(seg.argv))
    print("\n" + "=" * 78)
    print("要真正采集: 追加 --execute (并确认已获用户授权、光路探测通过)。")
    print("=" * 78)


def freshness_problems(window: Path, segments: Sequence[Segment]) -> list[str]:
    """拒绝复用的条件: 窗口目录非空 / 任一子段目录已存在。"""
    problems: list[str] = []
    if window.exists():
        if not window.is_dir():
            problems.append(f"窗口路径已存在且不是目录: {window}")
        elif any(window.iterdir()):
            problems.append(f"窗口目录已存在且非空 (不允许复用): {window}")
    for seg in segments:
        if seg.out.exists():
            problems.append(f"子段目录已存在 (子进程用 exist_ok=False): {seg.out}")
    return problems


def acquire_lock(
    wait_s: float,
    *,
    slm_number: int,
    cam_id: int,
    own_paths: Sequence[str | Path] = (),
) -> tuple[Any | None, dict[str, Any]]:
    """等设备空闲并拿 bench 锁 (**仅** ``window`` 模式调用)。

    ``wait_s <= 0`` 时窗口按历史约定既不等窗口锁也不持锁 (证据记 ``disabled``)。
    注意这**不**意味着子段会跳过锁: 子脚本仍会 ``acquire``, 0 只是让它被占时立刻
    失败。因此 ``wait_s <= 0`` 只适合「窗口必须自己持锁」以外的场景, 不能用来把
    「窗口不持锁 + 子段等待 0」拼在一起。
    """
    record: dict[str, Any] = {
        "requested_wait_s": wait_s,
        "slm_number": slm_number,
        "cam_id": cam_id,
        "started_at": utc_now(),
    }
    if wait_s <= 0:
        record.update(status="disabled", ok=True, note="--wait-until-idle 0: 窗口不等也不持锁")
        return None, record

    from ao_shaping.tools.kernels.device_mutex import DeviceLock, wait_until_idle

    lock = DeviceLock.default()
    verdict = wait_until_idle(
        wait_s,
        lock=lock,
        slm_number=slm_number,
        cam_id=cam_id,
        own_paths=tuple(str(p) for p in own_paths),
    )
    record.update(
        ok=bool(verdict.ok),
        reason=str(getattr(verdict, "reason", "")),
        reason_text=str(getattr(verdict, "reason_text", "")),
        finished_at=utc_now(),
    )
    if not verdict.ok:
        record["status"] = "wait-failed"
        return None, record

    holder = lock.holder()
    if holder is None:
        if not lock.acquire(timeout=max(1.0, min(wait_s, 30.0))):
            record.update(status="acquire-failed", ok=False)
            return None, record
    record["status"] = "held"
    record["holder"] = _holder_payload(lock)
    return lock, record


def _holder_payload(lock: Any) -> dict[str, Any]:
    holder = lock.holder()
    if holder is None:
        return {"holder": None}
    return {
        "pid": getattr(holder, "pid", 0),
        "ppid": getattr(holder, "ppid", 0),
        "cmdline": getattr(holder, "cmdline", ""),
        "acquired_at": getattr(holder, "acquired_at", ""),
    }


def release_lock(lock: Any | None) -> dict[str, Any]:
    """释放锁并回读状态; 释放失败只记录, 不掩盖子段退出码。"""
    record: dict[str, Any] = {"released_at": utc_now()}
    if lock is None:
        record.update(lock_released="disabled", holder=None)
        return record
    error: str | None = None
    try:
        lock.release()
    except Exception as exc:  # 证据优先, 不掩盖退出码
        error = f"{type(exc).__name__}: {exc}"
    record["error"] = error
    try:
        record["holder"] = _holder_payload(lock)
        record["lock_released"] = record["holder"]["holder"] is None
    except Exception as exc:  # 回读失败只记录
        record["lock_released"] = "unknown"
        record["holder_error"] = f"{type(exc).__name__}: {exc}"
    return record


def open_lock_probe() -> Any | None:
    """懒导入一把**不 acquire** 的锁对象, 只用于回读持有者 (segment 模式)。

    导入 ``device_mutex`` 不会打开任何设备; 失败时返回 ``None`` (证据记 unknown),
    绝不让「读不到锁状态」变成「去抢锁」。
    """
    try:
        from ao_shaping.tools.kernels.device_mutex import DeviceLock
    except Exception:
        return None
    try:
        return DeviceLock.default()
    except Exception:
        return None


def lock_probe(lock: Any | None, *, at: str | None = None) -> dict[str, Any]:
    """**只读**回读锁状态 (谁在持有 / 是否空闲); 从不 ``release``。

    窗口外的持有者可能是别的会话: 只记录, 不动。
    """
    record: dict[str, Any] = {"at": at or utc_now()}
    if lock is None:
        record.update(status="not-probed", holder=None)
        return record
    try:
        holder = lock.holder()
    except Exception as exc:  # 回读失败只记录
        record.update(status="unknown", holder=None)
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    if holder is None:
        record.update(status="free", holder=None)
    else:
        record.update(
            status="held",
            holder={
                "pid": getattr(holder, "pid", 0),
                "ppid": getattr(holder, "ppid", 0),
                "cmdline": getattr(holder, "cmdline", ""),
                "acquired_at": getattr(holder, "acquired_at", ""),
            },
        )
    return record


def cooldown(seconds: float, *, after: str, lock: Any | None) -> dict[str, Any]:
    """段间冷却; 每片记录一次锁持有者, 冷却结束回读实际墙钟。"""
    record: dict[str, Any] = {
        "after_segment": after,
        "requested_s": seconds,
        "started_at": utc_now(),
    }
    if seconds <= 0:
        record.update(elapsed_s=0.0, skipped=True, finished_at=utc_now())
        return record
    started = time.monotonic()
    slices: list[dict[str, Any]] = []
    while True:
        remaining = seconds - (time.monotonic() - started)
        if remaining <= 0:
            break
        time.sleep(min(COOLDOWN_SLICE_S, remaining))
        if lock is not None:
            slices.append({"at": utc_now(), **_holder_payload(lock)})
    record.update(
        elapsed_s=round(time.monotonic() - started, 1),
        skipped=False,
        finished_at=utc_now(),
    )
    if slices:
        record["holder_samples"] = slices
    return record


def run_segment(seg: Segment, *, window: Path, timeout_s: float) -> dict[str, Any]:
    """跑一个子段: 日志落盘, 记录退出码与产物; 超时杀进程树。"""
    log_path = window / "segments" / f"{seg.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "name": seg.name,
        "purpose": seg.purpose,
        "argv": list(seg.argv),
        "out": str(seg.out),
        "log": str(log_path),
        "started_at": utc_now(),
        "timeout_s": timeout_s,
    }
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {seg.name} @ {record['started_at']}\n")
        handle.write("# argv: " + " ".join(seg.argv) + "\n\n")
        handle.flush()
        try:
            completed = subprocess.run(
                list(seg.argv),
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_s if timeout_s > 0 else None,
                check=False,
            )
            record["exit_code"] = completed.returncode
            record["timed_out"] = False
        except subprocess.TimeoutExpired:
            record["exit_code"] = None
            record["timed_out"] = True
    record["duration_s"] = round(time.monotonic() - started, 1)
    record["finished_at"] = utc_now()
    record["out_exists"] = seg.out.is_dir()
    record["out_files"] = _top_level_entries(seg.out)
    record["status"] = (
        "timeout"
        if record["timed_out"]
        else ("ok" if record["exit_code"] == 0 else "failed")
    )
    write_json(window / "segments" / f"{seg.name}.json", record)
    return record


def _top_level_entries(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    try:
        return sorted(entry.name for entry in path.iterdir())
    except OSError:
        return []


def run_window(
    segments: Sequence[Segment],
    *,
    window: Path,
    cooldown_s: float,
    wait_s: float,
    timeout_s: float,
    slm_number: int,
    cam_id: int,
    execute: bool,
    lock_mode: str = LOCK_MODE_WINDOW,
    extra: dict[str, Any] | None = None,
) -> int:
    """窗口主流程: 校验 → (等锁 | 交给子段) → 逐段 → 冷却 → 证据。

    ``lock_mode`` 决定**谁来拿 bench 锁**, 两者不能混用 (见模块 docstring):

    * ``"window"`` (默认): 窗口 ``wait_until_idle`` + 持锁到窗口结束, 子段必须已经
      用 ``--wait-until-idle 0`` 关掉自己的等待 (否则父子互锁);
    * ``"segment"``: 窗口完全不 ``acquire``, 每个子段自己拿锁; 窗口只串行、冷却,
      并在每段结束后**只读**回读锁状态、结束时记录最终锁证据, 从不释放别人的锁。
    """
    if lock_mode not in LOCK_MODES:
        print(
            f"拒绝执行: 未知锁模式 {lock_mode!r}; 只支持 {', '.join(LOCK_MODES)}",
            file=sys.stderr,
        )
        return 2

    if not execute:
        print_plan(
            segments,
            window=window,
            cooldown_s=cooldown_s,
            wait_s=wait_s,
            lock_mode=lock_mode,
        )
        return 0

    problems = freshness_problems(window, segments)
    if problems:
        print("拒绝执行: 目标目录不可复用", file=sys.stderr)
        for item in problems:
            print(f"  - {item}", file=sys.stderr)
        return 2

    window.mkdir(parents=True, exist_ok=False)
    plan_payload = {
        "window": str(window),
        "created_at": utc_now(),
        "python": sys.executable,
        "lock_mode": lock_mode,
        "lock_mode_note": LOCK_MODE_NOTES[lock_mode],
        "cooldown_s": cooldown_s,
        "wait_until_idle_s": wait_s,
        "segment_timeout_s": timeout_s,
        "segments": describe_segments(segments),
        "extra": dict(extra or {}),
    }
    write_json(window / "window_plan.json", plan_payload)
    print(f"[window] 目录 {window}")
    print(f"[window] 锁模式: {lock_mode}")
    print(f"[window] {len(segments)} 段: {', '.join(s.name for s in segments)}")

    segment_managed = lock_mode == LOCK_MODE_SEGMENT
    lock: Any | None = None
    probe: Any | None = None
    lock_record: dict[str, Any] = {}
    lock_states: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    cooldowns: list[dict[str, Any]] = []
    stopped_after: str | None = None
    stop_reason: str | None = None
    try:
        if segment_managed:
            # 只读探针: 回读持有者用, 绝不 acquire, 更不替别人 release。
            probe = open_lock_probe()
            lock_record = {
                "mode": LOCK_MODE_SEGMENT,
                "status": "segment-managed",
                "ok": True,
                "requested_wait_s": wait_s,
                "probe": "opened" if probe is not None else "unavailable",
                "note": LOCK_MODE_NOTES[LOCK_MODE_SEGMENT],
                "started_at": utc_now(),
            }
            print(
                f"[window] 锁交给子段自取 (段内 --wait-until-idle {wait_s:g} s); "
                f"窗口不持锁, 仅串行 + {cooldown_s:g}s 冷却"
            )
        else:
            lock, lock_record = acquire_lock(
                wait_s,
                slm_number=slm_number,
                cam_id=cam_id,
                own_paths=(Path(__file__).resolve(),),
            )
            print(f"[window] 设备等待/加锁: {lock_record.get('status')}")

        if segment_managed or lock_record.get("ok"):
            evidence_lock = probe if segment_managed else lock
            for index, seg in enumerate(segments):
                if index:
                    cool = cooldown(
                        cooldown_s,
                        after=segments[index - 1].name,
                        lock=evidence_lock,
                    )
                    cooldowns.append(cool)
                    print(
                        f"[window] 冷却 {cool['elapsed_s']:g}s "
                        f"(after {cool['after_segment']})"
                    )
                print(f"[window] 段 {seg.name} 开始: {seg.out}")
                record = run_segment(seg, window=window, timeout_s=timeout_s)
                results.append(record)
                print(
                    f"[window] 段 {seg.name} 结束: status={record['status']} "
                    f"exit={record['exit_code']} {record['duration_s']}s"
                )

                if segment_managed:
                    state = lock_probe(probe)
                    state["segment"] = seg.name
                    lock_states.append(state)
                    record["lock_after"] = state
                    write_json(window / "segments" / f"{seg.name}.json", record)
                    print(
                        f"[window] 段 {seg.name} 结束锁状态: {state.get('status')}"
                        + (
                            f" (持锁者 {state['holder'].get('pid')}, "
                            "非本窗口所持, 不释放)"
                            if state.get("status") == "held" and state.get("holder")
                            else ""
                        )
                    )

                if record["status"] != "ok":
                    stopped_after = seg.name
                    stop_reason = "segment-failed"
                    break
                if segment_managed and state.get("status") == "held":
                    # 子段没放锁: 下一段只会干等, 停在这里; 但仍不替别人解锁。
                    stopped_after = seg.name
                    stop_reason = "lock-not-released"
                    break
        else:
            stopped_after = "device-lock"
            stop_reason = "device-lock"
    except KeyboardInterrupt:
        stopped_after = "keyboard-interrupt"
        stop_reason = "keyboard-interrupt"
        raise
    finally:
        if segment_managed:
            final = lock_probe(probe)
            final_status = final.get("status")
            released = {
                "mode": LOCK_MODE_SEGMENT,
                "released_by_window": False,
                "window_held_lock": False,
                "final_state": final,
                "lock_released": (
                    True
                    if final_status == "free"
                    else ("unknown" if final_status != "held" else False)
                ),
                "released_at": utc_now(),
            }
        else:
            # 只释放本窗口 acquire 到的那把锁; 别人的锁永远不碰。
            released = release_lock(lock)
        write_json(window / "released.json", released)
        all_ok = (
            bool(results)
            and stopped_after is None
            and all(item["status"] == "ok" for item in results)
        )
        if segment_managed and released.get("lock_released") is not True:
            all_ok = False
        write_json(
            window / "window.json",
            {
                **plan_payload,
                "finished_at": utc_now(),
                "device_lock": lock_record,
                "segment_lock_states": lock_states,
                "released": released,
                "cooldowns": cooldowns,
                "segments": results,
                "stopped_after": stopped_after,
                "stop_reason": stop_reason,
                "all_ok": all_ok,
            },
        )
        if segment_managed:
            print(f"[window] 最终锁状态: {released['final_state'].get('status')} "
                  f"(窗口未释放任何锁)")
        else:
            print(f"[window] 锁释放: {released.get('lock_released')}")

    if not lock_record.get("ok"):
        print(
            "设备未空出, 一段都没跑。见 window.json 的 device_lock 记录。",
            file=sys.stderr,
        )
        return 3
    if stopped_after is not None:
        print(f"窗口在段 {stopped_after} 停止 ({stop_reason}); 未自动续跑。", file=sys.stderr)
        return 1
    return 0


def add_window_args(parser: Any, *, default_cooldown: float = 200.0) -> None:
    """两个窗口脚本共用的 CLI (冷却 / 等待 / 超时 / 执行开关)。"""
    parser.add_argument(
        "--cooldown-s",
        type=float,
        default=default_cooldown,
        help="段间冷却秒数 (合并版要求 ≥200 s; 背对背 open/close 会假死)",
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=600.0,
        help=(
            "等设备空闲的最长秒数。window 模式: 窗口等并持锁 (0 = 窗口不等不持锁); "
            "segment 模式: 原样转发给子段, 子段自己等并持锁 (0 只表示被占立刻失败, "
            "不是跳过锁)"
        ),
    )
    parser.add_argument(
        "--segment-timeout-s",
        type=float,
        default=3600.0,
        help="单段子进程超时 (0 = 不限)",
    )
    parser.add_argument(
        "--slm-number", type=int, default=1, help="SLM 编号 (默认 1)"
    )
    parser.add_argument("--cam-id", type=int, default=0, help="相机编号 (默认 0)")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正占用设备; 缺省只打印计划 (不建目录、不导入设备模块)",
    )
