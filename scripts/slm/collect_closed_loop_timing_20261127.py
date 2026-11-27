"""GUI 闭环逐轮 capture/fit/adapt/candidate 分项耗时采集器.

背景
----
``closed_loop_speed_analysis_20261125.md`` §5 第 1 项指出：GUI 闭环历史上只有单轮
fault，没有多轮 ``rounds.jsonl``，因此无法把逐轮耗时拆成 capture/fit/adapt/candidate。
当前工作树里 ``closed_loop_shaping.py`` / ``closed_loop_shaping_page.py`` 由别的车道
改写，本脚本**不改这两处**，只通过 ``ClosedLoopRunner`` 的公开钩子收集：

* ``on_event``：状态迁移的时间线（``适配重拟合`` → ``adaptation`` → ``生成第 N 轮候选``
  → ``下发第 N 轮候选``），据此拆出 fit / adapt / candidate；
* ``on_frame``：每轮采集完成的时刻，配合 ``TimingBench`` 装饰器拆出 SLM 写入与 CCD 取帧；
* ``Bench`` 装饰器本身不改变任何设备调用顺序。

安全契约
--------
* 默认（无 ``--execute``）是 plan-only：不导入设备模块、不加设备锁、不创建 ``--out``；
  只把解析后的配置打到 stdout 供离线复核。
* ``--execute`` 才懒导入 ``device_mutex`` 与 ``HardwareBench``，按
  ``--wait-until-idle`` 等设备空出并拿共享锁；锁在 ``finally`` 里释放，释放失败
  单独记录，绝不掩盖运行退出码。
* ``candidate_policy`` 固定为 ``auto``，整场无人值守；门禁失败仍由 runner 自己回滚。

证据
----
``<out>/rounds.jsonl``：每轮一行，含四个分项毫秒数与门禁/分数/决策。
``<out>/session.json``：配置、设备回读、锁记录、状态迁移时间线与运行结论。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from ao_shaping.gui.slm.closed_loop_shaping import (
    AdaptationPolicy,
    ClosedLoopConfig,
    ClosedLoopRunner,
    LoopEvent,
    OnlineTransferAdapter,
    analysis_grid,
    target_field,
)

logger = logging.getLogger(__name__)

ROUNDS_SCHEMA = "collect_closed_loop_timing/rounds/1"
SESSION_SCHEMA = "collect_closed_loop_timing/session/1"
LOCK_POLL_S = 0.5
GENERATE_PREFIX = "生成第 "
GENERATE_SUFFIX = " 轮候选"
DISPATCH_PREFIX = "下发第 "
DISPATCH_SUFFIX = " 轮候选"
REFIT_NOTE = "适配重拟合"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class TimingBench:
    """Bench 装饰器：逐次记录 ``apply_phase`` / ``capture`` 的墙钟耗时。

    只包装，不改变设备调用次数与顺序；最近一次的耗时供 ``on_frame`` 采样。
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.last_apply_ms: float | None = None
        self.last_capture_ms: float | None = None
        self.apply_calls = 0
        self.capture_calls = 0

    def open(self) -> None:
        self.inner.open()

    def describe(self) -> dict[str, Any]:
        return self.inner.describe()

    def apply_phase(self, phase: np.ndarray) -> str:
        started = time.perf_counter()
        result = self.inner.apply_phase(phase)
        self.last_apply_ms = (time.perf_counter() - started) * 1000.0
        self.apply_calls += 1
        return result

    def capture(self) -> Any:
        started = time.perf_counter()
        result = self.inner.capture()
        self.last_capture_ms = (time.perf_counter() - started) * 1000.0
        self.capture_calls += 1
        return result

    def close(self) -> None:
        self.inner.close()


class Timeline:
    """把 runner 事件与帧回调用同一条单调时钟串起来。"""

    def __init__(self) -> None:
        self._origin: float | None = None
        self.events: list[dict[str, Any]] = []
        self._frames: dict[str, dict[str, Any]] = {}
        #: gate 失败时 runner 会发 ``round`` 事件但不写进最终摘要；这里兜底留证
        self.round_records: dict[str, Any] = {}

    def _stamp(self) -> float:
        now = time.perf_counter()
        if self._origin is None:
            self._origin = now
        return round((now - self._origin) * 1000.0, 3)

    def on_event(self, event: LoopEvent) -> None:
        note = event.payload if isinstance(event.payload, str) else None
        self.events.append(
            {
                "t_ms": self._stamp(),
                "kind": event.kind,
                "state": event.state.value if event.state is not None else None,
                "note": note,
            }
        )
        if event.kind == "round" and hasattr(event.payload, "phase_id"):
            self.round_records[str(event.payload.phase_id)] = event.payload

    def on_frame(
        self,
        captured: Any,
        record: Any,
        apply_ms: float | None = None,
        capture_ms: float | None = None,
    ) -> None:
        if record is None:
            return
        self._frames[str(record.phase_id)] = {
            "t_ms": self._stamp(),
            "frame_id": int(captured.frame_id),
            "monotonic_ns": int(captured.monotonic_ns),
            "apply_phase_ms": apply_ms,
            "ccd_capture_ms": capture_ms,
        }

    def stage_timings(self) -> dict[str, dict[str, float | None]]:
        """按 phase_id 拆出 fit/adapt/candidate；capture 由调用方合并帧采样。

        事件对的口径与 ``ClosedLoopRunner.run`` 一一对应：

        * ``fit_ms``：``适配重拟合`` → 紧随其后的 ``adaptation`` 事件；
        * ``adapt_ms``：``adaptation`` → ``生成第 N 轮候选``（含 forward 切换）；
        * ``candidate_ms``：``生成第 N 轮候选`` → ``下发第 N 轮候选``（refine_phase）。
        """
        stages: dict[str, dict[str, float | None]] = {}
        refit_started: float | None = None
        adaptation_at: float | None = None
        candidate_started: dict[int, float] = {}

        for item in self.events:
            note = item.get("note") or ""
            stamp = float(item["t_ms"])
            if item["kind"] == "state" and note == REFIT_NOTE:
                refit_started = stamp
            elif item["kind"] == "adaptation":
                adaptation_at = stamp
                if refit_started is not None:
                    stages.setdefault("__refit", {})["value"] = stamp - refit_started
                refit_started = None
            else:
                index = _round_from_note(note, GENERATE_PREFIX, GENERATE_SUFFIX)
                if index is not None:
                    candidate_started[index] = stamp
                    adapt_ms = (
                        stamp - adaptation_at if adaptation_at is not None else None
                    )
                    stages[f"r{index:02d}"] = {
                        "fit_ms": stages.get("__refit", {}).get("value"),
                        "adapt_ms": adapt_ms,
                        "candidate_ms": None,
                    }
                    adaptation_at = None
                    continue
                index = _round_from_note(note, DISPATCH_PREFIX, DISPATCH_SUFFIX)
                if index is not None and index in candidate_started:
                    entry = stages.setdefault(f"r{index:02d}", {})
                    entry["candidate_ms"] = stamp - candidate_started[index]

        stages.pop("__refit", None)
        return stages

    def frame(self, phase_id: str) -> dict[str, Any] | None:
        return self._frames.get(phase_id)


def _round_from_note(note: str, prefix: str, suffix: str) -> int | None:
    if not note.startswith(prefix) or not note.endswith(suffix):
        return None
    if len(note) <= len(prefix) + len(suffix):
        return None
    try:
        return int(note[len(prefix) : -len(suffix)].strip())
    except ValueError:
        return None


def checkpoint_grid_size(checkpoint: str) -> int | None:
    """从冻结检查点读 ``grid_size``；文件/目录均可，缺失返回 ``None``。

    ``FrozenDefocusForward`` 按检查点的 ``grid_size`` 构建 SLM 控制网格，
    ``refine_phase`` 喂给它的相位必须同尺寸，否则第一轮候选就会 shape 失配。
    """
    if not checkpoint:
        return None
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        path = path / "frozen_parameters.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        value = int(data.get("grid_size", 0))
    except (OSError, ValueError, TypeError):
        return None
    return value if value > 0 else None


def build_config(args: argparse.Namespace) -> tuple[ClosedLoopConfig, np.ndarray]:
    centre = (float(args.centre_x), float(args.centre_y))
    xy = analysis_grid(centre, int(args.span))
    target = target_field(
        xy,
        centre_xy=centre,
        half_width=float(args.half_width),
        half_height=float(args.half_height),
    )
    # start_phase 尺寸必须与 checkpoint 的 SLM 控制网格一致；--grid-size 只是
    # 无检查点时的兜底。自动对齐避免用户手填失配（plan-only 不构造 forward，
    # 手填错误要到占设备后的第一轮候选才暴露，曾致 2026-11-27 上机失败）。
    grid_size = int(args.grid_size)
    ckpt_grid = checkpoint_grid_size(args.checkpoint)
    if ckpt_grid is not None and ckpt_grid != grid_size:
        logger.info(
            "start_phase 网格从 checkpoint grid_size=%d 对齐（--grid-size=%d 忽略）",
            ckpt_grid,
            grid_size,
        )
        grid_size = ckpt_grid
    start_phase = np.zeros((grid_size, grid_size), dtype=np.float64)
    config = ClosedLoopConfig(
        xy=xy,
        target=target,
        start_phase=start_phase,
        max_rounds=int(args.max_rounds),
        candidate_policy="auto",
        adapt=bool(args.adapt),
        register_rejected_samples=bool(args.register_rejected_samples),
        exposure_ms=float(args.exposure_ms),
        step_iterations=int(args.step_iterations),
    )
    return config, xy


def plan_doc(args: argparse.Namespace, config: ClosedLoopConfig) -> dict[str, Any]:
    return {
        "schema": SESSION_SCHEMA,
        "mode": "plan",
        "execute": False,
        "out": str(Path(args.out).resolve()),
        "config": {
            "centre_xy": [args.centre_x, args.centre_y],
            "span": args.span,
            "half_width": args.half_width,
            "half_height": args.half_height,
            "grid_size": int(config.start_phase.shape[0]),
            "grid_size_cli": args.grid_size,
            "max_rounds": config.max_rounds,
            "exposure_ms": config.exposure_ms,
            "adapt": config.adapt,
            "candidate_policy": config.candidate_policy,
            "refit_every": int(args.refit_every),
            "register_rejected_samples": bool(args.register_rejected_samples),
            "step_iterations": config.step_iterations,
        },
        "device": {
            "slm_number": args.slm_number,
            "wavelength": args.wavelength,
            "camera": args.camera,
            "cam_id": args.cam_id,
            "settle_s": args.settle_s,
            "discard_frames": args.discard_frames,
        },
        "adapter": {
            "checkpoint": args.checkpoint,
            "session_dir": args.session_dir,
            "source_dataset": args.source_dataset,
            "device": "cpu",
        },
        "wait_until_idle_s": args.wait_until_idle,
        "note": "plan-only：未导入设备模块、未加锁、未创建输出目录。",
    }


def adapter_inputs_error(args: argparse.Namespace) -> str | None:
    if not args.adapt:
        return None
    names = ("checkpoint", "session_dir", "source_dataset")
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        return f"启用适配但缺少参数：{', '.join(missing)}"
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.exists():
        return f"checkpoint 不存在：{checkpoint}"
    # seed_from_source() 读 <source>/manifest.json，source 必须是目录，不是 npz 文件。
    dataset = Path(args.source_dataset).expanduser()
    if not dataset.is_dir():
        return f"source_dataset 不是目录（应含 manifest.json）：{dataset}"
    manifest = dataset / "manifest.json"
    if not manifest.is_file():
        return f"source_dataset 缺 manifest.json：{manifest}"
    try:
        rows = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"manifest.json 解析失败：{exc}"
    labels = {str(row.get("label")) for row in rows}
    if "reference_start" not in labels:
        return "source_dataset 缺 reference_start 标定行"
    # 与 fit.geometry() 对齐：每个正 ramp 都要有同周期的负 ramp，否则适配会在
    # 占设备后的 refit 处抛 "missing negative ramp"。提前到离线阶段挡住。
    for label in labels:
        parts = label.split("_")
        if len(parts) == 4 and parts[0] == "ramp" and parts[3] == "1":
            negative = f"ramp_{parts[1]}_{parts[2]}_-1"
            if negative not in labels:
                return f"source_dataset 缺负 ramp：{negative}"
    return None


def acquire_device_lock(wait_s: float) -> Any | None:
    """Acquire ``DeviceLock.default()``, polling up to ``wait_s`` seconds."""
    from ao_shaping.tools.kernels.device_mutex import DeviceLock

    lock = DeviceLock.default()
    deadline = time.monotonic() + max(0.0, float(wait_s))
    warned = False
    while True:
        if lock.acquire(timeout=0.0):
            return lock
        if not warned:
            logger.warning("bench device lock busy, waiting up to %.1fs", wait_s)
            warned = True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(LOCK_POLL_S, remaining))


def release_device_lock(lock: Any, record: dict[str, Any]) -> None:
    """Release in ``finally``; release failure is evidence, never masks exit code."""
    acquired = record.get("acquired_utc")
    if acquired is not None:
        record["held_s"] = round(
            (
                datetime.now(timezone.utc) - datetime.fromisoformat(str(acquired))
            ).total_seconds(),
            3,
        )
    try:
        lock.release()
        record["released"] = True
    except Exception as exc:  # policy: 只记录释放失败，不改变闭环退出码
        record["released"] = False
        record["release_error"] = repr(exc)
    record["released_utc"] = _utc_now()


def _round_rows(
    records: Any,
    timeline: Timeline,
    bench: TimingBench | None,
    started_utc: str,
) -> list[dict[str, Any]]:
    stages = timeline.stage_timings()
    rows: list[dict[str, Any]] = []
    for record in records:
        phase_id = str(record.phase_id)
        stage = stages.get(phase_id, {})
        frame = timeline.frame(phase_id)
        # 帧回调当场采样 bench 耗时；旧调用若没采样则退回最近一次值（兼容离线单测）
        fallback_apply = bench.last_apply_ms if bench is not None else None
        fallback_capture = bench.last_capture_ms if bench is not None else None
        apply_ms = (frame or {}).get("apply_phase_ms")
        capture_ms = (frame or {}).get("ccd_capture_ms")
        if frame is not None and apply_ms is None:
            apply_ms = fallback_apply
        if frame is not None and capture_ms is None:
            capture_ms = fallback_capture
        total_capture = (
            None if apply_ms is None or capture_ms is None else apply_ms + capture_ms
        )
        rows.append(
            {
                "schema": ROUNDS_SCHEMA,
                "utc": started_utc,
                "round_index": int(record.round_index),
                "phase_id": phase_id,
                "frame_id": int(record.frame_id),
                "gate_ok": bool(record.gate.ok),
                "gate_reasons": list(record.gate.reasons),
                "decision": record.decision,
                "adaptation": record.adaptation,
                "measured_score": record.measured_score,
                "predicted_score": record.predicted_score,
                "fit_ms": _ms(stage.get("fit_ms")),
                "adapt_ms": _ms(stage.get("adapt_ms")),
                "candidate_ms": _ms(stage.get("candidate_ms")),
                "capture_ms": _ms(total_capture),
                "apply_phase_ms": _ms(apply_ms),
                "ccd_capture_ms": _ms(capture_ms),
                "note": record.note,
            }
        )
    return rows


def _ms(value: Any) -> float | None:
    return None if value is None else round(float(value), 3)


def write_evidence(out_dir: Path, rows: list[dict[str, Any]], session: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds_path = out_dir / "rounds.jsonl"
    with rounds_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (out_dir / "session.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GUI 闭环逐轮分项耗时采集；无 --execute 为 plan-only，不占设备。"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--out", required=True, help="证据目录（--execute 时创建）")
    parser.add_argument("--execute", action="store_true", help="真正占用 SLM/CCD 运行")
    parser.add_argument("--max-rounds", type=int, default=8)
    # 默认曝光取探针验证过的 0.05 ms（8-bit 峰值约 117，留足饱和裕度）；
    # 20 ms 在 8-bit 下会顶满量程，gate 会因过曝失效。
    parser.add_argument("--exposure-ms", type=float, default=0.05)
    # 默认中心取光路探针实测质心 / frozen origin（原始 CCD 帧像素坐标）。
    # analysis_grid 直接在原始帧采样，若用 (500,500) 窗口会落在空白区、分数失真。
    parser.add_argument("--centre-x", type=float, default=1228.0)
    parser.add_argument("--centre-y", type=float, default=1107.0)
    parser.add_argument("--span", type=int, default=4)
    parser.add_argument("--half-width", type=float, default=2.0)
    parser.add_argument("--half-height", type=float, default=2.0)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--step-iterations", type=int, default=60)
    # 在线适配重拟合间隔：默认 2 保持现状语义；设 1 可在首个被接受轮次后立即重拟合。
    parser.add_argument(
        "--refit-every",
        type=int,
        default=2,
        help="在线适配重拟合间隔（默认 2）；设 1 可在首个被接受轮次后立即重拟合",
    )
    # 默认关闭：被拒轮次仍只作为判读证据，开启后才登记为适配样本。
    parser.add_argument(
        "--register-rejected-samples",
        action="store_true",
        help="门禁通过但被拒的轮次也登记为适配样本（默认关闭）",
    )
    parser.add_argument("--no-adapt", dest="adapt", action="store_false", default=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--session-dir", default="")
    parser.add_argument("--source-dataset", default="")
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--wavelength", type=int, default=1064)
    parser.add_argument("--camera", default="daheng")
    parser.add_argument("--cam-id", default=0)
    parser.add_argument("--settle-s", type=float, default=0.15)
    parser.add_argument("--discard-frames", type=int, default=3)
    parser.add_argument("--wait-until-idle", type=float, default=300.0)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    bench_factory: Callable[[], Any] | None = None,
    adapter_factory: Callable[[np.ndarray], Any] | None = None,
) -> int:
    """跑一轮采集；工厂参数只给离线端到端测试注入假设备，生产路径不传。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    config, xy = build_config(args)

    if not args.execute:
        print(json.dumps(plan_doc(args, config), ensure_ascii=False, indent=2))
        return 0

    # 注入的适配器工厂跳过路径校验（与页面的 _adapter_factory 语义一致）
    if adapter_factory is None:
        problem = adapter_inputs_error(args)
        if problem is not None:
            logger.error("适配输入校验失败，未占用设备：%s", problem)
            return 2

    if bench_factory is None:

        def bench_factory() -> Any:
            from ao_shaping.gui.slm.closed_loop_bench import HardwareBench

            return HardwareBench(
                slm_number=int(args.slm_number),
                wavelength=int(args.wavelength),
                camera_type=str(args.camera),
                cam_id=args.cam_id,
                exposure_ms=float(args.exposure_ms),
                settle_s=float(args.settle_s),
                discard_frames=int(args.discard_frames),
            )

    if adapter_factory is None:

        def adapter_factory(grid: np.ndarray) -> Any:
            return OnlineTransferAdapter(
                checkpoint=args.checkpoint,
                session_dir=args.session_dir,
                span=int(args.span),
                source_dataset=args.source_dataset,
                # 重拟合间隔由 CLI 控制：1 表示首个被接受轮次后立即重拟合。
                policy=AdaptationPolicy(refit_every=int(args.refit_every)),
                # refine_phase 的 init/target 在 CPU（runner 调用点不传 device），
                # forward 必须同设备，否则 cuda/cpu 混算报 device 失配。
                device="cpu",
            )

    lock_record: dict[str, Any] = {
        "acquired_utc": None,
        "released_utc": None,
        "released": None,
        "release_error": None,
        "held_s": None,
    }
    lock = acquire_device_lock(args.wait_until_idle) if args.wait_until_idle > 0 else None
    if args.wait_until_idle > 0 and lock is None:
        logger.error("等待设备空闲超时，未启动闭环")
        return 3
    if lock is not None:
        lock_record["acquired_utc"] = _utc_now()

    started_utc = _utc_now()
    timeline = Timeline()
    summary: Any = None
    bench: TimingBench | None = None
    bench_info: dict[str, Any] = {}
    exit_code = 0
    try:
        adapter = adapter_factory(xy) if config.adapt else None
        inner = bench_factory()
        bench = TimingBench(inner)

        def on_frame(captured: Any, record: Any) -> None:
            # 必须在回调当场采样：全局 last_* 到末尾会被后续轮覆盖
            timeline.on_frame(
                captured,
                record,
                bench.last_apply_ms if bench is not None else None,
                bench.last_capture_ms if bench is not None else None,
            )

        runner = ClosedLoopRunner(
            bench,
            config=config,
            forward=None,
            adapter=adapter,
            on_event=timeline.on_event,
            on_frame=on_frame,
        )
        summary = runner.run()
        bench_info = dict(inner.describe())
        state = str(summary.state.value if hasattr(summary.state, "value") else summary.state)
        logger.info("闭环结束：state=%s reason=%s rounds=%d", state, summary.reason, len(summary.rounds))
        if state.upper() not in {"COMPLETED", "STOPPED"}:
            exit_code = 4
    except Exception as exc:  # policy: 现场异常要落盘再退出，不能静默
        logger.exception("闭环采集异常：%s", exc)
        exit_code = 5
    finally:
        if lock is not None:
            release_device_lock(lock, lock_record)

    # gate 失败轮不会进 summary.rounds，但 on_frame/on_event 已留证，必须落盘
    records_by_id: dict[str, Any] = {}
    if summary is not None:
        for record in getattr(summary, "rounds", ()):
            records_by_id[str(record.phase_id)] = record
    records_by_id.update(timeline.round_records)
    rows = _round_rows(
        [records_by_id[key] for key in sorted(records_by_id)],
        timeline,
        bench,
        started_utc,
    )
    session: dict[str, Any] = {
        "schema": SESSION_SCHEMA,
        "mode": "execute",
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "exit_code": exit_code,
        "config": plan_doc(args, config)["config"],
        "device": bench_info,
        "lock": lock_record,
        "state": (
            str(summary.state.value if hasattr(summary.state, "value") else summary.state)
            if summary is not None
            else None
        ),
        "reason": getattr(summary, "reason", None) if summary is not None else None,
        "rounds_recorded": len(rows),
        "timeline": timeline.events,
    }
    write_evidence(Path(args.out), rows, session)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
