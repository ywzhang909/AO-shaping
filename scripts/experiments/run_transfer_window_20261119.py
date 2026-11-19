"""一次窗口串行跑 transfer 放行段: K=1 / K=4 / ccd_mode light / ccd_mode none。

对应计划
--------
* `docs/dev/hw_plan_consolidated_20261118.md` §4 (S 段: K>1 真闭环、ccd_mode light/none)
* `docs/dev/next_hw_plan_transfer_candidates_issue247_20261117.md` 任务 A (K=4)
* `docs/dev/hw_ccd_mode_test_plan_20261118.md` L1 / N1

为什么不是一行 ``python -m ... run``
-----------------------------------
``transfer run`` 不加 bench 锁、没有 ``--wait-until-idle``、且 ``run_session`` 用
``exist_ok=False`` 建目录。本脚本用 ``_window.py`` 的信封补上「等锁 → 段间冷却 →
每段独立 ``--out`` → 释放 → 证据落盘」, 并把每段 argv 固定成计划里的写法, 避免
手敲时漏掉 ``--dataset/--checkpoint`` (light/none 缺这两个会 fail-fast)。

采集数据一律落 ``data/experiments/<window>/<segment>/``。

纪律
----
* 没有 ``--execute``: 只打印每段的完整 argv 与落盘目录, 不建目录、不碰设备;
* light / none 必须显式给 ``--dataset`` 与 ``--checkpoint`` (计划要求, 防静默退化);
* 段间冷却默认 200 s (背对背 SLM open/close 会触发驱动板假死 ``ReadSU=-1000``);
* 任一段非零退出即停止整个窗口, 不自动续跑 (非零退出通常是驱动板/USB 故障);
* 占用期不做任何分析: 判读在设备下线后离线做。

用法 (dry-run 默认):

    python scripts/experiments/run_transfer_window_20261119.py \
      --window data/experiments/transfer_window_20261119 \
      --dataset data/debug/slm_transfer/<session>/capture \
      --checkpoint data/debug/slm_transfer/<session>/fit/frozen_parameters.json \
      --segments k1 k4 light none --shapes square --side 1200

加 ``--execute`` 才真正占用设备 (需先获用户授权并通过六步光路探测)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _window import (  # 同目录信封, 必须在 sys.path 调整之后导入
    DATA_ROOT,
    ROOT,
    Segment,
    add_window_args,
    run_window,
)

#: 计划里各段的默认含义 (evidence 里写清楚这一段回答什么)。
PURPOSES = {
    "k1": "K=1 基线: 同窗口对照, 与 K=4 比同量级 (不测加速)",
    "k4": "K=4 真闭环: 只验在线实现可用 (优化器占端到端 <0.5%)",
    "light": "ccd_mode light: 复用探针 + 每块一次锚点 + 终检 A-B-B-A 的质量损失",
    "none": "ccd_mode none: 纯模型相位上屏的远场形态 (acceptance=open_loop)",
}

#: 需要显式 dataset + checkpoint 的段 (计划: --reuse-latest 不为这两个模式解析路径)。
NEEDS_MODEL = ("light", "none")


def build_segments(args: argparse.Namespace) -> list[Segment]:
    """把 CLI 展开成「段名 → 完整 argv」; 不触碰设备。"""
    if not args.window:
        raise SystemExit("--window 必填: 证据与数据都要有落脚点")
    window = Path(args.window)
    if not window.is_absolute():
        window = ROOT / window

    dataset = Path(args.dataset) if args.dataset else None
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if dataset is not None and not dataset.is_absolute():
        dataset = ROOT / dataset
    if checkpoint is not None and not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    model_args: list[str] = []
    if dataset is not None:
        model_args += ["--dataset", str(dataset)]
    if checkpoint is not None:
        model_args += ["--checkpoint", str(checkpoint)]

    shared: list[str] = []
    # 与窗口加锁的目标保持同一设备 (否则窗口锁 #1、子段开 #2)。
    shared += ["--slm-number", str(args.slm_number), "--cam-id", str(args.cam_id)]
    if args.shapes:
        shared += ["--shapes", *args.shapes]
    if args.side is not None:
        shared += ["--side", str(args.side)]
    if args.iterations is not None:
        shared += ["--iterations", str(args.iterations)]
    if args.count is not None:
        shared += ["--count", str(args.count)]
    if args.frames is not None:
        shared += ["--frames", str(args.frames)]
    if args.seed is not None:
        shared += ["--seed", str(args.seed)]
    shared += list(args.transfer_arg or [])

    segments: list[Segment] = []
    for name in args.segments:
        extra: list[str] = []
        requires: tuple[str, ...] = ()
        if name == "k1":
            extra = ["--candidates", "1"]
        elif name == "k4":
            extra = ["--candidates", "4"]
        elif name in NEEDS_MODEL:
            extra = ["--ccd-mode", name]
            requires = ("dataset", "checkpoint")
        else:  # pragma: no cover - argparse choices 已挡住
            raise SystemExit(f"未知段: {name}")
        segments.append(
            Segment(
                name=name,
                argv=(
                    sys.executable,
                    "-m",
                    "ao_shaping.tools.slm.transfer",
                    "run",
                    "--out",
                    str(window / name),
                    "--execute",
                    *model_args,
                    *extra,
                    *shared,
                ),
                out=window / name,
                purpose=PURPOSES[name],
                requires=requires,
            )
        )
    return segments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "一次窗口串行跑 transfer 放行段 (k1/k4/light/none); "
            "不带 --execute 只打印计划。"
        )
    )
    parser.add_argument(
        "--window",
        default=None,
        help=f"窗口目录 (建议 {DATA_ROOT}/transfer_window_<date>); 不允许已存在非空",
    )
    parser.add_argument(
        "--segments",
        nargs="+",
        default=["k1", "k4"],
        choices=["k1", "k4", "light", "none"],
        help="按给定顺序执行 (默认 k1 k4; light/none 需要 --dataset 与 --checkpoint)",
    )
    parser.add_argument("--dataset", default=None, help="复用采集会话的 capture 目录")
    parser.add_argument(
        "--checkpoint", default=None, help="复用检查点 (frozen_parameters.json)"
    )
    parser.add_argument(
        "--shapes", nargs="+", default=["square"], help="目标形状 (默认 square)"
    )
    parser.add_argument("--side", type=int, default=1200, help="目标边长 px (默认 1200)")
    parser.add_argument("--iterations", type=int, default=None, help="透传 --iterations")
    parser.add_argument("--count", type=int, default=None, help="透传 --count")
    parser.add_argument("--frames", type=int, default=None, help="透传 --frames")
    parser.add_argument("--seed", type=int, default=None, help="透传 --seed")
    parser.add_argument(
        "--transfer-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="其它原样透传给 transfer run 的参数 (可重复)",
    )
    add_window_args(parser)
    args = parser.parse_args(argv)

    segments = build_segments(args)

    missing: list[str] = []
    for seg in segments:
        for need in seg.requires:
            value = getattr(args, need, None)
            if not value:
                missing.append(f"段 {seg.name} 需要 --{need}")
            else:
                path = Path(value)
                path = path if path.is_absolute() else ROOT / path
                if not path.exists():
                    missing.append(f"段 {seg.name} 的 --{need} 不存在: {path}")
    if missing:
        print("拒绝执行:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 2

    window = Path(args.window)
    window = window if window.is_absolute() else ROOT / window
    return run_window(
        segments,
        window=window,
        cooldown_s=args.cooldown_s,
        wait_s=args.wait_until_idle,
        timeout_s=args.segment_timeout_s,
        slm_number=args.slm_number,
        cam_id=args.cam_id,
        execute=args.execute,
        extra={
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "shapes": list(args.shapes),
            "side": args.side,
            "segments": list(args.segments),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
