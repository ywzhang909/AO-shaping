"""一次窗口串行采对称二值光栅的**多周期**扫描: issue #238 剩余实验增量。

对应计划
--------
* `docs/dev/issue238_binary_gray_capture_plan_20261119.md` §6: 16 px (run1) 的系统
  ±13 来自像素响应模糊 —— e0 地板偏高、±1 极小分裂被抹平; 「加大周期 (32/48 px)
  是唯一有新增价值的实验增量」, 采集脚本已支持 ``--period-px``, 无需改代码。
* 同文 §6.1: ``run2_period32`` 在 725 帧 (gray 0..928, 36 个灰度层) 报
  ``SantecError ... -10032 USB其他错误`` 中止, finally 关闭 SLM/相机/锁均 ok 但平场
  归零失败, 已采部分退化, **必须在全新目录完整重跑**。

为什么需要这层信封, 而不是逐周期手敲 ``issue238_gray_phase_probes.py``
----------------------------------------------------------------------
(a) 探针脚本自带 ``--wait-until-idle`` 且段内自己 ``acquire`` bench 锁: 逐周期手敲
    会形成「释放 → 立刻重开」的背对背 SLM open/close, 这正是驱动板假死的触发方式
    (``ReadSU=-1000`` / ``Open=-10002``, 需用户手动重启);
(b) 合并计划要求一次占用跑完所有段、段间 ≥200 s 冷却、占用期不做任何分析。窗口负责
    **串行 + 冷却**, 但**不**持锁: 每个周期脚本在整段采集期间自己拿着 bench 锁
    (``lock_mode="segment"``), 窗口把真实的 ``--wait-until-idle`` 转发给它;
(c) ``--wait-until-idle 0`` **不会**让子脚本跳过锁 —— 探针仍会 ``acquire``, 0 只把
    等待上限设成 0, 被占就立刻 ``exit 2``。2026-11-19 ``run3_period_sweep`` 的
    ``period32`` 正是踩了这个坑 (窗口持锁 + 子段 0 ⇒ 子段被父进程的锁挡在门外);
(d) 重跑要求目录全新 (子脚本用 ``exist_ok=False`` 建 ``--out``), 由 ``run_window``
    统一挡: 窗口目录或任一子段目录已存在即拒绝, 避免把退化数据混进新证据。

本脚本因此只是 ``_window.py`` 的瘦封装: 每周期一段、独立 ``--out``、argv 固定,
不做任何判读。

落盘
----
采集数据落 ``data/experiments/issue238_binary_phase_20261119/<window>/period<P>/``;
窗口证据 (``window_plan.json`` / ``window.json`` / ``segments/*.log`` / ``released.json``)
落窗口根目录。

纪律
----
* 没有 ``--execute``: 只打印每段完整 argv 与落盘目录, 不建目录、不导入设备模块;
* 一个周期 = 一次完整灰度扫描 (粗步 32 + 864..1023 步 8, 约 51 图案 × 帧数),
  **单周期预期 ~10 min**, 两个周期加上段间冷却约 25 min, 期间不要开第二个会话;
* 只采集: g2pi 地标峰 / ±1 极小分裂 / 系统包络的拟合全部在设备下线后离线做;
* 任一段非零退出即停止整个窗口, 不自动续跑 (非零退出通常是驱动板 / USB 故障,
  继续只会扩大损失)。

用法 (dry-run 默认):

    python scripts/experiments/run_issue238_period_sweep_20261119.py \
      --run-dir data/experiments/issue238_binary_phase_20261119/run3_period_sweep \
      --periods-px 32 48

加 ``--execute`` 才真正占用设备 (需先获用户授权并通过六步光路探测)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _window import (  # 同目录信封, 必须在 sys.path 调整之后导入
    DATA_ROOT,
    LOCK_MODE_SEGMENT,
    ROOT,
    Segment,
    add_window_args,
    run_window,
)

#: 周期合法性 (px): 过小被像素响应模糊吃掉对比度, 过大则一个周期内光斑不再均匀。
PERIOD_MIN_PX = 8
PERIOD_MAX_PX = 256

#: 计划点名的两个周期各自的含义 (证据里写清楚这一段回答什么)。
PURPOSES: dict[int, str] = {
    32: "P=32 px 干净重跑 (run2 在 725 帧 USB -10032 中止): 重测 g2pi 地标峰",
    48: "P=48 px: 更大周期进一步压像素响应模糊, 收窄 ±1 极小分裂",
}


def purpose_for(period_px: int) -> str:
    """段用途一行说明; 计划外周期给通用描述。"""
    return PURPOSES.get(
        period_px, f"P={period_px} px: 该周期的 g2pi 地标峰与 ±1 极小分裂"
    )


def ordered_periods(
    periods_px: list[int], order: str = "as-given"
) -> list[int]:
    """展开窗口的周期拍摄顺序 (纯函数, 不建目录、不碰设备)。

    * ``as-given``: 原样返回 (历史行为);
    * ``palindrome``: 先按给定顺序、再反向重复一遍, 即 ``32,48,64`` 展开成
      ``32,48,64,64,48,32``。这样每个周期在时间上各有前后两个邻段, 周期效应
      与整段束功率漂移不再共线, 离线可按段块归一。中间周期会出现两次 (两段
      同为该周期, 时间上仍严格对称)。

    单周期输入无法构成回文, 由 :func:`validate` 拒绝。
    """
    base = [int(p) for p in periods_px]
    if order == "as-given":
        return base
    if order != "palindrome":
        raise ValueError(f"unknown period order {order!r}")
    return base + base[::-1]


def resolve_run_dir(raw: str | None) -> Path | None:
    """``--run-dir`` 解析: 相对路径按仓库根; 缺省返回 ``None`` (交给校验报错)。"""
    if not raw:
        return None
    run_dir = Path(raw)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    return run_dir


def validate(args: argparse.Namespace) -> list[str]:
    """窗口前置校验: 全部问题一次列出; **不建目录、不碰设备**。"""
    problems: list[str] = []
    if not args.run_dir:
        problems.append(
            "--run-dir 必填: 窗口目录既是证据落脚点, 也是各周期 --out 的父目录"
            f" (建议 {DATA_ROOT}/issue238_binary_phase_20261119/run3_period_sweep)"
        )
    bad = [p for p in args.periods_px if p < PERIOD_MIN_PX or p > PERIOD_MAX_PX]
    if bad:
        problems.append(
            f"--periods-px 越界 {bad}: 周期必须在 "
            f"{PERIOD_MIN_PX}..{PERIOD_MAX_PX} px 之间 (过小受像素响应模糊限制)"
        )
    seen: set[int] = set()
    dupes: list[int] = []
    for period_px in args.periods_px:
        if period_px in seen and period_px not in dupes:
            dupes.append(period_px)
        seen.add(period_px)
    if dupes:
        problems.append(f"--periods-px 重复 {dupes}: 同一周期只跑一次")
    if getattr(args, "period_order", "as-given") == "palindrome":
        if len(set(args.periods_px)) < 2:
            problems.append(
                "--period-order palindrome 至少需要 2 个不同周期 "
                "(单周期展开后仍只有一个周期, 无法与漂移分离)"
            )
    if getattr(args, "power_reference", "off") == "abba":
        if getattr(args, "power_reference_frames", 2) < 1:
            problems.append("--power-reference-frames >= 1 (abba 每侧至少 1 帧)")
    return problems


def shared_flags(args: argparse.Namespace) -> list[str]:
    """透传给探针脚本的可选项 (只转发显式给出的)。

    ``--frames-per-level`` 总是转发 (本窗口默认 20, 与 run2 的口径一致);
    ``--sweep-step`` 不转发: binary 模式忽略它。``--power-reference`` 只在
    非 ``off`` 时转发, 缺省 argv 与历史窗口逐字一致。
    """
    flags: list[str] = ["--frames-per-level", str(args.frames_per_level)]
    for name, value in (
        ("--dense-lo", args.dense_lo),
        ("--dense-hi", args.dense_hi),
        ("--dense-step", args.dense_step),
        ("--checker-frames", args.checker_frames),
        ("--settle-s", args.settle_s),
    ):
        if value is not None:
            flags += [name, str(value)]
    if args.exposure_bracket is not None:
        flags += ["--exposure-bracket", *(str(ms) for ms in args.exposure_bracket)]
    power_reference = getattr(args, "power_reference", "off")
    if power_reference != "off":
        # 只在显式启用时转发: 缺省 argv 与历史窗口完全一致。
        flags += ["--power-reference", str(power_reference)]
        if power_reference == "abba":
            flags += [
                "--power-reference-frames",
                str(args.power_reference_frames),
            ]
    return flags


def build_segments(args: argparse.Namespace, run_dir: Path) -> list[Segment]:
    """把 CLI 展开成「周期 → 完整子进程 argv」; 不触碰设备、不建目录。"""
    probe = ROOT / "scripts" / "slm" / "issue238_gray_phase_probes.py"
    flags = shared_flags(args)
    segments: list[Segment] = []
    occurrences: dict[int, int] = {}
    ordered = ordered_periods(
        args.periods_px, getattr(args, "period_order", "as-given")
    )
    for period_px in ordered:
        occurrences[period_px] = occurrences.get(period_px, 0) + 1
        repeat = occurrences[period_px]
        # as-given 下 validate() 已拒绝重复周期, 名字仍是 period<P>;
        # palindrome 第二遍才加 _r2 后缀, 避免同名同目录。
        name = f"period{period_px}" if repeat == 1 else f"period{period_px}_r{repeat}"
        out = run_dir / name
        segments.append(
            Segment(
                name=name,
                argv=(
                    sys.executable,
                    str(probe),
                    "--out",
                    str(out),
                    "--mode",
                    "binary",
                    "--period-px",
                    str(period_px),
                    "--execute",
                    # 子段锁模式: 窗口不持锁, 真实等待值原样转发; 子脚本自己
                    # acquire/release 整段。0 只表示「被占就立刻失败」, 不是跳过锁。
                    "--wait-until-idle",
                    str(args.wait_until_idle),
                    # 与窗口加锁的目标保持同一设备 (否则窗口锁 #1、子段开 #2)。
                    "--slm-number",
                    str(args.slm_number),
                    "--cam-id",
                    str(args.cam_id),
                    *flags,
                ),
                out=out,
                purpose=purpose_for(period_px),
            )
        )
    return segments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "一次窗口串行采对称二值光栅多周期扫描 (issue #238 §6.1: 32/48 px); "
            "不带 --execute 只打印计划。"
        )
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "窗口目录 (相对路径按仓库根解析); 不允许已存在非空 "
            f"如 {DATA_ROOT}/issue238_binary_phase_20261119/run3_period_sweep"
        ),
    )
    parser.add_argument(
        "--periods-px",
        nargs="+",
        type=int,
        default=[32, 48],
        help="光栅周期 (panel px), 每周期一段; 默认 32 48",
    )
    parser.add_argument(
        "--frames-per-level",
        type=int,
        default=20,
        help="[透传] 每灰度层帧数 (默认 20, 与 run2 口径一致)",
    )
    parser.add_argument(
        "--dense-lo", type=int, default=None, help="[透传] 密集带低灰度 (子脚本默认 864)"
    )
    parser.add_argument(
        "--dense-hi", type=int, default=None, help="[透传] 密集带高灰度 (子脚本默认 1023)"
    )
    parser.add_argument(
        "--dense-step", type=int, default=None, help="[透传] 密集带灰度步 (子脚本默认 8)"
    )
    parser.add_argument(
        "--checker-frames",
        type=int,
        default=None,
        help="[透传] 棋盘每灰度帧数 (子脚本默认 20)",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=None,
        help="[透传] 每次上屏后的稳定时间 (子脚本默认 0.3)",
    )
    parser.add_argument(
        "--exposure-bracket",
        type=float,
        nargs="+",
        default=None,
        metavar="MS",
        help="[透传] 曝光梯度 (ms), 原样转发 (子脚本要求 >=4 步且全为正)",
    )
    parser.add_argument(
        "--period-order",
        choices=("as-given", "palindrome"),
        default="as-given",
        help=(
            "周期拍摄顺序: as-given = 原样 (默认); palindrome = 先升后降, "
            "如 32 48 64 -> 32 48 64 64 48 32, 使周期效应与整段漂移可分"
        ),
    )
    parser.add_argument(
        "--power-reference",
        choices=("off", "abba"),
        default="off",
        help=(
            "[透传] 束功率参考轨: off = 历史行为 (不转发); "
            "abba = 每段前后各采一组平场参考 (固定曝光)"
        ),
    )
    parser.add_argument(
        "--power-reference-frames",
        type=int,
        default=2,
        help="[透传] abba 每侧参考帧数 (默认 2)",
    )
    add_window_args(parser)
    args = parser.parse_args(argv)

    problems = validate(args)
    if problems:
        print("拒绝执行: 参数不合法", file=sys.stderr)
        for item in problems:
            print(f"  - {item}", file=sys.stderr)
        return 2

    run_dir = resolve_run_dir(args.run_dir)
    assert run_dir is not None  # validate() 已挡
    segments = build_segments(args, run_dir)
    return run_window(
        segments,
        window=run_dir,
        cooldown_s=args.cooldown_s,
        wait_s=args.wait_until_idle,
        timeout_s=args.segment_timeout_s,
        slm_number=args.slm_number,
        cam_id=args.cam_id,
        execute=args.execute,
        # 每段脚本自取自放 bench 锁; 窗口只串行 + 冷却 + 只读回读锁状态 (#238
        # period32 复盘: 窗口持锁 + 子段 --wait-until-idle 0 必然父子互锁)。
        lock_mode=LOCK_MODE_SEGMENT,
        extra={
            "run_dir": str(run_dir),
            "lock_mode": LOCK_MODE_SEGMENT,
            "periods_px": list(args.periods_px),
            "period_order": args.period_order,
            "periods_px_ordered": ordered_periods(
                args.periods_px, args.period_order
            ),
            "power_reference": args.power_reference,
            "power_reference_frames": (
                args.power_reference_frames
                if args.power_reference == "abba"
                else None
            ),
            "frames_per_level": args.frames_per_level,
            "exposure_bracket": (
                list(args.exposure_bracket)
                if args.exposure_bracket is not None
                else None
            ),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
