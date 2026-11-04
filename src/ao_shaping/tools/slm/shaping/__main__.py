"""``python -m ao_shaping.tools.slm.shaping``：整形路线建议器 CLI。

读取一次 transfer 会话 / 采集目录的离线产物（或仅靠 ``--set`` 提供实测判据），
构建探测报告，交给 router 选一条整形路线，打印建议。

本命令是**建议器**：绝不执行任何整形 / 优化。输出含「需人工确认后才执行」，
由用户确认后自行运行对应方法。

用法示例：:

    python -m ao_shaping.tools.slm.shaping \\
        --session-dir data/debug/slm_transfer/square1200_20261026_1350 \\
        --intent stable_small
    python -m ao_shaping.tools.slm.shaping --set snr=1.0   # no_go，退出码 2
    python -m ao_shaping.tools.slm.shaping --session-dir <dir> --json --out plan.json

退出码：路由到 ``no_go`` 返回 2，其它返回 0。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ao_shaping.tools.slm.shaping.probe import build_report, parse_overrides
from ao_shaping.tools.slm.shaping.router import (
    RoutePlan as _RoutePlan,
    choose_route,
    format_plan,
    plan_to_dict,
)

# 支持的任务意图（与 router 契约一致）。
_INTENT_CHOICES = (
    "auto",
    "extended_field",
    "new_mask",
    "fixed_shape",
    "uniformity",
    "cold_start",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ao_shaping.tools.slm.shaping",
        description="整形路线建议器：读取探测产物并输出推荐方法与红线，绝不执行整形。",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="一次 transfer 会话/采集目录；不提供则全靠 --set 提供判据",
    )
    parser.add_argument(
        "--intent",
        choices=_INTENT_CHOICES,
        default="auto",
        help="任务意图（auto|extended_field|new_mask|fixed_shape|uniformity|cold_start）",
    )
    parser.add_argument(
        "--shape",
        default="square",
        help="目标形状（square|rectangle|triangle 等）",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="冻结模型显式路径（判定 has_checkpoint）",
    )
    parser.add_argument(
        "--seed-run",
        type=Path,
        default=None,
        help="种子运行显式路径（判定 has_seed_run）",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖任意 ProbeReport 字段（可重复；值先尝试 json.loads，失败按字符串）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="只输出 plan 的 JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="把 plan JSON 写盘（与输出共存）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回退出码（no_go→2，其它→0）。"""
    args = _build_parser().parse_args(argv)

    overrides = _coerce_overrides(args.set)
    report = build_report(
        session_dir=args.session_dir,
        intent=args.intent,
        shape=args.shape,
        checkpoint=args.checkpoint,
        seed_run=args.seed_run,
        overrides=overrides,
    )
    plan: _RoutePlan = choose_route(report)
    plan_dict: dict[str, Any] = plan_to_dict(plan)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(plan_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    rendered = (
        json.dumps(plan_dict, ensure_ascii=False, indent=2)
        if args.json
        else format_plan(plan)
    )
    print(rendered)

    return 2 if plan.route_id == "no_go" else 0


def _coerce_overrides(pairs: list[str]) -> dict[str, Any]:
    """CLI 的 --set 覆盖对 → dict（值经 JSON 尝试 / 字符串兜底）。"""
    return parse_overrides(pairs)


if __name__ == "__main__":
    raise SystemExit(main())
