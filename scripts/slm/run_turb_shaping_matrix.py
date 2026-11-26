"""湍流整形对比择优矩阵 (#253 持续优化轮) —— 多变体 × 多湍流强度的离线扫掠。

对每个 (变体, cn2) 单元: 经 ``slm-turb-shaping`` 的 ``run_arm`` 跑一次数字孪生
闭环 (不复制其接线), 再经报告生成器的 ``build_report`` 出单元报告 (相位/光斑
双 GIF); 汇总矩阵报告给出**每档湍流下哪个变体收敛最好/最快**的对比择优表与
逐单元链接。固定种子 → 全矩阵可复现; ``--dry-run`` 只打印单元计划, 零执行零读
数据 (CI 环境契约)。

MIL / GS-refine 暂不入矩阵: 二者的 sim 孪生没有湍流通道 (``_SimBench`` 钉死
cn2=0, 且 ``slm_shaping_bench.forward_intensity`` 的 cn2 是实测死配置) —— 给
孪生开扰动通道是独立改造, 见 issue #253 注记。

用法 (仓库根):
    python scripts/slm/run_turb_shaping_matrix.py --epochs 60 --out-root data/turb_matrix
    python scripts/slm/run_turb_shaping_matrix.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

#: 矩阵轴 (2026-11-21 标定后冻结): 三个 SPGD 族变体 × 三档湍流强度。
#: cn2 与 distance_m(=500, runner 默认) 是退化的生成器旋钮, 档位按**实测 σ_turb**
#: 标定 (~0.09 / ~0.40 / ~1.8 rad); 第一版网格 (distance=1) 实测 σ 仅 0.006-0.03
#: rad, 湍流近乎名义值, 无实验区分度, 已作废重跑 (台账第 179 批)。
#: 追加变体只增尾不插位 (slot_of 基线索引不变, 老单元种子配对永久稳定);
#: gs_refine = M5a freeform (湍流通道免费来自 SimPibCCD 真光路);
#: model_in_loop = M5b 解析孪生注入 (#253 后续批已落地: _SimBench disturbance,
#: 验收语义要求准静态, 故其单元固定 turb_mode=static)。
VARIANTS = ("spgd", "square", "heuristic", "gs_refine", "model_in_loop")
CN2_GRID = (1e-14, 2e-13, 2e-12)
BASE_DIST_SEED = 20261121


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--cam-size", type=int, default=256)
    ap.add_argument("--n-max", type=int, default=4)
    ap.add_argument("--target-side", type=float, default=120.0)
    ap.add_argument("--pop-size", type=int, default=12,
                    help="heuristic 变体种群规模 (M5c: 种群须随维度配平)")
    ap.add_argument("--out-root", type=Path, default=Path("data/turb_matrix"))
    ap.add_argument(
        "--report-root", type=Path, default=None,
        help="报告根 (默认 report/turbulence_shaping/matrix_<UTC>)",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印单元计划, 不执行")
    ap.add_argument("--variants", default=None,
                    help="逗号分隔覆盖扫掠变体, 缺省用冻结网格 VARIANTS")
    ap.add_argument("--cn2-grid", default=None,
                    help="逗号分隔覆盖 cn2 档位, 缺省用冻结网格 CN2_GRID")
    return ap.parse_args(argv)


def axes(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[float, ...]]:
    variants = tuple(v.strip() for v in args.variants.split(",")) if args.variants else VARIANTS
    cn2_grid = tuple(float(x) for x in args.cn2_grid.split(",")) if args.cn2_grid else CN2_GRID
    return variants, cn2_grid


#: dist_seed 槽位: 老变体复现历史 cn2-major enumerate (stride=3, m2b..m5c 全部批次:
#: spgd@2e-12=6, square@2e-13=4, heuristic@2e-12=8), gs_refine 用独立基址段 12+
#: (最大老槽 8, 无碰撞)。**历史槽位与 gs_refine 基址都是冻结契约** —— 改动任何
#: 一个都会让已提交报告不可配对。MIL (M5b) 用基址段 24+ (gs_refine 占 12..14, 无碰撞)。
VARIANTS = ("spgd", "square", "heuristic", "gs_refine", "model_in_loop")
_HISTORICAL_SLOT_STRIDE = 3
GS_REFINE_SEED_BASE = 12
MIL_SEED_BASE = 24


def slot_of(variant: str, cn2: float) -> int:
    """dist_seed 槽位 (冻结契约见 ``_HISTORICAL_SLOT_STRIDE``); 与子扫掠枚举顺序无关。"""
    if variant == "gs_refine":
        return GS_REFINE_SEED_BASE + CN2_GRID.index(cn2)
    if variant == "model_in_loop":
        return MIL_SEED_BASE + CN2_GRID.index(cn2)
    return CN2_GRID.index(cn2) * _HISTORICAL_SLOT_STRIDE + VARIANTS.index(variant)


def cell_plan(args: argparse.Namespace, variant: str, cn2: float, slot: int) -> dict:
    """一个单元的完整配置字典 (传给 run_arm 的 TurbShapingParams)。"""
    return {
        "algorithm": variant,
        "search_algorithm": "pso" if variant == "heuristic" else "ga",
        "pop_size": args.pop_size,
        # MIL 的 Step A 拟合 + 逐轮验收假设准静态波前 (dynamic 下验收必然被
        # 新屏翻盘, 整格只会得到"保平场"空结论) —— 静态屏才是它的适用域。
        "turb_mode": "static" if variant == "model_in_loop" else "dynamic",
        "cn2": cn2,
        "dist_seed": BASE_DIST_SEED + slot,
        "epochs": args.epochs,
        "cam_size": args.cam_size,
        "n_max": args.n_max,
        "target_side": args.target_side,
        "out_dir": str(Path(args.out_root) / f"{variant}_cn2{cn2:g}"),
        "report": False,
        "dry_run": False,
    }


def run_matrix(args: argparse.Namespace) -> list[dict]:
    from ao_shaping.runners.slm.turb_shaping_runner import TurbShapingParams, run_arm

    results: list[dict] = []
    variants, cn2_grid = axes(args)
    for cn2 in cn2_grid:
        for variant in variants:
            slot = slot_of(variant, cn2)
            plan = cell_plan(args, variant, cn2, slot)
            print(f"== cell {variant} cn2={cn2:g} (slot {slot})", flush=True)
            summary = run_arm(TurbShapingParams(**plan))
            results.append({"variant": variant, "cn2": cn2, **summary})
    return results


def report_matrix(results: list[dict], report_root: Path, args: argparse.Namespace) -> Path:
    from scripts.slm.generate_turbulence_shaping_report import build_report

    report_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for res in results:
        cell_dir = report_root / f"{res['variant']}_cn2{res['cn2']:g}"
        build_report(
            Path(res["run_dir"]), cell_dir, fps=6.0, phase_grid=256,
            max_frames=24, max_side=160,
        )
        rows.append({
            "variant": res["variant"],
            "cn2": res["cn2"],
            "sigma_total_rad": res["measured"]["sigma_total_rad"],
            "evals": int(res["measured"]["calls"]),
            # 链接相对**汇总表自身所在目录** (report_root) 解析, 不是其父目录
            "report": (cell_dir / "report.md").relative_to(report_root).as_posix(),
        })

    by_cn2: dict[float, list[dict]] = {}
    for row in rows:
        by_cn2.setdefault(row["cn2"], []).append(row)

    lines = [
        f"# 湍流整形对比择优矩阵 (epochs={args.epochs}, cam={args.cam_size}, "
        f"n_max={args.n_max}, side={args.target_side:g}, pop={args.pop_size})",
        "",
        "| 变体 | cn2 | 实测 σ_total [rad] | 光学求值 | 单元报告 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['cn2']:g} | {row['sigma_total_rad']:.4f} "
            f"| {row['evals']} | [{Path(row['report']).name}]({row['report']}) |"
        )
    lines += [
        "",
        "## 口径",
        "",
        "- 屏协议: spgd/square/heuristic/gs_refine = dynamic (逐光学求值独立屏); "
        "model_in_loop = static (Step A 拟合与逐轮验收假设准静态波前, 见 cell_plan 注);",
        "- 幅度**实测** (stats), 收敛快慢/终值对比见各单元 GIF 与 metrics.png; "
        "本矩阵为**对比择优**; MIL 的孪生湍流通道自 M5b 入列 (`_SimBench` "
        "disturbance 注入; GS-refine 通道见 matrix_20261121_m5a; #253 注记);",
        f"- 种子: dist_seed = {BASE_DIST_SEED} + 冻结网格序号 "
        "(``slot_of(variant, cn2)``; 子扫掠与基线逐单元配对)。",
        "",
    ]
    out = report_root / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    (report_root / "matrix.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def main() -> int:
    args = parse_args()
    if args.dry_run:
        dry_variants, dry_cn2 = axes(args)
        plan = [
            cell_plan(args, variant, cn2, slot_of(variant, cn2))
            for cn2 in dry_cn2 for variant in dry_variants
        ]
        print(json.dumps({"cells": plan}, ensure_ascii=False, indent=2))
        return 0
    results = run_matrix(args)
    import datetime as _dt

    report_root = args.report_root or (
        ROOT / "report" / "turbulence_shaping"
        / f"matrix_{_dt.datetime.now(_dt.UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    out = report_matrix(results, Path(report_root), args)
    print(f"matrix report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
