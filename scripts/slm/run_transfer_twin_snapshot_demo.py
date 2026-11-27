"""#254 T3/T4 数字孪生首验 —— 在孪生语料上端到端执行一次完整防泄漏流程.

不做设备动作: 语料由 ``transfer.simulation.generate_dataset`` (已知真值的
线性相机孪生) 生成, 随后原样调用采集侧将使用的同一条工具链 ——

1. ``snapshot.freeze``: 三层快照 v1-twin (哈希清单 + 防泄漏口径校验);
2. ``branch_fit.freeze``: 四分支预注册 spec (基线 / +learn_psf / +learn_geometry
   / +两者, 即 2026-11-24 两项适配的自然重构; **孪生预演用**, 正式四分支以
   采集侧 plan.md 为准) 统一预算并行训练, selection 层打分, 候选冻结;
3. ``branch_fit.score-final``: 候选冻结后对封存 final_test 层一次性打分 (审计)。

产物: ``report/shape_trials/transfer_twin_snapshot_demo_<UTC>/`` 收
spec / 快照清单 / candidates_frozen / final_test_result 的 JSON 与本报告;
训练工作目录 (inputs.npz、iterations.pkl 等大文件) 留在 ``data/`` 不入库。

边界 (报告同步声明): 孪生语料不是硬件结论 —— 本流程验证的是**机制与产物链**
(冻结可复核、预算唯一、封存一次性); 真实 v1 冻结与正式分支集合在持有
``data/debug/slm_transfer`` 语料和 plan.md 的采集侧机器上用同一命令执行。

用法::

    python scripts/slm/run_transfer_twin_snapshot_demo.py [--iterations 150]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.tools.slm.transfer import branch_fit as bf
from ao_shaping.tools.slm.transfer import snapshot as snap
from ao_shaping.tools.slm.transfer.simulation import generate_dataset

#: 四分支 = 11-24 两项适配 × 基线的自然重构 (孪生预演专用, 非正式预注册)。
BRANCHES = [
    {"name": "A_baseline", "overrides": {"adaptation": "alignment"}},
    {"name": "B_psf", "overrides": {"adaptation": "alignment", "learn_psf": True}},
    {"name": "C_geometry", "overrides": {"adaptation": "alignment",
                                         "learn_geometry": True}},
    {"name": "D_psf_geometry", "overrides": {"adaptation": "alignment",
                                             "learn_psf": True,
                                             "learn_geometry": True}},
]


def build_corpus(root: Path, *, count: int = 10) -> Path:
    """三份孪生采集 + 把 cap_c 的最后一条探针行改标 final_test (封存层)。"""
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    checkpoint = Path(generate_dataset(root / "cap_a", power=1.0, count=count,
                                       seed=102419))
    generate_dataset(root / "cap_b", power=1.1, count=count, seed=20241)
    generate_dataset(root / "cap_c", power=1.2, count=count, seed=30361)
    manifest = root / "cap_c" / "manifest.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    probes = [r for r in rows if r["split"] != "calibration"]
    probes[-1]["split"] = "final_test"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    return checkpoint


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path,
                    default=ROOT / "data/twin_snapshot_demo")
    ap.add_argument("--out", type=Path, default=None,
                    help="报告目录 (默认 report/shape_trials/"
                         "transfer_twin_snapshot_demo_<UTC日期>)")
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--span", type=int, default=120)
    ap.add_argument("--jobs", type=int, default=1,
                    help="分支训练并行度 (>1 走独立子进程, 与顺序路径同结论已由 CI 钉)")
    args = ap.parse_args(argv)

    stamp = datetime.now(UTC).strftime("%Y%m%d")
    report_dir = args.out or (ROOT / "report/shape_trials"
                              / f"transfer_twin_snapshot_demo_{stamp}")
    work = args.data_root
    corpus, run_root = work / "corpus", work / "run"
    report_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = build_corpus(corpus)
    manifest = snap.freeze(corpus, report_dir / "snapshot", "v1-twin")
    manifest_path = report_dir / "snapshot" / "v1-twin" / "snapshot_manifest.json"
    print(snap.render_plan(manifest), flush=True)

    spec = {
        "schema": bf.SPEC_SCHEMA, "snapshot_id": "v1-twin",
        "pre_registered_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "budget": {"checkpoint": str(checkpoint.resolve()),
                   "iterations": args.iterations, "span": args.span,
                   "device": "cpu"},
        "branches": BRANCHES,
    }
    spec_path = report_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    candidates = bf.freeze_candidates(spec_path, manifest_path, run_root, jobs=args.jobs)
    final = bf.score_final_once(run_root / "candidates_frozen.json", spec_path,
                                manifest_path)
    for name in ("candidates_frozen.json", "final_test_result.json",
                 "final_test_audit.jsonl"):
        src = run_root / name
        if src.is_file():
            shutil.copy2(src, report_dir / name)

    winner = candidates["winner"]
    lines = [
        "# 传输模型三层防泄漏快照与四分支预训练 —— 数字孪生首验 (#254 T3/T4)",
        "",
        f"- 生成时间 (UTC): {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- 快照: `v1-twin`, root_hash `{manifest['root_hash'][:16]}…`, "
        f"层计数 `{manifest['layer_counts']}`, 封存层 `{manifest['sealed_layers']}`",
        f"- 预算 (全分支共享): iterations={args.iterations}, span={args.span}, "
        "device=cpu; 分支只可覆盖模型旋钮白名单 (spec `budget`/`branches` 分离)",
        f"- 候选冻结: winner **{winner}**, spec_sha256 "
        f"`{candidates['spec_sha256'][:16]}…`; 冻结时刻 `final_test=None` (契约)",
        f"- 封存层一次性打分: correlation={final['final_test_quality']['correlation']:.4f}, "
        f"relative_l2={final['final_test_quality']['relative_l2']:.4f} "
        f"(审计行已入 final_test_audit.jsonl, 二次打分会被拒)",
        "",
        "## 分支 selection 层打分 (唯一晋级依据)",
        "",
        "| 分支 | 覆盖 | score | correlation | relative_l2 |",
        "|---|---|---|---|---|",
    ]
    for branch in candidates["branches"]:
        q = branch["quality"]
        lines.append(
            f"| {branch['name']} | `{branch['overrides']}` | {q['score']:.4f} | "
            f"{q['correlation']:.4f} | {q['relative_l2']:.4f} |")
    lines += [
        "",
        "## 口径与边界",
        "",
        "- 语料 = `transfer.simulation.generate_dataset` 已知真值孪生 (cap_a/b/c, "
        "cap_c 末行改标 final_test), **不是硬件数据**; 本实验验证的是 T3/T4 工具链"
        "端到端 (冻结可复核、预算结构性唯一、封存一次性打分、并行同结论), "
        "不构成任何模型质量或效率结论;",
        "- 四分支是 2026-11-24 两项适配 (PSF 通过 opt-in / 几何证伪) 的自然重构, "
        "仅用于孪生预演; 正式四分支集合与打分口径以采集侧 plan.md 为准;",
        "- 真实 v1 冻结与正式训练在持有 `data/debug/slm_transfer` 语料的采集侧执行: "
        "`python -m ao_shaping.tools.slm.transfer.snapshot freeze --root "
        "data/debug/slm_transfer --id v1 --out data/snapshots [--plan-only]` → "
        "`python -m ao_shaping.tools.slm.transfer.branch_fit freeze --spec ... "
        "--manifest ...`;",
        "- 训练工作目录 (inputs.npz / iterations.pkl / 各分支 fit 产物) 留在 "
        "`data/twin_snapshot_demo/run/`, 不入库; 入库 JSON 仅 spec / 快照清单 / "
        "candidates_frozen / final_test_result / audit。",
        "",
    ]
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {report_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
