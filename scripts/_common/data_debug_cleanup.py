"""data/debug/<top>/<run> 报告引用判定 + 归入 dump/（dry-run 默认）。

用法:
    python scripts/_common/data_debug_cleanup.py                # dry-run，列出将移动的 run
    python scripts/_common/data_debug_cleanup.py --apply        # 真的移动到 dump/data_debug/
    python scripts/_common/data_debug_cleanup.py --manifest out.json

判定口径：
  1) report/**/*.{md,json} 中解析出的 data/<path> 引用（复用 data_reference_inventory.scan）
  2) 保守兜底：run 名在任意报告正文中出现即视为引用
  3) 代码兜底：run 名在 src/ scripts/ tests/ 的源码中出现即视为运行依赖
  4) **在飞保护**：run 内最新文件比 ``--min-age-hours``（默认 12 h）还新 → 一律不动。
     其它会话可能正在同一台机器上跑硬件实验，其 run 目录随时在增长；搬走正在写的
     run 会毁掉在飞的工作，也会让报告引用失效。宁可漏搬，不可搬错。

**已知盲区（务必阅读）**：口径 1 只认"逐条路径引用"。若某报告把整个
``data/debug/<top>/`` 声明为实验数据根（例：``transfer_uncertainty_20261104.md`` 把
``data/debug/transfer_uncertainty_20261104/`` 整目录声明为数据根，正文引用其
bootstrap 委员会成员），则其下未被逐条引用的子目录会被误判为可移动。
``PROTECTED_RUNS`` 就是为此类情况设的人工豁免；新增豁免时必须写明依据报告。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_TOPS = {"slm_transfer"}
CODE_DIRS = ["src", "scripts", "tests"]

#: 默认在飞保护窗口（小时）。低于这个年龄的 run 视为"正在跑"。
DEFAULT_MIN_AGE_HOURS = 12.0

# 人工豁免：报告把整个目录声明为数据根，子目录未被逐条引用但仍属该实验的证据。
# 格式: "data/debug/<top>/<run>" -> 依据报告
PROTECTED_RUNS: dict[str, str] = {
    "data/debug/transfer_uncertainty_20261104/committee_0":
        "report/shape_trials/transfer_uncertainty/transfer_uncertainty_20261104.md 声明整目录为数据根",
    "data/debug/transfer_uncertainty_20261104/committee_1":
        "report/shape_trials/transfer_uncertainty/transfer_uncertainty_20261104.md 声明整目录为数据根",
    "data/debug/transfer_uncertainty_20261104/committee_2":
        "report/shape_trials/transfer_uncertainty/transfer_uncertainty_20261104.md 声明整目录为数据根",
    "data/debug/transfer_uncertainty_20261104/committee_3":
        "report/shape_trials/transfer_uncertainty/transfer_uncertainty_20261104.md 声明整目录为数据根",
}


def load_inventory():
    """按文件路径加载 data_reference_inventory（它是脚本, 不是包模块）。"""
    p = ROOT / "scripts" / "_common" / "data_reference_inventory.py"
    spec = importlib.util.spec_from_file_location("inventory", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def newest_mtime(p: Path) -> float:
    """run 目录内最新文件的 mtime；空目录返回 0。"""
    newest = 0.0
    for f in p.rglob("*"):
        if f.is_file():
            newest = max(newest, f.stat().st_mtime)
    return newest


def collect_refs() -> tuple[set[str], str]:
    inv = load_inventory()
    per_file = inv.scan(ROOT / "report", ROOT)
    refs: set[str] = set()
    corpus: list[str] = []
    for f, keys in per_file.items():
        refs.update(keys)
        try:
            corpus.append(f.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    return refs, "\n".join(corpus)


def collect_code_text() -> str:
    parts = []
    for d in CODE_DIRS:
        for f in (ROOT / d).rglob("*.py"):
            try:
                parts.append(f.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--manifest", default="")
    ap.add_argument(
        "--min-age-hours",
        type=float,
        default=DEFAULT_MIN_AGE_HOURS,
        help="run 内最新文件比这个小时数还新 → 在飞保护，一律不动（默认 12）",
    )
    args = ap.parse_args()

    refs, report_text = collect_refs()
    code_text = collect_code_text()

    debug = ROOT / "data" / "debug"
    dump_root = ROOT / "dump" / "data_debug"

    moves: list[tuple[str, Path]] = []
    keeps: list[tuple[str, str, Path]] = []
    cutoff = time.time() - args.min_age_hours * 3600
    for top in sorted(p for p in debug.iterdir() if p.is_dir()):
        if top.name in SKIP_TOPS:
            continue
        for run in sorted(p for p in top.iterdir() if p.is_dir()):
            key = f"data/debug/{top.name}/{run.name}"
            by_ref = any(r == key or r.startswith(key + "/") for r in refs)
            by_text = run.name in report_text
            by_code = run.name in code_text
            protected = key in PROTECTED_RUNS
            in_flight = newest_mtime(run) > cutoff
            if by_ref or by_text or by_code or protected or in_flight:
                why = []
                if by_ref:
                    why.append("ref")
                if by_text:
                    why.append("text")
                if by_code:
                    why.append("code")
                if protected:
                    why.append("manual")
                if in_flight:
                    why.append("in-flight")
                keeps.append((key, "+".join(why), run))
            else:
                moves.append((key, run))

    sizes = {key: dir_size(run) for key, run in moves}
    total = sum(sizes.values())
    for key, _ in moves:
        print(f"MOVE {key}  {sizes[key] / 1e9:.3f} GB")
    print(f"\n-- keep ({len(keeps)}) --")
    for key, why, _ in keeps:
        print(f"KEEP {key}  [{why}]")
    print(f"\nmove count={len(moves)} total={total / 1e9:.2f} GB")

    failed: list[str] = []
    skipped_exists: list[str] = []
    moved_count = 0
    if args.apply:
        for key, run in moves:
            dest = dump_root / run.parent.name / run.name
            if dest.exists():
                print(f"SKIP exists: {dest}")
                skipped_exists.append(key)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(run), str(dest))
                moved_count += 1
            except (PermissionError, OSError) as exc:
                print(f"FAILED {key}: {exc}")
                failed.append(f"{key}: {exc}")
        print(f"moved {moved_count} dirs -> {dump_root}")
    else:
        print("(dry-run; pass --apply to move)")

    if args.manifest:
        Path(args.manifest).write_text(
            json.dumps(
                {
                    "moves": [{"key": k, "bytes": sizes[k]} for k, _ in moves],
                    "keeps": [{"key": k, "why": w} for k, w, _ in keeps],
                    "moved_count": moved_count,
                    "failed": failed,
                    "skipped_exists": skipped_exists,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
