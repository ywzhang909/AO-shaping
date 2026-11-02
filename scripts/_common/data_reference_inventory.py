"""扫描 report/ 对 data/ 的引用，生成引用对照清单（可复用工具，2026-11-01）。

用法:
    python scripts/_common/data_reference_inventory.py [--report-dir report] [--data-dir data]
    python scripts/_common/data_reference_inventory.py --tops     # 只列零引用的 data 顶层条目

判定口径: 静态扫描 ``report/**/*.{md,json}`` 中形如 ``data/...`` 的字符串（含 ``../data/...``），
存在性按本机文件系统判定。仅代表"报告引用"维度，不代表运行时依赖
（``*_configs`` / ``dm_adj.txt`` 等仍可能被主程序读取，移动前需另行确认）。
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

REF_RE = re.compile(r"(?:\.\./|\.\.\\|['\"(`\[]|\s|^)((?:\.\.[/\\])*data[/\\][^\s'\"`)\]}>,，。；：*]+)")


def scan(report_dir: Path, repo_root: Path) -> dict[Path, dict[str, bool]]:
    per_file: dict[Path, dict[str, bool]] = defaultdict(dict)
    for f in sorted(report_dir.rglob("*")):
        if f.suffix.lower() not in {".md", ".json"} or not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in REF_RE.finditer(text):
            raw = m.group(1).rstrip(".:")
            if raw.startswith("../"):
                resolved = (f.parent / raw.replace("/", "\\")).resolve()
            else:
                resolved = (repo_root / raw.replace("\\", "/")).resolve()
            try:
                key = str(resolved.relative_to(repo_root)).replace("\\", "/")
            except ValueError:
                continue
            per_file[f][key] = resolved.exists()
    return per_file


def write_inventory(per_file: dict[Path, dict[str, bool]], out: Path, repo_root: Path) -> None:
    all_refs: dict[str, int] = defaultdict(int)
    for refs in per_file.values():
        for r in refs:
            all_refs[r] += 1
    exists = {r: any(v.get(r) for v in per_file.values()) for r in all_refs}

    lines = [
        "# report → data 引用对照清单",
        "",
        f"> 生成日期: {date.today().isoformat()}；工具: `scripts/_common/data_reference_inventory.py`。",
        "> 仅静态扫描 `report/**/*.{md,json}` 中的 `data/...` 字符串，存在性按本机文件系统判定；",
        "> 不代表运行时依赖（configs/标定文件等仍可能被主程序读取）。",
        "",
        "## 汇总",
        "",
        f"- 含 `data/` 引用的报告文件: {len(per_file)} / 扫描总数见上次运行；唯一被引路径 {len(all_refs)} 条，"
        f"本机不存在 {sum(1 for v in exists.values() if not v)} 条。",
        "",
        "## 逐报告引用明细",
        "",
    ]
    for f in sorted(per_file):
        lines.append(f"### {f.relative_to(repo_root).as_posix()}")
        lines.append("")
        for r in sorted(per_file[f]):
            lines.append(f"- {'✅ 存在' if per_file[f][r] else '❌ 缺失'} `{r}`")
        lines.append("")

    data_dir = repo_root / "data"
    referenced_tops = {r.split("/")[1] for r in all_refs if r.startswith("data/") and "/" in r}
    actual_tops = sorted(p.name for p in data_dir.iterdir()) if data_dir.exists() else []
    zero_ref = [t for t in actual_tops if t not in referenced_tops]
    lines += [
        "## 零报告引用的 data 顶层条目（潜在可清理候选）",
        "",
    ]
    lines += [f"- `{t}`" for t in zero_ref] or ["- （无）"]
    lines += [
        "",
        "> 仅“报告引用”维度；清理/移动前需 grep `src/`、`scripts/` 确认运行时依赖。",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"written: {out} ({len(all_refs)} unique refs, {len(per_file)} files, zero-ref tops: {len(zero_ref)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-dir", default="report")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=None, help="清单输出路径，默认 <report-dir>/data_reference_inventory.md")
    ap.add_argument("--tops", action="store_true", help="只打印零引用的 data 顶层条目")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    report_dir = (repo_root / args.report_dir).resolve()
    data_dir = (repo_root / args.data_dir).resolve()
    per_file = scan(report_dir, repo_root)

    if args.tops:
        all_refs = {r for refs in per_file.values() for r in refs}
        tops = {r.split("/")[1] for r in all_refs if r.startswith("data/") and r.count("/") >= 1}
        for p in sorted(data_dir.iterdir()):
            if p.name not in tops:
                print(p.name)
        return

    out = Path(args.out) if args.out else report_dir / "data_reference_inventory.md"
    write_inventory(per_file, out, repo_root)


if __name__ == "__main__":
    main()
