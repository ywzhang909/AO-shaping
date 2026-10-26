"""drivers/ 家族文档树与磁盘的双向清单守卫 (item 85).

解析每份 AGENTS.md 的「结构/目录结构」围栏树, 与磁盘逐名对账:
- 文档列出但磁盘不存在的 .py/目录 => 幻影 token (历史例: ccd/ffmpeg.py、dm/Drv_UDPST/)。
- 磁盘存在但文档漏列的 .py/目录 => 漏登 (item 85 前: drivers/_lazy.py 等)。

规则: 完整性只统计 .py 文件与目录 (排除 __pycache__); 其它文件 (.md/.txt) 仅在
文档列出那一刻校验存在, 不强制登记。新驱动文件落盘后必须同步更新对应
AGENTS.md 的树块, 否则本测试变红。
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DRIVERS = REPO / "src" / "ao_shaping" / "drivers"

# (doc 相对 drivers/, 树块根行, scope 目录 相对 drivers/)
DOC_TREES: list[tuple[str, str, str]] = [
    ("AGENTS.md", "drivers/", "."),
    ("ccd/AGENTS.md", "ccd/", "ccd"),
    ("dm/AGENTS.md", "dm/", "dm"),
    ("sim/AGENTS.md", "sim/", "sim"),
    ("slm/AGENTS.md", "slm/", "slm"),
    ("tm/AGENTS.md", "tm/", "tm"),
    ("wfs/AGENTS.md", "wfs/", "wfs"),
]


def _marker_col(line: str) -> int | None:
    """First box-drawing marker column, or None."""
    for marker in ("├── ", "└── "):
        col = line.find(marker)
        if col != -1:
            return col
    return None


def _parse_tree(doc_path: Path, root_line: str) -> list[tuple[str, bool]] | None:
    """Extract [(relpath-from-scope, is_dir)] from the fenced tree; None if missing."""
    lines = doc_path.read_text(encoding="utf-8").split("\n")
    start = None
    for k, line in enumerate(lines):
        if line.strip() == "```" and k + 1 < len(lines) and lines[k + 1].rstrip() == root_line:
            start = k + 1
            break
    if start is None:
        return None
    entries: list[tuple[str, bool]] = []
    ancestors: list[tuple[int, str]] = []
    for line in lines[start + 1 :]:
        if line.strip() == "```":
            break
        col = _marker_col(line)
        if col is None:
            continue
        label = line[col + 4 :].split("#", 1)[0].strip()
        if not label:
            continue
        is_dir = label.endswith("/")
        name = label.rstrip("/")
        depth = col // 4
        while ancestors and ancestors[-1][0] >= depth:
            ancestors.pop()
        # ancestor chain: names of all shallower surviving entries
        parent_parts = [n for _, n in ancestors]
        entries.append(("/".join(parent_parts + [name]), is_dir))
        ancestors.append((depth, name))
    return entries


def _relevant_children(dir_path: Path) -> set[str]:
    """Disk children counted by the completeness rule: .py files + dirs (no __pycache__)."""
    out: set[str] = set()
    if not dir_path.is_dir():
        return out
    for child in dir_path.iterdir():
        if child.name == "__pycache__":
            continue
        if child.is_dir() or child.suffix == ".py":
            out.add(child.name + ("/" if child.is_dir() else ""))
    return out


def test_driver_doc_trees_match_disk() -> None:
    diffs: list[str] = []
    for doc_rel, root_line, scope_rel in DOC_TREES:
        doc_path = DRIVERS / doc_rel
        scope_dir = DRIVERS / scope_rel
        entries = _parse_tree(doc_path, root_line)
        if entries is None:
            diffs.append(f"{doc_rel}: 未见根行为 {root_line!r} 的围栏树块")
            continue
        top: set[str] = set()
        listed_dirs: set[str] = set()
        for rel, is_dir in entries:
            name = rel + ("/" if is_dir else "")
            if name == "__pycache__/":
                diffs.append(f"{doc_rel}: 噪声 __pycache__/ 不应登记, 请从树块删除")
                continue
            if "/" not in rel:
                top.add(name)
            if is_dir:
                listed_dirs.add(rel)
            target = scope_dir / rel
            if not target.exists():
                diffs.append(f"{doc_rel}: 幻影 token {name!r} (磁盘不存在)")
            elif is_dir and not target.is_dir():
                diffs.append(f"{doc_rel}: {name!r} 在磁盘上不是目录")
        disk_top = _relevant_children(scope_dir)
        missing_top = sorted(disk_top - top)
        if missing_top:
            diffs.append(f"{doc_rel}: 漏登顶层 {missing_top}")
        for rel in sorted(listed_dirs):
            doc_kids = {
                e.rsplit("/", 1)[1] + ("/" if d else "")
                for e, d in entries
                if "/" in e and e.rsplit("/", 1)[0] == rel
            }
            if not doc_kids:
                # 该父目录的子内容由它自己的 scope 文档负责 (如根树里的 ccd/、dm/…),
                # 只有当文档确实列出了至少一个子项时才做完整性对账
                continue
            disk_kids = _relevant_children(scope_dir / rel)
            missing = sorted(disk_kids - doc_kids)
            if missing:
                diffs.append(f"{doc_rel}: 目录 {rel}/ 漏登 {missing}")
    assert not diffs, "drivers 家族文档树与磁盘漂移:\n  " + "\n  ".join(diffs)
