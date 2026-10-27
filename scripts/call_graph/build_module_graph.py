#!/usr/bin/env python3
"""Build the global (module-aggregated) call graph mermaid.

Reads ``call_graph_final.json`` (193 parsed function nodes / 290 edges) and
``recovered.json`` (bare-name -> dotted-name recovery), aggregates functions
into package-level nodes, and emits ``diagram/global-call-graph.mmd``.

Aggregation rules:
  * ``ao_shaping.drivers.ccd.common.capture_x``  -> package node ``ao_shaping.drivers.ccd``
  * class references (last segment uppercase, e.g. ``...adam.AdaMOD``)
    -> their module package (class segment dropped)
  * package-level re-export aliases (``ao_shaping.utils.gen_date_dir``)
    -> fallback to 2-level package when the 3-level name is not a real package
  * bare-name nodes (file-private helpers, no module info) -> one bucket node
  * unresolved targets -> one ``__unresolved__`` node

Idempotent: same input -> byte-identical output. No network, no randomness.

Usage:  python3 scripts/call_graph/build_module_graph.py
"""

from __future__ import annotations

import json
from loguru import logger
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/dev/call_graph_wiki" / "raw" / "assets"
OUT = ROOT / "docs/dev/call_graph_wiki" / "diagram" / "global-call-graph.mmd"

#: top-level subgraphs (by 2-level prefix); everything else is a standalone node
SUBGROUPS = [
    "ao_shaping.runners",
    "ao_shaping.optimizer",
    "ao_shaping.drivers",
    "ao_shaping.algorithm",
    "ao_shaping.utils",
    "ao_shaping.model",
    "ao_shaping.tools",
]

BARE = "__bare_helpers__"
UNRESOLVED = "__unresolved__"


def package_of(dotted: str, known: set[str]) -> str:
    """Map a dotted qualname (function or class) to its package node name."""
    parts = dotted.split(".")
    if parts[1] == "main":
        return "ao_shaping.main"
    if parts[-1][0].isupper():  # class reference -> its module
        parts = parts[:-1]
    pkg3 = ".".join(parts[:3])
    if pkg3 in known or len(parts) > 3:
        return pkg3
    if len(parts) == 3:  # re-export alias at package level (ao_shaping.utils.X)
        return ".".join(parts[:2])
    return pkg3


def subgroup_of(pkg: str) -> str | None:
    for s in SUBGROUPS:
        if pkg == s or pkg.startswith(s + "."):
            return s
    return None


def node_id(pkg: str) -> str:
    return "n_" + pkg.replace(".", "_")


def main() -> None:
    data = json.loads((ASSETS / "call_graph_final.json").read_text(encoding="utf-8"))
    rec = json.loads((ASSETS / "recovered.json").read_text(encoding="utf-8"))
    nodes: dict[str, dict | None] = data["nodes"]
    edges: list[dict] = data["edges"]

    groups: dict[str, set[str]] = defaultdict(set)
    n_bare = 0
    for qual, info in nodes.items():
        if info is None:
            continue  # ghost re-export node
        if "." not in qual:
            n_bare += 1
            continue  # file-private helper bucket (no module info)
        groups[package_of(qual, set())].add(qual)
    known_pkgs = set(groups)

    # re-resolve group keys now that `known` is available (re-export aliases)
    groups2: dict[str, set[str]] = defaultdict(set)
    for qual, info in nodes.items():
        if info is None or "." not in qual:
            continue
        groups2[package_of(qual, known_pkgs)].add(qual)
    groups = dict(groups2)

    agg: dict[tuple[str, str], int] = defaultdict(int)
    bare_edge_count = 0
    for e in edges:
        src, dst, calls = e["from"], e["to"], int(e.get("calls", 1))
        if "." not in src:  # call from a file-private helper
            bare_edge_count += calls
            continue
        sp = package_of(src, known_pkgs)
        if "." in dst:
            agg[(sp, package_of(dst, known_pkgs))] += calls
            continue
        r = rec.get(dst, dst)  # bare-name recovery
        resolved = None
        if isinstance(r, list):
            cands = [c for c in r if isinstance(c, str) and "." in c]
            if len(cands) == 1:
                resolved = cands[0]
        elif isinstance(r, str) and "." in r and r != dst:
            resolved = r
        if resolved:
            agg[(sp, package_of(resolved, known_pkgs))] += calls
        else:
            agg[(sp, UNRESOLVED)] += calls

    # edges whose target is a bare-name node -> helper bucket
    for (a, b), c in list(agg.items()):
        if b == BARE:
            continue
    # (bare targets never become their own package via package_of; they were
    #  already either resolved or UNRESOLVED above)

    # declare every package that appears in edges but has no own node
    edge_pkgs = {a for a, _ in agg} | {b for _, b in agg if b not in (BARE, UNRESOLVED)}
    extra_pkgs = sorted(edge_pkgs - set(groups))
    for p in extra_pkgs:
        groups[p] = set()  # zero-count external/class-reference package

    lines = ["flowchart TD"]

    # standalone nodes (main / zero-count extras first, sorted)
    standalone = [p for p in sorted(groups) if subgroup_of(p) is None]
    for p in standalone:
        cnt = len(groups[p])
        if cnt:
            lines.append(f'    {node_id(p)}["{p} ({cnt} 函数)"]')
        else:
            lines.append(f'    {node_id(p)}["{p} (外部引用包)"]')
    # subgraphs
    for s in SUBGROUPS:
        members = [p for p in sorted(groups) if subgroup_of(p) == s]
        if not members:
            continue
        lines.append(f"    subgraph sg_{s.split('.')[1]} ['{s}']")
        for p in members:
            cnt = len(groups[p])
            label = p.replace("ao_shaping.", "")
            if cnt == 0:
                label += " (外部引用包)"
            else:
                label += f" ({cnt} 函数)"
            lines.append(f"        {node_id(p)}[\"{label}\"]")
        lines.append("    end")
    lines.append(f'    {node_id(BARE)}["文件内私有 helper（裸名，{n_bare} 个）"]')
    lines.append(f'    {node_id(UNRESOLVED)}["未解析目标（动态分发/未恢复裸名）"]')

    # edges (weight-1 edges and self-loops hidden for readability)
    for (a, b), c in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0])):
        if c < 2 or a == b:
            continue
        lines.append(f"    {node_id(a)} -->|{c}| {node_id(b)}")

    kept = sum(1 for c in agg.values() if c >= 2)
    n_self = sum(c for (a, b), c in agg.items() if a == b)
    lines += [
        "",
        f"%% 193 个已解析函数聚合为 {sum(len(v) for v in groups.values() if v)} 个包级节点；",
        f"%% 另有 {n_bare} 个文件内私有 helper（裸名）归入单桶。",
        "%% 290 条边按包聚合，边权 = 调用次数（权重 1 的边与包内自环已隐去，全量见 raw/assets/call_graph_final.json）。",
        f"%% 包内自环共 {n_self} 次调用未画；类引用目标（如 AdaMOD / Recorder）已并入其所在包。",
        "%% 生成: scripts/call_graph/build_module_graph.py（幂等）。",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"wrote {OUT} ({OUT.stat().st_size} bytes), "
          f"{len([p for p in groups if groups[p]])} package nodes (+{len(extra_pkgs)} extra), "
          f"{kept} labeled edges (of {len(agg)} aggregated)")


if __name__ == "__main__":
    main()
