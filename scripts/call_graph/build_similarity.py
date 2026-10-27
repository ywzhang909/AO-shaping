"""Build the "similar functions" relation layer on top of the AST call graph.

Complements build_call_graph.py: the call graph answers "who calls whom",
this script answers "which functions are near-duplicates / sibling
implementations of the same idea" (similar-function relations).

Input : docs/dev/call_graph_wiki/raw/assets/call_graph_final.json
Output: docs/dev/call_graph_wiki/raw/assets/similar_functions.json

Idempotent: safe to re-run after any call-graph regeneration. No network,
no ML — a deterministic weighted score over four structural signals so the
result is reproducible and auditable (each pair carries its per-dimension
breakdown and a human-readable 理由).

Score model (tuned 2026-10-13, validated against the 193 resolved nodes):
  0.45 * jaccard(callee sets)  * min(1, shared_callees/2)
       -- structural similarity; the min() damper kills the Jaccard
          inflation when two nodes share only 1 trivial callee
  0.15 * name similarity       (1.0 if identical short name, 0.0 for the
       generic 'run' CLI entry, else token Jaccard on the short name)
  0.10 * same file (1.0) / same module (0.5) / else 0.0
  0.05 * jaccard(caller sets)
  0.05 * (shared_callees >= 5)
Pairs >= 0.30 are LINKED (shown on wiki pages); unions at >= 0.35 form
CLUSTERS (shown in relations/similar-functions.md).
"""
from __future__ import annotations

import json
from loguru import logger
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

P = Path(__file__).parent
WIKI = P.parents[1] / "docs" / "dev" / "call_graph_wiki"
CG = WIKI / "raw" / "assets" / "call_graph_final.json"
OUT = WIKI / "raw" / "assets" / "similar_functions.json"

LINK_THRESHOLD = 0.30
CLUSTER_THRESHOLD = 0.35
STOP_TOKENS = {"ao", "shaping"}


def short_name(qual: str) -> str:
    return qual.split(".")[-1]


def name_tokens(qual: str) -> set[str]:
    return {
        t
        for t in re.split(r"[^a-z0-9_]+", short_name(qual).lower())
        if t and t not in STOP_TOKENS
    }


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main() -> None:
    data = json.loads(CG.read_text())
    nodes = {q: v for q, v in data["nodes"].items() if v is not None}
    # drop ghost re-export nodes (bare names) — real symbols only
    qs = [q for q in nodes if "." in q and q not in ("ao_shaping.main.cli", "main.main")]

    callees: dict[str, set[str]] = defaultdict(set)
    callers: dict[str, set[str]] = defaultdict(set)
    for e in data["edges"]:
        callees[e["from"]].add(e["to"])
        callers[e["to"]].add(e["from"])

    pairs = []
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            a, b = qs[i], qs[j]
            shared = callees[a] & callees[b]
            sh = len(shared)
            jo = jaccard(callees[a], callees[b]) * min(1.0, sh / 2)
            ji = jaccard(callers[a], callers[b])
            sa, sb = short_name(a), short_name(b)
            if sa == sb:
                jn = 0.0 if sa == "run" else 1.0
                same_name = True
            else:
                jn = jaccard(name_tokens(a), name_tokens(b))
                same_name = False
            if nodes[a]["file"] == nodes[b]["file"]:
                fam, fam_label = 1.0, f"同文件 {nodes[a]['file'].split('/')[-1]}"
            elif nodes[a]["module"] == nodes[b]["module"]:
                fam, fam_label = 0.5, f"同模块 {nodes[a]['module']}"
            else:
                fam, fam_label = 0.0, "不同模块"
            s = 0.45 * jo + 0.15 * jn + 0.10 * fam + 0.05 * ji + 0.05 * (sh >= 5)
            if s < LINK_THRESHOLD:
                continue
            reasons = []
            if same_name:
                reasons.append(f"同名 `{sa}`")
            if sh >= 5:
                reasons.append(f"共享 {sh} 个被调函数（调用结构高度相似）")
            elif sh >= 2:
                reasons.append(f"共享 {sh} 个被调函数")
            if fam_label != "不同模块":
                reasons.append(fam_label)
            if ji >= 0.5:
                reasons.append("调用方高度重叠")
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "score": round(s, 3),
                    "jaccard_callees": round(jaccard(callees[a], callees[b]), 3),
                    "shared_callees": sh,
                    "name_sim": round(jn, 3),
                    "jaccard_callers": round(ji, 3),
                    "fam": fam_label,
                    "reason": "；".join(reasons) if reasons else "—",
                }
            )

    pairs.sort(key=lambda p: -p["score"])

    # union-find clusters at the higher threshold
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        if p["score"] >= CLUSTER_THRESHOLD:
            ra, rb = find(p["a"]), find(p["b"])
            if ra != rb:
                parent[ra] = rb
    clusters = defaultdict(list)
    for p in pairs:
        if p["score"] >= CLUSTER_THRESHOLD and find(p["a"]) == find(p["b"]):
            clusters[find(p["a"])].extend([p["a"], p["b"]])
    clusters = [sorted(set(v)) for v in clusters.values() if len(set(v)) > 1]
    clusters.sort(key=len, reverse=True)

    out = {
        "generated_on": date.today().isoformat(),
        "method": {
            "source": "call_graph_final.json (AST static call graph)",
            "script": "scripts/call_graph/build_similarity.py",
            "weights": {
                "jaccard_callees": 0.45,
                "name_similarity": 0.15,
                "same_file_or_module": 0.10,
                "jaccard_callers": 0.05,
                "many_shared_callees_bonus": 0.05,
            },
            "link_threshold": LINK_THRESHOLD,
            "cluster_threshold": CLUSTER_THRESHOLD,
            "notes": [
                "callees Jaccard is damped by min(1, shared/2) to avoid "
                "inflation from a single trivial shared callee",
                "generic CLI entry name 'run' contributes 0 to name similarity",
            ],
        },
        "pairs": pairs,
        "clusters": clusters,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    logger.info(f"wrote {OUT} — {len(pairs)} linked pairs, {len(clusters)} clusters")
    for p in pairs:
        logger.info(f"  {p['score']:.3f}  {p['a']}  ~  {p['b']}   ({p['reason']})")


if __name__ == "__main__":
    main()
