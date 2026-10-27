"""Merge recovery into final call_graph.json with full node metadata."""
from __future__ import annotations
import json
from loguru import logger
import sys
import importlib.util
from pathlib import Path

P = Path(__file__).resolve().parent          # scripts/call_graph/
REPO = P.parents[1]                            # repo root
WIKI = REPO / "docs" / "dev" / "call_graph_wiki"
ASSETS = WIKI / "raw" / "assets"
sys.path.insert(0, str(P))
src = (P / "build_call_graph.py").read_text().split('if __name__ == "__main__":')[0]
ns = {"__file__": str(P / "build_call_graph.py")}
exec(compile(src, "bcg", "exec"), ns)
mods = ns["load_modules"]()

base = ASSETS
data = json.loads((base / "call_graph.json").read_text())
rec = json.loads((base / "recovered.json").read_text())

# index every repo function by simple name AND qualname
by_simple = {}
by_qualified = {}
for d, m in mods.items():
    for fn in m.funcs.values():
        by_simple.setdefault(fn.name, []).append(fn)
        by_qualified[fn.qualname] = fn
    for fn in m.methods.values():
        by_qualified[fn.qualname] = fn
        by_simple.setdefault(fn.name.split(".")[-1], []).append(fn)

def find_fn(qual: str):
    """Resolve a qualname (module[.sub].name) to a Func."""
    if qual in by_qualified:
        return by_qualified[qual]
    parts = qual.split(".")
    for i in range(len(parts) - 1, 0, -1):
        cand = ".".join(parts[:i])
        if cand in mods:
            rest = ".".join(parts[i:])
            mm = mods[cand]
            if rest in mm.funcs:
                return mm.funcs[rest]
            if rest in mm.methods:
                return mm.methods[rest]
    return None

def resolve(t):
    if t in data["nodes"] and data["nodes"][t] is not None:
        return t
    if "." not in t and t in rec:
        v = rec[t]
        if isinstance(v, str):
            return v
    return t

def fn_to_info(fn):
    return {
        "qualname": fn.qualname,
        "module": fn.module,
        "file": fn.file,
        "lineno": fn.lineno,
        "end": fn.end,
        "args": fn.args,
        "returns": fn.returns,
        "doc": fn.doc,
        "decorators": fn.decorators,
    }

# Final nodes: entry targets + everything reachable
final_nodes = {}
for q in list(data["nodes"]) + [e["from"] for e in data["edges"]] + [resolve(e["to"]) for e in data["edges"]]:
    if q in final_nodes:
        continue
    fn = find_fn(q)
    if fn is not None:
        final_nodes[q] = fn_to_info(fn)
    else:
        # external (non-repo): record the name only
        final_nodes[q] = None

edges = {}
for e in data["edges"]:
    a, b = e["from"], resolve(e["to"])
    edges[(a, b)] = edges.get((a, b), 0) + e["calls"]

out = {
    "generated_from": "main.py -> ao_shaping.main.cli (click group, 20 subcommands)",
    "nodes": final_nodes,
    "edges": [{"from": a, "to": b, "calls": n} for (a, b), n in sorted(edges.items())],
    "cmd_entries": data["cmd_entries"],
}
(base / "call_graph_final.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

n_res = sum(1 for v in final_nodes.values() if v)
logger.info(f"final nodes: {len(final_nodes)} (resolved {n_res}, external {len(final_nodes)-n_res}), edges {len(edges)}")
