"""Second pass: recover unresolved names by repo-wide name match; dump function list."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
spec = importlib.util.spec_from_file_location("bcg", str(Path(__file__).parent / "build_call_graph.py"))
# don't run build() (it writes); import module-level fns only
import ast as _ast

# Re-import pieces by exec of the module without __main__
src = (Path(__file__).parent / "build_call_graph.py").read_text()
src = src.split('if __name__ == "__main__":')[0]
ns = {"__file__": str(Path(__file__).parent / "build_call_graph.py")}
exec(compile(src, "bcg", "exec"), ns)

mods = ns["load_modules"]()
print(f"modules: {len(mods)}", file=sys.stderr)

# index all repo functions by simple name
by_name = defaultdict(list)
for d, m in mods.items():
    for fn in m.funcs.values():
        by_name[fn.name].append(fn)
    for fn in m.methods.values():
        by_name[fn.name.split(".")[-1]].append(fn)

P = Path(__file__).resolve().parent  # scripts/call_graph/
ASSETS = P.parents[1] / "docs" / "dev" / "call_graph_wiki" / "raw" / "assets"
data = json.loads((ASSETS / "call_graph.json").read_text())
nodes = data["nodes"]
recovered = {}
for q, info in nodes.items():
    if info is None and "." not in q:
        cands = by_name.get(q, [])
        if len(cands) == 1:
            recovered[q] = cands[0].qualname
        elif len(cands) > 1:
            recovered[q] = [c.qualname for c in cands]

print(f"unresolved simple names: {sum(1 for q,i in nodes.items() if i is None and '.' not in q)}", file=sys.stderr)
print(f"recovered uniquely: {sum(1 for v in recovered.values() if isinstance(v,str))}", file=sys.stderr)
amb = {k:v for k,v in recovered.items() if isinstance(v,list)}
print(f"ambiguous ({len(amb)}):", json.dumps(amb, indent=1, ensure_ascii=False), file=sys.stderr)

# apply unique recovery
for q, tgt in recovered.items():
    if isinstance(tgt, str) and tgt not in nodes:
        m = mods[tgt.split(".")[0] if tgt.count(".")==1 else ".".join(tgt.split(".")[:-1])] if False else None
        # find func object
        for d, mm in mods.items():
            pass
print("recovery map saved")
(ASSETS / "recovered.json").write_text(json.dumps(recovered, ensure_ascii=False, indent=1))

# Dump resolved function inventory grouped by module for the wiki
inv = defaultdict(list)
for q, info in nodes.items():
    if info is not None:
        inv[info["module"]].append(info)
for k in sorted(inv):
    print(f"\n== {k} ({len(inv[k])}) ==")
    for fn in sorted(inv[k], key=lambda x: x["lineno"]):
        loc = fn["qualname"].split(".")[-1]
        print(f"  {loc}  L{fn['lineno']}-{fn['end']}  args={len(fn['args'])}  doc={len(fn['doc'])}B  deco={fn['decorators']}")
