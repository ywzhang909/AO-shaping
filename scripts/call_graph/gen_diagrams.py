"""Generate mermaid call-graph diagrams from call_graph_final.json."""
from __future__ import annotations
import json
import re
from pathlib import Path
from collections import defaultdict

base = Path(__file__).resolve().parents[2] / "docs" / "dev" / "call_graph_wiki" / "raw" / "assets"
data = json.loads((base / "call_graph_final.json").read_text())
nodes, edges = data["nodes"], data["edges"]
cmd = data["cmd_entries"]  # symbol -> module.func

def short(qual):
    return qual.split(".")[-1]

def layer_of(qual):
    if qual in ("main.main", "ao_shaping.main.cli"):
        return 0  # entry
    if "runners" in qual or qual.startswith("ao_shaping.tools"):
        return 1  # runner / tool
    if "optimizer" in qual:
        return 2  # optimizer
    if "algorithm" in qual:
        return 3
    if "drivers" in qual:
        return 4  # hardware drivers
    if "utils" in qual or "display" in qual:
        return 5  # utils/display
    if "gui" in qual:
        return 6
    return 7

# ---- Diagram 1: top-level (main -> commands -> runners -> core optimizer) ----
core = {}
for sym, entry in cmd.items():
    core[sym] = entry
tools = {
    "slm-diagnose": "ao_shaping.tools.slm.panel.diagnose.main",
    "slm-lut": "ao_shaping.tools.slm.calibration.lut_runner.run",
}

lines = ["flowchart TD"]
lines.append("    subgraph L0[入口 entry]")
lines.append("        MAIN[\"main.py<br/>cli() __main__\"]")
lines.append("        CLI[\"ao_shaping.main.cli<br/>(click group, 20 命令)\"]")
lines.append("        MAIN --> CLI")
lines.append("    end")

lines.append("    subgraph L1[CLI 命令 → runner 入口]")
for name, entry in list(core.items()) + list(tools.items()):
    safe = name.replace("-", "_").replace(".", "_")
    fn = entry.split(".")[-1]
    lines.append(f'        C{safe}["{name}"]')
    lines.append(f'        CLI --> C{safe}["{name}"]')
    if nodes.get(entry):
        lines.append(f'        C{safe} --> R{safe}["{short(entry)}<br/>{entry.split(".")[0]}...{entry.split(".")[1] if "." in entry else ""}"]')
    else:
        lines.append(f'        C{safe} --> R{safe}["{entry} (ext)"]')
lines.append("    end")

# depth-1: runner -> optimizer (only optimizer layer, to keep diagram readable)
depth1 = defaultdict(set)
for e in edges:
    a, b = e["from"], e["to"]
    if nodes.get(a) and layer_of(a) == 1 and layer_of(b) == 2:
        depth1[a].add(b)

lines.append("    subgraph L2[核心优化器 optimizer]")
for a, bs in depth1.items():
    safe = a.replace(".", "_").replace("-", "_")
    for b in sorted(bs):
        if b in nodes and nodes[b]:
            lines.append(f'        O{b.replace(".", "_")}["{b.split(".")[-1]}<br/>{b.rsplit(".",1)[0]}"]')
            lines.append(f'        R{a.replace(".", "_").replace("-", "_")} --> O{b.replace(".", "_")}')
        else:
            lines.append(f'        O{b.replace(".", "_")}["{b} (ext)"]')
            lines.append(f'        R{a.replace(".", "_").replace("-", "_")} --> O{b.replace(".", "_")}')
lines.append("    end")

(base / "diagram_commands.mmd").write_text("\n".join(lines))
print("wrote diagram_commands.mmd")

# ---- Diagram 2: full per-layer flow (entry->runner->optimizer->driver/util) ----
lines2 = ["flowchart TD"]
seen_nodes = set()
for e in edges:
    a, b = e["from"], e["to"]
    if b not in nodes or not nodes[b]:
        continue
    if a == "main.main":
        a = "ao_shaping.main.cli"
    la, lb = layer_of(a), layer_of(b)
    if la not in (0, 1, 2, 3, 4, 5, 6) or lb not in (0, 1, 2, 3, 4, 5, 6):
        continue
    # only keep edges between adjacent layers or 1->2 etc (skip util->util noise? keep all)
    ida = "n" + a.replace(".", "_")
    idb = "n" + b.replace(".", "_")
    if a not in seen_nodes:
        seen_nodes.add(a)
        lines2.append(f'    {ida}["{a}"]')
    if b not in seen_nodes:
        seen_nodes.add(b)
        lines2.append(f'    {idb}["{b}"]')
    lines2.append(f"    {ida} --> {idb}")

(base / "diagram_full.mmd").write_text("\n".join(lines2))
print("wrote diagram_full.mmd")

# ---- Per-command call chain (text, for wiki) ----
out = []
for name in list(cmd) + list(tools):
    entry = core.get(name) or tools.get(name)
    out.append(f"## {name}  →  {entry}")
    stack = [entry]
    emitted = 0
    while stack and emitted < 60:
        cur = stack.pop()
        if cur in out[0]:  # visited
            pass
        # find calls from cur
        kids = [e["to"] for e in edges if e["from"] == cur]
        for k in kids:
            if k in (cur,):
                continue
            out.append(f"    - {k}")
        break  # only one level of direct calls shown per command here
    out.append("")

# simpler: per command show direct callees + their direct callees (2 levels)
out2 = []
for name in list(cmd) + list(tools):
    entry = core.get(name) or tools.get(name)
    out2.append(f"### {name}  ({entry})")
    direct = sorted({e["to"] for e in edges if e["from"] == entry})
    for k in direct:
        k2 = sorted({e["to"] for e in edges if e["from"] == k})
        out2.append(f"- {k}" + ("" if not k2 else ""))
        for kk in k2:
            out2.append(f"  - {kk}")
    out2.append("")
(base / "per_command_chain.txt").write_text("\n".join(out2))
print("wrote per_command_chain.txt")
