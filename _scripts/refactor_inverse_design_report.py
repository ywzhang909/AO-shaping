"""Mechanically split build_report in generate_inverse_design_report.py into
section helpers. Pure text splicing from ast.get_source_segment -- no manual
transcription, so the generated markdown stays byte-identical (golden-diffed)."""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "scripts/ml/generate_inverse_design_report.py"
text = SRC.read_text(encoding="utf-8")
tree = ast.parse(text)

func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_report")

# The last two statements are the tail (`body = ...`, `return ...`).
tail = func.body[-2:]
tail_srcs = [ast.get_source_segment(text, s) for s in tail]
body_assign = tail_srcs[0].split("\n")[0]      # `    body = "\n".join(lines)`
ret_stmt = tail_srcs[1].split("\n")[0]          # `    return insert_header(...)`

setup_srcs = []
header_stmt = None
appends = []

for stmt in func.body[:-2]:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        tgt = stmt.targets[0]
        if isinstance(tgt, ast.Name) and tgt.id == "lines" and isinstance(stmt.value, ast.List):
            header_stmt = ast.get_source_segment(text, stmt)
            continue
        setup_srcs.append(ast.get_source_segment(text, stmt))
        continue
    if isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Name) and stmt.test.id == "figures_ok" and len(stmt.body) == 1:
        inner = stmt.body[0]
        if isinstance(inner, ast.AugAssign) and isinstance(inner.value, ast.List):
            appends.append(ast.get_source_segment(text, inner.value))
            continue
    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "lines" and isinstance(stmt.value, ast.List):
        appends.append(ast.get_source_segment(text, stmt.value))
        continue
    raise AssertionError(f"unrecognized statement: {ast.dump(stmt)[:120]}")

assert header_stmt is not None
assert len(appends) == 6, len(appends)
B, C, D, E, F, G = appends


def wrap_return(lit: str) -> str:
    """`[...]` literal segment -> `return [...]` statement at 4-space body indent.

    The literal's internal lines keep their original (8-space) indent, which is
    already correct relative to a 4-space body; only the opener line needs the
    `return` prefix at 4 spaces.
    """
    out, started = [], False
    for ln in lit.split("\n"):
        if not started:
            if ln.strip() == "":
                continue
            out.append("    return " + ln.strip())
            started = True
        else:
            out.append(ln)
    return "\n".join(out)


def helper(name, doc, params, lit):
    return f"def {name}({params}):\n    \"\"\"{doc}\"\"\"\n" + wrap_return(lit)


helpers_src = "\n\n\n".join(
    [
        helper("_forward_figures", "§2 正向图表块（figures_ok 时插入）。", "fwd: dict", B),
        helper("_inverse_figures", "§3 逆向图表块（figures_ok 时插入）。", "invst: dict", D),
        helper("_phase_figures", "§4 瞳孔相位图表块（figures_ok 时插入）。", "ph: dict", F),
    ]
)

TAIL_MAP = {
    "SEARCH_ROWS": "search_rows",
    "COMBINED_ROWS": "combined_rows",
    "INVERSE_ROWS": "inverse_rows",
    "TAXONOMY_ROWS": "taxonomy_rows",
    "PHASE_SMOOTH_ROWS": "phase_smooth_rows",
    "CROSSTALK_ROWS": "crosstalk_rows",
    "INVERSE_TRANSFER_ROWS": "inverse_transfer_rows",
}
g = wrap_return(G)
for orig, new in TAIL_MAP.items():
    g = g.replace("*" + orig, "*" + new)
tail_helper = (
    "def _tail_lines(*, search_rows, combined_rows, inverse_rows, taxonomy_rows,\n"
    "                phase_smooth_rows, crosstalk_rows, inverse_transfer_rows):\n"
    '    """§5–§14 静态正文 + 术语表（无 stats 依赖，行变量经 kwarg 解包）。"""\n'
    + g
)

new_body = [
    "    " + header_stmt.strip(),
    "    if figures_ok:",
    "        lines += _forward_figures(fwd)",
    "    lines += " + C.strip(),
    "    if figures_ok:",
    "        lines += _inverse_figures(invst)",
    "    lines += " + E.strip(),
    "    if figures_ok:",
    "        lines += _phase_figures(ph)",
    "    lines += _tail_lines(search_rows=SEARCH_ROWS, combined_rows=COMBINED_ROWS,\n"
    "        inverse_rows=INVERSE_ROWS, taxonomy_rows=TAXONOMY_ROWS,\n"
    "        phase_smooth_rows=PHASE_SMOOTH_ROWS, crosstalk_rows=CROSSTALK_ROWS,\n"
    "        inverse_transfer_rows=INVERSE_TRANSFER_ROWS)",
    "    " + body_assign.strip(),
    "    " + ret_stmt.strip(),
]

new_func = "def build_report(stats: dict, figures_ok: bool) -> str:\n"
if setup_srcs:
    new_func += "    " + "\n    ".join(s.strip() for s in setup_srcs) + "\n"
new_func += "\n".join(new_body) + "\n"

old_start = text.index("def build_report(")
nxt = text.index("\ndef ", old_start + 10)
new_text = (
    text[:old_start]
    + new_func
    + "\n\n\n"
    + helpers_src
    + "\n\n\n"
    + tail_helper
    + "\n"
    + text[nxt:]
)
SRC.write_text(new_text, encoding="utf-8")
print("rewritten, bytes:", len(new_text))
