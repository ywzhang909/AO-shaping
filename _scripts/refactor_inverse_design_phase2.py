"""Phase 2: split _tail_lines in generate_inverse_design_report.py into
per-section helpers (## 5 .. ## 14). Same mechanical AST splicing as phase 1;
output must stay byte-identical (golden-diffed)."""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "scripts/ml/generate_inverse_design_report.py"
text = SRC.read_text(encoding="utf-8")
tree = ast.parse(text)

func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_tail_lines")
ret = next(n for n in func.body if isinstance(n, ast.Return))
lst = ret.value
assert isinstance(lst, ast.List)

# Cut points: string elements that are a "## N." section header.
cuts = []
for i, el in enumerate(lst.elts):
    if isinstance(el, ast.Constant) and isinstance(el.value, str) and el.value.startswith("## "):
        cuts.append(i)
titles = [lst.elts[i].value for i in cuts]
assert [t.split(".")[0] for t in titles] == [f"## {n}" for n in range(5, 15)], titles

# Segment boundaries (elements inclusive of the header line).
bounds = cuts + [len(lst.elts)]
segments = []
for a, b in zip(bounds, bounds[1:]):
    seg = lst.elts[a:b]
    num = int(titles[a - cuts[0] if False else list(cuts).index(a)].split(".")[0][3:])
    # starred row-var names, in first-appearance order
    names = []
    for el in seg:
        for sub in ast.walk(el):
            if isinstance(sub, ast.Starred):
                nm = sub.value.id
                if nm not in names:
                    names.append(nm)
    segments.append((num, names, seg))

helpers = []
calls = []
for num, names, seg in segments:
    hname = f"_tail_section_{num}"
    param = ", ".join(names)
    body = "    return [\n"
    for el in seg:
        body += "        " + ast.get_source_segment(text, el).strip() + ",\n"
    body += "    ]"
    helpers.append(f"def {hname}({param}):\n    \"\"\"§{num} 小节。\"\"\"\n" + body)
    calls.append(hname + "(" + ", ".join(names) + ")")

# New _tail_lines body: return the concatenation of the section lists.
new_func = (
    "def _tail_lines(*, search_rows, combined_rows, inverse_rows, taxonomy_rows,\n"
    "                phase_smooth_rows, crosstalk_rows, inverse_transfer_rows):\n"
    '    """§5–§14 静态正文 + 术语表（行变量经 kwarg 传入各小节）。"""\n'
    "    return ("
    + " +\n       ".join(calls)
    + "\n    )"
)

old_start = text.index("def _tail_lines(")
nxt = text.index("\ndef ", old_start + 10)
new_text = text[:old_start] + new_func + "\n\n\n" + "\n\n\n".join(helpers) + "\n" + text[nxt:]
SRC.write_text(new_text, encoding="utf-8")
print("phase2 rewritten, bytes:", len(new_text))
