"""Phase-B mechanical split for generate_inverse_shaping_model_report.build_report.

Zero text mutation beyond a uniform 4-space re-indent: each top-level
``add("## N. ...")`` heading starts a section partition; every partition is
lifted verbatim into a ``_section_N(add, cfg, rows, meta, total,
stage_totals, floors)`` helper. ``build_report`` becomes setup + one call per
section + return. Rendered markdown stays byte-identical (golden-diffed on
both floors arms).
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "scripts/ml/generate_inverse_shaping_model_report.py"
INDENT = " " * 4

SIG = "(add, cfg, rows, meta, total, stage_totals, floors)"


def seg(text: str, node: ast.AST) -> str:
    s = ast.get_source_segment(text, node)
    assert s is not None, "get_source_segment returned None"
    return s


def is_assign(s: ast.stmt) -> bool:
    return isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign))


def heading_number(s: ast.stmt) -> str | None:
    """If s is an ``add("## N. ...")`` call, return the section number 'N'."""
    if not isinstance(s, ast.Expr):
        return None
    call = s.value
    if not isinstance(call, ast.Call) or not (
        isinstance(call.func, ast.Name) and call.func.id == "add"
    ):
        return None
    if not call.args:
        return None
    a0 = call.args[0]
    if not (isinstance(a0, ast.Constant) and isinstance(a0.value, str)):
        return None
    v = a0.value
    if not v.startswith("## "):
        return None
    rest = v[len("## "):]
    # number is the run of digits right after "## "
    i = 0
    while i < len(rest) and rest[i].isdigit():
        i += 1
    if i == 0:
        return None
    return rest[:i]


def reindent(block_src: str) -> str:
    return "\n".join(
        (INDENT + line) if line.strip() else "" for line in block_src.split("\n")
    )


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    src_lines = text.split("\n")
    tree = ast.parse(text)
    func = next(
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "build_report"
    )
    body = func.body
    doc_node = body[0]
    assert isinstance(doc_node, ast.Expr), "expected docstring first"
    stmts = body[1:]

    # setup = leading Assign/AnnAssign run
    setup_end = 0
    for i, s in enumerate(stmts):
        if is_assign(s):
            setup_end = i + 1
        else:
            break
    setup = stmts[:setup_end]
    rest = stmts[setup_end:]
    assert isinstance(rest[-1], ast.Return), "last stmt must be return"
    render, return_node = rest[:-1], rest[-1]

    # partition at H2 headings
    parts: list[tuple[str, list[ast.stmt]]] = []
    head_idx = []
    for i, s in enumerate(render):
        if heading_number(s) is not None:
            head_idx.append(i)
    assert head_idx, "no H2 headings"
    # partition at H2 headings: bounds = [0] + head_idx + [len(render)]
    bounds = [0] + head_idx + [len(render)]
    seen: set[str] = set()
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        part = render[a:b]
        if a == 0:
            parts.append(("preamble", part))
            continue
        num = heading_number(part[0])
        assert num is not None, f"section start not a heading: {ast.unparse(part[0])[:50]}"
        assert num not in seen, f"duplicate section {num}"
        seen.add(num)
        parts.append((num, part))
        for s in part:
            assert not is_assign(s), f"top-level Assign in section {num}: {ast.unparse(s)[:50]}"

    def line_span(nodes: list[ast.stmt]) -> str:
        """Verbatim original source lines covering these statements (keeps
        the original 4-space indentation, so no re-indent is needed)."""
        lo = min(n.lineno for n in nodes)
        hi = max(n.end_lineno for n in nodes)
        return "\n".join(src_lines[lo - 1 : hi])

    helpers_src: list[str] = []
    calls_src: list[str] = []
    for num, part in parts:
        block = line_span(part)
        helpers_src.append(f"\n\ndef _section_{num}{SIG} -> None:\n{block}\n")
        calls_src.append(f"{INDENT}_section_{num}(add, cfg, rows, meta, total, stage_totals, floors)\n")

    setup_src = line_span(setup)
    ret_src = line_span([return_node])
    new_func = (
        "def build_report(cfg: ShapingConfig, rows: list[dict], meta: dict, floors: list[dict] | None) -> str:\n"
        + INDENT
        + seg(text, doc_node)
        + "\n"
        + setup_src
        + "\n"
        + "".join(calls_src)
        + ret_src
        + "\n"
    )

    a, b = func.lineno, func.end_lineno
    new_text = (
        "\n".join(src_lines[: a - 1])
        + "\n"
        + new_func
        + "".join(helpers_src)
        + "\n".join(src_lines[b:])
        + "\n"
    )
    new_text = new_text.rstrip("\n") + "\n"
    ast.parse(new_text)
    TARGET.write_text(new_text, encoding="utf-8")
    print(f"split {len(parts)} parts: {[n for n, _ in parts]}")


if __name__ == "__main__":
    main()
