"""Generic mechanical section-splitter for markdown report builders.

Usage: refactor_report_sections.py <file.py> <func_name>

Algorithm (zero text mutation beyond the verbatim span lift):
  1. locate the target function; first body stmt = docstring
  2. setup = leading run of Assign/AnnAssign/AugAssign (the data-compute layer)
  3. render = everything after setup up to the final `return`
  4. split render at every top-level ``add("## ...")`` heading -> partitions
     named by POSITION (_part_0, _part_1, ...) so duplicate/non-numeric
     headings never collide
  5. each partition becomes ``def _part_k(add, <used setup vars>, <used
     function args>) -> None:`` containing the partition's verbatim original
     source lines (so no re-indent is even needed -- the partition lines are
     already at build-report-body depth, and the helper body is at the same
     depth because the helper is at module level... NO: helper is at module
     level, its body needs 4-space. So we DO re-indent the partition body by
     one level. The partition's original lines are at 4-space (inside
     build_report); inside the new module-level helper they must be 8-space.
     Hence reindent(+4).
  6. build_report becomes: docstring + setup + one call per partition + return

Each partition receives only the setup vars / function args it actually uses
(free-name analysis), so helper signatures stay minimal. Module-level globals
and builtins are left unpas... they are visible to the helpers too.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def targets_of(s: ast.stmt):
    out = []
    if isinstance(s, ast.Assign):
        out += s.targets
    elif isinstance(s, (ast.AugAssign, ast.AnnAssign)):
        out.append(s.target)
    return out


def names_of(node) -> set[str]:
    return {nm.id for nm in ast.walk(node) if isinstance(nm, ast.Name)}


def is_assign(s: ast.stmt) -> bool:
    return isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign))


def is_h2_heading(s: ast.stmt) -> bool:
    if not isinstance(s, ast.Expr):
        return False
    call = s.value
    if not isinstance(call, ast.Call):
        return False
    if not (isinstance(call.func, ast.Name) and call.func.id == "add"):
        return False
    if not call.args:
        return False
    a0 = call.args[0]
    return isinstance(a0, ast.Constant) and isinstance(a0.value, str) and a0.value.startswith("## ")


def reindent(block_src: str, extra: int = 4) -> str:
    pad = " " * extra
    return "\n".join((pad + ln) if ln.strip() else "" for ln in block_src.split("\n"))


def main() -> None:
    target = Path(sys.argv[1])
    fname = sys.argv[2]
    text = target.read_text(encoding="utf-8")
    src_lines = text.split("\n")
    tree = ast.parse(text)
    func = next(
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fname
    )

    # function arg names (no self/cls -- these are module-level functions)
    arg_names = [a.arg for a in func.args.args]

    body = func.body
    doc_node = body[0]
    assert isinstance(doc_node, ast.Expr), "expected docstring first"
    stmts = body[1:]

    # setup prefix
    se = 0
    for i, s in enumerate(stmts):
        if is_assign(s):
            se = i + 1
        else:
            break
    setup = stmts[:se]
    rest = stmts[se:]
    assert isinstance(rest[-1], ast.Return), "last stmt must be return"
    render, return_node = rest[:-1], rest[-1]

    # setup var names
    setup_vars: list[str] = []
    setup_set: set[str] = set()
    for s in setup:
        for t in targets_of(s):
            for nm in names_of(t):
                if nm not in setup_set:
                    setup_set.add(nm)
                    setup_vars.append(nm)

    # partition at H2 headings
    hi = [i for i, s in enumerate(render) if is_h2_heading(s)]
    bounds = [0] + hi + [len(render)]
    partitions: list[list[ast.stmt]] = []
    for k in range(len(bounds) - 1):
        partitions.append(render[bounds[k]:bounds[k + 1]])

    # free names per partition
    def free_names(part):
        f = set()
        for s in part:
            for nm in ast.walk(s):
                if isinstance(nm, ast.Name) and isinstance(nm.ctx, ast.Load):
                    f.add(nm.id)
        return f

    # per partition: which setup vars + which function args are used
    order = arg_names + setup_vars  # preference order for helper signature
    helpers_src: list[str] = []
    calls_src: list[str] = []
    def line_span(nodes):
        lo = min(n.lineno for n in nodes)
        hi2 = max(n.end_lineno for n in nodes)
        return "\n".join(src_lines[lo - 1 : hi2])

    for k, part in enumerate(partitions):
        fn = free_names(part)
        params = [n for n in order if n in fn and n != "add"]
        block = line_span(part)  # verbatim original lines (already at 4-space depth)
        sig = f"({', '.join(['add'] + params)})"
        defn = f"\n\ndef _part_{k}{sig} -> None:\n{block}\n"
        helpers_src.append(defn)
        arglist = ", ".join(["add"] + params)
        calls_src.append(f"    _part_{k}({arglist})\n")

    # assemble new build func
    def span(nodes):
        lo = min(n.lineno for n in nodes)
        hi2 = max(n.end_lineno for n in nodes)
        return "\n".join(src_lines[lo - 1:hi2])

    setup_src = span(setup)
    ret_src = span([return_node])
    doc_src = ast.get_source_segment(text, doc_node)
    new_func = (
        f"def {fname}({', '.join(arg_names) if arg_names else ''}):\n"
    )
    # reconstruct signature faithfully from source
    a = func.lineno
    # take original def line(s) up to the colon (signature may span multiple lines)
    # find the line where the def's closing paren is
    sig_lines = src_lines[func.lineno - 1:]
    depth = 0
    sig_end = None
    for j, ln in enumerate(sig_lines):
        depth += ln.count("(") - ln.count(")")
        if depth == 0 and (":" in ln) and "(" in "".join(sig_lines[: j + 1]):
            sig_end = j
            break
    sig_block = "\n".join(sig_lines[: sig_end + 1])
    new_func = sig_block + "\n" + "    " + doc_src + "\n" + setup_src + "\n" + "".join(calls_src) + ret_src + "\n"

    before = src_lines[: a - 1]
    after = src_lines[func.end_lineno:]
    new_text = "\n".join(before) + "\n" + new_func + "".join(helpers_src) + "\n".join(after) + "\n"
    new_text = new_text.rstrip("\n") + "\n"
    ast.parse(new_text)
    target.write_text(new_text, encoding="utf-8")
    print(f"split {fname} into {len(partitions)} parts")


if __name__ == "__main__":
    main()
