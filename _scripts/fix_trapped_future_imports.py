"""Fix 23 __init__.py files where `from __future__ import annotations` got
trapped inside a multi-line module docstring (F-5 slice-1 defect from d45e4e19).

Transform: remove the import line from inside the docstring, re-insert it as a
top-level statement immediately after the docstring's closing delimiter.

Hard verification per file (aborts on any failure, file left untouched):
  1. body[0] is still the docstring (Constant, str)
  2. the import is a top-level ImportFrom in body
  3. compile() sets CO_FUTURE_ANNOTATIONS (0x1000000)  -> PEP 563 actually on
  4. docstring text content is preserved (minus the removed import line)
  5. exactly one import line, at column 0, in the original
"""
from __future__ import annotations

import sys
import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
DRY = "--apply" not in sys.argv
IMPORT = "from __future__ import annotations\n"
CO_FUTURE = 0x01000000

files = []
for base in ("src", "scripts"):
    files += list((REPO / base).rglob("*.py"))

trapped = []
for p in files:
    text = p.read_text(encoding="utf-8-sig")
    if IMPORT not in text and "from __future__ import annotations" not in text:
        continue
    m = ast.parse(text)
    doc_str = ast.get_docstring(m, clean=False)  # raw string incl. the trapped line
    top_import = any(
        isinstance(x, ast.ImportFrom) and x.module == "__future__"
        and any(a.name == "annotations" for a in x.names)
        for x in m.body
    )
    doc_contains = bool(doc_str and "from __future__" in doc_str)
    if (not top_import) and doc_contains:
        trapped.append(p)

print(f"candidates trapped: {len(trapped)}")

ok, skipped = [], []
for p in trapped:
    raw = p.read_text(encoding="utf-8")
    # BOM guard: these should be clean (no BOM); if BOM present, skip (separate risk).
    if raw.startswith("\ufeff"):
        skipped.append((str(p), "BOM present"))
        continue
    lines = raw.splitlines(keepends=True)
    # locate the standalone import line (column 0)
    idxs = [i for i, ln in enumerate(lines) if ln == IMPORT]
    if len(idxs) != 1:
        skipped.append((str(p), f"import line occurrences={len(idxs)}"))
        continue
    imp = idxs[0]
    # docstring end line (1-based) from AST
    m = ast.parse(raw)
    d0 = m.body[0]
    end_lineno = d0.end_lineno  # 1-based line where docstring closes
    # remove import line
    new_lines = lines[:imp] + lines[imp + 1:]
    # after removal, docstring end is at 1-based (end_lineno - 1); 0-based index end_lineno-2
    # insert immediately after that line
    insert_at = (end_lineno - 1) - 1 + 1  # 0-based position just after closing line
    # guard: the closing line may be the file's last line with no trailing newline
    # (e.g. micro_dm/__init__.py); without \n the import fuses onto the `"""` line.
    if insert_at > 0 and not new_lines[insert_at - 1].endswith("\n"):
        new_lines[insert_at - 1] += "\n"
    new_lines = new_lines[:insert_at] + [IMPORT] + new_lines[insert_at:]
    new_text = "".join(new_lines)

    # ---- verify ----
    nm = ast.parse(new_text)
    new_doc0 = ast.get_docstring(nm, clean=False)
    if not new_doc0:
        skipped.append((str(p), "post: body[0] not docstring"))
        continue
    has_top = any(
        isinstance(x, ast.ImportFrom) and x.module == "__future__"
        and any(a.name == "annotations" for a in x.names)
        for x in nm.body
    )
    if not has_top:
        skipped.append((str(p), "post: import not top-level"))
        continue
    code = compile(new_text, str(p), "exec")
    if not (code.co_flags & CO_FUTURE):
        skipped.append((str(p), "post: CO_FUTURE not set"))
        continue
    # docstring content preserved except the removed import line (strict)
    old_doc = ast.get_docstring(m, clean=False)
    new_doc = ast.get_docstring(nm, clean=False)
    old_clean = "\n".join(
        ln for ln in old_doc.split("\n") if ln.strip() != "from __future__ import annotations"
    )
    if old_clean != new_doc:
        skipped.append((str(p), "docstring content mismatch"))
        continue

    if DRY:
        if not ok:
            show = p.read_text(encoding="utf-8").splitlines()[:3]
            show_new = new_text.splitlines()[:4]
            print(f"  [dry] {p}: docstring now opens with {show_new[0]!r}")
    else:
        p.write_text(new_text, encoding="utf-8")
    ok.append(str(p))

print(f"\nFIXED ({len(ok)}):")
for f in ok:
    print("  ", f)
print(f"\nSKIPPED ({len(skipped)}):")
for f, why in skipped:
    print("  ", f, "->", why)
