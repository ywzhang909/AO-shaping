import ast, pathlib, sys
CO=0x1000000
bad_trapped=bad_flag=bad_doc=0
checked=0
root=pathlib.Path(__file__).resolve().parent.parent
for p in list(root.joinpath("src").rglob("__init__.py"))+list(root.joinpath("scripts").rglob("__init__.py")):
    t=p.read_text(encoding="utf-8-sig")
    if "from __future__ import annotations" not in t: continue
    m=ast.parse(t); checked+=1
    doc=ast.get_docstring(m,clean=False)
    if doc and "from __future__" in doc:
        bad_trapped+=1; print("STILL TRAPPED:",p)
    if not doc:
        bad_doc+=1; print("NO DOCSTRING:",p)
    top=any(isinstance(x,ast.ImportFrom) and x.module=="__future__" and any(a.name=="annotations" for a in x.names) for x in m.body)
    flag=bool(compile(t,str(p),"exec").co_flags & CO)
    if not top: bad_flag+=1; print("NOT TOP-LEVEL:",p)
    if not flag: bad_flag+=1; print("CO_FUTURE NOT SET:",p)
print(f"checked __init__ with import: {checked}")
print(f"trapped={bad_trapped} not-top-or-noflag={bad_flag} no-docstring={bad_doc}")
print("RESULT:", "ALL CLEAN" if not(bad_trapped or bad_flag or bad_doc) else "PROBLEMS")
