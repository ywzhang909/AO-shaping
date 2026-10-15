"""Scan __init__.py re-exports for stale names + alias patterns."""
import ast
import pathlib

ROOT = pathlib.Path("src")
FILES = [p for p in ROOT.rglob("__init__.py") if "__pycache__" not in str(p)]

def local_bindings(tree):
    bound = set()
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            if isinstance(n, ast.Import):
                for a in n.names:
                    bound.add((a.asname or a.name).split(".")[0])
            else:
                star = False
                for a in n.names:
                    if a.name == "*":
                        star = True
                    else:
                        bound.add(a.asname or a.name)
                if star:
                    bound.add("__STAR__")
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    bound.add(t.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            bound.add(n.target.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, (ast.If, ast.Try)):
            for sub in ast.walk(n):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    if isinstance(sub, ast.Import):
                        for a in sub.names:
                            bound.add((a.asname or a.name).split(".")[0])
                    else:
                        for a in sub.names:
                            if a.name == "*":
                                bound.add("__STAR__")
                            else:
                                bound.add(a.asname or a.name)
                elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    bound.add(sub.name)
    return bound

stale = []
for p in sorted(FILES):
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    except SyntaxError:
        continue
    allnames = []
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == "__all__" and isinstance(n.value, ast.List):
                    allnames = [e.value for e in n.value.elts if isinstance(e, ast.Constant)]
    bound = local_bindings(tree)
    if "__STAR__" in bound or not allnames:
        continue
    for name in allnames:
        if name not in bound:
            stale.append((str(p), name))

print("=== __all__ stale re-exports (name not bound in module) ===")
for p, n in stale:
    print(f"  {p}: {n}")
print(f"total: {len(stale)}")
