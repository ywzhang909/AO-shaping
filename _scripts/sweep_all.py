"""Full runtime sweep: for every __init__.py with an __all__, report entries that fail getattr."""
import ast
import importlib
import pathlib

ROOT = pathlib.Path("src")
inits = sorted(p for p in ROOT.rglob("__init__.py") if "__pycache__" not in str(p))

def all_of(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(p))
    except SyntaxError:
        return None
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    if isinstance(n.value, ast.List):
                        return [e.value for e in n.value.elts if isinstance(e, ast.Constant)]
    return None

total_missing = 0
for p in inits:
    names = all_of(p)
    if not names:
        continue
    modpath = p.with_suffix("").as_posix().replace("/", ".")
    # only test modules under src/ao_shaping or src/ml
    if not (modpath.startswith("ao_shaping") or modpath.startswith("ml")):
        continue
    try:
        mod = importlib.import_module(modpath)
    except Exception as e:
        print(f"[skip import] {modpath}: {type(e).__name__}: {e}")
        continue
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        total_missing += len(missing)
        print(f"{modpath}: {len(missing)}/{len(names)} stale")
        for n in missing:
            print(f"    - {n}")
print(f"\nTOTAL stale __all__ entries: {total_missing}")
