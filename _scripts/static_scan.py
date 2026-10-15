"""Static scan: long functions / deep nesting / cyclomatic complexity + compat imports."""
import ast
import pathlib
from collections import Counter

ROOT = pathlib.Path("src")
SCRIPTS = pathlib.Path("scripts")
FILES = [p for p in list(ROOT.rglob("*.py")) + list(SCRIPTS.rglob("*.py"))
         if "__pycache__" not in str(p) and "vendor" not in str(p) and ".venv" not in str(p)]

BRACES = (ast.If, ast.For, ast.While, ast.With, ast.AsyncFor, ast.AsyncWith, ast.Try)

def max_depth(node):
    best = [0]
    def walk(n, depth):
        if isinstance(n, BRACES):
            depth += 1
            best[0] = max(best[0], depth)
        for c in ast.iter_child_nodes(n):
            walk(c, depth)
    walk(node, 0)
    return best[0]

def cyclomatic(node):
    c = 1
    for n in ast.walk(node):
        if isinstance(n, (ast.If, ast.For, ast.While, ast.AsyncFor, ast.IfExp, ast.Try, ast.comprehension)):
            c += 1
        if isinstance(n, ast.BoolOp):
            c += len(n.values) - 1
    return c

long_f, deep_f, cc_f = [], [], []
for p in FILES:
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    except SyntaxError:
        continue
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        length = (fn.end_lineno or fn.lineno) - fn.lineno + 1
        d = max_depth(fn)
        cc = cyclomatic(fn)
        if length > 80:
            long_f.append((length, str(p), fn.name, fn.lineno, d, cc))
        if d >= 5:
            deep_f.append((d, str(p), fn.name, fn.lineno, cc, length))
        if cc >= 20:
            cc_f.append((cc, str(p), fn.name, fn.lineno, length, d))

print("=== LONG (>80 lines): %d ===" % len(long_f))
for l, p, q, ln, d, cc in sorted(long_f, reverse=True)[:40]:
    print(f"{l:5d}  {p}:{q}@{ln}  depth={d} cc={cc}")
print("\n=== DEEP (>=5): %d ===" % len(deep_f))
for d, p, q, ln, cc, l in sorted(deep_f, reverse=True)[:40]:
    print(f"d{d}  {p}:{q}@{ln}  cc={cc} lines={l}")
print("\n=== CC (>=20): %d ===" % len(cc_f))
for cc, p, q, ln, l, d in sorted(cc_f, reverse=True)[:40]:
    print(f"cc{cc:3d}  {p}:{q}@{ln}  lines={l} depth={d}")

# compat imports: try: import X except ImportError
print("\n=== COMPAT IMPORTS (try/except ImportError): count per file ===")
cnt = Counter()
samples = {}
for p in FILES:
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    except SyntaxError:
        continue
    for n in ast.walk(tree):
        if isinstance(n, ast.Try):
            names = []
            for h in n.handlers:
                t = h.type
                if isinstance(t, ast.Name) and t.id in ("ImportError", "OSError"):
                    for stmt in n.body:
                        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                            if isinstance(stmt, ast.Import):
                                names.append(stmt.names[0].name + (f" as {stmt.names[0].asname}" if stmt.names[0].asname else ""))
                            else:
                                m = (stmt.module or "")
                                nms = [a.name + (f" as {a.asname}" if a.asname else "") for a in stmt.names]
                                names.append(f"from {m} import " + ", ".join(nms))
            if names:
                cnt[str(p)] += len(names)
                samples.setdefault(str(p), names[:6])
for p, c in sorted(cnt.items(), key=lambda kv: -kv[1]):
    print(f"{c:3d}  {p}")
    for s in samples[p]:
        print(f"      {s}")
