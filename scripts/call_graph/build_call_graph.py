"""Build a function-call graph rooted at main.py for the AO-shaping repo.

Outputs JSON (nodes/edges) + stats. Resolution:
  - top-level functions and class methods per module
  - imports (module imports, from-imports)
  - calls: Name, Attribute (module-qualified), partial (fn passed as arg)
External (non-repo) callees are marked, not expanded.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

# scripts/call_graph/<this file> -> repo root is two levels up
REPO = Path(__file__).resolve().parents[2]
SRC_DIRS = [REPO / "src", REPO]  # root main.py + src/

STDLIB = set(sys.stdlib_module_names)


def mod_dotted(path: Path) -> str | None:
    rel = path.relative_to(REPO)
    if rel == Path("main.py"):
        return "main"
    parts = list(rel.parts)
    if parts[0] in ("libs", "tests", "docker", "notebooks"):
        return None
    if parts[0] == "src":
        parts = parts[1:]
    parts = list(Path(*parts).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    return ".".join(parts)


class Func:
    def __init__(self, qualname: str, module: str, f: ast.AST, file: Path):
        self.qualname = qualname          # dotted: module.Class.method
        self.module = module
        self.name = qualname.split(".")[-1]
        self.cls = None
        self.lineno = f.lineno
        self.end = getattr(f, "end_lineno", f.lineno)
        self.args = []
        for a in f.args.args:
            self.args.append(a.arg)
        if f.args.vararg:
            self.args.append("*" + f.args.vararg.arg)
        if f.args.kwonlyargs:
            self.args.extend(a.arg for a in f.args.kwonlyargs)
        if f.args.kwarg:
            self.args.append("**" + f.args.kwarg.arg)
        self.defaults_n = len(f.args.defaults)
        self.returns = ast.unparse(f.returns) if f.returns else None
        self.doc = ast.get_docstring(f, clean=False) or ""
        self.decorators = [ast.unparse(d) for d in f.decorator_list]
        self.file = str(file.relative_to(REPO))
        self.node = f


class Module:
    def __init__(self, dotted: str, file: Path):
        self.dotted = dotted
        self.file = file
        self.tree = None
        self.funcs: dict[str, Func] = {}          # local name -> Func (top-level)
        self.methods: dict[str, Func] = {}        # Class.method (qualname w/o module)
        self.imports: dict[str, str] = {}         # local name -> target module
        self.from_imports: dict[str, tuple[str, str]] = {}  # local -> (mod, orig)
        self.aliases: dict[str, str] = {}         # alias -> real module dotted


def load_modules() -> dict[str, Module]:
    mods: dict[str, Module] = {}
    for base in SRC_DIRS:
        for p in sorted(base.rglob("*.py")):
            if ".venv" in p.parts or "__pycache__" in p.parts:
                continue
            if p == REPO / "main.py" or p.parent != REPO:
                pass
            d = mod_dotted(p)
            if d is None or d in mods:
                continue
            try:
                text = p.read_text(encoding="utf-8")
                tree = ast.parse(text)
            except SyntaxError:
                try:
                    text = p.read_text(encoding="utf-8-sig")
                    tree = ast.parse(text)
                except Exception as e:
                    print(f"PARSE FAIL {p}: {e}", file=sys.stderr)
                    continue
            except Exception as e:
                print(f"PARSE FAIL {p}: {e}", file=sys.stderr)
                continue
            m = Module(d, p)
            m.tree = tree
            mods[d] = m
            index_module(m, tree)
    return mods


def index_module(m: Module, tree: ast.AST):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            f = Func(node.name, m.dotted, node, m.file)
            m.funcs[node.name] = f
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fq = f"{node.name}.{sub.name}"
                    f = Func(f"{m.dotted}.{fq}", m.dotted, sub, m.file)
                    f.cls = node.name
                    m.methods[fq] = f
        elif isinstance(node, ast.Import):
            for a in node.names:
                tgt = a.name
                m.imports[a.asname or a.name] = tgt
                if a.asname:
                    m.aliases[a.asname] = tgt
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            mod = node.module
            for a in node.names:
                if a.name == "*":
                    continue
                local = a.asname or a.name
                m.from_imports[local] = (mod, a.name)
                m.imports[local] = mod  # for name-call resolution of submodules
                if a.asname:
                    m.aliases[local] = mod


def is_repo_module(name: str, mods: dict[str, Module]) -> bool:
    parts = name.split(".")
    first = parts[0]
    if first in STDLIB:
        return False
    return first in ("ao_shaping", "ml", "main", "calculators") or (mods.get(name) is not None)


def resolve_call(callee: ast.expr, func: Func, m: Module, mods: dict[str, Module]):
    """Return (target_qualname_or_None, kind) where kind in {'func','method','module','ext','unknown'}"""
    if isinstance(callee, ast.Name):
        nm = callee.id
        if nm in m.funcs:
            return m.funcs[nm].qualname, "func"
        if nm in m.methods:
            return f"{m.dotted}.{nm}", "method"
        if nm in m.from_imports:
            mod, orig = m.from_imports[nm]
            full = f"{mod}.{orig}"
            if full in mods:
                # from module import submodule  -> target is the module
                return full, "module"
            if is_repo_module(full, mods):
                return full, "ext"  # repo symbol in another module's namespace, not resolved further
            return None, "unknown"
        if nm in m.imports:
            return m.imports[nm], "module"
        # local var / builtin / external
        return None, "local"
    if isinstance(callee, ast.Attribute):
        # walk base
        base = callee.value
        if not isinstance(base, ast.Name):
            return None, "expr"
        # chain of attributes after first
        attrs = []
        cur = callee
        while isinstance(cur, ast.Attribute) and not isinstance(cur.value, ast.Name):
            attrs.append(cur.attr)
            cur = cur.value
        # cur is now Name(attr-chain root)
        root = cur.id if isinstance(cur, ast.Name) else None
        chain = [root] + list(reversed(attrs)) + [callee.attr]
        first = chain[0]
        # module import alias
        if first in m.aliases:
            tgt_mod = m.aliases[first]
            rest = chain[1:]
            # try to resolve rest as submodule path or func
            cand_mod = tgt_mod
            for i in range(len(rest) - 1):
                c2 = ".".join([tgt_mod] + rest[: i + 1])
                if c2 in mods:
                    cand_mod = c2
                    break
            final = rest[-1] if rest else None
            if cand_mod in mods:
                if final is None:
                    return cand_mod, "module"
                fq = f"{cand_mod}.{final}"
                mm = mods[cand_mod]
                if final in mm.funcs:
                    return f"{cand_mod}.{final}", "func"
                if final in mm.methods:
                    return fq, "method"
                return cand_mod, "module"
            return None, "ext"
        if first in m.from_imports:
            mod, orig = m.from_imports[first]
            rest = chain[1:]
            if not rest:
                return None, "local"
            # orig.attr(...)  e.g. from x import Y; Y.fn()
            full = f"{mod}.{orig}"
            if full in mods:
                mm = mods[full]
                fin = rest[-1]
                if fin in mm.funcs:
                    return f"{full}.{fin}", "func"
                if fin in mm.methods:
                    return f"{full}.{fin}", "method"
                return full, "module"
            return None, "ext"
        return None, "ext"
    return None, "unknown"


def collect_calls(func: Func, m: Module, mods: dict[str, Module]):
    """Return list of (target, kind, lineno) for calls inside func."""
    out = []
    for node in ast.walk(func.node):
        if isinstance(node, ast.Call):
            tgt, kind = resolve_call(node.func, func, m, mods)
            if tgt:
                out.append((tgt, kind, node.lineno))
    return out


def build():
    mods = load_modules()
    print(f"modules loaded: {len(mods)}", file=sys.stderr)
    # entries
    entries = {
        "main.main": None,  # placeholder
    }
    # root main: __main__ block calls cli()
    mmain = mods["ao_shaping.main"]
    entries["ao_shaping.main.cli"] = mmain

    # command registrations in ao_shaping.main: from runners import X ; cli.add_command(X, name=...)
    # map via runners __init__ _LAZY_RUNNERS (parsed via AST)
    init_tree = ast.parse((REPO / "src/ao_shaping/runners/__init__.py").read_text(encoding="utf-8-sig"))
    lazy, attrs = {}, {}
    for node in ast.walk(init_tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            tname = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            tname = node.targets[0].id
        else:
            continue
        if tname in ("_LAZY_RUNNERS", "_IMPORT_ATTRS") and isinstance(node.value, ast.Dict):
            out = {}
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                    out[k.value] = v.value
            if tname == "_LAZY_RUNNERS":
                lazy = out
            else:
                attrs = out

    cmd_entries = {}
    for name, sym in lazy.items():
        attr = attrs.get(name, "run")
        cmd_entries[name] = f"{sym}.{attr}"
    print(json.dumps(cmd_entries, indent=2), file=sys.stderr)

    # BFS
    nodes: dict[str, dict] = {}
    edges: dict[tuple, int] = {}

    def add_node(qual: str):
        if qual in nodes:
            return nodes[qual]
        # find func object
        fobj = None
        if "." not in qual:
            # module-level function in module 'main'
            mm = mods.get("main")
            if mm and qual in mm.funcs:
                fobj = mm.funcs[qual]
        else:
            parts = qual.split(".")
            # module could be 1+ dotted; find longest module prefix in mods
            for i in range(len(parts) - 1, 0, -1):
                cand = ".".join(parts[:i])
                if cand in mods:
                    mm = mods[cand]
                    rest = ".".join(parts[i:])
                    if rest in mm.funcs:
                        fobj = mm.funcs[rest]
                    elif rest in mm.methods:
                        fobj = mm.methods[rest]
                    break
        info = None
        if fobj is not None:
            info = {
                "qualname": fobj.qualname if not fobj.cls else fobj.qualname,
                "module": fobj.module,
                "file": fobj.file,
                "lineno": fobj.lineno,
                "end": fobj.end,
                "args": fobj.args,
                "returns": fobj.returns,
                "doc": fobj.doc,
                "decorators": fobj.decorators,
            }
        nodes[qual] = info
        return info

    def bfs(start_qual: str):
        # start may be module or function
        stack = [start_qual]
        while stack:
            cur = stack.pop()
            # expand module
            if cur in mods:
                m = mods[cur]
                # module-level calls: module is a node; its functions get discovered via their own calls when reached
                continue
            info = add_node(cur)
            if info is None:
                continue
            m = mods.get(info["module"])
            if m is None:
                continue
            fobj = None
            parts = cur.split(".")
            for i in range(len(parts) - 1, 0, -1):
                cand = ".".join(parts[:i])
                if cand in mods:
                    mm = mods[cand]
                    rest = ".".join(parts[i:])
                    if rest in mm.funcs:
                        fobj = mm.funcs[rest]
                    elif rest in mm.methods:
                        fobj = mm.methods[rest]
                    break
            if fobj is None:
                continue
            for tgt, kind, ln in collect_calls(fobj, m, mods):
                edges[(cur, tgt)] = edges.get((cur, tgt), 0) + 1
                if tgt in mods:  # module target
                    stack.append(tgt)
                else:
                    # function target: check it's a repo function (has info after add)
                    add_node(tgt)
                    stack.append(tgt)

    entry_points = []
    # main.__main__ -> cli
    entry_points.append("ao_shaping.main.cli")
    for sym in cmd_entries.values():
        entry_points.append(sym)
    # tools entries
    entry_points.append("ao_shaping.tools.slm.panel.diagnose.main")
    entry_points.append("ao_shaping.tools.slm.calibration.lut_runner.run")

    for e in entry_points:
        bfs(e)

    # stats
    repo_fns = [q for q, i in nodes.items() if i is not None and (i["module"] in mods)]
    ext = [q for q, i in nodes.items() if i is None]
    print(f"nodes: total={len(nodes)} resolved={len(repo_fns)} unresolved={len(ext)}", file=sys.stderr)
    print(f"edges: {len(edges)}", file=sys.stderr)
    from collections import Counter
    per_mod = Counter((i["module"] if i else q) for q, i in nodes.items())
    for k, v in per_mod.most_common():
        print(f"  {k}: {v}", file=sys.stderr)

    out = {
        "entries": entry_points,
        "nodes": {q: i for q, i in nodes.items()},
        "edges": [
            {"from": a, "to": b, "calls": n} for (a, b), n in edges.items()
        ],
        "cmd_entries": cmd_entries,
    }
    return out


if __name__ == "__main__":
    data = build()
    import sys as _s
    outp = REPO / "docs/dev/call_graph_wiki/raw/assets/call_graph.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"wrote {outp}", file=_s.stderr)
