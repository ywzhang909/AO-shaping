#!/usr/bin/env python3
"""ponytail-lean 快速扫描：死函数 / 未 import 模块候选 / 同名孪生候选。

用法:
    python3 ponytail_lean_scan.py <repo_root> [extra_dir ...]

默认扫描 <repo_root>/src（定义收集范围）；额外目录（如 tests scripts）仅计入引用。
AO-shaping 全量（src+tests+scripts，~70 万行）单次约 35s。

输出 JSON:
  dead_strict        全树 0 命中（除 def 行）且本文件 0 额外使用 —— 可报死
  dead_maybe         全树 0 命中，本文件 1+ 额外使用（可能 docstring 自述）—— 需人眼
  unimported_candidates  import 图未命中的 src 模块 —— 全部是误报候选，
                         必须按 references/ponytail-lean-audit.md 第 2 步全树
                         grep（含字符串惰性 import/测试常量表）逐个复核后才能报
  twin_name_candidates   跨 >=3 个 src 模块的同名顶层函数 —— 需 difflib
                         SequenceMatcher 行级 ratio >= 0.4 才按孪生报

注意：import 图只含 level-0 的 import / from import。相对导入、字符串惰性
注册（register_*_lazy）、延迟 import 均不在图内，这是 unimported 列表必须
人工复核的原因。
"""
from __future__ import annotations

import ast
import collections
import json
import os
import re
import sys

IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]{2,}")
SKIP_DIRS = ("__pycache__", ".venv", ".git", "node_modules")


def py_files(roots):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in SKIP_DIRS]
            for f in fn:
                if f.endswith(".py"):
                    yield os.path.join(dp, f)


def main() -> None:
    repo = os.path.abspath(sys.argv[1])
    ref_dirs = [repo] + [os.path.abspath(a) for a in sys.argv[2:]]

    texts: dict[str, str] = {}
    per_file: dict[str, collections.Counter] = {}
    all_words: collections.Counter = collections.Counter()
    for p in py_files(ref_dirs):
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                t = fh.read()
        except OSError:
            continue
        rel = os.path.relpath(p, repo)
        texts[rel] = t
        words = IDENT.findall(t)
        per_file[rel] = collections.Counter(words)
        all_words.update(words)

    # import 图（level-0 仅；前缀全展开）
    imported: set[str] = set()
    for rel, t in texts.items():
        try:
            tree = ast.parse(t)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    parts = a.name.split(".")
                    imported.update(".".join(parts[:k]) for k in range(1, len(parts) + 1))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                parts = node.module.split(".")
                imported.update(".".join(parts[:k]) for k in range(1, len(parts) + 1))

    # src/ 顶层定义
    defns: dict[str, dict[str, int]] = {}
    for p in py_files([os.path.join(repo, "src")]):
        if "egg-info" in p:
            continue
        rel = os.path.relpath(p, repo)
        if not rel.startswith("src/") or rel.endswith("__init__.py"):
            continue
        t = texts.get(rel)
        if t is None:
            continue
        try:
            tree = ast.parse(t)
        except SyntaxError:
            continue
        defs: dict[str, int] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs[node.name] = node.lineno
            elif isinstance(node, ast.ClassDef):
                defs[node.name] = node.lineno
        if defs:
            defns[rel] = defs

    def modpath(rel: str):
        r = rel[len("src/"):]
        if r.endswith("__init__.py"):
            return None
        if r.endswith(".py"):
            r = r[:-3]
        return r.replace("/", ".")

    unimported = []
    for rel in defns:
        m = modpath(rel)
        if m and not any(x == m or x.startswith(m + ".") for x in imported):
            unimported.append(rel)

    dead_strict: list[list] = []
    dead_maybe: list[list] = []
    for rel, defs in defns.items():
        own = per_file[rel]
        for name, ln in defs.items():
            total = all_words[name]
            own_c = own[name]
            own_extra = own_c - 1  # 扣掉 def 行本身
            other = total - own_c
            if other == 0:
                row = [rel, ln, name, own_extra]
                (dead_strict if own_extra == 0 else dead_maybe).append(row)

    name_locs: dict[str, list] = collections.defaultdict(list)
    for rel, defs in defns.items():
        for name, ln in defs.items():
            name_locs[name].append([rel, ln])
    twins = {
        n: v
        for n, v in name_locs.items()
        if len({x[0] for x in v}) >= 3
        and n not in ("main", "run", "__init__")
        and not n.startswith("_")
    }
    twins = dict(sorted(twins.items(), key=lambda kv: -len({x[0] for x in kv[1]}))[:30])

    out = {
        "dead_strict": sorted(dead_strict),
        "dead_maybe": sorted(dead_maybe),
        "unimported_candidates": sorted(unimported),
        "twin_name_candidates": twins,
    }
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
    print()


if __name__ == "__main__":
    main()
