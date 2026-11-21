"""#251 P1-A — every ``--dry-run`` path must be self-contained in CI.

CI 容器保证 **不存在** ``data/`` 与任何机器绝对路径 (见
``docs/ci-environment-contract.md``)。ci169 的 #5/#8 两红是同一模式:
dry-run 在开发机上写就、悄悄读了 ``data/`` 下的真实产物, push 后才在容器里
爆 ``FileNotFoundError``。第 170 批的修法 (合成回退) 逐条正确, 但通道没有封:
下一个 dry-run 仍可以读另一个不存在的文件。

本守卫静态扫描 ``scripts/**/*.py`` 中带 ``--dry-run`` 的脚本: 任何
dry-run 可达函数里对 ``data/`` (或 ``libs/gxipy``) 的读取调用, 其同函数内
必须存在**守卫式回退** (``is_file()`` / ``exists()`` 判断, 或
``synthetic*`` 回退调用), 否则违规。豁免清单逐条带理由, 收紧单向。
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _ROOT / "scripts"

#: 读取外部文件的最常见调用/方法名 (AST 属性末端或裸名)。
_READERS = {
    "open",
    "load",
    "loadtxt",
    "loads",
    "read_text",
    "read_bytes",
    "read_csv",
    "read_json",
    "loadfrozen",
}

#: 视为"有回退守卫"的证据 (同函数体内出现任一即合规)。
_FALLBACK_MARKERS = {"is_file", "exists", "synthetic", "dry_run_default", "missing"}

#: 违规豁免: file -> reason。收紧单向 —— 只允许删条目, 新违规不得入列。
_EXEMPT: dict[str, str] = {}


def _reads_data(node: ast.AST) -> bool:
    """节点子树内是否有对 ``data/`` 或 ``libs/gxipy`` 的读取调用。"""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name not in _READERS:
            continue
        blob = ast.dump(sub)
        if "data/" in blob or "data\\\\" in blob or "gxipy" in blob:
            return True
    return False


def _has_fallback(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr in _FALLBACK_MARKERS:
            return True
        if isinstance(sub, ast.Call):
            name = getattr(sub.func, "attr", None) or getattr(sub.func, "id", None)
            if isinstance(name, str) and name.startswith("synthetic"):
                return True
    return False


def _dry_run_functions(tree: ast.AST) -> list[ast.AST]:
    """dry-run 可达函数: 名字含 dry_run/dryrun, 或体内引用 dry_run。"""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            blob = ast.dump(node)
            if "dry_run" in node.name or "dryrun" in node.name or "dry_run" in blob:
                out.append(node)
    return out


def test_dry_run_paths_have_a_guarded_fallback_for_external_reads() -> None:
    offenders: list[str] = []
    for path in sorted(_SCRIPTS.rglob("*.py")):
        rel = path.relative_to(_SCRIPTS).as_posix()
        if rel in _EXEMPT:
            continue
        source = path.read_text(encoding="utf-8-sig", errors="replace")
        if "--dry-run" not in source:
            continue
        tree = ast.parse(source)
        for fn in _dry_run_functions(tree):
            if _reads_data(fn) and not _has_fallback(fn):
                offenders.append(f"{rel}::{fn.name}")
    assert not offenders, (
        "dry-run 路径读取 data/ 或 gxipy 却无合成/存在性回退 —— CI 容器上必 "
        "FileNotFoundError (#251 P1-A, docs/ci-environment-contract.md)。"
        f"违规: {offenders}"
    )


def test_exemption_list_is_shrinking_not_stale() -> None:
    """豁免条目必须仍对应真实存在的文件; 空豁免集本身就是目标状态。"""
    for rel in _EXEMPT:
        assert (_SCRIPTS / rel).is_file(), f"豁免清单指向不存在的脚本: {rel}"


def test_the_guard_itself_catches_a_bare_read() -> None:
    """负样本 (验收 #8): 无回退的 data/ 读取必须被判违规。"""
    src = (
        "import argparse\n\n\n"
        "def run_dry_run(args):\n"
        "    return open('data/debug/x.json').read()\n"
    )
    tree = ast.parse(src)
    fns = _dry_run_functions(tree)
    assert len(fns) == 1 and _reads_data(fns[0]) and not _has_fallback(fns[0])
    guarded = (
        "import argparse\n\n\n"
        "def run_dry_run(args):\n"
        "    p = __import__('pathlib').Path('data/debug/x.json')\n"
        "    if p.is_file():\n"
        "        return p.read_text()\n"
        "    return synthetic_x()\n"
    )
    gtree = ast.parse(guarded)
    gfns = _dry_run_functions(gtree)
    assert len(gfns) == 1 and not (_reads_data(gfns[0]) and not _has_fallback(gfns[0]))
