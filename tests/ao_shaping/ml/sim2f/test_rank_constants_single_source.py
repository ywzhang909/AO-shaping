"""``NDIM_*`` may have exactly one definition site in the whole ``ml`` package.

Item 116 consolidated the tensor ranks into :mod:`ml.sim2f.config`, which ``sim2f.optics``
and ``sim2f.losses`` import. Items 122 and 123 made ``ml.zernike.losses`` and
``ml.phase.dataset`` consumers from further subpackages, so the scan is package-wide: a
re-declaration in any subpackage drifts silently, and two shape guards can then mean different
ranks under one name, sending training after the wrong axis convention instead of raising.
The mirror rule is also pinned: a rank that one module laid out for itself stays private --
hoisting it into the hub would advertise a convention nobody else asked for.

The check is AST-only -- it parses source instead of importing it -- so it cannot be skipped
for an environmental reason (no torch, no device, no corpus).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[4] / "src" / "ml"
HUB = "sim2f/config.py"
RANK_NAMES = {"NDIM_MATRIX", "NDIM_BARE_MAP", "NDIM_BATCH_MAP", "NDIM_BATCHED_MAP"}
RANK_VALUES = {
    "NDIM_MATRIX": 2,
    "NDIM_BARE_MAP": 2,
    "NDIM_BATCH_MAP": 3,
    "NDIM_BATCHED_MAP": 4,
}
#: Every module that is allowed to *consume* the ranks, relative to ``PKG``. A module that
#: stops referencing them is reported, because the guard would then be watching fewer call
#: sites than the consolidation claimed.
CONSUMERS = (
    "sim2f/optics.py",
    "sim2f/losses.py",
    "sim2f/bench.py",
    "zernike/losses.py",
    "phase/dataset.py",
)
HUB_MODULE = "ml.sim2f.config"
#: A rank one module chose for its own layout. It must stay in that module: promoting it
# to the hub would imply a package-wide convention, and a second module declaring the same
# private name would imply the layouts agree when nothing enforces it.
LOCAL_RANKS = {"_NDIM_CHANNELLED_IMAGE": "phase/dataset.py"}


def _module_defs(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in RANK_NAMES and isinstance(node.value, ast.Constant)
    }


def _imported_names(tree: ast.AST, module: str) -> set[str]:
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


class TestRankConstantsSingleSource:
    def test_package_directory_is_reachable(self):
        assert PKG.is_dir(), f"expected package directory {PKG}"

    def test_each_rank_name_is_defined_only_in_the_hub(self):
        owners: dict[str, list[str]] = {name: [] for name in RANK_NAMES}
        for module in sorted(PKG.rglob("*.py")):
            for name in _module_defs(module):
                owners[name].append(module.relative_to(PKG).as_posix())
        for name, modules in owners.items():
            assert modules == [HUB], f"{name} defined in {modules}; expected {HUB} only"

    def test_rank_values_match_the_documented_axes(self):
        assert _module_defs(PKG / HUB) == RANK_VALUES

    def test_module_local_ranks_stay_in_their_module(self):
        for name, owner in sorted(LOCAL_RANKS.items()):
            declared, referenced = [], []
            for module in sorted(PKG.rglob("*.py")):
                body = module.read_text(encoding="utf-8")
                rel = module.relative_to(PKG).as_posix()
                tree = ast.parse(body)
                if any(
                    isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == name
                    for node in tree.body
                ):
                    declared.append(rel)
                if re.search(rf"\b{name}\b", body):
                    referenced.append(rel)
            assert declared == [owner], f"{name} declared in {declared}; expected {owner} only"
            assert referenced == [owner], f"{name} referenced outside {owner}: {referenced}"

    def test_consumers_import_ranks_instead_of_redeclaring(self):
        for consumer in CONSUMERS:
            path = PKG / consumer
            assert path.is_file(), f"missing consumer {consumer}"
            assert _module_defs(path) == {}, f"{consumer} re-declares rank constants"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            used = {n.id for n in ast.walk(tree)
                    if isinstance(n, ast.Name) and n.id in RANK_NAMES}
            assert used, f"{consumer} references no rank constant at all"
            imported = _imported_names(tree, HUB_MODULE)
            assert used <= imported, (
                f"{consumer} uses {sorted(used - imported)} without importing them")
