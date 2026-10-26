"""``NDIM_*`` may have exactly one definition site in the package.

Item 116 consolidated the tensor ranks into :mod:`ml.sim2f.config`, which ``optics`` and
``losses`` now import. A per-module re-declaration would drift silently: two shape guards
could then mean different ranks under one name, and training would follow the wrong axis
convention instead of raising.
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[4] / "src" / "ml" / "sim2f"
RANK_NAMES = {"NDIM_MATRIX", "NDIM_BARE_MAP", "NDIM_BATCH_MAP", "NDIM_BATCHED_MAP"}
RANK_VALUES = {
    "NDIM_MATRIX": 2,
    "NDIM_BARE_MAP": 2,
    "NDIM_BATCH_MAP": 3,
    "NDIM_BATCHED_MAP": 4,
}


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

    def test_each_rank_name_is_defined_only_in_config(self):
        owners: dict[str, list[str]] = {name: [] for name in RANK_NAMES}
        for module in sorted(PKG.glob("*.py")):
            for name in _module_defs(module):
                owners[name].append(module.stem)
        for name, modules in owners.items():
            assert modules == ["config"], f"{name} defined in {modules}; expected config only"

    def test_rank_values_match_the_documented_axes(self):
        assert _module_defs(PKG / "config.py") == RANK_VALUES

    def test_consumers_import_ranks_instead_of_redeclaring(self):
        for consumer in ("optics.py", "losses.py"):
            path = PKG / consumer
            assert _module_defs(path) == {}, f"{consumer} re-declares rank constants"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            used = {n.id for n in ast.walk(tree)
                    if isinstance(n, ast.Name) and n.id in RANK_NAMES}
            assert used, f"{consumer} references no rank constant at all"
            imported = _imported_names(tree, "ml.sim2f.config")
            assert used <= imported, (
                f"{consumer} uses {sorted(used - imported)} without importing them")
