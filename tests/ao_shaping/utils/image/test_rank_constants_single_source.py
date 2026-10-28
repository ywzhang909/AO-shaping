"""``NDIM_*`` in ``ao_shaping`` may have exactly one definition site.

Item 120 moved the array-rank conventions of this package into :mod:`ranks` because
``beam_metrics``, ``target.metrics`` and ``exposure_fusion`` were all spelling the same shape
guard with a bare literal. A per-module re-declaration would let those guards drift apart --
one helper rejecting a rank its neighbour accepts is a silent behaviour difference, not a
style one -- so the single-source property is asserted here rather than left to review.

The check is AST-only: it parses source instead of importing it, so it runs without numpy or
a device and cannot be skipped for an environmental reason.

The scan covers the whole ``ao_shaping`` package, not just ``utils.image``: the ranks are a
package-wide convention now that ``algorithm.signal_processing.differentiable_shaping``
consumes them, and a scan limited to the hub's own folder would let a second definition
appear one package over and read as a pass (the same widening as item 122 did for ``ml``).
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[4] / "src" / "ao_shaping"
HUB = "utils/image/ranks.py"
RANK_VALUES = {"NDIM_SINGLE_CHANNEL": 2, "NDIM_SPATIAL_STACK": 3}
CONSUMERS = (
    "utils/image/exposure_fusion.py",
    "utils/image/beam_metrics.py",
    "utils/image/target/metrics.py",
    "algorithm/signal_processing/differentiable_shaping.py",
    "algorithm/signal_processing/differentiable_defocus_uniformity.py",
)
HUB_MODULE = "ao_shaping.utils.image.ranks"


def _rank_defs(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("NDIM_") and isinstance(node.value, ast.Constant)
    }


def _imported_from(tree: ast.AST, module: str) -> set[str]:
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


class TestRankConstantsSingleSource:
    def test_package_directory_is_reachable(self):
        assert PKG.is_dir(), f"expected package directory {PKG}"

    def test_rank_names_are_defined_only_in_the_hub(self):
        owners: dict[str, list[str]] = {}
        for module in sorted(PKG.rglob("*.py")):
            for name in _rank_defs(module):
                owners.setdefault(name, []).append(module.relative_to(PKG).as_posix())
        assert set(owners) >= set(RANK_VALUES), f"hub ranks missing: {sorted(RANK_VALUES - set(owners))}"
        for name, modules in sorted(owners.items()):
            assert modules == [HUB], f"{name} defined in {modules}; expected {HUB} only"

    def test_rank_values_match_the_documented_axes(self):
        assert _rank_defs(PKG / HUB) == RANK_VALUES

    def test_consumers_import_ranks_instead_of_redeclaring(self):
        for consumer in CONSUMERS:
            path = PKG / consumer
            assert path.is_file(), f"missing consumer {consumer}"
            assert _rank_defs(path) == {}, f"{consumer} re-declares a rank constant"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            used = {
                n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and n.id.startswith("NDIM_")
            }
            assert used, f"{consumer} references no rank constant at all"
            imported = _imported_from(tree, HUB_MODULE)
            assert used <= imported, (
                f"{consumer} uses {sorted(used - imported)} without importing them from "
                f"{HUB_MODULE}")
