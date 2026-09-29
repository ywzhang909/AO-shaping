"""Structure, shim-fidelity and leaf-layer guards for the split ``target`` package.

The former flat ``ao_shaping.utils.image.targets`` module was split by type into
``ao_shaping.utils.image.target.{patterns,metrics,square,ccd,objective}`` and
``targets`` became a backward-compatible shim. These tests pin:

* every submodule exports its expected symbols;
* the package re-exports the full former surface;
* the ``targets`` shim exposes the *same objects* (identity, not copies), so
  ``from ao_shaping.utils.image.targets import X`` cannot silently diverge;
* the package stays a leaf (no torch / algorithm / drivers / optimizer / runners
  imports), matching the module's documented contract.

Behaviour itself is covered by ``test_targets.py`` / ``test_shaping_objective.py``
(which import through the shim and therefore also exercise the shim).
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

import ao_shaping.utils.image.target as target_pkg
import ao_shaping.utils.image.targets as legacy_shim

PACKAGE_DIR = Path(target_pkg.__file__).resolve().parent

# module -> symbols that MUST be defined in that submodule
EXPECTED_EXPORTS: dict[str, tuple[str, ...]] = {
    "patterns": ("create_target_shape", "create_target_mask", "load_target_image"),
    "metrics": (
        "TARGET_SHAPE_CHOICES",
        "TargetShape",
        "target_shape_roi",
        "spot_waist_sigma",
        "roi_energy_loss",
        "roi_pib_metric",
        "rms_pib_terms",
        "rmse_shape_metric",
        "SHAPE_STAGE_WEIGHTS",
        "shape_stage",
        "shape_stage_from_energy",
        "shape_metric",
    ),
    "square": ("compute_square_side", "build_square_target_amplitude"),
    "ccd": (
        "crop_resize_to_grid",
        "square_target_from_measurement",
        "generate_target_mask",
        "build_ccd_target",
        "build_target_from_frame",
        "_resize_bilinear",
        "_rectangle_mask",
        "_SHAPE_ALIASES",
    ),
    "objective": (
        "ShapeScoringParams",
        "ShapingObjectiveParams",
        "ObjectiveResult",
        "SHAPING_OBJECTIVE_CHOICES",
        "GUARDED_OBJECTIVES",
        "ShapingObjective",
        "_update_dynamic_weights",
        "_resolve_init_weights",
    ),
}

ALL_EXPECTED = tuple(
    name for names in EXPECTED_EXPORTS.values() for name in names
)


class TestSubmoduleStructure:
    @pytest.mark.parametrize("module", sorted(EXPECTED_EXPORTS))
    def test_submodule_is_importable(self, module: str) -> None:
        importlib.import_module(f"ao_shaping.utils.image.target.{module}")

    @pytest.mark.parametrize("module", sorted(EXPECTED_EXPORTS))
    def test_submodule_defines_its_symbols(self, module: str) -> None:
        mod = importlib.import_module(f"ao_shaping.utils.image.target.{module}")
        for name in EXPECTED_EXPORTS[module]:
            assert hasattr(mod, name), f"{module} is missing {name}"

class TestPackageReExports:
    @pytest.mark.parametrize("name", ALL_EXPECTED)
    def test_package_exposes_symbol(self, name: str) -> None:
        assert hasattr(target_pkg, name), f"target package is missing {name}"

    def test_dunder_all_covers_the_full_surface(self) -> None:
        assert set(ALL_EXPECTED) <= set(target_pkg.__all__)

    def test_dunder_all_has_no_missing_attributes(self) -> None:
        for name in target_pkg.__all__:
            assert hasattr(target_pkg, name), f"__all__ lists absent name {name}"


class TestLegacyShim:
    """``targets`` must re-export the SAME objects as the package - no copies."""

    @pytest.mark.parametrize("name", ALL_EXPECTED)
    def test_shim_symbol_is_identical_object(self, name: str) -> None:
        assert getattr(legacy_shim, name) is getattr(target_pkg, name), (
            f"targets.{name} is not target.{name}"
        )

    def test_shim_exposes_the_same_all(self) -> None:
        assert tuple(legacy_shim.__all__) == tuple(target_pkg.__all__)

    def test_shim_keeps_private_weight_helpers(self) -> None:
        # Existing tests / the PIB optimizer import these from ``targets``.
        assert legacy_shim._update_dynamic_weights is target_pkg._update_dynamic_weights
        assert legacy_shim._resolve_init_weights is target_pkg._resolve_init_weights


class TestLeafLayer:
    """The package must stay a pure-NumPy leaf (no torch / higher packages)."""

    FORBIDDEN_PREFIXES = (
        "torch",
        "ao_shaping.algorithm",
        "ao_shaping.drivers",
        "ao_shaping.optimizer",
        "ao_shaping.runners",
    )

    def _imported_modules(self, path: Path) -> list[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        mods: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                mods.append(node.module or "")
        return mods

    def test_package_modules_have_no_forbidden_imports(self) -> None:
        modules = sorted(PACKAGE_DIR.glob("*.py"))
        assert modules, "no package modules found"
        for path in modules:
            for mod in self._imported_modules(path):
                assert not mod.startswith(self.FORBIDDEN_PREFIXES), (
                    f"{path.name} imports forbidden module {mod!r}"
                )

    def test_shim_has_no_forbidden_imports(self) -> None:
        for mod in self._imported_modules(
            PACKAGE_DIR.parent / "targets.py"
        ):
            if mod.startswith("ao_shaping.utils.image.target"):
                continue
            assert not mod.startswith(self.FORBIDDEN_PREFIXES), (
                f"targets.py shim imports forbidden module {mod!r}"
            )


class TestBehaviourThroughBothPaths:
    """A couple of hand-computed smoke checks, via package and via shim."""

    def test_create_target_shape_square_same_both_paths(self) -> None:
        for mod in (target_pkg, legacy_shim):
            out = mod.create_target_shape("square", 7, side=3)
            assert out.shape == (7, 7)
            assert out.sum() == 9.0

    def test_target_shape_roi_same_both_paths(self) -> None:
        for mod in (target_pkg, legacy_shim):
            roi = mod.target_shape_roi((10, 10), (5, 5), "square", 3)
            assert roi.dtype == bool
            assert roi.sum() == 9

    def test_tracking_choices_identical(self) -> None:
        assert target_pkg.TARGET_SHAPE_CHOICES == legacy_shim.TARGET_SHAPE_CHOICES
        assert set(target_pkg.TARGET_SHAPE_CHOICES) == {
            "circle",
            "square",
            "rectangle",
            "annular",
            "grid",
            "cross",
            "gaussian",
            "pentagon",
        }

    def test_square_amplitude_both_paths(self) -> None:
        expected = np.zeros((6, 6))
        expected[1:5, 1:5] = 1.0
        for mod in (target_pkg, legacy_shim):
            out = mod.build_square_target_amplitude(6, 6, 4)
            assert np.array_equal(out, expected)
