"""``ao_shaping.tools.slm`` / leaf-package facade contracts (post-2026-10-12).

Locked invariants:

1. **No eager re-export at package level.** The submodules import ``click`` /
   ``loguru`` / numpy / the SLM and camera drivers at module scope, so a
   package-level re-export would make *every* ``import ao_shaping.tools.slm.<x>``
   pay for all of them. ``tools/slm/__init__.py`` (and the shared leaf packages
   ``kernels`` / ``zernike`` / ``ccd`` / ``wfs``) must contain **no** module-scope
   ``import`` / ``from ... import`` statement.
2. **``__init__.py`` must exist.** A directory without one is silently dropped by
   ``find_packages()``-style packaging.

The 2026-10-12 reorganisation moved the shared measurement kernels and the
WFS / CCD / Zernike probes out of ``tools/slm/`` into sibling packages; this file
covers those siblings too. ``slm/calibration/__init__.py`` is the one package
that *deliberately* re-exports (its public calibrator API), so it is exempt from
the no-eager-import rule.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[4] / "src" / "ao_shaping" / "tools"
_SLM_PKG = _PKG_ROOT / "slm"
_MICRO_DM_PKG = _PKG_ROOT / "micro_dm"

#: Sibling packages carved out of ``tools/slm/`` — all must stay import-free at
#: package level so ``import ao_shaping.tools.<x>.<y>`` stays cheap.
_LEAF_PACKAGES = ("kernels", "zernike", "ccd", "wfs")


def _top_level_imports(path: Path) -> list[ast.Import | ast.ImportFrom]:
    """Return module-scope ``import`` / ``from ... import`` nodes.

    ``from __future__ import annotations`` is exempt: it is a compiler directive,
    not a runtime import -- nothing is loaded, so it costs an ``import
    ao_shaping.tools.<x>.<y>`` nothing. Every module in this repo is required to
    carry it (README: "所有模块文件第一行应包含此导入"), so counting it would fail
    the lazy-import contract for every compliant package at once.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
    ]


class TestToolsSlmPackage:
    def test_init_exists(self) -> None:
        assert (_SLM_PKG / "__init__.py").is_file()

    def test_init_has_no_eager_imports(self) -> None:
        assert _top_level_imports(_SLM_PKG / "__init__.py") == []

    @pytest.mark.parametrize("module_name", sorted(p.name for p in _SLM_PKG.glob("*.py")))
    def test_no_facade_star_import_anywhere(self, module_name: str) -> None:
        """``from ao_shaping.tools.slm import *`` would rely on a facade surface."""
        source = (_SLM_PKG / module_name).read_text(encoding="utf-8")
        assert "from ao_shaping.tools.slm import *" not in source


class TestLeafPackages:
    @pytest.mark.parametrize("sub", _LEAF_PACKAGES)
    def test_init_exists_and_has_no_eager_imports(self, sub: str) -> None:
        init = _PKG_ROOT / sub / "__init__.py"
        assert init.is_file(), f"{sub}/__init__.py is missing"
        assert _top_level_imports(init) == [], f"{sub} does eager re-export"

    @pytest.mark.parametrize("sub", ("slm/panel",))
    def test_subpackage_init_exists(self, sub: str) -> None:
        assert (_PKG_ROOT / sub / "__init__.py").is_file()

    def test_calibration_package_reexports_its_api(self) -> None:
        """The calibration package is the one allowed to re-export its public API."""
        import ao_shaping.tools.slm.calibration as cal

        for name in ("SantecCalibrator", "SLMCCDCalibrator", "CalibrationMethod"):
            assert hasattr(cal, name), f"calibration package lost {name}"


class TestToolsMicroDmPackage:
    def test_init_exists(self) -> None:
        assert (_MICRO_DM_PKG / "__init__.py").is_file()

    def test_init_has_no_eager_imports(self) -> None:
        init = _MICRO_DM_PKG / "__init__.py"
        assert _top_level_imports(init) == []

    def test_submodule_is_importable(self) -> None:
        """Regression for the 12 docstrings that still said the pre-move path."""
        import importlib

        mod = importlib.import_module(
            "ao_shaping.tools.micro_dm.micro_dm_image_collect"
        )
        assert hasattr(mod, "main")


class TestMicroDmDocstringsAreCurrent:
    def test_no_stale_module_path_in_docstrings(self) -> None:
        source = (_MICRO_DM_PKG / "micro_dm_image_collect.py").read_text(encoding="utf-8")
        assert "ao_shaping.tools.micro_dm_image_collect" not in source
        assert "python -m ao_shaping.tools.micro_dm.micro_dm_image_collect" in source
