"""Guard test: the `tools/slm` package docstring must catalogue the package.

The package docstring (`src/ao_shaping/tools/slm/__init__.py`) is the index of the
package. After the 2026-10-12 reorganisation the package holds only SLM-specific
pieces: the ``panel/`` and ``calibration/`` subpackages, the ``cartographer/``
subpackage and the ``params`` module. The shared kernels / Zernike / WFS / CCD
probes moved to sibling packages (``tools/kernels`` etc.) and are documented by
*their* own package docstrings.

The guard is deliberately loose: it only checks that each name appears somewhere
in the docstring — not a bullet, indentation, phrasing or ordering.
"""

from __future__ import annotations

from pathlib import Path

import ao_shaping.tools.slm as slm_tools

PACKAGE_DIR: Path = Path(slm_tools.__file__).parent
DOCSTRING: str = slm_tools.__doc__ or ""

#: Subpackages the SLM package must still document.
_SUBPACKAGES: tuple[str, ...] = ("panel", "calibration", "cartographer")


def _module_stems() -> list[str]:
    """Return the stems of top-level modules in the package, excluding ``__init__``."""
    return sorted(
        path.stem for path in PACKAGE_DIR.glob("*.py") if path.stem != "__init__"
    )


def test_every_top_level_module_is_catalogueued() -> None:
    missing = [stem for stem in _module_stems() if stem not in DOCSTRING]
    assert not missing, (
        "tools/slm package docstring is missing an entry for: "
        f"{missing}. Add a bullet to the catalogue block in "
        "src/ao_shaping/tools/slm/__init__.py."
    )


def test_every_subpackage_is_catalogueued() -> None:
    missing = [name for name in _SUBPACKAGES if name not in DOCSTRING]
    assert not missing, f"tools/slm docstring does not name subpackage(s): {missing}"


def test_enumeration_finds_modules() -> None:
    """Sanity guard: the glob must actually find the package's modules."""
    stems = _module_stems()
    assert stems, f"no modules found in {PACKAGE_DIR}"
    assert "params" in stems, "params.py is not on disk"


def test_docstring_is_not_empty() -> None:
    """Sanity guard so the assertions above cannot pass on an empty docstring."""
    assert DOCSTRING.strip(), "package docstring is empty"
