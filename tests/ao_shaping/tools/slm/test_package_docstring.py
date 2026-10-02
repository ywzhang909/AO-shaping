"""Guard test: the `tools/slm` package docstring must catalogue every module.

The package docstring (`src/ao_shaping/tools/slm/__init__.py`) is the index of the
package: it lists each top-level module with a one-line Chinese description. That
index drifted silently before (six modules were missing), so this test pins the
invariant "every module on disk is named in the docstring".

Deliberately loose: it only checks that the module *stem* appears somewhere in the
docstring. It does not require a bullet, a particular indentation, a particular
`— description` phrasing, or any ordering — prose style is not what this test is
protecting.
"""

from __future__ import annotations

from pathlib import Path

import ao_shaping.tools.slm as slm_tools

PACKAGE_DIR: Path = Path(slm_tools.__file__).parent
DOCSTRING: str = slm_tools.__doc__ or ""

# Modules that were missing from the catalogue when this guard was written
# (2026-10-01). Asserted by name so the test documents what it protects.
NEWLY_DOCUMENTED: tuple[str, ...] = (
    "slm_zernike_response",
    "slm_zernike_correction",
    "slm_zernike_common",
    "slm_wfs_probe",
    "slm_wfs_reference",
    "delta_explorer",
)


def _module_stems() -> list[str]:
    """Return the stems of top-level modules in the package, excluding `__init__`.

    Top level only: the `cartographer/` subpackage is documented as a single
    `cartographer/` entry, so its files must not be demanded individually.
    """
    return sorted(
        path.stem for path in PACKAGE_DIR.glob("*.py") if path.stem != "__init__"
    )


def test_every_module_is_catalogueued() -> None:
    """Every top-level module stem must appear in the package docstring."""
    missing = [stem for stem in _module_stems() if stem not in DOCSTRING]
    assert not missing, (
        "modules/slm package docstring is missing an entry for: "
        f"{missing}. Add a `- <module>.py — <description>` bullet to the "
        "module-catalogue block in src/ao_shaping/tools/slm/__init__.py."
    )


def test_newly_documented_modules_are_named() -> None:
    """The six previously-missing modules must be named in the docstring."""
    missing = [stem for stem in NEWLY_DOCUMENTED if stem not in DOCSTRING]
    assert not missing, f"docstring regressed: missing {missing}"


def test_enumeration_finds_modules() -> None:
    """Sanity guard: the glob must actually find the package's modules."""
    stems = _module_stems()
    assert stems, f"no modules found in {PACKAGE_DIR}"
    for stem in NEWLY_DOCUMENTED:
        assert stem in stems, f"{stem}.py is not on disk; update NEWLY_DOCUMENTED"


def test_docstring_is_not_empty() -> None:
    """Sanity guard so the assertions above cannot pass on an empty docstring."""
    assert DOCSTRING.strip(), "package docstring is empty"
