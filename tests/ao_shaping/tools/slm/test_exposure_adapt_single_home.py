"""The exposure ladder lives in one module; ``focus_probe`` only re-exports it.

``adapt_exposure`` / ``frame_full_scale`` / ``robust_peak`` and the accepted-band
constants grew inside :mod:`ao_shaping.tools.slm.panel.focus_probe`, where the
focus sweep was the only user. Four other probe families now import the ladder
from there, so it moved to its own module (that file was 1139 lines against a
1097 ratchet baseline).

Two things are pinned here, because both rot silently:

* the names ``focus_probe`` exports must be **the same objects** as the ones in
  ``exposure_adapt`` -- a re-export that is a copy would let a caller and a
  consumer disagree about the accepted band with no error anywhere;
* ``exposure_adapt`` must not import ``focus_probe``. The reverse direction is
  the one that exists, so an import the other way is a cycle that only shows up
  when someone imports the leaf first.
"""
from __future__ import annotations

import ast

import pytest

from ao_shaping.tools.slm.panel import exposure_adapt, focus_probe

#: Names ``focus_probe`` has always exported for the ladder.
REEXPORTED = (
    "ACCEPT_HIGH",
    "ACCEPT_LOW",
    "TARGET_FRACTION",
    "ExposureSetting",
    "ExposureStep",
    "adapt_exposure",
    "frame_full_scale",
    "robust_peak",
)


@pytest.mark.parametrize("name", REEXPORTED)
def test_focus_probe_reexports_the_same_object(name: str) -> None:
    assert getattr(focus_probe, name) is getattr(exposure_adapt, name), (
        f"focus_probe.{name} is not exposure_adapt.{name}: a copy, not a re-export"
    )


def test_the_ladder_does_not_import_the_focus_sweep() -> None:
    tree = ast.parse(
        open(exposure_adapt.__file__, encoding="utf-8-sig").read()
    )
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    offenders = [m for m in imported if "focus_probe" in m]
    assert not offenders, (
        f"exposure_adapt imports {offenders}; the dependency runs the other way "
        "(focus_probe imports exposure_adapt), so this is a cycle"
    )
