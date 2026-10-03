"""Subpackage anchor tests for :mod:`ao_shaping.utils.image`.

Two things are pinned here:

1. Modules moved into the ``image`` subpackage stay importable from their
   **canonical** subpackage path.
2. The ``ao_shaping.utils.hardware_utils`` **alias is gone** (TODO R-22).

Point 2 used to be the opposite: the module carried a 16-line
``sys.modules[__name__] = ...`` shim at ``utils/hardware_utils.py``, kept alive by
the claim that "tests reset ``_frames_dir`` / ``_frame_counter`` via the legacy
name". That reasoning dissolved once every importer moved to the canonical path —
importing ``ao_shaping.utils.image.hardware_utils`` binds the *same* module
object, so globals mutated through it are still the real ones. Keeping a second
name for one module is exactly the "two homes" hazard the repo already forbids
elsewhere (see the ``ZernikeDM.generate_phase`` / ``PatternHelper._zernike_to_uint16``
anti-pattern rows).

Deliberately **no** compatibility shim: the old path must raise.
"""

from __future__ import annotations

import importlib

import pytest

CANONICAL = "ao_shaping.utils.image.hardware_utils"
LEGACY = "ao_shaping.utils.hardware_utils"


def test_canonical_module_imports() -> None:
    module = importlib.import_module(CANONICAL)
    assert module.__name__ == CANONICAL
    assert hasattr(module, "open_camera")
    assert hasattr(module, "init_frame_recording")


def test_legacy_alias_path_is_gone() -> None:
    """No shim: the second name for this module must not exist."""
    with pytest.raises(ImportError):
        importlib.import_module(LEGACY)


def test_the_alias_module_file_is_deleted() -> None:
    """The physical file must be gone too, not just unreachable by import."""
    from pathlib import Path

    import ao_shaping.utils as utils_pkg

    old = Path(utils_pkg.__file__).resolve().parent / "hardware_utils.py"
    assert not old.exists(), f"{old} still exists; delete it, do not leave a dead file"


def test_no_source_file_imports_the_legacy_path() -> None:
    """Static guard against the alias creeping back into a call site."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "src"
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "ao_shaping.utils.hardware_utils" in text:
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f"legacy import path still referenced in: {offenders}"