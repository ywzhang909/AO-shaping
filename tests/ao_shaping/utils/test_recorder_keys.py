"""The ``Recorder`` record-dict key names are an on-disk contract with one source.

Three separate guarantees, each catching a different way the contract breaks:

1. **Value pins.** The constants' strings are what column names land on disk in the
   pickled history and the sidecar HDF5/CSV that ``report/`` generators and
   ``ml/hwdataset`` read back. A rename here silently orphans every previously
   saved run, so the expected strings are written out literally in this file --
   deliberately *not* derived from the module, which would make the test tautological.
2. **No second vocabulary.** Until 2026-10 one writer used bare literals while
   ``ml/hwdataset`` carried three independent private copies of ``_phase`` / ``_c``
   / ``_img`` (``index.py``, ``records.py``, ``zernike_dataset.py``). Two spellings of
   one format is the "duplicate implementation drifts" failure this repo keeps getting
   bitten by, so a module-level string constant equal to a recorder key is rejected
   anywhere outside the owning module.
3. **The writer uses the constant,** not a stale literal beside it.

Trade-off, stated rather than hidden: guarantee 2 keys on the *value*, so a future
module that legitimately needs an unrelated ``"_c"`` constant will trip it and must
either reference the shared name or carry an explicit reason. With short generic keys
that is the cheaper error, because the fix is one line and the alternative is a
split vocabulary that only shows up as mismatched columns weeks later.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ao_shaping.utils.io import recorder_keys
from ao_shaping.utils.io.file import Recorder
from ao_shaping.utils.io.recorder_keys import RECORD_ID, RECORD_KEYS

_SRC = Path(__file__).resolve().parents[3] / "src"
_OWNER = (_SRC / "ao_shaping" / "utils" / "io" / "recorder_keys.py").resolve()

#: Byte-exact. Editing a value in the module must fail here, not in a report.
_EXPECTED = {
    "RECORD_ID": "_id",
    "RECORD_EPOCH": "_epoch",
    "RECORD_IMAGE": "_img",
    "RECORD_COEFFS": "_c",
    "RECORD_VOLTAGES": "_v",
    "RECORD_PHASE": "_phase",
    "RECORD_GRAD": "_grad",
    "RECORD_DIFF": "_diff",
    "RECORD_GATE": "_gate",
    "RECORD_RADIUS": "_r",
    "RECORD_ROI_ENERGY": "_p%",
}


def _module_level_key_literals(root: Path) -> list[str]:
    """``file:line: NAME = '...'`` for module-level constants equal to a recorder key."""
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.resolve() == _OWNER:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            else:
                continue
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                continue
            if value.value not in set(RECORD_KEYS):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    offenders.append(
                        f"{path.relative_to(root)}:{node.lineno}: "
                        f"{target.id} = {value.value!r}"
                    )
    return offenders


def test_key_values_are_pinned_byte_for_byte() -> None:
    actual = {name: getattr(recorder_keys, name) for name in _EXPECTED}
    assert actual == _EXPECTED, (
        f"a recorder key value changed; saved runs stop matching. "
        f"diff: { {k: (actual[k], v) for k, v in _EXPECTED.items() if actual[k] != v} }"
    )


def test_every_declared_key_is_in_the_registry() -> None:
    """A constant left out of ``RECORD_KEYS`` would escape every key-set consumer."""
    assert set(RECORD_KEYS) == set(_EXPECTED.values())
    assert len(RECORD_KEYS) == len(set(RECORD_KEYS)), "duplicate key in RECORD_KEYS"


def test_no_module_re_declares_a_recorder_key() -> None:
    offenders = _module_level_key_literals(_SRC)
    assert not offenders, (
        "these spell a recorder key as their own module-level constant instead of "
        "importing it from ao_shaping.utils.io.recorder_keys:\n  "
        + "\n  ".join(offenders)
    )


def test_the_redeclaration_scan_actually_bites(tmp_path: Path) -> None:
    """Guard against a vacuous scan: a guard that finds nothing is worse than none."""
    planted = tmp_path / "pkg" / "consumer.py"
    planted.parent.mkdir(parents=True)
    planted.write_text('_IMAGE_KEY = "_img"\nOK_KEY = "not_a_recorder_key"\n', encoding="utf-8")

    offenders = _module_level_key_literals(tmp_path)
    assert offenders == ["pkg/consumer.py:1: _IMAGE_KEY = '_img'"], (
        f"scan missed the planted re-declaration: {offenders}"
    )


def test_writer_populates_the_pinned_id_key() -> None:
    """``Recorder.append`` must fill the constant, not a literal that can drift."""
    recorder = Recorder(mark="J")
    recorder.append({"J": 1.0})
    assert recorder.history[0][RECORD_ID] == 0
    assert RECORD_ID == "_id"

    # An explicit id is respected rather than overwritten.
    recorder.append({RECORD_ID: 7, "J": 2.0})
    assert recorder.history[1][RECORD_ID] == 7


def test_recorder_keys_is_a_zero_import_leaf() -> None:
    """The owning module may import nothing, so any layer can use it without a cycle."""
    tree = ast.parse(_OWNER.read_text(encoding="utf-8-sig"))
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
    ]
    assert not imports, f"recorder_keys gained imports: {ast.dump(imports[0])[:120]}"


@pytest.mark.parametrize("key", sorted(RECORD_KEYS))
def test_keys_are_recorder_prefixed_or_plain_names(key: str) -> None:
    assert key.startswith("_") or key == "J", f"unexpected key shape: {key!r}"
