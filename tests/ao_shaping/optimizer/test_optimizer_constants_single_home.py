"""R-69 P0 guard: the four optimizer-wide constants must have exactly one home.

``optimizer/constants.py`` was created to centralise the optimizer factory and
``OPTIMIZER_MAP`` (issue #46). It also holds four values that the sensorless
optimizers share: the camera probe frame count, the auto-exposure target
brightness, the ideal spot radius, and the DM exit-voltage policy.

Three modules under ``optimizer/`` still declared their own copies after that
commit, so the same name was spelled twice with the same value in two places --
the "two spellings of one thing drift" failure this repo keeps getting bitten
by. A divergent *value* is the dangerous case: the DM exit policy in
``tools/train_data_collect.py`` is deliberately ``False`` (that tool wants the
DM zeroed on exit), which is why it is **not** folded into the canonical
``True`` -- folding it would silently change hardware behaviour.

This test fails if a module-level copy of any of the four comes back inside the
optimizer package; the canonical home is ``optimizer/constants.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OPTIMIZER_ROOT = REPO_ROOT / "src" / "ao_shaping" / "optimizer"
CANONICAL = OPTIMIZER_ROOT / "constants.py"

#: Name -> value the canonical module pins. A value change would silently alter
#: DM exit behaviour or the auto-exposure target.
CANONICAL_VALUES = {
    "CAM_SAMPLE_ITER": "1",
    "TEST_EXPOSURE_TIME_BRIGHTNESS": "220",
    "IDEAL_SPOT_RADIUS": 'int(os.environ.get("IDEAL_SPOT_RADIUS", "6"))',
    "KEEP_VOLTAGE_WHEN_EXIT": "True",
}


def _module_assign_names(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _declared_values(path: Path) -> dict[str, ast.expr]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
        return {}
    out: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
            value = node.value
        else:
            continue
        for name in targets:
            out[name] = value
    return out


def _imported_from_canonical(path: Path) -> set[str]:
    """Names this module takes from ``optimizer/constants.py``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "ao_shaping.optimizer.constants"
        ):
            names |= {alias.name for alias in node.names}
    return names


def test_canonical_constants_have_one_home() -> None:
    offenders: list[str] = []
    for path in sorted(OPTIMIZER_ROOT.rglob("*.py")):
        if path == CANONICAL:
            continue
        redeclared = _module_assign_names(path) & set(CANONICAL_VALUES)
        if redeclared:
            offenders.append(
                f"{path.relative_to(REPO_ROOT).as_posix()}: {sorted(redeclared)}"
            )
    assert not offenders, (
        "these re-declare a constant that belongs to "
        "ao_shaping/optimizer/constants.py -- import it from there instead:\n  "
        + "\n  ".join(offenders)
    )


def test_canonical_module_defines_all_four() -> None:
    declared = _declared_values(CANONICAL)
    missing = sorted(set(CANONICAL_VALUES) - set(declared))
    assert not missing, f"{missing} missing from optimizer/constants.py"
    for name, expected in CANONICAL_VALUES.items():
        # Compare the parsed trees: quote style is not part of the value.
        assert ast.dump(declared[name]) == ast.dump(
            ast.parse(expected, mode="eval").body
        ), f"{name} changed in optimizer/constants.py: expected {expected}"


def test_the_three_migrated_modules_take_their_values_from_the_canonical_home() -> None:
    """Provenance, not just the absence of a local copy.

    A module that neither declares nor imports a name cannot be using it, so
    this pins the migration itself rather than only its absence.
    """
    expected = {
        "slm_zernike_pib.py": {
            "CAM_SAMPLE_ITER",
            "IDEAL_SPOT_RADIUS",
            "TEST_EXPOSURE_TIME_BRIGHTNESS",
        },
        "slm_square_shaping.py": {"CAM_SAMPLE_ITER", "TEST_EXPOSURE_TIME_BRIGHTNESS"},
        "gready_cam.py": {"KEEP_VOLTAGE_WHEN_EXIT"},
    }
    for filename, names in expected.items():
        path = OPTIMIZER_ROOT / "wfless" / filename
        imported = _imported_from_canonical(path)
        assert names <= imported, (
            f"{filename} must take {sorted(names)} from "
            f"ao_shaping.optimizer.constants; it imports {sorted(imported)}"
        )


def test_the_scan_actually_bites(tmp_path: Path) -> None:
    """A guard that finds nothing is worse than none -- prove it can fail."""
    planted = tmp_path / "pkg" / "consumer.py"
    planted.parent.mkdir(parents=True)
    planted.write_text("CAM_SAMPLE_ITER = 1\n", encoding="utf-8")

    assert _module_assign_names(planted) & set(CANONICAL_VALUES) == {
        "CAM_SAMPLE_ITER"
    }
