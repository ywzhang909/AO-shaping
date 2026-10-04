"""Tests for the measured Zernike base-matrix data set (R-34).

The 66 measured 360x360 maps used to ship as ASCII ``.txt`` files that the
loader walked with ``Path.glob("*.txt")``. That is **lexicographic** order, so
``1, 10, 11, ..., 66, 7, 8, 9`` -- and because
:meth:`ZernikeCentroidCalculator.get_centroid` is a *positional* weighted sum,
65 of the 66 slots held the wrong map. A pure tilt-x command (``coef[1] = 1``)
therefore reported a centroid sitting on the array centre instead of shifting it
~35 px.

These tests pin the three things that keep that from coming back: the stored mode
numbers must be ``1..N`` in order, the physical identity of the low modes must
match the canonical Zernike order, and the shipped data must round-trip bit-exactly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.utils.wavefront import wavefront_calc as wc
from ao_shaping.utils.wavefront.wavefront_calc import (
    ZernikeCentroidCalculator,
    get_zernike_base_matrixs,
)

SIDE = 360
N_MODES = 66
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "tuning_devices" / "dm_unit_compute.py"
PACKAGE_INIT = ROOT / "scripts" / "tuning_devices" / "__init__.py"


def _ramp_axis(matrix: np.ndarray) -> str:
    """Classify a map by which axis it ramps along ("x", "y" or "sym")."""
    dx = float(np.abs(np.diff(matrix, axis=1)).mean())
    dy = float(np.abs(np.diff(matrix, axis=0)).mean())
    if dx > 2 * dy:
        return "x"
    if dy > 2 * dx:
        return "y"
    return "sym"


# --- the data file ---------------------------------------------------------
def test_loader_returns_the_full_mode_stack():
    matrices = get_zernike_base_matrixs()
    assert matrices.shape == (N_MODES, SIDE, SIDE)
    assert matrices.dtype == np.float64
    assert np.isfinite(matrices).all()


def test_stored_mode_numbers_are_one_to_n_in_order():
    """The data-level guard against a reordered stack.

    ``get_centroid`` multiplies ``matrices[i]`` by ``coef[i]`` positionally, so
    a permuted file would silently answer with wrong centroids rather than
    raising. ``get_zernike_base_matrixs`` validates this on every load.
    """
    with np.load(wc._STD_WAVEFRONT_PATH) as payload:
        modes = payload["modes"]
    assert np.array_equal(modes, np.arange(1, N_MODES + 1))


def test_low_modes_match_the_canonical_zernike_order():
    """mode 1 = piston, 2 = tilt x, 3 = tilt y, 4 = defocus.

    This is the direct regression test for the lexicographic-glob bug: with the
    old ordering, index 1 held ``10.txt`` (a rotationally symmetric mode), so the
    "x" assertion below failed.
    """
    matrices = get_zernike_base_matrixs()

    piston = matrices[0]
    assert np.unique(piston).size == 2, "mode 1 should be a binary piston mask"

    assert _ramp_axis(matrices[1]) == "x", "mode 2 should be the x tilt"
    assert _ramp_axis(matrices[2]) == "y", "mode 3 should be the y tilt"
    # defocus / astigmatism are rotationally symmetric about the array centre
    assert _ramp_axis(matrices[3]) == "sym"
    assert _ramp_axis(matrices[4]) == "sym"


def test_all_modes_are_distinct():
    matrices = get_zernike_base_matrixs()
    flat = matrices.reshape(N_MODES, -1)
    assert np.unique(flat, axis=0).shape[0] == N_MODES


# --- the physical consequence ---------------------------------------------
def test_pure_tilt_x_moves_the_centroid_in_x_only():
    """End-to-end proof the ordering is right, with the pre-fix value quoted.

    Before the fix this returned (179.500, 179.496) -- i.e. no shift at all,
    because a symmetric mode was standing in for the tilt.
    """
    calc = ZernikeCentroidCalculator()
    coef = np.zeros(N_MODES)
    coef[1] = 1.0  # mode 2 = tilt x
    (dx, dy), _img = calc.get_centroid(coef)

    assert dx == pytest.approx(144.291, abs=0.01)
    assert dy == pytest.approx(179.5, abs=0.01)
    # ~35.7 px left of the array centre, in mm
    assert calc.center_coordinate(dx, dy)[0] == pytest.approx(-35.71, abs=0.05)
    assert calc.center_coordinate(dx, dy)[1] == pytest.approx(-0.5, abs=0.05)


def test_pure_tilt_y_moves_the_centroid_in_y_only():
    calc = ZernikeCentroidCalculator()
    coef = np.zeros(N_MODES)
    coef[2] = 1.0  # mode 3 = tilt y
    (dx, dy), _img = calc.get_centroid(coef)

    assert dx == pytest.approx(179.5, abs=0.5)
    assert dy != pytest.approx(179.5, abs=1.0), "the y tilt must not stay centred"


# --- packaging / path anchoring -------------------------------------------
def test_default_path_is_absolute_and_exists():
    assert wc._STD_WAVEFRONT_PATH.is_absolute()
    assert wc._STD_WAVEFRONT_PATH.is_file()
    assert wc._STD_WAVEFRONT_PATH.suffix == ".npz"


def test_default_load_works_from_any_cwd(tmp_path, monkeypatch):
    """The old default was the relative string "scripts/tuning_devices/..."."""
    monkeypatch.chdir(tmp_path)
    assert get_zernike_base_matrixs().shape == (N_MODES, SIDE, SIDE)


def test_missing_file_raises_instead_of_falling_back(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        get_zernike_base_matrixs(tmp_path / "nope.npz")
    assert "nope.npz" in str(exc.value)


def test_legacy_ascii_directory_is_gone():
    """The ASCII data set must not creep back in beside the .npz."""
    legacy = sorted(wc._STD_WAVEFRONT_PATH.parent.glob("*.txt"))
    assert legacy == [], f"legacy ASCII bases are back: {[p.name for p in legacy]}"


# --- validation of a malformed file ---------------------------------------
def _write(tmp_path: Path, matrices: np.ndarray, modes: np.ndarray) -> Path:
    target = tmp_path / "bad.npz"
    np.savez_compressed(target, modes=modes, matrices=matrices)
    return target


def test_reordered_modes_are_rejected(tmp_path):
    matrices = np.zeros((4, 8, 8))
    bad_modes = np.array([1, 3, 2, 4], dtype=np.int16)
    with pytest.raises(ValueError, match="mode numbers"):
        get_zernike_base_matrixs(_write(tmp_path, matrices, bad_modes))


def test_non_square_stack_is_rejected(tmp_path):
    matrices = np.zeros((3, 8, 9))
    modes = np.array([1, 2, 3], dtype=np.int16)
    with pytest.raises(ValueError, match="S, S"):
        get_zernike_base_matrixs(_write(tmp_path, matrices, modes))


def test_compressed_npz_round_trip_is_bit_exact(tmp_path):
    """The 56.4 MB -> 23.2 MB claim in the docstring depends on this."""
    rng = np.random.default_rng(7)
    original = rng.normal(scale=4.6634, size=(6, 16, 16))
    modes = np.arange(1, 7, dtype=np.int16)

    path = tmp_path / "rt.npz"
    np.savez_compressed(path, modes=modes, matrices=original)

    loaded = get_zernike_base_matrixs(path)
    assert np.array_equal(loaded, original), "npz round-trip lost or altered bits"
    assert loaded.dtype == np.float64


# --- duplicate suppression -------------------------------------------------
def test_script_does_not_duplicate_the_loader_again():
    """``dm_unit_compute.py`` carried a verbatim copy of five functions.

    Two of them were the buggy loader, so the copy was not harmless. This guard
    follows the repo convention of ``test_common_helpers_not_reintroduced.py``.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    for symbol in (
        "def get_zernike_base_matrixs",
        "class ZernikeCentroidCalculator",
        "def centroid_calculation",
        "def calculate_derotation",
        "def to_color",
    ):
        assert symbol not in source, f"{symbol} was re-duplicated into {SCRIPT.name}"


def test_script_delegates_to_the_canonical_module():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from ao_shaping.utils.wavefront.wavefront_calc import" in source
    assert "ZernikeCentroidCalculator" in source  # imported, not defined


def test_tuning_devices_package_init_imports_cleanly():
    """It used to import three modules that do not exist (only .m files do)."""
    assert PACKAGE_INIT.is_file()
    spec = importlib.util.spec_from_file_location(
        "_tuning_devices_init_probe", PACKAGE_INIT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]