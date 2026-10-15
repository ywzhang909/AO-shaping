"""Rule 4b: an active-mode ``_c`` is a real Zernike record, not junk (issue #40).

The square family's zernike basis drives only a subset of the basis -- defocus
plus spherical by default -- and stores ``_c`` over just those modes. So
``len(_c)`` is 2, which is neither a triangular Zernike count nor a perfect
square, and every such record used to be excluded wholesale as
``odd_coefficient_length``. That is the "zernike basis yields 0% of the corpus"
half of #40; the freeform half always worked.

Three things have to hold for the fix to be honest, and each is pinned here:

1. classification stops dropping these records, and only when the sidecar
   actually declares the modes;
2. the scatter lands each coefficient on its own Noll mode -- a panel built from
   the wrong mode looks like valid data, which is the failure this repo treats as
   a red line;
3. records written before the sidecar existed keep their old fate, so the change
   cannot retroactively reinterpret old corpora.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from ml.hwdataset.index import ExclusionReason, PhaseSource, build_hw_index
from ml.hwdataset.transforms import expand_active_modes_to_noll, zernike_coeffs_to_panel

_STAMP = "20260101_000000"

#: What the square zernike basis drives by default: defocus (2,0) + spherical (4,0).
_ACTIVE = [(2, 0), (4, 0)]
_N_MAX = 4
_N_TERMS = 15  # (4+1)(4+2)/2


def _frame() -> np.ndarray:
    return np.zeros((16, 16), dtype=np.uint8)


def _write_family(
    root: Path,
    family: str,
    *,
    n_terms: int,
    sidecar: dict | None,
    n: int = 3,
) -> Path:
    """Write one run directory holding a ``_c`` payload, plus an optional sidecar."""
    run_dir = root / family / _STAMP
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        i: {
            "_c": np.linspace(0.1, 0.9, n_terms),
            "_img": _frame(),
            "exp_t": 2.0,
        }
        for i in range(n)
    }
    stem = f"run_{_STAMP}"
    (run_dir / f"{stem}.pkl").write_bytes(pickle.dumps(payload))
    if sidecar is not None:
        (run_dir / f"{stem}.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return run_dir


def _index(root: Path):
    return build_hw_index(root)


# ---------------------------------------------------------------------------
# 1. classification
# ---------------------------------------------------------------------------


def test_active_mode_record_is_indexed_as_zernike(tmp_path: Path) -> None:
    """The 0% becomes 100%: length 2 with a declared mode list is a Zernike record."""
    _write_family(
        tmp_path,
        "slm_square_x",
        n_terms=len(_ACTIVE),
        sidecar={"n_max": _N_MAX, "active_modes": [list(m) for m in _ACTIVE]},
    )
    index = _index(tmp_path)

    assert index.counts_by_source() == {PhaseSource.ZERNIKE: 3}
    ref = index.records[0]
    assert ref.n_max == _N_MAX
    assert ref.n_terms == len(_ACTIVE)
    assert ref.active_modes == tuple(_ACTIVE)


def test_without_the_sidecar_key_the_record_is_still_excluded(tmp_path: Path) -> None:
    """No declaration, no opinion: old corpora keep their old fate.

    This is the guard against retroactively reinterpreting data recorded before
    the sidecar existed, where the same length-2 vector could have meant anything.
    """
    _write_family(tmp_path, "slm_square_x", n_terms=2, sidecar={"n_max": _N_MAX})
    index = _index(tmp_path)

    assert len(index) == 0
    assert index.excluded == {ExclusionReason.ODD_COEFFICIENT_LENGTH: 3}


def test_declared_modes_must_match_the_coefficient_count(tmp_path: Path) -> None:
    """A length/mode disagreement is a corrupt sidecar, so it is not trusted."""
    _write_family(
        tmp_path,
        "slm_square_x",
        n_terms=2,
        sidecar={"n_max": _N_MAX, "active_modes": [[2, 0], [4, 0], [3, -1]]},
    )
    index = _index(tmp_path)

    assert len(index) == 0
    assert index.excluded == {ExclusionReason.ODD_COEFFICIENT_LENGTH: 3}


def test_n_max_is_required_to_claim_the_record(tmp_path: Path) -> None:
    """Without ``n_max`` there is no basis length to scatter into."""
    _write_family(
        tmp_path,
        "slm_square_x",
        n_terms=len(_ACTIVE),
        sidecar={"active_modes": [list(m) for m in _ACTIVE]},
    )
    index = _index(tmp_path)

    assert len(index) == 0
    assert index.excluded == {ExclusionReason.ODD_COEFFICIENT_LENGTH: 3}


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [[2, 0, 0]],
        [[2]],
        [["a", 0]],
        [True, 0],
        "defocus",
        7,
    ],
)
def test_malformed_active_modes_fall_back_to_exclusion(
    tmp_path: Path, bad: object
) -> None:
    """A malformed declaration must never be half-trusted into an index entry."""
    _write_family(
        tmp_path, "slm_square_x", n_terms=2, sidecar={"n_max": _N_MAX, "active_modes": bad}
    )
    index = _index(tmp_path)

    assert len(index) == 0
    assert index.excluded == {ExclusionReason.ODD_COEFFICIENT_LENGTH: 3}


def test_full_length_zernike_records_are_unaffected(tmp_path: Path) -> None:
    """A record that already spans the whole basis keeps ``active_modes`` unset."""
    _write_family(
        tmp_path,
        "slm_square_x",
        n_terms=_N_TERMS,
        sidecar={"n_max": _N_MAX, "active_modes": [list(m) for m in _ACTIVE]},
    )
    index = _index(tmp_path)

    assert index.counts_by_source() == {PhaseSource.ZERNIKE: 3}
    # rule 1 (triangular + n_max) fires first, so the scatter must NOT run.
    assert index.records[0].active_modes is None


# ---------------------------------------------------------------------------
# 2. the scatter
# ---------------------------------------------------------------------------


def test_scatter_places_each_coefficient_on_its_own_mode() -> None:
    """Defocus -> Noll 4 (index 3), spherical -> Noll 11 (index 10), rest zero."""
    full = expand_active_modes_to_noll(
        np.array([1.0, 0.5]), active_modes=_ACTIVE, n_max=_N_MAX
    )

    assert len(full) == _N_TERMS
    assert full[3] == 1.0
    assert full[10] == 0.5
    assert np.count_nonzero(full) == 2, "undriven modes must stay exactly zero"


def test_scatter_is_not_order_insensitive() -> None:
    """Values follow their declared mode, so swapping must move them."""
    swapped = expand_active_modes_to_noll(
        np.array([1.0, 0.5]), active_modes=[(4, 0), (2, 0)], n_max=_N_MAX
    )
    ordered = expand_active_modes_to_noll(
        np.array([1.0, 0.5]), active_modes=_ACTIVE, n_max=_N_MAX
    )

    assert not np.array_equal(swapped, ordered)


def test_scatter_reconstructs_the_same_panel_as_a_full_vector() -> None:
    """The point of the whole change: identical phase, not merely a valid one.

    A panel built from the wrong modes is still a well-formed array, so only a
    comparison against the equivalent full-length vector can catch it.
    """
    coeffs = np.array([0.3, -0.7])
    scattered = expand_active_modes_to_noll(
        coeffs, active_modes=_ACTIVE, n_max=_N_MAX
    )
    hand_built = np.zeros(_N_TERMS)
    hand_built[3], hand_built[10] = 0.3, -0.7

    assert np.array_equal(scattered, hand_built)

    kwargs = {"n_max": _N_MAX, "resolution": (120, 192), "radius": 60.0}
    assert np.array_equal(
        zernike_coeffs_to_panel(scattered, **kwargs),
        zernike_coeffs_to_panel(hand_built, **kwargs),
    )


def test_scatter_rejects_a_length_mismatch() -> None:
    """Silently zipping to the shorter one would drop a coefficient."""
    with pytest.raises(ValueError, match="one-to-one"):
        expand_active_modes_to_noll(
            np.array([1.0, 0.5, 0.25]), active_modes=_ACTIVE, n_max=_N_MAX
        )


def test_scatter_rejects_a_mode_outside_the_basis() -> None:
    """A mode the basis does not contain must not be placed on a neighbour."""
    with pytest.raises(ValueError, match="not in the Noll set"):
        expand_active_modes_to_noll(
            np.array([1.0, 0.5]), active_modes=[(2, 0), (99, 0)], n_max=_N_MAX
        )


def test_scatter_honours_a_larger_n_max() -> None:
    """The same two modes inside a bigger basis land at the same positions."""
    full = expand_active_modes_to_noll(
        np.array([1.0, 0.5]), active_modes=_ACTIVE, n_max=6
    )

    assert len(full) == 28
    assert full[3] == 1.0
    assert full[10] == 0.5


# ---------------------------------------------------------------------------
# 3. the index cache
# ---------------------------------------------------------------------------


def test_active_modes_survive_the_index_cache(tmp_path: Path) -> None:
    """A cached index must still scatter, or the loader raises on a short ``_c``.

    Encoding and decoding are separate functions that both silently accept a
    missing key, so this is the only check that they agree.
    """
    _write_family(
        tmp_path,
        "slm_square_x",
        n_terms=len(_ACTIVE),
        sidecar={"n_max": _N_MAX, "active_modes": [list(m) for m in _ACTIVE]},
    )
    cache = tmp_path / "index_cache.json"

    scanned = build_hw_index(tmp_path, index_cache=cache)
    loaded = build_hw_index(tmp_path, index_cache=cache)

    assert cache.exists()
    assert loaded.records[0].active_modes == tuple(_ACTIVE)
    assert loaded.records[0].active_modes == scanned.records[0].active_modes


def test_a_full_length_record_stays_unset_through_the_cache(tmp_path: Path) -> None:
    """``None`` must survive as ``None``, not come back as an empty tuple."""
    _write_family(tmp_path, "slm_square_x", n_terms=_N_TERMS, sidecar={"n_max": _N_MAX})
    cache = tmp_path / "index_cache.json"

    build_hw_index(tmp_path, index_cache=cache)
    loaded = build_hw_index(tmp_path, index_cache=cache)

    assert loaded.records[0].active_modes is None
