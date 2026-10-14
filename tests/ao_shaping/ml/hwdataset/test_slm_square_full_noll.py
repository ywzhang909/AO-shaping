"""Tests for the ``spgd-square --debug`` full-Noll ``_c`` corpus indexing (TODO H-23).

The ``spgd-square`` Zernike-basis optimiser works in ACTIVE-mode space
(Defocus (2,0) + Spherical (4,0) by default, length 2), but the debug sidecar
records ``_c`` in **full Noll order** (``(n_max+1)(n_max+2)//2`` = 15 for the
default ``n_max=4``) with zeros on the inactive slots. This is what makes
``ml.hwdataset.index`` classify the record as :class:`PhaseSource.ZERNIKE`
(rule 1: sidecar ``n_max`` present AND ``len(_c) == (n_max+1)(n_max+2)//2``)
instead of dropping it as :class:`ExclusionReason.ODD_COEFFICIENT_LENGTH`
(a length-2 vector is neither triangular nor a perfect square).

Every fixture is written into a pytest ``tmp_path`` as a synthetic pickle, so
the suite never touches the real ``data/debug`` corpus. No camera/CCD is opened.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ml.hwdataset.index import (
    ExclusionReason,
    PhaseSource,
    build_hw_index,
    family_of,
)

# Small stand-in for the two-level real layout
# (``data/debug/<family>_<ts>/<ts>/<prefix>_<ts>_<ts>.pkl``).
_STAMP = "20260101_000000"


def _frame(shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    """Return a small ``uint8`` CCD frame (2-D, as the index requires)."""
    return (np.arange(shape[0] * shape[1], dtype=np.uint16) % 251).astype(np.uint8).reshape(shape)


def _write_sidecar(dir: Path, name: str, **fields: Any) -> Path:
    """Write a sidecar JSON file into ``dir``."""
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / name
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


def _write_square_zernike_pkl(
    dir: Path, n_terms: int = 15
) -> Path:
    """Write a ``slm_square`` debug pickle with a sidecar ``n_max=4``.

    Mirrors ``data/debug/slm_square/<ts>/``: no ``_phase``, a ``_c`` vector, a
    per-record ``exp_t``, and a ``<stem>.json`` sidecar carrying ``n_max`` (the
    square sidecar always records ``basis`` / ``n_max`` / ``zernike_radius`` /
    ``zernike_mask``).

    Args:
        dir: Family directory (``slm_square``).
        n_terms: Length of ``_c``. 15 = full Noll for ``n_max=4``; 2 = the bare
            active-mode vector (the pre-fix defect).

    Returns:
        The written ``.pkl`` path.
    """
    run_dir = dir / _STAMP
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"square_zernike_{_STAMP}"
    # A full-length `_c` is all rule 1 needs. The two non-zero entries are placed
    # arbitrarily here (this fixture only exercises CLASSIFICATION, which keys on
    # length + sidecar n_max); mode placement is asserted separately, against the
    # real `_scatter_active_to_noll`, in TestSquareFullNollReconstruction.
    c = np.zeros(n_terms, dtype=np.float64)
    if n_terms >= 2:
        c[0] = 0.3
        c[1] = -0.2
    payload = {
        i: {
            "_c": np.asarray(c, dtype=np.float64),
            "_grad": np.zeros(n_terms, dtype=np.float64),
            "_img": _frame(),
            "exp_t": 2.0,
        }
        for i in range(3)
    }
    path = run_dir / f"{stem}.pkl"
    path.write_bytes(pickle.dumps(payload))
    # The square sidecar always records n_max; that is what rule 1 keys on.
    _write_sidecar(
        run_dir,
        f"{stem}.json",
        family="spgd-square",
        basis="zernike",
        n_max=4,
        zernike_radius=600.0,
        zernike_mask=None,
    )
    return path


class TestSquareFullNollIndex:
    """A full-Noll ``spgd-square`` sidecar record indexes as ZERNIKE (H-23)."""

    def test_family_is_slm_square(self, tmp_path: Path) -> None:
        """The square debug tree resolves to the ``slm_square`` family."""
        path = tmp_path / "slm_square" / _STAMP / "x.pkl"
        assert family_of(path) == "slm_square"

    def test_full_noll_sidecar_is_zernike(self, tmp_path: Path) -> None:
        """(a) A full-Noll ``_c`` (15) + sidecar ``n_max=4`` -> ZERNIKE n_max=4."""
        pkl = _write_square_zernike_pkl(tmp_path / "slm_square", n_terms=15)
        index = build_hw_index([pkl])
        assert len(index) == 3, "all 3 records must be indexed (was 0/3 before H-23)"
        ref = index.records[0]
        assert ref.source is PhaseSource.ZERNIKE
        assert ref.n_max == 4
        assert ref.n_terms == 15
        assert ref.family == "slm_square"
        # Exposure comes from the per-record ``exp_t``.
        assert ref.exposure_ms == 2.0

    def test_bare_length2_no_sidecar_is_excluded(self, tmp_path: Path) -> None:
        """(c) A bare length-2 ``_c`` (no matching sidecar) is still excluded.

        This guards against over-broad matching: a length-2 vector matches no
        accepted length (neither triangular nor a perfect square), so rule 5
        must still drop it as :class:`ExclusionReason.ODD_COEFFICIENT_LENGTH`
        even when it is not rescued by rule 1 (no sidecar ``n_max``).
        """
        # No sidecar written -> rule 1 cannot fire; length 2 -> rule 5.
        run_dir = tmp_path / "slm_square" / _STAMP
        run_dir.mkdir(parents=True, exist_ok=True)
        stem = f"square_bare_{_STAMP}"
        payload = {
            i: {
                "_c": np.array([0.3, -0.2], dtype=np.float64),
                "_grad": np.zeros(2, dtype=np.float64),
                "_img": _frame(),
                "exp_t": 2.0,
            }
            for i in range(3)
        }
        path = run_dir / f"{stem}.pkl"
        path.write_bytes(pickle.dumps(payload))
        index = build_hw_index([path])
        assert len(index) == 0
        assert index.excluded.get(ExclusionReason.ODD_COEFFICIENT_LENGTH, 0) == 3


class TestSquareFullNollReconstruction:
    """The active modes land on the right Noll slots and reconstruct identically."""

    def test_active_modes_at_correct_noll_indices(self) -> None:
        """(b) ``_scatter_active_to_noll`` places defocus/spherical at Noll 4/11."""
        from ao_shaping.optimizer.wfless.slm_square_shaping import (
            ZERNIKE_ACTIVE_MODES,
            _scatter_active_to_noll,
            _zernike_indices,
            calc_n_zernike_terms,
        )

        n_max = 4
        nk = calc_n_zernike_terms(n_max)  # 15
        noll = _zernike_indices(n_max)
        active_pos = [
            i for i, m in enumerate(noll) if m in ZERNIKE_ACTIVE_MODES
        ]
        # Noll 4 = (2,0) defocus -> 0-based index 3; Noll 11 = (4,0) spherical -> 10.
        assert active_pos == [3, 10]
        defocus, spherical = 0.7, 1.3
        full = _scatter_active_to_noll(
            np.array([defocus, spherical], dtype=np.float64), active_pos, nk
        )
        assert full.shape == (nk,)
        # Active modes land on the same Noll slots with the same radian amplitudes.
        assert np.isclose(full[3], defocus)
        assert np.isclose(full[10], spherical)
        # Every inactive Noll slot is exactly zero.
        inactive = [i for i in range(nk) if i not in active_pos]
        assert np.all(full[inactive] == 0.0)
        # Cross-validate the hand-written scatter against the canonical
        # (n,m)->Noll-array converter (RZern-based, repo single source of
        # truth). The scatter must place the two active modes at exactly the
        # same Noll positions with the same amplitudes as the canonical API.
        from ao_shaping.utils.wavefront.zernike_utils import coefficients_to_array

        canonical = coefficients_to_array(
            {(2, 0): defocus, (4, 0): spherical}, n_max
        )
        assert canonical.shape == full.shape
        assert np.array_equal(full, canonical)

    def test_reconstruction_identical_to_active_modes(self) -> None:
        """(b) Full-Noll ``_c`` reconstructs the same panel as the active modes.

        The corpus consumer (``ml.hwdataset.transforms.zernike_coeffs_to_panel``)
        reads ``_c`` as a **full Noll-order** vector. Reconstructing from the
        recorded full-Noll ``_c`` must be numerically identical to reconstructing
        from only the active modes (the representation the freeform/basis path
        feeds the forward model): the active modes land on the same Noll slots
        with the same radian amplitudes, and the inactive (zero) slots add
        nothing.
        """
        from ao_shaping.optimizer.wfless.slm_square_shaping import (
            ZERNIKE_ACTIVE_MODES,
            _scatter_active_to_noll,
            _zernike_indices,
            calc_n_zernike_terms,
        )
        from ml.hwdataset.transforms import zernike_coeffs_to_panel

        n_max = 4
        nk = calc_n_zernike_terms(n_max)
        noll = _zernike_indices(n_max)
        active_pos = [
            i for i, m in enumerate(noll) if m in ZERNIKE_ACTIVE_MODES
        ]
        defocus, spherical = 0.7, -1.3
        # The optimiser's in-run work vector (active-mode space, length 2).
        active = np.array([defocus, spherical], dtype=np.float64)
        # The recorded debug ``_c`` (full Noll order, length 15).
        full = _scatter_active_to_noll(active, active_pos, nk)
        # The same coefficients expressed as a plain Noll-order flat vector the
        # way a reader of the corpus would rebuild it from the sidecar: zeros in
        # every slot, the two active modes filled in at their Noll positions.
        # This is the "basis path" reference: the active modes alone, without
        # the scatter helper, must give the identical panel.
        reference = np.zeros(nk, dtype=np.float64)
        reference[active_pos[0]] = float(active[0])  # Noll 4 (2,0) defocus
        reference[active_pos[1]] = float(active[1])  # Noll 11 (4,0) spherical

        resolution = (96, 64)
        radius = 28.0
        from_full = zernike_coeffs_to_panel(
            full, n_max=n_max, resolution=resolution, radius=radius
        )
        from_ref = zernike_coeffs_to_panel(
            reference, n_max=n_max, resolution=resolution, radius=radius
        )
        # Both must be real, nonzero panels (guards against a degenerate zero).
        assert np.all(np.isfinite(from_full))
        assert float(np.max(np.abs(from_full))) > 1e-3
        # Bit-identical: the scattered full-Noll vector and the hand-built
        # reference vector describe the same phase (the inactive zero slots
        # contribute nothing), so the recorded representation reconstructs
        # exactly the panel the basis path would have produced.
        assert np.array_equal(from_full, from_ref)
