"""The four ``zernike_slm_response_matrix`` sidecars have to be checked against each other.

``save_zernike_slm_response_matrix`` writes ``.matrix.npy`` / ``.variance.npy`` / ``.pinv.npy``
plus a metadata JSON, and ``to_dict`` declares ``matrix_shape`` / ``n_spots`` / ``n_modes`` --
none of which ``from_dict`` reads. A directory in which one file was regenerated, or copied in
from another calibration, therefore loads without a word, and ``apply_zernike_correction`` then
drives the SLM from a pseudoinverse that does not belong to the matrix it is multiplying.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.optimizer.wf.interaction_matrix import (
    ZernikeSLMResponseMatrixResult,
    load_zernike_slm_response_matrix,
    save_zernike_slm_response_matrix,
)
from ao_shaping.utils.wavefront.matrix_utils import compute_pinv

N_SPOTS = 50
N_MODES = 10


def _result() -> ZernikeSLMResponseMatrixResult:
    matrix = np.random.default_rng(0).normal(size=(2 * N_SPOTS, N_MODES))
    matrix[:N_MODES, :] += np.eye(N_MODES) * 10
    variance = np.abs(np.random.default_rng(1).normal(size=matrix.shape)) * 0.01
    return ZernikeSLMResponseMatrixResult(
        matrix=matrix,
        variance_matrix=variance,
        slm_wavelength_nm=1064,
        wfs_resolution="768",
        pupil_diameter_mm=2.24,
        magnitude_rad=0.5,
        n_cycles=3,
        n_averages=5,
        timestamp="2026-10-26T12:00:00",
        pinv_matrix=compute_pinv(matrix),
    )


class TestConsistentSet:
    def test_consistent_set_still_loads(self, tmp_path: Path) -> None:
        base = tmp_path / "m"
        original = _result()
        save_zernike_slm_response_matrix(original, base)

        loaded = load_zernike_slm_response_matrix(base)

        np.testing.assert_allclose(loaded.matrix, original.matrix)
        np.testing.assert_allclose(loaded.variance_matrix, original.variance_matrix)
        np.testing.assert_allclose(loaded.pinv_matrix, original.pinv_matrix)


class TestMixedSetIsRejected:
    def test_variance_from_another_calibration_is_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "m"
        save_zernike_slm_response_matrix(_result(), base)
        np.save(base.with_suffix(".variance.npy"), np.zeros((N_SPOTS, N_MODES)))

        with pytest.raises(ValueError, match="variance"):
            load_zernike_slm_response_matrix(base)

    def test_pinv_that_is_not_the_matrices_pseudoinverse_is_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "m"
        save_zernike_slm_response_matrix(_result(), base)
        np.save(base.with_suffix(".pinv.npy"), np.zeros((2 * N_SPOTS, N_MODES)))

        with pytest.raises(ValueError, match="pinv"):
            load_zernike_slm_response_matrix(base)

    def test_declared_matrix_shape_disagreeing_with_array_is_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "m"
        save_zernike_slm_response_matrix(_result(), base)
        meta_path = base.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["matrix_shape"] = [1, 1]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(ValueError, match="matrix_shape"):
            load_zernike_slm_response_matrix(base)

    def test_declared_spot_and_mode_counts_are_checked(self, tmp_path: Path) -> None:
        base = tmp_path / "m"
        save_zernike_slm_response_matrix(_result(), base)
        meta_path = base.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["n_spots"] = 7
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(ValueError, match="n_spots"):
            load_zernike_slm_response_matrix(base)

    def test_legacy_metadata_without_declared_shapes_still_loads(self, tmp_path: Path) -> None:
        base = tmp_path / "m"
        save_zernike_slm_response_matrix(_result(), base)
        meta_path = base.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("matrix_shape", "n_spots", "n_modes"):
            meta.pop(key, None)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        loaded = load_zernike_slm_response_matrix(base)

        assert loaded.n_spots == N_SPOTS
        assert loaded.n_modes == N_MODES
