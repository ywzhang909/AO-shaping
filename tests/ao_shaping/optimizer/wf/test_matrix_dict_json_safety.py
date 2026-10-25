"""``to_dict`` promises JSON output; ``amplitude_optimization`` broke that promise.

``ZernikeResponseMatrixResult.to_dict`` documents 转换为字典用于JSON序列化 and
``DMResponseMatrixResult.to_dict`` documents "JSON-serializable dict (ndarrays -> lists)", yet
both converted every matrix field with ``.tolist()`` while leaving the nested per-mode
diagnostics untouched, so ``json.dumps`` raised on the whole payload. The HDF5 save/load pair
does round-trip these diagnostics -- and casts ``best_idx``/``optimal_*`` to ``int``/``float``
-- so the dict path was the odd one out and had to agree with the h5 path on types.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ao_shaping.optimizer.wf.dm_response_matrix import DMResponseMatrixResult
from ao_shaping.optimizer.wf.zernike_response_matrix import ZernikeResponseMatrixResult

_ARRAY_FIELDS = {
    "zernike": ("test_amps", "responses", "linearity"),
    "dm": ("test_voltages", "responses", "linearity"),
}
_SCORE_FIELD = {"zernike": "optimal_amplitude", "dm": "optimal_voltage"}


def _zernike_result() -> ZernikeResponseMatrixResult:
    matrix = np.random.default_rng(0).normal(size=(6, 5))
    return ZernikeResponseMatrixResult(
        matrix=matrix,
        variance_matrix=np.abs(np.random.default_rng(1).normal(size=(6, 5))) * 0.01,
        n_max=3,
        magnitude=0.5,
        wavelength_nm=1064,
        n_averages=3,
        n_cycles=2,
        timestamp="2026-10-26T00:00:00",
        excluded_piston=True,
        excluded_tip_tilt=False,
        pinv_matrix=np.linalg.pinv(matrix),
        amplitude_optimization={
            0: {
                "test_amps": np.array([0.1, 0.2, 0.3, 0.4]),
                "responses": np.array([1.0, 2.0, 3.0, 4.0]),
                "linearity": np.array([0.0, 0.0, 0.0]),
                "best_idx": np.int64(2),
                "optimal_amplitude": np.float64(0.3),
            },
            3: {
                "test_amps": np.array([0.1, 0.2, 0.3, 0.4]),
                "responses": np.array([0.5, 1.0, 1.5, 2.0]),
                "linearity": np.array([0.1, 0.0, 0.1]),
                "best_idx": 1,
                "optimal_amplitude": 0.2,
            },
        },
    )


def _dm_result() -> DMResponseMatrixResult:
    matrix = np.random.default_rng(2).normal(size=(8, 4))
    return DMResponseMatrixResult(
        matrix=matrix,
        variance_matrix=np.abs(np.random.default_rng(3).normal(size=(8, 4))) * 0.01,
        n_actuators=8,
        n_averages=3,
        n_cycles=2,
        timestamp="2026-10-26T00:00:00",
        pinv_matrix=np.linalg.pinv(matrix),
        amplitude_optimization={
            1: {
                "test_voltages": np.array([0.1, 0.2, 0.3, 0.4]),
                "responses": np.array([1.0, 2.0, 3.0, 4.0]),
                "linearity": np.array([0.0, 0.0, 0.0]),
                "best_idx": np.int64(2),
                "optimal_voltage": np.float64(0.3),
            },
            5: {
                "test_voltages": np.array([0.1, 0.2, 0.3, 0.4]),
                "responses": np.array([0.5, 1.0, 1.5, 2.0]),
                "linearity": np.array([0.1, 0.0, 0.1]),
                "best_idx": 1,
                "optimal_voltage": 0.2,
            },
        },
    )


_BUILDERS = {"zernike": _zernike_result, "dm": _dm_result}


@pytest.mark.parametrize("kind", list(_BUILDERS))
class TestJsonSafety:
    def test_to_dict_output_is_json_serializable(self, kind: str) -> None:
        json.dumps(_BUILDERS[kind]().to_dict())

    def test_amplitude_entries_use_lists_not_arrays(self, kind: str) -> None:
        payload = _BUILDERS[kind]().to_dict()
        for key, diag in payload["amplitude_optimization"].items():
            for field in _ARRAY_FIELDS[kind]:
                assert not isinstance(diag[field], np.ndarray), f"{key}:{field} stayed an array"
                assert isinstance(diag[field], list), f"{key}:{field} is {type(diag[field])}"
            for field in ("best_idx", _SCORE_FIELD[kind]):
                assert not isinstance(diag[field], np.generic), (
                    f"{key}:{field} is a numpy scalar; json.dumps rejects those as well as arrays"
                )


@pytest.mark.parametrize("kind", list(_BUILDERS))
class TestRoundTrip:
    def test_json_round_trip_restores_arrays_and_int_keys(self, kind: str) -> None:
        original = _BUILDERS[kind]()
        restored = type(original).from_dict(json.loads(json.dumps(original.to_dict())))
        score_field = _SCORE_FIELD[kind]

        assert set(restored.amplitude_optimization) == set(original.amplitude_optimization), (
            "JSON turns integer keys into strings; mode/actuator indices are ints in-process"
        )
        for idx in original.amplitude_optimization:
            diag = restored.amplitude_optimization[idx]
            for field in _ARRAY_FIELDS[kind]:
                np.testing.assert_allclose(
                    diag[field], original.amplitude_optimization[idx][field]
                )
                assert isinstance(diag[field], np.ndarray), f"{field} is {type(diag[field])}"
            assert isinstance(diag["best_idx"], int) and not isinstance(diag["best_idx"], bool)
            assert isinstance(diag[score_field], float)

    def test_in_memory_round_trip_still_matches(self, kind: str) -> None:
        original = _BUILDERS[kind]()
        restored = type(original).from_dict(original.to_dict())
        for idx, diag in original.amplitude_optimization.items():
            for field in _ARRAY_FIELDS[kind]:
                np.testing.assert_allclose(restored.amplitude_optimization[idx][field], diag[field])

    def test_absent_amplitude_stays_none(self, kind: str) -> None:
        result = _BUILDERS[kind]()
        result.amplitude_optimization = None
        restored = type(result).from_dict(result.to_dict())
        assert restored.amplitude_optimization is None
