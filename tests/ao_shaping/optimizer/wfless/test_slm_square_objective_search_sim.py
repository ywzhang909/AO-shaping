"""Square-shaping search branches on the 2f-Fourier sim: SPGD and heuristic.

Covers the two search drivers in ``optimizer.wfless.slm_square_shaping`` that
``tests/ao_shaping/runners/test_slm_gsnet_objective_cli.py`` only exercises at
the *dispatcher* level (it stubs the whole execution path). Those tests prove
the objective reaches the config; they cannot prove the search actually
consumes it, which is where a sign error would silently optimise the wrong
direction.

The Oracle review flagged exactly this risk for the heuristic branch: both
``run_heuristic_search(maximize=True)`` and the SPGD ``diff = pos - neg`` are
higher-is-better, so a double negation in
:func:`square_shaping_objective` would be invisible to a wiring-only test yet
would invert every run on the bench.

Sim harness: a Gaussian far-field blob standing in for the CCD frame, and a
no-op SLM. Both are injected by patching the two names the optimizer resolves at
call time (``create_camera`` and ``Santec``), so the real objective plumbing,
Recorder bookkeeping and both search drivers all execute.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

from ao_shaping.optimizer.wfless import slm_square_shaping as sq


class _SimCamera:
    """Minimal ``BaseCamera`` stand-in returning a fixed Gaussian blob."""

    def __init__(self, shape: tuple[int, int] = (64, 64)) -> None:
        self.shape = shape
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
        self._img = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 5.0**2)))
        self._img *= 200.0
        self._img = self._img.astype(np.float64)

    # -- context-manager / device protocol ---------------------------------
    def __enter__(self) -> "_SimCamera":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def is_connected(self) -> bool:
        return True

    # -- capture ------------------------------------------------------------
    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = False) -> np.ndarray:
        return self._img.copy()

    def reset_window(self, center: Any, size: Any) -> Any:
        return self.shape, center

    def set_exposure_time(self, exposure_time_ms: float) -> None:
        return None

    exposure_time_ms = 80.0

    def get_exposure_time(self) -> float:
        return self.exposure_time_ms

    def enable_auto_exposure(self, *a: object, **k: object) -> None:
        return None

    def get_auto_exposure_state(self) -> Any:
        return None

    def set_auto_exposure_range(self, *a: object, **k: object) -> None:
        return None

    @staticmethod
    def get_cam_list() -> list:
        return ["sim"]


class _SimSLM:
    """No-op SLM: records writes so the phase path is genuinely exercised."""

    def __init__(self, *a: object, **k: object) -> None:
        self.writes: list[np.ndarray] = []

    def __enter__(self) -> "_SimSLM":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def is_connected(self) -> bool:
        return True

    def display_phase(self, phase: np.ndarray, *a: object, **k: object) -> None:
        self.writes.append(np.asarray(phase))

    def display_data(self, data: np.ndarray, *a: object, **k: object) -> None:
        self.writes.append(np.asarray(data))

    def create_phase_from_array(self, phase: np.ndarray, *a: object, **k: object) -> np.ndarray:
        self.writes.append(np.asarray(phase))
        return np.zeros(phase.shape, dtype=np.uint16)

    def display_memory(self, *a: object, **k: object) -> None:
        return None

    def set_grayscale(self, gray: object, *a: object, **k: object) -> None:
        return None

    def get_displayed_memory_number(self) -> int:
        return 0

    @property
    def _max_gray(self) -> int:
        return 1023


@pytest.fixture
def sim_bench(monkeypatch: pytest.MonkeyPatch):
    """Patch the two device factories the optimizer resolves at call time."""
    slm_holder: dict[str, _SimSLM] = {}

    def _make_camera(*a: object, **k: object) -> _SimCamera:
        return _SimCamera()

    def _make_slm(*a: object, **k: object) -> _SimSLM:
        slm = _SimSLM()
        slm_holder["slm"] = slm
        return slm

    monkeypatch.setattr(sq, "create_camera", _make_camera)
    monkeypatch.setattr(sq, "Santec", _make_slm)
    return slm_holder


def _run(objective: str, algorithm: str, epochs: int, **over: Any) -> Any:
    cfg = sq.SlmSquareConfig(
        cam_type="sim",
        algorithm=algorithm,
        target_side=20,
        cam_size=64,
        phase_grid=6,
        random_seed=0,
        objective=objective,
        **over,
    )
    return sq.optimize_slm_square(
        center=(32, 32), epochs=epochs, config=cfg
    )


class TestSquareSearchBranches:
    @pytest.mark.parametrize("algorithm", ["spgd", "ga"])
    def test_quality_branch_runs(self, sim_bench, algorithm: str) -> None:
        recorder = _run("quality", algorithm, epochs=2)

        assert len(recorder) > 0
        assert len(recorder.columns) > 0

    @pytest.mark.parametrize("algorithm", ["spgd", "ga"])
    def test_pearson_branch_runs(self, sim_bench, algorithm: str) -> None:
        recorder = _run("pearson", algorithm, epochs=2)

        assert len(recorder) > 0
        assert len(recorder.columns) > 0

    def test_spgd_writes_phase_to_slm(self, sim_bench) -> None:
        _run("pearson", "spgd", epochs=2)

        assert sim_bench.get("slm") is not None
        assert sim_bench["slm"].writes, "SPGD must actually push phase to the SLM"

    def test_heuristic_writes_phase_to_slm(self, sim_bench) -> None:
        _run("pearson", "ga", epochs=2, pop_size=4)

        assert sim_bench.get("slm") is not None
        assert sim_bench["slm"].writes, "the heuristic must load phases too"

    def test_recorded_pearson_column_is_the_loss(self, sim_bench) -> None:
        """The Recorder must expose the objective under its own name."""
        recorder = _run("pearson", "spgd", epochs=2)

        assert any("pearson" in str(c) for c in recorder.columns), recorder.columns

    def test_recorded_quality_column_present(self, sim_bench) -> None:
        recorder = _run("quality", "spgd", epochs=2)

        assert any("quality" in str(c) for c in recorder.columns), recorder.columns


class TestObjectiveDirection:
    """The search must move *toward* better objective values, not away."""

    def test_score_is_higher_is_better_for_pearson(self) -> None:
        """A frame closer to the target must score HIGHER under 'pearson'.

        This is the invariant a double negation would break: the dispatcher
        negates the loss once so the search stack stays higher-is-better.
        """
        img = np.zeros((64, 64), dtype=np.float64)
        yy, xx = np.mgrid[0:64, 0:64]
        img += 50.0 * np.exp(-(((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 6.0**2)))

        good = sq.square_objective_score(
            img=img, cv=0.2, encircled_energy=0.8, aspect_ratio=1.0,
            center=(32, 32), side=20, objective="pearson",
        )
        # A flat, structureless frame anti-correlates with the ROI target.
        bad = sq.square_objective_score(
            img=np.full((64, 64), 50.0),
            cv=0.9, encircled_energy=0.2, aspect_ratio=1.4,
            center=(32, 32), side=20, objective="pearson",
        )

        assert good > bad, "pearson score must reward the more target-like frame"

    def test_pearson_score_is_negated_loss(self) -> None:
        from ao_shaping.utils.image.target.metrics import pearson_shape_metric

        img = np.zeros((64, 64), dtype=np.float64)
        yy, xx = np.mgrid[0:64, 0:64]
        img += 50.0 * np.exp(-(((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 6.0**2)))

        # The square dispatcher scores against a SQUARE target box (aspect 1.0),
        # not the 4:3 rectangle used elsewhere in the report.
        loss, _ = pearson_shape_metric(img, (32.0, 32.0), "square", 20, 1.0)
        score = sq.square_objective_score(
            img=img, cv=0.0, encircled_energy=0.0, aspect_ratio=1.0,
            center=(32, 32), side=20, objective="pearson",
        )

        assert score == pytest.approx(-loss, abs=1e-9)

    def test_quality_score_stays_in_unit_interval(self) -> None:
        for cv, ee, ar in [(0.0, 1.0, 1.0), (1.0, 0.0, 2.0), (0.5, 0.5, 1.0)]:
            score = sq.square_objective_score(
                img=np.zeros((8, 8)), cv=cv, encircled_energy=ee,
                aspect_ratio=ar, center=(4, 4), side=4, objective="quality",
            )
            assert 0.0 <= score <= 1.0
