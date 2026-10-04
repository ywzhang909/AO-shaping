"""Energy guard, flat init and end-of-epoch resync on the 2f-Fourier sim.

These three changes all come from the 2026-10-01 real-hardware run of
``slm-gsnet spgd`` (report/fouriergsnet_pipeline/hardware_run_20261001.md), which
was a random walk (dec=0.487) that ended 35.9% worse than it started:

  * the freeform init was ``uniform(-pi, pi)`` -- full-amplitude random phase,
    which destroys the focus (measured: 0-order peak 225 -> 17) and whose best
    iterate was the initial state;
  * ``square_quality_score`` is ``w_cv*exp(-2*cv) + w_ee*ee`` with no absolute
    energy term, so the search is free to raise the score by scattering light
    out of the box; the run traded 6x of encircled energy (0.158 -> 0.026) for a
    +0.0118 uniformity gain;
  * the last SLM write of every epoch was the REJECTED ``-delta`` perturbation,
    so the panel sat 2*delta away from the state the loop believed it was at.

The sim harness is the same style as
``test_slm_square_objective_search_sim.py`` (patch the two device factories the
optimizer resolves at call time), so the real objective plumbing, the Recorder
bookkeeping and the actual SPGD loop all execute.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ao_shaping.optimizer.wfless import slm_square_shaping as sq


class _State:
    """Shared mutable bench state: how many times the SLM has been written."""

    def __init__(self) -> None:
        self.writes = 0
        self.energy_after_write = 1.0


class _GuardedCamera:
    """Gaussian blob whose in-box energy drops once the SLM is driven.

    Reproduces the hardware failure mode: the objective's energy term collapses,
    so without a guard the search walks straight into an empty box.
    """

    def __init__(self, state: _State, shape: tuple[int, int] = (64, 64)) -> None:
        self.state = state
        self.shape = shape
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
        self._img = 200.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 5.0**2)))

    def __enter__(self) -> "_GuardedCamera":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def is_connected(self) -> bool:
        return True

    def get_numpy_image(self, n_sample: int = 1, skip_first: bool = False) -> np.ndarray:
        # The initial display is the reference (blob centred in the ROI); every
        # later write MOVES the blob far outside the target box while leaving the
        # frame total essentially unchanged.
        #
        # It must not be modelled as a uniform dim: `square_encircled_energy` is
        # box_sum / frame_sum, so a global brightness change cancels exactly and
        # the guard would be blind to it. The real hardware failure was light
        # being scattered OUT of the box into the rest of the frame, which is
        # what this reproduces.
        if self.state.writes < 2 or self.state.energy_after_write >= 1.0:
            return self._img.copy()
        moved = np.zeros_like(self._img)
        # dump the same total flux into a corner far outside the centred box
        h, w = self.shape
        flux = float(self._img.sum())
        corner = np.zeros_like(self._img)
        corner[:6, :6] = flux / 36.0
        moved[:] = corner
        return moved

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


class _RecordingSLM:
    def __init__(self, state: _State, *a: object, **k: object) -> None:
        self.state = state
        self.writes: list[np.ndarray] = []

    def __enter__(self) -> "_RecordingSLM":
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
        self.state.writes += 1

    def display_data(self, data: np.ndarray, *a: object, **k: object) -> None:
        self.writes.append(np.asarray(data))
        self.state.writes += 1

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
def guard_bench(monkeypatch: pytest.MonkeyPatch):
    state = _State()
    state.energy_after_write = 0.05      # perturbation scatters 95% of the light
    holder: dict[str, Any] = {"slm": _RecordingSLM(state), "state": state}

    def _make_camera(*a: object, **k: object) -> _GuardedCamera:
        return _GuardedCamera(state)

    def _make_slm(*a: object, **k: object) -> _RecordingSLM:
        return holder["slm"]

    monkeypatch.setattr(sq, "create_camera", _make_camera)
    monkeypatch.setattr(sq, "Santec", _make_slm)
    return holder


def _run(epochs: int, **over: Any) -> Any:
    cfg = sq.SlmSquareConfig(
        cam_type="sim",
        algorithm="spgd",
        target_side=20,
        cam_size=64,
        phase_grid=6,
        random_seed=0,
        objective="quality",
        **over,
    )
    return sq.optimize_slm_square(center=(32, 32), epochs=epochs, config=cfg)


class TestFlatInit:
    def test_default_init_is_flat_not_random(self, guard_bench) -> None:
        """The first phase written must be flat (all zeros).

        Regression: the freeform init used to be ``uniform(-pi, pi)``, i.e.
        full-amplitude random phase. On the bench that dropped the 0-order peak
        from 225 to 17 and made the initial state the best iterate of the run.
        """
        _run(epochs=1)

        first = guard_bench["slm"].writes[0]
        assert np.allclose(first, 0.0), (
            "freeform init must start from the flat (best-focus) state, "
            f"got max|phase|={np.abs(first).max():.4f}"
        )

    def test_explicit_amplitude_opts_into_random_init(self, guard_bench) -> None:
        """``init_amplitude_rad`` still allows a bounded random start."""
        _run(epochs=1, init_amplitude_rad=0.05)

        first = guard_bench["slm"].writes[0]
        assert np.abs(first).max() <= 0.05 + 1e-9
        assert np.abs(first).max() > 0.0


class TestEnergyGuard:
    def test_guard_gates_epochs_that_empty_the_box(self, guard_bench) -> None:
        """A collapsing in-ROI energy must SKIP the epoch, not just lower J.

        This is the decisive regression: the slm-pib family only subtracts 1e3
        from the objective, which protects the best-so-far track but still lets
        ``diff = pos_q - neg_q`` build a huge real gradient from the penalised
        pair, so the coefficients keep walking toward an empty box.
        """
        recorder = _run(epochs=4, max_roi_energy_loss=0.6)

        gates = [row.get("gate") for row in recorder.history]
        assert "energy" in gates, f"expected gated epochs, got gates={gates}"
        # Every gated row must report a zero gradient and an unchanged parameter
        # vector, i.e. no motion toward the rejected state.
        for row in recorder.history:
            if row.get("gate") == "energy":
                assert float(row["_diff"]) == 0.0
                assert np.allclose(np.asarray(row["_grad"]), 0.0)

    def test_guard_disabled_keeps_every_epoch(self, guard_bench) -> None:
        """max_roi_energy_loss=0 is the documented opt-out."""
        recorder = _run(epochs=4, max_roi_energy_loss=0.0)

        gates = [row.get("gate") for row in recorder.history]
        assert "energy" not in gates, f"guard fired while disabled: {gates}"

    def test_guard_does_not_gate_a_healthy_bench(self, monkeypatch) -> None:
        """With the energy preserved, nothing may be gated.

        Guards against a guard that fires on ordinary noise, which would silently
        freeze every search.
        """
        state = _State()
        state.energy_after_write = 1.0
        monkeypatch.setattr(sq, "create_camera", lambda *a, **k: _GuardedCamera(state))
        monkeypatch.setattr(sq, "Santec", lambda *a, **k: _RecordingSLM(state))

        recorder = _run(epochs=3, max_roi_energy_loss=0.6)

        gates = [row.get("gate") for row in recorder.history]
        assert "energy" not in gates, f"false-positive gating on a stable bench: {gates}"

    def test_out_of_range_guard_is_rejected(self, guard_bench) -> None:
        with pytest.raises(ValueError, match="max_roi_energy_loss"):
            _run(epochs=1, max_roi_energy_loss=1.5)


class TestEndOfEpochResync:
    def test_panel_is_left_on_the_accepted_state(self, guard_bench) -> None:
        """The final write of an epoch must be the accepted state, not ``-delta``.

        The last write used to be the rejected negative perturbation, leaving the
        panel 2*delta away in every dimension while the loop believed it was at
        ``_params``.
        """
        _run(epochs=2, max_roi_energy_loss=0.0)

        writes = guard_bench["slm"].writes
        assert len(writes) >= 5, f"expected the resync write, got {len(writes)}"
        # The resync re-displays the flat (zero) parameter vector, so the last
        # write must be all zeros even though the loop was perturbing.
        assert np.allclose(writes[-1], 0.0), (
            "the panel must be left showing the accepted parameters; "
            f"last write max|phase|={np.abs(writes[-1]).max():.4f}"
        )
