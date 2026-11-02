"""Tests for local phase fine-tuning and its hardware-safety boundaries.

A lower loss alone is not enough for this stage.  These checks pin the reasons a
separate refiner exists instead of reusing the free-phase teacher:

* optimisation must really propagate through ``Sim2f`` and improve a requested
  shape, while all reported metrics remain reproducible from the shared
  :func:`ml.sim2f.losses.objective_terms` implementation (the repository has
  already been burnt by duplicated objectives drifting apart);
* both trust-region parameterisations must bound the update even under an
  intentionally excessive learning rate, and rejection must return the exact
  input phase so a speculative update cannot reach hardware;
* a non-flat starting phase must retain its structure, and update blur must
  measurably suppress spatial frequencies that a fitted bench cannot be trusted
  to predict;
* disabling ``keep_best`` must still leave a gentle decaying schedule useful,
  while malformed grids and empty targets fail before optimisation rather than
  being silently broadcast.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.sim2f.config import RefineConfig, Sim2fConfig
from ml.sim2f.finetune import PhaseRefiner, RefineResult
from ml.sim2f.losses import objective_terms
from ml.sim2f.optics import Sim2f
from ml.sim2f.targets import TargetSample, draw_target

#: The reference bench's camera sampling at one quarter of its pupil grid, so
#: these tests exercise the real shaping difficulty without paying its FFT cost.
SHAPE = Sim2fConfig(
    grid=64,
    region_span_px=900.0,
    ccd_px=64,
    ccd_pixel_um=4.4,
    exposure_ms=0.0,
)


def _square(seed: int = 7) -> TargetSample:
    """Build the deterministic centred target used by the refinement checks.

    Args:
        seed: NumPy generator seed.  The square size is fixed, but an explicit
            generator keeps the call faithful to the public target API.

    Returns:
        A centred square occupying 30 percent of the camera width.
    """
    return draw_target(
        "square",
        SHAPE.ccd_px,
        np.random.default_rng(seed),
        size_frac=(0.3, 0.3),
    )


def _flat_phase() -> np.ndarray:
    """Return a float32 flat phase on the pupil grid.

    Returns:
        A ``(grid, grid)`` zero phase in radians.
    """
    return np.zeros((SHAPE.grid, SHAPE.grid), dtype=np.float32)


def _high_frequency_energy(update: np.ndarray) -> float:
    """Measure update roughness with the interior five-point Laplacian.

    Ignoring the one-pixel border avoids turning the refiner's replicate-padded
    Gaussian boundary condition into an artificial wrap-around discontinuity.

    Args:
        update: Two-dimensional phase update in radians.

    Returns:
        Mean absolute discrete Laplacian over the interior pixels.
    """
    laplacian = (
        -4.0 * update[1:-1, 1:-1]
        + update[:-2, 1:-1]
        + update[2:, 1:-1]
        + update[1:-1, :-2]
        + update[1:-1, 2:]
    )
    return float(np.mean(np.abs(laplacian)))


def test_refinement_improves_objective_and_encircled_energy() -> None:
    """Autograd reaches the phase and finds an accepted improvement from flat.

    A centred square already contains almost all flat-spot energy, so a strong
    positive EE weight makes the energy direction observable instead of letting
    shape matching trade a tiny amount of EE away.
    """
    sim = Sim2f(SHAPE, device="cpu")
    sample = _square()
    cfg = RefineConfig(iters=30, lr=0.02, w_ee=10.0)

    result = PhaseRefiner(sim, cfg).finetune(
        _flat_phase(), sample.intensity, support=sample.support
    )

    assert result.accepted is True
    assert result.objective < result.objective_init
    assert result.ee > result.ee_init


def test_reported_metrics_match_the_shared_objective() -> None:
    """Every reported score is reproducible from the returned camera frame.

    This prevents the refiner from acquiring a local objective copy whose
    normalisation or frame preparation can drift from training and evaluation.
    """
    sim = Sim2f(SHAPE, device="cpu")
    sample = _square(seed=11)
    cfg = RefineConfig(iters=15)
    result = PhaseRefiner(sim, cfg).finetune(
        _flat_phase(), sample.intensity, support=sample.support
    )

    frame = torch.as_tensor(result.intensity)[None, None]
    target = torch.as_tensor(sample.intensity)[None]
    support = torch.as_tensor(sample.support)[None].float()
    objective, l1, ee, ncc = objective_terms(
        frame,
        target,
        support,
        w_ee=cfg.w_ee,
        metric=cfg.metric,
        prepare=cfg.prepare,
    )

    assert result.objective == pytest.approx(float(objective), rel=1e-5)
    assert result.l1 == pytest.approx(float(l1), rel=1e-5)
    assert result.ee == pytest.approx(float(ee), rel=1e-5)
    assert result.ncc == pytest.approx(float(ncc), rel=1e-5)


@pytest.mark.parametrize("trust_region", ["clip", "tanh"])
def test_trust_region_bounds_large_adam_steps(trust_region: str) -> None:
    """Both mappings enforce the radius when Adam tries to overshoot it."""
    sim = Sim2f(SHAPE, device="cpu")
    sample = _square(seed=5)
    radius = 0.1
    cfg = RefineConfig(
        iters=3,
        lr=0.5,
        max_delta_rad=radius,
        trust_region=trust_region,
    )

    result = PhaseRefiner(sim, cfg).finetune(
        _flat_phase(), sample.intensity, support=sample.support
    )

    assert result.accepted is True
    assert result.delta_max_rad > 0.9 * radius
    assert result.delta_max_rad <= radius + 1e-6
    assert np.abs(result.phase - result.phase_init).max() <= radius + 1e-6


def test_rejection_returns_the_input_phase_bit_identically() -> None:
    """An update below the required gain cannot leak into a hardware command."""
    sim = Sim2f(SHAPE, device="cpu")
    sample = _square()
    phase_init = _flat_phase()
    cfg = RefineConfig(iters=10, min_gain=10.0)

    result = PhaseRefiner(sim, cfg).finetune(
        phase_init, sample.intensity, support=sample.support
    )

    assert result.accepted is False
    assert np.array_equal(result.phase, phase_init)
    assert result.delta_max_rad == 0.0


def test_nonflat_starting_structure_stays_inside_local_radius() -> None:
    """Fine-tuning changes, rather than replaces, an existing smooth phase."""
    sim = Sim2f(SHAPE, device="cpu")
    sample = _square()
    yy, xx = np.mgrid[: SHAPE.grid, : SHAPE.grid]
    phase_init = (
        0.3 * np.sin(2.0 * np.pi * xx / SHAPE.grid)
        + 0.2 * np.cos(2.0 * np.pi * yy / SHAPE.grid)
    ).astype(np.float32)
    radius = 0.1
    cfg = RefineConfig(iters=10, lr=0.1, max_delta_rad=radius)

    result = PhaseRefiner(sim, cfg).finetune(
        phase_init, sample.intensity, support=sample.support
    )
    update = result.phase - result.phase_init

    assert result.accepted is True
    assert np.any(update != 0.0)
    assert np.abs(update).max() <= radius + 1e-6


def test_update_sigma_suppresses_high_spatial_frequencies() -> None:
    """Gaussian update blur produces a clearly smoother local correction."""
    sim = Sim2f(SHAPE, device="cpu")
    sample = _square()
    phase_init = _flat_phase()
    common = {"iters": 20, "lr": 0.02, "max_delta_rad": 0.5}

    rough = PhaseRefiner(sim, RefineConfig(**common, update_sigma_px=0.0)).finetune(
        phase_init, sample.intensity, support=sample.support
    )
    smooth = PhaseRefiner(sim, RefineConfig(**common, update_sigma_px=2.0)).finetune(
        phase_init, sample.intensity, support=sample.support
    )
    rough_energy = _high_frequency_energy(rough.phase - rough.phase_init)
    smooth_energy = _high_frequency_energy(smooth.phase - smooth.phase_init)

    assert rough.accepted is True
    assert smooth.accepted is True
    assert rough_energy > 0.0
    assert smooth_energy < 0.5 * rough_energy


def test_last_iterate_improves_with_keep_best_disabled() -> None:
    """A gentle decaying schedule leaves its final iterate better than flat."""
    sim = Sim2f(SHAPE, device="cpu")
    sample = _square()
    cfg = RefineConfig(
        iters=20,
        lr=0.005,
        lr_final_frac=0.05,
        keep_best=False,
    )

    result = PhaseRefiner(sim, cfg).finetune(
        _flat_phase(), sample.intensity, support=sample.support
    )

    assert isinstance(result, RefineResult)
    assert result.phase.shape == (SHAPE.grid, SHAPE.grid)
    assert result.accepted is True
    assert result.objective < result.objective_init


def test_bad_phase_and_target_inputs_are_rejected() -> None:
    """Wrong pupil/camera grids and an energy-free target fail loudly."""
    sim = Sim2f(SHAPE, device="cpu")
    sample = _square()
    refiner = PhaseRefiner(sim, RefineConfig(iters=1))

    with pytest.raises(ValueError, match="phase_init: expected"):
        refiner.finetune(
            np.zeros((SHAPE.grid - 1, SHAPE.grid), dtype=np.float32),
            sample.intensity,
        )
    with pytest.raises(ValueError, match="target: expected"):
        refiner.finetune(
            _flat_phase(),
            np.zeros((SHAPE.ccd_px - 1, SHAPE.ccd_px), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="target has no energy"):
        refiner.finetune(
            _flat_phase(),
            np.zeros((SHAPE.ccd_px, SHAPE.ccd_px), dtype=np.float32),
        )
