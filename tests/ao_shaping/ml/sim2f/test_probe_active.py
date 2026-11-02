"""Tests for :mod:`ml.sim2f.probe_active` -- posterior algebra, selection, recovery.

Three kinds of test, separated on purpose:

* **algebra** on a hand-built design (does ``linear_posterior`` shrink the trace
  monotonically, does ``select_probe`` return the mode that actually reduces the
  trace most, is a tie broken deterministically);
* **recovery** on a tiny ideal bench (``grid=32``, ``ccd=64``): a planted
  single-mode aberration comes back and the posterior never grows;
* **validation**: a mismatched shape or a duplicated candidate raises instead of
  silently returning something plausible.

The tiny bench is ideal on purpose.  The device used by
``scripts/ml/sim2f_probe_active.py`` carries unmodelled ``OpticsErrors``; a test
that reproduced those numbers would be measuring this machine's CUDA build, not
the estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.sim2f.config import AberrationConfig, Sim2fConfig
from ml.sim2f.fit import FitConfig
from ml.sim2f.optics import Sim2f
from ml.sim2f.probe_active import (
    PosteriorEstimate,
    linear_posterior,
    probe_active,
    select_probe,
)


def _posterior(
    cov: np.ndarray, *, gram: np.ndarray | None = None, sigma2: float = 1.0
) -> PosteriorEstimate:
    """A posterior with the given covariance and a unit-amplitude gram."""
    cov = np.asarray(cov, dtype=np.float64)
    n_modes = cov.shape[0]
    return PosteriorEstimate(
        coeffs=np.zeros(n_modes, dtype=np.float64),
        cov=cov,
        sigma2=sigma2,
        rank=n_modes,
        condition=1.0,
        n_measurements=0,
        gram=np.eye(n_modes, dtype=np.float64) if gram is None else gram,
    )


def test_trace_decreases_with_measurements() -> None:
    """Every added probe block lowers ``trace(cov)``; it never grows."""
    rng = np.random.default_rng(11)
    n_modes, n_pixels = 3, 400
    design = rng.normal(size=(n_pixels, n_modes))
    truth = np.array([0.30, -0.20, 0.10])
    residual = design @ truth + rng.normal(scale=0.05, size=n_pixels)
    prior = np.eye(n_modes) * 1.5**2

    traces = []
    for count in (1, 2, 3):
        block = design[: count * (n_pixels // 3)]
        values = residual[: count * (n_pixels // 3)]
        estimate = linear_posterior(
            block, values, prior_cov=prior, n_measurements=count
        )
        assert estimate.coeffs.shape == (n_modes,)
        assert estimate.confidence().shape == (n_modes,)
        assert estimate.gram is not None and estimate.gram.shape == (n_modes, n_modes)
        traces.append(estimate.trace())

    assert traces[0] > traces[1] > traces[2]
    assert np.allclose(
        linear_posterior(design, residual).cov, linear_posterior(design, residual).cov.T
    )


def test_select_probe_picks_largest_variance_reduction() -> None:
    """The greedy choice is the candidate with the biggest trace reduction."""
    cov = np.diag([0.80, 0.02])
    posterior = _posterior(cov)
    amplitude = 0.25
    sigma2 = posterior.sigma2

    def reduction(mode: int) -> float:
        direction = amplitude * posterior.gram[:, mode - 2]
        cov_direction = posterior.cov @ direction
        return float(cov_direction @ cov_direction) / float(
            sigma2 + direction @ cov_direction
        )

    chosen = select_probe(posterior, (2, 3), amplitude_rad=amplitude)
    assert chosen == 2  # the uncertain mode
    assert reduction(2) > reduction(3)

    flipped = select_probe(
        _posterior(np.diag([0.02, 0.80])), (2, 3), amplitude_rad=amplitude
    )
    assert flipped == 3

    # Equal variances: the tie goes to the smaller Noll index, deterministically.
    tied = _posterior(np.eye(2) * 0.25)
    assert select_probe(tied, (2, 3), amplitude_rad=amplitude) == 2
    assert select_probe(tied, (3, 2), amplitude_rad=amplitude) == 2


def test_probe_active_recovers_planted_mode() -> None:
    """A single-mode aberration is recovered on an ideal tiny bench."""
    cfg = Sim2fConfig(grid=64, ccd_px=64)
    bench = Sim2f(cfg)
    fit_cfg = FitConfig.from_aberration(AberrationConfig(n_modes=4))
    truth = np.zeros(4, dtype=np.float64)
    truth[2] = 0.40  # Noll 4
    truth_map = bench.aberration(truth[None])[0]

    def measure(phase: torch.Tensor) -> torch.Tensor:
        return bench.forward(phase, aberration=truth_map[None], detector=False)

    result = probe_active(
        bench,
        fit_cfg,
        measure=measure,
        budget=3,
        candidates=(2, 3, 4, 5),
        fit_cfg=fit_cfg,
    )

    assert len(result.selected_modes) == 3
    # Re-probing is allowed by design (a repeat still shrinks the posterior), so
    # the assertion is that the bank is explored and every pick is a candidate,
    # not that the three picks are distinct.
    assert set(result.selected_modes) <= {2, 3, 4, 5}
    assert len(set(result.selected_modes)) >= 2
    assert abs(result.coeffs[2] - 0.40) < 0.12
    assert np.all(np.abs(result.coeffs[[0, 1, 3]]) < 0.12)

    traces = [estimate.trace() for estimate in result.estimates]
    assert all(later < earlier for earlier, later in zip(traces, traces[1:]))
    assert result.final.trace() == pytest.approx(traces[-1])
    assert np.allclose(result.final.coeffs, result.coeffs)


def test_validation_errors() -> None:
    """A bad shape, a duplicate candidate and a bad amplitude all raise."""
    design = np.ones((10, 2))
    with pytest.raises(ValueError):
        linear_posterior(design, np.ones(9))  # residual/rows mismatch
    with pytest.raises(ValueError):
        linear_posterior(np.ones(10), np.ones(10))  # not 2-D
    with pytest.raises(ValueError):
        linear_posterior(design, np.ones(10), prior_cov=np.eye(3))

    posterior = _posterior(np.eye(2))
    with pytest.raises(ValueError):
        select_probe(posterior, (2, 2), amplitude_rad=0.3)  # duplicate candidate
    with pytest.raises(ValueError):
        select_probe(posterior, (), amplitude_rad=0.3)  # empty candidate list
    with pytest.raises(ValueError):
        select_probe(posterior, (2, 3), amplitude_rad=0.0)
    with pytest.raises(ValueError):
        select_probe(posterior, (2, 4), amplitude_rad=0.3)  # outside 2..3

    cfg = Sim2fConfig(grid=64, ccd_px=64)
    bench = Sim2f(cfg)
    fit_cfg = FitConfig.from_aberration(AberrationConfig(n_modes=4))

    def measure(phase: torch.Tensor) -> torch.Tensor:
        return bench.forward(phase, detector=False)

    with pytest.raises(ValueError):
        probe_active(
            bench,
            fit_cfg,
            measure=measure,
            budget=2,
            candidates=(2, 2),
            fit_cfg=fit_cfg,
        )
    with pytest.raises(ValueError):
        probe_active(
            bench,
            fit_cfg,
            measure=measure,
            budget=2,
            order=(4,),
            fit_cfg=fit_cfg,
        )
