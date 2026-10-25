"""Tests for ``scripts/ml/measure_sim2f_kl_basis.py``.

What is pinned, and why each is load-bearing
============================================
The script's answer ("do not wire a KL basis") rests on two things it *computes*
rather than asserts, so both are checked here:

1. :func:`capture` -- the variance-fraction definition.  If it were wrong, every
   Part 1 ratio would be wrong in the same direction and the decision rule would
   be evaluated on a bogus quantity.  It is compared against a brute-force
   ``trace(P S P) / trace(S)`` built by explicit matrix products, and the KL
   basis is required to beat the Noll basis *on its own covariance* (its
   optimality condition -- a self-consistency check, not a result).
2. :func:`build_response_general` -- the local generalisation that makes a
   combined (KL) push measurable at all, because
   :func:`ml.sim2f.fit.build_response` takes Noll *indices* and one-hot-encodes
   them internally.  The whole KL column of Part 2 is only comparable to the
   Noll column if this function reproduces the real probe path on the one-hot
   bank; that equality is what is asserted (the script's
   ``reduction_max_rel_diff`` field is the same check at run time).

No hardware and no detector: the bench is the bare float64 CPU
:class:`ml.sim2f.config.Sim2fConfig` with ``detector=False``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from ml.sim2f.config import AberrationConfig, Sim2fConfig
from ml.sim2f.fit import FitConfig, build_response, probe_noll
from scripts.ml.measure_sim2f_kl_basis import (
    _basis_kl,
    _basis_noll,
    _dense_vectors,
    build_response_general,
    capture,
)


def _bench():
    """The bench every test here measures on (bare, float64, CPU, no detector)."""
    from ml.sim2f.optics import Sim2f

    return Sim2f(Sim2fConfig(), device="cpu", dtype=torch.float64)


def _covariance() -> np.ndarray:
    """A small anisotropic covariance: diagonal ladder plus one strong pair."""
    diagonal = np.linspace(1.0, 0.02, 19)
    cov = np.diag(diagonal)
    cov[7, 8] = cov[8, 7] = 0.4 * np.sqrt(diagonal[7] * diagonal[8])
    return cov


def test_capture_matches_the_explicit_projector() -> None:
    """``capture`` equals ``trace(P S P)/trace(S)`` with ``P = B B^T`` written out."""
    cov = _covariance()
    basis = _basis_noll(5)
    projector = basis @ basis.T
    expected = float(np.trace(projector @ cov @ projector)) / float(np.trace(cov))
    residual = np.eye(19) - projector
    expected_residual = float(np.sqrt(np.trace(residual @ cov @ residual)))

    got = capture(basis, cov)

    assert got["fraction"] == pytest.approx(expected, rel=1e-12)
    assert got["residual_rad"] == pytest.approx(expected_residual, rel=1e-12)


def test_kl_beats_noll_on_its_own_covariance_at_every_partial_k() -> None:
    """The KL basis is variance-optimal for the covariance it was built from.

    This is the script's stated non-negotiability: it *must* win on E1, which is
    why that column is not evidence.  What the test pins is only that the
    measurement machinery has the property -- if it did not, ``capture`` or
    ``_basis_kl`` would be wrong.

    A tie is legitimate and is asserted as such: this synthetic covariance is
    diagonal except for one correlated pair at rows 7/8, and a rotation preserves
    the trace of any block it acts on.  So the gain appears only at the one ``K``
    whose window *splits* the pair (K=8: the strong member is in, the weak one
    out); a window that contains both, or neither, gains nothing.
    """
    cov = _covariance()
    for k in (2, 3, 5, 8, 12):
        noll = capture(_basis_noll(k), cov)["fraction"]
        kl = capture(_basis_kl(cov, k), cov)["fraction"]
        assert kl >= noll - 1e-12, f"K={k}: kl {kl} lost to noll {noll}"
        if k == 8:
            assert kl > noll, f"K={k}: the window splits the correlated pair, so it must gain"


def test_full_k_captures_everything_for_both_bases() -> None:
    """At K = 19 both bases span the space, so the variance argument is vacuous."""
    cov = _covariance()
    for family in (_basis_noll, _basis_kl):
        basis = family(19) if family is _basis_noll else family(cov, 19)
        got = capture(basis, cov)
        assert got["fraction"] == pytest.approx(1.0, abs=1e-12)
        assert got["residual_rad"] == pytest.approx(0.0, abs=1e-12)


def test_general_response_reduces_to_build_response_on_the_one_hot_bank() -> None:
    """The local generalisation reproduces the real probe path exactly.

    ``build_response`` cannot express a combined push -- its columns come from
    :func:`ml.sim2f.fit._unit_coefficient`'s one-hot vectors -- so the KL columns
    are built by ``build_response_general``.  That is only legitimate if the two
    agree on the bank the real function *can* express.
    """
    bench = _bench()
    base = FitConfig.from_aberration(AberrationConfig(n_modes=21))
    cfg = FitConfig(
        n_modes=base.n_modes,
        probe_modes=(4, 5, 6),
        probe_frozen=base.probe_frozen,
        fit_frozen=base.fit_frozen,
        probe_amplitude_rad=base.probe_amplitude_rad,
        iters=base.iters,
        lr=base.lr,
    )
    chosen = probe_noll(cfg)
    assert chosen == (4, 5, 6)

    def measure(phase: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return bench.forward(phase, detector=False)

    real = build_response(bench, cfg, measure)
    onehot = [np.eye(cfg.n_modes)[index - 2] for index in chosen]
    copy = build_response_general(bench, cfg, onehot, measure)

    scale = float(np.abs(real.matrix).max())
    assert scale > 0.0
    assert float(np.abs(copy["matrix"] - real.matrix).max()) / scale < 1e-12
    assert copy["rank"] == real.rank
    assert copy["cond"] == pytest.approx(real.condition, rel=1e-12)


def test_a_combined_push_is_not_a_one_hot_push() -> None:
    """A dense coefficient vector gives a response no one-hot bank can produce.

    Guards the reason ``build_response_general`` exists at all: if a mixture
    behaved like a single mode, the whole Part 2 comparison would be moot.
    """
    bench = _bench()
    cfg = FitConfig.from_aberration(AberrationConfig(n_modes=21))

    def measure(phase: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return bench.forward(phase, detector=False)

    onehot = [np.eye(cfg.n_modes)[4 - 2], np.eye(cfg.n_modes)[8 - 2]]
    # Unit-norm mixtures, so the rotation is orthogonal and the comparison is
    # about *which* phase is pushed, never about a different probe amplitude.
    mixed = [np.zeros(cfg.n_modes) for _ in range(2)]
    mixed[0][4 - 2] = 1.0 / np.sqrt(2.0)
    mixed[0][8 - 2] = 1.0 / np.sqrt(2.0)
    mixed[1][4 - 2] = 1.0 / np.sqrt(2.0)
    mixed[1][8 - 2] = -1.0 / np.sqrt(2.0)

    plain = build_response_general(bench, cfg, onehot, measure)
    combined = build_response_general(bench, cfg, mixed, measure)

    # A combined push really is the sum of its parts: the first column,
    # (Z4 + Z8)/sqrt(2), matches the same superposition of the two one-hot
    # columns.  (Not exactly -- a +/-0.35 rad secant of a two-mode sum is not the
    # sum of the two one-mode secants; the script reports that gap as
    # `cond_if_linear`.)  What matters here is that the generalised builder
    # produces a genuine mixture, which the Noll bank cannot express.
    superposed = (plain["matrix"][:, 0] + plain["matrix"][:, 1]) / np.sqrt(2.0)
    cosine = float(
        np.dot(combined["matrix"][:, 0], superposed)
        / (np.linalg.norm(combined["matrix"][:, 0]) * np.linalg.norm(superposed))
    )
    assert cosine > 0.999
    assert not np.allclose(combined["matrix"], plain["matrix"])
    assert combined["s_max"] == pytest.approx(plain["s_max"], rel=0.2)


def test_dense_vectors_keep_tip_and_tilt_at_zero() -> None:
    """A KL basis over the high-order block must not invent a tip/tilt push."""
    basis = _basis_kl(_covariance(), 3)
    vectors = _dense_vectors(basis, 21)
    assert len(vectors) == 3
    for vector in vectors:
        assert vector.shape == (21,)
        assert vector[0] == 0.0
        assert vector[1] == 0.0
        assert np.linalg.norm(vector) == pytest.approx(1.0, rel=1e-12)
