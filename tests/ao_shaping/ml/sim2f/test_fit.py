"""Tests for :mod:`ml.sim2f.fit` -- probe algebra, trust region, real-twin recovery.

Two kinds of test, deliberately separated:

* a **linear sensor stub** where the camera frame is an exact affine function of
  the Zernike coefficients (an aberration map goes through a fixed linear mixing
  operator).  The response matrix is then exact, so the assertions are about the
  *algebra* -- does the fit invert the matrix it measured, does the trust region
  hold, is a rank-deficient bank reported as a failure -- rather than about
  optics;
* one **real-twin integration test** on ``Sim2fConfig.calibrated()``, where the
  numbers are measurements (coefficient RMS error, cosine similarity, the noise
  ladder).  Its thresholds are the measured values with an order of magnitude of
  room, and the measured numbers are quoted where they are used.

The stub is linear on purpose, and that is *not* a model of the bench: the real
frame is quadratic in the phase over the interval a probe spans, which is exactly
why ``probe`` does not return the linear solve as its answer (see
:func:`ml.sim2f.fit.probe`).  A linear bench is what isolates the algebra from
that physics.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.sim2f.config import AberrationConfig, Sim2fConfig
from ml.sim2f.fit import (
    FitConfig,
    build_response,
    noise_ladder,
    probe,
    probe_noll,
    refine,
    refit_noll,
)

_MODES = 6  # Noll 2..7
_FIT_FIELDS_FROM_ABERRATION = (
    "n_modes",
    "probe_modes",
    "probe_frozen",
    "fit_frozen",
    "probe_amplitude_rad",
    "iters",
    "lr",
)


class LinearSensorBench:
    """Pupil phase in, camera frame out, through a fixed *linear* operator.

    ``phase`` and ``aberration`` are **added** (exactly as
    :meth:`ml.sim2f.optics.Sim2f.field` adds them), the sum is projected onto the
    Zernike basis by a fixed pseudo-inverse, and the coefficients are mixed into
    the camera frame: ``frame = 1 + M @ c_total``.  The pedestal keeps every pixel
    positive, so ``prepare_frame`` (median removal, clip at zero) is affine and the
    response matrix is the exact Jacobian.  Everything is torch, so autograd
    reaches the coefficients exactly as it does on the real bench.
    """

    def __init__(
        self, *, modes: int = _MODES, grid: int = 16, ccd: int = 16, seed: int = 0
    ) -> None:
        self.grid = grid
        self.ccd_px = ccd
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.n_modes = modes
        generator = np.random.default_rng(seed)
        pupil = generator.normal(size=(modes, grid, grid))
        pupil -= pupil.mean(axis=(1, 2), keepdims=True)
        pupil /= np.abs(pupil).max(axis=(1, 2), keepdims=True)
        self.pupil_basis = torch.as_tensor(pupil, dtype=self.dtype)
        # How a pupil phase is expressed in the basis (a real bench's Zernike
        # decomposition), and how the coefficients reach the sensor.
        self._pinv = torch.as_tensor(
            np.linalg.pinv(pupil.reshape(modes, -1)), dtype=self.dtype
        )
        mixer = generator.normal(size=(ccd * ccd, modes))
        mixer /= np.abs(mixer).sum(axis=1, keepdims=True)
        self.mixer = torch.as_tensor(mixer * 0.05, dtype=self.dtype)
        # A fixed structure in the *flat* frame, like the bench's own defocus /
        # 0-order background.  Without it the model frame at a flat command is a
        # constant, `prepare_frame` removes it, and the fit starts from an
        # identically-zero prediction -- where the projected gain is zero and so
        # is every gradient.  The real bench is never that flat.
        flat = generator.normal(size=ccd * ccd)
        flat -= flat.mean()
        self.flat_pattern = torch.as_tensor(flat / np.abs(flat).max() * 0.02, dtype=self.dtype)

    def aberration(self, coeffs) -> torch.Tensor:
        """``(B, K)`` radians -> ``(B, grid, grid)`` pupil map (differentiable)."""
        values = coeffs if torch.is_tensor(coeffs) else torch.as_tensor(coeffs)
        values = values.to(dtype=self.dtype)
        if values.dim() == 1:
            values = values[None]
        return torch.einsum("bk,kxy->bxy", values, self.pupil_basis)

    def forward(self, phase, *, aberration=None, illumination=None, detector=False, **kwargs):
        """Camera frames for a batch of pupil phases plus an optional aberration."""
        tensor = phase if torch.is_tensor(phase) else torch.as_tensor(phase)
        tensor = tensor.to(dtype=self.dtype)
        if tensor.dim() == 2:
            tensor = tensor[None]
        batch = int(tensor.shape[0])
        total = tensor
        if aberration is not None:
            extra = aberration if torch.is_tensor(aberration) else torch.as_tensor(aberration)
            total = total + extra.to(dtype=self.dtype)
        coefficients = total.reshape(batch, -1) @ self._pinv
        frame = 1.0 + self.flat_pattern + coefficients @ self.mixer.T
        return frame.reshape(batch, 1, self.ccd_px, self.ccd_px)


class DuplicateModeBench(LinearSensorBench):
    """Last two pupil modes are identical, so the bank has a null direction."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pupil_basis[-1] = self.pupil_basis[-2].clone()


def _cfg(**overrides) -> AberrationConfig:
    """A small probe contract: six coefficients, 120 Adam steps."""
    base = dict(n_modes=_MODES, probe_amplitude_rad=0.35, fit_iters=120, fit_lr=0.05)
    base.update(overrides)
    return AberrationConfig(**base)


def _command(bench: LinearSensorBench) -> torch.Tensor:
    """The probe / refit command (zeros; the stub ignores it, the bench does not)."""
    return torch.zeros(1, bench.grid, bench.grid)


def _measure(bench: LinearSensorBench, truth: np.ndarray):
    """A camera read with the given true coefficients behind it.

    The bench state is applied *on top of* whatever command is sent -- the same
    composition the real bench performs -- so a probe push and a closed-loop
    refit both see it.
    """

    def call(phase: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return bench.forward(phase, aberration=bench.aberration(truth[None]))

    return call


_TRUTH = np.array([0.2, -0.1, 0.3, 0.05, -0.25, 0.15])


class TestConfiguration:
    """The budget must come from the dataclasses, never from a literal here."""

    def test_aberration_fields_are_copied_not_defaulted(self):
        cfg = _cfg(fit_iters=37, fit_lr=0.02, probe_amplitude_rad=0.2, fit_frozen=(1, 2))
        fit_cfg = FitConfig.from_aberration(cfg)
        assert fit_cfg.iters == 37
        assert fit_cfg.lr == pytest.approx(0.02)
        assert fit_cfg.probe_amplitude_rad == pytest.approx(0.2)
        assert fit_cfg.fit_frozen == (1, 2)
        assert fit_cfg.n_modes == cfg.n_modes
        # Fields that also exist in AberrationConfig carry no default of their
        # own, so a parser literal cannot silently shadow the calibrated value.
        fields = FitConfig.__dataclass_fields__
        for name in _FIT_FIELDS_FROM_ABERRATION:
            assert fields[name].default is dataclasses.MISSING
            assert fields[name].default_factory is dataclasses.MISSING

    def test_unknown_override_raises(self):
        with pytest.raises(ValueError, match="unknown FitConfig field"):
            FitConfig.from_aberration(_cfg(), trust_radius=1.0)

    def test_probe_noll_defaults_to_high_order(self):
        # AberrationConfig documents "empty means the first modes after 4", and the
        # coefficient vector ends at n_modes + 1.
        assert probe_noll(_cfg()) == (4, 5, 6, 7)

    def test_probe_noll_honours_explicit_modes_and_freeze(self):
        cfg = _cfg(probe_modes=(4, 5, 6, 7), probe_frozen=(5,))
        assert probe_noll(cfg) == (4, 6, 7)

    def test_refit_noll_excludes_frozen(self):
        # The contract default freezes Noll 1-3 (piston + tip/tilt): once the
        # target decides where the energy goes, the spot position identifies
        # nothing about the bench's tilt.
        assert refit_noll(_cfg()) == (4, 5, 6, 7)
        assert refit_noll(_cfg(fit_frozen=())) == (2, 3, 4, 5, 6, 7)
        assert refit_noll(_cfg(fit_frozen=(1, 5))) == (2, 3, 4, 6, 7)


class TestProbeAlgebra:
    """On a linear sensor the response matrix is exact, so the algebra is testable."""

    def test_linear_step_tracks_the_measured_matrix(self):
        bench = LinearSensorBench()
        response = build_response(
            bench, FitConfig.from_aberration(_cfg()), _measure(bench, _TRUTH)
        )
        assert response.rank == len(response.noll) == 4
        assert response.condition < 1e4
        assert response.gain == pytest.approx(1.0, rel=0.1)
        # Probed modes come back in the right direction and magnitude.  It is not
        # exact and cannot be: `prepare_frame` clips half of any zero-mean frame,
        # so a secant measured on *prepared* frames is a biased derivative (the
        # same bias the real bench has, which is why `probe` does not return this
        # vector -- 3% here, 170% on the speckle bench).
        measured = np.array([response.linear[j - 2] for j in response.noll])
        reference = np.array([_TRUTH[j - 2] for j in response.noll])
        np.testing.assert_allclose(measured, reference, rtol=0.2, atol=0.01)
        assert response.linear[:2].tolist() == [0.0, 0.0]
        # Only about half the frame difference is explained: that is the clipping
        # bias, and the reason the linear step is evidence rather than the answer.
        assert response.linear_residual < 0.6

    def test_probe_returns_the_truth(self):
        bench = LinearSensorBench()
        result = probe(bench, _cfg(), measure=_measure(bench, _TRUTH))
        assert result.ok, result.reason
        assert result.rank == 4
        np.testing.assert_allclose(result.coeffs, _TRUTH, atol=0.02)
        # The probe stage fits tip/tilt too (only piston is frozen): on a flat spot
        # the spot shift *is* the tilt information.
        assert result.free_noll == (2, 3, 4, 5, 6, 7)
        assert len(result.history) == result.iterations == 120
        assert result.residual < 0.01 < result.residual_start

    def test_rank_deficient_bank_is_a_documented_failure(self):
        bench = DuplicateModeBench()
        result = probe(bench, _cfg(), measure=_measure(bench, _TRUTH))
        assert not result.ok
        assert "rank-deficient" in (result.reason or "")
        # Noll 6 and 7 share a pupil map, so the bank spans 3 of its 4 columns.
        assert result.rank == 3
        assert "7" in (result.reason or "")
        # The vector is the unchanged starting point, never a partial answer.
        np.testing.assert_allclose(result.coeffs, 0.0)

    def test_all_dark_frame_is_a_documented_failure(self):
        bench = LinearSensorBench()

        def dark(phase: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, 1, bench.ccd_px, bench.ccd_px)

        result = probe(bench, _cfg(), measure=dark)
        assert not result.ok
        assert "rank-deficient" in (result.reason or "")
        assert result.rank == 0
        np.testing.assert_allclose(result.coeffs, 0.0)

    def test_measured_frame_can_be_passed_explicitly(self):
        bench = LinearSensorBench()
        frame = _measure(bench, _TRUTH)(_command(bench))
        result = probe(
            bench, _cfg(), measure=_measure(bench, _TRUTH), measured=frame
        )
        assert result.ok, result.reason
        np.testing.assert_allclose(result.coeffs, _TRUTH, atol=0.02)


class TestRefine:
    """The closed-loop stage: frozen command, free aberration, bounded step."""

    def test_recovers_from_a_perturbed_start(self):
        bench = LinearSensorBench()
        frame = _measure(bench, _TRUTH)(_command(bench))
        start = _TRUTH + np.array([0.4, 0.0, -0.3, 0.1, 0.0, -0.2])
        # All modes free: with tip/tilt frozen the model could not reproduce a
        # frame that carries them and the other modes would absorb the difference
        # -- which is precisely why the loop's refit freezes them *and* commands
        # the correction separately.
        result = refine(bench, _command(bench), frame, start, cfg=_cfg(fit_frozen=()))
        assert result.ok, result.reason
        assert result.residual < result.residual_start
        np.testing.assert_allclose(result.coeffs, _TRUTH, atol=0.02)
        assert result.free_noll == (2, 3, 4, 5, 6, 7)

    def test_trust_region_bounds_every_step(self):
        bench = LinearSensorBench()
        truth = np.zeros(_MODES)
        frame = _measure(bench, truth)(_command(bench))
        # Start far away with an absurd learning rate: the box is the only thing
        # that can stop the step.
        fit_cfg = FitConfig.from_aberration(_cfg(fit_lr=50.0, fit_iters=3), trust_rad=0.05)
        start = np.full(_MODES, 0.9)
        result = refine(
            bench, _command(bench), frame, start, cfg=_cfg(), fit_cfg=fit_cfg
        )
        assert result.ok, result.reason
        assert np.abs(result.coeffs - start).max() <= 0.05 + 1e-6

    def test_frozen_modes_keep_their_incoming_value(self):
        bench = LinearSensorBench()
        frame = _measure(bench, _TRUTH)(_command(bench))
        start = np.array([0.5, -0.5, 0.0, 0.0, 0.0, 0.0])
        result = refine(
            bench, _command(bench), frame, start, cfg=_cfg(fit_frozen=(1, 2, 3))
        )
        assert result.ok, result.reason
        assert result.free_noll == (4, 5, 6, 7)
        # `fit_frozen=(1, 2, 3)` freezes Noll 1-3, i.e. rows 0-1 of the vector
        # (piston is not in it).  Frozen on entry, frozen on exit: the fit may not
        # move them even to make the residual smaller (the loop commands that part
        # separately), so Noll 4 is free and does move off its 0.0 start.
        np.testing.assert_allclose(result.coeffs[:2], start[:2])
        assert result.residual < result.residual_start

    def test_no_free_modes_is_a_documented_failure(self):
        bench = LinearSensorBench()
        frame = _measure(bench, np.zeros(_MODES))(_command(bench))
        cfg = _cfg(fit_frozen=tuple(range(1, _MODES + 2)))
        result = refine(bench, _command(bench), frame, np.zeros(_MODES), cfg=cfg)
        assert not result.ok
        assert "no free modes" in (result.reason or "")

    def test_empty_frame_is_a_documented_failure(self):
        bench = LinearSensorBench()
        empty = torch.zeros(1, 1, bench.ccd_px, bench.ccd_px)
        result = refine(
            bench, _command(bench), empty, np.zeros(_MODES), cfg=_cfg()
        )
        assert not result.ok
        assert "empty measurement" in (result.reason or "")

    def test_wrong_vector_length_raises(self):
        bench = LinearSensorBench()
        frame = _measure(bench, np.zeros(_MODES))(_command(bench))
        with pytest.raises(ValueError, match="coefficient"):
            refine(bench, _command(bench), frame, np.zeros(3), cfg=_cfg())


class TestDeterminism:
    """Same bench, same seed, same vector -- bit for bit."""

    def test_repeated_probe_matches(self):
        bench = LinearSensorBench()
        first = probe(bench, _cfg(), measure=_measure(bench, _TRUTH))
        second = probe(bench, _cfg(), measure=_measure(bench, _TRUTH))
        assert np.array_equal(first.coeffs, second.coeffs)
        assert first.history == second.history


# ----------------------------------------------------------------------
# Real twin
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def twin():
    """The calibrated 2f bench (one instance per module: it builds a Zernike basis)."""
    from ml.sim2f.optics import Sim2f

    return Sim2f(Sim2fConfig.calibrated(), device="cpu")


def _twin_state(twin, seed: int = 3, **state_overrides):
    """Draw one bench state and build its (aberration, illumination) tensors."""
    from ml.sim2f.optics import draw_aberration, write_aberration

    rng = np.random.default_rng(seed)
    state = draw_aberration(twin, AberrationConfig(**state_overrides), rng)
    aberration, illumination = write_aberration(twin, state)
    return state, aberration, illumination


def _twin_measure(twin, aberration, illumination=None, *, exposure=None, seed: int = 7):
    """Camera read for the twin; ``exposure=None`` switches the detector off."""
    generator = torch.Generator(device=twin.device).manual_seed(seed)

    def call(phase: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return twin.forward(
                phase,
                aberration=aberration,
                illumination=illumination,
                detector=exposure is not None,
                exposure_ms=exposure,
                generator=generator if exposure is not None else None,
            )

    return call


class TestTwinRecovery:
    """Acceptance: the fit recovers coefficients, and noise is what breaks it.

    Measured on this bench (one drawn state, high-order RMS 0.167 rad, detector
    off, flat probe command): coefficient RMS error 0.0046 rad, cosine 1.000,
    frame residual 0.39 -> 0.02 over 80 Adam steps.  A zero estimate would score
    0.167 rad and the inflated linear step ~0.35 rad, so these thresholds
    separate a working fit from both failure modes.
    """

    def test_probe_recovers_the_drawn_state(self, twin):
        # Nominal illumination on *both* sides: the estimator only knows what its
        # model contains, and the model has no illumination term (see the module
        # docstring).  The next test measures what that omission costs.
        state, aberration, _ = _twin_state(twin, seed=3)
        result = probe(
            twin, AberrationConfig(), measure=_twin_measure(twin, aberration)
        )
        assert result.ok, result.reason
        assert result.rank == 19
        assert result.linear is not None
        error = result.coeffs - state.coeffs
        rmse = float(np.sqrt(np.mean(error**2)))
        assert rmse < 0.05, f"coefficient RMS error {rmse:.4f} rad"
        assert result.residual < 0.1 < result.residual_start
        high = slice(2, None)
        cosine = float(
            result.coeffs[high] @ state.coeffs[high]
            / (np.linalg.norm(result.coeffs[high]) * np.linalg.norm(state.coeffs[high]))
        )
        assert cosine > 0.95

    def test_unmodelled_illumination_degrades_the_fit(self, twin):
        # Measured: nominal illumination 0.005 rad; the same estimator with the
        # state's 6% centre jitter left out of the model 0.17 rad -- i.e. the
        # estimate is dominated by that unmodelled term.  Pinned because it is
        # the stated reason `run_loop` reports both configurations.
        state, aberration, jittered = _twin_state(twin, seed=3, illum_jitter_frac=0.06)
        clean = probe(twin, AberrationConfig(), measure=_twin_measure(twin, aberration))
        noisy = probe(
            twin, AberrationConfig(), measure=_twin_measure(twin, aberration, jittered)
        )
        clean_err = float(np.sqrt(np.mean((clean.coeffs - state.coeffs) ** 2)))
        noisy_err = float(np.sqrt(np.mean((noisy.coeffs - state.coeffs) ** 2)))
        assert clean_err < 0.05
        assert noisy_err > 3.0 * clean_err

    def test_noise_ladder_falls_and_refuses_hopeless_rungs(self, twin):
        # Measured (this state): detector off 0.005 rad; 0.15 ms (9 counts peak)
        # refused; 5 ms 0.053; 20 ms 0.074.  The ladder makes "error falls as noise
        # falls" measurable -- and it also shows the *inversion* between 5 and
        # 20 ms, where a brighter frame fits its own residual better (0.11 vs 0.37)
        # but the coefficient error is no longer set by noise but by how weakly the
        # remaining directions are observed.  Reported, not smoothed over.
        state, _, _ = _twin_state(
            twin, seed=3, illum_jitter_frac=0.0, illum_width=(1.0, 1.0)
        )
        rungs = noise_ladder(
            twin,
            AberrationConfig(),
            coeffs=state.coeffs,
            exposures=(None, 0.15, 5.0, 20.0),
            seed=3,
        )
        assert [rung["exposure_ms"] for rung in rungs] == [None, 0.15, 5.0, 20.0]
        reference, hopeless, brighter, brightest = rungs
        assert reference["coeff_rmse"] is not None
        # The noise-free rung is the best one and the hopeless rung is refused.
        assert reference["coeff_rmse"] < 0.05
        assert hopeless["coeff_rmse"] is None
        assert hopeless["peak_counts"] < 20.0
        # Every measurable detector rung is worse than the noise-free one, which is
        # the direction the acceptance criterion claims; the ladder is monotone in
        # *residual*, not in coefficient error (see the docstring).
        assert brighter["coeff_rmse"] is not None
        assert brightest["coeff_rmse"] is not None
        assert min(brighter["coeff_rmse"], brightest["coeff_rmse"]) > reference["coeff_rmse"]
        assert brightest["residual"] < brighter["residual"]
        assert brightest["peak_counts"] > brighter["peak_counts"]

    def test_read_noise_axis_is_reported(self, twin):
        # The read-noise axis only exists once the detector is on, so this rung is
        # at a fixed exposure and the read noise is raised past the signal.
        state, _, _ = _twin_state(
            twin, seed=3, illum_jitter_frac=0.0, illum_width=(1.0, 1.0)
        )
        rungs = noise_ladder(
            twin,
            AberrationConfig(),
            coeffs=state.coeffs,
            exposures=(5.0,),
            read_sigmas=(None, 60.0),
            seed=3,
        )
        assert [rung["read_sigma_counts"] for rung in rungs] == [None, 60.0]
        assert rungs[0]["coeff_rmse"] is not None
        # 60 counts of read noise on a ~50-count frame is below the floor: refused,
        # not fitted.
        assert rungs[1]["coeff_rmse"] is None

