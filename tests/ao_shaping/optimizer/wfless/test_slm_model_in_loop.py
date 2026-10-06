"""Offline contract tests for the SLM model-in-the-loop runner.

WHAT is pinned
--------------
1. ``SlmModelInLoopConfig.__post_init__`` -- every invariant it actually
   enforces, with the exact message each violation raises, plus the focal-length
   derivation from the measured tilt-shift scale.
2. The coupled-parameter invariants the docstring *promises* but the code does
   **not** enforce, pinned as explicit latent-bug canaries (see
   ``TestUnvalidatedInvariants``).
3. ``trust_region_clamp`` and ``acceptance_verdict`` -- the two pure guards that
   decide whether a round is committed or rolled back.
4. Frame preprocessing, the model-grid resample and the probe-phase generator,
   because all three silently corrupt the physics when they are wrong.
5. The composite quality score, and -- deliberately -- the fact that the legacy
   ``compute_quality_score`` this runner replaced is *saturated* and must never
   be used for the bake-off.
6. The CLI and the library entry point, which must resolve to one and the same
   config object.
7. One full simulated optimisation, asserting the stage/round/accept schema of
   the history rows.

WHY
---
This runner replaces a hand-tuned per-round learning rate and a
magnitude-limited coefficient step with three pieces of machinery that are all
offline-verifiable: a calibrated bench model (a *bake-off*, so it can only be
trusted if it is scored before it is committed), a trust region on the
coefficient step, and an acceptance guard that rolls a bad round back. Each of
those has a documented hardware failure mode behind it, and each is cheap to
pin here.

These tests run with no hardware and no SDK: the bench is the in-repo simulated
backend and the forward model is torch on ``device="cpu"``.

One fact about the runtime budget shaped the layout of this file: importing
``slm_model_in_loop`` costs ~28 s, because its module-level import of
``slm_gs_refine`` (needed for ``_prepare_frame``) drags in the driver package.
The import-isolation probe needs a *clean* interpreter, so it necessarily pays
that cost a second time; it is therefore wrapped in ``lru_cache`` and shared by
every isolation test, making it 28 s rather than 28 s each. The remaining ~2 s is
the simulated end-to-end round.

An earlier revision ran the probe on a background thread to overlap it with the
parent's own import. That was reverted: two concurrent ~1.5 GB torch imports
thrash memory badly enough on this machine that the pair took over 400 s --
far worse than running them back to back.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import cast

# ---------------------------------------------------------------------------
# Import probe
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
"""Repository root.

``tests/ao_shaping/optimizer/wfless/test_slm_model_in_loop.py`` -- parents[0] is
``wfless``, [1] ``optimizer``, [2] ``ao_shaping``, [3] ``tests``, [4] the repo
root.
"""

#: Third-party namespaces that must never be imported by an ``ao_shaping`` import.
#: ``drivers/device_base._load_sdk()`` defers these, so seeing one in
#: ``sys.modules`` means the lazy-SDK contract has been broken.
_VENDOR_SDK_ROOTS = frozenset({"gxipy", "gxi", "MvCameraControl"})

_PROBE_SCRIPT = r"""
import json, sys

# A *clean* interpreter, because the property under test is what a fresh
# process actually sees when a user imports the runner.
before = set(sys.modules)
from ao_shaping.optimizer.wfless import slm_model_in_loop
added = set(sys.modules) - before

print("@@PROBE@@" + json.dumps({
    "target": slm_model_in_loop.__name__,
    "n_added": len(added),
    "n_drivers": len([m for m in added if m.startswith("ao_shaping.drivers")]),
    "bench_loaded": "ao_shaping.drivers.sim.slm_shaping_bench" in sys.modules,
    "sdk_loaded": sorted(
        m for m in added if m.split(".")[0] in {"gxipy", "gxi", "MvCameraControl"}
    ),
}))
"""

_probe_result: dict = {}


@lru_cache(maxsize=1)
def _probe() -> dict:
    """Run the import probe once per session and memoise the verdict.

    Costs ~28 s, so both isolation tests share this single call. Any failure is
    turned into a ``pytest.fail`` here rather than returned, so a broken probe
    can never look like a passing one.
    """
    if _probe_result:
        return _probe_result

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), str(REPO_ROOT / "libs"), env.get("PYTHONPATH", "")]
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        pytest.fail("import probe timed out after 600 s")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"import probe failed: {type(exc).__name__}: {exc}")

    for line in proc.stdout.splitlines():
        if line.startswith("@@PROBE@@"):
            _probe_result.update(json.loads(line[len("@@PROBE@@") :]))
            return _probe_result

    pytest.fail(
        "import probe produced no marker "
        f"(rc={proc.returncode})\nstdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}"
    )


# --- module imports --------------------------------------------------------

import numpy as np
import pytest

from ao_shaping.drivers.sim.slm_shaping_bench import composite_from_pib_cv
from ao_shaping.optimizer.wfless.slm_gs_refine import (
    _TILT_SHIFT_SCALE_MEASURED_AT_NM,
    _TILT_SHIFT_SCALE_PX,
    _prepare_frame,
)
from ao_shaping.utils.wavefront import focal_length_from_camera_pixel
from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    AcceptanceVerdict,
    ModelInLoopStatus,
    SlmModelInLoopConfig,
    _probe_phase,
    _quality,
    _to_model_grid,
    acceptance_verdict,
    optimize_slm_model_in_loop,
    trust_region_clamp,
)
from ao_shaping.runners.runner_common import SlmModelInLoopParams
from ao_shaping.runners.slm.model_in_loop_runner import run
from ao_shaping.utils.image.beam_metrics import compute_quality_score


# ---------------------------------------------------------------------------
# 1. Config validation -- what __post_init__ actually enforces
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """The invariants that must hold before any hardware opens.

    Each exists because the violated assumption produced a *silently bad run*
    rather than a crash: too few probes leaves Adam in a random stationary point
    of a non-convex fit, a near-flat probe pupil makes the fit blind, and a
    non-positive damping factor silently disables or inverts the rejection
    response.
    """

    def test_n_rounds_must_be_at_least_one(self) -> None:
        """Zero rounds would calibrate the bench and then never shape anything."""
        with pytest.raises(ValueError, match="n_rounds must be >= 1"):
            SlmModelInLoopConfig(n_rounds=0)

    def test_probe_count_must_be_at_least_two(self) -> None:
        """One probe is not a degenerate case, it is a broken configuration."""
        with pytest.raises(ValueError, match="probe_count must be >= 2"):
            SlmModelInLoopConfig(probe_count=1)

    def test_probe_spread_must_exceed_three_radians(self) -> None:
        """A near-flat pupil's focus barely responds to a smooth aberration."""
        with pytest.raises(ValueError, match="probe_spread must be >= 3.0"):
            SlmModelInLoopConfig(probe_spread=2.9)

    @pytest.mark.parametrize(
        "build",
        [
            lambda: SlmModelInLoopConfig(step_a_iterations=0),
            lambda: SlmModelInLoopConfig(step_b_iterations=0),
        ],
        ids=["step_a_iterations", "step_b_iterations"],
    )
    def test_step_iterations_must_be_at_least_one(
        self, build: Callable[[], SlmModelInLoopConfig]
    ) -> None:
        """Zero iterations in either step would commit an unshaped phase."""
        with pytest.raises(ValueError, match="must be >= 1"):
            build()

    @pytest.mark.parametrize("factor", [0.0, 1.0, -0.1, 1.5])
    def test_damp_factor_must_be_a_strict_fraction(self, factor: float) -> None:
        """0 disables damping and 1 never damps; both defeat the rejection path."""
        with pytest.raises(ValueError, match="damp_lr_factor must lie in"):
            SlmModelInLoopConfig(damp_lr_factor=factor)

    def test_target_side_must_be_at_least_one(self) -> None:
        """A zero-width target box has zero area and no gradient."""
        with pytest.raises(ValueError, match="target_side must be >= 1"):
            SlmModelInLoopConfig(target_side=0)

    def test_region_must_be_at_least_two(self) -> None:
        """A 1x1 pupil grid cannot support a Zernike fit."""
        with pytest.raises(ValueError, match="region must be >= 2"):
            SlmModelInLoopConfig(region=1)

    def test_n_orders_must_be_at_least_one(self) -> None:
        """Zero Zernike orders leaves nothing to fit."""
        with pytest.raises(ValueError, match="n_orders must be >= 1"):
            SlmModelInLoopConfig(n_orders=0)

    def test_far_field_padding_must_be_at_least_one(self) -> None:
        """Padding is the far-field scale anchor; zero has no meaning."""
        with pytest.raises(ValueError, match="far_field_padding must be >= 1"):
            SlmModelInLoopConfig(far_field_padding=0)

    def test_panel_span_must_be_positive(self) -> None:
        """The model grid samples the pupil at panel_span_px/region * SLM pitch."""
        with pytest.raises(ValueError, match="panel_span_px must be > 0"):
            SlmModelInLoopConfig(panel_span_px=0.0)

    def test_frozen_modes_are_normalised_to_a_sorted_unique_tuple(self) -> None:
        """Frozen modes are 1-based Noll indices held at zero for the whole fit.

        Normalisation happens in ``__post_init__`` so the CLI and the library
        path cannot disagree about which modes were held.
        """
        config = SlmModelInLoopConfig(
            frozen_modes=cast("tuple[int, ...]", [3, 1, 2, 1])
        )
        assert config.frozen_modes == (1, 2, 3)
        assert isinstance(config.frozen_modes, tuple)


class TestUnvalidatedInvariants:
    """Invariants the docstring *promises* but ``__post_init__`` never checks.

    These are canaries, not endorsements. Each asserts the **current permissive
    behaviour** so that adding the missing validation turns the test red and
    forces the assertion to be rewritten into a ``pytest.raises``. That is the
    intended signal: the hardware failure each one guards against is documented
    in the config's own docstrings.
    """

    def test_config_is_mutable_after_validation(self) -> None:
        """LATENT BUG: the config is ``@dataclass``, not ``frozen=True``.

        It cannot be frozen as written, because ``__post_init__`` assigns
        ``frozen_modes`` and ``focal_length_m`` -- but that also means every
        invariant checked above can be bypassed with one attribute write after
        construction, which is exactly what a runner that mutates ``config``
        between the CLI parse and the optimizer call would do.
        """
        config = SlmModelInLoopConfig()
        config.n_rounds = 0  # accepted, even though __post_init__ rejects it
        assert config.n_rounds == 0

    def test_efficiency_weight_is_not_required_to_be_non_zero(self) -> None:
        """LATENT BUG: ``w_efficiency`` docstring says it "must stay non-zero".

        Optimising ``-CV`` alone empties the target box -- hardware measured
        encircled energy collapsing to 0.002 -- and nothing stops that config
        being constructed and run.
        """
        config = SlmModelInLoopConfig(w_efficiency=0.0, w_uniformity=0.4)
        assert config.w_efficiency == 0.0

    def test_weights_are_not_required_to_sum_to_one(self) -> None:
        """LATENT BUG: the composite is documented as a *weighted mean*.

        Unnormalised weights silently rescale the score, and because the score
        is compared against thresholds and against the flat baseline, a
        mis-scaled score changes which rounds are accepted.
        """
        config = SlmModelInLoopConfig(w_efficiency=0.6, w_uniformity=0.6)
        assert config.w_efficiency + config.w_uniformity == pytest.approx(1.2)

    def test_target_side_has_no_upper_bound(self) -> None:
        """LATENT BUG: nothing ties the target box to the far-field size.

        A target wider than the analytic far field can never be filled, so every
        round is rejected for a reason that looks like a physics failure.
        """
        config = SlmModelInLoopConfig(target_side=10_000)
        assert config.target_side == 10_000

    def test_frozen_modes_accept_zero_and_negative_noll_indices(self) -> None:
        """LATENT BUG: Noll indices are 1-based, so 0 and negatives are typos.

        ``int(m)`` is applied without a sign check, so an index that does not
        exist is silently accepted and simply never matches.
        """
        assert SlmModelInLoopConfig(frozen_modes=(1, 0, -2)).frozen_modes == (-2, 0, 1)

    def test_frozen_modes_silently_truncate_floats(self) -> None:
        """LATENT BUG: ``int(m)`` truncates rather than rejecting a fractional index."""
        config = SlmModelInLoopConfig(
            frozen_modes=cast("tuple[int, ...]", (1.9,))
        )
        assert config.frozen_modes == (1,)

    def test_camera_pixel_pitch_is_not_validated(self) -> None:
        """LATENT BUG: a zero pitch reaches the focal-length derivation unchecked.

        ``camera_pixel_um`` is only read when ``focal_length_m <= 0``, so pinning
        the focal length lets a nonsense pitch through unchallenged.
        """
        config = SlmModelInLoopConfig(camera_pixel_um=0.0, focal_length_m=0.125)
        assert config.camera_pixel_um == 0.0


# ---------------------------------------------------------------------------
# 2. Focal length resolution
# ---------------------------------------------------------------------------


class TestFocalLengthResolution:
    """``focal_length_m=0`` derives the 2f focal length from the measured scale.

    The measured tilt-shift scale fixes only the ratio ``f / p_cam``, so exactly
    one external anchor is unavoidable; the CCD pixel pitch is used because it
    is a datasheet constant rather than a bench assembly choice. Resolution
    happens once in ``__post_init__`` so the runner, a library caller and these
    tests can never disagree about what geometry a run used.
    """

    def test_zero_focal_length_is_derived_from_the_measured_scale(self) -> None:
        """The derivation is the measured one, not a hard-coded magic number."""
        config = SlmModelInLoopConfig(focal_length_m=0.0)
        expected = focal_length_from_camera_pixel(
            wavelength_nm=float(config.slm_wavelength),
            camera_pixel_um=config.camera_pixel_um,
            slm_pixel_um=config.slm_pixel_um,
            focal_scale_px=_TILT_SHIFT_SCALE_PX,
            scale_measured_at_nm=_TILT_SHIFT_SCALE_MEASURED_AT_NM,
        )
        assert config.focal_length_m == pytest.approx(expected)
        assert config.focal_length_m > 0.0

    def test_explicit_focal_length_is_pinned_verbatim(self) -> None:
        """An explicit value is a datasheet anchor and must not be overwritten."""
        config = SlmModelInLoopConfig(focal_length_m=0.125)
        assert config.focal_length_m == pytest.approx(0.125)

    def test_camera_pixel_pitch_scales_the_derived_focal_length(self) -> None:
        """f is proportional to p_cam, so doubling the pitch doubles f."""
        wide = SlmModelInLoopConfig(focal_length_m=0.0, camera_pixel_um=4.4)
        narrow = SlmModelInLoopConfig(focal_length_m=0.0, camera_pixel_um=2.2)
        assert wide.focal_length_m == pytest.approx(2.0 * narrow.focal_length_m)


# ---------------------------------------------------------------------------
# 3. Trust region and acceptance guard
# ---------------------------------------------------------------------------


class TestTrustRegionClamp:
    """The per-round coefficient change is capped in L2 norm.

    Step A and Step B chase each other: without a bound, the aberration fit
    progressively absorbs the shaping phase. Clamping the *step* keeps the two
    estimators coupled but not co-adapting.
    """

    def test_step_inside_the_region_is_untouched(self) -> None:
        """A small step passes through unchanged and reports no clamp."""
        reference = np.zeros(10, dtype=np.float64)
        candidate = reference + 0.01
        clamped, step_l2, was_clamped = trust_region_clamp(candidate, reference, 0.5)
        assert was_clamped is False
        assert step_l2 == pytest.approx(0.031622776601683793)
        np.testing.assert_array_equal(clamped, candidate)

    def test_step_beyond_the_region_lands_exactly_on_the_sphere(self) -> None:
        """An over-long step is rescaled onto the boundary, not truncated."""
        reference = np.zeros(4, dtype=np.float64)
        candidate = np.full(4, 10.0, dtype=np.float64)
        clamped, step_l2, was_clamped = trust_region_clamp(candidate, reference, 0.5)
        assert was_clamped is True
        assert step_l2 == pytest.approx(20.0)  # the *unclamped* norm is reported
        assert float(np.linalg.norm(clamped - reference)) == pytest.approx(0.5)

    def test_clamping_scales_the_step_not_the_vector(self) -> None:
        """Clamping the vector would drag coefficients toward the origin.

        This is why the helper is written the way it is, so it is pinned: the
        clamped result must stay near ``reference``, not shrink toward zero.
        """
        reference = np.full(4, 3.0, dtype=np.float64)
        candidate = np.full(4, 10.0, dtype=np.float64)
        clamped, _, was_clamped = trust_region_clamp(candidate, reference, 0.5)
        assert was_clamped is True
        assert float(np.linalg.norm(clamped)) > float(np.linalg.norm(reference))

    def test_zero_radius_disables_the_guard(self) -> None:
        """``max_l2 <= 0`` is the documented opt-out."""
        reference = np.zeros(4, dtype=np.float64)
        candidate = np.full(4, 10.0, dtype=np.float64)
        clamped, step_l2, was_clamped = trust_region_clamp(candidate, reference, 0.0)
        assert was_clamped is False
        assert step_l2 == pytest.approx(20.0)
        np.testing.assert_array_equal(clamped, candidate)

    def test_zero_length_step_is_not_a_clamp(self) -> None:
        """A zero step must not report a clamp, or the history log lies."""
        reference = np.ones(4, dtype=np.float64)
        clamped, step_l2, was_clamped = trust_region_clamp(reference, reference, 0.5)
        assert was_clamped is False
        assert step_l2 == pytest.approx(0.0)

    def test_mismatched_shapes_are_rejected(self) -> None:
        """A shape mismatch is a caller bug, not something to broadcast."""
        with pytest.raises(ValueError, match="candidate shape"):
            trust_region_clamp(np.zeros(4), np.zeros(5), 0.5)

    def test_input_is_not_mutated(self) -> None:
        """The caller keeps using its own candidate; the helper returns a copy."""
        reference = np.zeros(3, dtype=np.float64)
        candidate = np.full(3, 9.0, dtype=np.float64)
        trust_region_clamp(candidate, reference, 0.1)
        np.testing.assert_array_equal(candidate, np.full(3, 9.0))


class TestAcceptanceVerdict:
    """A round is committed only if it is not a regression on either axis.

    Both axes are needed. The score alone would accept a round that improves EE
    while the uniformity collapses; the loss alone would accept a round whose
    Step-A fit improves but whose realised spot gets worse.
    """

    def test_clear_improvement_is_accepted(self) -> None:
        """Better on both axes commits."""
        verdict = acceptance_verdict(1.0, 0.5, 0.4, 0.8)
        assert verdict.accepted is True
        assert verdict.reason == "accepted"
        assert bool(verdict) is True

    def test_worse_score_is_rejected_even_with_a_better_loss(self) -> None:
        """The score is the quantity the bake-off actually compares."""
        verdict = acceptance_verdict(1.0, 0.5, 0.8, 0.4)
        assert verdict.accepted is False
        assert verdict.reason == "score_degraded"
        assert verdict.score_degraded is True
        assert verdict.loss_worsened is False

    def test_worse_loss_is_rejected_even_with_a_better_score(self) -> None:
        """A better score on a worse fit is the overfitting failure mode."""
        verdict = acceptance_verdict(1.0, 2.0, 0.4, 0.8)
        assert verdict.accepted is False
        assert verdict.reason == "loss_worsened"

    def test_both_axes_can_fail_at_once(self) -> None:
        """The reason tag joins both, so the log distinguishes the combinations."""
        verdict = acceptance_verdict(1.0, 2.0, 0.9, 0.1)
        assert verdict.accepted is False
        assert verdict.reason == "loss_worsened+score_degraded"

    def test_tolerated_regression_is_accepted(self) -> None:
        """Inside the configured windows a regression is noise, not a regression."""
        verdict = acceptance_verdict(
            1.0,
            1.0 + 1e-9,
            0.5,
            0.5 - 1e-9,
            loss_delta=1e-4,
            score_eps=1e-3,
        )
        assert verdict.accepted is True

    @pytest.mark.parametrize(
        "loss_before,loss_after,score_after",
        [
            (float("nan"), 0.5, 0.5),
            (1.0, float("nan"), 0.5),
            (1.0, 0.5, float("nan")),
            (float("inf"), 0.5, 0.5),
        ],
    )
    def test_non_finite_measurements_are_rejected(
        self, loss_before: float, loss_after: float, score_after: float
    ) -> None:
        """Absence of evidence is not evidence of absence: refuse, do not commit."""
        verdict = acceptance_verdict(loss_before, loss_after, 0.4, score_after)
        assert verdict.accepted is False
        assert verdict.reason == "non_finite"

    def test_first_round_skips_the_score_test(self) -> None:
        """``score_before=None`` means "nothing accepted yet", not "reject all".

        The flat baseline was never optimised for this target, so its encircled
        energy is high by accident; comparing round one against it would reject
        every genuine attempt.
        """
        verdict = acceptance_verdict(1.0, 0.5, None, 0.01)
        assert verdict.accepted is True
        assert verdict.score_degraded is False

    def test_non_finite_score_before_falls_back_to_the_loss_test(self) -> None:
        """A NaN incumbent score must not reject on the score axis."""
        verdict = acceptance_verdict(1.0, 0.5, float("nan"), 0.01)
        assert verdict.score_degraded is False
        assert verdict.accepted is True

    def test_verdict_is_a_frozen_value_object(self) -> None:
        """The verdict is returned to callers and must not be mutable after the fact."""
        verdict = AcceptanceVerdict(True, False, False, "accepted")
        # ``setattr`` because the point of the test is that the attribute is
        # read-only; a plain assignment would be rejected statically *and*
        # dynamically, which is exactly the pair of facts being pinned.
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(verdict, "accepted", False)


# ---------------------------------------------------------------------------
# 4. Frame / grid / probe primitives
# ---------------------------------------------------------------------------


class TestPrepareFrame:
    """Background subtraction before any ratio metric.

    A CCD frame carries symmetric read noise, so roughly half its pixels are
    negative and ``PIB`` -- which divides the in-target sum by the whole-frame
    sum -- exceeds 1 (measured 1.0120 on a sim frame with 7164/14400 negative
    pixels). Any optimiser driving on that signal is chasing read noise.
    """

    def test_output_is_non_negative(self) -> None:
        """Clipping after subtraction is what makes PIB <= 1 and CV finite."""
        frame = np.array([[-5.0, 0.0], [1.0, 2.0]], dtype=np.float64)
        out = _prepare_frame(frame)
        assert float(out.min()) >= 0.0

    def test_output_is_contiguous_and_floating(self) -> None:
        """The resampler downstream requires a contiguous float array."""
        out = _prepare_frame(np.arange(16, dtype=np.float64).reshape(4, 4))
        assert out.flags["C_CONTIGUOUS"]
        assert np.issubdtype(out.dtype, np.floating)

    def test_floating_input_preserves_its_precision(self) -> None:
        """float32 in, float32 out -- no silent upcast or downcast."""
        assert _prepare_frame(np.zeros((4, 4), dtype=np.float32)).dtype == np.float32
        assert _prepare_frame(np.zeros((4, 4), dtype=np.float64)).dtype == np.float64

    def test_integer_input_is_promoted_to_float(self) -> None:
        """An integer detector frame cannot be background-subtracted in place."""
        out = _prepare_frame(np.arange(16, dtype=np.uint8).reshape(4, 4))
        assert np.issubdtype(out.dtype, np.floating)

    def test_non_finite_pixels_are_zeroed_before_the_median(self) -> None:
        """NaNs would otherwise poison the median that estimates the background."""
        frame = np.array([[1.0, np.nan], [1.0, 1.0]], dtype=np.float64)
        out = _prepare_frame(frame)
        assert np.isfinite(out).all()

    def test_flat_field_collapses_to_zero(self) -> None:
        """A uniform frame has no beam, so nothing may survive the background."""
        out = _prepare_frame(np.full((8, 8), 17.0, dtype=np.float64))
        np.testing.assert_array_equal(out, np.zeros((8, 8)))


class TestToModelGrid:
    """The far-field frame is resampled onto the forward model's own grid.

    Step A's loss compares the model's prediction with the measurement, so both
    must live on the same grid. The camera oversamples the far field relative to
    the model, so this crops a ``region * scale`` window on the frozen optical
    axis and reduces it to ``region``.
    """

    def test_integer_scale_uses_a_block_average(self) -> None:
        """Extra camera pixels are a finer sampling of one cell, not new samples."""
        frame = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
        out = _to_model_grid(frame, (32, 32), 16, 4.0)
        assert out.shape == (16, 16)
        assert out.dtype == np.float64
        assert out.flags["C_CONTIGUOUS"]

    def test_unit_scale_is_a_plain_crop(self) -> None:
        """scale == 1 means the camera already samples on the model grid."""
        frame = np.arange(32 * 32, dtype=np.float64).reshape(32, 32)
        out = _to_model_grid(frame, (16, 16), 32, 1.0)
        assert out.shape == (32, 32)

    def test_non_integer_scale_falls_back_to_interpolation(self) -> None:
        """A real bench's geometry rarely lands on an integer scale."""
        frame = np.arange(96 * 96, dtype=np.float64).reshape(96, 96)
        out = _to_model_grid(frame, (48, 48), 24, 3.3)
        assert out.shape == (24, 24)
        assert np.isfinite(out).all()

    def test_output_is_always_region_squared(self) -> None:
        """The model grid size comes from ``region``, never from the camera."""
        for scale in (1.0, 2.0, 2.5, 4.0):
            frame = np.zeros((128, 128), dtype=np.float64)
            assert _to_model_grid(frame, (64, 64), 8, scale).shape == (8, 8)

    def test_centre_off_frame_is_zero_padded_not_an_error(self) -> None:
        """A stale centre must not crash the run or silently shift the grid."""
        frame = np.ones((64, 64), dtype=np.float64)
        out = _to_model_grid(frame, (2, 2), 16, 2.0)
        assert out.shape == (16, 16)
        assert float(out.max()) == pytest.approx(1.0)
        assert float(out.min()) == pytest.approx(0.0)

    def test_non_positive_scale_is_rejected(self) -> None:
        """A zero or negative scale has no meaning and would divide by zero."""
        with pytest.raises(ValueError, match="camera_px_per_model_px must be > 0"):
            _to_model_grid(np.zeros((16, 16)), (8, 8), 8, 0.0)


class TestProbePhase:
    """The Step-A probes must be strong and reproducible.

    Two probes is the documented minimum: a single probe leaves Adam in
    whichever stationary point of the non-convex intensity fit it entered.
    """

    def test_shape_and_dtype(self) -> None:
        """A probe is a pupil-grid phase in radians."""
        probe = _probe_phase(region=32, spread=4.0, seed=0)
        assert probe.shape == (32, 32)
        assert np.issubdtype(probe.dtype, np.floating)

    def test_same_seed_is_bit_identical(self) -> None:
        """Reproducibility is what makes a rejected round diagnosable."""
        a = _probe_phase(region=32, spread=4.0, seed=7)
        b = _probe_phase(region=32, spread=4.0, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_probes(self) -> None:
        """Otherwise the probe cycle is not a cycle."""
        a = _probe_phase(region=32, spread=4.0, seed=1)
        b = _probe_phase(region=32, spread=4.0, seed=2)
        assert not np.array_equal(a, b)

    def test_spread_is_actually_realised(self) -> None:
        """A probe that is too flat makes the fit blind; pin the realised sigma."""
        probe = _probe_phase(region=64, spread=4.0, seed=0)
        assert float(np.std(probe)) == pytest.approx(4.0, rel=0.35)

    def test_probe_is_finite(self) -> None:
        """A NaN probe silently blinds the fit rather than raising."""
        assert np.isfinite(_probe_phase(region=32, spread=4.0, seed=0)).all()


# ---------------------------------------------------------------------------
# 5. Quality score
# ---------------------------------------------------------------------------


def _metrics(cv: float, ee: float) -> dict:
    """A square-metrics dict shaped like the real ``_metrics_at`` output."""
    return {"uniformity_cv": cv, "encircled_energy": ee, "aspect_ratio": 1.0}


class TestQualityScore:
    """The composite score is EE + 1/(1+CV); the legacy one must stay unused.

    The legacy ``compute_quality_score`` scores uniformity as
    ``exp(-((CV/0.3)**2))``, which underflows to ~6e-5 at the CV ~0.9 this task
    reaches and to exactly 0 at the CV ~11 of an unshaped focus. Both score 0 on
    uniformity, so the composite collapses to ``0.3*aspect + 0.3*EE`` and ranks a
    near-delta focus **above** a genuinely uniform square -- the metric
    saturation trap in its purest form.
    """

    def test_composite_matches_the_documented_formula(self) -> None:
        """score = w_ee * EE + w_cv / (1 + CV)."""
        assert _quality({"encircled_energy": 0.8, "uniformity_cv": 0.9}) == (
            pytest.approx(0.6 * 0.8 + 0.4 / 1.9)
        )

    def test_uniform_square_beats_an_unshaped_focus(self) -> None:
        """The reason the legacy score is banned: the ordering inverts."""
        shaped = _quality({"encircled_energy": 0.87, "uniformity_cv": 0.9})
        unshaped = _quality({"encircled_energy": 0.99, "uniformity_cv": 11.0})
        assert shaped > unshaped

    def test_the_saturating_kernel_it_replaced_and_the_ranking_it_breaks(self) -> None:
        """Why this runner has its own ``_quality``: the kernel it replaced was wrong.

        The retired kernel was ``f_uni = exp(-((CV/0.3)**2))``, i.e. a saturating
        exponential. On this measured pair it gave the uniform square **0.5610**
        and the unshaped focus **0.5970** -- ranking the unshaped focus *above*
        the uniform square, because CV=11 underflows that kernel to exactly 0 and
        an unfocused spot is naturally concentrated (EE=0.99). That inversion is
        what made the first bake-off discard the correct phase and keep the flat
        field.

        The canonical ``compute_quality_score`` no longer has this defect -- its
        kernels are now the non-saturating ``1/(1+|AR-1|)`` and ``1/(1+CV)``, so
        both scores are pinned here as a regression check that the trap cannot
        reopen. This test no longer asserts the saturation exists; it asserts the
        ordering is correct, and records the retired numbers as the reason.
        """
        # Retired saturating kernel, reproduced locally so the failure mode stays
        # visible without depending on `compute_quality_score` regressing to it.
        def legacy(cv: float, ee: float) -> float:
            f_uni = math.exp(-((cv / 0.3) ** 2))
            return 0.3 * 1.0 + 0.4 * f_uni + 0.3 * ee

        assert legacy(cv=0.9, ee=0.87) == pytest.approx(0.5610493639216347)
        assert legacy(cv=11.0, ee=0.99) == pytest.approx(0.3 * 1.0 + 0.3 * 0.99)
        assert legacy(cv=11.0, ee=0.99) > legacy(cv=0.9, ee=0.87)  # the inversion

        # Canonical, as shipped now: same inputs, ordering restored.
        shaped = compute_quality_score(_metrics(cv=0.9, ee=0.87))
        unshaped = compute_quality_score(_metrics(cv=11.0, ee=0.99))
        assert shaped > unshaped
        # 0.3*1/(1+0) + 0.4/(1+0.9) + 0.3*0.87 = 0.3 + 0.21053 + 0.261
        assert shaped == pytest.approx(0.3 + 0.4 / 1.9 + 0.3 * 0.87)
        # 0.3 + 0.4/12 + 0.3*0.99
        assert unshaped == pytest.approx(0.3 + 0.4 / 12.0 + 0.3 * 0.99)

    def test_missing_cv_is_treated_as_uniformity_failure(self) -> None:
        """Absent CV must score 0 on the CV axis, not raise or default to 0."""
        assert _quality({"encircled_energy": 0.8}) == pytest.approx(0.6 * 0.8)

    def test_missing_ee_is_treated_as_zero_efficiency(self) -> None:
        """Absent EE must not default to 1.0 and fake a perfect score."""
        assert _quality({"uniformity_cv": 0.0}) == pytest.approx(0.4)

    def test_weights_are_honoured(self) -> None:
        """The caller-supplied weights reach the composite unchanged."""
        assert _quality({"encircled_energy": 1.0, "uniformity_cv": 0.0}, 1.0, 0.0) == (
            pytest.approx(1.0)
        )

    def test_score_decreases_monotonically_with_cv(self) -> None:
        """A strictly worse CV must never improve the score."""
        scores = [
            _quality({"encircled_energy": 0.8, "uniformity_cv": cv})
            for cv in (0.5, 2.0, 8.0)
        ]
        assert scores == sorted(scores, reverse=True)

    def test_runner_reuses_the_canonical_bench_composite(self) -> None:
        """The runner must not fork the composite; it reuses the bench helper."""
        assert composite_from_pib_cv(0.8, 0.9, w_pib=0.6, w_unif=0.4) == (
            pytest.approx(0.6 * 0.8 + 0.4 / 1.9)
        )


# ---------------------------------------------------------------------------
# 6. CLI / library parity
# ---------------------------------------------------------------------------


class TestCliLibraryParity:
    """The Click command and the library function must agree exactly.

    A drift here means the documented CLI invocation silently runs different
    physics than the tested library path -- the worst possible failure mode,
    because both look healthy.

    The parity is currently **NOT exact**, and the gaps are pinned below as
    latent findings rather than papered over:

    * ``sim_aberration`` has no CLI option, so a simulated run driven from the
      command line silently uses an empty aberration and cannot reproduce the
      library path.
    * ``output_subdir``, ``progress_every`` and ``callback`` are library-only.
    * ``save_best_image`` is a CLI-only field with no config counterpart.
    * The two field *orders* differ, so the two dataclasses must only ever be
      bridged by keyword.
    """

    #: Config fields with no ``SlmModelInLoopParams`` counterpart.
    LIBRARY_ONLY_FIELDS = {"sim_aberration", "output_subdir", "progress_every", "callback"}

    #: Param fields with no ``SlmModelInLoopConfig`` counterpart.
    CLI_ONLY_FIELDS = {"save_best_image"}

    def test_shared_field_names_are_a_valid_mapping(self) -> None:
        """Every shared name must exist on both sides -- a typo would TypeError."""
        config_fields = {f.name for f in dataclasses.fields(SlmModelInLoopConfig)}
        param_fields = set(SlmModelInLoopParams.__dataclass_fields__)
        shared = config_fields & param_fields
        assert shared, "no shared fields at all -- the mapping is broken"
        assert shared <= config_fields
        assert shared <= param_fields

    def test_no_cli_option_exists_for_the_simulated_aberration(self) -> None:
        """LATENT BUG: the CLI cannot reproduce a simulated library run.

        ``sim_aberration`` is how the simulated bench injects a known
        aberration, so its absence from the CLI means ``slm-model-in-loop`` on
        the simulated backend always runs with an empty aberration.
        """
        param_fields = set(SlmModelInLoopParams.__dataclass_fields__)
        assert "sim_aberration" not in param_fields
        assert "sim_aberration" in self.LIBRARY_ONLY_FIELDS

    def test_library_only_and_cli_only_fields_are_disjoint(self) -> None:
        """A name cannot be both unreachable from the CLI and CLI-only."""
        assert not (self.LIBRARY_ONLY_FIELDS & self.CLI_ONLY_FIELDS)

    def test_field_order_does_not_match_so_construction_must_be_keyword(self) -> None:
        """LATENT BUG: positional bridging would silently shuffle values.

        ``dtype`` sits at index 3 in the config but index 25 in the params, so
        ``SlmModelInLoopConfig(*params_values)`` would assign the wrong physics
        to every field from position 3 onward without raising.
        """
        config_order = [f.name for f in dataclasses.fields(SlmModelInLoopConfig)]
        param_order = list(SlmModelInLoopParams.__dataclass_fields__)
        assert config_order != param_order
        assert param_order.index("dtype") != config_order.index("dtype")

    def test_cli_entry_point_is_callable(self) -> None:
        """A renamed or re-signatured Click command must fail here, not on the bench."""
        assert callable(run)


# ---------------------------------------------------------------------------
# 7. Import isolation
# ---------------------------------------------------------------------------


def _module_level_imports() -> set[str]:
    """Return the absolute module names in the runner's **module body**.

    Only ``Module.body`` statements are walked, never function bodies. That
    distinction is the whole point: ``slm_gs_refine`` does
    ``from ao_shaping.drivers.ccd.common import create_camera`` *inside* a
    function (line 623), which is the deferred pattern the project requires, so
    a function-local import must not be flagged here. Reading the AST rather than
    ``sys.modules`` is what lets this check distinguish the two.
    """
    import ast

    import ao_shaping.optimizer.wfless.slm_model_in_loop as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


class TestImportIsolation:
    """Importing the runner must not load a native SDK or touch a device.

    WHAT IS PINNED
    --------------
    Two separable claims, deliberately tested by two different mechanisms because
    one subprocess cannot measure both:

    1. **No vendor SDK is loaded** -- asserted in a *clean* child interpreter
       (``subprocess``, ~28 s). This is the enforceable hardware-safety contract
       from ``drivers/AGENTS.md``: package import must not load native DLLs.
    2. **The runner's own module body imports no driver** -- asserted
       statically from its AST (free). This is the code-hygiene contract, and it
       is what makes claim 1 sufficient: the runner adds no driver *of its own*.

    LATENT FINDING (measured, not papered over)
    --------------------------------------------
    A clean import of this runner loads **62** ``ao_shaping.drivers.*`` modules,
    which violates the "package import is side-effect free" spirit of the lazy
    contract. The attribution is exact: the runner imports
    ``slm_gs_refine`` for ``_prepare_frame``, and that module's line 70 does
    ``from ao_shaping.drivers.sim.slm_shaping_bench import ...`` at module level,
    which in turn runs ``drivers.ccd.common``'s ``register_camera`` side effect.
    Fixing it needs a source change in ``slm_gs_refine``, so it is reported, not
    patched here. The ``*_transitively`` canary below pins the current behaviour
    so that fixing it turns this file red.
    """

    def test_import_loads_no_vendor_sdk_module(self) -> None:
        """The hard contract: no ``gxipy`` / ``gxi`` / ``MvCameraControl`` in sys.modules."""
        payload = _probe()
        assert payload["sdk_loaded"] == [], (
            "importing the runner loaded a native vendor SDK module: "
            f"{payload['sdk_loaded']}"
        )

    def test_runner_module_body_imports_no_driver_directly(self) -> None:
        """The runner's own top-level imports contain no ``ao_shaping.drivers``."""
        names = _module_level_imports()
        offenders = sorted(
            name
            for name in names
            if name.startswith("ao_shaping.drivers")
            or name.split(".")[0] in _VENDOR_SDK_ROOTS
        )
        assert offenders == [], (
            "the runner's module body imports driver/SDK modules directly: "
            f"{offenders}"
        )

    def test_declared_module_level_dependencies_are_the_documented_ones(self) -> None:
        """Pin the dependency set so a refactor cannot silently widen the graph.

        ``slm_gs_refine`` is the module that drags in the driver package (see the
        class docstring); naming it here is what makes the attribution auditable.

        The second name is ``bench_kernels``, not the older ``slm_bench_probe``:
        commit ``54d9613`` merged that module and five siblings into two kernels, and
        this assertion had not been updated. A stale member of a ``>=`` set does not
        fail -- the set only grows -- so the pin quietly stopped pinning anything, and
        the whole file stopped importing. Kept explicit here because the failure mode of
        an ``>=`` pin is silence, not an error.
        """
        assert _module_level_imports() >= {
            "ao_shaping.optimizer.wfless.slm_gs_refine",
            "ao_shaping.tools.slm.bench_kernels",
        }

    def test_driver_modules_arrive_transitively_via_slm_gs_refine(self) -> None:
        """LATENT FINDING canary: the clean import is *not* driver-free.

        Measured 62 driver modules. This is the eager-import defect described in
        the class docstring. When someone defers ``slm_gs_refine``'s bench import
        and the graph goes clean, ``n_drivers`` drops to 0 and this test fails,
        which is the intended prompt to rewrite it as an assertion of 0.
        """
        payload = _probe()
        assert payload["n_drivers"] > 0, (
            "the runner's import graph is now driver-free -- good, but this "
            "canary must be rewritten to assert the fixed behaviour"
        )
        assert payload["bench_loaded"] is True, (
            "the driver modules should arrive via slm_gs_refine's line-70 "
            "import of ao_shaping.drivers.sim.slm_shaping_bench"
        )

    def test_the_import_really_happened(self) -> None:
        """Guards against a vacuous pass if the probe silently imported nothing."""
        payload = _probe()
        assert payload["target"] == "ao_shaping.optimizer.wfless.slm_model_in_loop"
        assert payload["n_added"] > 0


# ---------------------------------------------------------------------------
# 8. End-to-end simulated optimisation
# ---------------------------------------------------------------------------

_SIM_KWARGS: dict = {
    "n_rounds": 2,
    "seed": 0,
    "device": "cpu",
    "dtype": "float64",
    "target_side": 24,
    "probe_count": 2,
    "n_calibration_probes": 2,
    "step_a_iterations": 3,
    "step_b_iterations": 3,
    "region": 32,
    "cam_type": "sim",
    "cam_size": 128,
    "far_field_padding": 1,
    "panel_span_px": 8.0,
    "settle_discard": 0,
    "n_eval_frames": 1,
    "max_rejection_streak": 3,
    "sim_aberration": {},
}

#: Keys on *every* history row -- the calibration baseline and the rounds alike.
_SHARED_KEYS = {
    "stage",
    "round",
    # The Recorder's required index key. `utils.io.file.save_recorder_debug_artifacts`
    # does `data[int(rec["_epoch"])] = item`, so a row without it raises KeyError from
    # inside the pkl path -- which is precisely why this runner's `--debug` artefacts
    # could never be written. It is shared by both schemas because both are recorded.
    "_epoch",
    "accepted",
    "reason",
    "score_before",
    "score_after",
    "cv_after",
    "ee_after",
    "uniformity_cv",
    "coeff_norm",
    "coeff_step_l2",
    "clamp_applied",
    "probe_count",
    "step_a_lr",
    "rejection_streak",
}

#: Calibration-baseline only. These are one-shot facts about the Stage-1
#: geometry solve, which runs once before the loop and so has no per-round value.
_CALIBRATION_ONLY_KEYS = {"geometry_correlation", "n_calibration_probes"}

#: Round only. The Step-A/Step-B cost ledger, which is meaningless on the
#: baseline row because no fit has happened yet when that row is written.
_ROUND_ONLY_KEYS = {
    "loss_before",
    "loss_after",
    "step_a_iterations",
    "step_b_iterations",
    "coeff_step_applied",
    "shape_loss_initial",
    "shape_loss_final",
}

_CALIBRATION_KEYS = _SHARED_KEYS | _CALIBRATION_ONLY_KEYS
_ROUND_KEYS = _SHARED_KEYS | _ROUND_ONLY_KEYS

_sim_cache: dict = {}


def _sim_result() -> dict:
    """Run the tiny simulated optimisation once and memoise it.

    The simulated round costs a few seconds and a dozen assertions need it, so
    it runs once per module. A deep copy is handed back so no test can mutate
    the shared result for the next one.
    """
    if "value" not in _sim_cache:
        result = optimize_slm_model_in_loop(SlmModelInLoopConfig(**_SIM_KWARGS))
        _sim_cache["value"] = {"result": result, "history": result.recorder.history}
    return copy.deepcopy(_sim_cache["value"])


class TestSimEndToEnd:
    """One full offline run: calibrate, then fit -> shape -> re-measure twice.

    This is the only test that exercises the actual loop, and it is the one that
    catches a regression in the *ordering* of the guards -- a rollback that
    stops rolling back, a calibration that stops gating, a phase that stops being
    committed on acceptance.
    """

    def test_history_has_one_calibration_row_plus_one_row_per_round(self) -> None:
        """The baseline is recorded before the loop, then each round appends."""
        assert len(_sim_result()["history"]) == _SIM_KWARGS["n_rounds"] + 1

    def test_calibration_row_is_the_pre_loop_baseline(self) -> None:
        """Row 0 is the flat baseline: accepted, no fit, zero coefficient norm."""
        row = _sim_result()["history"][0]
        assert row["stage"] == "calibration"
        assert row["round"] == 0
        assert row["accepted"] is True
        assert row["reason"] == "baseline"
        assert row["coeff_norm"] == 0.0
        assert row["probe_count"] == 0

    def test_simulated_bench_reports_zero_collected_calibration_probes(self) -> None:
        """The simulated bench short-circuits calibration, so the count is 0.

        ``_calibrate_geometry`` returns ``(bench.true_geometry(), [])`` for a
        ``_SimBench`` -- the true geometry is known analytically, so no speckle
        probes are displayed. The row therefore records
        ``n_calibration_probes == 0`` even though ``config.n_calibration_probes``
        is 2. This is intentional and documented in the helper's docstring
        ("empty for the simulated bench"), and it is pinned here because the
        alternative -- a reporter that trusts this field as "probes displayed"
        -- silently under-counts every simulated run.
        """
        row = _sim_result()["history"][0]
        assert row["n_calibration_probes"] == 0
        assert _SIM_KWARGS["n_calibration_probes"] == 2

    def test_round_rows_are_labelled_round_with_a_one_based_index(self) -> None:
        """Round rows carry ``stage="round"`` and a 1-based ``round`` counter.

        They are not the same thing: ``stage`` is a category string, ``round`` is
        the index. A reporter that conflates them mis-buckets every row.
        """
        rounds = _sim_result()["history"][1:]
        assert [row["stage"] for row in rounds] == ["round"] * len(rounds)
        assert [row["round"] for row in rounds] == list(range(1, len(rounds) + 1))

    def test_round_rows_carry_the_full_step_a_and_step_b_record(self) -> None:
        """Both optimisation steps are logged per round, not just the outcome."""
        for row in _sim_result()["history"][1:]:
            assert not (_ROUND_KEYS - set(row)), sorted(_ROUND_KEYS - set(row))

    def test_history_uses_two_distinct_row_schemas(self) -> None:
        """The two row schemas are disjoint in both directions, by design.

        The calibration row has no Step-A/Step-B fields (no fit has run when it is
        written) and the round rows have no Stage-1 fields (the geometry solve
        happened once). A reporter that assumes one flat schema KeyErrors here,
        which is exactly the coupling this pins shut.
        """
        history = _sim_result()["history"]
        assert not _ROUND_ONLY_KEYS & set(history[0])
        assert not _CALIBRATION_ONLY_KEYS & set(history[1])
        assert _CALIBRATION_ONLY_KEYS and _ROUND_ONLY_KEYS
        assert not _CALIBRATION_ONLY_KEYS & _ROUND_ONLY_KEYS

    def test_both_row_schemas_are_pinned_exactly(self) -> None:
        """Exact key sets, so an added or dropped column cannot pass unnoticed."""
        history = _sim_result()["history"]
        assert set(history[0]) == _CALIBRATION_KEYS
        for row in history[1:]:
            assert set(row) == _ROUND_KEYS

    def test_history_rows_are_flat_scalars_only(self) -> None:
        """Every value must survive a CSV/DataFrame round trip.

        The runner keeps phase arrays out of the rows entirely and records only
        scalars, so a reporter never has to special-case a rejected round.
        """
        for row in _sim_result()["history"]:
            for key, value in row.items():
                assert isinstance(
                    value, (bool, int, float, str)
                ), f"{key} is {type(value).__name__}, not a scalar"

    def test_acceptance_flags_are_real_booleans(self) -> None:
        """``accepted`` gates a coefficient commit; a truthy int would not do."""
        for row in _sim_result()["history"]:
            assert isinstance(row["accepted"], bool)
            assert isinstance(row["clamp_applied"], bool)

    def test_trust_region_bounds_the_applied_step(self) -> None:
        """The applied L2 can never exceed the radius, and never exceeds the raw."""
        radius = float(SlmModelInLoopConfig().trust_region_c_l2)
        for row in _sim_result()["history"][1:]:
            assert row["coeff_step_applied"] <= radius + 1e-9
            assert row["coeff_step_applied"] <= row["coeff_step_l2"] + 1e-9

    def test_rejection_streak_counts_consecutive_rejections(self) -> None:
        """The streak recorded on a row is the count *before* that round."""
        streak = 0
        for row in _sim_result()["history"][1:]:
            assert row["rejection_streak"] == streak
            streak = 0 if row["accepted"] else streak + 1

    def test_every_rejection_names_a_machine_readable_reason(self) -> None:
        """A rejection with an empty reason cannot be triaged from the log."""
        for row in _sim_result()["history"][1:]:
            if not row["accepted"]:
                assert row["reason"] and isinstance(row["reason"], str)

    def test_baseline_score_is_self_consistent(self) -> None:
        """The bake-off needs the flat score recorded before and after identically."""
        row = _sim_result()["history"][0]
        assert row["score_before"] == pytest.approx(row["score_after"])

    def test_status_is_a_known_terminal_state(self) -> None:
        """An unrecognised status means the run neither shaped nor reported why."""
        status = _sim_result()["result"].status
        assert status in {
            ModelInLoopStatus.COMPLETED,
            ModelInLoopStatus.ABORTED_UNIDENTIFIABLE,
            ModelInLoopStatus.ABORTED_REJECTION_STREAK,
            ModelInLoopStatus.BAKE_OFF_REJECTED,
        }

    def test_best_phase_is_always_finite_and_safely_writable(self) -> None:
        """``best_phase`` falls back to flat, so it must never be NaN or empty."""
        phase = np.asarray(_sim_result()["result"].best_phase)
        assert phase.ndim == 2
        assert np.isfinite(phase).all()

    def test_target_side_is_converted_into_model_pixels(self) -> None:
        """The model only sees its own grid, so the side must be converted down."""
        result = _sim_result()["result"]
        assert result.target_side_camera == _SIM_KWARGS["target_side"]
        assert 0 < int(result.target_side_model) < _SIM_KWARGS["region"]

    def test_result_carries_the_config_it_ran_with(self) -> None:
        """Provenance: the resolved config must come back on the result."""
        result = _sim_result()["result"].config
        assert isinstance(result, SlmModelInLoopConfig)
        assert result.n_rounds == _SIM_KWARGS["n_rounds"]

    def test_geometry_correlation_is_reported(self) -> None:
        """The bake-off gate reads this number, so it must be on the row."""
        assert math.isfinite(_sim_result()["history"][0]["geometry_correlation"])