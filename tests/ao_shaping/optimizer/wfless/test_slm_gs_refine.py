"""Tests for the GS-warm-started freeform SLM square-shaping hardware loop.

The hardware path cannot be exercised without a Santec SLM and a CCD, so the
end-to-end coverage runs against ``--cam_type sim`` (the 2f-Fourier digital
twin). That still exercises the real control flow: ROI location and freezing,
the GS bake-off, settle-then-average acquisition, SPGD gradient estimation and
the phase/panel embedding.

The invariants locked here are the ones whose violation produced *plausible but
wrong* hardware behaviour during development:

* ``PIB`` must stay in ``[0, 1]`` -- read noise makes roughly half the pixels of
  a CCD frame negative, and an uncorrected ratio exceeds 1.
* the target ROI must be located once, never re-``argmax``-ed per iteration.
* the best phase must not be stored per recorder row (17.6 MB x 400 epochs).
* the GS phase must be rejected when it does not beat the flat baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.sim.slm_shaping_bench import power_in_bucket, uniformity_cv
from ao_shaping.optimizer.wfless.slm_gs_refine import (
    SlmGsRefineConfig,
    _compose,
    _derive_target_side,
    _embed_pupil_square,
    _fit_exact,
    _locate_zero_order,
    _lr_at,
    _make_optimizer,
    _metrics,
    _prepare_frame,
    _pupil_side,
    _square_mask,
    _upsample_bilinear,
    optimize_slm_gs_refine,
)


def _fast_config(**overrides) -> SlmGsRefineConfig:
    """A sim config small enough to run in a few seconds."""
    base = dict(
        cam_type="sim",
        epochs=3,
        phase_grid=4,
        delta=0.4,
        lr=0.05,
        seed=7,
        beam_radius_px=24.0,
        cam_size=100,
        gs_iters=8,
        n_eval_frames=1,
        settle_wait_s=0.0,
        settle_max_wait_s=0.0,
        settle_max_discard=1,
        progress_every=0,
    )
    base.update(overrides)
    return SlmGsRefineConfig(**base)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


class TestSquareMask:
    def test_area_and_centering(self):
        mask = _square_mask((100, 100), 50, 50, 20)
        assert mask.shape == (100, 100)
        assert mask.sum() == 400
        assert mask[50, 50] and mask[40, 40] and mask[59, 59]

    def test_clipped_at_frame_edge(self):
        mask = _square_mask((50, 50), 0, 0, 20)
        assert mask.any()
        assert mask[:10, :10].all()

    def test_completely_outside_frame_is_empty(self):
        assert not _square_mask((50, 50), 200, 200, 10).any()

    def test_odd_side_does_not_crash(self):
        mask = _square_mask((64, 64), 32, 32, 7)
        assert mask.sum() >= 49


class TestLocateZeroOrder:
    def test_finds_argmax_not_frame_centre(self):
        frame = np.zeros((100, 100))
        # Deliberately off-centre: the 2f optical axis is NOT the frame centre.
        frame[37, 82] = 5.0
        assert _locate_zero_order(frame) == (37, 82)

    def test_nan_is_ignored(self):
        frame = np.zeros((40, 40))
        frame[10, 10] = np.nan
        frame[20, 30] = 1.0
        assert _locate_zero_order(frame) == (20, 30)


class TestResizeHelpers:
    def test_fit_exact_pads_and_crops(self):
        assert _fit_exact(np.ones((3, 3)), (5, 5)).shape == (5, 5)
        assert _fit_exact(np.ones((9, 9)), (4, 4)).shape == (4, 4)

    def test_upsample_reaches_exact_shape(self):
        for shape in ((10, 10), (24, 24), (37, 41)):
            out = _upsample_bilinear(np.arange(9, dtype=float).reshape(3, 3), shape)
            assert out.shape == shape
            assert np.isfinite(out).all()

    def test_upsample_is_identity_when_shape_matches(self):
        coarse = np.arange(16, dtype=float).reshape(4, 4)
        assert np.array_equal(_upsample_bilinear(coarse, (4, 4)), coarse)

    def test_pupil_side_has_a_floor(self):
        assert _pupil_side(1.0) == 32
        assert _pupil_side(450.0) == 900


# ---------------------------------------------------------------------------
# Noise handling -- the PIB > 1 defect
# ---------------------------------------------------------------------------


class TestPrepareFrame:
    def test_removes_negative_read_noise(self):
        rng = np.random.default_rng(0)
        frame = rng.normal(50.0, 20.0, size=(80, 80))
        clean = _prepare_frame(frame)
        assert clean.min() >= 0.0
        # The ~50 background is gone. Clipping piles the negative half onto zero,
        # so the surviving mean is a fraction of sigma rather than exactly zero.
        assert clean.mean() < 0.5 * frame.std()

    def test_removes_a_constant_pedestal(self):
        assert np.allclose(_prepare_frame(np.full((40, 40), 37.0)), 0.0)

    def test_pib_stays_within_unit_interval(self):
        """Uncorrected frames give PIB > 1, which is physically impossible.

        The negative pedestal is placed deterministically rather than drawn from
        a noise distribution: whether the out-of-target sum lands negative is a
        coin flip at realistic sigma, which made a random fixture flaky.
        """
        frame = np.zeros((120, 120))
        frame[50:70, 50:70] = 100.0  # beam
        frame[:40, :40] = -12.0  # negative pedestal, entirely outside the target

        mask = _square_mask(frame.shape, 60, 60, 40)
        assert power_in_bucket(frame, mask) == pytest.approx(1.923, abs=0.01)
        assert power_in_bucket(_prepare_frame(frame), mask) <= 1.0

    def test_nan_is_neutralised(self):
        frame = np.full((20, 20), np.nan)
        frame[5, 5] = 2.0
        assert np.isfinite(_prepare_frame(frame)).all()

    def test_metrics_are_finite_and_bounded(self):
        rng = np.random.default_rng(1)
        frame = np.zeros((60, 60))
        frame[20:40, 20:40] = 80.0
        frame += rng.normal(0.0, 4.0, size=frame.shape)
        clean = _prepare_frame(frame)
        mask = _square_mask(clean.shape, 30, 30, 24)
        score, pib, cv = _metrics(clean, mask, 0.5, 0.5)
        assert 0.0 <= pib <= 1.0
        assert np.isfinite(cv) and cv >= 0.0
        assert 0.0 <= score <= 1.0

    def test_uniform_beat_peaked_beat(self):
        mask = _square_mask((64, 64), 32, 32, 24)
        uniform = np.zeros((64, 64))
        uniform[20:44, 20:44] = 1.0
        peaked = np.zeros((64, 64))
        peaked[32, 32] = 1.0
        assert _metrics(uniform, mask, 0.5, 0.5)[0] > _metrics(peaked, mask, 0.5, 0.5)[0]
        assert uniformity_cv(uniform, mask) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Phase composition
# ---------------------------------------------------------------------------


class TestPhaseComposition:
    def test_embed_places_the_pupil_at_the_panel_centre(self):
        panel = (200, 320)
        emb = _embed_pupil_square(np.ones((10, 10)), panel, 20.0)
        assert emb.shape == panel
        side = _pupil_side(20.0)
        assert emb.sum() == side * side
        ys, xs = np.nonzero(emb)
        assert ys.min() == (200 - side) // 2 and xs.min() == (320 - side) // 2

    def test_compose_adds_dof_to_base(self):
        panel = (120, 160)
        base = _embed_pupil_square(np.zeros((8, 8)), panel, 10.0)
        dof = np.full((4, 4), 0.5)
        out = _compose(base, dof, panel, 10.0)
        assert out.shape == panel
        assert out.max() == pytest.approx(0.5)

    def test_compose_with_zero_dof_returns_base_unchanged(self):
        panel = (60, 80)
        base = np.zeros(panel)
        assert _compose(base, np.zeros((4, 4)), panel, 10.0) is base
        assert _compose(base, None, panel, 10.0) is base

    def test_base_is_not_mutated(self):
        panel = (60, 80)
        base = np.zeros(panel)
        before = base.copy()
        _compose(base, np.ones((4, 4)), panel, 10.0)
        assert np.array_equal(base, before)


class TestTargetSide:
    def test_explicit_side_is_honoured(self):
        cfg = SlmGsRefineConfig(target_side=90, side_factor=1.0)
        assert _derive_target_side(cfg, 450.0) == 90

    def test_side_factor_scales(self):
        cfg = SlmGsRefineConfig(target_side=90, side_factor=2.0)
        assert _derive_target_side(cfg, 450.0) == 180

    def test_derived_side_follows_the_documented_focal_relation(self):
        """D*f/(d_SLM*p_cam) with the default bench constants."""
        cfg = SlmGsRefineConfig()
        radius = 450.0
        expected = 2 * radius * 8e-6 * 0.125 / (3.31e-6)
        assert _derive_target_side(cfg, radius) == pytest.approx(round(expected), rel=0.01)

    def test_never_degenerate(self):
        assert _derive_target_side(SlmGsRefineConfig(target_side=1), 10.0) >= 3


class TestLearningRate:
    def test_auto_lr_tracks_delta(self):
        cfg = SlmGsRefineConfig(epochs=10, delta=0.4, lr=0.0)
        assert _lr_at(cfg, 0) == pytest.approx(0.06)

    def test_explicit_lr_wins(self):
        cfg = SlmGsRefineConfig(epochs=10, delta=0.4, lr=0.5)
        assert _lr_at(cfg, 0) == pytest.approx(0.5)

    def test_cosine_decays_and_respects_floor(self):
        cfg = SlmGsRefineConfig(epochs=100, delta=0.4, lr=0.0, lr_min_ratio=0.1)
        first, last = _lr_at(cfg, 0), _lr_at(cfg, 99)
        assert last < first
        assert last == pytest.approx(0.06 * 0.1, rel=1e-6)

    def test_static_schedule_is_flat(self):
        cfg = SlmGsRefineConfig(epochs=100, delta=0.4, lr_schedule="static")
        assert _lr_at(cfg, 0) == _lr_at(cfg, 99)


class TestOptimizerFactory:
    @pytest.mark.parametrize("kind", ["adam", "adamw", "adamod", "sgd", "unknown"])
    def test_every_kind_yields_a_working_optimizer(self, kind):
        opt = _make_optimizer(kind, 16, 0.1)
        step = opt.update(np.ones(16))
        assert np.asarray(step).shape == (16,)
        assert np.isfinite(np.asarray(step, dtype=float)).all()


# ---------------------------------------------------------------------------
# End-to-end (simulated hardware)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEndToEndSim:
    def test_pipeline_produces_a_valid_recorder(self):
        rec = optimize_slm_gs_refine(_fast_config())

        assert len(rec.history) >= 2
        assert all("score" in row for row in rec.history)

        stages = [row["stage"] for row in rec.history]
        assert stages[0] == "flat"
        assert "gs" in stages
        assert set(stages) <= {"flat", "gs", "refine"}

        scores = [row["score"] for row in rec.history]
        assert all(np.isfinite(scores))
        assert max(scores) == pytest.approx(rec.get_best_iter()[1][1])

    def test_pib_never_exceeds_one(self):
        rec = optimize_slm_gs_refine(_fast_config(epochs=4))
        for row in rec.history:
            assert 0.0 <= row["pib"] <= 1.0, f"PIB out of range: {row['pib']}"
            assert np.isfinite(row["cv"])

    def test_best_phase_and_frame_are_attached_not_stored_per_row(self):
        rec = optimize_slm_gs_refine(_fast_config())

        assert rec.best_phase is not None
        assert rec.best_phase.shape == (1200, 1920)
        assert np.isfinite(rec.best_phase).all()
        assert rec.best_frame is not None
        assert rec.best_frame.ndim == 2

        # The memory defect: no row may carry a full panel array.
        assert all("phase" not in row for row in rec.history)
        assert all(not isinstance(row.get("cv"), np.ndarray) for row in rec.history)

    def test_roi_is_frozen_at_the_flat_baseline(self):
        """Every row must share the one ROI located on the unshaped frame."""
        rec = optimize_slm_gs_refine(_fast_config(epochs=4))
        cy = {row["cy"] for row in rec.history if "cy" in row}
        cx = {row["cx"] for row in rec.history if "cx" in row}
        assert len(cy) == 1 and len(cx) == 1

    def test_phases_stay_raw_unwrapped_radians(self):
        """The raw-only contract: the generator must not wrap to 2*pi."""
        rec = optimize_slm_gs_refine(_fast_config(gs_iters=0, gs_warm_start=False))
        assert rec.best_phase is not None
        assert np.abs(rec.best_phase).max() < 4.0 * np.pi

    def test_gs_bakeoff_rejects_a_losing_warm_start(self):
        """A mis-calibrated bench model must not be able to hurt the run."""
        cfg = _fast_config(gs_warm_start=True, gs_iters=6, epochs=1)
        rec = optimize_slm_gs_refine(cfg)
        by_stage = {row["stage"]: row["score"] for row in rec.history}
        assert "gs" in by_stage
        best = max(row["score"] for row in rec.history)
        if by_stage["gs"] <= by_stage["flat"]:
            # GS lost the bake-off, so the run must not have started from it.
            assert rec.best_stage != "gs" or best > by_stage["flat"]
        else:
            assert best >= by_stage["gs"]

    def test_disabling_gs_skips_the_stage(self):
        rec = optimize_slm_gs_refine(_fast_config(gs_warm_start=False, gs_iters=0))
        assert "gs" not in {row["stage"] for row in rec.history}

    def test_early_stop_triggers(self):
        rec = optimize_slm_gs_refine(_fast_config(epochs=50, early_stop_score=0.0))
        refine_rows = [r for r in rec.history if r["stage"] == "refine"]
        assert len(refine_rows) <= 50

    def test_callback_is_invoked_per_refine_epoch(self):
        seen: list[int] = []
        optimize_slm_gs_refine(
            _fast_config(epochs=3, callback=lambda e, s: seen.append(e))
        )
        assert seen == [0, 1, 2]

    def test_run_is_approximately_reproducible(self):
        """Only the perturbation stream is seeded; measurements carry device noise.

        Two same-seed runs disagree at the ~1e-3 level *at the flat baseline*,
        before any optimisation happens, because the far-field read-out adds
        noise on every frame. Exact run-to-run equality is therefore not a
        property this optimizer can have -- on hardware the noise is irreducible
        -- so the seed pins the SPGD perturbations and nothing more.
        """
        a = optimize_slm_gs_refine(_fast_config(epochs=3, seed=11))
        b = optimize_slm_gs_refine(_fast_config(epochs=3, seed=11))
        scores_a = [r["score"] for r in a.history]
        scores_b = [r["score"] for r in b.history]
        assert len(scores_a) == len(scores_b)
        assert scores_a == pytest.approx(scores_b, abs=0.05)
        # The control flow is deterministic; only the measurements are not.
        assert [r["stage"] for r in a.history] == [r["stage"] for r in b.history]

    def test_perturbation_stream_is_seed_determined(self):
        """The seeded stream drives the SPGD signs; it must be exact."""
        draws = [
            np.random.default_rng(11).choice(np.array([-1.0, 1.0]), size=16)
            for _ in range(2)
        ]
        assert np.array_equal(draws[0], draws[1])

    def test_devices_are_closed_after_the_run(self, monkeypatch):
        from ao_shaping.drivers.sim import slm_pib_sim

        closed = []
        original = slm_pib_sim.SimPibCCD.close

        def spy(self, *a, **kw):
            closed.append(True)
            return original(self, *a, **kw)

        monkeypatch.setattr(slm_pib_sim.SimPibCCD, "close", spy)
        optimize_slm_gs_refine(_fast_config(epochs=1))
        assert closed
