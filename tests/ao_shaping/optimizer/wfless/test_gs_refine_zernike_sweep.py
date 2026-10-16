"""Tests for the GS-refine Zernike-order sweep driver.

The driver runs ONE GS-refine optimisation, fits its best phase into Zernike
coefficients at several truncation orders, then re-optimises each coefficient
vector with SPGD and re-scores every run on a single frozen square mask.

These tests exercise only the pure helpers (target-box resolution, sweep plan,
stage-C config assembly, clip warning, offline scoring) - never the CLI or any
hardware. The orchestration entry point is a plain callable so it can be driven
with tiny injected parameters; ``main`` is a thin argparse shell around it.
"""
from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms
from scripts.slm.run_gs_refine_zernike_sweep import (
    CAM_WINDOW_TARGET_MARGIN,
    DEFAULT_SIM_FAR_FIELD_PADDING,
    MIN_TARGET_SIDE,
    ZERNIKE_CLIP,
    build_stage_c_config,
    check_zernike_clip,
    probe_flat_field,
    resolve_target_side,
    score_frame,
    sweep_plan,
)


class TestTargetSide:
    """The shared target box must fit the camera window for every order."""

    def test_shared_target_side_fits_camera_window(self):
        side = resolve_target_side(w0=3.0, cam_size=256)
        assert 4.0 <= side <= 256 / CAM_WINDOW_TARGET_MARGIN

        # Tiny window: the clamp holds and the box still fits inside the frame.
        side_tiny = resolve_target_side(w0=3.0, cam_size=6)
        assert side_tiny < 6
        assert side_tiny <= 6 / CAM_WINDOW_TARGET_MARGIN + 1e-9

    def test_rejects_degenerate_target_box(self):
        with pytest.raises(ValueError):
            resolve_target_side(w0=3.0, cam_size=256, override=300.0)
        with pytest.raises(ValueError):
            resolve_target_side(w0=3.0, cam_size=256, override=256.0)


class TestSweepPlan:
    def test_sweep_plan_orders(self):
        plan = sweep_plan([4, 6, 8])
        assert [p["n_max"] for p in plan] == [4, 6, 8]
        assert [p["n_modes"] for p in plan] == [
            calc_n_zernike_terms(n) for n in (4, 6, 8)
        ]


class TestStageCConfig:
    def test_seed_coefficients_match_zernike_radius(self):
        coeffs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        cfg = build_stage_c_config(
            center=(128, 128),
            coefficients=coeffs,
            n_max=4,
            side=40.0,
            beam_radius_px=450.0,
            cam_type="sim",
            cam_id=0,
            exposure_time_ms=1.0,
            cam_size=256,
            slm_number=1,
            epochs=4,
            delta=0.2,
            lr=0.0,
            seed=0,
        )
        assert cfg.slm.zernike_radius == 450.0
        assert np.array_equal(cfg.slm.init_c, coeffs)
        assert cfg.camera.target_size == 40.0
        assert cfg.camera.name == "rms_pib"
        assert cfg.camera.target_shape == "square"
        assert cfg.center == (128, 128)
        assert cfg.epochs == 4

    def test_energy_guard_passthrough(self):
        cfg = build_stage_c_config(
            center=(128, 128),
            coefficients=np.zeros(5),
            n_max=4,
            side=40.0,
            beam_radius_px=450.0,
            cam_type="sim",
            cam_id=0,
            exposure_time_ms=1.0,
            cam_size=256,
            slm_number=1,
            epochs=4,
            delta=0.02,
            lr=0.01,
            seed=0,
            max_energy_loss=0.0,
        )
        assert cfg.camera.max_roi_energy_loss == 0.0


class TestClipWarning:
    def test_zernike_clip_warning(self):
        assert check_zernike_clip(1.0) is None
        warn = check_zernike_clip(ZERNIKE_CLIP + 1.0)
        assert warn is not None
        assert "clip" in warn.lower() or str(ZERNIKE_CLIP) in warn


class TestScoreFrame:
    def test_score_frame_reports_all_metrics(self):
        frame = np.zeros((64, 64), dtype=np.float64)
        frame[28:36, 28:36] = 100.0
        mask = np.zeros((64, 64), dtype=bool)
        mask[28:36, 28:36] = True

        scores = score_frame(frame, mask, center_xy=(32, 32), side=8)

        assert set(scores) >= {"pib", "cv", "ee", "ar", "quality"}
        assert all(np.isfinite(v) for v in scores.values())
        assert scores["pib"] > 0.9
        assert scores["ee"] > 0.9


class TestSimBenchGeometry:
    """The default sim bench must yield a resolvable target box, not the floor.

    The far-field spot size in pixels is ``N * padding / (pi * beam_w0)``, so it
    is invariant under a proportional scaling of the panel and the beam. These
    tests shrink both by 10x to keep the FFT cheap while exercising the exact
    default padding and the real probe path.
    """

    @staticmethod
    def _shrink_sim_bench(monkeypatch):
        import ao_shaping.drivers.sim.slm_pib_sim as sim

        class _SmallSystem(sim.SimPibSystem):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("slm_shape", (120, 192))
                kwargs.setdefault("beam_w0", 40.0)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(sim, "SimPibSystem", _SmallSystem)

    def test_default_sim_config_target_side_above_floor(self, monkeypatch):
        self._shrink_sim_bench(monkeypatch)
        _center, w0, shape = probe_flat_field(
            "sim", cam_id=0, exposure_time_ms=0.0, seed=0, cam_size=256
        )
        side = resolve_target_side(w0, 256)
        assert shape == (256, 256)
        # The box is ``2 * w0``; it must clear the floor rather than clamp to it.
        assert w0 > MIN_TARGET_SIDE / 2
        assert side > MIN_TARGET_SIDE

    def test_probe_configures_sim_far_field_padding(self, monkeypatch):
        import ao_shaping.drivers.sim.slm_pib_sim as sim

        self._shrink_sim_bench(monkeypatch)
        probe_flat_field("sim", cam_id=0, exposure_time_ms=0.0, seed=0, cam_size=256)
        assert sim.get_system().far_field_padding == DEFAULT_SIM_FAR_FIELD_PADDING

    def test_probe_flat_field_passes_cam_size(self, monkeypatch):
        import ao_shaping.drivers.ccd.common as ccd_common

        captured: dict = {}

        class _FakeCam:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_numpy_image(self, *args, **kwargs):
                yy, xx = np.mgrid[0:256, 0:256]
                return np.exp(-((xx - 128.0) ** 2 + (yy - 128.0) ** 2) / (2 * 8.0**2))

        def fake_create_camera(camera_type, cam_id=0, exposure_time_ms=20.0, **kwargs):
            captured["camera_type"] = camera_type
            captured["kwargs"] = kwargs
            return _FakeCam()

        monkeypatch.setattr(ccd_common, "create_camera", fake_create_camera)
        probe_flat_field("sim", cam_id=0, exposure_time_ms=0.0, seed=0, cam_size=256)
        assert captured["kwargs"].get("cam_size") == 256
