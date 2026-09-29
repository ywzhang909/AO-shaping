"""Rule-parity tests for ``slm_zernike_shaping.ObjectiveSpec``.

``slm_zernike_shaping`` is the sandbox copy of ``slm_zernike_pib`` where the
inline ``(objective, target_shape)`` if-else was refactored into a single nested
``ObjectiveSpec.resolve(objective, target_shape)`` value object. These tests pin
that the resolver reproduces the ORIGINAL rules EXACTLY:

* shape membership + ``target_shape`` only valid for
  ``pib``/``rmse``/``shape``/``roi_pib``/``rms_pib``;
* supplying ``target_shape`` promotes ``pib`` -> ``shape`` (but leaves
  ``roi_pib``/``rms_pib``/``rmse`` as the ROI selector only);
* the shape family defaults to ``"rectangle"``;
* the exact ``ValueError`` messages and their ordering.

No hardware is opened.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ao_shaping.optimizer.wfless.slm_zernike_shaping import ObjectiveSpec
from ao_shaping.utils.image.targets import SHAPING_OBJECTIVE_CHOICES


# (objective, target_shape) -> (resolved name, resolved shape)
RESOLVE_TABLE: tuple[tuple[str, str | None, str, str | None], ...] = (
    ("pib", None, "pib", None),
    ("pib", "square", "shape", "square"),          # promotion
    ("PIB", "SQUARE", "shape", "square"),          # case-insensitive
    ("roi_pib", None, "roi_pib", "rectangle"),     # ROI family default
    ("roi_pib", "square", "roi_pib", "square"),    # ROI selector, no promotion
    ("rms_pib", None, "rms_pib", "rectangle"),
    ("rms_pib", "circle", "rms_pib", "circle"),
    ("rmse", None, "rmse", "rectangle"),
    ("rmse", "circle", "rmse", "circle"),
    ("rmse_out", None, "rmse_out", "rectangle"),   # + outside penalty, ROI family
    ("rmse_out", "square", "rmse_out", "square"),
    ("shape", None, "shape", "rectangle"),         # defining param, defaulted
    ("shape", "pentagon", "shape", "pentagon"),
    ("radiu", None, "radiu", None),
    ("avg_radiu", None, "avg_radiu", None),
)


class TestResolveTable:
    @pytest.mark.parametrize("objective,target_shape,name,shape", RESOLVE_TABLE)
    def test_resolution(self, objective, target_shape, name, shape) -> None:
        spec = ObjectiveSpec.resolve(objective, target_shape)
        assert spec.name == name
        assert spec.shape == shape

    @pytest.mark.parametrize("objective,target_shape,name,shape", RESOLVE_TABLE)
    def test_shape_family_never_null_shape(self, objective, target_shape, name, shape) -> None:
        spec = ObjectiveSpec.resolve(objective, target_shape)
        from ao_shaping.utils.image.targets import TARGET_SHAPE_CHOICES

        assert (spec.name, spec.shape) == (name, shape)
        if name in ("shape", "roi_pib", "rms_pib", "rmse", "rmse_out"):
            assert spec.shape in TARGET_SHAPE_CHOICES


class TestValidation:
    def test_shape_rejected_for_radiu(self) -> None:
        with pytest.raises(ValueError, match="target_shape can only be used with"):
            ObjectiveSpec.resolve("radiu", "square")

    def test_shape_rejected_for_avg_radiu(self) -> None:
        with pytest.raises(ValueError, match="target_shape can only be used with"):
            ObjectiveSpec.resolve("avg_radiu", "circle")

    def test_unknown_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="target_shape must be one of"):
            ObjectiveSpec.resolve("pib", "BOGUS")

    def test_unknown_objective_without_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="objective must be one of"):
            ObjectiveSpec.resolve("bogus", None)

    def test_unknown_objective_with_shape_raises_compat_error_first(self) -> None:
        # Original ordering: the target_shape/objective compatibility check (A)
        # fires BEFORE the final objective-membership check (D).
        with pytest.raises(ValueError, match="target_shape can only be used with"):
            ObjectiveSpec.resolve("bogus", "square")


class TestSpecShapeInvariants:
    def test_is_frozen(self) -> None:
        spec = ObjectiveSpec.resolve("pib", None)
        assert dataclasses.is_dataclass(spec)
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "shape"  # type: ignore[misc]

    def test_objective_choices_are_the_eight(self) -> None:
        assert set(SHAPING_OBJECTIVE_CHOICES) == {
            "pib",
            "radiu",
            "avg_radiu",
            "rmse",
            "rmse_out",
            "shape",
            "roi_pib",
            "rms_pib",
        }


class TestSimEndToEnd:
    """The refactored optimizer still runs a full SPGD loop on the 2f sim.

    ``SimSLMPib`` never implemented the ``from_params`` classmethod the real
    ``Santec`` exposes (a pre-existing sim gap, unrelated to this refactor), so
    it is shimmed here. The camera goes through the registry
    (``cam_type="sim"``). No hardware is touched.
    """

    def test_spgd_runs_end_to_end(self, monkeypatch) -> None:
        import ao_shaping.optimizer.wfless.slm_zernike_shaping as opt
        from ao_shaping.drivers.sim.slm_pib_sim import (
            SimSLMPib,
            register_sim_camera,
            reset_system,
        )
        from ao_shaping.runners.runner_common import (
    CameraParamsPib,
    ObjectiveTarget,
    SlmParamsPib,
)

        register_sim_camera()
        reset_system(seed=42)
        monkeypatch.setattr(
            SimSLMPib,
            "from_params",
            classmethod(lambda cls, params, **kw: cls()),
            raising=False,
        )
        monkeypatch.setattr(opt, "Santec", SimSLMPib)

        config = opt.SlmZernikePibConfig(
            center="shape",
            epochs=2,
            algorithm="spgd",
            camera=CameraParamsPib(target=ObjectiveTarget(name="pib"), cam_type="sim", cam_size=250),
            slm=SlmParamsPib(),
        )
        recorder = opt.optimize_slm_zernike_pib(config)

        # init row + one row per epoch
        assert len(recorder.history) == 3
        # the objective's own column is recorded on every row
        assert "pib" in recorder.history[0]
        assert "pib" in recorder.history[-1]


class TestRmseOutMetric:
    """``rmse_out_metric`` = normalised RMSE + w_out * (1 - in_target_energy)."""

    def test_equals_rmse_plus_weighted_outside_fraction(self) -> None:
        from ao_shaping.utils.image.targets import (
            rmse_out_metric,
            rmse_shape_metric,
        )

        img = np.zeros((40, 40), dtype=np.float64)
        img[18:22, 18:22] = 1.0
        center = (20.0, 20.0)
        rmse, energy = rmse_shape_metric(img, center, "square", 4)
        j, e2 = rmse_out_metric(img, center, "square", 4, w_outside=2.0)
        assert e2 == pytest.approx(energy)
        assert j == pytest.approx(rmse + 2.0 * (1.0 - energy))

    def test_more_outside_light_raises_j(self) -> None:
        from ao_shaping.utils.image.targets import rmse_out_metric

        center = (20.0, 20.0)
        inside = np.zeros((40, 40))
        inside[18:22, 18:22] = 1.0
        outside = np.zeros((40, 40))
        outside[0, 0] = 1.0
        j_in, e_in = rmse_out_metric(inside, center, "square", 4, w_outside=1.0)
        j_out, e_out = rmse_out_metric(outside, center, "square", 4, w_outside=1.0)
        assert e_in == pytest.approx(1.0)
        assert e_out == pytest.approx(0.0)
        assert j_out > j_in

    def test_dark_frame_is_strongly_penalised(self) -> None:
        from ao_shaping.utils.image.targets import rmse_out_metric

        j, energy = rmse_out_metric(
            np.zeros((20, 20)), (10.0, 10.0), "square", 3, w_outside=1.0
        )
        assert j == pytest.approx(1e3 + 1.0)
        assert energy == 0.0

