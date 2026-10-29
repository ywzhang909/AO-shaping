"""``slm-gs-refine`` 参数单一来源: 一处尺寸输入、一张形状表、一处派生。

四条不变量, 每条都对应一种**静默**的错结果:

1. **目标尺寸只有一个输入。** ``--gs-target-side-px`` 曾能在
   ``--target-side`` 之外静默覆盖 GS 模型里的边长, 于是横幅、评分掩膜与 GS
   目标可以各自说不同的尺寸。删掉它之后, 相机像素 -> 模型远场像素的换算是
   :func:`bench_target_side_px` 一个纯函数, 公式必须与删除前逐值一致 (含把
   ``<3`` 夹到 3)。
2. **形状旋钮不生效就报错。** ``--target-grid-nx`` 对圆形是解析成功、然后什么
   都不做。现在 ``build_gs_refine_config`` 直接拒绝这种组合, 并说明该形状真正
   读取哪些旋钮。
3. **横幅只列该形状读的旋钮**, 由 ``SHAPE_PARAMETER_FIELDS`` 渲染, 没有第二份
   if 链。
4. **验证配方只有一份。** ``VALIDATED_GS_REFINE_RECIPE`` 是 preset 而不是
   dataclass 默认值, GUI 的 ``GS_REFINE_RECIPE`` 是它的副本。

No hardware, no torch, no network, no files written.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from click.testing import CliRunner

from ao_shaping.optimizer.wfless.slm_gs_refine import (
    VALIDATED_GS_REFINE_RECIPE,
    SlmGsRefineConfig,
    _gs_pupil_phase,
    bench_target_side_px,
)
from ao_shaping.runners.runner_common import (
    SlmGsRefineParams,
    build_gs_refine_config,
)
from ao_shaping.runners.slm.gs_refine_runner import _FIELD_FLAGS, _shape_params_line
from ao_shaping.utils.image.target.metrics import TARGET_SHAPE_CHOICES
from ao_shaping.utils.image.target.shapes import SHAPE_PARAMETER_FIELDS

#: Every per-shape knob the CLI exposes (union over all shapes).
_ALL_SHAPE_KNOBS: tuple[str, ...] = tuple(
    sorted({name for names in SHAPE_PARAMETER_FIELDS.values() for name in names})
)


# ---------------------------------------------------------------------------
# 1. one target-size input
# ---------------------------------------------------------------------------


class TestOneTargetSizeInput:
    def test_the_second_input_flag_is_gone(self):
        from ao_shaping.runners.slm.gs_refine_runner import run

        result = CliRunner().invoke(run, ["--help"])
        assert result.exit_code == 0, result.output
        assert "--gs-target-side-px" not in result.output
        # The one remaining size input and its multiplier are still advertised.
        assert "--target-side" in result.output
        assert "--side-factor" in result.output

    def test_the_second_input_field_is_gone(self):
        config_fields = {f.name for f in dataclasses.fields(SlmGsRefineConfig)}
        params_fields = {f.name for f in dataclasses.fields(SlmGsRefineParams)}
        assert "gs_target_side_px" not in config_fields
        assert "gs_target_side_px" not in params_fields

    @pytest.mark.parametrize(
        ("camera_side_px", "camera_pixel_um", "far_field_pixel_size"),
        [
            # Realistic bench numbers: 2.2 um camera pitch, 3.45 um model pitch.
            (100, 2.2, 3.45e-6),
            (0, 2.2, 3.45e-6),
            (1, 2.2, 3.45e-6),
            (90, 2.2, 1e-6),
            (90, 2.2, 1e-4),
            (100, 0.0, 3.45e-6),  # pitch <= 0 is floored, not divided by zero
            (400, 5.5, 2.2e-6),
        ],
    )
    def test_conversion_matches_the_deleted_formula(
        self, camera_side_px: int, camera_pixel_um: float, far_field_pixel_size: float
    ):
        """The formula deleted from ``_gs_pupil_phase``, written out again here.

        ``max(3, round(side * max(p_cam, 1e-6) * 1e-6 / far_field_pixel_size))``
        is the pre-refactor expression verbatim; the helper must agree value for
        value, including the ``<3`` clamp.
        """
        cam_pitch_m = max(camera_pixel_um, 1e-6) * 1e-6
        expected = max(
            3, int(round(camera_side_px * cam_pitch_m / far_field_pixel_size))
        )
        assert (
            bench_target_side_px(
                camera_side_px,
                camera_pixel_um=camera_pixel_um,
                far_field_pixel_size=far_field_pixel_size,
            )
            == expected
        )

    def test_the_clamp_is_reachable(self):
        """A tiny side on a coarse model grid must come out at 3, not 1 or 2."""
        assert (
            bench_target_side_px(1, camera_pixel_um=2.2, far_field_pixel_size=1e-3) == 3
        )

    def test_gs_pupil_phase_has_no_override_branch(self):
        """Two configs that differed only in the deleted field are now identical.

        Before the change ``gs_target_side_px`` silently replaced the converted
        side; now the only route to the bench side is the conversion, so the same
        optics give the same GS target whatever a caller might have wanted to
        override.
        """
        base: dict[str, Any] = {
            "target_side": 90,
            "gs_iters": 4,
            "gs_warm_start": True,
            "far_field_padding": 2,
            "panel_pixel_um": 8.0,
            "camera_pixel_um": 2.2,
            "beam_radius_px": 60.0,
            "slm_wavelength": 1064,
            "seed": 0,
        }
        first = SlmGsRefineConfig(**base)
        second = SlmGsRefineConfig(**base)
        phase_a = _gs_pupil_phase(first, 60.0, 40)
        phase_b = _gs_pupil_phase(second, 60.0, 40)
        assert phase_a is not None
        assert phase_b is not None
        # Same inputs, same GS target: no hidden second size can divert it.
        assert phase_a.shape == phase_b.shape
        assert (phase_a == phase_b).all()

    def test_side_factor_still_scales_the_explicit_size(self):
        """``side_factor`` semantics are unchanged (regression guard)."""
        from ao_shaping.optimizer.wfless.slm_gs_refine import _derive_target_side

        assert (
            _derive_target_side(
                SlmGsRefineConfig(target_side=90, side_factor=2.0), 450.0
            )
            == 180
        )


# ---------------------------------------------------------------------------
# 2. a shape knob that does nothing is an error
# ---------------------------------------------------------------------------


class TestIrrelevantKnobsAreRejected:
    @pytest.mark.parametrize(
        ("shape", "knob", "value"),
        [
            ("circle", "target_grid_nx", 3),
            ("spot", "target_thickness", 0.2),
            ("square", "target_aspect_ratio", 3.0),
            ("pentagon", "target_sigma_ratio", 0.4),
            ("triangle", "target_inner_ratio", 0.3),
            ("star", "target_line_width", 0.1),
            ("rectangle", "target_inner_ratio", 0.3),
            ("annular", "target_thickness", 0.2),
            ("gaussian", "target_grid_ny", 7),
            ("grid", "target_aspect_ratio", 3.0),
            ("cross", "target_sigma_ratio", 0.4),
        ],
    )
    def test_rejects_with_the_shape_name(self, shape: str, knob: str, value: Any):
        params = SlmGsRefineParams(target_shape=shape, **{knob: value})
        with pytest.raises(ValueError) as excinfo:
            build_gs_refine_config(params, debug=False)
        message = str(excinfo.value)
        assert shape in message
        # The message must say what the shape DOES read, not only what is wrong.
        assert "reads" in message

    @pytest.mark.parametrize(
        ("shape", "knob", "value"),
        [
            ("rectangle", "target_aspect_ratio", 3.0),
            ("annular", "target_inner_ratio", 0.4),
            ("gaussian", "target_sigma_ratio", 0.3),
            ("grid", "target_line_width", 0.04),
            ("grid", "target_grid_nx", 3),
            ("grid", "target_grid_ny", 7),
            ("cross", "target_thickness", 0.09),
            ("circle", "target_side", 60),
        ],
    )
    def test_accepts_the_knob_the_shape_reads(self, shape: str, knob: str, value: Any):
        params = SlmGsRefineParams(target_shape=shape, **{knob: value})
        config = build_gs_refine_config(params, debug=False)
        assert getattr(config, knob) == value

    def test_defaults_are_accepted_for_every_shape(self):
        for shape in TARGET_SHAPE_CHOICES:
            config = build_gs_refine_config(
                SlmGsRefineParams(target_shape=shape), debug=False
            )
            assert config.target_shape == shape


# ---------------------------------------------------------------------------
# 3. one shape table
# ---------------------------------------------------------------------------


class TestShapeParameterTable:
    def test_table_covers_the_whole_vocabulary(self):
        assert set(SHAPE_PARAMETER_FIELDS) == set(TARGET_SHAPE_CHOICES)

    def test_every_field_is_a_real_params_field_with_a_flag(self):
        params_fields = {f.name for f in dataclasses.fields(SlmGsRefineParams)}
        for shape, names in SHAPE_PARAMETER_FIELDS.items():
            for name in names:
                assert name in params_fields, f"{shape} names {name}"
                assert name in _FIELD_FLAGS, f"{name} has no CLI flag"

    def test_spot_is_the_alias_of_circle(self):
        assert SHAPE_PARAMETER_FIELDS["spot"] == SHAPE_PARAMETER_FIELDS["circle"] == ()

    @pytest.mark.parametrize("shape", TARGET_SHAPE_CHOICES)
    def test_banner_renders_and_lists_only_the_read_knobs(self, shape: str):
        params = SlmGsRefineParams(target_shape=shape)
        line = _shape_params_line(params)
        assert line
        for name in _ALL_SHAPE_KNOBS:
            flag = _FIELD_FLAGS[name]
            if name in SHAPE_PARAMETER_FIELDS[shape]:
                assert flag in line, (shape, flag)
            else:
                assert flag not in line, (shape, flag)

    def test_a_shape_sized_by_the_side_alone_says_so(self):
        assert _shape_params_line(SlmGsRefineParams(target_shape="circle")) == (
            "edge = target side"
        )


# ---------------------------------------------------------------------------
# 4. one validated recipe
# ---------------------------------------------------------------------------


class TestValidatedRecipe:
    def test_every_key_is_a_params_field(self):
        params_fields = {f.name for f in dataclasses.fields(SlmGsRefineParams)}
        assert set(VALIDATED_GS_REFINE_RECIPE) <= params_fields

    def test_it_is_a_preset_not_a_dataclass_default(self):
        """The recipe must stay distinguishable from the library defaults.

        If it ever equalled every dataclass default it would no longer be
        evidence of anything, and the GUI could quietly lose the validated
        configuration without a single test noticing.
        """
        params = SlmGsRefineParams()
        differing = {
            key
            for key, value in VALIDATED_GS_REFINE_RECIPE.items()
            if getattr(params, key) != value
        }
        assert differing

    def test_the_gui_name_is_a_copy(self):
        from ao_shaping.gui.slm.shaping_controls import GS_REFINE_RECIPE

        assert GS_REFINE_RECIPE == dict(VALIDATED_GS_REFINE_RECIPE)
        assert GS_REFINE_RECIPE is not VALIDATED_GS_REFINE_RECIPE

    def test_mutating_the_gui_copy_leaves_the_validated_table_intact(self):
        from ao_shaping.gui.slm import shaping_controls

        before = dict(VALIDATED_GS_REFINE_RECIPE)
        shaping_controls.GS_REFINE_RECIPE["epochs"] = 999
        try:
            assert VALIDATED_GS_REFINE_RECIPE == before
        finally:
            shaping_controls.GS_REFINE_RECIPE.clear()
            shaping_controls.GS_REFINE_RECIPE.update(before)
