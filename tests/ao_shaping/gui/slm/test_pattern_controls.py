"""Tests for the pattern_controls refactor: init-carried defaults, raw radian
phase generation, binary 0/pi output, raw-gray overrides and the dispatcher.

The rewritten ``XXXControl`` classes must:
- carry SLM context (id/wavelength/pitch/panel/bits) in ``__init__`` — never
  read ``st.session_state`` for defaults;
- expose ``generate_phase_rad(params) -> float64 (height, width)`` raw unwrapped
  radians (no mod-2pi, no uint16 conversion);
- keep binary patterns (棋盘格/二元光栅/达曼光栅) at exactly 0.0/pi;
- keep the raw-uint16 gray contract for 平场 (amplitude coupling) and the
  slm-required contract for the default driver conversion + GS方形整形;
- dispatch through ``_build_control`` from the SLM object's own properties.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ao_shaping.gui.slm import pyarrow_probe
from ao_shaping.gui.slm.pattern_controls import (
    PATTERN_REGISTRY,
    BlazedGratingControl,
    CircularGratingControl,
    FlatControl,
    GSSquareControl,
    HalfHalfPhaseControl,
    LensControl,
    LinearGratingControl,
    PatternControl,
    SteadyPhaseControl,
    VortexPhaseControl,
    ZernikeControl,
    _build_control,
    generate_phase_gray,
)
from ao_shaping.utils.wavefront.zernike_calc import zernike_modes

# The order rendered by the old multi_slm_controller selectbox — must be preserved.
CANONICAL_REGISTRY_ORDER = [
    "平场",
    "线性光栅",
    "圆形光栅",
    "透镜",
    "全息光栅",
    "闪耀光栅",
    "棋盘格",
    "二元光栅",
    "微透镜阵列",
    "湍流相位屏",
    "Zernike",
    "达曼光栅",
    "涡旋相位",
    "半半相位",
    "GS方形整形",
    "GS方形整形(高斯版)",
    "稳像法整形",
]


def _mock_slm() -> MagicMock:
    slm = MagicMock()
    slm.slm_number = 0
    slm.wavelength = 1064.0
    slm.Pitch_um = 8.0
    slm.Panel_Res = (128, 96)  # (width, height)
    slm.Gray_Scale_bits = 10
    slm.create_phase_from_array = MagicMock(
        side_effect=lambda rad: np.zeros(rad.shape, dtype=np.uint16)
    )
    return slm


class TestInitContext:
    def test_prefix_uses_slm_id(self) -> None:
        assert FlatControl(slm_id=1).prefix == "slm1"
        assert FlatControl(slm_id=7).prefix == "slm7"

    def test_defaults_and_ranges(self) -> None:
        ctrl = FlatControl()
        assert (ctrl.width, ctrl.height, ctrl.bits) == (1920, 1200, 10)
        assert ctrl.max_gray == 1023
        assert ctrl.default_radius == 600  # min(1920, 1200) // 2

    def test_init_values_are_coerced(self) -> None:
        # GUI values may arrive as strings — __init__ must coerce them.
        slm_id: Any = "2"
        wavelength: Any = "1064.5"
        pixel_pitch_um: Any = "8.5"
        width: Any = "640"
        height: Any = "480"
        ctrl = FlatControl(
            slm_id=slm_id,
            wavelength=wavelength,
            pixel_pitch_um=pixel_pitch_um,
            width=width,
            height=height,
            bits=8,
        )
        assert ctrl.slm_id == 2
        assert ctrl.wavelength == 1064.5
        assert ctrl.pixel_pitch_um == 8.5
        assert (ctrl.width, ctrl.height) == (640, 480)
        assert ctrl.max_gray == 255

    def test_registry_order_matches_ui_selectbox(self) -> None:
        assert list(PATTERN_REGISTRY.keys()) == CANONICAL_REGISTRY_ORDER

    def test_every_registered_class_has_generate_phase_rad(self) -> None:
        for pattern_type, cls in PATTERN_REGISTRY.items():
            assert hasattr(cls, "generate_phase_rad"), pattern_type
            assert callable(cls.generate_phase_rad), pattern_type


class TestGeneratePhaseRad:
    @pytest.mark.parametrize(
        "pattern_type, params",
        [
            ("线性光栅", {"period": 64.0, "phase_range": 2.0}),
            ("圆形光栅", {"radius": 64.0, "phase_range": 2.0}),
            ("棋盘格", {"period": 50}),
            ("Zernike", {"n_max": 4, "radius": 55, "coefficients": {(2, 0): 1.0}}),
            (
                "涡旋相位",
                {"topological_charge": 1, "wavelength_nm": 1064, "pixel_pitch_um": 8},
            ),
        ],
    )
    def test_rad_phase_is_float64_panel_shaped(
        self, pattern_type: str, params: dict
    ) -> None:
        cls = PATTERN_REGISTRY[pattern_type]
        ctrl = cls(width=128, height=96, wavelength=1064.0, pixel_pitch_um=8.0)
        phase = ctrl.generate_phase_rad(params)
        assert isinstance(phase, np.ndarray)
        assert phase.dtype == np.float64
        assert phase.shape == (96, 128)

    def test_linear_grating_is_unwrapped_raw_radians(self) -> None:
        ctrl = LinearGratingControl(width=128, height=128)
        phase = ctrl.generate_phase_rad({"period": 64.0, "phase_range": 4.0 / np.pi})
        # Centered grid: phase = (x/period)·phase_range·π ∈ [-w/2/period·pr·π, +w/2/period·pr·π].
        # phase_range=4.0/π means 4.0 radians of swing.
        # Raw radians: values are NOT grayscale and NOT mod-2pi-wrapped.
        assert phase.min() >= -4.0 - 1e-9 and phase.max() <= 4.0 + 1e-9
        assert phase.max() > 1.0  # modulated

    def test_zernike_rad_preserves_coefficient_amplitude(self) -> None:
        ctrl = ZernikeControl(width=128, height=128)
        phase = ctrl.generate_phase_rad(
            {"n_max": 4, "radius": 55, "coefficients": {(4, 0): 2.0}}
        )
        assert phase.dtype == np.float64
        assert float(np.max(np.abs(phase))) > 1e-6  # real modulation, not flat


class TestBinaryPatterns:
    @pytest.mark.parametrize(
        "pattern_type, params",
        [
            ("棋盘格", {"period": 50}),
            ("二元光栅", {"a": 2, "b": 3, "direction": "horizontal"}),
        ],
    )
    def test_binary_rad_phase_is_exactly_0_or_pi(
        self, pattern_type: str, params: dict
    ) -> None:
        cls = PATTERN_REGISTRY[pattern_type]
        ctrl = cls(width=128, height=128)
        phase = ctrl.generate_phase_rad(params)
        values = np.unique(phase)
        assert values.shape == (2,)
        assert np.isclose(values, 0.0).any()
        assert np.isclose(values, float(np.pi)).any()


class TestFlatRawGrayContract:
    def test_flat_rad_phase_is_ideal_interpretation(self) -> None:
        ctrl = FlatControl(width=128, height=96)
        phase = ctrl.generate_phase_rad({"flat_gray": 512})
        expected = 512 / ctrl.max_gray * 2.0 * np.pi
        assert phase.shape == (96, 128)
        assert np.allclose(phase, expected)

    def test_flat_gray_is_raw_uint16_without_slm(self) -> None:
        ctrl = FlatControl(width=128, height=96)
        gray = ctrl.generate_phase_gray({"flat_gray": 7})
        assert gray.dtype == np.uint16
        assert gray.shape == (96, 128)
        assert np.all(gray == 7)


class TestSlmRequiredPaths:
    def test_default_gray_requires_slm(self) -> None:
        ctrl = LinearGratingControl()
        with pytest.raises(ValueError, match="需要 slm 对象"):
            ctrl.generate_phase_gray({"period": 64.0, "phase_range": 2.0})

    def test_gs_square_requires_slm(self) -> None:
        ctrl = GSSquareControl(width=128, height=128)
        with pytest.raises(ValueError):
            ctrl.generate_phase_rad({})

    def test_dispatcher_unknown_pattern_raises(self) -> None:
        with pytest.raises(ValueError, match="未知相位图类型"):
            generate_phase_gray(_mock_slm(), "不存在的图案", {})

    def test_dispatcher_flat_routes_to_raw_gray(self) -> None:
        slm = _mock_slm()
        gray = generate_phase_gray(slm, "平场", {"flat_gray": 100})
        assert gray.dtype == np.uint16
        assert gray.shape == (96, 128)
        assert np.all(gray == 100)

    def test_dispatcher_linear_routes_through_driver(self) -> None:
        slm = _mock_slm()
        gray = generate_phase_gray(
            slm, "线性光栅", {"period": 64.0, "phase_range": 2.0}
        )
        assert gray.dtype == np.uint16
        assert gray.shape == (96, 128)
        # The driver conversion received the RAW radian phase (float64, unwrapped).
        assert slm.create_phase_from_array.call_count == 1
        rad_arg = slm.create_phase_from_array.call_args[0][0]
        assert rad_arg.dtype == np.float64
        assert rad_arg.shape == (96, 128)
        assert not np.allclose(rad_arg, rad_arg % (2 * np.pi))  # not pre-mod-2pi


class TestVortexContract:
    def test_vortex_rad_phase_shape_and_modulation(self) -> None:
        ctrl = VortexPhaseControl(width=128, height=128, wavelength=1064.0)
        phase = ctrl.generate_phase_rad(
            {"topological_charge": 1, "wavelength_nm": 1064, "pixel_pitch_um": 8}
        )
        assert phase.dtype == np.float64
        assert phase.shape == (128, 128)
        # A topological charge 1 vortex is azimuthally modulated (raw phase,
        # not wrapped): the centered-grid arctan2 spans ~2*pi across the panel.
        assert float(np.max(phase) - np.min(phase)) > 5.0


class TestDefaultsRangesSpec:
    """Group 1 — every registered control carries non-empty defaults, and every
    range key is backed by a default (the min_value-fallback bug regression)."""

    @pytest.mark.parametrize("pattern_type", list(PATTERN_REGISTRY.keys()))
    def test_defaults_nonempty_and_ranges_keys_in_defaults(
        self, pattern_type: str
    ) -> None:
        cls = PATTERN_REGISTRY[pattern_type]
        ctrl = cls(width=128, height=96, wavelength=1064.0, pixel_pitch_um=8.0)
        assert ctrl.defaults, f"{pattern_type}: DEFAULTS must not be empty"
        for key in ctrl.ranges:
            assert key in ctrl.defaults, (
                f"{pattern_type}: range key '{key}' missing from defaults"
            )


class TestRenderEmbedsDefaults:
    """Group 2 — render() returns the stored defaults (not min_value / not
    st.session_state). This is the regression test for the min_value fallback
    bug: a bare-mode number_input without value= returns min_value."""

    @pytest.mark.parametrize(
        "pattern_type",
        ["线性光栅", "圆形光栅", "透镜", "涡旋相位", "稳像法整形", "GS方形整形"],
    )
    def test_render_params_embed_defaults(self, pattern_type: str) -> None:
        cls = PATTERN_REGISTRY[pattern_type]
        ctrl = cls(width=128, height=96, wavelength=1064.0, pixel_pitch_um=8.0)
        params = ctrl.render("slm0")
        for key, default in ctrl.defaults.items():
            if key not in params:
                # Conditional params (e.g. gs_p_cam only emitted when > 0).
                continue
            if isinstance(default, bool):
                assert params[key] is default, f"{pattern_type}: {key}"
            else:
                assert params[key] == default, f"{pattern_type}: {key}"


class TestDefaultsRegression:
    """Group 3 — defaults must produce a physically meaningful (modulated)
    phase, not the all-zero grating from a period=1.0 min_value fallback."""

    def test_linear_grating_defaults_produce_modulated_phase(self) -> None:
        ctrl = LinearGratingControl(width=128, height=128)
        phase = ctrl.generate_phase_rad(dict(ctrl.defaults))
        assert phase.dtype == np.float64
        assert phase.shape == (128, 128)
        assert float(np.max(phase) - np.min(phase)) > 1.0


class TestContextDerivedDefaults:
    """Group 4 — context-derived defaults are resolved in __init__ from the
    SLM context, not hardcoded."""

    def test_lens_radius_defaults_to_panel_radius(self) -> None:
        ctrl = LensControl(width=128, height=96)
        assert ctrl.defaults["lens_radius"] == ctrl.default_radius == 48

    def test_vortex_wavelength_defaults_to_slm_wavelength(self) -> None:
        ctrl = VortexPhaseControl(width=128, height=96, wavelength=1064.0)
        assert ctrl.defaults["wavelength_nm"] == int(ctrl.wavelength) == 1064

    def test_steady_phase_defaults_are_physical(self) -> None:
        ctrl = SteadyPhaseControl(width=128, height=96)
        assert ctrl.defaults["focal_length_mm"] == 300.0
        assert ctrl.defaults["waist_radius_um"] == 3600.0
        assert ctrl.defaults["flat_top_half_length_um"] == 150.0
        assert ctrl.defaults["blaze_period"] == 4.0
        assert ctrl.defaults["blaze_angle_deg"] == 45.0


class TestZernikeEditorFallback:
    """Regression — a broken/mismatched pyarrow must not take the page down.

    ``st.data_editor`` imports pyarrow *inside* the widget call, so a failing
    import (e.g. a cp313 pyarrow wheel left in a cp314 interpreter's
    site-packages: ``ModuleNotFoundError: No module named 'pyarrow.lib'``)
    aborts the whole Streamlit script run — selecting "Zernike" in the
    phase-type dropdown wiped every element after it ("页面自动退出").  The
    control must probe the capability first and fall back to plain,
    pyarrow-free widgets.
    """

    def test_modes_parity_and_count(self) -> None:
        modes = zernike_modes(5)
        assert len(modes) == 21  # sum(n + 1) for n in 0..5
        assert (0, 0) in modes
        assert (2, 0) in modes and (2, 1) not in modes  # n - |m| must be even

    def test_modes_is_canonical_zernike_math(self) -> None:
        """``zernike_modes`` must live in utils.zernike_calc, not the GUI layer."""
        from ao_shaping.utils import zernike_calc

        assert hasattr(zernike_calc, "zernike_modes")
        assert zernike_calc.zernike_modes is zernike_modes
        # ZernikeControl no longer owns a _modes method.
        assert not hasattr(ZernikeControl, "_modes")

    def test_render_falls_back_when_data_editor_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            pyarrow_probe,
            "probe_pyarrow",
            lambda: (False, "ModuleNotFoundError: No module named 'pyarrow.lib'"),
        )
        ctrl = ZernikeControl(width=128, height=96)
        params = ctrl.render("slm0")

        assert params["n_max"] == 5
        assert params["radius"] == float(ctrl.default_radius) == 48.0
        assert set(params["coefficients"]) == set(zernike_modes(5))
        # Every mode defaults to 0.0 rad (no piston special-case).
        assert all(value == 0.0 for (n, m), value in params["coefficients"].items())

    def test_data_editor_probe_matches_pyarrow_importability(self) -> None:
        """The probe must mirror the real pyarrow import state, not assume."""
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            expected, reason = False, f"{type(exc).__name__}: {exc}"
        else:
            expected, reason = True, ""
        available, probe_reason = pyarrow_probe.probe_pyarrow()
        assert available is expected
        assert (probe_reason == "") is expected
        if not expected:
            assert reason.split(":")[0] in probe_reason  # e.g. ModuleNotFoundError

    def test_pandas_compat_probe_matches_real_import_state(self) -> None:
        """``pyarrow_pandas_compat_ok`` must mirror the real import state.

        ``st.data_editor`` calls ``pa.Table.from_pandas`` internally, which
        does ``from pyarrow.pandas_compat import ...``.  When that submodule
        is missing the widget aborts the script run — the same fatal pattern
        as a plain ``pyarrow.lib`` failure.  The probe must catch it.
        """
        try:
            import pyarrow.pandas_compat  # noqa: F401
        except ImportError:
            expected = False
        else:
            expected = True
        assert pyarrow_probe.pyarrow_pandas_compat_ok() is expected

    def test_capability_fails_when_pandas_compat_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even with pyarrow importable, a broken pandas_compat must fail the probe."""
        monkeypatch.setattr(pyarrow_probe, "pyarrow_pandas_compat_ok", lambda: False)
        available, reason = pyarrow_probe.probe_pyarrow()
        assert available is False
        assert "pandas_compat" in reason

    def test_diagnostics_report_reason_and_repair_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reason = "ModuleNotFoundError: No module named 'pyarrow.lib'"
        monkeypatch.setattr(pyarrow_probe, "probe_pyarrow", lambda: (False, reason))
        monkeypatch.setattr(pyarrow_probe, "pyarrow_version", lambda: None)
        monkeypatch.setattr(pyarrow_probe, "pyarrow_pandas_compat_ok", lambda: False)
        rows = dict(pyarrow_probe.pyarrow_diagnostics())
        assert reason in rows["pyarrow"]
        assert rows["pyarrow.pandas_compat"] == "不可用"
        assert "Python" in rows and "解释器接受的扩展后缀" in rows
        assert "pip install" in rows["修复命令"]

    def test_debug_panel_lists_editor_path_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pyarrow_probe, "probe_pyarrow", lambda: (True, ""))
        monkeypatch.setattr(pyarrow_probe, "pyarrow_pandas_compat_ok", lambda: True)
        rows = dict(pyarrow_probe.pyarrow_diagnostics())
        assert rows["系数编辑器"] == "st.data_editor (系数表)"
        assert rows["pyarrow.pandas_compat"] == "可用"
        assert "修复命令" not in rows  # nothing to repair


class TestMaxGrayOverride:
    """Group 5 — the driver's real 2π max-grayscale (slm._max_gray) overrides
    the bits-derived fallback in __init__ and flows into defaults/ranges."""

    def test_max_gray_override(self) -> None:
        ctrl = FlatControl(max_gray=993)
        assert ctrl.max_gray == 993
        assert ctrl.defaults["flat_gray"] == 0
        assert ctrl.ranges["flat_gray"] == (0, 993)

        ctrl2 = HalfHalfPhaseControl(max_gray=993)
        assert ctrl2.max_gray == 993
        assert ctrl2.defaults["flat_gray"] == 496
        assert ctrl2.ranges["flat_gray"] == (0, 993)

        # No override -> bits-derived fallback (10 bits -> 1023).
        ctrl3 = FlatControl()
        assert ctrl3.max_gray == 1023
        assert ctrl3.defaults["flat_gray"] == 0

    def test_generate_phase_rad_uses_max_gray(self) -> None:
        ctrl = FlatControl(max_gray=993)
        phase = ctrl.generate_phase_rad({"flat_gray": 993})
        assert phase.shape == (1200, 1920)
        assert np.allclose(phase, 2 * np.pi)

    def test_context_synced_widgets_declared(self) -> None:
        assert PatternControl.CONTEXT_SYNCED_WIDGETS == ()
        assert FlatControl.CONTEXT_SYNCED_WIDGETS == ("flat_gray",)
        assert HalfHalfPhaseControl.CONTEXT_SYNCED_WIDGETS == ("halfhalf_flat_gray",)
        assert BlazedGratingControl.CONTEXT_SYNCED_WIDGETS == ("blazed_calc_wl",)
        assert VortexPhaseControl.CONTEXT_SYNCED_WIDGETS == ("vortex_wavelength",)

    def test_build_control_propagates_max_gray(self) -> None:
        slm_with = _mock_slm()
        slm_with._max_gray = 993
        ctrl = _build_control("平场", slm_with)
        assert ctrl.max_gray == 993

        # No _max_gray attribute → MagicMock child is not an int → None →
        # bits-derived fallback (proves the isinstance guard).
        slm_without = _mock_slm()
        ctrl2 = _build_control("平场", slm_without)
        assert ctrl2.max_gray == 1023


class TestPhaseRangeNPiUnits:
    """phase_range 输入框为 N*pi 单位，默认 2 (=2π)，发送时自动乘以 π。

    验证:
    - render 返回的 phase_range 默认为 2.0 (即 2π)
    - generate_phase_rad 内部将 N*pi 乘以 π 得到弧度
    - 转化为灰度后值域 [0, max_gray] 与 max_gray 一致
    """

    def test_phase_range_default_is_2pi(self) -> None:
        """All phase_range controls default to 2.0 (= 2π)."""
        for pattern_type, cls in PATTERN_REGISTRY.items():
            if "phase_range" in cls.DEFAULTS:
                ctrl = cls(width=128, height=96, wavelength=1064.0, pixel_pitch_um=8.0)
                assert ctrl.defaults["phase_range"] == 2.0, (
                    f"{pattern_type}: phase_range default "
                    f"should be 2.0 (2π), got {ctrl.defaults['phase_range']}"
                )

    def test_linear_grating_phase_range_conversion(self) -> None:
        """phase_range=2.0 (N*pi) → 2π radians in generate_phase_rad."""
        ctrl = LinearGratingControl(width=128, height=128)
        phase = ctrl.generate_phase_rad({"period": 64.0, "phase_range": 2.0})
        assert phase.dtype == np.float64
        # phase_range=2π means max phase = (63/64)*2π ≈ 6.22
        assert float(np.max(np.abs(phase))) > 6.0

    def test_circular_grating_phase_range_conversion(self) -> None:
        ctrl = CircularGratingControl(width=128, height=128)
        phase = ctrl.generate_phase_rad({"radius": 64.0, "phase_range": 2.0})
        assert phase.dtype == np.float64
        assert float(np.max(np.abs(phase))) > 6.0

    def test_halfhalf_phase_range_conversion(self) -> None:
        ctrl = HalfHalfPhaseControl(width=128, height=128)
        phase = ctrl.generate_phase_rad(
            {
                "flat_gray": 512,
                "split_direction": "左右",
                "period": 4.0,
                "phase_range": 2.0,
                "blaze_direction": "vertical",
            }
        )
        assert phase.dtype == np.float64
        assert phase.shape == (128, 128)

    def test_blazed_grating_phase_range_conversion(self) -> None:
        ctrl = BlazedGratingControl(width=128, height=128)
        phase = ctrl.generate_phase_rad(
            {"period": 64.0, "phase_range": 2.0, "direction": "vertical"}
        )
        assert phase.dtype == np.float64
        assert float(np.max(np.abs(phase))) > 6.0

    def test_phase_to_gray_matches_max_gray(self) -> None:
        """When phase_range=2π (N*pi=2.0), gray output fills [0, max_gray].

        The driver's create_phase_from_array maps radian phase ∈ [0, 2π)
        to grayscale ∈ [0, max_gray].  With phase_range=2π, the max phase
        on the panel approaches 2π, so the max gray approaches max_gray.
        """
        max_gray = 993
        slm = MagicMock()
        slm._max_gray = max_gray
        slm.create_phase_from_array = MagicMock(
            side_effect=lambda rad: np.round(
                np.mod(np.asarray(rad, dtype=np.float64), 2 * np.pi)
                / (2 * np.pi)
                * max_gray
            ).astype(np.uint16)
        )

        ctrl = LinearGratingControl(width=128, height=128)
        gray = ctrl.generate_phase_gray({"period": 64.0, "phase_range": 2.0}, slm=slm)
        assert gray.dtype == np.uint16
        # Max gray should approach max_gray (phase wraps near 2π)
        assert gray.max() >= max_gray * 0.95, (
            f"Expected gray max >= {max_gray * 0.95:.0f}, got {gray.max()}"
        )
        assert gray.min() == 0

    def test_phase_range_render_defaults(self) -> None:
        """render() should return phase_range=2.0 for all phase_range controls."""
        for pattern_type in [
            "线性光栅",
            "圆形光栅",
            "全息光栅",
            "半半相位",
            "闪耀光栅",
        ]:
            cls = PATTERN_REGISTRY[pattern_type]
            ctrl = cls(width=128, height=96)
            params = ctrl.render("slm0")
            if "phase_range" in params:
                assert params["phase_range"] == 2.0, (
                    f"{pattern_type}: render phase_range should be 2.0, "
                    f"got {params['phase_range']}"
                )
