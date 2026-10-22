"""Tests for the acquisition preflight stage (focus compensation + de-steer).

The 2f bench sits permanently **out of focus**, and a forward model fitted on
an uncorrected bench predicts a **broader** spot than the hardware produces
(measured width ratio 1.328, CV bias -0.3009 over 51/51 records --
``report/shape_trials/experiment_summary_20261022.md`` §2).  So the Fresnel
focus phase has to be on the panel for every displayed pupil phase.

What is under test here is the **contract**, not the optics:

1. the overlay is ``((focus + tilt) + panel)`` -- order and sign are written
   down and pinned, because floating-point addition is not associative and the
   sign of the focus phase is carried by the (possibly negative) calibrated
   focal length;
2. with the compensation **off** the panel phase is handed back untouched --
   the *same object*, so a caller cannot even accidentally change it;
3. a broken calibration is an **explicit** error, never a silent zero
   compensation (which would put the run back on the uncorrected bench);
4. the de-steer ramp agrees with the probe script that owns that physics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.tools.kernels.bench_kernels import fresnel_panel
from ao_shaping.tools.slm.panel import focus_preflight as fp

PANEL_H, PANEL_W = 1200, 1920
WAVELENGTH_NM = 1064.0
PIXEL_UM = 8.0


def _write_focus_json(
    directory: Path,
    *,
    focal_mm: float = 1783.5418257111967,
    diopter_m1: float = 0.5606821133007323,
    radius: int = 450,
    center_x: int = 960,
    center_y: int = 600,
    wavelength: int = 1064,
) -> Path:
    """Write a calibration evidence file in the nested layout."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "focus.json"
    path.write_text(
        json.dumps(
            {
                "focus": {
                    "best_focal_mm": focal_mm,
                    "best_diopter_m1": diopter_m1,
                    "gain_vs_flat": 31.359632520449363,
                },
                "settings": {
                    "center_x": center_x,
                    "center_y": center_y,
                    "radius": radius,
                    "wavelength": wavelength,
                    "slm_number": 1,
                    "cam_type": "daheng",
                    "cam_id": 0,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


class TestCalibrationLoading:
    """The evidence file is the only thing that may set the focus phase."""

    def test_reads_the_nested_layout(self, tmp_path: Path) -> None:
        _write_focus_json(tmp_path)
        calibration = fp.load_focus_calibration(tmp_path)
        assert calibration.focal_mm == pytest.approx(1783.5418257111967)
        assert calibration.diopter_m1 == pytest.approx(0.5606821133007323)
        assert calibration.gain_vs_flat == pytest.approx(31.359632520449363)
        assert calibration.radius_px == 450
        assert (calibration.center_x, calibration.center_y) == (960, 600)
        assert calibration.wavelength_nm == pytest.approx(1064.0)
        assert calibration.slm_number == 1
        assert calibration.cam_type == "daheng"
        assert calibration.cam_id == 0
        assert calibration.source_path.endswith("focus.json")
        assert calibration.source_mtime_utc.endswith("+00:00")

    def test_reads_the_flat_layout(self, tmp_path: Path) -> None:
        """``probe_session_focus`` writes the evidence flat, not nested."""
        (tmp_path / "focus.json").write_text(
            json.dumps(
                {
                    "best_focal_mm": 1549.3,
                    "best_diopter_m1": 0.6454,
                    "gain_vs_flat": 29.0,
                    "radius_px": 450,
                    "centre_panel_px": [960, 600],
                    "slm_number": 1,
                    "cam_type": "daheng",
                    "cam_id": 0,
                }
            ),
            encoding="utf-8",
        )
        calibration = fp.load_focus_calibration(tmp_path)
        assert calibration.focal_mm == pytest.approx(1549.3)
        assert (calibration.center_x, calibration.center_y) == (960, 600)
        assert calibration.radius_px == 450

    def test_missing_file_is_an_explicit_error(self, tmp_path: Path) -> None:
        with pytest.raises(fp.FocusPreflightError, match="焦面标定证据不存在"):
            fp.load_focus_calibration(tmp_path / "nope")

    def test_missing_field_is_an_explicit_error(self, tmp_path: Path) -> None:
        (tmp_path / "focus.json").write_text(
            json.dumps({"focus": {"best_diopter_m1": 0.5}, "settings": {}}),
            encoding="utf-8",
        )
        with pytest.raises(fp.FocusPreflightError, match="best_focal_mm"):
            fp.load_focus_calibration(tmp_path)

    def test_zero_focal_length_is_rejected(self, tmp_path: Path) -> None:
        _write_focus_json(tmp_path, focal_mm=0.0)
        with pytest.raises(fp.FocusPreflightError, match="会聚"):
            fp.load_focus_calibration(tmp_path)

    def test_unreadable_json_is_an_explicit_error(self, tmp_path: Path) -> None:
        (tmp_path / "focus.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(fp.FocusPreflightError, match="不可读"):
            fp.load_focus_calibration(tmp_path)

    def test_a_broken_evidence_file_never_degrades_to_zero(self, tmp_path: Path) -> None:
        """The whole point: no silent zero compensation."""
        (tmp_path / "focus.json").write_text(
            json.dumps({"focus": {"best_focal_mm": None}, "settings": {}}),
            encoding="utf-8",
        )
        with pytest.raises(fp.FocusPreflightError):
            fp.build_focus_preflight(focus_dir=tmp_path, enabled=True)


class TestFocusPhase:
    """The focus phase is ``fresnel_panel`` verbatim, sign included."""

    def test_matches_fresnel_panel_bit_for_bit(self, tmp_path: Path) -> None:
        _write_focus_json(tmp_path)
        calibration = fp.load_focus_calibration(tmp_path)
        expected = fresnel_panel(
            calibration.focal_mm,
            radius=450,
            pupil_center=(960, 600),
            panel_shape=(PANEL_H, PANEL_W),
            wavelength_nm=1064.0,
        )
        actual = fp.build_focus_phase(
            calibration, panel_shape=(PANEL_H, PANEL_W)
        )
        assert np.array_equal(actual, expected)

    def test_a_non_positive_focal_length_is_rejected(self, tmp_path: Path) -> None:
        """Only a **converging** correction is a legal compensation.

        A zero focal length is not a lens; a negative one means the probe's
        winner landed on the diverging branch.  Either way, silently applying
        it would make an uncorrected bench worse instead of better.
        """
        for focal in (0.0, -1000.0):
            _write_focus_json(tmp_path, focal_mm=focal)
            with pytest.raises(fp.FocusPreflightError, match="会聚"):
                fp.load_focus_calibration(tmp_path)

    def test_the_focus_phase_is_converging(self, tmp_path: Path) -> None:
        """The quadratic is zero at the disc centre and positive outwards."""
        _write_focus_json(tmp_path)
        calibration = fp.load_focus_calibration(tmp_path)
        phase = fp.build_focus_phase(calibration, panel_shape=(PANEL_H, PANEL_W))
        assert float(phase[600, 960]) == 0.0
        assert float(phase[600, 1060]) > 0.0
        assert float(phase.max()) > 0.0

    def test_zero_is_outside_the_illuminated_disc(self, tmp_path: Path) -> None:
        _write_focus_json(tmp_path)
        calibration = fp.load_focus_calibration(tmp_path)
        phase = fp.build_focus_phase(calibration, panel_shape=(PANEL_H, PANEL_W))
        assert phase[0, 0] == 0.0
        assert phase[5, 5] == 0.0
        # The quadratic is zero at the disc centre and grows outwards.
        assert float(phase[600, 960]) == 0.0
        assert float(phase[600, 1060]) != 0.0

    def test_zernike_decomposition_matches_the_analytic_defocus(
        self, tmp_path: Path
    ) -> None:
        """``pi·r²/(λf)`` on a disc is exactly piston + Noll 4."""
        _write_focus_json(tmp_path)
        calibration = fp.load_focus_calibration(tmp_path)
        phase = fp.build_focus_phase(calibration, panel_shape=(PANEL_H, PANEL_W))
        decomposition = fp.zernike_decomposition(
            phase, radius_px=450, pupil_center=(960, 600)
        )
        edge = (
            math.pi
            * (450 * PIXEL_UM * 1e-6) ** 2
            / (WAVELENGTH_NM * 1e-9 * (calibration.focal_mm * 1e-3))
        )
        # rho^2 = 1/2 + (1/(2*sqrt(3))) * sqrt(3)(2 rho^2 - 1)
        assert decomposition["noll_4_defocus_rad"] == pytest.approx(
            edge / (2 * math.sqrt(3)), rel=1e-9
        )
        assert decomposition["noll_1_piston_rad"] == pytest.approx(
            edge / 2, rel=1e-9
        )
        # A pure quadratic has no other mode: the fit must be that clean.
        others = [
            value
            for name, value in decomposition["modes"].items()
            if not name.startswith(("noll1_", "noll4_"))
        ]
        assert max(abs(v) for v in others) < 1e-9


class TestOverlayContract:
    """``total = ((focus + tilt) + panel)`` -- order and sign pinned."""

    @staticmethod
    def _preflight(
        focus: np.ndarray | None, tilt: np.ndarray | None
    ) -> fp.FocusPreflight:
        return fp.FocusPreflight(
            requested=True,
            enabled=True,
            focus_phase_rad=focus,
            tilt_phase_rad=tilt,
        )

    def test_the_sum_is_exactly_focus_then_tilt_then_panel(self) -> None:
        focus = np.full((4, 6), 0.25, dtype=np.float64)
        tilt = np.full((4, 6), -0.5, dtype=np.float64)
        panel = np.full((4, 6), 1.75, dtype=np.float64)
        total = self._preflight(focus, tilt).apply_to_panel_phase(panel)
        expected = (focus + tilt) + panel
        assert np.array_equal(total, expected)

    def test_a_focus_only_overlay_is_focus_plus_panel(self) -> None:
        focus = np.full((4, 6), 0.25, dtype=np.float64)
        panel = np.full((4, 6), 1.75, dtype=np.float64)
        total = self._preflight(focus, None).apply_to_panel_phase(panel)
        assert np.array_equal(total, focus + panel)

    def test_a_tilt_only_overlay_is_tilt_plus_panel(self) -> None:
        tilt = np.full((4, 6), -0.5, dtype=np.float64)
        panel = np.full((4, 6), 1.75, dtype=np.float64)
        total = self._preflight(None, tilt).apply_to_panel_phase(panel)
        assert np.array_equal(total, tilt + panel)

    def test_the_panel_phase_is_not_mutated(self) -> None:
        focus = np.full((4, 6), 0.25, dtype=np.float64)
        panel = np.full((4, 6), 1.75, dtype=np.float64)
        before = panel.copy()
        self._preflight(focus, None).apply_to_panel_phase(panel)
        assert np.array_equal(panel, before)

    def test_disabled_returns_the_very_same_object(self) -> None:
        """Not "the same values" -- the same object.

        A caller that adds the compensation to the panel phase must not be
        able to change the array it was handed when the compensation is off.
        """
        panel = np.full((4, 6), 1.75, dtype=np.float64)
        preflight = fp.FocusPreflight(requested=False, enabled=False)
        assert preflight.apply_to_panel_phase(panel) is panel

    def test_enabled_without_any_compensation_returns_the_same_object(self) -> None:
        panel = np.full((4, 6), 1.75, dtype=np.float64)
        preflight = fp.FocusPreflight(requested=True, enabled=True)
        assert preflight.apply_to_panel_phase(panel) is panel


class TestDescribe:
    """The metadata is what a later reader has to trust."""

    def test_every_field_is_present(self, tmp_path: Path) -> None:
        _write_focus_json(tmp_path)
        preflight = fp.build_focus_preflight(
            focus_dir=tmp_path,
            enabled=True,
            panel_shape=(PANEL_H, PANEL_W),
            pupil_center=(960, 600),
            run_pupil_center=(960, 600),
        )
        payload = preflight.describe()
        for key in (
            "requested",
            "enabled",
            "disable_reason",
            "overlay_order",
            "overlay_contract",
            "calibration",
            "focus_phase_stats",
            "tilt_phase_stats",
            "tilt_specs",
            "tilt_source",
            "warnings",
            "recorded_utc",
        ):
            assert key in payload, f"missing {key}"
        assert payload["requested"] is True
        assert payload["enabled"] is True
        assert payload["overlay_order"] == ["focus", "tilt", "panel"]
        calibration = payload["calibration"]
        assert calibration["focal_mm"] == pytest.approx(1783.5418257111967)
        assert calibration["diopter_m1"] == pytest.approx(0.5606821133007323)
        assert calibration["gain_vs_flat"] == pytest.approx(31.359632520449363)
        assert calibration["source_path"].endswith("focus.json")
        stats = payload["focus_phase_stats"]
        assert stats["shape"] == [PANEL_H, PANEL_W]
        assert stats["max_abs_rad"] > 0.0
        assert 0 < stats["nonzero_px"] < stats["total_px"]

    def test_a_disabled_preflight_still_records_why(self) -> None:
        payload = fp.FocusPreflight(
            requested=False, enabled=False, disable_reason="未开启"
        ).describe()
        assert payload["requested"] is False
        assert payload["enabled"] is False
        assert payload["disable_reason"] == "未开启"
        assert payload["calibration"] is None

    def test_the_record_is_written_even_when_disabled(self, tmp_path: Path) -> None:
        preflight = fp.FocusPreflight(
            requested=False, enabled=False, disable_reason="未开启"
        )
        path = fp.write_preflight_record(tmp_path / "run", preflight)
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["enabled"] is False
        assert payload["disable_reason"] == "未开启"


class TestDisabledAndUnrequested:
    """The default path must not touch anything."""

    def test_disabled_preflight_reads_nothing(self, tmp_path: Path) -> None:
        """A broken evidence file must not fail a run that asked for no
        compensation -- and must not be read either."""
        (tmp_path / "focus.json").write_text("{not json", encoding="utf-8")
        preflight = fp.build_focus_preflight(focus_dir=tmp_path, enabled=False)
        assert preflight.enabled is False
        assert preflight.requested is False
        assert preflight.focus_phase_rad is None

    def test_requested_without_evidence_is_recorded_as_not_effective(
        self, tmp_path: Path
    ) -> None:
        preflight = fp.build_focus_preflight(focus_dir=None, enabled=True)
        assert preflight.requested is True
        assert preflight.enabled is False
        assert "焦面标定证据" in preflight.disable_reason
        # The live-probe plan is written out so the operator sees what *would*
        # have been done -- no device is opened in this path.
        assert preflight.live_probe_plan is not None
        assert preflight.live_probe_plan["action"] == "probe_session_focus"

    def test_a_stale_pupil_centre_is_flagged(self, tmp_path: Path) -> None:
        _write_focus_json(tmp_path, center_x=960, center_y=600)
        preflight = fp.build_focus_preflight(
            focus_dir=tmp_path,
            enabled=True,
            panel_shape=(PANEL_H, PANEL_W),
            pupil_center=(1000, 480),
            run_pupil_center=(1000, 480),
        )
        assert preflight.warnings
        assert "相差" in preflight.warnings[0]


class TestTiltCompensation:
    """The de-steer ramp must agree with the probe script that owns it."""

    #: The bench's measured ramp scale: a 2*pi ramp over ``period`` panel px
    #: moves the spot by ``TILT_SHIFT_SCALE / period`` camera px.
    SLOPE = 7400.0
    TEST_PERIOD_PX = 240

    @staticmethod
    def _axis_map() -> tuple:
        shift = TestTiltCompensation.SLOPE / TestTiltCompensation.TEST_PERIOD_PX
        return (
            fp.TiltAxisResponse(
                panel_axis=0, camera_axis=1, shift_px=shift,
                period_px=TestTiltCompensation.TEST_PERIOD_PX,
            ),
            fp.TiltAxisResponse(
                panel_axis=1, camera_axis=0, shift_px=-shift,
                period_px=TestTiltCompensation.TEST_PERIOD_PX,
            ),
        )

    def test_matches_the_probe_script_bit_for_bit(self) -> None:
        """``src/`` may not import ``scripts/``, so the formula is duplicated --
        this test is what keeps the two copies from drifting apart."""
        from scripts.slm.hw_gs_defocus_probe import (
            RampAxisMap,
            RampAxisResponse,
            compensating_ramp,
        )

        responses = self._axis_map()
        axis_map = RampAxisMap(
            tuple(
                RampAxisResponse(
                    panel_axis=r.panel_axis,
                    camera_axis=r.camera_axis,
                    shift_px=r.shift_px,
                    period_px=r.period_px,
                )
                for r in responses
            )
        )
        desired = (3.5, -2.25)
        mine, mine_specs = fp.build_tilt_phase(
            responses, desired, (PANEL_H, PANEL_W)
        )
        theirs, their_specs = compensating_ramp(
            desired, axis_map, (PANEL_H, PANEL_W)
        )
        assert np.array_equal(mine, theirs)
        assert [s.period_px for s in mine_specs] == [
            s.period_px for s in their_specs
        ]
        assert [s.panel_axis for s in mine_specs] == [
            s.panel_axis for s in their_specs
        ]
        assert [s.sign for s in mine_specs] == [s.sign for s in their_specs]

    def test_the_sign_follows_the_measured_response(self) -> None:
        """A ramp whose phase gradient is reversed moves the spot the other way."""
        responses = self._axis_map()
        phase, specs = fp.build_tilt_phase(responses, (3.5, 0.0), (PANEL_H, PANEL_W))
        assert len(specs) == 1
        # camera axis 0 is driven by panel axis 1, whose measured slope is
        # negative, so a positive desired shift needs a negative ramp.
        assert specs[0].panel_axis == 1
        assert specs[0].sign == -1.0
        assert float(np.abs(phase).max()) > 0.0

    def test_a_zero_desired_shift_produces_no_ramp(self) -> None:
        phase, specs = fp.build_tilt_phase(
            self._axis_map(), (0.0, 0.0), (PANEL_H, PANEL_W)
        )
        assert specs == ()
        assert not np.any(phase)

    def test_the_axis_map_is_read_from_the_probe_script_layout(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "axis_map.json"
        shift = self.SLOPE / self.TEST_PERIOD_PX
        path.write_text(
            json.dumps(
                {
                    "axis_map": {
                        "panel_axis_0": {
                            "panel_axis": 0,
                            "camera_axis": 1,
                            "shift_px": shift,
                            "period_px": self.TEST_PERIOD_PX,
                        },
                        "panel_axis_1": {
                            "panel_axis": 1,
                            "camera_axis": 0,
                            "shift_px": -shift,
                            "period_px": self.TEST_PERIOD_PX,
                        },
                        "measured_vs_expected_ratio": 1.01,
                    }
                }
            ),
            encoding="utf-8",
        )
        responses = fp.load_tilt_axis_map(path)
        assert len(responses) == 2
        assert responses[0].panel_axis == 0
        assert responses[0].camera_axis == 1
        assert responses[1].shift_px == pytest.approx(-shift)

    def test_a_tilt_map_without_a_desired_shift_is_rejected(
        self, tmp_path: Path
    ) -> None:
        _write_focus_json(tmp_path)
        path = tmp_path / "axis_map.json"
        path.write_text(json.dumps({"axis_map": {}}), encoding="utf-8")
        with pytest.raises(fp.FocusPreflightError, match="没有可用的实测响应"):
            fp.load_tilt_axis_map(path)

    def test_the_whole_chain_from_the_probe_calibration_matches(
        self, tmp_path: Path
    ) -> None:
        """End to end: the probe's own calibration -> JSON -> this module's ramp.

        The probe's ``calibrate_chirp_tilt`` measures the panel-ramp ->
        camera-shift map, ``axis_map_to_dict`` writes it out and
        ``compensation_for_dz`` turns it into the de-steer ramp.  This module
        consumes the **same** measurement and must produce the **same** ramp,
        otherwise the preflight would de-steer by a different amount than the
        probe measured.
        """
        from scripts.slm.hw_gs_defocus_probe import (
            ChirpTiltCalibration,
            RampAxisMap,
            RampAxisResponse,
            axis_map_to_dict,
            compensation_for_dz,
            predict_shift_px,
        )

        calibration = ChirpTiltCalibration(
            axis_map=RampAxisMap(
                (
                    RampAxisResponse(
                        panel_axis=0, camera_axis=1,
                        shift_px=7400.0 / 240, period_px=240,
                    ),
                    RampAxisResponse(
                        panel_axis=1, camera_axis=0,
                        shift_px=-7400.0 / 240, period_px=240,
                    ),
                )
            ),
            raw_shift_px_per_mm=(-5.5, 5.2),
            cal_dz_mm=1.5,
            test_period_px=240,
            cal_centroid_xy=(960.0, 600.0),
        )
        path = tmp_path / "axis_map.json"
        path.write_text(
            json.dumps({"axis_map": axis_map_to_dict(calibration.axis_map)}),
            encoding="utf-8",
        )
        responses = fp.load_tilt_axis_map(path)
        dz = 1.5
        desired = tuple(
            -value
            for value in predict_shift_px(calibration.raw_shift_px_per_mm, dz)
        )
        mine, mine_specs = fp.build_tilt_phase(
            responses, desired, (PANEL_H, PANEL_W)
        )
        theirs, their_specs, realised = compensation_for_dz(
            dz, calibration, (PANEL_H, PANEL_W)
        )
        assert np.array_equal(mine, theirs)
        assert len(mine_specs) == len(their_specs)
        for got, want in zip(mine_specs, their_specs, strict=True):
            assert got.panel_axis == want.panel_axis
            assert got.period_px == want.period_px
            assert got.sign == want.sign
        assert realised != (0.0, 0.0)
