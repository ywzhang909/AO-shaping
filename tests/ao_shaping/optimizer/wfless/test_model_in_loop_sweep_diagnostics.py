"""Diagnostics for the sweep-based bench geometry calibration.

The 2026-10-22 campaign died in the geometry stage with a bare
``KeyError: 'defocus'``: the defocus width response was degenerate (that
sweep was noise-dominated -- FWHM bouncing over 33-76 px with no trend), the
mode was dropped from the fit, and the next line read the dropped group
unconditionally. The failure read like a crash instead of the signal-to-noise
verdict it actually is.

These tests pin the replacement:

* a diagnostic error (:class:`SweepCalibrationError`) that names the failure,
  lists the dropped modes with their reasons and raw widths, and gives the two
  concrete next steps;
* a ``dropped`` record (``mode -> reason``) the caller can persist, filled in
  place and populated on the failure path too;
* bit-for-bit identical numbers when every mode responds -- the fix must not
  move a single digit of a good calibration.

Everything here is synthetic and hardware-free: the widths come from the
forward model itself, not from a camera.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
    SweepCalibrationError,
    SweepRecord,
    ZernikeCoefficientOptimizer,
    _pupil_amplitude,
    _spot_fwhm,
    _zernike_phase,
    calibrate_bench_geometry_from_sweep,
)


@pytest.fixture(autouse=True)
def _single_threaded_torch():
    """Pin torch to one thread for this module, then hand the pool back.

    Same reasoning as ``test_model_in_loop_shaping.py``: the calibration runs
    hundreds of 64x64 forward passes, where torch's default pool costs far
    more in context switching than it saves. Scoped rather than set at import,
    so the process-global thread budget the harness asserts on is untouched.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# ---------------------------------------------------------------------------
# Synthetic bench
# ---------------------------------------------------------------------------

REGION = 64
FAR_FIELD_SIZE = 512
SEMI_AXIS = 32
MODEL_TO_PANEL = 4.0
TRUE_CAM_PER_MODEL = 0.9
TRUE_WAIST = 20.0
WAIST_CANDIDATES = [16.0, 20.0, 24.0, 32.0]

MODE_NM = {
    "defocus": (2, 0),
    "astig_x": (2, -2),
    "astig_y": (2, 2),
    "coma_x": (3, -1),
    "coma_y": (3, 1),
}
TRUE_OFFSETS = {
    "defocus": 0.4,
    "astig_x": -0.3,
    "astig_y": 0.25,
    "coma_x": -0.2,
    "coma_y": 0.15,
}
COEFFS = (-2.0, -1.0, 1.0, 2.0)

#: Widths that *shrink* as |c| grows: the anti-correlation a noise-dominated
#: sweep produces (the fitted w^2 curvature comes out negative, i.e. the mode
#: "responds" the wrong way). The shape is irrelevant -- what matters is that
#: the width^2 fit cannot open upward.
DEGENERATE_WIDTHS = {-2.0: 30.0, -1.0: 45.0, 1.0: 46.0, 2.0: 31.0}


def _model_widths() -> dict[str, dict[float, float]]:
    """The width each swept mode produces on the synthetic bench."""
    opt = ZernikeCoefficientOptimizer(
        n_orders=1,
        region=REGION,
        dtype="float64",
        device="cpu",
        far_field_size=FAR_FIELD_SIZE,
    )
    zeros = np.zeros(opt.n_coefficients)
    amp = _pupil_amplitude(REGION, TRUE_WAIST, SEMI_AXIS)
    widths: dict[str, dict[float, float]] = {}
    for mode, nm in MODE_NM.items():
        offset = TRUE_OFFSETS[mode]
        widths[mode] = {}
        for coefficient in COEFFS:
            model = opt.forward_intensity(
                zeros,
                _zernike_phase(REGION, SEMI_AXIS, nm, coefficient - offset),
                amp,
            )
            widths[mode][coefficient] = float(_spot_fwhm(model) * TRUE_CAM_PER_MODEL)
    return widths


def _records(degenerate: tuple[str, ...] = ()) -> list[SweepRecord]:
    """A full sweep: flat, two tilt axes, and one record per mode/coefficient."""
    widths = _model_widths()
    k_tilt = TRUE_CAM_PER_MODEL * FAR_FIELD_SIZE / (np.pi * SEMI_AXIS)
    records: list[SweepRecord] = [
        SweepRecord("flat", "flat", 0.0, "", 12.0, 100.0, 200.0, 90.0)
    ]
    for coefficient in (-1.0, -0.5, 0.5, 1.0):
        jitter = 0.01 * coefficient  # deterministic, breaks the exact symmetry
        records.append(
            SweepRecord(
                f"tx{coefficient}",
                "tilt",
                coefficient,
                "x",
                12.0,
                100.0 + k_tilt * coefficient + jitter,
                200.0,
                80.0,
            )
        )
        records.append(
            SweepRecord(
                f"ty{coefficient}",
                "tilt",
                coefficient,
                "y",
                12.0,
                100.0,
                200.0 + k_tilt * coefficient - jitter,
                80.0,
            )
        )
    for mode in MODE_NM:
        for coefficient in COEFFS:
            width = (
                DEGENERATE_WIDTHS[coefficient]
                if mode in degenerate
                else widths[mode][coefficient]
            )
            records.append(
                SweepRecord(
                    f"{mode}{coefficient}",
                    mode,
                    coefficient,
                    "",
                    width,
                    0.0,
                    0.0,
                    60.0,
                    0.9,
                )
            )
    return records


def _calibrate(records, **kwargs):
    return calibrate_bench_geometry_from_sweep(
        records,
        region=REGION,
        far_field_size=FAR_FIELD_SIZE,
        model_to_panel=MODEL_TO_PANEL,
        waist_candidates=WAIST_CANDIDATES,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Reference values, captured from the implementation *before* the diagnostic
# error landed (2026-10-22). The fix must not move a single digit of a good
# calibration: same numbers, same notes.
# ---------------------------------------------------------------------------

REF_ALL_NORMAL_NOTES = (
    "tilt slope 4.584 cam px/rad (x:+4.594, y:+4.574, via ['centroid (no "
    "usable phase correlation)']), semi-axis 32 model px, P=512; focal scale "
    "from Zernike tilt slope 4.58 cam px/rad via ['centroid (no usable phase "
    "correlation)']; modes fitted astig_x+astig_y+coma_x+coma_y+defocus; joint "
    "rel-RMS 4.0%; bench offsets astig_x -0.300, astig_y +0.300, coma_x -0.300, "
    "coma_y +0.300, defocus +0.600 rad; curvatures astig_x 148.1, astig_y 145.4, "
    "coma_x 18.2, coma_y 18.7, defocus 232.8; defocus asymmetry -68%; waist fitted"
)
REF_ALL_NORMAL = {
    "panel_disc_radius": 128,
    "region": 64,
    "beam_waist_panel_px": 80.0,
    "far_field_size": 512,
    "camera_px_per_model_px": 0.9000000000000012,
    "spot_fwhm_camera_px": 12.0,
    "spot_fwhm_model_px": 9.331777421785404,
    "correlation": 0.9599427811850411,
}

REF_COMA_DEGENERATE_NOTES = (
    "tilt slope 4.584 cam px/rad (x:+4.594, y:+4.574, via ['centroid (no "
    "usable phase correlation)']), semi-axis 32 model px, P=512; focal scale "
    "from Zernike tilt slope 4.58 cam px/rad via ['centroid (no usable phase "
    "correlation)']; modes fitted astig_x+astig_y+coma_y+defocus; joint "
    "rel-RMS 4.6%; bench offsets astig_x -0.300, astig_y +0.300, coma_y +0.300, "
    "defocus +0.600 rad; curvatures astig_x 148.1, astig_y 145.4, coma_y 18.7, "
    "defocus 232.8; defocus asymmetry -68%; waist fitted"
)
REF_COMA_DEGENERATE = {
    "panel_disc_radius": 128,
    "region": 64,
    "beam_waist_panel_px": 80.0,
    "far_field_size": 512,
    "camera_px_per_model_px": 0.9000000000000012,
    "spot_fwhm_camera_px": 12.0,
    "spot_fwhm_model_px": 9.331777421785404,
    "correlation": 0.9535053562571063,
}


def _assert_matches_reference(geometry, reference: dict) -> None:
    for field, expected in reference.items():
        actual = getattr(geometry, field)
        if isinstance(expected, float):
            assert actual == pytest.approx(expected, rel=1e-12), field
        else:
            assert actual == expected, field


# ---------------------------------------------------------------------------
# The three scenarios
# ---------------------------------------------------------------------------


class TestSweepCalibrationDiagnostics:
    """The defocus-degenerate crash, and its two harmless cousins."""

    def test_defocus_degenerate_raises_a_diagnostic_error_not_a_keyerror(self) -> None:
        """The 2026-10-22 crash: a degenerate defocus must explain itself."""
        with pytest.raises(SweepCalibrationError) as excinfo:
            _calibrate(_records(("defocus",)))

        error = excinfo.value
        message = str(error)
        # A ValueError subclass, so every existing ``except ValueError`` keeps
        # working -- but never a KeyError.
        assert isinstance(error, ValueError)
        assert not isinstance(error, KeyError)
        # The message must name the failure, the evidence and the two next
        # steps, in the words the operator will search for.
        for keyword in (
            "离焦",  # defocus
            "退化",  # degenerate
            "提高曝光",  # raise the exposure
            "二阶矩半径",  # second-moment radius
            "25%",  # the width-resolution criterion
        ):
            assert keyword in message, keyword
        assert "defocus" in message
        # The dropped mode, its reason and its raw widths are all in there.
        assert "curvature" in message
        assert "30.0, 45.0, 46.0, 31.0" in message

    def test_the_error_carries_the_dropped_modes_and_the_surviving_ones(self) -> None:
        with pytest.raises(SweepCalibrationError) as excinfo:
            _calibrate(_records(("defocus",)))

        error = excinfo.value
        assert set(error.dropped) == {"defocus"}
        assert "curvature" in error.dropped["defocus"]
        assert error.usable_modes == ("astig_x", "astig_y", "coma_x", "coma_y")
        assert error.widths["defocus"] == [30.0, 45.0, 46.0, 31.0]

    def test_all_modes_respond_calibrates_and_reports_no_dropped_modes(self) -> None:
        """Regression: a good sweep must come out bit-for-bit unchanged."""
        dropped: dict[str, str] = {}
        geometry = _calibrate(_records(), dropped=dropped)

        assert dropped == {}
        assert geometry.method == "sweep"
        _assert_matches_reference(geometry, REF_ALL_NORMAL)
        assert geometry.calibration_notes == REF_ALL_NORMAL_NOTES

    def test_a_degenerate_non_defocus_mode_is_recorded_and_the_fit_continues(
        self,
    ) -> None:
        """A dead coma must not take the run down -- but it must be recorded."""
        dropped: dict[str, str] = {}
        geometry = _calibrate(_records(("coma_x",)), dropped=dropped)

        assert set(dropped) == {"coma_x"}
        assert "curvature" in dropped["coma_x"]
        _assert_matches_reference(geometry, REF_COMA_DEGENERATE)
        # The pre-fix notes survive verbatim; the dropped mode is appended.
        assert geometry.calibration_notes.startswith(REF_COMA_DEGENERATE_NOTES)
        assert "; dropped modes coma_x" in geometry.calibration_notes

    def test_the_dropped_record_is_published_even_when_the_calibration_fails(
        self,
    ) -> None:
        """The caller keeps the evidence even when the run dies."""
        dropped: dict[str, str] = {}
        with pytest.raises(SweepCalibrationError):
            _calibrate(_records(("defocus",)), dropped=dropped)

        assert set(dropped) == {"defocus"}
        assert "curvature" in dropped["defocus"]

    def test_a_defocus_group_with_one_unusable_point_is_diagnosed(self) -> None:
        """Same crash by another route: the defocus group never formed."""
        records: list[SweepRecord] = []
        for record in _records():
            if record.mode != "defocus":
                records.append(record)
            elif record.coefficient == -2.0:
                records.append(record)
            elif record.coefficient == -1.0:
                # One unusable width leaves a single usable defocus point, so
                # the group never forms -- and the run used to die with a
                # KeyError on the missing group.
                records.append(
                    SweepRecord(
                        record.label,
                        record.mode,
                        record.coefficient,
                        record.axis,
                        0.0,
                        record.centroid_x,
                        record.centroid_y,
                        record.peak,
                        record.hollowness,
                    )
                )

        with pytest.raises(SweepCalibrationError) as excinfo:
            _calibrate(records)

        message = str(excinfo.value)
        assert "defocus" in message
        assert "finite positive FWHM" in message

    def test_every_mode_degenerate_keeps_the_historical_wording(self) -> None:
        """The pre-existing message is preserved, with the evidence added."""
        with pytest.raises(SweepCalibrationError) as excinfo:
            _calibrate(_records(tuple(MODE_NM)))

        message = str(excinfo.value)
        assert "no swept mode produced a usable width response" in message
        assert "被丢弃的模式" in message
        assert "提高曝光" in message
        assert set(excinfo.value.dropped) == set(MODE_NM)
