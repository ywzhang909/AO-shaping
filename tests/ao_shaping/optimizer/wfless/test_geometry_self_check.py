"""The geometry self-check must catch what the correlation floor cannot.

On this bench a geometry fitted on an uncorrected pupil scored 0.831 and cleared the
correlation floor, while a usable one scored 0.848 -- so the floor is blind to a
defocus mismatch. These tests pin the replacement: the measured flat spot's share of
the camera window, which is 4.4% for the usable geometry and 36% for the broken one.
"""

from __future__ import annotations

import pytest

from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
    GEOMETRY_MAX_SPOT_WINDOW_FRACTION,
    BenchGeometry,
)


def _geometry(**overrides: float) -> BenchGeometry:
    """A geometry that passes, so each test changes exactly one field."""
    payload: dict[str, float] = {
        "panel_disc_radius": 450,
        "region": 128,
        "beam_waist_panel_px": 450.0,
        "far_field_size": 1024,
        "camera_px_per_model_px": 1.0467,
        "spot_fwhm_camera_px": 13.19,
        "spot_fwhm_model_px": 8.64,
        "correlation": 0.8476,
        "method": "sweep",
    }
    payload.update(overrides)
    return BenchGeometry(**payload)  # type: ignore[arg-type]


class TestSpotShareCheck:
    def test_a_usable_geometry_passes(self) -> None:
        assert _geometry().self_check_failures(300) == []

    def test_the_defocus_fitted_geometry_is_refused(self) -> None:
        """The geometry that really ran on 2026-10-19, verbatim.

        Its correlation (0.8309) is *below* the usable one's (0.8476) yet it cleared
        the 0.5 floor, so only the spot share separates them.
        """
        broken = _geometry(
            camera_px_per_model_px=2.6121,
            spot_fwhm_camera_px=106.85,
            correlation=0.8309,
        )
        failures = broken.self_check_failures(300)
        assert len(failures) == 1
        message = failures[0]
        assert f"{106.85:.1f} px" in message
        assert "36%" in message
        # The message has to name the cause, or the reader cannot act on it.
        assert "defocus" in message

    @pytest.mark.parametrize(
        ("fwhm", "expected_passes"),
        [
            (300 * 0.04, True),
            (300 * 0.10, True),
            (300 * 0.15, True),  # exactly at the bound still passes
            (300 * 0.16, False),
            (300 * 0.36, False),
        ],
    )
    def test_the_bound_is_inclusive(self, fwhm: float, expected_passes: bool) -> None:
        failures = _geometry(spot_fwhm_camera_px=fwhm).self_check_failures(300)
        assert (failures == []) is expected_passes

    def test_the_bound_is_dimensionless_in_the_camera_window(self) -> None:
        """The same fraction must be judged the same on a bigger window."""
        small = _geometry(spot_fwhm_camera_px=30.0).self_check_failures(300)  # 10%
        large = _geometry(spot_fwhm_camera_px=300.0).self_check_failures(3000)  # 10%
        assert small == [] and large == []
        wide = _geometry(spot_fwhm_camera_px=300.0).self_check_failures(300)  # 100%
        assert wide != []


class TestDegenerateInput:
    @pytest.mark.parametrize("fwhm", [0.0, -1.0, float("nan"), float("inf")])
    def test_an_unusable_width_is_refused_rather_than_divided_by(self, fwhm: float) -> None:
        failures = _geometry(spot_fwhm_camera_px=fwhm).self_check_failures(300)
        assert len(failures) == 1
        assert "not a usable width" in failures[0]

    @pytest.mark.parametrize("cam_size", [0, -300])
    def test_a_nonpositive_window_is_rejected(self, cam_size: int) -> None:
        with pytest.raises(ValueError, match="cam_size must be positive"):
            _geometry().self_check_failures(cam_size)


class TestDoesNotDuplicateTheCorrelationFloor:
    """The check must be independent, or it adds nothing to the gate."""

    def test_it_refuses_a_geometry_the_correlation_floor_would_accept(self) -> None:
        good_correlation = 0.8476
        broken = _geometry(spot_fwhm_camera_px=106.85, correlation=good_correlation)
        assert broken.correlation >= 0.5  # the floor would wave this through
        assert broken.self_check_failures(300) != []

    def test_it_passes_a_geometry_the_correlation_floor_would_reject(self) -> None:
        """A tight spot with poor correlation is the floor's business, not this one's."""
        weak = _geometry(correlation=0.11)
        assert weak.correlation < 0.5
        assert weak.self_check_failures(300) == []


def test_the_bound_sits_between_the_two_measured_geometries() -> None:
    """Pin the constant to the measurements that justify it.

    If a future bench shifts both numbers, this fails loudly instead of the constant
    silently drifting away from the evidence behind it.
    """
    usable_share = 13.19 / 300
    broken_share = 106.85 / 300
    assert usable_share < GEOMETRY_MAX_SPOT_WINDOW_FRACTION < broken_share