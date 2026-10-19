"""Image symmetry and correction search must work before optical use."""

import numpy as np
import pytest

from ao_shaping.tools.slm.panel.alignment_probe import scan_coma, symmetry_metrics


def _spot(dx=0, dy=0, tail=0):
    y, x = np.mgrid[:161, :161]
    core = np.exp(-((x-80-dx)**2 + (y-80-dy)**2)/128)
    wing = np.exp(-((x-100-dx)**2 + (y-80-dy)**2)/128)
    return 5 + 150 * (core + tail*wing)


def test_symmetry_ignores_exposure_background_and_translation():
    base = symmetry_metrics(_spot(), 50)
    moved = symmetry_metrics(3 * _spot(9, -7) + 13, 50)
    assert base["asymmetry"] < 1e-10
    assert moved["asymmetry"] == pytest.approx(base["asymmetry"], abs=1e-8)
    assert moved["center_x"] == pytest.approx(89, abs=.05)
    assert moved["center_y"] == pytest.approx(73, abs=.05)
    assert symmetry_metrics(_spot(tail=.5), 50)["left_right"] > .08


def test_coma_scan_recovers_symmetric_candidate_without_losing_focus():
    # Independent synthetic response: the two controls create orthogonal wings.
    y, x = np.mgrid[:161, :161]
    core = np.exp(-((x-80)**2 + (y-80)**2)/128)
    wing_x = np.exp(-((x-100)**2 + (y-80)**2)/128)
    wing_y = np.exp(-((x-80)**2 + (y-100)**2)/128)

    def capture(c, label):
        return 100*(core + abs(c[0]-.5)*wing_x + abs(c[1]+.5)*wing_y), .5

    result = scan_coma(capture, lambda c: c, span_rad=1, points=5,
                       passes=1, half_width=50)
    assert result["coefficients_rad"] == pytest.approx([.5, -.5])
    assert result["final"]["asymmetry"] < result["baseline"]["asymmetry"]
    assert result["final"]["rms_radius_px"] < result["baseline"]["rms_radius_px"]
    assert all("timestamp_utc" in row for row in result["rows"])


def test_invalid_or_edge_images_fail():
    with pytest.raises(ValueError, match="energy"):
        symmetry_metrics(np.zeros((161, 161)), 50)
    with pytest.raises(ValueError, match="edge"):
        symmetry_metrics(_spot(-65, 0), 50)
