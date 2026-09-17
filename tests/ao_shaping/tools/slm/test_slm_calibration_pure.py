from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from ao_shaping.tools.slm.calibration import SLMCCDCalibrator, SLMLUTCalibrator


def test_moments_finds_intensity_centroid() -> None:
    yy, xx = np.mgrid[:21, :21]
    image = np.exp(-(((xx - 10) ** 2 + (yy - 10) ** 2) / 8.0))

    center = cast(Any, SLMCCDCalibrator)._moments(image)

    assert center == pytest.approx([10.0, 10.0])


def test_fwhm1d_counts_half_maximum_span() -> None:
    profile = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

    assert cast(Any, SLMCCDCalibrator)._fwhm1d(profile) == 3.0


def test_split_pattern_uses_calibrated_beam_center() -> None:
    calibrator = cast(Any, object.__new__(SLMLUTCalibrator))
    calibrator.panel_res = (4, 6)
    calibrator.calib = {"beam_center": np.array([1.0, 3.0])}
    calibrator.factory_2pi = 255.0

    pattern = calibrator._split_pattern(50.0, 100.0)

    assert pattern.shape == (4, 6)
    assert pattern[:, :3] == pytest.approx(50.0 * 2.0 * np.pi / 255.0)
    assert pattern[:, 3:] == pytest.approx(100.0 * 2.0 * np.pi / 255.0)


def test_center_uses_calibrated_center() -> None:
    calibrator = cast(Any, object.__new__(SLMLUTCalibrator))
    calibrator.calib = {"center": np.array([3.0, 4.0])}
    image = np.zeros((8, 8), dtype=np.float64)

    assert calibrator._center(image) == pytest.approx([3.0, 4.0])


def test_fringe_phase_recovers_sinusoid_period() -> None:
    calibrator = cast(Any, object.__new__(SLMLUTCalibrator))
    calibrator.window = 64
    calibrator.calib = {"center": np.array([40.0, 40.0])}
    x = np.arange(80)
    image = np.broadcast_to(np.sin(2.0 * np.pi * x / 8.0), (80, 80)).copy()

    phase, index, period = calibrator._fringe_phase(image)

    assert np.isfinite(phase)
    assert index == 8
    assert period == pytest.approx(8.0)
