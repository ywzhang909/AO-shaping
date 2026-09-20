from __future__ import annotations

import numpy as np
from click.testing import CliRunner

from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    TARGET_SHAPE_CHOICES,
    gauss_center,
    shape_metric,
    target_shape_roi,
)
from ao_shaping.runners.slm_pib_runner import run


def test_gauss_center_is_subpixel_and_background_robust() -> None:
    yy, xx = np.mgrid[:64, :80]
    image = 12.0 + 100.0 * np.exp(-((xx - 31.2) ** 2 + (yy - 21.7) ** 2) / (2 * 3.0**2))

    center = gauss_center(image, half_win=20)

    np.testing.assert_allclose(center, (31.2, 21.7), atol=1e-9)


def test_target_shape_roi_tracks_center_and_aspect_ratio() -> None:
    first = target_shape_roi((64, 80), (40, 30), "rectangle", 20, 4 / 3)
    second = target_shape_roi((64, 80), (50, 45), "rectangle", 20, 4 / 3)

    ys, xs = np.nonzero(first)
    assert first.shape == (64, 80)
    assert np.isclose(
        (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1),
        4 / 3,
        atol=0.05,
    )
    assert not np.array_equal(first, second)
    assert first[30, 40]
    assert second[45, 50]


def test_shape_metric_uses_dynamic_roi_energy() -> None:
    image = np.zeros((64, 80), dtype=np.float64)
    image[20:30, 30:40] = 1.0
    shifted = np.zeros_like(image)
    shifted[25:35, 35:45] = 1.0

    centered = shape_metric(
        image,
        (35, 25),
        (39.5, 31.5),
        target_shape="rectangle",
        target_size=10,
    )
    translated = shape_metric(
        shifted,
        (40, 30),
        (39.5, 31.5),
        target_shape="rectangle",
        target_size=10,
    )

    assert centered[1] == 1.0
    assert translated[1] == 1.0
    assert translated[0] > centered[0]


def test_slm_pib_cli_exposes_shape_objective_options() -> None:
    result = CliRunner().invoke(run, ["--help"])

    assert result.exit_code == 0, result.output
    for option in (
        "--target-shape",
        "--target-size",
        "--target-aspect-ratio",
        "--target-center-smooth",
    ):
        assert option in result.output
    assert "shape" in result.output
    assert set(TARGET_SHAPE_CHOICES) == {
        "circle",
        "square",
        "rectangle",
        "annular",
        "grid",
        "cross",
        "gaussian",
        "pentagon",
    }
