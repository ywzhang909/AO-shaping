"""Quantitative CCD exposure must be bounded and scale consistently."""

import numpy as np
import pytest

from ao_shaping.utils.image.exposure_control import (
    choose_exposure,
    exposure_normalized_frame,
)


def test_exposure_scaling_is_invariant_for_the_same_light() -> None:
    short = np.array([[10, 20], [40, 80]], dtype=np.uint16)
    long = short * 4
    np.testing.assert_allclose(
        exposure_normalized_frame(short, 0.05),
        exposure_normalized_frame(long, 0.2),
    )


def test_auto_exposure_never_crosses_daheng_ceiling() -> None:
    dim = np.full((4, 4), 100, dtype=np.uint16)
    decision = choose_exposure(dim, 0.02)
    assert decision.next_ms == 0.4
    assert not decision.accepted


def test_saturated_frame_is_rejected() -> None:
    saturated = np.full((4, 4), 65535, dtype=np.uint16)
    decision = choose_exposure(saturated, 0.02)
    assert not decision.accepted
    assert decision.next_ms < 0.02


def test_invalid_auto_exposure_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        choose_exposure(np.zeros((2, 2), dtype=np.float32), 0.02)
    with pytest.raises(ValueError):
        choose_exposure(np.zeros((2, 2), dtype=np.uint16), 0.41)
