"""Coordinate contract for metrics on the simulated SLM shaping bench."""

import numpy as np

from ao_shaping.drivers.sim.slm_shaping_bench import (
    compute_metrics,
    power_in_bucket,
    zero_order_fraction,
)


def test_off_axis_center_uses_xy_coordinates() -> None:
    target = np.zeros((32, 32))
    target[15:17, 15:17] = 1.0
    intensity = np.zeros((32, 32))
    intensity[6, 23] = 10.0
    center = (23, 6)

    assert power_in_bucket(intensity, target, center=center) == 1.0
    assert zero_order_fraction(intensity, center) == 1.0
    assert compute_metrics(intensity, target, center=center)["PIB"] == 1.0
    assert compute_metrics(intensity, target)["zero_order"] == 1.0
