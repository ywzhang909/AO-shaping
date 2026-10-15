"""Convergence tests: the two target-mask factories must agree.

``ao_shaping.utils.image.target.patterns.create_target_shape`` is the single
canonical implementation; ``ao_shaping.algorithm.signal_processing.
differentiable_shaping.create_target_mask`` is a thin delegate over it. These
tests pin that contract so the two cannot silently drift apart again.

The delegate returns ``float64`` while the canonical factory returns
``float32``; for the binary shapes the round-trip is exact, and for the
gaussian it differs only by float32 rounding (asserted with a tight
``atol``). No raw byte snapshot of the old implementation is embedded — the
properties below are the durable contract.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
    create_target_mask,
)
from ao_shaping.utils.image.target.patterns import TargetShape, create_target_shape

# Shapes accepted by BOTH factories (the delegate's full set).
SHARED_SHAPES = [
    "square",
    "rectangle",
    "triangle",
    "star",
    "circle",
    "gaussian",
    "spot",
]

GRID = 64
SIZE = 20


def _patterns_kwargs(shape: str, grid: int, size: int) -> dict:
    """The exact parameter mapping the delegate applies (documented contract)."""
    kwargs: dict = {"center": (grid / 2.0, grid / 2.0)}
    if shape == "square":
        kwargs["side"] = size
    elif shape == "rectangle":
        # ``side`` is the SHORT side; long side = side * aspect_ratio.
        kwargs["side"] = size / 2.0
        kwargs["aspect_ratio"] = 2.0
    elif shape in {"triangle", "star", "circle", "spot"}:
        kwargs["radius_ratio"] = size / grid
    else:  # gaussian
        kwargs["radius_ratio"] = 4.0 * (size / 6.0) / grid
    return kwargs


@pytest.mark.parametrize("shape", SHARED_SHAPES)
def test_delegate_matches_canonical_factory(shape: str) -> None:
    expected = create_target_shape(
        cast(TargetShape, shape), (GRID, GRID), **_patterns_kwargs(shape, GRID, SIZE)
    )
    actual = create_target_mask(shape, (GRID, GRID), SIZE)

    assert actual.dtype == np.float64
    assert expected.dtype == np.float32
    assert actual.shape == expected.shape == (GRID, GRID)
    assert np.allclose(actual, expected.astype(np.float64), rtol=0, atol=1e-6)


@pytest.mark.parametrize("shape", ["triangle", "star", "spot"])
def test_factory_now_accepts_new_shapes(shape: str) -> None:
    out = create_target_shape(cast(TargetShape, shape), (GRID, GRID), radius_ratio=0.3)
    assert out.shape == (GRID, GRID)
    assert out.dtype == np.float32
    assert float(out.max()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("shape", SHARED_SHAPES)
def test_output_dtype_shape_and_peak(shape: str) -> None:
    out = create_target_shape(
        cast(TargetShape, shape), (GRID, GRID), **_patterns_kwargs(shape, GRID, SIZE)
    )
    assert out.dtype == np.float32
    assert out.shape == (GRID, GRID)
    assert float(out.max()) == pytest.approx(1.0, abs=1e-6)


def test_circle_symmetric_about_both_axes() -> None:
    out = create_target_shape("circle", (GRID, GRID), radius_ratio=0.3)
    assert np.array_equal(out, np.flip(out, axis=0))
    assert np.array_equal(out, np.flip(out, axis=1))


def test_star_symmetric_about_vertical_axis() -> None:
    out = create_target_shape("star", (GRID, GRID), radius_ratio=0.3)
    assert np.array_equal(out, np.flip(out, axis=1))


def test_triangle_apex_up_base_down() -> None:
    out = create_target_shape("triangle", (GRID, GRID), radius_ratio=0.3)
    rows = np.nonzero(out.any(axis=1))[0]
    top, bottom = int(rows[0]), int(rows[-1])
    # Apex (top) is a single point; the base (bottom) is the widest row.
    assert int(out[top].sum()) < int(out[bottom].sum())


def test_unknown_shape_message_lists_all_shapes() -> None:
    with pytest.raises(ValueError) as exc:
        create_target_shape(cast(TargetShape, "bogus"), (GRID, GRID))
    msg = str(exc.value)
    for name in (
        "gaussian",
        "circle",
        "spot",
        "square",
        "annular",
        "grid",
        "cross",
        "rectangle",
        "pentagon",
        "triangle",
        "star",
    ):
        assert name in msg
