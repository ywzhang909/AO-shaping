"""R-28 D3: characterise the argmax spot-location helpers before merging any.

There are four ways this repo turns "brightest pixel" into a spot coordinate:

    spots_calc.center_of_brightness        numpy   -> (x, y)   raw argmax
    spots_calc.center_of_brightness_cupy   cupy    -> (x, y)   raw argmax
    spots_calc.center_of_brightness_numba  numba   -> (x, y)   hand-rolled arithmetic
    beam_metrics.zero_order_center          numpy   -> (x, y)   argmax + dark-frame guard
                                                     + optional sub-window refine

The first three are the same function behind three backends. The fourth is NOT
interchangeable with them, and this module exists mostly to record why.

The divergence is the dark frame. A bare ``argmax`` on an all-zero image returns
index 0, so the "centre" comes back as the top-left corner:

    center_of_brightness(np.zeros((5, 5)))   -> (0, 0)      # meaningless
    zero_order_center(np.zeros((5, 5)))      -> (2, 2)      # (w//2, h//2)

That is not a rounding detail -- it is the reason the repo's own bench notes
carry a rule that a dark frame must never be located with a bare argmax, and why
``zero_order_center`` grew the guard in the first place. These tests pin both
behaviours so the difference cannot be closed by accident, whichever direction a
future refactor goes.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.beam_metrics import zero_order_center
from ao_shaping.utils.image.spots_calc import (
    center_of_brightness,
    center_of_brightness_numba,
)


def _spot(shape=(64, 80), peak=(40, 56), sigma=5.0) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    return np.exp(-((yy - peak[0]) ** 2 + (xx - peak[1]) ** 2) / (2 * sigma**2))


# ---------------------------------------------------------------------------
# the three backends agree
# ---------------------------------------------------------------------------


def test_numpy_and_numba_backends_agree_on_a_clean_spot() -> None:
    img = _spot()
    assert center_of_brightness(img) == center_of_brightness_numba(img)
    assert center_of_brightness(img) == (56, 40)  # (x, y)


@pytest.mark.parametrize("seed", range(25))
def test_numpy_and_numba_agree_on_random_frames(seed: int) -> None:
    """Ties broken the same way, or the two backends disagree on real data.

    The frame is given a unique maximum so ``argmax`` has no tie to resolve; that
    is the case where the hand-rolled ``flat // w`` / ``flat % w`` could plausibly
    diverge from ``unravel_index``.
    """
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(2, 40)), int(rng.integers(2, 40))
    img = rng.random((h, w))
    img[rng.integers(0, h), rng.integers(0, w)] = 10.0
    assert center_of_brightness(img) == center_of_brightness_numba(img)


def test_both_backends_return_python_ints() -> None:
    """Numba returns numpy ints; callers compare against tuples and format them."""
    x, y = center_of_brightness(_spot())
    assert type(x) is int and type(y) is int
    x, y = center_of_brightness_numba(_spot())
    assert type(x) is int and type(y) is int


def test_backends_agree_on_a_saturated_flat_frame() -> None:
    """A clipped frame is all-equal, so argmax lands on index 0 for both."""
    flat = np.full((8, 8), 255.0)
    assert center_of_brightness(flat) == center_of_brightness_numba(flat)


# ---------------------------------------------------------------------------
# the documented fork: dark frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(5, 5), (4, 8), (9, 3)])
def test_bare_argmax_returns_the_corner_on_a_dark_frame(shape) -> None:
    """Documented, pinned, and expected -- do not "fix" it silently."""
    dark = np.zeros(shape, dtype=np.float64)
    assert center_of_brightness(dark) == (0, 0)
    assert center_of_brightness_numba(dark) == (0, 0)


@pytest.mark.parametrize("shape", [(5, 5), (4, 8), (9, 3)])
def test_zero_order_center_returns_the_frame_centre_on_a_dark_frame(shape) -> None:
    h, w = shape
    assert zero_order_center(dark_frame := np.zeros(shape)) == (w // 2, h // 2)
    assert dark_frame.shape == shape  # guard against a typo in the fixture


def test_the_fork_is_real_on_a_realistic_dark_frame() -> None:
    """Both helpers see the same pixels and still disagree.

    Asserted explicitly rather than left to the two tests above, because "they
    differ" is the fact a future merge attempt needs to see.
    """
    dark = np.zeros((64, 80))
    assert center_of_brightness(dark) != zero_order_center(dark)


# ---------------------------------------------------------------------------
# what zero_order_center adds, and why refactor callers should want it
# ---------------------------------------------------------------------------


def test_zero_order_center_refines_within_a_sub_window() -> None:
    """Refine is a LOCAL centroid, so it only moves within the search window.

    The half-window is ``max(min(h, w) // 20, 8)`` -- on a 40x40 frame that is
    +-8 px -- so grains further apart than that are not candidates. Two grains
    two pixels apart are well inside it, and the refined answer lands between
    them while the unrefined one stays on whichever maximum numpy listed first.

    Returns ``(x, y)``; row-major ordering makes the left grain the argmax.
    """
    img = np.zeros((40, 40))
    img[20, 18] = 100.0
    img[20, 20] = 100.0
    assert zero_order_center(img, refine=False) == (18, 20), "argmax snaps to a grain"
    assert zero_order_center(img, refine=True) == (19, 20), "refine averages them"


def test_refine_cannot_see_beyond_its_window() -> None:
    """Pins the window size, because "refine" reads like "find the real peak".

    A grain 40 rows away is invisible to it: the refined answer stays on the
    global argmax instead of jumping to the other one.
    """
    img = np.zeros((80, 80))
    img[20, 40] = 100.0
    img[60, 40] = 1.0  # far outside the +-8 window
    assert zero_order_center(img, refine=True) == (40, 20)


def test_zero_order_center_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        zero_order_center(np.zeros((3, 4, 5)))


def test_center_of_brightness_has_no_dark_frame_guard() -> None:
    """If a future change adds one, this is the test that must change with it."""
    with pytest.raises(TypeError):
        center_of_brightness(np.zeros((4, 4)), refine=True)