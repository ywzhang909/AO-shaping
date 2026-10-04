"""The `far_field_padding` default is family-dependent, and measured so.

`ZernikeAmpConfig.far_field_padding` defaults to 10, and its own docstring warns
"the right value depends on the family's ``fov_px``, so re-run the sweep for a new
family rather than copying the default". That warning was not acted on: a sweep of
all 6 usable corpus families found **10 is optimal for exactly one of them**.

These tests pin the measured map and the fallback contract, so the numbers cannot
drift silently and so an unmeasured `fov_px` degrades to the documented default
instead of being guessed at.
"""

from __future__ import annotations

import pytest

from ml.zernike.models import (
    NO_VALID_PADDING_FOV,
    PADDING_BY_FOV_PX,
    ZernikeAmpConfig,
    recommended_padding,
)


def test_the_default_is_optimal_for_at_most_one_measured_family():
    """The reason this exists at all.

    Measured (scripts/sweep_far_field_padding.py, Z=0, 48 samples/family):
    fov 64 -> 8, fov 248 -> 14, fov 320 -> 16. Only the mixed 64/1944
    ``model_in_loop_hw_sweep`` family peaks at the default 10.
    """
    assert ZernikeAmpConfig.far_field_padding == 10
    optimal = [
        fov for fov, pad in PADDING_BY_FOV_PX.items() if pad == ZernikeAmpConfig.far_field_padding
    ]
    assert len(optimal) <= 1, (
        f"the default is optimal for {optimal}, so it is no longer a family-specific "
        "value and this table should be revisited"
    )


def test_the_optimum_tracks_fov_px():
    """The real signal in the sweep, stronger than any single argmax.

    Larger camera windows need more zero-padding to reach the same angular
    extent, so ``padding`` must increase with ``fov_px``.
    """
    ordered = sorted(PADDING_BY_FOV_PX)
    pads = [PADDING_BY_FOV_PX[fov] for fov in ordered]
    assert pads == sorted(pads), f"padding must be non-decreasing in fov_px: {dict(PADDING_BY_FOV_PX)}"


@pytest.mark.parametrize("fov", sorted(PADDING_BY_FOV_PX))
def test_recommended_padding_returns_the_measured_value(fov: int):
    assert recommended_padding(fov) == PADDING_BY_FOV_PX[fov]


@pytest.mark.parametrize("fov", [None, 999])
def test_unmeasured_fov_falls_back_to_the_documented_default(fov):
    """Never guess: an unknown window returns the dataclass default."""
    assert recommended_padding(fov) == ZernikeAmpConfig.far_field_padding


def test_full_sensor_fov_is_marked_as_having_no_valid_padding():
    """``fov_px=1944`` is negative at *every* padding.

    That is not a padding problem: ``slm_gsnet_square`` stores freeform phase
    cells rather than Zernike coefficients, so a Zernike-parameterised forward
    model has nothing to fit. Recording it prevents someone "fixing" -1.11 by
    tuning padding.
    """
    assert 1944 in NO_VALID_PADDING_FOV
    assert 1944 not in PADDING_BY_FOV_PX
    assert recommended_padding(1944) == ZernikeAmpConfig.far_field_padding


def test_every_measured_padding_is_a_positive_int():
    for fov, pad in PADDING_BY_FOV_PX.items():
        assert isinstance(pad, int) and pad >= 1, f"fov {fov}: bad padding {pad!r}"


def test_the_catastrophic_padding_is_not_recommended_anywhere():
    """``model_in_loop_hw_collect`` falls to R2 = -4.46 at pad 20.

    A large padding is not a safe "conservative" choice on a narrow window, so
    guard against a future table entry reintroducing it for fov 64.
    """
    assert PADDING_BY_FOV_PX[64] == 8
    assert PADDING_BY_FOV_PX[64] < 20