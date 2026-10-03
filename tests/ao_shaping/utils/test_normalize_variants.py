"""R-28 D4: three different "normalise by the peak" behaviours, not one.

The TODO counted "6+ places" that normalise an image by its maximum. There are
indeed six -- ``x / x.max()`` in ``gui/slm/pattern_controls.py``,
``drivers/sim/fouriergsnet_env.py`` and four times in
``drivers/sim/slm_shaping_bench.py`` -- and there are already three canonical
helpers that they are not using:

    x / x.max()                        the six inline sites
    beam_metrics.normalize_pattern     peak, with a guard, float32 out
    wavefront_calc.normalize_01        min-max, constant -> zeros

They are not interchangeable. Measured, per input:

    input              x/x.max()      normalize_pattern   normalize_01
    zeros              NaN + warning  0.0                 0.0
    negative peak       1.0           -1.0  (unchanged)   0.0
    contains NaN       NaN            finite              NaN
    constant 5.0       1.0            1.0                 0.0
    int32 input dtype  preserved      float32             float64

The first row is the one that matters. A dark frame divided by its own maximum
is 0/0, so the six inline sites manufacture NaN on exactly the input the repo's
bench notes say must never be trusted -- while ``normalize_pattern`` returns a
clean zero. That is a real difference in kind, not rounding.

The second row is a latent contradiction: ``normalize_pattern`` documents itself
as normalising to ``[0, 1]`` but returns its input untouched when the peak is
non-positive, which is neither normalised nor in range.

So this file pins the divergence rather than merging the implementations. Any
future attempt to route the six sites through a helper has to satisfy these
assertions first, and two of them will fail -- which is the point.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ao_shaping.utils.image.beam_metrics import normalize_pattern
from ao_shaping.utils.wavefront.wavefront_calc import normalize_01

CASES = {
    "zeros": np.zeros((4, 4)),
    "negative_peak": -np.ones((4, 4)),
    "with_nan": np.array([[1.0, np.nan], [2.0, 3.0]]),
    "constant": np.full((4, 4), 5.0),
    "normal": np.arange(16, dtype=float).reshape(4, 4),
}


# ---------------------------------------------------------------------------
# row 1: a dark frame is where the six inline sites break
# ---------------------------------------------------------------------------


def test_inline_division_by_max_makes_nan_on_a_dark_frame() -> None:
    """0/0. Pinned because it is the failure the repo warns about elsewhere."""
    dark = np.zeros((4, 4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = dark / dark.max()
    assert not np.isfinite(out).all()
    assert any("divide" in str(w.message) for w in caught)


def test_the_canonical_helper_returns_zeros_there_instead() -> None:
    dark = np.zeros((4, 4))
    assert np.array_equal(normalize_pattern(dark, "peak"), np.zeros((4, 4), np.float32))
    assert np.array_equal(normalize_01(dark), np.zeros((4, 4)))


def test_the_two_helpers_agree_on_a_dark_frame_and_on_a_normal_one() -> None:
    """Where they agree, a merge would be safe -- which is only two of five rows."""
    for key in ("zeros", "normal"):
        a = np.asarray(normalize_pattern(CASES[key], "peak"), dtype=np.float64)
        b = np.asarray(normalize_01(CASES[key]), dtype=np.float64)
        assert np.allclose(a, b), f"{key}: {a.ravel()[:4]} vs {b.ravel()[:4]}"


# ---------------------------------------------------------------------------
# row 2: "normalises to [0, 1]" is not true for a non-positive peak
# ---------------------------------------------------------------------------


def test_normalize_pattern_leaves_a_non_positive_peak_untouched() -> None:
    """Documented as [0, 1]; actually returns the input when ``max <= 0``.

    Deliberate as a guard -- there is no meaningful peak to divide by -- but it
    means the return value is not in [0, 1] for that input, which the docstring
    does not say.
    """
    arr = CASES["negative_peak"]
    out = normalize_pattern(arr, "peak")
    assert np.array_equal(out, arr.astype(np.float32))
    assert out.max() < 0.0, "the [0,1] claim does not hold here"


def test_the_six_inline_sites_would_flip_the_sign_instead() -> None:
    """``-1 / -1 == 1``, so the inline form claims a value the peak form refuses."""
    arr = CASES["negative_peak"]
    assert np.allclose(arr / arr.max(), 1.0)


# ---------------------------------------------------------------------------
# rows 3-5: NaN handling, the constant case, and dtype
# ---------------------------------------------------------------------------


def test_only_normalize_pattern_scrubs_nan() -> None:
    """It calls ``nan_to_num`` first; the other two propagate."""
    arr = CASES["with_nan"]
    assert np.isfinite(normalize_pattern(arr, "peak")).all()
    assert not np.isfinite(arr / arr.max()).all()
    assert not np.isfinite(normalize_01(arr)).all()


def test_normalize_01_maps_a_constant_to_zero_where_peak_gives_one() -> None:
    arr = CASES["constant"]
    assert np.array_equal(normalize_01(arr), np.zeros((4, 4)))
    assert np.allclose(normalize_pattern(arr, "peak"), 1.0)


def test_dtypes_are_three_different_answers() -> None:
    """int32 in; int32 out, float32 out, float64 out."""
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    assert (arr / arr.max()).dtype == np.float64  # true division promotes
    assert normalize_pattern(arr, "peak").dtype == np.float32
    assert normalize_01(arr).dtype == np.float64


# ---------------------------------------------------------------------------
# the docstring that describes a delegation which does not happen
# ---------------------------------------------------------------------------


def test_normalize_01_does_not_delegate_and_is_not_peak_normalisation() -> None:
    """``normalize_01`` claims to delegate to ``normalize_pattern``.

    It does not -- it re-implements min-max inline. And ``normalize_pattern``'s
    default mode is ``peak``, so even a real delegation would compute a different
    function. Both halves of that docstring are wrong, and the test that proves
    it is a value comparison rather than a read of the prose.
    """
    arr = np.array([[0.0, 1.0], [2.0, 4.0]])
    assert np.allclose(normalize_01(arr), (arr - 0.0) / (4.0 - 0.0))  # min-max
    assert np.allclose(normalize_pattern(arr, "peak"), arr / 4.0)  # peak
    # On an array whose min is far from zero the two cannot agree:
    shifted = np.array([[10.0, 11.0], [12.0, 14.0]])
    assert not np.allclose(normalize_01(shifted), normalize_pattern(shifted, "peak"))


def test_the_two_modes_define_different_targets_not_a_better_worse_choice() -> None:
    """``peak`` normalises the maximum to 1; ``sum`` normalises the *total* to 1.

    On a constant 5.0 array those are different answers -- 1.0 versus 1/16 --
    which is the whole reason both modes exist and the reason a merge has to
    pick one rather than average them.
    """
    arr = CASES["constant"]  # 16 cells of 5.0
    assert np.allclose(normalize_pattern(arr, "peak"), 1.0)
    assert np.allclose(normalize_pattern(arr, "sum"), 1.0 / arr.size)
    assert not np.allclose(normalize_pattern(arr, "peak"), normalize_pattern(arr, "sum"))


def test_unknown_mode_is_rejected_rather_than_silently_ignored() -> None:
    with pytest.raises(ValueError, match="Unknown normalize mode"):
        normalize_pattern(CASES["normal"], "nonsense")  # type: ignore[arg-type]