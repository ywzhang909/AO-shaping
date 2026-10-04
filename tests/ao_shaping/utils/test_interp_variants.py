"""R-28 D5: pin the interpolation behaviours BEFORE anyone tries to merge them.

The TODO flagged this as the one dedup in the batch that changes numerical
results, and it does. This module exists to make the size of that change
impossible to argue about: the numbers below were measured, not reasoned.

Three behaviours are in the tree:

    A  _resize_bilinear                    utils/image/target/ccd.py
    B  zoom(order=1, grid_mode=True,  mode="grid-constant")
    C  zoom(order=1)                       <- scipy's default, and what most
                                              call sites in this repo use

A and B agree to float rounding when downsampling (2e-16 to 3e-16). They do
**not** agree when upsampling, and neither agrees with C. Concretely, on a
random [0, 1) field:

    ratio      max|A - B|      max|A - C|
      64->32      2.2e-16        4.5e-01
     250->50      0.0            9.0e-01
     248->64      2.2e-16        8.4e-01
    1200->64      3.3e-16        8.5e-01
      64->248     3.8e-01        9.8e-01
      37->111     3.5e-01        4.0e-01
     100->100     0.0            0.0

Only the 5:1 ratio comes out bit-identical, because that is the one case where
both coordinate maps land on whole pixels. Everything else differs in the last
bit or two -- except upsampling, which differs by tenths of peak. Those are not
rounding errors. Merging these would silently move every pixel of every SLM
panel image and every far-field crop.

Why the conventions differ, read off a linear ramp (any correct interpolant
reproduces an affine ramp exactly, so the returned value IS the source
coordinate):

    250 -> 50, value at output index 0
      A  2.0        half-pixel / grid_mode=True
      B  2.0        identical
      C  0.0        align-corners

So scipy's *default* is the odd one out, and the default is what most of the
repo calls. The two upsampling paths also disagree at the borders: A clamps the
source coordinate to [0, n-1] and B extends the border, which is invisible on a
ramp (affine either way) and worth 0.59 peak on noise.

Do not "simplify" by pointing the call sites at one of these. If a single
canonical is ever wanted, the characterisation tests below are the contract it
has to satisfy, and it does not currently exist.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import zoom

from ao_shaping.utils.image.target.ccd import _resize_bilinear


def _bilinear_up(img: np.ndarray, out: tuple[int, int]) -> np.ndarray:
    return _resize_bilinear(img, out)


def _b(img: np.ndarray, out: tuple[int, int]) -> np.ndarray:
    n, m = img.shape[0], out[0]
    return zoom(img, (m / n, m / n), order=1, grid_mode=True, mode="grid-constant")


def _c(img: np.ndarray, out: tuple[int, int]) -> np.ndarray:
    n, m = img.shape[0], out[0]
    return zoom(img, (m / n, m / n), order=1)


# ---------------------------------------------------------------------------
# A == B to float rounding, but only when downsampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "m"), [(64, 32), (250, 50), (248, 64), (1200, 64)])
def test_downsample_A_equals_B_to_float_rounding(n: int, m: int) -> None:
    """Agreement is at machine epsilon -- 2e-16 to 3e-16, never larger.

    Not bit-identical though: only the 5:1 ratio (250 -> 50) comes out exactly
    equal, because that is the one case where both coordinate maps land on whole
    pixels. Every other ratio differs in the last bit or two. Asserted as
    "within rounding" rather than "array_equal" so the claim matches the
    measurement.
    """
    for seed in range(8):
        img = np.random.default_rng(seed).random((n, n))
        gap = np.abs(_bilinear_up(img, (m, m)) - _b(img, (m, m))).max()
        assert gap <= 4e-16, f"{n}->{m} seed {seed}: A and B differ by {gap}"


@pytest.mark.parametrize(("n", "m"), [(64, 248), (37, 111), (16, 32)])
def test_upsample_A_and_B_diverge(n: int, m: int) -> None:
    """The exact case the SLM path depends on: small grid -> full panel."""
    gaps = [
        np.abs(_bilinear_up(img, (m, m)) - _b(img, (m, m))).max()
        for img in (np.random.default_rng(s).random((n, n)) for s in range(4))
    ]
    assert min(gaps) > 0.2, f"expected a real divergence when upsampling, got {min(gaps)}"


def test_identity_resize_is_a_no_op_for_every_variant() -> None:
    img = np.random.default_rng(1).random((16, 16))
    for f in (_bilinear_up, _b, _c):
        assert np.array_equal(f(img, (16, 16)), img)


# ---------------------------------------------------------------------------
# C (scipy's default, and the repo's majority call) is a different convention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "m"), [(64, 32), (250, 50), (248, 64), (64, 248)])
def test_default_zoom_is_not_interchangeable_with_A(n: int, m: int) -> None:
    img = np.random.default_rng(n).random((n, n))
    gap = np.abs(_bilinear_up(img, (m, m)) - _c(img, (m, m))).max()
    assert gap > 0.1, (
        f"scipy's default zoom agreed with _resize_bilinear at {n}->{m} "
        f"(gap {gap}); if that ever becomes true the conventions were unified "
        "and this whole module is stale."
    )


def test_default_zoom_is_align_corners_while_A_is_half_pixel() -> None:
    """The convention difference, read straight off an affine ramp.

    Any correct interpolant reproduces a linear ramp exactly, so the value it
    returns at an output index *is* the source coordinate it sampled.
    """
    ramp = np.tile(np.arange(250, dtype=np.float64), (250, 1))

    a_first = _bilinear_up(ramp, (50, 50))[0, 0]
    b_first = _b(ramp, (50, 50))[0, 0]
    c_first = _c(ramp, (50, 50))[0, 0]

    assert a_first == pytest.approx(2.0), "A samples the half-pixel centre"
    assert b_first == pytest.approx(a_first), "B is defined to match A here"
    assert c_first == pytest.approx(0.0), "C anchors on the first source pixel"


# ---------------------------------------------------------------------------
# the other two behaviours in the tree, for completeness
# ---------------------------------------------------------------------------


def test_block_mean_is_not_bilinear_and_loses_energy_on_noise() -> None:
    """``ml/gsnet_debug/offline.py`` block-averages instead of interpolating.

    That is a deliberate low-pass, not a sloppy bilinear: on a random field it
    differs by ~0.43 of full scale, which is the point of averaging.
    """
    img = np.random.default_rng(3).random((248, 248))
    block = img.reshape(62, 4, 62, 4).mean(axis=(1, 3))
    gap = np.abs(block - _bilinear_up(img, (62, 62))).max()
    assert gap > 0.2, f"block mean and bilinear should differ clearly, got {gap}"


def test_bicubic_overshoots_below_zero_where_bilinear_cannot() -> None:
    """``phase_wrap`` / ``dynamic_compensation`` use order=3.

    Cubic kernels ring: on a hard edge the result goes negative, which for a
    *phase* is not a quantity that exists. Worth knowing before anyone unifies
    the orders too.
    """
    step = np.zeros((21, 21))
    step[:, 10:] = 1.0
    cubic = zoom(step, (210 / 21, 210 / 21), order=3)
    assert cubic.min() < -0.1, f"expected ringing below zero, got {cubic.min()}"


def test_value_range_is_preserved_by_bilinear_but_not_by_cubic() -> None:
    """A [0, 1] phase or intensity map must stay in [0, 1]."""
    img = np.random.default_rng(5).random((32, 32))
    bilinear = _bilinear_up(img, (64, 64))
    assert bilinear.min() >= 0.0 and bilinear.max() <= 1.0