"""float64 precision is unreachable: the panel quantises before it can matter.

The working precision of the forward model is not a free choice, because the phase
that finally reaches the LCOS panel is quantised to the device's 10-bit gray map
(2*pi = 993 levels at 1064 nm, i.e. 6.33e-3 rad per level). A float32 mantissa is
five orders of magnitude finer than one gray level, so the two precisions must
produce the *same* pattern on the panel -- not merely a close one.

These tests pin that byte-level equivalence rather than a tolerance, because a
tolerance would leave open the possibility that some array finally crosses a
quantisation boundary and shifts a pixel by one level. If that ever happens, this
file is where it should surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.slm.phase_display import phase_to_slm_grayscale

#: 2*pi over the device's gray levels at 1064 nm -- the smallest phase change the
#: panel can express at all.
GRAY_LEVEL_RAD = 2.0 * np.pi / 993.0


def _random_phase(region: int, seed: int, spread: float = 4.0) -> np.ndarray:
    """A raw unwrapped-radian pupil phase, the shape a generator really emits."""
    return np.random.default_rng(seed).normal(0.0, 1.0, (region, region)) * spread


class TestByteIdenticalQuantisation:
    @pytest.mark.parametrize("region", [64, 128, 256, 512])
    @pytest.mark.parametrize("spread", [1.0, 4.0, 12.0])
    def test_the_gray_maps_are_identical(self, region: int, spread: float) -> None:
        """Not ``allclose`` -- identical. A one-level shift would be a real change."""
        phase64 = _random_phase(region, seed=11, spread=spread)
        phase32 = phase64.astype(np.float32).astype(np.float64)
        gray64 = phase_to_slm_grayscale(phase64, slm=None)
        gray32 = phase_to_slm_grayscale(phase32, slm=None)
        assert gray64.shape == gray32.shape
        differing = int((gray64 != gray32).sum())
        assert differing == 0, (
            f"{differing}/{gray64.size} pixels shifted between precisions at "
            f"region={region}, spread={spread}; float32 is no longer equivalent"
        )
        assert np.array_equal(gray64, gray32)

    @pytest.mark.parametrize("region", [128, 512])
    def test_the_phase_difference_is_far_below_one_gray_level(self, region: int) -> None:
        """The headroom, stated in the unit that matters: gray levels."""
        phase64 = _random_phase(region, seed=23)
        phase32 = phase64.astype(np.float32).astype(np.float64)
        worst = float(np.abs(phase64 - phase32).max())
        assert worst < 0.1 * GRAY_LEVEL_RAD, (
            f"float32 phase error {worst:.3e} rad is {worst / GRAY_LEVEL_RAD:.2%} of "
            "one gray level -- the equivalence argument needs it well inside"
        )

    def test_a_deliberate_half_level_shift_does_change_the_map(self) -> None:
        """Guard against the assertions above passing for the wrong reason.

        If ``phase_to_slm_grayscale`` ignored its input, every equivalence test here
        would pass trivially. Half a gray level is enough to move some pixels.
        """
        phase = _random_phase(256, seed=5)
        shifted = phase + 0.5 * GRAY_LEVEL_RAD
        assert not np.array_equal(
            phase_to_slm_grayscale(phase, slm=None),
            phase_to_slm_grayscale(shifted, slm=None),
        )
