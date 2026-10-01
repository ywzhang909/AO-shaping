"""Contract tests for the simulated camera's noise model.

The defect these lock down
--------------------------
``far_field_noisy`` summed zero-mean read noise and then applied
``clip(img, 0, None)``. Clipping a symmetric distribution at zero rectifies it:
half the pixels that should have read slightly negative are pinned to 0, turning
zero-mean noise into a **DC pedestal proportional to the pixel count**.

Measured on the 1200x1920 panel with the default ``noise_adu``:

    shot noise only (signal-proportional)   1.005x the signal   <- correct
    read noise after ``clip(·, 0, None)``   4598x  the signal   <- defect

The spot holds ~150 px of signal while the frame has 2.3 M pixels, so the
pedestal carried ~3000x the light. The optimiser's power-in-bucket metric
divides by the frame total, so it was measuring that pedestal: ``pib`` sat at
~0.0145 and SPGD's ``J(+d) - J(-d)`` was pure noise, which is why ``pib`` and
``combined`` ran to completion while never moving.

A real detector's floor is flat *below zero* in raw ADU and is clipped once, at
digitisation -- not re-rectified on every frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem


@pytest.fixture
def system():
    s = SimPibSystem(noise_adu=5.0)
    s.set_phase_rad(np.zeros((s.slm_h, s.slm_w)))
    s.dm_optics.set_voltages(np.zeros(s.dm_optics.n_actuators))
    return s


class TestNoiseFloorDoesNotDominate:
    def test_read_noise_must_not_create_a_pedestal(self, system):
        """The frame's mean offset must stay far below the signal's peak.

        The *total* is a bad probe here: zero-mean noise summed over 2.3 M pixels
        has a standard deviation of ``sigma * sqrt(N)`` ~ 758, inherently the same
        order as the 152 signal total, so the sum fluctuates even when the noise is
        perfectly centred. The pedestal defect shows up in the *mean*, which is
        what rectification destroys.
        """
        clean = system.far_field()
        noisy = system.far_field_noisy()
        assert abs(float(noisy.mean())) < float(clean.max()) * 0.01, (
            f"frame mean offset {float(noisy.mean()):.4f} vs peak "
            f"{float(clean.max()):.1f}: read noise is rectified into a pedestal"
        )

    def test_dark_region_stays_dark(self, system):
        """Far from the spot there is no light, so there must be no pedestal."""
        noisy = system.far_field_noisy()
        corner = noisy[:200, :200]
        signal_peak = float(system.far_field().max())
        assert abs(float(corner.mean())) < signal_peak * 0.01, (
            "an unilluminated corner is glowing -- the noise floor is not zero-mean"
        )


class TestSignalStillVisible:
    def test_spot_survives_the_noise(self, system):
        noisy = system.far_field_noisy()
        assert float(noisy.max()) > 0
        assert np.isfinite(noisy).all()

    def test_bucket_holds_the_signal(self, system):
        """Noise must not displace the spot out of the bucket.

        Deliberately *not* asserted as a fraction of the noisy frame total: with
        per-pixel read noise over 2.3 M pixels the total carries ~1458 of
        statistical noise (sigma*sqrt(N) ~ 758), so a ratio against it is
        dominated by noise regardless of the optics. What the model must
        guarantee is that the bucket still collects the light.
        """
        clean = system.far_field()
        noisy = system.far_field_noisy()
        peak = np.unravel_index(np.argmax(clean), clean.shape)
        r = 18
        sl = (slice(peak[0] - r, peak[0] + r), slice(peak[1] - r, peak[1] + r))
        assert clean[sl].sum() > clean.sum() * 0.8, (
            "the clean spot does not sit in the bucket"
        )
        assert noisy[sl].sum() > clean[sl].sum() * 0.8, (
            "noise displaced the signal out of the bucket"
        )
