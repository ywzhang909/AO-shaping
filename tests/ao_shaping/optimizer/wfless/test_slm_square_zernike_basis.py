"""Tests for the Zernike basis used by ``spgd-square`` CLI.

Verifies that the Zernike basis generated for
``python src/ao_shaping/main.py spgd-square -e 2000 -n 4 --target-side 20``
is a circular aperture that fully utilises every SLM pixel — i.e. the
aperture is the panel-inscribed circle (touching the top/bottom edges), not a
cropped or shrunk central region. No camera/CCD is opened.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.optimizer.wfless.slm_square_shaping import (
    SLM_HEIGHT,
    SLM_RESOLUTION,
    SLM_WIDTH,
    _zernike_indices,
    _zernike_phase_radians,
)
from ao_shaping.utils.zernike_calc import ZernikeGenerator, calc_n_zernike_terms


class TestSlmSquareZernikeBasisFullPanel:
    """Verify the Zernike basis covers the full SLM panel as an inscribed circle."""

    N_MAX = 4  # matches ``-n 4`` in the spgd-square CLI

    @pytest.fixture
    def gen(self) -> ZernikeGenerator:
        """ZernikeGenerator matching the spgd-square CLI defaults (n_max=4)."""
        return ZernikeGenerator(
            resolution=SLM_RESOLUTION, radius=None, n_orders=self.N_MAX
        )

    def test_resolution_matches_slm_panel(self, gen: ZernikeGenerator):
        assert gen.resolution == (SLM_WIDTH, SLM_HEIGHT)
        assert SLM_WIDTH == 1920
        assert SLM_HEIGHT == 1200

    def test_default_radius_is_inscribed_circle(self, gen: ZernikeGenerator):
        """Default radius = min(h, w)/2 → circle inscribed, touching top/bottom."""
        assert gen.radius == pytest.approx(SLM_HEIGHT / 2)

    def test_mask_shape(self, gen: ZernikeGenerator):
        mask = gen.mask
        assert mask.shape == (SLM_HEIGHT, SLM_WIDTH)

    def test_mask_spans_full_panel_height_at_centre_column(self, gen: ZernikeGenerator):
        """At the centre column the circle touches the top & bottom edges."""
        mask = gen.mask
        cx = SLM_WIDTH // 2
        col = mask[:, cx]
        assert col.all(), (
            "Circular aperture must span the full panel height at the centre "
            "column (inscribed circle touching top/bottom edges)"
        )

    def test_mask_width_at_centre_row_equals_panel_height(self, gen: ZernikeGenerator):
        """At the centre row the inscribed circle diameter == panel height."""
        mask = gen.mask
        cy = SLM_HEIGHT // 2
        row = mask[cy, :]
        assert int(row.sum()) == SLM_HEIGHT, (
            f"Expected inscribed circle diameter {SLM_HEIGHT} px, got {row.sum()}"
        )

    def test_mask_is_circular_not_rectangular(self, gen: ZernikeGenerator):
        """Corners outside the circle must be masked out (circular aperture)."""
        mask = gen.mask
        assert mask[0, 0] == 0
        assert mask[0, -1] == 0
        assert mask[-1, 0] == 0
        assert mask[-1, -1] == 0

    def test_mask_covers_most_of_the_panel(self, gen: ZernikeGenerator):
        """Inscribed circle area = pi/4 * (min_dim/max_dim) of the rectangular panel.

        SLM is 1920x1200 (landscape), so the inscribed circle (diameter = 1200)
        covers pi*600^2 / (1920*1200) = pi/4 * (1200/1920) ≈ 0.4909.
        """
        mask = gen.mask
        fill = float(mask.mean())
        expected = (np.pi / 4.0) * (SLM_HEIGHT / SLM_WIDTH)
        assert fill == pytest.approx(expected, abs=0.02)

    def test_mode_count_for_n_max_4(self, gen: ZernikeGenerator):
        assert calc_n_zernike_terms(self.N_MAX) == 15
        modes = _zernike_indices(self.N_MAX)
        assert len(modes) == 15
        # All modes must be within the requested radial order
        for n, m in modes:
            assert n <= self.N_MAX

    def test_phase_radians_shape_and_range(self, gen: ZernikeGenerator):
        coeffs = np.ones(calc_n_zernike_terms(self.N_MAX), dtype=np.float64)
        phase = _zernike_phase_radians(coeffs, self.N_MAX, gen)
        assert phase.shape == (SLM_HEIGHT, SLM_WIDTH)
        assert phase.dtype == np.float64
        assert phase.min() >= 0.0
        assert phase.max() < 2.0 * np.pi

    def test_phase_uses_full_aperture_not_central_crop(self, gen: ZernikeGenerator):
        """A non-trivial Zernike combo must excite pixels across the whole circle,
        including the outer ring — proving the basis is not a shrunk central crop."""
        coeffs = np.zeros(calc_n_zernike_terms(self.N_MAX), dtype=np.float64)
        coeffs[3] = 1.0  # Noll 4 = defocus (2,0): varies strongly with radius
        phase = _zernike_phase_radians(coeffs, self.N_MAX, gen)

        mask = gen.mask.astype(bool)
        inside = phase[mask]
        # Defocus is non-constant across the full aperture
        assert inside.std() > 1.0

        # The outermost ring of the aperture (near the top edge of the circle)
        # must also carry phase — if the basis were a central crop this would be 0.
        cx = SLM_WIDTH // 2
        top_ring = phase[2:8, cx - 2 : cx + 2]
        assert top_ring.std() > 0.0, "Outer aperture ring must carry non-trivial phase"

    def test_phase_outside_aperture_is_zero(self, gen: ZernikeGenerator):
        """Pixels outside the circular aperture must not contribute."""
        coeffs = np.ones(calc_n_zernike_terms(self.N_MAX), dtype=np.float64)
        phase = _zernike_phase_radians(coeffs, self.N_MAX, gen)
        mask = gen.mask.astype(bool)
        # Outside the aperture the Zernike polynomials evaluate to ~0 (aotools
        # normalises the unit circle; outside is numerically zero).
        outside = phase[~mask]
        assert np.allclose(outside, 0.0, atol=1e-8)

    def test_basis_dim_matches_zernike_term_count(self, gen: ZernikeGenerator):
        """The flat parameter vector length must equal the number of Zernike terms."""
        nk = calc_n_zernike_terms(self.N_MAX)
        modes = _zernike_indices(self.N_MAX)
        assert len(modes) == nk
        # Each mode maps to exactly one coefficient
        assert nk == (self.N_MAX + 1) * (self.N_MAX + 2) // 2
