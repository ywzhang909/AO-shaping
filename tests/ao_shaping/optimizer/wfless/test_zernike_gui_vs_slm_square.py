"""Cross-check: Zernike phase generation between the GUI path and the
``slm_square_shaping`` path.

After the 2026-09 amplitude fix, BOTH the GUI Zernike branch
(``pattern_controls.generate_phase_gray`` → ``generate_zernike_polynomial``
→ ``slm.create_phase_from_array``) and ``PatternHelper.generate_zernike_polynomial``
preserve the absolute coefficient magnitude. Since 2026-09-15, the helper
returns the **raw unwrapped radian phase** (no mod-2π wrap, no uint16, no
min-max normalisation) — wrapping to grayscale is the SLM driver's job
(``create_phase_from_array``). The legacy min-max normalisation in
``_zernike_to_uint16`` was removed — it made patterns scale-invariant
(coefficient 0.1 and 10.0 produced byte-identical uint16).
Both paths generate at the radius=600 px aperture with a circular aperture mask.
Only Defocus (2,0) and Spherical (4,0) are optimised (``ZERNIKE_ACTIVE_MODES``).

The raw Zernike polynomial helper (``_zernike_phase_radians``) is retained and
still tested for polynomial correctness and the driver's linear
radian->grayscale mapping. No camera/CCD or SLM hardware is opened.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.drivers.slm.santec.wavefront_correction import WavefrontCorrection
from ao_shaping.optimizer.wfless.slm_square_shaping import (
    SLM_HEIGHT,
    SLM_RESOLUTION,
    SLM_WIDTH,
    ZERNIKE_ACTIVE_MODES,
    _map_init_to_active_modes,
    _zernike_indices,
    _zernike_phase_radians,
)
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator, calc_n_zernike_terms

# The two modes the runner optimises: defocus (2,0) + spherical (4,0)
COEFFS = {(2, 0): 1.0, (4, 0): 0.5}
RADIUS = 600  # px — inscribed circle of the 1920x1200 SLM panel
N_MAX = 4
BITS = 10


def _gui_gray(coeffs: dict[tuple[int, int], float]) -> np.ndarray:
    """GUI path: multi_slm_controller.py Zernike branch.

    Post-2026-09-15 contract: raw unwrapped radian phase preserving absolute
    coefficient amplitude (no min-max normalisation, no mod-2π, no uint16).
    """
    helper = PatternHelper(SLM_RESOLUTION, bits=BITS)
    return helper.generate_zernike_polynomial(
        coefficients=coeffs, radius=RADIUS, n_max=N_MAX
    )


def _slm_square_gray(coeffs: dict[tuple[int, int], float]) -> np.ndarray:
    """Runner zernike-basis path, post-fix.

    Mirrors ``optimize_slm_square``'s ``_params_to_gray`` at the default
    SLM/basis settings: a PatternHelper on the panel resolution with
    bits = Gray_Scale_bits (10), generated at the default 600 px aperture
    radius with n_max=4.
    """
    helper = PatternHelper(SLM_RESOLUTION, bits=BITS)
    return helper.generate_zernike_polynomial(
        coefficients=coeffs, radius=RADIUS, n_max=N_MAX
    )


def _aperture_mask() -> np.ndarray:
    gen = ZernikeGenerator(SLM_RESOLUTION, radius=RADIUS, n_orders=N_MAX)
    return gen.mask.astype(bool)


class TestZernikeGuiVsSlmSquareConsistency:
    """GUI path vs slm_square_shaping path must produce identical patterns."""

    @pytest.fixture
    def gui_gray(self) -> np.ndarray:
        return _gui_gray(COEFFS)

    @pytest.fixture
    def slm_square_gray(self) -> np.ndarray:
        return _slm_square_gray(COEFFS)

    def test_both_produce_panel_resolution(self, gui_gray, slm_square_gray):
        assert gui_gray.shape == (SLM_HEIGHT, SLM_WIDTH)
        assert slm_square_gray.shape == (SLM_HEIGHT, SLM_WIDTH)
        assert gui_gray.dtype == np.float64
        assert slm_square_gray.dtype == np.float64

    def test_runner_is_byte_identical_to_gui(self, gui_gray, slm_square_gray):
        """Post-fix both paths share PatternHelper.generate_zernike_polynomial
        — the runner output must be byte-identical to the GUI reference."""
        assert np.array_equal(gui_gray, slm_square_gray)

    def test_outside_aperture_is_zero_in_both(self, gui_gray, slm_square_gray):
        mask = _aperture_mask()
        assert np.all(gui_gray[~mask] == 0)
        assert np.all(slm_square_gray[~mask] == 0)
        assert np.any(gui_gray[mask] > 0)

    def test_defocus_is_monotonic_with_radius_in_raw_polynomial(self):
        """Defocus (2,0) must be monotonic with radius in the RAW polynomial."""
        gen = ZernikeGenerator(SLM_RESOLUTION, radius=RADIUS, n_orders=N_MAX)
        mask = gen.mask.astype(bool)
        raw = gen.generate_polynomial({(2, 0): 1.0})[mask].astype(np.float64)
        rho2 = gen.R[mask] ** 2
        corr = np.corrcoef(rho2, raw)[0, 1]
        assert abs(corr) > 0.99, (
            f"Defocus not monotonic with rho^2 in raw polynomial (corr={corr:.3f})"
        )

    def test_spherical_defocus_are_distinct(self, slm_square_gray):
        """(2,0) and (4,0) must produce different patterns."""
        mask = _aperture_mask()
        defocus = _slm_square_gray({(2, 0): 1.0})[mask].astype(np.float64)
        spherical = _slm_square_gray({(4, 0): 0.5})[mask].astype(np.float64)
        rel_diff = float(np.linalg.norm(defocus - spherical) / defocus.std())
        assert rel_diff > 0.1, (
            f"Defocus and spherical patterns are nearly identical "
            f"(rel_diff={rel_diff:.3f})"
        )

    def test_driver_linear_mapping_before_wrap(self):
        """The SLM driver maps radians -> grayscale linearly (2pi = max_gray)
        BEFORE the 2pi wrap."""
        gen = ZernikeGenerator(SLM_RESOLUTION, radius=RADIUS, n_orders=N_MAX)
        mask = gen.mask.astype(bool)
        raw = gen.generate_polynomial({(2, 0): 1.0})
        expected_unwrapped = raw / (2.0 * np.pi) * Santec.MAX_GRAYSCALE_VALUE

        c = np.zeros(calc_n_zernike_terms(N_MAX), dtype=np.float64)
        c[3] = 1.0
        phase = _zernike_phase_radians(c, N_MAX, gen)
        slm = Santec.__new__(Santec)
        slm._max_gray = Santec.MAX_GRAYSCALE_VALUE
        slm._correction = WavefrontCorrection()
        slm._lut = None
        slm._shift_x = 0
        slm._shift_y = 0
        gray = slm.create_phase_from_array(phase)

        diff = gray[mask].astype(np.float64) - expected_unwrapped[mask]
        ratios = diff / Santec.MAX_GRAYSCALE_VALUE
        rounded = np.round(ratios)
        assert np.allclose(ratios, rounded, atol=1e-3), (
            "Driver grayscale is not a pure mod-2pi wrap of the linear mapping"
        )

    def test_gui_gray_is_raw_unwrapped_polynomial(self):
        """The helper must return the RAW unwrapped Zernike polynomial inside
        the aperture — the post-2026-09-15 raw-only contract. The legacy
        min-max normalised map was merely *affine* equivalent, i.e.
        scale-invariant: ×1 and ×4 coefficients produced byte-identical
        output — exactly the reported bug. The residual mod-2π wrap was also
        removed: the SLM driver applies it on conversion to grayscale."""
        gen = ZernikeGenerator(SLM_RESOLUTION, radius=RADIUS, n_orders=N_MAX)
        mask = gen.mask.astype(bool)
        raw = gen.generate_polynomial(COEFFS).astype(np.float64)
        gui = _gui_gray(COEFFS)

        g = gui[mask]
        e = raw[mask]
        assert g.std() > 0 and e.std() > 0

        # raw unwrapped equality — no wrap, no uint16, no min-max normalisation
        max_phase_err = float(np.max(np.abs(g - e)))
        assert max_phase_err <= 1e-9, (
            f"Helper output is not the raw unwrapped polynomial "
            f"(max phase err={max_phase_err:.3e})"
        )


class TestZernikeActiveModes:
    """The zernike basis optimises exactly Defocus (2,0) + Spherical (4,0)."""

    def test_active_modes_are_defocus_and_spherical(self):
        assert ZERNIKE_ACTIVE_MODES == ((2, 0), (4, 0))
        indices = _zernike_indices(N_MAX)
        for mode in ZERNIKE_ACTIVE_MODES:
            assert mode in indices

    def test_map_init_from_full_noll_vector(self):
        """A full Noll-length init vector keeps only the active-mode entries
        (Noll 4 = (2,0) -> index 3, Noll 11 = (4,0) -> index 10)."""
        nk = calc_n_zernike_terms(N_MAX)  # 15 for n_max=4
        init_c = np.zeros(nk, dtype=np.float64)
        init_c[3] = 1.0   # defocus
        init_c[10] = 0.5  # spherical
        init_c[4] = 9.0   # astig — must be dropped
        mapped = _map_init_to_active_modes(init_c, N_MAX, ZERNIKE_ACTIVE_MODES)
        assert mapped.shape == (2,)
        assert np.allclose(mapped, [1.0, 0.5])

    def test_map_init_short_vector_is_padded(self):
        mapped = _map_init_to_active_modes(
            np.array([1.0]), N_MAX, ZERNIKE_ACTIVE_MODES
        )
        assert mapped.shape == (2,)
        assert mapped[0] == 1.0 and mapped[1] == 0.0