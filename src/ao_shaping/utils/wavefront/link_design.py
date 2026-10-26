"""Telescope far-field link design: bidirectional scale conversion + DFT far-field engine.

Single source of truth for the telescope diagnostic link of issue #204
(``notebooks/telescope_link_spot_size_design.ipynb``). The telescope (M_t = 10)
inserts an extra magnification between the SLM and the diagnostic focus that the
plain 2f SLM-conjugate mappings elsewhere in this package do not carry:

* Pupil plane = exit pupil (90 mm). The SLM window 1125x1125 @ 8 um maps 1:1 in
  pixel index to 1125x1125 @ 80 um on the exit pupil.
* Effective focal length ``f_eff = f_c * M_t = 2 m``: every focal-plane scale
  law uses ``f_eff``, so ``W_ccd = f_eff * theta_far`` is exactly consistent
  with the conversion coefficient ``L / (f_c * M_t) = 500``.
* DFT kernel ``E(u) = B @ pupil @ B.T`` with
  ``B[j, i] = exp(-2j*pi*u[j]*x[i]/(lam*f_eff))`` — same physics as
  :mod:`ao_shaping.utils.wavefront.fraunhofer`, implemented in plain numpy so
  the utils package stays torch-free.

Locked design point (issue #204, do not change):

================  ==================
quantity        value
================  ==================
lambda          1064 nm
L (target)      1 km
M_t             10
f_c             200 mm
beam diameter   9 mm (SLM active window)
exit pupil      90 mm
objective       100 mm (nominal)
SLM pixel       8 um, 1125x1125
CCD pixel       2.2 um
carrier ramp    2 mrad (SLM side)
================  ==================

Golden acceptance values (asserted cell-by-cell in the notebook and by
``tests/ao_shaping/utils/test_link_design_param_sweep.py``): 200 um / 91 px
design point, Airy first zero 28.8 um / 13 px (D4FT measured 28.0 um),
carrier offset 400 um / 182 px (measured 397 um), nu_max 470 m^-1 / 266 px,
carrier 1879 m^-1 / 67 px, coherent cells 8.46 (90 mm) / 9.40 (100 mm),
depth of focus +/-131 m, field +/-1 mm = +/-455 px.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "TelescopeLinkParams",
    "conversion_scale",
    "far_to_ccd",
    "ccd_to_far",
    "far_field_dft",
    "circular_pupil",
    "square_pupil",
    "carrier_phase",
    "grating_factor",
    "airy_first_zero",
    "sinc_first_zero",
    "carrier_shift",
]

_B_CACHE: dict[tuple[float, float, float, int], np.ndarray] = {}


class TelescopeLinkParams:
    """Locked parameters of the telescope diagnostic link (issue #204).

    All attributes in metres / radians / plain units unless noted.
    """

    def __init__(
        self,
        wavelength: float = 1.064e-6,
        target_distance: float = 1.0e3,
        telescope_magnification: float = 10.0,
        focus_focal_length: float = 0.200,
        beam_diameter: float = 0.009,
        exit_pupil_diameter: float = 0.090,
        objective_diameter: float = 0.100,
        slm_pixel: float = 8.0e-6,
        ccd_pixel: float = 2.2e-6,
        pupil_grid: int = 1125,
        carrier_ramp: float = 2.0e-3,
    ) -> None:
        self.wavelength = wavelength
        self.target_distance = target_distance
        self.telescope_magnification = telescope_magnification
        self.focus_focal_length = focus_focal_length
        self.beam_diameter = beam_diameter
        self.exit_pupil_diameter = exit_pupil_diameter
        self.objective_diameter = objective_diameter
        self.slm_pixel = slm_pixel
        self.ccd_pixel = ccd_pixel
        self.pupil_grid = pupil_grid
        self.carrier_ramp = carrier_ramp

    @property
    def exit_pupil_pixel(self) -> float:
        """Exit-pixel pitch: SLM window maps 1:1 in pixel index (80 um)."""
        return self.slm_pixel * self.telescope_magnification

    @property
    def effective_focal_length(self) -> float:
        """f_eff = f_c * M_t = 2 m: use for every focal-plane scale law."""
        return self.focus_focal_length * self.telescope_magnification

    @property
    def scale(self) -> float:
        """Far-field / CCD linear coefficient L/(f_c*M_t) = 500."""
        return self.target_distance / self.effective_focal_length


def conversion_scale(params: TelescopeLinkParams) -> float:
    """Return the far-field / CCD linear coefficient (500 at the design point)."""
    return params.scale


def far_to_ccd(
    w_far_m: float, params: TelescopeLinkParams
) -> tuple[float, float]:
    """Convert a far-field spot size (m, at ``target_distance``) to the CCD.

    Returns ``(w_ccd_m, w_ccd_in_ccd_pixels)``.
    """
    alpha = w_far_m / params.target_distance
    w_ccd = params.effective_focal_length * alpha
    return w_ccd, w_ccd / params.ccd_pixel


def ccd_to_far(w_ccd_m: float, params: TelescopeLinkParams) -> float:
    """Convert a CCD spot size (m) back to the far field (m)."""
    return w_ccd_m * params.scale


def far_field_dft(
    pupil: np.ndarray,
    dx: float,
    lam: float,
    f_eff: float,
    du_out: float = 2.15e-6,
    n_out: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """DFT far-field engine (issue #204 locked grid by default).

    Args:
        pupil: ``(N, N)`` complex pupil-plane field, pixel pitch ``dx`` (m).
        dx: Pupil-plane pixel pitch (m). Design: ``exit_pupil_pixel`` = 80 um.
        lam: Wavelength (m).
        f_eff: Effective focal length (m). Design: ``f_c * M_t`` = 2 m.
        du_out: Focal-plane pixel pitch (m). Design: 2.15 um (~CCD 2.2 um).
        n_out: Focal-plane grid side. Design: 1024 (half-window +/-1.1 mm).

    Returns:
        ``(E, u)``: focal-plane field and coordinate vector (origin at the
        focal centre, spacing ``du_out``).
    """
    n_in = pupil.shape[0]
    key = (float(lam), float(f_eff), float(dx), int(n_out))
    B = _B_CACHE.get(key)
    if B is None:
        x = (np.arange(n_in) - n_in // 2) * dx
        u = (np.arange(n_out) - n_out // 2) * du_out
        B = np.exp(-2j * np.pi * np.outer(u, x) / (lam * f_eff))
        _B_CACHE[key] = B
    u = (np.arange(n_out) - n_out // 2) * du_out
    return B @ pupil @ B.T, u


def circular_pupil(n: int, dx: float, diameter: float) -> np.ndarray:
    """Circular (Hadamard-free) pupil mask of ``diameter`` on an ``n`` grid."""
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x)
    return (np.hypot(xx, yy) < diameter / 2.0).astype(float)


def square_pupil(n: int, dx: float, width: float) -> np.ndarray:
    """Square pupil mask of side ``width`` on an ``n`` grid (x-aligned)."""
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x)
    return ((np.abs(xx) < width / 2.0) & (np.abs(yy) < width / 2.0)).astype(float)


def carrier_phase(n: int, dx: float, theta: float, lam: float) -> np.ndarray:
    """Linear phase ramp (carrier) over the x axis, focal shift ``f_eff*theta``."""
    x = (np.arange(n) - n // 2) * dx
    return np.exp(2j * np.pi * theta * x / lam)


def grating_factor(n: int, dx: float, nu: float, m: float = 0.5) -> np.ndarray:
    """Cosine grating of spatial frequency ``nu`` (m^-1), modulation ``m``."""
    x = (np.arange(n) - n // 2) * dx
    return 1.0 + m * np.cos(2.0 * np.pi * nu * x)


def airy_first_zero(lam: float, f: float, diameter: float) -> float:
    """Analytic first-zero radius of a circular pupil: 1.22*lam*f/D (m)."""
    return 1.22 * lam * f / diameter


def sinc_first_zero(lam: float, f: float, width: float) -> float:
    """Analytic half-width to the first zero of a square pupil: lam*f/w (m)."""
    return lam * f / width


def carrier_shift(f_eff: float, theta: float) -> float:
    """Focal-plane shift produced by a carrier of angle ``theta`` (m)."""
    return f_eff * theta
