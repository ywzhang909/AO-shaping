"""Optional 0-order-deflecting phase overlay for the 2f Fourier SLM bench.

This module produces an **additive panel-phase term, in raw unwrapped radians**,
that is summed onto the shaped phase *before* the driver converts radian phase
to grayscale.  It never wraps and never normalises: the single radian->grayscale
wrap point remains ``Santec.create_phase_from_array`` (see the "SLM phase
raw-only contract" red line in the repo ``AGENTS.md``).

Why the mode distinction matters on a 2f bench
----------------------------------------------
On this bench the SLM sits at the **front focal plane** of the 2f transform
lens, so the field at the SLM is Fourier-transformed onto the camera:

* ``BLAZE`` (a linear "blaze" phase ramp) is a *prism*: it displaces the whole
  spot **laterally** on the camera.  A ramp of 2*pi over ``P`` panel pixels moves
  the spot by ``TILT_SHIFT_SCALE / P`` camera pixels.
* ``FRESNEL_ON_AXIS`` (an on-axis quadratic lens phase) acts as an extra lens
  placed at the front focal plane: it changes only the **axial** focus, so the
  spot stays put and merely defocuses / bloats.  It gives **no lateral
  displacement** -- that is the physically correct answer and this module does
  not fake one.
* ``FRESNEL_OFF_AXIS`` (a quadratic centred away from the panel axis) *does*
  steer laterally, because expanding the centred quadratic about the offset
  produces a residual **linear** term -- that linear term is the prism that
  actually steers the spot.  The pure quadratic part only defocuses.

``TILT_SHIFT_SCALE`` (imported from
:mod:`ao_shaping.utils.slm.bench_constants`) is the **measured** bench constant
for the blaze case.  It was measured on 2026-09-30 and the module that defines it
explicitly warns that the naive ``f * lambda / d_slm`` is ~17.5x too large here
and that the **panel <-> camera axes are swapped 90 deg**.  So the blaze shift is
derived from ``TILT_SHIFT_SCALE``, never from first principles.

Panel <-> camera 90 deg axis swap
---------------------------------
The panel and camera frames are rotated 90 deg relative to each other (again per
``bench_constants``).  Consequently a ramp along **panel x** displaces the spot
along **camera y**, and a ramp along **panel y** displaces it along **camera x**.
``predicted_shift_px`` applies exactly that swap (panel x -> camera y); only the
**magnitude** is calibrated (``TILT_SHIFT_SCALE``), the sign is a fixed
convention of this module because no signed measurement is available.

Layering
--------
Imports are limited to the standard library, numpy, the zero-import
``bench_constants`` leaf and ``pattern_helper``.  In particular this module does
**not** import ``ao_shaping.drivers``, ``ao_shaping.gui`` or
``ao_shaping.algorithm`` (layering red line).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ao_shaping.utils.slm.bench_constants import TILT_SHIFT_SCALE
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper

__all__ = [
    "ZeroOrderOverlay",
    "OverlayParams",
    "apply_zero_order_overlay",
    "predicted_shift_px",
    "grating_first_order_efficiency",
    "camera_pixel_um_from_bench",
]

_TWO_PI = 2.0 * np.pi

# Module-level cache for PatternHelper instances keyed by (width, height).
# This avoids rebuilding full coordinate grids on every call in a closed loop.
_HELPER_CACHE: dict[tuple[int, int], PatternHelper] = {}


def _get_helper(width: int, height: int) -> PatternHelper:
    """Return a cached PatternHelper for the given panel geometry."""
    key = (width, height)
    helper = _HELPER_CACHE.get(key)
    if helper is None:
        helper = PatternHelper(resolution=(width, height))
        _HELPER_CACHE[key] = helper
    return helper


class ZeroOrderOverlay(str, Enum):
    """Which optional 0-order overlay (if any) to sum onto the panel phase."""

    NONE = "none"
    BLAZE = "blaze"
    FRESNEL_ON_AXIS = "fresnel_on_axis"
    FRESNEL_OFF_AXIS = "fresnel_off_axis"


@dataclass(frozen=True)
class OverlayParams:
    """Parameters for :func:`apply_zero_order_overlay` / :func:`predicted_shift_px`.

    All lengths/angles are plain SI-friendly engineering units; conversion to SI
    metres happens inside the functions so callers never have to think about it.

    Attributes:
        mode: Overlay kind.  Coerced from its string value in ``__post_init__``.
        blaze_period_px: Ramp period, in panel pixels, of the ``BLAZE`` grating.
            Must be > 0 (and >= 2 to stay above the Nyquist floor, see
            :func:`grating_first_order_efficiency`).
        blaze_direction: ``"vertical"`` (ramp along panel **x**, the same
            convention as :meth:`PatternHelper.linear_grating`) or
            ``"horizontal"`` (ramp along panel **y**).
        fresnel_focal_mm: Focal length of the Fresnel lens phase, in mm.  Must be
            non-zero.
        off_axis_x_um: Off-axis centre x offset for ``FRESNEL_OFF_AXIS``, in um.
        off_axis_y_um: Off-axis centre y offset for ``FRESNEL_OFF_AXIS``, in um.
        wavelength_nm: Wavelength, in nm.  Must be > 0.
        panel_pixel_um: SLM panel pixel pitch, in um.  Must be > 0.
        lens_focal_mm: Focal length of the 2f transform lens, in mm; used only to
            *predict* the lateral shift of ``FRESNEL_OFF_AXIS`` (it does not enter
            the panel phase).
    """

    mode: ZeroOrderOverlay = ZeroOrderOverlay.NONE
    blaze_period_px: float = 64.0
    blaze_direction: str = "vertical"
    fresnel_focal_mm: float = 500.0
    off_axis_x_um: float = 0.0
    off_axis_y_um: float = 0.0
    wavelength_nm: float = 1064.0
    panel_pixel_um: float = 8.0
    lens_focal_mm: float = 125.0

    def __post_init__(self) -> None:
        # Coerce ``mode`` to the enum.  ``str(enum_member)`` is
        # "ZeroOrderOverlay.BLAZE" (not the value) on Python 3.11+, so read the
        # member's ``.value`` when present and only stringify otherwise.  This
        # accepts both the enum member and its string value.
        raw_mode = getattr(self.mode, "value", self.mode)
        try:
            coerced = ZeroOrderOverlay(str(raw_mode))
        except ValueError as exc:
            valid = ", ".join(m.value for m in ZeroOrderOverlay)
            raise ValueError(
                f"mode must be one of {{{valid}}}; got {self.mode!r}"
            ) from exc
        # frozen dataclass: bypass the immutability guard for this normalisation.
        object.__setattr__(self, "mode", coerced)

        if not (self.blaze_period_px > 0):
            raise ValueError(
                f"blaze_period_px must be > 0; got {self.blaze_period_px!r}"
            )
        if self.fresnel_focal_mm == 0:
            raise ValueError(
                "fresnel_focal_mm must be non-zero (it is a focal length)"
            )
        if not (self.panel_pixel_um > 0):
            raise ValueError(
                f"panel_pixel_um must be > 0; got {self.panel_pixel_um!r}"
            )
        if not (self.wavelength_nm > 0):
            raise ValueError(
                f"wavelength_nm must be > 0; got {self.wavelength_nm!r}"
            )
        if self.blaze_direction not in ("vertical", "horizontal"):
            raise ValueError(
                "blaze_direction must be 'vertical' or 'horizontal'; "
                f"got {self.blaze_direction!r}"
            )


def camera_pixel_um_from_bench(params: OverlayParams) -> float:
    """Derive the camera pixel pitch in micrometres from bench constants.

    The bench relation is
        ``TILT_SHIFT_SCALE = f * lambda / (d_slm * p_cam)``
    where ``TILT_SHIFT_SCALE`` is the measured focal scale in camera pixels per
    2*pi ramp per panel pixel, so
        ``p_cam = f * lambda / (d_slm * TILT_SHIFT_SCALE)``.

    This matches the convention in
    :func:`ao_shaping.utils.wavefront.matrix_utils.camera_pixel_um_from_focal_scale`
    (same formula, different parameterisation). The result is in micrometres.

    Args:
        params: Overlay parameters providing wavelength, panel pixel pitch,
            lens focal length and the measured focal scale context.

    Returns:
        Camera pixel pitch in micrometres.

    Raises:
        ValueError: If the derived ``p_cam`` is not positive or finite.
    """
    lam_m = params.wavelength_nm * 1e-9
    d_slm_m = params.panel_pixel_um * 1e-6
    f_m = params.lens_focal_mm * 1e-3
    k = float(TILT_SHIFT_SCALE)

    if not (lam_m > 0.0 and d_slm_m > 0.0 and f_m > 0.0 and k > 0.0):
        raise ValueError(
            "wavelength_nm, panel_pixel_um, lens_focal_mm and TILT_SHIFT_SCALE "
            "must all be positive, got "
            f"({params.wavelength_nm!r}, {params.panel_pixel_um!r}, "
            f"{params.lens_focal_mm!r}, {k!r})"
        )

    p_cam_m = lam_m * f_m / (d_slm_m * k)
    if not (np.isfinite(p_cam_m) and p_cam_m > 0.0):
        raise ValueError(
            f"derived camera pixel pitch is not positive/finite: {p_cam_m!r}"
        )

    return float(p_cam_m * 1e6)


def _build_overlay(
    shape: tuple[int, int],
    params: OverlayParams,
    helper: PatternHelper | None = None,
) -> np.ndarray:
    """Build the raw-radian overlay for the non-``NONE`` modes.

    Uses :class:`PatternHelper` so the blaze ramp and the on-axis quadratic
    match :meth:`PatternHelper.linear_grating` / :meth:`PatternHelper.generate_focus`
    exactly (same integer-pixel coordinate convention, same paraxial form).
    """
    height, width = shape
    # PatternHelper(resolution=(width, height)); it stores height=resolution[1].
    if helper is None:
        helper = _get_helper(width, height)

    wavelength_m = params.wavelength_nm * 1e-9
    pixel_m = params.panel_pixel_um * 1e-6
    f_slm_m = params.fresnel_focal_mm * 1e-3

    if params.mode is ZeroOrderOverlay.BLAZE:
        # Match PatternHelper.linear_grating exactly: direction="vertical"
        # ramps along panel x (self.xx), "horizontal" ramps along panel y
        # (self.yy).  phase_range = 2*pi spans one full wave per period.
        return helper.linear_grating(
            period=params.blaze_period_px,
            phase_range=_TWO_PI,
            direction=params.blaze_direction,
        )

    if params.mode is ZeroOrderOverlay.FRESNEL_ON_AXIS:
        # Paraxial lens phase (pi / (lambda f)) * (R**2 * pixel_size**2), the
        # exact form used by PatternHelper.generate_focus (SI units).  The
        # absolute scale is not physically critical for an *overlay* because the
        # driver wraps mod 2*pi downstream; only the shape/gradient matter.
        return helper.generate_focus(
            focal_length=f_slm_m,
            wavelength=wavelength_m,
            pixel_size=pixel_m,
        )

    if params.mode is ZeroOrderOverlay.FRESNEL_OFF_AXIS:
        xx = helper.xx
        yy = helper.yy
        # Offset converted from um to panel pixels.
        x0_px = params.off_axis_x_um / params.panel_pixel_um
        y0_px = params.off_axis_y_um / params.panel_pixel_um
        # Centred quadratic about (x0, y0).  Expanding (x - x0)**2 + (y - y0)**2
        # gives the on-axis quadratic (x**2 + y**2) PLUS the linear steering term
        #   -2*pi*(x*x0 + y*y0) / (lambda*f) * pixel**2
        # (the written '+' form is the same term with the offset-sign
        # convention folded into x0).  That linear term is the prism that
        # actually steers the spot; the quadratic part only defocuses.  We build
        # it as an explicit on-axis quadratic + linear term so the decomposition
        # is transparent; the difference from the on-axis Fresnel is exactly the
        # linear term (up to an irrelevant constant).
        quad_on_axis = (np.pi / wavelength_m / f_slm_m) * (
            (xx**2 + yy**2) * pixel_m**2
        )
        linear_steer = (
            -2.0
            * np.pi
            * (xx * x0_px + yy * y0_px)
            * pixel_m**2
            / (wavelength_m * f_slm_m)
        )
        return quad_on_axis + linear_steer

    # Unreachable: mode was coerced in __post_init__; kept defensive.
    raise ValueError(f"unhandled overlay mode: {params.mode!r}")


def apply_zero_order_overlay(
    panel_phase: np.ndarray,
    params: OverlayParams,
    helper: PatternHelper | None = None,
) -> np.ndarray:
    """Return ``panel_phase`` plus the optional 0-order overlay, in raw radians.

    The result is always a **new** array of the same shape and (when the input is
    floating) the same float dtype; the input is never mutated.  Nothing is
    wrapped mod 2*pi and nothing is normalised -- the driver performs the single
    radian->grayscale wrap.

    Args:
        panel_phase: 2-D panel phase in raw unwrapped radians.
        params: Overlay selection and geometry.
        helper: Optional cached :class:`PatternHelper` for the panel geometry.
            When provided, it is reused; otherwise a module-level cache keyed by
            ``(width, height)`` is used to avoid rebuilding coordinate grids on
            every call in a closed loop.

    Returns:
        A new array equal to ``panel_phase + overlay`` (raw radians).  For
        ``ZeroOrderOverlay.NONE`` the input array is returned unchanged (same
        object) -- there is no overlay to add.

    Raises:
        TypeError: If ``panel_phase`` is not a numpy array.
        ValueError: If ``panel_phase`` is not 2-D.
    """
    if not isinstance(panel_phase, np.ndarray):
        raise TypeError(
            f"panel_phase must be a numpy.ndarray; got {type(panel_phase).__name__}"
        )
    if panel_phase.ndim != 2:
        raise ValueError(
            f"panel_phase must be 2-D (height, width); got ndim={panel_phase.ndim}"
        )

    if params.mode is ZeroOrderOverlay.NONE:
        # Documented identity: no overlay, so return the caller's array as-is.
        return panel_phase

    overlay = _build_overlay(panel_phase.shape, params, helper=helper)

    target_dtype = (
        panel_phase.dtype
        if np.issubdtype(panel_phase.dtype, np.floating)
        else np.float64
    )
    out = panel_phase.astype(target_dtype, copy=True)
    out = out + overlay.astype(target_dtype, copy=False)
    assert out.shape == panel_phase.shape, "overlay must preserve the panel shape"
    return out


def predicted_shift_px(params: OverlayParams) -> tuple[float, float]:
    """Predict the LATERAL spot displacement ``(dx, dy)`` in camera pixels.

    * ``NONE`` -> ``(0.0, 0.0)``.
    * ``BLAZE`` -> magnitude ``TILT_SHIFT_SCALE / blaze_period_px`` (the measured
      bench focal scale: a 2*pi ramp over ``P`` panel px shifts by
      ``TILT_SHIFT_SCALE / P`` camera px).  The **panel <-> camera axes are
      swapped 90 deg**, so a ramp along panel x (``direction="vertical"``)
      displaces along **camera y**, and a ramp along panel y
      (``direction="horizontal"``) displaces along **camera x**.  This module
      applies panel-x -> camera-y; only the magnitude is calibrated, the sign is
      a fixed convention.
    * ``FRESNEL_ON_AXIS`` -> ``(0.0, 0.0)``.  An on-axis Fresnel lens placed at
      the front focal plane changes only the axial focus; it produces **no**
      lateral displacement.  This is the physically correct answer.
    * ``FRESNEL_OFF_AXIS`` -> the standard ``f2 * x0 / f_slm`` relation: the
      linear term of the centred quadratic tilts the wavefront by roughly
      ``x0 / f_slm`` (small angle), which the 2f lens maps to a focal-plane
      displacement ``f2 * (x0 / f_slm)``.  The camera pixel pitch is derived
      from the measured bench focal scale via
      ``p_cam = f * lambda / (d_slm * TILT_SHIFT_SCALE)`` (with ``f`` being the
      2f lens focal length), so the result is converted to **camera pixels** as
      ``(f2 * x0 / f_slm) / p_cam``.  Its **sign** is what callers should rely on.

    Returns:
        ``(dx, dy)`` in camera pixels.
    """
    if params.mode is ZeroOrderOverlay.NONE:
        return (0.0, 0.0)

    if params.mode is ZeroOrderOverlay.BLAZE:
        magnitude = TILT_SHIFT_SCALE / params.blaze_period_px
        # 90 deg panel <-> camera swap: panel-x ramp -> camera-y shift.
        if params.blaze_direction == "horizontal":
            # ramp along panel y -> camera x
            return (magnitude, 0.0)
        # ramp along panel x -> camera y
        return (0.0, magnitude)

    if params.mode is ZeroOrderOverlay.FRESNEL_ON_AXIS:
        return (0.0, 0.0)

    if params.mode is ZeroOrderOverlay.FRESNEL_OFF_AXIS:
        # Derive camera pixel pitch from bench constants:
        # p_cam = f * lambda / (d_slm * TILT_SHIFT_SCALE) (in metres)
        # The lateral displacement in camera pixels is f2 * (x0/f_slm) / p_cam
        # where x0 is in metres, f2 in metres, p_cam in metres.
        p_cam_um = camera_pixel_um_from_bench(params)
        p_cam_m = p_cam_um * 1e-6
        f2_m = params.lens_focal_mm * 1e-3
        f_slm_m = params.fresnel_focal_mm * 1e-3
        x0_m = params.off_axis_x_um * 1e-6
        y0_m = params.off_axis_y_um * 1e-6
        dx_px = (f2_m * x0_m / f_slm_m) / p_cam_m if p_cam_m > 0 else 0.0
        dy_px = (f2_m * y0_m / f_slm_m) / p_cam_m if p_cam_m > 0 else 0.0
        return (float(dx_px), float(dy_px))

    raise ValueError(f"unhandled overlay mode: {params.mode!r}")


def grating_first_order_efficiency(period_px: float) -> float:
    """First-order diffraction efficiency of a discrete blaze grating.

    A blaze grating quantised to ``period_px`` panel pixels per period puts
    ``sin(pi/N)/(pi/N)`` of the energy in the wanted first order, i.e.

        efficiency = np.sinc(1/N)**2

    where ``N = period_px`` and ``np.sinc(x) == sin(pi*x)/(pi*x)``, so
    ``np.sinc(1/N)**2 == sinc^2(pi/N)`` exactly.  Reference values:
    ``N=2 -> 0.405``, ``N=4 -> 0.811``, ``N=8 -> 0.950``, ``N=16 -> 0.987``.

    Nyquist floor: ``period_px`` must be **>= 2** panel pixels (the smallest
    resolvable period); below that the blaze cannot be resolved at all.  The
    caller is responsible for staying above it.

    This is a **GUI-only advisory** -- it is not part of the phase contract and
    does not affect the emitted radians.

    Args:
        period_px: Grating period in panel pixels.

    Returns:
        First-order efficiency in the range ``(0, 1]``.
    """
    return float(np.sinc(1.0 / period_px) ** 2)
