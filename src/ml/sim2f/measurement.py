"""Measured-camera transfer: the stage between the optics and the numbers.

Why this module exists
======================
:class:`~ml.sim2f.calibrate.CalibratedBench` already fits the entrance beam, the
pupil aberration and a camera zoom/shift.  Those describe *the optical path and
the mapping onto the sensor*, but a measured frame also carries effects that are
neither: the sensor's own point-spread function (charge diffusion, the lenslet /
cover-glass stack) and a uniform pedestal (stray light, dark offset).  A fit that
is free to spend Zernike coefficients will happily absorb both -- defocus and
spherical explain a wider spot, and a low-order term can fake a pedestal's
effect on the score -- so the reported aberration describes a bench that does
not exist.

This module isolates those two terms as a **post-affine** transfer applied to the
predicted camera frame:

.. code-block:: text

    Sim2f.forward  ->  camera affine  ->  detector MTF  ->  + uniform background

Nothing here propagates light: the FFT stays in :mod:`ml.sim2f.optics` (package
rule 1).  The detector MTF is a separable Gaussian convolution applied *after*
the camera affine -- it is not an optical propagation blur.  The background is a
constant, and both are the identity at their nominal values.

What is fitted, and what is not
===============================
* ``psf_sigma_px`` -- isotropic detector-MTF width in CCD pixels, ``[0, 4]``.
  It is fitted **after** the geometry fit, with beam widths and camera zoom
  frozen: all three change the measured spot size and are degenerate jointly.
* ``background_fraction`` -- the integrated background as a fraction of the
  frame's own total, ``[0, 0.5]``.  Per pixel it is
  ``background_fraction * frame.sum() / (ccd_px * ccd_px)``.  It is structural
  and **not fitted by default**: prepared/median-subtracted metrics cannot see a
  uniform offset, while on raw frames the closed-form gain absorbs it, so gain
  and background are degenerate.  ``fit_measurement_background`` is the explicit
  experimental escape hatch.
* ``zero_order_fraction`` -- an optional incoherent, fixed zero-order template
  term.  It is **structural and off by default**: applying it needs a measured
  template (``MeasurementTransfer.apply(..., template=...)``), which a first
  pass does not have, and fitting its weight without one would trade against
  the background with no way to tell them apart.  The coherent counterpart is
  :attr:`ml.sim2f.config.OpticsErrors.zero_order_eta`, a pupil-domain amplitude
  that interferes with the modulated field; this term is incoherent and lives on
  the sensor.  They are different parameters and must not both be free in one
  fit, since either could explain the same unmodulated light.

Identifiability
===============
:func:`ml.sim2f.losses.prepare_frame` subtracts the frame median, so a uniform
offset is invisible to the prepared-frame L1/NCC.  :func:`pedestal_mismatch`
is a raw-domain term available to the opt-in background experiment, not a claim
that the pedestal is identifiable in the default fit: the closed-form gain is a
competing explanation for the same raw-frame offset.  The default measurement
pass fits detector-MTF sigma alone, after the geometry knobs are frozen.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = [
    "INIT_BACKGROUND_FRACTION",
    "INIT_PSF_SIGMA_PX",
    "MAX_BACKGROUND_FRACTION",
    "MAX_PSF_SIGMA_PX",
    "PEDESTAL_WEIGHT",
    "MeasurementTransfer",
    "apply_transfer",
    "camera_affine",
    "gaussian_blur",
    "pedestal_mismatch",
]

#: Guard for the normalisation denominators below.
_EPS = 1e-12

#: Fit bounds.  A camera PSF wider than four CCD pixels would be resolved by the
#: detector and is a finding, not something to hide in a fit; a frame that is
#: more than half background has no usable signal left for a shape score.
MAX_PSF_SIGMA_PX = 4.0
MAX_BACKGROUND_FRACTION = 0.5

#: Where the fit starts.  Sigma may not start at zero: the Gaussian kernel's
#: derivative with respect to sigma vanishes there, so a zero start has no
#: gradient to leave with.  Half a pixel is "near identity" and still inside the
#: measurable range of a real sensor.
INIT_PSF_SIGMA_PX = 0.5
INIT_BACKGROUND_FRACTION = 0.0

#: Weight of the raw-domain pedestal term in the calibration objective.  Small on
#: purpose: it exists to identify the background, not to outvote the shape score.
PEDESTAL_WEIGHT = 0.1

#: A sigma below this is numerically indistinguishable from the identity, but the
#: kernel must still be buildable; it is not a physical lower bound.
_MIN_SIGMA_PX = 1e-3

#: Kernel half-width at the widest allowed sigma (three standard deviations).
_MAX_RADIUS = int(math.ceil(3.0 * MAX_PSF_SIGMA_PX))


def _as_tensor(value: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Lift a Python float or a tensor onto ``like``'s dtype and device.

    Args:
        value: A scalar or an existing tensor (gradients preserved).
        like: The frame whose dtype/device the result must match.

    Returns:
        A tensor holding ``value``.
    """
    if torch.is_tensor(value):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(float(value), dtype=like.dtype, device=like.device)


def camera_affine(
    frame: torch.Tensor,
    shift_x: torch.Tensor,
    shift_y: torch.Tensor,
    log_zoom: torch.Tensor,
) -> torch.Tensor:
    """Apply the camera-domain zoom and shift to a batch of predicted frames.

    The affine is expressed in ``grid_sample``'s own normalised coordinates
    (``align_corners=False``), where the frame centre is exactly ``0``: an output
    pixel at ``u`` samples the input at ``u / s``, shifted.  That is a pure
    similarity transform about the optical axis -- no shear, no rotation -- which
    is what the bench's camera-vs-model mapping can plausibly be off by.

    Args:
        frame: ``(n, 1, ccd_px, ccd_px)`` predicted intensity.
        shift_x: How far the predicted content moves along x, camera pixels.
        shift_y: How far the predicted content moves along y, camera pixels.
        log_zoom: Log of the magnification.

    Returns:
        The warped frames, same shape.
    """
    scale = torch.exp(log_zoom)
    batch = int(frame.shape[0])
    height = int(frame.shape[-2])
    width = int(frame.shape[-1])
    theta = torch.zeros(
        (batch, 2, 3), dtype=frame.dtype, device=frame.device, requires_grad=False
    )
    inv = 1.0 / scale
    theta[:, 0, 0] = inv
    theta[:, 1, 1] = inv
    theta[:, 0, 2] = -2.0 * shift_x / (scale * float(width))
    theta[:, 1, 2] = -2.0 * shift_y / (scale * float(height))
    grid = F.affine_grid(theta, frame.shape, align_corners=False)
    return F.grid_sample(frame, grid, mode="bilinear", align_corners=False, padding_mode="zeros")


def gaussian_blur(frame: torch.Tensor, sigma_px: float | torch.Tensor) -> torch.Tensor:
    """Separable isotropic detector-MTF blur in CCD pixels, by convolution.

    No FFT: :mod:`ml.sim2f.optics` owns the only propagator in the package, and a
    camera PSF is a measurement effect, not a propagation step.  The kernel is
    built from the current sigma (so autograd differentiates through it), applied
    as a row and then a column convolution, and the result is renormalised to the
    input's total flux -- a PSF redistributes light, it does not create it.  At
    ``sigma == 0`` the input is returned untouched, bit for bit.

    Args:
        frame: ``(n, 1, h, w)`` intensity.
        sigma_px: Gaussian width in pixels; scalar float or differentiable
            0-dim tensor.

    Returns:
        The blurred frames, same shape.

    Raises:
        ValueError: If ``sigma_px`` is negative.
    """
    sigma = _as_tensor(sigma_px, frame)
    if float(sigma.detach()) < 0.0:
        raise ValueError(f"sigma_px must be >= 0, got {float(sigma.detach())}")
    if not sigma.requires_grad and float(sigma) == 0.0:
        return frame
    sigma = sigma.clamp_min(_MIN_SIGMA_PX)
    radius = max(1, min(_MAX_RADIUS, int(math.ceil(3.0 * float(sigma.detach())))))
    offsets = torch.arange(-radius, radius + 1, dtype=frame.dtype, device=frame.device)
    weights = torch.exp(-0.5 * (offsets / sigma) ** 2)
    weights = weights / weights.sum()
    rows = F.conv2d(
        F.pad(frame, (radius, radius, 0, 0), mode="replicate"),
        weights.view(1, 1, 1, -1),
    )
    blurred = F.conv2d(
        F.pad(rows, (0, 0, radius, radius), mode="replicate"),
        weights.view(1, 1, -1, 1),
    )
    flux = frame.sum(dim=(-2, -1), keepdim=True)
    return blurred * flux / blurred.sum(dim=(-2, -1), keepdim=True).clamp_min(_EPS)


def apply_transfer(
    frame: torch.Tensor,
    psf_sigma_px: float | torch.Tensor,
    background_fraction: float | torch.Tensor,
) -> torch.Tensor:
    """Apply the fitted measurement transfer to predicted camera frames.

    Args:
        frame: ``(n, 1, ccd_px, ccd_px)`` predicted intensity.
        psf_sigma_px: Gaussian PSF width, CCD pixels (0 is the identity).
        background_fraction: Integrated background as a fraction of the frame's
            own total; the per-pixel offset is
            ``fraction * frame.sum() / (ccd_px * ccd_px)``.  A frame that sums to
            one therefore gets ``fraction / (ccd_px * ccd_px)`` per pixel, and a
            frame at any other exposure scale gets the same *relative* pedestal
            -- the number means the same physical thing either way.

    Returns:
        The transferred frames, same shape.
    """
    height = int(frame.shape[-2])
    width = int(frame.shape[-1])
    blurred = gaussian_blur(frame, psf_sigma_px)
    total = frame.sum(dim=(-2, -1), keepdim=True)
    return blurred + _as_tensor(background_fraction, frame) * total / float(height * width)


def pedestal_mismatch(
    prediction: torch.Tensor, measured: torch.Tensor, gain: torch.Tensor
) -> torch.Tensor:
    """Raw-domain pedestal mismatch, in units of the measured frame's spread.

    :func:`~ml.sim2f.losses.prepare_frame` subtracts the median before scoring,
    which makes a uniform offset invisible to the shape objective by
    construction.  This term is the counterpart that keeps the background
    identifiable: the median is a location statistic, so
    ``median(measured) - gain * median(prediction)`` isolates the offset that
    ``gain`` (an exposure scale) cannot explain.

    The measured median also contains the sensor's dark offset.  The fitted
    fraction is therefore the *combined* uniform pedestal in model intensity
    units, which is exactly the quantity a measurement transfer has to reproduce;
    it is not a claim about the SLM's zero order.

    Args:
        prediction: ``(n, 1, h, w)`` model frames, transfer already applied.
        measured: ``(n, 1, h, w)`` measured counts.
        gain: ``(n,)`` exposure scale from the prepared-frame projection.

    Returns:
        ``(n,)`` per-frame mismatch; differentiable in ``prediction``.
    """
    predicted_median = _median(prediction)
    measured_median = _median(measured)
    spread = measured.flatten(1).std(dim=1, unbiased=False).clamp_min(_EPS)
    return ((measured_median - gain * predicted_median).abs() / spread)


def _median(frame: torch.Tensor) -> torch.Tensor:
    """Per-frame median via ``kthvalue`` (deterministic on CUDA, unlike median)."""
    flat = frame.flatten(1)
    return flat.kthvalue((flat.shape[1] - 1) // 2 + 1, dim=1).values


@dataclass(frozen=True)
class MeasurementTransfer:
    """The fitted, camera-domain part of the bench state.

    Attributes:
        psf_sigma_px: Isotropic Gaussian PSF width in CCD pixels; 0 is identity.
        background_fraction: Integrated uniform background as a fraction of the
            frame's own total; 0 is identity.
        zero_order_fraction: Weight of an incoherent fixed zero-order template.
            0 (the default) disables the term; it is never fitted in this pass.
    """

    psf_sigma_px: float = 0.0
    background_fraction: float = 0.0
    zero_order_fraction: float = 0.0

    def __post_init__(self) -> None:
        """Reject values a transfer cannot have.

        Raises:
            ValueError: If a field is negative or not finite.
        """
        for name in ("psf_sigma_px", "background_fraction", "zero_order_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value}")

    def is_identity(self) -> bool:
        """Whether the transfer leaves a frame exactly unchanged."""
        return (
            float(self.psf_sigma_px) == 0.0
            and float(self.background_fraction) == 0.0
            and float(self.zero_order_fraction) == 0.0
        )

    def to_dict(self) -> dict[str, float]:
        """Round-trippable field dump for ``calibration.json``."""
        return {
            "psf_sigma_px": float(self.psf_sigma_px),
            "background_fraction": float(self.background_fraction),
            "zero_order_fraction": float(self.zero_order_fraction),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> MeasurementTransfer:
        """Read a transfer, treating a missing block as the identity.

        Calibrations written before this stage existed have no ``measurement``
        key.  They must load as the identity, not as a failure, so that old
        artifacts and their reported numbers stay reproducible.

        Args:
            payload: The ``measurement`` object from ``calibration.json``.

        Returns:
            The transfer; the identity when ``payload`` is ``None`` or empty.
        """
        if not payload:
            return cls()
        return cls(
            psf_sigma_px=float(payload.get("psf_sigma_px", 0.0)),
            background_fraction=float(payload.get("background_fraction", 0.0)),
            zero_order_fraction=float(payload.get("zero_order_fraction", 0.0)),
        )

    def apply(
        self,
        frame: torch.Tensor,
        *,
        template: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the transfer to predicted frames.

        Args:
            frame: ``(n, 1, ccd_px, ccd_px)`` predicted intensity, already
                camera-affine warped.
            template: Optional fixed zero-order template on the camera grid.
                Required for the zero-order term; without it the term is
                silently inert, which is what "off by default" means.

        Returns:
            The transferred frames, same shape.
        """
        out = apply_transfer(frame, self.psf_sigma_px, self.background_fraction)
        if template is not None and float(self.zero_order_fraction) > 0.0:
            out = out + float(self.zero_order_fraction) * template
        return out
