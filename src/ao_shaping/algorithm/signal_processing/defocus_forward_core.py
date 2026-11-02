"""Shared physical forward core for the defocus model-in-loop chain.

``PhysicalForwardFit`` (fitting) and ``FrozenDefocusForward`` (frozen
inference) implement the same physics twice: the panel geometry, the
Gaussian illumination amplitude, the camera sampling grids and the FFT
irradiance scaling were copy-pasted between them, and the wavelength /
panel constants were hardcoded at each site (issue #199 R2/P5). This
module is the single source for that shared math.

The bench-level physical constants (panel size, pixel pitch, wavelength)
themselves live in the zero-import leaf
:mod:`ao_shaping.utils.slm.bench_constants` and are re-exported here under
the names the two forward classes import, so this module never becomes a
second declaration site for them.

Every helper is written so a class that switches to it keeps its exact
arithmetic: the expressions are the ones the classes used inline, in the
same evaluation order, so the extraction moves no number. That property
is pinned bit-for-bit by
``tests/ao_shaping/algorithm/test_defocus_forward_core.py``.

What deliberately stays per class:

* the Zernike basis construction -- the frozen forward samples Noll modes
  from the full 1920x1920 panel grid (``map_coordinates`` at +7 px), the
  fitter generates them on its own control grid;
* the refit post-processing (blaze broadening, gain, additive
  background), which only the frozen forward applies.

``torch`` is passed in by the caller, so this module imports without the
ML extra -- matching both consumers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ao_shaping.utils.slm.bench_constants import (
    LASER_WAVELENGTH_M,
    SLM_PANEL_H,
    SLM_PANEL_W,
    SLM_PITCH_M,
)

#: SLM panel geometry (height, width) in pixels. Delegated to
#: :mod:`ao_shaping.utils.slm.bench_constants` (R-47 single source) -- these
#: aliases keep the names both forward classes already import.
PANEL_H, PANEL_W = SLM_PANEL_H, SLM_PANEL_W
#: Panel pitch expressed in panel pixels: a ``size`` x ``size`` control grid
#: covers the full 1920 px width, i.e. ``pitch = PANEL_PITCH_PX / size``.
PANEL_PITCH_PX = 1920.0
#: Pupil half-span in panel pixels (the illuminated disk is 2x this).
PANEL_HALF_SPAN_PX = 960.0
#: SLM pixel pitch in metres.
PANEL_PIXEL_PITCH_M = SLM_PITCH_M
#: Laser wavelength in metres (YAG).
WAVELENGTH_M = LASER_WAVELENGTH_M
#: Illumination ceiling in panel pixels (rows at/above it are masked out).
PANEL_ILLUMINATION_CEILING_PX = 1200.0
#: Irradiance reference the scaling below is normalised against.
IRRADIANCE_REFERENCE_PITCH = 15.0
IRRADIANCE_REFERENCE_FFT_SIZE = 512.0


def diopter_scale(wavelength_m: float = WAVELENGTH_M) -> float:
    """Diopter value of one unit of Zernike Noll 4 on this panel.

    ``Z4 = sqrt(3) * (2 r^2 - 1)`` over a pupil of half-span
    ``PANEL_HALF_SPAN_PX`` panel pixels at ``PANEL_PIXEL_PITCH_M``; the
    formula is the one both forward classes used inline.
    """
    return float(
        np.pi
        * (PANEL_HALF_SPAN_PX * PANEL_PIXEL_PITCH_M) ** 2
        / (2 * np.sqrt(3) * wavelength_m)
    )


def axial_plane_correction(defocus_diopter: float, offset_m: float):
    """Compose a calibrated paraxial ABCD map with input free space P(offset).

    For the intensity kernel, d=A/B and the measured ramp scale is proportional
    to B. Right multiplication by P(offset) gives B'=B+A*offset, hence
    d'=d/(1+d*offset), scale'=scale*(1+d*offset). Output chirp has unit modulus.
    Positive offset increases the distance from the input plane to the lens.
    Offset is conditional on the nominal calibration, not identifiable jointly
    with unrestricted d and ramp scale from single-plane intensity alone.
    """
    if not np.isfinite([defocus_diopter, offset_m]).all():
        raise ValueError("finite axial offset and curvature required")
    scale = 1. + float(defocus_diopter)*float(offset_m)
    if scale <= 0:
        raise ValueError("axial correction crosses a singular imaging plane")
    return float(defocus_diopter)/scale, scale


def lens_abcd(focal_length_m: Any, slm_offset_m: Any, camera_offset_m: Any):
    """Elements of P(f+camera_offset) L(f) P(f+slm_offset).

    Offsets are signed distances in metres; positive means farther from the
    lens. Tensor arguments retain autograd. This is a thin, unbounded paraxial
    lens, not a finite lens stop or a nonparaxial aberration model.
    """
    f = focal_length_m
    a = -camera_offset_m / f
    b = f - slm_offset_m * camera_offset_m / f
    c = -1.0 / f
    d = -slm_offset_m / f
    return a, b, c, d


def collins_field(
    torch: Any, field: Any, input_x_m: Any, input_y_m: Any,
    output_x_m: Any, output_y_m: Any, *, a: Any, b: Any, d: Any,
    wavelength_m: float = WAVELENGTH_M,
):
    """Sample the coherent Collins integral on independent Cartesian grids.

    ``field`` is (..., Ny, Nx); input/output coordinates are increasing 1-D
    vectors in metres relative to their optical axes. Input sampling is uniform.
    Quadrature includes dx*dy/(i*lambda*B), but drops the unobservable global
    longitudinal phase. All ABCD scalars may be tensors, including learnable B.
    Direct separable matrix quadrature avoids changing camera pitch with B or
    detaching propagation distances through a cached ASM transfer function.

    The caller must establish input-phase and output-fringe sampling convergence.
    This operator intentionally does not model lens clipping, camera integration,
    electronic bias, SLM response or arbitrary rotated camera coordinates.
    """
    axes = (input_x_m, input_y_m, output_x_m, output_y_m)
    if any(v.ndim != 1 or len(v) < 2 for v in axes):
        raise ValueError("coordinate axes must be 1-D with at least two samples")
    if not field.is_complex() or tuple(field.shape[-2:]) != (len(input_y_m), len(input_x_m)):
        raise ValueError("complex field must match input axes")
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("positive finite wavelength required")
    for axis in axes:
        if not torch.isfinite(axis).all() or not torch.all(axis[1:] > axis[:-1]):
            raise ValueError("coordinate axes must be finite and increasing")
    for axis in (input_x_m, input_y_m):
        if not torch.allclose(axis[1:]-axis[:-1], (axis[1]-axis[0]).expand(len(axis)-1)):
            raise ValueError("input axes must be uniformly sampled")
    scalars = [torch.as_tensor(v, device=field.device) for v in (a, b, d)]
    if any(v.ndim != 0 or not torch.isfinite(v) for v in scalars):
        raise ValueError("finite scalar ABCD elements required")
    if abs(scalars[1].detach().item()) < 1e-12:
        raise ValueError("B=0 imaging planes require an imaging operator")
    r2_in = input_y_m[:, None]**2 + input_x_m[None, :]**2
    r2_out = output_y_m[:, None]**2 + output_x_m[None, :]**2
    factor = np.pi / (wavelength_m * b)
    prepared = field * torch.exp(1j * factor * a * r2_in)
    ky = torch.exp(-2j * factor * output_y_m[:, None] * input_y_m[None, :])
    kx = torch.exp(-2j * factor * output_x_m[:, None] * input_x_m[None, :])
    # Coordinate precision must not silently downcast a differentiable field.
    dtype = torch.promote_types(prepared.dtype, ky.dtype)
    sampled = ky.to(dtype) @ prepared.to(dtype) @ kx.to(dtype).T
    area = (input_x_m[1]-input_x_m[0]) * (input_y_m[1]-input_y_m[0])
    return sampled * torch.exp(1j * factor * d * r2_out) * area / (1j*wavelength_m*b)


def panel_grid(torch: Any, size: int, *, device: Any = None):
    """Panel-pixel coordinates and illumination mask of a control grid.

    Returns ``(panel_x, panel_y, panel_mask)`` where ``panel_mask`` is the
    rows below :data:`PANEL_ILLUMINATION_CEILING_PX` as a float tensor.
    """
    pitch = PANEL_PITCH_PX / size
    y, x = torch.meshgrid(
        torch.arange(size, dtype=torch.float32, device=device),
        torch.arange(size, dtype=torch.float32, device=device),
        indexing="ij",
    )
    panel_x = x * pitch
    panel_y = y * pitch
    return panel_x, panel_y, (panel_y < PANEL_ILLUMINATION_CEILING_PX).float()


def gaussian_amplitude(
    torch: Any, panel_x: Any, panel_y: Any, center: Any, width: Any, mask: Any
) -> Any:
    """Gaussian illumination amplitude on the panel, zero outside ``mask``."""
    return (
        torch.exp(
            -(((panel_x - center[0]) / width[0]) ** 2)
            - ((panel_y - center[1]) / width[1]) ** 2
        )
        * mask
    )


def subpixel_offsets(torch: Any, pixel_samples: int, *, device: Any = None):
    """Sub-pixel sampling offsets as ``(x, y)`` tensors, ``iy``-major.

    The order (and the ``(ix, iy)`` component order) matches both classes'
    historical loops, which fixes the order the per-offset samples are
    averaged in.
    """
    return [
        torch.tensor(
            [(ix + 0.5) / pixel_samples - 0.5, (iy + 0.5) / pixel_samples - 0.5],
            dtype=torch.float32,
            device=device,
        )
        for iy in range(pixel_samples)
        for ix in range(pixel_samples)
    ]


def camera_sample_grids(
    torch: Any,
    xy: Any,
    origin: Any,
    inverse: Any,
    offsets,
    *,
    fft_size: int,
    pitch: float,
):
    """Normalised ``grid_sample`` coordinates for the camera sampling points.

    ``xy`` is the ``(H, W, 2)`` camera grid, ``origin`` the shared optical-axis
    intercept and ``inverse`` the inverse of the panel -> camera ramp matrix.
    """
    grids = []
    for offset in offsets:
        frequencies = (xy + offset - origin) @ inverse.T
        indices = frequencies * (fft_size * pitch) + fft_size // 2
        grids.append(2 * indices / (fft_size - 1) - 1)
    return grids


def sample_intensity(torch: Any, intensity: Any, grids) -> Any:
    """Bilinear-sample ``intensity`` at the camera points, averaged over offsets."""
    return sum(
        torch.nn.functional.grid_sample(
            intensity[:, None],
            grid[None].expand(len(intensity), -1, -1, -1),
            align_corners=True,
            padding_mode="zeros",
        )[:, 0]
        for grid in grids
    ) / len(grids)


def irradiance_scale(pitch: float, fft_size: int) -> float:
    """Scale that preserves physical irradiance units when pitch/FFT change.

    ``(pitch**2 * fft_size / (15**2 * 512))**2`` -- the reference pair
    (15 px pitch, 512 FFT) is the one both classes hardcoded.
    """
    return (
        pitch**2
        * fft_size
        / (IRRADIANCE_REFERENCE_PITCH**2 * IRRADIANCE_REFERENCE_FFT_SIZE)
    ) ** 2


def defocus_phase(
    torch: Any,
    base_phase: Any,
    defocus: Any,
    diopter: Any,
    z4: Any,
    *,
    residual_values: Any = None,
    residual_basis: Any = None,
) -> Any:
    """``base + defocus * diopter * z4`` plus optional residual Noll modes.

    ``defocus`` may be a plain float (frozen forward) or a tensor
    (fitter's trainable parameter); the expression is evaluated exactly as
    both classes did, so the two callers keep their own rounding.
    """
    phase = base_phase + defocus * diopter * z4
    if residual_basis is not None:
        phase = phase + torch.einsum("i,ijk->jk", residual_values, residual_basis)
    return phase
