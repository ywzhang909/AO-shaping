"""Opt-in reference forward model for SLM -> thin lens -> camera propagation.

The production ``FrozenDefocusForward`` path deliberately keeps its calibrated
FFT kernel. This module adds a separate, explicitly sampled model for diagnosing
what that baseline cannot represent:

``U_CCD = P(z2) { aperture * lens * aberration * P(z1)[A exp(i phi_SLM)] }``

``z1`` is the signed SLM-to-lens distance offset from ``f`` and ``z2`` is the
signed lens-to-camera distance offset. Propagation uses the separable Collins
integral already shared by the fitter, so all coordinates are metric and
``phi_SLM``, ``z1`` and ``z2`` retain autograd. A finite lens aperture is
load-bearing: without it, an ideal thin lens at the back focal plane makes the
intensity independent of the SLM axial offset. This model is a reference for
fitting and Jacobian checks, not a drop-in replacement for the 2048-point FFT
production path; callers must establish spatial/frequency sampling convergence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ao_shaping.algorithm.signal_processing.defocus_forward_core import (
    WAVELENGTH_M,
    collins_field,
)


def as_field_tensor(torch: Any, value: Any, reference: Any):
    """Coerce an optional scalar/tensor to ``reference`` device and real dtype."""
    if torch.is_tensor(value):
        return value.to(device=reference.device)
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def thin_lens_phase(torch: Any, x_m: Any, y_m: Any, focal_length_m: Any,
                    wavelength_m: float = WAVELENGTH_M):
    """Paraxial thin-lens phase ``exp(-i*pi*r^2/(lambda*f))`` on a metric grid."""
    f = torch.as_tensor(focal_length_m, device=x_m.device)
    if not bool(torch.isfinite(f)) or float(torch.as_tensor(f).detach()) == 0.:
        raise ValueError("finite nonzero focal length required")
    r2 = y_m[:, None] ** 2 + x_m[None, :] ** 2
    return torch.exp(-1j * np.pi * r2 / (float(wavelength_m) * f))


def free_space_field(torch: Any, field: Any, input_x_m: Any, input_y_m: Any,
                     output_x_m: Any, output_y_m: Any, distance_m: Any,
                     wavelength_m: float = WAVELENGTH_M):
    """Propagate between independent uniform metric planes with ``P(z)``.

    ``distance_m`` may be a tensor. An exactly zero distance is treated as the
    identity, which also gives autograd a defined identity branch.
    """
    distance = torch.as_tensor(distance_m, device=field.device)
    if not bool(torch.isfinite(distance)):
        raise ValueError("finite propagation distance required")
    if bool(torch.all(distance == 0)):
        if len(input_x_m) != len(output_x_m) or len(input_y_m) != len(output_y_m):
            raise ValueError("zero propagation requires matching output axes")
        if not (torch.allclose(input_x_m, output_x_m) and torch.allclose(input_y_m, output_y_m)):
            raise ValueError("zero propagation requires identical output axes")
        return field
    return collins_field(torch, field, input_x_m, input_y_m, output_x_m, output_y_m,
                         a=1., b=distance, d=1., wavelength_m=wavelength_m)


def full_chain_field(
    torch: Any,
    phase: Any,
    input_x_m: Any,
    input_y_m: Any,
    *,
    wavelength_m: float = WAVELENGTH_M,
    focal_length_m: Any,
    slm_offset_m: Any = 0.,
    camera_offset_m: Any = 0.,
    amplitude: Any = 1.,
    lens_aperture: Any = 1.,
    lens_aberration_rad: Any = 0.,
    lens_x_m: Any = None,
    lens_y_m: Any = None,
    camera_x_m: Any = None,
    camera_y_m: Any = None,
):
    """Return the complex field at the camera plane of the explicit chain.

    ``input_x_m``/``input_y_m`` describe the SLM field. By default the lens is
    sampled on those same axes; ``camera_x_m``/``camera_y_m`` may be an
    independent output grid. ``lens_aperture``, ``lens_aberration_rad`` and
    ``amplitude`` are broadcastable scalar fields or one field per phase.
    """
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("positive finite wavelength required")
    if phase.ndim == 2:
        phase = phase[None]
    if phase.ndim != 3:
        raise ValueError("phase must be (Ny, Nx) or (batch, Ny, Nx)")
    if tuple(phase.shape[-2:]) != (len(input_y_m), len(input_x_m)):
        raise ValueError("phase shape must match input axes")

    field = as_field_tensor(torch, amplitude, phase.real)
    if field.ndim == 2:
        field = field[None]
    field = field.to(dtype=phase.dtype) * torch.exp(1j * phase)
    lens_x = input_x_m if lens_x_m is None else lens_x_m
    lens_y = input_y_m if lens_y_m is None else lens_y_m
    field = free_space_field(
        torch, field, input_x_m, input_y_m, lens_x, lens_y,
        torch.as_tensor(focal_length_m, device=field.device) + slm_offset_m,
        wavelength_m=wavelength_m,
    )
    aperture = as_field_tensor(torch, lens_aperture, field.real)
    aberration = as_field_tensor(torch, lens_aberration_rad, field.real)
    field = field * aperture * thin_lens_phase(
        torch, lens_x, lens_y, focal_length_m, wavelength_m
    ) * torch.exp(1j * aberration)
    out_x = lens_x if camera_x_m is None else camera_x_m
    out_y = lens_y if camera_y_m is None else camera_y_m
    return free_space_field(
        torch, field, lens_x, lens_y, out_x, out_y,
        torch.as_tensor(focal_length_m, device=field.device) + camera_offset_m,
        wavelength_m=wavelength_m,
    )


def sample_field_intensity(torch: Any, intensity: Any, output_x_m: Any, output_y_m: Any,
                           points_xy_m: Any):
    """Bilinearly sample ``intensity`` at arbitrary metric camera points.

    ``points_xy_m`` is ``(H, W, 2)`` with components ``(x, y)``. The output grid
    must be uniform and increasing.
    """
    if tuple(intensity.shape[-2:]) != (len(output_y_m), len(output_x_m)):
        raise ValueError("intensity shape must match output axes")
    if points_xy_m.ndim != 3 or points_xy_m.shape[-1] != 2:
        raise ValueError("points_xy_m must be (H, W, 2)")
    dx = output_x_m[1] - output_x_m[0]
    dy = output_y_m[1] - output_y_m[0]
    ix = (points_xy_m[..., 0] - output_x_m[0]) / dx
    iy = (points_xy_m[..., 1] - output_y_m[0]) / dy
    grid = torch.stack((2 * ix / (len(output_x_m) - 1) - 1,
                        2 * iy / (len(output_y_m) - 1) - 1), dim=-1)
    sampled = torch.nn.functional.grid_sample(
        intensity[:, None], grid[None].expand(len(intensity), -1, -1, -1),
        align_corners=True, padding_mode="zeros",
    )[:, 0]
    return sampled


def pixel_integrated_intensity(
    torch: Any,
    intensity: Any,
    output_x_m: Any,
    output_y_m: Any,
    camera_xy_m: Any,
    *,
    pixel_pitch_xy_m: tuple[float, float],
    samples_per_axis: int = 1,
):
    """Average bilinear samples over sub-pixel positions of each camera pixel."""
    if samples_per_axis < 1:
        raise ValueError("samples_per_axis must be positive")
    pitch_x, pitch_y = pixel_pitch_xy_m
    if not np.isfinite([pitch_x, pitch_y]).all() or pitch_x <= 0 or pitch_y <= 0:
        raise ValueError("positive finite camera pixel pitch required")
    points = torch.as_tensor(camera_xy_m, dtype=output_x_m.dtype, device=output_x_m.device)
    sampled = None
    for iy in range(samples_per_axis):
        for ix in range(samples_per_axis):
            offset = torch.tensor(
                ((ix + .5) / samples_per_axis - .5) * pitch_x,
                dtype=points.dtype, device=points.device,
            ), torch.tensor(
                ((iy + .5) / samples_per_axis - .5) * pitch_y,
                dtype=points.dtype, device=points.device,
            )
            value = sample_field_intensity(
                torch, intensity, output_x_m, output_y_m,
                points + torch.stack(offset, -1),
            )
            sampled = value if sampled is None else sampled + value
    return sampled / samples_per_axis**2


def full_chain_intensity(
    torch: Any,
    phase: Any,
    input_x_m: Any,
    input_y_m: Any,
    **kwargs: Any,
):
    """Return camera intensity, optionally sampled at metric pixel centres.

    Pass ``camera_xy_m`` and ``pixel_pitch_xy_m`` for measured camera sampling.
    Otherwise the intensity on the output field grid is returned.
    """
    camera_xy_m = kwargs.pop("camera_xy_m", None)
    pixel_pitch_xy_m = kwargs.pop("pixel_pitch_xy_m", None)
    samples_per_axis = int(kwargs.pop("samples_per_axis", 1))
    camera_x_m = kwargs.get("camera_x_m")
    camera_y_m = kwargs.get("camera_y_m")
    field = full_chain_field(torch, phase, input_x_m, input_y_m, **kwargs)
    intensity = field.abs() ** 2
    if camera_xy_m is None:
        return intensity
    if camera_x_m is None or camera_y_m is None:
        raise ValueError("camera sampling requires camera_x_m and camera_y_m")
    if pixel_pitch_xy_m is None:
        return sample_field_intensity(torch, intensity, camera_x_m, camera_y_m,
                                      torch.as_tensor(camera_xy_m, device=field.device))
    return pixel_integrated_intensity(
        torch, intensity, camera_x_m, camera_y_m, camera_xy_m,
        pixel_pitch_xy_m=pixel_pitch_xy_m, samples_per_axis=samples_per_axis,
    )
