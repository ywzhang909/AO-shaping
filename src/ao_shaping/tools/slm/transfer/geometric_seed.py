"""Non-GS seeds from separable irradiance transport and fitted optics.

Two constructions live here, both of which avoid Gerchberg-Saxton:

``square_transport_phase``
    Exact for a separable target (square): the measured illumination's two
    marginals are mapped onto uniform intervals, and the resulting frequency
    profile is integrated into a raw-radian phase. A separable phase cannot
    produce a non-separable far field, so this construction is square-only by
    mathematics, not by convenience.

``target_backprojection_phase``
    Shape-general: the target is back-projected through the fitted optics
    (one inverse FFT of the target amplitude) and its phase is used directly.
    No iteration, no measurement feedback. It is the non-GS construction
    available for triangle / pentagon / star / circle; its start is blurrier
    than the transport seed's, so it is compared against the GS seed before use.

Both return **raw radians**, never wrapped or min-max normalized.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid


def square_transport_phase(forward, centre, side):
    """Map illumination marginals to uniform intervals, then subtract static phase.

    The square follows the two measured ramp axes; any small camera rotation
    is subsequently corrected by differentiable optimization on a CCD mask.
    Raw radians are returned, never wrapped or min-max normalized.
    """
    if not np.isfinite(side) or side <= 0:
        raise ValueError("side must be finite and positive")
    p = forward.params
    amplitude = forward.amplitude.detach().cpu().numpy()
    matrix = np.asarray(p.ramp_matrix)
    phase = np.zeros_like(amplitude, dtype=np.float64)
    nyquist = .5 / forward.pitch
    qcentre = np.linalg.solve(matrix, np.asarray(centre) - p.origin_xy)
    for axis in (0, 1):
        marginal = np.square(amplitude).sum(axis=0 if axis == 0 else 1)
        cdf = (np.cumsum(marginal) - .5 * marginal) / marginal.sum()
        frequency = side / np.linalg.norm(matrix[:, axis]) * (cdf - .5) + qcentre[axis]
        if np.max(np.abs(frequency)) >= nyquist:
            raise ValueError("target exceeds phase control Nyquist bandwidth; increase grid_size")
        profile = 2 * np.pi * cumulative_trapezoid(frequency, dx=forward.pitch, initial=0)
        phase += profile[None, :] if axis == 0 else profile[:, None]
    phase -= p.defocus_diopter * p.diopter_scale * forward.z4.detach().cpu().numpy()
    if forward.residual_basis is not None:
        phase -= np.einsum("i,ijk->jk", forward.residual_values.detach().cpu().numpy(),
                           forward.residual_basis.detach().cpu().numpy())
    return phase.astype(np.float32)


def target_backprojection_phase(forward, centre, side, shape):
    """Back-project the target through the fitted optics; no GS iteration.

    The target is sampled on the model's own FFT grid, turned into a far-field
    amplitude, and inverse-transformed once. The phase of that pupil is the
    seed; the fitted static aberration (defocus + residual Noll modes) is
    subtracted exactly as in the other two seeds, because the model adds it to
    the base phase it is given.

    Returns ``(phase, diagnostics)``. ``diagnostics`` carries the implied
    spatial frequency of the seed in cycles per panel pixel, so a caller can
    record whether the device can represent it (``0.5`` is the limit).
    """
    from ao_shaping.algorithm.signal_processing.multishape_objective import fft_grid_target

    p = forward.params
    n, g = p.fft_size, p.grid_size
    half = (n - g) // 2
    target = fft_grid_target(forward, centre, side, shape)
    far_field = np.sqrt(np.clip(target, 0., None))
    # Inverse of ``focal_field``: fftshift(fft2(ifftshift(field))).
    pupil = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far_field), norm="ortho"))
    phase = np.angle(pupil)[half:half + g, half:half + g].astype(np.float64)
    static = p.defocus_diopter * p.diopter_scale * forward.z4.detach().cpu().numpy()
    if forward.residual_basis is not None:
        static = static + np.einsum("i,ijk->jk", forward.residual_values.detach().cpu().numpy(),
                                    forward.residual_basis.detach().cpu().numpy())
    phase = phase - static
    # Implied spatial frequency: the largest phase step across one control
    # cell, converted to cycles per panel pixel (the device's actual limit).
    panel_per_cell = 1920 / g
    step = np.abs(np.diff(phase, axis=1)).max()
    diagnostics = dict(
        implied_cycles_per_panel_px=float(step / (2 * np.pi) / panel_per_cell),
        nyquist_cycles_per_panel_px=0.5,
        target_amplitude_sum=float(far_field.sum()),
    )
    return phase.astype(np.float32), diagnostics


def export_frozen_parameters(checkpoint, out, *, grid=640, fft=2048, geometry_dataset=None):
    """Export the fitted optical parameters at a finer control resolution.

    The optical parameters are copied unchanged; only the control resolution
    (and, optionally, the measured ramp mapping) differs. A finer control grid
    is a *sampling* change, so the result still needs CCD validation before it
    is used for control.
    """
    from dataclasses import replace
    import json
    from pathlib import Path

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import load_frozen_parameters

    checkpoint = Path(checkpoint)
    out = Path(out)
    if grid <= 0 or 1920 % grid or 1200 % (1920 // grid) or fft < grid:
        raise ValueError("grid must tile panel exactly, FFT must cover grid")
    original = load_frozen_parameters(checkpoint).effective_parameters()
    params = replace(original, grid_size=grid, fft_size=fft)
    if geometry_dataset is not None:
        from .fit import geometry
        matrix, _ = geometry(json.loads((Path(geometry_dataset) / "manifest.json").read_text()))
        params = replace(params, ramp_matrix=tuple(tuple(v) for v in matrix))
    out.mkdir(parents=True, exist_ok=False)
    (out / "frozen_parameters.json").write_text(json.dumps(params.to_dict(), indent=2), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(dict(source=str(checkpoint.resolve()),
        source_grid=original.grid_size, grid=grid, fft=fft,
        geometry_dataset=str(geometry_dataset) if geometry_dataset else None,
        status="optical parameters unchanged; control resolution changed, requires CCD validation"), indent=2), encoding="utf-8")
    return out / "frozen_parameters.json"


def main():
    """Export the same fitted optical parameters at a finer control resolution."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=640)
    parser.add_argument("--fft", type=int, default=2048)
    parser.add_argument("--geometry-dataset", type=Path, help="Replace ramp mapping using fresh signed-ramp measurements")
    args = parser.parse_args()
    export_frozen_parameters(args.checkpoint, args.out, grid=args.grid, fft=args.fft,
                             geometry_dataset=args.geometry_dataset)


if __name__ == "__main__":
    main()
