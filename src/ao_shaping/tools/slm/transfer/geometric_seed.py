"""Non-GS square seed from separable irradiance transport and fitted optics."""
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


def main():
    """Export the same fitted optical parameters at a finer control resolution."""
    import argparse
    from dataclasses import replace
    import json
    from pathlib import Path

    from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import load_frozen_parameters

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=640)
    parser.add_argument("--fft", type=int, default=2048)
    parser.add_argument("--geometry-dataset", type=Path, help="Replace ramp mapping using fresh signed-ramp measurements")
    args = parser.parse_args()
    if args.grid <= 0 or 1920 % args.grid or 1200 % (1920 // args.grid) or args.fft < args.grid:
        parser.error("grid must tile panel exactly, FFT must cover grid")
    original = load_frozen_parameters(args.checkpoint).effective_parameters()
    params = replace(original, grid_size=args.grid, fft_size=args.fft)
    if args.geometry_dataset:
        from .fit import geometry
        matrix, _ = geometry(json.loads((args.geometry_dataset / "manifest.json").read_text()))
        params = replace(params, ramp_matrix=tuple(tuple(v) for v in matrix))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "frozen_parameters.json").write_text(json.dumps(params.to_dict(), indent=2), encoding="utf-8")
    (args.out / "provenance.json").write_text(json.dumps(dict(source=str(args.checkpoint.resolve()),
        source_grid=original.grid_size, grid=args.grid, fft=args.fft,
        geometry_dataset=str(args.geometry_dataset) if args.geometry_dataset else None,
        status="optical parameters unchanged; control resolution changed, requires CCD validation"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
