"""Bench 2f forward physics: calibrated (reduced) propagator + ground-truth twin.

THE SINGLE SOURCE OF TRUTH for the 2f bench propagation used by issue #165.

The calibrated operator (:class:`Bench2FPropagator`) is the reduced physical
forward the model-in-loop closed loop optimises against: illumination x
aperture x Zernike-reduced phase -> centred zero-pad -> ``|fftshift(fft2(
ifftshift(.), norm="ortho"))|^2`` -> detector affine -> peak normalisation.
The padding arithmetic and the shift convention are imported from
:func:`ao_shaping.utils.wavefront.fraunhofer.focal_field` — this module never
re-derives a different centred-FFT rule.

The digital twin (:class:`Bench2FGroundTruth`) is a strict superset carrying
physics the reduced model cannot represent (super-Gaussian illumination, a
fixed out-of-span freeform aberration, detector noise), so a network trained
through the twin and evaluated against the propagator is a non-circular test.

This module is deliberately **not** re-exported from
``utils/wavefront/__init__.py``: that ``__init__`` eagerly imports every
submodule, and ``ao_shaping.utils`` must stay a torch-free leaf package
(torch is an optional dependency group). Callers import the submodule
directly, exactly like :mod:`ao_shaping.utils.wavefront.fraunhofer`::

    from ao_shaping.utils.wavefront.bench2f import Bench2FPropagator

Leaf contract: this module imports stdlib + numpy + torch + the two sibling
wavefront helpers only — never ``algorithm`` / ``drivers`` / ``runners`` /
``optimizer`` / ``ml``, and never ``scipy`` (the fit is a torch linear least
squares).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from ao_shaping.utils.wavefront.fraunhofer import focal_field
from ao_shaping.utils.wavefront.zernike_calc import (
    ZernikeGenerator,
    calc_n_zernike_terms,
)

__all__ = [
    "Bench2FGeometry",
    "Bench2FPropagator",
    "Bench2FGroundTruth",
    "CCDSensor",
    "ZernikeAberrationField",
]

#: JSON keys written by ``BenchGeometry.to_json`` that map onto a *differently
#: named* field of :class:`Bench2FGeometry`.
_JSON_KEY_ALIASES: dict[str, str] = {
    "panel_disc_radius": "panel_disc_radius_px",
}


def _centered_radius(region: int) -> torch.Tensor:
    """Pixel distance to the centre of an even ``region x region`` grid.

    The centre is at index ``region / 2`` (axis values
    ``arange(region) - region / 2``), matching
    :meth:`ZernikeCoefficientOptimizer.native_amplitude`
    (``ao_shaping.algorithm.signal_processing``) — the repo's physical-model
    beam convention, under which the central pixel sits at ``r = 0`` exactly
    and the raw Gaussian peaks at exactly 1.

    Args:
        region: Side length of the square pupil grid, pixels.

    Returns:
        ``(region, region)`` float64 tensor of radial distances in model px.
    """
    axis = torch.arange(region, dtype=torch.float64) - region / 2.0
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.hypot(xx, yy)


def _gaussian_waist_model_px(geometry: Bench2FGeometry) -> float:
    """Gaussian illumination waist in model pixels (physics contract #1).

    Args:
        geometry: Bench geometry carrying the panel/pupil scaling.

    Returns:
        ``beam_waist_panel_px * region / (2 * panel_disc_radius_px)``.
    """
    return (
        geometry.beam_waist_panel_px
        * geometry.region
        / (2.0 * geometry.panel_disc_radius_px)
    )


def _build_pupil_amplitude(
    geometry: Bench2FGeometry, exponent: float
) -> torch.Tensor:
    """Illumination x hard circular aperture, peak-normalised, float32.

    ``exponent == 2.0`` evaluates the exact Gaussian form
    ``exp(-2 r^2 / w0^2)``; any other exponent evaluates the super-Gaussian
    ``exp(-2 (r / w0)^exponent)``.  Both use the same waist and the same hard
    mask of radius ``region / 2`` (no soft edge is allowed).

    Args:
        geometry: Bench geometry.
        exponent: Illumination exponent (2.0 == Gaussian).

    Returns:
        ``(1, 1, region, region)`` amplitude with peak exactly 1.
    """
    region = geometry.region
    r = _centered_radius(region)
    w0 = _gaussian_waist_model_px(geometry)
    if exponent == 2.0:
        amplitude = torch.exp(-2.0 * r.pow(2) / (w0 * w0))
    else:
        amplitude = torch.exp(-2.0 * (r / w0).pow(exponent))
    aperture = (r <= region / 2.0).to(amplitude.dtype)
    amplitude = amplitude * aperture
    peak = amplitude.max()
    if peak > 0:
        amplitude = amplitude / peak
    return amplitude.to(torch.float32).reshape(1, 1, region, region)


def _normalised_intensity(
    field: torch.Tensor, geometry: Bench2FGeometry
) -> torch.Tensor:
    """Far-field intensity chain: pad -> centred FFT -> |F|^2 -> affine -> peak.

    Implements physics contract #2 on top of :func:`focal_field` (the single
    centred-padding rule of the repo).

    Args:
        field: Complex ``(B, 1, region, region)`` pupil field.
        geometry: Bench geometry (zero-pad size, gain, background).

    Returns:
        Real ``(B, 1, far_field_size, far_field_size)`` intensity with each
        sample's maximum normalised to 1 (a zero maximum is left untouched).
    """
    spectrum = focal_field(field, geometry.far_field_size)
    intensity = spectrum.real.pow(2) + spectrum.imag.pow(2)
    intensity = geometry.gain * intensity + geometry.background
    peak = intensity.amax(dim=(-2, -1), keepdim=True)
    return intensity / torch.where(peak > 0, peak, torch.ones_like(peak))


def _check_phase_batch(phase: torch.Tensor, region: int) -> None:
    """Validate a ``(B, 1, region, region)`` pupil-phase batch.

    Args:
        phase: Candidate phase tensor.
        region: Required square side length.

    Raises:
        ValueError: If the shape is wrong.
    """
    if phase.ndim != 4 or phase.shape[1:] != (1, region, region):
        raise ValueError(
            f"phase must have shape (B, 1, {region}, {region}), got {tuple(phase.shape)}"
        )


@dataclass(frozen=True)
class Bench2FGeometry:
    """Geometry + detector affine of the calibrated 2f bench model.

    Attributes:
        region: Pupil model grid side (px); even, >= 16.
        far_field_size: Far-field FFT grid side (centre zero-pad target).
        panel_disc_radius_px: Illuminated disc radius on the 1920x1200 panel.
        beam_waist_panel_px: Gaussian illumination waist, panel px.
        camera_px_per_model_px: Far-field sampling: 1 far-field px == this
            many camera px.
        gain: Detector gain (multiplies intensity).
        background: Additive detector background (same units as intensity).
    """

    region: int = 64
    far_field_size: int = 512
    panel_disc_radius_px: float = 450.0
    beam_waist_panel_px: float = 450.0
    camera_px_per_model_px: float = 8.0
    gain: float = 1.0
    background: float = 0.0

    def __post_init__(self) -> None:
        """Validate the grid/aperture invariants the physics contract needs.

        Raises:
            ValueError: On an odd or tiny ``region``, a far-field grid smaller
                than the pupil, or a non-positive panel disc radius.
        """
        if self.region % 2 or self.region < 16:
            raise ValueError(
                f"region must be an even integer >= 16, got {self.region}"
            )
        if self.far_field_size < self.region:
            raise ValueError(
                f"far_field_size ({self.far_field_size}) must be >= region "
                f"({self.region})"
            )
        if self.panel_disc_radius_px <= 0:
            raise ValueError(
                "panel_disc_radius_px must be positive, got "
                f"{self.panel_disc_radius_px}"
            )

    @classmethod
    def from_bench_geometry_json(
        cls, path: str | os.PathLike[str], **overrides: float | int
    ) -> "Bench2FGeometry":
        """Load geometry from a ``bench_geometry.json`` artefact.

        The file schema is the one produced by
        ``ao_shaping.optimizer.wfless.model_in_loop_shaping.BenchGeometry.to_json``.
        ``panel_disc_radius`` is mapped onto ``panel_disc_radius_px``; unknown
        keys (calibration scores, notes, future fields) are ignored; missing
        keys keep the dataclass defaults; ``gain`` / ``background`` are not in
        the schema and default to 1.0 / 0.0 unless given in ``overrides``.

        Args:
            path: Path to the JSON file.
            **overrides: Field values applied after the JSON mapping.

        Returns:
            The constructed :class:`Bench2FGeometry`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the JSON is malformed or not a mapping.
        """
        text = Path(path).read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed bench_geometry JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"bench_geometry JSON must be a mapping, got {type(payload).__name__}"
            )
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, float | int] = {}
        for key, value in payload.items():
            target = _JSON_KEY_ALIASES.get(str(key), str(key))
            if target in known:
                kwargs[target] = value
        kwargs.update(overrides)
        return cls(**kwargs)


class ZernikeAberrationField:
    """Zernike basis on the pupil grid + linear least-squares fit.

    ``n_terms = calc_n_zernike_terms(n_max)`` (Noll 1976, piston included).
    The basis is zero outside the circular aperture of radius ``region / 2``.
    The basis engine is the canonical :class:`ZernikeGenerator` — this class
    never hand-rolls a Zernike polynomial or an index table.
    """

    def __init__(self, region: int, n_max: int = 10) -> None:
        """Build the (masked) Zernike basis on a ``region x region`` grid.

        Args:
            region: Square pupil grid side, pixels; even, >= 16.
            n_max: Maximum Zernike radial order (Noll orders 1..n_max).

        Raises:
            ValueError: On an invalid ``region`` or ``n_max < 1``.
        """
        if region % 2 or region < 16:
            raise ValueError(
                f"region must be an even integer >= 16, got {region}"
            )
        if n_max < 1:
            raise ValueError(f"n_max must be >= 1, got {n_max}")
        self._region = int(region)
        self._n_max = int(n_max)
        self._n_terms = calc_n_zernike_terms(self._n_max)

        # Canonical Noll basis: one unit-coefficient generate_noll per mode.
        # eval_grid is NaN outside the unit circle, so the hard aperture mask
        # (radius region/2, the generator default) must be applied with where()
        # rather than multiplication.
        generator = ZernikeGenerator(
            (self._region, self._region), n_orders=self._n_max
        )
        inside = generator.mask.astype(bool)
        basis = np.zeros((self._n_terms, self._region, self._region))
        for index in range(self._n_terms):
            coefficients = np.zeros(self._n_terms)
            coefficients[index] = 1.0
            basis[index] = np.where(inside, generator.generate_noll(coefficients), 0.0)
        self._basis = torch.from_numpy(basis).to(torch.float32)
        self._basis_matrix = (
            self._basis.reshape(self._n_terms, -1).T.to(torch.float64)
        )  # (region^2, n_terms) double, for the least-squares fit.
        # 伪逆缓存一次算好：基底是固定的，之后每次拟合只是一次矩阵乘。
        # 这既比每次重解快得多，也绕开了 torch.linalg.lstsq 的批量 gels
        # 驱动在某些形状下的内部 LAPACK 断言失败。
        self._basis_pinv = torch.linalg.pinv(self._basis_matrix)

    @property
    def n_terms(self) -> int:
        """Number of Zernike terms (Noll orders 1..n_max, piston included)."""
        return self._n_terms

    def basis(self) -> torch.Tensor:
        """Zernike basis on the pupil grid.

        Returns:
            ``(n_terms, region, region)`` float32 tensor, zero outside the
            circular aperture.
        """
        return self._basis.clone()

    def fit(self, phase: torch.Tensor) -> torch.Tensor:
        """Linear least-squares fit of Zernike coefficients to a phase field.

        Plain (not non-negative) least squares over the whole grid —
        ``coef = pinv(basis_flat) @ phase_flat``.  The basis is already zero
        outside the aperture, so no mask weighting is needed.  The solve runs in
        float64 and the result is cast back to the input dtype.

        The pseudo-inverse is **cached**: the basis is fixed, so it is computed
        once and the fit becomes a single matmul.  That is both much faster
        than solving per call and avoids ``torch.linalg.lstsq``'s batched
        ``gels`` driver, which trips an internal LAPACK assert on some shapes
        (observed: "false INTERNAL ASSERT FAILED ... Argument 6 has illegal
        value" inside the training loop).

        Args:
            phase: ``(B, 1, region, region)`` phase in radians.

        Returns:
            ``(B, n_terms)`` coefficients in radians, Noll order.

        Raises:
            ValueError: If ``phase`` has the wrong shape.
        """
        _check_phase_batch(phase, self._region)
        rhs = phase.reshape(phase.shape[0], -1).to(torch.float64)
        return (rhs @ self._basis_pinv.T).to(phase.dtype)

    def render(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Synthesise the Zernike phase field of flat coefficients.

        Args:
            coefficients: ``(B, n_terms)`` coefficients in radians, Noll order.

        Returns:
            ``(B, 1, region, region)`` phase field in radians.

        Raises:
            ValueError: If ``coefficients`` has the wrong shape.
        """
        if coefficients.ndim != 2 or coefficients.shape[1] != self._n_terms:
            raise ValueError(
                f"coefficients must have shape (B, {self._n_terms}), "
                f"got {tuple(coefficients.shape)}"
            )
        basis = self._basis.to(coefficients.dtype)
        return torch.einsum("bn,nij->bij", coefficients, basis).unsqueeze(1)


class Bench2FPropagator:
    """Calibrated 2f forward: illumination x aperture x phase -> zero-pad FFT
    -> |F|^2 -> gain*intensity + background -> peak normalisation.

    The phase it propagates is FIRST REDUCED onto the Zernike span
    (:class:`ZernikeAberrationField`), so the operator carries the same
    ~70-parameter capacity as the model-in-loop physical model.  That
    reduction is the calibration: the network then optimises the same
    objective the closed loop does.
    """

    def __init__(self, geometry: Bench2FGeometry, *, n_max: int = 10) -> None:
        """Build the propagator for a bench geometry.

        Args:
            geometry: Calibrated bench geometry.
            n_max: Maximum Zernike radial order of the reduction.
        """
        self._geometry = geometry
        self._field = ZernikeAberrationField(geometry.region, n_max)
        self._amplitude = _build_pupil_amplitude(geometry, exponent=2.0)

    @property
    def geometry(self) -> Bench2FGeometry:
        """The bench geometry this propagator was calibrated to."""
        return self._geometry

    @property
    def field(self) -> ZernikeAberrationField:
        """The Zernike reduction model applied to every propagated phase."""
        return self._field

    def pupil_amplitude(self) -> torch.Tensor:
        """Gaussian illumination x circular aperture (physics contract #1).

        ``exp(-2 r^2 / w0^2)`` with ``w0_model = beam_waist_panel_px * region /
        (2 * panel_disc_radius_px)``, times a HARD circular mask of radius
        ``region / 2`` (no soft edge), peak-normalised.

        Returns:
            ``(1, 1, region, region)`` float32 amplitude with peak 1.
        """
        return self._amplitude.clone()

    def reduce_phase(self, phase: torch.Tensor) -> torch.Tensor:
        """Project a per-pixel phase onto the Zernike span.

        Args:
            phase: ``(B, 1, region, region)`` phase in radians.

        Returns:
            ``(B, 1, region, region)`` Zernike-reduced phase in radians.
        """
        return self._field.render(self._field.fit(phase))

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        """Propagate a pupil phase to the peak-normalised far field.

        The phase is first reduced onto the Zernike span (that is the
        calibration), then::

            U = A * exp(i phi)
            I = |fftshift(fft2(ifftshift(pad(U), norm='ortho'))|^2
            I = gain * I + background;  I /= max(I)

        Differentiable w.r.t. ``phase`` (no ``no_grad`` / ``detach``).

        Args:
            phase: ``(B, 1, region, region)`` pupil phase in radians.

        Returns:
            ``(B, 1, far_field_size, far_field_size)`` peak-normalised
            intensity.

        Raises:
            ValueError: If ``phase`` has the wrong shape.
        """
        _check_phase_batch(phase, self._geometry.region)
        return self._intensity(self.reduce_phase(phase))

    def forward_from_coefficients(
        self, coefficients: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        """Propagate an explicit Zernike field plus a per-pixel phase.

        Total pupil phase = ``field.render(coefficients) + phase``.  The
        Zernike reduction is NOT re-applied — the caller has already split
        the phase into its in-span and out-of-span parts.

        Args:
            coefficients: ``(B, n_terms)`` Zernike coefficients in radians.
            phase: ``(B, 1, region, region)`` per-pixel phase in radians.

        Returns:
            ``(B, 1, far_field_size, far_field_size)`` peak-normalised
            intensity.

        Raises:
            ValueError: If either input has the wrong shape.
        """
        _check_phase_batch(phase, self._geometry.region)
        total = self._field.render(coefficients) + phase
        return self._intensity(total)

    def back_propagate(self, far_field: torch.Tensor) -> torch.Tensor:
        """GS back step: centred inverse FFT from the padded far field.

        Uses the canonical centred inverse
        ``ifft2(ifftshift(far_field))`` scaled by ``(region / S)**2``, i.e. the
        adjoint of the zero-padded centred forward up to the padding
        normalisation; the centre ``region x region`` window of the result is
        returned.  The raw inverse-FFT output still carries the pupil in the
        wrapped (``ifftshift``) frame, so a final ``fftshift`` re-centres it
        before the crop: feeding the complex output
        ``fftshift(fft2(ifftshift(pad(U)), norm="ortho"))`` of the centred
        forward through this method returns ``U * (region/S)**2 / S`` exactly
        (un-normalised ``ifft2`` of an ortho forward carries the 1/S), in the
        same centred frame the forward consumes.

        Like the classic GS algorithm, the aperture mask, the detector gain
        and the background are **NOT inverted** — this is a practical GS
        approximation, not an exact adjoint of :meth:`forward`.

        Args:
            far_field: Complex ``(B, 1, S, S)`` far field, ``S >= region``.

        Returns:
            Complex ``(B, 1, region, region)`` pupil field.

        Raises:
            ValueError: If the input is not complex, not 4-D, or too small.
        """
        if not far_field.is_complex():
            raise ValueError(
                f"far_field must be complex, got dtype {far_field.dtype}"
            )
        side = far_field.shape[-1]
        region = self._geometry.region
        if far_field.ndim != 4 or far_field.shape[1:] != (1, side, side):
            raise ValueError(
                f"far_field must have shape (B, 1, S, S), got {tuple(far_field.shape)}"
            )
        if side < region:
            raise ValueError(
                f"far_field side {side} smaller than region {region}"
            )
        pupil = torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(far_field, dim=(-2, -1)), dim=(-2, -1)
            ),
            dim=(-2, -1),
        ) * (region / side) ** 2
        half = (side - region) // 2
        return pupil[..., half : half + region, half : half + region]

    def _intensity(self, phase: torch.Tensor) -> torch.Tensor:
        """Run the shared intensity chain on an already-reduced phase.

        Args:
            phase: ``(B, 1, region, region)`` phase in radians.

        Returns:
            ``(B, 1, far_field_size, far_field_size)`` peak-normalised
            intensity.
        """
        field = self._amplitude * torch.exp(1j * phase)
        return _normalised_intensity(field, self._geometry)


class CCDSensor:
    """A CCD detector: exposure, saturation, black level, quantisation.

    The far field a camera records is **not** the far field the optics make.
    Three things happen between them, and all three are visible in a frame:

    1. **Exposure** scales the photon count.  Too much and the peak runs into
       the full well, so the core of the spot flattens into a plateau
       (**over-exposure**, clipped at the top).
    2. **Too little** and the signal falls to the black level, so the dim
       speckle is chopped off (**under-exposure**, clipped at the bottom) and
       the read noise is all that is left.
    3. **Quantisation** to the ADC's bit depth.

    ``apply`` takes a peak-normalised intensity (max 1) and returns what the
    camera would record, in the same normalised units.

    The order is the physical one: photons (Poisson) -> read noise (Gaussian)
    -> the ADC's clip -> the ADC's quantisation.
    """

    def __init__(
        self,
        *,
        exposure: float = 1.0,
        full_well: float = 1.0,
        black_level: float = 0.0,
        bit_depth: int = 12,
        shot_noise: float = 0.0,
        read_noise: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> None:
        if exposure <= 0.0:
            raise ValueError(f"exposure must be > 0, got {exposure}")
        if full_well <= 0.0:
            raise ValueError(f"full_well must be > 0, got {full_well}")
        if not 0.0 <= black_level < full_well:
            raise ValueError(
                f"black_level must be in [0, full_well), got {black_level}"
            )
        if not 1 <= bit_depth <= 24:
            raise ValueError(f"bit_depth must be in 1..24, got {bit_depth}")
        if shot_noise < 0.0 or read_noise < 0.0:
            raise ValueError("noise levels must be >= 0")
        self.exposure = float(exposure)
        self.full_well = float(full_well)
        self.black_level = float(black_level)
        self.bit_depth = int(bit_depth)
        self.shot_noise = float(shot_noise)
        self.read_noise = float(read_noise)
        self._generator = generator

    def apply(self, intensity: torch.Tensor) -> torch.Tensor:
        """Map a peak-normalised intensity to a recorded frame.

        Args:
            intensity: Non-negative intensity, peak-normalised (max 1).

        Returns:
            The recorded frame, in the same normalised units.
        """
        scaled = intensity * self.exposure
        if self.shot_noise > 0.0:
            variance = self.shot_noise**2
            scaled = (
                torch.poisson(
                    (scaled / variance).clamp_min(0.0), generator=self._generator
                )
                * variance
            )
        if self.read_noise > 0.0:
            scaled = scaled + self.read_noise * torch.randn(
                scaled.shape,
                dtype=scaled.dtype,
                device=scaled.device,
                generator=self._generator,
            )
        scaled = scaled.clamp(self.black_level, self.full_well)
        levels = float(2**self.bit_depth - 1)
        return torch.round(scaled * levels) / levels

    def saturated_fraction(self, intensity: torch.Tensor) -> torch.Tensor:
        """Fraction of the frame that hit the full well (the over-exposure tell)."""
        return (intensity * self.exposure >= self.full_well).to(intensity.dtype).mean(
            dim=(-2, -1)
        )


class Bench2FGroundTruth:
    """Ground truth of the 2f bench.  Strictly a SUPERSET of Bench2FPropagator.

    Extra physics the reduced calibrated model CANNOT represent — this is what
    makes the comparison non-circular:
      * super-Gaussian illumination (exponent > 2) instead of a pure Gaussian;
      * a fixed per-pixel FREE-FORM aberration field outside the Zernike span;
      * detector noise: shot noise (Poisson, scaled by signal) + Gaussian read
        noise.

    Everything else (aperture, zero-pad size, gain, background, peak
    normalisation) is identical to :class:`Bench2FPropagator`.  With
    ``freeform=None``, ``illumination_exponent=2.0`` and zero noise the twin
    equals the propagator on any phase inside the Zernike span, so every extra
    ingredient is genuinely optional.
    """

    def __init__(
        self,
        geometry: Bench2FGeometry,
        *,
        freeform: torch.Tensor | None = None,
        illumination_exponent: float = 2.6,
        shot_noise: float = 0.0,
        read_noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        """Build the digital twin.

        Args:
            geometry: Bench geometry (shared with the calibrated model).
            freeform: ``(region, region)`` fixed out-of-span aberration in
                radians; ``None`` means zeros.
            illumination_exponent: Super-Gaussian exponent; 2.0 == Gaussian
                (same illumination as the calibrated model).
            shot_noise: Relative Poisson noise level (0 == noiseless); the
                peak-normalised intensity is resampled as a Poisson count with
                unit-signal relative sigma equal to this value.
            read_noise: Additive Gaussian read-noise sigma in peak-normalised
                units (0 == noiseless).
            seed: RNG seed; two twins with the same seed produce identical
                noisy outputs.

        Raises:
            ValueError: On a bad exponent / negative noise / wrong
                ``freeform`` shape.
        """
        if illumination_exponent < 1.0:
            raise ValueError(
                f"illumination_exponent must be >= 1.0, got {illumination_exponent}"
            )
        if shot_noise < 0.0 or read_noise < 0.0:
            raise ValueError(
                "shot_noise and read_noise must be non-negative, got "
                f"{shot_noise} / {read_noise}"
            )
        region = geometry.region
        if freeform is None:
            freeform_tensor = torch.zeros(region, region, dtype=torch.float32)
        else:
            if freeform.shape != (region, region):
                raise ValueError(
                    f"freeform must have shape ({region}, {region}), "
                    f"got {tuple(freeform.shape)}"
                )
            freeform_tensor = freeform.to(torch.float32)
        self._geometry = geometry
        self._freeform = freeform_tensor
        self._amplitude = _build_pupil_amplitude(geometry, illumination_exponent)
        self._shot_noise = float(shot_noise)
        self._read_noise = float(read_noise)
        self._generator = torch.Generator().manual_seed(int(seed))

    @property
    def geometry(self) -> Bench2FGeometry:
        """The bench geometry this twin simulates."""
        return self._geometry

    def pupil_amplitude(self) -> torch.Tensor:
        """Super-Gaussian illumination x hard circular aperture.

        Returns:
            ``(1, 1, region, region)`` float32 amplitude, peak 1.
        """
        return self._amplitude.clone()

    def complex_far_field(
        self, phase: torch.Tensor, *, aberration: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Propagate to the far field and return the **complex** spectrum.

        This is the same propagation :meth:`forward` performs, minus the
        intensity/peak-normalisation and the detector noise.  It exists so a
        Gerchberg-Saxton iteration can run *through* the ground truth: GS needs
        the far field's phase, which the intensity alone does not carry.

        Args:
            phase: ``(B, 1, region, region)`` pupil phase in radians.
            aberration: Optional extra pupil aberration in radians, same shape
                as ``phase``.  This is how one bench serves many training
                samples: each sample can carry its own Zernike / freeform
                aberration instead of every sample sharing one bench.

        Returns:
            ``(B, 1, far_field_size, far_field_size)`` complex far field,
            peak-normalised so that ``max|F| == 1``.
        """
        _check_phase_batch(phase, self._geometry.region)
        total = phase
        if aberration is not None:
            _check_phase_batch(aberration, self._geometry.region)
            total = total + aberration
        field = self._amplitude * torch.exp(
            1j * (total + self._freeform.reshape(1, 1, *self._freeform.shape))
        )
        spectrum = focal_field(field, self._geometry.far_field_size)
        peak = spectrum.abs().amax(dim=(-2, -1), keepdim=True)
        return spectrum / torch.where(peak > 0, peak, torch.ones_like(peak))

    def forward(
        self, phase: torch.Tensor, *, aberration: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Propagate with the extra ground-truth physics.

        Same chain as :meth:`Bench2FPropagator.forward` except: the phase is
        NOT Zernike-reduced, the freeform aberration is added to the pupil
        phase, the illumination may be super-Gaussian, and detector noise is
        applied AFTER peak normalisation.  ``shot_noise`` resamples the
        (clamped) intensity as Poisson counts with relative level
        ``shot_noise`` at the peak; ``read_noise`` then adds Gaussian noise in
        the same peak-normalised units.  Both default to 0, keeping the twin
        deterministic.

        Args:
            phase: ``(B, 1, region, region)`` pupil phase in radians.

        Returns:
            ``(B, 1, far_field_size, far_field_size)`` peak-normalised
            intensity (plus noise when configured).

        Raises:
            ValueError: If ``phase`` has the wrong shape.
        """
        _check_phase_batch(phase, self._geometry.region)
        total = phase
        if aberration is not None:
            _check_phase_batch(aberration, self._geometry.region)
            total = total + aberration
        field = self._amplitude * torch.exp(
            1j * (total + self._freeform.reshape(1, 1, *self._freeform.shape))
        )
        intensity = _normalised_intensity(field, self._geometry)
        if self._shot_noise > 0.0:
            counts_per_unit = 1.0 / (self._shot_noise * self._shot_noise)
            intensity = torch.poisson(
                intensity.clamp(min=0.0) * counts_per_unit,
                generator=self._generator,
            ) / counts_per_unit
        if self._read_noise > 0.0:
            intensity = intensity + self._read_noise * torch.randn(
                intensity.shape,
                dtype=intensity.dtype,
                device=intensity.device,
                generator=self._generator,
            )
        return intensity
