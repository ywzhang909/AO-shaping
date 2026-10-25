"""Frozen configuration contract for :mod:`ml.sim2f`.

This module is the **single shared interface** that ``optics`` / ``targets`` /
``teacher`` / ``synth`` / ``dataset`` / ``model`` / ``losses`` / ``fit`` /
``loop`` / ``train`` / ``evaluate`` all import.  It deliberately imports
**stdlib only** -- no ``torch``, no ``numpy``, no device code -- so the parser,
the cache metadata and the report generator can all read the same numbers
without pulling the optional ``ml`` dependency group into the process.

Why a frozen dataclass
----------------------
``train._build_parser`` / ``__main__._build_parser`` read **every** CLI default
from these dataclasses instead of writing literals into ``add_argument``.  That
is not a style preference: this repository shipped a bug where ``train_amp``'s
parser hard-wrote ``default=1`` for ``far_field_padding`` and silently
overrode a calibrated value of ``10`` on every run
(``report/zernike_phase2amp/report.md``).  A frozen dataclass makes "the parser
default" and "the dataclass default" the same object, so they cannot drift.

The optics this contract describes
----------------------------------
A 2f Fourier bench: the SLM sits in the front focal plane of a lens and the
camera in the back focal plane.  **The reference implementation is
``ao_shaping.drivers.sim.bench_twin``** (the calibrated digital twin behind
``report/shape_trials/``), and every name/unit below is chosen so a
``BenchTwinConfig`` can be built from this config by renaming fields -- that is
what ``tests/ao_shaping/ml/sim2f/test_twin_parity.py`` checks.  Nothing here is
an invented bench.

The pupil field is sampled on the **effective pupil pitch** of the model grid,
``pupil_pitch = region_span_px / grid * panel_pixel`` (900/128 x 8 um = 56.25 um
at the reference geometry); the Fourier transform of that array is sampled on

    ``far_field_pitch_m = wavelength * focal_length / (fft_length * pupil_pitch)``

with ``fft_length = grid * far_field_padding``, which has nothing to do with the
camera's pixel pitch ``ccd_pixel_um``.  On the real bench the camera pixel is
*finer* than that natural pitch -- 2.2 um against 4.618 um, i.e.
``sampling_ratio = far_field_pitch / ccd_pitch = 2.10`` -- so every place that
samples the model at camera coordinates must integrate over the camera pixel
area instead of pretending the FFT sample *is* a camera pixel.  The hardware
reports measured that integration to move the prediction by only 0.1-0.2%
(``report/shape_trials/model_in_loop/physical_forward_sampling_20261022_r2.md``),
so it is correctness, not the dominant error term.
:attr:`Sim2fConfig.exact_fft_n` is the (expensive) length at which the two
pitches would coincide.

``far_field_padding`` is the **angular-scale** knob, and it is *not* a cosmetic
one: the twin was calibrated at 4 and ``bench_twin``'s own constructor default
of 10 puts the far-field scale off by a factor.  It is decoupled from both grids
on purpose.

Noll coefficients
-----------------
``coefficients`` throughout this package are **radians** for Noll indices
``2 .. n_modes + 1`` (0-based row ``i`` <-> Noll index ``i + 2``), i.e. the
array *includes* tip and tilt (Noll 2, 3) and *excludes* piston (Noll 1).
Phase generators in this repository return **raw unwrapped radians**; the only
``mod 2*pi`` wrap lives in the SLM driver, so nothing here wraps either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "AberrationConfig",
    "DEVICE_SPECS",
    "LoopConfig",
    "OpticsErrors",
    "Sim2fConfig",
    "SynthConfig",
    "TARGET_KINDS",
    "TeacherConfig",
    "TrainConfig",
    "fast_len",
]


#: Target shape families understood by :mod:`ml.sim2f.targets`.  Order is the
#: on-disk ``kind`` index, so **appending is safe and reordering is not**: a
#: cached ``pairs.npz`` stores ``kind`` as an integer.
TARGET_KINDS: tuple[str, ...] = (
    "square",
    "rect",
    "disc",
    "ring",
    "cross",
    "two_spot",
    "line",
    "gaussian",
)

#: Accepted ``device`` values.  ``"auto"`` resolves to CUDA when available.
DEVICE_SPECS = ("auto", "cuda", "cpu")


def fast_len(n: int) -> int:
    """Smallest 5-smooth (``2^a * 3^b * 5^c``) integer ``>= n``.

    cuFFT is dramatically faster on these lengths, and the difference is not a
    micro-optimisation at ``fft_length >= 512`` -- a prime-ish length of the same
    magnitude costs several times more.

    Args:
        n: Lower bound.  Must be positive.

    Returns:
        The smallest 5-smooth integer ``>= n``.

    Raises:
        ValueError: If ``n`` is not a positive integer.

    >>> fast_len(4819)
    4860
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"fast_len needs a positive int, got {n!r}")
    # Start from the next power of two -- always a valid 5-smooth upper bound.
    best = 1 << (n - 1).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            # Smallest power of two that lifts this 3^b * 5^c to >= n.
            candidate = p35
            while candidate < n:
                candidate *= 2
            if candidate < best:
                best = candidate
            p35 *= 3
        p5 *= 5
    return best


@dataclass(frozen=True)
class OpticsErrors:
    """Optical non-idealities of the 2f bench -- mirror of ``TwinErrorModel``.

    Field-for-field the same knobs, names and units as
    :class:`ao_shaping.drivers.sim.bench_twin.TwinErrorModel`, so a
    ``BenchTwin`` can be built from this object by renaming fields (which is
    what the parity test does).  Two deliberate additions:

    * ``zero_order_eta`` -- ``ZeroOrderModel.eta``.  A **coherent** pupil-domain
      background, ``U = U_in * [eta + (1 - eta) * exp(i*phi)]``
      (``drivers/sim/nonideal.py``), i.e. the unmodulated reflection forming the
      0-order bulge on top of the shaped spot.  It is *not* a detector term and
      it does *not* conserve the modulated energy.
    * ``lut_table`` is not carried: only the generalised gamma form is modelled,
      because that is what the reference twin fitted (``slm_lut_gamma = 1.0``,
      i.e. linear in this round).

    Every default is the identity -- ``OpticsErrors()`` is an ideal bench.
    """

    lut_gamma: float = 1.0
    lut_phase_max_rad: float = 2.0 * math.pi
    gray_levels: int = 1023
    slm_zernike: tuple[tuple[int, float], ...] = ()
    slm_zernike_n_max: int = 6
    slm_roughness_sigma_rad: float = 0.0
    slm_roughness_seed: int = 0
    defocus_m: float = 0.0
    tilt_x_rad: float = 0.0
    tilt_y_rad: float = 0.0
    lens_zernike: tuple[tuple[int, float], ...] = ()
    lens_zernike_n_max: int = 4
    zero_order_eta: float = 0.0

    def __post_init__(self) -> None:
        for name in ("lut_gamma", "lut_phase_max_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")
        if isinstance(self.gray_levels, bool) or not isinstance(self.gray_levels, int):
            raise ValueError(f"gray_levels must be an int, got {self.gray_levels!r}")
        if self.gray_levels < 1:
            raise ValueError(f"gray_levels must be >= 1, got {self.gray_levels}")
        if not (0.0 <= float(self.zero_order_eta) < 1.0):
            raise ValueError(f"zero_order_eta must be in [0, 1), got {self.zero_order_eta!r}")
        for name in (
            "slm_roughness_sigma_rad",
            "tilt_x_rad",
            "tilt_y_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if not math.isfinite(float(self.defocus_m)):
            raise ValueError(f"defocus_m must be finite, got {self.defocus_m!r}")
        for name in ("slm_zernike", "lens_zernike"):
            modes = getattr(self, name)
            if not isinstance(modes, tuple):
                raise ValueError(
                    f"{name} must be a tuple of (noll, radians) pairs "
                    f"(hashable, so the config stays a frozen dataclass), got {type(modes)!r}"
                )
            for item in modes:
                if len(item) != 2 or not all(isinstance(v, (int, float)) for v in item):
                    raise ValueError(f"{name} entries must be (noll: int, rad: float), got {item!r}")
        for name in ("slm_zernike_n_max", "lens_zernike_n_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an int >= 2, got {value!r}")

    @property
    def is_identity(self) -> bool:
        """True when nothing at all is injected (bit-identical to the ideal bench)."""
        return (
            float(self.lut_gamma) == 1.0
            and not self.slm_zernike
            and float(self.slm_roughness_sigma_rad) == 0.0
            and float(self.defocus_m) == 0.0
            and float(self.tilt_x_rad) == 0.0
            and float(self.tilt_y_rad) == 0.0
            and not self.lens_zernike
            and float(self.zero_order_eta) == 0.0
        )

    @classmethod
    def calibrated(cls) -> "OpticsErrors":
        """The fitted twin of 2026-10-23 (``sim_twin_params_20261023.json``).

        Only two knobs are non-identity in that round: the camera sits
        **10.0847 mm** beyond focus (``f^2 / 1549.33 mm``) and **10%** of the
        light stays in the 0-order.  Panel static aberration and roughness were
        fitted to zero on purpose -- adding low-order Zernike to widen the spot
        turned the flat top into a spike and made CV worse.
        """
        return cls(
            lut_gamma=1.0,
            lut_phase_max_rad=2.0 * math.pi,
            gray_levels=1023,
            defocus_m=0.010084735,
            zero_order_eta=0.1,
        )

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable dump (tuples become lists, for run metadata)."""
        return {
            "lut_gamma": float(self.lut_gamma),
            "lut_phase_max_rad": float(self.lut_phase_max_rad),
            "gray_levels": int(self.gray_levels),
            "slm_zernike": [[int(n), float(c)] for n, c in self.slm_zernike],
            "slm_zernike_n_max": int(self.slm_zernike_n_max),
            "slm_roughness_sigma_rad": float(self.slm_roughness_sigma_rad),
            "slm_roughness_seed": int(self.slm_roughness_seed),
            "defocus_m": float(self.defocus_m),
            "tilt_x_rad": float(self.tilt_x_rad),
            "tilt_y_rad": float(self.tilt_y_rad),
            "lens_zernike": [[int(n), float(c)] for n, c in self.lens_zernike],
            "lens_zernike_n_max": int(self.lens_zernike_n_max),
            "zero_order_eta": float(self.zero_order_eta),
        }


@dataclass(frozen=True)
class Sim2fConfig:
    """Geometry, sampling and detector model of the simulated 2f bench.

    Every field is physical and is *used* -- there is no cosmetic parameter.
    The derived properties below are the only sanctioned way to ask "what pitch
    / what PSF / what ratio is this", so a caller cannot invent a second,
    slightly different conversion.

    Attributes:
        grid: Pupil sampling grid side, in model samples.  Also the side of the
            phase map the U-Net emits.  Even and ``>= 64``.
        region_span_px: SLM panel pixels covered by ``grid``.  ``grid`` and this
            together set the *effective* pupil pitch; 128 / 900 at the reference
            geometry.
        panel_pixel_um: Physical SLM panel pixel pitch, um (8 um LCOS class).
        pupil_fill: Support diameter of the Zernike basis / U-Net mask, as a
            fraction of ``grid``.  A **modelling** disc, not an aperture: the
            entrance amplitude is the Gaussian of ``beam_w0_px``.
        wavelength_m: Vacuum wavelength, metres.
        focal_length_m: Focal length of the Fourier lens, metres.
        far_field_padding: Zero-padding factor of the focal-plane FFT;
            ``fft_length = grid * far_field_padding``.  This is the *angular
            scale* knob the twin was calibrated on (4 reproduces the measured
            far-field scale; the twin's own constructor default of 10 does not).
        ccd_px: Camera window side, in camera pixels.  Also the side of the
            spot / target images.  A multiple of 4 so a 4-level U-Net can halve
            it cleanly.
        ccd_pixel_um: Camera pixel pitch, um.  Independent of the pupil pitch;
            see :attr:`sampling_ratio`.  2.2 um for the Daheng MER2-507.
        beam_w0_px: Gaussian entrance beam **1/e^2 intensity radius, in panel
            pixels** (``BenchTwinConfig.beam_w0_px``, same unit).  There is no
            hard aperture: hard truncation manufactures Airy rings the bench
            does not have.
        phase_blur_px: Std-dev, in model samples, of the Gaussian blur applied to
            the commanded phase before it modulates the field.  Models LCOS
            crosstalk.  **0 by default**, and must stay 0 for the BenchTwin
            parity test -- the reference twin does not model crosstalk.
        errors: :class:`OpticsErrors` -- LUT, panel aberration, alignment and
            0-order knobs (mirror of ``TwinErrorModel``).  Identity by default.
        exposure_ms: Exposure, ms.  ``0`` (default) switches the whole detector
            off and the frame is the ideal normalised intensity.  Quantisation
            would otherwise sit in the middle of the teacher's gradient path.
        peak_counts_per_ms: Counts per ms at the peak of the *ideally focused
            flat* frame (``counts_reference="flat"``), i.e. a fixed radiometric
            constant -- never renormalised per frame, or a defocused frame would
            silently regain its full peak.
        counts_reference: ``"flat"`` (normalise against the ideal focused flat
            peak -- the reference convention) or ``"frame"`` (per-frame peak;
            only for diagnostics, it cancels exactly the defocus spreading the
            twin is supposed to show).
        read_sigma_counts: Gaussian read-noise std-dev, counts.
        dark_level_counts: Global dark offset, counts.
        shot_k: Poisson gain (counts per photo-electron); 0 = no shot noise,
            which is what the reference twin fitted.
        prnu_sigma: Fixed multiplicative pixel-response non-uniformity std-dev.
        dark_fpn_sigma: Fixed per-pixel dark offset std-dev, counts.
        quantize_bits: ADC depth; ``0`` = no detector.  8 for the Daheng camera.
        detector_seed: Seed of the fixed PRNU / dark-FPN maps (fixed => the same
            bench every call, as on hardware).
    """

    # -- geometry: mirror of ``BenchTwinConfig`` --------------------------
    # Same names, same units, same defaults as ``drivers/sim/bench_twin.py``
    # except ``far_field_padding`` (that config ships 10, the fitted twin of
    # 2026-10-23 uses 4) -- so a BenchTwin can be built from this config by
    # renaming fields, and the parity test compares the two directly.
    grid: int = 128
    region_span_px: float = 900.0
    panel_pixel_um: float = 8.0
    pupil_fill: float = 0.75
    wavelength_m: float = 1.064e-6
    focal_length_m: float = 0.125
    far_field_padding: int = 4
    ccd_px: int = 256
    ccd_pixel_um: float = 2.2
    beam_w0_px: float = 120.0
    phase_blur_px: float = 0.0
    errors: OpticsErrors = OpticsErrors()

    # -- detector / radiometry: fitted twin of 2026-10-23 ------------------
    # Defaults are the *reference values* but everything is inert until
    # ``exposure_ms > 0``: a quantised, noisy frame must never be the default
    # because quantisation kills the gradient the teacher optimises through.
    # ``Sim2fConfig.calibrated()`` turns the whole detector on.
    exposure_ms: float = 0.0
    peak_counts_per_ms: float = 133.5
    counts_reference: Literal["flat", "frame"] = "flat"
    read_sigma_counts: float = 0.0
    dark_level_counts: float = 26.75
    shot_k: float = 0.0
    prnu_sigma: float = 0.0
    dark_fpn_sigma: float = 0.0
    quantize_bits: int = 0
    detector_seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.grid, bool) or not isinstance(self.grid, int):
            raise ValueError(f"grid must be an int, got {self.grid!r}")
        if self.grid < 64 or self.grid % 2:
            raise ValueError(f"grid must be even and >= 64, got {self.grid}")
        if isinstance(self.ccd_px, bool) or not isinstance(self.ccd_px, int):
            raise ValueError(f"ccd_px must be an int, got {self.ccd_px!r}")
        if self.ccd_px < 16 or self.ccd_px % 4:
            raise ValueError(f"ccd_px must be a multiple of 4 and >= 16, got {self.ccd_px}")
        if isinstance(self.far_field_padding, bool) or not isinstance(self.far_field_padding, int):
            raise ValueError(f"far_field_padding must be an int, got {self.far_field_padding!r}")
        if self.far_field_padding < 1:
            raise ValueError(f"far_field_padding must be >= 1, got {self.far_field_padding}")
        for name, low in (
            ("region_span_px", 0.0),
            ("panel_pixel_um", 0.0),
            ("wavelength_m", 0.0),
            ("focal_length_m", 0.0),
            ("ccd_pixel_um", 0.0),
            ("beam_w0_px", 0.0),
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= low:
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")
        if not (0.0 < float(self.pupil_fill) <= 1.0):
            raise ValueError(f"pupil_fill must be in (0, 1], got {self.pupil_fill!r}")
        if not isinstance(self.errors, OpticsErrors):
            raise ValueError(f"errors must be an OpticsErrors, got {type(self.errors)!r}")
        if self.counts_reference not in ("flat", "frame"):
            raise ValueError(
                f"counts_reference must be 'flat' or 'frame', got {self.counts_reference!r}"
            )
        if isinstance(self.quantize_bits, bool) or not isinstance(self.quantize_bits, int):
            raise ValueError(f"quantize_bits must be an int, got {self.quantize_bits!r}")
        if self.quantize_bits and not (1 <= self.quantize_bits <= 16):
            raise ValueError(f"quantize_bits must be 0 (off) or 1..16, got {self.quantize_bits}")
        for name in (
            "phase_blur_px",
            "exposure_ms",
            "read_sigma_counts",
            "dark_level_counts",
            "shot_k",
            "prnu_sigma",
            "dark_fpn_sigma",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")

    # -- pitches ----------------------------------------------------------
    @property
    def pupil_pitch_m(self) -> float:
        """Effective pupil sampling pitch, metres.

        ``region_span_px / grid`` panel pixels per model sample, times the
        panel pitch.  This is what the Fourier transform actually sees; it
        equals :attr:`panel_pixel_um` only when ``region_span_px == grid``.
        Reference: 900 / 128 panel px × 8 um = 56.25 um.
        """
        return float(self.region_span_px) / float(self.grid) * (float(self.panel_pixel_um) * 1e-6)

    @property
    def ccd_pitch_m(self) -> float:
        """Camera pixel pitch, metres (field name kept for the old callers)."""
        return float(self.ccd_pixel_um) * 1e-6

    @property
    def beam_w0_model_px(self) -> float:
        """Gaussian illumination 1/e² *intensity* radius, in pupil-grid samples.

        The panel-pixel value divided by the sampling ratio of the pupil grid;
        reference: 120 / (900 / 128) = 17.07 samples.  The forward model uses
        the Gaussian only -- there is no hard aperture (see the class docstring).
        """
        return float(self.beam_w0_px) * float(self.grid) / float(self.region_span_px)

    # -- pupil ------------------------------------------------------------
    @property
    def pupil_radius_px(self) -> float:
        """Zernike / mask support radius, in pupil-grid samples.

        A *modelling* disc (the basis the aberrations are expanded on and the
        U-Net's input mask), **not** an aperture in the optical path: the
        entrance amplitude is the Gaussian of :attr:`beam_w0_model_px`.
        """
        return 0.5 * float(self.pupil_fill) * float(self.grid)

    @property
    def aperture_m(self) -> float:
        """Effective aperture diameter, metres (hard-disc equivalent).

        ``pupil_fill * grid * pupil_pitch`` = 5.4 mm at the reference geometry.
        Diagnostics only -- the Gaussian tapers before this disc, so treat it
        as an upper bound on the diffracting aperture.
        """
        return float(self.pupil_fill) * float(self.grid) * float(self.pupil_pitch_m)

    @property
    def psf_scale_m(self) -> float:
        """Diffraction scale ``lambda * f / D``, metres (**estimate**).

        The optical resolution limit at the focal plane: no target feature
        smaller than roughly this can be produced, whatever the phase map says.
        Because :attr:`aperture_m` is a hard-disc equivalent of a Gaussian
        entrance, this is an estimate (the twin's own FWHM disagrees with the
        hard-disc Airy value by ~25% -- see
        ``report/shape_trials/sim_vs_hw/sim_vs_hw_20261023.md``).  The
        authoritative number is the measured FWHM of the ideal flat frame.
        """
        return float(self.wavelength_m) * float(self.focal_length_m) / self.aperture_m

    # -- focal plane sampling --------------------------------------------
    @property
    def fft_length(self) -> int:
        """Zero-padded FFT length = ``grid * far_field_padding`` (reference: 512)."""
        return int(self.grid) * int(self.far_field_padding)

    @property
    def far_field_pitch_m(self) -> float:
        """Natural focal-plane pitch of the FFT, metres.

        ``wavelength * focal_length / (fft_length * pupil_pitch)``.  This is the
        spacing between *samples of the transform*, not between camera pixels.
        Reference: 4.618 um against a 2.2 um camera pixel.
        """
        return (
            float(self.wavelength_m)
            * float(self.focal_length_m)
            / (self.fft_length * float(self.pupil_pitch_m))
        )

    @property
    def sampling_ratio(self) -> float:
        """``far_field_pitch / ccd_pitch`` -- how many camera pixels a model sample spans.

        ``< 1``: the model grid is finer than the camera, so the camera pixel
        *bins* several samples (the honest case -- camera pixels integrate).
        ``> 1``: the camera oversamples the model grid, so each camera pixel
        only covers part of one sample and the resampling is an interpolation;
        the information limit is then the model's sampling, not the camera.
        ``1`` exactly: the two grids coincide (see :attr:`exact_fft_n`).

        The reference geometry sits at **2.10** -- the camera oversamples, so
        the model grid is the information limit, and this is the same regime the
        hardware reports validated (pixel integration moves the prediction by
        only 0.1-0.2%: ``report/shape_trials/model_in_loop/
        physical_forward_sampling_20261022_r2.md``).
        """
        return self.far_field_pitch_m / self.ccd_pitch_m

    @property
    def exact_fft_n(self) -> int:
        """FFT length at which ``far_field_pitch_m == ccd_pitch_m``.

        Rounded up to a 5-smooth length.  This is the only length at which the
        transform grid *is* the camera grid; it is reported so a fidelity check
        can be run, but it is not the default because its cost is quadratic in
        a number several times the pupil grid (1075 -> 1080 here, against 512).
        """
        needed = (
            float(self.wavelength_m)
            * float(self.focal_length_m)
            / (float(self.pupil_pitch_m) * self.ccd_pitch_m)
        )
        return max(fast_len(int(math.ceil(needed))), int(self.grid))

    # -- camera -----------------------------------------------------------
    @property
    def ccd_fov_m(self) -> float:
        """Camera window width, metres (``ccd_px * ccd_pitch``)."""
        return float(self.ccd_px) * self.ccd_pitch_m

    @property
    def fov_in_psf(self) -> float:
        """Camera window width in diffraction scales -- the shaping budget.

        A target can only carry structure down to ~1 PSF, so this is
        approximately the number of independent cells across the window.
        """
        return self.ccd_fov_m / self.psf_scale_m

    @property
    def pixel_ratio_to_psf(self) -> float:
        """Camera pixels per diffraction scale (``psf_scale / ccd_pitch``)."""
        return self.psf_scale_m / self.ccd_pitch_m

    @property
    def gray_max(self) -> int:
        """Highest code value of the detector (``2**quantize_bits - 1``; 0 = no detector)."""
        return (1 << int(self.quantize_bits)) - 1 if self.quantize_bits else 0

    @property
    def detector_enabled(self) -> bool:
        """Whether the detector model (counts + noise + quantisation) is switched on."""
        return float(self.exposure_ms) > 0.0

    def describe(self) -> dict[str, float | int]:
        """Flatten the derived numbers into a JSON-serialisable mapping.

        Used by cache metadata, run configs and the report, so the three never
        disagree about e.g. the sampling ratio that produced a run.

        Returns:
            Mapping of derived name -> value (ints stay ints).
        """
        return {
            "grid": int(self.grid),
            "ccd_px": int(self.ccd_px),
            "fft_length": int(self.fft_length),
            "exact_fft_n": int(self.exact_fft_n),
            "pupil_pitch_m": self.pupil_pitch_m,
            "beam_w0_model_px": self.beam_w0_model_px,
            "aperture_m": self.aperture_m,
            "psf_scale_m": self.psf_scale_m,
            "far_field_pitch_m": self.far_field_pitch_m,
            "sampling_ratio": self.sampling_ratio,
            "ccd_fov_m": self.ccd_fov_m,
            "fov_in_psf": self.fov_in_psf,
            "pixel_ratio_to_psf": self.pixel_ratio_to_psf,
            "zero_order_eta": float(self.errors.zero_order_eta),
            "defocus_m": float(self.errors.defocus_m),
            "detector_enabled": bool(self.detector_enabled),
        }

    def to_dict(self) -> dict[str, float | int | bool | str]:
        """Round-trippable field dump (physics fields plus :meth:`describe`)."""
        payload: dict[str, float | int | bool | str] = {
            "grid": int(self.grid),
            "region_span_px": float(self.region_span_px),
            "panel_pixel_um": float(self.panel_pixel_um),
            "pupil_fill": float(self.pupil_fill),
            "wavelength_m": float(self.wavelength_m),
            "focal_length_m": float(self.focal_length_m),
            "far_field_padding": int(self.far_field_padding),
            "ccd_px": int(self.ccd_px),
            "ccd_pixel_um": float(self.ccd_pixel_um),
            "beam_w0_px": float(self.beam_w0_px),
            "phase_blur_px": float(self.phase_blur_px),
            "exposure_ms": float(self.exposure_ms),
            "peak_counts_per_ms": float(self.peak_counts_per_ms),
            "counts_reference": str(self.counts_reference),
            "read_sigma_counts": float(self.read_sigma_counts),
            "dark_level_counts": float(self.dark_level_counts),
            "shot_k": float(self.shot_k),
            "prnu_sigma": float(self.prnu_sigma),
            "dark_fpn_sigma": float(self.dark_fpn_sigma),
            "quantize_bits": int(self.quantize_bits),
            "detector_seed": int(self.detector_seed),
            "errors": self.errors.as_dict(),
        }
        payload.update(self.describe())
        return payload

    @classmethod
    def calibrated(cls) -> "Sim2fConfig":
        """The fitted twin of ``report/shape_trials/sim_vs_hw/artifacts/sim_twin_params_20261023.json``.

        Geometry is unchanged (that is not an "error"), but the optical errors
        and the detector are switched on with the *fitted* values.  Use this for
        the closed-loop / reference-environment run; use the bare constructor
        for teacher labels and U-Net supervision.
        """
        base = cls()
        return cls(
            **{
                **{
                    name: getattr(base, name)
                    for name in (
                        "grid",
                        "region_span_px",
                        "panel_pixel_um",
                        "pupil_fill",
                        "wavelength_m",
                        "focal_length_m",
                        "far_field_padding",
                        "ccd_px",
                        "ccd_pixel_um",
                        "beam_w0_px",
                        "phase_blur_px",
                    )
                },
                "errors": OpticsErrors.calibrated(),
                "exposure_ms": 0.15,
                "peak_counts_per_ms": 133.5,
                "counts_reference": "flat",
                "read_sigma_counts": 0.5,
                "dark_level_counts": 26.75,
                "shot_k": 0.0,
                "prnu_sigma": 0.015,
                "dark_fpn_sigma": 2.0,
                "quantize_bits": 8,
                "detector_seed": 0,
            }
        )


@dataclass(frozen=True)
class TeacherConfig:
    """Differentiable phase optimisation that produces the *labels*.

    The teacher runs on the **aberration-free** bench: its output is the phase
    that shapes the ideal spot into the requested target.  Aberrations are
    handled downstream (probe estimate + closed-loop residual fit), which is
    what lets one network serve every aberration state.

    The objective is a *distribution* objective, in the same two parts the
    :class:`~ml.sim2f.loop` driver reports:

    * ``L_shape`` -- L1 (or MSE) between the predicted and target intensity
      after each is normalised to unit sum, i.e. "does the energy land in the
      right places";
    * ``w_ee * (1 - EE)`` -- ``EE`` is the fraction of the predicted energy that
      falls inside the target support, i.e. "does the energy stay in the target
      instead of spraying over the whole window".

    Both are needed: ``L_shape`` alone is scale-free and happily empties the
    target, and ``EE`` alone is maximised by a delta function (the repo measured
    exactly this failure for ``-CV``-only square shaping).

    Attributes:
        iters: Adam iterations per target.  This is the teacher's wall-clock
            knob and the amortisation the network is asked to replace.  ``800``
            is the measured knee, not a guess -- same target (``gaussian``,
            ``w_ee=0.25``, ``lr=0.06``): 80 iters -> L1 0.368 / NCC 0.904,
            400 -> 0.061 / 0.992, 1200 -> 0.017 / 0.9995.  A 160-iteration
            teacher would label the corpus with an optimum a quarter of the way
            down that curve, and the reported amortisation gap would then be an
            artefact of the label quality.  Costs ~3.5 s/sample on CUDA
            (~35 min for the 600-sample corpus).
        lr: Initial Adam learning rate on the free phase.
        lr_final_frac: Cosine schedule floor, as a fraction of ``lr``.
        w_ee: Weight of the encircled-energy term.
        w_tv: Total-variation weight on the phase.  Default ``0``: the repo
            measured a phase-smoothness penalty *harmful* for phase prediction
            (paired dR2 -0.026 at w=0.003, 0/3 seeds positive), and the three
            earlier shaping audits all failed with a smoothing term on.  Kept
            as an off-by-default knob with that history, not a recommendation.
        metric: ``"l1"`` (default) or ``"mse"`` for ``L_shape``.
        prepare: Remove the detector pedestal (median subtract, clip at 0)
            before forming any ratio -- :func:`ml.sim2f.losses.objective_terms`'s
            ``prepare`` switch.  On by default because the trainer and the
            evaluator score prepared frames; leaving it off here would give the
            teacher a different objective from the one the report measures
            (measured 0.7% on the ideal bench, 2.5-2.8% on ``cross`` / ``rect``,
            and larger once the detector pedestal is in play).
        init: ``"flat"`` starts from zero phase (honest "from scratch");
            ``"conjugate"`` is not offered here on purpose -- the teacher is
            aberration-free, so there is nothing to conjugate.
    """

    iters: int = 800
    lr: float = 0.06
    lr_final_frac: float = 0.1
    w_ee: float = 0.25
    w_tv: float = 0.0
    metric: Literal["l1", "mse"] = "l1"
    prepare: bool = True
    init: Literal["flat"] = "flat"

    def __post_init__(self) -> None:
        if self.iters < 1:
            raise ValueError(f"iters must be >= 1, got {self.iters}")
        if not (0.0 < float(self.lr)):
            raise ValueError(f"lr must be > 0, got {self.lr!r}")
        if not (0.0 <= float(self.lr_final_frac) <= 1.0):
            raise ValueError(f"lr_final_frac must be in [0, 1], got {self.lr_final_frac!r}")
        for name in ("w_ee", "w_tv"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        if self.metric not in ("l1", "mse"):
            raise ValueError(f"metric must be 'l1' or 'mse', got {self.metric!r}")
        if self.init != "flat":
            raise ValueError(f"init must be 'flat', got {self.init!r}")

    def to_dict(self) -> dict[str, float | int | str]:
        """Round-trippable field dump."""
        return {
            "iters": int(self.iters),
            "lr": float(self.lr),
            "lr_final_frac": float(self.lr_final_frac),
            "w_ee": float(self.w_ee),
            "w_tv": float(self.w_tv),
            "metric": str(self.metric),
            "prepare": bool(self.prepare),
            "init": str(self.init),
        }


@dataclass(frozen=True)
class SynthConfig:
    """How the supervised corpus is drawn and split.

    Attributes:
        count: Number of ``(target, teacher phase)`` pairs.  The teacher label
            depends only on the target (the bench is aberration-free), so this
            is also the number of teacher optimisations.
        seed: Master seed.  Also the cache identity: changing it changes the
            corpus.
        target_kinds: Shape families to draw from; must be a subset of
            :data:`TARGET_KINDS`.
        size_frac: ``(min, max)`` target size as a fraction of ``ccd_px``.
        min_feature_px: Floor on a target's **thinnest bright feature**, camera
            pixels on the reference window (``ccd_px = 256``).  ``24`` is not a
            taste knob: the calibrated twin's PSF FWHM is **16.0 CCD px /
            35.2 um** (measured, ``report/shape_trials/sim_vs_hw/``), so a
            feature below ``1.5 x FWHM`` is a delta function to this bench, not a
            shape.  Measured with the 800-iteration teacher
            (:func:`ml.sim2f.losses.objective_terms`): ``line`` -- drawn at
            1.5-3.0 px, i.e. 0.12 x FWHM -- stalls at L1 1.62 / NCC 0.38 while
            ``gaussian`` reaches 0.029.  :func:`ml.sim2f.targets.draw_target`
            therefore grows each family until its thinnest feature clears this
            floor and raises otherwise (no silent resampling, no unreachable
            pair in the cache).
        train_frac: Fraction of pairs in the training split.
        val_frac: Fraction of pairs in the validation split.  The remainder is
            the test split.
        batch_size: DataLoader batch size (kept here so the cache metadata and
            the trainer's parser agree).
    """

    count: int = 600
    seed: int = 20261024
    target_kinds: tuple[str, ...] = TARGET_KINDS
    size_frac: tuple[float, float] = (0.16, 0.44)
    min_feature_px: float = 24.0
    train_frac: float = 0.8
    val_frac: float = 0.1
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"count must be >= 1, got {self.count}")
        if not self.target_kinds:
            raise ValueError("target_kinds must not be empty")
        unknown = [kind for kind in self.target_kinds if kind not in TARGET_KINDS]
        if unknown:
            raise ValueError(f"unknown target kinds {unknown}; known: {list(TARGET_KINDS)}")
        if len(set(self.target_kinds)) != len(self.target_kinds):
            raise ValueError(f"target_kinds has duplicates: {list(self.target_kinds)}")
        low, high = (float(self.size_frac[0]), float(self.size_frac[1]))
        if not (0.0 < low <= high < 1.0):
            raise ValueError(f"size_frac must satisfy 0 < low <= high < 1, got {self.size_frac!r}")
        if not math.isfinite(float(self.min_feature_px)) or float(self.min_feature_px) <= 0.0:
            raise ValueError(
                f"min_feature_px must be finite and > 0, got {self.min_feature_px!r}"
            )
        if not (0.0 < float(self.train_frac) < 1.0):
            raise ValueError(f"train_frac must be in (0, 1), got {self.train_frac!r}")
        if not (0.0 <= float(self.val_frac) < 1.0):
            raise ValueError(f"val_frac must be in [0, 1), got {self.val_frac!r}")
        if float(self.train_frac) + float(self.val_frac) >= 1.0:
            raise ValueError(
                "train_frac + val_frac must leave a test split, got "
                f"{self.train_frac} + {self.val_frac}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

    def to_dict(self) -> dict[str, object]:
        """Round-trippable field dump."""
        return {
            "count": int(self.count),
            "seed": int(self.seed),
            "target_kinds": list(self.target_kinds),
            "size_frac": [float(v) for v in self.size_frac],
            "min_feature_px": float(self.min_feature_px),
            "train_frac": float(self.train_frac),
            "val_frac": float(self.val_frac),
            "batch_size": int(self.batch_size),
        }


@dataclass(frozen=True)
class AberrationConfig:
    """The aberration state the closed loop must survive, and how it is probed.

    ``n_modes`` counts Noll modes ``4 .. n_modes + 1`` (0-based row ``i`` <->
    Noll ``i + 2``).  Tip/tilt are drawn separately as a *spot offset in camera
    pixels* (``tilt_px``) because that is the observable an operator would
    recognise, and converted to radians internally.

    Attributes:
        n_modes: Number of Zernike modes (Noll 4..) in the aberration state.
        rms_rad: ``(min, max)`` RMS of the drawn high-order aberration, radians.
        tilt_px: ``(min, max)`` magnitude of the tip/tilt spot offset, camera
            pixels.  Drawn on a random azimuth.
        illum_jitter_frac: Illumination centre jitter, as a fraction of the
            pupil radius.
        illum_width: ``(min, max)`` illumination waist scale.
        probe_amplitude_rad: Amplitude of each Zernike probe applied during the
            pre-loop probe stage.
        probe_modes: Noll indices probed.  Empty means "the first
            ``n_modes`` modes after 4" (one probe image per mode).
        probe_frozen: Noll indices held at zero during the *probe-stage* fit.
            Only piston is frozen: on a flat spot the tilt (Noll 2, 3) is
            directly visible as a spot shift, and correcting it is what puts a
            later shaped spot back on target.
        fit_iters: Adam iterations of one aberration fit (probe stage or
            closed-loop refit).
        fit_lr: Adam learning rate for the coefficients.
        fit_frozen: Noll indices held at zero during the *closed-loop* refit.
            Defaults to piston/tip/tilt: once the target itself fixes where the
            energy goes, the spot position no longer identifies tilt, and
            letting it move would absorb real shaping error into a degenerate
            direction.
    """

    n_modes: int = 21
    rms_rad: tuple[float, float] = (0.05, 0.45)
    tilt_px: tuple[float, float] = (0.0, 6.0)
    illum_jitter_frac: float = 0.06
    illum_width: tuple[float, float] = (0.9, 1.1)
    probe_amplitude_rad: float = 0.35
    probe_modes: tuple[int, ...] = ()
    probe_frozen: tuple[int, ...] = (1,)
    fit_iters: int = 80
    fit_lr: float = 0.05
    fit_frozen: tuple[int, ...] = (1, 2, 3)

    def __post_init__(self) -> None:
        if self.n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {self.n_modes}")
        low, high = (float(self.rms_rad[0]), float(self.rms_rad[1]))
        if not (0.0 <= low <= high):
            raise ValueError(f"rms_rad must satisfy 0 <= low <= high, got {self.rms_rad!r}")
        low, high = (float(self.tilt_px[0]), float(self.tilt_px[1]))
        if not (0.0 <= low <= high):
            raise ValueError(f"tilt_px must satisfy 0 <= low <= high, got {self.tilt_px!r}")
        low, high = (float(self.illum_width[0]), float(self.illum_width[1]))
        if not (0.0 < low <= high):
            raise ValueError(f"illum_width must satisfy 0 < low <= high, got {self.illum_width!r}")
        if not (0.0 <= float(self.illum_jitter_frac) < 1.0):
            raise ValueError(
                f"illum_jitter_frac must be in [0, 1), got {self.illum_jitter_frac!r}"
            )
        if not (0.0 < float(self.probe_amplitude_rad) <= math.pi):
            raise ValueError(
                f"probe_amplitude_rad must be in (0, pi], got {self.probe_amplitude_rad!r}"
            )
        if int(self.fit_iters) < 1:
            raise ValueError(f"fit_iters must be >= 1, got {self.fit_iters}")
        if float(self.fit_lr) <= 0.0:
            raise ValueError(f"fit_lr must be > 0, got {self.fit_lr!r}")
        for noll in (*self.probe_modes, *self.probe_frozen, *self.fit_frozen):
            if int(noll) < 1:
                raise ValueError(f"Noll indices are 1-based and >= 1, got {noll}")

    @property
    def coefficient_noll(self) -> tuple[int, ...]:
        """Noll indices covered by :class:`~ml.sim2f.fit`'s coefficient vector."""
        return tuple(range(2, self.n_modes + 2))

    def to_dict(self) -> dict[str, object]:
        """Round-trippable field dump."""
        return {
            "n_modes": int(self.n_modes),
            "rms_rad": [float(v) for v in self.rms_rad],
            "tilt_px": [float(v) for v in self.tilt_px],
            "illum_jitter_frac": float(self.illum_jitter_frac),
            "illum_width": [float(v) for v in self.illum_width],
            "probe_amplitude_rad": float(self.probe_amplitude_rad),
            "probe_modes": [int(v) for v in self.probe_modes],
            "probe_frozen": [int(v) for v in self.probe_frozen],
            "fit_iters": int(self.fit_iters),
            "fit_lr": float(self.fit_lr),
            "fit_frozen": [int(v) for v in self.fit_frozen],
        }


@dataclass(frozen=True)
class TrainConfig:
    """U-Net training contract.

    The network learns ``target spot -> ideal-bench phase`` only.  It is never
    shown an aberration, which is why the supervised leg (teacher phase) and the
    physics leg (ideal-bench forward model) can both be evaluated without
    knowing the aberration state.

    Attributes:
        width: Channels in the first encoder stage; doubles per level.
        depth: Number of pooling levels.  ``ccd_px`` must be divisible by
            ``2 ** depth``.
        batch_size: Training batch size.
        steps: Optimiser steps (one epoch is defined as one pass over the cache,
            but training is step-based so the budget does not drift with
            ``count``).
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        w_sup: Weight of the **piston-free phasor** loss against the teacher.
        w_phy: Weight of the physics loss (forward the predicted phase on the
            ideal bench, compare with the target).  Neither leg alone is
            sufficient: a global phase offset is invisible to the physics and
            pins nothing, and the teacher phase alone is one arbitrary sample
            from a manifold the physics only constrains weakly.
        w_tv: Total-variation weight on the phasor.  Off by default; see
            :class:`TeacherConfig`.
        val_every: Steps between validation passes.
        seed: Torch / NumPy seed.  With ``cudnn_deterministic`` a rerun is
            bit-identical (the repo measured a 0.006 R2 spread without it).
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
        cudnn_deterministic: Sets ``torch.backends.cudnn.deterministic`` and
            ``benchmark=False``.  Costs ~4% wall clock and buys reruns that are
            bit-identical; default on because every comparison in the report is
            paired.
        num_workers: DataLoader workers.  Default ``0``: the corpus is a few
            hundred in-memory tensors, so workers only add Windows spawn cost.
        out_dir: Run directory for checkpoints, ``config.json``,
            ``metrics.json`` and ``history.csv``.
        amp: Use CUDA autocast for the U-Net (not for the FFT legs, which stay
            float32 either way).
    """

    width: int = 32
    depth: int = 5
    batch_size: int = 8
    steps: int = 1500
    lr: float = 2.0e-4
    weight_decay: float = 0.0
    w_sup: float = 1.0
    w_phy: float = 1.0
    w_tv: float = 0.0
    val_every: int = 100
    seed: int = 0
    device: str = "auto"
    cudnn_deterministic: bool = True
    num_workers: int = 0
    out_dir: str = "runs/sim2f"
    amp: bool = False

    def __post_init__(self) -> None:
        if int(self.width) < 4:
            raise ValueError(f"width must be >= 4, got {self.width}")
        if int(self.depth) < 2:
            raise ValueError(f"depth must be >= 2, got {self.depth}")
        if int(self.batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if int(self.steps) < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if float(self.lr) <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr!r}")
        for name in ("w_sup", "w_phy", "w_tv"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        if float(self.w_sup) == 0.0 and float(self.w_phy) == 0.0:
            raise ValueError("at least one of w_sup / w_phy must be > 0")
        if int(self.val_every) < 1:
            raise ValueError(f"val_every must be >= 1, got {self.val_every}")
        if self.device not in DEVICE_SPECS:
            raise ValueError(f"device must be one of {DEVICE_SPECS}, got {self.device!r}")
        if int(self.num_workers) < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")

    def to_dict(self) -> dict[str, object]:
        """Round-trippable field dump."""
        return {
            "width": int(self.width),
            "depth": int(self.depth),
            "batch_size": int(self.batch_size),
            "steps": int(self.steps),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "w_sup": float(self.w_sup),
            "w_phy": float(self.w_phy),
            "w_tv": float(self.w_tv),
            "val_every": int(self.val_every),
            "seed": int(self.seed),
            "device": str(self.device),
            "cudnn_deterministic": bool(self.cudnn_deterministic),
            "num_workers": int(self.num_workers),
            "out_dir": str(self.out_dir),
            "amp": bool(self.amp),
        }


@dataclass(frozen=True)
class LoopConfig:
    """Closed-loop evaluation contract.

    One evaluation sample is one aberration state.  The loop is:

    1. **probe** -- apply a known Zernike probe (plus a flat reference),
       measure, fit ``c_hat`` (:mod:`ml.sim2f.fit`);
    2. **shape** -- ``phi_cmd = phase_net(target) - sum(c_hat * Z)``;
    3. **measure** -- camera frame, optionally with detector noise;
    4. **refit** -- re-fit ``c`` against *this* frame to absorb what the probe
       estimate missed, and go back to 2 for the configured count of rounds.

    The report compares ``flat`` (no phase), ``net_open`` (network phase, no
    aberration correction), ``net_probe`` (probe estimate only) and
    ``net_closed`` (probe + closed-loop refit), plus ``oracle`` (true
    aberration), so the two correction legs are separated.

    Attributes:
        samples: Number of aberration states evaluated.
        rounds: Closed-loop refinement rounds (each = measure + refit + reshape).
        probe_fit: Run the pre-loop probe stage.
        refit: Run the closed-loop refit stage.
        use_detector: Switch the detector model on for the loop (8-bit counts,
            read noise, dark/PRNU maps, per :class:`Sim2fConfig`).  Off by
            default, and the caller must then also pass
            ``Sim2fConfig.calibrated()``-style detector fields -- this flag only
            gates whether the loop *reads* frames through the detector.
        seed: Seed for the aberration draw and detector noise.
        out_dir: Directory for ``eval_loop.json`` and per-sample CSV.
    """

    samples: int = 24
    rounds: int = 2
    probe_fit: bool = True
    refit: bool = True
    use_detector: bool = False
    seed: int = 7
    out_dir: str = "runs/sim2f_loop"

    def __post_init__(self) -> None:
        if int(self.samples) < 1:
            raise ValueError(f"samples must be >= 1, got {self.samples}")
        if int(self.rounds) < 0:
            raise ValueError(f"rounds must be >= 0, got {self.rounds}")
        if not self.probe_fit and not self.refit:
            raise ValueError("probe_fit and refit cannot both be off -- nothing would be corrected")
        if int(self.seed) < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")

    def to_dict(self) -> dict[str, object]:
        """Round-trippable field dump."""
        return {
            "samples": int(self.samples),
            "rounds": int(self.rounds),
            "probe_fit": bool(self.probe_fit),
            "refit": bool(self.refit),
            "use_detector": bool(self.use_detector),
            "seed": int(self.seed),
            "out_dir": str(self.out_dir),
        }
