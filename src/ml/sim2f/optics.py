"""Differentiable 2f bench forward model -- the only FFT in this package.

The bench
=========
The SLM sits in the front focal plane of a lens and the camera in the back focal
plane, so the camera measures ``|FT(pupil field)|**2``.  This module is the one
implementation of that map: :class:`Sim2f`.  The teacher, the physics loss, the
aberration fit and the closed loop all call it (rule 1 of :mod:`ml.sim2f`),
because a second propagator would make the teacher's labels inconsistent with
what the loop measures.

Pupil -> focal plane
====================
1. ``A`` -- the Gaussian entrance beam, ``exp(-r**2 / (2 * w0**2))`` with
   ``w0 = beam_w0_model_px * width_scale``, peak-normalised to 1.  There is **no
   hard aperture**: hard truncation manufactures Airy rings the bench does not
   have (``Sim2fConfig`` says so, and ``bench_twin`` matches it).  The pupil disc
   of ``pupil_radius_px`` is the *modelling* support of the Zernike basis and of
   the U-Net mask, not an aperture in the optical path.
2. ``U = A * exp(1j * phi)``, where ``phi`` has been through the panel stages
   (crosstalk blur, panel static aberration, device LUT round trip) and then the
   optical stages (lens aberration, camera defocus, the requested aberration).
   ``phi`` is **raw unwrapped radians** at every public boundary (rule 3); only
   the LUT round trip wraps, because a 2*pi-periodic device really does.
3. ``fftshift(fft2(ifftshift(center_pad(U, fft_n)), norm="ortho"))`` -- the
   repository's centred-FFT convention, optical axis at the array centre, and
   ``norm="ortho"`` so the transform preserves total power.
4. Optional alignment: a beam tilt translates the *focal* pattern by
   ``focal_length * tilt`` (Fourier shift).  Multiplying the focal field by a
   linear phase instead would translate the *pupil* and leave the intensity
   unchanged, which is not what a beam tilt does (``bench_twin`` measured this).

Camera sampling is *not* an FFT-grid read
=========================================
The FFT of an ``fft_n``-point pupil sample is sampled on its own natural pitch

    ``far_field_pitch_m = wavelength * focal_length / (fft_n * pupil_pitch)``

which has nothing to do with ``ccd_pixel_um``.  A camera pixel therefore covers

    ``ccd_pitch_m / far_field_pitch_m = 1 / sampling_ratio``

**model cells**, and :class:`Sim2f` integrates the model intensity over that area
with two separable overlap matrices ``Wx`` / ``Wy``; each row sums to
``1 / sampling_ratio``.  The reciprocal is not an accident of convention:
:attr:`Sim2fConfig.sampling_ratio` is defined as ``far_field_pitch / ccd_pitch``
(i.e. cell size over pixel size), so at the reference geometry (2.10) one camera
pixel spans ``0.476`` of a model sample -- the camera oversamples and the model's
sampling is the information limit.  Using ``sampling_ratio`` itself as the pixel
width would mis-scale the frame by ``1 / sampling_ratio ** 2`` in area (4.4x
here) and move a measured tilt by the same factor; the tests pin the correct
version against the measured centroid.

Detector
========
``Sim2fConfig.exposure_ms == 0`` (the default) means "no detector": ``forward``
returns model intensity.  With ``exposure_ms > 0`` the frame goes through
:meth:`Sim2f.counts` -- absolute photometry, fixed PRNU/dark maps, shot and read
noise, ADC quantisation -- because a quantised, noisy frame must never be the
default: quantisation kills the gradient the teacher optimises through.

Selected non-idealities (``Sim2fConfig.errors``) also run on the *phase* path,
which is why the device LUT round trip uses a straight-through estimator: its
forward value is the phase the panel really displays, its derivative is the
identity.  Without that, no gradient would reach the teacher's parameters at all.

Precision policy
================
The resampling matrices, the Zernike basis and the static error fields are built
in ``numpy`` / ``float64`` because they are *constructed* numbers, not learned
ones; they are then cast to the instance dtype (``float32`` unless a caller asks
for ``float64`` to run a finite-difference check).  Everything downstream runs in
that dtype with autograd on the phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from ao_shaping.utils.wavefront.zernike_calc import (
    ZernikeGenerator,
    calc_n_zernike_terms,
)
from ao_shaping.utils.wavefront.zernike_utils import generate_zernike_phase
from ml.sim2f.config import AberrationConfig, Sim2fConfig
from ml.sim2f._device import _resolve_device

__all__ = [
    "AberrationState",
    "DEFAULT_COVARIANCE_PATH",
    "Sim2f",
    "draw_aberration",
    "write_aberration",
]

#: Packaged von Karman Zernike covariance for the reference geometry
#: (``n_modes`` 21 -> Noll 4..22, grid 128).  Anchored to the package, never to
#: the CWD: the driver layer already paid for that lesson (a class-body
#: ``np.loadtxt("data/dm_adj.txt")`` made ``import ao_shaping`` fail anywhere but
#: the repo root).  Regenerate with
#: ``scripts/ml/build_sim2f_aberration_covariance.py``.
DEFAULT_COVARIANCE_PATH = (
    Path(__file__).parent / "assets" / "zernike_cov_kolmogorov_n21.npz"
)

#: Floor used when reducing a ``(cos, sin)`` pair to an angle, so an exactly zero
#: phasor resolves to angle 0 instead of NaN.
_PHASOR_EPS = 1e-12
#: Rank of a two-axis array read as a matrix: a coefficient table, or a square
#: covariance. These checks ask whether the argument is a matrix at all, not whether a
#: batch axis still has to be added.
NDIM_MATRIX = 2

#: Rank of an unbatched spatial map, i.e. ``(grid, grid)``. An input at this rank gets an
#: axis prepended before it reaches the pipeline. Separate from NDIM_MATRIX even though the
#: number is the same: one shape is promoted, the other is only validated.
NDIM_BARE_MAP = 2

#: How many scalars a two-component vector carries -- an x/y offset, or a pair of
#: coefficients. This counts elements of a flat vector, never tensor axes.
NPAIR_COMPONENTS = 2

#: Rank of a batched spatial map with no channel axis: ``(B, grid, grid)``.
NDIM_BATCH_MAP = 3

#: Rank of a batched map carrying an explicit channel axis: ``(B, C, n, n)``.
NDIM_BATCHED_MAP = 4


def _overlap_matrix(pixels: int, cells: int, ratio: float) -> np.ndarray:
    """Pixel-area integration matrix from the FFT grid to one camera axis.

    Entry ``(k, j)`` is the length of the overlap between the interval covered by
    camera pixel ``k`` and FFT cell ``j``, measured **in FFT-cell units**, so
    every row sums to ``1 / ratio`` (the pixel width in cells) wherever the camera
    window falls inside the model window.  ``ratio`` is
    ``far_field_pitch / ccd_pitch``, i.e. cell size over pixel size.

    Both axes use the repository's centred-FFT indexing: the coordinate of index
    ``i`` of an axis of length ``n`` is ``(i - n / 2)`` samples, and index ``i``
    covers ``[i - n / 2 - 0.5, i - n / 2 + 0.5]``.  ``n`` is even in every
    supported configuration (``grid`` even, ``ccd_px`` a multiple of 4).

    Args:
        pixels: Camera pixels along the axis.
        cells: FFT cells along the axis.
        ratio: ``far_field_pitch / ccd_pitch`` for this axis.

    Returns:
        ``(pixels, cells)`` float64 overlap matrix.
    """
    width = 1.0 / ratio
    index = np.arange(pixels, dtype=np.float64)
    cam_lo = (index - pixels / 2.0 - 0.5) * width
    cam_hi = cam_lo + width

    cells_index = np.arange(cells, dtype=np.float64)
    cell_lo = cells_index - cells / 2.0 - 0.5
    cell_hi = cell_lo + 1.0

    lo = np.maximum(cam_lo[:, None], cell_lo[None, :])
    hi = np.minimum(cam_hi[:, None], cell_hi[None, :])
    return np.clip(hi - lo, 0.0, None)


def _gaussian_kernel(sigma_px: float) -> np.ndarray:
    """Normalised 1-D Gaussian kernel sampled on integer offsets.

    Args:
        sigma_px: Standard deviation in model samples. Must be positive.

    Returns:
        Float64 kernel of length ``2 * ceil(4 * sigma) + 1`` summing to 1.
    """
    radius = max(1, int(math.ceil(4.0 * float(sigma_px))))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / float(sigma_px)) ** 2)
    return kernel / kernel.sum()


def _orders_for(n_modes: int) -> int:
    """Smallest Zernike radial order whose basis covers ``n_modes`` terms.

    Args:
        n_modes: Number of Noll terms required (piston included).

    Returns:
        Radial order ``n`` with ``calc_n_zernike_terms(n) >= n_modes``.
    """
    order = 1
    while calc_n_zernike_terms(order) < int(n_modes):
        order += 1
    return order


def _noll_map(payload: object) -> dict[int, float]:
    """Normalise a Noll-index mapping to ``{int_noll: float_radians}``.

    ``OpticsErrors`` carries the pairs as a tuple of ``(noll, radians)`` (so the
    config stays hashable); a plain mapping (JSON, a test double) is accepted too.

    Args:
        payload: Tuple of pairs, mapping, or empty.

    Returns:
        ``{int_noll: float_radians}``.
    """
    if not payload:
        return {}
    if isinstance(payload, dict):
        return {int(key): float(value) for key, value in payload.items()}
    return {int(pair[0]): float(pair[1]) for pair in payload}


def _static_zernike(
    coefficients: dict[int, float], n_max: int, grid: int, radius_px: float
) -> np.ndarray:
    """Pupil phase of a Noll-index coefficient dict, via the canonical engine.

    Args:
        coefficients: Noll index -> radians.
        n_max: Radial order for the basis.
        grid: Pupil grid side.
        radius_px: Normalisation radius, in model samples. This is
            :attr:`Sim2fConfig.pupil_radius_px` -- the config's own "Zernike
            support" -- so the panel and lens modes share one aperture
            convention with :meth:`Sim2f.zernike_basis`. (``bench_twin`` uses
            ``region / 2`` instead; the two coincide at ``pupil_fill == 1``.)

    Returns:
        ``(grid, grid)`` float64 radians, zero outside the support disc.
    """
    raw = generate_zernike_phase(
        {int(k): float(v) for k, v in coefficients.items()},
        resolution=(grid, grid),
        n_max=int(n_max),
        radius=float(radius_px),
    )
    return np.nan_to_num(
        np.asarray(raw, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )


def _roughness(sigma_rad: float, seed: int, grid: int) -> np.ndarray:
    """Fixed-seed Gaussian panel roughness with the requested RMS.

    Args:
        sigma_rad: Target RMS, radians.
        seed: Generator seed, so the same bench is reproducible.
        grid: Pupil grid side.

    Returns:
        ``(grid, grid)`` float64 radians.
    """
    rng = np.random.default_rng(int(seed))
    raw = rng.normal(0.0, 1.0, size=(grid, grid))
    rms = float(np.sqrt(np.mean(raw * raw)))
    if rms <= 0.0:
        return np.zeros((grid, grid), dtype=np.float64)
    return raw * (float(sigma_rad) / rms)


class Sim2f:
    """The differentiable 2f bench: pupil phase in, camera frame out.

    All geometry, sampling and non-ideality comes from
    :class:`~ml.sim2f.config.Sim2fConfig`; this class adds no physical parameter
    of its own, so a cached corpus is fully described by its config.

    Attributes:
        cfg: The frozen bench configuration.
        errors: ``cfg.errors`` -- the optical non-ideality contract, kept as an
            attribute so callers can ask what this bench injects.
        device: Device every tensor lives on.
        dtype: Floating dtype of the model (``torch.float32`` normally).
        grid: Pupil grid side, in model samples.
        ccd_px: Camera window side, in camera pixels.
        fft_n: Zero-padded FFT length, ``grid * far_field_padding``.
    """

    def __init__(
        self,
        cfg: Sim2fConfig,
        *,
        device: str | torch.device = "auto",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Build the bench: pupil, camera sampling, basis and error fields.

        Args:
            cfg: Frozen bench configuration.
            device: ``"auto"``, a device string or a :class:`torch.device`.
            dtype: Model floating dtype. ``torch.float32`` for training and
                inference; ``torch.float64`` makes the finite-difference
                gradient check meaningful.

        Raises:
            ValueError: If ``dtype`` is not a floating torch dtype.
        """
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating torch dtype, got {dtype!r}")

        self.cfg: Sim2fConfig = cfg
        self.errors = cfg.errors
        self.device: torch.device = _resolve_device(device)
        self.dtype: torch.dtype = dtype
        self.grid: int = int(cfg.grid)
        self.ccd_px: int = int(cfg.ccd_px)
        self.fft_n: int = int(cfg.fft_length)

        # -- pupil coordinates -------------------------------------------------
        # Origin at ``(grid - 1) / 2``, i.e. the *same* coordinate grid the
        # canonical Zernike engine uses (``zernike_calc._build_cached_grid``).
        # Sharing it matters: with a half-pixel offset the engine's unit circle
        # cuts a crescent through the support disc, every mode comes back NaN
        # there, and the basis grows a zeroed rim that biased the tilt slope by
        # 4%.  A half-pixel displacement of the sampled pupil does not move the
        # far-field *intensity* (translating a pupil field only adds a linear
        # phase to its transform), so this costs nothing downstream.
        axis_px = np.arange(self.grid, dtype=np.float64) - (self.grid - 1) / 2.0
        self._x_px = np.broadcast_to(axis_px[None, :], (self.grid, self.grid)).copy()
        self._y_px = np.broadcast_to(axis_px[:, None], (self.grid, self.grid)).copy()
        radius_px = float(cfg.pupil_radius_px)
        support = np.hypot(self._x_px, self._y_px) <= radius_px
        self._support_np = support
        self._mask = torch.as_tensor(
            support.astype(np.float64), dtype=dtype, device=self.device
        ).view(1, 1, self.grid, self.grid)

        # -- camera sampling matrices -----------------------------------------
        ratio = float(cfg.sampling_ratio)
        wx = _overlap_matrix(self.ccd_px, self.fft_n, ratio)
        self._wx = torch.as_tensor(wx, dtype=dtype, device=self.device)
        self._wy_t = torch.as_tensor(wx.T.copy(), dtype=dtype, device=self.device)
        expected_row = 1.0 / ratio
        row_sum = float(wx.sum(axis=1).min())
        if row_sum < 0.999 * expected_row:
            logger.warning(
                "camera window {} px x {:.3g} m is wider than the model window "
                "{} px x {:.3g} m: outer camera pixels see no model sample "
                "(min row sum {:.4g} vs pixel width {:.4g} cells). Increase "
                "far_field_padding.",
                self.ccd_px,
                cfg.ccd_pitch_m,
                self.fft_n,
                cfg.far_field_pitch_m,
                row_sum,
                expected_row,
            )

        # -- panel crosstalk kernel -------------------------------------------
        self._blur_sigma = float(cfg.phase_blur_px)
        self._blur_radius = (
            max(1, int(math.ceil(4.0 * self._blur_sigma))) if self._blur_sigma > 0 else 0
        )
        self._blur_kernel = (
            torch.as_tensor(
                _gaussian_kernel(self._blur_sigma), dtype=dtype, device=self.device
            )
            if self._blur_sigma > 0
            else None
        )

        # -- Zernike basis (canonical engine only, never a local table) ---------
        self._basis_tensor: torch.Tensor | None = None
        self._basis_np = self._build_basis(3)
        # Radians per model sample produced by a unit coefficient on each tilt
        # mode. Taken from the *canonical* basis (rather than a hard-coded
        # ``2 / radius``) so the tilt <-> offset conversion cannot drift if the
        # engine's normalisation changes. The support disc is symmetric about the
        # optical axis, so the least-squares slope is exact.
        denom_x = float((self._x_px**2 * support).sum())
        denom_y = float((self._y_px**2 * support).sum())
        self._tilt_slope_px = float(
            (self._basis_np[1] * self._x_px * support).sum() / denom_x
        )
        self._tilt_slope_py = float(
            (self._basis_np[2] * self._y_px * support).sum() / denom_y
        )
        if self._tilt_slope_px <= 0.0 or self._tilt_slope_py <= 0.0:
            raise ValueError(
                "tilt basis slope must be positive; got "
                f"{self._tilt_slope_px!r} / {self._tilt_slope_py!r}"
            )

        # -- static error fields (all None / off for an ideal bench) -----------
        self._panel_phase: torch.Tensor | None = None
        self._lens_phase: torch.Tensor | None = None
        self._defocus_phase: torch.Tensor | None = None
        self._zero_weights: tuple[float, float] | None = None
        self._tilt_freq: torch.Tensor | None = None
        self._prnu: torch.Tensor | None = None
        self._dark: torch.Tensor | None = None
        self._counts_reference_peak: float | None = None
        self._build_error_fields()

        # Ideal (nominal) illumination: centred, unit width, no jitter.
        self._nominal_illum = self.illumination()

        logger.debug(
            "Sim2f ready: grid={} fft_n={} ccd_px={} ratio={:.4g} device={} dtype={} "
            "errors={} exposure_ms={}",
            self.grid,
            self.fft_n,
            self.ccd_px,
            ratio,
            self.device,
            dtype,
            "ideal" if self.errors.is_identity else "non-ideal",
            cfg.exposure_ms,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_error_fields(self) -> None:
        """Precompute every static field the non-idealities need.

        Nothing here touches the RNG per call: the PRNU / dark maps and the panel
        roughness are fixed by their seeds, exactly as on hardware.
        """
        errors = self.errors
        # Panel / lens modes are normalised on ``grid / 2``, not on
        # ``pupil_radius_px``: they are *device* aberrations spread over the whole
        # sampled panel, and ``bench_twin`` generates them with ``radius =
        # region / 2``. Using the modelling disc instead mismatched them by 2.6e-2
        # (flat frame) at the default ``pupil_fill = 0.75``. The loop's own
        # aberration basis keeps ``pupil_radius_px`` (see ``zernike_basis``).
        panel_radius = self.grid / 2.0

        panel = np.zeros((self.grid, self.grid), dtype=np.float64)
        slm_zernike = _noll_map(errors.slm_zernike)
        if slm_zernike:
            panel += _static_zernike(
                slm_zernike, errors.slm_zernike_n_max, self.grid, panel_radius
            )
        if errors.slm_roughness_sigma_rad > 0.0:
            panel += _roughness(
                errors.slm_roughness_sigma_rad, errors.slm_roughness_seed, self.grid
            )
        if np.any(panel):
            self._panel_phase = torch.as_tensor(
                panel, dtype=self.dtype, device=self.device
            )

        lens_zernike = _noll_map(errors.lens_zernike)
        if lens_zernike:
            lens = _static_zernike(
                lens_zernike, errors.lens_zernike_n_max, self.grid, panel_radius
            )
            self._lens_phase = torch.as_tensor(
                lens, dtype=self.dtype, device=self.device
            )

        if errors.defocus_m != 0.0:
            # Collins/ABCD: a detector at f + dz is the pupil quadratic phase
            # exp(-i pi dz r^2 / (lambda f^2)). bench_twin documents that this is
            # bit-equivalent to the focal-plane angular-spectrum form (and much
            # cheaper -- no extra FFT pair).
            #
            # Centre at ``grid / 2``, NOT at ``(grid - 1) / 2`` like the rest of
            # the pupil grid: ``bench_twin`` centres its quadratic on ``w / 2``,
            # and a half-sample offset adds a ~0.36 rad linear term over the pupil
            # (a spurious tilt worth ~0.5 model samples of spot shift), which is
            # far above the parity budget.
            axis_m = (
                np.arange(self.grid, dtype=np.float64) - self.grid / 2.0
            ) * float(self.cfg.pupil_pitch_m)
            r2 = axis_m[None, :] ** 2 + axis_m[:, None] ** 2
            defocus = (
                -math.pi
                * float(errors.defocus_m)
                * r2
                / (float(self.cfg.wavelength_m) * float(self.cfg.focal_length_m) ** 2)
            )
            self._defocus_phase = torch.as_tensor(
                defocus, dtype=self.dtype, device=self.device
            )

        if errors.zero_order_eta > 0.0:
            eta = min(max(float(errors.zero_order_eta), 0.0), 1.0)
            self._zero_weights = (math.sqrt(1.0 - eta), math.sqrt(eta))

        if errors.tilt_x_rad != 0.0 or errors.tilt_y_rad != 0.0:
            self._tilt_freq = torch.fft.fftfreq(
                self.fft_n, dtype=self.dtype, device=self.device
            ) * float(self.fft_n)

        self._build_detector_maps()

    def _build_detector_maps(self) -> None:
        """Build the fixed PRNU / dark maps, if this bench has a detector."""
        cfg = self.cfg
        if float(cfg.exposure_ms) <= 0.0:
            return
        shape = (self.ccd_px, self.ccd_px)
        seed = int(cfg.detector_seed)
        if float(cfg.prnu_sigma) > 0.0:
            rng = np.random.default_rng(seed + 0x51A3)
            prnu = rng.normal(1.0, float(cfg.prnu_sigma), size=shape)
            self._prnu = torch.as_tensor(prnu, dtype=self.dtype, device=self.device)
        dark = np.full(shape, float(cfg.dark_level_counts), dtype=np.float64)
        if float(cfg.dark_fpn_sigma) > 0.0:
            rng = np.random.default_rng(seed + 0xDA17)
            dark = dark + rng.normal(0.0, float(cfg.dark_fpn_sigma), size=shape)
        if np.any(dark):
            self._dark = torch.as_tensor(dark, dtype=self.dtype, device=self.device)

    def _build_basis(self, count: int) -> np.ndarray:
        """Build the first ``count`` Noll modes in float64 numpy.

        NaN handling: ``RZern.eval_grid`` returns NaN outside the unit circle.
        ``nan_to_num`` turns that exterior into exact zeros, which *is* the
        support mask (the generator shares this class's coordinate origin and
        radius), so no second masking step is applied -- masking NaNs with a
        multiply would keep the NaN, and re-masking the already-zeroed exterior
        used to cut a zeroed rim a sample wide.

        Args:
            count: Number of modes (Noll 1..count).

        Returns:
            ``(count, grid, grid)`` float64 array, zero outside the support.
        """
        order = _orders_for(count)
        generator = ZernikeGenerator(
            resolution=(self.grid, self.grid),
            radius=float(self.cfg.pupil_radius_px),
            n_orders=order,
        )
        nk = calc_n_zernike_terms(order)
        basis = np.zeros((count, self.grid, self.grid), dtype=np.float64)
        for index in range(count):
            coefficients = np.zeros(nk, dtype=np.float64)
            coefficients[index] = 1.0
            basis[index] = np.nan_to_num(
                generator.generate_noll(coefficients), nan=0.0, posinf=0.0, neginf=0.0
            )
        return basis

    # ------------------------------------------------------------------
    # Pupil-plane quantities
    # ------------------------------------------------------------------
    @property
    def pupil_mask(self) -> torch.Tensor:
        """Zernike / U-Net support disc, ``(1, 1, grid, grid)``.

        A *modelling* disc, not an aperture: the optical path has no hard edge
        (the entrance amplitude is the Gaussian of :attr:`nominal_illumination`).
        """
        return self._mask

    @property
    def nominal_illumination(self) -> torch.Tensor:
        """Centred, unit-width Gaussian beam, ``(1, 1, grid, grid)``."""
        return self._nominal_illum

    def illumination(
        self,
        center_px: tuple[float, float] = (0.0, 0.0),
        width_scale: float = 1.0,
    ) -> torch.Tensor:
        """Gaussian beam amplitude, peak-normalised to 1.

        ``exp(-r**2 / (2 * w0**2))`` -- the ``bench_twin`` convention for
        ``beam_w0_px`` -- with ``w0 = beam_w0_model_px * width_scale``. No
        aperture is applied (see the class docstring).

        Args:
            center_px: Beam centre offset from the optical axis, in model
                samples, as ``(x, y)`` (x = column, y = row).
            width_scale: Multiplier on the nominal waist.

        Returns:
            ``(1, 1, grid, grid)`` amplitude, maximum exactly 1.

        Raises:
            ValueError: If ``width_scale`` is not positive.
        """
        scale = float(width_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"width_scale must be finite and > 0, got {width_scale!r}")
        center_x = float(center_px[0])
        center_y = float(center_px[1])
        w0 = float(self.cfg.beam_w0_model_px) * scale
        r2 = (self._x_px - center_x) ** 2 + (self._y_px - center_y) ** 2
        amplitude = np.exp(-r2 / (2.0 * w0 * w0))
        peak = float(amplitude.max())
        amplitude = amplitude / peak
        return torch.as_tensor(
            amplitude, dtype=self.dtype, device=self.device
        ).view(1, 1, self.grid, self.grid)

    def zernike_basis(self, count: int) -> torch.Tensor:
        """Zernike modes Noll ``1 .. count`` on the pupil grid.

        Args:
            count: Number of modes. Row ``i`` is Noll index ``i + 1``.

        Returns:
            ``(count, grid, grid)`` real tensor in the model dtype, zero outside
            the support disc.

        Raises:
            ValueError: If ``count`` is not a positive integer.
        """
        if int(count) < 1:
            raise ValueError(f"count must be >= 1, got {count!r}")
        count = int(count)
        if self._basis_np.shape[0] < count:
            self._basis_np = self._build_basis(count)
            self._basis_tensor = None
        if self._basis_tensor is None or self._basis_tensor.shape[0] < count:
            self._basis_tensor = torch.as_tensor(
                self._basis_np, dtype=self.dtype, device=self.device
            )
        return self._basis_tensor[:count]

    def aberration(self, coeffs: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Zernike phase map from Noll coefficients ``2 .. K + 1``.

        Args:
            coeffs: ``(B, K)`` coefficients in **radians**. Row ``k`` of the
                output weights Noll index ``k + 2`` (tip/tilt included, piston
                excluded).

        Returns:
            ``(B, K) -> (B, grid, grid)`` phase in radians. Differentiable w.r.t.
            ``coeffs``.

        Raises:
            ValueError: If ``coeffs`` is not rank 2 or is empty in the mode axis.
        """
        tensor = coeffs if isinstance(coeffs, torch.Tensor) else torch.as_tensor(coeffs)
        if tensor.dim() != NDIM_MATRIX:
            raise ValueError(f"coeffs must be (B, K), got shape {tuple(tensor.shape)}")
        n_coeffs = int(tensor.shape[1])
        if n_coeffs < 1:
            raise ValueError("coeffs must have at least one coefficient")
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        basis = self.zernike_basis(n_coeffs + 1)[1 : n_coeffs + 1]
        return torch.einsum("bk,khw->bhw", tensor, basis)

    def blur_phase(self, phase: torch.Tensor) -> torch.Tensor:
        """Smooth a commanded phase with the panel's crosstalk kernel.

        Separable Gaussian, ``sigma = phase_blur_px``, radius ``ceil(4 sigma)``,
        reflect padding so the edge does not see a zero-phase ring. The kernel is
        a fixed buffer, so this adds no parameter and no RNG. Off by default: the
        calibrated twin does not model crosstalk, and the BenchTwin parity check
        requires ``phase_blur_px == 0``.

        Args:
            phase: ``(B, grid, grid)`` phase in radians.

        Returns:
            ``(B, grid, grid)`` blurred phase (the input itself when
            ``phase_blur_px == 0``).

        Raises:
            ValueError: If ``phase`` is not ``(B, grid, grid)``.
        """
        if phase.dim() != NDIM_BATCH_MAP or phase.shape[-2:] != (self.grid, self.grid):
            raise ValueError(
                f"phase must be (B, {self.grid}, {self.grid}), got "
                f"{tuple(phase.shape)}"
            )
        if self._blur_kernel is None:
            return phase
        batch = int(phase.shape[0])
        radius = self._blur_radius
        kernel = self._blur_kernel
        weight_y = kernel.view(1, 1, -1, 1).repeat(batch, 1, 1, 1)
        weight_x = kernel.view(1, 1, 1, -1).repeat(batch, 1, 1, 1)
        work = phase.unsqueeze(1)
        work = F.pad(work, (0, 0, radius, radius), mode="reflect")
        work = F.conv2d(work, weight_y, groups=batch)
        work = F.pad(work, (radius, radius, 0, 0), mode="reflect")
        work = F.conv2d(work, weight_x, groups=batch)
        return work.squeeze(1)

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------
    def field(
        self,
        phase: torch.Tensor | np.ndarray,
        *,
        aberration: torch.Tensor | np.ndarray | None = None,
        illumination: torch.Tensor | np.ndarray | None = None,
        aligned: bool = True,
    ) -> torch.Tensor:
        """Propagate a batch of pupil phases to the (padded) focal field.

        Args:
            phase: ``(B, grid, grid)`` radians or ``(B, 2, grid, grid)``
                ``(cos, sin)`` phasor -- the phasor is normalised to unit
                magnitude and reduced to an angle, so its scale does not matter.
            aberration: Optional aberration phase in radians, ``(grid, grid)``,
                ``(1, grid, grid)`` or ``(B, grid, grid)``. Applied *after* the
                panel round trip and the lens phase: it stands for the bench state
                the loop must estimate, not for the panel's own error.
            illumination: Optional amplitude, ``(1, 1, grid, grid)`` or
                ``(B, 1, grid, grid)``. Defaults to
                :attr:`nominal_illumination`.
            aligned: When ``False`` the alignment stages (camera defocus and beam
                tilt) are skipped. Only :meth:`counts` uses it, to build the fixed
                "flat, in-focus" photometric reference.

        Returns:
            ``(B, 1, fft_n, fft_n)`` complex field, optical axis at the array
            centre.

        Raises:
            ValueError: If the phase or a broadcast operand does not match the
                bench geometry.
        """
        phase_rad = self._as_phase(phase)
        phase_rad = self.blur_phase(phase_rad)

        # Panel family: the static aberration rides on the *command*, and the sum
        # then goes through the device round trip -- the order the panel sees.
        if self._panel_phase is not None:
            phase_rad = phase_rad + self._panel_phase
        actual = self._lut_round_trip(phase_rad)

        total = actual
        if self._lens_phase is not None:
            total = total + self._lens_phase
        if aligned and self._defocus_phase is not None:
            total = total + self._defocus_phase
        if aberration is not None:
            total = total + self._as_aberration(aberration, actual.shape[0])

        amplitude = self._as_illumination(illumination, actual.shape[0])
        u = torch.polar(amplitude, total)
        if self._zero_weights is not None:
            # Two-component LCoS pupil: the unmodulated part keeps the incident
            # amplitude, so it interferes with the shaped field instead of merely
            # adding a background. Energy in the *modulated* channel therefore
            # drops by eta -- never assume energy conservation when it is on.
            weight_mod, weight_zero = self._zero_weights
            u = weight_mod * u + weight_zero * amplitude

        pad = self.fft_n - self.grid
        if pad < 0:
            raise ValueError(
                f"fft_n ({self.fft_n}) must be >= grid ({self.grid})"
            )
        pad_lo = pad // 2
        pad_hi = pad - pad_lo
        u = F.pad(u, (pad_lo, pad_hi, pad_lo, pad_hi))
        u = u.unsqueeze(1)
        focal = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(u, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )
        if aligned and self._tilt_freq is not None:
            focal = self._apply_tilt(focal)
        return focal

    def _lut_round_trip(self, phase: torch.Tensor) -> torch.Tensor:
        """Device round trip: phase -> gray -> actual phase (straight-through).

        The forward value is what the panel really displays (wrapped to ``2*pi``,
        quantised to ``gray_levels``, then mapped through ``lut_gamma``).
        Quantisation has zero derivative, so the gradient is passed through
        unchanged (straight-through estimator); without that the teacher could not
        optimise through a quantised panel at all. A linear LUT (``gamma == 1``)
        is the identity and is skipped, keeping the ideal path bit-identical.

        Args:
            phase: ``(B, grid, grid)`` commanded phase, radians.

        Returns:
            ``(B, grid, grid)`` actually displayed phase, radians.
        """
        gamma = float(self.errors.lut_gamma)
        if gamma == 1.0:
            return phase
        levels = int(self.errors.gray_levels)
        if levels < 2:
            return phase
        two_pi = 2.0 * math.pi
        wrapped = torch.remainder(phase, two_pi)
        gray = torch.round(wrapped / two_pi * levels).clamp_(0.0, float(levels))
        shown = float(self.errors.lut_phase_max_rad) * (gray / float(levels)) ** gamma
        return phase + (shown - phase).detach()

    def _apply_tilt(self, focal: torch.Tensor) -> torch.Tensor:
        """Translate the focal pattern by ``focal_length * tilt`` metres.

        Fourier shift in the focal plane. Multiplying the focal field by a linear
        phase instead would *translate the pupil* and leave the intensity
        unchanged (``bench_twin`` documents that measurement), which is not what a
        beam tilt does.

        Args:
            focal: ``(B, 1, fft_n, fft_n)`` complex focal field.

        Returns:
            The translated field.
        """
        assert self._tilt_freq is not None
        dx = float(self.cfg.focal_length_m) * float(self.errors.tilt_x_rad)
        dy = float(self.cfg.focal_length_m) * float(self.errors.tilt_y_rad)
        freq = self._tilt_freq
        shift = torch.exp(
            -2j
            * math.pi
            * (freq[:, None] * dy + freq[None, :] * dx)
            / (float(self.fft_n) * float(self.cfg.far_field_pitch_m))
        )
        unshifted = torch.fft.ifftshift(focal, dim=(-2, -1))
        moved = torch.fft.ifft2(
            torch.fft.fft2(unshifted, norm="ortho") * shift, norm="ortho"
        )
        return torch.fft.fftshift(moved, dim=(-2, -1))

    def resample(self, intensity: torch.Tensor) -> torch.Tensor:
        """Integrate model intensity over camera-pixel areas.

        Args:
            intensity: ``(B, 1, fft_n, fft_n)`` non-negative model intensity.

        Returns:
            ``(B, 1, ccd_px, ccd_px)`` camera-frame intensity. Each output value
            is a sum of model samples weighted by overlap area, so the units are
            "model intensity times camera-pixel area in model cells".

        Raises:
            ValueError: If ``intensity`` is not ``(B, 1, fft_n, fft_n)``.
        """
        if intensity.dim() != NDIM_BATCHED_MAP or intensity.shape[-2:] != (self.fft_n, self.fft_n):
            raise ValueError(
                f"intensity must be (B, 1, {self.fft_n}, {self.fft_n}), got "
                f"{tuple(intensity.shape)}"
            )
        batch = int(intensity.shape[0])
        work = intensity.reshape(batch, self.fft_n, self.fft_n)
        xx = torch.matmul(self._wx, work)
        xy = torch.matmul(xx, self._wy_t)
        return xy.reshape(batch, 1, self.ccd_px, self.ccd_px)

    def forward(
        self,
        phase: torch.Tensor | np.ndarray,
        *,
        aberration: torch.Tensor | np.ndarray | None = None,
        illumination: torch.Tensor | np.ndarray | None = None,
        detector: bool | None = None,
        generator: torch.Generator | None = None,
        exposure_ms: float | None = None,
    ) -> torch.Tensor:
        """Pupil phase -> camera frame (the single forward entry point).

        Args:
            phase: ``(B, grid, grid)`` radians or ``(B, 2, grid, grid)`` phasor.
            aberration: Optional aberration phase, see :meth:`field`.
            illumination: Optional amplitude, see :meth:`field`.
            detector: ``None`` (default) follows the bench: counts iff the config
                has ``exposure_ms > 0``. ``False`` forces pure intensity (what the
                physics loss wants on any bench). ``True`` forces counts and
                raises if the bench has no exposure.
            generator: Optional torch RNG for the detector noise.
            exposure_ms: Exposure for this call, overriding the config's. Passing
                it always implies the detector.

        Returns:
            ``(B, 1, ccd_px, ccd_px)`` frame in the model dtype: model intensity,
            or counts when the detector is active.

        Raises:
            ValueError: If counts are requested from a bench with no exposure.
        """
        intensity = self.field(
            phase, aberration=aberration, illumination=illumination
        ).abs().square()
        frame = self.resample(intensity)

        exposure = float(self.cfg.exposure_ms) if exposure_ms is None else float(exposure_ms)
        if exposure_ms is not None:
            use_detector = True
        elif detector is None:
            use_detector = float(self.cfg.exposure_ms) > 0.0
        else:
            use_detector = bool(detector)
        if not use_detector:
            return frame
        if exposure <= 0.0:
            raise ValueError(
                "the detector needs a positive exposure; Sim2fConfig.exposure_ms is "
                f"{self.cfg.exposure_ms!r} and exposure_ms={exposure_ms!r}"
            )
        return self.counts(frame, exposure, generator=generator)

    def counts(
        self,
        frame: torch.Tensor,
        exposure_ms: float,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Model intensity -> detector counts (the calibrated photometry chain).

        Applies, in the order the hardware does:

        1. ``counts = fraction * peak_counts_per_ms * exposure_ms``, where
           ``fraction`` is the model intensity relative to the flat, in-focus
           frame peak when ``counts_reference == "flat"`` (so spreading the light
           really lowers the counts) or relative to the frame's own peak when it
           is ``"frame"``;
        2. add the dark map (global level + fixed-pattern noise), then multiply by
           the fixed PRNU map -- both seeded by ``detector_seed``, so they are
           identical between frames;
        3. shot noise ``shot_k * sqrt(signal)`` (``shot_k = 0`` disables it: the
           calibrated bench fitted zero), Gaussian read noise, and finally
           ``quantize_bits`` rounding/clipping.

        None of this is differentiable -- a detector is not part of the phase
        gradient -- so the result is detached from ``frame``.

        Args:
            frame: ``(B, 1, ccd_px, ccd_px)`` model intensity.
            exposure_ms: Exposure time in milliseconds. Must be >= 0.
            generator: Optional torch RNG for the shot / read noise.

        Returns:
            ``(B, 1, ccd_px, ccd_px)`` counts in the model dtype.

        Raises:
            ValueError: If ``exposure_ms`` is negative.
        """
        exposure = float(exposure_ms)
        if not math.isfinite(exposure) or exposure < 0.0:
            raise ValueError(f"exposure_ms must be finite and >= 0, got {exposure_ms!r}")

        cfg = self.cfg
        intensity = frame.detach().to(self.dtype)
        if cfg.counts_reference == "flat":
            fraction = intensity / max(self._flat_reference_peak(), 1e-12)
        else:
            per_frame = intensity.flatten(1).amax(dim=1).clamp_min(1e-12)
            fraction = intensity / per_frame.view(-1, 1, 1, 1)
        signal = fraction * float(cfg.peak_counts_per_ms) * exposure

        if self._dark is not None:
            signal = signal + self._dark
        if self._prnu is not None:
            signal = signal * self._prnu

        shot_k = float(cfg.shot_k)
        read_sigma = float(cfg.read_sigma_counts)
        if shot_k > 0.0:
            sigma = shot_k * signal.clamp_min(0.0).sqrt()
            signal = signal + self._noise(signal.shape, generator) * sigma
        if read_sigma > 0.0:
            signal = signal + self._noise(signal.shape, generator) * read_sigma
        if int(cfg.quantize_bits) > 0:
            signal = torch.round(signal).clamp_(0.0, float(cfg.gray_max))
        return signal.to(self.dtype)

    def _noise(
        self, shape: tuple[int, ...], generator: torch.Generator | None
    ) -> torch.Tensor:
        """Standard-normal draw on the model device/dtype.

        Args:
            shape: Output shape.
            generator: Optional torch RNG.

        Returns:
            Standard-normal tensor.
        """
        if generator is None:
            return torch.randn(shape, dtype=self.dtype, device=self.device)
        try:
            return torch.randn(
                shape, dtype=self.dtype, device=self.device, generator=generator
            )
        except (RuntimeError, TypeError):
            # A CPU generator cannot feed a CUDA tensor (and vice versa).
            return torch.randn(
                shape, dtype=self.dtype, device=generator.device
            ).to(self.device)

    def _flat_reference_peak(self) -> float:
        """Peak of the flat, in-focus model frame -- the photometric reference.

        The real bench's absolute photometry is calibrated on a flat field already
        focused onto the detector, so the reference must skip defocus and tilt
        (both are alignment errors, not part of "flat") while keeping the panel's
        own aberration (that *is* the panel). Cached: it is a bench constant.

        Returns:
            The reference peak, in model-intensity units.
        """
        if self._counts_reference_peak is None:
            flat = torch.zeros(
                1, self.grid, self.grid, dtype=self.dtype, device=self.device
            )
            intensity = self.field(flat, aligned=False).abs().square()
            self._counts_reference_peak = float(self.resample(intensity).max())
        return self._counts_reference_peak

    # ------------------------------------------------------------------
    # Tilt <-> spot offset
    # ------------------------------------------------------------------
    def tilt_coeffs_for_offset_px(
        self, offset_px: np.ndarray | tuple[float, float]
    ) -> np.ndarray:
        """Noll 2/3 coefficients (radians) that place the spot at ``offset_px``.

        A pupil ramp of slope ``s`` rad per pupil sample shifts the focal spot by
        ``lambda * f * s / (2 * pi * pupil_pitch)`` metres. With the canonical
        basis slope ``s = c * tilt_slope`` this inverts to
        ``c = 2 * pi * pupil_pitch * offset_m / (lambda * f * tilt_slope)``.

        The sign is ``+``: with the centred forward FFT of this module a phase
        increasing along ``+column`` moves the spot towards ``+column``. Verified
        numerically against the measured centroid: a requested ``(3, -2)`` px
        lands at ``(3.007, -2.006)`` px.

        Args:
            offset_px: ``(2,)`` spot offset in camera pixels, ``(x, y)`` with x =
                column and y = row, measured from the frame centre.

        Returns:
            ``(2,)`` float64 coefficients for Noll 2 (x) and Noll 3 (y).

        Raises:
            ValueError: If ``offset_px`` does not have two elements.
        """
        offset = np.asarray(offset_px, dtype=np.float64).reshape(-1)
        if offset.size != NPAIR_COMPONENTS:
            raise ValueError(f"offset_px must have two elements, got {offset_px!r}")
        scale = self._rad_per_camera_px()
        return np.array(
            [
                offset[0] * scale / self._tilt_slope_px,
                offset[1] * scale / self._tilt_slope_py,
            ],
            dtype=np.float64,
        )

    def offset_px_for_tilt_coeffs(
        self, coeffs: np.ndarray | tuple[float, float]
    ) -> np.ndarray:
        """Inverse of :meth:`tilt_coeffs_for_offset_px`.

        Args:
            coeffs: ``(2,)`` coefficients for Noll 2 (x) and Noll 3 (y), radians.

        Returns:
            ``(2,)`` float64 spot offset in camera pixels, ``(x, y)``.

        Raises:
            ValueError: If ``coeffs`` does not have two elements.
        """
        values = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        if values.size != NPAIR_COMPONENTS:
            raise ValueError(f"coeffs must have two elements, got {coeffs!r}")
        scale = self._rad_per_camera_px()
        return np.array(
            [
                values[0] * self._tilt_slope_px / scale,
                values[1] * self._tilt_slope_py / scale,
            ],
            dtype=np.float64,
        )

    def _rad_per_camera_px(self) -> float:
        """Ramp slope (rad per pupil sample) that moves the spot one camera pixel."""
        return (
            2.0
            * math.pi
            * float(self.cfg.pupil_pitch_m)
            * float(self.cfg.ccd_pitch_m)
            / (float(self.cfg.wavelength_m) * float(self.cfg.focal_length_m))
        )

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------
    def _as_phase(self, phase: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Normalise the phase argument to ``(B, grid, grid)`` radians.

        Args:
            phase: ``(B, grid, grid)`` radians or ``(B, 2, grid, grid)`` phasor.

        Returns:
            ``(B, grid, grid)`` radians in the model dtype/device.

        Raises:
            ValueError: If the rank/shape is not one of the two accepted forms.
        """
        tensor = phase if isinstance(phase, torch.Tensor) else torch.as_tensor(phase)
        if tensor.dim() == NDIM_BATCHED_MAP:
            if tensor.shape[1] != 2:
                raise ValueError(
                    "a rank-4 phase must be (B, 2, grid, grid) (cos, sin) channels, "
                    f"got shape {tuple(tensor.shape)}"
                )
            cos = tensor[:, 0]
            sin = tensor[:, 1]
            norm = torch.sqrt(cos * cos + sin * sin).clamp_min(_PHASOR_EPS)
            tensor = torch.atan2(sin / norm, cos / norm)
        elif tensor.dim() != NDIM_BATCH_MAP:
            raise ValueError(
                "phase must be (B, grid, grid) radians or (B, 2, grid, grid) "
                f"phasor, got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[-2:] != (self.grid, self.grid):
            raise ValueError(
                f"phase spatial size {tuple(tensor.shape[-2:])} != grid "
                f"{(self.grid, self.grid)}"
            )
        return tensor.to(device=self.device, dtype=self.dtype)

    def _as_aberration(
        self, aberration: torch.Tensor | np.ndarray, batch: int
    ) -> torch.Tensor:
        """Broadcast an aberration phase to ``(B, grid, grid)``.

        Args:
            aberration: ``(grid, grid)``, ``(1, grid, grid)`` or
                ``(B, grid, grid)`` radians.
            batch: Target batch size.

        Returns:
            ``(B, grid, grid)`` radians; gradients are preserved.

        Raises:
            ValueError: If the shape cannot broadcast to the batch.
        """
        tensor = (
            aberration
            if isinstance(aberration, torch.Tensor)
            else torch.as_tensor(aberration)
        )
        if tensor.dim() == NDIM_BARE_MAP:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != NDIM_BATCH_MAP or tensor.shape[-2:] != (self.grid, self.grid):
            raise ValueError(
                f"aberration must be (grid, grid), (1, grid, grid) or "
                f"(B, grid, grid) with grid={self.grid}, got "
                f"{tuple(tensor.shape)}"
            )
        if tensor.shape[0] not in (1, batch):
            raise ValueError(
                f"aberration batch {tensor.shape[0]} cannot broadcast to phase "
                f"batch {batch}"
            )
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        return tensor.expand(batch, self.grid, self.grid)

    def _as_illumination(
        self,
        illumination: torch.Tensor | np.ndarray | None,
        batch: int,
    ) -> torch.Tensor:
        """Broadcast an illumination amplitude to ``(B, grid, grid)``.

        Args:
            illumination: ``None`` (nominal), ``(1, 1, grid, grid)`` or
                ``(B, 1, grid, grid)``.
            batch: Target batch size.

        Returns:
            ``(B, grid, grid)`` amplitude.

        Raises:
            ValueError: If the shape cannot broadcast to the batch.
        """
        if illumination is None:
            return self._nominal_illum.view(1, self.grid, self.grid).expand(
                batch, self.grid, self.grid
            )
        tensor = (
            illumination
            if isinstance(illumination, torch.Tensor)
            else torch.as_tensor(illumination)
        )
        if tensor.dim() == NDIM_BARE_MAP:
            tensor = tensor.view(1, 1, self.grid, self.grid)
        elif tensor.dim() == NDIM_BATCH_MAP:
            tensor = tensor.unsqueeze(1)
        if tensor.dim() != NDIM_BATCHED_MAP or tensor.shape[1] != 1:
            raise ValueError(
                "illumination must be (1, 1, grid, grid) or (B, 1, grid, grid), "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[-2:] != (self.grid, self.grid):
            raise ValueError(
                f"illumination spatial size {tuple(tensor.shape[-2:])} != grid "
                f"{(self.grid, self.grid)}"
            )
        if tensor.shape[0] not in (1, batch):
            raise ValueError(
                f"illumination batch {tensor.shape[0]} cannot broadcast to phase "
                f"batch {batch}"
            )
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        return tensor.reshape(tensor.shape[0], self.grid, self.grid).expand(
            batch, self.grid, self.grid
        )


@dataclass(frozen=True)
class AberrationState:
    """One bench state: Zernike aberration plus illumination placement.

    The coefficients include tip/tilt (Noll 2, 3), because a mirror or lens
    misalignment shows up as a spot offset that the closed loop must also remove.
    :attr:`tilt_px` is the same information in the unit an operator recognises.

    Attributes:
        coeffs: ``(K,)`` float64 radians for Noll ``2 .. K + 1``.
        illum_center_px: Beam centre offset from the optical axis, model samples,
            ``(x, y)``.
        illum_width_scale: Multiplier on the nominal beam waist.
        px_per_rad: ``(2,)`` camera pixels per radian of Noll 2/3 -- the scale
            :attr:`tilt_px` needs. Filled by :func:`draw_aberration`; ``None`` for
            a state built by hand without optics, in which case :attr:`tilt_px`
            raises rather than silently guessing.
    """

    coeffs: np.ndarray
    illum_center_px: tuple[float, float]
    illum_width_scale: float
    px_per_rad: np.ndarray | None = None

    @property
    def tilt_px(self) -> np.ndarray:
        """``(2,)`` tip/tilt spot offset in camera pixels, ``(x, y)``.

        Returns:
            The offset implied by coefficients ``coeffs[:2]``.

        Raises:
            ValueError: If the state was not produced by :func:`draw_aberration`
                (no pixel scale available).
        """
        if self.px_per_rad is None:
            raise ValueError(
                "AberrationState has no px_per_rad scale, so radians cannot be "
                "converted to camera pixels; build it with draw_aberration(sim, ...)"
            )
        return np.asarray(self.coeffs[:2], dtype=np.float64) * np.asarray(
            self.px_per_rad, dtype=np.float64
        )


@lru_cache(maxsize=8)
def _covariance_factor(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a Kolmogorov covariance asset and factor it for drawing.

    The factor ``L = V diag(sqrt(clip(lambda, 0)))`` has ``L L^T = Sigma``, so
    ``L @ g`` is ``N(0, Sigma)`` for ``g ~ N(0, I)``.  Going through one
    eigendecomposition per *path* (not per draw) is what keeps the correlated
    path cheap enough for a corpus run.

    Negative eigenvalues are **clipped at zero**, deliberately: ``Sigma`` is a
    covariance projected through ``pinv``, so round-off leaves eigenvalues of
    order ``-1e-16 * max`` that must not reach ``sqrt``.  The clip cannot hide a
    rank-deficient asset -- the packaged one has ``min/max = 1.3e-2`` -- and
    ``sqrt`` of a clipped-to-zero eigenvalue simply yields a zero direction
    instead of a NaN.

    Args:
        path: Path to the ``.npz`` (``covariance``, ``noll``, plus provenance
            scalars).

    Returns:
        ``(noll, factor)``: the Noll indices covered, ``(K,)``, and the ``(K, K)``
        draw factor in ``rad``.

    Raises:
        FileNotFoundError: If the asset is missing.
        ValueError: If the asset is malformed (wrong shape, mismatched ``noll``,
            non-finite entries, or no positive eigenvalue at all).
    """
    asset = Path(path)
    if not asset.is_file():
        raise FileNotFoundError(
            f"Kolmogorov covariance asset not found: {asset}. Generate it with "
            "scripts/ml/build_sim2f_aberration_covariance.py, or leave "
            f"aberration_model='iid'. Packaged asset: {DEFAULT_COVARIANCE_PATH}"
        )
    with np.load(asset) as data:
        covariance = np.asarray(data["covariance"], dtype=np.float64)
        noll = np.asarray(data["noll"], dtype=np.int64)
    if covariance.ndim != NDIM_MATRIX or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            f"{asset} has covariance shape {covariance.shape}, expected (K, K)"
        )
    if noll.shape != (covariance.shape[0],):
        raise ValueError(
            f"{asset} has noll shape {noll.shape} but covariance shape "
            f"{covariance.shape}"
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{asset} has non-finite covariance entries")

    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    if float(eigenvalues.max()) <= 0.0:
        raise ValueError(f"{asset} covariance has no positive eigenvalue")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    factor = eigenvectors * np.sqrt(eigenvalues)[None, :]
    logger.debug(
        "loaded Kolmogorov Zernike covariance {} (Noll {}..{}, eig {:.3e}..{:.3e} rad^2)",
        asset.name,
        int(noll[0]),
        int(noll[-1]),
        float(eigenvalues.min()),
        float(eigenvalues.max()),
    )
    return noll, factor


def _correlated_high_order(white: np.ndarray, cfg: AberrationConfig) -> np.ndarray:
    """Map i.i.d. normals onto the correlated von Karman ensemble.

    ``white`` is consumed as-is, so the correlated and i.i.d. paths use **the
    same number of RNG draws** -- a seed cannot mean "the same noise" across
    models (that is impossible: the two ensembles differ), but it does mean the
    rest of the state (tilt, illumination) stays aligned for a given seed.

    Args:
        white: ``(K,)`` standard normals, ``K = n_modes - 2``.
        cfg: Distribution contract; ``covariance_path`` must be set.

    Returns:
        ``(K,)`` coefficients in ``rad``, distributed ``N(0, Sigma)``.

    Raises:
        ValueError: If the config has no path, or the asset covers a different
            number of modes.
    """
    if not cfg.covariance_path:
        raise ValueError(
            "aberration_model='kolmogorov' requires covariance_path; "
            f"see {DEFAULT_COVARIANCE_PATH}"
        )
    noll, factor = _covariance_factor(str(cfg.covariance_path))
    if factor.shape[0] != white.size:
        raise ValueError(
            f"covariance asset covers {factor.shape[0]} modes "
            f"(Noll {int(noll[0])}..{int(noll[-1])}) but n_modes="
            f"{cfg.n_modes} asks for {white.size} (Noll 4..{cfg.n_modes + 1}); "
            "rebuild it with scripts/ml/build_sim2f_aberration_covariance.py "
            f"--n-modes {cfg.n_modes}"
        )
    return factor @ white


def draw_aberration(
    sim: Sim2f,
    cfg: AberrationConfig,
    rng: np.random.Generator,
) -> AberrationState:
    """Draw one bench state from :class:`~ml.sim2f.config.AberrationConfig`.

    The high-order modes are drawn and then rescaled so the vector's RMS is
    uniform in ``cfg.rms_rad`` -- drawing both the RMS and the shape from one
    distribution would concentrate every state near the same magnitude.  The
    *shape* of the ensemble is ``cfg.aberration_model``:

    * ``"iid"`` (default): ``N(0, 1)`` per mode.  This is what every corpus so
      far was generated with, and the draw order is pinned by
      ``tests/ao_shaping/ml/sim2f/test_aberration_covariance.py`` -- adding a
      correlation option must not perturb these bytes.
    * ``"kolmogorov"``: ``N(0, Sigma)`` from a von Karman covariance asset
      (strongly correlated modes, variance falling with radial order).  The
      ``rms_rad`` rescale is applied *after* the correlated draw, so the
      amplitude knob -- and therefore the difficulty range -- is identical
      between the two models.

      That ordering has a measured cost worth knowing: fixing the vector's RMS
      puts every draw on a sphere, and an anisotropic ensemble squeezes toward
      isotropy on a sphere (Bingham).  Over 2000 draws the realised diagonal
      ladder is compressed by 0.85x at low order and 1.27x at the highest mode
      (measured in ``test_end_to_end_shape_survives_the_rms_rescale``), so the
      defocus-to-Noll-22 ratio lands at ~24x instead of the asset's 35x --
      still an order of magnitude away from the 1x an i.i.d. draw gives.  If the
      *exact* ``Sigma`` is ever needed, the amplitude has to come from the
      ensemble (e.g. draw the RMS from the physically implied Rayleigh/"energy
      in the pupil" distribution) instead of from an independent uniform
      ``rms_rad``.

    Tip/tilt is drawn as a spot offset (uniform magnitude on a uniform azimuth)
    because that is the observable, and converted through the bench.

    Args:
        sim: The bench the state belongs to (its geometry defines the scale).
        cfg: Aberration distribution contract.
        rng: NumPy generator. The caller owns seeding.

    Returns:
        The drawn :class:`AberrationState`, coefficients in float64 radians.
    """
    n_modes = int(cfg.n_modes)
    coeffs = np.zeros(n_modes, dtype=np.float64)

    n_high = max(0, n_modes - 2)
    if n_high > 0:
        high = rng.standard_normal(n_high)
        if cfg.aberration_model == "kolmogorov":
            high = _correlated_high_order(high, cfg)
        rms = float(np.sqrt(np.mean(high * high)))
        target = float(rng.uniform(*cfg.rms_rad))
        if rms > 0.0:
            coeffs[2:] = high * (target / rms)

    if n_modes >= 2:
        magnitude = float(rng.uniform(*cfg.tilt_px))
        azimuth = float(rng.uniform(0.0, 2.0 * math.pi))
        offset = np.array(
            [magnitude * math.cos(azimuth), magnitude * math.sin(azimuth)],
            dtype=np.float64,
        )
        coeffs[:2] = sim.tilt_coeffs_for_offset_px(offset)

    jitter = float(cfg.illum_jitter_frac) * float(sim.cfg.pupil_radius_px)
    radius = jitter * math.sqrt(float(rng.uniform(0.0, 1.0)))
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    center = (radius * math.cos(angle), radius * math.sin(angle))
    width_scale = float(rng.uniform(*cfg.illum_width))

    px_per_rad = np.array(
        [
            sim.offset_px_for_tilt_coeffs([1.0, 0.0])[0],
            sim.offset_px_for_tilt_coeffs([0.0, 1.0])[1],
        ],
        dtype=np.float64,
    )
    state = AberrationState(
        coeffs=coeffs,
        illum_center_px=(float(center[0]), float(center[1])),
        illum_width_scale=width_scale,
        px_per_rad=px_per_rad,
    )
    logger.debug(
        "drew aberration: rms={:.3f} rad tilt={} px illum=({:+.2f},{:+.2f}) samples "
        "width_scale={:.3f}",
        float(np.sqrt(np.mean(coeffs[2:] ** 2))) if n_high else 0.0,
        np.round(state.tilt_px, 3).tolist(),
        state.illum_center_px[0],
        state.illum_center_px[1],
        width_scale,
    )
    return state


def write_aberration(
    sim: Sim2f, state: AberrationState
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a state into the two tensors :meth:`Sim2f.field` accepts.

    Args:
        sim: The bench.
        state: Drawn or hand-built state.

    Returns:
        ``(aberration, illumination)`` with shapes ``(1, grid, grid)`` radians and
        ``(1, 1, grid, grid)`` amplitude, both on the bench's device/dtype.
    """
    coeffs = np.asarray(state.coeffs, dtype=np.float64).reshape(1, -1)
    aberration = sim.aberration(coeffs)
    illumination = sim.illumination(state.illum_center_px, state.illum_width_scale)
    return aberration, illumination
