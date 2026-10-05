"""Inverse (beam-shaping) design toolkit: independent evaluator + design operators.

This is the reusable core behind ``report/loss_defects/PROCESS.md``. It started as
copy-pasted helpers in a dozen exploratory scripts; it is now the single
implementation, and those scripts are thin CLIs over it.

**A library module — no CLI, no ``__main__``.** Run the studies through the
``scripts/inverse_*.py`` wrappers.

Why an *independent* evaluator
------------------------------
The learned model must not grade its own work: its absolute score is uncorrelated
with real far-field quality across restarts (PROCESS.md attempt 7), and its gradients
are anti-aligned (attempt 8). So every design produced here is scored by
:class:`~ao_shaping.drivers.sim.slm_pib_sim.SimPibSystem` -- a separate numpy
``|FFT(pupil * e^{i phi})|^2`` with its own Gaussian illumination -- using the
canonical :func:`~ao_shaping.utils.image.target.metrics.rms_pib_terms`.

Two calibration facts that are easy to get wrong, both learned the hard way:

* The simulator is calibrated for the real 1920x1200 bench panel. A 64x64 ``SimPibSystem``
  is out of regime and returns **NaN**, so a small pupil map must be *embedded* into
  the calibrated disc (:func:`embed_phase`).
* :func:`sim_far_field` masks the pupil to the inscribed circle, because outside the
  disc there is **no light**, not zero phase.

Both are why :func:`score_phase` raises rather than returning a plausible-looking
number when the geometry is wrong.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton
from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem
from ao_shaping.utils.image.target.metrics import rms_pib_terms
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator, fit_zernike
from ml.zernike.losses import LossConfig, composite_loss, roi_mask
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel

__all__ = [
    "GRID", "N_MAX", "PADDING", "SIZE_FRAC", "ASPECT", "COARSE_GRID",
    "PANEL_H", "PANEL_W", "DISC_RADIUS", "SIM_PADDING", "SIM_WINDOW",
    "PIXEL_UM", "GS_ITERS",
    "square_amplitude", "target_tensor", "inscribed_mask", "roi",
    "embed_phase", "centre_crop", "sim_far_field", "score_phase",
    "coefficients_to_phase", "score_coefficients",
    "build_model", "model_far_field",
    "gs_phase", "gs_coefficients",
    "gradient_freeform", "gradient_zernike",
    "usable_records", "load_corpus_sample",
    "pearson", "spearman",
]

#: Discretisation shared by every study in PROCESS.md.
GRID, N_MAX, PADDING = 64, 15, 12
SIZE_FRAC, ASPECT = 0.375, 4.0 / 3.0
#: ``slm_gs_refine``'s default ``--phase-grid``.
COARSE_GRID = 24

#: The real bench panel, used only to derive the illumination *ratio*.
BENCH_PANEL_H, BENCH_PANEL_W = 1200, 1920
DISC_RADIUS = 450

#: The evaluator runs the 64x64 pupil **directly**, not embedded in the 1200x1920
#: panel. Two reasons, both learned the hard way:
#:
#: 1. Embedding makes the far field cover the *panel's* angular range, so the beam
#:    core lands on ~6 px of a 64 px crop. A design targeting a 24 px square in its own
#:    grid then aims at something 4x larger than the measurement window, and every
#:    comparison reads as noise.
#: 2. A 64x64 `SimPibSystem` with a *narrow* waist returns NaN (out of calibrated
#:    regime). Scaling the waist with the grid keeps it finite.
#:
#: Running the pupil directly also makes the evaluator's far-field grid identical to
#: the grid Gerchberg-Saxton produces, so the two are comparable pixel-for-pixel.
PANEL_H, PANEL_W = GRID, GRID
SIM_PADDING, SIM_WINDOW = 1, GRID
#: Waist scaled from the bench ratio (450 / (1920/2)) onto this grid.
BEAM_W0 = DISC_RADIUS * GRID / (BENCH_PANEL_W / 2)

PIXEL_UM = 8.0
GS_ITERS = 40


# ----------------------------------------------------------------------
# Targets and masks
# ----------------------------------------------------------------------
def square_amplitude(
    grid: int = GRID, size_frac: float = SIZE_FRAC, aspect: float = ASPECT
) -> np.ndarray:
    """Binary square target amplitude, centred, ``size_frac * grid`` on the short side."""
    amp = np.zeros((grid, grid), dtype=np.float64)
    half_w = int(size_frac * grid * aspect / 2)
    half_h = int(size_frac * grid / 2)
    c = grid // 2
    amp[c - half_h : c + half_h, c - half_w : c + half_w] = 1.0
    return amp


def target_tensor(
    grid: int = GRID, size_frac: float = SIZE_FRAC, aspect: float = ASPECT
) -> torch.Tensor:
    """The square target as a ``(1, 1, grid, grid)`` tensor, ready for a loss."""
    return torch.as_tensor(
        square_amplitude(grid, size_frac, aspect), dtype=torch.float32
    )[None, None]


def inscribed_mask(grid: int = GRID) -> torch.Tensor:
    """``(1, 1, grid, grid)`` unit amplitude inside the inscribed circle, zero outside.

    Required when pushing a *phase* through :meth:`ZernikeAmpModel.forward`: that method
    uses the input phasor raw, so an unmasked phase-only pupil would carry unit
    amplitude outside the illuminated disc while the simulator lights a finite disc.
    """
    radius = grid / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(grid, dtype=torch.float32) + 0.5,
        torch.arange(grid, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    inside = ((ys - radius) ** 2 + (xs - radius) ** 2) <= radius * radius
    return inside.float()[None, None]


def roi(
    grid: int = GRID, size_frac: float = SIZE_FRAC, aspect: float = ASPECT
) -> torch.Tensor:
    """ROI mask for the loss, matching the evaluator's box exactly."""
    return roi_mask(
        (grid, grid), (grid / 2, grid / 2), "rectangle", size_frac * grid, aspect
    )


# ----------------------------------------------------------------------
# Independent evaluator
# ----------------------------------------------------------------------
def embed_phase(phase: np.ndarray) -> np.ndarray:
    """Mask a ``(grid, grid)`` pupil phase to the illuminated disc.

    Outside the inscribed circle there is no light, so the amplitude there is zero --
    a bare phase would imply unit amplitude there and disagree with the simulator
    off-aperture.
    """
    pupil = np.asarray(phase, dtype=np.float64)
    if pupil.shape != (GRID, GRID):
        raise ValueError(f"phase must be ({GRID}, {GRID}), got {pupil.shape}")
    radius = GRID / 2.0
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    disc = ((ys + 0.5 - radius) ** 2 + (xs + 0.5 - radius) ** 2) <= radius * radius
    return np.where(disc, pupil, 0.0)


def centre_crop(image: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Crop the ``grid x grid`` window **centred** on the array.

    The 0-order of a ``SimPibSystem`` far field sits at the array centre, so a
    top-left crop (``image[:grid, :grid]``) takes a corner ~1000x dimmer than the beam
    and every ROI metric computed on it describes off-axis sidelobes rather than the
    spot. That bug ran through the whole first pass of this study and was only caught by
    rendering a pred-vs-true image, where the "independent" panel came out blank.
    """
    image = np.asarray(image)
    if image.shape[0] < grid or image.shape[1] < grid:
        raise ValueError(f"image {image.shape} smaller than grid {grid}")
    if image.shape[0] == grid and image.shape[1] == grid:
        return image
    start_r = (image.shape[0] - grid) // 2
    start_c = (image.shape[1] - grid) // 2
    return image[start_r : start_r + grid, start_c : start_c + grid]


def sim_far_field(phase: np.ndarray, padding: int = SIM_PADDING) -> np.ndarray:
    """Independent simulator far field for a pupil phase, cropped to ``grid x grid``.

    The crop is **centred on the 0-order** (see :func:`centre_crop`).

    Args:
        phase: ``(grid, grid)`` pupil phase in radians.
        padding: Far-field zero-padding factor. The centre-cropped ``grid x grid`` output
            of a padding-``p`` FFT covers ``1/p`` of the padded angular extent, so this
            sets the **angular scale** of the result. Pass the same value the model under
            test was built with, or the two are not comparable pixel-for-pixel. The
            default matches the scale Gerchberg-Saxton designs on.

    Raises:
        RuntimeError: If the far field is non-finite, which means the geometry is out
            of the simulator's calibrated regime. A silently-zero metric is worse than
            an error -- that failure mode is exactly how the first evaluator in
            PROCESS.md attempt 1 produced all-zero results for every configuration.
    """
    system = SimPibSystem(
        slm_shape=(PANEL_H, PANEL_W),
        ccd_res=(512, 512),
        beam_w0=float(BEAM_W0),
        noise_adu=0.0,
        seed=0,
        far_field_padding=int(padding),
        far_field_window=SIM_WINDOW,
    )
    system.set_phase_rad(embed_phase(phase))
    full = np.asarray(system.far_field(), dtype=np.float64)
    if not np.all(np.isfinite(full)):
        raise RuntimeError(
            "simulator far field is non-finite -- geometry out of regime. A 64x64 "
            "SimPibSystem returns NaN; embed the pupil in the calibrated panel instead."
        )
    return centre_crop(full, GRID)


def score_phase(
    phase: np.ndarray,
    size_frac: float = SIZE_FRAC,
    aspect: float = ASPECT,
    padding: int = SIM_PADDING,
) -> float:
    """``pib + uniformity`` on the independent simulator (higher is better).

    ``padding`` is forwarded to :func:`sim_far_field`; see there for why it sets the
    angular scale.
    """
    pib, uni = rms_pib_terms(
        sim_far_field(phase, padding),
        (GRID / 2, GRID / 2),
        "rectangle",
        size_frac * GRID,
        aspect,
    )
    return float(pib + uni)


def coefficients_to_phase(coefficients: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    """Zernike coefficients (non-piston, Noll order) -> ``(grid, grid)`` pupil phase.

    ``nan_to_num`` is not cosmetic: ``ZernikeGenerator`` returns **NaN outside the
    aperture disc**, and embedding that NaN into the panel makes the whole far field
    non-finite, which reads as "every metric is 0" rather than as an error.
    """
    generator = ZernikeGenerator((GRID, GRID), n_orders=n_max)
    phase = generator.generate_noll(np.asarray(coefficients, dtype=np.float64))
    return np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)


def score_coefficients(
    coefficients: np.ndarray,
    size_frac: float = SIZE_FRAC,
    aspect: float = ASPECT,
    n_max: int = N_MAX,
    padding: int = SIM_PADDING,
) -> float:
    """:func:`score_phase` for a Zernike coefficient vector."""
    return score_phase(
        coefficients_to_phase(coefficients, n_max), size_frac, aspect, padding
    )


# ----------------------------------------------------------------------
# Model and design operators
# ----------------------------------------------------------------------
def build_model(
    n_max: int = N_MAX, grid: int = GRID, padding: int = PADDING, seed: int = 0
) -> ZernikeAmpModel:
    """A deterministic ``ZernikeAmpModel`` with the study's geometry.

    The seed is fixed because ``coefficients`` initialises to zeros *deterministically* --
    ``torch.manual_seed`` cannot make two model instances differ, which is why every
    restart in these studies is drawn explicitly rather than seeded.
    """
    torch.manual_seed(seed)
    return ZernikeAmpModel(
        ZernikeAmpConfig(
            n_max=n_max, grid=grid, far_field_padding=padding, normalization="peak"
        )
    )


def model_far_field(model: ZernikeAmpModel, phase: torch.Tensor) -> torch.Tensor:
    """Learned-model far field for a freeform phase, aperture-masked.

    ``ZernikeAmpModel.forward`` validates only that its input is ``(B, 1, g, g)`` at the
    basis resolution -- ``measured = complex(phase_cos, phase_sin)`` *is* the input
    phasor -- so any phase, freeform included, goes straight through. No separate
    freeform-capable model is needed.
    """
    phasor = torch.polar(torch.ones_like(phase), phase) * inscribed_mask(phase.shape[-1])
    return model(phasor.real, phasor.imag)


def gs_phase(
    size_frac: float = SIZE_FRAC, aspect: float = ASPECT, iterations: int = GS_ITERS
) -> np.ndarray:
    """Canonical Fourier Gerchberg-Saxton pupil phase, used directly (no projection)."""
    result = gerchberg_saxton(
        np.ones((GRID, GRID), dtype=np.float64),
        square_amplitude(GRID, size_frac, aspect),
        iterations=iterations,
        cell_spacing=PIXEL_UM * 1e-6,
        propagation="fft",
    )
    return np.asarray(result.phase, dtype=np.float64)


def gs_coefficients(
    size_frac: float = SIZE_FRAC,
    aspect: float = ASPECT,
    n_max: int = N_MAX,
    iterations: int = GS_ITERS,
    init_phase: np.ndarray | None = None,
) -> np.ndarray:
    """GS pupil phase projected onto the Zernike basis (the *lossy* path).

    Piston is dropped with ``[1:]``: ``fit_zernike`` returns every term in Noll order
    including piston, while a model holds the ``K`` non-piston modes only.
    """
    phase = gs_phase(size_frac, aspect, iterations)
    if init_phase is not None:
        phase = phase + np.asarray(init_phase, dtype=np.float64)
    return fit_zernike(phase, n_max=n_max)[1:]


def gradient_freeform(
    init_coarse: np.ndarray,
    model: ZernikeAmpModel,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: LossConfig | None = None,
    steps: int = 60,
    lr: float = 0.02,
) -> np.ndarray:
    """Optimise a coarse freeform phase grid; returns the ``(grid, grid)`` phase.

    The optimisation variable is ``init_coarse`` at ``COARSE_GRID`` resolution,
    bilinearly upsampled to the model grid -- the ``slm_gs_refine`` ``phase-grid``
    parameterisation, which is what lets a low-DOF grid represent a square far field
    that smooth low-order Zernike cannot.
    """
    weights = weights or LossConfig(w_mse=1.0)
    coarse = torch.nn.Parameter(
        torch.as_tensor(np.asarray(init_coarse, np.float32))[None, None]
    )
    optimiser = torch.optim.AdamW([coarse], lr=lr)
    for _ in range(steps):
        optimiser.zero_grad()
        up = F.interpolate(
            coarse, size=(GRID, GRID), mode="bilinear", align_corners=False
        )
        loss = composite_loss(model_far_field(model, up), target, mask, weights)[
            "_mean_total"
        ]
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        up = F.interpolate(
            coarse, size=(GRID, GRID), mode="bilinear", align_corners=False
        )
    return up[0, 0].numpy().astype(np.float64)


def gradient_zernike(
    init_coefficients: np.ndarray,
    model: ZernikeAmpModel,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: LossConfig | None = None,
    steps: int = 60,
    lr: float = 0.02,
) -> np.ndarray:
    """Refine a Zernike coefficient vector against the learned model.

    Note this path reads only ``model.coefficients``: the fitted forward state is never
    consulted, so an unfitted and a fitted model give bit-identical results
    (PROCESS.md attempt 12).
    """
    weights = weights or LossConfig(w_mse=1.0)
    with torch.no_grad():
        model.coefficients.copy_(
            torch.as_tensor(np.asarray(init_coefficients, np.float32))
        )
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        optimiser.zero_grad()
        loss = composite_loss(
            model.correction_far_field(), target, mask, weights
        )["_mean_total"]
        loss.backward()
        optimiser.step()
    return model.coefficients_array()


# ----------------------------------------------------------------------
# Real corpus
# ----------------------------------------------------------------------
def usable_records(index) -> list:
    """Corpus records that can be materialised (a real, non-placeholder sample)."""
    return [r for r in index.records if r.key is not None and r.key >= 0]


def load_corpus_sample(
    index,
    position: int = 0,
    grid: int = GRID,
    family: str = "slm_zernike_shaping",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One real measured sample as ``(phase_cos, phase_sin, image)``, each ``(1,1,g,g)``.

    ``phase_cos``/``phase_sin`` are the measured pupil phasor, i.e. what
    :meth:`ZernikeAmpModel.forward` consumes; ``image`` is the measured far field. The
    pair is what a forward pred-vs-true panel compares.
    """
    from ml.hwdataset import MaterialiserConfig
    from ml.hwdataset.records import Materialiser

    usable = usable_records(index.filter(families=[family]))
    sample = Materialiser(
        config=MaterialiserConfig(grid=grid), use_cache=True
    ).materialise(usable[position])

    def _t(array) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array, dtype=np.float32))[None, None]

    return _t(sample.phase_cos), _t(sample.phase_sin), _t(sample.image)


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def pearson(a, b) -> float:
    """Pearson correlation; 0.0 when either input is constant."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a**2).sum() * (b**2).sum()))
    return float((a * b).sum() / denom) if denom else 0.0


def spearman(a, b) -> float:
    """Rank correlation via argsort-of-argsort (no scipy dependency).

    Prefer this over :func:`pearson` for the small-n sweeps in this study: those
    relationships contain one or two extreme cells, and Pearson is dominated by them
    while the ranks are not.

    A constant input yields ``0.0`` rather than ``+/-1``: ``argsort`` of a constant
    series still returns ``0..n-1``, so the degeneracy has to be detected before
    ranking.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom else 0.0
