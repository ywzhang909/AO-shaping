"""Closed-loop SLM far-field beam-shaping simulation bench.

This module provides a self-contained, hardware-free simulation of **single
phase-only SLM** far-field beam shaping (the 2f Fourier model):

    gaussian input field  ->  SLM phase exp(1j*phi)  ->  Fraunhofer far field
    ->  intensity on the (zero-padded) camera grid

It re-uses ``beam_backend`` only for the input beam (``gaussian_pupil`` /
``make_beam_config``); the focal plane is computed directly as a zero-padded
FFT. It provides:

* objective-function / quality metrics (PIB, efficiency, uniformity/CV,
  Strehl, overlap, zero-order fraction, structure similarity) used as the
  *objective functions* reported in the paper survey, and
* reference *phase generators* / optimization loops that reproduce, at a
  simulation level, the representative methods found in the literature
  (Gerchberg-Saxton, IFTA, differentiable gradient descent, SPGD sensorless
  black-box, Zernike-basis SPGD, and an analytic single-plate "amplitude"
  target).

The intent is to let a single, reproducible script exercise many beam-shaping
methods on one optical model and compare them with identical metrics.

All functions are pure numpy/torch (no hardware). The torch path is optional
(``_TORCH_AVAILABLE``); methods fall back to numpy when torch is absent.

Note on the physical model
--------------------------
The 2f Fourier bench (SLM front focus -> f-lens -> camera back focus) maps the
SLM pupil field to its **Fourier transform** (the lens phase + propagation to
``z=f`` equals a Fraunhofer transform up to a global phase). The pupil is
therefore zero-padded to ``far_field_size`` before the FFT so the focal plane is
properly sampled: a same-size FFT samples it at ``w_far/dx_far =
aperture/(pi*w0) = 3.5/pi = 1.11`` px per waist radius -- a constant of the
model independent of ``n_grid`` -- which aliases the lens-phase-modulated pupil
into a lattice of dots (see ``far_field_padding``). The zero-order spot sits at
the global intensity maximum (located by ``argmax``, never by geometry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ao_shaping.drivers.sim.beam_backend import (
    BeamSimConfig,
    gaussian_pupil,
    make_beam_config,
    turbulence_phase,
)

try:
    import torch
    import torch.nn.functional as F  # noqa: F401

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShapingBenchConfig:
    """Parameters for the 2f-Fourier SLM shaping bench.

    Attributes:
        n_grid: SLM / simulation grid edge length (square).
        aperture_size: SLM aperture width in metres.
        wavelength: laser wavelength in metres.
        focal_length: lens focal length in metres (sets the 2f scale).
        cn2: atmospheric turbulence strength (0 => no turbulence).
        l_max: turbulence outer scale (m).
        l_min: turbulence inner scale (m).
        target_side_px: target square/region side length in **far-field
            (camera) pixels**, i.e. pixels of the zero-padded far-field grid
            (``far_field_size``), not of the pupil grid.
        target_kind: "square" | "circle" | "gaussian" | "annulus".
        zero_order_margin_px: guard radius (far-field px) around the zero-order
            spot when reporting its power fraction (phase-only SLM keeps the
            undiffracted 0th order at the pattern center).
        far_field_padding: integer zero-padding factor applied to the pupil
            before the Fraunhofer FFT, i.e. the far-field grid is
            ``n_grid * far_field_padding``.  Without padding a same-size FFT
            samples the focal plane at ``w_far/dx_far = aperture/(pi*w0) =
            3.5/pi = 1.11`` px per waist radius *independently of n_grid*,
            which cannot represent a focused spot (it aliases into a lattice of
            dots).  8x gives ~8.9 px/waist (enough for aberration morphology);
            16x matches the repo house standard (``SimPibSystem`` and
            ``generate_zernike_farfield_sim_report`` both use 512 -> 8192).
        seed: RNG seed for reproducibility.
    """

    n_grid: int = 256
    aperture_size: float = 12e-3  # 12 mm SLM aperture
    wavelength: float = 532e-9  # 532 nm
    focal_length: float = 0.125  # 125 mm lens
    cn2: float = 0.0  # 0 => clean (no turbulence) baseline
    l_max: float = 0.1
    l_min: float = 1e-3
    target_side_px: int = 60
    target_kind: str = "square"
    zero_order_margin_px: float = 10.0
    far_field_padding: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.far_field_padding < 1:
            raise ValueError(
                f"far_field_padding must be >= 1, got {self.far_field_padding}"
            )

    @property
    def far_field_size(self) -> int:
        """Edge length of the zero-padded far-field (camera) grid."""
        return self.n_grid * self.far_field_padding

    @property
    def far_field_pixel_size(self) -> float:
        """Far-field (camera) pixel pitch in metres.

        For a Fraunhofer FFT zero-padded to ``M = n_grid * far_field_padding``
        the focal-plane pitch is ``lambda*f/(M*dx_pupil) =
        lambda*f/(far_field_padding*aperture_size)`` -- independent of n_grid.
        """
        return self.wavelength * self.focal_length / (
            self.aperture_size * self.far_field_padding
        )

    def make_beam_config(self) -> BeamSimConfig:
        return make_beam_config(
            n_grid=self.n_grid,
            aperture_size=self.aperture_size,
            wavelength=self.wavelength,
            cn2=self.cn2,
            l_max=self.l_max,
            l_min=self.l_min,
            propagation_distance=self.focal_length,
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ShapingResult:
    """Outcome of a single shaping run on the bench."""

    method: str
    phase: np.ndarray
    intensity: np.ndarray
    metrics: dict[str, float]
    n_iters: int
    history: list[dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Forward model (2f Fourier)
# ---------------------------------------------------------------------------
def _pad_centred(arr: np.ndarray, cfg: ShapingBenchConfig) -> np.ndarray:
    """Zero-pad a square pupil-grid array to the far-field grid, centred.

    The pupil occupies the central ``n_grid x n_grid`` block; the padding
    region is the (dark) area outside the SLM aperture.
    """
    n = arr.shape[0]
    m = cfg.far_field_size
    if m <= n:
        return arr
    out = np.zeros((m, m), dtype=arr.dtype)
    start = (m - n) // 2
    out[start : start + n, start : start + n] = arr
    return out


def _fraunhofer_intensity(field: np.ndarray, cfg: ShapingBenchConfig) -> np.ndarray:
    """Zero-padded Fraunhofer (focal-plane) intensity of a pupil field.

    The 2f Fourier bench maps the SLM pupil field to its Fourier transform
    (the lens phase + propagation to ``z=f`` is exactly this, up to a global
    phase).  The pupil is centred and **zero-padded** to ``far_field_size``
    before the FFT so the focal plane is properly oversampled; an un-padded
    same-size FFT samples the focal plane at only ~1.1 px per waist radius
    (a model constant, independent of ``n_grid``) and aliases into a lattice.

    Transform convention: ``F = fftshift(fft2(ifftshift(f)))`` so that a
    centred pupil maps to a centred far field, matching
    ``SimPibSystem.far_field`` and the repo ``beam_simulation._ft`` helper.

    Args:
        field: Pupil field on the ``n_grid`` grid (complex).
        cfg: Bench config (supplies the padding factor).

    Returns:
        Non-negative intensity on the ``far_field_size`` grid (unnormalised).
    """
    padded = _pad_centred(field, cfg)
    focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    return np.abs(focal) ** 2


def forward_intensity(phase: np.ndarray, cfg: ShapingBenchConfig) -> np.ndarray:
    """Propagate a phase-only SLM pattern to the far field; return intensity.

    Gaussian input field times ``exp(1j*phase)`` (phase-only SLM) propagated to
    the focal plane by a **zero-padded** Fraunhofer FFT. Returns a
    non-negative intensity array on the ``far_field_size`` grid (unnormalised).
    """
    beam_cfg = cfg.make_beam_config()
    field = gaussian_pupil(beam_cfg).astype(np.complex128) * np.exp(1j * np.asarray(phase))
    return _fraunhofer_intensity(field, cfg)


def make_target(cfg: ShapingBenchConfig) -> np.ndarray:
    """Build the normalised far-field target pattern (sum=1) on the camera grid.

    Target is centred on the **far-field (padded)** grid; its size is
    ``cfg.target_side_px`` far-field pixels so the objective is scale-stable.
    """
    n = cfg.far_field_size
    y, x = np.ogrid[:n, :n]
    cx, cy = n // 2, n // 2
    side = int(cfg.target_side_px)
    target = np.zeros((n, n))
    if cfg.target_kind == "square":
        target[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2] = 1.0
    elif cfg.target_kind == "circle":
        target[(x - cx) ** 2 + (y - cy) ** 2 <= (side / 2) ** 2] = 1.0
    elif cfg.target_kind == "gaussian":
        s = side / 4
        target = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2))
    elif cfg.target_kind == "annulus":
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        target[(r >= side / 2 - side / 4) & (r <= side / 2 + side / 4)] = 1.0
    else:
        raise ValueError(f"unknown target_kind {cfg.target_kind!r}")
    return target / target.sum()


# ---------------------------------------------------------------------------
# Objective functions / quality metrics
# ---------------------------------------------------------------------------
def _rolled_support(target: np.ndarray, center: np.ndarray | None) -> np.ndarray:
    """Boolean target support, rolled so its centre sits on ``center``.

    ``center`` is ``(col, row)`` (the ``np.unravel_index(...)[::-1]`` convention
    used by every caller). Rows (axis 0) are shifted by the *row* offset and
    columns (axis 1) by the *column* offset -- mixing these up transposes the
    support for off-axis beams, which silently lets an optimizer chase a
    support that no longer covers the beam. When ``center`` is None the support
    is used as-is (grid-centred). Callers must use the *same* centre for every
    metric so the bucket consistently follows the beam (matching the hardware
    closed-loop ``argmax`` rule).
    """
    sup = target > 0
    if center is not None:
        row_shift = int(round(center[1] - target.shape[0] // 2))
        col_shift = int(round(center[0] - target.shape[1] // 2))
        sup = np.roll(sup, (row_shift, col_shift), axis=(0, 1))
    return sup


def power_in_bucket(
    intensity: np.ndarray,
    target: np.ndarray,
    *,
    center: np.ndarray | None = None,
) -> float:
    """Power-in-bucket: fraction of total power inside the target region.

    The target region is taken from ``target`` (its support). If ``center`` is
    given the target is re-centred there (the target follows the measured
    zero-order / beam centroid), mirroring the hardware closed-loop.
    """
    total = intensity.sum()
    if total <= 0:
        return 0.0
    return float(intensity[_rolled_support(target, center)].sum() / total)


def efficiency(
    intensity: np.ndarray,
    target: np.ndarray,
    *,
    center: np.ndarray | None = None,
) -> float:
    """Encircled / bucket energy = power in the target support."""
    return power_in_bucket(intensity, target, center=center)


def uniformity_cv(
    intensity: np.ndarray,
    target: np.ndarray,
    *,
    center: np.ndarray | None = None,
) -> float:
    """Coefficient of variation of intensity *within* the target support.

    Lower is better (0 = perfectly uniform). Returns +inf if the target
    support is empty of power. The support is rolled to ``center`` exactly as
    in :func:`power_in_bucket`, so PIB and CV always describe the same region.
    """
    vals = intensity[_rolled_support(target, center)]
    if vals.size == 0 or vals.mean() <= 0:
        return float("inf")
    return float(vals.std() / vals.mean())


def strehl(intensity: np.ndarray, target: np.ndarray) -> float:
    """Normalized overlap (Strehl-like) between simulated and target patterns.

    Uses the cosine similarity (normalized inner product) of the two
    energy-distributed patterns; 1.0 = perfect match.
    """
    a = intensity
    b = target
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def zero_order_fraction(
    intensity: np.ndarray,
    center: np.ndarray,
    *,
    zero_order_margin_px: float = 10.0,
) -> float:
    """Fraction of total power inside the central zero-order guard band.

    Args:
        intensity: Far-field intensity on the (padded) camera grid.
        center: Zero-order location in ``(row, col)`` of the intensity array.
        zero_order_margin_px: Guard radius in far-field (camera) pixels. The
            default is ~1 Airy radius (10 px at the 8x padding); scale it with
            ``cfg.zero_order_margin_px`` for other paddings.
    """
    n = intensity.shape[0]
    y, x = np.ogrid[:n, :n]
    r = np.sqrt((x - int(center[0])) ** 2 + (y - int(center[1])) ** 2)
    m = r <= zero_order_margin_px
    total = intensity.sum()
    return float(intensity[m].sum() / total) if total > 0 else 0.0


def compute_metrics(
    intensity: np.ndarray,
    target: np.ndarray,
    *,
    center: np.ndarray | None = None,
    zero_order_margin_px: float = 10.0,
) -> dict[str, float]:
    """Compute the full objective-function suite used in the survey.

    ``center`` is the target-box centre: ``None`` (default) uses the target's
    own grid-centred support, which is correct for this **centred** simulation
    bench. Do NOT default it to the intensity ``argmax``: for a speckle-like
    field the global maximum hops between near-equal grains under a ~1e-3 model
    perturbation, so an argmax-rolled box makes PIB/CV discontinuous and lets an
    optimizer chase a box that does not cover the beam. The zero-order guard is
    still centred on the intensity peak (that is what it measures).
    """
    peak = np.unravel_index(np.argmax(intensity), intensity.shape)[::-1]
    return {
        "PIB": power_in_bucket(intensity, target, center=center),
        "efficiency": efficiency(intensity, target, center=center),
        "CV": uniformity_cv(intensity, target, center=center),
        "Strehl": strehl(intensity, target),
        "zero_order": zero_order_fraction(
            intensity, peak, zero_order_margin_px=zero_order_margin_px
        ),
    }


def composite_from_pib_cv(
    pib: float,
    cv: float,
    *,
    w_pib: float = 0.5,
    w_unif: float = 0.5,
) -> float:
    """Composite score formula, shared by the numpy and torch objective paths.

    Uniformity is scored as ``1 / (1 + cv)`` rather than a clipped linear map.
    A clipped term ``1 - min(cv/cv_ref, 1)`` saturates to zero for every
    achievable flat-top CV (CV within a target box is not scale-free, so no
    single ``cv_ref`` works), which silently reduces the objective to pure
    bucket energy and rewards concentrating light over flattening it.
    ``1/(1+cv)`` is monotone and never saturates, so it rewards every
    uniformity gain with no threshold to tune.
    """
    return w_pib * pib + w_unif * (1.0 / (1.0 + cv))


def composite_score(
    m: dict[str, float],
    *,
    w_pib: float = 0.5,
    w_unif: float = 0.5,
) -> float:
    """Composite scalar score to *maximize* (the SPGD / differentiable reward).

    Combines bucket energy with a uniformity term so the optimizer does not
    merely dump energy into the box while leaving it ragged (a known failure
    mode of CV-only objectives).
    """
    pib = m.get("PIB", 0.0)
    cv = m.get("CV", float("inf"))
    return composite_from_pib_cv(pib, cv, w_pib=w_pib, w_unif=w_unif)


# ---------------------------------------------------------------------------
# Method 1: Gerchberg-Saxton (amplitude <-> phase constraints)
# ---------------------------------------------------------------------------
def gs_shape(
    cfg: ShapingBenchConfig,
    *,
    n_iters: int = 200,
    relax: float = 1.0,
    seed: int | None = None,
    verbose: bool = False,
    return_phase_only: bool = True,
    base_phase: np.ndarray | None = None,
) -> ShapingResult:
    """Gerchberg-Saxton shaping of a single phase-only SLM.

    Amplitude constraint = sqrt(target) in the far field, phase constraint =
    arbitrary (SLM plane). The final phase is the SLM pattern.

    GS runs on the **padded far-field grid** (the pupil is zero-padded to
    ``far_field_size``), so the forward/backward FFT pair is an exact,
    correctly-sampled transform and the returned intensity matches
    ``make_target``'s grid. The pupil-support constraint keeps only the central
    ``n_grid`` block (light exists only inside the SLM aperture).

    Args:
        base_phase: Fixed pupil phase (raw radians, shape ``(n_grid, n_grid)``)
            that the GS solution is added to, e.g. a measured aberration
            pre-correction ``-Z_est``. It is applied *inside* the pupil
            constraint, so GS shapes the corrected pupil and the returned phase
            is the total ``base_phase + delta``.
    """
    beam_cfg = cfg.make_beam_config()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    target = make_target(cfg)
    amp_target = np.sqrt(target)
    amp_slm = _pad_centred(np.abs(gaussian_pupil(beam_cfg)), cfg).astype(np.float64)
    if base_phase is None:
        base = np.zeros(amp_slm.shape, dtype=np.float64)
    else:
        base = _pad_centred(
            np.asarray(base_phase, dtype=np.float64), cfg
        )
    # Initial field: gaussian input in the SLM plane
    field = amp_slm * np.exp(1j * (base + rng.normal(0, 0.1, size=amp_slm.shape)))

    history = []
    for i in range(n_iters):
        # SLM -> far field
        ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
        # amplitude constraint in far field
        ff = ff * (amp_target / (np.abs(ff) + 1e-12)) * relax
        # relax back toward the target amplitude
        ff = ff * (1 - relax) + amp_target * np.exp(1j * np.angle(ff)) * relax
        # far field -> SLM plane
        field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ff)))
        # phase-only constraint in SLM plane (keep gaussian amplitude, set phase)
        field = amp_slm * np.exp(1j * np.angle(field))
        if verbose and i % 20 == 0:
            inten = _fraunhofer_intensity(field, cfg)
            inten = inten / inten.max()
            center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
            m = compute_metrics(inten, target, center=center)
            history.append({"iter": i, **m})
    inten = _fraunhofer_intensity(field, cfg)
    inten = inten / (inten.sum() + 1e-12)
    center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
    metrics = compute_metrics(inten, target, center=center)
    n = cfg.n_grid
    pad = (cfg.far_field_size - n) // 2
    pupil_phase = np.angle(field)[pad : pad + n, pad : pad + n]
    return ShapingResult(
        method="gerchberg_saxton",
        phase=pupil_phase,
        intensity=inten,
        metrics=metrics,
        history=history,
        n_iters=n_iters,
    )


# ---------------------------------------------------------------------------
# Method 2: differentiable gradient descent (torch, optional)
# ---------------------------------------------------------------------------
def differentiable_shape(
    cfg: ShapingBenchConfig,
    *,
    n_iters: int = 300,
    lr: float = 0.05,
    seed: int | None = None,
) -> ShapingResult:
    """Differentiable far-field shaping: gradient descent on the SLM phase.

    Loss = 1 - overlap(target, sim) + lambda * (1 - PIB). Requires torch.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("differentiable_shape requires torch")
    beam_cfg = cfg.make_beam_config()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    target = make_target(cfg)
    tgt = torch.as_tensor(target, dtype=torch.float64).double()
    n = cfg.n_grid
    m = cfg.far_field_size
    pad = (m - n) // 2
    init_phase = torch.as_tensor(
        rng.normal(0, 0.1, size=(n, n)), dtype=torch.float64
    ).double()
    param = torch.nn.Parameter(init_phase)
    opt = torch.optim.Adam([param], lr=lr)

    in_amp = torch.as_tensor(
        np.abs(gaussian_pupil(beam_cfg)), dtype=torch.float64
    ).double()

    def intensity(phase: Any) -> Any:
        field = in_amp * torch.exp(1j * phase)
        if m > n:
            padded = torch.zeros((m, m), dtype=torch.complex128)
            padded[pad : pad + n, pad : pad + n] = field
            field = padded
        focal = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field)))
        return focal.real**2 + focal.imag**2

    history = []
    for i in range(n_iters):
        opt.zero_grad()
        inten = intensity(param)
        inten = inten / inten.sum()
        # target overlap (cosine-ish, normalized)
        a = inten - inten.mean()
        b = tgt - tgt.mean()
        overlap = (a * b).sum() / (a.norm() * b.norm() + 1e-12)
        # PIB: power in target support
        sup = (tgt > 0).double()
        pib = (inten * sup).sum() / inten.sum()
        loss = -overlap - 0.5 * pib
        loss.backward()
        opt.step()
        if i % 50 == 0:
            with torch.no_grad():
                inten_np = intensity(param).numpy()
                inten_np = inten_np / inten_np.max()
                center = np.unravel_index(np.argmax(inten_np), inten_np.shape)[::-1]
                metrics_hist = compute_metrics(inten_np, target, center=center)
                history.append({"iter": i, **metrics_hist})
    with torch.no_grad():
        inten_np = intensity(param).numpy()
        inten_np = inten_np / (inten_np.sum() + 1e-12)
    center = np.unravel_index(np.argmax(inten_np), inten_np.shape)[::-1]
    metrics = compute_metrics(inten_np, target, center=center)
    return ShapingResult(
        method="differentiable",
        phase=param.detach().numpy(),
        intensity=inten_np,
        metrics=metrics,
        history=history,
        n_iters=n_iters,
    )


# ---------------------------------------------------------------------------
# Method 3: SPGD (sensorless black-box gradient) on the SLM phase
# ---------------------------------------------------------------------------
def spgd_shape(
    cfg: ShapingBenchConfig,
    *,
    n_iters: int = 600,
    delta: float = 0.1,
    lr: float = 0.02,
    seed: int | None = None,
    dim: int | None = None,
) -> ShapingResult:
    """Sensorless SPGD on a low-dimensional (Zernike / coarse) phase basis.

    The SLM phase is parameterised as a small number of freeform coefficients
    (a coarse grid) so that random-gradient search is tractable in the sim.
    The reward is the composite score (PIB + uniformity).
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    target = make_target(cfg)
    d = dim or 16  # 16x16 freeform basis
    n_par = d * d
    phase_flat = rng.normal(0, 0.05, size=n_par)

    def upsample(vec: np.ndarray) -> np.ndarray:
        """Map the d×d freeform coefficient grid to the full SLM phase (kron upsample)."""
        ph = np.kron(vec.reshape(d, d), np.ones((cfg.n_grid // d, cfg.n_grid // d)))
        if ph.shape != (cfg.n_grid, cfg.n_grid):
            ph = ph[: cfg.n_grid, : cfg.n_grid]
        return ph

    def eval_score(vec: np.ndarray) -> float:
        inten = forward_intensity(upsample(vec), cfg)
        inten = inten / inten.max()
        center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
        m = compute_metrics(inten, target, center=center)
        return composite_score(m)

    score = eval_score(phase_flat)
    g = np.zeros(n_par)
    history = [
        {"iter": 0, "score": score, "PIB": 0, "CV": float("inf")}
    ]
    for i in range(1, n_iters):
        sgn = rng.choice([-1, 1], size=n_par)
        cand = phase_flat + delta * sgn
        s2 = eval_score(cand)
        g = 0.95 * g + 0.05 * sgn * (s2 - score)
        phase_flat = phase_flat + lr * g
        score = eval_score(phase_flat)
        if i % 50 == 0:
            inten = forward_intensity(upsample(phase_flat), cfg)
            inten = inten / inten.max()
            center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
            m = compute_metrics(inten, target, center=center)
            history.append({"iter": i, "score": score, "PIB": m["PIB"], "CV": m["CV"]})
    ph = upsample(phase_flat)
    inten = forward_intensity(ph, cfg)
    inten = inten / (inten.sum() + 1e-12)
    center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
    metrics = compute_metrics(inten, target, center=center)
    metrics["score"] = composite_score(metrics)
    return ShapingResult(
        method="spgd_freeform",
        phase=ph,
        intensity=inten,
        metrics=metrics,
        history=history,
        n_iters=n_iters,
    )


# ---------------------------------------------------------------------------
# Method 4: analytic "amplitude" target (single-plate amplitude-only baseline)
# ---------------------------------------------------------------------------
def analytic_amplitude_target(cfg: ShapingBenchConfig) -> ShapingResult:
    """Trivial baseline: the target itself (amplitude shaping, no phase).

    Serves as a reference for what a pure-amplitude (non-phase-only) SLM would
    produce -- it bounds the achievable quality for a phase-only device.
    """
    target = make_target(cfg)
    metrics = compute_metrics(target, target)
    return ShapingResult(
        method="analytic_amplitude_baseline",
        phase=np.zeros((cfg.n_grid, cfg.n_grid)),
        intensity=target,
        metrics=metrics,
        n_iters=0,
    )


__all__ = [
    "ShapingBenchConfig",
    "ShapingResult",
    "forward_intensity",
    "make_target",
    "compute_metrics",
    "composite_score",
    "composite_from_pib_cv",
    "gs_shape",
    "differentiable_shape",
    "spgd_shape",
    "analytic_amplitude_target",
]
