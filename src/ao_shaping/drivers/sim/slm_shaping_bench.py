"""Closed-loop SLM far-field beam-shaping simulation bench.

This module provides a self-contained, hardware-free simulation of **single
phase-only SLM** far-field beam shaping (the 2f Fourier model):

    gaussian input field  ->  SLM phase exp(1j*phi)  ->  lens(f) + propagation
    ->  far-field (Fraunhofer) intensity on the camera grid

It re-uses the canonical beam-simulation backend
(``ao_shaping.drivers.sim.beam_backend``) for propagation, and provides:

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
``beam_backend.focal_plane`` implements ``lens(phase) + propagation(z=f)``, i.e.
the far-field (Fraunhofer) focal plane of a lens placed at the SLM plane. This
matches the experimental 2f Fourier bench described in ``docs`` (SLM front
focus -> f-lens -> camera back focus), where the camera grid is a
spatial-frequency grid and the zero-order spot sits at the global intensity
maximum (located by ``argmax``, never by geometry).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ao_shaping.drivers.sim.beam_backend import (
    BeamSimConfig,
    focal_plane,
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
        target_side_px: target square/region side length in *camera* pixels.
        target_kind: "square" | "circle" | "gaussian" | "annulus".
        zero_order_margin_px: guard band (px) around the zero-order spot when
            excluding it from the shaping region (phase-only SLM keeps the
            undiffracted 0th order at the pattern center).
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
    zero_order_margin_px: int = 8
    seed: int = 0

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
def forward_intensity(phase: np.ndarray, cfg: ShapingBenchConfig) -> np.ndarray:
    """Propagate a phase-only SLM pattern to the far field; return intensity.

    Gaussian input field times ``exp(1j*phase)`` (phase-only SLM) propagated to
    the focal plane. Returns a non-negative intensity array (unnormalised).
    """
    beam_cfg = cfg.make_beam_config()
    field = gaussian_pupil(beam_cfg).astype(np.complex128) * np.exp(1j * np.asarray(phase))
    ff = focal_plane(field, beam_cfg, focal_length=cfg.focal_length)
    return np.abs(ff) ** 2


def make_target(cfg: ShapingBenchConfig) -> np.ndarray:
    """Build the normalised far-field target pattern (sum=1) on the grid.

    Target is centred on the grid; its size is ``cfg.target_side_px`` camera
    pixels so the objective is scale-stable.
    """
    n = cfg.n_grid
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
def _zero_order_mask(cfg: ShapingBenchConfig, center: np.ndarray) -> np.ndarray:
    """Boolean mask of the zero-order exclusion region (False = valid region)."""
    n = cfg.n_grid
    y, x = np.ogrid[:n, :n]
    r = np.sqrt((x - int(center[1])) ** 2 + (y - int(center[0])) ** 2)
    return r > cfg.zero_order_margin_px


def _target_mask(cfg: ShapingBenchConfig) -> np.ndarray:
    n = cfg.n_grid
    half = n // 2
    s = cfg.target_side_px
    return (
        (np.arange(n) >= half - s // 2)
        & (np.arange(n) < half + s // 2)
    )[:, None] & (
        (np.arange(n) >= half - s // 2) & (np.arange(n) < half + s // 2)
    )[None, :]


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
    sup = target > 0
    if center is not None:
        dy, dx = int(round(center[0] - target.shape[0] // 2)), int(
            round(center[1] - target.shape[1] // 2)
        )
        sup = np.roll(sup, (dy, dx), axis=(0, 1))
    return float(intensity[sup].sum() / total)


def efficiency(intensity: np.ndarray, target: np.ndarray) -> float:
    """Encircled / bucket energy = power in the (centred) target support."""
    return power_in_bucket(intensity, target)


def uniformity_cv(intensity: np.ndarray, target: np.ndarray) -> float:
    """Coefficient of variation of intensity *within* the target support.

    Lower is better (1.0 = perfectly uniform). Returns +inf if the target
    support is empty of power.
    """
    sup = target > 0
    vals = intensity[sup]
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


def zero_order_fraction(intensity: np.ndarray, center: np.ndarray) -> float:
    """Fraction of total power inside the central zero-order guard band."""
    n = intensity.shape[0]
    y, x = np.ogrid[:n, :n]
    r = np.sqrt((x - int(center[1])) ** 2 + (y - int(center[0])) ** 2)
    m = r <= 6
    total = intensity.sum()
    return float(intensity[m].sum() / total) if total > 0 else 0.0


def compute_metrics(
    intensity: np.ndarray,
    target: np.ndarray,
    *,
    center: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the full objective-function suite used in the survey."""
    if center is None:
        center = np.unravel_index(np.argmax(intensity), intensity.shape)[::-1]
    return {
        "PIB": power_in_bucket(intensity, target, center=center),
        "efficiency": efficiency(intensity, target),
        "CV": uniformity_cv(intensity, target),
        "Strehl": strehl(intensity, target),
        "zero_order": zero_order_fraction(intensity, center),
    }


def composite_score(
    m: dict[str, float],
    *,
    w_pib: float = 0.5,
    w_unif: float = 0.5,
    cv_ref: float = 0.3,
) -> float:
    """Composite scalar score to *maximize* (the SPGD / differentiable reward).

    Combines bucket energy with a uniformity term so the optimizer does not
    merely dump energy into the box while leaving it ragged (a known failure
    mode of CV-only objectives).
    """
    pib = m.get("PIB", 0.0)
    cv = m.get("CV", float("inf"))
    cv_term = 1.0 - min(cv / max(cv_ref, 1e-9), 1.0)
    return w_pib * pib + w_unif * cv_term


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
) -> ShapingResult:
    """Gerchberg-Saxton shaping of a single phase-only SLM.

    Amplitude constraint = sqrt(target) in the far field, phase constraint =
    arbitrary (SLM plane). The final phase is the SLM pattern.
    """
    beam_cfg = cfg.make_beam_config()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    target = make_target(cfg)
    amp_target = np.sqrt(target)
    # Initial field: gaussian input in the SLM plane
    field = gaussian_pupil(beam_cfg).astype(np.complex128)
    field = field * np.exp(1j * rng.normal(0, 0.1, size=field.shape))

    history = []
    for i in range(n_iters):
        # SLM -> far field
        ff = focal_plane(field, beam_cfg, focal_length=cfg.focal_length)
        # amplitude constraint in far field
        ff = ff * (amp_target / (np.abs(ff) + 1e-12)) * relax
        # relax back toward the target amplitude
        ff = ff * (1 - relax) + amp_target * np.exp(1j * np.angle(ff)) * relax
        # far field -> SLM plane
        field = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(ff)))
        # phase-only constraint in SLM plane (keep gaussian amplitude, set phase)
        amp_slm = np.abs(gaussian_pupil(beam_cfg))
        field = amp_slm * np.exp(1j * np.angle(field))
        if verbose and i % 20 == 0:
            inten = np.abs(focal_plane(field, beam_cfg, focal_length=cfg.focal_length)) ** 2
            inten = inten / inten.max()
            center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
            m = compute_metrics(inten, target, center=center)
            history.append({"iter": i, **m})
    inten = np.abs(focal_plane(field, beam_cfg, focal_length=cfg.focal_length)) ** 2
    inten = inten / inten.max()
    center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
    metrics = compute_metrics(inten, target, center=center)
    return ShapingResult(
        method="gerchberg_saxton",
        phase=np.angle(field),
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
    init_phase = torch.as_tensor(rng.normal(0, 0.1, size=(cfg.n_grid, cfg.n_grid)), dtype=torch.float64).double()
    param = torch.nn.Parameter(init_phase)
    opt = torch.optim.Adam([param], lr=lr)

    # precompute the (static) lens + propagation as a torch function
    lam = beam_cfg.wavelength
    dx = beam_cfg.pixel_size
    n = cfg.n_grid
    fx = np.fft.fftshift(np.fft.fftfreq(n, dx))
    fy = np.fft.fftshift(np.fft.fftfreq(n, dx))
    fxg, fyg = np.meshgrid(fx, fy)
    kz = np.sqrt(np.abs((2 * np.pi) ** 2 * ((1 / lam) ** 2 - fxg**2 - fyg**2)))
    kz = torch.as_tensor(kz, dtype=torch.float64).double()
    transfer = torch.exp(1j * torch.as_tensor(cfg.focal_length, dtype=torch.float64) * kz)
    lens_ph = -torch.as_tensor(2 * np.pi / lam, dtype=torch.float64) * (
        torch.as_tensor(fxg**2 + fyg**2, dtype=torch.float64) / (2 * cfg.focal_length)
    )
    lens_ph = torch.exp(1j * lens_ph)
    in_amp = torch.as_tensor(gaussian_pupil(beam_cfg), dtype=torch.float64).double()

    def fft(x):
        return torch.fft.fftshift(torch.fft.ifftshift(torch.fft.fft2(x)))

    def ifft(x):
        return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(x)))

    history = []
    for i in range(n_iters):
        opt.zero_grad()
        field = in_amp * torch.exp(1j * param)
        field = field * lens_ph
        ff = fft(field) * transfer
        ff = ifft(ff)
        inten = (ff.real**2 + ff.imag**2)
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
                inten_np = (ff.real**2 + ff.imag**2).numpy()
                inten_np = inten_np / inten_np.max()
                center = np.unravel_index(np.argmax(inten_np), inten_np.shape)[::-1]
                m = compute_metrics(inten_np, target, center=center)
                history.append({"iter": i, **m})
    with torch.no_grad():
        field = in_amp * torch.exp(1j * param)
        field = field * lens_ph
        ff = fft(field) * transfer
        ff = ifft(ff)
        inten_np = (ff.real**2 + ff.imag**2).numpy()
        inten_np = inten_np / inten_np.max()
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
    inten = inten / inten.max()
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
    "gs_shape",
    "differentiable_shape",
    "spgd_shape",
    "analytic_amplitude_target",
]
