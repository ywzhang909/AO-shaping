"""Reference optimization methods for the simulated 2f SLM shaping bench."""

from __future__ import annotations

import numpy as np

from ao_shaping.drivers.sim.beam_backend import focal_plane, gaussian_pupil
from ao_shaping.drivers.sim.slm_shaping_bench import (
    ShapingBenchConfig,
    ShapingResult,
    composite_score,
    compute_metrics,
    forward_intensity,
    make_target,
)


def gs_shape(
    cfg: ShapingBenchConfig,
    *,
    n_iters: int = 200,
    relax: float = 1.0,
    seed: int | None = None,
    verbose: bool = False,
    return_phase_only: bool = True,
) -> ShapingResult:
    """Apply Gerchberg-Saxton amplitude constraints on the simulated bench."""
    beam_cfg = cfg.make_beam_config()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    target = make_target(cfg)
    amp_target = np.sqrt(target)
    field = gaussian_pupil(beam_cfg).astype(np.complex128)
    field = field * np.exp(1j * rng.normal(0, 0.1, size=field.shape))

    history = []
    for i in range(n_iters):
        ff = focal_plane(field, beam_cfg, focal_length=cfg.focal_length)
        ff = ff * (amp_target / (np.abs(ff) + 1e-12)) * relax
        ff = ff * (1 - relax) + amp_target * np.exp(1j * np.angle(ff)) * relax
        field = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(ff)))
        amp_slm = np.abs(gaussian_pupil(beam_cfg))
        field = amp_slm * np.exp(1j * np.angle(field))
        if verbose and i % 20 == 0:
            inten = np.abs(focal_plane(field, beam_cfg, focal_length=cfg.focal_length)) ** 2
            inten = inten / inten.max()
            center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
            history.append({"iter": i, **compute_metrics(inten, target, center=center)})
    inten = np.abs(focal_plane(field, beam_cfg, focal_length=cfg.focal_length)) ** 2
    inten = inten / inten.max()
    center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
    return ShapingResult(
        method="gerchberg_saxton",
        phase=np.angle(field),
        intensity=inten,
        metrics=compute_metrics(inten, target, center=center),
        history=history,
        n_iters=n_iters,
    )


def differentiable_shape(
    cfg: ShapingBenchConfig,
    *,
    n_iters: int = 300,
    lr: float = 0.05,
    seed: int | None = None,
) -> ShapingResult:
    """Optimize the simulated SLM phase by differentiating the far field."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("differentiable_shape requires torch") from exc

    beam_cfg = cfg.make_beam_config()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    target = make_target(cfg)
    tgt = torch.as_tensor(target, dtype=torch.float64)
    init_phase = torch.as_tensor(
        rng.normal(0, 0.1, size=(cfg.n_grid, cfg.n_grid)), dtype=torch.float64
    )
    param = torch.nn.Parameter(init_phase)
    opt = torch.optim.Adam([param], lr=lr)

    lam = beam_cfg.wavelength
    dx = beam_cfg.pixel_size
    n = cfg.n_grid
    fx = np.fft.fftshift(np.fft.fftfreq(n, dx))
    fy = np.fft.fftshift(np.fft.fftfreq(n, dx))
    fxg, fyg = np.meshgrid(fx, fy)
    kz = np.sqrt(np.abs((2 * np.pi) ** 2 * ((1 / lam) ** 2 - fxg**2 - fyg**2)))
    kz = torch.as_tensor(kz, dtype=torch.float64)
    transfer = torch.exp(1j * torch.as_tensor(cfg.focal_length, dtype=torch.float64) * kz)
    lens_ph = -torch.as_tensor(2 * np.pi / lam, dtype=torch.float64) * (
        torch.as_tensor(fxg**2 + fyg**2, dtype=torch.float64) / (2 * cfg.focal_length)
    )
    lens_ph = torch.exp(1j * lens_ph)
    in_amp = torch.as_tensor(gaussian_pupil(beam_cfg), dtype=torch.float64)

    def fft(x):
        return torch.fft.fftshift(torch.fft.ifftshift(torch.fft.fft2(x)))

    def ifft(x):
        return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(x)))

    history = []
    for i in range(n_iters):
        opt.zero_grad()
        field = in_amp * torch.exp(1j * param)
        field = field * lens_ph
        ff = ifft(fft(field) * transfer)
        inten = ff.real**2 + ff.imag**2
        inten = inten / inten.sum()
        a = inten - inten.mean()
        b = tgt - tgt.mean()
        overlap = (a * b).sum() / (a.norm() * b.norm() + 1e-12)
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
                history.append({"iter": i, **compute_metrics(inten_np, target, center=center)})
    with torch.no_grad():
        field = in_amp * torch.exp(1j * param)
        field = field * lens_ph
        ff = ifft(fft(field) * transfer)
        inten_np = (ff.real**2 + ff.imag**2).numpy()
        inten_np = inten_np / inten_np.max()
    center = np.unravel_index(np.argmax(inten_np), inten_np.shape)[::-1]
    return ShapingResult(
        method="differentiable",
        phase=param.detach().numpy(),
        intensity=inten_np,
        metrics=compute_metrics(inten_np, target, center=center),
        history=history,
        n_iters=n_iters,
    )


def spgd_shape(
    cfg: ShapingBenchConfig,
    *,
    n_iters: int = 600,
    delta: float = 0.1,
    lr: float = 0.02,
    seed: int | None = None,
    dim: int | None = None,
) -> ShapingResult:
    """Run freeform SPGD on the simulated SLM phase."""
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    target = make_target(cfg)
    d = dim or 16
    n_par = d * d
    phase_flat = rng.normal(0, 0.05, size=n_par)

    def upsample(vec: np.ndarray) -> np.ndarray:
        ph = np.kron(vec.reshape(d, d), np.ones((cfg.n_grid // d, cfg.n_grid // d)))
        if ph.shape != (cfg.n_grid, cfg.n_grid):
            ph = ph[: cfg.n_grid, : cfg.n_grid]
        return ph

    def eval_score(vec: np.ndarray) -> float:
        inten = forward_intensity(upsample(vec), cfg)
        inten = inten / inten.max()
        center = np.unravel_index(np.argmax(inten), inten.shape)[::-1]
        return composite_score(compute_metrics(inten, target, center=center))

    score = eval_score(phase_flat)
    g = np.zeros(n_par)
    history = [{"iter": 0, "score": score, "PIB": 0, "CV": float("inf")}]
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


def analytic_amplitude_target(cfg: ShapingBenchConfig) -> ShapingResult:
    """Return the ideal amplitude target as a comparison baseline."""
    target = make_target(cfg)
    return ShapingResult(
        method="analytic_amplitude_baseline",
        phase=np.zeros((cfg.n_grid, cfg.n_grid)),
        intensity=target,
        metrics=compute_metrics(target, target),
        n_iters=0,
    )
