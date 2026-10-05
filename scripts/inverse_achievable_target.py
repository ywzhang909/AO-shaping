"""Does an ACHIEVABLE target beat a binary square for inverse design?

Attempt 8 found plain `mse` is the best-aligned inverse objective (+0.084 between
best- and worst-converged restarts) while the physical terms are anti-aligned, and
that the objective itself is worth nothing versus any other (+0.0002). So the lever is
not the loss -- it is *what the loss matches*.

Hypothesis: a binary square is UNACHIEVABLE in a 135-coefficient Zernike basis, so the
MSE residual is dominated by an irreducible mismatch and its gradient carries little
information about direction. Replacing the target with the best pattern the model can
actually produce for the same goal should make the gradient informative.

The achievable target is built from canonical pieces only, no new maths:
  1. canonical Fourier Gerchberg-Saxton on a square target amplitude -> pupil phase
  2. canonical `fit_zernike` to project that phase onto the model's Zernike basis
  3. the model's own `correction_far_field` for those coefficients -> achievable intensity
The model under test is used only to BUILD the target, never to score it; scoring is
on the independent sim, paired by restart.

Arms: binary square (current baseline) vs the GS-derived achievable intensity, plus
the achievable target's own quality as a reference point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\Projects\TIFO\AO-shaping\src")))
sys.path.insert(0, str(Path(r"D:\Projects\TIFO\AO-shaping\scripts")))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ao_shaping.algorithm.signal_processing.gerchberg_saxton import (  # noqa: E402
    gerchberg_saxton,
)
from ao_shaping.utils.wavefront.zernike_calc import fit_zernike  # noqa: E402
from inverse_design_sim_eval import (  # noqa: E402
    ASPECT,
    GRID,
    N_MAX,
    PADDING,
    SIZE_FRAC,
    score_on_sim,
)
from ml.zernike.losses import LossConfig, composite_loss, roi_mask  # noqa: E402
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402

RESTARTS = 16
STEPS = 60
SIGMA = 0.6
N_MODES = N_MAX * (N_MAX + 3) // 2
PIXEL_UM = 8.0


def square_amplitude(grid: int) -> np.ndarray:
    amp = np.zeros((grid, grid), dtype=np.float64)
    half_w = int(SIZE_FRAC * grid * ASPECT / 2)
    half_h = int(SIZE_FRAC * grid / 2)
    c = grid // 2
    amp[c - half_h : c + half_h, c - half_w : c + half_w] = 1.0
    return amp


def binary_target(grid: int) -> torch.Tensor:
    return torch.as_tensor(square_amplitude(grid), dtype=torch.float32)[None, None]


def achievable_target(grid: int) -> tuple[torch.Tensor, np.ndarray]:
    """GS pupil phase -> Zernike projection -> the model's own achievable intensity.

    Returns (target tensor, coefficients used).
    """
    target_amp = square_amplitude(grid)
    source_amp = np.ones((grid, grid), dtype=np.float64)
    gs = gerchberg_saxton(
        source_amp,
        target_amp,
        iterations=40,
        cell_spacing=PIXEL_UM * 1e-6,
        propagation="fft",
    )
    # it_zernike returns all terms in Noll order INCLUDING piston; the model
    # holds the K non-piston modes only (Noll 2..136). Piston is also physically
    # unobservable here -- it carries no gradient in this basis.
    coefficients = fit_zernike(np.asarray(gs.phase, dtype=np.float64), n_max=N_MAX)[1:]
    assert len(coefficients) == N_MODES, f"got {len(coefficients)}, want {N_MODES}"
    model = ZernikeAmpModel(
        ZernikeAmpConfig(
            n_max=N_MAX, grid=grid, far_field_padding=PADDING, normalization="peak"
        )
    )
    with torch.no_grad():
        model.coefficients.copy_(torch.as_tensor(np.asarray(coefficients, dtype=np.float32)))
        intensity = model.correction_far_field()
    return intensity, coefficients


def design(start: np.ndarray, target: torch.Tensor, mask, steps: int) -> np.ndarray:
    model = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING, normalization="peak")
    )
    with torch.no_grad():
        model.coefficients.copy_(torch.as_tensor(start.astype(np.float32)))
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        composite_loss(model.correction_far_field(), target, mask, LossConfig(w_mse=1.0))[
            "_mean_total"
        ].backward()
        opt.step()
    return model.coefficients_array()


def main() -> None:
    mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle", SIZE_FRAC * GRID, ASPECT)
    flat = score_on_sim(np.zeros(N_MODES), GRID, N_MAX)["shape_sum"]

    binary = binary_target(GRID)
    gs_target, gs_coeffs = achievable_target(GRID)

    print("=" * 92)
    print("ACHIEVABLE (GS-derived) TARGET vs BINARY SQUARE for inverse design")
    print("=" * 92)
    print(f"flat = {flat:.4f}")
    print(f"GS coefficients: n={len(gs_coeffs)} absmax={np.abs(gs_coeffs).max():.4f} rad")
    print(f"achievable target: peak={float(gs_target.max()):.4f} "
          f"mean={float(gs_target.mean()):.5f} "
          f"(binary square mean={float(binary.mean()):.5f})")
    gs_sim = score_on_sim(np.asarray(gs_coeffs, dtype=np.float64), GRID, N_MAX)["shape_sum"]
    print(f"sim shape_sum of the GS target coefficients themselves: {gs_sim:.4f}")

    starts = [
        np.random.default_rng(52000 + r).normal(0.0, SIGMA, N_MODES) for r in range(RESTARTS)
    ]
    arms = {"binary_square": binary, "gs_achievable": gs_target}
    rows = []
    for name, target in arms.items():
        for idx, start in enumerate(starts):
            coeffs = design(start, target, mask, STEPS)
            rows.append(
                {
                    "target": name,
                    "restart": idx,
                    "sim": score_on_sim(coeffs, GRID, N_MAX)["shape_sum"],
                }
            )

    def arr(name: str) -> np.ndarray:
        return np.array([r["sim"] for r in rows if r["target"] == name])

    base = arr("binary_square")
    enriched = arr("gs_achievable")
    d = enriched - base
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0

    print(f"\n{'target':<16}{'sim mean':>11}{'sim sd':>10}{'vs flat':>10}")
    for name in arms:
        s = arr(name)
        print(f"{name:<16}{s.mean():>11.4f}{s.std():>10.4f}{s.mean() - flat:>+10.4f}")

    print("\npaired delta (gs_achievable - binary_square), same restart:")
    print(f"  delta={d.mean():+.4f} +/- {se:.4f}  positives={int((d > 0).sum())}/{len(d)}"
          + (f"  t={d.mean() / se:+.2f}" if se else ""))

    print("\nreference points:")
    print(f"  flat                     {flat:.4f}")
    print(f"  GS target coefficients   {gs_sim:.4f}")

    out = Path("report/loss_defects/achievable_target.json")
    out.write_text(json.dumps({"rows": rows, "gs_coefficients": gs_coeffs.tolist()}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()