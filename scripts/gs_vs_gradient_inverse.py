"""Is canonical GS a BETTER inverse designer than gradient descent on the model?

Attempt 9 found the GS-derived target gave no improvement (delta -0.0212 +/- 0.0665),
but it also surfaced an unexpected number: the GS coefficients themselves score
1.0998 on the independent sim, versus ~0.95-1.02 mean for gradient inverse design.
That suggests the open-loop GS solve beats optimising the model's MSE -- plausible,
since GS enforces BOTH amplitude constraints every iteration in the Fourier domain
while Adam on a 135-coefficient Zernike basis lands in a mediocre local optimum
dominated by restart noise.

GS is deterministic, so a single 1.0998 proves nothing. This varies the thing that
actually makes GS non-deterministic -- the INITIAL random phase -- over many draws,
and compares the two distributions paired by that same initial phase. Same starts,
same budget, scored on the independent sim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\Projects\TIFO\AO-shaping\src")))
sys.path.insert(0, str(Path(r"D:\Projects\TIFO\AO-shaping\scripts")))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton  # noqa: E402
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

DRAWS = 16
STEPS = 60
SIGMA = 0.6
N_MODES = N_MAX * (N_MAX + 3) // 2
PIXEL_UM = 8.0
GS_ITERS = 40


def square_amplitude(grid: int) -> np.ndarray:
    amp = np.zeros((grid, grid), dtype=np.float64)
    half_w = int(SIZE_FRAC * grid * ASPECT / 2)
    half_h = int(SIZE_FRAC * grid / 2)
    c = grid // 2
    amp[c - half_h : c + half_h, c - half_w : c + half_w] = 1.0
    return amp


def gs_coefficients(init_phase: np.ndarray, grid: int) -> np.ndarray:
    """Canonical Fourier GS from a random start, projected onto the Zernike basis."""
    gs = gerchberg_saxton(
        np.ones((grid, grid), dtype=np.float64),
        square_amplitude(grid),
        iterations=GS_ITERS,
        cell_spacing=PIXEL_UM * 1e-6,
        propagation="fft",
    )
    combined = init_phase + np.asarray(gs.phase, dtype=np.float64)
    # [1:] drops piston: the model holds the K non-piston Noll modes only.
    return fit_zernike(combined, n_max=N_MAX)[1:]


def gradient_design(start: np.ndarray, target, mask, steps: int) -> np.ndarray:
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
    target = torch.as_tensor(square_amplitude(GRID), dtype=torch.float32)[None, None]
    flat = score_on_sim(np.zeros(N_MODES), GRID, N_MAX)["shape_sum"]

    print("=" * 92)
    print("CANONICAL GS vs GRADIENT inverse design (paired by initial phase)")
    print("=" * 92)
    print(f"draws={DRAWS} steps={STEPS} sigma={SIGMA} gs_iters={GS_ITERS} flat={flat:.4f}")

    rows = []
    for d in range(DRAWS):
        rng = np.random.default_rng(61000 + d)
        init = rng.normal(0.0, SIGMA, (GRID, GRID))
        grad_start = rng.normal(0.0, SIGMA, N_MODES)

        gs_c = gs_coefficients(init, GRID)
        gs_sim = score_on_sim(gs_c, GRID, N_MAX)["shape_sum"]

        grad_c = gradient_design(grad_start, target, mask, STEPS)
        grad_sim = score_on_sim(grad_c, GRID, N_MAX)["shape_sum"]

        rows.append({"draw": d, "gs": gs_sim, "gradient": grad_sim, "flat": flat})

    gs = np.array([r["gs"] for r in rows])
    grad = np.array([r["gradient"] for r in rows])
    d = gs - grad
    se = d.std(ddof=1) / np.sqrt(len(d))

    print(f"\n{'draw':>6}{'GS':>10}{'gradient':>11}{'GS - grad':>11}")
    for r in rows:
        print(f"{r['draw']:>6}{r['gs']:>10.4f}{r['gradient']:>11.4f}{r['gs'] - r['gradient']:>+11.4f}")

    print(f"\n{'':<6}{'mean':>10}{'sd':>10}{'min':>10}{'max':>10}")
    for name, arr in (("GS", gs), ("gradient", grad)):
        print(f"{name:<6}{arr.mean():>10.4f}{arr.std():>10.4f}{arr.min():>10.4f}{arr.max():>10.4f}")

    print(f"\npaired GS - gradient = {d.mean():+.4f} +/- {se:.4f}  "
          f"positives={int((d > 0).sum())}/{len(d)}  t={d.mean() / se:+.2f}")
    print(f"GS beats flat in {int((gs > flat).sum())}/{len(d)}; "
          f"gradient beats flat in {int((grad > flat).sum())}/{len(d)}")

    verdict = "GS IS BETTER" if d.mean() > 2 * se else "no difference"
    print(f"\nVERDICT: {verdict}")

    out = Path("report/loss_defects/gs_vs_gradient.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()