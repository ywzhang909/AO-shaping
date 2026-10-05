"""Is "refinement is a restart, not a gradient" robust across BOTH ROI and objective?

Attempt 13 left exactly one ROI-independent conclusion, but it was measured on a
single objective (`mse`). A claim that survives a nuisance sweep on one axis and not
another is only half-robust, so this repeats the sweep over objective as well:

    axes: 9 ROI geometries  x  2 objectives (mse, physical)
    for each: x = (proposal quality)  y = (refinement delta)

If refinement is genuinely acting as a restart -- rescuing weak proposals, degrading
strong ones -- the monotone relationship must appear for BOTH objectives. If it only
appears for `mse`, then the surviving conclusion is objective-specific and must be
weakened again.

Prediction if the "restart" reading is right: spearman(x, y) strongly negative in
both objectives, with similar magnitude. Prediction if refinement carries real
directional information from the objective: y would track how well the refinement
optimised its own objective, which attempt 8 already showed does not track reality.
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
from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem  # noqa: E402
from ao_shaping.utils.image.target.metrics import rms_pib_terms  # noqa: E402
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator, fit_zernike  # noqa: E402
from ml.zernike.losses import LossConfig, composite_loss, roi_mask  # noqa: E402
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402

GRID, N_MAX, PADDING = 64, 15, 12
N_MODES = N_MAX * (N_MAX + 3) // 2
PIXEL_UM, GS_ITERS, REFINE_STEPS, DRAWS = 8.0, 40, 60, 10
PANEL_H, PANEL_W, DISC_R = 1200, 1920, 450

SIZE_FRACS = [0.25, 0.375, 0.50]
ASPECTS = [1.0, 4.0 / 3.0, 1.5]
OBJECTIVES = {
    "mse": LossConfig(w_mse=1.0),
    "physical": LossConfig(w_mse=0.0, w_pib=1.0, w_uniformity=1.0),
}


def square_amplitude(grid: int, size_frac: float, aspect: float) -> np.ndarray:
    amp = np.zeros((grid, grid), dtype=np.float64)
    half_w = int(size_frac * grid * aspect / 2)
    half_h = int(size_frac * grid / 2)
    c = grid // 2
    amp[c - half_h : c + half_h, c - half_w : c + half_w] = 1.0
    return amp


def far_field(coefficients: np.ndarray) -> np.ndarray:
    gen = ZernikeGenerator((GRID, GRID), n_orders=N_MAX)
    pupil = np.nan_to_num(gen.generate_noll(np.asarray(coefficients, np.float64)), nan=0.0)
    cy, cx = PANEL_H // 2, PANEL_W // 2
    phase = np.zeros((PANEL_H, PANEL_W), dtype=np.float64)
    phase[cy - GRID // 2 : cy + GRID // 2, cx - GRID // 2 : cx + GRID // 2] = pupil
    s = SimPibSystem(
        slm_shape=(PANEL_H, PANEL_W), ccd_res=(512, 512), beam_w0=float(DISC_R),
        noise_adu=0.0, seed=0, far_field_padding=4, far_field_window=1024,
    )
    s.set_phase_rad(phase)
    crop = np.asarray(s.far_field(), dtype=np.float64)[:GRID, :GRID]
    if not np.all(np.isfinite(crop)):
        raise RuntimeError("non-finite far field")
    return crop


def score(coefficients, size_frac: float, aspect: float) -> float:
    crop = far_field(coefficients)
    pib, uni = rms_pib_terms(crop, (GRID / 2, GRID / 2), "rectangle", size_frac * GRID, aspect)
    return float(pib + uni)


def gs_coefficients(init, size_frac: float, aspect: float) -> np.ndarray:
    gs = gerchberg_saxton(
        np.ones((GRID, GRID), dtype=np.float64),
        square_amplitude(GRID, size_frac, aspect),
        iterations=GS_ITERS, cell_spacing=PIXEL_UM * 1e-6, propagation="fft",
    )
    return fit_zernike(init + np.asarray(gs.phase, np.float64), n_max=N_MAX)[1:]


def refine(start, weights, mask, target, steps: int = REFINE_STEPS):
    model = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING, normalization="peak")
    )
    with torch.no_grad():
        model.coefficients.copy_(torch.as_tensor(np.asarray(start, np.float32)))
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        composite_loss(model.correction_far_field(), target, mask, weights)[
            "_mean_total"
        ].backward()
        opt.step()
    return model.coefficients_array()


def pearson(a, b) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    return float((a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum()))


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))


def main() -> None:
    print("=" * 100)
    print("IS 'REFINEMENT IS A RESTART' ROBUST ACROSS ROI *AND* OBJECTIVE?")
    print("=" * 100)

    rows = []
    for size_frac in SIZE_FRACS:
        for aspect in ASPECTS:
            mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle",
                            size_frac * GRID, aspect)
            target = torch.as_tensor(
                square_amplitude(GRID, size_frac, aspect), dtype=torch.float32
            )[None, None]
            flat = score(np.zeros(N_MODES), size_frac, aspect)
            for d in range(DRAWS):
                rng = np.random.default_rng(99000 + d)
                gs_c = gs_coefficients(rng.normal(0.0, 0.6, (GRID, GRID)), size_frac, aspect)
                gs_score = score(gs_c, size_frac, aspect)
                row = {"size_frac": size_frac, "aspect": aspect, "draw": d,
                       "flat": flat, "gs": gs_score}
                for name, weights in OBJECTIVES.items():
                    row[f"refined_{name}"] = score(
                        refine(gs_c, weights, mask, target), size_frac, aspect
                    )
                rows.append(row)

    print(f"\n{'ROI':>14}{'GS-flat':>10}" + "".join(f"{f'delta {k}':>14}" for k in OBJECTIVES))
    cells = []
    for size_frac in SIZE_FRACS:
        for aspect in ASPECTS:
            sel = [r for r in rows if r["size_frac"] == size_frac
                   and abs(r["aspect"] - aspect) < 1e-9]
            gs_minus_flat = np.mean([r["gs"] - r["flat"] for r in sel])
            deltas = {
                k: float(np.mean([r[f"refined_{k}"] - r["gs"] for r in sel]))
                for k in OBJECTIVES
            }
            cells.append((gs_minus_flat, deltas))
            print(f"{size_frac:>7.3f}x{aspect:<6.3f}{gs_minus_flat:>+10.4f}"
                  + "".join(f"{deltas[k]:>+14.4f}" for k in OBJECTIVES))

    xs = np.array([c[0] for c in cells])
    print(f"\n{'objective':<12}{'pearson':>10}{'spearman':>11}{'verdict':>28}")
    verdicts = {}
    for k in OBJECTIVES:
        ys = np.array([c[1][k] for c in cells])
        p, s = pearson(xs, ys), spearman(xs, ys)
        ok = s < -0.6
        verdicts[k] = ok
        print(f"{k:<12}{p:>+10.4f}{s:>+11.4f}"
              f"{('monotone as predicted' if ok else 'NOT monotone'):>28}")

    both = all(verdicts.values())
    print(f"\nVERDICT: {'ROBUST across ROI x objective' if both else 'OBJECTIVE-DEPENDENT -- weaken the claim'}")
    print("Reading: y decreases as x (proposal quality) increases in BOTH objectives means")
    print("refinement rescues weak proposals and degrades strong ones regardless of what")
    print("it is optimising -- i.e. restart behaviour, not directional information.")

    out = Path("report/loss_defects/restart_claim_robustness.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()