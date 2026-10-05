"""Are the headline conclusions artefacts of the evaluator's ROI geometry?

Every result in attempts 8-12 was measured with ONE arbitrary ROI: a
`SIZE_FRAC=0.375`, `ASPECT=4/3` box at `(grid/2, grid/2)` on a `[:grid,:grid]` crop of
the sim far field. If the GS-vs-gradient ranking and the refinement-hurts finding are
artefacts of that box rather than properties of the methods, they are worthless.

This re-tests both headline claims across a sweep of ROI geometries, holding the
methods and the starts fixed. A conclusion that survives a wrong ROI is worth
something; one that flips is a measurement of my ROI, not of the method.

The two claims:
  1. canonical GS beats gradient inverse design from a random start  (attempt 10)
  2. gradient refinement on top of GS makes it worse                (attempt 11)
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


def square_amplitude(grid: int, size_frac: float, aspect: float) -> np.ndarray:
    amp = np.zeros((grid, grid), dtype=np.float64)
    half_w = int(size_frac * grid * aspect / 2)
    half_h = int(size_frac * grid / 2)
    c = grid // 2
    amp[c - half_h : c + half_h, c - half_w : c + half_w] = 1.0
    return amp


def far_field(coefficients: np.ndarray) -> np.ndarray:
    """Independent sim far field, 64x64 crop. Shared by every ROI variant."""
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


def score(coefficients: np.ndarray, size_frac: float, aspect: float) -> float:
    crop = far_field(coefficients)
    pib, uni = rms_pib_terms(
        crop, (GRID / 2, GRID / 2), "rectangle", size_frac * GRID, aspect
    )
    return float(pib + uni)


def gs_coefficients(init: np.ndarray, size_frac: float, aspect: float) -> np.ndarray:
    gs = gerchberg_saxton(
        np.ones((GRID, GRID), dtype=np.float64),
        square_amplitude(GRID, size_frac, aspect),
        iterations=GS_ITERS, cell_spacing=PIXEL_UM * 1e-6, propagation="fft",
    )
    return fit_zernike(init + np.asarray(gs.phase, np.float64), n_max=N_MAX)[1:]


def gradient_design(start: np.ndarray, target, mask, steps: int = REFINE_STEPS):
    model = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING, normalization="peak")
    )
    with torch.no_grad():
        model.coefficients.copy_(torch.as_tensor(np.asarray(start, np.float32)))
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        composite_loss(model.correction_far_field(), target, mask, LossConfig(w_mse=1.0))[
            "_mean_total"
        ].backward()
        opt.step()
    return model.coefficients_array()


def main() -> None:
    print("=" * 104)
    print("ROBUSTNESS: do the headline conclusions survive a change of evaluator ROI?")
    print("=" * 104)

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
                init = rng.normal(0.0, 0.6, (GRID, GRID))
                gs_c = gs_coefficients(init, size_frac, aspect)
                grad_c = gradient_design(rng.normal(0.0, 0.6, N_MODES), target, mask)
                rows.append(
                    {
                        "size_frac": size_frac, "aspect": aspect, "draw": d, "flat": flat,
                        "gs": score(gs_c, size_frac, aspect),
                        "grad": score(grad_c, size_frac, aspect),
                        "gs_refined": score(
                            gradient_design(gs_c, target, mask), size_frac, aspect
                        ),
                    }
                )

    def sel(sf: float, asp: float) -> list[dict]:
        return [r for r in rows if r["size_frac"] == sf and abs(r["aspect"] - asp) < 1e-9]

    print(f"\n{'size_frac':>10}{'aspect':>8}{'flat':>9}{'GS':>9}{'grad':>9}"
          f"{'GS-grad':>10}{'t':>7}{'GS+ref':>9}{'ref delta':>11}{'t':>7}")
    verdicts = []
    for sf in SIZE_FRACS:
        for asp in ASPECTS:
            s = sel(sf, asp)
            gs = np.array([r["gs"] for r in s])
            gr = np.array([r["grad"] for r in s])
            rf = np.array([r["gs_refined"] for r in s])
            d1, d2 = gs - gr, rf - gs
            t1 = d1.mean() / (d1.std(ddof=1) / np.sqrt(len(d1)))
            t2 = d2.mean() / (d2.std(ddof=1) / np.sqrt(len(d2)))
            verdicts.append((sf, asp, d1.mean(), t1, d2.mean(), t2))
            print(
                f"{sf:>10.3f}{asp:>8.3f}{s[0]['flat']:>9.4f}{gs.mean():>9.4f}{gr.mean():>9.4f}"
                f"{d1.mean():>+10.4f}{t1:>+7.2f}{rf.mean():>9.4f}{d2.mean():>+11.4f}{t2:>+7.2f}"
            )

    print("\n" + "=" * 104)
    c1 = [v for v in verdicts if v[3] > 0]
    c2 = [v for v in verdicts if v[5] < 0]
    print(f"claim 1  GS > gradient        : holds in {len(c1)}/{len(verdicts)} ROI settings")
    print(f"claim 2  refinement hurts GS   : holds in {len(c2)}/{len(verdicts)} ROI settings")
    robust = len(c1) == len(verdicts) and len(c2) == len(verdicts)
    print(f"\nVERDICT: {'BOTH ROBUST across the ROI sweep' if robust else 'NOT ROBUST -- ROI-dependent'}")

    out = Path("report/loss_defects/roi_robustness.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()