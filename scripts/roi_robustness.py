"""Are the headline conclusions artefacts of the evaluator's ROI geometry?

Re-measures both headline claims across a sweep of ROI geometries on the CORRECTED
evaluator, holding the methods and the starts fixed. A conclusion that survives a
wrong ROI is worth something; one that flips is a measurement of the ROI, not of the
method.

The two claims:
  1. canonical GS beats gradient inverse design from a random start
  2. gradient refinement on top of GS makes it worse

**This is a re-run, not the original.** The first version of this script and of
`restart_claim_robustness.py` each carried their own copy of the evaluator, and that
copy cropped the simulator's far field with ``[:64, :64]`` -- the top-left CORNER,
while the 0-order sits at the array CENTRE. Every number below was therefore taken on
a patch ~1e5 times dimmer than the beam. See the CORRECTION section of
``report/loss_defects/PROCESS.md``.

The evaluator, ROI, GS and gradient operators are no longer defined here: they come
from ``ml.zernike.inverse_design``, which runs the 64x64 pupil directly so its
far-field grid matches the grid GS designs on, pixel for pixel. That also makes
``SIZE_FRAC`` mean what it says -- a fraction of the *beam's* frame rather than of a
stray corner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ml.zernike import inverse_design as inv  # noqa: E402

GRID, N_MAX, REFINE_STEPS, DRAWS = inv.GRID, inv.N_MAX, 60, 10
N_MODES = N_MAX * (N_MAX + 3) // 2

SIZE_FRACS = [0.25, 0.375, 0.50]
ASPECTS = [1.0, 4.0 / 3.0, 1.5]


def gs_coefficients(init_phase: np.ndarray, size_frac: float, aspect: float) -> np.ndarray:
    """GS proposal, warm-started from a random phase (the original protocol)."""
    return inv.gs_coefficients(size_frac, aspect, n_max=N_MAX, init_phase=init_phase)


def gradient_design(start: np.ndarray, target, mask) -> np.ndarray:
    return inv.gradient_zernike(
        start, inv.build_model(n_max=N_MAX, grid=GRID, padding=inv.PADDING), target, mask,
        steps=REFINE_STEPS,
    )


def main() -> None:
    print("=" * 104)
    print("ROBUSTNESS on the CORRECTED evaluator: do the headline conclusions survive a ROI sweep?")
    print("=" * 104)
    print(f"grid={GRID} n_max={N_MAX} padding={inv.PADDING} sim_window={inv.SIM_WINDOW} "
          f"draws={DRAWS} refine_steps={REFINE_STEPS}")

    rows: list[dict] = []
    for size_frac in SIZE_FRACS:
        for aspect in ASPECTS:
            mask = inv.roi(GRID, size_frac, aspect)
            target = inv.target_tensor(GRID, size_frac, aspect)
            flat = inv.score_coefficients(np.zeros(N_MODES), size_frac, aspect, n_max=N_MAX)
            for d in range(DRAWS):
                rng = np.random.default_rng(99000 + d)
                init = rng.normal(0.0, 0.6, (GRID, GRID))
                gs_c = gs_coefficients(init, size_frac, aspect)
                grad_c = gradient_design(rng.normal(0.0, 0.6, N_MODES), target, mask)
                rows.append({
                    "size_frac": size_frac, "aspect": aspect, "draw": d, "flat": flat,
                    "gs": inv.score_coefficients(gs_c, size_frac, aspect, n_max=N_MAX),
                    "grad": inv.score_coefficients(grad_c, size_frac, aspect, n_max=N_MAX),
                    "gs_refined": inv.score_coefficients(
                        gradient_design(gs_c, target, mask), size_frac, aspect, n_max=N_MAX
                    ),
                })

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
