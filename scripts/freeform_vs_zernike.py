"""Freeform phase: the parameterisation the hardware runner actually uses.

Every attempt in PROCESS.md so far optimised a 135-coefficient Zernike vector. The
shipped `slm_gs_refine` optimises a *freeform* phase grid instead, so a Zernike-only
conclusion cannot be transferred to freeform unexamined. This closes that gap.

No new architecture is needed: `ZernikeAmpModel.forward` validates only that its input
is `(B, 1, g, g)` at the basis resolution, and `measured = complex(phase_cos, phase_sin)`
*is* the input phasor. So any phase -- including a freeform one -- can be pushed through
the model. (An earlier note in this file claimed freeform needed a new forward model;
that was wrong.)

Two things freeform changes relative to the Zernike path:

* **GS no longer needs a projection bottleneck.** The Zernike arm runs
  `fit_zernike(gs.phase, n_max)`, and attempt 12 found that projection is actively
  harmful at n_max=20 (it lands *below* flat). In freeform the GS pupil phase is used
  directly, so that failure mode disappears.
* **The parameterisation is far richer** -- a 24x24 grid upsampled to 64x64 is 576 DOF
  against Zernike's 135 -- so it can represent square far fields that low-order
  Zernike cannot. The repo's own anti-pattern table says as much.

Arms, all scored on the independent sim (higher `pib + uniformity` is better):
  gs_zernike    GS -> fit_zernike -> Zernike gradient design   (the old path)
  gs_freeform   GS pupil phase used directly, no projection
  grad_freeform freeform gradient design from a random start
  gs_fm_refine  freeform gradient refinement started from gs_freeform

**This is a re-run.** The first version carried its own evaluator, which cropped the
far field with `[:64, :64]` -- the top-left corner, where the value is ~1e5 times dimmer
than the beam at the array centre. Its "+0.1272 for removing the projection bottleneck"
and its spearman -0.8667 / -0.9167 were both measured there. Both are void until this
run replaces them. Everything now comes from `ml.zernike.inverse_design`, so the crop
cannot drift again.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ao_shaping.utils.wavefront.zernike_calc import (  # noqa: E402
    ZernikeGenerator,
    fit_zernike,
)
from ml.zernike import inverse_design as inv  # noqa: E402
from ml.zernike.losses import LossConfig, composite_loss  # noqa: E402

GRID, N_MAX, PADDING = inv.GRID, inv.N_MAX, inv.PADDING
GS_ITERS, STEPS, DRAWS, COARSE = inv.GS_ITERS, 60, 8, inv.COARSE_GRID

SIZE_FRACS = [0.25, 0.375, 0.50]
ASPECTS = [1.0, 4.0 / 3.0, 1.5]

MODEL = inv.build_model(n_max=N_MAX, grid=GRID, padding=PADDING, seed=0)
WEIGHTS = LossConfig(w_mse=1.0)


def _resize(phase, size: int) -> np.ndarray:
    """Bilinear downsample to the coarse grid (kept local: freeform-specific)."""
    t = torch.as_tensor(np.asarray(phase, dtype=np.float32))[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[
        0, 0
    ].numpy()


def _zernike_path(gs_phase: np.ndarray, target, mask, size_frac: float, aspect: float) -> float:
    """GS -> fit_zernike -> Zernike gradient design: the old attempts 10-14 path."""
    model = inv.build_model(n_max=N_MAX, grid=GRID, padding=PADDING, seed=0)
    with torch.no_grad():
        model.coefficients.copy_(
            torch.as_tensor(fit_zernike(gs_phase, n_max=N_MAX)[1:].astype(np.float32))
        )
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(STEPS):
        opt.zero_grad()
        composite_loss(model.correction_far_field(), target, mask, WEIGHTS)[
            "_mean_total"
        ].backward()
        opt.step()
    gen = ZernikeGenerator((GRID, GRID), n_orders=N_MAX)
    phase = np.nan_to_num(gen.generate_noll(model.coefficients_array()))
    return inv.score_phase(phase, size_frac, aspect)


def main() -> None:
    print("=" * 100)
    print("FREEFORM vs ZERNIKE on the CORRECTED evaluator: does the projection bottleneck matter?")
    print("=" * 100)
    print(f"grid={GRID} n_max={N_MAX} coarse={COARSE} gs_iters={GS_ITERS} "
          f"steps={STEPS} draws={DRAWS} sim_window={inv.SIM_WINDOW}")
    print("score = pib + uniformity, HIGHER is better")

    rows: list[dict] = []
    for size_frac in SIZE_FRACS:
        for aspect in ASPECTS:
            mask = inv.roi(GRID, size_frac, aspect)
            target = inv.target_tensor(GRID, size_frac, aspect)
            gs_p = inv.gs_phase(size_frac, aspect, iterations=GS_ITERS)
            for d in range(DRAWS):
                rng = np.random.default_rng(77000 + d)
                rows.append({
                    "size_frac": size_frac, "aspect": aspect, "draw": d,
                    "flat": inv.score_phase(np.zeros((GRID, GRID)), size_frac, aspect),
                    "gs_zernike": _zernike_path(gs_p, target, mask, size_frac, aspect),
                    "gs_freeform": inv.score_phase(gs_p, size_frac, aspect),
                    "grad_freeform": inv.score_phase(
                        inv.gradient_freeform(
                            rng.normal(0.0, 0.6, (COARSE, COARSE)), MODEL, target, mask,
                            WEIGHTS, STEPS,
                        ), size_frac, aspect,
                    ),
                    "gs_fm_refined": inv.score_phase(
                        inv.gradient_freeform(
                            _resize(gs_p, COARSE), MODEL, target, mask, WEIGHTS, STEPS,
                        ), size_frac, aspect,
                    ),
                })

    keys = ("flat", "gs_zernike", "gs_freeform", "grad_freeform", "gs_fm_refined")
    cells: list[dict] = []
    print(f"\n{'ROI':>14}" + "".join(f"{k:>14}" for k in keys))
    for size_frac in SIZE_FRACS:
        for aspect in ASPECTS:
            sel = [r for r in rows if r["size_frac"] == size_frac
                   and abs(r["aspect"] - aspect) < 1e-9]
            mean = {k: float(np.mean([r[k] for r in sel])) for k in keys}
            cells.append(mean)
            print(f"{size_frac:>7.3f}x{aspect:<6.3f}" + "".join(f"{mean[k]:>14.4f}" for k in keys))

    def report(label: str, a: str, b: str) -> None:
        d = np.array([c[a] - c[b] for c in cells])
        print(f"  {label:<34} mean={d.mean():+.4f}  positives={int((d > 0).sum())}/{len(d)}")

    print("\n--- paired across the 9 ROI cells ---")
    report("GS_freeform - GS_zernike", "gs_freeform", "gs_zernike")
    report("grad_freeform - GS_freeform", "grad_freeform", "gs_freeform")
    report("GS_freeform - flat", "gs_freeform", "flat")
    report("grad_freeform - flat", "grad_freeform", "flat")
    report("GS_zernike - flat", "gs_zernike", "flat")

    x = np.array([c["gs_freeform"] - c["flat"] for c in cells])
    y = np.array([c["gs_fm_refined"] - c["gs_freeform"] for c in cells])
    print("\n--- does refinement behave like a restart in freeform space? ---")
    print(f"  spearman(GS_freeform - flat, refinement delta) = {inv.spearman(x, y):+.4f}")
    print("  Zernike space, corner-cropped evaluator: -0.8667 (mse) / -0.9167 (physical)")
    print("  Zernike space, corrected evaluator:       +0.9333 (mse) / +0.9500 (physical)")
    print("  A restart would be strongly NEGATIVE. Positive means refinement gains more")
    print("  when the proposal is better -- directional information, i.e. a gradient.")

    out = Path("report/loss_defects/freeform_vs_zernike.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
