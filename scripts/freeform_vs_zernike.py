"""Freeform phase: the parameterisation the hardware runner actually uses.

Every attempt in PROCESS.md so far optimised a 135-coefficient Zernike vector. The
shipped `slm_gs_refine` optimises a *freeform* phase grid instead, and attempt 13
showed the Zernike conclusions are ROI-conditional, so they cannot be transferred to
freeform unexamined. This closes that gap.

No new architecture is needed: `ZernikeAmpModel.forward` validates only that its input
is `(B, 1, g, g)` at the basis resolution, and `measured = complex(phase_cos, phase_sin)`
*is* the input phasor. So any phase -- including a freeform one -- can be pushed through
the model. (An earlier note in this file claimed freeform needed a new forward model;
that was wrong.)

Two things freeform changes relative to the Zernike path:

* **GS no longer needs a projection bottleneck.** Attempts 10-14 all ran
  `fit_zernike(gs.phase, n_max)`, and attempt 12 found that projection is actively
  harmful at n_max=20 (it lands *below* flat). In freeform the GS pupil phase is used
  directly, so that failure mode disappears.
* **The parameterisation is far richer** -- a 24x24 grid upsampled to 64x64 is 576 DOF
  against Zernike's 135 -- so it can represent square far fields that low-order
  Zernike cannot. The repo's own anti-pattern table says as much.

Arms, all scored on the independent sim:
  gs_zernike    GS -> fit_zernike -> Zernike gradient design   (the old path)
  gs_freeform   GS pupil phase used directly, no projection
  grad_freeform freeform gradient design from a random start
  gs_fm_refine  freeform gradient refinement started from gs_freeform

The aperture is masked explicitly. `forward` uses the input phasor raw, so a phase-only
pupil would have unit amplitude *outside* the illuminated disc, whereas the sim
illuminates a finite disc. Masking to the inscribed circle makes the two agree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\Projects\TIFO\AO-shaping\src")))
sys.path.insert(0, str(Path(r"D:\Projects\TIFO\AO-shaping\scripts")))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton  # noqa: E402
from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem  # noqa: E402
from ao_shaping.utils.image.target.metrics import rms_pib_terms  # noqa: E402
from ao_shaping.utils.wavefront.zernike_calc import (  # noqa: E402
    ZernikeGenerator,
    fit_zernike,
)
from ml.zernike.losses import LossConfig, composite_loss, roi_mask  # noqa: E402
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402

GRID, N_MAX, PADDING = 64, 15, 12
PIXEL_UM, GS_ITERS, STEPS, DRAWS = 8.0, 40, 60, 8
COARSE = 24  # slm_gs_refine's default --phase-grid
PANEL_H, PANEL_W, DISC_R = 1200, 1920, 450
SIZE_FRACS = [0.25, 0.375, 0.50]
ASPECTS = [1.0, 4.0 / 3.0, 1.5]


def inscribed_mask(grid: int) -> torch.Tensor:
    """Unit amplitude inside the inscribed circle, zero outside.

    Without this the freeform pupil has unit amplitude everywhere while the sim
    illuminates a finite disc, so model and evaluator disagree off-aperture.
    """
    r = grid / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(grid, dtype=torch.float32) + 0.5,
        torch.arange(grid, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    return (((ys - r) ** 2 + (xs - r) ** 2) <= r * r).float()[None, None]


def square_amplitude(grid: int, size_frac: float, aspect: float) -> np.ndarray:
    amp = np.zeros((grid, grid), dtype=np.float64)
    hw, hh = int(size_frac * grid * aspect / 2), int(size_frac * grid / 2)
    c = grid // 2
    amp[c - hh : c + hh, c - hw : c + hw] = 1.0
    return amp


def sim_far_field(phase: np.ndarray) -> np.ndarray:
    """Independent sim far field for an arbitrary pupil phase (no Zernike involved)."""
    pupil = np.asarray(phase, dtype=np.float64)
    if pupil.shape != (GRID, GRID):
        raise ValueError(f"expected {(GRID, GRID)}, got {pupil.shape}")
    # Outside the disc there is no light, not zero phase.
    r = GRID / 2.0
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    disc = ((ys + 0.5 - r) ** 2 + (xs + 0.5 - r) ** 2) <= r * r
    pupil = np.where(disc, pupil, 0.0)
    cy, cx = PANEL_H // 2, PANEL_W // 2
    panel = np.zeros((PANEL_H, PANEL_W), dtype=np.float64)
    panel[cy - GRID // 2 : cy + GRID // 2, cx - GRID // 2 : cx + GRID // 2] = pupil
    s = SimPibSystem(
        slm_shape=(PANEL_H, PANEL_W), ccd_res=(512, 512), beam_w0=float(DISC_R),
        noise_adu=0.0, seed=0, far_field_padding=4, far_field_window=1024,
    )
    s.set_phase_rad(panel)
    crop = np.asarray(s.far_field(), dtype=np.float64)[:GRID, :GRID]
    if not np.all(np.isfinite(crop)):
        raise RuntimeError("non-finite far field")
    return crop


def score(phase: np.ndarray, size_frac: float, aspect: float) -> float:
    crop = sim_far_field(phase)
    pib, uni = rms_pib_terms(
        crop, (GRID / 2, GRID / 2), "rectangle", size_frac * GRID, aspect
    )
    return float(pib + uni)


def gs_phase(size_frac: float, aspect: float) -> np.ndarray:
    """GS pupil phase, used directly -- no Zernike projection."""
    gs = gerchberg_saxton(
        np.ones((GRID, GRID), dtype=np.float64),
        square_amplitude(GRID, size_frac, aspect),
        iterations=GS_ITERS, cell_spacing=PIXEL_UM * 1e-6, propagation="fft",
    )
    return np.asarray(gs.phase, dtype=np.float64)


def _model() -> ZernikeAmpModel:
    torch.manual_seed(0)
    return ZernikeAmpModel(
        ZernikeAmpConfig(
            n_max=N_MAX, grid=GRID, far_field_padding=PADDING, normalization="peak"
        )
    )


def model_far_field(model: ZernikeAmpModel, phase: torch.Tensor) -> torch.Tensor:
    """Learned-model far field for a freeform phase, with the aperture masked."""
    phasor = torch.polar(torch.ones_like(phase), phase) * inscribed_mask(GRID)
    return model(phasor.real, phasor.imag)


def _resize(phase, size: int, mode: str) -> np.ndarray:
    t = torch.as_tensor(np.asarray(phase, dtype=np.float32))[None, None]
    kw = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(t, size=(size, size), mode=mode, **kw)[0, 0].numpy()


def gradient_freeform(init_coarse, model, target, mask, steps: int) -> np.ndarray:
    coarse = torch.nn.Parameter(
        torch.as_tensor(np.asarray(init_coarse, np.float32))[None, None]
    )
    opt = torch.optim.AdamW([coarse], lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        up = F.interpolate(coarse, size=(GRID, GRID), mode="bilinear", align_corners=False)
        loss = composite_loss(
            model_far_field(model, up), target, mask, LossConfig(w_mse=1.0)
        )["_mean_total"]
        loss.backward()
        opt.step()
    with torch.no_grad():
        up = F.interpolate(coarse, size=(GRID, GRID), mode="bilinear", align_corners=False)
    return up[0, 0].numpy().astype(np.float64)


def zernike_path(gs_p, target, mask, size_frac, aspect) -> float:
    """GS -> fit_zernike -> Zernike gradient design: the old attempts 10-14 path."""
    m = _model()
    with torch.no_grad():
        m.coefficients.copy_(
            torch.as_tensor(fit_zernike(gs_p, n_max=N_MAX)[1:].astype(np.float32))
        )
    opt = torch.optim.AdamW(m.parameters(), lr=0.02)
    for _ in range(STEPS):
        opt.zero_grad()
        composite_loss(m.correction_far_field(), target, mask, LossConfig(w_mse=1.0))[
            "_mean_total"
        ].backward()
        opt.step()
    gen = ZernikeGenerator((GRID, GRID), n_orders=N_MAX)
    return score(np.nan_to_num(gen.generate_noll(m.coefficients_array())), size_frac, aspect)


def main() -> None:
    print("=" * 100)
    print("FREEFORM vs ZERNIKE: does the projection bottleneck matter?")
    print("=" * 100)
    print(f"grid={GRID} n_max={N_MAX} coarse={COARSE} gs_iters={GS_ITERS} "
          f"steps={STEPS} draws={DRAWS}")

    model = _model()
    rows: list[dict] = []
    for size_frac in SIZE_FRACS:
        for aspect in ASPECTS:
            mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle",
                            size_frac * GRID, aspect)
            target = torch.as_tensor(
                square_amplitude(GRID, size_frac, aspect), dtype=torch.float32
            )[None, None]
            gs_p = gs_phase(size_frac, aspect)
            for d in range(DRAWS):
                rng = np.random.default_rng(77000 + d)
                rows.append({
                    "size_frac": size_frac, "aspect": aspect, "draw": d,
                    "flat": score(np.zeros((GRID, GRID)), size_frac, aspect),
                    "gs_zernike": zernike_path(gs_p, target, mask, size_frac, aspect),
                    "gs_freeform": score(gs_p, size_frac, aspect),
                    "grad_freeform": score(
                        gradient_freeform(
                            rng.normal(0.0, 0.6, (COARSE, COARSE)), model, target, mask, STEPS
                        ), size_frac, aspect,
                    ),
                    "gs_fm_refined": score(
                        gradient_freeform(
                            _resize(gs_p, COARSE, "bilinear"), model, target, mask, STEPS
                        ), size_frac, aspect,
                    ),
                })

    keys = ("flat", "gs_zernike", "gs_freeform", "grad_freeform", "gs_fm_refined")
    cells = []
    print(f"\n{'ROI':>14}" + "".join(f"{k:>14}" for k in keys))
    for size_frac in SIZE_FRACS:
        for aspect in ASPECTS:
            sel = [r for r in rows if r["size_frac"] == size_frac
                   and abs(r["aspect"] - aspect) < 1e-9]
            mean = {k: float(np.mean([r[k] for r in sel])) for k in keys}
            cells.append(mean)
            print(f"{size_frac:>7.3f}x{aspect:<6.3f}" + "".join(f"{mean[k]:>14.4f}" for k in keys))

    def report(label, a, b):
        d = np.array([c[a] - c[b] for c in cells])
        print(f"  {label:<34} mean={d.mean():+.4f}  positives={int((d > 0).sum())}/{len(d)}")

    print("\n--- paired across the 9 ROI cells ---")
    report("GS_freeform - GS_zernike", "gs_freeform", "gs_zernike")
    report("grad_freeform - GS_freeform", "grad_freeform", "gs_freeform")
    report("GS_freeform - flat", "gs_freeform", "flat")
    report("grad_freeform - flat", "grad_freeform", "flat")

    def rank(a):
        return np.argsort(np.argsort(a)).astype(float)

    x = np.array([c["gs_freeform"] - c["flat"] for c in cells])
    y = np.array([c["gs_fm_refined"] - c["gs_freeform"] for c in cells])
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    sp = float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))
    print("\n--- restart hypothesis in freeform space ---")
    print(f"  spearman(GS_freeform - flat, refinement delta) = {sp:+.4f}")
    print("  Zernike space gave -0.8667 (mse) / -0.9167 (physical)")

    out = Path("report/loss_defects/freeform_vs_zernike.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()