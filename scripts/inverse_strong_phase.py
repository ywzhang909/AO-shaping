"""Does synthetic strong-phase augmentation buy INVERSE capability?

The forward search found no arm that improves forward R2 (best: phasor noise at
+0.0036 +/- 0.0045, 1/3 paired -- inside the noise). But the forward direction was never the
broken one: the measured failure is that **neither model can be inverted**, and the diagnosis
is that the corpus only ever shows mild bench aberrations mapping to broad spots, so a GS
proposal is out of distribution.

C4 (`+strong 25%`) is the arm built for exactly that, and it costs -0.0211 R2 in the forward
direction -- which is the expected price of deliberately adding out-of-distribution data, not
a failure of the idea. So it is judged on the axis it was built for.

Compared here:
    A0  incumbent, real corpus only
    C4  real corpus + 25% synthetic strong phases
    C5  real corpus + 25% synthetic strong phases + phasor noise

Padding is the trap that has voided three separate results in this project, so it is stated
rather than inherited: everything -- training targets, the model, and the evaluator -- runs at
the SAME padding, and the GS proposal the inverse loop starts from is regenerated at that same
padding by :func:`gs_phase_at_scale`. A GS phase designed at padding 1 and scored at padding
12 reads as worse than flat (measured: 0.8103 vs 1.2918) for reasons that have nothing to do
with the model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from compare_unet_baseline import _peak_normalise, _select_records  # noqa: E402
from forward_search import (  # noqa: E402
    APERTURE_R,
    BEAM_W0,
    CoeffForward,
    _aperture,
    fwd,
    train_arm,
    Arm,
)
from ml.hwdataset import (  # noqa: E402
    HwPhaseImageDataset,
    MaterialiserConfig,
    build_hw_index,
)
from ml.zernike import augment as aug  # noqa: E402
from ml.zernike import inverse_design as inv  # noqa: E402
from ml.zernike.losses import LossConfig, composite_loss, roi_mask  # noqa: E402
from ao_shaping.utils.wavefront.zernike_calc import fit_zernike  # noqa: E402
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402
from ml.zernike.train_amp import AmpTrainConfig, collect_split  # noqa: E402

GRID, N_MAX, PADDING = inv.GRID, 20, 1
SEEDS = [0, 1, 2]
EPOCHS, BS, LR = 60, 64, 0.02
SIZE_FRAC, ASPECT = inv.SIZE_FRAC, inv.ASPECT


def inverse_score(model, basis, start, target, mask, device) -> float:
    wrapped = CoeffForward(model, basis, stacked=False).to(device).eval()
    theta = torch.nn.Parameter(torch.as_tensor(start, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([theta], lr=LR)
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        composite_loss(wrapped(theta), target, mask, LossConfig(w_mse=1.0))["_mean_total"].backward()
        opt.step()
    return inv.score_coefficients(
        theta.detach().cpu().numpy()[0], SIZE_FRAC, ASPECT, n_max=N_MAX, padding=PADDING
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 104)
    print("DOES STRONG-PHASE AUGMENTATION BUY INVERSE CAPABILITY?")
    print("=" * 104)
    print(f"seeds={SEEDS} padding={PADDING} (training, evaluator AND the GS target all share it)")

    # The library's VALIDATED GS phase. A hand-rolled scale-matched GS was tried first and is
    # deleted: it used the pupil phase in place of a back-propagation, so it was not a
    # Gerchberg-Saxton iteration at all, and it scored 0.7761 against a flat reference of
    # 1.3711 while the surrounding print claimed the opposite. A wrong propagator that still
    # emits a plausible-looking number is worse than no propagator.
    #
    # Padding is therefore 1 for this script -- the scale GS actually designs on. The cost is
    # that forward R2 at padding 1 is lower than at the family's optimum 12 (0.53-0.74 vs
    # ~0.88), so these are NOT comparable with forward_search.json. That is the trade: a
    # self-consistent inverse measurement, stated rather than hidden.
    gs = inv.gs_phase(SIZE_FRAC, ASPECT, iterations=40)
    flat = inv.score_phase(np.zeros((GRID, GRID)), SIZE_FRAC, ASPECT, padding=PADDING)
    gs_score = inv.score_phase(gs, SIZE_FRAC, ASPECT, padding=PADDING)
    gs_coeff = fit_zernike(gs, n_max=N_MAX)[1:].astype(np.float32)
    print(f"\nreferences at padding={PADDING}: flat={flat:.4f}  library GS={gs_score:.4f}")
    if gs_score <= flat:
        print("  WARNING: GS does not beat flat here -- the target scale is wrong and any")
        print("  inverse number below would be meaningless. Stopping rather than reporting it.")
        raise SystemExit(1)
    print(f"  GS beats flat by {gs_score - flat:+.4f} -- the target is aimed correctly")

    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=["slm_zernike_shaping"])
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=GRID), use_cache=True)
    cfg0 = AmpTrainConfig(
        families=("slm_zernike_shaping",), n_max=N_MAX, grid=GRID, epochs=EPOCHS, lr=LR,
        batch_size=BS, max_train=512, max_val=128, beam_samples=16, use_wandb=False,
        save_checkpoint=False,
    )
    basis = ZernikeAmpModel(ZernikeAmpConfig(n_max=N_MAX, grid=GRID)).basis.detach().clone().to(device)
    n_modes = basis.shape[0]
    target = inv.target_tensor(GRID, SIZE_FRAC, ASPECT).to(device)
    mask = roi_mask(
        (GRID, GRID), (GRID / 2, GRID / 2), "rectangle", SIZE_FRAC * GRID, ASPECT
    ).to(device)

    arms = {
        "A0 incumbent": Arm("A0"),
        "C4 +strong 25%": Arm("C4", strong_frac=0.25),
        "C5 +noise+strong": Arm("C5", noise_std=0.05, strong_frac=0.25),
    }
    rows = []
    for name, arm in arms.items():
        for seed in SEEDS:
            cfg = AmpTrainConfig(**{**cfg0.__dict__, "seed": seed})
            torch.manual_seed(seed)
            tr, _ = _select_records(dataset, cfg)
            t = collect_split(dataset, tr, device)
            x = torch.cat([t["phase_cos"], t["phase_sin"]], dim=1)
            y = _peak_normalise(t["target"].clone())
            strong = None
            if arm.strong_frac > 0:
                strong = aug.strong_phase_pairs(
                    300, grid=GRID, beam_w0=BEAM_W0, far_field_padding=PADDING,
                    aperture_radius=APERTURE_R, n_max=N_MAX, seed=seed + 99,
                )
            model = train_arm(arm, x, y, seed, device, strong)
            rng = np.random.default_rng(4242 + seed)
            row = {
                "arm": name, "seed": seed,
                "inv_rand": inverse_score(
                    model, basis, rng.normal(0.0, 0.6, (1, n_modes)).astype(np.float32),
                    target, mask, device,
                ),
                "inv_gs": inverse_score(model, basis, gs_coeff[None], target, mask, device),
            }
            rows.append(row)
            print(f"  {name:<20} seed={seed} inv(rand)={row['inv_rand']:.4f}  "
                  f"(flat={flat:.4f}, GS={gs_score:.4f})")

    print("\n" + "=" * 104)
    print(f"means over {len(SEEDS)} seeds -- flat={flat:.4f}")
    print("=" * 104)
    base = {r["seed"]: r["inv_rand"] for r in rows if r["arm"] == "A0 incumbent"}
    print(f"{'arm':<22}{'inv(rand)':>11}{'vs flat':>10}{'dR2 vs A0':>12}{'pos':>7}")
    for name in arms:
        sel = [r for r in rows if r["arm"] == name]
        d = np.array([r["inv_rand"] - base[r["seed"]] for r in sel])
        m = float(np.mean([r["inv_rand"] for r in sel]))
        print(f"{name:<22}{m:>11.4f}{m - flat:>+10.4f}{d.mean():>+12.4f}"
              f"{int((d > 0).sum()):>4}/{len(d)}")

    out = ROOT / "report" / "loss_defects" / "inverse_strong_phase.json"
    out.write_text(json.dumps(
        {"flat": flat, "gs": gs_score, "padding": PADDING, "rows": rows}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()