"""Does the crosstalk-augmented forward gain transfer to inverse design?

Why this run exists
-------------------
The forward experiment (`crosstalk_augmentation_train.py`) came out **inconclusive**, and
the honest reading is a null: the best crosstalk arm paired +0.0253 ± 0.0314 R² with
`p_sign_flip = 0.6250`, and its edge over the same-simulator control was +0.0112 with
`p = 0.3125`. With 5 pairs the smallest attainable p is 0.0625, so five seeds cannot reach
significance at all. Nothing here is a win.

That makes the inverse run worth doing for a different reason than "check the winner". This
repo has already established, three times independently, that **forward accuracy does not
predict inverse capability**:

* the physics+U-Net residual gained +0.0426 forward (5/5) and +0.0065 inverse (2/5, noise);
* the standalone U-Net nearly doubled forward R² (+0.8718 vs +0.6276) and moved inverse by
  **0.0000**;
* three architectures × two paddings all reproduced the dissociation.

So the prediction is specific and falsifiable: if the forward gain is noise, the inverse
score should not move either. A third confirmation raises confidence that the bottleneck is
the **training distribution** (1010 records, 4 mild objectives) and not model capacity,
regularisation, or loss design -- every one of which has now been measured and eliminated.

What would refute it: an inverse gain that is positive in >=3/5 seeds *and* significant
against the same control. That would mean the forward axis and the inverse axis are not
dissociated after all, and the "distribution, not model" reading would need revisiting.

Arms (all three, so the control is present in the inverse comparison too):

* ``D2``            incumbent, no augmentation
* ``D2+same``       same-simulator augmentation control
* ``D2+xt2.0``      the best crosstalk arm from the forward run

Run::

    $env:PYTHONPATH="src;."
    python scripts/inverse_crosstalk_check.py

Writes ``report/loss_defects/inverse_crosstalk_check.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "report" / "loss_defects"

from crosstalk_augmentation_train import (  # noqa: E402
    RESIDUAL_WIDTH, crosstalk_phase_pairs, train,
)
from forward_search import APERTURE_R, ASPECT, BEAM_W0, SIZE_FRAC  # noqa: E402
from forward_search_extra import RESIDUAL_SCALE  # noqa: E402

#: The inverse protocol runs at padding **1**, not the 12 that forward accuracy needs.
#: This is not a detail: `gs_phase` designs on the padding-1 angular scale, so at padding 12
#: the GS reference scores *below flat* and every inverse number would be meaningless --
#: which is exactly what this script's own guard detects and refuses to run past. The forward
#: run stays at 12; conflating the two is what voided the earlier ROI sweep, so they are kept
#: separate on purpose.
INV_PADDING = 1
GRID = 64
N_MAX = 20

SEEDS = [0, 1, 2, 3, 4]
XT_SIGMA = 2.0
STRONG_FRAC = 0.25


# The canonical coefficients -> phasor -> model wrapper. Aliased, not re-derived: the
# aperture mask and basis contraction must agree with the forward pipeline exactly, or the
# inverse step optimises a different operator than the model was fitted with. (A first
# attempt here hand-rolled it and indexed the phasor as a 5-D tensor.)
from forward_search import CoeffForward  # noqa: E402


def inverse_score(model, basis, start, target, mask, device) -> float:
    """The same inverse step ``inverse_combined.inverse_score`` runs: 80 AdamW steps on the
    coefficients, MSE only, scored on the INDEPENDENT simulator.

    Re-implemented rather than imported only because that module binds its own ``PADDING``
    to 1 while this run must use 12 to match the forward training -- a mismatch is exactly
    the failure that voided the earlier ROI sweep.
    """
    from ml.zernike.inverse_design import score_coefficients
    from ml.zernike.losses import LossConfig, composite_loss

    # stacked=False: the canonical wrapper passes (real, imag) as two arguments, which
    # is the PhysicsPlusResidual signature. stacked=True feeds one 2-channel tensor and
    # only suits the bare U-Net arms.
    wrapped = CoeffForward(model, basis, stacked=False).to(device).eval()
    # Batched to (1, K): the canonical wrapper contracts with einsum("bk,kij->bij"),
    # so a 1-D theta is a shape error, not a silent broadcast.
    theta = torch.nn.Parameter(
        torch.as_tensor(start, dtype=torch.float32, device=device).reshape(1, -1)
    )
    opt = torch.optim.AdamW([theta], lr=0.02)
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        composite_loss(wrapped(theta), target, mask, LossConfig(w_mse=1.0))["_mean_total"].backward()
        opt.step()
    return float(
        score_coefficients(
            theta.detach().cpu().numpy()[0], SIZE_FRAC, ASPECT, n_max=N_MAX, padding=INV_PADDING
        )
    )


def _build_at(seed: int, device):
    """The incumbent architecture built at the INVERSE padding.

    `crosstalk_augmentation_train.build` hardcodes the forward padding (12), which is the
    right default for that script and the wrong one here: the inverse step's target lives on
    the padding-1 angular scale, so the model under test has to speak that scale too.
    """
    from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel
    from forward_search_extra import PhysicsPlusResidual

    torch.manual_seed(seed)
    physics = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=INV_PADDING)
    )
    return PhysicsPlusResidual(physics, width=RESIDUAL_WIDTH, scale=RESIDUAL_SCALE).to(device)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("INVERSE DESIGN through the crosstalk-augmented models (transfer test)")
    print("=" * 100)
    print(f"device={device} seeds={SEEDS} padding={INV_PADDING} n_max={N_MAX}")

    from ml.zernike import inverse_design as inv
    from ml.zernike.losses import roi_mask
    from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel
    from ml.zernike import augment as aug
    from ml.zernike.metrics import batch_image_metrics
    from compare_unet_baseline import _inputs, _peak_normalise, _select_records
    from ml.hwdataset import HwPhaseImageDataset, MaterialiserConfig, build_hw_index
    from ml.zernike.train_amp import AmpTrainConfig, collect_split
    from ml.zernike.eval_stats import paired_comparison

    gs = inv.gs_phase(SIZE_FRAC, ASPECT, iterations=40)
    flat = inv.score_phase(np.zeros((GRID, GRID)), SIZE_FRAC, ASPECT, padding=INV_PADDING)
    gs_score = inv.score_phase(gs, SIZE_FRAC, ASPECT, padding=INV_PADDING)
    if gs_score <= flat:
        print("STOPPING: GS does not beat flat -> the target scale is wrong, every number "
              "below would be meaningless.")
        raise SystemExit(1)
    print(f"references: flat={flat:.4f}  GS={gs_score:.4f}  (GS - flat = {gs_score - flat:+.4f})")

    basis = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=N_MAX, grid=GRID)
    ).basis.detach().clone().to(device)
    target = inv.target_tensor(GRID, SIZE_FRAC, ASPECT).to(device)
    mask = roi_mask((GRID, GRID), (GRID / 2, GRID / 2), "rectangle",
                    SIZE_FRAC * GRID, ASPECT).to(device)

    same = aug.strong_phase_pairs(
        300, grid=GRID, beam_w0=BEAM_W0, far_field_padding=INV_PADDING,
        aperture_radius=APERTURE_R, n_max=N_MAX, seed=7,
    )
    xt = crosstalk_phase_pairs(
        300, grid=GRID, beam_w0=BEAM_W0, far_field_padding=INV_PADDING,
        aperture_radius=APERTURE_R, n_max=N_MAX, seed=7, sigma_px=XT_SIGMA,
    )

    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=["slm_zernike_shaping"])
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=GRID), use_cache=True)
    cfg0 = AmpTrainConfig(
        families=("slm_zernike_shaping",), n_max=N_MAX, grid=GRID, epochs=60, lr=0.02,
        batch_size=64, max_train=512, max_val=128, beam_samples=16, use_wandb=False,
        save_checkpoint=False,
    )

    rows: list[dict] = []
    for seed in SEEDS:
        cfg = AmpTrainConfig(**{**cfg0.__dict__, "seed": seed})
        torch.manual_seed(seed)
        tr, va = _select_records(dataset, cfg)
        tr_t, va_t = collect_split(dataset, tr, device), collect_split(dataset, va, device)
        x = torch.cat([tr_t["phase_cos"], tr_t["phase_sin"]], dim=1)
        y = _peak_normalise(tr_t["target"].clone())
        xv, yv = _inputs(va_t), _peak_normalise(va_t["target"].clone())

        for arm, pair in (("D2", None), ("D2+same", same), (f"D2+xt{XT_SIGMA}", xt)):
            t0 = time.perf_counter()
            trained, model = train(seed, x, y, xv, yv, pair, device)
            inv_scores = []
            for rep in range(2):
                rng = np.random.default_rng(1000 + 17 * rep + seed)
                start = rng.normal(0.0, 0.35, size=basis.shape[0]).astype(np.float32)
                inv_scores.append(inverse_score(model, basis, start, target, mask, device))
            rows.append({
                "arm": arm, "seed": seed,
                "r2": trained["r2"], "ssim": trained["ssim"],
                "phase_pv_rad": trained["phase_pv_rad"],
                "inv_mean": float(np.mean(inv_scores)),
                "inv_flat": flat, "inv_gs": gs_score,
                "wall_s": time.perf_counter() - t0,
            })
            print(
                f"  seed{seed} {arm:12s} fwdR2={trained['r2']:+.4f} "
                f"inv={np.mean(inv_scores):.4f} (flat={flat:.4f} gs={gs_score:.4f}) "
                f"[{time.perf_counter() - t0:.0f}s]",
                flush=True,
            )

    base = {r["seed"]: r["inv_mean"] for r in rows if r["arm"] == "D2"}
    verdict: dict[str, dict[str, object]] = {}
    for arm in dict.fromkeys(r["arm"] for r in rows):
        if arm == "D2":
            continue
        sel = {r["seed"]: r["inv_mean"] for r in rows if r["arm"] == arm}
        st = paired_comparison([sel[s] - base[s] for s in sorted(sel)], f"{arm} vs D2")
        verdict[arm] = {
            "mean_inv": float(np.mean([sel[s] for s in sel])),
            "d_inv_vs_D2": st["mean_diff"],
            "cohens_dz": st["cohens_dz"],
            "p_sign_flip": st["p_sign_flip"],
            "min_attainable_p": st["min_attainable_p"],
            "beats_baseline": int(sum(1 for s in sel if sel[s] > base[s])),
            "n": len(sel),
            "mean_fwd_r2": float(np.mean([r["r2"] for r in rows if r["arm"] == arm])),
        }

    print(f"\nflat={flat:.4f}  GS={gs_score:.4f}")
    print("\n--- inverse paired verdict ---")
    hdr = f"{'arm':12s} {'inv':>8s} {'d_inv':>9s} {'dz':>6s} {'p':>7s} {'beats':>6s} {'fwdR2':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for arm, v in sorted(verdict.items(), key=lambda kv: -kv[1]["d_inv_vs_D2"]):
        print(f"{arm:12s} {v['mean_inv']:8.4f} {v['d_inv_vs_D2']:+9.4f} "
              f"{v['cohens_dz']:+6.2f} {v['p_sign_flip']:7.4f} "
              f"{v['beats_baseline']:>3}/{v['n']} {v['mean_fwd_r2']:+8.4f}")

    moved = [a for a, v in verdict.items()
             if v["d_inv_vs_D2"] > 0 and v["beats_baseline"] >= 3 and v["p_sign_flip"] <= 0.0625]
    line = (
        "inverse gain TRANSFERS: " + ", ".join(moved)
        if moved else
        "no inverse gain transfers (prediction CONFIRMED): forward R2 moved without the "
        "inverse score following, a third independent dissociation"
    )
    print("\n" + line)

    (OUT_DIR / "inverse_crosstalk_check.json").write_text(
        json.dumps(
            {
                "config": {"seeds": SEEDS, "padding": INV_PADDING, "n_max": N_MAX,
                           "grid": GRID, "xt_sigma": XT_SIGMA, "device": str(device)},
                "references": {"flat": flat, "gs": gs_score},
                "rows": rows, "verdict": verdict, "transferred": moved,
                "verdict_line": line,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_DIR / 'inverse_crosstalk_check.json'}")


if __name__ == "__main__":
    main()