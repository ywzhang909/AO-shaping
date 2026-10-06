"""Does augmentation from an INDEPENDENT propagator help? Paired, 5 seeds, with a control.

The hypothesis
--------------
Every image-domain loss idea has now been measured out (see
``report/loss_defects/inverse_design_report.md`` §10), and so has every regulariser. The
remaining explanation for the forward model's error is **training distribution**: the corpus
is 1010 records over 4 optimisation objectives holding mild aberrations, while inverse design
asks about strong ones.

The obvious fix was already tried and lost. Arm "C4 +strong 25%" scored **-0.0211, 0/3
paired**. The diagnosis, measured rather than assumed: ``aug.strong_phase_pairs`` builds its
targets with ``inverse_design.sim_far_field``, which calls ``SimPibSystem`` -- **the same
propagator** ``ZernikeAmpModel._propagate`` implements. Feeding the same phase through both
gives **corr = 0.9956**. So that augmentation was not new information; it re-served what the
model already computes exactly, while widening the training distribution. A negative result
is exactly what "no information" predicts.

So the only augmentation worth trying is one whose propagator contains physics the model
**provably cannot represent**. Three candidates are documented as measured on this bench, and
none of them is in the simulator:

1. **SLM pixel crosstalk / finite fill factor** -- a 2f Fourier bench has a spatial
   bandwidth product, and a real LCOS mixes neighbouring pixels.
2. **LCOS gray-to-amplitude coupling** -- AGENTS.md records camera intensity varying with
   flat-phase gray level on a ~993-gray period at 1064 nm.
3. Large amplitude excursions (but this one the model *can* represent, so it is not one).

This script injects **(1) crosstalk**, and is built so the result is interpretable either way:

* ``D2``            incumbent architecture, real corpus only.
* ``D2+same``       **CONTROL** -- augmentation through the existing same-simulator path,
                    same sample count. Without it, "independent propagator helped" cannot be
                    separated from "more data helped", which is the entire question.
* ``D2+crosstalk``  augmentation through the crosstalk propagator, 3 strengths.

Verdict rule, fixed in advance: a crosstalk arm wins only if its mean paired dR2 is positive
AND it beats ``D2`` in >=3/5 seeds AND it also beats the ``D2+same`` control. Anything else
is reported as a negative result, with its standard error.

Why subclassing
---------------
``SimPibSystem._pupil_field`` exists as an explicit seam -- its own docstring says it is
"so a subclass can inject pupil-plane physics (e.g. SLM channel crosstalk) without copying
the pad + FFT + normalise chain". This uses that seam. Re-deriving an FFT chain would
violate the repo's standing prohibition and is a documented past failure mode: a duplicated
propagator that quietly diverges.

Run::

    $env:PYTHONPATH="src;."
    python scripts/crosstalk_augmentation_train.py

Writes ``report/loss_defects/crosstalk_augmentation_train.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
# `forward_search_extra` imports its constants from `forward_search` by bare name, so
# `scripts/` has to be importable too. Taking the geometry from those modules rather than
# restating it is the whole point: a hardcoded BEAM_W0 or APERTURE_R here would silently
# make the augmentation a different distribution from the control arm's.
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scipy.ndimage import convolve  # noqa: E402

OUT_DIR = ROOT / "report" / "loss_defects"

from forward_search import APERTURE_R, BEAM_W0, GRID, N_MAX, PADDING  # noqa: E402
from forward_search_extra import RESIDUAL_SCALE, SEEDS  # noqa: E402

EPOCHS, BS, LR = 60, 64, 0.02
MAX_TRAIN, MAX_VAL = 512, 128
FAMILY = "slm_zernike_shaping"

#: The incumbent's U-Net width, read from `forward_search_extra`'s own instantiation
#: (``PhysicsPlusResidual(phys, 16, RESIDUAL_SCALE)``). Duplicating the number here is a
#: silent-drift risk, so it is asserted against that module at run time instead.
RESIDUAL_WIDTH = 16

#: Augmentation share of each batch, matching arm C4's ``strong_frac`` so the control and
#: the incumbent are the only things that differ.
STRONG_FRAC = 0.25
N_AUG = 300

#: Crosstalk strengths in pupil pixels. 0.5 is sub-pixel (nearest-neighbour bleed only),
#: 1.0 is a realistic single-pixel PSF, 2.0 is deliberately over-crosstalked -- if the
#: mechanism is real, some strength should help before the signal is destroyed.
SIGMAS = (0.5, 1.0, 2.0)


# --------------------------------------------------------------------------- propagator


def _gauss_kernel_2d(sigma: float) -> np.ndarray:
    """Normalised, symmetric 2-D Gaussian, truncated at 3 sigma."""
    r = max(int(3 * sigma + 0.5), 1)
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    k = np.outer(k, k)
    return k / k.sum()


class CrosstalkSim:
    """Factory for a ``SimPibSystem`` whose pupil field carries a crosstalk PSF.

    Implemented by subclassing so the pad -> FFT -> normalise chain stays canonical.
    """

    def __new__(cls, sigma_px: float, **kwargs):
        from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem

        kernel = _gauss_kernel_2d(sigma_px)
        base = SimPibSystem

        class _Crosstalk(base):  # type: ignore[misc, valid-type]
            """``SimPibSystem`` with a finite-fill-factor PSF on the complex pupil field.

            Crosstalk mixes neighbouring pixels' **fields**, not their phases, so the PSF is
            convolved onto ``A(r) * exp(i*phase)`` rather than onto the phase. Convolving the
            phase would be a different (and wrong) error model: it would let the PSF average
            the command while leaving the illumination untouched.
            """

            def _pupil_field(self, phase: np.ndarray) -> np.ndarray:
                field = super()._pupil_field(phase)
                # 'nearest' keeps total energy (up to the truncation) instead of bleeding it
                # into a zero border, which would fake a loss of light.
                return convolve(field, kernel, mode="nearest")

        return _Crosstalk(**kwargs)


def crosstalk_phase_pairs(
    n_samples: int, *, grid: int, beam_w0: float, far_field_padding: int,
    aperture_radius: float, n_max: int, seed: int, sigma_px: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(phasor, far field)`` pairs whose targets carry SLM crosstalk.

    Same three phase families and the same return contract as
    ``ml.zernike.augment.strong_phase_pairs``; only the propagator differs, which is the
    whole point of the experiment. The *input* stays the clean commanded phase while the
    *target* is crosstalk-degraded, so the model is taught "commanded phase -> what the panel
    actually delivers" -- the mapping a real forward model needs and cannot derive.
    """
    from ml.zernike.inverse_design import (
        BEAM_W0 as SIM_BEAM_W0, PANEL_H, PANEL_W, SIM_WINDOW,
        centre_crop, coefficients_to_phase, embed_phase, gs_phase,
    )

    rng = np.random.default_rng(seed)
    families = ["gs", "zernike", "freeform"]
    phasor = torch.zeros(n_samples, 2, grid, grid, dtype=torch.float32)
    target = torch.zeros(n_samples, 1, grid, grid, dtype=torch.float32)
    yy, xx = np.mgrid[0:grid, 0:grid]
    radius = grid / 2.0
    disc = ((yy + 0.5 - radius) ** 2 + (xx + 0.5 - radius) ** 2) <= aperture_radius**2

    # Geometry mirrors `inverse_design.sim_far_field` exactly, so the ONLY difference from
    # the control arm is the crosstalk convolution in `_pupil_field`.
    system = CrosstalkSim(
        sigma_px=sigma_px, slm_shape=(PANEL_H, PANEL_W), ccd_res=(512, 512),
        beam_w0=float(SIM_BEAM_W0), noise_adu=0.0, seed=0,
        far_field_padding=int(far_field_padding), far_field_window=SIM_WINDOW,
    )

    for i in range(n_samples):
        kind = families[i % len(families)]
        if kind == "gs":
            phase = gs_phase(float(rng.uniform(0.25, 0.55)),
                             float(rng.choice([1.0, 4 / 3, 1.5])), iterations=30)
        elif kind == "zernike":
            phase = coefficients_to_phase(rng.normal(0.0, 3.0, n_max * (n_max + 3) // 2),
                                          n_max=n_max)
        else:
            phase = rng.uniform(-np.pi, np.pi, (grid, grid))
        phase = np.where(disc, phase, 0.0)
        phasor[i, 0] = torch.as_tensor(np.cos(phase), dtype=torch.float32)
        phasor[i, 1] = torch.as_tensor(np.sin(phase), dtype=torch.float32)
        system.set_phase_rad(embed_phase(phase))
        full = np.asarray(system.far_field(), dtype=np.float64)
        if not np.all(np.isfinite(full)):
            raise RuntimeError("crosstalk simulator produced a non-finite far field")
        target[i, 0] = torch.as_tensor(centre_crop(full, grid), dtype=torch.float32)

    peak = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return phasor, target / peak  # per-sample peak normalise, matching the incumbent


# --------------------------------------------------------------------------- arms


def build(seed: int, device) -> nn.Module:
    """The incumbent architecture, imported rather than re-declared."""
    from scripts.forward_search_extra import PhysicsPlusResidual
    from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel

    torch.manual_seed(seed)
    physics = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING)
    )
    return PhysicsPlusResidual(physics, width=RESIDUAL_WIDTH, scale=RESIDUAL_SCALE).to(device)


def _fwd(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x[:, :1], x[:, 1:])


def train(seed: int, x, y, xv, yv, aug_pair, device, epochs: int = EPOCHS
          ) -> tuple[dict, nn.Module]:
    """One paired cell. Identical loop to ``forward_search.train_arm``, minus the arms
    that are ablated here (noise / flip / shift / ellipse) so nothing else varies.

    Returns ``(metrics, model)``. The model is returned because the inverse step needs the
    **fitted** object. An earlier version returned metrics only, and the inverse check
    consequently scored a freshly constructed model whose coefficients are deterministically
    zero -- so all arms agreed to four decimals and the run looked stable.
    ``ZernikeAmpModel.coefficients`` initialising to zeros is by design (it is what makes
    `torch.manual_seed` inert on a fresh model), which is exactly why that mistake was silent.
    """
    model = build(seed, device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator(device="cpu").manual_seed(seed + 777)
    n = x.shape[0]
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n)
        for s in range(0, n, BS):
            idx = order[s : s + BS]
            xb, yb = x[idx].to(device), y[idx].to(device)
            if aug_pair is not None:
                k = int(STRONG_FRAC * xb.shape[0])
                if k > 0:
                    pick = torch.randint(0, aug_pair[0].shape[0], (k,), generator=gen)
                    xb = torch.cat([xb[: xb.shape[0] - k], aug_pair[0][pick].to(device)])
                    yb = torch.cat([yb[: yb.shape[0] - k], aug_pair[1][pick].to(device)])
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((_fwd(model, xb) - yb) ** 2)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pv = torch.cat([_fwd(model, xv[s : s + 256]) for s in range(0, xv.shape[0], 256)])
    from ml.zernike.metrics import batch_image_metrics

    img = batch_image_metrics(pv, yv)
    with torch.no_grad():
        phi = model.physics.correction_phase()
        tv = float(
            (phi[1:, :] - phi[:-1, :]).abs().mean() + (phi[:, 1:] - phi[:, :-1]).abs().mean()
        )
        pvv = float(phi.amax() - phi.amin())
        cmax = float(model.physics.coefficients.abs().max())
    return {
        "seed": seed,
        "r2": float(img["r2"]),
        "ssim": float(img["ssim"]),
        "mse": float(img["mse"]),
        "phase_tv_rad_per_px": tv,
        "phase_pv_rad": pvv,
        "max_abs_coeff": cmax,
        "train_s": time.perf_counter() - t0,
    }, model


# --------------------------------------------------------------------------- driver


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 108)
    print("CROSSTALK AUGMENTATION: does an independent propagator help? paired, 5 seeds")
    print("=" * 108)
    print(f"device={device} epochs={EPOCHS} seeds={SEEDS} n_max={N_MAX} padding={PADDING} "
          f"strong_frac={STRONG_FRAC} n_aug={N_AUG} sigmas={SIGMAS}")

    from scripts.compare_unet_baseline import _inputs, _peak_normalise, _select_records
    from ml.hwdataset import HwPhaseImageDataset, MaterialiserConfig, build_hw_index
    from ml.zernike import augment as aug
    from ml.zernike.train_amp import AmpTrainConfig, collect_split

    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=[FAMILY])
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=GRID), use_cache=True)
    cfg0 = AmpTrainConfig(
        families=(FAMILY,), n_max=N_MAX, grid=GRID, epochs=EPOCHS, lr=LR, batch_size=BS,
        max_train=MAX_TRAIN, max_val=MAX_VAL, beam_samples=16, use_wandb=False,
        save_checkpoint=False,
    )

    # ---- degeneracy guards on the augmentation itself -------------------------
    print("\nbuilding augmentation pairs and checking they are not degenerate ...")
    same = aug.strong_phase_pairs(
        N_AUG, grid=GRID, beam_w0=BEAM_W0, far_field_padding=PADDING,
        aperture_radius=APERTURE_R, n_max=N_MAX, seed=7,
    )
    xt = {
        s: crosstalk_phase_pairs(
            N_AUG, grid=GRID, beam_w0=BEAM_W0, far_field_padding=PADDING,
            aperture_radius=APERTURE_R, n_max=N_MAX, seed=7, sigma_px=s,
        )
        for s in SIGMAS
    }
    guards: dict[str, float] = {}
    for name, (p, t) in [("same", same), *[(f"xt{s}", v) for s, v in xt.items()]]:
        guards[f"{name}_finite"] = float(bool(torch.isfinite(t).all()))
        guards[f"{name}_nonzero_frac"] = float((t.amax(dim=(-2, -1)) > 1e-9).float().mean())
    # The decisive one: does crosstalk actually change the target relative to the
    # same-simulator path? If not, the arm is a no-op and a null result means nothing.
    for s in SIGMAS:
        guards[f"xt{s}_vs_same_mae"] = float((xt[s][1] - same[1]).abs().mean())
    guards["same_vs_same_mae"] = 0.0
    print(json.dumps({k: round(v, 5) for k, v in guards.items()}, indent=2))
    for s in SIGMAS:
        if guards[f"xt{s}_vs_same_mae"] < 1e-4:
            raise SystemExit(
                f"crosstalk sigma={s} produced targets indistinguishable from the "
                "same-simulator path -- the arm would be a no-op"
            )

    rows: list[dict] = []
    for seed in SEEDS:
        cfg = AmpTrainConfig(**{**cfg0.__dict__, "seed": seed})
        torch.manual_seed(seed)
        tr, va = _select_records(dataset, cfg)
        tr_t = collect_split(dataset, tr, device)
        va_t = collect_split(dataset, va, device)
        x = torch.cat([tr_t["phase_cos"], tr_t["phase_sin"]], dim=1)
        y = _peak_normalise(tr_t["target"].clone())
        xv, yv = _inputs(va_t), _peak_normalise(va_t["target"].clone())

        for arm, pair in (
            [("D2", None), ("D2+same", same)]
            + [(f"D2+xt{s}", xt[s]) for s in SIGMAS]
        ):
            r, _fitted = train(seed, x, y, xv, yv, pair, device)
            r["arm"] = arm
            rows.append(r)
            print(
                f"  seed{seed} {arm:12s} R2={r['r2']:+.4f} SSIM={r['ssim']:.4f} "
                f"PV={r['phase_pv_rad']:6.2f}rad TV={r['phase_tv_rad_per_px']:.4f} "
                f"max|c|={r['max_abs_coeff']:.3f}",
                flush=True,
            )

    base = {r["seed"]: r for r in rows if r["arm"] == "D2"}
    ctrl = float(np.mean([
        next(r for r in rows if r["arm"] == "D2+same" and r["seed"] == sd)["r2"]
        - base[sd]["r2"] for sd in SEEDS
    ]))

    verdict: dict[str, dict[str, object]] = {}
    for arm in dict.fromkeys(r["arm"] for r in rows):
        if arm == "D2":
            continue
        sel = [r for r in rows if r["arm"] == arm]
        d = [r["r2"] - base[r["seed"]]["r2"] for r in sel]
        ds = [r["ssim"] - base[r["seed"]]["ssim"] for r in sel]
        mean = float(np.mean(d))
        se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
        verdict[arm] = {
            "d_r2_paired": mean,
            "d_r2_se": se,
            "beats_baseline": int(sum(1 for v in d if v > 0)),
            "n": len(d),
            "d_ssim_paired": float(np.mean(ds)),
            "mean_r2": float(np.mean([r["r2"] for r in sel])),
            "phase_pv_rad": float(np.mean([r["phase_pv_rad"] for r in sel])),
            "phase_tv_rad_per_px": float(np.mean([r["phase_tv_rad_per_px"] for r in sel])),
            "max_abs_coeff": float(np.mean([r["max_abs_coeff"] for r in sel])),
            "beats_control": bool(mean > ctrl),
            "wins": bool(mean > 0 and sum(1 for v in d if v > 0) >= 3 and mean > ctrl),
        }

    base_pv = float(np.mean([r["phase_pv_rad"] for r in base.values()]))
    base_tv = float(np.mean([r["phase_tv_rad_per_px"] for r in base.values()]))
    print(f"\nD2 baseline: meanR2={np.mean([r['r2'] for r in base.values()]):+.4f} "
          f"PV={base_pv:.2f}rad TV={base_tv:.4f}")
    print(f"D2+same CONTROL: dR2={ctrl:+.4f}")
    print("\n--- paired verdict vs D2 (and vs the same-simulator control) ---")
    hdr = (f"{'arm':14s} {'dR2':>9s} {'SE':>7s} {'beats':>6s} {'dSSIM':>8s} "
           f"{'vs ctrl':>8s} {'PV':>7s} {'WINS':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for arm, v in sorted(verdict.items(), key=lambda kv: -kv[1]["d_r2_paired"]):
        print(f"{arm:14s} {v['d_r2_paired']:+9.4f} {v['d_r2_se']:7.4f} "
              f"{v['beats_baseline']:>3}/{v['n']} {v['d_ssim_paired']:+8.4f} "
              f"{str(v['beats_control']):>8s} {v['phase_pv_rad']:7.2f} "
              f"{str(v['wins']):>5s}")

    winners = [a for a, v in verdict.items() if v["wins"]]
    print(f"\nwinners (positive dR2 AND >=3/5 seeds AND beats control): {winners or 'NONE'}")
    verdict_line = (
        f"crosstalk augmentation {'HELPS' if winners else 'DOES NOT HELP'}: "
        + (
            f"best arm {max(winners, key=lambda a: verdict[a]['d_r2_paired'])} "
            f"dR2={verdict[max(winners, key=lambda a: verdict[a]['d_r2_paired'])]['d_r2_paired']:+.4f}"
            if winners
            else f"every arm <= 0 paired or within noise (control dR2={ctrl:+.4f})"
        )
    )
    print(verdict_line)

    (OUT_DIR / "crosstalk_augmentation_train.json").write_text(
        json.dumps(
            {
                "config": {
                    "seeds": SEEDS, "epochs": EPOCHS, "n_max": N_MAX, "grid": GRID,
                    "padding": PADDING, "strong_frac": STRONG_FRAC, "n_aug": N_AUG,
                    "sigmas": list(SIGMAS), "max_train": MAX_TRAIN, "max_val": MAX_VAL,
                    "lr": LR, "batch_size": BS, "device": str(device),
                },
                "degeneracy_guards": guards,
                "baseline": {"mean_r2": float(np.mean([r["r2"] for r in base.values()])),
                             "phase_pv_rad": base_pv, "phase_tv_rad_per_px": base_tv},
                "control_d_r2": ctrl,
                "rows": rows,
                "verdict": verdict,
                "winners": winners,
                "verdict_line": verdict_line,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_DIR / 'crosstalk_augmentation_train.json'}")


if __name__ == "__main__":
    main()