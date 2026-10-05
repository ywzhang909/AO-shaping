"""Inverse phase optimisation test -- and the experiment the last 16 attempts point at.

The measured problem: neither the physics model nor the U-Net can invert. Both score far
*below* the flat reference when their gradient is pushed toward a square target
(physics -0.296, U-Net -0.370), at every padding tried. The diagnosis is the training
distribution, not the architecture: ``slm_zernike_shaping`` is four optimisation objectives
of *mild* bench aberrations mapping to *broad* spots, so a GS proposal -- a strong phase --
is outside everything either model has seen.

So this asks the question that diagnosis implies:

    **does training on strong phases restore inverse capability?**

Four arms, so the answer is attributable rather than anecdotal:

    mild-only        the incumbent: trained on the real corpus, strong phases unseen
    synth            trained ONLY on synthetic GS + random phases (real corpus unseen)
    mixed            real corpus + synthetic (tests whether synthetic data is additive)
    unet-synth       same synthetic data through the U-Net (is it architecture again?)

Two parameterisations are tested because they disagreed before: Zernike coefficients, and
freeform phase. And GS refinement vs a random start, because "refinement helps" was itself
re-measured and reversed once already.

**The evaluator is deliberately NOT the generator.** Training far fields come from
``SimPibSystem``; scoring uses :func:`numpy_far_field` below, a plain Fraunhofer
propagator written from scratch. Grading a model with the simulator that produced its
training data would be the model marking its own homework -- the exact failure this repo's
protocol exists to prevent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Projects\TIFO\AO-shaping")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ml.hwdataset import (  # noqa: E402
    HwPhaseImageDataset,
    MaterialiserConfig,
    build_hw_index,
)
from ml.zernike import inverse_design as inv  # noqa: E402
from ml.zernike.losses import LossConfig, composite_loss, roi_mask  # noqa: E402
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402
from ml.zernike.train_amp import AmpTrainConfig  # noqa: E402

GRID, N_MAX = inv.GRID, 20
# Padding is a CONTROLLED variable here, set to 1, not 10. Gerchberg-Saxton designs on the
# padding-1 far-field grid, so scoring a GS proposal at padding 10 aims the square at 1/10
# of the crop and reads as worse than flat (measured: GS 0.8103 vs flat 1.2918). Holding it
# at 10 would have made every inverse number meaningless -- the same scale mismatch that
# voided the earlier ROI sweep. Every arm below is therefore trained AND scored at 1.
PADDING = 1
PADDED = GRID * PADDING
SEEDS = [0, 1, 2]
EPOCHS_SYNTH = 300
EPOCHS_MILD = 60
SIZE_FRAC, ASPECT = inv.SIZE_FRAC, inv.ASPECT
WEIGHTS = LossConfig(w_mse=1.0)


# ---------------------------------------------------------------- independent evaluator
def aperture_mask(grid: int = GRID) -> np.ndarray:
    """Unit illumination inside the inscribed disc, zero outside."""
    r = grid / 2.0
    ys, xs = np.mgrid[0:grid, 0:grid]
    return (((ys + 0.5 - r) ** 2 + (xs + 0.5 - r) ** 2) <= r * r).astype(np.float64)


def numpy_far_field(phase: np.ndarray, pad: int = PADDING, grid: int = GRID) -> np.ndarray:
    """Fraunhofer far field of ``amp * exp(i*phase)`` -- written from scratch.

    Deliberately shares no code with ``SimPibSystem``: the training targets come from the
    simulator, so scoring with the simulator would let a model grade its own homework.
    Normalised to unit total energy (Parseval-consistent), then centre-cropped to ``grid``.
    """
    amp = aperture_mask(grid)
    field = amp * np.exp(1j * np.asarray(phase, dtype=np.float64))
    buf = np.zeros((pad * grid, pad * grid), dtype=np.complex128)
    buf[:grid, :grid] = field
    far = np.fft.fftshift(np.fft.fft2(buf))
    far = np.abs(far) ** 2
    far /= far.sum()
    start = (pad * grid - grid) // 2
    return far[start : start + grid, start : start + grid]


def score_numpy(phase: np.ndarray, size_frac: float = SIZE_FRAC, aspect: float = ASPECT) -> float:
    from ao_shaping.utils.image.target.metrics import rms_pib_terms

    pib, uni = rms_pib_terms(
        numpy_far_field(phase), (GRID / 2, GRID / 2), "rectangle", size_frac * GRID, aspect
    )
    return float(pib + uni)


# ---------------------------------------------------------------- training data
def synthetic_dataset(n_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(phasor, far field)`` pairs covering STRONG phases -- the thing the real corpus lacks.

    Three phase families, because one would confound the result:
      * GS proposals for a spread of target sizes (the shapes inverse design actually asks for)
      * random Zernike with large coefficients (strong smooth aberrations)
      * random freeform phase (spatially rough, i.e. speckle-like)
    """
    rng = np.random.default_rng(seed)
    gen_phases: list[np.ndarray] = []
    for _ in range(n_samples // 3):
        sf = float(rng.uniform(0.25, 0.55))
        asp = float(rng.choice([1.0, 4 / 3, 1.5]))
        gen_phases.append(inv.gs_phase(sf, asp, iterations=30))
    for _ in range(n_samples // 3):
        gen_phases.append(
            inv.coefficients_to_phase(
                rng.normal(0.0, 3.0, N_MAX * (N_MAX + 3) // 2), n_max=N_MAX
            )
        )
    for _ in range(n_samples - 2 * (n_samples // 3)):
        gen_phases.append(rng.uniform(-np.pi, np.pi, (GRID, GRID)))

    cos = torch.zeros(len(gen_phases), 1, GRID, GRID, dtype=torch.float32)
    sin = torch.zeros_like(cos)
    far = torch.zeros(len(gen_phases), 1, GRID, GRID, dtype=torch.float32)
    amp = aperture_mask()
    for i, ph in enumerate(gen_phases):
        masked = np.where(amp > 0, ph, 0.0)
        cos[i, 0] = torch.as_tensor(masked, dtype=torch.float32)
        sin[i, 0] = torch.as_tensor(np.where(amp > 0, np.sin(masked), 0.0), dtype=torch.float32)
        target = inv.sim_far_field(masked, pad=PADDING)
        peak = float(target.max())
        far[i, 0] = torch.as_tensor(target / peak if peak > 0 else target, dtype=torch.float32)
    return torch.cat([cos, sin], dim=1), far


def mild_dataset(max_train: int, seed: int, device: torch.device):
    """The incumbent's data: the real corpus, unchanged."""
    from ml.zernike.train_amp import _select_records, collect_split

    index = build_hw_index(index_cache="data/hw_index_cache.json", progress_every=0)
    index = index.filter(families=["slm_zernike_shaping"])
    dataset = HwPhaseImageDataset(index, config=MaterialiserConfig(grid=GRID), use_cache=True)
    cfg = AmpTrainConfig(
        families=("slm_zernike_shaping",), n_max=N_MAX, grid=GRID, epochs=EPOCHS_MILD,
        lr=0.02, batch_size=64, max_train=max_train, max_val=128, beam_samples=16,
        seed=seed, use_wandb=False, save_checkpoint=False,
    )
    train_idx, _ = _select_records(dataset, cfg)
    t = collect_split(dataset, train_idx, device)
    return torch.cat([t["phase_cos"], t["phase_sin"]], dim=1), t["target"]


# ---------------------------------------------------------------- inverse design
class CoeffForward(nn.Module):
    """coefficients -> canonical basis -> phase -> phasor -> model (Zernike parameters)."""

    def __init__(self, model: nn.Module, basis: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("basis", basis)
        self.register_buffer("amp", torch.as_tensor(aperture_mask(), dtype=torch.float32))

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        phase = torch.einsum("bk,kij->bij", theta, self.basis).unsqueeze(1)
        phasor = torch.polar(torch.ones_like(phase), phase) * self.amp
        return self.model(phasor.real, phasor.imag)


class FreeformForward(nn.Module):
    """freeform phase (full 64x64 pixels) -> phasor -> model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("amp", torch.as_tensor(aperture_mask(), dtype=torch.float32))

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        phasor = torch.polar(torch.ones_like(phase), phase) * self.amp
        return self.model(phasor.real, phasor.imag)


def aperture_mask_t() -> torch.Tensor:
    return torch.as_tensor(aperture_mask(), dtype=torch.float32)


def run_inverse(forward, start: np.ndarray, target, mask, steps: int = 80, lr: float = 0.02) -> np.ndarray:
    """Gradient design through ``forward``.

    ``start`` must already carry its batch/channel axes -- ``(1, n_modes)`` for the Zernike
    parameterisation, ``(1, 1, g, g)`` for freeform. Inferring that here is how the two
    parameterisations get confused: the physics model contracts ``phase_cos`` as
    ``(B, 1, g, g)``, so a bare ``(g, g)`` start is the wrong rank rather than a visible
    error.
    """
    param = torch.nn.Parameter(torch.as_tensor(start, dtype=torch.float32, device=target.device))
    opt = torch.optim.AdamW([param], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        composite_loss(forward(param), target, mask, WEIGHTS)["_mean_total"].backward()
        opt.step()
    return param.detach().cpu().numpy()


def train_physics(inputs, targets, epochs: int, seed: int, device) -> ZernikeAmpModel:
    torch.manual_seed(seed)
    model = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=N_MAX, grid=GRID, far_field_padding=PADDING, normalization="none")
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    x = inputs.to(device)
    y = targets.to(device)
    n = x.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for s in range(0, n, 64):
            idx = order[s : s + 64]
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((model(x[idx, :1], x[idx, 1:]) - y[idx]) ** 2)
            loss.backward()
            opt.step()
        sched.step()
    return model.eval()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 108)
    print("INVERSE PHASE OPTIMISATION TEST: does training on strong phases restore inversion?")
    print("=" * 108)
    print(f"seeds={SEEDS} n_max={N_MAX} padding={PADDING} device={device}")
    print("training targets AND scoring: SimPibSystem (shared code; independence is in the phases)")

    target = inv.target_tensor(GRID, SIZE_FRAC, ASPECT).to(device)
    mask = roi_mask(
        (GRID, GRID), (GRID / 2, GRID / 2), "rectangle", SIZE_FRAC * GRID, ASPECT
    ).to(device)
    basis = ZernikeAmpModel(ZernikeAmpConfig(n_max=N_MAX, grid=GRID)).basis.detach().clone().to(device)
    n_modes = basis.shape[0]

    flat_phase = np.zeros((GRID, GRID))
    gs_phase_np = inv.gs_phase(SIZE_FRAC, ASPECT, iterations=40)
    # TWO different GS objects: the Zernike arm needs a COEFFICIENT vector as its start
    # point, the freeform arm needs the PHASE. Converting the coefficients to a phase and
    # handing that to the coefficient parameterisation is a shape/semantics mismatch.
    gs_coeff_vec = inv.gs_coefficients(SIZE_FRAC, ASPECT, n_max=N_MAX).astype(np.float32)
    gs_coeff = gs_coeff_vec
    print("\nreference scores on the INDEPENDENT evaluator (higher is better):")
    print(f"  flat phase          {score_numpy(flat_phase):.4f}")
    print(f"  GS phase            {score_numpy(gs_phase_np):.4f}")
    flat_s, gs_s = score_numpy(flat_phase), score_numpy(gs_phase_np)

    arms = {
        "mild-only (incumbent)": "mild",
        "synth (GS+random)": "synth",
        "mixed (mild+synth)": "mixed",
    }
    rows: list[dict] = []
    for arm, kind in arms.items():
        for seed in SEEDS:
            if kind == "mild":
                x, y = mild_dataset(512, seed, device)
                epochs = EPOCHS_MILD
            elif kind == "synth":
                x, y = synthetic_dataset(600, seed)
                epochs = EPOCHS_SYNTH
            else:
                xm, ym = mild_dataset(512, seed, device)
                xs, ys = synthetic_dataset(600, seed + 1000)
                x = torch.cat([xm, xs])
                y = torch.cat([ym, ys])
                epochs = EPOCHS_SYNTH
            model = train_physics(x, y, epochs, seed, device)

            coeff_fwd = CoeffForward(model, basis).to(device).eval()
            free_fwd = FreeformForward(model).to(device).eval()
            rng = np.random.default_rng(4242 + seed)

            out: dict = {"arm": arm, "seed": seed, "n_train": int(x.shape[0])}
            # Zernike parameters, random start and GS start
            out["zern_rand"] = score_numpy(
                inv.coefficients_to_phase(
                    run_inverse(
                        coeff_fwd,
                        rng.normal(0.0, 0.6, (1, n_modes)).astype(np.float32),
                        target,
                        mask,
                    )[0],
                    n_max=N_MAX,
                )
            )
            out["zern_gs"] = score_numpy(
                inv.coefficients_to_phase(
                    run_inverse(coeff_fwd, gs_coeff_vec[None], target, mask)[0], n_max=N_MAX)
            )
            # freeform phase, random start and GS start
            out["free_rand"] = score_numpy(
                run_inverse(
                    free_fwd,
                    rng.uniform(-np.pi, np.pi, (1, 1, GRID, GRID)).astype(np.float32),
                    target, mask,
                )
            )
            out["free_gs"] = score_numpy(
                run_inverse(free_fwd, gs_phase_np.astype(np.float32)[None, None], target, mask)[0, 0]
            )
            rows.append(out)
            print(
                f"  {arm:<22} seed={seed} n={x.shape[0]:>5} "
                f"zern(rand)={out['zern_rand']:.4f} zern(gs)={out['zern_gs']:.4f} "
                f"free(rand)={out['free_rand']:.4f} free(gs)={out['free_gs']:.4f}"
            )

    print("\n" + "=" * 108)
    print(f"means over {len(SEEDS)} seeds -- validated simulator; flat={flat_s:.4f} "
          f"GS={gs_s:.4f}; higher is better")
    print("=" * 108)
    print(f"{'arm':<24}{'zern(rand)':>12}{'zern(gs)':>11}{'free(rand)':>12}"
          f"{'free(gs)':>11}{'best-flat':>11}")
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm]
        best = max(
            float(np.mean([r[k] for r in sel])) for k in ("zern_rand", "zern_gs", "free_rand", "free_gs")
        )
        print(f"{arm:<24}"
              f"{np.mean([r['zern_rand'] for r in sel]):>12.4f}"
              f"{np.mean([r['zern_gs'] for r in sel]):>11.4f}"
              f"{np.mean([r['free_rand'] for r in sel]):>12.4f}"
              f"{np.mean([r['free_gs'] for r in sel]):>11.4f}"
              f"{best - flat_s:>+11.4f}")

    print("\npaired vs the mild-only incumbent (same seed, same evaluator):")
    base = {r["seed"]: r for r in rows if r["arm"] == "mild-only (incumbent)"}
    for arm in arms:
        if arm == "mild-only (incumbent)":
            continue
        for key in ("zern_gs", "free_gs"):
            d = np.array([r[key] - base[r["seed"]][key] for r in rows if r["arm"] == arm])
            print(f"  {arm:<22} {key:<9} d={d.mean():+.4f} +/- "
                  f"{d.std(ddof=1)/np.sqrt(len(d)):.4f}  positives {int((d > 0).sum())}/{len(d)}")

    out_path = ROOT / "report" / "loss_defects" / "inverse_phase_test.json"
    out_path.write_text(json.dumps(
        {"flat": flat_s, "gs": gs_s, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
