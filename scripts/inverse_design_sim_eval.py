"""Inverse (shaping) design on the learned model, scored on an INDEPENDENT simulator.

Why this script exists
----------------------
``ZernikeAmpModel.forward`` is a *forward* model: it returns
``measured_phasor * exp(i * correction)``. That makes it useless for inverse design
in the obvious way -- hand it a zero phasor to "use the coefficients alone" and the
field is identically zero, so the output is identically zero and **every gradient is
exactly 0.0**. ``ZernikeAmpModel.correction_far_field()`` is the seam that fixes it.

But a self-consistent inverse experiment proves nothing: the same model both
synthesises the phase and scores it, so its own error cancels. This script therefore
scores the designed phase on :class:`SimPibSystem`, an independent numpy
``|FFT(pupil * e^{i phi})|^2`` with its own Gaussian illumination model, using the
canonical ``rms_pib_terms``.

Results and caveats live in ``report/loss_defects/PROCESS.md``. Headline, from the
replicated run: every objective beat the flat reference in 12/12 paired runs, but the
*ordering between* objectives is not resolved -- the single-sample run that appeared
to favour ``physical`` reversed sign on replication.

Usage
-----
    python scripts/inverse_design_sim_eval.py                # replicated run
    python scripts/inverse_design_sim_eval.py --samples 1 --seeds 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.drivers.sim.slm_pib_sim import SimPibSystem  # noqa: E402
from ao_shaping.utils.image.target.metrics import rms_pib_terms  # noqa: E402
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator  # noqa: E402
from ml.hwdataset import MaterialiserConfig, build_hw_index  # noqa: E402
from ml.hwdataset.records import Materialiser  # noqa: E402
from ml.zernike.losses import LossConfig, composite_loss, roi_mask  # noqa: E402
from ml.zernike.models import ZernikeAmpConfig, ZernikeAmpModel  # noqa: E402

# Discretisation shared with `inverse_design_accuracy_ladder.py`, which imports
# these rather than re-declaring them. Kept as module constants (not argparse-only
# defaults) so there is exactly one source of truth for the grid.
GRID, N_MAX, PADDING = 64, 15, 12
SIZE_FRAC, ASPECT = 0.375, 4.0 / 3.0
# The simulator is calibrated for the real bench panel, so the model's small pupil
# map has to be embedded into the illuminated disc. A 64x64 sim panel is out of
# regime and returns NaN -- verified by sweeping panel shape x waist x padding.
PANEL_H, PANEL_W = 1200, 1920
DISC_RADIUS = 450
SIM_PADDING, SIM_WINDOW = 4, 1024

OBJECTIVES: dict[str, LossConfig] = {
    "mse": LossConfig(w_mse=1.0),
    "physical": LossConfig(w_mse=0.0, w_pib=1.0, w_uniformity=1.0),
    "anchored": LossConfig(w_mse=1.0, w_shape_gap=1.0),
}


def target_square(grid: int) -> torch.Tensor:
    """Binary square target, centred, matching the ``SIZE_FRAC``/``ASPECT`` ROI."""
    t = torch.zeros(1, 1, grid, grid)
    half_w = int(SIZE_FRAC * grid * ASPECT / 2)
    half_h = int(SIZE_FRAC * grid / 2)
    c = grid // 2
    t[..., c - half_h : c + half_h, c - half_w : c + half_w] = 1.0
    return t


def score_on_sim(coefficients: np.ndarray, grid: int, n_max: int) -> dict[str, float]:
    """Push a coefficient vector through the independent sim and score the far field."""
    generator = ZernikeGenerator((grid, grid), n_orders=n_max)
    pupil = generator.generate_noll(np.asarray(coefficients, dtype=np.float64))
    # `ZernikeGenerator` returns NaN OUTSIDE the aperture disc: the pupil has no
    # light there, not undefined light. Embedding the NaN into the panel makes the
    # whole far field non-finite and every metric silently reads 0.
    pupil = np.nan_to_num(pupil, nan=0.0, posinf=0.0, neginf=0.0)

    cy, cx = PANEL_H // 2, PANEL_W // 2
    phase = np.zeros((PANEL_H, PANEL_W), dtype=np.float64)
    phase[cy - grid // 2 : cy + grid // 2, cx - grid // 2 : cx + grid // 2] = pupil

    system = SimPibSystem(
        slm_shape=(PANEL_H, PANEL_W),
        ccd_res=(512, 512),
        beam_w0=float(DISC_RADIUS),
        noise_adu=0.0,
        seed=0,
        far_field_padding=SIM_PADDING,
        far_field_window=SIM_WINDOW,
    )
    system.set_phase_rad(phase)
    crop = np.asarray(system.far_field(), dtype=np.float64)[:grid, :grid]
    if not np.all(np.isfinite(crop)):
        raise RuntimeError("sim far field is non-finite -- geometry out of regime")

    pib, uni = rms_pib_terms(
        crop, (grid / 2, grid / 2), "rectangle", SIZE_FRAC * grid, ASPECT
    )
    return {"pib_term": float(pib), "uniformity": float(uni), "shape_sum": float(pib + uni)}


def fit_forward(index, position: int, grid: int, n_max: int, padding: int, steps: int, lr: float):
    """Fit the forward model to one real measured sample; return its coefficients.

    A real target is required: a model fitted to nothing is not a forward model, so
    inverting it would test nothing.
    """
    usable = [r for r in index.records if r.key is not None and r.key >= 0]
    ref = usable[position]
    sample = Materialiser(
        config=MaterialiserConfig(grid=grid), use_cache=True
    ).materialise(ref)

    cos = torch.as_tensor(np.asarray(sample.phase_cos, dtype=np.float32))[None, None]
    sin = torch.as_tensor(np.asarray(sample.phase_sin, dtype=np.float32))[None, None]
    target = torch.as_tensor(np.asarray(sample.image, dtype=np.float32))[None, None]

    torch.manual_seed(0)
    model = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=n_max, grid=grid, far_field_padding=padding, normalization="peak")
    )
    reference = model._normalize(target.clone())  # noqa: SLF001
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        optimiser.zero_grad()
        ((model(cos, sin) - reference) ** 2).mean().backward()
        optimiser.step()
    with torch.no_grad():
        residual = float(((model(cos, sin) - reference) ** 2).mean())
    # `coefficients_array` already returns a detached numpy array.
    return model.coefficients_array(), residual


def design(
    weights: LossConfig,
    start: np.ndarray | None,
    grid: int,
    n_max: int,
    padding: int,
    seed: int,
    steps: int,
    lr: float,
) -> np.ndarray:
    """Run inverse design on the learned model via ``correction_far_field``."""
    torch.manual_seed(seed)
    model = ZernikeAmpModel(
        ZernikeAmpConfig(n_max=n_max, grid=grid, far_field_padding=padding, normalization="peak")
    )
    if start is not None:
        with torch.no_grad():
            model.coefficients.copy_(torch.as_tensor(np.asarray(start, dtype=np.float32)))

    mask = roi_mask((grid, grid), (grid / 2, grid / 2), "rectangle", SIZE_FRAC * grid, ASPECT)
    target = target_square(grid)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        optimiser.zero_grad()
        composite_loss(model.correction_far_field(), target, mask, weights)[
            "_mean_total"
        ].backward()
        optimiser.step()
    return model.coefficients_array()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=GRID)
    parser.add_argument("--n-max", type=int, default=N_MAX)
    parser.add_argument("--padding", type=int, default=PADDING)
    parser.add_argument("--samples", type=int, default=4, help="real corpus samples to fit")
    parser.add_argument("--seeds", type=int, default=3, help="design seeds per sample")
    parser.add_argument("--fit-steps", type=int, default=120)
    parser.add_argument("--fit-lr", type=float, default=0.03)
    parser.add_argument("--design-steps", type=int, default=60)
    parser.add_argument("--design-lr", type=float, default=0.02)
    parser.add_argument(
        "--index-cache", default="data/hw_index_cache.json", help="hw corpus index cache"
    )
    parser.add_argument(
        "--out",
        default="report/loss_defects/inverse_design_replicates.json",
        help="where to write the per-run records",
    )
    args = parser.parse_args()

    index = build_hw_index(index_cache=args.index_cache).filter(
        families=["slm_zernike_shaping"]
    )

    rows: list[dict[str, float]] = []
    header = f"{'sample':>7}{'seed':>5}{'fit MSE':>10}{'flat':>9}" + "".join(
        f"{name:>11}" for name in OBJECTIVES
    )
    print("=" * len(header))
    print("INVERSE DESIGN on the learned model, SCORED on an independent sim")
    print("=" * len(header))
    print(header)

    for position in range(args.samples):
        start, fit_mse = fit_forward(
            index, position, args.grid, args.n_max, args.padding, args.fit_steps, args.fit_lr
        )
        flat = score_on_sim(np.zeros_like(start), args.grid, args.n_max)
        for seed in range(args.seeds):
            row: dict[str, float] = {"sample": position, "seed": seed, "flat": flat["shape_sum"]}
            for name, weights in OBJECTIVES.items():
                coefficients = design(
                    weights, start, args.grid, args.n_max, args.padding,
                    seed, args.design_steps, args.design_lr,
                )
                row[name] = score_on_sim(coefficients, args.grid, args.n_max)["shape_sum"]
            rows.append(row)
            print(
                f"{position:>7}{seed:>5}{fit_mse:>10.5f}{row['flat']:>9.4f}"
                + "".join(f"{row[name]:>11.4f}" for name in OBJECTIVES)
            )

    print("\n" + "=" * 72)
    print(f"paired deltas vs `mse`  ({len(rows)} pairs)")
    print("=" * 72)
    baseline = [r["mse"] for r in rows]
    for name in OBJECTIVES:
        if name == "mse":
            continue
        deltas = [r[name] - b for r, b in zip(rows, baseline)]
        wins = sum(1 for d in deltas if d > 0)
        print(
            f"  {name:<10} mean={np.mean(deltas):+.4f} "
            f"spread={max(deltas) - min(deltas):.4f} positives={wins}/{len(deltas)}"
        )
    print("\nmeans: " + ", ".join(f"{n}={np.mean([r[n] for r in rows]):.4f}" for n in OBJECTIVES))
    beaten = sum(1 for r in rows if all(r[n] > r["flat"] for n in OBJECTIVES))
    print(f"all objectives beat flat in {beaten}/{len(rows)} paired runs")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())