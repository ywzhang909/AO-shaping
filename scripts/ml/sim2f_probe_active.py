"""Is a posterior-driven probe order better than the fixed Noll bank at equal budget?

The question
============
Each probe costs one SLM write plus one exposure, so on a bench the probe stage of
:mod:`ml.sim2f.fit` runs under a hard frame budget.  :func:`ml.sim2f.fit.probe`
spends that budget on a fixed bank (Noll ``4 .. n_modes + 1``, in Noll order).
:mod:`ml.sim2f.probe_active` argues that once a few frames exist the *order* is a
free parameter: a mode the camera can barely see buys little information per
frame, while a mode the residual still excites buys a lot, and greedy
D-optimality on the posterior covariance can price that difference.

This script measures whether that argument survives contact with a bench that
carries unmodelled aberration.  It is the budget-vs-error curve, not a single
number: at two probes a fixed bank has only touched defocus and astigmatism, and
the active selector may or may not have done better.

Protocol, per seed
==================
1. Draw a Noll ``4..11`` coefficient vector: a random azimuth per mode, then the
   whole vector rescaled so its RMS is uniform in ``AberrationConfig.rms_rad``.
   That is the seed's truth.
2. ``device`` -- a fresh :class:`~ml.sim2f.optics.Sim2f` whose ``OpticsErrors``
   carry that truth plus panel roughness (sigma 0.05 rad).  ``model`` -- a
   separate, ideal ``Sim2f`` with the same geometry and no errors at all.  The
   two are never the same object; the model never sees the truth.
3. ``measure`` is the device frame, uncropped: this lane is about probing, not
   about a small ROI.
4. :func:`ml.sim2f.probe_active.compare_probe_strategies` runs the fixed bank and
   the active selector through the *identical* code path, each once to the
   largest budget, with every smaller budget read off as a prefix.

Scored per (seed, strategy, budget): coefficient RMSE over the eight fitted modes
(Noll ``4..11``), plus the posterior diagnostics (``trace``, ``condition``,
``rank``, ``sigma2``, per-mode confidence) and the modes actually pushed.  The
active arm's win or loss is reported as a per-seed ratio ``active / fixed``;
with five seeds this is a direction, not a p-value.

Two truth vectors, because the bench has two conventions
-------------------------------------------------------
``OpticsErrors.slm_zernike`` is a **panel** aberration: it is normalised on
``grid / 2`` (see ``Sim2f._build_error_fields``), while the loop's coefficient
vector is normalised on the **pupil** radius (``grid * pupil_fill / 2``).  A mode
drawn with coefficient ``c`` therefore reaches only a fraction of ``c`` inside
the pupil -- measured at 0.32x .. 0.56x per mode for a seed here, leaving the
roughness behind at 0.05 rad rms.  Scoring a fit against the drawn number would
charge that representation offset to the estimator and flatten the comparison,
so both metrics are reported:

``coeff_rmse``
    RMSE against the drawn coefficient vector, as handed to ``OpticsErrors``.
    This is the literal "error vs truth" number; it carries the same convention
    offset in *both* arms, so it is reported but not used for the ratio.
``coeff_rmse_basis``
    RMSE against the drawn aberration **re-expressed in the fitted basis**: the
    least-squares projection, over the model's pupil mask, of a roughness-free
    panel carrying the same ``slm_zernike`` (Lane A's ``in_basis_device_coeffs``
    without the roughness).  This is the recovery error proper, and it is the
    metric the figure and the ratio use.

Honest limitations
==================
* **Offline surrogate.**  ``device`` is a :class:`~ml.sim2f.optics.Sim2f`, not the
  bench: no shot noise (``exposure_ms = 0``), no detector drift, no SLM
  calibration error beyond the modelled Zernike set and roughness.  A real bench
  adds noise that the posterior would have to estimate, and ``sigma2`` stands in
  for that here.
* **Five seeds, one bench family.**  The spread is a range over five draws, not a
  confidence interval, and every draw comes from one ``AberrationConfig``
  (i.i.d. mode amplitudes, Noll ``4..11`` only).
* **Truth is in-basis.**  Once the radius convention is accounted for, the drawn
  aberration lies exactly in the fitted mode family, so the coefficient floor is
  zero; the roughness (0.05 rad, not representable) is what keeps the *frame*
  error from reaching it.
* **One probe amplitude.**  Both arms share ``probe_amplitude_rad = 0.35`` and
  the module's own SNR-driven enlargement, so the comparison is about order
  only, not about how hard each mode is pushed.

Usage::

    python scripts/ml/sim2f_probe_active.py
    python scripts/ml/sim2f_probe_active.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ml.sim2f.config import AberrationConfig, OpticsErrors, Sim2fConfig
from ml.sim2f.fit import probe_noll
from ml.sim2f.optics import Sim2f
from ml.sim2f.probe_active import StrategyOutcome, compare_probe_strategies

#: Report this script feeds; the report links the artifacts below, so the key is
#: carried in ``summary.json``'s ``meta`` and never used to write the report.
REPORT_KEY = "report/shape_trials/fov_generalization/fov_generalization_20261101.md"

#: Pinned artifact locations (never the report root, never ``artifacts/`` itself).
OUT_DIR = REPO / "report/shape_trials/fov_generalization/artifacts/probe"
FIG_DIR = REPO / "report/shape_trials/fov_generalization/figures"
FIG_NAME = "fig_probe_budget.png"

#: ``--dry-run`` writes here instead, so a smoke test cannot touch the artifacts.
DRY_ROOT = Path(tempfile.gettempdir()) / "opencode" / "sim2f_probe_active_dry"

#: Fitted coefficient vector: Noll ``2 .. N_MODES + 1`` (tip/tilt stay at zero).
N_MODES = 10

#: The eight modes that are both drawn and probed, in Noll order.
PROBE_NOLL = tuple(range(4, 12))

#: Rows of the coefficient vector that carry the drawn/probed modes.
FITTED = slice(PROBE_NOLL[0] - 2, PROBE_NOLL[-1] - 1)

#: Hard probe budgets (pushes), read as prefixes of one run.
BUDGETS = (2, 4, 6, 8, 10)

#: Seeds per strategy, each a fresh device/model pair.
N_SEEDS = 5

#: Unmodelled panel roughness carried by every device; the model has none.
ROUGHNESS_SIGMA_RAD = 0.05

#: Strategies, in figure/table order.
STRATEGIES = ("fixed", "active")


# ----------------------------------------------------------------------
# Contract and truth
# ----------------------------------------------------------------------
def aberration_cfg() -> AberrationConfig:
    """The probe/fit contract, identical to Lane A's ``aberration_cfg()``."""
    return AberrationConfig(
        n_modes=N_MODES,
        probe_modes=PROBE_NOLL,
        probe_frozen=(1,),
        fit_frozen=(1, 2, 3),
        probe_amplitude_rad=0.35,
        fit_iters=80,
        fit_lr=0.05,
    )


def draw_truth(cfg: AberrationConfig, rng: np.random.Generator) -> np.ndarray:
    """One ``(n_modes,)`` coefficient vector with a uniform-RMS magnitude.

    Each mode gets a random azimuth (its coefficient is the projection of a unit
    phasor, ``cos(theta)``), and the vector is then rescaled so its RMS is
    uniform in ``cfg.rms_rad`` -- drawing magnitude and shape from one
    distribution would concentrate every state near the same RMS.  Rows for
    Noll 2/3 (tip/tilt) stay at zero: they are outside the probe bank.

    Args:
        cfg: Supplies ``n_modes`` and the ``rms_rad`` range.
        rng: Seeded generator; the only source of randomness here.

    Returns:
        ``(n_modes,)`` radians, Noll ``2 .. n_modes + 1`` in row order.
    """
    n_modes = int(cfg.n_modes)
    azimuth = rng.uniform(0.0, 2.0 * math.pi, size=len(PROBE_NOLL))
    shape = np.cos(azimuth)
    rms = float(rng.uniform(*cfg.rms_rad))
    shape = shape / math.sqrt(float(np.mean(shape**2)))
    vector = np.zeros(n_modes, dtype=np.float64)
    for mode, value in zip(PROBE_NOLL, shape * rms, strict=True):
        vector[mode - 2] = float(value)
    return vector


def build_benches(
    grid: int, ccd: int, truth: np.ndarray, seed: int
) -> tuple[Sim2f, Sim2f]:
    """A device carrying ``truth`` plus roughness, and an independent ideal model."""
    errors = OpticsErrors(
        slm_zernike=tuple((int(mode), float(truth[mode - 2])) for mode in PROBE_NOLL),
        slm_roughness_sigma_rad=ROUGHNESS_SIGMA_RAD,
        slm_roughness_seed=seed,
    )
    device = Sim2f(Sim2fConfig(grid=grid, ccd_px=ccd, errors=errors), device="cpu")
    model = Sim2f(Sim2fConfig(grid=grid, ccd_px=ccd), device="cpu")
    if device is model or device.errors == model.errors:
        raise RuntimeError("device and model must not share physics")
    return device, model


def in_basis_truth(truth: np.ndarray, grid: int, ccd: int, model: Sim2f) -> np.ndarray:
    """Re-express the drawn panel aberration in the loop's fitted basis.

    ``OpticsErrors.slm_zernike`` normalises each mode on ``grid / 2`` (the whole
    panel) while the coefficient vector is normalised on the pupil radius, so the
    drawn numbers are *not* the coefficients a fit should recover.  This projects
    a roughness-free panel carrying the same modes onto the model's Zernike basis
    over the pupil mask -- the model supplies geometry only, never a value.

    Args:
        truth: ``(n_modes,)`` drawn coefficients, radians.
        grid: Model grid side.
        ccd: Camera window side.
        model: Ideal bench; its ``zernike_basis`` and ``pupil_mask`` are used.

    Returns:
        ``(n_modes,)`` coefficients in the fitted basis, radians (zeros on the
        tip/tilt rows, which the panel does not carry).
    """
    panel_bench = Sim2f(
        Sim2fConfig(
            grid=grid,
            ccd_px=ccd,
            errors=OpticsErrors(
                slm_zernike=tuple(
                    (int(mode), float(truth[mode - 2])) for mode in PROBE_NOLL
                )
            ),
        ),
        device="cpu",
    )
    panel = getattr(panel_bench, "_panel_phase", None)
    vector = np.zeros(N_MODES, dtype=np.float64)
    if panel is None:
        return vector
    basis = model.zernike_basis(N_MODES + 1).detach().cpu().numpy()
    mask = model.pupil_mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(bool)
    panel_np = panel.detach().cpu().numpy()
    coefficients, *_ = np.linalg.lstsq(basis[:, mask].T, panel_np[mask], rcond=None)
    for mode in PROBE_NOLL:
        vector[mode - 2] = float(coefficients[mode - 1])
    return vector


def device_measure(device: Sim2f):
    """The camera read: uncropped device frame at the pushed phase."""
    return lambda phase: device.forward(phase)


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def coefficient_rmse(coeffs: np.ndarray, truth: np.ndarray) -> float:
    """RMS of ``coeffs - truth`` over the eight fitted modes (Noll ``4..11``)."""
    estimate = np.asarray(coeffs, dtype=np.float64)[FITTED]
    target = np.asarray(truth, dtype=np.float64)[FITTED]
    return float(np.sqrt(np.mean((estimate - target) ** 2)))


def repeats(modes: tuple[int, ...]) -> int:
    """How many of ``modes`` repeat a mode already probed earlier in the run."""
    return len(modes) - len(set(modes))


def record(
    seed: int,
    outcome: StrategyOutcome,
    truth: np.ndarray,
    basis_truth: np.ndarray,
) -> dict[str, Any]:
    """One flat row: truth error plus the posterior diagnostics for this budget."""
    confidence = np.asarray(outcome.confidence, dtype=np.float64)[FITTED]
    return {
        "seed": int(seed),
        "strategy": str(outcome.strategy),
        "budget": int(outcome.budget),
        "coeff_rmse": coefficient_rmse(outcome.coeffs, truth),
        "coeff_rmse_basis": coefficient_rmse(outcome.coeffs, basis_truth),
        "trace": float(outcome.trace),
        "condition": float(outcome.condition),
        "rank": int(outcome.rank),
        "sigma2": float(outcome.sigma2),
        "mean_confidence": float(np.mean(confidence)),
        "repeats": repeats(outcome.selected_modes),
        "selected_modes": [int(mode) for mode in outcome.selected_modes],
        "amplitudes": [float(value) for value in outcome.amplitudes],
        "coeffs": [float(value) for value in np.asarray(outcome.coeffs)],
    }


def run_seed(
    seed: int,
    grid: int,
    ccd: int,
    budgets: tuple[int, ...],
    cfg: AberrationConfig,
) -> dict[str, Any]:
    """Draw a truth, build the pair, and run both strategies to ``max(budgets)``."""
    rng = np.random.default_rng(seed)
    truth = draw_truth(cfg, rng)
    device, model = build_benches(grid, ccd, truth, seed)
    basis_truth = in_basis_truth(truth, grid, ccd, model)
    outcomes = compare_probe_strategies(
        device,
        cfg,
        measure=device_measure(device),
        budgets=budgets,
        model=model,
        candidates=probe_noll(cfg),
    )
    rows = [
        record(seed, outcome, truth, basis_truth)
        for strategy in STRATEGIES
        for outcome in outcomes[strategy]
    ]
    return {
        "seed": int(seed),
        "rms_rad": float(np.sqrt(np.mean(truth[FITTED] ** 2))),
        "rms_basis_rad": float(np.sqrt(np.mean(basis_truth[FITTED] ** 2))),
        "truth_noll": {str(mode): float(truth[mode - 2]) for mode in PROBE_NOLL},
        "truth_basis_noll": {
            str(mode): float(basis_truth[mode - 2]) for mode in PROBE_NOLL
        },
        "records": rows,
    }


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def _mean_std(values: list[float]) -> dict[str, float]:
    """Mean and sample std (``ddof=1`` when n > 1, else 0)."""
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {"mean": float(np.mean(array)), "std": std, "n": int(array.size)}


def _cell(run: dict[str, Any], strategy: str, budget: int) -> dict[str, Any]:
    """The one record for ``(strategy, budget)`` in ``run``."""
    return next(
        row
        for row in run["records"]
        if row["strategy"] == strategy and row["budget"] == budget
    )


def aggregate(
    runs: list[dict[str, Any]], budgets: tuple[int, ...]
) -> dict[str, dict[str, Any]]:
    """Mean/std across seeds for every ``(strategy, budget)`` cell."""
    table: dict[str, dict[str, Any]] = {}
    for strategy in STRATEGIES:
        for budget in budgets:
            cells = [_cell(run, strategy, budget) for run in runs]
            table[f"{strategy}|{budget}"] = {
                "strategy": strategy,
                "budget": int(budget),
                "coeff_rmse": _mean_std([cell["coeff_rmse"] for cell in cells]),
                "coeff_rmse_basis": _mean_std(
                    [cell["coeff_rmse_basis"] for cell in cells]
                ),
                "trace": _mean_std([cell["trace"] for cell in cells]),
                "condition": _mean_std([cell["condition"] for cell in cells]),
                "rank": _mean_std([float(cell["rank"]) for cell in cells]),
                "sigma2": _mean_std([cell["sigma2"] for cell in cells]),
                "mean_confidence": _mean_std(
                    [cell["mean_confidence"] for cell in cells]
                ),
                "repeats": _mean_std([float(cell["repeats"]) for cell in cells]),
            }
    return table


def compare(
    runs: list[dict[str, Any]], budgets: tuple[int, ...], metric: str
) -> dict[str, Any]:
    """Per-seed ``active / fixed`` ratio of ``metric``, and the direction it points."""
    per_budget: dict[str, Any] = {}
    for budget in budgets:
        ratios = [
            _cell(run, "active", budget)[metric] / _cell(run, "fixed", budget)[metric]
            for run in runs
        ]
        per_budget[str(budget)] = {
            "mean_ratio": float(np.mean(ratios)),
            "std_ratio": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
            "per_seed": [float(value) for value in ratios],
            "active_better_seeds": int(sum(1 for value in ratios if value < 1.0)),
            "n_seeds": len(ratios),
        }
    means = {budget: cell["mean_ratio"] for budget, cell in per_budget.items()}
    return {
        "metric": metric,
        "ratio": per_budget,
        "budgets_active_better": [
            int(budget) for budget, value in means.items() if value < 1.0
        ],
        "budgets_active_worse": [
            int(budget) for budget, value in means.items() if value >= 1.0
        ],
        "mean_ratio_all_budgets": float(np.mean(list(means.values()))),
    }


# ----------------------------------------------------------------------
# Artifacts
# ----------------------------------------------------------------------
def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Flat one-row-per-(seed, strategy, budget) table."""
    fields = [
        "seed",
        "strategy",
        "budget",
        "coeff_rmse",
        "coeff_rmse_basis",
        "trace",
        "condition",
        "rank",
        "sigma2",
        "mean_confidence",
        "repeats",
        "selected_modes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{name: row[name] for name in fields if name != "selected_modes"},
                    "selected_modes": " ".join(
                        str(mode) for mode in row["selected_modes"]
                    ),
                }
            )


def plot(
    table: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    budgets: tuple[int, ...],
    path: Path,
) -> None:
    """Panel (a) RMSE vs budget, panel (b) per-seed active/fixed ratio."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colours = {"fixed": "tab:gray", "active": "tab:red"}
    for strategy in STRATEGIES:
        cells = [table[f"{strategy}|{budget}"] for budget in budgets]
        mean = np.array([cell["coeff_rmse_basis"]["mean"] for cell in cells])
        std = np.array([cell["coeff_rmse_basis"]["std"] for cell in cells])
        left.plot(budgets, mean, "o-", color=colours[strategy], label=strategy)
        left.fill_between(
            budgets, mean - std, mean + std, color=colours[strategy], alpha=0.18
        )
        drawn = np.array([cell["coeff_rmse"]["mean"] for cell in cells])
        left.plot(
            budgets,
            drawn,
            ":",
            color=colours[strategy],
            alpha=0.6,
            label=f"{strategy}, vs drawn vector",
        )
    left.set_xlabel("probe budget (pushes)")
    left.set_ylabel("coefficient RMSE, rad (Noll 4..11)")
    left.set_title("(a) recovery vs budget, mean +/- std (dotted: drawn vector)")
    left.set_xticks(list(budgets))
    left.grid(alpha=0.3)
    left.legend(fontsize=8)

    for index, budget in enumerate(budgets):
        values = comparison["ratio"][str(budget)]["per_seed"]
        right.plot(
            [budget] * len(values),
            values,
            "o",
            color="tab:blue",
            alpha=0.45,
            markersize=5,
        )
        right.plot(
            [budget - 0.18, budget + 0.18],
            [np.mean(values)] * 2,
            "-",
            color="tab:red",
            linewidth=2.0,
            label="mean ratio" if index == 0 else None,
        )
    right.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    right.set_xlabel("probe budget (pushes)")
    right.set_ylabel("active / fixed coefficient RMSE")
    right.set_title("(b) per-seed ratio (below 1.0 = active wins)")
    right.set_xticks(list(budgets))
    right.grid(alpha=0.3)
    right.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--ccd", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument(
        "--seed", type=int, default=0, help="base seed; seed i uses base + i"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "one seed, budgets 2 and 4, artifacts under a temp dir: checks the "
            "code path without touching the report"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the comparison and write the four artifacts."""
    args = _parse(argv)
    started = time.perf_counter()

    if args.dry_run:
        seeds = 1
        budgets = (2, 4)
        out_dir = DRY_ROOT / "artifacts" / "probe"
        fig_dir = DRY_ROOT / "figures"
    else:
        seeds = int(args.seeds)
        budgets = tuple(int(value) for value in args.budgets)
        out_dir = OUT_DIR
        fig_dir = FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = aberration_cfg()
    runs: list[dict[str, Any]] = []
    for index in range(seeds):
        seed = int(args.seed) + index
        torch.manual_seed(seed)
        np.random.seed(seed)
        run = run_seed(seed, int(args.grid), int(args.ccd), budgets, cfg)
        runs.append(run)
        print(
            f"seed {seed}: "
            + " ".join(
                f"b{budget} fixed={_cell(run, 'fixed', budget)['coeff_rmse_basis']:.4f} "
                f"active={_cell(run, 'active', budget)['coeff_rmse_basis']:.4f}"
                for budget in budgets
            )
        )

    table = aggregate(runs, budgets)
    comparison = compare(runs, budgets, "coeff_rmse_basis")
    literal = compare(runs, budgets, "coeff_rmse")
    runtime = round(time.perf_counter() - started, 2)

    meta = {
        "report_key": REPORT_KEY,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "runtime_s": runtime,
        "grid": int(args.grid),
        "ccd_px": int(args.ccd),
        "seeds": seeds,
        "seed_base": int(args.seed),
        "budgets": list(budgets),
        "candidates": [int(mode) for mode in probe_noll(cfg)],
        "n_modes": int(cfg.n_modes),
        "fitted_noll": [int(mode) for mode in PROBE_NOLL],
        "probe_amplitude_rad": float(cfg.probe_amplitude_rad),
        "rms_rad": [float(value) for value in cfg.rms_rad],
        "roughness_sigma_rad": ROUGHNESS_SIGMA_RAD,
        "device": "cpu",
        "exposure_ms": 0.0,
        "primary_metric": (
            "coeff_rmse_basis: coefficient RMSE over Noll 4..11 against the drawn "
            "aberration re-expressed in the fitted basis, radians"
        ),
        "secondary_metric": (
            "coeff_rmse: coefficient RMSE against the drawn panel-normalized "
            "vector; both arms carry the same ~0.4x radius-convention offset"
        ),
        "limitation": (
            "offline Sim2f surrogate (no shot noise), 5 seeds, one AberrationConfig; "
            "the ratio is a direction, not a p-value"
        ),
    }

    (out_dir / "runs.json").write_text(
        json.dumps({"meta": meta, "runs": runs}, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "aggregate": table,
                "comparison": comparison,
                "comparison_literal": literal,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv([row for run in runs for row in run["records"]], out_dir / "trials.csv")
    plot(table, comparison, budgets, fig_dir / FIG_NAME)

    for title, payload, metric in (
        ("in-basis (primary)", comparison, "coeff_rmse_basis"),
        ("drawn vector (literal)", literal, "coeff_rmse"),
    ):
        print()
        print(title)
        print(f"{'budget':>7} {'fixed':>9} {'active':>9} {'ratio':>7} {'winner':>8}")
        for budget in budgets:
            fixed = table[f"fixed|{budget}"][metric]
            active = table[f"active|{budget}"][metric]
            ratio = payload["ratio"][str(budget)]["mean_ratio"]
            print(
                f"{budget:>7} {fixed['mean']:>9.4f} {active['mean']:>9.4f} "
                f"{ratio:>7.3f} {'active' if ratio < 1.0 else 'fixed':>8}"
            )
        print(
            f"mean ratio over budgets {payload['mean_ratio_all_budgets']:.3f}; "
            f"active better at {payload['budgets_active_better']}, "
            f"worse at {payload['budgets_active_worse']}"
        )
    print()
    print("repeat diagnostics (mean repeats per seed):")
    for budget in budgets:
        print(
            f"  budget {budget:>2}: fixed "
            f"{table[f'fixed|{budget}']['repeats']['mean']:.1f} "
            f"active {table[f'active|{budget}']['repeats']['mean']:.1f}"
        )
    print(f"runtime {runtime}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
