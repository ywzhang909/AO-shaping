"""Does a physics-informed residual generalise outside the fitted window?

The question this script answers is the one that matters for a small-ROI
camera: a forward model fitted on a 96x96 window sees a small part of the far
field, so it can match that window for the wrong reasons and be badly wrong
everywhere else.  :mod:`ml.sim2f.pinn` adds a residual phase ``delta`` to the
commanded phase and regularises it with constraints the hardware really
imposes (aperture support, a Zernike band limit, detector linearity, surface
smoothness).  Here that residual is fitted on window pixels only and then
scored on the **whole** sensor, which the fit never sees.

Six arms, per seed:

``base``
    Stage 1 only -- :func:`ml.sim2f.fit.probe` on the window with Noll modes
    4..11, no residual.
``data_only``
    Residual fitted with ``physics=False`` (window L1 + the TV regulariser).
``pinn``
    The same residual fitted with every physics term active.
``band_only``
    Residual fitted with only the Zernike-band term (data + band + smooth).
``support_only``
    Residual fitted with only the aperture-support term (data + support + smooth).
``oracle``
    The true device error vector (Noll 4..21 of ``OpticsErrors.slm_zernike``)
    as an upper bound *within the Zernike family*; the device's
    ``slm_roughness`` (sigma 0.05 rad) is not representable and is not
    corrected, so this is a bound, not perfection.

Three extra arms are a **declared sensitivity sweep of one knob**, not a search
for a winner: ``pinn_support_1.0``/``pinn_support_0.1``/``pinn_support_0.01``
set the ``support`` weight to 1.0/0.1/0.01 with ``data``, ``band``, ``exposure``
and ``smooth`` held at :data:`ml.sim2f.pinn.DEFAULT_WEIGHTS`.  They exist
because the all-on ``pinn`` arm is indistinguishable from ``support_only`` --
the support term is what wrecks the out-of-window prediction -- so the sweep
says how much of the damage the term is responsible for.  Every weight is
reported, including the ones that lose; choosing a weight *on* ``l1_out`` is
selection and the resulting number is optimistic.

The primary metric is ``l1_out`` -- relative L1 on the pixels outside the
window -- from :func:`ml.sim2f.extrapolate.extrapolation_metrics`.  It is a
ratio whose denominator is the true energy outside the window, so it is only
interpretable where that energy is non-trivial: on the ``tiny`` phase the
denominator is near zero and ``l1_out`` is a ratio of noise.  Every record
therefore also carries ``l1_out_denom`` (that denominator, from the raw frame)
and ``out_energy_frac`` (that denominator over the full-frame sum), and the
summary reports ``out_energy_frac`` per phase so the report can say which
phases have a readable ``l1_out``.  ``l1_out`` itself is left untouched.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from ao_shaping.utils.wavefront.zernike_utils import (
    generate_zernike_phase,
)
from ml.sim2f.config import AberrationConfig, OpticsErrors, Sim2fConfig
from ml.sim2f.extrapolate import extrapolation_metrics, frame_np
from ml.sim2f.fit import probe
from ml.sim2f.losses import prepare_frame
from ml.sim2f.optics import Sim2f
from ml.sim2f.pinn import DEFAULT_WEIGHTS, fit_residual
from ml.sim2f.window import CameraWindow, WindowedBench

ARMS = (
    "base",
    "data_only",
    "pinn",
    "band_only",
    "support_only",
    "pinn_support_1.0",
    "pinn_support_0.1",
    "pinn_support_0.01",
    "oracle",
)
METRICS = (
    "l1_out",
    "l1_in",
    "ncc_out",
    "ee_out_err",
    "centroid_err_px",
    "d86_err_px",
    "l1_out_denom",
    "out_energy_frac",
)
TERMS = ("data", "support", "band", "smooth", "total")

#: Support weights of the declared sensitivity sweep (addition 1), with the
#: label used in the arm name.  The other terms stay at
#: :data:`ml.sim2f.pinn.DEFAULT_WEIGHTS`; ``1.0`` is deliberately a duplicate of
#: the all-on ``pinn`` arm so the existing numbers stay comparable.
SUPPORT_SWEEP: tuple[tuple[str, float], ...] = (
    ("1.0", 1.0),
    ("0.1", 0.1),
    ("0.01", 0.01),
)


def _sweep_weights(support_w: float) -> dict[str, float]:
    """``DEFAULT_WEIGHTS`` with only the ``support`` weight changed."""
    return {**DEFAULT_WEIGHTS, "support": float(support_w)}


#: The fitted arms: ``(name, physics, weights)``.  ``pinn``/``data_only`` are the
#: two the brief asks for; ``band_only`` and ``support_only`` are diagnostics that
#: say *which* physics term earns its keep, since a single all-on arm cannot.
#: The ``pinn_support_*`` arms are the declared support-weight sensitivity sweep.
FITS: tuple[tuple[str, bool, dict[str, float] | None], ...] = (
    ("data_only", False, None),
    ("pinn", True, None),
    (
        "band_only",
        True,
        {"data": 1.0, "support": 0.0, "band": 1.0, "exposure": 0.0, "smooth": 0.05},
    ),
    (
        "support_only",
        True,
        {"data": 1.0, "support": 1.0, "band": 0.0, "exposure": 0.0, "smooth": 0.05},
    ),
    *((f"pinn_support_{label}", True, _sweep_weights(w)) for label, w in SUPPORT_SWEEP),
)

#: The unmodelled bench state every arm has to survive.
ERRORS = OpticsErrors(
    slm_zernike=((4, 0.35), (6, -0.28), (8, 0.22), (12, 0.20), (15, -0.16)),
    slm_roughness_sigma_rad=0.05,
    slm_roughness_seed=3,
)


def _coeffs(noll_values: dict[int, float], n_modes: int = 20) -> np.ndarray:
    """``(n_modes,)`` Noll ``2..n_modes+1`` coefficient vector from a Noll dict."""
    out = np.zeros(n_modes, dtype=np.float64)
    for noll, value in noll_values.items():
        out[int(noll) - 2] = float(value)
    return out


def _train_phases(
    sim: Sim2f, count: int, seed: int, *, rms: float = 0.45
) -> torch.Tensor:
    """``(count, grid, grid)`` commanded phases drawn from Noll 4..21."""
    rng = np.random.default_rng(seed)
    draw = rng.normal(size=(count, 18))
    draw *= rms / np.sqrt((draw**2).mean(axis=1, keepdims=True))
    coeffs = np.concatenate([np.zeros((count, 3)), draw], axis=1)
    return sim.aberration(coeffs)


def _test_phases(sim: Sim2f) -> dict[str, torch.Tensor]:
    """Three held-out commands: a diffraction-limited core, an oversized spot, a draw."""
    tiny = torch.zeros(1, sim.grid, sim.grid, device=sim.device)
    oversized = sim.aberration(_coeffs({4: 1.4})[None])
    held_out = _train_phases(sim, 1, seed=991, rms=0.6)
    return {"tiny": tiny, "oversized": oversized, "held_out": held_out}


def _panel_phase(sim: Sim2f, noll_values: dict[int, float]) -> torch.Tensor:
    """The device's static panel Zernike, on the panel's own ``grid / 2`` radius.

    ``Sim2f`` normalises the *device* panel modes on ``grid / 2`` -- they are
    spread over the whole sampled panel, as ``bench_twin`` generates them --
    while :meth:`Sim2f.zernike_basis` and :meth:`Sim2f.aberration` use
    ``pupil_radius_px``.  Feeding the coefficient vector to ``aberration``
    therefore builds the wrong phase (rms 0.45 rad off at ``grid = 64``, i.e.
    worse than doing nothing), so the map is regenerated at the panel radius.
    ``slm_roughness`` is *not* added: it is not representable in the Zernike
    family, which is exactly why the oracle is a bound and not perfection.
    """
    raw = generate_zernike_phase(
        {int(k): float(v) for k, v in noll_values.items()},
        resolution=(int(sim.grid), int(sim.grid)),
        n_max=int(sim.cfg.errors.slm_zernike_n_max),
        radius=float(sim.grid) / 2.0,
    )
    return torch.as_tensor(np.nan_to_num(raw), device=sim.device, dtype=sim.dtype)


def _assert_oracle_matches_device(device: Sim2f, oracle_ab: torch.Tensor) -> None:
    """Pin the oracle to the device's static Zernike panel, excluding roughness."""
    reference = Sim2f(
        Sim2fConfig(
            grid=device.grid,
            ccd_px=device.ccd_px,
            errors=OpticsErrors(
                slm_zernike=device.errors.slm_zernike,
                slm_zernike_n_max=device.errors.slm_zernike_n_max,
            ),
        ),
        device=device.device,
        dtype=device.dtype,
    )
    assert reference._panel_phase is not None
    torch.testing.assert_close(oracle_ab, reference._panel_phase, rtol=0.0, atol=0.0)


def _stage1(
    device: Sim2f, model: Sim2f, window: CameraWindow
) -> tuple[np.ndarray, float]:
    """Probe-stage aberration estimate from window pixels only."""
    wbench = WindowedBench(model, window)

    def measure(phase: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return window.crop(device.forward(phase))

    result = probe(wbench, AberrationConfig(n_modes=10), measure=measure)
    return result.coeffs, float(result.residual)


def _score(
    device: Sim2f,
    model: Sim2f,
    phase: torch.Tensor,
    aberration: torch.Tensor | None,
    delta: np.ndarray | None,
    window: CameraWindow,
) -> dict[str, float]:
    """Full-sensor metrics for one prediction against the true device frame.

    Besides the ratio metrics, records the **denominator** of ``l1_out``: the
    true energy outside the window, both absolutely (``l1_out_denom``, raw frame
    units) and as a fraction of the full-frame true energy
    (``out_energy_frac``).  Where that fraction is near zero -- the ``tiny``
    phase -- ``l1_out`` is a ratio of noise and must not be read as an error.
    """
    with torch.no_grad():
        truth = prepare_frame(device.forward(phase))
        command = phase
        if delta is not None:
            command = (
                phase
                + torch.as_tensor(delta, device=phase.device, dtype=phase.dtype)[None]
            )
        predicted = prepare_frame(model.forward(command, aberration=aberration))
        raw_truth = frame_np(truth, normalise=False)
    metrics = extrapolation_metrics(predicted, truth, window).as_dict()
    outside = ~window.mask((raw_truth.shape[1], raw_truth.shape[0])).astype(bool)
    denom = float(raw_truth[outside].sum())
    total = float(raw_truth.sum())
    metrics["l1_out_denom"] = denom
    metrics["out_energy_frac"] = denom / total if total > 0 else float("nan")
    return metrics


def _plot_losses(curves: dict[str, dict[str, list[float]]], path: Path) -> None:
    """Per-term loss curves, PINN against the data-only ablation."""
    fig, axes = plt.subplots(1, len(TERMS), figsize=(3.0 * len(TERMS), 3.2))
    for axis, term in zip(axes, TERMS):
        for arm, style in (("pinn", "-"), ("data_only", "--")):
            series = np.asarray(curves[arm][term], dtype=np.float64)
            if series.size == 0:
                continue
            axis.plot(series, style, label=arm, linewidth=1.2)
        axis.set_yscale("symlog", linthresh=1e-3)
        axis.set_title(term)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("physics terms: PINN vs data-only")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_cases(
    cases: dict[str, dict[str, np.ndarray]], window: CameraWindow, path: Path
) -> None:
    """Truth / base / PINN / oracle for each test phase, with the window drawn."""
    names = list(cases)
    columns = ("truth", "base", "pinn", "oracle")
    fig, axes = plt.subplots(
        len(names), len(columns), figsize=(3.0 * len(columns), 3.0 * len(names))
    )
    axes = np.atleast_2d(axes)
    for row, name in enumerate(names):
        panels = cases[name]
        scale = float(panels["truth"].max())
        for column, key in enumerate(columns):
            axis = axes[row, column]
            axis.imshow(panels[key], cmap="inferno", vmin=0.0, vmax=scale)
            if key == "truth":
                x0, y0 = window.offset()
                axis.add_patch(
                    plt.Rectangle(
                        (x0 - 0.5, y0 - 0.5),
                        window.width,
                        window.height,
                        fill=False,
                        edgecolor="cyan",
                        linewidth=1.0,
                    )
                )
            if row == 0:
                axis.set_title(key)
            if column == 0:
                axis.set_ylabel(name)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("full-sensor prediction (window outlined on truth)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("report/shape_trials/fov_generalization/artifacts/pinn"),
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--ccd", type=int, default=256)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--train-phases", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument(
        "--exposures",
        type=float,
        nargs=2,
        default=None,
        help=(
            "capture every training phase at two exposures for the detector-"
            "linearity term; off by default because this bench and the model "
            "share one photometric chain, which makes the term identically "
            "zero (it would only double the runtime)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the six-arm comparison and write the artifacts."""
    args = _parse(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    device = Sim2f(Sim2fConfig(grid=args.grid, ccd_px=args.ccd, errors=ERRORS))
    model = Sim2f(Sim2fConfig(grid=args.grid, ccd_px=args.ccd))
    window = CameraWindow.centered(args.window, args.ccd)
    exposures = None if args.exposures is None else tuple(args.exposures)

    oracle_coeffs = _coeffs({noll: value for noll, value in ERRORS.slm_zernike})
    oracle_ab = _panel_phase(model, dict(ERRORS.slm_zernike))
    _assert_oracle_matches_device(device, oracle_ab)
    tests = _test_phases(model)

    runs: list[dict[str, object]] = []
    curves: dict[str, dict[str, list[float]]] = {"pinn": {}, "data_only": {}}
    coefficients: dict[str, np.ndarray] = {"oracle": oracle_coeffs}
    cases: dict[str, dict[str, np.ndarray]] = {name: {} for name in tests}

    # Stage 1 is deterministic -- the bench has no shot noise at ``exposure_ms=0``
    # and ``probe`` seeds nothing -- so it is solved once and shared by every
    # seed.  ``base``/``oracle`` are therefore identical across seeds and their
    # reported std is 0, which is the truth about them, not a collapsed sample.
    base_coeffs, base_residual = _stage1(device, model, window)
    base_ab = model.aberration(base_coeffs[None])
    coefficients["base"] = base_coeffs

    for seed in range(int(args.seeds)):
        torch.manual_seed(seed)
        np.random.seed(seed)
        phases = _train_phases(model, int(args.train_phases), seed)

        fits = {}
        for arm, physics, weights in FITS:
            fit = fit_residual(
                device,
                model,
                phases,
                window=window,
                epochs=int(args.epochs),
                lr=float(args.lr),
                seed=seed,
                physics=physics,
                weights=weights,
                aberration=base_ab,
                exposures=exposures,
            )
            fits[arm] = fit
            coefficients[f"{arm}_{seed}"] = fit.coeffs
            coefficients[f"{arm}_delta_{seed}"] = fit.delta
            if arm in curves:
                for term in TERMS:
                    curves[arm].setdefault(term, [])
                    if seed == 0:
                        curves[arm][term] = list(fit.history.get(term, []))
            runs.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "phase": "fit",
                    "l1_start": fit.l1_start,
                    "l1_end": fit.l1_end,
                    "ok": fit.ok,
                    "reason": fit.reason,
                    "weights": fit.weights,
                }
            )

        arms = {
            "base": (base_ab, None),
            "oracle": (oracle_ab, None),
            **{arm: (base_ab, fits[arm].delta) for arm, _, _ in FITS},
        }
        for name, phase in tests.items():
            for arm in ARMS:
                aberration, delta = arms[arm]
                metrics = _score(device, model, phase, aberration, delta, window)
                runs.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "phase": name,
                        "metrics": metrics,
                        "l1_start": None,
                        "l1_end": None,
                    }
                )
                if seed == 0:
                    with torch.no_grad():
                        command = phase
                        if delta is not None:
                            command = (
                                phase
                                + torch.as_tensor(
                                    delta, device=phase.device, dtype=phase.dtype
                                )[None]
                            )
                        predicted = prepare_frame(
                            model.forward(command, aberration=aberration)
                        )
                        truth = prepare_frame(device.forward(phase))
                    cases[name][arm] = predicted[0, 0].cpu().numpy()
                    cases[name]["truth"] = truth[0, 0].cpu().numpy()

    rows = [r for r in runs if r["phase"] != "fit"]
    summary: dict[str, object] = {
        "command": " ".join(["python", "-m", "scripts.ml.sim2f_pinn_extrapolation"]),
        "argv": vars(args) | {"out": str(out), "exposures": exposures},
        "config": {
            "grid": args.grid,
            "ccd_px": args.ccd,
            "window": args.window,
            "train_phases": args.train_phases,
            "epochs": args.epochs,
            "lr": args.lr,
            "seeds": args.seeds,
            "weights": DEFAULT_WEIGHTS,
            "support_sweep": [w for _, w in SUPPORT_SWEEP],
            "arm_weights": {
                name: (
                    weights
                    if weights is not None
                    else (
                        "DEFAULT_WEIGHTS (physics=True)"
                        if physics
                        else "physics=False (data+smooth)"
                    )
                )
                for name, physics, weights in FITS
            },
            "errors": {
                "slm_zernike": list(ERRORS.slm_zernike),
                "slm_roughness_sigma_rad": ERRORS.slm_roughness_sigma_rad,
            },
        },
        "arms": {},
        "stage1": {},
        "runtime_s": None,
    }
    for arm in ARMS:
        summary["arms"][arm] = {}
        for metric in METRICS:
            values = np.array(
                [r["metrics"][metric] for r in rows if r["arm"] == arm],
                dtype=np.float64,
            )
            summary["arms"][arm][metric] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "n": int(values.size),
            }
    for arm in (name for name, _, _ in FITS):
        starts = [
            r["l1_start"] for r in runs if r["arm"] == arm and r["phase"] == "fit"
        ]
        ends = [r["l1_end"] for r in runs if r["arm"] == arm and r["phase"] == "fit"]
        summary["arms"][arm]["window_l1"] = {
            "start": float(np.nanmean(starts)),
            "end": float(np.nanmean(ends)),
        }
    summary["stage1"] = {
        "probe_n_modes": 10,
        "probe_noll": "4..11",
        "residual": float(base_residual),
    }

    # Per-phase aggregation (addition 2 of the brief).  The pooled means above
    # are dominated by ``tiny``, whose ``l1_out`` has a near-zero denominator, so
    # every arm is also broken out per test phase.  ``out_energy_frac`` is a
    # property of the true frame alone (identical for every arm), so it is also
    # stated once per phase -- it is what makes ``l1_out`` interpretable there.
    summary["by_phase"] = {}
    for phase in tests:
        phase_rows = [r for r in rows if r["phase"] == phase]
        entry: dict[str, object] = {
            "n_seeds": len({r["seed"] for r in phase_rows}),
            "arms": {},
        }
        for arm in ARMS:
            cell: dict[str, dict[str, float | int]] = {}
            for metric in METRICS:
                values = np.array(
                    [r["metrics"][metric] for r in phase_rows if r["arm"] == arm],
                    dtype=np.float64,
                )
                cell[metric] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                    "n": int(values.size),
                }
            entry["arms"][arm] = cell
        for metric in ("out_energy_frac", "l1_out_denom"):
            values = np.array(
                [r["metrics"][metric] for r in phase_rows if r["arm"] == "base"],
                dtype=np.float64,
            )
            entry[metric] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "n": int(values.size),
            }
        summary["by_phase"][phase] = entry
    summary["phase_out_energy_frac"] = {
        phase: summary["by_phase"][phase]["out_energy_frac"]["mean"] for phase in tests
    }

    summary["runtime_s"] = round(time.perf_counter() - started, 1)
    (out / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(out / "coeffs.npz", **coefficients)
    _plot_losses(curves, out / "loss_curves.png")
    _plot_cases(cases, window, out / "fig_pinn_cases.png")

    for arm in ARMS:
        entry = summary["arms"][arm]
        print(
            f"{arm:>10}: l1_out={entry['l1_out']['mean']:.4f} "
            f"l1_in={entry['l1_in']['mean']:.4f} "
            f"ee_out_err={entry['ee_out_err']['mean']:.4f} "
            f"centroid_err_px={entry['centroid_err_px']['mean']:.2f} "
            f"d86_err_px={entry['d86_err_px']['mean']:.2f}"
        )
    for phase, block in summary["by_phase"].items():
        print(
            f"[{phase}] out_energy_frac={block['out_energy_frac']['mean']:.4g} "
            f"l1_out_denom={block['l1_out_denom']['mean']:.4g}"
        )
        for arm in ARMS:
            cell = block["arms"][arm]
            print(
                f"  {arm:>18}: l1_out={cell['l1_out']['mean']:.4f}"
                f"±{cell['l1_out']['std']:.4f} "
                f"ncc_out={cell['ncc_out']['mean']:.4f} "
                f"l1_in={cell['l1_in']['mean']:.4f}"
            )
    print(f"runtime {summary['runtime_s']}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
