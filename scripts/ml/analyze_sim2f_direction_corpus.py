"""Does the 2f twin predict the *change* a real measurement sees?

`report/sim2f_bench/twin_vs_hw_corpus/20261027/report.md` answers one question about
absolute frames: over the 2026-10-25 corpus the twin's prediction is **not similar**
to the measured frame (normalised L1 never reaches the 0.5 gate at any geometry scale
``k``). That verdict is correct and deliberately harsh, but it is not the question a
gradient-based optimizer asks. An optimizer never needs the absolute frame; it needs
the *direction* -- perturb the phase, and ask whether the predicted change points the
same way as the measured change.

This script therefore re-reads the **same** corpus and compares differences instead of
frames:

* records come from :func:`scripts.ml.validate_sim2f_vs_hw_corpus.measure_corpus` --
  the single corpus loader, deduplication rule, twin, ``k*`` selection and metrics are
  reused verbatim, so no second implementation of any of that exists here;
* every unordered pair of distinct ``(phase, frame)`` records becomes one
  ``(predicted_plus/minus, measured_plus/minus, core)`` quadruple, handed to the
  canonical :func:`ao_shaping.utils.image.physical_model_metrics.compute_directional_metrics`;
* the *null* is a deliberately mismatched pairing (the predicted difference of pair A
  against the measured difference of pair B). Mismatched pairs carry no information
  about each other, so whatever cosine they reach is the noise/geometry floor the real
  pairs must beat.

Criteria are fixed here, in :data:`COSINE_GATE` / :data:`SIGN_GATE` / :data:`CONTROL_MARGIN`,
before any number is computed:

* ``direction_ok`` -- median cosine over real pairs ``>= COSINE_GATE`` **and** it clears
  the mismatched-pair median by ``>= CONTROL_MARGIN``;
* ``sign_ok`` -- median sign agreement ``>= SIGN_GATE``.

Nothing here is a device run: the corpus is read from ``data/debug/`` and the twin runs
on CPU.

CLI::

    python scripts/ml/analyze_sim2f_direction_corpus.py
    python scripts/ml/analyze_sim2f_direction_corpus.py --limit-files 6   # smoke
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
# Order matters: `src` lands in front of the repo root so `ml` resolves to
# `src/ml`. The root itself is needed only for `scripts.*` -- the shared `_common`
# helpers and the corpus loader reused below. `scripts/` is never put on the path:
# its `scripts/ml/` folder would shadow `src/ml`
# (see `tests/ao_shaping/scripts/test_script_file_form_imports.py`).
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from ao_shaping.tools.slm.transfer.radiometry import count_rate
from ao_shaping.utils.image.physical_model_metrics import compute_directional_metrics
from scripts._common import markdown_table, savefig
from scripts.ml.validate_sim2f_vs_hw_corpus import (
    K_GRID,
    describe,
    measure_corpus,
    zero_mean_ncc,
)

OUT_DIR = ROOT / "report" / "simulation" / "sim_credibility"
ARTIFACT_DIR = OUT_DIR / "artifacts"
FIG_DIR = OUT_DIR / "figures"
STAMP = "20261030"

#: Written before the measurement: a pair's direction is accepted when the median
#: cosine clears this gate ...
COSINE_GATE = 0.5
#: ... by at least this much over the mismatched-pair control ...
CONTROL_MARGIN = 0.2
#: ... and the median sign agreement clears this gate.
SIGN_GATE = 0.75
#: A pair is scored only on pixels where the measured change reaches this fraction
#: of its own peak (see :func:`_support`), and only when that leaves this many.
DIFF_FRACTION = 0.15
MIN_CORE_PX = 20
#: Pixels at most this fraction of each frame's peak are treated as background, where
#: the frame difference is read noise; at least this many such pixels are required.
DARK_FRACTION = 0.10
MIN_DARK_PX = 200
#: Deterministic control pairing.
CONTROL_SEED = 0
CONTROL_TRIALS = 400
#: A subset needs this many scored pairs before its gates are declared passed/failed.
MIN_PAIRS = 10
#: Registration tolerance for the post-hoc upper bound, in measured-window pixels.
SHIFT_PX = 3


def _support(measured_plus: np.ndarray, measured_minus: np.ndarray) -> np.ndarray:
    """Pixels where the **measured** change is large: ``|Δ| >= DIFF_FRACTION * max|Δ|``.

    The support is picked from the measurement alone, never from the model, so the twin
    cannot choose the pixels it is scored on. A frame-intensity mask (5% of peak) would
    admit the background, where the frame difference is read noise and carries no
    directional information -- scoring a direction there drags every cosine toward zero
    for reasons that have nothing to do with the model.
    """
    delta = np.abs(np.asarray(measured_plus, dtype=np.float64) - np.asarray(measured_minus, dtype=np.float64))
    peak = float(np.max(delta))
    if not math.isfinite(peak) or peak <= 0.0:
        return np.zeros(delta.shape, dtype=bool)
    return delta >= DIFF_FRACTION * peak


def _difference_snr(
    measured_plus: np.ndarray, measured_minus: np.ndarray, support: np.ndarray
) -> dict[str, float | int]:
    """Measured-change amplitude against the frame-difference read noise.

    The noise sigma is read off pixels that are dark in **both** frames (the window's
    background), where a frame-to-frame difference is read noise only. A measured change
    whose RMS inside the support does not clear that floor cannot decide any direction
    question, whatever the model predicts -- so this number gates the interpretation of
    the cosine instead of letting a zero mean "the twin is wrong".
    """
    plus = np.asarray(measured_plus, dtype=np.float64)
    minus = np.asarray(measured_minus, dtype=np.float64)
    dark = (plus <= DARK_FRACTION * float(np.max(plus))) & (minus <= DARK_FRACTION * float(np.max(minus)))
    delta = plus - minus
    empty = {"snr": float("nan"), "noise_sigma": float("nan"), "delta_rms": float("nan"), "dark_px": int(dark.sum())}
    if int(dark.sum()) < MIN_DARK_PX or not support.any():
        return empty
    noise = float(np.std(delta[dark]))
    rms = float(np.sqrt(np.mean(delta[support] ** 2)))
    return {
        "snr": rms / noise if noise > 0.0 else float("nan"),
        "noise_sigma": noise,
        "delta_rms": rms,
        "dark_px": int(dark.sum()),
    }


def _selftest() -> dict[str, float]:
    """The metric plumbing must score an exact match as 1 and a swap as -1.

    A direction analysis that reports ~0 for every pair is only believable if the
    instrument itself is known to read 1 on a perfect agreement; this runs on every
    invocation and raises instead of silently reporting a plausible-looking zero.
    """
    rng = np.random.default_rng(7)
    first, second = rng.random((16, 16)), rng.random((16, 16))
    core = np.ones((16, 16), dtype=bool)
    identical = compute_directional_metrics(first, second, first, second, core)["direction_cosine"]
    swapped = compute_directional_metrics(first, second, second, first, core)["direction_cosine"]
    if not (abs(identical - 1.0) < 1e-12 and abs(swapped + 1.0) < 1e-12):
        raise RuntimeError(f"directional metric self-test failed: identical={identical!r} swapped={swapped!r}")
    return {"identical": float(identical), "swapped": float(swapped)}


def _parallel_cosine(delta: np.ndarray, frame: np.ndarray, support: np.ndarray) -> float:
    """How much of a difference is "the frame got brighter" rather than "the light moved".

    A between-record difference on a real bench carries a global intensity term (session
    drift, an exposure re-read): that term is parallel to the frame itself. Projecting the
    difference onto the zero-mean frame measures how much of it is that term, so a pair
    whose difference is mostly drift is visible in the numbers instead of being read as
    model error.
    """
    d = np.asarray(delta, dtype=np.float64)[support]
    f = np.asarray(frame, dtype=np.float64)[support]
    if d.size == 0:
        return float("nan")
    d = d - d.mean()
    f = f - f.mean()
    nd, nf = float(np.linalg.norm(d)), float(np.linalg.norm(f))
    if nd == 0.0 or nf == 0.0:
        return float("nan")
    return float(d @ f / (nd * nf))


def _shift_cosine(
    delta_predicted: np.ndarray, delta_measured: np.ndarray, support: np.ndarray, shift: tuple[int, int]
) -> float:
    """Cosine after translating the predicted difference by ``shift`` pixels."""
    rolled = np.roll(np.roll(delta_predicted, shift[0], axis=0), shift[1], axis=1)
    moved = np.roll(np.roll(support, shift[0], axis=0), shift[1], axis=1)
    a = rolled[moved] - rolled[moved].mean()
    b = delta_measured[moved] - delta_measured[moved].mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(a @ b / (na * nb))


def _best_shift(
    delta_predicted: np.ndarray, delta_measured: np.ndarray, support: np.ndarray, radius: int = SHIFT_PX
) -> dict[str, float | int]:
    """Largest cosine over integer translations within ``radius`` pixels.

    A *post-hoc* upper bound, not a gate: the change may be right while its registration
    is wrong (the twin's ``k*``/axis is a single global guess). If a small shift recovers
    the cosine, the defect is registration, not physics -- which is a different, and
    cheaper, thing to fix. Reported as an upper bound because it maximises over
    ``(2*radius+1)**2`` hypotheses.
    """
    best = {"cosine": float("nan"), "dy": 0, "dx": 0}
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            value = _shift_cosine(delta_predicted, delta_measured, support, (dy, dx))
            if math.isfinite(value) and (not math.isfinite(best["cosine"]) or value > best["cosine"]):
                best = {"cosine": value, "dy": dy, "dx": dx}
    return best


def _direction_row(
    left: dict[str, Any],
    right: dict[str, Any],
    measured: tuple[np.ndarray, np.ndarray],
    predicted: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any] | None:
    """One pair: measured/predicted differences scored by the canonical metric."""
    core = _support(*measured)
    if int(core.sum()) < MIN_CORE_PX:
        return None
    metrics = compute_directional_metrics(predicted[0], predicted[1], measured[0], measured[1], core)
    delta_measured = measured[0] - measured[1]
    delta_predicted = predicted[0] - predicted[1]
    frame = 0.5 * (measured[0] + measured[1])
    return {
        "left": left["path"],
        "right": right["path"],
        "left_index": left["index"],
        "right_index": right["index"],
        "mode_id_left": left["mode_id"],
        "mode_id_right": right["mode_id"],
        "coefficient_left": left["coefficient"],
        "coefficient_right": right["coefficient"],
        "probe_left": left["probe"],
        "probe_right": right["probe"],
        "same_mode": left["mode_id"] == right["mode_id"],
        "same_file": left["path"] == right["path"],
        "core_px": int(core.sum()),
        "ncc_left": zero_mean_ncc(measured[0], predicted[0]),
        "ncc_right": zero_mean_ncc(measured[1], predicted[1]),
        "drift_measured": _parallel_cosine(delta_measured, frame, core),
        "drift_predicted": _parallel_cosine(delta_predicted, frame, core),
        "best_shift": _best_shift(delta_predicted, delta_measured, core),
        "snr": _difference_snr(measured[0], measured[1], core),
        "direction": metrics,
    }


def _prepare(measurement: dict[str, Any]) -> list[dict[str, Any]]:
    """Exposure-normalise both sides once, then hold (record, measured, predicted).

    Measured frames go through :func:`ao_shaping.tools.slm.transfer.radiometry.count_rate`
    -- the repo's only exposure normalisation entry point -- so pairs captured at
    different exposures are still comparable. The twin prediction is a relative
    intensity (``detector=False``) and carries no exposure, so only scale-free
    quantities (correlation, cosine, sign agreement) are read off a pair.
    """
    prepared: list[dict[str, Any]] = []
    for record in measurement["distinct"]:
        measured = np.asarray(record["measured"], dtype=np.float64)
        if record.get("exposure_ms"):
            measured = np.asarray(count_rate(measured, {"exposure_ms": record["exposure_ms"]}))
        prepared.append(
            {
                "record": record,
                "measured": measured,
                # `twin_at_k` is the twin sampled onto the measured grid at `k*`
                # by `measure_corpus` itself -- no second resampling here.
                "predicted": np.asarray(record["twin_at_k"], dtype=np.float64),
            }
        )
    return prepared


def _verdict(rows: Sequence[dict[str, Any]], control: Sequence[float], *, pairing: str) -> dict[str, Any]:
    """Apply the pre-registered gates to one pairing of the same rows.

    ``enough_pairs`` is part of the verdict: below :data:`MIN_PAIRS` a median is reported
    but no gate is declared passed, so a thin subset cannot masquerade as a result.
    """
    cosines = [r["direction"]["direction_cosine"] for r in rows if r["direction"]["direction_cosine"] is not None]
    signs = [
        r["direction"]["direction_sign_agreement"]
        for r in rows
        if r["direction"]["direction_sign_agreement"] is not None
    ]
    ncc = [value for r in rows for value in (r["ncc_left"], r["ncc_right"]) if math.isfinite(value)]
    real, ctrl, sign = describe(cosines), describe(control), describe(signs)
    enough = len(cosines) >= MIN_PAIRS
    return {
        "pairing": pairing,
        "pairs": len(cosines),
        "enough_pairs": bool(enough),
        "median_cosine": real["median"],
        "median_control_cosine": ctrl["median"],
        "control_margin": real["median"] - ctrl["median"],
        "median_sign_agreement": sign["median"],
        "median_absolute_ncc": describe(ncc)["median"],
        "median_snr": describe([r["snr"]["snr"] for r in rows])["median"],
        "median_drift_measured": describe([r["drift_measured"] for r in rows])["median"],
        "median_cosine_best_shift": describe([r["best_shift"]["cosine"] for r in rows])["median"],
        "direction_ok": bool(
            enough
            and real["median"] >= COSINE_GATE
            and (real["median"] - ctrl["median"]) >= CONTROL_MARGIN
        ),
        "sign_ok": bool(enough and sign["median"] >= SIGN_GATE),
        "cosine_gate": COSINE_GATE,
        "control_margin_gate": CONTROL_MARGIN,
        "sign_gate": SIGN_GATE,
    }


def _score(prepared: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(prepared, 2):
        row = _direction_row(
            left["record"],
            right["record"],
            (left["measured"], right["measured"]),
            (left["predicted"], right["predicted"]),
        )
        if row is not None:
            rows.append(row)

    rng = np.random.default_rng(CONTROL_SEED)
    control: list[float] = []
    for _ in range(CONTROL_TRIALS):
        a, b = rng.choice(len(prepared), size=2, replace=False)
        c, d = rng.choice(len(prepared), size=2, replace=False)
        if len({int(a), int(b), int(c), int(d)}) < 3:
            continue
        core = _support(prepared[int(a)]["measured"], prepared[int(b)]["measured"])
        if int(core.sum()) < MIN_CORE_PX:
            continue
        metrics = compute_directional_metrics(
            prepared[int(c)]["predicted"],
            prepared[int(d)]["predicted"],
            prepared[int(a)]["measured"],
            prepared[int(b)]["measured"],
            core,
        )
        if metrics["direction_cosine"] is not None:
            control.append(float(metrics["direction_cosine"]))

    cosines = [r["direction"]["direction_cosine"] for r in rows if r["direction"]["direction_cosine"] is not None]
    signs = [
        r["direction"]["direction_sign_agreement"]
        for r in rows
        if r["direction"]["direction_sign_agreement"] is not None
    ]
    ncc = [value for r in rows for value in (r["ncc_left"], r["ncc_right"]) if math.isfinite(value)]

    real = describe(cosines)
    ctrl = describe(control)
    verdict = _verdict(rows, control, pairing="all")

    within = [r for r in rows if r["same_file"]]
    within_verdict = _verdict(within, control, pairing="same_file")
    primary = within_verdict if within_verdict["pairs"] >= MIN_PAIRS else verdict

    return {
        "rows": rows,
        "control_cosines": control,
        "primary": primary,
        "summary": {
            "real": real,
            "control": ctrl,
            "sign": describe(signs),
            "absolute_ncc": describe(ncc),
            "same_mode": describe(
                [r["direction"]["direction_cosine"] for r in rows if r["same_mode"]]
            ),
            "cross_mode": describe(
                [r["direction"]["direction_cosine"] for r in rows if not r["same_mode"]]
            ),
            "snr": describe([r["snr"]["snr"] for r in rows]),
            "drift_measured": describe([r["drift_measured"] for r in rows]),
            "drift_predicted": describe([r["drift_predicted"] for r in rows]),
            "verdict": verdict,
            "verdict_same_file": within_verdict,
            "primary_pairing": primary["pairing"],
        },
    }


def _figure(result: dict[str, Any]) -> Path:
    rows = result["rows"]
    cosines = [r["direction"]["direction_cosine"] for r in rows if r["direction"]["direction_cosine"] is not None]
    snr = [r["snr"]["snr"] for r in rows]
    drift = [r["drift_measured"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axes[0].hist(cosines, bins=24, range=(-1.0, 1.0), color="#3b6ea5", edgecolor="white", label="all pairs")
    axes[0].hist(
        np.clip(result["control_cosines"], -1.0, 1.0),
        bins=24,
        range=(-1.0, 1.0),
        histtype="step",
        color="#b0483a",
        label="mismatched control",
    )
    same_file = [r["direction"]["direction_cosine"] for r in rows if r["same_file"]]
    if same_file:
        axes[0].hist(
            same_file,
            bins=24,
            range=(-1.0, 1.0),
            histtype="step",
            color="#2f7a4f",
            label="same dump",
        )
    axes[0].axvline(COSINE_GATE, color="black", linestyle="--", linewidth=1.0, label=f"gate {COSINE_GATE:g}")
    axes[0].set_xlabel("direction cosine (predicted Δ vs measured Δ)")
    axes[0].set_ylabel("pairs")
    axes[0].legend(fontsize=8)
    axes[1].scatter(drift, cosines, s=18, color="#3b6ea5")
    axes[1].axhline(COSINE_GATE, color="black", linestyle="--", linewidth=1.0)
    axes[1].axhline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("measured Δ ∥ frame (share that is global brightness drift)")
    axes[1].set_ylabel("direction cosine")
    fig.tight_layout()
    path = FIG_DIR / f"direction_corpus_{STAMP}.png"
    savefig(fig, path)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-files", type=int, default=None, help="smoke-run cap on pickles")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    selftest = _selftest()
    measurement = measure_corpus(args.limit_files, progress_every=args.progress_every)
    prepared = _prepare(measurement)
    result = _score(prepared)
    fig = _figure(result)
    elapsed = time.perf_counter() - started

    payload = {
        "k_star": measurement["k_star"],
        "k_grid": list(K_GRID),
        "distinct_pairs": len(prepared),
        "exposures_ms": sorted({r["exposure_ms"] for r in measurement["distinct"]}),
        "replicas": describe([r.get("replicas", 1) for r in measurement["distinct"]]),
        "corpus": measurement["config"],
        "summary": result["summary"],
        "selftest": selftest,
        "control_trials": len(result["control_cosines"]),
        "rows": result["rows"],
        "figure": str(fig.relative_to(ROOT)).replace("\\", "/"),
        "elapsed_s": elapsed,
        "producer": "scripts/ml/analyze_sim2f_direction_corpus.py",
    }
    out = ARTIFACT_DIR / f"direction_corpus_{STAMP}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    verdict = result["primary"]
    snr = result["summary"]["snr"]
    drift = result["summary"]["drift_measured"]
    logger.info(
        f"pairs={verdict['pairs']} pairing={verdict['pairing']} "
        f"median_cosine={verdict['median_cosine']:.4f} "
        f"control={verdict['median_control_cosine']:.4f} "
        f"sign={verdict['median_sign_agreement']:.3f} "
        f"snr={snr['median']:.2f} drift={drift['median']:.3f} -> {out}"
    )
    print(markdown_table(
        ("pair statistics", "value"),
        (
            ("distinct records", len(prepared)),
            ("pairs scored (all)", result["summary"]["verdict"]["pairs"]),
            ("pairs scored (same dump)", result["summary"]["verdict_same_file"]["pairs"]),
            ("primary pairing", verdict["pairing"]),
            ("median direction cosine", f"{verdict['median_cosine']:.4f}"),
            ("median control cosine", f"{verdict['median_control_cosine']:.4f}"),
            ("median sign agreement", f"{verdict['median_sign_agreement']:.4f}"),
            ("median absolute-frame ncc", f"{verdict['median_absolute_ncc']:.4f}"),
            ("median measured Δ SNR", f"{snr['median']:.2f}"),
            ("median measured Δ drift cosine", f"{drift['median']:.3f}"),
            ("median predicted Δ drift cosine", f"{result['summary']['drift_predicted']['median']:.3f}"),
            ("median cosine, best ±3 px shift (upper bound)", f"{verdict['median_cosine_best_shift']:.3f}"),
            ("direction_ok", str(verdict["direction_ok"])),
            ("sign_ok", str(verdict["sign_ok"])),
            ("elapsed", f"{elapsed:.1f} s"),
        ),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
