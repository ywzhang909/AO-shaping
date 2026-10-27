"""Two-arm comparison of the sim2f closed loop: ``iid`` vs ``kolmogorov`` states.

What this measures
==================
The sim2f ablation was run with one aberration ensemble: ``AberrationConfig``'s
default ``aberration_model="iid"``, which draws the high-order Zernike
coefficients ``N(0, 1)`` per mode.  Real turbulence is nothing like that -- its
modes are strongly correlated and the variance falls steeply with radial order --
so ``aberration_model="kolmogorov"`` (von Karman covariance, commit ``38303f58``)
exists.  This driver runs **the same corpus and the same checkpoint** through
``ml.sim2f.loop.run_loop`` twice, changing *only* the ensemble the per-sample
states are drawn from, and asks whether the net-vs-flat ordering survives.

Why one corpus (and one checkpoint) is enough
=============================================
The ablation is an **evaluation-ensemble** comparison, not a training-distribution
one:

* :func:`ml.sim2f.synth.build_pairs` runs the teacher on the *aberration-free*
  bench (``Sim2f(sim_cfg)`` with no state at all) and its archive
  (``pairs.npz``, :data:`ml.sim2f.schema.CACHE_FIELDS`) carries **no aberration
  field** -- ``target``, ``support``, ``phase``, ``spot``, ``teacher_metrics``,
  ``kind``, ``seed``, ``split``, and that is all eight;
* the state is drawn per sample *inside* the loop:
  ``loop.run_loop`` L479 ``state = draw_aberration(bench, states, rng)`` with
  ``states = AberrationConfig() if aberration_cfg is None else aberration_cfg``
  (L442) and ``rng = np.random.default_rng([cfg.seed, index])`` (L478).

So ``aberration_model`` cannot reach a label or a weight.  ``verify_conceptual``
re-checks this at runtime instead of taking it on trust.

Determinism / pairing
=====================
The state RNG is ``[seed, sample_index]`` and does not depend on the ensemble, and
:func:`ml.sim2f.optics.draw_aberration` consumes the **same number of draws** on
both paths (``standard_normal(n_high)`` first, then the tilt / illumination
uniforms).  Therefore for a given sample the two arms share the total RMS
(exactly: both are rescaled to the same ``target`` drawn after it), the tip/tilt
and the illumination state -- only ``coeffs[2:]``'s *shape* differs.  The driver
asserts that pairing.

Nothing here is committed: ``runs/`` is untracked by design.

Usage
=====
    python runs/sim2f_kl_ab/run_two_arms.py
    python runs/sim2f_kl_ab/run_two_arms.py --no-noise-floor   # 2 runs instead of 4
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from math import comb
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

DEFAULT_CACHE = "runs/sim2f_fair/corpus"
DEFAULT_CHECKPOINT = "runs/sim2f_fair/train/model_best.pt"
DEFAULT_BASELINE = "runs/sim2f_fair/loop/a_no_noise/eval_loop.json"

ARMS: tuple[str, ...] = ("flat", "net_open", "net_probe", "net_closed", "oracle")
#: Family comparison the fair report uses: "does the arm's family-mean objective
#: come in under ``flat``'s?"  Counted over families, never over samples.
CHALLENGERS: tuple[str, ...] = ("net_open", "net_probe", "net_closed", "oracle")

#: Verb text of the pre-registration.  Written here, in the source, **before** the
#: arms were run, and rendered verbatim into ``two_arms.md``.  Editing it after
#: seeing numbers is exactly what it exists to prevent.
PRE_REGISTRATION: dict[str, Any] = {
    "written_before_the_run": True,
    "expectation": (
        "The ordering does NOT change.  ``flat`` still beats every net arm on the "
        "calibrated bench, under both ensembles."
    ),
    "reasoning": (
        "The calibrated 2f twin's dominant optical error is the static defocus "
        "(10.0847 mm), not turbulence.  The drawn aberration state is a *second* "
        "error on top of it: 0.05-0.45 rad RMS over Noll 4..22 plus 0-6 px of "
        "tip/tilt.  Changing the *shape* of that ensemble's high-order part cannot "
        "repair the defocus-dominated flat spot, which is what the fair run "
        "measured (``net_open`` 1.4588 vs ``flat`` 1.1160); the plausible outcome "
        "is therefore that the numbers move and the ordering does not."
    ),
    "what_would_be_significant": (
        "``flat`` losing the aggregate under the kolmogorov ensemble, or the "
        "net-vs-flat family count rising above 0/7 far enough to change the "
        "published verdict.  Either would be reported as a significant result and "
        "not folded into 'the numbers moved'."
    ),
    "what_is_not_evidence": (
        "A change smaller than the state-draw noise floor measured by the "
        "seed-swap replicates.  The detector is OFF in this arm (``a_no_noise``), "
        "so the *only* stochasticity left is which 24 aberration states were "
        "drawn -- a different 24 states is the right noise floor, not a rerun of "
        "the same 24."
    ),
}


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _now() -> float:
    """Wall-clock seconds (monotonic)."""
    return time.perf_counter()


def _git(*args: str) -> str:
    """Run a git command from the repo root, returning stripped stdout."""
    try:
        done = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _reuse_arm(arm: str, out_dir: Path) -> dict[str, Any]:
    """Re-read an arm's existing ``eval_loop.json`` instead of running the loop.

    Args:
        arm: Arm label (also the directory name under ``loop/``).
        out_dir: Directory holding ``eval_loop.json``.

    Returns:
        The same record shape :func:`run_arm` returns, with ``wall_s`` = 0.

    Raises:
        FileNotFoundError: If the report is missing.
    """
    path = out_dir / "eval_loop.json"
    if not path.is_file():
        raise FileNotFoundError(f"--reuse needs an existing report: {path}")
    return {
        "arm": arm,
        "wall_s": 0.0,
        "out_dir": out_dir.relative_to(REPO).as_posix() if out_dir.is_relative_to(REPO) else out_dir.as_posix(),
        "eval_loop": path.relative_to(REPO).as_posix(),
        "payload": _load_json(path),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically enough for a report artefact (UTF-8, no BOM).

    Non-finite floats are replaced with ``None`` on the way out, exactly as
    ``ml.sim2f.loop`` does for its own payload: a NaN makes the file invalid JSON
    for every strict reader (RFC 8259 has no NaN), and every NaN here means "not
    applicable" rather than "a number".
    """
    path.write_text(
        json.dumps(_finite(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _finite(value: Any) -> Any:
    """Replace NaN / ±inf with ``None``, recursively.

    Args:
        value: Any JSON-shaped object.

    Returns:
        The same structure with non-finite floats replaced by ``None``.
    """
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(value: float | None, spec: str = ".4f") -> str:
    """Format an optional float, ``'-'`` when absent or not finite."""
    if value is None:
        return "-"
    number = float(value)
    if not math.isfinite(number):
        return "-"
    return format(number, spec)


# ----------------------------------------------------------------------
# 1. Conceptual verification -- re-check the claim at runtime
# ----------------------------------------------------------------------
def verify_conceptual(cache_dir: Path) -> dict[str, Any]:
    """Prove the corpus carries no aberration state and the labels cannot see it.

    Three independent checks, none of which trusts a docstring:

    1. the archive's key set equals :data:`ml.sim2f.schema.CACHE_FIELDS` -- no
       extra field could hide a state;
    2. no archive field name *mentions* aberration / zernike / state / coeff;
    3. the teacher's own docstring/behaviour: ``build_pairs`` constructs
       ``Sim2f(sim_cfg)`` and calls ``teacher.optimise`` with no aberration
       argument, and the forward call inside passes none.  Checked by source
       scan of the two modules for ``draw_aberration`` / ``AberrationConfig``.

    Args:
        cache_dir: Corpus directory holding ``pairs.npz`` + ``meta.json``.

    Returns:
        A JSON-able evidence record.
    """
    import inspect

    from ml.sim2f import schema, synth
    from ml.sim2f.synth import load_pairs

    arrays, meta = load_pairs(cache_dir)
    declared = tuple(field.name for field in schema.CACHE_FIELDS)
    on_disk = tuple(sorted(arrays))
    suspicious_tokens = ("aberr", "zern", "state", "coeff", "turb")
    suspicious = [name for name in on_disk if any(t in name.lower() for t in suspicious_tokens)]

    synth_src = inspect.getsource(synth)
    teacher_src = inspect.getsource(__import__("ml.sim2f.teacher", fromlist=["x"]))
    _schema_src = inspect.getsource(schema)

    return {
        "archive_fields": list(on_disk),
        "declared_fields": list(declared),
        "fields_match_schema": on_disk == tuple(sorted(declared)),
        "fields_naming_an_aberration_state": suspicious,
        "carries_aberration_state": bool(suspicious),
        "synth_documents_an_aberration_free_bench": "aberration-free" in synth_src.lower(),
        "schema_declares_phase_aberration_free": "ABERRATION-FREE" in _schema_src,
        "synth_passes_an_aberration_kwarg": "aberration=" in synth_src,
        "synth_mentions_draw_aberration": "draw_aberration" in synth_src,
        "synth_mentions_AberrationConfig": "AberrationConfig" in synth_src,
        "teacher_documents_an_aberration_free_bench": "aberration-free bench" in teacher_src,
        "teacher_passes_an_aberration_kwarg": "aberration=" in teacher_src,
        "corpus_meta_aberration_block": meta.get("aberration"),
        "corpus_meta": {
            key: meta.get(key) for key in ("version", "counts", "wall_s", "git_commit")
        },
        "loop_draw_site": "src/ml/sim2f/loop.py L442 (states) / L478-483 (draw_aberration)",
        "note": (
            "The teacher runs on the aberration-free bench by construction: "
            "`build_pairs` builds Sim2f(sim_cfg), never draws a state, and its only "
            "forward call is `sim.forward(zeros, detector=False)`. The teacher's own "
            "module docstring says so ('Why the teacher runs on the aberration-free "
            "bench'). So `aberration_model` cannot reach a label."
        ),
    }


# ----------------------------------------------------------------------
# 2. The two arms
# ----------------------------------------------------------------------
def build_configs() -> dict[str, Any]:
    """Build the two ``AberrationConfig`` objects, differing only in the model.

    Returns:
        ``{arm: AberrationConfig}``.

    Raises:
        ValueError: If the two configs differ anywhere except in
            ``aberration_model`` / ``covariance_path`` -- the whole comparison
            rests on that.
    """
    from ml.sim2f.config import AberrationConfig
    from ml.sim2f.optics import DEFAULT_COVARIANCE_PATH

    iid = AberrationConfig()
    ko = AberrationConfig(
        aberration_model="kolmogorov", covariance_path=str(DEFAULT_COVARIANCE_PATH)
    )
    left, right = iid.to_dict(), ko.to_dict()
    differing = {key for key in left if left[key] != right[key]}
    allowed = {"aberration_model", "covariance_path"}
    if differing - allowed:
        raise ValueError(
            "the two arms must differ only in the ensemble, but these also differ: "
            f"{sorted(differing - allowed)}"
        )
    return {"iid": iid, "kolmogorov": ko}


def run_arm(
    arm: str,
    aberration_cfg: Any,
    *,
    cache_dir: Path,
    checkpoint: Path,
    out_dir: Path,
    samples: int,
    rounds: int,
    seed: int,
    ladder: tuple[float | None, ...] | None,
) -> dict[str, Any]:
    """Run one arm's five-arm loop and return the payload plus its wall clock.

    Args:
        arm: ``"iid"`` or ``"kolmogorov"`` (also ``"iid_seed8"`` / ... for the
            noise-floor replicates).
        aberration_cfg: The ensemble contract.
        cache_dir: Corpus directory.
        checkpoint: Network checkpoint.
        out_dir: Report directory.
        samples: Number of states.
        rounds: Closed-loop rounds.
        seed: ``LoopConfig.seed`` -- drives the sample order *and* the states.
        ladder: Exposures for the fit-error noise ladder.

    Returns:
        ``{"arm", "wall_s", "out_dir", "eval_loop", "payload"}``.
    """
    from ml.sim2f.config import LoopConfig
    from ml.sim2f.loop import run_loop

    cfg = LoopConfig(
        samples=samples,
        rounds=rounds,
        probe_fit=True,
        refit=True,
        use_detector=False,
        seed=seed,
        out_dir=str(out_dir),
    )
    started = _now()
    payload = run_loop(
        cfg,
        cache_dir=cache_dir,
        checkpoint=checkpoint,
        aberration_cfg=aberration_cfg,
        device="auto",
        out=out_dir,
        ladder_exposures=ladder,
    )
    wall_s = _now() - started
    return {
        "arm": arm,
        "wall_s": float(wall_s),
        "out_dir": out_dir.relative_to(REPO).as_posix() if out_dir.is_relative_to(REPO) else out_dir.as_posix(),
        "eval_loop": (out_dir / "eval_loop.json").relative_to(REPO).as_posix(),
        "payload": payload,
    }


# ----------------------------------------------------------------------
# 3. Baseline reproduction
# ----------------------------------------------------------------------
def baseline_diff(
    payload: dict[str, Any], baseline_path: Path
) -> dict[str, Any]:
    """Per-arm, per-column absolute difference against the published baseline.

    Args:
        payload: The ``iid`` arm's ``eval_loop`` payload.
        baseline_path: ``runs/sim2f_fair/loop/a_no_noise/eval_loop.json``.

    Returns:
        ``{arms: {arm: {column: abs_diff}}, "max_abs_diff": float, "bit_exact": bool,
        "baseline": str, "metadata_match": {...}}``.
    """
    baseline = _load_json(baseline_path)
    columns = list(baseline["columns"])
    diffs: dict[str, dict[str, float]] = {}
    worst = 0.0
    for arm in ARMS:
        row: dict[str, float] = {}
        for column in columns:
            value = payload["aggregate"][arm][column]
            reference = baseline["aggregate"][arm][column]
            delta = abs(float(value) - float(reference))
            row[column] = delta
            worst = max(worst, delta)
        diffs[arm] = row
    meta_keys = ("n_samples", "columns", "arms")
    shared = {
        key: value
        for key, value in payload["aberration"].items()
        if key in baseline["aberration"]
    }
    return {
        "baseline": baseline_path.relative_to(REPO).as_posix()
        if baseline_path.is_relative_to(REPO)
        else baseline_path.as_posix(),
        "abs_diff": diffs,
        "max_abs_diff": float(worst),
        "bit_exact": bool(worst == 0.0),
        "exact_columns": int(sum(1 for arm in ARMS for value in diffs[arm].values() if value == 0.0)),
        "total_columns": int(len(ARMS) * len(columns)),
        "metadata_match": {
            key: payload.get(key) == baseline.get(key) for key in meta_keys
        },
        "paths_differ_by_design": {
            key: {"this_run": payload.get(key), "baseline": baseline.get(key)}
            for key in ("cache", "checkpoint")
            if payload.get(key) != baseline.get(key)
        },
        "loop_config_match": {
            key: payload["loop"].get(key) == baseline["loop"].get(key)
            for key in ("samples", "rounds", "probe_fit", "refit", "use_detector", "seed")
        },
        "aberration_shared_keys_match": bool(
            shared == baseline["aberration"]
        ),
        "aberration_extra_keys": sorted(
            set(payload["aberration"]) - set(baseline["aberration"])
        ),
        "aberration_missing_keys": sorted(
            set(baseline["aberration"]) - set(payload["aberration"])
        ),
    }


# ----------------------------------------------------------------------
# 4. Per-family table
# ----------------------------------------------------------------------
def family_table(payload: dict[str, Any]) -> dict[str, Any]:
    """Per-family objectives and the report's ``net < flat`` counts.

    Args:
        payload: A loop payload.

    Returns:
        ``{"families": {kind: {...}}, "counts": {arm: {"wins", "of", "families"}}}``.
    """
    families: dict[str, Any] = {}
    for kind, entry in payload["per_kind"].items():
        objectives = {arm: float(entry["arms"][arm]["objective"]) for arm in ARMS}
        families[kind] = {
            "n": int(entry["n"]),
            "objective": objectives,
            "closed_coeff_rmse": float(entry["closed_coeff_rmse"]),
            "beats_flat": {arm: objectives[arm] < objectives["flat"] for arm in CHALLENGERS},
        }
    counts: dict[str, Any] = {}
    for arm in CHALLENGERS:
        winners = [kind for kind, entry in families.items() if entry["beats_flat"][arm]]
        counts[arm] = {"wins": len(winners), "of": len(families), "families": winners}
    return {"families": families, "counts": counts}


def ordering(payload: dict[str, Any]) -> dict[str, Any]:
    """Aggregate ranking plus the flat-vs-net statements the verdict needs.

    Args:
        payload: A loop payload.

    Returns:
        Rank by objective (lower is better), whether ``flat`` wins, and every
        challenger's gap to ``flat``.
    """
    aggregate = payload["aggregate"]
    ranked = sorted(ARMS, key=lambda arm: float(aggregate[arm]["objective"]))
    flat = float(aggregate["flat"]["objective"])
    return {
        "rank_by_objective": ranked,
        "best": ranked[0],
        "flat_is_best": ranked[0] == "flat",
        "flat_objective": flat,
        "net_beats_flat_aggregate": {
            arm: float(aggregate[arm]["objective"]) < flat for arm in CHALLENGERS
        },
        "gap_to_flat": {
            arm: float(aggregate[arm]["objective"]) - flat for arm in CHALLENGERS
        },
    }


# ----------------------------------------------------------------------
# 5. What the ensemble actually changed
# ----------------------------------------------------------------------
def _high_order_matrix(payload: dict[str, Any]) -> np.ndarray:
    """Per-sample drawn high-order coefficients from a loop payload.

    Args:
        payload: A loop payload (``per_sample[].truth.coeffs``).

    Returns:
        ``(n_samples, n_high)`` float64, columns = Noll ``4 .. n_modes + 1``.
    """
    rows = [
        np.asarray(sample["truth"]["coeffs"], dtype=np.float64)[2:]
        for sample in payload["per_sample"]
    ]
    return np.vstack(rows)


def _state_matrix_pairs(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    """Tilt and illumination fields of each drawn state.

    Args:
        payload: A loop payload.

    Returns:
        ``{"tilt_px": (N, 2), "illum_center_px": (N, 2), "illum_width_scale": (N,),
        "rms": (N,)}`` -- the parts ``aberration_model`` must not touch.
    """
    tilt, center, width, rms = [], [], [], []
    for sample in payload["per_sample"]:
        truth = sample["truth"]
        tilt.append(truth["tilt_px"])
        center.append(truth["illum_center_px"])
        width.append(truth["illum_width_scale"])
        coeffs = np.asarray(truth["coeffs"], dtype=np.float64)
        rms.append(float(np.sqrt(np.mean(coeffs[2:] ** 2))))
    return {
        "tilt_px": np.asarray(tilt, dtype=np.float64),
        "illum_center_px": np.asarray(center, dtype=np.float64),
        "illum_width_scale": np.asarray(width, dtype=np.float64),
        "rms": np.asarray(rms, dtype=np.float64),
    }


def _noll(i: int) -> int:
    """0-based high-order column -> Noll index (``coeffs[2]`` is Noll 4)."""
    return i + 4


def mode_rms(matrix: np.ndarray) -> list[float]:
    """Per-column RMS of a coefficient matrix."""
    return [float(np.sqrt(np.mean(matrix[:, i] ** 2))) for i in range(matrix.shape[1])]


def draw_many(cfg: Any, *, seed: int, count: int) -> np.ndarray:
    """Draw ``count`` states with the loop's own seeding scheme and return coeffs.

    Uses ``np.random.default_rng([seed, index])`` exactly as
    ``loop.run_loop`` does, so the realised ladder is the loop's ladder rather
    than a re-derivation.

    Args:
        cfg: ``AberrationConfig``.
        seed: ``LoopConfig.seed``.
        count: Number of indices to draw over (used for the stable ensemble
            estimate; the loop itself only visits the sampled corpus indices).

    Returns:
        ``(count, n_modes)`` float64 coefficient matrix, radians.
    """
    from ml.sim2f.config import Sim2fConfig
    from ml.sim2f.optics import Sim2f, draw_aberration

    bench = Sim2f(Sim2fConfig.calibrated(), device="cpu")
    rows = []
    for index in range(count):
        rng = np.random.default_rng([int(seed), int(index)])
        rows.append(np.asarray(draw_aberration(bench, cfg, rng).coeffs, dtype=np.float64))
    return np.vstack(rows)


def correlation_structure(matrix: np.ndarray) -> dict[str, Any]:
    """Strongest off-diagonal correlation of the high-order coefficients.

    Args:
        matrix: ``(N, n_high)`` coefficients, columns = Noll ``4 .. n_modes + 1``.

    Returns:
        ``{"n", "strongest": {...}, "top_pairs": [...], "median_abs": float,
        "matrix": [[...]]}``.
    """
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    corr = np.corrcoef(finite, rowvar=False)
    n_high = corr.shape[0]
    pairs = []
    for i in range(n_high):
        for j in range(i + 1, n_high):
            pairs.append(
                {
                    "noll_a": _noll(i),
                    "noll_b": _noll(j),
                    "r": float(corr[i, j]),
                }
            )
    pairs.sort(key=lambda item: abs(item["r"]), reverse=True)
    off = np.abs(np.asarray([item["r"] for item in pairs], dtype=np.float64))
    return {
        "n": int(finite.shape[0]),
        "strongest": pairs[0],
        "top_pairs": pairs[:5],
        "median_abs_offdiag": float(np.median(off)) if off.size else 0.0,
        "mean_abs_offdiag": float(np.mean(off)) if off.size else 0.0,
        "matrix": [[float(value) for value in row] for row in corr],
        "noll_labels": [_noll(i) for i in range(n_high)],
    }


def asset_shrinkage(matrix: np.ndarray, covariance_path: Path) -> dict[str, Any]:
    """Reproduce the documented Bingham shrinkage: realised share over asset share.

    ``draw_aberration`` rescales every draw to a fixed total RMS, which puts the
    ensemble on a sphere; an anisotropic shape on a sphere is squeezed toward
    isotropy.  The documented measurement (``test_end_to_end_shape_survives_the
    _rms_rescale``: 2000 draws) is low order **0.85x** and the highest mode
    **1.27x** of its *asset* share, and a defocus-to-Noll-22 span of **35x -> 24x**.
    This recomputes it from the driver's own independent draw, so the claim is
    either confirmed here or not.

    Args:
        matrix: ``(N, n_high)`` high-order coefficients from
            :func:`draw_many` (already Noll ``4 ..``).
        covariance_path: The von Karman asset the ensemble came from.

    Returns:
        The per-mode realised/asset share ratio, its extremes, and the
        defocus-to-highest span before and after the rescale.
    """
    with np.load(covariance_path) as data:
        target = np.asarray(data["covariance"], dtype=np.float64)
        noll = np.asarray(data["noll"], dtype=np.int64)

    empirical = np.cov(matrix, rowvar=False)
    realised_var = np.diag(empirical)
    asset_var = np.diag(target)
    if realised_var.size != asset_var.size:
        return {"comparable": False, "reason": "draw and asset cover different mode counts"}

    share = realised_var / realised_var.mean()
    expected = asset_var / asset_var.mean()
    ratio = share / expected

    lowest = int(np.argmin(ratio))
    highest = int(np.argmax(ratio))
    defocus = int(np.where(noll == 4)[0][0])
    top = int(np.where(noll == int(noll[-1]))[0][0])
    return {
        "comparable": True,
        "n": int(matrix.shape[0]),
        "noll": [int(value) for value in noll],
        "share_ratio": [float(value) for value in ratio],
        "min_ratio": {"noll": int(noll[lowest]), "value": float(ratio[lowest])},
        "max_ratio": {"noll": int(noll[highest]), "value": float(ratio[highest])},
        "low_order_mean_ratio_noll_4_6": float(np.mean(ratio[noll <= 6])),
        "highest_mode_ratio": {"noll": int(noll[top]), "value": float(ratio[top])},
        "asset_span_defocus_to_highest": float(expected[defocus] / expected[top]),
        "realised_span_defocus_to_highest": float(share[defocus] / share[top]),
        "documented": {
            "low_order": 0.85,
            "highest_mode": 1.27,
            "asset_span": 35.0,
            "realised_span": 24.0,
            "source": "tests/ao_shaping/ml/sim2f/test_aberration_covariance.py"
            "::test_end_to_end_shape_survives_the_rms_rescale",
        },
    }


def ensemble_difference(
    iid_payload: dict[str, Any],
    ko_payload: dict[str, Any],
    *,
    seed: int,
    large_n: int,
) -> dict[str, Any]:
    """Compare the two realised ensembles: ladder, shrinkage, correlation, pairing.

    Args:
        iid_payload: The ``iid`` arm payload.
        ko_payload: The ``kolmogorov`` arm payload.
        seed: ``LoopConfig.seed`` used for the large draw.
        large_n: Number of draws for the stable estimate.

    Returns:
        A JSON-able record with the *actual* 24 states and the ``large_n`` estimate.
    """
    configs = build_configs()
    n_modes = int(configs["iid"].n_modes)
    from ml.sim2f.optics import DEFAULT_COVARIANCE_PATH

    # Both are already high-order only (columns = Noll 4..n_modes+1).
    actual_iid = _high_order_matrix(iid_payload)
    actual_ko = _high_order_matrix(ko_payload)
    large_iid = draw_many(configs["iid"], seed=seed, count=large_n)[:, 2:]
    large_ko = draw_many(configs["kolmogorov"], seed=seed, count=large_n)[:, 2:]

    def ladder(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
        iid_rms = mode_rms(left)
        ko_rms = mode_rms(right)
        return {
            "noll": [_noll(i) for i in range(len(iid_rms))],
            "iid_rms": iid_rms,
            "kolmogorov_rms": ko_rms,
            "ratio_ko_over_iid": [k / i if i > 0 else None for i, k in zip(iid_rms, ko_rms)],
        }

    paired_fields = {}
    state_iid = _state_matrix_pairs(iid_payload)
    state_ko = _state_matrix_pairs(ko_payload)
    for name in ("tilt_px", "illum_center_px", "illum_width_scale", "rms"):
        left, right = state_iid[name], state_ko[name]
        paired_fields[name] = {
            "identical": bool(np.array_equal(left, right)),
            "max_abs_diff": float(np.max(np.abs(left - right))) if left.shape == right.shape else None,
        }
    paired_high_order = {
        "identical": bool(np.array_equal(actual_iid, actual_ko)),
        "max_abs_diff": float(np.max(np.abs(actual_iid - actual_ko))),
        "max_abs_diff_including_tilt": float(
            np.max(
                np.abs(
                    np.vstack([np.asarray(s["truth"]["coeffs"]) for s in iid_payload["per_sample"]])
                    - np.vstack(
                        [np.asarray(s["truth"]["coeffs"]) for s in ko_payload["per_sample"]]
                    )
                )
            )
        ),
    }
    sample_order = {
        "identical": [s["index"] for s in iid_payload["per_sample"]]
        == [s["index"] for s in ko_payload["per_sample"]],
        "indices": [int(s["index"]) for s in iid_payload["per_sample"]],
        "kinds": [str(s["kind"]) for s in iid_payload["per_sample"]],
    }
    total_rms_paired = {
        "identical_per_sample": bool(np.array_equal(state_iid["rms"], state_ko["rms"])),
        "iid_mean": float(np.mean(state_iid["rms"])),
        "kolmogorov_mean": float(np.mean(state_ko["rms"])),
        "max_abs_diff": float(np.max(np.abs(state_iid["rms"] - state_ko["rms"]))),
    }
    return {
        "n_modes": n_modes,
        "actual_24_states": {
            "ladder": ladder(actual_iid, actual_ko),
            "correlation_iid": correlation_structure(actual_iid),
            "correlation_kolmogorov": correlation_structure(actual_ko),
        },
        "large_draw": {
            "n": int(large_n),
            "seed": int(seed),
            "seeding": "np.random.default_rng([seed, index]) for index in range(n) -- the loop's own scheme",
            "ladder": ladder(large_iid, large_ko),
            "correlation_iid": correlation_structure(large_iid),
            "correlation_kolmogorov": correlation_structure(large_ko),
        },
        "bingham_shrinkage": asset_shrinkage(large_ko, Path(DEFAULT_COVARIANCE_PATH)),
        "covariance_asset": str(DEFAULT_COVARIANCE_PATH),
        "pairing": {
            "sample_order": sample_order,
            "fields_the_ensemble_must_not_touch": paired_fields,
            "high_order_only_differs": paired_high_order,
            "total_rms": total_rms_paired,
        },
    }


# ----------------------------------------------------------------------
# 6. Noise floor (state-draw replicates)
# ----------------------------------------------------------------------


def _sign_flip_pvalue(diffs: list[float], *, seed: int = 0, iterations: int = 200_000) -> dict[str, Any]:
    """Exact sign-flip p when cheap, Monte-Carlo above the exact limit.

    ``ml.zernike.eval_stats.sign_flip_pvalue`` enumerates all ``2**n`` assignments
    in pure Python.  At ``n = 18`` (its documented regime) that is 4.7 M inner
    steps; at ``n = 24`` it is 400 M and would take hours, so above the limit this
    uses the *same statistic* by Monte-Carlo.  The p is reported with the method
    it came from, so nobody quotes a Monte-Carlo p as exact.

    Args:
        diffs: Paired differences.
        seed: Monte-Carlo seed (documented, so the number is reproducible).
        iterations: Monte-Carlo iterations above the exact limit.

    Returns:
        ``{"p", "method", "min_attainable_p", "n"}``.
    """
    from ml.zernike.eval_stats import min_attainable_pvalue, sign_flip_pvalue

    clean = [float(value) for value in diffs if not math.isnan(float(value))]
    n_pairs = len(clean)
    if n_pairs == 0:
        return {"p": None, "method": "n/a", "min_attainable_p": float("nan"), "n": 0}
    if n_pairs <= 18:
        return {
            "p": float(sign_flip_pvalue(clean)),
            "method": "exact (ml.zernike.eval_stats.sign_flip_pvalue)",
            "min_attainable_p": float(min_attainable_pvalue(n_pairs)),
            "n": n_pairs,
        }
    values = np.asarray(clean, dtype=np.float64)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(int(seed))
    extreme = 0
    done = 0
    chunk = 20_000
    while done < iterations:
        size = min(chunk, iterations - done)
        signs = rng.integers(0, 2, size=(size, n_pairs), dtype=np.int8) * 2 - 1
        means = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(means >= observed - 1e-15))
        done += size
    return {
        "p": float((extreme + 1) / (iterations + 1)),
        "method": f"monte_carlo ({iterations} sign flips, seed={seed})",
        "min_attainable_p": float(min_attainable_pvalue(n_pairs)),
        "n": n_pairs,
    }


def _sign_test_pvalue(diffs: list[float], *, tol: float = 1e-6) -> float:
    """Exact two-sided binomial sign test (cheap, exact at any n).

    Differences within ``tol`` are dropped as ties rather than counted as
    negatives, so a round-off-only arm (``oracle``) is reported as having nothing
    to test instead of a spurious p-value.

    Args:
        diffs: Paired differences.
        tol: Absolute magnitude below which a difference counts as a tie.

    Returns:
        ``min(1, 2 * sum_{i=k..n} C(n, i) / 2**n)`` with ``k`` the larger tail.
    """
    clean = [
        float(value)
        for value in diffs
        if not math.isnan(float(value)) and abs(float(value)) > tol
    ]
    n_pairs = len(clean)
    if n_pairs == 0:
        return float("nan")
    positive = sum(1 for value in clean if value > 0.0)
    negative = n_pairs - positive
    tail = max(positive, negative)
    return float(min(1.0, 2.0 * sum(comb(n_pairs, i) for i in range(tail, n_pairs + 1)) / 2**n_pairs))


def paired_analysis(iid_payload: dict[str, Any], ko_payload: dict[str, Any]) -> dict[str, Any]:
    """The powerful comparison: per-sample paired differences, ``ko - iid``.

    The two arms visit the **same** corpus indices, share tip/tilt and the
    illumination state, and are rescaled to the *same* total high-order RMS, so a
    per-sample difference isolates the ensemble's shape.  That is far more
    sensitive than comparing two aggregates drawn from different state sets, and
    it is the right instrument for "did the ensemble move the numbers".

    Args:
        iid_payload: iid arm payload.
        ko_payload: kolmogorov arm payload.

    Returns:
        Per-arm paired summaries plus the per-sample flat-vs-net gap difference.
    """
    from ml.zernike.eval_stats import cohens_dz

    left = iid_payload["per_sample"]
    right = ko_payload["per_sample"]
    if [s["index"] for s in left] != [s["index"] for s in right]:
        return {"comparable": False, "reason": "sample order differs between arms"}

    columns = ("objective", "ncc", "l1", "ee")
    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        entry: dict[str, Any] = {"per_sample": {}, "columns": {}}
        objectives = []
        for position, (iid_sample, ko_sample) in enumerate(zip(left, right)):
            values = {
                column: float(ko_sample["arms"][arm][column])
                - float(iid_sample["arms"][arm][column])
                for column in columns
            }
            entry["per_sample"][str(iid_sample["index"])] = values
            objectives.append(values["objective"])
            del position
        for column in columns:
            diffs = [entry["per_sample"][key][column] for key in entry["per_sample"]]
            stats = _sign_flip_pvalue(diffs, seed=0)
            spread = float(np.std(diffs, ddof=1))
            degenerate = spread < 1e-6
            entry["columns"][column] = {
                "mean_diff": float(np.mean(diffs)),
                "std_diff": spread,
                "cohens_dz": None if degenerate else float(cohens_dz(diffs)),
                "p_sign_flip": None if degenerate else stats["p"],
                "p_method": stats["method"],
                "min_attainable_p": stats["min_attainable_p"],
                "p_sign_test": _sign_test_pvalue(diffs),
                "n_lower": int(sum(1 for value in diffs if value < -1e-6)),
                "n_higher": int(sum(1 for value in diffs if value > 1e-6)),
                "n_tied": int(sum(1 for value in diffs if abs(value) <= 1e-6)),
                "n_pairs": len(diffs),
                "degenerate": bool(degenerate),
            }
        entry["objective_diffs"] = [float(value) for value in objectives]
        per_arm[arm] = entry

    gaps = []
    for iid_sample, ko_sample in zip(left, right):
        gap_iid = float(iid_sample["arms"]["net_open"]["objective"]) - float(
            iid_sample["arms"]["flat"]["objective"]
        )
        gap_ko = float(ko_sample["arms"]["net_open"]["objective"]) - float(
            ko_sample["arms"]["flat"]["objective"]
        )
        gaps.append(gap_ko - gap_iid)
    gap_stats = _sign_flip_pvalue(gaps, seed=0)

    oracle_diffs = per_arm["oracle"]["objective_diffs"]
    others = [
        abs(value)
        for arm in ("flat", "net_open", "net_probe", "net_closed")
        for value in per_arm[arm]["objective_diffs"]
    ]
    return {
        "comparable": True,
        "n_pairs": len(left),
        "sign_convention": "difference = kolmogorov - iid; negative = kolmogorov scored lower (better objective)",
        "arms": per_arm,
        "net_open_minus_flat_gap": {
            "mean_diff": float(np.mean(gaps)),
            "std_diff": float(np.std(gaps, ddof=1)),
            "cohens_dz": float(cohens_dz(gaps)),
            "p_sign_flip": gap_stats["p"],
            "p_method": gap_stats["method"],
            "min_attainable_p": gap_stats["min_attainable_p"],
            "p_sign_test": _sign_test_pvalue(gaps),
            "n_gap_widened": int(sum(1 for value in gaps if value > 0.0)),
            "n_pairs": len(gaps),
            "per_sample": [float(value) for value in gaps],
        },
        "oracle_pairing_check": {
            "max_abs_objective_diff": float(max(abs(value) for value in oracle_diffs)),
            "largest_estimated_arm_diff": float(max(others)) if others else None,
            "interpretation": (
                "`oracle` cancels the true coefficients exactly, so its frame is the "
                "same in both arms up to float32 round-off. It therefore measures how "
                "well the two runs are paired: anything above round-off would mean the "
                "runs differed in more than the ensemble."
            ),
        },
    }


def noise_floor(
    *,
    base: dict[str, dict[str, Any]],
    replicates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """How far the aggregate moves when only the *drawn 24 states* change.

    With the detector off there is no measurement noise at all, so the honest
    floor for "is this difference real" is a second draw of 24 states from the
    same ensemble.  Reported as the per-arm absolute change plus the change in
    the flat-vs-net gap, which is the quantity the verdict is about.

    Args:
        base: ``{ensemble: run record}`` at ``LoopConfig.seed``.
        replicates: ``{ensemble: run record}`` at the replicate seed.

    Returns:
        A JSON-able floor record per ensemble.
    """
    floor: dict[str, Any] = {}
    for ensemble, run in base.items():
        other = replicates.get(ensemble)
        if other is None:
            continue
        left, right = run["payload"]["aggregate"], other["payload"]["aggregate"]
        per_arm = {
            arm: {
                column: abs(float(left[arm][column]) - float(right[arm][column]))
                for column in ("objective", "ncc", "l1", "ee")
            }
            for arm in ARMS
        }
        gaps = {
            arm: {
                "base": float(left[arm]["objective"]) - float(left["flat"]["objective"]),
                "replicate": float(right[arm]["objective"]) - float(right["flat"]["objective"]),
            }
            for arm in CHALLENGERS
        }
        for arm, entry in gaps.items():
            entry["abs_change"] = abs(entry["base"] - entry["replicate"])
        floor[ensemble] = {
            "base_seed": int(run["payload"]["loop"]["seed"]),
            "replicate_seed": int(other["payload"]["loop"]["seed"]),
            "per_arm_abs_change": per_arm,
            "flat_vs_net_gap": gaps,
            "max_abs_change_objective": max(
                per_arm[arm]["objective"] for arm in ARMS
            ),
            "max_abs_change_gap": max(entry["abs_change"] for entry in gaps.values()),
        }
    return floor


# ----------------------------------------------------------------------
# 7. Rendering
# ----------------------------------------------------------------------
def _arm_table(payload: dict[str, Any]) -> list[str]:
    """Aggregate metric table for one payload."""
    lines = [
        "| arm | objective | ncc | l1 | ee | cv_in_support | iou50 |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        row = payload["aggregate"][arm]
        lines.append(
            f"| `{arm}` | {row['objective']:.6f} | {row['ncc']:.6f} | {row['l1']:.6f} | "
            f"{row['ee']:.6f} | {row['cv_in_support']:.4f} | {row['iou50']:.4f} |"
        )
    return lines


def _family_lines(table: dict[str, Any]) -> list[str]:
    """Per-family objective table plus the win counts."""
    lines = [
        "| family | n | flat | net_open | net_probe | net_closed | oracle | net_open < flat |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for kind, entry in sorted(
        table["families"].items(), key=lambda item: item[1]["objective"]["flat"]
    ):
        obj = entry["objective"]
        mark = "yes" if entry["beats_flat"]["net_open"] else "no"
        lines.append(
            f"| {kind} | {entry['n']} | {obj['flat']:.4f} | {obj['net_open']:.4f} | "
            f"{obj['net_probe']:.4f} | {obj['net_closed']:.4f} | {obj['oracle']:.4f} | {mark} |"
        )
    wins = table["counts"]
    lines.append("")
    for arm in CHALLENGERS:
        entry = wins[arm]
        detail = ", ".join(entry["families"]) if entry["families"] else "none"
        lines.append(f"- `{arm}` beats `flat` in **{entry['wins']}/{entry['of']}** families: {detail}")
    return lines


def render_markdown(
    *,
    payload_json: dict[str, Any],
) -> str:
    """Render ``two_arms.md`` from the comparison payload.

    Args:
        payload_json: The dict also written to ``two_arms.json``.

    Returns:
        The markdown document.
    """
    setup = payload_json["setup"]
    verification = payload_json["conceptual_verification"]
    baseline = payload_json["baseline"]
    arms = payload_json["arms"]
    tables = payload_json["per_family"]
    ensemble = payload_json["ensemble"]
    floor = payload_json["noise_floor"]
    verdict = payload_json["verdict"]

    lines: list[str] = []
    lines.append("# Two-arm comparison — `iid` vs `kolmogorov` aberration ensemble")
    lines.append("")
    lines.append(
        "Same corpus, same checkpoint, same 24 samples, same seed, same rounds, "
        "detector off — **only** the high-order coefficient ensemble differs."
    )
    lines.append("")
    lines.append(f"- corpus: `{setup['cache']}`")
    lines.append(f"- checkpoint: `{setup['checkpoint']}`")
    lines.append(
        f"- loop: samples={setup['samples']}, rounds={setup['rounds']}, seed={setup['seed']}, "
        f"use_detector={setup['use_detector']}, probe_fit=True, refit=True"
    )
    lines.append(f"- baseline reproduced: `{baseline['baseline']}`")
    lines.append(f"- mode: {setup['mode']}")
    lines.append(f"- git HEAD: `{setup['git_head']}`")
    lines.append("")

    lines.append("## 0. Pre-registration (text written before the numbers)")
    lines.append("")
    lines.append("> " + PRE_REGISTRATION["expectation"])
    lines.append("")
    lines.append(f"Reasoning: {PRE_REGISTRATION['reasoning']}")
    lines.append("")
    lines.append(f"Significant outcome: {PRE_REGISTRATION['what_would_be_significant']}")
    lines.append("")
    lines.append(f"Not evidence: {PRE_REGISTRATION['what_is_not_evidence']}")
    lines.append("")

    lines.append("## 1. Conceptual verification — does the corpus carry an aberration state?")
    lines.append("")
    lines.append(
        f"**No.** `pairs.npz` holds exactly the {len(verification['archive_fields'])} "
        f"fields declared by `ml.sim2f.schema.CACHE_FIELDS`: "
        f"`{', '.join(verification['archive_fields'])}`."
    )
    lines.append("")
    lines.append(
        f"- fields naming an aberration/zernike/state/coefficient: "
        f"{verification['fields_naming_an_aberration_state'] or 'none'}"
    )
    lines.append(
        f"- `build_pairs` passes an `aberration=` kwarg: "
        f"**{verification['synth_passes_an_aberration_kwarg']}** "
        f"(mentions `draw_aberration`: {verification['synth_mentions_draw_aberration']}, "
        f"`AberrationConfig`: {verification['synth_mentions_AberrationConfig']}); "
        f"its docstring says the teacher bench is ABERRATION-FREE: "
        f"{verification['synth_documents_an_aberration_free_bench']}"
    )
    lines.append(
        f"- the teacher documents an aberration-free bench: "
        f"**{verification['teacher_documents_an_aberration_free_bench']}**; "
        f"passes an `aberration=` kwarg: **{verification['teacher_passes_an_aberration_kwarg']}**"
    )
    lines.append(f"- corpus `meta.json` has an aberration block: {verification['corpus_meta_aberration_block']}")
    lines.append("")
    lines.append(
        "The state is drawn **inside** the loop, per sample: `loop.run_loop` L442 "
        "`states = AberrationConfig() if aberration_cfg is None else aberration_cfg`, "
        "L478 `rng = np.random.default_rng([int(cfg.seed), int(index)])`, "
        "L479 `state = draw_aberration(bench, states, rng)`. So `aberration_model` "
        "cannot reach a label or a weight: **the two arms share corpus and checkpoint**, "
        "and the comparison is an evaluation-ensemble one."
    )
    lines.append("")

    lines.append("## 2. Commands and wall clock")
    lines.append("")
    lines.append("One command; it issues four `loop.run_loop` calls internally.")
    lines.append("")
    lines.append("| # | what | command | wall clock |")
    lines.append("|---|---|---|---|")
    total = payload_json["commands"][0]["command"] if payload_json["commands"] else ""
    lines.append(
        f"| 1 | whole driver (4 loop runs) | `{total}` | "
        f"**{payload_json['driver_wall_s']:.1f} s** |"
    )
    for entry in payload_json["commands"]:
        lines.append(
            f"| — | internal `run_loop` → `{entry['arm']}` | "
            f"`runs/sim2f_kl_ab/loop/{entry['arm']}/` | {entry['wall_s']:.1f} s |"
        )
    lines.append("")
    lines.append(
        "Per-run clock is measured around `run_loop` alone, so it excludes interpreter "
        "start, `ml.sim2f` import and CUDA init — which is why the first run is the "
        "slowest and why these do not add up to the driver total. The published "
        "`runs/sim2f_fair` figure for one loop run is **83.6 s** as a *whole process*; "
        "see `COMMANDS.md` for the like-for-like process measurement."
    )
    lines.append("")

    lines.append("## 3. Baseline reproduction (iid arm vs the published `a_no_noise`)")
    lines.append("")
    lines.append(
        f"max |Δ| over {baseline['total_columns']} columns = "
        f"**{baseline['max_abs_diff']:.3e}** "
        f"({baseline['exact_columns']}/{baseline['total_columns']} columns exactly equal); "
        f"bit-exact: **{baseline['bit_exact']}**."
    )
    lines.append("")
    lines.append("| arm | column | published | iid arm | abs diff |")
    lines.append("|---|---|---|---|---|")
    for arm in ARMS:
        for column, delta in baseline["abs_diff"][arm].items():
            published = baseline["values"]["published"][arm][column]
            got = baseline["values"]["iid"][arm][column]
            lines.append(
                f"| `{arm}` | {column} | {published:.9f} | {got:.9f} | {delta:.3e} |"
            )
    lines.append("")
    lines.append(
        f"config match: loop={baseline['loop_config_match']}, "
        f"aberration block shared keys={baseline['aberration_shared_keys_match']} "
        f"(extra keys in this run: {baseline['aberration_extra_keys']}), "
        f"metadata={baseline['metadata_match']}"
    )
    if baseline["paths_differ_by_design"]:
        lines.append(
            f"(paths differ by design: {list(baseline['paths_differ_by_design'])})"
        )
    lines.append("")

    lines.append("## 4. Aggregate, both arms")
    lines.append("")
    for ensemble_name in ("iid", "kolmogorov"):
        lines.append(f"### {ensemble_name}")
        lines.append("")
        lines.extend(_arm_table(arms[ensemble_name]["payload"]))
        lines.append("")
        rank = arms[ensemble_name]["ordering"]
        lines.append(
            f"rank by objective (lower is better): "
            f"`{' < '.join(rank['rank_by_objective'])}` — `flat` best: **{rank['flat_is_best']}**"
        )
        lines.append("")

    lines.append("## 5. Per family — does `net` beat `flat`?")
    lines.append("")
    for ensemble_name in ("iid", "kolmogorov"):
        lines.append(f"### {ensemble_name}")
        lines.append("")
        lines.extend(_family_lines(tables[ensemble_name]))
        lines.append("")

    lines.append("## 6. What the kolmogorov ensemble actually changed")
    lines.append("")
    lines.append(
        "### Pairing (the comparison is paired, not independent)"
    )
    lines.append("")
    pairing = ensemble["pairing"]
    lines.append(
        f"- sample order identical across arms: **{pairing['sample_order']['identical']}** "
        f"({len(pairing['sample_order']['indices'])} samples, "
        f"{len(set(pairing['sample_order']['kinds']))} families: "
        f"{', '.join(dict.fromkeys(pairing['sample_order']['kinds']))})"
    )
    for name, entry in pairing["fields_the_ensemble_must_not_touch"].items():
        if name == "rms":
            continue  # covered by the total-RMS bullet below
        exact = entry["identical"] or (
            entry["max_abs_diff"] is not None and entry["max_abs_diff"] < 1e-12
        )
        lines.append(
            f"- `{name}` identical: **{exact}** (max |Δ| {_fmt(entry['max_abs_diff'], '.3e')})"
        )
    lines.append(
        f"- total high-order RMS identical per sample (to float64 round-off): "
        f"**{pairing['total_rms']['max_abs_diff'] < 1e-12}** "
        f"(mean iid {pairing['total_rms']['iid_mean']:.4f} rad vs kolmogorov "
        f"{pairing['total_rms']['kolmogorov_mean']:.4f} rad, max |Δ| "
        f"{pairing['total_rms']['max_abs_diff']:.3e})"
    )
    lines.append(
        f"- only `coeffs[2:]` differs: max |Δ| over all 21 coefficients = "
        f"**{pairing['high_order_only_differs']['max_abs_diff_including_tilt']:.3e}**, "
        f"over the 19 high-order ones = {pairing['high_order_only_differs']['max_abs_diff']:.3e}"
    )
    if payload_json.get("paired", {}).get("comparable"):
        oracle = payload_json["paired"]["oracle_pairing_check"]
        lines.append(
            f"- **measured pairing proof**: the `oracle` arm cancels the true "
            f"coefficients exactly, so its objective can only differ by round-off. It "
            f"differs by **{oracle['max_abs_objective_diff']:.2e}** between the two "
            f"arms, against {oracle['largest_estimated_arm_diff']:.2e} for the "
            f"estimated arms — i.e. the runs are paired to float32 precision and only "
            f"the ensemble separated them."
        )
    lines.append("")

    for scope, label in (("actual_24_states", "the 24 states the loop actually drew"),
                         ("large_draw", "a large draw (%d) for the stable estimate" % ensemble["large_draw"]["n"])):
        block = ensemble[scope]
        lines.append(f"### Per-mode RMS ladder — {label}")
        lines.append("")
        lines.append(
            "| Noll | iid RMS (rad) | kolmogorov RMS (rad) | ratio ko/iid |"
        )
        lines.append("|---|---|---|---|")
        ladder = block["ladder"]
        for index, noll in enumerate(ladder["noll"]):
            ratio = ladder["ratio_ko_over_iid"][index]
            lines.append(
                f"| {noll} | {ladder['iid_rms'][index]:.5f} | "
                f"{ladder['kolmogorov_rms'][index]:.5f} | {_fmt(ratio, '.3f')} |"
            )
        lines.append("")
        for arm_name in ("correlation_iid", "correlation_kolmogorov"):
            corr = block[arm_name]
            strongest = corr["strongest"]
            lines.append(
                f"- `{arm_name.replace('correlation_', '')}`: strongest off-diagonal "
                f"**r = {strongest['r']:+.3f}** (Noll {strongest['noll_a']}–{strongest['noll_b']}), "
                f"median |r| {corr['median_abs_offdiag']:.3f}, mean |r| {corr['mean_abs_offdiag']:.3f}, "
                f"n = {corr['n']}"
            )
            top = ", ".join(
                f"{item['noll_a']}–{item['noll_b']} {item['r']:+.2f}"
                for item in corr["top_pairs"][:3]
            )
            lines.append(f"  - top 3 pairs: {top}")
        if scope == "actual_24_states":
            lines.append("")
            lines.append(
                "  ⚠️ **At n = 24 this correlation block is noise, not structure.** "
                "With 171 pairs the sampling sd of a single r is ≈ 0.21, so a max |r| "
                "near 0.7 (and a median |r| near 0.19) is what an *uncorrelated* "
                "ensemble produces — which is exactly why the iid reference row looks "
                "as structured as the kolmogorov one here. Read the 4096-draw block "
                "below for the real correlation structure."
            )
        lines.append("")

    shrink = ensemble.get("bingham_shrinkage", {})
    lines.append("### Bingham shrinkage — does the documented cost show up?")
    lines.append("")
    if shrink.get("comparable"):
        documented = shrink["documented"]
        lines.append(
            "Fixed total RMS puts every draw on a sphere, and an anisotropic shape on "
            "a sphere is squeezed toward isotropy. The documented cost "
            "(`" + documented["source"].split("::")[-1] + "`) is low order "
            f"**{documented['low_order']:.2f}×** and the highest mode "
            f"**{documented['highest_mode']:.2f}×** of its *asset* share, with the "
            f"defocus-to-Noll-{shrink['noll'][-1]} span falling "
            f"{documented['asset_span']:.0f}× → {documented['realised_span']:.0f}×."
        )
        lines.append("")
        lines.append("| quantity | documented | measured here |")
        lines.append("|---|---|---|")
        lines.append(
            f"| low-order mean share ratio (Noll 4–6) | {documented['low_order']:.2f} | "
            f"**{shrink['low_order_mean_ratio_noll_4_6']:.3f}** |"
        )
        lines.append(
            f"| highest mode (Noll {shrink['highest_mode_ratio']['noll']}) share ratio | "
            f"{documented['highest_mode']:.2f} | **{shrink['highest_mode_ratio']['value']:.3f}** |"
        )
        lines.append(
            f"| defocus-to-Noll-{shrink['noll'][-1]} span, asset | {documented['asset_span']:.0f}× | "
            f"{shrink['asset_span_defocus_to_highest']:.1f}× |"
        )
        lines.append(
            f"| defocus-to-Noll-{shrink['noll'][-1]} span, realised | "
            f"{documented['realised_span']:.0f}× | **{shrink['realised_span_defocus_to_highest']:.1f}×** |"
        )
        lines.append(
            f"| extreme share ratio | — | min {shrink['min_ratio']['value']:.3f} "
            f"(Noll {shrink['min_ratio']['noll']}), max {shrink['max_ratio']['value']:.3f} "
            f"(Noll {shrink['max_ratio']['noll']}) |"
        )
        lines.append("")
        lines.append(
            f"Measured on an independent draw of n = {shrink['n']} from the driver's own "
            "`np.random.default_rng([seed, index])` scheme, so this is not the test's "
            "numbers being echoed back."
        )
    else:
        lines.append(f"Not comparable: {shrink.get('reason')}")
    lines.append("")

    floor_lines = payload_json["noise_floor_lines"]
    lines.append("## 7. Noise floor — a different 24 states from the same ensemble")
    lines.append("")
    lines.extend(floor_lines)
    lines.append("")

    lines.append("## 8. Verdict")
    lines.append("")
    lines.extend(verdict["lines"])
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 9. Verdict
# ----------------------------------------------------------------------
def build_verdict(
    *,
    iid_payload: dict[str, Any],
    ko_payload: dict[str, Any],
    iid_rank: dict[str, Any],
    ko_rank: dict[str, Any],
    iid_families: dict[str, Any],
    ko_families: dict[str, Any],
    paired: dict[str, Any],
    floor: dict[str, Any],
) -> dict[str, Any]:
    """Decide the pre-registered question: did the net-vs-flat ordering change?

    The pre-registered question is specifically ``flat`` versus the net arms, so
    the verdict separates it from a reshuffle *inside* the net group:

    * ``flat_vs_net_ordering_changed`` -- does ``flat`` still beat every net arm on
      the aggregate, and does the family count stay flat?  This is the claim.
    * ``within_net_ordering_changed`` -- the net arms' internal order.  Reported,
      but explicitly as unresolvable when the arms sit closer to each other than a
      second draw of 24 states moves them, because a reordering of three
      statistically indistinguishable numbers is not a finding.

    Args:
        iid_payload: iid arm payload.
        ko_payload: kolmogorov arm payload.
        iid_rank: :func:`ordering` for iid.
        ko_rank: :func:`ordering` for kolmogorov.
        iid_families: :func:`family_table` for iid.
        ko_families: :func:`family_table` for kolmogorov.
        paired: :func:`paired_analysis` output.
        floor: :func:`noise_floor` output.

    Returns:
        ``{"flat_vs_net_ordering_changed", "within_net_ordering_changed", "lines", ...}``.
    """
    net_arms = ("net_open", "net_probe", "net_closed", "oracle")

    flat_vs_net_changed = (
        iid_rank["flat_is_best"] != ko_rank["flat_is_best"]
    ) or (
        iid_rank["net_beats_flat_aggregate"] != ko_rank["net_beats_flat_aggregate"]
    )
    iid_net_order = [arm for arm in iid_rank["rank_by_objective"] if arm in net_arms]
    ko_net_order = [arm for arm in ko_rank["rank_by_objective"] if arm in net_arms]
    within_net_changed = iid_net_order != ko_net_order

    iid_spread = max(float(iid_payload["aggregate"][a]["objective"]) for a in net_arms) - min(
        float(iid_payload["aggregate"][a]["objective"]) for a in net_arms
    )
    ko_spread = max(float(ko_payload["aggregate"][a]["objective"]) for a in net_arms) - min(
        float(ko_payload["aggregate"][a]["objective"]) for a in net_arms
    )
    draw_noise = max((entry["max_abs_change_objective"] for entry in floor.values()), default=None)
    within_net_resolvable = (
        None if draw_noise is None else max(iid_spread, ko_spread) > draw_noise
    )

    flat_wins_iid = iid_families["counts"]["net_open"]["wins"]
    flat_wins_ko = ko_families["counts"]["net_open"]["wins"]
    total_families = iid_families["counts"]["net_open"]["of"]

    lines: list[str] = []
    lines.append("### The pre-registered question: does `flat` still beat the net arms?")
    lines.append("")
    if not flat_vs_net_changed:
        lines.append(
            "**No — the net-vs-flat ordering did not change, and the pre-registered "
            "expectation held.** `flat` is the best arm on the aggregate under both "
            f"ensembles (iid {iid_rank['flat_objective']:.4f}; "
            f"kolmogorov {ko_rank['flat_objective']:.4f}), `net_open` beats it in "
            f"**{flat_wins_iid}/{total_families}** families under iid and "
            f"**{flat_wins_ko}/{total_families}** under kolmogorov, and the "
            "`net_open` − `flat` gap did not close: "
            f"{iid_rank['gap_to_flat']['net_open']:+.4f} → "
            f"{ko_rank['gap_to_flat']['net_open']:+.4f}."
        )
    else:
        lines.append(
            "**YES — this is the significant outcome the pre-registration named.** "
            "`flat`'s position relative to the net arms changed; see the per-family "
            "tables above and the paired test below before drawing a conclusion."
        )
    lines.append("")

    if within_net_changed:
        lines.append("### Inside the net group: a reshuffle that is not resolvable")
        lines.append("")
        lines.append(
            f"The net arms' internal order is `{' < '.join(iid_net_order)}` under iid and "
            f"`{' < '.join(ko_net_order)}` under kolmogorov — so it **did** reorder. But "
            f"the whole net group spans only {iid_spread:.4f} (iid) / {ko_spread:.4f} "
            f"(kolmogorov), while a second draw of 24 states moves a single arm's "
            f"objective by up to {_fmt(draw_noise, '.4f')}. "
            f"Resolvable: **{within_net_resolvable}**."
        )
        lines.append("")
        lines.append(
            "The fair run already had this shape (`net_closed` − `net_probe` = 0.0035), "
            "so the reshuffle is a property of a 24-state draw, not of the ensemble."
        )
        lines.append("")

    lines.append("### Did the ensemble move the numbers? (paired, the sensitive test)")
    lines.append("")
    if paired.get("comparable"):
        lines.append(
            "Per-sample paired differences, `kolmogorov − iid` (negative = kolmogorov "
            "scored a *lower*, better objective). Negative mean with a small p means the "
            "ensemble genuinely eases that arm; the pairing is exact."
        )
        lines.append("")
        lines.append("| arm | mean Δobjective | sd | Cohen's d_z | p (sign flip) | p (sign test) | lower/higher/tied |")
        lines.append("|---|---|---|---|---|---|---|")
        for arm in ARMS:
            stats = paired["arms"][arm]["columns"]["objective"]
            if stats["degenerate"]:
                lines.append(
                    f"| `{arm}` | {stats['mean_diff']:+.2e} | {stats['std_diff']:.2e} | "
                    f"— (pairing check) | — | — | "
                    f"{stats['n_lower']}/{stats['n_higher']}/{stats['n_tied']} |"
                )
            else:
                lines.append(
                    f"| `{arm}` | {stats['mean_diff']:+.4f} | {stats['std_diff']:.4f} | "
                    f"{stats['cohens_dz']:+.2f} | {_fmt(stats['p_sign_flip'], '.3g')} | "
                    f"{_fmt(stats['p_sign_test'], '.3g')} | "
                    f"{stats['n_lower']}/{stats['n_higher']}/{stats['n_tied']} |"
                )
        lines.append("")
        any_stats = next(
            (entry["columns"]["objective"] for entry in paired["arms"].values()), {}
        )
        lines.append(
            f"Method: exact sign-flip enumeration over 2²⁴ = 16.7 M assignments is "
            f"impractical in pure Python, so p is Monte-Carlo over 200 000 sign flips "
            f"(seed 0) using the same statistic as "
            f"`ml.zernike.eval_stats.sign_flip_pvalue`; the minimum attainable p over "
            f"{any_stats.get('n_pairs', 24)} pairs is "
            f"{_fmt(any_stats.get('min_attainable_p'), '.2e')}. The exact binomial sign "
            f"test is shown alongside; it drops |Δ| ≤ 1e-6 as ties. `oracle` is a "
            f"pairing check, not a measurement: its differences are float32 round-off."
        )
        lines.append("")
        gap = paired["net_open_minus_flat_gap"]
        lines.append(
            f"The pre-registered quantity — the per-sample `net_open − flat` gap — "
            f"moved by {gap['mean_diff']:+.4f} (sd {gap['std_diff']:.4f}, "
            f"d_z {gap['cohens_dz']:+.2f}, sign-flip p = {_fmt(gap['p_sign_flip'], '.3g')}, "
            f"sign-test p = {_fmt(gap['p_sign_test'], '.3g')}); it widened in "
            f"{gap['n_gap_widened']}/{gap['n_pairs']} samples."
        )
        lines.append("")
        oracle = paired["oracle_pairing_check"]
        lines.append(
            f"Pairing check: `oracle` differs by at most "
            f"{oracle['max_abs_objective_diff']:.2e} between the arms (float32 "
            f"round-off, since it cancels the true aberration exactly), against "
            f"{oracle['largest_estimated_arm_diff']:.2e} for the estimated arms."
        )
    else:
        lines.append(f"Paired analysis unavailable: {paired.get('reason')}")
    lines.append("")

    lines.append("### Reading")
    lines.append("")
    if not flat_vs_net_changed:
        lines.append(
            "The ensemble changes what the loop must survive, not who wins. Two "
            "mechanisms are visible in the measurement and both are consistent with "
            "the calibrated twin's dominant error being the static defocus "
            "(10.0847 mm): (1) the flat spot itself improves under kolmogorov, because "
            "a total-RMS-matched von Kármán draw puts far less energy in high order "
            "(Noll 22 sits at 0.37× its iid RMS — see the ladder), and high-order "
            "aberration destroys a spot faster per radian² than low order, so matched "
            "RMS is not matched difficulty; (2) the amortised phase does not convert "
            "that into a win, because it is still commanded onto a bench whose "
            "remaining error is the defocus the network was never asked to model."
        )
        lines.append("")
        lines.append(
            "Scope: this is **this** calibrated bench, **this** corpus and **this** "
            "checkpoint. It is not a statement about turbulence in general, and it is "
            "not evidence that the network is insensitive to aberration *shape* — it is "
            "evidence that on a defocus-dominated bench the shape of the high-order "
            "ensemble does not decide the ordering."
        )
    else:
        lines.append(
            "The ensemble changed the verdict, so the fair run's conclusion does not "
            "transfer to a spatially correlated state distribution. Before quoting "
            "that, check the per-family table and the paired test above: the fair "
            "report's claim is per-family, not the cross-kind mean."
        )
    lines.append("")
    return {
        "flat_vs_net_ordering_changed": bool(flat_vs_net_changed),
        "within_net_ordering_changed": bool(within_net_changed),
        "within_net_resolvable": None if within_net_resolvable is None else bool(within_net_resolvable),
        "within_net_spread": {"iid": float(iid_spread), "kolmogorov": float(ko_spread)},
        "draw_noise_floor": None if draw_noise is None else float(draw_noise),
        "iid_net_order": iid_net_order,
        "kolmogorov_net_order": ko_net_order,
        "flat_still_best": bool(iid_rank["flat_is_best"] and ko_rank["flat_is_best"]),
        "net_open_family_wins": {"iid": flat_wins_iid, "kolmogorov": flat_wins_ko},
        "gap_to_flat": {
            "iid": iid_rank["gap_to_flat"],
            "kolmogorov": ko_rank["gap_to_flat"],
        },
        "lines": lines,
    }


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Run both arms, the optional noise-floor replicates, and write the report."""
    from ml.sim2f.fit import DEFAULT_LADDER_EXPOSURES

    parser = argparse.ArgumentParser(
        prog="python runs/sim2f_kl_ab/run_two_arms.py",
        description="iid vs kolmogorov aberration ensemble, same corpus + checkpoint.",
    )
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--floor-seed", type=int, default=8)
    parser.add_argument("--large-n", type=int, default=4096)
    parser.add_argument("--out-root", default=str(HERE))
    parser.add_argument("--no-noise-floor", action="store_true")
    parser.add_argument("--no-ladder", action="store_true")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="read existing loop/<arm>/eval_loop.json instead of running the loop "
        "(report regeneration without compute; the json records which mode produced it)",
    )
    args = parser.parse_args(argv)

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = REPO / out_root
    cache_dir = REPO / args.cache
    checkpoint = REPO / args.checkpoint
    baseline_path = REPO / args.baseline
    ladder = None if args.no_ladder else DEFAULT_LADDER_EXPOSURES

    driver_started = _now()
    configs = build_configs()
    verification = verify_conceptual(cache_dir)

    runs: dict[str, dict[str, Any]] = {}
    replicates: dict[str, dict[str, Any]] = {}
    commands: list[dict[str, Any]] = []
    if args.reuse:
        for ensemble in ("iid", "kolmogorov"):
            runs[ensemble] = _reuse_arm(ensemble, out_root / "loop" / ensemble)
        if not args.no_noise_floor:
            for ensemble in ("iid", "kolmogorov"):
                replicates[ensemble] = _reuse_arm(
                    f"{ensemble}_seed{args.floor_seed}",
                    out_root / "loop" / f"{ensemble}_seed{args.floor_seed}",
                )
        commands.append(
            {
                "command": f"python runs/sim2f_kl_ab/run_two_arms.py --reuse "
                f"--samples {args.samples} --rounds {args.rounds} --seed {args.seed}",
                "arm": "reused",
                "wall_s": 0.0,
            }
        )
    else:
        for ensemble in ("iid", "kolmogorov"):
            out_dir = out_root / "loop" / ensemble
            runs[ensemble] = run_arm(
                ensemble,
                configs[ensemble],
                cache_dir=cache_dir,
                checkpoint=checkpoint,
                out_dir=out_dir,
                samples=args.samples,
                rounds=args.rounds,
                seed=args.seed,
                ladder=ladder,
            )
            commands.append(
                {
                    "command": (
                        "python runs/sim2f_kl_ab/run_two_arms.py "
                        f"--samples {args.samples} --rounds {args.rounds} --seed {args.seed}"
                    ),
                    "arm": ensemble,
                    "wall_s": runs[ensemble]["wall_s"],
                }
            )

        if not args.no_noise_floor:
            for ensemble in ("iid", "kolmogorov"):
                out_dir = out_root / "loop" / f"{ensemble}_seed{args.floor_seed}"
                replicates[ensemble] = run_arm(
                    f"{ensemble}_seed{args.floor_seed}",
                    configs[ensemble],
                    cache_dir=cache_dir,
                    checkpoint=checkpoint,
                    out_dir=out_dir,
                    samples=args.samples,
                    rounds=args.rounds,
                    seed=args.floor_seed,
                    ladder=None,
                )
                commands.append(
                    {
                        "command": (
                            "python runs/sim2f_kl_ab/run_two_arms.py "
                            f"--samples {args.samples} --rounds {args.rounds} "
                            f"--floor-seed {args.floor_seed}  # noise-floor replicate"
                        ),
                        "arm": f"{ensemble}_seed{args.floor_seed}",
                        "wall_s": replicates[ensemble]["wall_s"],
                    }
                )

    comparison = baseline_diff(runs["iid"]["payload"], baseline_path)
    comparison["values"] = {
        "published": {
            arm: {column: float(value) for column, value in row.items()}
            for arm, row in _load_json(baseline_path)["aggregate"].items()
        },
        "iid": {
            arm: {column: float(value) for column, value in row.items()}
            for arm, row in runs["iid"]["payload"]["aggregate"].items()
        },
    }

    tables = {ensemble: family_table(runs[ensemble]["payload"]) for ensemble in runs}
    ranks = {ensemble: ordering(runs[ensemble]["payload"]) for ensemble in runs}
    ensemble_diff = ensemble_difference(
        runs["iid"]["payload"], runs["kolmogorov"]["payload"], seed=args.seed, large_n=args.large_n
    )
    paired = paired_analysis(runs["iid"]["payload"], runs["kolmogorov"]["payload"])
    floor = noise_floor(base=runs, replicates=replicates) if replicates else {}
    verdict = build_verdict(
        iid_payload=runs["iid"]["payload"],
        ko_payload=runs["kolmogorov"]["payload"],
        iid_rank=ranks["iid"],
        ko_rank=ranks["kolmogorov"],
        iid_families=tables["iid"],
        ko_families=tables["kolmogorov"],
        paired=paired,
        floor=floor,
    )

    floor_lines: list[str] = []
    if floor:
        floor_lines.append(
            f"Same ensemble, `LoopConfig.seed={args.floor_seed}` instead of {args.seed} — "
            "the detector is off, so this is pure state-draw variation."
        )
        floor_lines.append("")
        floor_lines.append("| ensemble | arm | |Δobjective| | |Δncc| | |Δ net_open−flat gap| |")
        floor_lines.append("|---|---|---|---|---|")
        for ensemble, entry in floor.items():
            for arm in ARMS:
                gap = entry["flat_vs_net_gap"].get(arm)
                floor_lines.append(
                    f"| {ensemble} | `{arm}` | "
                    f"{entry['per_arm_abs_change'][arm]['objective']:.4f} | "
                    f"{entry['per_arm_abs_change'][arm]['ncc']:.4f} | "
                    f"{_fmt(gap['abs_change'], '.4f') if gap else '-'} |"
                )
        floor_lines.append("")
        floor_lines.append(
            "max |Δobjective| over arms: "
            + ", ".join(
                f"{ensemble} {entry['max_abs_change_objective']:.4f}"
                for ensemble, entry in floor.items()
            )
            + "; max |Δ gap to flat|: "
            + ", ".join(
                f"{ensemble} {entry['max_abs_change_gap']:.4f}" for ensemble, entry in floor.items()
            )
            + "."
        )
    else:
        floor_lines.append("Noise floor not measured (`--no-noise-floor`).")

    status = _git("status", "--porcelain", "--", "src/ml/sim2f", "src/ao_shaping", "scripts", "report", "docs")
    payload = {
        "generated_by": "runs/sim2f_kl_ab/run_two_arms.py",
        "preregistration": PRE_REGISTRATION,
        "conceptual_verification": verification,
        "setup": {
            "cache": str(args.cache),
            "checkpoint": str(args.checkpoint),
            "baseline": str(args.baseline),
            "samples": int(args.samples),
            "rounds": int(args.rounds),
            "seed": int(args.seed),
            "floor_seed": None if args.no_noise_floor else int(args.floor_seed),
            "use_detector": False,
            "large_n": int(args.large_n),
            "mode": (
                "report regenerated from existing loop reports (--reuse), no loop run"
                if args.reuse
                else "ran ml.sim2f.loop.run_loop for every arm"
            ),
            "git_head": _git("rev-parse", "HEAD"),
            "tracked_sources_dirty_snapshot": status,
            "tracked_sources_dirty_note": (
                "Snapshot of `git status --porcelain` over the source trees at run "
                "time. It is RECORDED, not produced: this driver writes only under "
                "runs/sim2f_kl_ab/ (git-untracked) and edits no tracked file. The one "
                "sim2f entry, src/ml/sim2f/bench.py, predates this run (a concurrent "
                "lane owns it); the iid arm still reproduces the published baseline "
                "bit for bit, so it does not affect these numbers."
            ),
            "driver_wrote_only": "runs/sim2f_kl_ab/ (git-untracked)",
            "aberration_configs": {name: cfg.to_dict() for name, cfg in configs.items()},
        },
        "commands": commands,
        "driver_wall_s": float(_now() - driver_started),
        "arms": {
            name: {
                "wall_s": run["wall_s"],
                "eval_loop": run["eval_loop"],
                "payload": run["payload"],
                "ordering": ranks[name],
            }
            for name, run in runs.items()
        },
        "replicates": {
            name: {
                "wall_s": run["wall_s"],
                "eval_loop": run["eval_loop"],
                "aggregate": run["payload"]["aggregate"],
                "seed": run["payload"]["loop"]["seed"],
            }
            for name, run in replicates.items()
        },
        "baseline": comparison,
        "per_family": tables,
        "ensemble": ensemble_diff,
        "paired": paired,
        "noise_floor": floor,
        "noise_floor_lines": floor_lines,
        "verdict": verdict,
    }
    _write_json(out_root / "two_arms.json", payload)
    (out_root / "two_arms.md").write_text(render_markdown(payload_json=payload), encoding="utf-8")

    # Printed table: the same numbers, for the log.
    for ensemble in ("iid", "kolmogorov"):
        print(f"\n=== {ensemble} ===")
        print("\n".join(_arm_table(runs[ensemble]["payload"])))
        print(f"rank: {' < '.join(ranks[ensemble]['rank_by_objective'])}")
    print(
        f"\nbaseline reproduction: max |Δ| = {comparison['max_abs_diff']:.3e} "
        f"({comparison['exact_columns']}/{comparison['total_columns']} exact)"
    )
    print(
        f"verdict: flat_vs_net_ordering_changed={verdict['flat_vs_net_ordering_changed']} "
        f"within_net_ordering_changed={verdict['within_net_ordering_changed']} "
        f"(resolvable={verdict['within_net_resolvable']})"
    )
    print("\n".join(verdict["lines"]))
    print(f"\nreport -> {out_root / 'two_arms.md'} + two_arms.json")
    print(f"driver wall clock: {payload['driver_wall_s']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
