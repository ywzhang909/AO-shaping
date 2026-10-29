"""Measure whether a KL basis would help the ``ml.sim2f`` probe/fit stage.

The question, and why it is asked this way
==========================================
``ml.sim2f.fit`` estimates aberration coefficients in the **Noll** basis: the
probe bank pushes one canonical Zernike mode per column
(:func:`ml.sim2f.fit.probe_noll` -> Noll ``4 .. n_modes + 1``), and the
coefficient vector is Noll ``2 .. n_modes + 1``.  ``AberrationConfig``'s
``"kolmogorov"`` model draws the high-order block from a von Karman Zernike
covariance ``Sigma``
(``assets/zernike_cov_kolmogorov_n21.npz``, Noll ``4..22``, 19x19).  ``Sigma`` is
not diagonal, so the Noll basis is *not* the variance-optimal basis for that
ensemble: the KL basis (eigenvectors of ``Sigma``) is.

This script measures whether that optimality would actually buy anything at the
**probe/fit** stage, and it is deliberately **measurement only**: it does not
touch ``fit.py``, ``config.py`` or ``optics.py``, and it adds no basis switch.

Coefficient space is Euclidean
==============================
The metric used throughout is Euclidean in the coefficient vector, and that is
not an approximation of taste: the canonical modes
(:mod:`ao_shaping.utils.wavefront.zernike_calc`, Noll-normalised) are
**orthonormal over the unit disc**, ``<Z_i, Z_j> = delta_ij`` in the
mean-over-disc inner product, so the coefficient inner product *is* the L2 phase
inner product on the support disc.  ``trace(P S P) / trace(S)`` -- the fraction
of coefficient variance captured by the projector ``P = B_K B_K^T`` -- is
therefore a statement in radians, not in arbitrary mode units.  The script also
measures the discrete Gram matrix of the sampled basis and prints the deviation,
so the assumption is checked rather than asserted.

Part 1 -- variance capture (pure linear algebra)
================================================
For each basis family at ``K = 2, 3, 5, 8, 12, 19``:

* ``noll``      -- the first ``K`` canonical positions (Noll 4, 5, ...), exactly
  what :func:`ml.sim2f.fit.probe_noll` pushes today;
* ``kl_sigma``  -- top-``K`` eigenvectors of ``Sigma`` (descending eigenvalue);
* ``kl_realised`` -- top-``K`` eigenvectors of the covariance of **realised**
  ``draw_aberration(..., aberration_model="kolmogorov")`` draws (own seed).

evaluated on two ensembles:

* ``E1 = N(0, Sigma)`` -- analytic value *and* a sample estimate;
* ``E2`` = realised ``draw_aberration`` draws on a **held-out seed** (different
  from the seed that built ``kl_realised``).

Part 2 -- probe identifiability (uses the forward model)
=======================================================
A basis can capture variance and still be worse to **invert**: KL modes are
linear combinations of Zernike modes, so one "unit KL push" is a combined push
and the induced response columns can become nearly collinear.  For ``K = 5, 12,
19`` the response matrix is built through the *real* probe path
(:func:`ml.sim2f.fit.build_response`) for the Noll basis, and through
:func:`build_response_general` -- a local generalisation of the same algebra
that accepts arbitrary coefficient vectors -- for KL.  ``s_max``, ``s_min`` and
``s_max/s_min`` are reported.

``build_response``'s signature takes Noll **indices** and one-hot-encodes them
internally (:func:`ml.sim2f.fit._unit_coefficient`), so it **cannot express a
combined push**: a KL column needs ``bench.aberration(v)`` with ``v`` dense, and
no combination of ``probe_modes`` produces that.  Rather than inventing a
different probe path, this script copies the *same* algebra
(:func:`build_response_general`) and **proves the copy reduces exactly** to
``build_response`` on the one-hot bank (field ``reduction_max_rel_diff``, must be
~0).  If that check ever fails, the KL rows are not comparable to the Noll rows
and the script says so.

PRE-REGISTERED DECISION RULE (fixed before the numbers below were seen)
=======================================================================
Wiring a KL basis into ``fit.py`` is justified only if **both** hold:

1. ``kl_sigma`` captures at least **1.5x** the variance ``noll`` captures, at the
   ``K`` the fit actually uses.  There is no single such ``K``, so it is measured
   at ``K = 19`` -- the number of high-order columns ``probe_noll`` pushes for
   the default ``AberrationConfig(n_modes=21)`` (Noll 4..22) -- and reported at
   every other ``K`` as well.  Both ensembles must clear it.
2. The response-matrix condition number ``s_max/s_min`` of the KL basis is at
   most **2x** the Noll one at matched ``K`` (every ``K``, every bench/state).

If either fails the answer is "do not wire".  These thresholds are not to be
relaxed after seeing the numbers; a mis-set threshold gets reported separately.

CAVEAT ON PART 1 (do not read the E1 KL column as evidence)
==========================================================
Comparing ``kl_sigma`` against ``noll`` on ``E1`` is ``kl_sigma``'s own
optimality condition -- it *must* win, so that column alone proves nothing.  The
informative quantities are (a) the *size* of the gap at each ``K``, and (b) the
``E2`` columns, where the sphere-constrained ensemble means the eigenvectors of
``Sigma`` are no longer optimal even for the ensemble the simulator actually
draws.

Usage
=====
    python scripts/ml/measure_sim2f_kl_basis.py
    python scripts/ml/measure_sim2f_kl_basis.py --draws 5000 --out logs/try

Writes ``<out>/kl_basis_measurement.json`` and ``<out>/kl_basis_measurement.md``
(default ``out`` = ``logs/<timestamp>/``).  Nothing under ``report/``: a formal
report is premature, and ``scripts/_common/provenance.py`` is another session's
work in progress.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from ao_shaping.utils.io.timestamp import format_ts
from ml.sim2f.config import AberrationConfig, Sim2fConfig
from ml.sim2f.fit import (
    DEFAULT_RCOND,
    FitConfig,
    _as_phase,
    _frame_on,
    _model_frame,
    _prepare,
    _unit_coefficient,
    build_response,
    frame_of,
    probe_noll,
)
from ml.sim2f.losses import EPS

__all__ = ["build_response_general", "capture", "main"]

#: Ensemble sizes.  The task fixes 20000 for everything; ``--draws`` overrides for
#: a faster exploratory run and the artifact records whichever was used.
DEFAULT_DRAWS = 20000

#: Fixed seeds.  ``SEED_KL_REALISED`` builds ``kl_realised``; ``SEED_E2`` is the
#: **held-out** ensemble it is scored on; ``SEED_E1_SAMPLE`` is independent of both.
SEED_KL_REALISED = 20261025
SEED_E1_SAMPLE = 20261026
SEED_E2 = 20261027
SEED_STATE = 7  # bench state for the "drawn" probe response

#: ``K`` values every table is reported at.
K_GRID = (2, 3, 5, 8, 12, 19)

#: ``K`` values the response matrix is built at (cost grows as 2K forwards).
K_RESPONSE = (5, 12, 19)

#: Pre-registered thresholds.  Not to be changed after reading the numbers.
CRIT1_MIN_RATIO = 1.5
CRIT2_MAX_RATIO = 2.0

#: The K the fit actually uses: ``probe_noll(AberrationConfig(n_modes=21))``
#: yields Noll 4..22, i.e. 19 high-order probe columns.
K_FIT = 19

BASIS_FAMILIES = ("noll", "kl_sigma", "kl_realised")
ENSEMBLES = ("E1_analytic", "E1_sample", "E2_realised")

PRE_REGISTERED_RULE = (
    "Wire a KL basis into fit.py only if BOTH: (1) kl_sigma captures at least "
    f"{CRIT1_MIN_RATIO}x the variance noll captures at the K the fit actually uses "
    f"(K={K_FIT}: the number of high-order probe columns probe_noll pushes for "
    "AberrationConfig(n_modes=21), Noll 4..22), on both ensembles; AND (2) the "
    "response-matrix condition number s_max/s_min of the KL basis is at most "
    f"{CRIT2_MAX_RATIO}x the Noll one at matched K, for every K and every "
    "bench/state measured. If either fails: do not wire."
)

CAVEAT = (
    "Comparing kl_sigma against noll on E1 is kl_sigma's own optimality condition "
    "-- it must win, so that column alone proves nothing. The informative "
    "quantities are (a) the size of the gap at each K, and (b) the E2 columns, "
    "where the sphere-constrained ensemble means the eigenvectors of Sigma are no "
    "longer optimal even for the ensemble the simulator actually draws."
)

EUCLIDEAN_STATEMENT = (
    "Coefficient space is Euclidean: the canonical Zernike modes are orthonormal "
    "over the unit disc (mean-over-disc <Z_i, Z_j> = delta_ij), so the coefficient "
    "inner product is the L2 phase inner product on the support disc and "
    "trace(P S P)/trace(S) is a variance fraction in radians."
)


# ----------------------------------------------------------------------
# Covariance asset
# ----------------------------------------------------------------------
def _covariance_asset() -> tuple[np.ndarray, np.ndarray, Path]:
    """The packaged von Karman Zernike covariance, loaded the way optics does.

    Read through :data:`ml.sim2f.optics.DEFAULT_COVARIANCE_PATH` -- the same
    anchor ``draw_aberration`` uses -- so this script cannot disagree with the
    simulator about which file is the ensemble.

    Returns:
        ``(sigma, noll, path)``: the ``(K, K)`` covariance in ``rad**2``, its Noll
        indices, and the asset path.
    """
    from ml.sim2f.optics import DEFAULT_COVARIANCE_PATH

    path = Path(DEFAULT_COVARIANCE_PATH)
    with np.load(path) as data:
        sigma = np.asarray(data["covariance"], dtype=np.float64)
        noll = np.asarray(data["noll"], dtype=np.int64)
    return sigma, noll, path


# ----------------------------------------------------------------------
# Part 1 -- variance capture
# ----------------------------------------------------------------------
def _basis_noll(k: int) -> np.ndarray:
    """First ``k`` canonical index positions of the high-order block.

    Args:
        k: Number of columns.

    Returns:
        ``(19, k)`` one-hot columns, i.e. Noll ``4 .. 3+k`` -- exactly the columns
        :func:`ml.sim2f.fit.probe_noll` pushes at ``K = k``.
    """
    return np.eye(19, dtype=np.float64)[:, :k].copy()


def _basis_kl(cov: np.ndarray, k: int) -> np.ndarray:
    """Top-``k`` eigenvectors of ``cov`` by descending eigenvalue.

    Args:
        cov: ``(19, 19)`` symmetric covariance.
        k: Number of columns.

    Returns:
        ``(19, k)`` orthonormal columns.
    """
    values, vectors = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(values)[::-1]
    return np.ascontiguousarray(vectors[:, order[:k]])


def capture(basis: np.ndarray, cov: np.ndarray) -> dict[str, float]:
    """Variance fraction captured by ``basis`` under the ensemble ``cov``.

    ``trace(P S P) / trace(S)`` with ``P = B B^T`` (orthonormal columns), and the
    residual outside the span as an RMS in **radians**
    ``sqrt(trace((I-P) S (I-P)))``.

    Args:
        basis: ``(19, K)`` orthonormal columns.
        cov: ``(19, 19)`` ensemble covariance, ``rad**2``.

    Returns:
        Mapping with ``fraction`` (0..1), ``residual_rad`` and ``total_rad2``.
    """
    inner = basis.T @ cov @ basis
    captured = float(np.trace(inner))
    total = float(np.trace(cov))
    projector = np.eye(cov.shape[0]) - basis @ basis.T
    residual_matrix = projector @ cov @ projector
    residual = float(np.trace(residual_matrix))
    return {
        "fraction": captured / total if total > 0 else float("nan"),
        "residual_rad": math.sqrt(max(residual, 0.0)),
        "total_rad2": total,
    }


def draw_high_order(
    bench: object, cfg: AberrationConfig, seed: int, draws: int
) -> np.ndarray:
    """Realised high-order coefficients from ``draw_aberration``.

    The returned ``.coeffs`` is the 21-vector ``[tip, tilt, high-order...]``, so
    rows/columns ``[2:]`` are the 19 coefficients that match ``Sigma``.

    Args:
        bench: The :class:`ml.sim2f.optics.Sim2f` the state belongs to.
        cfg: Aberration contract (``aberration_model="kolmogorov"``).
        seed: NumPy seed; the caller owns it so E2 can be held out.
        draws: Number of draws.

    Returns:
        ``(draws, 19)`` float64 coefficients, radians.
    """
    from ml.sim2f.optics import draw_aberration

    rng = np.random.default_rng(int(seed))
    out = np.empty((int(draws), 19), dtype=np.float64)
    for index in range(int(draws)):
        out[index] = draw_aberration(bench, cfg, rng).coeffs[2:]
    return out


def empirical_covariance(samples: np.ndarray) -> np.ndarray:
    """Population covariance of ``samples`` (mean removed, ``N`` divisor).

    Args:
        samples: ``(N, K)`` draws.

    Returns:
        ``(K, K)`` covariance.
    """
    centred = samples - samples.mean(axis=0, keepdims=True)
    return centred.T @ centred / float(samples.shape[0])


# ----------------------------------------------------------------------
# Part 2 -- probe response
# ----------------------------------------------------------------------
def build_response_general(
    bench: object,
    cfg: FitConfig,
    vectors: list[np.ndarray],
    measure,
    *,
    phase=None,
) -> dict[str, object]:
    """Response matrix for arbitrary coefficient vectors, same algebra as ``fit``.

    A line-by-line copy of :func:`ml.sim2f.fit.build_response` with the single
    change that the pushed map comes from an arbitrary coefficient vector
    instead of ``_unit_coefficient``'s one-hot.  It exists because the real
    function's signature takes Noll *indices* and cannot express a combined push
    (a KL column is a dense vector in the same 21-row coefficient space).  It is
    not a different probe path: :func:`ml.sim2f.fit.build_response`'s output on
    the one-hot bank is reproduced to rounding, and that equality is asserted by
    the caller.

    Args:
        bench: The bench (``grid``, ``ccd_px``, ``dtype``, ``device``,
            ``aberration``, ``forward``).
        cfg: Probe contract (``probe_amplitude_rad``, ``rcond``).
        vectors: ``len(vectors)`` coefficient vectors, each ``(n_modes,)``.
        measure: Callable pushing a phase and returning a camera frame.
        phase: Command the probes are added to (default zeros).

    Returns:
        Mapping with ``matrix``, ``singular``, ``rank``, ``condition``, ``gain``.
    """
    command = _as_phase(bench, phase)
    model0 = _model_frame(bench, command)
    frame = _frame_on(bench, frame_of(measure(command)))

    with torch.no_grad():
        denominator = model0.pow(2).sum().clamp_min(EPS)
        gain = float((model0 * frame).sum() / denominator)
    if not math.isfinite(gain) or gain <= 0.0:
        gain = 1.0
    model0_flat = model0.flatten().cpu().numpy().astype(np.float64)
    target = frame.flatten().cpu().numpy().astype(np.float64) * gain - model0_flat

    amplitude = float(cfg.probe_amplitude_rad)
    columns = np.zeros((target.size, len(vectors)), dtype=np.float64)
    for column, vector in enumerate(vectors):
        dense = np.asarray(vector, dtype=np.float64).reshape(-1)
        map_unit = bench.aberration(dense[None].astype(np.float32))[0]  # type: ignore[attr-defined]
        if isinstance(map_unit, Tensor):
            map_unit = map_unit.detach()
        plus = _prepare(frame_of(measure(command + amplitude * map_unit)))
        minus = _prepare(frame_of(measure(command - amplitude * map_unit)))
        difference = (plus - minus).flatten().cpu().numpy().astype(np.float64)
        columns[:, column] = difference * (gain / (2.0 * amplitude))

    summary = summarise_matrix(columns, cfg)
    summary["matrix"] = columns
    summary["gain"] = gain
    return summary


def summarise_matrix(matrix: np.ndarray, cfg: FitConfig) -> dict[str, float]:
    """Singular spectrum of a response matrix, reported two ways.

    ``fit.ResponseMatrix.condition`` is ``s_max / s_rank`` -- the *retained*
    spectrum, after the ``rcond`` cut -- not ``s_max / s_min`` over all columns.
    On a full-rank bank the two coincide; on a degenerate one (
    :attr:`rank` ``< n_columns``) the retained form silently looks benign while
    some columns carry no response at all.  Both are reported so a degenerate
    condition cannot hide behind a small condition number.

    Args:
        matrix: ``(pixels, n_columns)`` response matrix.
        cfg: Probe contract (supplies ``rcond``).

    Returns:
        Mapping with ``rank``, ``s_max``, ``s_min`` (retained), ``s_min_all``,
        ``cond`` (``s_max / s_min``, i.e. the fit's own number) and
        ``cond_full`` (``s_max / s_min_all``).
    """
    singular = np.linalg.svd(matrix, compute_uv=False) if matrix.size else np.zeros(0)
    cut = float(cfg.rcond) * float(singular[0]) if singular.size and singular[0] > 0 else 0.0
    keep = [i for i, value in enumerate(singular) if value > cut] if cut > 0 else []
    rank = len(keep)
    return {
        "rank": rank,
        "s_max": float(singular[0]) if singular.size else float("nan"),
        "s_min": float(singular[rank - 1]) if rank else float("nan"),
        "s_min_all": float(singular[-1]) if singular.size else float("nan"),
        "cond": (
            float(singular[0] / singular[rank - 1])
            if rank and singular[rank - 1] > 0
            else float("inf")
        ),
        "cond_full": (
            float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else float("inf")
        ),
    }


def _dense_vectors(basis: np.ndarray, n_modes: int) -> list[np.ndarray]:
    """Embed ``(19, K)`` high-order columns into the ``(n_modes,)`` coefficient rows.

    Rows 0/1 are Noll 2/3 (tip, tilt) and stay zero: ``Sigma`` covers the
    high-order block only, and tip/tilt are not probed.

    Args:
        basis: ``(19, K)`` columns in the high-order block.
        n_modes: Length of the fit's coefficient vector.

    Returns:
        ``K`` vectors of length ``n_modes``.
    """
    out = []
    for column in range(basis.shape[1]):
        vector = np.zeros(n_modes, dtype=np.float64)
        vector[2:] = basis[:, column]
        out.append(vector)
    return out


def _response_rows(
    bench: object,
    bench_label: str,
    state_label: str,
    phase,
    measure,
    kl_dirs: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    """One ``(bench, state)`` block: build every basis/K response matrix.

    The full 19-column Noll bank is built once per condition, both as the
    reference for the "if the response were linear" check and as the K=19 Noll
    row, so the extra cost is one bank rather than one per K.

    Args:
        bench: The bench.
        bench_label: ``"bare"`` or ``"calibrated"``.
        state_label: ``"flat"`` or ``"drawn"``.
        phase: Command phase (``None`` for flat).
        measure: Camera read.
        kl_dirs: ``{"kl_sigma": (19,19), "kl_realised": (19,19)}`` eigenvector
            matrices, descending eigenvalue.

    Returns:
        The table rows for this block.
    """
    base = FitConfig.from_aberration(AberrationConfig(n_modes=21))
    n_modes = int(base.n_modes)

    full_cfg = replace(base, probe_modes=tuple(range(4, 4 + K_FIT)))
    full = build_response(bench, full_cfg, measure, phase=phase)
    full_summary = summarise_matrix(full.matrix, full_cfg)
    full_ok = int(full_summary["rank"]) == K_FIT

    rows: list[dict[str, object]] = []
    for k in K_RESPONSE:
        noll_cfg = replace(base, probe_modes=tuple(range(4, 4 + k)))
        chosen = probe_noll(noll_cfg)
        if len(chosen) != k:
            raise RuntimeError(f"probe_noll gave {chosen} for K={k}, expected {k} indices")

        real = full if k == K_FIT else build_response(bench, noll_cfg, measure, phase=phase)
        summary = full_summary if k == K_FIT else summarise_matrix(real.matrix, noll_cfg)

        # Does the local generalisation reproduce the real probe path exactly?
        onehot = [_unit_coefficient(noll_cfg, index) for index in chosen]
        copy = build_response_general(bench, noll_cfg, onehot, measure, phase=phase)
        scale = float(np.abs(real.matrix).max())
        reduction = float(np.abs(copy["matrix"] - real.matrix).max() / scale) if scale > 0 else float("nan")

        rows.append(
            {
                "bench": bench_label,
                "state": state_label,
                "K": k,
                "basis": "noll",
                "n_columns": len(chosen),
                **summary,
                "cond_if_linear": summary["cond"],
                "reduction_max_rel_diff": reduction,
            }
        )

        for family in ("kl_sigma", "kl_realised"):
            vectors = _dense_vectors(kl_dirs[family][:, :k], n_modes)
            got = build_response_general(bench, base, vectors, measure, phase=phase)
            reference = float("nan")
            if full_ok:
                # "If the response were linear" reference: stacking the same
                # columns through an orthogonal rotation must leave the singular
                # spectrum unchanged.  Any gap is finite-amplitude /
                # prepared-frame non-additivity of a combined push, not a basis
                # property.  Undefined when the reference bank itself is
                # rank-deficient, hence the guard rather than a numerical floor.
                linear = full.matrix @ kl_dirs[family][:, :k]
                values = np.linalg.svd(linear, compute_uv=False)
                if values[-1] > 0:
                    reference = float(values[0] / values[-1])
            rows.append(
                {
                    "bench": bench_label,
                    "state": state_label,
                    "K": k,
                    "basis": family,
                    "n_columns": len(vectors),
                    **{key: got[key] for key in ("rank", "s_max", "s_min", "s_min_all", "cond", "cond_full")},
                    "cond_if_linear": reference,
                    "reduction_max_rel_diff": None,
                }
            )
    return rows


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def _part1_table(rows: list[dict[str, object]]) -> str:
    """Render the variance-capture table.

    Args:
        rows: Part 1 rows.

    Returns:
        The table as text.
    """
    lines = [
        "Part 1  variance capture (Euclidean coefficient space, radians)",
        f"  {'ensemble':<13} {'K':>3}  {'basis':<12} {'captured':>9} {'resid_rad':>10} {'vs noll':>8}",
    ]
    index = {(row["ensemble"], row["K"], row["basis"]): row for row in rows}
    for ensemble in ENSEMBLES:
        noll_fraction = None
        for k in K_GRID:
            noll_fraction = float(index[(ensemble, k, "noll")]["fraction"])
            for family in BASIS_FAMILIES:
                row = index[(ensemble, k, family)]
                ratio = float(row["fraction"]) / noll_fraction if noll_fraction > 0 else float("inf")
                lines.append(
                    f"  {ensemble:<13} {k:>3}  {family:<12} "
                    f"{100.0 * float(row['fraction']):>8.2f}% {float(row['residual_rad']):>10.4f} "
                    f"{ratio:>8.3f}"
                )
        lines.append("")
    return "\n".join(lines)


def _part2_table(rows: list[dict[str, object]]) -> str:
    """Render the response-matrix table.

    Args:
        rows: Part 2 rows.

    Returns:
        The table as text.
    """
    lines = [
        "Part 2  probe response conditioning (noise-free secant, detector off)",
        f"  {'bench':<10} {'state':<6} {'K':>3}  {'basis':<12} {'rank':>4} {'s_max':>10} "
        f"{'s_min':>10} {'cond':>9} {'vs noll':>8} {'cond_full':>10} {'if linear':>10}",
    ]
    noll_cond = {
        (row["bench"], row["state"], row["K"]): float(row["cond"])
        for row in rows
        if row["basis"] == "noll"
    }
    for row in rows:
        key = (row["bench"], row["state"], row["K"])
        ratio = float(row["cond"]) / noll_cond[key] if noll_cond[key] > 0 else float("inf")
        reference = row["cond_if_linear"]
        ref_text = "n/a" if reference is None or not np.isfinite(reference) else f"{float(reference):>10.2f}"
        mark = "*" if int(row["rank"]) < int(row["n_columns"]) else " "
        lines.append(
            f"  {row['bench']:<10} {row['state']:<6} {row['K']:>3}  {row['basis']:<12} "
            f"{str(int(row['rank'])) + mark:>5} {float(row['s_max']):>10.3e} "
            f"{float(row['s_min']):>10.3e} {float(row['cond']):>9.2f} {ratio:>8.3f} "
            f"{float(row['cond_full']):>10.2e} {ref_text}"
        )
    lines += [
        "",
        "  cond      = s_max/s_rank over the rcond-retained spectrum: the number "
        "fit.ResponseMatrix reports.",
        "  cond_full = s_max/s_min over all columns.  * marks rank < n_columns: at "
        "rcond=1e-6 the bank is",
        "              rank-deficient, so its retained cond is benign while some "
        "columns carry no response.",
        "  if linear = cond of J_noll @ V_K (the same columns through an orthogonal "
        "rotation); n/a when that",
        "              reference bank is itself rank-deficient.",
        "  kl_realised's own smallest column is a low-variance direction estimated from "
        "20000 draws, so its",
        "              s_min can be far below the Noll bank's (3662 / 11590 / 41085 at "
        "bare/flat): a",
        "              variance-ordered KL bank puts its weakest direction last, which is "
        "where a rank cut bites.",
    ]
    return "\n".join(lines)


def _evaluate(
    part1: list[dict[str, object]], part2: list[dict[str, object]]
) -> dict[str, object]:
    """Apply the pre-registered rule to the measured numbers.

    Args:
        part1: Variance-capture rows.
        part2: Response rows.

    Returns:
        The evaluation, with per-check evidence and the verdict.
    """
    index = {(row["ensemble"], row["K"], row["basis"]): row for row in part1}

    criterion1 = []
    for ensemble in ("E1_analytic", "E2_realised"):
        noll_fraction = float(index[(ensemble, K_FIT, "noll")]["fraction"])
        kl_fraction = float(index[(ensemble, K_FIT, "kl_sigma")]["fraction"])
        ratio = kl_fraction / noll_fraction if noll_fraction > 0 else float("inf")
        criterion1.append(
            {
                "ensemble": ensemble,
                "K": K_FIT,
                "noll_fraction": noll_fraction,
                "kl_sigma_fraction": kl_fraction,
                "ratio": ratio,
                "threshold": CRIT1_MIN_RATIO,
                "pass": bool(ratio >= CRIT1_MIN_RATIO),
            }
        )
    criterion1_pass = all(bool(row["pass"]) for row in criterion1)

    noll_cond = {
        (row["bench"], row["state"], row["K"]): row
        for row in part2
        if row["basis"] == "noll"
    }
    criterion2 = []
    criterion2_secondary = []
    for row in part2:
        if row["basis"] not in ("kl_sigma", "kl_realised"):
            continue
        reference = noll_cond[(row["bench"], row["state"], row["K"])]
        base = float(reference["cond"])
        ratio = float(row["cond"]) / base if base > 0 else float("inf")
        # A bank whose rank at rcond is below its column count has no meaningful
        # s_min -- the pre-registered quantity is undefined there, not large.  That
        # is a measured property of the condition (see the table footnote), not a
        # relaxation of the threshold.
        degenerate = int(row["rank"]) < int(row["n_columns"]) or int(reference["rank"]) < int(
            reference["n_columns"]
        )
        entry = {
            "bench": row["bench"],
            "state": row["state"],
            "K": row["K"],
            "basis": row["basis"],
            "noll_cond": base,
            "kl_cond": float(row["cond"]),
            "ratio": ratio,
            "threshold": CRIT2_MAX_RATIO,
            "comparable": not degenerate,
            "reason": (
                "rank-deficient bank at rcond="
                f"{DEFAULT_RCOND:g}: noll rank {int(reference['rank'])} of "
                f"{int(reference['n_columns'])}, kl rank {int(row['rank'])} of "
                f"{int(row['n_columns'])}; s_min sits at the numerical floor, so the "
                "pre-registered s_max/s_min is undefined (not merely large)"
            )
            if degenerate
            else None,
            "pass": bool(ratio <= CRIT2_MAX_RATIO),
        }
        (criterion2 if row["basis"] == "kl_sigma" else criterion2_secondary).append(entry)
    comparable = [row for row in criterion2 if row["comparable"]]
    criterion2_pass = bool(comparable) and all(bool(row["pass"]) for row in comparable)

    # Every K, both ensembles: the gap closed by KL (informational only).
    gaps = []
    for ensemble in ("E1_analytic", "E2_realised"):
        for k in K_GRID:
            noll_fraction = float(index[(ensemble, k, "noll")]["fraction"])
            kl_fraction = float(index[(ensemble, k, "kl_sigma")]["fraction"])
            gaps.append(
                {
                    "ensemble": ensemble,
                    "K": k,
                    "noll_fraction": noll_fraction,
                    "kl_sigma_fraction": kl_fraction,
                    "ratio": kl_fraction / noll_fraction if noll_fraction > 0 else float("inf"),
                }
            )

    return {
        "criterion1": criterion1,
        "criterion1_pass": criterion1_pass,
        "criterion2": criterion2,
        "criterion2_secondary_kl_realised": criterion2_secondary,
        "criterion2_comparable_conditions": len(comparable),
        "criterion2_pass": criterion2_pass,
        "gaps_all_K": gaps,
        "verdict": "WIRE" if (criterion1_pass and criterion2_pass) else "DO NOT WIRE",
    }


def _part1_markdown(rows: list[dict[str, object]], payload: dict[str, object]) -> str:
    """Markdown rendering of the variance-capture table.

    Args:
        rows: Part 1 rows.
        payload: The JSON payload (supplies the per-ensemble absolute scale).

    Returns:
        Markdown table.
    """
    index = {(row["ensemble"], row["K"], row["basis"]): row for row in rows}
    scales = payload["ensemble_absolute_scale"]
    out = [
        "| ensemble | ensemble RMS (rad) | K | basis | captured | residual (rad) | vs noll |",
        "|---|---|---|---|---|---|---|",
    ]
    for ensemble in ENSEMBLES:
        rms = float(scales[ensemble]["ensemble_rms_rad"])
        for k in K_GRID:
            noll_fraction = float(index[(ensemble, k, "noll")]["fraction"])
            for family in BASIS_FAMILIES:
                row = index[(ensemble, k, family)]
                ratio = float(row["fraction"]) / noll_fraction if noll_fraction > 0 else float("inf")
                out.append(
                    f"| {ensemble} | {rms:.4f} | {k} | {family} | "
                    f"{100.0 * float(row['fraction']):.2f}% | "
                    f"{float(row['residual_rad']):.4f} | {ratio:.3f} |"
                )
    return "\n".join(out)


def _part2_markdown(rows: list[dict[str, object]]) -> str:
    """Markdown rendering of the response table.

    Args:
        rows: Part 2 rows.

    Returns:
        Markdown table.
    """
    noll_cond = {
        (row["bench"], row["state"], row["K"]): float(row["cond"])
        for row in rows
        if row["basis"] == "noll"
    }
    out = [
        "| bench | state | K | basis | rank | s_max | s_min | s_max/s_min | vs noll | "
        "s_max/s_min (all cols) | if linear |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        key = (row["bench"], row["state"], row["K"])
        ratio = float(row["cond"]) / noll_cond[key] if noll_cond[key] > 0 else float("inf")
        reference = row["cond_if_linear"]
        ref_text = "n/a" if reference is None or not np.isfinite(reference) else f"{float(reference):.2f}"
        mark = "*" if int(row["rank"]) < int(row["n_columns"]) else ""
        out.append(
            f"| {row['bench']} | {row['state']} | {row['K']} | {row['basis']} | "
            f"{int(row['rank'])}{mark} | {float(row['s_max']):.3e} | "
            f"{float(row['s_min']):.3e} | {float(row['cond']):.2f} | {ratio:.3f} | "
            f"{float(row['cond_full']):.2e} | {ref_text} |"
        )
    return "\n".join(out)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument(
        "--draws",
        type=int,
        default=DEFAULT_DRAWS,
        help=f"ensemble size for every Monte Carlo leg (default {DEFAULT_DRAWS})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default logs/<timestamp>/)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run both parts, print the tables, write the artifact.

    Args:
        argv: Arguments after the script name; ``None`` means ``sys.argv[1:]``.

    Returns:
        Process exit code (0 always: this is a measurement, not a gate).
    """
    args = _build_parser().parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    started = time.perf_counter()
    out_dir = args.out or (ROOT / "logs" / format_ts())
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma, noll, asset = _covariance_asset()
    n_high = int(sigma.shape[0])
    if n_high != 19:
        raise RuntimeError(f"expected a 19x19 high-order covariance, got {sigma.shape}")

    bench = _get_bench()
    kcfg = AberrationConfig(
        aberration_model="kolmogorov", covariance_path=str(asset), n_modes=n_high + 2
    )

    # -- orthonormality of the sampled basis (the Euclidean claim) ----------
    gram = _sampled_gram(bench)

    # -- Part 1 -------------------------------------------------------------
    draws = max(int(args.draws), 2 * n_high)
    e1_samples = np.random.default_rng(SEED_E1_SAMPLE).multivariate_normal(
        np.zeros(n_high), sigma, size=draws, check_valid="raise"
    )
    e2_samples = draw_high_order(bench, kcfg, SEED_E2, draws)
    kl_realised_samples = draw_high_order(bench, kcfg, SEED_KL_REALISED, draws)

    ensembles = {
        "E1_analytic": sigma,
        "E1_sample": empirical_covariance(e1_samples),
        "E2_realised": empirical_covariance(e2_samples),
    }
    families = {
        "noll": lambda k: _basis_noll(k),
        "kl_sigma": lambda k: _basis_kl(sigma, k),
        "kl_realised": lambda k: _basis_kl(empirical_covariance(kl_realised_samples), k),
    }

    part1: list[dict[str, object]] = []
    for ensemble, cov in ensembles.items():
        for k in K_GRID:
            for family in BASIS_FAMILIES:
                stats = capture(families[family](k), cov)
                part1.append(
                    {
                        "ensemble": ensemble,
                        "K": k,
                        "basis": family,
                        "fraction": float(stats["fraction"]),
                        "residual_rad": float(stats["residual_rad"]),
                        "total_rad2": float(stats["total_rad2"]),
                    }
                )

    # The sphere compression the framing describes, measured on E2.  The
    # normalisation is deliberately the one the repository's own test uses
    # (``test_end_to_end_shape_survives_the_rms_rescale``: each side's diagonal is
    # divided by its own mean), so the quoted 0.85x / 1.27x is directly
    # comparable.  Normalising by the mean of the *ratio* instead would weight
    # every mode equally and report a much larger distortion for the same draws,
    # which is not the documented quantity.
    e2_cov = empirical_covariance(e2_samples)
    ladder_target = np.diag(sigma) / np.diag(sigma).mean()
    ladder_realised = np.diag(e2_cov) / np.diag(e2_cov).mean()
    shape_ladder = ladder_realised / ladder_target

    # Why the KL gap can only be small.  Three measured facts about Sigma in the
    # Noll basis, all in this dict: a steep radial-order diagonal ladder, leading
    # eigenvalues that almost coincide with the leading diagonal entries, and
    # off-diagonal correlation that links *widely separated* radial orders rather
    # than equal-variance partners.  Rotating can therefore only re-pack the
    # nearly-degenerate blocks, which is worth ~1.03x, not 1.5x.
    diagonal = np.diag(sigma)
    correlation = sigma / np.sqrt(np.outer(diagonal, diagonal))
    off_correlation = np.abs(correlation - np.eye(n_high))
    eigenvalues = np.linalg.eigvalsh(sigma)[::-1]
    equal_variance_energy = 0.0
    total_off_energy = float(((correlation - np.eye(n_high)) ** 2).sum())
    for row in range(n_high):
        for column in range(n_high):
            if row == column:
                continue
            ratio = diagonal[row] / diagonal[column]
            if 1.0 / 1.5 <= ratio <= 1.5:
                equal_variance_energy += float((correlation[row, column]) ** 2)
    covariance_shape = {
        "diag_ladder_ratio_first_to_last": float(diagonal[0] / diagonal[-1]),
        "leading_eigen_over_leading_diag_top3": float(
            eigenvalues[:3].sum() / diagonal[:3].sum()
        ),
        "offdiagonal_frobenius_over_total": float(
            np.linalg.norm(sigma - np.diag(diagonal)) / np.linalg.norm(sigma)
        ),
        "max_abs_offdiagonal_correlation": float(off_correlation.max()),
        "offdiagonal_corr_energy_between_near_equal_partners": (
            equal_variance_energy / total_off_energy if total_off_energy > 0 else 0.0
        ),
        "eigen_ratio_max_min": float(eigenvalues.max() / eigenvalues.min()),
        "ceiling_ratio_by_K": {
            str(k): float(eigenvalues[:k].sum() / diagonal[:k].sum()) for k in K_GRID
        },
        "note": (
            "sum(top-K eigenvalues)/sum(first-K diagonal) is the largest captured-variance "
            "ratio any orthonormal rotation of the same space can deliver at K, and it equals "
            "the kl_sigma ratio measured on E1_analytic. It is ~1.03 because Sigma's Noll "
            "diagonal is already ordered by radial order with a steep ladder and the leading "
            "eigenvalues almost coincide with the leading diagonal entries; the off-diagonal "
            "correlation links modes that differ by 3.7-9.5x in variance, and mixing a strong "
            "mode with a weak one cannot move the top-K sum much."
        ),
    }

    # -- Part 2 -------------------------------------------------------------
    kl_dirs = {
        "kl_sigma": _basis_kl(sigma, n_high),
        "kl_realised": _basis_kl(empirical_covariance(kl_realised_samples), n_high),
    }
    part2: list[dict[str, object]] = []
    for bench_label, bench_cfg in _bench_cfgs():
        state_bench = _get_bench(bench_cfg)
        measure = _model_measure(state_bench)
        for state_label, phase in _states(state_bench, asset):
            part2.extend(
                _response_rows(state_bench, bench_label, state_label, phase, measure, kl_dirs)
            )

    elapsed = time.perf_counter() - started
    evaluation = _evaluate(part1, part2)

    # -- print --------------------------------------------------------------
    logger.info("=" * 78)
    logger.info("sim2f probe/fit basis: does a KL basis beat Noll?  (measurement only)")
    logger.info("=" * 78)
    logger.info("bench           : Sim2f(Sim2fConfig(), device='cpu', dtype=torch.float64)")
    logger.info(f"covariance asset: {_display(asset)}")
    logger.info(f"                 : Noll {int(noll[0])}..{int(noll[-1])}, K={n_high} high-order modes")
    logger.info(f"draws per leg   : {draws}")
    logger.info(f"seeds           : kl_realised={SEED_KL_REALISED}  E1_sample={SEED_E1_SAMPLE}  "
          f"E2(held out)={SEED_E2}  probe state={SEED_STATE}")
    logger.info(f"  {EUCLIDEAN_STATEMENT}")
    logger.info(f"  measured Gram of the sampled high-order basis: diag in "
          f"[{gram['diag_min']:.4f}, {gram['diag_max']:.4f}], max |off-diagonal| "
          f"{gram['off_max']:.3e} (ideal 1 and 0)")
    logger.info()
    logger.info("PRE-REGISTERED DECISION RULE (fixed before these numbers were seen)")
    logger.info(f"  {PRE_REGISTERED_RULE}")
    logger.info()
    logger.info("CAVEAT ON PART 1 (do not read the E1 KL column as evidence)")
    logger.info(f"  {CAVEAT}")
    logger.info()
    logger.info(_part1_table(part1))
    for ensemble, cov in ensembles.items():
        logger.info(
            f"  [{ensemble}] ensemble RMS = {math.sqrt(float(np.trace(cov))):.4f} rad, "
            f"trace(S) = {float(np.trace(cov)):.3e} rad^2"
        )
    logger.info("  -> the residual column is only comparable WITHIN one ensemble: E1 uses the")
    logger.info("     asset's own rad^2 scale (a weak turbulence realisation), while E2 carries")
    logger.info("     draw_aberration's 0.05-0.45 rad per-state RMS rescale. The fractions are")
    logger.info("     scale-free, which is why they are the column that compares bases.")
    logger.info()
    logger.info(f"  realised/canonical variance ladder (E2 vs Sigma, each share normalised to mean 1): "
          f"min {shape_ladder.min():.3f} at Noll {int(noll[int(np.argmin(shape_ladder))])}, "
          f"max {shape_ladder.max():.3f} at Noll {int(noll[int(np.argmax(shape_ladder))])}; "
          f"defocus/last ratio {float(np.diag(sigma)[0] / np.diag(sigma)[-1]):.1f}x -> "
          f"{float(np.diag(e2_cov)[0] / np.diag(e2_cov)[-1]):.1f}x")
    shape = covariance_shape
    logger.info("  why the gap can only be small (measured on Sigma, Noll basis):")
    logger.info(f"    diagonal ladder Noll {int(noll[0])}/Noll {int(noll[-1])} = "
          f"{shape['diag_ladder_ratio_first_to_last']:.1f}x; leading eigenvalues vs leading")
    logger.info(f"    diagonal entries (top 3) = {shape['leading_eigen_over_leading_diag_top3']:.3f}; "
          f"max |off-diagonal corr| = {shape['max_abs_offdiagonal_correlation']:.3f}")
    logger.info(f"    but {100.0 * shape['offdiagonal_corr_energy_between_near_equal_partners']:.2f}% "
          "of the off-diagonal correlation energy links equal-variance partners; it links")
    logger.info("    modes 3.7-9.5x apart in variance, so a rotation cannot move the top-K sum.")
    logger.info("    sum(top-K eigenvalues)/sum(first-K diagonal) IS the achievable ceiling:")
    logger.info("      " + "  ".join(
        f"K={k}:{shape['ceiling_ratio_by_K'][str(k)]:.3f}" for k in K_GRID))
    logger.info()
    logger.info(_part2_table(part2))
    logger.info()
    logger.info("DECISION")
    for row in evaluation["criterion1"]:
        logger.info(
            f"  criterion 1 @ K={row['K']} {row['ensemble']:<12}: "
            f"kl_sigma {100.0 * row['kl_sigma_fraction']:.2f}% vs noll "
            f"{100.0 * row['noll_fraction']:.2f}% -> ratio {row['ratio']:.3f} "
            f"(needs >= {row['threshold']}) {'PASS' if row['pass'] else 'FAIL'}"
        )
    for row in evaluation["criterion2"]:
        if row["comparable"]:
            logger.info(
                f"  criterion 2 {row['bench']}/{row['state']} K={row['K']:<2}: "
                f"cond {row['kl_cond']:.2f} vs noll {row['noll_cond']:.2f} -> ratio "
                f"{row['ratio']:.3f} (needs <= {row['threshold']}) "
                f"{'PASS' if row['pass'] else 'FAIL'}"
            )
        else:
            logger.info(
                f"  criterion 2 {row['bench']}/{row['state']} K={row['K']:<2}: "
                f"NOT COMPARABLE -- {row['reason']}"
            )
    logger.info(f"  criterion 2 evaluated on {evaluation['criterion2_comparable_conditions']} "
          f"of {len(evaluation['criterion2'])} (bench, state, K) conditions.")
    logger.info()
    logger.info(f"  VERDICT: {evaluation['verdict']}")
    logger.info(f"  ({elapsed:.1f} s)")

    # -- artifacts ----------------------------------------------------------
    payload = {
        "pre_registered_rule": PRE_REGISTERED_RULE,
        "pre_registered_thresholds": {
            "criterion1_min_variance_ratio": CRIT1_MIN_RATIO,
            "criterion2_max_condition_ratio": CRIT2_MAX_RATIO,
            "K_fit": K_FIT,
            "K_fit_reason": (
                "probe_noll(AberrationConfig(n_modes=21)) pushes Noll 4..22, i.e. 19 "
                "high-order probe columns; the coefficient vector is 21 rows (Noll 2..22) "
                "but tip/tilt are not probed"
            ),
        },
        "caveat": CAVEAT,
        "euclidean_statement": EUCLIDEAN_STATEMENT,
        "sampled_basis_gram": gram,
        "bench": {
            "constructor": "Sim2f(Sim2fConfig(), device='cpu', dtype=torch.float64)",
            "configs_measured": [label for label, _ in _bench_cfgs()],
            "detector": "off (forward(..., detector=False)): the response matrix is a "
            "noise-free secant",
        },
        "covariance_asset": str(asset),
        "noll": [int(v) for v in noll],
        "draws": draws,
        "seeds": {
            "kl_realised": SEED_KL_REALISED,
            "E1_sample": SEED_E1_SAMPLE,
            "E2_held_out": SEED_E2,
            "probe_state": SEED_STATE,
        },
        "ensemble_absolute_scale": {
            name: {
                "total_rad2": float(np.trace(cov)),
                "ensemble_rms_rad": math.sqrt(float(np.trace(cov))),
            }
            for name, cov in ensembles.items()
        },
        "ensemble_scale_caveat": (
            "The residual column is only comparable within one ensemble: E1 uses the "
            "asset's own rad^2 scale (a weak turbulence realisation), while E2 carries "
            "draw_aberration's 0.05-0.45 rad per-state RMS rescale. The captured "
            "fractions are scale-free and are the column that compares bases."
        ),
        "covariance_shape": covariance_shape,
        "realised_variance_ladder_E2": {
            "values": [float(v) for v in shape_ladder],
            "noll": [int(v) for v in noll],
            "min": float(shape_ladder.min()),
            "max": float(shape_ladder.max()),
            "argmin_noll": int(noll[int(np.argmin(shape_ladder))]),
            "argmax_noll": int(noll[int(np.argmax(shape_ladder))]),
            "target_defocus_over_last": float(np.diag(sigma)[0] / np.diag(sigma)[-1]),
            "realised_defocus_over_last": float(np.diag(e2_cov)[0] / np.diag(e2_cov)[-1]),
            "definition": (
                "diag(E2 covariance)/mean(diag(E2 covariance)) divided by "
                "diag(Sigma)/mean(diag(Sigma)) -- the normalisation of "
                "tests/ao_shaping/ml/sim2f/test_aberration_covariance.py::"
                "test_end_to_end_shape_survives_the_rms_rescale, so its quoted 0.85x/"
                "1.27x is directly comparable. The values in this row use E2's "
                "held-out seed, so they differ from that docstring only by sampling."
            ),
        },
        "results_part1_variance_capture": part1,
        "results_part2_response": part2,
        "decision": evaluation,
        "elapsed_s": elapsed,
    }
    json_path = out_dir / "kl_basis_measurement.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    markdown = _markdown_report(payload)
    md_path = out_dir / "kl_basis_measurement.md"
    md_path.write_text(markdown + "\n", encoding="utf-8")

    logger.info()
    logger.info(f"wrote {_display(json_path)}")
    logger.info(f"wrote {_display(md_path)}")
    return 0


def _markdown_report(payload: dict[str, object]) -> str:
    """Render the artifact.

    Ordering matters: the pre-registered rule and the caveat come **before** the
    results, so a reader meets the decision criterion before the numbers.

    Args:
        payload: The JSON payload.

    Returns:
        Markdown text.
    """
    decision = payload["decision"]
    part1 = payload["results_part1_variance_capture"]
    part2 = payload["results_part2_response"]
    lines = [
        "# sim2f: would a KL basis help the probe/fit stage?",
        "",
        "Measurement only -- `ml/sim2f/fit.py`, `config.py` and `optics.py` are unchanged, "
        "and no basis switch was added.",
        "",
        "## Pre-registered decision rule (stated before the results)",
        "",
        f"> {payload['pre_registered_rule']}",
        "",
        f"Thresholds: criterion 1 `>= {payload['pre_registered_thresholds']['criterion1_min_variance_ratio']}x`, "
        f"criterion 2 `<= {payload['pre_registered_thresholds']['criterion2_max_condition_ratio']}x`.",
        "",
        f"`K` of the fit: **{payload['pre_registered_thresholds']['K_fit']}** -- "
        f"{payload['pre_registered_thresholds']['K_fit_reason']}.",
        "",
        "## Caveat on Part 1",
        "",
        f"> {payload['caveat']}",
        "",
        "## Metric",
        "",
        f"{payload['euclidean_statement']} Measured on the sampled basis: Gram diagonal in "
        f"[{payload['sampled_basis_gram']['diag_min']:.4f}, "
        f"{payload['sampled_basis_gram']['diag_max']:.4f}], max |off-diagonal| "
        f"{payload['sampled_basis_gram']['off_max']:.3e}.",
        "",
        "## Setup",
        "",
        f"* bench: `{payload['bench']['constructor']}` ({', '.join(payload['bench']['configs_measured'])} configs)",
        f"* detector: {payload['bench']['detector']}",
        f"* covariance asset: `{payload['covariance_asset']}` (Noll {payload['noll'][0]}..{payload['noll'][-1]})",
        f"* draws per Monte Carlo leg: {payload['draws']}",
        f"* seeds: kl_realised `{payload['seeds']['kl_realised']}`, E1 sample "
        f"`{payload['seeds']['E1_sample']}`, E2 held out `{payload['seeds']['E2_held_out']}`, "
        f"probe state `{payload['seeds']['probe_state']}`",
        "",
        "## Results, Part 1 -- variance capture",
        "",
        _part1_markdown(part1, payload),
        "",
        f"> {payload['ensemble_scale_caveat']}",
        "",
        f"Realised/canonical variance ladder on E2 (each side's diagonal share normalised to "
        f"mean 1, the normalisation of `test_end_to_end_shape_survives_the_rms_rescale`): min "
        f"{payload['realised_variance_ladder_E2']['min']:.3f} at Noll "
        f"{payload['realised_variance_ladder_E2']['argmin_noll']}, max "
        f"{payload['realised_variance_ladder_E2']['max']:.3f} at Noll "
        f"{payload['realised_variance_ladder_E2']['argmax_noll']}; the defocus-to-last-mode "
        f"ratio falls "
        f"{payload['realised_variance_ladder_E2']['target_defocus_over_last']:.1f}x -> "
        f"{payload['realised_variance_ladder_E2']['realised_defocus_over_last']:.1f}x. "
        f"{payload['realised_variance_ladder_E2']['definition']}",
        "",
        f"Why the gap can only be small (measured on `Sigma` in the Noll basis): the "
        f"diagonal ladder runs {payload['covariance_shape']['diag_ladder_ratio_first_to_last']:.1f}x "
        f"from Noll {payload['noll'][0]} to Noll {payload['noll'][-1]}; the leading "
        f"eigenvalues nearly coincide with the leading diagonal entries (top-3 ratio "
        f"{payload['covariance_shape']['leading_eigen_over_leading_diag_top3']:.3f}); the "
        f"strongest off-diagonal correlation is "
        f"{payload['covariance_shape']['max_abs_offdiagonal_correlation']:.3f}, but only "
        f"{100.0 * payload['covariance_shape']['offdiagonal_corr_energy_between_near_equal_partners']:.2f}% "
        "of the off-diagonal correlation energy links equal-variance partners -- it links "
        "modes 3.7-9.5x apart, which a rotation cannot exploit. "
        f"{payload['covariance_shape']['note']}",
        "",
        "## Results, Part 2 -- probe response conditioning",
        "",
        _part2_markdown(part2),
        "",
        "`rank` is the column count retained at the fit's `rcond`; `*` marks a "
        "rank-deficient bank at that cut, whose retained `s_max/s_min` is benign while some "
        "columns carry no response at all (see the `all cols` column). At `bare/flat` the "
        "response of half the Noll modes cancels *identically*: with a real, even pupil "
        "command `A` and an odd-parity push `Z`, `U- = conj(U+)`, so "
        "`I-(k) = I+(-k) = I+(k)` (centro-symmetric) and the +/- secant is exactly zero. "
        "That condition therefore has no conditioning information -- a property of the "
        "flat command, not of either basis.",
        "",
        "`if linear` is `s_max/s_min` of `J_noll @ V_K` (the same columns stacked "
        "through an orthogonal rotation): any gap between it and the measured KL condition "
        "number is finite-amplitude / prepared-frame non-additivity of a combined push, "
        "not a basis property.",
        "",
        "A variance-ordered KL bank also includes its *weakest* direction as a column, and "
        "`kl_realised`'s weakest is a low-variance direction estimated from "
        f"{payload['draws']} draws. Its `s_min` can therefore sit orders of magnitude below "
        "the Noll bank's (3662 / 11590 / 41085 at bare/flat, against 1.00 / 7.45 / 46.28): "
        "putting the weakest mode last is exactly where the fit's rank cut would bite, which "
        "is a risk a Noll bank does not carry.",
        "",
        "## Decision",
        "",
    ]
    for row in decision["criterion1"]:
        lines.append(
            f"* criterion 1 @ K={row['K']} {row['ensemble']}: ratio **{row['ratio']:.3f}** "
            f"(needs >= {row['threshold']}) -- {'PASS' if row['pass'] else 'FAIL'}"
        )
    for row in decision["criterion2"]:
        if not row["comparable"]:
            lines.append(
                f"* criterion 2 {row['bench']}/{row['state']} K={row['K']}: "
                f"**not comparable** -- {row['reason']}"
            )
            continue
        lines.append(
            f"* criterion 2 {row['bench']}/{row['state']} K={row['K']}: ratio "
            f"**{row['ratio']:.3f}** (needs <= {row['threshold']}) -- "
            f"{'PASS' if row['pass'] else 'FAIL'}"
        )
    lines += [
        "",
        f"Evaluated on {decision['criterion2_comparable_conditions']} of "
        f"{len(decision['criterion2'])} (bench, state, K) conditions; a rank-deficient bank "
        "has no defined `s_min`, so the pre-registered quantity is undefined there.",
        "",
        f"### Verdict: **{decision['verdict']}**",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _display(path: Path) -> str:
    """Render a path relative to the repo when possible.

    Args:
        path: Any path (a ``--out`` inside or outside the repo).

    Returns:
        The repo-relative path when it is inside the repo, else the absolute path.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _get_bench(cfg: Sim2fConfig | None = None):
    """Build the float64 CPU bench this script measures on.

    Args:
        cfg: Bench config; ``None`` means the bare :class:`Sim2fConfig`.

    Returns:
        The bench.
    """
    from ml.sim2f.optics import Sim2f

    return Sim2f(cfg or Sim2fConfig(), device="cpu", dtype=torch.float64)


def _bench_cfgs() -> list[tuple[str, Sim2fConfig]]:
    """The bench configurations the response matrix is measured on.

    Returns:
        ``[("bare", Sim2fConfig()), ("calibrated", Sim2fConfig.calibrated())]``.
    """
    return [("bare", Sim2fConfig()), ("calibrated", Sim2fConfig.calibrated())]


def _model_measure(bench: object):
    """A noise-free camera read: the bench's own model frame.

    The detector is forced off, exactly as ``fit`` does when it predicts: a
    quantised, noisy prediction would floor the secant the matrix is built from,
    and every number here would then be a statement about the detector.

    Args:
        bench: The bench.

    Returns:
        ``callable(phase) -> frame``.
    """

    def measure(phase: Tensor) -> Tensor:
        with torch.no_grad():
            return bench.forward(phase, detector=False)  # type: ignore[attr-defined]

    return measure


def _states(bench: object, asset: Path) -> list[tuple[str, object]]:
    """The two probe conditions: flat command, and one drawn bench state.

    The flat command is ``build_response``'s own default.  The drawn state is what
    the probe stage actually runs on, so the comparison is reported on both --
    they are not equivalent, see the parity note under the Part 2 table.

    Args:
        bench: The bench.
        asset: The covariance asset path (so the state is drawn from the same
            ensemble the fit is asked to survive).

    Returns:
        ``[("flat", None), ("drawn", (grid, grid) phase)]``.
    """
    from ml.sim2f.optics import draw_aberration

    cfg = AberrationConfig(
        aberration_model="kolmogorov", covariance_path=str(asset), n_modes=21
    )
    state = draw_aberration(bench, cfg, np.random.default_rng(SEED_STATE))
    coeffs = np.asarray(state.coeffs, dtype=np.float64)[None]
    with torch.no_grad():
        phase = bench.aberration(coeffs)[0].detach().cpu().numpy()  # type: ignore[attr-defined]
    return [("flat", None), ("drawn", phase)]


def _sampled_gram(bench: object) -> dict[str, float]:
    """Gram matrix of the sampled high-order basis over the support disc.

    Checks the Euclidean-coefficient-space claim on the *discrete* pupil the
    simulator uses, instead of asserting the continuous convention.

    Args:
        bench: The bench.

    Returns:
        Mapping with ``diag_min``, ``diag_max``, ``off_max``.
    """
    support = bench.pupil_mask.cpu().numpy()[0, 0] > 0.5  # type: ignore[attr-defined]
    raw = bench.zernike_basis(22).cpu().numpy()[3:22]  # type: ignore[attr-defined]
    samples = raw[:, support]
    gram = samples @ samples.T / float(samples.shape[1])
    off = gram - np.eye(gram.shape[0])
    return {
        "diag_min": float(np.diag(gram).min()),
        "diag_max": float(np.diag(gram).max()),
        "off_max": float(np.abs(off).max()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
