"""Closed-loop ablation on the calibrated twin: five arms, per-family numbers.

The five arms, exactly
======================
``flat``        zero phase -- the bench's own spot, the baseline to beat.
``net_open``    the U-Net phase, no aberration correction.  This is the
                "amortisation only" arm: it is what the network can do when
                nobody measures the optical path.
``net_probe``   ``net - sum_j c_hat_j Z_j`` with ``c_hat`` from the pre-loop probe
                stage (:func:`ml.sim2f.fit.probe`).
``net_closed``  the same, then ``LoopConfig.rounds`` rounds of measure ->
                :func:`ml.sim2f.fit.refine` -> re-command.
``oracle``      ``net - sum_j c_true_j Z_j``: the ceiling, exact knowledge of the
                bench state.  It still sits on the aberrated bench with the same
                detector, so it is the best *this* bench allows, not an ideal
                bench.

Every arm is measured on the same bench, in the same bench state, and scored with
:func:`ml.sim2f.losses.objective_terms` (the teacher's own ``L_shape + w_ee *
(1 - EE)``) plus :func:`ml.sim2f.losses.shaping_metrics`, on **prepared** frames.
Samples are aberration states: a target is drawn from the corpus and paired with a
state from :func:`ml.sim2f.optics.draw_aberration`, so the network is evaluated on
the targets it was trained for but on an optical path it never saw.

What this can and cannot claim
==============================
Teacher, network and loop share one forward model, so this is an *amortisation* +
*closed-loop estimation* measurement, never a sim-to-real result.  A matched model
also means the estimator could recover the state from a single frame by gradient
descent; the probe bank's contribution is the observability evidence (rank,
condition) plus the linear step, not a model the fit could not otherwise have.

The arms are not tuned to make ``net_closed >= net_probe >= net_open >= flat``
come out.  Where the measurement disagrees with that ordering, the disagreement is
what the report shows; the two known causes, both measured here, are

* the calibrated detector at its fitted exposure (0.15 ms) puts the frame below
  the fixed-pattern noise floor, so the estimate is the noise direction
  (:func:`ml.sim2f.fit.probe` returns a documented failure there and the arms
  collapse to ``net_open``);
* the aberration state's **illumination** jitter is not in the estimated
  coefficient vector and is nearly degenerate with tip/tilt, so at the default
  ``illum_jitter_frac = 0.06`` the probe estimate is dominated by that unmodelled
  term.

Both are reported rather than hidden: ``--exposure-ms`` and ``--illum-jitter`` are
bench/state settings, and the JSON records which ones produced the table.

Determinism
===========
Same seed => same numbers.  Every random draw (aberration state, detector noise)
is seeded from ``LoopConfig.seed`` and the sample index; no RNG is taken from the
global stream.  Each arm gets its own noise realisation (a *different* seed), so
per-arm differences carry an independent noise floor -- the honest choice, since a
shaped frame and a flat frame cannot share a noise field meaningfully.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from ml.sim2f.config import AberrationConfig, LoopConfig, Sim2fConfig, TeacherConfig
from ml.sim2f._device import _resolve_device
from ml.sim2f.fit import (
    DEFAULT_LADDER_EXPOSURES,
    FitConfig,
    FitResult,
    noise_ladder,
    probe,
    refine,
)
from ml.sim2f.losses import from_phasor
from ml.sim2f.model import build_unet, network_input
from ml.sim2f.optics import Sim2f, draw_aberration, write_aberration

__all__ = ["ARMS", "LOOP_NAME", "LOOP_CSV", "build_parser", "format_arms", "main", "run_loop"]

#: Arm labels, in report order.
ARMS: tuple[str, ...] = ("flat", "net_open", "net_probe", "net_closed", "oracle")

#: Report files written into ``LoopConfig.out_dir``.
LOOP_NAME = "eval_loop.json"
LOOP_CSV = "eval_loop.csv"

#: Metric columns of the aggregate tables.  The first four exist for every row;
#: the rest are :func:`ml.sim2f.losses.shaping_metrics` diagnostics.
METRIC_COLUMNS: tuple[str, ...] = (
    "objective",
    "l1",
    "ncc",
    "ee",
    "cv_in_support",
    "iou50",
    "peak_ratio",
)

#: Geometry fields the corpus and the loop bench must agree on.  A mismatch makes
#: the numbers incomparable rather than merely noisy (the network's phase map is
#: `grid` wide and its target is `ccd_px` wide), so it raises instead of warning.
_GEOMETRY_FIELDS: tuple[str, ...] = (
    "grid",
    "ccd_px",
    "far_field_padding",
    "pupil_fill",
    "region_span_px",
    "panel_pixel_um",
    "ccd_pixel_um",
    "wavelength_m",
    "focal_length_m",
    "beam_w0_px",
    "phase_blur_px",
)


# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------
def _check_geometry(label: Sim2fConfig, bench: Sim2fConfig) -> None:
    """Require the loop bench to share the corpus' geometry.

    Args:
        label: The bench the labels / network were made on (from the cache).
        bench: The bench the loop measures on.

    Raises:
        ValueError: If any geometry field differs.  Optical *errors* and the
            detector are expected to differ -- that is the point of the loop --
            but a different grid or angular scale is not comparable.
    """
    for name in _GEOMETRY_FIELDS:
        left = getattr(label, name)
        right = getattr(bench, name)
        same = (
            int(left) == int(right)
            if isinstance(left, int)
            else math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=0.0)
        )
        if not same:
            raise ValueError(
                f"loop bench disagrees with the corpus on {name}: {right!r} vs "
                f"{left!r}. Only the aberration/detector block may differ; build the "
                "bench with dataclasses.replace(Sim2fConfig.calibrated(), ...) so the "
                "geometry stays the corpus'."
            )


def _load_network(
    checkpoint: str | Path, device: torch.device
) -> tuple[torch.nn.Module, Tensor, Sim2fConfig, dict[str, Any]]:
    """Rebuild the U-Net from a checkpoint.

    ``evaluate._config_from_checkpoint`` is the package's single checkpoint
    reader, so the loop cannot disagree with ``eval`` about which bench or which
    PSF context a checkpoint carries.

    Args:
        checkpoint: Path to ``model_best.pt`` / ``model_last.pt``.
        device: Device to build on.

    Returns:
        ``(model, context, label_sim_cfg, raw_payload)``.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
    """
    from ml.sim2f.evaluate import _config_from_checkpoint

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    train_cfg, sim_cfg, context = _config_from_checkpoint(payload)
    context = context.to(device=device, dtype=torch.float32)
    model = build_unet(train_cfg, sim_cfg, device=device, context=context)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, context, sim_cfg, payload


class _Measurer:
    """Push a phase, return a camera frame -- with the bench state applied.

    The state lives in the closure, never in the estimator: that is the whole
    contract of a camera read, and it is what lets the same ``probe`` run against
    a stub bench in the tests.

    Attributes:
        count: Number of frames read so far (the noise stream is derived from it).
    """

    def __init__(
        self,
        bench: Sim2f,
        aberration: Tensor | None,
        illumination: Tensor | None,
        *,
        detector: bool,
        seed: int,
    ) -> None:
        """Store the bench state and the noise seed.

        Args:
            bench: The bench.
            aberration: ``(1, grid, grid)`` true aberration phase, radians.
            illumination: ``(1, 1, grid, grid)`` entrance amplitude.
            detector: Whether frames go through the detector model.
            seed: Base noise seed; frame ``n`` uses ``seed + n``.
        """
        self.bench = bench
        self.aberration = aberration
        self.illumination = illumination
        self.detector = bool(detector)
        self.seed = int(seed)
        self.count = 0

    def __call__(self, phase: Tensor) -> Tensor:
        """Read one frame.

        Args:
            phase: ``(1, grid, grid)`` command, radians.

        Returns:
            ``(1, 1, ccd_px, ccd_px)`` frame (counts when the detector is on).
        """
        generator = None
        if self.detector:
            generator = torch.Generator(device=self.bench.device).manual_seed(
                self.seed + self.count
            )
        self.count += 1
        with torch.no_grad():
            return self.bench.forward(
                phase,
                aberration=self.aberration,
                illumination=self.illumination,
                detector=self.detector,
                generator=generator,
            )


def _command(bench: Sim2f, net_phase: Tensor, coeffs: np.ndarray) -> Tensor:
    """Form the loop's command: network phase minus the estimated aberration.

    Args:
        bench: The bench (its basis defines the map).
        net_phase: ``(1, grid, grid)`` network phase, radians.
        coeffs: ``(n_modes,)`` estimate, radians (all zeros => no correction).

    Returns:
        ``(1, grid, grid)`` command phase, radians.  The bench *adds* the true
        aberration, so subtracting the estimate is what cancels it.
    """
    values = np.asarray(coeffs, dtype=np.float32).reshape(-1)
    if not np.any(values):
        return net_phase
    delta = bench.aberration(values[None])
    return net_phase - delta


def _arm_seed(seed: int, sample: int, tag: str) -> int:
    """Deterministic noise seed for one (sample, arm) measurement stream.

    Args:
        seed: ``LoopConfig.seed``.
        sample: Sample index in cache order.
        tag: ``"probe"`` or an arm name.

    Returns:
        A non-negative integer seed.
    """
    position = {"probe": 0, **{name: index + 1 for index, name in enumerate(ARMS)}}[tag]
    return (int(seed) * 7919 + int(sample) * 104729 + position * 1543) % (2**31 - 1)


def _score(
    frame: Tensor,
    target: Tensor,
    support: Tensor,
    *,
    teacher: TeacherConfig,
) -> dict[str, float]:
    """Score one measured frame with the teacher's own objective and metrics.

    The composition lives in ``evaluate._row``: importing it keeps this table and
    ``eval.json`` numerically identical for the same input, which is the only way
    the two reports can be compared at all.

    Args:
        frame: ``(1, 1, ccd_px, ccd_px)`` measured frame.
        target: ``(1, 1, ccd_px, ccd_px)`` requested distribution.
        support: ``(1, 1, ccd_px, ccd_px)`` analytic support.
        teacher: Supplies ``w_ee`` and ``metric``.

    Returns:
        The metric row, including ``objective``.
    """
    from ml.sim2f.evaluate import _row

    return _row(
        frame,
        target,
        support,
        w_ee=teacher.w_ee,
        metric=teacher.metric,
        prepare=True,
    )


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """Column-wise mean of metric rows.

    Args:
        rows: Non-empty list of rows with identical keys.

    Returns:
        The mean of each column.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("cannot average an empty list of metric rows")
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def _cosine(left: np.ndarray, right: np.ndarray, rows: tuple[int, ...]) -> float:
    """Cosine similarity of two coefficient vectors, restricted to ``rows``.

    Args:
        left: ``(n_modes,)`` first vector.
        right: ``(n_modes,)`` second vector.
        rows: 0-based row positions to compare (the fitted / refit set).

    Returns:
        The cosine, or ``0.0`` when either side is zero.
    """
    a = np.array([left[row] for row in rows], dtype=np.float64)
    b = np.array([right[row] for row in rows], dtype=np.float64)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / norm) if norm > 0.0 else 0.0


def _coefficient_error(estimate: np.ndarray, truth: np.ndarray, rows: tuple[int, ...]) -> dict[str, float]:
    """RMS / max coefficient error and cosine similarity over ``rows``.

    Args:
        estimate: ``(n_modes,)`` estimate, radians.
        truth: ``(n_modes,)`` truth, radians.
        rows: 0-based row positions to compare.

    Returns:
        ``{"coeff_rmse", "coeff_max", "cos"}``.
    """
    error = np.array([estimate[row] - truth[row] for row in rows], dtype=np.float64)
    return {
        "coeff_rmse": float(np.sqrt(np.mean(error**2))) if error.size else 0.0,
        "coeff_max": float(np.abs(error).max()) if error.size else 0.0,
        "cos": _cosine(estimate, truth, rows),
    }


def _finite(value: Any) -> Any:
    """Replace non-finite floats with ``None`` (standards-compliant JSON).

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


# ----------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------
def run_loop(
    cfg: LoopConfig,
    *,
    cache_dir: str | Path,
    checkpoint: str | Path,
    bench_cfg: Sim2fConfig | None = None,
    aberration_cfg: AberrationConfig | None = None,
    device: str | torch.device = "auto",
    out: str | Path | None = None,
    ladder_exposures: tuple[float | None, ...] | None = DEFAULT_LADDER_EXPOSURES,
) -> dict[str, Any]:
    """Run the five arms on every sampled state and write the report.

    Args:
        cfg: Loop contract (sample count, rounds, seeds, output directory).
        cache_dir: Corpus directory (``pairs.npz`` + ``meta.json``).
        checkpoint: Network checkpoint trained on that corpus.
        bench_cfg: Bench to measure on.  Defaults to
            :meth:`ml.sim2f.config.Sim2fConfig.calibrated`; only the
            aberration / detector block may differ from the corpus, the geometry
            must match (:func:`_check_geometry`).
        aberration_cfg: Distribution the states are drawn from.  Defaults to
            :class:`~ml.sim2f.config.AberrationConfig`.
        device: ``"auto"``, ``"cuda"``, ``"cpu"`` or a device.
        out: Report directory; defaults to ``cfg.out_dir``.
        ladder_exposures: Exposures for the fit-error noise ladder (``None``
            entries are the detector-less bench).  ``None`` skips the ladder.

    Returns:
        The report payload (also written to ``eval_loop.json``).

    Raises:
        FileNotFoundError: If the corpus or the checkpoint is missing.
        ValueError: If the bench geometry disagrees with the corpus, or the
            corpus is empty.
    """
    from ml.sim2f.train import _apply_determinism, load_corpus, seed_everything

    resolved = _resolve_device(device)
    seed_everything(int(cfg.seed))
    _apply_determinism(True)

    corpus = load_corpus(cache_dir)
    model, context, label_cfg, _payload = _load_network(checkpoint, resolved)
    bench_config = Sim2fConfig.calibrated() if bench_cfg is None else bench_cfg
    _check_geometry(label_cfg, bench_config)
    bench = Sim2f(bench_config, device=str(resolved))
    states = AberrationConfig() if aberration_cfg is None else aberration_cfg
    fit_cfg = FitConfig.from_aberration(states)
    teacher = corpus.teacher_cfg
    refit_rows = tuple(
        j - 2 for j in range(2, int(states.n_modes) + 2) if j not in set(states.fit_frozen)
    )

    available = corpus.n_samples
    wanted = min(int(cfg.samples), available)
    generator = torch.Generator().manual_seed(int(cfg.seed))
    order = torch.randperm(available, generator=generator)[:wanted].tolist()
    if wanted < int(cfg.samples):
        logger.info(
            "requested {} sample(s) but the corpus holds {}; running {}",
            cfg.samples, available, wanted,
        )

    logger.info(
        "loop: {} sample(s) x {} arm(s), rounds={}, probe={}, refit={}, detector={}, "
        "bench(exposure={} ms, defocus={} m, eta={}), illum_jitter={}",
        wanted, len(ARMS), cfg.rounds, cfg.probe_fit, cfg.refit, cfg.use_detector,
        bench_config.exposure_ms, bench_config.errors.defocus_m,
        bench_config.errors.zero_order_eta, states.illum_jitter_frac,
    )

    grid = bench.grid
    zeros = torch.zeros(1, grid, grid, device=resolved)
    per_sample: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    truth_for_ladder: np.ndarray | None = None

    for position, index in enumerate(order):
        kind = corpus.kinds()[index]
        target = corpus.target[index : index + 1].to(device=resolved, dtype=torch.float32)
        support = corpus.support[index : index + 1].to(device=resolved, dtype=torch.float32)

        rng = np.random.default_rng([int(cfg.seed), int(index)])
        state = draw_aberration(bench, states, rng)
        truth = np.asarray(state.coeffs, dtype=np.float64)
        if truth_for_ladder is None:
            truth_for_ladder = truth
        aberration, illumination = write_aberration(bench, state)

        # One camera per arm, each with its own noise stream: a shaped frame and
        # a flat frame cannot share a noise field meaningfully, so the arms are
        # independent draws rather than common random numbers.
        readers = {
            tag: _Measurer(
                bench,
                aberration,
                illumination,
                detector=cfg.use_detector,
                seed=_arm_seed(cfg.seed, index, tag),
            )
            for tag in ("flat", "net_open", "probe", "net_probe", "net_closed", "oracle")
        }

        with torch.no_grad():
            net_phase = from_phasor(model(network_input(target, context)))

        # ``net - sum c_j Z_j``, bound to this sample's network phase: the loop's
        # re-commanding step, written once so every arm forms it identically.
        command = partial(_command, bench, net_phase)

        arm_rows: dict[str, dict[str, float]] = {}

        # -- flat -----------------------------------------------------------
        flat_frame = readers["flat"](zeros)
        arm_rows["flat"] = _score(flat_frame, target, support, teacher=teacher)

        # -- net_open -------------------------------------------------------
        open_frame = readers["net_open"](net_phase)
        arm_rows["net_open"] = _score(open_frame, target, support, teacher=teacher)

        # -- probe stage ----------------------------------------------------
        if cfg.probe_fit:
            probe_result = probe(
                bench, states, measure=readers["probe"], phase=zeros, fit_cfg=fit_cfg
            )
        else:
            # The "no probe" ablation: nothing is measured, so the vector is zero
            # and `net_probe` collapses onto `net_open` by construction -- the
            # stage's contribution is exactly what those two rows differ by.
            probe_result = FitResult(
                ok=False,
                reason="probe stage disabled (LoopConfig.probe_fit=False)",
                coeffs=np.zeros(int(states.n_modes), dtype=np.float64),
                free_noll=(),
                residual=float("nan"),
                residual_start=float("nan"),
                iterations=0,
                history=(),
            )
        c_probe = probe_result.coeffs if probe_result.ok else np.zeros(int(states.n_modes))
        probe_info = {
            "ok": bool(probe_result.ok),
            "reason": probe_result.reason,
            "rank": int(probe_result.rank),
            "condition": float(probe_result.condition),
            "residual_start": float(probe_result.residual_start),
            "residual": float(probe_result.residual),
            "iterations": int(probe_result.iterations),
            "linear_residual": None
            if probe_result.linear_residual is None
            else float(probe_result.linear_residual),
            "linear_coeff_rmse": float(
                np.sqrt(np.mean((probe_result.linear - truth) ** 2))
            )
            if probe_result.linear is not None
            else None,
            # Bounded, so it stays readable when the one-step inverse is huge:
            # the linear solve is reported as evidence about the bank, not as an
            # estimate (see fit.probe).
            "linear_cos": _cosine(probe_result.linear, truth, refit_rows)
            if probe_result.linear is not None
            else None,
            **_coefficient_error(c_probe, truth, refit_rows),
        }
        if not probe_result.ok:
            logger.warning("sample {}: probe stage failed ({})", index, probe_result.reason)

        # -- net_probe ------------------------------------------------------
        probe_frame = readers["net_probe"](command(c_probe))
        arm_rows["net_probe"] = _score(probe_frame, target, support, teacher=teacher)

        # -- net_closed -----------------------------------------------------
        closed_reader = readers["net_closed"]
        current = c_probe.copy()
        current_phase = command(current)
        current_frame = probe_frame
        rounds: list[dict[str, Any]] = [
            {
                "round": 0,
                "objective": arm_rows["net_probe"]["objective"],
                "l1": arm_rows["net_probe"]["l1"],
                "residual": None,
                "ok": True,
                "reason": None,
                **_coefficient_error(current, truth, refit_rows),
            }
        ]
        for index_round in range(int(cfg.rounds) if cfg.refit else 0):
            result = refine(
                bench, current_phase, current_frame, current, cfg=states, fit_cfg=fit_cfg
            )
            if not result.ok:
                rounds.append(
                    {
                        "round": index_round + 1,
                        "objective": rounds[-1]["objective"],
                        "l1": rounds[-1]["l1"],
                        "residual": float(result.residual),
                        "ok": False,
                        "reason": result.reason,
                        **_coefficient_error(current, truth, refit_rows),
                    }
                )
                logger.info(
                    "sample {} round {}: refit rejected ({})",
                    index, index_round + 1, result.reason,
                )
                break
            current = result.coeffs
            current_phase = command(current)
            current_frame = closed_reader(current_phase)
            row = _score(current_frame, target, support, teacher=teacher)
            rounds.append(
                {
                    "round": index_round + 1,
                    "objective": row["objective"],
                    "l1": row["l1"],
                    "residual": float(result.residual),
                    "ok": True,
                    "reason": None,
                    **_coefficient_error(current, truth, refit_rows),
                }
            )
        arm_rows["net_closed"] = _score(current_frame, target, support, teacher=teacher)
        improvements = [
            float(rounds[step + 1]["objective"]) <= float(rounds[step]["objective"])
            for step in range(len(rounds) - 1)
        ]

        # -- oracle ---------------------------------------------------------
        oracle_frame = readers["oracle"](command(truth))
        arm_rows["oracle"] = _score(oracle_frame, target, support, teacher=teacher)

        for arm in ARMS:
            for column in METRIC_COLUMNS:
                csv_rows.append(
                    {
                        "sample": int(index),
                        "kind": kind,
                        "arm": arm,
                        "round": 0,
                        "metric": column,
                        "value": float(arm_rows[arm][column]),
                    }
                )
        for entry in rounds:
            csv_rows.append(
                {
                    "sample": int(index),
                    "kind": kind,
                    "arm": "net_closed",
                    "round": int(entry["round"]),
                    "metric": "objective",
                    "value": float(entry["objective"]),
                }
            )

        per_sample.append(
            {
                "index": int(index),
                "kind": kind,
                "truth": {
                    "coeffs": truth.tolist(),
                    "tilt_px": state.tilt_px.tolist(),
                    "high_order_rms_rad": float(np.sqrt(np.mean(truth[2:] ** 2)))
                    if truth.size > 2
                    else 0.0,
                    "illum_center_px": list(state.illum_center_px),
                    "illum_width_scale": float(state.illum_width_scale),
                },
                "probe": probe_info,
                "closed": {
                    "coeffs": current.tolist(),
                    "rounds": rounds,
                    "monotone": bool(all(improvements)),
                    "improved_rounds": int(sum(improvements)),
                    "round_count": int(len(rounds) - 1),
                    **_coefficient_error(current, truth, refit_rows),
                },
                "arms": arm_rows,
            }
        )
        logger.info(
            "sample {} kind={} high_rms={:.3f} | flat={:.4f} open={:.4f} probe={:.4f} "
            "closed={:.4f} oracle={:.4f} | probe_ok={} closed_coeff_rmse={:.3f}",
            index,
            kind,
            per_sample[-1]["truth"]["high_order_rms_rad"],
            arm_rows["flat"]["objective"],
            arm_rows["net_open"]["objective"],
            arm_rows["net_probe"]["objective"],
            arm_rows["net_closed"]["objective"],
            arm_rows["oracle"]["objective"],
            probe_result.ok,
            per_sample[-1]["closed"]["coeff_rmse"],
        )

    aggregate = {
        arm: _mean_rows([sample["arms"][arm] for sample in per_sample]) for arm in ARMS
    }
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({sample["kind"] for sample in per_sample}):
        members = [sample for sample in per_sample if sample["kind"] == kind]
        by_kind[kind] = {
            "n": len(members),
            "arms": {arm: _mean_rows([sample["arms"][arm] for sample in members]) for arm in ARMS},
            "closed_coeff_rmse": float(
                sum(sample["closed"]["coeff_rmse"] for sample in members) / len(members)
            ),
        }
    round_total = sum(sample["closed"]["round_count"] for sample in per_sample)
    round_improved = sum(sample["closed"]["improved_rounds"] for sample in per_sample)

    ladder: list[dict[str, Any]] = []
    if ladder_exposures is not None and truth_for_ladder is not None:
        ladder = noise_ladder(
            bench,
            states,
            coeffs=truth_for_ladder,
            exposures=tuple(ladder_exposures),
            seed=int(cfg.seed),
            fit_cfg=fit_cfg,
        )

    payload = {
        "cache": str(Path(cache_dir)),
        "checkpoint": str(Path(checkpoint)),
        "loop": cfg.to_dict(),
        "bench": bench_config.to_dict(),
        "aberration": states.to_dict(),
        "teacher": teacher.to_dict(),
        "n_samples": len(per_sample),
        "arms": list(ARMS),
        "columns": list(METRIC_COLUMNS),
        "aggregate": aggregate,
        "per_kind": by_kind,
        "rounds": {
            "rounds_per_sample": int(cfg.rounds) if cfg.refit else 0,
            "improved": int(round_improved),
            "total": int(round_total),
            "improved_fraction": float(round_improved / round_total) if round_total else None,
            "monotone_samples": int(sum(1 for sample in per_sample if sample["closed"]["monotone"])),
            "samples": len(per_sample),
        },
        "probe": {
            "ok": int(sum(1 for sample in per_sample if sample["probe"]["ok"])),
            "samples": len(per_sample),
            "coeff_rmse": float(
                sum(sample["probe"]["coeff_rmse"] for sample in per_sample) / len(per_sample)
            ),
            "linear_coeff_rmse": float(
                sum(
                    sample["probe"]["linear_coeff_rmse"]
                    for sample in per_sample
                    if sample["probe"]["linear_coeff_rmse"] is not None
                )
                / max(
                    1,
                    sum(
                        1
                        for sample in per_sample
                        if sample["probe"]["linear_coeff_rmse"] is not None
                    ),
                )
            ),
            "linear_cos": float(
                sum(
                    sample["probe"]["linear_cos"]
                    for sample in per_sample
                    if sample["probe"]["linear_cos"] is not None
                )
                / max(
                    1,
                    sum(1 for sample in per_sample if sample["probe"]["linear_cos"] is not None),
                )
            ),
        },
        "noise_ladder": ladder,
        "per_sample": per_sample,
        "notes": (
            "Five arms on one bench state each, scored with the teacher's own "
            "objective on prepared frames. Matched forward model: this is an "
            "amortisation + closed-loop-estimation result, not sim-to-real. "
            "`oracle` uses the true coefficients and is the ceiling of this bench, "
            "not an ideal bench. Per-kind rows are the claim; the cross-kind mean "
            "is not. `net_closed` round 0 is the net_probe frame."
        ),
    }
    payload = _finite(payload)

    out_dir = Path(cfg.out_dir if out is None else out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / LOOP_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / LOOP_CSV, csv_rows)
    logger.info("loop report -> {} ({} + {})", out_dir, LOOP_NAME, LOOP_CSV)
    logger.info("\n{}", format_arms(aggregate))
    for kind, entry in by_kind.items():
        logger.info(
            "kind {:<9} n={:<3} objective flat={:.4f} open={:.4f} probe={:.4f} "
            "closed={:.4f} oracle={:.4f}",
            kind,
            entry["n"],
            entry["arms"]["flat"]["objective"],
            entry["arms"]["net_open"]["objective"],
            entry["arms"]["net_probe"]["objective"],
            entry["arms"]["net_closed"]["objective"],
            entry["arms"]["oracle"]["objective"],
        )
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the long-format per-sample metric table.

    Long format (one row per sample x arm x round x metric) rather than one wide
    row per sample: ``net_closed`` has several rounds and the other arms have one,
    and a wide table would have to leave most of its columns empty.

    Args:
        path: Destination file.
        rows: Rows keyed by ``sample``, ``kind``, ``arm``, ``round``, ``metric``,
            ``value``.
    """
    columns = ("sample", "kind", "arm", "round", "metric", "value")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [row["sample"], row["kind"], row["arm"], row["round"], row["metric"],
                 f"{float(row['value']):.10g}"]
            )


def format_arms(aggregate: dict[str, dict[str, float]]) -> str:
    """Render the per-arm aggregate as a fixed-width table.

    Args:
        aggregate: ``{arm: metric row}``.

    Returns:
        A header plus one line per arm.
    """
    width = max((len(name) for name in aggregate), default=5)
    header = " " * (width + 2) + "  ".join(f"{column:>12}" for column in METRIC_COLUMNS)
    lines = [header]
    for name, row in aggregate.items():
        cells = [
            f"{row[column]:>12.4f}" if column in row else f"{'-':>12}" for column in METRIC_COLUMNS
        ]
        lines.append(f"{name:<{width}}  " + "  ".join(cells))
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the ``loop`` parser, every default read from the dataclasses.

    Returns:
        The configured parser.
    """
    cfg = LoopConfig()
    states = AberrationConfig()
    bench = Sim2fConfig.calibrated()
    parser = argparse.ArgumentParser(
        prog="python -m ml.sim2f loop",
        description=(
            "Run the five-arm closed-loop ablation (flat / net_open / net_probe / "
            "net_closed / oracle) on the calibrated 2f twin."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache", required=True, help="corpus directory (pairs.npz)")
    parser.add_argument("--checkpoint", required=True, help="model_best.pt / model_last.pt")
    parser.add_argument("--samples", type=int, default=cfg.samples)
    parser.add_argument("--rounds", type=int, default=cfg.rounds)
    parser.add_argument(
        "--no-probe", action="store_true", help="skip the probe stage (net_probe == net_open)"
    )
    parser.add_argument(
        "--no-refit", action="store_true", help="skip the closed-loop rounds"
    )
    parser.add_argument(
        "--detector", action="store_true", help="read frames through the detector model"
    )
    parser.add_argument(
        "--exposure-ms",
        type=float,
        default=None,
        help=(
            "override the bench exposure (the calibrated twin uses "
            f"{bench.exposure_ms} ms, which is below the fit's noise floor; the "
            "JSON records what was used)"
        ),
    )
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--out-dir", default=cfg.out_dir)
    parser.add_argument(
        "--illum-jitter",
        type=float,
        default=states.illum_jitter_frac,
        help="illumination centre jitter of the drawn states (not fitted; see the module doc)",
    )
    parser.add_argument(
        "--ladder",
        default=",".join(str(value) for value in DEFAULT_LADDER_EXPOSURES),
        help="comma-separated exposures for the fit-error noise ladder ('off' disables)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` on success, ``2`` on a rejected configuration.
    """
    args = build_parser().parse_args(argv)
    try:
        cfg = LoopConfig(
            samples=args.samples,
            rounds=args.rounds,
            probe_fit=not args.no_probe,
            refit=not args.no_refit,
            use_detector=bool(args.detector),
            seed=args.seed,
            out_dir=args.out_dir,
        )
        states = AberrationConfig(illum_jitter_frac=float(args.illum_jitter))
        bench_cfg = Sim2fConfig.calibrated()
        if args.exposure_ms is not None:
            bench_cfg = replace(bench_cfg, exposure_ms=float(args.exposure_ms))
        if str(args.ladder).strip().lower() in ("off", "none", ""):
            ladder: tuple[float | None, ...] | None = None
        else:
            ladder = tuple(
                None if part.strip().lower() in ("none", "off") else float(part)
                for part in str(args.ladder).split(",")
                if part.strip()
            )
    except ValueError as exc:
        logger.error("invalid configuration: {}", exc)
        return 2
    try:
        run_loop(
            cfg,
            cache_dir=Path(args.cache),
            checkpoint=Path(args.checkpoint),
            bench_cfg=bench_cfg,
            aberration_cfg=states,
            device=args.device,
            ladder_exposures=ladder,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("loop failed: {}", exc)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

