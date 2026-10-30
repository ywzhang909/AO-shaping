"""Physical-forward-model multishape shaping, from a from-zero model to CCD-anchored blocks.

This runner is the single entry point for the flow documented in
``report/shape_trials/square1200/square1200_20261026.md``:

1. **model** -- capture a phase-diverse probe set on the current optical path,
   initialize a *nominal* checkpoint from that capture alone (no historical
   model), fit the physical forward model on it, and export the same optical
   parameters at a finer control resolution (plus a wide-ramp variant whose
   carrier mapping is measured again);
2. **shaping** -- for every requested shape, run the four CCD-anchored
   optimization stages (non-GS seed -> Mono12 -> continuation -> dense), where
   every block is accepted only when the *measured* core CV drops;
3. **replay** -- re-display the archived seed and final phases of every shape
   at a common bit depth and at two exposures, so the stages can be compared
   under identical conditions;
4. **record** -- turn the runs into :class:`~ao_shaping.utils.io.file.Recorder`
   records (one per CCD block: phase, measured far field, phase change, CV
   before/after, acceptance) and save them as CSV + HDF5 + an animated GIF.

The device is opened only by the stage that needs it, and only with
``--execute``; without that flag the runner only prints the plan.

The stages are the production ones -- ``capture`` / ``fit`` / ``geometric_seed``
/ ``validate`` / ``replay_phases`` of :mod:`ao_shaping.tools.slm.transfer` --
called in-process, so there is exactly one implementation of the update loop.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import numpy as np
from loguru import logger

from ao_shaping.tools.slm.transfer import capture as capture_stage
from ao_shaping.tools.slm.transfer import fit as fit_stage
from ao_shaping.tools.slm.transfer import recording, replay_phases, validate as validate_stage
from ao_shaping.tools.slm.transfer.light_guard import OpticalPathInvalid
from ao_shaping.tools.slm.transfer.protocol import derive_replay_exposures
from ao_shaping.tools.slm.transfer.session_record import (
    finish_segment, measured_peak_rates, optical_session_field, record_optical_session)
from ao_shaping.tools.slm.transfer.stages import STAGE_RECIPES, StageRecipe
from ao_shaping.utils.image.target import TRANSFER_SHAPE_CHOICES


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _seed_kind(shape: str, requested: str) -> str:
    """The initial-phase construction to use for a shape.

    ``transport`` is exact for a square and mathematically impossible for the
    others (a separable phase cannot produce a non-separable far field), so
    ``auto`` picks the shape-general non-GS back-projection for everything
    else. ``gs`` stays available as the comparison arm.

    ``plane`` and ``random`` are the **严格冷启动** route: they come from
    ``cold_start.cold_phase``, which reads neither the target nor the forward
    model. They must not be pooled with ``transport`` / ``backprojection`` in a
    report -- the latter two use the fitted optics, so a win there does not show
    that the loop works from nothing.
    """
    if requested != "auto":
        return requested
    return "transport" if shape == "square" else "backprojection"


@dataclass
class _Journal:
    """Append-only stage journal; one line per stage start / end."""

    path: Path
    entries: list[dict] = field(default_factory=list)

    def write(self) -> None:
        self.path.write_text(json.dumps(self.entries, indent=2, default=str), encoding="utf-8")

    def record(self, stage: str, status: str, **extra) -> None:
        from ao_shaping.tools.slm.transfer.capture import clock

        self.entries.append(dict(stage=stage, status=status, **clock(), **extra))
        self.write()


def _validate_args(*, checkpoint: Path, out: Path, zero_xy, shapes, recipe: StageRecipe,
                   initial_phase: str, seed_run: Path | None, device: str, camera: str,
                   cam_id: int, slm_number: int, frames: int, side: float, period: float,
                   window_span: float, zero_radius: float, window_step: float | None = None,
                   blocks: int | None = None, updates: int | None = None,
                   adaptive_exposure: bool = True, profile_timing: bool = False,
                   async_capture_write: bool = False, jacobian_guard: bool = True,
                   jacobian_low: float = 0.5, jacobian_high: float = 0.85,
                   jacobian_floor: float = 0.02, spectral_weight: float = 0.0,
                   spectral_cutoff: float | None = None,
                   objective: str = "cv", weight_schedule: str = "constant") -> SimpleNamespace:
    """Build the argument namespace ``validate.run`` expects."""
    if weight_schedule not in ("constant", "staged"):
        raise ValueError(f"unknown weight schedule {weight_schedule!r}")
    if weight_schedule == "staged" and objective == "cv":
        # The CV-only loss has no energy/shape term to stage, so "staged" would
        # be a silent no-op. Fail instead of pretending the run was scheduled.
        raise ValueError("--weight-schedule staged needs --objective shape or formation; "
                         "the cv objective has nothing to stage")
    return SimpleNamespace(
        checkpoint=checkpoint, out=out, zero_xy=list(zero_xy), target_xy=None,
        period=period, side=side, window_span=window_span,
        window_step=window_step if window_step is not None else recipe.window_step,
        zero_radius=zero_radius, core_margin=recipe.core_margin, meter_target=True,
        bit_depth=recipe.bit_depth, slm_number=slm_number, camera=camera, cam_id=cam_id,
        shapes=list(shapes), blocks=blocks if blocks is not None else recipe.blocks,
        updates=updates if updates is not None else recipe.updates, frames=frames,
        mode="anchored", seed_run=seed_run, seed_policy=recipe.seed_policy,
        initial_phase=initial_phase, random_seed=102626,
        phase_lr=recipe.phase_lr, phase_rms_cap=recipe.phase_rms_cap,
        adaptive_step=True, adaptive_exposure=adaptive_exposure, fixed_exposure_ms=None,
        exploration_temperature=0.0, exploration_budget=0.25, exploration_window=4,
        accept_metric=recipe.accept_metric, objective=objective, abba=False,
        ccd_check=recipe.ccd_check, device=device, execute=True,
        profile_timing=profile_timing, async_capture_write=async_capture_write,
        jacobian_guard=jacobian_guard, jacobian_low=jacobian_low,
        jacobian_high=jacobian_high, jacobian_floor=jacobian_floor,
        spectral_weight=spectral_weight, spectral_cutoff=spectral_cutoff,
        weight_schedule=weight_schedule,
    )


def _plan_dict(*, out: Path, execute: bool, **params: Any) -> dict[str, Any]:
    """The plan printed without ``--execute`` and journalled when a run starts.

    Split out of :func:`run` so :func:`execute_campaign` journals the *same* dict a
    plan-only run prints: the journal's ``plan`` entry is what a later reader uses to
    answer "这条路 lane 到底带了哪些旗标", so the two must never be built twice with
    slightly different key sets.
    """
    plan = dict(out=str(out), shapes=list(params["shapes"]), side=params["side"],
                period=params["period"],
                window_span=params["window_span"], window_step=params["window_step"],
                zero_radius=params["zero_radius"],
                capture_grid=params["capture_grid"], capture_count=params["capture_count"],
                fit_iterations=params["fit_iterations"], control_grid=params["control_grid"],
                fft_size=params["fft_size"],
                initial_phase=params["initial_phase"], seed_bakeoff=params["seed_bakeoff"],
                gs_seed=params["gs_seed"],
                execute=execute, resume=params["resume"], order=params["order"],
                record_responses=params["record_responses"],
                replay_exposures=("auto" if params["replay_exposures_auto"]
                                  else list(params["replay_exposures"])),
                jacobian_guard=params["jacobian_guard"], jacobian_low=params["jacobian_low"],
                jacobian_high=params["jacobian_high"], jacobian_floor=params["jacobian_floor"],
                spectral_weight=params["spectral_weight"],
                spectral_cutoff=params["spectral_cutoff"],
                objective=params["objective"], weight_schedule=params["weight_schedule"],
                stages=[dict(name=recipe.name, blocks=recipe.blocks, updates=recipe.updates,
                             bit_depth=recipe.bit_depth, ccd_check=recipe.ccd_check,
                             window_step=recipe.window_step) for recipe in STAGE_RECIPES])
    # ``validate.session_plan`` prices one shape at a single (bit_depth,
    # window_step, groups) tuple per stage; hand it the recipe table so the
    # printed plan reflects the real stage sizes instead of the runner's own
    # --window-span/--blocks defaults. Only the resume/order path needs it: the
    # historical flag-only plan stays byte-identical.
    if plan["resume"] or plan["order"] != "shape":
        from types import SimpleNamespace as _Namespace

        from ao_shaping.tools.slm.transfer.validate import session_plan

        plan["session_plan"] = session_plan(_Namespace(
            out=out, shapes=list(params["shapes"]), frames=params["frames"],
            session_groups=[dict(bit_depth=recipe.bit_depth, window_step=recipe.window_step,
                                 groups=recipe.blocks) for recipe in STAGE_RECIPES]))
    return plan


def execute_campaign(*, out: Path, should_stop: Callable[[], bool] | None = None,
                     **kwargs: Any) -> dict[str, Any]:
    """Run the *executing* half of :func:`run` — this one opens the device.

    Why it is a function of its own: the GUI campaign page runs it in a background
    thread (``Thread(target=...)`` bodies may not touch ``st.session_state``), and
    copying the runner's 38-parameter call plus its model/replay/records sequencing
    into the page would fork the flow into two implementations that drift.

    Arguments
    ---------
    ``**kwargs`` uses the click parameter names of ``slm-transfer-shapes`` (minus
    ``--out``), so ``campaign.CampaignSettings.kwargs()`` expands straight in.
    Omitted values fall back to **click's own defaults** (read off ``run.params``)
    rather than a second table written here: two copies of the same default drift
    silently, and click is the contract the lane scripts already bind to.

    ``should_stop`` is a cooperative, **segment-granularity** stop: it is asked once
    before each CCD block group (i.e. before each ``(shape, stage)`` segment) and once
    before ``replay`` and before ``records``. A single block (~24-40 s: one A1(+B)
    capture, its optimization and its write) is an atomic transaction — interrupting
    it mid-way leaves a half-written ``manifest.json``, so it is never polled inside
    a block. "Stop" therefore means "stop after this segment", not "cut the device".
    A segment that never starts is journalled ``stopped`` (``campaign_progress``
    reads any unknown status verbatim and does not raise), and this function returns
    normally — an operator pressing stop is not a failure.

    Returns:
        ``{"out": str, "stages": int, "blocks": int, "stopped": bool}``. ``stages`` /
        ``blocks`` are the **planned** totals (``campaign.campaign_matrix``: 4 shapes
        → 16 / 112); what actually ran is read back from the directory by
        ``campaign_progress.collect_campaign_progress``.

    Raises:
        OpticalPathInvalid: no usable light on this path (someone blocked the beam, the
            laser is off, the path walked away). Expected on the bench, not a defect:
            the exception carries the exposure ladder readings plus ``reason`` /
            ``cause`` / ``report()``. Only :func:`run` turns it into exit code 3 — a
            programmatic caller (the GUI) gets the object and can render the numbers.
    """
    from ao_shaping.tools.slm.transfer.campaign import campaign_matrix, matrix_totals

    defaults = {param.name: param.default for param in run.params
                if param.name not in ("out_dir", "execute", "records_only")}
    unknown = sorted(set(kwargs) - set(defaults))
    if unknown:
        raise TypeError(f"execute_campaign 不认识这些参数: {unknown}"
                        f"（可用: {sorted(defaults)}）")
    values: dict[str, Any] = {**defaults, **kwargs}

    shapes = list(values["shapes"])
    replay_exposures = list(values["replay_exposures"])
    side = values["side"]
    period = values["period"]
    window_span = values["window_span"]
    window_step = values["window_step"]
    zero_radius = values["zero_radius"]
    capture_grid = values["capture_grid"]
    capture_count = values["capture_count"]
    fit_iterations = values["fit_iterations"]
    fit_span = values["fit_span"]
    amplitude_center_lr = values["amplitude_center_lr"]
    control_grid = values["control_grid"]
    fft_size = values["fft_size"]
    frames = values["frames"]
    seed = values["seed"]
    camera = values["camera"]
    cam_id = values["cam_id"]
    slm_number = values["slm_number"]
    device = values["device"]
    initial_phase = values["initial_phase"]
    seed_bakeoff = values["seed_bakeoff"]
    gs_seed = values["gs_seed"]
    replay_exposures_auto = values["replay_exposures_auto"]
    record_responses = values["record_responses"]
    model_dir = values["model_dir"]
    resume = values["resume"]
    order = values["order"]
    profile_timing = values["profile_timing"]
    async_capture_write = values["async_capture_write"]
    jacobian_guard = values["jacobian_guard"]
    jacobian_low = values["jacobian_low"]
    jacobian_high = values["jacobian_high"]
    jacobian_floor = values["jacobian_floor"]
    spectral_weight = values["spectral_weight"]
    spectral_cutoff = values["spectral_cutoff"]
    objective = values["objective"]
    weight_schedule = values["weight_schedule"]

    totals = matrix_totals(campaign_matrix(shapes))
    summary = {"out": str(out), "stages": totals["segments"], "blocks": totals["blocks"],
               "stopped": False}

    def stop_requested() -> bool:
        return bool(should_stop is not None and should_stop())

    out.mkdir(parents=True, exist_ok=True)
    journal = _Journal(path=out / "journal.json")
    journal.record("plan", "complete",
                   plan=_plan_dict(out=out, execute=True, **values))

    from ao_shaping.tools.slm.transfer import bootstrap as bootstrap_stage
    from ao_shaping.tools.slm.transfer import geometric_seed

    # ---------------------------------------------------------------- model
    if model_dir is not None:
        root = model_dir.resolve()
        capture_dir = root / "capture"
        fit_dir = root / "fit"
        model_640 = root / "model640"
        wide_dir = root / "model_wide640"
        journal.record("model", "reused", source=str(root))
        # 复用模型时整段采集被跳过, 于是**没有人在模型加载前看过一眼光路**。
        # 那份模型是在原来那束光上拟合的: 光路失效后继续用, 第一个变暗的读数要
        # 等到整形段第一块才出现 (2026-10-28 那次是 48 s 的曝光阶梯空转), 而模型
        # 加载/目标/种子准备都已经跑完。少数几帧先问一句「有没有光」, 没有就抛
        # ``OpticalPathInvalid`` (CLI 由 :func:`run` 映射成退出码 3, GUI 由调用方
        # 直接渲染 ``report()``)。
        capture_stage.precheck_light(camera=camera, cam_id=cam_id,
                                     label=f"model reuse {root.name}")
    else:
        root = out
        capture_dir = root / "capture"
        journal.record("capture", "running")
        capture_stage.capture(SimpleNamespace(
            out=capture_dir, camera=camera, cam_id=cam_id, slm_number=slm_number,
            grid=capture_grid, count=capture_count, frames=frames, seed=seed,
            execute=True, exposure_max_steps=12, save_dataset=True))
        journal.record("capture", "complete", out=str(capture_dir))

        # The nominal checkpoint comes from this capture alone: the model is
        # trained from zero, not warm-started from a historical one.
        nominal = root / "initial"
        nominal.mkdir(parents=True, exist_ok=True)
        frozen, provenance = bootstrap_stage.bootstrap(capture_dir)
        (nominal / "frozen_parameters.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
        (nominal / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

        fit_dir = root / "fit"
        model_640 = fit_dir / "model640"
        wide_dir = fit_dir / "model_wide640"
        journal.record("fit", "running")
        fit_stage.fit(SimpleNamespace(
            dataset=capture_dir, checkpoint=nominal / "frozen_parameters.json",
            out=fit_dir, iterations=fit_iterations, span=fit_span, device=device,
            adaptation="full", amplitude_center_lr=amplitude_center_lr, radiometry="exposure",
            record_responses=record_responses))
        journal.record("fit", "complete", out=str(fit_dir))

        model_640 = root / "model640"
        wide_dir = root / "model_wide640"
        geometric_seed.export_frozen_parameters(fit_dir / "frozen_parameters.json", model_640,
                                                grid=control_grid, fft=fft_size)
        journal.record("model", "complete", model640=str(model_640))

    # The zero order is measured on this path, never assumed.
    rows = _json(capture_dir / "manifest.json")
    reference = next(row for row in rows if row["label"] == "reference_start")
    zero_xy = [float(v) for v in np.round(fit_stage.spot(reference))]
    journal.record("geometry", "complete", zero_xy=zero_xy)

    if not (model_640 / "frozen_parameters.json").is_file():
        geometric_seed.export_frozen_parameters(fit_dir / "frozen_parameters.json", model_640,
                                                grid=control_grid, fft=fft_size)
        journal.record("model", "complete", model640=str(model_640))

    # The wide-ramp variant needs a fresh signed-ramp measurement at the
    # control resolution actually used for shaping.
    if not (wide_dir / "frozen_parameters.json").is_file():
        geometry_dir = root / "geometry_wide"
        journal.record("capture_geometry", "running")
        capture_stage.capture(SimpleNamespace(
            out=geometry_dir, camera=camera, cam_id=cam_id, slm_number=slm_number,
            grid=control_grid, count=6, frames=frames, seed=seed + 1,
            execute=True, exposure_max_steps=12, save_dataset=True,
            ramp_periods=[16, 32]))
        journal.record("capture_geometry", "complete", out=str(geometry_dir))
        geometric_seed.export_frozen_parameters(model_640 / "frozen_parameters.json",
                                                wide_dir, grid=control_grid, fft=fft_size,
                                                geometry_dataset=geometry_dir)
        journal.record("wide_model", "complete", out=str(wide_dir))

    # ------------------------------------------------------- seed bake-off
    chosen: dict[str, str] = {}
    if seed_bakeoff:
        for shape in shapes:
            candidates = ["transport", "backprojection", "gs"] if shape == "square" \
                else ["backprojection", "gs"]
            if not gs_seed:
                candidates = [kind for kind in candidates if kind != "gs"]
            if len(candidates) == 1:
                # 单一候选无需实测比较, 直接采用 (仍写入 journal 以便溯源)。
                chosen[shape] = candidates[0]
                journal.record("seed_bakeoff", "skipped", shape=shape, candidates=candidates,
                               chosen=candidates[0], reason="single candidate")
                logger.info("seed bake-off {}: single candidate -> {}", shape, candidates[0])
                continue
            scores: dict[str, float] = {}
            for kind in candidates:
                bake_dir = out / "seed_bakeoff" / f"{shape}_{kind}"
                manifest_path = bake_dir / "manifest.json"
                if manifest_path.is_file():
                    scores[kind] = float(_json(manifest_path)[0]["metrics"]["cv"])
                    continue
                args = _validate_args(
                    checkpoint=model_640 / "frozen_parameters.json", out=bake_dir,
                    zero_xy=zero_xy, shapes=[shape], recipe=STAGE_RECIPES[0],
                    initial_phase=kind, seed_run=None, device=device, camera=camera,
                    cam_id=cam_id, slm_number=slm_number, frames=frames, side=side,
                    period=period, window_span=window_span, zero_radius=zero_radius,
                    window_step=window_step, blocks=0, updates=0, adaptive_exposure=False)
                validate_stage.run(args)
                if not manifest_path.is_file():
                    raise RuntimeError(
                        f"seed bake-off {shape}/{kind} finished without writing "
                        f"{manifest_path}; a seed-only pass (blocks=0) must still "
                        f"measure its seed row")
                scores[kind] = float(_json(manifest_path)[0]["metrics"]["cv"])
            best = min(scores, key=lambda k: scores[k])
            chosen[shape] = best
            journal.record("seed_bakeoff", "complete", shape=shape, scores=scores, chosen=best)
            logger.info("seed bake-off {}: {} -> {}", shape, scores, best)
    else:
        chosen = {shape: _seed_kind(shape, initial_phase) for shape in shapes}

    # ------------------------------------------------------------- shaping
    recorder = None
    try:
        from ao_shaping.utils.io.file import Recorder

        recorder = Recorder(mark="cv_candidate", mode="min")
    # policy: Recorder is optional telemetry for this stage loop
    except Exception as exc:  # pragma: no cover - Recorder is part of the repo
        logger.warning("Recorder unavailable: {}", exc)

    stopped = run_stages(
        out=out, shapes=shapes, order=order, resume=resume, journal=journal,
        chosen=chosen, model_640=model_640, wide_dir=wide_dir, zero_xy=zero_xy,
        initial_phase=initial_phase, device=device, camera=camera, cam_id=cam_id,
        slm_number=slm_number, frames=frames, side=side, period=period,
        window_span=window_span, zero_radius=zero_radius, window_step=window_step,
        recorder=recorder, profile_timing=profile_timing,
        async_capture_write=async_capture_write,
        jacobian_guard=jacobian_guard, jacobian_low=jacobian_low,
        jacobian_high=jacobian_high, jacobian_floor=jacobian_floor,
        spectral_weight=spectral_weight, spectral_cutoff=spectral_cutoff,
        objective=objective, weight_schedule=weight_schedule,
        should_stop=should_stop)

    # -------------------------------------------------------------- replay
    runs = [out / "shape" / shape / recipe.name for shape in shapes for recipe in STAGE_RECIPES]
    replay_dir = out / "common_replay"
    if stopped or stop_requested():
        # 停止是**段粒度**的: 整形段边界停下后不再开重放, 把余下工作留给 --resume
        # (重放目录只有写了 ``released.json`` 才被当作已完成, 见下)。
        journal.record("replay", "stopped", reason="operator stop before replay")
        journal.record("session", "stopped", out=str(out))
        summary["stopped"] = True
        click.echo(f"Session stopped at a segment boundary, before replay: {out}")
        return summary
    # ``released.json`` is written only after every block replayed *and* both
    # devices were released, so it is the completion marker. ``config.json`` is
    # not: it is written before the first block, so a replay that aborted on a
    # saturated core (2026-10-29, lane 01) would be skipped as "reused" and the
    # lane would silently report a session with no usable replay.
    if resume and (replay_dir / "released.json").is_file():
        # Same contract as a reused stage: the directory says it already ran.
        journal.record("replay", "reused", out=str(replay_dir))
    else:
        journal.record("replay", "running")
        exposures = list(replay_exposures)
        if replay_exposures_auto:
            observations = measured_peak_rates(runs)
            exposures = derive_replay_exposures(observations)
            journal.record("replay", "ladder_derived", exposures=exposures,
                           n_blocks=len(observations))
            logger.info("replay ladder derived from {} measured blocks: {} ms",
                        len(observations), exposures)
        replay_phases.replay(SimpleNamespace(
            initial_run=out / "shape" / shapes[0] / STAGE_RECIPES[0].name, runs=runs,
            out=replay_dir, exposures=exposures, frames=frames,
            camera=camera, cam_id=cam_id, slm_number=slm_number, include_best=False))
        journal.record("replay", "complete", out=str(replay_dir), exposures=exposures)

    # -------------------------------------------------------------- record
    if stop_requested():
        # 重放已经跑完, 但操作员在回收表之前按了停止: 回收表可由 --records-only
        # 从 manifest 重建 (与下面 "no captures" 分支同一理由), 所以这里不是丢数据。
        journal.record("record", "stopped", reason="operator stop before records")
        journal.record("session", "stopped", out=str(out))
        summary["stopped"] = True
        click.echo(f"Session stopped before records: {out}")
        return summary
    records_marker = out / "records" / "process.csv"
    if resume and records_marker.is_file():
        journal.record("record", "reused", out=str(records_marker.parent))
    elif recorder is None or not recorder.history:
        # ``--resume`` on an otherwise finished lane re-enters no stage, so there
        # is no capture to export. ``save_records`` raises "history is empty",
        # which would fail a session whose only remaining work was the replay.
        # ``--records-only`` rebuilds the table from the stage manifests.
        journal.record("record", "skipped", reason="no captures this run; use --records-only")
    elif recorder is not None:
        saved = recording.save_records(recorder, out / "records", mark="process",
                                       gif_name="process.gif", fps=2.0,
                                       title="多形状物理预测模型闭环整形 (逐 CCD 块)")
        journal.record("record", "complete", **saved)
        click.echo(json.dumps(saved, indent=2, ensure_ascii=False))
    record_optical_session(out)
    journal.record("session", "complete", out=str(out), **optical_session_field(out))
    summary.update(optical_session_field(out))
    click.echo(f"Session complete: {out}")
    return summary


def _rebuild_records(out: Path, shapes: list[str]) -> None:
    """Rebuild the Recorder table from the run directories already on disk.

    Offline and device-free: the run directories are the source of truth, so
    the record table can be re-derived (after a fix, or with a new column)
    without re-running anything.
    """
    from ao_shaping.utils.io.file import Recorder

    recorder = Recorder(mark="cv_candidate", mode="min")
    for shape in shapes:
        for recipe in STAGE_RECIPES:
            stage_dir = out / "shape" / shape / recipe.name
            if not (stage_dir / "manifest.json").is_file():
                continue
            added = recording.append_run(recorder, stage_dir, f"{shape}/{recipe.name}")
            logger.info("recorded {} blocks from {}", added, stage_dir)
    saved = recording.save_records(recorder, out / "records", mark="process",
                                   gif_name="process.gif", fps=2.0,
                                   title="多形状物理预测模型闭环整形 (逐 CCD 块)")
    click.echo(json.dumps(saved, indent=2, ensure_ascii=False))


def _stage_sequence(shapes, order: str) -> list:
    """The order in which ``(shape, stage index)`` pairs run.

    ``shape`` is the historical nesting: one shape's four stages back to back.
    ``stage`` is recipe-major: stage 1 for every shape, then stage 2, and so on.
    With the bench already warm and an exposure landed, stage-major keeps the
    same bit depth / window step / exposure policy across consecutive runs, so a
    mid-session abort costs one stage of one shape instead of every later stage.
    No wall-clock saving is claimed here: the two orders measure the same stages.
    """
    pairs = [(shape, index) for shape in shapes for index in range(len(STAGE_RECIPES))]
    # ``sorted`` is stable, so within one stage the shape order is preserved.
    return sorted(pairs, key=lambda pair: pair[1]) if order == "stage" else pairs


def run_stages(*, out: Path, shapes: list[str], order: str, resume: bool, journal,
               chosen: dict[str, str], model_640: Path, wide_dir: Path, zero_xy,
               initial_phase: str, device: str, camera: str, cam_id: int, slm_number: int,
               frames: int, side: float, period: float, window_span: float,
               zero_radius: float, window_step: float, recorder,
               profile_timing: bool = False, async_capture_write: bool = False,
               jacobian_guard: bool = True, jacobian_low: float = 0.5,
               jacobian_high: float = 0.85, jacobian_floor: float = 0.02,
               spectral_weight: float = 0.0,
               spectral_cutoff: float | None = None,
               objective: str = "cv", weight_schedule: str = "constant",
               should_stop: Callable[[], bool] | None = None) -> bool:
    """Run every ``(shape, stage)`` in ``order``, reusing finished stages on ``--resume``.

    Split out of :func:`run` so the ordering and the reuse rule are testable
    without opening a device: a caller can drive this with ``validate.run``
    monkeypatched and read back which stage directories were entered. On
    ``--resume`` a stage whose ``manifest.json`` exists is *not* re-entered --
    it is journalled ``reused`` and its directory becomes the next stage's seed,
    which is what makes finishing a half-done shape a continuation rather than a
    re-measurement. Without ``--resume`` the old contract holds and a populated
    directory raises ``FileExistsError`` inside ``validate.run``.

    ``should_stop`` is checked **before each segment only** (return value ``True``
    means the operator stopped it): one segment is 4-12 CCD blocks of one shape and
    every single block is an atomic capture+optimize+write transaction, so a stop
    that cut inside a block would leave a half-written ``manifest.json`` on disk.
    The segment that never starts is journalled ``stopped``.

    A completed segment also writes its measured wall-clock cost into its own
    ``cost.json`` (elapsed from the journal's ``running``/``complete`` timestamps,
    blocks from the segment manifest). After that, a **segment-boundary pause**
    request found in the campaign directory (``campaign_control``) is honoured:
    the segment is journalled ``paused``, the request file is cleared, and the
    loop returns ``True`` so no replay starts. That request is the on-disk twin
    of ``should_stop``; both stop at the same boundary, so a paused run resumes
    as ``reused``.

    Returns:
        ``True`` when a stop or pause request ended the loop early, ``False``
        otherwise.
    """
    for shape, index in _stage_sequence(shapes, order):
        recipe = STAGE_RECIPES[index]
        stage_dir = out / "shape" / shape / recipe.name
        if resume and (stage_dir / "manifest.json").is_file():
            journal.record(f"{shape}/{recipe.name}", "reused", out=str(stage_dir))
            continue
        if should_stop is not None and should_stop():
            journal.record(f"{shape}/{recipe.name}", "stopped",
                           reason="operator stop before segment start")
            logger.info("stop requested; {} not started", f"{shape}/{recipe.name}")
            return True
        # Each stage continues from the nearest earlier stage that is on disk,
        # which under --resume is the reused directory just above.
        seed_run = next((out / "shape" / shape / earlier.name
                         for earlier in reversed(STAGE_RECIPES[:index])
                         if (out / "shape" / shape / earlier.name / "manifest.json").is_file()), None)
        checkpoint = (wide_dir if recipe.wide_model else model_640) / "frozen_parameters.json"
        kind = chosen[shape] if index == 0 else _seed_kind(shape, initial_phase)
        args = _validate_args(
            checkpoint=checkpoint, out=stage_dir, zero_xy=zero_xy, shapes=[shape],
            recipe=recipe, initial_phase=kind, seed_run=seed_run, device=device,
            camera=camera, cam_id=cam_id, slm_number=slm_number, frames=frames,
            side=side, period=period, window_span=window_span, zero_radius=zero_radius,
            window_step=window_step if index == 0 else None,
            profile_timing=profile_timing, async_capture_write=async_capture_write,
            jacobian_guard=jacobian_guard, jacobian_low=jacobian_low,
            jacobian_high=jacobian_high, jacobian_floor=jacobian_floor,
            spectral_weight=spectral_weight, spectral_cutoff=spectral_cutoff,
            objective=objective, weight_schedule=weight_schedule)
        # validate.run only reads this optional flag; absent it keeps the
        # historical "new directory or fail" contract.
        args.resume = resume
        key = f"{shape}/{recipe.name}"
        journal.record(key, "running")
        validate_stage.run(args)
        if recorder is not None:
            added = recording.append_run(recorder, stage_dir, key)
            logger.info("recorded {} blocks from {}", added, stage_dir)
        journal.record(key, "complete", out=str(stage_dir))
        # 段耗时（journal 里刚写下的两条时间戳）× manifest 的真实块数落进段目录的
        # cost.json，供成本面板外推；段边界的协作式暂停也在这一步判定。
        if finish_segment(journal, key, stage_dir, stage=recipe.name, shape=shape,
                          fallback_blocks=recipe.blocks, out=out):
            return True
    return False


@click.command(name="slm-transfer-shapes")
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True,
              help="运行根目录 (每个阶段写入其子目录)")
@click.option("--shapes", type=click.Choice(list(TRANSFER_SHAPE_CHOICES)),
              multiple=True, default=("square", "triangle", "star", "circle"),
              show_default=True, help="要整形的形状 (同一尺寸, 由 --side 给出); 可多次指定")
@click.option("--side", type=float, default=1200.0, show_default=True,
              help="目标边长 (CCD 原生像元)")
@click.option("--period", type=float, default=128.0, show_default=True,
              help="载波周期 (面板像元)")
@click.option("--window-span", type=float, default=820.0, show_default=True,
              help="相机窗口半宽 (CCD 原生像元)")
@click.option("--window-step", type=float, default=4.0, show_default=True,
              help="第一段的相机采样步长 (CCD 原生像元); 后续段按流程表")
@click.option("--zero-radius", type=float, default=60.0, show_default=True,
              help="零级排除半径 (CCD 原生像元)")
@click.option("--capture-grid", type=int, default=384, show_default=True,
              help="探针采集的控制网格")
@click.option("--capture-count", type=int, default=12, show_default=True,
              help="探针相位的数量 (训练/验证各半)")
@click.option("--fit-iterations", type=int, default=150, show_default=True,
              help="物理预测模型的拟合迭代数 (从零训练)")
@click.option("--fit-span", type=int, default=120, show_default=True,
              help="拟合窗口半宽 (CCD 原生像元)")
@click.option("--amplitude-center-lr", type=float, default=4.0, show_default=True,
              help="照明位置学习率 (面板像元/步); 从零训练需要它走完标称偏差")
@click.option("--control-grid", type=int, default=640, show_default=True,
              help="整形阶段的控制网格")
@click.option("--fft-size", type=int, default=2048, show_default=True,
              help="整形阶段的 FFT 尺寸")
@click.option("--frames", type=int, default=3, show_default=True, help="每次采集的帧数")
@click.option("--seed", type=int, default=20261027, show_default=True, help="探针族随机种子")
@click.option("--camera", default="daheng", show_default=True)
@click.option("--cam-id", type=int, default=0, show_default=True)
@click.option("--slm-number", type=int, default=1, show_default=True)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto",
              show_default=True, help="计算设备 (auto = 有 CUDA 用 CUDA)")
@click.option("--initial-phase",
              type=click.Choice(["auto", "plane", "random", "transport", "backprojection", "gs"]),
              default="auto", show_default=True,
              help="初相位构造; auto = 正方形用非 GS 传输, 其余用非 GS 反投影; "
                   "plane/random = 严格冷启动 (不读目标、不用物理预测模型)")

@click.option("--seed-bakeoff/--no-seed-bakeoff", default=True, show_default=True,
              help="先实测各候选初相位的核心 CV, 逐形状择优 (对比择优)")
@click.option("--gs-seed/--no-gs-seed", default=True, show_default=True,
              help="对比择优是否允许 GS 初相位候选; 无 GS 冷启动方案须用 --no-gs-seed")
@click.option("--replay-exposures", type=float, multiple=True, default=(50.0, 100.0),
              show_default=True,
              help="共同重放的曝光 (ms); 重复旗标 (--replay-exposures 20 --replay-exposures 50)。"
                   "注意 click 的 multiple=True 不接受空格分隔的多个值, 生成计划时须逐项展开")
@click.option("--replay-exposures-auto", is_flag=True, default=False,
              help="按本 lane 已测各块的核心峰值率推导共同重放曝光 (覆盖 --replay-exposures)。"
                   "固定 ladder 只在一个 lane 内成立: 200 px 的核在 125 ms 就满量程, "
                   "而 1200 px 在 225 ms 仍可用, 故不能跨尺寸共用")
@click.option("--record-responses/--no-record-responses", default=False, show_default=True,
              help="拟合时记录逐批物理响应统计 (fit --record-responses), 写入 fit/")
@click.option("--model-dir", type=click.Path(path_type=Path), default=None,
              help="复用已有会话目录 (含 capture/ 与 fit/), 跳过模型采集与拟合")
@click.option("--records-only", is_flag=True,
              help="只从已有的分段目录重建 Recorder 记录 (离线, 不打开设备)")
@click.option("--resume", is_flag=True,
              help="跳过已有 manifest.json 的阶段 (journal 记 reused) 并沿用其目录作为下一段种子")
@click.option("--order", "order", type=click.Choice(["shape", "stage"]), default="shape",
              show_default=True,
              help="shape = 逐形状跑完四段; stage = 先跑所有形状的第 1 段, 再第 2 段 (连续几次运行共用同一 bit depth / window step / 曝光策略, 中断只损失一个形状的一段)")
@click.option("--execute", is_flag=True, help="真正打开设备执行 (否则只打印计划)")
@click.option("--profile-timing/--no-profile-timing", default=False,
              help="在每个采集的 manifest 行里记录 slm_write/settle/read/reduce/save 分段耗时 (ms); "
                   "只做测量, 不改任何采集值。默认关闭: 跨 lane 比较时要显式选择。")
@click.option("--async-capture-write/--no-async-capture-write", default=False,
              help="把采集 npz 落盘放到后台线程, 与下一次采集的液晶稳定重叠; "
                   "只改变字节何时落盘, 不改变内容。默认关闭。")
@click.option("--jacobian-guard/--no-jacobian-guard", default=True,
              help="逐块用 A1/B 的实测差分校验模型步方向, 并按一致性收缩下一块的 "
                   "phase_rms_cap (issue #213 质量项 3a)。默认开启: 只读已有帧, "
                   "不改变接受/拒绝判定。跨 lane 比较旧行为时用 --no-jacobian-guard。")
@click.option("--jacobian-low", type=float, default=0.5, show_default=True,
              help="方向相关性低于此值判为 contradict (cap 减半)")
@click.option("--jacobian-high", type=float, default=0.85, show_default=True,
              help="方向相关性高于此值判为 agree (cap 恢复)")
@click.option("--jacobian-floor", type=float, default=0.02, show_default=True,
              help="连续被否定时 cap 的下限 (rad, 照明加权 RMS)")
@click.option("--spectral-weight", type=float, default=0.0, show_default=True,
              help="相位增量带外功率占比的惩罚权重 (issue #213 质量项 1b)。"
                   "默认 0 = 关闭: 器件 MTF 尚未实测, 开启前应先确认 cutoff。")
@click.option("--spectral-cutoff", type=float, default=None,
              help="通过带边缘, 单位 cycles/控制像素 (0..0.5, 0.5 = 控制网格 Nyquist)。"
                   "缺省 = 0.35 (Nyquist 的 70%); 实测 MTF 后覆盖。")
@click.option("--objective", type=click.Choice(["cv", "shape", "formation"]), default="cv",
              show_default=True,
              help="优化目标族。cv = 只磨核心均匀度 (历史行为); shape 加能量与形状项; "
                   "formation 再启用 partition / peripheral / floor / boundary 四项 "
                   "(这几项此前从未在本 campaign 中被计算过 —— 抽取它们等于换实验变体, "
                   "须单独列为一组, 不得与 cv 变体的结果混读)。验收判据不随此选项改变。")
@click.option("--weight-schedule", type=click.Choice(["constant", "staged"]), default="constant",
              show_default=True,
              help="损失权重随时间变化的方式 (issue #213 质量项 4c)。constant = 历史常数; "
                   "staged = 前 1/3 块偏 energy、中段偏 shape、后 1/3 偏 cv2。只移动优化路径, "
                   "不改验收判据; 每块用到的权重写进 summary 条目。需 --objective 非 cv。")
def run(out_dir: Path, shapes: tuple[str, ...], side: float, period: float, window_span: float,
        window_step: float, zero_radius: float, capture_grid: int, capture_count: int,
        fit_iterations: int, fit_span: int, amplitude_center_lr: float, control_grid: int,
        fft_size: int, frames: int, seed: int, camera: str, cam_id: int, slm_number: int,
        device: str, initial_phase: str, seed_bakeoff: bool, gs_seed: bool,
        replay_exposures: tuple[float, ...],
        replay_exposures_auto: bool,
        record_responses: bool,
        model_dir: Path | None, records_only: bool, resume: bool, order: str,
        execute: bool, profile_timing: bool, async_capture_write: bool,
        jacobian_guard: bool, jacobian_low: float, jacobian_high: float,
        jacobian_floor: float, spectral_weight: float,
        spectral_cutoff: float | None, objective: str,
        weight_schedule: str) -> None:
    """物理预测模型闭环整形：从零训练模型 + 目标整形，逐形状执行。

    典型用法::

        python src/ao_shaping/main.py slm-transfer-shapes \\
            --out data/debug/slm_transfer/session \\
            --shapes square --shapes triangle --execute
    """
    shapes = list(shapes)
    replay_exposures = list(replay_exposures)
    out = out_dir.resolve()
    # Fail at argument time, not after the model is loaded: a staged schedule on
    # the CV-only loss has nothing to stage and would silently run unscheduled.
    if weight_schedule == "staged" and objective == "cv":
        raise click.UsageError("--weight-schedule staged needs --objective shape or formation; "
                               "the cv objective has nothing to stage")
    if records_only:
        _rebuild_records(out, shapes)
        return
    params: dict[str, Any] = dict(
        shapes=shapes, side=side, period=period, window_span=window_span,
        window_step=window_step, zero_radius=zero_radius, capture_grid=capture_grid,
        capture_count=capture_count, fit_iterations=fit_iterations, fit_span=fit_span,
        amplitude_center_lr=amplitude_center_lr, control_grid=control_grid,
        fft_size=fft_size, frames=frames, seed=seed, camera=camera, cam_id=cam_id,
        slm_number=slm_number, device=device, initial_phase=initial_phase,
        seed_bakeoff=seed_bakeoff, gs_seed=gs_seed, replay_exposures=replay_exposures,
        replay_exposures_auto=replay_exposures_auto, record_responses=record_responses,
        model_dir=model_dir, resume=resume, order=order, profile_timing=profile_timing,
        async_capture_write=async_capture_write, jacobian_guard=jacobian_guard,
        jacobian_low=jacobian_low, jacobian_high=jacobian_high,
        jacobian_floor=jacobian_floor, spectral_weight=spectral_weight,
        spectral_cutoff=spectral_cutoff, objective=objective,
        weight_schedule=weight_schedule)
    plan = _plan_dict(out=out, execute=execute, **params)
    click.echo(json.dumps(plan, indent=2, ensure_ascii=False))
    if not execute:
        click.echo("\n未指定 --execute : 只打印计划, 不打开设备。")
        return
    # The executing half lives in ``execute_campaign`` so the GUI campaign page can
    # drive the identical sequence from a background thread instead of a second copy.
    #
    # 「光路无光」的退出码映射放在**这里**，不放在 ``execute_campaign`` 里：那是本模块
    # 唯一的库函数（GUI 在后台线程调用它），而 ``capture.py`` / ``validate.py`` 的既有
    # 约定也是「库抛 ``OpticalPathInvalid``，只有 ``main`` 把它翻成退出码 3」。翻早了
    # 异常对象就消失在一句 ``SystemExit(3)`` 里 —— 调用方只剩一个数字，实测曝光阶梯、
    # ``reason`` / ``cause`` 全都拿不到。这一条是**预期**的失败（有人挡光 / 激光没出光
    # / 光路漂走），不是缺陷：只报实测数字，不打印 traceback；普通异常照旧是退出码 1。
    try:
        execute_campaign(out=out, **params)
    except OpticalPathInvalid as exc:
        click.echo(exc.report(), err=True)
        raise SystemExit(exc.exit_code) from None


if __name__ == "__main__":
    run()
