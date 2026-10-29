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
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np
from loguru import logger

from ao_shaping.tools.slm.transfer import capture as capture_stage
from ao_shaping.tools.slm.transfer import fit as fit_stage
from ao_shaping.tools.slm.transfer import recording, replay_phases, validate as validate_stage
from ao_shaping.tools.slm.transfer.protocol import derive_replay_exposures
from ao_shaping.utils.image.target import TRANSFER_SHAPE_CHOICES


@dataclass(frozen=True)
class StageRecipe:
    """One CCD-anchored optimization stage of the shaping chain."""

    name: str
    blocks: int
    updates: int
    phase_lr: float
    phase_rms_cap: float
    bit_depth: int
    ccd_check: bool
    accept_metric: str
    window_step: float
    core_margin: float | None = None
    wide_model: bool = False
    seed_policy: str = "strict"


#: The four stages of the square1200 chain, in order. Stage 1 starts from the
#: non-GS geometric seed; stages 2-4 continue from the previous stage's
#: accepted phase. Only stage 4 uses the wide-ramp model and the finer camera
#: sampling, which is where the resolution limit of the first three shows up.
STAGE_RECIPES: tuple[StageRecipe, ...] = (
    StageRecipe(name="transport_cv", blocks=4, updates=15, phase_lr=0.03,
                phase_rms_cap=0.25, bit_depth=8, ccd_check=False, accept_metric="loss",
                window_step=4.0),
    StageRecipe(name="mono12_cv", blocks=8, updates=20, phase_lr=0.025,
                phase_rms_cap=0.20, bit_depth=12, ccd_check=True, accept_metric="cv",
                window_step=4.0),
    StageRecipe(name="mono12_continue", blocks=12, updates=15, phase_lr=0.02,
                phase_rms_cap=0.15, bit_depth=12, ccd_check=True, accept_metric="cv",
                window_step=4.0),
    StageRecipe(name="mono12_dense", blocks=4, updates=20, phase_lr=0.015,
                phase_rms_cap=0.12, bit_depth=12, ccd_check=True, accept_metric="cv",
                window_step=2.0, core_margin=8.0, wide_model=True, seed_policy="transfer"),
)


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _measured_peak_rates(runs: list[Path]) -> list[tuple[float, float, float]]:
    """``(core_peak, full_scale, exposure_ms)`` for every block already measured.

    Read from the stage manifests of *this* lane, so the replay ladder is sized
    by the same bench and the same shaped spot it will replay. A missing manifest
    (``--resume`` into a half-done lane) simply contributes nothing.
    """
    observations: list[tuple[float, float, float]] = []
    for run in runs:
        manifest_path = Path(run) / "manifest.json"
        if not manifest_path.is_file():
            continue
        for row in _json(manifest_path):
            observations.append((row.get("core_peak"), row.get("full_scale"),
                                 row.get("exposure_ms")))
    return observations


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
                   adaptive_exposure: bool = True) -> SimpleNamespace:
    """Build the argument namespace ``validate.run`` expects."""
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
        accept_metric=recipe.accept_metric, objective="cv", abba=False,
        ccd_check=recipe.ccd_check, device=device, execute=True,
    )


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
               zero_radius: float, window_step: float, recorder) -> None:
    """Run every ``(shape, stage)`` in ``order``, reusing finished stages on ``--resume``.

    Split out of :func:`run` so the ordering and the reuse rule are testable
    without opening a device: a caller can drive this with ``validate.run``
    monkeypatched and read back which stage directories were entered. On
    ``--resume`` a stage whose ``manifest.json`` exists is *not* re-entered --
    it is journalled ``reused`` and its directory becomes the next stage's seed,
    which is what makes finishing a half-done shape a continuation rather than a
    re-measurement. Without ``--resume`` the old contract holds and a populated
    directory raises ``FileExistsError`` inside ``validate.run``.
    """
    for shape, index in _stage_sequence(shapes, order):
        recipe = STAGE_RECIPES[index]
        stage_dir = out / "shape" / shape / recipe.name
        if resume and (stage_dir / "manifest.json").is_file():
            journal.record(f"{shape}/{recipe.name}", "reused", out=str(stage_dir))
            continue
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
            window_step=window_step if index == 0 else None)
        # validate.run only reads this optional flag; absent it keeps the
        # historical "new directory or fail" contract.
        args.resume = resume
        journal.record(f"{shape}/{recipe.name}", "running")
        validate_stage.run(args)
        if recorder is not None:
            added = recording.append_run(recorder, stage_dir, f"{shape}/{recipe.name}")
            logger.info("recorded {} blocks from {}", added, stage_dir)
        journal.record(f"{shape}/{recipe.name}", "complete", out=str(stage_dir))


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
def run(out_dir: Path, shapes: tuple[str, ...], side: float, period: float, window_span: float,
        window_step: float, zero_radius: float, capture_grid: int, capture_count: int,
        fit_iterations: int, fit_span: int, amplitude_center_lr: float, control_grid: int,
        fft_size: int, frames: int, seed: int, camera: str, cam_id: int, slm_number: int,
        device: str, initial_phase: str, seed_bakeoff: bool, gs_seed: bool,
        replay_exposures: tuple[float, ...],
        replay_exposures_auto: bool,
        record_responses: bool,
        model_dir: Path | None, records_only: bool, resume: bool, order: str,
        execute: bool) -> None:
    """物理预测模型闭环整形：从零训练模型 + 目标整形，逐形状执行。

    典型用法::

        python src/ao_shaping/main.py slm-transfer-shapes \\
            --out data/debug/slm_transfer/session \\
            --shapes square --shapes triangle --execute
    """
    shapes = list(shapes)
    replay_exposures = list(replay_exposures)
    out = out_dir.resolve()
    if records_only:
        _rebuild_records(out, shapes)
        return
    plan = dict(out=str(out), shapes=list(shapes), side=side, period=period,
                window_span=window_span, window_step=window_step, zero_radius=zero_radius,
                capture_grid=capture_grid, capture_count=capture_count,
                fit_iterations=fit_iterations, control_grid=control_grid, fft_size=fft_size,
                initial_phase=initial_phase, seed_bakeoff=seed_bakeoff, gs_seed=gs_seed,
                execute=execute, resume=resume, order=order,
                record_responses=record_responses,
                replay_exposures=("auto" if replay_exposures_auto else list(replay_exposures)),
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
            out=out, shapes=shapes, frames=frames,
            session_groups=[dict(bit_depth=recipe.bit_depth, window_step=recipe.window_step,
                                 groups=recipe.blocks) for recipe in STAGE_RECIPES]))
    click.echo(json.dumps(plan, indent=2, ensure_ascii=False))
    if not execute:
        click.echo("\n未指定 --execute : 只打印计划, 不打开设备。")
        return

    out.mkdir(parents=True, exist_ok=True)
    journal = _Journal(path=out / "journal.json")
    journal.record("plan", "complete", plan=plan)

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
        params, provenance = bootstrap_stage.bootstrap(capture_dir)
        (nominal / "frozen_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
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

    run_stages(
        out=out, shapes=shapes, order=order, resume=resume, journal=journal,
        chosen=chosen, model_640=model_640, wide_dir=wide_dir, zero_xy=zero_xy,
        initial_phase=initial_phase, device=device, camera=camera, cam_id=cam_id,
        slm_number=slm_number, frames=frames, side=side, period=period,
        window_span=window_span, zero_radius=zero_radius, window_step=window_step,
        recorder=recorder)

    # -------------------------------------------------------------- replay
    runs = [out / "shape" / shape / recipe.name for shape in shapes for recipe in STAGE_RECIPES]
    replay_dir = out / "common_replay"
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
            observations = _measured_peak_rates(runs)
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
    journal.record("session", "complete", out=str(out))
    click.echo(f"Session complete: {out}")


if __name__ == "__main__":
    run()
