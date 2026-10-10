"""Zernike coefficient search CLI runners (GA and greedy local search).

Two commands, one module:

* :func:`run` → ``ga-zernike`` — genetic algorithm over Zernike coefficients,
  minimising WFS-measured wavefront RMS via SLM phase.
* :func:`greedy_zernike_run` → ``greedy-zernike`` — greedy local search, or any
  black-box heuristic from ``algorithm/heuristic`` via ``--algorithm``.

They were separate modules until 2026-10-04. The merge is real sharing, not
co-location: both build the same WFS/SLM option trio, both call an optimizer in
``optimizer/wf`` that returns the same Recorder shape, and both then persist it
through the same three steps -- ``get_best_iter()``, optional debug artefacts,
``save_best`` into ``flatten_zernike/<date>``. That persistence tail is now
:func:`_persist_results`.

Usage::

    python -m ao_shaping.runners.zernike_search_runner [opts]                 -> ga-zernike
    python -m ao_shaping.runners.zernike_search_runner greedy-zernike [opts] -> greedy-zernike

Or via the unified CLI::

    python -m ao_shaping.main ga-zernike -n 4
    python -m ao_shaping.main greedy-zernike --algorithm spgd
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

import click
import numpy as np
from loguru import logger

from ao_shaping.drivers import MlaRes
from ao_shaping.optimizer.wf.ga_zernike import optimizer_ga
from ao_shaping.optimizer.wf.greedy_zernike import optimizer_greedy
from ao_shaping.runners.runner_common import (
    GaZernikeParams,
    GreedyZernikeParams,
    WfsParams,
    ZernikeSlmParams,
)
from ao_shaping.utils.cli.params import with_params
from ao_shaping.utils.io.cli_helpers import (
    get_date_dir_name,
    get_debug_mode,
    resolve_debug,
    setup_coredumpy,
)
from ao_shaping.utils.io.file import (
    build_debug_save_paths,
    save_optimization_debug_artifacts,
)

#: Selectors accepted by :func:`main` for `python -m` dispatch.
COMMAND_SELECTORS = ("ga-zernike", "greedy-zernike")


def _persist_results(
    records: object,
    *,
    root_dir: Path,
    tag: str,
    opt_wavefront: np.ndarray,
    note_fmt: str,
    want_artifacts: bool,
) -> tuple[int, float]:
    """Write debug artefacts and the best Zernike set for one search run.

    Both search commands ended with this identical sequence; only the artefact
    directory tag, the wording of the plot note and which wavefront counts as
    "optimal" differed.

    Args:
        records: Recorder returned by the optimizer.
        root_dir: Data root from ``--dir``.
        tag: Sub-directory name for the debug artefacts.
        opt_wavefront: Wavefront to label as the optimum. The two callers index
            ``_wavefront`` differently -- GA uses element 0, greedy uses 1 --
            and that difference is preserved rather than silently normalised,
            because it changes which image is written.
        note_fmt: Annotation for the comparison plot, formatted with
            ``step`` and ``metric``.
        want_artifacts: Whether to dump the debug artefacts at all.

    Returns:
        ``(best_step, best_metric)`` from the recorder.
    """
    _min_iter, (best_step, best_metric) = records.get_best_iter()

    if want_artifacts:
        save_dir, saved_file = build_debug_save_paths(root_dir, tag)
        save_optimization_debug_artifacts(
            records=records,
            save_dir=save_dir,
            saved_file_name=saved_file,
            min_epoch=best_step,
            min_metric=best_metric,
            best_coeff_key="_c",
            init_wavefront=records.first["_wavefront"][0],
            opt_wavefront=opt_wavefront,
            init_title="初始波前",
            opt_title="最优波前",
            plot_params_note=note_fmt.format(step=best_step, metric=best_metric),
        )

    records.save_best(
        saved_dir=root_dir / "flatten_zernike" / get_date_dir_name(),
        target="_c",
        process_fn=np.round,
        fmt="%.6f",
    )
    return best_step, best_metric


# --------------------------------------------------------------------------- #
# ga-zernike
# --------------------------------------------------------------------------- #


@click.command(name="ga-zernike")
@click.pass_context
@with_params(GaZernikeParams, kw_name="params")
@with_params(WfsParams, kw_name="wfs")
@with_params(ZernikeSlmParams, kw_name="slm")
def run(
    ctx: click.Context,
    params: GaZernikeParams,
    wfs: WfsParams,
    slm: ZernikeSlmParams,
) -> None:
    """使用遗传算法优化Zernike系数进行波前校正."""
    debug = get_debug_mode()

    # Convert wfs_res from str to MlaRes
    wfs_res_enum = MlaRes.from_str(wfs.wfs_res)

    click.echo("GA-Zernike优化参数:")
    click.echo(f"  种群大小: {params.population_size}")
    click.echo(f"  迭代代数: {params.n_generations}")
    click.echo(f"  交叉概率: {params.crossover_prob}")
    click.echo(f"  变异概率: {params.mutation_prob}")
    click.echo(f"  锦标赛大小: {params.tournament_size}")
    click.echo(f"  精英数量: {params.elite_count}")
    click.echo(f"  最大Zernike阶数: {params.n_max}")
    click.echo(f"  波长: {slm.wavelength} nm")
    click.echo(f"  WFS分辨率: {wfs_res_enum}")
    click.echo(f"  瞳孔直径: {wfs.pupil_diameter}")
    click.echo(f"  瞳孔中心: {wfs.pupil_center}")
    click.echo(f"  早停阈值: {params.early_stop_threshold}")
    click.echo(f"  SLM编号: {slm.slm_number}")
    click.echo(f"  去除倾斜: {wfs.remove_tilt}")
    click.echo(f"  X偏移: {slm.shift_x}")
    click.echo(f"  Y偏移: {slm.shift_y}")

    recorder = optimizer_ga(
        wfs_type=wfs.wfs_type,
        n_generations=params.n_generations,
        population_size=params.population_size,
        crossover_prob=params.crossover_prob,
        mutation_prob=params.mutation_prob,
        tournament_size=params.tournament_size,
        elite_count=params.elite_count,
        n_max=params.n_max,
        wavelength=slm.wavelength,
        wfs_res=wfs_res_enum,
        pupil_diameter=wfs.pupil_diameter,
        pupil_center=wfs.pupil_center,
        early_stop_threshold=params.early_stop_threshold,
        slm_number=slm.slm_number,
        remove_tilt=wfs.remove_tilt,
        shift_x=slm.shift_x,
        shift_y=slm.shift_y,
    )

    min_iter, _ = recorder.get_best_iter()
    min_gen, min_rms = _persist_results(
        recorder,
        root_dir=Path(params.dir),
        tag="ga_zernike",
        opt_wavefront=min_iter["_wavefront"][0],
        note_fmt="Min RMS: {metric:.3f} @ gen {step}",
        want_artifacts=debug or params.show,
    )

    click.echo("\nGA-Zernike优化完成!")
    click.echo(f"  最佳RMS: {min_rms:.4f} @ generation {min_gen}")


# --------------------------------------------------------------------------- #
# greedy-zernike
# --------------------------------------------------------------------------- #


@click.command(name="greedy-zernike")
@click.pass_context
@with_params(GreedyZernikeParams, kw_name="params")
@with_params(WfsParams, kw_name="wfs")
@with_params(ZernikeSlmParams, kw_name="slm")
def greedy_zernike_run(
    ctx: click.Context,
    params: GreedyZernikeParams,
    wfs: WfsParams,
    slm: ZernikeSlmParams,
) -> None:
    """Zernike波前优化器 - 贪婪局部搜索 / 启发式搜索

    通过SLM进行波前校正，最小化WFS测量的波前RMS值。
    ``--algorithm spgd`` 使用贪婪局部搜索:
    1. 随机初始化N个位置，选取最优作为起始点
    2. 每次迭代采样n个随机扰动方向
    3. 评估所有候选(当前位置+n个扰动)，选择最优
    其他取值 (ga/pso/sa/hc/rs/cem/de) 使用 algorithm/heuristic 的启发式搜索。

    调试模式: ``main.py --debug greedy-zernike`` / 本命令 ``--debug`` / ``DEBUG=1``。
    """
    debug = resolve_debug(ctx, params.debug)

    records = optimizer_greedy(
        wfs_type=wfs.wfs_type,
        epochs=params.epochs,
        n_init=params.n_init,
        n_directions=params.n_directions,
        perturbation_scale=params.perturbation_scale,
        init_z=None,
        pupil_center=wfs.pupil_center,
        pupil_diameter=wfs.pupil_diameter,
        early_stop_threshold=params.early_stop_threshold,
        wavelength=slm.wavelength,
        shift_x=slm.shift_x,
        shift_y=slm.shift_y,
        n_max=params.n_max,
        wfs_res=MlaRes.from_str(wfs.wfs_res),
        remove_tilt=wfs.remove_tilt,
        slm_number=slm.slm_number,
        algorithm=params.algorithm,
        pop_size=params.pop_size,
    )

    min_iter, (min_epoch, min_rms) = records.get_best_iter()
    _persist_results(
        records,
        root_dir=Path(params.dir),
        tag="greedy_zernike",
        opt_wavefront=min_iter["_wavefront"][1],
        note_fmt="Min RMS: {metric:.3f} @ epoch {step}",
        want_artifacts=debug,
    )

    click.echo(f"贪婪优化完成，最优RMS值: {min_rms:.4f} @ epoch {min_epoch}")


# --------------------------------------------------------------------------- #
# `python -m` entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> NoReturn:
    """Dispatch to a command by selector for `python -m` execution.

    With no selector the module runs ``ga-zernike``.

    Raises:
        SystemExit: Always; the chosen Click command decides the code.
    """
    args = list(sys.argv[1:] if argv is None else argv)

    command: click.Command = run
    label = "ga-zernike"
    if args and args[0] in COMMAND_SELECTORS:
        label = args[0]
        args = args[1:]
        command = greedy_zernike_run

    try:
        command(args)
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"❌ 运行时错误: {e}")
        logger.exception("{} runner failed", label)
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    setup_coredumpy()
    main()
