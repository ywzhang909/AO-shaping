"""Model-in-the-loop square shaping on real SLM + CCD hardware.

Each round alternates two steps. **Step A** displays strong, varied random pupil
probes and refits one shared Zernike aberration of the forward model against the
measured frames, minimising loss(measured CCD, predicted far field). **Step B**
freezes that aberration and optimises a full-pixel SLM phase for a square target.
The phase is displayed, measured, and warm-starts the next round.

Two guards keep the steps from chasing each other: a trust region on the
coefficient change, and an acceptance test that rolls back a round whose fit or
whose measured quality regressed.

This is the model-in-the-loop counterpart of ``slm-gs-refine``. Prefer this one
when the bench's aberration is worth identifying; prefer ``slm-gs-refine`` when
it is not, or when the drift is fast relative to a round. The run aborts rather
than committing a phase when it concludes the bench is outside the model.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from ao_shaping.runners.runner_common import (
    SlmModelInLoopParams,
    build_model_in_loop_config,
    parse_point,
    save_model_in_loop_results,
)
from ao_shaping.utils.cli.params import with_params
from ao_shaping.utils.io.cli_helpers import get_debug_mode


@click.command()
@click.pass_context
@with_params(SlmModelInLoopParams, kw_name="params")
def run(ctx: click.Context, params: SlmModelInLoopParams) -> None:
    """SLM model-in-the-loop square shaping (SLM + CCD closed loop).

    Every round: fit the forward model's aberration from measured probe frames,
    then synthesise a square far-field spot with that aberration frozen. The
    fitted phase is only adopted if it measurably beats what is already held, and
    the final phase is kept only if it beats the flat baseline.

    \b
    Exit codes:
      0  a shaped phase was adopted (or flat was correctly kept)
      2  the geometry bake-off failed, or the bench stayed outside the model --
         the run refused to commit a phase. Use slm-gs-refine for that bench.
    """
    debug = get_debug_mode()
    root_dir = (
        Path(ctx.parent.obj.get("dir", "data"))
        if ctx.parent is not None and ctx.parent.obj is not None
        else Path("data")  # python -m standalone: no main group context
    )

    click.echo("=" * 68)
    click.echo("SLM Model-in-the-Loop Square Shaping")
    click.echo("=" * 68)
    click.echo(
        f"Camera: {params.cam_type} #{params.cam_id} size={params.cam_size} "
        f"exposure={params.exposure_time_ms or 'device default'}ms"
    )
    click.echo(f"SLM: #{params.slm_number} @ {params.slm_wavelength}nm")
    click.echo(
        f"Rounds: {params.n_rounds}  seed={params.seed}  "
        f"warm_start={'on' if params.warm_start else 'off'}"
    )
    click.echo(
        f"Step A: {params.probe_count} probes/round, spread={params.probe_spread} rad, "
        f"{params.step_a_iterations} iters, lr={params.step_a_lr}, n_orders={params.n_orders}"
    )
    click.echo(
        f"Step B: {params.step_b_iterations} iters, lr={params.step_b_lr}, "
        f"target={params.target_side} cam px, w_ee={params.w_efficiency} w_cv={params.w_uniformity}"
    )
    click.echo(
        f"Frozen modes: {params.frozen_modes or '(none)'} "
        "(piston/tip/tilt are unidentifiable from far-field INTENSITY)"
    )
    if params.forward_checkpoint is not None:
        click.echo(
            f"Step-A seed: {params.forward_checkpoint} "
            f"(assume_unverified_geometry={params.assume_unverified_geometry})"
        )
    click.echo(
        f"Guards: trust region {params.trust_region_c_l2} rad, "
        f"loss delta {params.acceptance_loss_delta}, score eps {params.acceptance_score_eps}, "
        f"abort after {params.max_rejection_streak} rejections"
    )
    click.echo(
        f"Model grid: region={params.region} padding={params.far_field_padding} "
        f"panel_span={params.panel_span_px}px  correlation floor="
        f"{params.min_geometry_correlation}"
    )

    if params.cam_type != "sim":
        click.echo("!" * 68)
        click.echo("硬件模式: 请确认光斑已对准 SLM 且 CCD 已就位。")
        click.echo(f"面板坐标下的光斑中心: {parse_point(params.pupil_center)} (须在面板上实测)")
        click.echo("每一轮都要重新采集探针帧, 耗时按 (probe_count+1) x 轮数 估计。")
        click.echo("Ctrl+C 会在当前轮结束后释放设备。")
        click.echo("!" * 68)
    click.echo("=" * 68)

    from ao_shaping.optimizer.wfless.slm_model_in_loop import optimize_slm_model_in_loop

    config = build_model_in_loop_config(params, debug=debug)

    try:
        result = optimize_slm_model_in_loop(config)
    except RuntimeError as exc:
        click.echo("")
        click.echo("!" * 68)
        click.echo(f"运行中止, 未下发任何整形相位: {exc}")
        click.echo("该台架不在正向模型描述范围内; 改用 slm-gs-refine。")
        click.echo("!" * 68)
        raise SystemExit(2) from exc

    recorder = result.recorder
    rounds = [row for row in recorder.history if row.get("stage") == "round"]
    accepted = [row for row in rounds if row.get("accepted")]

    click.echo("")
    click.echo("=" * 68)
    click.echo("Per-round summary")
    click.echo("=" * 68)
    header = (
        f"{'rd':>3} {'ok':>3} {'reason':<26} {'score':>8} {'CV':>8} "
        f"{'EE':>7} {'lossA':>9} {'|dc|':>7} {'probes':>7} {'lr':>7}"
    )
    click.echo(header)
    for row in recorder.history:
        score_before = row.get("score_before")
        before_txt = (
            "  n/a " if score_before is None or not np.isfinite(score_before)
            else f"{score_before:6.3f}->"
        )
        click.echo(
            f"{row.get('round', 0):>3} "
            f"{('Y' if row.get('accepted') else 'n'):>3} "
            f"{str(row.get('reason', '')):<26} "
            f"{before_txt}{row.get('score_after', float('nan')):.3f} "
            f"{row.get('uniformity_cv', float('nan')):>8.4f} "
            f"{row.get('ee_after', float('nan')):>7.4f} "
            f"{row.get('loss_after', float('nan')):>9.5f} "
            f"{row.get('coeff_step_l2', 0.0):>7.3f} "
            f"{row.get('probe_count', 0):>7} "
            f"{row.get('step_a_lr', 0.0):>7.4f}"
        )

    geometry = result.geometry
    click.echo("")
    click.echo(f"Status              : {result.status}")
    click.echo(f"Rounds accepted     : {len(accepted)} / {len(rounds)}")
    if geometry is not None:
        click.echo(
            f"Geometry            : {geometry.method}, correlation "
            f"{geometry.correlation:.4f}, camera/model px "
            f"{geometry.camera_px_per_model_px:.4f}"
        )
    click.echo(f"Target side         : {result.target_side_camera} cam px "
               f"({result.target_side_model} model px)")
    click.echo(f"Flat baseline score : {result.flat_score:.4f}")
    click.echo(f"Best shaped score   : {result.best_score:.4f}")

    save_result = save_model_in_loop_results(result, params, root_dir=root_dir, debug=debug)
    click.echo(f"History saved       : {save_result.csv_file}")
    if save_result.phase_file is not None:
        click.echo(
            f"Best phase saved    : {save_result.phase_file} "
            "(raw radians; display via Santec.create_phase_from_array)"
        )
    if save_result.geometry_file is not None:
        click.echo(f"Geometry saved      : {save_result.geometry_file}")
    if save_result.image_file is not None:
        click.echo(f"Best image saved    : {save_result.image_file}")
    click.echo(f"Recorder pkl/json   : {save_result.recorder_png.parent}")
    if debug:
        click.echo(f"Debug artefacts     : {save_result.recorder_png}")

    if result.status in ("aborted_unidentifiable", "aborted_rejection_streak"):
        click.echo("")
        click.echo("!" * 68)
        click.echo("该台架无法被当前正向模型描述, 已保留平场/最优已验证相位。")
        click.echo("建议改用: python src/ao_shaping/main.py slm-gs-refine ...")
        click.echo("!" * 68)
        raise SystemExit(2)
    click.echo("Done.")


if __name__ == "__main__":
    run()
