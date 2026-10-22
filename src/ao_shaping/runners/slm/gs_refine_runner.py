"""Click CLI runner for GS-warm-started freeform SLM square shaping.

Hardware port of the simulation-only ``iterative_zernike_shaping`` pipeline
(``initial 0.662 -> GS 0.812 -> refinement 0.849``) onto a real Santec SLM and
a real CCD (Daheng by default, MiiCam also supported).

========================= 用法 =========================

CLI 两种入口等价::

    python src/ao_shaping/main.py slm-gs-refine [OPTIONS]
    python -m ao_shaping.runners.slm.gs_refine_runner [OPTIONS]

示例::

    # 无硬件自检 (2f-Fourier 数值仿真)
    python src/ao_shaping/main.py slm-gs-refine --cam-type sim -e 20

    # 大恒 CCD + Santec SLM #1 @1064nm
    python src/ao_shaping/main.py slm-gs-refine --cam-type daheng --cam-id 0 -e 400

    # 目标方形边长显式指定 (相机像素)
    python src/ao_shaping/main.py slm-gs-refine --target-side 90

    # 从平场起步 (跳过 GS 预整形)
    python src/ao_shaping/main.py slm-gs-refine --no-gs-warm-start

========================= 方法说明 =========================

三段式流程, 与仿真的胜出配方一一对应:

1. **平场基线**: 下发平场相位读一帧, 用 ``argmax`` 定位 0 级 (2f 光路上 0 级
   落在相机几何中心纯属巧合), **冻结** 方形 ROI。全程不再重新定位 —— bench
   上已记录 ``argmax`` 滚动 ROI 会让 PIB/CV 不连续 (散斑场最大值在近似等值的
   颗粒间跳变), 优化器会去追一个已经不含光束的方框。

2. **GS 预矫正 (对比择优)**: 用 bench 前向模型算 GS 相位并下发实测。若 GS 分数
   不高于平场就丢弃 GS、从平场继续 —— 模型参数 (相机像素间距 / 光斑半径) 错了
   时, 代价只是两次测量, 而不是整轮跑坏。

3. **全频随机相位 SPGD 细化**: 粗网格 (默认 24x24 = 576 自由度) 全频随机相位, 每个 epoch
   扰动正负两次读帧, 用差分估计梯度, Adam 更新。仿真的细化靠 torch autograd
   穿过解析远场 FFT, 硬件上唯一的前向模型就是测量本身、不可微, 因此换成无感知
   (SPGD) 形式 —— 与既有 ``spgd-square`` / ``slm-gsnet`` 一致。

目标函数与仿真**同一个函数** ``composite_from_pib_cv(PIB, CV)``, 只是喂真实
CCD 帧, 所以硬件分数可与仿真的 0.849 直接对比。

三处硬件特有的正确性处理 (详见 optimizer 模块 docstring):
- 逐帧去背景并裁负: 对称读出噪声让约一半像素为负, 直接算 ``PIB`` 会 >1
  (实测 7164/14400 负像素 → PIB=1.0120), 优化器会开始追噪声。
- 每帧等**稳定**而非等固定时长: 驱动自动翻转时间估算对灰度统计相近的相位返回
  0 ms, 而面板实际还在弛豫。
- 连续写入不指定 memory slot: 同槽连写是固件 no-op, 面板不刷新。

结果输出 (默认 ``data/debug/slm_gs_refine/<日期>/``):
- 优化历史 CSV (recorder, 每轮一行)
- 最优相位 ``.npy`` (raw 未包裹弧度, 1200x1920)
- 最优远场图 PNG (``--save-best-image``)
"""

from __future__ import annotations

from pathlib import Path

import click

from ao_shaping.runners.runner_common import (
    SlmGsRefineParams,
    build_gs_refine_config,
    save_gs_refine_results,
)
from ao_shaping.utils.cli.params import with_params
from ao_shaping.utils.cli.exposure import (
    DEFAULT_EXPOSURE_TOKEN,
    ExposureChoice,
    resolve_exposure,
)
from ao_shaping.utils.io.cli_helpers import resolve_debug


@click.command()
@click.pass_context
@with_params(SlmGsRefineParams, kw_name="params")
def run(ctx: click.Context, params: SlmGsRefineParams) -> None:
    """SLM GS 预整形 + 全频随机相位 SPGD 细化 (SLM + CCD 闭环)。

    把远场光斑整形为均匀方形。先用 GS 算一个开环预整形相位 (只在实测优于平场时
    采用), 再用无感知 SPGD 细化全频随机相位。

    目标函数与仿真一致: ``0.5*PIB + 0.5/(1+CV)``, 读取真实 CCD 帧计算。
    """
    if params.cam_type == "daheng":
        # Resolve first: it rejects 0 / negative / unparseable with the shared
        # message naming `auto`. Both tokens are then refused, because the
        # single legacy rung they replaced (a non-positive float) was refused
        # here too -- refusing only `auto` would have opened the camera where
        # the old code errored out.
        exposure = resolve_exposure(
            params.exposure_time_ms, param_hint="--exposure-time-ms"
        )
        if isinstance(exposure, ExposureChoice):
            raise click.BadParameter(
                "set an explicit positive exposure in ms for Daheng "
                f"(ceiling: AO_DAHENG_MAX_EXPOSURE_MS, else probed from the "
                f"image); 'auto'/'{DEFAULT_EXPOSURE_TOKEN}' are not allowed here",
                param_hint="--exposure-time-ms",
            )
    debug = resolve_debug(ctx)
    root_dir = (
        Path(ctx.parent.obj.get("dir", "data"))
        if ctx.parent is not None and ctx.parent.obj is not None
        else Path("data")  # python -m 独立运行时无 main 组上下文
    )

    click.echo("=" * 64)
    click.echo("SLM GS Warm Start + Freeform SPGD Refinement")
    click.echo("=" * 64)
    click.echo(f"Camera: {params.cam_type} #{params.cam_id} size={params.cam_size}")
    click.echo(f"SLM: #{params.slm_number} @ {params.slm_wavelength}nm")
    click.echo(
        f"Epochs: {params.epochs}  grid: {params.phase_grid}x{params.phase_grid}"
    )
    click.echo(f"SPGD: delta={params.delta} lr={params.lr} opt={params.optimizer_type}")
    click.echo(
        f"GS warm start: {'on' if params.gs_warm_start else 'off'} "
        f"(iters={params.gs_iters})"
    )
    if params.gs_method != "plain":
        click.echo(
            f"GS strategy: {params.gs_method}  gamma={params.gs_gamma} "
            f"clip={params.gs_weight_clip} beta={params.gs_beta} "
            f"hio_fraction={params.gs_hio_fraction}"
        )
    # focal_length_m defaults to 0 = "derive from the measured focal scale and the
    # camera pixel pitch", so echo what will actually be used rather than the raw
    # 0 (the config resolves it in __post_init__).
    from ao_shaping.drivers.slm.factory import DEFAULT_WAVELENGTH_NM
    from ao_shaping.optimizer.wfless.slm_gs_refine import (
        _TILT_SHIFT_SCALE_MEASURED_AT_NM,
        _TILT_SHIFT_SCALE_PX,
        focal_length_from_camera_pixel,
    )

    _f_m = params.focal_length_m
    _f_note = "pinned"
    if _f_m <= 0.0:
        _f_m = focal_length_from_camera_pixel(
            wavelength_nm=float(params.slm_wavelength or DEFAULT_WAVELENGTH_NM),
            camera_pixel_um=params.camera_pixel_um,
            slm_pixel_um=params.panel_pixel_um,
            focal_scale_px=_TILT_SHIFT_SCALE_PX,
            scale_measured_at_nm=_TILT_SHIFT_SCALE_MEASURED_AT_NM,
        )
        _f_note = "derived"
    click.echo(
        f"Bench model: beam_r={params.beam_radius_px}px panel={params.panel_pixel_um}um "
        f"cam={params.camera_pixel_um}um f={_f_m:.5f}m ({_f_note}) "
        f"pad={params.far_field_padding}"
    )
    click.echo(
        f"Wavelength: {params.slm_wavelength or 'ask device'} nm "
        f"(focal scale {_TILT_SHIFT_SCALE_PX:.0f} measured @ "
        f"{_TILT_SHIFT_SCALE_MEASURED_AT_NM:.0f}nm)"
    )
    click.echo(
        f"Target side: {params.target_side or 'auto'} px (factor {params.side_factor})"
    )
    click.echo(f"SLM pupil center: {params.pupil_center or 'panel center'}")
    click.echo(
        f"Objective: 0.5-style composite w_pib={params.w_pib} w_unif={params.w_unif}"
    )
    if params.cam_type != "sim":
        click.echo("!" * 64)
        click.echo("硬件模式: 请确认光斑已对准 SLM 且 CCD 已就位。")
        click.echo("Ctrl+C 会在当前 epoch 结束后释放设备。")
        click.echo("!" * 64)
    click.echo("=" * 64)

    from ao_shaping.optimizer.wfless.slm_gs_refine import optimize_slm_gs_refine

    config = build_gs_refine_config(params, debug=debug)

    recorder = optimize_slm_gs_refine(config)

    best_iter, (best_epoch, best_score) = recorder.get_best_iter()
    stage_scores = {
        row["stage"]: row["score"]
        for row in recorder.history
        if row["stage"] in ("flat", "gs")
    }

    click.echo("\n" + "=" * 64)
    click.echo("Optimization Complete")
    click.echo("=" * 64)
    click.echo(f"Evaluations: {len(recorder.history)}")
    if "flat" in stage_scores:
        click.echo(f"Flat baseline : {stage_scores['flat']:.4f}")
    if "gs" in stage_scores:
        verdict = (
            "adopted"
            if stage_scores["gs"] > stage_scores["flat"]
            else "rejected (kept flat)"
        )
        click.echo(f"GS warm start : {stage_scores['gs']:.4f}  [{verdict}]")
    click.echo(
        f"Best score    : {best_score:.4f} (stage={best_iter['stage']}, epoch={best_epoch})"
    )
    click.echo(f"  PIB: {best_iter.get('pib', float('nan')):.4f}")
    click.echo(f"  CV : {best_iter.get('cv', float('nan')):.4f}")

    save_result = save_gs_refine_results(recorder, params, root_dir=root_dir, debug=debug)
    click.echo(f"History saved: {save_result.csv_file}")
    if save_result.phase_file is not None:
        click.echo(
            f"Best phase saved: {save_result.phase_file} "
            "(raw radians; display via Santec.create_phase_from_array)"
        )
    if save_result.image_file is not None:
        click.echo(f"Best image saved: {save_result.image_file}")
    if debug:
        click.echo(f"Debug data saved to: {save_result.debug_png}")


if __name__ == "__main__":
    run()
