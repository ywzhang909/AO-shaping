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
    python src/ao_shaping/main.py slm-gs-refine --cam_type sim -e 20

    # 大恒 CCD + Santec SLM #1 @1064nm
    python src/ao_shaping/main.py slm-gs-refine --cam_type daheng --cam-id 0 -e 400

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

2. **GS 预矫正 (bake-off)**: 用 bench 前向模型算 GS 相位并下发实测。若 GS 分数
   不高于平场就丢弃 GS、从平场继续 —— 模型参数 (相机像素间距 / 光斑半径) 错了
   时, 代价只是两次测量, 而不是整轮跑坏。

3. **自由相位 SPGD 细化**: 粗网格 (默认 24x24 = 576 自由度) 自由相位, 每个 epoch
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

结果输出 (默认 ``data/slm_gs_refine/<日期>/``):
- 优化历史 CSV (recorder, 每轮一行)
- 最优相位 ``.npy`` (raw 未包裹弧度, 1200x1920)
- 最优远场图 PNG (``--save-best-image``)
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from ao_shaping.runners.runner_common import SlmGsRefineParams, with_params
from ao_shaping.utils.io.cli_helpers import get_debug_mode


@click.command()
@click.pass_context
@with_params(SlmGsRefineParams, kw_name="params")
def run(ctx: click.Context, params: SlmGsRefineParams) -> None:
    """SLM GS 预整形 + 自由相位 SPGD 细化 (SLM + CCD 闭环)。

    把远场光斑整形为均匀方形。先用 GS 算一个开环预整形相位 (只在实测优于平场时
    采用), 再用无感知 SPGD 细化自由相位。

    目标函数与仿真一致: ``0.5*PIB + 0.5/(1+CV)``, 读取真实 CCD 帧计算。
    """
    debug = get_debug_mode()
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
    click.echo(f"Epochs: {params.epochs}  grid: {params.phase_grid}x{params.phase_grid}")
    click.echo(f"SPGD: delta={params.delta} lr={params.lr} opt={params.optimizer_type}")
    click.echo(
        f"GS warm start: {'on' if params.gs_warm_start else 'off'} "
        f"(iters={params.gs_iters})"
    )
    # focal_length_m defaults to 0 = "derive from the measured focal scale and the
    # camera pixel pitch", so echo what will actually be used rather than the raw
    # 0 (the config resolves it in __post_init__).
    from ao_shaping.optimizer.wfless.slm_gs_refine import (
        _DEFAULT_WAVELENGTH_NM,
        _TILT_SHIFT_SCALE_MEASURED_AT_NM,
        _TILT_SHIFT_SCALE_PX,
        focal_length_from_camera_pixel,
    )

    _f_m = params.focal_length_m
    _f_note = "pinned"
    if _f_m <= 0.0:
        _f_m = focal_length_from_camera_pixel(
            wavelength_nm=float(params.slm_wavelength or _DEFAULT_WAVELENGTH_NM),
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
    click.echo(f"Target side: {params.target_side or 'auto'} px (factor {params.side_factor})")
    click.echo(f"Objective: 0.5-style composite w_pib={params.w_pib} w_unif={params.w_unif}")
    if params.cam_type != "sim":
        click.echo("!" * 64)
        click.echo("硬件模式: 请确认光斑已对准 SLM 且 CCD 已就位。")
        click.echo("Ctrl+C 会在当前 epoch 结束后释放设备。")
        click.echo("!" * 64)
    click.echo("=" * 64)

    from ao_shaping.optimizer.wfless.slm_gs_refine import (
        SlmGsRefineConfig,
        optimize_slm_gs_refine,
    )
    from ao_shaping.utils.io.file import gen_date_dir, gen_date_str

    config = SlmGsRefineConfig(
        epochs=params.epochs,
        seed=params.seed,
        early_stop_score=params.early_stop_score,
        target_side=params.target_side,
        side_factor=params.side_factor,
        w_pib=params.w_pib,
        w_unif=params.w_unif,
        phase_grid=params.phase_grid,
        delta=params.delta,
        lr=params.lr,
        optimizer_type=params.optimizer_type,
        lr_schedule=params.lr_schedule,
        gs_iters=params.gs_iters,
        gs_warm_start=params.gs_warm_start,
        panel_pixel_um=params.panel_pixel_um,
        camera_pixel_um=params.camera_pixel_um,
        beam_radius_px=params.beam_radius_px,
        focal_length_m=params.focal_length_m,
        far_field_padding=params.far_field_padding,
        gs_target_side_px=params.gs_target_side_px,
        cam_type=params.cam_type,
        cam_id=params.cam_id,
        exposure_time_ms=params.exposure_time_ms,
        cam_size=params.cam_size,
        slm_number=params.slm_number,
        slm_wavelength=params.slm_wavelength,
        n_eval_frames=params.n_eval_frames,
        settle_wait_s=params.settle_wait_s,
        settle_tol=params.settle_tol,
        settle_max_wait_s=params.settle_max_wait_s,
    )

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
        verdict = "adopted" if best_iter["stage"] == "gs" else "rejected (kept flat)"
        click.echo(f"GS warm start : {stage_scores['gs']:.4f}  [{verdict}]")
    click.echo(f"Best score    : {best_score:.4f} (stage={best_iter['stage']}, epoch={best_epoch})")
    click.echo(f"  PIB: {best_iter.get('pib', float('nan')):.4f}")
    click.echo(f"  CV : {best_iter.get('cv', float('nan')):.4f}")

    save_dir = gen_date_dir(root_dir / "slm_gs_refine")
    csv_file = save_dir / f"slm_gs_refine_{gen_date_str()}.csv"
    recorder.save_dataframe(csv_file)
    click.echo(f"History saved: {csv_file}")

    best_phase = getattr(recorder, "best_phase", None)
    if best_phase is not None:
        phase_file = save_dir / "best_phase.npy"
        np.save(phase_file, np.asarray(best_phase, dtype=np.float64))
        click.echo(
            f"Best phase saved: {phase_file} "
            "(raw radians; display via Santec.create_phase_from_array)"
        )

    if params.save_best_image:
        img = getattr(recorder, "best_frame", None)
        if img is not None:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img, cmap="gray")
            ax.set_title(
                f"slm-gs-refine best: score={best_score:.4f} "
                f"PIB={best_iter.get('pib', 0):.4f} CV={best_iter.get('cv', 0):.4f}"
            )
            ax.axis("off")
            img_file = save_dir / "best_far_field.png"
            fig.savefig(img_file, dpi=150, bbox_inches="tight")
            plt.close(fig)
            click.echo(f"Best image saved: {img_file}")

    if debug:
        click.echo(f"Debug data saved to: {save_dir}")


if __name__ == "__main__":
    run()
