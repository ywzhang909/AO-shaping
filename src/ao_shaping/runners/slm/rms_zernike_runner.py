from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from ao_shaping.algorithm.gradient.adam import search_optimal_delta
from ao_shaping.drivers import MlaRes, ThorlabWFS
from ao_shaping.drivers.slm import ZernikeSLM
from ao_shaping.optimizer.wf.rms_by_zernike import optimizer_rms_slm
from ao_shaping.runners.runner_common import (
    RmsZernikeParams,
    WfsParams,
    ZernikeSlmParams,
    with_params,
)
from ao_shaping.utils.io.cli_helpers import (
    resolve_debug,
    setup_coredumpy,
)
from ao_shaping.utils.io.file import (
    build_debug_save_paths,
    save_optimization_debug_artifacts,
)
from ao_shaping.utils.wavefront.matrix_utils import calc_n_zernike_terms


def _auto_delta_detect_rms(
    min_delta: float = 0.1,
    max_delta: float = 2.0,
    delta_step: float = 0.2,
    n_directions: int = 3,
    pupil_center: tuple[float, float] = (0, 0),
    pupil_diameter: float = 4.6,
    wfs_exposure_time: float = 0.0,
    wavelength: int = 532,
    shift_x: int = 0,
    shift_y: int = 0,
    n_max: int = 4,
    wfs_res: MlaRes = MlaRes.Res1024,
    remove_tilt: bool = False,
    slm_number: int = 1,
) -> tuple[float, dict]:
    n_zernike = calc_n_zernike_terms(n_max)

    with (
        ZernikeSLM(
            slm_number=slm_number,
            wavelength=wavelength,
            n_max=n_max,
            shift_x=shift_x,
            shift_y=shift_y,
        ) as slm,
        ThorlabWFS(
            wfs_res,
            exposure_time=wfs_exposure_time,
            use_custom_ref=False,
            high_speed=True,
            pupil_diameter=pupil_diameter,
            pupil_center=pupil_center,
        ) as wfs,
    ):

        def objective_fn(params: np.ndarray) -> float:
            wfs.take_image(3)
            wf, statics = wfs.get_wavefront(cancel_tile=remove_tilt)
            return statics.get("rms", np.inf)

        def apply_fn(params: np.ndarray) -> None:
            slm.send_zernike(params)

        perturb_mask = np.ones(n_zernike, dtype=np.float64)
        if n_zernike > 0:
            perturb_mask[0] = 0

        best_delta, info = search_optimal_delta(
            param_dim=n_zernike,
            objective_fn=objective_fn,
            apply_fn=apply_fn,
            min_delta=min_delta,
            max_delta=max_delta,
            n_magnitude_steps=max(5, int(delta_step)),
            n_samples_per_delta=n_directions,
            clip_min=-50.0,
            clip_max=50.0,
            perturb_mask=perturb_mask,
            verbose=True,
        )

        return best_delta, {
            "baseline_rms": info["baseline_obj"],
            "best_rms": info.get(
                "best_obj", info["baseline_obj"] - info.get("optimal_delta", 0)
            ),
            "best_delta": info["optimal_delta"],
            "all_results": info.get("fine_results", []),
        }


@click.command()
@click.pass_context
@with_params(RmsZernikeParams, kw_name="params")
@with_params(WfsParams, kw_name="wfs")
@with_params(ZernikeSlmParams, kw_name="slm")
def run(
    ctx: click.Context,
    params: RmsZernikeParams,
    wfs: WfsParams,
    slm: ZernikeSlmParams,
) -> None:
    """Zernike波前优化器 (基于SLM的RMS最小化)

    使用Zernike多项式通过SLM进行波前校正，最小化WFS测量的波前RMS值。
    ``--algorithm spgd`` 使用梯度/SPGD；其他取值 (ga/pso/sa/hc/rs/cem/de) 使用
    algorithm/heuristic 的启发式搜索。

    调试模式: ``main.py --debug rms-zernike`` / 本命令 ``--debug`` / ``DEBUG=1``。
    """
    debug = resolve_debug(ctx, params.debug)

    delta = params.delta
    if delta <= 0:
        delta, delta_info = _auto_delta_detect_rms(
            min_delta=params.min_delta,
            max_delta=params.max_delta,
            delta_step=params.delta_step,
            n_directions=params.n_directions,
            pupil_center=wfs.pupil_center,
            pupil_diameter=wfs.pupil_diameter,
            wfs_exposure_time=wfs.exposure_time_ms,
            wavelength=slm.wavelength,
            shift_x=slm.shift_x,
            shift_y=slm.shift_y,
            n_max=params.n_max,
            wfs_res=MlaRes.from_str(wfs.wfs_res),
            remove_tilt=wfs.remove_tilt,
            slm_number=slm.slm_number,
        )
        click.echo(
            f"Detected optimal delta: {delta:.2f} (baseline RMS: {delta_info['baseline_rms']:.4f}, best RMS: {delta_info['best_rms']:.4f})"
        )
        click.echo(
            "Note: The optimizer will use its internal scheduler, but this detection provides guidance on optimal perturbation amplitudes."
        )

    init_v = [0 for _ in range(calc_n_zernike_terms(params.n_max))]
    records = optimizer_rms_slm(
        init_z=init_v,
        epochs=params.epochs,
        delta=delta,
        lr=params.lr,
        pupil_center=wfs.pupil_center,
        pupil_diameter=wfs.pupil_diameter,
        early_stop_threshold=params.early_stop_threshold,
        wavelength=slm.wavelength,
        shift_x=slm.shift_x,
        shift_y=slm.shift_y,
        n_max=params.n_max,
        wfs_res=wfs.wfs_res,
        wfs_exposure_time=wfs.exposure_time_ms,
        remove_tilt=wfs.remove_tilt,
        slm_number=slm.slm_number,
        slm_wait_time=slm.wait_time,
        n_init_positions=params.n_init_positions,
        init_range=params.init_range,
        lr_schedule=params.lr_schedule,
        lr_min=params.lr_min,
        delta_schedule=params.delta_schedule,
        delta_min=params.delta_min,
        optimizer_type=params.optimizer,
        beta1=params.beta1,
        weight_decay=params.weight_decay,
        mini_batch=params.mini_batch,
        gradient_clip=params.gradient_clip,
        stagnation_patience=params.stagnation_patience,
        stagnation_delta_boost=params.stagnation_delta_boost,
        freeze_high_order_threshold=params.freeze_threshold,
        early_stop_window=params.early_stop_window,
        early_stop_min_epochs=params.early_stop_min_epochs,
        early_stop_patience=params.early_stop_patience,
        n_frames=params.n_frames,
        algorithm=params.algorithm,
        pop_size=params.pop_size,
    )
    root_dir = Path(params.dir)

    min_iter, (min_epoch, min_rms) = records.get_best_iter()
    save_dir, _ = build_debug_save_paths(root_dir, "flatten_zernike")
    records.save_best(saved_dir=save_dir, target="_c", process_fn=np.round, fmt="%.6f")
    records.save_array_sidecars(save_dir)
    if debug:
        _, saved_file_name = build_debug_save_paths(root_dir, "flatten_zernike")

        save_optimization_debug_artifacts(
            records=records,
            save_dir=save_dir,
            saved_file_name=saved_file_name,
            min_epoch=min_epoch,
            min_metric=min_rms,
            best_coeff_key="_c",
            init_wavefront=records.first["_wavefront"][0],
            opt_wavefront=min_iter["_wavefront"][1],
            init_title="Init wavefront",
            opt_title="Opt wavefront",
            plot_params_note=f"Min RMS: {min_rms:.3f} @ epoch {min_epoch}",
        )

        has_intensity = "_pos_intensity" in records.first
        if has_intensity:
            all_intensities = []
            all_dev_x = []
            all_dev_y = []
            for rec in records.history:
                if "_pos_intensity" in rec and rec["_pos_intensity"] is not None:
                    all_intensities.append(rec["_pos_intensity"])
                if "_pos_dev_x" in rec and rec["_pos_dev_x"] is not None:
                    all_dev_x.append(rec["_pos_dev_x"])
                if "_pos_dev_y" in rec and rec["_pos_dev_y"] is not None:
                    all_dev_y.append(rec["_pos_dev_y"])
            if all_intensities:
                np.savez_compressed(
                    save_dir / "wfs_debug_data.npz",
                    intensities=np.array(all_intensities),
                    dev_x=np.array(all_dev_x) if all_dev_x else np.array([]),
                    dev_y=np.array(all_dev_y) if all_dev_y else np.array([]),
                )
                click.echo(f"WFS debug数据已保存: {save_dir / 'wfs_debug_data.npz'}")

    click.echo(f"波前优化完成，最优RMS值: {min_rms:.4f} @ epoch {min_epoch}")


if __name__ == "__main__":
    setup_coredumpy()
    run()