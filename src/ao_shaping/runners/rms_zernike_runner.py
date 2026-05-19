import click
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ao_shaping.utils import gen_date_dir, gen_file_path_uuid
from ao_shaping.utils.display import plot_funcs
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms
from ao_shaping.utils.cli_helpers import parse_tuple, setup_coredumpy, get_date_dir_name, get_debug_mode
from ao_shaping.drivers import MlaRes, Thorlab_WFS
from ao_shaping.drivers.slm import ZernikeSLM
from ao_shaping.algorithm.adam import search_optimal_delta
from ao_shaping.optimizer.wf.rms_by_zernike import optimizer_rms


def _get_wfs_res(res_str: str) -> MlaRes:
    res_map = {
        '320': MlaRes.Res320,
        '512': MlaRes.Res512,
        '768': MlaRes.Res768,
        '1024': MlaRes.Res1024,
        '1280': MlaRes.Res1280,
    }
    return res_map.get(res_str, MlaRes.Res1024)


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
        Thorlab_WFS(
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
            return statics.get('rms', np.inf)
        
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
            'baseline_rms': info['baseline_obj'],
            'best_rms': info.get('best_obj', info['baseline_obj'] - info.get('optimal_delta', 0)),
            'best_delta': info['optimal_delta'],
            'all_results': info.get('fine_results', []),
        }


@click.command()
@click.option("-d", "--dir", default="data", help="数据保存根目录 (default: data)")
@click.option("-e", "--epochs", default=20000, help="优化迭代次数 (default: 20000)")
@click.option("-n", "--n-max", default=4, help="Zernike最大阶数 (default: 4)")
@click.option("--lr", default=0.01, help="学习率 (default: 0.01)")
@click.option("--delta", default=0.0, help="初始delta值 (default: 0.0)")
@click.option("-r", "--wfs_res", default='1024', help="WFS分辨率 (default: 1024)")
@click.option("-p", "--pupil_diameter", default=2.7, help="瞳孔直径 (default: 2.7)")
@click.option("-c", "--pupil_center", callback=parse_tuple, default="(0,0)", help="瞳孔中心坐标 (default: (0,0))")
@click.option("--exposure-time-ms", default=0.0, type=float, help="WFS曝光时间 (毫秒, default: 0.0=自动曝光)")
@click.option("-t", "--early_stop_threshold", default=0.12, help="早停阈值 (default: 0.12)")
@click.option("--wavelength", default=532, help="SLM波长 (nm, default: 532)")
@click.option("--shift-x", default=0, help="SLM X方向平移 (像素, default: 0)")
@click.option("--shift-y", default=0, help="SLM Y方向平移 (像素, default: 0)")
@click.option("--wait-time", default=0.3, help="SLM 液晶翻转等待时间(秒, default: 0.3) ")
@click.option("--slm-number", default=1, help="SLM设备编号 (default: 1)")
@click.option("--remove-tilt", is_flag=True, help="移除波前测量中的倾斜项")
@click.option("--min-delta", default=0.01, help="自动检测最小delta (数量级扫描, default: 0.01)")
@click.option("--max-delta", default=100.0, help="自动检测最大delta (数量级扫描, default: 100.0)")
@click.option("--delta-step", default=5, help="数量级扫描步数 (用于细粒度扫描, default: 5)")
@click.option("--n-directions", default=5, help="每个delta采样次数防噪声 (default: 5)")
@click.option("--n-init-positions", default=0, help="多起点优化：随机初始位置数量 (default: 0, 禁用)")
@click.option("--init-range", default=1.0, help="多起点初始化的随机范围 (default: 1.0)")
@click.option("--lr-schedule", default="static", type=click.Choice(["static", "cosine", "exp", "linear"]), help="学习率调度类型 (default: static)")
@click.option("--lr-min", default=1e-6, type=float, help="学习率最小值 (default: 1e-6)")
@click.option("--delta-schedule", default="static", type=click.Choice(["static", "cosine", "exp", "linear"]), help="Delta调度类型 (default: static)")
@click.option("--delta-min", default=1e-7, type=float, help="Delta最小值 (default: 1e-7)")
@click.option("--optimizer", default="adamod", type=click.Choice(["adamod", "adamw"]), help="优化器类型 (default: adamod)")
@click.option("--beta1", default=0.95, type=float, help="Adam beta1参数 (default: 0.95)")
@click.option("--weight-decay", default=1e-2, type=float, help="AdamW权重衰减 (default: 1e-2)")
@click.option("--mini-batch", default=1, type=int, help="SPGD mini-batch大小 (default: 1)")
@click.option("--gradient-clip", default=0.0, type=float, help="梯度裁剪阈值 (default: 0.0, 禁用)")
@click.option("--stagnation-patience", default=30, type=int, help="停滞检测轮数 (default: 30)")
@click.option("--stagnation-delta-boost", default=1.5, type=float, help="停滞时delta倍增 (default: 1.5)")
@click.option("--freeze-threshold", default=None, type=float, help="冻结高阶模式阈值 (default: None)")
@click.option("--early-stop-window", default=0, type=int, help="早停滑动窗口大小 (default: 0)")
@click.option("--early-stop-min-epochs", default=0, type=int, help="早停最小轮数 (default: 0)")
@click.option("--early-stop-patience", default=0, type=int, help="早停耐心值 (default: 0)")
@click.option("--n-frames", default=10, type=int, help="WFS帧平均数 (default: 10)")
def run(dir, epochs, lr, delta, n_max, wfs_res, pupil_diameter, pupil_center, early_stop_threshold,
        exposure_time_ms, wavelength, shift_x, shift_y, slm_number, remove_tilt, wait_time,
        min_delta, max_delta, delta_step, n_directions,
        n_init_positions, init_range,
        lr_schedule, lr_min, delta_schedule, delta_min,
        optimizer, beta1, weight_decay, mini_batch, gradient_clip,
        stagnation_patience, stagnation_delta_boost, freeze_threshold,
        early_stop_window, early_stop_min_epochs, early_stop_patience, n_frames):
    """Zernike波前优化器 (基于SLM的RMS最小化)

    使用Zernike多项式通过SLM进行波前校正，最小化WFS测量的波前RMS值。
    DEBUG环境变量控制调试模式。
    """
    debug = get_debug_mode()
    
    if delta > 0:
        pass
    else:
        delta, delta_info = _auto_delta_detect_rms(
            min_delta=min_delta,
            max_delta=max_delta,
            delta_step=delta_step,
            n_directions=n_directions,
            pupil_center=pupil_center,
            pupil_diameter=pupil_diameter,
            wfs_exposure_time=exposure_time_ms,
            wavelength=wavelength,
            shift_x=shift_x,
            shift_y=shift_y,
            n_max=n_max,
            wfs_res=_get_wfs_res(wfs_res),
            remove_tilt=remove_tilt,
            slm_number=slm_number,
        )
        click.echo(f"Detected optimal delta: {delta:.2f} (baseline RMS: {delta_info['baseline_rms']:.4f}, best RMS: {delta_info['best_rms']:.4f})")
        click.echo("Note: The optimizer will use its internal scheduler, but this detection provides guidance on optimal perturbation amplitudes.")

    init_v = [0 for _ in range(calc_n_zernike_terms(n_max))]
    records = optimizer_rms(
        init_z=init_v,
        epochs=epochs,
        delta=delta,
        lr=lr,
        pupil_center=pupil_center,
        pupil_diameter=pupil_diameter,
        early_stop_threshold=early_stop_threshold,
        wavelength=wavelength,
        shift_x=shift_x,
        shift_y=shift_y,
        n_max=n_max,
        wfs_res=wfs_res,
        wfs_exposure_time=exposure_time_ms,
        remove_tilt=remove_tilt,
        slm_number=slm_number,
        slm_wait_time=wait_time,
        n_init_positions=n_init_positions,
        init_range=init_range,
        lr_schedule=lr_schedule,
        lr_min=lr_min,
        delta_schedule=delta_schedule,
        delta_min=delta_min,
        optimizer_type=optimizer,
        beta1=beta1,
        weight_decay=weight_decay,
        mini_batch=mini_batch,
        gradient_clip=gradient_clip,
        stagnation_patience=stagnation_patience,
        stagnation_delta_boost=stagnation_delta_boost,
        freeze_high_order_threshold=freeze_threshold,
        early_stop_window=early_stop_window,
        early_stop_min_epochs=early_stop_min_epochs,
        early_stop_patience=early_stop_patience,
        n_frames=n_frames,
    )
    root_dir = Path(dir)

    min_iter, (min_epoch, min_rms) = records.get_best_iter()
    save_dir = root_dir / "flatten_zernike" / get_date_dir_name()
    records.save_best(saved_dir=save_dir, target="_c", process_fn=np.round, fmt="%.6f")
    records.save_array_sidecars(save_dir)
    if debug:
        saved_file_name = gen_file_path_uuid(save_dir, 'pkl')
        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        rms_values = records.get_sublist()
        plot_funcs["rms_history"](rms_values, ax[0, 0], min_epoch, min_rms)
        plot_funcs["voltages"](min_iter["_c"], ax[0, 1], f"Min RMS: {min_rms:.3f} @ epoch {min_epoch}")
        im = plot_funcs["wavefront"](records.first["_wavefront"][0], ax[1, 0], "Init wavefront")
        plt.colorbar(im, ax=ax[1, 0], orientation='horizontal')
        im = plot_funcs["wavefront"](min_iter["_wavefront"][1], ax[1, 1], "Opt wavefront")
        plt.colorbar(im, ax=ax[1, 1], orientation='horizontal')

        plt.savefig(saved_file_name.with_suffix('.png'))
        plt.close()
        records.save_dataframe(saved_file_name.with_suffix('.zip'), sidecar_dir=save_dir, compression='zip')

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
