import click
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.utils import gen_date_dir, gen_file_path_uuid
from ao_shaping.optimizer.wf.rms_by_zernike import optimizer_rms
from ao_shaping.utils.display import plot_funcs
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms
from ao_shaping.utils.cli_helpers import parse_tuple, setup_coredumpy, get_date_dir_name, get_debug_mode


@click.command()
@click.option("-d", "--dir", default="data", help="数据保存根目录 (default: data)")
@click.option("-e", "--epochs", default=20000, help="优化迭代次数 (default: 20000)")
@click.option("-n", "--n-max", default=4, help="Zernike最大阶数 (default: 4)")
@click.option("-r", "--wfs_res", default='1024', help="WFS分辨率 (default: 1024)")
@click.option("-p", "--pupil_diameter", default=2.7, help="瞳孔直径 (default: 2.7)")
@click.option("-c", "--pupil_center", callback=parse_tuple, default="(0,0)", help="瞳孔中心坐标 (default: (0,0))")
@click.option("-t", "--early_stop_threshold", default=0.12, help="早停阈值 (default: 0.12)")
@click.option("--wavelength", default=532, help="SLM波长 (nm, default: 532)")
@click.option("--shift-x", default=0, help="SLM X方向平移 (像素, default: 0)")
@click.option("--shift-y", default=0, help="SLM Y方向平移 (像素, default: 0)")
@click.option("--slm-number", default=1, help="SLM设备编号 (default: 1)")
@click.option("--remove-tilt", is_flag=True, help="移除波前测量中的倾斜项")
@click.option("--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)")
def run(dir, epochs, n_max, wfs_res, pupil_diameter, pupil_center, early_stop_threshold,
        wavelength, shift_x, shift_y, slm_number, remove_tilt, show):
    """Zernike波前优化器 (基于SLM的RMS最小化)

    使用Zernike多项式通过SLM进行波前校正，最小化WFS测量的波前RMS值。
    DEBUG环境变量控制调试模式。
    """
    debug = get_debug_mode()
    init_v = [0 for _ in range(calc_n_zernike_terms(n_max))]
    records = optimizer_rms(
        init_z=init_v,
        epochs=epochs,
        pupil_center=pupil_center,
        pupil_diameter=pupil_diameter,
        early_stop_threshold=early_stop_threshold,
        wavelength=wavelength,
        shift_x=shift_x,
        shift_y=shift_y,
        n_max=n_max,
        wfs_res=wfs_res,
        remove_tilt=remove_tilt,
        slm_number=slm_number,
    )
    root_dir = Path(dir)

    min_iter, (min_epoch, min_rms) = records.get_best_iter()

    if debug:
        save_dir = gen_date_dir(root_dir / "rms_zernike")
        saved_file_name = gen_file_path_uuid(save_dir, 'pkl')

        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        rms_values = records.get_sublist()
        plot_funcs["rms_history"](rms_values, ax[0, 0], min_epoch, min_rms)
        plot_funcs["voltages"](min_iter["_c"], ax[0, 1], f"Min RMS: {min_rms:.3f} @ epoch {min_epoch}")
        im = plot_funcs["wavefront"](records.first["_wavefront"][0], ax[1, 0], "初始波前")
        plt.colorbar(im, ax=ax[1, 0], orientation='horizontal')
        im = plot_funcs["wavefront"](min_iter["_wavefront"][1], ax[1, 1], "最优波前")
        plt.colorbar(im, ax=ax[1, 1], orientation='horizontal')

        plt.savefig(saved_file_name.with_suffix('.png'))
        plt.close()
        records.save_dataframe(saved_file_name.with_suffix('.zip'), compression='zip')

    save_dir = root_dir / "flatten_zernike" / get_date_dir_name()
    records.save_best(saved_dir=save_dir, target="_c", process_fn=np.round, fmt="%.6f")

    click.echo(f"波前优化完成，最优RMS值: {min_rms:.4f} @ epoch {min_epoch}")


if __name__ == "__main__":
    setup_coredumpy()
    run()
