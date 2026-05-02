import click
import re
import coredumpy
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.utils import gen_date_dir, gen_file_path_uuid, logger
from ao_shaping.optimizer.wf.rms_by_zernike import optimizer_rms
from ao_shaping.utils.display import plot_funcs
from ao_shaping.utils.matrix_utils import calc_n_zernike_terms


def parse_tuple(ctx, param, value):
    """解析元组格式的参数，支持 'x,y' 或 '(x,y)' 格式"""
    # 移除空格和括号
    s_clean = re.sub(r'[()\s]', '', str(value))
    try:
        parts = s_clean.split(',')
        if len(parts) != 2:
            raise ValueError("Must have exactly two integers")
        x, y = map(int, parts)
        return (x, y)
    except Exception:
        raise click.BadParameter(
            f"Invalid center format: {value}. Expected formats: 'x,y' or '(x,y)'"
        )


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
@click.option("--debug", is_flag=True, help="是否开启调试模式 (default: False)")
@click.option("--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)")
def run(dir, epochs, n_max, wfs_res, pupil_diameter, pupil_center, early_stop_threshold,
        wavelength, shift_x, shift_y, slm_number, remove_tilt, debug, show):
    """Zernike波前优化器 (基于SLM的RMS最小化)

    使用Zernike多项式通过SLM进行波前校正，最小化WFS测量的波前RMS值。
    """
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
        # 绘制RMS变化趋势
        rms_values = records.get_sublist()
        plot_funcs["rms_history"](rms_values, ax[0, 0], min_epoch, min_rms)
        # 绘制最优Zernike系数
        plot_funcs["voltages"](min_iter["_c"], ax[0, 1], f"Min RMS: {min_rms:.3f} @ epoch {min_epoch}")
        # 绘制初始波前
        im = plot_funcs["wavefront"](records.first["_wavefront"][0], ax[1, 0], "初始波前")
        plt.colorbar(im, ax=ax[1, 0], orientation='horizontal')
        # 绘制最优波前
        im = plot_funcs["wavefront"](min_iter["_wavefront"][1], ax[1, 1], "最优波前")
        plt.colorbar(im, ax=ax[1, 1], orientation='horizontal')

        plt.savefig(saved_file_name.with_suffix('.png'))
        plt.close()
        records.save_dataframe(saved_file_name.with_suffix('.zip'), compression='zip')

    # 保存最优Zernike系数
    dir_name = datetime.now().strftime("%Y%m%d")
    save_dir = root_dir / "flatten_zernike" / dir_name
    records.save_best(saved_dir=save_dir, target="_c", process_fn=np.round, fmt="%.6f")

    click.echo(f"波前优化完成，最优RMS值: {min_rms:.4f} @ epoch {min_epoch}")


if __name__ == "__main__":
    try:
        coredumpy.patch_except(directory='logs/debug/error')
    except:
        logger.error("coredumpy初始化失败")
    run()
