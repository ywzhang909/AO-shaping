import click
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.utils import gen_date_dir, gen_file_path_uuid
from ao_shaping.optimizer.wf.rms import optimizer_rms
from ao_shaping.utils.display import plot_funcs
from ao_shaping.utils.cli_helpers import parse_tuple, setup_coredumpy, get_date_dir_name
from ao_shaping.config import DM_N_ACTUATORS


@click.command()
@click.option("-d", "--dir", default="data", help="数据保存根目录 (default: data)")
@click.option("-e", "--epochs", default=20_000, help="优化迭代次数 (default: 20000)")
@click.option("-r", "--wfs_res", default='768', help="WFS分辨率 (default: 768)")
@click.option("-p", "--pupil_diameter", default=2.7, help="瞳孔直径 (default: 2.7)")
@click.option("-c", "--pupil_center", callback=parse_tuple, default="(0,0)", help="瞳孔中心坐标 (default: (0,0))")
@click.option("-t", "--early_stop_threshold", default=0.0, help="早停阈值 (default: 0.0)")
@click.option("--debug", is_flag=True, help="是否开启调试模式 (default: False)")
@click.option("--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)")
def run(dir, epochs, wfs_res, pupil_diameter, pupil_center, early_stop_threshold, debug, show):
    """波前优化器"""
    init_v = [0 for _ in range(DM_N_ACTUATORS)]
    records = optimizer_rms(
        init_v=init_v,
        epochs=epochs,
        wfs_res=wfs_res,
        pupil_diameter=pupil_diameter,
        pupil_center=pupil_center,
        early_stop_threshold=early_stop_threshold,
    )
    root_dir = Path(dir)

    min_iter, (min_epoch, min_rms) = records.get_best_iter()

    if debug:
        save_dir = gen_date_dir(root_dir / "wf")
        saved_file_name = gen_file_path_uuid(save_dir, 'pkl')

        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        rms_values = records.get_sublist()
        plot_funcs["rms_history"](rms_values, ax[0, 0], min_epoch, min_rms)
        plot_funcs["voltages"](min_iter["_v"], ax[0, 1], f"Min J: {min_rms:.3f} @ epoch {min_epoch}")
        im = plot_funcs["wavefront"](records.first["_wavefront"][0], ax[1, 0], "init wavefront")
        plt.colorbar(im, ax=ax[1, 0], orientation='horizontal')
        im = plot_funcs["wavefront"](min_iter["_wavefront"][1], ax[1, 1], "opt wavefront")
        plt.colorbar(im, ax=ax[1, 1], orientation='horizontal')

        plt.savefig(saved_file_name.with_suffix('.png'))
        plt.close()
        records.save_dataframe(saved_file_name.with_suffix('.zip'), compression='zip')

    save_dir = root_dir / "flatten_voltages" / get_date_dir_name()
    records.save_best(saved_dir=save_dir, target="_v", process_fn=np.round, fmt="%d")

    click.echo(f"波前优化完成，最优RMS值: {min_rms:.4f} @ epoch {min_epoch}")


if __name__ == "__main__":
    setup_coredumpy()
    run()
