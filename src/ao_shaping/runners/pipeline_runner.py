import os

import click
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.utils import gen_date_dir, gen_file_path_uuid, logger
from ao_shaping.utils.display import plot_funcs
from ao_shaping.optimizer.wf.rms import optimizer_rms
from ao_shaping.optimizer.wfless.pib import optimize_pib
from ao_shaping.utils.cli_helpers import setup_coredumpy, get_date_dir_name
from ao_shaping.config import DM_N_ACTUATORS
from ao_shaping.utils.cli_helpers import get_debug_mode


@click.command()
@click.option("-d", "--dir", default="data", help="数据保存根目录 (default: data)")
@click.option("-f", "--load_file", default=None, help="加载优化结果文件 (default: None)")
@click.option("-e", "--epochs", default=8_000, help="优化迭代次数 (default: 8000)")
@click.option("-E", "--wf_epochs", default=8_000, help="WF优化迭代次数 (default: 8000)")
@click.option("-R", "--wfs_res", type=click.Choice(['768', '512']), default='768', help="WFS分辨率 (default: 768)")
@click.option("-p", "--pupil_diameter", default=2.7, help="瞳孔直径 (default: 2.7)")
@click.option("-c", "--cam_id", default=lambda: os.environ.get('Far_Cam_ID', 0), help="远场光斑CCD设备ID (default: Far_Cam_ID/0)")
@click.option("-t", "--exposure_time_ms", default=0, help="远场光斑CCD曝光时间 (毫秒) (default: 0，自动选取曝光)")
@click.option("-s", "--cam_size", default=160, help="相机开窗大小 (default: 160)")
@click.option("-r", "--rms_threshold", default=0.12, help="RMS阈值 (default: 0.12)")
@click.option("-u", "--dm_unit_mask", type=click.Choice(['all','inner','outer']), default='all', help="DM单元掩码 (default: all)")
def run(dir, load_file, epochs, wf_epochs, wfs_res, pupil_diameter, cam_id, exposure_time_ms, cam_size, rms_threshold, dm_unit_mask):
    """串行优化器（先波前优化，再轴向光束优化）

    DEBUG环境变量控制调试模式。
    """
    debug = get_debug_mode()
    if load_file:
        last_v = np.loadtxt(load_file)
        init_v = last_v.tolist()
    else:
        init_v = []

    wf_records = optimizer_rms(
        init_v=init_v,
        pupil_diameter=pupil_diameter,
        wfs_res=wfs_res,
        early_stop_threshold=rms_threshold,
        epochs=wf_epochs)

    min_iter, (min_epoch, min_rms) = wf_records.get_best_iter()
    logger.info(f"WF优化完成，最佳迭代：{min_epoch}，RMS：{min_rms:.4f}")

    if debug:
        init_wf, min_wf = wf_records.first["_wavefront"][0], min_iter["_wavefront"][0]

    dm_available = np.ones(DM_N_ACTUATORS, dtype=bool)
    dm_available[0] = False
    if dm_unit_mask == 'inner':
        dm_available[21:] = False
    elif dm_unit_mask == 'outer':
        dm_available[:39] = False

    ccd_records = optimize_pib(
        cam_id=cam_id, center="mass",
        exposure_time_ms=exposure_time_ms,
        cam_size=cam_size,
        dm_unit_mask=dm_available,
        target_max_brightness=0,
        epochs=epochs, lr=0.9, delta=0.9,
        shrink_iter=20, shrink_ratio=0.8, r_bucket=int(os.environ.get("IDEAL_SPOT_RADIUS", 6)),
        init_v=min_iter["_v"], show=False)
    max_pid_iter, (max_epoch, max_pib) = ccd_records.get_best_iter()
    last_V = max_pid_iter["_v"]

    save_dir = Path(dir) / "flatten_voltages" / get_date_dir_name()
    def np_array_to_int(arr):
        return arr.astype(int)
    wf_records.save_best(saved_dir=save_dir, target="_v", process_fn=np_array_to_int, fmt="%d")
    ccd_records.save_best(saved_dir=save_dir, target="_v", process_fn=np_array_to_int, fmt="%d")

    if debug:
        fig, ax = plt.subplots(2, 4, figsize=(12, 8))
        rms_values = wf_records.get_sublist()
        plot_funcs["rms_history"](rms_values, ax[0, 0], min_epoch, min_rms)
        pib_values = ccd_records.get_sublist()[1:]
        plot_funcs["pib_history"](pib_values, ax[1, 0])
        axim = plot_funcs["wavefront"](init_wf, ax[0, 1], "Init WF")
        plt.colorbar(axim, ax=ax[0, 1], orientation='vertical', fraction=0.046, pad=0.04)
        axim = plot_funcs["wavefront"](min_wf, ax[1, 1], "Best WF")
        plt.colorbar(axim, ax=ax[1, 1], orientation='vertical')
        axim1 = plot_funcs["img"](ccd_records.first["_img"], ax[0, 2], "Init CCD")
        axim2 = plot_funcs["img"](max_pid_iter["_img"], ax[1, 2], "Best CCD")
        plt.colorbar(axim2, ax=ax[1, 2], fraction=0.046, pad=0.04, orientation='vertical')
        plot_funcs["voltage_comparison"](init_v, last_V, ax[0, 3], "Voltage Comparison")
        voltages = np.array(wf_records.get_sublist("_v")+ccd_records.get_sublist("_v")[1:])
        plot_funcs["voltage_heatmap"](voltages, ax[1, 3], "Voltage History")

        plt.tight_layout()
        save_dir = gen_date_dir(f'{dir}/pipeline')
        saved_file_name = gen_file_path_uuid(save_dir)
        wf_records.save_dataframe(saved_file_name.with_suffix('.wfs.pkl'), compression='zip')
        ccd_records.save_dataframe(saved_file_name.with_suffix('.ccd.pkl'), compression='zip')
        plt.savefig(saved_file_name.with_suffix('.png'))

        with open(saved_file_name.with_suffix('.json'), 'w' ,encoding='utf8') as f:
            import json
            json.dump({
                'dir': dir,
                'load_file': load_file,
                'epochs': epochs,
                'wfs_res': wfs_res,
                'pupil_diameter': pupil_diameter,
                'cam_id': cam_id,
                'exposure_time_ms': exposure_time_ms,
                'cam_size': cam_size,
                'rms_threshold': rms_threshold,
                'debug': debug
            }, f, ensure_ascii=False, indent=4)

    click.echo(f"组合优化完成，最优RMS值: {min_rms:.4f} @ epoch {min_epoch}, 最优PIB值: {max_pib:.4f} @ epoch {max_epoch}")


if __name__ == "__main__":
    setup_coredumpy()
    run()
