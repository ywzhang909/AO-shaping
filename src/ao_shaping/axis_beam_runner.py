import os
import json
import re
from datetime import datetime

import click
import coredumpy

import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.optimizer.wfless.pib import optimize_pib
from ao_shaping.utils.file import gen_file_path_uuid, gen_date_dir, get_init_V_by_rms, logger
from ao_shaping.utils.display import plot_funcs


def parse_tuple(ctx, param, value):
    """解析元组格式的参数，支持 'x,y' 或 '(x,y)' 格式"""
    if value is None:
        return None
    if value.lower() in ["mass", "max", "shape"]:
        return value.lower()
    # 移除空格和括号
    s_clean = re.sub(r'[()\s]', '', str(value))
    try:
        parts = s_clean.split(',')
        if len(parts) != 2:
            raise ValueError("Must have exactly two integers")
        x, y = map(int, parts)
        return (x, y)
    except Exception as e:
        raise click.BadParameter(
            f"Invalid center format: {value}. Expected formats: 'x,y' or '(x,y)'"
        )


@click.command()
@click.option("-d", "--root_dir", default="data", help="数据保存根目录 (default: data)")
@click.option("-f", "--load_file", default='rms', help="加载优化结果文件 (default: None), 若为'rms'，则使用RMS优化结果初始化")
@click.option("--cam_id", default=lambda: os.environ.get('Far_Cam_ID', 0), help="远场光斑CCD设备ID (default: Far_Cam_ID/0)")
@click.option("-c", "--center", callback=parse_tuple, default="mass", help="场光斑CCD中心位置 (example: 665,403)")
@click.option("-t","--exposure_time_ms", default=60, help="远场光斑CCD曝光时间 (毫秒) (default: 60)")
@click.option("-e", "--epochs", default=4_000, help="优化迭代次数 (default: 4000)")
@click.option("-r", "--r_bucket", default=0, help="渲染半径桶大小 (default: 0，环围半径)")
@click.option("--delta", default=2, help="优化步长 (default: 2)")
@click.option("--lr", default=0.0, help="优化学习率 (default: None，表示基于环围半径动态学习率衰减)")
@click.option("--weight_decay", default=0.0, help="权重衰减 (default: 0.0)")
@click.option("--shrink_iter", default=200, help="优化迭代次数后收缩半径桶和步长 (default: 300)")
@click.option("--shrink_ratio", default=0.8, help="收缩半径桶和步长比例 (default: 0.8)")
@click.option("-s", "--cam_size", default=200, help="相机开窗大小 (default: 200*200)")
@click.option("-b", "--target_max_brightness", default=90, help="目标最大亮度值 (default: 90)")
@click.option("--debug", is_flag=True, help="是否开启调试模式 (default: False)")
@click.option("--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)")
def run(root_dir, load_file, cam_id, center, exposure_time_ms, epochs, r_bucket, 
        delta, lr, weight_decay, shrink_iter, shrink_ratio, cam_size, target_max_brightness, debug, show):
    """轴向光束优化器"""
    
    
    # 处理初始电压
    if load_file.lower() == 'rms':
        init_v = get_init_V_by_rms()
    elif os.path.exists(load_file):
        last_v = np.loadtxt(load_file)
        init_v = last_v.tolist()
    else:
        logger.warning(f"load_file {load_file} not exists")
        init_v = []

    config = {
        'root_dir': root_dir,
        'load_file': load_file,
        'cam_id': cam_id,
        'center': center,
        'exposure_time_ms': exposure_time_ms,
        'target_max_brightness': target_max_brightness,
        'epochs': epochs,
        'r_bucket': r_bucket,
        'delta': delta,
        'lr': lr,
        'weight_decay': weight_decay,
        'shrink_iter': shrink_iter,
        'shrink_ratio': shrink_ratio,
        'cam_size': cam_size,
        'debug': debug,
        'show': show
    }
    logger.info(config)

    dm_unit_mask = np.ones(64, dtype=bool)
    dm_unit_mask[0] = False
    # dm_unit_mask[38:] = False
    res_list = optimize_pib(
        center=center,
        r_bucket=r_bucket,
        epochs=epochs,
        delta=delta,
        lr=lr,
        exposure_time_ms=exposure_time_ms,
        shrink_iter=shrink_iter,
        shrink_ratio=shrink_ratio,
        cam_id=cam_id,
        show=show,
        init_v=init_v,
        cam_size=cam_size,
        target_max_brightness=target_max_brightness,
        dm_unit_mask=dm_unit_mask,
        dm_neibor_diff=400,
        dm_max_voltage=300,
        dm_min_voltage=-200,
    )
    # 保存结果
    res_df = res_list.dataframe

    saved_dir = f'{root_dir}/flatten_voltages/{datetime.now().strftime("%Y%m%d")}'
    res_list.save_best(saved_dir, target="_v",
                        process_fn=lambda x: np.around(x).astype(int), fmt="%d")
    best_iter, (max_j_id, max_j) = res_list.get_best_iter()    
    if debug:
        save_dir = gen_date_dir(f'{root_dir}/wf-less')
        saved_file_name = gen_file_path_uuid(save_dir, 'pkl')
        res_df.to_pickle(saved_file_name, compression='zip')

        with open(saved_file_name.with_suffix('.json'), 'w' ,encoding='utf8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        # init image
        plot_funcs["img"](res_df.iloc[0]["_img"], ax[0, 0], f"Init Image, pib={res_df.iloc[0]['pib']:.3f}")
        # best image
        axim = plot_funcs["img"](res_df.iloc[max_j_id]["_img"], ax[0, 1], f"Best Image, pib={max_j:.3f}")
        cbar = fig.colorbar(axim, ax=[ax[0, 0], ax[0, 1]], orientation='horizontal')
        # pib history
        plot_funcs["pib_history"](res_df["pib"], ax[1, 0])
        # best voltages plot bar
        plot_funcs["voltages"](best_iter["_v"], ax[1, 1], "Best Voltages")
        
        plt.savefig(saved_file_name.with_suffix('.png'))
        plt.close()
    
    click.echo(f"轴向光束优化完成，最优PIB值: {max_j:.4f} @ epoch {max_j_id}")


if __name__ == "__main__":
    try:
        coredumpy.patch_except(directory='logs/debug/error')
    except:
        logger.error("coredumpy初始化失败")
        pass  # 忽略coredumpy初始化错误
    run()
