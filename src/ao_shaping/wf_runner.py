import argparse
import coredumpy
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ao_shaping.utils import gen_date_dir, gen_file_path_uuid
from ao_shaping.optimizer.wf.rms import optimizer_rms

def run(args):
    # init_V = get_init_V_by_rms()
    init_V = [0 for _ in range(64)]
    records = optimizer_rms(
        init_v=init_V.copy(),
        epochs=args.epochs,
        wfs_res=args.wfs_res,
        pupil_diameter=args.pupil_diameter,
        early_stop_threshold=args.early_stop_threshold,
    )
    root_dir = Path(args.dir)

    min_iter, (min_epoch, min_rms) = records.get_best_iter()
    
    if args.debug:
        save_dir = gen_date_dir(root_dir / "wf")
        saved_file_name = gen_file_path_uuid(save_dir, 'pkl')

        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        # 绘制J的变化趋势
        ax[0, 0].plot(records.get_sublist())
        ax[0, 0].scatter(min_epoch, min_rms, color="red", marker="*", s=100)
        ax[0, 0].text(min_epoch, min_rms, f"Min J: {min_rms:.3f} @ epoch {min_epoch}")
        ax[0, 0].set_xlabel("Epoch")
        ax[0, 0].set_ylabel(f"{records.mark}")
        ax[0, 0].set_title(f"{records.mark} trend")
        # 绘制保存的电压
        ax[0, 1].bar(range(64), min_iter["_v"])
        ax[0, 1].set_xlabel("DM Unit")
        ax[0, 1].set_ylabel("Voltage")
        ax[0, 1].set_title(f"Min J: {min_rms:.3f} @ epoch {min_epoch}")
        # 绘制保存的初始波前
        ax[1, 0].imshow(records.first["_wavefront"][0], cmap='gray')
        ax[1, 0].set_title("init wavefront")
        ax[1, 0].axis('off')
        # 绘制保存的最优波前
        ax[1, 1].imshow(min_iter["_wavefront"][1], cmap='gray')
        ax[1, 1].set_title("opt wavefront")
        ax[1, 1].axis('off')
        
        plt.savefig(saved_file_name.with_suffix('.png'))
        plt.close()

        records.save_dataframe(saved_file_name.with_suffix('.zip'), compression='zip')
    
    # 在data/flatten_voltages下生成名称为当前日期的目录并保存电压
    dir_name = datetime.now().strftime("%Y%m%d")
    save_dir = root_dir / "flatten_voltages" / dir_name
    records.save_best(saved_dir=save_dir, target="_v", process_fn=np.round, fmt="%d")

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="data")
    parser.add_argument("-e", "--epochs", type=int, default=20_000)
    parser.add_argument("-r", "--wfs_res", type=str, default='768')
    parser.add_argument("-p", "--pupil_diameter", type=float, default=2.7)
    parser.add_argument("-t", "--early_stop_threshold", type=float, default=0.0)
    parser.add_argument("--debug", action='store_true', default=False, help="是否开启调试模式 (default: False)")
    parser.add_argument("--show", action='store_true', default=False, help="显示远场光斑CCD图像和优化历史 (default: False)")
    args = parser.parse_args()
    coredumpy.patch_except(directory='logs/debug/error')

    run(args)

if __name__ == "__main__":
    init()