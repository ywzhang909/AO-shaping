import os
import sys
import json
import re
from datetime import datetime

import argparse
import coredumpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ao_shaping.optimizer.wfless.pib import optimize_pib
from ao_shaping.utils.file import gen_file_path_uuid, gen_date_dir, logger

def parse_tuple(s):
    if s is None:
        return None
    # 移除空格和括号
    s_clean = re.sub(r'[()\s]', '', s)
    try:
        parts = s_clean.split(',')
        if len(parts) != 2:
            raise ValueError("Must have exactly two integers")
        x, y = map(int, parts)
        return (x, y)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid center format: {s}. Expected formats: 'x,y' or '(x,y)'"
        )

def run(args:argparse.Namespace): 
    root_dir = args.root_dir
    load_file = args.load_file
    if load_file:
        last_v = np.loadtxt(load_file)
        args.init_v = last_v.tolist()
    logger.info(args.__dict__)

    res_list = optimize_pib(
        center=args.center,
        r_bucket=args.r_bucket,
        epochs=args.epochs,
        delta=args.delta,
        lr=args.lr,
        exposure_time_ms=args.exposure_time_ms,
        shrink_iter=args.shrink_iter,
        shrink_ratio=args.shrink_ratio,
        cam_id=args.cam_id,
        show=args.show,
        init_v=args.init_v,
        cam_size=args.cam_size,
    )
    # 保存结果
    res_df = res_list.dataframe
    max_j_id = res_df['pib'].argmax()
    last_V = res_df.iloc[max_j_id]["_v"]
    max_j = res_df.iloc[max_j_id]['pib']
    logger.info(f"{max_j_id} -> {max_j}")

    saved_dir = f'{root_dir}/flatten_voltages/{datetime.now().strftime("%Y%m%d")}'
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    np.savetxt(f'{saved_dir}/to_load_V-{max_j}.csv', np.around(last_V).astype(int), fmt="%d")
        
    if args.debug:
        save_dir = gen_date_dir(f'{root_dir}/wf-less')
        saved_file_name = gen_file_path_uuid(save_dir, 'pkl')
        res_df.to_pickle(saved_file_name, compression='zip')

        with open(saved_file_name.with_suffix('.json'), 'w' ,encoding='utf8') as f:
            json.dump(args.__dict__, f, ensure_ascii=False, indent=4)

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        # init image
        ax[0, 0].imshow(res_df.iloc[0]["_img"])
        ax[0, 0].set_title(f"Init Image, pib={res_df.iloc[0]['pib']:.3f}")
        ax[0, 0].axis("off")
        # best image
        ax[0, 1].imshow(res_df.iloc[max_j_id]["_img"])
        ax[0, 1].set_title(f"Best Image, pib={max_j:.3f}")
        ax[0, 1].axis("off")
        # pib history
        ax[1, 0].plot(res_df["pib"])
        ax[1, 0].set_title("PIB History")
        ax[1, 0].set_xlabel("Epoch")
        ax[1, 0].set_ylabel("PIB")
        # best voltages plot bar
        ax[1, 1].bar(range(64), last_V)
        ax[1, 1].set_title("Best Voltages")
        ax[1, 1].set_xlabel("Unit ID")
        ax[1, 1].set_ylabel("Voltage")
            
        plt.tight_layout()
        plt.savefig(saved_file_name.with_suffix('.png'))
        plt.close()
    
def init():
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--root_dir", type=str, default="data", help="数据保存根目录 (default: data)")
    args.add_argument("-f", "--load_file", type=str, default=None, help="加载优化结果文件 (default: None)")
    args.add_argument("--cam_id", type=int, default=os.environ.get('Far_Cam_ID', 0), help="远场光斑CCD设备ID (default: Far_Cam_ID/0)")
    args.add_argument("-c", "--center", type=parse_tuple, default=None, help="场光斑CCD中心位置 (default: (665, 403))")
    args.add_argument("-t","--exposure_time_ms", type=int, default=800, help="远场光斑CCD曝光时间 (毫秒) (default: 60)")
    args.add_argument("-e", "--epochs", type=int, default=4_000, help="优化迭代次数 (default: 4000)")
    args.add_argument("-r", "--r_bucket", type=float, default=18, help="渲染半径桶大小 (default: 18)")
    args.add_argument("--delta", type=float, default=2, help="优化步长 (default: 2)")
    args.add_argument("--lr", type=float, default=2, help="优化学习率 (default: 2)")
    args.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减 (default: 0.0)")
    args.add_argument("--shrink_iter", type=int, default=300, help="优化迭代次数后收缩半径桶和步长 (default: 300)")
    args.add_argument("--shrink_ratio", type=float, default=0.8, help="收缩半径桶和步长比例 (default: 0.8)")
    args.add_argument("-s", "--cam_size", type=int, default=200, help="相机开窗大小 (default: 200*200)")
    args.add_argument("--debug", action='store_true', default=False, help="是否开启调试模式 (default: False)")
    args.add_argument("--show", action='store_true', default=False, help="显示远场光斑CCD图像和优化历史 (default: False)")

    args = args.parse_args()
    coredumpy.patch_except(directory='logs/debug/error')

    run(args)

    if args.debug:
        coredumpy.dump(directory='logs/debug')

    sys.exit(0)

if __name__ == "__main__":
    init()
