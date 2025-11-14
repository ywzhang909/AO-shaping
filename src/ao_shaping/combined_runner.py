from typing import Literal
import os, sys

import argparse
import coredumpy
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from ao_shaping.utils import gen_date_dir, gen_file_path_uuid
from ao_shaping.optimizer.wf.rms import optimizer_rms
from ao_shaping.optimizer.wfless.pib import optimize_pib

def run(args):
    if args.load_file:
        last_v = np.loadtxt(args.load_file)
        args.init_v = last_v.tolist()
    else:
        args.init_v = []

    wf_records = optimizer_rms(
        init_v=args.init_v,
        pupil_diameter=args.pupil_diameter,
        wfs_res=args.wfs_res,
        early_stop_threshold=args.rms_threshold,
        epochs=20_000)

    min_iter, (min_epoch, min_rms) = wf_records.get_best_iter()
    init_v = min_iter["_v"]
    init_wf, min_wf = wf_records.first["_wavefront"][0], min_iter["_wavefront"][0]

    dm_available = np.ones(64, dtype=bool)
    dm_available[0] = False
    dm_available[21:] = False

    ccd_records = optimize_pib(
        cam_id=args.cam_id, center=None, exposure_time_ms=args.exposure_time_ms, cam_size=args.cam_size,
        dm_unit_mask=dm_available,
        epochs=args.epochs, lr=0.9, delta=0.9, shrink_iter=20, shrink_ratio=0.8,
        init_v=init_v, show=False)
    max_pid_iter, (max_epoch, max_pib) = ccd_records.get_best_iter()
    last_V = max_pid_iter["_v"]
    if args.debug:
        fig, ax = plt.subplots(2, 4, figsize=(12, 8))
        # rms history
        ax[0, 1].plot(wf_records.get_sublist())
        ax[0, 1].scatter(min_epoch, min_rms, color='r', marker='*', label='Min RMS')
        ax[0, 1].text(min_epoch, min_rms, f"{min_rms:.4f}", color='r')
        ax[0, 1].set_title("RMS History")
        ax[0, 1].set_xlabel("Epoch")
        ax[0, 1].set_ylabel("RMS")
        # pib history
        ax[1, 1].plot(ccd_records.get_sublist()[1:])
        ax[1, 1].scatter(max_epoch, max_pib, color='r', marker='*', label='Max PIB')
        ax[1, 1].text(max_epoch, max_pib, f"{max_pib:.4f}", color='r')
        ax[1, 1].set_title("PIB History")
        ax[1, 1].set_xlabel("Epoch")
        ax[1, 1].set_ylabel("PIB")
        # init wf
        ax[0, 2].imshow(init_wf)
        ax[0, 2].set_title("Init WF")
        ax[0, 2].set_xlabel("Pixel ID")
        ax[0, 2].set_ylabel("Amplitude")
        # best wf
        ax[1, 2].imshow(min_wf)
        ax[1, 2].set_title("Best WF")
        ax[1, 2].set_xlabel("Pixel ID")
        ax[1, 2].set_ylabel("Amplitude")
        # best voltages plot bar
        ax[0, 3].bar(range(64), init_v, color='r')
        ax[0, 3].bar(range(64), last_V, color='b')
        ax[0, 3].set_title("Best Voltages")
        ax[0, 3].set_xlabel("Unit ID")
        ax[0, 3].set_ylabel("Voltage")
        # voltage history
        voltages = np.array(wf_records.get_sublist("_v")+ccd_records.get_sublist("_v")[1:])
        ax[1, 3].imshow(voltages.T, aspect='auto')
        ax[1, 3].set_title("Voltage History")
        ax[1, 3].set_xlabel("Epoch")
        ax[1, 3].set_ylabel("Voltage")
            
        plt.tight_layout()
        save_dir = gen_date_dir(f'{args.root_dir}/{__file__}')
        saved_file_name = gen_file_path_uuid(save_dir)
        wf_records.save_dataframe(saved_file_name.with_suffix('.wfs.pkl'), compression='zip')
        ccd_records.save_dataframe(saved_file_name.with_suffix('.ccd.pkl'), compression='zip')

        with open(saved_file_name.with_suffix('.json'), 'w' ,encoding='utf8') as f:
            import json
            json.dump(args.__dict__, f, ensure_ascii=False, indent=4)
        
    # 在data/flatten_voltages下生成名称为当前日期的目录并保存电压
    dir_name = datetime.now().strftime("%Y%m%d")
    save_dir = args.root_dir / "flatten_voltages" / dir_name
    wf_records.save_best(saved_dir=save_dir, target="_v", process_fn=round, fmt="%d")
    ccd_records.save_best(saved_dir=save_dir, target="_v", process_fn=round, fmt="%d")

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="data")
    parser.add_argument("-f", "--load_file", type=str, default=None, help="加载优化结果文件 (default: None)")
    parser.add_argument("-e", "--epochs", type=int, default=8_000)
    parser.add_argument("-R", "--wfs_res", type=str, choices=['768', '512'], default='768')
    parser.add_argument("-p", "--pupil_diameter", type=float, default=2.7)
    parser.add_argument("-c", "--cam_id", type=str, default=os.environ['Far_Cam_ID'])
    parser.add_argument("-t", "--exposure_time_ms", type=int, default=500)
    parser.add_argument("-s", "--cam_size", type=int, default=160)
    parser.add_argument("-r", "--rms_threshold", type=float, default=0.12)
    parser.add_argument("--debug", action='store_true', default=False, help="是否开启调试模式 (default: False)")
    # parser.add_argument("--show", action='store_true', default=False, help="显示远场光斑CCD图像和优化历史 (default: False)")
    args = parser.parse_args()
    coredumpy.patch_except(directory='logs/debug/error')

    run(args)

    if args.debug:
        coredumpy.dump(directory='logs/debug')

    sys.exit(0)

if __name__ == "__main__":
    init()