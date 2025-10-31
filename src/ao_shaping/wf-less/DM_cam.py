import os
from loguru import logger
import datetime
import json
import tqdm
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from ao_shaping.drivers import CameraStreamManager, NlightDM
from ao_shaping.utils.file import gen_file_path_uuid, gen_date_dir
from ao_shaping.utils.display import ImageVoltagesDisplay

# adam parameters
beta1 = 0.9
beta2 = 0.99
beta3 = 0.9995

# cool_momentum_spgd parameters
Rho_0 = 0.9

# metropolis parameters
METROPOLIS_ALPHA = 0.8

# camera parameters
CAM_EXP_TIME_ADJ_RATE = 0
CAM_SAMPLE_ITER = 10

# dm parameters
KEEP_VOLTAGE_WHEN_EXIT = True


def optimizer(
    center,
    r_bucket,
    epochs,
    delta=1,
    lr=1,
    exposure_time_ms=80,
    shrank_iter=0,
    shrank_ratio=0.9,
    cam_id=0,
    show=True,
    init_v=[],
    cam_size=250,
    **kwargs
):
    delta = abs(delta)
    epochs = int(epochs)
    
    with CameraStreamManager(cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=False) as cam,\
            NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT, max_neibor_diff=200) as dm:
        # dm.reset_all()

        if not init_v:
            _init_v = np.zeros(dm.DM_Num, dtype=np.float64)
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 1)
        
        m = np.zeros_like(_init_v, dtype=np.float64)
        v = np.zeros_like(_init_v, dtype=np.float64)
        s = 0
        
        # 使用传入的相机尺寸参数
        img_size = (cam_size, cam_size)
        img_size, _ = cam.reset_window(center, img_size)
        init_img = cam.get_numpy_image(1)
        img_size = init_img.shape[::-1]
        xv, yv = np.ogrid[-img_size[0]//2:img_size[0]//2, -img_size[1]//2:img_size[1]//2]

        imgmesh_dist = ((xv) ** 2 + (yv) ** 2).transpose()
        dist = np.sqrt(imgmesh_dist)
        bucket_mask = dist < r_bucket
        pib_mask = dist < 5
        
        if show:
            window = ImageVoltagesDisplay(img_size)
            window.init_window()

        def calc_pib(img, r):
            if shrank_iter <= 0:
                return np.sum(img[bucket_mask]).astype(float)
            in_power = np.sum(img[dist < r]).astype(float)
            return in_power
        
        def test_pib(img):
            return np.sum(img[pib_mask]).astype(float)
        
        def calc_j(img):
            return calc_pib(img, r_bucket)

        init_pid = test_pib(init_img)
        history = [
            {
                "J": calc_j(init_img),
                "pib": init_pid,
                "_v": _init_v,
                "_img": init_img,
                "_diff": 0,
                "gamma": lr,
                "r": r_bucket,
                "delta": delta,
                "_epoch": 0,
            }
        ]
        with tqdm.tqdm(
            total=epochs, desc=f"iter {epochs}", dynamic_ncols=True
        ) as bar:
            for epoch in range(1,epochs+1):
                disturb_v = np.random.binomial(1, 0.5, (dm.DM_Num,)).astype(float) * 2.0 - 1.0

                disturb_v = disturb_v * delta
                disturb_v[0] = 0.0

                dm.send_voltages(_init_v + disturb_v)
                pos_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                pos_j = calc_j(pos_img)

                dm.send_voltages(_init_v - disturb_v)
                neg_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                neg_j = calc_j(neg_img)

                # if (pos_j + neg_j) == 0 and CAM_EXP_TIME_ADJ_RATE > 1:
                #     cam.reset_explore_time(cam.explore_time * CAM_EXP_TIME_ADJ_RATE)
                #     continue

                diff = pos_j - neg_j
                gradient = -diff * disturb_v
                m = beta1 * m + (1 - beta1) * (gradient)
                v = beta2 * v + (1 - beta2) * (gradient**2)
                m_hat = m / (1 - beta1 ** (epoch))
                v_hat = v / (1 - beta2 ** (epoch))
                gamma = lr / (np.sqrt(v_hat) + 1e-8)
                s = beta3 * s + (1 - beta3) * gamma
                learning_rate = np.where(gamma<s, gamma, s)
                update = _init_v - learning_rate * m_hat
                   
                if dm.check_dm_unit_grad_safe(update):
                    _init_v = update
                else:
                    logger.warning("相邻单元压差过大，放弃本次结果")

                log = {
                    "J": (pos_j + neg_j) / 2,
                    "pib": test_pib(pos_img),
                    "_diff": diff,
                    "gamma": lr,
                    "r": r_bucket,
                    "delta": delta,
                    "_epoch": epoch,
                    "_v": _init_v,
                    "_img": pos_img,
                }
                history.append(log)
                # earlying schedule
                if shrank_iter > 0 and epoch % shrank_iter == shrank_iter - 1:
                    r_bucket = max(shrank_ratio * r_bucket, 4.0)
                    delta = max(delta * shrank_ratio, 0.6)
                    lr = max(lr * shrank_ratio, 0.8)

                if show:
                    window.render(
                        pos_img, _init_v, dm.V_Min, dm.V_Max, center, r_bucket, f"{epoch}: PIB={log['pib']:.3f}"
                    )
                
                bar.set_postfix({k: v for k, v in log.items() if k[0] != "_"})
                bar.update(1)

        return history

def run(args:argparse.Namespace): 
    opti_args = args.__dict__
    root_dir = opti_args.pop('root_dir')
    load_file = opti_args.pop('load_file')
    if load_file:
        last_v = np.loadtxt(load_file)
        opti_args['init_v'] = last_v.tolist()
    
    res_list = optimizer(**opti_args)
    # 保存结果
    res_df = pd.DataFrame(res_list)
    if root_dir:
        save_dir = gen_date_dir(root_dir)
        saved_file_name = gen_file_path_uuid(save_dir, 'pkl')
        res_df.to_pickle(saved_file_name, compression='zip')
    max_j_id = res_df['pib'].argmax()
    last_V = res_df.iloc[max_j_id]["_v"]
    max_j = res_df.iloc[max_j_id]['pib']
    logger.info(f"{max_j_id} -> {max_j}")

    saved_dir = f'data/flatten_voltages/{datetime.datetime.now().strftime("%Y%m%d")}'
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    np.savetxt(f'{saved_dir}/to_load_V-{max_j}.csv', np.around(last_V).astype(int), fmt="%d")

    with open(saved_file_name.with_suffix('.json'), 'w' ,encoding='utf8') as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=4)
        
    if args.show:
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
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--root_dir", type=str, default="data/wf-less", help="数据保存根目录 (default: data/wf-less)")
    args.add_argument("--load_file", type=str, default=None, help="加载优化结果文件 (default: None)")
    args.add_argument("--cam_id", type=int, default=1, help="远场光斑CCD设备ID (default: 1)")
    args.add_argument("--center", type=tuple, default=(662,420), help="远场光斑CCD中心位置 (default: (665, 403))")
    args.add_argument("--exposure_time_ms", type=int, default=50, help="远场光斑CCD曝光时间 (毫秒) (default: 60)")
    args.add_argument("--epochs", type=int, default=4_000, help="优化迭代次数 (default: 4000)")
    args.add_argument("--r_bucket", type=float, default=18, help="渲染半径桶大小 (default: 18)")
    args.add_argument("--delta", type=float, default=2, help="优化步长 (default: 2)")
    args.add_argument("--lr", type=float, default=2, help="优化学习率 (default: 2)")
    args.add_argument("--shrank_iter", type=int, default=300, help="优化迭代次数后收缩半径桶和步长 (default: 300)")
    args.add_argument("--show", type=bool, default=True, help="显示远场光斑CCD图像和优化历史 (default: True)")
    args.add_argument("--cam_size", type=int, default=200, help="相机开窗大小 (default: 200*200)")
    
    args = args.parse_args()
    run(args)
