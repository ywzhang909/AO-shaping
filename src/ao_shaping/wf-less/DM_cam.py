import os
from loguru import logger
import datetime
import json
import tqdm
import argparse

import numpy as np
import pandas as pd

import pygame
import matplotlib.pyplot as plt

from ao_shaping.drivers import CameraStreamManager, NlightDM

# display settings
VOLT_HEIGHT = 200
LOG_J_HEIGHT = 200
# 定义背景颜色
BACKGROUND_COLOR = (0, 0, 0)
# 定义折线颜色
LINE_COLOR = (0, 255, 0)

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
IMG_SIZE = (250, 250)

# dm parameters
KEEP_VOLTAGE_WHEN_EXIT = True

def gen_file_name(dir, postfix: str = ''):
    if not os.path.exists(dir):
        os.makedirs(dir)
    fname = os.listdir(dir)
    if postfix:
        fname = len([_ for _ in fname if _.endswith(postfix)]) + 1
    else:
        fname = len(fname) + 1

    if not postfix:  # make dir
        path = os.path.join(dir, str(fname))
        if not postfix and not os.path.exists(path):
            os.makedirs(path)
    else:
        if postfix[0] != ".":
            postfix = "." + postfix
        path = os.path.join(dir, str(fname)) + postfix
    return path


def render(window, img, log, center, r, info="", img_size=IMG_SIZE) -> None:
    canvas = pygame.surfarray.make_surface(img.transpose())
    pygame.draw.circle(canvas, (255, 0, 0), center, r, 1)
    pygame.display.set_caption(info)
    window.blit(canvas, (0,0))
    # 绘制电压图
    # 清空之前绘制的条形统计图
    plot_area = pygame.Rect(0, img_size[1], img_size[0], VOLT_HEIGHT)
    window.fill(BACKGROUND_COLOR, plot_area)
    volts = log[-1]['_v']
    bar_width = int(img_size[0] / len(volts))
    for i,value in enumerate(volts):
        normed_v = (value-NlightDM.V_Min)/(NlightDM.V_Max-NlightDM.V_Min)
        color = (int(normed_v*255), int((1-normed_v)*255), 0)
        x = int(i * bar_width)
        y = int(img_size[1] + VOLT_HEIGHT)
        height = int((value / NlightDM.V_Max) *  VOLT_HEIGHT)
        pygame.draw.line(window, color, (x, y), (x, y - height), bar_width)
    
    pygame.event.pump()
    pygame.display.update()


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
            NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT) as dm:
        # dm.reset_all()

        if not init_v:
            _init_v = np.zeros(dm.DM_Num, dtype=np.float64)
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 1)

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
            total_height = VOLT_HEIGHT + LOG_J_HEIGHT + (cam.cam_height or 400)
            pygame.init()
            window = pygame.display.set_mode((cam.cam_width or 640, total_height))

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
                pos_img = cam.get_numpy_image(8)
                pos_j = calc_j(pos_img)

                dm.send_voltages(_init_v - disturb_v)
                neg_img = cam.get_numpy_image(8)
                neg_j = calc_j(neg_img)

                # if (pos_j + neg_j) == 0 and CAM_EXP_TIME_ADJ_RATE > 1:
                #     cam.reset_explore_time(cam.explore_time * CAM_EXP_TIME_ADJ_RATE)
                #     continue

                diff = pos_j - neg_j
                gradient = -diff * disturb_v

                if epoch == 1:
                    m = np.zeros_like(_init_v, dtype=np.float64)
                    v = np.zeros_like(_init_v, dtype=np.float64)
                    s = 0

                m = beta1 * m + (1 - beta1) * (gradient)
                v = beta2 * v + (1 - beta2) * (gradient**2)

                m_hat = m / (1 - beta1 ** (epoch + 1))
                v_hat = v / (1 - beta2 ** (epoch + 1))

                gamma = lr / (np.sqrt(v_hat) + 1e-8)
                s = beta3 * s + (1 - beta3) * gamma
                learning_rate = np.where(gamma<s, gamma, s)
                update = learning_rate * m_hat

                _to_update_v = np.clip(_init_v - update, dm.V_Min, dm.V_Max)
                if dm.check_dm_unit_grad_safe(_to_update_v):
                    _init_v = _to_update_v
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
                    render(
                        window, pos_img, history, center, r_bucket, f"{epoch}: PIB={log['pib']:.3f}", img_size
                    )
                
                bar.set_postfix({k: v for k, v in log.items() if k[0] != "_"})
                bar.update(1)

        return history

def run(args:argparse.Namespace):
    root_dir = args.root_dir
    res_list = optimizer(**args.__dict__)
    # 保存结果
    res_df = pd.DataFrame(res_list)
    saved_file_name = gen_file_name(os.path.join(root_dir,'wf-less'), 'pkl')
    res_df.to_pickle(saved_file_name, compression='zip')
    max_j_id = res_df['pib'].argmax()
    last_V = res_df.iloc[max_j_id]["_v"]
    max_j = res_df.iloc[max_j_id]['pib']
    logger.info(f"{max_j_id} -> {max_j}")

    saved_dir = f'data/flatten_voltages/{datetime.datetime.now().strftime("%Y%m%d")}'
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    np.savetxt(f'{saved_dir}/to_load_V-{max_j}.csv', np.around(last_V).astype(int), fmt="%d")

    with open(saved_file_name+'-args.json', 'w' ,encoding='utf8') as f:
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
        plt.savefig(saved_file_name+'-plot.png')
        plt.close()
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--root_dir", type=str, default="data", help="数据保存根目录 (default: data)")
    args.add_argument("--cam_id", type=int, default=1, help="远场光斑CCD设备ID (default: 1)")
    args.add_argument("--center", type=tuple, default=(665, 415), help="远场光斑CCD中心位置 (default: (665, 415))")
    args.add_argument("--exposure_time_ms", type=int, default=50, help="远场光斑CCD曝光时间 (毫秒) (default: 60)")
    args.add_argument("--epochs", type=int, default=4_000, help="优化迭代次数 (default: 4000)")
    args.add_argument("--r_bucket", type=float, default=18, help="渲染半径桶大小 (default: 18)")
    args.add_argument("--delta", type=float, default=2, help="优化步长 (default: 2)")
    args.add_argument("--lr", type=float, default=2, help="优化学习率 (default: 2)")
    args.add_argument("--shrank_iter", type=int, default=300, help="优化迭代次数后收缩半径桶和步长 (default: 300)")
    args.add_argument("--show", type=bool, default=True, help="显示远场光斑CCD图像和优化历史 (default: True)")
    args.add_argument("--cam_size", type=int, default=250, help="相机开窗大小 (default: 250*250)")
    
    args = args.parse_args()
    run(args)
