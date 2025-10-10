import os
import time
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tqdm
import pygame
import utils
from drivers import CameraStreamManager, NlightDM

ROOT_DIR = r"D:\ao-project\data"

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
Rho_0 = 0.99

# metropolis parameters
METROPOLIS_ALPHA = 0.8

# camera parameters
CAM_EXP_TIME = 100
CAM_EXP_TIME_ADJ_RATE = 0
IMG_SIZE = (200, 200)

# dm parameters
KEEP_VOLTAGE_WHEN_EXIT = True

V_MAX = 499
V_MIN = -300
UPDATE_MAX = 20
DM_Adj = np.loadtxt('data\dm_adj.txt')
Tolerance = 300
print(f"相邻单元矩阵加载完成:{DM_Adj.shape},最大压差:{Tolerance}")

def check_dm_unit_grad_safe(vs, adj_mat=DM_Adj, tolerance=Tolerance):
    assert len(vs) == DM_Adj.shape[0] == DM_Adj.shape[1]
    diff_mat = (vs[:,None] - vs[None,:]) * adj_mat
    return not np.any(diff_mat[diff_mat > tolerance])


def render(window, img, log, center, r, info="") -> None:
    canvas = pygame.surfarray.make_surface(img.transpose())
    pygame.draw.circle(canvas, (255, 0, 0), center, r, 1)
    pygame.display.set_caption(info)
    window.blit(canvas, (0,0))
    
    # 绘制电压图
    # 清空之前绘制的条形统计图
    plot_area = pygame.Rect(0, IMG_SIZE[1], IMG_SIZE[0], VOLT_HEIGHT)
    window.fill(BACKGROUND_COLOR, plot_area)
    volts = log[-1]['_v']
    bar_width = int(IMG_SIZE[0] / len(volts))
    for i,value in enumerate(volts):
        normed_v = (value-V_MIN)/(-V_MIN+V_MAX)
        color = (int(normed_v*255), int((1-normed_v)*255), 0)
        x = int(i * bar_width)
        y = int(IMG_SIZE[1] + VOLT_HEIGHT)
        height = int((value / V_MAX) *  VOLT_HEIGHT)
        pygame.draw.line(window, color, (x, y), (x, y - height), bar_width)
    
    pygame.event.pump()
    pygame.display.update()


def optimizer(
    r_bucket,
    epochs,
    delta=1,
    lr=1,
    weight_decay=0.001,
    shrank_iter=0,
    weights_class=1,
    pid_weighted_ratio=1.0,
    show=True,
    init_v=None,
    center="max"
):
    delta = abs(delta)
    epochs = int(epochs)

    with CameraStreamManager(cam_id=0, explosure_time=CAM_EXP_TIME, skip_sampling=True) as cam,\
            NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT) as dm:
        # dm.reset_all()

        if init_v is None:
            _init_v = np.zeros(dm.dm_num, dtype=np.float64)
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 1)
        init_img = cam.get_numpy_image()

        # center = utils.centroid(init_img, cam.xv, cam.yv, 30)
        if center == 'max':
            center = np.unravel_index(np.argmax(init_img), init_img.shape)
            center = (center[1], center[0])
        elif center == 'centroid':
            center = utils.centroid(init_img, cam.xv, cam.yv, 30)
        else:
            center = center

        print(f"{center=}")
        img_size, center = cam.reset_window(center, IMG_SIZE)

        init_img = cam.get_numpy_image()
        xv, yv = np.meshgrid(
            np.arange(-img_size[0] // 2, img_size[0] // 2),
            np.arange(-img_size[1] // 2, img_size[1] // 2),
        )

        imgmesh_dist = (xv) ** 2 + (yv) ** 2
        weighted_mask = ( - imgmesh_dist/np.max(imgmesh_dist) + 1) ** (weights_class)
        bucket_mask = imgmesh_dist < r_bucket**2
        fix_r_mask = imgmesh_dist < 20**2
        
        if show:
            total_height = VOLT_HEIGHT + LOG_J_HEIGHT + cam.cam_height
            pygame.init()
            window = pygame.display.set_mode((cam.cam_width, total_height))

        def calc_pib(img, r):
            if shrank_iter <= 0:
                return np.sum(img[bucket_mask]) / np.sum(bucket_mask)
            in_power = np.sum(img[imgmesh_dist < r**2])
            return in_power

        def calc_weighted_power(img) -> float:
            return np.sum(weighted_mask * img) / np.sum(weighted_mask)
        def calc_j(img):
            weighted_ratio = (0, 10 * pid_weighted_ratio, 10 * (1-pid_weighted_ratio)) # PIB越大越集中，锥形权越大约均匀
            _pib_r_20 = np.sum(img[fix_r_mask])*weighted_ratio[1] / np.sum(fix_r_mask) * weighted_ratio[0]\
                if weighted_ratio[0] else 0
            _pib = calc_pib(img, r_bucket) * weighted_ratio[1] if weighted_ratio[1] else 0
            _wp = calc_weighted_power(img) * weighted_ratio[-1] if weighted_ratio[-1] else 0
            return _pib+_wp+_pib_r_20
            
            # return  np.max(img)
        
        history = [
            {
                "J": calc_j(init_img),
                "_v": _init_v,
                "_img": init_img,
                "_diff": 0,
                "gamma": lr,
                "r": r_bucket,
                "_delta": delta,
                "_epoch": -1,
            }
        ]
        with tqdm.tqdm(
            total=epochs, desc="iter {epochs}", dynamic_ncols=True
        ) as bar:
            for epoch in range(epochs):
                disturb_v = np.random.binomial(1, 0.5, (dm.dm_num,)).astype(float) * 2.0 - 1.0

                disturb_v = disturb_v * delta
                disturb_v[0] = 0

                dm.send_voltages(_init_v + disturb_v, 0.005)
                img = cam.get_numpy_image(2)
                pos_j = calc_j(img)

                dm.send_voltages(_init_v - disturb_v, 0.005)
                img = cam.get_numpy_image(2)
                neg_j = calc_j(img)

                if (pos_j + neg_j) == 0 and CAM_EXP_TIME_ADJ_RATE > 1:
                    cam.reset_explore_time(cam.explore_time * CAM_EXP_TIME_ADJ_RATE)
                    continue

                gradient = np.sign(pos_j - neg_j) * disturb_v
                _to_update_v = _init_v + gradient * lr
                
                if check_dm_unit_grad_safe(_to_update_v):
                    _init_v = _to_update_v
                else:
                    print("相邻单元压差过大，放弃本次结果")

                log = {
                    "J": (pos_j + neg_j) / 2,
                    "_gamma": lr,
                    "r": r_bucket,
                    "_delta": delta,
                    "_epoch": epoch,
                    "_v": _init_v,
                    "_img": img
                }
                history.append(log)
                # earlying schedule
                if shrank_iter > 0 and epoch % shrank_iter == shrank_iter - 1:
                    r_bucket = max(0.8 * r_bucket, 5)
                    delta = max(delta * 0.8, 0.7)
                    # pid_weighted_ratio = min(pid_weighted_ratio * 0.7, 0)

                if np.sum(img[img == 255]) / 255 > 2 and CAM_EXP_TIME_ADJ_RATE > 1:
                    print(
                        f"exp time = {cam.reset_explore_time(cam.explore_time * CAM_EXP_TIME_ADJ_RATE)}"
                    )

                if show:
                    render(
                        window, img, history, center, r_bucket, f"{epoch}: J={log['J']:.3f}"
                    )
                

                bar.set_postfix({k: v for k, v in log.items() if k[0] != "_"})
                bar.update(1)

        return history


def run():
    # init_V = np.load('last_v.npz')['v'] \
    #                  if os.path.exists("last_v.npz") else None
    # init_V = np.random.random((64,))*100 - 50
    init_V = np.zeros((64,))
    init_V[0] = 0

    res_list = optimizer(
        init_v=init_V.copy(),
        epochs=10000, 
        r_bucket=10, 
        delta=1, # 扰动要和桶半径匹配，不要扰动会导致质心出桶
        lr=1, 
        weights_class=1,
        pid_weighted_ratio=1,
        shrank_iter=0,
        center=(776, 470))
    
    res_df = pd.DataFrame(res_list)
    res_df.to_pickle('record.pkl', compression=None)
    max_j_id = -1
    last_V = res_df.iloc[max_j_id]["_v"]
    print(f"{max_j_id} -> {res_df.iloc[max_j_id]['J']}")
    
    def get_nerbors(unit_id):
        return (a for a in np.where(DM_Adj[unit_id, :] == 1)[0])
    
    base_unit_id = np.argmin(np.abs(last_V[1:]))+1
    checked_mask = np.zeros_like(DM_Adj, dtype=bool)
    def reset_nerbors(unit_id, v):
        min, max = v[unit_id]-Tolerance, v[unit_id]+Tolerance
        for nerbor in get_nerbors(unit_id):
            if not checked_mask[unit_id, nerbor]:
                v[nerbor] = np.clip(v[nerbor], min, max)
                checked_mask[unit_id, nerbor] = checked_mask[nerbor, unit_id] = True
                reset_nerbors(nerbor, v)

    reset_nerbors(base_unit_id, last_V)
    np.savez('last_v', v=last_V)
    np.savetxt('to_load_V.csv', np.around(last_V), fmt="%d")

    fig, ax = plt.subplots(2, 1)
    ax[0].bar(x=np.arange(64)-0.25, height=init_V, width=0.5)
    ax[0].bar(x=np.arange(64)+0.25, height=last_V, width=0.5)
    ax[1].plot(res_df['J'])
    plt.show()

if __name__ == "__main__":
    run()
