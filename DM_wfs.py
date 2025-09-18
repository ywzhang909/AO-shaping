import os
import time
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tqdm
import pygame
import utils
from drivers import CameraStreamManager, MlaRes, NlightDM, WFSManager

ROOT_DIR = "./data"

# display settings
VOLT_HEIGHT = 200
LOG_J_HEIGHT = 200
# 定义背景颜色
BACKGROUND_COLOR = (0, 0, 0)
# 定义折线颜色
LINE_COLOR = (0, 255, 0)

# adam parameters
beta1 = 0.9
beta2 = 0.999
beta3 = 0.9999

# cool_momentum_spgd parameters
Rho_0 = 0.99

# metropolis parameters
METROPOLIS_ALPHA = 0.8

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

def learning_schedule(
    lr, epoch, epochs, method: Literal["static", "cosin", "exp", "linear"] = "static"
):
    if method == "static":
        return lr
    # 余弦退火
    elif method == "cosin":
        lr = lr * np.cos(np.pi * epoch / epochs) + 1e-6
        return lr
    # 指数衰减
    elif method == "exp":
        lr = lr * np.exp(-epoch / epochs) + 1e-6
        return lr
    # 线性衰减
    elif method == "linear":
        lr = lr * (1 - epoch / epochs) + 1e-6
        return lr
    else:
        raise ValueError("method must be static, cosin, exp or linear")


def gen_file_name(dir, postfix: str = None):
    fname = os.listdir(dir)
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
        
    # 清空之前绘制的折线统计图
    line_plot_area = pygame.Rect(0, IMG_SIZE[1] + VOLT_HEIGHT, IMG_SIZE[0], LOG_J_HEIGHT)
    window.fill(BACKGROUND_COLOR, line_plot_area)
    if len(log) > 1:
        min_sum = min(v["J"] for v in log)
        max_sum = min(v["J"] for v in log)
        points = []
        num_points = len(log)
        for i, value in enumerate(log):
            j_value = value['J']
            # 均匀分布 x 轴坐标
            x = int(i * (IMG_SIZE[0] / (num_points - 1)))
            y = IMG_SIZE[1] + VOLT_HEIGHT + LOG_J_HEIGHT - int(
                (j_value - min_sum) / (max_sum - min_sum) * LOG_J_HEIGHT
            ) if max_sum != min_sum else IMG_SIZE[1] + VOLT_HEIGHT + LOG_J_HEIGHT // 2
            points.append((x, y))
        pygame.draw.lines(window, LINE_COLOR, False, points, 2)
    
    pygame.event.pump()
    pygame.display.update()


def optimizer(
    epochs,
    delta=1,
    lr=1,
    weight_decay=0.001,
    algorithm: Literal[
        "spgd", "adam", "nadam", "adamod", "cool_momentum_spgd"
    ] = "adamod",
    lr_schedul: Literal[
        "static", "cosin", "exp", "linear"
    ] = "static",
    metropolis_temperature=0,
    v0=0,
    init_v=None
):
    delta = abs(delta)
    epochs = int(epochs)

    with WFSManager(MlaRes.Res768, use_custom_ref=False, high_speed=True) as wfs,\
            NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT) as dm:

        if init_v is None:
            _init_v = np.zeros(dm.dm_num, dtype=np.float64)
            _init_v[0] = v0
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 1)
        
        def calc_j():
            wfs.take_image()
            return wfs.get_rms()[-1]['rms']

        
        history = [
            {
                "J": calc_j(),
                "_v": _init_v,
                "diff": 0,
                "gamma": lr,
                "_delta": delta,
                "_epoch": -1,
            }
        ]
        with tqdm.tqdm(
            total=epochs, desc=f"{algorithm} iter {epochs}", dynamic_ncols=True
        ) as bar:
            for epoch in range(epochs):
                disturb_v = np.random.binomial(1, 0.5, (dm.dm_num,)).astype(float) * 2.0 - 1.0

                disturb_v = disturb_v * delta
                disturb_v[0] = 0

                dm.send_voltages(_init_v + disturb_v)
                pos_j = calc_j()

                dm.send_voltages(_init_v - disturb_v)
                neg_j = calc_j()

                diff = pos_j - neg_j
                gradient = diff * disturb_v
                lr = learning_schedule(lr, epoch, epochs, method=lr_schedul)
                if algorithm == "spgd":
                    update = lr * gradient - lr * weight_decay * _init_v

                elif algorithm.lower() in ("adam", "nadam", "adamod"):
                    if epoch == 0:
                        m = np.zeros_like(_init_v, dtype=np.float64)
                        v = np.zeros_like(_init_v, dtype=np.float64)
                        s = 0

                    m = beta1 * m + (1 - beta1) * (gradient)
                    v = beta2 * v + (1 - beta2) * (gradient**2)

                    m_hat = m / (1 - beta1 ** (epoch + 1))
                    v_hat = v / (1 - beta2 ** (epoch + 1))

                    if algorithm == "nadam":
                        m_hat = beta1 * m_hat + (1 - beta1) * (
                            gradient - lr * weight_decay * _init_v
                        ) / (1 - beta1 ** (epoch + 1))
                    elif algorithm == "adamod":
                        gamma = lr / (np.sqrt(v_hat) + 1e-8)
                        s = beta3 * s + (1 - beta3) * gamma
                        learning_rate = np.where(gamma<s, gamma, s)
                        update = learning_rate * m_hat
                    elif algorithm == "adam":
                        update = lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                    else:
                        pass

                elif algorithm == "cool_momentum_spgd":
                    if epoch == 0:
                        cooling_rate = (1 - Rho_0) ** (1 / epochs)
                        momentum = np.zeros_like(_init_v, dtype=np.float64)

                    rho_n = 1 - (1 - Rho_0) / (cooling_rate**epoch)
                    learning_rate = lr * (1 + rho_n) / 2
                    update = rho_n * momentum + learning_rate * (gradient)
                    momentum = update

                update = np.clip(update, -UPDATE_MAX+delta, UPDATE_MAX-delta)
                if metropolis_temperature > 0:
                    # 使用模拟退火接受或拒绝更新
                    last_J = history[-1]["J"]
                    metropolis = (max(pos_j, neg_j) - last_J) / last_J
                    if metropolis > 0 or np.random.rand() < np.exp(
                        metropolis / metropolis_temperature
                    ):
                        _to_update_v = np.clip(_init_v - update, V_MIN, V_MAX)
                    metropolis_temperature *= METROPOLIS_ALPHA
                else:
                    _to_update_v = np.clip(_init_v - update, V_MIN, V_MAX)
                
                if check_dm_unit_grad_safe(_to_update_v):
                    _init_v = _to_update_v
                else:
                    print("相邻单元压差过大，放弃本次结果")

                avg_j = (pos_j + neg_j) / 2
                if avg_j > 0.2:
                    delta = 3
                elif avg_j > 0.1:
                    delta = 2
                else:
                    delta = 1
                
                log = {
                    "J": avg_j,
                    "_diff": diff,
                    "_gamma": lr,
                    "delta": delta,
                    "_epoch": epoch,
                    "_v": _init_v,
                }
                history.append(log)
                # earlying schedule

                bar.set_postfix({k: f'{v:.4f}' for k, v in log.items() if k[0] != "_"})
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
        delta=2, # 扰动要和桶半径匹配，不要扰动会导致质心出桶
        lr=1.2, 
        algorithm="adamod",
        metropolis_temperature=0,
        lr_schedul="static")
    
    res_df = pd.DataFrame(res_list)
    res_df.to_pickle('record.pkl', compression=None)
    max_j_id = res_df['J'].idxmin()
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
