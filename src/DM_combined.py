# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd

import tqdm
import pygame
import utils.utils as utils
from drivers import MlaRes, NlightDM, Thorlab_wfs, CameraStreamManager

ROOT_DIR = "./data/wf"

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

# camera parameters
CAM_EXP_TIME = 60
CAM_EXP_TIME_ADJ_RATE = 0
IMG_SIZE = (250, 250)

# wavefront sensor parameters
Pupil_Diameter = 2.7

# dm parameters
KEEP_VOLTAGE_WHEN_EXIT = True
UPDATE_MAX = 20
DM_Adj = np.loadtxt('data\dm_adj.txt')
Tolerance = 300
print("DM adjacency matrix loaded: {}, max voltage difference: {}".format(DM_Adj.shape, Tolerance))

def check_dm_unit_grad_safe(vs, adj_mat=DM_Adj, tolerance=Tolerance):
    assert len(vs) == DM_Adj.shape[0] == DM_Adj.shape[1]
    diff_mat = (vs[:,None] - vs[None,:]) * adj_mat
    return not np.any(diff_mat[diff_mat > tolerance])

def learning_schedule(
    lr, epoch, epochs, method="static"
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
def delta_schedule(rms):
    if rms > 0.2:
        return 2
    elif rms > 0.13:
        return 1
    else:
        return 0.8

def gen_file_name(dir, postfix=None):
    if postfix:
        postfix = postfix if postfix.startswith('.') else '.' + postfix
        fname = [f for f in os.listdir(dir) if f.endswith(postfix)]
    else:
        fname = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
    fname = max([int(f.split('.')[0]) for f in fname]) + 1 if fname else '1'

    if not postfix:  # make dir
        path = os.path.join(dir, str(fname))
        if not postfix and not os.path.exists(path):
            os.makedirs(path)
    else:
        path = os.path.join(dir, str(fname)) + postfix

    print("save path : {}".format(path))
    return path

def render(window,
            history, r,
            info="") -> None:
    img = history[-1]['_img']
    center = history[-1]['_center']
    
    _, img_y, img_x = img.shape
    pos_img_canvas = pygame.surfarray.make_surface(img[0].transpose())
    pygame.draw.circle(pos_img_canvas, (255, 0, 0), center, r, 1)
    pygame.display.set_caption(info)
    window.blit(pos_img_canvas, (0,0))

    neg_img_canvas = pygame.surfarray.make_surface(img[1].transpose())
    pygame.draw.circle(neg_img_canvas, (255, 0, 0), center, r, 1)
    window.blit(neg_img_canvas, (img_y+5,0))
    
    # 绘制电压图
    # 清空之前绘制的条形统计图
    volts = (history[-1]['_v'][0] + 300)/800
    plot_area = pygame.Rect(0, IMG_SIZE[1], IMG_SIZE[0], VOLT_HEIGHT)
    window.fill(BACKGROUND_COLOR, plot_area)
    bar_width = int(IMG_SIZE[0] / len(volts))
    for i,value in enumerate(volts):
        color = (int(value*255), int((1-value)*255), 0)
        x = int(i * bar_width)
        y = int(IMG_SIZE[1] + VOLT_HEIGHT)
        height = int(value *  VOLT_HEIGHT)
        pygame.draw.line(window, color, (x, y), (x, y - height), bar_width)
    
    pygame.event.pump()
    pygame.display.update()

def optimizer(
    epochs,
    rms_threshold=0.9,
    lr=0.9,
    weight_decay=0.001,
    algorithm="adamod",
    lr_schedul="static",
    init_v=[],
    r_bucket=6.4,
    center="max",
    show=True,
):
    epochs = int(epochs)

    with Thorlab_wfs(MlaRes.Res768, use_custom_ref=False, high_speed=True, pupil_diameter=Pupil_Diameter) as wfs,\
            CameraStreamManager(cam_id=0, explosure_time=CAM_EXP_TIME, skip_sampling=False) as cam,\
            NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT) as dm:
        if not init_v:
            _init_v = np.zeros(dm.DM_Num, dtype=np.float64)
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 0.5)

        def calc_rms(n_sample=5):
            wfs.take_image(n_sample)
            wf, statics = wfs.get_wavefront()
            return wf, statics['rms']
        
        def calc_pib(n_sample=3):
            img = cam.get_numpy_image(n_sample)
            return img, np.sum(img[bucket_mask]).astype(float)

        # Set up camera window for visualization
        if center == 'max':
            init_img = cam.get_numpy_image(10)
            _center = np.unravel_index(np.argmax(init_img), init_img.shape)[::-1]
        elif center == 'centroid':
            init_img = cam.get_numpy_image(10)
            _center = utils.centroid(init_img, cam.xv, cam.yv, 30)
        else:
            _center = center

        (img_sx, img_sy), _ = cam.reset_window(_center, IMG_SIZE)

        def refind_center():
            init_img = cam.get_numpy_image(10)
            center = np.unravel_index(np.argmax(init_img), init_img.shape)
            c_x, c_y = center[1]+1, center[0]+1
            print("({},{}) value: {}".format(c_x, c_y, init_img[center]))
            xv, yv = np.ogrid[:img_sx, :img_sy]

            imgmesh_dist = (xv-c_x) ** 2 + (yv-c_y) ** 2
            dist = np.sqrt(imgmesh_dist)
            bucket_mask = dist < r_bucket

            return (c_x, c_y), bucket_mask

        _center, bucket_mask = refind_center()
        wf, rms = calc_rms(5)
        img, pib = calc_pib(1)
        delta = delta_schedule(rms)
        history = [
            {
                "J": pib,
                "rms": (rms,),
                "pib": (pib,),
                "_center": _center,
                "_v": (_init_v,),
                "_gamma": lr,
                "delta": delta,
                "_epoch": 0,
                "_wavefront": wf[np.newaxis, ...],
                "_img": img[np.newaxis, ...],
            }
        ]
        last_mode = 'rms' if rms < rms_threshold else 'pib'
        
        if show:
            pygame.init()
            total_height = VOLT_HEIGHT*2 + cam.cam_height
            window = pygame.display.set_mode((cam.cam_width*2, total_height))

        with tqdm.tqdm(
            total=epochs, desc="{} iter {}".format(algorithm, epochs), dynamic_ncols=True
        ) as bar:
            for epoch in range(1,epochs+1):
                disturb_v = np.random.binomial(1, 0.5, (dm.DM_Num,)).astype(float) * 2.0 - 1.0

                disturb_v = disturb_v * delta
                disturb_v[0] = 0

                pos_vs = dm.send_voltages(_init_v + disturb_v)
                pos_wf, pos_rms = calc_rms()
                pos_img, pos_pib = calc_pib()

                neg_vs = dm.send_voltages(_init_v - disturb_v)
                neg_wf, neg_rms = calc_rms()
                neg_img, neg_pib = calc_pib()

                # 如果rms大于阈值, 使用rms作用目标函数；否则使用pib
                if min(pos_rms, neg_rms) > rms_threshold:
                    diff = (pos_rms-neg_rms)
                    last_mode = 'rms'
                else:
                    diff = -(pos_pib-neg_pib)
                    if last_mode == 'rms':
                        _center, bucket_mask = refind_center()
                    last_mode = 'pib'
                gradient = -diff * disturb_v
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

                _to_update_v = _init_v - update
                if check_dm_unit_grad_safe(_to_update_v):
                    _init_v = _to_update_v
                else:
                    print("相邻单元压差过大，放弃本次结果")

                delta = delta_schedule(rms)
                history.append({
                    "J": np.mean([pos_pib, neg_pib]),
                    "rms": (pos_rms, neg_rms),
                    "pib": (pos_pib, neg_pib),
                    "_center": _center,
                    "_v": (pos_vs, neg_vs),
                    "_gamma": lr,
                    "delta": delta,
                    "_epoch": epoch,
                    "_wavefront": np.stack((pos_wf, neg_wf), axis=0),
                    "_img": np.stack((pos_img, neg_img), axis=0),
                })

                if show:
                    render(
                        window, history, r_bucket, f"{epoch}: J={history[-1]['J']:.3f}"
                    )

                bar.set_postfix({k: '{:.4f}'.format(v) for k, v in history[-1].items() if k[0] != "_"})
                bar.update(1)
    return history

def run():
    init_V = np.loadtxt("rms-0.087.csv")
    wfs_history = optimizer(
        init_v=init_V.copy(),
        epochs=10000,
        algorithm="adamod")

    # Find best voltage
    wfs_df = pd.DataFrame(wfs_history)
    wfs_df.columns = [c[1:] if c.startswith('_') else c for c in wfs_df.columns]
    min_J_idx = wfs_df["J"].argmin()
    best_v = wfs_df.iloc[min_J_idx]["v"]
    best_J = wfs_df.iloc[min_J_idx]["J"]
    # Save final voltages
    np.savetxt('final_J-{:.4f}.csv'.format(best_J), best_v, fmt="%d")


if __name__ == "__main__":
    run()
