import os

import json
from typing import Literal

import numpy as np
import pandas as pd

import tqdm
import pygame
import matplotlib.pyplot as plt

from ao_shaping.drivers import CameraStreamManager, NlightDM
from ao_shaping.utils import centroid

ROOT_DIR = "data/wf-less"

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
CAM_EXP_TIME = 60
CAM_EXP_TIME_ADJ_RATE = 0
IMG_SIZE = (250, 250)

# dm parameters
KEEP_VOLTAGE_WHEN_EXIT = True

V_MAX = 499
V_MIN = -300
UPDATE_MAX = 20
DM_Adj = np.loadtxt('data/dm_adj.txt')
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


def gen_file_name(dir, postfix: str = ''):
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
    shrank_ratio=0.9,
    weights_class=1,
    pid_weighted_ratio=1.0,
    algorithm: Literal[
        "spgd", "adam", "nadam", "adamod", "cool_momentum_spgd", "gready"
    ] = "adamod",
    lr_schedul: Literal[
        "static", "cosin", "exp", "linear"
    ] = "static",
    metropolis_temperature=0,
    v0=0,
    show=True,
    init_v=None,
    center="max"
):
    delta = abs(delta)
    epochs = int(epochs)

    with CameraStreamManager(cam_id=0, exposure_time_ms=CAM_EXP_TIME, skip_sampling=False) as cam,\
            NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT) as dm:
        # dm.reset_all()

        if init_v is None:
            _init_v = np.zeros(dm.DM_Num, dtype=np.float64)
            _init_v[0] = v0
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 1)
        
        if center == 'max':
            init_img = cam.get_numpy_image(20)
            center = np.unravel_index(np.argmax(init_img), init_img.shape)
            center = (center[1], center[0])
            print(f'{center=} : {init_img[center]}')
        elif center == 'centroid':
            init_img = cam.get_numpy_image(20)
            center = centroid(init_img, cam.xv, cam.yv, 30)
            print(f'{center=} : {init_img[center]}')
        else:
            center = center

        img_size, _ = cam.reset_window(center, IMG_SIZE)
        init_img = cam.get_numpy_image(1)
        img_size = init_img.shape[::-1]
        xv, yv = np.ogrid[-img_size[0]//2:img_size[0]//2, -img_size[1]//2:img_size[1]//2]

        imgmesh_dist = ((xv) ** 2 + (yv) ** 2).transpose()
        dist = np.sqrt(imgmesh_dist)
        weighted_mask = ( - imgmesh_dist/np.max(imgmesh_dist) + 1) ** (weights_class)
        bucket_mask = dist < r_bucket
        pib_mask = dist < 5
        
        if show:
            total_height = VOLT_HEIGHT + LOG_J_HEIGHT + cam.cam_height
            pygame.init()
            window = pygame.display.set_mode((cam.cam_width, total_height))

        def calc_pib(img, r):
            if shrank_iter <= 0:
                return np.sum(img[bucket_mask]).astype(float)
            in_power = np.sum(img[dist < r]).astype(float)
            return in_power
        
        def test_pib(img):
            return np.sum(img[pib_mask]).astype(float)

        def calc_weighted_power(img) -> float:
            return np.sum(weighted_mask * img) / np.sum(weighted_mask)

        def encircled_radius(image: np.ndarray,
                    ratio: float = 0.86,
                    pixel_size: float = 1.17,   # px → mm
                    dist_step: float = 0.5,     # 直方图 bin 宽
                    ):
            """
            计算光斑质心与 encircled-energy 半径
            Parameters
            ----------
            image : 2-D ndarray, float or int
            ratio : 能量包围比，默认 0.86
            pixel_size : 像素物理尺寸（缩放系数），默认 1.17
            dist_step : 直方图距离间隔，默认 0.5 px
            center_offset : 坐标偏移，默认 (23.5, 23.5)

            Returns
            -------
            (cx, cy) : 质心（已乘 pixel_size）
            radius   : encircled-energy 半径（已乘 pixel_size）
            """
            max_d = dist.max()
            bins = int(np.ceil(max_d / dist_step)) + 1
            hist, bin_edges = np.histogram(dist, bins=bins, range=(0, bins*dist_step), weights=image)
            bin_centers = bin_edges[:-1] + dist_step / 2
            cum = np.cumsum(hist)
            target = ratio * np.sum(image)

            idx = np.searchsorted(cum, target, side='right')
            if idx == 0:
                radius = dist_step * 0.5
            else:
                radius = bin_centers[idx-1]
            radius *= pixel_size
            return radius
        
        def calc_j(img):
            return calc_pib(img, r_bucket)

        
        # utils.disp(init_img, cam.xv, cam.yv, r_bucket, 15)
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
            total=epochs, desc=f"{algorithm} iter {epochs}", dynamic_ncols=True
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
                lr = learning_schedule(lr, epoch, epochs, method=lr_schedul)
                if algorithm == "spgd":
                    update = lr * gradient - lr * weight_decay * _init_v

                elif algorithm.lower() in ("adam", "nadam", "adamod"):
                    if epoch == 1:
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
                    
                elif algorithm == "gready":
                    update = lr * np.sign(-diff) * disturb_v
                    
                else:
                    raise NameError(f'{algorithm} not supported!')
                    

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
                    # pid_weighted_ratio = min(pid_weighted_ratio * 0.7, 0)

                if show:
                    render(
                        window, pos_img, history, center, r_bucket, f"{epoch}: J={log['J']:.3f}"
                    )
                
                bar.set_postfix({k: v for k, v in log.items() if k[0] != "_"})
                bar.update(1)

        return history

def bayes_opt():
    from skopt import gp_minimize
    from skopt.space import Categorical, Integer, Real

    space = [
        Integer(10, 20, name="r_bucket"),
        Integer(0.4, 2, name="delta"),
        Real(0.4, 2, name="lr"),
        Real(0.0001, 0.01, name="weight_decay"),
        Integer(2000, 6000, name="shrank_iter"),
        Integer(0, 100, name="v0"),
        Integer(5,55, name="weights_class"),
        Real(0.2, 0.8, name="pid_weighted_ratio"),
        Categorical(
            ["adam", "nadam", "adamod"], name="algorithm"
        ),  # 添加算法选项
    ]

    def objective(params):
        r_bucket, delta, lr, weight_decay, shrank_iter, v0, ratio, weights_class, algorithm = params
        dfhistory = optimizer(
            r_bucket=r_bucket,
            delta=delta,
            lr=lr,
            weight_decay=weight_decay,
            shrank_iter=shrank_iter,
            v0=v0,
            weights_class=weights_class,
            algorithm=algorithm,
            pid_weighted_ratio=ratio,
            epochs=int(shrank_iter*4)
        )
        return -(dfhistory.iloc[-20:]["J"]).mean()

    result = gp_minimize(objective, space, n_calls=100, random_state=0)
    # 输出最优超参数
    (
        best_r_bucket,
        best_delta,
        best_gamma,
        best_weight_decay,
        best_shrank_iter,
        best_weights_class,
        best_algorithm,
    ) = result.x
    print(result)
    print(
        f"Best r_bucket: {best_r_bucket}, Best delta: {best_delta}, Best gamma: {best_gamma}, Best weight_decay: {best_weight_decay}, Best shrank_iter: {best_shrank_iter}, Best weights_class: {best_weights_class}, Best algorithm: {best_algorithm}"
    )

    # 使用最优超参数运行优化器
    dfhistory = optimizer(
        best_r_bucket,
        best_delta,
        best_gamma,
        best_weight_decay,
        best_shrank_iter,
        best_weights_class,
        best_algorithm,
    )
    return dfhistory

def run():
    # init_V = np.load('last_v-0.07.npz')['v']
    # init_V = np.loadtxt('rms-0.087.csv')
    #                  if os.path.exists("last_v.npz") else None
    
    # init_V = np.random.random((64,))*100 - 50
    # init_V[0] = 0
    
    # import pathlib
    # volt_files = list(pathlib.Path('data').glob('to_load_V-*.csv'))
    # volt_files.sort(key=lambda x: float(x.stem.split('-')[-1]))
    # init_V = np.loadtxt(volt_files[-1])
    
    init_V = np.zeros((64,))

    args = dict(
        init_v=init_V.tolist(),
        epochs=4_000,
        r_bucket=18,
        delta=2, # 扰动要和桶半径匹配，不要扰动会导致质心出桶
        lr=2,
        weights_class=1,
        algorithm="adamod",
        metropolis_temperature=0,
        lr_schedul="static",
        pid_weighted_ratio=1,
        shrank_iter=300,
        center="max"
    )

    res_list = optimizer(**args)
    
    res_df = pd.DataFrame(res_list)
    saved_file_name = gen_file_name(ROOT_DIR, 'pkl')
    res_df.to_pickle(saved_file_name, compression='zip')
    max_j_id = res_df['pib'].argmax()
    last_V = res_df.iloc[max_j_id]["_v"]
    max_j = res_df.iloc[max_j_id]['pib']
    print(f"{max_j_id} -> {max_j}")

    def get_nerbors(unit_id):
        return (a for a in np.where(DM_Adj[unit_id, :] == 1)[0])
    
    # base_unit_id = np.argmin(np.abs(last_V[1:]))+1
    # checked_mask = np.zeros_like(DM_Adj, dtype=bool)
    # def reset_nerbors(unit_id, v):
    #     min, max = v[unit_id]-Tolerance, v[unit_id]+Tolerance
    #     for nerbor in get_nerbors(unit_id):
    #         if not checked_mask[unit_id, nerbor]:
    #             v[nerbor] = np.clip(v[nerbor], min, max)
    #             checked_mask[unit_id, nerbor] = checked_mask[nerbor, unit_id] = True
    #             reset_nerbors(nerbor, v)

    # reset_nerbors(base_unit_id, last_V)
    np.savetxt(f'data/to_load_V-{max_j}.csv', np.around(last_V).astype(int), fmt="%d")

    with open(saved_file_name+'-args.json', 'w' ,encoding='utf8') as f:
        json.dump(args, f, ensure_ascii=False, indent=4)
        
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
    run()
