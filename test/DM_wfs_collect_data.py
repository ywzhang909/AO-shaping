import os
import time
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tqdm
from drivers import MlaRes, NlightDM, WFSManager, CameraStreamManager

ROOT_DIR = "./data"

# metrics
R_Bucket = 10
Img_Size = (200,200)

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
    if postfix:
        postfix = postfix if postfix.startswith('.') else f'.{postfix}'
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
        
    print(f"save path : {path}")
    return path

def save_list(dir, data):
    f_name = len(os.listdir(dir))+1
    save_path = os.path.join(dir, str(f_name)) + '.pkl'
    res_df = pd.DataFrame(data)
    res_df.to_pickle(save_path)
    print(f"\n{len(res_df)} saved @ {save_path}")
    del res_df

def optimizer(
    save_dir,
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

    with WFSManager(MlaRes.Res768, use_custom_ref=False, high_speed=True, pupil_diameter=2.8) as wfs,\
            NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT) as dm, \
            CameraStreamManager(cam_id=0, explosure_time=80, skip_sampling=False) as f_cam ,\
            CameraStreamManager(cam_id=1, explosure_time=400, skip_sampling=False) as n_cam:

        if init_v is None:
            _init_v = np.zeros(dm.dm_num, dtype=np.float64)
            _init_v[0] = v0
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 0.5)
        
        init_img = f_cam.get_numpy_image()
        # init_n_img = n_cam.get_numpy_image()

        # center = np.unravel_index(np.argmax(init_img), init_img.shape)
        # center = (center[1], center[0])
        center = (776, 470)

        print(f"{center=}")
        img_size, center = f_cam.reset_window(center, Img_Size)
        
        w,h = img_size
        cood = np.mgrid[-w//2:w//2, -h//2:h//2]
        bucket_mask = (cood[0]**2 + cood[1]**2) < R_Bucket**2
        def calc_bucket(img, r=10):
            return np.sum(img[bucket_mask])
        
        def calc_j():
            wfs.take_image()
            wf, statics = wfs.get_wavefront()
            return wf, statics

        wf, statics = calc_j()
        f_img = f_cam.get_numpy_image()
        history = [
            {
                "J": statics['rms'],
                "_v": _init_v,
                "_diff": 0,
                "_gamma": lr,
                "delta": delta,
                "_epoch": -1,
                "_f_cam": f_img[np.newaxis, ...],
                "_n_cam": n_cam.get_numpy_image()[np.newaxis, ...],
                "_wavefront": wf[np.newaxis, ...],
                "_statics": statics,
                "pib": calc_bucket(f_img)
            }
        ]
        Js_log = [history[-1]["J"]]
        with tqdm.tqdm(
            total=epochs, desc=f"{algorithm} iter {epochs}", dynamic_ncols=True
        ) as bar:
            for epoch in range(epochs):
                disturb_v = np.random.binomial(1, 0.5, (dm.dm_num,)).astype(float) * 2.0 - 1.0

                disturb_v = disturb_v * delta
                disturb_v[0] = 0

                pos_vs = dm.send_voltages(_init_v + disturb_v)
                pos_wf, pos_statics = calc_j()
                pos_j = pos_statics['rms']
                pos_f_img = f_cam.get_numpy_image()
                pos_n_img = n_cam.get_numpy_image()

                ng_vs = dm.send_voltages(_init_v - disturb_v)
                neg_wf, neg_statics = calc_j()
                neg_j = neg_statics['rms']
                neg_f_img = f_cam.get_numpy_image()
                neg_n_img = n_cam.get_numpy_image()
                
                diff = pos_statics['rms'] - neg_statics['rms']
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
                # if avg_j > 1.2:
                #     delta = 2
                #     lr = 1.2
                # elif avg_j > 0.5:
                #     delta = 1.8
                #     lr = 1.1
                # elif avg_j > 0.3:
                #     delta = 1.3
                #     lr = 1.0
                # elif avg_j > 0.2:
                #     delta = 1
                #     lr = 0.9
                # else:
                #     delta = 0.9
                #     lr = 0.8
                
                log = {
                    "J": avg_j,
                    "_diff": diff,
                    "_gamma": lr,
                    "delta": delta,
                    "pib": calc_bucket(neg_f_img),
                    "_epoch": epoch,
                    "_v": _init_v,
                    "_pos_v": pos_vs,
                    "_neg_v": ng_vs,
                    "_wavefront": np.stack((pos_wf, neg_wf)),
                    "_statics": {"pos": pos_statics, "neg": neg_statics},
                    "_f_cam": np.stack((pos_f_img, neg_f_img)),
                    "_n_cam": np.stack((pos_n_img, neg_n_img))
                }
                Js_log.append(log["J"])
                history.append(log)
                if len(history) >= 1000:
                    save_list(save_dir, history)
                    del history
                    history = []
                    
                bar.set_postfix({k: f'{v:.4f}' for k, v in log.items() if k[0] != "_"})
                bar.update(1)
        if history:
            save_list(save_dir, history)
            del history
            
        return Js_log

def run():
    # init_V = np.load('last_v.npz')['v'] \
    #                  if os.path.exists("last_v.npz") else np.zeros((64,))
    init_V = np.random.random((64,))*200 - 80
    # init_V = np.zeros((64,))
    init_V[0] = 0
    dir_name = gen_file_name(ROOT_DIR)
    res_list = optimizer(
        save_dir=dir_name,
        init_v=init_V.copy(),
        epochs=10000, 
        delta=1,
        lr=1, 
        algorithm="adamod",
        metropolis_temperature=0,
        lr_schedul="static")

    plt.plot(res_list, label='rms')
    plt.legend()
    plt.savefig(os.path.join(dir_name, 'j.png'))
    plt.close()

if __name__ == "__main__":
    for iter in range(1):
        print(f"Collect data @ iter:{iter}")
        run()