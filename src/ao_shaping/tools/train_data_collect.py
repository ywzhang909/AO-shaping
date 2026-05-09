import time
from typing import Literal
import asyncio
import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tqdm
from ao_shaping.drivers import MlaRes, NlightDM, Thorlab_WFS, CameraStreamManager
from ao_shaping.utils import gen_file_path_uuid, gen_file_path_inc

ROOT_DIR = "./data/img2img"

# adam parameters
beta1 = 0.9
beta2 = 0.999
beta3 = 0.9999

# cool_momentum_spgd parameters
Rho_0 = 0.99

# metropolis parameters
METROPOLIS_ALPHA = 0.8

# dm parameters
KEEP_VOLTAGE_WHEN_EXIT = False

async def async_to_pickle(df, file_path, **kwargs):
    """异步执行to_pickle操作"""
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, lambda: df.to_pickle(file_path, **kwargs))

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

def optimizer(
    saved_dir,
    epochs,
    delta=1.0,
    lr=1.0,
    weight_decay=0.001,
    algorithm: Literal[
        "spgd", "adam", "nadam", "adamod", "cool_momentum_spgd"
    ] = "adamod",
    lr_schedul: Literal[
        "static", "cosin", "exp", "linear"
    ] = "static",
    metropolis_temperature=0,
    v0=0,
    init_v=None,
):
    delta = abs(delta)
    epochs = int(epochs)

    with Thorlab_WFS(MlaRes.Res512, use_custom_ref=False, high_speed=False, pupil_diameter=0.0) as wfs,\
            NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT) as dm, \
            CameraStreamManager(cam_id=1, exposure_time_ms=20) as cam_axis, \
            CameraStreamManager(cam_id=0, exposure_time_ms=20) as cam_focal:

        if init_v is None:
            _init_v = np.zeros(dm.DM_Num, dtype=np.float64)
            _init_v[0] = v0
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 1)

        def get_data():
            wfs.take_image()
            wf, statics = wfs.get_wavefront()
            return statics, wf, cam_axis.get_numpy_image(), cam_focal.get_numpy_image()

        wf_statics, wf, cam_axis_img, cam_focal_img = get_data()
        history = [
            {
                "J": wf_statics['rms'],
                "_v": _init_v,
                "_diff": 0,
                "_gamma": lr,
                "delta": delta,
                "_epoch": 0,
                "_cam_axis": cam_axis_img,
                "_cam_focal": cam_focal_img,
                "_wavefront": wf
            }
        ]
        J_v_history = [
            {
                "J": wf_statics['rms'],
                "v": _init_v,
                "_epoch": 0,
            }
        ]
        with tqdm.tqdm(
            total=epochs, desc=f"{algorithm} iter {epochs}", dynamic_ncols=True
        ) as bar:
            s_time = time.perf_counter()
            for epoch in range(1, epochs+1):
                disturb_v = np.random.binomial(1, 0.5, (dm.DM_Num,)).astype(float) * 2.0 - 1.0

                disturb_v = disturb_v * delta
                disturb_v[0] = 0

                dm.send_voltages(_init_v + disturb_v)
                wf_statics, wf, cam_axis_img, cam_focal_img = get_data()
                pos_j = wf_statics['rms']
                history.append({
                    "J": pos_j,
                    "_diff": pos_j - history[-1]["J"],
                    "_epoch": epoch,
                    "_v": _init_v + disturb_v,
                    "_cam_axis": cam_axis_img,
                    "_cam_focal": cam_focal_img,
                    "_wavefront": wf,
                    "_time": time.perf_counter() - s_time
                })

                dm.send_voltages(_init_v - disturb_v)
                wf_statics, wf, cam_axis_img, cam_focal_img = get_data()
                neg_j = wf_statics['rms']
                history.append({
                    "J": neg_j,
                    "_diff": pos_j - neg_j,
                    "_epoch": epoch,
                    "_v": _init_v + disturb_v,
                    "_cam_axis": cam_axis_img,
                    "_cam_focal": cam_focal_img,
                    "_wavefront": wf,
                    "_time": time.perf_counter() - s_time
                })

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

                update = np.clip(update, -dm.max_neibor_diff+delta, dm.max_neibor_diff-delta)
                if metropolis_temperature > 0:
                    # 使用模拟退火接受或拒绝更新
                    last_J = history[-1]["J"]
                    metropolis = (max(pos_j, neg_j) - last_J) / last_J
                    if metropolis > 0 or np.random.rand() < np.exp(
                        metropolis / metropolis_temperature
                    ):
                        _to_update_v = np.clip(_init_v - update, dm.V_Min, dm.V_Max)
                    metropolis_temperature *= METROPOLIS_ALPHA
                else:
                    _to_update_v = np.clip(_init_v - update, dm.V_Min, dm.V_Max)

                if dm.check_dm_unit_grad_safe(_to_update_v):
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

                log = history[-1]
                J_v_history.append({
                    "J": avg_j,
                    "v": _init_v,
                    "_epoch": epoch,
                })
                # earlying schedule

                bar.set_postfix({k: f'{v:.4f}' for k, v in log.items() if k[0] != "_"})
                bar.update(1)

                if len(history) >= 1000 or epoch == epochs-1:
                    res_df = pd.DataFrame(history)
                    del history
                    history = []
                    file_path = gen_file_path_inc(saved_dir, "pkl")
                    asyncio.run(async_to_pickle(res_df, file_path))

        return J_v_history

def run():
    init_V = np.random.random((64,))*100 - 50
    init_V[0] = 0

    save_dir = gen_file_path_uuid(ROOT_DIR)

    res_list = optimizer(
        saved_dir=save_dir,
        init_v=init_V.copy(),
        epochs=10000,
        delta=2, # 扰动要和桶半径匹配，不要扰动会导致质心出桶
        lr=1.2,
        algorithm="adamod",
        metropolis_temperature=0,
        lr_schedul="static")

    res_df = pd.DataFrame(res_list)
    max_j_id = res_df['J'].argmax()
    last_V = res_df.iloc[max_j_id]["_v"]
    print(f"{max_j_id} -> {res_df.iloc[max_j_id]['J']}")

    fig, ax = plt.subplots(2, 1)
    ax[0].bar(x=np.arange(64)-0.25, height=init_V, width=0.5)
    ax[0].bar(x=np.arange(64)+0.25, height=last_V, width=0.5)
    ax[1].plot(res_df['J'])
    plt.savefig(gen_file_path_uuid(save_dir, "png"))

if __name__ == "__main__":
   for _ in range(50):
        run()
