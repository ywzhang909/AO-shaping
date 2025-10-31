import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tqdm
from ao_shaping.drivers import MlaRes, NlightDM, Thorlab_WFS
from ao_shaping.utils import gen_date_dir, gen_file_path_uuid, get_init_V_by_rms
from ao_shaping.algorithm.adam import AdaMOD

ROOT_DIR = "./data/wf"

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

V_MAX = NlightDM.V_Max
V_MIN = NlightDM.V_Min

def save_list(dir, data):
    f_name = len(os.listdir(dir))+1
    save_path = os.path.join(dir, str(f_name)) + '.pkl'
    res_df = pd.DataFrame(data)
    res_df.to_pickle(save_path)
    print(f"\n{len(res_df)} saved @ {save_path}")
    del res_df
    
def schedule_lr_delta(rms):
    '''
    schedule the learning rate and momentum factor based on the rms of the wavefront
    
    Args:
        rms (float): rms of the wavefront
    
    Returns:
        tuple: A tuple containing the learning rate (lr) and delta (disturb voltage).
    '''
    if rms > 0.3:
        return 2, 3
    elif rms > 0.25:
        return 1.2, 2
    elif rms > 0.18:
        return 1.0,1.1
    elif rms > 0.12:
        return 0.9, 0.9
    elif rms > 0.08:
        return 0.8, 0.8
    else:
        return 0.7, 0.7

def optimizer(
    epochs,
    metropolis_temperature=0,
    v0=0,
    init_v=None
):
    epochs = int(epochs)

    with Thorlab_WFS(MlaRes.Res768, use_custom_ref=False, high_speed=True, pupil_diameter=2.7) as wfs,\
            NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT) as dm:

        if init_v is None:
            _init_v = np.zeros(dm.DM_Num, dtype=np.float64)
            _init_v[0] = v0
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 0.5)

        def calc_j():
            wfs.take_image(5)
            wf, statics = wfs.get_wavefront()
            return wf, statics

        wf, statics = calc_j()
        lr, delta = schedule_lr_delta(statics['rms'])
        optimizer = AdaMOD(dim=dm.DM_Num, lr=lr, beta1=beta1, beta2=beta2, beta3=beta3)

        history = [
            {
                "J": statics['rms'],
                "_v": _init_v,
                "_diff": 0,
                "_gamma": lr,
                "delta": delta,
                "_epoch": 0,
                "_wavefront": wf[np.newaxis, ...],
                "_statics": statics
            }
        ]

        with tqdm.tqdm(
            total=epochs, desc=f"{statics['rms']:.3f} iter {epochs}", dynamic_ncols=True
        ) as bar:
            for epoch in range(1, epochs+1):
                disturb_v = np.random.binomial(1, 0.5, (dm.DM_Num,)).astype(float) * 2.0 - 1.0

                disturb_v = disturb_v * delta
                disturb_v[0] = 0

                pos_vs = dm.send_voltages(_init_v + disturb_v)
                pos_wf, pos_statics = calc_j()
                pos_j = pos_statics['rms']

                ng_vs = dm.send_voltages(_init_v - disturb_v)
                neg_wf, neg_statics = calc_j()
                neg_j = neg_statics['rms']
                
                diff = pos_statics['rms'] - neg_statics['rms']
                gradient = -diff * disturb_v
                
                avg_j = (pos_j + neg_j) / 2
                lr, delta = schedule_lr_delta(avg_j)
                optimizer.lr = lr
                update = optimizer.update(gradient)
                update = np.clip(update, -dm.max_iter_diff+delta, dm.max_iter_diff-delta)
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
                
                if dm.check_dm_unit_grad_safe(_to_update_v):
                    _init_v = _to_update_v
                else:
                    print("相邻单元压差过大，放弃本次结果")
                
                log = {
                    "J": avg_j,
                    "_diff": diff,
                    "_gamma": optimizer.lr,
                    "delta": delta,
                    "_epoch": epoch,
                    "_v": _init_v,
                    "_pos_v": pos_vs,
                    "_neg_v": ng_vs,
                    "_wavefront": np.stack([pos_wf, neg_wf]),
                    "_statics": {"pos": pos_statics, "neg": neg_statics},
                }
                history.append(log)
                    
                bar.set_postfix({k: f'{v:.4f}' for k, v in log.items() if k[0] != "_"})
                bar.update(1)

        return history

def run():
    # init_V = get_init_V_by_rms()
    init_V = [0 for _ in range(64)]
    res_list = optimizer(
        init_v=init_V.copy(),
        epochs=15000,
        metropolis_temperature=0)

    res_df = pd.DataFrame(res_list)
    save_dir = gen_date_dir(ROOT_DIR)
    saved_file_name = gen_file_path_uuid(save_dir, 'pkl')
    res_df.to_pickle(saved_file_name)
    min_id = res_df["J"].argmin()
    min_iter = res_df.iloc[min_id]
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    # 绘制J的变化趋势
    ax[0, 0].scatter(min_iter["_epoch"], min_iter["J"], color="red", marker="*", s=100)
    ax[0, 0].plot(res_df["_epoch"], res_df["J"])
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel("J")
    ax[0, 0].set_title(f"Min J: {min_iter['J']:.3f} @ epoch {min_iter['_epoch']}")
    # 绘制保存的电压
    ax[0, 1].bar(range(64), min_iter["_v"])
    ax[0, 1].set_xlabel("DM Unit")
    ax[0, 1].set_ylabel("Voltage")
    ax[0, 1].set_title(f"Min J: {min_iter['J']:.3f} @ epoch {min_iter['_epoch']}")
    # 绘制保存的初始波前
    ax[1, 0].imshow(res_df.iloc[0]["_wavefront"][0], cmap='gray')
    ax[1, 0].set_title("init wavefront")
    ax[1, 0].axis('off')
    # 绘制保存的最优波前
    ax[1, 1].imshow(min_iter["_wavefront"][1], cmap='gray')
    ax[1, 1].set_title("opt wavefront")
    ax[1, 1].axis('off')
    
    plt.savefig(saved_file_name.with_suffix('.png'))
    plt.close()
    
    # 在data/flatten_voltages下生成名称为当前日期的目录并保存电压
    dir_name = datetime.now().strftime("%Y%m%d")
    os.makedirs(os.path.join("data/flatten_voltages", dir_name), exist_ok=True)
    np.savetxt(os.path.join("data/flatten_voltages", dir_name, f'rms-{min_iter["J"]:.3f}.csv'), min_iter["_v"], fmt="%d")

if __name__ == "__main__":
    for iter in range(1):
        print(f"Collect data @ iter:{iter}")
        run()
