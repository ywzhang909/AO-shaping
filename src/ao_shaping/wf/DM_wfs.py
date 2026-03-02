from typing import Literal
import os
from datetime import datetime
from loguru import logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tqdm
from ao_shaping.drivers import MlaRes, NlightDM, Thorlab_WFS
from ao_shaping.utils import gen_date_dir, gen_file_path_uuid
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
    elif rms > 0.2:
        return 1, 1.2
    elif rms > 0.15:
        return 0.9, 0.9
    elif rms > 0.08:
        return 0.8, 0.8
    else:
        return 0.7, 0.7

def optimizer_rms(
    epochs,
    wfs_res: Literal['512', '768'] = '768',
    init_v:list[float]=[],
    pupil_diameter:float=2.24,
    early_stop_threshold:float=0.12,
):
    epochs = int(epochs)

    with NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT) as dm:
        if not init_v:
            _init_v = np.zeros(dm.DM_Num, dtype=np.float64)
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 0.5)
        
        if wfs_res == '768':
            wfs_res_config = MlaRes.Res768
        elif wfs_res == '512':
            wfs_res_config = MlaRes.Res512
        else:
            raise ValueError(f"wfs_res must be '512' or '768', but got {wfs_res}")
        
        with Thorlab_WFS(wfs_res_config, use_custom_ref=False, high_speed=True, pupil_diameter=pupil_diameter) as wfs:
                
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
                    _to_update_v = np.clip(_init_v - update, V_MIN, V_MAX)
                    
                    if dm.check_dm_unit_grad_safe(_to_update_v):
                        _init_v = _to_update_v
                    else:
                        logger.warning(f"相邻单元压差大于{dm.max_neibor_diff}，放弃本次结果")
                    
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

                    if avg_j < early_stop_threshold:
                        logger.info(f"Early stop at epoch {epoch} with J={avg_j:.4f}")
                        break
                        
                    bar.set_postfix({k: f'{v:.4f}' for k, v in log.items() if k[0] != "_"})
                    bar.update(1)

        return history

def run():
    # init_V = get_init_V_by_rms()
    init_V = [0 for _ in range(64)]
    res_list = optimizer_rms(
        init_v=init_V.copy(),
        epochs=20_000)

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
    run()
