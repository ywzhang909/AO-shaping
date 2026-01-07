from typing import Literal, Sequence
import tqdm

import numpy as np

from ao_shaping.drivers import MlaRes, NlightDM, Thorlab_WFS
from ao_shaping.algorithm.adam import AdaMOD
from ao_shaping.utils import logger, Recorder

KEEP_VOLTAGES = True


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
        return 1.5, 2
    elif rms > 0.2:
        return 1.1, 1.2
    elif rms > 0.15:
        return 1, 1
    elif rms > 0.11:
        return 0.9, 0.9
    elif rms > 0.08:
        return 0.8, 0.8
    else:
        return 0.7, 0.7

def optimizer_rms(
    epochs,
    wfs_res: Literal['512', '768'] = '768',
    init_v: Sequence[float | int]=[],
    pupil_center:tuple[float, float]=(0,0),
    pupil_diameter:float=2.24,
    early_stop_threshold:float=0.12,
) -> Recorder:
    epochs = int(epochs)
    recorder = Recorder(mark='rms', mode='min')
    
    with NlightDM(keep_when_exit=KEEP_VOLTAGES) as dm:
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
        
        with Thorlab_WFS(wfs_res_config, 
                         use_custom_ref=False, high_speed=True, 
                         pupil_diameter=pupil_diameter,
                         pupil_center=pupil_center) as wfs:
                
            def calc_j():
                wfs.take_image(5)
                wf, statics = wfs.get_wavefront()
                return wf, statics

            wf, statics = calc_j()
            lr, delta = schedule_lr_delta(statics['wighted_rms'])
            optimizer = AdaMOD(dim=dm.DM_Num, lr=lr, beta3=0.9999)

            recorder.append(
                {
                    "rms": statics['rms'],
                    "_v": _init_v,
                    "_diff": 0,
                    "_gamma": lr,
                    "delta": delta,
                    "_epoch": 0,
                    "_wavefront": wf[np.newaxis, ...],
                    "_statics": statics
                }
            )

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
                    _to_update_v = np.clip(_init_v - update, dm.V_Min, dm.V_Max)
                    
                    if dm.check_dm_unit_grad_safe(_to_update_v):
                        _init_v = _to_update_v
                    else:
                        logger.warning(f"相邻单元压差大于{dm.max_neibor_diff}，放弃本次结果")
                    
                    log = {
                        "rms": avg_j,
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
                    recorder.append(log)
                    bar.set_postfix(recorder.last_info_dict)
                    if avg_j < early_stop_threshold:
                        logger.info(f"Early stop at epoch {epoch} with rms={avg_j:.4f}")
                        break
                    
                    bar.update(1)

                # end iter
            if KEEP_VOLTAGES:
                best_voltage, _ = recorder.get_best_target('_v')
                dm.send_voltages(best_voltage)

        return recorder
