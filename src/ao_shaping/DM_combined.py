# -*- coding: utf-8 -*-
import numpy as np

import tqdm
import json
from loguru import logger

from ao_shaping.drivers import MlaRes, NlightDM, Thorlab_WFS, CameraStreamManager
from ao_shaping.algorithm.adam import AdaMOD
from ao_shaping.utils import gen_date_dir, gen_file_path_uuid
from ao_shaping.display import FrameInfo, AutoDisplay

ROOT_DIR = "./data/wf"

# adam parameters
beta1 = 0.9
beta2 = 0.999
beta3 = 0.9995

# camera parameters
CAM_EXP_TIME = 50
CAM_EXP_TIME_ADJ_RATE = 0
IMG_SIZE = (200, 200)

# wavefront sensor parameters
Pupil_Diameter = 2.26

# dm parameters
KEEP_VOLTAGE_WHEN_EXIT = True

def schedule(rms):
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
        return 1.0,1.1
    elif rms > 0.15:
        return 0.9, 0.9
    elif rms > 0.08:
        return 0.8, 0.8
    else:
        return 0.7, 0.7

def optimizer(
    epochs,
    img_size=IMG_SIZE,
    rms_threshold=0.15,
    init_v:list=[],
    r_bucket=6.4
):
    epochs = int(epochs)
    _init_v = np.array(init_v)

    frames = [
        FrameInfo("fspot", "far spot", "Image2DWithBucketFrame"),
        FrameInfo("wf", "wavefront", "Image2DFrame"),
        FrameInfo("voltage", "voltages", "VoltageFrame"),
        FrameInfo("pib", "PIB", "LogFrame"),
        FrameInfo("rms", "RMS", "LogFrame"),
        FrameInfo("info", "info", "TextFrame"),
    ]

    with Thorlab_WFS(MlaRes.Res512, use_custom_ref=False, high_speed=True, pupil_diameter=Pupil_Diameter) as wfs,\
            CameraStreamManager(cam_id=0, exposure_time_ms=CAM_EXP_TIME, skip_sampling=False) as cam,\
            NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT) as dm, \
            AutoDisplay(frames) as display:
        assert len(init_v) == dm.DM_Num
        dm.send_voltages(_init_v, 0.5)

        def calc_rms(n_sample=5):
            wfs.take_image(n_sample)
            wf, statics = wfs.get_wavefront()
            return wf, statics['rms']
        
        def calc_pib(bucket_mask, n_sample=5):
            img = cam.get_numpy_image(n_sample)
            return img, np.sum(img[bucket_mask]).astype(float)

        # Set up camera window for visualization
        init_img = cam.get_numpy_image(10)
        _center = np.unravel_index(np.argmax(init_img), init_img.shape)[::-1]
        (img_sx, img_sy), _ = cam.reset_window(_center, img_size)
        xv, yv = np.ogrid[:img_sx, :img_sy]
        def refind_center(img):
            center = np.unravel_index(np.argmax(img), img.shape)
            c_x, c_y = center[1]+1, center[0]+1
            logger.info("({},{}) value: {}".format(c_x, c_y, img[center]))
            imgmesh_dist = (xv-c_x) ** 2 + (yv-c_y) ** 2
            dist = np.sqrt(imgmesh_dist)
            bucket_mask = dist < r_bucket

            return (c_x, c_y), bucket_mask.transpose()

        _center, bucket_mask = refind_center(cam.get_numpy_image(10))
        wf, rms = calc_rms(5)
        img, pib = calc_pib(bucket_mask, 1)
        lr, delta = schedule(rms)
        optimizer = AdaMOD(dim=dm.DM_Num, lr=lr, beta1=beta1, beta2=beta2, beta3=beta3)
        history = [
            {
                "J_pib": pib,
                "J_rms": rms,
                "_rms": (rms,),
                "_pib": (pib,),
                "_center": _center,
                "_v": (_init_v,),
                "_gamma": lr,
                "delta": delta,
                "_epoch": 0,
                "_wavefront": wf[np.newaxis, ...],
                "_img": img[np.newaxis, ...],
            }
        ]
        mode = 'rms' if 6 > rms > rms_threshold else 'pib'

        with tqdm.tqdm(
            total=epochs, desc="{} iter {}".format(mode, epochs), dynamic_ncols=True
        ) as bar:
            for epoch in range(1,epochs+1):
                disturb_v = np.random.binomial(1, 0.5, (dm.DM_Num,)).astype(float) * 2.0 - 1.0

                disturb_v = disturb_v * delta
                disturb_v[0] = 0

                cam_sample, wf_sample = (1, 7) if mode == 'rms' else (10, 1)
                pos_vs = dm.send_voltages(_init_v + disturb_v)
                pos_wf, pos_rms = calc_rms(wf_sample)
                pos_img, pos_pib = calc_pib(bucket_mask, cam_sample)

                neg_vs = dm.send_voltages(_init_v - disturb_v)
                neg_wf, neg_rms = calc_rms(wf_sample)
                neg_img, neg_pib = calc_pib(bucket_mask, cam_sample)

                # 如果rms大于阈值, 使用rms作用目标函数；否则使用pib
                if min(pos_rms, neg_rms) > rms_threshold and max(pos_rms, neg_rms) < 6.0:
                    diff = -(pos_rms-neg_rms)
                    mode = 'rms'
                else:
                    diff = (pos_pib-neg_pib)
                    if mode == 'rms':
                        _center, bucket_mask = refind_center(pos_img if pos_pib>neg_pib else neg_img)
                    mode = 'pib'
                gradient = diff * disturb_v
                lr, delta = schedule(rms)
                optimizer.lr = lr
                update = optimizer.update(gradient)
                _to_update_v = np.clip(_init_v - update, -dm.max_iter_diff+delta, dm.max_iter_diff-delta)
                if dm.check_dm_unit_grad_safe(_to_update_v):
                    _init_v = _to_update_v
                else:
                    logger.warning(f"相邻单元压差大于{dm.max_neibor_diff}，放弃本次结果")
                
                J_pib = np.mean([pos_pib, neg_pib])
                J_rms = np.mean([pos_rms, neg_rms])
                history.append({
                    "J_pib": J_pib,
                    "J_rms": J_rms,
                    "_rms": (pos_rms, neg_rms),
                    "_pib": (pos_pib, neg_pib),
                    "_center": _center,
                    "_v": (pos_vs, neg_vs),
                    "_gamma": optimizer.lr,
                    "delta": delta,
                    "_epoch": epoch,
                    "_wavefront": np.stack((pos_wf, neg_wf), axis=0),
                    "_img": np.stack((pos_img, neg_img), axis=0),
                })
                info = {k: '{:.4f}'.format(v) for k, v in history[-1].items() if k[0] != "_"}

                frame_data = {
                    "fspot": {'img': neg_img, 'center': _center, 'r': r_bucket},
                    "wf": {'img': neg_wf},
                    "voltage": {'volts': _init_v},
                    "pib": {'value': J_pib},
                    "rms": {'value': J_rms},
                    "info": {'text': json.dumps(info, indent=4)},
                }
                display.render(frame_data=frame_data, info=f"Epoch {epoch}/{epochs}:{J_pib=:.4f},{J_rms=:.4f}")
                
                bar.set_postfix(info)
                bar.update(1)
    return history

def run():
    init_V = np.zeros((64,))
    wfs_history = optimizer(
        init_v=init_V.tolist(),
        epochs=100_000)

    # Find best voltage
    id_max = np.argmax([h["J"] for h in wfs_history])
    best_v = wfs_history[id_max]["_v"][0]
    best_J = wfs_history[id_max]["J_pib"]
    # Save final voltages
    np.savetxt('final_J-{:.4f}.csv'.format(best_J), best_v, fmt="%d")


if __name__ == "__main__":
    run()
