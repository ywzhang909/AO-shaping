# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import zoom

import tqdm
import pygame

from ao_shaping.drivers import MlaRes, NlightDM, Thorlab_WFS, CameraStreamManager
from ao_shaping.algorithm.adam import AdaMOD
from ao_shaping.utils import gen_date_dir, gen_file_path_uuid

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
IMG_SIZE = (200, 200)

# wavefront sensor parameters
Pupil_Diameter = 2.7

# dm parameters
KEEP_VOLTAGE_WHEN_EXIT = True
UPDATE_MAX = 20
DM_Adj = np.loadtxt('data/dm_adj.txt')
Tolerance = 300
print("DM adjacency matrix loaded: {}, max voltage difference: {}".format(DM_Adj.shape, Tolerance))

def check_dm_unit_grad_safe(vs, adj_mat=DM_Adj, tolerance=Tolerance):
    assert len(vs) == DM_Adj.shape[0] == DM_Adj.shape[1]
    diff_mat = (vs[:,None] - vs[None,:]) * adj_mat
    return not np.any(diff_mat[diff_mat > tolerance])

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
    elif rms > 0.18:
        return 1.0,1.1
    elif rms > 0.15:
        return 0.9, 0.9
    elif rms > 0.1:
        return 0.8, 0.8
    else:
        return 0.7, 0.7


def render(window,
            history, r,
            info="") -> None:
    img = history[-1]['_img']
    wavefront = history[-1]['_wavefront']
    center = history[-1]['_center']
    
    _, img_y, img_x = img.shape
    pos_img_canvas = pygame.surfarray.make_surface(img[0].transpose())
    pygame.draw.circle(pos_img_canvas, (255, 0, 0), center, r, 1)
    pygame.display.set_caption(info)
    window.blit(pos_img_canvas, (0,0))
    # 扩大wavefront使得大小和img一样
    wf = wavefront[0]
    zoom_factors = (img_y / wf.shape[0], img_x / wf.shape[1])
    disp_wf = zoom(wf, zoom_factors, order=1)  # 双线性插值
    # fill nan in disp_wf
    disp_wf[np.isnan(disp_wf)] = 0
    wavefront_canvas = pygame.surfarray.make_surface(disp_wf.transpose())
    window.blit(wavefront_canvas, (img_y+5,0))
    
    # 绘制电压图
    # 清空之前绘制的条形统计图
    volts = history[-1]['_v'][0]
    plot_area = pygame.Rect(0, IMG_SIZE[1], IMG_SIZE[0], VOLT_HEIGHT)
    window.fill(BACKGROUND_COLOR, plot_area)
    bar_width = int(IMG_SIZE[0] / len(volts))
    for i,value in enumerate(history[-1]['_v'][0]):
        normed_v = (value + 300)/800
        color = (int(normed_v*255), int((1-normed_v)*255), 0)
        x = int(i * bar_width)
        y = int(IMG_SIZE[1] + VOLT_HEIGHT)
        height = int(value *  VOLT_HEIGHT/ 500)
        pygame.draw.line(window, color, (x, y), (x, y - height), bar_width)
    
    pygame.event.pump()
    pygame.display.update()

def optimizer(
    epochs,
    rms_threshold=0.15,
    init_v:list=[],
    r_bucket=6.4,
    show=True,
):
    epochs = int(epochs)
    _init_v = np.array(init_v)
    with Thorlab_WFS(MlaRes.Res768, use_custom_ref=False, high_speed=True, pupil_diameter=Pupil_Diameter) as wfs,\
            CameraStreamManager(cam_id=1, exposure_time_ms=CAM_EXP_TIME, skip_sampling=False) as cam,\
            NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT) as dm:     
        assert len(init_v) == dm.DM_Num
        dm.send_voltages(_init_v, 0.5)

        def calc_rms(n_sample=10):
            wfs.take_image(n_sample)
            wf, statics = wfs.get_wavefront()
            return wf, statics['rms']
        
        def calc_pib(bucket_mask, n_sample=5):
            img = cam.get_numpy_image(n_sample)
            return img, np.sum(img[bucket_mask]).astype(float)

        # Set up camera window for visualization
        init_img = cam.get_numpy_image(10)
        _center = np.unravel_index(np.argmax(init_img), init_img.shape)[::-1]
        (img_sx, img_sy), _ = cam.reset_window(_center, IMG_SIZE)
        xv, yv = np.ogrid[:img_sx, :img_sy]
        def refind_center(img):
            center = np.unravel_index(np.argmax(img), img.shape)
            c_x, c_y = center[1]+1, center[0]+1
            print("({},{}) value: {}".format(c_x, c_y, img[center]))
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
                "J": pib,
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
        
        if show:
            pygame.init()
            total_height = VOLT_HEIGHT*2 + cam.cam_height
            window = pygame.display.set_mode((cam.cam_width*2, total_height))

        with tqdm.tqdm(
            total=epochs, desc="{} iter {}".format(mode, epochs), dynamic_ncols=True
        ) as bar:
            for epoch in range(1,epochs+1):
                disturb_v = np.random.binomial(1, 0.5, (dm.DM_Num,)).astype(float) * 2.0 - 1.0

                disturb_v = disturb_v * delta
                disturb_v[0] = 0

                pos_vs = dm.send_voltages(_init_v + disturb_v)
                pos_wf, pos_rms = calc_rms()
                pos_img, pos_pib = calc_pib(bucket_mask)

                neg_vs = dm.send_voltages(_init_v - disturb_v)
                neg_wf, neg_rms = calc_rms()
                neg_img, neg_pib = calc_pib(bucket_mask)

                # 如果rms大于阈值, 使用rms作用目标函数；否则使用pib
                if min(pos_rms, neg_rms) > rms_threshold and max(pos_pib, neg_pib) < 3.0:
                    diff = (pos_rms-neg_rms)
                    mode = 'rms'
                else:
                    diff = -(pos_pib-neg_pib)
                    if mode == 'rms':
                        _center, bucket_mask = refind_center(pos_img if pos_pib>neg_pib else neg_img)
                    mode = 'pib'
                gradient = diff * disturb_v
                lr, delta = schedule(rms)
                optimizer.lr = lr
                update = optimizer.update(gradient)
                _to_update_v = _init_v - update
                if check_dm_unit_grad_safe(_to_update_v):
                    _init_v = _to_update_v
                else:
                    print("相邻单元压差过大，放弃本次结果")
                history.append({
                    "J": np.mean([pos_pib, neg_pib]),
                    "J_rms": np.mean([pos_rms, neg_rms]),
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

                if show:
                    render(
                        window, history, r_bucket, f"{epoch}: J={history[-1]['J']:.3f}"
                    )

                bar.set_postfix({k: '{:.4f}'.format(v) for k, v in history[-1].items() if k[0] != "_"})
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
    best_J = wfs_history[id_max]["J"]
    # Save final voltages
    np.savetxt('final_J-{:.4f}.csv'.format(best_J), best_v, fmt="%d")


if __name__ == "__main__":
    run()
