
import tqdm
import numpy as np

from ao_shaping.drivers import CameraStreamManager, NlightDM
from ao_shaping.algorithm.adam import AdaMOD
from ao_shaping.utils import ImageVoltagesDisplay, logger, Recorder
from ao_shaping.utils.spots_calc import centroid, radius

# adam parameters
beta1 = 0.9
beta2 = 0.99
beta3 = 0.9999

# metropolis parameters
METROPOLIS_ALPHA = 0.8

# camera parameters
CAM_EXP_TIME_ADJ_RATE = 0
CAM_SAMPLE_ITER = 5

# dm parameters
KEEP_VOLTAGE_WHEN_EXIT = True


def learning_schedule(power_radio, ideal_r=6) -> tuple[float, float]:
    '''
    Learning schedule for power radio.
    If power radio is less than or equal to 7, return power radio.
    Otherwise, return power radio raised to the power of epoch divided by update_iter.

    Args:
        epoch (int): Current epoch.
        power_radio (float): Power radio.
    return:
        lr (float): Learning rate.
        delta (float): distribution.
    '''
    if power_radio <= 2*ideal_r:
        return 0.9, 0.9
    elif power_radio <= 3*ideal_r:
        return 1, 1
    elif power_radio <= 4*ideal_r:
        return 1.5, 2
    elif power_radio <= 6*ideal_r:
        return 2, 3
    else:
        return 2, 4

def optimize_pib(
    center,
    epochs,
    r_bucket=0,
    delta=1,
    lr=1,
    exposure_time_ms=80,
    shrink_iter=0,
    shrink_ratio=0.9,
    cam_id=0,
    show=True,
    init_v=[],
    cam_size=250,
    dm_unit_mask=None,
    **kwargs
):
    delta = abs(delta)
    epochs = int(epochs)
    recorder = Recorder(mark="pib", mode="max")
    
    with CameraStreamManager(cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=False) as cam,\
            NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT, max_neibor_diff=200) as dm:
        optimizer = AdaMOD(dm.DM_Num, lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, **kwargs)

        if dm_unit_mask is None:
            dm_unit_mask = dm.default_dm_unit_mask

        if init_v is None or len(init_v) == 0:
            _init_v = np.zeros(dm.DM_Num, dtype=np.float64)
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 1)

        if center is None:
            _img = cam.get_numpy_image(10)
            center = centroid(np.where(_img > np.mean(_img), 1, 0))

        img_size = (cam_size, cam_size)
        img_size, _ = cam.reset_window(center, img_size)
        init_img = cam.get_numpy_image(1)
        img_size = init_img.shape[::-1]
        xv, yv = np.ogrid[-img_size[0]//2:img_size[0]//2, -img_size[1]//2:img_size[1]//2]

        if r_bucket <= 0:
            r_bucket = radius(init_img, center='origin', energy=0.9) * shrink_ratio
        
        # r_bucket * shrink_ratio ^ N = 5
        # N = log_ shrink_ratio (5 / r_bucket)
        # iter = total_epochs / N
        init_r, update_iter = r_bucket, epochs // (np.log(5 / r_bucket) / np.log(shrink_ratio))

        imgmesh_dist = ((xv) ** 2 + (yv) ** 2).transpose()
        dist = np.sqrt(imgmesh_dist)
        bucket_mask = dist < r_bucket
        pib_mask = dist < 5
        
        if show:
            window = ImageVoltagesDisplay(img_size)
            window.init_window()

        def calc_j(img):
            if shrink_iter <= 0:
                in_power = np.sum(img[bucket_mask]).astype(float)
            else:
                in_power = np.sum(img[dist < r_bucket]).astype(float)
            in_power_ratio = in_power / np.sum(img[img>2]).astype(float)
            return in_power, in_power_ratio
        
        def test_pib(img):
            return np.sum(img[pib_mask]).astype(float)

        j, pib_ratio = calc_j(init_img)
        recorder.append(
            {
                "J": j,
                "pib": test_pib(init_img),
                "p%": pib_ratio,
                "max_r": init_r,
                "_v": _init_v,
                "_img": init_img,
                "_diff": 0,
                "lr": lr,
                "r": r_bucket,
                "delta": delta,
                "_epoch": 0,
            }
        )
        with tqdm.tqdm(
            total=epochs, desc=f"iter {epochs}", dynamic_ncols=True
        ) as bar:
            for epoch in range(1,epochs+1):
                disturb_v = np.random.binomial(1, 0.5, (dm.DM_Num,)).astype(float) * 2.0 - 1.0

                disturb_v = disturb_v * delta * dm_unit_mask

                dm.send_voltages(_init_v + disturb_v)
                pos_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                pos_j, pos_pib_ratio = calc_j(pos_img)

                dm.send_voltages(_init_v - disturb_v)
                neg_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                neg_j, neg_pib_ratio = calc_j(neg_img)

                # if (pos_j + neg_j) == 0 and CAM_EXP_TIME_ADJ_RATE > 1:
                #     cam.reset_explore_time(cam.explore_time * CAM_EXP_TIME_ADJ_RATE)
                #     continue

                diff = pos_j - neg_j
                gradient = -diff * disturb_v
                update = optimizer.update(gradient)
                _to_update_v = np.clip(_init_v - update, dm.V_Min, dm.V_Max)
                if dm.check_dm_unit_grad_safe(_to_update_v):
                    _init_v = _to_update_v
                else:
                    logger.warning("相邻单元压差过大，放弃本次结果")

                avg_pib_ratio = (pos_pib_ratio + neg_pib_ratio) / 2
                log = {
                    "J": (pos_j + neg_j) / 2,
                    "p%": avg_pib_ratio,
                    "max_r": init_r,
                    "pib": test_pib(pos_img),
                    "_diff": diff,
                    "lr": optimizer.lr,
                    "r": r_bucket,
                    "delta": delta,
                    "_epoch": epoch,
                    "_v": _init_v,
                    "_img": pos_img,
                }
                recorder.append(log)
                # earlying schedule
                if epoch % update_iter == update_iter - 1 and log['J'] > 0:
                    init_r = max(init_r * shrink_ratio, 5)

                if (shrink_iter > 0 and epoch % shrink_iter == shrink_iter - 1 and avg_pib_ratio > 0.1) or avg_pib_ratio > 0.5:
                    power_radio = radius(pos_img, center='origin', energy=0.9)
                    _pr = power_radio * shrink_ratio
                    r_bucket = min(r_bucket, _pr, init_r)
                    optimizer.lr, delta = learning_schedule(epoch, power_radio)
                    # delta = max(delta * shrink_ratio, 0.6)
                    # optimizer.lr = max(lr * shrink_ratio, 0.8)
                    

                if show:
                    if not window.render(
                        pos_img, _init_v, dm.V_Min, dm.V_Max, center, r_bucket, f"{epoch}: PIB={log['pib']:.3f}"
                    ):
                        break
                
                bar.set_postfix({k: v for k, v in log.items() if k[0] != "_"})
                bar.update(1)
        if show:
            window.close()
        return recorder

