import os
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
CAM_SAMPLE_ITER = 5
IDEAL_SPOT_RADIUS = int(os.environ.get("IDEAL_SPOT_RADIUS", 7))

# dm parameters
KEEP_VOLTAGE_WHEN_EXIT = True


def learning_schedule(power_radio, ideal_r=IDEAL_SPOT_RADIUS) -> tuple[float, float]:
    '''
    Learning schedule for power radio.
    If power radio is less than or equal to 7, return power radio.
    Otherwise, return power radio raised to the power of epoch divided by update_iter.

    Args:
        epoch (int): Current epoch.
        power_radio (float): Power radio.
        ideal_r (float): Ideal spot radius.
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
    elif power_radio <= 5*ideal_r:
        return 2.2, 4
    else:
        return 3, 5

def optimize_pib(
    center,
    epochs,
    r_bucket=0,
    delta:float=1,
    lr:float=0,
    exposure_time_ms:int=80,
    shrink_iter:int=0,
    shrink_ratio:float=0.9,
    cam_id=0,
    show:bool=False,
    init_v=[],
    cam_size=250,
    target_max_brightness = 40,
    dm_unit_mask=None,
    dm_neibor_diff=200,
    dm_max_voltage=None,
    dm_min_voltage=None,
    **kwargs
):
    delta = abs(delta)
    epochs = int(epochs)
    recorder = Recorder(mark="pib", mode="max")
    
    with CameraStreamManager(cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=False) as cam,\
            NlightDM(keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT, max_neibor_diff=dm_neibor_diff, max_voltage=dm_max_voltage, min_voltage=dm_min_voltage) as dm:
        if dm_unit_mask is None:
            dm_unit_mask = dm.default_dm_unit_mask
            if dm_unit_mask[0]:
                logger.warning("dm_unit_mask[0] is True, which means the first unit is active.")
        
        if init_v is None or len(init_v) == 0:
            _init_v = np.zeros(dm.DM_Num, dtype=np.float64)
        else:
            _init_v = np.array(init_v)
        dm.send_voltages(_init_v, 1)

        cam.autoset_exposure_time_ms(n_sample=10, target_max_brightness=255, threshold=0.2)
        _img = cam.get_numpy_image(10)
        if center is None:
            h,w = _img.shape
            # TODO: 如果环围半径较小，使用质心而非形心;如果中间存在空洞使用形心，否则质心
            center = centroid(np.where(_img > np.max(_img[:max(int(h//50),2),:max(int(w//50),2)])
                                       , 1, 0))
        elif isinstance(center, str):
            _img = cam.get_numpy_image(10)
            if center == "mass":
                center = centroid(_img)
            elif center == 'max':
                center = np.unravel_index(np.argmax(_img), _img.shape)[::-1]
            elif center == 'shape':
                center = centroid(
                    np.where(_img > np.max(_img[:max(int(h//50),2),:max(int(w//50),2)]), 1, 0))
                
            else:
                raise ValueError(f"known center: {center}")

        else:
            center = center    
        logger.info(f"Centroid: {center}, Max brightness: {np.max(_img)} @ {cam.exposure_time_ms}ms")

        img_size = (cam_size, cam_size)
        img_size, _ = cam.reset_window(center, img_size)
        if 0<target_max_brightness<255:
            init_img = cam.autoset_exposure_time_ms(
                n_sample=CAM_SAMPLE_ITER, target_max_brightness=target_max_brightness)
        else:
            cam.reset_exposure_time(exposure_time_ms)
            init_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
        logger.debug(f"Inital Image Max brightness: {np.max(init_img)} @ {cam.exposure_time_ms}ms")
        img_size = init_img.shape[::-1]
        xv, yv = np.ogrid[-img_size[0]//2:img_size[0]//2, -img_size[1]//2:img_size[1]//2]

        if r_bucket <= 0:
            r_bucket = radius(init_img, center=center, energy=0.9) * shrink_ratio
        
        # r_bucket * shrink_ratio ^ 0.8N = IDEAL_SPOT_RADIUS
        # 0.8N = log_ shrink_ratio (IDEAL_SPOT_RADIUS / r_bucket)
        # iter = total_epochs / 0.8N
        init_r, update_iter = r_bucket, epochs * 0.8 // (np.log(IDEAL_SPOT_RADIUS / r_bucket) / np.log(shrink_ratio))

        imgmesh_dist = ((xv) ** 2 + (yv) ** 2).transpose()
        dist = np.sqrt(imgmesh_dist)
        bucket_mask = dist <= r_bucket
        pib_mask = dist <= IDEAL_SPOT_RADIUS
        
        if show:
            window = ImageVoltagesDisplay(img_size)
            window.init_window()

        # TODO: 修改成util中的函数、工具类
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
        if lr <= 0:
            lr, delta = learning_schedule(radius(init_img, center=center, energy=0.9))

        optimizer = AdaMOD(dm.DM_Num, lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, **kwargs)
        recorder.append(
            {
                "J": j,
                "pib": test_pib(init_img),
                "_p%": pib_ratio,
                "_max_r": init_r,
                "_v": _init_v,
                "_img": init_img,
                "_diff": 0,
                "lr": optimizer.lr,
                "r": r_bucket,
                "delta": delta,
                "_epoch": 0,
                "exp_t": cam.exposure_time_ms,
                "max_brt": np.max(init_img),
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
                _to_update_v = np.clip(_init_v - update, dm.min_voltage, dm.max_voltage)
                if dm.check_dm_unit_grad_safe(_to_update_v):
                    _init_v = _to_update_v
                else:
                    logger.warning(f"相邻单元压差大于{dm.max_neibor_diff}，放弃本次结果")

                avg_pib_ratio = (pos_pib_ratio + neg_pib_ratio) / 2
                log = {
                    "J": (pos_j + neg_j) / 2,
                    "_p%": avg_pib_ratio,
                    "_max_r": init_r,
                    "pib": test_pib(pos_img),
                    "_diff": diff,
                    "lr": optimizer.lr,
                    "r": r_bucket,
                    "delta": delta,
                    "_epoch": epoch,
                    "_v": _init_v,
                    "_img": pos_img,
                    "exp_t": cam.exposure_time_ms,
                    "max_brt": np.max(pos_img),
                }
                recorder.append(log)
                # earlying schedule
                if epoch % update_iter == update_iter - 1 and log['J'] > 0:
                    init_r = max(init_r * shrink_ratio, 5)

                if (shrink_iter > 0 and epoch % shrink_iter == shrink_iter - 1 and avg_pib_ratio > 0.1) or avg_pib_ratio > 0.5:
                    power_radio = radius(pos_img, center='origin', energy=0.9)
                    _pr = power_radio * shrink_ratio
                    r_bucket = min(r_bucket, _pr, init_r)
                    if not lr:
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

