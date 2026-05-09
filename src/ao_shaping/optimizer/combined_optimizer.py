import os
import tqdm
import numpy as np

from ao_shaping.drivers import CameraStreamManager, NlightDM, Thorlab_WFS
from ao_shaping.algorithm.adam import AdaMOD
from ao_shaping.utils import ImageVoltagesDisplay, logger, Recorder
from ao_shaping.utils.spots_calc import centroid, radius, power_in_bucket_mask, pib_ratio_mask

# adam parameters
beta1 = 0.9
beta2 = 0.99
beta3 = 0.9999

# metropolis parameters
METROPOLIS_ALPHA = 0.8

# camera parameters
CAM_SAMPLE_ITER = 1
TEST_EXPOSURE_TIME_BRIGHTNESS = 220
IDEAL_SPOT_RADIUS = int(os.environ.get("IDEAL_SPOT_RADIUS", 6))

# dm parameters
KEEP_VOLTAGE_WHEN_EXIT = True


def learning_schedule(power_radius, ideal_r=IDEAL_SPOT_RADIUS) -> tuple[float, float]:
    '''
    Learning schedule for power radio.
    If power radio is less than or equal to ideal_r, return power radio.
    Otherwise, return power radio raised to the power of epoch divided by update_iter.

    Args:
        epoch (int): Current epoch.
        power_radio (float): Power radio.
        ideal_r (float): Ideal spot radius.
    return:
        lr (float): Learning rate.
        delta (float): distribution.
    '''
    if power_radius <= ideal_r:
        return 1.5, 1
    elif power_radius <= 2*ideal_r:
        return 2, 2
    elif power_radius <= 3*ideal_r:
        return 2.5, 3
    elif power_radius <= 4*ideal_r:
        return 3, 4
    else:
        return 4, 5

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
    """优化PIB（Power in Bucket）

    Args:
        center (str or tuple): 中心位置。
        epochs (int): 迭代次数。
        r_bucket (float): 桶半径。如果设置为0，则根据功率半径自动调整。
        delta (float): 分布参数。
        lr (float): 学习率。如果设置为0，则根据功率半径自动调整。
        exposure_time_ms (int): 曝光时间（毫秒）。如果设置为0，则自动曝光。
        shrink_iter (int): 收缩迭代次数。如果设置为0，则不进行收缩。
        shrink_ratio (float): 收缩比例。
        cam_id (int): 相机ID。
        show (bool): 是否显示图像。
        init_v (list): 初始电压。
        cam_size (int): 相机图像大小。
        target_max_brightness (float): 目标最大亮度。如果设置为0，则迭代过程中不自动调整曝光时间。
        dm_unit_mask (list): DM单元掩码。
        dm_neibor_diff (float): DM邻居电压差。
        dm_max_voltage (float): DM最大电压。
        dm_min_voltage (float): DM最小电压。
        **kwargs: 其他参数。

    """

    delta = abs(delta)
    epochs = int(epochs)
    recorder = Recorder(mark="pib", mode="max")
    
    with CameraStreamManager(cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=False) as cam,\
            Thorlab_WFS() as wfs,\
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

        img_size = (cam_size, cam_size)
        cam.reset_window(center, img_size)

        _img = cam.autoset_exposure_time_ms(target_max_brightness=TEST_EXPOSURE_TIME_BRIGHTNESS)
        if center is None:
            (h, w), margin = _img.shape
            center = centroid(np.where(_img > np.max(_img[:max(int(h // 50), 2), :max(int(w // 50), 2)])
                                     , 1, 0))
            (cx, cy) = center
            if np.all(_img[cy - margin: cy + margin, cx - margin: cx + margin] >= np.max(_img) * 0.4):
                center = centroid(_img)
        elif isinstance(center, str):
            _img = cam.get_numpy_image(10)
            if center == "mass":
                center = centroid(_img)
            elif center == 'max':
                center = np.unravel_index(np.argmax(_img), _img.shape)[::-1]
            elif center == 'shape':
                center = centroid(
                    np.where(_img > np.max(_img[:max(int(h // 50), 2), :max(int(w // 50), 2)]), 1, 0))
            else:
                raise ValueError(f"known center: {center}")
        logger.info(f"Centroid: {center}, Max brightness: {np.max(_img)} @ {cam.exposure_time}ms")

        img_size, _ = cam.reset_window(center, img_size)

        if 0 < target_max_brightness < 255 and target_max_brightness > 0:
            auto_exposure = True
            init_img = cam.autoset_exposure_time_ms(
                target_max_brightness=target_max_brightness, twice_valid=True)
        elif exposure_time_ms > 0:
            auto_exposure = False
            cam.exposure_time = exposure_time_ms
            init_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
        else:
            init_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
        logger.debug(f"Inital Image Max brightness: {np.max(init_img)} @ {cam.exposure_time}ms")
        img_size = init_img.shape[::-1]
        xv, yv = np.ogrid[-img_size[0] // 2:img_size[0] // 2, -img_size[1] // 2:img_size[1] // 2]

        if r_bucket <= 0:
            r_bucket = radius(init_img, center=center, energy=0.6) * shrink_ratio
            _fix_bucket = False
        else:
            _fix_bucket = True

        init_r, update_iter = r_bucket, epochs * 0.8 // (np.log(IDEAL_SPOT_RADIUS / r_bucket) / np.log(shrink_ratio))

        imgmesh_dist = ((xv) ** 2 + (yv) ** 2).transpose()
        dist = np.sqrt(imgmesh_dist)
        bucket_mask = (dist <= r_bucket)
        pib_mask = (dist <= IDEAL_SPOT_RADIUS)
        
        if show:
            window = ImageVoltagesDisplay(img_size)
            window.init_window()

        j, pib_ratio = power_in_bucket_mask(init_img, bucket_mask)
        pib_scaler = 1.0

        optimizer = AdaMOD(dm.DM_Num, lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, **kwargs)
        if lr == 0:
            optimizer.lr, delta = learning_schedule(radius(init_img, center=center, energy=0.8))
        recorder.append(
            {
                "J": j,
                "pib": pib_ratio_mask(init_img, pib_mask, pib_scaler),
                "_raw_pib": pib_ratio_mask(init_img, pib_mask),
                "_p%": pib_ratio,
                "_max_r": init_r,
                "_v": _init_v,
                "_img": init_img,
                "_diff": 0,
                "lr": optimizer.lr,
                "r": r_bucket,
                "delta": delta,
                "_epoch": 0,
                "exp_t": cam.exposure_time,
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
                pos_j, pos_pib_ratio = power_in_bucket_mask(pos_img, bucket_mask)

                dm.send_voltages(_init_v - disturb_v)
                neg_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                neg_j, neg_pib_ratio = power_in_bucket_mask(neg_img, bucket_mask)
                
                if show:
                    if not window.render(
                        pos_img, _init_v, dm.V_Min, dm.V_Max, center, r_bucket, f"{epoch}"
                    ):
                        break
                
                avg_brightness = np.mean([np.max(pos_img), np.max(neg_img)])
                if avg_brightness == 255 and auto_exposure:
                    _resampled_img = cam.autoset_exposure_time_ms(target_max_brightness, twice_valid=False)
                    pib_scaler = avg_brightness / np.max(_resampled_img)

                diff = pos_j - neg_j
                gradient = -diff * disturb_v
                update = optimizer.update(gradient)
                _to_update_v = np.clip(_init_v - update, dm.min_voltage, dm.max_voltage)
                if dm.check_dm_unit_grad_safe(_to_update_v):
                    _init_v = _to_update_v
                else:
                    logger.warning(f"相邻单元压差大于{dm.max_neibor_diff}，放弃本次结果")
              
                # learning schedule
                pib, pib_ratio = pib_ratio_mask(pos_img, pib_mask, pib_scaler), (pos_pib_ratio+neg_pib_ratio)/2
                J = (pos_j + neg_j) / 2

                if epoch % update_iter == update_iter - 1:
                    init_r = max(init_r * shrink_ratio, IDEAL_SPOT_RADIUS)

                if (epoch % update_iter == update_iter - 1 or
                     epoch % shrink_iter == shrink_iter - 1 or
                     pib_ratio >= 0.99) and pib > 0 and not _fix_bucket:
                    power_radio = radius(pos_img, center=center, energy=0.8)
                    _pr = power_radio * shrink_ratio
                    _r = max(r_bucket*shrink_ratio+1, IDEAL_SPOT_RADIUS, r_bucket)
                    r_bucket = min(_r, _pr, init_r)
                    bucket_mask = dist <= r_bucket
                    if lr == 0:
                        optimizer.lr, delta = learning_schedule(r_bucket)
                
                log = {
                    "J": J,
                    "_p%": pib_ratio,
                    "_max_r": init_r,
                    "pib": pib,
                    "_raw_pib": pib_ratio_mask(pos_img, pib_mask),
                    "_diff": diff,
                    "lr": optimizer.lr,
                    "r": r_bucket,
                    "delta": delta,
                    "_epoch": epoch,
                    "_v": _init_v,
                    "_img": pos_img,
                    "exp_t": cam.exposure_time,
                    "max_brt": avg_brightness,
                }
                recorder.append(log)
                bar.set_postfix({k: v for k, v in log.items() if k[0] != "_"})
                bar.update(1)
        if show:
            window.close()
        return recorder