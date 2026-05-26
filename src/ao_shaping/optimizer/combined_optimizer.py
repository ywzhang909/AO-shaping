"""Combined PIB optimization with adaptive bucket radius.

This module provides a variant of the PIB (Power-in-Bucket) optimizer that uses
AdaMOD and an adaptive bucket radius strategy. It is registered as the 'combined'
CLI command in main.py.

The optimization loop:
1. Opens the camera and DM (supports any DM subclass via the dm parameter).
2. Uses AdaMOD for gradient-based optimization with a disturbance-based policy.
3. Dynamically shrinks the bucket radius as focus improves.
4. Returns a Recorder with full optimization history.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import tqdm

from ao_shaping.algorithm.adam import AdaMOD
from ao_shaping.drivers import CameraStreamManager
from ao_shaping.drivers.dm._registry import get_dm_registry
from ao_shaping.drivers.dm.base import DM
from ao_shaping.utils import ImageVoltagesDisplay, Recorder, logger
from ao_shaping.utils.spots_calc import centroid, pib_ratio_mask, power_in_bucket_mask, radius

BETA1: float = 0.9
BETA2: float = 0.99
BETA3: float = 0.9999
METROPOLIS_ALPHA: float = 0.8

CAM_SAMPLE_ITER: int = 1
TEST_EXPOSURE_TIME_BRIGHTNESS: int = 220
IDEAL_SPOT_RADIUS: int = int(os.environ.get("IDEAL_SPOT_RADIUS", "6"))
KEEP_VOLTAGE_WHEN_EXIT: bool = True


def learning_schedule(power_radius: float, ideal_r: int = IDEAL_SPOT_RADIUS) -> tuple[float, float]:
    """Map power radius to (learning_rate, delta).

    Args:
        power_radius: Current power-containing radius.
        ideal_r: Target spot radius.

    Returns:
        Tuple of (lr, delta).
    """
    if power_radius <= ideal_r:
        return 1.5, 1.0
    if power_radius <= 2 * ideal_r:
        return 2.0, 2.0
    if power_radius <= 3 * ideal_r:
        return 2.5, 3.0
    if power_radius <= 4 * ideal_r:
        return 3.0, 4.0
    return 4.0, 5.0


def optimize_pib(
    center: str | tuple[int, int] | None,
    epochs: int = 4_000,
    r_bucket: float = 0.0,
    delta: float = 1.0,
    lr: float = 0.0,
    exposure_time_ms: float = 80.0,
    shrink_iter: int = 0,
    shrink_ratio: float = 0.9,
    cam_id: int = 0,
    show: bool = False,
    init_v: list[float] | None = None,
    cam_size: int = 250,
    target_max_brightness: float = 40.0,
    dm_unit_mask: list[bool] | None = None,
    dm_neibor_diff: float = 200.0,
    dm_max_voltage: float | None = None,
    dm_min_voltage: float | None = None,
    dm: DM | None = None,
    **kwargs: Any,
) -> Recorder:
    """Optimize PIB (Power in Bucket) with AdaMOD and adaptive bucket radius.

    This optimizer opens the camera and DM, then runs a disturbance-based
    gradient loop that dynamically shrinks the bucket radius as the spot
    becomes more concentrated.

    Args:
        center: Spot center. Pass a ``(x, y)`` tuple, ``"mass"``, ``"max"``,
            ``"shape"``, or ``None`` for auto-detection.
        epochs: Optimisation iterations.
        r_bucket: Initial bucket radius. ``0`` = auto-detect.
        delta: Disturbance magnitude.
        lr: AdaMOD learning rate. ``0`` = auto-schedule.
        exposure_time_ms: Camera exposure (ms). ``0`` = auto.
        shrink_iter: Shrink bucket every N steps. ``0`` = disable.
        shrink_ratio: Bucket shrink factor per shrink step.
        cam_id: Camera device ID.
        show: Show live Pygame display.
        init_v: Initial DM voltage vector.
        cam_size: Camera region-of-interest size.
        target_max_brightness: Target max pixel brightness. ``0`` = disable auto-exposure.
        dm_unit_mask: Boolean mask for enabled DM actuators.
        dm_neibor_diff: Max neighbour voltage difference safety check.
        dm_max_voltage: DM max voltage limit.
        dm_min_voltage: DM min voltage limit.
        dm: DM instance. If ``None``, a default NlightDM is created.
        **kwargs: Additional kwargs forwarded to the AdaMOD constructor.

    Returns:
        ``Recorder`` with full per-epoch history (J, pib, voltages, etc.).
    """
    delta = abs(delta)
    epochs = int(epochs)
    recorder = Recorder(mark="pib", mode="max")

    with CameraStreamManager(
        cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=False
    ) as cam:
        if dm is None:
            dm = get_dm_registry().create(
                "nlight",
                keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT,
                max_neibor_diff=dm_neibor_diff,
                max_voltage=dm_max_voltage,
                min_voltage=dm_min_voltage,
            )
            dm.open()
            _dm_owned = True
        else:
            if not dm.is_connected():
                dm.open()
            _dm_owned = False

        try:
            if dm_unit_mask is None:
                _dm_mask = dm.default_dm_unit_mask  # type: ignore[union-attr]
                dm_unit_mask = list(_dm_mask)  # type: ignore[assignment]
                if _dm_mask[0]:
                    logger.warning("dm_unit_mask[0] is True — first unit is active.")

            if init_v is None or len(init_v) == 0:
                _init_v = np.zeros(dm.DM_Num, dtype=np.float64)  # type: ignore[union-attr]
            else:
                _init_v = np.array(init_v, dtype=np.float64)
            dm.send_voltages(_init_v, 0.5)

            img_size = (cam_size, cam_size)

            _img = cam.autoset_exposure_time_ms(
                target_max_brightness=TEST_EXPOSURE_TIME_BRIGHTNESS
            )

            _center: tuple[int, int]
            if center is None:
                h, w = _img.shape
                margin = int(IDEAL_SPOT_RADIUS)
                _center = (int(centroid(
                    np.where(
                        _img > np.max(_img[:max(int(h // 50), 2), :max(int(w // 50), 2)]),
                        1,
                        0,
                    )
                )[0]), int(centroid(
                    np.where(
                        _img > np.max(_img[:max(int(h // 50), 2), :max(int(w // 50), 2)]),
                        1,
                        0,
                    )
                )[1]))
                cx, cy = _center
                if np.all(
                    _img[cy - margin : cy + margin, cx - margin : cx + margin]
                    >= np.max(_img) * 0.4
                ):
                    _center = (int(centroid(_img)[0]), int(centroid(_img)[1]))
            elif isinstance(center, str):
                _img = cam.get_numpy_image(10)
                if center == "mass":
                    _center = (int(centroid(_img)[0]), int(centroid(_img)[1]))
                elif center == "max":
                    _center = (int(np.unravel_index(np.argmax(_img), _img.shape)[::-1][0]),
                               int(np.unravel_index(np.argmax(_img), _img.shape)[::-1][1]))
                elif center == "shape":
                    h, w = _img.shape
                    _center = (int(centroid(
                        np.where(
                            _img > np.max(_img[:max(int(h // 50), 2), :max(int(w // 50), 2)]),
                            1,
                            0,
                        )
                    )[0]), int(centroid(
                        np.where(
                            _img > np.max(_img[:max(int(h // 50), 2), :max(int(w // 50), 2)]),
                            1,
                            0,
                        )
                    )[1]))
                else:
                    raise ValueError(f"Unknown center: {center}")
            else:
                _center = center  # type: ignore[assignment]

            center = _center
            logger.info(f"Centroid: {center}, Max brightness: {np.max(_img)} @ {cam.exposure_time}ms")

            img_size, _ = cam.reset_window(center, img_size)

            if 0 < target_max_brightness < 255 and target_max_brightness > 0:
                auto_exposure = True
                init_img = cam.autoset_exposure_time_ms(
                    target_max_brightness=target_max_brightness, twice_valid=True
                )
            elif exposure_time_ms > 0:
                auto_exposure = False
                cam.exposure_time = int(exposure_time_ms)
                init_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
            else:
                init_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
            logger.debug(f"Initial image max brightness: {np.max(init_img)} @ {cam.exposure_time}ms")

            img_size = init_img.shape[::-1]
            xv, yv = np.ogrid[
                -img_size[0] // 2 : img_size[0] // 2,
                -img_size[1] // 2 : img_size[1] // 2,
            ]

            if r_bucket <= 0:
                r_bucket = radius(init_img, center=center, energy=0.6) * shrink_ratio
                _fix_bucket = False
            else:
                _fix_bucket = True

            init_r = r_bucket
            update_iter = max(
                1,
                int(epochs * 0.8 // max(np.log(IDEAL_SPOT_RADIUS / max(r_bucket, 1e-6)) / np.log(shrink_ratio), 1)),
            )

            imgmesh_dist = (xv**2 + yv**2).transpose()
            dist = np.sqrt(imgmesh_dist)
            bucket_mask = dist <= r_bucket
            pib_mask = dist <= IDEAL_SPOT_RADIUS

            window: ImageVoltagesDisplay | None = None
            if show:
                window = ImageVoltagesDisplay(img_size)
                window.init_window()

            j, _ = power_in_bucket_mask(init_img, bucket_mask)
            pib_scaler = 1.0
            optimizer = AdaMOD(dm.DM_Num, lr=lr, beta1=BETA1, beta2=BETA2, beta3=BETA3, **kwargs)  # type: ignore[union-attr]
            if lr == 0:
                optimizer.lr, delta = learning_schedule(radius(init_img, center=center, energy=0.8))

            recorder.append(
                {
                    "J": j,
                    "pib": pib_ratio_mask(init_img, pib_mask, pib_scaler),
                    "_raw_pib": pib_ratio_mask(init_img, pib_mask),
                    "_p%": 0.0,
                    "_max_r": init_r,
                    "_v": _init_v.copy(),
                    "_img": init_img,
                    "_diff": 0,
                    "lr": optimizer.lr,
                    "r": r_bucket,
                    "delta": delta,
                    "_epoch": 0,
                    "exp_t": cam.exposure_time,
                    "max_brt": float(np.max(init_img)),
                }
            )

            with tqdm.tqdm(total=epochs, desc=f"iter {epochs}", dynamic_ncols=True) as bar:
                for epoch in range(1, epochs + 1):
                    disturb_v = (
                        np.random.binomial(1, 0.5, (dm.DM_Num,)).astype(float) * 2.0 - 1.0  # type: ignore[union-attr]
                    )
                    disturb_v = disturb_v * delta * np.asarray(dm_unit_mask, dtype=float)

                    dm.send_voltages(_init_v + disturb_v)
                    pos_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                    pos_j, _ = power_in_bucket_mask(pos_img, bucket_mask)

                    dm.send_voltages(_init_v - disturb_v)
                    neg_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                    neg_j, _ = power_in_bucket_mask(neg_img, bucket_mask)

                    if show and window is not None:
                        if not window.render(
                            pos_img, _init_v, dm.V_Min, dm.V_Max,  # type: ignore[union-attr]
                            center, int(r_bucket), str(epoch),
                        ):
                            break

                    avg_brightness = float(np.mean([np.max(pos_img), np.max(neg_img)]))
                    if avg_brightness >= 255 and auto_exposure:
                        _resampled_img = cam.autoset_exposure_time_ms(
                            target_max_brightness, twice_valid=False
                        )
                        pib_scaler = avg_brightness / max(np.max(_resampled_img), 1)

                    diff = pos_j - neg_j
                    gradient = -diff * disturb_v
                    update = optimizer.update(gradient)
                    _to_update_v = np.clip(
                        _init_v - update,
                        dm.min_voltage,  # type: ignore[union-attr]
                        dm.max_voltage,  # type: ignore[union-attr]
                    )

                    if dm.check_dm_unit_grad_safe(_to_update_v):  # type: ignore[union-attr]
                        _init_v = _to_update_v
                    else:
                        logger.warning("Neighbour voltage difference too large — skip update")

                    pib, pib_ratio = (
                        pib_ratio_mask(pos_img, pib_mask, pib_scaler),
                        (pos_j + neg_j) / 2.0,
                    )
                    J = (pos_j + neg_j) / 2.0

                    if epoch % update_iter == update_iter - 1:
                        init_r = max(init_r * shrink_ratio, IDEAL_SPOT_RADIUS)

                    if (
                        epoch % update_iter == update_iter - 1
                        or (shrink_iter > 0 and epoch % shrink_iter == shrink_iter - 1)
                        or pib_ratio >= 0.99
                    ) and pib > 0 and not _fix_bucket:
                        power_radio = radius(pos_img, center=center, energy=0.8)
                        _pr = power_radio * shrink_ratio
                        _r = max(r_bucket * shrink_ratio + 1, IDEAL_SPOT_RADIUS, r_bucket)
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
                        "_v": _init_v.copy(),
                        "_img": pos_img,
                        "exp_t": cam.exposure_time,
                        "max_brt": avg_brightness,
                    }
                    recorder.append(log)
                    bar.set_postfix({k: v for k, v in log.items() if k[0] != "_"})
                    bar.update(1)

        finally:
            if show and window is not None:
                window.close()
            try:
                dm.send_voltages(np.zeros(dm.DM_Num))  # type: ignore[union-attr]
            except Exception:
                pass
            if _dm_owned:
                dm.close()

    return recorder
