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

from typing import Any

import numpy as np
import tqdm

from ao_shaping.algorithm.gradient.adam import AdaMOD
from ao_shaping.drivers.dm._registry import get_dm_registry
from ao_shaping.drivers.dm.base import DM
from ao_shaping.drivers.ccd.common import (
    get_camera_exposure_ms,
    resample_on_saturation,
)
from ao_shaping.utils import ImageVoltagesDisplay, Recorder, logger
from ao_shaping.optimizer.spid_pib_loop import open_pib_bench, spid_pib_epoch
from ao_shaping.utils.image.spots_calc import (
    pib_ratio_mask,
    power_in_bucket_mask,
    radius,
)
# Camera-injection seam shared with wfless.pib: a single global in
# ao_shaping.optimizer._camera_inject (previously each optimizer carried its own
# copy, so injecting the simulator through one left the other on the real driver).
from ao_shaping.optimizer._camera_inject import (
    _camera_cls,  # noqa: F401  (backward-compat re-export)
    _camera_override,  # noqa: F401  (backward-compat re-export)
)
from ao_shaping.optimizer.constants import (
    IDEAL_SPOT_RADIUS,
    KEEP_VOLTAGE_WHEN_EXIT,
    TEST_EXPOSURE_TIME_BRIGHTNESS,
)
from ao_shaping.utils.io.recorder_keys import (
    RECORD_DIFF,
    RECORD_EPOCH,
    RECORD_IMAGE,
    RECORD_ROI_ENERGY,
    RECORD_VOLTAGES,
)

BETA1: float = 0.9
BETA2: float = 0.99
BETA3: float = 0.9999


def learning_schedule(
    power_radius: float, ideal_r: int = IDEAL_SPOT_RADIUS
) -> tuple[float, float]:
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
    exposure_time_ms: float = 0.0,
    cam_type: str = "",
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

    with open_pib_bench(
        cam_type=cam_type,
        cam_id=cam_id,
        exposure_time_ms=exposure_time_ms,
        cam_size=cam_size,
        target_max_brightness=target_max_brightness,
        center=center,
        dm=dm,
        create_dm=lambda: get_dm_registry().create(
            "nlight",
            keep_when_exit=KEEP_VOLTAGE_WHEN_EXIT,
            max_neibor_diff=dm_neibor_diff,
            max_voltage=dm_max_voltage,
            min_voltage=dm_min_voltage,
        ),
        init_v=init_v,
        dm_unit_mask=dm_unit_mask,
        init_fallback_target_max=TEST_EXPOSURE_TIME_BRIGHTNESS,
    ) as bench:
        cam = bench.cam
        dm = bench.dm
        _init_v = bench.init_v
        dm_unit_mask = bench.unit_mask
        center = bench.center
        init_img = bench.init_img
        img_size = bench.img_size
        auto_exposure = exposure_time_ms == 0
        try:
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
                int(
                    epochs
                    * 0.8
                    // max(
                        np.log(IDEAL_SPOT_RADIUS / max(r_bucket, 1e-6))
                        / np.log(shrink_ratio),
                        1,
                    )
                ),
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
            optimizer = AdaMOD(
                dm.DM_Num, lr=lr, beta1=BETA1, beta2=BETA2, beta3=BETA3, **kwargs
            )  # type: ignore[union-attr]
            if lr == 0:
                optimizer.lr, delta = learning_schedule(
                    radius(init_img, center=center, energy=0.8)
                )

            recorder.append(
                {
                    "J": j,
                    "pib": pib_ratio_mask(init_img, pib_mask, pib_scaler),
                    "_raw_pib": pib_ratio_mask(init_img, pib_mask),
                    RECORD_ROI_ENERGY: 0.0,
                    "_max_r": init_r,
                    RECORD_VOLTAGES: _init_v.copy(),
                    RECORD_IMAGE: init_img,
                    RECORD_DIFF: 0,
                    "lr": optimizer.lr,
                    "r": r_bucket,
                    "delta": delta,
                    RECORD_EPOCH: 0,
                    "exp_t": get_camera_exposure_ms(cam),
                    "max_brt": float(np.max(init_img)),
                }
            )

            def _render_epoch_frame(pos_img, voltages, epoch_number):
                if not show or window is None:
                    return True
                return window.render(
                    pos_img,
                    voltages,
                    dm.V_Min,
                    dm.V_Max,
                    center,
                    int(r_bucket),
                    str(epoch_number),
                )

            def _saturate_epoch(pos_frame, neg_frame):
                nonlocal pib_scaler
                avg = float(np.mean([np.max(pos_frame), np.max(neg_frame)]))
                if avg >= 255 and auto_exposure:
                    resampled = resample_on_saturation(
                        pos_frame,
                        cam,
                        exposure_time_ms,
                        target_max_brightness,
                        auto_exposure_fn=lambda c, t, n_sample=1: c.auto_exposure(
                            target_max=t,
                            twice_valid=False,
                            n_sample=max(n_sample, 20),
                        ),
                        saturation_threshold=avg,
                    )
                    pib_scaler = avg / max(np.max(resampled), 1)
                return avg

            with tqdm.tqdm(
                total=epochs, desc=f"iter {epochs}", dynamic_ncols=True
            ) as bar:
                for epoch in range(1, epochs + 1):
                    _epoch = spid_pib_epoch(
                        dm=dm,
                        cam=cam,
                        init_v=_init_v,
                        delta=delta,
                        unit_mask=dm_unit_mask,
                        rng=np.random,
                        optimizer=optimizer,
                        evaluate=lambda img, _mask=bucket_mask: power_in_bucket_mask(img, _mask),
                        maximize=True,
                        diff_sign=1.0,
                        epoch=epoch,
                        on_render=_render_epoch_frame,
                        on_saturation=_saturate_epoch,
                    )
                    if _epoch.aborted:
                        break
                    pos_img = _epoch.pos_img
                    neg_img = _epoch.neg_img
                    pos_j, neg_j = _epoch.pos_value, _epoch.neg_value
                    diff = _epoch.diff
                    gradient = _epoch.gradient
                    _init_v = _epoch.init_v
                    avg_brightness = _epoch.brightness
                    if not _epoch.applied:
                        logger.warning(
                            "Neighbour voltage difference too large — skip update"
                        )

                    pib, pib_ratio = (
                        pib_ratio_mask(pos_img, pib_mask, pib_scaler),
                        (pos_j + neg_j) / 2.0,
                    )
                    J = (pos_j + neg_j) / 2.0

                    if epoch % update_iter == update_iter - 1:
                        init_r = max(init_r * shrink_ratio, IDEAL_SPOT_RADIUS)

                    if (
                        (
                            epoch % update_iter == update_iter - 1
                            or (
                                shrink_iter > 0
                                and epoch % shrink_iter == shrink_iter - 1
                            )
                            or pib_ratio >= 0.99
                        )
                        and pib > 0
                        and not _fix_bucket
                    ):
                        power_radio = radius(pos_img, center=center, energy=0.8)
                        _pr = power_radio * shrink_ratio
                        _r = max(
                            r_bucket * shrink_ratio + 1, IDEAL_SPOT_RADIUS, r_bucket
                        )
                        r_bucket = min(_r, _pr, init_r)
                        bucket_mask = dist <= r_bucket
                        if lr == 0:
                            optimizer.lr, delta = learning_schedule(r_bucket)

                    log = {
                        "J": J,
                        RECORD_ROI_ENERGY: pib_ratio,
                        "_max_r": init_r,
                        "pib": pib,
                        "_raw_pib": pib_ratio_mask(pos_img, pib_mask),
                        RECORD_DIFF: diff,
                        "lr": optimizer.lr,
                        "r": r_bucket,
                        "delta": delta,
                        RECORD_EPOCH: epoch,
                        RECORD_VOLTAGES: _init_v.copy(),
                        RECORD_IMAGE: pos_img,
                        "exp_t": get_camera_exposure_ms(cam),
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
            except Exception as exc:
                # Teardown only: raising here would mask whatever ended the run.
                # Silent is worse -- the mirror keeps its last (possibly large)
                # voltages and nobody at the bench can tell.
                logger.error(
                    "DM 归零失败 ({}): {} —— 变形镜仍停留在最后一组电压",
                    type(exc).__name__,
                    exc,
                )
            if bench.owns_dm:
                bench.dm.close()

    return recorder
