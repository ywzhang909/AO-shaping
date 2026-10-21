"""Shared bench + SPGD-epoch mechanics for the two ``optimize_pib`` paths.

``optimizer/wfless/pib.py`` (SPGD + adaptive-neighbourhood + tabu search) and
``optimizer/combined_optimizer.py`` (AdaMOD + adaptive bucket radius) are a
same-family fork: both open a camera and a DM, take a flat-field reading, and
then run a paired-measurement SPGD loop.  The mechanics were duplicated
verbatim and had already drifted apart, so a fix applied to one path silently
missed the other.

This module single-homes the parts that are *mechanics* and leaves the parts
that are *policy* at the call site.

Mechanics (here)
    * opening the camera and the DM, and the flat-field reading: probe ->
      centre -> brightness log -> window reset -> initial image;
    * one paired-measurement epoch: perturb, measure both polarities, build
      the SPGD gradient, apply the update under the voltage clip and the
      neighbour-difference safety check.

Policy (at the call site)
    * the perturbation stream and the objective -- the two paths deliberately
      differ (they draw from different RNG streams, which is why they walk
      apart at epoch 1; see ``test_spid_pib_path_characterization.py``);
    * the saturation policy, including which brightness statistic counts as
      "too bright" (one path takes the peak of the two frames, the other their
      mean) and what remedy follows;
    * the recorder columns, the learning-rate table, the radius-shrink rule
      and the exit behaviour (only the combined path zeroes the DM on exit).

Three readings of the flat field are *not* interchangeable between the paths
and are therefore exposed rather than unified:

* ``PibBench.center`` is the centre detected on the probe frame, while
  ``PibBench.window_center`` is what ``cam.reset_window`` returned.  The
  combined path measures against the detected centre, the SPGD path against
  the window centre; unifying them would move the SPGD path's metrics.
* the initial image's brightness target falls back to a different constant per
  path, so it is a parameter.

Three further details are load-bearing and must not be "tidied up":

* the DM factory is injected rather than looked up here, because the
  simulation harness patches ``get_dm_registry`` on the *calling* module -- a
  lookup in this module would bypass the patch and reach for real hardware;
* ``cam.reset_window`` is called positionally, because the real driver names
  its second parameter ``size`` while the simulation shim names it
  ``img_size``, so a keyword call would break one of the two;
* the voltage clip reads ``dm.V_Min`` / ``dm.V_Max``.  One path spelled the
  same bound ``dm.min_voltage`` / ``dm.max_voltage``; those are properties on
  the base DM class that *return* ``V_Min`` / ``V_Max``, so the two spellings
  were always the same number and one home is enough.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from ao_shaping.algorithm.gradient.adam import Base
from ao_shaping.drivers.ccd.common import (
    capture_with_exposure,
    get_camera_exposure_ms,
)
from ao_shaping.drivers.dm.base import DM
from ao_shaping.optimizer._camera_inject import _camera_cls
from ao_shaping.optimizer.constants import (
    CAM_SAMPLE_ITER,
    TEST_EXPOSURE_TIME_BRIGHTNESS,
)
from ao_shaping.optimizer.spgd import spgd_gradient
from ao_shaping.utils import logger
from ao_shaping.utils.image.beam_metrics import smart_zero_order_center
from ao_shaping.utils.image.hardware_utils import (
    log_center_brightness,
    resolve_spot_center,
)

__all__ = [
    "PibBench",
    "SpidPibEpoch",
    "open_pib_bench",
    "spid_pib_epoch",
]


@dataclass(frozen=True)
class PibBench:
    """The opened bench plus the flat-field reading taken before epoch 1.

    Attributes:
        cam: The opened camera.
        dm: The opened DM (created here, or the one the caller injected).
        owns_dm: Whether this call created the DM, and therefore whether the
            caller is responsible for closing it.  The caller closes it
            because its own teardown also covers the live window and, for the
            combined path, zeroing the DM.
        init_v: The initial DM voltages, already sent to the hardware.
        unit_mask: The active-actuator mask, normalised to float64.
        center: Centre detected on the probe frame.
        window_center: Centre returned by ``cam.reset_window``.
        img_size: ``init_img``'s shape as ``(width, height)``.
        init_img: The initial (flat-field) image.
        probe_img: The probe image the centre was detected on.
    """

    cam: Any
    dm: DM
    owns_dm: bool
    init_v: np.ndarray
    unit_mask: np.ndarray
    center: tuple[int, int]
    window_center: tuple[int, int]
    img_size: tuple[int, int]
    init_img: np.ndarray
    probe_img: np.ndarray


@dataclass(frozen=True)
class SpidPibEpoch:
    """One paired-measurement epoch of a SPGD-PIB loop.

    Attributes:
        disturb_v: The signed perturbation that was applied.
        pos_img: Frame captured at ``init_v + disturb_v``.
        neg_img: Frame captured at ``init_v - disturb_v``.
        pos_value: Objective measured on ``pos_img``.
        neg_value: Objective measured on ``neg_img``.
        pos_ratio: Ratio reported alongside ``pos_value``.
        neg_ratio: Ratio reported alongside ``neg_value``.
        diff: Logged objective difference, scaled by the caller's sign.
        gradient: The SPGD gradient handed to ``optimizer.update``.
        init_v: The voltages after the update (unchanged if it was rejected).
        brightness: The frame-brightness statistic the path reported.
        applied: Whether the update passed the neighbour-difference check.
        aborted: Whether the caller asked to stop (live window closed).
    """

    disturb_v: np.ndarray
    pos_img: np.ndarray
    neg_img: np.ndarray
    pos_value: float
    neg_value: float
    pos_ratio: float
    neg_ratio: float
    diff: float
    gradient: np.ndarray
    init_v: np.ndarray
    brightness: float
    applied: bool
    aborted: bool


@contextmanager
def open_pib_bench(
    *,
    cam_type: str,
    cam_id: int,
    exposure_time_ms: float,
    cam_size: int,
    target_max_brightness: float,
    center: Any,
    dm: DM | None,
    create_dm: Callable[[], DM],
    init_v: Sequence[float] | np.ndarray | None,
    dm_unit_mask: Sequence[bool] | np.ndarray | None,
    init_fallback_target_max: int,
    on_probe: Callable[[np.ndarray, tuple[int, int]], None] | None = None,
) -> Iterator[PibBench]:
    """Open the camera + DM and take the flat-field reading.

    Args:
        cam_type: Camera backend name.
        cam_id: Camera device id.
        exposure_time_ms: Fixed exposure in ms (0 = auto / keep).
        cam_size: Side length of the square region of interest.
        target_max_brightness: Auto-exposure target peak (0 = keep).
        center: Centre spec (``None`` / mode string / ``(x, y)``).
        dm: An already-created DM, or ``None`` to create one.
        create_dm: Factory for the DM, called only when ``dm`` is ``None``.
        init_v: Initial DM voltages (empty / ``None`` = all zeros).
        dm_unit_mask: Active-actuator mask (``None`` = the DM's default).
        init_fallback_target_max: Brightness target used for the initial
            image when ``target_max_brightness`` is 0.  The two paths use
            different constants here, so it is a parameter.
        on_probe: Optional hook called with the probe image and the detected
            centre, before the brightness log.

    Yields:
        The opened bench and its flat-field reading.
    """
    with _camera_cls(cam_type)(
        cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=False
    ) as cam:
        if dm is None:
            dm = create_dm()
            dm.open()
            owns_dm = True
        else:
            if not dm.is_connected():
                dm.open()
            owns_dm = False

        # The DM is closed by the *caller* (its ``finally`` also tears down
        # the live window and, for the combined path, zeroes the DM).  This
        # frame only has to cover the case where the bench itself fails before
        # handing the DM over -- a failed probe capture would otherwise leave
        # the DM open.  ``yielded`` is set before the ``yield`` so that an
        # exception raised by the caller's body is not mistaken for a failed
        # setup (the caller's own ``finally`` closes the DM in that case).
        yielded = False
        try:
            if dm_unit_mask is None:
                dm_unit_mask = dm.default_dm_unit_mask
                if dm_unit_mask[0]:
                    logger.warning(
                        "dm_unit_mask[0] is True, which means the first unit is active."
                    )
            mask = np.asarray(dm_unit_mask, dtype=np.float64)

            if init_v is None or len(init_v) == 0:
                _init_v = np.zeros(dm.DM_Num, dtype=np.float64)
            else:
                _init_v = np.array(init_v, dtype=np.float64)
            dm.send_voltages(_init_v, 0.5)

            probe_img = capture_with_exposure(
                cam,
                0.0,
                TEST_EXPOSURE_TIME_BRIGHTNESS,
                auto_n_sample=20,
            )
            detected = resolve_spot_center(
                cam,
                probe_img,
                center,
                detect_fn=smart_zero_order_center,
                n_sample=10,
            )
            if on_probe is not None:
                on_probe(probe_img, detected)
            log_center_brightness(probe_img, detected, get_camera_exposure_ms(cam))

            # Positional on purpose: the real driver names the second
            # parameter ``size`` while the simulation shim names it
            # ``img_size``, so a keyword call would break one of the two.
            img_size, window_center = cam.reset_window(detected, (cam_size, cam_size))
            logger.info(f"reset window center @ {window_center}")

            init_img = capture_with_exposure(
                cam,
                exposure_time_ms,
                target_max_brightness or init_fallback_target_max,
                auto_n_sample=20,
            )
            logger.debug(
                f"Initial image max brightness: {np.max(init_img)} "
                f"@ {get_camera_exposure_ms(cam)}ms"
            )

            bench = PibBench(
                cam=cam,
                dm=dm,
                owns_dm=owns_dm,
                init_v=_init_v,
                unit_mask=mask,
                center=detected,
                window_center=window_center,
                img_size=init_img.shape[::-1],
                init_img=init_img,
                probe_img=probe_img,
            )
            yielded = True
            yield bench
        finally:
            if owns_dm and not yielded:
                dm.close()


def spid_pib_epoch(
    *,
    dm: DM,
    cam: Any,
    init_v: np.ndarray,
    delta: float,
    unit_mask: np.ndarray,
    rng: Any,
    optimizer: Base,
    evaluate: Callable[[np.ndarray], tuple[float, float]],
    maximize: bool,
    diff_sign: float,
    epoch: int = 0,
    on_render: Callable[[np.ndarray, np.ndarray, int], bool] | None = None,
    on_saturation: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> SpidPibEpoch:
    """Run one paired-measurement SPGD epoch and apply the resulting update.

    The perturbation is drawn from ``rng``, which must expose
    ``binomial(1, 0.5, (n,))`` -- a ``numpy.random.Generator`` for the SPGD
    path, the ``numpy.random`` module itself for the combined path.  The two
    streams differ on purpose: it is why the paths walk apart at epoch 1.

    Args:
        dm: The DM to drive.
        cam: The opened camera.
        init_v: The voltages to perturb around.
        delta: Perturbation magnitude.
        unit_mask: Active-actuator mask, as float64.
        rng: Source of the +/- perturbation pattern.
        optimizer: The algorithm optimizer that turns the gradient into a step.
        evaluate: Maps a frame to ``(value, ratio)``.
        maximize: Whether the objective is maximised.
        diff_sign: Sign applied to the logged objective difference.
        epoch: The epoch number, handed to the render hook.
        on_render: Optional live-view hook, called with the positive frame,
            the voltages and the epoch number; returning ``False`` aborts the
            epoch before any update is applied.
        on_saturation: Optional hook that owns the path's saturation policy.
            It is called with the two frames, applies whatever remedy the path
            wants (resampling the exposure, rescaling the optimizer), and
            returns the brightness statistic the path reports.

    Returns:
        The epoch's measurements and the voltages after the update.
    """
    disturb_v = rng.binomial(1, 0.5, (dm.DM_Num,)).astype(float) * 2.0 - 1.0
    disturb_v = disturb_v * delta * unit_mask

    dm.send_voltages(init_v + disturb_v)
    pos_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
    pos_value, pos_ratio = evaluate(pos_img)

    dm.send_voltages(init_v - disturb_v)
    neg_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
    neg_value, neg_ratio = evaluate(neg_img)

    if on_render is not None and not on_render(pos_img, init_v, epoch):
        return SpidPibEpoch(
            disturb_v=disturb_v,
            pos_img=pos_img,
            neg_img=neg_img,
            pos_value=pos_value,
            neg_value=neg_value,
            pos_ratio=pos_ratio,
            neg_ratio=neg_ratio,
            diff=0.0,
            gradient=np.zeros_like(init_v),
            init_v=init_v,
            brightness=0.0,
            applied=False,
            aborted=True,
        )

    brightness = 0.0
    if on_saturation is not None:
        brightness = on_saturation(pos_img, neg_img)

    diff = (float(pos_value) - float(neg_value)) * diff_sign
    gradient = spgd_gradient(pos_value, neg_value, disturb_v, maximize=maximize)
    update = optimizer.update(gradient)
    to_update_v = np.clip(init_v - update, dm.V_Min, dm.V_Max)

    applied = dm.check_dm_unit_grad_safe(to_update_v)
    return SpidPibEpoch(
        disturb_v=disturb_v,
        pos_img=pos_img,
        neg_img=neg_img,
        pos_value=pos_value,
        neg_value=neg_value,
        pos_ratio=pos_ratio,
        neg_ratio=neg_ratio,
        diff=diff,
        gradient=gradient,
        init_v=to_update_v if applied else init_v,
        brightness=brightness,
        applied=applied,
        aborted=False,
    )
