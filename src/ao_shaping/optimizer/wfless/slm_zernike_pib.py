"""SPGD optimization for PIB using SLM with Zernike coefficient control.

This module replaces DM voltage control with SLM phase modulation via Zernike
polynomial coefficients. The optimization perturbs Zernike coefficients,
displays the resulting phase pattern on the SLM, and measures the PIB metric
from the camera image.

Key differences from pib.py:
- DM (NlightDM) → SLM (Santec)
- Voltage vectors → Zernike coefficient vectors
- dm.send_voltages(v) → slm.display_data(phase_pattern)
- No neighbor voltage safety checks (SLM has no such constraint)
- No tabu search / adaptive neighborhood search (simplified)

Example:
    >>> from ao_shaping.optimizer.wfless.slm_zernike_pib import optimize_slm_zernike_pib
    >>> recorder = optimize_slm_zernike_pib(
    ...     center="shape",
    ...     epochs=2000,
    ...     n_max=4,
    ...     delta=0.1,
    ...     cam_id=0,
    ...     slm_number=1,
    ... )
"""

from __future__ import annotations

import inspect
import os
import time

import tqdm
import numpy as np
import matplotlib.pylab as plt

from ao_shaping.drivers import CameraStreamManager
from ao_shaping.drivers.slm import Santec
from ao_shaping.drivers.slm.santec import MEMORY_MODE_INTERNAL
from ao_shaping.optimizer.wfless.slm_square_shaping import _zernike_indices
from ao_shaping.optimizer.spgd import spgd_gradient
from ao_shaping.utils.pattern_helper import PatternHelper
from ao_shaping.algorithm.adam import AdaMOD, Adam, AdamW, Base, Muno, MunoW, SGD
from ao_shaping.utils import logger, Recorder
from ao_shaping.utils.file import gen_date_dir, gen_date_str
from ao_shaping.utils.spots_calc import centroid, radius
from ao_shaping.algorithm.target_func import ImageTargetFunc
from ao_shaping.utils.zernike_calc import calc_n_zernike_terms

# adam parameters
beta1 = 0.9
beta2 = 0.99
beta3 = 0.9999

# camera parameters
CAM_SAMPLE_ITER = 1
ADVISE_EXPOSURE_TIME_BRIGHTNESS = int(255 / 3)
TEST_EXPOSURE_TIME_BRIGHTNESS = 220
IDEAL_SPOT_RADIUS = int(os.environ.get("IDEAL_SPOT_RADIUS", 6))

# Safety caps for learning_schedule: a single SPGD step must not be able to
# traverse the Zernike coefficient clip range (+/-5 in the loop). Uncapped the
# schedule returned lr=6 / delta=5, which saturated the clip every step and
# diverged on hardware (PIB 0.70 -> 0.08, peak brightness 199 -> 15).
SCHEDULE_MAX_LR = 1.0
SCHEDULE_MAX_DELTA = 0.5

# slm parameters
SLM_RESPONSE_TIME_S = 0.3  # Santec SLM-200 response time ~300ms
# On exit, leave the best-found phase on the SLM. The candidate set includes
# the initial (flat when starting from zeros, or a loaded) phase: if the search
# never improved on it, that phase is restored instead. Set False to skip
# touching the SLM on exit.
SLM_APPLY_BEST_ON_EXIT = True

# SLM resolution (from Santec.Panel_Res = (1920, 1200))
SLM_WIDTH = 1920
SLM_HEIGHT = 1200
SLM_RESOLUTION = (SLM_WIDTH, SLM_HEIGHT)

OPTIMIZER_MAP = {
    "adam": Adam,
    "adamw": AdamW,
    "adamod": AdaMOD,
    "sgd": SGD,
    "muno": Muno,
    "munow": MunoW,
}


def _create_optimizer(optimizer_type: str, dim: int, lr: float, **kwargs) -> Base:
    """Create the configured optimizer while filtering unsupported kwargs."""
    optimizer_cls = OPTIMIZER_MAP.get(optimizer_type.lower(), AdaMOD)
    filtered_kwargs = {}
    signature = inspect.signature(optimizer_cls.__init__)
    for key, value in kwargs.items():
        if key in signature.parameters:
            filtered_kwargs[key] = value
    return optimizer_cls(dim, lr=lr, **filtered_kwargs)


ZERNIKE_APERTURE_RADIUS = 300.0
"""Aperture radius (px) the Zernike phase is defined over.

Must match the illuminated beam radius on the SLM. The panel is 1920x1200, so
PatternHelper's default is half the short side = 600 px; this bench's beam
radius is only ~300 px. With a 600 px aperture only the inner half of the
polynomial lands on the beam, so every mode is nearly CONSTANT across the
illuminated area -- and a constant phase does not change the far field, i.e.
the correction silently does nothing (hardware-verified: flat vs a 2 rad
defocus were indistinguishable until this was matched).
"""


# Memory-slot range used for phase writes (never repeat a slot consecutively).
_SLOT_MIN, _SLOT_MAX = 2, 125
_SLOT_STATE = {"slot": _SLOT_MIN - 1}


def _display(slm, gray) -> int:
    """Display ``gray`` on a **rotating explicit memory slot**.

    ``slm.display_data(gray)`` (default slot handling) produced NO optical effect
    on this bench: verified on BOTH cameras, a 5 rad tilt did not move the spot
    and a uniform 0 vs 900 changed nothing. ``slm.display_data(gray,
    memory_number=<distinct slot>, memory_mode=MEMORY_MODE_INTERNAL)`` -- the
    mechanism ``tools/slm/slm_diagnose.py`` uses -- does refresh the LCOS. Since
    writing the *same* slot twice is a firmware no-op, slots are rotated.
    """
    _SLOT_STATE["slot"] = (
        _SLOT_MIN if _SLOT_STATE["slot"] >= _SLOT_MAX else _SLOT_STATE["slot"] + 1
    )
    slm.display_data(
        gray,
        memory_number=_SLOT_STATE["slot"],
        memory_mode=MEMORY_MODE_INTERNAL,
    )
    return _SLOT_STATE["slot"]


def _zernike_to_phase(
    coeffs: np.ndarray,
    n_max: int,
    pattern_helper: PatternHelper,
    radius: float | None = None,
) -> np.ndarray:
    """Convert a flat Zernike coefficient array to a radian phase pattern.

    Args:
        coeffs: Flat array of Zernike coefficients (amplitudes in wavelengths).
        n_max: Maximum Zernike radial order.
        pattern_helper: PatternHelper instance for phase generation.

    Returns:
        float64 raw radian phase pattern array with shape (SLM_HEIGHT, SLM_WIDTH).
        No mod-2π here — the caller must convert to grayscale via
        ``slm.create_phase_from_array()`` before ``slm.display_data()``
        (2026-09: PatternHelper no longer performs phase→gray; the SLM
        driver applies the 2π wrap on radian→grayscale conversion).
    """
    modes = _zernike_indices(n_max)
    coeffs_dict: dict[tuple[int, int], float] = {}
    for i, (n, m) in enumerate(modes):
        if i < len(coeffs):
            coeffs_dict[(n, m)] = float(coeffs[i])
    return pattern_helper.generate_zernike_polynomial(
        n_max=n_max,
        coefficients=coeffs_dict,
        radius=ZERNIKE_APERTURE_RADIUS if radius is None else radius,
    )


def learning_schedule(
    power_radius: float,
    ideal_r: float = IDEAL_SPOT_RADIUS,
    gradient_history: list[float] | None = None,
    pib_history: list[float] | None = None,
    epoch: int = 0,
) -> tuple[float, float]:
    """Dynamic learning rate scheduler based on power radius and convergence state.

    Args:
        power_radius: Current power radius.
        ideal_r: Ideal spot radius.
        gradient_history: Recent gradient magnitude history for convergence detection.
        pib_history: Recent PIB value history for convergence detection.
        epoch: Current iteration number.

    Returns:
        (lr, delta): Learning rate and perturbation amplitude.
    """
    # Base parameters: segmented by power_radius
    if power_radius <= ideal_r:
        base_lr, base_delta = 1.5, 1
    elif power_radius <= 2 * ideal_r:
        base_lr, base_delta = 2, 2
    elif power_radius <= 3 * ideal_r:
        base_lr, base_delta = 2.5, 3
    elif power_radius <= 4 * ideal_r:
        base_lr, base_delta = 3, 4
    elif power_radius <= 5 * ideal_r:
        base_lr, base_delta = 4.5, 5
    else:
        base_lr, base_delta = 6, 5

    # If no convergence history, return base parameters directly (still capped)
    if gradient_history is None or pib_history is None or len(gradient_history) < 5:
        return (
            min(base_lr, SCHEDULE_MAX_LR),
            min(base_delta, SCHEDULE_MAX_DELTA),
        )

    # Convergence state detection
    recent_grads = (
        list(gradient_history[-10:])
        if len(gradient_history) >= 10
        else gradient_history
    )
    recent_pibs = list(pib_history[-10:]) if len(pib_history) >= 10 else pib_history

    # Gradient trend (lower variance = more stable convergence)
    grad_mean = np.mean(recent_grads)
    grad_std = np.std(recent_grads) if len(recent_grads) > 1 else 0
    grad_cv = grad_std / (grad_mean + 1e-8)  # Coefficient of variation

    # PIB trend
    pib_mean = np.mean(recent_pibs)
    pib_std = np.std(recent_pibs) if len(recent_pibs) > 1 else 0
    pib_trend = (
        (recent_pibs[-1] - recent_pibs[0]) / (len(recent_pibs) + 1e-8)
        if len(recent_pibs) > 1
        else 0
    )

    # Dynamic adjustment factors
    lr_factor = 1.0
    delta_factor = 1.0

    # Case 1: Small gradient variance (stable convergence) → reduce lr and delta
    if grad_cv < 0.1:
        lr_factor = 0.5
        delta_factor = 0.5
    # Case 2: Medium gradient variance (normal fluctuation) → keep
    elif grad_cv < 0.3:
        lr_factor = 0.8
        delta_factor = 0.8
    # Case 3: Large gradient variance (oscillation) → reduce lr significantly
    elif grad_cv > 0.8:
        lr_factor = 0.3
        delta_factor = 1.2  # Increase exploration

    # Case 4: PIB not improving (possible local optimum)
    if abs(pib_trend) < 1e-5 and pib_std < 0.01:
        delta_factor = max(delta_factor, 1.5)
        lr_factor = min(lr_factor, 0.7)
    # Case 5: PIB decreasing (diverging) → reduce lr
    elif pib_trend < -0.001:
        lr_factor = 0.4
        delta_factor = 0.6
    # Case 6: PIB increasing (normal convergence) → keep or fine-tune
    elif pib_trend > 0.001:
        lr_factor = min(lr_factor, 1.1)

    # Early epoch warmup
    if epoch < 20:
        lr_factor *= 1.2
        delta_factor *= 1.1

    # Clamp adjustment range
    lr_factor = np.clip(lr_factor, 0.2, 2.0)
    delta_factor = np.clip(delta_factor, 0.3, 2.5)

    final_lr = min(base_lr * lr_factor, SCHEDULE_MAX_LR)
    final_delta = min(base_delta * delta_factor, SCHEDULE_MAX_DELTA)

    return final_lr, final_delta


def optimize_slm_zernike_pib(
    center,
    epochs,
    n_max: int = 4,
    r_bucket=0,
    delta: float = 0.1,
    lr: float = 0,
    exposure_time_ms: float = 80.0,
    shrink_iter: int = 0,
    shrink_ratio: float = 0.9,
    cam_id=0,
    show: bool = False,
    init_c=None,
    cam_size=250,
    target_max_brightness=40,
    slm_number: int = 1,
    slm_wavelength: int = 1064,
    optimizer_type: str = "adamod",
    random_seed: int | None = None,
    objective: str = "pib",
    zernike_radius: float = ZERNIKE_APERTURE_RADIUS,
    shift_x: int | None = 0,
    shift_y: int | None = 0,
    **kwargs,
):
    """Optimize PIB (Power in Bucket) using SLM with Zernike coefficient control.

    This function uses the SPGD (Stochastic Parallel Gradient Descent) algorithm
    to optimize Zernike coefficients displayed on an SLM, maximizing the power
    in a bucket (PIB) metric measured by a camera.

    Args:
        center: Center position for PIB calculation. Can be None (auto-detect),
            "mass" (centroid), "max" (brightest pixel), "shape" (threshold-based),
            or a tuple (x, y).
        epochs: Number of optimization iterations.
        n_max: Maximum Zernike radial order. Controls the number of modes.
        r_bucket: Bucket radius. If 0, auto-adjusted based on power radius.
        delta: Perturbation amplitude for Zernike coefficients.
        lr: Learning rate. If 0, auto-adjusted via learning_schedule.
        exposure_time_ms: Camera exposure time in ms. If 0, auto-exposure.
        shrink_iter: Shrink iteration count. If 0, no shrinking.
        shrink_ratio: Shrink ratio for bucket radius.
        cam_id: Camera device ID.
        show: Whether to display images during optimization.
        init_c: Initial Zernike coefficients. If None, starts from zeros.
        cam_size: Camera window size.
        target_max_brightness: Target max brightness for auto-exposure.
        slm_number: SLM device number (1-8).
        slm_wavelength: SLM wavelength in nm.
        optimizer_type: Optimizer type for gradient stage (adam/adamod/sgd/muno).
        random_seed: Random seed for reproducibility.
        objective: Optimization target: 'pib' (maximize), 'radiu' (minimize radius),
            'avg_radiu' (maximize average radius).
        **kwargs: Additional optimizer parameters.

    Returns:
        Recorder: Optimization history recorder.
    """
    delta = abs(delta)
    epochs = int(epochs)
    rng = np.random.default_rng(random_seed)

    if objective not in ("pib", "radiu", "avg_radiu"):
        raise ValueError(
            f"objective must be one of ('pib', 'radiu', 'avg_radiu'), got {objective}"
        )

    # Optimization mode mapping: pib and avg_radiu are maximized, radiu is minimized
    objective_mode = "max" if objective in ("pib", "avg_radiu") else "min"
    recorder = Recorder(mark=objective, mode=objective_mode)

    # History for convergence detection
    _gradient_history: list[float] = []
    _pib_history: list[float] = []
    _max_history_len = 50

    # Zernike mode count and index mapping
    nk = calc_n_zernike_terms(n_max)
    zernike_modes = _zernike_indices(n_max)

    # SLM pattern helper
    pattern_helper = PatternHelper(resolution=SLM_RESOLUTION, bits=10)

    with (
        CameraStreamManager(
            cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=False
        ) as cam,
        Santec(
            slm_number=slm_number,
            wavelength=slm_wavelength,
            shift_x=shift_x,
            shift_y=shift_y,
        ) as slm,
    ):
        # Initialize Zernike coefficients
        if init_c is None or len(init_c) == 0:
            _init_c = np.zeros(nk, dtype=np.float64)
        else:
            _init_c = np.array(init_c, dtype=np.float64)
            if len(_init_c) < nk:
                padded = np.zeros(nk, dtype=np.float64)
                padded[: len(_init_c)] = _init_c
                _init_c = padded
            elif len(_init_c) > nk:
                _init_c = _init_c[:nk]

        # Reset SLM to flat phase
        initial_phase = slm.create_phase_from_array(
            _zernike_to_phase(_init_c, n_max, pattern_helper, zernike_radius)
        )
        _display(slm, initial_phase)
        time.sleep(SLM_RESPONSE_TIME_S)

        # Auto-exposure for initial image
        _img = cam.autoset_exposure_time_ms(
            target_max_brightness=TEST_EXPOSURE_TIME_BRIGHTNESS
        )

        def intellij_center(img):
            (h, w) = img.shape
            margin = int(IDEAL_SPOT_RADIUS)
            center = centroid(
                np.where(
                    img > np.max(img[: max(int(h // 50), 2), : max(int(w // 50), 2)]),
                    1,
                    0,
                )
            )
            (cx, cy) = center
            if np.all(
                img[cy - margin : cy + margin, cx - margin : cx + margin]
                >= np.max(img) * 0.4
            ):
                center = centroid(img)
            return center

        if center is None:
            center = intellij_center(_img)
        elif isinstance(center, str):
            _img = cam.get_numpy_image(10)
            if center == "mass":
                center = centroid(_img)
            elif center == "max":
                center = np.unravel_index(np.argmax(_img), _img.shape)[::-1]
            elif center == "shape":
                (h, w) = _img.shape
                center = centroid(
                    np.where(
                        _img
                        > np.max(_img[: max(int(h // 50), 2), : max(int(w // 50), 2)]),
                        1,
                        0,
                    )
                )
            else:
                raise ValueError(f"known center: {center}")
        else:
            center = center

        if show:
            plt.imshow(_img, cmap="gray")
            plt.scatter(x=center[0], y=center[1], c="red", s=5)
            plt.show()

        logger.info(
            f"Centroid brightness: {_img[center[::-1]]}@{center}, "
            f"Max brightness: {np.max(_img)} @ {cam.exposure_time}ms"
        )

        img_size = (cam_size, cam_size)
        img_size, center = cam.reset_window(center, img_size)
        logger.info(f"reset window center @ {center}")

        if exposure_time_ms > 0:
            cam.exposure_time = exposure_time_ms
            init_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
        elif 0 < target_max_brightness < 255 and target_max_brightness > 0:
            init_img = cam.autoset_exposure_time_ms(
                target_max_brightness=target_max_brightness, twice_valid=True
            )
        else:
            init_img = cam.autoset_exposure_time_ms(
                target_max_brightness=ADVISE_EXPOSURE_TIME_BRIGHTNESS, twice_valid=True
            )
        logger.debug(
            f"Initial Image Max brightness: {np.max(init_img)} @ {cam.exposure_time}ms"
        )
        img_size = init_img.shape[::-1]

        if r_bucket <= 0:
            _w, _h = img_size
            r_bucket = ImageTargetFunc(_w, _h, center).radius(init_img, energy=0.99)
            r_bucket = min(r_bucket, cam_size // 2) * shrink_ratio
            _fix_bucket = False
            logger.info(f"Use dynamic radiu @ {r_bucket}")
        else:
            _fix_bucket = True

        if (
            shrink_ratio <= 0
            or np.isclose(shrink_ratio, 1.0)
            or r_bucket <= IDEAL_SPOT_RADIUS
        ):
            update_iter = max(1, epochs)
        else:
            shrink_span = np.log(IDEAL_SPOT_RADIUS / r_bucket) / np.log(shrink_ratio)
            if np.isfinite(shrink_span) and shrink_span > 0:
                update_iter = max(1, int(epochs * 0.8 // shrink_span))
            else:
                update_iter = max(1, epochs)
        _init_r = r_bucket

        target_func = ImageTargetFunc.build_from_init_image(init_img)

        # Objective calculation functions
        def test_pib(img):
            return target_func.pib(img, IDEAL_SPOT_RADIUS)[1]

        to_min = 1
        if objective == "pib":
            # Maximize PIB: negate the SPGD estimate so `_init_c - update`
            # ascends the objective. Same convention as pib.py's intended
            # `to_min = -1`; without this flip the loop minimizes PIB.
            to_min = -1

            def calc_objective(img):
                pib, pib_ratio = target_func.pib(img, r_bucket)
                return pib, pib_ratio
        elif objective == "radiu":

            def calc_objective_radiu(img):
                r = target_func.radius(img, energy=0.99)
                return r, 0.0

            calc_objective = calc_objective_radiu
        elif objective == "avg_radiu":
            # Maximize average radius: same sign flip as PIB.
            to_min = -1

            def calc_objective_avg(img):
                return target_func.avg_radius(img, moment=1.0)

            calc_objective = calc_objective_avg

        j, pib_ratio = calc_objective(init_img)

        optimizer = _create_optimizer(
            optimizer_type=optimizer_type,
            dim=nk,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            **kwargs,
        )
        if lr == 0:
            optimizer.lr, delta = learning_schedule(
                radius(init_img, center=center, energy=0.8),
                gradient_history=_gradient_history,
                pib_history=_pib_history,
                epoch=0,
            )

        # Track the objective's OWN value. For "pib" that is the exposure-
        # independent bucket ratio (matches the logged column); for the other
        # objectives it is the value the gradient uses -- e.g. the encircle
        # radius, which must be MINIMISED (a `>` comparison would keep the worst).
        best_objective = float(test_pib(init_img)) if objective == "pib" else float(j)
        # Baseline objective of the initial phase (flat when ``init_c`` is
        # empty). Used on exit to decide between the best phase and flat.
        _initial_objective = best_objective
        best_j = float(j)
        best_objective_ratio = float(pib_ratio)
        best_c = _init_c.copy()
        best_img = init_img.copy()
        last_best_epoch = 0

        recorder.append(
            {
                "J": j,
                objective: best_objective,
                "_p%": pib_ratio,
                "_max_r": _init_r,
                "_c": _init_c,
                "_img": init_img,
                "_diff": 0,
                "lr": optimizer.lr,
                "r": r_bucket,
                "delta": delta,
                "_epoch": 0,
                "exp_t": cam.exposure_time,
                "max_brt": np.max(init_img),
                "_grad": np.zeros_like(_init_c),
                "optimizer": optimizer_type,
                f"best_{objective}": best_objective,
            }
        )

        with tqdm.tqdm(
            total=epochs, desc=f"slm_zernike iter {epochs}", dynamic_ncols=True
        ) as bar:
            for epoch in range(1, epochs + 1):
                # Generate random perturbation (±1 pattern)
                disturb_c = rng.binomial(1, 0.5, (nk,)).astype(float) * 2.0 - 1.0
                disturb_c = disturb_c * delta

                # Positive perturbation
                _pos_c = np.clip(_init_c + disturb_c, -5.0, 5.0)
                pos_phase = slm.create_phase_from_array(
                    _zernike_to_phase(_pos_c, n_max, pattern_helper, zernike_radius)
                )
                _display(slm, pos_phase)
                time.sleep(SLM_RESPONSE_TIME_S)
                pos_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                pos_obj, pos_obj_ratio = calc_objective(pos_img)

                # Negative perturbation
                _neg_c = np.clip(_init_c - disturb_c, -5.0, 5.0)
                neg_phase = slm.create_phase_from_array(
                    _zernike_to_phase(_neg_c, n_max, pattern_helper, zernike_radius)
                )
                _display(slm, neg_phase)
                time.sleep(SLM_RESPONSE_TIME_S)
                neg_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                neg_obj, neg_obj_ratio = calc_objective(neg_img)

                # Auto-exposure adjustment if saturated
                max_brightness = max([np.max(pos_img), np.max(neg_img)])
                if max_brightness == 255 and exposure_time_ms == 0:
                    _resample_img = cam.autoset_exposure_time_ms(
                        target_max_brightness, twice_valid=False
                    )
                    optimizer.scale_momentum(np.sum(_resample_img) / np.sum(pos_img))

                pos_j, neg_j = pos_obj, neg_obj
                # `diff` is kept for logging; the SPGD sign comes from the
                # shared helper (optimizer/spgd.py) so it cannot be
                # hand-inverted again (this site maximised/minimised the wrong
                # way until the to_min fix). Cast to float: bucket sums are
                # unsigned.
                diff = (float(pos_j) - float(neg_j)) * to_min
                gradient = spgd_gradient(
                    pos_j, neg_j, disturb_c, maximize=(to_min == -1)
                )
                update = optimizer.update(gradient)
                _to_update_c = np.clip(_init_c - update, -5.0, 5.0)
                _init_c = _to_update_c

                # Value logged under the objective's own name and used by the
                # Recorder to pick its best row: the bucket ratio for "pib",
                # otherwise the objective the gradient optimises (e.g. radius).
                objective_val = (
                    float(test_pib(pos_img)) if objective == "pib" else float(pos_j)
                )
                objective_ratio = (pos_obj_ratio + neg_obj_ratio) / 2
                J = (pos_j + neg_j) / 2

                # Bucket radius shrink
                if epoch % update_iter == update_iter - 1:
                    _init_r = max(_init_r * shrink_ratio, IDEAL_SPOT_RADIUS)

                if (
                    (
                        epoch % update_iter == update_iter - 1
                        or (shrink_iter > 0 and epoch % shrink_iter == shrink_iter - 1)
                        or objective_ratio >= 0.99
                    )
                    and not _fix_bucket
                    and objective_val > 0
                ):
                    power_radio = radius(pos_img, center=center, energy=0.8)
                    _pr = power_radio * shrink_ratio
                    _r = max(r_bucket * shrink_ratio + 1, IDEAL_SPOT_RADIUS, r_bucket)
                    r_bucket = min(_r, _pr, _init_r)
                    if lr == 0:
                        _grad_mag = float(np.linalg.norm(gradient))
                        _gradient_history.append(_grad_mag)
                        _pib_history.append(float(objective_val))
                        if len(_gradient_history) > _max_history_len:
                            _gradient_history.pop(0)
                            _pib_history.pop(0)
                        optimizer.lr, delta = learning_schedule(
                            power_radius=r_bucket,
                            gradient_history=_gradient_history,
                            pib_history=_pib_history,
                            epoch=epoch,
                        )

                # Track best result in the objective's own direction.
                improved = (
                    objective_val > best_objective + 1e-4
                    if objective_mode == "max"
                    else objective_val < best_objective - 1e-4
                )
                if improved:
                    best_objective = float(objective_val)
                    best_j = float(J)
                    best_objective_ratio = float(objective_ratio)
                    # objective_val / pos_img were measured on the PERTURBED
                    # phase `_pos_c`, so the saved coefficients must be `_pos_c`
                    # too. Saving the clean `_init_c` made `save_best` write a
                    # configuration that was never measured -- re-applying it
                    # did not reproduce the reported metric (hardware-verified).
                    best_c = _pos_c.copy()
                    best_img = pos_img.copy()
                    last_best_epoch = epoch

                log = {
                    "J": J,
                    "_p%": objective_ratio,
                    "_max_r": _init_r,
                    "pib": objective_val,
                    "_diff": diff,
                    "lr": optimizer.lr,
                    "r": r_bucket,
                    "delta": delta,
                    "_epoch": epoch,
                    "_c": _init_c,
                    "_img": pos_img,
                    "exp_t": cam.exposure_time,
                    "max_brt": max_brightness,
                    "_grad": gradient,
                }
                recorder.append(log)

                bar.set_postfix({k: v for k, v in log.items() if k[0] != "_"})
                bar.update(1)

        # On exit, leave the SLM at the best phase found. The initial (flat or
        # loaded) phase is one of the candidates: if the search never improved
        # on it, restore that instead of a worse "best".
        if SLM_APPLY_BEST_ON_EXIT:
            improved = (
                best_objective > _initial_objective + 1e-4
                if objective_mode == "max"
                else best_objective < _initial_objective - 1e-4
            )
            if improved:
                best_phase = slm.create_phase_from_array(
                    _zernike_to_phase(best_c, n_max, pattern_helper, zernike_radius)
                )
                _display(slm, best_phase)
                time.sleep(SLM_RESPONSE_TIME_S)
                logger.info(
                    "SLM left at best {} phase: {:.4f} @ epoch {} "
                    "(initial {:.4f})",
                    objective,
                    best_objective,
                    last_best_epoch,
                    _initial_objective,
                )
            else:
                slm.set_grayscale(0)
                logger.info(
                    "No {} improvement over the initial phase ({:.4f}); "
                    "SLM left at flat",
                    objective,
                    _initial_objective,
                )

        return recorder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SPGD PIB optimization using SLM with Zernike coefficients"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=2000, help="Number of iterations"
    )
    parser.add_argument(
        "-n", "--n_max", type=int, default=4, help="Max Zernike radial order"
    )
    parser.add_argument(
        "-c", "--center", type=str, default="shape", help="Center detection method"
    )
    parser.add_argument(
        "-r", "--r_bucket", type=float, default=0, help="Bucket radius (0=auto)"
    )
    parser.add_argument(
        "-d", "--delta", type=float, default=0.1, help="Perturbation amplitude"
    )
    parser.add_argument("--lr", type=float, default=0, help="Learning rate (0=auto)")
    parser.add_argument(
        "-t", "--exposure_time_ms", type=float, default=80.0, help="Exposure time (ms)"
    )
    parser.add_argument("--cam_id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--slm_number", type=int, default=1, help="SLM device number")
    parser.add_argument(
        "--slm_wavelength", type=int, default=1064, help="SLM wavelength (nm)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="adamod", help="Optimizer type"
    )
    parser.add_argument(
        "--objective", type=str, default="pib", help="Optimization target"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--show", action="store_true", help="Show images during optimization"
    )
    parser.add_argument("--cam_size", type=int, default=250, help="Camera window size")

    args = parser.parse_args()

    recorder = optimize_slm_zernike_pib(
        center=args.center,
        epochs=args.epochs,
        n_max=args.n_max,
        r_bucket=args.r_bucket,
        delta=args.delta,
        lr=args.lr,
        exposure_time_ms=args.exposure_time_ms,
        cam_id=args.cam_id,
        slm_number=args.slm_number,
        slm_wavelength=args.slm_wavelength,
        optimizer_type=args.optimizer,
        objective=args.objective,
        random_seed=args.seed,
        show=args.show,
        cam_size=args.cam_size,
    )

    best_iter, (_, best_val) = recorder.get_best_iter()
    logger.info(
        f"Optimization complete. Best {args.objective}: {best_val:.4f} @ epoch {best_iter.get('_epoch', 'N/A')}"
    )
    save_file = (
        gen_date_dir("data") / f"slm_zernike_{args.objective}_{gen_date_str()}.csv"
    )
    recorder.save_dataframe(save_file)
    logger.info(f"Results saved to: {save_file}")
