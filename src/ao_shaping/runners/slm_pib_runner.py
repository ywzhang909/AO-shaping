"""Run slm-pib — PIB shaping with a Santec SLM driven by Zernike coefficients.

Usage:
    python src/ao_shaping/main.py slm-pib [OPTIONS]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import click
import numpy as np
from loguru import logger

from ao_shaping.algorithm.heuristic.search import heuristic_algorithm_choices
from ao_shaping.drivers import MIICamera, Santec
from ao_shaping.drivers.ccd import DahengCamera
from ao_shaping.drivers.ccd.common import create_camera
from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    ALGORITHM_CHOICES,
    optimize_slm_zernike_pib,
)
from ao_shaping.runners.runner_common import (
    CameraParamsPib,
    HeuristicParams,
    ObjectiveParamsPib,
    RunParams,
    SlmParams,
    SpgdParamsPib,
    build_debug_save_paths,
    config_payload,
    camera_options,
    run_options,
    save_optimization_debug_artifacts,
    slm_extended_options,
)
from ao_shaping.utils.io.cli_helpers import get_date_dir_name, get_debug_mode, parse_tuple, setup_coredumpy
from ao_shaping.utils.io.file import Recorder, save_recorder_debug_artifacts

# Backward-compatible aliases — tests and external callers may still import
# the old names from this module.
ObjectiveParams = ObjectiveParamsPib
CameraParams = CameraParamsPib
SpgdParams = SpgdParamsPib

# --- debug artifact fields -------------------------------------------------

DEBUG_EXCLUDE = {"_img", "_grad", "_diff"}

_DEBUG_IMG_KEYS = ("_img", "_diff")
_DEBUG_1D_KEYS = ("_c",)
_DEBUG_2D_KEYS = ("_grad",)
_DEBUG_SCALAR_KEYS = (
    "J",
    "_p%",
    "_max_r",
    "_r",
    "lr",
    "delta",
    "r",
    "exp_t",
    "max_brt",
    "_epoch",
    "J0",
    "r0",
    "J0_r",
    "dJ0",
    "dJ0_r",
)
_DEBUG_OBJECTIVE_KEYS = ("pib", "radiu", "avg_radiu")


def _config_payload(obj: Any) -> dict[str, Any]:
    """Serialize a search-config dataclass for the JSON debug sidecar.

    Drops unset (None) fields and fields left at their dataclass default;
    the identity field (``algorithm`` / ``name``) is always kept so the
    search family survives the round trip.
    """
    return config_payload(obj)


def _save_debug_artifacts(
    res: Recorder,
    objective: "ObjectiveParamsPib",
    config: "SlmParams",
    obj_or_heur: "ObjectiveParamsPib | HeuristicParams",
    root_dir: str,
) -> Any:
    """Write PNG/pkl/json debug artifacts for the recorded search.

    Delegates to :func:`ao_shaping.utils.io.file.save_recorder_debug_artifacts`
    with the slm-pib key set. ``config`` is accepted for signature stability
    (the SLM config is part of the JSON sidecar via ``obj_or_heur``).
    """
    return save_recorder_debug_artifacts(
        res,
        root_dir=root_dir,
        subdir_prefix=f"slm_pib_{objective.name}",
        scalar_keys=_DEBUG_SCALAR_KEYS,
        objective_keys=_DEBUG_OBJECTIVE_KEYS,
        img_keys=_DEBUG_IMG_KEYS,
        d1_keys=_DEBUG_1D_KEYS,
        d2_keys=_DEBUG_2D_KEYS,
        json_payload=_config_payload(obj_or_heur),
        title=f"slm-pib {objective.name} search",
    )


# --- option decorators (applied to both subcommands) -----------------------
# ``run_options`` / ``slm_extended_options`` come from runner_common;
# ``_camera_options`` stays local because slm-pib defaults cam_size=250.


def _camera_options(fn):
    fn = click.option("--cam-id", default=0, help="CCD camera device ID")(fn)
    fn = click.option(
        "--cam_type",
        type=click.Choice(["miicam", "daheng", "sim"]),
        default="daheng",
        help="CCD camera backend (sim = 2f-Fourier numerical simulation, no hardware).",
    )(fn)
    fn = click.option(
        "--exposure_time_ms",
        type=float,
        default=80.0,
        help="CCD exposure time in ms (0 = auto-exposure).",
    )(fn)
    fn = click.option(
        "--cam_size", type=int, default=250, help="CCD window size in pixels."
    )(fn)
    fn = click.option(
        "-c",
        "--center",
        default=None,
        help="Center: 'mass' / 'max' / 'shape' or 'x,y'.",
    )(fn)
    return fn


def _objective_options(fn):
    fn = click.option(
        "--objective",
        type=click.Choice(["pib", "radiu", "avg_radiu", "shape", "roi_pib"]),
        default="pib",
        help="Optimization objective.",
    )(fn)
    fn = click.option(
        "--target_max_brightness",
        type=int,
        default=40,
        help="Target max brightness for auto-exposure.",
    )(fn)
    fn = click.option(
        "-r",
        "--r_bucket",
        type=int,
        default=0,
        help="Bucket radius (0 = auto from power radius).",
    )(fn)
    fn = click.option(
        "--target_size", type=float, default=44.0, help="Target extent in camera px."
    )(fn)
    fn = click.option(
        "--target_aspect_ratio",
        type=float,
        default=4.0 / 3.0,
        help="Width:height ratio for a rectangular target.",
    )(fn)
    fn = click.option(
        "--target_center_smooth",
        type=int,
        default=3,
        help="Frames averaged for the target centre estimate.",
    )(fn)
    fn = click.option(
        "--target_shape",
        type=click.Choice(["rectangle", "circle", "square", "ellipse", "annulus"]),
        default=None,
        help="Target ROI shape (implies the 'shape' objective).",
    )(fn)
    fn = click.option(
        "--shape_schedule",
        is_flag=True,
        default=False,
        help="Use the coarse->fine shaping weight schedule.",
    )(fn)
    fn = click.option(
        "--max_energy_loss",
        type=float,
        default=0.6,
        help="Max allowed in-ROI energy loss fraction (0 disables the guard).",
    )(fn)
    fn = click.option(
        "--w_uniformity", type=float, default=2.0, help="Uniformity penalty weight."
    )(fn)
    fn = click.option(
        "--w_peak", type=float, default=0.5, help="Peak penalty weight."
    )(fn)
    fn = click.option(
        "--w_displacement", type=float, default=0.0, help="Displacement penalty weight."
    )(fn)
    fn = click.option(
        "--log_uniformity",
        is_flag=True,
        default=False,
        help="Use log1p(u) instead of u/(1+u) for the uniformity term.",
    )(fn)
    return fn


# --- shared execution helpers ------------------------------------------------


def _objective_mode(objective: str) -> str:
    """Map an objective name to the recorder mode (max/min)."""
    return "max" if objective in ("pib", "avg_radiu", "shape", "roi_pib") else "min"


def _load_initial_coeffs(
    load_file: str | None, init_c: str | None
) -> np.ndarray | None:
    """Resolve the initial Zernike coefficients from a file or an explicit value."""
    if load_file is not None:
        arr = np.load(load_file)
        logger.info("Loaded initial Zernike coefficients from {}", load_file)
        return arr
    if init_c is not None:
        if init_c.strip().startswith("{"):
            import json

            coeffs = json.loads(init_c)
            return np.array(list(coeffs.values()))
        return np.array([float(x) for x in init_c.split(",")])
    return None


def _apply_disturbance(
    coeffs: np.ndarray | None, magnitude: float, rng: np.random.Generator
) -> np.ndarray:
    """Add a random Gaussian perturbation of the given magnitude to coefficients."""
    base = np.zeros_like(coeffs) if coeffs is None else coeffs.copy()
    base = base + rng.standard_normal(base.shape) * magnitude
    return base


@dataclass
class SlmPibConfig:
    """Aggregates every parameter group for one slm-pib run.

    Uses the slm-pib-specific dataclasses (``CameraParamsPib``,
    ``SpgdParamsPib``, ``ObjectiveParamsPib``) so the defaults stay
    consistent with the click decorators.
    """

    run: RunParams
    camera: CameraParamsPib
    slm: SlmParams
    objective: ObjectiveParamsPib
    search: SpgdParamsPib | HeuristicParams


def _optimizer_kwargs(
    cfg: SlmPibConfig,
    search: SpgdParamsPib | HeuristicParams,
) -> dict[str, Any]:
    """Flatten the aggregated config into the optimizer's flat keyword arguments."""
    slm = cfg.slm
    cam = cfg.camera
    obj = cfg.objective

    zernike_radius: float | None = (
        slm.zernike_radius if slm.zernike_radius > 0 else None
    )

    kwargs: dict[str, Any] = {
        "center": cam.center,
        "epochs": search.epochs,
        "n_max": slm.n_max,
        "r_bucket": obj.r_bucket,
        "exposure_time_ms": cam.exposure_time_ms,
        "cam_id": cam.cam_id,
        "cam_type": cam.cam_type,
        "cam_size": cam.cam_size,
        "target_max_brightness": obj.target_max_brightness,
        "slm_number": slm.slm_number,
        "slm_wavelength": slm.slm_wavelength,
        "objective": obj.name,
        "target_shape": obj.target_shape,
        "target_size": obj.target_size,
        "target_aspect_ratio": obj.target_aspect_ratio,
        "target_center_smooth": obj.target_center_smooth,
        "shape_schedule": obj.shape_schedule,
        "max_roi_energy_loss": obj.max_roi_energy_loss,
        "w_uniformity": obj.w_uniformity,
        "w_peak": obj.w_peak,
        "w_displacement": obj.w_displacement,
        "log_uniformity": obj.log_uniformity,
        "zernike_radius": zernike_radius,
        "shift_x": slm.shift_x,
        "shift_y": slm.shift_y,
        "random_seed": None,
    }

    if isinstance(search, SpgdParamsPib):
        kwargs.update(
            {
                "delta": search.delta,
                "lr": search.lr,
                "optimizer_type": search.optimizer_type,
                "shrink_iter": search.shrink_iter,
                "shrink_ratio": search.shrink_ratio,
                "algorithm": "spgd",
                "pop_size": None,
                "show": search.show,
            }
        )
    else:  # HeuristicParams
        kwargs.update(
            {
                "algorithm": search.algorithm,
                "pop_size": search.pop_size,
                "show": search.show,
            }
        )

    return kwargs


def _parse_objective_args(**opts) -> ObjectiveParamsPib:
    """Build an :class:`ObjectiveParamsPib` from the shared objective click options."""
    return ObjectiveParamsPib(
        name=opts["objective"],
        target_max_brightness=opts["target_max_brightness"],
        r_bucket=opts["r_bucket"],
        target_size=opts["target_size"],
        target_aspect_ratio=opts["target_aspect_ratio"],
        target_center_smooth=opts["target_center_smooth"],
        target_shape=opts["target_shape"],
        shape_schedule=opts["shape_schedule"],
        max_roi_energy_loss=opts["max_energy_loss"],
        w_uniformity=opts["w_uniformity"],
        w_peak=opts["w_peak"],
        w_displacement=opts["w_displacement"],
        log_uniformity=opts["log_uniformity"],
    )


# --- click group + subcommands ---------------------------------------------


@click.group(invoke_without_command=True)
@click.pass_context
def run(ctx: click.Context) -> None:
    """PIB shaping with a Santec SLM driven by Zernike coefficients.

    Without a subcommand, the SPGD (gradient) search runs.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(spgd, **ctx.params)


@click.command(name="spgd")
@run_options
@_camera_options
@slm_extended_options
@_objective_options
@click.option("-e", "--epochs", type=int, default=2000, help="Optimization iterations.")
@click.option("--delta", type=float, default=0.2, help="SPGD perturbation amplitude (rad).")
@click.option("--lr", type=float, default=0.0, help="SPGD learning rate (0 = auto).")
@click.option(
    "--optimizer_type",
    type=click.Choice(
        ["adam", "adamw", "adamod", "sgd", "muno", "munow"], case_sensitive=False
    ),
    default="adamod",
    show_default=True,
    help="SPGD gradient optimizer.",
)
@click.option("--shrink_iter", type=int, default=0, help="Iterations before radius/step shrink.")
@click.option("--shrink_ratio", type=float, default=0.9, help="Radius/step shrink ratio.")
@click.option("--show", is_flag=True, default=False, help="Open a live display window.")
@click.pass_context
def spgd(
    ctx: click.Context,
    dir: str,
    debug: bool,
    cam_id: int,
    cam_type: str,
    exposure_time_ms: float,
    cam_size: int,
    center,
    slm_number: int,
    slm_wavelength: int,
    n_max: int,
    shift_x: int,
    shift_y: int,
    zernike_radius: float,
    load_file: str | None,
    init_c: str | None,
    **obj_opts,
) -> None:
    """Run the SPGD (Stochastic Parallel Gradient Descent) search."""
    run_cfg = RunParams(dir=dir, debug=debug)
    camera = CameraParamsPib(
        cam_id=cam_id,
        cam_type=cam_type,
        exposure_time_ms=exposure_time_ms,
        cam_size=cam_size,
        center=center,
    )
    slm = SlmParams(
        slm_number=slm_number,
        slm_wavelength=slm_wavelength,
        n_max=n_max,
        shift_x=shift_x,
        shift_y=shift_y,
        zernike_radius=zernike_radius,
        load_file=load_file,
        init_c=init_c,
    )
    objective = _parse_objective_args(**obj_opts)
    search = SpgdParamsPib(
        epochs=obj_opts.pop("epochs", 2000),
        delta=obj_opts.pop("delta", 0.2),
        lr=obj_opts.pop("lr", 0.0),
        optimizer_type=obj_opts.pop("optimizer_type", "adamod"),
        shrink_iter=obj_opts.pop("shrink_iter", 0),
        shrink_ratio=obj_opts.pop("shrink_ratio", 0.9),
        show=obj_opts.pop("show", False),
    )
    _execute(run_cfg, camera, slm, objective, search)


@click.command(name="heuristic")
@run_options
@_camera_options
@slm_extended_options
@_objective_options
@click.option(
    "--algorithm",
    type=click.Choice(heuristic_algorithm_choices()),
    default="ga",
    help="Black-box search algorithm.",
)
@click.option("--pop_size", type=int, default=None, help="Population size (ga/pso/cem/de).")
@click.option("-e", "--epochs", type=int, default=2000, help="Optimization iterations.")
@click.option("--show", is_flag=True, default=False, help="Open a live display window.")
@click.pass_context
def heuristic(
    ctx: click.Context,
    dir: str,
    debug: bool,
    cam_id: int,
    cam_type: str,
    exposure_time_ms: float,
    cam_size: int,
    center,
    slm_number: int,
    slm_wavelength: int,
    n_max: int,
    shift_x: int,
    shift_y: int,
    zernike_radius: float,
    load_file: str | None,
    init_c: str | None,
    algorithm: str,
    pop_size: int | None,
    epochs: int,
    show: bool,
    **obj_opts,
) -> None:
    """Run a black-box heuristic search (ga/pso/sa/hc/rs/cem/de)."""
    run_cfg = RunParams(dir=dir, debug=debug)
    camera = CameraParamsPib(
        cam_id=cam_id,
        cam_type=cam_type,
        exposure_time_ms=exposure_time_ms,
        cam_size=cam_size,
        center=center,
    )
    slm = SlmParams(
        slm_number=slm_number,
        slm_wavelength=slm_wavelength,
        n_max=n_max,
        shift_x=shift_x,
        shift_y=shift_y,
        zernike_radius=zernike_radius,
        load_file=load_file,
        init_c=init_c,
    )
    objective = _parse_objective_args(**obj_opts)
    search = HeuristicParams(algorithm=algorithm, pop_size=pop_size, epochs=epochs, show=show)
    _execute(run_cfg, camera, slm, objective, search)


run.add_command(spgd, name="spgd")
run.add_command(heuristic, name="heuristic")


def _execute(
    run_cfg: RunParams,
    camera: CameraParamsPib,
    slm: SlmParams,
    objective: ObjectiveParamsPib,
    search: SpgdParamsPib | HeuristicParams,
) -> None:
    """Shared execution path for both search families."""
    setup_coredumpy()
    if camera.center is not None and not isinstance(camera.center, str):
        camera.center = parse_tuple(camera.center)

    kwargs = _optimizer_kwargs(
        SlmPibConfig(run_cfg, camera, slm, objective, search), search
    )
    init_c = _load_initial_coeffs(slm.load_file, slm.init_c)
    if init_c is not None:
        kwargs["init_c"] = init_c

    res = optimize_slm_zernike_pib(**kwargs)
    mode = _objective_mode(objective.name)
    if run_cfg.debug:
        _save_debug_artifacts(res, objective, slm, search, run_cfg.dir)

    final = res.history[-1]
    logger.info("SLM PIB done. Final {}", {k: final[k] for k in ("J", objective.name)})


if __name__ == "__main__":
    run()
