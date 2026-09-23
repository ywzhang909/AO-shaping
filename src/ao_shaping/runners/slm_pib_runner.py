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
    TARGET_SHAPE_CHOICES,
    optimize_slm_zernike_pib,
)
from ao_shaping.runners.runner_common import (
    build_debug_save_paths,
    save_optimization_debug_artifacts,
)
from ao_shaping.utils.io.cli_helpers import setup_coredumpy
from ao_shaping.utils.io.file import Recorder

# --- debug artifact fields -------------------------------------------------

_DEBUG_IMG_KEYS = ("_img",)
_DEBUG_1D_KEYS = ("_c",)
_DEBUG_2D_KEYS = ("_grad",)
_DEBUG_SCALAR_KEYS = (
    "J",
    "_p%",
    "_max_r",
    "_r",
    "_diff",
    "lr",
    "delta",
    "r",
    "exp_t",
    "max_brt",
    "_epoch",
    "w_pib",
    "w_rms",
    "pib_term",
    "rms_term",
)
_DEBUG_OBJECTIVE_KEYS = (
    "pib",
    "radiu",
    "avg_radiu",
    "rmse",
    "shape",
    "roi_pib",
    "rms_pib",
)


def _effective_objective_key(name: str, target_shape: Any) -> str:
    """Return the dict key the optimizer actually records for the objective value.

    The optimizer remaps ``objective`` to ``"shape"`` when ``target_shape`` is
    supplied (and the objective is not ``roi_pib``/``rms_pib``/``rmse``), so the
    recorder row key changes accordingly — the runner must follow the same remapping.
    """
    if target_shape is not None and name not in ("roi_pib", "rms_pib", "rmse"):
        return "shape"
    return name


def _config_payload(obj: Any) -> dict[str, Any]:
    """Serialize a search-config dataclass for the JSON debug sidecar.

    Drops unset (None) fields and fields left at their dataclass default;
    the identity field (``algorithm`` / ``name``) is always kept so the
    search family survives the round trip.
    """
    payload: dict[str, Any] = {}
    for key, value in obj.__dict__.items():
        if value is None:
            continue
        default = getattr(type(obj), key, None)
        if value == default and key not in ("algorithm", "name"):
            continue
        payload[key] = value
    return payload


def _save_debug_artifacts(
    res: Recorder,
    objective: "ObjectiveParams",
    config: "SlmParams",
    obj_or_heur: "ObjectiveParams | HeuristicParams",
    root_dir: str,
) -> Any:
    """Write PNG/pkl/json/h5 debug artifacts for the recorded search.

    ``objective`` and ``config`` are the slm-pib dataclasses (kept for the
    signature so the figure labels stay stable); ``obj_or_heur`` is whichever
    search-config object actually ran (SPGD or heuristic) and its ``__dict__``
    is the payload round-tripped in the json sidecar.
    """
    from ao_shaping.utils.io.file import save_history_hdf5

    eff_key = _effective_objective_key(objective.name, objective.target_shape)
    save_dir, saved_file_name = build_debug_save_paths(
        os.path.join(root_dir, "debug"),
        f"slm_pib_{eff_key}_{datetime.now():%Y%m%d_%H%M%S}",
    )
    png_path = saved_file_name.with_suffix(".png")
    pkl_path = saved_file_name.with_suffix(".pkl")
    json_path = saved_file_name.with_suffix(".json")
    h5_path = saved_file_name.with_suffix(".h5")

    data: dict[str, Any] = {}
    for rec in res.history:
        item: dict[str, Any] = {}
        for k in _DEBUG_SCALAR_KEYS + _DEBUG_OBJECTIVE_KEYS:
            if k in rec:
                item[k] = float(rec[k])
        for k in _DEBUG_IMG_KEYS:
            if k in rec:
                item[k] = np.asarray(rec[k])
        for k in _DEBUG_1D_KEYS:
            if k in rec:
                item[k] = np.asarray(rec[k], dtype=float)
        for k in _DEBUG_2D_KEYS:
            if k in rec:
                item[k] = np.asarray(rec[k])
        for k in list(rec.keys()):
            if k.startswith("best_"):
                item[k] = float(rec[k])
        if "_phase" in rec:
            item["_phase"] = np.asarray(rec["_phase"])
        data[int(rec["_epoch"])] = item

    save_optimization_debug_artifacts(
        data=data,
        png_path=png_path,
        pkl_path=pkl_path,
        json_path=json_path,
        title=f"slm-pib {eff_key} search",
        json_payload=_config_payload(obj_or_heur),
    )
    # Full-fidelity export: SLM phases + CCD frames per epoch in one HDF5.
    save_history_hdf5(
        res.history,
        h5_path,
        metadata={**_config_payload(obj_or_heur), "objective": eff_key},
    )
    return png_path


# --- parameter dataclasses -------------------------------------------------


@dataclass
class RunParams:
    """Global / run-wide options shared by both subcommands."""

    dir: str = "data"
    debug: bool = False


@dataclass
class CameraParams:
    """CCD camera options."""

    cam_id: int = 0
    cam_type: str = "daheng"
    exposure_time_ms: float = 80.0
    cam_size: int = 250
    center: Any = None


@dataclass
class SlmParams:
    """Santec SLM options."""

    slm_number: int = 1
    slm_wavelength: int = 1064
    n_max: int = 4
    shift_x: int = 0
    shift_y: int = 0
    zernike_radius: float = 0.0
    load_file: Any = None
    init_c: Any = None


@dataclass
class ObjectiveParams:
    """Imaging objective options (shared by both search families)."""

    name: str = "pib"
    target_max_brightness: int = 40
    r_bucket: int = 0
    target_size: float = 44.0
    target_aspect_ratio: float = 4.0 / 3.0
    target_center_smooth: int = 3
    target_shape: str | None = None
    shape_schedule: bool = False
    max_roi_energy_loss: float = 0.6
    w_uniformity: float = 2.0
    w_peak: float = 0.5
    w_displacement: float = 0.0
    log_uniformity: bool = False
    w_ema_decay: float = 0.9
    w_floor: float = 0.1
    w_temperature: float = 8.0


@dataclass
class SpgdParams:
    """SPGD (gradient) search options."""

    epochs: int = 2000
    delta: float = 0.2
    lr: float = 0.0
    optimizer_type: str = "adamod"
    shrink_iter: int = 0
    shrink_ratio: float = 0.9
    show: bool = False


@dataclass
class HeuristicParams:
    """Black-box heuristic search options."""

    algorithm: str = "ga"
    pop_size: int | None = None
    epochs: int = 2000
    show: bool = False


@dataclass
class SlmPibConfig:
    """Aggregates every parameter group for one slm-pib run."""

    run: RunParams
    camera: CameraParams
    slm: SlmParams
    objective: ObjectiveParams
    search: SpgdParams | HeuristicParams


# --- option decorators (applied to both subcommands) -----------------------


def _run_options(fn):
    fn = click.option("-d", "--dir", default="data", help="Data root directory.")(fn)
    fn = click.option(
        "--debug", is_flag=True, default=False, help="Enable debug mode."
    )(fn)
    return fn


def _camera_options(fn):
    fn = click.option("--cam-id", default=0, help="CCD camera device ID")(fn)
    fn = click.option(
        "--cam_type",
        type=click.Choice(["miicam", "daheng"]),
        default="daheng",
        help="CCD camera backend.",
    )(fn)
    fn = click.option(
        "--exposure_time_ms",
        type=float,
        default=80.0,
        help="CCD exposure time in ms (0 = auto-exposure).",
    )(fn)
    fn = click.option(
        "--auto-exposure",
        is_flag=True,
        default=False,
        help="Auto-find a safe fixed exposure before optimizing (one probe pass).",
    )(fn)
    fn = click.option(
        "--auto-target-peak",
        type=float,
        default=160.0,
        help="Target peak brightness for --auto-exposure.",
    )(fn)
    fn = click.option(
        "--auto-n-frames",
        type=int,
        default=5,
        help="Frames for the auto 0-order centre median.",
    )(fn)
    fn = click.option(
        "--cam_size", type=int, default=250, help="CCD window size in pixels."
    )(fn)
    fn = click.option(
        "-c",
        "--center",
        default=None,
        help="Center: 'auto' / 'mass' / 'max' / 'shape' or 'x,y'.",
    )(fn)
    return fn


def _slm_options(fn):
    fn = click.option(
        "--slm_number", type=int, default=1, help="Santec SLM device number (1-8)."
    )(fn)
    fn = click.option(
        "--slm_wavelength",
        type=int,
        default=1064,
        help="SLM operating wavelength (nm).",
    )(fn)
    fn = click.option(
        "-n", "--n_max", type=int, default=4, help="Max Zernike radial order."
    )(fn)
    fn = click.option(
        "--shift_x", type=int, default=0, help="SLM phase X shift (pixels)."
    )(fn)
    fn = click.option(
        "--shift_y", type=int, default=0, help="SLM phase Y shift (pixels)."
    )(fn)
    fn = click.option(
        "--zernike_radius",
        type=float,
        default=0.0,
        help="Zernike aperture radius (pixels); 0 = default.",
    )(fn)
    fn = click.option(
        "-f",
        "--load_file",
        type=str,
        default=None,
        help="Path to a prior Zernike coefficient file to load.",
    )(fn)
    fn = click.option(
        "--init_c",
        type=str,
        default=None,
        help="Initial Zernike coefficients (JSON or comma-separated).",
    )(fn)
    return fn


def _objective_options(fn):
    fn = click.option(
        "--objective",
        type=click.Choice(["pib", "radiu", "avg_radiu", "rmse", "shape", "roi_pib", "rms_pib"]),
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
        type=click.Choice(list(TARGET_SHAPE_CHOICES)),
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
    fn = click.option("--w_peak", type=float, default=0.5, help="Peak penalty weight.")(
        fn
    )
    fn = click.option(
        "--w_displacement", type=float, default=0.0, help="Displacement penalty weight."
    )(fn)
    fn = click.option(
        "--log_uniformity",
        is_flag=True,
        default=False,
        help="Use log1p(u) instead of u/(1+u) for the uniformity term.",
    )(fn)
    fn = click.option(
        "--w_ema_decay",
        type=float,
        default=0.9,
        help="EMA decay for the adaptive PIB/RMS weights of the 'rms_pib' objective.",
    )(fn)
    fn = click.option(
        "--w_floor",
        type=float,
        default=0.1,
        help="Minimum weight floor per term of the 'rms_pib' objective (0..0.5).",
    )(fn)
    fn = click.option(
        "--w_temperature",
        type=float,
        default=8.0,
        help="Softmax temperature for the 'rms_pib' weight update.",
    )(fn)
    return fn


# --- shared execution helpers ------------------------------------------------


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


def _optimizer_kwargs(
    cfg: SlmPibConfig,
    search: SpgdParams | HeuristicParams,
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
        "w_ema_decay": obj.w_ema_decay,
        "w_floor": obj.w_floor,
        "w_temperature": obj.w_temperature,
        "zernike_radius": zernike_radius,
        "shift_x": slm.shift_x,
        "shift_y": slm.shift_y,
        "record_phase": cfg.run.debug,
        "random_seed": None,
    }

    if isinstance(search, SpgdParams):
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


def _parse_objective_args(**opts) -> ObjectiveParams:
    """Build an :class:`ObjectiveParams` from the shared objective click options."""
    return ObjectiveParams(
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
        w_ema_decay=opts["w_ema_decay"],
        w_floor=opts["w_floor"],
        w_temperature=opts["w_temperature"],
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
@_run_options
@_camera_options
@_slm_options
@_objective_options
@click.option("-e", "--epochs", type=int, default=2000, help="Optimization iterations.")
@click.option(
    "--delta", type=float, default=0.2, help="SPGD perturbation amplitude (rad)."
)
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
@click.option(
    "--shrink_iter", type=int, default=0, help="Iterations before radius/step shrink."
)
@click.option(
    "--shrink_ratio", type=float, default=0.9, help="Radius/step shrink ratio."
)
@click.option("--show", is_flag=True, default=False, help="Open a live display window.")
@click.pass_context
def spgd(
    ctx: click.Context,
    dir: str,
    debug: bool,
    cam_id: int,
    cam_type: str,
    exposure_time_ms: float,
    auto_exposure: bool,
    auto_target_peak: float,
    auto_n_frames: int,
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
    camera = CameraParams(
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
    search = SpgdParams(
        epochs=obj_opts.pop("epochs", 2000),
        delta=obj_opts.pop("delta", 0.2),
        lr=obj_opts.pop("lr", 0.0),
        optimizer_type=obj_opts.pop("optimizer_type", "adamod"),
        shrink_iter=obj_opts.pop("shrink_iter", 0),
        shrink_ratio=obj_opts.pop("shrink_ratio", 0.9),
        show=obj_opts.pop("show", False),
    )
    _execute(
        run_cfg,
        camera,
        slm,
        objective,
        search,
        auto_exposure,
        auto_target_peak,
        auto_n_frames,
    )


@click.command(name="heuristic")
@_run_options
@_camera_options
@_slm_options
@_objective_options
@click.option(
    "--algorithm",
    type=click.Choice(heuristic_algorithm_choices()),
    default="ga",
    help="Black-box search algorithm.",
)
@click.option(
    "--pop_size", type=int, default=None, help="Population size (ga/pso/cem/de)."
)
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
    auto_exposure: bool,
    auto_target_peak: float,
    auto_n_frames: int,
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
    camera = CameraParams(
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
    search = HeuristicParams(
        algorithm=algorithm, pop_size=pop_size, epochs=epochs, show=show
    )
    _execute(
        run_cfg,
        camera,
        slm,
        objective,
        search,
        auto_exposure,
        auto_target_peak,
        auto_n_frames,
    )


run.add_command(spgd, name="spgd")
run.add_command(heuristic, name="heuristic")


def _resolve_auto_camera(
    camera: CameraParams,
    find_exposure: bool,
    auto_target_peak: float,
    auto_n_frames: int,
) -> None:
    """Probe the camera for a safe fixed exposure / 0-order centre before optimizing.

    Opens its own probe camera (closed again before the real run), so a
    failure never blocks the run: on any error the CLI-provided values are
    kept and ``'auto'`` centre falls back to the smart ``'shape'`` detection.
    Uses ``auto_find_exposure_and_center`` from the shared hardware utils so
    any runner can reuse the same probe logic.
    """
    from ao_shaping.utils.image.hardware_utils import (
        auto_find_exposure_and_center,
        open_camera,
    )

    try:
        probe = open_camera(camera.cam_type, camera.cam_id, camera.exposure_time_ms)
        try:
            exposure, center = auto_find_exposure_and_center(
                probe,
                find_exposure=find_exposure,
                find_center=(camera.center == "auto"),
                target_peak=auto_target_peak,
                n_frames=auto_n_frames,
            )
        finally:
            close_fn = getattr(probe, "close", None)
            if callable(close_fn):
                close_fn()
        if exposure is not None:
            camera.exposure_time_ms = float(exposure)
        if center is not None:
            camera.center = (int(center[0]), int(center[1]))
    except Exception as exc:  # probe is best-effort, never blocks the run
        logger.warning(
            "Auto exposure/centre probe failed ({}); using CLI camera values", exc
        )
        if camera.center == "auto":
            camera.center = "shape"


def _execute(
    run_cfg: RunParams,
    camera: CameraParams,
    slm: SlmParams,
    objective: ObjectiveParams,
    search: SpgdParams | HeuristicParams,
    auto_exposure: bool = False,
    auto_target_peak: float = 160.0,
    auto_n_frames: int = 5,
) -> None:
    """Shared execution path for both search families."""
    setup_coredumpy()
    if auto_exposure or camera.center == "auto":
        _resolve_auto_camera(camera, auto_exposure, auto_target_peak, auto_n_frames)
    if camera.center is not None and not isinstance(camera.center, str):
        camera.center = (int(camera.center[0]), int(camera.center[1]))
    elif isinstance(camera.center, str) and "," in camera.center:
        x_str, y_str = (p.strip() for p in camera.center.split(","))
        camera.center = (int(x_str), int(y_str))

    kwargs = _optimizer_kwargs(
        SlmPibConfig(run_cfg, camera, slm, objective, search), search
    )
    init_c = _load_initial_coeffs(slm.load_file, slm.init_c)
    if init_c is not None:
        kwargs["init_c"] = init_c

    res = optimize_slm_zernike_pib(**kwargs)
    if run_cfg.debug:
        _save_debug_artifacts(res, objective, slm, search, run_cfg.dir)

    eff_key = _effective_objective_key(objective.name, objective.target_shape)
    final = res.history[-1]
    logger.info("SLM PIB done. Final {}", {k: final[k] for k in ("J", eff_key)})


if __name__ == "__main__":
    run()
