"""Run slm-gsnet — square far-field shaping via FREEFORM per-pixel SLM phase.

This is the Gerchberg-Saxton / differentiable "shaping" family runner: instead of
driving the SLM with a low-order Zernike basis (which physically cannot synthesise
a true square far-field — low-order Zernike modes are smooth and circularly
symmetric), it optimises the SLM's *free* per-pixel phase (a ``phase_grid²``
freeform DOF) with SPGD (gradient) or a black-box heuristic search. The target is
a uniform, high-energy square, scored by the combined quality score
(uniformity CV + encircled energy + aspect) — the same ``optimize_slm_square``
objective as the ``spgd-square`` command, but freeform-first.

Usage:
    python src/ao_shaping/main.py slm-gsnet                    # SPGD (default, no options)
    python src/ao_shaping/main.py slm-gsnet spgd [OPTIONS]     # SPGD gradient search
    python src/ao_shaping/main.py slm-gsnet heuristic [OPTIONS]  # black-box heuristic

Offline dry-run (no hardware) — options live on the subcommand, not the group:
    python src/ao_shaping/main.py slm-gsnet spgd --cam_type sim --epochs 50
    python src/ao_shaping/main.py slm-gsnet heuristic --cam_type sim --epochs 50 --algorithm ga
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
from ao_shaping.optimizer.wfless.slm_square_shaping import optimize_slm_square
from ao_shaping.runners.runner_common import (
    build_debug_save_paths,
    save_optimization_debug_artifacts,
)
from ao_shaping.utils.io.cli_helpers import get_debug_mode, parse_tuple, setup_coredumpy

# --- debug artifact fields -------------------------------------------------
# The square-shaping recorder stores the metric keys below. Image-like entries
# (leading-underscore ``_img``/``_diff``) are the CCD frames and are rendered by
# the shared artifact writer; the scalar keys drive the JSON sidecar + summary.

_DEBUG_EXCLUDE = {"_img", "_grad"}

_DEBUG_SCALAR_KEYS = (
    "J",
    "quality",
    "cv",
    "ee",
    "ar",
    "side",
    "lr",
    "delta",
    "_diff",
    "exp_t",
    "max_brt",
    "mean_b",
    "target_mean_b",
    "best_quality",
    "_epoch",
)

_IMG_KEYS = ("_img",)
_1D_KEYS = ("_c", "_grad")


def _save_debug_artifacts(
    res: "RecorderLike",
    objective: "ObjectiveParams",
    root_dir: str,
) -> Any:
    """Write PNG/pkl/json debug artifacts for the recorded square-shaping search.

    Mirrors :func:`slm_pib_runner._save_debug_artifacts` but for the square
    objective key set. The JSON sidecar carries the objective + search config so
    the run is reproducible from the artifact directory alone.
    """
    save_dir, saved_file_name = build_debug_save_paths(
        os.path.join(root_dir, "debug"),
        f"slm_gsnet_{objective.name}_{datetime.now():%Y%m%d_%H%M%S}",
    )
    png_path = saved_file_name.with_suffix(".png")
    pkl_path = saved_file_name.with_suffix(".pkl")
    json_path = saved_file_name.with_suffix(".json")

    data: dict[str, Any] = {}
    for rec in res.history:
        item: dict[str, Any] = {}
        for k in _DEBUG_SCALAR_KEYS:
            if k in rec:
                item[k] = float(rec[k])
        for k in _IMG_KEYS:
            if k in rec:
                item[k] = np.asarray(rec[k])
        for k in _1D_KEYS:
            if k in rec:
                item[k] = np.asarray(rec[k], dtype=float)
        data[int(rec["_epoch"])] = item

    save_optimization_debug_artifacts(
        data=data,
        png_path=png_path,
        pkl_path=pkl_path,
        json_path=json_path,
        title=f"slm-gsnet {objective.name} search",
        json_payload=_config_payload(objective),
    )
    return png_path


def _config_payload(obj: Any) -> dict[str, Any]:
    """Serialize a parameter dataclass for the JSON debug sidecar."""
    payload: dict[str, Any] = {}
    for key, value in obj.__dict__.items():
        if value is None:
            continue
        default = getattr(type(obj), key, None)
        if value == default:
            continue
        payload[key] = value
    return payload


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
    cam_size: int = 300
    center: Any = None


@dataclass
class SlmParams:
    """Santec SLM options."""

    slm_number: int = 1
    slm_wavelength: int = 1064
    n_max: int = 4
    zernike_radius: float = 0.0


@dataclass
class ObjectiveParams:
    """Square-shaping objective options."""

    name: str = "square"
    target_side: int = 0
    target_mean_brightness: float = 0.0
    side_factor: float = 1.5
    target_max_brightness: int = 200
    w_uniformity: float = 0.4
    w_efficiency: float = 0.6
    w_aspect: float = 0.0


@dataclass
class SpgdParams:
    """SPGD (gradient) search options."""

    epochs: int = 2000
    delta: float = 0.1
    lr: float = 0.0
    optimizer_type: str = "adamod"
    show: bool = False


@dataclass
class HeuristicParams:
    """Black-box heuristic search options."""

    algorithm: str = "ga"
    pop_size: int | None = None
    epochs: int = 2000
    show: bool = False


@dataclass
class SlmGsnetConfig:
    """Aggregates every parameter group for one slm-gsnet run."""

    run: RunParams
    camera: CameraParams
    slm: SlmParams
    objective: ObjectiveParams
    search: SpgdParams | HeuristicParams


# --- option decorators (applied to both subcommands) -----------------------


def _run_options(fn):
    fn = click.option(
        "-d", "--dir", default="data", help="Data root directory."
    )(fn)
    fn = click.option(
        "--debug", is_flag=True, default=False, help="Enable debug mode."
    )(fn)
    return fn


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
        "--cam_size", type=int, default=300, help="CCD window size in pixels."
    )(fn)
    fn = click.option(
        "-c",
        "--center",
        default=None,
        help="Spot center: 'shape' / 'max' / 'mass' / 'centroid_thresh' or 'x,y'.",
    )(fn)
    return fn


def _slm_options(fn):
    fn = click.option(
        "--slm_number", type=int, default=1, help="Santec SLM device number (1-8)."
    )(fn)
    fn = click.option(
        "--slm_wavelength", type=int, default=1064, help="SLM operating wavelength (nm)."
    )(fn)
    fn = click.option(
        "-n", "--n_max", type=int, default=4, help="Max Zernike radial order."
    )(fn)
    fn = click.option(
        "--zernike_radius",
        type=float,
        default=0.0,
        help="Zernike aperture radius (pixels); 0 = default (SLM short side / 2).",
    )(fn)
    return fn


def _objective_options(fn):
    fn = click.option(
        "--target-side",
        type=int,
        default=0,
        help="Target square side (pixels); 0 = auto from spot size. Mutually "
        "exclusive with --target-mean-brightness.",
    )(fn)
    fn = click.option(
        "--target-mean-brightness",
        type=float,
        default=0.0,
        help="Target square mean brightness (gray). >0 auto-derives side by "
        "energy conservation. Mutually exclusive with --target-side.",
    )(fn)
    fn = click.option(
        "--side-factor", type=float, default=1.5, help="Auto side-length factor."
    )(fn)
    fn = click.option(
        "--target-max-brightness",
        type=int,
        default=200,
        help="Target max brightness for auto-exposure.",
    )(fn)
    fn = click.option(
        "--w_uniformity", type=float, default=0.4, help="Uniformity (CV) weight."
    )(fn)
    fn = click.option(
        "--w_efficiency", type=float, default=0.6, help="Encircled-energy weight."
    )(fn)
    fn = click.option(
        "--w_aspect", type=float, default=0.0, help="Aspect-ratio weight."
    )(fn)
    return fn


# --- shared execution helpers ------------------------------------------------


def _parse_center(raw: Any) -> tuple[int, int] | str | None:
    """Normalize the ``--center`` option to the optimizer's accepted form."""
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = raw.split(",")
        if len(parts) == 2:
            try:
                return (int(parts[0].strip()), int(parts[1].strip()))
            except ValueError:
                return raw  # named mode: shape/max/mass/centroid_thresh
        return raw
    return raw


def _optimizer_kwargs(cfg: SlmGsnetConfig) -> dict[str, Any]:
    """Flatten the aggregated config into the optimizer's keyword arguments."""
    cam = cfg.camera
    slm = cfg.slm
    obj = cfg.objective
    search = cfg.search

    center = _parse_center(cam.center)
    zernike_radius: float | None = slm.zernike_radius if slm.zernike_radius > 0 else None

    kwargs: dict[str, Any] = {
        "center": center,
        "epochs": search.epochs,
        "n_max": slm.n_max,
        "target_side": obj.target_side,
        "target_mean_brightness": obj.target_mean_brightness,
        "side_factor": obj.side_factor,
        "exposure_time_ms": cam.exposure_time_ms,
        "cam_id": cam.cam_id,
        "show": search.show,
        "cam_size": cam.cam_size,
        "target_max_brightness": obj.target_max_brightness,
        "slm_number": slm.slm_number,
        "slm_wavelength": slm.slm_wavelength,
        "w_uniformity": obj.w_uniformity,
        "w_efficiency": obj.w_efficiency,
        "w_aspect": obj.w_aspect,
        # Freeform per-pixel phase is the ONLY DOF that can synthesise a square
        # far-field (low-order Zernike is smooth and cannot). This runner is the
        # freeform-first entry point of the square-shaping family.
        "basis": "freeform",
        "phase_grid": 24,
        "zernike_radius": zernike_radius,
        "random_seed": None,
    }

    if isinstance(search, SpgdParams):
        kwargs.update(
            {
                "lr": search.lr,
                "delta": search.delta,
                "optimizer_type": search.optimizer_type,
                "algorithm": "spgd",
                "pop_size": None,
            }
        )
    else:  # HeuristicParams — black-box search has no learning rate / perturbation
        kwargs["lr"] = 0.0
        kwargs.update(
            {
                "algorithm": search.algorithm,
                "pop_size": search.pop_size,
            }
        )

    return kwargs


def _parse_objective_args(**opts) -> ObjectiveParams:
    """Build an :class:`ObjectiveParams` from the shared objective click options."""
    return ObjectiveParams(
        name="square",
        target_side=opts["target_side"],
        target_mean_brightness=opts["target_mean_brightness"],
        side_factor=opts["side_factor"],
        target_max_brightness=opts["target_max_brightness"],
        w_uniformity=opts["w_uniformity"],
        w_efficiency=opts["w_efficiency"],
        w_aspect=opts["w_aspect"],
    )


def _maybe_sim_patch(cam_type: str) -> None:
    """Wire the pure-numpy 2f-Fourier sim into the square-shaping optimizer.

    ``optimize_slm_square`` hard-codes ``MIICamera(...)`` and ``Santec(...)`` in
    its ``with`` block. For an offline dry-run we monkeypatch both names in the
    optimizer module to the sim stand-ins so no hardware is touched (and no DVI
    hang). No-op unless ``cam_type == "sim"``.
    """
    if cam_type != "sim":
        return
    from ao_shaping.drivers.sim.slm_pib_sim import (
        register_sim_camera,
        reset_system,
        SimPibCCD,
        SimSLMPib,
    )

    import ao_shaping.optimizer.wfless.slm_square_shaping as opt

    register_sim_camera()
    reset_system(seed=42)
    opt.MIICamera = SimPibCCD
    opt.Santec = SimSLMPib


# --- click group + subcommands ---------------------------------------------


@click.group(invoke_without_command=True)
@click.pass_context
def run(ctx: click.Context) -> None:
    """Square far-field shaping via FREEFORM per-pixel SLM phase.

    The phase basis is always freeform (per-pixel) — the only DOF that can
    synthesise a true square far-field (low-order Zernike cannot). Without a
    subcommand, the SPGD (gradient) search runs.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(spgd, **ctx.params)


@click.command(name="spgd")
@_run_options
@_camera_options
@_slm_options
@_objective_options
@click.option("-e", "--epochs", type=int, default=2000, help="Optimization iterations.")
@click.option("--delta", type=float, default=0.1, help="SPGD perturbation amplitude (rad).")
@click.option("--lr", type=float, default=0.0, help="SPGD learning rate (0 = auto).")
@click.option(
    "--optimizer_type",
    type=click.Choice(["adam", "adamw", "adamod", "sgd", "muno", "munow"], case_sensitive=False),
    default="adamod",
    show_default=True,
    help="SPGD gradient optimizer.",
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
    cam_size: int,
    center,
    slm_number: int,
    slm_wavelength: int,
    n_max: int,
    zernike_radius: float,
    epochs: int,
    delta: float,
    lr: float,
    optimizer_type: str,
    show: bool,
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
        zernike_radius=zernike_radius,
    )
    objective = _parse_objective_args(**obj_opts)
    search = SpgdParams(
        epochs=epochs,
        delta=delta,
        lr=lr,
        optimizer_type=optimizer_type,
        show=show,
    )
    _execute(SlmGsnetConfig(run_cfg, camera, slm, objective, search))


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
    zernike_radius: float,
    algorithm: str,
    pop_size: int | None,
    epochs: int,
    show: bool,
    **obj_opts,
) -> None:
    """Run a black-box heuristic freeform search (ga/pso/sa/hc/rs/cem/de)."""
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
        zernike_radius=zernike_radius,
    )
    objective = _parse_objective_args(**obj_opts)
    search = HeuristicParams(algorithm=algorithm, pop_size=pop_size, epochs=epochs, show=show)
    _execute(SlmGsnetConfig(run_cfg, camera, slm, objective, search))


run.add_command(spgd, name="spgd")
run.add_command(heuristic, name="heuristic")


def _execute(cfg: SlmGsnetConfig) -> None:
    """Shared execution path for both search families."""
    setup_coredumpy()
    _maybe_sim_patch(cfg.camera.cam_type)

    kwargs = _optimizer_kwargs(cfg)
    res = optimize_slm_square(**kwargs)

    if cfg.run.debug:
        _save_debug_artifacts(res, cfg.objective, cfg.run.dir)

    final = res.history[-1]
    logger.info(
        "SLM GSNET done. Final quality={:.4f} (CV={:.4f}, EE={:.4f}, AR={:.4f})",
        final.get("quality", float("nan")),
        final.get("cv", float("nan")),
        final.get("ee", float("nan")),
        final.get("ar", float("nan")),
    )


if __name__ == "__main__":
    run()
