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
    CameraParams,
    HeuristicParams,
    ObjectiveParamsSquare,
    RunParams,
    SlmParams,
    SpgdParams,
    config_payload,
    parse_center,
    run_options,
    camera_options,
    slm_options,
)
from ao_shaping.utils.io.file import Recorder, save_recorder_debug_artifacts
from ao_shaping.utils.io.cli_helpers import setup_coredumpy

# --- debug artifact fields -------------------------------------------------
# The square-shaping recorder stores the metric keys below. Image-like entries
# (leading-underscore ``_img``/``_diff``) are the CCD frames and are rendered by
# the shared artifact writer; the scalar keys drive the JSON sidecar + summary.

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
    res: Recorder,
    objective: "ObjectiveParamsSquare",
    root_dir: str,
) -> Any:
    """Write PNG/pkl/json debug artifacts for the recorded square-shaping search.

    Delegates to the shared :func:`ao_shaping.utils.io.file.save_recorder_debug_artifacts`
    with the square-shaping key set. The JSON sidecar carries the objective
    config so the run is reproducible from the artifact directory alone.
    """
    return save_recorder_debug_artifacts(
        res,
        root_dir=root_dir,
        subdir_prefix=f"slm_gsnet_{objective.name}",
        scalar_keys=_DEBUG_SCALAR_KEYS,
        img_keys=_IMG_KEYS,
        d1_keys=_1D_KEYS,
        json_payload=_config_payload(objective),
        title=f"slm-gsnet {objective.name} search",
    )


def _config_payload(obj: Any) -> dict[str, Any]:
    """Serialize a parameter dataclass for the JSON debug sidecar."""
    return config_payload(obj)


# --- parameter dataclasses -------------------------------------------------


@dataclass
class SlmGsnetConfig:
    """Aggregates every parameter group for one slm-gsnet run."""

    run: RunParams
    camera: CameraParams
    slm: SlmParams
    objective: ObjectiveParamsSquare
    search: SpgdParams | HeuristicParams


# --- option decorators (applied to both subcommands) -----------------------


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


def _optimizer_kwargs(cfg: SlmGsnetConfig) -> dict[str, Any]:
    """Flatten the aggregated config into the optimizer's keyword arguments."""
    cam = cfg.camera
    slm = cfg.slm
    obj = cfg.objective
    search = cfg.search

    center = parse_center(cam.center)
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


def _parse_objective_args(**opts) -> ObjectiveParamsSquare:
    """Build an :class:`ObjectiveParamsSquare` from the shared objective click options."""
    return ObjectiveParamsSquare(
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
@run_options
@camera_options
@slm_options
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
@run_options
@camera_options
@slm_options
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
