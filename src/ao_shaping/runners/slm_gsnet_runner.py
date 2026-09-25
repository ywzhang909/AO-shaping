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

from dataclasses import dataclass
from typing import Any

import click
import numpy as np
from loguru import logger

from ao_shaping.optimizer.wfless.slm_square_shaping import (
    SlmSquareConfig,
    optimize_slm_square,
)
from ao_shaping.runners.runner_common import (
    CameraParams,
    HeuristicParams,
    ObjectiveParamsSquare,
    RunParams,
    SlmParams,
    SpgdParams,
    config_payload,
    parse_center,
    with_params,
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


# --- shared execution helpers ------------------------------------------------


def _build_square_config(cfg: SlmGsnetConfig) -> SlmSquareConfig:
    """Build the optimizer config object directly (no flat kwargs flattening).

    Freeform per-pixel phase is the ONLY DOF that can synthesise a square
    far-field (low-order Zernike is smooth and cannot). This runner is the
    freeform-first entry point of the square-shaping family.
    ``center`` / ``epochs`` are not part of the config — they stay positional
    arguments of :func:`optimize_slm_square`.
    """
    cam = cfg.camera
    slm = cfg.slm
    obj = cfg.objective
    search = cfg.search

    zernike_radius: float | int | None = (
        slm.zernike_radius if slm.zernike_radius > 0 else None
    )

    common: dict[str, Any] = dict(
        n_max=slm.n_max,
        target_side=obj.target_side,
        target_mean_brightness=obj.target_mean_brightness,
        side_factor=obj.side_factor,
        exposure_time_ms=cam.exposure_time_ms,
        cam_id=cam.cam_id,
        show=search.show,
        cam_size=cam.cam_size,
        target_max_brightness=obj.target_max_brightness,
        slm_number=slm.slm_number,
        slm_wavelength=slm.slm_wavelength,
        w_uniformity=obj.w_uniformity,
        w_efficiency=obj.w_efficiency,
        w_aspect=obj.w_aspect,
        basis="freeform",
        phase_grid=24,
        zernike_radius=zernike_radius,
        random_seed=cfg.run.seed,
    )

    if isinstance(search, SpgdParams):
        common.update(
            algorithm="spgd",
            pop_size=None,
            lr=search.lr,
            delta=search.delta,
            optimizer_type=search.optimizer_type,
        )
    else:  # HeuristicParams — black-box search has no learning rate / perturbation
        common.update(
            algorithm=search.algorithm,
            pop_size=search.pop_size,
            lr=0.0,
        )

    return SlmSquareConfig(**common)


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
@with_params(RunParams, kw_name="run")
def run(ctx: click.Context, run: RunParams) -> None:
    """Square far-field shaping via FREEFORM per-pixel SLM phase.

    The phase basis is always freeform (per-pixel) — the only DOF that can
    synthesise a true square far-field (low-order Zernike cannot). Without a
    subcommand, the SPGD (gradient) search runs.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(spgd, **ctx.params)


@click.command(name="spgd")
@click.pass_context
@with_params(RunParams, kw_name="run")
@with_params(CameraParams, kw_name="camera")
@with_params(SlmParams, kw_name="slm")
@with_params(ObjectiveParamsSquare, kw_name="objective")
@with_params(SpgdParams, kw_name="search")
def spgd(
    ctx: click.Context,
    run: RunParams,
    camera: CameraParams,
    slm: SlmParams,
    objective: ObjectiveParamsSquare,
    search: SpgdParams,
) -> None:
    """Run the SPGD (Stochastic Parallel Gradient Descent) search."""
    _execute(SlmGsnetConfig(run, camera, slm, objective, search))


@click.command(name="heuristic")
@click.pass_context
@with_params(RunParams, kw_name="run")
@with_params(CameraParams, kw_name="camera")
@with_params(SlmParams, kw_name="slm")
@with_params(ObjectiveParamsSquare, kw_name="objective")
@with_params(HeuristicParams, kw_name="search")
def heuristic(
    ctx: click.Context,
    run: RunParams,
    camera: CameraParams,
    slm: SlmParams,
    objective: ObjectiveParamsSquare,
    search: HeuristicParams,
) -> None:
    """Run a black-box heuristic freeform search (ga/pso/sa/hc/rs/cem/de)."""
    _execute(SlmGsnetConfig(run, camera, slm, objective, search))


run.add_command(spgd, name="spgd")
run.add_command(heuristic, name="heuristic")


def _execute(cfg: SlmGsnetConfig) -> None:
    """Shared execution path for both search families."""
    setup_coredumpy()
    _maybe_sim_patch(cfg.camera.cam_type)

    config = _build_square_config(cfg)
    res = optimize_slm_square(
        center=parse_center(cfg.camera.center),
        epochs=cfg.search.epochs,
        config=config,
    )

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
