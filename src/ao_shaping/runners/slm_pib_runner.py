"""Run slm-pib — PIB shaping with a Santec SLM driven by Zernike coefficients.

Usage:
    python src/ao_shaping/main.py slm-pib [OPTIONS]
"""

from __future__ import annotations

from typing import Any

import click
import numpy as np
from loguru import logger

from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    SlmZernikePibConfig,
    optimize_slm_zernike_pib,
)
from ao_shaping.runners.runner_common import (
    CameraParamsPib,
    HeuristicParams,
    ObjectiveParamsPib,
    RunParams,
    SlmParams,
    SlmParamsPib,
    SpgdParamsPib,
    config_payload,
    with_params,
)
from ao_shaping.utils.io.cli_helpers import setup_coredumpy
from ao_shaping.utils.io.file import (
    Recorder,
    save_recorder_debug_artifacts,
)

# Backward-compatible aliases — tests and external callers may still import
# the old names from this module.
ObjectiveParams = ObjectiveParamsPib
CameraParams = CameraParamsPib
SpgdParams = SpgdParamsPib

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
    "w_ee",
    "pib_term",
    "rms_term",
    "ee_term",
    # Cross-objective metric panel (recorded every epoch by the optimizer, so
    # runs driven by different objectives can be compared on identical columns).
    "m_shape",
    "m_energy",
    "m_rmse",
    "m_roi_pib",
    "m_pib",
    "m_pib7",
    "m_rms_pib",
    "m_rms_t",
    "m_ee",
    "m_brt",
    # Objective columns — merged from _DEBUG_OBJECTIVE_KEYS for a single pass.
    "pib",
    "radiu",
    "avg_radiu",
    "rmse",
    "shape",
    "roi_pib",
    "rms_pib",
)

# Objective columns passed separately to the shared debug-artifact helper
# (see ``save_recorder_debug_artifacts``); kept in sync with the scalar set.
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
    return config_payload(obj)


def _save_debug_artifacts(
    res: Recorder,
    objective: "ObjectiveParamsPib",
    config: "SlmParamsPib",
    obj_or_heur: "SpgdParamsPib | HeuristicParams",
    root_dir: str,
) -> Any:
    """Write PNG/pkl/json/h5 debug artifacts for the recorded search.

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


def _build_slm_pib_config(
    run: RunParams,
    camera: CameraParamsPib,
    slm: SlmParamsPib,
    search: SpgdParamsPib | HeuristicParams,
    init_c: np.ndarray | None = None,
) -> SlmZernikePibConfig:
    """Build the optimizer config directly from the fused parameter objects.

    ``CameraParamsPib`` fuses the camera + objective fields and
    ``SlmParamsPib`` carries the SLM + Zernike-coefficient fields, so both are
    passed to :class:`SlmZernikePibConfig` as nested objects — no flat kwargs
    flattening. The run-level switches ``center`` and ``epochs`` are set from
    the camera centre and the search epochs.
    """
    if init_c is not None:
        # The config is a plain dataclass: the parsed array is carried on the
        # ``SlmParamsPib`` object itself (the dataclass field is normally a
        # raw ``str``).
        slm.init_c = init_c

    cfg: dict[str, Any] = {
        "center": camera.center,
        "epochs": search.epochs,
        "camera": camera,
        "slm": slm,
        "record_phase": run.debug,
        "random_seed": run.seed,
    }

    if isinstance(search, SpgdParamsPib):
        cfg.update(
            algorithm="spgd",
            pop_size=None,
            delta=search.delta,
            lr=search.lr,
            optimizer_type=search.optimizer_type,
            shrink_iter=search.shrink_iter,
            shrink_ratio=search.shrink_ratio,
            show=search.show,
        )
    else:  # HeuristicParams
        cfg.update(
            algorithm=search.algorithm,
            pop_size=search.pop_size,
            show=search.show,
        )

    return SlmZernikePibConfig(**cfg)


# --- click group + subcommands ---------------------------------------------


@click.group(invoke_without_command=True)
@click.pass_context
@with_params(RunParams, kw_name="run")
def run(ctx: click.Context, run: RunParams) -> None:
    """PIB shaping with a Santec SLM driven by Zernike coefficients.

    Without a subcommand, the SPGD (gradient) search runs.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(spgd, **ctx.params)


@click.command(name="spgd")
@click.pass_context
@with_params(RunParams, kw_name="run")
@with_params(CameraParamsPib, kw_name="camera")
@with_params(SlmParamsPib, kw_name="slm")
@with_params(SpgdParamsPib, kw_name="search")
def spgd(
    ctx: click.Context,
    run: RunParams,
    camera: CameraParamsPib,
    slm: SlmParamsPib,
    search: SpgdParamsPib,
) -> None:
    """Run the SPGD (Stochastic Parallel Gradient Descent) search."""
    _execute(run, camera, slm, search)


@click.command(name="heuristic")
@click.pass_context
@with_params(RunParams, kw_name="run")
@with_params(CameraParamsPib, kw_name="camera")
@with_params(SlmParamsPib, kw_name="slm")
@with_params(HeuristicParams, kw_name="search")
def heuristic(
    ctx: click.Context,
    run: RunParams,
    camera: CameraParamsPib,
    slm: SlmParamsPib,
    search: HeuristicParams,
) -> None:
    """Run a black-box heuristic search (ga/pso/sa/hc/rs/cem/de)."""
    _execute(run, camera, slm, search)


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
    camera: CameraParamsPib,
    slm: SlmParamsPib,
    search: SpgdParamsPib | HeuristicParams,
) -> None:
    """Shared execution path for both search families."""
    setup_coredumpy()
    if camera.auto_exposure or camera.center == "auto":
        _resolve_auto_camera(
            camera,
            camera.auto_exposure,
            camera.auto_target_peak,
            camera.auto_n_frames,
        )
    if camera.center is not None and not isinstance(camera.center, str):
        camera.center = (int(camera.center[0]), int(camera.center[1]))
    elif isinstance(camera.center, str) and "," in camera.center:
        x_str, y_str = (p.strip() for p in camera.center.split(","))
        camera.center = (int(x_str), int(y_str))

    init_c = _load_initial_coeffs(slm.load_file, slm.init_c)
    config = _build_slm_pib_config(run_cfg, camera, slm, search, init_c)
    res = optimize_slm_zernike_pib(config)
    if run_cfg.debug:
        _save_debug_artifacts(res, camera, slm, search, run_cfg.dir)

    eff_key = _effective_objective_key(camera.name, camera.target_shape)
    # Report the best record (by the effective objective) instead of the last
    # one: the final evaluation may be a guard-penalised row (energy guard
    # returns J-1e3 and the history[-1] record would show -999 even though the
    # search found a good optimum).
    final: dict[str, Any]
    if res.history and eff_key in res.history[0]:
        final = max(
            res.history,
            key=lambda row: row.get(eff_key, -float("inf")),
        )
    else:
        final = res.history[-1]
    logger.info("SLM PIB done. Final {}", {k: final[k] for k in ("J", eff_key)})


if __name__ == "__main__":
    run()
