"""Run ``slm-pib`` / ``spgd-square`` — SLM far-field shaping, one runner.

Both CLI families drive the *same* device pair (Santec SLM + CCD, camera
feedback, Zernike coefficients) and only differ in what they optimise:

* **``spgd`` / ``heuristic``** (``slm-pib``) -> ``optimize_slm_zernike_pib``
  Power-in-Bucket / RMS / Pearson ... against an arbitrary target shape
  (``ObjectiveTarget``). Config container: ``SlmZernikePibConfig``.
* **``square``** (``spgd-square``) -> ``optimize_slm_square``
  Uniform **square** far field, scored by the combined quality score
  (uniformity CV + encircled energy + aspect). Config container:
  ``SlmSquareConfig``.

They were separate modules (``pib_runner.py`` + ``square_runner.py``) whose
option groups, config plumbing and sim wiring had already started to drift —
``--cam-type sim`` replaced the camera on one command and opened a real Santec
on the other. One module, one set of shared helpers, one sim patch.

Usage:
    python src/ao_shaping/main.py slm-pib spgd [OPTIONS]         # PIB, SPGD
    python src/ao_shaping/main.py slm-pib heuristic [OPTIONS]    # PIB, black-box
    python src/ao_shaping/main.py spgd-square [OPTIONS]          # square shaping
    python src/ao_shaping/main.py slm-pib square [OPTIONS]       # same command

Without a subcommand ``slm-pib`` runs the SPGD (gradient) search.

Offline dry-run (no hardware) — ``--cam-type sim`` covers **both** devices, so
neither command can reach for hardware on that path:

    python src/ao_shaping/main.py slm-pib spgd --cam-type sim -e 3
    python src/ao_shaping/main.py spgd-square --cam-type sim -e 3

``spgd-square`` additionally exposes ``--slm-type`` for symmetry with the other
SLM runners; it must accompany ``--cam-type sim`` (a simulated SLM only exists
once the twin is installed) and is otherwise redundant.

Module-local entry points (``python -m``) are equally supported:

    python -m ao_shaping.runners.slm.slm_shaping_runner spgd --help
    python -m ao_shaping.runners.slm.slm_shaping_runner square --help

Results (default ``data/``):

* ``--debug`` -> ``data/debug/slm_square_<timestamp>/`` PNG + PKL + JSON
  sidecar. This is the corpus ``ml.hwdataset`` indexes for phase->image
  training, which the square family previously never fed.
* ``data/slm_square/<date>/`` -> best-iteration CSV, best Zernike
  coefficients, and the best far-field PNG (``--save-best-image``).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
from loguru import logger

from ao_shaping.optimizer.wfless.slm_square_shaping import (
    ZERNIKE_ACTIVE_MODES,
    SlmSquareConfig,
    _zernike_indices,
    optimize_slm_square,
)
from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    SLM_RESOLUTION,
    SlmZernikePibConfig,
    _display,
    optimize_slm_zernike_pib,
)
from ao_shaping.runners.runner_common import (
    CameraParamsPib,
    HeuristicParams,
    ObjectiveParamsPib,
    RunParams,
    SlmParams,
    SlmParamsPib,
    SlmSquareParams,
    SpgdParamsPib,
    ZernikeSlmParams,
    config_payload,
    parse_center,
    patch_sim_pib_shaping,
    patch_sim_square_shaping,
    resolve_spgd_delta,
)
from ao_shaping.utils.cli.params import with_params
from ao_shaping.utils.image.targets import ObjectiveSpec
from ao_shaping.utils.io.cli_helpers import get_debug_mode, setup_coredumpy
from ao_shaping.utils.io.file import (
    _DATA_MODE_OBJECTIVE_KEYS,
    Recorder,
    save_recorder_debug_artifacts,
)
from ao_shaping.display.frames import save_best_image

# Backward-compatible aliases / re-exports — tests and external callers
# import the parameter dataclasses from the runner module they configure, so
# they stay importable from here.
ObjectiveParams = ObjectiveParamsPib
CameraParams = CameraParamsPib
SpgdParams = SpgdParamsPib
__all__ = [
    "CameraParams",
    "HeuristicParams",
    "ObjectiveParams",
    "RunParams",
    "SlmParams",
    "SlmParamsPib",
    "SlmSquareParams",
    "SpgdParams",
    "SpgdParamsPib",
    "ZernikeSlmParams",
    "heuristic",
    "run",
    "spgd",
    "square",
]

# --- debug artifact fields -------------------------------------------------

_DEBUG_IMG_KEYS = ("_img", "_full_frame_img")
_DEBUG_1D_KEYS = ("_c",)
# ``_phase`` is the display-ready grayscale actually sent to the SLM on each
# epoch (present when ``record_phase`` / ``--debug``); persisted so the report
# can show the sent phase alongside the measured spot (``_img``).
_DEBUG_2D_KEYS = ("_grad", "_phase")
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
    # Objective columns — canonical set from _DATA_MODE_OBJECTIVE_KEYS.
    *_DATA_MODE_OBJECTIVE_KEYS,
)

# Objective columns passed separately to the shared debug-artifact helper
# (see ``save_recorder_debug_artifacts``); re-exported from the canonical
# definition in ``utils/io/file.py`` so there is a single source of truth.
_DEBUG_OBJECTIVE_KEYS = _DATA_MODE_OBJECTIVE_KEYS

# The square-shaping Recorder stores a different column set (the quality score
# and its CV / EE / AR components, not the PIB panel above). Both families go
# through the same artifact writer, so the keys are declared per family.
_SQUARE_DEBUG_SCALAR_KEYS = (
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
_SQUARE_DEBUG_IMG_KEYS = ("_img",)
_SQUARE_DEBUG_1D_KEYS = ("_c", "_grad")


def _effective_objective_key(name: str, target_shape: Any) -> str:
    """Return the dict key the optimizer actually records for the objective value.

    Delegates to :meth:`ObjectiveSpec.resolve` - the single authority for the
    objective/target-shape pairing - instead of repeating the rule here. The old
    inline copy was already stale: it hard-coded ``("roi_pib", "rms_pib", "rmse")``
    as the shape-preserving set, so any newer objective (``pearson``) was
    silently reported as ``"shape"``. That made the runner pick the ``m_shape``
    panel column for the best-row of a Pearson run: wrong numbers, no error.
    """
    return ObjectiveSpec.resolve(name, target_shape).name


def _config_payload(obj: Any) -> dict[str, Any]:
    """Serialize a search-config dataclass for the JSON debug sidecar.

    Drops unset (None) fields and fields left at their dataclass default;
    the identity field (``algorithm`` / ``name``) is always kept so the
    search family survives the round trip.
    """
    return config_payload(obj)


def _save_debug_artifacts(
    res: Recorder,
    camera: "CameraParamsPib",
    config: "SlmParamsPib",
    obj_or_heur: "ObjectiveParamsPib | SpgdParamsPib | HeuristicParams",
    root_dir: str,
) -> Any:
    """Write PNG/pkl/json/h5 debug artifacts for the recorded search.

    Delegates to :func:`ao_shaping.utils.io.file.save_recorder_debug_artifacts`
    with the slm-pib key set.

    The JSON sidecar records the **objective, camera, SLM and search identity**
    so an artifact is self-describing. The runner passes a
    ``CameraParamsPib`` (which inherits both ``CameraParams`` and
    ``ObjectiveParamsPib``) as the camera argument, and the search config
    (``SpgdParamsPib`` / ``HeuristicParams``) as the third, so each block is
    read from the object that actually owns it rather than by duck-typing one
    argument for everything.
    """
    # Objective identity lives on the camera/objective param bundle.
    target = getattr(camera, "target", None)
    payload = {
        **_config_payload(obj_or_heur),
        "objective": getattr(target, "name", None),
        "target_shape": getattr(target, "target_shape", None),
        "target_size": getattr(camera, "target_size", None),
        "max_roi_energy_loss": getattr(camera, "max_roi_energy_loss", None),
        "w_uniformity": getattr(camera, "w_uniformity", None),
        "w_peak": getattr(camera, "w_peak", None),
        "w_pearson": getattr(camera, "w_pearson", None),
        "w_displacement": getattr(camera, "w_displacement", None),
        "shape_schedule": getattr(camera, "shape_schedule", None),
        "log_uniformity": getattr(camera, "log_uniformity", None),
        # Camera identity.
        "cam_type": getattr(camera, "cam_type", None),
        "cam_id": getattr(camera, "cam_id", None),
        "exposure_time_ms": getattr(camera, "exposure_time_ms", None),
        "cam_size": getattr(camera, "cam_size", None),
        "center": getattr(camera, "center", None),
        # Perturbation / step size (SpgdParamsPib); absent for the heuristic
        # branch, hence the getattr default.
        "delta": getattr(obj_or_heur, "delta", None),
        "lr": getattr(obj_or_heur, "lr", None),
        "epochs": getattr(obj_or_heur, "epochs", None),
        # Search identity.
        "n_eval_frames": getattr(obj_or_heur, "n_eval_frames", None),
        "noise_gate_k": getattr(obj_or_heur, "noise_gate_k", None),
        "fold_ratio": getattr(obj_or_heur, "fold_ratio", None),
        "optimizer_type": getattr(obj_or_heur, "optimizer_type", None),
        # SLM identity. ``n_max`` lives here (on SlmParams), NOT on the search
        # config — reading it from ``obj_or_heur`` silently dropped it, which
        # left the DOF count (the variable that sets the SPGD signal dilution)
        # unrecorded in every artefact.
        "n_max": getattr(config, "n_max", None),
        "zernike_radius": getattr(config, "zernike_radius", None),
        "slm_number": getattr(config, "slm_number", None),
        "slm_wavelength": getattr(config, "slm_wavelength", None),
    }
    # Drop unset entries so the sidecar stays a clean record of what was set.
    payload = {k: v for k, v in payload.items() if v is not None}
    name = getattr(target, "name", None) or "run"
    return save_recorder_debug_artifacts(
        res,
        root_dir=root_dir,
        subdir_prefix=f"slm_pib_{name}",
        scalar_keys=_DEBUG_SCALAR_KEYS,
        objective_keys=_DEBUG_OBJECTIVE_KEYS,
        img_keys=_DEBUG_IMG_KEYS,
        d1_keys=_DEBUG_1D_KEYS,
        d2_keys=_DEBUG_2D_KEYS,
        json_payload=payload,
        title=f"slm-pib {name} search",
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
            n_eval_frames=search.n_eval_frames,
            fold_ratio=search.fold_ratio,
            noise_gate_k=search.noise_gate_k,
            abba_sampling=search.abba_sampling,
            max_peak=search.max_peak,
        )
    else:  # HeuristicParams
        cfg.update(
            algorithm=search.algorithm,
            pop_size=search.pop_size,
            show=search.show,
            n_eval_frames=search.n_eval_frames,
        )

    return SlmZernikePibConfig(**cfg)


def _resolve_auto_camera(
    camera: CameraParams,
    slm_params: SlmParamsPib,
    find_exposure: bool,
    auto_target_peak: float,
    auto_n_frames: int,
) -> None:
    """Probe the camera for a safe fixed exposure / 0-order centre before optimizing.

    Set the SLM to flat before opening a probe camera, so prior run phases
    cannot bias the exposure. Both probe devices close before the real run.
    Uses ``auto_find_exposure_and_center`` from the shared hardware utils so
    any runner can reuse the same probe logic.
    """
    from ao_shaping.utils.image.hardware_utils import (
        auto_find_exposure_and_center,
        open_camera,
    )

    from ao_shaping.optimizer.wfless.slm_zernike_pib import Santec

    try:
        with Santec.from_params(slm_params) as probe_slm:
            flat = probe_slm.create_phase_from_array(
                np.zeros((SLM_RESOLUTION[1], SLM_RESOLUTION[0]), dtype=np.float32)
            )
            _display(probe_slm, flat)
            time.sleep(0.3)
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


def _resolve_run_context(ctx: click.Context) -> tuple[Path, bool]:
    """Read ``(root_dir, debug)`` inherited from the top-level ``main`` group.

    ``square`` deliberately does **not** take a ``RunParams`` group: that group
    declares ``-d/--dir`` while ``SlmSquareParams`` declares ``-d/--delta``, and
    two options claiming one flag is exactly the silent collision the CLI
    contract freeze guards against. The run-wide switches are therefore read off
    the parent context (``main.py --dir / --debug``) or the ``DEBUG``
    environment variable. ``ctx.parent`` is ``None`` under ``python -m``, which
    falls back to ``data/`` and the env flag.
    """
    parent = ctx.parent
    obj = parent.obj if parent is not None and isinstance(parent.obj, dict) else {}
    return Path(obj.get("dir", "data")), bool(
        get_debug_mode() or obj.get("debug", False)
    )


# --- click group + subcommands ---------------------------------------------


@click.group(invoke_without_command=True)
@click.pass_context
@with_params(RunParams, kw_name="run")
def run(ctx: click.Context, run: RunParams) -> None:
    """SLM shaping with a Santec SLM driven by Zernike coefficients.

    Without a subcommand, the SPGD (gradient) search runs.

    Subcommands: ``spgd`` / ``heuristic`` shape an arbitrary target shape
    (Power-in-Bucket family); ``square`` optimises a uniform square far field.
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


# ---------------------------------------------------------------------------
# square shaping (``spgd-square``)
# ---------------------------------------------------------------------------


def _parse_zernike_mask(raw: str | None) -> np.ndarray | None:
    """Parse ``--zernike-mask`` (``"0,0,0,1,..."``) into a 0/1 int array."""
    if raw is None:
        return None
    return np.array([int(x.strip()) for x in raw.split(",")], dtype=int)


def _active_zernike_modes(
    n_max: int, mask: np.ndarray | None
) -> tuple[list[int], list[tuple[int, int]]]:
    """Return ``(active_positions, active_modes)`` for the SPGD DOF vector.

    ``active_positions`` index the full Noll vector that ``init_c`` is
    expressed in; ``active_modes`` are the ``(n, m)`` pairs, for the startup
    banner and the dropped-coefficient warning. Without a mask the basis is
    :data:`ZERNIKE_ACTIVE_MODES` (defocus + spherical). With one, the mask is
    padded/truncated to the Noll length and Noll 1-3 (piston/tip/tilt) are forced
    off — a gradient on global phase or on a beam shift is not a far-field
    shaping DOF.
    """
    noll = _zernike_indices(n_max)
    if mask is None:
        active = [i for i, mode in enumerate(noll) if mode in ZERNIKE_ACTIVE_MODES]
        return active, list(ZERNIKE_ACTIVE_MODES)

    trimmed = np.asarray(mask, dtype=int).ravel()
    nk = len(noll)
    if trimmed.size < nk:
        trimmed = np.pad(trimmed, (0, nk - trimmed.size), constant_values=0)
    elif trimmed.size > nk:
        trimmed = trimmed[:nk]
    trimmed[0] = trimmed[1] = trimmed[2] = 0
    active = [i for i in range(nk) if trimmed[i] == 1]
    return active, [noll[i] for i in active]


def _resolve_init_coeffs(params: SlmSquareParams) -> np.ndarray | list[float] | None:
    """Resolve the starting Zernike vector from ``--init-coeffs`` / defaults.

    ``--init-coeffs`` accepts a Noll-index dict (``{"4": 1.0, "11": 0.5}``) or a
    flat Noll-ordered array. With the Zernike basis and no explicit value the
    default start is the GUI's Defocus(2,0) + Spherical(4,0); the freeform basis
    starts flat (``None``), because the optimizer generates that phase itself.
    """
    nk = (params.n_max + 1) * (params.n_max + 2) // 2

    if params.init_coeffs is not None:
        try:
            parsed = json.loads(params.init_coeffs)
        except json.JSONDecodeError as exc:
            click.echo(f"Error parsing --init-coeffs: {exc}", err=True)
            sys.exit(1)
        if isinstance(parsed, dict):
            from ao_shaping.utils.wavefront.zernike_calc import noll_to_nm
            from ao_shaping.utils.wavefront.zernike_utils import (
                parse_zernike_coefficients,
            )

            by_mode = parse_zernike_coefficients(parsed, n_max=params.n_max)
            coeffs = np.zeros(nk, dtype=np.float64)
            for j in range(nk):
                mode = noll_to_nm(j + 1)
                if mode in by_mode:
                    coeffs[j] = by_mode[mode]
            return coeffs
        if isinstance(parsed, list):
            return [float(v) for v in parsed]
        click.echo(
            "Error parsing --init-coeffs: expected a Noll-index dict or a "
            f"Noll-ordered array, got {type(parsed).__name__}",
            err=True,
        )
        sys.exit(1)

    if params.basis == "zernike":
        from ao_shaping.utils.wavefront.zernike_calc import noll_to_nm

        coeffs = np.zeros(nk, dtype=np.float64)
        for j in range(nk):
            mode = noll_to_nm(j + 1)
            if mode == (2, 0):
                coeffs[j] = params.init_defocus
            elif mode == (4, 0):
                coeffs[j] = params.init_spherical
        return coeffs

    return None


def _build_square_config(
    params: SlmSquareParams,
    slm_params: ZernikeSlmParams,
    zernike_mask: np.ndarray | None,
) -> SlmSquareConfig:
    """Build the square-shaping optimizer config from the flat param groups.

    ``center`` and ``epochs`` are not part of the container — they stay
    positional arguments of :func:`optimize_slm_square`, the same split
    :class:`SlmZernikePibConfig` uses. ``zernike_mask`` arrives already parsed
    (it is a CSV string on the CLI and an array in the config, so it is
    translated once at the edge rather than by rebinding the parameter field).
    """
    # ``--delta`` defaults to None so Click keeps "user typed nothing" distinct
    # from "user typed 0.1": the lr==0 adaptive schedule *reassigns* delta every
    # epoch, so an explicit flag has to be marked pinned or it is a no-op.
    delta, delta_pinned = resolve_spgd_delta(params.delta)
    return SlmSquareConfig(
        n_max=params.n_max,
        target_side=params.target_side,
        target_mean_brightness=params.target_mean_brightness,
        side_factor=params.side_factor,
        delta=delta,
        delta_pinned=delta_pinned,
        lr=params.lr,
        exposure_time_ms=params.exposure_ms,
        cam_id=params.cam_id,
        cam_type=params.cam_type,
        show=params.show,
        init_c=_resolve_init_coeffs(params),
        cam_size=params.cam_size,
        target_max_brightness=params.target_brightness,
        slm_number=slm_params.slm_number,
        slm_wavelength=slm_params.wavelength,
        optimizer_type=params.optimizer,
        random_seed=params.seed,
        w_uniformity=params.w_uniformity,
        w_efficiency=params.w_efficiency,
        w_aspect=params.w_aspect,
        w_pbr=params.w_pbr,
        basis=params.basis,
        phase_grid=params.phase_grid,
        zernike_radius=params.zernike_radius,
        zernike_mask=zernike_mask,
        rotation_search_deg=params.rotation_search_deg,
        algorithm=params.algorithm,
        pop_size=params.pop_size,
    )


def _square_json_payload(
    params: SlmSquareParams, slm_params: ZernikeSlmParams, config: SlmSquareConfig
) -> dict[str, Any]:
    """Self-describing sidecar: the CLI params plus the resolved config.

    ``config_payload`` drops fields left at their dataclass default, which is the
    right trade for a diff against a baseline but the wrong one for a lone
    artefact directory: a run with every knob at default would record no knobs at
    all. The resolved config is therefore written out field by field, so a
    reader never has to know which default was in force when the run happened.
    """
    payload = {**_config_payload(params), **_config_payload(slm_params)}
    payload.update(
        {
            "family": "spgd-square",
            "basis": config.basis,
            "algorithm": config.algorithm,
            "delta": config.delta,
            "delta_pinned": config.delta_pinned,
            "lr": config.lr,
            "n_max": config.n_max,
            "zernike_radius": config.zernike_radius,
            "zernike_mask": (
                np.asarray(config.zernike_mask).tolist()
                if config.zernike_mask is not None
                else None
            ),
            "cam_type": config.cam_type,
            "cam_id": config.cam_id,
            "cam_size": config.cam_size,
            "exposure_time_ms": config.exposure_time_ms,
            "target_side": config.target_side,
            "target_mean_brightness": config.target_mean_brightness,
            "side_factor": config.side_factor,
            "w_uniformity": config.w_uniformity,
            "w_efficiency": config.w_efficiency,
            "w_aspect": config.w_aspect,
            "w_pbr": config.w_pbr,
            "random_seed": config.random_seed,
            "pop_size": config.pop_size,
            "optimizer_type": config.optimizer_type,
            "rotation_search_deg": config.rotation_search_deg,
        }
    )
    return {k: v for k, v in payload.items() if v is not None}


def _echo_square_banner(
    params: SlmSquareParams,
    slm_params: ZernikeSlmParams,
    config: SlmSquareConfig,
    active_modes: list[tuple[int, int]],
) -> None:
    """Print the run configuration for one square-shaping run."""
    click.echo("=" * 60)
    click.echo("SLM Square Beam Uniformity Optimization (SPGD)")
    click.echo("=" * 60)
    click.echo(
        f"Zernike order: {params.n_max} "
        f"({(params.n_max + 1) * (params.n_max + 2) // 2} terms)"
    )
    if params.target_mean_brightness > 0:
        click.echo(
            f"Target: mean brightness = {params.target_mean_brightness} (auto side)"
        )
    else:
        click.echo(
            f"Target side: {'auto' if params.target_side <= 0 else params.target_side} px"
        )
    click.echo(f"Optimizer: {params.optimizer}")
    click.echo(
        f"Basis: {params.basis}"
        + (
            f" (grid={params.phase_grid})"
            if params.basis == "freeform"
            else f" (radius={params.zernike_radius})"
        )
    )
    if params.basis == "zernike" and params.init_coeffs is None:
        click.echo(
            f"Init Zernike: Defocus(2,0)={params.init_defocus}, "
            f"Spherical(4,0)={params.init_spherical}"
        )
    if params.basis == "zernike":
        if config.zernike_mask is not None:
            click.echo(f"Zernike mask: {np.asarray(config.zernike_mask).tolist()}")
        click.echo(f"Active modes: {active_modes}")
    if params.rotation_search_deg > 0:
        click.echo(
            f"Rotation search: ±{params.rotation_search_deg / 2.0:.1f}° "
            "(SPGD extra DOF)"
        )
    click.echo(f"Epochs: {params.epochs}")
    click.echo(f"SLM: #{slm_params.slm_number} @ {slm_params.wavelength}nm")
    click.echo(f"Camera: ID={params.cam_id}, size={params.cam_size}")
    click.echo(
        f"Weights: CV={params.w_uniformity}, EE={params.w_efficiency}, "
        f"AR={params.w_aspect}"
    )
    click.echo("=" * 60)


def _save_square_summary(recorder: Recorder, root_dir: Path) -> tuple[Path, dict]:
    """Write the CSV history + best Zernike coefficients under ``slm_square/``.

    Kept apart from the ``--debug`` HDF5/PKL artefact writer: these two files are
    the historical ``spgd-square`` outputs and their paths/names are part of the
    run contract. Returns the directory and the best recorded iteration.
    """
    from ao_shaping.utils.io.file import gen_date_dir, gen_date_str

    save_dir = gen_date_dir(root_dir / "slm_square")
    csv_file = save_dir / f"slm_square_{gen_date_str()}.csv"
    recorder.save_dataframe(csv_file)
    click.echo(f"Results saved: {csv_file}")

    best_iter, (best_epoch, best_val) = recorder.get_best_iter()
    click.echo("\n" + "=" * 60)
    click.echo("Optimization Complete")
    click.echo("=" * 60)
    click.echo(f"Best quality: {best_val:.4f} @ epoch {best_epoch}")
    click.echo(f"  CV: {best_iter.get('cv', 'N/A')}")
    click.echo(f"  EE: {best_iter.get('ee', 'N/A')}")
    click.echo(f"  AR: {best_iter.get('ar', 'N/A')}")
    click.echo(f"  Side: {best_iter.get('side', 'N/A')} px")

    best_c = best_iter.get("_c")
    if best_c is not None:
        coeffs_file = save_dir / "best_coefficients.csv"
        np.savetxt(coeffs_file, best_c, fmt="%.6f")
        click.echo(f"Best coefficients saved: {coeffs_file}")
    return save_dir, best_iter


def _save_square_best_image(save_dir: Path, best_iter: dict) -> None:
    """Render the best far field, marking the target box half-diagonal."""
    best_img = best_iter.get("_img")
    if best_img is None:
        return

    side = best_iter.get("side", 0)
    img_file = save_best_image(
        best_img,
        save_dir / "best_square.png",
        title=(
            f"Best square (CV={best_iter.get('cv', 0):.4f}, "
            f"EE={best_iter.get('ee', 0):.4f})"
        ),
        marker_side=side,
    )
    click.echo(f"Best image saved: {img_file}")


@click.command(name="square")
@click.pass_context
@with_params(SlmSquareParams, kw_name="params")
@with_params(ZernikeSlmParams, kw_name="slm_params")
def square(
    ctx: click.Context, params: SlmSquareParams, slm_params: ZernikeSlmParams
) -> None:
    """SLM方形光斑整形优化器 (uniform square far field, camera feedback).

    使用SPGD算法优化SLM上的Zernike系数，通过相机反馈产生均匀方形远场光斑。

    优化目标: 最小化目标方形区域内光强的变异系数(CV = std/mean)，同时可选
    加权环围能量(EE)和宽高比(AR)指标。

    目标方形参数二选一: 边长(px) 或 平均亮度 (--target-side 与
    --target-mean-brightness 互斥)。
    """
    if params.target_side > 0 and params.target_mean_brightness > 0:
        raise click.UsageError(
            "--target-side 与 --target-mean-brightness 互斥: "
            "方形边长(px) 与 方形平均亮度只能二选一"
        )
    if params.slm_type == "sim" and params.cam_type != "sim":
        # The optimizer resolves its SLM from one module global, and a "sim" SLM
        # only exists once the sim patch is installed — which is keyed on the
        # camera backend. Failing loudly beats opening real hardware.
        raise click.UsageError(
            "--slm-type sim 需要同时指定 --cam-type sim (数字孪生 bench 成对提供)"
        )

    root_dir, debug = _resolve_run_context(ctx)
    zernike_mask = _parse_zernike_mask(params.zernike_mask)
    config = _build_square_config(params, slm_params, zernike_mask)
    # Must land before optimize_slm_square resolves either device.
    patch_sim_square_shaping(config.cam_type)

    active_pos, active_modes = _active_zernike_modes(params.n_max, config.zernike_mask)
    if config.basis == "zernike" and config.init_c is not None:
        dropped = [
            i for i, v in enumerate(config.init_c) if v != 0 and i not in active_pos
        ]
        if dropped:
            click.echo(
                f"警告: zernike 基活动模式 {active_modes} "
                f"(Noll 索引 {active_pos}); 非活动系数被忽略: {dropped}"
            )
    _echo_square_banner(params, slm_params, config, active_modes)

    recorder = optimize_slm_square(
        center=parse_center(params.center),
        epochs=params.epochs,
        config=config,
    )

    save_dir, best_iter = _save_square_summary(recorder, root_dir)
    if params.save_best_image:
        _save_square_best_image(save_dir, best_iter)
    if debug:
        save_recorder_debug_artifacts(
            recorder,
            root_dir=str(root_dir),
            subdir_prefix="slm_square",
            scalar_keys=_SQUARE_DEBUG_SCALAR_KEYS,
            img_keys=_SQUARE_DEBUG_IMG_KEYS,
            d1_keys=_SQUARE_DEBUG_1D_KEYS,
            json_payload=_square_json_payload(params, slm_params, config),
            title="spgd-square search",
        )
        click.echo(f"\nDebug data saved to: {root_dir / 'debug'}")


run.add_command(spgd, name="spgd")
run.add_command(heuristic, name="heuristic")
run.add_command(square, name="square")


def _execute(
    run_cfg: RunParams,
    camera: CameraParamsPib,
    slm: SlmParamsPib,
    search: SpgdParamsPib | HeuristicParams,
) -> None:
    """Shared execution path for both search families."""
    setup_coredumpy()
    # Must precede the auto-camera probe *and* the run: with --cam-type sim the
    # optimizer's SLM resolves to the digital twin, never a real Santec. Before
    # this wiring the flag only replaced the camera, so a "sim" pib run died on
    # `SantecError -10002` (no USB) after burning the probe.
    patch_sim_pib_shaping(camera.cam_type)
    if camera.auto_exposure or camera.center == "auto":
        _resolve_auto_camera(
            camera,
            slm,
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
