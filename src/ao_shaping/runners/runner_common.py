"""Shared parameter dataclasses + ``with_params`` click integration.

A single place for the parameter groups that appear in 3+ runner files,
keeping each runner focused on what makes it unique while pulling the CLI
option declarations out of every file.

Concrete file/save-path/visualisation helpers have been relocated to their
canonical homes:

* :func:`build_debug_save_paths` → :mod:`ao_shaping.utils.io.file`
* :func:`save_optimization_debug_artifacts`,
  :func:`_save_data_mode_debug_artifacts`,
  :func:`_infer_objective_key` → :mod:`ao_shaping.utils.io.file`
* :func:`make_debug_wavefront_ax_plots`,
  :func:`save_recorder_artifacts` → :mod:`ao_shaping.utils.image.display`
* :func:`resolve_dm` → :mod:`ao_shaping.drivers.dm._registry`

This module keeps only the parameter dataclasses (with their click option
metadata) plus the ``with_params`` machinery that turns them into CLI options.

Parameter dataclasses are grouped by role:

* 纯硬件参数 (pure hardware)     — device config: CCD camera / SLM / WFS
* 算法参数 (algorithm)           — search knobs: SPGD, heuristic, ...
* 目标参数 (objective)           — target & quality weights (PIB, square)
* 可视化与输出参数 (viz/output)   — run-wide output dir + debug visualisation
* 融合参数 (fused)               — composite groups combining roles above
  (e.g. CameraParamsPib = camera + objective: the slm-pib target definition)

dataclass-click mechanism
-------------------------
**The mechanism itself lives in :mod:`ao_shaping.utils.cli_params`** (TODO R-36)
-- a zero-``ao_shaping``-import leaf shared with ``tools/slm/params.py`` so both
layers can declare options with the same convention without depending on each
other. This module only *uses* it: every parameter class below is written in
that convention.

Each field is declared ``name: Annotated[T, option(...)] = default``. The
dataclass field default is the single source of truth for the CLI default —
``with_params`` injects it into the click option (an explicit ``default=``
inside ``option(...)`` raises ``TypeError``). ``option`` is a delayed
``click.option``: inside ``Annotated`` it returns a ``_DelayedCall`` that
``with_params`` applies to the command in *reversed* declaration order, so the
CLI help lists the options top-to-bottom in the same order the class reads.

Option names come from the declaration, flags first (e.g.
``option("-c", "--center")``) — never repeat the field name as the first
positional. The click type is inferred from the field annotation
(``str``/``int``/``float``/``bool``/``Path``; ``Optional[x]`` / ``x | None``
strips to ``x``) unless ``type=``, ``callback=``, ``is_flag`` or ``multiple``
is given explicitly. A union of two concrete types (e.g.
``str | tuple[int, int]``) MUST pass an explicit ``type=``.

The wrapped command receives one keyword argument per decorator, named by
``kw_name``, holding a fully-populated instance of the parameter class.

Accepted CLI help diffs (vs. the pre-refactor per-runner decorator stacks):

1. ``RunParams`` lists ``-d/--dir`` first (previously ``--debug`` first).
2. Fused ``CameraParamsPib`` lists the objective block before the camera
   block, with the ``--auto-*`` flags after ``-c/--center`` (byte-identical
   to the pre-refactor slm-pib CLI help).
3. ``SlmParamsPib`` lists ``--slm_number`` first (previously ``--init_c``
   first — the decorator-stack help was reversed).
4. ``SpgdParamsPib`` lists ``--show`` before ``--shrink_iter`` (previously
   ``--shrink_iter`` first).
5. Objective blocks read top-to-bottom (previously reversed).
6. ``slm-gsnet`` gains a ``--seed`` option (from ``RunParams``); it had no
   seed option before — its optimizer kwargs hardcoded ``random_seed=None``.

Note: never paste Windows paths with ``\\U``/``\\u`` escapes into docstrings or
comments verbatim — they start unicode escapes and raise ``SyntaxError``.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Annotated, Any, cast

import click

from ao_shaping.algorithm.heuristic.search import heuristic_algorithm_choices
from ao_shaping.utils.image.target import (
    SHAPING_OBJECTIVE_CHOICES,
    SQUARE_OBJECTIVE_CHOICES,
    TARGET_SHAPE_CHOICES,
)
from ao_shaping.utils.io.cli_helpers import parse_tuple
from ao_shaping.utils.wavefront.matrix_utils import (
    focal_length_from_camera_pixel,
)

from ao_shaping.drivers.dm import list_dm_types
from ao_shaping.drivers.slm.santec.slm200_constants import PANEL_RES

# The dataclass-click mechanism (the ``Annotated[..., option(...)]`` convention,
# the ``with_params`` collector and object delivery) is a zero-``ao_shaping``
# leaf shared with ``tools/slm/params.py``. Defined here, not here.
from ao_shaping.utils.cli_params import ClickGroup, option, with_params

# slm-pib 的 Zernike 孔径半径默认值: SLM 面板短边的一半 (PANEL_RES = (1920, 1200) -> 600 px),
# 与方形整形 (slm_square_shaping) 及 GUI 的默认值一致; 取短边保证基圆完整落在面板内。
DEFAULT_ZERNIKE_RADIUS = min(PANEL_RES) / 2.0

# Bench geometry for the 2f-Fourier path.
#
# The measured focal scale (a 2*pi ramp over P SLM px moves the spot
# _TILT_SHIFT_SCALE/P camera px) constrains only the RATIO f/p_cam:
#
#     focal_scale = lambda * f / (d_slm * p_cam)
#
# so recovering either one needs one quantity from outside the alignment
# measurement. We anchor on the CCD pixel pitch, because that is a per-camera
# datasheet constant we have confirmed out of band (2.2 um for the Daheng
# MER2-507-23GM, recorded in drivers/AGENTS.md), whereas the lens focal length
# is a bench ASSEMBLY choice that silently changes whenever somebody swaps
# optics -- which is exactly the kind of value that must not be baked in.
#
# Consequence worth knowing: the GS *target side* consumes f/p_cam only
# (slm_gs_refine._derive_target_side), so it is pinned by the measured scale and
# is completely insensitive to which anchor we pick. Only the GS propagation
# phase uses f absolutely, and its bake-off stage catches a wrong f.
#
# Values this replaced, and why:
#   * camera_pixel_um = 3.31 um implied a focal scale of 5023 -- 33% below the
#     measured 7400 -- biasing the GS target angular size by the same factor.
#   * focal_length_m = 0.125 m is now DERIVED (see _DEFAULT_FOCAL_LENGTH_M).
_TILT_SHIFT_SCALE_PX = 7400.0
_DEFAULT_CAMERA_PIXEL_UM = 2.2
_DEFAULT_FOCAL_LENGTH_M = focal_length_from_camera_pixel(
    wavelength_nm=1064.0,
    camera_pixel_um=_DEFAULT_CAMERA_PIXEL_UM,
    slm_pixel_um=8.0,
    focal_scale_px=_TILT_SHIFT_SCALE_PX,
)

# Snapshot of DM types taken BEFORE asyn_micro_dm registration below. The
# dm_matrix_runner at HEAD computed its own DM_TYPES at import time before
# asyn_micro was registered (runners/__init__.py imports dm_matrix_runner
# before full_voltage_runner), so its --dm_type choice list excluded asyn_micro.
# Preserve that exact ordering for byte-identical help output.
DM_TYPES_PRE_ASYN_MICRO = list_dm_types()

# Importing the async driver module registers the "asyn_micro" DM type (side
# effect). It is normally registered by micro_drive.full_voltage_runner, which is
# imported AFTER this module in runners/__init__.py — without this import,
# DM_TYPES below would miss asyn_micro and the --dm_type choice list would
# silently shrink from 7 to 6 entries. The submodule is imported directly
# rather than via ``dm.micro``, whose __getattr__ resolves the async driver
# lazily (that laziness is what keeps DM_TYPES_PRE_ASYN_MICRO above honest).
import ao_shaping.drivers.dm.micro.asyn_driver  # noqa: F401

DM_TYPES = list_dm_types()


# ---------------------------------------------------------------------------
# 纯硬件参数 | pure hardware parameters — device configuration
# (CCD camera / Santec SLM / Thorlabs WFS)
# ---------------------------------------------------------------------------


#: SPGD perturbation amplitude used when the caller does not pass ``--delta``.
#: Mirrors the historical per-family defaults so omitting the flag is
#: behaviour-preserving.
DEFAULT_SPGD_DELTA = 0.1


def resolve_spgd_delta(
    delta: float | None, *, default: float = DEFAULT_SPGD_DELTA
) -> tuple[float, bool]:
    """Resolve a ``--delta`` option into ``(value, pinned)``.

    ``SpgdParams.delta`` defaults to ``None`` so a runner can tell "the user
    typed ``--delta``" apart from "the user typed nothing" -- Click collapses both
    cases into one value otherwise. That distinction matters because the
    ``lr == 0`` adaptive schedule *reassigns* ``delta`` on every epoch: an
    explicit ``--delta`` used to be silently overwritten unless the caller also
    passed an explicit ``--lr``, making the flag a no-op.

    Args:
        delta: The raw option value, or ``None`` when the flag was omitted.
        default: Value to use when the flag was omitted.

    Returns:
        ``(delta, pinned)`` where ``pinned`` is ``True`` only if the caller
        supplied the value explicitly (and therefore wants it respected).
    """
    if delta is None:
        return abs(float(default)), False
    return abs(float(delta)), True


@dataclass
class CameraParams:
    """CCD camera options (slm-gsnet family)."""

    cam_id: Annotated[int, option("--cam-id", help="CCD camera device ID")] = 0
    cam_type: Annotated[
        str,
        option(
            "--cam_type",
            type=click.Choice(["miicam", "daheng", "sim"]),
            help="CCD camera backend (sim = 2f-Fourier numerical simulation, no hardware).",
        ),
    ] = "daheng"
    exposure_time_ms: Annotated[
        float,
        option(
            "--exposure_time_ms",
            help=(
                "CCD exposure time in ms. 0 (the default) means 'no fixed value': "
                "the camera keeps whatever it is set to, and auto-exposure "
                "brackets it when a target brightness is available. On Daheng a "
                "0 that reaches the driver is CLAMPED to the device minimum "
                "(~0.02 ms), so bracket the exposure explicitly with "
                "slm_drift_probe before trusting an unbracketed run."
            ),
        ),
    ] = 0.0
    cam_size: Annotated[
        int, option("--cam_size", help="CCD window size in pixels.")
    ] = 300
    center: Annotated[
        str | tuple[int, int] | None,
        option(
            "-c",
            "--center",
            type=click.STRING,
            help="Spot center: 'shape' / 'max' / 'mass' / 'centroid_thresh' or 'x,y'.",
        ),
    ] = None


@dataclass
class SlmParams:
    """Santec SLM options (slm-gsnet family)."""

    slm_number: Annotated[
        int, option("--slm_number", help="Santec SLM device number (1-8).")
    ] = 1
    slm_wavelength: Annotated[
        int,
        option(
            "--slm_wavelength",
            help=(
                "SLM operating wavelength (nm); 0 (the default) asks the "
                "device which wavelength it is programmed for."
            ),
        ),
    ] = 0
    n_max: Annotated[int, option("-n", "--n_max", help="Max Zernike radial order.")] = 4
    zernike_radius: Annotated[
        float,
        option(
            "--zernike_radius", help="Zernike aperture radius (pixels); 0 = default."
        ),
    ] = 0.0


@dataclass
class WfsParams:
    """Thorlabs WFS options (rms-zernike, ga-zernike, greedy-zernike share these)."""

    wfs_res: Annotated[
        str, option("-r", "--wfs_res", help="WFS分辨率 (default: 1024)")
    ] = "1024"
    pupil_diameter: Annotated[
        float, option("-p", "--pupil_diameter", help="瞳孔直径 (default: 2.7)")
    ] = 2.7
    pupil_center: Annotated[
        str | tuple[float, float],
        option(
            "-c",
            "--pupil_center",
            callback=parse_tuple,
            help="瞳孔中心坐标 (default: (0,0))",
        ),
    ] = "(0,0)"
    exposure_time_ms: Annotated[
        float,
        option(
            "--exposure-time-ms",
            type=float,
            help="WFS曝光时间 (毫秒, default: 0.0=自动曝光)",
        ),
    ] = 0.0
    remove_tilt: Annotated[
        bool, option("--remove-tilt", is_flag=True, help="移除波前测量中的倾斜项")
    ] = False
    wfs_type: Annotated[
        str,
        option(
            "--wfs_type",
            type=click.Choice(["thorlab", "sim"]),
            help="波前传感器类型 (sim=仿真 Shack-Hartmann, 无需硬件)",
        ),
    ] = "thorlab"


@dataclass
class ZernikeSlmParams:
    """Zernike SLM options (rms-zernike, ga-zernike, greedy-zernike share these)."""

    wavelength: Annotated[
        int, option("--wavelength", help="SLM波长 (nm, default: 532)")
    ] = 532
    shift_x: Annotated[
        int, option("--shift-x", help="SLM X方向平移 (像素, default: 0)")
    ] = 0
    shift_y: Annotated[
        int, option("--shift-y", help="SLM Y方向平移 (像素, default: 0)")
    ] = 0
    slm_number: Annotated[
        int, option("--slm-number", help="SLM设备编号 (default: 1)")
    ] = 1
    wait_time: Annotated[
        float, option("--wait-time", help="SLM 液晶翻转等待时间(秒, default: 0.3) ")
    ] = 0.3


@dataclass
class ThorlabWfsDriverParams:
    """Thorlab WFS 驱动级选项 (dm-matrix, hadamard-matrix, zernike-matrix share these)."""

    mla_index: Annotated[
        str,
        option(
            "--mla-index",
            type=click.Choice(["512", "540", "600", "768", "1280"]),
            help="MLA分辨率 (默认: 512)",
        ),
    ] = "512"
    exp_time: Annotated[
        float, option("--exp-time", help="WFS曝光时间 (ms, 0=自动)")
    ] = 0.0
    auto_exposure: Annotated[
        bool,
        option("--auto-exposure/--no-auto-exposure", help="启用WFS自动曝光 (默认开启)"),
    ] = True
    high_speed: Annotated[
        bool, option("--high-speed", is_flag=True, help="启用高速模式")
    ] = False
    use_custom_ref: Annotated[
        bool, option("--use-custom-ref", is_flag=True, help="使用自定义参考文件")
    ] = False


# ---------------------------------------------------------------------------
# 算法参数 | algorithm parameters — search / optimisation knobs
# ---------------------------------------------------------------------------


@dataclass
class SpgdParams:
    """SPGD (gradient) search options (slm-gsnet family)."""

    epochs: Annotated[
        int, option("-e", "--epochs", help="Optimization iterations.")
    ] = 2000
    delta: Annotated[
        float | None,
        option(
            "--delta",
            type=float,
            help=(
                "SPGD perturbation amplitude (rad). Omit to let the adaptive "
                "schedule choose it; passing a value PINS it and disables the "
                "schedule's delta update."
            ),
        ),
    ] = None
    lr: Annotated[float, option("--lr", help="SPGD learning rate (0 = auto).")] = 0.0
    optimizer_type: Annotated[
        str,
        option(
            "--optimizer_type",
            type=click.Choice(
                ["adam", "adamw", "adamod", "sgd", "muno", "munow"],
                case_sensitive=False,
            ),
            show_default=True,
            help="SPGD gradient optimizer.",
        ),
    ] = "adamod"
    show: Annotated[
        bool, option("--show", is_flag=True, help="Open a live display window.")
    ] = False


@dataclass
class HeuristicParams:
    """Black-box heuristic search options (slm-gsnet heuristic subcommand)."""

    algorithm: Annotated[
        str,
        option(
            "--algorithm",
            type=click.Choice(heuristic_algorithm_choices()),
            help="Black-box search algorithm.",
        ),
    ] = "ga"
    pop_size: Annotated[
        int | None,
        option("--pop_size", type=int, help="Population size (ga/pso/cem/de)."),
    ] = None
    epochs: Annotated[
        int, option("-e", "--epochs", help="Optimization iterations.")
    ] = 2000
    n_eval_frames: Annotated[
        int,
        option(
            "--n-eval-frames",
            help="Frames averaged per candidate camera read (1 = legacy).",
        ),
    ] = 1
    show: Annotated[
        bool, option("--show", is_flag=True, help="Open a live display window.")
    ] = False


# slm-pib variant of the shared SPGD params (larger default perturbation +
# radius/step shrink knobs; --show stays on the base class).
@dataclass
class SpgdParamsPib(SpgdParams):
    """SPGD options — slm-pib uses a larger default perturbation (0.2 rad)."""

    delta: Annotated[
        float, option("--delta", help="SPGD perturbation amplitude (rad).")
    ] = 0.2
    shrink_iter: Annotated[
        int, option("--shrink_iter", help="Iterations before radius/step shrink.")
    ] = 0
    shrink_ratio: Annotated[
        float, option("--shrink_ratio", help="Radius/step shrink ratio.")
    ] = 0.9
    n_eval_frames: Annotated[
        int,
        option("--n-eval-frames", help="Frames averaged per camera read (1 = legacy)."),
    ] = 1
    fold_ratio: Annotated[
        float,
        option(
            "--fold-ratio",
            help="Brightness-fold epoch gate ratio (0 disables; see slm_zernike_pib).",
        ),
    ] = 0.5
    noise_gate_k: Annotated[
        float,
        option(
            "--noise-gate-k",
            help="Noise-aware update gate sigma multiplier (0 disables).",
        ),
    ] = 3.0
    abba_sampling: Annotated[
        bool,
        option(
            "--abba-sampling",
            is_flag=True,
            help=(
                "Enable ABBA sampling: capture 4 frames per epoch in order (+ - - +) "
                "instead of 2 (+ -) to cancel linear slow drift (4x captures/epoch). "
                "Default OFF for byte-identical behavior."
            ),
        ),
    ] = False


# ---------------------------------------------------------------------------
# 目标参数 | objective parameters — imaging target & quality weights
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveTarget:
    """The imaging objective, with its target shape nested underneath.

    ``target_shape`` is a *subordinate* field of the objective: it defines the
    target for ``shape``, and only selects the ROI for ``roi_pib`` / ``rms_pib``
    / ``rmse`` / ``rmse_out``. It is unused for ``pib`` / ``radiu`` /
    ``avg_radiu``. Validate a combination with
    :class:`~ao_shaping.utils.image.target.ObjectiveSpec`, the single source of
    truth for those rules.
    """

    name: Annotated[
        str,
        option(
            "--objective",
            # Derived from the canonical tuple so the CLI can never drift from
            # the resolver (see plan R4).
            type=click.Choice(list(SHAPING_OBJECTIVE_CHOICES)),
            help="Optimization objective.",
        ),
    ] = "pib"
    target_shape: Annotated[
        str | None,
        option(
            "--target_shape",
            type=click.Choice(list(TARGET_SHAPE_CHOICES)),
            help="Target ROI shape (implies the 'shape' objective).",
        ),
    ] = None


@dataclass
class ObjectiveParamsPib:
    """Imaging objective options (shared by both slm-pib search families)."""

    target: Annotated[
        ObjectiveTarget,
        ClickGroup(),
    ] = field(default_factory=ObjectiveTarget)
    target_max_brightness: Annotated[
        int,
        option(
            "--target_max_brightness", help="Target max brightness for auto-exposure."
        ),
    ] = 40
    r_bucket: Annotated[
        int,
        option("-r", "--r_bucket", help="Bucket radius (0 = auto from power radius)."),
    ] = 0
    target_size: Annotated[
        float, option("--target_size", help="Target extent in camera px.")
    ] = 64.0
    target_aspect_ratio: Annotated[
        float,
        option(
            "--target_aspect_ratio", help="Width:height ratio for a rectangular target."
        ),
    ] = 4.0 / 3.0
    target_center_smooth: Annotated[
        int,
        option(
            "--target_center_smooth",
            help="Frames averaged for the target centre estimate.",
        ),
    ] = 3
    shape_schedule: Annotated[
        bool,
        option(
            "--shape_schedule",
            is_flag=True,
            help="Use the coarse->fine shaping weight schedule.",
        ),
    ] = False
    max_roi_energy_loss: Annotated[
        float,
        option(
            "--max_energy_loss",
            help="Max allowed in-ROI energy loss fraction (0 disables the guard).",
        ),
    ] = 0.6
    w_uniformity: Annotated[
        float, option("--w_uniformity", help="Uniformity penalty weight.")
    ] = 2.0
    w_peak: Annotated[float, option("--w_peak", help="Peak penalty weight.")] = 0.5
    w_displacement: Annotated[
        float, option("--w_displacement", help="Displacement penalty weight.")
    ] = 0.0
    log_uniformity: Annotated[
        bool,
        option(
            "--log_uniformity",
            is_flag=True,
            help="Use log1p(u) instead of u/(1+u) for the uniformity term.",
        ),
    ] = False
    w_ema_decay: Annotated[
        float,
        option(
            "--w_ema_decay",
            help="EMA decay for the adaptive PIB/RMS weights of the 'rms_pib' objective.",
        ),
    ] = 0.9
    w_floor: Annotated[
        float,
        option(
            "--w_floor",
            help="Minimum weight floor per term of the 'rms_pib' objective (0..0.5).",
        ),
    ] = 0.1
    w_temperature: Annotated[
        float,
        option(
            "--w_temperature",
            help="Softmax temperature for the 'rms_pib' weight update.",
        ),
    ] = 8.0
    # Initial weights of the 'rms_pib' objective (None = default 1/3 each).
    # Provided terms are kept exactly; unprovided terms share the remainder.
    w_pib_init: Annotated[
        float | None,
        option(
            "--w_pib_init",
            type=float,
            help="Initial PIB weight of the 'rms_pib' objective (default 1/3).",
        ),
    ] = None
    w_rms_init: Annotated[
        float | None,
        option(
            "--w_rms_init",
            type=float,
            help="Initial RMS (in-ROI uniformity) weight of the 'rms_pib' objective (default 1/3).",
        ),
    ] = None
    w_ee_init: Annotated[
        float | None,
        option(
            "--w_ee_init",
            help="Initial encircled-energy weight of the 'rms_pib' objective (default 1/3).",
        ),
    ] = None

    # --- Compatibility (read-only forwarding; remove after one release) ------
    # Reads keep working unchanged, so consumers — including the core of
    # slm_zernike_pib — need no edits. Flat *construction*
    # (``ObjectiveParamsPib(name=...)``) is intentionally a TypeError so that
    # every missed call site surfaces loudly instead of drifting silently.
    # Migrate those to ``ObjectiveParamsPib(target=ObjectiveTarget(...))``.

    @property
    def name(self) -> str:
        """Objective name — forwards to ``self.target.name``."""
        return self.target.name

    @property
    def target_shape(self) -> str | None:
        """Target ROI shape — forwards to ``self.target.target_shape``."""
        return self.target.target_shape


@dataclass
class ObjectiveParamsSquare:
    """Square-shaping objective options (slm-gsnet family)."""

    name: str = "square"
    target_side: Annotated[
        int,
        option(
            "--target-side",
            help="Target square side (pixels); 0 = auto from spot size. Mutually "
            "exclusive with --target-mean-brightness.",
        ),
    ] = 0
    target_mean_brightness: Annotated[
        float,
        option(
            "--target-mean-brightness",
            help="Target square mean brightness (gray). >0 auto-derives side by "
            "energy conservation. Mutually exclusive with --target-side.",
        ),
    ] = 0.0
    side_factor: Annotated[
        float, option("--side-factor", help="Auto side-length factor.")
    ] = 1.5
    target_max_brightness: Annotated[
        int,
        option(
            "--target-max-brightness", help="Target max brightness for auto-exposure."
        ),
    ] = 200
    w_uniformity: Annotated[
        float, option("--w_uniformity", help="Uniformity (CV) weight.")
    ] = 0.4
    w_efficiency: Annotated[
        float, option("--w_efficiency", help="Encircled-energy weight.")
    ] = 0.6
    w_aspect: Annotated[float, option("--w_aspect", help="Aspect-ratio weight.")] = 0.0
    w_pbr: Annotated[
        float,
        option(
            "--w_pbr",
            help=(
                "Background-suppression (peak-to-background) weight. "
                "0 = off (default), which keeps runs reproducible. PBR is the "
                "box peak divided by the mean of the rest of the frame, "
                "median-subtracted and clipped so read noise cannot push it "
                "above 1. Transcribed from the prose definition in Liu et al., "
                "Acta Photonica Sinica 2023, 52(6):0629002."
            ),
        ),
    ] = 0.0
    objective: Annotated[
        str,
        option(
            "--objective",
            type=click.Choice(SQUARE_OBJECTIVE_CHOICES),
            help="Square-shaping objective (larger is better): 'quality' = combined "
            "uniformity+energy+aspect score (default); 'pearson' = FourierGSNet "
            "1 - Pearson correlation loss. WARNING: 'pearson' is mean-centred, so it "
            "is invariant to intensity scale and cannot see absolute energy — the "
            "search can improve the correlation by pushing light OUT of the target "
            "box (the same failure mode AGENTS.md records for bare -CV, where "
            "hardware EE collapsed to 0.002). Unlike 'slm-pib', this square path "
            "has NO encircled-energy guard, so prefer 'quality' unless you are "
            "monitoring the 'ee' Recorder column and will stop the run yourself.",
        ),
    ] = "quality"


# ---------------------------------------------------------------------------
# 可视化与输出参数 | visualisation & output parameters — run-wide controls
# ---------------------------------------------------------------------------


@dataclass
class RunParams:
    """Global / run-wide options shared by subcommands.

    Not tied to a single device or objective role: ``dir`` is the output
    root, ``debug`` enables debug-artifact visualisation, ``seed`` controls
    the random stream (reproducible only in 'sim' mode).
    """

    dir: Annotated[str, option("-d", "--dir", help="Data root directory.")] = "data"
    debug: Annotated[
        bool, option("--debug", is_flag=True, help="Enable debug mode.")
    ] = False
    seed: Annotated[
        int | None,
        option(
            "--seed",
            type=int,
            help="Random seed for reproducible runs (reproducible only in 'sim' mode).",
        ),
    ] = None


# ---------------------------------------------------------------------------
# 单命令参数 | single-command parameters — rms-zernike / greedy-zernike
# (声明序 == help 序; 硬件块由 WfsParams / ZernikeSlmParams 单独提供)
# ---------------------------------------------------------------------------


@dataclass
class RmsZernikeParams:
    """SLM-Zernike RMS 优化的全部 CLI 参数 (rms-zernike, 单命令无子命令)。

    字段即搜索结果/调度超参 + 输出控制 (dir/debug); WFS 与 SLM 硬件块
    分别由 ``WfsParams`` / ``ZernikeSlmParams`` 提供。
    """

    dir: Annotated[
        str, option("-d", "--dir", help="数据保存根目录 (default: data)")
    ] = "data"
    epochs: Annotated[
        int, option("-e", "--epochs", help="优化迭代次数 (default: 20000)")
    ] = 20000
    n_max: Annotated[
        int, option("-n", "--n-max", help="Zernike最大阶数 (default: 4)")
    ] = 4
    lr: Annotated[float, option("--lr", help="学习率 (default: 0.01)")] = 0.01
    delta: Annotated[float, option("--delta", help="初始delta值 (default: 0.0)")] = 0.0
    early_stop_threshold: Annotated[
        float, option("-t", "--early_stop_threshold", help="早停阈值 (default: 0.12)")
    ] = 0.12
    min_delta: Annotated[
        float,
        option("--min-delta", help="自动检测最小delta (数量级扫描, default: 0.01)"),
    ] = 0.01
    max_delta: Annotated[
        float,
        option("--max-delta", help="自动检测最大delta (数量级扫描, default: 100.0)"),
    ] = 100.0
    delta_step: Annotated[
        int, option("--delta-step", help="数量级扫描步数 (用于细粒度扫描, default: 5)")
    ] = 5
    n_directions: Annotated[
        int, option("--n-directions", help="每个delta采样次数防噪声 (default: 5)")
    ] = 5
    n_init_positions: Annotated[
        int,
        option(
            "--n-init-positions", help="多起点优化：随机初始位置数量 (default: 0, 禁用)"
        ),
    ] = 0
    init_range: Annotated[
        float, option("--init-range", help="多起点初始化的随机范围 (default: 1.0)")
    ] = 1.0
    lr_schedule: Annotated[
        str,
        option(
            "--lr-schedule",
            type=click.Choice(["static", "cosine", "exp", "linear"]),
            help="学习率调度类型 (default: static)",
        ),
    ] = "static"
    lr_min: Annotated[
        float, option("--lr-min", type=float, help="学习率最小值 (default: 1e-6)")
    ] = 1e-6
    delta_schedule: Annotated[
        str,
        option(
            "--delta-schedule",
            type=click.Choice(["static", "cosine", "exp", "linear"]),
            help="Delta调度类型 (default: static)",
        ),
    ] = "static"
    delta_min: Annotated[
        float, option("--delta-min", type=float, help="Delta最小值 (default: 1e-7)")
    ] = 1e-7
    optimizer: Annotated[
        str,
        option(
            "--optimizer",
            type=click.Choice(["adamod", "adamw"]),
            help="优化器类型 (default: adamod)",
        ),
    ] = "adamod"
    beta1: Annotated[
        float, option("--beta1", type=float, help="Adam beta1参数 (default: 0.95)")
    ] = 0.95
    weight_decay: Annotated[
        float,
        option("--weight-decay", type=float, help="AdamW权重衰减 (default: 1e-2)"),
    ] = 1e-2
    mini_batch: Annotated[
        int, option("--mini-batch", type=int, help="SPGD mini-batch大小 (default: 1)")
    ] = 1
    gradient_clip: Annotated[
        float,
        option("--gradient-clip", type=float, help="梯度裁剪阈值 (default: 0.0, 禁用)"),
    ] = 0.0
    stagnation_patience: Annotated[
        int,
        option("--stagnation-patience", type=int, help="停滞检测轮数 (default: 30)"),
    ] = 30
    stagnation_delta_boost: Annotated[
        float,
        option(
            "--stagnation-delta-boost",
            type=float,
            help="停滞时delta倍增 (default: 1.5)",
        ),
    ] = 1.5
    freeze_threshold: Annotated[
        float | None,
        option(
            "--freeze-threshold", type=float, help="冻结高阶模式阈值 (default: None)"
        ),
    ] = None
    early_stop_window: Annotated[
        int,
        option("--early-stop-window", type=int, help="早停滑动窗口大小 (default: 0)"),
    ] = 0
    early_stop_min_epochs: Annotated[
        int,
        option("--early-stop-min-epochs", type=int, help="早停最小轮数 (default: 0)"),
    ] = 0
    early_stop_patience: Annotated[
        int, option("--early-stop-patience", type=int, help="早停耐心值 (default: 0)")
    ] = 0
    n_frames: Annotated[
        int, option("--n-frames", type=int, help="WFS帧平均数 (default: 10)")
    ] = 10
    algorithm: Annotated[
        str,
        option(
            "--algorithm",
            type=click.Choice(
                list(heuristic_algorithm_choices()), case_sensitive=False
            ),
            show_default=True,
            help="搜索算法: spgd (梯度/SPGD) 或启发式 (ga/pso/sa/hc/rs/cem/de)",
        ),
    ] = "spgd"
    pop_size: Annotated[
        int | None,
        option(
            "--pop_size",
            type=int,
            help="种群规模 (ga/pso/cem/de 使用; 默认取算法默认值)",
        ),
    ] = None
    debug: Annotated[
        bool,
        option("--debug", is_flag=True, help="启用调试模式: 保存 pkl/json 与汇总图"),
    ] = False


@dataclass
class GreedyZernikeParams:
    """贪婪/启发式 Zernike 优化 CLI 参数 (greedy-zernike, 单命令无子命令)。"""

    dir: Annotated[
        str, option("-d", "--dir", help="数据保存根目录 (default: data)")
    ] = "data"
    epochs: Annotated[
        int, option("-e", "--epochs", help="优化迭代次数 (default: 2000)")
    ] = 2000
    n_max: Annotated[
        int, option("-n", "--n-max", help="Zernike最大阶数 (default: 4)")
    ] = 4
    early_stop_threshold: Annotated[
        float, option("-t", "--early_stop_threshold", help="早停阈值 (default: 0.12)")
    ] = 0.12
    show: Annotated[
        bool,
        option(
            "--show",
            is_flag=True,
            help="显示远场光斑CCD图像和优化历史 (default: False)",
        ),
    ] = False
    n_init: Annotated[
        int, option("--n-init", help="初始随机位置数量 (default: 10)")
    ] = 10
    n_directions: Annotated[
        int, option("--n-directions", help="每次迭代的随机方向数量 (default: 5)")
    ] = 5
    perturbation_scale: Annotated[
        float, option("--perturbation-scale", help="扰动幅度缩放因子 (default: 5.0)")
    ] = 5.0
    algorithm: Annotated[
        str,
        option(
            "--algorithm",
            type=click.Choice(
                list(heuristic_algorithm_choices()), case_sensitive=False
            ),
            show_default=True,
            help="搜索算法: spgd (贪婪局部搜索) 或启发式 (ga/pso/sa/hc/rs/cem/de)",
        ),
    ] = "spgd"
    pop_size: Annotated[
        int | None,
        option(
            "--pop_size",
            type=int,
            help="种群规模 (ga/pso/cem/de 使用; 默认取算法默认值)",
        ),
    ] = None
    debug: Annotated[
        bool,
        option("--debug", is_flag=True, help="启用调试模式: 保存 pkl/json 与汇总图"),
    ] = False


@dataclass
class GaZernikeParams:
    """遗传算法 Zernike 优化 CLI 参数 (ga-zernike, 单命令无子命令)。"""

    dir: Annotated[
        str, option("-d", "--dir", help="数据保存根目录 (default: data)")
    ] = "data"
    population_size: Annotated[
        int, option("--population-size", help="种群大小 (default: 50)")
    ] = 50
    n_generations: Annotated[
        int, option("--n-generations", help="GA迭代代数 (default: 2000)")
    ] = 2000
    crossover_prob: Annotated[
        float, option("--crossover-prob", help="交叉概率 (default: 0.7)")
    ] = 0.7
    mutation_prob: Annotated[
        float, option("--mutation-prob", help="变异概率 (default: 0.15)")
    ] = 0.15
    tournament_size: Annotated[
        int, option("--tournament-size", help="锦标赛选择大小 (default: 3)")
    ] = 3
    elite_count: Annotated[
        int, option("--elite-count", help="精英个体数量 (default: 2)")
    ] = 2
    n_max: Annotated[
        int, option("-n", "--n-max", help="最大Zernike径向阶数 (default: 4)")
    ] = 4
    early_stop_threshold: Annotated[
        float, option("--early-stop-threshold", help="早停RMS阈值 (default: 0.01)")
    ] = 0.01
    show: Annotated[
        bool, option("--show", is_flag=True, help="显示优化历史 (default: False)")
    ] = False


@dataclass
class SlmSquareParams:
    """SLM 方形光斑 SPGD 整形 CLI 参数 (spgd-square, 单命令无子命令)。"""

    epochs: Annotated[
        int, option("-e", "--epochs", help="优化迭代次数 (default: 2000)")
    ] = 2000
    n_max: Annotated[
        int, option("-n", "--n-max", help="Zernike最大径向阶数 (default: 4)")
    ] = 4
    center: Annotated[
        str,
        option(
            "-c",
            "--center",
            help="光斑中心检测: shape(智能argmax锚定,默认)/centroid_thresh(亮度重心)"
            "/max(峰值位置)/mass(质心,易被杂散光拉偏)/'x,y'(固定坐标) (default: shape)",
        ),
    ] = "shape"
    target_side: Annotated[
        int,
        option(
            "--target-side",
            help="目标方形边长(像素), 0=自动; 与 --target-mean-brightness 互斥 (default: 0)",
        ),
    ] = 0
    target_mean_brightness: Annotated[
        float,
        option(
            "--target-mean-brightness",
            help="目标方形平均亮度(灰度), >0 时由总亮度能量守恒自动推导边长; "
            "与 --target-side 互斥 (default: 0 = 不启用)",
        ),
    ] = 0.0
    side_factor: Annotated[
        float, option("--side-factor", help="自动边长倍率 (default: 1.5)")
    ] = 1.5
    delta: Annotated[float, option("-d", "--delta", help="扰动幅度 (default: 0.1)")] = (
        0.1
    )
    lr: Annotated[float, option("--lr", help="学习率, 0=自动 (default: 0)")] = 0.0
    exposure_ms: Annotated[
        float,
        option(
            "-t",
            "--exposure-ms",
            help=(
                "相机曝光时间ms. 0(默认)=不固定, 由设备保持/自动曝光; "
                "Daheng 上 0 会被驱动钳到设备最小值(~0.02ms), "
                "请先用 slm_drift_probe 重新 bracket"
            ),
        ),
    ] = 0.0
    cam_id: Annotated[int, option("--cam-id", help="相机设备ID (default: 0)")] = 0
    cam_type: Annotated[
        str,
        option(
            "--cam_type",
            type=click.Choice(["miicam", "daheng", "sim"]),
            help="CCD 相机后端 (sim = 2f-Fourier 数值仿真, 无需硬件)。",
        ),
    ] = "daheng"
    slm_type: Annotated[
        str,
        option(
            "--slm_type",
            type=click.Choice(["santec", "sim"]),
            help="SLM 后端 (sim = 2f-Fourier 数值仿真, 无需硬件; 需配合 --cam_type sim)。",
        ),
    ] = "santec"
    cam_size: Annotated[
        int, option("-s", "--cam-size", help="相机开窗大小 (default: 300)")
    ] = 300
    optimizer: Annotated[
        str,
        option(
            "--optimizer",
            help="优化器: adam/adamod/sgd/muno (default: adamod; 仅 --algorithm spgd 生效)",
        ),
    ] = "adamod"
    algorithm: Annotated[
        str,
        option(
            "--algorithm",
            type=click.Choice(
                list(heuristic_algorithm_choices()), case_sensitive=False
            ),
            show_default=True,
            help="搜索算法: spgd (SPGD 梯度) 或启发式 (ga/pso/sa/hc/rs/cem/de)",
        ),
    ] = "spgd"
    pop_size: Annotated[
        int | None,
        option(
            "--pop_size",
            type=int,
            help="种群规模 (ga/pso/cem/de 使用; 默认取算法默认值)",
        ),
    ] = None
    seed: Annotated[
        int | None, option("--seed", type=int, help="随机种子 (default: None)")
    ] = None
    show: Annotated[bool, option("--show", is_flag=True, help="显示中间图像")] = False
    target_brightness: Annotated[
        int, option("--target-brightness", help="目标最大亮度 (default: 200)")
    ] = 200
    w_uniformity: Annotated[
        float, option("--w-uniformity", help="均匀性权重 (default: 0.4)")
    ] = 0.4
    w_efficiency: Annotated[
        float, option("--w-efficiency", help="能量效率权重 (default: 0.6)")
    ] = 0.6
    w_aspect: Annotated[
        float, option("--w-aspect", help="宽高比权重 (default: 0.0)")
    ] = 0.0
    w_pbr: Annotated[
        float,
        option(
            "--w-pbr",
            help=(
                "背景抑制 (PBR) 权重 (default: 0.0=关闭, 保持既有结果可复现)。"
                "PBR = 目标框峰值 / 框外均值, 已做扣中位数+截零, "
                "避免读出噪声把比值推到 >1。定义转录自 "
                "刘卉等, 光子学报 2023, 52(6):0629002。"
            ),
        ),
    ] = 0.0
    basis: Annotated[
        str,
        option(
            "--basis",
            type=click.Choice(["freeform", "zernike"]),
            help="相位参数化: zernike(默认, 与GUI一致: radius=600 + defocus(2,0) + spherical(4,0))/freeform(自由相位, 可合成方形)",
        ),
    ] = "zernike"
    phase_grid: Annotated[
        int,
        option("--phase-grid", help="freeform 相位网格边长 (dim=grid²) (default: 24)"),
    ] = 24
    zernike_radius: Annotated[
        int,
        option(
            "--zernike-radius",
            help="Zernike 孔径半径(px), 默认 600 = SLM 面板短边一半 (与GUI一致)",
        ),
    ] = 600
    zernike_mask: Annotated[
        str | None,
        option(
            "--zernike-mask",
            type=str,
            help="0/1 binary mask for Zernike modes (comma-separated), e.g. '0,0,0,1,0,0,0,0,0,0,0,1' for defocus+spherical only. Noll 1-3 forced to 0. Overrides --basis zernike defaults.",
        ),
    ] = None
    rotation_search_deg: Annotated[
        float,
        option(
            "--rotation-search",
            help="SLM↔相机相对旋转搜索范围(度, 0~360; 0=关闭旋转校正)。>0 时旋转角作为额外 SPGD 自由度在 ±range/2 内搜索",
        ),
    ] = 0.0
    init_defocus: Annotated[
        float, option("--init-defocus", help="初始 Defocus (2,0) 系数 (default: 1.0)")
    ] = 1.0
    init_spherical: Annotated[
        float,
        option("--init-spherical", help="初始 Spherical (4,0) 系数 (default: 0.5)"),
    ] = 0.5
    init_coeffs: Annotated[
        str | None,
        option(
            "--init-coeffs",
            type=str,
            help='初始Zernike系数JSON (Noll 索引 dict 或 Noll 序数组); zernike 基只优化 Defocus(2,0)[Noll 4] 与 Spherical(4,0)[Noll 11], e.g. \'{"4":1.0,"11":0.5}\'',
        ),
    ] = None
    save_best_image: Annotated[
        bool, option("--save-best-image", is_flag=True, help="保存最优图像")
    ] = False


# ---------------------------------------------------------------------------
# 融合参数 | fused parameters — composite groups combining roles above
# ---------------------------------------------------------------------------


@dataclass
class SlmParamsPib(SlmParams):
    """Extended Santec SLM options (slm-pib family: adds phase shifting + coefficient loading)."""

    zernike_radius: Annotated[
        float,
        option(
            "--zernike_radius",
            help=f"Zernike aperture radius (pixels); default = SLM 面板短边的一半 ({DEFAULT_ZERNIKE_RADIUS} px).",
        ),
    ] = DEFAULT_ZERNIKE_RADIUS
    shift_x: Annotated[int, option("--shift_x", help="SLM phase X shift (pixels).")] = 0
    shift_y: Annotated[int, option("--shift_y", help="SLM phase Y shift (pixels).")] = 0
    load_file: Annotated[
        str | None,
        option(
            "-f",
            "--load_file",
            type=str,
            help="Path to a prior Zernike coefficient file to load.",
        ),
    ] = None
    init_c: Annotated[
        str | None,
        option(
            "--init_c",
            type=str,
            help="Initial Zernike coefficients (JSON or comma-separated).",
        ),
    ] = None


@dataclass
class CameraParamsPib(CameraParams, ObjectiveParamsPib):
    """CCD camera options — slm-pib uses a smaller default window (250 px).

    Fused group for slm-pib: combines the camera (纯硬件) role with the PIB
    objective (目标) role, so one parameter object carries the whole target
    definition (window centre/size + target-shape weights + auto-exposure).
    """

    cam_size: Annotated[
        int, option("--cam_size", help="CCD window size in pixels.")
    ] = 250
    center: Annotated[
        str | tuple[int, int] | None,
        option(
            "-c",
            "--center",
            type=click.STRING,
            help="Center: 'auto' / 'mass' / 'max' / 'shape' or 'x,y'.",
        ),
    ] = None
    auto_exposure: Annotated[
        bool,
        option(
            "--auto-exposure",
            is_flag=True,
            help="Auto-find a safe fixed exposure before optimizing (one probe pass).",
        ),
    ] = False
    auto_target_peak: Annotated[
        float,
        option(
            "--auto-target-peak", help="Target peak brightness for --auto-exposure."
        ),
    ] = 160.0
    auto_n_frames: Annotated[
        int,
        option("--auto-n-frames", help="Frames for the auto 0-order centre median."),
    ] = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_center(raw: Any) -> tuple[int, int] | str | None:
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


#: Fields that always survive serialization because they identify the run.
_IDENTITY_FIELDS = ("algorithm", "name")


def _config_value(value: Any) -> Any:
    """Recursively turn a nested parameter dataclass into a plain dict."""
    if is_dataclass(value) and not isinstance(value, type):
        return _config_group(value)
    return value


def _config_group(obj: Any) -> dict[str, Any]:
    """Serialize one dataclass level, recursing into nested parameter groups."""
    group: dict[str, Any] = {}
    for key, value in vars(obj).items():
        if value is None:
            continue
        # MISSING (not None) so a ``default_factory`` field — whose class
        # attribute is deleted — is never mistaken for "left at default".
        default = getattr(type(obj), key, MISSING)
        if value == default and key not in _IDENTITY_FIELDS:
            continue
        group[key] = _config_value(value)
    return group


def config_payload(obj: Any) -> dict[str, Any]:
    """Serialize a parameter dataclass for the JSON debug sidecar.

    Drops unset (None) fields and fields left at their dataclass default;
    the identity fields (``algorithm`` / ``name``) are always kept so the
    search family survives the round trip. Nested parameter groups (e.g.
    ``ObjectiveParamsPib.target``) are emitted as nested objects rather than
    flattened, so the sidecar mirrors the parameter structure; readers must
    therefore tolerate both shapes.
    """
    return _config_group(obj)


# ---------------------------------------------------------------------------
# nlight_dm 参数 | nlight_dm runner parameters (wf / pib / pipeline / combined)
# ---------------------------------------------------------------------------


@dataclass
class WfRunnerParams:
    """波前优化器 (wf) 的全部 CLI 参数。"""

    dir: Annotated[
        str, option("-d", "--dir", help="数据保存根目录 (default: data)")
    ] = "data"
    epochs: Annotated[
        int, option("-e", "--epochs", help="优化迭代次数 (default: 20000)")
    ] = 20_000

    early_stop_threshold: Annotated[
        float, option("-t", "--early_stop_threshold", help="早停阈值 (default: 0.0)")
    ] = 0.0
    disturbance_cn2: Annotated[
        float,
        option(
            "--disturbance-cn2",
            type=float,
            help="仿真 WFS 湍流强度 cn2 (0=不加像差, 默认 0)",
        ),
    ] = 0.0
    lr: Annotated[
        float | None,
        option(
            "--lr",
            type=float,
            help="覆盖自动学习率 (默认按硬件标定的自动调度)",
        ),
    ] = None
    delta: Annotated[
        float | None,
        option(
            "--delta",
            type=float,
            help="覆盖 SPGD 扰动幅度 (V); 自动调度按真实硬件标度, 对仿真 DM 过小",
        ),
    ] = None
    show: Annotated[
        bool,
        option(
            "--show",
            is_flag=True,
            help="显示远场光斑CCD图像和优化历史 (default: False)",
        ),
    ] = False
    dm_type: Annotated[
        str | None,
        option(
            "--dm_type",
            type=click.Choice(DM_TYPES, case_sensitive=False),
            help="变形镜类型 (default: auto-detect). 若未指定且仅一个DM在线则自动选取，否则报错.",
        ),
    ] = None


@dataclass
class PibRunnerParams:
    """轴向光束优化器 (pib) 的全部 CLI 参数。"""

    root_dir: Annotated[
        str, option("-d", "--root_dir", help="数据保存根目录 (default: data)")
    ] = "data"
    load_file: Annotated[
        str,
        option(
            "-f",
            "--load_file",
            help="加载优化结果文件 (default: None), 若为'rms',则使用RMS优化结果初始化",
        ),
    ] = "rms"
    cam_id: Annotated[
        str,
        option("--cam_id", help="远场光斑CCD设备ID (default: Far_Cam_ID/0)"),
    ] = cast(str, lambda: os.environ.get("FAR_CAM_ID", "0"))
    cam_type: Annotated[
        str,
        option(
            "--cam_type",
            type=click.Choice(["miicam", "daheng", "sim"]),
            help="CCD 相机后端 (sim = 2f-Fourier 数值仿真, 需配合 --dm_type sim)。",
        ),
    ] = "miicam"
    center: Annotated[
        str | tuple[float, float] | None,
        option(
            "-c",
            "--center",
            callback=parse_tuple,
            help="场光斑CCD中心位置 (example: 665,403)",
        ),
    ] = "mass"
    exposure_time_ms: Annotated[
        int,
        option(
            "-t",
            "--exposure_time_ms",
            help=(
                "远场光斑CCD曝光时间 (毫秒). 0(默认)=不固定, 由设备保持/自动曝光; "
                "Daheng 上 0 会被驱动钳到设备最小值(~0.02ms)"
            ),
        ),
    ] = 0.0
    epochs: Annotated[
        int, option("-e", "--epochs", help="优化迭代次数 (default: 4000)")
    ] = 4_000
    r_bucket: Annotated[
        int,
        option(
            "-r",
            "--r_bucket",
            help="半径桶大小 (default: 0,环围半径)。若设置为0,则根据功率半径自动调整。",
        ),
    ] = 0
    delta: Annotated[float, option("--delta", help="优化步长 (default: 2)")] = 2.0
    lr: Annotated[
        float,
        option("--lr", help="优化学习率 (default: 0.0,表示基于环围半径动态学习率衰减)"),
    ] = 0.0
    weight_decay: Annotated[
        float, option("--weight_decay", help="权重衰减 (default: 0.0)")
    ] = 0.0
    optimizer_type: Annotated[
        str,
        option(
            "--optimizer_type",
            type=click.Choice(
                ["adam", "adamw", "adamod", "sgd", "muno", "munow"],
                case_sensitive=False,
            ),
            show_default=True,
            help="梯度阶段使用的优化器类型",
        ),
    ] = "adamod"
    shrink_iter: Annotated[
        int,
        option(
            "--shrink_iter",
            help="优化迭代次数后收缩半径桶和步长 (default: 200)。若设置为0，则不进行收缩。",
        ),
    ] = 200
    shrink_ratio: Annotated[
        float, option("--shrink_ratio", help="收缩半径桶和步长比例 (default: 0.8)")
    ] = 0.8
    enable_adaptive_search: Annotated[
        bool,
        option(
            "--enable_adaptive_search",
            is_flag=True,
            help="启用局部最优后的自适应邻域搜索",
        ),
    ] = False
    search_interval: Annotated[
        int, option("--search_interval", show_default=True, help="邻域搜索触发间隔")
    ] = 120
    search_warmup: Annotated[
        int,
        option("--search_warmup", show_default=True, help="邻域搜索启动前的最小迭代数"),
    ] = 200
    search_patience: Annotated[
        int,
        option(
            "--search_patience",
            show_default=True,
            help="最佳 PIB 无提升时触发搜索的等待轮数",
        ),
    ] = 100
    search_samples: Annotated[
        int,
        option(
            "--search_samples", show_default=True, help="每次邻域搜索评估的候选解数量"
        ),
    ] = 8
    search_radius: Annotated[
        float | None,
        option(
            "--search_radius",
            type=float,
            help="邻域搜索初始半径，默认跟随 delta 自适应",
        ),
    ] = None
    tabu_memory_size: Annotated[
        int, option("--tabu_memory_size", show_default=True, help="禁忌记忆表容量")
    ] = 128
    cam_size: Annotated[
        int, option("-s", "--cam_size", help="相机开窗大小 (default: 200*200)")
    ] = 200
    target_max_brightness: Annotated[
        int,
        option(
            "-b",
            "--target_max_brightness",
            help="目标最大亮度值 (default: 90), 若为0则不自动调整曝光时间",
        ),
    ] = 90
    objective: Annotated[
        str,
        option(
            "-o",
            "--objective",
            type=click.Choice(["pib", "radiu", "avg_radiu"]),
            show_default=True,
            help="优化目标函数: pib(最大化PIB), radiu(最小化半径), avg_radiu(最大化平均半径)",
        ),
    ] = "pib"
    show: Annotated[
        bool,
        option(
            "--show",
            is_flag=True,
            help="显示远场光斑CCD图像和优化历史 (default: False)",
        ),
    ] = False
    dm_type: Annotated[
        str | None,
        option(
            "--dm_type",
            type=click.Choice(DM_TYPES, case_sensitive=False),
            help="变形镜类型 (default: auto-detect). 若未指定且仅一个DM在线则自动选取，否则报错.",
        ),
    ] = None
    debug_flag: Annotated[
        bool | None,
        option(
            "--debug",
            is_flag=True,
            help="启用调试模式: 保存 pkl/json 与汇总图 (初始/最优光斑, 目标曲线, 最优电压)",
        ),
    ] = None


@dataclass
class PipelineRunnerParams:
    """串行优化器 (pipeline) 的全部 CLI 参数。"""

    dir: Annotated[
        str, option("-d", "--dir", help="数据保存根目录 (default: data)")
    ] = "data"
    load_file: Annotated[
        str | None,
        option("-f", "--load_file", help="加载优化结果文件 (default: None)"),
    ] = None
    epochs: Annotated[
        int, option("-e", "--epochs", help="优化迭代次数 (default: 8000)")
    ] = 8_000
    wf_epochs: Annotated[
        int, option("-E", "--wf_epochs", help="WF优化迭代次数 (default: 8000)")
    ] = 8_000
    cam_id: Annotated[
        str,
        option("--cam_id", help="远场光斑CCD设备ID (default: Far_Cam_ID/0)"),
    ] = cast(str, lambda: os.environ.get("Far_Cam_ID", 0))
    exposure_time_ms: Annotated[
        int,
        option(
            "-t",
            "--exposure_time_ms",
            help="远场光斑CCD曝光时间 (毫秒) (default: 0，自动选取曝光)",
        ),
    ] = 0
    cam_size: Annotated[
        int, option("-s", "--cam_size", help="相机开窗大小 (default: 160)")
    ] = 160
    rms_threshold: Annotated[
        float, option("--rms_threshold", help="RMS阈值 (default: 0.12)")
    ] = 0.12
    dm_unit_mask: Annotated[
        str,
        option(
            "-u",
            "--dm_unit_mask",
            type=click.Choice(["all", "inner", "outer"]),
            help="DM单元掩码 (default: all)",
        ),
    ] = "all"
    dm_type: Annotated[
        str | None,
        option(
            "--dm_type",
            type=click.Choice(DM_TYPES, case_sensitive=False),
            help="变形镜类型 (default: auto-detect). 若未指定且仅一个DM在线则自动选取，否则报错.",
        ),
    ] = None


@dataclass
class CombinedRunnerParams:
    """AdaMOD 综合PIB优化器 (combined) 的全部 CLI 参数。"""

    root_dir: Annotated[
        str, option("-d", "--root_dir", help="数据保存根目录 (default: data)")
    ] = "data"
    load_file: Annotated[
        str | None,
        option("-f", "--load_file", help="加载初始电压文件 (default: None)"),
    ] = None
    epochs: Annotated[
        int, option("-e", "--epochs", help="优化迭代次数 (default: 4000)")
    ] = 4_000
    r_bucket: Annotated[
        int,
        option("-r", "--r_bucket", help="半径桶大小 (default: 0, 环围半径自动调整)"),
    ] = 0
    delta: Annotated[float, option("--delta", help="优化步长 (default: 1)")] = 1.0
    lr: Annotated[
        float, option("--lr", help="优化学习率 (default: 0.0, 动态学习率衰减)")
    ] = 0.0
    shrink_iter: Annotated[
        int, option("--shrink_iter", help="收缩半径桶的迭代间隔 (default: 0, 不收缩)")
    ] = 0
    shrink_ratio: Annotated[
        float, option("--shrink_ratio", help="收缩半径桶比例 (default: 0.9)")
    ] = 0.9
    target_max_brightness: Annotated[
        int,
        option("-b", "--target_max_brightness", help="目标最大亮度值 (default: 40)"),
    ] = 40
    show: Annotated[
        bool,
        option(
            "--show",
            is_flag=True,
            help="显示远场光斑CCD图像和优化历史 (default: False)",
        ),
    ] = False
    dm_type: Annotated[
        str | None,
        option(
            "--dm_type",
            type=click.Choice(DM_TYPES, case_sensitive=False),
            help="变形镜类型 (default: auto-detect)",
        ),
    ] = None
    debug_flag: Annotated[
        bool | None,
        option(
            "--debug",
            is_flag=True,
            help="启用调试模式: 保存 pkl/json 与汇总图 (初始/最优光斑, PIB 曲线, 最优电压)",
        ),
    ] = None


@dataclass
class DmMatrixRunnerParams:
    """DM响应矩阵标定 (dm-matrix) 的全部 CLI 参数。"""

    disturb_voltage: Annotated[
        float, option("--voltage", help="扰动电压 (0=自动优化, 默认: 50.0)")
    ] = 50.0
    n_averages: Annotated[
        int, option("--n-averages", help="每次WFS读取次数 M (默认: 20)")
    ] = 20
    n_cycles: Annotated[
        int, option("--n-cycles", help="正负交替循环次数 N (默认: 1)")
    ] = 1
    wait_time: Annotated[
        float, option("--wait", help="电压施加后等待时间 (秒, 默认: 0.1)")
    ] = 0.1
    output_path: Annotated[
        str, option("--output", help="输出文件路径 (默认: data/dm_response_matrix)")
    ] = "data/dm_response_matrix"
    dm_unit_mask_str: Annotated[
        str | None,
        option(
            "--dm-unit-mask",
            help="DM单元掩码 (逗号分隔的0/1列表, 默认: 全部有效, actuator 0禁用)",
        ),
    ] = None
    compute_inverses: Annotated[
        bool, option("--no-inverses", flag_value=False, help="不计算逆矩阵")
    ] = True
    cancel_tile: Annotated[
        bool, option("--cancel-tile", is_flag=True, help="测量时去除WFS的tip/tilt")
    ] = False
    auto_optimize_voltage: Annotated[
        bool,
        option(
            "--auto-optimize/--no-auto-optimize",
            help="自动优化每路扰动电压 (voltage=0时, 默认开启)",
        ),
    ] = True
    optimize_n_avg: Annotated[
        int, option("--optimize-n-avg", help="电压优化时的WFS读取次数 (默认: 10)")
    ] = 10
    display: Annotated[
        bool,
        option("--display/--no-display", help="显示实时pygame显示 (暂未实现)"),
    ] = False
    debug: Annotated[
        bool | None,
        option("--debug", is_flag=True, help="启用调试模式 (保存原始测量数据)"),
    ] = None
    dm_type: Annotated[
        str | None,
        option(
            "--dm_type",
            type=click.Choice(DM_TYPES_PRE_ASYN_MICRO, case_sensitive=False),
            help="变形镜类型 (default: auto-detect). 若未指定且仅一个DM在线则自动选取，否则报错.",
        ),
    ] = None
    mode: Annotated[
        str,
        option(
            "--mode",
            type=click.Choice(["sequential", "hadamard"]),
            help="校准模式: sequential=逐单元推拉; hadamard=哈达玛模式同时推拉 (测量次数更少, 所有单元同时扰动)",
        ),
    ] = "sequential"
    hadamard_order: Annotated[
        int | None,
        option(
            "--hadamard-order",
            help="哈达玛矩阵阶数 (mode=hadamard时使用); None=自动 (>=有效单元数的最小2的幂). mode=sequential时忽略",
        ),
    ] = None


@dataclass
class HadamardMatrixRunnerParams:
    """Hadamard响应矩阵校准 (hadamard-matrix) 的全部 CLI 参数。"""

    mode_order: Annotated[
        int, option("--mode-order", help="Hadamard矩阵阶数 (2的幂次, 默认8)")
    ] = 8
    magnitude: Annotated[float, option("--magnitude", help="扰动幅度 (波长)")] = 0.5
    n_averages: Annotated[int, option("--n-averages", help="每次WFS读取次数 (M)")] = 10
    n_cycles: Annotated[int, option("--n-cycles", help="正负交替循环次数 (N)")] = 1
    wait_time: Annotated[float, option("--wait", help="等待时间 (秒)")] = 0.1
    output_path: Annotated[str, option("--output", help="输出文件路径")] = (
        "data/hadamard_response_matrix"
    )
    resolution: Annotated[str, option("--resolution", help="SLM分辨率 (宽,高)")] = (
        "1920,1080"
    )
    wavelength: Annotated[int, option("--wavelength", help="工作波长 (nm)")] = 1064

    compute_inverses: Annotated[
        bool, option("--no-inverses", flag_value=False, help="不计算逆矩阵")
    ] = True
    display: Annotated[
        bool, option("--display/--no-display", help="显示实时pygame显示")
    ] = False
    debug: Annotated[
        bool | None, option("--debug", is_flag=True, help="启用调试模式")
    ] = None


# ---------------------------------------------------------------------------
# --- full-voltage / alt-voltage (micro_drive) ---
# ---------------------------------------------------------------------------


@dataclass
class FullVoltageRunnerParams:
    """全量交替电压下发 (full-voltage) 的全部 CLI 参数。"""

    ips_str: Annotated[
        str | None,
        option(
            "--ips",
            help="Controller IPs, comma-separated (default: 192.168.0.101~126, 全部 26 台)",
        ),
    ] = None
    alt_voltage: Annotated[
        float,
        option(
            "--voltage", required=True, help="Voltage for ALL units (V, [-20, 120])"
        ),
    ] = field(default_factory=lambda: 0.0)
    alt_freq: Annotated[
        float, option("--freq", help="Alternation frequency (Hz, default: 1.0)")
    ] = 1.0
    alt_duration: Annotated[
        float,
        option("--duration", help="Duration in seconds (0=until Ctrl+C, default: 0)"),
    ] = 0.0
    relay_on: Annotated[
        bool,
        option(
            "--relay-on/--no-relay-on",
            help="Auto relay on before starting (default: True)",
        ),
    ] = True
    home_voltage: Annotated[
        float, option("--home-voltage", help="Home voltage on shutdown (default: 0.0)")
    ] = 0.0
    timeout: Annotated[
        float,
        option("--timeout", help="Controller connect/send timeout (s, default: 10.0)"),
    ] = 10.0
    debug: Annotated[
        bool, option("--debug", is_flag=True, help="Enable debug logging")
    ] = False


@dataclass
class AltVoltageRunnerParams:
    """交替电压下发 (alt-voltage) 的全部 CLI 参数。"""

    ip: Annotated[
        str,
        option(
            "--ip", required=True, help="Controller IP address (e.g., 192.168.0.101)"
        ),
    ] = field(default_factory=lambda: "")
    port: Annotated[
        int | None, option("--port", help="TCP port (default: 10000 + last IP octet)")
    ] = None
    alt_voltage: Annotated[
        float,
        option("--voltage", required=True, help="Input voltage for alternation (V)"),
    ] = field(default_factory=lambda: 0.0)
    alt_freq: Annotated[
        float, option("--freq", help="Alternation frequency (Hz, default: 1.0)")
    ] = 1.0
    alt_duration: Annotated[
        float,
        option("--duration", help="Duration in seconds (0=until Ctrl+C, default: 0)"),
    ] = 0.0
    channel_str: Annotated[
        str | None,
        option(
            "--channels",
            help="Channels to alternate (comma-separated, e.g. 0,1,2 or 'all' for all 50)",
        ),
    ] = None
    ping_first: Annotated[
        bool,
        option(
            "--ping-first/--no-ping-first",
            help="Ping test before connecting (default: True)",
        ),
    ] = True
    relay_on: Annotated[
        bool,
        option(
            "--relay-on/--no-relay-on",
            help="Auto relay on before starting (default: True)",
        ),
    ] = True
    debug: Annotated[
        bool, option("--debug", is_flag=True, help="Enable debug logging")
    ] = False
    adc_enabled: Annotated[
        bool,
        option(
            "--adc-enabled/--no-adc-enabled",
            help="Enable ADC voltage acquisition (default: False)",
        ),
    ] = False
    adc_device: Annotated[
        str, option("--adc-device", help="NI DAQ device name (default: Dev1)")
    ] = "Dev1"
    adc_channel: Annotated[
        str, option("--adc-channel", help="Analog input channel (default: ai0)")
    ] = "ai0"
    adc_sample_rate: Annotated[
        int, option("--adc-sample-rate", help="ADC sample rate in Hz (default: 5000)")
    ] = 5000
    adc_samples_per_read: Annotated[
        int, option("--adc-samples-per-read", help="Samples per ADC read (default: 10)")
    ] = 10


@dataclass
class SlmGsRefineParams:
    """SLM GS warm-start + sensorless freeform refinement (slm-gs-refine)."""

    # --- objective ---------------------------------------------------------
    target_side: Annotated[
        int,
        option(
            "--target-side",
            help="Target square side in camera pixels (0 = derive from pupil image).",
        ),
    ] = 0
    side_factor: Annotated[
        float, option("--side-factor", help="Multiplier on the target side.")
    ] = 1.0
    w_pib: Annotated[
        float, option("--w_pib", help="Composite weight on power-in-bucket.")
    ] = 0.5
    w_unif: Annotated[
        float, option("--w-unif", help="Composite weight on 1/(1+CV).")
    ] = 0.5

    # --- freeform parameterisation ----------------------------------------
    phase_grid: Annotated[
        int,
        option(
            "--phase-grid",
            help="Coarse freeform phase grid edge; DOF = grid**2 (default: 24).",
        ),
    ] = 24
    delta: Annotated[
        float, option("--delta", help="SPGD perturbation amplitude in radians.")
    ] = 0.35
    lr: Annotated[
        float, option("--lr", help="Learning rate (0 = auto = 0.15*delta).")
    ] = 0.0
    optimizer_type: Annotated[
        str,
        option(
            "--optimizer",
            type=click.Choice(["adam", "adamw", "adamod", "sgd"]),
            help="Gradient optimizer applied to the SPGD estimate.",
        ),
    ] = "adam"
    lr_schedule: Annotated[
        str,
        option(
            "--lr-schedule",
            type=click.Choice(["static", "cosine"]),
            help="Learning-rate schedule.",
        ),
    ] = "cosine"

    # --- GS warm start -----------------------------------------------------
    gs_iters: Annotated[
        int, option("--gs-iters", help="Gerchberg-Saxton iterations (0 = skip GS).")
    ] = 200
    gs_warm_start: Annotated[
        bool,
        option(
            "--gs-warm-start/--no-gs-warm-start",
            help="Compute a GS phase and keep it only if it beats flat.",
        ),
    ] = True

    # --- bench model -------------------------------------------------------
    panel_pixel_um: Annotated[
        float, option("--panel-pixel-um", help="SLM pixel pitch in um.")
    ] = 8.0
    camera_pixel_um: Annotated[
        float,
        option(
            "--camera-pixel-um",
            help=(
                "CCD pixel pitch in um -- the measurement ANCHOR for the bench "
                "model, because it is a per-camera datasheet constant (2.2um "
                "for the Daheng MER2-507-23GM). The measured focal scale only "
                "fixes the ratio f/p_cam, so one anchor is unavoidable; pass it "
                "explicitly for any other camera."
            ),
        ),
    ] = _DEFAULT_CAMERA_PIXEL_UM
    beam_radius_px: Annotated[
        float,
        option(
            "--beam-radius-px",
            help="Illuminated pupil radius on the panel, in panel pixels.",
        ),
    ] = 450.0
    focal_length_m: Annotated[
        float,
        option(
            "--focal-length-m",
            help=(
                "2f lens focal length in metres. 0 (the default) DERIVES it from "
                "the measured focal scale and --camera-pixel-um; pass it "
                "explicitly to pin a specific lens."
            ),
        ),
    ] = 0.0
    far_field_padding: Annotated[
        int,
        option(
            "--far-field-padding",
            help="GS far-field zero-padding factor (cost is quadratic).",
        ),
    ] = 3
    gs_target_side_px: Annotated[
        int,
        option(
            "--gs-target-side-px",
            help="Override the bench-space target side (0 = derive).",
        ),
    ] = 0

    # --- hardware ----------------------------------------------------------
    cam_type: Annotated[
        str,
        option(
            "--cam_type",
            type=click.Choice(["daheng", "miicam", "sim"]),
            help="CCD backend (sim = 2f-Fourier numerical simulation, no hardware).",
        ),
    ] = "daheng"
    cam_id: Annotated[int, option("--cam-id", help="CCD device ID.")] = 0
    exposure_time_ms: Annotated[
        float, option("--exposure_time_ms", help="CCD exposure in ms (0 = default).")
    ] = 0.0
    cam_size: Annotated[
        int, option("--cam_size", help="CCD window size in pixels.")
    ] = 300
    slm_number: Annotated[
        int, option("--slm_number", help="Santec SLM device number (1-8).")
    ] = 1
    slm_wavelength: Annotated[
        int,
        option(
            "--slm_wavelength",
            help=(
                "SLM operating wavelength (nm); 0 (the default) asks the "
                "device which wavelength it is programmed for."
            ),
        ),
    ] = 0

    n_eval_frames: Annotated[
        int, option("--n-eval-frames", help="Frames averaged per measurement.")
    ] = 4
    settle_wait_s: Annotated[
        float, option("--settle-wait-s", help="Initial LCOS settle wait (s).")
    ] = 0.5
    settle_tol: Annotated[
        float, option("--settle-tol", help="Relative stability tolerance.")
    ] = 0.02
    settle_max_wait_s: Annotated[
        float, option("--settle-max-wait-s", help="Cap on settle wait (s).")
    ] = 6.0

    # --- control -----------------------------------------------------------
    epochs: Annotated[
        int, option("-e", "--epochs", help="SPGD refinement iterations.")
    ] = 400
    early_stop_score: Annotated[
        float,
        option("--early-stop-score", help="Stop once the composite reaches this."),
    ] = 0.0
    seed: Annotated[
        int | None, option("--seed", help="RNG seed for reproducible perturbations.")
    ] = None
    save_best_image: Annotated[
        bool, option("--save-best-image", help="Save the best far-field PNG.")
    ] = True
