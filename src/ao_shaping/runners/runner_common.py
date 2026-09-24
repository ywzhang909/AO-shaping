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

import functools
import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Union, cast, get_args, get_origin, get_type_hints

import click

from ao_shaping.algorithm.heuristic.search import heuristic_algorithm_choices
from ao_shaping.utils.image.targets import TARGET_SHAPE_CHOICES
from ao_shaping.utils.io.cli_helpers import parse_tuple

from ao_shaping.drivers.dm import list_dm_types

# Snapshot of DM types taken BEFORE asyn_micro_dm registration below. The
# dm_matrix_runner at HEAD computed its own DM_TYPES at import time before
# asyn_micro was registered (runners/__init__.py imports dm_matrix_runner
# before full_voltage_runner), so its --dm_type choice list excluded asyn_micro.
# Preserve that exact ordering for byte-identical help output.
DM_TYPES_PRE_ASYN_MICRO = list_dm_types()

# Importing asyn_micro_dm registers the "asyn_micro" DM type (side effect).
# It is normally registered by micro_drive.full_voltage_runner, which is
# imported AFTER this module in runners/__init__.py — without this import,
# DM_TYPES below would miss asyn_micro and the --dm_type choice list would
# silently shrink from 6 to 5 entries.
import ao_shaping.drivers.dm.asyn_micro_dm  # noqa: F401

DM_TYPES = list_dm_types()


# ---------------------------------------------------------------------------
# dataclass-click machinery (B2 copy of the dataclass-click convention:
# Annotated[...] metadata + with_params collector, object delivery)
# ---------------------------------------------------------------------------


class _DelayedCall:
    """A ``click.option`` declaration captured but not yet applied."""

    __slots__ = ("callable", "args", "kwargs")

    def __init__(self, callable: object, args: tuple, kwargs: dict[str, Any]) -> None:
        self.callable = callable
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        merged = dict(self.kwargs)
        merged.update(kwargs)
        return self.callable(*self.args, *args, **merged)


class _DelayedFunction:
    """Wrap ``click.option`` so it can be invoked inside ``Annotated[...]``."""

    def __init__(self, fn: object) -> None:
        self.fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> _DelayedCall:
        return _DelayedCall(self.fn, args, kwargs)


option = _DelayedFunction(click.option)


_TYPE_INFERENCE: dict[type[Any], click.ParamType] = {
    str: click.STRING,
    int: click.INT,
    float: click.FLOAT,
    bool: click.BOOL,
    Path: click.Path(path_type=Path),
}


def _strip_optional(tp: Any) -> Any:
    if get_origin(tp) in (Union, UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) != 1:
            raise TypeError(
                f"Cannot infer click type from union {tp!r} — pass an explicit type=."
            )
        return args[0]
    return tp


def _patch_names(decl: tuple, name: str) -> tuple:
    # Prepend the field name so click maps user input back onto the field.
    # Declarations must therefore be flags-first (no duplicated first positional).
    return (name, *decl)


def _patch_click_types(name: str, field_type: Any, kwargs: dict[str, Any]) -> None:
    if (
        "type" in kwargs
        or "callback" in kwargs
        or kwargs.get("is_flag")
        or kwargs.get("multiple")
    ):
        return
    stripped = _strip_optional(field_type)
    try:
        kwargs["type"] = _TYPE_INFERENCE[stripped]
    except KeyError:
        raise TypeError(
            f"Field {name!r}: cannot infer click type from {field_type!r} — pass an explicit type=."
        ) from None


def _patch_defaults(name: str, field: object, kwargs: dict[str, Any]) -> None:
    if "default" in kwargs:
        raise TypeError(
            f"Field {name!r}: default must live on the dataclass field, not in option()."
        )
    value = field.default
    if value is not MISSING:
        kwargs["default"] = value


def _copy_delayed_call(d: _DelayedCall) -> _DelayedCall:
    return _DelayedCall(
        d.callable,
        tuple(d.args),
        {k: (list(v) if isinstance(v, list) else v) for k, v in d.kwargs.items()},
    )


def _collect_click_annotations(cls: type) -> list[tuple[Any, Any, _DelayedCall]]:
    hints = get_type_hints(cls, include_extras=True)
    collected: list[tuple[Any, Any, _DelayedCall]] = []
    for f in fields(cls):
        hint = hints.get(f.name)
        if hint is None or get_origin(hint) is not Annotated:
            continue
        meta = get_args(hint)[1:]
        delayed = next((m for m in meta if isinstance(m, _DelayedCall)), None)
        if delayed is None:
            continue
        collected.append((f, get_args(hint)[0], delayed))
    return collected


def with_params(arg_class: type, *, kw_name: str) -> Any:
    """Attach the click options of ``arg_class`` to a click command.

    Every ``Annotated[..., option(...)]`` field becomes a click option (help
    text order == dataclass declaration order). The wrapped command receives
    ``kw_name=arg_class(**provided_fields)`` — one keyword argument per
    decorator, containing a fully-populated parameter instance.
    """

    def decorator(fn: Any) -> Any:
        # Apply in REVERSED declaration order so click's cumulative
        # __click_params__ yields help in the same order the class reads.
        for fld, field_type, delayed in reversed(
            _collect_click_annotations(arg_class)
        ):
            dc = _copy_delayed_call(delayed)
            dc.args = _patch_names(delayed.args, fld.name)
            _patch_click_types(fld.name, field_type, dc.kwargs)
            _patch_defaults(fld.name, fld, dc.kwargs)
            fn = dc.callable(*dc.args, **dc.kwargs)(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            obj_kwargs = {}
            for f in fields(arg_class):
                if f.name in kwargs:
                    obj_kwargs[f.name] = kwargs.pop(f.name)
            kwargs[kw_name] = arg_class(**obj_kwargs)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# 纯硬件参数 | pure hardware parameters — device configuration
# (CCD camera / Santec SLM / Thorlabs WFS)
# ---------------------------------------------------------------------------


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
        option("--exposure_time_ms", help="CCD exposure time in ms (0 = auto-exposure)."),
    ] = 80.0
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
        int, option("--slm_wavelength", help="SLM operating wavelength (nm).")
    ] = 1064
    n_max: Annotated[int, option("-n", "--n_max", help="Max Zernike radial order.")] = 4
    zernike_radius: Annotated[
        float, option("--zernike_radius", help="Zernike aperture radius (pixels); 0 = default.")
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
        option("--exposure-time-ms", type=float, help="WFS曝光时间 (毫秒, default: 0.0=自动曝光)"),
    ] = 0.0
    remove_tilt: Annotated[
        bool, option("--remove-tilt", is_flag=True, help="移除波前测量中的倾斜项")
    ] = False


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
        float, option("--delta", help="SPGD perturbation amplitude (rad).")
    ] = 0.1
    lr: Annotated[float, option("--lr", help="SPGD learning rate (0 = auto).")] = 0.0
    optimizer_type: Annotated[
        str,
        option(
            "--optimizer_type",
            type=click.Choice(["adam", "adamw", "adamod", "sgd", "muno", "munow"], case_sensitive=False),
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


# ---------------------------------------------------------------------------
# 目标参数 | objective parameters — imaging target & quality weights
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveParamsPib:
    """Imaging objective options (shared by both slm-pib search families)."""

    name: Annotated[
        str,
        option(
            "--objective",
            type=click.Choice(["pib", "radiu", "avg_radiu", "rmse", "shape", "roi_pib", "rms_pib"]),
            help="Optimization objective.",
        ),
    ] = "pib"
    target_max_brightness: Annotated[
        int, option("--target_max_brightness", help="Target max brightness for auto-exposure.")
    ] = 40
    r_bucket: Annotated[
        int, option("-r", "--r_bucket", help="Bucket radius (0 = auto from power radius).")
    ] = 0
    target_size: Annotated[
        float, option("--target_size", help="Target extent in camera px.")
    ] = 64.0
    target_aspect_ratio: Annotated[
        float,
        option("--target_aspect_ratio", help="Width:height ratio for a rectangular target."),
    ] = 4.0 / 3.0
    target_center_smooth: Annotated[
        int,
        option("--target_center_smooth", help="Frames averaged for the target centre estimate."),
    ] = 3
    target_shape: Annotated[
        str | None,
        option(
            "--target_shape",
            type=click.Choice(list(TARGET_SHAPE_CHOICES)),
            help="Target ROI shape (implies the 'shape' objective).",
        ),
    ] = None
    shape_schedule: Annotated[
        bool,
        option("--shape_schedule", is_flag=True, help="Use the coarse->fine shaping weight schedule."),
    ] = False
    max_roi_energy_loss: Annotated[
        float,
        option("--max_energy_loss", help="Max allowed in-ROI energy loss fraction (0 disables the guard)."),
    ] = 0.6
    w_uniformity: Annotated[
        float, option("--w_uniformity", help="Uniformity penalty weight.")
    ] = 2.0
    w_peak: Annotated[
        float, option("--w_peak", help="Peak penalty weight.")
    ] = 0.5
    w_displacement: Annotated[
        float, option("--w_displacement", help="Displacement penalty weight.")
    ] = 0.0
    log_uniformity: Annotated[
        bool,
        option("--log_uniformity", is_flag=True, help="Use log1p(u) instead of u/(1+u) for the uniformity term."),
    ] = False
    w_ema_decay: Annotated[
        float,
        option("--w_ema_decay", help="EMA decay for the adaptive PIB/RMS weights of the 'rms_pib' objective."),
    ] = 0.9
    w_floor: Annotated[
        float,
        option("--w_floor", help="Minimum weight floor per term of the 'rms_pib' objective (0..0.5)."),
    ] = 0.1
    w_temperature: Annotated[
        float,
        option("--w_temperature", help="Softmax temperature for the 'rms_pib' weight update."),
    ] = 8.0
    # Initial weights of the 'rms_pib' objective (None = default 1/3 each).
    # Provided terms are kept exactly; unprovided terms share the remainder.
    w_pib_init: Annotated[
        float | None,
        option("--w_pib_init", type=float, help="Initial PIB weight of the 'rms_pib' objective (default 1/3)."),
    ] = None
    w_rms_init: Annotated[
        float | None,
        option("--w_rms_init", type=float, help="Initial RMS (in-ROI uniformity) weight of the 'rms_pib' objective (default 1/3)."),
    ] = None
    w_ee_init: Annotated[
        float | None,
        option("--w_ee_init", type=float, help="Initial encircled-energy weight of the 'rms_pib' objective (default 1/3)."),
    ] = None


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
        int, option("--target-max-brightness", help="Target max brightness for auto-exposure.")
    ] = 200
    w_uniformity: Annotated[
        float, option("--w_uniformity", help="Uniformity (CV) weight.")
    ] = 0.4
    w_efficiency: Annotated[
        float, option("--w_efficiency", help="Encircled-energy weight.")
    ] = 0.6
    w_aspect: Annotated[
        float, option("--w_aspect", help="Aspect-ratio weight.")
    ] = 0.0


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
        option("--seed", type=int, help="Random seed for reproducible runs (reproducible only in 'sim' mode)."),
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

    dir: Annotated[str, option("-d", "--dir", help="数据保存根目录 (default: data)")] = "data"
    epochs: Annotated[
        int, option("-e", "--epochs", help="优化迭代次数 (default: 20000)")
    ] = 20000
    n_max: Annotated[int, option("-n", "--n-max", help="Zernike最大阶数 (default: 4)")] = 4
    lr: Annotated[float, option("--lr", help="学习率 (default: 0.01)")] = 0.01
    delta: Annotated[
        float, option("--delta", help="初始delta值 (default: 0.0)")
    ] = 0.0
    early_stop_threshold: Annotated[
        float, option("-t", "--early_stop_threshold", help="早停阈值 (default: 0.12)")
    ] = 0.12
    min_delta: Annotated[
        float, option("--min-delta", help="自动检测最小delta (数量级扫描, default: 0.01)")
    ] = 0.01
    max_delta: Annotated[
        float, option("--max-delta", help="自动检测最大delta (数量级扫描, default: 100.0)")
    ] = 100.0
    delta_step: Annotated[
        int, option("--delta-step", help="数量级扫描步数 (用于细粒度扫描, default: 5)")
    ] = 5
    n_directions: Annotated[
        int, option("--n-directions", help="每个delta采样次数防噪声 (default: 5)")
    ] = 5
    n_init_positions: Annotated[
        int, option("--n-init-positions", help="多起点优化：随机初始位置数量 (default: 0, 禁用)")
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
        float, option("--weight-decay", type=float, help="AdamW权重衰减 (default: 1e-2)")
    ] = 1e-2
    mini_batch: Annotated[
        int, option("--mini-batch", type=int, help="SPGD mini-batch大小 (default: 1)")
    ] = 1
    gradient_clip: Annotated[
        float, option("--gradient-clip", type=float, help="梯度裁剪阈值 (default: 0.0, 禁用)")
    ] = 0.0
    stagnation_patience: Annotated[
        int, option("--stagnation-patience", type=int, help="停滞检测轮数 (default: 30)")
    ] = 30
    stagnation_delta_boost: Annotated[
        float,
        option("--stagnation-delta-boost", type=float, help="停滞时delta倍增 (default: 1.5)"),
    ] = 1.5
    freeze_threshold: Annotated[
        float | None,
        option("--freeze-threshold", type=float, help="冻结高阶模式阈值 (default: None)"),
    ] = None
    early_stop_window: Annotated[
        int, option("--early-stop-window", type=int, help="早停滑动窗口大小 (default: 0)")
    ] = 0
    early_stop_min_epochs: Annotated[
        int, option("--early-stop-min-epochs", type=int, help="早停最小轮数 (default: 0)")
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
            type=click.Choice(list(heuristic_algorithm_choices()), case_sensitive=False),
            show_default=True,
            help="搜索算法: spgd (梯度/SPGD) 或启发式 (ga/pso/sa/hc/rs/cem/de)",
        ),
    ] = "spgd"
    pop_size: Annotated[
        int | None,
        option("--pop_size", type=int, help="种群规模 (ga/pso/cem/de 使用; 默认取算法默认值)"),
    ] = None
    debug: Annotated[
        bool,
        option("--debug", is_flag=True, help="启用调试模式: 保存 pkl/json 与汇总图"),
    ] = False


@dataclass
class GreedyZernikeParams:
    """贪婪/启发式 Zernike 优化 CLI 参数 (greedy-zernike, 单命令无子命令)。"""

    dir: Annotated[str, option("-d", "--dir", help="数据保存根目录 (default: data)")] = "data"
    epochs: Annotated[
        int, option("-e", "--epochs", help="优化迭代次数 (default: 2000)")
    ] = 2000
    n_max: Annotated[int, option("-n", "--n-max", help="Zernike最大阶数 (default: 4)")] = 4
    early_stop_threshold: Annotated[
        float, option("-t", "--early_stop_threshold", help="早停阈值 (default: 0.12)")
    ] = 0.12
    show: Annotated[
        bool, option("--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)")
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
            type=click.Choice(list(heuristic_algorithm_choices()), case_sensitive=False),
            show_default=True,
            help="搜索算法: spgd (贪婪局部搜索) 或启发式 (ga/pso/sa/hc/rs/cem/de)",
        ),
    ] = "spgd"
    pop_size: Annotated[
        int | None,
        option("--pop_size", type=int, help="种群规模 (ga/pso/cem/de 使用; 默认取算法默认值)"),
    ] = None
    debug: Annotated[
        bool,
        option("--debug", is_flag=True, help="启用调试模式: 保存 pkl/json 与汇总图"),
    ] = False


# ---------------------------------------------------------------------------
# 融合参数 | fused parameters — composite groups combining roles above
# ---------------------------------------------------------------------------


@dataclass
class SlmParamsPib(SlmParams):
    """Extended Santec SLM options (slm-pib family: adds phase shifting + coefficient loading)."""

    shift_x: Annotated[
        int, option("--shift_x", help="SLM phase X shift (pixels).")
    ] = 0
    shift_y: Annotated[
        int, option("--shift_y", help="SLM phase Y shift (pixels).")
    ] = 0
    load_file: Annotated[
        str | None,
        option("-f", "--load_file", type=str, help="Path to a prior Zernike coefficient file to load."),
    ] = None
    init_c: Annotated[
        str | None,
        option("--init_c", type=str, help="Initial Zernike coefficients (JSON or comma-separated)."),
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
        option("--auto-target-peak", help="Target peak brightness for --auto-exposure."),
    ] = 160.0
    auto_n_frames: Annotated[
        int, option("--auto-n-frames", help="Frames for the auto 0-order centre median.")
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


def config_payload(obj: Any) -> dict[str, Any]:
    """Serialize a parameter dataclass for the JSON debug sidecar.

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
    wfs_res: Annotated[
        str, option("-r", "--wfs_res", help="WFS分辨率 (default: 768)")
    ] = "768"
    pupil_diameter: Annotated[
        float, option("-p", "--pupil_diameter", help="瞳孔直径 (default: 2.7)")
    ] = 2.7
    pupil_center: Annotated[
        str | tuple[float, float] | None,
        option(
            "-c",
            "--pupil_center",
            callback=parse_tuple,
            help="瞳孔中心坐标 (default: (0,0))",
        ),
    ] = "(0,0)"
    early_stop_threshold: Annotated[
        float, option("-t", "--early_stop_threshold", help="早停阈值 (default: 0.0)")
    ] = 0.0
    show: Annotated[
        bool,
        option("--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)"),
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
        option("-t", "--exposure_time_ms", help="远场光斑CCD曝光时间 (毫秒) (default: 60)"),
    ] = 60
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
                ["adam", "adamw", "adamod", "sgd", "muno", "munow"], case_sensitive=False
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
        bool, option("--enable_adaptive_search", is_flag=True, help="启用局部最优后的自适应邻域搜索")
    ] = False
    search_interval: Annotated[
        int, option("--search_interval", show_default=True, help="邻域搜索触发间隔")
    ] = 120
    search_warmup: Annotated[
        int, option("--search_warmup", show_default=True, help="邻域搜索启动前的最小迭代数")
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
        option("--search_samples", show_default=True, help="每次邻域搜索评估的候选解数量"),
    ] = 8
    search_radius: Annotated[
        float | None,
        option("--search_radius", type=float, help="邻域搜索初始半径，默认跟随 delta 自适应"),
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
        option("--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)"),
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
    wfs_res: Annotated[
        str,
        option(
            "-R",
            "--wfs_res",
            type=click.Choice(["768", "512"]),
            help="WFS分辨率 (default: 768)",
        ),
    ] = "768"
    pupil_diameter: Annotated[
        float, option("-p", "--pupil_diameter", help="瞳孔直径 (default: 2.7)")
    ] = 2.7
    cam_id: Annotated[
        str,
        option("-c", "--cam_id", help="远场光斑CCD设备ID (default: Far_Cam_ID/0)"),
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
        float, option("-r", "--rms_threshold", help="RMS阈值 (default: 0.12)")
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
    cam_id: Annotated[
        str,
        option("--cam_id", help="远场光斑CCD设备ID (default: Far_CAM_ID/0)"),
    ] = cast(str, lambda: os.environ.get("FAR_CAM_ID", "0"))
    center: Annotated[
        str,
        option("-c", "--center", help="场光斑CCD中心位置 (example: 665,403)"),
    ] = "mass"
    exposure_time_ms: Annotated[
        int,
        option("-t", "--exposure_time_ms", help="远场光斑CCD曝光时间 (毫秒) (default: 80)"),
    ] = 80
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
    cam_size: Annotated[
        int, option("-s", "--cam_size", help="相机开窗大小 (default: 250)")
    ] = 250
    target_max_brightness: Annotated[
        int,
        option("-b", "--target_max_brightness", help="目标最大亮度值 (default: 40)"),
    ] = 40
    show: Annotated[
        bool,
        option("--show", is_flag=True, help="显示远场光斑CCD图像和优化历史 (default: False)"),
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
    pupil_diameter: Annotated[
        float, option("--pupil-diameter", help="瞳孔直径 (mm, 默认: 2.0)")
    ] = 2.0
    pupil_center: Annotated[
        str | tuple[float, float],
        option(
            "--pupil-center",
            callback=parse_tuple,
            help="瞳孔中心坐标 (默认: (0,0))",
        ),
    ] = "(0,0)"
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
    magnitude: Annotated[
        float, option("--magnitude", help="扰动幅度 (波长)")
    ] = 0.5
    n_averages: Annotated[
        int, option("--n-averages", help="每次WFS读取次数 (M)")
    ] = 10
    n_cycles: Annotated[
        int, option("--n-cycles", help="正负交替循环次数 (N)")
    ] = 1
    wait_time: Annotated[
        float, option("--wait", help="等待时间 (秒)")
    ] = 0.1
    output_path: Annotated[
        str, option("--output", help="输出文件路径")
    ] = "data/hadamard_response_matrix"
    resolution: Annotated[
        str, option("--resolution", help="SLM分辨率 (宽,高)")
    ] = "1920,1080"
    wavelength: Annotated[
        int, option("--wavelength", help="工作波长 (nm)")
    ] = 1064
    mla_index: Annotated[
        str,
        option(
            "--mla-index",
            type=click.Choice(["512", "540", "600", "768", "1280"]),
            help="MLA分辨率",
        ),
    ] = "512"
    exp_time: Annotated[
        float, option("--exp-time", help="曝光时间 (ms, 0=自动)")
    ] = 0.0
    auto_exposure: Annotated[
        bool, option("--auto-exposure/--no-auto-exposure", help="启用WFS自动曝光")
    ] = True
    high_speed: Annotated[
        bool, option("--high-speed", is_flag=True, help="启用高速模式")
    ] = False
    use_custom_ref: Annotated[
        bool, option("--use-custom-ref", help="使用自定义参考文件")
    ] = False
    pupil_diameter: Annotated[
        float, option("--pupil-diameter", help="瞳孔直径 (mm)")
    ] = 2.0
    pupil_center: Annotated[
        str | tuple[float, float],
        option("--pupil-center", callback=parse_tuple, help="瞳孔中心坐标"),
    ] = "(0,0)"
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
        option("--ips", help="Controller IPs, comma-separated (default: 192.168.0.101~126, 全部 26 台)"),
    ] = None
    alt_voltage: Annotated[
        float,
        option("--voltage", required=True, help="Voltage for ALL units (V, [-20, 120])"),
    ] = field(default_factory=lambda: 0.0)
    alt_freq: Annotated[
        float, option("--freq", help="Alternation frequency (Hz, default: 1.0)")
    ] = 1.0
    alt_duration: Annotated[
        float, option("--duration", help="Duration in seconds (0=until Ctrl+C, default: 0)")
    ] = 0.0
    relay_on: Annotated[
        bool, option("--relay-on/--no-relay-on", help="Auto relay on before starting (default: True)")
    ] = True
    home_voltage: Annotated[
        float, option("--home-voltage", help="Home voltage on shutdown (default: 0.0)")
    ] = 0.0
    timeout: Annotated[
        float, option("--timeout", help="Controller connect/send timeout (s, default: 10.0)")
    ] = 10.0
    debug: Annotated[
        bool, option("--debug", is_flag=True, help="Enable debug logging")
    ] = False


@dataclass
class AltVoltageRunnerParams:
    """交替电压下发 (alt-voltage) 的全部 CLI 参数。"""

    ip: Annotated[
        str,
        option("--ip", required=True, help="Controller IP address (e.g., 192.168.0.101)"),
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
        float, option("--duration", help="Duration in seconds (0=until Ctrl+C, default: 0)")
    ] = 0.0
    channel_str: Annotated[
        str | None,
        option("--channels", help="Channels to alternate (comma-separated, e.g. 0,1,2 or 'all' for all 50)"),
    ] = None
    ping_first: Annotated[
        bool, option("--ping-first/--no-ping-first", help="Ping test before connecting (default: True)")
    ] = True
    relay_on: Annotated[
        bool, option("--relay-on/--no-relay-on", help="Auto relay on before starting (default: True)")
    ] = True
    debug: Annotated[
        bool, option("--debug", is_flag=True, help="Enable debug logging")
    ] = False
    adc_enabled: Annotated[
        bool, option("--adc-enabled/--no-adc-enabled", help="Enable ADC voltage acquisition (default: False)")
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