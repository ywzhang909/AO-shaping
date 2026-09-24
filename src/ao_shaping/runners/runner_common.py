"""Shared helpers for AO-Shaping runner scripts.

A single place for the patterns that appear verbatim in 3+ runner files,
keeping each runner focused on what makes it unique while pulling
boilerplate out of every file.

Concrete file/save-path/visualisation helpers have been relocated to their
canonical homes:

* :func:`build_debug_save_paths` → :mod:`ao_shaping.utils.io.file`
* :func:`save_optimization_debug_artifacts`,
  :func:`_save_data_mode_debug_artifacts`,
  :func:`_infer_objective_key` → :mod:`ao_shaping.utils.io.file`
* :func:`make_debug_wavefront_ax_plots`,
  :func:`save_recorder_artifacts` → :mod:`ao_shaping.utils.image.display`
* :func:`resolve_dm` → :mod:`ao_shaping.drivers.dm._registry`

This module re-exports them for backward compatibility, and keeps the
parameter dataclasses + click option decorators that are genuinely
runner-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import click

# Re-exports (backward compatibility for any runner that still imports here)
from ao_shaping.drivers.dm._registry import (  # noqa: F401
    create_dm,
    list_reachable_dm_types,
    resolve_dm,
)
from ao_shaping.utils.io.cli_helpers import parse_tuple  # noqa: F401
from ao_shaping.utils.image.display import (  # noqa: F401
    make_debug_wavefront_ax_plots,
    save_recorder_artifacts,
)
from ao_shaping.utils.io.file import (  # noqa: F401
    build_debug_save_paths,
    save_optimization_debug_artifacts,
)


# ---------------------------------------------------------------------------
# Shared parameter dataclasses (SLM family runners)
# ---------------------------------------------------------------------------


@dataclass
class RunParams:
    """Global / run-wide options shared by subcommands."""

    dir: str = "data"
    debug: bool = False
    seed: int | None = None


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
    shift_x: int = 0
    shift_y: int = 0
    zernike_radius: float = 0.0
    load_file: Any = None
    init_c: Any = None


@dataclass
class SpgdParams:
    """SPGD (gradient) search options."""

    epochs: int = 2000
    delta: float = 0.1
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


# ---------------------------------------------------------------------------
# slm-pib-specific parameter dataclasses
# ---------------------------------------------------------------------------
# These mirror the shared dataclasses above but with slm-pib defaults
# (smaller CCD window, larger SPGD perturbation, PIB objective fields).
# They live here so slm_pib_runner can import them instead of re-defining
# the same shape locally.


@dataclass
class CameraParamsPib(CameraParams):
    """CCD camera options — slm-pib uses a smaller default window (250 px)."""

    cam_size: int = 250


@dataclass
class SpgdParamsPib(SpgdParams):
    """SPGD options — slm-pib uses a larger default perturbation (0.2 rad)."""

    delta: float = 0.2


@dataclass
class ObjectiveParamsPib:
    """Imaging objective options (shared by both slm-pib search families)."""

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
    # Initial weights of the 'rms_pib' objective (None = default 1/3 each).
    # Provided terms are kept exactly; unprovided terms share the remainder.
    w_pib_init: float | None = None
    w_rms_init: float | None = None
    w_ee_init: float | None = None


@dataclass
class ObjectiveParamsSquare:
    """Square-shaping objective options (slm-gsnet family)."""

    name: str = "square"
    target_side: int = 0
    target_mean_brightness: float = 0.0
    side_factor: float = 1.5
    target_max_brightness: int = 200
    w_uniformity: float = 0.4
    w_efficiency: float = 0.6
    w_aspect: float = 0.0


# ---------------------------------------------------------------------------
# Shared click option decorators (SLM family runners)
# ---------------------------------------------------------------------------


def run_options(fn):
    """``-d/--dir`` + ``--debug`` shared by every SLM runner subcommand."""
    fn = click.option("-d", "--dir", default="data", help="Data root directory.")(fn)
    fn = click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")(fn)
    return fn


def seed_option(fn):
    """``--seed`` random seed (reproducible only in 'sim' mode)."""
    fn = click.option(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible runs (reproducible only in 'sim' mode).",
    )(fn)
    return fn


def camera_options(fn):
    """CCD camera options: ``--cam-id``, ``--cam_type``, ``--exposure_time_ms``,
    ``--cam_size``, ``-c/--center``."""
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


def slm_options(fn):
    """Santec SLM options: ``--slm_number``, ``--slm_wavelength``, ``-n/--n_max``,
    ``--zernike_radius``."""
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
        help="Zernike aperture radius (pixels); 0 = default.",
    )(fn)
    return fn


def slm_extended_options(fn):
    """Extended SLM options: :func:`slm_options` + ``--shift_x``, ``--shift_y``,
    ``--load_file``, ``--init_c``."""
    fn = slm_options(fn)
    fn = click.option("--shift_x", type=int, default=0, help="SLM phase X shift (pixels).")(fn)
    fn = click.option("--shift_y", type=int, default=0, help="SLM phase Y shift (pixels).")(fn)
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


# ---------------------------------------------------------------------------
# Zernike WFS / SLM option decorators (rms-zernike, ga-zernike,
# greedy-zernike, slm-offset share these)
# ---------------------------------------------------------------------------


def wfs_options(fn):
    """ThorlabWFS options: ``-r/--wfs_res``, ``-p/--pupil_diameter``,
    ``-c/--pupil_center`` (parse_tuple callback), ``--exposure-time-ms``,
    ``--remove-tilt``."""
    fn = click.option(
        "-r", "--wfs_res", default="1024", help="WFS分辨率 (default: 1024)"
    )(fn)
    fn = click.option(
        "-p", "--pupil_diameter", default=2.7, help="瞳孔直径 (default: 2.7)"
    )(fn)
    fn = click.option(
        "-c",
        "--pupil_center",
        callback=parse_tuple,
        default="(0,0)",
        help="瞳孔中心坐标 (default: (0,0))",
    )(fn)
    fn = click.option(
        "--exposure-time-ms",
        default=0.0,
        type=float,
        help="WFS曝光时间 (毫秒, default: 0.0=自动曝光)",
    )(fn)
    fn = click.option(
        "--remove-tilt", is_flag=True, help="移除波前测量中的倾斜项"
    )(fn)
    return fn


def zernike_slm_options(fn):
    """Zernike SLM options: ``--wavelength``, ``--shift-x``, ``--shift-y``,
    ``--slm-number``, ``--wait-time``."""
    fn = click.option(
        "--wavelength", default=532, help="SLM波长 (nm, default: 532)"
    )(fn)
    fn = click.option("--shift-x", default=0, help="SLM X方向平移 (像素, default: 0)")(fn)
    fn = click.option("--shift-y", default=0, help="SLM Y方向平移 (像素, default: 0)")(fn)
    fn = click.option("--slm-number", default=1, help="SLM设备编号 (default: 1)")(fn)
    fn = click.option(
        "--wait-time", default=0.3, help="SLM 液晶翻转等待时间(秒, default: 0.3) "
    )(fn)
    return fn


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
