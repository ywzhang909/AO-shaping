#!/usr/bin/env python
"""GS-refine -> Zernike-order sweep -> SPGD uniformity experiment driver.

Runs a three-stage experiment that asks: *how much of the GS-refine freeform
phase survives truncation to a low-order Zernike basis, and can SPGD on those
coefficients recover the uniformity?*

Stage A - one GS-refine optimisation (``optimize_slm_gs_refine``) produces a
    freeform best phase and a flat baseline on the same bench.
Stage B - the best phase is least-squares fitted into Noll-order Zernike
    coefficients at several truncation orders (``fit_best_phase``).
Stage C - each coefficient vector seeds an SPGD run on the Zernike basis
    (``optimize_slm_zernike_pib``, objective ``rms_pib`` + square target).

Every run's best frame is re-scored offline on ONE frozen square mask so the
orders are comparable. The driver is hardware-ready (``--cam-type daheng`` /
``miicam``) and runs end-to-end with zero hardware (``--cam-type sim``).

The target box is probed from a flat far field and clamped to the camera
window, which sidesteps the known ``cam_size`` default bug where a target
larger than the window makes ``square_mask`` clip to the whole frame and pins
``pib`` at exactly 1.0.

Split into segments
-------------------

One call doing Stage A plus all eight orders takes ~55 min on the bench, which
does not fit in a single 10-minute slot. So the driver can be **split into
independent calls**, each recording its own segment:

1. **Stage A, once** -- ``--stage-a-only`` runs Stage A, writes
   ``<group>/stage_a.npz`` (best phase, flat frame, best frame) and
   ``<group>/stage_a.json`` (geometry, epochs actually run, best score, the
   config), then exits.
2. **One order per call** -- ``--stage-a-in <group>/stage_a.npz`` reads that
   state back, skips the flat probe and Stage A, and runs Stage B/C for the
   orders named by ``--orders``.
3. **Stage A in chunks** -- ``--stage-a-only --stage-a-in <prev chunk>`` is
   *both* at once: it reads the previous chunk's end state and continues the
   same optimisation for one more chunk. ``--epochs`` stays the **total** number
   of refinement epochs (the learning-rate schedule's horizon), ``--chunk-epochs``
   says how many this chunk runs (capped by the epochs still left), and
   ``--time-budget-s`` caps the chunk's wall clock. Without this, a 250-epoch
   Stage A needs ~50 min on the bench, which cannot fit in a 10-minute slot.

Every call writes only its own segment: ``summary.json`` lists just the orders
processed by that call. Give each call its own ``--group`` so the artefacts of
the eight orders live side by side under one ``--out`` root.

Usage::

    python scripts/slm/run_gs_refine_zernike_sweep.py --cam-type sim \\
        --cam-size 256 --epochs 4 --gs-iters 20 --spgd-epochs 4 --n-max 4 6

    # Segment 1: Stage A only (writes stage_a.npz / stage_a.json)
    python scripts/slm/run_gs_refine_zernike_sweep.py --cam-type daheng \\
        --defocus-focal-mm 1597.4 --stage-a-only --time-budget-s 540 \\
        --group zernike_stage_a --out data/debug/hw_zernike_sweep_20261021

    # Segment 2: one order, reading the Stage A result back
    python scripts/slm/run_gs_refine_zernike_sweep.py --cam-type daheng \\
        --defocus-focal-mm 1597.4 \\
        --stage-a-in data/debug/hw_zernike_sweep_20261021/zernike_stage_a/stage_a.npz \\
        --orders 4 --group zernike_order_nmax4 \\
        --out data/debug/hw_zernike_sweep_20261021

    # Hardware run: the 2f bench camera is not on the 2f plane, so the pupil
    # Fresnel focus correction has to be applied explicitly.
    python scripts/slm/run_gs_refine_zernike_sweep.py --cam-type daheng \\
        --defocus-focal-mm 1597.4

Exit codes
----------

0 = the sweep ran; 2 = bad arguments; 3 = **the optical path is blocked** (the
flat far field carries no localisable spot, so every Zernike order would be
measured against noise -- see :class:`PathBlockedError`).

One variable group per invocation
---------------------------------

This driver probes ONE group of variables per invocation: the Zernike
truncation order ``n_max``. The group's identity is ``--group`` (default
``main``), its fixed parameters are ``--group-meta KEY=VALUE`` (repeatable,
recorded verbatim), and the artefacts land in ``<out>/<group>/``. The root
directory keeps a ``manifest.json`` indexing every group recorded under it, so
several groups of one sweep share a root and stay separable.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "libs") not in sys.path:
    sys.path.insert(0, str(ROOT / "libs"))

from ao_shaping.algorithm.signal_processing.square_shaping import (
    power_in_bucket,
    uniformity_cv,
)
from ao_shaping.drivers.slm.factory import DEFAULT_WAVELENGTH_NM
from ao_shaping.gui.utils.phase_overlay import PhaseOverlay
from ao_shaping.optimizer.wfless.gs_refine_zernike_fit import (
    fit_best_phase,
    panel_phase_from_coefficients,
)
from ao_shaping.optimizer.wfless._measure import locate_zero_order
from ao_shaping.optimizer.wfless.slm_gs_refine import (
    SlmGsRefineConfig,
    optimize_slm_gs_refine,
)
from ao_shaping.optimizer.wfless.slm_square_shaping import (
    square_aspect_ratio,
    square_encircled_energy,
    square_quality_score,
)
from ao_shaping.optimizer.wfless.slm_zernike_pib import (
    CAM_WINDOW_TARGET_MARGIN,
    TARGET_BOX_WAIST_FACTOR,
    ZERNIKE_CLIP,
    SlmZernikePibConfig,
    optimize_slm_zernike_pib,
)
from ao_shaping.runners.runner_common import (
    CameraParamsPib,
    ObjectiveTarget,
    SlmParamsPib,
    patch_sim_pib_shaping,
)
from ao_shaping.tools.kernels.bench_kernels import fresnel_panel
from ao_shaping.utils.cli.exposure import DEFAULT_EXPOSURE_TOKEN
from ao_shaping.utils.image.beam_metrics import zero_order_center
from ao_shaping.utils.image.frame_analysis import assess_spot
from ao_shaping.utils.image.target.metrics import spot_waist_sigma
from ao_shaping.utils.image.targets import square_mask
from ao_shaping.utils.slm.bench_constants import (
    BEAM_CENTER_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    SLM_PITCH_M,
)
from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms
from scripts._common.group_manifest import (
    group_output_dir,
    group_record,
    parse_group_meta,
    update_manifest,
)
from ao_shaping.utils.io.recorder_keys import (
    RECORD_EPOCH,
    RECORD_IMAGE,
)

#: 这一组扫的变量名 —— 写进分组清单的 ``variable`` 字段。
#: 泽尼克这一组的自变量是截断阶数 n_max; 离焦那一组扫的是 ``dz_mm``。
GROUP_VARIABLE = "n_max"

MIN_TARGET_SIDE = 4.0
"""Floor for the shared target box, in camera pixels."""

DEFAULT_SIM_FAR_FIELD_PADDING = 16
"""Zero-padding factor for the simulated far-field FFT (``--cam-type sim``).

The library default is ``4`` (``slm_pib_sim.FAR_FIELD_PADDING``), which is tuned
for point-PIB work where a ~2 px diffraction-limited spot is adequate. This
experiment instead needs a *square target box that is resolvable in camera
pixels*: the box is ``TARGET_BOX_WAIST_FACTOR * w0``, so a ~1.7 px spot (padding
4) clamps to the :data:`MIN_TARGET_SIDE` floor and the whole sweep degenerates
(the objective diverges to ~-999, SPGD cannot improve, and every order reports
``after == before``). The far-field spot size in pixels is
``N * padding / (pi * beam_w0)``, so it is scale-invariant in the panel size.

The FFT cost scales as ``padding**2``, so the box size and the runtime trade
off against each other. Measured on the 1200x1920 panel (``beam_w0=400``,
``far_field_window=1024``, ``cam_size=256``):

    padding   spot w0   target side   ROI px   one ``far_field`` call
         4       1.72            3         9     0.71 s  (clamps to floor)
         8       3.44            7        49     2.23 s  (below the default)
        16       6.84           14       196     7.88 s  <-- this default
        24      10.29           21       441    16.93 s
        32      13.70           27       729    32.21 s

Padding 16 is the smallest value whose ROI (~196 pixels) is large enough for a
stable ``uniformity_cv`` under shot noise, while still costing ~3.5x padding 8.
Raise ``--sim-far-field-padding`` for a larger box at quadratic cost, or lower
it for a faster, noisier run.
"""

DEFAULT_N_MAX: tuple[int, ...] = (4, 6, 8, 11, 15)
"""Default truncation orders (15 / 28 / 45 / 78 / 136 Noll modes)."""

OBJECTIVE = "rms_pib"
"""Stage-C objective; the square target shape selects the ROI."""


# ---------------------------------------------------------------------------
# Pure helpers (importable without touching hardware)
# ---------------------------------------------------------------------------


def build_focus_overlay(
    focal_mm: float | None,
    *,
    beam_radius_px: float,
) -> PhaseOverlay | None:
    """Build the pupil Fresnel focus-correction overlay (``None`` = no overlay).

    The 2f bench camera is *not* on the 2f plane: a flat pupil phase loses
    focus as soon as the detector leaves it, so the real bench needs a
    quadratic phase on the pupil to move the focus back onto the detector
    (``ao_shaping.tools.slm.panel.focus_probe`` measured 1/f = +0.626 m^-1 on
    2026-10-20, i.e. f_v = 1597.4 mm, on this bench -- a bench measurement,
    not a constant, so re-measure after moving the camera). The overlay is
    applied by the optimizers on the handle they open, so it reaches every
    frame of a stage without changing the phase the optimiser itself computes.

    ``None`` (the default) means "no correction", which keeps the run
    byte-identical to a run without ``--defocus-focal-mm``.

    Args:
        focal_mm: Focal length of the correction in mm (``None`` = no overlay).
        beam_radius_px: Illuminated pupil radius on the panel, in pixels.

    Returns:
        A :class:`PhaseOverlay` carrying the panel-sized raw-radian phase, or
        ``None``.
    """
    if focal_mm is None:
        return None
    phase = fresnel_panel(
        float(focal_mm),
        radius=int(round(beam_radius_px)),
        pupil_center=(int(BEAM_CENTER_PANEL[0]), int(BEAM_CENTER_PANEL[1])),
        panel_shape=(SLM_PANEL_H, SLM_PANEL_W),
        wavelength_nm=DEFAULT_WAVELENGTH_NM,
        pixel_um=SLM_PITCH_M * 1e6,
    )
    return PhaseOverlay(phase_rad=phase)


def resolve_target_side(
    w0: float,
    cam_size: int,
    override: float = 0.0,
) -> float:
    """Resolve the shared square-target side, clamped to the camera window.

    The box is ``TARGET_BOX_WAIST_FACTOR * w0`` (the flat spot's waist
    diameter), floored at :data:`MIN_TARGET_SIDE` and capped at
    ``cam_size / CAM_WINDOW_TARGET_MARGIN`` so it always fits the window. An
    explicit ``override`` bypasses the clamp but is still validated.

    Args:
        w0: Flat-field spot waist sigma in camera pixels.
        cam_size: Camera window edge in pixels.
        override: Explicit side in pixels; ``0`` (or negative) means auto.

    Returns:
        The target side in camera pixels, strictly less than ``cam_size``.

    Raises:
        ValueError: If the resolved box is degenerate or does not fit the
            window (``side >= cam_size``).
    """
    cam_size = int(cam_size)
    if cam_size < 2:
        raise ValueError(f"cam_size must be at least 2, got {cam_size}")

    if override and override > 0:
        side = float(override)
    else:
        upper = cam_size / CAM_WINDOW_TARGET_MARGIN
        side = float(
            min(max(TARGET_BOX_WAIST_FACTOR * float(w0), MIN_TARGET_SIDE), upper)
        )

    if side >= cam_size:
        raise ValueError(
            f"target side {side:.1f}px does not fit the {cam_size}px camera "
            f"window (must be < cam_size); raise --cam-size or lower "
            f"--target-side"
        )
    if side < 1.0:
        raise ValueError(f"target side {side:.1f}px is degenerate")
    return side


def sweep_plan(n_max_list: tuple[int, ...] | list[int]) -> list[dict[str, int]]:
    """Build the per-order sweep plan, preserving the requested order.

    Args:
        n_max_list: Zernike truncation orders to sweep.

    Returns:
        One ``{"n_max": int, "n_modes": int}`` dict per order, where
        ``n_modes = calc_n_zernike_terms(n_max)``.
    """
    return [
        {"n_max": int(n_max), "n_modes": int(calc_n_zernike_terms(int(n_max)))}
        for n_max in n_max_list
    ]


def parse_orders(text: str) -> tuple[int, ...]:
    """把 ``--orders 3,4,6`` 解析成阶数元组。

    Args:
        text: 逗号分隔的整数串。

    Returns:
        阶数元组 (保持书写顺序)。

    Raises:
        ValueError: 串里不是整数, 或解析后为空。
    """
    orders: list[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            orders.append(int(chunk))
        except ValueError as exc:
            raise ValueError(
                f"--orders 需要逗号分隔的整数, 收到 {chunk!r}"
            ) from exc
    if not orders:
        raise ValueError("--orders 不能为空")
    return tuple(orders)


def resolve_orders(params: SweepParams) -> tuple[int, ...]:
    """本次调用要处理的阶: ``--orders`` 优先, 否则用 ``--n-max``。

    Args:
        params: 扫描参数。

    Returns:
        阶数元组。
    """
    if params.orders:
        return tuple(int(n) for n in params.orders)
    return tuple(int(n) for n in params.n_max_list)


def group_fields(group: str, meta: dict[str, str] | None = None) -> dict[str, Any]:
    """这一组的身份字段 (组名 / 自变量 / 固定参数)。

    ``summary.json`` 与根目录的 ``manifest.json`` 都用这三个字段, 所以定义只有一处。
    固定参数**逐字**保留 —— 清单是这一组的事实记录, 推断类型会让人以为参数被归一化过。

    Args:
        group: 组名 (``--group``)。
        meta: 这一组的固定参数 (``--group-meta``)。

    Returns:
        ``{"group": ..., "variable": ..., "group_meta": {...}}``。
    """
    return {
        "group": str(group),
        "variable": GROUP_VARIABLE,
        "group_meta": {str(k): str(v) for k, v in (meta or {}).items()},
    }


def check_zernike_clip(
    max_abs_coefficient: float,
    clip: float = ZERNIKE_CLIP,
) -> str | None:
    """Return a warning if a fitted coefficient exceeds the optimizer clip.

    The SPGD loop truncates coefficients at ``ZERNIKE_CLIP`` rad, so a fit that
    exceeds it is silently altered. Returns ``None`` when the fit is safe.

    Args:
        max_abs_coefficient: ``max(|coefficients|)`` from the Zernike fit.
        clip: The optimizer's coefficient clip in radians.

    Returns:
        A human-readable warning string, or ``None`` if within the clip.
    """
    if not np.isfinite(max_abs_coefficient):
        return f"non-finite Zernike coefficient (max|c|={max_abs_coefficient})"
    if max_abs_coefficient > clip:
        return (
            f"max|Zernike coefficient| = {max_abs_coefficient:.3f} rad exceeds "
            f"the optimizer clip {clip:.1f} rad; the SPGD loop will truncate it"
        )
    return None


def _exposure_token(exposure_time_ms: float) -> str:
    """Spell this driver's exposure in the vocabulary Stage C accepts.

    The Stage-C optimizer resolves ``--exposure-time-ms`` through the shared
    CLI vocabulary, which **rejects a bare 0** ("0 is not a synonym for
    'auto'"). This driver has always used ``0`` to mean "keep the device
    default", so it is translated into the token that says exactly that
    instead of aborting the whole sweep at Stage C.

    Args:
        exposure_time_ms: The driver's own exposure value in ms (``<= 0``
            means "keep the device default").

    Returns:
        The token or the numeric string to hand to the optimizer.
    """
    if float(exposure_time_ms) <= 0.0:
        return DEFAULT_EXPOSURE_TOKEN
    return f"{float(exposure_time_ms):g}"


def build_stage_c_config(
    *,
    center: tuple[int, int],
    coefficients: np.ndarray,
    n_max: int,
    side: float,
    beam_radius_px: float,
    cam_type: str,
    cam_id: int,
    exposure_time_ms: float,
    cam_size: int,
    slm_number: int,
    epochs: int,
    delta: float,
    lr: float,
    seed: int,
    max_energy_loss: float = 0.6,
    send_overlay: PhaseOverlay | None = None,
) -> SlmZernikePibConfig:
    """Assemble the Stage-C SPGD config for one truncation order.

    The fitted coefficients seed the optimizer through ``slm.init_c`` and the
    Zernike aperture is pinned to ``beam_radius_px`` so the fit and the
    re-optimisation share one basis.

    Args:
        center: 0-order centre ``(x, y)`` in camera-frame coordinates.
        coefficients: Noll-order seed coefficients (raw radians).
        n_max: Zernike truncation order for this run.
        side: Shared square-target side in camera pixels.
        beam_radius_px: Illuminated pupil radius on the panel, in pixels.
        cam_type: Camera backend (``sim`` / ``daheng`` / ``miicam``).
        cam_id: Camera device id.
        exposure_time_ms: Fixed exposure in ms (0 = device default).
        cam_size: Camera window edge in pixels.
        slm_number: SLM device number.
        epochs: SPGD iterations.
        delta: SPGD perturbation amplitude in radians.
        lr: Adam learning rate (0 = auto).
        seed: RNG seed.
        max_energy_loss: In-ROI energy-loss fraction that abandons an
            evaluation (``0`` disables the guard). The guard is what stops the
            search emptying the target box, but on a small box it can also
            reject every perturbation of an already-good fitted phase; lower it
            (or set ``0``) to let SPGD explore.
        send_overlay: Focus-correction overlay applied to the SLM this run
            opens. ``None`` (default) leaves the send path untouched.

    Returns:
        A fully-specified :class:`SlmZernikePibConfig`.
    """
    camera = CameraParamsPib(
        cam_type=cam_type,
        cam_id=cam_id,
        exposure_time_ms=_exposure_token(exposure_time_ms),
        cam_size=cam_size,
        target=ObjectiveTarget(name=OBJECTIVE, target_shape="square"),
        target_size=float(side),
        max_roi_energy_loss=float(max_energy_loss),
    )
    slm = SlmParamsPib(
        slm_number=slm_number,
        n_max=int(n_max),
        zernike_radius=float(beam_radius_px),
        init_c=np.asarray(coefficients, dtype=np.float64),
    )
    return SlmZernikePibConfig(
        center=center,
        epochs=int(epochs),
        algorithm="spgd",
        delta=float(delta),
        lr=float(lr),
        random_seed=int(seed),
        camera=camera,
        slm=slm,
        send_overlay=send_overlay,
    )


def score_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    center_xy: tuple[int, int],
    side: float,
) -> dict[str, float]:
    """Score one far-field frame on the shared square mask.

    Args:
        frame: 2-D camera frame.
        mask: Boolean square support (the frozen shared mask).
        center_xy: Square centre ``(x, y)`` in camera-frame coordinates.
        side: Square side in camera pixels.

    Returns:
        ``{"pib", "cv", "ee", "ar", "quality"}`` - all floats.
    """
    arr = np.asarray(frame, dtype=np.float64)
    side_i = int(round(side))
    pib = float(power_in_bucket(arr, mask))
    cv = float(uniformity_cv(arr, mask))
    ee = float(square_encircled_energy(arr, center_xy, side_i))
    ar = float(square_aspect_ratio(arr, center_xy, side_i))
    quality = float(square_quality_score(cv, ee, ar))
    return {"pib": pib, "cv": cv, "ee": ee, "ar": ar, "quality": quality}


def _metrics_identical(
    a: dict[str, float],
    b: dict[str, float],
    tol: float = 1e-9,
) -> bool:
    """Return ``True`` when two score dicts agree on every metric within ``tol``.

    Used to detect a Stage-C SPGD run that never moved off its own starting
    point: the optimizer's best row is then the epoch-0 row, so the re-scored
    ``after`` metrics are bit-for-bit the ``before`` metrics. A genuine
    improvement - however small - breaks the equality.
    """
    return all(abs(float(a[key]) - float(b[key])) <= tol for key in a)


# ---------------------------------------------------------------------------
# Flat-field probe
# ---------------------------------------------------------------------------


def configure_sim_system(seed: int, far_field_padding: int) -> None:
    """Reset the process-wide sim system with ``seed`` and the requested padding.

    ``slm_pib_sim.reset_system`` does not take a padding argument, so the padding
    is set on the returned system *before* any far field is computed (at that
    point ``_far_field`` and ``_pad_buffer`` are still ``None``, so the change is
    safe and the next ``far_field()`` uses it). The existing disturbance is
    carried over so a caller that installed one keeps it.

    Args:
        seed: RNG seed for the simulated noise model.
        far_field_padding: Zero-padding factor for the far-field FFT.
    """
    from ao_shaping.drivers.sim.slm_pib_sim import get_system, reset_system

    existing = get_system()
    system = reset_system(seed=seed, disturbance=existing.disturbance)
    system.far_field_padding = int(far_field_padding)


class PathBlockedError(RuntimeError):
    """平场远场里没有光 —— 光路受阻, 继续跑只是在对着死路测噪声。

    2026-10-21 实测: 光束没进 2f 透镜时, 平场画面里只剩读出噪声, 而噪声照样能
    算出零级位置与腰径 —— 于是整套 8 个泽尼克阶次都跑在一条**死路**上, 而且
    看上去毫无异常。判据必须看**结构** (见
    :func:`ao_shaping.utils.image.frame_analysis.assess_spot`)。
    """


def probe_flat_field(
    cam_type: str,
    cam_id: int,
    exposure_time_ms: float,
    seed: int,
    cam_size: int,
    far_field_padding: int = DEFAULT_SIM_FAR_FIELD_PADDING,
    *,
    send_overlay: PhaseOverlay | None = None,
    slm_number: int = 1,
) -> tuple[tuple[int, int], float, tuple[int, int]]:
    """Probe a flat-phase far field for the 0-order centre, waist and shape.

    The probe opens the *same* camera the Stage-C optimizer will open, so the
    returned frame shape and centre are in the exact geometry Stage C scores.
    In simulation the process-level system is reset with ``seed`` and configured
    with ``far_field_padding`` first, making the flat far field deterministic and
    wide enough for the target box to clear :data:`MIN_TARGET_SIDE`. The 0-order
    is located by ``argmax`` (never by geometry - the optical axis is not the
    frame centre).

    🔴 **光路自检**: 抓到的平场画面先判一次「有没有可评分的局域光斑」。光路被
    挡住时画面里只有读出噪声, 而噪声照样能给出零级位置与腰径 —— 于是后面 8 个
    泽尼克阶次全都跑在一条死路上, 且看上去毫无异常。所以这里直接抛
    :class:`PathBlockedError`, 由调用方中止整轮。

    Args:
        cam_type: Camera backend (``sim`` / ``daheng`` / ``miicam``).
        cam_id: Camera device id.
        exposure_time_ms: Fixed exposure in ms (0 = device default).
        seed: RNG seed for the simulated system.
        cam_size: Camera window edge in pixels. Forwarded to ``create_camera``
            so the probe frame matches the requested window (the sim CCD
            otherwise falls back to its 250 px default).
        far_field_padding: Sim far-field FFT padding (ignored for hardware).

    Returns:
        ``(center_xy, w0_px, frame_shape)`` where ``center_xy`` is ``(x, y)``.

    Raises:
        PathBlockedError: The flat frame carries no localisable spot (the path is
            not delivering light).
    """
    if cam_type == "sim":
        configure_sim_system(seed, far_field_padding)

    from types import SimpleNamespace

    from ao_shaping.drivers.slm.factory import open_camera, open_slm
    from ao_shaping.gui.utils.phase_overlay import apply_overlays_to_handle
    from ao_shaping.optimizer.wfless._measure import display_and_average
    from ao_shaping.utils.slm.phase_embed import panel_shape_from_slm

    # ``BenchDeviceParams`` 是 **Protocol** (结构化类型), 不能直接实例化 ——
    # 工厂只按属性取用, 所以给一个同形状的命名空间即可。
    device_params = SimpleNamespace(
        cam_type=cam_type,
        cam_id=cam_id,
        exposure_time_ms=exposure_time_ms,
        cam_size=cam_size,
        slm_number=slm_number,
        slm_wavelength=DEFAULT_WAVELENGTH_NM,
    )
    slm = open_slm(device_params)
    cam = open_camera(device_params)
    with slm, cam:
        # 🔴 **面板必须由本函数摆成「平相位 + 焦面矫正」**, 不能只开相机抓一帧:
        # 抓到的会是面板上**上一次**留下的相位。2026-10-21 13:14 实测: 焦面标定组
        # 刚跑完, 面板上留的是**未矫正**的平相位 ⇒ 画面是一圈弥散环 ⇒ 峰值停在
        # 读出噪声本底上 ⇒ 被判成「光路受阻」(退出码 3), 整整一轮白跑。
        if send_overlay is not None:
            apply_overlays_to_handle(slm, send_overlay)
        panel_shape = panel_shape_from_slm(slm)
        # display_and_average 负责「等面板稳定」(按稳定性判, 不按时间); 它返回的
        # 是减过中位数的帧, 所以下面再抓一张**原始**帧来量腰径与判光路。
        display_and_average(
            cam,
            slm,
            np.zeros(panel_shape, dtype=np.float64),
            n_frames=1,
            settle_wait_s=0.5,
            settle_tol=0.02,
            settle_max_wait_s=6.0,
            settle_max_discard=40,
        )
        frame = np.asarray(cam.get_numpy_image(), dtype=np.float64)
    assessment = assess_spot(frame)
    if assessment.blocked:
        raise PathBlockedError(assessment.reason)
    center = zero_order_center(frame)
    w0 = float(spot_waist_sigma(frame, center))
    return center, w0, (int(frame.shape[0]), int(frame.shape[1]))


def _crop_to_shape(
    frame: np.ndarray,
    center_xy: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Crop ``frame`` to ``shape`` centred on ``center_xy`` (no-op if equal).

    Stage A and Stage C can still open different camera windows on hardware
    (Stage A windows the CCD explicitly, Stage C relies on the backend), so the
    Stage-A baselines are re-cropped onto the Stage-C geometry before scoring.
    """
    arr = np.asarray(frame, dtype=np.float64)
    h, w = int(shape[0]), int(shape[1])
    if arr.shape == (h, w):
        return arr
    if arr.shape[0] < h or arr.shape[1] < w:
        raise ValueError(f"cannot crop frame {arr.shape} to larger shape {(h, w)}")
    cx, cy = int(center_xy[0]), int(center_xy[1])
    x0 = max(0, min(cx - w // 2, arr.shape[1] - w))
    y0 = max(0, min(cy - h // 2, arr.shape[0] - h))
    return arr[y0 : y0 + h, x0 : x0 + w]


def _patch_sim_stage_c_camera_window() -> None:
    """Make the Stage-C optimizer's camera honour ``cam_size`` for the sim backend.

    ``create_camera`` reads only ``cam_type`` / ``cam_id`` / ``exposure_time_ms``
    / ``skip_sampling`` off a params object, so ``optimize_slm_zernike_pib``'s
    ``create_camera(config.camera)`` silently drops ``cam_size`` and the sim CCD
    falls back to its 250 px default. Stage A already forwards it (via
    ``drivers.slm.factory.open_camera``), so without this the probe, Stage A and
    Stage C would score on three different windows and ``frame_shape`` would not
    match the requested ``--cam-size``. Rebinding the optimizer module's
    ``create_camera`` mirrors ``patch_sim_pib_shaping``'s rebinding of its
    ``Santec`` name; hardware backends filter ``cam_size`` out of their
    ``accepted_kwargs``, so this is a no-op for them.
    """
    import ao_shaping.optimizer.wfless.slm_zernike_pib as pib_opt

    original = pib_opt.create_camera

    def create_camera_with_window(
        camera_type: Any,
        cam_id: int | str = 0,
        exposure_time_ms: float = 20.0,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(camera_type, str):
            cam_size = getattr(camera_type, "cam_size", None)
            if cam_size is not None:
                kwargs.setdefault("cam_size", cam_size)
        return original(
            camera_type, cam_id=cam_id, exposure_time_ms=exposure_time_ms, **kwargs
        )

    pib_opt.create_camera = create_camera_with_window


def _window_mask_centre(
    flat_frame: np.ndarray,
    persisted: tuple[int, int] | None,
    side: float,
) -> tuple[tuple[int, int], str]:
    """续跑块的目标框中心 —— 一律**窗口内**坐标。

    框是画在**平场帧**上的, 而平场帧是相机裁出来的 ROI 窗口 (256x256), 所以框
    中心必须是这个窗口自己的坐标 (零级在 ~128)。全幅传感器坐标 (例如
    (1434, 973)) 落在窗口之外, 框整个跑到画面外 —— 2026-10-21 实测的分块续跑
    失败正是这么来的。

    优先用交接文件里记的窗口内中心; 缺了 (老版交接文件) 或者它落在窗口之外
    (有人把全幅坐标写进去了) ⇒ 从平场帧重新定位, 与优化器自己用的峰值定位
    同一套 routine。

    Args:
        flat_frame: 交接文件里的平场帧 (ROI 窗口坐标系)。
        persisted: 交接文件里记的窗口内中心 ``(x, y)``; ``None`` = 没记。
        side: 共享方形目标框边长 (相机像素)。

    Returns:
        ``((x, y), how)`` —— 框中心与取值途径 (``"persisted"`` / ``"relocated"``)。
    """
    frame = np.asarray(flat_frame, dtype=np.float64)
    side_px = max(1, int(round(side)))
    if persisted is not None:
        cx, cy = int(persisted[0]), int(persisted[1])
        if square_mask(frame.shape, cy, cx, side_px).any():
            return (cx, cy), "persisted"
    # ``locate_zero_order`` 返回的是 (行, 列), 这里要的是 (列, 行)。
    cy, cx = locate_zero_order(frame)
    return (int(cx), int(cy)), "relocated"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class SweepParams:
    """All knobs for one sweep run (mirrors the CLI)."""

    cam_type: str = "sim"
    cam_id: int = 0
    exposure_time_ms: float = 0.0
    cam_size: int = 256
    sim_far_field_padding: int = DEFAULT_SIM_FAR_FIELD_PADDING
    slm_number: int = 1
    beam_radius_px: float = 450.0
    epochs: int = 400
    gs_iters: int = 200
    spgd_epochs: int = 200
    n_max_list: tuple[int, ...] = DEFAULT_N_MAX
    delta: float = 0.02
    lr: float = 0.01
    seed: int = 0
    out: str = ""
    target_side: float = 0.0
    max_energy_loss: float = 0.6
    save_images: bool = False
    defocus_focal_mm: float | None = None
    """Pupil Fresnel focus-correction focal length in mm (``None`` = off).

    Recorded in ``summary.json``'s ``config`` so a run can be reproduced.
    """

    group: str = "main"
    """Identity of the variable group this invocation probes.

    The sweep writes into ``<out>/<group>/`` and the root gets a ``manifest.json``
    listing every group recorded under it.
    """

    group_meta: dict[str, str] = field(default_factory=dict)
    """Fixed parameters of this group, recorded verbatim (see ``--group-meta``)."""

    stage_a_only: bool = False
    """只跑 Stage A, 落盘后退出 (不再跑 Stage B/C)。"""

    stage_a_in: str = ""
    """读回 Stage A 产物的路径 (``stage_a.npz``); 非空 = 跳过平场探测与 Stage A。"""

    orders: tuple[int, ...] = ()
    """本次调用要处理的阶 (非空时**优先于** ``n_max_list``)。"""

    time_budget_s: float = 0.0
    """Stage A 的墙钟预算 (s); 0 = 不限。"""

    chunk_epochs: int = 0
    """本块 Stage A 跑多少个精化 epoch (0 = 跑满 ``epochs``)。

    「一次扫描不超过 10 分钟」把 Stage A 切成若干块后, ``epochs`` 保持**总轮数**
    (学习率调度的总地平线不变), 本字段只说这一块跑多少轮; 实际轮数还会被
    「总轮数 - 已跑轮数」夹住, 所以最后一块自动收口。
    """


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-serialisable values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# Stage A 的落盘 / 读回 (分段扫描的接缝)
# ---------------------------------------------------------------------------

#: Stage A 落盘的两个文件名 (基名)。``--stage-a-only`` 写它们, ``--stage-a-in`` 读它们。
STAGE_A_NPZ_NAME = "stage_a.npz"
STAGE_A_JSON_NAME = "stage_a.json"


@dataclass
class StageAResult:
    """一次 Stage A 跑出来的全部可复用状态 (落盘与读回共用这一个形状)。

    Stage B 每次都从 ``best_phase`` 重新拟合, 所以「可复用的东西」就是这三个
    数组 (最佳相位 + 两张基线帧) 加上一份几何元数据 —— 没有别的隐藏状态。
    """

    best_phase: np.ndarray
    """Stage A 的最佳相位 (面板尺寸的 raw 弧度)。"""

    flat_frame: np.ndarray
    """平场基线帧 (raw 相机帧)。"""

    best_frame: np.ndarray
    """最佳相位对应的远场帧 (raw 相机帧)。"""

    center_xy: tuple[int, int]
    """零级位置 ``(x, y)`` (**全幅**传感器坐标)。

    这是平场探测在全幅画面上找到的零级 —— Stage C 用它摆 ROI 窗口, 所以它必须
    留在全幅坐标系里。它**不是**评分用的框中心。
    """

    w0: float
    """平场光斑腰径 (相机像素)。"""

    side: float
    """共享方形目标框边长 (相机像素)。"""

    probe_shape: tuple[int, int]
    """平场探测时的画面尺寸 ``(h, w)``。"""

    best_stage: str
    """最佳结果来自哪个阶段 (flat / gs / refine)。"""

    best_score: float
    """最佳复合分数。"""

    epochs_executed: int
    """实际跑掉的精化 epoch 数 (被墙钟预算截断时小于请求值)。"""

    elapsed_s: float = 0.0
    """Stage A 实测墙钟耗时 (s) —— 「一次扫描不超过 10 分钟」这条约束的账本。"""

    config: dict[str, Any] = field(default_factory=dict)
    """本次 Stage A 用的配置 (含 ``defocus_focal_mm``), 供读回时复现。"""

    dof: np.ndarray | None = None
    """精化阶段的自由参数量 (grid×grid)。续跑必须带上它, 否则等于从头再来。"""

    base_panel: np.ndarray | None = None
    """精化所依附的基座相位 (GS 面板或平场), **未叠加**零级叠加层。"""

    best_phase_plain: np.ndarray | None = None
    """最佳相位的**未叠加**版本 (显示时再加叠加层; 续跑要的是这个)。"""

    epochs_total: int = 0
    """到这一块为止累计跑掉的精化轮数 (= 下一块的起点)。"""

    mask_center_xy: tuple[int, int] | None = None
    """目标框中心 ``(x, y)`` —— **窗口内**坐标 (ROI 窗口自己的坐标系)。

    评分用的方形框就画在这里, 所以它必须是**窗口内**坐标: 平场帧是相机裁出来的
    ROI 窗口 (256x256), 而 :attr:`center_xy` 是全幅传感器坐标 —— 拿后者当框中心,
    框整个落在窗口之外 (2026-10-21 实测: "target square ... is empty")。
    """

    full_frame_center: tuple[int, int] | None = None
    """ROI 窗口在全幅传感器上的中心 ``(x, y)``。

    下一块据此把窗口摆在**同一个**位置: 重新定位零级会落到近等的散斑颗粒上,
    窗口一平移, 窗口坐标系就跟着挪, 冻结的框就指错了地方。
    """


def _resumable_arrays(stage_a: StageAResult) -> bool:
    """交接文件里够不够续跑一块。

    DOF 与基座相位都必须在, 而且不能是**空数组**: 老版交接文件干脆没有这两个
    键 (读回来是 None), 而写坏的文件可能留下长度为 0 的数组 —— 两者都意味着
    「这一块没法接着跑」, 必须当场报错, 而不是带着空 DOF 往下跑、在更难懂的地方
    炸掉。
    """
    for name in ("dof", "base_panel"):
        value = getattr(stage_a, name, None)
        if value is None or np.asarray(value).size == 0:
            return False
    return True


def save_stage_a(result: StageAResult, out_dir: Path) -> tuple[Path, Path]:
    """把 Stage A 的产物落盘。

    数组进 ``stage_a.npz`` (raw 数组, 不做任何归一化 —— 读回时要能逐位复现
    评分), 元数据进 ``stage_a.json`` (人可读, 记录几何与实际跑掉的 epoch 数)。

    Args:
        result: Stage A 的产物。
        out_dir: 组目录 (不存在会被创建)。

    Returns:
        ``(npz_path, json_path)``。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / STAGE_A_NPZ_NAME
    arrays: dict[str, np.ndarray] = {
        "best_phase": np.asarray(result.best_phase, dtype=np.float64),
        "flat_frame": np.asarray(result.flat_frame, dtype=np.float64),
        "best_frame": np.asarray(result.best_frame, dtype=np.float64),
    }
    # 续跑需要的三件 (自由参数量 / 基座相位 / 未叠加的最佳相位): **值为 None 就
    # 不写这个键**。写成空数组的话, 读回来是「长度 0 的数组」而不是 None ——
    # 「不可续跑」的守卫就形同虚设, 带着空 DOF 往下跑会在更难懂的地方炸掉
    # (2026-10-21 复核时发现的老坑)。缺键 = None, 读回语义才与写出一致。
    if result.dof is not None:
        arrays["dof"] = np.asarray(result.dof, dtype=np.float64)
    if result.base_panel is not None:
        arrays["base_panel"] = np.asarray(result.base_panel, dtype=np.float64)
    if result.best_phase_plain is not None:
        arrays["best_phase_plain"] = np.asarray(
            result.best_phase_plain, dtype=np.float64
        )
    np.savez(npz_path, **arrays)
    json_path = out_dir / STAGE_A_JSON_NAME
    payload = {
        "center_xy": [int(result.center_xy[0]), int(result.center_xy[1])],
        # 框中心 (**窗口内**坐标) 与摆窗口用的全幅零级位置分开放: 这两个坐标
        # 系混用就是 2026-10-21 那次分块续跑失败的根因。
        "mask_center_xy": (
            None
            if result.mask_center_xy is None
            else [int(result.mask_center_xy[0]), int(result.mask_center_xy[1])]
        ),
        "full_frame_center": (
            None
            if result.full_frame_center is None
            else [int(result.full_frame_center[0]), int(result.full_frame_center[1])]
        ),
        "w0": float(result.w0),
        "side": float(result.side),
        "probe_shape": [int(result.probe_shape[0]), int(result.probe_shape[1])],
        "best_stage": str(result.best_stage),
        "best_score": float(result.best_score),
        "epochs_executed": int(result.epochs_executed),
        "epochs_total": int(result.epochs_total),
        "elapsed_s": float(result.elapsed_s),
        "config": _jsonable(result.config),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return npz_path, json_path


def load_stage_a(path: str | Path) -> StageAResult:
    """读回 Stage A 的产物。

    ``path`` 给到 ``stage_a.npz`` 即可, 同目录的 ``stage_a.json`` 自动配对;
    给目录也行 (取目录里的 ``stage_a.npz``)。

    Args:
        path: ``stage_a.npz`` 的路径, 或它所在的目录。

    Returns:
        读回的 :class:`StageAResult`。

    Raises:
        FileNotFoundError: ``stage_a.npz`` 或配对的 ``stage_a.json`` 不存在。
        ValueError: 元数据缺字段, 或 npz 里的数组不齐。
    """
    npz_path = Path(path)
    if npz_path.is_dir():
        npz_path = npz_path / STAGE_A_NPZ_NAME
    json_path = npz_path.with_name(STAGE_A_JSON_NAME)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Stage A 产物不存在: {npz_path}")
    if not json_path.is_file():
        raise FileNotFoundError(f"Stage A 元数据不存在: {json_path}")

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"Stage A 元数据必须是 JSON 对象: {json_path}")
    with np.load(npz_path) as data:
        arrays = {
            key: np.asarray(data[key], dtype=np.float64)
            for key in ("best_phase", "flat_frame", "best_frame")
        }
        # 续跑用的两件是**可选**的 (旧版交接文件没有它们 ⇒ 只能从头跑)
        for key in ("dof", "base_panel", "best_phase_plain"):
            if key in data:
                arrays[key] = np.asarray(data[key], dtype=np.float64)
    try:
        center = meta["center_xy"]
        shape = meta["probe_shape"]
        mask_center = meta.get("mask_center_xy")
        full_center = meta.get("full_frame_center")
        return StageAResult(
            best_phase=arrays["best_phase"],
            flat_frame=arrays["flat_frame"],
            best_frame=arrays["best_frame"],
            center_xy=(int(center[0]), int(center[1])),
            w0=float(meta["w0"]),
            side=float(meta["side"]),
            probe_shape=(int(shape[0]), int(shape[1])),
            best_stage=str(meta.get("best_stage", "")),
            best_score=float(meta.get("best_score", float("nan"))),
            epochs_executed=int(meta.get("epochs_executed", 0)),
            epochs_total=int(meta.get("epochs_total", meta.get("epochs_executed", 0))),
            elapsed_s=float(meta.get("elapsed_s", 0.0)),
            dof=arrays.get("dof"),
            base_panel=arrays.get("base_panel"),
            best_phase_plain=arrays.get("best_phase_plain"),
            config=dict(meta.get("config") or {}),
            mask_center_xy=(
                None
                if mask_center is None
                else (int(mask_center[0]), int(mask_center[1]))
            ),
            full_frame_center=(
                None
                if full_center is None
                else (int(full_center[0]), int(full_center[1]))
            ),
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Stage A 元数据缺字段或格式不对: {json_path} ({exc})") from exc


def _scalar_columns(history: list[dict[str, Any]]) -> list[str]:
    """Column names whose values are scalars across every recorded row."""
    if not history:
        return []
    columns: list[str] = []
    for key in history[0]:
        if all(
            isinstance(row.get(key), (int, float, str, bool, type(None)))
            for row in history
        ):
            columns.append(key)
    return columns


def _write_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    """Write the scalar columns of a recorder history to CSV."""
    columns = _scalar_columns(history)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in history:
            handle.write(",".join(str(row.get(col, "")) for col in columns) + "\n")


def _save_frame_png(frame: np.ndarray, path: Path, title: str) -> None:
    """Save a single frame as a PNG (Agg backend, no display)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.asarray(frame, dtype=np.float64), cmap="inferno")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _flat_frame(history: list[dict[str, Any]]) -> np.ndarray:
    """Extract the flat-baseline frame from a Stage-A history."""
    for row in history:
        if row.get("stage") == "flat" and row.get(RECORD_IMAGE) is not None:
            return np.asarray(row[RECORD_IMAGE], dtype=np.float64)
    raise RuntimeError("Stage-A history has no flat frame (record_debug off?)")


#: Stage A 遇到 CCD 饱和时的曝光退避次数。每次把曝光减半后重跑 Stage A。
#: 为什么必须退避而不是直接失败: GS 的 SPGD 细化会把光**越收越紧**, 目标框只有
#: ~9 px 时, 收敛后的方斑峰值随轮次上升, 撞到 CCD 的 250 counts 饱和线
#: (2026-10-20 实测: 0.4 ms 下第 ~310 轮饱和, 整轮 22 分钟的 Stage A 作废)。
SATURATION_BACKOFF_ATTEMPTS = 3
#: 曝光退避的下限 (ms)。再低就没有可用信号了, 那时应当报「光太弱」而不是继续退避。
MIN_BACKOFF_EXPOSURE_MS = 0.02


def _run_stage_a_with_saturation_backoff(
    stage_a: SlmGsRefineConfig, params: SweepParams
) -> Any:
    """跑 Stage A, 遇 CCD 饱和就把曝光减半后重跑。

    ``slm_gs_refine.measure`` 对饱和帧直接抛 ``ValueError`` (这是对的: 饱和帧的
    指标没有意义)。但 GS 的 SPGD 细化本来就是「越收越紧」的过程, 越到后面越亮,
    于是收敛前的某一轮会先撞线。此时整轮 Stage A 的时间全部作废 —— 所以这里
    接住这个异常、曝光减半重跑, 并把降下来的曝光同步给 Stage C (``params``),
    让三段用同一个曝光, 基线才可比。
    """
    config = stage_a
    for attempt in range(SATURATION_BACKOFF_ATTEMPTS + 1):
        try:
            return optimize_slm_gs_refine(config)
        except ValueError as exc:
            saturated = "saturat" in str(exc).lower()
            if not saturated or attempt == SATURATION_BACKOFF_ATTEMPTS:
                raise
            current = float(config.exposure_time_ms)
            if current <= MIN_BACKOFF_EXPOSURE_MS:
                raise
            reduced = max(MIN_BACKOFF_EXPOSURE_MS, current * 0.5)
            print(
                f"[饱和退避] Stage A 在 {current:.4f} ms 饱和 ({exc}); "
                f"曝光降到 {reduced:.4f} ms 重跑",
                flush=True,
            )
            config = replace(config, exposure_time_ms=reduced)
            # Stage C 必须用同一个曝光, 否则两段的基线不可比。
            params.exposure_time_ms = reduced
    raise RuntimeError("unreachable: Stage A saturation backoff exhausted")


def run_sweep(params: SweepParams) -> dict[str, Any]:
    """Run the GS-refine -> Zernike-fit -> SPGD sweep, one segment per call.

    Three ways to call it, all recording only their own segment:

    * **默认**: 平场探测 + Stage A + Stage B/C (``--n-max`` 里的每个阶)。
    * **``--stage-a-only``**: 只跑 Stage A, 落盘 ``stage_a.npz`` /
      ``stage_a.json`` 后返回 (``orders`` 为空)。
    * **``--stage-a-in``**: 读回 Stage A 的产物, 跳过平场探测与 Stage A,
      直接跑 Stage B/C (只跑 ``--orders`` 指名的阶)。

    Args:
        params: Fully-specified sweep parameters.

    Returns:
        The summary dict that is also written to ``<group>/summary.json``.
        Its ``orders`` list holds **only** the orders processed by this call.
    """
    out_root = (
        Path(params.out)
        if params.out
        else Path("data/debug/slm_gs_refine_zernike")
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    # 组目录: <根目录>/<组名>/。一次调用只探这一组 (自变量 = n_max)。
    out_dir = group_output_dir(out_root, params.group)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[out] 根目录 {out_root} / 组目录 {out_dir} (组名 {params.group})")

    if params.cam_type == "sim":
        patch_sim_pib_shaping(params.cam_type)
        _patch_sim_stage_c_camera_window()

    n_max_list = resolve_orders(params)
    stage_a: StageAResult
    stage_a_source = ""

    # 分块续跑要用: 续跑块的状态与本块轮数 (不走续跑时都是空/0)。
    resume_state: dict[str, Any] | None = None
    chunk_count = 0

    if params.stage_a_in:
        # --- Segment 2+: read the Stage A result back ---------------------
        stage_a = load_stage_a(params.stage_a_in)
        stage_a_source = str(params.stage_a_in)
        center_xy = stage_a.center_xy
        w0 = stage_a.w0
        probe_shape = stage_a.probe_shape
        side = stage_a.side
        best_phase = stage_a.best_phase
        flat_frame = stage_a.flat_frame
        best_frame = stage_a.best_frame
        # The baselines are measured in Stage A's exposure. If Stage A backed
        # off from saturation, Stage C must use the SAME exposure, otherwise
        # the flat/GS baselines and the Stage-C numbers are not comparable.
        recorded_exposure = stage_a.config.get("exposure_time_ms")
        if recorded_exposure is not None:
            recorded = float(recorded_exposure)
            if abs(recorded - float(params.exposure_time_ms)) > 1e-9:
                print(
                    f"[stage A] 曝光以 Stage A 记录为准 ({recorded} ms), "
                    f"忽略命令行的 {params.exposure_time_ms} ms —— "
                    f"基线可比性要求两段用同一个曝光",
                    flush=True,
                )
                params.exposure_time_ms = recorded
        # Same argument for the focus correction: Stage C must optimise the
        # optical configuration the baselines were taken in.
        recorded_focus = stage_a.config.get("defocus_focal_mm")
        if params.defocus_focal_mm is None and recorded_focus is not None:
            params.defocus_focal_mm = float(recorded_focus)
            print(
                f"[stage A] 焦面矫正沿用 Stage A 记录的 {params.defocus_focal_mm} mm",
                flush=True,
            )
        # --- Focus correction (2f bench camera is not on the 2f plane) -----
        # Both optimizers accept a send-pipeline overlay, so ONE overlay covers
        # Stage A (GS) and Stage C (SPGD): the three stages then share the same
        # optical correction and their baselines stay comparable.
        # 叠加层在**采用 Stage A 记录的焦距之后**才建: 否则会拿命令行的焦距建叠加层,
        # 而基线是在 Stage A 的焦距下拍的 —— 两段的光学状态就不一样了。
        send_overlay = build_focus_overlay(
            params.defocus_focal_mm, beam_radius_px=params.beam_radius_px
        )
        logger.info(
            f"[stage A] 读回 {stage_a_source}: center={center_xy} w0={w0:.2f}px "
            f"target_side={side:.2f}px epochs={stage_a.epochs_executed} "
            f"best_score={stage_a.best_score:.4f}"
        )
        # 续跑这一块 Stage A (--stage-a-only 与 --stage-a-in 同时给): 把上一块
        # 的末态交给优化器接着跑。
        if params.stage_a_only:
            if not _resumable_arrays(stage_a):
                raise SystemExit(
                    "参数错误: --stage-a-in 读回的交接文件里没有续跑需要的 "
                    "DOF / 基座相位 (缺失或长度为 0; 旧版交接文件只能从头跑)"
                )
            done = int(stage_a.epochs_total)
            remaining = max(0, int(params.epochs) - done)
            chunk_count = (
                min(int(params.chunk_epochs), remaining)
                if int(params.chunk_epochs) > 0
                else remaining
            )
            # 框中心必须是**窗口内**坐标: 平场帧是相机裁出来的 ROI 窗口 (256x256),
            # 而 stage_a.json 里的 center_xy 是**全幅**传感器坐标 (例如 (1434, 973))
            # —— 拿后者当框中心, 框整个落在窗口之外, 续跑块两秒就挂
            # (2026-10-21 实测: "target square (side=12px at (973,1434)) is empty")。
            mask_center, mask_center_source = _window_mask_centre(
                stage_a.flat_frame, stage_a.mask_center_xy, side
            )
            if mask_center_source == "relocated":
                print(
                    f"[stage A] 交接文件里没有可用的窗口内零级位置, 改从平场帧"
                    f"重新定位: {mask_center}",
                    flush=True,
                )
            resume_state = {
                "base_panel": np.asarray(stage_a.base_panel, dtype=np.float64),
                "dof": np.asarray(stage_a.dof, dtype=np.float64),
                "flat_frame": np.asarray(stage_a.flat_frame, dtype=np.float64),
                "center_y": int(mask_center[1]),
                "center_x": int(mask_center[0]),
                # 全幅零级位置: 下一块据此把 ROI 窗口摆在同一个地方 (冻结 ROI)。
                # 拿不到就退回「重新发现零级」—— 实测两者一致 (chunk-1/chunk-2
                # 的窗口都是 (1429, 971))。
                "full_frame_center": (
                    None
                    if stage_a.full_frame_center is None
                    else (
                        int(stage_a.full_frame_center[0]),
                        int(stage_a.full_frame_center[1]),
                    )
                ),
                "best_score": float(stage_a.best_score),
                "best_phase": np.asarray(
                    stage_a.best_phase_plain
                    if stage_a.best_phase_plain is not None
                    else stage_a.best_phase,
                    dtype=np.float64,
                ),
                "best_frame": np.asarray(stage_a.best_frame, dtype=np.float64),
                "best_stage": str(stage_a.best_stage),
                "epoch_offset": done,
            }
            print(
                f"[stage A] 续跑: 已跑 {done} 轮, 本块跑 {chunk_count} 轮 "
                f"(总轮数 {params.epochs})",
                flush=True,
            )
    else:
        # --- Segment 1: probe the flat far field ---------------------------
        # 叠加层必须在探测**之前**建好: 平场探测要自己把面板摆成
        # 「平相位 + 这张矫正」(见 probe_flat_field 的说明)。
        send_overlay = build_focus_overlay(
            params.defocus_focal_mm, beam_radius_px=params.beam_radius_px
        )
        center_xy, w0, probe_shape = probe_flat_field(
            params.cam_type,
            params.cam_id,
            params.exposure_time_ms,
            params.seed,
            params.cam_size,
            params.sim_far_field_padding,
            send_overlay=send_overlay,
            slm_number=params.slm_number,
        )
        side = resolve_target_side(w0, params.cam_size, params.target_side)
        logger.info(
            f"[probe] cam_type={params.cam_type} probe_shape={probe_shape} "
            f"center={center_xy} w0={w0:.2f}px target_side={side:.2f}px"
        )

    if not params.stage_a_in or params.stage_a_only:
        # --- Stage A: one GS-refine optimisation --------------------------
        # (续跑块也走这里: --stage-a-only + --stage-a-in)
        stage_a_cfg = SlmGsRefineConfig(
            epochs=params.epochs,
            seed=params.seed,
            gs_iters=params.gs_iters,
            target_side=int(round(side)),
            side_factor=1.0,
            cam_type=params.cam_type,
            cam_id=params.cam_id,
            exposure_time_ms=params.exposure_time_ms,
            cam_size=params.cam_size,
            slm_number=params.slm_number,
            beam_radius_px=params.beam_radius_px,
            record_debug=True,
            send_overlay=send_overlay,
            time_budget_s=float(params.time_budget_s),
            chunk_epochs=int(chunk_count),
            resume_state=resume_state,
        )
        rec_a = _run_stage_a_with_saturation_backoff(stage_a_cfg, params)
        best_phase = np.asarray(rec_a.best_phase, dtype=np.float64)
        best_frame = np.asarray(rec_a.best_frame, dtype=np.float64)
        # 续跑块的历史里**没有** flat 行 (平场是上一块测的, 重测会引入漂移) ⇒
        # 沿用上一块的基线帧。
        has_flat = any(r["stage"] == "flat" for r in rec_a.history)
        flat_frame = (
            _flat_frame(rec_a.history)
            if has_flat
            else np.asarray(stage_a.flat_frame, dtype=np.float64)
        )
        chunk_best = max((r["score"] for r in rec_a.history), default=float("-inf"))
        best_score = (
            max(float(stage_a.best_score), float(chunk_best))
            if resume_state is not None
            else chunk_best
        )
        epochs_executed = int(getattr(rec_a, "epochs_executed", 0))
        logger.info(
            f"[stage A] best_stage={rec_a.best_stage} "
            f"best_score={best_score:.4f} "
            f"epochs={epochs_executed}/{params.epochs} "
            f"evals={len(rec_a.history)}"
        )
        # 框中心取**优化器冻结的窗口内**零级位置 (recorder.center_*), 而不是
        # center_xy —— 后者是全幅传感器坐标, 拿来当框中心会让框落在窗口之外。
        mask_center_xy = (
            int(getattr(rec_a, "center_x", 0)),
            int(getattr(rec_a, "center_y", 0)),
        )
        full_frame_center = getattr(rec_a, "full_frame_center", None)
        stage_a = StageAResult(
            best_phase=best_phase,
            flat_frame=flat_frame,
            best_frame=best_frame,
            center_xy=center_xy,
            w0=w0,
            side=side,
            probe_shape=probe_shape,
            best_stage=str(rec_a.best_stage),
            best_score=float(best_score),
            epochs_executed=epochs_executed,
            epochs_total=int(
                getattr(rec_a, "epochs_total", epochs_executed)
            ),
            elapsed_s=float(getattr(rec_a, "elapsed_s", 0.0)),
            dof=np.asarray(getattr(rec_a, "dof", np.zeros(0)), dtype=np.float64),
            base_panel=np.asarray(
                getattr(rec_a, "base_panel", np.zeros(0)), dtype=np.float64
            ),
            best_phase_plain=np.asarray(
                getattr(rec_a, "best_phase_plain", best_phase), dtype=np.float64
            ),
            config=_jsonable(asdict(params)),
            mask_center_xy=mask_center_xy,
            full_frame_center=(
                None
                if full_frame_center is None
                else (int(full_frame_center[0]), int(full_frame_center[1]))
            ),
        )
        if params.stage_a_only:
            npz_path, json_path = save_stage_a(stage_a, out_dir)
            logger.info(f"[stage A] 落盘 {npz_path} / {json_path}")
            summary = {
                **group_fields(params.group, params.group_meta),
                "config": _jsonable(asdict(params)),
                "cam_size": int(params.cam_size),
                "probe_shape": [int(probe_shape[0]), int(probe_shape[1])],
                "frame_shape": [int(probe_shape[0]), int(probe_shape[1])],
                # center_xy = 全幅探测到的零级 (Stage C 摆窗口用);
                # mask_center_xy = 窗口内的框中心 (评分用)。两个坐标系别混。
                "center_xy": [int(center_xy[0]), int(center_xy[1])],
                "mask_center_xy": [
                    int(mask_center_xy[0]),
                    int(mask_center_xy[1]),
                ],
                "full_frame_center": (
                    None
                    if full_frame_center is None
                    else [
                        int(full_frame_center[0]),
                        int(full_frame_center[1]),
                    ]
                ),
                "w0_px": float(w0),
                "target_side": float(side),
                "beam_radius_px": float(params.beam_radius_px),
                "best_stage": str(stage_a.best_stage),
                "stage_a": {
                    "source": str(npz_path),
                    "epochs_executed": int(stage_a.epochs_executed),
                    "best_score": float(stage_a.best_score),
                    "elapsed_s": float(stage_a.elapsed_s),
                },
                "baselines": {},
                "orders": [],
            }
            (out_dir / "summary.json").write_text(
                json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
            )
            update_manifest(
                out_root,
                group_record(
                    group=params.group,
                    variable=GROUP_VARIABLE,
                    meta=params.group_meta,
                    dir_name=str(params.group),
                    n_rows=0,
                ),
            )
            logger.info(f"[done] wrote {out_dir / 'summary.json'} (Stage A only)")
            return summary

    # --- Stage B + C: per-order fit and SPGD ------------------------------
    runs: list[dict[str, Any]] = []
    for entry in sweep_plan(n_max_list):
        n_max = entry["n_max"]
        fit = fit_best_phase(best_phase, n_max, params.beam_radius_px)
        clip_warning = check_zernike_clip(fit.max_abs_coefficient)
        if clip_warning:
            logger.info(f"[warn] n_max={n_max}: {clip_warning}")

        cfg = build_stage_c_config(
            center=center_xy,
            coefficients=fit.coefficients,
            n_max=n_max,
            side=side,
            beam_radius_px=params.beam_radius_px,
            cam_type=params.cam_type,
            cam_id=params.cam_id,
            exposure_time_ms=params.exposure_time_ms,
            cam_size=params.cam_size,
            slm_number=params.slm_number,
            epochs=params.spgd_epochs,
            delta=params.delta,
            lr=params.lr,
            seed=params.seed,
            max_energy_loss=params.max_energy_loss,
            send_overlay=send_overlay,
        )
        rec_c = optimize_slm_zernike_pib(cfg)
        runs.append(
            {"entry": entry, "fit": fit, "clip_warning": clip_warning, "rec": rec_c}
        )

    # --- Shared frozen mask, on the Stage-C frame geometry ----------------
    ref_frame = np.asarray(runs[0]["rec"].history[0][RECORD_IMAGE], dtype=np.float64)
    ref_shape = (int(ref_frame.shape[0]), int(ref_frame.shape[1]))
    c_cx, c_cy = zero_order_center(ref_frame)
    mask = square_mask(ref_shape, c_cy, c_cx, int(round(side)))
    if not mask.any():
        raise ValueError(
            f"shared target square (side={side:.1f}px at ({c_cx},{c_cy})) is empty"
        )
    mask_center = (int(c_cx), int(c_cy))

    flat_scored = _crop_to_shape(flat_frame, zero_order_center(flat_frame), ref_shape)
    gs_scored = _crop_to_shape(best_frame, zero_order_center(best_frame), ref_shape)
    baselines = {
        "flat": score_frame(flat_scored, mask, mask_center, side),
        "gs_freeform": score_frame(gs_scored, mask, mask_center, side),
    }
    if params.save_images:
        _save_frame_png(flat_scored, out_dir / "baseline_flat.png", "flat")
        _save_frame_png(gs_scored, out_dir / "baseline_gs_freeform.png", "gs freeform")

    orders: list[dict[str, Any]] = []
    for run in runs:
        entry = run["entry"]
        fit = run["fit"]
        rec_c = run["rec"]
        n_max = entry["n_max"]

        before_frame = np.asarray(rec_c.history[0][RECORD_IMAGE], dtype=np.float64)
        # ``get_best_iter`` returns ``(best_row, (index, value))`` - not an int.
        best_row, (best_index, best_value) = rec_c.get_best_iter()
        after_frame = np.asarray(best_row[RECORD_IMAGE], dtype=np.float64)
        before = score_frame(before_frame, mask, mask_center, side)
        after = score_frame(after_frame, mask, mask_center, side)
        # A Stage-C run that never moved off its own starting point leaves the
        # best row == the epoch-0 row, so ``after`` is bit-for-bit ``before``.
        # That is a legitimate (if disappointing) result, but it must never be
        # invisible: warn and record the flag.
        spgd_improved = not _metrics_identical(before, after)
        spgd_epochs_executed = int(
            rec_c.history[-1].get(RECORD_EPOCH, len(rec_c.history) - 1)
        )
        js = [float(row["J"]) for row in rec_c.history if row.get("J") is not None]
        if not spgd_improved:
            logger.warning(
                "n_max={} SPGD produced no improvement: the best metrics equal "
                "the pre-SPGD fit metrics (best_epoch={}, best_objective={:.6g}); "
                "recorded as spgd_improved=false",
                n_max,
                int(best_index),
                float(best_value),
            )
        logger.info(
            f"n_max={n_max} modes={fit.n_modes} "
            f"fit_rms={fit.rms_residual_rad:.4f}rad "
            f"max|c|={fit.max_abs_coefficient:.3f} | "
            f"pib {before['pib']:.2f}->{after['pib']:.2f}  "
            f"cv {before['cv']:.2f}->{after['cv']:.2f}  "
            f"ee {before['ee']:.2f}->{after['ee']:.2f}  "
            f"improved={spgd_improved}"
        )

        _write_history_csv(rec_c.history, out_dir / f"spgd_nmax{n_max}.csv")
        np.save(out_dir / f"best_phase_nmax{n_max}.npy", best_phase)
        np.save(out_dir / f"fit_coefficients_nmax{n_max}.npy", fit.coefficients)
        # 这一阶的**泽尼克拟合相位** (Stage B 的拟合还原到面板的 raw 弧度)。
        # 远场补拍组要用**本轮**的相位拍: 上一轮的相位对应的是上一轮的焦面矫正
        # (10-20 是 f_v=1597.4 mm), 沿用就等于拿别的光路状态的相位去拍这一轮。
        np.save(
            out_dir / f"phase_nmax{n_max}.npy",
            panel_phase_from_coefficients(
                fit.coefficients,
                params.beam_radius_px,
                (SLM_PANEL_H, SLM_PANEL_W),
            ),
        )
        if params.save_images:
            _save_frame_png(
                before_frame, out_dir / f"before_nmax{n_max}.png", f"n_max={n_max} fit"
            )
            _save_frame_png(
                after_frame, out_dir / f"after_nmax{n_max}.png", f"n_max={n_max} spgd"
            )

        orders.append(
            {
                "n_max": n_max,
                "n_modes": int(fit.n_modes),
                "fit_rms_rad": float(fit.rms_residual_rad),
                "max_abs_coefficient": float(fit.max_abs_coefficient),
                "clip_warning": run["clip_warning"],
                "before": before,
                "after": after,
                "best_epoch": int(best_index),
                "best_objective": float(best_value),
                "spgd_improved": bool(spgd_improved),
                "spgd_epochs_executed": spgd_epochs_executed,
                "spgd_J_min": float(min(js)) if js else None,
                "spgd_J_max": float(max(js)) if js else None,
                "spgd_J_final": float(js[-1]) if js else None,
            }
        )

    summary = {
        **group_fields(params.group, params.group_meta),
        "config": _jsonable(asdict(params)),
        "cam_size": int(params.cam_size),
        "probe_shape": [int(probe_shape[0]), int(probe_shape[1])],
        "frame_shape": [int(ref_shape[0]), int(ref_shape[1])],
        "center_xy": [int(center_xy[0]), int(center_xy[1])],
        "mask_center_xy": [int(mask_center[0]), int(mask_center[1])],
        "w0_px": float(w0),
        "target_side": float(side),
        "beam_radius_px": float(params.beam_radius_px),
        "best_stage": str(stage_a.best_stage),
        "stage_a": {
            "source": stage_a_source,
            "epochs_executed": int(stage_a.epochs_executed),
            "best_score": float(stage_a.best_score),
            "elapsed_s": float(stage_a.elapsed_s),
        },
        "baselines": baselines,
        "orders": orders,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    # 根目录的分组清单: 每组写完立刻更新 —— 后面那一组崩掉时, 这一组已经被索引。
    update_manifest(
        out_root,
        group_record(
            group=params.group,
            variable=GROUP_VARIABLE,
            meta=params.group_meta,
            dir_name=str(params.group),
            n_rows=len(orders),
        ),
    )
    logger.info(f"[done] wrote {out_dir / 'summary.json'}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cam-type", default="sim", choices=["sim", "daheng", "miicam"])
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--exposure-time-ms", type=float, default=0.0)
    parser.add_argument("--cam-size", type=int, default=256)
    parser.add_argument(
        "--sim-far-field-padding",
        type=int,
        default=DEFAULT_SIM_FAR_FIELD_PADDING,
        help=(
            "Sim far-field FFT zero-padding factor (--cam-type sim only). "
            f"Default {DEFAULT_SIM_FAR_FIELD_PADDING}: the library default 4 gives "
            "a ~1.7 px spot that clamps the target box to its floor."
        ),
    )
    parser.add_argument("--slm-number", type=int, default=1)
    parser.add_argument("--beam-radius-px", type=float, default=450.0)
    parser.add_argument("--epochs", type=int, default=400, help="Stage-A SPGD epochs.")
    parser.add_argument("--gs-iters", type=int, default=200, help="GS iterations.")
    parser.add_argument(
        "--spgd-epochs", type=int, default=200, help="Stage-C SPGD epochs per order."
    )
    parser.add_argument(
        "--n-max",
        type=int,
        nargs="+",
        default=list(DEFAULT_N_MAX),
        help="Zernike truncation orders to sweep.",
    )
    parser.add_argument(
        "--orders",
        default="",
        help=(
            "只处理这几个 n_max (逗号分隔, 如 --orders 3,4,6)。与 --n-max 联用时 "
            "--orders 优先: 分段扫描时每次调用只跑一个阶, 整轮时间才压得下来。"
        ),
    )
    parser.add_argument(
        "--stage-a-only",
        action="store_true",
        help=(
            "只跑 Stage A: 把最佳相位与两张基线帧落盘到 <组目录>/stage_a.npz "
            "(配 stage_a.json), 然后退出 (不再跑 Stage B/C)。"
        ),
    )
    parser.add_argument(
        "--stage-a-in",
        default="",
        help=(
            "读回 Stage A 的产物 (路径给到 stage_a.npz 即可, 同目录的 stage_a.json "
            "自动配对), 跳过平场探测与 Stage A, 直接跑 Stage B/C。"
        ),
    )
    parser.add_argument(
        "--chunk-epochs",
        type=int,
        default=0,
        help=(
            "本块 Stage A 跑多少个精化 epoch (0 = 跑满 --epochs)。"
            "分块续跑用: 每块 <=10 分钟, 但总轮数不变 (学习率按全局 epoch 继续)"
        ),
    )
    parser.add_argument(
        "--time-budget-s",
        type=float,
        default=0.0,
        help=(
            "Stage A 的墙钟预算 (s); 0 = 不限。超预算就提前收工并保留目前的最佳"
            "相位 —— 「一次扫描不超过 10 分钟」由这个参数保证, 而不是靠掐表掐进程。"
        ),
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.02,
        help=(
            "SPGD perturbation amplitude in radians. Only honoured when "
            "--lr is non-zero: with --lr 0 the optimizer's own schedule "
            "overrides it (capped at 0.5 rad, larger than a typical fitted "
            "coefficient, which makes SPGD reject every step)."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Adam learning rate (0 = optimizer auto-schedule, which also overrides --delta).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="",
        help=(
            "Output root directory. The group's artefacts go to <out>/<group>/ "
            "and the root gets a manifest.json indexing every group recorded "
            "under it (default: data/debug/slm_gs_refine_zernike/<timestamp>)."
        ),
    )
    parser.add_argument(
        "--target-side",
        type=float,
        default=0.0,
        help="Explicit target side in px (0 = auto-probe).",
    )
    parser.add_argument(
        "--max-energy-loss",
        type=float,
        default=0.6,
        help=(
            "In-ROI energy-loss fraction that abandons an SPGD evaluation "
            "(0 disables the guard). Lower it if every perturbation of an "
            "already-good fitted phase is being rejected."
        ),
    )
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument(
        "--defocus-focal-mm",
        type=float,
        default=None,
        help=(
            "Pupil Fresnel focus-correction focal length in mm (omit = no "
            "correction). The 2f bench camera is not on the 2f plane, so a "
            "hardware run needs this overlay; ao_shaping.tools.slm.panel."
            "focus_probe measured 1/f = +0.626 m^-1 (f_v = 1597.4 mm) on "
            "2026-10-20 -- a bench measurement, re-measure after moving the "
            "camera."
        ),
    )
    parser.add_argument(
        "--group",
        default="main",
        help=(
            "Identity of the variable group this invocation probes (default "
            "'main'). The sweep writes into <out>/<group>/; the root keeps a "
            "manifest.json indexing every group recorded under it."
        ),
    )
    parser.add_argument(
        "--group-meta",
        dest="group_meta_pairs",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Fixed parameters of this group (repeatable), recorded verbatim in "
            "the group's summary.json and in the root manifest.json, e.g. "
            "--group-meta exposure_ms=0.2 --group-meta phase=hardware_gs"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the sweep."""
    args = _build_parser().parse_args(argv)
    try:
        group_meta = parse_group_meta(args.group_meta_pairs)
    except ValueError as exc:
        logger.info(f"参数错误: {exc}")
        return 2
    # --stage-a-only 与 --stage-a-in **可以**同时给: 那是「续跑这一块 Stage A」
    # (读回上一块的末态, 接着跑, 再把新末态落盘)。两者互斥会让「分块但总强度不变」
    # 做不到 —— 一块跑不完就得整轮重跑, 所以这里放开。
    try:
        orders = parse_orders(args.orders) if args.orders else ()
    except ValueError as exc:
        logger.info(f"参数错误: {exc}")
        return 2
    if args.time_budget_s < 0:
        logger.info(f"参数错误: --time-budget-s 不能为负, 收到 {args.time_budget_s}")
        return 2
    params = SweepParams(
        cam_type=args.cam_type,
        cam_id=args.cam_id,
        exposure_time_ms=args.exposure_time_ms,
        cam_size=args.cam_size,
        sim_far_field_padding=args.sim_far_field_padding,
        slm_number=args.slm_number,
        beam_radius_px=args.beam_radius_px,
        epochs=args.epochs,
        gs_iters=args.gs_iters,
        spgd_epochs=args.spgd_epochs,
        n_max_list=tuple(args.n_max),
        delta=args.delta,
        lr=args.lr,
        seed=args.seed,
        out=args.out,
        target_side=args.target_side,
        max_energy_loss=args.max_energy_loss,
        save_images=args.save_images,
        defocus_focal_mm=args.defocus_focal_mm,
        group=args.group,
        group_meta=group_meta,
        stage_a_only=bool(args.stage_a_only),
        stage_a_in=str(args.stage_a_in),
        orders=orders,
        time_budget_s=float(args.time_budget_s),
        chunk_epochs=int(args.chunk_epochs),
    )
    try:
        run_sweep(params)
    except PathBlockedError as exc:
        # 光路受阻: 继续跑 8 个泽尼克阶次只是在对着死路测噪声, 而且结果看上去
        # 毫无异常。给出操作者要动手的检查清单, 并用**专属**退出码 (3) 与
        # 「参数错误」(2) 区分开。
        logger.info(f"光路受阻: {exc}")
        logger.info("  请先确认 光束→SLM 面板 → 2f 透镜 → CCD 三段光路。")
        return 3
    except (FileNotFoundError, ValueError) as exc:
        # 读不回 Stage A 产物 (路径写错 / 文件损坏) 属于「参数错误」: 那条
        # 命令行本身就跑不起来, 与光路无关。
        logger.info(f"参数错误: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
