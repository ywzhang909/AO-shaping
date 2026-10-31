"""Hardware closed loop for GS-warm-started freeform SLM square shaping.

This is the hardware port of the simulation-only pipeline in
:mod:`ao_shaping.optimizer.wfless.iterative_zernike_shaping`, whose verified
result (2026-10-01) was ``initial 0.662 -> GS 0.812 -> refinement 0.849``.

What survives the port, and what does not
-----------------------------------------
``gs_shape`` is an **open-loop** calculation: it needs a forward *model*, not a
measurement, so it ports unchanged. It supplies the ``0.662 -> 0.812`` step.

The refinement does **not** port. In simulation it differentiates the composite
objective through an analytic far-field FFT with torch autograd; on hardware the
only available forward model *is* the measurement, which has no derivative. The
refinement therefore becomes **sensorless (SPGD)**: perturb the SLM phase, read
two far-field frames, and estimate the gradient from their difference. That is
the standard substitution for a non-differentiable plant, and is what the
existing ``spgd-square`` / ``slm-gsnet`` runners already do.

The objective is deliberately **the same function** the simulation used --
``composite_from_pib_cv(power_in_bucket(...), uniformity_cv(...))`` -- evaluated
on the measured CCD frame instead of a simulated one. That is what makes the
hardware numbers comparable to the simulated 0.849 rather than to some
differently-scaled proxy.

Three hardware-specific decisions that are easy to get wrong
-----------------------------------------------------------
1. **The ROI is located once and then frozen.** On a speckle field the global
   maximum hops between near-equal grains under a ~1e-3 perturbation, so an
   ``argmax``-rolled target box makes PIB/CV discontinuous and the optimizer
   chases a box that no longer covers the beam. We therefore locate the 0-order
   by ``argmax`` on the **unshaped** frame, before any phase is applied, and
   never re-locate it. (See ``report/slm_pib_bench/report.md`` §7.3.)

2. **The GS phase is admitted only if it wins a bake-off.** The GS phase comes
   from a *model* of this bench (aperture, focal length, camera pixel pitch). If
   any of those inputs is wrong, GS can make the real spot *worse*. So we measure
   the flat phase and the GS phase and keep the better one. A mis-calibrated
   bench model then costs two measurements instead of wrecking the run.

3. **Every measurement waits for the panel to settle.** The driver's automatic
   LCOS flip-time estimate is driven by how much the gray map changed, so two
   phases with similar gray statistics make it report **0.0 ms** while the panel
   is still relaxing (observed: the same ramp read fwhm 43.2 px immediately and
   12.8 px three seconds later, centroid shifted 62 px). A single-shot read
   therefore silently records an unsettled frame. We discard frames until two
   consecutive readings agree.

Phase conventions
-----------------
All phases here are **raw unwrapped radians**; nothing in this module does
``mod 2 pi``. The single wrap point is the driver's ``create_phase_from_array``.
Consecutive writes go through ``slm.display_data(gray)`` with **no**
``memory_number``, so the driver rotates memory slots itself -- writing the same
slot twice is a firmware no-op that does not refresh the panel.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, replace
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.algorithm import make_optimizer
from ao_shaping.algorithm.signal_processing.square_shaping import (
    UniformSquareShaper,
    composite_from_pib_cv,
    gs_shape,
    power_in_bucket,
    uniformity_cv,
)
from ao_shaping.drivers.sim.slm_shaping_bench import (
    ShapingBenchConfig,
)
from ao_shaping.drivers.slm.factory import (
    DEFAULT_WAVELENGTH_NM,
    open_camera,
    open_slm,
    require_drive_board_entry,
    sync_device_wavelength,
)
from ao_shaping.gui.utils.phase_overlay import (
    PhaseOverlay,
    apply_overlays_to_handle,
)
from ao_shaping.optimizer.spgd import spgd_gradient
from ao_shaping.optimizer.wfless._measure import display_and_average, locate_zero_order
from ao_shaping.utils.image.exposure_control import (
    DAHENG_EXPOSURE_LIMIT_ENV,
    configured_daheng_exposure_limit,
    resolve_daheng_exposure_limit,
)
from ao_shaping.utils.image.target.patterns import create_target_shape
from ao_shaping.utils.image.target.metrics import TARGET_SHAPE_CHOICES
from ao_shaping.utils.image.targets import square_mask
from ao_shaping.utils.io.file import Recorder
from ao_shaping.utils.slm.bench_constants import (
    BLAZE_PERIOD_MIN_PX,
    MAX_SLM_NUMBER,
    MIN_CAM_SIZE_PX,
    TILT_SHIFT_SCALE as _TILT_SHIFT_SCALE_PX,
)
from ao_shaping.utils.slm.phase_disturbance import load_panel_disturbance
from ao_shaping.utils.slm.phase_embed import (
    compose_pupil_dof,
    embed_pupil_square,
    panel_shape_from_slm,
    pupil_side,
)
from ao_shaping.utils.wavefront.matrix_utils import (
    focal_length_from_camera_pixel,
)
from ao_shaping.utils.io.recorder_keys import (
    RECORD_EPOCH,
    RECORD_IMAGE,
    RECORD_PHASE,
)

__all__ = ["VALIDATED_GS_REFINE_RECIPE", "SlmGsRefineConfig", "optimize_slm_gs_refine"]

# Measured bench constants (src/ao_shaping/drivers/AGENTS.md, 2026-09).
_DEFAULT_PANEL_PIXEL_UM = 8.0
_DEFAULT_BEAM_RADIUS_PX = 450.0
# The measured 2f focal scale for this bench: a 2*pi phase ramp over P SLM pixels
# displaces the spot TILT_SHIFT_SCALE/P camera px. See
# `ao_shaping.tools.slm.slm_bench_probe` (the tilt probe that measures it).
# ``_TILT_SHIFT_SCALE_PX`` is imported from ``ao_shaping.utils.slm.bench_constants``
# (single source of truth); do not re-declare it here.
# The wavelength at which _TILT_SHIFT_SCALE_PX was measured. K = lambda*f/(d*p_cam)
# scales WITH wavelength, so a scale measured at 1064nm must be referred to another
# wavelength before use -- otherwise the derived focal length inherits a spurious
# factor lambda/lambda_ref. Passing this makes the derived focal length
# wavelength-invariant, as a lens focal length must be.
_TILT_SHIFT_SCALE_MEASURED_AT_NM = 1064.0
# The CCD pixel pitch is the measurement ANCHOR, not a derived quantity: it is a
# per-camera datasheet constant (2.2 um for the Daheng MER2-507-23GM, recorded in
# drivers/AGENTS.md), whereas the lens focal length is a bench ASSEMBLY choice
# that changes whenever somebody swaps optics. The measured focal scale only
# constrains the ratio f/p_cam, so exactly one anchor is unavoidable.
#
# The value this replaces, 3.31 um, was a guess implying a focal scale of 5023 --
# 33% below the measured 7400-7600 -- which biased the GS target angular size by
# that same factor.
_DEFAULT_CAMERA_PIXEL_UM = 2.2
# Focal length is DERIVED from the anchor plus the measured scale: on this bench
# (1064 nm, p_cam 2.2 um, d_slm 8 um, scale 7400) that is 0.1224 m, i.e. the
# 125 mm nominal lens to within 2% -- the same 2% by which the three datasheet
# values disagree with the measured scale. Pass ``focal_length_m`` explicitly to
# pin a specific lens instead.
_DEFAULT_FOCAL_LENGTH_M = focal_length_from_camera_pixel(
    wavelength_nm=DEFAULT_WAVELENGTH_NM,
    camera_pixel_um=_DEFAULT_CAMERA_PIXEL_UM,
    slm_pixel_um=_DEFAULT_PANEL_PIXEL_UM,
    focal_scale_px=_TILT_SHIFT_SCALE_PX,
    scale_measured_at_nm=_TILT_SHIFT_SCALE_MEASURED_AT_NM,
)
# The GS far-field pixel pitch is ``lambda*f/(aperture*padding)`` -- independent
# of ``n_grid`` -- so padding only sets how finely the focal plane is sampled,
# at a cost quadratic in it. The simulation used 8 because its ``n_grid`` was
# only 512; on the real bench ``n_grid`` is ~900 (one sample per panel pixel), so
# padding 8 would mean an 7200**2 FFT (GB-scale). Padding 3 still resolves the
# target square with >200 far-field pixels and keeps the FFT at 2700**2.
_DEFAULT_FAR_FIELD_PADDING = 3


def _pitch_backend_label(cam_type: str) -> str:
    """Backend label used in the pixel-pitch provenance phrase.

    ``sim`` is the twin-optics backend, and saying "sim twin" is what tells a
    reader the 2.2 um came from the simulation's own optics rather than from a
    datasheet.
    """
    name = str(cam_type or "").strip().lower()
    return "sim twin" if name == "sim" else name


_STAGE_FLAT = "flat"
_STAGE_GS = "gs"
_STAGE_REFINE = "refine"

#: A target shape covering less than this fraction of the ROI frame is treated as
#: a broken geometry (wrong coordinate frame, or a size that collapsed to a few
#: pixels) rather than as a very small but valid target: PIB/CV over a handful of
#: pixels is pure noise, and the optimizer would chase it for a whole run.
_MIN_TARGET_AREA_FRAC = 1e-3

#: The interleaved-replay validated ``slm-gs-refine`` recipe
#: (``report/shape_trials/gs_refine_live_20261012.md``; re-checked 2026-10-14).
#:
#: This is a **preset**, not a dataclass default: every value here was measured
#: on the 2f experiment system as one interleaved replay, and changing one
#: silently abandons that evidence. The library defaults in
#: :class:`SlmGsRefineConfig` are the generic ones; the GUI's
#: ``GS_REFINE_RECIPE`` is a copy of this table and the CLI defaults are
#: deliberately left alone.
#:
#: ``panel_pixel_um`` is the **pitch** (``PITCH_UM`` = 8.0), not the active-pixel
#: size 7.8: the bench model samples on the panel lattice, and the 0.2 μm gap is
#: real device geometry — using 7.8 shrinks the aperture by 2.5 % and voids the
#: replay evidence below.
VALIDATED_GS_REFINE_RECIPE: dict[str, object] = {
    "target_side": 100,
    "side_factor": 1.0,
    "w_pib": 0.5,
    "w_unif": 0.5,
    "phase_grid": 12,
    "delta": 0.35,
    "confirm_best": True,
    "epochs": 12,
    "seed": 0,
    "gs_warm_start": True,
    "gs_iters": 12,
    "gs_method": "plain",
    "panel_pixel_um": 8.0,
    "camera_pixel_um": 2.2,
    "beam_radius_px": 450.0,
    "pupil_center": "960,690",
    "far_field_padding": 3,
    "cam_type": "daheng",
    "cam_id": 0,
    "exposure_time_ms": 0.05,
    "cam_size": 250,
    "n_eval_frames": 4,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SlmGsRefineConfig:
    """Single-parameter config for :func:`optimize_slm_gs_refine`.

    Devices are opened and closed *inside* the optimizer; reusing a device
    across runs is forbidden (the SDK handles are not re-entrant).
    """

    # --- control -----------------------------------------------------------
    epochs: int = 400
    """SPGD refinement iterations (each costs two far-field reads)."""

    seed: int | None = None
    early_stop_score: float = 0.0
    """Stop once the composite score reaches this (0 = run all epochs)."""

    time_budget_s: float = 0.0
    """Stage A 的墙钟预算 (s); 0 = 不限。超预算就**提前收工**: 保留目前最好的相位,
    把实际跑掉的 epoch 数记进记录器, 让「一次扫描不超过 10 分钟」这条约束由配置保证,
    而不是靠掐表掐进程。"""
    chunk_epochs: int = 0
    """本块要跑多少个**精化** epoch (0 = 用 ``epochs``)。

    「一次扫描不超过 10 分钟」把 Stage A 切成若干块后, ``epochs`` 必须保持
    **总轮数** —— 学习率调度的总地平线就靠它, 改了调度就不是同一条轨迹了。
    本字段只说这一块跑多少轮。
    """

    resume_state: dict[str, Any] | None = None
    """从上一块续跑的状态 (由 :func:`resume_state_from_recorder` 产出)。

    块边界必须能续上: 学习率按**全局** epoch 继续 (不热重启), 最佳值跨块取全局最优,
    基线帧与零级位置沿用。否则 250 轮就被切成 6 段互不相干的优化。
    """

    # --- objective ---------------------------------------------------------
    target_side: int = 0
    """Characteristic target size in **camera** pixels. 0 = derive from the pupil
    image.

    Its per-shape meaning is the one the CLI documents: square edge / circle and
    pentagon diameter / rectangle SHORT side / annular OUTER diameter / grid and
    cross bounding size / gaussian characteristic size (``sigma = ratio*side``).
    """

    side_factor: float = 1.0
    """Multiplier applied to the target side.

    It scales **whichever** size ``target_side`` resolved to: an explicit
    ``target_side`` is multiplied by it, and so is the auto-derived one. There is
    no second, independent way to set the target size.
    """

    target_shape: str = "square"
    """Far-field target shape; a member of ``TARGET_SHAPE_CHOICES``.

    ``square`` (the default) reproduces every earlier run bit for bit. The other
    shapes change both the GS target **and** the measured mask, which is the
    point: scoring a circle with a square box would measure empty corners.
    """

    target_aspect_ratio: float = 2.0
    """``rectangle`` long/short side ratio (long axis horizontal); must be >= 1."""

    target_inner_ratio: float = 0.6
    """``annular`` inner/outer radius ratio; must be in (0, 1)."""

    target_sigma_ratio: float = 0.25
    """``gaussian`` sigma as a fraction of the characteristic size; must be > 0.

    The default 0.25 matches the GS family's own gaussian target (``side/4``), so
    the default gaussian run aims at and scores the same profile.
    """

    target_line_width: float = 0.02
    """``grid``/``cross`` line width as a fraction of the characteristic size."""

    target_grid_nx: int = 5
    """``grid`` number of vertical lines."""

    target_grid_ny: int = 5
    """``grid`` number of horizontal lines."""

    target_thickness: float = 0.05
    """``cross`` line width (both strokes) as a fraction of the characteristic size."""

    w_pib: float = 0.5
    w_unif: float = 0.5
    """Composite weights; the default reproduces the simulated objective."""
    exclude_zero_order_for_cv: bool = False
    """Use the flat-reference zero-order core as a fixed CV exclusion."""

    # --- freeform parameterisation ----------------------------------------
    phase_grid: int = 24
    """Edge length of the coarse freeform phase grid; DOF = ``phase_grid**2``.

    The panel has ~2.3 M pixels; perturbing all of them makes the SPGD gradient
    estimate pure noise. 24x24 = 576 DOF is what ``spgd-square --basis freeform``
    uses and is enough to synthesise a square.
    """

    delta: float = 0.35
    """SPGD perturbation amplitude in radians on the coarse grid."""

    confirm_best: bool = False

    lr: float = 0.0
    """Adam learning rate (0 = auto -> ``0.15 * delta``)."""

    optimizer_type: str = "adam"
    lr_schedule: str = "cosine"
    lr_min_ratio: float = 0.05
    """Fraction of the initial LR kept at the final epoch."""

    # --- GS warm start -----------------------------------------------------
    gs_iters: int = 200
    gs_warm_start: bool = True
    gs_relax: float = 1.0
    gs_method: str = "plain"
    """GS 变体: plain / weighted / hio / hio-weighted / adaptive-feedback。"""
    gs_gamma: float = 0.5
    """加权 GS 累积反馈系数 γ, 0≤γ<1。γ=0 是退化哨兵 (权重恒为 1 ⇒ 与 plain GS 等价)。"""
    gs_weight_clip: float = 4.0
    """加权 GS 权重上限 (裁剪, 须 >1)。"""
    gs_beta: float = 0.9
    """HIO 松弛系数或自适应目标域反馈上限 (0≤β≤1)。"""
    gs_hio_fraction: float = 0.7
    """hio-weighted 前段 HIO 迭代占比 (0≤f≤1)。"""

    # --- bench model (only used to compute the GS phase) -------------------
    panel_pixel_um: float = _DEFAULT_PANEL_PIXEL_UM
    camera_pixel_um: float = _DEFAULT_CAMERA_PIXEL_UM
    """Camera pixel pitch -- the bench model's measurement anchor. **Verify this
    against your camera** (2.2 um for the Daheng MER2-507-23GM). The measured
    focal scale fixes only the ratio ``f/p_cam``, so one anchor is unavoidable;
    this is the one we chose because it is a datasheet constant rather than a
    bench assembly choice."""

    beam_radius_px: float = _DEFAULT_BEAM_RADIUS_PX
    """Illuminated pupil radius on the panel, in panel pixels."""
    pupil_center: tuple[int, int] | None = None

    focal_length_m: float = 0.0
    """2f lens focal length in metres. 0 (the default) derives it from
    ``camera_pixel_um`` and the measured focal scale; pass a value to pin a
    specific lens."""
    far_field_padding: int = _DEFAULT_FAR_FIELD_PADDING

    # --- hardware ----------------------------------------------------------
    cam_type: str = "daheng"
    cam_id: int = 0
    exposure_time_ms: float = 0.0
    """0 = keep the device default."""
    cam_size: int = 300
    slm_number: int = 1
    phase_disturbance_file: str | None = None
    """Fixed raw-radian panel bias added to every displayed phase."""
    blaze_period_px: int = -1
    """-1 disables the carrier, 0 probes candidates, >=4 probes one period."""
    blaze_direction: str = "vertical"
    slm_wavelength: int = 0
    """SLM operating wavelength in nm. 0 (the default) asks the DEVICE for its
    configured wavelength instead of assuming this bench's 1064nm -- the lab has
    more than one SLM (see the serial-number conflict in drivers/AGENTS.md), and
    a wrong wavelength reprograms the panel's phase table."""

    n_eval_frames: int = 4
    settle_wait_s: float = 0.5
    settle_tol: float = 0.02
    settle_max_wait_s: float = 6.0
    settle_max_discard: int = 40

    # --- send-pipeline overlay (GUI 离焦 / 底相位 T4 叠加层) ---------------
    send_overlay: PhaseOverlay | None = field(default=None, repr=False)
    """Send-pipeline overlay (defocus + base phase) to apply to the SLM this
    run opens. ``None`` (default) = no overlay, so CLI runs are untouched. The
    GUI entry fills this from the current session via ``build_send_overlay``;
    :func:`apply_overlays_to_handle` writes it onto the freshly-opened handle
    (driver slot for a real Santec, no-op for a sim SLM that lacks the setters).
    """

    # --- bookkeeping -------------------------------------------------------
    callback: Callable[[int, float], None] | None = field(default=None, repr=False)
    progress_every: int = 10
    record_debug: bool = False

    def __post_init__(self) -> None:
        """Resolve the derived bench-model quantities once, here.

        Doing it in ``__post_init__`` rather than at each use site means the
        runner, the library caller and the tests all see the same resolved
        config, and no consumer can accidentally use the raw 0.

        The wavelength is not known yet (the SLM is opened later, and 0 means
        "ask the device"), so this uses the bench fallback for the geometry and
        :func:`sync_device_wavelength` recomputes it from the device's answer.
        """
        self._focal_length_pinned = self.focal_length_m > 0.0
        if not self._focal_length_pinned:
            self.focal_length_m = self._derive_focal_length()

        # --- range validation (moved here from the optimizer body so a bad
        # value never reaches the hardware context managers) ---
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if self.blaze_period_px < -1 or 0 < self.blaze_period_px < BLAZE_PERIOD_MIN_PX:
            raise ValueError(
                f"blaze_period_px must be -1 (off), 0 (auto), or >={BLAZE_PERIOD_MIN_PX}"
            )
        if self.blaze_direction not in ("vertical", "horizontal"):
            raise ValueError("blaze_direction must be vertical or horizontal")
        if self.phase_grid < 2:
            raise ValueError(f"phase_grid must be at least 2, got {self.phase_grid}")
        if self.delta < 0:
            raise ValueError(f"delta must be non-negative, got {self.delta}")
        if self.n_eval_frames < 1:
            raise ValueError(
                f"n_eval_frames must be at least 1, got {self.n_eval_frames}"
            )
        if self.gs_iters < 0 or (self.gs_warm_start and self.gs_iters == 0):
            raise ValueError(
                f"gs_iters must be positive when GS is enabled, got {self.gs_iters}"
            )
        if self.gs_relax <= 0:
            raise ValueError(f"gs_relax must be positive, got {self.gs_relax}")
        if self.gs_method not in (
            "plain",
            "weighted",
            "hio",
            "hio-weighted",
            "adaptive-feedback",
        ):
            raise ValueError(
                "gs_method must be one of 'plain', 'weighted', 'hio', "
                f"'hio-weighted', 'adaptive-feedback', got {self.gs_method!r}"
            )
        # γ≥1 makes the cumulative feedback unit-gain and it oscillates; γ=0 is a
        # deliberate degenerate sentinel (weight stays 1 ⇒ identical to plain GS)
        if not (0.0 <= self.gs_gamma < 1.0):
            raise ValueError(f"gs_gamma must be in [0, 1), got {self.gs_gamma}")
        if self.gs_weight_clip <= 1.0:
            raise ValueError(f"gs_weight_clip must be > 1.0, got {self.gs_weight_clip}")
        # Fienup relaxation range; β=0 freezes the noise region
        if not (0.0 <= self.gs_beta <= 1.0):
            raise ValueError(f"gs_beta must be in [0, 1], got {self.gs_beta}")
        if not (0.0 <= self.gs_hio_fraction <= 1.0):
            raise ValueError(
                f"gs_hio_fraction must be in [0, 1], got {self.gs_hio_fraction}"
            )
        if self.w_pib < 0 or self.w_unif < 0:
            raise ValueError(
                f"w_pib and w_unif must be non-negative, got "
                f"w_pib={self.w_pib}, w_unif={self.w_unif}"
            )
        if self.w_pib <= 0 and self.w_unif <= 0:
            raise ValueError("at least one of w_pib / w_unif must be positive")
        if self.settle_wait_s <= 0:
            raise ValueError(
                f"settle_wait_s must be positive, got {self.settle_wait_s}"
            )
        if self.settle_tol <= 0:
            raise ValueError(f"settle_tol must be positive, got {self.settle_tol}")
        if self.settle_max_wait_s <= 0:
            raise ValueError(
                f"settle_max_wait_s must be positive, got {self.settle_max_wait_s}"
            )
        if self.settle_max_discard < 1:
            raise ValueError(
                f"settle_max_discard must be at least 1, got {self.settle_max_discard}"
            )
        if self.side_factor <= 0:
            raise ValueError(f"side_factor must be positive, got {self.side_factor}")
        if self.target_side < 0:
            raise ValueError(
                f"target_side must be non-negative, got {self.target_side}"
            )
        if self.target_shape not in TARGET_SHAPE_CHOICES:
            raise ValueError(
                f"target_shape must be one of {TARGET_SHAPE_CHOICES}, "
                f"got {self.target_shape!r}"
            )
        if not (np.isfinite(self.target_aspect_ratio) and self.target_aspect_ratio >= 1):
            raise ValueError(
                f"target_aspect_ratio must be >= 1 and finite, got "
                f"{self.target_aspect_ratio}"
            )
        if not (0.0 < self.target_inner_ratio < 1.0):
            raise ValueError(
                f"target_inner_ratio must be in (0, 1), got {self.target_inner_ratio}"
            )
        if not (np.isfinite(self.target_sigma_ratio) and self.target_sigma_ratio > 0):
            raise ValueError(
                f"target_sigma_ratio must be positive and finite, got "
                f"{self.target_sigma_ratio}"
            )
        if not (np.isfinite(self.target_line_width) and self.target_line_width > 0):
            raise ValueError(
                f"target_line_width must be positive and finite, got "
                f"{self.target_line_width}"
            )
        if not (np.isfinite(self.target_thickness) and self.target_thickness > 0):
            raise ValueError(
                f"target_thickness must be positive and finite, got "
                f"{self.target_thickness}"
            )
        if self.target_grid_nx < 1:
            raise ValueError(
                f"target_grid_nx must be at least 1, got {self.target_grid_nx}"
            )
        if self.target_grid_ny < 1:
            raise ValueError(
                f"target_grid_ny must be at least 1, got {self.target_grid_ny}"
            )
        if self.panel_pixel_um <= 0:
            raise ValueError(
                f"panel_pixel_um must be positive, got {self.panel_pixel_um}"
            )
        if self.camera_pixel_um <= 0:
            raise ValueError(
                f"camera_pixel_um must be positive, got {self.camera_pixel_um}"
            )
        if self.beam_radius_px <= 0:
            raise ValueError(
                f"beam_radius_px must be positive, got {self.beam_radius_px}"
            )
        if self.far_field_padding < 1:
            raise ValueError(
                f"far_field_padding must be at least 1, got {self.far_field_padding}"
            )
        if self.cam_size < 0:
            raise ValueError(f"cam_size must be non-negative, got {self.cam_size}")
        if 0 < self.cam_size < MIN_CAM_SIZE_PX:
            raise ValueError(
                f"cam_size must be >={MIN_CAM_SIZE_PX} or 0 (full frame), "
                f"got {self.cam_size}"
            )
        if self.slm_number < 1 or self.slm_number > MAX_SLM_NUMBER:
            raise ValueError(
                f"slm_number must be in 1..{MAX_SLM_NUMBER}, got {self.slm_number}"
            )
        if self.slm_wavelength < 0:
            raise ValueError(
                f"slm_wavelength must be non-negative, got {self.slm_wavelength}"
            )
        if self.exposure_time_ms < 0:
            raise ValueError(
                f"exposure_time_ms must be non-negative, got {self.exposure_time_ms}"
            )
        if self.cam_type == "daheng" and self.exposure_time_ms > 0:
            ceiling = configured_daheng_exposure_limit()
            if ceiling > 0 and self.exposure_time_ms > ceiling:
                raise ValueError(
                    f"Daheng exposure_time_ms {self.exposure_time_ms} ms exceeds "
                    f"the {ceiling} ms ceiling ({DAHENG_EXPOSURE_LIMIT_ENV})"
                )
        if self.early_stop_score < 0:
            raise ValueError(
                f"early_stop_score must be non-negative, got {self.early_stop_score}"
            )
        if self.time_budget_s < 0:
            raise ValueError(
                f"time_budget_s must be non-negative, got {self.time_budget_s}"
            )
        if self.progress_every < 1:
            raise ValueError(
                f"progress_every must be at least 1, got {self.progress_every}"
            )
        if self.lr_min_ratio < 0 or self.lr_min_ratio > 1:
            raise ValueError(f"lr_min_ratio must be in 0..1, got {self.lr_min_ratio}")

    def _derive_focal_length(self) -> float:
        """Derive the 2f focal length from the anchor and the measured scale."""
        return focal_length_from_camera_pixel(
            wavelength_nm=float(self.slm_wavelength or DEFAULT_WAVELENGTH_NM),
            camera_pixel_um=self.camera_pixel_um,
            slm_pixel_um=self.panel_pixel_um,
            focal_scale_px=_TILT_SHIFT_SCALE_PX,
            scale_measured_at_nm=_TILT_SHIFT_SCALE_MEASURED_AT_NM,
        )


# ---------------------------------------------------------------------------
# Objective metrics
#
# Geometry and image helpers (panel shape, pupil side, square mask, phase
# embedding, zero-order location) are deliberately NOT re-implemented here: this
# module already imports them from ``utils.slm.phase_embed``,
# ``utils.image.targets``, ``utils.image.resample`` and
# ``optimizer.wfless._measure``. Local copies of those lived here once and
# drifted (``_embed_pupil_square`` clamped where the canonical raises), so they
# were deleted rather than kept in sync.
# ---------------------------------------------------------------------------


def _metrics(
    frame: np.ndarray,
    mask: np.ndarray,
    w_pib: float,
    w_unif: float,
    cv_mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return ``(composite, pib, cv)`` for a measured frame.

    ``composite`` is the simulated objective, evaluated on real pixels.
    """
    pib = power_in_bucket(frame, mask)
    cv = uniformity_cv(frame, mask if cv_mask is None else cv_mask)
    return composite_from_pib_cv(pib, cv, w_pib=w_pib, w_unif=w_unif), pib, cv


# ---------------------------------------------------------------------------
# Target shape geometry
#
# One resolved geometry feeds BOTH the GS target (bench far-field pixels) and the
# measured mask (camera pixels), so the phase cannot aim at one shape while the
# objective scores another. The pattern rasteriser itself is never re-implemented
# here -- every shape comes from the canonical
# ``utils.image.target.patterns.create_target_shape``.
# ---------------------------------------------------------------------------

#: The shapes the GS family expresses natively through
#: ``SquareShapingConfig.target_kind``. The CLI spelling ``annular`` maps onto the
#: algorithm layer's ``annulus``: the two names must never drift apart, so the
#: mapping lives in exactly this one dict (a test asserts every entry).
_GS_NATIVE_TARGET_KINDS: dict[str, str] = {
    "square": "square",
    "circle": "circle",
    "gaussian": "gaussian",
    "annular": "annulus",
}

#: Intensity contour that defines the *measured support* of a gaussian target:
#: the 1/e^2 radius (``r = 2 sigma``), the standard beam-width convention. A
#: gaussian pattern never reaches exactly zero in float32, so a plain ``> 0``
#: test would make the entire frame the target and reduce PIB to a total-power
#: ratio while CV averages over empty pixels.
_GAUSSIAN_SUPPORT_LEVEL = float(np.exp(-2.0))


@dataclass(frozen=True)
class TargetGeometry:
    """One shape's resolved geometry, in pixels of the grid it is rasterised on.

    All lengths are in that grid's pixels. The same numbers describe the target
    in **camera** pixels for the measured mask and in **bench far-field** pixels
    for the GS phase, which is exactly what keeps the two on one shape.
    """

    side: float
    aspect_ratio: float
    inner_radius: float
    outer_radius: float
    sigma: float
    line_width_px: float
    thickness_px: float
    nx: int
    ny: int


def target_geometry(config: SlmGsRefineConfig, side: float) -> TargetGeometry:
    """Resolve ``--target-shape``'s parameters for characteristic size ``side``.

    ``side`` keeps its documented per-shape meaning: square edge / circle and
    pentagon diameter / rectangle SHORT side / annular OUTER diameter / grid and
    cross bounding size / gaussian characteristic size (``sigma = ratio*side``).
    The aspect ratio, ring ratio, sigma ratio, line width and cross thickness are
    all expressed *relative to* that one size, so every shape is compared at the
    same feature scale.

    Args:
        config: The run configuration supplying the shape parameters.
        side: Characteristic size in pixels of the target grid.

    Returns:
        The resolved geometry.

    Raises:
        ValueError: If ``side`` is not positive.
    """
    if not np.isfinite(side) or side <= 0:
        raise ValueError(f"target characteristic size must be positive, got {side!r}")
    outer = float(side) / 2.0
    return TargetGeometry(
        side=float(side),
        aspect_ratio=float(config.target_aspect_ratio),
        inner_radius=float(config.target_inner_ratio) * outer,
        outer_radius=outer,
        sigma=float(config.target_sigma_ratio) * float(side),
        line_width_px=float(config.target_line_width) * float(side),
        thickness_px=float(config.target_thickness) * float(side),
        nx=int(config.target_grid_nx),
        ny=int(config.target_grid_ny),
    )


def _bench_probe(config: SlmGsRefineConfig, radius_px: float) -> ShapingBenchConfig:
    """The GS forward-model bench config for this run's pupil and optics.

    Single construction site: :func:`_gs_pupil_phase` aims this probe at the
    target shape, and :func:`_prepare_refine_context` reads its
    ``far_field_pixel_size`` to report the derived bench-space target side. Both
    must see the same optics, so the probe is built once here rather than twice.
    """
    n_grid = pupil_side(radius_px)
    return ShapingBenchConfig(
        n_grid=n_grid,
        aperture_size=n_grid * config.panel_pixel_um * 1e-6,
        wavelength=config.slm_wavelength * 1e-9,
        focal_length=config.focal_length_m,
        far_field_padding=config.far_field_padding,
        target_side_px=8,
        seed=config.seed if config.seed is not None else 0,
    )


def bench_target_side_px(
    camera_side_px: int,
    *,
    camera_pixel_um: float,
    far_field_pixel_size: float,
) -> int:
    """Convert a **camera**-pixel target side to the model's far-field pixels.

    GS constrains a target expressed in the forward model's own far-field grid,
    while ``target_side`` is documented in camera pixels. The two lattices differ
    by ``p_cam / far_field_pixel_size``, and that ratio is the whole conversion --
    there is exactly one place that may set the bench-space side.

    Args:
        camera_side_px: Target side in camera pixels.
        camera_pixel_um: Camera pixel pitch (the measurement anchor).
        far_field_pixel_size: Model far-field pixel pitch in metres.

    Returns:
        The side in model far-field pixels, at least 3 (below that the target
        cannot be rasterised).
    """
    cam_pitch_m = max(camera_pixel_um, 1e-6) * 1e-6
    return max(3, int(round(camera_side_px * cam_pitch_m / far_field_pixel_size)))


def _rasterise_target(
    shape: str,
    grid_shape: tuple[int, int],
    cy: float,
    cx: float,
    geom: TargetGeometry,
) -> np.ndarray:
    """Rasterise ``shape`` on a ``grid_shape`` canvas centred on ``(cy, cx)``.

    The geometry is **not** re-implemented here: every pattern comes from
    :func:`~ao_shaping.utils.image.target.patterns.create_target_shape`, the
    canonical factory. This function only (a) converts the resolved
    :class:`TargetGeometry` into that factory's kwargs, and (b) gives ``grid`` /
    ``cross`` a ``side``-sized canvas, because the factory draws those two across
    the *whole* array while ``--target-side`` is documented as their bounding
    size. Rasterising on the bounding canvas and placing it is a crop, not a
    second geometry.

    Args:
        shape: A member of ``TARGET_SHAPE_CHOICES``.
        grid_shape: ``(h, w)`` of the output canvas.
        cy: Centre row (``y``) in canvas pixels.
        cx: Centre column (``x``) in canvas pixels.
        geom: The resolved geometry for this canvas.

    Returns:
        ``(h, w)`` float32 intensity pattern in ``[0, 1]``.
    """
    height, width = int(grid_shape[0]), int(grid_shape[1])
    m = float(min(height, width))
    if shape in {"grid", "cross"}:
        # Bounding canvas: the factory's line positions span its whole array, so
        # the canvas edge IS the bounding size. The factory's `line_width` /
        # `thickness` are fractions of the canvas half-size, and its stroke width
        # is twice the half-width -- so passing the pixel width divided by the
        # canvas edge gives exactly `geom.line_width_px` / `geom.thickness_px`.
        canvas = max(3, int(round(geom.side)))
        pattern = create_target_shape(
            shape,
            (canvas, canvas),
            line_width=geom.line_width_px / canvas,
            thickness=geom.thickness_px / canvas,
            nx=geom.nx,
            ny=geom.ny,
        )
        out = np.zeros((height, width), dtype=np.float32)
        y0 = int(round(cy)) - canvas // 2
        x0 = int(round(cx)) - canvas // 2
        ys, ye = max(0, y0), min(height, y0 + canvas)
        xs, xe = max(0, x0), min(width, x0 + canvas)
        if ye > ys and xe > xs:
            out[ys:ye, xs:xe] = pattern[ys - y0 : ye - y0, xs - x0 : xe - x0]
        return out
    if shape == "square":
        return create_target_shape(
            shape, (height, width), side=geom.side, center=(cx, cy)
        )
    if shape in {"circle", "spot", "pentagon"}:
        # Both are sized by a radius = side/2 -> radius_ratio = side / min(h, w).
        return create_target_shape(
            shape,
            (height, width),
            radius_ratio=geom.side / m,
            center=(cx, cy),
        )
    if shape == "gaussian":
        # The factory takes sigma = radius/2 with radius = ratio*min/2.
        return create_target_shape(
            shape,
            (height, width),
            radius_ratio=4.0 * geom.sigma / m,
            center=(cx, cy),
        )
    if shape == "annular":
        return create_target_shape(
            shape,
            (height, width),
            inner_radius=geom.inner_radius,
            outer_radius=geom.outer_radius,
            center=(cx, cy),
        )
    if shape == "rectangle":
        return create_target_shape(
            shape,
            (height, width),
            side=geom.side,
            aspect_ratio=geom.aspect_ratio,
            center=(cx, cy),
        )
    # grid / cross are handled above; anything left is a genuine unknown shape and
    # the factory is the authority on the message.
    return create_target_shape(shape, (height, width), side=geom.side, center=(cx, cy))


def _target_mask(
    frame: np.ndarray,
    cy: int,
    cx: int,
    side: int,
    config: SlmGsRefineConfig,
) -> np.ndarray:
    """Build the frozen target **shape** mask, or say which frame the centre came from.

    框中心必须是 ``frame`` 的**窗口内**坐标: ``frame`` 是相机裁出来的 ROI 窗口
    (通常是 256x256, 零级在 ~128)。而**全幅**传感器坐标 (未裁剪画面上的零级,
    例如 2592x1944 传感器上的 (1434, 973)) 落在窗口之外, 框就整个跑到画面外 ——
    2026-10-21 实测的分块续跑失败正是这么来的, 而报错只有一句
    "target square ... is empty", 两个坐标系被静默混用, 根本看不出是哪一处。

    所以这里把形状、特征尺寸、两个坐标系与画幅尺寸都写进报错里。

    掩膜按 ``--target-shape`` 生成 (形状掩膜, 不是方框切片):

    * ``square`` 走 :func:`~ao_shaping.utils.image.targets.square_mask` —— 它的
      半像素边界约定 (``round(cy ± side/2)`` 切片) 是历史口径, 与形状工厂的
      "``|x-cx| <= half`` 含端点" 差一个像素; 方斑必须逐位不变, 所以这里显式
      委托, 由测试锁定两者相同。
    * 其余形状由 :func:`_rasterise_target` 出图案后取阈值。``gaussian`` 取 1/e²
      等强线 (``r <= 2 sigma``, 严格大于判据), 其余取 ``> 0``。

    形状掩膜的意义: 对方斑以外的形状, **方框切片会把框内的空白像素算进 CV 与
    PIB 的分母** —— cross / grid / annular 的框内大部分是空的, 于是 CV 被空白
    拉高、PIB 被空白稀释, 优化器去追一个它其实无法填满的框。
    """
    geom = target_geometry(config, side)
    shape = config.target_shape
    if shape == "square":
        mask = square_mask(frame.shape, cy, cx, side)
    else:
        pattern = _rasterise_target(shape, frame.shape, cy, cx, geom)
        level = _GAUSSIAN_SUPPORT_LEVEL if shape == "gaussian" else 0.0
        mask = pattern > level
    area_frac = float(mask.mean()) if mask.size else 0.0
    if mask.any() and area_frac >= _MIN_TARGET_AREA_FRAC:
        return mask
    h, w = frame.shape[0], frame.shape[1]
    raise ValueError(
        f"target shape {shape!r} (characteristic size={side}px at window-local "
        f"({cy},{cx}), aspect={geom.aspect_ratio}, inner={geom.inner_radius:.1f}, "
        f"outer={geom.outer_radius:.1f}, sigma={geom.sigma:.1f}, "
        f"nx={geom.nx}, ny={geom.ny}, line_width={geom.line_width_px:.1f}px, "
        f"thickness={geom.thickness_px:.1f}px) covers "
        f"{area_frac * 100.0:.3f}% of the {h}x{w} ROI frame: the mask centre must "
        f"be a WINDOW-LOCAL coordinate (inside the cropped ROI, 0..{min(h, w) - 1}), "
        f"not a FULL-FRAME sensor coordinate (e.g. (1434, 973) on a 2592x1944 "
        f"sensor), and the shape must be large enough to cover at least "
        f"{_MIN_TARGET_AREA_FRAC * 100.0:.1f}% of it. The two coordinate frames "
        f"must not be mixed."
    )


def _camera_frame_shape(cam: Any) -> tuple[int, int] | None:
    """相机当前窗口的尺寸 ``(h, w)``; 拿不到就返回 ``None``。

    只用于「零级离传感器边缘太近」这个检查 —— 续跑块跳过了零级重定位, 手上
    没有全幅画面, 只能问相机自己的尺寸。相机刚打开时窗口就是全幅, 所以这里读
    到的就是传感器尺寸。
    """
    width = int(getattr(cam, "cam_width", 0) or 0)
    height = int(getattr(cam, "cam_height", 0) or 0)
    if width > 0 and height > 0:
        return height, width
    return None


def _resume_full_frame_center(
    resume: dict[str, Any] | None,
) -> tuple[int, int] | None:
    """续跑状态里带的**全幅**零级位置 ``(x, y)``; 没有就 ``None``。

    有它 ⇒ 下一块把 ROI 窗口摆在同一个位置 (冻结 ROI 的前提); 没有 ⇒ 照旧
    重新定位零级。
    """
    if not resume:
        return None
    raw = resume.get("full_frame_center")
    if raw is None:
        return None
    try:
        return int(raw[0]), int(raw[1])
    except (TypeError, ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# GS warm start
# ---------------------------------------------------------------------------


def _derive_target_side(config: SlmGsRefineConfig, radius_px: float) -> int:
    """Target square side in camera pixels.

    An explicit ``target_side`` wins. Otherwise use the geometric image of the
    illuminated pupil, ``D * f / (d_SLM * p_cam)`` -- the size a flat phase
    naturally produces, which is a sensible starting target.
    """
    factor = max(config.side_factor, 1e-6)
    if config.target_side > 0:
        return max(3, int(round(config.target_side * factor)))
    aperture_m = 2.0 * radius_px * config.panel_pixel_um * 1e-6
    cam_pitch_m = max(config.camera_pixel_um, 1e-6) * 1e-6
    return max(3, int(round(aperture_m * config.focal_length_m / cam_pitch_m * factor)))


def _gs_target_config(
    config: SlmGsRefineConfig, probe: ShapingBenchConfig, bench_side: int
) -> ShapingBenchConfig:
    """Aim the GS forward model at ``config.target_shape``.

    Two disjoint routes, and both end in the **same** GS loop:

    * shapes the algorithm layer expresses natively (``square`` / ``circle`` /
      ``gaussian`` / ``annular``) set ``target_kind`` -- plus the resolved sigma
      and ring radii, because the algorithm layer's own defaults are a fixed
      ``side/4`` sigma and a fixed 1:3 ring at ``side/4 .. 3*side/4``, which is
      not the geometry ``--target-side`` documents (outer diameter = side,
      inner = ``--target-inner-ratio`` of it). Without the override GS would aim
      at one ring while the measured mask scored another.
    * the rest (``rectangle`` / ``grid`` / ``cross`` / ``pentagon``) inject the
      far-field amplitude rasterised by the canonical target factory
      (``SquareShapingConfig.target_amplitude``). No GS loop and no geometry is
      copied into this module: the same :func:`_rasterise_target` that builds the
      measured mask builds this amplitude.

    ``target_shape == "square"`` returns ``probe`` untouched, so the historical
    square path stays bit-for-bit what it always was.
    """
    shape = config.target_shape
    if shape == "square":
        return probe
    kind = _GS_NATIVE_TARGET_KINDS.get(shape)
    geom = target_geometry(config, bench_side)
    if kind is not None:
        overrides: dict[str, Any] = {"target_kind": kind}
        if kind == "gaussian":
            overrides["target_gaussian_sigma_px"] = geom.sigma
        elif kind == "annulus":
            overrides["target_inner_radius_px"] = geom.inner_radius
            overrides["target_outer_radius_px"] = geom.outer_radius
        return replace(probe, **overrides)
    far_field = probe.far_field_size
    centre = (far_field // 2, far_field // 2)
    pattern = _rasterise_target(shape, (far_field, far_field), centre[1], centre[0], geom)
    return replace(probe, target_amplitude=np.sqrt(pattern.astype(np.float64)))


def _gs_pupil_phase(
    config: SlmGsRefineConfig, radius_px: float, target_side_cam: int
) -> np.ndarray | None:
    """Compute the GS warm-start phase on the pupil square (raw radians).

    ``config.gs_method`` selects the open-loop GS variant. ``"plain"`` keeps the
    historical path byte-for-byte (it calls :func:`gs_shape` directly); the other
    variants route through :class:`UniformSquareShaper` for weighted feedback,
    Fienup HIO, or adaptive target-domain feedback. Their measured uniformity
    must be compared against plain GS before adoption.

    ``config.target_shape`` aims the target at that shape (see
    :func:`_gs_target_config`); ``"square"`` is the untouched default.

    Returns ``None`` when the warm start is disabled or GS degenerates, in which
    case the caller simply proceeds from the flat phase.
    """
    if not config.gs_warm_start or config.gs_iters <= 0:
        return None

    n_grid = pupil_side(radius_px)
    seed = config.seed if config.seed is not None else 0
    aperture_m = n_grid * config.panel_pixel_um * 1e-6
    probe = _bench_probe(config, radius_px)
    # GS constrains a target expressed in *bench* far-field pixels, so the
    # camera-pixel target side must be converted before GS aims at it.
    bench_side = bench_target_side_px(
        target_side_cam,
        camera_pixel_um=config.camera_pixel_um,
        far_field_pixel_size=probe.far_field_pixel_size,
    )
    cfg = _gs_target_config(config, replace(probe, target_side_px=bench_side), bench_side)

    logger.info(
        "GS warm start: shape={} n_grid={} aperture={:.3f}mm bench_target={}px "
        "(camera side {}px @ {:.2f}um/px)",
        config.target_shape,
        n_grid,
        aperture_m * 1e3,
        bench_side,
        target_side_cam,
        config.camera_pixel_um,
    )
    try:
        if config.gs_method == "plain":
            # Historical path, deliberately NOT routed through the new class:
            # this is the default and must stay bit-for-bit what it always was.
            phase = gs_shape(
                cfg, n_iters=config.gs_iters, relax=config.gs_relax, seed=seed
            ).phase
        else:
            logger.info(
                "GS strategy={} gamma={} weight_clip={} beta={} hio_fraction={}",
                config.gs_method,
                config.gs_gamma,
                config.gs_weight_clip,
                config.gs_beta,
                config.gs_hio_fraction,
            )
            phase = (
                UniformSquareShaper(
                    cfg,
                    n_iters=config.gs_iters,
                    method=config.gs_method,
                    gamma=config.gs_gamma,
                    weight_clip=config.gs_weight_clip,
                    beta=config.gs_beta,
                    hio_fraction=config.gs_hio_fraction,
                    relax=config.gs_relax,
                    seed=seed,
                )
                .run()
                .phase
            )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        logger.warning("GS warm start failed ({}); continuing from flat", exc)
        return None

    phase = np.asarray(phase, dtype=np.float64)
    if not np.isfinite(phase).all():
        logger.warning("GS phase is non-finite; discarding warm start")
        return None
    return phase


# ---------------------------------------------------------------------------
# Optimizer / LR helpers
# ---------------------------------------------------------------------------


def _lr_at(config: SlmGsRefineConfig, epoch: int) -> float:
    """Learning rate for ``epoch``, optionally cosine-decayed."""
    base = config.lr if config.lr > 0 else 0.15 * max(config.delta, 1e-6)
    if config.lr_schedule != "cosine" or config.epochs <= 1:
        return base
    frac = epoch / max(1, config.epochs - 1)
    floor = config.lr_min_ratio
    return base * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * frac)))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def resume_state_from_recorder(
    recorder: Any,
    *,
    flat_frame: np.ndarray,
    center_y: int | None = None,
    center_x: int | None = None,
) -> dict[str, Any]:
    """把一块的末态打包成「下一块的起点」。

    「一次扫描不超过 10 分钟」把 Stage A 切成若干块后, 每一块都要从上一块的
    末态继续 —— 否则块边界会重置学习率调度, 250 轮就变成 6 段互不相干的优化。

    ⚠️ 打包的是**未叠加**的相位: 零级叠加是显示时加的 (驱动层), 进了状态下一块
    就会叠加两次。焦面矫正走的是**另一个通道** (``set_phase_overlay``), 本来就不在
    相位里, 所以不用管。

    Args:
        recorder: 一块跑完的 :class:`Recorder`。
        flat_frame: 平场基线帧 (续跑时沿用, 免得再测一次)。
        center_y: 零级**行**坐标 —— 必须是**窗口内**坐标 (平场帧所在的那张 ROI
            窗口的坐标系)。``None`` ⇒ 取记录器里冻结的窗口内零级位置。
        center_x: 零级**列**坐标 (同上)。

    Returns:
        可直接塞进 :attr:`SlmGsRefineConfig.resume_state` 的字典。
    """
    if center_y is None or center_x is None:
        # 默认取记录器冻结的**窗口内**零级位置 —— 全幅坐标不能当框中心用。
        center_y = int(getattr(recorder, "center_y", 0))
        center_x = int(getattr(recorder, "center_x", 0))
    full_center = getattr(recorder, "full_frame_center", None)
    return {
        "base_panel": np.asarray(recorder.base_panel, dtype=np.float64),
        "dof": np.asarray(recorder.dof, dtype=np.float64),
        "flat_frame": np.asarray(flat_frame, dtype=np.float64),
        "center_y": int(center_y),
        "center_x": int(center_x),
        # 全幅零级位置: 下一块据此摆**同一个** ROI 窗口 (冻结 ROI 的前提)。
        # 没有它 (仿真路径 / 旧版交接文件) ⇒ 下一块照旧重新定位。
        "full_frame_center": (
            None if full_center is None else (int(full_center[0]), int(full_center[1]))
        ),
        "best_score": float(recorder.best_score),
        "best_phase": np.asarray(recorder.best_phase_plain, dtype=np.float64),
        "best_frame": np.asarray(recorder.best_frame, dtype=np.float64),
        "best_stage": str(recorder.best_stage),
        # 下一块的**起点** = 已经跑过的总轮数 (不是本块的起始偏移:
        # 第一块的起始偏移是 0, 但它跑完之后下一块该从 2 接着跑)。
        "epoch_offset": int(getattr(recorder, "epochs_total", 0)),
    }


def optimize_slm_gs_refine(config: SlmGsRefineConfig) -> Recorder:
    """Shape the far field into a uniform square on real SLM + CCD hardware.

    Pipeline: flat baseline -> GS warm start (bake-off) -> SPGD freeform
    refinement. The objective is the composite used in simulation, read from
    the CCD.

    Args:
        config: Fully-specified configuration. Devices are opened and closed
            here; they are never reused across runs.

    Returns:
        A :class:`Recorder` whose ``mark`` is ``"score"`` in ``mode="max"``.
        The recorder also carries ``epochs_executed`` (refinement epochs that
        actually ran) and ``stopped_on_budget`` (the wall-clock budget cut the
        run short), so a caller can tell a finished run from a truncated one.
    """
    if config.cam_type == "sim":
        # Registers the "sim" camera type as an import side effect.
        import ao_shaping.drivers.sim.slm_pib_sim  # noqa: F401

    rng = np.random.default_rng(config.seed)
    recorder = Recorder(mark="score", mode="max")

    slm = open_slm(config)
    # Send-pipeline overlay (GUI 离焦 / 底相位): apply to the SLM this run opens,
    # so the closed loop sends the same overlay the GUI's manual path uses.
    # No-op for a sim SLM (no overlay setters); default ``None`` leaves CLI runs
    # untouched.
    if config.send_overlay is not None:
        apply_overlays_to_handle(slm, config.send_overlay)
    cam = None
    try:
        cam = open_camera(config)
        with slm, cam:
            # 采集入口守卫 (#186): 驱动板不应答在第一帧前失败; 仿真面板 no-op。
            require_drive_board_entry(slm)
            original_slot = None
            original_gray = None
            if config.cam_type != "sim":
                from ao_shaping.drivers.slm.santec import SantecError

                try:
                    original_slot = slm.get_displayed_memory_number()
                except SantecError as exc:
                    if exc.code != 1:
                        raise
                if original_slot is None:
                    original_gray = slm.get_current_grayscale()
            try:
                return _run_gs_refine_session(
                    config, slm, cam, rng, recorder, protected_slot=original_slot
                )
            finally:
                if config.cam_type != "sim":
                    try:
                        if original_slot is None:
                            slm.set_grayscale(original_gray)
                        else:
                            slm._display_memory(original_slot)
                    # policy: restoring the panel's initial display is reported AND re-raised: an unrestored SLM is a real bench hazard for whoever runs next
                    except Exception:
                        logger.exception("failed to restore the initial SLM display")
                        raise
    finally:
        for dev in (cam, slm):
            if dev is None:
                continue
            try:
                dev.close()
            # policy: teardown closes what it can without masking the failure that ended the run
            except Exception:  # cleanup must not mask the real error
                logger.debug("device close failed during cleanup", exc_info=True)


@dataclass
class _RefineContext:
    """Mutable state shared by the stages of one GS-refine session.

    The session is a sequence of stages (flat baseline, GS warm start, SPGD
    refinement) that hand measurements to each other, so the state they share
    -- the best phase so far, the target mask, the measured-frame bookkeeping --
    lives here instead of in a chain of ``nonlocal`` closures. It also carries
    the Stage-A wall-clock budget and the chunk-resume state.
    """

    config: SlmGsRefineConfig
    slm: Any
    cam: Any
    recorder: Recorder
    rng: np.random.Generator
    panel_shape: tuple[int, int]
    work_slots: list[int]
    radius_px: float
    target_side: int
    center_xy: tuple[int, int]
    signal_local_xy: tuple[int, int]
    overlay: Any = None
    disturbance: np.ndarray | None = None
    blaze_choice: Any = None
    ceiling: float | None = None
    measurement_count: int = 0
    last_capture_utc: str = ""
    last_exposure_ms: float | None = None
    cv_mask: np.ndarray | None = None
    mask: np.ndarray | None = None
    mask_area_frac: float = 0.0
    cy: int = 0
    cx: int = 0
    base_panel: np.ndarray | None = None
    best_score: float = -math.inf
    best_stage: str = _STAGE_FLAT
    best_phase: np.ndarray | None = None
    best_frame: np.ndarray | None = None
    # --- Stage-A wall-clock budget + chunk-resume state (issue #167 merge) ---
    budget_s: float = 0.0
    session_started: float = 0.0
    epochs_executed: int = 0
    stopped_on_budget: bool = False
    resume: dict[str, Any] | None = None
    epoch_offset: int = 0
    grid: int = 0
    dof: np.ndarray | None = None
    chunk_count: int = 0
    # 摆 ROI 窗口用的**全幅**零级位置 (仿真路径没有全幅发现 ⇒ None)。
    full_frame_center: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        # The wall-clock budget covers the WHOLE Stage A session (flat probe +
        # GS + refinement), not just the refinement loop: the promise is about
        # the caller's 10-minute slot, so the clock starts when the session
        # starts -- i.e. when this context is created.
        self.budget_s = float(self.config.time_budget_s)
        self.session_started = time.monotonic()
        self.resume = self.config.resume_state
        self.grid = max(2, int(self.config.phase_grid))
        self.chunk_count = (
            int(self.config.chunk_epochs)
            if int(self.config.chunk_epochs) > 0
            else int(self.config.epochs)
        )

    def displayed_phase(self, phase: np.ndarray) -> np.ndarray:
        """The phase actually shown: plus the disturbance, plus the overlay."""
        from ao_shaping.utils.wavefront.zero_order_overlay import (
            apply_zero_order_overlay,
        )

        result = phase if self.disturbance is None else phase + self.disturbance
        return (
            result
            if self.overlay is None
            else apply_zero_order_overlay(result, self.overlay)
        )

    def measure(self, phase: np.ndarray) -> np.ndarray:
        """Display a phase and return the measured frame."""
        memory_number = None
        if self.config.cam_type != "sim":
            memory_number = self.work_slots[
                self.measurement_count % len(self.work_slots)
            ]
            self.measurement_count += 1
        frame = display_and_average(
            self.cam,
            self.slm,
            self.displayed_phase(phase),
            n_frames=self.config.n_eval_frames,
            settle_wait_s=self.config.settle_wait_s,
            settle_tol=self.config.settle_tol,
            settle_max_wait_s=self.config.settle_max_wait_s,
            settle_max_discard=self.config.settle_max_discard,
            memory_number=memory_number,
        )
        if self.config.cam_type == "daheng" and np.max(frame) >= 250:
            raise ValueError(
                "Daheng CCD frame saturated; reduce exposure before continuing"
            )
        if self.config.cam_type == "daheng":
            actual_ms = float(self.cam.get_exposure_time_ms())
            if not 0 < actual_ms <= (self.ceiling or 0.0) + 1e-9:
                raise RuntimeError(f"unsafe Daheng exposure readback: {actual_ms} ms")
            self.last_exposure_ms = actual_ms
        self.last_capture_utc = datetime.now(timezone.utc).isoformat(
            timespec="microseconds"
        )
        return frame

    def record(
        self,
        stage: str,
        frame: np.ndarray,
        mask: np.ndarray,
        phase: np.ndarray,
        *,
        other_frame: np.ndarray | None = None,
        other_phase: np.ndarray | None = None,
        positive_wins: bool | None = None,
        **extra: Any,
    ) -> float:
        """Append one evaluation row to the recorder; return its composite score."""
        score, pib, cv = _metrics(
            frame, mask, self.config.w_pib, self.config.w_unif, self.cv_mask
        )
        from ao_shaping.utils.image.beam_metrics import compute_shape_iou

        row = {
            "stage": stage,
            "score": score,
            "pib": pib,
            "cv": cv,
            **(
                {"cv_full": uniformity_cv(frame, mask)}
                if self.cv_mask is not None
                else {}
            ),
            "shape_iou": compute_shape_iou(frame, mask),
            "target_side": self.target_side,
            "target_shape": self.config.target_shape,
            "mask_area_frac": self.mask_area_frac,
            RECORD_EPOCH: len(self.recorder.history),
            "timestamp_utc": self.last_capture_utc,
            "exposure_ms_readback": (
                self.last_exposure_ms if self.config.cam_type == "daheng" else None
            ),
            **extra,
        }
        if self.config.record_debug:
            row[RECORD_IMAGE] = np.asarray(frame).copy()
            row[RECORD_PHASE] = np.asarray(
                self.slm.create_phase_from_array(self.displayed_phase(phase))
            ).copy()
            if other_frame is not None and other_phase is not None:
                other_img = np.asarray(other_frame).copy()
                other_gray = np.asarray(
                    self.slm.create_phase_from_array(self.displayed_phase(other_phase))
                ).copy()
                if positive_wins:
                    row.update(
                        _img_pos=row[RECORD_IMAGE],
                        _phase_pos=row[RECORD_PHASE],
                        _img_neg=other_img,
                        _phase_neg=other_gray,
                    )
                else:
                    row.update(
                        _img_pos=other_img,
                        _phase_pos=other_gray,
                        _img_neg=row[RECORD_IMAGE],
                        _phase_neg=row[RECORD_PHASE],
                    )
        self.recorder.append(row)
        return score


def _resolve_daheng_ceiling(ctx: "_RefineContext") -> float:
    """Resolve the Daheng exposure ceiling before any measurement can fire one.

    The ceiling is set by the optical path, not the sensor, so it is resolved
    per session: the environment variable, else probe the image itself.
    """
    config = ctx.config
    if not configured_daheng_exposure_limit():
        ctx.cam.reset_exposure_time(0.02)
        display_and_average(
            ctx.cam,
            ctx.slm,
            ctx.displayed_phase(np.zeros(ctx.panel_shape, dtype=np.float64)),
            n_frames=1,
            settle_wait_s=config.settle_wait_s,
            settle_tol=config.settle_tol,
            settle_max_wait_s=config.settle_max_wait_s,
            settle_max_discard=config.settle_max_discard,
            memory_number=ctx.work_slots[-1],
        )
    try:
        ceiling = resolve_daheng_exposure_limit(camera=ctx.cam)
    except ValueError as exc:
        if "no ladder frame carries signal" in str(exc):
            raise RuntimeError("入射光强不足：请检查光路/对准/激光出光") from exc
        raise
    if config.exposure_time_ms > ceiling + 1e-9:
        logger.warning(
            "Daheng exposure_time_ms {} ms exceeds resolved ceiling {} ms; "
            "clamping to ceiling",
            config.exposure_time_ms,
            ceiling,
        )
        config.exposure_time_ms = ceiling
    # The probe left the camera at the last ladder step; re-arm the
    # working exposure before the first measurement.
    ctx.cam.reset_exposure_time(config.exposure_time_ms)
    return ceiling


def _locate_zero_order_and_window(ctx: _RefineContext) -> None:
    """Locate the zero order (or reuse the frozen one) and set the CCD ROI window.

    A chunk resume carries the previous chunk's full-frame zero-order position
    (``full_frame_center``); reusing it keeps the ROI window pixel-identical
    across chunks so the frozen target mask stays valid. Otherwise the zero
    order is re-located on the full sensor. Blaze calibration, when enabled,
    runs on a full-frame shot but never moves the ROI on a resume. Sets
    ``ctx.full_frame_center`` and ``ctx.signal_local_xy``.
    """
    config = ctx.config
    panel_shape = ctx.panel_shape
    # 续跑块沿用上一块记录的**全幅**零级位置, **不重新定位**: 重新定位会落在一个
    # 近等的散斑颗粒上, ROI 窗口随之平移一两像素 —— 窗口坐标系整个挪动, 冻结的
    # 目标框就指到了别处, 指标在块边界上不可比。
    recorded_full_center = _resume_full_frame_center(config.resume_state)
    # 摆窗口用的全幅位置 (仿真路径没有全幅发现 ⇒ None)。
    full_frame_center: tuple[int, int] | None = None
    if config.cam_type != "sim":
        discovery = None
        if recorded_full_center is not None:
            full_x, full_y = recorded_full_center
            logger.info(
                "沿用续跑状态里的全幅零级位置 ({}, {}): 跳过零级重定位, "
                "ROI 窗口与上一块逐像素一致",
                full_x,
                full_y,
            )
        else:
            # Locate the zero order on the full sensor before restricting the CCD ROI.
            discovery = display_and_average(
                ctx.cam,
                ctx.slm,
                ctx.displayed_phase(np.zeros(panel_shape, dtype=np.float64)),
                n_frames=1,
                settle_wait_s=config.settle_wait_s,
                settle_tol=config.settle_tol,
                settle_max_wait_s=config.settle_max_wait_s,
                settle_max_discard=config.settle_max_discard,
                memory_number=ctx.work_slots[2],
            )
            if config.cam_type == "daheng" and np.max(discovery) >= 255:
                raise ValueError(
                    "Daheng full-frame discovery saturated; reduce exposure"
                )
            full_y, full_x = locate_zero_order(discovery)
        if config.blaze_period_px >= 0:
            from ao_shaping.optimizer.wfless._blaze_calibration import select_blaze

            if discovery is None:
                # blaze 标定要在**全幅**画面上找零级与信号阶, 续跑块跳过零级重定位
                # 后没有全幅帧, 这里补拍一张 (只用于标定, 不参与定位)。
                discovery = display_and_average(
                    ctx.cam,
                    ctx.slm,
                    ctx.displayed_phase(np.zeros(panel_shape, dtype=np.float64)),
                    n_frames=1,
                    settle_wait_s=config.settle_wait_s,
                    settle_tol=config.settle_tol,
                    settle_max_wait_s=config.settle_max_wait_s,
                    settle_max_discard=config.settle_max_discard,
                    memory_number=ctx.work_slots[2],
                )
            blaze_probe_count = 0

            def capture_blaze(phase: np.ndarray) -> np.ndarray:
                nonlocal blaze_probe_count
                slot = ctx.work_slots[blaze_probe_count % 2]
                blaze_probe_count += 1
                frame = display_and_average(
                    ctx.cam,
                    ctx.slm,
                    phase if ctx.disturbance is None else phase + ctx.disturbance,
                    n_frames=1,
                    settle_wait_s=config.settle_wait_s,
                    settle_tol=config.settle_tol,
                    settle_max_wait_s=config.settle_max_wait_s,
                    settle_max_discard=config.settle_max_discard,
                    memory_number=slot,
                )
                if config.cam_type == "daheng" and np.max(frame) >= 250:
                    raise RuntimeError("blaze CCD probe saturated; reduce exposure")
                return frame

            try:
                choice = select_blaze(
                    capture_blaze,
                    panel_shape,
                    target_side=ctx.target_side,
                    window_size=config.cam_size or ctx.target_side,
                    period_px=config.blaze_period_px,
                    direction=config.blaze_direction,
                    flat_frame=discovery,
                )
                overlay = choice.params
                ctx.overlay = overlay
                ctx.blaze_choice = choice
            except (RuntimeError, ValueError) as exc:
                logger.warning("blaze calibration skipped: {}", exc)
                ctx.overlay = None
                ctx.blaze_choice = None
            if ctx.overlay is not None:
                if recorded_full_center is None:
                    full_x, full_y = ctx.blaze_choice.signal_xy
                else:
                    logger.warning(
                        "blaze 重新标定不挪动 ROI: 续跑块继续用记录的全幅零级位置 ({}, {})",
                        full_x,
                        full_y,
                    )
                logger.info(
                    "measured blaze: period={} direction={} zero={} signal={} separation={:.1f}px",
                    ctx.overlay.blaze_period_px,
                    ctx.overlay.blaze_direction,
                    ctx.blaze_choice.zero_xy,
                    ctx.blaze_choice.signal_xy,
                    ctx.blaze_choice.separation_px,
                )
        if ctx.overlay is not None:
            full_frame_center = (int(full_x), int(full_y))
        elif discovery is not None:
            zy, zx = np.unravel_index(
                int(np.argmax(discovery)), np.asarray(discovery).shape
            )
            full_frame_center = (int(zx), int(zy))
        elif recorded_full_center is not None:
            # 续跑块**跳过**了零级重定位 (没有全幅帧可看), 但它记着上一块用的是
            # 哪个位置 —— 那正是本块摆窗口用的位置, 必须原样带下去。漏掉这一支
            # 会让本块把 (0, 0) 记成「全幅零级」, 下一块于是重新定位, 冻结的
            # ROI 在第二道块边界上就失效了 (2026-10-23 修复: 此前这一支落到
            # 下面的 (0, 0) 兜底)。
            full_frame_center = (int(full_x), int(full_y))
        else:
            full_frame_center = (0, 0)
        signal_local_xy = full_frame_center
        ctx.signal_local_xy = signal_local_xy
        if config.cam_size > 0:
            half = config.cam_size // 2
            full_shape = (
                discovery.shape
                if discovery is not None
                else _camera_frame_shape(ctx.cam)
            )
            if full_shape is not None and (
                full_x - half < 0
                or full_y - half < 0
                or full_x + half >= full_shape[1]
                or full_y + half >= full_shape[0]
            ):
                raise ValueError(
                    "CCD zero order too close to edge for requested window"
                )
            actual_size, actual_center = ctx.cam.reset_window(
                center=(full_x, full_y), size=(config.cam_size, config.cam_size)
            )
            ctx.signal_local_xy = (
                int(full_x - actual_center[0] + actual_size[0] // 2),
                int(full_y - actual_center[1] + actual_size[1] // 2),
            )
            logger.info(
                "CCD ROI centered on full-frame zero order ({}, {})", full_x, full_y
            )
    ctx.full_frame_center = full_frame_center


def _declared_default(name: str) -> Any:
    """The dataclass default of ``SlmGsRefineConfig.<name>``.

    Read from the class (not from an instance) because ``__post_init__``
    rewrites ``focal_length_m`` in place: the default is the "nobody typed it"
    baseline, the instance value is the resolved one.
    """
    for spec in dataclass_fields(SlmGsRefineConfig):
        if spec.name == name:
            if spec.default is MISSING:
                raise RuntimeError(f"SlmGsRefineConfig.{name} has no default")
            return spec.default
    raise KeyError(name)


def _value_source(value: Any, name: str) -> str:
    """``"default"`` or ``"input"`` for a scalar config field.

    ``config`` cannot record whether a caller passed a value or took the default
    (the click params and the dataclass both collapse those cases), so this
    reports the only thing that is knowable: whether the value still equals the
    declared default.
    """
    return "default" if value == _declared_default(name) else "input"


def _log_resolved_parameters(
    config: SlmGsRefineConfig,
    *,
    panel_shape: tuple[int, int],
    radius_px: float,
    target_side: int,
    bench_side: int,
) -> None:
    """Log the run's resolved parameters **once**, each with its provenance.

    ``target_side`` and the bench-space side are resolved in two different
    places (``_derive_target_side`` and ``bench_target_side_px``), and the focal
    length in a third (``__post_init__``). Printing the resolved values together
    with ``input`` / ``derived`` / ``default`` is what makes a run reproducible
    from its log: the config alone cannot say which numbers were typed.
    """
    exposure_source = _value_source(config.exposure_time_ms, "exposure_time_ms")
    weights_source = (
        "default"
        if (
            config.w_pib == _declared_default("w_pib")
            and config.w_unif == _declared_default("w_unif")
        )
        else "input"
    )
    side_source = "input" if config.target_side > 0 else "derived"
    lines = (
        f"target_shape={config.target_shape} "
        f"({_value_source(config.target_shape, 'target_shape')})",
        f"target_side={'auto' if config.target_side <= 0 else config.target_side} "
        f"-> {target_side}px x side_factor={config.side_factor} ({side_source})",
        f"gs_target_side_px={bench_side} (derived: camera px -> model far-field px)",
        f"focal_length_m={config.focal_length_m:.6f} "
        f"({'pinned' if config._focal_length_pinned else 'derived'})",
        f"camera_pixel_um={config.camera_pixel_um} (anchor)",
        f"phase_grid={config.phase_grid} "
        f"({_value_source(config.phase_grid, 'phase_grid')})",
        f"w_pib={config.w_pib} w_unif={config.w_unif} ({weights_source})",
        f"exposure_ms={config.exposure_time_ms} ({exposure_source})",
        f"panel={panel_shape} beam_radius_px={radius_px} "
        f"cam={config.cam_type}#{config.cam_id}",
    )
    for line in lines:
        logger.info("slm-gs-refine: {}", line)


def _prepare_refine_context(
    config: SlmGsRefineConfig,
    slm: Any,
    cam: Any,
    recorder: Recorder,
    rng: np.random.Generator,
    protected_slot: int | None,
) -> _RefineContext:
    """Open the session: geometry, exposure ceiling, zero-order discovery, blaze.

    Everything the stages need that does not change during the run is resolved
    once, here. The Stage-A wall-clock budget and the chunk-resume state are
    derived from ``config`` when :class:`_RefineContext` is created.
    """
    sync_device_wavelength(config, slm)
    panel_shape = panel_shape_from_slm(slm)
    disturbance = load_panel_disturbance(config.phase_disturbance_file, panel_shape)

    ctx = _RefineContext(
        config=config,
        slm=slm,
        cam=cam,
        recorder=recorder,
        rng=rng,
        panel_shape=panel_shape,
        work_slots=[],
        radius_px=0.0,
        target_side=0,
        center_xy=(0, 0),
        signal_local_xy=(0, 0),
        disturbance=disturbance,
    )

    center_xy = config.pupil_center or (panel_shape[1] // 2, panel_shape[0] // 2)
    side = pupil_side(config.beam_radius_px)
    if (
        center_xy[0] - side // 2 < 0
        or center_xy[1] - side // 2 < 0
        or center_xy[0] - side // 2 + side > panel_shape[1]
        or center_xy[1] - side // 2 + side > panel_shape[0]
    ):
        raise ValueError("pupil square extends beyond the SLM panel")
    ctx.center_xy = center_xy
    work_slots = [slot for slot in (126, 127, 124) if slot != protected_slot]
    ctx.work_slots = work_slots
    if config.cam_type == "daheng":
        ctx.ceiling = _resolve_daheng_ceiling(ctx)
    radius_px = max(8.0, float(config.beam_radius_px))
    ctx.radius_px = radius_px
    target_side = _derive_target_side(config, radius_px)
    ctx.target_side = target_side
    if config.blaze_period_px >= 0 and config.cam_type == "sim":
        raise ValueError("blaze calibration requires a full-frame hardware CCD")
    _locate_zero_order_and_window(ctx)
    bench_side = bench_target_side_px(
        target_side,
        camera_pixel_um=config.camera_pixel_um,
        far_field_pixel_size=_bench_probe(config, radius_px).far_field_pixel_size,
    )
    _log_resolved_parameters(
        config, panel_shape=panel_shape, radius_px=radius_px,
        target_side=target_side, bench_side=bench_side,
    )
    ctx.work_slots = work_slots[:2]
    return ctx


def _apply_resume_state(ctx: _RefineContext) -> None:
    """Restore a previous chunk's end state as this chunk's start.

    A chunk resume skips the flat baseline and the GS warm start (both were
    measured by an earlier chunk) and continues from the frozen ROI, the DOF
    vector, the base panel and the global best carried in ``config.resume_state``.
    """
    resume = ctx.resume
    if resume is None:
        return
    # --- 续跑: 跳过平场基线与 GS 热启动, 从上一块的末态继续 --------------
    ctx.base_panel = np.asarray(resume["base_panel"], dtype=np.float64)
    ctx.dof = np.asarray(resume["dof"], dtype=np.float64)
    frame0 = np.asarray(resume["flat_frame"], dtype=np.float64)
    if "center_y" in resume and "center_x" in resume:
        cy, cx = int(resume["center_y"]), int(resume["center_x"])
    else:
        # 交接文件里没有窗口内的零级位置 ⇒ 从平场帧重新定位 (与优化器自己
        # 用的峰值定位同一套 routine)。
        cy, cx = locate_zero_order(frame0)
        logger.info(
            "续跑状态里没有窗口内的零级位置, 改从平场帧重新定位 (row={}, col={})",
            cy,
            cx,
        )
    ctx.cy = cy
    ctx.cx = cx
    ctx.mask = _target_mask(frame0, cy, cx, ctx.target_side, ctx.config)
    ctx.mask_area_frac = float(ctx.mask.mean())
    ctx.best_score = float(resume["best_score"])
    ctx.best_phase = np.asarray(resume["best_phase"], dtype=np.float64)
    ctx.best_frame = np.asarray(resume["best_frame"], dtype=np.float64)
    ctx.best_stage = str(resume["best_stage"])
    ctx.epoch_offset = int(resume.get("epoch_offset", 0))
    logger.info(
        "续跑 Stage A: 从全局 epoch {} 继续, 当前最佳 {:.4f} (stage={})",
        ctx.epoch_offset,
        ctx.best_score,
        ctx.best_stage,
    )


def _stage_flat_baseline(ctx: _RefineContext) -> None:
    """Stage 0: flat baseline; locate the 0-order; freeze the ROI."""
    flat_panel = np.zeros(ctx.panel_shape, dtype=np.float64)
    frame0 = ctx.measure(flat_panel)
    cy, cx = (
        (ctx.signal_local_xy[1], ctx.signal_local_xy[0])
        if ctx.overlay is not None
        else locate_zero_order(frame0)
    )
    ctx.cy = cy
    ctx.cx = cx
    ctx.mask = _target_mask(frame0, cy, cx, ctx.target_side, ctx.config)
    ctx.mask_area_frac = float(ctx.mask.mean())
    if ctx.config.exclude_zero_order_for_cv:
        from ao_shaping.utils.image.beam_metrics import zero_order_exclusion_mask

        core = zero_order_exclusion_mask(frame0, ctx.mask)
        ctx.cv_mask = ctx.mask & ~core
        logger.info(
            "CV excludes {} zero-order pixels from {} target pixels",
            int(core.sum()),
            int(ctx.mask.sum()),
        )
    score_flat = ctx.record(_STAGE_FLAT, frame0, ctx.mask, flat_panel, cy=cy, cx=cx)
    ctx.best_score = score_flat
    ctx.base_panel = flat_panel
    ctx.best_phase = flat_panel
    ctx.best_frame = frame0
    logger.info(
        "0-order at (row={}, col={}); target side {}px; flat score {:.4f}",
        cy,
        cx,
        ctx.target_side,
        score_flat,
    )


def _stage_gs_warm_start(ctx: _RefineContext) -> None:
    """Stage 1: GS warm start, admitted only if it wins."""
    config = ctx.config
    base_panel = ctx.base_panel
    gs_box = _gs_pupil_phase(config, ctx.radius_px, ctx.target_side)
    if gs_box is None:
        return
    gs_panel = embed_pupil_square(gs_box, ctx.panel_shape, ctx.radius_px, ctx.center_xy)
    gs_frame = ctx.measure(gs_panel)
    score_gs = ctx.record(_STAGE_GS, gs_frame, ctx.mask, gs_panel, cy=ctx.cy, cx=ctx.cx)
    if score_gs > ctx.best_score:
        ctx.base_panel = gs_panel
        ctx.best_score = score_gs
        ctx.best_phase = gs_panel
        ctx.best_frame = gs_frame
        ctx.best_stage = _STAGE_GS
    else:
        logger.info(
            "GS warm start {:.4f} did not beat flat {:.4f}; keeping flat "
            "(check --camera-pixel-um / --beam-radius-px)",
            score_gs,
            ctx.best_score,
        )


def _stage_spgd_refine(ctx: _RefineContext) -> None:
    """Stage 2: SPGD freeform refinement."""
    config = ctx.config
    # The DOF vector stays flat: ``Base.update`` keeps its moments at
    # ``(dim,)`` and would not broadcast against a ``(grid, grid)``
    # gradient. It is reshaped only when embedded into the panel.
    grid = ctx.grid
    dof = ctx.dof if ctx.dof is not None else np.zeros(grid * grid, dtype=np.float64)
    optimizer = make_optimizer(
        config.optimizer_type, dof.size, _lr_at(config, ctx.epoch_offset)
    )

    def phase_of(vec: np.ndarray) -> np.ndarray:
        return compose_pupil_dof(
            ctx.base_panel,
            vec.reshape(grid, grid),
            ctx.panel_shape,
            ctx.radius_px,
            ctx.center_xy,
        )

    for epoch in range(ctx.epoch_offset, ctx.epoch_offset + max(1, ctx.chunk_count)):
        # Checked BEFORE spending an epoch: a budget that is already spent must
        # not buy another pair of far-field reads. The best phase/frame found so
        # far are kept -- an unfinished run is still a usable measurement.
        if (
            ctx.budget_s > 0.0
            and (time.monotonic() - ctx.session_started) >= ctx.budget_s
        ):
            ctx.stopped_on_budget = True
            logger.info(
                "Stage A 墙钟预算 {:.1f}s 已用尽: 在第 {} 个精化 epoch 前收工, "
                "保留目前的最佳相位 (实跑 {} 个 epoch)",
                ctx.budget_s,
                epoch,
                ctx.epochs_executed,
            )
            break
        lr = _lr_at(config, epoch)
        if hasattr(optimizer, "lr"):
            optimizer.lr = lr

        delta = ctx.rng.choice(np.array([-1.0, 1.0]), size=dof.size) * config.delta
        # Measure first, then advance: the phase we score below is the one
        # the camera actually saw, never a dof we have not evaluated yet.
        phase_pos = phase_of(dof + delta)
        phase_neg = phase_of(dof - delta)
        frame_pos = ctx.measure(phase_pos)
        pos_timestamp = ctx.last_capture_utc
        frame_neg = ctx.measure(phase_neg)
        neg_timestamp = ctx.last_capture_utc
        pos = _metrics(frame_pos, ctx.mask, config.w_pib, config.w_unif, ctx.cv_mask)[0]
        neg = _metrics(frame_neg, ctx.mask, config.w_pib, config.w_unif, ctx.cv_mask)[0]

        take_pos = pos >= neg
        measured_phase = phase_pos if take_pos else phase_neg
        probe_score = pos if take_pos else neg
        measured_frame = frame_pos if take_pos else frame_neg
        confirmed = False
        if config.confirm_best and probe_score > ctx.best_score:
            measured_frame = ctx.measure(measured_phase)
            confirmed = True

        # issue #140: the persisted best must be the un-perturbed convergence
        # point (dof), not the winning ±delta probe (which carries a random
        # ±delta rad offset). Measure the clean dof every epoch and log its
        # score in a separate column so the record distinguishes "optimizer
        # advance" (dof) from "probe selection bias" (pos/neg).
        phase_dof = phase_of(dof)
        frame_dof = ctx.measure(phase_dof)
        dof_timestamp = ctx.last_capture_utc
        dof_score = _metrics(
            frame_dof, ctx.mask, config.w_pib, config.w_unif, ctx.cv_mask
        )[0]
        score = ctx.record(
            _STAGE_REFINE,
            measured_frame,
            ctx.mask,
            measured_phase,
            other_frame=frame_neg if take_pos else frame_pos,
            other_phase=phase_neg if take_pos else phase_pos,
            positive_wins=take_pos,
            epoch=epoch,
            lr=lr,
            pos=pos,
            neg=neg,
            probe_score=probe_score,
            confirmed=confirmed,
            pos_timestamp_utc=pos_timestamp,
            neg_timestamp_utc=neg_timestamp,
            dof_score=dof_score,
            dof_timestamp_utc=dof_timestamp,
        )
        ctx.epochs_executed += 1
        if config.record_debug and confirmed:
            row = ctx.recorder.history[-1]
            row["_img_pos"] = np.asarray(frame_pos).copy()
            row["_phase_pos"] = np.asarray(
                ctx.slm.create_phase_from_array(ctx.displayed_phase(phase_pos))
            ).copy()
            row["_img_neg"] = np.asarray(frame_neg).copy()
            row["_phase_neg"] = np.asarray(
                ctx.slm.create_phase_from_array(ctx.displayed_phase(phase_neg))
            ).copy()
        # issue #140: track the best by the un-perturbed dof score, and persist
        # the dof's own phase (the convergence point), not the ±delta probe.
        if dof_score > ctx.best_score:
            ctx.best_score = dof_score
            ctx.best_phase = phase_dof
            ctx.best_frame = frame_dof
            ctx.best_stage = _STAGE_REFINE

        grad = spgd_gradient(pos, neg, delta, maximize=True)
        dof = np.clip(dof - optimizer.update(grad), -4.0 * np.pi, 4.0 * np.pi)

        if config.callback is not None:
            config.callback(epoch, score)
        if config.progress_every and epoch % config.progress_every == 0:
            logger.info(
                "epoch {:4d} lr={:.4g} score={:.4f} (pos={:.4f} neg={:.4f}) "
                "best={:.4f} [{}]",
                epoch,
                lr,
                score,
                pos,
                neg,
                ctx.best_score,
                ctx.best_stage,
            )
        if config.early_stop_score > 0 and ctx.best_score >= config.early_stop_score:
            logger.info("early stop at epoch {} (score {:.4f})", epoch, ctx.best_score)
            break

    ctx.dof = dof


def _run_gs_refine_session(
    config: SlmGsRefineConfig,
    slm: Any,
    cam: Any,
    rng: np.random.Generator,
    recorder: Recorder,
    protected_slot: int | None = None,
) -> Recorder:
    """Run one GS-refine session: flat baseline, GS warm start, SPGD refinement.

    A chunk resume (``config.resume_state``) skips the flat baseline and the GS
    warm start and continues from the previous chunk's end state instead.
    """
    ctx = _prepare_refine_context(config, slm, cam, recorder, rng, protected_slot)
    if ctx.resume is not None:
        _apply_resume_state(ctx)
    else:
        _stage_flat_baseline(ctx)
        _stage_gs_warm_start(ctx)
    _stage_spgd_refine(ctx)

    best_row = max(recorder.history, key=lambda r: r["score"])
    logger.info(
        "best composite {:.4f} at stage={} epoch={} ({} evaluations); "
        "target_shape={} mask_area={:.3f}%",
        best_row["score"],
        best_row["stage"],
        best_row.get("epoch", "-"),
        len(recorder.history),
        config.target_shape,
        ctx.mask_area_frac * 100.0,
    )
    # Summary-level shape record: the history rows carry it too, but a consumer
    # reading only the summary (the CLI banner, the saved debug JSON) must be able
    # to tell which shape was scored without opening the CSV.
    recorder.target_shape = config.target_shape
    recorder.mask_area_frac = ctx.mask_area_frac
    # How many refinement epochs actually ran: the number of ``refine`` rows in
    # the history. A run cut short by the wall-clock budget must be visible as
    # "fewer epochs than asked for", not as a silent gap.
    recorder.epochs_executed = ctx.epochs_executed
    recorder.time_budget_s = ctx.budget_s
    recorder.stopped_on_budget = ctx.stopped_on_budget
    recorder.elapsed_s = round(time.monotonic() - ctx.session_started, 6)
    # 可续跑的状态: DOF 向量 + 基座相位 (都是**未叠加**的 —— 零级叠加是显示时加的,
    # 进了状态下一块就会叠加两次) + 全局 epoch 计数。
    recorder.base_panel = ctx.base_panel
    recorder.dof = ctx.dof
    recorder.epoch_offset = ctx.epoch_offset
    recorder.epochs_total = ctx.epoch_offset + ctx.epochs_executed
    # 冻结的零级中心: 续跑块必须沿用同一块 ROI (重新定位会把指标挪到别的位置)。
    recorder.center_y = int(ctx.cy)
    recorder.center_x = int(ctx.cx)
    # 摆 ROI 窗口用的**全幅**位置: 下一块据此把窗口摆在同一个地方 ——
    # 重新定位零级会落到近等的散斑颗粒上, 窗口一平移, 窗口坐标系就跟着挪。
    # 默认是零级; 开了 blaze 载波时是信号阶位置 (窗口是按它摆的)。
    # 仿真路径没有全幅发现 ⇒ None (下一块照旧重新定位)。
    recorder.full_frame_center = ctx.full_frame_center
    # The best phase and frame ride on the Recorder as attributes rather than as
    # row columns: a 1200x1920 float64 panel is 17.6 MB, so one column would cost
    # ~7 GB across 400 epochs, and ``save_dataframe`` would stringify every array
    # into a CSV cell.
    recorder.best_phase = ctx.displayed_phase(ctx.best_phase)
    # 未叠加零级的原始相位: 续跑时要用它 (显示时再加叠加层)。
    recorder.best_phase_plain = ctx.best_phase
    recorder.best_score = ctx.best_score
    recorder.best_frame = ctx.best_frame
    recorder.best_stage = ctx.best_stage
    choice = ctx.blaze_choice
    recorder.blaze_choice = (
        {
            "period_px": ctx.overlay.blaze_period_px,
            "direction": ctx.overlay.blaze_direction,
            "zero_xy": choice.zero_xy,
            "signal_xy": choice.signal_xy,
            "separation_px": choice.separation_px,
            "signal_peak": choice.signal_peak,
            "trials": choice.trials,
        }
        if ctx.overlay is not None
        else None
    )
    return recorder
