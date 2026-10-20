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
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
from ao_shaping.utils.image.targets import square_mask
from ao_shaping.utils.io.file import Recorder
from ao_shaping.utils.slm.bench_constants import (
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

__all__ = ["SlmGsRefineConfig", "optimize_slm_gs_refine"]

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

_STAGE_FLAT = "flat"
_STAGE_GS = "gs"
_STAGE_REFINE = "refine"


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

    # --- objective ---------------------------------------------------------
    target_side: int = 0
    """Target square side in **camera** pixels. 0 = derive from the pupil image."""

    side_factor: float = 1.0
    """Multiplier applied to the derived target side."""

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
    gs_target_side_px: int = 0
    """Override the bench-space target side (0 = derive from ``target_side``)."""

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
        if self.blaze_period_px < -1 or 0 < self.blaze_period_px < 4:
            raise ValueError("blaze_period_px must be -1 (off), 0 (auto), or >=4")
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
        if 0 < self.cam_size < 16:
            raise ValueError(
                f"cam_size must be >= 16 or 0 (full frame), got {self.cam_size}"
            )
        if self.slm_number < 1 or self.slm_number > 8:
            raise ValueError(f"slm_number must be in 1..8, got {self.slm_number}")
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


def _gs_pupil_phase(
    config: SlmGsRefineConfig, radius_px: float, target_side_cam: int
) -> np.ndarray | None:
    """Compute the GS warm-start phase on the pupil square (raw radians).

    ``config.gs_method`` selects the open-loop GS variant. ``"plain"`` keeps the
    historical path byte-for-byte (it calls :func:`gs_shape` directly); the other
    variants route through :class:`UniformSquareShaper` for weighted feedback,
    Fienup HIO, or adaptive target-domain feedback. Their measured uniformity
    must be compared against plain GS before adoption.

    Returns ``None`` when the warm start is disabled or GS degenerates, in which
    case the caller simply proceeds from the flat phase.
    """
    if not config.gs_warm_start or config.gs_iters <= 0:
        return None

    n_grid = pupil_side(radius_px)
    seed = config.seed if config.seed is not None else 0
    aperture_m = n_grid * config.panel_pixel_um * 1e-6
    probe = ShapingBenchConfig(
        n_grid=n_grid,
        aperture_size=aperture_m,
        wavelength=config.slm_wavelength * 1e-9,
        focal_length=config.focal_length_m,
        far_field_padding=config.far_field_padding,
        target_side_px=8,
        seed=seed,
    )
    # GS constrains a target expressed in *bench* far-field pixels, so the
    # camera-pixel target side must be converted before GS aims at it.
    if config.gs_target_side_px > 0:
        bench_side = config.gs_target_side_px
    else:
        cam_pitch_m = max(config.camera_pixel_um, 1e-6) * 1e-6
        bench_side = max(
            3, int(round(target_side_cam * cam_pitch_m / probe.far_field_pixel_size))
        )
    cfg = replace(probe, target_side_px=bench_side)

    logger.info(
        "GS warm start: n_grid={} aperture={:.3f}mm bench_target={}px "
        "(camera side {}px @ {:.2f}um/px)",
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
                    except Exception:
                        logger.exception("failed to restore the initial SLM display")
                        raise
    finally:
        for dev in (cam, slm):
            if dev is None:
                continue
            try:
                dev.close()
            except Exception:  # noqa: BLE001 - cleanup must not mask the real error
                logger.debug("device close failed during cleanup", exc_info=True)


@dataclass
class _RefineContext:
    """Mutable state shared by the stages of one GS-refine session.

    The session is a sequence of stages (flat baseline, GS warm start, SPGD
    refinement) that hand measurements to each other, so the state they share
    -- the best phase so far, the target mask, the measured-frame bookkeeping --
    lives here instead of in a chain of ``nonlocal`` closures.
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
    cy: int = 0
    cx: int = 0
    base_panel: np.ndarray | None = None
    best_score: float = -math.inf
    best_stage: str = _STAGE_FLAT
    best_phase: np.ndarray | None = None
    best_frame: np.ndarray | None = None

    def displayed_phase(self, phase: np.ndarray) -> np.ndarray:
        """The phase actually shown: plus the disturbance, plus the overlay."""
        from ao_shaping.utils.wavefront.zero_order_overlay import apply_zero_order_overlay

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
            memory_number = self.work_slots[self.measurement_count % len(self.work_slots)]
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
            "_epoch": len(self.recorder.history),
            "timestamp_utc": self.last_capture_utc,
            "exposure_ms_readback": (
                self.last_exposure_ms if self.config.cam_type == "daheng" else None
            ),
            **extra,
        }
        if self.config.record_debug:
            row["_img"] = np.asarray(frame).copy()
            row["_phase"] = np.asarray(
                self.slm.create_phase_from_array(self.displayed_phase(phase))
            ).copy()
            if other_frame is not None and other_phase is not None:
                other_img = np.asarray(other_frame).copy()
                other_gray = np.asarray(
                    self.slm.create_phase_from_array(self.displayed_phase(other_phase))
                ).copy()
                if positive_wins:
                    row.update(
                        _img_pos=row["_img"],
                        _phase_pos=row["_phase"],
                        _img_neg=other_img,
                        _phase_neg=other_gray,
                    )
                else:
                    row.update(
                        _img_pos=other_img,
                        _phase_pos=other_gray,
                        _img_neg=row["_img"],
                        _phase_neg=row["_phase"],
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
    ceiling = resolve_daheng_exposure_limit(camera=ctx.cam)
    if config.exposure_time_ms > ceiling + 1e-9:
        raise ValueError(
            f"Daheng exposure_time_ms {config.exposure_time_ms} ms exceeds the "
            f"resolved ceiling {ceiling} ms ({DAHENG_EXPOSURE_LIMIT_ENV} or probe)"
        )
    # The probe left the camera at the last ladder step; re-arm the
    # working exposure before the first measurement.
    ctx.cam.reset_exposure_time(config.exposure_time_ms)
    return ceiling


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
    once, here.
    """
    sync_device_wavelength(config, slm)
    panel_shape = panel_shape_from_slm(slm)
    disturbance = load_panel_disturbance(config.phase_disturbance_file, panel_shape)

    overlay = None
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
    if config.cam_type != "sim":
        # Locate the zero order on the full sensor before restricting the CCD ROI.
        discovery = display_and_average(
            cam,
            slm,
            ctx.displayed_phase(np.zeros(panel_shape, dtype=np.float64)),
            n_frames=1,
            settle_wait_s=config.settle_wait_s,
            settle_tol=config.settle_tol,
            settle_max_wait_s=config.settle_max_wait_s,
            settle_max_discard=config.settle_max_discard,
            memory_number=work_slots[2],
        )
        if config.cam_type == "daheng" and np.max(discovery) >= 255:
            raise ValueError("Daheng full-frame discovery saturated; reduce exposure")
        full_y, full_x = locate_zero_order(discovery)
        if config.blaze_period_px >= 0:
            from ao_shaping.optimizer.wfless._blaze_calibration import select_blaze

            blaze_probe_count = 0

            def capture_blaze(phase: np.ndarray) -> np.ndarray:
                nonlocal blaze_probe_count
                slot = work_slots[blaze_probe_count % 2]
                blaze_probe_count += 1
                frame = display_and_average(
                    cam, slm,
                    phase if disturbance is None else phase + disturbance,
                    n_frames=1, settle_wait_s=config.settle_wait_s,
                    settle_tol=config.settle_tol,
                    settle_max_wait_s=config.settle_max_wait_s,
                    settle_max_discard=config.settle_max_discard,
                    memory_number=slot,
                )
                if config.cam_type == "daheng" and np.max(frame) >= 250:
                    raise RuntimeError("blaze CCD probe saturated; reduce exposure")
                return frame

            choice = select_blaze(
                capture_blaze, panel_shape, target_side=target_side,
                window_size=config.cam_size or target_side,
                period_px=config.blaze_period_px,
                direction=config.blaze_direction, flat_frame=discovery,
            )
            overlay = choice.params
            ctx.overlay = overlay
            ctx.blaze_choice = choice
            full_x, full_y = choice.signal_xy
            logger.info(
                "measured blaze: period={} direction={} zero={} signal={} separation={:.1f}px",
                overlay.blaze_period_px, overlay.blaze_direction, choice.zero_xy,
                choice.signal_xy, choice.separation_px,
            )
        signal_local_xy = (full_x, full_y)
        ctx.signal_local_xy = signal_local_xy
        if config.cam_size > 0:
            half = config.cam_size // 2
            if (
                full_x - half < 0
                or full_y - half < 0
                or full_x + half >= discovery.shape[1]
                or full_y + half >= discovery.shape[0]
            ):
                raise ValueError(
                    "CCD zero order too close to edge for requested window"
                )
            actual_size, actual_center = cam.reset_window(
                center=(full_x, full_y), size=(config.cam_size, config.cam_size)
            )
            ctx.signal_local_xy = (
                int(full_x - actual_center[0] + actual_size[0] // 2),
                int(full_y - actual_center[1] + actual_size[1] // 2),
            )
            logger.info(
                "CCD ROI centered on full-frame zero order ({}, {})", full_x, full_y
            )
    logger.info(
        "slm-gs-refine: panel={} beam_radius={}px target_side={}px "
        "cam={}#{} exposure={}ms grid={}x{}",
        panel_shape,
        radius_px,
        target_side,
        config.cam_type,
        config.cam_id,
        config.exposure_time_ms,
        config.phase_grid,
        config.phase_grid,
    )
    ctx.work_slots = work_slots[:2]
    return ctx


def _stage_flat_baseline(ctx: _RefineContext) -> None:
    """Stage 0: flat baseline; locate the 0-order; freeze the ROI."""
    flat_panel = np.zeros(ctx.panel_shape, dtype=np.float64)
    frame0 = ctx.measure(flat_panel)
    cy, cx = (
        (ctx.signal_local_xy[1], ctx.signal_local_xy[0])
        if ctx.overlay is not None
        else locate_zero_order(frame0)
    )
    mask = square_mask(frame0.shape, cy, cx, ctx.target_side)
    if not mask.any():
        raise ValueError(
            f"target square (side={ctx.target_side}px at ({cy},{cx})) is empty"
        )
    ctx.cy = cy
    ctx.cx = cx
    ctx.mask = mask
    if ctx.config.exclude_zero_order_for_cv:
        from ao_shaping.utils.image.beam_metrics import zero_order_exclusion_mask

        core = zero_order_exclusion_mask(frame0, mask)
        ctx.cv_mask = mask & ~core
        logger.info(
            "CV excludes {} zero-order pixels from {} target pixels",
            int(core.sum()), int(mask.sum()),
        )
    score_flat = ctx.record(
        _STAGE_FLAT, frame0, mask, flat_panel, cy=cy, cx=cx
    )
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
    gs_panel = embed_pupil_square(
        gs_box, ctx.panel_shape, ctx.radius_px, ctx.center_xy
    )
    gs_frame = ctx.measure(gs_panel)
    score_gs = ctx.record(
        _STAGE_GS, gs_frame, ctx.mask, gs_panel, cy=ctx.cy, cx=ctx.cx
    )
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
    grid = max(2, int(config.phase_grid))
    dof = np.zeros(grid * grid, dtype=np.float64)
    optimizer = make_optimizer(config.optimizer_type, dof.size, _lr_at(config, 0))

    def phase_of(vec: np.ndarray) -> np.ndarray:
        return compose_pupil_dof(
            ctx.base_panel, vec.reshape(grid, grid), ctx.panel_shape,
            ctx.radius_px, ctx.center_xy,
        )

    for epoch in range(max(1, int(config.epochs))):
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
        pos = _metrics(
            frame_pos, ctx.mask, config.w_pib, config.w_unif, ctx.cv_mask
        )[0]
        neg = _metrics(
            frame_neg, ctx.mask, config.w_pib, config.w_unif, ctx.cv_mask
        )[0]

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


def _run_gs_refine_session(
    config: SlmGsRefineConfig,
    slm: Any,
    cam: Any,
    rng: np.random.Generator,
    recorder: Recorder,
    protected_slot: int | None = None,
) -> Recorder:
    """Run one GS-refine session: flat baseline, GS warm start, SPGD refinement."""
    ctx = _prepare_refine_context(
        config, slm, cam, recorder, rng, protected_slot
    )
    _stage_flat_baseline(ctx)
    _stage_gs_warm_start(ctx)
    _stage_spgd_refine(ctx)

    best_row = max(recorder.history, key=lambda r: r["score"])
    logger.info(
        "best composite {:.4f} at stage={} epoch={} ({} evaluations)",
        best_row["score"],
        best_row["stage"],
        best_row.get("epoch", "-"),
        len(recorder.history),
    )
    # The best phase and frame ride on the Recorder as attributes rather than as
    # row columns: a 1200x1920 float64 panel is 17.6 MB, so one column would cost
    # ~7 GB across 400 epochs, and ``save_dataframe`` would stringify every array
    # into a CSV cell.
    recorder.best_phase = ctx.displayed_phase(ctx.best_phase)
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
