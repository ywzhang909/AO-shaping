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
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.sim.slm_shaping_bench import (
    ShapingBenchConfig,
    UniformSquareShaper,
    composite_from_pib_cv,
    gs_shape,
    power_in_bucket,
    uniformity_cv,
)
from ao_shaping.optimizer.spgd import spgd_gradient
from ao_shaping.utils.io.file import Recorder
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
_TILT_SHIFT_SCALE_PX = 7400.0
# The wavelength at which _TILT_SHIFT_SCALE_PX was measured. K = lambda*f/(d*p_cam)
# scales WITH wavelength, so a scale measured at 1064nm must be referred to another
# wavelength before use -- otherwise the derived focal length inherits a spurious
# factor lambda/lambda_ref. Passing this makes the derived focal length
# wavelength-invariant, as a lens focal length must be.
_TILT_SHIFT_SCALE_MEASURED_AT_NM = 1064.0
_DEFAULT_WAVELENGTH_NM = 1064.0
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
    wavelength_nm=_DEFAULT_WAVELENGTH_NM,
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

    # --- freeform parameterisation ----------------------------------------
    phase_grid: int = 24
    """Edge length of the coarse freeform phase grid; DOF = ``phase_grid**2``.

    The panel has ~2.3 M pixels; perturbing all of them makes the SPGD gradient
    estimate pure noise. 24x24 = 576 DOF is what ``spgd-square --basis freeform``
    uses and is enough to synthesise a square.
    """

    delta: float = 0.35
    """SPGD perturbation amplitude in radians on the coarse grid."""

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
    """GS 变体: plain / weighted / hio / hio-weighted。默认 plain = 既有行为。"""
    gs_gamma: float = 0.5
    """加权 GS 累积反馈系数 γ, 0≤γ<1。γ=0 是退化哨兵 (权重恒为 1 ⇒ 与 plain GS 等价)。"""
    gs_weight_clip: float = 4.0
    """加权 GS 权重上限 (裁剪, 须 >1)。"""
    gs_beta: float = 0.9
    """HIO 反馈松弛系数 β (Fienup, 0≤β≤1)。β=0 是退化哨兵 (框外冻结)。"""
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
        :func:`_sync_device_wavelength` recomputes it from the device's answer.
        """
        self._focal_length_pinned = self.focal_length_m > 0.0
        if not self._focal_length_pinned:
            self.focal_length_m = self._derive_focal_length()

        # --- range validation (moved here from the optimizer body so a bad
        # value never reaches the hardware context managers) ---
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
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
        if self.gs_method not in ("plain", "weighted", "hio", "hio-weighted"):
            raise ValueError(
                "gs_method must be one of 'plain', 'weighted', 'hio', "
                f"'hio-weighted', got {self.gs_method!r}"
            )
        # γ≥1 makes the cumulative feedback unit-gain and it oscillates; γ=0 is a
        # deliberate degenerate sentinel (weight stays 1 ⇒ identical to plain GS)
        if not (0.0 <= self.gs_gamma < 1.0):
            raise ValueError(f"gs_gamma must be in [0, 1), got {self.gs_gamma}")
        if self.gs_weight_clip <= 1.0:
            raise ValueError(
                f"gs_weight_clip must be > 1.0, got {self.gs_weight_clip}"
            )
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
        if self.cam_size < 16:
            raise ValueError(f"cam_size must be at least 16, got {self.cam_size}")
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
            wavelength_nm=float(self.slm_wavelength or _DEFAULT_WAVELENGTH_NM),
            camera_pixel_um=self.camera_pixel_um,
            slm_pixel_um=self.panel_pixel_um,
            focal_scale_px=_TILT_SHIFT_SCALE_PX,
            scale_measured_at_nm=_TILT_SHIFT_SCALE_MEASURED_AT_NM,
        )


# ---------------------------------------------------------------------------
# Geometry / image helpers
# ---------------------------------------------------------------------------


def _panel_shape(slm: Any) -> tuple[int, int]:
    """Return the panel ``(height, width)`` for a real or simulated SLM."""
    res = getattr(slm, "Panel_Res", None)
    if res is not None:
        return int(res[1]), int(res[0])
    return 1200, 1920


def _pupil_side(radius_px: float) -> int:
    """Edge length of the square covering the illuminated pupil."""
    return max(32, int(round(2.0 * radius_px)))


def _square_mask(shape: tuple[int, int], cy: int, cx: int, side: int) -> np.ndarray:
    """Boolean square support of edge ``side`` centred on ``(cy, cx)``.

    Clipped to the frame. The bench metric functions only ever use the *support*
    of their ``target`` argument, so a boolean mask is a faithful stand-in for a
    target intensity pattern.
    """
    h, w = shape
    half = side / 2.0
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0c, y1c = max(0, y0), min(h, y1)
    x0c, x1c = max(0, x0), min(w, x1)
    mask = np.zeros(shape, dtype=bool)
    if y1c > y0c and x1c > x0c:
        mask[y0c:y1c, x0c:x1c] = True
    return mask


def _fit_exact(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Force ``arr`` to exactly ``shape`` by edge-padding then cropping."""
    h, w = shape
    out = np.asarray(arr, dtype=np.float64)
    ph, pw = max(0, h - out.shape[0]), max(0, w - out.shape[1])
    if ph or pw:
        out = np.pad(out, ((0, ph), (0, pw)), mode="edge")
    return out[:h, :w]


def _upsample_bilinear(coarse: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear upsample of a coarse grid to an exact ``shape``."""
    from scipy.ndimage import zoom

    if tuple(coarse.shape) == tuple(shape):
        return np.asarray(coarse, dtype=np.float64)
    factors = (shape[0] / coarse.shape[0], shape[1] / coarse.shape[1])
    return _fit_exact(zoom(coarse, factors, order=1, mode="nearest"), shape)


def _prepare_frame(frame: np.ndarray) -> np.ndarray:
    """Remove the detector background and return a non-negative frame.

    A CCD frame carries symmetric read noise, so roughly half its pixels are
    negative and an unprocessed frame yields ``PIB > 1`` (measured 1.012). The
    beam occupies a small fraction of the frame, so the median is a robust
    background estimate; clipping **after** subtraction is what makes ``PIB <= 1``
    and ``CV`` finite. Clipping the raw frame instead would rectify that noise
    into a pixel-count-sized DC pedestal.

    Note this is deliberately *not* the same convention as
    ``slm_square_shaping.square_quality_score``, which clips without subtracting:
    there the background fills the frame, so the median *is* the pedestal and
    subtracting it makes the ratio explode. The measurements behind both, and the
    4598x read-noise pedestal figure, are in
    ``report/slm_pib_bench/report.md`` §7.
    """
    clean = np.where(np.isfinite(frame), frame, 0.0)
    return np.clip(clean - np.median(clean), 0.0, None)


def _locate_zero_order(frame: np.ndarray) -> tuple[int, int]:
    """Locate the 0-order as the frame global maximum, returned as ``(cy, cx)``.

    On a 2f-Fourier bench the optical axis lands at the camera centre only by
    luck (measured: frame centre (1344, 760) vs 0-order (1441, 705)), so geometry
    must never be used for this.
    """
    masked = np.where(np.isfinite(frame), frame, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return int(iy), int(ix)


def _metrics(
    frame: np.ndarray, mask: np.ndarray, w_pib: float, w_unif: float
) -> tuple[float, float, float]:
    """Return ``(composite, pib, cv)`` for a measured frame.

    ``composite`` is the simulated objective, evaluated on real pixels.
    """
    pib = power_in_bucket(frame, mask)
    cv = uniformity_cv(frame, mask)
    return composite_from_pib_cv(pib, cv, w_pib=w_pib, w_unif=w_unif), pib, cv


def _embed_pupil_square(
    box: np.ndarray, panel_shape: tuple[int, int], radius_px: float
) -> np.ndarray:
    """Embed a pupil-square array into a full-panel array, centred.

    The freeform DOF and the GS phase are both computed on a square covering
    only the illuminated pupil. Everything outside stays flat, so no degree of
    freedom is spent on light that never reaches the far field.
    """
    ph, pw = panel_shape
    out = np.zeros((ph, pw), dtype=np.float64)
    side = min(_pupil_side(radius_px), ph, pw)
    y0, x0 = (ph - side) // 2, (pw - side) // 2
    out[y0 : y0 + side, x0 : x0 + side] = _fit_exact(box, (side, side))
    return out


def _compose(
    base_panel: np.ndarray,
    dof: np.ndarray | None,
    panel_shape: tuple[int, int],
    radius_px: float,
) -> np.ndarray:
    """Full-panel raw-radian phase = base (flat or GS) + embedded freeform DOF.

    ``base_panel`` is never mutated, so the accumulated DOF cannot be double
    counted when the best iterate is promoted.
    """
    if dof is None or not np.any(dof):
        return base_panel
    box = _upsample_bilinear(dof, (_pupil_side(radius_px),) * 2)
    return base_panel + _embed_pupil_square(box, panel_shape, radius_px)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _display_and_average(
    cam: Any,
    slm: Any,
    phase_rad: np.ndarray,
    *,
    n_frames: int,
    settle_wait_s: float,
    settle_tol: float,
    settle_max_wait_s: float,
    settle_max_discard: int,
    memory_number: int | None = None,
) -> np.ndarray:
    """Write ``phase_rad`` (raw radians), wait for the panel to settle, average.

    Real hardware uses a pair of explicit memory slots that exclude the slot
    displayed before the run. The simulator receives only the gray map because
    its ``display_data`` does not accept hardware slot arguments.
    """
    gray = slm.create_phase_from_array(phase_rad)
    if memory_number is None:
        slm.display_data(gray)
    else:
        slm.display_data(gray, memory_number=memory_number)

    if settle_wait_s > 0:
        time.sleep(settle_wait_s)

    def probe() -> tuple[float, float, float]:
        img = np.asarray(cam.get_numpy_image(n_sample=1), dtype=np.float64)
        cy, cx = _locate_zero_order(img)
        return float(img[cy, cx]), float(cy), float(cx)

    prev = probe()
    deadline = time.time() + max(0.0, settle_max_wait_s - settle_wait_s)
    for _ in range(int(settle_max_discard)):
        if time.time() >= deadline:
            break
        cur = probe()
        if (
            abs(cur[0] - prev[0]) <= settle_tol * max(prev[0], 1.0)
            and abs(cur[1] - prev[1]) <= settle_tol * 40.0
            and abs(cur[2] - prev[2]) <= settle_tol * 40.0
        ):
            break
        prev = cur

    acc = [
        np.asarray(cam.get_numpy_image(n_sample=1), dtype=np.float64)
        for _ in range(max(1, int(n_frames)))
    ]
    return _prepare_frame(np.mean(acc, axis=0))


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
    variants route through :class:`UniformSquareShaper`, which improves the
    *uniformity* of the flat-top square rather than plain error-reduction --
    weighted iterative feedback, Fienup HIO, or the documented
    "HIO first, weighted GS to polish" combination.

    Returns ``None`` when the warm start is disabled or GS degenerates, in which
    case the caller simply proceeds from the flat phase.
    """
    if not config.gs_warm_start or config.gs_iters <= 0:
        return None

    n_grid = _pupil_side(radius_px)
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
            phase = UniformSquareShaper(
                cfg,
                n_iters=config.gs_iters,
                method=config.gs_method,
                gamma=config.gs_gamma,
                weight_clip=config.gs_weight_clip,
                beta=config.gs_beta,
                hio_fraction=config.gs_hio_fraction,
                relax=config.gs_relax,
                seed=seed,
            ).run().phase
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


def _make_optimizer(kind: str, dim: int, lr: float) -> Any:
    """Create a gradient optimizer exposing a descent-direction ``update``."""
    from ao_shaping.algorithm import Adam, AdamW, SGD, AdaMOD

    table = {"adam": Adam, "adamw": AdamW, "adamod": AdaMOD, "sgd": SGD}
    cls = table.get(kind.strip().lower(), Adam)
    try:
        return cls(dim, lr=lr)
    except TypeError:
        return cls(dim)


def _lr_at(config: SlmGsRefineConfig, epoch: int) -> float:
    """Learning rate for ``epoch``, optionally cosine-decayed."""
    base = config.lr if config.lr > 0 else 0.15 * max(config.delta, 1e-6)
    if config.lr_schedule != "cosine" or config.epochs <= 1:
        return base
    frac = epoch / max(1, config.epochs - 1)
    floor = config.lr_min_ratio
    return base * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * frac)))


# ---------------------------------------------------------------------------
# Device construction
# ---------------------------------------------------------------------------


class _SlmCfg:
    """Minimal duck-type carrier for :meth:`Santec.from_params`."""

    def __init__(self, config: SlmGsRefineConfig) -> None:
        self.slm_number = config.slm_number
        # 0 means "ask the device": Santec treats None as "read the configured
        # wavelength", whereas a literal 0 would take the force-write branch in
        # _setup_wavelength and program a 0nm phase table.
        self.slm_wavelength = config.slm_wavelength or None
        self.shift_x = None
        self.shift_y = None


def _sync_device_wavelength(config: SlmGsRefineConfig, slm: Any) -> None:
    """Adopt the SLM's actual wavelength when the config did not pin one.

    Runs right after the SLM opens and before any phase math, so the GS
    propagation and the focal-length derivation both see the wavelength the
    panel is really programmed for. The simulated SLM has no wavelength
    attribute, in which case the bench fallback stays in place.
    """
    if config.slm_wavelength > 0:
        return
    reported = getattr(slm, "wavelength", None)
    if not isinstance(reported, int | float) or reported <= 0:
        # The simulated panel reports nothing. Adopt the bench fallback so no
        # downstream phase math ever sees a zero wavelength -- but say so,
        # because the caller asked the device and got nothing.
        config.slm_wavelength = int(_DEFAULT_WAVELENGTH_NM)
        logger.info(
            "SLM 未报告工作波长(仿真面板?), 几何推导沿用台架fallback {}nm; "
            "如实际不同请显式传 --slm-wavelength",
            _DEFAULT_WAVELENGTH_NM,
        )
        return
    config.slm_wavelength = int(reported)
    logger.info("波长改用 SLM 实际报告值 {}nm (原默认 0 = 询问设备)", reported)
    # The derived focal length needs no recompute: it is referred through
    # _TILT_SHIFT_SCALE_MEASURED_AT_NM, so it is wavelength-invariant (which a
    # lens focal length must be). What the device's wavelength DOES change is the
    # GS propagation, from here on.


def _open_slm(config: SlmGsRefineConfig) -> Any:
    """Build the SLM: real Santec, or the simulated panel for ``cam_type=sim``."""
    if config.cam_type == "sim":
        from ao_shaping.drivers.sim.slm_pib_sim import SimSLMPib

        return SimSLMPib()
    from ao_shaping.drivers.slm.santec import Santec

    return Santec.from_params(_SlmCfg(config))


def _open_cam(config: SlmGsRefineConfig) -> Any:
    """Build the camera and window it to ``cam_size``.

    ``create_camera`` forwards only the keys in the backend's ``accepted_kwargs``,
    and Daheng's allow-list does **not** include ``cam_size`` -- so the window has
    to be set explicitly with ``reset_window``.
    """
    from ao_shaping.drivers.ccd.common import create_camera

    if config.cam_type == "sim":
        return create_camera(
            "sim",
            cam_id=config.cam_id,
            exposure_time_ms=config.exposure_time_ms,
            cam_size=config.cam_size,
        )

    return create_camera(
        config.cam_type,
        cam_id=config.cam_id,
        exposure_time_ms=config.exposure_time_ms,
    )


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

    slm = _open_slm(config)
    cam = None
    try:
        cam = _open_cam(config)
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


def _run_gs_refine_session(
    config: SlmGsRefineConfig,
    slm: Any,
    cam: Any,
    rng: np.random.Generator,
    recorder: Recorder,
    protected_slot: int | None = None,
) -> Recorder:
    best_score = -math.inf
    best_stage = _STAGE_FLAT
    best_phase: np.ndarray | None = None
    best_frame: np.ndarray | None = None
    _sync_device_wavelength(config, slm)
    panel_shape = _panel_shape(slm)
    work_slots = [slot for slot in (126, 127, 124) if slot != protected_slot]
    if config.cam_type != "sim":
        # Locate the zero order on the full sensor before restricting the CCD ROI.
        discovery = _display_and_average(
            cam,
            slm,
            np.zeros(panel_shape, dtype=np.float64),
            n_frames=1,
            settle_wait_s=config.settle_wait_s,
            settle_tol=config.settle_tol,
            settle_max_wait_s=config.settle_max_wait_s,
            settle_max_discard=config.settle_max_discard,
            memory_number=work_slots[2],
        )
        full_y, full_x = _locate_zero_order(discovery)
        cam.reset_window(
            center=(full_x, full_y), size=(config.cam_size, config.cam_size)
        )
        logger.info("CCD ROI centered on full-frame zero order ({}, {})", full_x, full_y)
    radius_px = max(8.0, float(config.beam_radius_px))
    target_side = _derive_target_side(config, radius_px)
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

    work_slots = work_slots[:2]
    measurement_count = 0

    def measure(phase: np.ndarray) -> np.ndarray:
        nonlocal measurement_count
        memory_number = None
        if config.cam_type != "sim":
            memory_number = work_slots[measurement_count % len(work_slots)]
            measurement_count += 1
        return _display_and_average(
            cam,
            slm,
            phase,
            n_frames=config.n_eval_frames,
            settle_wait_s=config.settle_wait_s,
            settle_tol=config.settle_tol,
            settle_max_wait_s=config.settle_max_wait_s,
            settle_max_discard=config.settle_max_discard,
            memory_number=memory_number,
        )

    def record(
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
        score, pib, cv = _metrics(frame, mask, config.w_pib, config.w_unif)
        row = {
            "stage": stage,
            "score": score,
            "pib": pib,
            "cv": cv,
            "target_side": target_side,
            "_epoch": len(recorder.history),
            **extra,
        }
        if config.record_debug:
            row["_img"] = np.asarray(frame).copy()
            row["_phase"] = np.asarray(slm.create_phase_from_array(phase)).copy()
            if other_frame is not None and other_phase is not None:
                other_img = np.asarray(other_frame).copy()
                other_gray = np.asarray(
                    slm.create_phase_from_array(other_phase)
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
        recorder.append(row)
        return score

    # --- Stage 0: flat baseline; locate 0-order; freeze the ROI ------
    flat_panel = np.zeros(panel_shape, dtype=np.float64)
    frame0 = measure(flat_panel)
    cy, cx = _locate_zero_order(frame0)
    mask = _square_mask(frame0.shape, cy, cx, target_side)
    if not mask.any():
        raise ValueError(
            f"target square (side={target_side}px at ({cy},{cx})) is empty"
        )
    score_flat = record(_STAGE_FLAT, frame0, mask, flat_panel, cy=cy, cx=cx)
    best_score = score_flat
    best_phase = flat_panel
    best_frame = frame0
    logger.info(
        "0-order at (row={}, col={}); target side {}px; flat score {:.4f}",
        cy,
        cx,
        target_side,
        score_flat,
    )

    # --- Stage 1: GS warm start, admitted only if it wins -------------
    base_panel = flat_panel
    gs_box = _gs_pupil_phase(config, radius_px, target_side)
    if gs_box is not None:
        gs_panel = _embed_pupil_square(gs_box, panel_shape, radius_px)
        gs_frame = measure(gs_panel)
        score_gs = record(_STAGE_GS, gs_frame, mask, gs_panel, cy=cy, cx=cx)
        if score_gs > score_flat:
            base_panel = gs_panel
            best_score = score_gs
            best_phase = gs_panel
            best_frame = gs_frame
            best_stage = _STAGE_GS
        else:
            logger.info(
                "GS warm start {:.4f} did not beat flat {:.4f}; keeping flat "
                "(check --camera-pixel-um / --beam-radius-px)",
                score_gs,
                score_flat,
            )

    # --- Stage 2: SPGD freeform refinement --------------------------
    # The DOF vector stays flat: ``Base.update`` keeps its moments at
    # ``(dim,)`` and would not broadcast against a ``(grid, grid)``
    # gradient. It is reshaped only when embedded into the panel.
    grid = max(2, int(config.phase_grid))
    dof = np.zeros(grid * grid, dtype=np.float64)
    optimizer = _make_optimizer(
        config.optimizer_type, dof.size, _lr_at(config, 0)
    )

    def phase_of(vec: np.ndarray) -> np.ndarray:
        return _compose(
            base_panel, vec.reshape(grid, grid), panel_shape, radius_px
        )

    for epoch in range(max(1, int(config.epochs))):
        lr = _lr_at(config, epoch)
        if hasattr(optimizer, "lr"):
            optimizer.lr = lr

        delta = rng.choice(np.array([-1.0, 1.0]), size=dof.size) * config.delta
        # Measure first, then advance: the phase we score below is the one
        # the camera actually saw, never a dof we have not evaluated yet.
        phase_pos = phase_of(dof + delta)
        phase_neg = phase_of(dof - delta)
        frame_pos = measure(phase_pos)
        frame_neg = measure(phase_neg)
        pos = _metrics(frame_pos, mask, config.w_pib, config.w_unif)[0]
        neg = _metrics(frame_neg, mask, config.w_pib, config.w_unif)[0]

        take_pos = pos >= neg
        measured_phase = phase_pos if take_pos else phase_neg
        score = record(
            _STAGE_REFINE,
            frame_pos if take_pos else frame_neg,
            mask,
            measured_phase,
            other_frame=frame_neg if take_pos else frame_pos,
            other_phase=phase_neg if take_pos else phase_pos,
            positive_wins=take_pos,
            epoch=epoch,
            lr=lr,
            pos=pos,
            neg=neg,
        )
        if score > best_score:
            best_score = score
            best_phase = measured_phase
            best_frame = frame_pos if take_pos else frame_neg
            best_stage = _STAGE_REFINE

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
                best_score,
                best_stage,
            )
        if (
            config.early_stop_score > 0
            and best_score >= config.early_stop_score
        ):
            logger.info(
                "early stop at epoch {} (score {:.4f})", epoch, best_score
            )
            break

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
    recorder.best_phase = best_phase
    recorder.best_frame = best_frame
    recorder.best_stage = best_stage
    return recorder
