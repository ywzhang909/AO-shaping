"""SPGD optimization for square beam uniformity using SLM with Zernike coefficient control.

This module optimizes Zernike coefficients displayed on an SLM to produce a
uniform square intensity pattern in the far-field. The cost function minimizes
the coefficient of variation (CV = std/mean) of the intensity within a target
square region, optionally weighted with encircled energy and aspect ratio.

Key differences from slm_zernike_pib.py:
- Cost function: square uniformity (CV) instead of PIB
- Target: square region instead of circular bucket
- Additional metrics: encircled energy, squareness, aspect ratio

==================== 使用方法 ====================

CLI 入口 (推荐):
    python src/ao_shaping/main.py spgd-square -e 2000 -n 4 --target-side 0 \
        --w-uniformity 0.4 --w-efficiency 0.6 --w-aspect 0.0

    # 或用 --init-coeffs 指定起始 Zernike 系数 (Noll 索引 dict)。
    # zernike 基只优化 Defocus (2,0) [Noll 4] 与 Spherical (4,0) [Noll 11]:
    python src/ao_shaping/main.py spgd-square --init-coeffs '{"4":1.0,"11":0.5}'

编程调用:
    >>> from ao_shaping.optimizer.wfless.slm_square_shaping import optimize_slm_square
    >>> recorder = optimize_slm_square(
    ...     center="shape",        # 目标区域中心: "shape"(原始光斑质心, 默认) / "max" / (x, y)
    ...     epochs=2000,
    ...     n_max=4,               # Zernike 最高径向阶数
    ...     target_side=0,         # 目标方形边长(px), 0=自动按边长比; 与 target_mean_brightness 互斥
    ...     target_mean_brightness=0.0,  # 目标方形平均亮度, >0 时自动由总亮度推导边长
    ...     delta=0.1,             # SPGD 扰动幅度
    ...     cam_id=0,              # 相机 ID
    ...     slm_number=1,          # SLM 编号 (SLM200 多片级联时指定)
    ...     slm_wavelength=0,      # 0 = 询问设备实际波长 (勿假设某台 SLM)
    ...     optimizer_type="adamod",
    ...     w_uniformity=0.4,      # 均匀度 (CV) 权重
    ...     w_efficiency=0.6,      # 环围能量权重
    ...     w_aspect=0.0,          # 宽高比权重
    ... )
    >>> best_iter, (best_epoch, best_val) = recorder.get_best_iter()

==================== 算法说明 ====================

每一轮迭代:
1. 对当前系数向量加/减随机扰动 ±Δ (SPGD 扰动)
2. 扰动系数 → 相位图 (zernike: PatternHelper.generate_zernike_polynomial)
3. 相位图经 slm.display_data 写入内存槽并显示 (自动轮换槽位)
4. 相机采集远场图 → 提取目标方形区域 → 计算 cost
5. 按 cost 差值更新系数 (AdaMOD/Adam/SGD 等优化器)

代价函数 = w_uniformity * CV + w_efficiency * (1 - EE) + w_aspect * AR_penalty
其中 CV 为目标区域内光强变异系数, EE 为环围能量占比, AR 为方形度偏差。

注意事项:
- 依赖硬件 (SLM200 + Daheng/MiiCam 相机), 无模拟模式。
- Zernike 约定: 使用 aotools Noll 1976 约定 (见 zernike_utils 模块文档),
  Noll 5 = (2, -2), Noll 11 = (4, 0)。
- zernike 基 (--basis zernike) 只优化 Defocus (2,0) 与 Spherical (4,0) 两种
  模式 (与 GUI multi_slm_controller.py 的 Zernike 分支一致): 相位经
  PatternHelper.generate_zernike_polynomial 在 radius=600 px (默认) 圆形孔径内
  生成 min-max 归一化 uint16 灰度图并直接 display_data 下发, 不再走弧度换算。
- 整形过程不对 CCD 开窗 resize: cam_size 仅用于目标方形边长上限。
- IDEAL_SPOT_RADIUS 环境变量控制 spot 检测半径 (默认 6)。
- SLM 相位写入必须轮换内存槽 (同一槽重复 display 是 no-op, 见 AGENTS.md)。

Example:
    >>> from ao_shaping.optimizer.wfless.slm_square_shaping import optimize_slm_square
    >>> recorder = optimize_slm_square(
    ...     center="shape",
    ...     epochs=2000,
    ...     n_max=4,
    ...     target_side=100,
    ...     delta=0.1,
    ...     cam_id=0,
    ...     slm_number=1,
    ... )
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tqdm

from ao_shaping.algorithm.heuristic.search import (
    run_heuristic_search,
)
from ao_shaping.display import SlmSquareDisplay
from ao_shaping.drivers.ccd.common import (
    capture_with_exposure,
    create_camera,
    get_camera_exposure_ms,
    resample_on_saturation,
)
from ao_shaping.drivers.slm import Santec
from ao_shaping.optimizer.constants import (
    CAM_SAMPLE_ITER,
    TEST_EXPOSURE_TIME_BRIGHTNESS,
    create_optimizer,
)
from ao_shaping.optimizer.wfless._convergence_diagnostics import (
    MIN_HISTORY,
    convergence_lr_delta_factors,
)
from ao_shaping.optimizer.wfless.adaptive_spgd_core import (
    AdaptiveSpgdConfig,
    get_stage_weights,
    should_boost_stagnation,
)
from ao_shaping.utils import Recorder, logger
from ao_shaping.utils.image.beam_metrics import smart_zero_order_center
from ao_shaping.utils.image.exposure_control import (
    resolve_daheng_exposure_limit,
)
from ao_shaping.utils.image.hardware_utils import log_center_brightness
from ao_shaping.utils.image.spots_calc import centroid
from ao_shaping.utils.image.targets import (
    SQUARE_OBJECTIVE_CHOICES,
    pearson_shape_metric,
)
from ao_shaping.utils.slm.bench_constants import (
    BLAZE_PERIOD_MIN_PX,
    MAX_SLM_NUMBER,
    MIN_CAM_SIZE_PX,
)
from ao_shaping.utils.wavefront.pattern_helper import PatternHelper
from ao_shaping.utils.wavefront.zernike_calc import (
    ZernikeGenerator,
    calc_n_zernike_terms,
    noll_indices,
)
from ao_shaping.utils.io.recorder_keys import (
    RECORD_COEFFS,
    RECORD_DIFF,
    RECORD_EPOCH,
    RECORD_GRAD,
    RECORD_IMAGE,
)

# Adam parameters
beta1 = 0.9
beta2 = 0.99
beta3 = 0.9999

# Camera parameters
ADVISE_EXPOSURE_TIME_BRIGHTNESS = int(255 / 3)

# SLM parameters
SLM_RESPONSE_TIME_S = 0.3  # Santec SLM-200 response time ~300ms
SLM_RESET_ON_EXIT = True

#: Simulation / test injection point; ``None`` means the real driver.
_SLM_CLASS_OVERRIDE: type | None = None


def _slm_cls(slm_type: str = "") -> type:
    """Resolve the SLM driver class.

    Precedence: an explicit ``_SLM_CLASS_OVERRIDE`` wins (tests), then
    ``slm_type="sim"`` for the 2f-Fourier simulator, otherwise the module-level
    ``Santec`` binding.

    The real driver is read from the module global rather than re-imported, so
    ``monkeypatch.setattr(module, "Santec", fake)`` keeps working — the
    simulation tests inject their SLM exactly that way.
    """
    if _SLM_CLASS_OVERRIDE is not None:
        return _SLM_CLASS_OVERRIDE

    if slm_type == "sim":
        from ao_shaping.drivers.sim.slm_pib_sim import SimSLMPib

        return SimSLMPib

    if Santec is None:
        raise RuntimeError("Santec SLM driver not available. Install the SLM SDK.")
    return Santec


# SLM resolution (from Santec.Panel_Res = (1920, 1200))
SLM_WIDTH = 1920
SLM_HEIGHT = 1200
SLM_RESOLUTION = (SLM_WIDTH, SLM_HEIGHT)


# Zernike basis: only these modes are optimised by ``spgd-square``. Defocus
# (2,0) + Spherical (4,0) match the GUI Zernike branch (multi_slm_controller.py)
# at the 600 px aperture radius. Full Noll-length init vectors keep only these
# entries (Noll 4 = (2,0), Noll 11 = (4,0)).
ZERNIKE_ACTIVE_MODES: tuple[tuple[int, int], ...] = ((2, 0), (4, 0))

_ZERNIKE_MIN_MASK_LEN = 5  # piston + tip + tilt always masked; mask must cover ≥ Noll 5


# Noll-order (n, m) enumeration — canonical implementation lives in
# zernike_calc.noll_indices (v0.12+ consolidation); this alias keeps the
# historical import name working for internal callers.
_zernike_indices = noll_indices


def _map_init_to_active_modes(
    init_c: np.ndarray,
    n_max: int,
    active_modes: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Map an initial coefficient vector onto the active modes only.

    Full Noll-length inputs (len == nk terms, e.g. the runner's default
    init_defocus/init_spherical vector or ``--init-coeffs``) keep only the
    entries of the requested ``active_modes``; shorter inputs are padded /
    truncated to ``len(active_modes)``. Non-active entries are dropped.
    """
    nk = calc_n_zernike_terms(n_max)
    arr = np.asarray(init_c, dtype=np.float64)
    if len(arr) == nk and nk != len(active_modes):
        active_pos = [
            i for i, mode in enumerate(_zernike_indices(n_max)) if mode in active_modes
        ]
        arr = arr[active_pos]
    if len(arr) < len(active_modes):
        padded = np.zeros(len(active_modes), dtype=np.float64)
        padded[: len(arr)] = arr
        return padded
    return arr[: len(active_modes)]


def _scatter_active_to_noll(
    vec: np.ndarray,
    active_pos: list[int],
    nk: int,
    has_rotation: bool = False,
) -> np.ndarray:
    """Scatter an active-mode parameter vector into a full Noll-order vector.

    The ``spgd-square --debug`` sidecar records ``_c`` for the ML corpus
    (``ml/hwdataset``). ``ml.hwdataset.index`` classifies a ``_c`` as
    ``PhaseSource.ZERNIKE`` only when its length is the FULL Noll count
    ``(n_max+1)(n_max+2)//2`` (rule 1, matching the sidecar ``n_max``). The
    optimiser works in ACTIVE-mode space (e.g. two modes), and a length-2 vector
    is neither triangular nor a perfect square, so such a record is dropped as
    ``odd_coefficient_length`` (TODO H-23).

    This maps the work-vector onto full Noll order (zeros on inactive slots)
    WITHOUT touching the optimiser's own state. The recorded representation is
    the same one ``--init-coeffs`` accepts, so the Zernike forward
    reconstruction in ``ml.hwdataset.transforms`` reads it faithfully.

    Args:
        vec: Active-mode parameters, length ``len(active_pos)`` (plus one
            trailing rotation DOF when ``has_rotation``).
        active_pos: Noll indices of the optimised modes, in ``vec`` order.
        nk: Full Noll length, ``(n_max+1)(n_max+2)//2``.
        has_rotation: When True the last element of ``vec`` is a rotation DOF
            appended AFTER the ``nk`` Noll slots; the result is length ``nk+1``
            and deliberately does NOT classify as a pure Zernike record (the
            commanded phase really did carry a rotation).

    Returns:
        A ``float64`` array of length ``nk`` (or ``nk+1`` with rotation).
    """
    arr = np.asarray(vec, dtype=np.float64).ravel()
    n_active = len(active_pos)
    out = np.zeros(nk + (1 if has_rotation else 0), dtype=np.float64)
    if n_active:
        out[np.asarray(active_pos, dtype=int)] = arr[:n_active]
    if has_rotation:
        out[-1] = float(arr[n_active])
    return out


def _locate_square_center(img: np.ndarray, mode: str) -> tuple[int, int]:
    if mode == "shape":
        return smart_zero_order_center(img)
    if mode == "centroid_thresh":
        tx, ty = centroid(img, moment=1, threshold=0.1)
        return (int(round(tx)), int(round(ty)))
    if mode == "max":
        idx = np.unravel_index(int(np.argmax(img)), img.shape)
        return (int(idx[1]), int(idx[0]))
    if mode == "mass":
        return (int(centroid(img)[0]), int(centroid(img)[1]))
    raise ValueError(f"Unknown center mode: {mode}")


def _zernike_phase_radians(
    coeffs: np.ndarray,
    n_max: int,
    zernike_gen: ZernikeGenerator,
) -> np.ndarray:
    """Convert a flat Zernike coefficient array to a **wrapped radian** phase map.

    NOTE (2026-09): retained for documentation/reference/tests only. The
    ``spgd-square`` zernike basis no longer uses this helper — it generates
    GUI-consistent uint16 patterns via ``PatternHelper.generate_zernike_polynomial``
    (min-max normalised, circular aperture), optimising only
    ``ZERNIKE_ACTIVE_MODES`` = Defocus (2,0) + Spherical (4,0).

    The previous implementation returned a uint16 pattern through
    ``PatternHelper._zernike_to_uint16``, which min-max normalised the phase
    (scale-invariant, no mod-2pi wrap) and bypassed the SLM driver's
    device-specific grayscale mapping. This version produces a physically
    correct phase in [0, 2pi) that the driver converts to grayscale via
    ``create_phase_from_array`` (2pi = max_grayscale).

    Args:
        coeffs: Flat array of Zernike coefficients (radians per mode).
        n_max: Maximum Zernike radial order.
        zernike_gen: ZernikeGenerator sized to the SLM panel.

    Returns:
        float64 phase map in [0, 2pi), shape (SLM_HEIGHT, SLM_WIDTH).
    """
    modes = _zernike_indices(n_max)
    coeffs_dict: dict[tuple[int, int], float] = {}
    for i, (n, m) in enumerate(modes):
        if i < len(coeffs):
            coeffs_dict[(n, m)] = float(coeffs[i])
    phase = zernike_gen.generate_polynomial(coeffs_dict)
    return np.mod(np.nan_to_num(phase, nan=0.0), 2.0 * np.pi)


def _freeform_phase_radians(
    flat_params: np.ndarray,
    resolution: tuple[int, int],
    grid: int,
) -> np.ndarray:
    """Convert a flat free-form low-res phase grid to a raw radian phase map.

    Low-order Zernike modes are smooth and cannot synthesise a square
    far-field. A free-form phase grid (the SLM's native degree of freedom)
    can, so this is the default basis for square shaping.

    Returns the RAW unwrapped radians — no mod-2π here: the SLM driver
    ``create_phase_from_array`` applies the wrap on radian→grayscale
    conversion (2026-09: raw-only contract, same as every phase generator).

    Args:
        flat_params: Flat array of length ``grid * grid`` (radians per cell).
        resolution: SLM panel resolution as (width, height).
        grid: Side of the square parameter grid.

    Returns:
        float64 raw radian phase map, shape (height, width).
    """
    width, height = resolution
    cells = np.asarray(flat_params, dtype=np.float64).reshape(grid, grid)
    block_h = int(np.ceil(height / grid))
    block_w = int(np.ceil(width / grid))
    upsampled = np.kron(cells, np.ones((block_h, block_w), dtype=np.float64))
    upsampled = upsampled[:height, :width]
    if upsampled.shape != (height, width):  # pad if grid doesn't tile exactly
        padded = np.zeros((height, width), dtype=np.float64)
        padded[: upsampled.shape[0], : upsampled.shape[1]] = upsampled
        upsampled = padded
    return upsampled  # raw radians — the SLM driver wraps on conversion


def square_uniformity_cost(
    img: np.ndarray,
    center: tuple[int, int],
    side: int,
) -> tuple[float, float, float]:
    """Compute square uniformity cost metrics.

    The cost is the negative coefficient of variation (CV = std/mean) of the
    intensity within a target square region. Lower CV = more uniform.

    Args:
        img: Camera image (2D array).
        center: Center position (x, y) of the square region.
        side: Side length of the square region in pixels.

    Returns:
        Tuple of (cost, cv, mean_intensity):
        - cost: Negative CV (to maximize via minimization, range ~(-inf, 0])
        - cv: Coefficient of variation (std/mean), range [0, inf)
        - mean_intensity: Mean intensity in the square region.
    """
    h, w = img.shape
    cx, cy = center
    half = side // 2

    # Compute square bounds with clipping
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)

    region = img[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0, float("inf"), 0.0

    mean_val = float(np.mean(region))
    if mean_val < 1e-10:
        return 0.0, float("inf"), 0.0

    std_val = float(np.std(region))
    cv = std_val / mean_val

    # Negative CV as cost (minimize CV → maximize -CV)
    cost = -cv
    return cost, cv, mean_val


def square_encircled_energy(
    img: np.ndarray,
    center: tuple[int, int],
    side: int,
) -> float:
    """Compute the fraction of total energy within the square region.

    Args:
        img: Camera image (2D array).
        center: Center position (x, y) of the square region.
        side: Side length of the square region in pixels.

    Returns:
        Fraction of total energy in the square (0.0 to 1.0).
    """
    h, w = img.shape
    cx, cy = center
    half = side // 2

    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)

    total = float(np.sum(img))
    if total < 1e-10:
        return 0.0

    region_sum = float(np.sum(img[y0:y1, x0:x1]))
    return region_sum / total


def square_peak_to_background_ratio(
    img: np.ndarray,
    center: tuple[int, int],
    side: int,
    *,
    min_background_px: int = 64,
) -> float:
    """Peak-to-background ratio (PBR) of the target box against the frame.

    Transcribed from the prose definition in Liu et al., *A universal and
    improved mutation strategy for feedback-based wavefront shaping
    optimization algorithm*, Acta Photonica Sinica 2023, 52(6):0629002:
    "the ratio of the focused spot to the average intensity of the speckle
    background" (Peak-to-background Ratio). The paper states this only in
    prose -- it prints no equation for it -- so this is our reading of that
    sentence, not the authors' typeset formula.

    This is the one part of that paper that is genuinely additive to the square
    objective: ``square_quality_score`` already measures in-box uniformity
    (CV), energy capture (EE) and shape (AR), but nothing about how dark the
    *background* got relative to the peak.

    Read-noise handling is mandatory here, not optional: an unprocessed frame
    drives the ratio **above 1** and the optimizer then chases noise, so
    negatives are clipped at 0 *inside* this function and callers cannot get it
    wrong.

    Only clipping is applied -- deliberately NOT median subtraction, unlike
    ``prepare_frame`` (``ao_shaping.utils.image.beam_metrics``). The background occupies most of the frame,
    so the median *is* the pedestal and subtracting it makes the ratio explode.
    Clipping alone removes the negative half of the read-noise
    distribution while leaving the pedestal intact, so the denominator stays a
    real measurement.

    Args:
        img: Camera frame (2D array).
        center: Target-box centre in window-local ``(x, y)`` pixels. Callers must
            keep this **frozen** for the whole run -- re-locating a box by
            ``argmax`` every iteration makes the objective discontinuous on a
            speckle field.
        side: Target-box side length in pixels.
        min_background_px: Minimum background pixel count required for the ratio
            to be meaningful; below this the frame is too small to separate
            signal from background and0.0 is returned.

    Returns:
        ``peak / mean_background`` (0.0 when undefined). Larger is better.
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"img must be 2D, got shape {arr.shape}")

    # Clip the negative half of the symmetric read-noise distribution. This keeps
    # the pedestal (so the background mean is a real measurement) while stopping
    # noise from driving the ratio above 1.
    frame = np.clip(arr, 0.0, None)

    h, w = arr.shape
    cx, cy = int(center[0]), int(center[1])
    half = max(int(side) // 2, 1)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    if y1 <= y0 or x1 <= x0:
        return 0.0

    peak = float(frame[y0:y1, x0:x1].max())

    # Background = everything OUTSIDE the box. The box is excluded so the ratio
    # cannot be inflated by the very signal it is measuring.
    outside = np.ones(arr.shape, dtype=bool)
    outside[y0:y1, x0:x1] = False
    background = frame[outside]
    if background.size < max(int(min_background_px), 1):
        return 0.0

    background_mean = float(background.mean())
    if background_mean <= 0.0 or peak <= 0.0:
        return 0.0
    return float(peak / background_mean)


def square_aspect_ratio(
    img: np.ndarray,
    center: tuple[int, int],
    side: int,
    threshold_ratio: float = 0.5,
) -> float:
    """Compute the aspect ratio of the illuminated region within the square.

    Args:
        img: Camera image (2D array).
        center: Center position (x, y) of the square region.
        side: Side length of the square region in pixels.
        threshold_ratio: Intensity threshold as fraction of peak in region.

    Returns:
        Aspect ratio (width/height) of the thresholded region. 1.0 = perfect square.
    """
    h, w = img.shape
    cx, cy = center
    half = side // 2

    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)

    region = img[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0

    peak = float(np.max(region))
    if peak < 1e-10:
        return 0.0

    mask = region > peak * threshold_ratio
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return 0.0

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    height_px = float(r_max - r_min + 1)
    width_px = float(c_max - c_min + 1)

    if height_px < 1e-10:
        return 0.0

    return width_px / height_px


def square_quality_score(
    cv: float,
    encircled_energy: float,
    aspect_ratio: float,
    w_cv: float = 0.4,
    w_ee: float = 0.4,
    w_ar: float = 0.2,
    *,
    peak_to_background: float = 0.0,
    w_pbr: float = 0.0,
    pbr_reference: float = 1000.0,
) -> float:
    """Compute a combined quality score for the square beam.

    Args:
        cv: Coefficient of variation (lower is better).
        encircled_energy: Fraction of energy in the square (higher is better).
        aspect_ratio: Width/height ratio (1.0 is ideal).
        w_cv: Weight for uniformity component.
        w_ee: Weight for energy efficiency component.
        w_ar: Weight for aspect ratio component.
        peak_to_background: Peak-to-background ratio of the box against the rest
            of the frame (see :func:`square_peak_to_background_ratio`). Unused
            when ``w_pbr`` is 0.
        w_pbr: Weight for the background-suppression (PBR) component. **Defaults
            to 0.0 (off)** so existing runs keep reproducing bit-for-bit; same
            convention as ``w_ar``. Set e.g. ``0.2`` to enable it.
        pbr_reference: PBR value that maps to a full score, when ``w_pbr`` > 0.
            The mapping is logarithmic (see below), which is why this is not a
            simple saturation constant.

    Returns:
        Quality score in [0, 1] where 1 is perfect.
    """
    # Uniformity component: 1 at CV=0, decays exponentially
    uniformity = np.exp(-cv * 2.0)
    # Energy efficiency component: direct
    efficiency = encircled_energy
    # Aspect ratio component: 1 at ratio=1.0, decays
    ar_score = np.exp(-abs(aspect_ratio - 1.0) * 3.0)

    score = w_cv * uniformity + w_ee * efficiency + w_ar * ar_score

    # Background suppression. PBR is unbounded (it keeps growing as the
    # background is suppressed), so it MUST be compressed to stay inside this
    # function's documented [0, 1] contract.
    #
    # The obvious `pbr / (1 + pbr)` is wrong here: on measured bench data PBR is
    # ~250 (peak 79 counts against a ~0.31 counts background), which maps to
    # 0.996 -- effectively saturated, leaving the SPGD/Adam update no gradient to
    # follow. A log mapping centred on `pbr_reference` keeps the whole realistic
    # 1..1000 range usable: PBR 1 -> ~0.15, PBR 10 -> ~0.5, PBR 100 -> 1.0.
    if w_pbr:
        pbr = max(float(peak_to_background), 0.0)
        ref = max(float(pbr_reference), 1e-6)
        pbr_term = float(np.log1p(pbr) / np.log1p(ref))
        score += w_pbr * min(max(pbr_term, 0.0), 1.0)

    return float(min(max(score, 0.0), 1.0))


# ``SQUARE_OBJECTIVE_CHOICES`` is imported from ``utils.image.target.objective``
# next to the other objective vocabularies; re-exported here so the optimizer's
# public surface stays self-describing.


def square_objective_score(
    img: np.ndarray,
    cv: float,
    encircled_energy: float,
    aspect_ratio: float,
    center: tuple[int, int],
    side: int,
    objective: str = "quality",
    w_cv: float = 0.4,
    w_ee: float = 0.4,
    w_ar: float = 0.2,
    *,
    peak_to_background: float = 0.0,
    w_pbr: float = 0.0,
    pbr_reference: float = 1000.0,
) -> float:
    """Single HIGHER-IS-BETTER scoring entry point for square shaping.

    Every square-shaping consumer (SPGD plus/minus perturbation, the epoch
    rescore, the heuristic fitness and the initial baseline) routes through this
    function so the objective's sign convention is decided in exactly one place.
    Both SPGD (``diff = pos - neg``) and ``run_heuristic_search(maximize=True)``
    are higher-is-better, so this function never returns a negated value for
    ``"quality"`` — doing so would break its documented ``[0, 1]`` contract and
    flip the direction of every existing square run.

    Args:
        img: The measured far-field frame (needed by image-domain objectives).
        cv: Coefficient of variation inside the target box.
        encircled_energy: Fraction of frame energy inside the target box.
        aspect_ratio: Target-box width/height ratio.
        center: Target-box centre in window-local ``(x, y)`` pixels.
        side: Target-box side length in pixels.
        objective: One of :data:`SQUARE_OBJECTIVE_CHOICES`.
        w_cv: Uniformity weight (``"quality"`` only).
        w_ee: Energy-efficiency weight (``"quality"`` only).
        w_ar: Aspect-ratio weight (``"quality"`` only).
        peak_to_background: Box peak-to-background ratio (``"quality"`` only).
        w_pbr: Background-suppression weight (``"quality"`` only, default off).
        pbr_reference: PBR value scoring full marks (``"quality"`` only).

    Returns:
        A score where **larger is better**.
    """
    key = str(objective).lower()
    if key in ("quality", "shape", ""):
        return square_quality_score(
            cv,
            encircled_energy,
            aspect_ratio,
            w_cv,
            w_ee,
            w_ar,
            peak_to_background=peak_to_background,
            w_pbr=w_pbr,
            pbr_reference=pbr_reference,
        )
    if key == "pearson":
        # ``pearson_shape_metric`` is a LOSS (smaller is better); negate it once,
        # here, so the whole search stack can stay higher-is-better.
        loss, _energy = pearson_shape_metric(
            img,
            (float(center[0]), float(center[1])),
            "square",
            float(side),
            1.0,
        )
        return -float(loss)
    raise ValueError(
        f"objective must be one of {SQUARE_OBJECTIVE_CHOICES}, got {objective!r}"
    )


def learning_schedule(
    cv: float,
    encircled_energy: float,
    gradient_history: list[float] | None = None,
    cv_history: list[float] | None = None,
    epoch: int = 0,
) -> tuple[float, float]:
    """Dynamic learning rate scheduler based on optimization state.

    Args:
        cv: Current coefficient of variation.
        encircled_energy: Current encircled energy.
        gradient_history: Recent gradient magnitudes.
        cv_history: Recent CV values.
        epoch: Current iteration number.

    Returns:
        (lr, delta): Learning rate and perturbation amplitude.
    """
    # Base parameters based on CV level
    if cv > 0.5:
        base_lr, base_delta = 3.0, 3.0
    elif cv > 0.3:
        base_lr, base_delta = 2.5, 2.5
    elif cv > 0.15:
        base_lr, base_delta = 2.0, 2.0
    elif cv > 0.05:
        base_lr, base_delta = 1.5, 1.5
    else:
        base_lr, base_delta = 1.0, 1.0

    if (
        gradient_history is None
        or cv_history is None
        or len(gradient_history) < MIN_HISTORY
    ):
        return base_lr, base_delta

    # Convergence state detection (shared: _convergence_diagnostics). CV is
    # lower-is-better (rising CV = divergence) and needs no std gate for
    # stagnation, so higher_is_better=False / stagnant_needs_low_std=False.
    lr_factor, delta_factor = convergence_lr_delta_factors(
        gradient_history,
        cv_history,
        epoch,
        higher_is_better=False,
        stagnant_needs_low_std=False,
    )

    return base_lr * lr_factor, base_delta * delta_factor


@dataclass
class SlmSquareConfig:
    """Configuration bundle for :func:`optimize_slm_square`.

    Carries every keyword argument of :func:`optimize_slm_square` except the
    positional ``center`` / ``epochs`` (which stay required function
    arguments). When passed via ``config=``, its fields are substituted for
    the corresponding keyword arguments; any explicitly-passed keyword
    arguments still win (merged on top, see the fold in the optimizer).
    """

    n_max: int = 4
    target_side: int = 0
    target_mean_brightness: float = 0.0
    side_factor: float = 1.5
    delta: float = 0.1
    lr: float = 0
    exposure_time_ms: float = 0.0  # 0 = keep/auto, see CameraParams
    cam_id: int = 0
    cam_type: str = "daheng"
    show: bool = False
    init_c: list[float] | np.ndarray | None = None
    cam_size: int = 300
    blaze_period_px: int = -1
    blaze_direction: str = "vertical"
    target_max_brightness: int = 200
    slm_number: int = 1
    slm_wavelength: int = 0
    optimizer_type: str = "adamod"
    random_seed: int | None = None
    w_uniformity: float = 0.4
    w_efficiency: float = 0.6
    w_aspect: float = 0.0
    #: Background-suppression (PBR) weight. 0.0 = off, so runs that predate this
    #: term keep reproducing exactly. See `square_peak_to_background_ratio`.
    w_pbr: float = 0.0
    basis: str = "freeform"
    phase_grid: int = 24
    zernike_radius: float | int | None = None
    zernike_mask: np.ndarray | None = None
    rotation_search_deg: float = 0.0
    algorithm: str = "spgd"
    pop_size: int | None = None
    objective: str = "quality"
    max_roi_energy_loss: float = 0.6
    init_amplitude_rad: float = 0.0
    #: When True, ``delta`` is held fixed and the ``lr == 0`` adaptive schedule
    #: may only update the learning rate. Set by the CLI when the user actually
    #: typed ``--delta``; otherwise the schedule owns ``delta`` as before.
    delta_pinned: bool = False
    #: When True (default), the target box center is fixed from the initial
    #: detection and NOT re-detected after exposure adjustment. When False, the
    #: center is re-detected on the post-exposure frame (legacy behavior).
    fix_target: bool = True
    #: Adaptive closed-loop SPGD (T4): widen the perturbation and switch the
    #: objective when the score stagnates for ``stagnate_win`` epochs.
    adaptive: bool = False
    stagnate_win: int = 10
    stagnate_boost: float = 1.5
    min_improve_frac: float = 0.01
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the square-shaping fields before any hardware is touched.

        These rules used to live in ``optimize_slm_square`` *after* the
        camera and SLM context managers were entered — a bad value then
        meant a wasted device open/close cycle. Moving them here means
        programmatic callers (no Click layer) also get the guard.
        """
        if self.n_max < 1:
            raise ValueError(f"n_max must be at least 1, got {self.n_max}")
        if self.blaze_period_px < -1 or 0 < self.blaze_period_px < BLAZE_PERIOD_MIN_PX:
            raise ValueError(
                f"blaze_period_px must be -1 (off), 0 (auto), or >={BLAZE_PERIOD_MIN_PX}"
            )
        if self.blaze_direction not in ("vertical", "horizontal"):
            raise ValueError("blaze_direction must be vertical or horizontal")
        if self.blaze_period_px >= 0 and (
            self.target_side < 1 or self.exposure_time_ms <= 0
        ):
            raise ValueError(
                "off-axis shaping requires target_side and a fixed exposure"
            )
        if self.blaze_period_px >= 0 and self.cam_type == "sim":
            raise ValueError("blaze calibration requires a full-frame hardware CCD")
        if self.target_side < 0:
            raise ValueError(
                f"target_side must be non-negative, got {self.target_side}"
            )
        if self.target_mean_brightness < 0:
            raise ValueError(
                f"target_mean_brightness must be non-negative, "
                f"got {self.target_mean_brightness}"
            )
        if self.target_side > 0 and self.target_mean_brightness > 0:
            raise ValueError(
                "target_side 与 target_mean_brightness 互斥: "
                "方形边长(px) 与 方形平均亮度只能二选一 (both given)"
            )
        if self.side_factor <= 0:
            raise ValueError(f"side_factor must be positive, got {self.side_factor}")
        if self.delta < 0:
            raise ValueError(f"delta must be non-negative, got {self.delta}")
        if self.lr < 0:
            raise ValueError(f"lr must be non-negative, got {self.lr}")
        if self.cam_id < 0:
            raise ValueError(f"cam_id must be non-negative, got {self.cam_id}")
        if self.cam_size < MIN_CAM_SIZE_PX:
            raise ValueError(f"cam_size must be >={MIN_CAM_SIZE_PX}, got {self.cam_size}")
        if self.target_max_brightness <= 0:
            raise ValueError(
                f"target_max_brightness must be positive, got {self.target_max_brightness}"
            )
        if self.slm_number < 1 or self.slm_number > MAX_SLM_NUMBER:
            raise ValueError(
                f"slm_number must be in 1..{MAX_SLM_NUMBER}, got {self.slm_number}"
            )
        if self.slm_wavelength < 0:
            raise ValueError(
                f"slm_wavelength must be non-negative, got {self.slm_wavelength}"
            )
        if self.w_uniformity < 0:
            raise ValueError(
                f"w_uniformity must be non-negative, got {self.w_uniformity}"
            )
        if self.w_efficiency < 0:
            raise ValueError(
                f"w_efficiency must be non-negative, got {self.w_efficiency}"
            )
        if self.w_aspect < 0:
            raise ValueError(f"w_aspect must be non-negative, got {self.w_aspect}")
        if self.w_pbr < 0:
            raise ValueError(f"w_pbr must be non-negative, got {self.w_pbr}")
        if self.phase_grid < 2:
            raise ValueError(f"phase_grid must be at least 2, got {self.phase_grid}")
        if self.zernike_radius is not None and self.zernike_radius <= 0:
            raise ValueError(
                f"zernike_radius must be positive when set, got {self.zernike_radius!r}"
            )
        if self.rotation_search_deg < 0 or self.rotation_search_deg > 360:
            raise ValueError(
                f"rotation_search_deg 必须在 0~360 范围内, "
                f"实际 {self.rotation_search_deg}"
            )
        if self.max_roi_energy_loss < 0 or self.max_roi_energy_loss > 1:
            raise ValueError(
                "max_roi_energy_loss must be within 0..1 "
                f"(0 disables the guard), got {self.max_roi_energy_loss!r}"
            )
        if self.init_amplitude_rad < 0:
            raise ValueError(
                f"init_amplitude_rad must be non-negative, got {self.init_amplitude_rad}"
            )
        if self.stagnate_win < 1:
            raise ValueError(
                f"stagnate_win must be at least 1, got {self.stagnate_win}"
            )
        if self.stagnate_boost < 1.0:
            raise ValueError(
                f"stagnate_boost must be >= 1.0, got {self.stagnate_boost}"
            )
        if not (0.0 <= self.min_improve_frac < 1.0):
            raise ValueError(
                f"min_improve_frac must be in [0, 1), got {self.min_improve_frac}"
            )


def optimize_slm_square(
    center: tuple[int, int] | str | None,
    epochs: int,
    config: SlmSquareConfig | None = None,
    n_max: int = 4,
    target_side: int = 0,
    target_mean_brightness: float = 0.0,
    side_factor: float = 1.5,
    delta: float = 0.1,
    lr: float = 0,
    exposure_time_ms: float = 0.0,
    cam_id: int = 0,
    cam_type: str = "daheng",
    show: bool = False,
    init_c: list[float] | np.ndarray | None = None,
    cam_size: int = 300,
    blaze_period_px: int = -1,
    blaze_direction: str = "vertical",
    target_max_brightness: int = 200,
    slm_number: int = 1,
    slm_type: str = "",
    slm_wavelength: int = 0,
    optimizer_type: str = "adamod",
    random_seed: int | None = None,
    w_uniformity: float = 0.4,
    w_efficiency: float = 0.6,
    w_aspect: float = 0.0,
    w_pbr: float = 0.0,
    basis: str = "freeform",
    phase_grid: int = 24,
    zernike_radius: float | int | None = None,
    zernike_mask: np.ndarray | None = None,
    rotation_search_deg: float = 0.0,
    algorithm: str = "spgd",
    pop_size: int | None = None,
    objective: str = "quality",
    max_roi_energy_loss: float = 0.6,
    init_amplitude_rad: float = 0.0,
    adaptive: bool = False,
    stagnate_win: int = 10,
    stagnate_boost: float = 1.5,
    min_improve_frac: float = 0.01,
    **kwargs,
) -> Recorder:
    """Optimize square beam uniformity using SLM with Zernike coefficient control.

    This function uses the SPGD (Stochastic Parallel Gradient Descent) algorithm
    to optimize Zernike coefficients displayed on an SLM, minimizing the
    coefficient of variation (CV) of the intensity within a target square region.

    Args:
        center: Center position for the square target. Can be None (auto-detect
            original-spot centroid, same as "shape"), "shape" (argmax-anchored
            smart detection — argmax anchor + centroid refinement on flat cores),
            "centroid_thresh" (brightness centroid, threshold=0.1×max),
            "max" (brightest pixel / argmax), "mass" (full-image centroid,
            stray-light sensitive), or tuple (x, y) for a fixed center
            (not re-detected after exposure adjustment).
        epochs: Number of optimization iterations.
        n_max: Maximum Zernike radial order. Controls the number of modes.
        target_side: Side length of the target square in pixels.
            Mutually exclusive with target_mean_brightness (specify only one).
            If 0 and target_mean_brightness is also 0, auto-computed from spot
            size × side_factor.
        target_mean_brightness: Target mean brightness (average pixel value /
            gray level) of the square. Mutually exclusive with target_side.
            When positive, the side length is auto-derived by energy
            conservation: side = sqrt(total_brightness / target_mean_brightness),
            where total_brightness is the integrated intensity of the initial
            camera image.
        side_factor: Multiplier for auto-computed side length (default 1.5).
        delta: Perturbation amplitude for Zernike coefficients.
        lr: Learning rate. If 0, auto-adjusted via learning_schedule.
        exposure_time_ms: Camera exposure time in ms. If 0, auto-exposure.
        cam_id: Camera device ID.
        show: Whether to display images during optimization.
        init_c: Initial Zernike coefficients. If None, starts from zeros.
        cam_size: Camera window size.
        target_max_brightness: Target max brightness for auto-exposure.
        slm_number: SLM device number (1-8).
        slm_wavelength: SLM wavelength in nm (0 = ask the device).
        optimizer_type: Optimizer type (adam/adamod/sgd/muno).
        random_seed: Random seed for reproducibility.
        w_uniformity: Weight for uniformity (CV) in quality score.
        w_efficiency: Weight for encircled energy in quality score.
        w_aspect: Weight for aspect ratio in quality score.
        basis: Parameterisation basis ("zernike" or "freeform").
        phase_grid: Freeform grid side (dim = grid^2).
        zernike_radius: Zernike aperture radius in pixels. Defaults to
            min(SLM_HEIGHT, SLM_WIDTH) / 2 = 600, matching the GUI
            (multi_slm_controller.py Zernike branch).
        zernike_mask: 0/1 binary array selecting which Zernike Noll modes
            participate in SPGD. Length must be ≥ 5; Noll indices 1-3 (piston,
            tip, tilt) are forced to 0. When provided, overrides the hardcoded
            ZERNIKE_ACTIVE_MODES ((2,0),(4,0)). E.g. [0,0,0,1,0,0,0,0,0,0,0,1]
            activates only defocus (2,0) and spherical (4,0).
        rotation_search_deg: SLM↔相机相对旋转搜索范围(度, 0~360)。0=不启用
            旋转校正(默认); >0 时旋转角作为额外 SPGD 自由度, 在
            [-range/2, +range/2] 内搜索, 每轮下发相位图按当前角旋转补偿。
            起始值 0°, 扰动步长固定 1°。
        objective: Square-shaping objective, one of
            :data:`SQUARE_OBJECTIVE_CHOICES`. ``"quality"`` (default) is the
            historical combined score; ``"pearson"`` is the FourierGSNet
            ``1 - Pearson`` correlation loss, negated to higher-is-better by
            :func:`square_objective_score`.
        adaptive: Enable the adaptive SPGD closed loop (default False). When on,
            the quality weights are scheduled from
            :data:`AdaptiveSpgdConfig.stage_weights` — a
            (w_uniformity, w_efficiency, w_aspect) triple — switching at epoch
            thirds, and ``w_uniformity`` is temporarily boosted while the
            best-CV improvement stays below ``min_improve_frac`` over the last
            ``stagnate_win`` epochs. When off, behavior is identical to previous
            releases.
        stagnate_win: Stagnation-detection window in epochs (adaptive only).
        stagnate_boost: Multiplicative boost applied to ``w_uniformity`` while
            stagnating; 1.0 disables the boost (adaptive only).
        min_improve_frac: Stagnation trigger — best-CV improvement below this
            fraction over the window counts as stagnation (adaptive only).
        **kwargs: Additional optimizer parameters.

    Returns:
        Recorder: Optimization history recorder.
    """
    if config is None:
        config = SlmSquareConfig(
            n_max=n_max,
            target_side=target_side,
            target_mean_brightness=target_mean_brightness,
            side_factor=side_factor,
            delta=delta,
            lr=lr,
            exposure_time_ms=exposure_time_ms,
            cam_id=cam_id,
            cam_type=cam_type,
            show=show,
            init_c=init_c,
            cam_size=cam_size,
            blaze_period_px=blaze_period_px,
            blaze_direction=blaze_direction,
            target_max_brightness=target_max_brightness,
            slm_number=slm_number,
            slm_wavelength=slm_wavelength,
            optimizer_type=optimizer_type,
            random_seed=random_seed,
            w_uniformity=w_uniformity,
            w_efficiency=w_efficiency,
            w_aspect=w_aspect,
            basis=basis,
            phase_grid=phase_grid,
            zernike_radius=zernike_radius,
            zernike_mask=zernike_mask,
            rotation_search_deg=rotation_search_deg,
            algorithm=algorithm,
            pop_size=pop_size,
            objective=objective,
            adaptive=adaptive,
            stagnate_win=stagnate_win,
            stagnate_boost=stagnate_boost,
            min_improve_frac=min_improve_frac,
            kwargs=kwargs,
        )
    else:
        # Extra keyword args are merged into the config's escape-hatch dict.
        config.kwargs.update(kwargs)

    # `delta` is only adaptive when the caller did not pin it. Click collapses
    # "user typed --delta 0.1" and "user typed nothing" into one value, so the
    # runners resolve that distinction via ``resolve_spgd_delta`` and pass the
    # answer through the config (or, for direct callers, through kwargs). Read
    # it BEFORE the config block below, which folds it in further down.
    _delta_pinned = bool(kwargs.pop("delta_pinned", False))

    n_max = config.n_max
    target_side = config.target_side
    target_mean_brightness = config.target_mean_brightness
    side_factor = config.side_factor
    delta = config.delta
    lr = config.lr
    exposure_time_ms = config.exposure_time_ms
    cam_id = config.cam_id
    # Must be read back from the config: the ``cam_type`` function parameter
    # keeps its ``"daheng"`` default whenever a caller passes ``config=``
    # directly (e.g. the slm-gsnet runner), so without this line the camera
    # backend is silently ignored.
    cam_type = config.cam_type
    show = config.show
    init_c = config.init_c
    cam_size = config.cam_size
    blaze_period_px = config.blaze_period_px
    blaze_direction = config.blaze_direction
    target_max_brightness = config.target_max_brightness
    slm_number = config.slm_number
    slm_wavelength = config.slm_wavelength
    optimizer_type = config.optimizer_type
    random_seed = config.random_seed
    w_uniformity = config.w_uniformity
    w_efficiency = config.w_efficiency
    w_aspect = config.w_aspect
    w_pbr = float(config.w_pbr)
    basis = config.basis
    phase_grid = config.phase_grid
    zernike_radius = config.zernike_radius
    zernike_mask = config.zernike_mask
    rotation_search_deg = config.rotation_search_deg
    algorithm = config.algorithm
    pop_size = config.pop_size
    objective = str(config.objective).lower()
    max_roi_energy_loss = float(config.max_roi_energy_loss)
    init_amplitude_rad = float(config.init_amplitude_rad)
    _delta_pinned = _delta_pinned or bool(config.delta_pinned)
    _fix_target = bool(config.fix_target)
    kwargs = config.kwargs

    # Adaptive SPGD closed loop (opt-in via config.adaptive). The base weights
    # are frozen here — best-* tracking must stay comparable across epochs while
    # the scheduled weights (and the stagnation boost) only affect the SCORED
    # quality during a run.
    _base_w_uniformity, _base_w_efficiency, _base_w_aspect = (
        w_uniformity,
        w_efficiency,
        w_aspect,
    )
    _adaptive_cfg = (
        AdaptiveSpgdConfig(
            stagnate_win=int(config.stagnate_win),
            stagnate_boost=float(config.stagnate_boost),
            min_improve_frac=float(config.min_improve_frac),
            enable_stagnate_boost=True,
            boost_w_u_only=True,
        )
        if bool(config.adaptive)
        else None
    )
    _adaptive_boost_active = False

    delta = abs(delta)
    epochs = int(epochs)
    rng = np.random.default_rng(random_seed)

    # Reject an out-of-vocabulary objective HERE, before any device is opened.
    # The only other validation sits at the bottom of square_objective_score, i.e. in
    # per-epoch scoring -- after the SLM has been driven and the camera opened and
    # probed, so a typo reached the hardware and left the panel on whatever phase the
    # first perturbation wrote.
    #
    # This is deliberately STRICTER than square_objective_score, which still tolerates the
    # legacy aliases ``("quality", "shape", "")`` at its first branch. Nothing in ``src/``
    # passes ``"shape"`` or ``""`` -- the CLI and every runner default to ``"quality"`` and
    # declare the same ``SQUARE_OBJECTIVE_CHOICES`` vocabulary (imported from
    # ``utils.image.target.objective``) that this check enforces. The alias branch is left
    # alone because it is reachable by direct callers of the scorer; do not "reconcile" the
    # two by widening this check.
    _objective = str(objective).lower()
    if _objective not in SQUARE_OBJECTIVE_CHOICES:
        raise ValueError(
            f"objective must be one of {SQUARE_OBJECTIVE_CHOICES}, got {objective!r}"
        )

    # The recorded score is whatever ``objective`` selected, so name the column
    # accordingly. The default (``quality``/``shape``/``""``) keeps the historical
    # ``quality`` column name so existing default runs stay byte-identical; a
    # non-default objective (e.g. ``pearson``) records under its own name so a
    # downstream analysis can tell which objective produced the trace instead of
    # reading a Pearson score out of a column labelled ``quality``.
    _DEFAULT_OBJECTIVE_ALIASES = ("quality", "shape", "")
    _objective_key = str(objective).lower()
    _score_column = (
        "quality" if _objective_key in _DEFAULT_OBJECTIVE_ALIASES else _objective_key
    )
    recorder = Recorder(mark=_score_column, mode="max")

    # History for convergence detection
    _gradient_history: list[float] = []
    _cv_history: list[float] = []
    _max_history_len = 50

    # Phase parameterisation basis.
    #   - "zernike": active-mode coefficient vector, dim = len(ZERNIKE_ACTIVE_MODES)
    #   - "freeform": flat grid x grid phase map (radians), dim = grid^2
    # Low-order Zernike modes are smooth and cannot synthesise a square
    # far-field; freeform (the SLM's native per-pixel DOF) can, so it is the
    # default for square shaping.
    basis = str(basis).lower().strip()
    if basis not in ("zernike", "freeform"):
        raise ValueError(f"Unknown basis: {basis!r} (expected 'zernike' or 'freeform')")
    if basis == "zernike":
        # 与 GUI multi_slm_controller.py 一致: 默认孔径 = SLM 面板短边一半
        if zernike_radius is None:
            zernike_radius = min(SLM_HEIGHT, SLM_WIDTH) / 2.0
        _noll_modes = _zernike_indices(n_max)
        nk = len(_noll_modes)
        if zernike_mask is not None:
            # --- mask-based mode selection ---
            _mask = np.asarray(zernike_mask, dtype=int).ravel()
            if _mask.size < _ZERNIKE_MIN_MASK_LEN:
                raise ValueError(
                    f"zernike_mask 长度必须 >= {_ZERNIKE_MIN_MASK_LEN} "
                    f"(piston+tip+tilt 强制为0), 实际 {_mask.size}"
                )
            # Pad to nk or truncate
            if _mask.size < nk:
                _mask = np.pad(_mask, (0, nk - _mask.size), constant_values=0)
            elif _mask.size > nk:
                logger.warning(
                    "zernike_mask 长度 {} 超过 n_max={} 的 Noll 项数 {}, 截断",
                    _mask.size,
                    n_max,
                    nk,
                )
                _mask = _mask[:nk]
            # Noll 1,2,3 (piston, tip, tilt) 强制为 0
            _mask[0] = _mask[1] = _mask[2] = 0
            _active_modes: tuple[tuple[int, int], ...] = tuple(
                _noll_modes[i] for i in range(nk) if _mask[i] == 1
            )
            if not _active_modes:
                raise ValueError(
                    "zernike_mask 全为 0 (强制屏蔽 piston/tip/tilt 后): "
                    "至少需要一个活动模式"
                )
            logger.info(
                "zernike_mask 活动模式: {} (Noll 索引 {})",
                _active_modes,
                [i for i in range(nk) if _mask[i] == 1],
            )
        else:
            # --- default: hardcoded active modes (backward compatible) ---
            _active_modes = tuple(
                m for m in ZERNIKE_ACTIVE_MODES if m[0] <= n_max and abs(m[1]) <= n_max
            )
            if not _active_modes:
                raise ValueError(
                    f"n_max={n_max} 过小: Zernike 活动模式 {ZERNIKE_ACTIVE_MODES} "
                    "超出最大径向阶数上限"
                )
        _dim = len(_active_modes)
        logger.info(
            "Zernike 基启用 {} 个活动模式 (dim={}): {}",
            _dim,
            _dim,
            [(i + 1, m) for i, m in enumerate(_active_modes)],
        )
        _param_clip: tuple[float, float] | None = (-5.0, 5.0)
        _param_scale = 1.0
    else:
        _active_modes = ()
        _dim = int(phase_grid) * int(phase_grid)
        _param_clip = None
        # The auto learning schedule returns lr/delta ~1-3, tuned for Zernike
        # amplitudes. Freeform parameters are radians, so scale down.
        _param_scale = 0.1

    # --- SLM↔camera relative rotation search (extra SPGD DOF) ---
    _rotation_search_deg = float(rotation_search_deg)
    _has_rotation = _rotation_search_deg > 0.0
    _rot_clip: tuple[float, float] | None = (
        (-_rotation_search_deg / 2.0, _rotation_search_deg / 2.0)
        if _has_rotation
        else None
    )
    _rot_delta = 1.0  # fixed 1° perturbation step for the rotation DOF
    if _has_rotation:
        _dim += 1  # rotation angle is the last element of the parameter vector
        logger.info(
            "启用 SLM↔相机旋转校正: 搜索范围 ±{:.1f}°, 作为第 {} 维 SPGD 自由度 (初始 0°, 扰动 {:.1f}°)",
            _rotation_search_deg / 2.0,
            _dim,
            _rot_delta,
        )

    # Heuristic-search bounds: zernike coefficients are clipped to ±5, freeform
    # parameters are radians. Rotation is an extra DOF with its own bounds, which
    # the shared driver's single (lo, hi) cannot express — reject that combination
    # instead of searching with wrong bounds.
    if algorithm != "spgd" and _has_rotation:
        raise ValueError(
            "启发式搜索暂不支持 rotation_search_deg > 0 "
            "(旋转角是额外自由度, 与统一 bounds 不兼容); 请使用 --algorithm spgd"
        )
    _heuristic_bounds: tuple[float, float] = (
        _param_clip if _param_clip is not None else (-np.pi, np.pi)
    )

    # --- Debug-record representation -------------------------------------
    # `_c` / `_grad` are recorded in FULL Noll order for the Zernike basis so
    # `ml.hwdataset.index` classifies the debug record as ZERNIKE (rule 1) --
    # an active-mode vector of length 2 matches no accepted length (TODO H-23).
    # The optimiser's own work-vector stays active-mode; this only remaps what is
    # *recorded*. Freeform (`_c` = grid*grid radians) is passed through untouched.
    if basis == "zernike":
        _noll_position = {mode: i for i, mode in enumerate(_noll_modes)}
        _active_pos = [_noll_position[m] for m in _active_modes]

        def _record_vec(vec: np.ndarray) -> np.ndarray:
            return _scatter_active_to_noll(vec, _active_pos, nk, _has_rotation)

    else:

        def _record_vec(vec: np.ndarray) -> np.ndarray:
            return vec

    with (
        create_camera(
            cam_type,
            cam_id=cam_id,
            exposure_time_ms=exposure_time_ms,
            skip_sampling=False,
        ) as cam,
        _slm_cls(slm_type)(
            slm_number=slm_number, wavelength=slm_wavelength or None
        ) as slm,
    ):
        if cam_type == "daheng":
            # The exposure ceiling is set by the optical path, not the sensor,
            # so resolve it per session before any measurement can fire an
            # over-limit exposure: explicit config, env, else probe the image.
            ceiling = resolve_daheng_exposure_limit(camera=cam)
            if exposure_time_ms > ceiling + 1e-9:
                logger.warning(
                    "Daheng exposure_time_ms {} ms exceeds resolved ceiling {} ms; "
                    "clamping to ceiling",
                    exposure_time_ms,
                    ceiling,
                )
                exposure_time_ms = ceiling
            # The probe left the camera at the last ladder step; re-arm the
            # working exposure before the first measurement.
            cam.reset_exposure_time(exposure_time_ms)
        blaze_choice = None
        if blaze_period_px >= 0:
            from ao_shaping.optimizer.wfless._blaze_calibration import select_blaze

            actual_exposure = float(cam.reset_exposure_time(exposure_time_ms))
            if abs(actual_exposure - exposure_time_ms) > 0.001:
                raise RuntimeError("CCD did not accept the fixed blaze probe exposure")

            def capture_blaze(phase: np.ndarray) -> np.ndarray:
                slm.display_data(slm.create_phase_from_array(phase))
                time.sleep(SLM_RESPONSE_TIME_S)
                frame = np.asarray(cam.get_numpy_image(1), dtype=np.float64)
                if cam_type == "daheng" and float(frame.max()) >= 250:
                    raise RuntimeError("blaze CCD probe saturated; reduce exposure")
                return frame

            blaze_choice = select_blaze(
                capture_blaze,
                (SLM_HEIGHT, SLM_WIDTH),
                target_side=target_side,
                window_size=cam_size,
                period_px=blaze_period_px,
                direction=blaze_direction,
            )
            center = blaze_choice.signal_xy
            logger.info("measured blaze: {}", blaze_choice)
        recorder.blaze_choice = blaze_choice
        # Initialize parameter vector (zernike: mapped onto the active modes)
        if basis == "zernike":
            if init_c is None or len(init_c) == 0:
                _params = np.zeros(_dim, dtype=np.float64)
            else:
                _params = _map_init_to_active_modes(
                    np.asarray(init_c, dtype=np.float64), n_max, _active_modes
                )
        else:
            # Freeform init. HARDWARE MEASURED (2026-10-01, bench 2f/SLM #1 +
            # Daheng, report/fouriergsnet_pipeline/hardware_run_20261001.md): the
            # flat state is the BEST-focus state (FWHM 13.6 px, hollowness 0.92),
            # and a full-amplitude random start destroys it -- the failed run
            # began at 0-order peak 225 and ended at 17, with its best iterate
            # being the initial one. A dense +/-0.5 rad random phase was measured
            # to leave the in-box energy unchanged within 2.6%, i.e. it
            # redistributes light rather than focusing it, so starting there just
            # wastes the objective's dynamic range. Default to flat (zeros) and
            # let `init_amplitude_rad` request a small random escape.
            if init_amplitude_rad > 0.0:
                _params = rng.uniform(
                    -float(init_amplitude_rad), float(init_amplitude_rad), size=_dim
                ).astype(np.float64)
            else:
                _params = np.zeros(_dim, dtype=np.float64)

        # Rotation DOF starts at 0° (middle of the search range)
        if _has_rotation:
            _params = np.concatenate([_params, np.zeros(1, dtype=_params.dtype)])
        _opt_dim = _dim  # full dimension including rotation (used by optimizer)

        # GUI-consistent zernike generation (multi_slm_controller.py): uint16
        # grayscale, min-max normalised inside a circular aperture of radius
        # `zernike_radius`. Built once per run (generator cached in helper).
        _pattern_helper: PatternHelper | None = (
            PatternHelper(SLM_RESOLUTION, bits=slm.Gray_Scale_bits or 10)
            if basis == "zernike"
            else None
        )

        def _rotate_pattern(arr: np.ndarray, angle_deg: float) -> np.ndarray:
            """Rotate a phase map by angle_deg degrees around its center.

            傅里叶变换保旋转: SLM 近场相位旋转 θ → 相机远场同向旋转 θ,
            用于补偿 SLM↔相机 的相对装配旋转 (metrics 保持相机轴对齐不变)。
            """
            from scipy.ndimage import rotate

            if angle_deg == 0.0:
                return arr
            return rotate(
                arr,
                angle_deg,
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
            )

        def _params_to_gray(p: np.ndarray) -> np.ndarray:
            """Parameter vector -> uint16 grayscale phase map on the SLM panel.

            zernike: ``PatternHelper.generate_zernike_polynomial`` (radians,
            identical to the GUI Zernike branch) then the SLM driver
            ``slm.create_phase_from_array()``; freeform: raw radians
            converted by the SLM driver. If rotation search is enabled the LAST
            element of ``p`` is the rotation angle (degrees); the base pattern
            is generated from the remaining elements and then rotated.
            """
            if _has_rotation:
                _rot = float(p[-1])
                _head = p[:-1]
            else:
                _rot = 0.0
                _head = p
            if basis == "zernike":
                assert _pattern_helper is not None
                assert zernike_radius is not None  # set in the basis block above
                coeffs_dict = {
                    m: float(_head[i])
                    for i, m in enumerate(_active_modes)
                    if i < len(_head)
                }
                _rad = _pattern_helper.generate_zernike_polynomial(
                    coefficients=coeffs_dict,
                    radius=float(zernike_radius),
                    n_max=n_max,
                )
                if _has_rotation:
                    _rad = _rotate_pattern(_rad, _rot)
                if blaze_choice is not None:
                    from ao_shaping.utils.wavefront.zero_order_overlay import (
                        apply_zero_order_overlay,
                    )

                    _rad = apply_zero_order_overlay(_rad, blaze_choice.params)
                return slm.create_phase_from_array(_rad)
            _rad = _freeform_phase_radians(_head, SLM_RESOLUTION, int(phase_grid))
            if _has_rotation:
                _rad = _rotate_pattern(_rad, _rot)
            if blaze_choice is not None:
                from ao_shaping.utils.wavefront.zero_order_overlay import (
                    apply_zero_order_overlay,
                )

                _rad = apply_zero_order_overlay(_rad, blaze_choice.params)
            return slm.create_phase_from_array(_rad)

        def _apply_update(p: np.ndarray, upd: np.ndarray) -> np.ndarray:
            """Apply an optimizer update, respecting the basis bounds.

            With rotation search active, the last element (rotation angle) is
            clipped to `_rot_clip`; the head (Zernike/freeform part) follows the
            basis bounds.
            """
            p = p + upd
            if _has_rotation:
                assert _rot_clip is not None  # set when rotation search is enabled
                _head = p[:-1]
                _rot = float(np.clip(p[-1], *_rot_clip))
                if _param_clip is not None:
                    _head = np.clip(_head, *_param_clip)
                else:
                    _head = np.mod(_head + np.pi, 2.0 * np.pi) - np.pi
                return np.concatenate([_head, np.array([_rot], dtype=p.dtype)])
            if _param_clip is not None:
                return np.clip(p, *_param_clip)
            return np.mod(p + np.pi, 2.0 * np.pi) - np.pi

        # Display initial phase (uint16 grayscale, direct write — no radian wrap)
        slm.display_data(_params_to_gray(_params))
        time.sleep(SLM_RESPONSE_TIME_S)

        # Auto-exposure for initial image
        _img = capture_with_exposure(
            cam,
            exposure_time_ms if blaze_choice is not None else 0.0,
            TEST_EXPOSURE_TIME_BRIGHTNESS,
            auto_n_sample=20,
        )

        # Track the user-chosen mode for post-exposure re-detection:
        #   None/"shape"  : smart_zero_order_center (argmax + flat-core centroid)
        #   "centroid_thresh" : threshold centroid (threshold=0.1×max)
        #   "max"         : global argmax
        #   "mass"        : full-image centroid
        #   (x, y) tuple  : explicit — fixed, not re-detected
        _center_mode: str | None = None
        if center is None:
            _center_mode = "shape"
            center = _locate_square_center(_img, _center_mode)
        elif isinstance(center, str):
            _center_mode = center
            _img = cam.get_numpy_image(10)
            center = _locate_square_center(_img, _center_mode)
        else:
            _center_mode = "fixed"

        log_center_brightness(_img, center, get_camera_exposure_ms(cam))

        # NOTE: this runner does NOT resize the CCD window (user choice) —
        # full-frame capture is kept; cam_size only bounds the target side.

        # Set exposure
        init_img = capture_with_exposure(
            cam,
            exposure_time_ms,
            target_max_brightness or ADVISE_EXPOSURE_TIME_BRIGHTNESS,
            auto_n_sample=20,
        )
        logger.debug(
            f"Initial Image Max brightness: {np.max(init_img)} "
            f"@ {get_camera_exposure_ms(cam)}ms"
        )

        # Re-detect the spot centre on the (full-frame) image so the target box
        # is guaranteed to sit on the beam (AGENTS.md: 0-order located by argmax,
        # never by geometry — it can sit off the frame centre).
        # Post-exposure re-detection by the selected mode; explicit (x, y) tuples
        # are fixed and not re-detected. When fix_target=True (default), the
        # initial center is kept and NOT re-detected after exposure adjustment.
        if _center_mode != "fixed" and not _fix_target:
            center = _locate_square_center(init_img, _center_mode)
        logger.info(f"target box center @ {center} (fix_target={_fix_target})")

        # Compute target square side length
        if target_mean_brightness > 0:
            # 平均亮度模式: 能量守恒推导边长 side = sqrt(total_brightness / target_mean_brightness)
            _w, _h = init_img.shape[1], init_img.shape[0]
            total_brt = float(np.sum(init_img))
            if total_brt <= 0:
                raise ValueError(
                    "target_mean_brightness 模式需要初始图像总亮度 > 0 "
                    f"(当前 total={total_brt:.1f})"
                )
            target_side = int(round(np.sqrt(total_brt / target_mean_brightness)))
            target_side = max(target_side, 20)  # minimum 20px
            target_side = min(target_side, min(cam_size, _w, _h) - 4)
            logger.info(
                f"target_mean_brightness={target_mean_brightness:.2f} → "
                f"auto target_side={target_side} (total_brt={total_brt:.1f})"
            )
        elif target_side <= 0:
            # Auto-compute from spot size (90% encircled energy diameter)
            _w, _h = init_img.shape[1], init_img.shape[0]
            from ao_shaping.algorithm.goal_functions.target_func import ImageTargetFunc

            _target_func = ImageTargetFunc(_w, _h, center)
            spot_radius = _target_func.radius(init_img, energy=0.90)
            target_side = int(spot_radius * 2 * side_factor)
            target_side = max(target_side, 20)  # minimum 20px
            target_side = min(target_side, min(cam_size, _w, _h) - 4)
            logger.info(
                f"Auto target_side={target_side} (spot_r={spot_radius:.1f}, factor={side_factor})"
            )
        else:
            logger.info(f"User target_side={target_side}")

        # Compute initial metrics
        cost, cv, mean_int = square_uniformity_cost(init_img, center, target_side)
        ee = square_encircled_energy(init_img, center, target_side)
        ar = square_aspect_ratio(init_img, center, target_side)
        pbr = square_peak_to_background_ratio(init_img, center, target_side)
        quality = square_objective_score(
            init_img,
            cv,
            ee,
            ar,
            center,
            target_side,
            objective,
            w_uniformity,
            w_efficiency,
            w_aspect,
            peak_to_background=pbr,
            w_pbr=w_pbr,
        )

        best_quality = quality
        best_cost = cost
        best_cv = cv
        best_ee = ee
        best_ar = ar
        best_c = _params.copy()
        best_img = init_img.copy()
        last_best_epoch = 0

        # ------------------------------------------------------------------
        # In-ROI energy guard, armed from the initial (flat) frame.
        #
        # HARDWARE MEASURED (2026-10-01): without this guard the square search
        # is free to raise the combined score by SCATTERING light out of the
        # box, because `square_quality_score` is w_cv*exp(-2*cv) + w_ee*ee with
        # no absolute-energy term. The failed run traded 6x of encircled energy
        # (0.158 -> 0.026) for a +0.0118 uniformity gain and ended 35.9% worse.
        #
        # Deliberately NOT a bare sentinel: the slm-pib family only subtracts
        # 1e3 from the objective, which protects the best-so-far track but still
        # lets `diff = pos_q - neg_q` build a huge REAL gradient from the
        # penalised pair. Here the epoch is SKIPPED outright, so the
        # coefficients never move toward a state that empties the box.
        # ------------------------------------------------------------------
        _guard_enabled = float(max_roi_energy_loss) > 0.0
        _guard_ref_ee: float | None = None
        if _guard_enabled:
            if not 0.0 <= float(max_roi_energy_loss) <= 1.0:
                raise ValueError(
                    "max_roi_energy_loss must be within 0..1 "
                    f"(0 disables the guard), got {max_roi_energy_loss!r}"
                )
            _guard_ref_ee = float(ee) if np.isfinite(ee) and ee > 0.0 else None
            if _guard_ref_ee is None:
                _guard_enabled = False
                logger.warning(
                    "ROI energy guard disabled: initial encircled energy is not "
                    "positive (ee={})",
                    ee,
                )
            else:
                logger.info(
                    "ROI energy guard armed: reference ee={:.4f}, max loss {:.1%}",
                    _guard_ref_ee,
                    float(max_roi_energy_loss),
                )
        _n_energy_gated = 0

        # Record initial state
        recorder.append(
            {
                "J": cost,
                _score_column: quality,
                "cv": cv,
                "ee": ee,
                "ar": ar,
                "pbr": pbr,
                "side": target_side,
                RECORD_COEFFS: _record_vec(_params),
                RECORD_IMAGE: init_img,
                RECORD_DIFF: 0,
                "lr": lr,
                "delta": delta,
                RECORD_EPOCH: 0,
                "exp_t": get_camera_exposure_ms(cam),
                "max_brt": np.max(init_img),
                "mean_b": mean_int,
                "target_mean_b": target_mean_brightness,
                RECORD_GRAD: _record_vec(np.zeros_like(_params)),
                "optimizer": optimizer_type,
                f"best_{_score_column}": best_quality,
            }
        )

        # pygame 实时显示 (2×2 面板), 初始化失败则禁用显示但不中断优化
        display_stack = contextlib.ExitStack()
        display_ctx: SlmSquareDisplay | None = None
        if show:
            try:
                display_ctx = SlmSquareDisplay()
                display_stack.enter_context(display_ctx)
            except Exception as _e:
                logger.warning("pygame显示初始化失败, 本次运行禁用显示: {}", _e)
                display_ctx = None
                display_stack.close()

        # ── Black-box heuristic search branch (shared driver) ──────────────
        if algorithm != "spgd":
            last_eval: dict = {}

            def _evaluate_quality(params: np.ndarray) -> float:
                """Load one phase and return its HIGHER-IS-BETTER square score."""
                slm.display_data(_params_to_gray(params))
                time.sleep(SLM_RESPONSE_TIME_S)
                img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                _cost, _cv, _mean = square_uniformity_cost(img, center, target_side)
                _ee = square_encircled_energy(img, center, target_side)
                _ar = square_aspect_ratio(img, center, target_side)
                _pbr = square_peak_to_background_ratio(img, center, target_side)
                _q = square_objective_score(
                    img,
                    _cv,
                    _ee,
                    _ar,
                    center,
                    target_side,
                    objective,
                    w_uniformity,
                    w_efficiency,
                    w_aspect,
                    peak_to_background=_pbr,
                    w_pbr=w_pbr,
                )
                last_eval.update(
                    {
                        "img": img,
                        "cost": float(_cost),
                        "cv": float(_cv),
                        "ee": float(_ee),
                        "ar": float(_ar),
                        "pbr": float(_pbr),
                        "mean": float(_mean),
                        "gray": _params_to_gray(params),
                    }
                )
                return float(_q)

            with tqdm.tqdm(
                total=None, desc=f"slm_square {algorithm}", dynamic_ncols=True
            ) as hbar:

                def _on_evaluate(params: np.ndarray, value: float, index: int) -> None:
                    nonlocal best_quality, best_cost, best_cv, best_ee, best_ar
                    nonlocal best_c, best_img, last_best_epoch
                    if value > best_quality + 1e-6:
                        best_quality = float(value)
                        best_cost = last_eval["cost"]
                        best_cv = last_eval["cv"]
                        best_ee = last_eval["ee"]
                        best_ar = last_eval["ar"]
                        best_c = np.asarray(params, dtype=np.float64).copy()
                        best_img = last_eval["img"].copy()
                        last_best_epoch = index

                    row = {
                        "J": last_eval["cost"],
                        _score_column: float(value),
                        "cv": last_eval["cv"],
                        "ee": last_eval["ee"],
                        "ar": last_eval["ar"],
                        "pbr": last_eval["pbr"],
                        "side": target_side,
                        RECORD_COEFFS: _record_vec(
                            np.asarray(params, dtype=np.float64)
                        ),
                        RECORD_IMAGE: last_eval["img"],
                        RECORD_DIFF: 0.0,
                        "lr": 0.0,
                        "delta": float(delta),
                        RECORD_EPOCH: index,
                        "exp_t": get_camera_exposure_ms(cam),
                        "max_brt": float(np.max(last_eval["img"])),
                        "mean_b": last_eval["mean"],
                        "target_mean_b": target_mean_brightness,
                        RECORD_GRAD: _record_vec(np.zeros_like(params)),
                        "optimizer": algorithm,
                    }
                    recorder.append(row)

                    if display_ctx is not None:
                        try:
                            display_ctx.update(
                                img=last_eval["img"],
                                phase=last_eval["gray"],
                                quality=float(value),
                                cv=last_eval["cv"],
                                ee=last_eval["ee"],
                                epoch=index,
                                total_epochs=epochs,
                            )
                        except Exception:
                            pass

                    hbar.set_postfix({k: v for k, v in row.items() if k[0] != "_"})
                    hbar.update(1)

                result = run_heuristic_search(
                    algorithm,
                    _evaluate_quality,
                    dim=_dim,
                    iterations=epochs,
                    bounds=_heuristic_bounds,
                    x0=_params,
                    maximize=True,
                    seed=random_seed,
                    pop_size=pop_size,
                    on_evaluate=_on_evaluate,
                )

            # Leave the best phase found on the SLM.
            slm.display_data(_params_to_gray(result.best_x))
            logger.info(
                "{} search finished: best quality={:.4f} over {} evaluations "
                "(epoch {})",
                algorithm,
                result.best_value,
                result.evaluations,
                last_best_epoch,
            )
            display_stack.close()
            return recorder

        # Create optimizer
        optimizer = create_optimizer(
            optimizer_type=optimizer_type,
            dim=_opt_dim,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            **kwargs,
        )
        if lr == 0:
            optimizer.lr, delta = learning_schedule(
                cv=cv,
                encircled_energy=ee,
                gradient_history=_gradient_history,
                cv_history=_cv_history,
                epoch=0,
            )
            optimizer.lr *= _param_scale
            delta *= _param_scale

        # Main optimization loop
        with tqdm.tqdm(
            total=epochs, desc=f"slm_square iter {epochs}", dynamic_ncols=True
        ) as bar:
            for epoch in range(1, epochs + 1):
                # Adaptive SPGD (only active when --adaptive is on): schedule the
                # quality weights from stage_weights by epoch thirds, then detect
                # best-CV stagnation over the last `stagnate_win` epochs and
                # temporarily boost w_uniformity.
                if _adaptive_cfg is not None:
                    w_uniformity, w_efficiency, w_aspect = get_stage_weights(
                        _adaptive_cfg, epoch, epochs
                    )
                    if (
                        _adaptive_cfg.enable_stagnate_boost
                        and _adaptive_cfg.stagnate_boost > 1.0
                        and should_boost_stagnation(
                            _cv_history,
                            win=_adaptive_cfg.stagnate_win,
                            min_improve_frac=_adaptive_cfg.min_improve_frac,
                        )
                    ):
                        if not _adaptive_boost_active:
                            logger.info(
                                "Adaptive SPGD: best-CV stagnation over the last "
                                "{} epochs (win={}, min_improve_frac={}); boosting "
                                "w_uniformity x{} for this epoch",
                                _adaptive_cfg.stagnate_win,
                                _adaptive_cfg.stagnate_win,
                                _adaptive_cfg.min_improve_frac,
                                _adaptive_cfg.stagnate_boost,
                            )
                        _adaptive_boost_active = True
                        w_uniformity *= _adaptive_cfg.stagnate_boost
                    else:
                        _adaptive_boost_active = False

                # Generate random perturbation (±1 pattern), dim = _dim
                disturb_c = rng.binomial(1, 0.5, (_dim,)).astype(float) * 2.0 - 1.0
                disturb_c = disturb_c * delta

                if _has_rotation:
                    # rotation DOF uses a fixed 1° step, not the amplitude delta
                    disturb_c[-1] = np.sign(disturb_c[-1]) * _rot_delta
                    if disturb_c[-1] == 0.0:
                        disturb_c[-1] = _rot_delta

                # Positive perturbation
                _pos_c = _apply_update(_params, disturb_c)
                slm.display_data(_params_to_gray(_pos_c))
                time.sleep(SLM_RESPONSE_TIME_S)
                pos_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                pos_cost, pos_cv, pos_mean = square_uniformity_cost(
                    pos_img, center, target_side
                )
                pos_ee = square_encircled_energy(pos_img, center, target_side)
                pos_ar = square_aspect_ratio(pos_img, center, target_side)
                pos_pbr = square_peak_to_background_ratio(pos_img, center, target_side)
                pos_q = square_objective_score(
                    pos_img,
                    pos_cv,
                    pos_ee,
                    pos_ar,
                    center,
                    target_side,
                    objective,
                    w_uniformity,
                    w_efficiency,
                    w_aspect,
                    peak_to_background=pos_pbr,
                    w_pbr=w_pbr,
                )

                # Negative perturbation
                _neg_c = _apply_update(_params, -disturb_c)
                slm.display_data(_params_to_gray(_neg_c))
                time.sleep(SLM_RESPONSE_TIME_S)
                neg_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
                neg_cost, neg_cv, neg_mean = square_uniformity_cost(
                    neg_img, center, target_side
                )
                neg_ee = square_encircled_energy(neg_img, center, target_side)
                neg_ar = square_aspect_ratio(neg_img, center, target_side)
                neg_pbr = square_peak_to_background_ratio(neg_img, center, target_side)
                neg_q = square_objective_score(
                    neg_img,
                    neg_cv,
                    neg_ee,
                    neg_ar,
                    center,
                    target_side,
                    objective,
                    w_uniformity,
                    w_efficiency,
                    w_aspect,
                    peak_to_background=neg_pbr,
                    w_pbr=w_pbr,
                )

                # Auto-exposure adjustment if saturated
                max_brightness = max(np.max(pos_img), np.max(neg_img))
                if max_brightness == 255 and exposure_time_ms == 0:
                    _resample_img = resample_on_saturation(
                        pos_img,
                        cam,
                        exposure_time_ms,
                        target_max_brightness,
                        auto_exposure_fn=lambda c, t, n_sample=1: c.auto_exposure(
                            target_max=t, twice_valid=False, n_sample=max(n_sample, 20)
                        ),
                        saturation_threshold=float(max_brightness),
                    )
                    optimizer.scale_momentum(np.sum(_resample_img) / np.sum(pos_img))

                # In-ROI energy guard: abandon the whole epoch if either
                # perturbation has driven the in-box energy fraction below
                # (1 - max_roi_energy_loss) of the armed reference. Skipping
                # (rather than only penalising the score) is essential -- see the
                # arming comment above.
                _gated = False
                if _guard_enabled and _guard_ref_ee is not None:
                    _pos_loss = (_guard_ref_ee - pos_ee) / _guard_ref_ee
                    _neg_loss = (_guard_ref_ee - neg_ee) / _guard_ref_ee
                    if (
                        _pos_loss > max_roi_energy_loss
                        or _neg_loss > max_roi_energy_loss
                    ):
                        _gated = True
                        _n_energy_gated += 1
                        _gate = "energy"
                        if _n_energy_gated <= 5 or _n_energy_gated % 50 == 0:
                            logger.warning(
                                "epoch {}: in-ROI energy loss {:.1%}/{:.1%} exceeds "
                                "{:.1%} - epoch skipped (#{} gated)",
                                epoch,
                                _pos_loss,
                                _neg_loss,
                                float(max_roi_energy_loss),
                                _n_energy_gated,
                            )
                if _gated:
                    # Record honestly (unchanged coefficients, diff=0) and skip.
                    recorder.append(
                        {
                            "J": (pos_cost + neg_cost) / 2,
                            _score_column: float("nan"),
                            "cv": (pos_cv + neg_cv) / 2,
                            "ee": (pos_ee + neg_ee) / 2,
                            "ar": (pos_ar + neg_ar) / 2,
                            "pbr": (pos_pbr + neg_pbr) / 2,
                            "side": target_side,
                            RECORD_DIFF: 0.0,
                            "lr": optimizer.lr,
                            "delta": delta,
                            RECORD_EPOCH: epoch,
                            RECORD_COEFFS: _record_vec(_params.copy()),
                            RECORD_IMAGE: pos_img,
                            "exp_t": get_camera_exposure_ms(cam),
                            "max_brt": max_brightness,
                            "mean_b": (pos_mean + neg_mean) / 2,
                            "target_mean_b": target_mean_brightness,
                            RECORD_GRAD: _record_vec(np.zeros_like(disturb_c)),
                            "gate": _gate,
                        }
                    )
                    # Leave the panel on the accepted state, not on the rejected
                    # perturbation (see the end-of-epoch resync below).
                    slm.display_data(_params_to_gray(_params))
                    time.sleep(SLM_RESPONSE_TIME_S)
                    bar.update(1)
                    continue

                # SPGD gradient update. The objective is the combined quality
                # score (uniformity + energy + aspect), NOT -CV alone: optimising
                # -CV alone lets the optimizer minimise CV by EMPTYING the target
                # box (observed on hardware: EE collapsed to ~0.002).
                diff = pos_q - neg_q
                gradient = diff * disturb_c
                update = optimizer.update(gradient)
                _params = _apply_update(_params, update)

                # Metrics for the better perturbation
                if pos_q >= neg_q:
                    eval_img, eval_cv, eval_ee, eval_ar, eval_c, eval_pbr = (
                        pos_img,
                        pos_cv,
                        pos_ee,
                        pos_ar,
                        _pos_c.copy(),
                        pos_pbr,
                    )
                else:
                    eval_img, eval_cv, eval_ee, eval_ar, eval_c, eval_pbr = (
                        neg_img,
                        neg_cv,
                        neg_ee,
                        neg_ar,
                        _neg_c.copy(),
                        neg_pbr,
                    )

                quality = square_objective_score(
                    eval_img,
                    eval_cv,
                    eval_ee,
                    eval_ar,
                    center,
                    target_side,
                    objective,
                    w_uniformity,
                    w_efficiency,
                    w_aspect,
                    peak_to_background=eval_pbr,
                    w_pbr=w_pbr,
                )
                J = (pos_cost + neg_cost) / 2

                # Best-tracking uses the FROZEN base weights (adaptive contract):
                # the logged `quality` reflects the scheduled/boosted weights,
                # while best-* selection must stay comparable across epochs.
                _track_quality = (
                    square_objective_score(
                        eval_img,
                        eval_cv,
                        eval_ee,
                        eval_ar,
                        center,
                        target_side,
                        objective,
                        _base_w_uniformity,
                        _base_w_efficiency,
                        _base_w_aspect,
                        peak_to_background=eval_pbr,
                        w_pbr=w_pbr,
                    )
                    if _adaptive_cfg is not None
                    else quality
                )

                # Track best
                if _track_quality > best_quality + 1e-6:
                    best_quality = _track_quality
                    best_cost = float(J)
                    best_cv = eval_cv
                    best_ee = eval_ee
                    best_ar = eval_ar
                    best_c = eval_c.copy()
                    best_img = eval_img.copy()
                    last_best_epoch = epoch

                # Adaptive learning schedule. Only touches `delta` when the caller
                # did NOT pin one: an explicit `--delta` used to be silently
                # overwritten here whenever `lr` was left at 0 (its default),
                # which made the flag a no-op unless you also passed an explicit
                # `--lr`. `--delta` pinned is now reported, not discarded.
                if lr == 0:
                    _grad_mag = float(np.linalg.norm(gradient))
                    _gradient_history.append(_grad_mag)
                    _cv_history.append(eval_cv)
                    if len(_gradient_history) > _max_history_len:
                        _gradient_history.pop(0)
                        _cv_history.pop(0)
                    optimizer.lr, _auto_delta = learning_schedule(
                        cv=eval_cv,
                        encircled_energy=eval_ee,
                        gradient_history=_gradient_history,
                        cv_history=_cv_history,
                        epoch=epoch,
                    )
                    optimizer.lr *= _param_scale
                    if not _delta_pinned:
                        delta = _auto_delta * _param_scale
                    elif epoch == 1:
                        logger.info(
                            "delta pinned at {:.4g} rad by the caller; the adaptive "
                            "schedule will not override it",
                            delta,
                        )
                    delta *= _param_scale

                # Adaptive-only: the CV history is normally collected only for
                # the `lr == 0` schedule, but stagnation detection needs it for
                # ANY lr once --adaptive is on.
                elif _adaptive_cfg is not None:
                    _cv_history.append(eval_cv)
                    _max_history_len = max(_max_history_len, len(_cv_history))
                    if len(_cv_history) > _max_history_len:
                        _cv_history.pop(0)

                log = {
                    "J": J,
                    "gate": "applied",
                    _score_column: quality,
                    "cv": eval_cv,
                    "ee": eval_ee,
                    "ar": eval_ar,
                    "pbr": eval_pbr,
                    "side": target_side,
                    RECORD_DIFF: diff,
                    "lr": optimizer.lr,
                    "delta": delta,
                    RECORD_EPOCH: epoch,
                    RECORD_COEFFS: _record_vec(eval_c),
                    RECORD_IMAGE: eval_img,
                    "exp_t": get_camera_exposure_ms(cam),
                    "max_brt": max_brightness,
                    "mean_b": (pos_mean + neg_mean) / 2,
                    "target_mean_b": target_mean_brightness,
                    RECORD_GRAD: _record_vec(gradient),
                }
                recorder.append(log)

                if display_ctx is not None:
                    try:
                        display_ctx.update(
                            img=eval_img,
                            phase=_params_to_gray(eval_c),
                            quality=quality,
                            cv=eval_cv,
                            ee=eval_ee,
                            epoch=epoch,
                            total_epochs=epochs,
                        )
                    except Exception:
                        pass

                bar.set_postfix({k: v for k, v in log.items() if k[0] != "_"})
                bar.update(1)

                # Re-display the ACCEPTED state. Without this the last write of
                # every epoch is `_neg_c` (the rejected -delta perturbation), so
                # the panel sits on a state 2*delta away in all 576 dimensions
                # while the loop believes it is at `_params`. That desync is a
                # plausible contributor to the 0-order collapse seen on hardware
                # (peak 225 -> 17). It also makes the next epoch's baseline
                # ambiguous, which matters once the energy guard is armed.
                slm.display_data(_params_to_gray(_params))
                time.sleep(SLM_RESPONSE_TIME_S)

        # Reset SLM to flat phase on exit
        if SLM_RESET_ON_EXIT:
            slm.set_grayscale(0)

        logger.info(
            f"Optimization complete. Best quality={best_quality:.4f} "
            f"(CV={best_cv:.4f}, EE={best_ee:.4f}, AR={best_ar:.4f}) "
            f"@ epoch {last_best_epoch}"
        )
        if _guard_enabled:
            logger.info(
                "Energy guard: {} of {} epochs skipped", _n_energy_gated, epochs
            )

        display_stack.close()

        return recorder
