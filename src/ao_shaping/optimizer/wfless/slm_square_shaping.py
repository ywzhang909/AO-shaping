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
    ...     slm_wavelength=1064,   # SLM 标称波长 (nm), 1064 表示 NIR 模式
    ...     optimizer_type="adamod",
    ...     w_uniformity=0.4,      # 均匀性 (CV) 权重
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
- 依赖真实硬件 (SLM200 + Daheng/MiiCam 相机), 无模拟模式。
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
import inspect
import os
import time
from typing import TYPE_CHECKING

import tqdm
import numpy as np

from ao_shaping.drivers import CameraStreamManager
from ao_shaping.drivers.slm import SantecSLM200
from ao_shaping.algorithm.adam import AdaMOD, Adam, AdamW, Base, Muno, MunoW, SGD
from ao_shaping.utils import logger, Recorder
from ao_shaping.utils.file import gen_date_dir, gen_date_str
from ao_shaping.utils.pattern_helper import PatternHelper
from ao_shaping.utils.spots_calc import centroid, radius
from ao_shaping.utils.zernike_calc import (
    ZernikeGenerator,
    calc_n_zernike_terms,
    noll_to_nm,
)

if TYPE_CHECKING:
    import pygame

# Adam parameters
beta1 = 0.9
beta2 = 0.99
beta3 = 0.9999

# Camera parameters
CAM_SAMPLE_ITER = 1
ADVISE_EXPOSURE_TIME_BRIGHTNESS = int(255 / 3)
TEST_EXPOSURE_TIME_BRIGHTNESS = 220
IDEAL_SPOT_RADIUS = int(os.environ.get("IDEAL_SPOT_RADIUS", 6))

# SLM parameters
SLM_RESPONSE_TIME_S = 0.3  # Santec SLM-200 response time ~300ms
SLM_RESET_ON_EXIT = True

# SLM resolution (from SantecSLM200.Panel_Res = (1920, 1200))
SLM_WIDTH = 1920
SLM_HEIGHT = 1200
SLM_RESOLUTION = (SLM_WIDTH, SLM_HEIGHT)

OPTIMIZER_MAP = {
    "adam": Adam,
    "adamw": AdamW,
    "adamod": AdaMOD,
    "sgd": SGD,
    "muno": Muno,
    "munow": MunoW,
}

# Zernike basis: only these modes are optimised by ``spgd-square``. Defocus
# (2,0) + Spherical (4,0) match the GUI Zernike branch (multi_slm_controller.py)
# at the 600 px aperture radius. Full Noll-length init vectors keep only these
# entries (Noll 4 = (2,0), Noll 11 = (4,0)).
ZERNIKE_ACTIVE_MODES: tuple[tuple[int, int], ...] = ((2, 0), (4, 0))

_ZERNIKE_MIN_MASK_LEN = 5  # piston + tip + tilt always masked; mask must cover ≥ Noll 5


def _zernike_indices(n_max: int) -> list[tuple[int, int]]:
    """Return list of (n, m) pairs for all valid Zernike modes up to n_max.

    Uses noll_to_nm from zernike_calc for correctness.
    """
    n_terms = calc_n_zernike_terms(n_max)
    modes = []
    for j in range(1, n_terms + 1):
        n, m = noll_to_nm(j)
        if n <= n_max:
            modes.append((n, m))
    return modes


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


def _create_optimizer(optimizer_type: str, dim: int, lr: float, **kwargs) -> Base:
    """Create the configured optimizer while filtering unsupported kwargs."""
    optimizer_cls = OPTIMIZER_MAP.get(optimizer_type.lower(), AdaMOD)
    filtered_kwargs = {}
    signature = inspect.signature(optimizer_cls.__init__)
    for key, value in kwargs.items():
        if key in signature.parameters:
            filtered_kwargs[key] = value
    return optimizer_cls(dim, lr=lr, **filtered_kwargs)


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
    """Convert a flat free-form low-res phase grid to a wrapped radian phase map.

    Low-order Zernike modes are smooth and cannot synthesise a square
    far-field. A free-form phase grid (the SLM's native degree of freedom)
    can, so this is the default basis for square shaping.

    Args:
        flat_params: Flat array of length ``grid * grid`` (radians per cell).
        resolution: SLM panel resolution as (width, height).
        grid: Side of the square parameter grid.

    Returns:
        float64 phase map in [0, 2pi), shape (height, width).
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
    return np.mod(upsampled, 2.0 * np.pi)


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
) -> float:
    """Compute a combined quality score for the square beam.

    Args:
        cv: Coefficient of variation (lower is better).
        encircled_energy: Fraction of energy in the square (higher is better).
        aspect_ratio: Width/height ratio (1.0 is ideal).
        w_cv: Weight for uniformity component.
        w_ee: Weight for energy efficiency component.
        w_ar: Weight for aspect ratio component.

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
    return float(score)


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

    if gradient_history is None or cv_history is None or len(gradient_history) < 5:
        return base_lr, base_delta

    recent_grads = (
        list(gradient_history[-10:])
        if len(gradient_history) >= 10
        else gradient_history
    )
    recent_cvs = list(cv_history[-10:]) if len(cv_history) >= 10 else cv_history

    grad_mean = np.mean(recent_grads)
    grad_std = np.std(recent_grads) if len(recent_grads) > 1 else 0
    grad_cv = grad_std / (grad_mean + 1e-8)

    cv_trend = (
        (recent_cvs[-1] - recent_cvs[0]) / (len(recent_cvs) + 1e-8)
        if len(recent_cvs) > 1
        else 0
    )

    lr_factor = 1.0
    delta_factor = 1.0

    # Gradient variance control
    if grad_cv < 0.1:
        lr_factor = 0.5
        delta_factor = 0.5
    elif grad_cv < 0.3:
        lr_factor = 0.8
        delta_factor = 0.8
    elif grad_cv > 0.8:
        lr_factor = 0.3
        delta_factor = 1.2

    # CV trend control
    if abs(cv_trend) < 1e-5:  # Stagnant
        delta_factor = max(delta_factor, 1.5)
        lr_factor = min(lr_factor, 0.7)
    elif cv_trend > 0.001:  # CV increasing (diverging)
        lr_factor = 0.4
        delta_factor = 0.6
    elif cv_trend < -0.001:  # CV decreasing (converging)
        lr_factor = min(lr_factor, 1.1)

    # Early warmup
    if epoch < 20:
        lr_factor *= 1.2
        delta_factor *= 1.1

    lr_factor = np.clip(lr_factor, 0.2, 2.0)
    delta_factor = np.clip(delta_factor, 0.3, 2.5)

    return base_lr * lr_factor, base_delta * delta_factor


class _SPGDDisplay:
    """SPGD square-beam 优化实时 pygame 可视化窗口.

    2×2 面板:
        - 左上: SLM 下发相位 (grayscale)
        - 右上: Zernike 系数 (柱状图)
        - 左下: CCD 远场图像
        - 右下: 优化指标曲线 (quality / CV / EE)

    用法::

        with _SPGDDisplay() as disp:
            disp.update(phase_gray, coeffs, img, epoch, quality, cv, ee)
    """

    PANEL_W: int = 480
    PANEL_H: int = 480
    PAD: int = 8
    TITLE_H: int = 28
    BG: tuple[int, int, int] = (25, 25, 25)

    def __init__(self, active_modes: tuple[tuple[int, int], ...]) -> None:
        w = self.PANEL_W * 2 + self.PAD * 3
        h = self.PANEL_H * 2 + self.PAD * 3 + self.TITLE_H
        self._window_size = (w, h)
        self._active_modes = active_modes  # e.g. ((2,0),(4,0))
        self._phase: np.ndarray | None = None
        self._image: np.ndarray | None = None
        self._coeffs: np.ndarray | None = None
        self._quality_curve: list[float] = []
        self._cv_curve: list[float] = []
        self._ee_curve: list[float] = []
        self._epoch = 0

    def __enter__(self) -> "_SPGDDisplay":
        import pygame

        pygame.init()
        pygame.display.set_caption("SPGD Square 闭环优化")
        self._screen = pygame.display.set_mode(self._window_size)
        self._font = pygame.font.SysFont("consolas", 16)
        self._clock = pygame.time.Clock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import pygame

        pygame.quit()

    @staticmethod
    def _to_surface(arr: np.ndarray, cmap: str = "gray") -> "pygame.Surface":
        """numpy array → pygame Surface (with swapaxes for surfarray convention)."""
        import pygame

        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            arr = np.zeros((16, 16), dtype=np.float64)
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax - vmin < 1e-9:
            norm = np.zeros_like(arr, dtype=np.uint8)
        else:
            norm = ((arr - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        if cmap == "heat" and norm.ndim == 2:
            f = norm.astype(np.float64) / 255.0
            r = np.clip(1.5 - np.abs(4 * f - 3.0), 0.0, 1.0)
            g = np.clip(1.5 - np.abs(4 * f - 2.0), 0.0, 1.0)
            b = np.clip(1.5 - np.abs(4 * f - 1.0), 0.0, 1.0)
            rgb = np.dstack((r, g, b))
            rgb = (rgb * 255.0).astype(np.uint8)
            return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        if norm.ndim == 2:
            norm = np.dstack((norm, norm, norm))
        return pygame.surfarray.make_surface(norm.swapaxes(0, 1))

    def update(
        self,
        phase_gray: np.ndarray,
        coeffs: np.ndarray,
        img: np.ndarray,
        epoch: int,
        quality: float,
        cv: float,
        ee: float,
    ) -> None:
        """Update all panels and render."""
        self._phase = phase_gray
        self._coeffs = coeffs
        self._image = img
        self._epoch = epoch
        self._quality_curve.append(quality)
        self._cv_curve.append(cv)
        self._ee_curve.append(ee)
        self._render()

    def _draw_panel(self, index: int, title: str, surface: "pygame.Surface") -> None:
        import pygame

        col = index % 2
        row = index // 2
        x = self.PAD + col * (self.PANEL_W + self.PAD)
        y = self.TITLE_H + self.PAD + row * (self.PANEL_H + self.PAD)
        title_surf = self._font.render(title, True, (255, 255, 0))
        self._screen.blit(title_surf, (x + 4, y - self.TITLE_H + 2))
        scaled = pygame.transform.scale(surface, (self.PANEL_W, self.PANEL_H))
        self._screen.blit(scaled, (x, y))
        pygame.draw.rect(
            self._screen, (120, 120, 120), (x, y, self.PANEL_W, self.PANEL_H), 1
        )

    def _draw_coeffs_bar(self) -> None:
        """Panel 1 (index=1): bar chart of Zernike coefficients."""
        import pygame

        col = 1
        row = 0
        x = self.PAD + col * (self.PANEL_W + self.PAD)
        y = self.TITLE_H + self.PAD + row * (self.PANEL_H + self.PAD)
        area = pygame.Rect(x, y, self.PANEL_W, self.PANEL_H)
        self._screen.fill((0, 0, 0), area)
        if self._coeffs is None:
            return
        coeffs = self._coeffs
        n = len(coeffs)
        if n == 0:
            return
        bar_w = max(min((self.PANEL_W - 40) // n, 80), 10)
        gap = 8
        total_w = n * bar_w + (n - 1) * gap
        start_x = x + (self.PANEL_W - total_w) // 2
        cmax = max(abs(float(c)) for c in coeffs) or 1.0
        mid_y = y + self.PANEL_H // 2
        for i, c in enumerate(coeffs):
            bx = start_x + i * (bar_w + gap)
            val = float(c)
            bar_h = int(abs(val) / cmax * (self.PANEL_H // 2 - 20))
            color = (0, 180, 255) if val >= 0 else (255, 100, 100)
            if val >= 0:
                rect = pygame.Rect(bx, mid_y - bar_h, bar_w, bar_h)
            else:
                rect = pygame.Rect(bx, mid_y, bar_w, bar_h)
            pygame.draw.rect(self._screen, color, rect)
            # Label: Noll index
            noll = self._active_modes[i] if i < len(self._active_modes) else (0, 0)
            label = self._font.render(f"({noll[0]},{noll[1]})", True, (200, 200, 200))
            self._screen.blit(label, (bx + 2, y + self.PANEL_H - 18))
            # Value
            vlabel = self._font.render(f"{val:.2f}", True, (255, 255, 255))
            self._screen.blit(
                vlabel, (bx + 2, y + 6 if val >= 0 else y + self.PANEL_H - 34)
            )
        # Center line
        pygame.draw.line(
            self._screen, (80, 80, 80), (x, mid_y), (x + self.PANEL_W, mid_y), 1
        )

    def _draw_metrics_curve(self) -> None:
        """Panel 3 (index=3): multi-curve quality/CV/EE."""
        import pygame

        col = 1
        row = 1
        x = self.PAD + col * (self.PANEL_W + self.PAD)
        y = self.TITLE_H + self.PAD + row * (self.PANEL_H + self.PAD)
        area = pygame.Rect(x, y, self.PANEL_W, self.PANEL_H)
        self._screen.fill((0, 0, 0), area)
        curves = [
            (self._quality_curve, (0, 255, 0), "Quality"),
            (self._cv_curve, (100, 180, 255), "CV"),
            (self._ee_curve, (255, 200, 0), "EE"),
        ]
        # Draw legend
        for ci, (data, color, name) in enumerate(curves):
            lx = x + 10 + ci * 100
            pygame.draw.line(self._screen, color, (lx, y + 8), (lx + 20, y + 8), 2)
            t = self._font.render(name, True, color)
            self._screen.blit(t, (lx + 24, y + 2))
        if len(self._quality_curve) < 2:
            t = self._font.render("Waiting for data...", True, (150, 150, 150))
            self._screen.blit(t, (x + 20, y + self.PANEL_H // 2))
            return
        # Draw each curve
        for data, color, _name in curves:
            if len(data) < 2:
                continue
            cmin, cmax = min(data), max(data)
            span = (cmax - cmin) or 1.0
            points = []
            for i, val in enumerate(data):
                px = x + 10 + int(i * (self.PANEL_W - 20) / max(len(data) - 1, 1))
                py = y + self.PANEL_H - 30 - int(
                    (val - cmin) / span * (self.PANEL_H - 50)
                )
                points.append((px, py))
            pygame.draw.lines(self._screen, color, False, points, 2)
        # Epoch label
        t = self._font.render(
            f"Epoch {self._epoch} | Q={self._quality_curve[-1]:.4f}",
            True,
            (255, 255, 255),
        )
        self._screen.blit(t, (x + 10, y + self.PANEL_H - 22))

    def _render(self) -> None:
        import pygame

        self._screen.fill(self.BG)
        header = (
            f"SPGD Square | Epoch {self._epoch} | CV={self._cv_curve[-1]:.4f}"
            if self._cv_curve
            else "SPGD Square 闭环优化"
        )
        head_surf = self._font.render(header, True, (0, 255, 255))
        self._screen.blit(head_surf, (self.PAD, 6))
        # Panel 0: SLM phase
        if self._phase is not None:
            self._draw_panel(
                0, "SLM Phase (grayscale)", self._to_surface(self._phase, "gray")
            )
        # Panel 1: Zernike coefficients
        self._draw_coeffs_bar()
        # Panel 2: CCD image
        if self._image is not None:
            self._draw_panel(
                2, "CCD Far-field Image", self._to_surface(self._image, "heat")
            )
        # Panel 3: Metrics curve
        self._draw_metrics_curve()
        pygame.display.update()
        self._clock.tick(30)


def optimize_slm_square(
    center: tuple[int, int] | str | None,
    epochs: int,
    n_max: int = 4,
    target_side: int = 0,
    target_mean_brightness: float = 0.0,
    side_factor: float = 1.5,
    delta: float = 0.1,
    lr: float = 0,
    exposure_time_ms: float = 80.0,
    cam_id: int = 0,
    show: bool = False,
    init_c: list[float] | np.ndarray | None = None,
    cam_size: int = 300,
    target_max_brightness: int = 200,
    slm_number: int = 1,
    slm_wavelength: int = 1064,
    optimizer_type: str = "adamod",
    random_seed: int | None = None,
    w_uniformity: float = 0.4,
    w_efficiency: float = 0.6,
    w_aspect: float = 0.0,
    basis: str = "freeform",
    phase_grid: int = 24,
    zernike_radius: float | int | None = None,
    zernike_mask: np.ndarray | None = None,
    rotation_search_deg: float = 0.0,
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
        slm_wavelength: SLM wavelength in nm.
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
        **kwargs: Additional optimizer parameters.

    Returns:
        Recorder: Optimization history recorder.
    """
    delta = abs(delta)
    epochs = int(epochs)
    rng = np.random.default_rng(random_seed)

    # 目标方形参数二选一: 边长(像素) 或 平均亮度, 同时给出报错
    if target_side > 0 and target_mean_brightness > 0:
        raise ValueError(
            "target_side 与 target_mean_brightness 互斥: "
            "方形边长(px) 与 方形平均亮度只能二选一 (both given)"
        )

    recorder = Recorder(mark="quality", mode="max")

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
                    _mask.size, n_max, nk,
                )
                _mask = _mask[:nk]
            # Noll 1,2,3 (piston, tip, tilt) 强制为 0
            _mask[0] = _mask[1] = _mask[2] = 0
            _active_modes: tuple[tuple[int, int], ...] = tuple(
                _noll_modes[i] for i in range(nk) if _mask[i] == 1
            )
            if not _active_modes:
                raise ValueError(
                    f"zernike_mask 全为 0 (强制屏蔽 piston/tip/tilt 后): "
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
    if not (0.0 <= _rotation_search_deg <= 360.0):
        raise ValueError(
            f"rotation_search_deg 必须在 0~360 范围内, 实际 {_rotation_search_deg}"
        )
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

    with (
        CameraStreamManager(
            cam_id=cam_id, exposure_time_ms=exposure_time_ms, skip_sampling=False
        ) as cam,
        SantecSLM200(slm_number=slm_number, wavelength=slm_wavelength) as slm,
    ):
        # Initialize parameter vector (zernike: mapped onto the active modes)
        if basis == "zernike":
            if init_c is None or len(init_c) == 0:
                _params = np.zeros(_dim, dtype=np.float64)
            else:
                _params = _map_init_to_active_modes(
                    np.asarray(init_c, dtype=np.float64), n_max, _active_modes
                )
        else:
            # Small random init escapes the trivial flat-phase stationary point.
            _params = rng.uniform(-np.pi, np.pi, size=_dim).astype(np.float64)

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

            zernike: ``PatternHelper.generate_zernike_polynomial`` (identical
            to the GUI Zernike branch); freeform: wrapped radians converted by
            the SLM driver. If rotation search is enabled the LAST element of
            ``p`` is the rotation angle (degrees); the base pattern is generated
            from the remaining elements and then rotated.
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
                _gray = _pattern_helper.generate_zernike_polynomial(
                    coefficients=coeffs_dict,
                    radius=float(zernike_radius),
                    n_max=n_max,
                )
                return _rotate_pattern(_gray, _rot) if _has_rotation else _gray
            _rad = _freeform_phase_radians(_head, SLM_RESOLUTION, int(phase_grid))
            if _has_rotation:
                _rad = _rotate_pattern(_rad, _rot)
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
        _img = cam.autoset_exposure_time_ms(
            target_max_brightness=TEST_EXPOSURE_TIME_BRIGHTNESS
        )

        def _detect_center(img: np.ndarray) -> tuple[int, int]:
            """Locate the 0-order spot center, reusing axis_beam_runner's method.

            复用 axis_beam_runner (PIB) 的 0 级光斑定位方法:
            ``ImageTargetFunc.intelligen_center`` (algorithm/target_func.py):
            1. ``center_of_brightness`` = 全局最大像素 (argmax) —— 满足 AGENTS.md
               "0 级光斑用 argmax 定位, 绝不用几何默认" 的光轴不在帧中心的约束;
            2. 若该点邻域非空洞 (≥ 40% 峰值), 改用全图二阶矩质心
               ``center_of_mass`` 细化光斑中心 (平坦核心用质心更稳)。
            退化 (全零图) 时回退 argmax。
            """
            from ao_shaping.algorithm.target_func import ImageTargetFunc

            h, w = img.shape
            _tf = ImageTargetFunc(w, h, (w // 2, h // 2))
            _cx, _cy = _tf.intelligen_center(img)
            if not (np.isfinite(_cx) and np.isfinite(_cy)):
                idx = np.unravel_index(int(np.argmax(img)), img.shape)
                _cx, _cy = float(idx[1]), float(idx[0])
            return (int(round(_cx)), int(round(_cy)))

        def _locate_center(img: np.ndarray) -> tuple[int, int]:
            """按所选模式 `_center_mode` 定位 0 级光斑中心 (int 像素坐标).

            统一分派点, 初始采集 (auto-exposure 前) 与曝光后重检共用,
            保证用户所选模式在两次定位间一致。
            """
            if _center_mode in (None, "shape"):
                return _detect_center(img)
            if _center_mode == "centroid_thresh":
                tx, ty = centroid(img, moment=1, threshold=0.1)
                return (int(round(tx)), int(round(ty)))
            if _center_mode == "max":
                idx = np.unravel_index(np.argmax(img), img.shape)
                return (int(idx[1]), int(idx[0]))
            if _center_mode == "mass":
                return (int(centroid(img)[0]), int(centroid(img)[1]))
            raise ValueError(f"Unknown center mode: {_center_mode}")

        # 记录用户所选模式, 供曝光后重检复用:
        #   None/"shape"         : argmax 锚定智能检测 (默认; 平坦核心质心细化)
        #   "centroid_thresh"    : 亮度重心 (阈值质心, threshold=0.1×max)
        #   "max"                : 峰值位置 (全局 argmax)
        #   "mass"               : 全图质心 (无 argmax 锚定, 易被杂散光拉偏)
        #   (x, y) 元组           : 显式坐标 (fixed, 曝光后不重检)
        _center_mode: str | None = None
        if center is None:
            _center_mode = "shape"
            center = _locate_center(_img)
        elif isinstance(center, str):
            _center_mode = center
            _img = cam.get_numpy_image(10)
            center = _locate_center(_img)
        else:
            _center_mode = "fixed"  # 显式 (x, y): 保持用户给定值

        logger.info(
            f"Centroid brightness: {_img[center[1], center[0]]}@{center}, "
            f"Max brightness: {np.max(_img)} @ {cam.exposure_time}ms"
        )

        # 注意: 整形过程不对 CCD 开窗 resize (用户要求) —— 保持全帧采集。
        # 不再调用 cam.reset_window(...); cam_size 仅用于目标方形边长上限
        # (target_side <= min(cam_size, w, h) - 4, 见下方)。

        # Set exposure
        if exposure_time_ms > 0:
            cam.exposure_time = exposure_time_ms
            init_img = cam.get_numpy_image(CAM_SAMPLE_ITER)
        elif 0 < target_max_brightness < 255 and target_max_brightness > 0:
            init_img = cam.autoset_exposure_time_ms(
                target_max_brightness=target_max_brightness, twice_valid=True
            )
        else:
            init_img = cam.autoset_exposure_time_ms(
                target_max_brightness=ADVISE_EXPOSURE_TIME_BRIGHTNESS, twice_valid=True
            )
        logger.debug(
            f"Initial Image Max brightness: {np.max(init_img)} @ {cam.exposure_time}ms"
        )

        # Re-detect the spot centre on the (full-frame) image so the target box
        # is guaranteed to sit on the beam (AGENTS.md: 0-order located by argmax,
        # never by geometry — it can sit off the frame centre).
        # 注意: 曝光后按用户所选模式重检 (shape/max/mass/centroid_thresh);
        # 显式 (x, y) 元组 (fixed) 不重检, 保持用户给定坐标。
        if _center_mode != "fixed":
            center = _locate_center(init_img)
        logger.info(f"target box center @ {center}")

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
            from ao_shaping.algorithm.target_func import ImageTargetFunc

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
        quality = square_quality_score(cv, ee, ar, w_uniformity, w_efficiency, w_aspect)

        best_quality = quality
        best_cost = cost
        best_cv = cv
        best_ee = ee
        best_ar = ar
        best_c = _params.copy()
        best_img = init_img.copy()
        last_best_epoch = 0

        # Record initial state
        recorder.append(
            {
                "J": cost,
                "quality": quality,
                "cv": cv,
                "ee": ee,
                "ar": ar,
                "side": target_side,
                "_c": _params,
                "_img": init_img,
                "_diff": 0,
                "lr": lr,
                "delta": delta,
                "_epoch": 0,
                "exp_t": cam.exposure_time,
                "max_brt": np.max(init_img),
                "mean_b": mean_int,
                "target_mean_b": target_mean_brightness,
                "_grad": np.zeros_like(_params),
                "optimizer": optimizer_type,
                "best_quality": best_quality,
            }
        )

        # Create optimizer
        optimizer = _create_optimizer(
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

        # pygame 实时显示 (2×2 面板), 初始化失败则禁用显示但不中断优化
        display_stack = contextlib.ExitStack()
        display_ctx: _SPGDDisplay | None = None
        if show:
            try:
                display_ctx = _SPGDDisplay(active_modes=_active_modes)
                display_stack.enter_context(display_ctx)
            except Exception as _e:
                logger.warning("pygame显示初始化失败, 本次运行禁用显示: {}", _e)
                display_ctx = None
                display_stack.close()

        # Main optimization loop
        with tqdm.tqdm(
            total=epochs, desc=f"slm_square iter {epochs}", dynamic_ncols=True
        ) as bar:
            for epoch in range(1, epochs + 1):
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
                pos_q = square_quality_score(
                    pos_cv, pos_ee, pos_ar, w_uniformity, w_efficiency, w_aspect
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
                neg_q = square_quality_score(
                    neg_cv, neg_ee, neg_ar, w_uniformity, w_efficiency, w_aspect
                )

                # Auto-exposure adjustment if saturated
                max_brightness = max(np.max(pos_img), np.max(neg_img))
                if max_brightness == 255 and exposure_time_ms == 0:
                    _resample_img = cam.autoset_exposure_time_ms(
                        target_max_brightness, twice_valid=False
                    )
                    optimizer.scale_momentum(np.sum(_resample_img) / np.sum(pos_img))

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
                    eval_img, eval_cv, eval_ee, eval_ar, eval_c = (
                        pos_img,
                        pos_cv,
                        pos_ee,
                        pos_ar,
                        _pos_c.copy(),
                    )
                else:
                    eval_img, eval_cv, eval_ee, eval_ar, eval_c = (
                        neg_img,
                        neg_cv,
                        neg_ee,
                        neg_ar,
                        _neg_c.copy(),
                    )

                quality = square_quality_score(
                    eval_cv, eval_ee, eval_ar, w_uniformity, w_efficiency, w_aspect
                )
                J = (pos_cost + neg_cost) / 2

                # Track best
                if quality > best_quality + 1e-6:
                    best_quality = quality
                    best_cost = float(J)
                    best_cv = eval_cv
                    best_ee = eval_ee
                    best_ar = eval_ar
                    best_c = eval_c.copy()
                    best_img = eval_img.copy()
                    last_best_epoch = epoch

                # Adaptive learning schedule
                if lr == 0:
                    _grad_mag = float(np.linalg.norm(gradient))
                    _gradient_history.append(_grad_mag)
                    _cv_history.append(eval_cv)
                    if len(_gradient_history) > _max_history_len:
                        _gradient_history.pop(0)
                        _cv_history.pop(0)
                    optimizer.lr, delta = learning_schedule(
                        cv=eval_cv,
                        encircled_energy=eval_ee,
                        gradient_history=_gradient_history,
                        cv_history=_cv_history,
                        epoch=epoch,
                    )
                    optimizer.lr *= _param_scale
                    delta *= _param_scale

                log = {
                    "J": J,
                    "quality": quality,
                    "cv": eval_cv,
                    "ee": eval_ee,
                    "ar": eval_ar,
                    "side": target_side,
                    "_diff": diff,
                    "lr": optimizer.lr,
                    "delta": delta,
                    "_epoch": epoch,
                    "_c": eval_c,
                    "_img": eval_img,
                    "exp_t": cam.exposure_time,
                    "max_brt": max_brightness,
                    "mean_b": (pos_mean + neg_mean) / 2,
                    "target_mean_b": target_mean_brightness,
                    "_grad": gradient,
                }
                recorder.append(log)

                if display_ctx is not None:
                    try:
                        display_ctx.update(
                            phase_gray=_params_to_gray(eval_c),
                            coeffs=(eval_c[:-1] if _has_rotation else eval_c),
                            img=eval_img,
                            epoch=epoch,
                            quality=quality,
                            cv=eval_cv,
                            ee=eval_ee,
                        )
                    except Exception:
                        pass

                bar.set_postfix({k: v for k, v in log.items() if k[0] != "_"})
                bar.update(1)

        # Reset SLM to flat phase on exit
        if SLM_RESET_ON_EXIT:
            slm.set_grayscale(0)

        logger.info(
            f"Optimization complete. Best quality={best_quality:.4f} "
            f"(CV={best_cv:.4f}, EE={best_ee:.4f}, AR={best_ar:.4f}) "
            f"@ epoch {last_best_epoch}"
        )

        display_stack.close()

        return recorder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SPGD square beam uniformity optimization using SLM with Zernike coefficients"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=2000, help="Number of iterations"
    )
    parser.add_argument(
        "-n", "--n_max", type=int, default=4, help="Max Zernike radial order"
    )
    parser.add_argument(
        "-c",
        "--center",
        type=str,
        default="max",
        help="Center detection: shape/max/mass/centroid_thresh or 'x,y'",
    )
    parser.add_argument(
        "--target-side", type=int, default=0, help="Target square side (0=auto)"
    )
    parser.add_argument(
        "--side-factor", type=float, default=1.5, help="Side length factor for auto"
    )
    parser.add_argument(
        "-d", "--delta", type=float, default=0.1, help="Perturbation amplitude"
    )
    parser.add_argument("--lr", type=float, default=0, help="Learning rate (0=auto)")
    parser.add_argument(
        "-t", "--exposure_time_ms", type=float, default=80.0, help="Exposure time (ms)"
    )
    parser.add_argument("--cam_id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--slm_number", type=int, default=1, help="SLM device number")
    parser.add_argument(
        "--slm_wavelength", type=int, default=1064, help="SLM wavelength (nm)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="adamod", help="Optimizer type"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--show", action="store_true", help="Show images during optimization"
    )
    parser.add_argument("--cam_size", type=int, default=300, help="Camera window size")
    parser.add_argument(
        "--zernike-mask",
        type=str,
        default=None,
        help="0/1 binary mask for Zernike modes (comma-separated), e.g. '0,0,0,1,0,0,0,0,0,0,0,1' for defocus+spherical only. Noll 1-3 forced to 0.",
    )
    parser.add_argument(
        "--rotation-search",
        type=float,
        default=0.0,
        help="SLM↔相机相对旋转搜索范围(度, 0~360; 0=关闭旋转校正)",
    )

    args = parser.parse_args()

    center_arg: tuple[int, int] | str | None = args.center
    try:
        parts = args.center.split(",")
        if len(parts) == 2:
            center_arg = (int(parts[0]), int(parts[1]))
    except (ValueError, AttributeError):
        pass

    zernike_mask_arr = None
    if args.zernike_mask is not None:
        zernike_mask_arr = np.array(
            [int(x.strip()) for x in args.zernike_mask.split(",")],
            dtype=int,
        )

    recorder = optimize_slm_square(
        center=center_arg,
        epochs=args.epochs,
        n_max=args.n_max,
        target_side=args.target_side,
        side_factor=args.side_factor,
        delta=args.delta,
        lr=args.lr,
        exposure_time_ms=args.exposure_time_ms,
        cam_id=args.cam_id,
        slm_number=args.slm_number,
        slm_wavelength=args.slm_wavelength,
        optimizer_type=args.optimizer,
        random_seed=args.seed,
        show=args.show,
        cam_size=args.cam_size,
        zernike_mask=zernike_mask_arr,
        rotation_search_deg=args.rotation_search,
    )

    best_iter, (_, best_val) = recorder.get_best_iter()
    logger.info(
        f"Optimization complete. Best quality: {best_val:.4f} "
        f"@ epoch {best_iter.get('_epoch', 'N/A')}"
    )
    save_file = gen_date_dir("data") / f"slm_square_{gen_date_str()}.csv"
    recorder.save_dataframe(save_file)
    logger.info(f"Results saved to: {save_file}")
