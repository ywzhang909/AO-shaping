"""Offline target + phase matrix for the frozen SLM shaping campaign (2026-11-20).

冻结矩阵: 6 个形状 x 4 个尺寸 (CCD 原生像元) x 2 条方法 = 48 条相位:

* ``gs``    —— ``gerchberg_saxton`` 的纯傅里叶基线, 不依赖任何拟合参数作为"真值";
* ``model`` —— 冻结物理预测模型 (P3 checkpoint) 上的 128 控制格可微优化, **仅作诊断**
  (``model_phase_diagnostic_only=true``)。P3 checkpoint 双资格门 fail, 且其
  ``grid_size=128`` / ``fft_size=512`` 只覆盖零级周围 ±253.4 CCD px, 500/1000/1500 px
  目标大部分落在窗外 —— 那些尺寸不产出相位, manifest 记
  ``status="skipped_unrepresentable"``。

本脚本只做离线计算: 不 import/打开任何 SLM、CCD、激光, 不采集, 不 ``--execute``。
产物 = 目标强度 / 掩膜 / 预览 PNG + 每条相位 ``.npy`` + 含逐条 SHA256 的 manifest。
重复运行幂等: 指纹一致且文件 SHA256 未变则 ``skipped``, 变了则 ``overwritten``。

网格约定 (唯一一份, 见 ``panel_camera_xy``):

* 目标画在 **CCD 原生网格** 1944x2592 (Daheng MER2-507, 2.2 µm 像元);
* ``gs`` 在 **面板 600x960 网格** (面板 1200x1920 每 2 像元取一点) 上求解, 再把复振幅
  双线性升采样 x2 到面板原生 1200x1920; FFT 的频率跨度为面板物理跨度, 所以
  一个 FFT bin = ``slope/1920`` CCD px (x) / ``slope/1200`` (y), 与求解网格无关;
* ``model`` 的相位在冻结模型的 128x128 控制格上, 1 格 = 1920/128 = 15 面板像元。

运行::

    python scripts/slm/target_phase_matrix_20261120.py \
        --out data/debug/slm_transfer/dyn_phase_matrix_20261120
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy.ndimage import map_coordinates, zoom

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward,
    load_frozen_parameters,
    recentre_window,
)
from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton
from ao_shaping.algorithm.signal_processing.multishape_objective import (
    assert_target_within_fft_window,
    fft_max_side_px,
)
from ao_shaping.utils.image.target.patterns import create_target_shape
from ao_shaping.utils.slm.bench_constants import (
    LASER_WAVELENGTH_M,
    SLM_PANEL_H,
    SLM_PANEL_W,
    SLM_PITCH_M,
)

#: 冻结矩阵 (用户已确认, 不得改动)。
SHAPES: tuple[str, ...] = ("square", "circle", "triangle", "star", "rectangle", "annular")
SIZES_PX: tuple[int, ...] = (200, 500, 1000, 1500)
METHODS: tuple[str, ...] = ("gs", "model")

#: CCD 原生网格 (Daheng MER2-507: 2592x1944, 2.2 µm 像元)。
CCD_H, CCD_W = 1944, 2592
CCD_PIXEL_UM = 2.2

#: 面板物理跨度 (面板像元); GS 的 FFT 网格覆盖它, 1 bin = 1/1920 cycle/panel-px。
PANEL_SPAN = (float(SLM_PANEL_H), float(SLM_PANEL_W))

#: GS 求解网格: 面板每 2 像元取一点, 覆盖整块面板。理由见 ``gs_grid_report``。
GS_GRID: tuple[int, int] = (SLM_PANEL_H // 2, SLM_PANEL_W // 2)

#: 矩形长:短 = 2:1, 长边 = size; 环内径 = 0.6 x 外径。
RECT_ASPECT = 2.0
ANNULAR_INNER_RATIO = 0.6
#: 星形内/外半径比 (与 ``patterns._pat_poly`` 的 0.42 一致, 只用于解析面积自查)。
STAR_INNER_RATIO = 0.42

#: 0 级核心半径 (CCD px): 用"落在核心内的能量占比"估计不加载波时零级有多亮。
ZERO_CORE_RADIUS_PX = 10.0

DEFAULT_OUT = Path("data/debug/slm_transfer/dyn_phase_matrix_20261120")
DEFAULT_CHECKPOINT = Path("data/debug/slm_transfer/p3_fit_20261120/frozen_parameters.json")
#: checkpoint 缺失时目标退回 CCD 阵列中心。
CENTER_FALLBACK: tuple[float, float] = ((CCD_W - 1) / 2.0, (CCD_H - 1) / 2.0)

#: 与 ``multishape_objective.FFT_WINDOW_MARGIN_PX`` 同值: 目标不得贴到窗边。
FFT_WINDOW_MARGIN_PX = 8.0
#: 判定"欠采样"的门限: 目标最短跨度的独立分辨元数量。经验值 —— 少于 8 个分辨元
#: 连目标轮廓都拼不出来; 取 8 而不是 16, 因为 200 px 矩形 (短边 100 px) 在 GS 网格
#: (分辨元 6.33 px) 上正好 15.8 个, 属于"粗但可分辨", 不该被当成不可表示丢掉。
MIN_RESOLUTION_ELEMENTS = 8.0
#: GS 的标称传播距离 (propagation="fft" 下不参与计算, 只为库签名完整)。
GS_NOMINAL_DISTANCE_M = 0.1

SIZE_SEMANTICS: dict[str, str] = {
    "square": "边长 = size_px (CCD 原生像元)",
    "circle": "外接圆直径 = size_px",
    "triangle": "外接圆直径 = size_px (正三角, 尖朝上)",
    "star": "外接圆直径 = size_px (五角星, 内/外半径 0.42)",
    "rectangle": "长边 = size_px, 短边 = size_px/2 (长宽比 2:1)",
    "annular": "外径 = size_px, 内径 = 0.6 x 外径",
}


# ---------------------------------------------------------------------------
# 基本工具
# ---------------------------------------------------------------------------
def utc_stamp() -> str:
    """ISO-8601 UTC 时间戳 (秒精度)。"""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    """字节串的 SHA256。"""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """文件的 SHA256; 不存在时返回 ``""``。"""
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npy_bytes(array: np.ndarray) -> bytes:
    """``np.save`` 到内存 (确定性头部), 供"先算哈希再决定是否落盘"。"""
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array))
    return buffer.getvalue()


def git_commit(root: Path) -> str | None:
    """当前 HEAD; 不是一个 git 仓库时返回 ``None``。"""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


# ---------------------------------------------------------------------------
# 目标图 (CCD 原生网格, 复用 patterns.create_target_shape)
# ---------------------------------------------------------------------------
def shape_kwargs(shape: str, size_px: float, grid: tuple[int, int] = (CCD_H, CCD_W)) -> dict[str, float]:
    """尺寸语义 -> ``create_target_shape`` 参数 (唯一实现)。

    ``circle``/``triangle``/``star`` 走 ``radius_ratio`` (半径 = 外接圆半径);
    ``square`` 的 ``side`` 就是边长; ``rectangle`` 的 ``side`` 是**短边**
    (见 ``patterns._pat_rectangle``), 所以长边 = size 时传 ``side = size/2``;
    ``annular`` 传绝对像素半径。
    """
    size = float(size_px)
    if not np.isfinite(size) or size <= 0:
        raise ValueError(f"尺寸必须是有限正数, 收到 {size_px!r}")
    if shape == "square":
        return {"side": size}
    if shape == "rectangle":
        return {"side": size / RECT_ASPECT, "aspect_ratio": RECT_ASPECT}
    if shape == "annular":
        outer = size / 2.0
        return {"outer_radius": outer, "inner_radius": ANNULAR_INNER_RATIO * outer}
    if shape in {"circle", "triangle", "star"}:
        return {"radius_ratio": size / float(min(grid))}
    raise ValueError(f"未支持的目标形状: {shape!r}")


def analytic_area(shape: str, size_px: float) -> float:
    """尺寸语义下的解析面积 (px²), 用于和光栅化结果对账。"""
    size = float(size_px)
    if shape == "square":
        return size**2
    if shape == "rectangle":
        return size * size / RECT_ASPECT
    if shape == "circle":
        return float(np.pi) * (size / 2.0) ** 2
    if shape == "annular":
        outer = size / 2.0
        inner = ANNULAR_INNER_RATIO * outer
        return float(np.pi) * (outer**2 - inner**2)
    if shape == "triangle":
        return float(3.0 * np.sqrt(3.0) / 4.0) * (size / 2.0) ** 2
    if shape == "star":
        # 10 顶点交替 R / 0.42R: 面积 = 5 R r sin(36°)
        return 5.0 * (size / 2.0) * (STAR_INNER_RATIO * size / 2.0) * float(np.sin(np.deg2rad(36.0)))
    raise ValueError(f"未支持的目标形状: {shape!r}")


def build_target(
    shape: str, size_px: float, centre: tuple[float, float], *, grid: tuple[int, int] = (CCD_H, CCD_W)
) -> tuple[np.ndarray, np.ndarray]:
    """在 CCD 原生网格上生成 (float32 目标强度, bool 掩膜)。

    强度在掩膜内恒为 1.0 (硬边, 不做高斯/窗口平滑), 这样填充像素数可以直接和
    ``analytic_area`` 对账, 目标"形状 x 尺寸"的口径不会隐式漂移。
    """
    limit = float(min(grid)) - 2.0
    if float(size_px) > limit:
        raise ValueError(f"目标尺寸 {size_px:g} px 超过 {grid[0]}x{grid[1]} 网格的可用跨度 (上限 {limit:g} px)")
    raster = create_target_shape(shape, grid, center=centre, **shape_kwargs(shape, size_px, grid))
    mask = np.asarray(raster) > 0
    if not mask.any():
        raise ValueError(f"{shape}@{size_px:g} 生成的目标为空")
    return mask.astype(np.float32), mask


def resolve_centre(params: Any | None) -> tuple[float, float]:
    """目标中心 = 冻结模型的实测零点 (不加倾斜时目标就在零级); 无 checkpoint 时用阵列中心。"""
    if params is None:
        return CENTER_FALLBACK
    return (float(params.origin_xy[0]), float(params.origin_xy[1]))


def render_preview(target: np.ndarray, path: Path, title: str) -> None:
    """写一张目标预览 PNG (Agg, 完全离线)。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = np.flatnonzero(target.any(axis=1))
    cols = np.flatnonzero(target.any(axis=0))
    pad = 24
    r0, r1 = max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad + 1, target.shape[0])
    c0, c1 = max(int(cols[0]) - pad, 0), min(int(cols[-1]) + pad + 1, target.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.2, 5.2), dpi=110)
    axis.imshow(target[r0:r1, c0:c1], cmap="inferno", origin="upper", interpolation="nearest")
    axis.set_title(title, fontsize=9)
    axis.set_xlabel(f"CCD x [px]  (crop origin {c0})", fontsize=8)
    axis.set_ylabel(f"CCD y [px]  (crop origin {r0})", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# 网格/代表性: manifest 的核心数字
# ---------------------------------------------------------------------------
def panel_camera_xy(params: Any, grid: tuple[int, int]) -> np.ndarray:
    """FFT 索引 -> CCD 坐标 (``(H, W, 2)``, x 在 [...,0]) 的**唯一**映射。

    ``gerchberg_saxton(propagation="fft")`` 用 ``fftshift(fft2(field))``; 对一个覆盖
    整块面板 (``PANEL_SPAN`` 面板像元) 的 ``(H, W)`` 网格, bin ``(iy, ix)`` 的物理
    循环频率是 ``(ix - W//2)/(W*cell_x)`` cycles/panel-px, 其中 ``cell_x = 1920/W``,
    即 ``(ix - W//2)/1920``。再乘 ``ramp_matrix`` (cycle/panel-px -> CCD px) 加
    ``origin_xy``。``grid == PANEL_SPAN`` 时退化为面板逐像元网格。
    """
    height, width = int(grid[0]), int(grid[1])
    ramp = np.asarray(params.ramp_matrix, dtype=np.float64)
    origin = np.asarray(params.origin_xy, dtype=np.float64)
    cell_x = PANEL_SPAN[1] / width
    cell_y = PANEL_SPAN[0] / height
    fx = (np.arange(width) - width // 2) * (1.0 / (width * cell_x))
    fy = (np.arange(height) - height // 2) * (1.0 / (height * cell_y))
    xy = np.empty((height, width, 2), dtype=np.float64)
    xy[..., 0] = fx[None, :] * ramp[0, 0] + fy[:, None] * ramp[1, 0] + origin[0]
    xy[..., 1] = fx[None, :] * ramp[0, 1] + fy[:, None] * ramp[1, 1] + origin[1]
    return xy


def _ramp_slope(params: Any) -> float:
    """``ramp_matrix`` 两行范数的较大者: CCD px per (cycle/panel-px)。"""
    ramp = np.asarray(params.ramp_matrix, dtype=np.float64)
    return float(np.linalg.norm(ramp, axis=1).max())


def model_grid_report(params: Any) -> dict[str, Any]:
    """冻结模型的网格覆盖: 具体数字, 不猜。

    * 控制格 ``grid_size`` 覆盖整块面板 -> 1 格 = ``1920/grid_size`` 面板像元;
    * FFT 窗覆盖 ``[0, 0.5]`` cycle/panel-px, 即 ``±slope*0.5/pitch`` CCD px (硬边界);
    * ``fft_size`` 只改采样密度 (1 bin = ``slope/(fft_size*pitch)`` px), 不改覆盖;
    * ``fft_size/grid_size`` 倍零填充 -> 独立分辨元 = ``2*reach/grid_size`` px。
    """
    grid_size = int(params.grid_size)
    fft_size = int(params.fft_size)
    pitch = PANEL_SPAN[1] / grid_size
    slope = _ramp_slope(params)
    reach = slope * 0.5 / pitch
    return {
        "grid_size": grid_size,
        "fft_size": fft_size,
        "zero_pad_factor": fft_size // grid_size,
        "control_cell_panel_px": pitch,
        # 模型的控制格是 grid_size x grid_size 的**方形**域 (见 defocus_forward_core.panel_grid):
        # 两轴都是 pitch=1920/grid_size 面板像元, 所以 x/y 的频率标度相同。
        "control_domain_panel_px": [float(grid_size) * pitch, float(grid_size) * pitch],
        "slope_ccd_px_per_cycle_per_panel_px": slope,
        "window_half_span_px": reach,
        "usable_half_span_px": reach - FFT_WINDOW_MARGIN_PX,
        "window_span_px": 2.0 * reach,
        # 零填充后的 FFT 采样间隔 (只影响采样密度, 不影响覆盖)。
        "fft_bin_px": slope / (fft_size * pitch),
        # 一个**独立**分辨元: 2*reach / grid_size = 4*bin (4 倍零填充)。
        "resolution_px_per_dof": 2.0 * reach / grid_size,
        "max_target_side_px": float(fft_max_side_px(grid_size, slope)),
    }


def gs_grid_report(params: Any | None) -> dict[str, Any]:
    """GS 求解网格的覆盖与采样; 与模型网格并列, 便于解释为什么 GS 能做 1500 px。"""
    nh, nw = GS_GRID
    cell_y = PANEL_SPAN[0] / nh
    cell_x = PANEL_SPAN[1] / nw
    report: dict[str, Any] = {
        "solver_grid": [nh, nw],
        "solver_span_panel_px": list(PANEL_SPAN),
        "cell_panel_px": [cell_y, cell_x],
        "panel_span_px": list(PANEL_SPAN),
        "phase_saved_grid": [int(SLM_PANEL_H), int(SLM_PANEL_W)],
        "phase_upsample": "complex-field bilinear x2 (unit-magnitude field, then angle)",
        "why_this_grid": (
            "网格覆盖整块面板 (1200x1920 面板像元), 每 2 面板像元取一点: FFT 的"
            "频率跨度只由面板物理跨度决定, 与网格点数无关, 所以 1 bin 的 CCD 位移"
            "与面板原生网格相同; 网格减半只把目标边界的采样从 1 面板像元放宽到 2, "
            "而 1500 CCD px 目标仍落在 ±1900 px 的 FFT 可达窗内。"
        ),
        "fft_bin_px_x": None,
        "fft_bin_px_y": None,
        "reach_half_span_px": None,
        "resolution_px_per_dof": None,
        "nyquist_period_panel_px": 2.0 * cell_x,
    }
    if params is None:
        return report
    slope = _ramp_slope(params)
    # 频率 bin = 1/(grid*cell) cycles/panel-px -> CCD px = slope/(grid*cell);
    # bin 的 CCD 位移随 grid 增大而变小, 但 grid*cell = 面板跨度 恒为 1920。
    bin_x = slope / (nw * cell_x)
    bin_y = slope / (nh * cell_y)
    reach_y = slope * 0.5 / cell_y
    reach_x = slope * 0.5 / cell_x
    report.update(
        {
            "fft_bin_px_x": bin_x,
            "fft_bin_px_y": bin_y,
            # 覆盖半跨 = half_span_in_bins * bin_px = (n/2) * slope/(n*cell)。
            "reach_half_span_px": [reach_y, reach_x],
            # 独立分辨元 = 2*reach/n (无零填充时也等于 2*bin); 取两轴较粗者。
            "resolution_px_per_dof": max(2.0 * reach_y / nh, 2.0 * reach_x / nw),
        }
    )
    return report


def _representability_core(
    shape: str,
    size_px: float,
    mask: np.ndarray | None,
    centre: tuple[float, float],
    *,
    usable_half_span_px: float,
    window_span_px: float,
    max_side_px: float,
    resolution_px_per_dof: float,
    grid_label: str,
) -> dict[str, Any]:
    """共用代表性核算: FFT 窗 (硬, 切面积) + 有效分辨元 (软, 欠采样)。

    ``clipped_area_fraction`` 直接在已光栅化的掩膜上数: 中心窗
    ``±usable_half_span_px`` 之外的像素占比 —— 这是"该网格看不到多少目标"的确切数字。
    """
    usable = float(usable_half_span_px)
    dof = float(resolution_px_per_dof)
    extent_long = float(size_px)
    extent_short = extent_long / RECT_ASPECT if shape == "rectangle" else extent_long
    half_long = extent_long / 2.0
    window_ok = half_long <= usable
    elements_long = extent_long / dof
    elements_short = extent_short / dof
    resolution_ok = min(elements_long, elements_short) >= MIN_RESOLUTION_ELEMENTS

    clipped: float | None = None
    if mask is not None:
        rows = np.arange(mask.shape[0], dtype=np.float64)[:, None]
        cols = np.arange(mask.shape[1], dtype=np.float64)[None, :]
        inside = (np.abs(cols - centre[0]) <= usable) & (np.abs(rows - centre[1]) <= usable)
        filled = int(mask.sum())
        clipped = float(1.0 - int((mask & inside).sum()) / filled) if filled else None

    if not window_ok:
        clipped_text = (
            " (dry-run 未光栅化掩膜, 切面积占比在该模式下不给)"
            if clipped is None
            else f"; 目标有 {100.0 * clipped:.1f}% 的面积落在窗外"
        )
        reason = (
            f"{grid_label}: 目标半跨 {half_long:.1f} px 超过 FFT 窗可用半径 {usable:.1f} px "
            f"(窗总宽 {window_span_px:.1f} px, 居中目标上限边长 {max_side_px:.1f} px)"
            f"{clipped_text}"
        )
    elif not resolution_ok:
        reason = (
            f"{grid_label}: 窗前通过, 但最短跨度 {extent_short:.0f} px 只对应 "
            f"{elements_short:.1f} 个独立分辨元 (每个 {dof:.2f} px), "
            f"低于门限 {MIN_RESOLUTION_ELEMENTS:.0f}"
        )
    else:
        reason = (
            f"{grid_label}: 半跨 {half_long:.1f} px 在窗内 (余量 {usable - half_long:.1f} px); "
            f"长/短跨度分别 {elements_long:.0f}/{elements_short:.0f} 个独立分辨元 "
            f"(每个 {dof:.2f} px)"
            + ("" if clipped is None else f"; 窗外面积 {100.0 * clipped:.1f}%")
        )
    return {
        "grid": grid_label,
        "representable": bool(window_ok and resolution_ok),
        "window_ok": bool(window_ok),
        "resolution_ok": bool(resolution_ok),
        "half_span_px": half_long,
        "window_headroom_px": float(usable - half_long),
        "elements_across_long": elements_long,
        "elements_across_short": elements_short,
        "resolution_px_per_dof": dof,
        "usable_half_span_px": usable,
        "window_span_px": float(window_span_px),
        "max_target_side_px": float(max_side_px),
        "clipped_area_fraction": clipped,
        "reason": reason,
    }


def representability(
    shape: str,
    size_px: float,
    mask: np.ndarray | None,
    params: Any | None,
    centre: tuple[float, float],
    *,
    method: str = "model",
) -> dict[str, Any]:
    """``method`` 自己的相位网格下这个目标是否可表示 (逐方法给出, 不混用口径)。

    * ``model`` —— 冻结模型的 ``grid_size`` 控制格: 只能放 ≤490.7 px 的目标;
    * ``gs``    —— GS 求解网格 (600x960, 整块面板): 可达 ±1900 px, 1500 px 也放得下。
    """
    if params is None:
        return {
            "grid": "model" if method == "model" else "gs",
            "representable": False,
            "window_ok": False,
            "resolution_ok": False,
            "reason": "checkpoint 不可用: 冻结模型的 ramp_matrix / 照明参数缺失, 无法判定网格代表性",
            "clipped_area_fraction": None,
        }
    if method == "model":
        report = model_grid_report(params)
        return _representability_core(
            shape, size_px, mask, centre,
            usable_half_span_px=float(report["usable_half_span_px"]),
            window_span_px=float(report["window_span_px"]),
            max_side_px=float(report["max_target_side_px"]),
            resolution_px_per_dof=float(report["resolution_px_per_dof"]),
            grid_label=f"frozen_model_{int(report['grid_size'])}x{int(report['grid_size'])}",
        )
    gs = gs_grid_report(params)
    reach_x, reach_y = (float(v) for v in gs["reach_half_span_px"])
    return _representability_core(
        shape, size_px, mask, centre,
        usable_half_span_px=min(reach_x, reach_y) - FFT_WINDOW_MARGIN_PX,
        window_span_px=2.0 * min(reach_x, reach_y),
        max_side_px=2.0 * (min(reach_x, reach_y) - FFT_WINDOW_MARGIN_PX),
        resolution_px_per_dof=float(gs["resolution_px_per_dof"]),
        grid_label=f"gs_solver_{GS_GRID[0]}x{GS_GRID[1]}",
    )


# ---------------------------------------------------------------------------
# GS: 面板采样网格上的纯傅里叶基线
# ---------------------------------------------------------------------------
def panel_source_amplitude(params: Any | None, grid: tuple[int, int] = GS_GRID) -> np.ndarray:
    """GS 的源面振幅: 冻结模型拟合出的高斯照明在求解网格上取样 (无 checkpoint 时全 1)。"""
    height, width = int(grid[0]), int(grid[1])
    if params is None:
        return np.ones((height, width), dtype=np.float64)
    xx = np.arange(width, dtype=np.float64) * (PANEL_SPAN[1] / width)
    yy = np.arange(height, dtype=np.float64) * (PANEL_SPAN[0] / height)
    cx, cy = (float(v) for v in params.amplitude_center_xy)
    wx, wy = (float(v) for v in params.amplitude_width_xy)
    gx = np.exp(-(((xx - cx) / wx) ** 2))
    gy = np.exp(-(((yy - cy) / wy) ** 2))
    return gy[:, None] * gx[None, :]


def _sample_ccd(target: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """按 CCD 坐标 (x, y) 双线性采样 ``target``; 窗外为 0。"""
    return map_coordinates(target, [xy[..., 1], xy[..., 0]], order=1, mode="constant", cval=0.0)


def _far_field_diagnostics(
    intensity: np.ndarray,
    target_support: np.ndarray,
    xy: np.ndarray,
    centre: tuple[float, float],
    *,
    scope: str,
) -> dict[str, Any]:
    """目标内能量占比 / 0 级核心占比 / 目标内峰均比 (全部相对本网格的可达视场)。"""
    total = float(intensity.sum())
    if not np.isfinite(total) or total <= 0 or not np.isfinite(intensity).all():
        return {
            "energy_in_target_fraction": None,
            "zero_order_core_fraction_r10px": None,
            "peak_over_mean_in_target": None,
            "diagnostics_scope": scope,
            "null_reason": "远场强度非有限或总能量 <= 0",
        }
    inside = float(intensity[target_support].sum()) if target_support.any() else 0.0
    radius = np.hypot(xy[..., 0] - centre[0], xy[..., 1] - centre[1])
    core = float(intensity[radius <= ZERO_CORE_RADIUS_PX].sum())
    peak_ratio = None
    if target_support.any() and inside > 0:
        peak_ratio = float(intensity[target_support].max() / (inside / float(target_support.sum())))
    return {
        "energy_in_target_fraction": inside / total,
        "zero_order_core_fraction_r10px": core / total,
        "peak_over_mean_in_target": peak_ratio,
        "diagnostics_scope": scope,
        "null_reason": None,
        "energy_base": "sum of intensity over this grid's reachable field",
    }


def _upsample_phase(phase: np.ndarray, out_shape: tuple[int, int], factor: tuple[float, float]) -> np.ndarray:
    """复振幅双线性升采样后取角度 (避免对包裹相位直接插值产生假条纹)。"""
    field = np.exp(1j * np.asarray(phase, dtype=np.float64))
    real = zoom(field.real, factor, order=1)
    imag = zoom(field.imag, factor, order=1)
    if real.shape != out_shape:  # pragma: no cover - 整数倍时不会发生
        raise RuntimeError(f"升采样形状不符: {real.shape} != {out_shape}")
    return np.angle(real + 1j * imag)


def gs_phase(
    target: np.ndarray,
    params: Any,
    centre: tuple[float, float],
    *,
    iterations: int = 200,
) -> tuple[np.ndarray, dict[str, Any]]:
    """在 GS 求解网格上跑 ``gerchberg_saxton``, 返回 (面板原生弧度相位, 诊断)。

    目标振幅由**已落盘的目标图**按同一 ``panel_camera_xy`` 映射采样, 再按能量匹配
    缩放 (``sum(amp²) == sum(source²)``); 否则小目标会因振幅口径不同被过度聚焦。
    """
    xy = panel_camera_xy(params, GS_GRID)
    support = _sample_ccd(target, xy) > 0.5
    filled = int(support.sum())
    if filled == 0:
        raise RuntimeError("目标在 GS 的 FFT 网格上没有非零采样 (检查 centre / ramp_matrix)")
    source = panel_source_amplitude(params, GS_GRID)
    amplitude = np.where(support, 1.0, 0.0)
    scale = float(np.sqrt((source**2).sum() / (amplitude**2).sum()))
    result = gerchberg_saxton(
        source,
        amplitude * scale,
        iterations=int(iterations),
        cell_spacing=SLM_PITCH_M,
        distance=GS_NOMINAL_DISTANCE_M,
        wavelength=LASER_WAVELENGTH_M,
        propagation="fft",
    )
    solver_phase = np.asarray(result.phase, dtype=np.float64)
    panel_phase = _upsample_phase(
        solver_phase, (int(SLM_PANEL_H), int(SLM_PANEL_W)),
        (float(SLM_PANEL_H) / GS_GRID[0], float(SLM_PANEL_W) / GS_GRID[1]),
    )
    intensity = np.asarray(result.amplitude, dtype=np.float64) ** 2
    diagnostics = _far_field_diagnostics(intensity, support, xy, centre, scope="gs_fft_field")
    mse = float(result.error_history[-1]) if result.error_history else None
    diagnostics.update(
        {
            "gs_final_mse": mse,
            "gs_final_mse_normalized": None if mse is None else mse / scale**2,
            "gs_iterations": int(result.iterations),
            "gs_target_bins": filled,
            "gs_target_amplitude_scale": scale,
            "gs_source_amplitude": "frozen-model Gaussian illumination sampled on the solver grid",
            "static_aberration_compensation": "none (pure Fourier GS; no defocus/Zernike term applied)",
            "phase_range_rad": [float(panel_phase.min()), float(panel_phase.max())],
            "phase_wrapped": True,
        }
    )
    return panel_phase, diagnostics


# ---------------------------------------------------------------------------
# model: 冻结物理预测模型上的可微优化 (诊断专用)
# ---------------------------------------------------------------------------
def _model_seed(forward: Any, target: np.ndarray, centre: tuple[float, float]) -> np.ndarray:
    """模型优化初值: 在**冻结模型自己的 FFT 网格**上对本条目标做 80 步 GS。

    与 ``multishape_objective.gs_seed`` 同构 (同一份面板振幅嵌入 ``[half:half+g]``、
    同一套静态离焦 + Zernike 残差扣除), 但目标振幅来自**本条已落盘的目标图**,
    而不是 ``target_at`` 重画的形状。原因: ``target_at`` 把 annular 交给
    ``patterns._pat_annular`` 的默认值 (内径 0.2R / 外径 0.25*size), 与冻结矩阵的
    "内径 = 0.6 x 外径、外径 = size" 不是同一个环; 若直接复用, 种子和损失函数
    优化的会是两个不同的目标。
    """
    p = forward.params
    n, g = int(p.fft_size), int(p.grid_size)
    half = (n - g) // 2
    yy, xx = np.mgrid[:n, :n]
    frequency = np.stack((xx - n // 2, yy - n // 2), axis=-1) / (n * forward.pitch)
    camera_xy = frequency @ np.asarray(p.ramp_matrix).T + np.asarray(p.origin_xy)
    target_amplitude = np.sqrt(np.clip(_sample_ccd(target, camera_xy), 0.0, None))
    if not (target_amplitude > 0).any():
        raise RuntimeError("目标在模型 FFT 网格上没有非零采样 (检查 centre / ramp_matrix)")
    source = np.zeros((n, n), dtype=np.float64)
    source[half:half + g, half:half + g] = forward.amplitude.detach().cpu().numpy()
    result = gerchberg_saxton(
        source,
        target_amplitude,
        iterations=80,
        propagation="fft",
    )
    static = p.defocus_diopter * p.diopter_scale * forward.z4.detach().cpu().numpy()
    if forward.residual_basis is not None:
        static = static + np.einsum(
            "i,ijk->jk",
            forward.residual_values.detach().cpu().numpy(),
            forward.residual_basis.detach().cpu().numpy(),
        )
    return result.phase[half:half + g, half:half + g] - static


def model_phase(
    target: np.ndarray,
    params: Any,
    centre: tuple[float, float],
    shape: str,
    size_px: float,
    *,
    iterations: int = 150,
    lr: float = 0.05,
    seed: int = 0,
    device: str | None = None,
    sample_step_px: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """在冻结物理预测模型上优化 128x128 相位 (诊断), 目标就是已落盘的目标图。

    只用 ``FrozenDefocusForward.predict`` (含离焦/Zernike 残差与高斯照明), 不另造
    传播模型; 损失 = 目标窗外能量 (最小化) + 目标内掩膜 CV。初值见 ``_model_seed``
    (模型 FFT 网格上对本条目标做 80 步 GS, 已减去静态离焦 + Zernike 项)。
    """
    import torch

    report = model_grid_report(params)
    half_span = max(40.0, float(report["usable_half_span_px"]))
    half_span = min(half_span, float(report["usable_half_span_px"]))
    camera_xy = recentre_window((float(centre[0]), float(centre[1])),
                               half_span_px=half_span, step_px=sample_step_px)
    forward = FrozenDefocusForward(params, camera_xy, device=device)
    window_target = _sample_ccd(target, camera_xy)
    target_mask = window_target > 0.5
    if not target_mask.any():
        raise RuntimeError("目标在模型采样窗内没有像素 (检查 centre / 窗口跨度)")
    weight = torch.as_tensor(target_mask, dtype=torch.float32, device=forward._device)

    side = float(size_px) / RECT_ASPECT if shape == "rectangle" else float(size_px)
    assert_target_within_fft_window(forward, centre, side, shape)
    initial = _model_seed(forward, target, centre)
    torch.manual_seed(int(seed))
    phase = torch.tensor(initial, dtype=torch.float32, device=forward._device, requires_grad=True)
    optimiser = torch.optim.Adam([phase], lr=float(lr))

    background = float(params.background)
    n_target = float(target_mask.sum())
    best_phase = phase.detach().clone()
    best_loss = float("inf")
    history: list[float] = []
    for _ in range(int(iterations)):
        optimiser.zero_grad(set_to_none=True)
        image = torch.clamp(forward.predict(phase) - background, min=0.0)
        total = image.sum() + 1e-12
        inside = (image * weight).sum()
        loss_energy = 1.0 - inside / total
        masked = image * weight
        mean = masked.sum() / n_target
        cv = torch.sqrt(((masked - mean * weight) ** 2).sum() / n_target) / (mean + 1e-12)
        loss = loss_energy + 0.5 * cv
        loss.backward()
        optimiser.step()
        value = float(loss.detach().cpu())
        history.append(value)
        if value < best_loss:
            best_loss = value
            best_phase = phase.detach().clone()

    with torch.no_grad():
        image = torch.clamp(forward.predict(best_phase) - background, min=0.0).cpu().numpy()
    result = np.asarray(image, dtype=np.float64)
    # ``predict`` 返回 (B, H, W); 单条相位时 B=1, 去掉批次维以便和 camera_xy 对账。
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.shape != tuple(camera_xy.shape[:2]):
        raise RuntimeError(
            f"模型输出形状 {result.shape} 与采样窗 {tuple(camera_xy.shape[:2])} 不符"
        )
    diagnostics = _far_field_diagnostics(
        result, target_mask, camera_xy, centre, scope="frozen_model_camera_window"
    )
    diagnostics.update(
        {
            "model_iterations": int(iterations),
            "model_lr": float(lr),
            "model_best_loss": best_loss,
            "model_final_loss": history[-1] if history else None,
            "model_sample_half_span_px": half_span,
            "model_sample_step_px": float(sample_step_px),
            "model_window_shape": list(camera_xy.shape[:2]),
            "model_target_pixels_in_window": int(target_mask.sum()),
            "model_init": "_model_seed: 80-step GS on the model FFT grid against this entry's target, static defocus+Zernike subtracted",
            "seed": int(seed),
            "phase_range_rad": [float(np.min(best_phase.detach().cpu().numpy())),
                                float(np.max(best_phase.detach().cpu().numpy()))],
            "phase_wrapped": False,
        }
    )
    return best_phase.detach().cpu().numpy().astype(np.float64), diagnostics


# ---------------------------------------------------------------------------
# 批处理 / 幂等 / manifest
# ---------------------------------------------------------------------------
def _fingerprint(
    script_sha: str,
    checkpoint_sha: str,
    gs_iterations: int,
    model_iterations: int,
    seed: int,
    shapes: tuple[str, ...],
    sizes: tuple[int, ...],
    methods: tuple[str, ...],
) -> str:
    payload = json.dumps(
        {
            "script_sha256": script_sha,
            "checkpoint_sha256": checkpoint_sha,
            "gs_iterations": gs_iterations,
            "model_iterations": model_iterations,
            "seed": seed,
            "shapes": list(shapes),
            "sizes": list(sizes),
            "methods": list(methods),
            "gs_grid": list(GS_GRID),
            "panel_span": list(PANEL_SPAN),
            "ccd_grid": [CCD_H, CCD_W],
        },
        sort_keys=True,
    )
    return sha256_bytes(payload.encode("utf-8"))


def _trusted_previous(manifest: Path, fingerprint: str) -> dict[tuple[str, int, str], dict[str, Any]]:
    """上一次 manifest 的条目 (仅当指纹一致), 用于"不重算就跳过"的快路径。"""
    if not manifest.is_file():
        return {}
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("fingerprint") != fingerprint:
        return {}
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    for entry in payload.get("entries", []):
        try:
            key = (str(entry["shape"]), int(entry["size_px"]), str(entry["method"]))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = entry
    return out


def _artifacts_intact(entry: dict[str, Any], root: Path) -> bool:
    """上一条 manifest 记录的文件是否都还在且 SHA256 未变。"""
    if entry.get("status") not in {"written", "skipped", "overwritten"}:
        return False
    shas = entry.get("sha256") or {}
    for key in ("target", "mask", "phase"):
        rel = entry.get(f"{key}_path")
        digest = shas.get(key)
        if not rel or not digest:
            return False
        if sha256_file(root / rel) != digest:
            return False
    return True


def _write_array(path: Path, array: np.ndarray) -> tuple[str, str]:
    """写 ``.npy`` 并返回 (SHA256, written|skipped|overwritten)。"""
    payload = npy_bytes(array)
    digest = sha256_bytes(payload)
    if path.is_file():
        if sha256_file(path) == digest:
            return digest, "skipped"
        path.write_bytes(payload)
        return digest, "overwritten"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return digest, "written"


def _summarise(entries: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "written": 0, "skipped": 0, "overwritten": 0,
        "failed": 0, "skipped_unrepresentable": 0,
    }
    for entry in entries:
        status = str(entry.get("status", "failed"))
        counts[status] = counts.get(status, 0) + 1
    counts["total"] = len(entries)
    counts["representable_false"] = sorted(
        {f"{e['shape']}@{e['size_px']}" for e in entries if not e.get("representable", False)}
    )
    return counts


def _entry_common(shape: str, size: int, method: str, *, representable: bool) -> dict[str, Any]:
    return {
        "shape": shape,
        "size_px": size,
        "size_semantics": SIZE_SEMANTICS[shape],
        "method": method,
        "target_path": None,
        "mask_path": None,
        "preview_path": None,
        "phase_path": None,
        "sha256": {"target": None, "mask": None, "phase": None},
        "grid": None,
        "grid_kind": None,
        "representable": bool(representable),
        "model_phase_diagnostic_only": method == "model",
        # 本批没有任何 CCD 采集: model 相位只能是 model_only (P3 双门 fail);
        # gs 是纯傅里叶基线, 连冻结模型都不经过, 记 simulation_only。
        "acceptance": "model_only" if method == "model" else "simulation_only",
        "carrier": "none",
        "tilt_applied": False,
        "status": "failed",
        "notes": "",
        "diagnostics": {},
        "created_utc": utc_stamp(),
    }


def build_plan(
    shapes: tuple[str, ...],
    sizes: tuple[int, ...],
    methods: tuple[str, ...],
    params: Any | None,
    centre: tuple[float, float],
) -> dict[str, Any]:
    """``--dry-run`` 的清单: 只说会生成什么, 不碰磁盘。"""
    rows = []
    for shape in shapes:
        for size in sizes:
            for method in methods:
                rep = representability(shape, size, None, params, centre, method=method)
                skip = not rep["representable"]
                rows.append(
                    {
                        "shape": shape,
                        "size_px": size,
                        "method": method,
                        "representable": rep["representable"],
                        "action": "skip (unrepresentable)" if skip else "generate",
                        "clipped_area_fraction": rep.get("clipped_area_fraction"),
                        "reason": rep.get("reason"),
                    }
                )
    return {
        "out": None,
        "targets": [f"{s}_{z}" for s in shapes for z in sizes],
        "entries": rows,
        "centre_ccd_px": [float(centre[0]), float(centre[1])],
        "gs_grid": list(GS_GRID),
    }


def _write_manifest(
    out_dir: Path,
    *,
    entries: list[dict[str, Any]],
    fingerprint: str,
    script_sha: str,
    checkpoint: Path,
    checkpoint_sha: str,
    args: argparse.Namespace,
    shapes: tuple[str, ...],
    sizes: tuple[int, ...],
    methods: tuple[str, ...],
    model_report: dict[str, Any] | None,
    gs_report: dict[str, Any],
    centre: tuple[float, float],
) -> None:
    """原子写 manifest (每个尺寸后调用一次, 中断也留下可续跑的断点)。"""
    payload = {
        "created_utc": utc_stamp(),
        "script": "scripts/slm/target_phase_matrix_20261120.py",
        "script_sha256": script_sha,
        "git_commit": git_commit(Path(__file__).resolve().parents[2]),
        "fingerprint": fingerprint,
        "checkpoint_path": str(checkpoint).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_role": "diagnostic only: P3 failed both eligibility gates; not a hardware candidate",
        "gs_iterations": int(args.gs_iterations),
        "model_iterations": int(args.model_iterations),
        "seed": int(args.seed),
        "device": args.device,
        "requested": {"shapes": list(shapes), "sizes_px": list(sizes), "methods": list(methods)},
        "target_centre_ccd_px": [float(centre[0]), float(centre[1])],
        "ccd_grid": [CCD_H, CCD_W],
        "ccd_pixel_um": CCD_PIXEL_UM,
        "carrier": "none",
        "tilt_applied": False,
        "acceptance_policy": (
            "本批是离线计算, 没有任何 CCD 采集。`slm-transfer-shapes` 是 CCD 锚定的硬件"
            "闭环 (--side 是 CCD 原生像元), 不带 CCD 时其结果只能 acceptance=model_only, "
            "不能当作离线相位生成器的验收。P3 checkpoint 双资格门 fail, 因此 model 相位"
            "一律 model_phase_diagnostic_only=true / acceptance=model_only; gs 相位是纯"
            "傅里叶基线, 记 acceptance=simulation_only。两者都不是资格门, 也不构成\"通过\"。"
        ),
        "zero_order": {
            "core_radius_px": ZERO_CORE_RADIUS_PX,
            "note": (
                "本批不加倾斜/载波, 目标就在零级。每条相位若给出 "
                "diagnostics.zero_order_core_fraction_r10px, 那是该模型自己算出的 "
                "10 px 圆盘能量占比 (数值估计, 未实验验证); 给不出时为 null 并带 null_reason。"
            ),
        },
        "grids": {"gs": gs_report, "model": model_report},
        "entries": entries,
        "summary": _summarise(entries),
    }
    tmp = out_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_dir / "manifest.json")


def _print_summary(summary: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    print("\n=== target_phase_matrix_20261120 ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "representable_false"}, ensure_ascii=False))
    print(f"{'shape':<10}{'size':>6}{'method':<8}{'status':<26}{'repr':<7}clipped")
    for entry in entries:
        rep = entry.get("representability") or {}
        clipped = rep.get("clipped_area_fraction")
        clipped_text = "-" if clipped is None else f"{100.0 * clipped:.1f}%"
        print(
            f"{entry['shape']:<10}{entry['size_px']:>6}{entry['method']:<8}"
            f"{entry['status']:<26}{str(entry['representable']):<7}{clipped_text}"
        )
    if summary["representable_false"]:
        print(f"representable=false: {summary['representable_false']}")


def run(args: argparse.Namespace) -> int:
    """执行整批; 返回进程退出码 (非 0 当且仅当全部条目都失败)。"""
    out_dir = Path(args.out)
    shapes = tuple(args.shapes)
    sizes = tuple(int(v) for v in args.sizes)
    methods = METHODS if args.methods == "both" else (args.methods,)
    checkpoint = Path(args.checkpoint)

    script_sha = sha256_file(Path(__file__).resolve())
    checkpoint_sha = sha256_file(checkpoint)
    if not checkpoint.is_file() and "model" in methods:
        logger.warning(f"checkpoint 不存在: {checkpoint} —— model 路线会逐条失败")

    fingerprint = _fingerprint(
        script_sha, checkpoint_sha, args.gs_iterations, args.model_iterations, args.seed,
        shapes, sizes, methods,
    )

    params = load_frozen_parameters(checkpoint) if checkpoint.is_file() else None
    centre = resolve_centre(params)
    model_report = model_grid_report(params) if params is not None else None
    gs_report = gs_grid_report(params)
    logger.info(f"目标中心 (CCD px): ({centre[0]:.2f}, {centre[1]:.2f})")
    if params is not None and model_report is not None:
        logger.info(
            f"冻结模型: grid={model_report['grid_size']} fft={model_report['fft_size']}, "
            f"FFT 窗 ±{model_report['window_half_span_px']:.2f} px "
            f"(居中上限边长 {model_report['max_target_side_px']:.1f} px), "
            f"独立分辨元 {model_report['resolution_px_per_dof']:.2f} px"
        )
        logger.info(
            f"GS 求解网格 {GS_GRID[0]}x{GS_GRID[1]} (1 格 = 2 面板像元), "
            f"FFT bin {gs_report['fft_bin_px_x']:.2f}/{gs_report['fft_bin_px_y']:.2f} CCD px"
        )

    if args.dry_run:
        plan = build_plan(shapes, sizes, methods, params, centre)
        plan["out"] = str(out_dir)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        logger.info(f"--dry-run: 不创建 {out_dir}, 不写任何文件")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    targets_dir = out_dir / "targets"
    phases_dir = out_dir / "phases"
    previous = _trusted_previous(out_dir / "manifest.json", fingerprint)

    entries: list[dict[str, Any]] = []
    for shape in shapes:
        for size in sizes:
            stem = f"{shape}_{size}"
            target_path = targets_dir / f"{stem}_target.npy"
            mask_path = targets_dir / f"{stem}_mask.npy"
            preview_path = targets_dir / f"{stem}_preview.png"
            try:
                target, mask = build_target(shape, size, centre)
            except Exception as exc:
                logger.error(f"[{stem}] 目标生成失败: {exc}")
                for method in methods:
                    entry = _entry_common(shape, size, method, representable=False)
                    entry["notes"] = f"target build failed: {type(exc).__name__}: {exc}"
                    entries.append(entry)
                _write_manifest(
                    out_dir, entries=entries, fingerprint=fingerprint, script_sha=script_sha,
                    checkpoint=checkpoint, checkpoint_sha=checkpoint_sha, args=args,
                    shapes=shapes, sizes=sizes, methods=methods, model_report=model_report,
                    gs_report=gs_report, centre=centre,
                )
                continue

            target_sha, target_action = _write_array(target_path, target)
            mask_sha, mask_action = _write_array(mask_path, mask.astype(np.uint8))
            if target_action != "skipped" or not preview_path.is_file():
                try:
                    render_preview(target, preview_path, f"{shape} {size}px  (CCD native)")
                except Exception as exc:
                    logger.warning(f"[{stem}] 预览图失败: {exc}")

            base_note = (
                f"filled={int(mask.sum())} px, analytic={analytic_area(shape, size):.0f} px²; "
                f"{SIZE_SEMANTICS[shape]}"
            )

            for method in methods:
                key = (shape, size, method)
                rep = representability(shape, size, mask, params, centre, method=method)
                entry = _entry_common(shape, size, method, representable=rep["representable"])
                entry.update(
                    {
                        "target_path": str(target_path.relative_to(out_dir)).replace("\\", "/"),
                        "mask_path": str(mask_path.relative_to(out_dir)).replace("\\", "/"),
                        "preview_path": str(preview_path.relative_to(out_dir)).replace("\\", "/"),
                        "sha256": {"target": target_sha, "mask": mask_sha, "phase": None},
                        "representability": rep,
                        "notes": base_note,
                    }
                )

                if not rep["representable"]:
                    entry["status"] = "skipped_unrepresentable"
                    entry["notes"] = (
                        f"{base_note}; representable=false on {rep.get('grid')}: {rep.get('reason')}; "
                        "不产出相位 (该网格下目标不可表示, 硬造相位只会得到看似合法的假结果)"
                    )
                    entries.append(entry)
                    continue

                trusted = previous.get(key)
                if trusted is not None and _artifacts_intact(trusted, out_dir):
                    entry.update(
                        {
                            "status": "skipped",
                            "phase_path": trusted.get("phase_path"),
                            "grid": trusted.get("grid"),
                            "grid_kind": trusted.get("grid_kind"),
                            "sha256": {
                                "target": target_sha,
                                "mask": mask_sha,
                                "phase": (trusted.get("sha256") or {}).get("phase"),
                            },
                            "diagnostics": trusted.get("diagnostics", {}),
                            "notes": f"{base_note}; resumed: identical fingerprint, files intact",
                        }
                    )
                    entries.append(entry)
                    continue

                phase_path = phases_dir / f"{stem}_{method}_phase.npy"
                try:
                    if method == "gs":
                        assert params is not None
                        phase, diagnostics = gs_phase(
                            target, params, centre, iterations=args.gs_iterations
                        )
                        entry.update(
                            {
                                "grid": [int(SLM_PANEL_H), int(SLM_PANEL_W)],
                                "grid_kind": "panel_native_1200x1920",
                                "solver_grid": list(GS_GRID),
                                "solver_cell_panel_px": 2.0,
                                "notes": (
                                    f"{base_note}; GS on {GS_GRID[0]}x{GS_GRID[1]} panel-sampled "
                                    "grid (2 panel px/cell), complex-field bilinear x2 to the "
                                    "1200x1920 panel; target amplitude sampled from this entry's "
                                    "target .npy via the frozen ramp_matrix and energy-matched to "
                                    "the fitted illumination; no static aberration compensation; "
                                    "no carrier (target stays at the zero order)"
                                ),
                            }
                        )
                    else:
                        assert params is not None
                        phase, diagnostics = model_phase(
                            target, params, centre, shape, size,
                            iterations=args.model_iterations, seed=args.seed,
                            device=None if args.device == "auto" else args.device,
                        )
                        entry.update(
                            {
                                "grid": [int(params.grid_size), int(params.grid_size)],
                                "grid_kind": "frozen_model_control_grid_1cell=15panel_px",
                                "notes": (
                                    f"{base_note}; DIAGNOSTIC ONLY (P3 checkpoint failed both "
                                    "eligibility gates; no CCD acceptance, acceptance=model_only). "
                                    "Frozen forward model, 128x128 control grid, optimiser "
                                    "initialised by 80-step GS on the model FFT grid against this "
                                    "entry's target (static defocus+Zernike subtracted); no "
                                    "carrier (target stays at the zero order)."
                                ),
                            }
                        )
                    digest, action = _write_array(phase_path, phase)
                    entry.update(
                        {
                            "status": action,
                            "phase_path": str(phase_path.relative_to(out_dir)).replace("\\", "/"),
                            "sha256": {"target": target_sha, "mask": mask_sha, "phase": digest},
                            "diagnostics": diagnostics,
                        }
                    )
                except Exception as exc:
                    logger.error(f"[{stem}/{method}] 失败: {exc}")
                    entry["status"] = "failed"
                    entry["notes"] = f"{base_note}; {type(exc).__name__}: {exc}"
                entries.append(entry)

            _write_manifest(
                out_dir, entries=entries, fingerprint=fingerprint, script_sha=script_sha,
                checkpoint=checkpoint, checkpoint_sha=checkpoint_sha, args=args,
                shapes=shapes, sizes=sizes, methods=methods, model_report=model_report,
                gs_report=gs_report, centre=centre,
            )

    if not entries:
        logger.error("没有任何条目被处理 (检查 --shapes / --sizes)")
        return 1
    summary = _summarise(entries)
    _write_manifest(
        out_dir, entries=entries, fingerprint=fingerprint, script_sha=script_sha,
        checkpoint=checkpoint, checkpoint_sha=checkpoint_sha, args=args,
        shapes=shapes, sizes=sizes, methods=methods, model_report=model_report,
        gs_report=gs_report, centre=centre,
    )
    _print_summary(summary, entries)
    return 0 if summary["failed"] < len(entries) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="产物根目录 (默认 data/debug/slm_transfer/dyn_phase_matrix_20261120)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="冻结参数 json (裸文件, 不是会话目录)")
    parser.add_argument("--shapes", nargs="+", choices=list(SHAPES), default=list(SHAPES))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES_PX))
    parser.add_argument("--methods", choices=["gs", "model", "both"], default="both")
    parser.add_argument("--gs-iterations", type=int, default=200)
    parser.add_argument("--model-iterations", type=int, default=150)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将生成的清单, 不创建目录、不写文件")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """入口。"""
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
