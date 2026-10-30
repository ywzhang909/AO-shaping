"""Render the figure set for the pure-GS 1000 px control arm.

Run root (one experiment session)::

    <RUN>/sides/side_1000/frame_000.npz     GS 相位, 无闪耀光栅  (直接发送 GS)
    <RUN>/blaze/frame_000.npz               GS + 闪耀光栅 P=10
    <RUN>/blaze/frame_001.npz               GS + 闪耀光栅 P=12
    <RUN>/blaze/phase_001.npy               实际下发到面板的相位 (1920x1200, raw rad)

Flat-phase reference frames come from ``probe_optical_path_validity.py``
(``exposure_<ms>ms.npz``, keys ``frames`` (n,H,W) / ``requested_ms``).

四个交付变体 (报告 §结果):

1. 平相位 (flat phase)          —— 光路自身的光斑, 未整形
2. 直接发送 GS (GS, no blaze)   —— 方斑与 0 级重合, 0 级在框内饱和
3. 经过矫正后发送 GS            —— **本次未采集** (设备下线); 传 ``--corrected-frame``
                                   即自动占用该格, 否则画占位格并写明待办
4. 叠加闪耀光栅 (GS + blaze)    —— 方斑移出 0 级

几何口径 (与 ``hw_gs_square_scans.py`` 同一套, 这里只读不算):

* 面板 1920x1200, 光束中心 (960, 600), 半径 450 panel px;
* ``TILT_SHIFT_SCALE = 7400`` 相机 px per (1 / 面板 px) 空间频率, 即一个面板 px 的
  斜坡在焦面产生 ``7400/period`` 相机 px 位移;
* 面板分辨率模型的远场采样 = ``7400/1920`` (列) 与 ``7400/1200`` (行) 相机 px/样本,
  比 512 的 GS 网格 (``7400/900 = 8.22``) 细 2.1x / 1.3x。**模型必须建在面板分辨率
  上**, 因为 512 网格一个样本就跨 8.22 相机 px, 而衍射极限光斑 FWHM 只有 8.46 px。
* 面板 <-> 相机是 **90 度轴交换** (``zero_order_overlay.predicted_shift_px`` 口径):
  面板 y 斜坡 -> 相机 **x**, 面板 x 斜坡 -> 相机 **y**。于是模型**行**下标对相机 x
  (``MODEL_ROW_TO_CAM_X = 7400/1200``), 模型**列**下标对相机 y
  (``MODEL_COL_TO_CAM_Y = 7400/1920``)。相机上 1000x1000 px 的方框因此对应模型
  162 行 x 259 列 —— 方框在模型坐标里是**矩形**, 这不是 bug。

为什么不用 ``generate_hw_gs_square_scans_report.py``: 它要求三个阶段齐备
(含 ``zernike/measurements.csv``), 本会话没跑 zernike 阶段。

用法::

    python scripts/slm/generate_gs_control_report.py
    python scripts/slm/generate_gs_control_report.py --corrected-frame <npz>
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# 指标与几何全部复用既有实现, 本文件不另写一套
from ao_shaping.utils.image.beam_metrics import prepare_frame
from ao_shaping.utils.slm.bench_constants import (
    BEAM_CENTER_PANEL,
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    TILT_SHIFT_SCALE,
)
from scripts.slm.hw_gs_square_scans import measure_box
from scripts._common.plot_style import setup_cjk_font

DEFAULT_RUN = ROOT / "data/debug/hw_gs_square_scans/20261030_gs1000"
DEFAULT_FLAT = ROOT / "data/debug/optical_path_probe/gs_flat_halo/20261030_034710Z"
DEFAULT_OUT = ROOT / "report/slm/gs_control_1000px/figures"

# 面板 <-> 相机 90 度轴交换 (bench_constants 口径, 与
# ``zero_order_overlay.predicted_shift_px`` 完全一致): 面板 y 斜坡 -> 相机 x 位移,
# 面板 x 斜坡 -> 相机 y 位移。模型是**面板网格**上的 FFT, 于是模型行下标 (面板 y)
# 对应相机 x, 模型列下标 (面板 x) 对应相机 y。
MODEL_ROW_TO_CAM_X = TILT_SHIFT_SCALE / SLM_PANEL_H  # 6.167 相机 px / 模型行
MODEL_COL_TO_CAM_Y = TILT_SHIFT_SCALE / SLM_PANEL_W  # 3.854 相机 px / 模型列
FULL_SCALE = 255.0
# 方斑边缘阈值 (counts, 已减帧中位数): 高于 8 bit 量化的台阶, 低于方斑平台 11–15
_THR = 8.0


def _spread(vals: list[float]) -> tuple[float, float]:
    """剖面散布 = 10–90 百分位。

    用 min–max 会把个别误检剖面 (行/列恰好切到画面边缘的辉光) 也算进散布, 报出来
    比真实读数噪声大一两倍; 10–90 百分位描述的是「换一条剖面读数会差多少」。
    """
    if not vals:
        return (float("nan"),) * 2
    lo, hi = np.percentile(np.asarray(vals, float), [10, 90])
    return (float(lo), float(hi))

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})
setup_cjk_font()


# ---------------------------------------------------------------------------
# 读数
# ---------------------------------------------------------------------------
def read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_frame(path: pathlib.Path) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    arr = z["frame"] if "frame" in z else z["frames"][0]
    return np.asarray(arr, dtype=np.float64)


def load_flat(flat_dir: pathlib.Path, ms: float) -> np.ndarray:
    return load_frame(flat_dir / f"exposure_{ms:g}ms.npz")


def crop(frame: np.ndarray, cx: float, cy: float, half: int) -> tuple[np.ndarray, int, int]:
    """以 (cx, cy) 为心的方裁, 越界时平移而不是补零。"""
    h, w = frame.shape
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x0 = min(max(x0, 0), max(w - 2 * half, 0))
    y0 = min(max(y0, 0), max(h - 2 * half, 0))
    return frame[y0:y0 + 2 * half, x0:x0 + 2 * half], x0, y0


# ---------------------------------------------------------------------------
# 面板分辨率远场模型
# ---------------------------------------------------------------------------
def panel_far_field(phase: np.ndarray, radius: float = BEAM_RADIUS_PANEL) -> np.ndarray:
    """面板分辨率的 2f 焦面强度 (归一化到总和 1)。

    振幅取硬边圆孔 (半径 ``radius`` panel px, 中心 ``BEAM_CENTER_PANEL``) —— 光只
    存在于被照亮的那一块, 所以圆孔外的相位 (含铺满整块面板的闪耀斜坡) 不贡献。

    建模的是**理想 2pi 调制**: 不含 LUT 非线性、不含面板静态相差、不含相机响应。
    因此它给出的是「同一相位在理想器件上应有的样子」, 用来判断失败发生在 GS 算法 /
    相位下发, 还是发生在器件与光路。
    """
    h, w = phase.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = BEAM_CENTER_PANEL[1], BEAM_CENTER_PANEL[0]
    amp = ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2).astype(np.float64)
    field = amp * np.exp(1j * np.asarray(phase, dtype=np.float64))
    intensity = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    total = intensity.sum()
    return intensity / total if total > 0 else intensity


def cam_to_model(cam_x: np.ndarray | float, cam_y: np.ndarray | float, dc: tuple[float, float]):
    """相机 px -> 模型样本坐标 ``(列, 行)`` (浮点)。

    90 度轴交换: 相机 x 走模型的**行** (面板 y), 相机 y 走模型的**列** (面板 x)。
    """
    col = BEAM_CENTER_PANEL[0] + (np.asarray(cam_y, float) - dc[1]) / MODEL_COL_TO_CAM_Y
    row = BEAM_CENTER_PANEL[1] + (np.asarray(cam_x, float) - dc[0]) / MODEL_ROW_TO_CAM_X
    return col, row


def model_to_cam(col: np.ndarray | float, row: np.ndarray | float, dc: tuple[float, float]):
    """模型样本坐标 ``(列, 行)`` -> 相机 px (``cam_to_model`` 的逆)。"""
    cam_x = dc[0] + (np.asarray(row, float) - BEAM_CENTER_PANEL[1]) * MODEL_ROW_TO_CAM_X
    cam_y = dc[1] + (np.asarray(col, float) - BEAM_CENTER_PANEL[0]) * MODEL_COL_TO_CAM_Y
    return cam_x, cam_y


def model_box_pib(
    intensity: np.ndarray,
    dc: tuple[float, float],
    center: tuple[float, float],
    side: float,
    window: tuple[float, float, float, float] | None = None,
) -> float:
    """方框内的模型能量占比 —— 在**模型坐标系**里积分, 不重采样到相机网格。

    方框与 ``window`` 都写在**相机 px** 里, 所以先把模型样本下标换算成相机坐标
    (``model_to_cam``) 再判包含关系; 直接拿模型下标去比相机坐标会差一个
    ``MODEL_ROW_TO_CAM_X`` 的缩放, 方框会落到离方斑几百样本之外。

    ``window`` = ``(x0, x1, y0, y1)`` 相机 px, 用来把分母限制在相机看得见的视场
    (相机只有 2592x1944, 模型能算到更远), 否则模型分母里含了画面外的光。
    """
    h, w = intensity.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cam_x, cam_y = model_to_cam(xx, yy, dc)
    box = (
        (cam_x >= center[0] - side / 2)
        & (cam_x <= center[0] + side / 2)
        & (cam_y >= center[1] - side / 2)
        & (cam_y <= center[1] + side / 2)
    )
    if window is not None:
        x0, x1, y0, y1 = window
        view = (cam_x >= x0) & (cam_x <= x1) & (cam_y >= y0) & (cam_y <= y1)
    else:
        view = np.ones_like(box)
    denom = float(intensity[view].sum())
    return float(intensity[box & view].sum() / denom) if denom > 0 else float("nan")


def model_cv(intensity: np.ndarray, dc, center, side, center_frac: float = 0.0) -> float:
    """方框内模型强度的变异系数 (与硬件侧 ``uniformity_cv`` 同定义: std/mean)。

    ``center_frac`` > 0 时只取方框正中的那一块 (边长 ``center_frac * side``) ——
    避开方斑的软边缘与落在框心的残余级次, 只量方斑自身的平顶程度。
    """
    h, w = intensity.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cam_x, cam_y = model_to_cam(xx, yy, dc)
    box = (
        (cam_x >= center[0] - side / 2)
        & (cam_x <= center[0] + side / 2)
        & (cam_y >= center[1] - side / 2)
        & (cam_y <= center[1] + side / 2)
    )
    if center_frac > 0:
        half = side * center_frac / 2
        box &= (
            (cam_x >= center[0] - half)
            & (cam_x <= center[0] + half)
            & (cam_y >= center[1] - half)
            & (cam_y <= center[1] + half)
        )
    sub = intensity[box]
    return float(sub.std() / sub.mean()) if sub.size and sub.mean() > 0 else float("nan")


def resample_model_to_crop(
    intensity: np.ndarray, dc: tuple[float, float], x0: int, y0: int, n: int
) -> np.ndarray:
    """把模型强度重采样到 ``n x n`` 的相机裁切网格 (双线性)。

    模型一个样本跨 ``MODEL_ROW_TO_CAM_X`` (行) x ``MODEL_COL_TO_CAM_Y`` (列) 个相机
    px, 所以这是**放大**; 双线性只是让图好看, 定量一律走 :func:`model_box_pib`
    (在模型坐标系里积分)。
    """
    h, w = intensity.shape
    xs = x0 + np.arange(n)
    ys = y0 + np.arange(n)
    col, row = cam_to_model(xs, ys, dc)
    col = np.clip(col, 0, w - 1)
    row = np.clip(row, 0, h - 1)
    # 先按行 (y) 插值, 再按列 (x) 插值
    rows = np.empty((n, w))
    for j, yv in enumerate(row):
        i0 = int(np.floor(yv))
        i1 = min(i0 + 1, h - 1)
        t = yv - i0
        rows[j] = intensity[i0] * (1 - t) + intensity[i1] * t
    out = np.empty((n, n))
    for j in range(n):
        xv = col
        j0 = np.floor(xv).astype(int)
        j1 = np.minimum(j0 + 1, w - 1)
        t = xv - j0
        out[j] = rows[j, j0] * (1 - t) + rows[j, j1] * t
    return out


def airy_profile(r_px: np.ndarray, zero_radius_px: float) -> np.ndarray:
    """理想圆孔衍射的径向强度 (归一化峰值 1), ``r`` 以相机 px 计。"""
    v = 3.832 * np.asarray(r_px, float) / zero_radius_px
    v = np.where(np.abs(v) < 1e-6, 1e-6, v)
    j1 = _bessel_j1(v)
    return (2 * j1 / v) ** 2


def _bessel_j1(x: np.ndarray) -> np.ndarray:
    """J1 的多项式近似 (Abramowitz & Stegun 9.4.4/9.4.6), 免 scipy 依赖。"""
    ax = np.abs(x)
    small = ax < 3.0
    out = np.empty_like(ax)
    t = (ax[small] / 3.0) ** 2
    out[small] = ax[small] * (
        0.5 - 0.56249985 * t + 0.21093573 * t**2 - 0.03954289 * t**3
        + 0.00443319 * t**4 - 0.00031761 * t**5 + 0.00001109 * t**6
    )
    big = ~small
    z = 3.0 / ax[big]
    f = (
        0.79788456 + 0.00000156 * z + 0.01659667 * z**2 + 0.00017105 * z**3
        - 0.00249511 * z**4 + 0.00113653 * z**5 - 0.00020033 * z**6
    )
    th = (
        ax[big] - 2.35619449 + 0.12499612 * z + 0.00005650 * z**2
        - 0.00637879 * z**3 + 0.00074348 * z**4 + 0.00079824 * z**5
        - 0.00029166 * z**6
    )
    out[big] = np.sqrt(2 / (np.pi * ax[big])) * f * np.cos(th)
    return out


# ---------------------------------------------------------------------------
# 图 1: 四个交付变体
# ---------------------------------------------------------------------------
def fig_four_arms(ctx: dict, out: pathlib.Path) -> pathlib.Path:
    run = ctx["run"]
    side = ctx["side_px"]
    blaze = ctx["blaze"][ctx["blaze_primary"]]

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.6))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # --- A: 平相位 ---
    flat = ctx["flat_1ms"]
    med = float(np.median(flat))
    dc = ctx["dc"]
    sub, x0, y0 = crop(flat, dc[0], dc[1], 200)
    ax_a.imshow(sub, cmap="inferno", vmin=0, vmax=float(sub.max()))
    ax_a.set_title("A  平相位 (flat phase) — 光路本身")
    ax_a.text(
        0.02, 0.98,
        f"1 ms, 峰值 {sub.max():.0f} counts\n"
        f"FWHM  x {ctx['flat_fwhm_x']:.0f} px   y {ctx['flat_fwhm_y']:.0f} px\n"
        f"理想圆孔衍射 FWHM {ctx['airy_fwhm']:.1f} px  →  {ctx['flat_x_ratio']:.2f}x / {ctx['flat_y_ratio']:.2f}x\n"
        f"中心 ({dc[0]:.0f}, {dc[1]:.0f}) px",
        transform=ax_a.transAxes, va="top", ha="left", fontsize=8,
        color="white", bbox=dict(fc="black", alpha=0.55, lw=0),
    )
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    _scale_bar(ax_a, sub.shape, 100, "100 px")

    # --- B: 直接发送 GS (无闪耀光栅) ---
    s = ctx["sides"]
    bsub, bx0, by0 = crop(ctx["side_frame"], s["box_center_x"], s["box_center_y"], 700)
    ax_b.imshow(bsub, cmap="inferno", vmin=0, vmax=30)
    rect = mpatches.Rectangle(
        (s["box_center_x"] - side / 2 - bx0, s["box_center_y"] - side / 2 - by0),
        side, side, fill=False, ec="#39d0ff", lw=1.4,
    )
    ax_b.add_patch(rect)
    ax_b.set_title("B  直接发送 GS (无闪耀光栅)")
    ax_b.text(
        0.02, 0.98,
        f"{s['exposure_ms_readback']:.0f} ms, 色标上限 30 (0 级被截顶)\n"
        f"框内 pib {s['pib']:.3f}   cv {s['cv']:.2f}   score {s['score']:.3f}\n"
        f"方框与 0 级同心 → 0 级在框内饱和, 均匀度指标不可用\n"
        f"饱和像素 {int((ctx['side_frame'] >= 254).sum())}",
        transform=ax_b.transAxes, va="top", ha="left", fontsize=8,
        color="white", bbox=dict(fc="black", alpha=0.55, lw=0),
    )
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    _scale_bar(ax_b, bsub.shape, 200, "200 px")

    # --- C: 经过矫正后发送 GS ---
    if ctx["corrected_frame"] is not None:
        cf = ctx["corrected_frame"]
        csub, cx0, cy0 = crop(cf, s["box_center_x"], s["box_center_y"], 700)
        ax_c.imshow(csub, cmap="inferno", vmin=0, vmax=30)
        ax_c.add_patch(mpatches.Rectangle(
            (s["box_center_x"] - side / 2 - cx0, s["box_center_y"] - side / 2 - cy0),
            side, side, fill=False, ec="#39d0ff", lw=1.4))
        ax_c.set_title("C  经过矫正后发送 GS")
        ax_c.text(0.02, 0.98, ctx["corrected_note"], transform=ax_c.transAxes,
                  va="top", ha="left", fontsize=8, color="white",
                  bbox=dict(fc="black", alpha=0.55, lw=0))
        ax_c.set_xticks([])
        ax_c.set_yticks([])
    else:
        ax_c.set_facecolor("#e8e8ec")
        ax_c.text(
            0.5, 0.62, "本次未采集", ha="center", va="center",
            fontsize=15, color="#8a8a94", fontweight="bold",
        )
        ax_c.text(
            0.5, 0.34,
            "设备在跑到该变体之前下线\n(camera 不枚举 / SLM 写内存 -10032)\n"
            "复现命令见报告 §8 下次实验清单",
            ha="center", va="center", fontsize=8.5, color="#5a5a64",
        )
        ax_c.set_xticks([])
        ax_c.set_yticks([])
        ax_c.set_title("C  经过矫正后发送 GS — 待采集")

    # --- D: 叠加闪耀光栅 ---
    dsub, dx0, dy0 = crop(ctx["blaze_frame"], blaze["box_center_x"], blaze["box_center_y"], 700)
    ax_d.imshow(dsub, cmap="inferno", vmin=0, vmax=60)
    ax_d.add_patch(mpatches.Rectangle(
        (blaze["box_center_x"] - side / 2 - dx0, blaze["box_center_y"] - side / 2 - dy0),
        side, side, fill=False, ec="#39d0ff", lw=1.4))
    # 0 级位置 (取自平相位帧, 该帧 0 级未饱和)
    dcx, dcy = dc[0] - dx0, dc[1] - dy0
    if 0 <= dcx < dsub.shape[1] and 0 <= dcy < dsub.shape[0]:
        ax_d.add_patch(mpatches.Circle((dcx, dcy), 30, fill=False, ec="#ff5b5b", lw=1.2, ls="--"))
        # 标签锚点 clamp 到 axes 内, 避免左半截被切出面板
        margin = 24
        tx = float(np.clip(dcx - 150, margin, dsub.shape[1] - margin - 80))
        ty = float(np.clip(dcy + 130, margin, dsub.shape[0] - margin - 40))
        ax_d.annotate("0 级", (dcx, dcy), (tx, ty), color="#ff5b5b",
                      fontsize=9, arrowprops=dict(arrowstyle="->", color="#ff5b5b", lw=1))
    ax_d.set_title(f"D  叠加闪耀光栅 (P = {ctx['blaze_primary']})")
    ax_d.text(
        0.02, 0.98,
        f"{blaze['exposure_ms_readback']:.0f} ms, 色标上限 60\n"
        f"框内 pib {blaze['pib']:.3f}   cv {blaze['cv']:.2f}   score {blaze['score']:.3f}\n"
        f"位移 实测 {blaze['measured_shift_x_px']:.0f} px / 预测 {blaze['predicted_shift_x_px']:.0f} px\n"
        f"方斑右边缘 {ctx['square_right']:.0f} px (剖面 {ctx['square_right_range'][0]:.0f}–"
        f"{ctx['square_right_range'][1]:.0f}), 高 {ctx['square_h']:.0f} px\n"
        f"纵向中心 {ctx['square_cy']:.0f} px (剖面 {ctx['square_cy_range'][0]:.0f}–"
        f"{ctx['square_cy_range'][1]:.0f}); 左边缘与 0 级光晕相连, 量不出",
        transform=ax_d.transAxes, va="top", ha="left", fontsize=8,
        color="white", bbox=dict(fc="black", alpha=0.55, lw=0),
    )
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    _scale_bar(ax_d, dsub.shape, 200, "200 px")

    fig.suptitle("纯 GS 方斑控制组 — 四个变体 (1000 px 目标, 2f 实验系统)", fontsize=11.5)
    fig.tight_layout()
    p = out / "fig1_four_arms.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def _scale_bar(ax, shape, length: float, label: str) -> None:
    h, w = shape
    x0 = w * 0.06
    y0 = h * 0.94
    ax.plot([x0, x0 + length], [y0, y0], color="white", lw=2.4, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 - h * 0.02, label, color="white", ha="center", va="bottom",
            fontsize=7.5, bbox=dict(fc="black", alpha=0.5, lw=0))


# ---------------------------------------------------------------------------
# 图 2: 全帧概览
# ---------------------------------------------------------------------------
def fig_overview(ctx: dict, out: pathlib.Path) -> pathlib.Path:
    blaze = ctx["blaze"][ctx["blaze_primary"]]
    frame = ctx["blaze_frame"]
    side = ctx["side_px"]
    dc = ctx["dc"]

    fig, ax = plt.subplots(figsize=(11.0, 8.4))
    im = ax.imshow(frame, cmap="inferno", norm=LogNorm(vmin=1.0, vmax=FULL_SCALE))
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("counts / px (8 bit, 对数)")

    ax.add_patch(mpatches.Rectangle(
        (blaze["box_center_x"] - side / 2, blaze["box_center_y"] - side / 2),
        side, side, fill=False, ec="#39d0ff", lw=1.6,
        label=f"目标方框 1000 px (中心 {blaze['box_center_x']:.0f}, {blaze['box_center_y']:.0f})"))
    ax.add_patch(mpatches.Circle(dc, 40, fill=False, ec="#ff5b5b", lw=1.8, ls="--"))
    ax.annotate("0 级", xy=dc, xytext=(dc[0] - 180, dc[1] - 120),
                color="#ff5b5b", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#ff5b5b", lw=1.2))
    ax.annotate("", xy=(ctx["square_cx"], ctx["square_cy"]), xytext=dc,
                arrowprops=dict(arrowstyle="->", color="#7CFC00", lw=1.6))
    ax.text((dc[0] + ctx["square_cx"]) / 2, dc[1] - 90,
            f"闪耀位移 {blaze['measured_shift_x_px']:.0f} px (预测 {blaze['predicted_shift_x_px']:.0f})",
            color="#7CFC00", fontsize=9, ha="center",
            bbox=dict(fc="black", alpha=0.6, lw=0))
    ax.set_title(
        f"叠加闪耀光栅 P={ctx['blaze_primary']} — 全帧 ({frame.shape[1]}x{frame.shape[0]}), "
        f"{blaze['exposure_ms_readback']:.0f} ms"
    )
    ax.set_xlabel("x / 相机 px")
    ax.set_ylabel("y / 相机 px")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    p = out / "fig2_blaze_overview.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 图 3: 模型 vs 实测
# ---------------------------------------------------------------------------
def fig_model_vs_measured(ctx: dict, out: pathlib.Path) -> pathlib.Path:
    blaze = ctx["blaze"][ctx["blaze_primary"]]
    side = ctx["side_px"]
    model = ctx["blaze_model"]
    dc = ctx["dc"]

    cx, cy = blaze["box_center_x"], blaze["box_center_y"]
    sub, x0, y0 = crop(ctx["blaze_frame"], cx, cy, 700)
    msub = resample_model_to_crop(model, dc, x0, y0, sub.shape[0])

    m = prepare_frame(sub)

    # 模型的残余相干 0 级 (方斑正中单样本尖峰, ~214x 平台) 与实测的 0 级 (左下方,
    # 框外) 同源 —— 残余 0 级随闪耀一起搬进框内 (probe: 模型 argmax 在框内
    # (1886, 944), 实测框内最亮像素 (1884, 928), 偏差 2 px / 16 px), 对齐 OK。
    # 每个面板按**自身框内水平**缩放, 而不是按全局最大值: 模型残余 0 级是平台的
    # 214x, 若按全局 max 缩放, 平台被压到色标的 ~8% (不可见) —— 这正是 defect。
    box_left = int(cx - side / 2 - x0)
    box_top = int(cy - side / 2 - y0)
    box_mask = np.zeros_like(m, dtype=bool)
    box_mask[box_top:box_top + int(side), box_left:box_left + int(side)] = True
    model_inbox = msub[box_mask]
    meas_inbox = m[box_mask]
    vmax_model = 1.5 * float(np.median(model_inbox)) if np.median(model_inbox) > 0 else 1.0
    vmax_meas = 1.5 * float(np.median(meas_inbox)) if np.median(meas_inbox) > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.4))
    for ax, img, title, vmax in (
        (axes[0], msub, "理想模型 (面板分辨率, 同一相位)", vmax_model),
        (axes[1], m, f"实测 ({blaze['exposure_ms_readback']:.0f} ms)", vmax_meas),
    ):
        im = ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax)
        ax.add_patch(mpatches.Rectangle(
            (cx - side / 2 - x0, cy - side / 2 - y0), side, side,
            fill=False, ec="#39d0ff", lw=1.4))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)

    # 模型面板的色标单位是归一化强度, 实测面板是 counts/px; 各面板色标按自身框内
    # 中位数 1.5x 缩放, 残余 0 级已截顶。
    axes[0].text(
        0.02, 0.98,
        f"框内 pib {ctx['model_pib_box']:.3f}  (分母 = 相机视场内的模型光)\n"
        f"整个模型场口径 {ctx['model_pib_all']:.3f} —— 含画面外的光, 不可与实测比\n"
        f"框内 cv {ctx['model_cv_box']:.2f}  (剔残余级次 {ctx['model_cv_box_masked']:.2f})\n"
        f"相机视场内只有模型的 {ctx['model_in_view']*100:.1f}% 光\n"
        f"色标按框内中位数 1.5x 缩放; 残余 0 级已截顶",
        transform=axes[0].transAxes, va="top", ha="left", fontsize=8,
        color="white", bbox=dict(fc="black", alpha=0.55, lw=0),
    )
    axes[1].text(
        0.02, 0.98,
        f"框内 pib {blaze['pib']:.3f}\n"
        f"框内 cv {blaze['cv']:.2f}  (剔饱和 {ctx['measured_cv_ex_sat']:.2f})\n"
        f"框外 {ctx['outside_share']*100:.0f}% (其中大部分是均匀底, 见 fig6)\n"
        f"色标按框内中位数 1.5x 缩放",
        transform=axes[1].transAxes, va="top", ha="left", fontsize=8,
        color="white", bbox=dict(fc="black", alpha=0.55, lw=0),
    )
    fig.suptitle("同一相位: 理想模型 vs 实测 —— 色标各自按框内水平缩放", fontsize=11)
    fig.tight_layout()
    p = out / "fig3_model_vs_measured.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 图 4: 剖面
# ---------------------------------------------------------------------------
def fig_profiles(ctx: dict, out: pathlib.Path) -> pathlib.Path:
    blaze = ctx["blaze"][ctx["blaze_primary"]]
    side = ctx["side_px"]
    cx, cy = blaze["box_center_x"], blaze["box_center_y"]
    frame = ctx["blaze_frame"]
    model = ctx["blaze_model"]
    dc = ctx["dc"]

    H, W = frame.shape
    # 剖面窗以方框中心为心, 但必须裁到帧内 —— 方斑靠右, +800 会越界
    x0, x1 = max(0, int(cx - 800)), min(W, int(cx + 800))
    y0, y1 = max(0, int(cy - 800)), min(H, int(cy + 800))
    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    _all_vals: list[float] = []
    for ax, (cut, lab) in zip(
        axes,
        [("h", "水平剖面 (沿 y = %.0f)" % cy), ("v", "垂直剖面 (沿 x = %.0f)" % cx)],
    ):
        if cut == "h":
            meas = prepare_frame(frame)[int(round(cy)), x0:x1]
            coord = xs
            col, row = cam_to_model(coord, np.full_like(coord, cy, float), dc)
            prof = model[np.clip(row.round().astype(int), 0, model.shape[0] - 1),
                         np.clip(col.round().astype(int), 0, model.shape[1] - 1)]
        else:
            meas = prepare_frame(frame)[y0:y1, int(round(cx))]
            coord = ys
            col, row = cam_to_model(np.full_like(coord, cx, float), coord, dc)
            prof = model[np.clip(row.round().astype(int), 0, model.shape[0] - 1),
                         np.clip(col.round().astype(int), 0, model.shape[1] - 1)]

        prof = prof / prof.max() * meas.max() if prof.max() > 0 else prof
        ax.plot(coord, meas, lw=1.3, color="#c1121f", label="实测")
        ax.plot(coord, prof, lw=1.3, color="#1d4e89", label="理想模型")
        edges = (cx - side / 2, cx + side / 2) if cut == "h" else (cy - side / 2, cy + side / 2)
        for _i, edge in enumerate(edges):
            # 两条边线同名 → 图例只标第一条, 否则图例里出现两行一样的条目
            ax.axvline(edge, color="#39d0ff", lw=1.0, ls="--",
                       label="方框边 (1000 px)" if _i == 0 else None)
        ax.set_title(lab)
        ax.set_xlabel("相机 px")
        ax.set_ylabel("counts / px (已减中位数)")
        ax.grid(alpha=0.25)
        # 共享 y 轴上限用的采样 (供循环后统一设 ylim)
        _all_vals.extend(meas.tolist())
        _all_vals.extend(prof.tolist())
        # 水平剖面里, 框外左侧的 0 级峰值 (~1220 px) 标注
        if cut == "h":
            pk = int(np.argmax(meas))
            ax.annotate("0 级", xy=(coord[pk], meas[pk]),
                        xytext=(coord[pk] - 120, meas[pk] * 0.6),
                        color="#c1121f", fontsize=8,
                        arrowprops=dict(arrowstyle="->", color="#c1121f", lw=0.9))
        ax.legend(fontsize=8)

    # 两面板共享 y 上限, 与 suptitle 的「共享峰值缩放」一致
    _ymax = max(_all_vals) * 1.08 if _all_vals else 1.0
    for _ax in axes:
        _ax.set_ylim(0, _ymax)

    fig.suptitle("方斑剖面: 实测 vs 理想模型 (各面板模型按该面板实测峰值缩放)", fontsize=11)
    fig.tight_layout()
    p = out / "fig4_profiles.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 图 5: 平板光斑 vs 衍射极限
# ---------------------------------------------------------------------------
def fig_flat_psf(ctx: dict, out: pathlib.Path) -> pathlib.Path:
    flat = ctx["flat_1ms"]
    dc = ctx["dc"]
    sub, x0, y0 = crop(flat, dc[0], dc[1], 60)
    p = prepare_frame(sub)
    iy, ix = np.unravel_index(np.argmax(p), p.shape)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    im = axes[0].imshow(sub, cmap="inferno", vmin=0, vmax=float(sub.max()))
    axes[0].set_title("平相位光斑 (1 ms), 裁 120x120 px")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.01)

    r = np.arange(p.shape[1]) - ix
    hx = p[iy, :]
    vy = p[:, ix]
    axes[1].plot(r, hx, lw=1.3, color="#c1121f", label=f"x 向, FWHM {ctx['flat_fwhm_x']:.1f} px")
    axes[1].plot(r, vy, lw=1.3, color="#1d4e89", label=f"y 向, FWHM {ctx['flat_fwhm_y']:.1f} px")
    rho = np.linspace(float(r.min()), float(r.max()), 601)
    axes[1].plot(rho, airy_profile(np.abs(rho), ctx["airy_zero"]) * p.max() / 1.0,
                 lw=1.2, ls="--", color="#2a9d8f",
                 label=f"理想圆孔 (r=450 panel px): FWHM {ctx['airy_fwhm']:.2f} px")
    axes[1].set_ylim(-5, p.max() * 1.08)
    axes[1].set_xlabel("距峰值的相机 px")
    axes[1].set_ylabel("counts / px (已减中位数)")
    axes[1].set_title("剖面与衍射极限")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].text(0.02, 0.62,
                 f"实测 / 衍射极限:  x {ctx['flat_x_ratio']:.2f}x   y {ctx['flat_y_ratio']:.2f}x\n"
                 "x 向多出的部分集中在单侧肩部 → 光路有未校正像差",
                 transform=axes[1].transAxes, fontsize=8, va="top")
    fig.suptitle("光路自身的光斑: 非衍射极限, 这是方斑边缘偏软的第一个来源", fontsize=11)
    fig.tight_layout()
    out_p = out / "fig5_flat_psf.png"
    fig.savefig(out_p)
    plt.close(fig)
    return out_p


# ---------------------------------------------------------------------------
# 图 6: 8 bit 与那层弥散底
# ---------------------------------------------------------------------------
def fig_diffuse_floor(ctx: dict, out: pathlib.Path) -> pathlib.Path:
    blaze = ctx["blaze"][ctx["blaze_primary"]]
    side = ctx["side_px"]
    frame = ctx["blaze_frame"]
    dc = ctx["dc"]
    H, W = frame.shape

    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - dc[0], yy - dc[1])
    med = float(np.median(frame))
    q = (frame - med).clip(0)

    bins = np.array([0, 10, 25, 50, 100, 200, 400, 700, 1000, 1400, 1900])
    centres, means = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum():
            centres.append((lo + hi) / 2)
            means.append(float(q[m].mean()))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ax = axes[0]
    ax.semilogy(centres, means, "o-", lw=1.4, color="#c1121f",
                label=f"实测 P={ctx['blaze_primary']} ({blaze['exposure_ms_readback']:.0f} ms, 已减中位数)")
    ax.axhline(med, color="#888", lw=1.0, ls=":", label=f"该帧中位数 {med:.1f} counts/px")
    ax.axhline(ctx["floor_outside"], color="#1d4e89", lw=1.0, ls="--",
               label=f"框外均值 {ctx['floor_outside']:.2f} counts/px")
    ax.axhline(ctx["pedestal"], color="#2a9d8f", lw=1.0, ls="-.",
               label=f"0.05 ms 帧的静态偏置 {ctx['pedestal']:.2f} counts/px")
    ax.axvspan(0, 1, alpha=0)  # 占位, 保持 x 轴从 0 起
    ax.set_xlabel("距 0 级 (中心) 的相机 px")
    ax.set_ylabel("平均 counts / px")
    ax.set_title("径向亮度: 0 级 / 方斑平台 / 一直铺到画面边缘的底")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_ylim(max(min(means) * 0.5, 1e-3), max(means) * 1.5)
    # 最外环 (中心 1650 px) 已到画面角上, 均值 ~0.01 counts/px —— 仍画出来, 不丢点
    ax.annotate("最外环 (中心 1650 px, n=224689)\n均值 0.01 counts/px, 已到画面角",
                xy=(centres[-1], means[-1]), xytext=(centres[-1] - 320, means[-1] * 8),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

    ax = axes[1]
    pairs = ctx["derivative_pairs"]
    labels = [f"{a:g}→{b:g}" for a, b in pairs["pairs"]]
    vals = pairs["halo_per_ms"]
    colors = ["#c1121f" if v > 0 else "#8a8a94" for v in vals]
    ax.barh(labels, vals, color=colors, height=0.62)
    # 可读范围: 0 … 1.6e5; 最早一对 (-5.83e5) 超出, 用标记 + 注解说明, 不静默丢掉
    ax.set_xlim(0, 1.6e5)
    ax.axvline(0, color="#444", lw=0.8)
    # 超限条的标记 (左脊柱处)
    out_idx = 0
    ax.annotate(
        f"{labels[out_idx]}: {vals[out_idx]:.2e} counts/ms  (超出量程)\n"
        "0.05 ms 帧中位数 0、σ≈1, 该对 meaningless, 不引用",
        xy=(0, out_idx), xytext=(0.02, 0.18),
        transform=ax.transAxes, ha="left", va="center", fontsize=7.5,
        color="#8a8a94",
        arrowprops=dict(arrowstyle="->", color="#8a8a94", lw=0.8),
    )
    ax.set_xlabel("r>50 的计数增量 / 曝光增量  (counts/ms)")
    ax.set_title("为什么 1 ms 帧上的「弥散光占比」不可信")
    ax.grid(alpha=0.25, axis="x")
    ax.invert_yaxis()
    spread = max(vals) / min(v for v in vals if v > 0)
    # 注解移出图区, 不遮挡条形
    fig.text(
        0.5, 0.02,
        f"七对增量差 {spread:.0f}x, 且最早一对为负; 帧间偏置漂移 ≈ {ctx['pedestal_drift']:.1e} counts,\n"
        f"与 1 ms 处的整个光信号同量级。只有 20→50 ms 那一对 (光增量 "
        f"{ctx['long_pair_increment']:.1e} counts) 高于漂移, 它的框外份额 "
        f"{ctx['long_pair_share']*100:.0f}% 才可引用。",
        ha="center", va="bottom", fontsize=7.5,
        bbox=dict(fc="#fff4e6", ec="#e0a458", lw=0.8),
    )

    fig.suptitle("8 bit 相机的量程与那层铺满全帧的底", fontsize=11)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    p = out / "fig6_diffuse_floor.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_context(run: pathlib.Path, flat_dir: pathlib.Path, corrected: pathlib.Path | None) -> dict:
    sides = read_csv(run / "sides/measurements.csv")[0]
    blaze_rows = read_csv(run / "blaze/measurements.csv")
    blaze_rows = [r for r in blaze_rows if not r.get("error")]
    sides = {k: (float(v) if _is_num(v) else v) for k, v in sides.items()}
    blaze = {}
    for r in blaze_rows:
        rr = {k: (float(v) if _is_num(v) else v) for k, v in r.items()}
        # 帧/相位文件按 CSV 的行序编号 (frame_000 = 第一个周期), 不是按周期数值
        rr["frame_index"] = int(r["index"])
        blaze[int(rr["blaze_period_px"])] = rr
    primary = max(blaze)  # P=12: 0 级离框最近, 但位移最稳

    flat1 = load_flat(flat_dir, 1)
    flat005 = load_flat(flat_dir, 0.05)

    # 0 级: 平相位帧上取峰 (600 ms 的方斑帧里 0 级饱和, 定不到中心)
    p1 = prepare_frame(flat1)
    iy, ix = np.unravel_index(np.argmax(p1), p1.shape)
    dc = (float(ix), float(iy))

    def fwhm(prof):
        above = np.where(prof >= prof.max() / 2)[0]
        return float(above[-1] - above[0] + 1) if above.size else float("nan")

    fw_x = fwhm(p1[iy, max(0, ix - 60):ix + 60])
    fw_y = fwhm(p1[max(0, iy - 60):iy + 60, ix])

    airy_zero = 1.22 / (2 * BEAM_RADIUS_PANEL) * TILT_SHIFT_SCALE
    airy_fwhm = 0.843 * airy_zero

    pidx = blaze[primary]["frame_index"]
    frame = load_frame(run / "blaze" / f"frame_{pidx:03d}.npz")
    side_frame = load_frame(run / "sides/side_1000/frame_000.npz")
    phase = np.load(run / "blaze" / f"phase_{pidx:03d}.npy")

    model = panel_far_field(phase)
    H, W = frame.shape
    window = (0.0, float(W - 1), 0.0, float(H - 1))
    side_px = float(sides["box_side_px"])
    box_c = (blaze[primary]["box_center_x"], blaze[primary]["box_center_y"])

    model_pib_box = model_box_pib(model, dc, box_c, side_px, window)
    model_pib_all = model_box_pib(model, dc, box_c, side_px, None)
    model_cv_box = model_cv(model, dc, box_c, side_px)

    # GS 相位-only 近似的**残余相干 0 级**: 方斑正中一个单样本尖峰 (3.8e-3 vs 平台
    # 1.8e-5, 差 214x), 它随闪耀一起搬走, 所以在模型方框内。它单独就把 std/mean
    # 推到 1.6, 因此另给一个"剔掉残余级次"的 CV, 才是在说方斑本身的均匀度。
    _col, _row = cam_to_model(box_c[0], box_c[1], dc)
    _hx = side_px / 2 / MODEL_COL_TO_CAM_Y
    _hy = side_px / 2 / MODEL_ROW_TO_CAM_X
    _sub = model[int(_row - _hy):int(_row + _hy), int(_col - _hx):int(_col + _hx)]
    _floor = float(np.median(_sub))
    _hot = _sub > 20 * _floor
    model_cv_box_masked = (float(_sub[~_hot].std() / _sub[~_hot].mean())
                           if _sub[~_hot].size else float("nan"))
    residual_order_share = float(_sub[_hot].sum() / _sub.sum()) if _sub.sum() else float("nan")

    # 重定心的模型方框: 框心 = 真实 0 级 (1 ms 平场峰) + 计划位移。把"方框摆位误差"
    # 从算法质量里剥出来 —— 实测 0.737 里含着同一个摆位误差, 所以这个数**不可**直接
    # 与实测比, 只用于说明算法上限。
    _px = float(blaze[primary]["predicted_shift_x_px"])
    _py = float(blaze[primary]["predicted_shift_y_px"])
    recentre = (dc[0] + _px, dc[1] + _py)
    model_pib_box_recentred = model_box_pib(model, dc, recentre, side_px, window)
    model_cv_box_recentred = model_cv(model, dc, recentre, side_px)

    # 实测方斑是**散斑场**, 不是平顶: 剔掉饱和像素后箱内 CV 仍远高于模型
    _prep = prepare_frame(frame)
    _bx0, _bx1 = int(box_c[0] - side_px / 2), int(box_c[0] + side_px / 2)
    _by0, _by1 = int(box_c[1] - side_px / 2), int(box_c[1] + side_px / 2)
    _mbox = _prep[_by0:_by1, _bx0:_bx1]
    _msat = frame[_by0:_by1, _bx0:_bx1] >= 254
    measured_cv_ex_sat = (float(_mbox[~_msat].std() / _mbox[~_msat].mean())
                          if _mbox[~_msat].size else float("nan"))
    measured_sat_share = float(_msat.mean())

    # 残余 0 级就落在方框正中, 它的光晕把箱内 CV 推到 ~1。方框边缘又与方斑边缘
    # 重合 (方斑边缘是软的), 于是"箱内 CV"同时混了边缘失配和残余级次, 两个方向
    # 都不可比。给一个**方框正中 60%** 的 CV: 完全避开边缘与残余级次, 只量方斑
    # 自身是平顶还是散斑 —— 这才是模型与实测唯一可比的均匀度。
    _FRAC = 0.6
    _ch = side_px * _FRAC / 2
    _cen = np.zeros(frame.shape, bool)
    _cen[int(box_c[1] - _ch):int(box_c[1] + _ch),
         int(box_c[0] - _ch):int(box_c[0] + _ch)] = True
    measured_cv_center = (float(_prep[_cen].std() / _prep[_cen].mean())
                          if _prep[_cen].size else float("nan"))
    model_cv_center = model_cv(model, dc, box_c, side_px, center_frac=_FRAC)

    # 运行期冻结的 0 级 (600 ms 饱和帧上定位, 偏了) —— 方框摆位误差的来源
    try:
        _rc = json.loads((run / "blaze/run_config.json").read_text(encoding="utf-8"))
        dc_frozen = tuple(float(v) for v in _rc["zero_order_xy_frozen"])
    except (OSError, KeyError, ValueError):
        dc_frozen = None

    # 相机视场内的模型总份额
    hh, ww = model.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    cam_x, cam_y = model_to_cam(xx, yy, dc)
    view = (cam_x >= 0) & (cam_x <= W - 1) & (cam_y >= 0) & (cam_y <= H - 1)
    model_in_view = float(model[view].sum())

    # 方斑实测尺寸 —— 只量**能干净量**的那两条: 右边缘与纵向高度/中心。
    # 左边缘与 0 级光晕相连 (600 ms 下 0 级核饱和并拖出条纹), 量不出来; 横向宽度
    # 因此由「右边缘 − 设计边长」反推, 不是直接读数。
    # 单条剖面在散斑场上散布 ±20 px (相邻剖面差一个颗粒直径), 所以逐行/逐列各取
    # 一串剖面、报中位数, 并把散布一并记下来 —— 单个数会假装精确。
    med = float(np.median(frame))
    q = (frame - med).clip(0)
    _bx = int(round(box_c[0]))
    _by = int(round(box_c[1]))
    rights, heights, centres_y = [], [], []
    for yy in range(_by - int(side_px * 0.4), _by + int(side_px * 0.4) + 1, 20):
        if not (0 <= yy < H):
            continue
        idx = np.where(q[yy] > _THR)[0]
        if idx.size:
            rights.append(float(idx.max()))
    for xx in range(_bx - int(side_px * 0.3), _bx + int(side_px * 0.4) + 1, 20):
        if not (0 <= xx < W):
            continue
        idy = np.where(q[:, xx] > _THR)[0]
        if idy.size and (idy.max() - idy.min() + 1) > 0.5 * side_px:
            heights.append(float(idy.max() - idy.min() + 1))
            centres_y.append(float((idy.max() + idy.min()) / 2))
    square_right = float(np.median(rights)) if rights else float("nan")
    square_right_range = _spread(rights)
    square_h = float(np.median(heights)) if heights else float("nan")
    square_h_range = _spread(heights)
    square_cy = float(np.median(centres_y)) if centres_y else float("nan")
    square_cy_range = _spread(centres_y)
    # 横向中心由右边缘反推 (方斑设计边长已知), 宽度不单独读
    square_w = side_px
    square_cx = square_right - side_px / 2

    m = np.ones(frame.shape, bool)
    m[int(box_c[1] - side_px / 2):int(box_c[1] + side_px / 2),
      int(box_c[0] - side_px / 2):int(box_c[0] + side_px / 2)] = False
    prep = _prep
    outside_share = float(prep[m].sum() / prep.sum())

    # 增量对
    rr = np.hypot(np.mgrid[0:H, 0:W][1] - dc[0], np.mgrid[0:H, 0:W][0] - dc[1])
    core = rr < 50
    order = [0.05, 0.2, 1, 2, 5, 10, 20, 50]
    e2 = {e: load_flat(flat_dir, e) for e in order}
    pairs, halo_pm = [], []
    for a, b in zip(order[:-1], order[1:]):
        d = (e2[b][~core].sum() - e2[a][~core].sum()) / (b - a)
        pairs.append((a, b))
        halo_pm.append(float(d))

    long_inc = float((e2[50][~core].sum() - e2[20][~core].sum()))
    long_share = float((e2[50][~core].sum() - e2[20][~core].sum())
                       / (e2[50].sum() - e2[20].sum()))
    pedestal = float(flat005.mean())
    pedestal_drift = float(abs(e2[1].sum() - e2[2].sum()))

    return dict(
        run=run, flat_dir=flat_dir, sides=sides, blaze=blaze, blaze_primary=primary,
        side_px=side_px, dc=dc, flat_1ms=flat1, side_frame=side_frame,
        blaze_frame=frame, blaze_model=model,
        flat_fwhm_x=fw_x, flat_fwhm_y=fw_y, airy_zero=airy_zero, airy_fwhm=airy_fwhm,
        flat_x_ratio=fw_x / airy_fwhm, flat_y_ratio=fw_y / airy_fwhm,
        model_pib_box=model_pib_box, model_pib_all=model_pib_all,
        model_cv_box=model_cv_box, model_in_view=model_in_view,
        model_cv_box_masked=model_cv_box_masked,
        residual_order_share=residual_order_share,
        model_pib_box_recentred=model_pib_box_recentred,
        model_cv_box_recentred=model_cv_box_recentred,
        recentre=recentre,
        measured_cv_ex_sat=measured_cv_ex_sat,
        measured_sat_share=measured_sat_share,
        measured_cv_center=measured_cv_center,
        model_cv_center=model_cv_center,
        cv_center_frac=_FRAC,
        dc_frozen=dc_frozen,
        outside_share=outside_share,
        square_w=square_w, square_h=square_h, square_cx=square_cx, square_cy=square_cy,
        square_right=square_right, square_right_range=square_right_range,
        square_h_range=square_h_range, square_cy_range=square_cy_range,
        floor_outside=float(frame[m].mean()), pedestal=pedestal,
        frame_median=med,
        pedestal_drift=pedestal_drift,
        derivative_pairs=dict(pairs=pairs, halo_per_ms=halo_pm),
        long_pair_increment=long_inc, long_pair_share=long_share,
        corrected_frame=load_frame(corrected) if corrected else None,
        corrected_note="",
    )


def _is_num(v) -> bool:
    if v is None or v == "":
        return False
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-root", type=pathlib.Path, default=DEFAULT_RUN)
    ap.add_argument("--flat-dir", type=pathlib.Path, default=DEFAULT_FLAT)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--corrected-frame", type=pathlib.Path, default=None,
                    help="「经过矫正后发送 GS」那一变体的帧 (.npz, key=frame)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只画指定图 (four_arms overview model profiles psf floor)")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    ctx = build_context(args.run_root, args.flat_dir, args.corrected_frame)
    if ctx["corrected_frame"] is not None:
        res = measure_box(ctx["corrected_frame"],
                          (ctx["sides"]["box_center_x"], ctx["sides"]["box_center_y"]),
                          ctx["side_px"])
        ctx["corrected_note"] = (
            f"pib {res['pib']:.3f}  cv {res['cv']:.2f}  score {res['score']:.3f}"
        )

    wanted = args.only or ["four_arms", "overview", "model", "profiles", "psf", "floor"]
    made = []
    if "four_arms" in wanted:
        made.append(fig_four_arms(ctx, args.out))
    if "overview" in wanted:
        made.append(fig_overview(ctx, args.out))
    if "model" in wanted:
        made.append(fig_model_vs_measured(ctx, args.out))
    if "profiles" in wanted:
        made.append(fig_profiles(ctx, args.out))
    if "psf" in wanted:
        made.append(fig_flat_psf(ctx, args.out))
    if "floor" in wanted:
        made.append(fig_diffuse_floor(ctx, args.out))

    summary = {
        "run_root": str(args.run_root),
        "dc_px": ctx["dc"],
        "flat_fwhm_px": [ctx["flat_fwhm_x"], ctx["flat_fwhm_y"]],
        "airy_fwhm_px": ctx["airy_fwhm"],
        "airy_zero_px": ctx["airy_zero"],
        "flat_ratio_to_airy": [ctx["flat_x_ratio"], ctx["flat_y_ratio"]],
        "square_measured_px": [ctx["square_w"], ctx["square_h"]],
        "square_center_px": [ctx["square_cx"], ctx["square_cy"]],
        "square_right_edge_px": ctx["square_right"],
        "square_right_edge_range_px": list(ctx["square_right_range"]),
        "square_height_range_px": list(ctx["square_h_range"]),
        "square_center_y_range_px": list(ctx["square_cy_range"]),
        "square_left_edge_unmeasurable": "左边缘与 0 级光晕相连; 宽度由右边缘减设计边长反推",
        "model_pib_box": ctx["model_pib_box"],
        "model_pib_box_denominator": "相机视场内的模型光 —— 与实测 pib 同口径",
        "model_pib_box_whole_field": ctx["model_pib_all"],
        "model_pib_box_whole_field_note": "分母是整个 7400^2 模型场, 含落在画面外的光; 不可与实测 pib 直接比",
        "outside_box_share_model": 1.0 - ctx["model_pib_box"],
        "model_cv_box": ctx["model_cv_box"],
        "model_cv_box_ex_residual_order": ctx["model_cv_box_masked"],
        "model_residual_order_energy_share": ctx["residual_order_share"],
        "model_pib_box_recentred": ctx["model_pib_box_recentred"],
        "model_cv_box_recentred": ctx["model_cv_box_recentred"],
        "recentre_box_center_px": list(ctx["recentre"]),
        "predicted_square_right_edge_px": ctx["recentre"][0] + ctx["side_px"] / 2,
        "measured_cv_ex_saturated": ctx["measured_cv_ex_sat"],
        "measured_saturated_share_in_box": ctx["measured_sat_share"],
        "cv_center_frac": ctx["cv_center_frac"],
        "model_cv_center": ctx["model_cv_center"],
        "measured_cv_center": ctx["measured_cv_center"],
        "model_light_inside_camera_view": ctx["model_in_view"],
        "dc_frozen_px": list(ctx["dc_frozen"]) if ctx["dc_frozen"] else None,
        "dc_frozen_error_px": (list(np.subtract(ctx["dc"], ctx["dc_frozen"]))
                               if ctx["dc_frozen"] else None),
        "outside_box_share_measured": ctx["outside_share"],
        "outside_box_mean_counts": ctx["floor_outside"],
        "pedestal_0.05ms_counts": ctx["pedestal"],
        "frame_median_counts": ctx["frame_median"],
        "derivative_pairs_counts_per_ms": dict(
            zip([f"{a:g}->{b:g}" for a, b in ctx["derivative_pairs"]["pairs"]],
                ctx["derivative_pairs"]["halo_per_ms"])),
        "long_pair_increment_counts": ctx["long_pair_increment"],
        "long_pair_outside_share": ctx["long_pair_share"],
        "pedestal_drift_counts": ctx["pedestal_drift"],
        "figures": [str(p) for p in made],
    }
    (args.out.parent / "figure_numbers.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
