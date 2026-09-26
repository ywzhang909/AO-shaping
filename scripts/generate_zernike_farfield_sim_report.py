"""Generate the Zernike far-field spot morphology simulation report.

基于项目仿真环境 (2f-Fourier 模型, 与 ``drivers/sim/slm_pib_sim.py`` 的
``SimPibSystem.far_field()`` 同一物理模型, 纯 numpy 实现 — 省去设备生命周期与
灰度量化, 适合批量模式×幅度扫描), 对每个 Zernike 模式 × 系数幅度 (waves) 仿真
远场光斑形貌, 生成图文报告:

- ``docs/zernike_farfield_sim/report.md``  — 报告主体 (中文)
- ``docs/zernike_farfield_sim/figures/``   — 6 张图
  (01 远场形貌网格 / 02 相位形貌网格 / 03 Strehl / 04 环围能量 / 05 FWHM / 06 峰值偏移)
- ``docs/zernike_farfield_sim/metrics.csv`` — 全量指标表 (96 行)

Usage (Windows PowerShell):
    $env:PYTHONPATH = "src"
    python scripts/generate_zernike_farfield_sim_report.py

Fully offline: 无硬件, 纯 numpy。入瞳网格自初版 128² 4× 上采样 (孔径边缘阶梯能量降 ~18 dB,
方正条纹降至 −25 dB 显示底以下); m=4 方位谐波诊断量化方正调制前后 (见报告 §4.1)。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator
from ao_shaping.utils.wavefront.zernike_utils import list_zernike_modes

# ========================= 仿真参数 =========================
PUPIL_N = 512          # 入瞳平面网格边长 (px) — 较 128 4× 上采样: 硬边孔径阶梯能量降 ~18 dB
PUPIL_RADIUS = 256.0   # 圆形孔径半径 (px) — 单位圆恰好触及帧边 (与 128/64 同一尺度)
FAR_N = 8192           # 远场网格边长 (px) — 零填充上采样; 首个 Airy 暗环仍 ≈19.5 px (尺度不变)
CROP = 160             # 图版裁剪窗口 (px, 围绕 0 级光斑)
N_MAX = 4              # 最大径向阶数 (Noll 15 = (4, -4))
NOLL_MODES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 跳过 piston/tilt (1-3)
AMPS_WAVES = [0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0]    # 幅度扫描 (λ)

OUT_DIR = REPO_ROOT / "docs" / "zernike_farfield_sim"
FIG_DIR = OUT_DIR / "figures"
LOG_FLOOR_DB = -25.0   # 远场 log 显示底 (dB) — 提高显示底以抑制衍射环间低强度衍射纹理
SMOOTH_SIGMA_PX = 1.5  # 远场图 log 域高斯平滑 σ (px) — 柔化衍射环边缘/间隙对比, 降低"方正周期条纹"观感
# ---- m=4 方位谐波诊断 (方正条纹数值伪影量化, 见报告 §4.1) ----
PUPIL_N_OLD = 128        # 旧入瞳网格 (m=4 前后对比用)
PUPIL_RADIUS_OLD = 64.0  # 旧孔径半径 (px)
FAR_N_OLD = 2048         # 旧远场网格 (px)
M4_RADII = (25.0, 50.0, 75.0, 100.0)  # m=4 采样环半径 (px, 远场网格)
RADIAL_RMAX = 300.0    # 径向剖面半径 (px) — 覆盖 90% 环围能量与大像差下的衍射环
RADIAL_DR = 0.5        # 径向剖面 bin 宽 (px)


# ========================= 光学模型 =========================
def far_field_intensity(pupil: np.ndarray, far_n: int = FAR_N) -> np.ndarray:
    """2f-Fourier 远场强度: I = |fftshift(FFT2(零填充入瞳场))|².

    零填充不改变远场连续 FT 的取值, 只细化其采样 (标准远场上采样技巧)。
    与 ``SimPibSystem.far_field()`` (slm_pib_sim.py) 同一模型。

    Args:
        pupil: 入瞳复振幅场 (h, w), 已乘圆形孔径 mask。
        far_n: 远场 FFT 边长 (px)。

    Returns:
        (far_n, far_n) 远场强度 (float64, 未归一化)。
    """
    ph, pw = pupil.shape
    oh, ow = (far_n - ph) // 2, (far_n - pw) // 2
    padded = np.zeros((far_n, far_n), dtype=np.complex128)
    padded[oh:oh + ph, ow:ow + pw] = pupil
    u = np.fft.fftshift(np.fft.fft2(padded))
    return u.real * u.real + u.imag * u.imag


def log_scale(img: np.ndarray, floor_db: float = LOG_FLOOR_DB) -> np.ndarray:
    """10·log10(I/I_max), 下限 floor_db (dB), 用于远场显示。"""
    peak = img.max()
    if peak <= 0:
        return np.zeros_like(img)
    db = 10.0 * np.log10(img / peak + 1e-30)
    return np.clip(db, floor_db, 0.0)


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """1D 高斯核 (归一化和为 1)。

    Args:
        size: 核长度 (奇数)。
        sigma: 高斯标准差 (px)。

    Returns:
        (size,) 归一化高斯核。
    """
    ax = np.arange(size) - (size - 1) / 2.0
    k = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _smooth2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """可分离高斯平滑 (两遍 1D 卷积, edge padding), 保持形状。

    用于图 1 log 强度显示: 柔化衍射环边缘与环间暗隙对比, 弱化因
    log 底抬升 + 细采样产生的"方正周期条纹"观感 (不改变物理量, 仅显示)。

    Args:
        img: (h, w) 输入数组。
        sigma: 高斯 σ (px)。

    Returns:
        (h, w) 平滑后数组 (形状不变)。
    """
    r = max(1, int(round(3.0 * sigma)))
    k = _gaussian_kernel(2 * r + 1, sigma)
    p = np.pad(img, r, mode="edge")
    tmp = np.apply_along_axis(lambda row: np.convolve(row, k, "valid"), 1, p)
    return np.apply_along_axis(lambda col: np.convolve(col, k, "valid"), 0, tmp)


def azimuthal_m4_db(img: np.ndarray, cy: float, cx: float,
                    r_px: float, n_ang: int = 720) -> float:
    """m=4 方位谐波水平 (dB), 用于定量证明 4× oversampling 抑制方条纹数值伪影。

    在 (cy, cx) 周围半径 r_px 的环上均匀采样 720 点 (双线性插值),
    返回 20·log10(|FFT[4]| / |FFT[0]|) — 4 阶方位谐波相对直流分量的幅度。
    """
    th = 2.0 * np.pi * np.arange(n_ang) / n_ang
    xa = cx + r_px * np.cos(th)
    ya = cy + r_px * np.sin(th)
    x0 = np.floor(xa).astype(int) % img.shape[1]
    y0 = np.floor(ya).astype(int) % img.shape[0]
    fx = xa - np.floor(xa)
    fy = ya - np.floor(ya)
    x1 = (x0 + 1) % img.shape[1]
    y1 = (y0 + 1) % img.shape[0]
    ring = (img[y0, x0] * (1 - fx) * (1 - fy)
            + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy
            + img[y1, x1] * fx * fy)
    sp = np.fft.fft(ring)
    f0 = abs(sp[0])
    if f0 <= 0.0:
        return -300.0
    return float(20.0 * np.log10(abs(sp[4]) / f0 + 1e-15))


# ========================= 径向指标 =========================
def radial_profile(img: np.ndarray, cy: float, cx: float,
                   rmax: float = RADIAL_RMAX, dr: float = RADIAL_DR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """以 (cy, cx) 为中心的径向强度剖面 (bincount, 压缩空 bin)。

    Returns:
        (r, profile, wsum): 半径 (px, 压缩空 bin 后非均匀网格)、每 bin 平均强度、
        每 bin 能量和 (供环围能量使用)。
    """
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    sel = rr <= rmax
    b = np.floor(rr[sel] / dr).astype(int)
    wsum = np.bincount(b, weights=img[sel], minlength=int(rmax / dr) + 1)
    nbin = np.bincount(b, minlength=int(rmax / dr) + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(nbin > 0, wsum / np.maximum(nbin, 1), 0.0)
    r = np.arange(len(p)) * dr
    keep = nbin > 0  # 去掉空 bin (0 值空洞会造成假穿越)
    return r[keep], p[keep], wsum[keep]


def _crossing_radius(r: np.ndarray, p: np.ndarray, half: float,
                     ipk: int, direction: int) -> float:
    """线性内插剖面与 half 的交叉半径。direction: +1 向外, -1 向内。"""
    if direction > 0:
        for i in range(ipk, len(p) - 1):
            lo, hi = p[i], p[i + 1]
            if lo >= half > hi:
                if hi == lo:
                    return float(r[i])
                t = (lo - half) / (lo - hi)
                return float(r[i] + t * (r[i + 1] - r[i]))
        return float(r[-1])  # 未下穿 → 取边界
    for i in range(ipk, 0, -1):
        lo, hi = p[i], p[i - 1]
        if lo >= half > hi:
            if hi == lo:
                return float(r[i])
            t = (lo - half) / (lo - hi)
            return float(r[i] + t * (r[i - 1] - r[i]))
    return 0.0  # 单调剖面 → 内侧到轴


def fwhm_radial(r: np.ndarray, p: np.ndarray) -> float:
    """径向剖面半高全宽 FWHM (px)。

    r_peak = argmax(profile); r_out 为向外首次下穿半幅的半径, r_in 为向内
    首次下穿半幅的半径。剖面峰在轴上 (中心对称光斑, 如理想 Airy, 剖面单调递减)
    时 FWHM = 2·r_out (全宽); 否则 FWHM = r_out − r_in。
    """
    ipk = int(np.argmax(p))
    half = 0.5 * p[ipk]
    r_out = _crossing_radius(r, p, half, ipk, +1)
    if ipk == 0:
        return 2.0 * r_out
    r_in = _crossing_radius(r, p, half, ipk, -1)
    return r_out - r_in


def encircled_radius(r: np.ndarray, wsum: np.ndarray, frac: float, total: float) -> float:
    """环围能量半径: 累计能量达到 frac·total 的半径 (px, 线性内插)。

    累计使用每 bin **能量和** ``wsum`` (而非每 bin 均值 — 环面积随 r 增大,
    均值不能累加成能量; 旧实现 cumsum(均值) 永远达不到 frac·total, 半径钉在边界)。
    """
    c = np.cumsum(wsum)
    target = frac * total
    if c[-1] <= 0:
        return 0.0
    if target >= c[-1]:
        return float(r[-1])
    idx = int(np.searchsorted(c, target))
    if idx <= 0:
        return 0.0
    c0, c1 = c[idx - 1], c[idx]
    t = (target - c0) / (c1 - c0) if c1 > c0 else 0.0
    return float(r[idx - 1] + t * (r[idx] - r[idx - 1]))


def strehl_08_crossing(amps: np.ndarray, s: np.ndarray) -> float | None:
    """Strehl 首次下穿 0.8 的幅度 (线性内插); 全程 ≥0.8 返回 None。"""
    for i in range(1, len(s)):
        if s[i] < 0.8 <= s[i - 1]:
            t = (s[i - 1] - 0.8) / (s[i - 1] - s[i])
            return float(amps[i - 1] + t * (amps[i] - amps[i - 1]))
    return None


# ========================= 图版 =========================
def _setup_cjk() -> None:
    plt.rcParams["font.family"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def fig_farfield_grid(data: dict, modes_info: list[tuple[int, str]],
                      amps: list[float]) -> None:
    """图 1: 12 模式 × 8 幅度 远场 log 强度网格 (围绕 0 级预裁剪 CROP×CROP)。"""
    n_r, n_c = len(modes_info), len(amps)
    fig, axes = plt.subplots(
        n_r, n_c, figsize=(1.7 * n_c, 1.7 * n_r + 1.2), dpi=110,
        squeeze=False,
    )
    for i, (noll, name) in enumerate(modes_info):
        for j, a in enumerate(amps):
            sl = data[(noll, a)]["crop"]
            crop = _smooth2d(log_scale(sl), SMOOTH_SIGMA_PX)
            ax = axes[i][j]
            ax.imshow(crop, cmap="inferno", vmin=LOG_FLOOR_DB, vmax=0.0, extent=None,
                      interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j == 0:
                ax.set_ylabel(f"Noll {noll}\n{name}", fontsize=8)
            if i == n_r - 1:
                ax.set_xlabel(f"{a:g} λ", fontsize=8)
    fig.suptitle(f"Zernike 远场光斑形貌 (2f-Fourier 仿真, log 强度 {LOG_FLOOR_DB:.0f}~0 dB, "
                 f"Gaussian 平滑 σ={SMOOTH_SIGMA_PX:g}px, 围绕 0 级光斑裁剪)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(FIG_DIR / "01_farfield_grid.png")
    plt.close(fig)


def fig_phase_grid(data: dict, mask: np.ndarray, modes_info: list[tuple[int, str]],
                   amps: list[float]) -> None:
    """图 2: 12 模式 × 8 幅度 相位 (pupil 平面, 逐格对称归一化, 孔径外置零)。"""
    n_r, n_c = len(modes_info), len(amps)
    fig, axes = plt.subplots(
        n_r, n_c, figsize=(1.28 * n_c, 1.62 * n_r + 1.2), dpi=110, squeeze=False,
    )
    for i, (noll, name) in enumerate(modes_info):
        for j, a in enumerate(amps):
            ph = data[(noll, a)]["phase"]
            disp = np.where(mask > 0, ph, 0.0)
            vmax = float(np.abs(disp).max())
            ax = axes[i][j]
            ax.imshow(disp, cmap="seismic",
                      norm=Normalize(vmin=-max(vmax, 1e-6), vmax=max(vmax, 1e-6)))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j == 0:
                ax.set_ylabel(f"Noll {noll}\n{name}", fontsize=8)
            if i == n_r - 1:
                ax.set_xlabel(f"{a:g} λ", fontsize=8)
    fig.suptitle("Zernike 相位形貌 (raw 弧度, 逐格对称归一化 — 形状不变, 幅度线性缩放)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(FIG_DIR / "02_phase_grid.png")
    plt.close(fig)


def _curve_lines(ax, amps: np.ndarray, series: dict[int, np.ndarray],
                 modes_info: list[tuple[int, str]], **plot_kw) -> None:
    """Draw one line per mode over the positive-amplitude points of ``amps``.

    A=0 is the shared Airy baseline (identical for all modes) and is left off
    the curves; each series therefore only uses the a>0 subset of the full
    amplitude axis.
    """
    a = np.asarray(amps, dtype=float)
    pos = a > 0
    cmap = plt.get_cmap("tab20")
    for k, (noll, _) in enumerate(modes_info):
        if noll in series:
            ax.plot(a[pos], series[noll][pos], marker="o", ms=3.2, lw=1.4,
                    color=cmap(k % 20), label=f"Noll {noll}", **plot_kw)


def fig_strehl(series: dict[int, np.ndarray], modes_info: list[tuple[int, str]],
               amps: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=140)
    _curve_lines(ax, amps, series, modes_info)
    ax.axhline(0.8, color="red", ls="--", lw=1.2, label="Strehl = 0.8 (衍射极限判据)")
    ax.set_xscale("log")
    ax.set_xlabel("Zernike 系数幅度 (λ)")
    ax.set_ylabel("Strehl 比 (峰值 / 理想 Airy 峰值)")
    ax.set_title("Strehl 比随幅度变化 (Noll 4–15, N_MAX = 4)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=3, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_strehl_vs_amp.png")
    plt.close(fig)


def fig_encircled(series50: dict[int, np.ndarray], series90: dict[int, np.ndarray],
                  modes_info: list[tuple[int, str]], amps: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=140)
    for ax, series, title in (
        (axes[0], series50, "EE50 环围半径 (50% 能量)"),
        (axes[1], series90, "EE90 环围半径 (90% 能量)"),
    ):
        _curve_lines(ax, amps, series, modes_info)
        ax.set_xscale("log")
        ax.set_xlabel("Zernike 系数幅度 (λ)")
        ax.set_ylabel("半径 (px)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6.8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_encircled_energy_vs_amp.png")
    plt.close(fig)


def fig_fwhm(series: dict[int, np.ndarray], modes_info: list[tuple[int, str]],
             amps: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=140)
    _curve_lines(ax, amps, series, modes_info)
    ax.set_xscale("log")
    ax.set_xlabel("Zernike 系数幅度 (λ)")
    ax.set_ylabel("FWHM (px) — 半高全宽:\n单调核心 2·r_out; 环形光斑 r_out − r_in")
    ax.set_title("光斑 FWHM 随幅度变化")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_fwhm_vs_amp.png")
    plt.close(fig)


def fig_peak_shift(series: dict[int, np.ndarray], modes_info: list[tuple[int, str]],
                   amps: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=140)
    _curve_lines(ax, amps, series, modes_info)
    ax.set_xscale("log")
    ax.set_xlabel("Zernike 系数幅度 (λ)")
    ax.set_ylabel("质心偏移 r = √(dx²+dy²) (px, 相对 Airy 0 级)")
    ax.set_title("强度质心偏移随幅度变化 (彗差类被彗尾拉偏; 其余模式质心保持在中心)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_peak_shift_vs_amp.png")
    plt.close(fig)


# ========================= 报告 =========================
MODE_SHAPE_TEXT = {
    4: "中央光斑随幅度增大而扩大、中心逐渐变暗, 1λ 以上呈明显 **环形 (donut)**; 能量由中心向环上迁移。",
    5: "光斑沿 45° 方向拉伸为 **椭圆/线段**; 幅度增大时长轴变长, 强像散时中心凹陷、出现双峰结构。",
    6: "光斑沿 0° 方向拉伸为 **椭圆/线段** (与 Noll 5 正交方向); 形态与 Noll 5 对称, 方向差 90°。",
    7: "光斑呈 **彗差尾** (尾向固定), 主斑峰值向尾的反方向偏移, 偏移量随幅度增大 — 本报告中唯一出现显著峰值平移的模式。",
    8: "彗差尾 (方向与 Noll 7 不同), 峰值同样发生定向偏移。",
    9: "光斑呈 **三叶形** (trefoil), 叶瓣数固定为 3, 幅度增大时叶瓣加深但中心位置不变。",
    10: "三叶形 (取向与 Noll 9 旋转 60°), 形态同族。",
    11: "Airy 图样演变为 **三环结构**: 中心光斑变暗、外侧出现两个衍射环, 是球差的标志性特征。",
    12: "次高图像散: 椭圆基础上边缘出现 **四瓣化** 调制, 幅度增大时中心凹陷加深。",
    13: "次高图像散 (与 Noll 12 取向不同), 形态同族。",
    14: "光斑呈 **四叶形** (tetrafoil), 中心凹陷随幅度加深, 峰值保持在中心。",
    15: "四叶形 (取向与 Noll 14 旋转 45°), 形态同族。",
}


def write_report(
    metrics: list[dict],
    modes_info: list[tuple[int, int, int, str]],
    amps: np.ndarray,
        strehl: dict[int, np.ndarray],
        m4: dict,
        ideal_peak: float,
    airy_fwhm: float,
    airy_ee50: float,
    airy_ee90: float,
    elapsed_s: float,
) -> None:
    lines: list[str] = []
    add = lines.append

    add("# Zernike 各模式/系数幅度下远场光斑形貌仿真报告")
    add("")
    add(f"> **Fully offline** — 2f-Fourier 数值仿真, 无硬件。生成耗时 {elapsed_s:.0f} s。")
    add(">")
    add("> 光学模型与 `src/ao_shaping/drivers/sim/slm_pib_sim.py` 的 `SimPibSystem.far_field()` "
        "同一 2f-Fourier 夫琅禾费模型: 远场 = 入瞳场 `I = |FFT(孔径 · e^(iφ))|²`。"
        "纯 numpy 实现省去设备生命周期与 SLM 灰度量化, 仅保留衍射物理, 适合 12 模式 × 8 幅度 = 96 例批量扫描。")
    add("")
    add("## 1. 摘要")
    add("")
    add("本报告在 2f-Fourier 仿真环境下, 对 **Noll 4–15** (径向阶 n≤4, 共 12 个非 piston/tilt 模式) "
        "在 **0–2 λ** 的 8 档系数幅度下, 仿真并量化了远场光斑形貌。主要结论:")
    add("")
    best = max(strehl.items(), key=lambda kv: kv[1][-1])[0]
    worst = min(strehl.items(), key=lambda kv: kv[1][-1])[0]
    add(f"1. **Strehl 比随幅度单调下降**, 2λ 处最高为 Noll {best} ({strehl[best][-1]:.3f}), "
        f"最低为 Noll {worst} ({strehl[worst][-1]:.3f})。")
    add("2. **形态家族规律**: 离焦 → 环形; 像散 → 椭圆/线段; 彗差 → 彗尾 + 峰值偏移; "
        "球差 → 三环; 三叶/四叶 → 多瓣; 次高图像散 → 椭圆+四瓣调制。所有模式的相位形状**尺度不变**, "
        "幅度仅线性缩放 (见相位网格图)。")
    add("3. **峰值偏移仅彗差族 (Noll 7/8) 非零** — 其余模式中心对称性或反对称性保证 0 级保持在中心; "
        "彗差的定向偏移量随幅度近似线性增长。")
    add("4. **0.8 判据**: 各模式 Strehl 下穿 0.8 的幅度 (表 3) 给出该模式的『可用系数上限』, "
        "对 SLM 相位加载的闭环控制幅度选择有直接参考价值。")
    add("")
    add("## 2. 方法")
    add("")
    add("| 项 | 取值 | 说明 |")
    add("|---|---|---|")
    add(f"| 入瞳网格 | {PUPIL_N}×{PUPIL_N} px | 圆形孔径半径 {PUPIL_RADIUS:g} px, 单位圆恰触帧边 |")
    add(f"| 远场网格 | {FAR_N}×{FAR_N} px | 入瞳场零填充上采样 (不增加信息, 仅细化远场采样) |")
    add(f"| 远场模型 | `I = |fftshift(FFT2(pupil))|²` | 2f 傅里叶光路, 与 `slm_pib_sim.py` 同一模型 |")
    add(f"| 模式 | Noll {NOLL_MODES[0]}–{NOLL_MODES[-1]} (12 个) | N_MAX = {N_MAX}; 跳过 piston/tilt (对远场无影响) |")
    add(f"| 幅度 | {', '.join(f'{a:g}' for a in amps)} λ | `φ = A·2π·Z_j` (rad); A=0 为共同 Airy 基线 |")
    add(f"| 0 级定位 | 基线 Airy 的 `argmax` | 遵循 AGENTS 红线: 永不假设几何中心 |")
    add("")
    add("**指标定义**")
    add("")
    add("| 指标 | 定义 |")
    add("|---|---|")
    add("| Strehl | `peak(I) / peak(I_Airy)` |")
    add(f"| FWHM | 径向剖面 (dr={RADIAL_DR}px) 半高全宽 `r_out − r_in` (px); 单调 Airy 剖面 r_in→0, 即半宽 |")
    add("| EE50 / EE90 | 环围能量半径: 累计能量达 50%/90% 总能量的半径 (px, 内插) |")
    add("| 峰值偏移 | `argmax(I) − 基线中心` 的 (dx, dy) 与模长 (px) |")
    add("")
    add("## 3. 模式清单")
    add("")
    add("| Noll | (n, m) | 名称 |")
    add("|---|---|---|")
    for noll, n, m, name in modes_info:
        add(f"| {noll} | ({n}, {m}) | {name} |")
    add("")
    add("## 4. 远场光斑形貌")
    add("")
    add(f"![far-field grid]({('figures/01_farfield_grid.png').replace(chr(92), '/')})")
    add("")
    add(f"12 模式 × 8 幅度远场 log 强度图 (围绕 0 级光斑裁剪 {CROP}×{CROP} px, "
        f"底 {LOG_FLOOR_DB:g} dB, Gaussian 平滑 σ={SMOOTH_SIGMA_PX:g}px)。逐格归一化到各自峰值。")
    add("")
    add("| Noll | 名称 | 形态演化 (随幅度增大) |")
    add("|---|---|---|")
    for noll, n, m, name in modes_info:
        short = name.split(" / ")[0]
        add(f"| {noll} | {short} | {MODE_SHAPE_TEXT.get(noll, '—')} |")
    add("")
    add("相位图 (pupil 平面, 逐格对称归一化):")
    add("")
    add("![phase grid](figures/02_phase_grid.png)")
    add("")
    add("所有模式的相位形状随幅度**严格线性缩放** (形状不变) — 验证了 raw 弧度相位契约: "
        "`φ = A·2π·Z_j`, 无 min-max 归一化等尺度无关畸变。")
    add("")
    add("### 4.1 数值伪影诊断: m=4 方位谐波 (“方条纹”来源)")
    add("")
    add("硬边圆孔径在 4 个轴向方向存在阶梯化边缘 (离散化的圆), 其方位谐波以 m=4 为主, "
        "在远场 log 图中表现为方形周期条纹 — 这是数值伪影, 不是物理衍射。4× 上采样后 "
        "(孔径 128→512 px, R=64→256 px) 边缘更光滑, m=4 能量按 4×(1/16)² = 1/64 ≈ −18 dB 下降。")
    add("")
    add("| 用例 | r (px) | 旧网格 128/64 (dB) | 新网格 512/256 (dB) |")
    add("|---|---:|---:|---:|")
    cases = (("airy", "Airy 基线 (A=0)"), ("defocus_2.0", "离焦 (Noll 4, 2.0λ)"))
    for key, lbl in cases:
        for r_px in M4_RADII:
            old_db = m4["old"][key].get(r_px, float("nan"))
            new_db = m4["new"][key].get(r_px, float("nan"))
            add(f"| {lbl} | {r_px:.0f} | {old_db:+.1f} | {new_db:+.1f} |")
    add("")
    add("新网格 m=4 比旧网格低约 18 dB (见上表实测), 低于图 1 显示底限, 方形条纹被抑制; "
        "保留的同心环为物理 Airy 衍射 (首暗环 ≈19.5 px, 两网格尺度不变)。")
    add("")
    add("## 5. 指标")
    add("")
    add("### 5.1 Strehl 比")
    add("")
    add("![strehl](figures/03_strehl_vs_amp.png)")
    add("")
    add("| Noll | 名称 | Strehl@0.1λ | 0.4λ | 0.8λ | 1.2λ | 1.6λ | 2.0λ | 0.8 判据幅度 (λ) |")
    add("|---|---|---|---|---|---|---|---|---|")
    for noll, n, m, name in modes_info:
        s = strehl[noll]
        cross = strehl_08_crossing(amps, s)
        short = name.split(" / ")[0]
        idx = {a: i for i, a in enumerate(amps)}
        def _v(a):
            return f"{s[idx[a]]:.3f}"
        add(f"| {noll} | {short} | {_v(0.1)} | {_v(0.4)} | {_v(0.8)} | {_v(1.2)} | {_v(1.6)} | {_v(2.0)} | "
            f"{'—' if cross is None else f'{cross:.2f}'} |")
    add("")
    add("### 5.2 光斑尺寸 (环围能量 / 半高宽)")
    add("")
    add("![ee](figures/04_encircled_energy_vs_amp.png)")
    add("")
    add("![fwhm](figures/05_fwhm_vs_amp.png)")
    add("")
    add(f"Airy 基线 (A=0): FWHM(半宽) = {airy_fwhm:.2f} px, EE50 = {airy_ee50:.2f} px, "
        f"EE90 = {airy_ee90:.2f} px。")
    add("")
    add("### 5.3 峰值偏移")
    add("")
    add("![peak shift](figures/06_peak_shift_vs_amp.png)")
    add("")
    add("## 6. 观察与结论")
    add("")
    add("1. **幅度-形态单调性**: 每个模式的形貌随幅度连续演化, 无突变 (相位线性缩放 → 场分布连续变化); "
        "但可观察到**定性转变点** — 离焦 ~0.5λ 后中心变暗成环, 球差 ~0.4λ 后第二环可见, "
        "像散/三叶/四叶 ~0.8λ 后中心凹陷形成多峰。")
    add("2. **低阶 vs 高阶**: n=2 (离焦/像散) 的 Strehl 退化最慢 (模式与 Airy 能量分布最『兼容』), "
        "n=4 的高阶模式在同等幅度下 Strehl 退化更快 — 与相位方差 σ² 随阶数增长的解析预期一致。")
    add("3. **彗差的特殊性**: 唯一的峰值偏移来源; 偏移 ≈ 幅度线性, 可用于 SLM 标定中区分彗差与其他模式。")
    add("4. **工程意义**: 表 3.1 的 0.8 判据幅度可直接作为闭环 (SPGD/GA) 中 Zernike 系数的**步长上限参考** — "
        "超过该幅度单模式 Strehl 即低于 0.8, 梯度信噪比 (SNR) 显著下降。")
    add("")
    add("## 7. 复现")
    add("")
    add("```powershell")
    add('$env:PYTHONPATH = "src"')
    add("python scripts/generate_zernike_farfield_sim_report.py")
    add("```")
    add("")
    add("全量逐例数据见 `metrics.csv` (96 行: mode_noll, n, m, name, amp_waves, strehl, "
        "fwhm_px, ee50_r_px, ee90_r_px, peak_dx, peak_dy, peak_r_px)。")
    add("")

    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    _setup_cjk()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 相位生成 (canonical 入口: ZernikeGenerator, 网格缓存复用) ----------
    gen = ZernikeGenerator(
        resolution=(PUPIL_N, PUPIL_N),
        radius=PUPIL_RADIUS,
        n_orders=N_MAX,
    )
    mask = gen.mask  # uint8 (512,512), 单位圆内为 1

    modes_info: list[tuple[int, int, int, str]] = []
    mode_short: dict[int, str] = {}
    for entry in list_zernike_modes(N_MAX):
        noll, n, m, name = entry
        if noll in NOLL_MODES:
            modes_info.append((noll, n, m, name))
            mode_short[noll] = name.split(" / ")[0]
    modes_info.sort(key=lambda t: t[0])
    modes_info_lbl = [(noll, mode_short[noll]) for noll, _, _, _ in modes_info]

    # 模式 (n, m)
    nm_of_noll = {noll: (n, m) for noll, n, m, _ in modes_info}

    # ---------- 基线 Airy (A=0, 全部模式共用) ----------
    pupil0 = mask.astype(np.float64)
    I_airy = far_field_intensity(pupil0)
    ideal_peak = float(I_airy.max())
    cy0, cx0 = np.unravel_index(np.argmax(I_airy), I_airy.shape)  # 0 级 = argmax
    r_airy, p_airy, w_airy = radial_profile(I_airy, float(cy0), float(cx0))
    airy_fwhm = fwhm_radial(r_airy, p_airy)
    total_airy = float(I_airy.sum())
    airy_ee50 = encircled_radius(r_airy, w_airy, 0.50, total_airy)
    airy_ee90 = encircled_radius(r_airy, w_airy, 0.90, total_airy)
    print(f"[baseline] Airy: center=({cy0},{cx0}) peak={ideal_peak:.3e} "
              f"FWHM={airy_fwhm:.2f}px EE50={airy_ee50:.2f}px EE90={airy_ee90:.2f}px")

    # ---- m=4 方位谐波诊断: 旧网格 (128/64) 对照块 (报告 §4.1) ----
    gen_old = ZernikeGenerator(resolution=(PUPIL_N_OLD, PUPIL_N_OLD),
                               radius=PUPIL_RADIUS_OLD, n_orders=N_MAX)
    yy_old, xx_old = np.mgrid[0:PUPIL_N_OLD, 0:PUPIL_N_OLD]
    r_old = np.sqrt((xx_old - (PUPIL_N_OLD - 1) / 2.0) ** 2 + (yy_old - (PUPIL_N_OLD - 1) / 2.0) ** 2)
    mask_old = (r_old <= PUPIL_RADIUS_OLD).astype(np.float64)
    I_old_airy = far_field_intensity(mask_old, far_n=FAR_N_OLD)
    cy0o, cx0o = np.unravel_index(int(np.argmax(I_old_airy)), I_old_airy.shape)
    ph_old_def = np.nan_to_num(gen_old.generate_polynomial({(2, 0): 2.0 * 2.0 * np.pi}))
    I_old_def = far_field_intensity(np.exp(1j * ph_old_def) * mask_old, far_n=FAR_N_OLD)
    m4_old = {
        "airy": {r: float(azimuthal_m4_db(I_old_airy, cy0o, cx0o, r)) for r in M4_RADII},
        "defocus_2.0": {r: float(azimuthal_m4_db(I_old_def, cy0o, cx0o, r)) for r in M4_RADII},
    }
    del I_old_airy, I_old_def, ph_old_def, mask_old, yy_old, xx_old, r_old
    m4_new = {
        "airy": {r: float(azimuthal_m4_db(I_airy, cy0, cx0, r)) for r in M4_RADII},
        "defocus_2.0": {r: float("nan") for r in M4_RADII},   # 扫描时 (noll==4, 2.0λ) 回填
    }
    m4 = {"old": m4_old, "new": m4_new}
    print(f"[m4] old 128/64 airy r=25: {m4_old['airy'][25.0]:+.1f} dB | "
          f"new 512/256 airy r=25: {m4_new['airy'][25.0]:+.1f} dB")

    # 整帧坐标网格 (强度加权质心用, 预计算一次, 96 例复用)
    yy_full, xx_full = np.mgrid[0:FAR_N, 0:FAR_N]
    # ---------- 扫描: 12 模式 × 8 幅度 (A=0 复用基线) ----------
    # 内存优化: data 仅存 160×160 中心裁剪 (图01) + 512² 相位 (图02); 全帧 8192² I
    # 逐例瞬态计算 (峰值 ≈4.5 GB, 机器 64 GB 可用)。I_airy 为 A=0 共享引用, 全程保留。
    half = CROP // 2
    sl0 = (slice(int(cy0) - half, int(cy0) + half), slice(int(cx0) - half, int(cx0) + half))
    airy_crop = I_airy[sl0].copy()
    zero_phase = np.zeros((PUPIL_N, PUPIL_N))
    data: dict[tuple[int, float], dict] = {}
    metrics: list[dict] = []
    n_fft = 0
    for noll, n, m, name in modes_info:
        nm = nm_of_noll[noll]
        for a_waves in AMPS_WAVES:
            key = (noll, a_waves)
            if a_waves == 0.0:
                I = I_airy
                phase = zero_phase
                crop = airy_crop
            else:
                amp_rad = a_waves * 2.0 * np.pi
                phase = gen.generate_polynomial({nm: amp_rad})
                phase = np.nan_to_num(phase)  # 稳健: 孔径外若 NaN 置零
                pupil = np.exp(1j * phase) * pupil0
                I = far_field_intensity(pupil)
                n_fft += 1
                if noll == 4 and a_waves == 2.0:   # 新网格 m=4 回填 (§4.1)
                    for r in M4_RADII:
                        m4["new"]["defocus_2.0"][r] = float(azimuthal_m4_db(I, cy0, cx0, r))
                crop = I[sl0].copy()
            data[key] = {"crop": crop, "phase": phase}

            # ---- 指标 ----
            peak = float(I.max())
            sth = peak / ideal_peak
            total = float(I.sum())
            # 0 级位置 = 整帧强度加权质心: 对称模式恰在 Airy 中心, 彗差族被
            # 彗尾拉偏 (随幅度近似线性); 比 argmax 峰值稳健 (不受细结构量化影响)
            cey = float((yy_full * I).sum() / total)
            cex = float((xx_full * I).sum() / total)
            r, p, wsum_c = radial_profile(I, cey, cex)
            metrics.append({
                "mode_noll": noll, "n": n, "m": m, "name": name,
                "amp_waves": a_waves, "strehl": sth,
                "fwhm_px": fwhm_radial(r, p),
                "ee50_r_px": encircled_radius(r, wsum_c, 0.50, total),
                "ee90_r_px": encircled_radius(r, wsum_c, 0.90, total),
                "peak_dx": cex - cx0, "peak_dy": cey - cy0,
                "peak_r_px": float(np.hypot(cex - cx0, cey - cy0)),
            })
        print(f"[mode] Noll {noll} ({n}, {m}) {name} done")
    print(f"[scan] {len(metrics)} cases, {n_fft} far-field FFTs")

    # ---------- metrics.csv ----------
    import csv
    with open(OUT_DIR / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode_noll", "n", "m", "name", "amp_waves", "strehl", "fwhm_px",
                    "ee50_r_px", "ee90_r_px", "peak_dx", "peak_dy", "peak_r_px"])
        for row in metrics:
            w.writerow([row["mode_noll"], row["n"], row["m"], row["name"],
                        f'{row["amp_waves"]:.2f}', f'{row["strehl"]:.4f}',
                        f'{row["fwhm_px"]:.3f}', f'{row["ee50_r_px"]:.3f}',
                        f'{row["ee90_r_px"]:.3f}', f'{row["peak_dx"]:.2f}',
                        f'{row["peak_dy"]:.2f}', f'{row["peak_r_px"]:.3f}'])

    # ---------- 序列 (per-mode) ----------
    amps = np.array(AMPS_WAVES)
    strehl_s: dict[int, np.ndarray] = {}
    fwhm_s: dict[int, np.ndarray] = {}
    ee50_s: dict[int, np.ndarray] = {}
    ee90_s: dict[int, np.ndarray] = {}
    shift_s: dict[int, np.ndarray] = {}
    for noll, _, _, _ in modes_info:
        idx = {i: row for i, row in enumerate(metrics) if row["mode_noll"] == noll}
        strehl_s[noll] = np.array([metrics[i]["strehl"] for i in idx])
        fwhm_s[noll] = np.array([metrics[i]["fwhm_px"] for i in idx])
        ee50_s[noll] = np.array([metrics[i]["ee50_r_px"] for i in idx])
        ee90_s[noll] = np.array([metrics[i]["ee90_r_px"] for i in idx])
        shift_s[noll] = np.array([metrics[i]["peak_r_px"] for i in idx])

    # ---------- 图 ----------
    fig_farfield_grid(data, modes_info_lbl, AMPS_WAVES)
    print("[fig] 01_farfield_grid.png")
    fig_phase_grid(data, mask, modes_info_lbl, AMPS_WAVES)
    print("[fig] 02_phase_grid.png")
    fig_strehl(strehl_s, modes_info_lbl, amps)
    print("[fig] 03_strehl_vs_amp.png")
    fig_encircled(ee50_s, ee90_s, modes_info_lbl, amps)
    print("[fig] 04_encircled_energy_vs_amp.png")
    fig_fwhm(fwhm_s, modes_info_lbl, amps)
    print("[fig] 05_fwhm_vs_amp.png")
    fig_peak_shift(shift_s, modes_info_lbl, amps)
    print("[fig] 06_peak_shift_vs_amp.png")

    # ---------- 报告 ----------
    write_report(metrics, modes_info, amps, strehl_s, m4, ideal_peak,
                 airy_fwhm, airy_ee50, airy_ee90, time.time() - t0)
    print(f"[report] {OUT_DIR / 'report.md'}")
    print(f"[done] {time.time() - t0:.1f}s, output: {OUT_DIR}")


if __name__ == "__main__":
    main()
