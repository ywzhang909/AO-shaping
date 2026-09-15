"""SLM Zernike 工具集 — 共享常量与测量原语.

被 `slm_shift_calib` / `slm_zernike_response` / `slm_zernike_correction` /
`slm_wfs_reference` / `slm_zernike_report` 共同复用, 避免重复实现。

⚠️ 核心索引约定 (官方手册 + 硬件实测双重确认, **非标准 Noll 1976**)
------------------------------------------------------------------
`ThorlabWFS.get_zernike()` 返回 67 长数组, DLL 填 `coeff[1..66]` (index 0 未用),
排序为**顺序 m 枚举** (`m = -n..+n`):

    [1](0,0) [2](1,-1) [3](1,1) [4](2,-2) [5](2,0)defocus [6](2,2)
    [7](3,-3) [8](3,-1) [9](3,1)coma [10](3,3)
    [11](4,-4) [12](4,-2) [13](4,0)spherical [14](4,2) [15](4,4)

- 手册佐证: `roCMm` "derived from Zernike coefficient **Z[5]**" (球面波 RoC ← defocus);
  `arrayZernikeUm` "indices [1..66] are used instead of [0..65]"。
- 实测佐证: 加载 (2,0)→[5], (2,2)→[6], (3,1)→[9], (4,0)→[13]; 响应矩阵 14/14 强对角。

标准 Noll 中 defocus=4、spherical=11 ⇒ 驱动 docstring 的 "Noll 1976 约定" 不准确。
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.slm import Santec
from ao_shaping.drivers.wfs import ThorlabWFS
from ao_shaping.utils.pattern_helper import PatternHelper

PANEL_H, PANEL_W = 1200, 1920
MAX_EXPOSURE_MS = 7.0
DEFAULT_EXPOSURE_MS = 4.0
SETTLE_REDUNDANCY_S = 0.1          # 像素翻转估算之外额外等待
WFS_ZERNIKE_ORDER = 10             # orders 有效值 0=auto 或 2..10 (10 → 66 项); 15 非法!

# DLL 顺序 m 枚举: 1-based index -> (n, m)
DLL_ZERNIKE_ORDER: list[tuple[int, int]] = [
    (0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2),
    (3, -3), (3, -1), (3, 1), (3, 3),
    (4, -4), (4, -2), (4, 0), (4, 2), (4, 4),
    (5, -5), (5, -3), (5, -1), (5, 1), (5, 3), (5, 5),
]

DLL_ZERNIKE_NAMES: dict[int, str] = {
    1: "(0,0)piston", 2: "(1,-1)tiltA", 3: "(1,1)tiltB", 4: "(2,-2)astig",
    5: "(2,0)defocus", 6: "(2,2)astig", 7: "(3,-3)trefoil", 8: "(3,-1)coma",
    9: "(3,1)coma", 10: "(3,3)trefoil", 11: "(4,-4)quadrafoil",
    12: "(4,-2)2nd-astig", 13: "(4,0)spherical", 14: "(4,2)2nd-astig",
    15: "(4,4)quadrafoil",
}


def nm_of(index: int) -> tuple[int, int]:
    """DLL 1-based 索引 → (n, m)."""
    return DLL_ZERNIKE_ORDER[index - 1]


def index_of(nm: tuple[int, int]) -> int | None:
    """(n, m) → DLL 1-based 索引 (未收录返回 None)."""
    try:
        return DLL_ZERNIKE_ORDER.index(nm) + 1
    except ValueError:
        return None


def name_of(index: int) -> str:
    return DLL_ZERNIKE_NAMES.get(index, f"idx{index}")


def n_modes_upto(n_max: int) -> int:
    """含 piston 的模式总数 (n_max=4 → 15)."""
    return (n_max + 1) * (n_max + 2) // 2


# ─────────────────────────── 相位生成 / 下发 ───────────────────────────

def make_phase(ph: PatternHelper, coefficients: dict[tuple[int, int], float],
               radius: float, n_max: int = 4) -> np.ndarray:
    """Zernike 系数 (rad) → 弧度相位图. 走 PatternHelper (与 GUI 同链路)."""
    return ph.generate_zernike_polynomial(
        coefficients=coefficients, radius=float(radius), n_max=max(n_max, 4))


def flat_gray() -> np.ndarray:
    """纯平相位灰度 (gray=0). 扁平相位必须直接发 uint16, 严禁走 create_phase_from_array."""
    return np.full((PANEL_H, PANEL_W), 0, dtype=np.uint16)


def show_phase(slm: Santec, phase_rad: np.ndarray,
               extra_sleep: float = SETTLE_REDUNDANCY_S) -> None:
    """下发相位: ``wait_time_s=None`` 自动按像素翻转估算等待 + ``extra_sleep`` 冗余.

    估算见 ``Santec._estimate_pixel_flip_wait``:
    ``wait = max_gray_change / _max_gray × MAX_PIXEL_FLIP_TIME_S`` (上限 200 ms)。
    """
    slm.display_phase(phase_rad, wait_time_s=None)
    if extra_sleep > 0:
        time.sleep(extra_sleep)


# ─────────────────────────── 测量 ───────────────────────────

def measure_zernike(wfs: ThorlabWFS, n_avg: int = 3,
                    order: int = WFS_ZERNIKE_ORDER) -> np.ndarray | None:
    """多帧中位数聚合 WFS Zernike 读数 (µm, 67 长, index 0 未用).

    中位数聚合抑制单帧光子噪声/斑点抖动 (小幅度端尤其敏感)。
    """
    rows: list[np.ndarray] = []
    for _ in range(n_avg):
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        try:
            z = wfs.get_zernike(zernike_order=order)
        except Exception as e:
            logger.debug("get_zernike 失败: {}", e)
            continue
        if z is None or not np.isfinite(z).all():
            continue
        rows.append(np.asarray(z, dtype=float))
    return None if not rows else np.median(np.array(rows), axis=0)


def measure_wavefront(wfs: ThorlabWFS, n_avg: int = 3) -> tuple[np.ndarray, dict] | None:
    """多帧中位数波前 + 平均 stats (keys: rms/diff/mean/max/min)."""
    wfs_, stats = [], []
    for _ in range(n_avg):
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        wf, st = wfs.get_wavefront(cancel_tile=False)
        wfs_.append(wf)
        stats.append(st)
    if not wfs_:
        return None
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.array(wfs_), axis=0)
    mean_stats = {k: float(np.mean([s.get(k, np.nan) for s in stats]))
                  for k in ("rms", "diff", "mean", "max", "min")}
    return med, mean_stats


def added_tilt(z: np.ndarray, base: np.ndarray) -> tuple[float, np.ndarray]:
    """相对基线的附加倾斜 (λ): DLL [2],[3] 两分量, 返回 (范数, 分量)."""
    t = z[2:4] / 0.532
    b = base[2:4] / 0.532
    d = t - b
    return float(np.linalg.norm(d)), d


# ─────────────────────────── 自动定标 ───────────────────────────

def diagnose_beam_radius(slm: Santec, wfs: ThorlabWFS, ph: PatternHelper,
                         radii: list[float], amp_rad: float = 20.0,
                         n_avg: int = 3, extra_sleep: float = SETTLE_REDUNDANCY_S,
                         verbose: bool = True) -> tuple[float, list[dict]]:
    """Zernike R 扫描 → WFS defocus 响应最大者 ≈ **光束在 SLM 上的半径**.

    原理: 半径 R 的 defocus 归一化后, 光束 (半径 r_beam) 感受到的相位幅度
    ∝ A·(r_beam/R)² → R 越大响应越小; R < r_beam 时图案被裁切。故响应峰值出现在
    R ≈ r_beam。
    """
    out: list[dict] = []
    for r in radii:
        phase = make_phase(ph, {(2, 0): amp_rad}, float(r), n_max=5)
        show_phase(slm, phase, extra_sleep)
        z = measure_zernike(wfs, n_avg)
        if z is None:
            if verbose:
                print(f"   R={r:5.0f}px  无有效读数")
            continue
        d = float(z[5]) / 0.532           # DLL [5] = (2,0) defocus
        out.append({"radius": float(r), "defocus_lam": d})
        if verbose:
            print(f"   R={r:5.0f}px  defocus[5]={d:+.4f}λ")
    if not out:
        raise RuntimeError("光束半径诊断无有效数据")
    best = max(out, key=lambda r: abs(r["defocus_lam"]))
    if verbose:
        print(f"   → 光束半径 ≈ {best['radius']:.0f}px")
    return float(best["radius"]), out


def calibrate_center_shift(slm: Santec, wfs: ThorlabWFS, ph: PatternHelper,
                           r_beam: float, amp_rad: float = 20.0,
                           coarse_half: int = 300, coarse_step: int = 100,
                           fine_span: int = 40, fine_step: int = 5,
                           n_avg: int = 3,
                           extra_sleep: float = SETTLE_REDUNDANCY_S,
                           limit: int = 500, verbose: bool = True
                           ) -> tuple[int, int, list[dict]]:
    """defocus 零点法测光束中心偏移 → shift_x/shift_y (轴无关判据 ‖Δz_tilt‖).

    图案平移 ``(sx,sy)`` 后光束感受 ``P(b−s+ξ)``; 对 defocus 有梯度 ``∝2D(b−s)``
    → tip/tilt 关于 shift 线性, 零点即 ``s=b``。判据用**相对纯平的附加倾斜**。

    两级扫描: 粗扫 ``±coarse_half`` 步长 ``coarse_step`` → 细扫
    ``±fine_span`` 步长 ``fine_step`` (默认 5px, 提高定位精度)。
    """
    r_use = max(r_beam * 3.0, 450.0)      # 盘需足够大以免平移裁切光束
    phase = make_phase(ph, {(2, 0): amp_rad}, r_use, n_max=5)
    slm.set_shift(0, 0)
    slm.display_data(flat_gray(), wait_time_s=0.5)
    time.sleep(0.3)
    base = measure_zernike(wfs, max(n_avg, 3))
    if base is None:
        raise RuntimeError("基线测量失败")

    scan: list[dict] = []

    def eval_shift(sx: int, sy: int) -> float:
        slm.set_shift(int(np.clip(sx, -limit, limit)), int(np.clip(sy, -limit, limit)))
        show_phase(slm, phase, extra_sleep)
        z = measure_zernike(wfs, n_avg)
        if z is None:
            return float("inf")
        m, _ = added_tilt(z, base)
        return m

    def scan_axis(axis: str, values, other: int) -> float:
        best_v, best_m = float(values[0]), float("inf")
        for v in values:
            sx_ = int(v) if axis == "x" else int(other)
            sy_ = int(v) if axis == "y" else int(other)
            m = eval_shift(sx_, sy_)
            scan.append({"axis": axis, "shift": float(v), "added_norm": float(m)})
            if m < best_m:
                best_m, best_v = m, float(v)
        if verbose:
            print(f"   {axis} 轴最优 {best_v:.0f} (‖Δ‖={best_m:.4f}λ)")
        return best_v

    def fine_around(center: float) -> list[float]:
        return [center + d for d in range(-fine_span, fine_span + 1, fine_step)]

    coarse = list(range(-coarse_half, coarse_half + 1, coarse_step))
    sx0 = scan_axis("x", coarse, 0)
    sx = scan_axis("x", fine_around(sx0), 0)
    sy0 = scan_axis("y", coarse, int(sx))
    sy = scan_axis("y", fine_around(sy0), int(sx))
    if verbose:
        print(f"   → shift=({int(sx)},{int(sy)})")
    return int(sx), int(sy), scan


# ─────────────────────────── 有效性门控 ───────────────────────────

def wfs_validity(wfs: ThorlabWFS, n_avg: int = 1,
                 min_spot_ratio: float = 0.5) -> dict[str, Any]:
    """WFS 测量有效性门控 — 防止大幅度模式下子孔径光斑丢失导致的垃圾拟合.

    返回 dict: valid_ratio / n_valid / n_total / ok。
    ``ok=False`` 时该点读数应剔除或重测 (实测 R≈光束半径 + 大振幅时 DLL 拟合崩溃,
    ``|resp|`` 可暴涨到 10³ 量级)。
    """
    wfs.take_image(n_sample=1, dynamicNoiseCut=True)
    intensity, _ = wfs.get_spots_statics()
    arr = np.asarray(intensity, dtype=float)
    total = int(arr.size)
    valid = int(np.isfinite(arr).sum())
    ratio = valid / total if total else 0.0
    return {"valid_ratio": ratio, "n_valid": valid, "n_total": total,
            "ok": bool(ratio >= min_spot_ratio)}



# ─────────────────────────── 线性度指标 ───────────────────────────

def linearity_metrics(responses: list[np.ndarray], amps: list[float],
                      cv_threshold: float = 0.15, cos_threshold: float = 0.9
                      ) -> dict[str, Any]:
    """推拉响应向量的线性度 (**正确判据**).

    响应向量 ``resp = (z₊ − z₋)/2/a_waves`` **已按单位幅度归一化** → 线性响应表现为
    ``|resp|`` **恒定**。故不能用 slope/R² (线性时 slope≈0, R² 无意义), 而应用:

    - ``cv`` = std(|resp|)/mean(|resp|)  —— 幅度一致性 (越小越线性)
    - ``cos_min`` = 各幅度对之间响应向量夹角余弦最小值 —— 方向一致性
    - ``ok`` = cv < cv_threshold 且 cos_min > cos_threshold

    Args:
        responses: 各幅度下的响应向量列表 (等长)
        amps: 对应幅度 (rad), 仅用于排序

    Returns:
        dict: norms/cv/cos_min/ok/n_amps
    """
    order = np.argsort(np.asarray(amps, dtype=float))
    vecs = [np.asarray(responses[i], dtype=float) for i in order]
    norms = np.array([float(np.linalg.norm(v)) for v in vecs])
    mean_n = float(np.mean(norms)) if len(norms) else 0.0
    cv = float(np.std(norms) / mean_n) if mean_n > 0 else float("inf")
    cos_min = 1.0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            ni, nj = float(np.linalg.norm(vecs[i])), float(np.linalg.norm(vecs[j]))
            if ni > 0 and nj > 0:
                cos_min = min(cos_min, float(vecs[i] @ vecs[j] / (ni * nj)))
    ok = bool(cv < cv_threshold and cos_min > cos_threshold and len(vecs) >= 2)
    return {"norms": norms.tolist(), "cv": cv, "cos_min": cos_min,
            "n_amps": len(vecs), "ok": ok}
