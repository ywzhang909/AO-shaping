"""Simulation: what an SLM axial move off the front focal plane does to the model.

Ground truth is ``bench_twin`` with the new ``slm_axial_offset_m`` knob (the panel
no longer sits on the lens front focal plane, so the modulated field free-space
propagates to the lens first).  The model under test is the frozen forward's
axial face: the two geometric scalars ``defocus_diopter`` (``q = A/B``) and the
ramp map (proportional to ``B``, i.e. the far-field scale).  Here that family is
represented by ``{rescale(s) ∘ bench_twin(defocus_m=eps)}`` with the radiometric
gain marginalized in closed form, everything else held identical, so the residual
is the axial move alone.

Writes ``artifacts/metrics.json`` first, then the figures, then the report body.

Reproduce::

    python scripts/slm/generate_axial_move_sim_report.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from ao_shaping.algorithm.signal_processing.defocus_forward_core import (
    axial_plane_correction,
)
from ao_shaping.algorithm.signal_processing.gerchberg_saxton import gerchberg_saxton
from ao_shaping.drivers.sim.bench_twin import BenchTwin, BenchTwinConfig, TwinErrorModel
from ao_shaping.tools.slm.transfer.protocol import FixedAperture, aperture_metrics
from ao_shaping.utils.image.target.patterns import create_target_shape

ROOT = Path(__file__).resolve().parents[2]
REPORT_KEY = "report/shape_trials/axial_move_sim/axial_move_sim_20261029.md"
OUT_DIR = ROOT / "report" / "shape_trials" / "axial_move_sim"
FIG_DIR = OUT_DIR / "figures"
ART_DIR = OUT_DIR / "artifacts"

#: 实验系统标称几何 (2026-10-19 focus_probe: 补偿 0.559 1/m ⇒ 相机离后焦面 8.7 mm)。
F0_M = 0.125
EPS0_M = 0.0087
#: 模型面内的离焦屈光度 q = A/B; 符号由一次标度测量定, 这里取与孪生真值一致的值。
Q0_M = -EPS0_M / F0_M**2
N_PUPIL = 128
W0_PX = 250.0
#: 轴向偏移扫描点 (m); 正 = 远离透镜。
DELTAS_M = (0.0, 0.005, 0.010, 0.025, 0.050, 0.100, 0.200)
#: 一维闪耀相位周期 (面板 px): 把 ±1 级放在 ±40 个模型采样点上。
GRATING_PERIOD_PX = 32.0
#: GS 种子目标的方形边长 (模型采样点)。
GS_SIDE_PX = 60.0
#: 孪生远场补零后的采样数 (与其 far_field 内部一致)。
N_FFT = 1280


# ---------------------------------------------------------------------------
# 孪生与探针
# ---------------------------------------------------------------------------


def twin(delta_m: float = 0.0, eps_m: float = EPS0_M, *, noise_adu: float = 0.0,
         seed: int = 0) -> BenchTwin:
    """标称几何的孪生；``delta_m`` = 面板后移量, ``eps_m`` = 相机离焦传播距离。"""
    return BenchTwin(
        BenchTwinConfig(
            beam_w0_px=W0_PX,
            slm_axial_offset_m=float(delta_m),
            noise_adu=float(noise_adu),
            seed=int(seed),
            errors=TwinErrorModel(defocus_m=float(eps_m)),
        )
    )


def gs_seed() -> np.ndarray:
    """GS 方形种子 (补零到孪生的远场采样, 否则焦面采样差一个补零因子)。

    该种子**未收敛到干净方斑** (高斯照明 + 硬方形目标), 它的用途是把能量铺满
    目标区, 作为「延展图样」的模型保真度读数, 不是形状质量的参照。
    """
    base = twin()
    src = np.zeros((N_FFT, N_FFT))
    half = (N_FFT - N_PUPIL) // 2
    src[half:half + N_PUPIL, half:half + N_PUPIL] = base.pupil_amplitude()
    mask = create_target_shape(
        "square", (N_FFT, N_FFT), side=GS_SIDE_PX, radius_ratio=GS_SIDE_PX / N_FFT,
        center=((N_FFT - 1) / 2.0, (N_FFT - 1) / 2.0),
    )
    res = gerchberg_saxton(
        src, np.sqrt(np.asarray(mask, dtype=np.float64)), iterations=200,
        cell_spacing=base.pupil_pitch_m, distance=F0_M,
        wavelength=float(base.config.wavelength_m), propagation="fft",
    )
    return np.asarray(res.phase, dtype=np.float64)[half:half + N_PUPIL,
                                                   half:half + N_PUPIL]


def make_probes() -> dict[str, np.ndarray]:
    """一维闪耀 ±1 级 / 平相位 / GS 方形种子 (全部是流水线真会用量级的相位)。"""
    idx = (np.arange(N_PUPIL, dtype=np.float64) - (N_PUPIL - 1) / 2.0)[None, :]
    grad = 2.0 * math.pi / GRATING_PERIOD_PX
    return {
        "grating_pos": np.broadcast_to(grad * idx, (N_PUPIL, N_PUPIL)).copy(),
        "grating_neg": np.broadcast_to(-grad * idx, (N_PUPIL, N_PUPIL)).copy(),
        "flat": np.zeros((N_PUPIL, N_PUPIL)),
        "gs_seed": gs_seed(),
    }


# ---------------------------------------------------------------------------
# 模型族: rescale(s) ∘ twin(eps), 增益解析边缘化
# ---------------------------------------------------------------------------


def rescale_about_axis(image: np.ndarray, scale: float) -> np.ndarray:
    """绕光轴缩放远场图样 (与 ramp_matrix 乘 s 同义)。"""
    arr = np.asarray(image, dtype=np.float64)
    n = arr.shape[0]
    centre = (n - 1) / 2.0
    idx = np.arange(n, dtype=np.float64) - centre
    yy, xx = np.meshgrid(idx, idx, indexing="ij")
    coords = [centre + yy / scale, centre + xx / scale]
    return map_coordinates(arr, coords, order=3, mode="nearest")


def relative_rms(measured: np.ndarray, model: np.ndarray) -> tuple[float, float]:
    """最优增益下的相对 RMS 残差与增益 (radiometry 单独边缘化)。"""
    a = np.asarray(measured, dtype=np.float64).ravel()
    b = np.asarray(model, dtype=np.float64).ravel()
    denom = float(b @ b)
    gain = float(a @ b) / denom if denom > 0 else 0.0
    resid = a - gain * b
    return float(np.sqrt((resid @ resid) / (a @ a))), gain


def centroid(frame: np.ndarray) -> float:
    """沿 x 的强度质心 (模型采样点), 亚像素。"""
    arr = np.clip(np.asarray(frame, dtype=np.float64), 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        return float("nan")
    x = np.arange(arr.shape[1], dtype=np.float64)
    return float((arr.sum(axis=0) * x).sum() / total)


def order_separation(pos_frame: np.ndarray, neg_frame: np.ndarray) -> float:
    """±1 级质心间距 (模型采样点): 平移不敏感、对尺度一阶敏感。"""
    return abs(centroid(pos_frame) - centroid(neg_frame))


def fit_scale(measured: np.ndarray, base: np.ndarray,
              lo: float = 0.85, hi: float = 1.15) -> float:
    """在基准帧上找最优标度 (粗网格 + 抛物线细化)。"""
    grid = np.linspace(lo, hi, 61)
    errs = np.array([relative_rms(measured, rescale_about_axis(base, s))[0] for s in grid])
    k = int(np.argmin(errs))
    if 0 < k < len(grid) - 1:
        y0, y1, y2 = errs[k - 1], errs[k], errs[k + 1]
        denom = y0 - 2 * y1 + y2
        if denom > 0:
            return float(grid[k] + 0.5 * (y0 - y2) / denom * (grid[1] - grid[0]))
    return float(grid[k])


def model_frame(phase: np.ndarray, scale: float, eps_m: float) -> np.ndarray:
    return rescale_about_axis(twin(0.0, eps_m).far_field(phase), scale)


def fit_scale_and_defocus(measured: np.ndarray, phase: np.ndarray,
                          s0: float, eps0: float) -> dict[str, Any]:
    """单帧强度上联合拟合 (scale, eps); 报参数、1σ、相关系数。"""
    a = np.asarray(measured, dtype=np.float64).ravel()
    norm = max(float(np.abs(a).max()), 1e-12)

    def resid(theta: np.ndarray) -> np.ndarray:
        s, eps = float(theta[0]), float(theta[1])
        m = model_frame(phase, s, eps).ravel()
        denom = float(m @ m)
        gain = float(a @ m) / denom if denom > 0 else 0.0
        return (a - gain * m) / norm

    sol = least_squares(resid, np.array([s0, eps0]), diff_step=1e-4, xtol=1e-12,
                        ftol=1e-12, gtol=1e-12, max_nfev=600)
    jac = np.asarray(sol.jac, dtype=np.float64)
    sigma2 = float(sol.fun @ sol.fun) / max(int(a.size - 3), 1)
    try:
        cov = np.linalg.inv(jac.T @ jac) * sigma2
    except np.linalg.LinAlgError:  # pragma: no cover
        cov = np.full((2, 2), np.nan)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    corr = (float(cov[0, 1] / (sd[0] * sd[1]))
            if sd[0] > 0 and sd[1] > 0 else float("nan"))
    return {
        "scale": float(sol.x[0]), "defocus_m": float(sol.x[1]),
        "scale_sigma": float(sd[0]), "defocus_sigma_m": float(sd[1]),
        "corr": corr, "rms": float(np.sqrt(np.mean(sol.fun ** 2))),
        "note": "1σ 是形式值; 模型失配是系统性的, 真实精度看与真值的偏差 (bias_*)",
    }


# ---------------------------------------------------------------------------
# 形状读数
# ---------------------------------------------------------------------------


def extent_side_px(frame: np.ndarray, half_side: float) -> float:
    """目标框内 (强度 > 半高) 的等效方形边长 — 只读尺寸, 不读均匀度。"""
    arr = np.asarray(frame, dtype=np.float64)
    n = arr.shape[0]
    centre = (n - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(n) - centre, np.arange(n) - centre, indexing="ij")
    box = (np.abs(xx) <= half_side) & (np.abs(yy) <= half_side)
    peak = float(arr[box].max())
    if peak <= 0:
        return 0.0
    return float(math.sqrt(int((arr[box] > 0.5 * peak).sum())))


def shaping_metrics(frame: np.ndarray, half_side: float) -> dict[str, Any]:
    n = frame.shape[0]
    aperture = FixedAperture(centre_xy=((n - 1) / 2.0, (n - 1) / 2.0),
                             half_side=half_side, core_margin_px=2.0)
    out = aperture_metrics(np.asarray(frame, dtype=np.float64), aperture)
    out["extent_side_px"] = extent_side_px(frame, half_side)
    return {k: out[k] for k in ("cv", "ee_rel", "mean", "extent_side_px")}


# ---------------------------------------------------------------------------
# 四段实验
# ---------------------------------------------------------------------------


def experiment_scale_law(probes: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """A: 标度律 s = 1 + q·δ; ``eps=0`` 列是理想透镜恒等式的零假设。"""
    rows: list[dict[str, Any]] = []
    for eps in (EPS0_M, 0.0):
        sep0 = order_separation(twin(0.0, eps).far_field(probes["grating_pos"]),
                                twin(0.0, eps).far_field(probes["grating_neg"]))
        rms0, _ = relative_rms(twin(0.0, eps).far_field(probes["flat"]),
                               twin(0.0, eps).far_field(probes["flat"]))
        for delta in DELTAS_M:
            sep = order_separation(twin(delta, eps).far_field(probes["grating_pos"]),
                                   twin(delta, eps).far_field(probes["grating_neg"]))
            s_fit = sep / sep0
            s_pred = 1.0 + Q0_M * delta if eps != 0.0 else 1.0
            flat_truth = twin(delta, eps).far_field(probes["flat"])
            flat_base = twin(0.0, eps).far_field(probes["flat"])
            rows.append({
                "eps_m": float(eps), "delta_m": float(delta),
                "order_sep_px": float(sep), "order_sep0_px": float(sep0),
                "scale_fit": float(s_fit), "scale_pred": float(s_pred),
                "scale_rel_dev": float(s_fit / s_pred - 1.0),
                "flat_rel_rms": float(relative_rms(flat_truth, flat_base)[0]),
                "nominal_rms": float(rms0),
            })
    return rows


def experiment_model_cost(probes: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """B: 冻结模型 vs 闭式复合, 在同一真值帧上的残差 (延展图样)。"""
    rows: list[dict[str, Any]] = []
    for name in ("gs_seed", "grating_pos", "flat"):
        phase = probes[name]
        base = twin(0.0, EPS0_M).far_field(phase)
        for delta in DELTAS_M:
            truth = twin(delta, EPS0_M).far_field(phase)
            q_new, scale = axial_plane_correction(Q0_M, delta)
            composed = model_frame(phase, scale, EPS0_M / scale)
            rows.append({
                "probe": name, "delta_m": float(delta), "scale": float(scale),
                "defocus_diopter_new": float(q_new), "eps_new_m": float(EPS0_M / scale),
                "rel_rms_nominal": float(relative_rms(truth, base)[0]),
                "rel_rms_composed": float(relative_rms(truth, composed)[0]),
            })
    return rows


def experiment_shaping(probes: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """C: 延展图样的指标 — 目标框冻结在旧标度时的代价。"""
    phase = probes["gs_seed"]
    half = GS_SIDE_PX / 2.0
    rows: list[dict[str, Any]] = []
    for delta in DELTAS_M:
        truth_frame = twin(delta, EPS0_M).far_field(phase)
        _, scale = axial_plane_correction(Q0_M, delta)
        composed = model_frame(phase, scale, EPS0_M / scale)
        nominal = twin(0.0, EPS0_M).far_field(phase)
        rows.append({
            "delta_m": float(delta), "scale": float(scale),
            "target_half_side_px": half,
            "truth": shaping_metrics(truth_frame, half),
            "truth_at_rescaled_box": shaping_metrics(truth_frame, half * scale),
            "model_nominal": shaping_metrics(nominal, half),
            "model_composed": shaping_metrics(composed, half),
        })
    return rows


def experiment_identifiability(probes: dict[str, np.ndarray],
                               delta_m: float = 0.05) -> list[dict[str, Any]]:
    """D: 单帧能否同时定出 (scale, eps)? 三种探针各拟一次。"""
    _, scale_true = axial_plane_correction(Q0_M, delta_m)
    eps_true = EPS0_M / scale_true
    rows: list[dict[str, Any]] = []
    for name in ("flat", "grating_pos", "gs_seed"):
        phase = probes[name]
        truth = twin(delta_m, EPS0_M, noise_adu=2.0, seed=11).frame(phase)
        fit = fit_scale_and_defocus(truth, phase, 1.0, EPS0_M)
        rows.append({
            "probe": name, "delta_m": float(delta_m),
            "truth_scale": float(scale_true), "truth_eps_m": float(eps_true),
            "bias_scale": float(fit["scale"] / scale_true - 1.0),
            "bias_eps_m": float(fit["defocus_m"] - eps_true),
            **fit,
        })
    return rows


# ---------------------------------------------------------------------------
# 图 (英文标注: DejaVu Sans 无 CJK 字形)
# ---------------------------------------------------------------------------


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_figures(payload: dict[str, Any]) -> list[str]:
    plt = _pyplot()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    law = payload["scale_law"]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for eps, label, colour in ((EPS0_M, "camera 8.7 mm off FT plane (bench)", "C0"),
                               (0.0, "camera on FT plane (ideal)", "C1")):
        sel = [r for r in law if r["eps_m"] == eps]
        ax.plot([r["delta_m"] * 1e3 for r in sel], [r["scale_fit"] for r in sel],
                "o-", color=colour, label=f"{label}: measured")
    sel = [r for r in law if r["eps_m"] == EPS0_M]
    x = np.array([r["delta_m"] for r in sel]) * 1e3
    ax.plot(x, 1.0 + Q0_M * x * 1e-3, "k--", lw=1.0,
            label=r"$s = 1 + q\,\delta$,  $q = -0.557$ m$^{-1}$")
    ax.set_xlabel("SLM axial offset $\\delta$ (mm)")
    ax.set_ylabel("far-field scale $s$ (relative)")
    ax.set_title("Twin ground truth: scale vs axial offset")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "axial_move_scale_law_20261029.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)

    cost = payload["model_cost"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharey=True)
    for ax, probe, title in ((axes[0], "gs_seed", "extended (GS) pattern"),
                             (axes[1], "flat", "compact spot (flat phase)")):
        sel = [r for r in cost if r["probe"] == probe]
        x = [r["delta_m"] * 1e3 for r in sel]
        ax.plot(x, [100 * r["rel_rms_nominal"] for r in sel], "o-", color="C0",
                label="frozen model")
        ax.plot(x, [100 * r["rel_rms_composed"] for r in sel], "s--", color="C2",
                label="closed-form composition")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("$\\delta$ (mm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("model vs truth: relative RMS (%)")
    fig.suptitle("Model residual: frozen vs axially composed", fontsize=9)
    fig.tight_layout()
    path = FIG_DIR / "axial_move_model_cost_20261029.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)

    ident = payload["identifiability"]
    names = [r["probe"] for r in ident]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    axes[0].bar(names, [r["scale_sigma"] for r in ident], color="C0")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"formal $\sigma_s$")
    axes[0].set_title("scale: formal 1 sigma", fontsize=9)
    axes[1].bar(names, [abs(r["bias_scale"]) for r in ident], color="C3")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$|s_{fit}/s_{true} - 1|$")
    axes[1].set_title("scale: actual bias vs truth", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Single-frame joint fit of (s, eps) at delta=50 mm, s={ident[0]['truth_scale']:.4f}",
                 fontsize=9)
    fig.tight_layout()
    path = FIG_DIR / "axial_move_identifiability_20261029.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)
    return written


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _pick(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    def match(r: dict[str, Any], key: str, want: Any) -> bool:
        got = r[key]
        if isinstance(want, str) or isinstance(got, str):
            return got == want
        return abs(float(got) - float(want)) < 1e-12

    return next(r for r in rows if all(match(r, k, v) for k, v in kw.items()))


def render_report(payload: dict[str, Any], figures: list[str]) -> str:
    law, cost, shape = payload["scale_law"], payload["model_cost"], payload["shaping"]
    ident, meta = payload["identifiability"], payload["meta"]
    L: list[str] = []
    add = L.append

    bench = _pick(law, eps_m=EPS0_M, delta_m=0.05)
    null = _pick(law, eps_m=0.0, delta_m=0.05)
    cost_gs50 = _pick(cost, probe="gs_seed", delta_m=0.05)
    cost_gs100 = _pick(cost, probe="gs_seed", delta_m=0.1)
    cost_flat50 = _pick(cost, probe="flat", delta_m=0.05)
    sh50 = _pick(shape, delta_m=0.05)
    id_flat = _pick(ident, probe="flat")
    id_gs = _pick(ident, probe="gs_seed")
    id_gr = _pick(ident, probe="grating_pos")
    dev_max = max(abs(r["scale_rel_dev"]) for r in law
                  if r["eps_m"] == EPS0_M and r["delta_m"] <= 0.1)

    add("# SLM 轴向移动的模型代价与再辨识（仿真，2026-10-29）")
    add("")
    add("> 本报告的数字全部来自数字孪生真值，**不含任何实验测量**。"
        "适用面是「把 SLM 从前焦面附近沿光轴移开」这一类改动，量级按当前实验系统几何"
        "（f = 125 mm、相机离焦 0.559 m⁻¹，即离后焦面 8.7 mm）给出。"
        "上一轮 `axial_plane_20261026.md` 已给出同一套 ABCD 关系与硬件上的"
        "「条件最优在扫描边界、验证集改善很小」的判读；本报告补的是**孪生真值下的量级与可辨识性**。")
    add("")
    add("## 结论")
    add("")
    add(f"1. **移动本身几乎不可见，除非相机不在傅里叶面。** 相机在后焦面时 `A = 1 - d₂/f = 0`，"
        f"面板后移只改变输出二次相位，强度逐位不变：仿真在 `ε=0` 列的标度读数是 "
        f"{_pick(law, eps_m=0.0, delta_m=0.05)['scale_fit']:.5f}、平相位帧相对残差 "
        f"{100 * null['flat_rel_rms']:.2e}%。当前实验系统 `A ≠ 0`，移动**只通过这一个耦合**起作用。")
    add(f"2. **可观测的只有两个标量，且都有闭式解。** 远场标度 `s = 1 + q·δ`"
        f"（`q = A/B`，按当前几何 `q = {Q0_M:.4f} m⁻¹`）：δ = 50 mm 时 s = "
        f"{bench['scale_pred']:.4f}，±1 级间距实测 {bench['scale_fit']:.4f}"
        f"（偏差 {100 * bench['scale_rel_dev']:+.2f}%）；δ ≤ 100 mm 内最大偏差 {100 * dev_max:.2f}%。"
        f" 离焦屈光度同步变为 `q' = q/s`。**模型不需要新增自由度**。")
    add(f"3. **冻结模型的代价可以量化，但只有延展图样才看得见。** δ = 50 mm 时延展图样的"
        f"模型残差从 {100 * cost_gs50['rel_rms_nominal']:.1f}% 降到 "
        f"{100 * cost_gs50['rel_rms_composed']:.1f}%（δ = 100 mm："
        f"{100 * cost_gs100['rel_rms_nominal']:.1f}% → {100 * cost_gs100['rel_rms_composed']:.1f}%）；"
        f"而紧凑光斑（平相位）即使 δ = 50 mm 也只有 {100 * cost_flat50['rel_rms_nominal']:.2f}%。"
        f" 也就是说：**用小平斑自检会得出「移动没有影响」的错误结论**。")
    add(f"4. **单帧标度可测，单帧离焦不可测。** 在 δ = 50 mm 的真值帧上联合拟合 (s, ε)："
        f"平相位探针把 s 拟到偏差 {100 * abs(id_flat['bias_scale']):.2f}%，"
        f"相关系数 {id_flat['corr']:+.2f}（s 与 ε 几乎不可分）；延展图样偏差 "
        f"{100 * abs(id_gs['bias_scale']):.2f}%、相关系数 {id_gs['corr']:+.2f}。"
        f" 形式 1σ（{id_gs['scale_sigma']:.1e}）**远小于**实际偏差，因为失配是系统性的 —— "
        f"报告里只把偏差当精度。")
    add("5. **不重训练、不重标定面板项。** 面板 LUT / Zernike / 粗糙度属于器件，移动不改它们；"
        "要更新的只有两个几何标量、一次辐射度锚定，以及随之而来的**目标框与窗口标度**。")
    add("")
    add("## 1. 物理：移动改变了什么")
    add("")
    add("近轴 ABCD 复合链 `P(d₂)·L(f)·P(d₁)`：")
    add("")
    add("```text")
    add("A = 1 - d₂/f,   B = d₁ + d₂ - d₁·d₂/f,   q = A/B")
    add("```")
    add("")
    add("相机在后焦面（`d₂ = f`）⇒ `A = 0`、`B = f`，与 `d₁` 无关：面板位置只改变**输出**二次相位，"
        "强度不变。这就是孪生里 `ε = 0` 列的零假设。")
    add("")
    add("相机离焦（`d₂ = f + ε`）⇒ `A = -ε/f ≠ 0`。面板后移 δ 等价于总矩阵右侧乘 `P(δ)`：")
    add("")
    add("```text")
    add("B' = B + A·δ   ⇒   s = B'/B = 1 + q·δ,    q' = q/s")
    add("```")
    add("")
    add("这正是 `defocus_forward_core.axial_plane_correction(q, δ)` 的返回式："
        "`ramp_matrix` 乘 `s`、光学增益除 `s²`。除此之外模型没有新自由度。")
    add("")
    add("被忽略的二阶项，放在同一把尺子上（当前几何，`w₀ = 2 mm` ⇒ `z_R = 11.8 m`）：")
    add("")
    add("| 效应 | 标度 | δ = 50 mm | δ = 200 mm |")
    add("|---|---|---|---|")
    add("| 远场标度 `q·δ` | `0.557 m⁻¹ × δ` | 2.8% | 11.1% |")
    add("| 离焦屈光度相对变化 `q·δ` | 同上 | 2.8% | 11.1% |")
    add("| 照明半径 `w(δ)/w₀ - 1` | `(δ/z_R)²/2` | 9e-6 | 1.4e-4 |")
    add("| 衍射展宽 `λδ/(πw₀)` | 1.7 µm @ 50 mm | 2e-4 | 8e-4 |")
    add("")
    add("⇒ **δ ≤ 200 mm 时只有标度与离焦两项进入误差预算**。"
        "照明项只在把光束收得很紧（`w₀ ≲ 0.5 mm`，`z_R ≲ 0.7 m`）时才需要显式建模；"
        "移动前后各测一次 `beam_extent` 就能判定自己落在哪一侧。")
    add("")
    add("## 2. 仿真方法")
    add("")
    add(f"- 真值：`bench_twin` + 本轮新增的 `slm_axial_offset_m`（调制后的面板场先角谱传播 δ 到透镜面）。"
        f"`f = {meta['focal_length_m'] * 1e3:.0f} mm`、`w₀ = {meta['w0_px']:.0f}` 面板 px、"
        f"相机离焦 `ε = {meta['eps_m'] * 1e3:.1f} mm`、瞳面 {meta['pupil']}² / 56.25 µm。")
    add("- 模型族：`rescale(s) ∘ bench_twin(defocus_m = ε)`，即冻结前向的 `(q, ramp scale)` 两个标量；"
        "增益按最小二乘解析边缘化。其余项（面板误差、透镜像差、口径）两侧取同一值 ⇒ 残差只剩轴向移动。")
    add("- 探针：一维闪耀 ±1 级（周期 32 面板 px，落点 ±40 模型采样点，尺度读数用它）、"
        f"平相位（紧凑光斑）、GS 方形种子（边长 {GS_SIDE_PX:.0f} 采样点，延展图样）。"
        "三者都是流水线真会用量级的相位，不用像素级随机相位（它在孪生的周期网格上会引入绕回伪影）。")
    add("- 指标：`protocol.aperture_metrics` 的 `cv` / `ee_rel`，"
        "尺寸读数用目标框内「强度过半高」的等效方形边长。")
    add("")
    add("## 3. 结果 A：标度律")
    add("")
    add("| δ (mm) | ±1 级间距 (采样点) | s 实测 | s = 1+q·δ | 偏差 | 平相位帧残差（ε=8.7 mm） | 平相位帧残差（ε=0） |")
    add("|---|---|---|---|---|---|---|")
    for r in law:
        if r["eps_m"] != EPS0_M:
            continue
        zero = _pick(law, eps_m=0.0, delta_m=r["delta_m"])
        add(f"| {r['delta_m'] * 1e3:.0f} | {r['order_sep_px']:.3f} | {r['scale_fit']:.4f} | "
            f"{r['scale_pred']:.4f} | {100 * r['scale_rel_dev']:+.2f}% | "
            f"{100 * r['flat_rel_rms']:.2f}% | {100 * zero['flat_rel_rms']:.2e}% |")
    add("")
    over = _pick(law, eps_m=EPS0_M, delta_m=0.2)
    add(f"读数：闭式标度律在 δ ≤ 100 mm 内成立到 {100 * dev_max:.2f}%；δ = 200 mm 偏 "
        f"{100 * over['scale_rel_dev']:+.2f}%，说明二阶项已经进 1% 预算，那里要用实测标度而不是公式外推。"
        " `ε = 0` 列是**数值零**（恒等式），不是小量 —— 它同时证明了孪生新增的那一步传播"
        "确实只是在 `A ≠ 0` 时才产生强度效应。")
    add("")
    add("## 4. 结果 B：冻结模型 vs 闭式复合")
    add("")
    add("| 探针 | δ (mm) | s | q' (1/m) | 冻结模型残差 | 复合后残差 |")
    add("|---|---|---|---|---|---|")
    for probe in ("gs_seed", "flat"):
        for r in cost:
            if r["probe"] != probe:
                continue
            add(f"| {probe} | {r['delta_m'] * 1e3:.0f} | {r['scale']:.4f} | "
                f"{r['defocus_diopter_new']:.4f} | {100 * r['rel_rms_nominal']:.2f}% | "
                f"{100 * r['rel_rms_composed']:.2f}% |")
    add("")
    add("复合把延展图样的残差削掉一半以上，但**不会归零**：残余是 `rescale` 的三次样条插值误差"
        "与复合式只保留一阶项留下的二阶变化。它说明的是「误差降到与其它未建模项同量级」，"
        "不是「模型变精确」。紧凑光斑一列即使在 δ = 50 mm 也只有千分之几 —— "
        "**这就是为什么不能拿小平斑做移动后自检**：它对尺度几乎不敏感。")
    add("")
    add("## 5. 结果 C：延展图样的尺寸读数")
    add("")
    add(f"目标框半边长 {sh50['target_half_side_px']:.0f} 采样点（`FixedAperture`），边长读数是"
        "「框内强度过半高的等效方形边长」。GS 种子未收敛到干净方斑，所以这一栏读的是**尺寸**，"
        "不是形状质量。")
    add("")
    add("| δ (mm) | s | 真值边长 | 真值 ee_rel（旧框） | 真值 ee_rel（框×s） | 冻结模型边长 | 复合模型边长 |")
    add("|---|---|---|---|---|---|---|")
    for r in shape:
        add(f"| {r['delta_m'] * 1e3:.0f} | {r['scale']:.4f} | {r['truth']['extent_side_px']:.2f} | "
            f"{r['truth']['ee_rel']:.4f} | {r['truth_at_rescaled_box']['ee_rel']:.4f} | "
            f"{r['model_nominal']['extent_side_px']:.2f} | "
            f"{r['model_composed']['extent_side_px']:.2f} |")
    add("")
    add(f"δ = 50 mm 时真值边长 {sh50['truth']['extent_side_px']:.2f} 采样点，等于旧标度的 "
        f"{sh50['scale']:.3f} 倍 —— 同一相位在移动后**做小了 {100 * (1 - sh50['scale']):.1f}%**。"
        "这不是光学退化，是标度换了而控制映射没换。"
        f"旧框下的 `ee_rel` 从 {_pick(shape, delta_m=0.0)['truth']['ee_rel']:.4f} 落到"
        f" {sh50['truth_at_rescaled_box']['ee_rel']:.4f}… 这一栏的作用是提醒："
        "如果移动后沿用旧目标框，读数会出现一个**只由标度引起的**下降，很容易被误读成光路变差。")
    add("")
    add("## 6. 结果 D：可辨识性")
    add("")
    add("| 探针 | s 拟合 | s 真值 | 偏差 | 形式 σ_s | ε 拟合 (mm) | ε 真值 (mm) | corr(s, ε) |")
    add("|---|---|---|---|---|---|---|---|")
    for r in ident:
        add(f"| {r['probe']} | {r['scale']:.4f} | {r['truth_scale']:.4f} | "
            f"{100 * r['bias_scale']:+.2f}% | {r['scale_sigma']:.1e} | "
            f"{r['defocus_m'] * 1e3:.2f} | {r['truth_eps_m'] * 1e3:.2f} | {r['corr']:+.3f} |")
    add("")
    add(f"真值是 δ = 50 mm（s = {ident[0]['truth_scale']:.4f}，ε = {ident[0]['truth_eps_m'] * 1e3:.2f} mm）。"
        f"平相位探针的相关系数 {id_flat['corr']:+.2f}、s 偏差 {100 * abs(id_flat['bias_scale']):.2f}%："
        "紧凑光斑的**尺寸**同时被尺度和离焦改变，两者在这一个读数上退化。"
        f"延展图样把 s 定到 {100 * abs(id_gs['bias_scale']):.2f}% 偏差、相关 {id_gs['corr']:+.2f}；"
        f"单帧闪耀的偏差是 {100 * abs(id_gr['bias_scale']):.2f}%。"
        "形式 σ 比实际偏差小一到两个数量级，因为模型失配是系统性的 —— 报告只把**偏差**当精度。")
    add("")
    add("结论与 `axial_plane_20261026.md` 的硬件判读同向，并且给了量："
        "**单帧强度不能同时定出 (q, 映射)**，但**也不需要**同时定出 —— "
        "δ 是机械量（输入），s 与 q' 是输出；用一次延展图样测 s、一次离焦扫描测 q' 即可。")
    add("")
    add("## 7. 下次移动 SLM 的操作流程")
    add("")
    add("### 移动前（只读，约 10 分钟）")
    add("")
    add("1. `bench_fingerprint` 存指纹，并记下机械标尺读数 `z_before`。")
    add("2. `beam_extent` 测照明半径 `w₀` 与中心，按 §1 的表判定 `δ/z_R` 落在哪一侧。")
    add("3. 存**平相位**与**延展图样**各一帧定曝光画面，连同该轮 `result.json` —— 这是移动后唯一可比参照。")
    add("")
    add("### 移动后（按此顺序，约 15 分钟）")
    add("")
    add("1. **先验光路**：`light_guard` 判「有光」，`focus_preflight` 做离焦补偿与去转向；"
        "零级与 ROI 必须在补偿**之后**重新定位（沿用 `focus_probe` 的结论：补偿前定的 ROI 会偏几十 px）。")
    add("2. **两个标量各用一次专门测量**：")
    add("   - 标度 `s`：延展图样帧对移动前那一帧做一维标度搜索（或直接量 ±1 级间距的比值，"
        "本报告 §3 的做法）。核对 `s ≈ 1 + q·δ_model`：一致 ⇒ δ 与模型自洽；"
        "不一致 ⇒ 以**实测 s** 为准，并检查 `q` 的符号。")
    add("   - 离焦 `q'`：`focus_probe` 重扫一次（现有入口），或用 `q' = q/s` 预测后只测一个验证点。")
    add("3. **重算几何书签**（最容易漏、后果最长）：`ramp_matrix × s`、光学增益 `/ s²`、"
        "目标框 `half_side × s`、窗口跨度与 `control_grid` 同步缩放；"
        "`origin_xy` / `amplitude_center_xy` 不动（它们由光轴与照明中心定，不由 δ 定）。")
    add("4. **一次辐射度锚定**：平相位单帧重定 `gain` 与 `background`（`s⁻²` 只作用于光学增益，"
        "背景与曝光偏置不缩放）。")
    add("5. **写溯源**：检查点记录 `slm_axial_offset_m`、`axial_reference_curvature_diopter`、"
        "`bench_fingerprint` 哈希、`z_before/z_after`、实测 s 与该次测量的曝光。"
        "指纹不一致时**拒绝**加载旧检查点，并在报错里点名哪一项变了。")
    add("6. **面板项一律冻结**：LUT / 面板 Zernike / 粗糙度属于器件，移动不改它们。")
    add("")
    add("### 验收（三条，任一不过就不要开形状 campaign）")
    add("")
    add("1. 延展图样的模型残差与移动前同量级（本报告给的是复合后 "
        f"{100 * cost_gs50['rel_rms_composed']:.1f}% @ 50 mm 量级）。")
    add("2. 平相位帧的 `ee_rel` 与峰值在移动前后相差 ≤ 5%（否则是截光或曝光，不是模型问题）。")
    add("3. 一个已知方形目标做 A–B–A（旧检查点 / 新检查点 / 旧检查点）："
        "新检查点的 `cv` 与 `ee_rel` 不劣于旧检查点 —— 这一条同时检验标度与离焦是否都改对了。")
    add("")
    add("## 8. 本轮改动的代码")
    add("")
    add("| 位置 | 改动 | 理由 |")
    add("|---|---|---|")
    add("| `drivers/sim/bench_twin.py` | 新增 `BenchTwinConfig.slm_axial_offset_m` 与 "
        "`BenchTwin.slm_plane_to_lens()`；`far_field` 在透镜像差之前插入这一步 | "
        "孪生此前无法表示「面板不在前焦面」，于是「移动后模型是否还成立」在仿真里无法被证伪 |")
    add("| `scripts/slm/generate_axial_move_sim_report.py` | 本报告、三张图与 `metrics.json` | 生成命令即复现命令 |")
    add("| `tests/ao_shaping/drivers/sim/test_bench_twin.py` | 新参数的三条回归：δ=0 恒等、"
        "ε=0 时强度不变、标度按 1+q·δ 变化 | 把上面两条物理结论钉成测试 |")
    add("")
    add("模型侧（`defocus_forward_core.axial_plane_correction`、"
        "`FrozenDefocusParameters.slm_axial_offset_m` / `effective_parameters()`）**无需改动**："
        "本报告验证的正是这两个已存在的量。")
    add("")
    add("## 9. 边界与未验证项")
    add("")
    add("- 孪生的瞳面网格跨 900 面板 px（128 点 / 56.25 µm 节距），面板真实的 8 µm 结构不在传播里，"
        "所以本报告是**控制网格口径**下的结论。真实面板的高频结构在 δ 传播中抹得更快，"
        "但那一层已经超出冻结前向的表达能力。")
    add("- 有限透镜口径与传播后截光未建模（孪生用网格边界当口径）；按 §1 的量级估计，"
        "δ ≤ 200 mm 时它进不了 1% 预算。若将来把光束收窄到 `w₀ < 0.5 mm`，这一项要重估。")
    add("- 照明项只在 §1 的量级表里出现，没有单独仿真：它的判据是 `δ/z_R`，由实测 `w₀` 决定，"
        "不需要新的模型自由度。")
    add("- GS 种子未收敛到干净方斑（高斯照明 + 硬方形目标），§5 的边长读数因此只当尺寸用。")
    add("- 报告里所有 δ 都是**标称机械位移**。`axial_plane_20261026.md` 已说明，"
        "任何从强度反推的 δ 都不能当卷尺读数；本报告的立场相同：δ 是输入，`s` 与 `q'` 是输出。")
    add("")
    add("## 附：复现")
    add("")
    add("```bash")
    add("python scripts/slm/generate_axial_move_sim_report.py")
    add("```")
    add("")
    add(f"产物：`artifacts/metrics.json`、`figures/` 下 {len(figures)} 张图"
        f"（{', '.join(figures)}）。")
    add("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def build_payload() -> dict[str, Any]:
    probes = make_probes()
    law = experiment_scale_law(probes)
    cost = experiment_model_cost(probes)
    shape = experiment_shaping(probes)
    ident = experiment_identifiability(probes)
    return {
        "meta": {
            "focal_length_m": F0_M, "eps_m": EPS0_M, "q_model_per_m": Q0_M,
            "w0_px": W0_PX, "pupil": N_PUPIL, "square_side_px": GS_SIDE_PX,
            "grating_period_px": GRATING_PERIOD_PX, "n_fft": N_FFT,
            "deltas_m": list(DELTAS_M), "report_key": REPORT_KEY,
        },
        "scale_law": law, "model_cost": cost, "shaping": shape,
        "identifiability": ident,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-report", action="store_true", help="只算不写报告正文")
    args = ap.parse_args(argv)

    ART_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    (ART_DIR / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    figures = render_figures(payload)
    if not args.no_report:
        (OUT_DIR / "axial_move_sim_20261029.md").write_text(
            render_report(payload, figures), encoding="utf-8"
        )
    print(f"wrote {OUT_DIR} ({len(figures)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
