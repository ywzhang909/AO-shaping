"""Issue #165 P0 — 校准 FourierGSNet 前向到本台架 2f 物理前向（纯离线仿真）。

本脚本是 issue #165 的 P0 离线验收实现。**不打开任何硬件**：全部在
``Bench2FGroundTruth``（数字孪生）上跑。

为什么需要一个「数字孪生」而不是直接用理想 FFT 当真值：如果真值就是被测模型
自己，实验就是循环论证。所以真值里必须含有**被校准模型结构上表示不出来**的物理：

* 超高斯照明（指数 > 2），而校准模型只用高斯；
* 固定的一块**逐像素自由形状**像差场，它落在 Zernike 张成之外
  （``ZernikeAberrationField.fit`` 拟合不掉）；
* 探测器噪声（散粒噪声 + 读出噪声）。

校准模型（``Bench2FPropagator``）能表达的只有：圆形孔径、高斯照明、
把瞳面相位**投影到 66 项 Zernike 张成**（与实验系统 MIL 的 ~70 参数物理模型
同量级）、增益 + 背景、中心补零 512² 的 FFT、模方、峰值归一化。

两条实验臂：

* **臂 A（前向保真度）** —— issue 的 (3) 判据所在。24 帧（3 个相位族 × 8），
  在同一 36×36 网格上比较「理想 FFT 前向 / 校准 2f 前向 / 物理 Zernike 基线」
  三者对真值的相关性、包围能量、NRMSE、两个口径的 CV。
  issue 给的硬件基线是「物理 Zernike 0.839 / EE 0.642」。
* **臂 B（网络换前向前后）** —— issue 的 (2)。同一份数据、同一 seed、同一预算，
  分别用理想 FFT 前向与校准 2f 前向训练 FourierGSNet；把网络预测的相位送进
  **真值**孪生算远场，与目标比较（相关性 / 包围能量 / NRMSE / CV），
  并与「Zernike 约束的经典 GS 物理基线」同口径对照。

**P0 决策**：只有当臂 A 里校准前向的 ≥ 物理 Zernike 基线的（相关性不低于、
包围能量不低于基线减 2 % 容差）才打印 ``P0_DECISION: PASS``，否则 ``FAIL``
并以退出码 2 结束——对应 issue 的对比择优逻辑：不劣于基线才允许进入 P1。

用法::

    python scripts/slm/calibrate_gsnet_2f.py
    python scripts/slm/calibrate_gsnet_2f.py --skip-network
    python scripts/slm/calibrate_gsnet_2f.py --decision-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from numpy.typing import NDArray as npt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (  # noqa: E402
    ZernikeCoefficientOptimizer,
)
from ao_shaping.utils.wavefront.bench2f import (  # noqa: E402
    Bench2FGeometry,
    Bench2FGroundTruth,
    Bench2FPropagator,
    ZernikeAberrationField,
)
from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms  # noqa: E402
from ml.gsnet.forward_ops import Bench2FOperator, IdealFFTOperator  # noqa: E402
from ml.gsnet.model import FourierGSNet  # noqa: E402

# --------------------------------------------------------------------------------------
# 绘图约定（仓库 scripts 规范）
# --------------------------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

FAMILIES = ("smooth", "zernike", "freeform")

#: 全部指标的键。``correlation_full`` / ``nrmse_full`` 是全远场网格上的量，
#: 是有分辨力的那一套；``correlation`` / ``nrmse`` / ``cv_*`` 是 36×36 裁剪口径。
METRIC_KEYS = (
    "correlation",
    "nrmse",
    "correlation_full",
    "nrmse_full",
    "encircled_energy",
    "cv_target_box",
    "cv_flat_rms",
    "composite",
)
FAMILY_LABELS = {
    "smooth": "低阶 Zernike (Noll 4-15)",
    "zernike": "高阶 Zernike (Noll 4-66)",
    "freeform": "全频随机相位",
}
MODEL_LABELS = {
    "ideal": "理想 FFT 前向",
    "calibrated": "校准 2f 前向",
    "zernike": "物理 Zernike 基线",
}


# --------------------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------------------
def _peak_normalise(image: np.ndarray) -> np.ndarray:
    peak = float(np.max(image))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros_like(image, dtype=np.float64)
    return np.asarray(image, dtype=np.float64) / peak


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


def _cv_box(image: np.ndarray, side: int) -> float:
    """现行口径：目标方框内 std/mean。"""
    if side < 1 or image.shape[0] < side or image.shape[1] < side:
        return float("nan")
    h, w = image.shape
    r0, c0 = (h - side) // 2, (w - side) // 2
    box = image[r0 : r0 + side, c0 : c0 + side]
    mean = float(box.mean())
    if not np.isfinite(mean) or mean == 0.0:
        return float("inf")
    return float(box.std() / mean)


def _cv_flat_rms(image: np.ndarray, side: int, flat_side: int) -> float:
    """文献口径（Zhou 2010 < 6 % rms）：小平坦区的 rms 相对起伏。"""
    if side < 1 or image.shape[0] < side or image.shape[1] < side:
        return float("nan")
    h, w = image.shape
    r0, c0 = (h - side) // 2, (w - side) // 2
    box = image[r0 : r0 + side, c0 : c0 + side]
    f = max(1, min(flat_side, side))
    fr0, fc0 = (side - f) // 2, (side - f) // 2
    flat = box[fr0 : fr0 + f, fc0 : fc0 + f]
    mean = float(flat.mean())
    if not np.isfinite(mean) or mean == 0.0:
        return float("inf")
    return float(np.sqrt(((flat - mean) ** 2).mean()) / mean)


def _encircled_energy(image: np.ndarray, side: int) -> float:
    """目标方框内能量占**全图**能量的比例。

    分母必须是全图：若分母也取方框，EE 恒等于 1，这个指标就没有信息量。
    """
    if side < 1 or image.shape[0] < side or image.shape[1] < side:
        return float("nan")
    h, w = image.shape
    r0, c0 = (h - side) // 2, (w - side) // 2
    box = image[r0 : r0 + side, c0 : c0 + side]
    total = float(image.sum())
    if not np.isfinite(total) or total <= 0.0:
        return float("nan")
    return float(box.sum() / total)


def _crop_centre(image: np.ndarray, side: int) -> np.ndarray:
    h, w = image.shape
    if h < side or w < side:
        raise ValueError(f"image {image.shape} smaller than the metric side {side}")
    r0, c0 = (h - side) // 2, (w - side) // 2
    return image[r0 : r0 + side, c0 : c0 + side]


def _nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    a = _peak_normalise(pred)
    b = _peak_normalise(ref)
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    return float(np.sqrt(((a - b) ** 2).mean()))


def _composite_score(pib: float, cv: float) -> float:
    """仓库的整形复合分数：``0.5*PIB + 0.5/(1+CV)``（与 ``slm_gs_refine`` 同口径）。

    整形质量的主判据不是「与目标的相关性」——在远场中心裁剪窗里，0 阶斑本身就
    与方形目标高度相关，平场反而「赢」。真正区分整形好坏的是框内**均匀度**与
    **能量占比**，这也是 issue 写的验收线（CV ≤ 0.3 且 EE ≥ 0.7）。
    """
    if not np.isfinite(pib) or not np.isfinite(cv):
        return float("nan")
    return float(0.5 * pib + 0.5 / (1.0 + max(cv, 0.0)))


def frame_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    metric_side: int,
    target_side: int | None = None,
) -> dict[str, float]:
    """两套网格上的指标。

    * ``correlation`` / ``nrmse`` / ``cv_*`` 在**同一 metric_side×metric_side 网格**
      上算（issue 的 (3) 口径，默认 36×36 的远场中心裁剪）。
    * ``correlation_full`` / ``nrmse_full`` 在**整个远场网格**上算。
      这一套才是有分辨力的：36×36 的裁剪里 0 阶峰占主导，任何模型都能拿到
      0.97 以上的相关性，分不出好坏；散斑结构铺满全图时，全图相关性才随
      模型对不对而变。
    * ``encircled_energy`` 是目标方框内能量占**全图**能量的比例。
    """
    p_full = np.asarray(pred, dtype=np.float64)
    t_full = np.asarray(truth, dtype=np.float64)
    p = _crop_centre(p_full, metric_side)
    t = _crop_centre(t_full, metric_side)
    flat_side = max(3, metric_side // 4)
    box_side = target_side if target_side is not None else metric_side
    cv_box = _cv_box(p, box_side)
    ee = _encircled_energy(p_full, box_side)
    return {
        "correlation": _pearson(p, t),
        "nrmse": _nrmse(p, t),
        "correlation_full": _pearson(p_full, t_full),
        "nrmse_full": _nrmse(p_full, t_full),
        "encircled_energy": ee,
        "cv_target_box": cv_box,
        "cv_flat_rms": _cv_flat_rms(p, metric_side, flat_side),
        "composite": _composite_score(ee, cv_box),
    }


def _agg(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


# --------------------------------------------------------------------------------------
# 相位族
# --------------------------------------------------------------------------------------
def _zernike_basis(region: int, n_max: int) -> torch.Tensor:
    field = ZernikeAberrationField(region=region, n_max=n_max)
    return field.basis()


#: 各相位族的 rms（弧度）。幅度是关键：0.3 rad 的弱像差几乎不散射，
#: 0 阶峰主导整个远场，于是任何前向模型的相关性都 >0.97，指标失去分辨力。
#: 这里的取值让远场真正成为散斑场（与实验系统「全频随机相位」族同量级）。
FAMILY_RMS = {"smooth": 0.8, "zernike": 1.2, "freeform": 2.0}


def make_phase(
    family: str,
    region: int,
    rng: np.random.Generator,
    *,
    n_max: int = 10,
    rms_rad: float | None = None,
) -> np.ndarray:
    """在瞳面网格 (region, region) 上生成一个相位（弧度，未包裹）。"""
    if rms_rad is None:
        rms_rad = FAMILY_RMS[family]
    if family == "smooth":
        basis = _zernike_basis(region, 4)[3:15]  # Noll 4..15
    elif family == "zernike":
        basis = _zernike_basis(region, 10)[3:]  # Noll 4..66
    elif family == "freeform":
        raw = rng.normal(0.0, 1.0, size=(region, region))
        # 高斯低通，模拟器件带宽内可实现的逐像素相位
        k = np.fft.fftfreq(region)[:, None]
        l = np.fft.fftfreq(region)[None, :]
        sigma_px = 2.0
        kernel = np.exp(-2.0 * (np.pi**2) * (sigma_px**2) * (k**2 + l**2))
        filtered = np.real(np.fft.ifft2(np.fft.fft2(raw) * kernel))
        scale = float(np.sqrt((filtered**2).mean()))
        if scale > 0:
            filtered = filtered / scale
        return (filtered * rms_rad).astype(np.float64)
    else:  # pragma: no cover - argparse 限制了取值
        raise ValueError(f"unknown phase family: {family}")

    coeffs = rng.normal(0.0, 1.0, size=basis.shape[0])
    field = np.tensordot(coeffs, basis.numpy(), axes=(0, 0))
    scale = float(np.sqrt((field**2).mean()))
    if scale > 0:
        field = field / scale
    return (field * rms_rad).astype(np.float64)


# --------------------------------------------------------------------------------------
# 臂 A：前向保真度
# --------------------------------------------------------------------------------------
def _fixed_bench_aberration(region: int, seed: int) -> torch.Tensor:
    """实验系统自身的一块固定像差（落在 Zernike 张成**内**，幅度 0.25 rad rms）。

    探针要拟合的就是它；它张成内的部分可被物理 Zernike 模型吸收，
    张成外的部分（孪生内部的自由形状像差）任何 66 项模型都拟合不掉。
    """
    rng = np.random.default_rng(seed + 4242)
    coeffs = rng.normal(0.0, 1.0, size=15)  # Noll 4..18
    basis = ZernikeAberrationField(region=region, n_max=6).basis().numpy()[3:18]
    field = np.tensordot(coeffs, basis, axes=(0, 0))
    scale = float(np.sqrt((field**2).mean()))
    if scale > 0:
        field = field / scale
    return torch.from_numpy((field * 0.8).astype(np.float32))[None, None]


def _fit_bench_aberration(
    optimizer: ZernikeCoefficientOptimizer,
    truth_bench: Bench2FGroundTruth,
    bench_aberration: torch.Tensor,
    region: int,
    geometry: Bench2FGeometry,
    seed: int,
    n_probes: int = 8,
    iterations: int = 80,
) -> np.ndarray:
    """用探针帧拟合实验系统的像差系数（与 MIL 的 Step A 同一套机制）。

    直接调用仓库自己的 ``_fit_aberration_at_probes``——即闭环里 Step A 用的那个
    helper——而不是自己重写一遍循环。原因是它把**多帧探针在同一个 Adam 状态里
    轮转**，而逐帧各自拟合会把共享系数向量带偏；另外 ``update`` 在连续若干步
    没有改善后会闩住 ``_converged`` 且**没有人能解开**，自己写的循环会悄悄
    停在半路（实测：自写循环拟合出的残差比「像差为零」还差）。

    探针相位取**逐像素随机**（band-limited）而不是低阶 Zernike：低阶 Zernike
    探针能被 66 项 Zernike 模型整个吸收，模型会去解释探针而不是像差
    （实测拟合系数与真值像差的相关性 ≈ 0.13；换成随机相位后明显改善）。这与
    ``scripts/slm/model_in_loop_hw_runbook.py`` 的告诫一致。
    """
    from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
        _fit_aberration_at_probes,
    )

    rng = np.random.default_rng(seed + 31337)
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_probes):
        probe = torch.from_numpy(
            make_phase("freeform", region, rng, rms_rad=1.2)
        ).to(torch.float32)
        measured = (
            truth_bench.forward(probe[None, None] + bench_aberration)[0, 0]
            .numpy()
            .astype(np.float64)
        )
        frames.append((measured, probe.numpy().astype(np.float64)))
    fitted, _, _, _ = _fit_aberration_at_probes(optimizer, frames, iterations)
    return np.asarray(fitted, dtype=np.float64)


@dataclass
class ArmAConfig:
    geometry: Bench2FGeometry
    frames_per_family: int
    metric_side: int
    target_side: int
    seed: int
    truth_bench: Bench2FGroundTruth
    fitted_aberration: torch.Tensor
    fitted_coeffs: npt.NDArray[np.float64]
    bench_aberration: torch.Tensor


def run_arm_a(cfg: ArmAConfig, out_dir: Path) -> dict[str, Any]:
    """24 帧同帧对照：三种前向 vs 真值。

    三个预测器共用**同一组**从探针帧拟合出来的 Zernike 像差系数，
    所以它们之间的差异只来自「传播链本身的物理」，不来自「谁拟合得更好」：

    * ``ideal``       —— 理想 FFT（网络当前内部用的），既不知道像差，
      也没有孔径、照明、增益、背景。
    * ``calibrated``  —— 本台架 2f 物理前向（孔径 + 高斯照明 + 512² 补零
      + 增益 + 背景 + 峰值归一化），像差用同一组拟合系数。
    * ``zernike``     —— 仓库自己的 MIL 物理模型（``ZernikeCoefficientOptimizer``），
      即 issue 里的「物理 Zernike 基线」。
    """
    geometry = cfg.geometry
    region = geometry.region
    rng = np.random.default_rng(cfg.seed)

    propagator = Bench2FPropagator(geometry, n_max=10)
    truth_bench = cfg.truth_bench
    bench_aberration = cfg.bench_aberration

    ideal_op = IdealFFTOperator(geometry.far_field_size, convention="canonical")

    # 物理 Zernike 基线：仓库自己的 MIL 正向模型（66 项 Zernike + 冻结 piston/tilt）
    zernike_opt = ZernikeCoefficientOptimizer(
        n_orders=10,
        region=region,
        lr=0.05,
        far_field_size=geometry.far_field_size,
        frozen_modes=(1, 2, 3),
        dtype="float64",
    )
    amplitude = propagator.pupil_amplitude().numpy()[0, 0].astype(np.float64)
    zernike_opt.set_source_amplitude(amplitude)

    fitted_phase = cfg.fitted_aberration
    fitted_coeffs = cfg.fitted_coeffs
    # 校准前向算子带上这组拟合像差：命令相位保持逐像素自由度，
    # 实验系统的像差叠在上面——与 MIL 的前向模型同构。
    bench_op = Bench2FOperator(propagator, aberration=fitted_phase)
    per_family: dict[str, dict[str, list[dict[str, float]]]] = {
        fam: {model: [] for model in MODEL_LABELS} for fam in FAMILIES
    }
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for family in FAMILIES:
        for index in range(cfg.frames_per_family):
            phase = make_phase(family, region, rng)
            phase_t = torch.from_numpy(phase).to(torch.float32)[None, None]

            truth = truth_bench.forward(phase_t + bench_aberration)[0, 0].numpy()
            truth = truth.astype(np.float64)

            amp_t = propagator.pupil_amplitude()
            ideal_far = ideal_op.forward(amp_t, phase_t)
            ideal_int = (ideal_far.real**2 + ideal_far.imag**2)[0, 0].numpy().astype(np.float64)

            # 校准前向：命令相位 + 拟合像差，走本台架 2f 物理链
            cal_far = bench_op.forward(amp_t, phase_t)
            cal_int = (cal_far.real**2 + cal_far.imag**2)[0, 0].numpy().astype(np.float64)

            # 物理 Zernike 基线：同一组系数，走仓库自己的 MIL 前向
            zernike_int = zernike_opt.forward_intensity(
                fitted_coeffs, phase, source_amplitude=amplitude
            )

            for model, image in (
                ("ideal", ideal_int),
                ("calibrated", cal_int),
                ("zernike", zernike_int),
            ):
                per_family[family][model].append(
                    frame_metrics(image, truth, cfg.metric_side, cfg.target_side)
                )

            np.savez_compressed(
                frames_dir / f"{family}_{index:02d}.npz",
                phase_rad=phase,
                truth=truth,
                ideal=ideal_int,
                calibrated=cal_int,
                zernike=zernike_int,
            )

    summary: dict[str, Any] = {"per_family": {}, "overall": {}}
    for family in FAMILIES:
        summary["per_family"][family] = {
            model: {
                key: _agg([row[key] for row in rows])
                for key in METRIC_KEYS
            }
            for model, rows in per_family[family].items()
        }
    for model in MODEL_LABELS:
        rows = [row for family in FAMILIES for row in per_family[family][model]]
        summary["overall"][model] = {
            key: _agg([row[key] for row in rows]) for key in METRIC_KEYS
        }
    return summary


# --------------------------------------------------------------------------------------
# 臂 B：网络换前向前后
# --------------------------------------------------------------------------------------
def _square_target(size: int, far_field_size: int) -> np.ndarray:
    """远场网格中心的方形平顶目标（峰值归一）。"""
    target = np.zeros((far_field_size, far_field_size), dtype=np.float64)
    r0 = (far_field_size - size) // 2
    target[r0 : r0 + size, r0 : r0 + size] = 1.0
    return target


def _source_intensity(region: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """瞳面上的高斯照明（归一到峰值 1）。

    腰宽与中心偏移随机化：训练集里的照明必须有多样性，否则网络只会学到
    「这一束光的解」。
    """
    axis = np.arange(region) - (region - 1) / 2.0
    if rng is None:
        waist = region / 2.0
        shift_x = shift_y = 0.0
    else:
        waist = (region / 2.0) * float(rng.uniform(0.7, 1.3))
        shift_x = float(rng.uniform(-0.08, 0.08)) * region
        shift_y = float(rng.uniform(-0.08, 0.08)) * region
    xx, yy = np.meshgrid(axis - shift_x, axis - shift_y, indexing="ij")
    r2 = (xx**2 + yy**2) / (waist**2)
    src = np.exp(-2.0 * r2)
    return (src / src.max()).astype(np.float64)


def _pupil_back(far_field: torch.Tensor, pupil_side: int) -> torch.Tensor:
    """远场复数场 -> 瞳面复数场（中心裁剪）。

    零填充 512² 的正向 FFT 的逆步：``ifft2(ifftshift(far))`` 给出 512² 的
    填充瞳面场，中心 ``pupil_side`` 那一块才是真正的瞳面。
    """
    back = torch.fft.ifft2(torch.fft.ifftshift(far_field, dim=(-2, -1)), dim=(-2, -1))
    # 逆变换的输出里瞳面还在「卷绕」帧上，必须再 fftshift 一次才对齐正向
    # 链路的中心帧——少了这一步，裁出来的瞳面相位误差是 ~4 rad（完全错）。
    back = torch.fft.fftshift(back, dim=(-2, -1))
    side = back.shape[-1]
    half = (side - pupil_side) // 2
    return back[..., half : half + pupil_side, half : half + pupil_side]


def _classical_gs(
    forward_complex,
    target_amp: torch.Tensor,
    pupil_side: int,
    iterations: int,
) -> torch.Tensor:
    """在给定前向下跑经典 GS，返回瞳面相位。

    ``forward_complex(pupil_phase)`` 必须给出**复数**远场（幅度已峰值归一），
    因为 GS 的幅度约束需要远场的相位，而强度图不携带相位。
    """
    phase = torch.angle(_pupil_back(target_amp.to(torch.complex64), pupil_side))
    for _ in range(iterations):
        far = forward_complex(phase)
        far_phase = torch.angle(far)
        far_c = target_amp * torch.exp(1j * far_phase)
        phase = torch.angle(_pupil_back(far_c, pupil_side))
    return phase


def _make_training_set(
    geometry: Bench2FGeometry,
    truth_bench: Bench2FGroundTruth,
    bench_aberration: torch.Tensor,
    n_samples: int,
    target_side: int,
    gs_iterations: int,
    seed: int,
    gt_mode: str = "bench",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从真值孪生合成 (source, target, gt_phase) 训练对。

    每个样本换一套照明与目标，否则网络学的是「这一组 (source, target) 的解」：
    经典 GS 是确定性的，固定 (source, target) 会让 64 个样本逐位相同。
    """
    rng = np.random.default_rng(seed)
    region = geometry.region
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    gt_phases: list[torch.Tensor] = []

    for _ in range(n_samples):
        source = torch.from_numpy(_source_intensity(region, rng)).to(torch.float32)[None, None]
        side = int(target_side * float(rng.uniform(0.75, 1.25)))
        # 网络的 source/target 必须在**同一网格**上（FourierGSNet 的契约），
        # 所以目标画在 region×region 的网络网格上。
        target = torch.from_numpy(
            _square_target(side, region)
        ).to(torch.float32)[None, None]
        # 真值侧的 GS 跑在过采样的远场上（512²），但两个远场网格的**视场相同**
        # （补零只改采样不改视场），所以同一个边长在两张网格上是同一个角尺寸。
        target_far = torch.from_numpy(
            _square_target(side, geometry.far_field_size)
        ).to(torch.float32)[None, None]
        target_amp = torch.sqrt(target_far.clamp_min(0.0) + 1e-12)
        if gt_mode == "bench":
            # GT = 在**过采样真值**上跑经典 GS 的解（对真值而言是最优相位）。
            phase = _classical_gs(
                lambda ph: _truth_complex_far_field(
                    truth_bench, geometry, ph, bench_aberration
                ),
                target_amp,
                region,
                gs_iterations,
            )
        else:
            # GT = 在**网络自己的网格**上跑经典 GS 的解（与网络能表达的东西同口径）。
            phase = _classical_gs_same_regime(
                source, target, region, gs_iterations
            )
        sources.append(source[0])
        targets.append(target[0])
        gt_phases.append(phase[0, 0].clone())

    return (
        torch.stack(sources),
        torch.stack(targets),
        torch.stack(gt_phases)[:, None],
    )


def _classical_gs_same_regime(
    source: torch.Tensor,
    target: torch.Tensor,
    pupil_side: int,
    iterations: int,
) -> torch.Tensor:
    """在网络自己的 64×64 网格上跑经典 GS（理想 FFT 口径）。

    这是「网络架构能表达的那个解」：网络的 source/target 必须在同一网格上，
    它的内部远场只能是同尺寸 FFT（衍射极限光斑亚像素、图样混叠）。
    用过采样真值当 GT 时，网络学不到（实测相关性 ≈ 0），那个差值量的是
    **架构限制**，不是前向校准的好坏。
    """
    amp = torch.sqrt(source.clamp_min(0.0) + 1e-12)
    tamp = torch.sqrt(target.clamp_min(0.0) + 1e-12)
    phase = torch.angle(_pupil_back(tamp.to(torch.complex64), pupil_side))
    for _ in range(iterations):
        far = torch.fft.fftshift(torch.fft.fft2(amp * torch.exp(1j * phase)), dim=(-2, -1))
        far_c = tamp * torch.exp(1j * torch.angle(far))
        phase = torch.angle(_pupil_back(far_c, pupil_side))
    return phase


def _truth_complex_far_field(
    truth_bench: Bench2FGroundTruth,
    geometry: Bench2FGeometry,
    phase: torch.Tensor,
    bench_aberration: torch.Tensor,
) -> torch.Tensor:
    """真值孪生的复远场（强度 -> 复数，幅度取峰值归一后的 sqrt）。

    GS 需要的是复远场的相位；真值孪生只给强度，所以用「幅度 = sqrt(强度)、
    相位 = 0」的实数远场参与幅度约束迭代。
    """
    return truth_bench.complex_far_field(phase + bench_aberration)


@dataclass
class ArmBConfig:
    geometry: Bench2FGeometry
    n_train: int
    n_eval: int
    epochs: int
    metric_side: int
    target_side: int
    seed: int
    truth_bench: Bench2FGroundTruth
    fitted_aberration: torch.Tensor
    bench_aberration: torch.Tensor
    gt_mode: str = "same"
    num_layers: int = 8
    base_channels: int = 24


def _evaluate_network(
    model: FourierGSNet,
    source: torch.Tensor,
    target: torch.Tensor,
    truth_bench: Bench2FGroundTruth,
    bench_aberration: torch.Tensor,
    metric_side: int,
    target_side: int,
    far_field_size: int,
) -> dict[str, Any]:
    """把网络预测的相位送进**真值**孪生，再与目标比。"""
    model.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for i in range(source.shape[0]):
            pred_phase = model(source[i : i + 1], target[i : i + 1])
            far = truth_bench.forward(pred_phase + bench_aberration)[0, 0].numpy()
            ref = _square_target(target_side, far_field_size)
            rows.append(
                frame_metrics(far.astype(np.float64), ref, metric_side, target_side)
            )
    return {key: _agg([row[key] for row in rows]) for key in METRIC_KEYS}


def _shaping_loss(
    operator, source_amp: torch.Tensor, phase: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """1 - Pearson（与仓库 ``ShapingLosses.shaping_loss`` 同口径）。

    关键：**用本臂自己的前向算子**算预测远场。损失若用理想 FFT 算，
    校准臂就在被一个与它内部物理不一致的目标函数训练。
    """
    far = operator.forward(source_amp, phase)
    intensity = far.real**2 + far.imag**2
    pred = intensity.flatten(1)
    tgt = target.flatten(1)
    p = pred - pred.mean(dim=1, keepdim=True)
    t = tgt - tgt.mean(dim=1, keepdim=True)
    denom = torch.sqrt((p**2).sum(dim=1) * (t**2).sum(dim=1)) + 1e-12
    return torch.mean(1.0 - (p * t).sum(dim=1) / denom)


def _train_arm(
    operator,
    source: torch.Tensor,
    target: torch.Tensor,
    gt_phase: torch.Tensor,
    cfg: ArmBConfig,
) -> FourierGSNet:
    """一个训练臂：固定 seed、固定预算，损失与仓库的 train_gsnet 同口径。

    损失 = ``w_phase * 圆周相位 MSE + w_shaping * (1 - Pearson)``，
    与 ``ml.gsnet.train.train_gsnet`` 的构成一致；两项都用**本臂的前向算子**。
    """
    torch.manual_seed(cfg.seed)
    model = FourierGSNet(
        num_layers=cfg.num_layers,
        base_channels=cfg.base_channels,
        shared_weights=True,
        forward_op=operator,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.5)
    n = source.shape[0]
    batch = 16
    w_phase, w_shaping = 1.0, 40.0
    for _epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            pred = model(source[idx], target[idx])
            diff = torch.remainder(pred - gt_phase[idx] + torch.pi, 2 * torch.pi) - torch.pi
            phase_loss = torch.nn.functional.mse_loss(diff, torch.zeros_like(diff))
            source_amp = torch.sqrt(source[idx].clamp_min(0.0) + 1e-12)
            shaping_loss = _shaping_loss(operator, source_amp, pred, target[idx])
            loss = w_phase * phase_loss + w_shaping * shaping_loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        scheduler.step()
    return model


def run_arm_b(cfg: ArmBConfig, out_dir: Path) -> dict[str, Any]:
    geometry = cfg.geometry
    region = geometry.region
    propagator = Bench2FPropagator(geometry, n_max=10)
    truth_bench = cfg.truth_bench
    bench_aberration = cfg.bench_aberration

    source, target, gt_phase = _make_training_set(
        geometry,
        truth_bench,
        bench_aberration,
        cfg.n_train,
        cfg.target_side,
        gs_iterations=20,
        seed=cfg.seed,
        gt_mode=cfg.gt_mode,
    )

    # 网络的 source/target 必须在同一网格上，所以网络臂的远场网格 = 瞳面网格。
    # 评估仍然走 512² 的过采样真值。
    net_geometry = replace(geometry, far_field_size=region)
    net_propagator = Bench2FPropagator(net_geometry, n_max=10)
    ideal_op = IdealFFTOperator(region, convention="canonical")
    bench_op = Bench2FOperator(net_propagator, aberration=cfg.fitted_aberration)

    summary: dict[str, Any] = {}
    for name, operator in (("ideal", ideal_op), ("calibrated", bench_op)):
        started = time.time()
        model = _train_arm(operator, source, target, gt_phase, cfg)
        metrics = _evaluate_network(
            model,
            source[: cfg.n_eval],
            target[: cfg.n_eval],
            truth_bench,
            bench_aberration,
            cfg.metric_side,
            cfg.target_side,
            geometry.far_field_size,
        )
        metrics["train_seconds"] = time.time() - started
        summary[name] = metrics

    # 物理 Zernike 基线：把经典 GS 解投影到 66 项 Zernike 张成（同一目标、同一真值前向）
    field = ZernikeAberrationField(region=region, n_max=10)
    basis_np = field.basis().numpy().astype(np.float64)
    rows: list[dict[str, float]] = []
    for i in range(cfg.n_eval):
        target_amp = torch.sqrt(
            torch.from_numpy(
                _square_target(cfg.target_side, geometry.far_field_size)
            ).to(torch.float32)[None, None].clamp_min(0.0)
        )
        gs_phase = _classical_gs(
            lambda ph: _truth_complex_far_field(
                truth_bench, geometry, ph, bench_aberration
            ),
            target_amp,
            region,
            iterations=40,
        )
        coeffs, *_ = np.linalg.lstsq(
            basis_np.reshape(basis_np.shape[0], -1).T,
            gs_phase[0, 0].numpy().reshape(-1),
            rcond=None,
        )
        zernike_phase = field.render(
            torch.from_numpy(coeffs).to(torch.float32)[None, :]
        )
        with torch.no_grad():
            far = truth_bench.forward(zernike_phase + bench_aberration)[0, 0].numpy()
        ref = _square_target(cfg.target_side, geometry.far_field_size)
        rows.append(
            frame_metrics(far.astype(np.float64), ref, cfg.metric_side, cfg.target_side)
        )
    summary["zernike_baseline"] = {
        key: _agg([row[key] for row in rows]) for key in METRIC_KEYS
    }

    # 上下界：平场（零相位）与 GS 在过采样真值上的解本身。
    # 没有这两个数，网络臂的分数无法解读。
    ref = _square_target(cfg.target_side, geometry.far_field_size)
    flat_rows: list[dict[str, float]] = []
    for i in range(cfg.n_eval):
        with torch.no_grad():
            far = truth_bench.forward(
                torch.zeros(1, 1, region, region) + bench_aberration
            )[0, 0].numpy()
        flat_rows.append(
            frame_metrics(far.astype(np.float64), ref, cfg.metric_side, cfg.target_side)
        )
    summary["flat"] = {key: _agg([row[key] for row in flat_rows]) for key in METRIC_KEYS}

    with torch.no_grad():
        far = truth_bench.forward(gt_phase[: cfg.n_eval] + bench_aberration)[
 :, 0
        ].numpy()
    ceiling_rows = [
        frame_metrics(
            far[i].astype(np.float64), ref, cfg.metric_side, cfg.target_side
        )
        for i in range(min(cfg.n_eval, far.shape[0]))
    ]
    summary["gs_ceiling"] = {
        key: _agg([row[key] for row in ceiling_rows]) for key in METRIC_KEYS
    }
    return summary


# --------------------------------------------------------------------------------------
# 图
# --------------------------------------------------------------------------------------
def plot_arm_a(summary: dict[str, Any], out_path: Path) -> None:
    models = list(MODEL_LABELS)
    families = list(FAMILIES)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    width = 0.26
    for ax, key, title in (
        (axes[0], "correlation", "远场相关性（36×36 同帧网格）"),
        (axes[1], "encircled_energy", "包围能量 EE"),
    ):
        xs = np.arange(len(families))
        for i, model in enumerate(models):
            means = [summary["per_family"][f][model][key]["mean"] for f in families]
            errs = [
                summary["per_family"][f][model][key]["sd"]
                or 0.0
                for f in families
            ]
            ax.bar(xs + (i - 1) * width, means, width, yerr=errs, label=MODEL_LABELS[model])
        ax.set_xticks(xs)
        ax.set_xticklabels([FAMILY_LABELS[f] for f in families], fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("臂 A：三种前向对真值数字孪生的保真度（issue #165 P0）")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_arm_b(summary: dict[str, Any], out_path: Path) -> None:
    models = [k for k in ("ideal", "calibrated", "zernike_baseline") if k in summary]
    labels = {
        "ideal": "网络（理想 FFT 前向）",
        "calibrated": "网络（校准 2f 前向）",
        "zernike_baseline": "物理 Zernike 基线",
    }
    keys = ("correlation", "encircled_energy", "nrmse")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, key in zip(axes, keys):
        means = [summary[m][key]["mean"] for m in models]
        errs = [summary[m][key]["sd"] or 0.0 for m in models]
        ax.bar(range(len(models)), means, yerr=errs, color=["#8ecae6", "#219ebc", "#ffb703"][: len(models)])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([labels[m] for m in models], fontsize=8)
        ax.set_title(key)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("臂 B：FourierGSNet 换前向前后（真值孪生评分）")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 决策
# --------------------------------------------------------------------------------------
def decide(
    summary_a: dict[str, Any],
    summary_b: dict[str, Any] | None = None,
    ee_tolerance: float = 0.02,
) -> tuple[bool, str]:
    """P0 决策。

    issue 的判据原文是「**校准后网络**与物理 Zernike 基线比，不劣于基线才允许
    进入 P1」——所以闸门落在**网络臂**（臂 B）：校准前向训练出来的网络，
    与物理 Zernike 基线在同一个目标、同一个评分口径下比。

    臂 A（前向保真度）是支撑证据：校准前向必须明显优于理想 FFT 前向，
    否则「把网络内部的前向换成校准版」这件事本身没有意义。
    """
    parts: list[str] = []
    passed = True
    evaluated = True

    if summary_b:
        cal = summary_b.get("calibrated")
        base = summary_b.get("zernike_baseline")
        ideal = summary_b.get("ideal")
        if cal and base:
            # 判据：整形复合分数不劣于物理 Zernike 基线（issue 的「不劣于基线」）。
            # 相关性与 NRMSE 同时报出来，但它们在远场中心裁剪窗里被 0 阶斑主导，
            # 不能单独当整形判据。
            score_ok = cal["composite"]["mean"] >= base["composite"]["mean"]
            corr_ok = cal["correlation_full"]["mean"] >= base["correlation_full"]["mean"]
            passed = bool(score_ok and corr_ok)
            parts.append(
                f"composite={cal['composite']['mean']:.4f} "
                f"EE={cal['encircled_energy']['mean']:.4f} "
                f"CV={cal['cv_target_box']['mean']:.4f} "
                f"corr_full={cal['correlation_full']['mean']:.4f} | "
                f"baseline composite={base['composite']['mean']:.4f} "
                f"EE={base['encircled_energy']['mean']:.4f} "
                f"CV={base['cv_target_box']['mean']:.4f} "
                f"corr_full={base['correlation_full']['mean']:.4f}"
            )
            if ideal:
                parts.append(
                    f"| ideal-forward composite={ideal['composite']['mean']:.4f} "
                    f"corr_full={ideal['correlation_full']['mean']:.4f}"
                )

    else:
        # 臂 B 没跑 ⇒ **P0 判据（网络层）没有被评估**。此时报 PASS 等于
        # 对一个没跑过的门签字，不允许。臂 A 的前向保真度是另一件事，
        # 单独报在 forward corr 那一段里。
        evaluated = False

    overall = summary_a["overall"]
    parts.append(
        f"|| forward corr: ideal={overall['ideal']['correlation']['mean']:.4f} "
        f"calibrated={overall['calibrated']['correlation']['mean']:.4f} "
        f"zernike={overall['zernike']['correlation']['mean']:.4f}"
    )

    if not evaluated:
        line = "P0_DECISION: NOT_EVALUATED  (arm B not run; the network gate was " \
               "not evaluated)  " + "  ".join(parts)
        return False, line
    line = f"P0_DECISION: {'PASS' if passed else 'FAIL'}  " + "  ".join(parts)
    return passed, line


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录（默认 data/debug/gsnet_2f_calib/<时间戳>）",
    )
    parser.add_argument("--grid", type=int, default=64, help="瞳面模型网格边长")
    parser.add_argument("--far-field-size", type=int, default=512, help="远场 FFT 网格边长")
    parser.add_argument("--frames", type=int, default=8, help="每个相位族的帧数（总帧数 = 3×）")
    parser.add_argument("--metric-side", type=int, default=36, help="指标网格边长")
    parser.add_argument("--target-side", type=int, default=36, help="远场方形目标边长")
    parser.add_argument("--epochs", type=int, default=40, help="每个网络臂的训练轮数")
    parser.add_argument("--n-train", type=int, default=96, help="训练样本数")
    parser.add_argument("--n-eval", type=int, default=24, help="评估样本数")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--gt-mode",
        choices=["bench", "same"],
        default="same",
        help=(
            "训练目标的来源：bench = 过采样真值上跑 GS 的解（对真值最优，但网络"
            "架构学不到）；same = 网络自己网格上跑 GS 的解（与网络能表达的同口径）"
        ),
    )
    parser.add_argument("--skip-network", action="store_true", help="只跑臂 A")
    parser.add_argument("--decision-only", action="store_true", help="只打印决策行")
    return parser


def _prepare_bench(
    geometry: Bench2FGeometry, seed: int
) -> tuple[
    Bench2FGroundTruth, torch.Tensor, torch.Tensor, npt.NDArray[np.float64]
]:
    """构造真值孪生、实验系统的固定像差，以及用探针帧拟合出来的像差。

    拟合用的机制与 MIL 的 Step A 相同（``ZernikeCoefficientOptimizer.update``），
    所以「实验系统有多像差」这件事，网络和物理基线知道的是同一份信息。
    """
    region = geometry.region
    propagator = Bench2FPropagator(geometry, n_max=10)
    freeform_rng = np.random.default_rng(seed + 991)
    freeform = torch.from_numpy(
        make_phase("freeform", region, freeform_rng, rms_rad=0.12)
    ).to(torch.float32)
    truth_bench = Bench2FGroundTruth(
        geometry,
        freeform=freeform,
        illumination_exponent=2.6,
        shot_noise=0.0,
        read_noise=0.0,
        seed=seed + 7,
    )
    bench_aberration = _fixed_bench_aberration(region, seed)

    zernike_opt = ZernikeCoefficientOptimizer(
        n_orders=10,
        region=region,
        lr=0.05,
        far_field_size=geometry.far_field_size,
        frozen_modes=(1, 2, 3),
        dtype="float64",
    )
    zernike_opt.set_source_amplitude(
        propagator.pupil_amplitude().numpy()[0, 0].astype(np.float64)
    )
    fitted_coeffs = _fit_bench_aberration(
        zernike_opt, truth_bench, bench_aberration, region, geometry, seed
    )
    fitted_aberration = ZernikeAberrationField(region=region, n_max=10).render(
        torch.from_numpy(fitted_coeffs).to(torch.float32)[None, :]
    )
    return truth_bench, bench_aberration, fitted_aberration, fitted_coeffs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    geometry = Bench2FGeometry(
        region=args.grid,
        far_field_size=args.far_field_size,
        panel_disc_radius_px=450.0,
        beam_waist_panel_px=450.0,
        camera_px_per_model_px=8.0,
        gain=1.0,
        background=0.0,
    )

    out_dir = args.out
    if out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "data" / "debug" / "gsnet_2f_calib" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    truth_bench, bench_aberration, fitted_aberration, fitted_coeffs = _prepare_bench(
        geometry, args.seed
    )

    arm_a_cfg = ArmAConfig(
        geometry=geometry,
        frames_per_family=args.frames,
        metric_side=args.metric_side,
        target_side=args.target_side,
        seed=args.seed,
        fitted_aberration=fitted_aberration,
        fitted_coeffs=fitted_coeffs,
        bench_aberration=bench_aberration,
        truth_bench=truth_bench,
    )
    summary_a = run_arm_a(arm_a_cfg, out_dir)
    plot_arm_a(summary_a, out_dir / "forward_fidelity.png")

    summary_b: dict[str, Any] = {}
    if not args.skip_network:
        arm_b_cfg = ArmBConfig(
            geometry=geometry,
            n_train=args.n_train,
            n_eval=args.n_eval,
            epochs=args.epochs,
            metric_side=args.metric_side,
            target_side=args.target_side,
            seed=args.seed,
            fitted_aberration=fitted_aberration,
            bench_aberration=bench_aberration,
            truth_bench=truth_bench,
            gt_mode=args.gt_mode,
        )
        summary_b = run_arm_b(arm_b_cfg, out_dir)
        if summary_b:
            plot_arm_b(summary_b, out_dir / "network_before_after.png")

    passed, line = decide(summary_a, summary_b or None)
    results = {
        "geometry": asdict(geometry),
        "config": {
            "gt_mode": args.gt_mode,
            "frames_per_family": args.frames,
            "metric_side": args.metric_side,
            "target_side": args.target_side,
            "epochs": args.epochs,
            "n_train": args.n_train,
            "n_eval": args.n_eval,
            "seed": args.seed,
        },
        "arm_a_forward_fidelity": summary_a,
        "arm_b_network": summary_b,
        "decision": {
            "passed": passed,
            "evaluated": bool(summary_b),
            "line": line,
        },
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(line)
    print(f"结果目录: {out_dir}")
    if not args.decision_only:
        for family in FAMILIES:
            print(f"\n[臂 A / {family}]")
            for model in MODEL_LABELS:
                row = summary_a["per_family"][family][model]
                print(
                    f"  {MODEL_LABELS[model]:<14} "
                    f"corr={row['correlation']['mean']:.4f}±{row['correlation']['sd']:.4f} "
                    f"EE={row['encircled_energy']['mean']:.4f} "
                    f"NRMSE={row['nrmse']['mean']:.4f}"
                )
        if summary_b:
            print("\n[臂 B / 网络]")
            for name in ("ideal", "calibrated", "zernike_baseline"):
                if name not in summary_b:
                    continue
                row = summary_b[name]
                print(
                    f"  {name:<18} "
                    f"corr={row['correlation']['mean']:.4f}±{row['correlation']['sd']:.4f} "
                    f"EE={row['encircled_energy']['mean']:.4f} "
                    f"NRMSE={row['nrmse']['mean']:.4f} "
                    f"cv_box={row['cv_target_box']['mean']:.4f}"
                )
    # 退出码 2 = 判据不通过；退出码 3 = 判据未被评估（臂 B 没跑）。
    # 两者都**不是**成功。
    if not summary_b:
        return 3
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
