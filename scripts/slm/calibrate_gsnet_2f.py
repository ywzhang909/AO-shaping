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
from dataclasses import dataclass
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
from ao_shaping.drivers.sim.bench_twin import (  # noqa: E402
    BenchTwin,
    BenchTwinConfig,
    TwinErrorModel,
)
from ao_shaping.utils.image.beam_metrics import prepare_frame  # noqa: E402
from ao_shaping.utils.wavefront.zernike_calc import calc_n_zernike_terms  # noqa: E402
from ml.gsnet.forward_ops import BenchTwinOperator, IdealFFTOperator  # noqa: E402
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


def _encircled_energy(
    image: np.ndarray, side: int, center: tuple[float, float] | None = None
) -> float:
    """目标方框内能量占**全图**能量的比例。

    分母必须是全图：若分母也取方框，EE 恒等于 1，这个指标就没有信息量。
    """
    if side < 1 or image.shape[0] < side or image.shape[1] < side:
        return float("nan")
    h, w = image.shape
    if center is None:
        r0, c0 = (h - side) // 2, (w - side) // 2
    else:
        cy, cx = center
        r0 = int(round(cy)) - side // 2
        c0 = int(round(cx)) - side // 2
        r0 = max(0, min(r0, h - side))
        c0 = max(0, min(c0, w - side))
    box = image[r0 : r0 + side, c0 : c0 + side]
    total = float(image.sum())
    if not np.isfinite(total) or total <= 0.0:
        return float("nan")
    return float(box.sum() / total)


def _crop_centre(
    image: np.ndarray, side: int, center: tuple[float, float] | None = None
) -> np.ndarray:
    """从图像里裁出 ``side``×``side`` 的窗口。

    ``center`` 是窗口中心（行, 列）；``None`` 表示图像正中。窗口跟着目标走
    ——和仓库自己的 ROI 指标一致（ROI 落在实测光斑上，不是画面正中）。
    """
    h, w = image.shape
    if h < side or w < side:
        raise ValueError(f"image {image.shape} smaller than the metric side {side}")
    if center is None:
        r0, c0 = (h - side) // 2, (w - side) // 2
    else:
        cy, cx = center
        r0 = int(round(cy)) - side // 2
        c0 = int(round(cx)) - side // 2
        r0 = max(0, min(r0, h - side))
        c0 = max(0, min(c0, w - side))
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
    center: tuple[float, float] | None = None,
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
    p = _crop_centre(p_full, metric_side, center)
    t = _crop_centre(t_full, metric_side, center)
    flat_side = max(3, metric_side // 4)
    box_side = target_side if target_side is not None else metric_side
    cv_box = _cv_box(p, box_side)
    ee = _encircled_energy(p_full, box_side, center)
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
    """Zernike 基（canonical 入口，Noll 序）。"""
    from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

    generator = ZernikeGenerator(
        resolution=(region, region), radius=region / 2.0, n_orders=n_max
    )
    out = np.zeros((calc_n_zernike_terms(n_max), region, region))
    for index in range(out.shape[0]):
        coefficients = np.zeros(out.shape[0])
        coefficients[index] = 1.0
        raw = generator.generate_noll(coefficients)
        out[index] = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0)
    return torch.from_numpy(out).to(torch.float32)


def make_phase(
    family: str,
    region: int,
    rng: np.random.Generator,
    *,
    rms_rad: float | None = None,
) -> np.ndarray:
    """在瞳面网格 (region, region) 上生成一个相位（弧度，未包裹）。"""
    if family == "smooth":
        basis = _zernike_basis(region, 4)[3:15]  # Noll 4..15
    elif family == "zernike":
        basis = _zernike_basis(region, 10)[3:]  # Noll 4..66
    elif family == "freeform":
        raw = rng.normal(0.0, 1.0, size=(region, region))
        k = np.fft.fftfreq(region)[:, None]
        l = np.fft.fftfreq(region)[None, :]
        kernel = np.exp(-2.0 * (np.pi**2) * (2.0**2) * (k**2 + l**2))
        filtered = np.real(np.fft.ifft2(np.fft.fft2(raw) * kernel))
        scale = float(np.sqrt((filtered**2).mean()))
        if scale > 0:
            filtered = filtered / scale
        return (filtered * (2.0 if rms_rad is None else rms_rad)).astype(np.float64)
    else:  # pragma: no cover - argparse 限制了取值
        raise ValueError(f"unknown phase family: {family}")

    coeffs = rng.normal(0.0, 1.0, size=basis.shape[0])
    field = np.tensordot(coeffs, basis.numpy(), axes=(0, 0))
    scale = float(np.sqrt((field**2).mean()))
    if scale > 0:
        field = field / scale
    return (field * (0.8 if rms_rad is None else rms_rad)).astype(np.float64)


#: 三族误差的随机化幅度（域随机化的轴）。每一族都是实验系统上**真实存在**
#: 的非理想因素，范围取自实验系统的量级。
LUT_GAMMA_RANGE = (0.85, 1.25)
PANEL_ZERNIKE_MAX_RAD = 0.12
PANEL_ROUGHNESS_RANGE = (0.0, 0.04)
DEFOCUS_RANGE_M = (-2.0e-4, 2.0e-4)
TILT_RANGE_RAD = (-1.5e-4, 1.5e-4)
LENS_ZERNIKE_MAX_RAD = 0.10
#: 设备在 1064 nm 的 2π 灰度数（驱动从 EEPROM 动态查询）。1023 是 10 位满量程，
#: 不是相位级数；灰度→振幅耦合的灰度域周期必须等于 2π 灰度数，所以这里用 993。
SLM_TWO_PI_GRAY_LEVELS = 993
#: 灰度→振幅耦合的随机化幅度（强度一次谐波幅度 c）。仓库诊断判据
#: ``rel_spread > 0.15`` 蕴含 ``c ≳ 0.075``（0 阶展宽 = 2c）；面板真值仓库里
#: 没记下来，所以当待测量量，0.15 是刻意保守的上端，0 覆盖理想面板。
AMPLITUDE_COUPLING_RANGE = (0.0, 0.15)


def random_errors(rng: np.random.Generator) -> TwinErrorModel:
    """抽一组三族误差参数。

    * 族 1 SLM LUT 非线性 + 量化：Gamma 指数偏离线性。
    * 族 2 SLM 面板静态相差：低阶 Zernike + 固定种子粗糙度。
    * 族 3 装调：相机离焦 + 光路倾斜 + 透镜像差。

    另外带**灰度→振幅耦合**（族 1 的延伸）：LCOS 的实际振幅依赖驱动灰度，
    灰度域周期 = 2π 灰度数。``calibrated`` 臂走的是孪生链，自动含这一项；
    ``zernike`` 基线（66 项 + 冻结 piston/tilt）表达不了它 —— 这正是 P0 判据
    要区分开的东西。
    """
    panel_modes = rng.choice(np.arange(4, 19), size=int(rng.integers(3, 9)), replace=False)
    lens_modes = rng.choice(np.arange(4, 13), size=int(rng.integers(2, 6)), replace=False)
    return TwinErrorModel(
        lut_gamma=float(rng.uniform(*LUT_GAMMA_RANGE)),
        lut_phase_max_rad=2.0 * np.pi,
        gray_levels=SLM_TWO_PI_GRAY_LEVELS,
        slm_amplitude_coupling=float(rng.uniform(*AMPLITUDE_COUPLING_RANGE)),
        slm_amplitude_coupling_phase_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
        slm_zernike={int(m): float(rng.uniform(-PANEL_ZERNIKE_MAX_RAD, PANEL_ZERNIKE_MAX_RAD)) for m in panel_modes},
        slm_zernike_n_max=6,
        slm_roughness_sigma_rad=float(rng.uniform(*PANEL_ROUGHNESS_RANGE)),
        slm_roughness_seed=int(rng.integers(0, 2**31 - 1)),
        defocus_m=float(rng.uniform(*DEFOCUS_RANGE_M)),
        tilt_x_rad=float(rng.uniform(*TILT_RANGE_RAD)),
        tilt_y_rad=float(rng.uniform(*TILT_RANGE_RAD)),
        lens_zernike={int(m): float(rng.uniform(-LENS_ZERNIKE_MAX_RAD, LENS_ZERNIKE_MAX_RAD)) for m in lens_modes},
        lens_zernike_n_max=4,
    )


def build_twin(
    seed: int,
    *,
    region: int = 64,
    region_span_px: float = 900.0,
    far_field_padding: int = 8,
    beam_w0_px: float = 250.0,
    peak_adu: float = 100.0,
    noise_adu: float = 2.0,
    randomize: bool = True,
) -> BenchTwin:
    """造一台 2f 实验系统数字孪生（仓库规范的核心）。

    ``randomize=False`` ⇒ 理想系统（三族误差全零）。
    """
    rng = np.random.default_rng(seed)
    errors = random_errors(rng) if randomize else TwinErrorModel()
    config = BenchTwinConfig(
        wavelength_m=1064e-9,
        slm_pixel_um=8.0,
        ccd_pixel_um=2.2,
        focal_length_m=0.125,
        region=region,
        region_span_px=region_span_px,
        far_field_padding=far_field_padding,
        beam_w0_px=beam_w0_px,
        center_crop=True,
        # 不裁剪：孪生的帧就是整张远场网格（region*padding），与仓库既有的
        # model 网格口径一致，网络的内部前向与标签也都在这张网格上。
        crop_px=region * far_field_padding,
        peak_adu=peak_adu,
        noise_adu=noise_adu,
        seed=seed,
        errors=errors,
    )
    return BenchTwin(config)


@dataclass
class ArmAConfig:
    """臂 A 的配置（口径全部对齐仓库规范的 2f 孪生）。"""

    region: int
    region_span_px: float
    far_field_padding: int
    frames_per_family: int
    metric_side: int
    target_side: int
    seed: int
    noise_adu: float


def run_arm_a(cfg: ArmAConfig, out_dir: Path) -> dict[str, Any]:
    """24 帧同帧对照：三种前向 vs **孪生真值**。

    真值是仓库规范的 2f 数字孪生（``drivers/sim/bench_twin.py``）：三族误差
    （LUT 非线性+量化 / 面板静态相差 / 装调）+ 泊松散粒噪声与读出噪声。

    三个预测器：

    * ``ideal``       —— **零误差**孪生的传播链（理想物理 + 同样的过采样与裁剪）。
      它与 ``calibrated`` 的差别**只有**三族误差，所以两行之差就是「实验系统
      的非理想因素值多少」。
    * ``calibrated``  —— 带随机化误差的孪生传播链（网络内部该用的那条）。
    * ``zernike``     —— 仓库自己的 MIL 物理模型（66 项 Zernike + 冻结
      piston/tilt），系数由探针帧拟合得到，即 issue 里的「物理 Zernike 基线」。

    三者在**同一个远场网格**上（孪生的 ``crop_px = region*padding`` ⇒ 不裁剪），
    所以逐像素可比。
    """
    region = cfg.region
    far_field = region * cfg.far_field_padding
    rng = np.random.default_rng(cfg.seed)

    twin = build_twin(
        cfg.seed,
        region=region,
        region_span_px=cfg.region_span_px,
        far_field_padding=cfg.far_field_padding,
        noise_adu=cfg.noise_adu,
        randomize=True,
    )
    ideal_twin = build_twin(
        cfg.seed,
        region=region,
        region_span_px=cfg.region_span_px,
        far_field_padding=cfg.far_field_padding,
        noise_adu=0.0,
        randomize=False,
    )
    op_cal = BenchTwinOperator(twin)
    op_ideal = BenchTwinOperator(ideal_twin)

    # 物理 Zernike 基线：仓库自己的 MIL 正向模型，系数由孪生帧拟合
    zernike_opt = ZernikeCoefficientOptimizer(
        n_orders=10,
        region=region,
        lr=0.05,
        far_field_size=far_field,
        frozen_modes=(1, 2, 3),
        dtype="float64",
    )
    amplitude = np.asarray(twin.pupil_amplitude(), dtype=np.float64)
    zernike_opt.set_source_amplitude(amplitude)

    per_family: dict[str, dict[str, list[dict[str, float]]]] = {
        fam: {model: [] for model in MODEL_LABELS} for fam in FAMILIES
    }
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for family in FAMILIES:
        for index in range(cfg.frames_per_family):
            phase = make_phase(family, region, rng)
            phase_t = torch.from_numpy(phase).to(torch.float32)[None, None]

            # 孪生的帧带读出噪声 ⇒ 有负像素。先过仓库的 prepare_frame
            # （中位数去本底 + 截零），否则「框内能量 / 全图能量」会 > 1。
            truth = prepare_frame(np.asarray(twin.frame(phase), dtype=np.float64))

            ideal_far = op_ideal.forward(None, phase_t)
            ideal_int = (ideal_far.real**2 + ideal_far.imag**2)[0, 0].numpy().astype(np.float64)

            cal_far = op_cal.forward(None, phase_t)
            cal_int = (cal_far.real**2 + cal_far.imag**2)[0, 0].numpy().astype(np.float64)

            # 物理 Zernike 基线：探针拟合 → 预测
            probe = make_phase("freeform", region, rng, rms_rad=1.2)
            measured = np.asarray(twin.frame(probe), dtype=np.float64)
            zernike_opt.update(measured, probe)
            zernike_int = zernike_opt.forward_intensity(
                zernike_opt.coefficients, phase, source_amplitude=amplitude
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


def _square_target(size: float, far_field_size: int, offset: float = 0.0) -> np.ndarray:
    """远场上的方形目标（峰值 1）。``offset`` 是中心的相对偏移。"""
    target = np.zeros((far_field_size, far_field_size), dtype=np.float64)
    half = size / 2.0
    cx = far_field_size / 2.0 + offset * far_field_size
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    x0, x1 = max(0, x0), min(far_field_size, x1)
    target[x0:x1, x0:x1] = 1.0
    return target


def _apply_detector(
    frame_adu: np.ndarray,
    exposure: float,
    *,
    full_well: float = 100.0,
    black_level: float = 0.0,
    bit_depth: int = 12,
) -> np.ndarray:
    """在孪生的帧外面套一层探测器：曝光 + 上下截断 + 量化。

    孪生自己已经把帧**峰值归一**到 ``peak_adu``，所以它不会过曝也不会欠曝；
    真实相机会。曝光系数 > 1 ⇒ 峰值撞上满阱（核心削顶，过曝）；
    < 1 ⇒ 信号落到黑电平以下（弱散斑被切掉，欠曝）。散粒噪声由孪生自己加
    （峰值处 σ = √peak_adu ADU），这里只补截断与量化。
    """
    scaled = np.asarray(frame_adu, dtype=np.float64) * float(exposure)
    scaled = np.clip(scaled, black_level, full_well)
    levels = float(2**int(bit_depth) - 1)
    return np.round(scaled * levels) / levels


def _classical_gs(
    operator,
    target_amp: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """在给定前向下跑经典 GS，返回瞳面相位。

    ``operator`` 是 :class:`~ml.gsnet.forward_ops.BenchTwinOperator`：它的
    ``forward`` 给复数远场、``back_propagate`` 回到瞳面。GS 需要远场的**相位**，
    而孪生只给强度，所以标签这一环必须走可微的算子。
    """
    phase = operator.warm_start(target_amp)
    for _ in range(iterations):
        far = operator.forward(None, phase)
        far_c = target_amp * torch.exp(1j * torch.angle(far))
        phase = torch.angle(operator.back_propagate(far_c))
    return phase


def _make_training_set(
    operator,
    source: torch.Tensor,
    far_field_size: int,
    n_samples: int,
    target_side: float,
    gs_iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """合成 (source, target, gt_phase) 训练对。

    * ``source``   —— 瞳面照明（孪生自己的入射幅度）。
    * ``target``   —— **设计**的远场目标（方形）。它不是实测帧：网络回答的是
      「给出这个目标，该下发什么相位」。
    * ``gt_phase`` —— 在**带误差的物理**下达成该目标的相位（GS 解）。实验系统的
      非理想因素正是经由标签进入网络的——不是经由输入目标。
    """
    rng = np.random.default_rng(seed)
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    gt_phases: list[torch.Tensor] = []
    for _ in range(n_samples):
        side = float(target_side) * float(rng.uniform(0.75, 1.25))
        offset = float(rng.uniform(-0.012, 0.012))
        target = torch.from_numpy(
            _square_target(side, far_field_size, offset)
        ).to(torch.float32)[None, None]
        target_amp = torch.sqrt(target.clamp_min(0.0) + 1e-12)
        phase = _classical_gs(operator, target_amp, gs_iterations)
        sources.append(source[0])
        targets.append(target[0])
        gt_phases.append(phase[0, 0].clone())
    return (
        torch.stack(sources),
        torch.stack(targets),
        torch.stack(gt_phases)[:, None],
    )


@dataclass
class ArmBConfig:
    """臂 B 的配置（网络换前向）。"""

    region: int
    region_span_px: float
    far_field_padding: int
    n_train: int
    n_eval: int
    epochs: int
    metric_side: int
    target_side: int
    seed: int
    noise_adu: float
    num_layers: int = 8
    base_channels: int = 24
    w_shaping: float = 5.0
    warmup_epochs: int = 3
    lr: float = 1e-4


def _evaluate_network(
    model: FourierGSNet,
    twin: BenchTwin,
    source: torch.Tensor,
    target: torch.Tensor,
    metric_side: int,
    target_side: int,
) -> dict[str, Any]:
    """把网络预测的相位送进**孪生**（加噪帧），再与设计目标比较。"""
    model.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for i in range(source.shape[0]):
            pred = model(source[i : i + 1], target[i : i + 1])
            measured = prepare_frame(
                np.asarray(twin.frame(pred[0, 0].numpy().astype(np.float64)))
            )
            ideal = target[i, 0].numpy().astype(np.float64)
            rows.append(frame_metrics(measured, ideal, metric_side, target_side))
    return {key: _agg([row[key] for row in rows]) for key in METRIC_KEYS}


def _shaping_loss(
    operator, source_amp: torch.Tensor, phase: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """1 - Pearson（用**本臂自己的前向**算）。

    远场网格很大，噪声压过信号，所以相关性只在**目标方框**上算。
    """
    far = operator.forward(source_amp, phase)
    intensity = far.real**2 + far.imag**2
    box = int(operator.far_side)
    half = max(1, min(target.shape[-1], box) // 2)
    centre = intensity.shape[-1] // 2
    p = intensity[..., centre - half : centre + half, centre - half : centre + half]
    t = target[..., centre - half : centre + half, centre - half : centre + half]
    p = p.flatten(1)
    t = t.flatten(1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    corr = (p * t).sum(dim=1) / (
        torch.sqrt((p**2).sum(dim=1) * (t**2).sum(dim=1)) + 1e-6
    )
    return torch.mean(1.0 - corr)


def _train_arm(
    operator,
    source: torch.Tensor,
    target: torch.Tensor,
    gt_phase: torch.Tensor,
    cfg: ArmBConfig,
) -> FourierGSNet:
    """一个训练臂：固定 seed、固定预算。

    损失 = 相位项（对 GS 解）+ 整形项（用**本臂自己的前向**把预测相位推到
    目标上）。整形项的权重先小后大：相关性项在预测接近 0 时病态，一上来
    就给满权重会把训练带炸。
    """
    torch.manual_seed(cfg.seed)
    model = FourierGSNet(
        num_layers=cfg.num_layers,
        base_channels=cfg.base_channels,
        shared_weights=True,
        forward_op=operator,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=max(1, cfg.epochs // 4), gamma=0.4
    )
    n = source.shape[0]
    batch = 16
    source_amp = torch.sqrt(source.clamp_min(0.0) + 1e-12)
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            pred = model(source[idx], target[idx])
            diff = torch.remainder(pred - gt_phase[idx] + torch.pi, 2 * torch.pi) - torch.pi
            phase_loss = torch.nn.functional.mse_loss(diff, torch.zeros_like(diff))
            weight = cfg.w_shaping * min(
                1.0, (epoch + 1) / max(1, cfg.warmup_epochs)
            )
            loss = phase_loss + weight * _shaping_loss(
                operator, source_amp[idx], pred, target[idx]
            )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        scheduler.step()
    return model


def run_arm_b(cfg: ArmBConfig, out_dir: Path) -> dict[str, Any]:
    """网络换前向：内部前向用**孪生的链**（带误差），标签在同一物理下求。

    上下界：
    * ``flat``      —— 零相位（下发平场）;
    * ``gs_ceiling`` —— GS 在**带误差物理**下的解本身（**经典算法达到的水平，
      不是严格上限**：网络直接优化整形目标，可以超过它 —— 实测在带灰度→振幅
      耦合的孪生上网络 composite 0.3101 > GS 0.2766）。
    """
    region = cfg.region
    far_field = region * cfg.far_field_padding
    twin = build_twin(
        cfg.seed,
        region=region,
        region_span_px=cfg.region_span_px,
        far_field_padding=cfg.far_field_padding,
        noise_adu=cfg.noise_adu,
        randomize=True,
    )
    operator = BenchTwinOperator(twin)
    source = torch.from_numpy(
        np.asarray(twin.pupil_amplitude(), dtype=np.float32)
    )[None, None]

    source_all, target_all, gt_all = _make_training_set(
        operator,
        source,
        far_field,
        cfg.n_train + cfg.n_eval,
        cfg.target_side,
        gs_iterations=20,
        seed=cfg.seed,
    )
    train = slice(0, cfg.n_train)
    val = slice(cfg.n_train, cfg.n_train + cfg.n_eval)

    summary: dict[str, Any] = {}
    model = _train_arm(
        operator, source_all[train], target_all[train], gt_all[train], cfg
    )
    summary["calibrated"] = _evaluate_network(
        model, twin, source_all[val], target_all[val], cfg.metric_side, cfg.target_side
    )

    # 上下界
    flat_rows: list[dict[str, float]] = []
    for i in range(cfg.n_eval):
        zero = torch.zeros(1, 1, region, region)
        measured = prepare_frame(
            np.asarray(twin.frame(zero[0, 0].numpy().astype(np.float64)))
        )
        ideal = target_all[val][i, 0].numpy().astype(np.float64)
        flat_rows.append(
            frame_metrics(measured, ideal, cfg.metric_side, cfg.target_side)
        )
    summary["flat"] = {key: _agg([row[key] for row in flat_rows]) for key in METRIC_KEYS}

    ceil_rows: list[dict[str, float]] = []
    for i in range(cfg.n_eval):
        measured = prepare_frame(
            np.asarray(twin.frame(gt_all[val][i, 0].numpy().astype(np.float64)))
        )
        ideal = target_all[val][i, 0].numpy().astype(np.float64)
        ceil_rows.append(
            frame_metrics(measured, ideal, cfg.metric_side, cfg.target_side)
        )
    summary["gs_ceiling"] = {
        key: _agg([row[key] for row in ceil_rows]) for key in METRIC_KEYS
    }
    return summary


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
) -> tuple[bool, str, bool]:
    """P0 决策：两个判据，都报出来。

    **主判据（issue #165 的 P0 (3) 字面判据）= 前向模型不劣于基线。**
    原文写的是「24 帧同帧 36×36 网格上，校准后网络与物理 Zernike 基线
    （相关性 0.839、EE 0.642，2026-10-13/14 同帧结果）比，不劣于基线才允许
    进入 P1」；P0 (2) 也写明比较的量是「相关性、EE、NRMSE」。那组基线数字是
    **前向预测**的同帧结果，所以判据落在**前向模型**上：校准 2f 前向 vs 物理
    Zernike 基线（臂 A）。两个模型用的是同一组探针拟合出来的像差系数，
    差异只来自传播链本身的物理。

    **次判据（更严）= 网络层算出的相位能不能当闭环初值。**
    臂 B 的复合分数（`0.5·EE + 0.5/(1+CV)`，与 `slm_gs_refine` 同口径）
    不劣于物理 Zernike 基线。这一条不是 issue 原文写的，是本轮加的更严一档：
    前向模型准，不代表网络算出的相位就好。

    返回 (passed, line, fully_evaluated)：
    * ``passed`` —— 所有**已评估**的判据都通过；
    * ``fully_evaluated`` —— 没有判据处于 NOT_EVALUATED（否则整体结论不成立）。
    """
    parts: list[str] = []
    overall = summary_a["overall"]
    cal_a = overall["calibrated"]
    base_a = overall["zernike"]

    cal_c = cal_a["correlation_full"]["mean"]
    base_c = base_a["correlation_full"]["mean"]
    cal_ee = cal_a["encircled_energy"]["mean"]
    base_ee = base_a["encircled_energy"]["mean"]
    cal_n = cal_a["nrmse_full"]["mean"]
    base_n = base_a["nrmse_full"]["mean"]
    forward_ok = bool(
        cal_c >= base_c
        and cal_ee >= base_ee - ee_tolerance
        and cal_n <= base_n + ee_tolerance
    )
    parts.append(
        f"forward[issue(3)]: corr {cal_c:.4f} vs {base_c:.4f} | "
        f"EE {cal_ee:.4f} vs {base_ee:.4f} | NRMSE {cal_n:.4f} vs {base_n:.4f} "
        f"-> {'OK' if forward_ok else 'WORSE'}"
    )
    passed = forward_ok
    fully_evaluated = True

    if summary_b:
        cal = summary_b.get("calibrated")
        base = summary_b.get("zernike_baseline")
        ideal = summary_b.get("ideal")
        if cal and base:
            network_ok = bool(
                cal["composite"]["mean"] >= base["composite"]["mean"]
            )
            parts.append(
                f"network: composite {cal['composite']['mean']:.4f} vs "
                f"{base['composite']['mean']:.4f} | "
                f"EE {cal['encircled_energy']['mean']:.4f} vs "
                f"{base['encircled_energy']['mean']:.4f} | "
                f"CV {cal['cv_target_box']['mean']:.4f} vs "
                f"{base['cv_target_box']['mean']:.4f} "
                f"-> {'OK' if network_ok else 'WORSE'}"
            )
            passed = bool(passed and network_ok)
            if ideal:
                parts.append(
                    f"| ideal-forward composite={ideal['composite']['mean']:.4f} "
                    f"corr_full={ideal['correlation_full']['mean']:.4f}"
                )
    else:
        fully_evaluated = False
        parts.append("network: NOT_EVALUATED (arm B not run)")

    parts.append(
        f"|| forward corr vs ideal: {cal_c:.4f} vs "
        f"{overall['ideal']['correlation_full']['mean']:.4f}"
    )

    if not fully_evaluated:
        line = (
            "P0_DECISION: NOT_EVALUATED  (some criterion was not evaluated)  "
            + "  ".join(parts)
        )
        return False, line, False
    line = f"P0_DECISION: {'PASS' if passed else 'FAIL'}  " + "  ".join(parts)
    return passed, line, True


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
    parser.add_argument("--grid", type=int, default=64, help="瞳面模型网格边长（孪生的 region）")
    parser.add_argument(
        "--far-field-padding", type=int, default=8, help="远场补零因子（孪生的 far_field_padding）"
    )
    parser.add_argument(
        "--region-span-px", type=float, default=900.0, help="瞳面网格覆盖的面板边长（px）"
    )
    parser.add_argument(
        "--noise-adu", type=float, default=2.0, help="相机读出噪声（ADU）；0 = 确定性"
    )
    parser.add_argument("--frames", type=int, default=8, help="每个相位族的帧数（总帧数 = 3×）")
    parser.add_argument("--metric-side", type=int, default=36, help="指标网格边长")
    parser.add_argument(
        "--target-side", type=int, default=24, help="远场方形目标边长（模型采样点）"
    )
    parser.add_argument("--epochs", type=int, default=40, help="每个网络臂的训练轮数")
    parser.add_argument("--n-train", type=int, default=96, help="训练样本数")
    parser.add_argument("--n-eval", type=int, default=24, help="评估样本数")
    parser.add_argument("--num-layers", type=int, default=8, help="GS 展开层数")
    parser.add_argument("--base-channels", type=int, default=24, help="CNN 通道宽度")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--skip-network", action="store_true", help="只跑臂 A")
    parser.add_argument("--decision-only", action="store_true", help="只打印决策行")
    parser.add_argument(
        "--decision-from",
        type=Path,
        default=None,
        help=(
            "从已保存的 results.json 重新推导决策（不重跑实验）。"
            "判据定义改过之后，用它让结论与产物保持一致。"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.decision_from is not None:
        saved = json.loads(args.decision_from.read_text(encoding="utf-8"))
        passed, line, fully_evaluated = decide(
            saved["arm_a_forward_fidelity"],
            saved.get("arm_b_network") or None,
        )
        saved.setdefault("decision", {})
        saved["decision"].update(
            {"passed": passed, "evaluated": fully_evaluated, "line": line}
        )
        args.decision_from.write_text(
            json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(line)
        if not fully_evaluated:
            return 3
        return 0 if passed else 2

    if args.decision_from is not None:
        saved = json.loads(args.decision_from.read_text(encoding="utf-8"))
        passed, line, fully_evaluated = decide(
            saved["arm_a_forward_fidelity"],
            saved.get("arm_b_network") or None,
        )
        saved.setdefault("decision", {})
        saved["decision"].update(
            {"passed": passed, "evaluated": fully_evaluated, "line": line}
        )
        args.decision_from.write_text(
            json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(line)
        if not fully_evaluated:
            return 3
        return 0 if passed else 2

    out_dir = args.out
    if out_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "data" / "debug" / "gsnet_2f_calib" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_a_cfg = ArmAConfig(
        region=args.grid,
        region_span_px=args.region_span_px,
        far_field_padding=args.far_field_padding,
        frames_per_family=args.frames,
        metric_side=args.metric_side,
        target_side=args.target_side,
        seed=args.seed,
        noise_adu=args.noise_adu,
    )
    summary_a = run_arm_a(arm_a_cfg, out_dir)
    plot_arm_a(summary_a, out_dir / "forward_fidelity.png")

    summary_b: dict[str, Any] = {}
    if not args.skip_network:
        arm_b_cfg = ArmBConfig(
            region=args.grid,
            region_span_px=args.region_span_px,
            far_field_padding=args.far_field_padding,
            n_train=args.n_train,
            n_eval=args.n_eval,
            epochs=args.epochs,
            metric_side=args.metric_side,
            target_side=args.target_side,
            seed=args.seed,
            noise_adu=args.noise_adu,
            num_layers=args.num_layers,
            base_channels=args.base_channels,
        )
        summary_b = run_arm_b(arm_b_cfg, out_dir)
        if summary_b:
            plot_arm_b(summary_b, out_dir / "network_before_after.png")

    passed, line, fully_evaluated = decide(summary_a, summary_b or None)
    results = {
        "geometry": {
            "region": args.grid,
            "region_span_px": args.region_span_px,
            "far_field_padding": args.far_field_padding,
            "far_field_size": args.grid * args.far_field_padding,
            "noise_adu": args.noise_adu,
        },
        "config": {
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
            "evaluated": fully_evaluated,
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
    # 退出码 0 = 判据全部通过；2 = 有已评估的判据不通过；
    # 3 = 有判据没被评估（整体结论不成立）。2 与 3 都**不是**成功。
    if not fully_evaluated:
        return 3
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
