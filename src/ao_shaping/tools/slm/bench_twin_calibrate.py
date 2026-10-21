"""数字孪生的参数反演 (标定) 模块。

本模块**复用** :mod:`ao_shaping.drivers.sim.bench_twin` 的参数化 2f 数字孪生
核心 (不重写传播链), 在其之上加一层**反演**: 给定若干组「(命令相位, 实测
画面)」配对, 反推出孪生配置里的误差参数 (SLM LUT 非线性 / 面板静态相差 /
相机离焦 / 光路倾斜), 并给出**算法级预补偿** (把命令相位改成「让实际相位等于
命令相位」的那一张相位)。

两阶段反演
----------
* **Stage 1 (全局)** —— ``scipy.optimize.differential_evolution`` 粗调
  ``lut_gamma`` / ``lut_phase_max_rad`` / ``defocus_m`` (全局量)。
* **Stage 2 (空间)** —— ``scipy.optimize.minimize(method="L-BFGS-B")`` 精调
  面板 Zernike 系数 / 光路倾斜 / 透镜 Zernike 系数。

Stage 2 以 Stage 1 的解为初值, 且**同时放开 Stage 1 的粗参数**。原因:
面板 Noll 4 (离焦) 与相机离焦在远场上**近似简并** —— 瞳面二次相位与焦面
角谱离焦逐位等价 (见 ``bench_twin.apply_defocus`` 的说明), 只精调空间参数
时这条简并方向无法被打破, 拟合会把二者的份额固定在 Stage 1 的偶然落点上。

损失函数 (不使用纯 MSE)
-----------------------
每个样本 (命令相位 → 实测画面) 的单项损失由三项加权而成::

    L = w_ssim·(1 − SSIM) + w_energy·((Σsim − Σmeas)/Σmeas)² + w_corr·(1 − r)

* **空间项 SSIM** —— 结构相似性。优先用 ``skimage.metrics.structural_similarity``
  (``scikit-image`` 是本仓库的**已声明依赖**, 见 ``pyproject.toml``); 若运行环境
  没有 skimage, 自动回落到**自实现**的 numpy 高斯窗 SSIM (窗长 7、σ=1.5,
  反射填充), 回落时 ``ssim_backend()`` 返回 ``"numpy"``。
* **能量项** —— 总能量一致性, 取**未归一化**的 ADU 积分之比, 因此它对实测
  画面的**绝对标度**敏感 (实测画面整体缩放 20% ⇒ 能量项变大)。
  ⚠️ 因此仿真与实测必须**同一 ADU 标度**: ``fit_twin`` 默认把孪生的
  ``peak_adu`` 取为实测峰值的**中位数** (相机的 ADU 标度), 使二者可比。
* **相关项** —— peak 归一化后的 Pearson 相关系数 r, 取 ``1 − r``。

⚠️ **不做逐帧重新居中**: 图像只做 **peak 归一化**, 绝不按 argmax 重新居中 ——
平移误差 (光路倾斜) 正是要反演的量, 重新居中会把它藏掉。

本模块是**纯仿真/离线层**: 不 import 任何硬件驱动包, 不做设备 I/O。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize

from ao_shaping.drivers.sim.bench_twin import (
    BenchTwin,
    BenchTwinConfig,
    TwinErrorModel,
)
from loguru import logger

__all__ = [
    "TwinSample",
    "TwinFitResult",
    "twin_loss",
    "fit_twin",
    "save_fit",
    "load_fit",
    "compensated_phase",
    "parameter_vector",
    "apply_parameter_vector",
    "simulate_sample",
    "estimate_flux_scale",
    "ssim_backend",
    "DEFAULT_LOSS_WEIGHTS",
    "DEFAULT_BOUNDS",
]

# ---------------------------------------------------------------------------
# SSIM 后端 (skimage 优先, 缺失时自实现)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - 取决于运行环境是否装了 scikit-image
    from skimage.metrics import structural_similarity as _skimage_ssim

    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover - 回落到 numpy 实现
    _skimage_ssim = None
    _HAS_SKIMAGE = False

# 自实现 SSIM 的窗参数 (仅在 skimage 不可用时生效)
_NUMPY_SSIM_WIN = 7
_NUMPY_SSIM_SIGMA = 1.5


def ssim_backend() -> str:
    """当前使用的 SSIM 实现: ``"skimage"`` 或 ``"numpy"`` (自实现)。"""
    return "skimage" if _HAS_SKIMAGE else "numpy"


# ---------------------------------------------------------------------------
# 损失权重与默认边界
# ---------------------------------------------------------------------------

#: 联合损失的默认权重 (SSIM / 能量 / 相关)。
DEFAULT_LOSS_WEIGHTS: dict[str, float] = {"ssim": 1.0, "energy": 10.0, "corr": 1.0}

#: 各参数组的默认边界 (物理单位)。可被 ``fit_twin(bounds=...)`` 覆盖。
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "lut_gamma": (0.5, 2.5),
    "lut_phase_max_rad": (math.pi, 2.5 * math.pi),
    "defocus_m": (-5.0e-4, 5.0e-4),
    "slm_zernike": (-1.0, 1.0),
    "tilt": (-0.05, 0.05),
    "lens_zernike": (-1.0, 1.0),
}

#: 优化器内部使用的**量纲缩放** (把参数拉到 O(1), 否则 L-BFGS-B 的有限差分
#: 步长与收敛判据在不同量级的参数上无法兼顾)。
_PARAM_SCALES: dict[str, float] = {
    "lut_gamma": 1.0,
    "lut_phase_max_rad": 1.0,
    "defocus_m": 1.0e-4,
    "slm_zernike": 1.0,
    "tilt": 1.0e-4,
    "lens_zernike": 1.0,
}

#: Stage 2 中心差分梯度的步长 (在**缩放空间**里)。实测: 步长 1e-3 会被目标
#: 函数在 ~1e-5 量级上的抖动 (peak 归一化时 argmax 换像素) 淹没, 梯度变成噪声,
#: L-BFGS-B 停在错误的盆地里; 步长 1e-2 时同样的合成真值可复原到 ~1e-6 的损失。
_GRAD_STEP = 1.0e-2

#: 参数种类 → 边界字典里的键 (``tilt_x``/``tilt_y`` 共用 ``tilt``)。
_BOUND_KEY: dict[str, str] = {
    "lut_gamma": "lut_gamma",
    "lut_phase_max_rad": "lut_phase_max_rad",
    "defocus_m": "defocus_m",
    "slm_zernike": "slm_zernike",
    "tilt_x_rad": "tilt",
    "tilt_y_rad": "tilt",
    "lens_zernike": "lens_zernike",
}


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------


@dataclass
class TwinSample:
    """一次 (命令相位, 实测画面) 配对。

    Attributes:
        phase_rad: 命令相位 (瞳面网格, raw 弧度, ``region × region``)。
        image: 实测画面 (与孪生输出同网格; 绝对 ADU 标度应与孪生的
            ``peak_adu`` 一致, 否则能量项会带一个常数偏置)。
        label: 可选标识 (便于在报告里指认样本)。
    """

    phase_rad: np.ndarray
    image: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        self.phase_rad = np.asarray(self.phase_rad, dtype=np.float64)
        self.image = np.asarray(self.image, dtype=np.float64)


@dataclass
class TwinFitResult:
    """一次反演的结果。

    Attributes:
        config: 拟合后的孪生配置。
        loss: 最终联合损失 (各样本单项损失的均值)。
        stages: 每阶段的损失、参数量、评估次数等 (全部为 float)。
        per_sample: 每个样本的 SSIM / 相关 / 能量比 / 残差明细。
        n_evaluations: 目标函数的总评估次数 (每次评估 = 全部样本的一遍前向)。
    """

    config: BenchTwinConfig
    loss: float
    stages: dict[str, float] = field(default_factory=dict)
    per_sample: list[dict] = field(default_factory=list)
    n_evaluations: int = 0

    def to_dict(self) -> dict:
        """导出为 JSON 可序列化的字典 (配置走 ``BenchTwin.to_dict``)。"""
        return {
            "config": BenchTwin(self.config).to_dict(),
            "loss": float(self.loss),
            "stages": {str(k): float(v) for k, v in self.stages.items()},
            "per_sample": [_jsonable(entry) for entry in self.per_sample],
            "n_evaluations": int(self.n_evaluations),
        }


def _jsonable(entry: Mapping[str, Any]) -> dict:
    """把一条明细转成 JSON 可序列化的字典 (数值 → float, 其余原样)。"""
    out: dict[str, Any] = {}
    for key, value in entry.items():
        if isinstance(value, (bool, str)) or value is None:
            out[str(key)] = value
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[str(key)] = float(value)
        else:
            out[str(key)] = str(value)
    return out


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------


def _peak_normalise(image: np.ndarray) -> np.ndarray:
    """峰值归一化 (峰值 1); 峰值非正时返回全零。

    ⚠️ 只做 peak 归一化, **不重新居中** —— 重新居中会把光路倾斜误差藏掉。
    """
    arr = np.asarray(image, dtype=np.float64)
    peak = float(arr.max()) if arr.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(arr)
    return arr / peak


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """两个同形数组的 Pearson 相关系数 (常数数组时返回 0)。"""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        return 0.0
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt(float(xc @ xc) * float(yc @ yc)))
    if denom <= 0.0:
        return 0.0
    return float((xc @ yc) / denom)


def _ssim_numpy(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """自实现的 numpy SSIM (高斯窗 7×7、σ=1.5、反射填充)。

    仅在运行环境没有 ``skimage`` 时使用。口径与 Wang et al. 2004 一致:
    ``C1 = (0.01·L)²``, ``C2 = (0.03·L)²``, 其中 ``L = data_range = 1``。
    """
    from scipy.ndimage import gaussian_filter

    a = np.asarray(image_a, dtype=np.float64)
    b = np.asarray(image_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"image shape mismatch: {a.shape} vs {b.shape}")
    if min(a.shape) < 3:
        return 0.0

    win = min(_NUMPY_SSIM_WIN, min(a.shape))
    if win % 2 == 0:
        win -= 1
    pad = (win - 1) // 2
    ap = np.pad(a, pad, mode="reflect")
    bp = np.pad(b, pad, mode="reflect")

    def _blur(arr: np.ndarray) -> np.ndarray:
        return gaussian_filter(arr, _NUMPY_SSIM_SIGMA, mode="reflect")[pad:-pad, pad:-pad]

    mu_a = _blur(ap)
    mu_b = _blur(bp)
    var_a = _blur(ap * ap) - mu_a * mu_a
    var_b = _blur(bp * bp) - mu_b * mu_b
    cov_ab = _blur(ap * bp) - mu_a * mu_b

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return float(np.mean(numerator / denominator))


def _ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """peak 归一化后的 SSIM (skimage 优先, 否则 numpy 自实现)。"""
    a = _peak_normalise(image_a)
    b = _peak_normalise(image_b)
    if a.shape != b.shape:
        raise ValueError(f"image shape mismatch: {a.shape} vs {b.shape}")
    if not _HAS_SKIMAGE:
        return _ssim_numpy(a, b)
    win = 7
    if min(a.shape) < win:
        win = min(a.shape)
        if win % 2 == 0:
            win -= 1
    if win < 3:
        return 0.0
    return float(_skimage_ssim(a, b, data_range=1.0, win_size=win))


def twin_loss(
    twin: BenchTwin,
    samples: Sequence[TwinSample],
    *,
    weights: Mapping[str, float] | None = None,
) -> tuple[float, list[dict]]:
    """联合损失 + 每样本明细。

    损失 = 各项样本损失的**均值**::

        L = w_ssim·(1 − SSIM) + w_energy·((Σsim − Σmeas)/Σmeas)² + w_corr·(1 − r)

    * SSIM 与相关系数都在 **peak 归一化**后的图像上计算;
    * 能量项用**未归一化**的 ADU 积分 (因此对实测画面的绝对标度敏感);
    * 绝不做逐帧 argmax 重新居中。

    Args:
        twin: 参与比较的孪生 (取其 ``far_field`` 作为仿真画面)。
        samples: (命令相位, 实测画面) 配对序列。
        weights: 权重覆盖, 键为 ``"ssim"`` / ``"energy"`` / ``"corr"``。

    Returns:
        ``(联合损失, 每样本明细)``。明细含 ``ssim`` / ``corr`` / ``energy_rel``
        (带符号的相对能量偏差) / ``energy`` (平方) / ``loss`` / ``residual_rms``。

    Raises:
        ValueError: 样本为空, 或某样本的实测画面与仿真画面形状不一致。
    """
    items = list(samples)
    if not items:
        raise ValueError("twin_loss needs at least one sample")

    w = dict(DEFAULT_LOSS_WEIGHTS)
    if weights:
        w.update({str(k): float(v) for k, v in weights.items()})

    total = 0.0
    details: list[dict] = []
    for sample in items:
        sim = np.asarray(twin.far_field(sample.phase_rad), dtype=np.float64)
        meas = np.asarray(sample.image, dtype=np.float64)
        if sim.shape != meas.shape:
            raise ValueError(
                f"measured image shape {meas.shape} != simulated {sim.shape}"
            )

        ssim_value = _ssim(meas, sim)
        sim_n = _peak_normalise(sim)
        meas_n = _peak_normalise(meas)
        corr = _pearson(sim_n, meas_n)

        sum_sim = float(sim.sum())
        sum_meas = float(meas.sum())
        if sum_meas > 0.0:
            energy_rel = (sum_sim - sum_meas) / sum_meas
        else:
            energy_rel = 0.0 if sum_sim <= 0.0 else 1.0

        loss = (
            w["ssim"] * (1.0 - ssim_value)
            + w["energy"] * energy_rel * energy_rel
            + w["corr"] * (1.0 - corr)
        )
        total += loss
        details.append(
            {
                "label": sample.label,
                "ssim": float(ssim_value),
                "corr": float(corr),
                "energy_rel": float(energy_rel),
                "energy": float(energy_rel * energy_rel),
                "loss": float(loss),
                "residual_rms": float(np.sqrt(np.mean((sim_n - meas_n) ** 2))),
                "sum_sim": float(sum_sim),
                "sum_meas": float(sum_meas),
            }
        )

    return float(total / len(items)), details


# ---------------------------------------------------------------------------
# 参数向量 ↔ 配置
# ---------------------------------------------------------------------------


def _layout(config: BenchTwinConfig, which: str) -> list[tuple[str, Any]]:
    """参数组的布局: ``[(kind, key), ...]``。

    * ``coarse`` —— ``lut_gamma`` / ``lut_phase_max_rad`` / ``defocus_m``;
    * ``fine`` —— 面板 Zernike (按 Noll 序升序) + ``tilt_x_rad`` + ``tilt_y_rad``
      + 透镜 Zernike (按 Noll 序升序)。

    Zernike 的**模式集合**由配置里已有的 ``slm_zernike`` / ``lens_zernike``
    的键决定 —— 调用方用「配置里写哪些模式」来声明「要反演哪些模式」。
    """
    errors = config.errors
    if which == "coarse":
        return [
            ("lut_gamma", None),
            ("lut_phase_max_rad", None),
            ("defocus_m", None),
        ]
    if which == "fine":
        entries: list[tuple[str, Any]] = []
        entries += [("slm_zernike", int(k)) for k in sorted(errors.slm_zernike)]
        entries += [("tilt_x_rad", None), ("tilt_y_rad", None)]
        entries += [("lens_zernike", int(k)) for k in sorted(errors.lens_zernike)]
        return entries
    raise ValueError(f"unknown parameter set {which!r} (expected 'coarse' or 'fine')")


def _read_param(errors: TwinErrorModel, kind: str, key: Any) -> float:
    if kind == "lut_gamma":
        return float(errors.lut_gamma)
    if kind == "lut_phase_max_rad":
        return float(errors.lut_phase_max_rad)
    if kind == "defocus_m":
        return float(errors.defocus_m)
    if kind == "slm_zernike":
        return float(errors.slm_zernike[int(key)])
    if kind == "lens_zernike":
        return float(errors.lens_zernike[int(key)])
    if kind == "tilt_x_rad":
        return float(errors.tilt_x_rad)
    if kind == "tilt_y_rad":
        return float(errors.tilt_y_rad)
    raise ValueError(f"unknown parameter kind {kind!r}")


def _write_param(
    errors: TwinErrorModel, kind: str, key: Any, value: float
) -> None:
    if kind == "lut_gamma":
        errors.lut_gamma = float(value)
    elif kind == "lut_phase_max_rad":
        errors.lut_phase_max_rad = float(value)
    elif kind == "defocus_m":
        errors.defocus_m = float(value)
    elif kind == "slm_zernike":
        errors.slm_zernike[int(key)] = float(value)
    elif kind == "lens_zernike":
        errors.lens_zernike[int(key)] = float(value)
    elif kind == "tilt_x_rad":
        errors.tilt_x_rad = float(value)
    elif kind == "tilt_y_rad":
        errors.tilt_y_rad = float(value)
    else:
        raise ValueError(f"unknown parameter kind {kind!r}")


def _clone_config(config: BenchTwinConfig) -> BenchTwinConfig:
    """深拷贝配置 (误差字典与 LUT 表都复制, 不与原对象共享可变状态)。"""
    errors = config.errors
    cloned = TwinErrorModel(
        lut_gamma=errors.lut_gamma,
        lut_phase_max_rad=errors.lut_phase_max_rad,
        gray_levels=errors.gray_levels,
        lut_table=(
            None
            if errors.lut_table is None
            else np.array(errors.lut_table, dtype=np.float64)
        ),
        slm_zernike=dict(errors.slm_zernike),
        slm_zernike_n_max=errors.slm_zernike_n_max,
        slm_roughness_sigma_rad=errors.slm_roughness_sigma_rad,
        slm_roughness_seed=errors.slm_roughness_seed,
        defocus_m=errors.defocus_m,
        tilt_x_rad=errors.tilt_x_rad,
        tilt_y_rad=errors.tilt_y_rad,
        lens_zernike=dict(errors.lens_zernike),
        lens_zernike_n_max=errors.lens_zernike_n_max,
    )
    return replace(config, errors=cloned)


def _apply_layout(
    config: BenchTwinConfig, values: Sequence[float], layout: Sequence[tuple[str, Any]]
) -> BenchTwinConfig:
    """把一组物理参数写进配置的一个**副本** (不改动原配置)。"""
    values = list(values)
    if len(values) != len(layout):
        raise ValueError(
            f"parameter vector length {len(values)} != layout length {len(layout)}"
        )
    new_config = _clone_config(config)
    for (kind, key), value in zip(layout, values, strict=False):
        _write_param(new_config.errors, kind, key, float(value))
    return new_config


def parameter_vector(config: BenchTwinConfig, which: str) -> np.ndarray:
    """取出一组参数 (物理单位) 作为向量。

    Args:
        config: 孪生配置。
        which: ``"coarse"`` (全局量) 或 ``"fine"`` (空间量)。

    Returns:
        长度与该参数组布局一致的 float64 向量。

    Raises:
        ValueError: ``which`` 不是 ``"coarse"`` / ``"fine"``。
    """
    layout = _layout(config, which)
    return np.array(
        [_read_param(config.errors, kind, key) for kind, key in layout],
        dtype=np.float64,
    )


def apply_parameter_vector(
    config: BenchTwinConfig, values: Sequence[float], which: str
) -> BenchTwinConfig:
    """把一组参数 (物理单位) 写进配置的**副本**。

    Args:
        config: 原配置 (不被修改)。
        values: 参数向量, 顺序与 :func:`parameter_vector` 一致。
        which: ``"coarse"`` 或 ``"fine"``。

    Returns:
        新的 :class:`BenchTwinConfig`。

    Raises:
        ValueError: 向量长度与布局不符, 或 ``which`` 非法。
    """
    return _apply_layout(config, values, _layout(config, which))


def _clip_layout(
    config: BenchTwinConfig,
    which: str,
    bounds: Sequence[tuple[float, float]],
) -> BenchTwinConfig:
    """把配置里某参数组的值裁剪到给定边界内 (返回副本)。"""
    values = parameter_vector(config, which)
    clipped = [min(max(value, lo), hi) for value, (lo, hi) in zip(values, bounds, strict=False)]
    return _apply_layout(config, clipped, _layout(config, which))


def _resolve_bounds(
    config: BenchTwinConfig,
    which: str,
    bounds: Mapping[str, Any] | None,
) -> list[tuple[float, float]]:
    """解析某参数组的边界 (物理单位)。

    ``bounds`` 的形状::

        {
            "coarse": {"lut_gamma": (lo, hi), "lut_phase_max_rad": (lo, hi),
                       "defocus_m": (lo, hi)},
            "fine": {"slm_zernike": {4: (lo, hi), ...} | (lo, hi),
                     "tilt": (lo, hi),
                     "lens_zernike": {4: (lo, hi), ...} | (lo, hi)},
        }

    未给出的项沿用 :data:`DEFAULT_BOUNDS`。
    """
    layout = _layout(config, which)
    spec = None if bounds is None else bounds.get(which)
    out: list[tuple[float, float]] = []
    for kind, _key in layout:
        bound_key = _BOUND_KEY[kind]
        lo, hi = DEFAULT_BOUNDS[bound_key]
        if isinstance(spec, Mapping):
            value = spec.get(bound_key)
            if value is not None:
                if isinstance(value, Mapping):
                    per_mode = value.get(_key)
                    if per_mode is not None:
                        lo, hi = _as_pair(per_mode)
                else:
                    lo, hi = _as_pair(value)
        out.append((float(lo), float(hi)))
    return out


def _as_pair(value: Any) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        raise ValueError(f"bound must be a (lo, hi) pair, got {value!r}")
    lo, hi = value
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        raise ValueError(f"bound hi < lo: {(lo, hi)}")
    return lo, hi


# ---------------------------------------------------------------------------
# 采样辅助
# ---------------------------------------------------------------------------


def simulate_sample(
    twin: BenchTwin, phase_rad: np.ndarray, label: str = ""
) -> TwinSample:
    """用孪生生成一条 (命令相位, 画面) 样本 (合成真值用)。"""
    phase = np.asarray(phase_rad, dtype=np.float64)
    return TwinSample(phase_rad=phase, image=twin.far_field(phase), label=label)


def estimate_flux_scale(
    samples: Sequence[TwinSample], config: BenchTwinConfig
) -> float:
    """估计「实测 ADU 标度 → 孪生 ``peak_adu`` 标度」的换算因子。

    取实测峰值中位数与 ``config.peak_adu`` 之比。实测峰值全为 0 时返回 1.0。

    ⚠️ 与 :func:`fit_twin` 的自动处理**互为逆操作**: ``fit_twin`` 默认把孪生的
    ``peak_adu`` 改成实测峰值的中位数 (等价于把实测乘上本函数返回的因子)。
    只有在调用方想**保留**输入配置的 ``peak_adu``、转而去缩放实测画面时,
    才需要显式用到本函数。
    """
    peaks = [float(np.max(s.image)) for s in samples]
    peaks = [p for p in peaks if p > 0.0]
    if not peaks:
        return 1.0
    reference = float(config.peak_adu)
    if reference <= 0.0:
        return 1.0
    return reference / float(np.median(peaks))


# ---------------------------------------------------------------------------
# 反演
# ---------------------------------------------------------------------------


def fit_twin(
    samples: Sequence[TwinSample],
    config: BenchTwinConfig,
    *,
    bounds: Mapping[str, Any] | None = None,
    stage1: bool = True,
    stage2: bool = True,
    maxiter: int = 12,
    popsize: int = 8,
    maxiter_fine: int = 60,
    seed: int = 0,
    verbose: bool = False,
    peak_adu: float | None = None,
) -> TwinFitResult:
    """两阶段反演: 粗调全局量 + 精调空间量。

    Args:
        samples: (命令相位, 实测画面) 配对序列。
        config: 初始配置 (其几何参数被当作**已知量**; 误差参数是待反演量)。
            误差字典里已经写好的 Zernike 模式集合即「要反演的模式集合」。
        bounds: 边界覆盖, 形状见 :func:`_resolve_bounds` 的说明。
        stage1: 是否跑 Stage 1 (``differential_evolution`` 粗调)。
        stage2: 是否跑 Stage 2 (``L-BFGS-B`` 精调)。
        maxiter: Stage 1 的迭代上限 (默认 12, 配合 ``popsize`` 控制总时长)。
        popsize: Stage 1 的种群规模乘数 (默认 8)。
        maxiter_fine: Stage 2 的迭代上限 (默认 60)。
        seed: 随机种子 (DE 的种群初始化); 同 seed ⇒ 逐位可复现。
        verbose: 是否打印每阶段的损失下降。
        peak_adu: 覆盖拟合时孪生的 ``peak_adu`` (相机的 ADU 标度)。默认取
            **实测峰值的中位数**, 使仿真与实测同标度 (能量项才有意义)。

    Returns:
        :class:`TwinFitResult`。拟合过程**确定性** (噪声被关掉), ``loss``
        是各样本单项损失的均值。

    Raises:
        ValueError: 样本为空, 或样本的相位/画面形状不一致。
    """
    items = list(samples)
    if not items:
        raise ValueError("fit_twin needs at least one sample")

    base = _clone_config(config)
    # 反演必须确定性: 关掉噪声, 前向只走 far_field (无噪)。
    base.noise_adu = 0.0
    if peak_adu is None:
        peaks = [float(np.max(s.image)) for s in items]
        peaks = [p for p in peaks if p > 0.0]
        base.peak_adu = float(np.median(peaks)) if peaks else float(base.peak_adu)
    else:
        base.peak_adu = float(peak_adu)

    region = int(base.region)
    for index, sample in enumerate(items):
        if np.asarray(sample.phase_rad).shape != (region, region):
            raise ValueError(
                f"sample #{index} phase shape "
                f"{np.asarray(sample.phase_rad).shape} != ({region}, {region})"
            )

    counter = {"n": 0}

    def evaluate(candidate: BenchTwinConfig) -> tuple[float, list[dict]]:
        twin = BenchTwin(candidate)
        loss, details = twin_loss(twin, items)
        counter["n"] += 1
        return loss, details

    coarse_layout = _layout(base, "coarse")
    fine_layout = _layout(base, "fine")
    coarse_bounds = _resolve_bounds(base, "coarse", bounds)
    fine_bounds = _resolve_bounds(base, "fine", bounds)
    joint_layout = coarse_layout + fine_layout
    joint_bounds = coarse_bounds + fine_bounds

    # 起始点必须落在边界内 (调用方可能给出越界的初值) —— 否则拟合结果可能
    # 停在越界的初值上, 违反「反演结果始终落在 bounds 内」。
    base = _clip_layout(base, "coarse", coarse_bounds)
    base = _clip_layout(base, "fine", fine_bounds)

    def scales_for(layout: Sequence[tuple[str, Any]]) -> np.ndarray:
        return np.array(
            [_PARAM_SCALES[_BOUND_KEY[kind]] for kind, _ in layout], dtype=np.float64
        )

    def to_scaled(values: Sequence[float], layout: Sequence[tuple[str, Any]]):
        return np.asarray(values, dtype=np.float64) / scales_for(layout)

    def to_physical(y: np.ndarray, layout: Sequence[tuple[str, Any]]):
        return np.asarray(y, dtype=np.float64) * scales_for(layout)

    def objective(y: np.ndarray, layout: Sequence[tuple[str, Any]]) -> float:
        candidate = _apply_layout(base, to_physical(y, layout), layout)
        return evaluate(candidate)[0]

    stages: dict[str, float] = {}
    best_config = base
    best_loss, best_details = evaluate(base)
    stages["initial_loss"] = float(best_loss)

    # ---------------- Stage 1: 全局量 (differential evolution) --------
    if stage1:
        y0 = to_scaled(parameter_vector(base, "coarse"), coarse_layout)
        scaled_bounds = [
            (lo / scale, hi / scale)
            for (lo, hi), scale in zip(coarse_bounds, scales_for(coarse_layout), strict=False)
        ]
        y0 = np.array(
            [min(max(value, lo), hi) for value, (lo, hi) in zip(y0, scaled_bounds, strict=False)],
            dtype=np.float64,
        )
        before = counter["n"]
        result = differential_evolution(
            objective,
            bounds=scaled_bounds,
            args=(coarse_layout,),
            maxiter=int(maxiter),
            popsize=int(popsize),
            seed=int(seed),
            workers=1,
            polish=False,
            init="latinhypercube",
            tol=0.0,
            updating="immediate",
        )
        candidate = _apply_layout(
            base, to_physical(result.x, coarse_layout), coarse_layout
        )
        loss, details = evaluate(candidate)
        stages["stage1_loss"] = float(loss)
        stages["n_evaluations_stage1"] = float(counter["n"] - before)
        stages["n_params_coarse"] = float(len(coarse_layout))
        if loss < best_loss:
            best_config, best_loss, best_details = candidate, loss, details
        if verbose:
            logger.info(
                f"[fit_twin] stage1: loss {stages['initial_loss']:.6g} "
                f"-> {stages['stage1_loss']:.6g} "
                f"({stages['n_evaluations_stage1']:.0f} evaluations)"
            )

    # ---------------- Stage 2: 空间量 + 粗量联合精调 (L-BFGS-B) -------
    if stage2:
        start = np.concatenate(
            [
                to_scaled(parameter_vector(best_config, "coarse"), coarse_layout),
                to_scaled(parameter_vector(best_config, "fine"), fine_layout),
            ]
        )
        scaled_bounds = [
            (lo / scale, hi / scale)
            for (lo, hi), scale in zip(joint_bounds, scales_for(joint_layout), strict=False)
        ]
        start = np.array(
            [min(max(value, lo), hi) for value, (lo, hi) in zip(start, scaled_bounds, strict=False)],
            dtype=np.float64,
        )

        def joint_objective(y: np.ndarray) -> float:
            return objective(y, joint_layout)

        def joint_gradient(y: np.ndarray) -> np.ndarray:
            """中心差分梯度 (缩放空间里的步长 ``_GRAD_STEP``)。"""
            grad = np.zeros_like(y)
            step = _GRAD_STEP
            for i in range(y.size):
                plus = y.copy()
                plus[i] += step
                minus = y.copy()
                minus[i] -= step
                grad[i] = (joint_objective(plus) - joint_objective(minus)) / (
                    2.0 * step
                )
            return grad

        # 多起点: Stage 1 的解 + 「初始配置」重启点。
        # ⚠️ 面板 Noll 4 与相机离焦近似简并, 只跑 Stage 1 (粗参数) 时 DE 很容易
        # 落进一个**联合**问题的假极小值 (实测: 该点的联合梯度≈0, L-BFGS-B
        # 一步不动)。保留初始配置作为重启点, 取损失最小者。
        starts: list[np.ndarray] = [start]
        if stage1:
            restart = np.concatenate(
                [
                    to_scaled(parameter_vector(base, "coarse"), coarse_layout),
                    to_scaled(parameter_vector(base, "fine"), fine_layout),
                ]
            )
            restart = np.array(
                [
                    min(max(value, lo), hi)
                    for value, (lo, hi) in zip(restart, scaled_bounds, strict=False)
                ],
                dtype=np.float64,
            )
            if not np.array_equal(restart, start):
                starts.append(restart)

        before = counter["n"]
        best_stage2: tuple[float, np.ndarray] = (float("inf"), start)
        for candidate_start in starts:
            result = minimize(
                joint_objective,
                x0=candidate_start,
                jac=joint_gradient,
                method="L-BFGS-B",
                bounds=scaled_bounds,
                options={
                    "maxiter": int(maxiter_fine),
                    "maxfun": int(maxiter_fine) * (2 * len(joint_layout) + 4),
                    # 目标函数在 ~1e-6 量级上有抖动 (peak 归一化时 argmax 换像素),
                    # 梯度噪声地板 ~1e-4 ⇒ 收敛判据不能按「梯度趋零」设, 否则永远
                    # 达不到; 用「相对下降」判据, 实测同一合成真值下 ftol=1e-8 与
                    # ftol=1e-14 给出**逐位相同**的最终参数 (迭代数 48 vs 54)。
                    "ftol": 1e-8,
                    "gtol": 1e-6,
                },
            )
            value = float(result.fun)
            if value < best_stage2[0]:
                best_stage2 = (value, np.asarray(result.x, dtype=np.float64))

        candidate = _apply_layout(
            base, to_physical(best_stage2[1], joint_layout), joint_layout
        )
        loss, details = evaluate(candidate)
        stages["stage2_loss"] = float(loss)
        stages["n_evaluations_stage2"] = float(counter["n"] - before)
        stages["n_params_fine"] = float(len(fine_layout))
        stages["n_params_joint"] = float(len(joint_layout))
        stages["n_restarts_stage2"] = float(len(starts))
        if loss < best_loss:
            best_config, best_loss, best_details = candidate, loss, details
        if verbose:
            logger.info(
                f"[fit_twin] stage2: loss "
                f"{stages.get('stage1_loss', stages['initial_loss']):.6g} "
                f"-> {stages['stage2_loss']:.6g} "
                f"({stages['n_evaluations_stage2']:.0f} evaluations, "
                f"{len(starts)} restarts)"
            )

    stages["final_loss"] = float(best_loss)
    stages["peak_adu"] = float(best_config.peak_adu)
    stages["ssim_backend_skimage"] = 1.0 if ssim_backend() == "skimage" else 0.0

    return TwinFitResult(
        config=best_config,
        loss=float(best_loss),
        stages=stages,
        per_sample=list(best_details),
        n_evaluations=int(counter["n"]),
    )


# ---------------------------------------------------------------------------
# 存取与预补偿
# ---------------------------------------------------------------------------


def save_fit(result: TwinFitResult, path: str | Path) -> Path:
    """把拟合结果写成 JSON。

    Args:
        result: 拟合结果。
        path: 目标文件路径 (父目录会自动创建)。

    Returns:
        写入的路径。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target


def load_fit(path: str | Path) -> TwinFitResult:
    """读回 :func:`save_fit` 写出的拟合结果。

    Args:
        path: JSON 文件路径。

    Returns:
        还原的 :class:`TwinFitResult` (配置经 ``BenchTwin.from_dict`` 复原)。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件内容不是合法的拟合结果字典。
    """
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "config" not in payload:
        raise ValueError(f"{source} does not contain a fit result (no 'config' key)")
    config = BenchTwin.from_dict(payload["config"]).config
    return TwinFitResult(
        config=config,
        loss=float(payload.get("loss", 0.0)),
        stages={str(k): float(v) for k, v in (payload.get("stages") or {}).items()},
        per_sample=list(payload.get("per_sample") or []),
        n_evaluations=int(payload.get("n_evaluations", 0)),
    )


def compensated_phase(result: TwinFitResult, phase_rad: np.ndarray) -> np.ndarray:
    """用拟合好的孪生做算法级预补偿, 返回应当下发的命令相位。

    补偿的**顺序** (与孪生的正向链严格相反)::

        正向:  cmd → (+面板静态相差) → (LUT 灰度量化与非线性) → 实际相位
        补偿:  cmd' = (LUT 反查(cmd)) − (面板静态相差)

    * **LUT 反查**: 解 ``灰度 = levels·(目标相位 / lut_phase_max_rad)^(1/gamma)``
      (有实测 LUT 表时改为查表插值), 再换算回「等效灰度相位」 ``2π·灰度/levels``;
    * **面板静态相差**: 面板 Zernike 在正向链里是**先加到命令上**再进 LUT 的,
      所以必须从命令里**先减掉** (顺序反了就会差一个面板相差)。

    残余误差: 灰度量化 (≤ 一个量化级, 约 ``gamma·2π/gray_levels`` 量级) 与
    面板粗糙度 (随机量, 不补偿) —— 后者需要 ``slm_roughness_sigma_rad > 0``
    才会出现。

    Args:
        result: 拟合结果 (提供拟合后的配置)。
        phase_rad: 期望的**实际**相位 (raw 弧度, 瞳面网格)。

    Returns:
        应当下发给 SLM 的命令相位 (raw 弧度)。
    """
    twin = BenchTwin(result.config)
    return twin.compensate(np.asarray(phase_rad, dtype=np.float64))
