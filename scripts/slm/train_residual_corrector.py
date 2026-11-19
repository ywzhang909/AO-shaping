# -*- coding: utf-8 -*-
"""在**冻结**的物理拟合之上训练两个事后残差校正器（operator / cnn）。

不访问任何设备；只读一次已录制的 dataset 与一次已完成的 fit 运行。

数据（实测字段，探测后记录）
---------------------------
数据集目录（``--dataset``）
    ``manifest.json`` 每行至少含 ``file`` / ``label`` / ``split`` /
    ``exposure_ms`` / ``spot_xy`` / ``peak``；``split`` 取
    ``train``（56 行）/ ``test``（12 行）/ ``calibration``（10 行）。
    ``sample_%04d.npz`` 的键为 ``raw_frames`` (3,1944,2592) uint8、
    ``frame`` (1944,2592) float64、``model_phase`` (128,128) float32、
    ``panel_phase`` (1200,1920)、``panel_gray`` (1200,1920) uint16、
    ``exposure_probe_frames``。本脚本只用 ``raw_frames`` 与 ``model_phase``，
    与 ``fit.load_dataset`` 一致：``model_phase`` 形状已等于 ``grid``（128），
    直接用，不再从 panel 重采样。
fit 目录（``--fit``）
    ``iteration_%04d.npz`` 只有一个键 ``prediction``，(68,161,161) float32，
    行序 = 参与拟合的行序（train 56 + test 12，按 manifest 的 ``valid()`` 顺序）。
    ``result.json`` 的 ``metrics`` 逐行给出 fit 自己算的指标，
    ``provenance.json`` 的 ``rows`` 给出同一行序的 label/split/file。
    本脚本用它做**硬校验**：重算的 baseline 指标必须与 ``result.json`` 一致，
    否则说明实测图像或窗口与 fit 不同，直接报错。

实测图像重建（与 fit 逐位一致）
    ``centre = spot_xy(reference_start)``（reference_start 在 calibration 行）；
    相机窗口 ``centre + mgrid[-span:span+1:2]``（span=160, step=2 → 161x161）；
    ``measured = mean(raw_frames)[window] / exposure_ms``（``radiometry.count_rate``）。

只用 ``split == "train"`` 建统计量、拟合与选轮；``test`` 只用于评估。

两个校正器（接口一致：``forward(prediction, phase) -> corrected``）
------------------------------------------------------------------
``operator``（物理约束，7 个系数，MLP ≈ 6.4k 参数）
    在小 MLP 上把相位 16x16 + 每行 RMS/peak 映射到 7 个系数：4 个高阶瞳面像差
    幅度（Zernike (3,±1) 彗差、(4,0) 球差、(4,2) 二级散光）、1 个标量展宽增量、
    2 个静态图混合幅度（train 残差的径向剖面；train 实测均值的高通静态结构，
    对应 ``diagnose_static_floor.py`` 的静态模板）。
    **实现的是线性化（对像差系数一阶）模型，不是完整重传播**：像差 a 的一阶
    强度响应取输运方程 ``I' = I - div(I · ∇ψ)``，``ψ`` 是该 Zernike 单位幅度
    映射到相机网格的像差；展宽项为 ``G_{σ=1.2}(I) - I``。所有基函数由
    ``(prediction, phase)`` 与固定常量算出（stump 由 train 统计归一化），
    不引入任何未标定的瞳面振幅/增益，因此不伪造光学系统参数。
``cnn``（无约束，≈ 5.9k 参数）
    输入 = cat(prediction@64x64, phase@64x64) 两个通道，3 层 3x3 卷积，
    输出 64x64 残差双线性上采样到 161x161 加到预测上。预期会过拟合——
    这正是对照的意义。

两者都用 Adam + L2 weight decay，按 **train loss** 选最佳轮（test 不参与），
固定随机种子。

消融（`--kind cnn` 专用，默认关，关闭时训练行为与之前完全一致）
------------------------------------------------------------------
``--drop-phase``    只喂预测通道（卷积输入通道 1），相位整条通路移除。若 test
                    增益仍在，说明校正器吃的是与相位无关的观测算子/静态项。
``--shuffle-phase`` 相位换成**另一条 train 行**的（确定性 derangement，按
                    ``--seed``），预测与实测仍是本行。若增益仍在，说明相位通道
                    不携带可用信息。

用法::

    python scripts/slm/train_residual_corrector.py --kind operator \
        --dataset data/debug/slm_transfer/<run> --fit fit_m3_full \
        --iteration 17 --out report/.../residual_operator.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_transfer_fit_runs import METRICS, phase_family

import torch
from torch import nn
from torch.nn import functional as F

WINDOW_STEP_PX = 2
WINDOW_SPAN_PX = 160
BLUR_SIGMA_PX = 1.2
"""CCD PSF 的展宽刻度；训练只在 train 上估计幅度，σ 固定为标称值。"""

MODES: tuple[tuple[int, int], ...] = ((3, 1), (3, -1), (4, 0), (4, 2))
"""高阶瞳面像差：彗差 x / 彗差 y / 球差 / 二级散光。"""

N_COEFFS = len(MODES) + 1 + 2
MLP_HIDDEN = 24
CNN_CHANNELS = 24

# 影响模型 forward 的尺寸常量（作为默认值；bundle 的 config 会显式保存，
# 使后续脚本只凭 config + state_dict 即可回放，不依赖模块级全局常量的当前值）。
OPERATOR_PHASE_POOL = 16
"""OperatorCorrector 中 phase 的自适应池化边长（features = pool**2 + 2）。"""
OPERATOR_HIGHPASS_SIGMA = 2.0
"""build() 中静态 highpass 模板的 σ（已烘焙进 static_maps buffer）。"""
CNN_DOWNSAMPLE = 64
"""CnnCorrector 中 prediction/phase 的自适应池化边长。"""
CNN_RESIDUAL_SCALE_FACTOR = 0.1
"""CnnCorrector 残差缩放 = factor * prediction_scale。"""


def model_config() -> dict[str, Any]:
    """影响 build / FixedBasis / forward 的全部常量，写入权重 bundle 的 config。"""
    return {
        "WINDOW_SPAN_PX": WINDOW_SPAN_PX,
        "WINDOW_STEP_PX": WINDOW_STEP_PX,
        "CAMERA_GRID": 2 * WINDOW_SPAN_PX // WINDOW_STEP_PX + 1,
        "BLUR_SIGMA_PX": BLUR_SIGMA_PX,
        "MODES": [list(mode) for mode in MODES],
        "N_MODES": len(MODES),
        "N_COEFFS": N_COEFFS,
        "MLP_HIDDEN": MLP_HIDDEN,
        "CNN_CHANNELS": CNN_CHANNELS,
        "OPERATOR_PHASE_POOL": OPERATOR_PHASE_POOL,
        "OPERATOR_FEATURES": OPERATOR_PHASE_POOL * OPERATOR_PHASE_POOL + 2,
        "OPERATOR_HIGHPASS_SIGMA": OPERATOR_HIGHPASS_SIGMA,
        "CNN_DOWNSAMPLE": CNN_DOWNSAMPLE,
        "CNN_RESIDUAL_SCALE_FACTOR": CNN_RESIDUAL_SCALE_FACTOR,
    }


# ---------------------------------------------------------------- 数据装配

def valid(row: dict[str, Any], full_scale: float | None = None) -> bool:
    """与 ``fit.valid`` 逐字一致：峰值为满量程的比例门，兼容不同位深。"""
    scale = full_scale if full_scale is not None else row.get("full_scale", 255)
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = 255.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 255.0
    try:
        peak = float(row.get("peak", scale))
    except (TypeError, ValueError):
        peak = scale
    return (row.get("settled", {}).get("settled", True)
            and row.get("exposure_status", row.get("exposure_setting", {}).get("status", "ok")) == "ok"
            and peak < 250 / 255 * scale
            and float(row["exposure_ms"]) > 0)


def spot(row: dict[str, Any]) -> np.ndarray:
    if "spot_xy" in row:
        return np.asarray(row["spot_xy"], dtype=float)
    return np.array([row["spot"][f"centroid_{axis}"] for axis in ("x", "y")])


def radiance(image: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    """counts / exposure_ms —— 与 ``radiometry.count_rate`` 相同。"""
    return np.asarray(image) / float(row["exposure_ms"])


def assemble(dataset: Path, fit: Path, iteration: int) -> dict[str, Any]:
    """重建 fit 用的 68 行 (phase, measured, prediction)，并校验行序与指标。"""
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads((fit / "provenance.json").read_text(encoding="utf-8"))
    result = json.loads((fit / "result.json").read_text(encoding="utf-8"))
    config = json.loads((dataset / "config.json").read_text(encoding="utf-8"))

    usable = [row for row in manifest if valid(row)]
    references = [row for row in usable if row["label"] == "reference_start"]
    if not references:
        raise ValueError("valid reference_start required")
    centre = spot(references[0])
    yy, xx = np.mgrid[-WINDOW_SPAN_PX:WINDOW_SPAN_PX + 1:WINDOW_STEP_PX,
                      -WINDOW_SPAN_PX:WINDOW_SPAN_PX + 1:WINDOW_STEP_PX]
    xy = np.stack((xx + round(centre[0]), yy + round(centre[1])), axis=-1)
    selected = [row for row in usable if row.get("split") in {"train", "test", "validation"}]

    expected = provenance["rows"]
    if [row["label"] for row in selected] != [row["label"] for row in expected]:
        raise ValueError(
            f"{fit}: manifest-derived rows disagree with provenance row order; "
            "cannot align predictions")

    stored = np.load(fit / f"iteration_{iteration:04d}.npz")["prediction"]
    stored = np.asarray(stored, dtype=np.float32)
    if stored.shape[0] != len(selected):
        raise ValueError(f"{fit}: {stored.shape[0]} predictions for {len(selected)} rows")

    grid = int(config.get("grid", 128))
    phases, measured, frames = [], [], []
    for row in selected:
        path = dataset / row["file"]
        payload = np.load(path)
        try:
            # 新采集直接存了显示用的控制网格（与 fit.load_dataset 同一判据）。
            phase = np.asarray(payload["model_phase"])
            if phase.shape != (grid, grid):
                raise ValueError(f"{path.name}: model_phase {phase.shape} != {(grid, grid)}")
            frame = np.asarray(payload["raw_frames"]).mean(0)
        finally:
            payload.close()
        frames.append(radiance(frame[xy[..., 1].astype(int), xy[..., 0].astype(int)], row))
        phases.append(phase)
    measured = np.stack(frames).astype(np.float32)
    phases = np.stack(phases).astype(np.float32)

    # 硬校验：baseline 指标必须复现 result.json（否则窗口/曝光/行序有一处不同）。
    for index, (pred, target) in enumerate(zip(stored, measured, strict=True)):
        ours = row_metrics(pred, target)
        theirs = result["metrics"][index]
        if abs(ours["cv_measured"] - float(theirs["cv_measured"])) > 1e-5:
            raise ValueError(f"row {index}: cv_measured {ours['cv_measured']} vs "
                             f"{theirs['cv_measured']} — measured images differ from the fit")

    labels = [row["label"] for row in selected]
    splits = [row["split"] for row in selected]
    return {"rows": [{"index": i, "label": label, "split": split, "family": phase_family(label)}
                     for i, (label, split) in enumerate(zip(labels, splits, strict=True))],
            "phases": torch.from_numpy(phases),
            "measured": torch.from_numpy(measured),
            "predictions": torch.from_numpy(stored),
            "fit_iteration": int(result.get("iteration", -1)),
            "fit_loss": float(result.get("loss", float("nan")))}


# ---------------------------------------------------------------- 指标

def row_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """与 ``fit.metrics`` 逐项一致（correlation / cv / relative_l2）。"""
    cv_p = float(prediction.std() / max(float(prediction.mean()), 1e-9))
    cv_t = float(target.std() / max(float(target.mean()), 1e-9))
    p, t = prediction.ravel(), target.ravel()
    return {"correlation": float(np.corrcoef(p, t)[0, 1]),
            "cv_prediction": cv_p,
            "cv_measured": cv_t,
            "cv_abs_error": abs(cv_p - cv_t),
            "relative_l2": float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-9))}


def tensor_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return row_metrics(prediction.detach().cpu().numpy(), target.detach().cpu().numpy())


def mean_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """``row_metrics`` 的向量化版本（逐行同定义），用于逐轮 test 诊断。"""
    p = prediction.flatten(1)
    t = target.flatten(1)
    pc, tc = p - p.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
    correlation = (pc * tc).sum(1) / (pc.norm(dim=1) * tc.norm(dim=1)).clamp_min(1e-9)
    relative_l2 = (p - t).norm(dim=1) / t.norm(dim=1).clamp_min(1e-9)
    cv_p = p.std(1) / p.mean(1).clamp_min(1e-9)
    cv_t = t.std(1) / t.mean(1).clamp_min(1e-9)
    return {"correlation": float(correlation.mean()),
            "relative_l2": float(relative_l2.mean()),
            "cv_abs_error": float((cv_p - cv_t).abs().mean())}


METRIC_NAMES = tuple(name for name in METRICS
                     if name in {"correlation", "relative_l2", "cv_abs_error"})


# ---------------------------------------------------------------- 物理基函数

def zernike(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """实值 Zernike（未归一化，单位幅度由调用方缩放）。"""
    am = abs(m)
    radial = np.zeros_like(rho)
    for k in range((n - am) // 2 + 1):
        coeff = ((-1) ** k * math.factorial(n - k)
                 / (math.factorial(k)
                    * math.factorial((n + m) // 2 - k)
                    * math.factorial((n - m) // 2 - k)))
        radial += coeff * rho ** (n - 2 * k)
    return radial * (np.cos(am * theta) if m >= 0 else np.sin(am * theta))


def pupil_gradients(size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """每个模式在单位圆内单位幅度的 ∇ψ，映射到相机网格 (161x161)。"""
    coordinates = np.linspace(-1, 1, size)
    gx, gy = np.meshgrid(coordinates, coordinates)
    rho, theta = np.hypot(gx, gy), np.arctan2(gy, gx)
    inside = rho <= 1.0
    out = []
    for n, m in MODES:
        mode = np.where(inside, zernike(n, m, rho, theta), 0.0)
        scale = np.abs(mode[inside]).max()
        mode = mode / (scale if scale > 0 else 1.0)
        dy, dx = np.gradient(mode, coordinates, coordinates)
        out.append((_resize(dx, 2 * WINDOW_SPAN_PX // WINDOW_STEP_PX + 1),
                    _resize(dy, 2 * WINDOW_SPAN_PX // WINDOW_STEP_PX + 1)))
    return out


def _resize(array: np.ndarray, size: int) -> np.ndarray:
    """双线性重采样到 size x size（固定网格，不随批次变化）。"""
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None]
    return F.interpolate(tensor, size=(size, size), mode="bilinear",
                         align_corners=True)[0, 0].numpy()


def _ddx(field: torch.Tensor) -> torch.Tensor:
    padded = F.pad(field, (1, 1, 0, 0), mode="replicate")
    return 0.5 * (padded[..., 2:] - padded[..., :-2])


def _ddy(field: torch.Tensor) -> torch.Tensor:
    padded = F.pad(field, (0, 0, 1, 1), mode="replicate")
    return 0.5 * (padded[..., 2:, :] - padded[..., :-2, :])


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = (kernel / kernel.sum())[None, None, None, :]
    flat = image.reshape(-1, 1, image.shape[-2], image.shape[-1])
    flat = F.conv2d(F.pad(flat, (radius, radius, 0, 0), mode="replicate"), kernel)
    return F.conv2d(F.pad(flat, (0, 0, radius, radius), mode="replicate"),
                    kernel.transpose(-1, -2)).reshape(image.shape)


def transport_basis(image: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """一阶强度响应 ``-div(I ∇ψ)``，按单位幅度 ψ 计算。"""
    psi_x, psi_y = psi
    flux_x = image * psi_x
    flux_y = image * psi_y
    return -(_ddx(flux_x) + _ddy(flux_y))


class FixedBasis(nn.Module):
    """由 (prediction, phase) 与固定常量算出的 7 个基函数，全部无梯度。

    ``blur_sigma`` 是 forward 中展宽项的 σ；作为 buffer 保存，随 state_dict 一起
    序列化，使后续脚本只凭 config + state_dict 即可回放，不依赖模块级全局常量。
    """

    def __init__(self, static_maps: torch.Tensor, scales: torch.Tensor,
                 psi: list[tuple[torch.Tensor, torch.Tensor]],
                 blur_sigma: float = BLUR_SIGMA_PX) -> None:
        super().__init__()
        self.register_buffer("static_maps", static_maps)  # (2,H,W)
        self.register_buffer("scales", scales)            # (N_COEFFS,1,1)
        for index, (px, py) in enumerate(psi):
            self.register_buffer(f"psi_x_{index}", px)
            self.register_buffer(f"psi_y_{index}", py)
        self.n_modes = len(psi)
        self.register_buffer("blur_sigma", torch.tensor(float(blur_sigma), dtype=torch.float64))

    def forward(self, prediction: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        del phase  # 基函数只用 prediction；相位经 MLP 特征进入
        terms = [transport_basis(prediction, (getattr(self, f"psi_x_{index}"),
                                             getattr(self, f"psi_y_{index}")))
                 for index in range(self.n_modes)]
        terms.append(_gaussian_blur(prediction, float(self.blur_sigma)) - prediction)
        for index in range(self.static_maps.shape[0]):
            terms.append(self.static_maps[index].expand_as(prediction))
        basis = torch.stack(terms, dim=1)
        return basis / self.scales.unsqueeze(0)


# ---------------------------------------------------------------- 模型

class OperatorCorrector(nn.Module):
    """物理约束校正器：MLP → 7 个系数 × 固定物理基函数。"""

    def __init__(self, basis: FixedBasis,
                 phase_pool: int = OPERATOR_PHASE_POOL,
                 mlp_hidden: int = MLP_HIDDEN,
                 n_coeffs: int = N_COEFFS) -> None:
        super().__init__()
        self.basis = basis
        self.phase_pool = int(phase_pool)
        self.register_buffer("phase_pool_buf", torch.tensor(self.phase_pool))
        self.features = self.phase_pool * self.phase_pool + 2
        self.mlp = nn.Sequential(
            nn.Linear(self.features, mlp_hidden), nn.Tanh(),
            nn.Linear(mlp_hidden, n_coeffs))

    def forward(self, prediction: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(phase.unsqueeze(1), self.phase_pool).flatten(1)
        rms = phase.flatten(1).pow(2).mean(1, keepdim=True).sqrt()
        peak = phase.flatten(1).abs().amax(1, keepdim=True)
        coefficients = torch.tanh(self.mlp(torch.cat((pooled, rms, peak), dim=1)))
        basis = self.basis(prediction, phase)
        return prediction + (coefficients.unsqueeze(-1).unsqueeze(-1) * basis).sum(1)


class CnnCorrector(nn.Module):
    """无约束基线：64x64 上的 3 层卷积残差，上采样后相加。

    ``in_channels=1`` 是 ``--drop-phase`` 消融：只喂预测通道，相位整条通路消失
    （参数也随之减少，不是把相位当零填进去）。
    """

    def __init__(self, prediction_scale: float, phase_scale: float,
                 in_channels: int = 2, downsample: int = CNN_DOWNSAMPLE,
                 channels: int = CNN_CHANNELS,
                 residual_scale_factor: float = CNN_RESIDUAL_SCALE_FACTOR) -> None:
        super().__init__()
        self.register_buffer("prediction_scale", torch.tensor(float(prediction_scale), dtype=torch.float64))
        self.register_buffer("phase_scale", torch.tensor(float(phase_scale), dtype=torch.float64))
        self.in_channels = int(in_channels)
        self.downsample = int(downsample)
        channels = int(channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, 1, 3, padding=1))
        self.register_buffer("residual_scale",
                             torch.tensor(float(residual_scale_factor) * prediction_scale,
                                          dtype=torch.float64))

    def forward(self, prediction: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        pred_ds = F.adaptive_avg_pool2d(prediction.unsqueeze(1), self.downsample)
        channels = [pred_ds / float(self.prediction_scale)]
        if self.in_channels == 2:
            phase_ds = F.adaptive_avg_pool2d(phase.unsqueeze(1), self.downsample)
            channels.append(phase_ds / float(self.phase_scale))
        residual = self.net(torch.cat(channels, dim=1)) * float(self.residual_scale)
        residual = F.interpolate(residual, size=prediction.shape[-2:], mode="bilinear",
                                 align_corners=True)[:, 0]
        return prediction + residual


# ---------------------------------------------------------------- 训练

def loss_terms(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """与 fit 目标同形：correlation + 0.25 相对 MSE + 0.25 CV 失配。"""
    p = prediction.flatten(1)
    t = target.flatten(1)
    corr = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp_min(1e-9)
    relative_mse = ((p - t) ** 2).mean(1) / (t ** 2).mean(1).clamp_min(1e-9)
    cv_p = p.std(1) / p.mean(1).clamp_min(1e-9)
    cv_t = t.std(1) / t.mean(1).clamp_min(1e-9)
    return (1 - corr).mean() + 0.25 * relative_mse.mean() \
        + 0.25 * ((cv_p - cv_t) / cv_t.clamp_min(0.1)).square().mean()


def build(kind: str, data: dict[str, Any], train_mask: torch.Tensor,
          device: torch.device, drop_phase: bool = False) -> nn.Module:
    predictions = data["predictions"][train_mask]
    phases = data["phases"][train_mask]
    if kind == "operator":
        measured = data["measured"][train_mask]
        residual = measured - predictions
        radius = _radius(predictions.shape[-1])
        radial = torch.zeros_like(predictions[0])
        for value in radius.unique():
            mask = radius == value
            radial[mask] = residual[:, mask].mean()
        highpass = _gaussian_blur(measured.mean(0), 2.0) - measured.mean(0)
        static = torch.stack((radial, highpass))
        static = static / static.flatten(1).std(1).clamp_min(1e-9)[:, None, None]
        gradients = [(torch.from_numpy(px), torch.from_numpy(py))
                     for px, py in pupil_gradients(predictions.shape[-1])]
        raw_terms = [transport_basis(predictions, gradient) for gradient in gradients]
        raw_terms.append(_gaussian_blur(predictions, BLUR_SIGMA_PX) - predictions)
        raw_terms.append(static[0][None].expand_as(predictions))
        raw_terms.append(static[1][None].expand_as(predictions))
        scales = torch.stack([term.flatten(1).std(1).mean() for term in raw_terms])
        scales = scales.clamp_min(1e-6)[:, None, None]
        psi = [(px.to(device), py.to(device)) for px, py in gradients]
        basis = FixedBasis(static.to(device), scales.to(device), psi).to(device)
        return OperatorCorrector(basis).to(device)
    prediction_scale = float(predictions.std()) + 1e-9
    phase_scale = float(phases.std()) + 1e-9
    return CnnCorrector(prediction_scale, phase_scale,
                        in_channels=1 if drop_phase else 2).to(device)


def _radius(size: int) -> torch.Tensor:
    coordinates = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return torch.hypot(xx, yy).round().to(torch.int64)


def shuffled_phases(phases: torch.Tensor, splits: list[str], seed: int) -> torch.Tensor:
    """``--shuffle-phase``：每行换成**另一条 train 行**的相位（确定性 derangement）。

    train 行之间是一个无不动点的置换，所以"换成自己"不可能发生；test 行取
    某条固定的 train 相位。预测、实测、损失全部保持原行不动。
    """
    train_index = [index for index, split in enumerate(splits) if split == "train"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_index))
    while np.any(order == np.arange(len(train_index))):
        order = rng.permutation(len(train_index))
    out = phases.clone()
    for rank, row in enumerate(train_index):
        out[row] = phases[train_index[int(order[rank])]]
    for row, split in enumerate(splits):
        if split == "train":
            continue
        out[row] = phases[train_index[int(order[row % len(train_index)])]]
    return out


def train_corrector(model: nn.Module, predictions: torch.Tensor, phases: torch.Tensor,
                    measured: torch.Tensor, args: argparse.Namespace,
                    test: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    ) -> dict[str, Any]:
    """按 TRAIN loss 选轮；test 每轮只做记录（诊断），从不参与选择。"""
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    test_predictions, test_phases, test_measured = test
    trace_l2: list[float] = []
    trace_corr: list[float] = []
    best_loss, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(args.epochs):
        model.train()
        optimiser.zero_grad()
        loss = loss_terms(model(predictions, phases), measured)
        loss.backward()
        optimiser.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss, best_epoch = value, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.eval()
        with torch.no_grad():
            observed = mean_metrics(model(test_predictions, test_phases), test_measured)
        trace_l2.append(observed["relative_l2"])
        trace_corr.append(observed["correlation"])
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_train_loss": best_loss,
            "epochs": args.epochs,
            "test_peak_relative_l2": {"epoch": int(np.argmin(trace_l2)),
                                      "value": float(np.min(trace_l2))},
            "test_peak_correlation": {"epoch": int(np.argmax(trace_corr)),
                                      "value": float(np.max(trace_corr))},
            "test_trace_relative_l2": trace_l2,
            "test_trace_correlation": trace_corr}


# ---------------------------------------------------------------- 报告

def summarise(rows: list[dict[str, Any]], kind: str, split: str) -> dict[str, Any]:
    """按相位族聚合（与 compare_transfer_fit_runs 的族划分和指标名一致）。"""
    chosen = [row for row in rows if row["split"] == split]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in chosen:
        groups.setdefault(row["family"], []).append(row)
    groups["all_" + split] = chosen
    summary: dict[str, Any] = {}
    for family, entries in groups.items():
        block: dict[str, Any] = {"n": len(entries)}
        for name in METRIC_NAMES:
            base = np.array([entry["baseline"][name] for entry in entries])
            corr = np.array([entry[kind][name] for entry in entries])
            block[name] = {"baseline_mean": float(base.mean()),
                           f"{kind}_mean": float(corr.mean()),
                           "delta_mean": float(corr.mean() - base.mean())}
        block["improved_relative_l2"] = sum(
            1 for entry in entries
            if entry[kind]["relative_l2"] < entry["baseline"]["relative_l2"])
        block["improved_correlation"] = sum(
            1 for entry in entries
            if entry[kind]["correlation"] > entry["baseline"]["correlation"])
        summary[family] = block
    return summary


def print_table(rows: list[dict[str, Any]], kind: str, summary: dict[str, Any]) -> None:
    print(f"\n{kind}: per-row test metrics")
    print(f"{'idx':>3} {'label':<26} {'corr':>8} {'d_corr':>8} "
          f"{'rl2':>8} {'d_rl2':>8} {'cv_abs':>8} {'d_cv':>8}")
    for row in rows:
        if row["split"] != "test":
            continue
        b, c = row["baseline"], row[kind]
        print(f"{row['index']:>3} {row['label']:<26} {c['correlation']:>8.4f} "
              f"{c['correlation'] - b['correlation']:>+8.4f} {c['relative_l2']:>8.4f} "
              f"{c['relative_l2'] - b['relative_l2']:>+8.4f} {c['cv_abs_error']:>8.4f} "
              f"{c['cv_abs_error'] - b['cv_abs_error']:>+8.4f}")
    print(f"\n{kind}: test summary by family (deltas vs baseline)")
    print(f"{'family':<16} {'n':>2} {'corr':>8} {'d_corr':>8} {'rl2':>8} {'d_rl2':>8}")
    for family, block in summary.items():
        print(f"{family:<16} {block['n']:>2} {block['correlation'][f'{kind}_mean']:>8.4f} "
              f"{block['correlation']['delta_mean']:>+8.4f} "
              f"{block['relative_l2'][f'{kind}_mean']:>8.4f} "
              f"{block['relative_l2']['delta_mean']:>+8.4f}")


def bundle_config(args: argparse.Namespace) -> dict[str, Any]:
    """自检配置：重建模型架构与复现推理所需的全部超参数/常量/消融开关。"""
    config = model_config()
    config.update({
        "kind": args.kind,
        "seed": int(args.seed),
        "drop_phase": bool(args.drop_phase),
        "shuffle_phase": bool(args.shuffle_phase),
        "IN_CHANNELS": 1 if args.drop_phase else 2,
        "iteration": int(args.iteration),
        "device": str(args.device),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "epochs": int(args.epochs),
    })
    return config


def save_weights_bundle(path: Path, model: nn.Module, args: argparse.Namespace,
                        history: dict[str, Any]) -> None:
    """把选中的最佳模型权重 + 重建配置写成一个自包含的 torch bundle。"""
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    bundle = {
        "format_version": 1,
        "state_dict": state_dict,
        "kind": args.kind,
        "best_epoch": int(history["best_epoch"]),
        "best_train_loss": float(history["best_train_loss"]),
        "config": bundle_config(args),
        "fit": str(args.fit),
        "iteration": int(args.iteration),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


def rebuild_from_bundle(bundle: dict[str, Any]) -> nn.Module:
    """仅凭 bundle 里的 config + state_dict 重建模型，不读取任何模块级全局常量。

    供后续脚本 ``torch.load(path, weights_only=True)`` 后直接调用。固定基函数的
    静态图/scales/ψ_grad 与各类缩放因子都已作为 buffer 落在 state_dict 中，
    这里只按 config 里记录的形状参数实例化正确的层结构。
    """
    config = bundle["config"]
    state = bundle["state_dict"]
    kind = bundle["kind"]

    if kind == "operator":
        psi: list[tuple[torch.Tensor, torch.Tensor]] = []
        index = 0
        while f"basis.psi_x_{index}" in state:
            psi.append((state[f"basis.psi_x_{index}"], state[f"basis.psi_y_{index}"]))
            index += 1
        if len(psi) != int(config["N_MODES"]):
            raise ValueError(f"psi count {len(psi)} != config N_MODES {config['N_MODES']}")
        basis = FixedBasis(
            state["basis.static_maps"], state["basis.scales"], psi,
            blur_sigma=float(state["basis.blur_sigma"]))
        model: nn.Module = OperatorCorrector(
            basis, phase_pool=int(config["OPERATOR_PHASE_POOL"]),
            mlp_hidden=int(config["MLP_HIDDEN"]),
            n_coeffs=int(config["N_COEFFS"]))
    elif kind == "cnn":
        model = CnnCorrector(
            float(state["prediction_scale"]), float(state["phase_scale"]),
            in_channels=int(config["IN_CHANNELS"]),
            downsample=int(config["CNN_DOWNSAMPLE"]),
            channels=int(config["CNN_CHANNELS"]),
            residual_scale_factor=float(config["CNN_RESIDUAL_SCALE_FACTOR"]))
    else:
        raise ValueError(f"unknown kind: {kind}")

    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=17)
    parser.add_argument("--kind", required=True, choices=("operator", "cnn"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--save-weights", type=Path, default=None,
                        help="可选：把选中的最佳模型权重与重建配置存为 torch bundle")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--drop-phase", action="store_true",
                        help="消融：cnn 只吃预测通道，相位整条通路移除")
    parser.add_argument("--shuffle-phase", action="store_true",
                        help="消融：cnn 的相位换成另一条 train 行的相位（确定性、按 --seed）")
    args = parser.parse_args()
    if (args.drop_phase or args.shuffle_phase) and args.kind != "cnn":
        parser.error("--drop-phase / --shuffle-phase are cnn-only ablations")
    if args.drop_phase and args.shuffle_phase:
        parser.error("--drop-phase and --shuffle-phase are mutually exclusive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)

    data = assemble(args.dataset, args.fit, args.iteration)
    predictions = data["predictions"].to(device)
    phases = data["phases"].to(device)
    measured = data["measured"].to(device)
    splits = [row["split"] for row in data["rows"]]
    train_mask = torch.tensor([split == "train" for split in splits])
    test_mask = torch.tensor([split == "test" for split in splits])
    if args.shuffle_phase:
        phases = shuffled_phases(phases, splits, args.seed)

    model = build(args.kind, data, train_mask, device, drop_phase=args.drop_phase)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    history = train_corrector(model, predictions[train_mask], phases[train_mask],
                              measured[train_mask], args,
                              (predictions[test_mask], phases[test_mask],
                               measured[test_mask]))
    with torch.no_grad():
        corrected = model(predictions, phases)

    rows = []
    for index, entry in enumerate(data["rows"]):
        rows.append({**entry,
                     "baseline": tensor_metrics(predictions[index], measured[index]),
                     args.kind: tensor_metrics(corrected[index], measured[index])})
    summary = summarise(rows, args.kind, "test")
    train_summary = summarise(rows, args.kind, "train")

    payload = {
        "config": {
            "dataset": str(args.dataset), "fit": str(args.fit),
            "iteration": args.iteration, "kind": args.kind, "epochs": args.epochs,
            "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed,
            "device": str(device), "parameters": parameters,
            "fit_iteration": data["fit_iteration"], "fit_loss": data["fit_loss"],
            "train_rows": int(train_mask.sum()),
            "test_rows": sum(1 for split in splits if split == "test"),
            "correction_model": "linearised (first order in the aberration coefficients)",
            "drop_phase": bool(args.drop_phase),
            "shuffle_phase": bool(args.shuffle_phase),
            "epoch_selection": "best TRAIN loss only; test trace is diagnostic, never selected on",
        },
        "training": history,
        "rows": rows,
        "summary": {"test": summary, "train": train_summary},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    if args.save_weights is not None:
        save_weights_bundle(args.save_weights, model, args, history)
        print(f"  weights bundle -> {args.save_weights}")

    print(f"{args.kind}: {parameters} parameters, best epoch {history['best_epoch']} "
          f"(train loss {history['best_train_loss']:.6f}) -> {args.out}")
    print(f"  diagnostic (NOT used for selection): test relative_l2 min "
          f"{history['test_peak_relative_l2']['value']:.4f} @ epoch "
          f"{history['test_peak_relative_l2']['epoch']}; test correlation max "
          f"{history['test_peak_correlation']['value']:.4f} @ epoch "
          f"{history['test_peak_correlation']['epoch']}")
    print_table(rows, args.kind, summary)


if __name__ == "__main__":
    main()
