"""衍射效率扫描的产物写出 (CSV / JSON / PNG) —— **无硬件依赖**。

本模块只做"把已经算好的结果落盘": 不 import 任何设备驱动, 不碰硬件, 因此
可以离线重跑 (例如从保存下来的扫描点重建曲线)。

三类产物
--------
``write_measurements_csv``
    逐点扫描数据 (CSV)。首行是 ``#`` 注释, 带上拟合摘要, 方便肉眼看;
    之后是固定表头 + 每点一行。
``write_summary_json``
    体检结论 + 元数据 (JSON)。numpy 标量/数组会被转成原生类型;
    非有限浮点 (NaN/Inf) 转成 ``null``, 以免写出不是合法 JSON 的
    ``NaN``/``Infinity`` 记号。
``plot_efficiency``
    两联图: (a) 实测 ±1 级相对效率 vs 命令, 叠拟合模型并标注拟合参数;
    (b) 拟合相位深度 Δφ(u) vs 归一化命令。matplotlib 不可用时只告警,
    不写文件 (仓库里其它报告脚本同样处理)。

术语
----
中文叙述遵守仓库术语表: 实验系统, 瞳面, 加性背景 (杂散光) 项。
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.tools.slm.calibration.diffraction_efficiency.core import (
    PhaseResponseFit,
    SweepPoint,
)

try:
    import matplotlib

    matplotlib.use("Agg")  # 无界面环境 (headless) 必须先于 pyplot 设定
    import matplotlib.pyplot as plt

    HAS_PLOT = True
except ImportError:  # pragma: no cover - 取决于运行环境
    HAS_PLOT = False

__all__ = [
    "FOCUS_SWEEP_COLUMNS",
    "plot_efficiency",
    "plot_focus_sweep",
    "write_focus_sweep_csv",
    "write_measurements_csv",
    "write_summary_json",
]

#: 逐点列 (顺序即表头顺序)。
BASE_FIELDS: tuple[str, ...] = (
    "command",
    "e_zero",
    "e_plus",
    "e_minus",
    "eta_zero",
    "eta_plus_total",
    "separation_px",
    "residual_px",
    "peak_adu",
    "fwhm_px",
    "ellipticity",
)

#: 有拟合结果时追加的列。
FIT_FIELDS: tuple[str, ...] = ("model_eta_plus_total", "model_depth_rad")


def _fmt(value: Any) -> str:
    """CSV 单元格文本化: ``None`` → 空串, 浮点 → 6 位有效数字。"""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return ""
        return f"{number:.6g}"
    return str(value)


def _fit_comment(fit: PhaseResponseFit | None) -> str:
    """CSV 首行的 ``#`` 注释 (元数据)。"""
    stamp = datetime.now().isoformat(timespec="seconds")
    if fit is None:
        return f"# diffraction efficiency sweep | written {stamp} | fit: none\n"
    return (
        f"# diffraction efficiency sweep | written {stamp} | "
        f"fit: depth_max={fit.depth_max_rad:.6g} rad ({fit.depth_max_pi:.4f}pi), "
        f"gamma={fit.gamma:.6g}, leakage={fit.leakage:.6g}, "
        f"residual_rms={fit.residual_rms:.6g}, n_points={fit.n_points}\n"
    )


def _model_by_command(fit: PhaseResponseFit | None) -> dict[float, tuple[float, float]]:
    """把拟合的模型序列按命令值索引 (拟合会剔除无效点, 不能按下标对齐)。"""
    if fit is None:
        return {}
    return {
        float(command): (float(eta), float(depth))
        for command, eta, depth in zip(
            fit.commands,
            fit.model_eta_plus_total,
            fit.model_depth_rad,
            strict=False,
        )
    }


def write_measurements_csv(
    path: Path,
    points: Sequence[SweepPoint],
    fit: PhaseResponseFit | None = None,
) -> Path:
    """把扫描点写成 CSV (逐点一行 + 固定表头)。

    Args:
        path: 输出路径 (父目录会自动创建)。
        points: 扫描点。
        fit: 可选的拟合结果 —— 给了就追加 ``model_eta_plus_total`` 与
            ``model_depth_rad`` 两列 (按命令值对齐, 不是按下标)。

    Returns:
        写出的文件路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BASE_FIELDS) + (list(FIT_FIELDS) if fit is not None else [])
    models = _model_by_command(fit)

    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(_fit_comment(fit))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            row: dict[str, Any] = {
                "command": point.command,
                "e_zero": point.e_zero,
                "e_plus": point.e_plus,
                "e_minus": point.e_minus,
                "eta_zero": point.eta_zero,
                "eta_plus_total": point.eta_plus_total,
                "separation_px": point.separation_px,
                "residual_px": point.residual_px,
                "peak_adu": point.peak_adu,
                "fwhm_px": point.fwhm_px,
                "ellipticity": point.ellipticity,
            }
            if fit is not None:
                model = models.get(float(point.command))
                row["model_eta_plus_total"] = model[0] if model else None
                row["model_depth_rad"] = model[1] if model else None
            writer.writerow({key: _fmt(value) for key, value in row.items()})

    logger.info("扫描点 CSV 已写出: {}", path)
    return path


def _jsonable(obj: Any) -> Any:
    """把对象转成 ``json.dump`` 能吃的东西。

    numpy 标量/数组 → 原生类型; 非有限浮点 → ``None`` (JSON 里的 ``null``);
    其它不认识的对象 → ``str()``。
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        number = float(obj)
        return number if math.isfinite(number) else None
    if isinstance(obj, np.ndarray):
        return [_jsonable(value) for value in obj.tolist()]
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_summary_json(path: Path, summary: dict) -> Path:
    """把体检结论 + 元数据写成 JSON。

    Args:
        path: 输出路径 (父目录会自动创建)。
        summary: 结论字典 (numpy 标量会被转换)。

    Returns:
        写出的文件路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    logger.info("结论 JSON 已写出: {}", path)
    return path


def plot_efficiency(
    path: Path,
    points: Sequence[SweepPoint],
    fit: PhaseResponseFit | None = None,
) -> Path:
    """画两联图: 效率-命令曲线 + 拟合相位深度。

    面板 (a): 实测 ±1 级相对效率 vs 命令值, 叠上拟合模型, 并在图内标注
    拟合参数 (Δφ_max / γ / 泄漏 / 残差)。
    面板 (b): 拟合相位深度 Δφ(u) vs 归一化命令 ``u``, 并给一条理想的
    ``Δφ = 2π·u`` 参考线。

    归一化命令由模型反解 (``u = (Δφ/Δφ_max)^(1/γ)``), 所以不依赖调用方
    当时用的命令上限。

    Args:
        path: 输出图片路径 (父目录会自动创建)。
        points: 扫描点。
        fit: 可选的拟合结果。

    Returns:
        图片路径 (matplotlib 不可用时返回路径但不写文件)。
    """
    path = Path(path)
    if not HAS_PLOT:
        logger.warning("matplotlib 不可用, 跳过绘图: {}", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    font_config = {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
    with plt.rc_context(font_config):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)

        # --- 面板 (a): 效率 vs 命令 ---------------------------------
        ax = axes[0]
        measured = [
            (float(point.command), float(point.eta_plus_total))
            for point in points
            if point.eta_plus_total is not None
        ]
        if measured:
            ax.plot(
                [item[0] for item in measured],
                [item[1] for item in measured],
                "o",
                color="royalblue",
                markersize=5,
                label="实测 ±1 级相对效率",
            )
        if fit is not None:
            ax.plot(
                fit.commands,
                fit.model_eta_plus_total,
                "-",
                color="crimson",
                linewidth=2,
                label="拟合模型",
            )
            note = (
                f"$\\mathrm{{\\Delta\\varphi_{{max}}}} = {fit.depth_max_rad:.3f}$ rad "
                f"({fit.depth_max_pi:.3f}π)\n"
                f"$\\gamma = {fit.gamma:.3f}$\n"
                f"泄漏 = {fit.leakage:.3f}\n"
                f"残差 rms = {fit.residual_rms:.4f}"
            )
            ax.text(
                0.02,
                0.98,
                note,
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
            )
        ax.set_xlabel("命令值 (灰度 / 弧度)")
        ax.set_ylabel("±1 级相对衍射效率")
        ax.set_title("衍射效率 vs 命令")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # --- 面板 (b): 拟合相位深度 vs 归一化命令 ---------------------
        ax2 = axes[1]
        if fit is not None and fit.depth_max_rad > 0.0 and fit.gamma > 0.0:
            normalized = np.power(
                np.clip(fit.model_depth_rad / fit.depth_max_rad, 0.0, None),
                1.0 / fit.gamma,
            )
            ax2.plot(
                normalized,
                fit.model_depth_rad,
                "-o",
                color="crimson",
                markersize=4,
                label="拟合 Δφ(u)",
            )
            ax2.plot(
                [0.0, 1.0],
                [0.0, 2.0 * math.pi],
                "--",
                color="gray",
                label="理想 Δφ = 2π·u",
            )
        else:
            ax2.text(
                0.5,
                0.5,
                "无拟合结果",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
        ax2.set_xlabel("归一化命令 u")
        ax2.set_ylabel("拟合相位深度 Δφ (rad)")
        ax2.set_title("相位响应 Δφ(u)")
        ax2.grid(True, alpha=0.3)
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend(fontsize=9)

        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    logger.info("效率曲线图已写出: {}", path)
    return path


# ---------------------------------------------------------------------------
# 前置焦面扫描的产物 (离焦拒绝时**只有**这两份落盘)
# ---------------------------------------------------------------------------

#: 焦面扫描 CSV 的固定表头。
FOCUS_SWEEP_COLUMNS: tuple[str, ...] = (
    "diopter_m1",
    "focal_mm",
    "exposure_ms",
    "peak",
    "peak_fraction",
    "peak_per_ms",
    "fwhm_px",
    "hollowness",
    "status",
)


def write_focus_sweep_csv(path: Path, points: Sequence[Any]) -> Path:
    """把焦面扫描的逐点结果写成 CSV。

    表头见 :data:`FOCUS_SWEEP_COLUMNS`; ``focal_mm`` 在零焦度处是 ``inf``,
    写成空串 (不是合法 CSV 数值)。

    Args:
        path: 输出路径。
        points: :class:`~ao_shaping.tools.slm.panel.focus_probe.FocusPoint` 序列。

    Returns:
        写出的路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FOCUS_SWEEP_COLUMNS)
        for point in points:
            writer.writerow(
                [
                    _fmt(point.diopter_m1),
                    _fmt(point.focal_mm),
                    _fmt(point.exposure_ms),
                    _fmt(point.peak),
                    _fmt(point.peak_fraction),
                    _fmt(point.peak_per_ms),
                    _fmt(point.fwhm_px),
                    _fmt(point.hollowness),
                    str(point.status),
                ]
            )
    logger.info("焦面扫描 CSV 已写出: {}", path)
    return path


def plot_focus_sweep(
    path: Path,
    points: Sequence[Any],
    *,
    fwhm_max_px: float,
    best_diopter_m1: float | None = None,
) -> Path | None:
    """把焦面扫描画成 FWHM vs 1/f 曲线, 并画出门限线。

    离焦拒绝时这张图就是唯一的证据: 一眼能看出最佳点落在哪、离门限差多少,
    以及扫描范围是否够宽 (最佳点贴边 = 真正的焦面在范围外)。

    Args:
        path: 输出路径。
        points: :class:`~ao_shaping.tools.slm.panel.focus_probe.FocusPoint` 序列。
        fwhm_max_px: 在焦门限 (水平红线)。
        best_diopter_m1: 最佳点 (竖直虚线); None 则不画。

    Returns:
        写出的路径; matplotlib 不可用时返回 None (只告警)。
    """
    if not HAS_PLOT:  # pragma: no cover - 取决于运行环境
        logger.warning("matplotlib 不可用, 跳过焦面扫描图: {}", path)
        return None
    usable = [p for p in points if str(p.status) == "ok" and np.isfinite(p.fwhm_px)]
    if not usable:
        logger.warning("焦面扫描没有可用点, 跳过作图: {}", path)
        return None

    diopters = np.array([p.diopter_m1 for p in usable], dtype=np.float64)
    fwhm = np.array([p.fwhm_px for p in usable], dtype=np.float64)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    ax.plot(diopters, fwhm, "o-", color="#1f77b4", lw=1.4, ms=4.5, label="零级 FWHM")
    ax.axhline(
        float(fwhm_max_px), color="#d62728", ls="--", lw=1.4,
        label=f"在焦门限 {float(fwhm_max_px):.0f} px",
    )
    if best_diopter_m1 is not None and np.isfinite(best_diopter_m1):
        ax.axvline(
            float(best_diopter_m1), color="#2ca02c", ls=":", lw=1.4,
            label=f"最佳 1/f = {float(best_diopter_m1):.3f} m⁻¹",
        )
    ax.set_xlabel("离焦补偿 1/f (m⁻¹)")
    ax.set_ylabel("零级 FWHM (CCD px)")
    ax.set_title("测量前置: 焦面扫描")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("焦面扫描图已写出: {}", path)
    return path
