"""model-in-loop Step A 的自标定对比择优 (傅里叶域损失 / 瞳面仿射 / 行程缩放)。

在**已录的真实探针**上, 比较 Step A 的几个变体在**验证集探针**上的残差。

论文 (GRAPHYCS, Cho 等 2026) 给了自标定的结构, 但**没有**给「加 / 不加
傅里叶项 × 低探针数」的消融数字, 所以本脚本自己测 —— 这是本任务的核心证据。

输入
----
一次硬件 model-in-loop 运行的目录, 例如::

    data/debug/slm_model_in_loop/20261019_103208/

里面的 ``probes/*.npz`` 每一条是一次 (命令相位, 实测画面) 配对; ``run_config.json``
给出那次运行的模型配置; ``bench_geometry.json`` 给出相机→模型网格的换算;
``source_amplitude.npy`` 是那次运行标定过的瞳面振幅。

验证集探针 (holdout) 的 ``model_grid`` 已经落盘, 训练集探针的没有 —— 脚本用
**同一套**几何换算把它重建出来 (冻结的 ROI 中心取自平场画面的 0 级 argmax,
旋转象限取硬件路径的 1 个象限), 重建结果与验证集已落盘的 ``model_grid``
逐位一致 (相对 L1 误差 ~1e-11), 所以两半数据可以用同一把尺子量。

输出
----
``logs/selfcal_bakeoff/<时间戳>/`` 下的 ``bakeoff.csv`` 与 ``bakeoff.json``。
**不写** ``report/``: 那个目录下的每一份 markdown 都必须登记在
``scripts/_common/provenance.py`` 里并带溯源块, 而本脚本不在那份登记表内。

用法
----
::

    python scripts/slm/model_in_loop_selfcal_bakeoff.py \
        --run-dir data/debug/slm_model_in_loop/20261019_103208

术语: 中文叙述遵守 ``docs/terminology_glossary.md``。
"""

from __future__ import annotations

import argparse
from loguru import logger
import csv
import json
import math
import sys
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
    _fit_aberration_at_probes,
)
from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    SlmModelInLoopConfig,
    _make_optimizer,
    _to_model_grid,
    trust_region_clamp,
)
from ao_shaping.utils.wavefront.calibration_terms import (
    PupilAffine,
    TravelScale,
)
from ml.zernike.eval_stats import pearson

#: 硬件路径冻结的 ROI 中心 = 平场画面 0 级的 argmax (``zero_order_center``
#: 的 ``refine=False`` 分支)。
def _roi_center(flat_frame: np.ndarray) -> tuple[int, int]:
    """返回平场画面 0 级的 (x, y) 像素坐标。"""
    flat = np.asarray(flat_frame, dtype=np.float64)
    peak = float(flat.max())
    if peak <= 0.0:
        return (flat.shape[1] // 2, flat.shape[0] // 2)
    y, x = np.unravel_index(int(np.argmax(flat)), flat.shape)
    return (int(x), int(y))


def _peak_normalise(image: np.ndarray) -> np.ndarray:
    """按峰值归一化 (口径与仓库既有的峰值归一化损失一致)。"""
    array = np.asarray(image, dtype=np.float64)
    peak = float(array.max())
    if not math.isfinite(peak) or peak <= 0.0:
        return array
    return array / peak


def _ssim(predicted: np.ndarray, measured: np.ndarray) -> float:
    """峰值归一化后的 SSIM (skimage, 7x7 高斯窗)。

    与 ``src/ao_shaping/tools/slm/bench_twin_calibrate.py::_ssim`` 同一口径
    (skimage 优先, numpy 自实现回落), 复用它而不是再写一份 —— 本仓库已经
    多次被「重复实现漂移」咬伤。
    """
    from ao_shaping.tools.slm.bench_twin_calibrate import _ssim as canonical_ssim

    return float(canonical_ssim(predicted, measured))


@dataclass
class Probe:
    """一条 (命令相位, 实测远场) 配对。"""

    phase: np.ndarray
    measured: np.ndarray
    name: str


@dataclass
class Variant:
    """一个待比较的 Step A 变体。"""

    name: str
    fourier_weight: float = 0.0
    loss_kind: str = "mse"
    affine: bool = False
    travel: bool = False

    def describe(self) -> str:
        """返回一句话描述 (写进表格)。"""
        parts = []
        if self.fourier_weight > 0.0:
            parts.append(f"傅里叶项 α={self.fourier_weight:g}")
        if self.loss_kind != "mse":
            parts.append("图像域 L1")
        if self.affine:
            parts.append("τ_θ")
        if self.travel:
            parts.append("γ")
        return "+".join(parts) if parts else "基线"


#: 要比较的变体。基线 = 三项全关, 与改动前逐位相同。
VARIANTS: tuple[Variant, ...] = (
    Variant("baseline"),
    Variant("fourier", fourier_weight=7e-4),
    Variant("affine", affine=True),
    Variant("travel", travel=True),
    Variant("selfcal", affine=True, travel=True),
    Variant("fourier+selfcal", fourier_weight=7e-4, affine=True, travel=True),
    Variant("l1", loss_kind="l1"),
    Variant("fourier+l1", fourier_weight=7e-4, loss_kind="l1"),
)


def _load_probes(probes_dir: Path) -> tuple[list[Probe], list[Probe], np.ndarray]:
    """读取探针目录, 返回 (训练集, 验证集, 平场画面)。

    训练集探针的 ``model_grid`` 没有落盘, 这里用**同一套**几何换算重建:
    冻结的 ROI 中心取自平场画面的 0 级 argmax, 旋转象限取硬件路径的 1 个
    象限 (``camera_rotation_quarters = 0 if cam_type == "sim" else 1``)。
    """
    manifest = json.loads((probes_dir / "manifest.json").read_text(encoding="utf-8"))
    train_raw: list[tuple[str, np.ndarray, np.ndarray]] = []
    holdout: list[Probe] = []
    flat_frame: np.ndarray | None = None
    for entry in manifest:
        path = probes_dir / str(entry["file"])
        with np.load(path) as payload:
            phase = np.asarray(payload["phase"], dtype=np.float64)
            frame = np.asarray(payload["frame"], dtype=np.float64)
            grid = payload["model_grid"] if "model_grid" in payload.files else None
        if entry["split"] == "flat":
            flat_frame = frame
            continue
        if entry["split"] == "holdout":
            if grid is None:
                raise ValueError(f"{path} has no model_grid")
            holdout.append(
                Probe(phase=phase, measured=np.asarray(grid, dtype=np.float64),
                      name=str(entry["file"]))
            )
        elif entry["split"] == "train":
            train_raw.append((str(entry["file"]), phase, frame))
    if flat_frame is None:
        raise ValueError("the probe set has no flat baseline frame")
    return (
        [(name, phase, frame) for name, phase, frame in train_raw],
        holdout,
        flat_frame,
    )


def _build_config(run_dir: Path, overrides: dict[str, Any]) -> SlmModelInLoopConfig:
    """用那次运行自己的 ``run_config.json`` 重建配置。"""
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    known = {field.name for field in fields(SlmModelInLoopConfig)}
    kwargs = {key: value for key, value in payload.items() if key in known}
    kwargs.update(overrides)
    return SlmModelInLoopConfig(**kwargs)


def _evaluate(
    optimizer: Any,
    coefficients: np.ndarray,
    probes: list[Probe],
    source_amplitude: np.ndarray,
    footprint: int,
) -> dict[str, float]:
    """在探针上算 SSIM / Pearson r / 能量比 / 峰值归一化 MSE。

    两套口径都算:

    * **整幅** (1024² 的模型网格) —— 与仓库既有的
      :func:`~ao_shaping.optimizer.wfless.slm_model_in_loop._prediction_correlation`
      同口径 (MIL 循环的验证集门限 ``min_holdout_correlation`` 就是它);
    * **视场** (相机窗口在模型网格上占的那一小块) —— 物理上真正可比的那块。
      实测画面是相机窗口被**放大**到 1024² 的结果 (相机只看到模型视场的
      ~11%), 所以整幅比较里有一大片两边都是 0 的区域, 能量比会被那个
      「模型 PSF 比实测宽得多」的失配主导。

    Args:
        optimizer: 拟合用的优化器 (自标定量在它身上)。
        coefficients: 待评价的 Zernike 系数。
        probes: 待评价的探针。
        source_amplitude: 瞳面振幅。
        footprint: 相机视场在模型网格上的边长 (单元)。

    Returns:
        指标字典 (float)。
    """
    ssim_full: list[float] = []
    ssim_foot: list[float] = []
    correlations: list[float] = []
    energy_full: list[float] = []
    energy_foot: list[float] = []
    mse_full: list[float] = []
    mse_foot: list[float] = []
    half = max(int(footprint) // 2, 1)
    for probe in probes:
        predicted = optimizer.forward_intensity(
            coefficients, probe.phase, source_amplitude
        )
        reference = _peak_normalise(probe.measured)
        estimate = _peak_normalise(predicted)
        ssim_full.append(_ssim(predicted, probe.measured))
        correlations.append(
            pearson(np.asarray(estimate).ravel(), np.asarray(reference).ravel())
        )
        energy_full.append(
            float(np.sum(estimate) / max(float(np.sum(reference)), 1e-30))
        )
        mse_full.append(float(np.mean((estimate - reference) ** 2)))
        rows, cols = estimate.shape
        r0, c0 = rows // 2 - half, cols // 2 - half
        r1, c1 = r0 + 2 * half, c0 + 2 * half
        est_foot = estimate[r0:r1, c0:c1]
        ref_foot = reference[r0:r1, c0:c1]
        ssim_foot.append(
            _ssim(est_foot / max(float(est_foot.max()), 1e-30), ref_foot)
        )
        energy_foot.append(
            float(np.sum(est_foot) / max(float(np.sum(ref_foot)), 1e-30))
        )
        mse_foot.append(float(np.mean((est_foot - ref_foot) ** 2)))
    return {
        "ssim": float(np.mean(ssim_full)),
        "ssim_foot": float(np.mean(ssim_foot)),
        "pearson_r": float(np.mean(correlations)),
        "energy_ratio": float(np.mean(energy_full)),
        "energy_ratio_foot": float(np.mean(energy_foot)),
        "mse": float(np.mean(mse_full)),
        "mse_foot": float(np.mean(mse_foot)),
    }


def _render_table(rows: list[dict[str, Any]]) -> str:
    """把结果渲染成 markdown 表格 (写进 JSON 的 ``table_markdown`` 字段)。"""
    header = (
        "| 变体 | 说明 | 验证集 SSIM | 验证集 Pearson r | 能量比 | "
        "峰值归一化 MSE |"
    )
    lines = [header, "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| `{row['variant']}` | {row['describe']} | "
            f"{row['ssim_foot']:.4f} | {row['pearson_r']:.4f} | "
            f"{row['energy_ratio_foot']:.4f} | {row['mse_foot']:.3e} |"
        )
    return "\n".join(lines)


def _run_bakeoff(run_dir: Path) -> dict[str, Any]:
    """在一次运行的探针上跑完所有变体, 返回该次运行的结果。"""
    probes_dir = run_dir / "probes"
    geometry = json.loads(
        (run_dir / "bench_geometry.json").read_text(encoding="utf-8")
    )
    far_field_size = int(geometry["far_field_size"])
    camera_px_per_cell = float(geometry["camera_px_per_model_px"])
    source_amplitude = np.load(run_dir / "source_amplitude.npy")

    train_raw, holdout, flat_frame = _load_probes(probes_dir)
    roi_center = _roi_center(flat_frame)
    camera_frame_size = int(np.asarray(flat_frame).shape[0])

    # 重建训练集探针的 model_grid, 并用验证集已落盘的 model_grid 校验换算。
    train: list[Probe] = []
    for name, phase, frame in train_raw:
        grid = _to_model_grid(
            frame, roi_center, far_field_size, camera_px_per_cell, 1
        )
        train.append(Probe(phase=phase, measured=grid, name=name))
    if holdout:
        reference = holdout[0]
        rebuilt = _to_model_grid(
            np.asarray(
                np.load(probes_dir / reference.name)["frame"], dtype=np.float64
            ),
            roi_center,
            far_field_size,
            camera_px_per_cell,
            1,
        )
        deviation = float(
            np.mean(np.abs(rebuilt - reference.measured))
            / max(float(np.max(reference.measured)), 1e-30)
        )
        logger.info(f"model_grid rebuild deviation (holdout 0): {deviation:.3e}")
        if deviation > 1e-6:
            raise RuntimeError(
                f"the model-grid rebuild does not reproduce the recorded one "
                f"(relative L1 {deviation:.3e}) -- the geometry mapping drifted"
            )

    logger.info(
        f"probes: {len(train)} train / {len(holdout)} holdout, "
        f"region={train[0].phase.shape[0]}, far_field={far_field_size}"
    )

    rows: list[dict[str, Any]] = []
    clamped_rows: list[dict[str, Any]] = []
    footprint = int(round(camera_frame_size / camera_px_per_cell))
    for variant in VARIANTS:
        config = _build_config(
            run_dir,
            {
                "fourier_weight": variant.fourier_weight,
                "loss_kind": variant.loss_kind,
                "selfcal_affine": variant.affine,
                "selfcal_travel": variant.travel,
            },
        )
        affine = PupilAffine() if variant.affine else None
        travel = TravelScale() if variant.travel else None
        optimizer = _make_optimizer(
            config, None, source_amplitude, affine=affine, travel=travel
        )
        frames = [(probe.measured, probe.phase) for probe in train]
        coefficients, losses, iterations, optimizer = _fit_aberration_at_probes(
            optimizer, frames, int(config.step_a_iterations)
        )
        metrics = _evaluate(
            optimizer, coefficients, holdout, source_amplitude, footprint
        )
        train_metrics = _evaluate(
            optimizer, coefficients, train, source_amplitude, footprint
        )
        # The MIL loop does NOT commit the raw fit: it clamps the per-round
        # coefficient step with the trust region first. The same clamp is
        # applied here so the second table is what the loop would actually
        # commit (and the first table is Step A's raw output).
        clamped, step_l2, clamp_applied = trust_region_clamp(
            coefficients, np.zeros_like(coefficients),
            float(config.trust_region_c_l2),
        )
        clamped_metrics = _evaluate(
            optimizer, clamped, holdout, source_amplitude, footprint
        )
        row: dict[str, Any] = {
            "variant": variant.name,
            "describe": variant.describe(),
            "iterations": int(iterations),
            "train_loss_final": float(losses[-1]) if losses else float("nan"),
            "coeff_norm": float(np.linalg.norm(coefficients)),
            "coeff_step_l2": float(step_l2),
            "clamp_applied": bool(clamp_applied),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{key: value for key, value in metrics.items()},
        }
        state = optimizer.calibration_state()
        row["affine"] = state["affine"]
        row["travel"] = state["travel"]
        rows.append(row)
        clamped_row = {
            "variant": variant.name,
            "describe": variant.describe(),
            "coeff_norm": float(np.linalg.norm(clamped)),
            "coeff_step_l2": float(step_l2),
            "clamp_applied": bool(clamp_applied),
            **clamped_metrics,
        }
        clamped_rows.append(clamped_row)
        logger.info(
            f"{variant.name:16s} val SSIM {metrics['ssim_foot']:.4f}  "
            f"r {metrics['pearson_r']:.4f}  E {metrics['energy_ratio_foot']:.4f}  "
            f"MSE {metrics['mse_foot']:.3e}  |c| {np.linalg.norm(coefficients):.3f}"
        )
    return {
        "run_dir": str(run_dir),
        "roi_center": list(roi_center),
        "far_field_size": far_field_size,
        "camera_px_per_cell": camera_px_per_cell,
        "camera_frame_size": camera_frame_size,
        "footprint_cells": footprint,
        "n_train": len(train),
        "n_holdout": len(holdout),
        "rows": rows,
        "clamped_rows": clamped_rows,
    }


def _aggregate(runs: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """把多次运行的结果按变体平均 (逐指标)。"""
    metric_keys = [
        "ssim",
        "ssim_foot",
        "pearson_r",
        "energy_ratio",
        "energy_ratio_foot",
        "mse",
        "mse_foot",
    ]
    out: list[dict[str, Any]] = []
    for index, variant in enumerate(VARIANTS):
        merged: dict[str, Any] = {
            "variant": variant.name,
            "describe": variant.describe(),
        }
        for metric in metric_keys:
            values = [
                float(run[key][index][metric])
                for run in runs
                if metric in run[key][index]
            ]
            merged[metric] = float(np.mean(values)) if values else float("nan")
        out.append(merged)
    return out


def main(argv: list[str] | None = None) -> int:
    """跑对比择优并落盘结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        nargs="+",
        default=["data/debug/slm_model_in_loop/20261019_103208"],
        help="一次或多次硬件 model-in-loop 运行的目录。",
    )
    parser.add_argument(
        "--out-dir",
        default="logs/selfcal_bakeoff",
        help="结果落盘目录 (默认在 logs/ 下, 已被 .gitignore 忽略)。",
    )
    args = parser.parse_args(argv)

    runs: list[dict[str, Any]] = []
    for raw in args.run_dir:
        run_dir = Path(raw)
        logger.info(f"\n=== {run_dir} ===")
        result = _run_bakeoff(run_dir)
        runs.append(result)
        logger.info("-- Step A 原始拟合 --")
        logger.info(_render_table(result["rows"]))
        logger.info("-- 经逐轮信任域后 --")
        logger.info(_render_table(result["clamped_rows"]))

    rows = _aggregate(runs, "rows")
    clamped_rows = _aggregate(runs, "clamped_rows")
    table = _render_table(rows)
    clamped_table = _render_table(clamped_rows)
    logger.info()
    logger.info(f"== Step A 原始拟合 (未经信任域), {len(runs)} 次运行平均 ==")
    logger.info(table)
    logger.info()
    logger.info(f"== 经逐轮信任域后 (MIL 循环实际提交的), {len(runs)} 次运行平均 ==")
    logger.info(clamped_table)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "bakeoff.csv"
    fields_out = [
        "variant",
        "describe",
        "iterations",
        "train_loss_final",
        "coeff_norm",
        "coeff_step_l2",
        "clamp_applied",
        "train_ssim",
        "train_ssim_foot",
        "train_pearson_r",
        "train_energy_ratio",
        "train_energy_ratio_foot",
        "train_mse",
        "train_mse_foot",
        "ssim",
        "ssim_foot",
        "pearson_r",
        "energy_ratio",
        "energy_ratio_foot",
        "mse",
        "mse_foot",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_out)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields_out})
    json_path = out_dir / "bakeoff.json"
    json_path.write_text(
        json.dumps(
            {
                "runs": runs,
                "rows": rows,
                "clamped_rows": clamped_rows,
                "table_markdown": table,
                "clamped_table_markdown": clamped_table,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _plot(rows, out_dir / "bakeoff.png")
    logger.info(f"\nwrote {csv_path} and {json_path}")
    return 0


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    """画一张对比图 (SSIM / r / 能量比 三联)。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = [row["variant"] for row in rows]
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    panels = (
        ("ssim_foot", "验证集 SSIM (相机视场)"),
        ("pearson_r", "验证集 Pearson r (整幅)"),
        ("energy_ratio_foot", "能量比 (预测/实测, 相机视场)"),
    )
    for axis, (key, title) in zip(axes, panels):
        values = [float(row[key]) for row in rows]
        axis.bar(positions, values, color="#4c72b0")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
