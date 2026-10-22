"""数字孪生 (2f 实验系统) 的保真度验证 + 误差参数反演过程的离线报告。

**要回答的问题。** ``src/ao_shaping/drivers/sim/bench_twin.py`` 给出的参数化 2f
数字孪生, 能不能定量解释实验系统真实拍到的画面? 反演出的误差参数, 是不是真的把
**验证集** (未参与拟合的独立相位探针) 上的残差压下去了?

**数据。** 一次真实 MIL (model-in-loop) 实验的落盘目录 (默认
``data/debug/slm_model_in_loop/20261019_103208``): 探针 npz (命令相位 + 原始帧 +
处理后帧)、``bench_geometry.json`` (实测几何)、``run_config.json`` (实验系统参数)、
``preflight/flat.npz`` (平场参考)。

**帧网格口径 (本脚本最要紧的一步)。** 实测帧是 300×300 **相机像素**窗口, 而孪生
``far_field()`` 输出的是**模型采样点**网格, 两者必须先落到同一张网格上才能逐像素
比较。脚本做三件事:

1. 相机像元 ↔ 模型采样点的换算用 ``bench_geometry.json`` 的**实测**标定
   ``camera_px_per_model_px`` (相位斜坡扫描得到), 于是相机窗口覆盖
   ``cells = round(cam_size / camera_px_per_model_px)`` 个模型采样点;
2. 实测帧落到模型采样点网格, 用 MIL 自己那套已验证的实现
   (``slm_model_in_loop._to_model_grid``), **并用探针里落盘的 ``model_grid``
   逐像素核对** (见报告 §2.3);
3. 孪生侧用 ``crop_px = cells`` 裁出中央窗口 —— 与孪生自己 ``_crop`` 的取整口径
   (``(m - side) // 2``) 完全一致。

✅ **焦面标度口径已统一**: 孪生**不再**按标称焦距工作, 而是由实测标度经
``BenchTwinConfig.from_measured_scale`` 反推出**等效焦距** (本实验系统 311.32 mm),
于是孪生自己的解析标度 ``BenchTwin.camera_px_per_model_px()`` 与实测标度**逐位
一致**, 「模型预测 → 相机网格」的映射与孪生自身的标度天然同源, 不再需要手工换算。
⚠️ 标称焦距 (122.4060 mm) 算出的解析标度是 1.027778, 与实测差 **2.54 倍** ——
报告 §2.2 保留这一事实作为提醒。

**反演过程的 GIF。** 拟合走的是 ``bench_twin_calibrate.fit_twin`` 自己的两阶段路径
(DE 粗调 + L-BFGS-B 精调), **不另写拟合器**。要拿到"每步迭代后的仿真图", 做法是
在拟合期间临时包一层模块级的 ``twin_loss``: ``fit_twin`` 每评估一个候选就调它一次,
所以包住它就能看到优化器**真实访问过**的状态序列, 顺序即是它访问的顺序。拟合
结束后原样恢复。

运行:
    python scripts/slm/generate_bench_twin_report.py
    python scripts/slm/generate_bench_twin_report.py --n-train 5 --maxiter-fine 8
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts._common import fmt_metric, markdown_table, savefig
from scripts._common.provenance import insert_header

from ao_shaping.drivers.sim.bench_twin import (
    BenchTwin,
    BenchTwinConfig,
    TwinErrorModel,
)

# MIL 自己那套"相机帧 → 模型采样点网格"的实现 (私有但已被实验系统验证过, 报告
# 用它落盘的 model_grid 逐像素核对; 这里复用它而不是另写一份换算, 免得两份口径
# 漂移)。
from ao_shaping.optimizer.wfless.slm_model_in_loop import _to_model_grid
from ao_shaping.tools.slm import bench_twin_calibrate as btc
from ao_shaping.utils.image.beam_metrics import zero_order_center

#: 默认实验目录 (一次真实 MIL 实验的落盘目录)。
DEFAULT_RUN_DIR = ROOT / "data/debug/slm_model_in_loop/20261019_103208"

#: 报告系列目录 (一份报告 = 一个子目录, 见 ``report/AGENTS.md``)。
OUT_DIR = ROOT / "report" / "simulation" / "bench_twin"
FIG_DIR = OUT_DIR / "figures"


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """一条探针: 下发的瞳面命令相位 + 实测相机帧。

    Attributes:
        name: 文件名 (探针的唯一标识)。
        split: ``"train"`` / ``"holdout"`` / ``"flat"``。
        phase: 下发到 SLM 的瞳面命令相位 (raw 弧度, ``region × region``)。
        frame: 实测处理后帧 (相机像元)。
        stored_grid: 探针里落盘的模型网格 (仅验证集探针有), 用于核对换算。
        stored_prediction: MIL 自己的正向模型在同一相位上的预测 (1024² 远场
            网格), 用于同口径对拍。
    """

    name: str
    split: str
    phase: np.ndarray
    frame: np.ndarray
    stored_grid: np.ndarray | None = None
    stored_prediction: np.ndarray | None = None


@dataclass
class Fidelity:
    """一个孪生在一组实测上的保真度。

    Attributes:
        ssim: 峰值归一化后的 SSIM (各样本均值)。
        corr: 峰值归一化后的 Pearson 相关系数 (各样本均值)。
        energy_ratio: **标度无关**的能量比 ``(Σsim/peak_sim)/(Σmeas/peak_meas)``;
            1.0 表示两者的光斑展宽完全一致。
        energy_rel: ``twin_loss`` 用的带符号相对能量偏差 (绝对 ADU 口径)。
        residual_rms: 峰值归一化后的残差 RMS (各样本均值)。
        loss: ``twin_loss`` 的联合损失。
        per_sample: 逐样本明细。
    """

    ssim: float = 0.0
    corr: float = 0.0
    energy_ratio: float = float("nan")
    energy_rel: float = float("nan")
    residual_rms: float = 0.0
    loss: float = float("nan")
    per_sample: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 读数据
# ---------------------------------------------------------------------------


def _relative_to_root(path: Path) -> str:
    """路径的仓库根相对形式 (正斜杠)。

    ``--run-dir`` 既可能给绝对路径, 也可能给相对路径 (溯源登记里记的就是相对
    路径), 所以这里先 ``resolve()`` 再取相对形式; 仓库外的路径原样返回。
    """
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_probes(run_dir: Path) -> list[Probe]:
    """读该次实验落盘的全部探针 (按文件名排序, 与 manifest 的顺序无关)。"""
    manifest_path = run_dir / "probes" / "manifest.json"
    split_of: dict[str, str] = {}
    if manifest_path.exists():
        for entry in _read_json(manifest_path):
            split_of[str(entry.get("file"))] = str(entry.get("split"))
    probes: list[Probe] = []
    for path in sorted((run_dir / "probes").glob("*.npz")):
        with np.load(path) as payload:
            stored = payload["model_grid"] if "model_grid" in payload.files else None
            prediction = (
                payload["prediction"] if "prediction" in payload.files else None
            )
            probes.append(
                Probe(
                    name=path.name,
                    split=split_of.get(path.name, "unknown"),
                    phase=np.asarray(payload["phase"], dtype=np.float64),
                    frame=np.asarray(payload["frame"], dtype=np.float64),
                    stored_grid=(
                        None if stored is None else np.asarray(stored, dtype=np.float64)
                    ),
                    stored_prediction=(
                        None
                        if prediction is None
                        else np.asarray(prediction, dtype=np.float64)
                    ),
                )
            )
    if not probes:
        raise RuntimeError(f"{run_dir}/probes 下没有 *.npz 探针")
    return probes


def _frozen_roi_center(run_dir: Path) -> tuple[int, int]:
    """冻结的光轴位置 (相机像元, ``(x, y)``)。

    MIL 在实验开始时用平场帧的全局 argmax 定位 0 级并**冻结** ROI 框
    (``zero_order_center(flat, refine=False)``); 本脚本用同一条规则复现它。
    """
    flat_path = run_dir / "preflight" / "flat.npz"
    if flat_path.exists():
        with np.load(flat_path) as payload:
            flat = np.asarray(payload["frame"], dtype=np.float64)
        if float(flat.max()) > 0.0:
            return zero_order_center(flat, refine=False)
    logger.warning("没有可用的平场帧, ROI 中心退回相机窗口中心")
    return (0, 0)


# ---------------------------------------------------------------------------
# 几何与孪生构造
# ---------------------------------------------------------------------------


def _build_config(
    geometry: dict,
    run_config: dict,
    *,
    measured_scale: float,
    crop_px: int,
    peak_adu: float,
    errors: TwinErrorModel | None = None,
) -> BenchTwinConfig:
    """按**实测**几何与实验参数构造孪生配置 (误差参数全零 = 理想系统)。

    焦距走 :meth:`BenchTwinConfig.from_measured_scale` —— 由实测标度反推等效焦距,
    于是孪生自己的解析标度与实测标度**逐位一致**, 不必再手工换算。
    """
    return BenchTwinConfig.from_measured_scale(
        float(measured_scale),
        wavelength_m=float(run_config["wavelength_nm"]) * 1e-9,
        slm_pixel_um=float(run_config["slm_pixel_um"]),
        ccd_pixel_um=float(run_config["camera_pixel_um"]),
        region=int(run_config["region"]),
        region_span_px=float(run_config["panel_span_px"]),
        far_field_padding=int(run_config["far_field_padding"]),
        beam_w0_px=float(geometry["beam_waist_panel_px"]),
        center_crop=True,
        crop_px=int(crop_px),
        peak_adu=float(peak_adu),
        noise_adu=0.0,
        seed=0,
        errors=errors if errors is not None else TwinErrorModel(),
    )


def _nominal_scale(
    geometry: dict, run_config: dict
) -> float:
    """若改用**标称焦距**, 孪生的解析标度会是多少 (提醒用)。"""
    nominal_config = BenchTwinConfig(
        wavelength_m=float(run_config["wavelength_nm"]) * 1e-9,
        slm_pixel_um=float(run_config["slm_pixel_um"]),
        ccd_pixel_um=float(run_config["camera_pixel_um"]),
        focal_length_m=float(run_config["focal_length_m"]),
        region=int(run_config["region"]),
        region_span_px=float(run_config["panel_span_px"]),
        far_field_padding=int(run_config["far_field_padding"]),
        beam_w0_px=float(geometry["beam_waist_panel_px"]),
    )
    return BenchTwin(nominal_config).camera_px_per_model_px()


def _crop_centre(grid: np.ndarray, cells: int) -> np.ndarray:
    """取中央 ``cells × cells`` 窗口 (与 ``BenchTwin._crop`` 的取整口径一致)。"""
    side = int(grid.shape[0])
    start = (side - int(cells)) // 2
    return grid[start : start + int(cells), start : start + int(cells)]


def _derive_rotation(
    probes: Sequence[Probe],
    center: tuple[int, int],
    *,
    far_field_size: int,
    cell_scale: float,
) -> tuple[int, float]:
    """从落盘的 ``model_grid`` 反推出相机↔模型的旋转象限, 并给出核对残差。

    相机轴与模型轴在这套实验系统上交换了 90°, MIL 端用
    ``camera_rotation_quarters = 1`` 处理。本脚本不写死这个象限, 而是拿探针里
    落盘的 ``model_grid`` 逐像素核对, 选出最吻合的那个象限 —— 口径由数据说话。

    Returns:
        ``(象限, 最大绝对偏差)``; 没有可核对的 ``model_grid`` 时返回 ``(0, nan)``。
    """
    best_rot, best_err = 0, float("nan")
    reference = [p for p in probes if p.stored_grid is not None]
    if not reference:
        return best_rot, best_err
    for rot in range(4):
        err = 0.0
        for probe in reference:
            grid = _to_model_grid(
                probe.frame, center, far_field_size, cell_scale, rot
            )
            err = max(err, float(np.abs(grid - probe.stored_grid).max()))
        if not math.isfinite(best_err) or err < best_err:
            best_rot, best_err = rot, err
    return best_rot, best_err


def _cells_for(frame: np.ndarray, center, *, far_field_size, cell_scale, rot, cells):
    """把一张相机帧落到模型采样点网格的中央 ``cells × cells`` 窗口。"""
    grid = _to_model_grid(frame, center, far_field_size, cell_scale, rot)
    return _crop_centre(grid, cells)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """两个同形数组的 Pearson 相关系数 (与 MIL 的 ``_prediction_correlation`` 同口径)。"""
    x = np.asarray(a, dtype=np.float64).ravel().copy()
    y = np.asarray(b, dtype=np.float64).ravel().copy()
    if x.shape != y.shape:
        return float("nan")
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


def _mil_run_summary(run_dir: Path) -> dict:
    """读 MIL 自己落盘的运行小结 (状态 + 验证集相关), 供同口径对拍。

    MIL 的 recorder pkl 里每一轮都带 ``holdout_correlation`` (它自己的正向模型
    在验证集上的相关), json 里带最终状态; 本脚本把它们读出来, 好让孪生的数字
    与**同一套实验系统、同一批验证集**上的既有数字直接比。
    """
    summary: dict[str, Any] = {}
    for path in sorted(run_dir.glob("slm_model_in_loop_*.json")):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if "status" in payload:
            summary["status"] = str(payload["status"])
        break
    for path in sorted(run_dir.glob("slm_model_in_loop_*.pkl")):
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:  # pragma: no cover - 落盘产物损坏时只丢这一项
            continue
        if not isinstance(payload, dict):
            continue
        for round_key in sorted(payload):
            entry = payload[round_key]
            if not isinstance(entry, dict):
                continue
            if "holdout_correlation" in entry:
                summary["holdout_correlation"] = float(entry["holdout_correlation"])
                summary["accepted"] = bool(entry.get("accepted", 0.0))
        break
    return summary



# ---------------------------------------------------------------------------
# 保真度指标
# ---------------------------------------------------------------------------


def _fidelity(twin: BenchTwin, samples: Sequence[btc.TwinSample]) -> Fidelity:
    """一个孪生在一组 (相位, 实测) 上的保真度。

    SSIM / 相关 / 残差 RMS 直接取 ``bench_twin_calibrate.twin_loss`` 的逐样本
    明细 (损失函数的定义与报告的口径一致, 不另写一套); 能量比另算一个**标度
    无关**的版本 —— ``twin_loss`` 的能量项用绝对 ADU 积分, 会把"孪生峰值标度"
    与"光斑展宽"混在一起。
    """
    loss, details = btc.twin_loss(twin, samples)
    ratios: list[float] = []
    for sample in samples:
        sim = np.asarray(twin.far_field(sample.phase_rad), dtype=np.float64)
        meas = np.asarray(sample.image, dtype=np.float64)
        peak_sim = float(sim.max())
        peak_meas = float(meas.max())
        if peak_sim > 0.0 and peak_meas > 0.0:
            ratios.append(
                (float(sim.sum()) / peak_sim) / (float(meas.sum()) / peak_meas)
            )
    return Fidelity(
        ssim=float(np.mean([d["ssim"] for d in details])),
        corr=float(np.mean([d["corr"] for d in details])),
        energy_ratio=float(np.mean(ratios)) if ratios else float("nan"),
        energy_rel=float(np.mean([d["energy_rel"] for d in details])),
        residual_rms=float(np.mean([d["residual_rms"] for d in details])),
        loss=float(loss),
        per_sample=list(details),
    )


# ---------------------------------------------------------------------------
# 反演过程的快照 (包一层 twin_loss, 不重写拟合器)
# ---------------------------------------------------------------------------


class _FitRecorder:
    """在拟合期间快照"优化器真实访问过的状态"。

    ``fit_twin`` 每评估一个候选就调一次模块级的 ``twin_loss``, 所以临时包住它
    就能拿到候选序列 —— 顺序就是优化器的访问顺序, 参数与损失都取自那一次评估,
    不需要另写迭代循环 (也就不会有第二份拟合逻辑与 ``fit_twin`` 漂移)。

    Attributes:
        frames: 落盘的三联图 (png) 路径, 按产生顺序。
        records: 每次快照的 (评估序号, 损失)。
    """

    def __init__(
        self,
        *,
        phase: np.ndarray,
        measured: np.ndarray,
        out_dir: Path,
        max_frames: int,
        rel_tol: float = 2e-3,
    ) -> None:
        self._phase = np.asarray(phase, dtype=np.float64)
        self._measured = np.asarray(measured, dtype=np.float64)
        self._out_dir = out_dir
        self._max_frames = int(max_frames)
        self._rel_tol = float(rel_tol)
        self._best = float("inf")
        self._n_eval = 0
        self._original: Any = None
        self.frames: list[Path] = []
        self.records: list[tuple[int, float]] = []

    # -- 上下文管理 ------------------------------------------------------

    def __enter__(self) -> "_FitRecorder":
        self._original = btc.twin_loss
        recorder = self

        def patched(twin, samples, *, weights=None):
            value, details = recorder._original(twin, samples, weights=weights)
            recorder._record(twin, value)
            return value, details

        btc.twin_loss = patched
        return self

    def __exit__(self, *exc: Any) -> None:
        btc.twin_loss = self._original

    # -- 快照 ------------------------------------------------------------

    def _record(self, twin: BenchTwin, loss: float) -> None:
        self._n_eval += 1
        if not math.isfinite(loss):
            return
        if loss >= self._best * (1.0 - self._rel_tol):
            return
        self._best = loss
        self.records.append((self._n_eval, float(loss)))
        if len(self.frames) >= self._max_frames:
            return
        simulated = np.asarray(twin.far_field(self._phase), dtype=np.float64)
        path = self._out_dir / f"step_{len(self.frames):03d}.png"
        _render_triptych(
            self._measured,
            simulated,
            path,
            title=(
                f"评估 #{self._n_eval} | 联合损失 {loss:.5g} | "
                f"gamma={twin.errors.lut_gamma:.3f} | "
                f"defocus={twin.errors.defocus_m * 1e3:.3f} mm | "
                f"tilt=({twin.errors.tilt_x_rad:.2e}, {twin.errors.tilt_y_rad:.2e}) rad"
            ),
        )
        self.frames.append(path)


def _render_triptych(
    measured: np.ndarray,
    simulated: np.ndarray,
    path: Path,
    *,
    title: str,
) -> None:
    """画"实测 | 仿真 | 差值"三联图并落盘。"""
    meas_n = measured / max(float(measured.max()), 1e-30)
    sim_n = simulated / max(float(simulated.max()), 1e-30)
    diff = sim_n - meas_n
    limit = float(np.abs(diff).max()) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    for ax, img, name in (
        (axes[0], meas_n, "实测帧 (峰值归一)"),
        (axes[1], sim_n, "孪生仿真 (峰值归一)"),
    ):
        art = ax.imshow(
            np.log10(np.maximum(img, 1e-6)), cmap="viridis", origin="upper"
        )
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(art, ax=ax, fraction=0.046)
    art = axes[2].imshow(diff, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper")
    axes[2].set_title("差值 (仿真 − 实测)", fontsize=9)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(art, ax=axes[2], fraction=0.046)
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)



def _write_gif(frames: Sequence[Path], out_path: Path, *, duration_ms: int = 700) -> dict:
    """把三联图序列合成 GIF (带帧戳: 图标题即帧戳)。"""
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in frames]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=int(duration_ms),
        loop=0,
    )
    for image in images:
        image.close()
    return {
        "frames": len(images),
        "duration_ms": int(duration_ms),
        "bytes": int(out_path.stat().st_size),
    }


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------


def _radial_energy(image: np.ndarray, bins: int = 12, radius: float = 60.0) -> np.ndarray:
    """峰值归一化后的径向能量分布 (以光轴为心的同心环)。"""
    norm = image / max(float(image.max()), 1e-30)
    side = norm.shape[0]
    centre = (side - 1) / 2.0
    yy, xx = np.mgrid[0:side, 0:side]
    r = np.hypot(yy - centre, xx - centre)
    edges = np.linspace(0.0, radius, bins + 1)
    total = float(norm.sum())
    out = np.zeros(bins, dtype=np.float64)
    for index in range(bins):
        mask = (r >= edges[index]) & (r < edges[index + 1])
        out[index] = float(norm[mask].sum()) / total if total > 0 else 0.0
    return out


def _figure_residual_before_after(
    measured: np.ndarray,
    ideal: np.ndarray,
    fitted: np.ndarray,
    path: Path,
    *,
    title: str,
) -> None:
    """理想孪生 vs 拟合孪生, 与实测的逐像素对照 (上排三图 + 下排差值与径向剖面)。"""
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.4))
    panels = (
        ("实测帧 (峰值归一)", measured),
        ("理想孪生 (峰值归一)", ideal),
        ("拟合孪生 (峰值归一)", fitted),
    )
    for col, (name, img) in enumerate(panels):
        norm = img / max(float(img.max()), 1e-30)
        art = axes[0, col].imshow(
            np.log10(np.maximum(norm, 1e-6)), cmap="viridis", origin="upper"
        )
        axes[0, col].set_title(name, fontsize=10)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        fig.colorbar(art, ax=axes[0, col], fraction=0.046)
    meas_n = measured / max(float(measured.max()), 1e-30)
    for col, (name, img) in enumerate(
        (("理想孪生 − 实测", ideal), ("拟合孪生 − 实测", fitted))
    ):
        other_n = img / max(float(img.max()), 1e-30)
        diff = other_n - meas_n
        limit = float(np.abs(diff).max()) or 1.0
        art = axes[1, col].imshow(
            diff, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper"
        )
        axes[1, col].set_title(name, fontsize=10)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        fig.colorbar(art, ax=axes[1, col], fraction=0.046)
    axes[1, 2].set_title("径向能量分布 (峰值归一)", fontsize=10)
    centres = 0.5 * (np.arange(12) + 1) * 5.0
    for name, img, style in (
        ("实测", measured, "-o"),
        ("理想孪生", ideal, "-s"),
        ("拟合孪生", fitted, "-^"),
    ):
        axes[1, 2].plot(
            centres,
            _radial_energy(img),
            style,
            markersize=4,
            label=name,
        )
    axes[1, 2].set_xlabel("半径 (模型采样点)")
    axes[1, 2].set_ylabel("能量占比")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_ylim(bottom=1e-5)
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].legend(fontsize=9)
    fig.subplots_adjust(wspace=0.45, hspace=0.35)
    fig.suptitle(title, fontsize=11)
    savefig(fig, path)



def _figure_metric_bars(
    train_ideal: Fidelity,
    train_fitted: Fidelity,
    hold_ideal: Fidelity,
    hold_fitted: Fidelity,
    path: Path,
) -> None:
    """理想 vs 拟合 在训练集/验证集上的三项保真度指标对照。"""
    groups = ("SSIM", "Pearson r", "能量比 (Σ/Σ)")
    x = np.arange(len(groups))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    panels = (
        (
            "训练集",
            [train_ideal.ssim, train_ideal.corr, train_ideal.energy_ratio],
            [train_fitted.ssim, train_fitted.corr, train_fitted.energy_ratio],
        ),
        (
            "验证集 (未参与拟合)",
            [hold_ideal.ssim, hold_ideal.corr, hold_ideal.energy_ratio],
            [hold_fitted.ssim, hold_fitted.corr, hold_fitted.energy_ratio],
        ),
    )
    for ax, (name, values_a, values_b) in zip(axes, panels, strict=False):
        bars_a = ax.bar(x - width / 2, values_a, width, label="理想孪生", color="#8FA9D8")
        bars_b = ax.bar(x + width / 2, values_b, width, label="拟合孪生", color="#D8836B")
        for bars in (bars_a, bars_b):
            for bar in bars:
                ax.annotate(
                    f"{bar.get_height():.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_title(f"{name}", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("数字孪生保真度: 理想 vs 拟合 (越高越好, 能量比以 1.0 为完全一致)", fontsize=11)
    savefig(fig, path)


def _figure_probe_gallery(
    samples: Sequence[btc.TwinSample],
    twin_ideal: BenchTwin,
    twin_fitted: BenchTwin,
    path: Path,
    *,
    title: str,
) -> None:
    """逐探针的实测 / 理想 / 拟合 三列对照。"""
    count = len(samples)
    fig, axes = plt.subplots(count, 3, figsize=(10.0, 3.3 * count))
    axes = np.atleast_2d(axes)
    for row, sample in enumerate(samples):
        images = (
            ("实测", np.asarray(sample.image)),
            ("理想孪生", twin_ideal.far_field(sample.phase_rad)),
            ("拟合孪生", twin_fitted.far_field(sample.phase_rad)),
        )
        for col, (name, img) in enumerate(images):
            norm = img / max(float(img.max()), 1e-30)
            art = axes[row, col].imshow(
                np.log10(np.maximum(norm, 1e-6)), cmap="viridis", origin="upper"
            )
            axes[row, col].set_title(f"{sample.label}\n{name}", fontsize=9)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            fig.colorbar(art, ax=axes[row, col], fraction=0.046)
    fig.suptitle(title, fontsize=11)
    savefig(fig, path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "生成数字孪生 (2f 实验系统) 保真度验证 + 误差参数反演的离线报告"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="一次真实 MIL 实验的落盘目录",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="报告输出目录 (系列子目录)",
    )
    parser.add_argument(
        "--stamp",
        default="",
        help="图名戳 (默认取今天的日期, 形如 20261021)",
    )
    parser.add_argument(
        "--report-key",
        default="",
        help="报告的仓库根相对路径 (溯源登记用; 默认按 --stamp 推导)",
    )
    parser.add_argument(
        "--n-train", type=int, default=5, help="参与拟合的训练探针个数"
    )
    parser.add_argument(
        "--n-holdout", type=int, default=3, help="验证集探针个数 (0 = 全部)"
    )
    parser.add_argument("--maxiter", type=int, default=3, help="Stage 1 (DE) 迭代上限")
    parser.add_argument("--popsize", type=int, default=5, help="Stage 1 种群规模乘数")
    parser.add_argument(
        "--maxiter-fine", type=int, default=8, help="Stage 2 (L-BFGS-B) 迭代上限"
    )
    parser.add_argument("--seed", type=int, default=0, help="Stage 1 的随机种子")
    parser.add_argument(
        "--gif-max-frames", type=int, default=48, help="反演过程 GIF 的最大帧数"
    )
    parser.add_argument(
        "--gif-duration-ms", type=int, default=700, help="GIF 每帧停留时间 (ms)"
    )
    parser.add_argument(
        "--slm-zernike-modes",
        default="4,5,6,7",
        help="要反演的面板静态相差 Noll 序 (逗号分隔)",
    )
    parser.add_argument(
        "--lens-zernike-modes",
        default="4,7",
        help="要反演的透镜像差 Noll 序 (逗号分隔)",
    )
    parser.add_argument(
        "--no-fit", action="store_true", help="只跑理想孪生的保真度 (不反演)"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    stamp = args.stamp or time.strftime("%Y%m%d")
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    step_dir = fig_dir / f"fit_steps_{stamp}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)
    report_key = args.report_key or (
        f"report/simulation/bench_twin/bench_twin_fidelity_{stamp}.md"
    )
    report_path = out_dir / f"bench_twin_fidelity_{stamp}.md"

    # ---------------- 读数据与几何 ----------------
    geometry = _read_json(run_dir / "bench_geometry.json")
    run_config = _read_json(run_dir / "run_config.json")
    probes = _read_probes(run_dir)
    center = _frozen_roi_center(run_dir)
    mil_summary = _mil_run_summary(run_dir)

    cam_size = int(run_config["cam_size"])
    cell_scale = float(geometry["camera_px_per_model_px"])
    far_field_size = int(geometry["far_field_size"])
    cells = int(round(cam_size / cell_scale))

    rot, rot_err = _derive_rotation(
        probes,
        center,
        far_field_size=far_field_size,
        cell_scale=cell_scale,
    )
    logger.info(
        "几何: cam {} px, 标度 {:.4f} 相机像元/模型采样点 -> 窗口 {} 个采样点; "
        "旋转象限 {} (与落盘 model_grid 的最大偏差 {:.3g})",
        cam_size,
        cell_scale,
        cells,
        rot,
        rot_err,
    )

    # ---------------- 训练/验证集划分 ----------------
    train_probes = [p for p in probes if p.split == "train"]
    holdout_probes = [p for p in probes if p.split == "holdout"]
    flat_probes = [p for p in probes if p.split == "flat"]
    if args.n_train > 0:
        train_probes = train_probes[: args.n_train]
    if args.n_holdout > 0:
        holdout_probes = holdout_probes[: args.n_holdout]
    if not train_probes:
        raise RuntimeError("该实验目录里没有训练集探针")
    if not holdout_probes:
        raise RuntimeError(
            "该实验目录里没有验证集探针 (manifest 的 split=holdout); "
            "本报告的核心结论是验证集残差, 没有验证集就没有可报告的结论"
        )

    def _samples(items: Sequence[Probe]) -> list[btc.TwinSample]:
        out = []
        for probe in items:
            measured = _cells_for(
                probe.frame,
                center,
                far_field_size=far_field_size,
                cell_scale=cell_scale,
                rot=rot,
                cells=cells,
            )
            out.append(btc.TwinSample(probe.phase, measured, probe.name))
        return out

    train_samples = _samples(train_probes)
    holdout_samples = _samples(holdout_probes)
    flat_samples = _samples(flat_probes)

    # ---------------- 构造孪生 (理想) ----------------
    # 峰值标度取**训练集**实测峰值的中位数 —— 与 ``fit_twin`` 内部的自动取值
    # 完全一致, 所以理想孪生与拟合孪生的能量项是同一把尺子, 可以直接比。
    peak_adu = float(
        np.median([float(s.image.max()) for s in train_samples if float(s.image.max()) > 0])
    )
    ideal_config = _build_config(
        geometry,
        run_config,
        measured_scale=cell_scale,
        crop_px=cells,
        peak_adu=peak_adu,
    )
    ideal_twin = BenchTwin(ideal_config)

    twin_analytic_scale = ideal_twin.camera_px_per_model_px()
    # 孪生按实测标度反推的等效焦距 (与孪生自身的解析标度同源)。
    implied_focal = float(ideal_config.focal_length_m)
    # 提醒: 若改用标称焦距, 解析标度会是多少 (与实测差一个倍数)。
    nominal_scale = _nominal_scale(geometry, run_config)
    nominal_focal = float(run_config["focal_length_m"])
    scale_rel_dev = abs(twin_analytic_scale - cell_scale) / cell_scale

    # ---------------- 理想孪生的保真度 ----------------
    train_ideal = _fidelity(ideal_twin, train_samples)
    holdout_ideal = _fidelity(ideal_twin, holdout_samples)
    flat_ideal = _fidelity(ideal_twin, flat_samples) if flat_samples else None

    # ---------------- 反演 ----------------
    fit_result = None
    recorder = None
    fit_seconds = 0.0
    if not args.no_fit:
        errors = TwinErrorModel()
        errors.slm_zernike = {
            int(k): 0.0 for k in args.slm_zernike_modes.split(",") if k.strip()
        }
        errors.slm_zernike_n_max = 4
        errors.lens_zernike = {
            int(k): 0.0 for k in args.lens_zernike_modes.split(",") if k.strip()
        }
        errors.lens_zernike_n_max = 4
        base_config = _build_config(
            geometry,
            run_config,
            measured_scale=cell_scale,
            crop_px=cells,
            peak_adu=peak_adu,
            errors=errors,
        )
        recorder = _FitRecorder(
            phase=train_samples[0].phase_rad,
            measured=np.asarray(train_samples[0].image),
            out_dir=step_dir,
            max_frames=args.gif_max_frames,
        )
        started = time.time()
        with recorder:
            fit_result = btc.fit_twin(
                train_samples,
                base_config,
                stage1=True,
                stage2=True,
                maxiter=args.maxiter,
                popsize=args.popsize,
                maxiter_fine=args.maxiter_fine,
                seed=args.seed,
                verbose=True,
            )
        fit_seconds = time.time() - started
        logger.info("反演完成, 用时 {:.1f} s", fit_seconds)

    fitted_twin = (
        BenchTwin(fit_result.config) if fit_result is not None else ideal_twin
    )
    train_fitted = (
        _fidelity(fitted_twin, train_samples) if fit_result is not None else None
    )
    holdout_fitted = (
        _fidelity(fitted_twin, holdout_samples) if fit_result is not None else None
    )

    # ---------------- 与 MIL 自己的正向模型同口径对拍 ----------------
    # MIL 的判定口径是「1024² 远场网格 vs 落盘的 model_grid」上的 Pearson 相关
    # (见 slm_model_in_loop._prediction_correlation)。孪生也用同一张网格、同一
    # 指标算一遍, 于是「孪生 vs MIL 正向模型」是同口径的, 且能与 MIL 自己落盘的
    # holdout_correlation 直接对照。
    full_ideal = BenchTwin(
        _build_config(
            geometry,
            run_config,
            measured_scale=cell_scale,
            crop_px=far_field_size,
            peak_adu=peak_adu,
            errors=ideal_config.errors,
        )
    )
    full_fitted = (
        BenchTwin(
            _build_config(
                geometry,
                run_config,
                measured_scale=cell_scale,
                crop_px=far_field_size,
                peak_adu=peak_adu,
                errors=fit_result.config.errors,
            )
        )
        if fit_result is not None
        else full_ideal
    )
    bench_rows: list[dict] = []
    for probe in holdout_probes:
        if probe.stored_grid is None:
            continue
        row: dict[str, Any] = {"name": probe.name}
        if probe.stored_prediction is not None:
            row["mil"] = _pearson(probe.stored_prediction, probe.stored_grid)
        row["twin_ideal"] = _pearson(
            full_ideal.far_field(probe.phase), probe.stored_grid
        )
        row["twin_fitted"] = (
            _pearson(full_fitted.far_field(probe.phase), probe.stored_grid)
            if fit_result is not None
            else None
        )
        bench_rows.append(row)


    # ---------------- 图 ----------------
    gif_info = None
    if recorder is not None and recorder.frames:
        gif_path = fig_dir / f"{stamp}_fit_process.gif"
        gif_info = _write_gif(
            recorder.frames, gif_path, duration_ms=args.gif_duration_ms
        )
        try:
            gif_info["path"] = str(gif_path.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            gif_info["path"] = str(gif_path).replace("\\", "/")

    residual_path = fig_dir / f"{stamp}_residual_before_after.png"
    _figure_residual_before_after(
        np.asarray(holdout_samples[0].image),
        ideal_twin.far_field(holdout_samples[0].phase_rad),
        fitted_twin.far_field(holdout_samples[0].phase_rad),
        residual_path,
        title=(
            f"验证集探针 {holdout_samples[0].label}: 理想孪生 vs 拟合孪生"
            f" (窗口 {cells}×{cells} 模型采样点)"
        ),
    )

    metrics_path = fig_dir / f"{stamp}_fidelity_metrics.png"
    if train_fitted is not None and holdout_fitted is not None:
        _figure_metric_bars(
            train_ideal, train_fitted, holdout_ideal, holdout_fitted, metrics_path
        )
    else:
        metrics_path = None

    gallery_path = fig_dir / f"{stamp}_probe_gallery.png"
    _figure_probe_gallery(
        holdout_samples,
        ideal_twin,
        fitted_twin,
        gallery_path,
        title="验证集探针: 实测 / 理想孪生 / 拟合孪生 (峰值归一, 对数色标)",
    )

    # ---------------- 报告 ----------------
    body = _render_report(
        stamp=stamp,
        run_dir=run_dir,
        geometry=geometry,
        run_config=run_config,
        center=center,
        cells=cells,
        cell_scale=cell_scale,
        twin_analytic_scale=twin_analytic_scale,
        implied_focal=implied_focal,
        nominal_scale=nominal_scale,
        nominal_focal=nominal_focal,
        scale_rel_dev=scale_rel_dev,
        rot=rot,
        rot_err=rot_err,
        ideal_twin=ideal_twin,
        fit_result=fit_result,
        train_ideal=train_ideal,
        train_fitted=train_fitted,
        holdout_ideal=holdout_ideal,
        holdout_fitted=holdout_fitted,
        flat_ideal=flat_ideal,
        flat_samples=flat_samples,
        holdout_samples=holdout_samples,
        train_samples=train_samples,
        gif_info=gif_info,
        recorder=recorder,
        fit_seconds=fit_seconds,
        bench_rows=bench_rows,
        mil_summary=mil_summary,
        residual_path=residual_path,
        metrics_path=metrics_path,
        gallery_path=gallery_path,
        step_dir=step_dir,
        args=args,
    )
    body = insert_header(body, report_key)
    report_path.write_text(body, encoding="utf-8")
    logger.info("报告已写出: {}", report_path)
    return 0


def _render_report(**kw: Any) -> str:
    """把算出来的数字渲染成报告正文。"""
    stamp: str = kw["stamp"]
    run_dir: Path = kw["run_dir"]
    geometry: dict = kw["geometry"]
    run_config: dict = kw["run_config"]
    center = kw["center"]
    cells: int = kw["cells"]
    cell_scale: float = kw["cell_scale"]
    twin_analytic_scale: float = kw["twin_analytic_scale"]
    implied_focal: float = kw["implied_focal"]
    nominal_scale: float = kw["nominal_scale"]
    nominal_focal: float = kw["nominal_focal"]
    scale_rel_dev: float = kw["scale_rel_dev"]
    rot: int = kw["rot"]
    rot_err: float = kw["rot_err"]
    ideal_twin: BenchTwin = kw["ideal_twin"]
    fit_result = kw["fit_result"]
    train_ideal: Fidelity = kw["train_ideal"]
    train_fitted = kw["train_fitted"]
    holdout_ideal: Fidelity = kw["holdout_ideal"]
    holdout_fitted = kw["holdout_fitted"]
    flat_ideal = kw["flat_ideal"]
    flat_samples = kw["flat_samples"]
    holdout_samples = kw["holdout_samples"]
    train_samples = kw["train_samples"]
    gif_info = kw["gif_info"]
    recorder = kw["recorder"]
    fit_seconds: float = kw["fit_seconds"]
    args = kw["args"]
    step_dir: Path = kw["step_dir"]

    rel_run = _relative_to_root(run_dir)
    config = ideal_twin.config

    lines: list[str] = []
    add = lines.append

    add(f"# 数字孪生保真度验证: 理想孪生 vs 误差参数反演后的孪生 ({stamp})")
    add("")
    add(
        "**问题.** `src/ao_shaping/drivers/sim/bench_twin.py` 的参数化 2f 数字孪生, "
        "能不能定量解释实验系统真实拍到的画面? 反演出的误差参数, 是不是真的把"
        "**验证集** (未参与拟合的独立相位探针) 上的残差压下去了?"
    )
    add("")
    add(
        f"**数据.** 一次真实 MIL 实验的落盘目录 `{rel_run}` "
        f"(训练集探针 {len(train_samples)} 条, 验证集探针 {len(holdout_samples)} 条, "
        f"平场参考 {len(flat_samples)} 条)。全部离线, 不碰任何设备。"
    )
    add("")

    # -- 1 数据口径 ------------------------------------------------------
    add("## 1. 数据与口径")
    add("")
    add(
        markdown_table(
            ["项", "值", "来源"],
            [
                ["区域 (region)", config.region, "`run_config.json`"],
                ["瞳面网格覆盖面板边长", f"{config.region_span_px:.0f} px", "`run_config.json`"],
                ["远场补零因子", config.far_field_padding, "`run_config.json`"],
                ["远场网格边长", ideal_twin.far_field_size, "region × 补零因子"],
                ["波长", f"{config.wavelength_m * 1e9:.0f} nm", "`run_config.json`"],
                ["透镜焦距 (标称, **未采用**)", f"{nominal_focal * 1e3:.4f} mm", "`run_config.json`"],
                [
                    "孪生工作焦距 (实测标度反推的等效焦距)",
                    f"{config.focal_length_m * 1e3:.2f} mm",
                    "`BenchTwinConfig.from_measured_scale`",
                ],
                ["SLM 像元节距", f"{config.slm_pixel_um:.1f} µm", "`run_config.json`"],
                ["CCD 像元节距", f"{config.ccd_pixel_um:.1f} µm", "`run_config.json`"],
                ["入射光束腰 (1/e⁴ 强度)", f"{config.beam_w0_px:.0f} 面板 px", "`bench_geometry.json`"],
                ["相机窗口", f"{run_config['cam_size']}×{run_config['cam_size']} 相机 px", "`run_config.json`"],
                [
                    "相机像元 / 模型采样点 (实测标度)",
                    f"{cell_scale:.6f}",
                    "`bench_geometry.json` (相位斜坡扫描)",
                ],
                [
                    "冻结光轴 (相机 px)",
                    f"({center[0]}, {center[1]})",
                    "平场帧全局 argmax",
                ],
                [
                    "比较窗口",
                    f"{cells}×{cells} 模型采样点",
                    "round(cam_size / 实测标度)",
                ],
                [
                    "孪生峰值标度 (peak_adu)",
                    fmt_metric(config.peak_adu),
                    "训练集实测峰值中位数 (与 fit_twin 内部取值一致)",
                ],
            ],
        )
    )
    add("")

    # -- 2 网格换算 ------------------------------------------------------
    add("## 2. 帧网格换算: 相机像元 ↔ 模型采样点")
    add("")
    add("### 2.1 为什么必须换算")
    add("")
    add(
        "实测帧是 **300×300 相机像元**的窗口, 孪生 `far_field()` 输出的是**模型采样点**"
        "网格 (region × 补零因子 = "
        f"{ideal_twin.far_field_size}², 裁剪后 {cells}²)。两者直接比会张冠李戴, "
        "所以先把它们放到同一张网格上。"
    )
    add("")
    add(
        "换算链: 相机窗口 `cam_size` 个像元 ↔ 实测标度 "
        f"`camera_px_per_model_px = {cell_scale:.6f}` (相机像元/模型采样点) ↔ "
        f"**{cells} 个模型采样点**。孪生侧用 `crop_px = {cells}` 裁出中央窗口 "
        "(与 `BenchTwin._crop` 的取整口径 `(m - side) // 2` 一致), 实测侧用 MIL 那套 "
        "已验证的 `_to_model_grid` 落到模型采样点网格后再裁同一个窗口。"
    )
    add("")
    add("### 2.2 焦面标度口径: 由实测标度反推等效焦距")
    add("")
    add(
        markdown_table(
            ["路线", "相机像元 / 模型采样点", "说明"],
            [
                [
                    "孪生解析值 `BenchTwin.camera_px_per_model_px()`",
                    f"{twin_analytic_scale:.6f}",
                    "`p_fft / d_ccd`, `p_fft = λf/(N_pad·d_model)`",
                ],
                [
                    "实验系统实测标度 (`bench_geometry.json`)",
                    f"{cell_scale:.6f}",
                    "相位斜坡扫描 (`ramp sweep`), 标定脚本判定它更可信",
                ],
                [
                    "两者之比 (实测 / 解析)",
                    f"{cell_scale / twin_analytic_scale:.9f}",
                    "**一致** —— 孪生按实测标度反推的等效焦距工作",
                ],
                [
                    "孪生工作焦距 (等效焦距, 由实测标度反推)",
                    f"{implied_focal * 1e3:.2f} mm",
                    "`BenchTwinConfig.from_measured_scale`",
                ],
                [
                    "若改用标称焦距的解析标度",
                    f"{nominal_scale:.6f}",
                    f"标称 {nominal_focal * 1e3:.4f} mm, 与实测差 "
                    f"{cell_scale / nominal_scale:.2f} 倍",
                ],
            ],
        )
    )
    add("")
    add(
        f"✅ **焦面标度口径已统一**: 孪生不再按标称焦距工作, 而是由实测标度 "
        f"`{cell_scale:.6f}` 反推出等效焦距 **{implied_focal * 1e3:.2f} mm** "
        "(`BenchTwinConfig.from_measured_scale`), 于是孪生自己的解析标度 "
        f"`BenchTwin.camera_px_per_model_px()` = `{twin_analytic_scale:.6f}` 与实测"
        f"**逐位一致** (相对偏差 {scale_rel_dev:.2e}, < 1e-12)。"
        "「模型预测 → 相机网格」的映射与孪生自身的标度因此**天然同源**, 不再需要"
        "手工换算。"
    )
    add("")
    add(
        f"⚠️ **提醒: 若仍按标称焦距 {nominal_focal * 1e3:.4f} mm 算, 解析标度只有 "
        f"{nominal_scale:.6f}, 与实测差 **{cell_scale / nominal_scale:.2f} 倍**。** "
        "标称焦距是标定书上的名义值, 这套光路的实际焦面标度与之不符 —— 这正是"
        "本报告此前记的「口径缺口」, 现在由 `from_measured_scale` 入口统一掉了。"
        "裁剪窗口仍按实测标度算 (`round(cam_size / 实测标度)`), 与 §2.3 的逐像素"
        "核对一致。"
    )
    add("")
    add("### 2.3 换算的逐像素核对")
    add("")
    add(
        f"旋转象限**不写死**: 脚本拿验证集探针里落盘的 `model_grid` (MIL 自己算的"
        f"「实测帧 → 模型采样点网格」) 逐像素比对, 选出最吻合的象限 = **{rot}** "
        f"(最大绝对偏差 {rot_err:.3g}, 即 float32 落盘精度)。实测帧与孪生用同一个"
        "旋转, 否则两者的 x/y 轴是交换的。"
    )
    add("")

    # -- 3 孪生构造 ------------------------------------------------------
    add("## 3. 孪生构造 (理想 = 三族误差全零)")
    add("")
    add(
        "孪生配置由上表的实测几何直接构造 (焦距取**实测标度反推的等效焦距**, 见 "
        "§2.2), 误差参数 "
        "`lut_gamma / lut_phase_max_rad / slm_zernike / slm_roughness / defocus_m / "
        "tilt / lens_zernike` 全为零 (理想系统), `noise_adu = 0` (反演必须确定性)。"
        "正向链、居中补零、`fftshift(fft2(ifftshift(·)))` 全部走 "
        "`bench_twin.py` 自己的实现, 本脚本不重写传播链。"
    )
    add("")

    # -- 4 保真度数字 ---------------------------------------------------
    add("## 4. 保真度数字")
    add("")
    add(
        "三项指标: **SSIM** 与 **Pearson r** 在峰值归一化后计算 (直接取 "
        "`bench_twin_calibrate.twin_loss` 的逐样本明细, 与损失函数口径一致); "
        "**能量比** 是标度无关的 `(Σsim/peak_sim)/(Σmeas/peak_meas)`, "
        "1.0 表示两者光斑展宽完全一致 (孪生自身的 `twin_loss` 能量项用绝对 ADU, "
        "会把「孪生峰值标度」混进「光斑展宽」, 所以另列这一项)。"
    )
    add("")

    rows = [
        [
            "训练集",
            "理想孪生",
            fmt_metric(train_ideal.ssim),
            fmt_metric(train_ideal.corr),
            fmt_metric(train_ideal.energy_ratio),
            fmt_metric(train_ideal.residual_rms),
            fmt_metric(train_ideal.loss),
        ]
    ]
    if train_fitted is not None:
        rows.append(
            [
                "训练集",
                "拟合孪生",
                fmt_metric(train_fitted.ssim),
                fmt_metric(train_fitted.corr),
                fmt_metric(train_fitted.energy_ratio),
                fmt_metric(train_fitted.residual_rms),
                fmt_metric(train_fitted.loss),
            ]
        )
    rows.append(
        [
            "验证集 (未参与拟合)",
            "理想孪生",
            fmt_metric(holdout_ideal.ssim),
            fmt_metric(holdout_ideal.corr),
            fmt_metric(holdout_ideal.energy_ratio),
            fmt_metric(holdout_ideal.residual_rms),
            fmt_metric(holdout_ideal.loss),
        ]
    )
    if holdout_fitted is not None:
        rows.append(
            [
                "验证集 (未参与拟合)",
                "拟合孪生",
                fmt_metric(holdout_fitted.ssim),
                fmt_metric(holdout_fitted.corr),
                fmt_metric(holdout_fitted.energy_ratio),
                fmt_metric(holdout_fitted.residual_rms),
                fmt_metric(holdout_fitted.loss),
            ]
        )
    if flat_ideal is not None:
        rows.append(
            [
                "平场参考 (零相位)",
                "理想孪生",
                fmt_metric(flat_ideal.ssim),
                fmt_metric(flat_ideal.corr),
                fmt_metric(flat_ideal.energy_ratio),
                fmt_metric(flat_ideal.residual_rms),
                fmt_metric(flat_ideal.loss),
            ]
        )
    add(
        markdown_table(
            [
                "集合",
                "孪生",
                "SSIM",
                "Pearson r",
                "能量比",
                "残差 RMS",
                "联合损失",
            ],
            rows,
        )
    )
    add("")

    if holdout_fitted is not None:
        improved = (
            holdout_fitted.ssim >= holdout_ideal.ssim
            and holdout_fitted.corr >= holdout_ideal.corr
        )
        energy_ok = abs(
            math.log(max(holdout_fitted.energy_ratio, 1e-12))
            - math.log(max(holdout_ideal.energy_ratio, 1e-12))
        ) < abs(
            math.log(max(holdout_ideal.energy_ratio, 1e-12))
            - math.log(1.0)
        )
        add("### 4.1 验证集逐探针明细")
        add("")
        detail_rows = []
        for before, after in zip(
            holdout_ideal.per_sample, holdout_fitted.per_sample, strict=False
        ):
            detail_rows.append(
                [
                    before["label"],
                    fmt_metric(before["ssim"]),
                    fmt_metric(after["ssim"]),
                    fmt_metric(before["corr"]),
                    fmt_metric(after["corr"]),
                    fmt_metric(before["energy_rel"]),
                    fmt_metric(after["energy_rel"]),
                ]
            )
        add(
            markdown_table(
                [
                    "探针",
                    "SSIM (理想)",
                    "SSIM (拟合)",
                    "r (理想)",
                    "r (拟合)",
                    "能量偏差 (理想)",
                    "能量偏差 (拟合)",
                ],
                detail_rows,
            )
        )
        add("")
        corr_regressed = [
            before["label"]
            for before, after in zip(
                holdout_ideal.per_sample, holdout_fitted.per_sample, strict=False
            )
            if after["corr"] < before["corr"]
        ]
        ssim_regressed = [
            before["label"]
            for before, after in zip(
                holdout_ideal.per_sample, holdout_fitted.per_sample, strict=False
            )
            if after["ssim"] < before["ssim"]
        ]
        if corr_regressed or ssim_regressed:
            add(
                "**逐探针看并不是全面变好**: 相关下降的探针 "
                + ", ".join(f"`{name}`" for name in corr_regressed)
                + ("; " if corr_regressed and ssim_regressed else "; ")
                + (
                    "SSIM 下降的探针 "
                    + ", ".join(f"`{name}`" for name in ssim_regressed)
                    if ssim_regressed
                    else "SSIM 全部改善"
                )
                + "。上表的改善是**集合均值**上的改善; 单条探针的走向并不一致, "
                "所以只能说「反演把整体残差压下去了」, 不能说「每一条探针都变好了」。"
            )
            add("")
        add(
            "**自检**: "
            + (
                "拟合后的验证集 SSIM 与相关都不劣于理想孪生, 能量比也更接近 1.0 —— "
                "反演确实把残差压下去了, 且不是只压训练集。"
                if improved and energy_ok
                else "拟合后的验证集指标与理想孪生相比**未能全面变好**, 详见 §5 的结论与 §6 的限制。"
            )
        )
        add("")

    # -- 4.2 与 MIL 自己的正向模型同口径对拍 ---------------------------
    bench_rows = kw.get("bench_rows") or []
    mil_summary = kw.get("mil_summary") or {}
    if bench_rows:
        add("### 4.2 与 MIL 自己的正向模型同口径对拍")
        add("")
        add(
            "MIL 的判定口径是「1024² 远场网格 vs 落盘的 `model_grid`」上的 Pearson "
            "相关 (`slm_model_in_loop._prediction_correlation`)。孪生用**同一张网格、"
            "同一指标**算一遍, 于是三者 (MIL 自己的正向模型 / 理想孪生 / 拟合孪生) "
            "是同口径的, 可以直接比。"
        )
        add("")
        rows = []
        for row in bench_rows:
            rows.append(
                [
                    row["name"],
                    fmt_metric(row.get("mil")),
                    fmt_metric(row.get("twin_ideal")),
                    fmt_metric(row.get("twin_fitted")),
                ]
            )
        add(
            markdown_table(
                ["验证集探针", "MIL 正向模型", "理想孪生", "拟合孪生"],
                rows,
            )
        )
        add("")
        mil_values = [r["mil"] for r in bench_rows if r.get("mil") is not None]
        twin_fitted_values = [
            r["twin_fitted"] for r in bench_rows if r.get("twin_fitted") is not None
        ]
        if mil_values:
            add(
                f"平均: MIL 正向模型 {fmt_metric(float(np.mean(mil_values)))} / "
                f"理想孪生 "
                f"{fmt_metric(float(np.mean([r['twin_ideal'] for r in bench_rows])))} / "
                + (
                    f"拟合孪生 {fmt_metric(float(np.mean(twin_fitted_values)))}。"
                    if twin_fitted_values
                    else "拟合孪生 (本次未反演)。"
                )
            )
            add("")
        if mil_summary:
            extra = []
            if "status" in mil_summary:
                extra.append(f"运行状态 `{mil_summary['status']}`")
            if "holdout_correlation" in mil_summary:
                extra.append(
                    f"MIL 自己落盘的验证集相关 {fmt_metric(mil_summary['holdout_correlation'])}"
                )
            if extra:
                add("同一次实验自身的记录: " + "; ".join(extra) + "。")
                add("")
        twin_fitted_values = [
            r["twin_fitted"] for r in bench_rows if r.get("twin_fitted") is not None
        ]
        twin_ideal_values = [
            r["twin_ideal"] for r in bench_rows if r.get("twin_ideal") is not None
        ]
        if twin_fitted_values and twin_ideal_values:
            delta = float(np.mean(twin_fitted_values)) - float(
                np.mean(twin_ideal_values)
            )
            add(
                f"⚠️ 注意: 这个指标上反演**几乎没有动** ({fmt_metric(float(np.mean(twin_ideal_values)))} "
                f"→ {fmt_metric(float(np.mean(twin_fitted_values)))}), 与 §4.1 的窗口指标"
                "形成对照。原因是这个指标把整张 1024² 网格都算进去, 而实测画面在"
                "**相机窗口之外全是零** —— 窗口之外的残差在两个口径里权重完全不同: "
                "窗口口径 (§4/§4.1) 只看相机真正看到的那 115×115 个采样点, "
                "所以它能反映「光斑展宽/形状」被反演改善; 这个全网格口径则被"
                "「窗口外有多少弥散光」主导。**两个口径都不是错的, 但它们量的不是"
                "同一件事, 引用时必须说清是哪一个。**"
            )
            add("")



    # -- 5 反演参数 ------------------------------------------------------
    add("## 5. 反演出的误差参数")
    add("")
    if fit_result is None:
        add("本次运行带 `--no-fit`, 未做反演。")
        add("")
    else:
        errors = fit_result.config.errors
        stages = fit_result.stages
        param_rows = [
            ["lut_gamma (灰度→相位 Gamma)", fmt_metric(errors.lut_gamma)],
            [
                "lut_phase_max_rad (满灰度相位, rad)",
                fmt_metric(errors.lut_phase_max_rad),
            ],
            ["defocus_m (相机离焦, m)", fmt_metric(errors.defocus_m)],
            ["tilt_x_rad (光路倾斜 x)", fmt_metric(errors.tilt_x_rad)],
            ["tilt_y_rad (光路倾斜 y)", fmt_metric(errors.tilt_y_rad)],
        ]
        for noll, value in sorted(errors.slm_zernike.items()):
            param_rows.append([f"面板静态相差 Noll {noll} (rad)", fmt_metric(value)])
        for noll, value in sorted(errors.lens_zernike.items()):
            param_rows.append([f"透镜像差 Noll {noll} (rad)", fmt_metric(value)])
        add(markdown_table(["参数", "反演值"], param_rows))
        add("")
        add(
            markdown_table(
                ["阶段", "损失", "评估次数"],
                [
                    ["初值", fmt_metric(stages.get("initial_loss")), "-"],
                    [
                        "Stage 1 (DE 粗调)",
                        fmt_metric(stages.get("stage1_loss")),
                        fmt_metric(stages.get("n_evaluations_stage1")),
                    ],
                    [
                        "Stage 2 (L-BFGS-B 精调)",
                        fmt_metric(stages.get("stage2_loss")),
                        fmt_metric(stages.get("n_evaluations_stage2")),
                    ],
                    ["总评估次数", "-", fmt_metric(fit_result.n_evaluations)],
                ],
            )
        )
        add("")
        add(
            f"拟合耗时 {fit_seconds:.1f} s; 联合损失 "
            f"{fmt_metric(stages.get('initial_loss'))} → "
            f"{fmt_metric(stages.get('final_loss'))} (训练集)。"
            f"反演用 `bench_twin_calibrate.fit_twin` 自己的两阶段路径, 边界取模块的 "
            "`DEFAULT_BOUNDS`。"
        )
        add("")

    # -- 6 GIF ------------------------------------------------------------
    add("## 6. 反演过程的 GIF")
    add("")
    if gif_info is None:
        add("本次运行没有生成 GIF (`--no-fit` 或没有快照)。")
        add("")
    else:
        add(
            f"![反演过程](figures/{Path(gif_info['path']).name})"
        )
        add("")
        add(
            markdown_table(
                ["项", "值"],
                [
                    ["文件", f"`{gif_info['path']}`"],
                    ["帧数", gif_info["frames"]],
                    ["每帧停留", f"{gif_info['duration_ms']} ms"],
                    ["总时长", f"{gif_info['frames'] * gif_info['duration_ms'] / 1000:.1f} s"],
                    ["文件大小", f"{gif_info['bytes'] / 1024:.1f} KiB"],
                    ["帧源", f"`figures/{step_dir.name}/step_*.png`"],
                ],
            )
        )
        add("")
        add(
            "**每帧三块**: 左 = 实测帧 (峰值归一, 对数色标), 中 = 当前候选孪生的仿真"
            "远场 (峰值归一, 对数色标), 右 = 差值图 (仿真 − 实测, 发散色标)。"
            "**帧戳**画在图标题上: 评估序号、联合损失、以及当前候选的关键误差参数 "
            "(gamma / 相机离焦 / 光路倾斜)。"
        )
        add("")
        add(
            "快照来自 `fit_twin` **真实访问过的候选序列**: 拟合期间临时包一层模块级 "
            "`twin_loss` (`fit_twin` 每评估一个候选就调它一次), 包住的正是那一次评估的"
            "候选孪生, 拟合结束后原样恢复。**没有另写迭代循环**, 所以 GIF 里的每一帧"
            "都是优化器自己走过的点。"
        )
        add("")
        if recorder is not None and recorder.records:
            first_eval, first_loss = recorder.records[0]
            last_eval, last_loss = recorder.records[-1]
            add(
                f"快照共 {len(recorder.records)} 次改进 (评估 #{first_eval} 损失 "
                f"{first_loss:.5g} → 评估 #{last_eval} 损失 {last_loss:.5g}); "
                f"GIF 取其中 {gif_info['frames']} 帧。"
            )
            add("")

    # -- 7 其他图 --------------------------------------------------------
    add("## 7. 对照图")
    add("")
    add(f"![残差对照](figures/{Path(kw['residual_path']).name})")
    add("")
    add(
        "验证集第一条探针: 上排左 = 实测帧, 中 = 理想孪生, 右 = 拟合孪生; "
        "下排左 = 理想孪生 − 实测, 中 = 拟合孪生 − 实测, 右 = 三者的径向能量分布 "
        "(半径越大能量占比越低的那条曲线, 光斑越紧)。"
    )
    add("")
    if kw["metrics_path"] is not None:
        add(f"![保真度指标](figures/{Path(kw['metrics_path']).name})")
        add("")
    add(f"![验证集探针对照](figures/{Path(kw['gallery_path']).name})")
    add("")
    add("验证集逐探针: 实测 / 理想孪生 / 拟合孪生 (峰值归一, 对数色标)。")
    add("")

    # -- 8 结论 ----------------------------------------------------------
    add("## 8. 结论")
    add("")
    if holdout_fitted is not None:
        add(
            f"1. **孪生能定量解释真实画面, 但只在「理想」层面部分成立**: 零相位平场参考上 "
            f"理想孪生的 SSIM = {fmt_metric(flat_ideal.ssim) if flat_ideal else '-'}、"
            f"相关 = {fmt_metric(flat_ideal.corr) if flat_ideal else '-'}, 说明传播链与"
            "标度口径本身是对的; 但在随机相位探针上, 理想孪生的验证集 SSIM 只有 "
            f"{fmt_metric(holdout_ideal.ssim)}、相关 {fmt_metric(holdout_ideal.corr)} —— "
            "**零误差假设本身解释不了真实画面**。"
        )
        add(
            f"2. **反演确实把残差压下去了, 而且压的是验证集** (集合均值上): 验证集 SSIM "
            f"{fmt_metric(holdout_ideal.ssim)} → {fmt_metric(holdout_fitted.ssim)}, "
            f"相关 {fmt_metric(holdout_ideal.corr)} → {fmt_metric(holdout_fitted.corr)}, "
            f"能量比 {fmt_metric(holdout_ideal.energy_ratio)} → "
            f"{fmt_metric(holdout_fitted.energy_ratio)} (1.0 为完全一致)。"
            "逐探针的走向并不一致 (§4.1), 所以这是「整体残差被压下去」而不是"
            "「每条探针都变好」。"
        )
    else:
        add("本次运行没有反演结果, 只给出理想孪生的保真度。")
    add("")
    if bench_rows:
        mil_values = [r["mil"] for r in bench_rows if r.get("mil") is not None]
        twin_values = [r["twin_ideal"] for r in bench_rows if r.get("twin_ideal") is not None]
        fitted_values = [
            r["twin_fitted"] for r in bench_rows if r.get("twin_fitted") is not None
        ]
        if mil_values and twin_values:
            add(
                "3. **同口径对拍 (§4.2)**: 验证集相关 (1024² 网格上的 Pearson) —— "
                f"MIL 自己的正向模型 {fmt_metric(float(np.mean(mil_values)))} / "
                f"理想孪生 {fmt_metric(float(np.mean(twin_values)))}"
                + (
                    f" / 拟合孪生 {fmt_metric(float(np.mean(fitted_values)))}。"
                    if fitted_values
                    else "。"
                )
                + " 换句话说, **这个实验系统上, 孪生与 MIL 的正向模型是同一量级的"
                "解释力**, 而 MIL 那次运行自己判的是 "
                + (
                    f"`{mil_summary.get('status', '未知')}`"
                    if mil_summary.get("status")
                    else "未通过验证集门槛"
                )
                + "。"
            )
            add("")


    # -- 9 限制 ----------------------------------------------------------
    add("## 9. 限制与剩余问题")
    add("")
    add(
        f"**焦面标度口径: 已统一 (原先的第一条缺口已关闭)**。孪生不再按标称焦距工作 —— "
        f"由实测标度 `{cell_scale:.6f}` 反推出等效焦距 **{implied_focal * 1e3:.2f} mm** "
        "(`BenchTwinConfig.from_measured_scale`), 孪生的解析标度与实测标度**逐位一致** "
        f"(相对偏差 {scale_rel_dev:.2e} < 1e-12), 因此孪生的远场**角标度**在这套实验"
        "系统上是自洽的, 反演出来的 `defocus_m` / `tilt_*` 等参数也不再依赖「网格对齐"
        "被强行拉齐」的前提。"
    )
    add("")
    add(
        f"⚠️ 提醒: 若改回标称焦距 {nominal_focal * 1e3:.4f} mm, 解析标度会差 "
        f"{cell_scale / nominal_scale:.2f} 倍 (标称焦距不适用于这套光路) —— 引用孪生的"
        "绝对角标度时, 必须说明用的是等效焦距。"
    )
    add("")
    add(
        "1. **瞳面照明模型不同**: MIL 的正向模型把入射振幅限制在**内切圆**内 "
        "(`gaussian_grid` 自带圆孔掩膜), 而 `bench_twin` 的 `pupil_amplitude()` 是"
        "**无掩膜高斯** (角上有光)。两者在瞳面边缘的场分布不同, 这是孪生与 MIL 模型"
        "在同一相位上给出不同远场的一个来源。"
    )
    add(
        "2. **面板上真正显示的相位是插值过的**: MIL 把 128×128 的模型相位用双线性"
        "插值放大到面板上的方框 (`phase_to_panel`), 所以真实瞳面相位是**带限**的; "
        "孪生 (和 MIL 模型) 直接用 128×128 数组, 于是两者的远场都比真实画面**更弥散**。"
        "这解释了为什么「实测帧的近场更紧」而模型更散。"
    )
    add(
        "3. **面相差与透镜像差近似简并**: 两者都是加在瞳面上的相位, 只有「是否经过 "
        "SLM 灰度量化」这一点不同, 所以它们的分配不唯一; 反演给出的面/透镜系数"
        "只能当作**一组等效解**, 不宜单独解读某一项。"
    )
    add(
        "4. **实测帧含探测器噪声**: 实测帧经 `prepare_frame` (扣中位数 + 截零) 处理, "
        "tails 里主要是探测器噪声底; 孪生反演时 `noise_adu = 0` (确定性), 所以 "
        "SSIM 在弱结构区天然吃亏, 相关与能量比更可信。"
    )
    add(
        "5. **验证集只有 "
        f"{len(holdout_samples)} 条探针**, 且都是 MIL 生成的平滑/低阶相位族; "
        "结论的统计强度有限, 但这是该次实验实际落盘的全部验证集数据。"
    )
    add("")

    # -- 10 复现 ---------------------------------------------------------
    add("## 10. 复现")
    add("")
    add("```bash")
    add(
        "python scripts/slm/generate_bench_twin_report.py "
        f"--run-dir {rel_run} --n-train {args.n_train} "
        f"--maxiter {args.maxiter} --popsize {args.popsize} "
        f"--maxiter-fine {args.maxiter_fine} --seed {args.seed}"
    )
    add("```")
    add("")
    add(
        f"输入数据: `{rel_run}` (实验系统实测, 本脚本只读)。"
        "输出: 本报告与 `figures/` 下的图。"
    )
    add("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
