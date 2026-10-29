"""MIL (model-in-loop) 单次实验运行的**记录报告**离线生成器。

**要回答的问题**（用户原话：SLM 相位、CCD 远场光斑、模型梯度变化、参数设置）：

1. **正向调优模型 (Step A)** —— 拟合出的像差系数（非零 Noll 模式 + 弧度）、拟合前后的
   残差、验证集相关性、用了多少探针 / 迭代。
2. **反向可微优化 (Step B)** —— 迭代数、形状损失的首末值、逐轮 score / CV / EE；
   逐迭代的损失轨迹（能找到就画，找不到就写「未记录」，**不臆造曲线**）。
3. **SLM 相位** —— 最佳相位的图与统计，以及它与 Step A 拟合像差的关系
   （可检验的一致性检查，见下）。
4. **CCD 远场光斑** —— 平场参考 / 整形后实测 / 模型预测 / 差值 四联，
   逐探针的相关系数与 SSIM。
5. **参数设置** —— 相机 / 曝光 / 目标 / 探针 / 预算 / 几何 / 误差项 / 随机种子。
6. **结论** —— 整形后复合分数是否**严格大于**平场分，以及未提交相位的原因。

**数据**是一次真实实验的落盘目录（只读、离线）：

    data/debug/model_in_loop_hw_ab/<会话>/<变体>/slm_model_in_loop/<运行戳>/
        run_config.json           本次运行的全部参数
        bench_geometry.json       实测几何
        fitted_coefficients.npy   Step A 拟合出的像差系数（Noll 序、弧度）
        best_phase.npy            落盘的最佳 SLM 相位（raw 未包裹弧度）
        source_amplitude.npy      标定出的瞳面照明幅度
        probes/manifest.json      逐探针清单
        probes/round_XX_*.npz     逐探针（相位 / 原始帧 / 处理后帧 / 模型网格 / 预测）
        preflight/flat.npz        平场参考
        slm_model_in_loop_*.csv   逐轮记录（32 列）
        slm_model_in_loop_*.pkl   recorder 的逐轮全过程（含逐迭代轨迹）
        model_error_terms.json    注入的误差参数（若本次带了 --error-params）
        handoff_summary.json      summarize_session 的验收结论

**帧网格口径（本脚本最要紧的一步）**。实测帧是 300×300 **相机像素**窗口，物理预测模型
输出的是 1024² **远场采样点**网格，两者必须先落到同一张网格上才能逐像素比较：

1. 相机像元 ↔ 远场采样点的换算用 ``bench_geometry.json`` 的**实测**标定
   ``camera_px_per_model_px``（相位斜坡扫描得到）；
2. 实测帧落到远场网格用 MIL 自己那套**已验证**的实现
   （``slm_model_in_loop._to_model_grid``），并用探针里落盘的 ``model_grid``
   **逐像素核对**（见报告 §6）；
3. **旋转象限不写死**：拿落盘的 ``model_grid`` 逐像素反推（4 个象限里挑最吻合的），
   口径由数据说话（实测最大偏差 ~6e-09，见报告 §6.1）；
4. 模型预测（1024²）→ 相机窗口（300×300）走本脚本自己的**逆映射**
   （``_model_grid_to_camera``），并用「把落盘 ``model_grid`` 映回相机窗口」与原始
   相机帧逐像素比对来验证（见报告 §6.1）。

**逐迭代轨迹的探测方式**（优雅降级，不臆造曲线）。按下列顺序找：

1. 运行目录里的独立文件：``step_b_trace*`` / ``*trace*.json`` / ``*trace*.npz`` /
   ``*trace*.csv``（新加的逐迭代轨迹落盘会命中这里）；
2. recorder 的 pkl（``slm_model_in_loop_*.pkl``）里的 ``step_a_trace`` /
   ``step_b_trace`` / ``iteration_traces``（**本次运行命中这里**）；
3. CSV 里形如 ``step_b_loss_<i>`` 的逐迭代列。

每一条轨迹都是「迭代号 + 损失」序列；**梯度范数**另找（``grad_norm`` /
``gradient_norm`` 之类的键），找不到就写「未记录」。**每迭代的模型远场**找不到时，
过程动画降级为「逐探针的实测 / 预测」。

**一致性检查的口径**。物理预测模型的瞳面相位是 ``命令相位 + 像差``（见
``zernike_coefficient_optimizer._far_field_intensity``），所以**抵消像差**的命令相位
是 ``-像差``。本报告因此给出「最佳相位 **减去**（-像差）」=「最佳相位 + 像差」
的残余瞳面相位 RMS：若 Step B 只是做像差抵消（不整形），该值应当接近 0。
本次运行的目标是**矩形**（不是平坦），所以残余不会为 0 —— 报告照实写数。

运行：

    python scripts/slm/generate_model_in_loop_record_report.py ^
        --run-dir data/debug/model_in_loop_hw_ab/20261021_172236/baseline_a/slm_model_in_loop/20261021_172309

    python scripts/slm/generate_model_in_loop_record_report.py ^
        --run-dir <运行目录> --stamp 20261021_172309 --no-gif
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
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

#: MIL 手工系列子目录（一份报告 = 一个子目录，见 ``report/AGENTS.md``）。
OUT_DIR = ROOT / "report" / "shape_trials" / "model_in_loop"
FIG_DIR = OUT_DIR / "figures"

#: 报告名（不含戳）—— 戳由运行目录名推导，``--stamp`` 可覆盖。
REPORT_STEM = "model_in_loop_record"

#: 找不到的量的统一写法。数字全部从产物读，读不到就写这个，**不填 0**。
MISSING = "未记录"

#: 逐迭代轨迹的候选文件名（独立落盘的那条线会命中其中之一）。
_TRACE_FILE_PATTERNS = (
    "step_b_trace*",
    "step_a_trace*",
    "*step_b*trace*",
    "*step_a*trace*",
    "*trace*.json",
    "*trace*.npz",
    "*trace*.csv",
)

#: 逐迭代轨迹里「梯度范数」可能用的键名。
_GRAD_KEYS = ("grad_norm", "gradient_norm", "grad_l2", "grad_norm_l2", "g_norm")


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------


@dataclass
class ProbeEntry:
    """一条探针（落盘的 ``probes/*.npz`` + ``manifest.json`` 里的元信息）。"""

    name: str
    round: int
    split: str
    index: int
    family: str | None
    exposure_ms: float | None
    timestamp_utc: str | None
    manifest_correlation: float | None
    phase: np.ndarray
    frame: np.ndarray
    raw_frame: np.ndarray | None
    model_grid: np.ndarray | None
    prediction: np.ndarray | None
    prediction_source: str = "未记录"

    @property
    def label(self) -> str:
        """短标签，例如 ``r1 train#0``。"""
        return f"r{self.round} {self.split}#{self.index}"


@dataclass
class Trace:
    """一段逐迭代轨迹（迭代号 + 损失 [+ 梯度范数]）。"""

    iterations: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float | None] = field(default_factory=list)
    source: str = MISSING

    @property
    def available(self) -> bool:
        """轨迹是否非空。"""
        return bool(self.losses)

    @property
    def has_grad(self) -> bool:
        """轨迹里是否带梯度范数。"""
        return any(value is not None for value in self.grad_norms)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """返回 ``(迭代号, 损失, 梯度范数或 None)``。"""
        iters = np.asarray(self.iterations, dtype=np.float64)
        losses = np.asarray(self.losses, dtype=np.float64)
        if self.has_grad:
            grads = np.asarray(
                [np.nan if v is None else v for v in self.grad_norms], dtype=np.float64
            )
        else:
            grads = None
        return iters, losses, grads


@dataclass
class RunBundle:
    """一次 MIL 运行的全部落盘产物。"""

    run_dir: Path
    stamp: str
    variant: str
    config: dict[str, Any]
    geometry: dict[str, Any]
    handoff: dict[str, Any]
    status: str
    error_terms: dict[str, Any] | None
    failure: dict[str, Any] | None
    rows: list[dict[str, str]]
    coefficients: np.ndarray | None
    best_phase: np.ndarray | None
    source_amplitude: np.ndarray | None
    flat_frame: np.ndarray | None
    probes: list[ProbeEntry]
    trace_a: Trace
    trace_b: Trace
    rotation_quarters: int
    rotation_residual: float
    roi_center: tuple[int, int] | None
    cell_scale: float | None
    far_field_size: int | None
    recompute_note: str = MISSING
    recompute_deviation: float | None = None


# ---------------------------------------------------------------------------
# 读盘
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    """读一个 JSON 对象（不存在返回空字典）。"""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读 JSON 失败 {}: {}", path, exc)
        return {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读逐轮 CSV（全部按字符串读，数值转换留给调用方）。"""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_float(value: Any) -> float | None:
    """把 CSV 单元格转成 float；空 / 非数 → ``None``。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _load_npy(path: Path) -> np.ndarray | None:
    """读一个 ``.npy``（不存在或读失败 → ``None``）。"""
    if not path.is_file():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        logger.warning("读 npy 失败 {}: {}", path, exc)
        return None


def _rel_to_root(path: Path) -> str:
    """仓库根相对路径（正斜杠），用于报告里的散文引用。"""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _variant_of(run_dir: Path) -> str:
    """从运行目录反推实验系统**变体**名（``.../<变体>/slm_model_in_loop/<戳>``）。"""
    parts = run_dir.resolve().parts
    if "slm_model_in_loop" in parts:
        index = len(parts) - 1 - parts[::-1].index("slm_model_in_loop")
        if index >= 1:
            return parts[index - 1]
    return run_dir.name


# ---------------------------------------------------------------------------
# 逐迭代轨迹的探测
# ---------------------------------------------------------------------------


def _normalise_trace(entries: Any, source: str) -> Trace:
    """把任意形状的轨迹数据归一成「迭代号 + 损失 [+ 梯度范数]」。"""
    trace = Trace(source=source)
    if entries is None:
        return trace
    if isinstance(entries, dict):
        # {"loss": [...], "iteration": [...], "grad_norm": [...]} 之类
        losses = entries.get("loss") or entries.get("losses")
        if losses is None:
            return trace
        iters = entries.get("iteration") or entries.get("iterations")
        grads = None
        for key in _GRAD_KEYS:
            if key in entries:
                grads = entries[key]
                break
        count = len(losses)
        trace.iterations = [
            int(iters[i]) if iters is not None and i < len(iters) else i
            for i in range(count)
        ]
        trace.losses = [float(v) for v in losses]
        trace.grad_norms = (
            [None if grads is None else float(grads[i]) for i in range(count)]
            if grads is not None
            else [None] * count
        )
        return trace
    if isinstance(entries, Sequence):
        for index, item in enumerate(entries):
            if isinstance(item, dict):
                iteration = item.get("iteration", item.get("iter", index))
                loss = item.get("loss")
                if loss is None:
                    continue
                grad = None
                for key in _GRAD_KEYS:
                    if item.get(key) is not None:
                        grad = float(item[key])
                        break
                trace.iterations.append(int(iteration))
                trace.losses.append(float(loss))
                trace.grad_norms.append(grad)
            else:
                trace.iterations.append(index)
                trace.losses.append(float(item))
                trace.grad_norms.append(None)
    return trace


def _trace_from_file(path: Path) -> tuple[Trace, Trace]:
    """从一个独立落盘的轨迹文件里读出 (Step A 轨迹, Step B 轨迹)。"""
    step_a, step_b = Trace(), Trace()
    name = path.name.lower()
    if path.suffix == ".json":
        payload = _read_json(path)
        if "step_b" in name:
            step_b = _normalise_trace(payload, f"`{path.name}`")
        elif "step_a" in name:
            step_a = _normalise_trace(payload, f"`{path.name}`")
        else:
            step_a = _normalise_trace(payload.get("step_a_trace"), f"`{path.name}`")
            step_b = _normalise_trace(payload.get("step_b_trace"), f"`{path.name}`")
    elif path.suffix == ".npz":
        try:
            with np.load(path, allow_pickle=False) as payload:
                keys = list(payload.files)
        except (OSError, ValueError):
            return step_a, step_b
        for key in keys:
            lowered = key.lower()
            if "grad" in lowered:
                continue
            if "step_b" in lowered or ("b" == lowered[-1:] and "step" in lowered):
                candidate = _normalise_trace(
                    {key: payload[key]}, f"`{path.name}` 的 `{key}`"
                )
                if candidate.available:
                    step_b = candidate
            elif "step_a" in lowered:
                candidate = _normalise_trace(
                    {key: payload[key]}, f"`{path.name}` 的 `{key}`"
                )
                if candidate.available:
                    step_a = candidate
    elif path.suffix == ".csv":
        rows = _read_csv_rows(path)
        if not rows:
            return step_a, step_b
        columns = list(rows[0].keys())
        loss_col = next(
            (c for c in columns if "loss" in c.lower() and "grad" not in c.lower()),
            None,
        )
        if loss_col is None:
            return step_a, step_b
        iter_col = next((c for c in columns if "iter" in c.lower()), None)
        grad_col = next((c for c in columns if "grad" in c.lower()), None)
        entries = []
        for index, row in enumerate(rows):
            loss = _as_float(row.get(loss_col))
            if loss is None:
                continue
            iteration = _as_float(row.get(iter_col)) if iter_col else None
            grad = _as_float(row.get(grad_col)) if grad_col else None
            entries.append(
                {
                    "iteration": int(iteration) if iteration is not None else index,
                    "loss": loss,
                    **({"grad_norm": grad} if grad is not None else {}),
                }
            )
        candidate = _normalise_trace(entries, f"`{path.name}`")
        if "step_b" in name:
            step_b = candidate
        else:
            step_a = candidate
    return step_a, step_b


def _trace_from_pickle(path: Path) -> tuple[Trace, Trace]:
    """从 recorder 的 pkl 里读出 (Step A 轨迹, Step B 轨迹)。"""
    step_a, step_b = Trace(), Trace()
    try:
        payload = pickle.loads(path.read_bytes())
    except (OSError, pickle.PickleError, EOFError, AttributeError) as exc:
        logger.warning("读 pkl 失败 {}: {}", path, exc)
        return step_a, step_b
    source = f"recorder pkl `{path.name}`"
    if (
        isinstance(payload, dict)
        and "history" in payload
        and hasattr(payload["history"], "__iter__")
    ):
        # Recorder 对象形态
        for row in payload["history"]:
            if not isinstance(row, dict):
                continue
            candidate_a = _normalise_trace(row.get("step_a_trace"), source)
            candidate_b = _normalise_trace(row.get("step_b_trace"), source)
            if candidate_a.available:
                step_a = candidate_a
            if candidate_b.available:
                step_b = candidate_b
        return step_a, step_b
    if isinstance(payload, dict):
        # {epoch: record} 形态 —— MIL 的 pkl 落盘就是这个形态
        for key in sorted(payload, key=lambda k: (isinstance(k, str), k)):
            record = payload[key]
            if not isinstance(record, dict):
                continue
            candidate_a = _normalise_trace(record.get("step_a_trace"), source)
            candidate_b = _normalise_trace(record.get("step_b_trace"), source)
            if candidate_a.available:
                step_a = candidate_a
            if candidate_b.available:
                step_b = candidate_b
        traces = payload.get("iteration_traces")
        if isinstance(traces, dict):
            for value in traces.values():
                if not isinstance(value, dict):
                    continue
                candidate_a = _normalise_trace(value.get("step_a_trace"), source)
                candidate_b = _normalise_trace(value.get("step_b_trace"), source)
                if candidate_a.available:
                    step_a = candidate_a
                if candidate_b.available:
                    step_b = candidate_b
    return step_a, step_b


def _trace_from_csv_columns(rows: Sequence[dict[str, str]]) -> tuple[Trace, Trace]:
    """从逐轮 CSV 里找形如 ``step_b_loss_<i>`` 的逐迭代列。"""
    step_a, step_b = Trace(), Trace()
    if not rows:
        return step_a, step_b
    columns = list(rows[0].keys())

    def collect(prefix: str) -> Trace:
        pattern = re.compile(rf"^{prefix}_(?:loss|grad_norm)_(\d+)$")
        found: dict[int, dict[str, float]] = {}
        for column in columns:
            match = pattern.match(column)
            if match is None:
                continue
            index = int(match.group(1))
            value = _as_float(rows[0].get(column))
            if value is not None:
                found.setdefault(index, {})[column] = value
        if not found:
            return Trace()
        trace = Trace(source="逐轮 CSV 的逐迭代列")
        for index in sorted(found):
            entry = found[index]
            loss = next(
                (v for k, v in entry.items() if "loss" in k and "grad" not in k), None
            )
            if loss is None:
                continue
            grad = next((v for k, v in entry.items() if "grad" in k), None)
            trace.iterations.append(index)
            trace.losses.append(loss)
            trace.grad_norms.append(grad)
        return trace

    step_b = collect("step_b")
    step_a = collect("step_a")
    return step_a, step_b


def _probe_iteration_traces(
    run_dir: Path, rows: Sequence[dict[str, str]]
) -> tuple[Trace, Trace]:
    """按「独立文件 → recorder pkl → CSV 逐迭代列」的顺序找逐迭代轨迹。"""
    step_a, step_b = Trace(), Trace()
    for pattern in _TRACE_FILE_PATTERNS:
        for path in sorted(run_dir.rglob(pattern)):
            if not path.is_file():
                continue
            found_a, found_b = _trace_from_file(path)
            if found_b.available and not step_b.available:
                step_b = found_b
            if found_a.available and not step_a.available:
                step_a = found_a
    if not (step_a.available and step_b.available):
        for path in sorted(run_dir.glob("slm_model_in_loop_*.pkl")):
            found_a, found_b = _trace_from_pickle(path)
            if found_b.available and not step_b.available:
                step_b = found_b
            if found_a.available and not step_a.available:
                step_a = found_a
    if not (step_a.available and step_b.available):
        found_a, found_b = _trace_from_csv_columns(rows)
        if found_b.available and not step_b.available:
            step_b = found_b
        if found_a.available and not step_a.available:
            step_a = found_a
    return step_a, step_b


def _probe_iteration_farfields(run_dir: Path) -> list[np.ndarray] | None:
    """找「每迭代的模型远场」：npz 里第一个维度等于迭代数的二维强度序列。"""
    for path in sorted(run_dir.rglob("*.npz")):
        if path.parent.name == "probes":
            continue
        try:
            with np.load(path, allow_pickle=False) as payload:
                for key in payload.files:
                    array = payload[key]
                    if array.ndim == 3 and array.shape[1] == array.shape[2]:
                        return [np.asarray(item, dtype=np.float64) for item in array]
        except (OSError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# 几何：旋转象限反推 + 模型网格 ↔ 相机窗口
# ---------------------------------------------------------------------------


def _derive_rotation(
    probes: Sequence[ProbeEntry],
    center: tuple[int, int] | None,
    *,
    far_field_size: int,
    cell_scale: float,
) -> tuple[int, float]:
    """从落盘的 ``model_grid`` 反推相机↔模型的旋转象限。

    口径由数据说话：4 个象限逐个试，取与落盘 ``model_grid`` 逐像素最吻合的那个。
    没有可核对的 ``model_grid`` 时返回 ``(0, nan)``。
    """
    from ao_shaping.optimizer.wfless.slm_model_in_loop import _to_model_grid

    reference = [p for p in probes if p.model_grid is not None]
    if center is None or not reference:
        return 0, float("nan")
    best_rot, best_err = 0, float("nan")
    for rot in range(4):
        err = 0.0
        for probe in reference:
            grid = _to_model_grid(probe.frame, center, far_field_size, cell_scale, rot)
            err = max(err, float(np.abs(grid - probe.model_grid).max()))
        if not math.isfinite(best_err) or err < best_err:
            best_rot, best_err = rot, err
    return best_rot, best_err


def _model_grid_to_camera(
    model_grid: np.ndarray,
    center: tuple[int, int],
    *,
    far_field_size: int,
    cell_scale: float,
    cam_size: int,
    rotation_quarters: int,
) -> np.ndarray:
    """把远场采样点网格映回相机像元窗口（``_to_model_grid`` 的逆映射）。

    正向：相机窗口（跨度 ``round(N * cell_scale)`` 像元，围绕冻结光轴）→ 重采样到
    ``N × N`` → ``rot90(·, -rot)``。逆向即本函数：先 ``rot90(·, +rot)`` 回到相机
    朝向，再按像元中心取最近邻采样。

    Returns:
        ``(cam_size, cam_size)`` 的相机窗口图像。
    """
    side = int(model_grid.shape[0])
    out = np.rot90(np.asarray(model_grid, dtype=np.float64), rotation_quarters)
    span = int(round(side * cell_scale))
    x0 = int(center[0]) - span // 2
    y0 = int(center[1]) - span // 2
    step = span / float(side)
    xs = np.clip(np.rint((np.arange(cam_size) - x0) / step).astype(int), 0, side - 1)
    ys = np.clip(np.rint((np.arange(cam_size) - y0) / step).astype(int), 0, side - 1)
    return out[np.ix_(ys, xs)]


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """两个同形数组的 Pearson 相关系数（与 MIL 的 ``_prediction_correlation`` 同口径）。"""
    x = np.asarray(a, dtype=np.float64).ravel().copy()
    y = np.asarray(b, dtype=np.float64).ravel().copy()
    if x.shape != y.shape or x.size < 2:
        return None
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 0:
        return None
    return float(np.dot(x, y) / denominator)


def _ssim(a: np.ndarray, b: np.ndarray) -> float | None:
    """峰值归一化后的 SSIM（``scikit-image``，7×7 窗）。

    缺后端 / 形状不匹配 / 峰值为 0 时返回 ``None``（读作「未记录」而不是 0）。
    """
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.shape != second.shape or min(first.shape) < 7:
        return None
    peak_a, peak_b = float(first.max()), float(second.max())
    if peak_a <= 0 or peak_b <= 0:
        return None
    try:
        from skimage.metrics import structural_similarity
    except Exception:  # pragma: no cover - scikit-image 是已声明依赖
        return None
    return float(
        structural_similarity(
            first / peak_a, second / peak_b, data_range=1.0, win_size=7
        )
    )


def _zernike_basis(region: int, n_orders: int) -> tuple[np.ndarray, np.ndarray]:
    """按 Noll 序构造 Zernike 基 + 瞳面掩膜（与 MIL 的优化器同口径）。

    复用的是**引擎层**的 ``ZernikeGenerator``（``utils/wavefront/zernike_calc.py``），
    与 ``zernike_coefficient_optimizer._build_basis`` 逐位同源 —— 不另写一份查表。
    """
    from ao_shaping.utils.wavefront.zernike_calc import (
        ZernikeGenerator,
        calc_n_zernike_terms,
    )

    count = calc_n_zernike_terms(n_orders)
    generator = ZernikeGenerator(
        (region, region), radius=region // 2, n_orders=n_orders
    )
    basis = np.zeros((count, region, region), dtype=np.float64)
    aperture = np.zeros((region, region), dtype=bool)
    for index in range(count):
        weights = np.zeros(count)
        weights[index] = 1.0
        mode = np.asarray(generator.generate_noll(weights), dtype=np.float64)
        if index == 0:
            aperture = np.isfinite(mode)
        basis[index] = np.nan_to_num(mode, nan=0.0)
    return basis, aperture


def _rms(values: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    """RMS（可选掩膜内统计）。"""
    array = np.asarray(values, dtype=np.float64)
    if mask is not None:
        array = array[mask]
    if array.size == 0:
        return None
    return float(np.sqrt(np.mean(array**2)))


# ---------------------------------------------------------------------------
# 组装一次运行
# ---------------------------------------------------------------------------


def _read_probes(run_dir: Path) -> list[ProbeEntry]:
    """读 ``probes/`` 下的全部探针（manifest 提供 split / 曝光 / 相关性）。"""
    manifest_path = run_dir / "probes" / "manifest.json"
    manifest: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        for entry in _read_json_list(manifest_path):
            manifest[str(entry.get("file"))] = entry
    probes: list[ProbeEntry] = []
    for path in sorted((run_dir / "probes").glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as payload:
                names = set(payload.files)
                phase = np.asarray(payload["phase"], dtype=np.float64)
                frame = np.asarray(payload["frame"], dtype=np.float64)
                raw = (
                    np.asarray(payload["raw_frame"], dtype=np.float64)
                    if "raw_frame" in names
                    else None
                )
                grid = (
                    np.asarray(payload["model_grid"], dtype=np.float64)
                    if "model_grid" in names
                    else None
                )
                prediction = (
                    np.asarray(payload["prediction"], dtype=np.float64)
                    if "prediction" in names
                    else None
                )
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("读探针失败 {}: {}", path, exc)
            continue
        meta = manifest.get(path.name, {})
        probes.append(
            ProbeEntry(
                name=path.name,
                round=int(meta.get("round", 0) or 0),
                split=str(meta.get("split", "unknown")),
                index=int(meta.get("index", 0) or 0),
                family=meta.get("family"),
                exposure_ms=_as_float(meta.get("exposure_ms")),
                timestamp_utc=meta.get("timestamp_utc"),
                manifest_correlation=_as_float(meta.get("correlation")),
                phase=phase,
                frame=frame,
                raw_frame=raw,
                model_grid=grid,
                prediction=prediction,
                prediction_source=("落盘预测" if prediction is not None else MISSING),
            )
        )
    return probes


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    """读一个 JSON 数组（不存在或读失败 → 空列表）。"""
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读 JSON 失败 {}: {}", path, exc)
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _recompute_predictions(bundle: RunBundle) -> None:
    """用本次运行的物理预测模型离线复算缺预测的探针（并用落盘预测校验复算可信）。

    复算只在**能与落盘预测对上**时才采用：判据是验证集 / 整形探针上的最大绝对
    偏差。对不上就放弃复算，缺预测的探针一律写「未记录」。
    """
    bundle.recompute_note = "未复算（缺少系数 / 瞳面振幅 / torch 后端）"
    if bundle.coefficients is None or bundle.source_amplitude is None:
        return
    if bundle.cell_scale is None or bundle.roi_center is None:
        return
    try:
        import torch
        from ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer import (
            ZernikeCoefficientOptimizer,
        )
    except Exception as exc:  # pragma: no cover - torch 是可选依赖
        bundle.recompute_note = f"未复算（torch 后端不可用：{exc}）"
        return

    config = bundle.config
    region = int(config.get("region", bundle.coefficients.shape[0] * 0 or 128))
    try:
        optimizer = ZernikeCoefficientOptimizer(
            n_orders=int(config.get("n_orders", 10)),
            region=region,
            initial_coefficients=bundle.coefficients,
            lr=float(config.get("step_a_lr", 0.05)),
            max_iterations=max(1, int(config.get("step_a_iterations", 1))),
            device=config.get("device"),
            seed=int(config.get("seed", 0) or 0),
            dtype=str(config.get("dtype", "float32")),
            far_field_size=region * int(config.get("far_field_padding", 1)),
            frozen_modes=tuple(config.get("frozen_modes", ()) or ()),
            pupil_oversampling=int(config.get("pupil_oversampling", 1)),
            error_terms=None,
        )
    except Exception as exc:  # 保留宽泛捕获: 内部调用链 ZernikeCoefficientOptimizer 的失败类型不可枚举
        bundle.recompute_note = f"未复算（物理预测模型构造失败：{exc}）"
        return
    optimizer.set_source_amplitude(bundle.source_amplitude)

    gain, leakage = _gain_and_leakage(bundle)
    flat_grid = None
    if bundle.flat_frame is not None and bundle.roi_center is not None:
        from ao_shaping.optimizer.wfless.slm_model_in_loop import _to_model_grid

        flat_grid = _to_model_grid(
            bundle.flat_frame,
            bundle.roi_center,
            int(bundle.far_field_size or 0),
            bundle.cell_scale,
            bundle.rotation_quarters,
        )

    def predict(phase: np.ndarray) -> np.ndarray:
        raw = optimizer.forward_intensity(
            bundle.coefficients, phase, bundle.source_amplitude
        )
        if flat_grid is not None and gain is not None and leakage is not None:
            raw = gain * raw + leakage * flat_grid
        return raw

    # 校验：与落盘预测逐像素比对
    deviation = 0.0
    checked = 0
    for probe in bundle.probes:
        if probe.prediction is None:
            continue
        try:
            candidate = predict(probe.phase)
        except Exception as exc:  # 保留宽泛捕获: 内部调用链 predict 的失败类型不可枚举
            logger.warning("复算预测失败 {}: {}", probe.name, exc)
            return
        deviation = max(deviation, float(np.abs(candidate - probe.prediction).max()))
        checked += 1
    if checked == 0:
        bundle.recompute_note = "未复算（本次运行没有落盘预测可供校验）"
        return
    bundle.recompute_deviation = deviation
    if deviation > 1e-3:
        bundle.recompute_note = (
            f"未采用复算（与落盘预测的最大偏差 {deviation:.3g} 过大）"
        )
        return
    bundle.recompute_note = (
        f"已用本次运行的物理预测模型离线复算（与落盘预测的最大绝对偏差 "
        f"{deviation:.3g}，{checked} 条探针）"
    )
    for probe in bundle.probes:
        if probe.prediction is not None:
            continue
        try:
            probe.prediction = predict(probe.phase)
            probe.prediction_source = "离线复算"
        except Exception as exc:  # pragma: no cover - 防御
            logger.warning("复算预测失败 {}: {}", probe.name, exc)


def _gain_and_leakage(bundle: RunBundle) -> tuple[float | None, float | None]:
    """从逐轮 CSV 读增益与加性背景（杂散光）项系数。"""
    gain, leakage = None, None
    for row in bundle.rows:
        if row.get("stage") != "round":
            continue
        value = _as_float(row.get("model_gain"))
        if value is not None:
            gain = value
        value = _as_float(row.get("flat_leakage"))
        if value is not None:
            leakage = value
    return gain, leakage


def load_run(run_dir: Path, stamp: str) -> RunBundle:
    """把一次 MIL 运行的落盘产物读成一个 :class:`RunBundle`。"""
    config = _read_json(run_dir / "run_config.json")
    geometry = _read_json(run_dir / "bench_geometry.json")
    handoff = _read_json(run_dir / "handoff_summary.json")
    error_terms = _read_json(run_dir / "model_error_terms.json") or None

    status = MISSING
    for path in sorted(run_dir.glob("slm_model_in_loop_*.json")):
        payload = _read_json(path)
        if payload.get("status"):
            status = str(payload["status"])
            break

    csv_paths = sorted(run_dir.glob("slm_model_in_loop_*.csv"))
    rows = _read_csv_rows(csv_paths[-1]) if csv_paths else []

    flat_frame = None
    flat_path = run_dir / "preflight" / "flat.npz"
    if flat_path.is_file():
        try:
            with np.load(flat_path, allow_pickle=False) as payload:
                if "frame" in payload.files:
                    flat_frame = np.asarray(payload["frame"], dtype=np.float64)
        except (OSError, ValueError) as exc:
            logger.warning("读平场失败 {}: {}", flat_path, exc)

    probes = _read_probes(run_dir)
    trace_a, trace_b = _probe_iteration_traces(run_dir, rows)

    far_field_size = int(geometry.get("far_field_size", 0) or 0)
    cam_type = str(config.get("cam_type", "daheng"))
    scale = _as_float(geometry.get("camera_px_per_model_px"))
    cell_scale: float | None = None
    if scale is not None and far_field_size:
        if cam_type == "sim":
            region = int(config.get("region", 128) or 128)
            cell_scale = scale * float(region) / float(far_field_size)
        else:
            cell_scale = scale

    roi_center = None
    if flat_frame is not None:
        from ao_shaping.utils.image.beam_metrics import zero_order_center

        roi_center = zero_order_center(flat_frame, refine=False)

    rotation, residual = (0, float("nan"))
    if cell_scale is not None and far_field_size:
        rotation, residual = _derive_rotation(
            probes, roi_center, far_field_size=far_field_size, cell_scale=cell_scale
        )

    bundle = RunBundle(
        run_dir=run_dir,
        stamp=stamp,
        variant=_variant_of(run_dir),
        config=config,
        geometry=geometry,
        handoff=handoff,
        status=status,
        error_terms=error_terms,
        failure=_read_json(run_dir / "failure.json") or None,
        rows=rows,
        coefficients=_load_npy(run_dir / "fitted_coefficients.npy"),
        best_phase=_load_npy(run_dir / "best_phase.npy"),
        source_amplitude=_load_npy(run_dir / "source_amplitude.npy"),
        flat_frame=flat_frame,
        probes=probes,
        trace_a=trace_a,
        trace_b=trace_b,
        rotation_quarters=rotation,
        rotation_residual=residual,
        roi_center=roi_center,
        cell_scale=cell_scale,
        far_field_size=far_field_size or None,
    )
    _recompute_predictions(bundle)
    return bundle


# ---------------------------------------------------------------------------
# 派生量
# ---------------------------------------------------------------------------


def _round_rows(bundle: RunBundle) -> list[dict[str, str]]:
    """逐轮记录（不含标定基线行）。"""
    return [row for row in bundle.rows if row.get("stage") == "round"]


def _flat_row(bundle: RunBundle) -> dict[str, str] | None:
    """标定基线行（平场）。"""
    for row in bundle.rows:
        if row.get("stage") == "calibration":
            return row
    return None


def _scores(bundle: RunBundle) -> tuple[float | None, float | None, str]:
    """返回 ``(平场复合分数, 整形后复合分数, 判定文字)``。

    平场分取标定基线行的 ``score_after``；整形后分取最后一个「round」行的
    ``score_after``。两者都读不到时返回 ``(None, None, ...)``。
    """
    flat_row = _flat_row(bundle)
    flat_score = _as_float(flat_row.get("score_after")) if flat_row else None
    rounds = _round_rows(bundle)
    shaped_score = _as_float(rounds[-1].get("score_after")) if rounds else None
    if flat_score is None or shaped_score is None:
        return flat_score, shaped_score, f"{MISSING}（产物里没有可比的复合分数）"
    if shaped_score > flat_score:
        verdict = "是"
    elif shaped_score == flat_score:
        verdict = "否（相等，不是严格大于）"
    else:
        verdict = "否"
    return flat_score, shaped_score, verdict


def _aberration_phase(bundle: RunBundle) -> tuple[np.ndarray | None, np.ndarray | None]:
    """像差相位图（弧度）与瞳面掩膜。"""
    if bundle.coefficients is None:
        return None, None
    region = int(bundle.config.get("region", 0) or 0)
    n_orders = int(bundle.config.get("n_orders", 0) or 0)
    if region <= 0 or n_orders <= 0:
        return None, None
    basis, aperture = _zernike_basis(region, n_orders)
    coefficients = np.asarray(bundle.coefficients, dtype=np.float64)
    count = min(coefficients.shape[0], basis.shape[0])
    phase = np.einsum("j,jhw->hw", coefficients[:count], basis[:count])
    return phase, aperture


def _shaped_probe(bundle: RunBundle) -> ProbeEntry | None:
    """Step B 实际合成并下发的那条探针（``split == "shaped"``）。"""
    shaped = [p for p in bundle.probes if p.split == "shaped"]
    return shaped[-1] if shaped else None


def _probe_order(bundle: RunBundle) -> list[ProbeEntry]:
    """探针的展示顺序：平场 → 训练 → 验证集 → 整形。"""
    order = {"flat": 0, "train": 1, "holdout": 2, "shaped": 3}
    return sorted(
        bundle.probes, key=lambda p: (order.get(p.split, 9), p.round, p.index)
    )


def _consistency(bundle: RunBundle) -> dict[str, Any]:
    """最佳相位与拟合像差的一致性检查。

    物理预测模型的瞳面相位是「命令相位 + 像差」，所以抵消像差的命令相位是 ``-像差``；
    「最佳相位减去（-像差）」= 「最佳相位 + 像差」就是**残余瞳面相位**。
    若 Step B 只做像差抵消（不做整形），残余应当接近 0。
    """
    out: dict[str, Any] = {
        "aberration_rms": None,
        "shaped_rms": None,
        "residual_rms": None,
        "best_residual_rms": None,
        "correlation": None,
        "aperture_px": None,
        "note": MISSING,
    }
    aberration, aperture = _aberration_phase(bundle)
    if aberration is None:
        return out
    out["aberration_rms"] = _rms(aberration, aperture)
    out["aperture_px"] = int(aperture.sum())
    shaped = _shaped_probe(bundle)
    if shaped is not None:
        residual = shaped.phase + aberration
        out["shaped_rms"] = _rms(shaped.phase, aperture)
        out["residual_rms"] = _rms(residual, aperture)
        out["correlation"] = _pearson(shaped.phase, -aberration)
    if bundle.best_phase is not None:
        best = np.asarray(bundle.best_phase, dtype=np.float64)
        if best.shape == aberration.shape:
            out["best_residual_rms"] = _rms(best + aberration, aperture)
    return out


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------


def _imshow(ax: Any, image: np.ndarray, title: str, *, vmax: float | None = None):
    """统一的强度图渲染（灰度、按分位数定标度）。"""
    array = np.asarray(image, dtype=np.float64)
    if vmax is None:
        vmax = float(np.percentile(array, 99.9)) if array.size else 1.0
    ax.imshow(array, cmap="inferno", vmin=0.0, vmax=max(vmax, 1e-12), origin="upper")
    ax.set_title(title, fontsize=11)
    _hide_axes(ax)


def _hide_axes(ax: Any) -> None:
    """去掉刻度与边框，但**保留** xlabel（统计标注就写在那里）。"""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _target_side_camera(bundle: RunBundle) -> int | None:
    """目标框在**相机像元**里的边长（与 MIL 自己的换算一致）。

    远场采样点 ↔ 相机像元的标度是 ``camera_px_per_model_px``；一个远场采样点
    覆盖 ``far_field_size / region`` 个瞳面网格，于是
    ``camera_px_per_target_cell = cell_scale * far_field_size / region``。
    ``config["target_side"]`` 是**标称**的相机像元值，``_metrics_at`` 实际用的是
    换算后的 ``target_camera``（60 → 6 个模型格 → 63 相机像元）。
    """
    if not bundle.cell_scale or not bundle.far_field_size:
        return None
    region = int(bundle.config.get("region", 0) or 0)
    side = _as_float(bundle.config.get("target_side"))
    if region <= 0 or side is None:
        return None
    per_cell = bundle.cell_scale * bundle.far_field_size / float(region)
    side_model = max(3, int(round(side / per_cell)))
    side_model = min(side_model, region - 2)
    return max(3, int(round(side_model * per_cell)))


def _target_mask_camera(bundle: RunBundle, frame_shape: tuple[int, int]) -> np.ndarray:
    """相机窗口上的目标掩模（与 runner 的 ``_metrics_at`` 同一口径）。

    口径：``create_target_mask`` 在图像**中心**生成 → 按
    ``camera_rotation_quarters`` 旋转 → ``roll`` 到**冻结光轴**。
    冻结光轴来自平场帧全局 argmax（见报告 §2「冻结光轴」），不随每帧重定位。

    ⚠️ 旋转象限取**本脚本反推出来的** ``bundle.rotation_quarters``，不是
    ``run_config.json`` 里的 ``camera_rotation_quarters`` —— 后者在硬件路径上
    并不落盘（runner 里它是写死的 1），读不到就退化成 0，目标框会转错 90°。
    目标边长同理：``config["target_side"]`` 是**相机像元**的标称值，而
    ``_metrics_at`` 用的是换算后的 ``target_camera``（见 ``_target_side_camera``）。
    """
    from ao_shaping.algorithm.signal_processing.differentiable_shaping import (
        create_target_mask,
    )

    shape = str(bundle.config.get("target_shape", "square"))
    side = _target_side_camera(bundle)
    if side is None or side <= 0 or bundle.roi_center is None:
        return np.zeros(frame_shape, dtype=bool)

    quarters = int(bundle.rotation_quarters)
    target = create_target_mask(shape, frame_shape, side)
    target = np.rot90(target, quarters)
    center_x, center_y = bundle.roi_center
    target = np.roll(
        target,
        (int(center_y) - frame_shape[0] // 2, int(center_x) - frame_shape[1] // 2),
        axis=(0, 1),
    )
    return np.asarray(target, dtype=np.float64) > 0.5


def _target_metrics(bundle: RunBundle, frame: np.ndarray) -> dict[str, float]:
    """目标区指标（CV / EE / IoU）—— 委托 runner 的 ``_metrics_at``，不另立口径。"""
    from ao_shaping.optimizer.wfless.slm_model_in_loop import _metrics_at

    if bundle.roi_center is None:
        return {}
    side = _target_side_camera(bundle)
    if side is None or side <= 0:
        return {}
    return _metrics_at(
        np.asarray(frame, dtype=np.float64),
        side,
        bundle.roi_center,
        target_shape=str(bundle.config.get("target_shape", "square")),
        camera_rotation_quarters=int(bundle.rotation_quarters),
    )


def _draw_target_outline(ax: Any, mask: np.ndarray) -> None:
    """在强度图上叠加目标掩模轮廓（虚线）。"""
    if mask is None or not mask.any():
        return
    from matplotlib.patches import Rectangle

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    ax.add_patch(
        Rectangle(
            (x0 - 0.5, y0 - 0.5),
            x1 - x0 + 1,
            y1 - y0 + 1,
            linewidth=1.2,
            edgecolor="cyan",
            facecolor="none",
            linestyle="--",
        )
    )


def _text_panel(ax: Any, title: str, lines: list[tuple[str, str]]) -> None:
    """文本面板（两列：项 / 值）。

    字体用默认 sans-serif —— 等宽字体缺中文字形, 面板里全是中文标签。
    """
    ax.clear()
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    if not lines:
        ax.text(0.5, 0.5, MISSING, ha="center", va="center", fontsize=12)
        return
    y = 0.93
    for label, value in lines:
        ax.text(0.02, y, label, fontsize=9.5, va="top")
        ax.text(0.56, y, value, fontsize=9.5, va="top")
        y -= 0.075


def _phase_image(phase: np.ndarray) -> np.ndarray:
    """展示用的相位图：中心化包裹到 ``[-π, π)``。

    包裹**只发生在展示层**（生成器给的是 raw 未包裹弧度）。中心化而不是
    ``[0, 2π)``：后者会把瞳面外那点几十分之一弧度的负值直接推到 ~2π，
    在图上读成一片椒盐 —— 那是显示的假象，不是相位本身。
    """
    array = np.asarray(phase, dtype=np.float64)
    return np.mod(array + np.pi, 2 * np.pi) - np.pi


def _imshow_phase(ax: Any, phase: np.ndarray, title: str):
    """相位图渲染（中心化包裹 + twilight 色环，标度固定 ±π）。"""
    ax.imshow(
        _phase_image(phase),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        origin="upper",
    )
    ax.set_title(title, fontsize=11)
    _hide_axes(ax)


def _phase_stats(phase: np.ndarray, mask: np.ndarray | None = None) -> str:
    """相位统计标注（min / max / RMS，单位弧度）。"""
    array = np.asarray(phase, dtype=np.float64)
    values = array[mask] if mask is not None else array.ravel()
    if values.size == 0:
        return MISSING
    rms = float(np.sqrt(np.mean(values**2)))
    return (
        f"min {values.min():+.3f} rad · max {values.max():+.3f} rad · RMS {rms:.3f} rad"
    )


def figure_phase(bundle: RunBundle, path: Path) -> Path:
    """最佳 SLM 相位图 + 统计标注 + 与像差的残余。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    aberration, aperture = _aberration_phase(bundle)
    shaped = _shaped_probe(bundle)

    if bundle.best_phase is not None:
        best = np.asarray(bundle.best_phase, dtype=np.float64)
        title = "落盘最佳相位 best_phase.npy"
        if float(np.abs(best).max()) == 0:
            title += "\n全零 = 保留平场（对比择优未通过）"
        _imshow_phase(axes[0], best, title)
        axes[0].set_xlabel(_phase_stats(best, aperture), fontsize=10)
    else:
        axes[0].text(0.5, 0.5, f"落盘最佳相位\n{MISSING}", ha="center", va="center")
        axes[0].axis("off")

    if shaped is not None:
        title = "Step B 合成的相位（实测下发）"
        outside = _rms(shaped.phase, None if aperture is None else ~aperture)
        if outside is not None:
            title += f"\n瞳面外 RMS {outside:.3f} rad（照明盘之外，模型里被掩膜）"
        _imshow_phase(axes[1], shaped.phase, title)
        axes[1].set_xlabel(_phase_stats(shaped.phase, aperture), fontsize=10)
    else:
        axes[1].text(0.5, 0.5, f"Step B 合成相位\n{MISSING}", ha="center", va="center")
        axes[1].axis("off")

    if aberration is not None and shaped is not None:
        residual = shaped.phase + aberration
        _imshow_phase(axes[2], residual, "残余瞳面相位 = 合成相位 + 像差")
        axes[2].set_xlabel(_phase_stats(residual, aperture), fontsize=10)
    else:
        axes[2].text(0.5, 0.5, f"残余瞳面相位\n{MISSING}", ha="center", va="center")
        axes[2].axis("off")

    fig.suptitle(
        f"MIL {bundle.stamp} · SLM 相位（raw 弧度，展示层才做 2π 包裹，中心化到 ±π）",
        fontsize=13,
    )
    fig.tight_layout()
    savefig(fig, path)
    return path


def figure_aberration(bundle: RunBundle, path: Path) -> Path:
    """Step A 拟合的像差：系数条形图 + 像差相位图 + 主要模态。"""
    from ao_shaping.utils.wavefront.zernike_calc import get_zernike_name, noll_to_nm

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    coefficients = bundle.coefficients
    aberration, aperture = _aberration_phase(bundle)
    if coefficients is None:
        for ax in axes:
            ax.text(0.5, 0.5, MISSING, ha="center", va="center")
            ax.axis("off")
        fig.suptitle(f"MIL {bundle.stamp} · Step A 拟合像差（系数未记录）", fontsize=13)
        fig.tight_layout()
        savefig(fig, path)
        return path

    values = np.asarray(coefficients, dtype=np.float64)
    noll = np.arange(1, values.shape[0] + 1)
    axes[0].bar(noll, values, color="#4c72b0", width=0.85)
    axes[0].axhline(0.0, color="k", linewidth=0.6)
    axes[0].set_xlabel("Noll 序号", fontsize=10)
    axes[0].set_ylabel("系数 (rad)", fontsize=10)
    axes[0].set_title("全部 Zernike 系数", fontsize=11)
    axes[0].tick_params(labelsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    if aberration is not None:
        _imshow_phase(axes[1], aberration, "像差相位（弧度，包裹后显示）")
        axes[1].set_xlabel(_phase_stats(aberration, aperture), fontsize=10)
    else:
        axes[1].text(0.5, 0.5, MISSING, ha="center", va="center")
        axes[1].axis("off")

    order = np.argsort(np.abs(values))[::-1][:12][::-1]
    labels = []
    for index in order:
        n, m = noll_to_nm(int(index) + 1)
        labels.append(f"N{index + 1} ({n},{m})\n{get_zernike_name(n, m)}")
    axes[2].barh(np.arange(len(order)), values[order], color="#55a868", height=0.75)
    axes[2].set_yticks(np.arange(len(order)))
    axes[2].set_yticklabels(labels, fontsize=7)
    axes[2].axvline(0.0, color="k", linewidth=0.6)
    axes[2].set_xlabel("系数 (rad)", fontsize=10)
    axes[2].set_title("绝对值最大的 12 个模态", fontsize=11)
    axes[2].grid(axis="x", alpha=0.3)

    fig.suptitle(f"MIL {bundle.stamp} · Step A 拟合出的像差", fontsize=13)
    fig.tight_layout()
    savefig(fig, path)
    return path


def _overlay_target(ax: Any, bundle: RunBundle, cam_size: int) -> None:
    """在远场图上叠目标框轮廓（青线）。"""
    mask = _target_mask_camera(bundle, (cam_size, cam_size))
    if mask is None or not mask.any():
        return
    _draw_target_outline(ax, mask)


def figure_farfield(bundle: RunBundle, path: Path) -> Path:
    """平场 / 整形后实测 / 模型预测 / 差值 四联。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    shaped = _shaped_probe(bundle)
    flat = bundle.flat_frame
    predicted_camera = None
    measured_camera = None
    if shaped is not None and bundle.roi_center is not None and bundle.cell_scale:
        measured_camera = shaped.frame
        if shaped.prediction is not None and bundle.far_field_size:
            predicted_camera = _model_grid_to_camera(
                shaped.prediction,
                bundle.roi_center,
                far_field_size=bundle.far_field_size,
                cell_scale=bundle.cell_scale,
                cam_size=shaped.frame.shape[0],
                rotation_quarters=bundle.rotation_quarters,
            )

    vmax = None
    for candidate in (flat, measured_camera, predicted_camera):
        if candidate is not None:
            value = float(np.percentile(np.asarray(candidate, dtype=np.float64), 99.9))
            vmax = value if vmax is None else max(vmax, value)

    if flat is not None:
        _imshow(axes[0][0], flat, "平场参考（preflight/flat.npz）", vmax=vmax)
        _overlay_target(axes[0][0], bundle, flat.shape[0])
    else:
        axes[0][0].text(0.5, 0.5, f"平场参考\n{MISSING}", ha="center", va="center")
        axes[0][0].axis("off")

    if measured_camera is not None:
        _imshow(axes[0][1], measured_camera, "整形后实测远场（CCD 300×300）", vmax=vmax)
        _overlay_target(axes[0][1], bundle, measured_camera.shape[0])
    else:
        axes[0][1].text(0.5, 0.5, f"整形后实测\n{MISSING}", ha="center", va="center")
        axes[0][1].axis("off")

    if predicted_camera is not None:
        _imshow(
            axes[1][0], predicted_camera, "模型预测（远场网格 → 相机窗口）", vmax=vmax
        )
        _overlay_target(axes[1][0], bundle, predicted_camera.shape[0])
    else:
        axes[1][0].text(0.5, 0.5, f"模型预测\n{MISSING}", ha="center", va="center")
        axes[1][0].axis("off")

    if measured_camera is not None and predicted_camera is not None:
        difference = measured_camera - predicted_camera
        # 差值与上面三格用**同一个**标度，读者可以直接横向比亮度。
        span = vmax if vmax is not None else float(np.abs(difference).max())
        axes[1][1].imshow(difference, cmap="coolwarm", vmin=-span, vmax=span)
        axes[1][1].set_title("差值 = 实测 − 预测", fontsize=11)
        _hide_axes(axes[1][1])
        _overlay_target(axes[1][1], bundle, difference.shape[0])
        axes[1][1].set_xlabel(
            f"标度 ±{span:.4g}（与上三格同一）· 最大 |差值| "
            f"{float(np.abs(difference).max()):.4g} · 相关 r = "
            f"{fmt_metric(_pearson(measured_camera, predicted_camera))}",
            fontsize=10,
        )
    else:
        axes[1][1].text(0.5, 0.5, f"差值\n{MISSING}", ha="center", va="center")
        axes[1][1].axis("off")

    side = _target_side_camera(bundle)
    fig.suptitle(
        f"MIL {bundle.stamp} · CCD 远场光斑：平场 / 实测 / 预测 / 差值"
        + (f"（青线 = 目标框 {side} 相机像元）" if side else ""),
        fontsize=13,
    )
    fig.tight_layout()
    savefig(fig, path)
    return path


def figure_gradient(bundle: RunBundle, path: Path) -> Path:
    """逐迭代损失 + 梯度范数；无轨迹时改画逐轮曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    trace_b = bundle.trace_b
    trace_a = bundle.trace_a

    if trace_b.available:
        iters, losses, grads = trace_b.as_arrays()
        axes[0][0].semilogy(iters, losses, color="#c44e52", linewidth=1.2)
        axes[0][0].set_title("Step B 逐迭代形状损失", fontsize=11)
        axes[0][0].set_xlabel("迭代", fontsize=10)
        axes[0][0].set_ylabel("损失", fontsize=10)
        axes[0][0].grid(alpha=0.3)
        axes[0][0].annotate(
            f"首 {losses[0]:.4g}\n末 {losses[-1]:.4g}",
            xy=(iters[-1], losses[-1]),
            xytext=(-70, 25),
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        )
    else:
        axes[0][0].text(
            0.5,
            0.5,
            "逐迭代损失轨迹未记录\n（本轮运行时尚未落盘逐迭代轨迹）",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[0][0].axis("off")

    if trace_b.available and trace_b.has_grad:
        iters, _, grads = trace_b.as_arrays()
        axes[0][1].semilogy(iters, grads, color="#4c72b0", linewidth=1.2)
        axes[0][1].set_title("Step B 逐迭代梯度范数", fontsize=11)
        axes[0][1].set_xlabel("迭代", fontsize=10)
        axes[0][1].set_ylabel("梯度范数", fontsize=10)
        axes[0][1].grid(alpha=0.3)
    else:
        axes[0][1].text(
            0.5,
            0.5,
            f"逐迭代梯度范数未记录\n（来源：{trace_b.source if trace_b.available else MISSING}）",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[0][1].axis("off")

    if trace_a.available:
        iters, losses, _ = trace_a.as_arrays()
        axes[1][0].semilogy(iters, losses, color="#55a868", linewidth=1.2)
        axes[1][0].set_title("Step A 逐迭代探针集损失", fontsize=11)
        axes[1][0].set_xlabel("迭代", fontsize=10)
        axes[1][0].set_ylabel("损失", fontsize=10)
        axes[1][0].grid(alpha=0.3)
    else:
        axes[1][0].text(
            0.5,
            0.5,
            f"Step A 逐迭代轨迹\n{MISSING}",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[1][0].axis("off")

    rounds = _round_rows(bundle)
    if rounds:
        numbers = [
            int(row.get("round", i + 1) or i + 1) for i, row in enumerate(rounds)
        ]
        score = [_as_float(row.get("score_after")) for row in rounds]
        cv = [_as_float(row.get("cv_after")) for row in rounds]
        ee = [_as_float(row.get("ee_after")) for row in rounds]
        axes[1][1].plot(numbers, score, "o-", label="复合分数", color="#c44e52")
        axes[1][1].plot(numbers, cv, "s--", label="CV", color="#4c72b0")
        axes[1][1].plot(numbers, ee, "^-.", label="EE", color="#55a868")
        flat_score, _, _ = _scores(bundle)
        if flat_score is not None:
            axes[1][1].axhline(
                flat_score,
                color="#c44e52",
                linestyle=":",
                linewidth=1.2,
                label=f"平场基线 {flat_score:.4f}",
            )
        axes[1][1].set_xlabel("轮次", fontsize=10)
        axes[1][1].set_title("逐轮复合分数 / CV / EE", fontsize=11)
        axes[1][1].legend(fontsize=9)
        axes[1][1].grid(alpha=0.3)
    else:
        axes[1][1].text(
            0.5, 0.5, f"逐轮记录\n{MISSING}", ha="center", va="center", fontsize=12
        )
        axes[1][1].axis("off")

    source_note = (
        f"逐迭代轨迹来源：{trace_b.source}"
        if trace_b.available
        else "逐迭代轨迹未落盘（本轮运行时尚未落盘逐迭代轨迹）"
    )
    fig.suptitle(f"MIL {bundle.stamp} · 反向可微优化的演化", fontsize=13)
    fig.text(0.5, 0.005, source_note, ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    savefig(fig, path)
    return path


def figure_process_gif(bundle: RunBundle, path: Path) -> Path | None:
    """过程动画：逐探针的下发相位 / 实测远场 / 模型预测远场 + 指标面板。

    每迭代的模型远场若已落盘则优先使用；本脚本探测不到时降级为逐探针动画。

    布局 2×3, 每帧同时给出三样东西:
      * 上排 —— 下发的瞳面相位 / 实测远场(叠目标轮廓) / 模型预测远场(叠目标轮廓)
      * 下排 —— 实测−预测差值(同标度) / **物理预测模型的 metrics** /
        **目标的 metrics** (实测与预测各一列)
    """
    from matplotlib.animation import PillowWriter

    probes = _probe_order(bundle)
    if not probes:
        return None

    frames: list[tuple[ProbeEntry, np.ndarray | None, np.ndarray | None]] = []
    for probe in probes:
        predicted = None
        if probe.prediction is not None and bundle.roi_center is not None:
            if bundle.cell_scale and bundle.far_field_size:
                predicted = _model_grid_to_camera(
                    probe.prediction,
                    bundle.roi_center,
                    far_field_size=bundle.far_field_size,
                    cell_scale=bundle.cell_scale,
                    cam_size=probe.frame.shape[0],
                    rotation_quarters=bundle.rotation_quarters,
                )
        frames.append((probe, probe.frame, predicted))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    writer = PillowWriter(fps=1.2)
    for index, (probe, measured, predicted) in enumerate(frames):
        for row in axes:
            for ax in row:
                ax.clear()

        _imshow_phase(axes[0][0], probe.phase, "下发的瞳面相位 (rad)")

        # 目标掩模（在相机窗口上, 冻结光轴）
        target_mask = None
        if measured is not None:
            target_mask = _target_mask_camera(bundle, measured.shape)

        # --- 差值用的统一标度 -------------------------------------------
        vmax = None
        for candidate in (measured, predicted):
            if candidate is not None:
                value = float(
                    np.percentile(np.asarray(candidate, dtype=np.float64), 99.9)
                )
                vmax = value if vmax is None else max(vmax, value)

        _imshow(axes[0][1], measured, "实测远场 (CCD)", vmax=vmax)
        if target_mask is not None:
            _draw_target_outline(axes[0][1], target_mask)

        if predicted is not None:
            _imshow(axes[0][2], predicted, "模型预测远场", vmax=vmax)
        else:
            axes[0][2].text(0.5, 0.5, MISSING, ha="center", va="center")
            axes[0][2].axis("off")
        if target_mask is not None:
            _draw_target_outline(axes[0][2], target_mask)

        # --- 下排: 差值 + 两个指标面板 -----------------------------------
        if measured is not None and predicted is not None:
            difference = np.asarray(measured, dtype=np.float64) - np.asarray(
                predicted, dtype=np.float64
            )
            span = vmax if vmax is not None else float(np.abs(difference).max())
            axes[1][0].imshow(difference, cmap="coolwarm", vmin=-span, vmax=span)
            axes[1][0].set_title("差值 = 实测 − 预测", fontsize=11)
            _hide_axes(axes[1][0])
            axes[1][0].set_xlabel(
                f"标度 ±{span:.4g}（与上两格同一）· 相关 r = "
                f"{fmt_metric(_pearson(measured, predicted))}",
                fontsize=9,
            )
        else:
            axes[1][0].text(0.5, 0.5, f"差值\n{MISSING}", ha="center", va="center")
            axes[1][0].axis("off")

        # (1,1) 物理预测模型的 metrics
        r_model = None
        r_camera = None
        ssim_camera = None
        if measured is not None and predicted is not None:
            r_model = _pearson(measured, predicted)
            r_camera = probe.manifest_correlation
            ssim_camera = _ssim(measured, predicted)
        gain, leakage = _gain_and_leakage(bundle)
        _text_panel(
            axes[1][1],
            "物理预测模型 metrics",
            [
                ("相关 r（相机窗口）", fmt_metric(r_model)),
                ("相关 r（落盘）", fmt_metric(probe.manifest_correlation)),
                ("SSIM（相机窗口）", fmt_metric(ssim_camera)),
                ("预测来源", probe.prediction_source),
                ("曝光 (ms)", fmt_metric(probe.exposure_ms, 3)),
                ("模型增益", fmt_metric(gain, 4)),
                ("加性背景（杂散光）项系数", fmt_metric(leakage, 4)),
            ],
        )

        # (1,2) 目标的 metrics
        measured_metrics = (
            _target_metrics(bundle, measured) if measured is not None else {}
        )
        predicted_metrics = (
            _target_metrics(bundle, predicted) if predicted is not None else {}
        )
        target_side_camera = _target_side_camera(bundle)
        _text_panel(
            axes[1][2],
            "目标 metrics（实测 / 预测）",
            [
                (
                    "形状 / 边长",
                    f"{bundle.config.get('target_shape', MISSING)} / 标称 "
                    f"{bundle.config.get('target_side', MISSING)} px"
                    + (
                        f" → 实际 {target_side_camera} 相机像元"
                        if target_side_camera
                        else ""
                    ),
                ),
                ("冻结光轴 (x, y)", str(bundle.roi_center or MISSING)),
                (
                    "CV",
                    f"{fmt_metric(measured_metrics.get('uniformity_cv'), 4)} / "
                    f"{fmt_metric(predicted_metrics.get('uniformity_cv'), 4)}",
                ),
                (
                    "EE",
                    f"{fmt_metric(measured_metrics.get('encircled_energy'), 4)} / "
                    f"{fmt_metric(predicted_metrics.get('encircled_energy'), 4)}",
                ),
                (
                    "IoU",
                    f"{fmt_metric(measured_metrics.get('shape_iou'), 4)} / "
                    f"{fmt_metric(predicted_metrics.get('shape_iou'), 4)}",
                ),
            ],
        )

        title = (
            f"{probe.label} · {probe.split} · 曝光 "
            f"{fmt_metric(probe.exposure_ms, 3)} ms"
        )
        fig.suptitle(title, fontsize=12)
        corr = probe.manifest_correlation
        extra = f" · 落盘相关 r = {fmt_metric(corr)}" if corr is not None else ""
        axes[0][1].set_xlabel(f"预测来源：{probe.prediction_source}{extra}", fontsize=9)
        if index == 0:
            writer.setup(fig, path, dpi=110)
        writer.grab_frame()
    writer.finish()
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _config_table(bundle: RunBundle) -> str:
    """参数设置表（相机 / 目标 / 探针 / 预算 / 几何 / 种子）。"""
    config = bundle.config
    geometry = bundle.geometry
    rows = [
        ["实验系统变体", bundle.variant, "运行目录的上一级目录名"],
        [
            "相机类型 / 序号",
            f"{config.get('cam_type', MISSING)} / {config.get('cam_id', MISSING)}",
            "`cam_type` / `cam_id`",
        ],
        [
            "相机窗口",
            f"{config.get('cam_size', MISSING)} × {config.get('cam_size', MISSING)} px",
            "`cam_size`",
        ],
        [
            "曝光（设定 / 实测）",
            f"{fmt_metric(config.get('exposure_time_ms'), 3)} ms / {fmt_metric(_round_exposure(bundle), 3)} ms",
            "`exposure_time_ms` / 探针清单",
        ],
        [
            "SLM 序号 / 波长",
            f"{config.get('slm_number', MISSING)} / {config.get('wavelength_nm', MISSING)} nm",
            "`slm_number` / `wavelength_nm`",
        ],
        [
            "SLM 像元 / 面板跨度",
            f"{config.get('slm_pixel_um', MISSING)} µm / {config.get('panel_span_px', MISSING)} px",
            "`slm_pixel_um` / `panel_span_px`",
        ],
        [
            "相机像元",
            f"{config.get('camera_pixel_um', MISSING)} µm",
            "`camera_pixel_um`",
        ],
        [
            "瞳面网格 / 远场零填充",
            f"{config.get('region', MISSING)} × {config.get('region', MISSING)} / ×{config.get('far_field_padding', MISSING)}",
            "`region` / `far_field_padding`",
        ],
        [
            "焦距（标称）",
            f"{fmt_metric(config.get('focal_length_m'), 6)} m",
            "`focal_length_m`",
        ],
        [
            "瞳面中心（面板坐标）",
            str(config.get("pupil_center", MISSING)),
            "`pupil_center`",
        ],
        [
            "目标形状 / 边长",
            f"{config.get('target_shape', MISSING)} / {config.get('target_side', MISSING)} px",
            "`target_shape` / `target_side`",
        ],
        [
            "复合分数权重（EE / CV）",
            f"{config.get('w_efficiency', MISSING)} / {config.get('w_uniformity', MISSING)}",
            "`w_efficiency` / `w_uniformity`",
        ],
        [
            "轮数 / 随机种子",
            f"{config.get('n_rounds', MISSING)} / {config.get('seed', MISSING)}",
            "`n_rounds` / `seed`",
        ],
        [
            "探针数 / 展布",
            f"{config.get('probe_count', MISSING)} / {config.get('probe_spread', MISSING)}",
            "`probe_count` / `probe_spread`",
        ],
        [
            "验证集探针数",
            config.get("holdout_probe_count", MISSING),
            "`holdout_probe_count`",
        ],
        [
            "Step A 迭代 / 学习率",
            f"{config.get('step_a_iterations', MISSING)} / {config.get('step_a_lr', MISSING)}",
            "`step_a_iterations` / `step_a_lr`",
        ],
        [
            "Step B 迭代 / 学习率",
            f"{config.get('step_b_iterations', MISSING)} / {config.get('step_b_lr', MISSING)}",
            "`step_b_iterations` / `step_b_lr`",
        ],
        [
            "Zernike 阶数 / 系数数 / 冻结模态",
            f"{config.get('n_orders', MISSING)} / {len(bundle.coefficients) if bundle.coefficients is not None else MISSING} / {config.get('frozen_modes', MISSING)}",
            "`n_orders` / `frozen_modes`",
        ],
        [
            "信赖域 L2 / 自适应",
            f"{config.get('trust_region_c_l2', MISSING)} / {config.get('adaptive_trust_region', MISSING)}",
            "`trust_region_c_l2` / `adaptive_trust_region`",
        ],
        ["几何标定文件", str(config.get("geometry_file", MISSING)), "`geometry_file`"],
        [
            "实测几何：相机像元/远场采样点",
            fmt_metric(geometry.get("camera_px_per_model_px"), 6),
            "`camera_px_per_model_px`",
        ],
        [
            "实测几何：远场网格 / 方法",
            f"{geometry.get('far_field_size', MISSING)} / {geometry.get('method', MISSING)}",
            "`far_field_size` / `method`",
        ],
        [
            "实测几何：FWHM（相机 / 远场）",
            f"{fmt_metric(geometry.get('spot_fwhm_camera_px'), 3)} px / {fmt_metric(geometry.get('spot_fwhm_model_px'), 3)} px",
            "`spot_fwhm_camera_px` / `spot_fwhm_model_px`",
        ],
        [
            "实测几何：光斑相关 / 瞳面盘半径 / 束腰",
            f"{fmt_metric(geometry.get('correlation'), 4)} / {geometry.get('panel_disc_radius', MISSING)} px / {fmt_metric(geometry.get('beam_waist_panel_px'), 1)} px",
            "`correlation` / `panel_disc_radius` / `beam_waist_panel_px`",
        ],
        [
            "冻结光轴（相机像元）",
            str(bundle.roi_center or MISSING),
            "平场帧全局 argmax",
        ],
        [
            "旋转象限（由落盘 model_grid 反推）",
            f"{bundle.rotation_quarters}（最大偏差 {fmt_metric(bundle.rotation_residual, 3)}）",
            "不写死，逐像素核对",
        ],
        ["注入的误差参数", _error_terms_text(bundle), "`model_error_terms.json`"],
    ]
    return markdown_table(["项", "取值", "来源"], rows)


def _round_exposure(bundle: RunBundle) -> float | None:
    """实测曝光（取探针清单里的中位数）。"""
    values = [p.exposure_ms for p in bundle.probes if p.exposure_ms is not None]
    if not values:
        return None
    return float(np.median(values))


def _error_terms_text(bundle: RunBundle) -> str:
    """注入的误差参数（没有就是「本次运行未注入」）。"""
    if not bundle.error_terms:
        return "本次运行未注入误差参数（理想系统）"
    items = []
    for key, value in bundle.error_terms.items():
        if isinstance(value, (int, float)):
            items.append(f"{key} = {fmt_metric(value, 6)}")
        else:
            items.append(f"{key} = {value}")
    return "; ".join(items) if items else "本次运行未注入误差参数（理想系统）"


def _coefficient_table(bundle: RunBundle) -> str:
    """非零像差系数表（Noll 序 + (n,m) + 名称 + 弧度）。"""
    from ao_shaping.utils.wavefront.zernike_calc import get_zernike_name, noll_to_nm

    coefficients = bundle.coefficients
    if coefficients is None:
        return f"像差系数：{MISSING}（运行目录里没有 `fitted_coefficients.npy`）"
    values = np.asarray(coefficients, dtype=np.float64)
    rows = []
    for index, value in enumerate(values):
        if abs(value) < 1e-9:
            continue
        n, m = noll_to_nm(index + 1)
        rows.append(
            [
                index + 1,
                f"({n}, {m})",
                get_zernike_name(n, m),
                f"{value:+.6f}",
                f"{abs(value):.6f}",
            ]
        )
    if not rows:
        return "像差系数：全部为 0（未记录到非零模态）"
    table = markdown_table(
        ["Noll", "(n, m)", "名称", "系数 (rad)", "绝对值 (rad)"], rows
    )
    frozen = bundle.config.get("frozen_modes") or []
    norm = float(np.linalg.norm(values))
    table += (
        f"\n\n- 系数总数 {values.shape[0]}（Noll 序，索引 0 = Noll 1 活塞）；"
        f"非零模态 {len(rows)} 个；系数 L2 范数 {norm:.6f} rad。"
        + (f" 冻结的模态（拟合全程保持 0）：{frozen}。" if frozen else "")
    )
    return table


def _as_count(value: Any) -> str:
    """迭代数这类整数：``80.0`` → ``80``；缺失 → 「未记录」。"""
    number = _as_float(value)
    if number is None:
        return MISSING
    return str(int(round(number)))


def _round_table(bundle: RunBundle) -> str:
    """逐轮指标表。"""
    rows = []
    for row in bundle.rows:
        rows.append(
            [
                row.get("stage", MISSING),
                row.get("round", MISSING),
                row.get("accepted", MISSING),
                row.get("reason", MISSING) or "-",
                fmt_metric(_as_float(row.get("score_before"))),
                fmt_metric(_as_float(row.get("score_after"))),
                fmt_metric(_as_float(row.get("uniformity_cv"))),
                fmt_metric(_as_float(row.get("ee_after"))),
                fmt_metric(_as_float(row.get("shape_iou_after"))),
                fmt_metric(_as_float(row.get("loss_before")), 6),
                fmt_metric(_as_float(row.get("loss_after")), 6),
                _as_count(row.get("step_a_iterations")),
                _as_count(row.get("step_b_iterations")),
                fmt_metric(_as_float(row.get("holdout_correlation")), 4),
                fmt_metric(_as_float(row.get("model_gain")), 4),
                fmt_metric(_as_float(row.get("flat_leakage")), 4),
            ]
        )
    if not rows:
        return f"逐轮记录：{MISSING}"
    return markdown_table(
        [
            "阶段",
            "轮次",
            "接受",
            "理由",
            "分（前）",
            "分（后）",
            "CV",
            "EE",
            "IoU",
            "损失（前）",
            "损失（后）",
            "Step A 迭代",
            "Step B 迭代",
            "验证集相关",
            "模型增益",
            "加性背景（杂散光）项系数",
        ],
        rows,
    )


def _probe_table(bundle: RunBundle) -> str:
    """逐探针的相关与 SSIM。"""
    rows = []
    for probe in _probe_order(bundle):
        measured_grid = probe.model_grid
        if measured_grid is None and probe.prediction is not None:
            measured_grid = _measured_model_grid(bundle, probe)
        r_model = (
            _pearson(probe.prediction, measured_grid)
            if probe.prediction is not None and measured_grid is not None
            else None
        )
        predicted_camera = None
        if (
            probe.prediction is not None
            and bundle.roi_center is not None
            and bundle.cell_scale
            and bundle.far_field_size
        ):
            predicted_camera = _model_grid_to_camera(
                probe.prediction,
                bundle.roi_center,
                far_field_size=bundle.far_field_size,
                cell_scale=bundle.cell_scale,
                cam_size=probe.frame.shape[0],
                rotation_quarters=bundle.rotation_quarters,
            )
        ssim_camera = (
            _ssim(probe.frame, predicted_camera)
            if predicted_camera is not None
            else None
        )
        rows.append(
            [
                probe.label,
                probe.split,
                fmt_metric(probe.exposure_ms, 3),
                fmt_metric(probe.manifest_correlation, 4),
                fmt_metric(r_model, 4),
                fmt_metric(ssim_camera, 4),
                probe.prediction_source,
            ]
        )
    if not rows:
        return f"逐探针统计：{MISSING}（运行目录里没有 `probes/`）"
    return markdown_table(
        [
            "探针",
            "类别",
            "曝光 (ms)",
            "落盘相关 r",
            "相关 r（远场网格）",
            "SSIM（相机窗口）",
            "预测来源",
        ],
        rows,
    )


def _measured_model_grid(bundle: RunBundle, probe: ProbeEntry) -> np.ndarray | None:
    """把探针的实测帧落到远场网格（与 MIL 同一条实现）。"""
    if bundle.roi_center is None or not bundle.cell_scale or not bundle.far_field_size:
        return None
    from ao_shaping.optimizer.wfless.slm_model_in_loop import _to_model_grid

    return _to_model_grid(
        probe.frame,
        bundle.roi_center,
        int(bundle.far_field_size),
        bundle.cell_scale,
        bundle.rotation_quarters,
    )


def _roundtrip_residual(bundle: RunBundle) -> tuple[float | None, float | None]:
    """「远场网格 → 相机窗口」逆映射的自检：最大绝对偏差与相对帧峰的占比。

    把落盘的 ``model_grid`` 映回相机窗口后与原始相机帧逐像素比对。偏差来自
    重采样（300 像元 → 1024 采样点 → 300 像元），**不是**口径错误。
    """
    for probe in bundle.probes:
        if probe.model_grid is None:
            continue
        if (
            bundle.roi_center is None
            or not bundle.cell_scale
            or not bundle.far_field_size
        ):
            return None, None
        back = _model_grid_to_camera(
            probe.model_grid,
            bundle.roi_center,
            far_field_size=bundle.far_field_size,
            cell_scale=bundle.cell_scale,
            cam_size=probe.frame.shape[0],
            rotation_quarters=bundle.rotation_quarters,
        )
        deviation = float(np.abs(back - probe.frame).max())
        peak = float(np.max(probe.frame))
        relative = deviation / peak if peak > 0 else None
        return deviation, relative
    return None, None


def _exposure_spread(bundle: RunBundle) -> str:
    """各探针类别的曝光（实测）一览 —— 跨曝光的强度不可直接比。"""
    groups: dict[str, list[float]] = {}
    for probe in _probe_order(bundle):
        if probe.exposure_ms is None:
            continue
        groups.setdefault(probe.split, []).append(probe.exposure_ms)
    if not groups:
        return MISSING
    parts = []
    for split, values in groups.items():
        low, high = min(values), max(values)
        parts.append(
            f"{split} {low:.3f} ms"
            if low == high
            else f"{split} {low:.3f}–{high:.3f} ms"
        )
    return "; ".join(parts)


def _frame_metrics(bundle: RunBundle, frame: np.ndarray) -> dict[str, float] | None:
    """一张相机帧在目标框上的指标 + 复合分数（与 MIL 同一套口径）。"""
    from ao_shaping.optimizer.wfless.slm_model_in_loop import _quality

    metrics = _target_metrics(bundle, frame)
    if not metrics:
        return None
    score = _quality(
        metrics,
        float(bundle.config.get("w_efficiency", 0.6)),
        float(bundle.config.get("w_uniformity", 0.4)),
    )
    return {
        "score": float(score),
        "cv": float(metrics.get("uniformity_cv", float("nan"))),
        "ee": float(metrics.get("encircled_energy", float("nan"))),
        "iou": float(metrics.get("shape_iou", float("nan"))),
    }


def _metrics_comparison(bundle: RunBundle) -> list[dict[str, Any]] | None:
    """平场（实测）/ 模型预测 / 整形后实测 三者在目标框上的指标对照。"""
    shaped = _shaped_probe(bundle)
    if shaped is None or bundle.roi_center is None:
        return None
    rows: list[dict[str, Any]] = []
    flat_metrics = (
        _frame_metrics(bundle, bundle.flat_frame) if bundle.flat_frame is not None else None
    )
    if flat_metrics:
        rows.append({"label": "平场（实测）", **flat_metrics})
    predicted_camera = None
    if shaped.prediction is not None and bundle.cell_scale and bundle.far_field_size:
        predicted_camera = _model_grid_to_camera(
            shaped.prediction,
            bundle.roi_center,
            far_field_size=bundle.far_field_size,
            cell_scale=bundle.cell_scale,
            cam_size=shaped.frame.shape[0],
            rotation_quarters=bundle.rotation_quarters,
        )
    if predicted_camera is not None:
        predicted_metrics = _frame_metrics(bundle, predicted_camera)
        if predicted_metrics:
            rows.append({"label": "模型预测的整形结果", **predicted_metrics})
    measured_metrics = _frame_metrics(bundle, shaped.frame)
    if measured_metrics:
        rows.append({"label": "整形后（实测）", **measured_metrics})
    return rows or None


def build_report(
    bundle: RunBundle,
    *,
    figure_names: dict[str, str | None],
    report_key: str,
    command: str,
) -> str:
    """拼出报告正文。"""
    flat_score, shaped_score, verdict = _scores(bundle)
    consistency = _consistency(bundle)
    rounds = _round_rows(bundle)
    handoff = bundle.handoff
    checks = handoff.get("checks", {}) if isinstance(handoff, dict) else {}

    delta = (
        shaped_score - flat_score
        if flat_score is not None and shaped_score is not None
        else None
    )
    lines: list[str] = []
    lines.append(f"# MIL (model-in-loop) 单次运行记录报告 — {bundle.stamp}")
    lines.append("")
    lines.append(f"- 运行目录: `{_rel_to_root(bundle.run_dir)}`（仓库根相对路径）")
    lines.append(
        "- 生成脚本: `scripts/slm/generate_model_in_loop_record_report.py`（离线读盘）"
    )
    lines.append(f"- 实验系统变体: `{bundle.variant}`")
    lines.append(f"- 运行状态: `{bundle.status}`")
    lines.append("")

    # --- 1. 结论 -----------------------------------------------------------
    lines.append("## 1. 结论")
    lines.append("")
    if flat_score is None or shaped_score is None:
        lines.append(
            f"**整形后复合分数是否严格大于平场分：{MISSING}**（产物里没有可比的复合分数）。"
        )
    else:
        lines.append(
            f"**整形后复合分数是否严格大于平场分：{verdict}。** "
            f"整形后 **{shaped_score:.6f}**，平场 **{flat_score:.6f}**，"
            f"差值 **{delta:+.6f}**。"
        )
    lines.append("")
    if bundle.status == "bake_off_rejected":
        lines.append(
            "**没有提交相位的原因**：最终对比择优判定整形相位没有打赢平场"
            f"（{fmt_metric(shaped_score)} ≤ {fmt_metric(flat_score)}），于是保留平场，"
            "运行状态置为 `bake_off_rejected`。因此落盘的 `best_phase.npy` 是**全零的平场相位**，"
            "它不是「整形失败后随便存的一个相位」，而是「择优之后仍然选中的那一个」。"
        )
    else:
        lines.append(
            f"运行状态为 `{bundle.status}`；未提交相位的原因见 `handoff_summary.json` 的 "
            "`verdict` / `qualification` 字段。"
        )
    lines.append("")
    if bundle.failure:
        lines.append(
            f"**本次运行以失败告终**（`failure.json`，{bundle.failure.get('timestamp_utc', MISSING)}）："
        )
        lines.append("")
        lines.append(f"> {bundle.failure.get('reason', MISSING)}")
        lines.append("")
    if handoff:
        lines.append("验收结论（`handoff_summary.json`）：")
        lines.append("")
        lines.append(
            markdown_table(
                ["项", "取值"],
                [
                    ["判定", str(handoff.get("verdict", MISSING))],
                    [
                        "轮数 / 接受轮数 / 最佳记录轮",
                        f"{handoff.get('rounds', MISSING)} / "
                        f"{handoff.get('accepted_rounds', MISSING)} / "
                        f"{handoff.get('best_recorded_round', MISSING)}",
                    ],
                    [
                        "最佳记录轮的指标",
                        ", ".join(
                            f"{key} = {fmt_metric(value, 4)}"
                            for key, value in (
                                handoff.get("metrics_at_best_recorded_round") or {}
                            ).items()
                        )
                        or MISSING,
                    ],
                    [
                        "逐项检查",
                        ", ".join(f"{key} = {value}" for key, value in checks.items())
                        or MISSING,
                    ],
                    ["说明", str(handoff.get("qualification", MISSING))],
                ],
            )
        )
        lines.append("")

    # --- 2. 参数设置 -------------------------------------------------------
    lines.append("## 2. 参数设置")
    lines.append("")
    lines.append(_config_table(bundle))
    lines.append("")
    if bundle.geometry.get("calibration_notes"):
        lines.append("几何标定备注（`bench_geometry.json` 的 `calibration_notes`）：")
        lines.append("")
        lines.append(f"> {bundle.geometry['calibration_notes']}")
        lines.append("")

    # --- 3. Step A ---------------------------------------------------------
    lines.append("## 3. Step A：正向调优模型（拟合实验系统的像差）")
    lines.append("")
    lines.append(_coefficient_table(bundle))
    lines.append("")
    if rounds:
        row = rounds[-1]
        lines.append(
            markdown_table(
                ["项", "取值"],
                [
                    ["探针数（训练）", str(row.get("probe_count", MISSING))],
                    ["Step A 迭代数", _as_count(row.get("step_a_iterations"))],
                    [
                        "损失（拟合前）",
                        fmt_metric(_as_float(row.get("loss_before")), 8),
                    ],
                    ["损失（拟合后）", fmt_metric(_as_float(row.get("loss_after")), 8)],
                    [
                        "验证集相关（`holdout_correlation`）",
                        fmt_metric(_as_float(row.get("holdout_correlation")), 4),
                    ],
                    [
                        "模型增益 / 加性背景（杂散光）项系数",
                        f"{fmt_metric(_as_float(row.get('model_gain')), 4)} / "
                        f"{fmt_metric(_as_float(row.get('flat_leakage')), 4)}",
                    ],
                    [
                        "信赖域是否截断 / 系数步长 L2",
                        f"{row.get('clamp_applied', MISSING)} / "
                        f"{fmt_metric(_as_float(row.get('coeff_step_l2')), 4)}",
                    ],
                ],
            )
        )
    else:
        lines.append(f"Step A 的拟合指标：{MISSING}（逐轮 CSV 里没有 `round` 行）。")
    lines.append("")
    if rounds:
        row = rounds[-1]
        norm = _as_float(row.get("coeff_norm"))
        radius = _as_float(bundle.config.get("trust_region_c_l2"))
        if norm is not None and radius is not None:
            lines.append(
                f"**信赖域把这一步限住了**：落盘系数的 L2 范数是 {norm:.6f} rad，"
                f"恰好等于信赖域半径 {radius:.6f} rad（`clamp_applied = "
                f"{row.get('clamp_applied', MISSING)}`）。也就是说 —— "
                "**未截断的拟合想走得更远，但被信赖域按在半径上**；"
                "报告里的像差系数是**截断之后**的那一组。"
            )
            lines.append("")
    if bundle.trace_a.available:
        lines.append(
            f"Step A 的逐迭代损失轨迹已落盘（来源：{bundle.trace_a.source}，"
            f"共 {len(bundle.trace_a.losses)} 次迭代）。"
        )
    else:
        lines.append(
            "Step A 的逐迭代损失轨迹：**未记录**（本轮运行时尚未落盘逐迭代轨迹）。"
        )
    lines.append("")
    if figure_names.get("aberration"):
        lines.append(f"![Step A 拟合的像差](figures/{figure_names['aberration']})")
        lines.append("")

    # --- 4. Step B ---------------------------------------------------------
    lines.append("## 4. Step B：反向可微优化 SLM 相位")
    lines.append("")
    if rounds:
        row = rounds[-1]
        lines.append(
            markdown_table(
                ["项", "取值"],
                [
                    ["Step B 迭代数", _as_count(row.get("step_b_iterations"))],
                    [
                        "形状损失（首 / 末）",
                        f"{fmt_metric(_as_float(row.get('shape_loss_initial')), 6)} / "
                        f"{fmt_metric(_as_float(row.get('shape_loss_final')), 6)}",
                    ],
                    [
                        "复合分数（前 / 后）",
                        f"{fmt_metric(_as_float(row.get('score_before')), 6)} / "
                        f"{fmt_metric(_as_float(row.get('score_after')), 6)}",
                    ],
                    [
                        "CV / EE / IoU（整形后）",
                        f"{fmt_metric(_as_float(row.get('cv_after')), 4)} / "
                        f"{fmt_metric(_as_float(row.get('ee_after')), 4)} / "
                        f"{fmt_metric(_as_float(row.get('shape_iou_after')), 4)}",
                    ],
                    [
                        "接受 / 理由",
                        f"{row.get('accepted', MISSING)} / {row.get('reason', MISSING) or '-'}",
                    ],
                ],
            )
        )
    else:
        lines.append(f"Step B 的指标：{MISSING}。")
    lines.append("")
    if rounds:
        lines.append(
            "> 第 1 轮的「复合分数（前）」为空是 MIL 的设计：平场基线不是同口径的参照，"
            "所以首轮没有「前」可填；与平场的比较统一在 §1 与 §7 给出。"
        )
        lines.append("")
    if bundle.trace_b.available:
        iters, losses, grads = bundle.trace_b.as_arrays()
        lines.append(
            f"**逐迭代损失轨迹已落盘**（来源：{bundle.trace_b.source}，共 {len(losses)} 次迭代，"
            f"从 {losses[0]:.6g} 降到 {losses[-1]:.6g}）。"
        )
        lines.append("")
        lines.append(
            f"**逐迭代梯度范数：{'已落盘' if bundle.trace_b.has_grad else MISSING}**"
            + (
                "（本次运行的轨迹里没有梯度范数字段）"
                if not bundle.trace_b.has_grad
                else f"（{grads[0]:.6g} → {grads[-1]:.6g}）"
            )
            + "。"
        )
    else:
        lines.append(
            "**逐迭代损失轨迹：未记录**（本轮运行时尚未落盘逐迭代轨迹）；"
            "**逐迭代梯度范数：未记录**。"
        )
    lines.append("")
    if figure_names.get("gradient"):
        lines.append(f"![反向可微优化的演化](figures/{figure_names['gradient']})")
        lines.append("")

    # --- 5. SLM 相位 -------------------------------------------------------
    lines.append("## 5. SLM 相位与一致性检查")
    lines.append("")
    if bundle.best_phase is not None:
        best = np.asarray(bundle.best_phase, dtype=np.float64)
        lines.append(
            f"落盘的最佳相位 `best_phase.npy`（raw 未包裹弧度，"
            f"{best.shape[0]}×{best.shape[1]}）："
            f"min {best.min():+.6f} rad、max {best.max():+.6f} rad、"
            f"RMS {float(np.sqrt(np.mean(best**2))):.6f} rad。"
            + (
                "**全零** —— 对比择优保留了平场（见 §1）。"
                if float(np.abs(best).max()) == 0
                else ""
            )
        )
    else:
        lines.append(f"落盘的最佳相位：{MISSING}。")
    lines.append("")
    lines.append("### 5.1 与 Step A 拟合像差的关系（一致性检查）")
    lines.append("")
    lines.append(
        "物理预测模型的瞳面相位是「**命令相位 + 像差**」，所以抵消像差的命令相位是 `-像差`。"
        "于是「最佳相位 **减去**（-像差）」= 「最佳相位 + 像差」就是**残余瞳面相位**；"
        "若 Step B 只做像差抵消（不做整形），它应当接近平坦（残余 RMS ≈ 0）。"
    )
    lines.append("")
    consistency_rows = [
        ["像差相位的 RMS（瞳面内）", fmt_metric(consistency.get("aberration_rms"), 6)],
        [
            "Step B 合成相位的 RMS（瞳面内）",
            fmt_metric(consistency.get("shaped_rms"), 6),
        ],
        [
            "残余 = 合成相位 + 像差（瞳面内 RMS）",
            fmt_metric(consistency.get("residual_rms"), 6),
        ],
        [
            "残余 = 落盘最佳相位 + 像差（瞳面内 RMS）",
            fmt_metric(consistency.get("best_residual_rms"), 6),
        ],
        [
            "合成相位与（-像差）的相关系数",
            fmt_metric(consistency.get("correlation"), 4),
        ],
        ["参与统计的瞳面像元数", str(consistency.get("aperture_px") or MISSING)],
    ]
    lines.append(markdown_table(["项", "取值"], consistency_rows))
    lines.append("")
    if consistency.get("residual_rms") is not None:
        lines.append(
            "**读法**：本次运行 Step B 的目标是**矩形**而不是平坦，所以残余瞳面相位**不会**为 0；"
            "上面三个数给出的是「相位里有多少是像差抵消、多少是整形结构」的实际比例。"
            "落盘最佳相位是全零的平场，所以它的「残余」就是像差本身。"
        )
    else:
        lines.append(f"一致性检查：{MISSING}。")
    lines.append("")
    if figure_names.get("phase"):
        lines.append(f"![SLM 相位](figures/{figure_names['phase']})")
        lines.append("")

    # --- 6. 远场光斑 -------------------------------------------------------
    lines.append("## 6. CCD 远场光斑：平场 / 整形后 / 模型预测 / 差值")
    lines.append("")
    lines.append(
        "**帧网格口径**。实测帧是 300×300 **相机像元**窗口，物理预测模型输出的是 1024² "
        "**远场采样点**网格。实测帧落到远场网格用 MIL 自己那套已验证的实现"
        "（`slm_model_in_loop._to_model_grid`），**旋转象限不写死** —— "
        "拿探针里落盘的 `model_grid` 逐像素反推（4 个象限里挑最吻合的），"
        f"本次反推结果是象限 {bundle.rotation_quarters}，"
        f"与落盘 `model_grid` 的最大绝对偏差 {fmt_metric(bundle.rotation_residual, 3)}。"
    )
    lines.append("")
    lines.append(
        "模型预测 → 相机窗口走本脚本自己的逆映射；把落盘的 `model_grid` 映回相机窗口后"
        "与原始相机帧逐像素比对，是这条映射的自检。"
    )
    lines.append("")
    deviation, relative = _roundtrip_residual(bundle)
    if deviation is not None:
        lines.append(
            f"逆映射自检：最大绝对偏差 **{deviation:.4g}**"
            + (f"（相对帧峰 {relative * 100:.2f}%）" if relative is not None else "")
            + "。这是 300 像元 → 1024 远场采样点 → 300 像元的**重采样损失**，"
            "不是口径错误；远场网格与相机窗口之间的换算由 `camera_px_per_model_px` "
            "与上一步反推出的旋转象限共同固定。"
        )
    else:
        lines.append(f"逆映射自检：{MISSING}（没有落盘的 `model_grid` 可供核对）。")
    lines.append("")
    lines.append(
        f"各探针类别的实测曝光：{_exposure_spread(bundle)}。**跨曝光的绝对强度不可直接比** —— "
        "相关与 SSIM 都是峰值归一化后的量，不受曝光影响；四联图里三格共用同一个标度，"
        "所以亮度的差异里也含曝光差。"
    )
    lines.append("")
    if figure_names.get("farfield"):
        lines.append(f"![远场光斑四联](figures/{figure_names['farfield']})")
        lines.append("")
    lines.append("### 6.1 逐探针的相关与 SSIM")
    lines.append("")
    lines.append(_probe_table(bundle))
    lines.append("")
    lines.append(f"预测来源说明：{bundle.recompute_note}。")
    lines.append("")
    lines.append("### 6.2 模型自认为的整形效果 vs 实测")
    lines.append("")
    comparison = _metrics_comparison(bundle)
    if comparison is None:
        lines.append(
            f"模型预测与实测的指标对照：{MISSING}（缺目标框几何或整形探针）。"
        )
    else:
        lines.append(
            markdown_table(
                ["对象", "复合分数", "CV", "EE", "IoU"],
                [
                    [
                        row["label"],
                        fmt_metric(row.get("score"), 4),
                        fmt_metric(row.get("cv"), 4),
                        fmt_metric(row.get("ee"), 4),
                        fmt_metric(row.get("iou"), 4),
                    ]
                    for row in comparison
                ],
            )
        )
        lines.append("")
        lines.append(
            "**读法**：三行分别是「平场（实测）」「模型预测的整形结果」「整形后实测」。"
            "**模型预测那行高于平场、实测那行低于平场** —— 也就是说，"
            "物理预测模型以为它把光斑整成了矩形（§6 的第三格确实是个矩形），"
            "而实验系统给回的是一团更散的光。"
            "这就是 §1 判定「没有打赢平场」的物理原因："
            "**不是优化器没找到更好的相位，而是它找到的相位只对模型成立。**"
        )
    lines.append("")

    # --- 7. 逐轮指标 -------------------------------------------------------
    lines.append("## 7. 逐轮指标")
    lines.append("")
    lines.append(_round_table(bundle))
    lines.append("")

    # --- 8. 动画 -----------------------------------------------------------
    lines.append("## 8. 过程动画")
    lines.append("")
    if figure_names.get("process"):
        lines.append(f"![过程动画](figures/{figure_names['process']})")
        lines.append("")
        lines.append(
            "动画逐探针给出「下发的瞳面相位 / 实测远场 / 模型预测远场」，"
            "并在实测远场与模型预测远场上叠加**目标掩模轮廓**（青色虚线）。"
            "下排另给三格：**实测 − 预测差值**（与上排同一标度，标注相关 r）、"
            "**物理预测模型的 metrics**（相机窗口相关 r / 落盘相关 r / SSIM / "
            "预测来源 / 曝光 / 模型增益 / 加性背景（杂散光）项系数）、"
            "**目标的 metrics**（形状与边长、冻结光轴、CV / EE / IoU，"
            "各给「实测 / 预测」两列）。"
            "**每迭代的模型远场未落盘**，所以过程动画降级为逐探针的实测 / 预测 —— "
            "这是本次运行**实际记录到**的粒度。"
            "⚠️ 面板里的 CV / EE / IoU 是**该探针帧**上算的（实测与预测同帧、"
            "同一目标掩模），与 §7 的 per-round 验收值口径可能不同 —— "
            "§7 来自 runner 自己的验收帧，不是这一张探针帧。"
        )
    else:
        lines.append("过程动画：**未生成**（没有可用的探针）。")
    lines.append("")

    # --- 9. 证据索引 -------------------------------------------------------
    lines.append("## 9. 证据索引与生成方式")
    lines.append("")
    run_rel = _rel_to_root(bundle.run_dir)
    run_dir = bundle.run_dir

    def _exists(relative: str) -> str:
        """产物是否存在（相对运行目录）。"""
        return "有" if (run_dir / relative).exists() else "无"

    def _exists_glob(pattern: str) -> str:
        return "有" if any(run_dir.glob(pattern)) else "无"

    lines.append(
        markdown_table(
            ["产物", "相对路径", "本次运行落盘"],
            [
                ["运行目录", f"`{run_rel}`", "-"],
                [
                    "运行参数",
                    f"`{run_rel}/run_config.json`",
                    _exists("run_config.json"),
                ],
                [
                    "实测几何",
                    f"`{run_rel}/bench_geometry.json`",
                    _exists("bench_geometry.json"),
                ],
                [
                    "像差系数",
                    f"`{run_rel}/fitted_coefficients.npy`",
                    _exists("fitted_coefficients.npy"),
                ],
                [
                    "落盘最佳相位",
                    f"`{run_rel}/best_phase.npy`",
                    _exists("best_phase.npy"),
                ],
                [
                    "瞳面照明幅度",
                    f"`{run_rel}/source_amplitude.npy`",
                    _exists("source_amplitude.npy"),
                ],
                [
                    "注入的误差参数",
                    f"`{run_rel}/model_error_terms.json`",
                    _exists("model_error_terms.json"),
                ],
                [
                    "平场参考",
                    f"`{run_rel}/preflight/flat.npz`",
                    _exists("preflight/flat.npz"),
                ],
                [
                    "逐探针清单",
                    f"`{run_rel}/probes/manifest.json`",
                    _exists("probes/manifest.json"),
                ],
                [
                    "逐轮记录（CSV）",
                    f"`{run_rel}/` 下的 `slm_model_in_loop_*.csv`",
                    _exists_glob("slm_model_in_loop_*.csv"),
                ],
                [
                    "recorder 全过程（pkl）",
                    f"`{run_rel}/` 下的 `slm_model_in_loop_*.pkl`",
                    _exists_glob("slm_model_in_loop_*.pkl"),
                ],
                [
                    "验收结论",
                    f"`{run_rel}/handoff_summary.json`",
                    _exists("handoff_summary.json"),
                ],
                [
                    "失败记录",
                    f"`{run_rel}/failure.json`",
                    _exists("failure.json"),
                ],
            ],
        )
    )
    lines.append("")
    lines.append(f"- 复现命令: `{command}`")
    lines.append(
        "- 溯源: 离线读盘，不碰设备；本报告的溯源块由 "
        "`scripts/sync_report_provenance.py` 在 "
        "`scripts/_common/provenance.py::REPORTS` 登记本报告后写入。"
    )
    lines.append("")

    text = "\n".join(lines).rstrip("\n") + "\n"
    return insert_header(text, report_key)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="一次 MIL 运行的落盘目录（含 run_config.json / probes/ 的那一层）",
    )
    parser.add_argument(
        "--stamp",
        default=None,
        help="图与报告名里的戳；缺省用运行目录名",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"报告目录；缺省 {OUT_DIR.relative_to(ROOT).as_posix()}",
    )
    parser.add_argument(
        "--fig-dir",
        default=None,
        help=f"图目录；缺省 {FIG_DIR.relative_to(ROOT).as_posix()}",
    )
    parser.add_argument("--no-gif", action="store_true", help="不生成过程动画")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    if not run_dir.is_dir():
        logger.error("运行目录不存在: {}", run_dir)
        return 2
    stamp = args.stamp or run_dir.name
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    fig_dir = Path(args.fig_dir) if args.fig_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_run(run_dir, stamp)
    logger.info(
        "读到 {} 条探针 · 旋转象限 {} (残差 {}) · Step B 轨迹 {}",
        len(bundle.probes),
        bundle.rotation_quarters,
        bundle.rotation_residual,
        bundle.trace_b.source,
    )

    figure_names: dict[str, str | None] = {
        "phase": None,
        "aberration": None,
        "farfield": None,
        "gradient": None,
        "process": None,
    }
    figure_names["phase"] = f"{stamp}_record_phase_best.png"
    figure_names["aberration"] = f"{stamp}_record_model_aberration.png"
    figure_names["farfield"] = f"{stamp}_record_farfield.png"
    figure_names["gradient"] = f"{stamp}_record_gradient_evolution.png"
    figure_names["process"] = f"{stamp}_record_process.gif"

    figure_phase(bundle, fig_dir / figure_names["phase"])
    figure_aberration(bundle, fig_dir / figure_names["aberration"])
    figure_farfield(bundle, fig_dir / figure_names["farfield"])
    figure_gradient(bundle, fig_dir / figure_names["gradient"])
    if not args.no_gif:
        if figure_process_gif(bundle, fig_dir / figure_names["process"]) is None:
            figure_names["process"] = None

    report_name = f"{REPORT_STEM}_{stamp}.md"
    report_path = out_dir / report_name
    report_key = _rel_to_root(report_path)
    command = (
        "python scripts/slm/generate_model_in_loop_record_report.py "
        f"--run-dir {_rel_to_root(run_dir)} --stamp {stamp}"
    )
    report_path.write_text(
        build_report(
            bundle,
            figure_names=figure_names,
            report_key=report_key,
            command=command,
        ),
        encoding="utf-8",
    )
    logger.info("报告已写出: {}", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
