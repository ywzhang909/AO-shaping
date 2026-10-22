"""model-in-loop (MIL) 历史运行的**只读**损失审计 + 候选相位清单。

回答三个假设, 并把后续实验要用的相位清单落盘。**不碰硬件, 不写 ``src/``。**

要回答什么
----------
MIL 闭环里正向仿真与实测 CCD 画面差异明显, 整形成绩压不过平场。到底卡在哪:

* **假设 A (迭代次数不足)**: Step A / Step B 的损失在末段仍明显下降 ⇒ 加迭代有用。
* **假设 B (模型失配)**: 损失早已平台化, 而模型预测指标与实测指标存在**系统性**偏差
  ⇒ 加迭代没用, 必须改模型。
* **假设 C (信任域太紧)**: 拟合系数撞上 ``--max-coefficient-norm`` (``clamp_applied``)
  ⇒ 是护栏在拦, 不是步数不够。

判据全部由落盘数据复算:

1. **损失轨迹** —— 优先读 recorder ``*.pkl`` 里的 ``step_a_trace`` / ``step_b_trace``
   (逐迭代), 退化到 CSV 的 ``step_a_loss_<i>`` / ``step_b_loss_<i>`` 逐迭代列。
   给出总相对降幅、**末段斜率** (最后 10% / 20% 迭代做最小二乘)、
   「迭代翻倍还能降多少」的**外推值**、末段抖动 (std/mean)。
2. **信任域** —— ``clamp_applied`` 比例, 步长 L2 / 上限的比值 (``coeff_step_applied``
   与 ``coeff_step_l2``), 以及撞满时损失是否仍在下降。
3. **模型失配** —— 逐 run / 逐轮把**模型预测**与**实测**配对。预测由落盘的
   ``fitted_coefficients.npy`` + ``source_amplitude.npy`` + ``run_config.json``
   正向模型离线复算 (与 MIL 内的 ``forward_intensity`` 同一实现), 并用落盘的
   ``prediction`` 逐像素校验; **校验不通过的轮次一律排除**并如实标注
   (被回滚 / 多轮时落盘系数只对应最后一轮)。
   偏差分解成四类可解释量: **能量尺度 / 宽度(模糊) / 位置(平移) / 形状**。

口径
----
* 相机帧读的就是 ``prepare_frame`` 之后的画面 (``probes/*.npz`` 的 ``frame``),
  **不再二次** ``prepare_frame``。
* 相机 ↔ 模型网格换算复用 MIL 自己的 ``_to_model_grid`` / 反向映射,
  旋转象限用硬件路径的写死值 (``cam_type != "sim"`` → 1), 并用落盘
  ``model_grid`` 校验 (相对 L1 ≈1e-11)。
* 指标 (CV / EE / IoU / 复合分数) 走 ``_metrics_at`` + ``_quality``, 不另立口径。

输出
----
``logs/mil_loss_audit/<时间戳>/`` 下::

    audit.json          # 全量逐 run 诊断
    summary.md          # 中文摘要 (假设 A/B/C 判定 + 数字)
    candidates.json     # 候选相位清单 + 两两距离矩阵
    candidates/*.npy    # 每个候选的相位 (弧度, raw 未包裹)
    candidates_grid.png # 相位 / 实测光斑 / 模型预测 三联缩略图

用法
----
::

    python scripts/slm/analyze_model_in_loop_loss.py
    python scripts/slm/analyze_model_in_loop_loss.py --out-dir logs/mil_loss_audit

术语: 中文叙述遵守 ``docs/terminology_glossary.md`` 的规范术语表 (A/B 对照语境
用「变体」, 加性背景用「加性背景（杂散光）项」, 对比择优不写成英文)。
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from loguru import logger

from ao_shaping.optimizer.wfless.slm_model_in_loop import (
    SlmModelInLoopConfig,
    _make_optimizer,
    _metrics_at,
    _quality,
    _to_model_grid,
)
from ao_shaping.utils.image.beam_metrics import zero_order_center
from scripts._common import fmt_signed

#: 默认扫描根 (都是 ``data/debug`` 下的 MIL 系列)。
DEFAULT_ROOTS = (
    "data/debug/model_in_loop_hw_ab",
    "data/debug/slm_model_in_loop",
    "data/debug/model_in_loop_auto",
    "data/debug/model_in_loop_error_params",
)

#: 找不到的量统一写法 —— 数字全部从产物读, 读不到就写这个, **不填 0**。
MISSING = "未记录"

#: 复算预测与落盘预测的允许相对偏差 (相对该轮预测的峰值)。
#: 超过即判「不可复算」: 落盘系数与该轮实际使用的不一致 (回滚 / 多轮)。
RECOMPUTE_TOL = 1e-5

#: 本体相位距离: 去掉 piston 后的 L2 (弧度), 在 canonical 瞳面掩膜内统计。
PUPIL_RADIUS_FRACTION = 0.5


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------


@dataclass
class Probe:
    """一条探针 (``probes/*.npz`` + ``manifest.json`` 的元信息)。"""

    name: str
    round: int
    split: str
    index: int
    exposure_ms: float | None
    recorded_correlation: float | None
    phase: np.ndarray
    frame: np.ndarray
    model_grid: np.ndarray | None
    prediction: np.ndarray | None

    @property
    def label(self) -> str:
        """短标签, 例如 ``r1 shaped#0``。"""
        return f"r{self.round} {self.split}#{self.index}"


@dataclass
class Trace:
    """一段逐迭代轨迹。"""

    iterations: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    source: str = MISSING

    @property
    def available(self) -> bool:
        """轨迹是否非空。"""
        return len(self.losses) >= 2


@dataclass
class RunData:
    """一次 MIL 运行的落盘产物 (只读)。"""

    campaign: str
    path: Path
    config: dict[str, Any]
    geometry: dict[str, Any]
    rows: list[dict[str, str]]
    records: list[dict[str, Any]]
    status: str
    failure: str | None
    coefficients: np.ndarray | None
    source_amplitude: np.ndarray | None
    best_phase: np.ndarray | None
    flat_frame: np.ndarray | None
    probes: list[Probe]
    trace_a: Trace
    trace_b: Trace

    @property
    def stamp(self) -> str:
        """运行戳 (目录名)。"""
        return self.path.name

    @property
    def variant(self) -> str:
        """实验系统变体名 (``.../<变体>/slm_model_in_loop/<戳>``)。"""
        parts = self.path.parts
        if "slm_model_in_loop" in parts:
            index = len(parts) - 1 - parts[::-1].index("slm_model_in_loop")
            if index >= 1:
                return parts[index - 1]
        return self.path.parent.name

    @property
    def is_hardware(self) -> bool:
        """是否真实硬件运行 (``cam_type != "sim"``)。"""
        return str(self.config.get("cam_type", "daheng")) != "sim"


# ---------------------------------------------------------------------------
# 读盘
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    """读一个 JSON 对象 (不存在 / 坏文件 → 空字典)。"""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读 JSON 失败 {}: {}", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读逐轮 CSV (全部按字符串读, 数值转换留给调用方)。"""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_float(value: Any) -> float | None:
    """把单元格 / JSON 标量转成 float (空 / 非数 → ``None``)。"""
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
    """读一个 ``.npy`` (不存在 / 读失败 → ``None``)。"""
    if not path.is_file():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        logger.warning("读 npy 失败 {}: {}", path, exc)
        return None


def _read_probes(run_dir: Path) -> list[Probe]:
    """读 ``probes/`` 下的全部探针。"""
    probe_dir = run_dir / "probes"
    manifest_path = probe_dir / "manifest.json"
    manifest: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("读探针清单失败 {}: {}", manifest_path, exc)
            payload = []
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, dict):
                    manifest[str(entry.get("file"))] = entry
    probes: list[Probe] = []
    for path in sorted(probe_dir.glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as payload:
                names = set(payload.files)
                phase = np.asarray(payload["phase"], dtype=np.float64)
                frame = np.asarray(payload["frame"], dtype=np.float64)
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
            Probe(
                name=path.name,
                round=int(meta.get("round", 0) or 0),
                split=str(meta.get("split", "unknown")),
                index=int(meta.get("index", 0) or 0),
                exposure_ms=_as_float(meta.get("exposure_ms")),
                recorded_correlation=_as_float(meta.get("correlation")),
                phase=phase,
                frame=frame,
                model_grid=grid,
                prediction=prediction,
            )
        )
    return probes


def _trace_from_pickle(run_dir: Path) -> tuple[Trace, Trace]:
    """从 recorder pkl 读 (Step A 轨迹, Step B 轨迹)。

    只认 ``step_a_trace`` / ``step_b_trace`` 两个键 (recorder 的真实字段名),
    末轮优先 —— 多轮运行时后写的覆盖前面的, 与「最终状态」一致。
    """
    step_a, step_b = Trace(), Trace()
    for path in sorted(run_dir.glob("*.pkl")):
        try:
            payload = pickle.loads(path.read_bytes())
        except (OSError, pickle.PickleError, EOFError, AttributeError) as exc:
            logger.warning("读 pkl 失败 {}: {}", path, exc)
            continue
        if not isinstance(payload, dict):
            continue
        source = f"recorder pkl `{path.name}`"
        for key in sorted(payload, key=lambda k: (isinstance(k, str), k)):
            record = payload[key]
            if not isinstance(record, dict):
                continue
            for attr, target in (("step_a_trace", "a"), ("step_b_trace", "b")):
                entries = record.get(attr)
                if not isinstance(entries, list) or len(entries) < 2:
                    continue
                iterations: list[float] = []
                losses: list[float] = []
                for index, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        continue
                    loss = _as_float(entry.get("loss"))
                    if loss is None:
                        continue
                    iterations.append(float(_as_float(entry.get("iteration")) or index))
                    losses.append(loss)
                if len(losses) < 2:
                    continue
                trace = Trace(iterations=iterations, losses=losses, source=source)
                if target == "a":
                    step_a = trace
                else:
                    step_b = trace
    return step_a, step_b


def _trace_from_csv_columns(rows: Sequence[dict[str, str]]) -> tuple[Trace, Trace]:
    """从逐轮 CSV 里找形如 ``step_a_loss_<i>`` 的逐迭代列 (退化路径)。"""
    if not rows:
        return Trace(), Trace()
    columns = list(rows[0])
    out: dict[str, Trace] = {}
    for prefix in ("step_a", "step_b"):
        pattern = re.compile(rf"^{prefix}_(?:loss|grad_norm)_(\d+)$")
        found: dict[int, float] = {}
        for column in columns:
            match = pattern.match(column)
            if match is None:
                continue
            value = _as_float(rows[0].get(column))
            if value is not None:
                found[int(match.group(1))] = value
        if found:
            out[prefix] = Trace(
                iterations=[float(k) for k in sorted(found)],
                losses=[found[k] for k in sorted(found)],
                source="逐轮 CSV 的逐迭代列",
            )
        else:
            out[prefix] = Trace()
    return out["step_a"], out["step_b"]


def load_run(campaign: str, run_dir: Path) -> RunData:
    """把一次 MIL 运行读成 :class:`RunData` (不构造正向模型)。"""
    config = _read_json(run_dir / "run_config.json")
    geometry = _read_json(run_dir / "bench_geometry.json")
    csv_paths = sorted(run_dir.glob("*.csv"))
    rows = _read_csv_rows(csv_paths[-1]) if csv_paths else []
    records = _read_records(run_dir)
    status = ""
    for path in sorted(run_dir.glob("slm_model_in_loop_*.json")):
        payload = _read_json(path)
        if payload.get("status"):
            status = str(payload["status"])
            break
    failure = None
    failure_path = run_dir / "failure.json"
    if failure_path.is_file():
        payload = _read_json(failure_path)
        failure = str(payload.get("reason") or payload.get("error") or "failure.json")
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
    if flat_frame is None:
        flats = [p for p in probes if p.split == "flat"]
        if flats:
            flat_frame = flats[0].frame
    trace_a, trace_b = _trace_from_pickle(run_dir)
    if not (trace_a.available and trace_b.available):
        csv_a, csv_b = _trace_from_csv_columns(rows)
        if not trace_a.available:
            trace_a = csv_a
        if not trace_b.available:
            trace_b = csv_b
    return RunData(
        campaign=campaign,
        path=run_dir,
        config=config,
        geometry=geometry,
        rows=rows,
        records=records,
        status=status,
        failure=failure,
        coefficients=_load_npy(run_dir / "fitted_coefficients.npy"),
        source_amplitude=_load_npy(run_dir / "source_amplitude.npy"),
        best_phase=_load_npy(run_dir / "best_phase.npy"),
        flat_frame=flat_frame,
        probes=probes,
        trace_a=trace_a,
        trace_b=trace_b,
    )


def _read_records(run_dir: Path) -> list[dict[str, Any]]:
    """把 recorder pkl 的逐轮记录摊平成列表 (按轮次升序)。"""
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.pkl")):
        try:
            payload = pickle.loads(path.read_bytes())
        except (OSError, pickle.PickleError, EOFError, AttributeError):
            continue
        if not isinstance(payload, dict):
            continue
        for key in sorted(payload, key=lambda k: (isinstance(k, str), k)):
            record = payload[key]
            if not isinstance(record, dict):
                continue
            if "round" not in record:
                continue
            records.append({k: v for k, v in record.items()})
        if records:
            break
    records.sort(key=lambda r: float(r.get("round", 0) or 0))
    return records


def discover_runs(roots: Sequence[str]) -> list[tuple[str, Path]]:
    """在给定根目录下找出全部带 ``run_config.json`` 的运行目录。"""
    out: list[tuple[str, Path]] = []
    for raw in roots:
        base = Path(raw)
        if not base.is_dir():
            logger.warning("跳过不存在的根目录 {}", base)
            continue
        campaign = base.name
        for run_dir in sorted({p.parent for p in base.rglob("run_config.json")}):
            out.append((campaign, run_dir))
    return out


# ---------------------------------------------------------------------------
# 轨迹诊断
# ---------------------------------------------------------------------------


def _tail_fit(
    iterations: np.ndarray, losses: np.ndarray, fraction: float
) -> tuple[float, float, float]:
    """末段 ``fraction`` 比例的迭代上做线性拟合, 返回 ``(斜率, 斜率标准误, t)``。

    斜率单位是「每步损失变化」(负=还在降)。标准误来自最小二乘残差, 所以
    ``|t| < 2`` 就说明这段时间的趋势在抖动里读不出来 —— 用斜率外推「再加多少
    迭代能再降多少」必须以此为前提, 否则外推的是噪声。
    """
    count = max(4, int(round(losses.size * fraction)))
    count = min(count, losses.size)
    x = np.asarray(iterations[-count:], dtype=np.float64)
    y = np.asarray(losses[-count:], dtype=np.float64)
    if x.size < 4 or float(np.ptp(x)) <= 0.0:
        return float("nan"), float("nan"), float("nan")
    design = np.vstack([x, np.ones_like(x)]).T
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    dof = max(x.size - 2, 1)
    variance = float(residual @ residual) / dof
    try:
        covariance = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:  # pragma: no cover - 退化设计矩阵
        return float(coefficients[0]), float("nan"), float("nan")
    stderr = float(math.sqrt(max(variance * covariance[0, 0], 0.0)))
    slope = float(coefficients[0])
    if not math.isfinite(stderr) or stderr <= 0.0:
        return slope, stderr, float("nan")
    return slope, stderr, float(slope / stderr)


def trace_diagnostics(trace: Trace) -> dict[str, Any]:
    """一次 Step 的损失轨迹诊断 (总降幅 / 末段斜率 / 外推 / 抖动)。"""
    out: dict[str, Any] = {
        "available": trace.available,
        "source": trace.source,
        "iterations": int(len(trace.losses)),
    }
    if not trace.available:
        out.update({"note": f"{MISSING}（没有逐迭代轨迹）"})
        return out
    iterations = np.asarray(trace.iterations, dtype=np.float64)
    losses = np.asarray(trace.losses, dtype=np.float64)
    first, last = float(losses[0]), float(losses[-1])
    out["loss_first"] = first
    out["loss_last"] = last
    out["relative_drop"] = float((first - last) / first) if first != 0 else None
    # 迭代间距 (多数是 1, 但记录的也可能是 0,1,2,... 的累计号)
    steps = float(np.median(np.diff(iterations))) if iterations.size > 1 else 1.0
    steps = steps if steps > 0 else 1.0
    for fraction, label in ((0.1, "tail_10pct"), (0.2, "tail_20pct")):
        slope, stderr, t_stat = _tail_fit(iterations, losses, fraction)
        window = max(4, int(round(losses.size * fraction)))
        window = min(window, losses.size)
        segment = losses[-window:]
        mean = float(segment.mean())
        out[f"{label}_slope_per_step"] = slope
        out[f"{label}_slope_stderr"] = stderr
        out[f"{label}_slope_t"] = t_stat
        out[f"{label}_slope_relative"] = (
            float(slope / mean) if mean != 0.0 and math.isfinite(slope) else None
        )
        out[f"{label}_std"] = float(segment.std())
        out[f"{label}_mean"] = mean
        out[f"{label}_jitter"] = (
            float(segment.std() / abs(mean)) if mean != 0.0 else None
        )
        out[f"{label}_range_relative"] = (
            float((segment.max() - segment.min()) / abs(mean)) if mean != 0.0 else None
        )
        # 迭代翻倍的额外降幅: 按末段斜率线性外推, 再除以当前损失。
        # 只在 |t| > 2 (趋势可分辨) 时给数值, 否则写「不可分辨」——
        # 抖动主导的末段上按斜率外推外推的是噪声。
        remaining = losses.size * steps
        detectable = math.isfinite(t_stat) and abs(t_stat) > 2.0
        if detectable and last:
            extrapolated = float(-slope * remaining / last)
            out[f"{label}_extrapolated_extra_drop"] = extrapolated
            # 线性外推超过「把当前损失降到 0」时它已经没有物理意义, 单独标出来,
            # 汇总时只统计可信的那部分。
            out[f"{label}_extrapolation_reliable"] = bool(0.0 <= extrapolated < 1.0)
            if math.isfinite(stderr):
                out[f"{label}_extrapolated_extra_drop_ci"] = [
                    float(-(slope - 2 * stderr) * remaining / last),
                    float(-(slope + 2 * stderr) * remaining / last),
                ]
            out[f"{label}_extrapolation_note"] = (
                "趋势可分辨（|t|>2），按末段斜率线性外推"
                + ("" if 0.0 <= extrapolated < 1.0 else "；外推已超过当前损失，不可信")
            )
        else:
            out[f"{label}_extrapolated_extra_drop"] = None
            out[f"{label}_extrapolation_reliable"] = False
            out[f"{label}_extrapolation_note"] = (
                "未外推：末段趋势与抖动不可分辨（|t|≤2），按斜率外推只会放大噪声"
            )
        # 该段内的实际相对变化, 用来判断是否真在动
        out[f"{label}_observed_change"] = (
            float((segment[0] - segment[-1]) / segment[0])
            if segment.size > 1 and segment[0] != 0.0
            else None
        )
    return out


# ---------------------------------------------------------------------------
# 信任域诊断
# ---------------------------------------------------------------------------


def trust_region_diagnostics(run: RunData) -> dict[str, Any]:
    """``clamp_applied`` 比例 + 步长 / 上限比值 + 撞满时损失是否仍在降。

    ⚠️ 只有**真正跑了 Step A** 的轮次 (``step_a_iterations > 0``) 才有损失变化
    可言: ``step_a_iterations=0`` 的是零拟合对照 (loss 前后恒等), 把它们算成
    「未撞满轮的损失降幅」会把「没有拟合」误读成「拟合没有收益」。
    """
    limit = _as_float(run.config.get("trust_region_c_l2"))
    step_a_iterations = _as_float(run.config.get("step_a_iterations")) or 0.0
    rounds = [r for r in run.records if float(r.get("round", 0) or 0) >= 1]
    out: dict[str, Any] = {
        "limit": limit,
        "n_rounds": len(rounds),
        "n_clamped": 0,
        "step_a_iterations": step_a_iterations,
        "steps": [],
    }
    if not rounds:
        out["note"] = f"{MISSING}（没有逐轮记录）"
        return out
    clamped_loss_drops: list[float] = []
    free_loss_drops: list[float] = []
    ratios: list[float] = []
    for record in rounds:
        clamped = bool(record.get("clamp_applied"))
        step_l2 = _as_float(record.get("coeff_step_l2"))
        applied = _as_float(record.get("coeff_step_applied"))
        norm = _as_float(record.get("coeff_norm"))
        loss_before = _as_float(record.get("loss_before"))
        loss_after = _as_float(record.get("loss_after"))
        drop = None
        if (
            step_a_iterations > 0
            and loss_before
            and loss_after is not None
            and loss_before != 0.0
        ):
            drop = float((loss_before - loss_after) / loss_before)
        out["n_clamped"] += 1 if clamped else 0
        if applied is not None and limit:
            ratios.append(float(applied / limit))
        if clamped and drop is not None:
            clamped_loss_drops.append(drop)
        if not clamped and drop is not None:
            free_loss_drops.append(drop)
        out["steps"].append(
            {
                "round": int(float(record.get("round", 0) or 0)),
                "clamped": clamped,
                "step_l2_raw": step_l2,
                "step_l2_applied": applied,
                "applied_over_limit": (
                    float(applied / limit) if applied is not None and limit else None
                ),
                "coeff_norm": norm,
                "coeff_norm_over_limit": (
                    float(norm / limit) if norm is not None and limit else None
                ),
                "loss_before": loss_before,
                "loss_after": loss_after,
                "loss_relative_drop": drop,
                "accepted": bool(record.get("accepted")),
            }
        )
    out["fitted_rounds"] = int(step_a_iterations > 0) * len(rounds)
    out["clamped_fraction"] = float(out["n_clamped"] / len(rounds)) if rounds else None
    out["applied_over_limit_max"] = max(ratios) if ratios else None
    out["applied_over_limit_median"] = float(np.median(ratios)) if ratios else None
    out["clamped_loss_drop_median"] = (
        float(np.median(clamped_loss_drops)) if clamped_loss_drops else None
    )
    out["free_loss_drop_median"] = (
        float(np.median(free_loss_drops)) if free_loss_drops else None
    )
    out["n_clamped_with_loss"] = len(clamped_loss_drops)
    out["n_free_with_loss"] = len(free_loss_drops)
    out["loss_drop_positive_fraction_clamped"] = (
        float(
            sum(1 for value in clamped_loss_drops if value > 0.0)
            / len(clamped_loss_drops)
        )
        if clamped_loss_drops
        else None
    )
    return out


# ---------------------------------------------------------------------------
# 模型失配：预测复算 + 四类分解
# ---------------------------------------------------------------------------


def _config_for(run: RunData) -> Any | None:
    """用该 run 自己的 ``run_config.json`` 重建 :class:`SlmModelInLoopConfig`。

    ``geometry_file`` 必须指向**本 run** 的 ``bench_geometry.json``:
    落盘的原值常指向一个已被删除的临时目录, 构造会直接报错。
    """
    geometry_path = run.path / "bench_geometry.json"
    if not geometry_path.is_file():
        return None
    known = set(SlmModelInLoopConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in run.config.items() if k in known}
    kwargs["geometry_file"] = str(geometry_path)
    kwargs["device"] = kwargs.get("device") or "cpu"
    try:
        return SlmModelInLoopConfig(**kwargs)
    except (TypeError, ValueError) as exc:
        logger.warning("{}: 配置重建失败 {}", run.stamp, exc)
        return None


def _gain_leakage(run: RunData) -> tuple[float | None, float | None]:
    """从逐轮 CSV 读正向增益与加性背景 (杂散光) 项系数 (末轮优先)。"""
    gain = leakage = None
    for row in run.rows:
        if row.get("stage") != "round":
            continue
        value = _as_float(row.get("model_gain"))
        if value is not None:
            gain = value
        value = _as_float(row.get("flat_leakage"))
        if value is not None:
            leakage = value
    return gain, leakage


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    """两个同形数组的 Pearson 相关系数。"""
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size < 2:
        return None
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0:
        return None
    return float(np.dot(a, b) / denominator)


def _peak_normalised_mse(predicted: np.ndarray, measured: np.ndarray) -> float | None:
    """峰值归一化 MSE (与 MIL 的 ``_probe_set_loss`` 同一口径)。"""
    peak_m = float(np.max(measured))
    peak_p = float(np.max(predicted))
    if peak_m <= 0.0 or peak_p <= 0.0:
        return None
    estimate = np.asarray(predicted, dtype=np.float64) / peak_p
    target = np.asarray(measured, dtype=np.float64) / peak_m
    return float(np.mean((estimate - target) ** 2))


def _second_moment_radius(image: np.ndarray) -> float | None:
    """二阶矩半径 (以强度质心为中心, 像素为单位)。"""
    array = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    total = float(array.sum())
    if total <= 0.0:
        return None
    ys, xs = np.indices(array.shape, dtype=np.float64)
    cy = float((array * ys).sum() / total)
    cx = float((array * xs).sum() / total)
    variance = float((array * ((ys - cy) ** 2 + (xs - cx) ** 2)).sum() / total)
    return float(math.sqrt(max(variance, 0.0)))


def _centroid(image: np.ndarray) -> tuple[float, float] | None:
    """强度质心 ``(x, y)``。"""
    array = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    total = float(array.sum())
    if total <= 0.0:
        return None
    ys, xs = np.indices(array.shape, dtype=np.float64)
    return float((array * xs).sum() / total), float((array * ys).sum() / total)


def _best_integer_shift(
    first: np.ndarray, second: np.ndarray, search: int = 8
) -> tuple[int, int]:
    """在 ``±search`` 格内找让 ``first`` 与 ``second`` 相关最大的整格平移 ``(dy, dx)``。

    用一次 FFT 互相关定位 (零填充到两倍尺寸以避免循环卷绕), 再把结果夹到
    ``±search`` —— 逐格暴力搜索在 230×230 的视场上要 289 次整幅 roll, 太慢。
    """
    from scipy.signal import fftconvolve

    a = np.asarray(first, dtype=np.float64)
    a = a - a.mean()
    b = np.asarray(second, dtype=np.float64)
    b = b - b.mean()
    if not np.any(a) or not np.any(b):
        return 0, 0
    correlation = fftconvolve(a, b[::-1, ::-1], mode="same")
    dy_index, dx_index = np.unravel_index(
        int(np.argmax(correlation)), correlation.shape
    )
    dy = int(dy_index) - a.shape[0] // 2
    dx = int(dx_index) - a.shape[1] // 2
    return int(np.clip(dy, -search, search)), int(np.clip(dx, -search, search))


def decompose_mismatch(
    predicted: np.ndarray, measured: np.ndarray, footprint: int
) -> dict[str, Any]:
    """把「预测 vs 实测」的差异拆成能量 / 宽度 / 位置 / 形状四类。

    全部在**相机视场**那一小块上算 (相机只看到模型视场的一个窗口, 整幅比较里
    一大片两边都是 0, 会把能量比压向 1 而掩盖失配)。

    Args:
        predicted: 模型预测的远场网格。
        measured: 实测帧落到同一模型网格后的结果。
        footprint: 相机视场在该网格上占的边长 (格)。

    Returns:
        四类量与相关系数; 任一量为 0 时对应项写 ``None``。
    """
    half = max(int(footprint) // 2, 1)
    rows, cols = np.asarray(predicted).shape
    r0, c0 = rows // 2 - half, cols // 2 - half
    r1, c1 = r0 + 2 * half, c0 + 2 * half
    pred = np.maximum(np.asarray(predicted, dtype=np.float64)[r0:r1, c0:c1], 0.0)
    meas = np.maximum(np.asarray(measured, dtype=np.float64)[r0:r1, c0:c1], 0.0)
    out: dict[str, Any] = {"footprint_cells": int(2 * half)}

    # 整幅 (1024²) 对照: 相机视场之外两边都是 0, 但它能说明「裁剪本身」有没有
    # 参与失配 —— 若整幅与视场的宽度比差很多, 那差异主要来自视场换算而不是光学。
    full_pred = np.maximum(np.asarray(predicted, dtype=np.float64), 0.0)
    full_meas = np.maximum(np.asarray(measured, dtype=np.float64), 0.0)
    full_sum_p, full_sum_m = float(full_pred.sum()), float(full_meas.sum())
    out["energy_ratio_full"] = (
        float(full_sum_p / full_sum_m) if full_sum_m > 0 else None
    )
    full_radius_p = _second_moment_radius(full_pred)
    full_radius_m = _second_moment_radius(full_meas)
    out["width_ratio_full"] = (
        float(full_radius_p / full_radius_m)
        if full_radius_p and full_radius_m
        else None
    )
    out["correlation_full"] = _pearson(full_pred, full_meas)

    sum_p, sum_m = float(pred.sum()), float(meas.sum())
    out["energy_pred"] = sum_p
    out["energy_measured"] = sum_m
    out["energy_ratio"] = float(sum_p / sum_m) if sum_m > 0 else None

    radius_p = _second_moment_radius(pred)
    radius_m = _second_moment_radius(meas)
    out["radius_pred"] = radius_p
    out["radius_measured"] = radius_m
    out["width_ratio"] = float(radius_p / radius_m) if radius_p and radius_m else None

    centre_p = _centroid(pred)
    centre_m = _centroid(meas)
    if centre_p and centre_m:
        dx = centre_p[0] - centre_m[0]
        dy = centre_p[1] - centre_m[1]
        out["centroid_shift_cells"] = float(math.hypot(dx, dy))
        out["centroid_shift_dx_cells"] = float(dx)
        out["centroid_shift_dy_cells"] = float(dy)
    else:
        out["centroid_shift_cells"] = None
        out["centroid_shift_dx_cells"] = None
        out["centroid_shift_dy_cells"] = None

    # 形状: 先各自按总能量归一 (去掉尺度), 再整格平移到质心重合 (去掉平移),
    # 最后取相关系数。相关系数高 = 只剩「形状」能解释的部分也小。
    if sum_p > 0 and sum_m > 0:
        pred_normalised = pred / sum_p
        meas_normalised = meas / sum_m
        dy, dx = _best_integer_shift(pred_normalised, meas_normalised)
        aligned = np.roll(np.roll(pred_normalised, dy, axis=0), dx, axis=1)
        out["shape_shift_dy_cells"] = int(dy)
        out["shape_shift_dx_cells"] = int(dx)
        out["shape_correlation"] = _pearson(aligned, meas_normalised)
        residual = aligned - meas_normalised
        out["shape_residual_l1"] = float(np.abs(residual).sum())
    else:
        out["shape_correlation"] = None
        out["shape_residual_l1"] = None
    out["correlation_raw"] = _pearson(pred, meas)
    out["mse_peak_normalised"] = _peak_normalised_mse(pred, meas)
    return out


def analyse_mismatch(run: RunData) -> dict[str, Any]:
    """复算模型预测并与实测配对 (逐轮 + 整形探针 + c=0 对照)。"""
    out: dict[str, Any] = {"available": False, "note": MISSING}
    if not run.is_hardware:
        out["note"] = "仿真运行（无模型↔实测失配可言）"
        return out
    if not run.probes:
        out["note"] = f"{MISSING}（没有落盘探针）"
        return out
    if run.coefficients is None or run.source_amplitude is None:
        out["note"] = f"{MISSING}（缺 fitted_coefficients.npy / source_amplitude.npy）"
        return out
    config = _config_for(run)
    if config is None:
        out["note"] = f"{MISSING}（无法重建正向模型配置）"
        return out
    geometry = run.geometry
    far_field_size = int(geometry.get("far_field_size", 0) or 0)
    scale = _as_float(geometry.get("camera_px_per_model_px"))
    if not far_field_size or not scale:
        out["note"] = (
            f"{MISSING}（bench_geometry.json 缺少 far_field_size / camera_px_per_model_px）"
        )
        return out
    flats = [p for p in run.probes if p.split == "flat"]
    if not flats:
        out["note"] = f"{MISSING}（没有平场探针, 冻结光轴无从确定）"
        return out
    flat = flats[0]
    rotation = 0 if str(run.config.get("cam_type", "daheng")) == "sim" else 1
    centre = zero_order_center(flat.frame, refine=False)
    flat_grid = _to_model_grid(flat.frame, centre, far_field_size, scale, rotation)

    # 相机窗口在模型网格上的边长 (格): 相机窗口 width / (相机像元每格)
    cam_size = int(flat.frame.shape[0])
    footprint = max(8, int(round(cam_size / scale)))

    try:
        optimizer = _make_optimizer(config, None, run.source_amplitude)
    except Exception as exc:  # pragma: no cover - 依赖 torch 后端
        out["note"] = f"{MISSING}（正向模型构造失败：{exc}）"
        return out

    gain, leakage = _gain_leakage(run)

    def predict(coefficients: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """复算一条探针的模型预测 (含增益 / 加性背景项, 与 MIL 内一致)。"""
        predicted = optimizer.forward_intensity(
            coefficients, phase, run.source_amplitude
        )
        if gain is not None and leakage is not None:
            predicted = gain * predicted + leakage * flat_grid
        return np.asarray(predicted, dtype=np.float64)

    # --- 逐轮校验复算可信度 ------------------------------------------------
    by_round: dict[int, list[Probe]] = {}
    for probe in run.probes:
        if probe.split == "flat":
            continue
        by_round.setdefault(probe.round, []).append(probe)
    round_checks: list[dict[str, Any]] = []
    verified_rounds: list[int] = []
    for round_index in sorted(by_round):
        deviations: list[float] = []
        for probe in by_round[round_index]:
            if probe.prediction is None:
                continue
            candidate = predict(run.coefficients, probe.phase)
            peak = float(np.abs(probe.prediction).max()) or 1.0
            deviations.append(float(np.abs(candidate - probe.prediction).max()) / peak)
        deviation = max(deviations) if deviations else None
        verified = deviation is not None and deviation <= RECOMPUTE_TOL
        if verified:
            verified_rounds.append(round_index)
        round_checks.append(
            {
                "round": round_index,
                "n_with_recorded_prediction": len(deviations),
                "max_relative_deviation": deviation,
                "verified": verified,
            }
        )
    out["round_checks"] = round_checks
    out["verified_rounds"] = verified_rounds
    if not verified_rounds:
        worst = [
            c["max_relative_deviation"]
            for c in round_checks
            if c["max_relative_deviation"] is not None
        ]
        detail = f"{max(worst):.3g}" if worst else MISSING
        out["note"] = (
            f"不可复算：落盘系数与该轮实际使用的系数不一致（最大相对偏差 "
            f"{detail}）。常见原因是该轮被回滚（拒绝）或多轮运行的落盘系数只对应末轮。"
        )
        return out
    out["available"] = True
    out["note"] = (
        f"已复算并校验（{len(verified_rounds)}/{len(round_checks)} 轮通过, "
        f"相对偏差 ≤{RECOMPUTE_TOL:g}）"
    )
    out["gain"] = gain
    out["leakage"] = leakage
    out["roi_center"] = list(centre)
    out["rotation_quarters"] = rotation
    out["camera_px_per_model_px"] = scale
    out["far_field_size"] = far_field_size
    out["footprint_cells"] = footprint

    from scripts.slm.generate_model_in_loop_record_report import (
        _model_grid_to_camera,
        _target_side_camera,
    )

    # --- 逐探针配对 --------------------------------------------------------
    probe_rows: list[dict[str, Any]] = []
    for round_index in verified_rounds:
        for probe in by_round[round_index]:
            predicted = predict(run.coefficients, probe.phase)
            measured_grid = _to_model_grid(
                probe.frame, centre, far_field_size, scale, rotation
            )
            row: dict[str, Any] = {
                "round": round_index,
                "split": probe.split,
                "index": probe.index,
                "file": probe.name,
                "exposure_ms": probe.exposure_ms,
                "recorded_correlation": probe.recorded_correlation,
                "correlation": _pearson(predicted, measured_grid),
                "decomposition": decompose_mismatch(
                    predicted, measured_grid, footprint
                ),
            }
            probe_rows.append(row)
    out["probes"] = probe_rows

    # --- 整形探针: 相机域指标对照 -----------------------------------------
    shaped_probes: list[Probe] = []
    for round_index in verified_rounds:
        shaped_probes.extend([p for p in by_round[round_index] if p.split == "shaped"])
    if shaped_probes:
        probe = shaped_probes[-1]
        predicted = predict(run.coefficients, probe.phase)
        predicted_camera = _model_grid_to_camera(
            predicted,
            centre,
            far_field_size=far_field_size,
            cell_scale=scale,
            cam_size=int(probe.frame.shape[0]),
            rotation_quarters=rotation,
        )
        out["shaped_metrics"] = _metrics_comparison(run, probe, predicted_camera)
        out["shaped_decomposition"] = decompose_mismatch(
            predicted,
            _to_model_grid(probe.frame, centre, far_field_size, scale, rotation),
            footprint,
        )
        out["shaped_probe_file"] = probe.name
        out["shaped_phase_stats"] = _phase_stats(probe.phase)

    # --- 拟合增益: 拟合系数 vs c=0 (无像差) 在验证集探针上的残差 ----------
    holdout = [p for r in verified_rounds for p in by_round[r] if p.split == "holdout"]
    train = [p for r in verified_rounds for p in by_round[r] if p.split == "train"]
    out["fit_gain"] = {
        "holdout": _residual_comparison(
            holdout, run.coefficients, predict, centre, far_field_size, scale, rotation
        ),
        "train": _residual_comparison(
            train, run.coefficients, predict, centre, far_field_size, scale, rotation
        ),
    }
    return out


def _metrics_comparison(
    run: RunData,
    probe: Probe,
    predicted_camera: np.ndarray,
) -> dict[str, Any]:
    """整形探针在目标框上的「模型预测 vs 实测」指标 (同一套 SDK 口径)。

    目标框边长复用报告脚本的 ``_target_side_camera`` (它编码了「配置里记的是
    相机像元的标称值, 实际用的是换算后的值」这条易错口径), 指标本身走
    MIL 自己的 ``_metrics_at`` + ``_quality``。
    """
    from scripts.slm.generate_model_in_loop_record_report import _target_side_camera

    side = _target_side_camera(_TargetSideShim(run, probe))
    if side is None or side <= 0:
        return {"note": f"{MISSING}（无法换算目标框在相机像元里的边长）"}
    flats = [p for p in run.probes if p.split == "flat"]
    centre = zero_order_center(flats[0].frame, refine=False)
    rotation = 0 if not run.is_hardware else 1
    shape = str(run.config.get("target_shape", "square"))
    w_efficiency = float(run.config.get("w_efficiency", 0.6))
    w_uniformity = float(run.config.get("w_uniformity", 0.4))

    def score(frame: np.ndarray | None) -> dict[str, float] | None:
        if frame is None:
            return None
        metrics = _metrics_at(
            np.asarray(frame, dtype=np.float64),
            int(side),
            centre,
            target_shape=shape,
            camera_rotation_quarters=rotation,
        )
        if not metrics:
            return None
        return {
            "score": _quality(metrics, w_efficiency, w_uniformity),
            "cv": float(metrics.get("uniformity_cv", float("nan"))),
            "ee": float(metrics.get("encircled_energy", float("nan"))),
            "iou": float(metrics.get("shape_iou", float("nan"))),
        }

    measured = score(probe.frame)
    model = score(predicted_camera)
    flat = score(run.flat_frame)
    out: dict[str, Any] = {"target_side_camera_px": int(side)}
    for label, value in (
        ("flat_measured", flat),
        ("shaped_measured", measured),
        ("shaped_model", model),
    ):
        out[label] = value
    if measured and model:
        out["model_minus_measured"] = {
            key: float(model[key] - measured[key])
            for key in ("score", "cv", "ee", "iou")
        }
    return out


@dataclass
class _TargetSideShim:
    """只为复用报告脚本的 ``_target_side_camera`` 而做的极薄适配。"""

    run: RunData
    probe: Probe

    @property
    def config(self) -> dict[str, Any]:
        """该 run 的配置字典。"""
        return self.run.config

    @property
    def cell_scale(self) -> float:
        """相机像元每模型格。"""
        return float(self.run.geometry.get("camera_px_per_model_px", 0.0) or 0.0)

    @property
    def far_field_size(self) -> int:
        """远场网格边长。"""
        return int(self.run.geometry.get("far_field_size", 0) or 0)


def _residual_comparison(
    probes: Sequence[Probe],
    coefficients: np.ndarray,
    predict: Any,
    centre: tuple[int, int],
    far_field_size: int,
    scale: float,
    rotation: int,
) -> dict[str, Any]:
    """拟合系数与 c=0 在同一批探针上的残余 (峰值归一化 MSE + 相关系数)。"""
    if not probes:
        return {"note": f"{MISSING}（该 run 没有这一类探针）"}
    zeros = np.zeros_like(np.asarray(coefficients, dtype=np.float64))
    fitted_errors: list[float] = []
    zero_errors: list[float] = []
    fitted_correlations: list[float] = []
    zero_correlations: list[float] = []
    for probe in probes:
        measured = _to_model_grid(probe.frame, centre, far_field_size, scale, rotation)
        fitted = predict(coefficients, probe.phase)
        baseline = predict(zeros, probe.phase)
        error = _peak_normalised_mse(fitted, measured)
        base_error = _peak_normalised_mse(baseline, measured)
        correlation = _pearson(fitted, measured)
        base_correlation = _pearson(baseline, measured)
        if error is not None:
            fitted_errors.append(error)
        if base_error is not None:
            zero_errors.append(base_error)
        if correlation is not None:
            fitted_correlations.append(correlation)
        if base_correlation is not None:
            zero_correlations.append(base_correlation)
    out: dict[str, Any] = {
        "n": len(probes),
        "loss_fitted": float(np.mean(fitted_errors)) if fitted_errors else None,
        "loss_zero": float(np.mean(zero_errors)) if zero_errors else None,
        "correlation_fitted": float(np.mean(fitted_correlations))
        if fitted_correlations
        else None,
        "correlation_zero": float(np.mean(zero_correlations))
        if zero_correlations
        else None,
    }
    if out["loss_fitted"] is not None and out["loss_zero"]:
        out["loss_improvement_relative"] = float(
            1.0 - out["loss_fitted"] / out["loss_zero"]
        )
    if out["correlation_fitted"] is not None and out["correlation_zero"] is not None:
        out["correlation_gain"] = float(
            out["correlation_fitted"] - out["correlation_zero"]
        )
    return out


def _phase_stats(phase: np.ndarray) -> dict[str, Any]:
    """相位的 min / max / RMS / PV (弧度, raw 未包裹)。"""
    array = np.asarray(phase, dtype=np.float64)
    if array.size == 0:
        return {"note": MISSING}
    return {
        "shape": list(array.shape),
        "min_rad": float(array.min()),
        "max_rad": float(array.max()),
        "rms_rad": float(np.sqrt(np.mean(array**2))),
        "pv_rad": float(array.max() - array.min()),
    }


# ---------------------------------------------------------------------------
# 运行清单
# ---------------------------------------------------------------------------


def inventory(run: RunData) -> dict[str, Any]:
    """一次运行的清单条目 (曝光 / 窗口 / 损失种类 / 探针数 / 迭代数 / 结果)。"""
    config = run.config
    round_rows = [r for r in run.rows if r.get("stage") == "round"]
    flat_row = next((r for r in run.rows if r.get("stage") == "calibration"), None)
    flat_score = _as_float(flat_row.get("score_after")) if flat_row else None
    shaped_score = _as_float(round_rows[-1].get("score_after")) if round_rows else None
    probe_splits: dict[str, int] = {}
    for probe in run.probes:
        probe_splits[probe.split] = probe_splits.get(probe.split, 0) + 1
    round_iterations = [
        {
            "round": int(float(r.get("round", 0) or 0)),
            "step_a_iterations": _as_float(r.get("step_a_iterations")),
            "step_b_iterations": _as_float(r.get("step_b_iterations")),
            "accepted": bool(r.get("accepted")),
            "reason": r.get("reason") or None,
            "score_before": _as_float(r.get("score_before")),
            "score_after": _as_float(r.get("score_after")),
            "holdout_correlation": _as_float(r.get("holdout_correlation")),
            "clamp_applied": bool(r.get("clamp_applied")),
        }
        for r in round_rows
    ]
    return {
        "campaign": run.campaign,
        "variant": run.variant,
        "stamp": run.stamp,
        "path": _rel(run.path),
        "cam_type": config.get("cam_type"),
        "exposure_ms": _as_float(config.get("exposure_time_ms")),
        "round_exposure_ms": [_as_float(r.get("exposure_ms")) for r in round_rows],
        "cam_size": config.get("cam_size"),
        "region": config.get("region"),
        "n_orders": config.get("n_orders"),
        "far_field_padding": config.get("far_field_padding"),
        "target_shape": config.get("target_shape"),
        "target_side": config.get("target_side"),
        "loss_kind": config.get("loss_kind", MISSING),
        "fourier_weight": config.get("fourier_weight", MISSING),
        "selfcal_affine": config.get("selfcal_affine", MISSING),
        "selfcal_travel": config.get("selfcal_travel", MISSING),
        "error_params": config.get("error_params", MISSING),
        "probe_count": config.get("probe_count"),
        "holdout_probe_count": config.get("holdout_probe_count"),
        "probes_on_disk": probe_splits,
        "step_a_iterations": config.get("step_a_iterations"),
        "step_b_iterations": config.get("step_b_iterations"),
        "n_rounds_requested": config.get("n_rounds"),
        "rounds_recorded": len(round_rows),
        "round_iterations": round_iterations,
        "trust_region_c_l2": config.get("trust_region_c_l2"),
        "adaptive_trust_region": config.get("adaptive_trust_region"),
        "status": run.status or MISSING,
        "failure": run.failure,
        "flat_score": flat_score,
        "shaped_score": shaped_score,
        "score_delta": (
            float(shaped_score - flat_score)
            if flat_score is not None and shaped_score is not None
            else None
        ),
        "trace_a": trace_diagnostics(run.trace_a),
        "trace_b": trace_diagnostics(run.trace_b),
    }


def _rel(path: Path) -> str:
    """仓库根相对路径 (正斜杠)。"""
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# 候选相位清单
# ---------------------------------------------------------------------------


def _pupil_mask(region: int) -> np.ndarray:
    """canonical 瞳面掩膜 (半径 ``region//2`` 的圆), 与 Zernike 基同口径。"""
    radius = max(1, int(round(region * PUPIL_RADIUS_FRACTION)))
    ys, xs = np.indices((region, region), dtype=np.float64)
    centre = (region - 1) / 2.0
    return ((ys - centre) ** 2 + (xs - centre) ** 2) <= radius**2


def _phase_vector(phase: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """掩膜内的相位向量, 去掉 piston (掩膜内均值)。"""
    values = np.asarray(phase, dtype=np.float64)[mask]
    return values - float(values.mean())


def collect_candidates(
    analyses: Sequence[dict[str, Any]],
    runs: dict[str, RunData],
    *,
    want: int,
    min_improvement: float,
    region: int | None = None,
) -> dict[str, Any]:
    """挑出「模型确实把损失 / 指标降下来」的相位, 再做多样性挑选。

    选取口径 (逐条可复算):

    1. 该轮 ``accepted == True``;
    2. 该轮的实测复合分数 ``score_after`` 高于同一 run 的平场基线 ``score_before``
       / 标定基线, 相对改善 ≥ ``min_improvement``;
    3. 该轮的模型预测可复算 (``verified_rounds`` 含该轮);
    4. 相位来自该轮实际下发的整形探针 (``split == "shaped"``), 或运行末端的
       ``best_phase.npy`` (全零视为「保留平场」, 不收)。

    多样性: 以「去掉 piston 后的 L2 距离」为度量做最远点采样 (farthest-point),
    保证选中的相位彼此尽量不同; 只有一条合格候选时退化为按改善排序取前 ``want``。
    """
    pool: list[dict[str, Any]] = []
    for analysis in analyses:
        run = runs[analysis["path"]]
        if not run.is_hardware:
            continue
        mismatch = analysis.get("mismatch") or {}
        verified = set(mismatch.get("verified_rounds") or [])
        flat_row = next((r for r in run.rows if r.get("stage") == "calibration"), None)
        flat_score = _as_float(flat_row.get("score_after")) if flat_row else None
        for record in analysis["inventory"]["round_iterations"]:
            round_index = record["round"]
            if not record["accepted"] or round_index not in verified:
                continue
            row = next(
                (
                    r
                    for r in run.rows
                    if r.get("stage") == "round"
                    and int(float(r.get("round", 0) or 0)) == round_index
                ),
                None,
            )
            if row is None:
                continue
            before = _as_float(row.get("score_before")) or flat_score
            after = _as_float(row.get("score_after"))
            if before is None or after is None or before <= 0:
                continue
            improvement = float((after - before) / before)
            if improvement < min_improvement:
                continue
            shaped = [
                p for p in run.probes if p.round == round_index and p.split == "shaped"
            ]
            phase, phase_source = None, MISSING
            if shaped:
                phase = np.asarray(shaped[-1].phase, dtype=np.float64)
                phase_source = f"probes/`{shaped[-1].name}`（该轮实际下发的整形相位）"
            elif run.best_phase is not None and float(np.abs(run.best_phase).max()) > 0:
                phase = np.asarray(run.best_phase, dtype=np.float64)
                phase_source = "`best_phase.npy`（运行末端提交的相位）"
            if phase is None:
                continue
            pool.append(
                {
                    "run_path": _rel(run.path),
                    "campaign": run.campaign,
                    "variant": run.variant,
                    "stamp": run.stamp,
                    "round": round_index,
                    "phase_source": phase_source,
                    "phase": phase,
                    "region": int(phase.shape[0]),
                    "score_before": float(before),
                    "score_after": float(after),
                    "relative_improvement": improvement,
                    "cv_after": _as_float(row.get("cv_after")),
                    "ee_after": _as_float(row.get("ee_after")),
                    "iou_after": _as_float(row.get("shape_iou_after")),
                    "holdout_correlation_recorded": _as_float(
                        row.get("holdout_correlation")
                    ),
                    "accept_reason": row.get("reason"),
                    "decomposition": mismatch.get("shaped_decomposition"),
                    "shaped_metrics": mismatch.get("shaped_metrics"),
                    "holdout_correlation": mismatch.get("fit_gain", {})
                    .get("holdout", {})
                    .get("correlation_fitted"),
                    "mismatch": mismatch,
                    "run": run,
                    "shaped_probe": shaped[-1] if shaped else None,
                }
            )
    # 相位长度不同 -> 无法直接比距离, 所以**按 region 分组各自挑选**, 每组一份
    # 距离矩阵; 主组 = 条目最多的那个 (可用 --region 指定)。
    if not pool:
        return {"note": "没有合格候选", "candidates": [], "distance_matrix": None}
    counts: dict[int, int] = {}
    for item in pool:
        counts[item["region"]] = counts.get(item["region"], 0) + 1
    chosen_region = int(region) if region is not None else max(counts, key=counts.get)
    groups: dict[int, list[dict[str, Any]]] = {}
    for item in pool:
        groups.setdefault(item["region"], []).append(item)
    secondary = [
        _select_within_region(groups[value], want, int(value))
        for value in sorted(groups)
        if value != chosen_region
    ]
    primary_items = groups.get(chosen_region)
    if not primary_items:
        return {
            "note": f"指定 region={chosen_region} 下没有合格候选",
            "candidates": [],
            "distance_matrix": None,
            "region_counts": counts,
            "secondary_groups": secondary,
        }
    primary = _select_within_region(primary_items, want, chosen_region)
    primary.update(
        {
            "region_counts": counts,
            "pool_size": len(pool),
            "secondary_groups": secondary,
        }
    )
    return primary


def _select_within_region(
    pool: Sequence[dict[str, Any]], want: int, region: int
) -> dict[str, Any]:
    """在单一 region 的合格池里按「改善排序 + 最远点采样」选出至多 ``want`` 条。"""
    mask = _pupil_mask(region)
    items = [dict(item) for item in pool]
    for item in items:
        item["vector"] = _phase_vector(item["phase"], mask)
    items.sort(key=lambda item: -item["relative_improvement"])
    selected: list[dict[str, Any]] = [items[0]]
    remaining = items[1:]
    while remaining and len(selected) < want:
        best_index, best_distance = 0, -np.inf
        for index, item in enumerate(remaining):
            distance = min(
                float(np.linalg.norm(item["vector"] - other["vector"]))
                for other in selected
            )
            if distance > best_distance:
                best_index, best_distance = index, distance
        selected.append(remaining.pop(best_index))

    vectors = np.stack([item["vector"] for item in selected])
    n = len(selected)
    distance = np.zeros((n, n))
    correlation = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance[i, j] = float(np.linalg.norm(vectors[i] - vectors[j]))
            correlation[i, j] = float(
                np.dot(vectors[i], vectors[j])
                / max(
                    float(np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])),
                    1e-30,
                )
            )
    upper = distance[np.triu_indices(n, 1)] if n > 1 else np.zeros(1)
    return {
        "region": int(region),
        "pupil_mask_px": int(mask.sum()),
        "pool_size": len(items),
        "candidates": selected,
        "distance_matrix": distance.tolist(),
        "correlation_matrix": correlation.tolist(),
        "distance_summary": {
            "max": float(upper.max()) if n > 1 else 0.0,
            "min": float(upper.min()) if n > 1 else 0.0,
            "median": float(np.median(upper)) if n > 1 else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# 汇总判定
# ---------------------------------------------------------------------------


def _finite(values: Sequence[float | None]) -> list[float]:
    """过滤 ``None`` / 非数。"""
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _stat(values: Sequence[float | None], *, signed: bool = False) -> dict[str, Any]:
    """一组数的 mean / median / min / max / 样本数 (+ 可选: 负值占比)。"""
    clean = _finite(values)
    if not clean:
        return {"n": 0, "note": MISSING}
    out = {
        "n": len(clean),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }
    if signed:
        out["fraction_negative"] = float(
            sum(1 for value in clean if value < 0.0) / len(clean)
        )
    return out


def aggregate(analyses: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把逐 run 诊断汇总成假设 A/B/C 的判据。"""
    hardware = [a for a in analyses if a["is_hardware"]]
    out: dict[str, Any] = {}

    def trace_stats(subset: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
        """一组运行在某个 Step 上的轨迹统计。"""
        items = [a["inventory"][key] for a in subset]
        used = [item for item in items if item.get("available")]
        return {
            "runs": len(used),
            "relative_drop": _stat([item.get("relative_drop") for item in used]),
            "tail_10pct_slope_relative": _stat(
                [item.get("tail_10pct_slope_relative") for item in used]
            ),
            "tail_20pct_slope_relative": _stat(
                [item.get("tail_20pct_slope_relative") for item in used]
            ),
            "tail_10pct_slope_t": _stat(
                [item.get("tail_10pct_slope_t") for item in used]
            ),
            "tail_20pct_slope_t": _stat(
                [item.get("tail_20pct_slope_t") for item in used]
            ),
            "tail_10pct_extrapolated_extra_drop": _stat(
                [
                    item.get("tail_10pct_extrapolated_extra_drop")
                    if item.get("tail_10pct_extrapolation_reliable")
                    else None
                    for item in used
                ]
            ),
            "tail_20pct_extrapolated_extra_drop": _stat(
                [
                    item.get("tail_20pct_extrapolated_extra_drop")
                    if item.get("tail_20pct_extrapolation_reliable")
                    else None
                    for item in used
                ]
            ),
            "tail_20pct_jitter": _stat(
                [item.get("tail_20pct_jitter") for item in used]
            ),
            "tail_20pct_observed_change": _stat(
                [item.get("tail_20pct_observed_change") for item in used]
            ),
            "tail_20pct_mean": _stat([item.get("tail_20pct_mean") for item in used]),
            "loss_first": _stat([item.get("loss_first") for item in used]),
            "loss_last": _stat([item.get("loss_last") for item in used]),
            "iterations": _stat([item.get("iterations") for item in used]),
            "runs_with_trace": len(used),
            "runs_with_detectable_trend": sum(
                1
                for item in used
                if item.get("tail_20pct_extrapolated_extra_drop") is not None
            ),
            "runs_with_reliable_extrapolation": sum(
                1 for item in used if item.get("tail_20pct_extrapolation_reliable")
            ),
            "runs_still_moving": sum(
                1
                for item in used
                if (item.get("tail_20pct_observed_change") or 0.0) > 0.05
                and (item.get("tail_20pct_observed_change") or 0.0)
                > 3.0 * (item.get("tail_20pct_jitter") or 1.0)
            ),
            "runs_diverging": sum(
                1 for item in used if (item.get("relative_drop") or 0.0) < -0.05
            ),
        }

    def trace_block(key: str, budget_key: str) -> dict[str, Any]:
        """轨迹统计 + 按迭代预算分组 (不同预算的收敛程度完全不同, 不能混着平均)。"""
        block = trace_stats(hardware, key)
        budgets: dict[str, list[dict[str, Any]]] = {}
        for analysis in hardware:
            budget = analysis["inventory"].get(budget_key)
            label = MISSING if budget is None else f"{int(budget)}"
            budgets.setdefault(label, []).append(analysis)
        block["by_iteration_budget"] = [
            {"iteration_budget": label, **trace_stats(subset, key)}
            for label, subset in sorted(
                budgets.items(), key=lambda item: (-len(item[1]), item[0])
            )
            if any(a["inventory"][key].get("available") for a in subset)
        ]
        return block

    out["step_a"] = trace_block("trace_a", "step_a_iterations")
    out["step_b"] = trace_block("trace_b", "step_b_iterations")

    trust = [a["trust_region"] for a in hardware if a["trust_region"].get("n_rounds")]
    fitted = [t for t in trust if (t.get("step_a_iterations") or 0.0) > 0.0]
    out["trust_region"] = {
        "runs": len(trust),
        "rounds": int(sum(t["n_rounds"] for t in trust)),
        "clamped_rounds": int(sum(t["n_clamped"] for t in trust)),
        "clamped_round_fraction": (
            float(
                sum(t["n_clamped"] for t in trust) / sum(t["n_rounds"] for t in trust)
            )
            if trust
            else None
        ),
        "runs_with_any_clamp": int(sum(1 for t in trust if t["n_clamped"])),
        "fitted_runs": len(fitted),
        "fitted_rounds": int(sum(t["n_rounds"] for t in fitted)),
        "fitted_rounds_clamped": int(sum(t["n_clamped"] for t in fitted)),
        "fitted_clamped_fraction": (
            float(
                sum(t["n_clamped"] for t in fitted) / sum(t["n_rounds"] for t in fitted)
            )
            if fitted
            else None
        ),
        "zero_fit_rounds": int(
            sum(
                t["n_rounds"]
                for t in trust
                if (t.get("step_a_iterations") or 0.0) <= 0.0
            )
        ),
        "applied_over_limit": _stat([t.get("applied_over_limit_max") for t in trust]),
        "clamped_loss_drop_median": _stat(
            [t.get("clamped_loss_drop_median") for t in trust]
        ),
        "free_loss_drop_median": _stat([t.get("free_loss_drop_median") for t in trust]),
        "clamped_rounds_with_loss": int(
            sum(t.get("n_clamped_with_loss", 0) for t in trust)
        ),
        "free_rounds_with_loss": int(sum(t.get("n_free_with_loss", 0) for t in trust)),
        "loss_drop_positive_fraction_clamped": _stat(
            [t.get("loss_drop_positive_fraction_clamped") for t in trust]
        ),
        "coeff_norm_over_limit": _stat(
            [step["coeff_norm_over_limit"] for t in trust for step in t["steps"]]
        ),
        "step_l2_raw_over_limit": _stat(
            [
                (
                    float(step["step_l2_raw"]) / float(t["limit"])
                    if step.get("step_l2_raw") is not None and t.get("limit")
                    else None
                )
                for t in trust
                for step in t["steps"]
            ]
        ),
    }

    mismatch = [a["mismatch"] for a in hardware if a["mismatch"].get("available")]
    out["mismatch"] = {
        "runs_analysed": len(mismatch),
    }
    shaped_metrics = [
        m["shaped_metrics"]["model_minus_measured"]
        for m in mismatch
        if isinstance(m.get("shaped_metrics"), dict)
        and isinstance(m["shaped_metrics"].get("model_minus_measured"), dict)
    ]
    out["mismatch"]["model_minus_measured"] = {
        key: _stat([row.get(key) for row in shaped_metrics], signed=True)
        for key in ("score", "cv", "ee", "iou")
    }
    decompositions = [
        m["shaped_decomposition"] for m in mismatch if m.get("shaped_decomposition")
    ]
    out["mismatch"]["shaped_decomposition"] = {
        key: _stat([d.get(key) for d in decompositions])
        for key in (
            "energy_ratio",
            "width_ratio",
            "centroid_shift_cells",
            "shape_correlation",
            "correlation_raw",
            "mse_peak_normalised",
            "energy_ratio_full",
            "width_ratio_full",
            "correlation_full",
        )
    }
    # 分日期 (= 分实验会话) 的失配方向: 一整天一套光路/一套几何标定, 混着平均会
    # 把方向相反的会话互相抵消, 掩盖「这次会话到底错在哪一类」。
    by_date: dict[str, list[dict[str, Any]]] = {}
    for analysis in hardware:
        block = analysis["mismatch"]
        if not block.get("available"):
            continue
        stamp = str(analysis["inventory"]["stamp"])
        label = stamp[:8] if stamp[:8].isdigit() else "unknown"
        by_date.setdefault(label, []).append(block)
    out["mismatch"]["by_date"] = [
        {
            "date": label,
            "runs": len(subset),
            "energy_ratio": _stat(
                [
                    (s.get("shaped_decomposition") or {}).get("energy_ratio")
                    for s in subset
                ]
            ).get("median"),
            "width_ratio": _stat(
                [
                    (s.get("shaped_decomposition") or {}).get("width_ratio")
                    for s in subset
                ]
            ).get("median"),
            "centroid_shift_cells": _stat(
                [
                    (s.get("shaped_decomposition") or {}).get("centroid_shift_cells")
                    for s in subset
                ]
            ).get("median"),
            "shape_correlation": _stat(
                [
                    (s.get("shaped_decomposition") or {}).get("shape_correlation")
                    for s in subset
                ]
            ).get("median"),
            "correlation_raw": _stat(
                [
                    (s.get("shaped_decomposition") or {}).get("correlation_raw")
                    for s in subset
                ]
            ).get("median"),
            "score_bias": _stat(
                [
                    (
                        (s.get("shaped_metrics") or {}).get("model_minus_measured")
                        or {}
                    ).get("score")
                    for s in subset
                ]
            ).get("median"),
            "cv_bias": _stat(
                [
                    (
                        (s.get("shaped_metrics") or {}).get("model_minus_measured")
                        or {}
                    ).get("cv")
                    for s in subset
                ]
            ).get("median"),
            "camera_px_per_model_px": _stat(
                [s.get("camera_px_per_model_px") for s in subset]
            ).get("median"),
        }
        for label, subset in sorted(by_date.items())
    ]
    probe_decompositions = [
        row["decomposition"] for m in mismatch for row in m.get("probes", [])
    ]
    out["mismatch"]["probe_decomposition"] = {
        key: _stat([d.get(key) for d in probe_decompositions])
        for key in (
            "energy_ratio",
            "width_ratio",
            "centroid_shift_cells",
            "shape_correlation",
            "correlation_raw",
            "mse_peak_normalised",
        )
    }
    out["mismatch"]["fit_gain_holdout"] = {
        key: _stat([m["fit_gain"]["holdout"].get(key) for m in mismatch])
        for key in (
            "loss_fitted",
            "loss_zero",
            "loss_improvement_relative",
            "correlation_fitted",
            "correlation_zero",
            "correlation_gain",
        )
    }
    gain_improvements = _finite(
        [m["fit_gain"]["holdout"].get("loss_improvement_relative") for m in mismatch]
    )
    out["mismatch"]["fit_gain_holdout"]["fraction_improving"] = (
        float(sum(1 for v in gain_improvements if v > 0.0) / len(gain_improvements))
        if gain_improvements
        else None
    )
    out["scores"] = {
        "flat_score": _stat([a["inventory"].get("flat_score") for a in hardware]),
        "shaped_score": _stat([a["inventory"].get("shaped_score") for a in hardware]),
        "score_delta": _stat([a["inventory"].get("score_delta") for a in hardware]),
        "runs_beating_flat": sum(
            1 for a in hardware if (a["inventory"].get("score_delta") or 0.0) > 0.0
        ),
        "runs": len(hardware),
        "status_counts": _count_by([a["inventory"].get("status") for a in hardware]),
    }
    return out


def _count_by(values: Sequence[Any]) -> dict[str, int]:
    """按取值计数 (``None`` / 空 → 「未记录」)。"""
    counts: dict[str, int] = {}
    for value in values:
        label = MISSING if not value else str(value)
        counts[label] = counts.get(label, 0) + 1
    return counts


def verdicts(agg: dict[str, Any]) -> dict[str, Any]:
    """把汇总数字翻成假设 A/B/C 的判定 (每条带判据与数字)。"""
    out: dict[str, Any] = {}
    step_a = agg["step_a"]
    step_b = agg["step_b"]

    def tail_evidence(block: dict[str, Any], label: str) -> dict[str, Any]:
        groups = []
        for group in block["by_iteration_budget"]:
            groups.append(
                {
                    "iteration_budget": group["iteration_budget"],
                    "runs": group["runs_with_trace"],
                    "median_total_relative_drop": group["relative_drop"].get("median"),
                    "median_tail_observed_change": group[
                        "tail_20pct_observed_change"
                    ].get("median"),
                    "median_tail_slope_relative": group[
                        "tail_20pct_slope_relative"
                    ].get("median"),
                    "median_tail_slope_t": group["tail_20pct_slope_t"].get("median"),
                    "median_tail_jitter": group["tail_20pct_jitter"].get("median"),
                    "median_extrapolated_extra_drop_doubling": group[
                        "tail_20pct_extrapolated_extra_drop"
                    ].get("median"),
                    "runs_with_detectable_trend": group["runs_with_detectable_trend"],
                    "runs_with_reliable_extrapolation": group[
                        "runs_with_reliable_extrapolation"
                    ],
                    "runs_still_moving": group["runs_still_moving"],
                    "runs_diverging": group["runs_diverging"],
                }
            )
        # 「还在动」= 某个预算档 (≥3 次运行) 在末段还能观测到 >5% 的相对变化,
        # 且趋势可分辨 (|t|>2)。抖动主导或已经压平的档一律读成收敛。
        moving = [
            group
            for group in groups
            if group["runs"] >= 3
            and (group["median_tail_observed_change"] or 0.0) > 0.05
            and group["runs_with_detectable_trend"] >= 0.5 * group["runs"]
        ]
        return {
            "label": label,
            "median_total_relative_drop": block["relative_drop"].get("median"),
            "median_tail_slope_per_step_relative": block[
                "tail_20pct_slope_relative"
            ].get("median"),
            "median_tail_slope_t": block["tail_20pct_slope_t"].get("median"),
            "median_tail_observed_change": block["tail_20pct_observed_change"].get(
                "median"
            ),
            "median_tail_jitter": block["tail_20pct_jitter"].get("median"),
            "median_extrapolated_extra_drop_doubling": block[
                "tail_20pct_extrapolated_extra_drop"
            ].get("median"),
            "runs_with_detectable_trend": block["runs_with_detectable_trend"],
            "runs_with_reliable_extrapolation": block[
                "runs_with_reliable_extrapolation"
            ],
            "runs_still_moving": block["runs_still_moving"],
            "runs_diverging": block["runs_diverging"],
            "still_declining_above_noise": bool(moving),
            "declining_budgets": [group["iteration_budget"] for group in moving],
            "by_iteration_budget": groups,
            "runs_with_trace": block["runs_with_trace"],
        }

    evidence_a = [tail_evidence(step_a, "Step A"), tail_evidence(step_b, "Step B")]
    # 判定要区分「历史档」与「当前最大档」: 加迭代的收益只在还没收敛的那一档上,
    # 而当前实践用的是最大预算档。最大档已收敛 = 这个问题现在不再是瓶颈。
    all_groups = [
        (row["label"], group, row)
        for row in evidence_a
        for group in row["by_iteration_budget"]
    ]
    numeric = [
        (label, group, row)
        for label, group, row in all_groups
        if group["iteration_budget"] != MISSING
    ]
    largest_budget = (
        max(float(group["iteration_budget"]) for _, group, _ in numeric)
        if numeric
        else None
    )
    largest_groups = [
        (label, group)
        for label, group, _ in numeric
        if largest_budget is not None
        and float(group["iteration_budget"]) == largest_budget
    ]
    largest_moving = [
        (label, group)
        for label, group in largest_groups
        if group["runs"] >= 3
        and (group["median_tail_observed_change"] or 0.0) > 0.05
        and group["runs_with_reliable_extrapolation"] >= 0.5 * group["runs"]
    ]
    historical_moving = [
        (row["label"], group)
        for row in evidence_a
        for group in row["by_iteration_budget"]
        if group["runs"] >= 3
        and (group["median_tail_observed_change"] or 0.0) > 0.05
        and float(group["iteration_budget"] or 0) != largest_budget
        and group["runs_with_detectable_trend"] >= 0.5 * group["runs"]
    ]
    if largest_moving:
        a_verdict = "成立"
    elif historical_moving:
        a_verdict = "部分成立"
    else:
        a_verdict = "否证"
    out["A"] = {
        "verdict": a_verdict,
        "supported": a_verdict == "成立",
        "supported_on_historical_budgets": bool(historical_moving),
        "largest_iteration_budget": largest_budget,
        "largest_budget_still_moving": [
            f"{label} {group['iteration_budget']} 步" for label, group in largest_moving
        ],
        "historical_budgets_still_moving": [
            f"{label} {group['iteration_budget']} 步"
            for label, group in historical_moving
        ],
        "evidence": evidence_a,
        "reading": (
            "分档判定: 「当前最大迭代预算」那一档若末段仍能观测到 >5% 相对变化且趋势"
            "可分辨, 判「成立」; 只有历史小预算档未收敛则判「部分成立」; 都不满足判「否证」。"
            "外推值是「迭代翻倍」按末段斜率线性外推的额外相对降幅"
        ),
    }

    mismatch = agg["mismatch"]
    bias = mismatch["model_minus_measured"]
    score_bias = bias["score"].get("median")
    cv_bias = bias["cv"].get("median")
    ee_bias = bias["ee"].get("median")
    decomp = mismatch["shaped_decomposition"]
    width = decomp["width_ratio"].get("median")
    energy = decomp["energy_ratio"].get("median")
    shift = decomp["centroid_shift_cells"].get("median")
    shape = decomp["shape_correlation"].get("median")
    # 系统性 = 方向一致 (过半同号) 且中位偏差量级不小。
    sign_consistency = mismatch["model_minus_measured"]["score"].get(
        "fraction_negative"
    )
    cv_consistency = mismatch["model_minus_measured"]["cv"].get("fraction_negative")
    ee_consistency = mismatch["model_minus_measured"]["ee"].get("fraction_negative")
    systematic = (
        score_bias is not None
        and abs(score_bias) > 0.05
        and sign_consistency is not None
        and sign_consistency >= 0.75
    ) or (
        (cv_bias is not None and abs(cv_bias) > 0.02)
        or (ee_bias is not None and abs(ee_bias) > 0.02)
    )
    out["B"] = {
        "supported": bool(systematic),
        "evidence": {
            "median_score_bias_model_minus_measured": score_bias,
            "fraction_of_runs_model_below_measured": sign_consistency,
            "fraction_of_runs_model_cv_below_measured": cv_consistency,
            "fraction_of_runs_model_ee_below_measured": ee_consistency,
            "median_cv_bias": cv_bias,
            "median_ee_bias": ee_bias,
            "median_iou_bias": bias["iou"].get("median"),
            "median_energy_ratio": energy,
            "median_width_ratio": width,
            "median_centroid_shift_cells": shift,
            "median_shape_correlation": shape,
            "runs_analysed": mismatch["runs_analysed"],
        },
        "reading": (
            "模型与实测的指标偏差是否系统性 (方向一致且量级远超噪声); "
            "四类分解指出差异主要落在能量 / 宽度 / 位置 / 形状哪一类"
        ),
    }

    trust = agg["trust_region"]
    out["C"] = {
        "supported": bool(
            trust.get("fitted_clamped_fraction") is not None
            and trust["fitted_clamped_fraction"] >= 0.5
        ),
        "evidence": {
            "clamped_round_fraction": trust.get("clamped_round_fraction"),
            "clamped_rounds": trust.get("clamped_rounds"),
            "rounds": trust.get("rounds"),
            "fitted_rounds": trust.get("fitted_rounds"),
            "fitted_rounds_clamped": trust.get("fitted_rounds_clamped"),
            "fitted_clamped_fraction": trust.get("fitted_clamped_fraction"),
            "zero_fit_rounds": trust.get("zero_fit_rounds"),
            "median_applied_over_limit": trust.get("applied_over_limit", {}).get(
                "median"
            ),
            "median_coeff_norm_over_limit": trust.get("coeff_norm_over_limit", {}).get(
                "median"
            ),
            "median_loss_drop_when_clamped": trust.get(
                "clamped_loss_drop_median", {}
            ).get("median"),
            "median_loss_drop_when_free": trust.get("free_loss_drop_median", {}).get(
                "median"
            ),
            "clamped_rounds_with_loss": trust.get("clamped_rounds_with_loss"),
            "free_rounds_with_loss": trust.get("free_rounds_with_loss"),
            "median_step_l2_raw_over_limit": trust.get(
                "step_l2_raw_over_limit", {}
            ).get("median"),
        },
        "reading": (
            "撞满比例 ≥50% 即视为护栏在拦; 撞满轮的损失降幅与未撞满轮对比, "
            "若撞满时损失仍在降, 说明是半径而非步数在限"
        ),
    }
    return out


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """JSON 兜底: numpy 标量 / 数组 / Path。"""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _render_summary(
    *,
    generated: str,
    roots: Sequence[str],
    analyses: Sequence[dict[str, Any]],
    agg: dict[str, Any],
    verdict_block: dict[str, Any],
    candidates: dict[str, Any],
    out_dir: Path,
) -> str:
    """生成中文摘要 markdown。"""
    hardware = [a for a in analyses if a["is_hardware"]]
    lines: list[str] = []
    lines.append("# model-in-loop 损失审计与候选相位清单")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：{generated}")
    lines.append(f"- 产物目录：`{_rel(out_dir)}`")
    lines.append(f"- 扫描根：{', '.join(f'`{root}`' for root in roots)}")
    lines.append(
        f"- 运行数：**{len(analyses)}**（其中真实硬件 {len(hardware)}，"
        f"仿真 {len(analyses) - len(hardware)}）"
    )
    lines.append("")
    lines.append("## 0. 判定摘要")
    lines.append("")
    lines.append("| 假设 | 判定 | 关键数字 |")
    lines.append("|---|---|---|")
    a = verdict_block["A"]
    a_parts = []
    for row in a["evidence"]:
        for group in row["by_iteration_budget"]:
            if group["runs"] < 3:
                continue
            a_parts.append(
                f"{row['label']} {group['iteration_budget']} 步档：末段相对变化 "
                f"{_pct(group['median_tail_observed_change'])}"
                f"（抖动 {_pct(group['median_tail_jitter'])}，t 中位 "
                f"{fmt_signed(group['median_tail_slope_t'], '.3g', MISSING)}）"
                + (
                    f"，翻倍外推 {_pct(group['median_extrapolated_extra_drop_doubling'])}"
                    if group["runs_with_reliable_extrapolation"]
                    else "，翻倍外推不可信/不可分辨"
                )
            )
    lines.append(f"| A 迭代次数不足 | {a['verdict']} | " + "；".join(a_parts) + " |")
    b = verdict_block["B"]
    b_text = (
        f"模型−实测 复合分数 {fmt_signed(b['evidence']['median_score_bias_model_minus_measured'], '.4g', MISSING)}、"
        f"CV {fmt_signed(b['evidence']['median_cv_bias'], '.4g', MISSING)}（{_pct(b['evidence'].get('fraction_of_runs_model_cv_below_measured'))} 的运行为负）、"
        f"EE {fmt_signed(b['evidence']['median_ee_bias'], '.4g', MISSING)}、"
        f"IoU {fmt_signed(b['evidence'].get('median_iou_bias'), '.4g', MISSING)};"
        f"能量比 {fmt_signed(b['evidence']['median_energy_ratio'], '.4g', MISSING)}、"
        f"宽度比 {fmt_signed(b['evidence']['median_width_ratio'], '.4g', MISSING)}、"
        f"质心偏移 {fmt_signed(b['evidence']['median_centroid_shift_cells'], '.4g', MISSING)} 格、"
        f"形状相关 {fmt_signed(b['evidence']['median_shape_correlation'], '.4g', MISSING)}"
    )
    lines.append(f"| B 模型失配 | {'成立' if b['supported'] else '否证'} | {b_text} |")
    c = verdict_block["C"]
    c_text = (
        f"真正跑 Step A 的 {c['evidence']['fitted_rounds']} 轮里 "
        f"{c['evidence']['fitted_rounds_clamped']} 轮撞满"
        f"（{_pct(c['evidence']['fitted_clamped_fraction'])}），"
        f"步长/上限 中位 {fmt_signed(c['evidence']['median_applied_over_limit'], '.4g', MISSING)}、"
        f"拟合需求/上限 中位 {fmt_signed(c['evidence'].get('median_step_l2_raw_over_limit'), '.4g', MISSING)}；"
        f"撞满时损失降幅中位 {_pct(c['evidence']['median_loss_drop_when_clamped'])}"
        f"（{c['evidence'].get('clamped_rounds_with_loss')} 轮），"
        f"未撞满对照 {c['evidence'].get('free_rounds_with_loss')} 轮"
    )
    lines.append(
        f"| C 信任域太紧 | {'成立' if c['supported'] else '否证'} | {c_text} |"
    )
    lines.append("")
    lines.append("## 1. 损失轨迹（假设 A）")
    lines.append("")
    lines.append(
        "| Step | 迭代预算 | 运行数 | 总相对降幅（中位） | 末段 20% 实测相对变化（中位） "
        "| 末段 20% 每步相对变化 | 末段 t | 翻倍外推 | 末段仍在动(>5%) | 发散 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, label in (("step_a", "Step A"), ("step_b", "Step B")):
        for group in agg[key]["by_iteration_budget"]:
            lines.append(
                f"| {label} | {group['iteration_budget']} | {group['runs_with_trace']} | "
                f"{_pct(group['relative_drop'].get('median'))} | "
                f"{_pct(group['tail_20pct_observed_change'].get('median'))} | "
                f"{fmt_signed(group['tail_20pct_slope_relative'].get('median'), '.4g', MISSING)} | "
                f"{fmt_signed(group['tail_20pct_slope_t'].get('median'), '.3g', MISSING)} | "
                f"{_pct(group['tail_20pct_extrapolated_extra_drop'].get('median'))} | "
                f"{group['runs_still_moving']}/{group['runs_with_trace']} | "
                f"{group['runs_diverging']} |"
            )
    lines.append("")
    lines.append(
        "「每步相对变化」= 末段线性斜率 ÷ 末段均值；「翻倍外推」= 按该斜率再多跑"
        "同样步数能带来的额外相对降幅，**只在末段斜率 |t|>2 且外推值 <100% 时"
        "才计入**（否则该运行在这一列不计），逐运行的原始值与 `extrapolation_note` "
        "留在 JSON 里。**按迭代预算分行**是必须的：80 步与 600 步的 Step B 收敛"
        "程度差一个量级，混在一起做中位数会把两者都读错。"
    )
    lines.append("")
    lines.append("## 2. 信任域（假设 C）")
    lines.append("")
    trust = agg["trust_region"]
    lines.append(
        f"- 有逐轮记录的运行 {trust['runs']} 个，共 {trust['rounds']} 轮；"
        f"其中 {trust['clamped_rounds']} 轮触发夹取"
        f"（{_pct(trust['clamped_round_fraction'])}），涉及 {trust['runs_with_any_clamp']} 个运行。"
    )
    lines.append(
        f"- **真正跑了 Step A 的轮次（`step_a_iterations > 0`）**：{trust['fitted_rounds']} 轮，"
        f"其中 {trust['fitted_rounds_clamped']} 轮撞满信任域"
        f"（{_pct(trust['fitted_clamped_fraction'])}）；另有 {trust['zero_fit_rounds']} 轮是"
        "零拟合对照（`step_a_iterations=0`，损失前后恒等，不能当作「未撞满」的反例）。"
    )
    lines.append(
        f"- 实际施加步长 / 上限：中位 {fmt_signed(trust['applied_over_limit'].get('median'), '.4g', MISSING)}、"
        f"最大 {fmt_signed(trust['applied_over_limit'].get('max'), '.4g', MISSING)}。"
    )
    lines.append(
        f"- 提交系数 L2 / 上限：中位 {fmt_signed(trust['coeff_norm_over_limit'].get('median'), '.4g', MISSING)}、"
        f"最大 {fmt_signed(trust['coeff_norm_over_limit'].get('max'), '.4g', MISSING)}。"
    )
    lines.append(
        f"- 损失相对降幅（只统计跑了 Step A 的轮次）：撞满轮中位 "
        f"{_pct(trust['clamped_loss_drop_median'].get('median'))}"
        f"（{trust.get('clamped_rounds_with_loss')} 轮），"
        f"未撞满轮 {_pct(trust['free_loss_drop_median'].get('median'))}"
        f"（{trust.get('free_rounds_with_loss')} 轮）—— **没有可用的未撞满对照**，"
        "所以「松一点会不会更好」只能靠单轮实验回答，历史数据里不存在这个反事实。"
    )
    lines.append(
        f"- 拟合**想要**的步长 / 上限：中位 "
        f"{fmt_signed(trust['step_l2_raw_over_limit'].get('median'), '.4g', MISSING)}、最大 "
        f"{fmt_signed(trust['step_l2_raw_over_limit'].get('max'), '.4g', MISSING)} —— 这就是护栏把一步"
        "截成了几分之一。"
    )
    lines.append("")
    lines.append("## 3. 模型失配（假设 B）")
    lines.append("")
    mismatch = agg["mismatch"]
    lines.append(
        f"- 可复算配对的运行：**{mismatch['runs_analysed']}**（其余运行因系数回滚 / 多轮"
        f"落盘只对应末轮而无法复算，已在 JSON 里逐条标注原因）。"
    )
    lines.append("")
    lines.append("| 指标 | 模型 − 实测（中位） | n | 模型偏低的运行占比 |")
    lines.append("|---|---|---|---|")
    for key, label in (
        ("score", "复合分数"),
        ("cv", "CV"),
        ("ee", "EE"),
        ("iou", "IoU"),
    ):
        block = mismatch["model_minus_measured"][key]
        fraction = block.get("fraction_negative")
        lines.append(
            f"| {label} | {fmt_signed(block.get('median'), '.4g', MISSING)} | {block.get('n', 0)} | "
            f"{_pct(fraction) if fraction is not None else MISSING} |"
        )
    lines.append("")
    lines.append("### 代表性运行（逐 run 的配对，供核对）")
    lines.append("")
    representative = [
        analysis
        for analysis in reversed(analyses)
        if analysis["is_hardware"] and analysis["mismatch"].get("available")
    ][:6]
    lines.append(
        "| 运行 | 轮 | 模型−实测 分数 | 模型−实测 CV | 模型−实测 EE | 能量比 | 宽度比 "
        "| 质心偏移(格) | 形状相关 | 峰值归一化 MSE |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for analysis in representative:
        mismatch_block = analysis["mismatch"]
        metrics = (mismatch_block.get("shaped_metrics") or {}).get(
            "model_minus_measured"
        ) or {}
        decomposition = mismatch_block.get("shaped_decomposition") or {}
        lines.append(
            f"| `{analysis['inventory']['stamp']}` | "
            f"{max(mismatch_block.get('verified_rounds') or [0])} | "
            f"{fmt_signed(metrics.get('score'), '.4g', MISSING)} | {fmt_signed(metrics.get('cv'), '.4g', MISSING)} | "
            f"{fmt_signed(metrics.get('ee'), '.4g', MISSING)} | {fmt_signed(decomposition.get('energy_ratio'), '.4g', MISSING)} | "
            f"{fmt_signed(decomposition.get('width_ratio'), '.4g', MISSING)} | "
            f"{fmt_signed(decomposition.get('centroid_shift_cells'), '.4g', MISSING)} | "
            f"{fmt_signed(decomposition.get('shape_correlation'), '.4g', MISSING)} | "
            f"{fmt_signed(decomposition.get('mse_peak_normalised'), '.4g', MISSING)} |"
        )
    lines.append("")
    lines.append("### 差异的四类分解（整形探针，相机视场）")
    lines.append("")
    lines.append("| 类别 | 量 | 中位 | n |")
    lines.append("|---|---|---|---|")
    decomp = mismatch["shaped_decomposition"]
    rows = (
        ("能量尺度", "预测/实测总能量", "energy_ratio"),
        ("宽度（模糊）", "预测/实测二阶矩半径", "width_ratio"),
        ("位置（平移）", "质心偏移（模型格）", "centroid_shift_cells"),
        ("形状", "去尺度去平移后相关", "shape_correlation"),
        ("总体", "原始 Pearson r", "correlation_raw"),
        ("总体", "峰值归一化 MSE", "mse_peak_normalised"),
    )
    for label, quantity, key in rows:
        block = decomp[key]
        lines.append(
            f"| {label} | {quantity} | {fmt_signed(block.get('median'), '.4g', MISSING)} | {block.get('n', 0)} |"
        )
    lines.append("")
    lines.append("### 分日期（= 分实验会话）的失配方向")
    lines.append("")
    lines.append(
        "| 日期 | 运行数 | 相机像元/模型格 | 能量比 | 宽度比 | 质心偏移(格) "
        "| 形状相关 | 原始 r | 模型−实测 分数 | 模型−实测 CV |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for group in mismatch.get("by_date", []):
        lines.append(
            f"| {group['date']} | {group['runs']} | "
            f"{fmt_signed(group['camera_px_per_model_px'], '.4g', MISSING)} | "
            f"{fmt_signed(group['energy_ratio'], '.4g', MISSING)} | {fmt_signed(group['width_ratio'], '.4g', MISSING)} | "
            f"{fmt_signed(group['centroid_shift_cells'], '.4g', MISSING)} | "
            f"{fmt_signed(group['shape_correlation'], '.4g', MISSING)} | {fmt_signed(group['correlation_raw'], '.4g', MISSING)} | "
            f"{fmt_signed(group['score_bias'], '.4g', MISSING)} | {fmt_signed(group['cv_bias'], '.4g', MISSING)} |"
        )
    lines.append("")
    lines.append(
        "分日期不是为了好看：一整天共用一套光路与一套几何标定，而**宽度比在"
        "不同会话里方向会翻**（有的会话模型偏宽、有的偏窄），把方向相反的会话"
        "混成一个大中位数会把这一层完全掩盖。所以「模型该往哪个方向改」只能"
        "按会话读，不能按全局中位数读。"
    )
    lines.append("")
    lines.append("### 拟合增益（验证集探针，拟合系数 vs c=0）")
    lines.append("")
    gain = mismatch["fit_gain_holdout"]
    lines.append("| 量 | 中位 | n |")
    lines.append("|---|---|---|")
    for key, label in (
        ("loss_fitted", "拟合系数 峰值归一化 MSE"),
        ("loss_zero", "c=0 峰值归一化 MSE"),
        ("loss_improvement_relative", "相对改善"),
        ("correlation_fitted", "拟合系数 相关"),
        ("correlation_zero", "c=0 相关"),
        ("correlation_gain", "相关增益"),
    ):
        lines.append(
            f"| {label} | {fmt_signed(gain[key].get('median'), '.4g', MISSING)} | {gain[key].get('n', 0)} |"
        )
    lines.append("")
    scores = agg["scores"]
    lines.append(
        f"平场复合分数中位 {fmt_signed(scores['flat_score'].get('median'), '.4g', MISSING)}，"
        f"最后一轮**实测**复合分数中位 {fmt_signed(scores['shaped_score'].get('median'), '.4g', MISSING)}，"
        f"差中位 {fmt_signed(scores['score_delta'].get('median'), '.4g', MISSING)}；{scores['runs']} 个硬件运行里"
        f"有 **{scores['runs_beating_flat']}** 个的最后一轮实测分数高于平场。"
        "注意这是**该轮的实测值**，不是「提交结果」——"
        "是否真的提交由对比择优与验收另行判定（见各 run 的 `status`）。"
    )
    status_counts = scores.get("status_counts") or {}
    if status_counts:
        lines.append(
            "- 运行状态分布："
            + "、".join(
                f"`{key}` {value}" for key, value in sorted(status_counts.items())
            )
            + "。"
        )
    lines.append("")
    lines.append("## 4. 候选相位清单")
    lines.append("")
    if candidates.get("candidates"):
        summary = candidates["distance_summary"]
        counts = candidates.get("region_counts") or {}
        lines.append(
            f"- 合格池 {candidates['pool_size']} 条（region 分布 "
            f"{', '.join(f'{k}:{v}' for k, v in sorted(counts.items()))}）；"
            f"主组 region={candidates['region']} 选中 **{len(candidates['candidates'])}** 条"
            f"（瞳面膜 {candidates['pupil_mask_px']} 像素）。"
        )
        lines.append(
            f"- 主组两两 L2 距离（去 piston）：最小 {summary['min']:.4f} rad、"
            f"中位 {summary['median']:.4f} rad、最大 {summary['max']:.4f} rad。"
        )
        for group in candidates.get("secondary_groups") or []:
            lines.append(
                f"- 副组 region={group['region']}：合格 {group['pool_size']} 条、"
                f"选中 {len(group['candidates'])} 条，距离 "
                f"{group['distance_summary']['min']:.4f}–"
                f"{group['distance_summary']['max']:.4f} rad。"
                "不同 region 的相位长度不同，只能与同组比较。"
            )
        lines.append(
            "- 主组按「合格条目最多」自动选；若当前光路用 region=128，请加 "
            "`--region 128` 只取副组那一批（相位数组已一并落盘，文件名前缀 `r128`）。"
        )
        cv_values = [item.get("cv_after") for item in candidates["candidates"]]
        ee_values = [item.get("ee_after") for item in candidates["candidates"]]
        cv_median = _stat(cv_values).get("median")
        ee_median = _stat(ee_values).get("median")
        if cv_median is not None and ee_median is not None:
            lines.append(
                f"- ⚠️ 这些条目的改善**主要来自能量集中**（EE 中位 {fmt_signed(ee_median, '.4g', MISSING)}，"
                f"而 CV 中位 {fmt_signed(cv_median, '.4g', MISSING)} 仍然很高）：复合分数里 EE 权重 0.6、"
                "均匀度项 0.4/(1+CV) 上限只有 0.4，所以「分数上升」并不等于「更均匀」。"
                "用它们做探测训练集是可以的，但别当成「均匀整形成功」的样本。"
            )
        lines.append("")
        lines.append(
            "| # | 运行 | 轮 | 分数 前→后 | 相对改善 | 模型−实测 分数 | 相位文件 |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for index, item in enumerate(candidates["candidates"]):
            delta = None
            metrics = item.get("shaped_metrics") or {}
            if isinstance(metrics.get("model_minus_measured"), dict):
                delta = metrics["model_minus_measured"].get("score")
            lines.append(
                f"| {index + 1} | `{item['stamp']}` | {item['round']} | "
                f"{item['score_before']:.4f} → {item['score_after']:.4f} | "
                f"{_pct(item['relative_improvement'])} | {fmt_signed(delta, '.4g', MISSING)} | "
                f"`{item.get('phase_file', MISSING)}` |"
            )
    else:
        lines.append("- 没有合格候选。")
    lines.append("")
    lines.append("## 5. 下一步最该动的旋钮（按预计收益排序）")
    lines.append("")
    lines.extend(_recommendations(verdict_block, agg))
    lines.append("")
    lines.append("## 6. 口径与局限")
    lines.append("")
    lines.append(
        "- 预测一律由落盘 `fitted_coefficients.npy` + `source_amplitude.npy` + "
        "`run_config.json` 离线复算，并用落盘 `prediction` 逐像素校验；"
        f"校验阈 {RECOMPUTE_TOL:g}（相对该轮预测峰值）。"
    )
    lines.append(
        "- 相机帧取自 `probes/*.npz` 的 `frame`（已是 `prepare_frame` 之后），不再二次去本底。"
    )
    lines.append(
        "- 相机↔模型网格换算复用 MIL 的 `_to_model_grid`；旋转象限取硬件路径写死值 1；"
        "用落盘 `model_grid` 校验后相对 L1 误差约 1e-11。"
    )
    lines.append(
        "- 仿真运行（`cam_type=sim`）纳入清单与轨迹统计，但**不进入**失配统计 —— "
        "仿真里没有模型↔实测失配可言。"
    )
    lines.append("- 缺失字段一律写「未记录」，不填 0。")
    lines.append("")
    return "\n".join(lines)


def _recommendations(verdict_block: dict[str, Any], agg: dict[str, Any]) -> list[str]:
    """按判定给出旋钮排序 (含预计收益量级)。"""
    out: list[str] = []
    rank = 1
    trust = verdict_block["C"]
    mismatch = verdict_block["B"]
    iters = verdict_block["A"]
    if trust["supported"]:
        over = trust["evidence"]["median_applied_over_limit"]
        out.append(
            f"{rank}. **放开 / 自适应信任域半径**（真正跑 Step A 的 "
            f"{trust['evidence']['fitted_rounds']} 轮里 "
            f"{_pct(trust['evidence']['fitted_clamped_fraction'])} 撞满；"
            f"步长/上限中位 {fmt_signed(over, '.4g', MISSING)}、拟合需求/上限中位 "
            f"{fmt_signed(trust['evidence'].get('median_step_l2_raw_over_limit'), '.4g', MISSING)}）。"
            f"撞满轮的损失仍在降（中位 "
            f"{_pct(trust['evidence']['median_loss_drop_when_clamped'])}），"
            "却全部被硬截在半径上——这是唯一一处「损失还在降但被截断」的位置。"
            "注意历史数据里**没有**未撞满的对照（其余轮次是零拟合对照），"
            "所以收益量级只能由单轮实验给出，不能从历史估计：建议 "
            "`--adaptive-trust-region` + 单轮 + 保留落盘 `step_l2_raw`。"
        )
        rank += 1
    if mismatch["supported"]:
        evidence = mismatch["evidence"]
        dominant = _dominant_term(evidence)
        per_date = _dominant_by_date(agg["mismatch"])
        date_text = "、".join(f"{date}→{term}" for date, term in per_date)
        cv_consistency = evidence.get("fraction_of_runs_model_cv_below_measured")
        out.append(
            f"{rank}. **按会话修模型，不要一次改一个全局项**（全局中位偏差："
            f"模型−实测 复合分数 {fmt_signed(evidence['median_score_bias_model_minus_measured'], '.4g', MISSING)}、"
            f"CV {fmt_signed(evidence['median_cv_bias'], '.4g', MISSING)}、EE {fmt_signed(evidence['median_ee_bias'], '.4g', MISSING)}；"
            f"能量比 {fmt_signed(evidence['median_energy_ratio'], '.4g', MISSING)}、宽度比 "
            f"{fmt_signed(evidence['median_width_ratio'], '.4g', MISSING)}、质心偏移 "
            f"{fmt_signed(evidence['median_centroid_shift_cells'], '.4g', MISSING)} 格、去尺度去平移后相关 "
            f"{fmt_signed(evidence['median_shape_correlation'], '.4g', MISSING)}）。全局最突出的一类是"
            f"「{dominant}」，但**分会话看主导类别会变**（{date_text}）——"
            "所以先按会话用候选探针复测几何（尤其 `camera_px_per_model_px` 与离焦），"
            "再决定改哪一项。"
            f"CV 偏差的方向一致性 {_pct(cv_consistency)}（模型系统性把光斑预测得"
            "更均匀），而复合分数的偏差却接近 0 —— 两项在目标函数里互相抵消，"
            "所以优化器几乎无从下手；这一层不修，Step B 的目标函数就是错的。"
        )
        rank += 1
    if iters["supported_on_historical_budgets"]:
        moving = [
            (row["label"], group)
            for row in iters["evidence"]
            for group in row["by_iteration_budget"]
            if group["runs"] >= 3
            and (group["median_tail_observed_change"] or 0.0) > 0.05
            and group["runs_with_detectable_trend"] >= 0.5 * group["runs"]
        ]
        detail = "; ".join(
            f"{label} 的 {group['iteration_budget']} 步档末段仍有 "
            f"{_pct(group['median_tail_observed_change'])} 的实测降幅、"
            f"翻倍外推 {_pct(group['median_extrapolated_extra_drop_doubling'])}"
            for label, group in moving
        )
        out.append(
            f"{rank}. **加迭代只对未收敛的那一档有意义**（{detail}）。"
            "这是**分档**结论：当前最大预算那一档已收敛（末段变化 ≤5% 或趋势不可分辨），"
            "加迭代只花钱不改善；把预算整体翻倍会把两档一起放大，收益被已收敛的档摊薄。"
        )
        rank += 1
    else:
        extrapolated = [
            row["median_extrapolated_extra_drop_doubling"] for row in iters["evidence"]
        ]
        worst = max([v for v in extrapolated if v is not None] or [0.0])
        detectable = sum(row["runs_with_detectable_trend"] for row in iters["evidence"])
        total = sum(row["runs_with_trace"] for row in iters["evidence"])
        out.append(
            f"{rank}. **不要先加迭代**。{total} 条轨迹里只有 {detectable} 条在末段"
            f"还有能盖过抖动的下降趋势；可外推的那几条给出的额外降幅最大 "
            f"{_pct(worst)}。加一倍迭代在噪声里看不出来。"
        )
        rank += 1
    out.append(
        f"{rank}. **把候选相位清单当探测训练集先在实验系统上测**（见第 4 节）："
        "它们带完整的实测 / 预测配对与四类分解，能直接用来判断"
        "「模型错在能量、宽度、位置还是形状」，而不用再猜。"
    )
    return out


def _dominant_term(evidence: dict[str, Any]) -> str:
    """按离「无失配」的距离挑出最突出的那一类。"""
    candidates = {
        "能量尺度": abs((evidence.get("median_energy_ratio") or 1.0) - 1.0),
        "宽度（模糊）": abs((evidence.get("median_width_ratio") or 1.0) - 1.0),
        "位置（平移）": min(
            (evidence.get("median_centroid_shift_cells") or 0.0) / 5.0, 9.9
        ),
        "形状": 1.0 - (evidence.get("median_shape_correlation") or 1.0),
    }
    return max(candidates, key=lambda key: candidates[key])


def _dominant_by_date(mismatch: dict[str, Any]) -> list[tuple[str, str]]:
    """逐会话挑主导失配类别 (会话内的方向可以与全局中位数相反)。"""
    out: list[tuple[str, str]] = []
    for group in mismatch.get("by_date", []):
        evidence = {
            "median_energy_ratio": group.get("energy_ratio"),
            "median_width_ratio": group.get("width_ratio"),
            "median_centroid_shift_cells": group.get("centroid_shift_cells"),
            "median_shape_correlation": group.get("shape_correlation"),
        }
        out.append((group["date"], _dominant_term(evidence)))
    return out


def _pct(value: Any) -> str:
    """百分比格式化。"""
    if value is None:
        return MISSING
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return MISSING
    return f"{number * 100:.2f}%"


def _imshow_spot(axis: Any, image: np.ndarray, title: str) -> None:
    """强度图渲染 (按 99.9% 分位定标, 与报告脚本 ``_imshow`` 同口径)。"""
    array = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    peak = float(np.percentile(array, 99.9)) if array.size else 1.0
    axis.imshow(array, cmap="inferno", vmin=0.0, vmax=max(peak, 1e-12))
    axis.set_title(title, fontsize=8)


def _plot_candidates(chosen: Sequence[dict[str, Any]], path: Path) -> Path | None:
    """候选三联缩略图: 相位 (包裹展示) / 实测光斑 / 模型预测光斑。"""
    if not chosen:
        return None
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    rows = len(chosen)
    figure, axes = plt.subplots(rows, 3, figsize=(9.5, 2.6 * rows))
    if rows == 1:
        axes = np.atleast_2d(axes)
    for index, item in enumerate(chosen):
        phase = np.asarray(item["phase"], dtype=np.float64)
        wrapped = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        axes[index, 0].imshow(wrapped, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[index, 0].set_title(
            f"#{index + 1} {item['stamp']} r{item['round']} 相位", fontsize=8
        )
        probe = item.get("shaped_probe")
        if probe is not None:
            _imshow_spot(axes[index, 1], probe.frame, "实测光斑")
            predicted = item.get("predicted_camera")
            if predicted is not None:
                _imshow_spot(axes[index, 2], predicted, "模型预测")
        for axis in axes[index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "候选相位 / 实测 / 模型预测（每格按自身 99.9% 分位定标，仅比形状）", fontsize=11
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """跑全量审计并落盘。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        default=list(DEFAULT_ROOTS),
        help="扫描根目录（每个下面按 run_config.json 找运行）。",
    )
    parser.add_argument(
        "--out-dir",
        default="logs/mil_loss_audit",
        help="产物根目录（会在其下建时间戳子目录）。",
    )
    parser.add_argument(
        "--stamp",
        default=None,
        help="产物子目录名（默认当前 UTC 时间）。",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=10,
        help="候选相位条数。",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="候选门槛：实测复合分数相对基线的相对改善下限（默认 >0）。",
    )
    parser.add_argument(
        "--region",
        type=int,
        default=None,
        help=(
            "候选相位的瞳面网格边长；默认取合格池里出现最多的那个。"
            "不同 region 的相位无法直接比距离，跨 region 的合格候选会在 "
            "`other_region_pool` 里列出而不参与挑选。"
        ),
    )
    parser.add_argument(
        "--no-mismatch",
        action="store_true",
        help="跳过模型失配复算（不加载正向模型，纯轨迹 / 信任域审计）。",
    )
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = out_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)

    runs: dict[str, RunData] = {}
    for campaign, run_dir in discover_runs(args.roots):
        try:
            runs[run_dir.as_posix()] = load_run(campaign, run_dir)
        except Exception as exc:  # pragma: no cover - 防御
            logger.warning("读取运行失败 {}: {}", run_dir, exc)
    print(f"discovered {len(runs)} run dirs")

    analyses: list[dict[str, Any]] = []
    for key, run in runs.items():
        entry: dict[str, Any] = {
            "path": run.path.as_posix(),
            "is_hardware": run.is_hardware,
            "inventory": inventory(run),
            "trust_region": trust_region_diagnostics(run),
        }
        if args.no_mismatch:
            entry["mismatch"] = {"available": False, "note": "本次运行跳过失配复算"}
        else:
            try:
                entry["mismatch"] = analyse_mismatch(run)
            except Exception as exc:  # pragma: no cover - 防御
                logger.warning("{}: 失配复算失败 {}", run.stamp, exc)
                entry["mismatch"] = {"available": False, "note": f"复算失败：{exc}"}
        analyses.append(entry)
        print(
            f"  {run.campaign[:16]:16s} {run.stamp:24s} "
            f"traceA={len(run.trace_a.losses):4d} "
            f"traceB={len(run.trace_b.losses):4d} "
            f"mismatch={'yes' if entry['mismatch'].get('available') else 'no'}"
        )

    agg = aggregate(analyses)
    verdict_block = verdicts(agg)
    candidate_block = collect_candidates(
        analyses,
        runs,
        want=args.candidates,
        min_improvement=args.min_improvement,
        region=args.region,
    )

    # 相位数组另存 + 预测画面 (缩略图用), 供后续实验直接用
    primary_items = candidate_block.get("candidates", [])
    _materialise_candidates(primary_items, candidates_dir, "c")
    for group in candidate_block.get("secondary_groups", []) or []:
        _materialise_candidates(
            group.get("candidates", []), candidates_dir, f"r{group['region']}"
        )

    grid_path = _plot_candidates(primary_items, out_dir / "candidates_grid.png")

    serialisable_candidates = [_strip_candidate(item) for item in primary_items]
    serialisable_secondary = [
        {
            "region": group["region"],
            "pupil_mask_px": group["pupil_mask_px"],
            "pool_size": group["pool_size"],
            "distance_summary": group["distance_summary"],
            "distance_matrix": group["distance_matrix"],
            "correlation_matrix": group["correlation_matrix"],
            "candidates": [_strip_candidate(item) for item in group["candidates"]],
        }
        for group in candidate_block.get("secondary_groups", []) or []
    ]

    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    audit = {
        "generated_utc": generated,
        "roots": list(args.roots),
        "recompute_tolerance": RECOMPUTE_TOL,
        "aggregate": agg,
        "verdicts": verdict_block,
        "runs": analyses,
    }
    (out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    candidates_payload = {
        "generated_utc": generated,
        "selection_rule": (
            "该轮 accepted=True 且实测复合分数相对基线改善 ≥ "
            f"{args.min_improvement:g}，且该轮模型预测可复算（相对偏差 ≤ "
            f"{RECOMPUTE_TOL:g}）；相位取该轮实际下发的整形探针（无则取 best_phase.npy，"
            "全零视为保留平场不收）；再按去 piston 的 L2 距离做最远点采样保证多样性"
        ),
        "region": candidate_block.get("region"),
        "region_counts": candidate_block.get("region_counts"),
        "pupil_mask_px": candidate_block.get("pupil_mask_px"),
        "pool_size": candidate_block.get("pool_size"),
        "note": candidate_block.get("note"),
        "distance_summary": candidate_block.get("distance_summary"),
        "distance_matrix": candidate_block.get("distance_matrix"),
        "correlation_matrix": candidate_block.get("correlation_matrix"),
        "candidates": serialisable_candidates,
        "secondary_groups": serialisable_secondary,
    }
    (out_dir / "candidates.json").write_text(
        json.dumps(
            candidates_payload, indent=2, ensure_ascii=False, default=_json_default
        ),
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(
        _render_summary(
            generated=generated,
            roots=args.roots,
            analyses=analyses,
            agg=agg,
            verdict_block=verdict_block,
            candidates=candidate_block,
            out_dir=out_dir,
        ),
        encoding="utf-8",
    )
    if grid_path is not None:
        print(f"wrote {grid_path}")
    print(f"wrote {out_dir / 'audit.json'}")
    print(f"wrote {out_dir / 'candidates.json'}")
    print(f"wrote {out_dir / 'summary.md'}")
    print()
    print(
        "verdicts:",
        json.dumps(verdict_block, indent=2, ensure_ascii=False, default=_json_default),
    )
    return 0


def _materialise_candidates(
    items: Sequence[dict[str, Any]], directory: Path, prefix: str
) -> None:
    """把候选相位存成 ``.npy`` (弧度, raw 未包裹) 并补上预测画面 (缩略图用)。"""
    for index, item in enumerate(items):
        name = f"{prefix}{index + 1:02d}_{item['stamp']}_r{item['round']}.npy"
        np.save(directory / name, np.asarray(item["phase"], dtype=np.float64))
        item["phase_file"] = f"candidates/{name}"
        if item.get("shaped_probe") is None:
            continue
        if item.get("predicted_camera") is None and item.get("mismatch", {}).get(
            "available"
        ):
            try:
                item["predicted_camera"] = _predicted_camera_for(item)
            except Exception as exc:  # pragma: no cover - 防御
                logger.warning("{}: 预测画面复算失败 {}", item["stamp"], exc)


def _strip_candidate(item: dict[str, Any]) -> dict[str, Any]:
    """候选条目里去掉不可 JSON 序列化 / 体积过大的字段。"""
    drop = {"phase", "vector", "run", "shaped_probe", "predicted_camera", "mismatch"}
    return {key: value for key, value in item.items() if key not in drop}


def _predicted_camera_for(item: dict[str, Any]) -> np.ndarray | None:
    """为缩略图重算「预测的相机窗口画面」(缩略图专用, 不进统计)。"""
    run: RunData | None = item.get("run")
    probe: Probe | None = item.get("shaped_probe")
    if (
        run is None
        or probe is None
        or run.coefficients is None
        or run.source_amplitude is None
    ):
        return None
    config = _config_for(run)
    geometry = run.geometry
    far_field_size = int(geometry.get("far_field_size", 0) or 0)
    scale = _as_float(geometry.get("camera_px_per_model_px"))
    if config is None or not far_field_size or not scale:
        return None
    from scripts.slm.generate_model_in_loop_record_report import _model_grid_to_camera

    flats = [p for p in run.probes if p.split == "flat"]
    if not flats:
        return None
    centre = zero_order_center(flats[0].frame, refine=False)
    rotation = 0 if not run.is_hardware else 1
    flat_grid = _to_model_grid(flats[0].frame, centre, far_field_size, scale, rotation)
    optimizer = _make_optimizer(config, None, run.source_amplitude)
    predicted = optimizer.forward_intensity(
        run.coefficients, probe.phase, run.source_amplitude
    )
    gain, leakage = _gain_leakage(run)
    if gain is not None and leakage is not None:
        predicted = gain * predicted + leakage * flat_grid
    return _model_grid_to_camera(
        np.asarray(predicted, dtype=np.float64),
        centre,
        far_field_size=far_field_size,
        cell_scale=scale,
        cam_size=int(probe.frame.shape[0]),
        rotation_quarters=rotation,
    )


if __name__ == "__main__":
    raise SystemExit(main())
