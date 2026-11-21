"""SLM transfer fit 阶段 Zernike 系数更新方向一致性诊断 (纯离线)。

本脚本只做**观察性**分析：基于 ``data/debug/slm_transfer/<run>/fit/iterations.pkl``
中真实记录的参数轨迹，测量 ``residual_noll_*`` 系数的相邻步更新方向一致性、步长
与损失下降的关系，据此评估「能否调整 fit stage 的 residual lr」。

重要口径
--------
capture 目录没有 ``sample_*.npz``（``save_dataset=False``），无法用真实 forward
目标函数重跑反事实。因此任何「放大 lr 会怎样」的结论都是**一阶线性代理**
（step ∝ lr，方向不变），不是真实重跑验证。``recommendation`` 字段只给灰度建议，
不直接改默认值。

示例
----
::

    .venv\\Scripts\\python.exe scripts\\slm\\diagnose_zernike_direction_lr.py \\
        --data-dir data/debug/slm_transfer \\
        --out data/debug/slm_transfer/zernike_direction_lr.json \\
        --figure data/debug/slm_transfer/zernike_direction_lr.png
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

# 无显示环境（CI / 后台批跑）下用 Agg 后端，必须在 pyplot 之前设置。
matplotlib.use("Agg")

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.stats import spearmanr

ITERATIONS_FILENAME = "iterations.pkl"
NESTED_KEY = "residual_noll_radians"
FLAT_PREFIX = "residual_noll_"
EPS = 1e-15

#: 常见中文字体，按优先级探测；命中后用于渲染图内中文，避免出现方块（tofu）。
_CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Microsoft JhengHei",
    "STSong",
    "KaiTi",
    "FangSong",
    "SimSun",
)


def _resolve_cjk_font() -> str | None:
    """返回一个可用的中文字体名，找不到则返回 ``None``。"""
    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            return name
    return None


_CJK_FONT = _resolve_cjk_font()
if _CJK_FONT is not None:
    matplotlib.rcParams["font.sans-serif"] = [_CJK_FONT, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

#: 有中文字体时图内用中文，否则退回英文，保证任意环境都无方块。
_LABELS: dict[str, str] = (
    {
        "cos_title": "Zernike 更新方向一致性（逐迭代）",
        "iter": "迭代",
        "cos": "相邻步余弦相似度",
        "median": "中位数",
        "scatter_title": "方向一致步：步长 vs 损失下降",
        "step": "步长 ‖dZ‖",
        "lossdrop": "该步损失下降",
        "trend": "线性趋势",
    }
    if _CJK_FONT is not None
    else {
        "cos_title": "Zernike update direction consistency (per iteration)",
        "iter": "iteration",
        "cos": "cosine between consecutive steps",
        "median": "median",
        "scatter_title": "Consistent steps: step norm vs loss drop",
        "step": "step norm |dZ|",
        "lossdrop": "loss drop at step",
        "trend": "linear trend",
    }
)


def extract_residual(entry: dict) -> np.ndarray:
    """从一条历史 entry 中按 Noll 序号数值排序展平 residual Zernike 系数。

    优先采用新版 debug_weights 里的展平键 ``residual_noll_<noll>``；若不存在，
    退回旧版 ``parameters['residual_noll_radians']`` 的 ``{noll: float}`` 字典。

    Parameters
    ----------
    entry
        ``iterations.pkl`` 中的一个 dict，含 ``parameters`` 字段。

    Returns
    -------
    np.ndarray
        按 Noll 序号升序排列的 float 向量，形状 ``(D,)``。

    Raises
    ------
    ValueError
        ``parameters`` 缺失，或两种来源都找不到任何 residual_noll 系数。
    """
    parameters = entry.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("entry 缺少 'parameters' dict，无法提取 residual Zernike 系数")

    flat: dict[int, float] = {}
    for key, value in parameters.items():
        if not isinstance(key, str) or not key.startswith(FLAT_PREFIX):
            continue
        if key == NESTED_KEY:
            continue
        try:
            noll = int(key[len(FLAT_PREFIX) :])
        except ValueError:
            continue
        flat[noll] = float(value)
    if flat:
        return np.array([flat[k] for k in sorted(flat)], dtype=float)

    nested = parameters.get(NESTED_KEY)
    if isinstance(nested, dict):
        nested_values: dict[int, float] = {}
        for key, value in nested.items():
            try:
                noll = int(key)
            except (TypeError, ValueError):
                continue
            nested_values[noll] = float(value)
        if nested_values:
            return np.array([nested_values[k] for k in sorted(nested_values)], dtype=float)

    raise ValueError(
        f"entry['parameters'] 中既无 '{FLAT_PREFIX}*' 也无 '{NESTED_KEY}' residual Zernike 系数"
    )


def direction_cosines(dZ: np.ndarray) -> np.ndarray:
    """计算相邻增量向量的逐点余弦相似度。

    Parameters
    ----------
    dZ
        增量矩阵，形状 ``(N-1, D)``。

    Returns
    -------
    np.ndarray
        形状 ``(N-2,)``，``cos[i] = cos(dZ[i], dZ[i+1])``。零向量用 ``1e-15``
        保护分母，避免除零。
    """
    arr = np.asarray(dZ, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"dZ 必须是二维 (N-1, D) 数组，实际 shape={arr.shape}")
    if arr.shape[0] < 2:
        return np.array([], dtype=float)
    left = arr[:-1]
    right = arr[1:]
    denom = np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), EPS)
    return np.sum(left * right, axis=1) / denom


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """尾部滑动平均，输出与输入等长且索引对齐。"""
    out = np.empty_like(values)
    for i in range(values.size):
        lo = max(0, i - window + 1)
        out[i] = values[lo : i + 1].mean()
    return out


def _empty_analysis() -> dict:
    """迭代数不足 2 时的空指标字典（保持键集完整，便于汇总）。"""
    return {
        "n_iters": 0,
        "loss_first": None,
        "loss_last": None,
        "cos_mean": None,
        "cos_last10": None,
        "step_norm_mean": None,
        "step_norm_std": None,
        "loss_drop_mean": None,
        "consistent_active_frac": 0.0,
        "noise_floor_from": None,
        "cos_values": [],
        "consistent_step_norms": [],
        "consistent_loss_drops": [],
    }


def analyze_trajectory(history: list[dict]) -> dict:
    """分析单条 iterations.pkl 历史，返回该轨迹的方向一致性指标。

    ``cos[i]`` 比较 step i 与 step i+1 的方向；其对应的「该步」取 step i+1，
    即 ``step_norm[i+1]`` 与 ``loss_drop[i+1]``（后者为 loss[i+1] → loss[i+2]
    的下降）。``noise_floor_from`` 为平滑余弦首次连续 3 步 < 0.7 的迭代索引。
    """
    n = len(history)
    if n < 2:
        return _empty_analysis()

    Z = np.vstack([extract_residual(entry) for entry in history])
    losses = np.array([float(entry["loss"]) for entry in history], dtype=float)
    dZ = np.diff(Z, axis=0)  # (N-1, D), step k = Z[k+1] - Z[k]
    cos = direction_cosines(dZ)  # (N-2,), cos[i] 比较 step i 与 step i+1
    step_norm = np.linalg.norm(dZ, axis=1)  # (N-1,)
    loss_drop = losses[:-1] - losses[1:]  # (N-1,), 正值表示下降

    # 把 cos[i] 对齐到 step i+1 的步长与损失下降。
    aligned_step = step_norm[1:]
    aligned_drop = loss_drop[1:]

    cos_mean = float(np.mean(cos)) if cos.size else None
    cos_last10 = float(np.mean(cos[-10:])) if cos.size else None

    # 噪声地板：窗口 5 的平滑余弦首次连续 3 步 < 0.7。
    noise_floor_from: int | None = None
    if cos.size:
        smoothed = _moving_average(cos, window=5)
        below = smoothed < 0.7
        for i in range(max(0, below.size - 2)):
            if bool(below[i]) and bool(below[i + 1]) and bool(below[i + 2]):
                noise_floor_from = i + 1  # step i+1 对应的迭代索引
                break

    consistent = cos > 0.9
    if consistent.size:
        consistent_step_norms = [float(v) for v in aligned_step[consistent]]
        consistent_loss_drops = [float(v) for v in aligned_drop[consistent]]
        consistent_active_frac = float(
            np.mean(consistent & (aligned_drop > 0.0))
        )
    else:
        consistent_step_norms = []
        consistent_loss_drops = []
        consistent_active_frac = 0.0

    return {
        "n_iters": n,
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "cos_mean": cos_mean,
        "cos_last10": cos_last10,
        "step_norm_mean": float(np.mean(step_norm)),
        "step_norm_std": float(np.std(step_norm)),
        "loss_drop_mean": float(np.mean(loss_drop)),
        "consistent_active_frac": consistent_active_frac,
        "noise_floor_from": noise_floor_from,
        "cos_values": [float(v) for v in cos],
        "consistent_step_norms": consistent_step_norms,
        "consistent_loss_drops": consistent_loss_drops,
    }


def make_recommendation(metrics: dict) -> str:
    """基于汇总指标生成 residual lr 调整建议（一阶线性代理，非真实重跑）。"""
    frac = float(metrics.get("frac_runs_cos_gt_0_9") or 0.0)
    rho = metrics.get("step_vs_lossdrop_spearman")
    if frac >= 0.8 and rho is not None and float(rho) > 0.3:
        return (
            "方向高度一致且步长与损失下降正相关：早期/中期（余弦>0.9 且损失仍在降）"
            "可安全放大 residual lr 加速收敛；进入噪声地板（余弦<0.7）后应恢复或减小，避免抖动。"
        )
    if frac >= 0.8:
        return (
            "方向一致但步长-收益关系弱：放大 lr 收益不确定，"
            "建议先小倍数 A/B 灰度验证，不直接改默认值。"
        )
    return "方向一致性不足，不建议据此处方化调步。"


def aggregate(analyses: list[dict]) -> dict:
    """跨轨迹汇总指标，并生成 ``recommendation``。"""
    n_runs = len(analyses)
    cos_last10_values = [
        float(a["cos_last10"]) for a in analyses if a.get("cos_last10") is not None
    ]
    frac_runs_cos_gt_0_9 = (
        sum(1 for v in cos_last10_values if v > 0.9) / n_runs if n_runs else 0.0
    )
    active_values = [
        float(a["consistent_active_frac"])
        for a in analyses
        if a.get("consistent_active_frac") is not None
    ]
    noise_floor_count = sum(
        1 for a in analyses if a.get("noise_floor_from") is not None
    )

    step_samples: list[float] = []
    drop_samples: list[float] = []
    for a in analyses:
        steps = a.get("consistent_step_norms") or []
        drops = a.get("consistent_loss_drops") or []
        for step, drop in zip(steps, drops):
            step_samples.append(float(step))
            drop_samples.append(float(drop))

    rho: float | None = None
    if len(step_samples) >= 5 and len(step_samples) == len(drop_samples):
        result = spearmanr(np.asarray(step_samples), np.asarray(drop_samples))
        value = float(result.statistic)
        if np.isfinite(value):
            rho = value

    metrics: dict[str, Any] = {
        "n_runs": n_runs,
        "cos_last10_median": float(np.median(cos_last10_values))
        if cos_last10_values
        else None,
        "frac_runs_cos_gt_0_9": float(frac_runs_cos_gt_0_9),
        "consistent_active_frac_median": float(np.median(active_values))
        if active_values
        else None,
        "frac_runs_reach_noise_floor": float(noise_floor_count / n_runs)
        if n_runs
        else 0.0,
        "step_vs_lossdrop_spearman": rho,
        "n_spearman_samples": len(step_samples),
    }
    metrics["recommendation"] = make_recommendation(metrics)
    return metrics


def _discover_histories(data_dir: Path) -> list[Path]:
    """扫描 ``<data-dir>/*/fit/iterations.pkl``；也接受直接指向某个 pkl。"""
    if data_dir.is_file():
        return [data_dir]
    return sorted(data_dir.glob(f"*/fit/{ITERATIONS_FILENAME}"))


def _read_history(path: Path) -> list[dict]:
    """读取一条 iterations.pkl，返回 ``list[dict]``。"""
    with path.open("rb") as fh:
        history = pickle.load(fh)
    if not isinstance(history, list):
        raise ValueError(f"{path} 内容不是 list[dict]")
    return history


def _json_safe(obj: object) -> object:
    """递归把 numpy / Path 值转换为 JSON 原生类型。"""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _render_figure(analyses: list[dict], out_path: Path) -> None:
    """输出两子图诊断 PNG：余弦曲线 + 步长/损失下降散点。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    max_len = max((len(a.get("cos_values") or []) for a in analyses), default=0)
    for a in analyses:
        values = np.asarray(a.get("cos_values") or [], dtype=float)
        if values.size == 0:
            continue
        ax.plot(np.arange(1, values.size + 1), values, color="0.6", lw=0.8)
    if analyses and max_len:
        padded = np.full((len(analyses), max_len), np.nan)
        for i, a in enumerate(analyses):
            values = np.asarray(a.get("cos_values") or [], dtype=float)
            padded[i, : values.size] = values
        median = np.nanmedian(padded, axis=0)
        ax.plot(
            np.arange(1, max_len + 1),
            median,
            color="red",
            lw=2,
            label=_LABELS["median"],
        )
        ax.legend(loc="lower right", fontsize=8)
    ax.axhline(0.9, ls="--", color="tab:green", lw=1)
    ax.axhline(0.7, ls="--", color="tab:orange", lw=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(_LABELS["cos_title"])
    ax.set_xlabel(_LABELS["iter"])
    ax.set_ylabel(_LABELS["cos"])

    ax = axes[1]
    xs: list[float] = []
    ys: list[float] = []
    for a in analyses:
        for step, drop in zip(
            a.get("consistent_step_norms") or [], a.get("consistent_loss_drops") or []
        ):
            xs.append(float(step))
            ys.append(float(drop))
    if xs:
        ax.scatter(xs, ys, s=10, alpha=0.4)
        x_arr = np.asarray(xs)
        y_arr = np.asarray(ys)
        if x_arr.size >= 2 and np.ptp(x_arr) > 0:
            slope, intercept = np.polyfit(x_arr, y_arr, 1)
            order = np.argsort(x_arr)
            ax.plot(
                x_arr[order],
                slope * x_arr[order] + intercept,
                color="red",
                lw=2,
                label=_LABELS["trend"],
            )
            ax.legend(loc="best", fontsize=8)
    ax.set_title(_LABELS["scatter_title"])
    ax.set_xlabel(_LABELS["step"])
    ax.set_ylabel(_LABELS["lossdrop"])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    """编排扫描→读取→分析→汇总→写 JSON→可选画图，返回结果 dict。"""
    data_dir = Path(args.data_dir)
    paths = _discover_histories(data_dir)
    logger.info("发现 {} 个 iterations.pkl 候选（data_dir={}）", len(paths), data_dir)

    analyses: list[dict] = []
    used_paths: list[str] = []
    for path in paths:
        try:
            history = _read_history(path)
        except Exception as exc:
            logger.warning("跳过无法读取的 pkl {}：{}", path, exc)
            continue
        if len(history) < int(args.min_iters):
            logger.debug("跳过过短历史 {}（n={} < {}）", path, len(history), args.min_iters)
            continue
        try:
            analysis = analyze_trajectory(history)
        except Exception as exc:
            logger.warning("分析失败，跳过 {}：{}", path, exc)
            continue
        analyses.append(analysis)
        used_paths.append(str(path))

    result: dict[str, Any] = {
        "aggregate": aggregate(analyses),
        "analyses": analyses,
        "sources": used_paths,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(result), fh, ensure_ascii=False, indent=2)
    logger.info("汇总 JSON 已写入 {}", out)

    figure = getattr(args, "figure", None)
    if figure:
        _render_figure(analyses, Path(figure))
        logger.info("诊断图已写入 {}", figure)

    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/debug/slm_transfer"),
        help="扫描 <data-dir>/*/fit/iterations.pkl，也接受直接指向某个 iterations.pkl",
    )
    parser.add_argument(
        "--min-iters",
        type=int,
        default=30,
        help="过滤迭代数不足该值的历史（默认：30）",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="JSON 汇总输出路径（aggregate + 每条轨迹的 analyses）",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="可选：诊断图 PNG 输出路径",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = run(args)
    metrics = result["aggregate"]
    logger.info(
        "n_runs={} cos_last10_median={} spearman={} recommendation={}",
        metrics["n_runs"],
        metrics["cos_last10_median"],
        metrics["step_vs_lossdrop_spearman"],
        metrics["recommendation"],
    )


if __name__ == "__main__":
    main()
