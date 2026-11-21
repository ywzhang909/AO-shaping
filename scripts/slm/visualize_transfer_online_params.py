"""离线绘制 transfer 在线训练参数过程图（闭环轮次 + fit 迭代）。

数据来源全部是已落盘的真实产物，不运行设备、不修改上游工具：

* 闭环轮次级：``<session>/rounds.json`` 的指标 + 各
  ``fit`` / ``fit.roundN`` / ``anchored...`` 目录中的 ``frozen_parameters.json``；
  注意历史闭环存档只在 ``rounds.json`` 记录指标，不记录参数，本脚本从每轮
  fit 快照中补出参数，并明确标注这一口径。
* 迭代级：``<run>/fit/debug_weights/weights_history.jsonl``（由 fit.py 的
  ``--debug-weights`` 生成），记录每次 optimizer step 后的可训练参数。

输出 PNG 到 ``report/shape_trials/transfer_online_params/figures/``，
并写一份 ``artifacts/`` 文本摘要供报告核对。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SESSION = ROOT / "data" / "debug" / "slm_transfer" / "20261105_173655"
DEFAULT_TRACE = (
    ROOT
    / "data"
    / "debug"
    / "slm_transfer"
    / "debug_weights_real_force_20261121"
    / "fit"
    / "debug_weights"
    / "weights_history.jsonl"
)
DEFAULT_OUT = ROOT / "report" / "shape_trials" / "transfer_online_params"

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


def resolve_cjk_font() -> str | None:
    """返回可用的中文字体，找不到则让图内标签回退英文。"""
    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            return name
    return None


CJK_FONT = resolve_cjk_font()
if CJK_FONT is not None:
    matplotlib.rcParams["font.sans-serif"] = [CJK_FONT, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

LABELS_ZH = {
    "round_title": "transfer 在线训练参数随闭环轮次变化",
    "round_subtitle": "每轮 fit 最终快照（来自 frozen_parameters.json）",
    "metric_corr": "实测相关性",
    "metric_rel": "实测相对 L2",
    "iteration": "闭环轮次",
    "metric": "指标值",
    "param": "参数值（各参数独立纵轴）",
    "trace_title": "在线训练参数随 fit 迭代变化",
    "trace_subtitle": "逐迭代参数轨迹（来自 --debug-weights，真实采集）",
    "loss": "训练损失",
    "step": "fit 迭代",
    "normalized": "相对全程变化范围（Δ / max(极差, 1e-12)）",
    "best": "最优损失迭代",
    "no_rounds": "没有可用闭环轮次快照",
    "no_trace": "没有可用逐迭代权重轨迹",
}
LABELS_EN = {
    "round_title": "Online transfer parameters across closed-loop rounds",
    "round_subtitle": "Final fit snapshot per round (frozen_parameters.json)",
    "metric_corr": "Measured correlation",
    "metric_rel": "Measured relative L2",
    "iteration": "Closed-loop round",
    "metric": "Metric value",
    "param": "Parameter value (independent y-axis per trace)",
    "trace_title": "Online parameters across fit iterations",
    "trace_subtitle": "Per-iteration trace from --debug-weights (real capture)",
    "loss": "Training loss",
    "step": "Fit iteration",
    "normalized": "Change / full observed range (delta / max(range, 1e-12))",
    "best": "best loss iteration",
    "no_rounds": "No closed-loop parameter snapshots found",
    "no_trace": "No per-iteration weight trace found",
}
LABELS = LABELS_ZH if CJK_FONT else LABELS_EN

# 只画可训练/拟合的物理参数；ramp_matrix / grid_size 等诊断配置不是训练量。
ROUND_PARAM_KEYS = (
    "defocus_diopter",
    "origin_0",
    "origin_1",
    "amplitude_center_0",
    "amplitude_center_1",
    "amplitude_width_0",
    "amplitude_width_1",
    "residual_noll_5",
    "residual_noll_6",
    "residual_noll_7",
    "residual_noll_8",
    "residual_noll_9",
    "residual_noll_10",
    "residual_noll_11",
    "gain",
    "background",
    "sigma_ccd_px",
    "bias_counts",
)


def flatten_parameters(params: dict) -> dict[str, float]:
    """复用 fit.py::_flatten_parameters 的字段口径，避免图与产物不一致。"""
    flat: dict[str, float] = {
        "defocus_diopter": float(params.get("defocus_diopter", 0.0)),
        "gain": float(params.get("gain", 0.0)),
        "background": float(params.get("background", 0.0)),
        "sigma_ccd_px": float(params.get("sigma_ccd_px", 0.0)),
    }
    for key in ("origin", "amplitude_center", "amplitude_width"):
        for i, value in enumerate(params.get(key) or []):
            flat[f"{key}_{i}"] = float(value)
    for mode, value in (params.get("residual_noll_radians") or {}).items():
        flat[f"residual_noll_{mode}"] = float(value)
    flat["bias_counts"] = float(params.get("bias_counts", 0.0))
    return flat


def load_round_rows(session: Path) -> tuple[list[dict], list[str]]:
    """读取 rounds.json 和各 fit 快照，返回按轮次对齐的行。"""
    rounds_path = session / "rounds.json"
    rounds = json.loads(rounds_path.read_text(encoding="utf-8")) if rounds_path.exists() else []
    rows: list[dict] = []
    sources: list[str] = []
    for rnd in rounds:
        fit_dir = rnd.get("fit_directory") or "fit"
        snapshot = session / fit_dir / "frozen_parameters.json"
        if not snapshot.exists():
            continue
        flat = flatten_parameters(json.loads(snapshot.read_text(encoding="utf-8")))
        rows.append(
            {
                "round": int(rnd.get("round", len(rows))),
                "correlation": rnd.get("measured_correlation"),
                "relative_l2": rnd.get("measured_relative_l2"),
                "accepted": bool(rnd.get("accepted", False)),
                "parameters": flat,
            }
        )
        sources.append(str(snapshot.relative_to(session)).replace("\\", "/"))
    return rows, sources


def load_trace(path: Path) -> list[dict]:
    """读取 weights_history.jsonl；每行 iteration/loss/parameters。"""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            {
                "iteration": int(row["iteration"]),
                "loss": float(row["loss"]),
                "parameters": {k: float(v) for k, v in row["parameters"].items()},
            }
        )
    rows.sort(key=lambda r: r["iteration"])
    return rows


def plot_rounds(rows: list[dict], out: Path) -> None:
    """两面板：验收指标 + 每轮最终参数（归一化到初值，独立纵轴）。"""
    fig, (ax_metric, ax_param) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [1, 2.4]}
    )
    if not rows:
        ax_metric.text(0.5, 0.5, LABELS["no_rounds"], ha="center", va="center")
        ax_param.axis("off")
    else:
        rounds = [r["round"] for r in rows]
        corr = [r["correlation"] for r in rows if r["correlation"] is not None]
        rel = [r["relative_l2"] for r in rows if r["relative_l2"] is not None]
        ax_metric.plot(rounds[: len(corr)], corr, "o-", color="#4C78A8", label=LABELS["metric_corr"])
        ax_metric.plot(rounds[: len(rel)], rel, "s-", color="#E45756", label=LABELS["metric_rel"])
        for r in rows:
            color = "#2CA02C" if r["accepted"] else "#BAB0AC"
            ax_metric.axvspan(r["round"] - 0.4, r["round"] + 0.4, color=color, alpha=0.08, lw=0)
        ax_metric.set_ylabel(LABELS["metric"])
        ax_metric.legend(loc="best", fontsize=9)
        ax_metric.grid(alpha=0.3)

        keys = [k for k in ROUND_PARAM_KEYS if k in rows[0]["parameters"]]
        cmap = plt.get_cmap("tab20")
        for i, key in enumerate(keys):
            vals = np.array([r["parameters"][key] for r in rows], dtype=float)
            span = max(float(np.ptp(vals)), 1e-12)
            norm = (vals - vals[0]) / span
            ax_param.plot(rounds, norm, "o-", ms=4, lw=1.4, color=cmap(i % 20), label=key)
        ax_param.axhline(0.0, color="#888888", lw=0.8, ls="--")
        ax_param.set_ylabel(LABELS["normalized"])
        ax_param.set_xlabel(LABELS["iteration"])
        ax_param.grid(alpha=0.3)
        ax_param.legend(ncol=3, fontsize=7.5, loc="best")
    fig.suptitle(LABELS["round_title"], fontsize=14)
    ax_metric.set_title(LABELS["round_subtitle"], fontsize=9, color="#555555")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_trace(rows: list[dict], out: Path) -> None:
    """两面板：训练损失 + 18 个可训练参数逐迭代轨迹。"""
    fig, (ax_loss, ax_param) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [1, 2.6]}
    )
    if not rows:
        ax_loss.text(0.5, 0.5, LABELS["no_trace"], ha="center", va="center")
        ax_param.axis("off")
    else:
        iters = [r["iteration"] for r in rows]
        losses = [r["loss"] for r in rows]
        ax_loss.plot(iters, losses, "o-", color="#4C78A8")
        best = int(iters[int(np.argmin(losses))])
        ax_loss.axvline(best, color="#E45756", ls="--", lw=1, label=f"{LABELS['best']}={best}")
        ax_loss.set_ylabel(LABELS["loss"])
        ax_loss.legend(loc="best", fontsize=9)
        ax_loss.grid(alpha=0.3)

        keys = list(rows[0]["parameters"])
        cmap = plt.get_cmap("tab20")
        for i, key in enumerate(keys):
            vals = np.array([r["parameters"][key] for r in rows], dtype=float)
            span = max(float(np.ptp(vals)), 1e-12)
            norm = (vals - vals[0]) / span
            ax_param.plot(iters, norm, "o-", ms=4, lw=1.3, color=cmap(i % 20), label=key)
        ax_param.axhline(0.0, color="#888888", lw=0.8, ls="--")
        ax_param.set_ylabel(LABELS["normalized"])
        ax_param.set_xlabel(LABELS["step"])
        ax_param.grid(alpha=0.3)
        ax_param.legend(ncol=3, fontsize=7.5, loc="best")
    fig.suptitle(LABELS["trace_title"], fontsize=14)
    ax_loss.set_title(LABELS["trace_subtitle"], fontsize=9, color="#555555")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION, help="闭环存档目录")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE, help="weights_history.jsonl")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="报告目录")
    args = parser.parse_args()

    fig_dir = args.out / "figures"
    art_dir = args.out / "artifacts"
    fig_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    rounds, sources = load_round_rows(args.session)
    trace = load_trace(args.trace)
    plot_rounds(rounds, fig_dir / "online_params_by_round.png")
    plot_trace(trace, fig_dir / "online_params_by_iteration.png")

    summary = {
        "round_session": str(args.session.relative_to(ROOT) if args.session.is_relative_to(ROOT) else args.session),
        "round_sources": sources,
        "round_rows": [
            {
                "round": r["round"],
                "measured_correlation": r["correlation"],
                "measured_relative_l2": r["relative_l2"],
                "accepted": r["accepted"],
            }
            for r in rounds
        ],
        "trace_file": str(args.trace.relative_to(ROOT) if args.trace.is_relative_to(ROOT) else args.trace),
        "trace_iterations": [r["iteration"] for r in trace],
        "trace_losses": [r["loss"] for r in trace],
        "trace_parameters": list(trace[0]["parameters"]) if trace else [],
        "note": "rounds.json 不记录参数；轮次参数由每轮 frozen_parameters.json 补出。",
    }
    (art_dir / "transfer_online_params_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"rounds={len(rounds)} trace_iterations={len(trace)}")
    print(f"figures: {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
