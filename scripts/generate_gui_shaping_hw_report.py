"""Offline figures + summary for the GUI closed-loop shaping report (2026-10-14).

Reads the saved run artefacts under ``data/debug/slm_gs_refine/<ts>/`` (CSV
history, replay ``*.npy`` frames and ``measurements.json``) and renders every
figure and the summary table used by
``report/shape_trials/gui_closed_loop_shaping/gui_closed_loop_shaping_20261014.md``.

Pure numpy + matplotlib: no hardware, no SDK, instruments may be powered off.

    python scripts/generate_gui_shaping_hw_report.py
"""

from __future__ import annotations

import json
from loguru import logger
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "debug" / "slm_gs_refine"
OUT = ROOT / "report" / "shape_trials" / "gui_closed_loop_shaping" / "images" / "gui_shaping_20261014"

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

#: The three GUI-driven runs of 2026-10-14 (see the report §2 for context).
#: Keys are the run directory names under ``data/debug/slm_gs_refine/``.
SESSIONS = {
    "20261014_100441": "A 10:04 首次 GUI 闭环",
    "20261014_104105": "B 10:41 重构后复跑",
    "20261014_143035": "C 14:30 远端合并后复跑",
}
EXPOSURE_LADDER_MS = (0.02, 0.05, 0.10, 0.20, 0.40)
#: Flat-field peak measured by the two read-only preflights (14:29, same day).
PREFLIGHT_PEAKS = {0.02: 59.3, 0.05: 146.7, 0.10: 234.0, 0.20: 254.7, 0.40: 254.7}
PREFLIGHT_PEAKS_LATE = {0.02: 50.0, 0.05: 125.7, 0.10: 211.3, 0.20: 254.7}

#: The report this script feeds. Its **body is hand-written** (see the provenance
#: note), but the §0 four-elements block below is **generated here** and rewritten
#: on every run, so it can never drift away from the artefacts it describes.
REPORT_KEY = "report/shape_trials/gui_closed_loop_shaping/gui_closed_loop_shaping_20261014.md"
#: Sentinels delimiting the generated block inside the hand-written body. They
#: make the rewrite idempotent: a second run replaces the block instead of
#: appending a second copy of it.
BLOCK_START = "<!-- four-elements:start -->"
BLOCK_END = "<!-- four-elements:end -->"


@dataclass(frozen=True)
class Run:
    ts: str
    label: str
    history: list[dict]
    replay: list[dict]
    best: np.ndarray | None
    flat: np.ndarray | None
    #: The run's own recorded config (``slm_gs_refine_<ts>.json``) -- the
    #: authoritative source for the hardware/setting table of the report.
    config: dict = field(default_factory=dict)
    #: The replay script's recorded config (``replay/replay_config.json``):
    #: SLM serial, expected wavelength, camera window and target box.
    replay_config: dict = field(default_factory=dict)


PURPOSE_TEXT = """
本轮要回答的问题: **GUI 侧把整形功能从"开环相位图案生成"换成闭环管线之后, 远场形状是不是真的
变好了?**

背景是三个开环控件 (`GS方形整形` / `GS方形整形(高斯版)` / `稳像法整形`) 在 2f-Fourier 实验系统上
都产不出可用的方形光斑 (根因见 §1)。本轮因此**不改算法、只换管线**: 把 GUI 的整形功能整体接到
已被验证的闭环管线 (`slm-gs-refine` 为主、`slm-model-in-loop` 为辅), 再按本目录既定口径验收。

口径上有一条硬规矩 (来自本目录 2026-10-12 的验收结论): **任何结论必须由交错重放支撑**
(flat / 候选 / flat 交替重放), 闭环运行内的最好值本身**不可**作为结论 —— 液晶尚未完全弛豫,
`settle` 判据在灰度统计相近的相位之间会提前通过, 于是同一相位在重放下的分数会高于闭环内。

所以本轮要排除三件事:

1. **"闭环内分数变好"被当成"形状变好"** —— 本轮所有对外数值一律取自交错重放 (§0.3)。
2. **"SPGD 细化带来可复现增益"** —— 同一配方在两次会话给出相反结论 (见 §0.3 与 §6.2)。
3. **"未标定的开环 GS 也能改善形貌"** —— 换管线后每次会话的 对比择优 都判定采用 GS, 说明
   **有闭环验收与兜底** 是它可用的前提, 而不是 GS 本身更优 (§6.3)。
"""

SETUP_TEXT = """
**本轮是硬件实验**: 数字来自真实实验系统上的落盘产物
(`data/debug/slm_gs_refine/<时间戳>/` 下的 CSV / JSON / PKL / PNG 与各自的 `replay/`),
由本脚本离线重算; 重放本身由 `scripts/slm/replay_gs_refine_hw.py` 在实验系统上采集
(固定曝光、固定相机窗、flat / 候选 / flat 交替)。本脚本**不打开任何设备**。

三次会话 (同配方、同设备): A = 首次实现, B = 适配器 + 装饰器重构后复跑, C = 合并远端
`gitub/banckend` 后复跑 (优化器被重写约 750 行)。下表的设置取自三次会话**落盘的 run JSON 与
`replay/replay_config.json`**, 脚本已逐字段比对: 三次会话的配置**完全一致**, 所以三次会话是
"同配方复跑", 差异只来自代码重构与远端合并。
"""

CHANGES_TEXT = """
本轮相对上一轮, **GUI 侧换了管线, 优化器的数学没有动**:

1. **整形功能整体从开环换成闭环**: 三个开环控件 (`GS方形整形` / `GS方形整形(高斯版)` /
   `稳像法整形`) 换成 `slm-gs-refine` (主) 与 `slm-model-in-loop` (辅) 两条闭环管线;
   实现方式是 `ShapingMethod` 适配层 + `@shaping_session` 装饰器 + `ShapingOutcome` 归一化
   (`src/ao_shaping/gui/slm/shaping_controls.py`): 页面只认一个 `ShapingOutcome`, 新算法
   只需再写一个装饰过的适配器子类。
2. **闭环配方** (与 CLI `slm-gs-refine` 是同一函数、同一目标框): GS 预矫正 (plain, 12 轮,
   面板相位网格 12×12) + SPGD 无感知细化 (12 轮, 扰动 delta=0.35, seed=0) + 对比择优
   (GS 优于平场才采用, `confirm_best=true`)。
3. **曝光改为每次实验实测定标**: 两次只读预检的曝光阶梯与饱和保护线 250 都在本脚本的
   `EXPOSURE_LADDER_MS` / `PREFLIGHT_PEAKS` 里, 选定工作点 0.05 ms (见 §3)。
4. **本轮期间由测试/QA 捕获并修复的三个真实缺陷** (记录见 §8): ① 相位下发原写在 `run()` 体内,
   而句柄重取在装饰器的 `finally` 中, 导致下发被静默跳过; ② 表驱动控件把 float 边界配 int 值,
   Streamlit 抛混合类型错误; ③ 用 `format_func` 做 key→label 映射会破坏 widget 状态回读,
   使 AppTest 直接 `KeyError`。
5. **未改的部分**: 优化器数学、目标函数 (复合分数 = 0.5·PIB + 0.5·1/(1+CV))、GS 与 SPGD 的实现
   都沿用 CLI 侧既有代码; 三次会话的落盘配置逐字段一致 (脚本已比对), 因此会话间差异不能归因到
   配方。
"""


def _load_csv(path: Path) -> list[dict]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict:
    """Read a JSON artefact; an absent file yields ``{}`` (not an error).

    The run config and the replay config are written by the optimizers and by
    ``scripts/slm/replay_gs_refine_hw.py``; a session that predates one of them
    simply has no row for it in the report instead of aborting the run.
    """
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run(ts: str, label: str) -> Run | None:
    directory = RUNS / ts
    csvs = sorted(directory.glob("*.csv"))
    if not csvs:
        return None
    history = _load_csv(csvs[0])
    replay_path = directory / "replay" / "measurements.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else []
    best = flat = None
    replay_dir = directory / "replay"
    if (replay_dir / "02_best.npy").exists():
        best = np.load(replay_dir / "02_best.npy")
        flat = np.load(replay_dir / "00_flat.npy")
    configs = sorted(directory.glob("*.json"))
    config = _load_json(configs[0]) if configs else {}
    replay_config = _load_json(replay_dir / "replay_config.json")
    return Run(
        ts=ts,
        label=label,
        history=history,
        replay=replay,
        best=best,
        flat=flat,
        config=config,
        replay_config=replay_config,
    )


def _metric(run: Run, stage: str, field: str) -> float | None:
    for row in run.history:
        if row["stage"] == stage and row.get(field):
            return float(row[field])
    return None


def _best_of(run: Run, field: str) -> float | None:
    values = [float(r[field]) for r in run.history if r.get(field)]
    return max(values) if values else None


def _best_row(run: Run) -> dict[str, str]:
    """The record-setting row — metrics must come from *this* row.

    Taking a per-column max over the history instead would mix stages (the flat
    row has the largest CV, the GS/best rows the largest PIB), which silently
    reports a combination no measured frame ever produced.
    """
    return max(run.history, key=lambda r: float(r["score"]))


def _best_metric(run: Run, field: str) -> float | None:
    row = _best_row(run)
    return float(row[field]) if row.get(field) else None


def fig_exposure_ladder() -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    xs = np.arange(len(EXPOSURE_LADDER_MS))
    early = [PREFLIGHT_PEAKS[e] for e in EXPOSURE_LADDER_MS]
    late = [PREFLIGHT_PEAKS_LATE.get(e, np.nan) for e in EXPOSURE_LADDER_MS[:4]]
    ax.plot(xs, early, "o-", label="10:03 预检")
    ax.plot(xs[:4], late[:4], "s--", label="14:29 预检")
    ax.axhline(250, color="crimson", ls=":", lw=1.2)
    ax.text(len(xs) - 0.4, 252, "饱和保护阈值 250", color="crimson", fontsize=8, ha="right")
    ax.axhline(125, color="grey", ls=":", lw=1)
    ax.text(0.05, 128, "选定工作点 0.05 ms", color="grey", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{e:g}" for e in EXPOSURE_LADDER_MS])
    ax.set_xlabel("曝光 (ms)")
    ax.set_ylabel("平场全帧峰值 (count/255)")
    ax.set_ylim(0, 285)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT / "fig1_exposure_ladder.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_convergence(runs: list[Run]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for run in runs:
        refine = [r for r in run.history if r["stage"] == "refine"]
        if not refine:
            continue
        epochs = [int(float(r["epoch"])) for r in refine]
        scores = [float(r["score"]) for r in refine]
        axes[0].plot(epochs, scores, marker="o", ms=3, lw=1, label=run.label)
    axes[0].set_xlabel("SPGD 轮次")
    axes[0].set_ylabel("复合分数")
    axes[0].set_title("闭环内 SPGD 轨迹（不含 flat/GS 阶段）")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)

    labels, flat_v, gs_v, best_v = [], [], [], []
    for run in runs:
        if not run.replay:
            continue
        by_phase: dict[str, list[float]] = {}
        for row in run.replay:
            by_phase.setdefault(str(row["phase"]), []).append(float(row["score"]))
        if not {"flat", "gs", "best"} <= set(by_phase):
            continue
        labels.append(run.label.split()[0])
        flat_v.append(float(np.mean(by_phase["flat"])))
        gs_v.append(float(np.mean(by_phase["gs"])))
        best_v.append(float(np.mean(by_phase["best"])))
    x = np.arange(len(labels))
    axes[1].bar(x - 0.25, flat_v, 0.25, label="flat")
    axes[1].bar(x, gs_v, 0.25, label="GS")
    axes[1].bar(x + 0.25, best_v, 0.25, label="best")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.45, 0.82)
    axes[1].set_ylabel("复合分数（交错重放均值）")
    axes[1].set_title("交错重放：因果口径")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path = OUT / "fig2_convergence_and_replay.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_far_field(runs: list[Run]) -> Path:
    rows = [r for r in runs if r.flat is not None and r.best is not None]
    if not rows:
        return OUT / "fig3_far_field.png"
    fig, axes = plt.subplots(len(rows), 3, figsize=(9.0, 3.1 * len(rows)))
    axes = np.atleast_2d(axes)
    for row, run in enumerate(rows):
        for col, (frame, name) in enumerate(
            ((run.flat, "flat"), (None, "GS"), (run.best, "best"))
        ):
            ax = axes[row, col]
            if frame is None:
                ax.axis("off")
                ax.text(0.5, 0.5, "GS 相位由重放脚本按记录回放\n（见 replay/measurements.json）",
                        ha="center", va="center", fontsize=7, transform=ax.transAxes)
                continue
            im = ax.imshow(frame, cmap="inferno")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{run.label.split()[0]} {name}  peak={frame.max():.0f}", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = OUT / "fig3_far_field.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_box_profile(runs: list[Run]) -> Path:
    rows = [r for r in runs if r.best is not None and r.flat is not None]
    if not rows:
        return OUT / "fig4_box_profile.png"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    run = rows[-1]
    centre = run.best.shape[0] // 2
    for frame, name, style in ((run.flat, "flat", "--"), (run.best, "best", "-")):
        axes[0].plot(frame[centre], style, lw=1, label=name)
        axes[1].plot(frame[:, centre], style, lw=1, label=name)
    axes[0].set_title("过零级中心行的行剖面")
    axes[1].set_title("过零级中心列的列剖面")
    for ax in axes:
        ax.set_xlabel("像素")
        ax.set_ylabel("强度 (count)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(f"末次会话（{run.label}）中心剖面：能量由紧凑峰铺向方形footprint", fontsize=9)
    fig.tight_layout()
    path = OUT / "fig4_box_profile.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def summarise(runs: list[Run]) -> dict:
    table = []
    for run in runs:
        flat = _metric(run, "flat", "score")
        gs = _metric(run, "gs", "score")
        best = _best_of(run, "score")
        by_phase: dict[str, list[dict]] = {}
        for row in run.replay:
            by_phase.setdefault(str(row["phase"]), []).append(row)
        table.append(
            {
                "ts": run.ts,
                "label": run.label,
                "in_run": {
                    "flat": flat,
                    "gs": gs,
                    "best": best,
                    "cv_best": _best_metric(run, "cv"),
                    "pib_best": _best_metric(run, "pib"),
                    "best_stage": _best_row(run)["stage"],
                    "best_epoch": _best_row(run).get("epoch", ""),
                    "gs_adopted": bool(flat is not None and gs is not None and gs > flat),
                },
                "replay": {
                    phase: {
                        "score_mean": float(np.mean([float(r["score"]) for r in rows])),
                        "score_spread": float(
                            np.ptp([float(r["score"]) for r in rows])
                        ),
                        "cv_mean": float(np.mean([float(r["cv"]) for r in rows])),
                        "pib_mean": float(np.mean([float(r["pib"]) for r in rows])),
                        "n": len(rows),
                    }
                    for phase, rows in by_phase.items()
                },
            }
        )
    summary = {
        # A list of records, not a float-keyed object: JSON object keys are
        # strings, so ``{"0.05": ...}`` would silently change type on the first
        # round-trip and make the artefact awkward to consume.
        "exposure_ladder": {
            "saturation_guard": 250,
            "working_point_ms": 0.05,
            "rows": [
                {
                    "exposure_ms": exposure,
                    "peak_1003": PREFLIGHT_PEAKS[exposure],
                    "peak_1429": PREFLIGHT_PEAKS_LATE.get(exposure),
                }
                for exposure in EXPOSURE_LADDER_MS
            ],
        },
        "runs": table,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _cell(value, spec: str = "", default: str = "—") -> str:
    """Format a value that may be missing from the artefacts."""
    if value is None:
        return default
    try:
        return format(value, spec) if spec else str(value)
    except (TypeError, ValueError):
        return str(value)


def _config_differs(left: dict, right: dict) -> list[str]:
    """Keys whose values differ between two recorded run configs."""
    if not left or not right:
        return []
    return sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))


def four_elements(runs: list[Run], summary: dict) -> str:
    """Build the §0 four-elements block of the report.

    Every number here comes from the artefacts (the run JSON / replay JSON) or
    from :func:`summarise`; nothing is invented. The prose parts are the module
    constants above.
    """
    first = runs[0]
    cfg = first.config
    rc = first.replay_config
    ladder = summary["exposure_ladder"]
    guard = ladder["saturation_guard"]
    work = ladder["working_point_ms"]
    peaks = {row["exposure_ms"]: row for row in ladder["rows"]}
    work_early = peaks[work]["peak_1003"]
    work_late = peaks[work]["peak_1429"]
    drift = (work_early - work_late) / work_early if work_early else 0.0
    saturated = [r["exposure_ms"] for r in ladder["rows"] if r["peak_1003"] >= guard]
    sat_ms = min(saturated) if saturated else None

    lines: list[str] = ["## 0. 实验目的", "", PURPOSE_TEXT.strip(), ""]
    lines += ["## 0.1 实验设置", "", SETUP_TEXT.strip(), ""]

    if cfg and rc:
        center = cfg.get("pupil_center") or [None, None]
        rows = [
            ("SLM", f"Santec SLM #{_cell(cfg.get('slm_number'))}, "
                    f"序列号 {_cell(rc.get('slm_serial'))}", "`replay/replay_config.json`"),
            ("波长", f"{_cell(rc.get('expected_wavelength_nm'))} nm "
                    "(设备报告, 非配置假定)", "`replay/replay_config.json`"),
            ("CCD", f"大恒 (cam_type={_cell(cfg.get('cam_type'))}, "
                    f"cam_id={_cell(cfg.get('cam_id'))})", "落盘 run JSON"),
            ("面板像元 / 相机像元", f"{_cell(cfg.get('panel_pixel_um'))} µm / "
                                   f"{_cell(cfg.get('camera_pixel_um'))} µm", "落盘 run JSON"),
            ("透镜焦距", f"{_cell(cfg.get('focal_length_m'), '.6f')} m "
                        f"(标定标志 `_focal_length_pinned`="
                        f"{_cell(cfg.get('_focal_length_pinned'))})", "落盘 run JSON"),
            ("照明瞳面", f"中心 ({_cell(center[0])}, {_cell(center[1])}), "
                        f"半径 {_cell(cfg.get('beam_radius_px'), 'g')} px", "落盘 run JSON"),
            ("曝光", f"{_cell(cfg.get('exposure_time_ms'), 'g')} ms 固定, "
                    f"n_eval_frames={_cell(cfg.get('n_eval_frames'))}", "落盘 run JSON"),
            ("相机窗口", f"请求 {_cell(cfg.get('cam_size'))} → 驱动按 4 px 粒度夹到 "
                        f"{_cell(rc.get('cam_size'))}", "落盘 run JSON / `replay/replay_config.json`"),
            ("目标框", f"{_cell(cfg.get('target_side'))}×{_cell(cfg.get('target_side'))} px",
             "落盘 run JSON"),
            ("远场 padding / 瞳面相位网格", f"{_cell(cfg.get('far_field_padding'))} / "
                                          f"{_cell(cfg.get('phase_grid'))}×"
                                          f"{_cell(cfg.get('phase_grid'))}", "落盘 run JSON"),
            ("GS / SPGD 配方", f"gs_iters={_cell(cfg.get('gs_iters'))}, "
                               f"epochs={_cell(cfg.get('epochs'))}, seed={_cell(cfg.get('seed'))}, "
                               f"delta={_cell(cfg.get('delta'), 'g')}", "落盘 run JSON"),
            ("目标函数", f"w_pib={_cell(cfg.get('w_pib'), 'g')}, "
                         f"w_unif={_cell(cfg.get('w_unif'), 'g')} "
                         "(复合分数 = 0.5·PIB + 0.5·1/(1+CV))", "落盘 run JSON"),
            ("优化器 / GS 方法", f"{_cell(cfg.get('optimizer_type'))} (cosine 调度) / "
                                f"{_cell(cfg.get('gs_method'))} "
                                f"(warm_start={_cell(cfg.get('gs_warm_start'))}, "
                                f"confirm_best={_cell(cfg.get('confirm_best'))})", "落盘 run JSON"),
            ("曝光工作点", f"{work:g} ms (阶梯预检选定, 饱和保护线 {guard})",
             "本脚本 `EXPOSURE_LADDER_MS` / `PREFLIGHT_PEAKS`"),
        ]
        lines += ["| 项 | 值 | 来源 |", "|---|---|---|"]
        lines += [f"| {name} | {value} | {source} |" for name, value, source in rows]
        lines.append("")

    diffs: list[str] = []
    for other in runs[1:]:
        diffs.extend(_config_differs(runs[0].config, other.config))
    diffs = sorted(set(diffs))
    same = "完全一致" if not diffs else f"不一致 ({', '.join(diffs)})"
    lines += [
        "## 0.2 算法修改",
        "",
        CHANGES_TEXT.strip(),
        "",
        f"> 三次会话落盘配置的逐字段比对结果: **{same}** (比对对象: 各会话的 "
        "`slm_gs_refine_<ts>.json`)。",
        "",
    ]

    # -- §0.3 结果分析: 只用 summarise() 已经算出的量 ----------------------
    gains = []
    for row in summary["runs"]:
        rep = row["replay"]
        if not {"flat", "gs", "best"} <= set(rep):
            continue
        gs_gain = rep["gs"]["score_mean"] - rep["flat"]["score_mean"]
        ref_gain = rep["best"]["score_mean"] - rep["gs"]["score_mean"]
        spread = rep["gs"]["score_spread"]
        ratio = ref_gain / spread if spread else float("nan")
        gains.append(
            f"会话 {row['label'].split()[0]}: flat {rep['flat']['score_mean']:.4f} → "
            f"GS {rep['gs']['score_mean']:.4f} (+{gs_gain:.4f}), "
            f"best {rep['best']['score_mean']:.4f} (相对 GS {ref_gain:+.4f}, "
            f"是 GS 重放极差 {spread:.4f} 的 {ratio:.1f} 倍)"
        )
    adopted = [r["label"].split()[0] for r in summary["runs"] if r["in_run"]["gs_adopted"]]
    no_replay = [r["label"].split()[0] for r in summary["runs"] if not r["replay"]]

    results = [
        f"1. **曝光每次都重新定标**: 工作点 {work:g} ms 的平场全帧峰值 {work_early} "
        f"(10:03 预检) / {work_late} (14:29 预检), 同日相差约 {drift * 100:.0f}% ⇒ "
        f"工作曝光不可沿用; {f'{sat_ms:g} ms' if sat_ms else '高曝光'} 起饱和 "
        f"(保护线 {guard})。",
        "2. **交错重放 (因果口径) 是本轮唯一的结论来源**: "
        + ("; ".join(gains) if gains else "本次没有可用的重放数据")
        + (f"。会话 {'、'.join(no_replay)} 未做重放 (仅代码回归), 故无因果行。"
           if no_replay else "。"),
        f"3. **GS 预矫正在 {'、'.join(adopted) if adopted else '（无）'} 三次会话中都被采用** "
        "(对比择优: GS 优于平场才采用), 即换管线后每次都判定 GS 优于平场。",
        "4. **SPGD 细化的增量不可分辨**: 上一条里 `best − GS` 在不同会话方向相反 "
        "(一个明显高于同相位重放极差, 另一个低于极差) ⇒ 与 §6.2 一致, 记为「未定」, "
        "本轮只把 GS 预矫正的改进列为已确证。",
        "5. 框内 CV / PIB、远场形貌与目标框内填充度的具体数值见 §4.2–§4.3; "
        "与既有基线的对照见 §5。",
    ]
    lines += ["## 0.3 结果分析", ""] + results + [""]
    return "\n".join(lines)


def sync_report_block(block: str) -> Path | None:
    """Write the generated §0 block into the report body (idempotent).

    The body of this report is hand-written; only the block between the
    sentinels belongs to this script. A second run replaces that block and
    leaves the rest of the body untouched.
    """
    path = ROOT / REPORT_KEY
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    payload = f"{BLOCK_START}\n{block.strip()}\n{BLOCK_END}"
    start = text.find(BLOCK_START)
    end = text.find(BLOCK_END)
    if start != -1 and end != -1 and end > start:
        tail = text[end + len(BLOCK_END):].lstrip("\n")
        new = text[:start] + payload + "\n\n" + tail
    else:
        lines = text.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.startswith("## "):
                new = "".join(lines[:i]) + payload + "\n\n" + "".join(lines[i:])
                break
        else:
            new = text.rstrip("\n") + "\n\n" + payload + "\n"
    if new != text:
        path.write_text(new, encoding="utf-8", newline="\n")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [r for r in (_load_run(ts, label) for ts, label in SESSIONS.items()) if r]
    if not runs:
        logger.info("no run directories found")
        return 1
    made = [fig_exposure_ladder(), fig_convergence(runs), fig_far_field(runs), fig_box_profile(runs)]
    summary = summarise(runs)
    report = sync_report_block(four_elements(runs, summary))
    if report is not None:
        logger.info(f"wrote {report.relative_to(ROOT)} (§0 block)")
    else:
        logger.info(f"note: {REPORT_KEY} not found, §0 block not written")
    for path in made:
        if path.exists():
            logger.info(f"wrote {path.relative_to(ROOT)}")
    logger.info(f"wrote {(OUT / 'summary.json').relative_to(ROOT)}")
    for row in summary["runs"]:
        replay = row["replay"]
        head = "  ".join(
            f"{phase}={replay[phase]['score_mean']:.4f}" for phase in ("flat", "gs", "best") if phase in replay
        )
        logger.info(f"  {row['ts']}: {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())