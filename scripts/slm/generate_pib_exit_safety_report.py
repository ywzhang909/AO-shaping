"""Render the **offline** report for the issue #145 (H-28) exit-safety acceptance.

Issue #145 asks one question the digital twin cannot answer: after a shaping run
dies -- Ctrl-C, or the camera dropping out -- **what is the real SLM panel
actually displaying**? ``scripts/slm/verify_pib_exit_safety_hw.py`` runs the
scenarios on the instruments and writes ``summary.json`` plus the far-field
crops. This script reads those artifacts and renders one illustrated Chinese
report. It never constructs a camera or an SLM, so the evidence can be re-read
and re-rendered with the instruments powered down.

What the report has to keep straight
-----------------------------------
1. **Two independent answers, not one.** The firmware readback
   (``get_displayed_memory_number`` / ``get_current_grayscale``) names the state;
   the optical comparison is what catches a firmware *no-op* -- writing to the
   slot the LCOS already shows does not refresh the panel. The report prints
   both, and says which of the two is gated by the verdict.
2. **The exit branch is data, not an assumption.** ``_apply_best_on_exit``
   restores the best phase when the search improved, and falls back to flat
   gray-0 when it did not (or when the restore itself fails). Which branch fired
   is read from the write log, so a run where the search never improved is not
   quietly reported as "best phase restored".
3. **Numbers come from the artifacts.** Every figure and every cell is read from
   ``summary.json`` / the ``.npz`` crops; nothing is typed in by hand.

Usage::

    python scripts/slm/generate_pib_exit_safety_report.py
    python scripts/slm/generate_pib_exit_safety_report.py --run-dir report/slm/pib_exit_safety/artifacts/run_20261024_120000
    python scripts/slm/generate_pib_exit_safety_report.py --no-figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from scripts._common import fmt_num_or_dash
from scripts._common.provenance import insert_header

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["font.size"] = 10

REPORT_KEY = "report/slm/pib_exit_safety/pib_exit_safety_20261025.md"
OUT_DIR = ROOT / "report" / "slm" / "pib_exit_safety"
FIG_DIR = OUT_DIR / "figures"
ARTIFACT_ROOT = OUT_DIR / "artifacts"

#: Frame states the harness saves, longest suffix first so ``flat_repeat`` is not
#: mistaken for ``flat``.
FRAME_STATES: tuple[tuple[str, str], ...] = (
    ("flat_repeat", "平场重复帧（噪声底）"),
    ("last_iterate", "搜索最后一轮迭代"),
    ("exit", "退出写入的相位"),
    ("live", "崩溃后面板实测"),
    ("flat", "平场 gray-0 参照"),
)

#: What each gated predicate means, for the acceptance-criteria table. The gate
#: list itself is read from each record's ``verdict_keys``, so this table cannot
#: silently disagree with what the harness actually gated.
CHECK_MEANINGS: dict[str, str] = {
    "failure_reached_caller": "注入的异常确实到达了调用方（场景真的以失败告终）",
    "original_exception_survived_cleanup": "到达调用方的异常类型就是注入的那一种：清理没有用新异常覆盖它",
    "run_completed": "该场景不注入异常，正常运行到结束",
    "firmware_is_named_state": "固件回读得到的状态是本脚本记录过的某个写入（内存槽号一致，或直接灰度模式且灰度值为 0）",
    "exit_used_a_fresh_slot": "退出写入没有落在当前已显示的内存槽上；落在同一槽时 `display_memory` 是空操作，面板不会刷新",
    "optical_is_defined_state": "面板实测画面与某个已写入状态的相关性达到重复帧噪声底，说明面板显示的是写过的状态而非任意图案",
    "cleanup_under_5s": "从注入到函数返回不超过 5 s（报告同时给出实测值；验收要求为 1–2 s）",
    "energy_loss_violations_recorded": "`recorder` 上带有 int 类型的 `energy_loss_violations`",
    "epoch0_is_sane": "异常后紧接的正常短跑，其 epoch-0 的 `pib` 与峰值亮度处于健康量级（不是目标框未对准的坍缩症状）",
}

#: The six observation points of issue #145, each mapped to the evidence that
#: answers it. Kept here (not in the artifact) because it is a statement about
#: the issue, not about a particular run.
OBSERVATION_POINTS: tuple[tuple[str, str, str], ...] = (
    (
        "① Ctrl-C 后面板停在最佳相位或平场",
        "`exit_branch` + `firmware_is_named_state` + `exit_used_a_fresh_slot` + `optical_is_defined_state`",
        "场景 A / D / E",
    ),
    (
        "② 相机掉线后面板强制平场 gray-0",
        "`exit_branch == flat_fallback`、`firmware_mode == grayscale`、`current_grayscale == 0`",
        "场景 B",
    ),
    (
        "③ 原始异常不被清理吞掉",
        "`original_exception_survived_cleanup` + `outcome.exception_message`",
        "场景 A / B / D / E",
    ),
    (
        "④ 清理在 1–2 s 内完成",
        "`outcome.cleanup_s`（门禁 `cleanup_under_5s` 放宽到 5 s，给记录开销留余量）",
        "场景 A / B / D / E",
    ),
    (
        "⑤ 异常后紧接的正常短跑 epoch-0 正常",
        "`recorder.epoch0.pib` / `epoch0.max_brt`（门禁 `epoch0_is_sane`）",
        "场景 C",
    ),
    (
        "⑥ `energy_loss_violations` 以 int 出现在 recorder 上",
        "`recorder.energy_loss_violations` + `energy_loss_violations_recorded`",
        "场景 A / B / D / E",
    ),
)


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def find_latest_run(root: Path = ARTIFACT_ROOT) -> Path | None:
    """Newest ``run_*`` directory that actually carries a ``summary.json``.

    A run directory without a summary is a crashed harness, not evidence; it is
    skipped rather than rendered as an empty report.
    """
    if not root.is_dir():
        return None
    runs = [p for p in root.glob("run_*") if (p / "summary.json").is_file()]
    return sorted(runs)[-1] if runs else None


def load_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _fmt_optional(value: Any, digits: int = 2, dash: str = "—") -> str:
    return fmt_num_or_dash(value, digits, dash)


def load_frames(record: dict[str, Any], run_dir: Path) -> dict[str, np.ndarray]:
    """Read the saved crops for one scenario, keyed by state name."""
    panel = record.get("panel") or {}
    frames: dict[str, np.ndarray] = {}
    for name in panel.get("frame_files") or []:
        for state, _label in FRAME_STATES:
            if not name.endswith(f"_{state}.npz"):
                continue
            path = run_dir / "frames" / name
            if not path.is_file():
                logger.warning("frame listed but missing: {}", path)
                continue
            with np.load(path) as data:
                frames[state] = np.asarray(data["frame"], dtype=float)
    return frames


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_scenario_panels(
    record: dict[str, Any], frames: dict[str, np.ndarray], out_path: Path
) -> Path | None:
    """live / flat / exit / last-iterate crops side by side."""
    order = [s for s in ("live", "flat", "exit", "last_iterate") if s in frames]
    if not order:
        return None
    corr = (record.get("panel") or {}).get("corr") or {}
    corr_key = {
        "live": None,
        "flat": "live_vs_flat",
        "exit": "live_vs_exit",
        "last_iterate": "live_vs_last_iterate",
    }
    labels = dict(FRAME_STATES)
    vmax = max(float(np.percentile(frames[k], 99.5)) for k in order) or 1.0

    fig, axes = plt.subplots(1, len(order), figsize=(3.1 * len(order), 3.5))
    if len(order) == 1:
        axes = [axes]
    for ax, state in zip(axes, order):
        image = ax.imshow(frames[state], cmap="inferno", vmin=0.0, vmax=vmax)
        title = labels[state]
        key = corr_key[state]
        if key is not None:
            title = f"{title}\nρ(实测) = {fmt_num_or_dash(corr.get(key))}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("px")
        ax.set_ylabel("px")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{record.get('key')} {record.get('name')} — 退出分支 "
        f"{record.get('exit_branch')}，判定 {record.get('verdict')}",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def figure_correlations(summary: dict[str, Any], out_path: Path) -> Path | None:
    """Per-scenario optical correlation vs the repeat-frame noise floor."""
    records = [r for r in summary.get("scenarios") or [] if r.get("panel")]
    if not records:
        return None
    series = (
        ("live_vs_flat", "实测 vs 平场", "#4C78A8"),
        ("live_vs_exit", "实测 vs 退出相位", "#F58518"),
        ("live_vs_last_iterate", "实测 vs 最后迭代", "#54A24B"),
    )
    labels = [f"{r.get('key')}\n{r.get('verdict')}" for r in records]
    x = np.arange(len(records), dtype=float)
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(records) + 2.0), 3.8))
    for index, (key, label, color) in enumerate(series):
        values = [
            (r.get("panel") or {}).get("corr", {}).get(key) for r in records
        ]
        heights = [np.nan if v is None else float(v) for v in values]
        ax.bar(
            x + (index - 1) * width,
            heights,
            width,
            label=label,
            color=color,
        )

    floors = [
        (r.get("panel") or {}).get("corr", {}).get("flat_repeat") for r in records
    ]
    floor_heights = [np.nan if v is None else float(v) for v in floors]
    ax.scatter(
        x,
        floor_heights,
        marker="_",
        s=420,
        linewidths=1.8,
        color="black",
        label="重复帧噪声底（平场与平场重复帧）",
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("峰值归一后的相关系数 $\\rho$")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("面板实测画面与各候选状态的相关性")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _panel_state_text(record: dict[str, Any]) -> str:
    """One line naming what the controller says it displays."""
    panel = record.get("panel") or {}
    firmware = panel.get("firmware") or {}
    checks = record.get("checks") or {}
    if not firmware:
        return "未取得固件回读"
    slot = firmware.get("displayed_slot")
    gray = firmware.get("current_grayscale")
    if checks.get("firmware_mode") == "memory" and slot is not None:
        expected = checks.get("expected_slot")
        suffix = "" if expected is None else f"（退出写入槽 #{expected}）"
        return f"内存槽 #{slot}{suffix}"
    if gray is not None:
        return f"直接灰度模式，灰度值 {gray}"
    error = firmware.get("displayed_slot_error") or firmware.get("current_grayscale_error")
    return f"回读失败：{error}"


def _verdict_word(record: dict[str, Any]) -> str:
    verdict = record.get("verdict")
    return {"pass": "通过", "fail": "失败", "error": "未完成"}.get(verdict, str(verdict))


def _write_log_rows(record: dict[str, Any]) -> list[str]:
    """The tail of the run's write log, which is where the exit is visible."""
    lines: list[str] = []
    writes = record.get("writes") or {}
    for group, title in (("before", "注入前的最后几次写入"), ("after", "注入后的写入（即清理）")):
        rows = writes.get(group) or []
        if not rows:
            continue
        lines.append(f"**{title}**")
        lines.append("")
        lines.append("| # | 类型 | 内存槽 | 灰度范围 | 固定灰度 |")
        lines.append("|---|---|---|---|---|")
        for row in rows:
            span = (
                f"{row.get('min')}–{row.get('max')}"
                if row.get("min") is not None
                else "—"
            )
            lines.append(
                f"| {row.get('i')} | {row.get('kind')} | "
                f"{row.get('slot') if row.get('slot') is not None else '—'} | "
                f"{span} | {row.get('gray_value') if row.get('gray_value') is not None else '—'} |"
            )
        lines.append("")
    return lines


def _link(target: Path, report_dir: Path) -> str:
    """POSIX relative link from the report's directory to ``target``."""
    return Path(os.path.relpath(target, report_dir)).as_posix()


def _rel_or_abs(path: Path, base: Path = ROOT) -> str:
    """Repo-relative POSIX path when possible, absolute otherwise.

    A run directory may live outside the repository (a copy kept elsewhere), so
    the header must not raise ``ValueError`` just because it cannot be made
    relative.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _scenario_section(
    record: dict[str, Any],
    figures: dict[str, Path],
    run_dir: Path,
    report_dir: Path,
) -> list[str]:
    lines: list[str] = []
    key = record.get("key")
    lines.append(f"### 场景 {key} — {record.get('name')}")
    lines.append("")
    lines.append(f"{record.get('description') or ''}".strip())
    lines.append("")

    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    config = record.get("config") or {}
    pop_size = config.get("pop_size")
    lines.append(
        f"| 搜索配置 | algorithm=`{config.get('algorithm')}`，epochs={config.get('epochs')}，"
        f"pop_size={'—' if pop_size is None else pop_size}，"
        f"target_size={config.get('target_size')}，"
        f"n_max={config.get('n_max')} |"
    )
    injected = record.get("injected") or {}
    lines.append(
        f"| 注入的失败 | `{injected.get('kind')}`：{injected.get('detail') or '—'}"
        f"（触发：{'是' if injected.get('fired') else '否'}） |"
    )
    outcome = record.get("outcome") or {}
    lines.append(
        f"| 到达调用方的异常 | `{outcome.get('exception_type')}`：{outcome.get('exception_message') or '—'} |"
    )
    lines.append(f"| 退出分支 | `{record.get('exit_branch')}` |")
    lines.append(f"| 固件回读 | {_panel_state_text(record)} |")
    lines.append(f"| 从注入到返回 | {_fmt_optional(outcome.get('cleanup_s'), 2)} s |")
    lines.append(f"| 判定 | **{_verdict_word(record)}** |")
    if record.get("failed_keys"):
        lines.append(f"| 未通过的门禁 | {', '.join(f'`{k}`' for k in record['failed_keys'])} |")
    lines.append("")

    if record.get("error"):
        lines.append(f"> 该场景在采集前就出错：`{record['error']}`")
        lines.append("")

    checks = record.get("checks") or {}
    if checks:
        lines.append("**检查项**")
        lines.append("")
        gated = set(record.get("verdict_keys") or [])
        lines.append("| 检查 | 结果 | 说明 |")
        lines.append("|---|---|---|")
        for name, meaning in CHECK_MEANINGS.items():
            if name not in checks:
                continue
            value = checks[name]
            if value is None:
                rendered = "—"
            else:
                rendered = "✅" if value else "❌"
            gate = "（门禁）" if name in gated else ""
            lines.append(f"| `{name}`{gate} | {rendered} | {meaning} |")
        lines.append("")

    corr = (record.get("panel") or {}).get("corr") or {}
    if corr:
        lines.append("**光学比对**")
        lines.append("")
        lines.append("| 比较 | $\\rho$ |")
        lines.append("|---|---|")
        for name, label in (
            ("live_vs_flat", "实测 vs 平场 gray-0"),
            ("live_vs_exit", "实测 vs 退出写入的相位"),
            ("live_vs_last_iterate", "实测 vs 搜索最后一轮迭代"),
            ("flat_repeat", "平场与平场重复帧（噪声底）"),
        ):
            if name in corr:
                lines.append(f"| {label} | {fmt_num_or_dash(corr.get(name))} |")
        lines.append("")

    log_rows = _write_log_rows(record)
    if log_rows:
        lines.extend(log_rows)

    figure = figures.get(str(key))
    if figure is not None:
        lines.append(f"![{key} 面板状态]({_link(figure, report_dir)})")
        lines.append("")
    return lines


def render(
    summary: dict[str, Any],
    figures: dict[str, Path],
    correlation_figure: Path | None,
    run_dir: Path,
    report_dir: Path,
) -> str:
    records = summary.get("scenarios") or []
    lines: list[str] = []

    lines.append("# R-8 安全退出态的硬件验收（issue #145 / H-28）")
    lines.append("")
    lines.append(
        "本报告由 `scripts/slm/generate_pib_exit_safety_report.py` 从 "
        f"`{_rel_or_abs(run_dir)}/summary.json` 离线渲染；"
        "所有数值来自该目录下的产物，正文不手写数字。"
    )
    lines.append("")
    lines.append(
        "被验收的行为是：`slm_zernike_pib` 的搜索主体整体包在 "
        "`try: ... finally: _apply_best_on_exit()` 内，因此相机掉线、系数越界或 "
        "`KeyboardInterrupt` 都不会让面板停在搜索最后写入的任意图案上；"
        "清理自身失败时强制回到平场 gray-0。"
    )
    lines.append("")

    # ---- verdict table ---------------------------------------------------
    lines.append("## 结论")
    lines.append("")
    lines.append("| 场景 | 注入的失败 | 退出分支 | 固件回读 | 光学判定 | 清理耗时 (s) | 判定 |")
    lines.append("|---|---|---|---|---|---|---|")
    for record in records:
        checks = record.get("checks") or {}
        outcome = record.get("outcome") or {}
        injected = record.get("injected") or {}
        lines.append(
            f"| {record.get('key')} `{record.get('name')}` "
            f"| `{injected.get('kind') or '—'}` "
            f"| `{record.get('exit_branch')}` "
            f"| {_panel_state_text(record)} "
            f"| `{checks.get('optical_class')}` "
            f"| {_fmt_optional(outcome.get('cleanup_s'), 2)} "
            f"| **{_verdict_word(record)}** |"
        )
    lines.append("")

    passed = sum(1 for r in records if r.get("verdict") == "pass")
    failed = [r for r in records if r.get("verdict") != "pass"]
    if failed:
        lines.append(
            f"共 {len(records)} 个场景，{passed} 个通过、{len(failed)} 个未通过："
            + "、".join(f"`{r.get('key')}`" for r in failed)
            + "。未通过的场景逐项列在下方对应小节里。"
        )
    else:
        lines.append(
            f"共 {len(records)} 个场景，全部通过：每个失败场景都留下了可命名的面板状态，"
            "到达调用方的异常就是注入的那一个，清理没有吞掉它。"
        )
    lines.append("")

    # ---- acceptance criteria --------------------------------------------
    lines.append("## 验收口径")
    lines.append("")
    lines.append(
        "issue #145 列出六个观测点；下表把它们映射到本报告读取的字段。"
        "参与最终判定的检查项在各场景表格里标注为「门禁」，其余只作记录，"
        "避免把一个数字误当成已通过的门槛。"
    )
    lines.append("")
    lines.append("| 观测点 | 证据字段 | 覆盖场景 |")
    lines.append("|---|---|---|")
    for point, evidence, covered in OBSERVATION_POINTS:
        lines.append(f"| {point} | {evidence} | {covered} |")
    lines.append("")
    present = [str(r.get("key")) for r in records]
    missing = [key for key in "ABCDE" if key not in present]
    lines.append("本次运行包含的场景：" + "、".join(f"`{k}`" for k in present) + "。")
    if missing:
        lines.append("")
        lines.append(
            f"> 本次运行**没有**包含场景 {'、'.join(missing)}，"
            "依赖这些场景的观测点在本报告中不构成证据；需要时重跑采集脚本。"
        )
    lines.append("")
    gated = sorted({k for r in records for k in (r.get("verdict_keys") or [])})
    if gated:
        lines.append("参与判定的门禁项：" + "、".join(f"`{k}`" for k in gated) + "。")
        lines.append("")

    if correlation_figure is not None:
        lines.append(
            f"![面板实测画面与各候选状态的相关性]({_link(correlation_figure, report_dir)})"
        )
        lines.append("")

    # ---- per scenario ----------------------------------------------------
    lines.append("## 场景与结果")
    lines.append("")
    for record in records:
        lines.extend(_scenario_section(record, figures, run_dir, report_dir))

    # ---- division of labour with the offline suite ----------------------
    lines.append("## 与离线测试的分工")
    lines.append("")
    lines.append(
        "`tests/ao_shaping/optimizer/wfless/test_slm_zernike_pib_hardware_safety.py` "
        "在数字孪生上证明了控制流：`finally` 一定执行、调用顺序正确、"
        "清理自身失败时回落到 `set_grayscale(0)`、原始异常存活。它证明不了面板状态，因为假驱动没有固件："
    )
    lines.append("")
    lines.append(
        "1. `display_data(gray, memory_number=slot)` 在 `slot` 正是当前显示槽时是**空操作**，"
        "LCOS 不刷新。清理「写了最佳相位」可能写进了空处。本报告用写入日志（`exit_used_a_fresh_slot`）"
        "和光学比对两条独立证据核查这一点。"
    )
    lines.append(
        "2. `set_grayscale(0)` 是否真的让面板成为平场，是设备事实，只能在实验系统上读取。"
    )
    lines.append(
        "3. 真实拆除路径上的 `close()` 若在异常展开时抛错，会覆盖原始异常；"
        "只有走真实驱动才能观察到这一点。"
    )
    lines.append("")

    # ---- reproduction ----------------------------------------------------
    lines.append("## 复现")
    lines.append("")
    camera = summary.get("camera") or {}
    slm = summary.get("slm") or {}
    lines.append("```bash")
    lines.append("# 1) 采集（需要设备；SLM 用内存模式，且没有别的进程占用 USB 句柄）")
    lines.append("python scripts/slm/verify_pib_exit_safety_hw.py")
    lines.append("# 2) 离线渲染本报告")
    lines.append("python scripts/slm/generate_pib_exit_safety_report.py")
    lines.append("```")
    lines.append("")
    lines.append("| 实验系统 | 值 |")
    lines.append("|---|---|")
    lines.append(
        f"| SLM | #{slm.get('slm_number')}，video_mode={slm.get('video_mode')}，"
        f"λ={slm.get('wavelength_nm')} nm，2π 对应灰度 {slm.get('max_gray_for_2pi')} |"
    )
    baseline = summary.get("baseline_slot")
    lines.append(
        f"| 会话开始时的显示槽 | {baseline if baseline is not None else '直接灰度模式（回读错误码 1 属正常）'} |"
    )
    lines.append(
        f"| 相机 | {camera.get('type')} #{camera.get('cam_id')}，"
        f"画幅 {camera.get('frame_shape')}，验收曝光 "
        f"{_fmt_optional(camera.get('verification_exposure_ms'), 3)} ms |"
    )
    lines.append(f"| 产物目录 | `{_rel_or_abs(run_dir)}` |")
    lines.append(f"| 采集时间 | {summary.get('started_iso')} → {summary.get('finished_iso')} |")
    lines.append("")
    lines.append(
        "采集脚本把两个**搜索质量门禁**关掉了（`fold_ratio=0`、`noise_gate_k=-1`）。"
        "它们只决定一步梯度更新是否被提交，与退出路径无关；但本实验系统的平场远场是散斑，"
        "逐轮峰值在 2–4 倍间起伏，门禁会把每一轮都判成亮度折叠，搜索因此没有任何迭代可回填，"
        "「停在最佳相位」这条分支就永远走不到。关掉它们是为了让搜索真的移动面板。"
    )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "本文件由脚本生成，请勿手工编辑；溯源块由 "
        "`python scripts/sync_report_provenance.py` 插入。"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render the offline report for the issue #145 (H-28) exit-safety "
            "acceptance from the artefacts written by verify_pib_exit_safety_hw.py."
        )
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Artifact run directory (default: newest run_* carrying summary.json).",
    )
    parser.add_argument(
        "--out",
        default=str(OUT_DIR / Path(REPORT_KEY).name),
        help="Output markdown path.",
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip the figures.")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run()
    if run_dir is None or not (run_dir / "summary.json").is_file():
        print(
            "no artifact run with summary.json under "
            f"{ARTIFACT_ROOT}; run scripts/slm/verify_pib_exit_safety_hw.py first"
        )
        return 1

    summary = load_summary(run_dir)
    stamp = run_dir.name.removeprefix("run_")
    logger.info("rendering {} scenario(s) from {}", len(summary.get("scenarios") or []), run_dir)

    figures: dict[str, Path] = {}
    correlation_figure: Path | None = None
    if not args.no_figures:
        for record in summary.get("scenarios") or []:
            frames = load_frames(record, run_dir)
            if not frames:
                continue
            path = figure_scenario_panels(
                record, frames, FIG_DIR / f"{stamp}_{record.get('key')}_panel.png"
            )
            if path is not None:
                figures[str(record.get("key"))] = path
        correlation_figure = figure_correlations(
            summary, FIG_DIR / f"{stamp}_correlations.png"
        )

    out = Path(args.out)
    text = render(summary, figures, correlation_figure, run_dir, out.parent)
    # This generator overwrites the same report path on every run, so it must
    # re-stamp the provenance block: otherwise a re-render deletes it and
    # ``sync_report_provenance.py --check`` reports drift on a correct file.
    text = insert_header(text, REPORT_KEY)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    for path in list(figures.values()) + ([correlation_figure] if correlation_figure else []):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
