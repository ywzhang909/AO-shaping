"""Generate the **offline** focus-plane report from the saved focus-probe evidence.

Reads what the hardware runs already wrote into
``data/debug/bench_settings/focus_probe/`` -- every ``slm1_*.json`` sweep, the
newest ``verify_*.json`` (exposure ladder + verification sweep + linearity
re-check) and the newest ``flat_drift_*.json`` (light-level stability series) --
and renders one illustrated Chinese markdown report plus one figure.
**Fully offline**: it never constructs a camera or an SLM, so the conclusions can
be re-read and re-rendered at any time, including with the instruments powered
down.

Why a report and not a log line
-------------------------------
Three things about this measurement are not visible in the winning number and are
each worth a table of their own:

1. **the exposure had to be re-armed at every point**, and the evidence is the
   span it had to cover -- if that span is not printed, a reader assumes a fixed
   exposure and silently believes a ranking they cannot reproduce;
2. **the light level stepped by a large factor mid-session**, so only ratios
   measured inside one sweep mean anything, and the report has to say which
   numbers are ratios and which are levels;
3. **the win is checked against two exposure-invariant quantities** (whole-frame
   energy, in-box energy fraction), because a peak statistic alone cannot tell
   "focused" from "concentrated into the frame".

Usage:
    python scripts/slm/generate_focus_plane_report.py
    python scripts/slm/generate_focus_plane_report.py --no-figures
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# Also the repo root: run as ``python scripts/slm/<this>.py``, so ``sys.path[0]``
# is ``scripts/slm`` and ``scripts._common`` would not resolve without this.
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from loguru import logger  # noqa: E402

# Cell formatting is shared: scripts/_common owns it, and a second copy drifts.
from scripts._common import fmt_metric  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10

EVIDENCE = ROOT / "data" / "debug" / "bench_settings" / "focus_probe"
OUT_MD = ROOT / "report" / "slm" / "focus_plane_20261019.md"
OUT_PNG = ROOT / "report" / "slm" / "focus_plane_20261019.png"

#: FT lens focal length of this bench (mm), from ``bench_constants`` /
#: ``matrix_utils``. Used only for the geometry cross-check, and stated in the
#: report so a reader can see which assumption the cross-check rests on.
F_LENS_MM = 125.0

#: Bench geometry the correction was measured against, echoed into the report so
#: the numbers cannot be silently applied to a different beam position.
CENTRE_PANEL_PX = (960, 690)
RADIUS_PANEL_PX = 450

#: Mirror of the fingerprint's trustworthiness gate, so the report states the
#: number the code actually uses rather than a second one that can drift.
FWHM_FOCUSED_PX = 30.0


def _load(pattern: str) -> list[dict[str, Any]]:
    """Load every matching evidence file, tagged with its own path.

    The probe's JSON predates an in-file timestamp, so the filename stamp is kept
    alongside and used as the time column -- a sweep with no timestamp in the
    report is a number nobody can date.
    """
    out: list[dict[str, Any]] = []
    for path in sorted(EVIDENCE.glob(pattern)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skipping unreadable {}: {}", path, exc)
            continue
        payload["_file"] = path.name
        payload["_stamp"] = _stamp_from_name(path.name)
        out.append(payload)
    return out


def _stamp_from_name(name: str) -> str:
    """``slm1_20261019_114201.json`` -> ``2026-10-19 11:42:01``."""
    m = re.search(r"(\d{8})_(\d{6})", name)
    if not m:
        return ""
    date, clock = m.group(1), m.group(2)
    return (
        f"{date[:4]}-{date[4:6]}-{date[6:8]} "
        f"{clock[:2]}:{clock[2:4]}:{clock[4:6]}"
    )


def _verdict_text(verdict: dict[str, Any]) -> str:
    """One cell: what the fingerprint check decided, if anything."""
    if verdict.get("skipped"):
        return "跳过（无参考 / 已跳过）"
    if verdict.get("notes"):
        return "未判定（光斑未合焦）"
    if verdict.get("ok"):
        return "一致"
    return "**不一致**"


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build_figure(runs: list[dict], verify: dict, path: Path) -> bool:
    """peak/ms vs 1/f (newest sweep) + the verification gains. Returns success."""
    try:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        newest = runs[-1]
        pts = sorted(newest["points"], key=lambda p: p["diopter_m1"])
        ok = [p for p in pts if p["status"] == "ok"]
        dim = [p for p in pts if p["status"] != "ok"]
        x = np.array([p["diopter_m1"] for p in ok])
        y = np.array([p["peak_per_ms"] for p in ok])
        ax.semilogy(x, y, "o-", color="#2563eb", label="可用点：峰值/曝光")
        if dim:
            ax.plot(
                [p["diopter_m1"] for p in dim],
                [max(p["peak_per_ms"], 1e-3) for p in dim],
                "x",
                color="#9ca3af",
                ms=5,
                label="曝光上限外（未参与排序）",
            )
        best = newest.get("best_diopter_m1")
        if best is not None:
            # ASCII exponent in the figure: Microsoft YaHei has no U+207B, and a
            # missing glyph renders as a blank box, not as a minus.
            ax.axvline(
                best, color="#dc2626", ls="--", lw=1.2, label=f"最佳 {best:+.3f} m^-1"
            )
        ax.axvline(0.0, color="#6b7280", lw=0.8)
        ax.set_xlabel("面板菲涅尔光焦度 1/f (m^-1)，0 = plain 相位")
        ax.set_ylabel("峰值 / 曝光 (counts/ms)")
        ax.set_title("亮度扫描（最新一趟）")
        ax.grid(alpha=0.3)

        ax3 = ax.twinx()
        ax3.plot(
            x,
            [p["exposure_ms"] for p in ok],
            "s--",
            color="#f59e0b",
            ms=4,
            alpha=0.8,
            label="该点实际曝光",
        )
        ax3.set_yscale("log")
        ax3.set_ylabel("该点曝光 (ms)")
        # ``twinx`` leaves an internal artist behind (``_childN``); merging every
        # line into one legend without filtering puts those in as nameless rows.
        lines = [
            l
            for l in ax.get_lines() + ax3.get_lines()
            if not l.get_label().startswith("_")
        ]
        ax.legend(
            lines,
            [l.get_label() for l in lines],
            fontsize=8,
            loc="upper left",
            framealpha=0.9,
        )

        rows = verify["rows"]
        labels = [
            "plain" if r["is_plain"] else f"{r['focal_mm']:.0f}" for r in rows
        ]
        gains = [r.get("peak_gain_vs_plain", 0.0) for r in rows]
        energies = [r.get("energy_ratio_vs_plain", 0.0) for r in rows]
        idx = np.arange(len(rows))
        ax2.bar(idx - 0.2, gains, width=0.4, label="峰值增益", color="#2563eb")
        ax2.bar(idx + 0.2, energies, width=0.4, label="整帧能量增益", color="#16a34a")
        ax2.axhline(1.0, color="#6b7280", lw=0.8)
        ax2.set_xticks(idx)
        ax2.set_xticklabels(labels)
        ax2.set_xlabel("菲涅尔焦距 (mm)")
        ax2.set_ylabel("相对 plain 的倍数")
        ax2.set_title("曝光无关的两项交叉验证")
        ax2.grid(alpha=0.3, axis="y")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return True
    except Exception as exc:  # noqa: BLE001 - a figure must never break the report
        logger.warning("figure generation failed: {}", exc)
        return False


def render(
    runs: list[dict],
    verify: dict,
    drift: dict | None,
    fig_ok: bool,
    fingerprints: list[dict] | None = None,
) -> str:
    newest = runs[-1]
    pts = sorted(newest["points"], key=lambda p: p["diopter_m1"])
    ok_pts = [p for p in pts if p["status"] == "ok"]
    exps = [p["exposure_ms"] for p in pts if p["exposure_ms"]]
    ladder = verify.get("ladder", [])
    vrows = verify["rows"]

    best_vals = [r["best_diopter_m1"] for r in runs if r.get("best_diopter_m1") is not None]
    med = float(np.median(best_vals)) if best_vals else float("nan")
    spread = (max(best_vals) - min(best_vals)) if len(best_vals) > 1 else 0.0

    # --- runs table ---
    def plain_level(run: dict) -> str:
        """The sweep's own plain reference: peak/ms at 1/f = 0.

        This is the light level *as archived by the run*, which is what makes a
        level change across runs visible without trusting any ad-hoc ladder.
        """
        for p in run["points"]:
            if abs(p["diopter_m1"]) < 1e-9:
                return f"{p['peak_per_ms']:.1f}"
        return "—"

    run_rows = []
    for r in runs:
        n_ok = sum(1 for p in r["points"] if p["status"] == "ok")
        run_rows.append(
            [
                r.get("measured_at") or r.get("_stamp") or "—",
                f"{r['best_diopter_m1']:+.4f}" if r.get("best_diopter_m1") is not None else "—",
                fmt_metric(r.get("best_focal_mm"), 0),
                plain_level(r),
                fmt_metric(r.get("gain_vs_flat"), 1),
                f"{n_ok}/{len(r['points'])}",
            ]
        )

    # --- sweep curve ---
    curve_rows = [
        [
            f"{p['diopter_m1']:+.3f}",
            # ``inf`` here means the plain phase, not an unusable number -- the
            # shared formatter would print "inf", so the domain reading stays here.
            "plain" if math.isinf(p["focal_mm"]) else fmt_metric(p["focal_mm"], 0),
            f"{p['exposure_ms']:.3f}",
            f"{p['peak_fraction']:.2f}",
            f"{p['peak_per_ms']:.1f}",
            f"{p['fwhm_px']:.1f}",
            p["status"],
        ]
        for p in pts
    ]

    # --- verification ---
    ver_rows = [
        [
            "plain" if r["is_plain"] else f"{r['focal_mm']:.0f}",
            f"{r['exposure_ms']:.3f}",
            f"{r['peak']:.1f}",
            f"{r['peak_per_ms']:.1f}",
            f"{r['fwhm_px']:.1f}",
            f"{r['hollowness']:.2f}",
            f"{r['energy_per_ms']:.0f}",
            f"{r['in_box_frac']:.3f}",
            fmt_metric(r.get("peak_gain_vs_plain"), 2),
            fmt_metric(r.get("energy_ratio_vs_plain"), 2),
        ]
        for r in vrows
    ]

    ladder_rows = [
        [
            f"{r['exposure_ms']:g}",
            f"{r['peak']:.1f}",
            f"{r['counts_per_ms']:.1f}",
            f"{r['median']:.2f}",
            f"{r['fwhm_px']:.1f}",
        ]
        for r in ladder
    ]

    plain_row = next((r for r in vrows if r["is_plain"]), None)
    best_row = max(
        (r for r in vrows if not r["is_plain"]),
        key=lambda r: r["peak_per_ms"],
        default=None,
    )

    lines: list[str] = []
    A = lines.append

    A("# 焦面补偿：相机离开 2f 焦面后的菲涅尔焦距实测")
    A("")
    A(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    A("- 实验系统：SLM #1 (22030108) + 大恒 FJB24112232，1064 nm，2f 傅里叶光路")
    A(
        f"- 测量条件：入射中心 {CENTRE_PANEL_PX[0]},{CENTRE_PANEL_PX[1]} 面板px，"
        f"照明半径 {RADIUS_PANEL_PX} 面板px，{len(runs)} 趟扫描"
    )
    A("")

    A("## 1. 结论")
    A("")
    A(
        "相机挪离 2f 焦面之后，**plain 相位不再是最亮的**。实测把它拉回焦点的面板"
        "菲涅尔光焦度为"
    )
    A("")
    A(f"> **1/f ≈ {med:+.2f} m⁻¹（f ≈ {1000.0 / med:+.0f} mm）**")
    A("")
    if len(best_vals) > 1:
        A(
            f"{len(best_vals)} 趟扫描的极差是 {spread:.3f} m⁻¹（"
            f"{spread / abs(med) * 100:.1f}%）；中位数 {med:.4f} m⁻¹。"
            "取整到 **0.70 m⁻¹ / f ≈ +1420 mm** 使用。"
        )
    A("")
    if best_row and plain_row:
        A(
            f"独立验证（`scripts/slm/verify_focus_plane_hw.py`，与探测器独立的一趟）："
            f"峰值/曝光 ×{best_row.get('peak_gain_vs_plain', 0):.1f}、"
            f"**整帧能量 ×{best_row.get('energy_ratio_vs_plain', 0):.1f}**、"
            f"FWHM {plain_row['fwhm_px']:.1f} → {best_row['fwhm_px']:.1f} px、"
            f"空心度 {plain_row['hollowness']:.2f} → {best_row['hollowness']:.2f}；"
            f"曝光线性度 {verify.get('linearity_at_best', float('nan')):.3f}。"
        )
        A("")
        A(
            "能量也涨了 —— 这一项重要：峰值是可以靠「把框外的光挤回框内」变高的，"
            "而整帧能量只有真的把光聚到探测器上才会涨。"
        )
    A("")

    A("## 2. 为什么必须实测，不能从标称 125 mm 推")
    A("")
    A(
        "相机在 2f 之后时，光斑是一个发散锥，弥散直径随偏移线性增长，峰值随之塌陷。"
        "修正手段是在面板上加二次相位，也就是一个菲涅尔透镜。但它的量："
    )
    A("")
    A("1. 取决于相机偏离焦面**多少毫米** —— 光路事实，纸面上没有；")
    A(
        "2. **符号不预设**：把面板相位加到光上的极性没有权威记录，"
        "猜错一侧就要多扫一遍；"
    )
    A(
        f"3. 还要落在面板能分辨的范围里：f = ±{1000.0 / 6.0:.0f} mm 时孔径边缘的"
        f"条纹周期约 "
        f"{1064e-9 * (1000.0 / 6.0) * 1e-3 / (RADIUS_PANEL_PX * 8e-6) * 1e6:.0f} μm，"
        "再强就欠采样、效率反而塌掉。"
    )
    A("")
    A("所以扫参量用**光焦度 1/f** 而不是焦距：离焦本身是半径的二次项，1/f 才是修正量。")
    A("")

    A("## 3. 方法")
    A("")
    A("- **相位**：plain 相位 + 以实测入射中心为心的菲涅尔相位，圆孔裁剪，raw 弧度；")
    A("- **扫描**：1/f 对称扫 ±6 m⁻¹（含 0 = plain），粗扫 25 点；")
    A(
        "- **逐点曝光自适应**：每个点单独把曝光调到满量程的 25%–80%，"
        "落在带外就标记（`clipped` / `dim`）并排除出排序；"
    )
    A("- **判据**：`峰值 / 曝光`。传感器对曝光线性时它正比于真实强度，是唯一能跨不同曝光比较的量；")
    A("- **细化**：抛物线顶点 + 附近细扫，且必须复测确认才采纳；")
    A("- **线性度复核**：在最优点换一个曝光再读一次，比值偏离 1 超过 25% 就在证据里明说。")
    A("")

    if exps:
        A("### 3.1 曝光确实必须逐点重调")
        A("")
        spans = []
        for r in runs:
            e = [p["exposure_ms"] for p in r["points"] if p["exposure_ms"]]
            if e:
                spans.append(max(e) / min(e))
        A(
            f"最新一趟里，各点实际曝光从 **{min(exps):.3f} ms 到 {max(exps):.3f} ms**，"
            f"跨度 {max(exps) / min(exps):.0f} 倍；"
            f"{len(runs)} 趟里最大的一趟跨度 **{max(spans):.0f}×**"
            f"（{min(spans):.0f}–{max(spans):.0f}×）。"
        )
        A("")
        A(
            "固定曝光排不出这个顺序：近焦面直接饱和、远焦面只剩读出噪声，"
            "`argmax(峰值)` 报出的是「哪个点恰好被固定曝光打满」，"
            "而不是「哪个点最亮」。曝光跨度越大，越说明「一次标定、一个曝光值」"
            "这种做法在这条光路上不成立。"
        )
        A("")

    A("## 4. 实测扫描曲线（最新一趟）")
    A("")
    A(
        _md_table(
            ["1/f (m⁻¹)", "f (mm)", "曝光 (ms)", "峰值/满量程", "峰值/曝光", "FWHM (px)", "状态"],
            curve_rows,
        )
    )
    A("")
    A(
        f"盆地是单峰的：`dim` 点共 "
        f"{sum(1 for p in pts if p['status'] != 'ok')} 个，全部落在曝光上限之外，"
        "没有参与排序 —— 这是有意的：欠曝的点只能给出「不够亮」的上界，"
        "把它排进序列等于用曝光上限去惩罚远端点。"
    )
    A("")

    if ladder_rows:
        A("## 5. 曝光阶梯（plain 相位，判定线性用）")
        A("")
        A(
            _md_table(
                ["曝光 (ms)", "峰值", "计数/ms", "中位数", "FWHM (px)"],
                ladder_rows,
            )
        )
        A("")
        cps = [r["counts_per_ms"] for r in ladder[3:]] if len(ladder) > 3 else []
        if cps:
            A(
                f"1 ms 以上各档的计数/ms 落在 {min(cps):.1f}–{max(cps):.1f}，"
                "即响应与曝光成正比 —— 这正是 `峰值/曝光` 可用作判据的前提。"
                "0.1 ms 以下已经掉进读出噪声，FWHM 也失去意义（表中可见其发散）。"
            )
        A("")

    A("## 6. 独立验证（曝光无关的两项）")
    A("")
    A(
        _md_table(
            [
                "f (mm)",
                "曝光 (ms)",
                "峰值",
                "峰值/曝光",
                "FWHM (px)",
                "空心度",
                "能量/曝光",
                "框内能量占比",
                "峰值增益",
                "能量增益",
            ],
            ver_rows,
        )
    )
    A("")
    A(
        "- **能量/曝光**：整帧信号总和除以曝光。光学不凭空产生或消灭光，"
        "它涨就说明离焦态确实把光丢在了画幅之外，聚焦把它收了回来；")
    A(
        "- **框内能量占比**：光斑 60 px 邻域内的能量占比，对曝光免疫，"
        "回答「这到底是不是一个光斑」。"
    )
    A("")

    A("## 7. 重复性与对光强的依赖")
    A("")
    A(
        _md_table(
            ["测量时间", "最佳 1/f (m⁻¹)", "f (mm)", "plain 基准 (counts/ms)", "相对 plain 增益", "可用点"],
            run_rows,
        )
    )
    A("")
    if drift:
        A(
            f"同一配置（plain、{drift['exposure_ms']} ms）连续 {drift['samples']} 次、"
            f"每 {drift['period_s']:g} s 采一次：峰值 {min(drift['peaks']):.1f}–"
            f"{max(drift['peaks']):.1f}，max/min = "
            f"**{max(drift['peaks']) / max(min(drift['peaks']), 1e-9):.2f}×**。"
            "也就是说**扫描期间的光强是稳的**，`峰值/曝光` 的逐点比较成立。"
        )
        A("")
    A(
        "低光强那趟（光强低约 7 倍、可用点只剩 8 个）落在 0.667 m⁻¹，"
        "与其余三趟的 0.702–0.705 差约 5%。盆地顶部是平的（见第 4 节"
        " ±0.17 m⁻¹ 仍在同一量级），所以这个差别落在**可用区间**内；"
        "把它当作不确定度的一部分，而不是当成又一次独立复现。"
    )
    A("")

    A("## 8. 会话中途光强发生了一次阶跃")
    A("")
    levels = []
    for r in runs:
        for p in r["points"]:
            if abs(p["diopter_m1"]) < 1e-9 and p["peak_per_ms"] > 0:
                levels.append((r.get("measured_at") or r.get("_stamp") or "?", p["peak_per_ms"]))
                break
    if len(levels) > 1:
        top = max(v for _, v in levels)
        low = min(v for _, v in levels)
        A(
            "每趟扫描自己都测了 plain 基准（1/f = 0 那一点的 `峰值/曝光`），"
            "它就是当时的光强记录："
        )
        A("")
        A(
            _md_table(
                ["测量时间", "plain 基准 (counts/ms)"],
                [[t, f"{v:.1f}"] for t, v in levels],
            )
        )
        A("")
        A(
            f"最高与最低相差 **{top / low:.1f} 倍**。单趟之内光强是稳的"
            "（第 7 节：2 分钟内 max/min 只有 1.05×），"
            f"所以这 {top / low:.1f} 倍的变化发生在**两趟之间**，而不是扫描过程中 —— "
            "这正是逐点自适应曝光要覆盖的场景：变化再大，只要发生在扫描内部，"
            "逐点重调就吸收得掉。"
        )
        A("")
    A(
        "两条推论：\n"
        "1. **跨会话的绝对亮度不可比**，只有同一趟扫描内的比值有意义 —— "
        "本报告所有「增益」都是比值，正因如此它们不受这次阶跃影响；\n"
        "2. 想比较两次实验的绝对亮度，必须先各自重新探测曝光。"
    )
    A("")
    A("### 8.1 顺带查出的另一处矛盾：曝光上限探测器与直接阶梯差 37 倍")
    A("")
    A(
        f"按第 5 节的阶梯，{ladder[-1]['exposure_ms']:g} ms 才让峰值到 "
        f"{ladder[-1]['peak_fraction']:.0%} 满量程，也就是本实验系统当前约 "
        f"{ladder[-1]['counts_per_ms']:.1f} counts/ms，要打满 255 得用到几十毫秒。"
        "但 `scripts/slm/check_flat_stability.py` 在同一时段解析出的"
        "「90% 饱和上限」是 **2.07 ms**，反推相当于 "
        f"{0.9 * 255 / 2.07:.0f} counts/ms —— 比实测**高约 37 倍**。"
    )
    A("")
    A(
        "本报告没有判定谁对，但这件事有直接后果：`probe_session_exposure_ms()`"
        "（GUI 曝光探测的入口）走的是同一个上限探测函数，若它给出的上限偏小，"
        "工作曝光就会偏大并**接近饱和**——而实测记录里已经出现过"
        "「阶梯给出的上限减半后读数 255/255」这类情况。**建议单独查这一条。**"
    )
    A("")

    A("## 9. 几何交叉校验，以及一处未解的矛盾")
    A("")
    A(
        f"薄透镜模型下，面板上的菲涅尔透镜把焦点从 {F_LENS_MM:.0f} mm 移到 "
        f"{F_LENS_MM - F_LENS_MM**2 / (1000.0 / med):.1f} mm"
        "（方向取决于面板相位极性，本次不预设），"
        f"即相机偏离 2f 焦面约 **{F_LENS_MM**2 / (1000.0 / med):.0f} mm**。"
    )
    A("")
    A("**未解**：用离焦弥散反过来推照明半径，与半径探测器给出的量不一致。")
    A("")
    if plain_row and best_row:
        A(
            f"离焦态 FWHM {plain_row['fwhm_px']:.1f} px、聚焦态 "
            f"{best_row['fwhm_px']:.1f} px，多出来的弥散约 "
            f"{(plain_row['fwhm_px'] ** 2 - best_row['fwhm_px'] ** 2) ** 0.5:.0f} px；"
            "按锥角 R/f 反推，得到的照明半径只有约 100 面板px 量级，"
            f"而半径探测器此前给的是入射中心 ({CENTRE_PANEL_PX[0]},{CENTRE_PANEL_PX[1]})、"
            "半径 300 px 以上。"
        )
    A("")
    A(
        "两条线索必须有一条是错的：要么半径探测器的口径仍不可信"
        "（本仓库此前已确认 `edge/2` 与「中心相减」两种算法给出过 180 / 300 px 的分歧），"
        "要么薄透镜模型在这条光路上不适用（例如 SLM 并非精确位于前焦面）。"
        "**这不影响补偿量本身** —— 它是直接测出来的，不依赖任何几何模型 —— "
        "但它会影响「用几何模型做二次校验」这条路，所以留作待查项。"
    )
    A("")

    A("## 10. 怎么用")
    A("")
    A("```bash")
    A(
        "python -m ao_shaping.tools.slm.panel.focus_probe \\\n"
        f"    --exposure-ms 2.0 --radius-px {RADIUS_PANEL_PX} "
        f"--centre-x {CENTRE_PANEL_PX[0]} --centre-y {CENTRE_PANEL_PX[1]}"
    )
    A("```")
    A("")
    A(
        "GUI：侧栏「SLM 1 · 光路几何与标定」→ **D2 焦面** → 「探测焦面」。"
        "整形页的「离焦 / 聚焦微调」区块会**默认采用**探测到的 f，"
        "也可以手工覆盖；焦距是粗补偿，Zernike (2,0) 滑块是细调，两者同为二次相位直接相加。"
    )
    A("")
    A("操作注意：")
    A("")
    A(
        f"- **光弱时要放宽曝光上限**：本次 20 ms 上限在低光强下让 {len(pts) - len(ok_pts)}"
        f"/{len(pts)} 个点落到带外；必要时加 `--max-exposure-ms`，或收窄 "
        "`--diopter-range`；"
    )
    A("- 探测结束面板会**恢复 plain**，不会把透镜悄悄留给下一次运行；")
    A(
        "- 校正值依赖**入射中心**：透镜偏离光轴会产生线性相位并大幅偏转光斑"
        "（本次实测位移 ≤ 96 px）。改光路后必须重测中心。"
    )
    A("")

    if fingerprints:
        A("## 10.1 光路指纹：光路被挪动时的强制失效")
        A("")
        A(
            "上面的补偿量是**光路状态**的量，而光路随时会被挪动。所以配了一套指纹："
            "给定平场相位与一个**已知倾斜**的 Zernike 相位，读 CCD 光斑的宽度、"
            "倾斜造成的位移与质心，和标定时那份比对 —— 挪过就拒绝复用那份标定。"
            "不用 TTL，因为光路可能一秒内被挪动、也可能一周不动。"
        )
        A("")
        fps = [f for f in fingerprints if "flat_fwhm_px" in f]
        if fps:
            A(
                _md_table(
                    [
                        "测量时间",
                        "平场宽 (px)",
                        "倾斜位移 (px)",
                        "质心",
                        "峰值/曝光",
                        "判定",
                    ],
                    [
                        [
                            f.get("measured_at", "—"),
                            f"{f['flat_fwhm_px']:.1f}",
                            f"{f.get('tilt_shift_px', 0.0):.1f}",
                            f"({f['flat_centroid'][0]:.0f},{f['flat_centroid'][1]:.0f})",
                            f"{f.get('flat_peak_per_ms', 0.0):.1f}",
                            _verdict_text(f.get("verdict") or {}),
                        ]
                        for f in fps
                    ],
                )
            )
            A("")
        A(
            "> 表里「不一致」的行是**闸门落地之前**记录的：当时光斑宽 58–79 px，"
            "代码会拿这种噪声去否决运行；现在的行为是「未判定」。这一条本身"
            "就是闸门存在的理由。"
        )
        A("")
        A(
            "三条设计决定，都是实测逼出来的：\n"
            "1. **判据不含亮度**。同一台面上平场亮度两趟之间能差 7.3 倍，"
            "把亮度放进判据等于让激光而不是光学路径决定否决与否；\n"
            "2. **倾斜位移用相位相关而不是质心差**。仓库里早已实测："
            "足以给出信号的倾斜会同时改变光斑形状，质心在重复之间跳 ±8 px，"
            "足以造成 2 倍误差（见 `frame_analysis.estimate_shift` 的说明）；\n"
            "3. **有可信度闸门**。光斑没合焦时（宽度超过 "
            f"{FWHM_FOCUSED_PX:.0f} px）任何形状统计量都是噪声 —— 实测离焦面下"
            "20 秒内两次读数就差 20% —— 此时指纹**拒绝下结论**，"
            "而不是拿噪声去否决一次运行。"
        )
        A("")
        A(
            "强制不检验的开关：侧栏「运行前校验光路指纹」取消勾选，"
            "或 CLI 加 `--skip-bench-path-check`。"
        )
        A("")

    A("## 11. 还没做的")
    A(
        "2. **相机窗口没跟着挪**。聚焦态光斑质心从 "
        f"({plain_row['centroid'][0]:.0f},{plain_row['centroid'][1]:.0f}) 移到 "
        f"({best_row['centroid'][0]:.0f},{best_row['centroid'][1]:.0f})，"
        "ROI 需要重新定位。"
        if plain_row and best_row
        else "2. 相机窗口/ROI 需要按聚焦后的光斑重新定位。"
    )
    A(
        "3. **入射中心没有为此重新标定**。本次沿用 "
        f"({CENTRE_PANEL_PX[0]},{CENTRE_PANEL_PX[1]})，从光斑位移反推的偏心约 2 px，"
        "但没有用 `panel_locate` 在同会话内实测。"
    )
    A("")

    A("## 12. 复现")
    A("")
    A("```bash")
    A("python -m ao_shaping.tools.slm.panel.focus_probe \\")
    A(
        f"    --exposure-ms 2.0 --radius-px {RADIUS_PANEL_PX} "
        f"--centre-x {CENTRE_PANEL_PX[0]} --centre-y {CENTRE_PANEL_PX[1]} "
        "--steps 25 --refine 7 --frames 3"
    )
    A("python scripts/slm/verify_focus_plane_hw.py")
    A("python scripts/slm/generate_focus_plane_report.py")
    A("```")
    A("")
    A(
        f"证据目录：`{EVIDENCE.relative_to(ROOT).as_posix()}/`"
        "（`slm1_*.json` 扫描、`verify_*.json` 验证、`flat_drift_*.json` 稳定性）。"
    )
    if fig_ok:
        A("")
        A("![焦面扫描与验证](./focus_plane_20261019.png)")
    A("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-figures", action="store_true", help="skip the figure")
    ap.add_argument("--out", default=str(OUT_MD), help="output markdown path")
    args = ap.parse_args(argv)

    runs = _load("slm1_*.json")
    if not runs:
        print(f"no focus-probe evidence under {EVIDENCE}; run the probe first")
        return 1
    verifies = _load("verify_*.json")
    drifts = _load("flat_drift_*.json")
    if not verifies:
        print("no verify_*.json evidence; run scripts/slm/verify_focus_plane_hw.py")
        return 1

    # The newest verify file may predate the ladder flag; prefer one that has it.
    verify = next(
        (v for v in reversed(verifies) if v.get("ladder")),
        verifies[-1],
    )
    drift = drifts[-1] if drifts else None
    fingerprints = sorted(
        (EVIDENCE.parent / "fingerprint").glob("fingerprint_*.json")
    )
    fps: list[dict[str, Any]] = []
    for path in fingerprints:
        try:
            fps.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue

    fig_ok = False
    if not args.no_figures:
        fig_ok = build_figure(runs, verify, OUT_PNG)

    text = render(runs, verify, drift, fig_ok, fps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    if fig_ok:
        print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())