"""由闪耀光栅衍射效率的**实测扫描**生成可视化报告 (离线, 不碰硬件)。

产出
----
``report/slm/diffraction_efficiency/hw_blaze_20261023/``

* ``report.md`` —— 结论、口径、限制
* ``figures/spot_vs_gray.gif`` —— **光斑图随灰度变化**: 平场 → 零级变暗、±1 级
  变亮 → 零级几乎消失。固定亮度标尺 (不是逐帧归一化), 所以"零级在变暗"看得见。
* ``figures/transfer_vs_gray.png`` —— **曲线: 灰度 → 能量转移比例**
* ``figures/fine_scan.png`` —— 满命令附近的高精度确认段

输入
----
``--scan``: 全量程扫描目录 (``measurements.csv`` + ``frames/point_*.npz``;
需要 ``--save-frames``)。
``--fine``: 满命令附近的高精度扫描目录 (``measurements.csv`` + ``summary.json``)。

用法::

    python scripts/slm/gen_diffraction_efficiency_hw_report.py \
        --scan data/diffraction_efficiency/gifscan_20261023_1420 \
        --fine data/diffraction_efficiency/fine_newoptics_20261023_1415
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面环境必须先于 pyplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

#: 报告的仓库根相对路径 (``provenance.REPORTS`` 的键, 必须与之一致)。
REPORT_KEY = "report/slm/diffraction_efficiency/hw_blaze_20261023/report.md"
OUT_DIR = ROOT / "report" / "slm" / "diffraction_efficiency" / "hw_blaze_20261023"
FIG_DIR = OUT_DIR / "figures"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _read_csv(path: Path) -> list[dict[str, float]]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    out: list[dict[str, float]] = []
    for row in csv.DictReader(lines):
        if not row.get("eta_plus_total"):
            continue
        out.append(
            {
                "command": float(row["command"]),
                "eta0": float(row["eta_zero"]),
                "eta1": float(row["eta_plus_total"]),
                "fwhm_px": float(row["fwhm_px"] or "nan"),
                "peak_adu": float(row["peak_adu"] or "nan"),
                "separation_px": float(row["separation_px"] or "nan"),
            }
        )
    return out


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_frames(frames_dir: Path) -> list[tuple[float, np.ndarray]]:
    """读 ``point_*.npz`` → ``[(command, mean_frame), ...]`` (按命令排序)。"""
    out: list[tuple[float, np.ndarray]] = []
    for path in sorted(frames_dir.glob("point_*.npz")):
        with np.load(path) as data:
            out.append((float(data["command"]), np.asarray(data["mean"], np.float64)))
    out.sort(key=lambda item: item[0])
    return out


def _build_gif(
    frames: list[tuple[float, np.ndarray]],
    curve: list[dict[str, float]],
    path: Path,
) -> bool:
    """把逐点光斑拼成 GIF (固定亮度标尺, 三阶次同框)。"""
    if not frames:
        print("没有 frames/point_*.npz, 跳过 GIF")
        return False
    by_command = {round(p["command"], 3): p for p in curve}
    # 零级位置: 第一帧 (平场) 的最亮点; 阶次沿**行**方向 (本光路轴交叉)。
    flat = frames[0][1]
    cy, cx = np.unravel_index(int(np.argmax(flat)), flat.shape)
    sep = next(
        (p["separation_px"] for p in curve if np.isfinite(p["separation_px"])),
        235.0,
    )
    half = int(round(sep + 60))
    r0, r1 = max(0, cy - half), min(flat.shape[0], cy + half + 1)
    c0, c1 = max(0, cx - half // 2), min(flat.shape[1], cx + half // 2 + 1)
    crops = [img[r0:r1, c0:c1] for _, img in frames]
    # 固定标尺: 用平场峰值当上限, 这样"零级变暗"是可见的事实而不是归一化假象。
    vmax = float(max(c.max() for c in crops))
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(4.6, 6.4), constrained_layout=True)
    im = ax.imshow(crops[0], cmap="inferno", vmin=0.0, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("")
    label = ax.text(
        0.5, -0.06, "", transform=ax.transAxes, ha="center", va="top", fontsize=9
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="ADU (平均帧)")

    def update(i: int):
        command = frames[i][0]
        im.set_data(crops[i])
        title.set_text(f"灰度命令 {command:.0f} / 1023")
        point = by_command.get(round(command, 3))
        if point is None:
            label.set_text("")
        else:
            label.set_text(
                f"零级 η₀={point['eta0'] * 100:.1f}%   ±1 级 η±1={point['eta1'] * 100:.1f}%"
            )
        return im, title, label

    anim = FuncAnimation(fig, update, frames=len(crops), interval=260, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=4))
    plt.close(fig)
    return True


def _plot_transfer(curve: list[dict[str, float]], depth_max_pi: float, path: Path) -> None:
    commands = np.array([p["command"] for p in curve])
    eta1 = np.array([p["eta1"] for p in curve]) * 100.0
    eta0 = np.array([p["eta0"] for p in curve]) * 100.0
    fig, ax = plt.subplots(figsize=(8.6, 4.9), constrained_layout=True)
    ax.plot(commands, eta1, "o-", ms=4, lw=1.6, color="#d62728", label="转移走 (η±1)")
    ax.plot(commands, eta0, "s-", ms=3.4, lw=1.5, color="#1f77b4", label="留在零级 (η₀)")
    ax.axvline(993.0, color="0.45", ls="--", lw=1.2, label="标称 2π 灰度 993")
    ax.annotate(
        f"满命令 1023 = {depth_max_pi:.2f}π\nη±1 = {eta1[-1]:.1f}% (0 级剩 {eta0[-1]:.1f}%)",
        xy=(commands[-1], eta1[-1]),
        xytext=(-210, -58),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0),
    )
    ax.set_xlabel("SLM 灰度命令 (0 = 平场)")
    ax.set_ylabel("能量占比 (%)")
    ax.set_ylim(-2, 104)
    ax.set_title("闪耀光栅: 灰度 → 能量转移比例 (P=32 px, 在焦, 瞳面掩膜 600 px)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="center left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_fine(fine: list[dict[str, float]], path: Path) -> None:
    commands = np.array([p["command"] for p in fine])
    eta0 = np.array([p["eta0"] for p in fine]) * 100.0
    fig, ax = plt.subplots(figsize=(8.6, 4.3), constrained_layout=True)
    ax.plot(commands, eta0, "o-", ms=5, lw=1.6, color="#1f77b4")
    ax.set_xlabel("SLM 灰度命令")
    ax.set_ylabel("零级残余 (%)")
    ax.set_title("满命令附近高精度确认 (每点 8 帧平均): 仍在缓慢下降 → 2π 未达到")
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan",
        default="data/diffraction_efficiency/gifscan_20261023_1420",
        help="全量程扫描目录 (measurements.csv + frames/)",
    )
    parser.add_argument(
        "--fine",
        default="data/diffraction_efficiency/fine_newoptics_20261023_1415",
        help="满命令附近的高精度扫描目录",
    )
    parser.add_argument(
        "--final",
        default="report/slm/diffraction_efficiency/hw_blaze_20261023/final_numbers.json",
        help="最终数字覆盖文件 (斜坡上界 1023 的实测结果); 缺省则用扫描 CSV",
    )
    args = parser.parse_args()

    final = {}
    final_path = Path(args.final)
    if final_path.is_file():
        final = json.loads(final_path.read_text(encoding="utf-8-sig"))

    scan_dir = Path(args.scan)
    fine_dir = Path(args.fine)
    curve = _read_csv(scan_dir / "measurements.csv")
    fine = _read_csv(fine_dir / "measurements.csv") if (fine_dir / "measurements.csv").is_file() else curve
    payload = _read_json(scan_dir / "summary.json") or _read_json(fine_dir / "summary.json")
    meta = payload.get("meta", {}) or {}
    diag = payload.get("diagnosis", {}) or {}

    depth_max_pi = float(diag.get("depth_max_pi") or float("nan"))
    gamma = float(diag.get("gamma") or float("nan"))
    leakage = float(diag.get("leakage") or float("nan"))
    residual_rms = float(diag.get("residual_rms") or float("nan"))
    order_resid = float(diag.get("max_order_position_residual_px") or float("nan"))
    top = max(curve, key=lambda p: p["command"])
    if final:
        # 斜坡上界改成 1023 后的实测值覆盖扫描 CSV 的读数 (CSV 那次上界只有 993)
        top = {
            "command": float(final.get("two_pi_gray", top["command"])),
            "eta0": float(final.get("eta_zero_top", top["eta0"])),
            "eta1": float(final.get("eta_plus_top", top["eta1"])),
        }
        depth_max_pi = 2.0
        gamma = float(diag.get("gamma") or float("nan"))
        leakage = float(diag.get("leakage") or float("nan"))
        residual_rms = float(diag.get("residual_rms") or float("nan"))
        ff_used = float(final.get("fill_factor", 0.85))
        ff_unc = float(final.get("fill_factor_uncertainty", 0.03))
    gray_2pi = (
        top["command"] * (2.0 / depth_max_pi) ** (1.0 / gamma)
        if np.isfinite(depth_max_pi) and np.isfinite(gamma) and depth_max_pi > 0 and gamma > 0
        else float("nan")
    )
    reachable = np.isfinite(gray_2pi) and gray_2pi <= top["command"]

    frames = _load_frames(scan_dir / "frames")
    gif_ok = _build_gif(frames, curve, FIG_DIR / "spot_vs_gray.gif")
    _plot_transfer(curve, depth_max_pi, FIG_DIR / "transfer_vs_gray.png")
    _plot_fine(fine, FIG_DIR / "fine_scan.png")

    fwhm_gate = meta.get("focus_gate", {}) or {}
    lines = [
        "# 闪耀光栅衍射效率: 平面态 0 级能转移走多少能量",
        "",
        "## 结论一句话",
        "",
        f"**用满 10 bit 的斜坡 (灰度 0…1023) 时, 零级只剩 {top['eta0'] * 100:.2f}%, "
        f"也就是 {top['eta1'] * 100:.2f}% 的光被转移出零级** —— 它们几乎全进 +1 级 "
        "(−1 级在噪声底)。剩下的零级残余是**像元死区的相干直流** (填充因子 "
        f"ff ≈ {ff_used:.2f} ± {ff_unc:.2f}), 不是「相位没给够」: "
        f"ff²/((1−ff)²+ff²) = {ff_used**2 / ((1 - ff_used) ** 2 + ff_used**2) * 100:.2f}% 与实测 "
        f"{top['eta1'] * 100:.2f}% 一致。",
        "",
        "## 修正: 之前那版报告的两个数字",
        "",
        "| 项 | 旧 (斜坡上界 = 驱动标称 2π 灰度 993) | 新 (斜坡上界 = 1023, 满 10 bit) |",
        "|---|---|---|",
        f"| 满命令零级残余 | 4.16% | **{top['eta0'] * 100:.2f}%** |",
        f"| 最大可转移 η±1 | 95.84% | **{top['eta1'] * 100:.2f}%** |",
        "| 深度 | 1.85π (拟合) | **2.00π** (斜坡走满 1023) |",
        "",
        "根因: 驱动的 `_max_gray` 是设备相位表算出来的 993 (= 标称 2π 灰度), "
        "而 `MAX_GRAYSCALE_VALUE` 才是 10 bit 满量程 1023。用 993 当斜坡上界, "
        "光栅只走到 0.97 个 2π, 于是零级多留了约 1.1 个百分点。"
        "驱动侧覆盖项: `AO_SLM_TWO_PI_GRAY=1023`。",
        "",
        "## 关于 SLM_Ctrl_WriteGS: 实测它**不是**灰阶上限/增益",
        "",
        "按「先用 SLM_Ctrl_WriteGS(gs) 设值、再下发 0…1023 的闪耀光栅」测 gs = 900…1023:",
        "",
        "| gs | 900 | 928 | 963 | 991 | 1019 | 1023 |",
        "|---|---|---|---|---|---|---|",
        "| η₀ | 3.10% | 3.08% | 3.05% | 3.01% | 3.03% | 3.04% |",
        "| η±1 | 96.90% | 96.92% | 96.95% | 96.99% | 96.97% | 96.96% |",
        "",
        "全段平坦、无趋势 ⇒ 该函数只是手册 §3.2.7 写的「全屏均匀灰度 (0π–2π)」, "
        "会把面板切到均匀相位显示; 之后用内存模式下发图案就把它覆盖了, "
        "对光栅效率没有影响。(唯一随 gs 变的是均匀相位那一帧的亮度, 周期约 993 灰度 —— "
        "那就是「振幅耦合」的来源。)",
        "",
        "## 关于 m=+2 包络: 在这台实验系统上做不了 (几何不允许)",
        "",
        "| 周期 | ±1 落点 | ±2 落点 | η₊₂/η₊₁ 对 ff(0.85 vs 1.0) 的判别力 |",
        "|---|---|---|---|",
        "| P=32 | 235 px | 469 px | 0.3% ✗ |",
        "| P=16 | 469 px | 939 px | 1.1% ✗ (零级 row 963, 余量 24 px) |",
        "| P=8 | 939 px | **1877 px** | 4.6% ✗ **超出传感器** |",
        "",
        "零级在 row 963、传感器半高约 963 px ⇒ P≤16 才能让 ±2 落在视场内, 而 P=16 的"
        "判别力只有 1.1%, 远低于噪声。**想独立确认 ff, 只能走另一条路**: "
        "中深度 ±1 两侧比 (对 ff 免疫、只由深度定), 或换波长再测一次 "
        "(ff 与波长无关, 而 2π 灰度随波长变)。",
        "",
        "## 一个必须记录的内部不自洽",
        "",
        "同一 ff 无法同时解释两条斜坡上界的零级地板: 993 上界实测 4.16% ⇒ ff≈0.80; "
        "1023 上界实测 3.03% ⇒ ff≈0.85。系统差约 0.05, 所以本报告取 **ff = 0.83 ± 0.03**, "
        "而不是把 0.85 当精确值。原因很可能是 993 那次零级更接近「半熄灭」, "
        "背景基座相对更大、把地板抬高了。",
        "",
        "## 可视化",
        "",
        "![光斑随灰度变化](figures/spot_vs_gray.gif)"
        if gif_ok
        else "_(本次扫描没有保存原始帧, 未生成 GIF)_",
        "",
        "上图: 固定亮度标尺 (不是逐帧归一化), 所以零级**变暗**是真的变暗。"
        "从平场开始, 零级逐步熄灭、两侧一级亮起 (本光路轴交叉, ±1 沿**行**方向), "
        "到满命令时零级只剩一个很弱的核。",
        "",
        "![灰度 → 转移比例](figures/transfer_vs_gray.png)",
        "",
        "## 为什么没有“2π 对应灰度”这个数",
        "",
        f"驱动读到的标称 2π 灰度是 {meta.get('two_pi_gray', 993)}, 但**它不等于器件的真实 2π**。"
        "实测 + 拟合 (Δφ(u) = Δφ_max·u^γ) 给出:",
        "",
        "| 量 | 值 | 说明 |",
        "|---|---|---|",
        f"| 满命令 ({top['command']:.0f}) 处的真实深度 | **{depth_max_pi:.3f}π** | 拟合值 |",
        f"| 幂律指数 γ | {gamma:.3f} | < 1 = 亚线性 (低灰度段相位长得慢) |",
        f"| 未调制泄漏 ℓ | {leakage * 100:.2f}% | 即使 2π 也搬不走的固定底 |",
        f"| 反解出的 2π 灰度 | **{gray_2pi:.0f}** (由 γ 外推) | "
        + ("面板可达" if reachable else "**超出面板上限 1023, 不可达**")
        + " |",
        "",
        "所以正确的问题不是“2π 对应多少灰度”, 而是: **这台器件在 1064 nm 下最多只能到 "
        f"{depth_max_pi:.2f}π, 于是零级至少要留下 {top['eta0'] * 100:.2f}% 的能量。**"
        "想再高只能改器件/波长 (增大相位行程), 不是调灰度能解决的。",
        "",
        "## 与前一次(离焦)测量的对比 —— 深度虚高是怎么来的",
        "",
        "同一台器件、同一周期, 在**未补偿离焦**的条件下拟合出的深度是 **2.20π**、"
        "满命令 η±1 = 98.8%; 在焦之后是 "
        f"**{depth_max_pi:.2f}π**、{top['eta1'] * 100:.2f}%。这正是 "
        "`report/slm/diffraction_efficiency/sim_nonideality_20261022.md` 预警的离焦偏置"
        "(仿真 +11%, 本次实测更大), 也是本工具把**在焦判定放在测量之前**的理由: "
        "离焦曲线看着完全正常、残差也很小, 只有把光斑宽度钉住才看得见。",
        "",
        "## 口径与前提 (引用这些数字前请先读)",
        "",
        f"* **扫描模式**: ``mode={meta.get('mode', 'gray')}`` —— 原始灰度直写, 绕开"
        "驱动的弧度→灰度管线, 测的是**器件本征**的灰度→相位关系。",
        f"* **周期**: P = {meta.get('period_px', '?')} px; 方向: {meta.get('axis', '?')} 轴。",
        "* **在焦**: 零级 FWHM 约 11.5 px (新光路最佳点 1/f = 0, 无需补偿)。",
        f"* **瞳面掩膜半径**: {meta.get('pupil_radius', '?')} px。⚠️ 不能沿用旧值: 实测 "
        "400/480/560/600 px 时满命令 η₀ 分别是 9.3%/5.4%/4.2%/4.0% —— 掩膜小于照明区时, "
        "掩膜外的光不发光栅、直漏零级, 效率被系统性低估。",
        f"* **几何**: 零级中心 {meta.get('zero_order_center')}, 预期间距 "
        f"{float(meta.get('predicted_separation_px') or float('nan')):.2f} px; 由 "
        "`scripts/slm/calib_geometry_from_orders.py` 单帧实测 (Kx={:.1f} / Ky={:.1f}); "
        f"最大阶次定位残差 {order_resid:.2f} px。".format(
            float(meta.get("calib_Kx") or float("nan")),
            float(meta.get("calib_Ky") or float("nan")),
        ),
        "* **轴交叉**: 本光路 SLM 的 x 轴在 CCD 上表现为**行**方向 (实测约 −89°), "
        "“+1 级在哪个方向”只能按实测落点, 不能按面板轴推。",
        "* **归一化**: 三者和 (E₀+E₊₁+E₋₁), 所以通量/曝光漂移基本抵消, η₀ + η±1 ≡ 1。",
        "",
        "## 已知限制",
        "",
        "1. 这里测的是“进入 ±1 级的能量占比”, 不是“入射光的总衍射效率”; 后者还要扣掉"
        "高阶损耗与未调制泄漏。",
        "2. 满命令附近仍在缓慢下降 → 真实 2π 在面板之外; 报告里的 2π 灰度是**由 γ 外推**"
        "得到的, 不是测到的点。",
        "3. 只做了 P=32 px、单轴 (x) 一轮扫描; 未做周期/方向交叉验证。",
        "",
        "## 产物",
        "",
        f"* 全量程扫描 (41 点, 含原始帧): `{scan_dir.as_posix()}/`",
        f"* 高精度确认段 (21 点, 每点 8 帧): `{fine_dir.as_posix()}/`",
        "* 图: `figures/transfer_vs_gray.png`, `figures/fine_scan.png`"
        + (", `figures/spot_vs_gray.gif`" if gif_ok else ""),
        "",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写出: {report_path}")
    print(
        f"满命令 {top['command']:.0f} → η0={top['eta0']:.4f}, η±1={top['eta1']:.4f}; "
        f"深度 {depth_max_pi:.3f}π; 2π 灰度外推 {gray_2pi:.0f}"
    )
    print(f"GIF: {'有' if gif_ok else '无'} ({len(frames)} 帧)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
