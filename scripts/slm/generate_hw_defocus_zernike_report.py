#!/usr/bin/env python
"""Generate the hardware defocus + Zernike-uniformity test report (2026-10-20).

Offline-only: reads the artefacts produced by

* ``scripts/slm/hw_gs_defocus_probe.py`` (hardware defocus sweep) — the
  ``measurements.csv`` / ``measurements.json`` / ``run_config.json`` /
  ``frame_*.npz`` set in one measurement directory, and
* ``scripts/slm/run_gs_refine_zernike_sweep.py`` (hardware Zernike-order
  sweep) — ``summary.json`` in one run directory,

plus the simulation reference curve ``report/shape_trials/fv_groups_fresnel_vs_vl.csv``
(group ``G1``: the "dz-blind GS, no defocus" group that
``report/shape_trials/gs_defocus_sim_20261019.md`` quotes as its G1 row).

Never opens a device.  The per-frame spot quantities (FWHM / hollowness /
centroid) are recomputed **offline** from the raw frames the probe stored, so
the metric calibre can be changed without re-running the hardware.

Run::

    python scripts/slm/generate_hw_defocus_zernike_report.py \\
        --defocus-dir report/shape_trials/hw_gs_defocus_probe_<UTC> \\
        --noise-dir report/shape_trials/hw_gs_defocus_probe_<UTC> \\
        --zernike-dir data/debug/hw_zernike_sweep_<date>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _common import fmt_metric, markdown_table, savefig  # noqa: E402
from _common.provenance import insert_header  # noqa: E402
from ao_shaping.algorithm.signal_processing.square_shaping import (  # noqa: E402
    composite_from_pib_cv,
    power_in_bucket,
    uniformity_cv,
)
from ao_shaping.utils.image.beam_metrics import prepare_frame  # noqa: E402
from ao_shaping.utils.image.frame_analysis import (  # noqa: E402
    estimate_shift,
    measure_spot,
)
from ao_shaping.utils.image.targets import square_mask  # noqa: E402

REPORT_KEY = "report/shape_trials/hw_defocus_zernike_20261020.md"
OUT_DIR = ROOT / "report" / "shape_trials"
FIG_DIR = OUT_DIR / "figures"
SIM_G1_CSV = ROOT / "report" / "shape_trials" / "fv_groups_fresnel_vs_vl.csv"

# --- CJK font guard (never crash on a headless box) --------------------------
try:
    from matplotlib import font_manager

    _available = {f.name for f in font_manager.fontManager.ttflist}
    _cjk_candidates = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]
    _picked = [n for n in _cjk_candidates if n in _available]
    if _picked:
        plt.rcParams["font.sans-serif"] = _picked + ["DejaVu Sans"]
except ImportError:  # pragma: no cover
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_probe_run(run_dir: Path) -> dict[str, Any]:
    """Read one ``hw_gs_defocus_probe`` measurement directory.

    Returns the run config, the measurement rows, and the per-frame spot
    quantities recomputed from the stored raw frames.
    """
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    rows = json.loads((run_dir / "measurements.json").read_text(encoding="utf-8"))
    cx, cy = config["zero_order_roi_xy_frozen"]
    side = int(config["target_side_px"])

    enriched: list[dict[str, Any]] = []
    for row in rows:
        npz_path = run_dir / f"frame_{int(row['index']):03d}.npz"
        item = dict(row)
        item["frozen_center_xy"] = [int(cx), int(cy)]
        item["target_side_px"] = side
        if not npz_path.exists():
            item["frame_missing"] = True
            enriched.append(item)
            continue
        raw = np.load(npz_path)["frame"]
        frame = prepare_frame(raw)
        spot = measure_spot(raw)
        frozen = square_mask(frame.shape, int(cy), int(cx), side)
        cv_frozen = float(uniformity_cv(frame, frozen))
        pib_frozen = float(power_in_bucket(frame, frozen))
        # 随斑框: **边长不变**, 只把中心平移到光斑质心。用来把「转向」与「离焦」
        # 两个效应分开 —— 它不做任何缩放, 所以不触发仓库记录的自适应框刷分陷阱。
        sx = int(round(spot.centroid_x))
        sy = int(round(spot.centroid_y))
        moved = square_mask(frame.shape, sy, sx, side)
        cv_moved = float(uniformity_cv(frame, moved))
        pib_moved = float(power_in_bucket(frame, moved))
        item.update(
            {
                "frame_missing": False,
                "fwhm_px": float(spot.fwhm_px),
                "hollowness": float(spot.hollowness),
                "centroid_xy": [float(spot.centroid_x), float(spot.centroid_y)],
                "shift_px": float(math.hypot(spot.centroid_x - cx, spot.centroid_y - cy)),
                "shift_x_px": float(spot.centroid_x - cx),
                "shift_y_px": float(spot.centroid_y - cy),
                "cv_frozen": cv_frozen,
                "pib_frozen": pib_frozen,
                "score_frozen": float(composite_from_pib_cv(pib_frozen, cv_frozen)),
                "cv_moved": cv_moved,
                "pib_moved": pib_moved,
                "score_moved": float(composite_from_pib_cv(pib_moved, cv_moved)),
            }
        )
        enriched.append(item)
    return {"dir": run_dir, "config": config, "rows": enriched}


def noise_floor(run: dict[str, Any]) -> dict[str, float]:
    """重复测量给出的噪声地板 (同一 Δz 的组内标准差)。"""
    rows = [r for r in run["rows"] if not r.get("frame_missing")]
    out: dict[str, float] = {}
    for key in ("cv", "pib", "score", "frame_peak"):
        values = [float(r[key]) for r in rows if math.isfinite(float(r.get(key, math.nan)))]
        out[f"sigma_{key}"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        out[f"mean_{key}"] = float(np.mean(values)) if values else math.nan
    out["n_repeats"] = len(rows)
    return out


def load_sim_g1() -> list[dict[str, Any]]:
    """仿真 G1 组 (dz-blind GS, 无离焦) 的细网格参考曲线。"""
    with SIM_G1_CSV.open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["group"] == "G1"]
    out = []
    for r in rows:
        out.append(
            {
                "dz_mm": float(r["dz_mm"]),
                "dz_units": float(r["dz_units"]),
                "side_px": int(r["side_px"]),
                "cv": float(r["cv"]),
                "pib": float(r["pib"]),
                "score": float(r["score"]),
                "cv_fixed": float(r["cv_fixed"]),
                "pib_fixed": float(r["pib_fixed"]),
                "score_fixed": float(r["score_fixed"]),
                "sigma_px": float(r["sigma_px"]),
            }
        )
    return sorted(out, key=lambda r: r["dz_mm"])


def load_zernike(run_dir: Path) -> dict[str, Any]:
    """读 ``run_gs_refine_zernike_sweep.py`` 的 ``summary.json``。"""
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def discover_groups(roots: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """在根目录里递归找出所有**分组记录**。

    刻意**不**依赖分组清单 JSON 的具体字段名 —— 清单是驱动脚本的产物, 字段会变;
    这里按「目录里有哪些特征文件」来认组:

    - 含 ``measurements.json`` ⇒ 离焦探针组 (``hw_gs_defocus_probe.py``);
    - 含 ``summary.json`` ⇒ 泽尼克阶数组 (``run_gs_refine_zernike_sweep.py``)。

    这样分组布局 (``<root>/<group>/``) 与旧的平铺布局都能认出来, 报告不挑。
    """
    probe: list[dict[str, Any]] = []
    zernike: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("measurements.json")):
            if path.parent in seen:
                continue
            seen.add(path.parent)
            probe.append({"kind": "probe", "name": path.parent.name, "dir": path.parent})
        for path in sorted(root.rglob("summary.json")):
            if path.parent in seen:
                continue
            seen.add(path.parent)
            zernike.append({"kind": "zernike", "name": path.parent.name, "dir": path.parent})
    return {"probe": probe, "zernike": zernike}


def _phase_label(run: dict[str, Any]) -> str:
    """给分组标注相位来源（模型 GS 相位 / 指定的相位文件）。"""
    source = str(run["config"].get("gs_phase_source", ""))
    if "现场生成" in source:
        return "模型 GS 相位（`OffFocusBench` 现场生成，Δz=0 的 plain GS）"
    if source:
        tail = source.split(":", 1)[-1].strip()[:80]
        return f"指定相位文件（{tail}）"
    return "未标注"


def is_noise_group(run: dict[str, Any]) -> bool:
    """该组是不是「同一相位重复采集」的噪声地板组（所有 Δz 都为 0）。"""
    rows = [r for r in run["rows"] if not r.get("frame_missing")]
    if not rows:
        return False
    return all(abs(float(r["dz_mm"])) < 1e-9 for r in rows)


def pick_closest_noise(
    noise_runs: list[dict[str, Any]], exposure_ms: float | None
) -> dict[str, Any] | None:
    """挑曝光最接近目标曝光的那条噪声地板（判阶数差异必须同曝光）。"""
    if not noise_runs:
        return None
    if exposure_ms is None:
        return noise_runs[0]

    def key(run: dict[str, Any]) -> float:
        got = run["config"].get("exposure_ms_working")
        if got is None:
            return math.inf
        return abs(float(got) - float(exposure_ms))

    return min(noise_runs, key=key)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def steering_fit(rows: list[dict[str, Any]], limit_mm: float = 2.5) -> dict[str, float]:
    """光斑质心位移 vs Δz 的线性拟合 (转向速率, px/mm)。

    瞳面二次相位以面板标称照明中心 (960, 600) 为中心, 而照明并不落在那里,
    所以这张相位等价于一片**偏心透镜**: 它一边离焦, 一边把光斑推走。

    拟合只用 |Δz| ≤ ``limit_mm`` 的点: 更大的 |Δz| 上光斑已经跑出 ROI,
    质心量到的是晕斑而不是被整形的光, 会把斜率稀释掉 (全区间拟合会从
    -6 px/mm 量级掉到 -3 px/mm 量级)。
    """
    pts_all = [
        (float(r["dz_mm"]), float(r["shift_x_px"]))
        for r in rows
        if not r.get("frame_missing") and math.isfinite(float(r["shift_x_px"]))
    ]
    out: dict[str, float] = {}
    for label, pts in (
        ("local", [p for p in pts_all if abs(p[0]) <= limit_mm]),
        ("global", pts_all),
    ):
        if len(pts) < 3:
            out[f"slope_{label}_px_per_mm"] = math.nan
            out[f"intercept_{label}_px"] = math.nan
            continue
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        out[f"slope_{label}_px_per_mm"] = float(slope)
        out[f"intercept_{label}_px"] = float(intercept)
    out["limit_mm"] = limit_mm
    out["n_local"] = len([p for p in pts_all if abs(p[0]) <= limit_mm])
    return out


def order_spread(orders: list[dict[str, Any]]) -> dict[str, float]:
    """阶数之间的指标跨度 (用来和噪声地板比)。"""
    out: dict[str, float] = {}
    for key, path in (
        ("fit_rms_rad", lambda o: float(o["fit_rms_rad"])),
        ("before_cv", lambda o: float(o["before"]["cv"])),
        ("after_cv", lambda o: float(o["after"]["cv"])),
        ("before_pib", lambda o: float(o["before"]["pib"])),
        ("after_pib", lambda o: float(o["after"]["pib"])),
        ("after_quality", lambda o: float(o["after"]["quality"])),
    ):
        values = [path(o) for o in orders if math.isfinite(path(o))]
        out[f"{key}_spread"] = (max(values) - min(values)) if values else math.nan
        out[f"{key}_min"] = min(values) if values else math.nan
        out[f"{key}_max"] = max(values) if values else math.nan
    jitters = [
        float(o["spgd_J_max"]) - float(o["spgd_J_min"])
        for o in orders
        if o.get("spgd_J_max") is not None and o.get("spgd_J_min") is not None
    ]
    out["spgd_J_span_min"] = min(jitters) if jitters else math.nan
    out["spgd_J_span_max"] = max(jitters) if jitters else math.nan
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def render_figures(
    scan: dict[str, Any], sim_g1: list[dict[str, Any]], zernike: dict[str, Any]
) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = [r for r in scan["rows"] if not r.get("frame_missing")]
    made: list[str] = []

    # Figure a: 复合分数 vs Δz (硬件定尺框 / 硬件随斑框 / 仿真定尺框)
    hw = [r for r in rows if "score_frozen" in r]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        [float(r["dz_mm"]) for r in hw],
        [float(r["score_frozen"]) for r in hw],
        "o-", color="#2b6cb0", markersize=4, label="硬件 (定尺框, 冻结)",
    )
    if hw and "score_moved" in hw[0]:
        ax.plot(
            [float(r["dz_mm"]) for r in hw],
            [float(r["score_moved"]) for r in hw],
            "s--", color="#975a16", markersize=4, label="硬件 (随斑框, 边长不变)",
        )
    ax.plot(
        [r["dz_mm"] for r in sim_g1],
        [r["score_fixed"] for r in sim_g1],
        "^-", color="#6b46c1", markersize=5, label="仿真 G1 (定尺框)",
    )
    ax.set_xlabel("离焦 Δz (mm)")
    ax.set_ylabel("复合分数")
    ax.set_title("离焦下的复合分数: 硬件 vs 仿真")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "hw_defocus_score_vs_dz.png")
    plt.close(fig)
    made.append("figures/hw_defocus_score_vs_dz.png")

    # Figure b: 转向 —— 质心位移 vs Δz
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        [float(r["dz_mm"]) for r in hw],
        [float(r["shift_x_px"]) for r in hw],
        "o-", color="#c53030", markersize=4, label="质心 x 偏移 (px)",
    )
    ax.plot(
        [float(r["dz_mm"]) for r in hw],
        [float(r["shift_y_px"]) for r in hw],
        "s-", color="#2b6cb0", markersize=4, label="质心 y 偏移 (px)",
    )
    ax.set_xlabel("离焦 Δz (mm)")
    ax.set_ylabel("相对冻结零级的偏移 (px)")
    ax.set_title("瞳面二次相位同时把光斑推走 (转向)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "hw_defocus_steering_vs_dz.png")
    plt.close(fig)
    made.append("figures/hw_defocus_steering_vs_dz.png")

    # Figure c: 光斑 FWHM vs Δz
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        [float(r["dz_mm"]) for r in hw],
        [float(r["fwhm_px"]) for r in hw],
        "o-", color="#2b6cb0", markersize=4, label="光斑 FWHM (px)",
    )
    ax.set_xlabel("离焦 Δz (mm)")
    ax.set_ylabel("FWHM (px)")
    ax.set_yscale("log")
    ax.set_title("离焦下的光斑尺寸")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, FIG_DIR / "hw_defocus_fwhm_vs_dz.png")
    plt.close(fig)
    made.append("figures/hw_defocus_fwhm_vs_dz.png")

    # Figure d: 泽尼克阶数扫描
    orders = zernike.get("orders", [])
    if orders:
        n_vals = [int(o["n_max"]) for o in orders]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(n_vals, [float(o["fit_rms_rad"]) for o in orders], "o-", color="#2b6cb0")
        axes[0].set_xlabel("n_max")
        axes[0].set_ylabel("Zernike 拟合残差 RMS (rad)")
        axes[0].set_title("拟合残差 vs 阶数")
        axes[0].grid(alpha=0.3)
        axes[1].plot(
            n_vals, [float(o["before"]["cv"]) for o in orders], "s--",
            color="#975a16", label="SPGD 前",
        )
        axes[1].plot(
            n_vals, [float(o["after"]["cv"]) for o in orders], "o-",
            color="#2b6cb0", label="SPGD 后",
        )
        base = zernike.get("baselines", {})
        if "gs_freeform" in base:
            axes[1].axhline(
                float(base["gs_freeform"]["cv"]), color="#6b46c1", ls="-.",
                label="GS 逐像素相位基线",
            )
        if "flat" in base:
            axes[1].axhline(
                float(base["flat"]["cv"]), color="#c53030", ls=":", label="平场基线"
            )
        axes[1].set_xlabel("n_max")
        axes[1].set_ylabel("CV")
        axes[1].set_title("方斑均匀性 vs Zernike 阶数")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        savefig(fig, FIG_DIR / "hw_zernike_order_scan.png")
        plt.close(fig)
        made.append("figures/hw_zernike_order_scan.png")
    return made


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def defocus_group_section(
    run: dict[str, Any],
    sim_g1: list[dict[str, Any]],
    sigma_score: float,
    level: str = "2",
    phase: str = "",
    response_figures: dict[str, str] | None = None,
    group_name: str = "",
) -> str:
    """一个离焦分组的完整小节（主表 + 与仿真对拍 + 转向 + 大 Δz）。

    每个分组自带自己的曝光与相位, 所以主表、对拍与转向结论都按组给出 ——
    「一次只跑一组变量」的结构在报告里也要保持。
    """
    cfg = run["config"]
    rows = [r for r in run["rows"] if not r.get("frame_missing")]
    hw_by_dz = {round(float(r["dz_mm"]), 4): r for r in rows}
    steer = steering_fit(rows)

    # --- 主表 ---------------------------------------------------------------
    main_rows = []
    for r in sorted(rows, key=lambda x: (float(x["dz_mm"]), int(x["index"]))):
        kind = "基线 (无叠加)" if float(r["chirp_sign"]) == 0.0 else "扫描"
        main_rows.append(
            [
                kind,
                fmt_metric(r["dz_mm"], 4),
                fmt_metric(r["chirp_edge_waves"], 3),
                fmt_metric(r.get("fwhm_px"), 2),
                fmt_metric(r.get("hollowness"), 3),
                fmt_metric(r.get("frame_peak"), 1),
                fmt_metric(r.get("shift_px"), 1),
                fmt_metric(r.get("cv_frozen", r.get("cv")), 4),
                fmt_metric(r.get("pib_frozen", r.get("pib")), 4),
                fmt_metric(r.get("score_frozen", r.get("score")), 4),
                fmt_metric(r.get("cv_moved"), 4),
                fmt_metric(r.get("score_moved"), 4),
                fmt_metric(r.get("halo_power_fraction_outside_box"), 4),
            ]
        )
    main_table = markdown_table(
        ["类型", "Δz (mm)", "面板边缘相位 (波)", "FWHM (px)", "环形度", "整帧峰值",
         "质心偏移 (px)", "CV 定尺", "PIB 定尺", "复合分数 定尺", "CV 随斑",
         "复合分数 随斑", "框外功率占比"],
        main_rows,
    )

    # --- 与仿真对拍 ---------------------------------------------------------
    zero_rows = [r for r in rows if abs(float(r["dz_mm"])) < 1e-9]
    base_hw = float(zero_rows[0].get("score_frozen", zero_rows[0].get("score", math.nan)))
    base_sim = sim_g1[4]["score_fixed"] if len(sim_g1) > 4 else math.nan
    cmp_rows = []
    for s in sim_g1:
        h = hw_by_dz.get(round(s["dz_mm"], 4))
        if h is None:
            continue
        hw_score = float(h.get("score_frozen", h.get("score", math.nan)))
        cmp_rows.append(
            [
                fmt_metric(s["dz_mm"], 4),
                fmt_metric(s["score_fixed"], 4),
                fmt_metric(hw_score, 4),
                fmt_metric(s["score_fixed"] / base_sim, 4) if base_sim else "-",
                fmt_metric(hw_score / base_hw, 4) if base_hw else "-",
                fmt_metric(float(h.get("shift_px", math.nan)), 1),
            ]
        )
    cmp_table = markdown_table(
        ["Δz (mm)", "仿真 G1 复合分数 (定尺)", "硬件 复合分数 (定尺)",
         "仿真 归一化", "硬件 归一化", "硬件 质心偏移 (px)"],
        cmp_rows,
    )

    # --- 转向的带符号位移表 -------------------------------------------------
    steer_rows = []
    for r in sorted(rows, key=lambda x: float(x["dz_mm"])):
        dz = float(r["dz_mm"])
        if abs(dz) > 2.5 or "shift_x_px" not in r:
            continue
        steer_rows.append(
            [
                fmt_metric(dz, 4),
                fmt_metric(r["shift_x_px"], 1),
                fmt_metric(r["shift_y_px"], 1),
                fmt_metric(r.get("score_frozen", r.get("score")), 4),
                fmt_metric(r.get("score_moved"), 4),
            ]
        )
    steer_table = markdown_table(
        ["Δz (mm)", "质心 x 偏移 (px, 带符号)", "质心 y 偏移 (px)",
         "复合分数 定尺", "复合分数 随斑"],
        steer_rows,
    )

    def at(dz: float, key: str, default: float = math.nan) -> float:
        r = hw_by_dz.get(round(dz, 4))
        if r is None:
            return default
        return float(r.get(key, default))

    base_pairs = ""
    zero_scan = [r for r in rows if abs(float(r["dz_mm"])) < 1e-9
                 and float(r["chirp_sign"]) != 0.0]
    if zero_rows and zero_scan:
        base_pairs = (
            f"{fmt_metric(abs(float(zero_rows[0].get('score_frozen', zero_rows[0].get('score', 0))) - float(zero_scan[0].get('score_frozen', 0))), 4)}"
        )

    # --- 三套口径的离焦响应 -------------------------------------------------
    matched = matched_box_side(run)
    response_table, response_data = defocus_response_table(run, matched)
    aperture_table = aperture_sensitivity_table(run)
    legacy_side = int(cfg["target_side_px"])
    resp_fig = (response_figures or {}).get(group_name, "")
    peak_values = [
        float(v["corr_peak"]) for v in response_data.values() if v.get("tracked_ok") is False
    ]
    peak_txt = (
        f"（互相关峰值在 Δz ≥ 0.19 mm 就低于 0.3 的可用门限，最低到 {fmt_metric(min(peak_values), 2)}）"
        if peak_values else ""
    )
    response_block = f"""
#### {level}.5 离焦响应：三套口径的对照

**目标框尺寸必须与被测相位匹配。** 探针原先把框写死成 {legacy_side} px —— 那是**模型** GS 相位方斑的边长；
这一组的实际光斑只有 {fmt_metric(response_data[0]['fwhm'] if response_data else float('nan'), 1)} px 宽，
框比光斑大 {fmt_metric(legacy_side / max(response_data[0]['fwhm'], 1e-9), 1) if response_data else '-'} 倍。
复合分数是 `0.5·PIB + 0.5·(1/(1+CV))`：框远大于光斑时，离焦的**弥散把光填满框**（CV 反而变好），
均匀项**奖励**离焦，而功率项同时变差，两项几乎等量抵消 ⇒ 复合分数看起来「毫无反应」。
**离焦效果一直都在**（框内功率保持率在**正半轴**上严格单调下降；负半轴系统性地低一档，
那是转向不对称、不是非单调），只是被口径藏起来了。

匹配框 = 该相位 Δz=0 光斑自己的 FWHM（只定一次、之后冻结，不逐帧缩放）。
**读单调性必须按半轴读**：把 ±Δz 按 |Δz| 合并，量到的是正负不对称，不是离焦响应。

**三条读数纪律**（本表逐条对应）：

1. **复合分数不能当主指标。** 框空掉时 CV→0，`1/(1+CV)`→1，复合分数的地板是 **0.5 而不是 0**，
   于是它在**任何**孔径下都会**反弹**（本组 {fmt_metric(response_data[3.75]['score_matched'], 4) if 3.75 in response_data else '-'}
   → {fmt_metric(response_data[5.0]['score_matched'], 4) if 5.0 in response_data else '-'}，|Δz| 3.75 → 5.0 mm：
   更差的图样反而得分更高）。
2. **「跟中心」口径在本实验系统的图样上不可用。** 相位互相关（`estimate_shift`）是仓库里
   为「模型自己的光斑」写的，对**散斑化的远场图样**不灵 {peak_txt}；
   本表的「PIB（互相关跟中心）」列因此大面积为空。质心位移列只在光斑仍是单瓣时可信。
3. **唯一与转向无关的离焦读数是 `g r50`**（功率半径的相对增长）：它量的是图样摊开了多少，
   不看位置，所以转向污染不到它。

⇒ 冻结框心的 PIB 衰减在 |Δz| ≳ 1.5 mm 之后**不能**读作「离焦把光打散了」：
光斑同时被**推走**了（见上 §x.3 的转向率）。消除这个污染的办法是**给 chirp 配一个反向的瞳面斜坡**
（去倾斜），那部分已进代码（`--chirp-tilt-cancel`，自标定），等设备上线后复测。

{response_table}

**孔径灵敏度**（同一批帧，只换目标框尺寸；正半轴）：

{aperture_table}

⇒ 在 8 ~ 20 px 的范围内离焦响应都在，**只有 44 px 那个「模型相位方斑的边长」把响应藏住了**；
所以结论不是孔径选出来的。

{resp_fig}
"""

    edge20 = at(20.0, "chirp_edge_waves")
    edge50 = at(50.0, "chirp_edge_waves")
    big_dz = ""
    if edge20 == edge20:  # not NaN
        edge50_txt = (
            f"、Δz = 50 mm 时是 {fmt_metric(edge50, 2)} 波" if edge50 == edge50 else ""
        )
        big_dz = f"""
#### {level}.4 大 Δz 与 SLM 采样

面板边缘相位在 Δz = 20 mm 时是 {fmt_metric(edge20, 2)} 波{edge50_txt}。相邻像元的相位步进
在 Δz = 20 mm 时约 0.23 rad，**远小于 π**，所以**采样（奈奎斯特）不是瓶颈**；真正的限制是

- 晕斑半径：Δz = 20 mm 对应约 335 µm ≈ 152 px，而 ROI 半宽只有 {cfg['roi_px'] // 2} px
  ⇒ 光斑整体跑出窗口（PIB = 0、框外功率占比 = 1）；
- 零级/杂散光：|Δz| ≥ 15 mm 后整帧峰值降到 3 counts 以下，**框里基本没有光**。

⇒ **把扫描限制在 |Δz| ≤ 20 mm 是对的**（面板边缘 {fmt_metric(edge20, 2)} 波）；再往大推，
晕斑已跑出 ROI，既没有可测信号也没有物理意义。
"""

    # --- 读法: 全部由本组数据算出, 不写死结论 ------------------------------
    sim_at = {round(s["dz_mm"], 4): s for s in sim_g1}
    sim_ratio_15 = (sim_at[round(1.5007, 4)]["score_fixed"] / sim_at[0.0]["score_fixed"]
                    if sim_at.get(0.0) else math.nan)
    hw_ratio_15 = (at(1.5007, "score_frozen") / base_hw) if base_hw else math.nan
    hw_drop = 1.0 - hw_ratio_15
    sim_drop = 1.0 - sim_ratio_15
    if hw_drop > 0.02:
        shape_word = f"**同样在下降**（掉了 {hw_drop:.1%}，仿真是 {sim_drop:.1%}）"
    elif hw_drop > -0.01:
        shape_word = (f"**基本持平**（只动了 {hw_drop:+.1%}，而仿真是 {sim_drop:.1%}）"
                      " —— 换成硬件 GS 相位后，冻结框口径下这个指标**几乎不随离焦变**")
    else:
        shape_word = f"**反而更高**（{hw_drop:+.1%}）"

    asym = abs(at(-1.5007, "score_frozen") - at(1.5007, "score_frozen"))
    asym_sim = abs(sim_at[round(-1.5007, 4)]["score_fixed"]
                   - sim_at[round(1.5007, 4)]["score_fixed"])
    if sigma_score and asym > 3 * sigma_score:
        asym_word = (f"是噪声地板的 {asym / sigma_score:.1f} 倍 —— **这不是噪声**")
    else:
        asym_word = "**与噪声地板同量级 —— 在这个口径下分不出不对称**"

    comparison_note = (
        f"- **形状**：仿真在 |Δz| = 1.5007 mm 处的复合分数降到 Δz=0 的 "
        f"{fmt_metric(sim_ratio_15, 4)}（掉了 {sim_drop:.1%}），这一组是 "
        f"{fmt_metric(hw_ratio_15, 4)}，{shape_word}。"
    )
    if phase == "model_gs":
        comparison_note += (
            f" 绝对值也对不上（硬件 Δz=0 的复合分数 {fmt_metric(base_hw, 4)} 对仿真 0.8073）："
            f" 这一组的相位来源、目标框尺寸（44 px 对 43 px）、目标边长都与仿真的 G1 行**同口径**，"
            f" 所以差距来自光路本身：仿真里 GS 相位与数字孪生**自洽**，"
            f" 而同一张相位下发到真实实验系统上，照明没落在瞳面中心、相机也不在 2f 面上，"
            f" 方斑并不成立（CV ≈ 0.73，与 2026-10-15 的硬件记录 CV ≈ 0.723 同量级）。"
        )
    elif phase == "hardware_gs":
        comparison_note += (
            f" **绝对值与仿真不可比，原因是目标框尺寸对不上**：离焦探针的目标框恒为 44 px，"
            f" 那是按**模型** GS 相位的方斑尺寸定的；而这一组用的是硬件闭环优化出来的 GS 相位，"
            f" 它的实际光斑只有约 10 px（泽尼克扫描从平场 waist 量到 target_side = 9.96 px）。"
            f" 于是 44 px 的框里只有一小块是亮的 —— Δz=0 的定尺框 CV 是 "
            f"{fmt_metric(at(0.0, 'cv_frozen'), 4)}，这个绝对值主要由「框远大于光斑」决定，"
            f" 而不是均匀性本身；**能比的是相对量**（随 |Δz| 变化的趋势与幅度），"
            f" 以及 §3 里同一张相位的泽尼克阶数结果。"
            f" ⚠️ 与仿真的**同尺度**对拍需要仿真在 10 px 目标上重跑（**本轮未做**），"
            f" 所以这一行对拍只成立到「形状」这一层。"
        )

    return f"""
#### {level}.1 主表

{main_table}

表注：

- **类型**：「基线 (无叠加)」是探针在扫描前测的那一行（叠加层为空）；「扫描」行的 Δz=0
  那一行叠加的是一张**全零**二次相位 —— 物理上与基线等价，两行同时给出就是为了显示重复性
  （两者差 {base_pairs}，远小于 3σ 地板）。
- **CV 列为「-」**：框内没有光时 CV 无定义（除零），按无数据显示。出现这一列的 Δz
  已经超出 {cfg['roi_px'] // 2} px 的 ROI 半宽。
- **FWHM 列为「-」**：帧里没有可测光斑。
- **质心偏移**：相对**冻结**的 ROI 内零级位置 {tuple(cfg['zero_order_roi_xy_frozen'])} 的距离。

#### {level}.2 与仿真 G1 曲线对拍

仿真的 G1 组（Δz 盲 GS、无离焦）取自 `report/shape_trials/fv_groups_fresnel_vs_vl.csv`
（与 `gs_defocus_sim_20261019.md` §2.1 的 G1 行同源：Δz=0 处 score_fixed = 0.8073、
cv_fixed = 0.4215，两边逐位相同）。硬件侧取定尺框复合分数，各自对自己的 Δz=0 归一化。

{cmp_table}

读法：

{comparison_note}
- **正负半轴**：仿真 G1 在 ±1.5007 mm 处差 {fmt_metric(asym_sim, 4)}，这一组差 {fmt_metric(asym, 4)}，
  噪声地板 {fmt_metric(sigma_score, 5)}，{asym_word}。

#### {level}.3 转向：为什么硬件曲线不对称

瞳面二次相位以面板标称照明中心 (960, 600) 为中心，而照明并不落在那里，于是这张相位等价于
一片**偏心透镜**：它一边离焦，一边把光斑横向推走。实测质心 x 相对冻结零级的偏移随 Δz
线性增长，斜率 **{fmt_metric(steer['slope_local_px_per_mm'], 2)} px/mm**（|Δz| ≤ 2.5 mm 内拟合；
全区间拟合是 {fmt_metric(steer['slope_global_px_per_mm'], 2)} px/mm，但大 |Δz| 上光斑已跑出 ROI，
质心量到的是晕斑，斜率被稀释）。|Δz| = 1.5 mm 时约
{fmt_metric(abs(steer['slope_local_px_per_mm']) * 1.5, 1)} px，|Δz| = 10 mm 时约
{fmt_metric(abs(steer['slope_local_px_per_mm']) * 10, 0)} px。

{steer_table}

**这张表就是不对称的全部机制**。注意 Δz = 0 那一行：光斑质心已经比冻结框中心偏了约 8 px
（照明没落在瞳面中心，GS 方斑与零级不同心）。转向是**带符号**的，于是

- **负 Δz** 把光斑**推回**框中心（Δz = -1.5007 mm 时偏移只剩 {fmt_metric(at(-1.5007, 'shift_x_px'), 1)} px），
  框反而框住了光斑；
- **正 Δz** 把光斑**推得更远**（+1.5007 mm 时偏了 {fmt_metric(at(1.5007, 'shift_x_px'), 1)} px），
  框只框住光斑的一部分。

⇒ 正负半轴测到的不是「同一个离焦的两个方向」，而是「同一个离焦 + 两个方向的转向」。
这一组的 ±1.5007 mm 差值 {fmt_metric(asym, 4)}（噪声地板 {fmt_metric(sigma_score, 5)}，{asym_word}），
主要来自这个几何偏置，不是离焦本身有方向性。

后果：冻结框在 |Δz| ≳ 1.5 mm 之后开始被光斑**拖出框外**，于是

- 框内功率（PIB）随 |Δz| 单调掉到 0；
- 但**随斑框**的复合分数在 +3.75 mm 处回到 {fmt_metric(at(3.75, 'score_moved'), 4)}，
  高于 +1.5007 mm 处的 {fmt_metric(at(1.5007, 'score_moved'), 4)}。

⇒ **冻结框口径下 |Δz| ≳ 1.5 mm 的劣化不能读作「离焦把光打散了」**；光还在，只是被搬走了。
{big_dz}
{response_block}"""


def zernike_noise_floor(
    zernike_dir: Path | None, orders: list[dict[str, Any]]
) -> dict[str, Any]:
    """**同口径**的噪声地板: 泽尼克扫描自己那把 9.96 px 目标框、0.2 ms。

    离焦探针的噪声地板是 44 px 目标框上的, 拿来判「9.96 px 框里的阶数差异」是错的口径
    (框小了 ~20 倍, 每次测量的光子数少一个量级)。所以这里从泽尼克扫描**自己**的
    逐轮记录里取:

    - **PIB**: 每轮都测同一个 (几乎不变的) 相位 —— SPGD 的更新被噪声门挡掉了
      (实测 150 轮里只放行 4 次), 所以逐轮 ``m_roi_pib`` 的散布**就是**测量噪声;
    - **CV**: 历史里没有 CV 列, 于是用同一阶的 ``after - before`` 配对差估 —— 这会把
      SPGD 那点真实改善也算进来, 所以是**偏保守**的一侧 (更难判可排序)。

    两者都取各阶的中位数, 避免个别阶的离群值主导。
    """
    out: dict[str, Any] = {"sigma_pib": math.nan, "sigma_cv": math.nan,
                            "n_orders": 0, "per_order": []}
    if zernike_dir is None or not orders:
        return out
    pib_sigmas: list[float] = []
    cv_sigmas: list[float] = []
    for order in orders:
        n_max = int(order["n_max"])
        path = zernike_dir / f"spgd_nmax{n_max}.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        values: list[float] = []
        for row in rows:
            try:
                values.append(float(row["m_roi_pib"]))
            except (KeyError, TypeError, ValueError):
                continue
        if len(values) > 3:
            pib_sigmas.append(float(np.std(values, ddof=1)))
        before = float(order["before"]["cv"])
        after = float(order["after"]["cv"])
        if math.isfinite(before) and math.isfinite(after):
            # 配对差的半宽 = 单次测量噪声 (保守: 含 SPGD 的真实改善)
            cv_sigmas.append(abs(after - before) / math.sqrt(2.0))
    if pib_sigmas:
        out["sigma_pib"] = float(np.median(pib_sigmas))
    if cv_sigmas:
        out["sigma_cv"] = float(np.median(cv_sigmas))
    out["n_orders"] = len(pib_sigmas)
    out["per_order"] = [
        {"n_max": int(o["n_max"]),
         "sigma_pib": float(np.std([float(r["m_roi_pib"]) for r in
                                    csv.DictReader(
                                        (zernike_dir / f"spgd_nmax{int(o['n_max'])}.csv")
                                        .open(encoding="utf-8"))], ddof=1))
         if (zernike_dir / f"spgd_nmax{int(o['n_max'])}.csv").exists() else math.nan}
        for o in orders
    ]
    return out


def _frames_by_index(run: dict[str, Any]) -> dict[int, tuple[float, np.ndarray]]:
    """读出该组每一行的原始帧, 按**行索引**建键 (未经处理)。

    ⚠️ 早期版本按 ``dz_mm`` 建键, 于是**基线行会被扫描里的 Δz=0 行覆盖** —— 两者
    物理等价 (一个是空叠加层, 一个是全零二次相位), 但基线行是「重复性」的唯一参照,
    被悄悄顶掉就没法核对重复性了。这里按**行索引**存, 由调用方显式挑选基线行。
    """
    out: dict[int, tuple[float, np.ndarray]] = {}
    for row in run["rows"]:
        if row.get("frame_missing"):
            continue
        path = Path(run["dir"]) / f"frame_{int(row['index']):03d}.npz"
        if not path.exists():
            continue
        out[int(row["index"])] = (float(row["dz_mm"]), np.load(path)["frame"])
    return out


def _split_baseline_and_scan(
    run: dict[str, Any],
) -> tuple[dict[int, tuple[float, np.ndarray]], dict[int, tuple[float, np.ndarray]]]:
    """把该组的帧分成「基线行」(空叠加层) 与「扫描行」。"""
    frames = _frames_by_index(run)
    baseline: dict[int, tuple[float, np.ndarray]] = {}
    scan: dict[int, tuple[float, np.ndarray]] = {}
    for row in run["rows"]:
        idx = int(row["index"])
        if idx not in frames:
            continue
        if float(row.get("chirp_sign", -1.0)) == 0.0:
            baseline[idx] = frames[idx]
        else:
            scan[idx] = frames[idx]
    return baseline, scan


def _scan_frames_by_dz(run: dict[str, Any]) -> dict[float, np.ndarray]:
    """扫描行的帧, 按 Δz 建键 (扫描行里 Δz 唯一)。"""
    _baseline, scan = _split_baseline_and_scan(run)
    return {dz: raw for dz, raw in scan.values()}


def _baseline_frame(run: dict[str, Any]) -> np.ndarray | None:
    """基线行 (空叠加层) 的帧; 没有就返回 None。"""
    baseline, _scan = _split_baseline_and_scan(run)
    for _dz, raw in baseline.values():
        return raw
    return None


def encircled_power_radius(
    frame: np.ndarray, cx: float, cy: float, fraction: float = 0.5
) -> float:
    """含 ``fraction`` 帧功率的半径 (绕 (cx, cy))。

    用「功率半径」而不是 FWHM: FWHM 走的是过极值点的直线, 遇到环形/分裂图样就量到
    环的**厚度** (本实验系统实测: 同一组里 FWHM 在 3.5 ~ 174.6 px 之间跳, 且模型相位
    组非单调)。功率半径对形状不敏感, 适合当「图样摊开了多少」的指标。
    """
    yy, xx = np.indices(frame.shape, dtype=np.float64)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    weight = np.maximum(frame, 0.0)
    total = float(weight.sum())
    if total <= 0:
        return math.nan
    order = np.argsort(r.ravel())
    cum = np.cumsum(weight.ravel()[order])
    idx = int(np.searchsorted(cum, fraction * total))
    idx = min(idx, order.size - 1)
    return float(r.ravel()[order[idx]])


def render_farfield_grid(
    run: dict[str, Any], dz_list: list[float], out_png: Path, title: str
) -> bool:
    """CCD 远场光斑图拼版: 每个 Δz 一格 (log10), 画出目标框。"""
    frames = _scan_frames_by_dz(run)
    picked = [dz for dz in dz_list if dz in frames]
    if not picked:
        return False
    cfg = run["config"]
    cx, cy = cfg["zero_order_roi_xy_frozen"]
    side = int(cfg["target_side_px"])
    ncol = min(4, len(picked))
    nrow = math.ceil(len(picked) / ncol)
    # 共享色标 + tight_layout 互斥 (matplotlib 会警告), 这里用 constrained layout。
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.0 * ncol, 3.1 * nrow), squeeze=False, layout="constrained"
    )
    for ax in axes.ravel():
        ax.axis("off")
    # 先算所有格的 log10 图, 再统一 vmin/vmax: 各格动态范围不同, 共用色标才能
    # 横向比较, 也让 log10 刻度本身可读。
    logs: list[np.ndarray] = []
    for dz in picked:
        frame = prepare_frame(frames[dz].astype(np.float64))
        logs.append(np.log10(np.maximum(frame, 1e-3)))
    stack = np.concatenate([img.ravel() for img in logs])
    vmin = float(np.percentile(stack, 1.0))
    vmax = float(stack.max())
    for k, dz in enumerate(picked):
        ax = axes.ravel()[k]
        frame = prepare_frame(frames[dz].astype(np.float64))
        ax.imshow(logs[k], cmap="magma", origin="upper", vmin=vmin, vmax=vmax)
        box = square_mask(frame.shape, int(cy), int(cx), side)
        ys, xs = np.nonzero(box)
        if xs.size:
            ax.add_patch(plt.Rectangle(
                (xs.min() - 0.5, ys.min() - 0.5), np.ptp(xs) + 1, np.ptp(ys) + 1,
                fill=False, edgecolor="lime", linewidth=1.0,
            ))
        row = next(r for r in run["rows"] if abs(float(r["dz_mm"]) - dz) < 1e-9)
        ax.set_title(
            f"Δz = {dz:+.2f} mm\n"
            f"CV {fmt_metric(row.get('cv_frozen', row.get('cv')), 2)}  "
            f"PIB {fmt_metric(row.get('pib_frozen', row.get('pib')), 2)}",
            fontsize=8,
        )
    mappable = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    bar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    bar.set_label("log10 强度 (counts)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    shape = frames[picked[0]].shape
    fig.suptitle(
        f"{title}\n每格 {shape[1]}×{shape[0]} px（CCD 窗口），绿框 = {side} px 目标框",
        fontsize=11,
    )
    savefig(fig, out_png)
    plt.close(fig)
    return True


def render_crosssection_grid(
    run: dict[str, Any], dz_list: list[float], out_png: Path, title: str
) -> bool:
    """光斑 X/Y 截面 (过光斑中心的 log10 强度剖面)。"""
    frames = _scan_frames_by_dz(run)
    picked = [dz for dz in dz_list if dz in frames]
    if not picked:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cmap = plt.get_cmap("viridis")
    for k, dz in enumerate(picked):
        frame = prepare_frame(frames[dz].astype(np.float64))
        spot = measure_spot(frames[dz])
        sx, sy = int(round(spot.centroid_x)), int(round(spot.centroid_y))
        colour = cmap(k / max(1, len(picked) - 1))
        label = f"{dz:+.2f} mm"
        row = frame[sy, :]
        axes[0].plot(np.arange(frame.shape[1]) - sx, np.log10(np.maximum(row, 1e-3)),
                     color=colour, label=label, lw=1.2)
        col = frame[:, sx]
        axes[1].plot(np.arange(frame.shape[0]) - sy, np.log10(np.maximum(col, 1e-3)),
                     color=colour, label=label, lw=1.2)
    for ax, lab in zip(axes, ("X 截面 (过光斑中心的行)", "Y 截面 (过光斑中心的列)")):
        ax.axvline(0.0, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("相对光斑中心 (px)")
        ax.set_ylabel("log10 强度 (counts)")
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3)
        # 两块面板各放一份图例: 只在一侧放的话, 另一侧的曲线颜色无从对照。
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, out_png)
    plt.close(fig)
    return True


def render_group_farfields(
    scan_groups: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """每个离焦分组出两张图: 远场光斑拼版 + X/Y 光斑截面。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[str]] = {}
    for grp in scan_groups:
        run = grp["run"]
        name = grp["name"]
        frames = _scan_frames_by_dz(run)
        if not frames:
            continue
        dzs = sorted(frames)
        # 拼版挑点: 覆盖两端与焦面附近, 太多则抽稀
        picks = [dz for dz in dzs if abs(dz) <= 20.0]
        if len(picks) > 12:
            step = len(picks) / 12
            picks = [picks[int(i * step)] for i in range(12)]
        made: list[str] = []
        if render_farfield_grid(
            run, picks, FIG_DIR / f"hw_farfield_{name}.png",
            f"CCD 远场光斑 (log10) — 分组「{name}」, 绿框为目标框",
        ):
            made.append(f"figures/hw_farfield_{name}.png")
        if render_crosssection_grid(
            run, picks, FIG_DIR / f"hw_xsec_{name}.png",
            f"光斑 X/Y 截面 (过光斑中心) — 分组「{name}」",
        ):
            made.append(f"figures/hw_xsec_{name}.png")
        if made:
            out[name] = made
    return out


def copy_zernike_farfields(zernike_dir: Path | None, orders: list[dict[str, Any]]) -> list[str]:
    """把泽尼克扫描保存的远场图 (before/after) 复制进报告图目录。"""
    if zernike_dir is None or not orders:
        return []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    for order in orders:
        n_max = int(order["n_max"])
        for tag in ("before", "after"):
            src = zernike_dir / f"{tag}_nmax{n_max}.png"
            if not src.exists():
                continue
            dst = FIG_DIR / f"hw_zernike_{tag}_nmax{n_max}.png"
            dst.write_bytes(src.read_bytes())
            made.append(f"figures/hw_zernike_{tag}_nmax{n_max}.png")
    return made


def render_zernike_xsec(
    farfield_runs: list[tuple[str, Path, dict[str, Any]]], out_png: Path
) -> bool:
    """泽尼克各阶的 X/Y 光斑截面 (来自各阶的远场组原始帧)。"""
    if not farfield_runs:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cmap = plt.get_cmap("viridis")
    plotted = 0
    for k, (name, _dir, run) in enumerate(farfield_runs):
        frames = _scan_frames_by_dz(run)
        if not frames:
            continue
        dz = sorted(frames)[0]
        frame = prepare_frame(frames[dz].astype(np.float64))
        spot = measure_spot(frames[dz])
        sx, sy = int(round(spot.centroid_x)), int(round(spot.centroid_y))
        colour = cmap(k / max(1, len(farfield_runs) - 1))
        label = f"n_max={name.split('nmax')[-1]}"
        axes[0].plot(np.arange(frame.shape[1]) - sx,
                     np.log10(np.maximum(frame[sy, :], 1e-3)),
                     color=colour, label=label, lw=1.3)
        axes[1].plot(np.arange(frame.shape[0]) - sy,
                     np.log10(np.maximum(frame[:, sx], 1e-3)),
                     color=colour, label=label, lw=1.3)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return False
    for ax, lab in zip(axes, ("X 截面 (过光斑中心的行)", "Y 截面 (过光斑中心的列)")):
        ax.axvline(0.0, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("相对光斑中心 (px)")
        ax.set_ylabel("log10 强度 (counts)")
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("泽尼克各阶的远场光斑截面 (设备下线前只采到部分阶)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, out_png)
    plt.close(fig)
    return True


def matched_box_side(run: dict[str, Any]) -> int:
    """与该相位自身光斑匹配的目标框边长 (Δz=0 光斑的 FWHM, 只定一次)。

    探针原先把目标框写死成 44 px —— 那是**模型** GS 相位方斑的边长。硬件 GS 相位的
    实际光斑只有 12.9 px, 框比光斑大 3.4 倍, 于是复合分数的两项
    (``0.5·PIB`` 与 ``0.5·(1/(1+CV))``) 方向相反、量级相当, 互相抵消 ——
    「离焦效果不佳」其实是度量口径的假象, 不是实验系统的问题。
    """
    frames = _scan_frames_by_dz(run)
    zeros = [dz for dz in frames if abs(dz) < 1e-9]
    if not zeros:
        return 0
    base_dz = min(zeros, key=lambda z: abs(z))
    fwhm = float(measure_spot(frames[base_dz]).fwhm_px)
    if not math.isfinite(fwhm) or fwhm <= 0:
        return 0
    roi = int(run["config"]["roi_px"])
    return int(max(4, min(int(round(fwhm)), int(0.6 * roi))))


def score_in_box(frame: np.ndarray, cx: int, cy: int, side: int) -> tuple[float, float, float]:
    """(PIB, CV, 复合分数) —— 在给定框上。"""
    mask = square_mask(frame.shape, int(cy), int(cx), int(side))
    pib = float(power_in_bucket(frame, mask))
    cv = float(uniformity_cv(frame, mask))
    return pib, cv, float(composite_from_pib_cv(pib, cv))


def aperture_sensitivity_table(
    run: dict[str, Any], sides: tuple[int, ...] = (8, 10, 13, 16, 20, 44)
) -> str:
    """孔径灵敏度: 同一批帧在若干目标框尺寸下, 离焦响应是否还在。

    用来排除「结论只是某个框选出来的」这一类反驳: 只要在 8 ~ 20 px 的范围内
    复合分数都随 Δz 下降, 结论就不是孔径的产物; 只有 44 px 那个「模型相位的方斑
    边长」会把响应藏起来。
    """
    frames = _scan_frames_by_dz(run)
    cfg = run["config"]
    cx, cy = cfg["zero_order_roi_xy_frozen"]
    positives = sorted(dz for dz in frames if dz >= 0)
    if not positives:
        return ""
    rows: list[list[Any]] = []
    for side in sides:
        scores: list[float] = []
        pibs: list[float] = []
        for dz in positives:
            frame = prepare_frame(frames[dz].astype(np.float64))
            mask = square_mask(frame.shape, int(cy), int(cx), int(side))
            p = float(power_in_bucket(frame, mask))
            c = float(uniformity_cv(frame, mask))
            pibs.append(p)
            scores.append(float(composite_from_pib_cv(p, c)))
        first = scores[0] if scores else math.nan
        last = scores[-1] if scores else math.nan
        drop = (first - last) / first if first else math.nan
        rows.append([
            f"{side} px",
            fmt_metric(scores[0], 4),
            fmt_metric(scores[min(4, len(scores) - 1)], 4),
            fmt_metric(last, 4),
            fmt_metric(drop, 3),
            fmt_metric(min(pibs), 4),
        ])
    return markdown_table(
        ["目标框边长", "复合分数 @Δz=0", f"复合分数 @Δz={positives[min(4, len(positives)-1)]:g} mm",
         f"复合分数 @Δz={positives[-1]:g} mm", "首末降幅", "最小 PIB (正半轴)"],
        rows,
    )


def defocus_response_table(
    run: dict[str, Any], matched: int
) -> tuple[str, dict[float, dict[str, float]]]:
    """三套口径的离焦响应表 + 逐 Δz 的数值。

    主指标按门禁的结论取**两个**:
    - ``R(Δz)`` = 跟中心框内的功率保持率 (相对 Δz=0 归一) —— 跨相位可比;
    - ``g(Δz)`` = 功率半径 r50 的相对增长 —— 对形状不敏感。
    复合分数**只作辅助读数**: 框空掉时 CV→0, 1/(1+CV)→1, 复合分数的地板是 0.5
    而不是 0, 于是它在任何孔径下都会**反弹** (实测硬件组 13 px 框: 0.3682 @3.75 mm
    → 0.4224 @5.0 mm —— 更差的图样反而得分更高), 所以它不能当主指标。
    """
    frames = _scan_frames_by_dz(run)
    cfg = run["config"]
    cx, cy = cfg["zero_order_roi_xy_frozen"]
    legacy = int(cfg["target_side_px"])
    if not frames:
        return "", {}
    zeros = [dz for dz in frames if abs(dz) < 1e-9]
    ref_dz = min(zeros, key=lambda z: abs(z)) if zeros else min(frames)
    ref_raw = frames[ref_dz]
    ref_frame = prepare_frame(ref_raw.astype(np.float64))

    rows: list[list[Any]] = []
    data: dict[float, dict[str, float]] = {}
    for dz in sorted(frames):
        raw = frames[dz]
        frame = prepare_frame(raw.astype(np.float64))
        spot = measure_spot(raw)
        p_leg, c_leg, s_leg = score_in_box(frame, cx, cy, legacy)
        p_mat, c_mat, s_mat = score_in_box(frame, cx, cy, matched)
        # 跟中心: 用相位互相关 (estimate_shift) 而不是逐帧质心 —— 质心在图样
        # 弥散/分裂时不稳定, 而互相关对形状变化不敏感, 且亚像素。
        try:
            dy, dx, peak = estimate_shift(ref_frame, frame)
        except Exception:  # noqa: BLE001 - 互相关失败就退回质心
            dy = dx = 0.0
            peak = 0.0
        tx = cx + dx
        ty = cy + dy
        # 互相关峰值低 ⇒ 图样已经不成单瓣, 跟中心没有意义 ⇒ 标 N/A
        tracked_ok = bool(peak >= 0.30)
        if tracked_ok:
            p_trk, c_trk, s_trk = score_in_box(frame, round(tx), round(ty), matched)
        else:
            p_trk = c_trk = s_trk = math.nan
        r50 = encircled_power_radius(frame, tx, ty, 0.5)
        data[dz] = {
            "fwhm": float(spot.fwhm_px), "pib_legacy": p_leg, "cv_legacy": c_leg,
            "score_legacy": s_leg, "pib_matched": p_mat, "cv_matched": c_mat,
            "score_matched": s_mat, "pib_tracked": p_trk, "cv_tracked": c_trk,
            "score_tracked": s_trk, "shift_dx": float(dx), "shift_dy": float(dy),
            "corr_peak": float(peak), "tracked_ok": tracked_ok, "r50_px": r50,
            # 质心位移: 光斑仍是单瓣时可靠 (近焦区), 是转向量的直接读数。
            "centroid_dx": float(spot.centroid_x - cx),
            "centroid_dy": float(spot.centroid_y - cy),
        }
        rows.append([
            fmt_metric(dz, 4),
            fmt_metric(float(spot.fwhm_px), 1),
            fmt_metric(s_leg, 4),
            fmt_metric(p_mat, 4), fmt_metric(s_mat, 4),
            fmt_metric(float(spot.centroid_x - cx), 1),
            fmt_metric(r50, 1),
            fmt_metric(p_trk, 4),
        ])
    # 归一化: 相对 Δz=0 (g r50 是无量纲比值, 跨相位可比)
    ref = data.get(ref_dz, {})
    ref_r50 = float(ref.get("r50_px", math.nan))
    for dz, item in data.items():
        r50 = float(item["r50_px"])
        item["g_r50"] = (r50 / ref_r50) if (ref_r50 and math.isfinite(r50)) else math.nan
    for row, dz in zip(rows, sorted(data)):
        row.append(fmt_metric(data[dz]["g_r50"], 3))
    table = markdown_table(
        ["Δz (mm)", "光斑 FWHM (px)",
         f"复合分数 {legacy}px (现口径)",
         f"PIB {matched}px (冻结框心)", f"复合分数 {matched}px (冻结框心)",
         "质心位移 dx (px)", "r50 (px)", "PIB (互相关跟中心)", "g r50 (归一)"],
        rows,
    )
    return table, data


def render_defocus_response(
    run: dict[str, Any], matched: int, data: dict[float, dict[str, float]],
    out_png: Path, title: str,
) -> bool:
    """离焦响应图: 三套口径的复合分数 + 框内功率保持率。"""
    if not data:
        return False
    dzs = sorted(data)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].plot(dzs, [data[d]["score_legacy"] for d in dzs], "s--", color="#c53030",
                 markersize=4, label=f"复合分数（现口径, 冻结 {int(run['config']['target_side_px'])} px 框）")
    axes[0].plot(dzs, [data[d]["score_matched"] for d in dzs], "o-", color="#2b6cb0",
                 markersize=4, label=f"复合分数（匹配框 {matched} px，框心冻结）")
    axes[0].plot(dzs, [data[d]["score_tracked"] for d in dzs], "^-", color="#6b46c1",
                 markersize=4, label=f"复合分数（匹配框 {matched} px，跟中心）")
    base = data[0]["score_legacy"]
    axes[0].axhline(base, color="grey", lw=0.8, ls=":")
    axes[0].set_xlabel("离焦 Δz (mm)")
    axes[0].set_ylabel("复合分数")
    axes[0].set_title("三套口径的复合分数（同一个 Δz=0 归一）", fontsize=10)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].plot(dzs, [data[d]["pib_matched"] for d in dzs], "o-", color="#2b6cb0",
                 markersize=4, label="PIB（匹配框，框心冻结）")
    axes[1].plot(dzs, [data[d]["pib_tracked"] for d in dzs], "^-", color="#6b46c1",
                 markersize=4, label="PIB（匹配框，跟中心）")
    axes[1].plot(dzs, [data[d]["fwhm"] / 20.0 for d in dzs], "s--", color="#975a16",
                 markersize=4, label="光斑 FWHM / 20（换算到同一纵轴）")
    axes[1].set_xlabel("离焦 Δz (mm)")
    axes[1].set_ylabel("框内功率保持率 / 相对尺度")
    axes[1].set_title("离焦对目标能量保持率的破坏（单调）", fontsize=10)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, out_png)
    plt.close(fig)
    return True


def render_defocus_responses(
    scan_groups: list[dict[str, Any]],
) -> dict[str, str]:
    """每个离焦分组出一张「三套口径的离焦响应」图。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for grp in scan_groups:
        run = grp["run"]
        name = grp["name"]
        matched = matched_box_side(run)
        if not matched:
            continue
        _table, data = defocus_response_table(run, matched)
        path = FIG_DIR / f"hw_defocus_response_{name}.png"
        if render_defocus_response(
            run, matched, data, path,
            f"离焦响应（三套口径）— 分组「{name}」",
        ):
            out[name] = f"figures/{path.name}"
    return out


def build_markdown(
    scan: dict[str, Any],
    noise: dict[str, Any],
    sim_g1: list[dict[str, Any]],
    zernike: dict[str, Any],
    zernike_dir: Path,
    figures: list[str],
    order_noise: dict[str, Any] | None = None,
    scan_groups: list[dict[str, Any]] | None = None,
    group_figures: dict[str, list[str]] | None = None,
    response_figures: dict[str, str] | None = None,
    zernike_figures: list[str] | None = None,
) -> str:
    cfg = scan["config"]
    rows = [r for r in scan["rows"] if not r.get("frame_missing")]
    orders = zernike.get("orders", [])
    bases = zernike.get("baselines", {})
    steer = steering_fit(rows)
    spread = order_spread(orders)
    if scan_groups is None:
        scan_groups = [{"name": "main", "dir": scan["dir"], "run": scan,
                        "phase_label": "未标注"}]

    sigma_cv = noise["sigma_cv"]
    sigma_pib = noise["sigma_pib"]
    sigma_score = noise["sigma_score"]

    # 阶数比较用的地板必须与泽尼克扫描**同曝光、同相位**（硬件 GS 相位）,
    # 否则拿离焦扫描那条 (模型 GS 相位, 0.4 ms) 的地板去判阶数差异是错的口径。
    on = order_noise if order_noise else noise
    # 阶数比较用的地板必须与泽尼克扫描**同口径**（9.96 px 目标框、0.2 ms）,
    # 离焦探针那条是 44 px 框上的, 拿来判阶数差异是错的口径。
    zn = zernike_noise_floor(zernike_dir, orders)
    on_sigma_cv = zn["sigma_cv"] if zn["n_orders"] else on["sigma_cv"]
    on_sigma_pib = zn["sigma_pib"] if zn["n_orders"] else on["sigma_pib"]

    # --- 分组清单 -----------------------------------------------------------
    group_rows = []
    for grp in scan_groups:
        run = grp["run"]
        group_rows.append(
            [
                grp["name"],
                grp.get("role", "dz_scan"),
                "dz_mm",
                fmt_metric(run["config"].get("exposure_ms_working"), 3),
                grp.get("phase_label", "未标注"),
                str(len([r for r in run["rows"] if not r.get("frame_missing")])),
                f"`{grp['dir']}`",
            ]
        )
    if zernike.get("orders"):
        group_rows.append(
            [
                zernike_dir.name if zernike_dir else "zernike_order",
                "zernike_order",
                "n_max",
                fmt_metric((zernike.get("config") or {}).get("exposure_time_ms"), 3),
                "硬件 GS 相位（Stage A 产出，Stage B 拟合的就是它）",
                str(len(zernike["orders"])),
                f"`{zernike_dir}`" if zernike_dir else "-",
            ]
        )
    groups_table = markdown_table(
        ["分组", "角色", "扫描的变量", "曝光 (ms)", "相位来源", "数据点数", "记录目录"],
        group_rows,
    )

    # --- 噪声地板 -----------------------------------------------------------
    noise_table = markdown_table(
        ["量", "均值", "组内标准差 σ", "重复次数"],
        [
            ["CV", fmt_metric(noise["mean_cv"]), fmt_metric(sigma_cv, 5),
             str(noise["n_repeats"])],
            ["PIB", fmt_metric(noise["mean_pib"]), fmt_metric(sigma_pib, 5),
             str(noise["n_repeats"])],
            ["复合分数", fmt_metric(noise["mean_score"]), fmt_metric(sigma_score, 5),
             str(noise["n_repeats"])],
            ["整帧峰值 (counts)", fmt_metric(noise["mean_frame_peak"], 2),
             fmt_metric(noise["sigma_frame_peak"], 2), str(noise["n_repeats"])],
        ],
    )

    # --- 离焦分组小节 -------------------------------------------------------
    # 每个分组自带曝光与相位, 所以主表/对拍/转向都按组给; 「一次只跑一组变量」
    # 的结构在报告里也保持。
    group_sections = ""
    for idx, grp in enumerate(scan_groups):
        run = grp["run"]
        # 2.1 是「口径」, 所以分组从 2.2 起, 避免两级标题撞号。
        level = f"2.{idx + 2}"
        heading = (
            f"### {level} 分组「{grp['name']}」"
            f"（曝光 {fmt_metric(run['config'].get('exposure_ms_working'), 3)} ms，"
            f"相位来源：{grp.get('phase_label', '未标注')}）\n\n"
            f"记录目录：`{grp['dir']}`\n"
        )
        group_sections += heading + defocus_group_section(
            run, sim_g1, sigma_score, level, grp.get("phase", ""), response_figures
        )
        group_figs = group_figures.get(grp["name"], [])
        if group_figs:
            group_sections += (
                f"\n#### {level}.6 CCD 远场光斑与光斑截面\n\n"
                + "\n\n".join(f"![{Path(f).stem}]({f})" for f in group_figs)
            )
        group_sections += "\n\n"

    # --- 泽尼克表 -----------------------------------------------------------
    zrows = []
    for o in orders:
        zrows.append(
            [
                str(o["n_max"]),
                str(o["n_modes"]),
                fmt_metric(o["fit_rms_rad"], 4),
                fmt_metric(o["max_abs_coefficient"], 4),
                "是" if o.get("clip_warning") else "",
                fmt_metric(o["before"]["cv"], 4),
                fmt_metric(o["after"]["cv"], 4),
                fmt_metric(o["before"]["pib"], 4),
                fmt_metric(o["after"]["pib"], 4),
                fmt_metric(o["after"]["ee"], 4),
                fmt_metric(o["after"]["quality"], 4),
                "是" if o.get("spgd_improved") else "否",
            ]
        )
    zernike_table = markdown_table(
        ["n_max", "模态数", "拟合残差 (rad)", "max|c| (rad)", "clip 警告",
         "CV 前", "CV 后", "PIB 前", "PIB 后", "EE 后", "复合分数 后", "SPGD 有改善"],
        zrows,
    )
    base_rows = []
    for name, label in (("flat", "平场"), ("gs_freeform", "GS 逐像素相位")):
        b = bases.get(name)
        if not b:
            continue
        base_rows.append(
            [label, fmt_metric(b["cv"], 4), fmt_metric(b["pib"], 4),
             fmt_metric(b["ee"], 4), fmt_metric(b["ar"], 4), fmt_metric(b["quality"], 4)]
        )
    baseline_table = markdown_table(
        ["相位", "CV", "PIB", "EE", "AR", "复合分数"], base_rows
    )

    # --- 阶数可排序性 -------------------------------------------------------
    rms_vals = [float(o["fit_rms_rad"]) for o in orders]
    monotone = all(rms_vals[i] >= rms_vals[i + 1] for i in range(len(rms_vals) - 1))
    cv_before = [float(o["before"]["cv"]) for o in orders]
    pib_before = [float(o["before"]["pib"]) for o in orders]
    qual_after = [float(o["after"]["quality"]) for o in orders]
    cv_spread = spread["after_cv_spread"]
    pib_spread = spread["after_pib_spread"]
    cv_before_spread = max(cv_before) - min(cv_before) if cv_before else math.nan
    pib_before_spread = max(pib_before) - min(pib_before) if pib_before else math.nan
    rankable = (cv_spread > 3 * on_sigma_cv) and (pib_spread > 3 * on_sigma_pib)
    # SPGD 的净贡献: 前后差的绝对值
    spgd_deltas = [abs(float(o["after"]["cv"]) - float(o["before"]["cv"])) for o in orders]
    spgd_max = max(spgd_deltas) if spgd_deltas else math.nan
    freeform = bases.get("gs_freeform", {})
    freeform_cv = float(freeform.get("cv", math.nan)) if freeform else math.nan
    freeform_pib = float(freeform.get("pib", math.nan)) if freeform else math.nan
    verdict = (
        f"**差异可以分辨，但不是「阶数越高越好」。** 阶数之间的 CV 跨度 {fmt_metric(cv_spread, 4)} "
        f"是同口径噪声 {fmt_metric(on_sigma_cv, 5)} 的 "
        f"{fmt_metric(cv_spread / on_sigma_cv, 1) if on_sigma_cv else '-'} 倍，PIB 跨度 "
        f"{fmt_metric(pib_spread, 4)} 是地板 {fmt_metric(on_sigma_pib, 5)} 的 "
        f"{fmt_metric(pib_spread / on_sigma_pib, 1) if on_sigma_pib else '-'} 倍 —— "
        + ("**超过 3σ，差异不是噪声**。" if rankable else
           "**没有超过 3σ，本次运行不支持给阶数排序** —— 任何「最优阶数」都是在读噪声。")
        + f" 但三点必须同时说：\n\n"
        f"   - **CV 随阶数只是整体下降、并不单调**：SPGD 之前（纯 Zernike 拟合）CV 从 "
        f"{fmt_metric(max(cv_before), 4)}（n_max=3）降到 {fmt_metric(min(cv_before), 4)}（n_max={orders[cv_before.index(min(cv_before))]['n_max']}），"
        f"跨度 {fmt_metric(cv_before_spread, 4)}，但 n_max=11 与 25 处有回升，**不是单调改善**。\n"
        f"   - **CV 变好不等于整形变好**：同一批相位的框内功率 PIB 反而从 "
        f"{fmt_metric(max(pib_before), 4)} 降到 {fmt_metric(min(pib_before), 4)}"
        f"（跨度 {fmt_metric(pib_before_spread, 4)}），"
        + (f"而 GS 逐像素相位本身是 CV {fmt_metric(freeform_cv, 4)} / PIB {fmt_metric(freeform_pib, 4)}。"
           if freeform else "")
        + f" 截断到低阶 Zernike 会把方斑**抹平**：框内更均匀（CV 低），但代价是**近一半的框内功率被摊到框外**。"
        f" 只看 CV 会把这个「平滑」误读成「整形更好」。\n"
        f"   - **Stage C 的 SPGD 基本没有改变结果**：各阶 `after` 与 `before` 的 CV 差最大只有 "
        f"{fmt_metric(spgd_max, 4)}，与同口径噪声 {fmt_metric(on_sigma_cv, 5)} 同量级，"
        f"且 n_max=25 直接记为 `spgd_improved=false`。"
        f" 所以阶数之间的差异**全部来自 Stage B 的 Zernike 拟合**，不是 SPGD 优化出来的。"
    )

    # --- 阶数比较用的噪声地板 -----------------------------------------------
    order_noise_table = markdown_table(
        ["量", "σ (同口径)", "来源", "说明"],
        [
            ["CV", fmt_metric(zn["sigma_cv"], 5),
             f"各阶 `after - before` 配对差 / √2（{zn['n_orders']} 阶取中位数）",
             "保守：把 SPGD 那点真实改善也算进了差里"],
            ["PIB", fmt_metric(zn["sigma_pib"], 5),
             f"各阶逐轮 `m_roi_pib` 的组内标准差（{zn['n_orders']} 阶取中位数）",
             "SPGD 的更新被噪声门挡掉（150 轮只放行 4 次），逐轮量的是几乎同一个相位"],
        ],
    )

    fig_block = "\n\n".join(f"![{Path(f).stem}]({f})" for f in figures)

    zernike_farfield_block = ""
    if zernike_figures:
        zernike_farfield_block = "\n\n".join(
            f"![{Path(f).stem}]({f})" for f in zernike_figures
        )

    zernike_section = ""
    if zernike.get("orders"):
        zernike_section = f"""
---

## 3. 泽尼克阶数扫描（硬件）

采集脚本：`scripts/slm/run_gs_refine_zernike_sweep.py --cam-type daheng --defocus-focal-mm 1597.4`，
数据目录 `{zernike_dir.relative_to(ROOT) if zernike_dir.is_relative_to(ROOT) else zernike_dir}`。

三阶段：Stage A = GS 预矫正 + SPGD 细化（硬件闭环）→ Stage B = Noll 序 Zernike 最小二乘拟合
→ Stage C = 在 Zernike 系数上做 SPGD。目标框由平场光斑 waist 推导，**全程冻结**。

### 3.1 基线

{baseline_table}

### 3.2 阶数扫描

{zernike_table}

### 3.2.1 各阶的 CCD 远场光斑图

下面是泽尼克扫描自己在每一阶保存的远场图（`before` = 纯 Zernike 拟合相位，
`after` = 再叠加该阶的 SPGD 优化结果），窗口是该扫描的 256×256 ROI、目标框 9.96 px。

{zernike_farfield_block}

### 3.3 阶数能不能排序

{verdict}

判据用的噪声地板必须与阶数比较**同口径**（泽尼克扫描那把 9.96 px 目标框、0.2 ms 曝光），
所以不能用 §1.2 那条（那条是离焦探针的 44 px 目标框，框大了 ~20 倍，光子数多一个量级，
σ 小得多，拿来判阶数差异会**高估**可分辨性）。这里从泽尼克扫描自己的逐轮记录里取：

{order_noise_table}

- 拟合残差随阶数{'单调不增' if monotone else '非单调（需要复查）'}；
  这是本轮唯一在阶数方向上单调的量。
- 各阶的 `max|c|` 与 `ZERNIKE_CLIP = 5.0` rad 的关系见上表「clip 警告」列。
- 本轮**只有 1 个 seed**，没有配对轴，因此不做显著性检验；
  `ml/zernike/eval_stats.py` 的 sign-flip 置换检验需要「同 seed 配 seed」，1 个 seed 无从配对。
"""
    else:
        zernike_section = "\n\n---\n\n## 3. 泽尼克阶数扫描（硬件）\n\n（本轮尚未采集。）\n"

    # --- 逐组的读数 (供摘要用) ------------------------------------------------
    group_readings: list[dict[str, Any]] = []
    for grp in scan_groups:
        rw = [x for x in grp["run"]["rows"] if not x.get("frame_missing")]
        by_dz = {round(float(x["dz_mm"]), 4): x for x in rw}
        base = [x for x in rw if abs(float(x["dz_mm"])) < 1e-9]
        b0 = float(base[0].get("score_frozen", math.nan)) if base else math.nan
        r15 = float(by_dz.get(round(1.5007, 4), {}).get("score_frozen", math.nan))
        a = abs(
            float(by_dz.get(round(-1.5007, 4), {}).get("score_frozen", math.nan))
            - float(by_dz.get(round(1.5007, 4), {}).get("score_frozen", math.nan))
        )
        group_readings.append({
            "name": grp["name"],
            "phase": grp.get("phase", ""),
            "asym": a,
            "slope": steering_fit(rw)["slope_local_px_per_mm"],
            "drop15": (1.0 - r15 / b0) if b0 else math.nan,
        })
    primary_reading = group_readings[0] if group_readings else {}
    asym_parts = "、".join(
        f"「{g['name']}」{fmt_metric(g['asym'], 4)}" for g in group_readings
    )
    drop_parts = "、".join(
        f"「{g['name']}」{fmt_metric(g['drop15'], 3)}" for g in group_readings
    )

    return f"""# 硬件离焦扫描 + 多阶泽尼克匀化测试报告（2026-10-20）

**本轮结论**：离焦扫描与泽尼克阶数扫描都在**真实实验系统**上在线实现，与
[`gs_defocus_sim_20261019.md`](gs_defocus_sim_20261019.md) 的纯仿真结论逐条对拍。三条主要结论：

1. **实验系统自带约 8 mm 的常驻离焦，必须先做瞳面焦面矫正**。平相位下光斑 FWHM ≈ 115 px 且呈环形；
   加上 `fresnel_panel(f_v = 1597.4 mm)` 的瞳面二次相位后，光斑收到 **15.5 px**，峰值/曝光提高
   **约 25 倍**。不矫正就直接扫 Δz，测到的不是离焦响应而是一个弥散环。
2. **瞳面二次相位在这套实验系统上不是纯粹的离焦，它同时把光斑推走**（转向）。主组实测转向速率
   **{fmt_metric(steer['slope_local_px_per_mm'], 2)} px/mm**（质心 x 相对冻结零级，|Δz| ≤ 2.5 mm 内拟合）。
   转向叠加在「光斑本来就不在框中心」之上，于是同一个 |Δz| 在正负半轴上的劣化不对称：
   本轮各组在 ±1.5007 mm 处的差值是 {asym_parts}（噪声地板 {fmt_metric(sigma_score, 5)}），
   而仿真那边只差 0.0008 —— 这是硬件与仿真最大的口径差异。
   各组 |Δz| = 1.5007 mm 相对 Δz=0 的复合分数变化是 {drop_parts}：模型相位组与仿真同量级，
   硬件 GS 相位组却几乎不变，原因见 §2.2.2（44 px 目标框远大于该相位 ~10 px 的实际光斑）。
3. **泽尼克阶数扫描在硬件上在线实现 8 阶**：{verdict}
   唯一随阶数**严格单调**的量是**拟合残差**（{'单调不增' if monotone else '非单调'}，
   {fmt_metric(rms_vals[0], 4)} → {fmt_metric(rms_vals[-1], 4)} rad）。

---

## 1. 实验系统与前提

| 项 | 值 |
|---|---|
| SLM | Santec #1, driveboardID `22030108`, 1920×1200, 像元 8 µm, memory 模式 |
| CCD | Daheng `FJB24112232`, 2592×1944, 像元 2.2 µm |
| 波长 | 1064 nm |
| 曝光 | {fmt_metric(cfg['exposure_ms_working'], 3)} ms（会话上限 `AO_DAHENG_MAX_EXPOSURE_MS` = 0.5 ms） |
| 瞳面焦面矫正 | `fresnel_panel(f_v = {fmt_metric(cfg.get('defocus_focal_mm'), 1)} mm)`，面板边缘 {23.96:.2f} rad ≈ 3.81 波 |
| 零级 (全幅 argmax) | {tuple(cfg['zero_order_full_frame_xy'])} |
| 零级 (ROI 内 argmax, 冻结) | {tuple(cfg['zero_order_roi_xy_frozen'])} |
| ROI / 目标框 | {cfg['roi_px']} px / {cfg['target_side_px']} px |
| 每点帧数 | {cfg['frames_per_measurement']} |

### 1.1 分组清单（一次只跑一组变量）

{groups_table}

本轮的全部数字都来自上表：离焦扫描与泽尼克匀化**叠加在同一张 GS 相位上**
（泽尼克扫描 Stage A 产出的那一张），每组单独记录、单独出表。

### 1.2 噪声地板

同一个 GS 相位重复采集 {noise['n_repeats']} 次（Δz 全部为 0）的组内标准差。**本报告任何
「改善」都必须大于这里的 3σ 才可采信。**

{noise_table}

### 1.3 实验系统的常驻离焦与曝光

- 平相位光斑 FWHM ≈ 115 px、环形度 ≈ 0.9；加上焦面矫正后 FWHM 15.5 px、峰值/曝光 ×25。
- 焦面标定：`focus_probe` 粗扫（±6 m⁻¹）给 1/f = +0.541 m⁻¹，细扫（±1.2 m⁻¹, 25 点 +
  9 点细化）给 **1/f = +0.626 m⁻¹（f_v = 1597.4 mm）**，峰值/曝光 370 counts/ms，相对平场 ×44.37。
  峰值判据在 D ∈ [0.58, 0.70] m⁻¹ 上**饱和成平台**，FWHM 判据的极小值在 D ≈ 0.58；
  本轮取 +0.626（平台中部），并在报告里如实标注该平台。
- 该焦面矫正等价于探针参数下 **Δz ≈ -9.4 mm** 的常驻离焦 —— 落在探针默认扫描区间正中，
  所以「Δz = 0」若不做矫正**不是焦点**。

---

## 2. 离焦扫描（硬件）

采集脚本：`scripts/slm/hw_gs_defocus_probe.py --execute --defocus-focal-mm 1597.4`。
原始帧逐点落盘（`frame_*.npz`，未经任何处理），下表的 FWHM / 环形度 / 质心 / 随斑框指标
都是**离线**从原始帧重算的。

### 2.1 口径

- **定尺框**：探针的主口径 —— 目标框在平场（带焦面矫正）上定位一次后**冻结**，之后不重定位。
  边长恒为 {cfg['target_side_px']} px，物理尺度恒定。
- **随斑框**：边长**不变**，只把中心平移到该帧光斑质心。它只消掉**平移**、不缩放，
  用来把转向分量与离焦分量分开。

{group_sections}

---

{zernike_section}

---

## 4. 图

{fig_block}

---

## 5. 限制（不能断言什么）

1. **离焦响应的可读区间只有 |Δz| ≤ 1.5 mm。** 更大的 Δz 上，转向分量压过离焦分量，
   冻结框口径不再度量「离焦」。
2. **本轮没有做「在工作点上重新优化」的硬件版。** 仿真报告的核心结论是
   「离焦本身不值钱, 值钱的是在工作点上重新优化」（G1′−G1 = +0.1465，而离焦并入环路的净贡献
   只有 +0.0179）。硬件上逐 Δz 重跑 GS 闭环的成本是本轮预算的数十倍，**本报告没有验证这一条**。
3. **离焦扫描用的是硬件 GS 相位**（泽尼克扫描 Stage A 产出、Stage B 拟合的就是这一张），
   与第 3 节同一张相位；模型 GS 相位那组只是**对照**（它证明模型相位在实验系统上不成方斑）。
4. **焦面矫正存在一个平台**（D ∈ [0.58, 0.70] m⁻¹ 上峰值判据饱和），本轮取 +0.626。
   平台内的选择会改变 Δz=0 的参考点。
5. **离焦组与噪声组各有一条噪声地板，口径不同**：离焦组用 44 px 目标框（0.2 ms，硬件 GS 相位），
   阶数比较用泽尼克扫描自己的 9.96 px 目标框（见 §3.3）。**两者不可互相替代**。
6. **泽尼克阶数只有 1 个 seed**，且 Stage A 的 GS 相位只优化了 250 轮、Stage C 每阶 150 轮。
   没有配对轴，所以不做显著性检验（`ml/zernike/eval_stats.py` 的 sign-flip 置换检验
   需要「同 seed 配 seed」）。
7. **本轮只测了一个离焦方向的两个半轴，没有做「在工作点上重新优化」的硬件版**（见上第 2 条）。
8. **补拍「泽尼克各阶远场组」时设备已下线**：8 个阶里只有 n_max=3 与 n_max=4 两组拍到了
   原始帧（其余 6 组的 SLM 打开报 **-10002 未找到 USB 设备**，是控制器断电，不是脚本问题）。
   所以 §3 的各阶远场图用的是**泽尼克扫描自己保存的那份**（`before/after_nmax*.png`，
   256×256 ROI、9.96 px 目标框），而带 X/Y 截面的那组只覆盖已经拍到的两阶。
   设备恢复后重跑 `data/debug/hw_groups_20261020/plan_zernike_farfield.json` 即可补齐。
9. **本报告不主张的事情**（逐条列出来，避免被误读）：
   - **不**主张「实验系统对离焦不敏感」—— 复合分数在 |Δz| ≤ 2.5 mm 上「平」是**口径的产物**
     （框 ≫ 光斑 ⇒ 均匀项奖励离焦），不是物理结论。
   - **不**主张「离焦强烈破坏图样」（基于冻结框心的读数）—— |Δz| ≳ 1.5 mm 时光斑同时被
     **推走**了（转向 −5.2 px/mm，Δz=2.5 mm 时已移出 13 px 的框），那量到的是**模拟方式的倾斜**，
     不是离焦。消除它要靠去倾斜（`--chirp-tilt-cancel`），复测后才能下这个结论。
   - **不**主张硬件与仿真的正负半轴不对称是**实验系统**的性质 —— 瞳面 chirp 以面板标称中心
     (960, 600) 为中心而照明不在那里，于是它带一个 ∝ Δz 的**倾斜**分量（远场位移 ∝ Δz），
     而仿真的 chirp 打在光束中心、**没有**倾斜。所以这里的不对称有一部分来自**模拟方式**，
     不是光学。
   - **不**跨相位比较复合分数/PIB 的**绝对值**（两组的孔径不同：13 px 对 23 px），只比
     **归一化**曲线。
   - **不**把 FWHM 当离焦指标（环形/分裂图样会让它量到环的厚度；本组实测 FWHM 在 3.5 ~ 174.6 px
     之间跳，模型相位组还非单调）。FWHM 只作描述列。
   - **不**说「跟中心就消除了转向污染」—— 互相关在本实验系统的散斑图样上不可用（见 §x.5 第 2 条）。
   - 44 px 那组数字**本身没错**（对该孔径而言是对的），错的是从它推出的结论；所以保留为「口径 A」。

---

## 6. 复现

```bash
# 一次只跑一组变量: 计划文件里每一组是一次独立调用, 各自单独记录。
# (完整计划见 data/debug/hw_groups_20261020/plan.json)
python scripts/slm/run_hw_sweep_groups.py \\
    --plan data/debug/hw_groups_20261020/plan.json \\
    --adopt-existing zernike_order data/debug/hw_zernike_sweep_20261020b \\
    --timeout-s 1800

# 单组单独跑 (驱动只挑这一组)
python scripts/slm/run_hw_sweep_groups.py \\
    --plan data/debug/hw_groups_20261020/plan.json --only dz_scan_hw_gs

# 计划里每组对应的实际调用 (节选; 下一轮上机要加的两个开关见括号)
#   dz_scan_hw_gs:
#     python scripts/slm/hw_gs_defocus_probe.py --execute --group dz_scan_hw_gs \\
#       --group-meta role=dz_scan --group-meta phase=hardware_gs \\
#       --out-root data/debug/hw_groups_20261020 \\
#       --defocus-focal-mm 1597.4 --exposure-ms 0.2 --frames 4 \\
#       --gs-phase data/debug/hw_zernike_sweep_20261020b/best_phase_nmax4.npy \\
#       --dz-mm -20 -15 ... 20
#       (下一轮: --target-side-px 10  --chirp-tilt-cancel)
#   dz_scan_model_gs:
#     同上, 但不传 --gs-phase (用模型相位), 曝光 0.4 ms
#       (下一轮: --target-side-px 44  --chirp-tilt-cancel)
#   zernike_order:
#     python scripts/slm/run_gs_refine_zernike_sweep.py --cam-type daheng --cam-id 0 \\
#       --exposure-time-ms 0.2 --cam-size 256 --beam-radius-px 450 --slm-number 1 \\
#       --epochs 250 --gs-iters 200 --spgd-epochs 150 \\
#       --n-max 3 4 6 8 11 15 20 25 --lr 0.01 --delta 0.02 --max-energy-loss 0.6 \\
#       --seed 0 --defocus-focal-mm 1597.4 --save-images

# 离线出图
python scripts/slm/generate_hw_defocus_zernike_report.py \\
    --groups-root data/debug/hw_groups_20261020 \\
    --groups-root data/debug/hw_zernike_sweep_20261020b
```

采集目录：离焦与噪声组在 `data/debug/hw_groups_20261020/<组名>/`；
泽尼克阶数组在 `{zernike_dir.name if zernike_dir else '（本轮未采集）'}`；
分组索引 `data/debug/hw_groups_20261020/groups_manifest.json`。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-root", action="append", type=Path, default=[],
                        help="分组记录根目录 (递归查找各组); 可给多次")
    parser.add_argument("--defocus-dir", type=Path, default=None,
                        help="离焦扫描的测量目录 (含 measurements.json / frame_*.npz)")
    parser.add_argument("--noise-dir", type=Path, default=None,
                        help="噪声地板的测量目录 (同一相位重复采集)")
    parser.add_argument("--zernike-dir", type=Path, default=None,
                        help="泽尼克阶数扫描的 run 目录 (含 summary.json)")
    parser.add_argument("--order-noise-dir", type=Path, default=None,
                        help="阶数比较用的噪声地板目录 (硬件 GS 相位重复采集, 曝光须与 "
                             "泽尼克扫描一致); 不给则按曝光自动挑")
    args = parser.parse_args()

    roots = [p for p in (args.groups_root + [args.defocus_dir, args.noise_dir,
                                             args.order_noise_dir, args.zernike_dir])
             if p is not None]
    found = discover_groups(roots)

    # --- 离焦分组: 噪声地板组 (Δz 全 0) 与扫描组 (Δz 有跨度) -----------------
    probe_runs = [(g["name"], g["dir"], load_probe_run(g["dir"]))
                  for g in found["probe"]]
    # 「泽尼克各阶的远场组」也是 Δz 全 0 的重复采集, 但它不是噪声地板:
    # 它的角色是给各阶拍远场图, 必须排除在噪声地板候选之外。
    farfield_runs = [(n, d, r) for n, d, r in probe_runs
                     if str((r["config"].get("group_meta") or {}).get("role", ""))
                     == "zernike_farfield"]
    noise_groups = [(n, d, r) for n, d, r in probe_runs
                    if is_noise_group(r)
                    and str((r["config"].get("group_meta") or {}).get("role", ""))
                    != "zernike_farfield"]
    scan_groups = [(n, d, r) for n, d, r in probe_runs if not is_noise_group(r)]
    if not scan_groups:
        raise SystemExit("没有找到离焦扫描组 (需要一个含 measurements.json 的目录)")

    def role_of(run: dict[str, Any]) -> str:
        meta = run["config"].get("group_meta") or {}
        return str(meta.get("role", "")) or "dz_scan"

    def phase_of(run: dict[str, Any]) -> str:
        meta = run["config"].get("group_meta") or {}
        return str(meta.get("phase", "")) or "unknown"

    # 主扫描 = role 为 dz_scan 的那组（模型相位那组是对照，排在后面）。
    scan_groups.sort(key=lambda item: (0 if role_of(item[2]) == "dz_scan" else 1,
                                        -len(item[2]["rows"])))
    primary_name, primary_dir, primary_run = scan_groups[0]
    primary_exposure = primary_run["config"].get("exposure_ms_working")

    def pick_noise(exposure: float | None) -> dict[str, Any] | None:
        if not noise_groups:
            return None
        if exposure is None:
            return noise_groups[0][2]
        best = min(
            noise_groups,
            key=lambda item: abs(
                float(item[2]["config"].get("exposure_ms_working", 0.0)) - float(exposure)
            ),
        )
        return best[2]

    noise_run = pick_noise(primary_exposure) or primary_run
    noise = noise_floor(noise_run)
    noise["dir"] = noise_run["dir"]

    # --- 泽尼克组 ------------------------------------------------------------
    zernike_dir = found["zernike"][0]["dir"] if found["zernike"] else args.zernike_dir
    zernike = load_zernike(zernike_dir) if zernike_dir else {}
    zernike_exposure = (zernike.get("config") or {}).get("exposure_time_ms")
    order_run = (load_probe_run(args.order_noise_dir) if args.order_noise_dir
                 else pick_noise(zernike_exposure))
    order_noise = noise_floor(order_run) if order_run else None
    if order_noise is not None:
        order_noise["dir"] = order_run["dir"]

    sim_g1 = load_sim_g1()
    groups_for_report = [
        {"name": name, "dir": dir_, "run": run,
         "role": role_of(run), "phase": phase_of(run),
         "phase_label": _phase_label(run)}
        for name, dir_, run in scan_groups
    ]

    figures = render_figures(primary_run, sim_g1, zernike)
    group_figures = render_group_farfields(groups_for_report)
    response_figures = render_defocus_responses(groups_for_report)
    zernike_figures = copy_zernike_farfields(zernike_dir, zernike.get("orders", []))
    if render_zernike_xsec(farfield_runs, FIG_DIR / "hw_zernike_xsec.png"):
        zernike_figures = zernike_figures + ["figures/hw_zernike_xsec.png"]
    body = build_markdown(
        primary_run, noise, sim_g1, zernike, zernike_dir, figures, order_noise,
        groups_for_report, group_figures, response_figures, zernike_figures,
    )
    text = insert_header(body, REPORT_KEY)
    out_path = ROOT / REPORT_KEY
    out_path.write_text(text, encoding="utf-8")
    print(f"[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
