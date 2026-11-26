"""倾斜 (闪耀光栅) 与整形分别能从平相位 0 级利用多少能量 (2026-11-26)。

背景
----
2026-11-26 把「SLM 衍射效率」重新定义为 **从平相位 0 级可以转移走的能量份额**::

    η = E_blaze / F0

``F0`` 是平相位帧零级窗口的暗减积分 (探针
``relative_efficiency.baseline_zero_window_counts``), ``E_blaze`` 是闪耀帧
闪耀级窗口能量。旧口径 ``lobe.ratios[blaze_order]`` 的分母是闪耀帧三窗之和
(E0+E+1+E-1), 不是输入光, 会把效率抬高 (同一探针 39.78% vs 19.27%)。

本脚本只读已归档数据, 不碰设备、不写标定文件, 把三组历史口径并排算出来:

1. 探针 ``data/diffraction_efficiency/*/summary.json`` —— η (F0 零级窗口径)、
   零级残余、三窗未计入份额;
2. 效率匹配 ``data/debug/efficiency_match/20261123_050735Z/analysis.json`` ——
   同一 F0 圆盘口径下的 η_d = G1/F0 与 η_s = (S - S0)/F0, 含判定;
3. 方形整形扫描 ``report/shape_trials/hw_gs_square_scans/artifacts/*/sides``
   —— pib = 目标方框 / 整帧 (整帧口径), 不能与 η 相除。

三种分母不同, 结论段明确写出「不能互相代入」。

用法::

    python scripts/slm/analyze_diffraction_vs_shaping_utilization_20261126.py
    python scripts/slm/analyze_diffraction_vs_shaping_utilization_20261126.py --json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROBE_GLOB = "data/diffraction_efficiency/*/summary.json"
MATCH_ANALYSIS = Path("data/debug/efficiency_match/20261123_050735Z/analysis.json")
PIB_BATCHES: dict[str, Path] = {
    "20261023_gs_square (20 ms)": Path(
        "report/shape_trials/hw_gs_square_scans/artifacts/20261023_gs_square"
        "/sides/measurements.csv"
    ),
    "20261023_atomic_full (180 ms)": Path(
        "report/shape_trials/hw_gs_square_scans/artifacts/20261023_atomic_full"
        "/sides/measurements.csv"
    ),
}
FIRST_ORDERS: tuple[str, ...] = ("plus_one", "minus_one")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _drift_key(frame: dict[str, Any]) -> str:
    """漂移标记的复合键: ``pass:role#ordinal`` (frames[].id 是短名)。"""
    return f"{frame.get('pass')}:{frame.get('role')}#{frame.get('ordinal')}"


def _blaze_order(relative: dict[str, Any], lobe: dict[str, Any]) -> str:
    """闪耀级标注: 新探针在 relative_efficiency, 旧探针在 lobe.orders。"""
    order = relative.get("blaze_order") or lobe.get("orders", {}).get("blaze_order")
    if order:
        return str(order)
    energies = lobe.get("energies", {})
    present = [o for o in FIRST_ORDERS if o in energies]
    if not present:
        raise KeyError("探针没有 ±1 级能量, 无法确定闪耀级")
    return max(present, key=lambda o: float(energies[o]))


def probe_rows() -> list[dict[str, Any]]:
    """每个探针一行: η (F0 零级窗口径) 与零级残余/未计入份额。"""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(".").glob(PROBE_GLOB)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        lobe = payload["lobe"]
        relative = payload.get("relative_efficiency") or {}
        energies = {k: float(v) for k, v in lobe.get("energies", {}).items()}
        flat_zero = relative.get("baseline_zero_window_counts")
        order = _blaze_order(relative, lobe)
        blaze = energies.get(order, 0.0)
        if flat_zero and "blaze_energy_share" in relative:
            eta = float(relative["blaze_energy_share"])
        elif flat_zero:
            eta = blaze / float(flat_zero)
        else:
            eta = None
        quality = payload.get("quality") or {}
        peaks = payload.get("peaks") or {}
        full_scale = payload.get("full_scale") or {}
        saturated = any(
            bool(full_scale.get(world)) and peaks.get(f"{world}_counts", 0.0) >= full_scale[world]
            for world in ("baseline", "blazed")
        )
        rows.append(
            {
                "probe": path.parent.name,
                "timestamp_utc": payload.get("timestamp_utc"),
                "gray_depth": (payload.get("config") or {}).get("gray_depth"),
                "effective_exposure_ms": (quality.get("exposure_settle") or {}).get(
                    "effective_ms"
                ),
                "qualified": quality.get("qualified"),
                "n_slots_blaze": (payload.get("slots") or {}).get("blaze"),
                "peaks": {k: peaks.get(k) for k in ("baseline_counts", "blazed_counts")},
                "saturated": saturated,
                "roi_radius_px": lobe.get("roi_radius_px"),
                "separation_px": lobe.get("separation_px"),
                "blaze_order": order,
                "blaze_energy": blaze,
                "flat_zero_energy": float(flat_zero) if flat_zero else None,
                "eta": eta,
                "zero_residual_over_f0": (
                    energies["zero"] / float(flat_zero)
                    if flat_zero and "zero" in energies
                    else None
                ),
                "unaccounted_over_f0": (
                    1.0 - sum(energies.values()) / float(flat_zero)
                    if flat_zero
                    else None
                ),
                "lobe_share_old": (lobe.get("ratios") or {}).get(order),
            }
        )
    return rows


def match_section(path: Path) -> dict[str, Any] | None:
    """效率匹配: 同 F0 圆盘口径的 η_d / η_s 与判定。"""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload.get("stats") or {}
    photo = payload.get("photometry") or {}
    f0 = photo.get("f0_mean")
    s0 = photo.get("s0_mean")
    rounds = [
        {
            "pass": row.get("pass"),
            "blaze_frame": row.get("blaze_frame"),
            "shaping_frame": row.get("shaping_frame"),
            "eta_d": row.get("eta_d"),
            "eta_s": row.get("eta_s"),
            "d": row.get("d"),
        }
        for row in stats.get("rounds") or []
    ]
    # 脚本自身只报 sd; 均值在这里由逐轮值算, 与 stats.mean_d 交叉核对。
    eta_d_values = [float(r["eta_d"]) for r in rounds if r.get("eta_d") is not None]
    eta_s_values = [float(r["eta_s"]) for r in rounds if r.get("eta_s") is not None]
    d_values = [float(r["d"]) for r in rounds if r.get("d") is not None]
    # 漂移标记用的是 ``pass:role#ordinal`` 复合键 (frames[].id 只是短名)。
    drift_labels = set((payload.get("drift") or {}).get("drift_flat_labels") or [])
    box_rows: list[dict[str, Any]] = []
    for frame in payload.get("frames") or []:
        if frame.get("role") != "shaping" or not f0:
            continue
        s_value = frame.get("s")
        box_rows.append(
            {
                "id": frame.get("id"),
                "pass": frame.get("pass"),
                "s": s_value,
                "box_over_f0": (float(s_value) / float(f0)) if s_value is not None else None,
            }
        )
    return {
        "dir": str(path.parent),
        "verdict": payload.get("verdict"),
        "reasons": payload.get("reasons") or payload.get("reason"),
        "config": payload.get("config"),
        "exposure_ms": sorted(
            {
                float(f["exposure_ms"])
                for f in payload.get("frames") or []
                if f.get("exposure_ms") is not None
            }
        ),
        "thresholds": {
            k: (payload.get("thresholds") or {}).get(k)
            for k in ("r0_px", "r1_px", "side_px", "separation_px")
        },
        "f0_mean": f0,
        "f0_sd": photo.get("f0_sd"),
        "s0_mean": s0,
        "flat_box_over_f0": (float(s0) / float(f0)) if f0 and s0 else None,
        "stable_flat_frames": len(
            [
                f
                for f in payload.get("frames") or []
                if f.get("role", "").startswith("flat") and _drift_key(f) not in drift_labels
            ]
        ),
        "n_flat_frames": len(
            [f for f in payload.get("frames") or [] if f.get("role", "").startswith("flat")]
        ),
        "drift_warnings": (payload.get("drift") or {}).get("drift_warnings"),
        "rounds": rounds,
        "eta_d_mean": _mean(eta_d_values),
        "eta_s_mean": _mean(eta_s_values),
        "d_mean": _mean(d_values),
        "stats_mean_d": stats.get("mean_d"),
        "ci95": [stats.get("ci95_low"), stats.get("ci95_high")],
        "sd_eta_d": stats.get("sd_eta_d"),
        "sd_eta_s": stats.get("sd_eta_s"),
        "box_rows": box_rows,
        "box_over_f0_mean": _mean(
            [float(r["box_over_f0"]) for r in box_rows if r["box_over_f0"] is not None]
        ),
    }


def pib_rows(path: Path) -> list[dict[str, Any]]:
    """方形整形扫描: pib = 目标方框 / 整帧 (整帧口径)。"""
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: list[dict[str, Any]] = []
    for row in rows:
        peak = float(row.get("frame_peak") or "nan")
        out.append(
            {
                "side_px": int(float(row["side_px"])),
                "exposure_ms": float(row.get("exposure_ms_readback") or "nan"),
                "pib": float(row["pib"]),
                "frame_peak": peak,
                "saturated": peak >= 255.0,
            }
        )
    return out


def _pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def _render_probes(probes: list[dict[str, Any]]) -> None:
    print("## 1. 探针: 倾斜 (闪耀光栅) 的 η = E_blaze / F0 (平相位零级窗口径)\n")
    print(
        "| 探针 | 有效曝光 (ms) | 灰度 | 窗口半径 (px) | 级次间距 (px) | 闪耀级 | "
        "F0 (counts) | E_blaze | η (F0 口径) | 零级残余/F0 | 三窗未计入/F0 | 旧口径 ratios |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in probes:
        print(
            "| {probe} | {exp} | {gray} | {roi} | {sep} | {order} | {f0} | {bl} | {eta} | "
            "{zero} | {un} | {old} |".format(
                probe=row["probe"],
                exp=(
                    "—"
                    if row["effective_exposure_ms"] is None
                    else f"{row['effective_exposure_ms']}"
                ),
                gray=row["gray_depth"],
                roi=f"{float(row['roi_radius_px']):g}" if row["roi_radius_px"] else "—",
                sep=f"{float(row['separation_px']):g}" if row["separation_px"] else "—",
                order=row["blaze_order"],
                f0=(
                    "—"
                    if row["flat_zero_energy"] is None
                    else f"{row['flat_zero_energy']:.1f}"
                ),
                bl=f"{row['blaze_energy']:.1f}",
                eta=_pct(row["eta"]),
                zero=_pct(row["zero_residual_over_f0"]),
                un=_pct(row["unaccounted_over_f0"]),
                old=_pct(row["lobe_share_old"]),
            )
        )
    print()


def _render_match(match: dict[str, Any] | None) -> None:
    print("## 2. 效率匹配 (2026-11-23): 同一 F0 圆盘口径下的 η_d / η_s\n")
    if match is None:
        print("(缺少 analysis.json)\n")
        return
    cfg = match["config"] or {}
    th = match["thresholds"]
    print(
        "采集: 曝光 {exp} ms, 相机满量程 {fs} counts, R0={r0} px 圆盘 / "
        "R1={r1} px 圆盘 / 目标方框 {side} px, 级次间距 {sep} px; "
        "稳定平场帧 {stable}/{total} 帧参与 F0 平均。\n".format(
            exp=", ".join(f"{e:g}" for e in match["exposure_ms"]) or "—",
            fs=cfg.get("camera_full_scale"),
            r0=th["r0_px"],
            r1=th["r1_px"],
            side=th["side_px"],
            sep=th["separation_px"],
            stable=match["stable_flat_frames"],
            total=match["n_flat_frames"],
        )
    )
    print("| 轮 (blaze / shaping) | pass | η_d = G1/F0 (倾斜) | η_s = (S-S0)/F0 (整形) | d = η_s - η_d |")
    print("|---|---|---|---|---|")
    for row in match["rounds"]:
        print(
            "| {bf} / {sf} | {p} | {d} | {s} | {diff} |".format(
                bf=row["blaze_frame"],
                sf=row["shaping_frame"],
                p=row["pass"],
                d=_pct(row["eta_d"]),
                s=_pct(row["eta_s"]),
                diff="—" if row["d"] is None else f"{row['d']:.4f}",
            )
        )
    print(
        "\n均值: η_d = {d} (sd {sd_d:.4f}), η_s = {s} (sd {sd_s:.4f}), "
        "d = {diff:.4f}, 脚本 mean_d = {md:.4f}, CI95 = [{lo:.4f}, {hi:.4f}]".format(
            d=_pct(match["eta_d_mean"]),
            sd_d=match["sd_eta_d"],
            s=_pct(match["eta_s_mean"]),
            sd_s=match["sd_eta_s"],
            diff=match["d_mean"],
            md=match["stats_mean_d"],
            lo=match["ci95"][0],
            hi=match["ci95"][1],
        )
    )
    print(
        "整形帧目标方框占比 S/F0: 均值 {mean} (逐帧 {rows}); "
        "平相位同框 S0/F0 = {flat}。".format(
            mean=_pct(match["box_over_f0_mean"]),
            rows=", ".join(_pct(r["box_over_f0"]) for r in match["box_rows"]),
            flat=_pct(match["flat_box_over_f0"]),
        )
    )
    print(f"\nF0 = {match['f0_mean']:.1f} counts (sd {match['f0_sd']}), S0 = {match['s0_mean']:.1f}")
    print(f"\n判定: **{match['verdict']}**, 原因 {match['reasons']}")
    for warning in match["drift_warnings"] or []:
        print(f"- 漂移警告: {warning}")
    print()


def _render_pib(pib: dict[str, list[dict[str, Any]]]) -> None:
    print("## 3. 方形整形扫描: pib = 目标方框 / 整帧 (整帧口径)\n")
    for name, rows in pib.items():
        print(f"### {name}\n")
        if not rows:
            print("(缺少 measurements.csv)\n")
            continue
        print("| side_px | exposure_ms | pib | frame_peak | 饱和 |")
        print("|---|---|---|---|---|")
        for row in rows:
            print(
                f"| {row['side_px']} | {row['exposure_ms']:.1f} | {_pct(row['pib'])} | "
                f"{row['frame_peak']:.1f} | {'是' if row['saturated'] else '否'} |"
            )
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非 markdown")
    args = parser.parse_args()

    probes = probe_rows()
    match = match_section(MATCH_ANALYSIS)
    pib = {name: pib_rows(path) for name, path in PIB_BATCHES.items()}

    if args.json:
        print(
            json.dumps(
                {"probes": probes, "efficiency_match": match, "pib": pib},
                ensure_ascii=False,
                indent=1,
            )
        )
        return 0

    _render_probes(probes)
    _render_match(match)
    _render_pib(pib)

    print("## 4. 口径提醒\n")
    print("- η (探针) 的分母是**平相位零级窗口 F0**; η_d/η_s (效率匹配) 的分母是**平相位 R0 圆盘**;")
    print("  pib 与闭环 `window_fraction` 的分母是**整帧**。三者不能相除, 也不能相加。")
    print("- 饱和点的窗口/方框能量被截断: 20261023_atomic_full 全部点 frame_peak=255,")
    print("  probe_real_20261121 (5 ms) 的 blazed 帧峰值也顶到满量程。")
    print("- 效率匹配的 F0 只来自一份稳定平场帧 (其余被漂移轮标记), 判定为 undeterminable。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
