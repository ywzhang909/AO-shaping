"""光路变化后的自动校准自检：哪些存量标定还能用、哪些必须重测/重拟合。

2026-10-23 的场景：把 CCD 移到焦平面之后，此前所有**与离焦强相关**的标定都失效：

* 冻结模型的 ``defocus_diopter``（拟合时约 −0.645 m⁻¹ ⇒ 等效焦距 ~1.55 m）描述的是
  **旧光路**的离焦；CCD 到焦面后它应当≈0，否则模型的远场形状会系统性偏；
* ``focus.json``（``best_focal_mm`` ~1783 mm、增益 31×）是**旧离焦状态**下的最佳补偿；
* 模型采样窗（``camera_xy``）的中心建在旧零点上（实测已偏 72 px）；
* 目标尺寸标定（``signature`` + 有效窗口）是在旧光路下扫出来的。

本脚本只读存量产物、离线给出逐项判定与处置建议；能离线重建的（采样窗）直接给出新值，
需要设备的（焦点、模型重拟合、尺寸重扫）明确标 ``needs-device``。
"""

from __future__ import annotations

import argparse
from loguru import logger
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/debug/defocus_differentiable/20261023_101822"
CHECKPOINT = ROOT / "data/debug/physical_forward_defocus/uniformity_refit_20261022_r1"
FOCUS = ROOT / "data/debug/physical_forward_defocus/alignment_20261022_1812/focus.json"
CALIBRATION = ROOT / "data/calibration/target_size_calibration.json"


def _row(item: str, stored: str, expected: str, verdict: str, action: str) -> dict:
    return dict(item=item, stored=stored, expected_after_change=expected, verdict=verdict, action=action)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="存在需要重做的项时返回非零退出码")
    parser.add_argument("--json", type=Path, default=None, help="把报告写到该路径")
    args = parser.parse_args()

    rows: list[dict] = []

    # ---- 1. 模型的离焦项 ------------------------------------------------------
    parameters = json.loads((CHECKPOINT / "result.json").read_text(encoding="utf-8"))["parameters"]
    diopter = float(parameters["defocus_diopter"])
    implied_focal_mm = 1000.0 / abs(diopter) if diopter else float("inf")
    rows.append(_row(
        "模型离焦项 defocus_diopter", f"{diopter:+.4f} m⁻¹（等效焦距 {implied_focal_mm:.0f} mm）",
        "≈ 0（CCD 已在焦面）",
        "stale" if abs(diopter) > 0.05 else "reuse",
        "needs-device：在焦面上重采探针并重拟合（先做 B± 方向检验再用于优化）"))

    # ---- 2. 焦面补偿标定 ------------------------------------------------------
    if FOCUS.exists():
        focus = json.loads(FOCUS.read_text(encoding="utf-8"))
        best = float(focus.get("focus", {}).get("best_focal_mm", float("nan")))
        gain = float(focus.get("focus", {}).get("gain_vs_flat", float("nan")))
        rows.append(_row(
            "焦面补偿 focus.json", f"best_focal_mm={best:.1f}，增益 {gain:.1f}×",
            "补偿量应显著变小（原为旧离焦的补偿）",
            "stale", "needs-device：重跑焦点扫描（probe_session_focus / focus_preflight）"))
        implied = 1000.0 / best if best and np.isfinite(best) else float("nan")
        rows.append(_row(
            "两者一致性（模型 |d| vs 1/f）", f"|d|/（1/f）= {abs(diopter) / implied:.2f}" if implied else "不可比",
            "光路变化后两者应同时趋近 0", "info",
            "若变化后两者仍相差 >30% ⇒ 说明其一未重测"))
    else:
        rows.append(_row("焦面补偿 focus.json", "缺失", "—", "missing", "needs-device：重新标定"))

    # ---- 3. 模型采样窗 --------------------------------------------------------
    with np.load(CHECKPOINT / "inputs.npz") as payload:
        xy = payload["camera_xy"]
    window_centre = (float(xy[..., 0].min() + xy[..., 0].max()) / 2.0,
                     float(xy[..., 1].min() + xy[..., 1].max()) / 2.0)
    frames = sorted((RUN / "stage_e/rep0").glob("p0_A1_f*.npy"))
    if frames:
        frame = np.mean([np.load(f) for f in frames], axis=0)
        row, col = np.unravel_index(int(np.argmax(frame)), frame.shape)
        spot = (float(col), float(row))
        offset = float(np.hypot(spot[0] - window_centre[0], spot[1] - window_centre[1]))
        rows.append(_row(
            "模型采样窗中心 vs 实测零点", f"窗 ({window_centre[0]:.0f}, {window_centre[1]:.0f}) / "
            f"零点 ({spot[0]:.0f}, {spot[1]:.0f})，偏 {offset:.0f} px",
            "窗心 = 本次实测零点", "stale" if offset > 5 else "reuse",
            "可离线重建：recentre_window(spot, half_span=..., step=2) 后作为 camera_xy"))
    else:
        rows.append(_row("模型采样窗中心", "无实测帧可对照", "—", "unknown", "needs-device：抓一帧定零点"))

    # ---- 4. 目标尺寸标定 ------------------------------------------------------
    if CALIBRATION.exists():
        calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        rows.append(_row(
            "目标尺寸标定", f"signature={calibration.get('signature', '?')[:16]}，"
            f"best={calibration.get('best_size', float('nan')):.1f} px",
            "签名不变（λ/f/N/p 未变）但**有效窗口**依赖旧离焦",
            "stale", "needs-device：重拟合后重扫 40–240 px（有效性门 + HDR 度量）"))
    else:
        rows.append(_row("目标尺寸标定", "缺失", "—", "missing", "needs-device：先跑扫描"))

    # ---- 汇总 ----------------------------------------------------------------
    stale = [r for r in rows if r["verdict"] in {"stale", "missing"}]
    logger.info("光路变化后的自动校准自检")
    logger.info("=" * 100)
    width = max(len(r["item"]) for r in rows)
    for item in rows:
        logger.info(f"  {item['item']:<{width}} | {item['verdict']:<7} | 存量: {item['stored']}")
        logger.info(f"  {'':<{width}} | {'':<7} | 期望: {item['expected_after_change']}")
        logger.info(f"  {'':<{width}} | {'':<7} | 处置: {item['action']}")
    logger.info("-" * 100)
    logger.info(f"  需要重做: {len(stale)} / {len(rows)} 项；其中可离线重建的只有采样窗，"
          f"其余需设备（焦点扫描、模型重拟合、尺寸重扫）。")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dict(run=str(RUN), items=rows, stale=len(stale)),
                                        indent=2, ensure_ascii=False), encoding="utf-8")
    if args.check and stale:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
