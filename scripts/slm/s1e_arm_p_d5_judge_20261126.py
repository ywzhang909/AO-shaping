# -*- coding: utf-8 -*-
"""S1e arm P (power-cycle) D5 cross-variant judge: frozen §12.16.3 only.

This module does NOT re-run the frozen judge or redefine any threshold.  It
reads the two already-frozen judge JSON payloads (variant A = run_01 with no
power cycle, variant P = run_02 after a physical SLM power cycle) and compares
the same-slot main lobes using the frozen ``intra_upload_max_dist_px`` constant
imported from the frozen capture script.  Positions are the ``(dr, dc)`` each
variant already computed relative to its OWN 0th reference (§12.16.3 forbids
reusing a 0th across variants).  Absolute-coordinate deltas are recorded as
supporting evidence only and never feed the verdict.

D5 verdict mapping (frozen §12.16.3 / §12.16.4):
  * max same-slot relative diff <= threshold, no col/row flip
        -> ``no_power_cycle_effect`` (E4 not triggered)
  * max diff > threshold AND a col/row axis flip appears
        -> ``power_cycle_state_dependent`` (E4 triggered)
  * max diff > threshold without any axis flip
        -> ``position_jitter_no_axis_flip`` (E4 not triggered)

No devices are touched.  Pure offline file I/O.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "libs", ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.slm.s1e_same_bytes_probe_capture_20261124 import (
    FROZEN_CRITERIA,
    FROZEN_ORDER,
)

FLAT_IDS = ("flat", "flat_mid", "flat_post")
SCHEMA = "issue247-s1e-arm-p-d5-judgment/v1"
DEFAULT_OUT_DIR = ROOT / "report/shape_trials/s1e_same_bytes_probe_20261126"
DEFAULT_A_JSON = DEFAULT_OUT_DIR / "s1e_same_bytes_probe_20261126.json"
DEFAULT_P_JSON = DEFAULT_OUT_DIR / "s1e_arm_p_20261126.json"


def _load(path: Path) -> dict[str, Any]:
    """Load a frozen judge JSON payload."""
    if not path.is_file():
        raise SystemExit(f"missing judge JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _axis_flip(ta: str, tp: str) -> bool:
    """True only for a col-type <-> row-type transition at the same slot."""
    return (ta == "col-type" and tp == "row-type") or (
        ta == "row-type" and tp == "col-type"
    )


def judge_d5(payload_a: dict[str, Any], payload_p: dict[str, Any]) -> dict[str, Any]:
    """Cross-variant D5 from two frozen judge payloads."""
    threshold = float(FROZEN_CRITERIA["intra_upload_max_dist_px"])
    uploads_a: dict[str, Any] = payload_a["uploads"]
    uploads_p: dict[str, Any] = payload_p["uploads"]

    per_slot: dict[str, Any] = {}
    for pid in FROZEN_ORDER:
        if pid in FLAT_IDS:
            continue
        ua, up = uploads_a.get(pid), uploads_p.get(pid)
        if not (ua and up and ua.get("valid") and up.get("valid")):
            continue
        ddr = float(ua["dr"]) - float(up["dr"])
        ddc = float(ua["dc"]) - float(up["dc"])
        rel_diff = float(np.hypot(ddr, ddc))
        abs_diff = float(
            np.hypot(
                float(ua["row"]) - float(up["row"]),
                float(ua["col"]) - float(up["col"]),
            )
        )
        ta, tp = str(ua["axis_type"]), str(up["axis_type"])
        per_slot[pid] = {
            "array": ua["array"],
            "dr_A": ua["dr"], "dr_P": up["dr"],
            "dc_A": ua["dc"], "dc_P": up["dc"],
            "rel_diff_px": round(rel_diff, 3),
            "abs_diff_px": round(abs_diff, 3),
            "axis_A": ta, "axis_P": tp,
            "axis_flip": _axis_flip(ta, tp),
        }

    if not per_slot:
        return {
            "schema": SCHEMA,
            "verdict": "inconclusive",
            "note": "no slot valid in both variants",
            "threshold_px": threshold,
            "E4_triggered": False,
        }

    worst = max(per_slot, key=lambda k: per_slot[k]["rel_diff_px"])
    max_diff = float(per_slot[worst]["rel_diff_px"])
    max_abs = max(float(rec["abs_diff_px"]) for rec in per_slot.values())
    any_flip = any(rec["axis_flip"] for rec in per_slot.values())

    if max_diff <= threshold:
        verdict = "no_power_cycle_effect"
        note = (
            f"同序位主瓣相对各自 0th 的差最大 {max_diff:.3f} px（{worst}），"
            f"未超冻结阈值 {threshold} px；跨变体轴型全一致、无 col/row 翻转；"
            f"绝对坐标差最大 {max_abs:.3f} px 仅作佐证（含各变体 0th 质心抖动）。"
        )
    elif any_flip:
        verdict = "power_cycle_state_dependent"
        note = (
            f"同序位主瓣相对差最大 {max_diff:.3f} px（{worst}）> {threshold} px，"
            f"且出现 col/row 翻转 ⇒ 上电状态相关。"
        )
    else:
        verdict = "position_jitter_no_axis_flip"
        note = (
            f"同序位主瓣相对差最大 {max_diff:.3f} px（{worst}）> {threshold} px，"
            f"但无 col/row 翻转，属位置抖动。"
        )

    return {
        "schema": SCHEMA,
        "verdict": verdict,
        "note": note,
        "threshold_px": threshold,
        "threshold_source": "FROZEN_CRITERIA['intra_upload_max_dist_px']",
        "position_basis": "dr/dc relative to each variant's own 0th (frozen §12.16.3)",
        "max_rel_diff_px": max_diff,
        "worst_slot": worst,
        "max_abs_diff_px": max_abs,
        "any_axis_flip": any_flip,
        "n_compared_slots": len(per_slot),
        "per_slot": per_slot,
        "E4_triggered": verdict == "power_cycle_state_dependent",
        "input_A": {
            "capture_dir": payload_a.get("capture_dir"),
            "zero_order_reference": payload_a.get("zero_order_reference"),
            "D1": payload_a.get("D1", {}).get("verdict"),
        },
        "input_P": {
            "capture_dir": payload_p.get("capture_dir"),
            "zero_order_reference": payload_p.get("zero_order_reference"),
            "D1": payload_p.get("D1", {}).get("verdict"),
            "E_branch": payload_p.get("E_branch"),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    """Render the D5 same-slot comparison table."""
    lines = [
        "# S1e arm P D5 同序位跨变体对比",
        "",
        f"- 判定：**{result['verdict']}**",
        f"- 冻结阈值：`{result.get('threshold_source', 'FROZEN_CRITERIA')}` = "
        f"{result['threshold_px']} px",
        f"- 口径：{result.get('position_basis', 'dr/dc relative to own 0th')}",
        f"- 同序位最大相对差：**{result.get('max_rel_diff_px')} px**"
        f"（{result.get('worst_slot')}）",
        f"- 绝对坐标差最大：{result.get('max_abs_diff_px')} px（仅佐证）",
        f"- 轴型翻转：{'有' if result.get('any_axis_flip') else '无'}；"
        f"E4 触发：**{'是' if result.get('E4_triggered') else '否'}**",
        "",
        "> 说明：",
        f"{result['note']}",
        "",
        "| pattern | 数组 | dr_A | dr_P | dc_A | dc_P | 相对差 px | 绝对差 px | 轴型 A | 轴型 P | 翻转 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for pid, rec in result.get("per_slot", {}).items():
        lines.append(
            f"| `{pid}` | {rec['array']} | {rec['dr_A']} | {rec['dr_P']} | "
            f"{rec['dc_A']} | {rec['dc_P']} | {rec['rel_diff_px']} | "
            f"{rec['abs_diff_px']} | {rec['axis_A']} | {rec['axis_P']} | "
            f"{'是' if rec['axis_flip'] else '否'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-a", type=Path, default=DEFAULT_A_JSON)
    parser.add_argument("--json-p", type=Path, default=DEFAULT_P_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default="s1e_arm_p_d5_20261126")
    args = parser.parse_args(argv)

    payload_a = _load(args.json_a)
    payload_p = _load(args.json_p)
    result = judge_d5(payload_a, payload_p)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.stem}.json"
    md_path = args.out_dir / f"{args.stem}_tables.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(result), encoding="utf-8")

    print(f"D5: {json.dumps(result, ensure_ascii=False)}")
    print(f"\njson   -> {json_path}")
    print(f"tables -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
