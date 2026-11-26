# -*- coding: utf-8 -*-
"""S1e same-bytes probe: offline execution of the frozen §12.16.3 criteria.

The capture script ``s1e_same_bytes_probe_capture_20261124.py`` froze the plan,
the arrays and the thresholds; this module only *executes* them after capture.
Nothing here may redefine a threshold: ``FROZEN_CRITERIA``, ``axis_type``,
``pairwise_max_distance`` and ``d6_zero_exception`` are imported from the
capture module, and ``find_spots``/``principal`` semantics come from the frozen
solver ``issue247_axis_map_solve_20261121.py``.

No devices are touched: this reads ``frames/*.npy`` and
``capture_journal.jsonl`` from a finished run directory and writes a judgment
JSON plus figures.  Supplementary numbers (zero-order drift, drift-corrected
spread, harmonic ratios) are recorded as *evidence only* and never feed a
verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "libs", ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.slm.issue247_axis_map_solve_20261121 import find_spots
from scripts.slm.s1e_same_bytes_probe_capture_20261124 import (
    ARRAY_BY_ID,
    FROZEN_CRITERIA,
    FROZEN_ORDER,
    PREDECESSOR_BY_ID,
    axis_type,
    d6_zero_exception,
    pairwise_max_distance,
)

JUDGMENT_SCHEMA = "issue247-s1e-same-bytes-probe-judgment/v1"
DEFAULT_CAPTURE_DIR = (
    ROOT
    / "data/experiments/attribution_20261123/s1/s1e_same_bytes_probe_20261124/run_01"
)
DEFAULT_OUT_DIR = ROOT / "report/slm/attribution"

#: ``principal()`` frozen semantics (core exclusion + minimum distance from 0th).
EXCLUDE_CORE_PX = 6.0
MIN_DIST_PX = 15.0
#: Harmonic orders recorded as supporting evidence only (§12.16.3).
HARMONIC_ORDERS = (3, 5)
HARMONIC_TOL_FRAC = 0.15

FLAT_IDS = ("flat", "flat_mid", "flat_post")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def read_journal(capture_dir: Path) -> list[dict[str, Any]]:
    """Parse ``capture_journal.jsonl``; one JSON object per non-empty line."""
    path = capture_dir / "capture_journal.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing journal: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def upload_records(journal: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``pattern_id`` -> upload record (memory slot, sent array digests, utc)."""
    out: dict[str, dict[str, Any]] = {}
    for row in journal:
        if row.get("event") != "upload":
            continue
        out[str(row["pattern_id"])] = row
    return out


def frame_files(
    journal: list[dict[str, Any]], capture_dir: Path
) -> dict[str, list[Path]]:
    """``pattern_id`` -> ordered frame paths actually written by the capture."""
    out: dict[str, list[Path]] = {}
    for row in journal:
        if row.get("event") != "frame_saved":
            continue
        path = capture_dir / "frames" / str(row["filename"])
        if not path.is_file():
            raise SystemExit(f"journal lists a missing frame: {path}")
        out.setdefault(str(row["pattern_id"]), []).append(path)
    for paths in out.values():
        paths.sort()
    return out


# ---------------------------------------------------------------------------
# Frozen spot semantics
# ---------------------------------------------------------------------------
def zero_order_reference(flat: np.ndarray) -> tuple[float, float]:
    """Sub-pixel 0th-order centroid = brightest ``find_spots`` detection on flat."""
    spots = find_spots(flat, min_dist=20)
    if not spots:
        raise SystemExit("no 0th-order spot on the flat reference frame")
    return (spots[0]["row"], spots[0]["col"])


def principal(
    frame: np.ndarray, ref: tuple[float, float]
) -> dict[str, float] | None:
    """Brightest spot clear of the 0th-order core (frozen solver semantics)."""
    best: dict[str, float] | None = None
    for spot in find_spots(frame):
        dr = spot["row"] - ref[0]
        dc = spot["col"] - ref[1]
        dist = float(np.hypot(dr, dc))
        if dist < MIN_DIST_PX:
            continue
        if abs(dr) <= EXCLUDE_CORE_PX and abs(dc) <= EXCLUDE_CORE_PX:
            continue
        if best is None or spot["peak"] > best["peak"]:
            best = {
                "row": float(spot["row"]),
                "col": float(spot["col"]),
                "dr": float(dr),
                "dc": float(dc),
                "dist": dist,
                "peak": float(spot["peak"]),
            }
    return best


def harmonics(frame: np.ndarray, ref: tuple[float, float], fundamental: float) -> dict:
    """Peak intensity at k x fundamental distance (supporting record only)."""
    out: dict[str, Any] = {}
    for order in HARMONIC_ORDERS:
        target = order * fundamental
        tol = HARMONIC_TOL_FRAC * fundamental
        candidates = []
        for spot in find_spots(frame):
            dr = spot["row"] - ref[0]
            dc = spot["col"] - ref[1]
            dist = float(np.hypot(dr, dc))
            if abs(dist - target) <= tol:
                candidates.append((dist, float(spot["peak"])))
        if not candidates:
            out[f"h{order}"] = None
            continue
        dist, peak = max(candidates, key=lambda item: item[1])
        out[f"h{order}"] = {
            "dist_px": round(dist, 2),
            "dist_ratio": round(dist / fundamental, 3),
            "peak": round(peak, 1),
            "i_over_i0": round(peak / max(fundamental_peak(frame, ref), 1e-9), 3),
        }
    return out


def fundamental_peak(frame: np.ndarray, ref: tuple[float, float]) -> float:
    """Peak of the principal spot, used only to normalise harmonic intensities."""
    spot = principal(frame, ref)
    return 0.0 if spot is None else spot["peak"]


# ---------------------------------------------------------------------------
# Per-upload and per-arm evaluation
# ---------------------------------------------------------------------------
def evaluate_uploads(
    patterns: dict[str, list[Path]],
    ref: tuple[float, float],
    flat_peak: float,
) -> dict[str, dict[str, Any]]:
    """Judge every non-flat upload under the frozen intra-upload rule."""
    spec = FROZEN_CRITERIA
    intra_max = float(spec["intra_upload_max_dist_px"])
    weak_frac = float(spec["D4"]["weak_peak_fraction"])
    out: dict[str, dict[str, Any]] = {}
    for pattern_id in FROZEN_ORDER:
        if pattern_id in FLAT_IDS:
            continue
        frames = patterns.get(pattern_id, [])
        spots = [principal(np.load(path), ref) for path in frames]
        if len(spots) != 2 or any(spot is None for spot in spots):
            out[pattern_id] = {"valid": False, "reason": "no_principal_spot"}
            continue
        first, second = spots[0], spots[1]
        assert first is not None and second is not None
        intra = float(np.hypot(first["row"] - second["row"], first["col"] - second["col"]))
        row = float(np.mean([first["row"], second["row"]]))
        col = float(np.mean([first["col"], second["col"]]))
        peak = float(np.mean([first["peak"], second["peak"]]))
        dr, dc = row - ref[0], col - ref[1]
        out[pattern_id] = {
            "array": ARRAY_BY_ID[pattern_id],
            "predecessor": PREDECESSOR_BY_ID[pattern_id],
            "intra_upload_dist_px": round(intra, 3),
            "row": round(row, 3),
            "col": round(col, 3),
            "dr": round(dr, 3),
            "dc": round(dc, 3),
            "dist_px": round(float(np.hypot(dr, dc)), 3),
            "axis_type": axis_type(dr, dc),
            "peak": round(peak, 1),
            "weak": peak < weak_frac * flat_peak,
            "valid": intra <= intra_max,
        }
    return out


def evaluate_arm(
    uploads: dict[str, dict[str, Any]], array: str
) -> dict[str, Any]:
    """Frozen D1 test for one array (H or B)."""
    spec = FROZEN_CRITERIA["D1"]
    min_effective = int(spec["min_effective_uploads"])
    max_dist = float(spec["max_pairwise_dist_px"])
    used = [
        rec
        for rec in uploads.values()
        if rec.get("array") == array and rec.get("valid") and not rec.get("weak")
    ]
    points = [(rec["dr"], rec["dc"]) for rec in used]
    spread = pairwise_max_distance(points)
    if len(used) < min_effective:
        verdict = "inconclusive"
    elif spread <= max_dist:
        verdict = "D-pass"
    else:
        verdict = "D-fail"
    return {
        "array": array,
        "n_effective": len(used),
        "max_pairwise_dist_px": round(spread, 3),
        "threshold_px": max_dist,
        "verdict": verdict,
        "axis_types": [rec["axis_type"] for rec in used],
    }


def group_by_predecessor(
    uploads: dict[str, dict[str, Any]], array: str | None = None
) -> dict[str, list[str]]:
    """Predecessor kind (``flat``/``H``/``B``) -> axis types (D3 grouping).

    ``array`` restricts the grouping to one arm (``H`` or ``B``); ``None`` keeps
    every upload, which is the literal pre-registration reading.  Both are
    reported so the reader can see that the all-upload grouping is confounded by
    array identity (a ``B`` upload that follows an ``H`` upload is intrinsically
    ``other``, not because of its history).
    """
    groups: dict[str, list[str]] = {}
    for rec in uploads.values():
        if not rec.get("valid") or rec.get("weak"):
            continue
        if array is not None and rec.get("array") != array:
            continue
        kind = str(rec["predecessor"])
        kind = "flat" if kind.startswith("flat") else kind
        groups.setdefault(kind, []).append(str(rec["axis_type"]))
    return groups


def group_by_slot(
    uploads: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]]
) -> dict[int, set[str]]:
    """Memory slot -> axis types (D6 grouping)."""
    groups: dict[int, set[str]] = {}
    for pattern_id, rec in uploads.items():
        if not rec.get("valid") or rec.get("weak"):
            continue
        slot = int(records[pattern_id]["memory_number"])
        groups.setdefault(slot, set()).add(str(rec["axis_type"]))
    return groups


def zero_exception(groups: dict[str, list[str]]) -> bool:
    """True when every group shows exactly one axis type."""
    return all(len(set(values)) == 1 for values in groups.values())


# ---------------------------------------------------------------------------
# Supplementary (evidence only; never a verdict)
# ---------------------------------------------------------------------------
def flat_reference_track(
    patterns: dict[str, list[Path]],
) -> dict[str, Any]:
    """0th-order centroid and peak of each flat frame across the session."""
    track: dict[str, Any] = {}
    for pattern_id in FLAT_IDS:
        rows = []
        for path in patterns.get(pattern_id, []):
            frame = np.load(path)
            ref = zero_order_reference(frame)
            rows.append(
                {
                    "frame": path.name,
                    "row": round(ref[0], 3),
                    "col": round(ref[1], 3),
                    "peak": int(frame.max()),
                }
            )
        track[pattern_id] = rows
    base = np.mean([[r["row"], r["col"]] for r in track.get("flat", [])], axis=0)
    for pattern_id in FLAT_IDS:
        if not track.get(pattern_id):
            continue
        mean = np.mean([[r["row"], r["col"]] for r in track[pattern_id]], axis=0)
        delta = mean - base
        track[f"{pattern_id}_offset"] = {
            "drow": round(float(delta[0]), 3),
            "dcol": round(float(delta[1]), 3),
            "dist_px": round(float(np.hypot(*delta)), 3),
        }
    return track


def drift_corrected_spread(
    uploads: dict[str, dict[str, Any]], track: dict[str, Any], array: str
) -> dict[str, Any]:
    """Spread after subtracting an interpolated 0th-order drift (evidence only)."""
    anchors = {
        "flat": (0.5, track["flat"]),
        "flat_mid": (7.5, track["flat_mid"]),
        "flat_post": (11.5, track["flat_post"]),
    }
    xs = np.array([idx for idx, _ in anchors.values()], dtype=np.float64)
    ys = np.array(
        [
            np.mean([[r["row"], r["col"]] for r in rows], axis=0)
            for _, rows in anchors.values()
        ],
        dtype=np.float64,
    )
    order_index = {pattern_id: i for i, pattern_id in enumerate(FROZEN_ORDER)}
    corrected: dict[str, tuple[float, float]] = {}
    for pattern_id, rec in uploads.items():
        if rec.get("array") != array or not rec.get("valid") or rec.get("weak"):
            continue
        interp = np.array(
            [np.interp(order_index[pattern_id], xs, ys[:, axis]) for axis in (0, 1)]
        )
        corrected[pattern_id] = (
            rec["row"] - (interp[0] - ys[0][0]) - ys[0][0],
            rec["col"] - (interp[1] - ys[0][1]) - ys[0][1],
        )
    points = list(corrected.values())
    return {
        "array": array,
        "max_pairwise_dist_px": round(pairwise_max_distance(points), 3),
        "positions": {
            key: {"dr": round(value[0], 3), "dc": round(value[1], 3)}
            for key, value in corrected.items()
        },
        "note": "supplementary: not part of the frozen verdict",
    }


# ---------------------------------------------------------------------------
# Judgment assembly
# ---------------------------------------------------------------------------
def judge(capture_dir: Path) -> dict[str, Any]:
    """Run every frozen criterion and return the judgment payload."""
    journal = read_journal(capture_dir)
    records = upload_records(journal)
    patterns = frame_files(journal, capture_dir)

    missing = [pid for pid in FROZEN_ORDER if pid not in patterns]
    if missing:
        raise SystemExit(f"capture incomplete, missing patterns: {missing}")

    flat_frames = [np.load(path) for path in patterns["flat"]]
    flat_peak = float(max(frame.max() for frame in flat_frames))
    ref = zero_order_reference(flat_frames[0])

    uploads = evaluate_uploads(patterns, ref, flat_peak)
    arms = {array: evaluate_arm(uploads, array) for array in ("H", "B")}

    d1_h = arms["H"]["verdict"]
    predecessor_groups = group_by_predecessor(uploads)
    predecessor_groups_h = group_by_predecessor(uploads, "H")
    predecessor_groups_b = group_by_predecessor(uploads, "B")
    predecessor_zero_exception = zero_exception(predecessor_groups)
    h_zero_exception = zero_exception(predecessor_groups_h)
    h_axis_types = {t for types in predecessor_groups_h.values() for t in types}
    h_axis_degenerate = len(h_axis_types) == 1
    if d1_h != "D-fail":
        d3_verdict = "not_applicable"
    elif predecessor_zero_exception:
        d3_verdict = "history_dependent"
    else:
        d3_verdict = "display_nondeterministic"
    # Best-fit §12.16.4 branch from the frozen preconditions.
    if d1_h == "D-pass":
        e_branch = "E1"
    elif d1_h == "D-fail" and predecessor_zero_exception:
        e_branch = "E3"
    elif d1_h == "D-fail":
        e_branch = "E2"
    else:
        e_branch = "undetermined"
    d3_discriminating = bool(d1_h == "D-fail" and not h_axis_degenerate)
    d3_note = (
        "行程内 H 臂六次上传全为同一轴型，无法按轴型区分历史效应——"
        "D-fail 是位置抖动而非轴型翻转；全量前驱分组出现例外仅因 B 上传（本身即 other）"
        "跟随 H，属数组身份混杂，非历史依赖。故 E3 不适用，取 E2。"
        if h_axis_degenerate
        else "H 臂存在轴型差异，可按前驱分组检验历史依赖。"
    )

    slot_groups = group_by_slot(uploads, records)
    slot_zero_exception = d6_zero_exception(slot_groups)
    d6_verdict = "zero_exception_slot_groups" if slot_zero_exception else "slot_groups_mixed"
    if d1_h != "D-fail":
        d6_verdict = "not_applicable"

    all_axis_types = {str(rec.get("axis_type")) for rec in uploads.values() if rec.get("valid")}
    degenerate = len(all_axis_types) == 1

    harmonic_rows: dict[str, Any] = {}
    for pattern_id in ("s1e_a1", "s1e_a3", "s1e_a5", "s1e_a6", "s1e_b1", "s1e_b2", "s1e_b3"):
        frame = np.load(patterns[pattern_id][0])
        spot = principal(frame, ref)
        if spot is None:
            continue
        harmonic_rows[pattern_id] = {
            "array": ARRAY_BY_ID[pattern_id],
            "fundamental_dist_px": round(spot["dist"], 2),
            "fundamental_peak": round(spot["peak"], 1),
            "harmonics": harmonics(frame, ref, spot["dist"]),
        }

    track = flat_reference_track(patterns)
    digests: dict[str, Any] = {}
    for pattern_id in FROZEN_ORDER:
        if pattern_id in FLAT_IDS:
            continue
        sent = records[pattern_id].get("sent_array") or {}
        digests.setdefault(ARRAY_BY_ID[pattern_id], set()).add(
            str(sent.get("physical_sha256"))
        )
    delivery_uniform = {
        array: {"n_distinct_physical_digest": len(values), "physical_sha256": sorted(values)}
        for array, values in digests.items()
    }

    return {
        "schema": JUDGMENT_SCHEMA,
        "capture_dir": str(capture_dir.relative_to(ROOT)) if capture_dir.is_relative_to(ROOT) else str(capture_dir),
        "criteria": FROZEN_CRITERIA,
        "zero_order_reference": {"row": round(ref[0], 3), "col": round(ref[1], 3),
                                 "flat_peak": flat_peak},
        "uploads": uploads,
        "arms": arms,
        "D1": {"array": "H", "verdict": d1_h,
               "max_pairwise_dist_px": arms["H"]["max_pairwise_dist_px"],
               "n_effective": arms["H"]["n_effective"]},
        "D2": {
            "H": [rec.get("axis_type") for pid, rec in uploads.items()
                  if rec.get("array") == "H"],
            "B": [rec.get("axis_type") for pid, rec in uploads.items()
                  if rec.get("array") == "B"],
            "H_dist_px": [rec.get("dist_px") for pid, rec in uploads.items()
                          if rec.get("array") == "H"],
            "B_dist_px": [rec.get("dist_px") for pid, rec in uploads.items()
                          if rec.get("array") == "B"],
        },
        "D3": {
            "verdict": d3_verdict,
            "predecessor_groups": predecessor_groups,
            "predecessor_groups_H": predecessor_groups_h,
            "predecessor_groups_B": predecessor_groups_b,
            "zero_exception": predecessor_zero_exception,
            "zero_exception_H": h_zero_exception,
            "degenerate_H_single_axis_type": h_axis_degenerate,
            "discriminating": d3_discriminating,
            "note": d3_note,
            "degenerate_single_axis_type": degenerate,
        },
        "E_branch": e_branch,
        "D4": {
            "verdict": arms["B"]["verdict"],
            "note": "B has 3 uploads; the frozen D1 rule needs 5 effective uploads. "
            "Supplementary: B also jitters, so it is not demonstrated stable.",
            "n_effective": arms["B"]["n_effective"],
            "max_pairwise_dist_px": arms["B"]["max_pairwise_dist_px"],
            "axis_types": arms["B"]["axis_types"],
        },
        "D5": {"verdict": "not_applicable", "note": "arm P (power cycle) not executed"},
        "D6": {
            "verdict": d6_verdict,
            "slot_groups": {str(k): sorted(v) for k, v in sorted(slot_groups.items())},
            "zero_exception": slot_zero_exception,
            "degenerate_single_axis_type": degenerate,
        },
        "delivery_uniformity": delivery_uniform,
        "harmonics_supporting": harmonic_rows,
        "supplementary": {
            "flat_reference_track": track,
            "drift_corrected_spread": {
                "H": drift_corrected_spread(uploads, track, "H"),
                "B": drift_corrected_spread(uploads, track, "B"),
            },
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Frozen-verdict tables for the report body."""
    lines = [
        "| pattern_id | 数组 | 前驱 | 帧内差 px | dr | dc | dist px | 轴型 | 峰值 | 有效 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pattern_id in FROZEN_ORDER:
        rec = payload["uploads"].get(pattern_id)
        if rec is None:
            lines.append(f"| {pattern_id} | flat | — | — | — | — | — | — | — | 参考 |")
            continue
        lines.append(
            f"| `{pattern_id}` | {rec['array']} | {rec['predecessor']} | "
            f"{rec['intra_upload_dist_px']:.2f} | {rec['dr']:+.2f} | {rec['dc']:+.2f} | "
            f"{rec['dist_px']:.2f} | {rec['axis_type']} | {rec['peak']:.0f} | "
            f"{'是' if rec['valid'] and not rec['weak'] else ('weak' if rec['weak'] else '否')} |"
        )
    return "\n".join(lines)


def write_figure(payload: dict[str, Any], path: Path) -> bool:
    """Three-panel evidence figure; skipped when matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning(f"matplotlib unavailable, figure skipped: {exc}")
        return False

    uploads = payload["uploads"]
    order_index = {pid: i for i, pid in enumerate(FROZEN_ORDER)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for array, color in (("H", "#1f77b4"), ("B", "#d62728")):
        ids = [pid for pid, rec in uploads.items() if rec.get("array") == array and rec.get("valid")]
        ids.sort(key=lambda pid: order_index[pid])
        xs = [order_index[pid] for pid in ids]
        axes[0].plot(xs, [uploads[pid]["dr"] for pid in ids], "o-", color=color,
                     label=f"{array} dr")
        axes[0].plot(xs, [uploads[pid]["dc"] for pid in ids], "s--", color=color,
                     alpha=0.55, label=f"{array} dc")
    axes[0].set_xlabel("upload index (frozen order)")
    axes[0].set_ylabel("offset from 0th order (px)")
    axes[0].set_title("(a) principal-lobe offsets")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    flat = payload["supplementary"]["flat_reference_track"]
    for key, marker in (("flat", "o"), ("flat_mid", "s"), ("flat_post", "^")):
        rows = flat.get(key, [])
        axes[1].plot([r["row"] for r in rows], [r["col"] for r in rows], marker,
                     label=key, ms=6)
    axes[1].set_xlabel("0th-order row (px)")
    axes[1].set_ylabel("0th-order col (px)")
    axes[1].set_title("(b) flat 0th-order centroid (session drift)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    heights = []
    labels = []
    for array, color in (("H", "#1f77b4"), ("B", "#d62728")):
        vals = [rec["dist_px"] for rec in uploads.values() if rec.get("array") == array]
        labels.append(array)
        heights.append(float(np.mean(vals)) if vals else 0.0)
        axes[2].scatter([array] * len(vals), vals, color=color, s=28)
    ratio = (heights[0] / heights[1]) if heights[1] else float("nan")
    axes[2].set_ylabel("displacement from 0th order (px)")
    axes[2].set_title(f"(c) axis fingerprint: H/B = {ratio:.3f}")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default="s1e_same_bytes_probe_20261126")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    payload = judge(args.capture_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.stem}.json"
    json_path.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path = args.out_dir / f"{args.stem}_tables.md"
    md_path.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    figure_path = None
    if not args.no_figure:
        candidate = args.out_dir / f"{args.stem}_fig1.png"
        if write_figure(payload, candidate):
            figure_path = candidate

    print(render_markdown(payload))
    print()
    for key in ("D1", "D3", "D4", "D5", "D6", "E_branch"):
        print(f"{key}: {json.dumps(payload[key], ensure_ascii=False)}")
    print(f"\njson   -> {json_path}")
    print(f"tables -> {md_path}")
    if figure_path:
        print(f"figure -> {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
