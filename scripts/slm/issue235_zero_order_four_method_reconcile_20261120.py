"""Issue #235 P1: offline four-method zero-order centre reconciliation.

Why this exists
---------------
``docs/dev/hw_plan_consolidated_20261118.md`` §3 P1 asks for an offline
reconciliation of four zero-order-centre estimators on the archived ``flat.npy``
(``data/debug/issue235_center_20261117/``): ``argmax`` / ``centroid`` /
``box_centroid`` / ``ellipse``. If every pair agrees within 3 px, the default
ellipse centre is deemed usable. The CCD-side and panel-side knife centres are
reported separately and must never be subtracted from each other (different
coordinate frames: CCD 2592x1944 vs panel 1920x1200).

This script is **offline only**: it reads saved artefacts and never touches a
device. It reuses the production estimator
:func:`ao_shaping.utils.image.zero_order.zero_order_center_by_method` (the same
function exercised by ``tests/ao_shaping/utils/image/test_zero_order.py``), so
the four "centres" come from one authoritative implementation rather than four
ad-hoc re-implementations.

Outputs a single JSON next to ``--out`` with the four centres, all six pairwise
Euclidean distances (px), the <=3 px verdict, and the two knife-edge centres.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from ao_shaping.utils.image.zero_order import (
    DEFAULT_ZERO_ORDER_METHOD,
    ZERO_ORDER_METHODS,
    zero_order_center_by_method,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLAT = ROOT / "data" / "debug" / "issue235_center_20261117" / "flat.npy"
DEFAULT_SUMMARY = ROOT / "data" / "debug" / "issue235_center_20261117" / "summary.json"
DEFAULT_KNIFE = (
    ROOT
    / "report"
    / "shape_trials"
    / "model_direction_hw_20261117"
    / "issue235_knife_baseline_fit_20261117.json"
)
PAIR_TOLERANCE_PX = 3.0


def reconcile(flat: np.ndarray) -> dict:
    """Run the four production estimators and return centres + pairwise diffs."""
    # Production API returns (x, y); project records centres as (cy, cx).
    centres_cy_cx: dict[str, list[int]] = {}
    centres_xy: dict[str, list[int]] = {}
    for key, _label in ZERO_ORDER_METHODS:
        x, y = zero_order_center_by_method(flat, method=key)
        centres_xy[key] = [int(x), int(y)]
        centres_cy_cx[key] = [int(y), int(x)]

    pairs: list[dict] = []
    for a, b in itertools.combinations([k for k, _ in ZERO_ORDER_METHODS], 2):
        ya, xa = centres_cy_cx[a]
        yb, xb = centres_cy_cx[b]
        d = float(np.hypot(ya - yb, xa - xb))
        pairs.append(
            {
                "a": a,
                "b": b,
                "dy_px": ya - yb,
                "dx_px": xa - xb,
                "distance_px": round(d, 6),
                "within_tol": bool(d <= PAIR_TOLERANCE_PX + 1e-9),
            }
        )
    max_d = max(p["distance_px"] for p in pairs)
    all_within = all(p["within_tol"] for p in pairs)
    return {
        "centres_cy_cx": centres_cy_cx,
        "centres_xy": centres_xy,
        "pairwise": pairs,
        "max_pairwise_distance_px": round(max_d, 6),
        "tolerance_px": PAIR_TOLERANCE_PX,
        "all_pairs_within_tolerance": bool(all_within),
        "default_method": DEFAULT_ZERO_ORDER_METHOD,
        "default_centre_xy": centres_xy[DEFAULT_ZERO_ORDER_METHOD],
        "verdict": (
            "ellipse_centre_usable" if all_within else "ellipse_centre_not_confirmed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat", type=Path, default=DEFAULT_FLAT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--knife", type=Path, default=DEFAULT_KNIFE)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path (parent directories are created).",
    )
    args = parser.parse_args()

    flat = np.load(args.flat)
    result = reconcile(flat)

    # Cross-check the two centres already present in the archived summary: they
    # anchor our production-function reading to the number the capture script wrote.
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result["summary_crosscheck"] = {
        "archived_argmax_cy_cx": summary.get("zero_order_cy_cx"),
        "archived_box_centroid_cy_cx": summary.get("zero_order_box_centroid_cy_cx"),
        "recomputed_argmax_cy_cx": result["centres_cy_cx"]["argmax"],
        "recomputed_box_centroid_cy_cx": result["centres_cy_cx"]["box_centroid"],
        "note": (
            "archived box_centroid used a 0.15-threshold fixed-window local helper "
            "(float); production zero_order.py uses the half-max connected "
            "component then rounds to int. Compare after rounding."
        ),
    }

    # CCD-side vs panel-side knife centres: different frames, report separately.
    knife = json.loads(args.knife.read_text(encoding="utf-8"))
    result["knife_centres_separate_frames"] = {
        "slm_panel_find_beam_yx": knife.get("find_beam_on_slm_centre_yx"),
        "slm_panel_fit_yx": knife.get("fit_centre_yx"),
        "frame_note": (
            "Both are SLM-panel coordinates from the differential knife-edge fit "
            "(1920x1200 panel frame); they are NOT in the CCD frame of flat.npy "
            "(2592x1944) and must not be differenced against the zero-order spot."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
