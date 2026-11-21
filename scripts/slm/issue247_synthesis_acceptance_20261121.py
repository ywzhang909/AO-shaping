"""Issue #247 / A0-3: offline synthesis acceptance for registration patterns.

为什么需要它
------------
#247 的原始帧给出了互相矛盾的轴响应：``grating_x`` / ``tilt_x`` 的位移落在
CCD-x，而 ``ladder_x`` 的 +1/+3 级却落在 CCD-y。这个矛盾无法靠"再挑一个仿射
矩阵"解决——必须先用**离线 FFT** 把每个候选图样的远场 spot 预测算出来，确认
它们在 180° 对称族（Klein 四群：恒等 / 整体反号 / 轴交换 / 反号+轴交换）下
**互相可区分**，才把对应的采集图样加入 A1。

做法：对每个候选图样先**解析预测**有符号 spot（面板周期 -> 远场 bin 偏移），
再用 FFT 在预测位置附近验证确实存在局部极大。这样"预测"与"验证"分开，弱级
（ladder 的 +1/+3 只占零级约 1%）不会被全局强度阈值整个吃掉。

本脚本零设备：只构造相位、做 FFT、验证 spot、判可区分性，输出带哈希的证据
JSON。不写设备、不建采集目录、不导入相机/SLM 驱动。

用法::

    python scripts/slm/issue247_synthesis_acceptance_20261121.py --self-test
    python scripts/slm/issue247_synthesis_acceptance_20261121.py --out <evidence.json>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy.ndimage import maximum_filter

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_PATH = Path(__file__).resolve().parent / "issue247_geometry_capture_20261120.py"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_capture_module() -> Any:
    """Load the capture script as a module to reuse its phase builders.

    Single source of truth: the acceptance must judge the *exact* phase arrays the
    capture would write, not a re-implementation that could drift.
    """
    spec = importlib.util.spec_from_file_location(
        "issue247_geometry_capture_20261120", CAPTURE_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load capture module from {CAPTURE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cap = _load_capture_module()

PANEL_SHAPE = cap.DEFAULT_PANEL_SHAPE  # (rows, cols)

#: A spot counts as a distinct order only if it sits at least this many FFT bins
#: away from every other order (bins are integer far-field indices).
MIN_SPOT_SEPARATION_BINS = 2.0
#: A non-zero order must be at least this far from the zero order to be resolvable.
MIN_ORDER_DISTANCE_BINS = 3.0
#: Radius (bins) around a predicted spot in which an FFT local maximum counts as
#: confirmation.  Predicted bins are often non-integer (e.g. 1920/128 = 15 is
#: integer, but 1200/128 = 9.375 rounds to 9), so a whole-bin radius is needed.
VERIFY_RADIUS_BINS = 2


def blaze2d_phase(
    shape: tuple[int, int],
    *,
    period_x_px: float,
    period_y_px: float,
    sign_x: int = 1,
    sign_y: int = 1,
) -> np.ndarray:
    """Full-panel 2D linear phase ramp (a 2D prism / blaze).

    The far field of ``exp(i*phi)`` with a pure linear ramp is the aperture
    diffraction pattern shifted to a **single** spot at frequency
    ``(sign_y / period_y_px, sign_x / period_x_px)``.  One capture therefore
    encodes both axes' sign *and* magnitude at once: with ``period_x != period_y``
    the spot is off-axis in a diagonal direction, so axis swap, sign flip and
    rot180 each move it to a different place.

    Single source of truth: delegates to :func:`cap.blaze2d_phase`, the exact
    builder the capture script writes to the panel, so the acceptance cannot
    drift from what hardware would emit.
    """
    return cap.blaze2d_phase(
        shape,
        period_x_px=period_x_px,
        period_y_px=period_y_px,
        sign_x=sign_x,
        sign_y=sign_y,
    )


def build_candidates() -> list[dict[str, Any]]:
    """Candidate registration patterns: existing probes + the new 2D blaze.

    Each item carries the exact phase builder call the capture would use, so the
    acceptance judges the real arrays.
    """
    shape = PANEL_SHAPE
    items: list[dict[str, Any]] = [
        {
            "id": "flat",
            "kind": "flat",
            "role": "zero-order reference (no registration content)",
            "build": lambda: cap.flat_phase(shape),
        },
    ]
    for period in cap.GRATING_PERIODS_PX:
        for axis, label in ((1, "x"), (0, "y")):
            items.append(
                {
                    "id": f"grating_{label}_p{period}",
                    "kind": "grating",
                    "role": "per-axis scale (symmetric: branch-blind)",
                    "params": {"period_px": period, "axis": label},
                    "build": lambda p=period, a=axis: cap.grating_phase(
                        shape, period_px=p, axis=a
                    ),
                }
            )
    for axis, label in ((1, "x"), (0, "y")):
        for sign in (1, -1):
            items.append(
                {
                    "id": f"tilt_{label}_{'p' if sign > 0 else 'm'}",
                    "kind": "tilt",
                    "role": "prismatic sign / handedness (one axis at a time)",
                    "params": {
                        "period_px": cap.DEFAULT_TILT_PERIOD_PX,
                        "axis": label,
                        "sign": sign,
                    },
                    "build": lambda a=axis, s=sign: cap.tilt_phase(
                        shape, period_px=cap.DEFAULT_TILT_PERIOD_PX, axis=a, sign=s
                    ),
                }
            )
    for axis, label in ((1, "x"), (0, "y")):
        items.append(
            {
                "id": f"ladder_{label}",
                "kind": "ladder",
                "role": "one-sided blaze ladder: 180-deg branch resolver",
                "params": {"axis": label},
                "build": lambda a=axis: cap.ladder_phase(shape, axis=a),
            }
        )
    # New: 2D asymmetric blaze.  Unequal periods put the single spot off-axis so
    # that the Klein-four branch family maps it to four *different* positions.
    for sign_x, sign_y in ((1, 1), (1, -1)):
        items.append(
            {
                "id": f"blaze2d_sx{sign_x:+d}_sy{sign_y:+d}",
                "kind": "blaze2d",
                "role": "2D asymmetric marker: both axes + branch in one spot",
                "params": {
                    "period_x_px": 128.0,
                    "period_y_px": 192.0,
                    "sign_x": sign_x,
                    "sign_y": sign_y,
                },
                "build": lambda sx=sign_x, sy=sign_y: blaze2d_phase(
                    shape, period_x_px=128.0, period_y_px=192.0, sign_x=sx, sign_y=sy
                ),
            }
        )
    return items


def predict_spots(item: dict[str, Any], shape: tuple[int, int]) -> list[tuple[float, float]]:
    """Analytic signed far-field spot prediction ``(d_row, d_col)`` in bins.

    A phase ramp of period ``P`` over ``N`` panel pixels puts its first order at
    ``N/P`` FFT bins from the zero order.  The zero order ``(0, 0)`` is always
    included.  ``d_row`` is the row-frequency offset, ``d_col`` the column one,
    matching the panel's ``(rows, cols)`` = ``(y, x)`` convention.
    """
    kind = item["kind"]
    params = item.get("params", {})
    rows, cols = int(shape[0]), int(shape[1])
    if kind == "flat":
        return [(0.0, 0.0)]
    if kind == "grating":
        period = float(params["period_px"])
        axis = params["axis"]
        if axis == "x":
            return [(0.0, cols / period), (0.0, -cols / period), (0.0, 0.0)]
        return [(rows / period, 0.0), (-rows / period, 0.0), (0.0, 0.0)]
    if kind == "tilt":
        period = float(params["period_px"])
        axis = params["axis"]
        sign = int(params["sign"])
        n = cols if axis == "x" else rows
        effective = cap.effective_tilt_period(period, n)
        offset = float(sign) * n / effective
        return [(0.0, offset) if axis == "x" else (offset, 0.0), (0.0, 0.0)]
    if kind == "ladder":
        axis = params["axis"]
        n = cols if axis == "x" else rows
        spots: list[tuple[float, float]] = [(0.0, 0.0)]
        for order in cap.LADDER_ORDERS:
            period = cap.LADDER_BASE_PERIOD_PX / float(order)
            offset = n / period
            spots.append((0.0, offset) if axis == "x" else (offset, 0.0))
        return spots
    if kind == "blaze2d":
        sign_x = int(params["sign_x"])
        sign_y = int(params["sign_y"])
        period_x = float(params["period_x_px"])
        period_y = float(params["period_y_px"])
        # No zero order: a full-panel continuous ramp of unit amplitude sends
        # *all* the energy to its single first order, and the panel-wide sum of
        # exp(i*ramp) cancels (cols/period_x = 1920/128 = 15 is an exact integer,
        # so the DC bin is a null: ~1e-23 here).  Predicting (0, 0) makes
        # verify_spots hunt for a local maximum that physics does not put there.
        return [(float(sign_y) * rows / period_y, float(sign_x) * cols / period_x)]
    raise ValueError(f"unknown candidate kind {kind!r}")


def verify_spots(
    phase: np.ndarray, predicted: list[tuple[float, float]]
) -> list[dict[str, Any]]:
    """Confirm each predicted spot has an FFT local maximum nearby.

    Returns per-spot ``predicted`` / ``measured`` / ``intensity_rel`` (relative to
    the zero-order peak) / ``confirmed``.  The zero-order peak is the global apex
    for every candidate here, so ``intensity_rel`` is a stable reference.
    """
    field = np.exp(1j * phase)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    height, width = spectrum.shape
    centre = (height // 2, width // 2)
    zero_val = float(spectrum[centre[0], centre[1]])
    peak_map = maximum_filter(spectrum, size=3, mode="nearest")
    is_peak = spectrum == peak_map

    out: list[dict[str, Any]] = []
    for pred_row, pred_col in predicted:
        r0 = centre[0] + int(round(pred_row))
        c0 = centre[1] + int(round(pred_col))
        r_lo, r_hi = max(0, r0 - VERIFY_RADIUS_BINS), min(height, r0 + VERIFY_RADIUS_BINS + 1)
        c_lo, c_hi = max(0, c0 - VERIFY_RADIUS_BINS), min(width, c0 + VERIFY_RADIUS_BINS + 1)
        best_val = -1.0
        best_at: tuple[int, int] | None = None
        for r in range(r_lo, r_hi):
            for c in range(c_lo, c_hi):
                if is_peak[r, c]:
                    val = float(spectrum[r, c])
                    if val > best_val:
                        best_val = val
                        best_at = (r, c)
        if best_at is None:
            out.append(
                {
                    "predicted": [pred_row, pred_col],
                    "measured": None,
                    "intensity_rel": None,
                    "confirmed": False,
                }
            )
        else:
            out.append(
                {
                    "predicted": [pred_row, pred_col],
                    "measured": [best_at[0] - centre[0], best_at[1] - centre[1]],
                    "intensity_rel": (best_val / zero_val) if zero_val > 0 else None,
                    "confirmed": True,
                }
            )
    return out


def spot_set(spots: list[tuple[float, float]]) -> frozenset[tuple[float, float]]:
    """Signed-offset set of the non-zero-order predicted spots."""
    return frozenset((r, c) for (r, c) in spots if (r, c) != (0.0, 0.0))


def _transform(spots: frozenset[tuple[float, float]], kind: str) -> frozenset[tuple[float, float]]:
    if kind == "identity":
        return spots
    if kind == "sign_flip":  # rot180 acts the same way in the far-field domain
        return frozenset((-r, -c) for (r, c) in spots)
    if kind == "axis_swap":
        return frozenset((c, r) for (r, c) in spots)
    if kind == "sign_flip_axis_swap":
        return frozenset((-c, -r) for (r, c) in spots)
    raise ValueError(f"unknown transform {kind!r}")


BRANCH_FAMILY: tuple[str, ...] = (
    "identity",
    "sign_flip",
    "axis_swap",
    "sign_flip_axis_swap",
)


def analyse(item: dict[str, Any], shape: tuple[int, int]) -> dict[str, Any]:
    """Synthesise one candidate, predict its spots, verify them, judge them."""
    phase = item["build"]()
    predicted = predict_spots(item, shape)
    verified = verify_spots(phase, predicted)
    non_zero = spot_set(predicted)

    # Resolvability: orders must be clear of the zero order and of each other.
    min_order_dist = float(
        min((np.hypot(r, c) for (r, c) in non_zero), default=float("inf"))
    )
    min_pair = float("inf")
    ordered = sorted(non_zero)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            min_pair = min(min_pair, float(np.hypot(a[0] - b[0], a[1] - b[1])))

    has_orders = bool(non_zero)
    resolvable = (not has_orders) or (
        min_pair >= MIN_SPOT_SEPARATION_BINS and min_order_dist >= MIN_ORDER_DISTANCE_BINS
    )

    # Branch distinguishability: the four Klein-four images must be pairwise
    # distinct for the candidate to resolve the 180-deg family on its own.
    images = {kind: _transform(non_zero, kind) for kind in BRANCH_FAMILY}
    distinct = len({frozenset(v) for v in images.values()}) == len(BRANCH_FAMILY)

    # Every predicted spot -- including the zero order -- must carry an FFT local
    # maximum.  A reference pattern with no non-zero orders is therefore
    # vacuously confirmed, not rejected.
    all_confirmed = all(v["confirmed"] for v in verified)

    return {
        "id": item["id"],
        "kind": item["kind"],
        "role": item["role"],
        "params": item.get("params", {}),
        "phase_sha256": cap.digest(phase),
        "phase_shape": list(phase.shape),
        "predicted_spots": [list(s) for s in predicted],
        "verified_spots": verified,
        "non_zero_spots": sorted(non_zero),
        "min_spot_separation_bins": min_pair,
        "min_order_distance_bins": min_order_dist,
        "resolvable": resolvable,
        "fft_confirmed": all_confirmed,
        "branch_images": {k: sorted(v) for k, v in images.items()},
        "branch_distinguishable": distinct,
        "resolves_branch_alone": bool(resolvable and distinct),
    }


def run_acceptance() -> dict[str, Any]:
    """Analyse every candidate and write the aggregate verdict."""
    items = build_candidates()
    results = [analyse(item, PANEL_SHAPE) for item in items]

    # Every candidate must be resolvable (or a reference), and the set must
    # contain at least one candidate that resolves the branch on its own.
    all_resolvable = all(r["resolvable"] for r in results)
    all_confirmed = all(r["fft_confirmed"] for r in results)
    any_resolves = any(r["resolves_branch_alone"] for r in results)

    verdict = {
        "schema": "issue247-synthesis-acceptance/v1",
        "panel_shape": list(PANEL_SHAPE),
        "branch_family": list(BRANCH_FAMILY),
        "criteria": {
            "min_spot_separation_bins": MIN_SPOT_SEPARATION_BINS,
            "min_order_distance_bins": MIN_ORDER_DISTANCE_BINS,
            "verify_radius_bins": VERIFY_RADIUS_BINS,
        },
        "candidates": results,
        "all_resolvable": all_resolvable,
        "all_fft_confirmed": all_confirmed,
        "any_resolves_branch": any_resolves,
        "accepted": bool(all_resolvable and all_confirmed and any_resolves),
    }
    return verdict


def run_self_test() -> int:
    """Assert the offline invariants with no device and no directory creation."""
    verdict = run_acceptance()
    by_id = {c["id"]: c for c in verdict["candidates"]}

    # 1) The new 2D blaze must be resolvable, FFT-confirmed and resolve the
    #    branch on its own.
    for cid in ("blaze2d_sx+1_sy+1", "blaze2d_sx+1_sy-1"):
        item = by_id[cid]
        assert item["resolvable"], f"{cid} not resolvable"
        assert item["fft_confirmed"], f"{cid} predicted spots not confirmed by FFT"
        assert item["branch_distinguishable"], f"{cid} branch images not distinct"
        assert item["resolves_branch_alone"], f"{cid} does not resolve the branch"

    # 2) The 2D blaze's single spot must be off-axis in both components (both
    #    axes encoded at once) and its two axes must differ in magnitude.
    spot = by_id["blaze2d_sx+1_sy+1"]["non_zero_spots"]
    assert len(spot) == 1, f"expected one dominant off-axis spot, got {spot}"
    (dr, dc) = spot[0]
    assert dr != 0 and dc != 0, f"blaze2d spot must be diagonal, got {spot}"
    assert abs(dr) != abs(dc), f"blaze2d periods must differ, got {spot}"

    # 3) The one-sided ladder must resolve the branch (that is its whole job) and
    #    its +1/+3 orders must be FFT-confirmed even though they are weak.
    for cid in ("ladder_x", "ladder_y"):
        assert by_id[cid]["resolvable"], f"{cid} not resolvable"
        assert by_id[cid]["fft_confirmed"], f"{cid} weak orders not confirmed"
        assert by_id[cid]["resolves_branch_alone"], f"{cid} does not resolve the branch"

    # 4) A symmetric grating is branch-blind by design -- it must still be
    #    resolvable, but must NOT claim to resolve the branch on its own.
    grating = by_id["grating_x_p128"]
    assert grating["resolvable"], "grating must be resolvable"
    assert not grating["branch_distinguishable"], "symmetric grating must be branch-blind"

    # 5) Collective verdict.
    assert verdict["all_resolvable"], "some candidate is not resolvable"
    assert verdict["all_fft_confirmed"], "some predicted spot is not FFT-confirmed"
    assert verdict["any_resolves_branch"], "no candidate resolves the branch"
    assert verdict["accepted"], "acceptance must pass"

    logger.info(
        "issue247 synthesis acceptance self-test: all invariants passed "
        f"({len(verdict['candidates'])} candidates, accepted={verdict['accepted']})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Issue #247 A0-3 offline synthesis acceptance: predict each candidate "
            "registration pattern's signed far-field spots, verify them with an "
            "FFT, and judge resolvability and Klein-four branch distinguishability. "
            "Zero device, zero capture directories."
        )
    )
    parser.add_argument("--out", type=Path, default=None, help="evidence JSON path")
    parser.add_argument(
        "--self-test", action="store_true", help="assert offline invariants; no --out"
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    verdict = run_acceptance()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logger.info("synthesis acceptance evidence written: {}", args.out)

    logger.info(
        "synthesis acceptance: accepted={} all_resolvable={} all_confirmed={} any_resolves_branch={}",
        verdict["accepted"],
        verdict["all_resolvable"],
        verdict["all_fft_confirmed"],
        verdict["any_resolves_branch"],
    )
    for cand in verdict["candidates"]:
        logger.info(
            "  {:<24} resolvable={:<5} confirmed={:<5} branch_alone={:<5} spots={}",
            cand["id"],
            cand["resolvable"],
            cand["fft_confirmed"],
            cand["resolves_branch_alone"],
            cand["non_zero_spots"],
        )
    return 0 if verdict["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
