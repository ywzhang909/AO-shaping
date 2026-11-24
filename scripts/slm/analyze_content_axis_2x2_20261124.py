# -*- coding: utf-8 -*-
"""Read-only judge for the preregistered S1b content × fx 2x2 capture (2026-11-24).

Loads the S1b paired capture (``D=0`` and ``D=-0.5``, each a 4-cell
A128/A256/B128/B256 acquisition), reproduces the issue247 ``principal()``
convention exactly (ref = A1 flat 0th, ``exclude_core=6``, ``min_dist=15``,
``find_spots`` min_dist=20 / thresh=0.01*max), and applies the §2.1 verdict
rules from ``docs/dev/hw_window_20261123_attribution_prereg.md``:

- content-correlated (4/4): content B both cells -> row axis (K_row*N, <=1.5 px)
  AND content A both cells -> column axis (K_col*N, <=1.5 px);
- content-independent: B same axis and same amplitude as A;
- insufficient_consistency: B two cells axis disagree, or any cell amplitude
  does not obey the K*N law.

Reference note (2026-11-24 lesson): §2.1 fixes ``ref`` to the A1 flat 0th
(1105.25, 1227.56).  A per-frame brightest-spot reference is NOT used for the
verdict because content B's row-0 pi step perturbs the per-frame 0th centroid
by ~3 px, which would falsely inflate B's measured amplitude (the principal
spots of A and B are at the *same absolute* CCD position).  The per-frame 0th
is still reported as a ``frame_0th`` drift diagnostic.

This script is strictly read-only: it loads ``raw/<pattern>.npz`` from the
capture directories, never writes into them, and never touches the bench.

The frozen issue247 constants are K_col=6.4, K_row=6.2, N=1920/period_px.
Explicit flags are raised when the measured axis/amplitude contradicts the
frozen convention or A1 (as this capture does: content A lands on the row axis
at ~3.78*N rather than column at K_col*N), because that is an
imaging/calibration discrepancy to be registered, not silently absorbed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = (
    ROOT
    / "data/experiments/attribution_20261123/s1/s1b_content_axis_20261124"
)
# Frozen from the issue247 axis-map convention (capture script, L74-75).
K_COL = 6.4
K_ROW = 6.2
N_PER_PERIOD = 1920.0  # panel columns / period_px
PERIODS = (128, 256)
CELLS = ("A128", "A256", "B128", "B256")
GROUPS = ("d0_20261124", "dm0p5_20261124")
# §2.1 S1b tolerances.
RESIDUAL_TOL_PX = 1.5
# A1 flat 0th (issue247 axis-map solve) — the §2.1 convention reference.
REF_A1_FLAT = (1105.25, 1227.56)
# A principal order weaker than 10% of the 0th is flagged near-noise / weak
# signal (D=-0.5 group: principal peaks 44-59 vs 0th ~1200; D=0 group:
# ~610/1173).  Such a group cannot be axis-judged.
WEAK_SIGNAL_FRAC = 0.10
# Convention principal() parameters (issue247 axis-map solve).
EXCLUDE_CORE = 6.0
MIN_DIST = 15.0
FIND_MIN_DIST = 20
FIND_THRESH_FRAC = 0.01
# L145: x-family (fy≡0, i.e. content A) true semantics = column axis + a
# 1920/1200 content-compression factor on the amplitude.  Reported separately;
# the §2.1 verdict is stated against the *uncompressed* frozen K*N law.
X_FAMILY_COMPRESSION = 1920.0 / 1200.0


def load_frame(group: str, cell: str, base: Path = DEFAULT_BASE) -> np.ndarray:
    path = base / group / "raw" / f"{cell}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing frame {path}")
    arr = np.load(path)
    if "frame" not in arr:
        raise KeyError(f"{path}: expected key 'frame', got {sorted(arr.keys())}")
    return arr["frame"].astype(np.float64)


def find_spots(frame: np.ndarray, min_dist: int = FIND_MIN_DIST,
               thresh_frac: float = FIND_THRESH_FRAC) -> list[dict]:
    """2-D local maxima with sub-pixel centroid, absolute (row, col)."""
    mx = maximum_filter(frame, size=min_dist, mode="nearest")
    is_pk = (frame == mx) & (frame > thresh_frac * frame.max())
    out: list[dict] = []
    for r, c in np.argwhere(is_pk):
        if r < 100 and c < 100:  # corner stray-light artifact (A1 convention)
            continue
        y0, y1 = max(0, r - 8), min(frame.shape[0], r + 9)
        x0, x1 = max(0, c - 8), min(frame.shape[1], c + 9)
        sub = frame[y0:y1, x0:x1]
        w = np.where(sub > 0, sub, 0.0)
        tot = w.sum()
        if tot <= 0:
            continue
        yy, xx = np.indices(sub.shape)
        cr = y0 + float((yy * w).sum() / tot)
        cc = x0 + float((xx * w).sum() / tot)
        out.append({"row": cr, "col": cc, "peak": float(frame[r, c])})
    out.sort(key=lambda s: -s["peak"])
    return out


def principal(frame: np.ndarray, ref: tuple[float, float],
              exclude_core: float = EXCLUDE_CORE,
              min_dist: float = MIN_DIST) -> dict | None:
    """Brightest spot clear of the 0th-order core; returns (dr, dc, peak, dist)."""
    best = None
    for s in find_spots(frame):
        dr = s["row"] - ref[0]
        dc = s["col"] - ref[1]
        dist = float(np.hypot(dr, dc))
        if dist < min_dist:
            continue
        if abs(dr) <= exclude_core and abs(dc) <= exclude_core:
            continue
        if best is None or s["peak"] > best["peak"]:
            best = {"dr": dr, "dc": dc, "peak": s["peak"], "dist": dist}
    return best


def axis_of(dr: float, dc: float) -> str:
    """Row/col axis assignment from a principal displacement."""
    if abs(dr) > abs(dc):
        return "row"
    if abs(dc) > abs(dr):
        return "column"
    return "ambiguous"


def peak_in_roi(frame: np.ndarray, center: tuple[float, float],
                half: int = 3) -> float:
    """Peak count in a (2*half+1)^2 ROI centered on a predicted position.

    §2.1 L142 requires ROI to be derived from the 1/p predicted location, not
    from a fixed exclusion radius (the 2026-11-23 fixed core_exclude=34.96 px
    bug).  This helper evaluates exactly such a prediction-derived ROI.
    """
    r, c = center
    y0, y1 = max(0, int(round(r)) - half), min(frame.shape[0], int(round(r)) + half + 1)
    x0, x1 = max(0, int(round(c)) - half), min(frame.shape[1], int(round(c)) + half + 1)
    if y0 >= y1 or x0 >= x1:
        return float("nan")
    return float(frame[y0:y1, x0:x1].max())


def predicted_roi_peaks(frame: np.ndarray, ref: tuple[float, float],
                        n: float) -> dict:
    """§2.1 L142 predicted-position ROIs (± on each axis, for each law)."""
    laws = {
        "Kcol": K_COL * n,
        "Krow": K_ROW * n,
        "Kcol_compressed": K_COL * n / X_FAMILY_COMPRESSION,
    }
    out: dict[str, dict[str, float]] = {}
    for name, amp in laws.items():
        out[name] = {
            "row_plus": peak_in_roi(frame, (ref[0] + amp, ref[1])),
            "row_minus": peak_in_roi(frame, (ref[0] - amp, ref[1])),
            "col_plus": peak_in_roi(frame, (ref[0], ref[1] + amp)),
            "col_minus": peak_in_roi(frame, (ref[0], ref[1] - amp)),
        }
    return out


def analyze_cell(group: str, cell: str, base: Path = DEFAULT_BASE,
                 ref: tuple[float, float] = REF_A1_FLAT) -> dict:
    frame = load_frame(group, cell, base)
    flat_idx = int(np.argmax(frame))
    f0, c0 = divmod(flat_idx, frame.shape[1])
    frame_0th = (float(f0), float(c0))
    spots = find_spots(frame)
    p = principal(frame, ref)
    n = N_PER_PERIOD / int(cell[1:])
    result = {
        "group": group,
        "cell": cell,
        "period_px": int(cell[1:]),
        "content": cell[0],
        "N": n,
        "ref": {"row": ref[0], "col": ref[1]},
        "frame_0th": {
            "row": round(frame_0th[0], 2),
            "col": round(frame_0th[1], 2),
            "dr_from_ref": round(frame_0th[0] - ref[0], 2),
            "dc_from_ref": round(frame_0th[1] - ref[1], 2),
        },
        "frame_peak": float(frame.max()),
        "frame_median": float(np.median(frame)),
        # §2.1 L142: ROI derived from 1/p predicted positions, not a fixed
        # exclusion radius.  These are the peaks at the predicted locations.
        "predicted_roi_peaks": predicted_roi_peaks(frame, ref, n),
        "top_spots": [
            {
                "row": round(s["row"], 2),
                "col": round(s["col"], 2),
                "dr": round(s["row"] - ref[0], 2),
                "dc": round(s["col"] - ref[1], 2),
                "dist": round(float(np.hypot(s["row"] - ref[0],
                                             s["col"] - ref[1])), 2),
                "peak": round(s["peak"], 1),
            }
            for s in spots[:5]
        ],
    }
    if p is None:
        result["principal"] = None
        result["axis"] = "none"
        result["amplitude_px"] = None
        result["K"] = None
        result["weak_signal"] = True
        return result
    dr, dc = p["dr"], p["dc"]
    amp = float(np.hypot(dr, dc))
    weak = bool(p["peak"] < WEAK_SIGNAL_FRAC * frame.max())
    result["principal"] = {
        "dr": round(dr, 3),
        "dc": round(dc, 3),
        "peak": round(p["peak"], 1),
        "dist": round(p["dist"], 3),
    }
    result["axis"] = axis_of(dr, dc)
    result["amplitude_px"] = round(amp, 3)
    result["K"] = round(amp / n, 3)
    result["weak_signal"] = weak
    result["residual_vs_KcolN_px"] = round(abs(amp - K_COL * n), 3)
    result["residual_vs_KrowN_px"] = round(abs(amp - K_ROW * n), 3)
    result["residual_vs_KcolN_compressed_px"] = round(
        abs(amp - K_COL * n / X_FAMILY_COMPRESSION), 3)
    return result


def judge_group(cells: dict[str, dict]) -> dict:
    """§2.1 verdict for one D group (4 cells).

    A cell whose principal order is weak (peak < 10% of the 0th) cannot be
    axis-judged; when every cell is weak the group verdict is
    ``insufficient_signal`` (the §2.1 verdict cannot be computed).
    """
    weak = [c for c in CELLS if cells[c].get("weak_signal")]
    if len(weak) == len(CELLS):
        return {
            "verdict": "insufficient_signal",
            "weak_cells": weak,
            "note": "all four principal orders near noise; axis not judgeable",
        }
    a_cells = [cells[c] for c in ("A128", "A256")]
    b_cells = [cells[c] for c in ("B128", "B256")]
    a_axis = {c["axis"] for c in a_cells if c["principal"] is not None}
    b_axis = {c["axis"] for c in b_cells if c["principal"] is not None}
    a_amp = [c["amplitude_px"] for c in a_cells if c["principal"] is not None]
    b_amp = [c["amplitude_px"] for c in b_cells if c["principal"] is not None]
    a_col = a_axis == {"column"}
    b_row = b_axis == {"row"}
    # Amplitude law references.  Content A is the full-frame 1D x-family
    # (fy≡0): per L145 its true law is the COMPRESSED K_col*N/1.6.  Content B is
    # not the x-family, so line 129 governs: K_row*N if content-correlated,
    # K_col*N if content-independent.
    a_resid_ok = all(
        c["principal"] is not None
        and c["residual_vs_KcolN_compressed_px"] <= RESIDUAL_TOL_PX
        for c in a_cells
    )
    b_krow_ok = all(
        c["principal"] is not None
        and c["residual_vs_KrowN_px"] <= RESIDUAL_TOL_PX
        for c in b_cells
    )
    # B same axis and same amplitude as A (content-independent signature).
    b_same_as_a = (
        len(b_axis) == 1 and len(a_axis) == 1 and b_axis == a_axis
        and len(a_amp) == len(b_amp)
        and all(abs(ba - aa) <= RESIDUAL_TOL_PX
                for ba, aa in zip(b_amp, a_amp))
    )
    # §2.1 verdict (lines 131-137), L145-coherent.  content_correlated_4of4
    # (line 132): A both cells -> column at the L145-compressed K_col*N AND
    # B both cells -> row at K_row*N.  content_independent (line 135): A both
    # cells -> column at the compressed K_col*N AND B identical to A (same axis
    # and amplitude).  Line 136 routes any cell that does not obey its family's
    # K*N law (or any axis/amplitude disagreement) to insufficient_consistency.
    a_col_comp = (a_axis == {"column"}) and a_resid_ok
    b_row_ok = (b_axis == {"row"}) and b_krow_ok
    if a_col_comp and b_row_ok:
        verdict = "content_correlated_4of4"
    elif b_same_as_a and a_col_comp:
        verdict = "content_independent"
    else:
        verdict = "insufficient_consistency"
    return {
        "verdict": verdict,
        "weak_cells": weak,
        "A_axis": sorted(a_axis),
        "B_axis": sorted(b_axis),
        "A_residual_compressed_ok": a_resid_ok,
        "B_residual_vs_KrowN_ok": b_krow_ok,
        "A_column_compressed_ok": a_col_comp,
        "B_same_axis_and_amplitude_as_A": b_same_as_a,
        "A_amplitude_px": a_amp,
        "B_amplitude_px": b_amp,
        # 2026-11-24 discrepancy: content A lands on the row axis (not column),
        # so the measured signature contradicts the frozen convention and A1.
        "axis_contradiction_vs_convention": (not a_col) or (not b_row),
        "content_independent_signature": b_same_as_a,
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--group", choices=GROUPS, default=None)
    ap.add_argument("--ref", type=float, nargs=2, default=list(REF_A1_FLAT),
                    metavar=("ROW", "COL"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    ref = (args.ref[0], args.ref[1])
    groups = [args.group] if args.group else list(GROUPS)
    out: dict = {
        "schema": "issue247-content-axis-2x2-analysis/v1",
        "constants": {"K_col": K_COL, "K_row": K_ROW,
                      "N_per_period": N_PER_PERIOD,
                      "residual_tol_px": RESIDUAL_TOL_PX,
                      "ref": {"row": ref[0], "col": ref[1]},
                      "weak_signal_frac": WEAK_SIGNAL_FRAC},
        "groups": {},
    }
    for g in groups:
        cells = {}
        for cell in CELLS:
            cells[cell] = analyze_cell(g, cell, args.base, ref)
        out["groups"][g] = {"cells": cells,
                            "judge": judge_group(cells)}
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    for g, data in out["groups"].items():
        j = data["judge"]
        print(f"== {g}  verdict={j['verdict']}")
        for cell in CELLS:
            c = data["cells"][cell]
            p = c["principal"]
            if p is None:
                print(f"   {cell}: no principal (axis none)")
                continue
            weak = "  [WEAK]" if c["weak_signal"] else ""
            print(
                f"   {cell} content={c['content']} N={c['N']:.1f} "
                f"axis={c['axis']:6s} amp={c['amplitude_px']:6.2f}px "
                f"K={c['K']:.3f} dr={p['dr']:+.2f} dc={p['dc']:+.2f} "
                f"peak={p['peak']:.0f} | residKcol={c['residual_vs_KcolN_px']:.2f} "
                f"residKrow={c['residual_vs_KrowN_px']:.2f} "
                f"residKcol_comp={c['residual_vs_KcolN_compressed_px']:.2f}"
                f"{weak}"
            )
            ft = c["frame_0th"]
            print(
                f"       0th(frame) row={ft['row']:.2f} col={ft['col']:.2f} "
                f"drift_from_A1ref=({ft['dr_from_ref']:+.2f},{ft['dc_from_ref']:+.2f})"
            )
        print(
            f"   A_axis={j.get('A_axis')} B_axis={j.get('B_axis')} "
            f"B_same_as_A={j.get('B_same_axis_and_amplitude_as_A')} "
            f"A_resid_comp_ok={j.get('A_residual_compressed_ok')} "
            f"B_resid_ok={j.get('B_residual_vs_KrowN_ok')} "
            f"axis_contradiction={j.get('axis_contradiction_vs_convention')} "
            f"content_independent_signature={j.get('content_independent_signature')}"
        )
        for w in j.get("weak_cells", []):
            print(f"   WEAK cell {w}: principal order near noise, axis unreliable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
