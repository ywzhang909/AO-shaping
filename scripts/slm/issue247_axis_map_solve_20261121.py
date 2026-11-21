"""Issue #247: attempt the panel<->CCD 2x2 affine map from an A1 capture.

Reads ``<capture>/frames/*.npy`` (each pattern averaged over its frames), locates
each probe's principal far-field spot relative to the ``flat`` 0th order, and
tests whether one linear map (panel spatial frequency -> CCD-px displacement) can
explain the probes.  Writes ``axis_mapping_status<prefix>.json`` and a readable
``README_axis_mapping<prefix>.md`` next to the frames.  The frozen map itself
(``issue247_axis_map<prefix>.json``) is written ONLY when the rule below passes.

Two rounds of evidence:

* the A1 probes (grating / tilt / ladder, 2026-11-21) -- direction diagnostics.
  ``principal()`` picks the brightest order of each pattern, so a probe set that
  images both panel axes onto one CCD direction (as A1 measured) leaves the 2x2
  map undetermined no matter how the data are fitted.
* the 2-D diagonal blaze family (``blaze2d_px*_py*_sx+1_sy+-1``, 2026-11-21b) --
  four full-panel 2-D phase ramps, each producing a SINGLE off-axis spot whose
  position encodes both panel axes' sign and scale at once.  This is the only
  probe set here that pins a 2x2 map in two dimensions.

PRE-REGISTERED FREEZE RULE (``docs/dev/hw_window_20261121b_issue247_prereg.md``,
fixed before the 2026-11-21b capture and not to be relaxed afterwards).  Model
``(dr, dc) = A @ (fy, fx)`` with ``A`` 2x2 in CCD px per (cycle / panel-pixel),
least-squares over the four blaze2d points only: 8 equations, 4 unknowns, so the
residual is meaningful.  Sign flips alone are linearly dependent (flipping both
signs gives exactly ``-A v``), which is why two period pairs, 128/192 and
192/128, are required -- only then can an inconsistent map be detected.
``frozen = true`` if and only if ALL four hold:

1. blaze2d residual RMS <= 1.0 px;
2. max per-point relative error <= 2 % (|residual| / |measured displacement|);
3. the fitted A's two column directions differ by 88.544 deg +- 2 deg;
4. rank guard |det A| / sigma_max(A)^2 >= 0.10 (the two panel axes must stay
   linearly independent on the CCD; this is sigma_min / sigma_max for a 2x2).

Otherwise ``frozen = false`` and ``verdict`` names every failing condition.

The per-family 2x2 residuals (grating-only / tilt-only / ladder-only) are
auxiliary attribution evidence: they are printed and recorded so the A1 camp
conflict shows up as a number rather than only as angles, but they never freeze
or unfreeze anything.

If the blaze2d frames are absent (older capture dirs such as
``data/debug/hw_window_20261121/issue247_axis``) the family is skipped with a
clear line, the legacy ambiguous verdict is reproduced unchanged, and the exit
code is unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE_DIR = ROOT / "data/debug/hw_window_20261121/issue247_axis"

# Pre-registered thresholds.  Changing them after a capture invalidates the
# 2026-11-21b pre-registration, so they live here verbatim, not as CLI options.
FREEZE_RULES: dict[str, float] = {
    "residual_rms_max_px": 1.0,
    "max_rel_err_max": 0.02,
    "axis_separation_expected_deg": 88.544,
    "axis_separation_tol_deg": 2.0,
    "rank_guard_min": 0.10,
}

# panel spatial-frequency (cycles / panel-pixel) of each probe's principal order,
# as (k_row, k_col).  axis=1 (panel-x, columns) -> k_col only; axis=0 -> k_row.
# For blaze2d the single far-field spot sits at (fy, fx) = (sign_y/period_y_px,
# sign_x/period_x_px), i.e. k_row = fy, k_col = fx.
PROBES = [
    # id,                 kind,        k_row,        k_col
    ("grating_x_p128", "grating", 0.0, 1.0 / 128.0),
    ("grating_y_p128", "grating", 1.0 / 128.0, 0.0),
    ("grating_x_p256", "grating", 0.0, 1.0 / 256.0),
    ("grating_y_p256", "grating", 1.0 / 256.0, 0.0),
    ("tilt_x_plus", "tilt+", 0.0, 1.0 / 128.0),
    ("tilt_x_minus", "tilt-", 0.0, -1.0 / 128.0),
    ("tilt_y_plus", "tilt+", 1.0 / 120.0, 0.0),   # tilt_y effective period = 120 px
    ("tilt_y_minus", "tilt-", -1.0 / 120.0, 0.0),
    ("ladder_x", "ladder", 0.0, 1.0 / 96.0),      # order 1 patch
    ("ladder_y", "ladder", 1.0 / 96.0, 0.0),      # order 1 patch
    ("blaze2d_px128_py192_sx+1_sy+1", "blaze2d", 1.0 / 192.0, 1.0 / 128.0),
    ("blaze2d_px128_py192_sx+1_sy-1", "blaze2d", -1.0 / 192.0, 1.0 / 128.0),
    ("blaze2d_px192_py128_sx+1_sy+1", "blaze2d", 1.0 / 128.0, 1.0 / 192.0),
    ("blaze2d_px192_py128_sx+1_sy-1", "blaze2d", -1.0 / 128.0, 1.0 / 192.0),
]

#: Used when ``summary.json`` has no ``blaze2d_designs`` key (pre-2026-11-21b dirs).
BLAZE2D_FALLBACK_IDS: tuple[str, ...] = tuple(
    pid for pid, kind, _, _ in PROBES if kind == "blaze2d"
)
#: Families whose own 2x2 fit is reported as auxiliary attribution evidence.
FAMILIES: tuple[str, ...] = ("blaze2d", "grating", "tilt", "ladder")


# ---------------------------------------------------------------------------
# frame access
# ---------------------------------------------------------------------------
def frame_files(pattern: str, frames: Path) -> list[Path]:
    return sorted(frames.glob(f"*_{pattern}.npy"))


def load(pattern: str, frames: Path) -> np.ndarray:
    files = frame_files(pattern, frames)
    if not files:
        raise FileNotFoundError(f"no frames for pattern {pattern!r} in {frames}")
    return np.mean([np.load(f) for f in files], axis=0).astype(np.float64)


def as_repo_str(path: Path) -> str:
    """Repo-relative POSIX path when possible, absolute otherwise."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def find_spots(frame: np.ndarray, min_dist: int = 20, thresh_frac: float = 0.02):
    """2-D local maxima with sub-pixel centroid, absolute (row, col)."""
    mx = maximum_filter(frame, size=min_dist, mode="nearest")
    is_pk = (frame == mx) & (frame > thresh_frac * frame.max())
    spots = []
    for r, c in np.argwhere(is_pk):
        if r < 100 and c < 100:          # corner stray-light artifact
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
        spots.append({"row": cr, "col": cc, "peak": float(frame[r, c])})
    spots.sort(key=lambda s: -s["peak"])
    return spots


def principal(frame, ref, exclude_core=6.0, min_dist=15.0):
    """Brightest spot clear of the 0th-order core; returns (dr, dc, peak, dist)."""
    best = None
    for s in find_spots(frame):
        dr, dc = s["row"] - ref[0], s["col"] - ref[1]
        dist = float(np.hypot(dr, dc))
        if dist < min_dist:
            continue
        if abs(dr) <= exclude_core and abs(dc) <= exclude_core:
            continue
        if best is None or s["peak"] > best["peak"]:
            best = {"dr": dr, "dc": dc, "peak": s["peak"], "dist": dist}
    return best


def angle_deg(dr: float, dc: float) -> float:
    """CCD angle in degrees, measured from +col (dc) toward +row (dr)."""
    return float(np.degrees(np.arctan2(dr, dc)))


def unit(dr: float, dc: float) -> tuple[float, float]:
    n = float(np.hypot(dr, dc)) or 1.0
    return (dr / n, dc / n)


def angle_between(a, b) -> float:
    ua, ub = unit(*a), unit(*b)
    cosang = max(-1.0, min(1.0, ua[0] * ub[0] + ua[1] * ub[1]))
    return float(np.degrees(np.arccos(cosang)))


# ---------------------------------------------------------------------------
# 2x2 least squares
# ---------------------------------------------------------------------------
def fit_affine(points: list[tuple[float, float, float, float]]):
    """Fit ``(dr, dc) = A @ (fy, fx)``; returns (A, predicted, residual).

    ``points`` are ``(fy, fx, dr, dc)``.  ``A`` is 2x2 with columns
    ``(dr, dc) per unit fy`` and ``(dr, dc) per unit fx``.
    """
    x = np.array([[fy, fx] for fy, fx, _, _ in points], dtype=np.float64)
    y = np.array([[dr, dc] for _, _, dr, dc in points], dtype=np.float64)
    if np.linalg.matrix_rank(x) < 2:
        raise ValueError(
            "probe frequencies are collinear in (fy, fx); A is not identifiable"
        )
    a_t, *_ = np.linalg.lstsq(x, y, rcond=None)
    a = a_t.T
    pred = x @ a_t
    return a, pred, y - pred


def affine_metrics(points, a, residual) -> dict:
    """Residual / conditioning metrics of one 2x2 fit."""
    meas = np.array([float(np.hypot(dr, dc)) for _, _, dr, dc in points])
    res = np.hypot(residual[:, 0], residual[:, 1])
    sv = np.linalg.svd(a, compute_uv=False)
    s_max, s_min = float(sv[0]), float(sv[-1])
    det = float(np.linalg.det(a))
    guard = abs(det) / (s_max * s_max) if s_max > 0 else 0.0
    cols = {
        "fy": (float(a[0, 0]), float(a[1, 0])),
        "fx": (float(a[0, 1]), float(a[1, 1])),
    }
    return {
        "residual_norm_px": res,
        "measured_norm_px": meas,
        "residual_rms_px": float(np.sqrt(np.mean(res**2))),
        "max_point_residual_px": float(res.max()),
        "max_rel_err": float(np.max(res / meas)) if float(meas.min()) > 0 else float("inf"),
        "singular_values": [s_max, s_min],
        "condition_number": float(s_max / s_min) if s_min > 0 else float("inf"),
        "rank_guard": guard,
        "column_vectors": cols,
        "column_angles_deg": {k: angle_deg(*v) for k, v in cols.items()},
        "column_separation_deg": angle_between(cols["fy"], cols["fx"]),
    }


def family_of(kind: str) -> str:
    return "tilt" if kind.startswith("tilt") else kind


def blaze2d_ids(capture_dir: Path) -> list[str]:
    """Design ids of this capture: ``summary.json``'s ``blaze2d_designs`` or the table."""
    summary = capture_dir / "summary.json"
    listed: list[str] = []
    if summary.is_file():
        try:
            blob = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blob = {}
        listed = _collect_blaze2d_ids(blob)
    return listed or list(BLAZE2D_FALLBACK_IDS)


def _collect_blaze2d_ids(blob, depth: int = 0) -> list[str]:
    """Find ``blaze2d_designs`` anywhere in the summary (top level or nested)."""
    if depth > 3 or not isinstance(blob, dict):
        return []
    for key, value in blob.items():
        if key == "blaze2d_designs" and isinstance(value, list):
            ids = []
            for entry in value:
                pid = entry.get("id") if isinstance(entry, dict) else entry
                if isinstance(pid, str) and pid:
                    ids.append(pid)
            if ids:
                return ids
        found = _collect_blaze2d_ids(value, depth + 1)
        if found:
            return found
    return []


# ---------------------------------------------------------------------------
# readable report
# ---------------------------------------------------------------------------
def render_readme(status: dict) -> str:
    b = status["blaze2d"]
    lines = [
        "# #247 axis-mapping result (auto-generated)",
        "",
        f"- capture: `{status['source']}` (frames dir)",
        f"- frozen: **{str(status['frozen']).lower()}**",
        f"- verdict: `{status['verdict']}`",
        f"- map written: `{status['map_path']}`" if status["frozen"] else "- map written: none",
        "",
        "## Freeze rule (pre-registered 2026-11-21b)",
        "",
        "| condition | value | limit | pass |",
        "|---|---|---|---|",
    ]
    for c in b.get("conditions", []):
        lines.append(
            f"| {c['name']} | {c['observed']} | {c['limit']} | "
            f"{'PASS' if c['pass'] else 'FAIL'} |"
        )
    lines += ["", "## Fitted A (CCD px per cycle/panel-pixel)", ""]
    if b.get("A_matrix"):
        a = b["A_matrix"]
        lines += [
            "```",
            f"A = [[{a[0][0]:+12.3f}, {a[0][1]:+12.3f}],   # dr per (fy, fx)",
            f"     [{a[1][0]:+12.3f}, {a[1][1]:+12.3f}]]   # dc per (fy, fx)",
            "```",
            "",
            f"- singular values: s_max={b['singular_values'][0]:.3f}, "
            f"s_min={b['singular_values'][1]:.3f}, "
            f"cond={b['condition_number']:.2f}",
            f"- rank guard |det A| / s_max^2 = {b['rank_guard']:.4f}",
            f"- column separation = {b['column_separation_deg']:.3f} deg",
        ]
    else:
        lines.append("_not fitted: the blaze2d family is absent from this capture._")
    lines += ["", "## blaze2d points", ""]
    if b.get("per_point"):
        lines += [
            "| id | fy (1/px) | fx (1/px) | measured (dr, dc) px | predicted (dr, dc) px "
            "| residual px | |r| px |",
            "|---|---|---|---|---|---|---|",
        ]
        for p in b["per_point"]:
            lines.append(
                f"| `{p['id']}` | {p['fy']:+.6f} | {p['fx']:+.6f} "
                f"| ({p['measured']['dr']:+.3f}, {p['measured']['dc']:+.3f}) "
                f"| ({p['predicted']['dr']:+.3f}, {p['predicted']['dc']:+.3f}) "
                f"| ({p['residual']['dr']:+.3f}, {p['residual']['dc']:+.3f}) "
                f"| {p['residual_norm_px']:.3f} |"
            )
    else:
        lines.append("_none._")
    lines += ["", "## Per-family 2x2 fits (auxiliary attribution, never used to freeze)", ""]
    if status.get("per_family"):
        lines += [
            "| family | n | fit RMS px | max rel err | vs reference column dirs fy / fx (deg) |",
            "|---|---|---|---|---|",
        ]
        for fam, f in status["per_family"].items():
            if f.get("error"):
                lines.append(f"| {fam} | - | - | - | {f['error']} |")
                continue
            lines.append(
                f"| {fam} | {f['n_points']} | {f['residual_rms_px']:.3f} "
                f"| {f['max_rel_err'] * 100:.3f} % "
                f"| {f['column_delta_vs_reference_deg']['fy']:+.3f} / "
                f"{f['column_delta_vs_reference_deg']['fx']:+.3f} |"
            )
    lines += ["", "## Legacy direction diagnostics (grating / tilt / ladder)", ""]
    for key, value in status["family_direction_comparisons_deg"].items():
        shown = "n/a" if value is None else f"{value:.2f} deg"
        lines.append(f"- `{key}`: {shown}")
    lines += [
        "",
        f"- ladder order-3/order-1: `{json.dumps(status['ladder_linearity'])}`",
        "",
        "---",
        "",
        "Generated by `scripts/slm/issue247_axis_map_solve_20261121.py`.",
        "Re-run with `--out-prefix <tag>` to keep a previous result intact.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit the #247 panel->CCD 2x2 map; freeze it only if the "
                    "pre-registered blaze2d rule passes."
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=DEFAULT_CAPTURE_DIR,
        help=f"capture directory holding frames/ and summary.json "
             f"(default: {DEFAULT_CAPTURE_DIR})",
    )
    parser.add_argument(
        "--out-prefix",
        default="",
        help="suffix for the written artifacts, e.g. 'selftest' -> "
             "axis_mapping_status_selftest.json (default: '', the recorded names)",
    )
    args = parser.parse_args(argv)

    capture_dir = Path(args.capture_dir)
    frames = capture_dir / "frames"
    if not frames.is_dir():
        print(f"!! no frames directory: {frames}")
        return 2
    suffix = f"_{args.out_prefix}" if args.out_prefix else ""
    out_status = capture_dir / f"axis_mapping_status{suffix}.json"
    out_readme = capture_dir / f"README_axis_mapping{suffix}.md"
    out_map = capture_dir / f"issue247_axis_map{suffix}.json"

    flat = load("flat", frames)
    r0, c0 = np.unravel_index(int(np.argmax(flat)), flat.shape)
    # sub-pixel 0th-order centroid from the flat primary peak
    core = find_spots(flat, min_dist=20)
    ref = (core[0]["row"], core[0]["col"])
    print(f"0th order: argmax=({r0},{c0}) centroid=({ref[0]:.2f},{ref[1]:.2f})")

    measured = {}
    missing: list[str] = []
    for pid, kind, kr, kc in PROBES:
        if not frame_files(pid, frames):
            missing.append(pid)
            continue
        p = principal(load(pid, frames), ref)
        if p is None:
            print(f"  !! no principal spot for {pid}")
            missing.append(pid)
            continue
        measured[pid] = {**p, "k_row": kr, "k_col": kc, "kind": kind}
        print(f"  {pid:16s} d=({p['dr']:+7.2f},{p['dc']:+7.2f}) px  "
              f"dist={p['dist']:7.2f} peak={p['peak']:7.1f}  angle={angle_deg(p['dr'], p['dc']):+7.1f} deg")

    def vec(pid):
        m = measured.get(pid)
        return None if m is None else (m["dr"], m["dc"])

    def sub(pid_a, pid_b):
        a, b = vec(pid_a), vec(pid_b)
        if a is None or b is None:
            return None
        return (a[0] - b[0], a[1] - b[1])

    def ang(a, b):
        if a is None or b is None:
            return None
        return angle_between(a, b)

    # ---- family direction comparison (the consistency test) ----
    comparisons = {
        "panel_x__tilt_x_vs_ladder_x": ang(
            sub("tilt_x_plus", "tilt_x_minus"), vec("ladder_x")
        ),
        "panel_y__tilt_y_vs_ladder_y": ang(
            sub("tilt_y_plus", "tilt_y_minus"), vec("ladder_y")
        ),
        "panel_x_vs_panel_y__tilt_x_vs_tilt_y": ang(
            sub("tilt_x_plus", "tilt_x_minus"), sub("tilt_y_plus", "tilt_y_minus")
        ),
        "panel_x__grating_x_vs_ladder_x": ang(vec("grating_x_p128"), vec("ladder_x")),
    }

    # ladder order-3 / order-1 linearity (ladder has orders 1 and 3 of the base:
    # period base/1 = 96 px -> order 1, period base/3 = 32 px -> order 3, so the
    # order-3 spot must sit at 3x the order-1 distance).  order1 = the brightest
    # nonzero spot; order3 = the spot nearest 3*order1.
    ladder_lin = {}
    for pid, axis in (("ladder_x", "x"), ("ladder_y", "y")):
        if pid not in measured:
            continue
        d1 = measured[pid]["dist"]
        cand = []
        for s in find_spots(load(pid, frames)):
            dr, dc = s["row"] - ref[0], s["col"] - ref[1]
            dist = float(np.hypot(dr, dc))
            if dist <= 1.5 * d1:
                continue
            cand.append(dist)
        if cand:
            d3 = min(cand, key=lambda d: abs(d - 3.0 * d1))
            ladder_lin[pid] = {
                "order1_dist_px": round(d1, 2),
                "order3_dist_px": round(d3, 2),
                "ratio_vs_expected_3.0": round(d3 / d1, 3),
            }
            print(f"  {pid}: order1={d1:.1f}px order3={d3:.1f}px ratio={d3/d1:.2f} (expect 3.00)")

    tol_angle = 2.0  # degrees: a single affine map puts same-panel-axis probes within ~0
    core_cmps = ("panel_x__tilt_x_vs_ladder_x", "panel_y__tilt_y_vs_ladder_y",
                 "panel_x__grating_x_vs_ladder_x")
    consistent = all(
        comparisons[k] is not None and comparisons[k] <= tol_angle for k in core_cmps
    )
    # panel axes must also be distinct (the measured rotation is ~88.5 deg)
    sep_legacy = comparisons["panel_x_vs_panel_y__tilt_x_vs_tilt_y"]
    axes_distinct = sep_legacy is not None and abs(sep_legacy - 88.544) <= 5.0
    legacy_conflict = not (consistent and axes_distinct)

    # ---- blaze2d: the 2x2 fit that the freeze rule tests ----
    known = {pid for pid, _, _, _ in PROBES}
    listed = blaze2d_ids(capture_dir)
    unknown = [pid for pid in listed if pid not in known]
    if unknown:
        print(f"  !! summary lists unknown blaze2d ids (ignored): {unknown}")
    blaze_points = [pid for pid in listed if pid in known and pid in measured]

    blaze = {"present": False, "ids": listed, "found": blaze_points, "per_point": []}
    failing: list[str] = []
    conditions: list[dict] = []
    if len(blaze_points) < 4:
        print("\nblaze2d family absent — skipped "
              f"(found {len(blaze_points)}/{len(listed)} 2-D ramp patterns)")
        print("  the pre-registered freeze rule needs all four; the legacy "
              "ambiguous verdict is reproduced unchanged")
        blaze["skip_reason"] = f"{len(blaze_points)}/{len(listed)} blaze2d patterns found"
    else:
        points = [
            (measured[pid]["k_row"], measured[pid]["k_col"],
             measured[pid]["dr"], measured[pid]["dc"])
            for pid in blaze_points
        ]
        a, pred, residual = fit_affine(points)
        met = affine_metrics(points, a, residual)
        res_norm = met["residual_norm_px"]

        print("\n[blaze2d 2x2 least squares]  model (dr,dc) = A @ (fy,fx); "
              "8 equations / 4 unknowns")
        print(f"  {'id':38s} {'measured (dr,dc)':>21s} {'predicted (dr,dc)':>21s} "
              f"{'residual':>19s} {'|r| px':>7s}")
        per_point = []
        for i, pid in enumerate(blaze_points):
            fy, fx, dr, dc = points[i]
            rr = residual[i]
            pdr, pdc = float(pred[i][0]), float(pred[i][1])
            print(f"  {pid:38s} ({dr:+9.3f},{dc:+9.3f}) ({pdr:+9.3f},{pdc:+9.3f}) "
                  f"({rr[0]:+8.3f},{rr[1]:+8.3f}) {res_norm[i]:7.3f}")
            per_point.append({
                "id": pid,
                "fy": float(fy),
                "fx": float(fx),
                "measured": {"dr": round(float(dr), 4), "dc": round(float(dc), 4)},
                "predicted": {"dr": round(pdr, 4), "dc": round(pdc, 4)},
                "residual": {"dr": round(float(rr[0]), 4), "dc": round(float(rr[1]), 4)},
                "residual_norm_px": round(float(res_norm[i]), 4),
                "rel_err": round(float(res_norm[i] / met["measured_norm_px"][i]), 6),
            })

        s_max, s_min = met["singular_values"]
        print("  A (CCD px per cycle/panel-pixel), rows = CCD (dr,dc), "
              "cols = panel (fy,fx):")
        print(f"    [[{a[0, 0]:+12.3f}, {a[0, 1]:+12.3f}],")
        print(f"     [{a[1, 0]:+12.3f}, {a[1, 1]:+12.3f}]]")
        print(f"  singular values: s_max={s_max:.3f} s_min={s_min:.3f} "
              f"cond={met['condition_number']:.3f}")
        print(f"  rank guard |det A| / s_max^2 = {met['rank_guard']:.4f}")
        print(f"  residual RMS = {met['residual_rms_px']:.3f} px   "
              f"max point residual = {met['max_point_residual_px']:.3f} px   "
              f"max relative error = {met['max_rel_err'] * 100:.3f} %")
        print(f"  column directions: fy at {met['column_angles_deg']['fy']:+.3f} deg, "
              f"fx at {met['column_angles_deg']['fx']:+.3f} deg, "
              f"separation {met['column_separation_deg']:.3f} deg")

        rules = FREEZE_RULES
        sep_deg = met["column_separation_deg"]
        conditions = [
            {
                "name": "blaze2d_residual_rms",
                "pass": met["residual_rms_px"] <= rules["residual_rms_max_px"],
                "observed": f"{met['residual_rms_px']:.4f} px",
                "limit": f"<= {rules['residual_rms_max_px']:.1f} px",
                "value": round(met["residual_rms_px"], 6),
            },
            {
                "name": "blaze2d_max_rel_err",
                "pass": met["max_rel_err"] <= rules["max_rel_err_max"],
                "observed": f"{met['max_rel_err'] * 100:.4f} %",
                "limit": f"<= {rules['max_rel_err_max'] * 100:.1f} %",
                "value": round(met["max_rel_err"], 8),
            },
            {
                "name": "blaze2d_axis_separation",
                "pass": abs(sep_deg - rules["axis_separation_expected_deg"])
                <= rules["axis_separation_tol_deg"],
                "observed": f"{sep_deg:.4f} deg",
                "limit": (f"{rules['axis_separation_expected_deg']:.3f} "
                          f"+- {rules['axis_separation_tol_deg']:.1f} deg"),
                "value": round(sep_deg, 6),
            },
            {
                "name": "blaze2d_rank_guard",
                "pass": met["rank_guard"] >= rules["rank_guard_min"],
                "observed": f"{met['rank_guard']:.4f}",
                "limit": f">= {rules['rank_guard_min']:.2f}",
                "value": round(met["rank_guard"], 6),
            },
        ]
        failing = [c["name"] for c in conditions if not c["pass"]]
        print("  freeze rule:")
        for c in conditions:
            print(f"    {c['name']:28s} {'PASS' if c['pass'] else 'FAIL'}  "
                  f"observed {c['observed']}  limit {c['limit']}")

        blaze = {
            "present": True,
            "ids": listed,
            "found": blaze_points,
            "model": "(dr, dc) = A @ (fy, fx); fy = sign_y/period_y_px, fx = sign_x/period_x_px",
            "units": "CCD px per (cycle / panel-pixel)",
            "A_matrix": [[round(float(v), 4) for v in row] for row in a],
            "A_columns": {
                "fy": {"dr": round(float(a[0, 0]), 4), "dc": round(float(a[1, 0]), 4),
                       "angle_deg": round(met["column_angles_deg"]["fy"], 4)},
                "fx": {"dr": round(float(a[0, 1]), 4), "dc": round(float(a[1, 1]), 4),
                       "angle_deg": round(met["column_angles_deg"]["fx"], 4)},
            },
            "column_separation_deg": round(met["column_separation_deg"], 4),
            "singular_values": [round(s_max, 4), round(s_min, 4)],
            "condition_number": round(met["condition_number"], 4),
            "rank_guard": round(met["rank_guard"], 6),
            "residual_rms_px": round(met["residual_rms_px"], 4),
            "max_point_residual_px": round(met["max_point_residual_px"], 4),
            "max_rel_err": round(met["max_rel_err"], 8),
            "per_point": per_point,
            "conditions": conditions,
            "failing_conditions": failing,
        }

        if not failing and legacy_conflict:
            print("  !! note: the legacy grating/tilt/ladder diagnostics still conflict;")
            print("     the freeze rests on the blaze2d family alone, per the "
                  "pre-registered rule.")

    # ---- per-family 2x2 fits (attribution only) ----
    fam_fits: dict[str, dict] = {}
    fam_A: dict[str, np.ndarray] = {}
    for fam in FAMILIES:
        pts = [
            (m["k_row"], m["k_col"], m["dr"], m["dc"])
            for m in measured.values()
            if family_of(m["kind"]) == fam
        ]
        if len(pts) < 2:
            fam_fits[fam] = {"n_points": len(pts), "error": "fewer than 2 patterns"}
            continue
        x = np.array([[fy, fx] for fy, fx, _, _ in pts])
        if np.linalg.matrix_rank(x) < 2:
            fam_fits[fam] = {
                "n_points": len(pts),
                "error": "probe frequencies collinear in (fy, fx); A not identifiable",
            }
            continue
        a_f, pred_f, resid_f = fit_affine(pts)
        met_f = affine_metrics(pts, a_f, resid_f)
        fam_A[fam] = a_f
        fam_fits[fam] = {
            "n_points": len(pts),
            "A_matrix": [[round(float(v), 4) for v in row] for row in a_f],
            "column_separation_deg": round(met_f["column_separation_deg"], 4),
            "residual_rms_px": round(met_f["residual_rms_px"], 4),
            "max_rel_err": round(met_f["max_rel_err"], 8),
            "rank_guard": round(met_f["rank_guard"], 6),
            "exactly_determined": len(pts) == 2,
            "note": ("exactly determined (4 equations / 4 unknowns): the residual "
                     "is structural, not evidence") if len(pts) == 2 else "",
        }

    ref_family = "blaze2d" if "blaze2d" in fam_A else ("grating" if "grating" in fam_A else None)
    if ref_family is not None:
        ref_cols = {
            "fy": (float(fam_A[ref_family][0, 0]), float(fam_A[ref_family][1, 0])),
            "fx": (float(fam_A[ref_family][0, 1]), float(fam_A[ref_family][1, 1])),
        }
        for fam, entry in fam_fits.items():
            if "A_matrix" not in entry:
                continue
            a_f = fam_A[fam]
            cols = {"fy": (float(a_f[0, 0]), float(a_f[1, 0])),
                    "fx": (float(a_f[0, 1]), float(a_f[1, 1]))}
            signed = {}
            for key in ("fy", "fx"):
                d = angle_between(cols[key], ref_cols[key])
                cross = cols[key][0] * ref_cols[key][1] - cols[key][1] * ref_cols[key][0]
                signed[key] = round(d if cross >= 0 else -d, 3)
            entry["column_delta_vs_reference_deg"] = signed
            entry["reference_family"] = ref_family
        print(f"\n[per-family 2x2 fits]  auxiliary attribution only; never used to freeze"
              f"  (reference columns: {ref_family})")
        for fam in FAMILIES:
            entry = fam_fits[fam]
            if "A_matrix" not in entry:
                print(f"  {fam:8s} n={entry['n_points']}  skipped: {entry['error']}")
                continue
            d = entry["column_delta_vs_reference_deg"]
            note = "  [exactly determined]" if entry["exactly_determined"] else ""
            print(f"  {fam:8s} n={entry['n_points']}  "
                  f"rms={entry['residual_rms_px']:.3f}px  "
                  f"max_rel={entry['max_rel_err'] * 100:.3f}%  "
                  f"columns vs ref: fy {d['fy']:+7.3f} deg, fx {d['fx']:+7.3f} deg{note}")
            print(f"           A=[[{entry['A_matrix'][0][0]:+10.2f}, "
                  f"{entry['A_matrix'][0][1]:+10.2f}], "
                  f"[{entry['A_matrix'][1][0]:+10.2f}, "
                  f"{entry['A_matrix'][1][1]:+10.2f}]]")

    # ---- verdict ----
    frozen = bool(blaze["present"] and not failing)
    if frozen:
        verdict = "mapping_frozen"
    elif blaze["present"]:
        verdict = "mapping_still_ambiguous: " + ", ".join(failing)
    else:
        verdict = "mapping_still_ambiguous"

    if frozen:
        finding = (
            "the four 2-D diagonal blaze probes are jointly consistent with one "
            "reversible 2x2 panel-frequency -> CCD-pixel map, and the two panel "
            "axes stay linearly independent (rank guard)."
        )
        conclusion = (
            "FROZEN by the pre-registered 2026-11-21b rule. The map in "
            "issue247_axis_map.json is the fitted A; do not refit it on the same "
            "frames and call the result an independent confirmation."
        )
    elif blaze["present"]:
        finding = (
            "the 2-D diagonal blaze family does NOT satisfy the pre-registered rule: "
            + ", ".join(failing)
            + "."
        )
        conclusion = (
            "NOT frozen: the blaze2d 2x2 fit fails at least one pre-registered "
            "condition, so no map is written and #247 stays open."
        )
    else:
        finding = (
            "grating/tilt probes place panel-x diffraction on CCD COLS while the "
            "ladder places it on CCD ROWS; panel-x and panel-y BOTH image onto "
            "CCD COLS, so a single linear panel->CCD map cannot explain the data. "
            "Ladder order-3/order-1 is also far from the expected 3.0."
        )
        conclusion = (
            "NOT frozen. The A1 capture is internally inconsistent for one affine "
            "map; #247 stays open. Re-capture is needed with one probe family only "
            "(or a fix to whichever builder the conflict traces to) before a map "
            "can be frozen."
        )

    status = {
        "schema": "issue247-axis-mapping-status/v2",
        "frozen": frozen,
        "verdict": verdict,
        "source": as_repo_str(frames),
        "freeze_rule": {
            "pre_registration": "docs/dev/hw_window_20261121b_issue247_prereg.md",
            "applies_to": "blaze2d family only",
            **FREEZE_RULES,
        },
        "zero_order_reference_px": {"row": round(ref[0], 3), "col": round(ref[1], 3),
                                    "off_from_geometric_centre": {
                                        "dr": round(ref[0] - flat.shape[0] / 2, 2),
                                        "dc": round(ref[1] - flat.shape[1] / 2, 2)}},
        "angle_convention": "deg from +CCD-col (dc) toward +CCD-row (dr)",
        "measured_principal_spots": {
            pid: {"dr": round(m["dr"], 2), "dc": round(m["dc"], 2),
                  "dist": round(m["dist"], 2), "peak": round(m["peak"], 1),
                  "k_row_per_px": m["k_row"], "k_col_per_px": m["k_col"],
                  "angle_deg": round(angle_deg(m["dr"], m["dc"]), 2)}
            for pid, m in measured.items()},
        "family_direction_comparisons_deg": {
            k: (None if v is None else round(v, 2)) for k, v in comparisons.items()},
        "ladder_linearity": ladder_lin,
        "consistency": {
            "same_panel_axis_agree_deg_le": tol_angle,
            "panel_x_consistent": (comparisons["panel_x__tilt_x_vs_ladder_x"] is not None
                                   and comparisons["panel_x__tilt_x_vs_ladder_x"] <= tol_angle),
            "panel_y_consistent": (comparisons["panel_y__tilt_y_vs_ladder_y"] is not None
                                   and comparisons["panel_y__tilt_y_vs_ladder_y"] <= tol_angle),
            "panel_axes_distinct_expected_deg": 88.544,
            "panel_axes_distinct_deg": (None if sep_legacy is None else round(sep_legacy, 2)),
        },
        "legacy_conflict_observed": legacy_conflict,
        "blaze2d": blaze,
        "per_family": fam_fits,
        "missing_patterns": missing,
        "artifact_prefix": args.out_prefix,
        "finding": finding,
        "conclusion": conclusion,
    }
    if frozen:
        status["map_path"] = as_repo_str(out_map)
    else:
        status["map_path"] = None

    print("\n=== consistency ===")
    for k, v in comparisons.items():
        print(f"  {k}: " + ("n/a" if v is None else f"{v:.2f} deg"))
    print(f"  consistent={consistent} axes_distinct={axes_distinct} "
          f"(legacy grating/tilt/ladder diagnostics)")
    print(f"verdict: {verdict}  frozen={frozen}")

    out_status.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_status}")

    if frozen:
        out_map.write_text(json.dumps({
            "schema": "issue247-axis-map/v1",
            "frozen": True,
            "verdict": verdict,
            "A_ccd_px_per_panel_freq": status["blaze2d"]["A_matrix"],
            "A_columns": status["blaze2d"]["A_columns"],
            "model": status["blaze2d"]["model"],
            "units": status["blaze2d"]["units"],
            "source_frames": as_repo_str(frames),
            "freeze_rule": status["freeze_rule"],
            "blaze2d_points": status["blaze2d"]["per_point"],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out_map}")
    else:
        print(f"no map written (frozen=false): {out_map}")

    if suffix or not out_readme.exists():
        out_readme.write_text(render_readme(status), encoding="utf-8")
        print(f"wrote {out_readme}")
    else:
        print(f"kept existing {out_readme} (not overwritten; use --out-prefix for a copy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
