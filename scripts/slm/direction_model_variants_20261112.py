"""Consolidated per-mode Jacobian-direction diagnosis across forward-model variants.

Offline only (issue #234 / #235 / #237): this script never opens the SLM, the
camera or the laser.  It replays the archived issue234 single-mode screens
(±0.10 / ±0.05 / ±0.02 rad, Z5–Z11, two passes) and compares, per mode and per
model variant, the signed direction cosine between the model C−U differential
and the measured C−U differential.

Variants (all from ``defocus_forward_20261107_focus_r2_fit/forward_fit.json``):

* ``M0_fit``   -- M0: defocus_diopter = 0 (the "analytic" no-defocus baseline).
* ``M1_fit``   -- M1_defocus: the standard slm_phase fitted checkpoint model.
* ``M3_fit``   -- M3_illumination: defocus + fitted illumination envelope.
* ``M1_bg0``   -- M1_defocus with background forced to 0.
* ``M1_bg10``  -- M1_defocus with background forced to +10 counts/ms.
* ``M3_bg0``   -- M3_illumination with background forced to 0.

Because the direction metric is an affine-invariant Pearson on the C−U
differential, background and gain are *expected* to cancel exactly; the bg
variants exist to demonstrate that numerically rather than assume it.

Guard: the ``M1_fit`` amp 0.05 arm must reproduce the archived
``direction_screen_amp005_M1_defocus_signm1_model.json`` per-mode correlation
exactly (bit-for-bit), proving the measurement口径 is unchanged.

Outputs ``results.json`` under ``data/debug/slm_transfer/direction_variants_20261112/``
and prints a per-variant summary table.  With ``--figures-only`` (or after a full
run) also renders the report figures into
``report/shape_trials/direction_diagnosis_20261112/figures/`` from the JSON
artifacts (offline, no recompute).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

#: repo-root bootstrap so ``scripts._common`` and ``ao_shaping`` resolve when
#: this file is run as ``python scripts/slm/direction_model_variants_20261112.py``
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward, FrozenDefocusParameters,
)
from ao_shaping.tools.slm.transfer.jacobian_guard import direction_agreement

FIT = Path("data/debug/slm_transfer/defocus_forward_20261107_focus_r2_fit/forward_fit.json")
OUT_DIR = Path("data/debug/slm_transfer/direction_variants_20261112/")
#: report-side figure output (tracked in git; see report/AGENTS.md asset rules)
FIG_DIR = Path("report/shape_trials/direction_diagnosis_20261112/figures")
CROP = 128
MODES = list(range(5, 12))
AMPS = [0.02, 0.05, 0.10]
# run dir per amp (the amp 0.10 ladder lives in the "single_mode" directory)
RUN_DIRS = {
    0.02: "data/debug/slm_transfer/issue234_amp02_screen_20261107",
    0.05: "data/debug/slm_transfer/issue234_amp05_screen_20261107",
    0.10: "data/debug/slm_transfer/issue234_single_mode_screen_20261107",
}
# archive guard: amp 0.05 M1_defocus sign -1, origin=model
GUARD = Path(
    "data/debug/slm_transfer/issue234_amp05_screen_20261107/"
    "direction_screen_amp005_M1_defocus_signm1_model.json")

VARIANTS = [
    ("M0_fit", "M0", None),
    ("M1_fit", "M1_defocus", None),
    ("M3_fit", "M3_illumination", None),
    ("M1_bg0", "M1_defocus", 0.0),
    ("M1_bg10", "M1_defocus", 10.0),
    ("M3_bg0", "M3_illumination", 0.0),
]


def load_frame(run, run_pass, row):
    d = np.load(run / run_pass / row["file"])
    raw = d["raw_frames"].astype(np.float64).mean(axis=0)
    rate = raw / float(row.get("exposure_ms") or 1.0)
    return d["model_phase"].astype(np.float32), rate


def crop(img, ctr):
    x0 = int(round(ctr[0] - CROP / 2))
    y0 = int(round(ctr[1] - CROP / 2))
    return img[y0:y0 + CROP, x0:x0 + CROP]


def evaluate(run_dir, amp, model_key, bg_override):
    run = Path(run_dir)
    fit = json.loads(FIT.read_text())
    model = fit["models"][model_key]
    ramp = fit["ramp_matrix"]
    origin = np.asarray(model["origin"], dtype=np.float64)
    bg = model["background"] if bg_override is None else bg_override
    params = FrozenDefocusParameters(
        defocus_diopter=model["defocus_diopter"],
        origin_xy=(float(origin[0]), float(origin[1])),
        amplitude_center_xy=tuple(model["amplitude_center"]),
        amplitude_width_xy=tuple(model["amplitude_width"]),
        residual_noll_radians={},
        gain=model["gain"],
        background=bg,
        sigma_ccd_px=0.0,
        ramp_matrix=((ramp[0][0], ramp[0][1]), (ramp[1][0], ramp[1][1])),
        phase_sign=-1.0,
        phase_sign_scope="command",
    )
    yy, xx = np.mgrid[:CROP, :CROP].astype(np.float64)
    camera_xy = np.stack([xx + origin[0] - CROP / 2, yy + origin[1] - CROP / 2], axis=-1)
    forward = FrozenDefocusForward(params, camera_xy, device="cpu")
    amp_tag = f"{amp:.2f}"
    results = []
    for run_pass in ["pass1", "pass2"]:
        manifest = json.loads((run / run_pass / "manifest.json").read_text())
        for mode in MODES:
            lp = [r for r in manifest if f"_Z{mode}_+{amp_tag}" in r["label"]]
            lm = [r for r in manifest if f"_Z{mode}_-{amp_tag}" in r["label"]]
            if not lp or not lm:
                print(f"{run_pass} Z{mode:2d}  MISSING labels")
                continue
            pm, rm = load_frame(run, run_pass, lm[0])
            pp, rp = load_frame(run, run_pass, lp[0])
            with torch.no_grad():
                pred_m = forward.predict(torch.as_tensor(pm)).cpu().numpy()[0]
                pred_p = forward.predict(torch.as_tensor(pp)).cpu().numpy()[0]
            agr = direction_agreement(pred_m, pred_p, crop(rm, origin), crop(rp, origin))
            results.append({
                "pass": run_pass, "mode": mode, "corr": agr.correlation,
                "verdict": agr.verdict, "rel_l2": agr.relative_l2,
                "predicted_rms": agr.predicted_rms, "measured_rms": agr.measured_rms,
            })
    return results, params


def stats(results):
    corrs = np.asarray([r["corr"] for r in results], dtype=np.float64)
    finite = corrs[np.isfinite(corrs)]
    return {
        "n": int(len(results)),
        "n_finite": int(finite.size),
        "median": float(np.median(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
        "std_ddof1": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
        "n_ge_0p5": int(np.sum(finite >= 0.5)),
        "n_ge_0p5_over": f"{int(np.sum(finite >= 0.5))}/{int(finite.size)}",
    }


def _variant_short(name: str) -> str:
    """Figure-friendly variant label (drops the trailing ``_fit``/``_bg0`` suffix noise)."""
    return {
        "M0_fit": "M0 (analytic)",
        "M1_fit": "M1 (defocus)",
        "M3_fit": "M3 (defocus+env)",
        "M1_bg0": "M1 bg=0",
        "M1_bg10": "M1 bg=10",
        "M3_bg0": "M3 bg=0",
    }.get(name, name)


def write_figures(
    out_dir: Path = OUT_DIR,
    fig_dir: Path = FIG_DIR,
) -> list[Path]:
    """Render the direction-diagnosis figures from the JSON artifacts.

    Offline, reads only ``results.json`` / ``attribution.json``.  Produces:
    fig1 per-amp heatmap (variant x mode), fig2 median-vs-amp grouped bars,
    fig3 attribution-arm bars.  All use the repo CJK font helper.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from scripts._common.plot_style import setup_cjk_font

    setup_cjk_font()
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads((out_dir / "results.json").read_text())
    attribution = json.loads((out_dir / "attribution.json").read_text())
    gate = 0.50
    paths: list[Path] = []
    variants = list(results["variants"])
    amps = list(results["amp"]) if isinstance(results["amp"], list) else [results["amp"]]

    def _per_mode(v: dict) -> dict[int, float]:
        return {int(k): float(x) for k, x in v.get("per_mode_median", {}).items()}

    def _pass_count(v: dict) -> tuple[int, int]:
        finite = [r["corr"] for r in v["results"] if np.isfinite(r["corr"])]
        return int(sum(c >= gate for c in finite)), int(len(finite))

    # ---- fig1: heatmap per amp ----------------------------------------------
    fig, axes = plt.subplots(1, len(amps), figsize=(6.2 * len(amps), 4.6),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, amp in zip(axes, amps):
        entries = [e for e in results["variants"]
                   if abs(float(e["amp"]) - float(amp)) < 1e-9]
        # align rows to VARIANTS declaration order
        entries.sort(key=lambda e: [n for n, *_ in VARIANTS].index(e["variant"]))
        mat = np.full((len(entries), len(MODES)), np.nan)
        for i, entry in enumerate(entries):
            for j, m in enumerate(MODES):
                mat[i, j] = _per_mode(entry).get(m, np.nan)
        norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=1.0)
        im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(len(MODES)))
        ax.set_xticklabels([f"Z{m}" for m in MODES])
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels([_variant_short(e["variant"]) for e in entries])
        ax.set_title(f"amp {float(amp):.2f} rad  (median 方向相关)")
        for i in range(len(entries)):
            for j in range(len(MODES)):
                val = mat[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold" if val >= gate else "normal",
                            color="black" if val >= gate else "0.25")
        fig.colorbar(im, ax=ax, shrink=0.8, label="direction Pearson")
    fig.suptitle("可微物理预测模型 Jacobian 方向矩阵（行=变体，列=Zernike 模式）", y=1.02)
    p = fig_dir / "fig1_variant_mode_heatmap_20261112.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # ---- fig2: median vs amp (M0/M1/M3) -------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    core = ["M0_fit", "M1_fit", "M3_fit"]
    colors = ["#7f8c8d", "#3498db", "#27ae60"]
    width = 0.26
    for k, name in enumerate(core):
        meds, passf = [], []
        for amp in amps:
            entry = next((v for v in results["variants"] if v["variant"] == name
                          and abs(float(v["amp"]) - float(amp)) < 1e-9), None)
            meds.append(float(entry["stats"]["median"]) if entry else np.nan)
            passf.append(_pass_count(entry) if entry else (0, 0))
        xs = np.arange(len(amps)) + (k - 1) * width
        bars = ax.bar(xs, meds, width, label=_variant_short(name), color=colors[k])
        for x, (m, (npass, ntot)) in zip(xs, zip(meds, passf)):
            if np.isfinite(m):
                ax.text(x, m + 0.03, f"{m:.2f}\n{npass}/{ntot}", ha="center",
                        va="bottom", fontsize=7)
    ax.axhline(gate, color="crimson", ls="--", lw=1.4)
    ax.text(len(amps) - 0.5, gate + 0.02, "方向门禁 0.50", color="crimson",
            fontsize=8, ha="right")
    ax.set_xticks(range(len(amps)))
    ax.set_xticklabels([f"{float(a):.2f} rad" for a in amps])
    ax.set_ylabel("median 方向相关")
    ax.set_ylim(-0.1, 1.0)
    ax.set_title("median 方向相关 vs 幅值（每个变体标注 ≥0.50 单对占比）")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    p = fig_dir / "fig2_median_vs_amp_20261112.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # ---- fig3: attribution arms ---------------------------------------------
    arms = attribution.get("arms", {})
    order = ["M3_baseline", "M1_baseline", "M3_env_on_M1", "M1_env_on_M3",
             "M3_origin_on_M1", "M1_origin_on_M3", "M3_defocus_on_M1"]
    labels, meds, passes = [], [], []
    for name in order:
        a = arms.get(name)
        if not a:
            continue
        labels.append(name)
        meds.append(float(a["median"]))
        finite = [x for x in a["per_mode"].values() if np.isfinite(x)]
        passes.append(f"{int(sum(x >= gate for x in finite))}/{len(finite)}")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ypos = np.arange(len(labels))[::-1]
    colors2 = ["#27ae60" if "M3" in n else "#95a5a6" for n in labels]
    ax.barh(ypos, meds, color=colors2, height=0.62)
    for y, (m, ps) in zip(ypos, zip(meds, passes)):
        ax.text(m + 0.02, y, f"{m:.2f} ({ps})", va="center", fontsize=8)
    ax.axvline(gate, color="crimson", ls="--", lw=1.4)
    ax.text(gate + 0.02, len(labels) - 0.4, "门禁 0.50", color="crimson", fontsize=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-0.2, 1.0)
    ax.set_xlabel("median 方向相关（amp 0.05，标注 ≥0.50 单对占比）")
    ax.set_title("M3 方向通过归因：替换单个变体无法复现基线通过")
    fig.tight_layout()
    p = fig_dir / "fig3_attribution_arms_20261112.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    print(f"figures written: {', '.join(str(x) for x in paths)}")
    return paths


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fit = json.loads(FIT.read_text())

    # ---- guard: reproduce the archived amp 0.05 M1_fit per-mode corr ---------
    guard_results, _ = evaluate(RUN_DIRS[0.05], 0.05, "M1_defocus", None)
    archived = json.loads(GUARD.read_text())["results"]
    by_key = {(r["pass"], r["mode"]): r["corr"] for r in archived}
    guard = {"n": len(guard_results), "exact_match": True}
    max_delta = 0.0
    for r in guard_results:
        a = by_key.get((r["pass"], r["mode"]))
        if a is None:
            guard["exact_match"] = False
            guard.setdefault("missing", []).append((r["pass"], r["mode"]))
            continue
        max_delta = max(max_delta, abs(a - r["corr"]))
    guard["max_abs_delta"] = max_delta
    guard["median"] = stats(guard_results)["median"]
    print(f"GUARD: n={guard['n']} median={guard['median']:.6f} "
          f"exact={guard['exact_match']} max_delta={max_delta:.3e}")

    # ---- variant matrix -----------------------------------------------------
    all_out = {"schema": "direction_model_variants/1", "amp": AMPS,
               "variants": [], "guard": guard}
    for vname, model_key, bg_override in VARIANTS:
        for amp in AMPS:
            results, params = evaluate(RUN_DIRS[amp], amp, model_key, bg_override)
            st = stats(results)
            per_mode = {}
            for r in results:
                per_mode.setdefault(r["mode"], []).append(r["corr"])
            per_mode_median = {str(m): float(np.median(v)) for m, v in per_mode.items()}
            entry = {
                "variant": vname, "model": model_key, "amp": amp,
                "background_override": bg_override, "stats": st,
                "per_mode_median": per_mode_median, "results": results,
            }
            all_out["variants"].append(entry)
            print(f"{vname:8s} amp={amp:.2f}  n={st['n']}  median={st['median']:+.4f}  "
                  f"mean={st['mean']:+.4f}  >=0.5={st['n_ge_0p5_over']}")

    (OUT_DIR / "results.json").write_text(json.dumps(all_out, indent=1))
    print("wrote", OUT_DIR / "results.json")


if __name__ == "__main__":
    if "--figures-only" in sys.argv[1:]:
        write_figures()
    else:
        main()
