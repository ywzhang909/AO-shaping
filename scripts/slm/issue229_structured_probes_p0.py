"""Issue #229 P0 offline (twin) experiment — structured vs random single-mode
probe *identifiability*, and the new zernike_block fit tier.

Device-free. Uses the repo's ``FrozenDefocusForward`` 2f twin with an
*injected* 7-mode Zernike residual standing in for the unknown physical state.
No SLM/CCD/laser is touched; the real paired-CCD comparison is the separate
hardware plan in the companion report.

Why identifiability (not raw recovery) is the metric
----------------------------------------------------
The twin is (to first order) linear, so the injected offset ``c*`` lives in the
*base* frame; the incremental response of a probe set is ``J @ d`` where ``d``
is the probe's own coefficients and ``J`` the per-mode Jacobian. That
incremental response is design-independent once you condition on ``d`` — which
is exactly issue #229's null result (random and structured recover the same
NRMSE). What structured single-mode probes *do* buy is **per-mode
attributability**: probe one mode at a time so each Jacobian row is measured in
isolation, giving a well-conditioned, low-cross-leakage design. That is what the
identifiability gate (``zernike_block.identifiability_gate``) consumes to pick
which modes the new ``zernike_block`` tier may release.

So this experiment measures, for identical frame budgets:
  * the modal response-design condition number,
  * per-mode peak-to-leakage (isolated response vs cross-leakage),
  * the per-mode identifiability rate (fraction passing the gate),
and feeds the gate to show which modes the zernike_block tier releases.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataclasses import replace

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusForward, FrozenDefocusParameters,
)
from ao_shaping.tools.slm.transfer.structured_probes import (
    AMPLITUDE_LADDER, DEFAULT_MODES, structured_phase, structured_probe_plan,
)
from ao_shaping.tools.slm.transfer.zernike_block import (
    BLOCK_L2_CAP_RAD, identifiability_gate,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "data" / "debug" / "issue229_structured_probes"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEED = 2291104
GRID = 32
CAMERA = 256
FFT = 256
MODES = list(DEFAULT_MODES)          # [5,6,7,8,9,10,11]
AMPS = AMPLITUDE_LADDER              # (0.02, 0.05, 0.10)
A_REF = 0.10
INJECTED = {5: 0.12, 6: -0.08, 7: 0.15, 8: 0.02, 9: -0.10, 10: 0.07, 11: -0.05}
PEAK_LEAKAGE_FLOOR = 2.0             # a mode is identifiable if its isolated
                                     # response is >=2x its cross-leakage


def make_params(residual: dict[int, float]) -> FrozenDefocusParameters:
    return FrozenDefocusParameters(
        defocus_diopter=0.025, origin_xy=(128., 128.),
        amplitude_center_xy=(960., 600.), amplitude_width_xy=(300., 260.),
        residual_noll_radians={j: 0.0 for j in range(5, 12)},
        gain=1.0, background=0.0, sigma_ccd_px=0.0,
        ramp_matrix=((0., 5000.), (-5000., 0.)),
        grid_size=GRID, fft_size=FFT, pixel_samples=1)


def build_twin(residual: dict[int, float]) -> FrozenDefocusForward:
    truth = replace(make_params(residual), residual_noll_radians=dict(residual))
    yy, xx = np.mgrid[:CAMERA, :CAMERA]
    xy = np.stack((xx, yy), axis=-1)
    return FrozenDefocusForward(truth, xy, device="cpu")


def respond(fwd, phase) -> np.ndarray:
    return fwd.predict(torch.as_tensor(np.asarray(phase), dtype=torch.float32)).detach().numpy()[0]


def modal_design(probes_design: np.ndarray, J: np.ndarray):
    """Project a probe set onto the modal Jacobian basis.

    ``probes_design`` (n x M): the probe's own coefficients in the modal basis.
    ``J`` (M, P): per-mode unit-amplitude response (flattened).
    Returns the modal response design D = probes_design @ J  (n x P) — the
    (first-order) response each probe excites in the modal basis.
    """
    return probes_design @ J


def design_quality(D: np.ndarray) -> dict:
    """Identifiability of the per-mode estimator from the *design-space* Gram.

    ``D`` (n x M) is the probe design in the modal coefficient space. The
    per-mode response is ``y = D @ J.T`` (J fixed), so the identifiability of
    the mode weights is set by the condition of the design Gram ``D.T @ D``
    (M x M). A diagonal, balanced Gram (structured: one mode per probe) is
    well-conditioned; a mixed design (random: all modes per probe) couples the
    modes and is worse conditioned. This is the *design* metric (correct
    space), not the rank-limited image-space SVD.
    """
    gram = D.T @ D
    gram = 0.5 * (gram + gram.T)
    s = np.linalg.svd(gram, compute_uv=False)
    cond = float(s[0] / max(s[-1], 1e-12))
    # per-mode estimation variance (up to sigma^2): diagonal of the inverse
    # Gram. Uniform + low for a balanced structured design.
    try:
        inv = np.linalg.inv(gram)
        var = np.abs(np.diag(inv))
    except np.linalg.LinAlgError:
        var = np.full(len(MODES), float("inf"))
    return dict(cond=cond, per_mode_var=var, gram=gram)


def per_mode_peak_leakage(struct_probes_design: np.ndarray, J: np.ndarray) -> dict[int, float]:
    """Structured: probe mode i alone -> its response is J[:, i]; leakage is
    the sum of the other modes' responses that correlate with it."""
    # For a structured single-mode design, the per-mode design column is exactly
    # J[:, i] (isolated). Cross-leakage is the model's own off-diagonal coupling
    # captured by how much J[:, i] projects onto the other J[:, j].
    P = J.shape[1]
    out = {}
    for i, m in enumerate(MODES):
        v = J[i]                                   # (P,)
        nv = np.linalg.norm(v)
        peak = nv
        leak_sq = 0.0
        for j in range(len(MODES)):
            if j == i:
                continue
            proj = np.dot(v, J[j]) / (np.linalg.norm(J[j]) + 1e-12)
            leak_sq += proj ** 2
        leak = float(np.sqrt(leak_sq))
        out[m] = peak / (leak + 1e-12)
    return out


def main() -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    fwd = build_twin(INJECTED)                       # the "physical" twin
    base = respond(fwd, np.zeros((GRID, GRID)))

    # per-mode Jacobian (model, residual=0) — the linear response basis
    fwd0 = build_twin({j: 0.0 for j in MODES})
    base0 = respond(fwd0, np.zeros((GRID, GRID)))
    J = np.stack([(respond(fwd0, structured_phase(dict(mode=m, amplitude=1.0, sign=1), GRID))
                   - base0).ravel() for m in MODES])   # (M, P)

    # ------------------------------------------------------------------
    # Probe designs (identical frame budget)
    # ------------------------------------------------------------------
    plan = structured_probe_plan(modes=tuple(MODES), amplitudes=AMPS, repeats=1)
    # use the FULL structured design (all 7 modes present) so the design Gram
    # is full rank; the random arm gets the same number of probes for a fair
    # budget. (The holdout-mode split only matters for the fit/validation
    # bookkeeping, not for the design-identifiability figure.)
    struct_rows = list(plan)
    struct_design = np.array([[r["sign"] * r["amplitude"] if r["mode"] == m else 0.0
                               for m in MODES] for r in struct_rows])   # (n, M)

    rng = np.random.default_rng(SEED)
    n = len(struct_rows)
    rand_design = rng.normal(0, 0.06, size=(n, len(MODES)))

    D_struct = modal_design(struct_design, J)   # (n, P) image-space response
    D_rand = modal_design(rand_design, J)

    # identifiability: design-space Gram condition (the correct metric)
    q_struct = design_quality(struct_design)     # (n, M) coefficient space
    q_rand = design_quality(rand_design)

    # per-mode peak/leakage (structured isolates each mode)
    pl_struct = per_mode_peak_leakage(struct_design, J)
    ident_rate_struct = float(np.mean([v > PEAK_LEAKAGE_FLOOR for v in pl_struct.values()]))

    # identifiability gate on the *measured* +A cross-response (physical twin)
    cross_matrix, direction = measure_cross_response(fwd)
    gate = identifiability_gate(cross_matrix, direction)
    sel = sorted(gate["selected"]) or list(MODES)
    rej = gate["rejected"]

    # ------------------------------------------------------------------
    # Fig 1 — cross-response matrix (structured, measured +A diffs) + gate
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cross_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels(MODES, fontsize=9)
    ax.set_yticks(range(len(MODES)))
    ax.set_yticklabels(MODES, fontsize=9)
    for i in range(len(MODES)):
        for j in range(len(MODES)):
            ax.text(j, i, f"{cross_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="w" if abs(cross_matrix[i, j]) > 0.5 else "k")
    fig.colorbar(im, ax=ax, fraction=0.05)
    cond = float(np.linalg.cond(cross_matrix))
    ax.set_title("Measured +A cross-response (structured, M×M)\n"
                 f"condition number = {cond:.1f}")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_cross_response.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 2 — per-mode direction agreement (the gate's |r|)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    xs = np.arange(len(MODES))
    ax.bar(xs, [abs(direction[m]) for m in MODES],
           color=["#3aa06a" if m in sel else "#c0574f" for m in MODES])
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.text(0, 0.52, "gate floor 0.5", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Z{m}" for m in MODES], fontsize=9)
    ax.set_ylabel("|Pearson r| (measured vs model +A direction)")
    ax.set_title("Per-mode direction agreement (green = gate-passed, red = rejected)")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_identifiability_rate.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 3 — per-mode identifiability (structured isolates each mode)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    # Left: structured design Gram — each mode on its own axis (isolated)
    ax = axes[0]
    gs = q_struct["gram"] / max(np.abs(q_struct["gram"]).max(), 1e-12)
    im = ax.imshow(gs, cmap="viridis")
    ax.set_title("structured design Gram (isolated per mode)\n"
                 "each probe excites exactly one mode")
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels(MODES, fontsize=8)
    ax.set_yticks(range(len(MODES)))
    ax.set_yticklabels(MODES, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.05)
    # Right: per-mode peak/leakage from the *physical* cross-response
    ax = axes[1]
    pl = np.array([pl_struct[m] for m in MODES])
    ax.bar(xs, pl, color=["#3aa06a" if v > PEAK_LEAKAGE_FLOOR else "#c0574f"
                          for v in pl])
    ax.axhline(PEAK_LEAKAGE_FLOOR, color="k", ls="--", lw=1)
    ax.text(0, PEAK_LEAKAGE_FLOOR * 1.05, f"identifiability floor {PEAK_LEAKAGE_FLOOR}",
            fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Z{m}" for m in MODES], fontsize=8)
    ax.set_ylabel("peak/leakage (log)")
    ax.set_title(f"Per-mode peak/leakage (model basis)\n"
                 f"identifiability rate = {ident_rate_struct:.0%}")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Structured single-mode probes isolate each mode; the far-field "
                 "optical basis still cross-couples them (identifiability gate input)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_fit_tiers.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 4 — structured probe patterns
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(MODES), figsize=(13, 2.6), sharex=True)
    for ax, m in zip(axes, MODES):
        ax.imshow(structured_phase(dict(mode=m, amplitude=A_REF, sign=1), 64), cmap="viridis")
        ax.set_title(f"Z{m}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Structured single-mode probe phases (amplitude {A_REF} rad)", y=1.05, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_probe_patterns.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 5 — twin base image + single-mode response differences
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    axes[0].imshow(base, cmap="inferno")
    axes[0].set_title("twin base (residual=truth)")
    axes[0].axis("off")
    for ax, m in zip(axes[1:], [7, 9]):
        delta = respond(fwd, structured_phase(dict(mode=m, amplitude=A_REF, sign=1), GRID)) - base
        im = ax.imshow(delta, cmap="coolwarm")
        ax.set_title(f"ΔI, Z{m} @ {A_REF} rad")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_mode_response.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig 6 — online model-switch gate trajectory
    # ------------------------------------------------------------------
    from ao_shaping.tools.slm.transfer.online_augment import ModelSwitchGate
    gate_obj = ModelSwitchGate()
    gate_obj.incumbent_validation_nrmse = 0.50
    seq = [0.48, 0.47, 0.62, 0.60, 0.52, 0.50, 0.49]
    labels, passed = [], []
    for i, v in enumerate(seq, 1):
        verdict = gate_obj.check(v)
        labels.append(i)
        passed.append(verdict["passed"])
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ax.bar(labels, seq, color=["#3aa06a" if p else "#c0574f" for p in passed], alpha=0.9)
    ax.axhline(0.50, color="#2b6fba", lw=1.6, label="incumbent NRMSE")
    ax.axhline(0.50 * 1.10, color="#888", ls="--", lw=1.2, label="+10% cap")
    ax.set_xticks(labels)
    ax.set_ylabel("validation NRMSE")
    ax.set_title("Online model-switch gate: pass (green) / reject (red)\n"
                 "10% NRMSE cap + no guard regression; 2 consecutive fails -> rebuild")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_refeed_gate.png", dpi=140)
    plt.close(fig)

    results = dict(
        seed=SEED, elapsed_s=round(time.time() - t0, 2), injected=INJECTED,
        n_probes=n, frames=2 * n,
        condition_number=dict(structured=q_struct["cond"], random=q_rand["cond"]),
        per_mode_peak_leakage=pl_struct,
        identifiability_rate=dict(structured=ident_rate_struct,
                                  random_floor=f"cond {q_rand['cond']:.1f}"),
        gate=dict(selected=sel, rejected=rej, matrix_ok=gate["matrix_ok"],
                  diag_off_ratio=gate["diag_off_ratio"],
                  condition_number=gate["condition_number"]),
        cross_response_condition_number=cond,
        block_cap=BLOCK_L2_CAP_RAD,
        figures=[p.name for p in sorted(FIG.glob("*.png"))],
        device_free=True,
        note="FrozenDefocusForward twin, injected residual; device-free. "
             "Value of structured probes = per-mode identifiability (design "
             "conditioning), consistent with #229 null result on raw NRMSE. "
             "Hardware paired-CCD comparison is the separate plan.",
    )
    (OUT / "p0_results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"cond": results["condition_number"],
                      "ident_rate_struct": ident_rate_struct,
                      "gate": {"selected": sel, "rejected": rej,
                               "matrix_ok": gate["matrix_ok"]},
                      "peak_leakage": {m: round(v, 2) for m, v in pl_struct.items()},
                      "elapsed_s": results["elapsed_s"]}, indent=2, default=float))
    print("figures:", results["figures"])
    return results


def measure_cross_response(fwd):
    """Measured +A cross-response (physical twin) + per-mode direction agreement."""
    from ao_shaping.tools.slm.transfer.structured_probes import cross_response_matrix
    fwd0 = build_twin({j: 0.0 for j in MODES})
    base_phys = respond(fwd, np.zeros((GRID, GRID)))
    base_model = respond(fwd0, np.zeros((GRID, GRID)))
    physical = {}
    model = {}
    for m in MODES:
        phase = structured_phase(dict(mode=m, amplitude=1.0, sign=1), GRID)
        physical[m] = respond(fwd, phase) - base_phys
        model[m] = respond(fwd0, phase) - base_model
    matrix, _ = cross_response_matrix(base_phys, physical, amplitude_scale=1.0)
    direction = {}
    for m in MODES:
        a = (physical[m]).ravel()
        b = (model[m]).ravel()
        direction[m] = float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return matrix, direction


if __name__ == "__main__":
    main()
