"""The ``report/`` tree must stay linked to the scripts that produce it.

**What this pins.** On 2026-10-05 the experiment reports moved out of ``docs/``
into a repo-root ``report/`` so they would not be mixed with device
documentation. That move touched three things at once, and each can rot silently:

1. the **provenance header** inside every report (rendered from
   ``scripts/_common/provenance.py``) -- a generator rewrite or a hand edit drops
   it, and then a reader cannot tell what produced the numbers;
2. the **registry** -- a report added without a ``REPORTS`` entry is invisible to
   the index, and a renamed script leaves a dangling path in every report;
3. the **links** -- the header's script link is relative, so a report nested at a
   different depth silently points at the wrong place.

A grep cannot catch any of these: the header is machine-generated, the registry
is data, and a wrong-depth link still *looks* like a link. So the checks are
structural, the same reason ``test_conventions.py`` walks the AST instead of
matching lines.

The sync script is invoked as a subprocess rather than imported, because
``--check`` is the contract being tested -- importing it would let a refactor of
its internals silently redefine what "clean" means.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO / "report"
SYNC = REPO / "scripts" / "sync_report_provenance.py"

sys.path.insert(0, str(REPO))
from scripts._common.provenance import REPORTS, render_index, script_link


def test_index_resolves_reports_from_repository_root(monkeypatch):
    """Rendering outside the checkout must keep existing report links."""
    monkeypatch.chdir(REPORT_DIR)
    index = render_index()
    report = "shape_trials/transfer_uncertainty/transfer_uncertainty_20261104.md"
    assert f"]({report})" in index
    assert f"- `{report}`" not in index

_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

#: A link target only counts as a file reference if it ends in a known extension.
#: Without this, prose notation is misread as a link: ``report/slm/report2.md``
#: writes Zernike modes as ``[5](2,0)defocus``, and the tuple ``2,0`` was being
#: resolved as a path. Matched with :func:`re.search`, not :meth:`re.match` --
#: an anchored pattern would reject every multi-segment path and make the whole
#: check pass vacuously.
_LINKABLE = re.compile(
    r"\.(?:md|markdown|png|jpe?g|gif|csv|tsv|json|npy|npz|bmp|tiff?|txt|pdf|"
    r"xlsx?|docx?)$",
    re.IGNORECASE,
)

#: Links that were already broken before the 2026-10-05 move: ``report -> (href,
#: reason)``. Each needs a justification, because an unexplained baseline is just
#: a suppressed failure.
_PREEXISTING_BROKEN: dict[str, dict[str, str]] = {
    "report/shape_trials/energy_utilization_20261126/energy_utilization_20261126.md": {
        "figures/bw_w01_speed_fps.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w02_speed_timing.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w03_ideal_overview.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w04_ideal_lens.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w05_online_round.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w06_online_iteration.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w07_s1e_arm_a.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w08_s1e_arm_p.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w09_psf_validation.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w10_psf_independent.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w11_geometry_validation.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w12_geometry_holdout.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w13_hmws_linearity.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w14_hmws_focus.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w15_tsne_anchored.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w16_tsne_fit.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/bw_w17_kaxis.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/fig10_real_col_profile.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/fig11_real_metric_windows.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/fig12_real_abba_profiles.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/fig1_eta_compare.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/fig2_f0_breakdown.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/fig3_efficiency_match.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/fig4_pib.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/fig5_abba_cv.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/fig6_real_efficiency_match.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/fig7_real_abba_frames.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/fig8_real_square_shaping.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
        "figures/fig9_real_lobe_profile.png": (
            "figures were rendered locally in the authoring pass of upstream commit 2d9b1dea but never committed, and no in-repo script emits them; the report still cites them. Registered as debt in TODO.md batch 190 (2026-11-27) pending back-fill or a generator entry."
        ),
    },
    "report/shape_trials/model_in_loop/physical_forward_defocus_live_20261022.md": {
        "artifacts/physical_forward_rows_20261022.csv": (
            "per-phase metrics CSV, which .gitignore:7 (`*.csv`) keeps out of "
            "the repo entirely -- it exists only on the collecting machine, so "
            "the link cannot resolve on a clean clone."
        ),
    },
    "report/shape_trials/model_in_loop/physical_forward_sampling_20261022_r2.md": {
        "artifacts/sampling_radiometry_metrics_20261022_r2.csv": (
            "radiometry metrics CSV, which .gitignore:7 (`*.csv`) keeps out of "
            "the repo entirely -- it exists only on the collecting machine, so "
            "the link cannot resolve on a clean clone."
        ),
    },
    "report/shape_trials/gs_refine/gs_refine_live_20261012.md": {
        "../../../data/debug/slm_gs_refine/20261012_173710/slm_gs_refine_20261012_173710_20261012_173710.gif": (
            "hardware capture under data/debug/, which .gitignore:28 "
            "(`data/*`) keeps out of the repo entirely -- it exists only "
            "on the collecting machine, so the embed cannot resolve here."
        ),
        "../../../data/debug/slm_gs_refine/20261012_174456/slm_gs_refine_20261012_174456_20261012_174456.gif": (
            "hardware capture under data/debug/, which .gitignore:28 "
            "(`data/*`) keeps out of the repo entirely -- it exists only "
            "on the collecting machine, so the embed cannot resolve here."
        ),
        "../../../data/debug/slm_gs_refine/20261012_174627/slm_gs_refine_20261012_174628_20261012_174628.gif": (
            "hardware capture under data/debug/, which .gitignore:28 "
            "(`data/*`) keeps out of the repo entirely -- it exists only "
            "on the collecting machine, so the embed cannot resolve here."
        ),
        "../../../data/debug/slm_gs_refine/20261012_175632/slm_gs_refine_20261012_175632_20261012_175632.gif": (
            "hardware capture under data/debug/, which .gitignore:28 "
            "(`data/*`) keeps out of the repo entirely -- it exists only "
            "on the collecting machine, so the embed cannot resolve here."
        ),
        "../../../data/debug/slm_gs_refine/20261012_175810/slm_gs_refine_20261012_175811_20261012_175811.gif": (
            "hardware capture under data/debug/, which .gitignore:28 "
            "(`data/*`) keeps out of the repo entirely -- it exists only "
            "on the collecting machine, so the embed cannot resolve here."
        ),
        "../../../data/debug/slm_gs_refine/20261012_175929/slm_gs_refine_20261012_175929_20261012_175929.gif": (
            "hardware capture under data/debug/, which .gitignore:28 "
            "(`data/*`) keeps out of the repo entirely -- it exists only "
            "on the collecting machine, so the embed cannot resolve here."
        ),
        "../../../data/debug/slm_gs_refine/20261012_180222/slm_gs_refine_20261012_180222_20261012_180222.gif": (
            "hardware capture under data/debug/, which .gitignore:28 "
            "(`data/*`) keeps out of the repo entirely -- it exists only "
            "on the collecting machine, so the embed cannot resolve here."
        ),
    },
    "report/slm/slm_rms_spgd/0516.md": {
        "image/0516/1778923338908.png": (
            "screenshot never committed -- searched the whole repo by name, it "
            "exists nowhere. The embed has always rendered broken."
        ),
        "image/0516/1778923774713.png": (
            "screenshot never committed -- searched the whole repo by name, it "
            "exists nowhere. The embed has always rendered broken."
        ),
    },
    "report/shape_trials/model_in_loop/model_in_loop_live_20261013.md": {
        # Hardware debug artefacts under data/debug/, kept out by .gitignore
        # (`data/*`). They exist only on the collecting bench, so the relative
        # links resolve there but not in a clean clone.
        "../../../data/debug/slm_model_in_loop/20261013_105911/bench_geometry.json": (
            "bench geometry json under data/debug/, gitignored (`data/*`) -- "
            "exists only on the collecting machine."
        ),
        "../../../data/debug/slm_model_in_loop/20261013_114650/best_far_field.png": (
            "hardware capture under data/debug/, gitignored (`data/*`) -- "
            "exists only on the collecting machine."
        ),
        "../../../data/debug/slm_model_in_loop/20261013_145819/best_far_field.png": (
            "hardware capture under data/debug/, gitignored (`data/*`) -- "
            "exists only on the collecting machine."
        ),
        "../../../data/debug/slm_model_in_loop/20261013_152341/best_far_field.png": (
            "hardware capture under data/debug/, gitignored (`data/*`) -- "
            "exists only on the collecting machine."
        ),
        "../../../data/debug/slm_exposure_analysis/20261013_1935/analysis_pixelwise.json": (
            "exposure analysis json under data/debug/, gitignored (`data/*`) -- "
            "exists only on the collecting machine."
        ),
        "../../../data/debug/slm_model_compare/20261013_psf_trial/paired_observable_holdouts.csv": (
            "paired-holdout csv under data/debug/, gitignored (`data/*`) -- "
            "exists only on the collecting machine."
        ),
        "../../../data/debug/slm_model_compare/20261013_phase_bandwidth_trial/paired_holdouts.csv": (
            "paired-holdout csv under data/debug/, gitignored (`data/*`) -- "
            "exists only on the collecting machine."
        ),
        "../../../data/debug/slm_model_compare/20261013_2015_new_session/psf_paired.csv": (
            "paired csv under data/debug/, gitignored (`data/*`) -- "
            "exists only on the collecting machine."
        ),
        "../../../data/debug/slm_model_compare/20261013_2015_new_session/phase_bandwidth_paired.csv": (
            "paired csv under data/debug/, gitignored (`data/*`) -- "
            "exists only on the collecting machine."
        ),
        "../../../data/debug/slm_disturbance_compare/20261013_165039/disturbance.json": (
            "disturbance json under data/debug/, gitignored (`data/*`) -- "
            "exists only on the collecting machine."
        ),
    },
    "report/micro_deformable_mirror/voltage_test.md": {
        # R50 voltage-test screenshots under data/md_test/, gitignored (`data/*`).
        "../../data/md_test/md_img-100v/192.168.0.124/192.168.0.124-002.png": (
            "R50 voltage-test screenshot under data/md_test/, gitignored "
            "(`data/*`) -- exists only on the collecting machine."
        ),
        "../../data/md_test/md_img-100v/192.168.0.114/192.168.0.114-000.png": (
            "R50 voltage-test screenshot under data/md_test/, gitignored "
            "(`data/*`) -- exists only on the collecting machine."
        ),
        "../../data/md_test/md_img-100v/192.168.0.114/192.168.0.114-001.png": (
            "R50 voltage-test screenshot under data/md_test/, gitignored "
            "(`data/*`) -- exists only on the collecting machine."
        ),
        "../../data/md_test/md_img-100v/192.168.0.114/192.168.0.114-002.png": (
            "R50 voltage-test screenshot under data/md_test/, gitignored "
            "(`data/*`) -- exists only on the collecting machine."
        ),
    },
}


def _link_targets(text: str) -> list[str]:
    """File-ish link targets in ``text``, with anchors/line refs stripped."""
    out = []
    for href in _MARKDOWN_LINK.findall(text):
        href = href.strip()
        if href.startswith(("http://", "https://", "#", "mailto:")):
            continue
        href = href.split("#", 1)[0].split(":", 1)[0]  # drop anchor / :47 line ref
        if href and _LINKABLE.search(href):
            out.append(href)
    return out


def _report_markdown() -> list[Path]:
    """Every markdown file under ``report/`` except the generated index.

    ``report/AGENTS.md`` is the *layout spec* for the report tree, not a
    report -- it carries no provenance block and is never regenerated.
    """
    skip = {"report/README.md", "report/AGENTS.md"}
    return [
        p
        for p in sorted(REPORT_DIR.rglob("*.md"))
        if p.relative_to(REPO).as_posix() not in skip
    ]


def test_the_report_tree_exists_and_is_not_empty() -> None:
    """A vacuous pass is worse than a failure: assert there is something to guard."""
    assert REPORT_DIR.is_dir(), "report/ is missing -- the reports were never moved"
    assert len(_report_markdown()) >= 40, (
        f"only {len(_report_markdown())} reports found; the move should have brought "
        "the whole tree, so a near-empty report/ means the sweep went wrong"
    )


def test_sync_check_reports_no_drift() -> None:
    """``--check`` must exit 0: every header present and byte-identical."""
    proc = subprocess.run(
        [sys.executable, str(SYNC), "--check"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=False,
    )
    assert proc.returncode == 0, (
        "report provenance is stale -- run "
        "`python scripts/sync_report_provenance.py`:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


#: MIL 外围移除 (e7d9a305) 删掉 figures/artifacts 资产但保留了叙事报告 ——
#: 断链按文件登记为基线留痕 (报告是实验记录, 资产已按上游决定不恢复)。
_MIL_ORPHAN_REASON = (
    "MIL 外围移除 (e7d9a305) 删除了该资产; 叙事报告保留为实验记录, 断链属预期分叉"
)
_MIL_ORPHANED: dict[str, tuple[str, ...]] = {
    'report/shape_trials/model_in_loop/model_in_loop_hw_ab_20261021.md': (
        'figures/20261021_hw_ab_summary.png',
        'figures/20261021_metrics_curves_baseline_a.png',
        'figures/20261021_metrics_curves_baseline_b.png',
        'figures/20261021_mil_inverse_baseline_a.gif',
        'figures/20261021_mil_inverse_baseline_b.gif',
        'figures/20261021_panel_scan_baseline_a.png',
        'figures/20261021_panel_scan_baseline_b.png',
        'figures/20261021_panel_scan_large_1000.png',
        'figures/20261021_true_vs_pred_baseline_a_probes_p1.png',
        'figures/20261021_true_vs_pred_baseline_a_probes_p2.png',
        'figures/20261021_true_vs_pred_baseline_a_probes_p3.png',
        'figures/20261021_true_vs_pred_baseline_a_stats.png',
        'figures/20261021_true_vs_pred_baseline_b_probes_p1.png',
        'figures/20261021_true_vs_pred_baseline_b_probes_p2.png',
        'figures/20261021_true_vs_pred_baseline_b_probes_p3.png',
        'figures/20261021_true_vs_pred_baseline_b_stats.png',
    ),
    'report/shape_trials/model_in_loop/model_in_loop_live_20261013.md': (
        'figures/20261013_114650_xy_profile.png',
        'figures/20261013_145819_xy_profile.png',
        'figures/20261013_152341_xy_profile.png',
        'images/mil_correlation_comparison.png',
        'images/mil_deep_model_comparison.png',
        'images/mil_exposure_and_model_comparison.png',
        'images/mil_model_candidate_comparison.png',
        'images/mil_model_correlation_bar.png',
        'images/mil_model_residual.png',
        'images/mil_new_session_verification.png',
        'images/mil_phase_ccd_montage.png',
        'images/mil_prediction_comparison.png',
        'images/mil_replay_comparison.png',
        'images/mil_run_comparison.png',
        'images/mil_train_probes_distribution.png',
        'images/mil_warm_cold_replay.png',
        'images/mil_zernike_spectrum.png',
    ),
    'report/shape_trials/model_in_loop/model_in_loop_live_20261014.md': (
        'figures/model_in_loop_live_20261014_1115_zero_vs_refit_01_A_farfield.png',
        'figures/model_in_loop_live_20261014_1115_zero_vs_refit_01_A_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_1115_zero_vs_refit_02_B_farfield.png',
        'figures/model_in_loop_live_20261014_1115_zero_vs_refit_02_B_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_1115_zero_vs_refit_04_B_farfield.png',
        'figures/model_in_loop_live_20261014_1115_zero_vs_refit_04_B_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_1115_zero_vs_refit_05_A_farfield.png',
        'figures/model_in_loop_live_20261014_1115_zero_vs_refit_05_A_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_1115_zero_vs_refit_A_zero_step_phase.png',
        'figures/model_in_loop_live_20261014_1115_zero_vs_refit_B_10_step_phase.png',
        'figures/model_in_loop_live_20261014_20261014_105041_farfield.png',
        'figures/model_in_loop_live_20261014_20261014_105041_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_20261014_105041_phase.png',
        'figures/model_in_loop_live_20261014_20261014_105205_farfield.png',
        'figures/model_in_loop_live_20261014_20261014_105205_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_20261014_105205_phase.png',
        'figures/model_in_loop_live_20261014_20261014_105339_farfield.png',
        'figures/model_in_loop_live_20261014_20261014_105339_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_20261014_105339_phase.png',
        'figures/model_in_loop_live_20261014_20261014_105438_farfield.png',
        'figures/model_in_loop_live_20261014_20261014_105438_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_20261014_105438_phase.png',
        'figures/model_in_loop_live_20261014_20261014_105535_farfield.png',
        'figures/model_in_loop_live_20261014_20261014_105535_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_20261014_105535_phase.png',
        'figures/model_in_loop_live_20261014_20261014_110203_farfield.png',
        'figures/model_in_loop_live_20261014_20261014_110203_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_20261014_110203_phase.png',
        'figures/model_in_loop_live_20261014_20261014_110305_farfield.png',
        'figures/model_in_loop_live_20261014_20261014_110305_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_20261014_110305_phase.png',
        'figures/model_in_loop_live_20261014_20261014_170820_farfield.png',
        'figures/model_in_loop_live_20261014_20261014_170820_farfield_target.png',
        'figures/model_in_loop_live_20261014_20261014_170820_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_20261014_170820_phase.png',
        'figures/model_in_loop_live_20261014_20261014_171043_farfield.png',
        'figures/model_in_loop_live_20261014_20261014_171043_farfield_target.png',
        'figures/model_in_loop_live_20261014_20261014_171043_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_20261014_171043_phase.png',
        'figures/model_in_loop_live_20261014_20261014_171219_farfield.png',
        'figures/model_in_loop_live_20261014_20261014_171219_farfield_target.png',
        'figures/model_in_loop_live_20261014_20261014_171219_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_20261014_171219_phase.png',
        'figures/model_in_loop_live_20261014_20261014_171301_farfield.png',
        'figures/model_in_loop_live_20261014_20261014_171301_farfield_target.png',
        'figures/model_in_loop_live_20261014_20261014_171301_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_20261014_171301_phase.png',
        'figures/model_in_loop_live_20261014_20261014_171340_farfield.png',
        'figures/model_in_loop_live_20261014_20261014_171340_farfield_target.png',
        'figures/model_in_loop_live_20261014_20261014_171340_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261014_20261014_171340_phase.png',
    ),
    'report/shape_trials/model_in_loop/model_in_loop_live_20261016.md': (
        'figures/model_in_loop_live_20261016_20261016_1044_uniformity_weights_01_A_farfield.png',
        'figures/model_in_loop_live_20261016_20261016_1044_uniformity_weights_01_A_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261016_20261016_1044_uniformity_weights_02_B_farfield.png',
        'figures/model_in_loop_live_20261016_20261016_1044_uniformity_weights_02_B_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261016_20261016_1044_uniformity_weights_04_B_farfield.png',
        'figures/model_in_loop_live_20261016_20261016_1044_uniformity_weights_04_B_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261016_20261016_1044_uniformity_weights_05_A_farfield.png',
        'figures/model_in_loop_live_20261016_20261016_1044_uniformity_weights_05_A_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261016_20261016_1044_uniformity_weights_A_phase.png',
        'figures/model_in_loop_live_20261016_20261016_1044_uniformity_weights_B_phase.png',
        'figures/model_in_loop_live_20261016_20261016_1051_refit_steps_01_A_farfield.png',
        'figures/model_in_loop_live_20261016_20261016_1051_refit_steps_01_A_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261016_20261016_1051_refit_steps_02_B_farfield.png',
        'figures/model_in_loop_live_20261016_20261016_1051_refit_steps_02_B_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261016_20261016_1051_refit_steps_04_B_farfield.png',
        'figures/model_in_loop_live_20261016_20261016_1051_refit_steps_04_B_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261016_20261016_1051_refit_steps_05_A_farfield.png',
        'figures/model_in_loop_live_20261016_20261016_1051_refit_steps_05_A_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261016_20261016_1051_refit_steps_A_phase.png',
        'figures/model_in_loop_live_20261016_20261016_1051_refit_steps_B_phase.png',
        'figures/model_in_loop_live_20261016_20261016_1053_refit_steps_reverse_01_A_farfield.png',
        'figures/model_in_loop_live_20261016_20261016_1053_refit_steps_reverse_01_A_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261016_20261016_1053_refit_steps_reverse_02_B_farfield.png',
        'figures/model_in_loop_live_20261016_20261016_1053_refit_steps_reverse_02_B_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261016_20261016_1053_refit_steps_reverse_04_B_farfield.png',
        'figures/model_in_loop_live_20261016_20261016_1053_refit_steps_reverse_04_B_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261016_20261016_1053_refit_steps_reverse_05_A_farfield.png',
        'figures/model_in_loop_live_20261016_20261016_1053_refit_steps_reverse_05_A_farfield_xy_profile.png',
        'figures/model_in_loop_live_20261016_20261016_1053_refit_steps_reverse_A_phase.png',
        'figures/model_in_loop_live_20261016_20261016_1053_refit_steps_reverse_B_phase.png',
    ),
    'report/shape_trials/model_in_loop/model_in_loop_live_20261019.md': (
        '../../../docs/slm/model_in_loop_handoff.md',
        'assets/20261019_111941/holdout_ceiling.png',
        'assets/20261019_111941/holdout_frames.png',
        'assets/20261019_111941/holdout_frames_normalized.png',
        'figures/20261019_100334_farfield.png',
        'figures/20261019_100334_farfield_xy_profile.png',
        'figures/20261019_100334_phase.png',
        'figures/20261019_103208_farfield.png',
        'figures/20261019_103208_farfield_xy_profile.png',
        'figures/20261019_103208_phase.png',
        'figures/20261019_103259_farfield.png',
        'figures/20261019_103259_farfield_xy_profile.png',
        'figures/20261019_103259_phase.png',
        'figures/20261019_103347_farfield.png',
        'figures/20261019_103347_farfield_xy_profile.png',
        'figures/20261019_103347_phase.png',
        'figures/20261019_110323_farfield.png',
        'figures/20261019_110323_farfield_xy_profile.png',
        'figures/20261019_110323_phase.png',
        'figures/20261019_111359_farfield.png',
        'figures/20261019_111359_farfield_xy_profile.png',
        'figures/20261019_111359_phase.png',
        'figures/20261019_111941_farfield.png',
        'figures/20261019_111941_farfield_xy_profile.png',
        'figures/20261019_111941_phase.png',
    ),
    'report/shape_trials/model_in_loop/model_in_loop_record_20261021_172309.md': (
        'figures/20261021_172309_record_farfield.png',
        'figures/20261021_172309_record_gradient_evolution.png',
        'figures/20261021_172309_record_model_aberration.png',
        'figures/20261021_172309_record_phase_best.png',
    ),
    'report/shape_trials/model_in_loop/model_in_loop_uniformity_live_20261016.md': (
        '../../../docs/slm/model_in_loop_handoff.md',
    ),
    'report/shape_trials/model_in_loop/physical_forward_defocus_live_20261022.md': (
        'artifacts/physical_forward_summary_20261022.json',
        'figures/physical_forward_predictions_20261022.png',
        'figures/physical_forward_spot_profiles_20261022.png',
    ),
    'report/shape_trials/model_in_loop/physical_forward_sampling_20261022_r2.md': (
        'artifacts/fixed_phase_repeat_metrics_20261022_r2.json',
        'artifacts/sampling_radiometry_summary_20261022_r2.json',
        'figures/sampling_radiometry_20261022_r2.png',
    ),
}
for _f, _hs in _MIL_ORPHANED.items():
    _PREEXISTING_BROKEN.setdefault(_f, {}).update({h: _MIL_ORPHAN_REASON for h in _hs})

@pytest.mark.parametrize("path", _report_markdown(), ids=lambda p: p.as_posix()[-60:])
def test_every_report_carries_a_provenance_block(path: Path) -> None:
    """Each report states its script (or that it is hand-written) and its env."""
    key = path.relative_to(REPO).as_posix()
    assert key in REPORTS, (
        f"{key} has no entry in scripts/_common/provenance.py::REPORTS, so it is "
        "missing from report/README.md and gets no provenance header"
    )
    text = path.read_text(encoding="utf-8-sig")
    assert "<!-- provenance:start -->" in text, f"{key}: provenance header absent"
    assert "<!-- provenance:end -->" in text, f"{key}: provenance header unterminated"
    assert text.count("<!-- provenance:start -->") == 1, (
        f"{key}: stacked provenance headers -- the sync script is not idempotent"
    )
    assert "**生成脚本**" in text and "**运行环境**" in text, (
        f"{key}: header is missing the script or the environment field"
    )


_MAX_PENDING_REPORTS = 1


def _pending_report_keys() -> list[str]:
    """Registry keys whose report file does not exist yet.

    Derived with the exact rule render_index() applies when it lists entries
    under the index's pending section, so this guard and report/README.md
    cannot drift apart about what "pending" means.
    """
    return sorted(k for k in REPORTS if not (REPO / k).is_file())


@pytest.mark.parametrize("key", sorted(REPORTS), ids=lambda k: k[-55:])
def test_registry_script_paths_exist(key: str) -> None:
    """A registry entry naming a script that does not exist is worse than none."""
    prov = REPORTS[key]
    pending = not (REPO / key).is_file()
    for script in (prov.script, *prov.extra_scripts):
        if not script:
            continue
        tail = (
            " -- the report itself is also pending, so this entry waits on the "
            "author landing both the generator and the report (TODO 8.9 items 77/79)"
            if pending
            else ". Either the generator was renamed or the registry is stale."
        )
        assert (REPO / script).is_file(), f"{key} names {script}, which does not exist{tail}"


def test_pending_registry_entries_are_bounded_and_justified() -> None:
    """The pending state is a deliberate hole -- keep it small and explained.

    render_index() lists any registry entry whose report file is absent under
    the index's pending section. Legitimate for author-declared cases; it must
    never become a place where phantom entries quietly accumulate.
    """
    pending = _pending_report_keys()
    for key in pending:
        assert REPORTS[key].note, (
            f"{key} is pending (report file absent) but its registry note is "
            "empty -- add one saying what it is waiting on"
        )
    assert len(pending) <= _MAX_PENDING_REPORTS, (
        f"{len(pending)} registry entries point at reports that do not exist "
        f"(max {_MAX_PENDING_REPORTS}): {pending}. Either generate the reports "
        "or prune the entries."
    )


def test_every_generating_script_lives_under_scripts_or_tests() -> None:
    """The *generating* entry point must be a script or a test, never ``src/``.

    The report rule (AGENTS.md) puts markdown *generation* in ``scripts/``, so a
    ``src/`` primary script means the registry points at the code that computes
    the numbers rather than the code that writes the report. ``extra_scripts``
    are exempt: a legitimate one is a ``src/`` data producer that the renderer
    merely reads from.
    """
    for key, prov in REPORTS.items():
        if prov.script:
            assert prov.script.startswith(("scripts/", "tests/")), (
                f"{key}: generating script {prov.script} is not under scripts/ or "
                "tests/ -- report generation belongs in scripts/ (AGENTS.md)"
            )
        for extra in prov.extra_scripts:
            assert extra.startswith(("scripts/", "tests/", "src/")), (
                f"{key}: associated script {extra} is not a repo module path"
            )


@pytest.mark.parametrize("key", sorted(REPORTS), ids=lambda k: k[-55:])
def test_header_script_links_resolve(key: str) -> None:
    """The relative link in a header must point at a file that exists.

    Depth is supplied by the caller, so this also pins that a report nested more
    deeply than the registry author assumed gets a link matching its own depth.
    """
    if key in _pending_report_keys():
        pytest.skip(
            f"{key}: report not yet generated (the index lists it as pending), "
            "so there is no header whose links could resolve"
        )
    path = REPO / key
    depth = len(path.relative_to(REPO).parts) - 1
    prov = REPORTS[key]
    for script in (prov.script, *prov.extra_scripts):
        if not script:
            continue
        href = script_link(script, depth).split("](")[1].rstrip(")")
        assert (path.parent / href).resolve().is_file(), (
            f"{key}: header link {href!r} does not resolve "
            f"(depth={depth}, script={script})"
        )


def test_commands_name_the_same_script_as_the_registry() -> None:
    """The repro command must invoke the declared script.

    A copy-paste slip here is invisible to a reader who trusts the command, and
    it is the field people actually paste.
    """
    for key, prov in REPORTS.items():
        if not prov.command:
            assert not prov.script, (
                f"{key}: declares script {prov.script} but no repro command"
            )
            continue
        assert prov.script in prov.command, (
            f"{key}: command {prov.command!r} does not invoke the declared script "
            f"{prov.script!r}"
        )


def test_index_links_resolve() -> None:
    """Every link in the generated ``report/README.md`` must resolve on disk."""
    index = REPORT_DIR / "README.md"
    assert index.is_file(), "report/README.md is missing -- run the sync script"
    broken = [
        h
        for h in _link_targets(index.read_text(encoding="utf-8-sig"))
        if not (index.parent / h).exists()
    ]
    assert not broken, (
        f"report/README.md links to missing files: {_describe_broken(index.parent, broken)}"
    )


@pytest.mark.parametrize("path", _report_markdown(), ids=lambda p: p.as_posix()[-60:])



def test_report_relative_links_resolve(path: Path) -> None:
    """Relative links inside a report must still resolve after the move.

    The move preserved directory depth for almost every report, which is why the
    figure links kept working -- but a few changed depth
    (``micro_deformable_mirror/freq_test.md`` lost a level,
    ``centroid_test_visualization`` lost one), and those are exactly the ones a
    blanket rename silently breaks.
    """
    key = path.relative_to(REPO).as_posix()
    broken = [
        h
        for h in _link_targets(path.read_text(encoding="utf-8-sig"))
        if not (path.parent / h).exists()
    ]
    allowed = _PREEXISTING_BROKEN.get(key)
    if allowed is None:
        assert not broken, (
            f"{key}: broken relative links {_describe_broken(path.parent, broken)}"
        )
        return
    unexpected = [h for h in broken if h not in allowed]
    assert not unexpected, (
        f"{key}: new broken links {_describe_broken(path.parent, unexpected)} — only "
        "the baselined ones in _PREEXISTING_BROKEN are tolerated"
    )


def _is_gitignored(path: Path) -> bool | None:
    """True when git refuses to track ``path``, False when it would, None when git
    could not answer at all.

    ``check-ignore`` exits 0 for ignored and 1 for not-ignored; any other code is a
    git failure (rc=128 when the mount excludes this checkout's git metadata). Those
    are different answers: reporting a failure as "not ignored" tells the reader to
    delete a baseline entry that is in fact justified.
    """
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(path)],
        cwd=str(REPO),
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None


def _describe_broken(parent: Path, hrefs: list[str]) -> list[str]:
    """Label each unresolved link as policy-blocked or simply missing.

    The two need opposite fixes and look identical in a plain link list: a target
    git refuses to track can never be repaired by committing it, so the link itself
    is the defect (#192 -- ``artifacts/*.csv`` under the global ``*.csv`` rule),
    while a non-ignored target is a file that should have been committed.
    ``None`` from :func:`_is_gitignored` is reported as unknown rather than folded
    into "missing", which would push the reader toward committing a file that
    ``.gitignore`` will keep refusing.
    """
    labels = []
    for href in hrefs:
        verdict = _is_gitignored(parent / href)
        if verdict is True:
            labels.append(
                f"{href} [gitignored -- can never be committed: render the path "
                f"as code, not a link]"
            )
        elif verdict is None:
            labels.append(f"{href} [git unavailable -- ignored status unknown]")
        else:
            labels.append(f"{href} [not ignored -- missing file]")
    return labels


def test_the_preexisting_baseline_is_still_justified() -> None:
    """Each baselined link needs a real reason, and must still be unresolvable.

    The second half matters as much as the first: if someone re-attaches the
    missing screenshots, the entry must be removed rather than left guarding
    nothing, which is how a baseline rots into a no-op.

    "Unresolvable" deliberately means *in a clean clone*, not *on this machine*.
    A link into ``data/`` is gitignored (``.gitignore`` has ``data/*``), so it is
    absent for everyone who has not run the bench -- yet on the machine that
    collected it the file is sitting on disk and the href resolves fine. Keying
    the guard on ``.exists()`` alone therefore made it environment-dependent: it
    passed in CI and failed on the collecting machine, which is precisely the
    machine whose reports these entries were written for. Accepting either
    "broken" or "gitignored" keeps the anti-rot intent for genuinely re-attached
    files while making the check reproducible on both sides.
    """
    for key, links in _PREEXISTING_BROKEN.items():
        path = REPO / key
        assert path.is_file(), f"{key} is baselined but no longer exists"
        broken = {
            h
            for h in _link_targets(path.read_text(encoding="utf-8-sig"))
            if not (path.parent / h).exists()
        }
        for href, reason in links.items():
            assert reason and not reason.startswith("TBD"), f"{key} {href}: no reason"
            target = path.parent / href
            ignored = _is_gitignored(target)
            if ignored is None:
                pytest.skip(
                    "git cannot answer for this checkout (its git metadata is outside "
                    "the mounted tree, e.g. a linked worktree), so the gitignored half "
                    "of this check is unavailable here"
                )
            assert href in broken or ignored, (
                f"{key}: {href} is baselined as unresolvable but now resolves AND is "
                "tracked by git — remove the entry so new breakage in this report is "
                "caught again"
            )
