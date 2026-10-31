"""Shape guard for the sweep-calibration split, issue #205 batch 1.

#205 registered the repo-wide long-function / deep-nesting debt and asked for
per-family batches, each characterization-first and each sealed with a shape
guard so a split cannot grow back. Batch 1 (wfless family) split
``calibrate_bench_geometry_from_sweep`` (662 lines, nesting 4) into nine
module-level helpers by **verbatim** block moves: the bit-for-bit numeric
goldens already in ``test_model_in_loop_sweep_diagnostics.py``
(``REF_ALL_NORMAL_NOTES`` / ``REF_ALL_NORMAL``) pin that the move did not move
a single digit.

The goldens pin equality for today; they do not stop any of the ten functions
from growing back. So the ten are pinned here by AST: span (end_lineno -
lineno + 1) and maximum nesting depth (if / for / while / try / with), the
same convention as ``test_issue167_function_shape.py``. Ceilings are measured
value plus headroom -- except the nesting ceilings at the #167 criterion (<=3)
where the measured value already sits: the two filter/fit blocks moved
verbatim measure nesting 4, so their ceiling is 4 with flattening left to a
later #205 batch, not hidden here by raising the bar repo-wide.

Scope: these functions only. The rest of #205's inventory
(``optimize_slm_square`` at 1287, ``optimize_pib`` at nesting 8, ...) lands in
its own batches with its own ceilings -- pinning it here would turn this guard
into a repo-wide lint that no single batch can pay.

Batch 2 (scripts/slm family) added the ``build_markdown`` split of
``generate_diffraction_efficiency_sim_report.py`` (596 lines, nesting 0): the
section f-strings moved line-exact into three helpers behind a rebind header,
sealed here with the same span/nesting convention. Paths are repo-relative
from batch 2 on, since the guard now covers ``scripts/`` as well as ``src/``.
Batch 3 (same family) added the ``render_retest_section`` split of
``_hw_gs_square_scans_sections.py`` (537 lines, nesting 5) by the same
rebind-header pattern; its nesting-5 zernike block moved verbatim and keeps
ceiling 5 -- flattening is a later batch, not a raised bar here.
Batch 4 (wfless family) lifted the 193-line tqdm epoch block of
``optimize_pib`` (582 lines, nesting 8 -- the repo's deepest) into
``_run_spgd_epochs``: read-only closure names enter as keyword params, the six
outer locals the loop mutates are threaded back through a returned namespace,
and the per-epoch sha256 characterization in
``test_spid_pib_path_characterization.py`` is the behavior net. Nesting 8 fell
to 5 in the orchestrator and 6 in the lifted block -- the block's own deep
nesting moved verbatim and keeps its measured ceiling.
Batch 5 (same family, ``combined_optimizer``) lifted the 225-line
open_pib_bench With block the same way -- 16 read-only params, 5 threaded
mutables; the orchestrator went from nesting 6 to 0.
Batch 6 (tools/slm/transfer) lifted the 325-line open-camera With block of
``validate.run`` (535 lines, nesting 9) into ``_run_validate_session``; the
net is 7 transfer suites (test_transfer_workflow.py drives run() end-to-end),
and the resume-gate source pin in ``test_validate_blocks.py`` moved to two
hops (run -> session -> gate). Nesting 9 in the lifted block keeps its
measured ceiling -- flattening is a later batch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_FILE = "src/ao_shaping/optimizer/wfless/model_in_loop_shaping.py"
_PIB = "src/ao_shaping/optimizer/wfless/pib.py"
_COMB = "src/ao_shaping/optimizer/combined_optimizer.py"
_VAL = "src/ao_shaping/tools/slm/transfer/validate.py"
_SCRIPT = "scripts/slm/generate_diffraction_efficiency_sim_report.py"
_SECTIONS = "scripts/slm/_hw_gs_square_scans_sections.py"

#: (relative path under src, function name, span ceiling, nesting ceiling,
#:  the span measured when the guard was written)
TARGETS = [
    (_FILE, "calibrate_bench_geometry_from_sweep", 360, 3, 283),
    (_FILE, "_tilt_axis_slopes", 70, 3, 44),
    (_FILE, "_ramp_focal_scale", 70, 3, 39),
    (_FILE, "_fov_ratio_cross_check", 80, 3, 46),
    (_FILE, "_tilt_physical_cross_check", 80, 3, 48),
    (_FILE, "_sweep_width_groups", 140, 4, 100),
    (_FILE, "_fit_curvature_offsets", 100, 3, 60),
    (_FILE, "_defocus_asymmetry", 50, 3, 14),
    (_FILE, "_fit_joint_waist", 110, 4, 65),
    (_FILE, "_sweep_calibration_notes", 80, 3, 35),
    (_SCRIPT, "build_markdown", 90, 2, 45),
    (_SCRIPT, "_md_extend", 40, 2, 6),
    (_SCRIPT, "_md_metrics", 90, 2, 42),
    (_SCRIPT, "_md_sections_core", 230, 2, 160),
    (_SCRIPT, "_md_sections_studies", 240, 2, 171),
    (_SCRIPT, "_md_sections_tail", 330, 2, 245),
    (_SECTIONS, "render_retest_section", 60, 1, 14),
    (_SECTIONS, "_rt_extend", 40, 2, 6),
    (_SECTIONS, "_rt_preamble", 40, 1, 12),
    (_SECTIONS, "_rt_intro", 90, 2, 39),
    (_SECTIONS, "_rt_caliber", 160, 3, 95),
    (_SECTIONS, "_rt_atomicity", 120, 3, 58),
    (_SECTIONS, "_rt_blaze", 100, 4, 47),
    (_SECTIONS, "_rt_zernike", 110, 5, 52),
    (_SECTIONS, "_rt_verdict", 230, 1, 139),
    (_SECTIONS, "_rt_jitter", 150, 2, 88),
    (_SECTIONS, "_rt_contract", 120, 1, 69),
    (_PIB, "optimize_pib", 500, 5, 433),
    (_PIB, "_run_spgd_epochs", 300, 6, 239),
    (_COMB, "optimize_pib", 150, 1, 92),
    (_COMB, "_run_combined_epochs", 320, 6, 256),
    (_VAL, "run", 300, 5, 246),
    (_VAL, "_run_validate_session", 450, 9, 366),
]

NESTING_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith)


def _max_nesting(node: ast.AST, depth: int = 0) -> int:
    """Deepest if/for/while/try/with chain inside ``node``."""
    best = depth
    for child in ast.iter_child_nodes(node):
        child_depth = depth + 1 if isinstance(child, NESTING_NODES) else depth
        best = max(best, _max_nesting(child, child_depth))
    return best


def _measure(rel_path: str, func_name: str) -> tuple[int, int]:
    module = ast.parse((REPO / rel_path).read_text(encoding="utf-8"))
    hits = [
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == func_name
    ]
    assert len(hits) == 1, (
        f"expected exactly one {func_name} in {rel_path}, found {len(hits)} -- "
        "the guard would silently measure the wrong one"
    )
    node = hits[0]
    return node.end_lineno - node.lineno + 1, _max_nesting(node)


@pytest.mark.parametrize(("rel_path", "func_name", "span_max", "nest_max", "measured"), TARGETS)
def test_the_split_function_stays_within_its_shape_budget(
    rel_path: str, func_name: str, span_max: int, nest_max: int, measured: int
) -> None:
    span, nesting = _measure(rel_path, func_name)

    assert nesting <= nest_max, (
        f"{rel_path}::{func_name} nests {nesting} deep, over the #205 ceiling of "
        f"{nest_max}. Split the new branch out instead of nesting further."
    )
    assert span <= span_max, (
        f"{rel_path}::{func_name} is {span} lines, over its #205 ceiling of "
        f"{span_max} (measured {measured} when the guard was written). It is "
        "growing back -- split it again rather than raising the ceiling."
    )


def test_the_depth_measurement_is_not_silently_zero() -> None:
    """The walker must still see depth where depth exists.

    Pinned against one of the two blocks that genuinely sit at nesting 4
    (verbatim-moved filters, flattening left to a later batch): if the AST walk
    ever degraded to 0, every nesting ceiling above would pass vacuously.
    """
    _span, nesting = _measure(_FILE, "_sweep_width_groups")
    assert nesting == 4, (
        f"expected _sweep_width_groups to measure exactly 4 deep, got {nesting}; "
        "the nesting walker changed and the ceilings above mean nothing"
    )
