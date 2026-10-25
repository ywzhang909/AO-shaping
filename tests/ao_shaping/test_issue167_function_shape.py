"""Shape guard for the four functions issue #167 split.

#167 (the "long function / deep nesting" item carved out of #135) named exactly
four functions and asked for two things per function: split it, and check that
the maximum nesting afterwards is <= 3. The splits landed on main as 58e530e1,
2354bf73, 6bac5686 and dcd9c44b, each with its own regression (two Streamlit
call-sequence goldens and two numeric goldens).

What none of those regressions do is stop the four functions from growing back.
That is not a hypothetical: #167's own text records that the #131 GUI rework
"在 C 的渲染函数上继续加码" -- the 257-line renderer it started from had been
grown by other work, and nothing pins it from happening again. So the four
targets are pinned here by AST: span (end_lineno - lineno + 1) and maximum
nesting depth (counting if / for / while / try / with).

The ceilings are the issue's criterion plus headroom over the measured value,
not the measured value itself: a bound set exactly at today's span turns every
legitimate one-line edit into a golden regeneration. The headroom is what makes
the guard catch re-growth (tens of lines, a new nesting level) rather than
cosmetic churn.

Scope: these four functions only. #167 explicitly took the C item of #135 and
explicitly listed these four; the repo still carries a much larger long-function
debt (187 functions at nesting >= 4 at the time of writing, the longest being
``optimize_slm_zernike_pib`` at 1311 lines), which is a separate follow-up and
deliberately NOT pinned here -- pinning it would turn this guard into a
repo-wide lint that no single issue can pay down.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

#: (relative path under src, function name, span ceiling, nesting ceiling,
#:  the span measured when the guard was written)
TARGETS = [
    ("ao_shaping/optimizer/wfless/slm_gs_refine.py", "_run_gs_refine_session", 120, 3, 83),
    (
        "ao_shaping/algorithm/signal_processing/differentiable_shaping.py",
        "train_beam_shaping",
        160,
        3,
        121,
    ),
    ("ao_shaping/gui/slm/multi_slm_controller.py", "_render_slm_settings", 60, 3, 28),
    ("ao_shaping/gui/slm/multi_slm_controller.py", "render_phase_control", 60, 3, 17),
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
    module = ast.parse((SRC / rel_path).read_text(encoding="utf-8"))
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
        f"{rel_path}::{func_name} nests {nesting} deep, over the #167 ceiling of "
        f"{nest_max}. Split the new branch out instead of nesting further."
    )
    assert span <= span_max, (
        f"{rel_path}::{func_name} is {span} lines, over its #167 ceiling of "
        f"{span_max} (measured {measured} when the guard was written). It is "
        "growing back -- split it again rather than raising the ceiling."
    )


def test_the_depth_measurement_is_not_silently_zero() -> None:
    """The walker must still see depth where depth exists.

    Written against a function that is genuinely deep (nesting 8 at the time of
    writing) and that no issue has committed to flattening: if the AST walk ever
    degrades to returning 0, the four budgets above pass vacuously and the guard
    becomes decoration.
    """
    _span, nesting = _measure("ao_shaping/optimizer/wfless/pib.py", "optimize_pib")
    assert nesting >= 4, (
        f"expected optimize_pib to measure at least 4 deep, got {nesting}; the "
        "nesting walker is broken and the budgets above mean nothing"
    )
