"""Tests for the ``--objective`` option of the ``slm-gsnet`` square-shaping CLI.

Three layers are covered, in order of blast radius:

1. **Config surface** - :class:`ObjectiveParamsSquare` keeps ``objective="quality"``
   as its default and exposes it as a ``click.Choice(SQUARE_OBJECTIVE_CHOICES)``
   option, so adding the option cannot silently change an existing run.
2. **CLI surface** - both search subcommands (``spgd`` / ``heuristic``) accept
   every value of ``SQUARE_OBJECTIVE_CHOICES`` and reject everything else, and
   the parsed value really reaches ``SlmSquareConfig.objective``. The execution
   path is stubbed out, so **no camera or SLM is ever opened** (``gxipy`` is not
   even installed in this environment).
3. **Objective dispatcher** - :func:`square_objective_score` is a
   HIGHER-IS-BETTER dispatcher. ``"quality"`` forwards unchanged to
   :func:`square_quality_score` (its ``[0, 1]`` contract is *not* negated by the
   dispatcher), ``"pearson"`` negates :func:`pearson_shape_metric` in exactly one
   place, and the higher-is-better direction is asserted directly against a
   deliberately better / worse frame pair.

Locks that matter most:

* a *better* frame must score strictly higher under **every** objective, which
  is the only thing that catches a double sign flip in the Pearson branch;
* ``square_objective_score(..., "pearson")`` must equal
  ``-pearson_shape_metric(img, center, "square", side, 1.0)[0]`` exactly.

No hardware, no torch, no network, no files written.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import click
import numpy as np
import pytest
from click.testing import CliRunner

from ao_shaping.runners import slm_gsnet_runner
from ao_shaping.optimizer.wfless.slm_square_shaping import (
    SlmSquareConfig,
    square_objective_score,
    square_quality_score,
)
from ao_shaping.runners.runner_common import ObjectiveParamsSquare
from ao_shaping.runners.slm_gsnet_runner import heuristic, run, spgd
from ao_shaping.utils.image.target import SQUARE_OBJECTIVE_CHOICES as CANONICAL_CHOICES
from ao_shaping.utils.image.targets import pearson_shape_metric, target_shape_roi
from ao_shaping.utils.image.targets import SQUARE_OBJECTIVE_CHOICES as SHIM_CHOICES

# --- geometry of the synthetic frames --------------------------------------
# 64x64 window, a 16x16 target box centred on (32, 32) -> the ROI lands exactly
# on rows/cols 24..40, so "uniform inside the box" and "all energy in the box"
# are both hand-checkable.
FRAME_SHAPE = (64, 64)
#: ``square_objective_score`` annotates ``center`` as ``tuple[int, int]`` and
#: coerces to float internally, so pass ints and keep the declared type honest.
CENTER = (32, 32)
SIDE = 16
BOX = slice(24, 40)

#: ``cv``/``ee`` extreme corners used by the ``[0, 1]`` contract sweep.
CV_SAMPLES = (0.0, 0.5, 2.0, 20.0, 200.0)
EE_SAMPLES = (0.0, 0.25, 0.5, 1.0)
AR_SAMPLES = (0.25, 1.0, 4.0)


def _target_roi() -> np.ndarray:
    """The exact boolean ROI the ``"pearson"`` branch scores against."""
    return target_shape_roi(FRAME_SHAPE, CENTER, "square", float(SIDE), 1.0)


def _good_frame() -> np.ndarray:
    """A perfectly uniform square with *all* of its energy inside the box.

    CV is exactly 0, encircled energy exactly 1, and because the frame support
    equals the target ROI support the Pearson correlation is exactly 1 (so the
    metric's loss is ~0 and the dispatcher's score is ~0).
    """
    return _target_roi().astype(np.float64) * 100.0


def _bad_frame() -> np.ndarray:
    """A single hot pixel on a dim pedestal: high CV, ~7% energy in the box.

    Strictly worse than :func:`_good_frame` on *both* drivers of the quality
    score (uniformity and encircled energy) and clearly non-uniform in shape,
    so the Pearson correlation drops well below 1.
    """
    frame = np.full(FRAME_SHAPE, 15.0)
    frame[32, 32] = 400.0
    return frame


def _cv_ee(frame: np.ndarray) -> tuple[float, float]:
    """``(cv, ee)`` of ``frame`` w.r.t. the target box, computed from scratch.

    Deliberately re-implemented with plain NumPy instead of reusing the
    optimizer's private helpers: the test then pins the *ordering* the
    higher-is-better invariant depends on, independent of the code under test.
    """
    box = frame[BOX, BOX]
    mean = float(box.mean())
    assert mean > 0.0
    cv = float(box.std()) / mean
    ee = float(box.sum()) / float(frame.sum())
    return cv, ee


def _quality(obj: str, frame: np.ndarray) -> float:
    """``square_objective_score`` with the frame's own (cv, ee) and a square box."""
    cv, ee = _cv_ee(frame)
    return square_objective_score(frame, cv, ee, 1.0, CENTER, SIDE, objective=obj)


def _objective_option(command: click.Command) -> click.Parameter | None:
    """The click parameter declared as ``--objective`` on ``command``."""
    for param in command.params:
        if "--objective" in getattr(param, "opts", ()):
            return param
    return None


@pytest.fixture
def stub_execution(monkeypatch):
    """Neutralise the whole execution path so invoking the CLI is inert.

    Replaces ``optimize_slm_square`` with a recorder-returning stub and drops
    the two side-effecting helpers (``setup_coredumpy`` and the ``--cam_type
    sim`` global monkey-patch of ``Santec``). Guarantees no device is opened
    and no global module state is mutated.
    """
    captured: dict[str, object] = {}

    def _fake_optimize(*, center, epochs, config, **kwargs):
        captured["center"] = center
        captured["epochs"] = epochs
        captured["config"] = config
        return SimpleNamespace(
            history=[{"quality": 0.5, "cv": 0.1, "ee": 0.9, "ar": 1.0}]
        )

    monkeypatch.setattr(slm_gsnet_runner, "optimize_slm_square", _fake_optimize)
    monkeypatch.setattr(slm_gsnet_runner, "setup_coredumpy", lambda *a, **k: True)
    monkeypatch.setattr(slm_gsnet_runner, "_maybe_sim_patch", lambda cam_type: None)
    return captured


# ---------------------------------------------------------------------------
# 1. default preserved (the "byte-identical" regression guard)
# ---------------------------------------------------------------------------


def test_objective_params_square_default_is_quality() -> None:
    """The dataclass default must stay ``"quality"`` - no silent retarget."""
    assert ObjectiveParamsSquare().objective == "quality"


def test_optimizer_config_default_is_quality() -> None:
    """``SlmSquareConfig`` also defaults to the historical objective."""
    assert SlmSquareConfig().objective == "quality"


@pytest.mark.parametrize("command", [spgd, heuristic], ids=["spgd", "heuristic"])
def test_objective_click_option_default_is_quality(command: click.Command) -> None:
    """``--objective`` is a ``Choice(SQUARE_OBJECTIVE_CHOICES)`` defaulting to quality."""
    param = _objective_option(command)

    assert param is not None, "--objective is not declared on the command"
    assert param.name == "objective"
    assert param.default == "quality"
    assert isinstance(param.type, click.Choice)
    assert tuple(param.type.choices) == CANONICAL_CHOICES


@pytest.mark.parametrize("sub", ["spgd", "heuristic"])
def test_objective_appears_in_subcommand_help(sub: str) -> None:
    """Both subcommand helps advertise the option and both of its choices."""
    result = CliRunner().invoke(run, [sub, "--help"])

    assert result.exit_code == 0, result.output
    assert "--objective" in result.output
    assert "quality" in result.output
    assert "pearson" in result.output
    # The default is NOT rendered as click's "[default: quality]" column marker:
    # ``ObjectiveParamsSquare.objective`` does not pass ``show_default=True``
    # (unlike ``SpgdParams.optimizer_type``), so click only prints the value in
    # the help prose ("... score (default)"). The machine-readable default is
    # pinned by ``test_objective_click_option_default_is_quality`` instead.


# ---------------------------------------------------------------------------
# 2. accepted by both subcommands, and actually plumbed through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("objective", CANONICAL_CHOICES, ids=list(CANONICAL_CHOICES))
def test_objective_accepted_by_spgd_and_reaches_config(
    stub_execution, objective: str
) -> None:
    """``spgd`` accepts every square objective and hands it to the optimizer."""
    result = CliRunner().invoke(run, ["spgd", "--objective", objective, "-e", "3"])

    assert result.exit_code == 0, result.output
    assert result.exception is None
    config = stub_execution["config"]
    assert isinstance(config, SlmSquareConfig)
    assert config.objective == objective


@pytest.mark.parametrize("objective", CANONICAL_CHOICES, ids=list(CANONICAL_CHOICES))
def test_objective_accepted_by_heuristic_and_reaches_config(
    stub_execution, objective: str
) -> None:
    """``heuristic`` accepts every square objective and hands it to the optimizer."""
    result = CliRunner().invoke(run, ["heuristic", "--objective", objective, "-e", "3"])

    assert result.exit_code == 0, result.output
    assert result.exception is None
    config = stub_execution["config"]
    assert isinstance(config, SlmSquareConfig)
    assert config.objective == objective


def test_omitting_objective_reaches_config_as_quality(stub_execution) -> None:
    """Not passing ``--objective`` at all must behave exactly like before."""
    result = CliRunner().invoke(run, ["spgd", "-e", "2"])

    assert result.exit_code == 0, result.output
    assert stub_execution["config"].objective == "quality"


# ---------------------------------------------------------------------------
# 3. bogus values rejected by click
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sub", ["spgd", "heuristic"])
def test_bogus_objective_rejected_by_click(sub: str) -> None:
    """An unknown objective is a click usage error naming the valid choices."""
    result = CliRunner().invoke(run, [sub, "--objective", "bogus"])

    assert result.exit_code != 0
    assert "--objective" in result.output
    assert "bogus" in result.output
    for choice in CANONICAL_CHOICES:
        assert choice in result.output, f"valid choice {choice!r} missing from error"


# ---------------------------------------------------------------------------
# 4. the higher-is-better invariant (the sign-flip guard)
# ---------------------------------------------------------------------------


def test_bad_frame_is_worse_on_both_quality_drivers() -> None:
    """Precondition for the invariant below: the bad frame really is worse."""
    cv_good, ee_good = _cv_ee(_good_frame())
    cv_bad, ee_bad = _cv_ee(_bad_frame())

    assert cv_good < cv_bad  # more uniform
    assert ee_good > ee_bad  # more energy in the box
    assert (cv_good, ee_good) == (0.0, 1.0)


@pytest.mark.parametrize("objective", CANONICAL_CHOICES, ids=list(CANONICAL_CHOICES))
def test_better_frame_scores_higher(objective: str) -> None:
    """A better frame must score strictly higher under every square objective.

    This is the guard against a *double* sign flip: negating the Pearson loss
    twice, or negating the quality score, would make this assertion fail.
    """
    good = _quality(objective, _good_frame())
    bad = _quality(objective, _bad_frame())

    assert good > bad, (
        f"objective={objective!r} is not higher-is-better: good={good!r} <= bad={bad!r}"
    )


def test_pearson_branch_is_not_double_negated() -> None:
    """The Pearson loss for the bad frame is large, so its sign is observable."""
    loss_bad, _energy = pearson_shape_metric(
        _bad_frame(), CENTER, "square", float(SIDE), 1.0
    )
    assert loss_bad > 0.5  # 1 - corr, far from 0

    score = square_objective_score(
        _bad_frame(), 1.45, 0.07, 1.0, CENTER, SIDE, objective="pearson"
    )
    assert score < 0.0, "a poor frame must score below 0 under 'pearson'"


# ---------------------------------------------------------------------------
# 5. the sign is applied in exactly one place
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame_name", ["good", "bad"], ids=["good", "bad"])
def test_pearson_branch_equals_negated_metric(frame_name: str) -> None:
    """``square_objective_score(..., "pearson") == -pearson_shape_metric(...)[0]``."""
    frame = _good_frame() if frame_name == "good" else _bad_frame()
    loss, _energy = pearson_shape_metric(frame, CENTER, "square", float(SIDE), 1.0)

    score = square_objective_score(
        frame, 0.4, 0.5, 0.9, CENTER, SIDE, objective="pearson"
    )

    assert score == pytest.approx(-loss, rel=1e-12, abs=1e-15)
    # ... and explicitly *not* the un-negated loss. Only meaningful when the loss
    # is far from 0 (for the perfect square the loss is ~1e-14, where the two
    # signs are numerically indistinguishable).
    if abs(loss) > 0.1:
        assert score != pytest.approx(loss, rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------------------
# 6. unknown objectives raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "objective",
    ["not-an-objective", "rmse", "pib", "shape_score", "quality "],
    ids=lambda value: repr(value),
)
def test_unknown_objective_raises_value_error(objective: str) -> None:
    """Anything outside the vocabulary (and its documented aliases) raises."""
    with pytest.raises(ValueError, match="objective must be one of"):
        square_objective_score(
            _good_frame(), 0.1, 0.9, 1.0, CENTER, SIDE, objective=objective
        )


def test_value_error_names_the_vocabulary() -> None:
    """The error text lists the valid choices so the CLI help cannot drift."""
    with pytest.raises(ValueError) as excinfo:
        square_objective_score(
            _good_frame(), 0.1, 0.9, 1.0, CENTER, SIDE, objective="not-an-objective"
        )

    message = str(excinfo.value)
    assert "not-an-objective" in message
    for choice in CANONICAL_CHOICES:
        assert choice in message


# ---------------------------------------------------------------------------
# 7. the quality branch is unchanged, and its [0, 1] contract is not negated
# ---------------------------------------------------------------------------


def test_square_quality_score_unit_interval_contract() -> None:
    """The historical quality score is still ``1.0`` at best and ``>= 0.0``."""
    assert square_quality_score(0.0, 1.0, 1.0) == pytest.approx(1.0)

    worst = square_quality_score(50.0, 0.0, 100.0)
    expected_worst = (
        0.4 * float(np.exp(-100.0)) + 0.0 + 0.2 * float(np.exp(-3.0 * 99.0))
    )
    assert worst == pytest.approx(expected_worst, rel=1e-9)
    assert 0.0 <= worst < 1e-6


@pytest.mark.parametrize(
    "cv,ee,ar",
    list(itertools.product(CV_SAMPLES, EE_SAMPLES, AR_SAMPLES)),
    ids=lambda value: f"{value:.3f}",
)
def test_quality_branch_stays_in_unit_interval(cv: float, ee: float, ar: float) -> None:
    """``"quality"`` through the dispatcher stays in [0, 1] over the domain."""
    frame = np.full(FRAME_SHAPE, ee * 100.0)  # energy-fraction-consistent frame
    score = square_objective_score(frame, cv, ee, ar, CENTER, SIDE, objective="quality")

    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(square_quality_score(cv, ee, ar), rel=1e-12)


def test_dispatcher_does_not_negate_the_quality_score() -> None:
    """The ``"quality"`` branch forwards the historical function verbatim.

    A negation here (or anywhere upstream of it) would return a value in
    ``[-1, 0]`` and silently reverse the direction of every existing square run.
    """
    frame = _bad_frame()
    cv, ee = _cv_ee(frame)

    via_dispatcher = square_objective_score(
        frame, cv, ee, 1.0, CENTER, SIDE, objective="quality"
    )
    direct = square_quality_score(cv, ee, 1.0)

    assert via_dispatcher == direct  # exact, not negated, not re-scaled
    assert via_dispatcher == pytest.approx(
        0.4 * float(np.exp(-2.0 * cv)) + 0.4 * ee + 0.2
    )
    assert via_dispatcher > 0.0, "a quality score must be positive, not negated"


@pytest.mark.parametrize(
    "w_cv,w_ee,w_ar", [(0.4, 0.4, 0.2), (0.6, 0.3, 0.1), (0.0, 1.0, 0.0)]
)
def test_dispatcher_forwards_custom_weights(
    w_cv: float, w_ee: float, w_ar: float
) -> None:
    """Explicit ``w_cv``/``w_ee``/``w_ar`` reach the historical function."""
    frame = _bad_frame()
    cv, ee = _cv_ee(frame)

    assert square_objective_score(
        frame,
        cv,
        ee,
        1.0,
        CENTER,
        SIDE,
        objective="quality",
        w_cv=w_cv,
        w_ee=w_ee,
        w_ar=w_ar,
    ) == pytest.approx(square_quality_score(cv, ee, 1.0, w_cv, w_ee, w_ar), rel=1e-12)


@pytest.mark.parametrize("alias", ["quality", "shape", "", "QUALITY", "Quality"])
def test_quality_aliases_are_byte_identical(alias: str) -> None:
    """The documented ``"quality"`` aliases all return the exact same float."""
    frame = _good_frame()
    cv, ee = _cv_ee(frame)

    assert square_objective_score(
        frame, cv, ee, 1.0, CENTER, SIDE, objective=alias
    ) == square_objective_score(frame, cv, ee, 1.0, CENTER, SIDE, objective="quality")


# ---------------------------------------------------------------------------
# 8. the vocabulary is exported from both the canonical and the legacy path
# ---------------------------------------------------------------------------


def test_square_objective_choices_exported_from_both_paths() -> None:
    """Canonical package and legacy shim expose the same tuple."""
    assert CANONICAL_CHOICES == SHIM_CHOICES
    assert CANONICAL_CHOICES == ("quality", "pearson")
    assert isinstance(CANONICAL_CHOICES, tuple)
    assert all(isinstance(choice, str) for choice in CANONICAL_CHOICES)


def test_square_objective_choices_reexported_by_the_optimizer() -> None:
    """The optimizer module re-exports the same vocabulary it validates against."""
    from ao_shaping.optimizer.wfless.slm_square_shaping import (
        SQUARE_OBJECTIVE_CHOICES as OPTIMIZER_CHOICES,
    )

    assert OPTIMIZER_CHOICES == CANONICAL_CHOICES


def test_optimizer_config_rejects_objective_outside_the_vocabulary() -> None:
    """``optimize_slm_square`` validates the config objective, not just the CLI.

    The check happens in the optimizer, before any device is opened, so an
    out-of-vocabulary value reaching the config surface fails fast rather than
    silently running with the default.
    """
    from ao_shaping.optimizer.wfless.slm_square_shaping import optimize_slm_square

    for bogus in ("not-an-objective", "rmse", ""):
        with pytest.raises(ValueError, match="objective must be one of"):
            optimize_slm_square(
                center=CENTER, epochs=1, config=SlmSquareConfig(objective=bogus)
            )
