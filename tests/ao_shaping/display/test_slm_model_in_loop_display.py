"""``slm-model-in-loop --show``: the shared four-panel pygame live view.

The runner borrows the ``slm-pib`` live-view machinery rather than
reimplementing it, so these tests pin the *sharing* (identical panel
primitives, identical window geometry, identical ``closed`` contract) and the
one deliberate difference: the Zernike coefficient bars are replaced by the
forward model's predicted far field.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from ao_shaping.display import SlmModelInLoopDisplay, SlmZernikeDisplay
from ao_shaping.optimizer.wfless.slm_model_in_loop import _predicted_far_field
from ao_shaping.runners import runner_common
from ao_shaping.runners.runner_common import SlmModelInLoopParams
from ao_shaping.runners.slm.model_in_loop_runner import run as mil_run

#: The golden lives with the CLI contract tests, not with the display tests.
GOLDEN = pathlib.Path(__file__).parents[1] / "runners" / "cli_help_golden.json"

#: The four panels the feature is specified to show, in window order.
EXPECTED_PANELS = ["ccd", "phase", "predicted", "curve"]

#: The PIB layout must keep its coefficient-bar panel; the model-in-loop layout
#: must not grow one. Guards against the subclass mutating shared state.
PIB_PANELS = ["ccd", "phase", "coeff", "curve"]


# ---------------------------------------------------------------------------
# CLI wiring: the flag is shared with slm-pib / spgd-square
# ---------------------------------------------------------------------------


def test_show_flag_is_declared_on_the_command() -> None:
    """`--show` must exist on the click command, exactly once."""
    matches = [p for p in mil_run.params if "--show" in p.opts]
    assert len(matches) == 1, f"expected exactly one --show, got {matches}"
    assert matches[0].is_flag, "--show must be a boolean flag, not a value option"


def test_show_defaults_to_off() -> None:
    """Off by default: opening a window is opt-in, never implicit."""
    assert SlmModelInLoopParams().show is False


def test_show_flag_is_shared_with_the_other_slm_runners() -> None:
    """The flag is spelled exactly as everywhere else, so scripts and muscle
    memory carry over unchanged.

    Asserted through ``with_params`` -- the same collector the runner uses --
    because the ``Annotated`` metadata is a deferred click builder that only
    materialises into real options at collection time.
    """
    import click

    from ao_shaping.utils.cli.params import with_params

    def flags(dataclass) -> list[str]:
        @click.command()
        @with_params(dataclass, kw_name="params")
        def _command(params):  # pragma: no cover - never invoked
            pass

        return [opt for param in _command.params for opt in param.opts]

    for name in ("SpgdParams", "HeuristicParams", "SlmSquareParams"):
        assert "--show" in flags(getattr(runner_common, name)), (
            f"{name} is expected to expose the same --show flag"
        )
    assert "--show" in flags(SlmModelInLoopParams)


def test_config_accepts_and_defaults_show() -> None:
    """The optimizer config is the single source of truth for the runner."""
    from ao_shaping.optimizer.wfless.slm_model_in_loop import SlmModelInLoopConfig

    assert SlmModelInLoopConfig().show is False
    assert SlmModelInLoopConfig(show=True).show is True


def test_golden_help_contains_the_show_flag() -> None:
    """The frozen --help contract was updated for the new flag.

    Without this the option exists but the golden drifts silently; with it the
    golden's rendering of ``--show`` is pinned too.
    """
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    text = data["slm-model-in-loop"]
    assert "--show " in text or text.rstrip().endswith("--show"), "flag not rendered"
    assert "PREDICTED" in text, "the help must state the predicted-far-field panel"


# ---------------------------------------------------------------------------
# Display: shared machinery, one substituted panel
# ---------------------------------------------------------------------------


def test_display_is_the_pib_window_so_the_show_command_is_shared() -> None:
    """Subclassing the PIB display is what makes this a *shared* show command
    rather than a second implementation that will drift."""
    assert issubclass(SlmModelInLoopDisplay, SlmZernikeDisplay)


def test_display_panel_names_and_classes() -> None:
    """Exactly the four specified panels, built from the shared primitives."""
    display = SlmModelInLoopDisplay()
    assert [f.name for f in display.frame_list] == EXPECTED_PANELS
    assert [f.frame for f in display.frame_list] == [
        "Image2DWithBucketFrame",  # measured CCD + target overlay
        "Image2DFrame",  # phase on the SLM
        "Image2DFrame",  # predicted far field
        "EpochCurveFrame",  # metrics log curve
    ]


def test_display_reuses_the_pib_window_geometry() -> None:
    """Same 2x2 grid and panel size as the PIB view, so both windows look alike."""
    display = SlmModelInLoopDisplay()
    pib = SlmZernikeDisplay()
    assert display.grid == pib.grid == (2, 2)
    assert display.frame_size == pib.frame_size
    assert display.total_size == pib.total_size


def test_display_does_not_mutate_the_pib_layout() -> None:
    """Constructing the model-in-loop view must leave the PIB view alone.

    ``SlmModelInLoopDisplay`` rebinds ``self.frame_list`` after ``super().__init__``;
    if that ever leaked to the class or the base instance, ``slm-pib --show``
    would silently lose its coefficient bars.
    """
    pib = SlmZernikeDisplay()
    _ = SlmModelInLoopDisplay()
    assert [f.name for f in SlmZernikeDisplay().frame_list] == PIB_PANELS
    assert [f.name for f in pib.frame_list] == PIB_PANELS
    assert "predicted" not in PIB_PANELS


def test_display_target_defaults_to_square() -> None:
    """The model-in-loop objective is a square, so that is the overlay default."""
    display = SlmModelInLoopDisplay()
    assert display.target_shape == "square"
    assert display.closed is False


# ---------------------------------------------------------------------------
# The predicted-far-field helper
# ---------------------------------------------------------------------------


class _FakeOptimizer:
    """Stands in for ``ZernikeCoefficientOptimizer`` so no torch is needed."""

    def __init__(self, result=None, error: BaseException | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple] = []

    def forward_intensity(self, coefficients, phase_slm, source_amplitude=None):
        self.calls.append((coefficients, phase_slm, source_amplitude))
        if self.error is not None:
            raise self.error
        return self.result


def test_predicted_far_field_forwards_exactly_the_model_inputs() -> None:
    """It must drive the canonical forward model, not re-derive a far field.

    The panel therefore cannot disagree with the loss Step A optimises.
    """
    coeffs = np.array([0.1, -0.2, 0.3])
    phase = np.zeros((8, 8))
    amplitude = np.ones((8, 8))
    raw = np.arange(64, dtype=np.float32).reshape(8, 8)

    opt = _FakeOptimizer(result=raw)
    out = _predicted_far_field(opt, coeffs, phase, amplitude)

    assert out is not None
    assert out.dtype == np.float64, "display array must be float64"
    np.testing.assert_allclose(out, raw)
    assert len(opt.calls) == 1
    got_c, got_p, got_a = opt.calls[0]
    np.testing.assert_allclose(got_c, coeffs)
    np.testing.assert_allclose(got_p, phase)
    np.testing.assert_allclose(got_a, amplitude)


def test_predicted_far_field_degrades_instead_of_aborting_a_hardware_run() -> None:
    """Diagnostics must never kill a run that already spent probe acquisitions.

    The flat baseline is displayed before any fit exists, and a model/torch
    failure at render time must leave the run going with a blank panel.
    """
    for error in (ValueError("bad shape"), RuntimeError("backend blew up")):
        opt = _FakeOptimizer(error=error)
        assert _predicted_far_field(opt, np.zeros(3), np.zeros((8, 8)), np.ones((8, 8))) is None


def test_predicted_far_field_does_not_hide_unexpected_bugs() -> None:
    """Only the two documented failure modes are absorbed; a real bug surfaces."""
    opt = _FakeOptimizer(error=TypeError("wrong arg type"))
    with pytest.raises(TypeError):
        _predicted_far_field(opt, np.zeros(3), np.zeros((8, 8)), np.ones((8, 8)))


# ---------------------------------------------------------------------------
# End-to-end render, against a real pygame window
# ---------------------------------------------------------------------------


def test_all_four_panels_render_into_a_real_window() -> None:
    """Drive the display exactly as the optimizer does, on a dummy SDL driver.

    Guards the whole payload contract at once: every panel must actually receive
    and draw an image, the curve must gain a point per *round* but not per
    intra-round probe frame, and closing must stop further updates.
    """
    pygame = pytest.importorskip("pygame", reason="pygame not installed")
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    region, padding, cam = 32, 4, 120
    ff = region * padding  # far_field_size == region * far_field_padding

    def measured(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:cam, 0:cam]
        r = np.hypot(xx - 60, yy - 58)
        return np.exp(-((r / 12.0) ** 2)) * (0.55 + 0.45 * rng.random((cam, cam))) * 255.0

    def predicted(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        c = ff // 2
        r = np.hypot(*np.mgrid[0:ff, 0:ff][::-1])
        return np.exp(-((r / (ff / 20.0)) ** 2)) * (0.55 + 0.45 * rng.random((ff, ff)))

    display = SlmModelInLoopDisplay()
    display.init_window()
    try:
        assert (display.n_cols, display.n_rows) == (2, 2)
        assert sorted(display._frames) == sorted(EXPECTED_PANELS)

        def panel_sum(name: str) -> int:
            frame = display._frames[name]
            surf = display.window.subsurface(
                pygame.Rect(frame.left, frame.top, frame.width, frame.height)
            )
            return int(pygame.surfarray.array3d(surf).sum())

        # Flat baseline: no prediction exists yet, so the panel must be blank
        # rather than showing a fabricated or stale fit.
        assert display.update(
            measured=measured(0),
            phase=np.zeros((region, region)),
            predicted=None,
            center=(60, 58),
            r=20.0,
            info="flat baseline | score 0.4123",
            value=0.4123,
            epoch=0,
            total_epochs=3,
            target_size=40,
        )
        assert panel_sum("predicted") == 0, "predicted panel must stay blank"

        # Step A probe frames: real predictions, but no curve points.
        curve = display._frames["curve"].data
        points_before = len(curve["value"])
        for k in range(2):
            assert display.update(
                measured=measured(k + 1),
                phase=np.random.default_rng(k).normal(0, 3.0, (region, region)),
                predicted=predicted(k),
                center=(60, 58),
                r=20.0,
                info=f"round 1/3 step A probe {k + 1}/4",
                target_size=40,
            )
        assert len(display._frames["curve"].data["value"]) == points_before, (
            "intra-round probe frames must not add curve points"
        )

        # Round outcome: a curve point, and every panel drawn.
        assert display.update(
            measured=measured(5),
            phase=np.random.default_rng(9).normal(0, 1.5, (region, region)),
            predicted=predicted(9),
            center=(60, 58),
            r=20.0,
            info="round 1/3 accepted | score 0.5510 | loss 0.9->0.2",
            value=0.5510,
            epoch=1,
            total_epochs=3,
            target_size=40,
        )
        curve = display._frames["curve"].data
        assert len(curve["value"]) == 2, f"expected 2 curve points, got {len(curve['value'])}"
        assert curve["epoch"] == [0.0, 1.0]

        for name in EXPECTED_PANELS:
            assert panel_sum(name) > 0, f"panel {name!r} rendered nothing"
    finally:
        display.close()

    assert display.closed
    assert (
        display.update(measured=measured(0), phase=np.zeros((region, region)))
        is False
    ), "a closed window must refuse further updates so the loop can stop"


# ---------------------------------------------------------------------------
# The optimizer's call sites, checked without importing torch
# ---------------------------------------------------------------------------


def _live_display_update_calls() -> list[ast.Call]:
    """Every ``live_display.update(...)`` call in the optimizer, by AST.

    Parsed rather than executed on purpose: the optimizer body imports torch, which
    the CLI contract test already proves is unavailable in some environments, and a
    signature typo in these three call sites would otherwise only surface on a
    hardware run.
    """
    import ast
    import inspect

    import ao_shaping.optimizer.wfless.slm_model_in_loop as module

    tree = ast.parse(inspect.getsource(module))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "live_display"
    ]


def test_optimizer_calls_update_with_only_real_parameter_names() -> None:
    """Each of the optimizer's render points must bind to the real signature.

    Keyword typos are invisible to lsp on an unexecuted path and would raise
    ``TypeError`` mid-run -- on hardware, after minutes of probe acquisition.
    """
    import inspect

    calls = _live_display_update_calls()
    assert len(calls) == 3, (
        f"expected 3 render points (flat baseline, Step A probe, round outcome), "
        f"found {len(calls)}"
    )

    parameters = set(inspect.signature(SlmModelInLoopDisplay.update).parameters)
    assert "self" in parameters
    parameters.discard("self")

    for node in calls:
        used = {kw.arg for kw in node.keywords if kw.arg is not None}
        unknown = used - parameters
        assert not unknown, f"update() called with unknown keyword(s): {sorted(unknown)}"
        # ``measured`` and ``phase`` are the only two required arguments, so they
        # are exactly the ones that must be passed by keyword at every call site.
        assert {"measured", "phase"} <= used, sorted(used)


def test_optimizer_guards_every_render_point_on_the_display_handle() -> None:
    """``live_display`` is ``None`` unless ``--show``, so each use must be guarded.

    Unguarded use would crash every default (non-``--show``) run at the first
    render point, which is the overwhelmingly common case.
    """
    import inspect

    import ao_shaping.optimizer.wfless.slm_model_in_loop as module

    source = inspect.getsource(module.optimize_slm_model_in_loop)
    guarded = source.count("if live_display is not None")
    assert guarded >= 4, (
        "expected a None-guard for the baseline, the probe loop, the round "
        f"outcome and the close check; found {guarded}"
    )
    assert "live_display.close()" in source, "the window must be torn down"