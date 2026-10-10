"""Verify the opt-in Adaptive SPGD ``--adaptive`` closed loop for ``spgd-square``.

Covers three layers:

* :class:`SlmSquareConfig` — the four new fields default to the documented
  values and the three new validations reject out-of-range input.
* ``slm_shaping_runner._build_square_config`` — the four fields are threaded
  from the CLI params dataclass into the optimizer config. This is the
  canonical ``spgd-square`` runner; ``gsnet_runner`` has its own unrelated
  ``_build_square_config`` (different signature) and must NOT be used for
  ``spgd-square`` tests.
* the ``square`` click command — the four flags are real CLI options, and an
  ``--adaptive`` run through the (patched) optimizer exercises the loop.

No hardware is opened: ``optimize_slm_square`` is monkey-patched, so the
invocation path only touches pure functions (config building, sim patch
installation, banner echo).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from ao_shaping.optimizer.wfless.slm_square_shaping import SlmSquareConfig
from ao_shaping.runners.runner_common import SlmSquareParams, ZernikeSlmParams
from ao_shaping.runners.slm.slm_shaping_runner import _build_square_config, square


def _invoke(args: list[str], obj: dict | None = None):
    """Invoke the ``square`` command with ``optimize_slm_square`` patched out.

    Mirrors ``test_slm_square_runner_defaults.py``: the runner binds
    ``optimize_slm_square`` at module scope, so that is where the name must be
    patched (not in the optimizer module).
    """
    runner = CliRunner()
    captured: dict = {}

    def _fake_optimize(**kwargs):
        captured.update(kwargs)
        from ao_shaping.utils.io.file import Recorder

        rec = Recorder(mark="quality", mode="max")
        rec.append({"quality": 1.0, "_epoch": 0})
        return rec

    with patch(
        "ao_shaping.runners.slm.slm_shaping_runner.optimize_slm_square",
        side_effect=_fake_optimize,
    ):
        result = runner.invoke(
            square, args, obj={"dir": "data"} if obj is None else obj
        )
    return result, captured


class TestSlmSquareAdaptiveConfigDefaults:
    def test_adaptive_defaults(self):
        config = SlmSquareConfig()
        assert config.adaptive is False
        assert config.stagnate_win == 10
        assert config.stagnate_boost == pytest.approx(1.5)
        assert config.min_improve_frac == pytest.approx(0.01)


class TestSlmSquareAdaptiveConfigValidation:
    def test_stagnate_win_below_1_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            SlmSquareConfig(stagnate_win=0)
        assert str(excinfo.value) == "stagnate_win must be at least 1, got 0"

    def test_stagnate_boost_below_1_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            SlmSquareConfig(stagnate_boost=0.9)
        assert str(excinfo.value) == "stagnate_boost must be >= 1.0, got 0.9"

    def test_min_improve_frac_negative_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            SlmSquareConfig(min_improve_frac=-0.1)
        assert (
            str(excinfo.value) == "min_improve_frac must be in [0, 1), got -0.1"
        )

    def test_min_improve_frac_at_1_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            SlmSquareConfig(min_improve_frac=1.0)
        assert str(excinfo.value) == "min_improve_frac must be in [0, 1), got 1.0"

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"stagnate_win": 1}, (1, 1.5, 0.01)),
            ({"stagnate_boost": 1.0}, (10, 1.0, 0.01)),
            ({"min_improve_frac": 0.0}, (10, 1.5, 0.0)),
        ],
    )
    def test_boundary_values_accepted(self, kwargs, expected):
        config = SlmSquareConfig(**kwargs)
        assert config.stagnate_win == expected[0]
        assert config.stagnate_boost == pytest.approx(expected[1])
        assert config.min_improve_frac == pytest.approx(expected[2])


class TestBuildSquareConfigThreadsAdaptive:
    def test_defaults_thread_through(self):
        config = _build_square_config(SlmSquareParams(), ZernikeSlmParams(), None)
        assert config.adaptive is False
        assert config.stagnate_win == 10
        assert config.stagnate_boost == pytest.approx(1.5)
        assert config.min_improve_frac == pytest.approx(0.01)

    def test_explicit_values_thread_through(self):
        params = SlmSquareParams(
            adaptive=True,
            stagnate_win=5,
            stagnate_boost=2.0,
            min_improve_frac=0.05,
        )
        config = _build_square_config(params, ZernikeSlmParams(), None)
        assert config.adaptive is True
        assert config.stagnate_win == 5
        assert config.stagnate_boost == pytest.approx(2.0)
        assert config.min_improve_frac == pytest.approx(0.05)


class TestSlmSquareAdaptiveCli:
    def test_help_lists_adaptive_flags(self):
        result, _ = _invoke(["--help"])
        assert result.exit_code == 0, result.output
        for flag in (
            "--adaptive",
            "--stagnate-win",
            "--stagnate-boost",
            "--min-improve-frac",
        ):
            assert flag in result.output, flag
        assert "自适应 SPGD" in result.output

    def test_defaults_reach_config(self):
        result, captured = _invoke(["-e", "1"])
        assert result.exit_code == 0, result.output
        config = captured["config"]
        assert config.adaptive is False
        assert config.stagnate_win == 10
        assert config.stagnate_boost == pytest.approx(1.5)
        assert config.min_improve_frac == pytest.approx(0.01)

    def test_adaptive_flags_reach_config(self):
        result, captured = _invoke(
            [
                "-e",
                "1",
                "--adaptive",
                "--stagnate-win",
                "5",
                "--stagnate-boost",
                "2.0",
                "--min-improve-frac",
                "0.05",
            ]
        )
        assert result.exit_code == 0, result.output
        config = captured["config"]
        assert config.adaptive is True
        assert config.stagnate_win == 5
        assert config.stagnate_boost == pytest.approx(2.0)
        assert config.min_improve_frac == pytest.approx(0.05)