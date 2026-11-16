"""Tests for the additive cosine LR schedule in ``hw_differentiable_phase_opt.py``.

The hardware script gained ``--lr-schedule {flat,cosine}`` / ``--lr-peak`` so the
offline sweep recommendation (cosine @ peak=0.1, see
``report/slm/differentiable_opt/artifacts/lr_sweep_20261116/lr_sweep.md``) can be
validated on hardware. Two contracts matter:

1. ``flat`` (the default) must be bit-identical to the historical behavior: the
   optimizer is *never* touched — no ``set_lr`` call, Adam keeps the ``--lr``
   constant it was constructed with.
2. ``cosine`` must reproduce the sweep formula verbatim,
   ``peak * 0.5 * (1 + cos(pi * i / max(horizon, 1)))`` — no warmup, no floor —
   applied before every ``opt.update()``, and ``cosine`` without a positive
   finite ``--lr-peak`` must be rejected before any device work.

Device-free: only the pure helper, the parser and the ``main()`` guard are
exercised here.
"""
from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "hw_differentiable_phase_opt.py"


def _load_hw_module():
    """Import the script by path (``scripts/`` is not an installable package)."""
    spec = importlib.util.spec_from_file_location(
        "hw_diff_opt_lr_schedule_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hw = _load_hw_module()


class _RecordingOpt:
    """Stand-in optimizer that records every ``set_lr`` value."""

    def __init__(self) -> None:
        self.lrs: list[float] = []

    def set_lr(self, lr: float) -> None:
        self.lrs.append(float(lr))


class _NoTouchOpt:
    """Optimizer without ``set_lr``: any attribute access proves a violation."""

    def __getattr__(self, name: str):  # pragma: no cover - failure path
        raise AssertionError(f"flat schedule must never touch the optimizer ({name})")


def _expected_cosine_lr(peak: float, i: int, horizon: int) -> float:
    """The sweep formula, transcribed verbatim (same op order ⇒ same bits)."""
    return float(peak) * 0.5 * (1.0 + math.cos(math.pi * i / max(int(horizon), 1)))


# ---------------------------------------------------------------------------
# flat: the bit-identical default
# ---------------------------------------------------------------------------


def test_flat_schedule_never_touches_optimizer() -> None:
    opt = _NoTouchOpt()
    for i in range(5):
        out = hw._apply_scheduled_lr(opt, "flat", None, i, 120, "[T]")
        assert out is None


def test_flat_ignores_peak_and_horizon() -> None:
    """Even with a peak given, flat must stay inert (no set_lr, returns None)."""
    opt = _NoTouchOpt()
    assert hw._apply_scheduled_lr(opt, "flat", 0.1, 3, 0, "[T]") is None


# ---------------------------------------------------------------------------
# cosine: formula fidelity
# ---------------------------------------------------------------------------


def test_cosine_matches_sweep_formula_over_full_horizon() -> None:
    opt = _RecordingOpt()
    peak, horizon = 0.1, 120
    for i in range(horizon):
        lr = hw._apply_scheduled_lr(opt, "cosine", peak, i, horizon, "[T]")
        assert lr == _expected_cosine_lr(peak, i, horizon)
    assert opt.lrs == [_expected_cosine_lr(peak, i, horizon) for i in range(horizon)]
    assert len(opt.lrs) == horizon


def test_cosine_endpoints_and_midpoint() -> None:
    opt = _RecordingOpt()
    peak, horizon = 0.1, 120
    lr0 = hw._apply_scheduled_lr(opt, "cosine", peak, 0, horizon, "[T]")
    lrmid = hw._apply_scheduled_lr(opt, "cosine", peak, horizon // 2, horizon, "[T]")
    lrlast = hw._apply_scheduled_lr(opt, "cosine", peak, horizon - 1, horizon, "[T]")
    assert lr0 == peak                      # no warmup: step 0 is exactly peak
    assert lrmid == _expected_cosine_lr(peak, horizon // 2, horizon)
    assert lrmid == pytest.approx(0.05)     # cos(pi/2) ⇒ half peak (to float eps)
    assert 0.0 < lrlast < 1e-4              # no floor, strictly positive tail
    assert opt.lrs[0] > opt.lrs[1] > opt.lrs[2]  # monotonically decreasing


def test_cosine_horizon_zero_does_not_divide_by_zero() -> None:
    opt = _RecordingOpt()
    lr = hw._apply_scheduled_lr(opt, "cosine", 0.2, 0, 0, "[T]")
    assert lr == 0.2  # max(0, 1) ⇒ cos(0) ⇒ peak
    assert opt.lrs == [0.2]


@pytest.mark.parametrize("bad_peak", [None, float("nan"), float("inf"), 0.0, -0.05])
def test_cosine_requires_positive_finite_peak(bad_peak) -> None:
    opt = _RecordingOpt()
    with pytest.raises(ValueError, match="lr-peak"):
        hw._apply_scheduled_lr(opt, "cosine", bad_peak, 0, 120, "[T]")
    assert opt.lrs == []  # rejected before the optimizer is touched


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_parser_defaults_to_flat_without_peak() -> None:
    args = hw.build_parser().parse_args([])
    assert args.lr_schedule == "flat"
    assert args.lr_peak is None


def test_parser_accepts_cosine_with_peak() -> None:
    args = hw.build_parser().parse_args(["--lr-schedule", "cosine", "--lr-peak", "0.1"])
    assert args.lr_schedule == "cosine"
    assert args.lr_peak == pytest.approx(0.1)


def test_parser_rejects_unknown_schedule() -> None:
    with pytest.raises(SystemExit):
        hw.build_parser().parse_args(["--lr-schedule", "step"])


def test_main_guard_cosine_without_peak_exits_before_device_work(monkeypatch) -> None:
    """``main()`` must parser.error (SystemExit 2) before any hardware path."""
    monkeypatch.setattr(sys, "argv", ["hw_differentiable_phase_opt.py", "--lr-schedule", "cosine"])
    with pytest.raises(SystemExit) as excinfo:
        hw.main()
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# run_config recording (source-level: the P7 model_info payload)
# ---------------------------------------------------------------------------


def test_run_config_records_schedule_and_peak() -> None:
    src = inspect.getsource(hw._run_bench)
    assert 'model_info["lr_schedule"] = lr_sched' in src
    assert 'model_info["lr_peak"] = None if lr_peak is None else float(lr_peak)' in src
