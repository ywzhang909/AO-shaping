"""Offline unit tests for the issue #145 exit-safety hardware harness.

``scripts/slm/verify_pib_exit_safety_hw.py`` runs on the instruments, so the
suite cannot execute it end to end. What *can* be pinned offline is every
decision it makes from evidence it has already collected:

* the write log -- which write is the search's, which is the cleanup's, and
  whether the cleanup rotated to a different memory slot (writing the slot the
  LCOS already shows is a firmware no-op, which would make "restored the best
  phase" a false claim);
* the firmware readback -- a named memory slot, or direct-grayscale mode with
  gray 0, and nothing else counts as "a state we can name";
* the optical comparison -- the live frame has to match one of the written
  states at the repeat-frame noise floor;
* the exception that reached the caller, and the cleanup duration.

Those are pure functions of dicts and arrays, so they are tested here with
synthetic evidence. No camera, no SLM, no ``libs/``.

What is *not* covered here, on purpose, is anything that only the real panel
can answer; that is what the hardware run in
``report/slm/pib_exit_safety/`` is for.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from scripts.slm import verify_pib_exit_safety_hw as hw

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _scenario(**overrides: Any) -> hw.Scenario:
    base: dict[str, Any] = {
        "key": "A",
        "name": "A_unit",
        "algorithm": "spgd",
        "epochs": 2,
    }
    base.update(overrides)
    return hw.Scenario(**base)


def _bench() -> hw.Bench:
    """A Bench with no device behind it -- only the write log is exercised."""
    return hw.Bench(slm=None)


def _write(
    index: int,
    kind: str = "memory",
    slot: int | None = None,
    gray_value: int | None = None,
) -> dict[str, Any]:
    return {
        "i": index,
        "kind": kind,
        "slot": slot,
        "t": float(index),
        "min": 0 if kind == "memory" else None,
        "max": 993 if kind == "memory" else None,
        "gray_value": gray_value,
    }


def _checks(**overrides: Any) -> dict[str, Any]:
    """Call ``_checks`` with a passing default evidence set."""
    kwargs: dict[str, Any] = {
        "scenario": _scenario(interrupt_at_write=3),
        "outcome": "raised",
        "exc_type": "KeyboardInterrupt",
        "exc_message": "",
        "firmware": {"displayed_slot": 60, "current_grayscale": None},
        "exit_write": _write(11, slot=60),
        "writes": [_write(9, slot=58), _write(10, slot=59), _write(11, slot=60)],
        "corr": {
            "live_vs_flat": 0.20,
            "live_vs_exit": 0.9990,
            "live_vs_last_iterate": 0.40,
            "flat_repeat": 0.9985,
        },
        "cleanup_s": 0.8,
        "energy_loss_violations_present": True,
        "row0": {"pib": 0.159, "max_brt": 24.0},
    }
    kwargs.update(overrides)
    return hw._checks(**kwargs)


# ---------------------------------------------------------------------------
# Scenario / Bench: which write is which
# ---------------------------------------------------------------------------


def test_scenario_injects_only_when_a_trigger_is_set() -> None:
    assert _scenario(interrupt_at_write=3).injects
    assert _scenario(dropout_at_capture=5).injects
    assert not _scenario().injects


def test_exit_write_is_the_last_write_when_nothing_was_injected() -> None:
    bench = _bench()
    bench.writes = [_write(0, slot=5), _write(1, slot=6)]
    assert bench.exit_write()["slot"] == 6
    assert bench.writes_after_inject() == []


def test_exit_write_is_the_last_cleanup_write_after_injection() -> None:
    bench = _bench()
    bench.writes = [_write(0, slot=5), _write(1, slot=6)]
    for index, write in enumerate(bench.writes):
        write["t"] = float(index)
    bench.inject_t = 1.5
    bench.writes.append(_write(2, slot=7))

    assert [w["i"] for w in bench.writes_before_inject()] == [0, 1]
    assert [w["i"] for w in bench.writes_after_inject()] == [2]
    assert bench.exit_write()["slot"] == 7


def test_exit_write_is_none_when_the_cleanup_never_wrote() -> None:
    bench = _bench()
    bench.writes = [_write(0, slot=5)]
    bench.writes[0]["t"] = 0.0
    bench.inject_t = 1.0
    assert bench.exit_write() is None


def test_last_search_write_ignores_grayscale_writes() -> None:
    bench = _bench()
    bench.writes = [
        _write(0, slot=5),
        _write(1, "grayscale", gray_value=0),
        _write(2, slot=6),
        _write(3, "grayscale", gray_value=0),
    ]
    assert bench.last_search_write()["slot"] == 6


# ---------------------------------------------------------------------------
# Optics
# ---------------------------------------------------------------------------


def test_normalised_correlation_is_invariant_to_gain_and_offset() -> None:
    rng = np.random.default_rng(3)
    reference = rng.random((48, 48))
    scaled = 3.0 * reference + 50.0
    assert hw._corr(reference, scaled) == pytest.approx(1.0, abs=1e-9)


def test_correlation_drops_for_independent_frames() -> None:
    rng = np.random.default_rng(4)
    assert abs(hw._corr(rng.random((48, 48)), rng.random((48, 48)))) < 0.25


def test_correlation_is_nan_for_a_flat_frame() -> None:
    flat = np.full((16, 16), 7.0)
    assert np.isnan(hw._corr(flat, np.random.default_rng(5).random((16, 16))))
    assert np.isnan(hw._corr(flat, flat))


# ---------------------------------------------------------------------------
# Firmware readback
# ---------------------------------------------------------------------------


def test_memory_exit_is_named_only_when_the_readback_slot_matches() -> None:
    assert _checks()["firmware_is_named_state"] is True
    mismatch = _checks(firmware={"displayed_slot": 58, "current_grayscale": None})
    assert mismatch["firmware_is_named_state"] is False


def test_grayscale_exit_is_named_only_at_gray_zero() -> None:
    exit_write = _write(11, "grayscale", gray_value=0)
    ok = _checks(
        firmware={"displayed_slot": None, "current_grayscale": 0},
        exit_write=exit_write,
    )
    assert ok["firmware_is_named_state"] is True
    assert ok["firmware_left_memory_mode"] is True

    wrong = _checks(
        firmware={"displayed_slot": None, "current_grayscale": 7},
        exit_write=_write(11, "grayscale", gray_value=7),
    )
    assert wrong["firmware_is_named_state"] is False


def test_no_exit_write_can_never_name_a_state() -> None:
    checks = _checks(exit_write=None, firmware={"displayed_slot": 60})
    assert checks["firmware_is_named_state"] is False
    assert checks["firmware_mode"] == "unknown"


# ---------------------------------------------------------------------------
# Slot rotation -- the firmware no-op trap
# ---------------------------------------------------------------------------


def test_exit_write_into_the_displayed_slot_fails_the_fresh_slot_check() -> None:
    # The last memory write before the cleanup already targeted slot 60, so the
    # cleanup's own write to slot 60 would not have refreshed the panel.
    checks = _checks(
        exit_write=_write(11, slot=60),
        writes=[_write(10, slot=60), _write(11, slot=60)],
    )
    assert checks["exit_used_a_fresh_slot"] is False
    assert checks["exit_previous_slot"] == 60


def test_rotating_to_a_different_slot_passes() -> None:
    checks = _checks(
        exit_write=_write(11, slot=60),
        writes=[_write(10, slot=59), _write(11, slot=60)],
    )
    assert checks["exit_used_a_fresh_slot"] is True


def test_fresh_slot_check_does_not_apply_to_a_grayscale_exit() -> None:
    checks = _checks(
        firmware={"displayed_slot": None, "current_grayscale": 0},
        exit_write=_write(11, "grayscale", gray_value=0),
    )
    assert "exit_used_a_fresh_slot" not in checks


# ---------------------------------------------------------------------------
# Exception survival and timings
# ---------------------------------------------------------------------------


def test_a_cleanup_exception_replacing_the_original_is_caught() -> None:
    checks = _checks(exc_type="RuntimeError")
    assert checks["original_exception_survived_cleanup"] is False
    assert checks["failure_reached_caller"] is True


def test_a_returning_run_is_not_scored_as_a_failure() -> None:
    checks = _checks(
        scenario=_scenario(),
        outcome="returned",
        exc_type=None,
        exc_message=None,
        cleanup_s=None,
        firmware={"displayed_slot": 60},
    )
    assert checks["run_completed"] is True
    assert "failure_reached_caller" not in checks
    # Not applicable, not a failure: there is no injection to time.
    assert checks["cleanup_under_5s"] is None


def test_cleanup_over_five_seconds_fails_the_gate() -> None:
    assert _checks(cleanup_s=5.5)["cleanup_under_5s"] is False


def test_epoch0_gate_uses_the_measured_metric_levels() -> None:
    assert _checks(row0={"pib": 0.159, "max_brt": 24.0})["epoch0_is_sane"] is True
    assert _checks(row0={"pib": 0.01, "max_brt": 24.0})["epoch0_is_sane"] is False
    assert _checks(row0={"pib": 0.159, "max_brt": 3.0})["epoch0_is_sane"] is False
    assert "epoch0_is_sane" not in _checks(row0=None)


# ---------------------------------------------------------------------------
# Optical classification
# ---------------------------------------------------------------------------


def test_live_frame_matching_the_noise_floor_is_a_defined_state() -> None:
    checks = _checks()
    assert checks["optical_is_defined_state"] is True
    assert checks["optical_class"] == "exit"
    assert checks["optical_prefers_exit"] is True


def test_live_frame_matching_nothing_fails_the_optical_gate() -> None:
    checks = _checks(
        corr={
            "live_vs_flat": 0.20,
            "live_vs_exit": 0.30,
            "live_vs_last_iterate": 0.10,
            "flat_repeat": 0.9985,
        }
    )
    assert checks["optical_is_defined_state"] is False
    assert checks["optical_class"] == "exit"


def test_flat_fallback_optics_classify_as_flat() -> None:
    checks = _checks(
        firmware={"displayed_slot": None, "current_grayscale": 0},
        exit_write=_write(11, "grayscale", gray_value=0),
        corr={"live_vs_flat": 0.9988, "flat_repeat": 0.9990},
    )
    assert checks["optical_class"] == "flat"
    assert checks["optical_prefers_exit"] is False
    assert checks["optical_is_defined_state"] is True


# ---------------------------------------------------------------------------
# Verdict plumbing
# ---------------------------------------------------------------------------


def test_every_present_verdict_key_is_a_python_bool() -> None:
    checks = _checks()
    present = [key for key in hw.VERDICT_KEYS if key in checks]
    assert present, "the default evidence set must exercise the gates"
    for key in present:
        assert isinstance(checks[key], bool), f"{key} is not a bool: {checks[key]!r}"


def test_a_pass_has_no_failed_gate_and_a_failure_names_one() -> None:
    passing = _checks()
    failed = _checks(cleanup_s=9.0)
    assert [k for k in hw.VERDICT_KEYS if k in passing and passing[k] is False] == []
    assert failed["cleanup_under_5s"] is False


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


def test_json_default_converts_numpy_and_refuses_unknown_objects() -> None:
    assert hw._json_default(np.int64(3)) == 3
    assert hw._json_default(np.float32(0.5)) == pytest.approx(0.5)
    assert hw._json_default(np.array([1, 2])) == [1, 2]
    with pytest.raises(TypeError):
        hw._json_default(object())


def test_write_summary_persists_numpy_evidence(tmp_path) -> None:
    summary = {
        "issue": 145,
        "scenarios": [
            {
                "key": "A",
                "verdict": "pass",
                "checks": {
                    "optical_noise_floor": np.float64(0.9985),
                    "epoch0_is_sane": np.bool_(True),
                },
            }
        ],
    }
    hw._write_summary(tmp_path, summary)

    reloaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert reloaded["scenarios"][0]["checks"]["optical_noise_floor"] == pytest.approx(0.9985)
    assert reloaded["scenarios"][0]["checks"]["epoch0_is_sane"] is True
