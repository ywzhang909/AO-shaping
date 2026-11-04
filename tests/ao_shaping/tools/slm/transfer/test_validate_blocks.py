"""The resume gate must not swallow a seed-only pass.

``validate.run`` serves two callers: a shaping stage (``blocks > 0``) and the
seed bake-off, which asks for ``blocks=0, updates=0`` because it only needs the
seed and final rows to compare initial-phase constructions. Both need a
``manifest.json``.

The 2026-10-29 campaign died on exactly this: a resume gate added in
``c1c7aeed`` skipped the shape whenever the block list was empty, so the bake-off
wrote only ``config.json`` and every square/triangle/star lane raised
``FileNotFoundError: .../seed_bakeoff/<shape>_<kind>/manifest.json``.
"""
from __future__ import annotations

import inspect

from ao_shaping.tools.slm.transfer import validate


def test_a_seed_only_pass_is_not_skipped_without_resume():
    # This is the bake-off: zero optimization blocks, but the seed row is the
    # measurement being asked for, so the shape must still be entered.
    assert validate.skips_shape(resume=False, blocks=[]) is False


def test_a_normal_stage_is_not_skipped():
    assert validate.skips_shape(resume=False, blocks=[0, 1, 2]) is False
    assert validate.skips_shape(resume=True, blocks=[2, 3]) is False


def test_only_a_resume_with_nothing_outstanding_is_skipped():
    assert validate.skips_shape(resume=True, blocks=[]) is True
    assert validate.skips_shape(resume=True, blocks=()) is True


def test_run_uses_the_gate_and_not_a_bare_empty_check():
    # The defect was the call site, not the predicate: pin that ``run`` consults
    # ``skips_shape``, so a future edit cannot quietly restore ``if not blocks``.
    # The session body and shape loop both live inline in ``run`` (the earlier
    # ``_run_validate_session`` / ``_run_validate_shapes`` helpers are gone), so
    # the whole chain is pinned against ``run``'s own source.
    run_source = inspect.getsource(validate.run)
    assert "skips_shape(resume=resume, blocks=blocks)" in run_source
    # The resume offset must feed the gate from ``remaining_blocks`` -- the gate
    # is consulted on the *outstanding* block list, never the raw recipe range.
    assert "remaining_blocks(" in run_source
    assert "if not blocks" not in run_source


def test_run_opens_both_devices_through_the_retry_wrapper():
    run_source = inspect.getsource(validate.run)
    assert "with opened_device(create_camera(" in run_source
    assert "with opened_device(Santec(" in run_source
