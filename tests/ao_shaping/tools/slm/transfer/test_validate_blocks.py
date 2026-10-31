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
    # The defect was the call site, not the predicate: pin that the session body
    # consulted by ``run`` calls ``skips_shape``, so a future edit cannot quietly
    # restore ``if not blocks``. Since #205 batch 6 the session block lives in
    # ``_run_validate_session`` -- pin both hops: run must call the session body,
    # the session body must call the gate.
    run_source = inspect.getsource(validate.run)
    session_source = inspect.getsource(validate._run_validate_session)
    assert "_run_validate_session(" in run_source
    assert "skips_shape(resume=resume, blocks=blocks)" in session_source
