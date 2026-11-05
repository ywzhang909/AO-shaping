"""Per-shape resume requires recorded block verdicts, not just capture data."""

from __future__ import annotations

import json

import pytest

from ao_shaping.tools.slm.transfer.session_plan import remaining_blocks, stage_complete


def test_resume_offset_counts_manifest_verdicts_not_capture_groups(tmp_path):
    shape = tmp_path / "square"
    shape.mkdir()
    # Capture reached block 4, but only the seed and blocks 1--2 have verdicts.
    (shape / "square_groups.json").write_text(json.dumps([{}] * 4), encoding="utf-8")
    (shape / "summary.json").write_text(
        json.dumps([{"block": i} for i in range(4)]), encoding="utf-8"
    )
    (shape / "manifest.json").write_text(
        json.dumps([{"seed": True}, {"block": 0}, {"block": 1}]), encoding="utf-8"
    )

    assert remaining_blocks(shape, 4, resume=True) == [2, 3]
    assert not stage_complete(shape, 4)


def test_capture_groups_without_a_manifest_do_not_skip_validation(tmp_path):
    shape = tmp_path / "square"
    shape.mkdir()
    (shape / "square_groups.json").write_text(json.dumps([{}] * 4), encoding="utf-8")

    assert remaining_blocks(shape, 4, resume=True) == [0, 1, 2, 3]
    assert not stage_complete(shape, 4)


def test_complete_manifest_and_corrupt_manifest_have_distinct_outcomes(tmp_path):
    shape = tmp_path / "square"
    shape.mkdir()
    manifest = shape / "manifest.json"
    manifest.write_text(
        json.dumps([{"seed": True}] + [{"block": i} for i in range(4)]),
        encoding="utf-8",
    )
    assert remaining_blocks(shape, 4, resume=True) == []
    assert stage_complete(shape, 4)

    manifest.write_text("not json", encoding="utf-8")
    assert not stage_complete(shape, 4)
    with pytest.raises(ValueError, match="unreadable manifest"):
        remaining_blocks(shape, 4, resume=True)
