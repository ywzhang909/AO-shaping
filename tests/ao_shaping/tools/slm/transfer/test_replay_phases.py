"""Replay selection: every shape of a multi-shape session is replayed."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ao_shaping.tools.slm.transfer.replay_phases import _selections


def _run(tmp_path: Path, name: str, labels: list[str]) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    rows = [dict(label=label, file=f"capture_{index:04d}.npz", metrics=dict(cv=1.0 + index),
                 exposure_ms=50.0) for index, label in enumerate(labels)]
    (directory / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    return directory


def test_selections_cover_every_shape_seed_and_final(tmp_path):
    square_seed = _run(tmp_path, "square_stage", ["square_seed", "square_0_A1", "square_final"])
    triangle_seed = _run(tmp_path, "triangle_stage", ["triangle_seed", "triangle_0_A1",
                                                      "triangle_final"])
    args = SimpleNamespace(initial_run=square_seed, runs=[square_seed, triangle_seed],
                           include_best=False)
    selections = _selections(args)
    labels = [label for _, label in selections]
    assert labels.count("square_seed") == 1
    assert labels.count("triangle_seed") == 1
    assert labels.count("square_final") == 1
    assert labels.count("triangle_final") == 1


def test_a_shape_only_present_in_a_later_run_still_gets_its_seed(tmp_path):
    """The seed of a shape lives in that shape's own first stage."""
    square = _run(tmp_path, "square_stage", ["square_seed", "square_final"])
    triangle = _run(tmp_path, "triangle_stage", ["triangle_seed", "triangle_final"])
    args = SimpleNamespace(initial_run=square, runs=[square, triangle], include_best=False)
    selections = dict((label, directory) for directory, label in _selections(args))
    assert selections["triangle_seed"] == triangle


def test_the_seed_replay_label_is_not_mistaken_for_a_seed(tmp_path):
    run = _run(tmp_path, "stage", ["square_seed", "square_seed_replay", "square_final"])
    args = SimpleNamespace(initial_run=run, runs=[run], include_best=False)
    labels = [label for _, label in _selections(args)]
    assert "square_seed_replay" not in labels
    assert labels.count("square_seed") == 1


def test_include_best_picks_the_lowest_accepted_sequential_candidate(tmp_path):
    directory = tmp_path / "stage"
    directory.mkdir()
    labels = ["square_seed", "square_0_B", "square_1_B", "square_final"]
    rows = []
    for index, label in enumerate(labels):
        rows.append(dict(label=label, file=f"capture_{index:04d}.npz",
                         metrics=dict(cv=0.9 - 0.1 * index), exposure_ms=50.0))
    (directory / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps([
        dict(shape="square", block=0, accepted=True, acceptance="sequential_ccd"),
        dict(shape="square", block=1, accepted=False, acceptance="sequential_ccd"),
    ]), encoding="utf-8")
    args = SimpleNamespace(initial_run=directory, runs=[directory], include_best=True)
    labels = [label for _, label in _selections(args)]
    assert "square_0_B" in labels
    assert "square_1_B" not in labels


def test_a_missing_manifest_is_an_error_not_a_silent_empty_replay(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    args = SimpleNamespace(initial_run=empty, runs=[empty], include_best=False)
    with pytest.raises(FileNotFoundError):
        _selections(args)
