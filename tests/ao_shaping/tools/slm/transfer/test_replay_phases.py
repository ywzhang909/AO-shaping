"""Replay selection: every shape of a multi-shape session is replayed."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

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


def _grid_run(tmp_path: Path, name: str, step: float) -> Path:
    """A run whose phase-bearing captures all share one ``camera_xy`` grid."""
    directory = tmp_path / name
    directory.mkdir()
    rows, xy = [], None
    for index, label in enumerate(["square_seed", "square_final"]):
        xy = np.arange(0, 4, step, dtype=float)
        grid = np.stack(np.meshgrid(xy, xy, indexing="xy"), axis=-1)
        np.savez_compressed(directory / f"capture_{index:04d}.npz",
                            model_phase=np.zeros((2, 2)), camera_xy=grid,
                            target=np.ones(grid.shape[:2]), core=np.ones(grid.shape[:2], bool))
        rows.append(dict(label=label, file=f"capture_{index:04d}.npz", metrics=dict(cv=1.0)))
    (directory / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    return directory


def test_stages_at_different_sampling_steps_still_group_their_own_grids(tmp_path):
    """Stage 4 samples the window finer, so its grid legitimately differs.

    The replay must sample each phase through its own grid rather than reject
    the session: the cross-stage comparison is the whole point of the replay.
    """
    coarse = _grid_run(tmp_path, "square_transport_cv", 1.0)
    fine = _grid_run(tmp_path, "square_mono12_dense", 0.5)
    args = SimpleNamespace(initial_run=coarse, runs=[coarse, fine], include_best=False)
    selections = _selections(args)
    assert {directory.name for directory, _ in selections} == {"square_transport_cv", "square_mono12_dense"}
    shapes = {}
    for directory, label in selections:
        with np.load(directory / json.loads((directory / "manifest.json").read_text())[0]["file"]) as saved:
            shapes.setdefault(saved["camera_xy"].shape, []).append(label)
    assert len(shapes) == 2
