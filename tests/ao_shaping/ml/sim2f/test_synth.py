"""Tests for the corpus builder: schema, stratification, reproducibility.

The corpus is the contract between the teacher and the trainer, so the checks
here are about *identity*, not quality: the archive must satisfy
:func:`ml.sim2f.schema.validate_cache`, the splits must be stratified by target
kind (a global shuffle routinely leaves a kind out of validation entirely), and a
second build into a different directory must produce byte-identical arrays --
otherwise a training run could not be reproduced from its metadata.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch lives in the optional ml group")

from ml.sim2f.config import (
    TARGET_KINDS,
    Sim2fConfig,
    SynthConfig,
    TeacherConfig,
)
from ml.sim2f.schema import (
    CACHE_FILE,
    CACHE_FIELDS,
    CACHE_VERSION,
    META_FILE,
    SPLIT_NAMES,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from ml.sim2f.synth import build_pairs, load_pairs

#: Small bench, two shapes, a two-iteration teacher: the corpus mechanics are what
#: this file tests, not the label quality (that is ``test_teacher``).
SIM = Sim2fConfig(grid=64, region_span_px=900.0, ccd_px=64, ccd_pixel_um=4.4)
SYNTH = SynthConfig(count=20, seed=1234, target_kinds=("square", "disc"))
TEACHER = TeacherConfig(iters=2)


def _build(tmp_path, name: str = "corpus"):
    """Build one corpus and return ``(directory, arrays, meta)``."""
    directory = tmp_path / name
    returned = build_pairs(
        SIM, SYNTH, TEACHER, out_dir=directory, device="cpu"
    )
    arrays, meta = load_pairs(directory)
    return returned, arrays, meta


def test_archive_satisfies_the_schema(tmp_path) -> None:
    """Every declared field is present with its exact shape and dtype."""
    returned, arrays, _ = _build(tmp_path)
    assert returned.is_dir()
    assert (returned / CACHE_FILE).is_file()
    assert (returned / META_FILE).is_file()

    for field in CACHE_FIELDS:
        assert field.name in arrays, field.name
        assert arrays[field.name].dtype.name == field.dtype
        assert arrays[field.name].shape[0] == SYNTH.count
    assert arrays["phase"].shape == (SYNTH.count, SIM.grid, SIM.grid)
    assert arrays["target"].shape == (SYNTH.count, 1, SIM.ccd_px, SIM.ccd_px)

    # Content sanity: unit-sum targets, non-empty supports, finite phases.
    sums = arrays["target"].reshape(SYNTH.count, -1).sum(axis=1)
    assert sums == pytest.approx(np.ones(SYNTH.count), rel=1e-4)
    assert (arrays["support"].reshape(SYNTH.count, -1).sum(axis=1) >= 16).all()
    assert np.isfinite(arrays["phase"]).all()
    assert (arrays["kind"] >= 0).all() and (arrays["kind"] < len(TARGET_KINDS)).all()
    # The reference flat frame is the same optics for every sample.
    assert np.allclose(arrays["spot"], arrays["spot"][0])


def test_splits_are_stratified_by_kind(tmp_path) -> None:
    """Ten samples per kind split 8/1/1, so every split sees both kinds."""
    _, arrays, meta = _build(tmp_path)
    split = arrays["split"]
    assert set(np.unique(split)) == {SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST}

    kinds = [TARGET_KINDS[int(index)] for index in arrays["kind"]]
    for kind in SYNTH.target_kinds:
        members = [index for index, value in enumerate(kinds) if value == kind]
        assert len(members) == SYNTH.count // len(SYNTH.target_kinds)
        per_kind = {SPLIT_NAMES[int(split[index])] for index in members}
        assert per_kind == {"train", "val", "test"}, kind

    counts = meta["counts"]["split"]
    assert sum(counts.values()) == SYNTH.count
    assert counts["train"] == int((split == SPLIT_TRAIN).sum())
    assert counts["val"] == int((split == SPLIT_VAL).sum())
    assert counts["test"] == int((split == SPLIT_TEST).sum())
    assert sum(meta["counts"]["kind"].values()) == SYNTH.count


def test_metadata_describes_the_run(tmp_path) -> None:
    """``meta.json`` carries the three configs, the derived optics and versions."""
    _, _, meta = _build(tmp_path)
    assert meta["version"] == CACHE_VERSION
    for section in ("sim2f", "synth", "teacher", "derived"):
        assert section in meta

    assert meta["sim2f"]["grid"] == SIM.grid
    assert meta["sim2f"]["ccd_px"] == SIM.ccd_px
    assert meta["synth"]["count"] == SYNTH.count
    assert meta["synth"]["seed"] == SYNTH.seed
    assert meta["teacher"] == TEACHER.to_dict()

    # Derived numbers are the config's own, not a second computation.
    for key, value in SIM.describe().items():
        assert meta["derived"][key] == pytest.approx(value, rel=1e-12)
    assert meta["derived"]["sampling_ratio"] == pytest.approx(
        SIM.sampling_ratio, rel=1e-12
    )

    assert meta["wall_s"] > 0.0
    assert meta["torch_version"] == torch.__version__
    assert meta["numpy_version"] == np.__version__
    assert "git_commit" in meta  # may be None outside a checkout


def test_per_sample_seeds_are_recorded_and_reproducible(tmp_path) -> None:
    """Each sample's seed is stored, so one pair can be rebuilt on its own."""
    _, arrays, _ = _build(tmp_path)
    seeds = arrays["seed"]
    assert seeds.dtype == np.int64
    assert len(set(int(value) for value in seeds)) == SYNTH.count

    # The master seed alone determines the per-sample seeds.
    master = np.random.default_rng(SYNTH.seed)
    expected = master.integers(0, 2**31 - 1, size=SYNTH.count, dtype=np.int64)
    assert np.array_equal(seeds, expected)


def test_two_builds_are_byte_identical(tmp_path) -> None:
    """The same three configs produce identical arrays in a different directory."""
    _, first, _ = _build(tmp_path, "first")
    _, second, _ = _build(tmp_path, "second")
    assert set(first) == set(second)
    for name in first:
        assert np.array_equal(first[name], second[name]), name

    # And the metadata differs only in the wall clock.
    meta_first = json.loads((tmp_path / "first" / META_FILE).read_text())
    meta_second = json.loads((tmp_path / "second" / META_FILE).read_text())
    for key in meta_first:
        if key == "wall_s":
            continue
        assert meta_first[key] == meta_second[key], key


def test_load_pairs_rejects_missing_files(tmp_path) -> None:
    """A missing archive or sidecar is an error, not an empty corpus."""
    with pytest.raises(FileNotFoundError):
        load_pairs(tmp_path / "nope")


def test_load_pairs_detects_a_schema_violation(tmp_path) -> None:
    """A truncated archive fails the schema check instead of training quietly."""
    _, arrays, _ = _build(tmp_path)
    broken = dict(arrays)
    broken.pop("support")
    np.savez(tmp_path / "corpus" / CACHE_FILE, **broken)
    with pytest.raises(ValueError, match="missing field"):
        load_pairs(tmp_path / "corpus")


def test_build_pairs_returns_the_output_directory(tmp_path) -> None:
    """The return value is the directory holding the archive."""
    directory = build_pairs(
        SIM, SYNTH, TEACHER, out_dir=tmp_path / "out", device="cpu"
    )
    assert directory == tmp_path / "out"
    assert (directory / CACHE_FILE).is_file()


def test_metadata_records_the_feature_floor(tmp_path) -> None:
    """The floor is part of the corpus identity, so ``meta.json`` carries it."""
    assert SYNTH.min_feature_px == pytest.approx(24.0)
    _, _, meta = _build(tmp_path)
    assert meta["synth"]["min_feature_px"] == pytest.approx(SYNTH.min_feature_px)


def test_unreachable_target_stops_the_build(tmp_path) -> None:
    """A target thinner than the bench can shape fails the build, loudly.

    ``line`` needs ``3 x floor`` of length to stay a line, and this toy bench is
    64 px wide, so the reference floor of 24 px is impossible here: the build must
    raise instead of caching a pair whose label the optics cannot produce.
    """
    unbuildable = SynthConfig(count=2, seed=1234, target_kinds=("line",))
    with pytest.raises(ValueError) as excinfo:
        build_pairs(
            SIM, unbuildable, TEACHER, out_dir=tmp_path / "broken", device="cpu"
        )
    message = str(excinfo.value)
    assert "line" in message
    assert "24.00px feature floor" in message
    assert not (tmp_path / "broken" / CACHE_FILE).is_file()


def test_draw_error_carries_the_kind_and_the_seed(tmp_path, monkeypatch) -> None:
    """The synth-level error names the sample; the guard's numbers are preserved."""
    def _reject(*args, **kwargs):
        raise ValueError(
            "unreachable 'square' target: its thinnest feature is 2.00px, below the "
            "24.00px floor"
        )

    monkeypatch.setattr("ml.sim2f.synth.draw_target", _reject)
    with pytest.raises(ValueError) as excinfo:
        build_pairs(SIM, SYNTH, TEACHER, out_dir=tmp_path / "rejected", device="cpu")
    message = str(excinfo.value)
    assert "'square'" in message
    assert "seed" in message
    assert "2.00px" in message and "24.00px floor" in message
