"""Unit tests for the objective-stratified train/val split in `train_amp._select_records`.

The split used to be a plain 25%-of-files draw plus a `val[:128]` cap, which on the
``slm_zernike_shaping`` family happened to land both validation files on a single
optimisation objective (``rmse_out``). That made the single global coefficient vector
be judged on one bench state while trained on a mixture of four -- the root of the
large train/val gap. These tests pin the stratification invariants:

* a single-file objective never leaks into validation (it cannot be split);
* every multi-file objective keeps at least one file in validation;
* a file is never split across train/val (records inside one pickle are near-duplicate
  consecutive epochs);
* the split is deterministic for a fixed seed;
* the size caps truncate proportionally so the objective mix survives.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from ml.hwdataset import HwRecordRef, PhaseSource
from ml.zernike.train_amp import (
    AmpTrainConfig,
    _select_records,
    _stratified_cap,
    objective_of,
)


class _FakeDataset:
    """Just enough surface for ``_select_records`` (it reads ``records`` / ``_fov_px``)."""

    def __init__(self, records: list[HwRecordRef]) -> None:
        self.records = records

    def _fov_px(self, record: HwRecordRef) -> int:
        return 0


def _make_records(n_per_file: int = 10) -> tuple[list[HwRecordRef], dict[int, str]]:
    """Build records whose paths encode the four unevenly-sized objectives.

    Mirrors the real ``slm_zernike_shaping`` layout: ``rms_pib`` (4 files),
    ``rmse_out`` (3), ``shape`` (2), ``roi_pib`` (1). Returns the records and a map
    of record position -> objective.
    """
    layout = {"rms_pib": 4, "rmse_out": 3, "shape": 2, "roi_pib": 1}
    records: list[HwRecordRef] = []
    objective_of_pos: dict[int, str] = {}
    pos = 0
    for objective, n_files in layout.items():
        for f in range(n_files):
            stamp = 100_000 + f
            path = Path(
                f"data/slm_zernike_shaping_{objective}_20260926_{stamp:06d}_"
                f"20260926_{stamp:06d}.pkl"
            )
            for _ in range(n_per_file):
                records.append(
                    HwRecordRef(
                        path=path,
                        position=pos,
                        key=pos,
                        family="slm_zernike_shaping",
                        source=PhaseSource.PANEL_RAD,
                        n_terms=0,
                        n_max=None,
                        freeform_grid=None,
                        exposure_ms=1.0,
                        sidecar={},
                    )
                )
                objective_of_pos[pos] = objective
                pos += 1
    return records, objective_of_pos


def _split(cfg: AmpTrainConfig) -> tuple[list[int], list[int]]:
    records, _ = _make_records()
    return _select_records(_FakeDataset(records), cfg)


class TestStratifiedInvariants:
    def test_single_file_objective_never_in_validation(self) -> None:
        train, val = _split(AmpTrainConfig(seed=0))
        _, objective_of_pos = _make_records()
        # roi_pib is the only single-file objective: none of its records may be in val.
        assert all(objective_of_pos[i] != "roi_pib" for i in val)
        assert any(objective_of_pos[i] == "roi_pib" for i in train)

    def test_each_multi_file_objective_represented_in_validation(self) -> None:
        train, val = _split(AmpTrainConfig(seed=0))
        _, objective_of_pos = _make_records()
        val_objectives = {objective_of_pos[i] for i in val}
        for objective in ("rms_pib", "rmse_out", "shape"):
            assert objective in val_objectives, f"{objective} missing from validation"

    def test_no_file_is_split_across_train_and_val(self) -> None:
        records, _ = _make_records()
        train, val = _split(AmpTrainConfig(seed=0))
        # Map record position -> its file path, then assert each file sits on one side.
        owner: dict[str, set[str]] = defaultdict(set)
        for i in train:
            owner[str(records[i].path)].add("train")
        for i in val:
            owner[str(records[i].path)].add("val")
        assert all(len(sides) == 1 for sides in owner.values()), "a file leaked across splits"

    def test_split_covers_every_record_exactly_once(self) -> None:
        records, _ = _make_records()
        train, val = _split(AmpTrainConfig(seed=0))
        seen = set(train) | set(val)
        assert len(train) + len(val) == len(records)
        assert seen == set(range(len(records)))
        assert not (set(train) & set(val))

    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        a = _split(AmpTrainConfig(seed=7))
        b = _split(AmpTrainConfig(seed=7))
        assert sorted(a[0]) == sorted(b[0])
        assert sorted(a[1]) == sorted(b[1])


class TestStratifiedCap:
    """The proportional cap must retain the objective mix, not collapse to one objective."""

    def test_retains_all_files_that_fit(self) -> None:
        positions = list(range(30))  # three files of 10
        file_to_positions = {"a": positions[:10], "b": positions[10:20], "c": positions[20:30]}
        out = _stratified_cap(positions, file_to_positions, cap=20)
        assert len(out) == 20
        # Each of the three files contributes at least one record.
        a = sum(1 for i in out if i < 10)
        b = sum(1 for i in out if 10 <= i < 20)
        c = sum(1 for i in out if i >= 20)
        assert a > 0 and b > 0 and c > 0
        assert a + b + c == 20

    def test_cap_larger_than_length_is_a_noop(self) -> None:
        positions = list(range(5))
        file_to_positions = {"a": positions}
        assert _stratified_cap(positions, file_to_positions, cap=99) == positions

    def test_cap_zero_returns_everything(self) -> None:
        positions = list(range(5))
        file_to_positions = {"a": positions}
        assert _stratified_cap(positions, file_to_positions, cap=0) == positions

    def test_truncation_keeps_multi_file_objectives_in_val(self) -> None:
        # Small enough cap to force truncation of the 30-record validation split.
        train, val = _split(AmpTrainConfig(seed=0, max_val=20))
        _, objective_of_pos = _make_records()
        assert len(val) == 20
        val_objectives = {objective_of_pos[i] for i in val}
        # All three multi-file objectives that the stratified split put in val must
        # survive the proportional truncation.
        assert {"rms_pib", "rmse_out", "shape"} <= val_objectives


class TestObjectiveOf:
    def test_parses_each_objective_from_real_filename_shapes(self) -> None:
        assert (
            objective_of(
                Path(
                    "slm_zernike_shaping_rmse_out_20260926_163121_20260926_163121.pkl"
                )
            )
            == "rmse_out"
        )
        assert (
            objective_of(Path("slm_zernike_shaping_roi_pib_20260926_171311_20260926_171311.pkl"))
            == "roi_pib"
        )
        assert objective_of(Path("other_family_thing.pkl")) == "?"
