"""Tests for :mod:`ml.sim2f.recorder_io` -- the bench-session recorder adapter.

Hardware-free: the module under test imports no driver and opens no device,
so the whole file runs anywhere.

What is pinned here
===================
* the **canonical vocabulary**: every column whose name starts with an
  underscore must be one of :data:`ao_shaping.utils.io.recorder_keys.RECORD_KEYS`.
  A parallel spelling (``_pred`` / ``_twin`` / ...) would silently fork the
  on-disk format, so this is a hard gate, not a style preference;
* the **array contract**: the measured frame, the command phase and the panel
  grayscale keep their dtype and shape, and the sidecars reproduce the
  submitted arrays byte for byte;
* the **failure modes**: a missing ``mark`` must raise, not assert (an
  ``assert`` vanishes under ``python -O``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ao_shaping.utils.io.recorder_keys import (
    RECORD_COEFFS,
    RECORD_EPOCH,
    RECORD_ID,
    RECORD_IMAGE,
    RECORD_KEYS,
    RECORD_PHASE,
)
from ml.sim2f.recorder_io import (
    COLUMN_CONDITION,
    COLUMN_INDEX,
    COLUMN_KIND,
    CSV_NAME,
    CSV_STEM,
    META_NAME,
    SIDECAR_DIR_NAME,
    BenchRecorder,
    MissingMarkError,
)

#: Small stand-ins for the real bench sizes: a 16x16 command grid, a 12x10
#: camera frame and a 6x4 panel.  The real session uses 128x128 / 256x256 /
#: 1200x1920; the sizes only have to be *different from each other* for the
#: shape assertions to be meaningful.
GRID = 16
CCD = (12, 10)
PANEL = (6, 4)


def _sample(
    index: int,
    *,
    kind: str = "flat",
    condition: str = "a1",
    seed: int = 0,
    metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    """Build one sample's inputs.

    Args:
        index: Sample index.
        kind: Sample label.
        condition: Session condition.
        seed: RNG seed, so the arrays are reproducible.
        metrics: Metric mapping; defaults to a mark plus one extra metric.

    Returns:
        The keyword arguments of
        :meth:`ml.sim2f.recorder_io.BenchRecorder.append_sample`.
    """
    rng = np.random.default_rng(seed)
    return {
        "index": index,
        "kind": kind,
        "condition": condition,
        "frame": rng.random(CCD).astype(np.float32),
        "command_phase": rng.standard_normal((GRID, GRID)).astype(np.float32),
        "panel_grayscale": rng.integers(0, 1024, size=PANEL).astype(np.uint16),
        "metrics": {"ncc_net": 0.5 + index, "objective_l1": 1.5}
        if metrics is None
        else metrics,
    }


def _two_rows() -> BenchRecorder:
    """Return a recorder holding two samples.

    Returns:
        The recorder.
    """
    rec = BenchRecorder(mark="ncc_net")
    rec.append_sample(**_sample(0, seed=1))  # type: ignore[arg-type]
    rec.append_sample(**_sample(7, kind="shaped", condition="b1", seed=2))  # type: ignore[arg-type]
    return rec


class TestRowContent:
    """The rows carry the canonical columns with the documented dtypes."""

    def test_row_ids_and_epochs_follow_the_caller(self) -> None:
        """``_id`` increments; ``_epoch`` is the index the caller passed in."""
        rec = _two_rows()
        rows = rec.history
        assert len(rows) == 2
        assert [row[RECORD_ID] for row in rows] == [0, 1]
        assert [row[RECORD_EPOCH] for row in rows] == [0, 7]

    def test_array_columns_keep_dtype_and_shape(self) -> None:
        """The frame is float32, the command phase float32, the panel uint16."""
        rec = _two_rows()
        for row, sample in zip(rec.history, [_sample(0, seed=1), _sample(7, seed=2)]):
            image = row[RECORD_IMAGE]
            assert isinstance(image, np.ndarray)
            assert image.dtype == np.float32
            assert image.ndim == 2
            assert image.shape == CCD
            np.testing.assert_array_equal(image, sample["frame"])

            coeffs = row[RECORD_COEFFS]
            assert isinstance(coeffs, np.ndarray)
            assert coeffs.dtype == np.float32
            assert coeffs.ndim == 2
            assert coeffs.shape == (GRID, GRID)
            np.testing.assert_array_equal(coeffs, sample["command_phase"])

            phase = row[RECORD_PHASE]
            assert isinstance(phase, np.ndarray)
            assert phase.dtype == np.uint16
            assert phase.ndim == 2
            assert phase.shape == PANEL
            np.testing.assert_array_equal(phase, sample["panel_grayscale"])

    def test_mark_and_metrics_are_present(self) -> None:
        """The mark column and every metric land in the row."""
        rec = _two_rows()
        for row in rec.history:
            assert "ncc_net" in row
            assert "objective_l1" in row
        assert rec.history[0]["ncc_net"] == pytest.approx(0.5)
        assert rec.history[1]["ncc_net"] == pytest.approx(7.5)

    def test_plain_columns_are_plain(self) -> None:
        """The non-canonical bookkeeping columns carry ordinary names."""
        rec = _two_rows()
        row = rec.history[1]
        assert row[COLUMN_KIND] == "shaped"
        assert row[COLUMN_CONDITION] == "b1"
        assert row[COLUMN_INDEX] == 7


class TestCanonicalVocabulary:
    """The hard gate: no parallel spelling of a canonical key."""

    def test_underscore_keys_are_canonical(self) -> None:
        """Every underscore-prefixed key is one of ``RECORD_KEYS``."""
        rec = _two_rows()
        for row in rec.history:
            for key in row:
                if key.startswith("_"):
                    assert key in RECORD_KEYS, f"non-canonical key {key!r}"

    def test_saved_columns_are_canonical(self) -> None:
        """The same holds for the columns of the written frame."""
        rec = _two_rows()
        for col in rec.dataframe.columns:
            if col.startswith("_"):
                assert col in RECORD_KEYS, f"non-canonical column {col!r}"

    def test_a_metric_may_not_masquerade_as_a_canonical_key(self) -> None:
        """A metric named like a canonical key is rejected."""
        rec = BenchRecorder(mark="ncc_net")
        with pytest.raises(ValueError, match="canonical"):
            rec.append_sample(
                **_sample(0, metrics={"ncc_net": 0.5, RECORD_IMAGE: 1.0})  # type: ignore[arg-type]
            )

    def test_a_metric_may_not_start_with_an_underscore(self) -> None:
        """A metric may not claim the canonical namespace."""
        rec = BenchRecorder(mark="ncc_net")
        with pytest.raises(ValueError, match="underscore"):
            rec.append_sample(
                **_sample(0, metrics={"ncc_net": 0.5, "_twin": 1.0})  # type: ignore[arg-type]
            )


class TestFailureModes:
    """A missing mark must raise loudly, not assert."""

    def test_missing_mark_raises(self) -> None:
        """``mark`` absent from ``metrics`` -> ``MissingMarkError``."""
        rec = BenchRecorder(mark="ncc_net")
        with pytest.raises(MissingMarkError):
            rec.append_sample(**_sample(0, metrics={"objective_l1": 1.0}))  # type: ignore[arg-type]
        assert len(rec) == 0

    def test_unknown_condition_is_rejected(self) -> None:
        """A condition outside the A1/B1/B2/A2 quartet is rejected."""
        rec = BenchRecorder(mark="ncc_net")
        with pytest.raises(ValueError, match="condition"):
            rec.append_sample(**_sample(0, condition="c3"))  # type: ignore[arg-type]

    def test_a_non_2d_array_is_rejected(self) -> None:
        """A 1-D frame is refused instead of being silently reshaped."""
        rec = BenchRecorder(mark="ncc_net")
        sample = _sample(0)
        sample["frame"] = np.zeros(8, dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            rec.append_sample(**sample)  # type: ignore[arg-type]


class TestSave:
    """The session lands on disk in the canonical layout."""

    def test_csv_and_sidecars_are_written(self, tmp_path: Path) -> None:
        """CSV rows match the appended rows; every array cell has a sidecar."""
        rec = _two_rows()
        paths = rec.save(tmp_path)

        csv_path = Path(paths["csv"])
        assert csv_path.name == CSV_NAME
        assert csv_path.is_file()
        table = pd.read_csv(csv_path)
        assert len(table) == 2

        sidecar_dir = Path(paths["sidecars"])
        assert sidecar_dir.name == SIDECAR_DIR_NAME
        assert sidecar_dir.is_dir()
        files = sorted(sidecar_dir.glob("*.npy"))
        assert len(files) == 2 * 3  # two rows x three array columns

    def test_sidecars_reproduce_the_arrays_bit_for_bit(self, tmp_path: Path) -> None:
        """Every submitted array comes back unchanged from its sidecar."""
        rec = _two_rows()
        paths = rec.save(tmp_path)
        sidecar_dir = Path(paths["sidecars"])

        samples = [_sample(0, seed=1), _sample(7, seed=2)]
        expected = {
            RECORD_IMAGE: [sample["frame"] for sample in samples],
            RECORD_COEFFS: [sample["command_phase"] for sample in samples],
            RECORD_PHASE: [sample["panel_grayscale"] for sample in samples],
        }
        for column, arrays in expected.items():
            for position, array in enumerate(arrays):
                matches = [
                    path
                    for path in sorted(sidecar_dir.glob("*.npy"))
                    if column in path.name
                    and np.load(path).shape == np.asarray(array).shape
                    and np.array_equal(np.load(path), np.asarray(array))
                ]
                assert matches, f"no sidecar reproduces {column} of row {position}"
                assert len(matches) == 1

    def test_meta_reports_the_array_volume(self, tmp_path: Path) -> None:
        """The metadata names the columns and their byte volume."""
        rec = _two_rows()
        paths = rec.save(tmp_path)
        meta_path = Path(paths["json"])
        assert meta_path.name == META_NAME
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        assert meta["row_count"] == 2
        assert meta["mark"] == "ncc_net"
        assert set(meta["columns"]) >= {
            RECORD_ID,
            RECORD_EPOCH,
            RECORD_IMAGE,
            RECORD_COEFFS,
            RECORD_PHASE,
            COLUMN_KIND,
            COLUMN_CONDITION,
            COLUMN_INDEX,
        }
        assert meta["array_bytes"][RECORD_PHASE] == 2 * PANEL[0] * PANEL[1] * 2
        assert meta["array_bytes"][RECORD_IMAGE] == 2 * CCD[0] * CCD[1] * 4
        assert meta["sidecar_file_count"] == 6

    def test_save_is_idempotent(self, tmp_path: Path) -> None:
        """Saving the same recorder twice does not raise."""
        rec = _two_rows()
        first = rec.save(tmp_path)
        second = rec.save(tmp_path)
        assert first == second
        table = pd.read_csv(Path(second["csv"]))
        assert len(table) == 2
        files = sorted(Path(second["sidecars"]).glob("*.npy"))
        assert len(files) == 6

    def test_csv_stem_names_the_sidecars(self, tmp_path: Path) -> None:
        """The sidecar file names carry the CSV stem and the column name."""
        rec = _two_rows()
        paths = rec.save(tmp_path)
        names = {path.name for path in Path(paths["sidecars"]).glob("*.npy")}
        assert f"{CSV_STEM}_{RECORD_IMAGE}_0000.npy" in names
        assert f"{CSV_STEM}_{RECORD_COEFFS}_0001.npy" in names
        assert f"{CSV_STEM}_{RECORD_PHASE}_0001.npy" in names


class TestCorpusReadability:
    """The recorded rows are consumable by the repository's corpus reader."""

    def test_the_corpus_reader_keeps_and_classifies_the_rows(
        self, tmp_path: Path
    ) -> None:
        """A pickled session is indexed as panel-gray records, none excluded."""
        import pickle

        from ml.hwdataset import PhaseSource, build_hw_index

        rec = _two_rows()
        with open(tmp_path / "bench_session_rows.pkl", "wb") as handle:
            pickle.dump(rec.recorder, handle)

        index = build_hw_index(roots=[tmp_path])
        assert len(index.records) == 2
        assert all(ref.source is PhaseSource.PANEL_GRAY for ref in index.records)
        # ``_phase`` is uint16, so the panel-gray branch wins over ``_c``.
        assert all(ref.n_terms == GRID for ref in index.records)
