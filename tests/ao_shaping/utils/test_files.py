from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from ao_shaping.utils.io.file import Recorder, ROOT_DIR, save_history, save_history_hdf5


def test_root_dir_exists():
    """Test that ROOT_DIR points to the project root."""
    assert ROOT_DIR.exists()
    assert (ROOT_DIR / "src").exists()
    assert (ROOT_DIR / "libs").exists()


def test_save_history_from_list(tmp_path):
    history = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    file_path = tmp_path / "test_history.csv"
    save_history(history, file_path)

    csv_path = file_path.with_suffix(".csv")
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert df["a"].tolist() == [1, 3]
    assert df["b"].tolist() == [2, 4]


def test_save_history_from_dataframe(tmp_path):
    df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
    file_path = tmp_path / "test_history.csv"
    save_history(df, file_path)

    csv_path = file_path.with_suffix(".csv")
    assert csv_path.exists()
    loaded = pd.read_csv(csv_path)
    assert len(loaded) == 2
    assert loaded["x"].tolist() == [10, 20]


def test_save_history_creates_parent_dirs(tmp_path):
    file_path = tmp_path / "subdir" / "nested" / "history.csv"
    save_history([{"v": 1}], file_path)

    csv_path = file_path.with_suffix(".csv")
    assert csv_path.exists()
    loaded = pd.read_csv(csv_path)
    assert loaded["v"].tolist() == [1]


def test_add_record():
    recorder1 = Recorder("mark1", "target1")
    recorder2 = Recorder("mark1", "target1")
    recorder1.append({"_id": 1, "mark1": 0.1, "target1": 0.2})
    recorder2.append({"_id": 2, "mark1": 0.3, "target1": 0.4})
    recorder1 += recorder2
    assert len(recorder1) == 2
    assert recorder1[0]["_id"] == 1
    assert recorder1[1]["_id"] == 2

def test_merge_recorder():
    recorder1 = Recorder("mark1", "target1")
    recorder2 = Recorder("mark1", "target1")
    recorder1.append({"_id": 1, "mark1": 0.1, "target1": 0.2})
    recorder2.append({"_id": 2, "mark1": 0.3, "target1": 0.4})
    recorder1 += recorder2
    assert len(recorder1) == 2
    assert recorder1[0]["_id"] == 1
    assert recorder1[1]["_id"] == 2


class TestSaveHistoryHdf5:
    def test_list_of_dicts_roundtrip(self, tmp_path):
        history = [
            {"_id": 0, "loss": 1.0, "coeffs": np.array([1.0, 2.0])},
            {"_id": 1, "loss": 0.5, "coeffs": np.array([3.0, 4.0])},
        ]
        path = save_history_hdf5(
            history,
            tmp_path / "run.csv",
            metadata={"run": "test", "epochs": 2, "flag": True, "none_val": None},
        )
        assert path == tmp_path / "run.h5"
        with h5py.File(path, "r") as f:
            assert f["metadata"].attrs["run"] == "test"
            assert f["metadata"].attrs["epochs"] == 2
            assert f["metadata"].attrs["flag"] == True
            assert f["metadata"].attrs["none_val"] == ""
            np.testing.assert_allclose(f["scalars/loss"][:], [1.0, 0.5])
            np.testing.assert_array_equal(f["epochs/0000/coeffs"][:], [1.0, 2.0])
            np.testing.assert_array_equal(f["epochs/0001/coeffs"][:], [3.0, 4.0])

    def test_recorder_input(self, tmp_path):
        rec = Recorder("J", "max")
        rec.append({"_id": 0, "J": 0.1, "coeffs": np.array([1.0])})
        rec.append({"_id": 1, "J": 0.2, "coeffs": np.array([2.0])})
        path = save_history_hdf5(rec, tmp_path / "rec.h5")
        with h5py.File(path, "r") as f:
            np.testing.assert_allclose(f["scalars/J"][:], [0.1, 0.2])
            np.testing.assert_array_equal(f["epochs/0000/coeffs"][:], [1.0])
            np.testing.assert_array_equal(f["epochs/0001/coeffs"][:], [2.0])

    def test_dataframe_input(self, tmp_path):
        df = pd.DataFrame({"_id": [0, 1], "loss": [1.0, 0.5]})
        path = save_history_hdf5(df, tmp_path / "df.h5")
        with h5py.File(path, "r") as f:
            np.testing.assert_allclose(f["scalars/loss"][:], [1.0, 0.5])

    def test_empty_list_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_history_hdf5([], tmp_path / "empty.h5")

    def test_empty_recorder_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_history_hdf5(Recorder("J", "max"), tmp_path / "empty_rec.h5")

    def test_empty_dataframe_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_history_hdf5(pd.DataFrame(), tmp_path / "empty_df.h5")

    def test_csv_suffix_replaced_with_h5(self, tmp_path):
        path = save_history_hdf5([{"v": 1.0}], tmp_path / "run.csv")
        assert path == tmp_path / "run.h5"
        assert path.exists()

    def test_no_suffix_appends_h5(self, tmp_path):
        path = save_history_hdf5([{"v": 1.0}], tmp_path / "run")
        assert path == tmp_path / "run.h5"
        assert path.exists()

    def test_missing_scalar_values_become_nan(self, tmp_path):
        history = [
            {"_id": 0, "loss": 1.0},
            {"_id": 1, "loss": 0.5, "extra": 2.0},
        ]
        path = save_history_hdf5(history, tmp_path / "nan.h5")
        with h5py.File(path, "r") as f:
            np.testing.assert_allclose(f["scalars/loss"][:], [1.0, 0.5])
            np.testing.assert_allclose(
                f["scalars/extra"][:], [np.nan, 2.0], equal_nan=True
            )

    def test_list_and_tuple_array_columns(self, tmp_path):
        history = [
            {"_id": 0, "vec": [1.0, 2.0, 3.0]},
            {"_id": 1, "vec": (4.0, 5.0, 6.0)},
        ]
        path = save_history_hdf5(history, tmp_path / "vec.h5")
        with h5py.File(path, "r") as f:
            np.testing.assert_array_equal(f["epochs/0000/vec"][:], [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(f["epochs/0001/vec"][:], [4.0, 5.0, 6.0])

    def test_epoch_id_from_row_id(self, tmp_path):
        history = [
            {"coeffs": np.array([1.0])},  # no _id -> falls back to row index 0
            {"_id": 7, "coeffs": np.array([2.0])},
        ]
        path = save_history_hdf5(history, tmp_path / "ids.h5")
        with h5py.File(path, "r") as f:
            np.testing.assert_array_equal(f["epochs/0000/coeffs"][:], [1.0])
            np.testing.assert_array_equal(f["epochs/0007/coeffs"][:], [2.0])