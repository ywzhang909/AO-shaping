from pathlib import Path

import pandas as pd

from ao_shaping.utils.file import Recorder, ROOT_DIR, save_history


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