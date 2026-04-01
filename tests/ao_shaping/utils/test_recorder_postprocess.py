"""Test Recorder postprocess_feature functionality."""

import numpy as np
import pytest

from ao_shaping.utils.file import Recorder


class TestRecorderPostprocessFeature:
    """Test postprocess_feature method for Recorder."""

    def test_postprocess_adds_column(self):
        """Test that postprocess_feature adds a new column to dataframe."""
        rec = Recorder(mark="score")
        rec.append({"score": 10, "value": 100})
        rec.append({"score": 20, "value": 200})

        rec.postprocess_feature("double_value", lambda row: row["value"] * 2)

        df = rec.dataframe
        assert "double_value" in df.columns
        assert df["double_value"].iloc[0] == 200
        assert df["double_value"].iloc[1] == 400

    def test_postprocess_with_custom_column_name(self):
        """Test postprocess_feature with custom column name."""
        rec = Recorder(mark="score")
        rec.append({"score": 10, "value": 100})

        rec.postprocess_feature("calc", lambda row: row["value"] * 3, column="triple")

        df = rec.dataframe
        assert "triple" in df.columns
        assert df["triple"].iloc[0] == 300

    def test_postprocess_in_columns(self):
        """Test that postprocess columns appear in .columns property."""
        rec = Recorder(mark="score")
        rec.append({"score": 10, "value": 100})

        rec.postprocess_feature("computed", lambda row: row["value"] + 1)

        assert "computed" in rec.columns

    def test_get_best_iter_with_postprocess(self):
        """Test get_best_iter works with postprocess feature as mark."""
        rec = Recorder(mark="score", mode="max")
        rec.append({"score": 10, "value": 100})
        rec.append({"score": 5, "value": 200})
        rec.append({"score": 15, "value": 50})

        # Register a postprocess feature
        rec.postprocess_feature("inverse_score", lambda row: 1.0 / row["score"])

        # Get best by inverse_score (should be minimum score = 5, so max inverse = 0.2)
        rec.mode = "max"
        best_iter, (idx, val) = rec.get_best_iter("inverse_score")
        assert idx == 1  # row with score=5 has highest inverse
        assert abs(val - 0.2) < 1e-6

    def test_get_best_target_with_postprocess(self):
        """Test get_best_target works with postprocess feature."""
        rec = Recorder(mark="score", mode="max")
        rec.append({"score": 10, "value": 100})
        rec.append({"score": 20, "value": 200})

        rec.postprocess_feature("ratio", lambda row: row["value"] / row["score"])

        # ratio is 10 for both rows
        target_val, (idx, score_val) = rec.get_best_target("ratio")
        assert abs(target_val - 10.0) < 1e-6

    def test_get_best_target_min_mode_with_postprocess(self):
        """Test get_best_target with min mode and postprocess feature."""
        rec = Recorder(mark="radius", mode="min")
        rec.append({"radius": 5.0, "img": np.ones((10, 10))})
        rec.append({"radius": 3.0, "img": np.ones((10, 10)) * 2})
        rec.append({"radius": 8.0, "img": np.ones((10, 10)) * 3})

        rec.postprocess_feature("area", lambda row: np.pi * row["radius"] ** 2)

        # Best by radius (min) should be index 1
        best_iter, (idx, val) = rec.get_best_iter("radius")
        assert idx == 1
        assert val == 3.0

        # Best by area should also be index 1
        best_iter, (idx, val) = rec.get_best_iter("area")
        assert idx == 1

    def test_postprocess_with_underscore_prefix(self):
        """Test postprocess feature with underscore-prefixed column name."""
        rec = Recorder(mark="score")
        rec.append({"score": 10, "img": np.ones((5, 5))})

        rec.postprocess_feature(
            "second_moment",
            lambda row: float(np.sum(row["img"])),
            column="_second_moment",
        )

        df = rec.dataframe
        assert "_second_moment" in df.columns
        assert df["_second_moment"].iloc[0] == 25.0

    def test_postprocess_multiple_features(self):
        """Test registering multiple postprocess features."""
        rec = Recorder(mark="score")
        rec.append({"score": 10, "value": 100})
        rec.append({"score": 20, "value": 200})

        rec.postprocess_feature("double", lambda row: row["value"] * 2)
        rec.postprocess_feature("half", lambda row: row["value"] / 2)
        rec.postprocess_feature("sum", lambda row: row["value"] + row["score"])

        df = rec.dataframe
        assert "double" in df.columns
        assert "half" in df.columns
        assert "sum" in df.columns
        assert df["double"].iloc[0] == 200
        assert df["half"].iloc[0] == 50
        assert df["sum"].iloc[0] == 110

    def test_postprocess_with_failing_function(self):
        """Test that failing postprocess function doesn't crash."""
        rec = Recorder(mark="score")
        rec.append({"score": 10, "value": 100})

        def bad_func(row):
            return row["nonexistent_key"] * 2

        rec.postprocess_feature("bad", bad_func)

        df = rec.dataframe
        assert "bad" in df.columns
        assert df["bad"].iloc[0] is None

    def test_postprocess_before_append(self):
        """Test postprocess registered before any append."""
        rec = Recorder(mark="score")
        rec.postprocess_feature("computed", lambda row: row["value"] * 2)

        rec.append({"score": 10, "value": 50})

        df = rec.dataframe
        assert "computed" in df.columns
        assert df["computed"].iloc[0] == 100

    def test_get_best_target_underscore_fallback(self):
        """Test get_best_target falls back to underscore-prefixed column."""
        rec = Recorder(mark="score", mode="max")
        rec.append({"score": 10, "_computed": 100})
        rec.append({"score": 20, "_computed": 200})

        target_val, (idx, val) = rec.get_best_target("computed")
        assert target_val == 200
        assert idx == 1

    def test_postprocess_with_numpy_array_output(self):
        """Test postprocess feature that returns numpy array."""
        rec = Recorder(mark="score")
        rec.append({"score": 10, "v": np.array([1, 2, 3])})

        rec.postprocess_feature("scaled_v", lambda row: row["v"] * 2.0)

        df = rec.dataframe
        assert "scaled_v" in df.columns
        np.testing.assert_array_equal(df["scaled_v"].iloc[0], [2.0, 4.0, 6.0])
