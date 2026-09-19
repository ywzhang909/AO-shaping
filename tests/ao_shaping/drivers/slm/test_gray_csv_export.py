"""Regression-anchor tests for :meth:`Santec.save_gray_to_csv` (D-track).

Pins the EXACT contract of the grayscale CSV exporter: output must be loadable
by :meth:`Santec.load_gray_from_csv` (roundtrip), follow the Santec
correction-file layout (``Y/X`` header + row/col indices), and reject invalid
input (wrong shape, non-integer, out-of-range).

All tests are offline: the SLM SDK is mocked, panel resolution is monkeypatched
to a small value so roundtrips are cheap.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.modules["ao_shaping.drivers.slm.santec._slm_win"] = MagicMock()

from ao_shaping.drivers.slm.santec import driver as _santec_driver
from ao_shaping.drivers.slm.santec import Santec


@pytest.fixture(autouse=True)
def _ensure_mock_slm_sdk() -> None:
    with patch.dict(
        "sys.modules",
        {"ao_shaping.drivers.slm.santec._slm_win": MagicMock()},
        clear=False,
    ):
        yield


@pytest.fixture(autouse=True)
def _small_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the panel resolution so roundtrips are cheap.

    PANEL_RES = (width, height); data shape is (height, width).
    """
    monkeypatch.setattr(_santec_driver, "PANEL_RES", (16, 12))


class TestSaveGrayToCSV:
    def test_roundtrip_load_gray_from_csv(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        gray = np.random.default_rng(42).integers(0, 1024, size=(12, 16)).astype(
            np.uint16
        )
        path = tmp_path / "roundtrip.csv"

        Santec.save_gray_to_csv(gray, path)
        loaded = Santec.load_gray_from_csv(path)

        assert loaded.shape == gray.shape
        assert loaded.dtype == np.uint16
        np.testing.assert_array_equal(loaded, gray)

    def test_header_and_index_layout(self, tmp_path: pytest.TempPathFactory) -> None:
        gray = np.zeros((12, 16), dtype=np.uint16)
        path = tmp_path / "layout.csv"

        Santec.save_gray_to_csv(gray, path)

        text = path.read_text(encoding="utf-8")
        lines = text.strip().splitlines()
        header = lines[0].split(",")
        assert header[0] == "Y/X"
        assert header[1:] == [str(i) for i in range(16)]
        # row index column: lines[1..] first field == 0..11
        assert [line.split(",")[0] for line in lines[1:]] == [
            str(i) for i in range(12)
        ]
        # data region is integers (no float notation)
        assert all("," not in field or field.isdigit() for field in text.split(","))

    def test_value_range_enforced(self, tmp_path: pytest.TempPathFactory) -> None:
        out_of_range = np.full((12, 16), 1024, dtype=np.uint16)
        with pytest.raises(ValueError, match="灰度"):
            Santec.save_gray_to_csv(out_of_range, tmp_path / "bad_max.csv")

        negative = np.full((12, 16), -1, dtype=np.int32)
        with pytest.raises(ValueError, match="灰度"):
            Santec.save_gray_to_csv(negative, tmp_path / "bad_min.csv")

    def test_non_integer_values_rejected(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        float_gray = np.full((12, 16), 128.5)
        with pytest.raises(ValueError, match="整数"):
            Santec.save_gray_to_csv(float_gray, tmp_path / "float.csv")

    def test_wrong_shape_rejected(self, tmp_path: pytest.TempPathFactory) -> None:
        too_small = np.zeros((6, 6), dtype=np.uint16)
        with pytest.raises(ValueError, match="尺寸"):
            Santec.save_gray_to_csv(too_small, tmp_path / "small.csv")

    def test_parent_dir_created(self, tmp_path: pytest.TempPathFactory) -> None:
        gray = np.zeros((12, 16), dtype=np.uint16)
        path = tmp_path / "nested" / "sub" / "gray.csv"

        Santec.save_gray_to_csv(gray, path)

        assert path.exists()