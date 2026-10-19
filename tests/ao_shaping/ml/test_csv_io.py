"""Regression tests for the shared CSV loader (ml.csv_io).

The loader used to be duplicated verbatim in ``ml.zernike.dataset`` and
``ml.phase.dataset`` and had no coverage at all; these tests pin the contract
both copies agreed on so the single source cannot drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.csv_io import load_zernike_coefficients
from ml.phase.dataset import load_zernike_coefficients as phase_alias
from ml.zernike.dataset import load_zernike_coefficients as zernike_alias


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "coeffs.csv"
    p.write_text(body, encoding="utf-8")
    return p


class TestSharedIdentity:
    def test_both_packages_reexport_the_same_object(self) -> None:
        """Dedup kept the public paths but collapsed them to one function."""
        assert zernike_alias is phase_alias is load_zernike_coefficients


class TestLoading:
    def test_reads_second_row_and_skips_header(self, tmp_path: Path) -> None:
        csv = _write(tmp_path, "noll1,noll2,noll3\n0.5,-1.25,2.0\n")
        out = load_zernike_coefficients(csv)
        assert out.tolist() == [0.5, -1.25, 2.0]
        assert out.dtype == np.float32

    def test_n_terms_truncates(self, tmp_path: Path) -> None:
        csv = _write(tmp_path, "a,b,c,d\n1,2,3,4\n")
        assert load_zernike_coefficients(csv, 2).tolist() == [1.0, 2.0]

    def test_n_terms_larger_than_row_is_not_padded(self, tmp_path: Path) -> None:
        csv = _write(tmp_path, "a,b\n1,2\n")
        out = load_zernike_coefficients(csv, 5)
        assert out.shape == (2,)


class TestRejectsMalformed:
    @pytest.mark.parametrize("body", ["", "a,b,c\n"])
    def test_header_only_or_empty_raises(self, tmp_path: Path, body: str) -> None:
        csv = _write(tmp_path, body)
        with pytest.raises(ValueError, match="Empty or invalid CSV"):
            load_zernike_coefficients(csv)

    def test_non_numeric_value_raises(self, tmp_path: Path) -> None:
        csv = _write(tmp_path, "a,b\nx,y\n")
        with pytest.raises(ValueError):
            load_zernike_coefficients(csv)
