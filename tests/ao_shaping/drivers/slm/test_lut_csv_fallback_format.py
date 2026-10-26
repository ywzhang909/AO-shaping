"""The driver's LUT CSV fallback cannot parse the CSV that this repo's own writer emits.

``slm_lut.save_lut`` writes ``lut_inverse.csv`` through ``np.savetxt`` with no ``fmt=``, so
numpy's default ``%.18e`` puts values like ``1.000000000000000000e+02`` in the gray column.
``Santec.load_lut``'s fallback parses that file with ``dtype=np.uint16``, which cannot convert
scientific-notation floats; ``np.genfromtxt`` substitutes its invalid-fill value **instead of
raising**, and the resulting table still clears the existing ``ndim == 1 and size >= 1024``
guard. A corrupted LUT therefore reaches the panel with nothing but an info log to show for it.

The committed suite never sees this because its CSV fixture hand-writes the gray column as bare
integers -- a format no producer emits.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.modules["ao_shaping.drivers.slm.santec._slm_win"] = MagicMock()

from ao_shaping.drivers.slm.santec import Santec
from ao_shaping.utils.slm.slm_lut import save_lut

N_ENTRIES = 1024
GRAY_DOMAIN = 1023  # utils.slm.phase_display.DEFAULT_MAX_GRAYSCALE


@pytest.fixture(autouse=True)
def _ensure_mock_slm_sdk() -> Generator[None, None]:
    with patch.dict(
        "sys.modules",
        {"ao_shaping.drivers.slm.santec._slm_win": MagicMock()},
        clear=False,
    ):
        yield


def _lut_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g_values = np.arange(0, N_ENTRIES, 4, dtype=np.uint16)
    phi = np.linspace(0.0, 2 * np.pi, g_values.size)
    inverse_gray = np.clip(np.arange(N_ENTRIES, dtype=np.int64) + 100, 0, GRAY_DOMAIN).astype(
        np.uint16
    )
    return g_values, phi, inverse_gray


def _write_real_lut(directory: Path) -> np.ndarray:
    """Write a LUT through the canonical writer and return the table it stored."""
    g_values, phi, inverse_gray = _lut_arrays()
    save_lut(directory, g_values, phi, inverse_gray, meta={"method": "test"})
    return inverse_gray


class TestWriterReaderRoundTrip:
    def test_npz_path_loads_what_the_writer_stored(self, tmp_path: Path) -> None:
        """Control: the primary npz path is already correct."""
        expected = _write_real_lut(tmp_path)
        slm = Santec(slm_number=1)

        slm.load_lut(tmp_path)

        assert slm.lut is not None
        np.testing.assert_array_equal(slm.lut, expected)

    def test_csv_fallback_loads_what_the_writer_stored(self, tmp_path: Path) -> None:
        """The fallback must read the file the writer actually produces."""
        expected = _write_real_lut(tmp_path)
        (tmp_path / "lut.npz").unlink()
        slm = Santec(slm_number=1)

        slm.load_lut(tmp_path)

        assert slm.lut is not None, "the CSV fallback loaded nothing at all"
        assert slm.lut.size == expected.size, (
            f"fallback read {slm.lut.size} rows for a {expected.size}-row table"
        )
        np.testing.assert_array_equal(
            slm.lut,
            expected,
            err_msg=f"fallback parsed {slm.lut[:5]} where the writer stored {expected[:5]}",
        )

    def test_handwritten_integer_csv_still_loads(self, tmp_path: Path) -> None:
        """Back-compat anchor: the bare-integer format some fixtures hand-write."""
        lines = ["target_phi,gray"]
        for index in range(N_ENTRIES):
            gray = min(index + 100, GRAY_DOMAIN)
            lines.append(f"{index * 2 * np.pi / N_ENTRIES:.6f},{gray}")
        (tmp_path / "lut_inverse.csv").write_text("\n".join(lines), encoding="utf-8")
        expected = np.clip(np.arange(N_ENTRIES) + 100, 0, GRAY_DOMAIN).astype(np.uint16)
        slm = Santec(slm_number=1)

        slm.load_lut(tmp_path)

        assert slm.lut is not None
        np.testing.assert_array_equal(slm.lut, expected)


class TestDomainGuard:
    def test_gray_values_outside_the_panel_domain_are_rejected(self, tmp_path: Path) -> None:
        """A table of 65535s is corruption, not a LUT -- and it must not be adopted.

        This is the failure mode the parse bug produces, checked through the npz path so it is
        independent of how the CSV is read.
        """
        np.savez_compressed(
            tmp_path / "lut.npz",
            inverse_gray=np.full(N_ENTRIES, 65535, dtype=np.uint16),
        )
        slm = Santec(slm_number=1)

        slm.load_lut(tmp_path)

        assert slm.lut is None, (
            f"a table peaking at {int(np.asarray(slm.lut).max())} was adopted; the panel domain "
            f"is 0..{GRAY_DOMAIN}"
        )

    def test_in_domain_table_still_loads_through_npz(self, tmp_path: Path) -> None:
        """The guard must not reject a legitimate table."""
        expected = np.arange(N_ENTRIES, dtype=np.uint16)
        np.savez_compressed(tmp_path / "lut.npz", inverse_gray=expected)
        slm = Santec(slm_number=1)

        slm.load_lut(tmp_path)

        assert slm.lut is not None
        np.testing.assert_array_equal(slm.lut, expected)
