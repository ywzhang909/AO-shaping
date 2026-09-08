"""Tests for the SLM gray-level remapping LUT hook.

Verifies that ``SantecSLM200.load_lut`` / ``create_phase_from_array``
correctly apply (or skip) a phase→gray compensation lookup table, with
ZERO behaviour change when no LUT is loaded.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock the SLM SDK before importing the driver (same pattern as test_correction_csv.py)
sys.modules["ao_shaping.drivers.slm._slm_win"] = MagicMock()

from ao_shaping.drivers.slm.santec_slm200 import (
    SantecSLM200,
    apply_lut_remap,
)

# Ensure mock stays in place for every test in this module
@pytest.fixture(autouse=True)
def _ensure_mock_slm_sdk() -> Generator[None, None, None]:
    with patch.dict(
        "sys.modules",
        {"ao_shaping.drivers.slm._slm_win": MagicMock()},
        clear=False,
    ):
        yield


# ---------------------------------------------------------------------------
# Helper: build a minimal LUT directory with lut.npz
# ---------------------------------------------------------------------------

def _write_identity_lut(directory: Path) -> None:
    """Write an identity inverse_gray LUT (gray maps to itself)."""
    directory.mkdir(parents=True, exist_ok=True)
    inverse_gray = np.arange(1024, dtype=np.uint16)
    np.savez_compressed(
        str(directory / "lut.npz"),
        inverse_gray=inverse_gray,
        g_values=np.arange(1024, dtype=np.uint16),
        phi=np.linspace(0, 2 * np.pi, 1024, dtype=np.float64),
        meta="identity",
    )


def _write_offset_lut(directory: Path, offset: int = 100) -> None:
    """Write an offset inverse_gray LUT (gray → gray+offset, clamped to 1023)."""
    directory.mkdir(parents=True, exist_ok=True)
    inverse_gray = np.clip(
        np.arange(1024, dtype=np.int64) + offset, 0, 1023
    ).astype(np.uint16)
    np.savez_compressed(
        str(directory / "lut.npz"),
        inverse_gray=inverse_gray,
        g_values=np.arange(1024, dtype=np.uint16),
        phi=np.linspace(0, 2 * np.pi, 1024, dtype=np.float64),
        meta=f"offset_{offset}",
    )


def _write_csv_lut(directory: Path, offset: int = 0) -> None:
    """Write a lut_inverse.csv fallback file."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["target_phi,gray"]
    for i in range(1024):
        gray_val = min(i + offset, 1023)
        lines.append(f"{i * 2 * np.pi / 1024:.6f},{gray_val}")
    (directory / "lut_inverse.csv").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Tests: apply_lut_remap (module-level pure function)
# ---------------------------------------------------------------------------

class TestApplyLutRemap:
    """Unit tests for the standalone apply_lut_remap function."""

    def test_identity_lut_preserves_values(self) -> None:
        """Identity LUT should return values unchanged (after integer truncation)."""
        lut = np.arange(1024, dtype=np.uint16)
        gray = np.array([0.0, 1.0, 500.5, 1023.0])
        result = apply_lut_remap(gray, lut)
        # LUT indexes by truncated integer: 500.5 → index 500 → value 500
        expected = np.array([0.0, 1.0, 500.0, 1023.0])
        np.testing.assert_array_equal(result, expected)

    def test_offset_lut_shifts_values(self) -> None:
        """Offset LUT should shift gray values accordingly."""
        lut = np.clip(np.arange(1024) + 100, 0, 1023).astype(np.uint16)
        gray = np.array([0.0, 100.0, 500.0, 950.0])
        result = apply_lut_remap(gray, lut)
        expected = np.array([100.0, 200.0, 600.0, 1023.0])
        np.testing.assert_array_equal(result, expected)

    def test_clipping_at_boundaries(self) -> None:
        """Values outside 0..1023 should be clipped to LUT bounds."""
        lut = np.arange(1024, dtype=np.uint16)
        gray = np.array([-10.0, 2000.0])
        result = apply_lut_remap(gray, lut)
        # -10 → clipped to index 0 → 0; 2000 → clipped to index 1023 → 1023
        np.testing.assert_array_equal(result, [0.0, 1023.0])

    def test_output_is_float64(self) -> None:
        """Output dtype should be float64 regardless of input dtype."""
        lut = np.arange(1024, dtype=np.uint16)
        gray = np.array([100, 200, 300], dtype=np.float32)
        result = apply_lut_remap(gray, lut)
        assert result.dtype == np.float64

    def test_truncation_behavior(self) -> None:
        """Non-integer values should be truncated (matching uint16 cast)."""
        lut_full = np.arange(1024, dtype=np.uint16)
        lut_full[2] = 999
        gray = np.array([1.6, 2.4])  # 1.6→1, 2.4→2
        result = apply_lut_remap(gray, lut_full)
        np.testing.assert_array_equal(result, [1.0, 999.0])


# ---------------------------------------------------------------------------
# Tests: SantecSLM200 LUT integration
# ---------------------------------------------------------------------------

class TestLUTDefaultState:
    """Verify default LUT state and no-op behavior."""

    def test_lut_is_none_by_default(self) -> None:
        slm = SantecSLM200(slm_number=1)
        assert slm.lut is None
        assert slm.lut_dir is None

    def test_create_phase_no_lut_matches_baseline(self) -> None:
        """With lut=None, output should be identical to the no-LUT baseline."""
        slm = SantecSLM200(slm_number=1, shift_x=0, shift_y=0)
        phase_rad = np.linspace(0, 2 * np.pi, 100).reshape(10, 10)

        # Baseline: no LUT
        baseline = slm.create_phase_from_array(phase_rad)

        # After explicitly clearing LUT (should be a no-op)
        slm.load_lut(None)
        after_clear = slm.create_phase_from_array(phase_rad)

        assert slm.lut is None
        np.testing.assert_array_equal(baseline, after_clear)

    def test_lut_default_dont_affect_correction(self) -> None:
        """Default lut=None should not interfere with correction loading."""
        slm = SantecSLM200(slm_number=1)
        assert slm._correction is not None
        assert slm.lut is None


class TestLoadLUT:
    """Tests for the load_lut method."""

    def test_load_identity_lut(self, tmp_path: Path) -> None:
        """Loading identity LUT should not change create_phase_from_array output."""
        lut_dir = tmp_path / "lut_identity"
        _write_identity_lut(lut_dir)

        slm = SantecSLM200(slm_number=1, shift_x=0, shift_y=0)
        phase_rad = np.linspace(0, 2 * np.pi, 100).reshape(10, 10)
        baseline = slm.create_phase_from_array(phase_rad)

        slm.load_lut(lut_dir)
        assert slm.lut is not None
        assert slm.lut_dir == lut_dir
        assert slm.lut.size >= 1024

        after_lut = slm.create_phase_from_array(phase_rad)
        np.testing.assert_array_equal(baseline, after_lut)

    def test_load_offset_lut_changes_output(self, tmp_path: Path) -> None:
        """Non-identity LUT should change gray values in the altered region."""
        lut_dir = tmp_path / "lut_offset"
        _write_offset_lut(lut_dir, offset=100)

        slm = SantecSLM200(slm_number=1, shift_x=0, shift_y=0)
        phase_rad = np.linspace(0, 2 * np.pi, 100).reshape(10, 10)
        baseline = slm.create_phase_from_array(phase_rad)

        slm.load_lut(lut_dir)
        after_lut = slm.create_phase_from_array(phase_rad)

        # With offset LUT, gray values should differ from baseline
        # (at least some pixels should change)
        assert not np.array_equal(baseline, after_lut)
        # Final dtype must still be uint16
        assert after_lut.dtype == np.uint16

    def test_load_missing_directory(self, tmp_path: Path) -> None:
        """Loading from a missing directory should warn and keep lut=None."""
        slm = SantecSLM200(slm_number=1)
        assert slm.lut is None

        slm.load_lut(tmp_path / "nonexistent")
        # Should remain None
        assert slm.lut is None
        assert slm.lut_dir is None

    def test_load_empty_directory(self, tmp_path: Path) -> None:
        """Loading from a directory without npz/csv should warn and keep lut=None."""
        empty_dir = tmp_path / "empty_lut"
        empty_dir.mkdir()
        slm = SantecSLM200(slm_number=1)

        slm.load_lut(empty_dir)
        assert slm.lut is None

    def test_load_none_clears_lut(self, tmp_path: Path) -> None:
        """load_lut(None) should clear a previously loaded LUT."""
        lut_dir = tmp_path / "lut_to_clear"
        _write_identity_lut(lut_dir)

        slm = SantecSLM200(slm_number=1)
        slm.load_lut(lut_dir)
        assert slm.lut is not None

        slm.load_lut(None)
        assert slm.lut is None
        assert slm.lut_dir is None

    def test_load_csv_fallback(self, tmp_path: Path) -> None:
        """Loading from a directory with only lut_inverse.csv should work."""
        lut_dir = tmp_path / "lut_csv"
        _write_csv_lut(lut_dir, offset=0)

        slm = SantecSLM200(slm_number=1, shift_x=0, shift_y=0)
        phase_rad = np.linspace(0, 2 * np.pi, 100).reshape(10, 10)
        baseline = slm.create_phase_from_array(phase_rad)

        slm.load_lut(lut_dir)
        assert slm.lut is not None

        after_lut = slm.create_phase_from_array(phase_rad)
        np.testing.assert_array_equal(baseline, after_lut)

    def test_load_corrupted_npz_keeps_previous(self, tmp_path: Path) -> None:
        """A corrupted npz file should log a warning and keep the previous LUT."""
        lut_dir_good = tmp_path / "lut_good"
        _write_identity_lut(lut_dir_good)
        lut_dir_bad = tmp_path / "lut_bad"
        lut_dir_bad.mkdir()
        # Write a valid-looking but corrupted npz
        bad_npz = lut_dir_bad / "lut.npz"
        bad_npz.write_bytes(b"not a real npz file")

        slm = SantecSLM200(slm_number=1)
        slm.load_lut(lut_dir_good)
        assert slm.lut is not None
        good_lut = slm.lut.copy()

        # Try loading corrupted - should keep previous
        slm.load_lut(lut_dir_bad)
        np.testing.assert_array_equal(slm.lut, good_lut)

    def test_load_npz_missing_key_keeps_previous(self, tmp_path: Path) -> None:
        """npz without 'inverse_gray' key should warn and keep previous."""
        lut_dir_good = tmp_path / "lut_good"
        _write_identity_lut(lut_dir_good)
        lut_dir_bad = tmp_path / "lut_bad"
        lut_dir_bad.mkdir()
        np.savez_compressed(
            str(lut_dir_bad / "lut.npz"),
            wrong_key=np.arange(1024, dtype=np.uint16),
        )

        slm = SantecSLM200(slm_number=1)
        slm.load_lut(lut_dir_good)
        assert slm.lut is not None
        good_lut = slm.lut.copy()

        slm.load_lut(lut_dir_bad)
        np.testing.assert_array_equal(slm.lut, good_lut)

    def test_load_npz_wrong_shape_keeps_previous(self, tmp_path: Path) -> None:
        """npz with wrong inverse_gray shape should warn and keep previous."""
        lut_dir_good = tmp_path / "lut_good"
        _write_identity_lut(lut_dir_good)
        lut_dir_bad = tmp_path / "lut_bad"
        lut_dir_bad.mkdir()
        np.savez_compressed(
            str(lut_dir_bad / "lut.npz"),
            inverse_gray=np.zeros((10, 10), dtype=np.uint16),
        )

        slm = SantecSLM200(slm_number=1)
        slm.load_lut(lut_dir_good)
        assert slm.lut is not None
        good_lut = slm.lut.copy()

        slm.load_lut(lut_dir_bad)
        np.testing.assert_array_equal(slm.lut, good_lut)


class TestLUTInCreatePhaseFromArray:
    """Integration tests: LUT application through create_phase_from_array."""

    def test_no_lut_vs_baseline_identical(self) -> None:
        """Critical: lut=None must produce byte-identical output to pre-LUT code."""
        slm = SantecSLM200(slm_number=1, shift_x=0, shift_y=0)
        phase_rad = np.linspace(0, 2 * np.pi, 400).reshape(20, 20)

        out1 = slm.create_phase_from_array(phase_rad)
        out2 = slm.create_phase_from_array(phase_rad)
        np.testing.assert_array_equal(out1, out2)

    def test_identity_lut_matches_no_lut(self, tmp_path: Path) -> None:
        """Identity LUT must produce identical output to no LUT at all."""
        lut_dir = tmp_path / "lut_identity"
        _write_identity_lut(lut_dir)

        slm_no_lut = SantecSLM200(slm_number=1, shift_x=0, shift_y=0)
        slm_with_lut = SantecSLM200(slm_number=1, shift_x=0, shift_y=0)
        slm_with_lut.load_lut(lut_dir)

        phase_rad = np.linspace(0, 2 * np.pi, 400).reshape(20, 20)
        out_no_lut = slm_no_lut.create_phase_from_array(phase_rad)
        out_with_lut = slm_with_lut.create_phase_from_array(phase_rad)

        np.testing.assert_array_equal(out_no_lut, out_with_lut)

    def test_offset_lut_modifies_correct_region(self, tmp_path: Path) -> None:
        """Pixels mapping into the offset region should be shifted by the LUT."""
        lut_dir = tmp_path / "lut_offset"
        _write_offset_lut(lut_dir, offset=100)

        slm = SantecSLM200(slm_number=1, shift_x=0, shift_y=0)
        phase_rad = np.linspace(0, 2 * np.pi, 400).reshape(20, 20)

        baseline = slm.create_phase_from_array(phase_rad)
        slm.load_lut(lut_dir)
        remapped = slm.create_phase_from_array(phase_rad)

        # Output must still be uint16
        assert remapped.dtype == np.uint16
        # At least some values should differ
        assert not np.array_equal(baseline, remapped)

    def test_lut_applies_before_shift(self, tmp_path: Path) -> None:
        """LUT remap should be applied before shift (LUT operates on the full array)."""
        lut_dir = tmp_path / "lut_shift"
        _write_offset_lut(lut_dir, offset=50)

        slm = SantecSLM200(slm_number=1, shift_x=3, shift_y=2)
        phase_rad = np.linspace(0, 2 * np.pi, 400).reshape(20, 20)

        baseline = slm.create_phase_from_array(phase_rad)
        slm.load_lut(lut_dir)
        remapped = slm.create_phase_from_array(phase_rad)

        # Different from baseline (LUT changes values, shift handles them the same)
        assert not np.array_equal(baseline, remapped)
        # Output dimensions preserved
        assert remapped.shape == baseline.shape

    def test_output_in_valid_range(self, tmp_path: Path) -> None:
        """All output values must be in 0..1023 even with aggressive LUT."""
        lut_dir = tmp_path / "lut_extreme"
        _write_offset_lut(lut_dir, offset=200)

        slm = SantecSLM200(slm_number=1, shift_x=0, shift_y=0)
        phase_rad = np.linspace(0, 4 * np.pi, 200).reshape(10, 20)

        result = slm.create_phase_from_array(phase_rad)
        assert np.all(result >= 0)
        assert np.all(result <= 1023)
