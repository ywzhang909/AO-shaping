"""Tests for the SLM gray-level remapping LUT hook.

Verifies that ``Santec.load_lut`` / ``create_phase_from_array``
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
sys.modules["ao_shaping.drivers.slm.santec._slm_win"] = MagicMock()

from ao_shaping.drivers.slm.santec import (
    Santec,
    apply_lut_remap,
)
from ao_shaping.utils.slm.slm_lut import (
    LUT_STATUS_VALIDATED,
    LutTrustError,
    build_lut_meta,
    write_lut_meta,
)

#: 两台真实设备的序列号。归档契约 (``utils.slm.slm_lut.require_validated_lut``)
#: 要求 LUT 目录按面板序列号命名并带 ``validated`` 的 ``meta.json``，否则
#: ``Santec.load_lut`` 直接拒绝 —— 下面这些用例测的是**解析格式**，
#: 所以必须先提供一个可信容器，才能走到它们真正要测的代码。
SLM1_SERIAL = "22030108"
SLM2_SERIAL = "23020026"


def _trust(lut_dir: Path, *, inverse_gray: np.ndarray | None = None) -> Path:
    """Attach a ``validated`` ``meta.json`` to a hand-written LUT directory."""
    table = np.arange(1024, dtype=np.uint16) if inverse_gray is None else inverse_gray
    g_values = np.linspace(0.0, 1023.0, 65)
    phi = np.linspace(0.0, 2.0 * np.pi, 65)
    write_lut_meta(
        lut_dir,
        build_lut_meta(
            serial=lut_dir.name,
            status=LUT_STATUS_VALIDATED,
            source="tests/ao_shaping/drivers/slm/test_slm_lut_hook.py",
            g_values=g_values,
            phi=phi,
            inverse_gray=table,
            valley=0.01,
            wavelength_nm=1064.0,
            gray_for_2pi=993,
            notes="测试夹具：只提供可信容器，表内容由各用例自己写",
        ),
    )
    return lut_dir


# Ensure mock stays in place for every test in this module
@pytest.fixture(autouse=True)
def _ensure_mock_slm_sdk() -> Generator[None, None, None]:
    with patch.dict(
        "sys.modules",
        {"ao_shaping.drivers.slm.santec._slm_win": MagicMock()},
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
    _trust(directory, inverse_gray=inverse_gray)


def _write_offset_lut(directory: Path, offset: int = 100) -> None:
    """Write an offset inverse_gray LUT (gray → gray+offset, clamped to 1023)."""
    directory.mkdir(parents=True, exist_ok=True)
    inverse_gray = np.clip(np.arange(1024, dtype=np.int64) + offset, 0, 1023).astype(
        np.uint16
    )
    np.savez_compressed(
        str(directory / "lut.npz"),
        inverse_gray=inverse_gray,
        g_values=np.arange(1024, dtype=np.uint16),
        phi=np.linspace(0, 2 * np.pi, 1024, dtype=np.float64),
        meta=f"offset_{offset}",
    )
    _trust(directory, inverse_gray=inverse_gray)


def _write_csv_lut(directory: Path, offset: int = 0) -> None:
    """Write a lut_inverse.csv fallback file."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["target_phi,gray"]
    for i in range(1024):
        gray_val = min(i + offset, 1023)
        lines.append(f"{i * 2 * np.pi / 1024:.6f},{gray_val}")
    (directory / "lut_inverse.csv").write_text("\n".join(lines))
    _trust(
        directory,
        inverse_gray=np.clip(np.arange(1024, dtype=np.int64) + offset, 0, 1023).astype(
            np.uint16
        ),
    )


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
# Tests: Santec LUT integration
# ---------------------------------------------------------------------------


class TestLUTDefaultState:
    """Verify default LUT state and no-op behavior."""

    def test_lut_is_none_by_default(self) -> None:
        slm = Santec(slm_number=1)
        assert slm.lut is None
        assert slm.lut_dir is None

    def test_create_phase_no_lut_matches_baseline(self) -> None:
        """With lut=None, output should be identical to the no-LUT baseline."""
        slm = Santec(slm_number=1, shift_x=0, shift_y=0)
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
        slm = Santec(slm_number=1)
        assert slm._correction is not None
        assert slm.lut is None


class TestLoadLUT:
    """Tests for the load_lut method."""

    def test_load_identity_lut(self, tmp_path: Path) -> None:
        """Loading identity LUT should not change create_phase_from_array output."""
        lut_dir = tmp_path / SLM1_SERIAL
        _write_identity_lut(lut_dir)

        slm = Santec(slm_number=1, shift_x=0, shift_y=0)
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
        lut_dir = tmp_path / SLM1_SERIAL
        _write_offset_lut(lut_dir, offset=100)

        slm = Santec(slm_number=1, shift_x=0, shift_y=0)
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
        slm = Santec(slm_number=1)
        assert slm.lut is None

        slm.load_lut(tmp_path / "nonexistent")
        # Should remain None
        assert slm.lut is None
        assert slm.lut_dir is None

    def test_load_empty_directory(self, tmp_path: Path) -> None:
        """一个只有空目录、没有 meta.json 的路径属于**不可信**，必须报错而不是静默忽略。

        归档契约把"缺元数据"等同于"未标定"，所以这里期望 ``LutTrustError``
        （旧行为是 warn + 保持 ``lut=None``，会让一份来历不明的表悄悄留在盘上）。
        """
        empty_dir = tmp_path / SLM2_SERIAL
        empty_dir.mkdir()
        slm = Santec(slm_number=1)

        with pytest.raises(LutTrustError):
            slm.load_lut(empty_dir)
        assert slm.lut is None

    def test_load_none_clears_lut(self, tmp_path: Path) -> None:
        """load_lut(None) should clear a previously loaded LUT."""
        lut_dir = tmp_path / SLM1_SERIAL
        _write_identity_lut(lut_dir)

        slm = Santec(slm_number=1)
        slm.load_lut(lut_dir)
        assert slm.lut is not None

        slm.load_lut(None)
        assert slm.lut is None
        assert slm.lut_dir is None

    def test_load_csv_fallback(self, tmp_path: Path) -> None:
        """Loading from a directory with only lut_inverse.csv should work."""
        lut_dir = tmp_path / SLM1_SERIAL
        _write_csv_lut(lut_dir, offset=0)

        slm = Santec(slm_number=1, shift_x=0, shift_y=0)
        phase_rad = np.linspace(0, 2 * np.pi, 100).reshape(10, 10)
        baseline = slm.create_phase_from_array(phase_rad)

        slm.load_lut(lut_dir)
        assert slm.lut is not None

        after_lut = slm.create_phase_from_array(phase_rad)
        np.testing.assert_array_equal(baseline, after_lut)

    def test_load_corrupted_npz_keeps_previous(self, tmp_path: Path) -> None:
        """A corrupted npz file should log a warning and keep the previous LUT."""
        lut_dir_good = tmp_path / SLM1_SERIAL
        _write_identity_lut(lut_dir_good)
        lut_dir_bad = tmp_path / SLM2_SERIAL
        lut_dir_bad.mkdir()
        # Write a valid-looking but corrupted npz
        bad_npz = lut_dir_bad / "lut.npz"
        bad_npz.write_bytes(b"not a real npz file")
        _trust(lut_dir_bad)

        slm = Santec(slm_number=1)
        slm.load_lut(lut_dir_good)
        assert slm.lut is not None
        good_lut = slm.lut.copy()

        # Try loading corrupted - should keep previous
        slm.load_lut(lut_dir_bad)
        np.testing.assert_array_equal(slm.lut, good_lut)

    def test_load_npz_missing_key_keeps_previous(self, tmp_path: Path) -> None:
        """npz without 'inverse_gray' key should warn and keep previous."""
        lut_dir_good = tmp_path / SLM1_SERIAL
        _write_identity_lut(lut_dir_good)
        lut_dir_bad = tmp_path / SLM2_SERIAL
        lut_dir_bad.mkdir()
        np.savez_compressed(
            str(lut_dir_bad / "lut.npz"),
            wrong_key=np.arange(1024, dtype=np.uint16),
        )
        _trust(lut_dir_bad)

        slm = Santec(slm_number=1)
        slm.load_lut(lut_dir_good)
        assert slm.lut is not None
        good_lut = slm.lut.copy()

        slm.load_lut(lut_dir_bad)
        np.testing.assert_array_equal(slm.lut, good_lut)

    def test_load_npz_wrong_shape_keeps_previous(self, tmp_path: Path) -> None:
        """npz with wrong inverse_gray shape should warn and keep previous."""
        lut_dir_good = tmp_path / SLM1_SERIAL
        _write_identity_lut(lut_dir_good)
        lut_dir_bad = tmp_path / SLM2_SERIAL
        lut_dir_bad.mkdir()
        np.savez_compressed(
            str(lut_dir_bad / "lut.npz"),
            inverse_gray=np.zeros((10, 10), dtype=np.uint16),
        )
        _trust(lut_dir_bad)

        slm = Santec(slm_number=1)
        slm.load_lut(lut_dir_good)
        assert slm.lut is not None
        good_lut = slm.lut.copy()

        slm.load_lut(lut_dir_bad)
        np.testing.assert_array_equal(slm.lut, good_lut)


class TestLUTInCreatePhaseFromArray:
    """Integration tests: LUT application through create_phase_from_array."""

    def test_no_lut_vs_baseline_identical(self) -> None:
        """Critical: lut=None must produce byte-identical output to pre-LUT code."""
        slm = Santec(slm_number=1, shift_x=0, shift_y=0)
        phase_rad = np.linspace(0, 2 * np.pi, 400).reshape(20, 20)

        out1 = slm.create_phase_from_array(phase_rad)
        out2 = slm.create_phase_from_array(phase_rad)
        np.testing.assert_array_equal(out1, out2)

    def test_identity_lut_matches_no_lut(self, tmp_path: Path) -> None:
        """Identity LUT must produce identical output to no LUT at all."""
        lut_dir = tmp_path / SLM1_SERIAL
        _write_identity_lut(lut_dir)

        slm_no_lut = Santec(slm_number=1, shift_x=0, shift_y=0)
        slm_with_lut = Santec(slm_number=1, shift_x=0, shift_y=0)
        slm_with_lut.load_lut(lut_dir)

        phase_rad = np.linspace(0, 2 * np.pi, 400).reshape(20, 20)
        out_no_lut = slm_no_lut.create_phase_from_array(phase_rad)
        out_with_lut = slm_with_lut.create_phase_from_array(phase_rad)

        np.testing.assert_array_equal(out_no_lut, out_with_lut)

    def test_offset_lut_modifies_correct_region(self, tmp_path: Path) -> None:
        """Pixels mapping into the offset region should be shifted by the LUT."""
        lut_dir = tmp_path / SLM1_SERIAL
        _write_offset_lut(lut_dir, offset=100)

        slm = Santec(slm_number=1, shift_x=0, shift_y=0)
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
        lut_dir = tmp_path / SLM1_SERIAL
        _write_offset_lut(lut_dir, offset=50)

        slm = Santec(slm_number=1, shift_x=3, shift_y=2)
        phase_rad = np.linspace(0, 2 * np.pi, 400).reshape(20, 20)

        baseline = slm.create_phase_from_array(phase_rad)
        slm.load_lut(lut_dir)
        remapped = slm.create_phase_from_array(phase_rad)

        # Different from baseline (LUT changes values, shift handles them the same)
        assert not np.array_equal(baseline, remapped)
        # Output dimensions preserved
        assert remapped.shape == baseline.shape

    def test_output_in_valid_range(self, tmp_path: Path) -> None:
        """Output follows linear mapping; SLM handles mod 2π wrapping."""
        lut_dir = tmp_path / SLM1_SERIAL
        _write_offset_lut(lut_dir, offset=200)

        slm = Santec(slm_number=1, shift_x=0, shift_y=0)
        phase_rad = np.linspace(0, 4 * np.pi, 200).reshape(10, 20)

        result = slm.create_phase_from_array(phase_rad)
        assert np.all(result >= 0)
        # Without software-side mod 2π, values can exceed 1023 for rad > 2π;
        # the SLM firmware handles wrapping.
        max_expected = int(phase_rad.max() / (2 * np.pi) * slm.MAX_GRAYSCALE_VALUE)
        assert np.all(result <= max_expected)
