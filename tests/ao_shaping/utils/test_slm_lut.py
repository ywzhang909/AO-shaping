"""Offline tests for the pure-math SLM gray-to-phase LUT module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.utils.slm.slm_lut import (
    LUT_META_FILENAME,
    LUT_STATUS_PROVISIONAL,
    LUT_STATUS_VALIDATED,
    LUTData,
    LutSerialError,
    LutTrustError,
    blaze_ramp_gray,
    build_inverse_lut,
    build_lut_meta,
    depth_pattern,
    invert_checkerboard_scan,
    invert_depth_scan,
    invert_offset_scan,
    load_lut,
    lut_dir_for_serial,
    mark_lut_validated,
    normalize_efficiency,
    normalize_slm_serial,
    offset_pattern,
    read_lut_meta,
    require_validated_lut,
    save_lut,
    save_serial_lut,
    sinc_inv,
    stack_halves,
    write_lut_calibration,
    write_lut_meta,
)


class TestBlazeRampGray:
    """Tests for blaze_ramp_gray."""

    def test_shape_and_range(self):
        period, peak_gray, width = 256, 100, 512
        ramp = blaze_ramp_gray(period, peak_gray, width)
        assert ramp.shape == (1, width)
        assert ramp.min() >= 0
        assert ramp.max() <= peak_gray

    def test_max_equals_peak_gray(self):
        period, peak_gray, width = 256, 100, 512
        ramp = blaze_ramp_gray(period, peak_gray, width)
        assert ramp.max() == peak_gray

    def test_monotonic_within_period(self):
        period, peak_gray, width = 256, 100, 512
        ramp = blaze_ramp_gray(period, peak_gray, width)
        first = ramp[0, :period]
        assert np.all(np.diff(first) >= 0)

    def test_period_repeats(self):
        period, peak_gray, width = 256, 100, 512
        ramp = blaze_ramp_gray(period, peak_gray, width)
        np.testing.assert_array_equal(ramp[0, :period], ramp[0, period : 2 * period])


class TestDepthPattern:
    """Tests for depth_pattern."""

    def test_dtype_and_shape(self):
        pattern = depth_pattern(period=16, peak_gray=1023, height=100, width=200)
        assert pattern.dtype == np.uint16
        assert pattern.shape == (100, 200)

    def test_value_range(self):
        peak_gray = 1023
        pattern = depth_pattern(period=16, peak_gray=peak_gray, height=50, width=100)
        assert pattern.min() >= 0
        assert pattern.max() <= peak_gray

    def test_rows_identical(self):
        pattern = depth_pattern(period=16, peak_gray=1023, height=10, width=64)
        for r in range(1, pattern.shape[0]):
            np.testing.assert_array_equal(pattern[r], pattern[0])


class TestOffsetPattern:
    """Tests for offset_pattern."""

    def test_dtype_and_shape(self):
        pattern = offset_pattern(
            period=16, peak_gray=1023, gray_offset=100, bits=10, height=100, width=200
        )
        assert pattern.dtype == np.uint16
        assert pattern.shape == (100, 200)

    def test_value_range_within_bits(self):
        bits = 10
        pattern = offset_pattern(
            period=16, peak_gray=1023, gray_offset=100, bits=bits, height=50, width=100
        )
        assert pattern.min() >= 0
        assert pattern.max() < 2**bits

    def test_modulo_wrap(self):
        # gray_offset pushes values past 2**bits -> wraps around
        bits = 10
        period, peak_gray, gray_offset = 16, 1023, 900
        pattern = offset_pattern(
            period=period,
            peak_gray=peak_gray,
            gray_offset=gray_offset,
            bits=bits,
            height=1,
            width=period,
        )
        ramp = blaze_ramp_gray(period, peak_gray, period)
        expected = np.mod(ramp + gray_offset, 2**bits).astype(np.uint16)
        np.testing.assert_array_equal(pattern[0], expected[0])


class TestStackHalves:
    """Tests for stack_halves."""

    def test_stack_vertical(self):
        h, w = 20, 30
        ref = np.full((h // 2, w), 1, dtype=np.uint16)
        test = np.full((h // 2, w), 2, dtype=np.uint16)
        out = stack_halves(ref, test)
        assert out.shape == (h, w)
        np.testing.assert_array_equal(out[: h // 2], ref)
        np.testing.assert_array_equal(out[h // 2 :], test)

    def test_mismatched_axis_raises(self):
        ref = np.zeros((10, 20))
        test = np.zeros((10, 25))  # width mismatch
        with pytest.raises(ValueError):
            stack_halves(ref, test)


class TestNormalizeEfficiency:
    """Tests for normalize_efficiency."""

    def test_normalize(self):
        eta = np.array([0.0, 2.0, 4.0, 1.0])
        out = normalize_efficiency(eta)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0, 0.25])

    def test_zero_max_returns_zeros(self):
        eta = np.array([0.0, 0.0, 0.0])
        out = normalize_efficiency(eta)
        np.testing.assert_array_equal(out, np.zeros(3))

    def test_negative_max_returns_zeros(self):
        eta = np.array([-1.0, -2.0])
        out = normalize_efficiency(eta)
        np.testing.assert_array_equal(out, np.zeros(2))


class TestSincInv:
    """Tests for sinc_inv."""

    def test_inverse_roundtrip(self):
        for y in [0.01, 0.1, 0.5, 0.9, 0.999, 1.0]:
            x = sinc_inv(np.array([y]))[0]
            assert x <= 0
            np.testing.assert_allclose(np.sinc(x), y, atol=1e-6)

    def test_y_equals_one_is_zero(self):
        x = sinc_inv(np.array([1.0]))[0]
        assert x == 0.0

    def test_y_leq_zero_is_neg_one(self):
        x = sinc_inv(np.array([0.0]))[0]
        assert x == -1.0

    def test_y_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            sinc_inv(np.array([1.5]))

    def test_nan_propagates(self):
        x = sinc_inv(np.array([np.nan]))[0]
        assert np.isnan(x)

    def test_vectorized(self):
        ys = np.array([0.1, 0.5, 0.9])
        xs = sinc_inv(ys)
        np.testing.assert_allclose(np.sinc(xs), ys, atol=1e-6)


class TestInvertDepthScan:
    """Tests for invert_depth_scan (the critical recovery test)."""

    def test_recovers_linear_response(self):
        gray_for_2pi = 993
        g_values = np.arange(0, 1024, 16, dtype=float)
        phi_lin = 2 * np.pi * g_values / gray_for_2pi
        # Simulate blazed-grating efficiency with tiny noise.
        u = (phi_lin - 2 * np.pi) / (2 * np.pi)
        rng = np.random.default_rng(0)
        eta = np.sinc(u) ** 2 + 1e-6 * rng.standard_normal(g_values.size)
        phi_rec = invert_depth_scan(eta)
        np.testing.assert_allclose(phi_rec, phi_lin, atol=2e-2)

    def test_peak_is_two_pi(self):
        eta = np.array([0.0, 0.5, 1.0, 0.5, 0.2])
        phi = invert_depth_scan(eta)
        assert phi[2] == pytest.approx(2 * np.pi)

    def test_output_float64(self):
        eta = np.array([0.0, 0.5, 1.0, 0.5, 0.2])
        phi = invert_depth_scan(eta)
        assert phi.dtype == np.float64


class TestInvertOffsetScan:
    """Tests for invert_offset_scan."""

    def test_flat_eta_monotone_and_arcsin(self):
        gray_for_2pi = 993
        # Scan within [0, gray_for_2pi] so no saturation region is entered.
        g_values = np.arange(0, gray_for_2pi + 1, 16, dtype=float)
        eta = np.full(g_values.size, 5.0)  # constant
        phi = invert_offset_scan(eta, g_values, gray_for_2pi)
        assert np.all(np.diff(phi) >= 0)
        # dphi = 2*arcsin(sqrt(1)) = pi for constant eta
        i = 10
        expected = 2 * np.pi * (g_values[i] / gray_for_2pi) + np.pi
        assert phi[i] == pytest.approx(expected, rel=1e-6)

    def test_saturation_after_gray_for_2pi(self):
        gray_for_2pi = 993
        g_values = np.arange(0, 1024, 16, dtype=float)
        eta = np.full(g_values.size, 5.0)
        phi = invert_offset_scan(eta, g_values, gray_for_2pi)
        mask = g_values > gray_for_2pi
        if np.any(mask):
            np.testing.assert_allclose(phi[mask], 2 * np.pi, atol=1e-6)


class TestInvertCheckerboardScan:
    """Tests for invert_checkerboard_scan (two-branch 0..2*pi recovery)."""

    @staticmethod
    def _retained(phase: np.ndarray) -> np.ndarray:
        """Ideal checkerboard zero-order retained fraction for phase ``phase``."""
        return np.cos(phase / 2.0) ** 2

    def test_recovers_linear_phase_over_full_span(self):
        g_values = np.linspace(0.0, 1000.0, 101)
        phi_true = 2 * np.pi * g_values / g_values[-1]
        phi = invert_checkerboard_scan(self._retained(phi_true))
        assert phi.shape == phi_true.shape
        assert phi.dtype == np.float64
        np.testing.assert_allclose(phi, phi_true, atol=1e-9)
        assert phi[0] == pytest.approx(0.0, abs=1e-9)
        assert phi[-1] == pytest.approx(2 * np.pi, abs=1e-9)

    def test_recovers_quadratic_phase_over_full_span(self):
        # Phase quadratic in gray (saturating LC response), still monotone 0..2*pi.
        frac = np.linspace(0.0, 1.0, 64)
        phi_true = 2 * np.pi * frac**2
        r = self._retained(phi_true)
        phi = invert_checkerboard_scan(r)
        m = int(np.argmin(r))
        # Every point but the branch point is recovered exactly; the branch
        # point is pinned to pi, which costs at most one sample step there.
        keep = np.ones(phi.size, dtype=bool)
        keep[m] = False
        np.testing.assert_allclose(phi[keep], phi_true[keep], atol=1e-9)
        assert phi[m] == np.pi
        assert abs(phi[m] - phi_true[m]) <= phi_true[m + 1] - phi_true[m - 1]
        assert phi[0] == pytest.approx(0.0, abs=1e-9)
        assert phi[-1] == pytest.approx(2 * np.pi, abs=1e-9)
        assert np.all(np.diff(phi) >= 0.0)

    def test_branch_point_is_pi_and_branches_stay_on_their_side(self):
        phase = np.linspace(0.0, 2 * np.pi, 41)
        r = self._retained(phase)
        phi = invert_checkerboard_scan(r)
        m = int(np.argmin(r))
        assert 0 < m < r.size - 1
        assert phi[m] == np.pi
        assert np.all(phi[:m] < np.pi)
        assert np.all(phi[m + 1 :] > np.pi)
        assert np.all(np.diff(phi) > 0)

    def test_noisy_scan_stays_nondecreasing(self):
        rng = np.random.default_rng(7)
        g_values = np.linspace(0.0, 1000.0, 121)
        phi_true = 2 * np.pi * g_values / g_values[-1]
        r = self._retained(phi_true) + 0.01 * rng.standard_normal(g_values.size)
        phi = invert_checkerboard_scan(r)
        assert np.all(np.diff(phi) >= 0.0)
        assert np.all((phi >= 0.0) & (phi <= 2 * np.pi))
        # Within the noise cone: r +-0.01 moves phase by at most ~2*sqrt(0.01).
        np.testing.assert_allclose(phi, phi_true, atol=0.5)

    def test_over_unity_clipped_not_rescaled(self):
        # Drift can push ratios above 1; those points must clip to 1 (phase 0/2*pi).
        base = self._retained(np.linspace(0.0, 2 * np.pi, 41))
        r = base.copy()
        r[0] = 1.35
        r[-1] = 1.08
        np.testing.assert_allclose(
            invert_checkerboard_scan(r), invert_checkerboard_scan(base), atol=1e-12
        )

    def test_negative_clipped_like_zero(self):
        base = self._retained(np.linspace(0.0, 2 * np.pi, 41))
        m = int(np.argmin(base))
        r = base.copy()
        r[m] = -0.25
        phi_neg = invert_checkerboard_scan(r)
        r[m] = 0.0
        np.testing.assert_allclose(phi_neg, invert_checkerboard_scan(r), atol=1e-12)
        assert phi_neg[m] == np.pi

    def test_not_normalized_by_max(self):
        # Uniform contrast loss (max ratio 0.5) must shrink the span, not be
        # rescaled back to [0, 2*pi].
        r = 0.5 * self._retained(np.linspace(0.0, 2 * np.pi, 41))
        phi = invert_checkerboard_scan(r)
        assert phi[0] == pytest.approx(np.pi / 2.0, abs=1e-9)
        assert phi[-1] == pytest.approx(3 * np.pi / 2.0, abs=1e-9)

    @pytest.mark.parametrize(
        "bad",
        [np.zeros((3, 1)), np.zeros((2, 3)), [[0.1, 0.2], [0.3, 0.4]]],
    )
    def test_non_1d_raises(self, bad):
        with pytest.raises(ValueError, match="1-D"):
            invert_checkerboard_scan(bad)

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_short_scan_raises(self, n):
        with pytest.raises(ValueError, match="at least 3"):
            invert_checkerboard_scan(np.full(n, 0.5))

    @pytest.mark.parametrize(
        "bad", [[0.5, np.nan, 0.1], [0.5, np.inf, 0.1], [-np.inf, 0.5, 0.1]]
    )
    def test_nonfinite_raises(self, bad):
        with pytest.raises(ValueError, match="finite"):
            invert_checkerboard_scan(np.array(bad, dtype=float))

    def test_endpoint_minimum_raises(self):
        # Purely falling / rising scan: no pi crossing, so the two-branch range
        # is not identifiable.
        with pytest.raises(ValueError, match="interior"):
            invert_checkerboard_scan(np.linspace(1.0, 0.0, 7))
        with pytest.raises(ValueError, match="interior"):
            invert_checkerboard_scan(np.linspace(0.0, 1.0, 7))

    def test_feeds_build_inverse_lut(self):
        # Integer grays so the LUT round-trips exactly through uint16.
        g_values = np.arange(0, 1009, 16, dtype=float)  # 0..1008, 64 points
        phi = invert_checkerboard_scan(
            self._retained(2 * np.pi * g_values / g_values[-1])
        )
        t_grid, gray_lut = build_inverse_lut(phi, g_values, n=1024)
        assert t_grid.shape == (1024,)
        assert gray_lut.shape == (1024,)
        assert gray_lut.dtype == np.uint16
        assert np.all(np.diff(gray_lut) >= 0)
        # Endpoints of the phase grid map back to the ends of the gray scan.
        assert gray_lut[0] == int(g_values[0])
        assert gray_lut[-1] == int(g_values[-1])
        # The pi grid point lands on the branch point gray.
        branch = int(np.argmin(np.abs(t_grid - np.pi)))
        assert gray_lut[branch] == int(g_values[np.argmin(np.abs(phi - np.pi))])
        # Every entry follows the argmin rule of build_inverse_lut.
        for i in (0, 100, 300, 512, 700, 1023):
            idx = int(np.argmin(np.abs(phi - t_grid[i])))
            assert gray_lut[i] == g_values[idx]


class TestBuildInverseLut:
    """Tests for build_inverse_lut."""

    def test_monotone_phi(self):
        g_values = np.arange(0, 1024, 16)
        phi = 2 * np.pi * g_values / 993.0
        t, gray_lut = build_inverse_lut(phi, g_values, n=256)
        assert gray_lut.dtype == np.uint16
        assert t.shape == (256,)
        assert gray_lut.shape == (256,)
        assert np.all(np.diff(gray_lut) >= 0)
        # matches argmin rule
        for i in range(256):
            idx = np.argmin(np.abs(phi - t[i]))
            assert gray_lut[i] == g_values[idx]

    def test_non_monotone_raises(self):
        phi = np.array([0.0, 3.0, 2.0, 4.0])
        g_values = np.arange(4)
        with pytest.raises(ValueError):
            build_inverse_lut(phi, g_values)


class TestSaveLoadLut:
    """Tests for save_lut / load_lut roundtrip."""

    def test_roundtrip_npz(self, tmp_path):
        g_values = np.arange(0, 1024, 16, dtype=float)
        phi = 2 * np.pi * g_values / 993.0
        inverse_gray = np.arange(64, dtype=np.uint16)
        meta = {"gray_for_2pi": 993, "wavelength_nm": 1064}
        save_lut(tmp_path, g_values, phi, inverse_gray, meta)
        data = load_lut(tmp_path)
        assert isinstance(data, LUTData)
        np.testing.assert_array_equal(data.g_values, g_values)
        np.testing.assert_array_equal(data.phi, phi)
        np.testing.assert_array_equal(data.inverse_gray, inverse_gray)
        assert data.meta == meta

    def test_roundtrip_csv_fallback(self, tmp_path):
        g_values = np.arange(0, 1024, 16, dtype=float)
        phi = 2 * np.pi * g_values / 993.0
        inverse_gray = np.arange(64, dtype=np.uint16)
        save_lut(tmp_path, g_values, phi, inverse_gray)
        # Remove npz to force CSV fallback.
        (tmp_path / "lut.npz").unlink()
        data = load_lut(tmp_path)
        np.testing.assert_allclose(data.g_values, g_values)
        np.testing.assert_allclose(data.phi, phi)
        np.testing.assert_array_equal(data.inverse_gray, inverse_gray)

    def test_missing_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_lut(tmp_path)


# ---------------------------------------------------------------------------
# Serial-bound archive + trust gate
# ---------------------------------------------------------------------------

#: A real device serial shape; the archive is keyed by it, never by "default".
TEST_SERIAL = "22030108"
#: Points of the checkerboard scan this archive format was designed around.
N_GRAY_POINTS = 65
#: Driver contract: the inverse table indexes the panel's full 10-bit gray range.
INVERSE_LENGTH = 1024


def _scan_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A gray/phase scan plus the inverse table a calibration would archive."""
    g_values = np.linspace(0.0, 1023.0, N_GRAY_POINTS)
    phi = np.linspace(0.0, 2 * np.pi, N_GRAY_POINTS)
    inverse_gray = np.clip(np.arange(INVERSE_LENGTH) , 0, 1023).astype(np.uint16)
    return g_values, phi, inverse_gray


def _archive(
    root: Path,
    *,
    serial: str = TEST_SERIAL,
    status: str = LUT_STATUS_PROVISIONAL,
    valley: float = 0.47,
) -> Path:
    """Write one calibration into ``root`` and return its serial directory."""
    g_values, phi, inverse_gray = _scan_arrays()
    return save_serial_lut(
        slm_serial=serial,
        g_values=g_values,
        phi=phi,
        inverse_gray=inverse_gray,
        source="python scripts/slm/calib_slm_lut_checkerboard.py --out x",
        status=status,
        valley=valley,
        wavelength_nm=1064.0,
        gray_for_2pi=993,
        quality={"retained_at_first": 0.93, "retained_at_last": 0.81},
        notes="棋盘格双通道分析判定反演不可信，未激活",
        root_dir=root,
    )


class TestNormalizeSlmSerial:
    """An archive keyed by a broken serial is worse than no archive."""

    @pytest.mark.parametrize("serial", [TEST_SERIAL, "M01219666", "SN-1.2_3"])
    def test_device_like_serials_pass(self, serial: str) -> None:
        assert normalize_slm_serial(serial) == serial

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert normalize_slm_serial(f" {TEST_SERIAL} ") == TEST_SERIAL

    @pytest.mark.parametrize("serial", ["", "   ", "\t\n", None])
    def test_empty_identity_is_refused(self, serial: str | None) -> None:
        """``getattr(slm, "serial", "")`` on an unopened device must not write."""
        with pytest.raises(LutSerialError, match="列号"):
            normalize_slm_serial(serial)

    @pytest.mark.parametrize(
        "serial",
        ["a/b", "..\\x", "a\\b", "../escape", "..", ".", "a b", "a;b", "a|b", "x" * 65],
    )
    def test_path_unsafe_serials_are_refused(self, serial: str) -> None:
        with pytest.raises(LutSerialError, match="目录名"):
            normalize_slm_serial(serial)

    def test_dir_for_serial_layout(self, tmp_path: Path) -> None:
        got = lut_dir_for_serial(TEST_SERIAL, root_dir=tmp_path)
        assert got == tmp_path / "data" / "calib" / "lut" / TEST_SERIAL

    def test_save_refuses_and_creates_nothing(self, tmp_path: Path) -> None:
        """No ``data/calib/lut//`` fallback directory for an empty serial."""
        g_values, phi, inverse_gray = _scan_arrays()
        with pytest.raises(LutSerialError):
            save_serial_lut(
                slm_serial="",
                g_values=g_values,
                phi=phi,
                inverse_gray=inverse_gray,
                source="x",
                valley=0.1,
                wavelength_nm=1064.0,
                root_dir=tmp_path,
            )
        assert not (tmp_path / "data" / "calib" / "lut" / "").exists()
        assert not (tmp_path / "data").exists()


class TestLutArchiveWrite:
    """The on-disk shape the production loader is allowed to consume."""

    def test_layout_is_npz_plus_sidecar(self, tmp_path: Path) -> None:
        lut_dir = _archive(tmp_path)
        names = {p.name for p in lut_dir.iterdir()}
        assert {"lut.npz", LUT_META_FILENAME} <= names
        assert "lut_inverse.csv" in names and "lut_forward.csv" in names

    def test_npz_keeps_the_loader_field_names(self, tmp_path: Path) -> None:
        lut_dir = _archive(tmp_path)
        mark_lut_validated(lut_dir, verified_by="test", notes="cross-check")
        with np.load(lut_dir / "lut.npz", allow_pickle=False) as data:
            assert {"g_values", "phi", "inverse_gray"} <= set(data.files)
            table = data["inverse_gray"]
        assert table.dtype == np.uint16
        assert table.size == INVERSE_LENGTH
        assert int(table.max()) <= 1023
        data = load_lut(lut_dir)
        assert data.phi.size == N_GRAY_POINTS

    def test_default_status_is_provisional(self, tmp_path: Path) -> None:
        meta = read_lut_meta(_archive(tmp_path))
        assert meta["status"] == LUT_STATUS_PROVISIONAL

    def test_sidecar_records_provenance_and_quality(self, tmp_path: Path) -> None:
        meta = read_lut_meta(_archive(tmp_path, valley=0.4712))
        assert meta["serial"] == TEST_SERIAL
        assert meta["schema"] == 1
        assert meta["wavelength_nm"] == 1064.0
        assert meta["gray_for_2pi"] == 993
        assert meta["n_points"] == N_GRAY_POINTS
        assert meta["inverse_length"] == INVERSE_LENGTH
        assert meta["created_utc"].endswith("Z")
        assert "calib_slm_lut_checkerboard" in meta["source"]
        assert meta["quality"]["valley"] == pytest.approx(0.4712)
        assert meta["quality"]["retained_at_first"] == pytest.approx(0.93)
        assert "不可信" in meta["notes"]

    def test_no_temporary_files_survive_a_write(self, tmp_path: Path) -> None:
        lut_dir = _archive(tmp_path)
        assert [p for p in lut_dir.iterdir() if p.suffix == ".tmp"] == []

    def test_overwrite_is_atomic_and_leaves_no_junk(self, tmp_path: Path) -> None:
        lut_dir = _archive(tmp_path)
        before = (lut_dir / LUT_META_FILENAME).read_bytes()
        again = _archive(tmp_path, valley=0.20)
        assert again == lut_dir
        after = (lut_dir / LUT_META_FILENAME).read_bytes()
        assert before != after
        assert read_lut_meta(lut_dir)["quality"]["valley"] == pytest.approx(0.20)
        assert [p for p in lut_dir.iterdir() if p.suffix == ".tmp"] == []

    def test_failed_sidecar_write_keeps_the_previous_one(self, tmp_path: Path) -> None:
        """A crash mid-write must not destroy an archived LUT."""
        lut_dir = _archive(tmp_path)
        mark_lut_validated(lut_dir, verified_by="test", notes="cross-check")
        keep = read_lut_meta(lut_dir)
        broken = dict(keep)
        broken["extra"] = {"unserialisable": {1, 2}}  # a set is not JSON
        with pytest.raises(TypeError):
            write_lut_meta(lut_dir, broken)
        assert read_lut_meta(lut_dir) == keep
        assert [p for p in lut_dir.iterdir() if p.suffix == ".tmp"] == []

    def test_unknown_status_never_reaches_disk(self, tmp_path: Path) -> None:
        g_values, phi, inverse_gray = _scan_arrays()
        with pytest.raises(ValueError, match="status"):
            build_lut_meta(
                serial=TEST_SERIAL,
                source="x",
                g_values=g_values,
                phi=phi,
                inverse_gray=inverse_gray,
                valley=0.1,
                wavelength_nm=1064.0,
                status="trusted",
            )

    def test_wavelength_or_two_pi_gray_is_mandatory(self) -> None:
        g_values, phi, inverse_gray = _scan_arrays()
        with pytest.raises(ValueError, match="2π|波长"):
            build_lut_meta(
                serial=TEST_SERIAL,
                source="x",
                g_values=g_values,
                phi=phi,
                inverse_gray=inverse_gray,
                valley=0.1,
            )

    def test_valley_is_mandatory_and_numeric(self) -> None:
        g_values, phi, inverse_gray = _scan_arrays()
        with pytest.raises(ValueError, match="valley"):
            build_lut_meta(
                serial=TEST_SERIAL,
                source="x",
                g_values=g_values,
                phi=phi,
                inverse_gray=inverse_gray,
                valley=float("nan"),
                wavelength_nm=1064.0,
            )

    def test_mismatched_serial_and_directory_is_refused(self, tmp_path: Path) -> None:
        g_values, phi, inverse_gray = _scan_arrays()
        meta = build_lut_meta(
            serial=TEST_SERIAL,
            source="x",
            g_values=g_values,
            phi=phi,
            inverse_gray=inverse_gray,
            valley=0.1,
            wavelength_nm=1064.0,
        )
        with pytest.raises(ValueError, match="不匹配"):
            write_lut_calibration(tmp_path / "OTHER1", g_values, phi, inverse_gray, meta)
        assert not (tmp_path / "OTHER1").exists()


class TestLutTrustGate:
    """``require_validated_lut`` is what the production display path calls."""

    def test_provisional_is_refused_with_a_reason(self, tmp_path: Path) -> None:
        lut_dir = _archive(tmp_path)
        with pytest.raises(LutTrustError, match="provisional") as exc:
            require_validated_lut(lut_dir)
        assert "不可信" in str(exc.value)

    def test_validated_is_accepted(self, tmp_path: Path) -> None:
        lut_dir = _archive(tmp_path)
        mark_lut_validated(lut_dir, verified_by="order-balance re-analysis", notes="谷底<0.05")
        meta = require_validated_lut(lut_dir)
        assert meta["status"] == LUT_STATUS_VALIDATED
        assert meta["verified_by"] == "order-balance re-analysis"

    def test_missing_sidecar_is_refused_not_tolerated(self, tmp_path: Path) -> None:
        lut_dir = _archive(tmp_path)
        (lut_dir / LUT_META_FILENAME).unlink()
        with pytest.raises(LutTrustError, match=LUT_META_FILENAME):
            require_validated_lut(lut_dir)

    def test_missing_directory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(LutTrustError, match="不存在"):
            require_validated_lut(tmp_path / "nope")

    def test_unparseable_sidecar_is_refused(self, tmp_path: Path) -> None:
        lut_dir = _archive(tmp_path)
        (lut_dir / LUT_META_FILENAME).write_text("{not json", encoding="utf-8")
        with pytest.raises(LutTrustError, match="无法解析"):
            require_validated_lut(lut_dir)

    @pytest.mark.parametrize("drop", ["status", "serial", "quality", "source", "schema"])
    def test_incomplete_sidecar_is_refused(self, tmp_path: Path, drop: str) -> None:
        lut_dir = _archive(tmp_path)
        meta = json.loads((lut_dir / LUT_META_FILENAME).read_text(encoding="utf-8"))
        meta.pop(drop)
        (lut_dir / LUT_META_FILENAME).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        with pytest.raises(LutTrustError, match="必填字段|缺"):
            require_validated_lut(lut_dir)

    def test_bogus_status_string_is_refused(self, tmp_path: Path) -> None:
        lut_dir = _archive(tmp_path)
        meta = json.loads((lut_dir / LUT_META_FILENAME).read_text(encoding="utf-8"))
        meta["status"] = "validated "  # trailing space is not the literal token
        (lut_dir / LUT_META_FILENAME).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        with pytest.raises(LutTrustError, match="status"):
            require_validated_lut(lut_dir)

    def test_copied_sidecar_is_refused(self, tmp_path: Path) -> None:
        """Sidecar and directory must agree, or the LUT belongs to another panel."""
        src = _archive(tmp_path)
        meta = json.loads((src / LUT_META_FILENAME).read_text(encoding="utf-8"))
        mark_lut_validated(src, verified_by="test", notes="ok")
        other = _archive(tmp_path, serial="OTHER1")
        (other / LUT_META_FILENAME).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        with pytest.raises(LutTrustError, match="不匹配"):
            require_validated_lut(other)

    def test_promotion_requires_attribution(self, tmp_path: Path) -> None:
        lut_dir = _archive(tmp_path)
        with pytest.raises(ValueError, match="verified_by"):
            mark_lut_validated(lut_dir, verified_by=" ", notes="ok")
        with pytest.raises(ValueError, match="notes"):
            mark_lut_validated(lut_dir, verified_by="me", notes="")
        assert read_lut_meta(lut_dir)["status"] == LUT_STATUS_PROVISIONAL

    def test_promotion_keeps_the_original_provenance(self, tmp_path: Path) -> None:
        lut_dir = _archive(tmp_path, valley=0.47)
        before = read_lut_meta(lut_dir)
        after = mark_lut_validated(lut_dir, verified_by="v", notes="谷底复查通过")
        assert after["created_utc"] == before["created_utc"]
        assert after["quality"] == before["quality"]
        assert after["source"] == before["source"]
        assert before["notes"] in after["notes"]
        assert require_validated_lut(lut_dir)["status"] == LUT_STATUS_VALIDATED
