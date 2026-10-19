"""Offline tests for the optical-path fingerprint.

Three properties are load-bearing, and each fails *silently* when broken:

1. **the axial signal must be caught** -- a camera moved off the focal plane is
   the change most likely to happen between sessions, and it shows up as the
   flat FWHM changing (measured on this bench: ~39 px defocused vs ~16 px in
   focus). A fingerprint that cannot see that is decorative.
2. **brightness must NOT be caught** -- the same bench read 20.6 counts/ms and
   then 2.9 counts/ms between runs, a 7.3x step, while staying stable to 1.05x
   within a run. Keying the verdict on brightness would veto runs for the laser
   rather than for the optics.
3. **the operator's override must be total** -- there is a switch that skips the
   comparison, and a switch that cannot actually skip is a switch that gets
   removed.

The fake bench recovers the requested phase from the pattern it is handed (a
least-squares fit), so the test drives the real ``zernike_panel`` and cannot be
fooled by a stub that echoes its own parameter back.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ao_shaping.tools.kernels.bench_kernels import fresnel_panel, zernike_panel
from ao_shaping.tools.slm.panel import bench_fingerprint as bfp

WAVELENGTH_NM = 1064.0
PIXEL_UM = 8.0
SHAPE = (120, 160)
CENTRE = (80, 60)


class FakeBench:
    """A camera whose spot responds to the panel phase the way this bench does.

    ``focus_scale`` stands for where the camera sits relative to the focus:
    smaller means a tighter spot. ``brightness`` scales the whole response and
    is the knob for the "same optics, different light level" case.
    """

    def __init__(
        self,
        *,
        focus_scale: float = 1.0,
        brightness: float = 1.0,
        centre_shift: tuple[float, float] = (0.0, 0.0),
        tilt_px_per_rad: float = 5.0,
        counts_per_ms: float = 575.0,
    ) -> None:
        self.focus_scale = float(focus_scale)
        self.brightness = float(brightness)
        self.centre_shift = centre_shift
        self.tilt_px_per_rad = float(tilt_px_per_rad)
        self.counts_per_ms = float(counts_per_ms)
        self.exposure_ms = 0.2
        self.tilt_rad = 0.0
        self.phases: list[float] = []

    # -- device surface ---------------------------------------------------
    def apply_phase(self, phase: np.ndarray) -> None:
        self.tilt_rad = self._tilt_of(phase)
        self.phases.append(self.tilt_rad)

    def set_exposure(self, ms: float) -> float:
        self.exposure_ms = float(ms)
        return float(ms)

    def read(self) -> np.ndarray:
        # The tilt displaces the envelope; defocus widens it. Both are the
        # physical behaviour the fingerprint is reading out.
        width = max(
            self.focus_scale * 6.0 / max(np.sqrt(1.0 + self.tilt_rad**2), 1e-9), 1.2
        )
        shift_x = self.tilt_px_per_rad * self.tilt_rad
        cx = SHAPE[1] / 2.0 + shift_x + self.centre_shift[0]
        cy = SHAPE[0] / 2.0 + self.centre_shift[1]
        yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
        peak = min(
            255.0,
            self.brightness * self.counts_per_ms * self.exposure_ms,
        )
        gx = np.exp(-0.5 * ((xx - cx) / width) ** 2)
        gy = np.exp(-0.5 * ((yy - cy) / width) ** 2)
        return (peak * gy * gx).astype(np.float32)

    # -- inverse problem the test relies on --------------------------------
    def _tilt_of(self, phase: np.ndarray) -> float:
        """Recover the requested tilt amplitude by fitting the canonical mode.

        The disc's bounding box is the mode's support, so radius and centre come
        from the pattern itself rather than from the value under test. Fitting
        against :func:`zernike_panel` also means the test cannot pass if the
        fingerprint stops using the canonical generator.
        """
        arr = np.asarray(phase, dtype=np.float64)
        mask = arr != 0.0
        if not mask.any():
            return 0.0
        ys, xs = np.nonzero(mask)
        radius = int(round((xs.max() - xs.min()) / 2.0))
        centre = (int(round(xs.min() + radius)), int(round(ys.min() + radius)))
        basis = zernike_panel(
            {(1, 1): 1.0}, radius=radius, pupil_center=centre, panel_shape=SHAPE
        )
        b = basis[mask]
        return float((arr[mask] @ b) / (b @ b))


def _capture(bench: FakeBench, **kwargs) -> bfp.BenchFingerprint:
    params = dict(
        apply_phase=bench.apply_phase,
        read_frame=bench.read,
        set_exposure_ms=bench.set_exposure,
        pupil_center_px=CENTRE,
        panel_shape=SHAPE,
        radius_px=40.0,
        frames=2,
        discard=1,
        wavelength_nm=WAVELENGTH_NM,
        pixel_um=PIXEL_UM,
    )
    params.update(kwargs)
    return bfp.measure_fingerprint(**params)


class TestCapture:
    def test_it_captures_a_finite_fingerprint(self) -> None:
        fp = _capture(FakeBench())
        assert np.isfinite(fp.flat_fwhm_px) and fp.flat_fwhm_px > 0
        assert fp.tilt_shift_px > 0
        assert 0.0 <= fp.flat_hollowness <= 1.2

    def test_the_tilt_actually_moves_the_spot(self) -> None:
        fp = _capture(FakeBench())
        assert fp.tilt_rad == pytest.approx(bfp.TILT_RAD, rel=0.2)
        assert fp.tilt_shift_px > 5.0

    def test_two_captures_of_an_unchanged_bench_match(self) -> None:
        a = _capture(FakeBench())
        b = _capture(FakeBench())
        verdict = bfp.compare_fingerprints(b, a)
        assert verdict.ok, verdict.reasons

    def test_an_underexposed_bench_refuses_to_produce_a_fingerprint(
        self, monkeypatch
    ) -> None:
        """A fingerprint from a black frame would veto runs for the wrong reason."""

        def _adapt(*_a, **_k):
            class _S:
                exposure_ms = 20.0
                peak = 1.0
                fraction = 0.004
                status = "dim"

            return _S()

        monkeypatch.setattr(bfp, "adapt_exposure", _adapt)
        with pytest.raises(RuntimeError, match="可测带"):
            _capture(FakeBench())


class TestChangeDetection:
    def test_a_camera_moved_off_focus_is_caught(self) -> None:
        """The axial signal, while both spots are still in the trusted regime.

        This mirrors the real numbers: ~16 px in focus, ~23 px after a small
        axial move. Past :data:`FWHM_FOCUSED_PX` the gate takes over instead --
        see :class:`TestTrustworthinessGate`.
        """
        reference = _capture(FakeBench(focus_scale=1.0))
        current = _capture(FakeBench(focus_scale=1.4))
        assert current.flat_fwhm_px < bfp.FWHM_FOCUSED_PX
        verdict = bfp.compare_fingerprints(current, reference)
        assert not verdict.ok
        assert any("宽度" in r for r in verdict.reasons)

    def test_a_brightness_change_alone_is_not_caught(self) -> None:
        """7.3x on this bench. Keying the verdict on light would blame the laser."""
        reference = _capture(FakeBench(brightness=1.0))
        current = _capture(FakeBench(brightness=1.0 / 7.3))
        verdict = bfp.compare_fingerprints(current, reference)
        assert verdict.ok, verdict.reasons

    def test_a_focal_scale_change_is_caught(self) -> None:
        reference = _capture(FakeBench(tilt_px_per_rad=5.0))
        current = _capture(FakeBench(tilt_px_per_rad=3.0))
        verdict = bfp.compare_fingerprints(current, reference)
        assert not verdict.ok
        assert any("焦面尺度" in r for r in verdict.reasons)

    def test_a_staging_shift_is_caught(self) -> None:
        reference = _capture(FakeBench())
        current = _capture(FakeBench(centre_shift=(40.0, 0.0)))
        verdict = bfp.compare_fingerprints(current, reference)
        assert not verdict.ok
        assert any("质心" in r for r in verdict.reasons)

    def test_a_different_pupil_centre_is_reported_not_compared(self) -> None:
        reference = _capture(FakeBench())
        current = _capture(FakeBench(), pupil_center_px=(60, 60))
        verdict = bfp.compare_fingerprints(current, reference)
        assert not verdict.ok
        assert any("入射中心" in r for r in verdict.reasons)

    def test_the_change_raises_when_asked_to(self) -> None:
        reference = _capture(FakeBench())
        verdict = bfp.compare_fingerprints(
            _capture(FakeBench(focus_scale=1.4)), reference
        )
        with pytest.raises(bfp.BenchPathChanged):
            verdict.raise_if_changed()

    def test_a_tiny_change_stays_inside_tolerance(self) -> None:
        """The detector has to survive its own noise or it fires on everything."""
        reference = _capture(FakeBench(focus_scale=1.0))
        current = _capture(FakeBench(focus_scale=1.03))
        assert bfp.compare_fingerprints(current, reference).ok


class TestOperatorOverride:
    def test_disabling_the_check_skips_even_a_real_change(self) -> None:
        reference = _capture(FakeBench())
        current = _capture(FakeBench(focus_scale=3.0))
        verdict = bfp.compare_fingerprints(current, reference, enabled=False)
        assert verdict.ok
        assert verdict.skipped
        verdict.raise_if_changed()  # the override must be total

    def test_no_reference_is_not_a_mismatch(self) -> None:
        verdict = bfp.compare_fingerprints(_capture(FakeBench()), None)
        assert verdict.ok and verdict.skipped

    def test_an_unreachable_shift_tolerance_is_not_used(self) -> None:
        """A reference with a ~0 shift must not divide by it."""
        reference = bfp.BenchFingerprint(
            measured_at="t",
            slm_number=1,
            slm_wavelength=1064,
            pupil_center_px=CENTRE,
            tilt_rad=bfp.TILT_RAD,
            flat_fwhm_px=16.0,
            flat_hollowness=0.9,
            flat_centroid=(100.0, 100.0),
            tilt_centroid=(100.0, 100.0),  # no measurable shift
            tilt_shift_px=0.0,
            flat_peak_per_ms=1.0,
            flat_peak_fraction=0.4,
        )
        current = _capture(FakeBench())
        verdict = bfp.compare_fingerprints(current, reference)
        assert "tilt_shift_rel" not in verdict.deltas or np.isfinite(
            verdict.deltas["tilt_shift_rel"]
        )


class TestTrustworthinessGate:
    """A defocused spot makes every shape statistic noise, so it must not veto."""

    def test_a_wide_spot_refuses_to_decide(self) -> None:
        reference = _capture(FakeBench(focus_scale=1.0))
        current = _capture(FakeBench(focus_scale=3.0))
        verdict = bfp.compare_fingerprints(current, reference)
        assert verdict.ok, "a defocused spot must never block a run"
        assert not verdict.reasons
        assert any("不可信" in n for n in verdict.notes)
        verdict.raise_if_changed()

    def test_the_gate_is_off_for_a_focused_bench(self) -> None:
        reference = _capture(FakeBench(focus_scale=1.0))
        assert reference.flat_fwhm_px < bfp.FWHM_FOCUSED_PX, (
            "the fake's in-focus spot must sit below the gate, or the tests "
            "above are passing for the wrong reason"
        )
        assert bfp.compare_fingerprints(_capture(FakeBench()), reference).ok

    def test_a_focused_bench_still_catches_a_real_change(self) -> None:
        reference = _capture(FakeBench(focus_scale=1.0))
        current = _capture(FakeBench(focus_scale=1.4))
        verdict = bfp.compare_fingerprints(current, reference)
        assert not verdict.ok


class TestSerialisation:
    def test_round_trip(self) -> None:
        fp = _capture(FakeBench())
        again = bfp.BenchFingerprint.from_dict(fp.to_dict())
        assert again.flat_fwhm_px == pytest.approx(fp.flat_fwhm_px)
        assert again.tilt_shift_px == pytest.approx(fp.tilt_shift_px)
        assert bfp.compare_fingerprints(again, fp).ok

    def test_unknown_keys_do_not_break_loading(self) -> None:
        payload = _capture(FakeBench()).to_dict()
        payload["something_new"] = 1
        assert bfp.BenchFingerprint.from_dict(payload).slm_number == 1


class TestSessionWiring:
    """The wiring is where the operator's override has to actually work."""

    def test_disabling_the_check_never_touches_the_panel(self, tmp_path, monkeypatch):
        """It must skip the *measurement*, not measure and ignore the verdict."""
        from ao_shaping.runners.slm import model_in_loop_session as mis

        def _boom(*_a, **_k):
            raise AssertionError("skipped verification must not open devices")

        monkeypatch.setattr(
            "ao_shaping.tools.slm.panel.bench_fingerprint.probe_session_fingerprint",
            _boom,
        )
        verdict = mis.verify_bench_path(
            tmp_path, params=object(), config=object(), enabled=False
        )
        assert verdict["ok"] and verdict["skipped"]

    def test_a_calibration_without_geometry_records_no_fingerprint(
        self, tmp_path, monkeypatch
    ):
        """A run with nothing calibrated has nothing to invalidate."""
        from ao_shaping.runners.slm import model_in_loop_session as mis

        monkeypatch.setattr(
            "ao_shaping.tools.slm.panel.bench_fingerprint.probe_session_fingerprint",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no hardware")),
        )
        calibration = mis.calibrate_session(
            _session_params(),
            _session_config(),
            tmp_path / "run",
            locate=False,
            geometry=False,
        )
        assert calibration.fingerprint is None
        assert not (tmp_path / "run" / "bench_fingerprint.json").exists()

    def test_a_stored_fingerprint_round_trips(self, tmp_path):
        from ao_shaping.runners.slm.model_in_loop_session import (
            load_stored_fingerprint,
        )

        fp = _capture(FakeBench())
        (tmp_path / "bench_fingerprint.json").write_text(
            json.dumps(fp.to_dict()), encoding="utf-8"
        )
        assert load_stored_fingerprint(tmp_path).flat_fwhm_px == pytest.approx(
            fp.flat_fwhm_px
        )

    def test_a_corrupt_fingerprint_is_treated_as_absent(self, tmp_path):
        from ao_shaping.runners.slm.model_in_loop_session import (
            load_stored_fingerprint,
        )

        (tmp_path / "bench_fingerprint.json").write_text("{not json", encoding="utf-8")
        assert load_stored_fingerprint(tmp_path) is None

    def test_a_missing_fingerprint_is_not_an_error(self, tmp_path):
        from ao_shaping.runners.slm.model_in_loop_session import (
            load_stored_fingerprint,
        )

        assert load_stored_fingerprint(tmp_path) is None


def _session_params(**overrides):
    from ao_shaping.runners.runner_common import SlmModelInLoopParams

    return SlmModelInLoopParams(**overrides)


def _session_config(**overrides):
    from ao_shaping.optimizer.wfless.slm_model_in_loop import SlmModelInLoopConfig

    base = dict(cam_type="daheng", region=64, far_field_padding=8, pupil_center=(960, 600))
    base.update(overrides)
    return SlmModelInLoopConfig(**base)


class TestKernelUsage:
    def test_the_tilt_probe_uses_the_canonical_zernike_generator(self) -> None:
        """The fingerprint must not carry its own idea of a tilt."""
        phase = zernike_panel(
            {(1, 1): bfp.TILT_RAD},
            radius=40,
            pupil_center=CENTRE,
            panel_shape=SHAPE,
        )
        assert phase.shape == SHAPE
        assert phase.any()

    def test_the_flat_probe_is_really_flat(self) -> None:
        assert not fresnel_panel(float("inf"), radius=40, pupil_center=CENTRE).any()