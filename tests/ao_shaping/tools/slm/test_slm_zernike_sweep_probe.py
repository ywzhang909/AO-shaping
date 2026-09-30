"""Offline tests for the smooth-Zernike sweep probe.

No hardware: the camera and SLM are mocks, which is the point of the probe
taking devices by injection.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.tools.slm.slm_zernike_sweep_probe import (
    AUX_MODES,
    MODE_CODES,
    SINGLE_LOBE_MIN_HOLLOWNESS,
    SWEEP_MODE_NM,
    TILT_AXES_NM,
    SweepPoint,
    SweepResult,
    acquire_sweep,
    default_sweep_points,
    point_panel_phase,
    save_sweep_npz,
    sweep_to_recorder_rows,
    zernike_panel,
)

PANEL = (120, 160)
CENTRE = (80, 60)
RADIUS = 40


class MockSLM:
    """Records what it was asked to display."""

    def __init__(self) -> None:
        self.shown: list[np.ndarray] = []
        self.waits: list[float | None] = []

    def create_phase_from_array(self, phase: np.ndarray) -> np.ndarray:
        return np.asarray(phase, dtype=np.float32)

    def display_data(self, gray: np.ndarray, wait_time_s: float | None = None) -> None:
        self.shown.append(np.asarray(gray))
        self.waits.append(wait_time_s)


class MockCam:
    """Returns a Gaussian spot that moves with the displayed phase gradient."""

    def __init__(self, shift_per_grad: float = 2.0) -> None:
        self.shift_per_grad = shift_per_grad
        self._last_phase: np.ndarray | None = None
        self.reads = 0

    def _render(self) -> np.ndarray:
        phase = self._last_phase if self._last_phase is not None else np.zeros(PANEL)
        # A crude but monotone stand-in for the real response: the centroid
        # follows the phase gradient, so the test can assert on the *sign* and
        # *ordering* of a shift rather than on optical fidelity.
        grad = float(np.gradient(phase.astype(np.float64))[0].mean())
        img = np.zeros((80, 120), dtype=np.float64)
        cy = 40.0 + np.clip(grad * self.shift_per_grad, -15.0, 15.0)
        yy, xx = np.mgrid[0:80, 0:120]
        img = 100.0 * np.exp(-(((yy - cy) ** 2 + (xx - 60.0) ** 2) / (2 * 6.0**2)))
        return img

    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        self.reads += 1
        return self._render()


class WiringCam(MockCam):
    """A camera that reports the phase it was last asked to display."""

    def display_data(self, gray: np.ndarray, wait_time_s: float | None = None) -> None:
        self._last_phase = np.asarray(gray, dtype=np.float64)


class TestZernikePanelIsTheKernelOne:
    """The panel builder is the kernel's; only the probe-specific bits are new.

    ``zernike_panel`` itself is owned by ``slm_bench_probe`` and is tested
    there. What matters here is that the probe re-exports it rather than
    reimplementing it (a second copy silently drifted once already), and that it
    honours a non-default ``panel_shape``.
    """

    def test_reexported_from_the_kernel(self) -> None:
        from ao_shaping.tools.slm.slm_bench_probe import (
            zernike_panel as kernel_builder,
        )

        assert zernike_panel is kernel_builder

    def test_non_default_panel_shape_is_honoured(self) -> None:
        assert zernike_panel({(2, 0): 1.0}, RADIUS, CENTRE, PANEL).shape == PANEL


class TestPointPanelPhase:
    def test_flat_is_all_zero(self) -> None:
        assert not np.any(point_panel_phase(SweepPoint("flat"), CENTRE, RADIUS, PANEL))

    def test_ramp_uses_the_requested_axis_and_period(self) -> None:
        x_phase = point_panel_phase(SweepPoint("ramp", 60.0, "x"), CENTRE, RADIUS, PANEL)
        y_phase = point_panel_phase(SweepPoint("ramp", 60.0, "y"), CENTRE, RADIUS, PANEL)
        assert not np.allclose(x_phase, y_phase)
        # A 2*pi ramp over 60 px repeats every 60 columns.
        assert np.allclose(x_phase[:, 0], x_phase[:, 60], atol=1e-9)

    def test_ramp_without_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="ramp point needs axis"):
            point_panel_phase(SweepPoint("ramp", 60.0), CENTRE, RADIUS, PANEL)

    def test_tilt_needs_an_axis(self) -> None:
        with pytest.raises(ValueError, match="tilt point needs axis"):
            point_panel_phase(SweepPoint("tilt", 1.0), CENTRE, RADIUS, PANEL)

    def test_unknown_mode_raises_with_the_valid_set(self) -> None:
        with pytest.raises(ValueError, match="unknown sweep mode"):
            point_panel_phase(SweepPoint("trefoil"), CENTRE, RADIUS, PANEL)

    def test_every_declared_mode_builds(self) -> None:
        for mode in SWEEP_MODE_NM:
            phase = point_panel_phase(SweepPoint(mode, 0.5), CENTRE, RADIUS, PANEL)
            assert phase.shape == PANEL
            assert np.any(np.abs(phase) > 0.0), mode

    def test_tilt_axes_are_distinct_modes(self) -> None:
        assert TILT_AXES_NM["x"] != TILT_AXES_NM["y"]


class TestAcquireSweep:
    def test_flat_is_captured_first_and_exactly_once(self) -> None:
        """The panel retains the previous pattern, so flat must precede writes."""
        slm, cam = MockSLM(), WiringCam()
        points = [SweepPoint("defocus", 1.0), SweepPoint("defocus", -1.0)]
        result = acquire_sweep(
            cam, slm, points, pupil_center=CENTRE, zernike_radius=RADIUS,
            panel_shape=PANEL, n_frames=1, n_discard=0, max_wait_s=0.2,
        )
        assert result.tags() == ["flat", "defocus+1.00", "defocus-1.00"]
        assert not np.any(slm.shown[0]), "the first display must be the flat phase"

    def test_every_point_is_displayed_and_measured(self) -> None:
        slm, cam = MockSLM(), WiringCam()
        points = default_sweep_points(tilt=(1.0,), defocus=(1.0,), astig=(1.0,),
                                      coma=(1.0,), spherical=(1.0,), ramps=(120.0,))
        result = acquire_sweep(
            cam, slm, points, pupil_center=CENTRE, zernike_radius=RADIUS,
            panel_shape=PANEL, n_frames=1, n_discard=0, max_wait_s=0.2,
        )
        assert len(result.points) == len(points) + 1
        assert len(slm.shown) == len(points) + 1
        for measured in result.points:
            assert measured.fwhm_px > 0.0
            assert np.isfinite(measured.centroid_x)
            assert measured.phase_rad.shape == PANEL
            assert measured.frame.ndim == 2

    def test_settle_arguments_reach_the_measurement_kernel(self) -> None:
        slm, cam = MockSLM(), WiringCam()
        acquire_sweep(
            cam, slm, [SweepPoint("defocus", 1.0)], pupil_center=CENTRE,
            zernike_radius=RADIUS, panel_shape=PANEL, n_frames=1, n_discard=0,
            wait_time_s=0.25, stable_tol=0.01, max_wait_s=0.2,
        )
        assert all(w == 0.25 for w in slm.waits)

    def test_on_point_callback_fires_once_per_point(self) -> None:
        slm, cam = MockSLM(), WiringCam()
        seen: list[str] = []
        acquire_sweep(
            cam, slm, [SweepPoint("defocus", 1.0)], pupil_center=CENTRE,
            zernike_radius=RADIUS, panel_shape=PANEL, n_frames=1, n_discard=0,
            max_wait_s=0.2, on_point=lambda m: seen.append(m.point.tag()),
        )
        assert seen == ["flat", "defocus+1.00"]

    def test_provenance_is_recorded_on_the_result(self) -> None:
        slm, cam = MockSLM(), WiringCam()
        result = acquire_sweep(
            cam, slm, [], pupil_center=(7, 9), zernike_radius=33,
            panel_shape=PANEL, n_frames=1, n_discard=0, max_wait_s=0.2,
        )
        assert result.zernike_radius == 33
        assert result.pupil_center == (7, 9)
        assert result.panel_shape == PANEL


class TestRecorderRows:
    def _sweep(self) -> SweepResult:
        slm, cam = MockSLM(), WiringCam()
        return acquire_sweep(
            cam, slm, [SweepPoint("defocus", 1.0), SweepPoint("ramp", 120.0, "x")],
            pupil_center=CENTRE, zernike_radius=RADIUS, panel_shape=PANEL,
            n_frames=1, n_discard=0, max_wait_s=0.2,
        )

    def test_rows_carry_the_raw_process_arrays(self) -> None:
        rows = sweep_to_recorder_rows(self._sweep())
        assert len(rows) == 3
        for index, row in enumerate(rows):
            assert row["_epoch"] == index
            assert row["_phase"].shape == PANEL
            assert row["_img"].ndim == 2

    def test_strings_are_encoded_as_codes(self) -> None:
        """The standard serialiser coerces scalars to float and would eat strings."""
        rows = sweep_to_recorder_rows(self._sweep())
        for row in rows:
            assert isinstance(row["mode_id"], int)
            assert isinstance(row["axis_id"], int)
            assert not any(isinstance(v, str) for v in row.values())

    def test_every_mode_and_axis_used_is_in_the_code_table(self) -> None:
        result = SweepResult()
        for mode in (*SWEEP_MODE_NM, *AUX_MODES, "tilt"):
            for axis in ("", "x", "y"):
                result.points.append(type(result.points).__class__ and _fake(mode, axis))
        for measured in result.points:
            assert measured.point.mode in MODE_CODES
            assert measured.point.axis in ("", "x", "y")

    def test_peak_is_the_recorder_mark(self) -> None:
        rows = sweep_to_recorder_rows(self._sweep())
        assert all("peak" in row for row in rows)


def _fake(mode: str, axis: str):
    from ao_shaping.tools.slm.slm_zernike_sweep_probe import SweepPointResult

    return SweepPointResult(
        point=SweepPoint(mode, 1.0, axis), peak=1.0, fwhm_px=1.0, centroid_x=0.0,
        centroid_y=0.0, hollowness=1.0, phase_rad=np.zeros(PANEL),
        frame=np.zeros((4, 4)),
    )


class TestPersistence:
    def test_columns_round_trip_through_npz(self) -> None:
        slm, cam = MockSLM(), WiringCam()
        result = acquire_sweep(
            cam, slm, [SweepPoint("defocus", 1.5)], pupil_center=CENTRE,
            zernike_radius=RADIUS, panel_shape=PANEL, n_frames=1, n_discard=0,
            max_wait_s=0.2,
        )
        columns = result.as_columns()
        lengths = {len(v) for k, v in columns.items() if k not in
                   ("zernike_radius", "pupil_center")}
        assert len(lengths) == 1
        assert columns["zernike_radius"].shape == ()
        assert columns["pupil_center"].shape == (2,)

    def test_save_creates_parents(self, tmp_path) -> None:
        slm, cam = MockSLM(), WiringCam()
        result = acquire_sweep(
            cam, slm, [], pupil_center=CENTRE, zernike_radius=RADIUS,
            panel_shape=PANEL, n_frames=1, n_discard=0, max_wait_s=0.2,
        )
        target = tmp_path / "nested" / "dir" / "sweep_records.npz"
        written = save_sweep_npz(target, result)
        assert written.exists()
        with np.load(written, allow_pickle=True) as data:
            assert data["mode"].tolist() == ["flat"]


class TestSingleLobeGate:
    def test_hollowness_threshold_matches_the_fitters_constant(self) -> None:
        from ao_shaping.optimizer.wfless.model_in_loop_shaping import (
            _DEFOCUS_MIN_HOLLOWNESS,
        )

        assert SINGLE_LOBE_MIN_HOLLOWNESS == _DEFOCUS_MIN_HOLLOWNESS

    def test_a_ring_is_flagged_and_a_lobe_is_not(self) -> None:
        from ao_shaping.tools.slm.slm_zernike_sweep_probe import SweepPointResult

        def measured(hollow: float) -> SweepPointResult:
            return _fake("spherical", "")
        ring = measured(0.43)
        ring.hollowness = 0.43
        assert not ring.is_single_lobe()
        lobe = measured(0.9)
        assert lobe.is_single_lobe()


class TestDefaultSweepPoints:
    def test_covers_every_declared_mode(self) -> None:
        points = default_sweep_points()
        modes = {p.mode for p in points}
        assert set(SWEEP_MODE_NM) <= modes
        assert {"tilt", "ramp"} <= modes

    def test_both_axes_are_swept_for_tilt_and_ramp(self) -> None:
        points = default_sweep_points()
        for mode in ("tilt", "ramp"):
            assert {p.axis for p in points if p.mode == mode} == {"x", "y"}

    def test_coefficients_straddle_zero_so_an_offset_is_observable(self) -> None:
        for mode in (*SWEEP_MODE_NM, "tilt"):
            values = [p.coefficient for p in default_sweep_points() if p.mode == mode]
            assert any(v > 0 for v in values), mode
            assert any(v < 0 for v in values), mode

    def test_default_point_count(self) -> None:
        # 10 ramp + 4 tilt + 8 defocus + 8 astig + 8 coma + 4 spherical
        assert len(default_sweep_points()) == 42
