"""Offline tests for the Fresnel focus probe, against a synthetic bench.

The probe's whole reason to exist is one trap, and every test here is aimed at
it: **brightness changes by orders of magnitude across the sweep**, so a raw peak
cannot rank the points. Near focus the frame clips, a few dioptres away it sits in
the read noise, and the naive ``argmax(peak)`` then reports whichever point the
fixed exposure happened to saturate.

So the contract under test is:

1. every point is measured at **its own** exposure (never carried over blindly);
2. ranking uses ``peak / exposure_ms``, the only exposure-correctable quantity on
   a linear sensor;
3. the returned point is where the *intensity* is highest, not where the raw peak
   is highest -- and on this synthetic bench those two answers differ;
4. a point that cannot be brought into the measurable band (clipped at the floor,
   or dark at the ceiling) is excluded instead of being ranked;
5. the sweep extends outward when the winner lands on the edge of the range,
   because that means the answer is outside it.

The fake bench recovers the requested defocus from the phase it is handed (a
least-squares fit of the quadratic coefficient), so the test drives the real
``fresnel_panel`` and cannot be fooled by a stub that echoes a parameter back.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.tools.kernels.bench_kernels import fresnel_panel
from ao_shaping.tools.slm.panel import focus_probe as fp

WAVELENGTH_NM = 1064.0
PIXEL_UM = 8.0
FULL_SCALE = 255.0


class FakeBench:
    """A camera + panel whose brightness is a Lorentzian in dioptric power.

    ``intensity(D)`` peaks at ``true_diopter`` and falls off with ``width``;
    the frame peak is ``min(full_scale, intensity * counts_per_ms * exposure)``,
    i.e. linear in exposure until it clips -- exactly the situation the probe is
    built to survive.
    """

    def __init__(
        self,
        *,
        true_diopter: float = -1.7,
        width: float = 1.2,
        counts_per_ms: float = 2550.0,
        floor_intensity: float = 0.004,
        shape: tuple[int, int] = (120, 160),
        sigma: float = 4.0,
        extra_peak_at: float | None = None,
        extra_peak_intensity: float = 0.0,
    ) -> None:
        self.true_diopter = float(true_diopter)
        self.width = float(width)
        self.counts_per_ms = float(counts_per_ms)
        self.floor_intensity = float(floor_intensity)
        self.shape = shape
        self.sigma = float(sigma)
        self.extra_peak_at = extra_peak_at
        self.extra_peak_intensity = float(extra_peak_intensity)
        self.diopter = 0.0
        self.exposure_ms = 0.1
        self.phases: list[float] = []
        self.exposures: list[float] = []
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        self._yy, self._xx = yy, xx

    # -- device surface ---------------------------------------------------
    def apply_phase(self, phase: np.ndarray) -> None:
        self.diopter = self._diopter_of(phase)
        self.phases.append(self.diopter)

    def set_exposure(self, ms: float) -> float:
        self.exposure_ms = float(ms)
        self.exposures.append(float(ms))
        return float(ms)

    def read(self) -> np.ndarray:
        intensity = self.intensity_at(self.diopter)
        peak = min(FULL_SCALE, intensity * self.counts_per_ms * self.exposure_ms)
        cy, cx = self.shape[0] // 2, self.shape[1] // 2
        g = np.exp(
            -((self._xx - cx) ** 2 + (self._yy - cy) ** 2) / (2.0 * self.sigma**2)
        )
        return (g * peak).astype(np.float32)

    def intensity_at(self, diopter: float) -> float:
        v = 1.0 / (1.0 + ((diopter - self.true_diopter) / self.width) ** 2)
        if self.extra_peak_at is not None:
            v += self.extra_peak_intensity / (
                1.0 + ((diopter - self.extra_peak_at) / 0.05) ** 2
            )
        return v + self.floor_intensity

    # -- the inverse problem the test relies on ---------------------------
    def _diopter_of(self, phase: np.ndarray) -> float:
        """Invert ``fresnel_panel``: D = c * lambda / (pi * pixel**2)."""
        arr = np.asarray(phase, dtype=np.float64)
        mask = arr != 0.0
        if not mask.any():
            return 0.0
        ys, xs = np.nonzero(mask)
        cy, cx = float(ys.mean()), float(xs.mean())
        r2 = ((self._xx - cx) ** 2 + (self._yy - cy) ** 2)[mask]
        c = float((arr[mask] * r2).sum() / (r2**2).sum())
        return c * WAVELENGTH_NM * 1e-9 / (np.pi * (PIXEL_UM * 1e-6) ** 2)


def _run(bench: FakeBench, **kwargs) -> fp.FocusProbeResult:
    params = dict(
        apply_phase=bench.apply_phase,
        read_frame=bench.read,
        set_exposure_ms=bench.set_exposure,
        panel_shape=bench.shape,
        radius_px=40.0,
        centre_px=(bench.shape[1] / 2.0, bench.shape[0] / 2.0),
        coarse_points=25,
        diopter_range=6.0,
        refine_points=7,
        start_exposure_ms=0.1,
        frames=2,
        discard=1,
        wavelength_nm=WAVELENGTH_NM,
        pixel_um=PIXEL_UM,
    )
    params.update(kwargs)
    return fp.sweep_focus(**params)


class TestFresnelKernel:
    def test_plain_phase_for_infinite_focal_length(self) -> None:
        phase = fresnel_panel(float("inf"), radius=30, pupil_center=(80, 60))
        assert phase.shape == (fp.SLM_PANEL_H, fp.SLM_PANEL_W)
        assert not phase.any()

    def test_zero_focal_length_is_rejected(self) -> None:
        """Returning zeros here would look like a successful correction."""
        with pytest.raises(ValueError, match="not a lens"):
            fresnel_panel(0.0, radius=30, pupil_center=(80, 60))

    def test_phase_is_confined_to_the_aperture(self) -> None:
        phase = fresnel_panel(-1000.0, radius=20, pupil_center=(40, 30))
        assert phase[30, 50] != 0.0
        assert phase[0, 0] == 0.0
        assert (phase != 0.0).sum() <= np.pi * 21**2

    def test_sign_flips_with_the_focal_length(self) -> None:
        a = fresnel_panel(-800.0, radius=20, pupil_center=(40, 30))
        b = fresnel_panel(800.0, radius=20, pupil_center=(40, 30))
        assert np.allclose(a, -b, atol=1e-12)

    def test_a_centre_outside_the_panel_is_clamped_not_crashing(self) -> None:
        phase = fresnel_panel(-1000.0, radius=20, pupil_center=(5000, 5000))
        assert phase.shape == (fp.SLM_PANEL_H, fp.SLM_PANEL_W)
        assert phase.any()

    def test_phase_grows_quadratically_with_the_aperture(self) -> None:
        phase = fresnel_panel(-1000.0, radius=30, pupil_center=(60, 60))
        # Two radii on the same row: the phase ratio must be (r2/r1)^2.
        a = abs(phase[60, 60 + 10])
        b = abs(phase[60, 60 + 20])
        assert b / a == pytest.approx(4.0, rel=0.02)


class TestAdaptiveExposure:
    def _box(self, counts_per_ms: float):
        state = {"e": 0.1}

        def set_e(ms: float) -> float:
            state["e"] = float(ms)
            return float(ms)

        def read() -> np.ndarray:
            peak = min(FULL_SCALE, counts_per_ms * state["e"])
            frame = np.zeros((20, 20), dtype=np.float32)
            frame[8:12, 8:12] = peak
            return frame

        return state, set_e, read

    def test_a_bright_scene_is_backed_off_into_the_band(self) -> None:
        state, set_e, read = self._box(counts_per_ms=4000.0)
        got = fp.adapt_exposure(
            set_e, read, start_ms=0.5, floor_ms=0.02, ceiling_ms=20.0,
            full_scale=FULL_SCALE,
        )
        assert got.status == "ok"
        assert fp.ACCEPT_LOW <= got.fraction <= fp.ACCEPT_HIGH
        assert got.exposure_ms < 0.5, "a clipped start must be backed off"
        assert state["e"] == got.exposure_ms

    def test_a_dim_scene_is_reported_as_dim_not_ranked(self) -> None:
        _, set_e, read = self._box(counts_per_ms=0.5)
        got = fp.adapt_exposure(
            set_e, read, start_ms=0.1, floor_ms=0.02, ceiling_ms=1.0,
            full_scale=FULL_SCALE,
        )
        assert got.status == "dim"
        assert got.exposure_ms == pytest.approx(1.0)

    def test_a_scene_clipping_even_at_the_floor_is_reported_as_clipped(self) -> None:
        _, set_e, read = self._box(counts_per_ms=100000.0)
        got = fp.adapt_exposure(
            set_e, read, start_ms=0.1, floor_ms=0.02, ceiling_ms=20.0,
            full_scale=FULL_SCALE,
        )
        assert got.status == "clipped"
        assert got.exposure_ms == pytest.approx(0.02)

    def test_the_ceiling_bounds_the_search(self) -> None:
        _, set_e, read = self._box(counts_per_ms=4000.0)
        got = fp.adapt_exposure(
            set_e, read, start_ms=0.5, floor_ms=0.02, ceiling_ms=0.2,
            full_scale=FULL_SCALE,
        )
        assert got.exposure_ms <= 0.2


class TestSweep:
    def test_it_finds_the_true_diopter(self) -> None:
        bench = FakeBench(true_diopter=-1.7, width=1.2)
        result = _run(bench)
        assert result.best is not None
        assert result.best_diopter_m1 == pytest.approx(-1.7, abs=0.35)

    def test_the_positive_branch_is_found_too(self) -> None:
        """Sign is measured, not assumed: the camera may sit either side."""
        bench = FakeBench(true_diopter=+2.2, width=0.9)
        result = _run(bench)
        assert result.best_diopter_m1 == pytest.approx(+2.2, abs=0.35)

    def test_raw_peaks_are_equalised_but_peak_per_ms_is_not(self) -> None:
        """The trap, asserted directly.

        Adaptation lands every usable point near the same fraction of full
        scale, so raw peaks carry almost no information -- yet the ranking still
        resolves the focus, because it divides by the exposure that was needed.
        """
        bench = FakeBench(true_diopter=-1.7, width=1.2)
        result = _run(bench)
        usable = [p for p in result.points if p.status == "ok"]
        assert len(usable) >= 10
        peaks = np.array([p.peak for p in usable])
        ratios = np.array([p.peak_per_ms for p in usable])
        # Every usable peak was dragged into the same band ...
        assert peaks.max() <= fp.ACCEPT_HIGH * FULL_SCALE + 1.0
        assert peaks.min() >= fp.ACCEPT_LOW * FULL_SCALE - 1.0
        # ... while the exposure-corrected metric still spans an order of
        # magnitude, which is what makes the ranking possible at all.
        assert ratios.max() / max(ratios.min(), 1e-9) > 10.0
        assert result.gain_vs_flat is not None and result.gain_vs_flat > 1.5

    def test_the_exposure_is_rearmed_at_every_point(self) -> None:
        bench = FakeBench(true_diopter=-1.7, width=1.2)
        _run(bench)
        distinct = {round(e, 4) for e in bench.exposures}
        assert len(distinct) >= 4, (
            "a single fixed exposure cannot cover a sweep whose brightness "
            f"changes by more than an order of magnitude; saw {sorted(distinct)}"
        )

    def test_no_point_is_left_clipped_in_the_ranking(self) -> None:
        bench = FakeBench(true_diopter=-1.7, width=1.2)
        result = _run(bench)
        for point in result.points:
            if point.status == "ok":
                assert point.peak_fraction <= fp.ACCEPT_HIGH + 1e-6

    def test_out_of_band_points_are_excluded_and_counted(self) -> None:
        """Far from focus the light is below the noise floor at the ceiling."""
        bench = FakeBench(
            true_diopter=0.0, width=0.25, counts_per_ms=255.0,
            floor_intensity=1e-6,
        )
        result = _run(bench, coarse_points=9, diopter_range=6.0)
        skipped = [p for p in result.points if p.status != "ok"]
        assert skipped, "a narrow focus with a huge range must leave dim points"
        if result.best is not None:
            assert result.best.status == "ok"
        assert any("曝光带" in n for n in result.notes)

    def test_the_range_extends_when_the_winner_is_on_the_edge(self) -> None:
        bench = FakeBench(true_diopter=2.5, width=0.7)
        result = _run(bench, coarse_points=9, diopter_range=1.0)
        assert any("扩展到" in note for note in result.notes)
        assert result.best_diopter_m1 == pytest.approx(2.5, abs=0.4)

    def test_linearity_is_checked_at_the_winner(self) -> None:
        bench = FakeBench(true_diopter=-1.7, width=1.2)
        result = _run(bench)
        assert result.linearity_ratio is not None
        assert result.linearity_ratio == pytest.approx(1.0, abs=0.2)

    def test_a_nonlinear_camera_is_flagged(self) -> None:
        """Auto-gain would break peak/ms, so the probe must say so."""

        class _Gain(FakeBench):
            def read(self) -> np.ndarray:
                # Gain that rises with exposure: peak stops being linear in
                # exposure, which is the one assumption the ranking rests on.
                return super().read() * (1.0 + 50.0 * self.exposure_ms)

        bench = _Gain(true_diopter=-1.7, width=1.2)
        result = _run(bench)
        assert result.linearity_ratio is not None
        assert not (0.8 <= result.linearity_ratio <= 1.25)
        assert any("线性度" in n for n in result.notes)

    def test_an_explicit_diopter_list_is_honoured_verbatim(self) -> None:
        bench = FakeBench(true_diopter=-0.5, width=0.4)
        result = _run(
            bench,
            diopters=[-0.6, -0.5, -0.4, 0.0, 1.0],
            coarse_points=25,
            diopter_range=6.0,
        )
        seen = {round(p.diopter_m1, 6) for p in result.points}
        assert {-0.6, -0.5, -0.4, 0.0, 1.0} <= seen
        assert result.best_diopter_m1 == pytest.approx(-0.5, abs=0.05)

    def test_all_dark_points_give_no_best_but_still_explain_why(self) -> None:
        bench = FakeBench(
            true_diopter=0.0, width=0.1, counts_per_ms=1e-6, floor_intensity=0.0
        )
        result = _run(bench, coarse_points=5, diopter_range=1.0, refine_points=3)
        assert result.best is None
        assert result.best_focal_mm is None
        assert any("曝光带" in n for n in result.notes)


class TestParabolicVertex:
    def test_it_recovers_the_vertex_of_a_symmetric_peak(self) -> None:
        xs = [0.0, 1.0, 2.0]
        ys = [1.0, 2.0, 1.0]
        assert fp._parabolic_vertex(xs, ys) == pytest.approx(1.0)

    def test_it_declines_when_the_middle_is_not_a_maximum(self) -> None:
        assert fp._parabolic_vertex([0.0, 1.0, 2.0], [1.0, 0.5, 1.0]) is None

    def test_it_declines_on_a_flat_triple(self) -> None:
        assert fp._parabolic_vertex([0.0, 1.0, 2.0], [1.0, 1.0, 1.0]) is None

    def test_it_declines_on_an_uneven_grid(self) -> None:
        assert fp._parabolic_vertex([0.0, 1.0, 3.0], [1.0, 2.0, 1.0]) is None
