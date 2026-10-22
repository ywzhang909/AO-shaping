"""The simulated SLM/CCD must speak the **same** contract as the real bench.

What this locks down
--------------------
The acquisition scripts that produced the hardware reports
(``scripts/slm/hw_gs_defocus_probe.py``, ``scripts/slm/run_gs_refine_zernike_sweep.py``)
have to run unchanged against the digital twin. Three gaps used to make that
impossible:

1. ``slm.display_data(gray, wait_time_s)`` — the hardware driver's **second
   positional** parameter is ``wait_time_s`` (see :meth:`Santec.display_data`), and
   ``tools/kernels/bench_kernels.py`` calls it exactly that way. The simulation's
   ``display_data`` used to take ``memory_number`` second, so the positional call
   either raised ``TypeError`` or silently meant something else.
2. ``slm.set_phase_overlay(phase_rad | None)`` — a radian overlay added **before**
   the radian→gray conversion, and visible in the far field. Without it the
   simulated defocus response was fake (the spot never blurred).
3. The bench's metrology (pixel pitch, camera pixel, ROI, absolute photometry,
   detector noise, panel aberration) was not reachable from the twin, so a
   simulation could not be lined up with the measured bench.

The defaults of every knob below are the *pre-existing* behaviour, so the existing
callers' results are bit-for-bit unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.drivers.sim.slm_pib_sim import (
    BEAM_W0,
    CAM_WINDOW_PX,
    CCD_PIXEL_UM,
    FAR_FIELD_PADDING,
    SLM_H,
    SLM_PIXEL_UM,
    SLM_W,
    TWO_PI_GRAY,
    WAVELENGTH_NM,
    SimNoiseSpec,
    SimPibCCD,
    SimPibSystem,
    SimPanelAberration,
    SimSLMPib,
    get_system,
    reset_system,
)

PANEL_SHAPE = (SLM_H, SLM_W)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fwhm_1d(profile: np.ndarray) -> float:
    """Full width at half maximum with linearly interpolated crossings."""
    peak = float(profile.max())
    if peak <= 0:
        return 0.0
    half = 0.5 * peak
    top = int(np.argmax(profile))
    lo = top
    while lo > 0 and profile[lo] > half:
        lo -= 1
    hi = top
    while hi < profile.size - 1 and profile[hi] > half:
        hi += 1
    p_lo, p_top, p_hi = float(profile[lo]), float(profile[top]), float(profile[hi])
    left = lo + (half - p_lo) * (top - lo) / (p_top - p_lo) if p_top != p_lo else float(top)
    right = hi + (half - p_hi) * (hi - top) / (p_top - p_hi) if p_top != p_hi else float(top)
    return float(right - left)


def _spot_fwhm(image: np.ndarray) -> tuple[float, float]:
    """FWHM along the row and the column through the brightest pixel."""
    row, col = np.unravel_index(int(np.argmax(image)), image.shape)
    return _fwhm_1d(image[row, :]), _fwhm_1d(image[:, col])


def _halo_fraction(image: np.ndarray, radius_px: float = 20.0) -> float:
    """Fraction of the far-field power outside a disc around the 0-order.

    A defocus moves power out of the core, so this is the robust "the light got
    spread" probe (the core width alone is not: a phase-modulated pupil can
    narrow the core while dumping most of the power into a halo).
    """
    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width]
    outside = (yy - height // 2) ** 2 + (xx - width // 2) ** 2 > radius_px**2
    total = float(image.sum())
    if total <= 0:
        return 0.0
    return float(image[outside].sum() / total)


def _chirp_panel(peak_rad: float) -> tuple[np.ndarray, float]:
    """A smooth quadratic pupil phase and its chirp coefficient.

    The phase is ``phi(r) = peak_rad * (r / r_corner)**2`` over the whole panel
    (no hard edge, so the far field stays clean). The returned coefficient is
    ``k`` in ``phi = k * r**2`` with ``r`` in **metres**.

    The predicted effect on the 0-order spot is analytic: a pupil chirp of
    ``k`` defocuses the far field by ``b = k * w0**2`` (dimensionless), so the
    spot radius grows by ``sqrt(1 + b**2)``.
    """
    yy, xx = np.mgrid[0:PANEL_SHAPE[0], 0:PANEL_SHAPE[1]]
    r2_px = (xx - (PANEL_SHAPE[1] - 1) / 2.0) ** 2 + (yy - (PANEL_SHAPE[0] - 1) / 2.0) ** 2
    corner_px2 = float(r2_px.max())
    panel = peak_rad * r2_px / corner_px2
    pixel_m = SLM_PIXEL_UM * 1e-6
    k = peak_rad / (corner_px2 * pixel_m**2)
    return panel, float(k)


def _defocus_panel(peak_rad: float = 24.0) -> np.ndarray:
    """The hardware's focus-correction panel (a pupil chirp of the measured size).

    24 rad at the panel corner is the order of magnitude of the real thing: the
    hardware's ``fresnel_panel(f_v = 1549.3 mm)`` puts 23.96 rad (~3.8 waves) on
    the panel edge.
    """
    panel, _ = _chirp_panel(peak_rad)
    return panel


# ---------------------------------------------------------------------------
# 1. display_data accepts the hardware's calling forms
# ---------------------------------------------------------------------------


class TestDisplayDataContract:
    @pytest.fixture
    def slm(self) -> SimSLMPib:
        system = SimPibSystem(seed=42)
        panel = SimSLMPib(system=system)
        panel.open()
        return panel

    def test_positional_wait_time_is_accepted(self, slm: SimSLMPib) -> None:
        """``slm.display_data(gray, wait_time_s)`` — the bench_kernels call form."""
        gray = np.zeros(PANEL_SHAPE, dtype=np.uint16)
        slot = slm.display_data(gray, 0.5)
        assert isinstance(slot, int), "the return value must stay an int (a slot number)"
        assert slot == 0, "no memory slot was requested, so 0 is returned"

    def test_memory_number_keyword_still_works(self, slm: SimSLMPib) -> None:
        """Backwards compatibility: the old keyword form must be untouched."""
        gray = np.zeros(PANEL_SHAPE, dtype=np.uint16)
        assert slm.display_data(gray, memory_number=7) == 7

    def test_full_keyword_form_is_accepted(self, slm: SimSLMPib) -> None:
        gray = np.zeros(PANEL_SHAPE, dtype=np.uint16)
        slot = slm.display_data(
            gray, memory_number=7, memory_mode=0, wait_time_s=0.5
        )
        assert slot == 7

    def test_bare_call_still_works(self, slm: SimSLMPib) -> None:
        gray = np.zeros(PANEL_SHAPE, dtype=np.uint16)
        assert slm.display_data(gray) == 0

    def test_wait_time_does_not_change_the_optics(self, slm: SimSLMPib) -> None:
        """The wait is ignored on purpose: the simulated panel is instantaneous."""
        system = slm.system
        gray = slm.create_phase_from_array(np.zeros(PANEL_SHAPE, dtype=np.float64))
        slm.display_data(gray, 0.0)
        first = system.far_field().copy()
        slm.display_data(gray, 5.0)
        assert np.array_equal(first, system.far_field())

    def test_bench_kernel_chain_runs_on_the_simulated_panel(self) -> None:
        """The real measurement kernel must work against the simulated devices.

        This is the actual gap that blocked "run the hardware script in sim":
        ``bench_kernels.acquire_settled`` writes with
        ``slm.display_data(gray, wait_time_s)``.
        """
        from ao_shaping.tools.kernels.bench_kernels import acquire_settled

        system = SimPibSystem(seed=42)
        panel = SimSLMPib(system=system)
        cam = SimPibCCD(system=system, exposure_time_ms=1.0, cam_size=250)
        with panel, cam:
            result = acquire_settled(
                cam,
                panel,
                np.zeros(PANEL_SHAPE, dtype=np.float64),
                n_frames=2,
                n_discard=1,
                wait_time_s=0.5,
                min_wait_s=0.0,
                max_wait_s=0.3,
                n_stable=1,
            )
        assert result.frame.shape == (250, 250)
        assert np.isfinite(result.frame).all()


# ---------------------------------------------------------------------------
# 2. set_phase_overlay — the radian overlay
# ---------------------------------------------------------------------------


class TestPhaseOverlay:
    @pytest.fixture
    def slm(self) -> SimSLMPib:
        system = SimPibSystem(seed=42)
        panel = SimSLMPib(system=system)
        panel.open()
        return panel

    def test_overlay_is_added_before_the_radian_to_gray_conversion(
        self, slm: SimSLMPib
    ) -> None:
        """``gray = mod(cmd + overlay, 2pi) / 2pi * 2pi_gray`` — as on the hardware."""
        cmd = np.zeros(PANEL_SHAPE, dtype=np.float64)
        overlay = _defocus_panel(6.0)
        slm.set_phase_overlay(overlay)
        gray = slm.create_phase_from_array(cmd)
        expected = (
            np.mod(cmd + overlay, 2.0 * np.pi) / (2.0 * np.pi) * TWO_PI_GRAY
        ).astype(np.uint16)
        assert np.array_equal(gray, expected)

    def test_overlay_reaches_the_far_field(self, slm: SimSLMPib) -> None:
        """The far field must see the *sum*, otherwise the defocus response is fake."""
        cmd = np.zeros(PANEL_SHAPE, dtype=np.float64)
        slm.create_phase_from_array(cmd)
        flat = slm.system.far_field().copy()
        slm.set_phase_overlay(_defocus_panel(6.0))
        slm.create_phase_from_array(cmd)
        blurred = slm.system.far_field().copy()
        assert not np.array_equal(flat, blurred), (
            "the overlay must change the far field -- otherwise the simulated "
            "defocus response is not physical"
        )

    def test_overlay_defocus_broadens_the_spot(self, slm: SimSLMPib) -> None:
        """A defocusing overlay must make the 0-order spot *wider*."""
        cmd = np.zeros(PANEL_SHAPE, dtype=np.float64)
        slm.create_phase_from_array(cmd)
        flat_fwhm = min(_spot_fwhm(slm.system.far_field()))
        slm.set_phase_overlay(_defocus_panel(24.0))
        slm.create_phase_from_array(cmd)
        defocused_fwhm = min(_spot_fwhm(slm.system.far_field()))
        assert defocused_fwhm > 2.0 * flat_fwhm, (
            f"defocus did not broaden the spot: {flat_fwhm:.2f} -> {defocused_fwhm:.2f} px"
        )

    def test_broadening_matches_the_analytic_defocus(self, slm: SimSLMPib) -> None:
        """The overlay must act as a *real* pupil defocus, not a fudge.

        A pupil chirp ``phi = k r**2`` defocuses the far field by the
        dimensionless parameter ``b = k * w0**2``, so the 0-order spot radius
        grows by ``sqrt(1 + b**2)``. The simulated spot has to follow that law
        (within 25%), otherwise the overlay is not really a lens.
        """
        cmd = np.zeros(PANEL_SHAPE, dtype=np.float64)
        slm.create_phase_from_array(cmd)
        flat_fwhm = min(_spot_fwhm(slm.system.far_field()))
        panel, k = _chirp_panel(24.0)
        w0_m = BEAM_W0 * SLM_PIXEL_UM * 1e-6
        b = k * w0_m**2
        slm.set_phase_overlay(panel)
        slm.create_phase_from_array(cmd)
        measured = min(_spot_fwhm(slm.system.far_field())) / flat_fwhm
        predicted = float(np.sqrt(1.0 + b**2))
        assert measured == pytest.approx(predicted, rel=0.25), (
            f"measured broadening {measured:.2f}x vs analytic {predicted:.2f}x "
            f"(b = {b:.2f}) -- the overlay is not acting as a pupil defocus"
        )

    def test_the_hardware_focus_correction_panel_spreads_the_light(
        self, slm: SimSLMPib
    ) -> None:
        """The very panel the hardware scripts send must change the simulated spot.

        ``fresnel_panel(f_v = 1549.3 mm)`` is the focus correction the hardware
        run used; the simulated twin has to react to it, otherwise the simulated
        defocus scan is not comparable with the measured one.

        The signature of a defocus is *energy leaving the core*, so the assertion
        is on the halo fraction (power outside a 20 px disc around the 0-order),
        not on the core width: the correction panel is phase-modulated only
        inside the illuminated disc, so it pushes most of the light into a halo
        while the core itself stays narrow.
        """
        from ao_shaping.tools.kernels.bench_kernels import fresnel_panel

        cmd = np.zeros(PANEL_SHAPE, dtype=np.float64)
        slm.create_phase_from_array(cmd)
        flat_halo = _halo_fraction(slm.system.far_field())
        correction = fresnel_panel(1549.3)
        assert correction.shape == PANEL_SHAPE
        slm.set_phase_overlay(correction)
        slm.create_phase_from_array(cmd)
        corrected_halo = _halo_fraction(slm.system.far_field())
        assert corrected_halo > 10.0 * flat_halo, (
            f"the focus-correction panel did not spread the light: halo "
            f"{flat_halo:.4f} -> {corrected_halo:.4f}"
        )

    def test_clearing_the_overlay_restores_the_flat_far_field(
        self, slm: SimSLMPib
    ) -> None:
        cmd = np.zeros(PANEL_SHAPE, dtype=np.float64)
        slm.create_phase_from_array(cmd)
        flat = slm.system.far_field().copy()
        slm.set_phase_overlay(_defocus_panel(6.0))
        slm.create_phase_from_array(cmd)
        slm.set_phase_overlay(None)
        slm.create_phase_from_array(cmd)
        assert np.array_equal(flat, slm.system.far_field())

    def test_overlay_is_stored_as_a_copy(self, slm: SimSLMPib) -> None:
        """The caller mutating its array must not change the overlay."""
        overlay = np.zeros(PANEL_SHAPE, dtype=np.float64)
        slm.set_phase_overlay(overlay)
        overlay[0, 0] = 1.234
        stored = slm.get_phase_overlay()
        assert stored is not None
        assert stored[0, 0] == 0.0

    def test_get_phase_overlay_returns_none_when_unset(self, slm: SimSLMPib) -> None:
        assert slm.get_phase_overlay() is None

    def test_size_mismatch_warns_and_skips_instead_of_raising(
        self, slm: SimSLMPib
    ) -> None:
        """Same contract as the hardware: warn, skip, keep sending."""
        cmd = np.zeros(PANEL_SHAPE, dtype=np.float64)
        slm.create_phase_from_array(cmd)
        flat = slm.system.far_field().copy()
        slm.set_phase_overlay(np.zeros((64, 64), dtype=np.float64))
        gray = slm.create_phase_from_array(cmd)  # must not raise
        assert gray.shape == PANEL_SHAPE
        assert np.array_equal(flat, slm.system.far_field()), (
            "a mismatched overlay must be skipped, not applied"
        )


# ---------------------------------------------------------------------------
# 3. The metrology knobs
# ---------------------------------------------------------------------------


class TestDefaultsAreUnchanged:
    def test_far_field_is_still_peak_normalised_to_100(self) -> None:
        assert np.isclose(SimPibSystem(seed=42).far_field().max(), 100.0)

    def test_camera_window_default_is_unchanged(self) -> None:
        system = SimPibSystem(seed=42)
        assert SimPibCCD(system=system).cam_size == CAM_WINDOW_PX == 250

    def test_counts_mode_is_off_by_default(self) -> None:
        system = SimPibSystem(seed=42)
        assert system.counts_mode is False
        assert system.peak_counts_per_ms is None

    def test_panel_aberration_is_off_by_default(self) -> None:
        system = SimPibSystem(seed=42)
        assert system.panel_aberration is None
        clean = system.far_field()
        other = SimPibSystem(seed=42).far_field()
        assert np.array_equal(clean, other)

    def test_noise_spec_without_photometry_is_rejected(self) -> None:
        """A counts noise model needs a counts scale -- fail loudly, not silently."""
        with pytest.raises(ValueError):
            SimPibSystem(seed=42, noise_spec=SimNoiseSpec(read_sigma_counts=0.18))

    def test_default_padding_stays_the_measured_angular_scale(self) -> None:
        """``FAR_FIELD_PADDING`` is the value that reproduces the measured scale.

        The measured bench scale is ``lambda*f/(p*d_cam) = 7557`` camera px per
        panel px (Kx = 7546.4 / Ky = 7495.9), and the twin's convention is
        "one far-field sample = one camera pixel" => ``N_pad = 7557``, i.e.
        ``padding = 7557/1920 = 3.94 ~ 4``.
        """
        assert FAR_FIELD_PADDING == 4
        system = SimPibSystem(seed=42)
        row_um, col_um = system.far_field_pitch_um
        # The column axis is the one the camera sees across the panel width.
        assert col_um == pytest.approx(CCD_PIXEL_UM, rel=0.05)
        # ... and the reported scale must match the measured one.
        scale = (
            WAVELENGTH_NM
            * 1e-9
            * system.focal_length_m
            / (system.slm_pixel_um * 1e-6 * CCD_PIXEL_UM * 1e-6)
        )
        assert scale == pytest.approx(7557.0, rel=0.01)


class TestPhotometry:
    def test_counts_scale_with_the_exposure(self) -> None:
        """0.2 ms must land on the measured flat-field peak (26.7 counts)."""
        system = SimPibSystem(seed=42, peak_counts_per_ms=133.5)
        frame = system.counts_frame(0.2)
        assert frame.max() == pytest.approx(26.7, rel=0.02)

    def test_counts_are_proportional_to_the_exposure(self) -> None:
        system = SimPibSystem(seed=42, peak_counts_per_ms=133.5)
        short = system.counts_frame(0.2).max()
        long = system.counts_frame(0.4).max()
        assert long == pytest.approx(2.0 * short, rel=1e-6)

    def test_camera_reads_counts_when_the_photometry_is_configured(self) -> None:
        system = SimPibSystem(seed=42, peak_counts_per_ms=133.5)
        cam = SimPibCCD(system=system, exposure_time_ms=0.2, cam_size=250)
        cam.open()
        frame = cam.get_numpy_image()
        assert frame.shape == (250, 250)
        assert frame.max() == pytest.approx(26.7, rel=0.05)

    def test_read_noise_sigma_is_reproducible(self) -> None:
        """The measured bench floor is sigma ~ 0.18 counts on a 26.7-count peak."""
        system = SimPibSystem(
            seed=42,
            peak_counts_per_ms=133.5,
            noise_spec=SimNoiseSpec(read_sigma_counts=0.18),
        )
        frames = np.stack([system.counts_frame(0.2) for _ in range(8)])
        dark = frames[:, :200, :200].reshape(8, -1)
        assert float(dark.std(axis=1).mean()) == pytest.approx(0.18, rel=0.25)

    def test_uint8_quantisation_is_applied(self) -> None:
        system = SimPibSystem(
            seed=42,
            peak_counts_per_ms=133.5,
            noise_spec=SimNoiseSpec(quantize_bits=8),
        )
        frame = system.counts_frame(0.2)
        assert np.all(frame == np.round(frame))
        assert frame.max() <= 255.0

    def test_counts_frame_rejects_a_negative_exposure(self) -> None:
        system = SimPibSystem(seed=42, peak_counts_per_ms=133.5)
        with pytest.raises(ValueError):
            system.counts_frame(-1.0)


class TestPanelAberration:
    def test_zernike_aberration_changes_the_far_field(self) -> None:
        """Real panels are not ideal: the flat-field spot is much wider than 2f."""
        clean = SimPibSystem(seed=42).far_field()
        aberrated = SimPibSystem(
            seed=42,
            panel_aberration=SimPanelAberration(zernike_rad={4: 0.5}),
        ).far_field()
        assert not np.array_equal(clean, aberrated)

    def test_roughness_changes_the_far_field(self) -> None:
        clean = SimPibSystem(seed=42).far_field()
        rough = SimPibSystem(
            seed=42,
            panel_aberration=SimPanelAberration(roughness_sigma_rad=0.05, roughness_seed=7),
        ).far_field()
        assert not np.array_equal(clean, rough)

    def test_empty_aberration_is_a_no_op(self) -> None:
        clean = SimPibSystem(seed=42).far_field()
        empty = SimPibSystem(
            seed=42, panel_aberration=SimPanelAberration()
        ).far_field()
        assert np.array_equal(clean, empty)


class TestSystemFactoryKnobs:
    def test_reset_system_forwards_the_knobs(self) -> None:
        system = reset_system(seed=7, far_field_padding=8, cam_window_px=123)
        try:
            assert system.far_field_padding == 8
            assert system.cam_window_px == 123
        finally:
            reset_system(seed=42)

    def test_reset_system_forwards_the_photometry(self) -> None:
        system = reset_system(seed=7, peak_counts_per_ms=100.0)
        try:
            assert system.counts_mode is True
            assert system.peak_counts_per_ms == 100.0
        finally:
            reset_system(seed=42)

    def test_get_system_returns_the_existing_instance(self) -> None:
        first = get_system()
        assert get_system() is first
