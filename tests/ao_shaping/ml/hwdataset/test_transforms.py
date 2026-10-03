from __future__ import annotations

"""Tests for :mod:`ml.hwdataset.transforms`.

These tests verify the coherent block mean, geometry equivalence with the
canonical helpers, and the public API contracts for phase/far-field transforms.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from ao_shaping.optimizer.wfless import slm_square_shaping
from ao_shaping.runners import gsnet_offline

from ml.hwdataset.transforms import (
    DEFAULT_PANEL_CENTER,
    DEFAULT_PANEL_RADIUS,
    DEFAULT_SLM_MAX_GRAY,
    coherent_block_mean,
    crop_panel_roi,
    farfield_frame_to_grid,
    freeform_grid_to_panel,
    grayscale_to_phase_rad,
    phase_to_grid,
    zernike_coeffs_to_panel,
)


class TestGrayscaleToPhaseRad:
    """Tests for grayscale_to_phase_rad."""

    def test_basic_values(self) -> None:
        """0 -> 0, max_gray -> just under 2*pi, monotone."""
        two_pi = 2.0 * np.pi
        gray = np.array([0, DEFAULT_SLM_MAX_GRAY], dtype=np.uint16)
        phase = grayscale_to_phase_rad(gray, max_gray=DEFAULT_SLM_MAX_GRAY)
        assert phase.shape == (2,)
        assert np.isclose(phase[0], 0.0, atol=1e-6)
        assert phase[1] < two_pi
        assert np.isclose(phase[1], two_pi, atol=1e-5)

    def test_monotone_and_shape(self) -> None:
        """Monotonic and shape preserved."""
        gray = np.linspace(0, DEFAULT_SLM_MAX_GRAY, 10, dtype=np.float32)
        phase = grayscale_to_phase_rad(gray, max_gray=DEFAULT_SLM_MAX_GRAY)
        assert phase.shape == gray.shape
        assert np.all(np.diff(phase) >= -1e-6)

    def test_max_gray_variants(self) -> None:
        """max_gray=993 vs 1023 differ as expected."""
        g = np.array([500, 993], dtype=np.float32)
        p993 = grayscale_to_phase_rad(g, max_gray=993)
        p1023 = grayscale_to_phase_rad(g, max_gray=1023)
        assert not np.allclose(p993, p1023)

    def test_invalid_max_gray(self) -> None:
        """ValueError for max_gray < 1."""
        with pytest.raises(ValueError, match="max_gray"):
            grayscale_to_phase_rad(np.array([0]), max_gray=0)


class TestCropPanelRoi:
    """Tests for crop_panel_roi."""

    def test_inside_panel(self) -> None:
        """Centre + radius well inside panel -> exactly 2*radius per axis."""
        phase = np.zeros((1200, 1920), dtype=np.float32)
        phase[600, 960] = 1.0
        roi = crop_panel_roi(phase, (960, 600), 100.0)
        assert roi.shape == (200, 200)
        # rows 500..699, cols 860..1059 -> panel (600, 960) lands at (100, 100)
        assert roi[100, 100] == 1.0
        assert np.count_nonzero(roi) == 1

    def test_clamped_near_edge(self) -> None:
        """Centre near edge -> clamped, result smaller, no exception."""
        phase = np.zeros((120, 120), dtype=np.float32)
        roi = crop_panel_roi(phase, (10, 10), 50.0)
        assert roi.shape == (60, 60)
        assert roi.shape[0] <= 100
        assert roi.shape[1] <= 100

    def test_radius_exceeding_panel_yields_whole_panel(self) -> None:
        """A radius wider than the panel degrades to the whole panel, not an error."""
        phase = np.arange(120 * 120, dtype=np.float32).reshape(120, 120)
        roi = crop_panel_roi(phase, (60, 60), 10_000.0)
        assert roi.shape == (120, 120)
        assert np.array_equal(roi, phase)

    def test_roi_missing_the_panel_raises(self) -> None:
        """A centre far off-panel is a caller bug and must be loud."""
        with pytest.raises(ValueError, match="does not intersect"):
            crop_panel_roi(np.zeros((120, 120), dtype=np.float32), (5000, 5000), 10.0)

    def test_radius_zero_or_negative(self) -> None:
        """radius <= 0 returns input unchanged."""
        phase = np.ones((50, 60), dtype=np.float32)
        roi0 = crop_panel_roi(phase, (30, 25), 0.0)
        roineg = crop_panel_roi(phase, (30, 25), -5.0)
        assert np.array_equal(roi0, phase)
        assert np.array_equal(roineg, phase)

    def test_c_contiguous(self) -> None:
        """Result is C-contiguous."""
        phase = np.zeros((100, 100), dtype=np.float32)
        roi = crop_panel_roi(phase, (50, 50), 10.0)
        assert roi.flags["C_CONTIGUOUS"]

    def test_not_2d(self) -> None:
        """ValueError if not 2D."""
        with pytest.raises(ValueError, match="2D"):
            crop_panel_roi(np.array([1, 2, 3]), (0, 0), 1.0)


class TestCoherentBlockMean:
    """Tests for coherent_block_mean."""

    def test_coherence_beats_arithmetic_mean(self) -> None:
        """Coherent mean better than arithmetic mean on wrapped phase."""
        rng = np.random.default_rng(123)
        x = np.linspace(0, 2 * np.pi * 3, 120)
        y = np.linspace(0, 2 * np.pi * 2, 120)
        X, Y = np.meshgrid(x, y)
        phi_true = (3.0 * np.sin(X / 9.0) + 2.0 * np.cos(Y / 7.0)) % (2.0 * np.pi)
        phi_true = phi_true.astype(np.float32)

        cos_g, sin_g = coherent_block_mean(phi_true, 4)
        phi_coh = np.arctan2(sin_g, cos_g)

        block_h = int(np.ceil(120 / 4))
        block_w = int(np.ceil(120 / 4))
        phi_pad = np.pad(phi_true, ((0, block_h * 4 - 120), (0, block_w * 4 - 120)))
        phi_arith = np.zeros((4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                sl = phi_pad[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]
                phi_arith[i, j] = float(np.mean(sl))

        assert phi_coh.shape == (4, 4)
        assert phi_arith.shape == (4, 4)
        assert not np.allclose(phi_coh, phi_arith, atol=1e-1)

    def test_geometry_equivalence_with_canonical(self) -> None:
        """Matches pupil_phase_to_grid reference on cos/sin."""
        rng = np.random.default_rng(42)
        phi = rng.uniform(0, 2 * np.pi, (1200, 1920)).astype(np.float32)
        cos_g, sin_g = coherent_block_mean(phi, 64)
        cos_ref = gsnet_offline.pupil_phase_to_grid(np.cos(phi), 64)
        sin_ref = gsnet_offline.pupil_phase_to_grid(np.sin(phi), 64)
        assert np.allclose(cos_g, cos_ref, atol=1e-5)
        assert np.allclose(sin_g, sin_ref, atol=1e-5)

    def test_non_divisible_padding(self) -> None:
        """1200 not divisible by 64; output (64,64) and matches reference."""
        rng = np.random.default_rng(7)
        phi = rng.uniform(0, 2 * np.pi, (1200, 1920)).astype(np.float32)
        cos_g, sin_g = coherent_block_mean(phi, 64)
        assert cos_g.shape == (64, 64)
        assert sin_g.shape == (64, 64)
        cos_ref = gsnet_offline.pupil_phase_to_grid(np.cos(phi), 64)
        sin_ref = gsnet_offline.pupil_phase_to_grid(np.sin(phi), 64)
        assert np.allclose(cos_g, cos_ref, atol=1e-5)
        assert np.allclose(sin_g, sin_ref, atol=1e-5)

    def test_tiny_input(self) -> None:
        """Tiny input 5x7 -> grid 4."""
        phi = np.array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]], dtype=np.float32)
        cos_g, sin_g = coherent_block_mean(phi, 4)
        assert cos_g.shape == (4, 4)
        assert sin_g.shape == (4, 4)

    def test_invalid_grid(self) -> None:
        """ValueError for grid < 1."""
        with pytest.raises(ValueError, match="grid"):
            coherent_block_mean(np.zeros((10, 10)), 0)

    def test_invalid_shape(self) -> None:
        """ValueError if not 2D."""
        with pytest.raises(ValueError, match="2D"):
            coherent_block_mean(np.array([0, 1, 2]), 2)


def _disc_mask(shape, center_yx, radius):
    """Boolean mask of a disc, `shape` = (H, W), `center_yx` = (row, col)."""
    height, width = shape
    cy, cx = center_yx
    grid_y, grid_x = np.ogrid[:height, :width]
    return (grid_x - cx) ** 2 + (grid_y - cy) ** 2 <= (radius - 1e-6) ** 2


def _ramp_disc_phase(shape, center_yx, radius, peak=3.0):
    """A disc of smooth radial phase on a flat background.

    The background is exactly phase 0, so its recovered phase is 0 and the
    recovered phase is a clean discriminator between "disc" and "no disc".
    Deliberately smooth: a uniformly RANDOM phase has a coherent magnitude of
    only ~1/sqrt(pixels_per_block) by construction, so any assertion about high
    coherence on noise would assert the opposite of the physics.
    """
    height, width = shape
    cy, cx = center_yx
    grid_y, grid_x = np.ogrid[:height, :width]
    r2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    inside = _disc_mask(shape, center_yx, radius)
    phase = np.zeros((height, width), dtype=np.float32)
    phase[inside] = (peak * r2[inside] / radius**2).astype(np.float32)
    return phase


def _phase_of(cos_grid, sin_grid):
    """Recover the local mean phase from the (cos, sin) pair."""
    return np.arctan2(sin_grid, cos_grid)


class TestPhaseToGrid:
    """Tests for phase_to_grid."""

    def test_end_to_end_disc_phase(self) -> None:
        """A smooth disc phase is recovered with high coherence and real extent."""
        phase = _ramp_disc_phase(
            (1200, 1920),
            (DEFAULT_PANEL_CENTER[1], DEFAULT_PANEL_CENTER[0]),
            DEFAULT_PANEL_RADIUS,
        )
        cos_g, sin_g = phase_to_grid(phase, 64)
        coherence = np.hypot(cos_g, sin_g)
        recovered = _phase_of(cos_g, sin_g)
        assert coherence.shape == recovered.shape == (64, 64)

        lit = np.abs(recovered) > 1.0
        assert int(lit.sum()) > 1500, int(lit.sum())
        # Cells away from the disc edge see an almost constant phase, so the
        # coherent magnitude must be ~1 there (measured 0.999).
        assert float(np.median(coherence[lit])) > 0.99

    def test_roi_crop_concentrates_the_signal(self) -> None:
        """The ROI crop is what makes the pupil visible, not a detail.

        Without the crop, a 64-cell grid over the whole 1200x1920 panel gives each
        cell ~19x30 px, so a 500 px disc covers only ~53x34 of the 4096 cells and
        every one of them is diluted by the flat background. Measured on this
        corpus geometry: mean |recovered phase| 1.14 cropped vs 0.50 whole, and
        2070 vs 927 cells above 1 rad. The 1.8x bound leaves margin on both.
        """
        phase = _ramp_disc_phase(
            (1200, 1920),
            (DEFAULT_PANEL_CENTER[1], DEFAULT_PANEL_CENTER[0]),
            DEFAULT_PANEL_RADIUS,
        )
        cropped_cos, cropped_sin = phase_to_grid(phase, 64)
        whole_cos, whole_sin = coherent_block_mean(phase, 64)
        assert cropped_cos.shape == whole_cos.shape == (64, 64)

        cropped_phase = np.abs(_phase_of(cropped_cos, cropped_sin))
        whole_phase = np.abs(_phase_of(whole_cos, whole_sin))
        assert cropped_phase.mean() > 1.8 * whole_phase.mean()
        assert int((cropped_phase > 1.0).sum()) > 1.8 * int((whole_phase > 1.0).sum())

    def test_default_roi_contains_the_widest_corpus_structure(self) -> None:
        """The default ROI must contain the widest structure in the corpus.

        `slm_pib_*` phases reach radius ~479 px (69% of all usable records), so a
        450 px default -- the illumination radius -- would clip them. A disc is
        fully inside an axis-aligned square of half-width h exactly when h >= R,
        because the disc's extreme along each axis sits at radius R. So the exact
        test is that NO disc pixel is lost by the crop.
        """
        shape = (1200, 1920)
        cy, cx = DEFAULT_PANEL_CENTER[1], DEFAULT_PANEL_CENTER[0]
        r_struct = 479.0
        phase = _ramp_disc_phase(shape, (cy, cx), r_struct)
        full_mask = _disc_mask(shape, (cy, cx), r_struct).astype(np.float32)
        disc_pixels = int(full_mask.sum())

        roi = crop_panel_roi(phase, DEFAULT_PANEL_CENTER, DEFAULT_PANEL_RADIUS)
        assert roi.shape == (1000, 1000)
        # Compare the MASKS, not the non-zero counts: the ramp is exactly 0 at the
        # disc centre, so one in-disc pixel legitimately reads as zero.
        kept_mask = crop_panel_roi(full_mask, DEFAULT_PANEL_CENTER, DEFAULT_PANEL_RADIUS)
        assert int(kept_mask.sum()) == disc_pixels
        # Every disc pixel strictly off-centre carries a non-zero phase, i.e. the
        # crop kept the structure instead of blanking it. The centre pixel is
        # legitimately 0 because the radial ramp starts at r = 0.
        lit = kept_mask > 0
        lit[int(DEFAULT_PANEL_RADIUS), int(DEFAULT_PANEL_RADIUS)] = False
        assert np.all(roi[lit] > 0)

        # A too-small ROI provably does lose disc area -- the regression this guards.
        clipped = crop_panel_roi(full_mask, DEFAULT_PANEL_CENTER, 450.0)
        assert int(clipped.sum()) < disc_pixels

        # The grid still resolves the structure instead of diluting it away.
        cos_g, sin_g = phase_to_grid(phase, 64)
        assert float(np.max(np.abs(_phase_of(cos_g, sin_g)))) > 0.9 * 3.0


class TestZernikeCoeffsToPanel:
    """Tests for zernike_coeffs_to_panel."""

    def test_basic_panel(self) -> None:
        """15-term vector -> finite, zero outside aperture."""
        coeffs = np.zeros(15, dtype=np.float32)
        coeffs[0] = 0.1
        coeffs[3] = 0.2
        phase = zernike_coeffs_to_panel(
            coeffs,
            n_max=4,
            resolution=(64, 48),
            radius=20.0,
        )
        assert phase.shape == (48, 64)
        assert np.all(np.isfinite(phase))
        cy, cx = 24, 32
        Y, X = np.ogrid[:48, :64]
        outside = (X - cx) ** 2 + (Y - cy) ** 2 > (20.0 + 1e-6) ** 2
        assert np.allclose(phase[outside], 0.0, atol=1e-6)

    def test_non_triangular_raises(self) -> None:
        """Non-triangular length propagates ValueError."""
        coeffs = np.zeros(7, dtype=np.float32)
        with pytest.raises(ValueError):
            zernike_coeffs_to_panel(
                coeffs,
                n_max=3,
                resolution=(32, 32),
                radius=10.0,
            )


class TestFreeformGridToPanel:
    """Tests for freeform_grid_to_panel."""

    def test_bit_identical_to_canonical(self) -> None:
        """Bit-identical to slm_square_shaping._freeform_phase_radians."""
        rng = np.random.default_rng(99)
        g = 24
        cells = rng.uniform(0, 2 * np.pi, size=(g * g,)).astype(np.float32)
        panel = freeform_grid_to_panel(cells, grid=g, resolution=(1920, 1200))
        canonical = slm_square_shaping._freeform_phase_radians(
            cells.reshape(g, g),
            resolution=(1920, 1200),
            grid=g,
        )
        assert panel.shape == canonical.shape
        assert np.array_equal(panel, canonical.astype(np.float32))

    def test_non_square_raises(self) -> None:
        """ValueError for a length that is not a perfect square (99, not 100)."""
        with pytest.raises(ValueError, match="perfect square"):
            freeform_grid_to_panel(np.zeros(99), grid=10, resolution=(100, 100))

    def test_square_length_is_accepted(self) -> None:
        """100 == 10*10 is a VALID freeform grid and must not raise."""
        panel = freeform_grid_to_panel(np.zeros(100), grid=10, resolution=(100, 100))
        assert panel.shape == (100, 100)
        assert np.array_equal(panel, np.zeros((100, 100), dtype=np.float32))

    def test_grid_argument_mismatch_raises(self) -> None:
        """A `grid` that disagrees with len(coeffs) is a caller bug, not a guess."""
        with pytest.raises(ValueError, match="disagrees"):
            freeform_grid_to_panel(np.zeros(100), grid=7, resolution=(100, 100))
        with pytest.raises(ValueError, match="disagrees"):
            freeform_grid_to_panel(np.zeros(576), grid=32, resolution=(1920, 1200))

    def test_non_tiling_grid(self) -> None:
        """Grid that does not tile evenly."""
        rng = np.random.default_rng(5)
        g = 7
        cells = rng.uniform(0, 2 * np.pi, size=(g * g,)).astype(np.float32)
        panel = freeform_grid_to_panel(cells, grid=g, resolution=(100, 100))
        assert panel.shape == (100, 100)
        assert np.all(np.isfinite(panel))


class TestFarfieldFrameToGrid:
    """Tests for farfield_frame_to_grid."""

    def test_abs255(self) -> None:
        """"abs255" keeps ABSOLUTE intensity -- the exposure-time signal.

        The frame's maximum is deliberately NOT 255: a peak-normalising
        implementation would return 1.0 here and this test would still pass, so
        the max has to differ from full scale to discriminate the two modes.
        """
        frame = np.array([[0, 50], [100, 64]], dtype=np.uint8)
        res = farfield_frame_to_grid(frame, 2, mode="abs255")
        assert res.shape == (2, 2)
        assert np.max(res) == pytest.approx(100.0 / 255.0, abs=1e-6)
        assert np.all(res >= 0.0) and np.all(res <= 1.0)

    def test_abs255_is_not_peak_normalised(self) -> None:
        """Regression: "abs255" must not secretly peak-normalise.

        Routing "abs255" through the peak-normalising helper makes every frame
        come back with max == 1.0, which silently deletes the brightness
        information that encodes the exposure time the model is given as input.
        """
        dim = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        absolute = farfield_frame_to_grid(dim, 2, mode="abs255")
        peaked = farfield_frame_to_grid(dim, 2, mode="peak")
        assert np.max(absolute) == pytest.approx(40.0 / 255.0, abs=1e-6)
        assert np.max(peaked) == pytest.approx(1.0)

    def test_abs255_clips_above_full_scale(self) -> None:
        """A float frame above 255 is clipped to 1.0, never wrapped."""
        frame = np.array([[0.0, 300.0], [255.0, 64.0]], dtype=np.float32)
        res = farfield_frame_to_grid(frame, 2, mode="abs255")
        assert np.max(res) == pytest.approx(1.0)
        assert np.all(res <= 1.0)

    def test_peak(self) -> None:
        """peak mode gives max == 1.0."""
        frame = np.array([[10, 20], [30, 40]], dtype=np.float32)
        res = farfield_frame_to_grid(frame, 2, mode="peak")
        assert np.max(res) == pytest.approx(1.0)

    def test_raw(self) -> None:
        """raw returns unchanged values."""
        frame = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        res = farfield_frame_to_grid(frame, 2, mode="raw")
        assert np.allclose(res, frame, atol=1e-6)

    def test_bogus_mode(self) -> None:
        """ValueError for invalid mode."""
        frame = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="mode"):
            farfield_frame_to_grid(frame, 2, mode="bogus")

    def test_all_zero_no_nan(self) -> None:
        """All-zero frame produces no NaN."""
        frame = np.zeros((20, 20), dtype=np.float32)
        res = farfield_frame_to_grid(frame, 4, mode="abs255")
        assert np.all(np.isfinite(res))

    def test_invalid_grid(self) -> None:
        """ValueError for grid < 1."""
        with pytest.raises(ValueError, match="grid"):
            farfield_frame_to_grid(np.zeros((10, 10)), 0)

    def test_invalid_shape(self) -> None:
        """ValueError if not 2D."""
        with pytest.raises(ValueError, match="2D"):
            farfield_frame_to_grid(np.array([0, 1]), 2)
