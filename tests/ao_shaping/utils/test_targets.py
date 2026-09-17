"""Regression anchors for ``ao_shaping.utils.targets``.

Pins the exact current behavior of the target-pattern generation module
(parametric shapes, masks, image loading, square-target sizing and the
CCD-frame -> SLM-grid target pipeline). All inputs are small and
hand-computable; no hardware, no network, no real target images.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.targets import (
    build_square_target_amplitude,
    compute_square_side,
    create_target_mask,
    create_target_shape,
    crop_resize_to_grid,
    load_target_image,
    square_target_from_measurement,
)


class TestCreateTargetShape:
    """Each shape branch of ``create_target_shape`` (7x7 grid, center (3, 3))."""

    def test_gaussian(self) -> None:
        out = create_target_shape("gaussian", 7, radius_ratio=0.3)
        assert out.shape == (7, 7)
        assert out.dtype == np.float32
        assert out.max() == 1.0  # normalized by peak
        assert out[3, 3] == 1.0  # center pixel is the peak
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
        assert np.allclose(out, out.T)  # radially symmetric

    def test_gaussian_as_amplitude(self) -> None:
        intensity = create_target_shape("gaussian", 7, radius_ratio=0.3)
        amp = create_target_shape("gaussian", 7, radius_ratio=0.3, as_amplitude=True)
        assert amp.dtype == np.float32
        assert np.allclose(amp, np.sqrt(intensity))

    def test_circle(self) -> None:
        # radius = 0.3 * 7 / 2 = 1.05 -> center (3,3) + 4 axis neighbors
        out = create_target_shape("circle", 7, radius_ratio=0.3)
        assert out.shape == (7, 7)
        assert out.dtype == np.float32
        assert out.max() == 1.0
        assert out.sum() == 5.0
        assert set(np.unique(out)) == {0.0, 1.0}

    def test_circle_as_amplitude_binary(self) -> None:
        intensity = create_target_shape("circle", 7, radius_ratio=0.3)
        amp = create_target_shape("circle", 7, radius_ratio=0.3, as_amplitude=True)
        assert np.array_equal(amp, intensity)  # sqrt of 0/1 mask is itself

    def test_square(self) -> None:
        out = create_target_shape("square", 7, side=3)
        assert out.shape == (7, 7)
        assert out.dtype == np.float32
        assert out.sum() == 9.0  # 3x3 block
        assert out[2:5, 2:5].sum() == 9.0
        assert out[1, 1] == 0.0

    def test_square_default_side(self) -> None:
        # side defaults to int(radius) = int(1.05) = 1 -> single center pixel
        out = create_target_shape("square", 7, radius_ratio=0.3)
        assert out.sum() == 1.0
        assert out[3, 3] == 1.0

    def test_annular(self) -> None:
        # inner_r = 0.2*1.05 = 0.21, outer_r = 0.5*7/2 = 1.75
        # ring pixels: r=1 (4 neighbors) + r=sqrt(2) (4 corners) = 8
        out = create_target_shape("annular", 7, radius_ratio=0.3)
        assert out.shape == (7, 7)
        assert out.dtype == np.float32
        assert out.sum() == 8.0
        assert out[3, 3] == 0.0  # center hole (r=0 < inner_r)
        assert out[2, 3] == 1.0  # r=1 inside the ring

    def test_grid(self) -> None:
        # lw = 0.2*7/2 = 0.7 -> vertical lines at x=0,3,6 and horizontal at y=0,3,6
        out = create_target_shape("grid", 7, nx=3, ny=3, line_width=0.2)
        assert out.shape == (7, 7)
        assert out.dtype == np.float32
        assert out.sum() == 33.0  # 3*7 + 3*7 - 9 crossings
        assert out[3, :].sum() == 7.0  # middle row fully lit
        assert out[0, 0] == 1.0

    def test_cross(self) -> None:
        # th = 0.05*7/2 = 0.175 -> only row 3 and column 3
        out = create_target_shape("cross", 7, thickness=0.05)
        assert out.shape == (7, 7)
        assert out.dtype == np.float32
        assert out.sum() == 13.0  # row 3 + col 3 - shared center
        assert out[3, :].sum() == 7.0
        assert out[:, 3].sum() == 7.0

    def test_rectangle(self) -> None:
        # side=3 is the SHORT side; long side = 3*2 = 6 -> rows 2..4, all cols
        out = create_target_shape("rectangle", 7, side=3, aspect_ratio=2.0)
        assert out.shape == (7, 7)
        assert out.dtype == np.float32
        assert out.sum() == 21.0  # 7 wide x 3 tall
        assert out[2:5, :].sum() == 21.0

    def test_pentagon(self) -> None:
        # point-up regular pentagon, circumradius 1.05 -> pixels (2,3) and (3,3)
        out = create_target_shape("pentagon", 7, radius_ratio=0.3)
        assert out.shape == (7, 7)
        assert out.dtype == np.float32
        assert set(np.unique(out)) == {0.0, 1.0}
        assert out[3, 3] == 1.0  # polygon center is always inside
        assert out.sum() == 2.0

    def test_tuple_size(self) -> None:
        out = create_target_shape("circle", (5, 7), radius_ratio=0.3)
        assert out.shape == (5, 7)
        assert out.dtype == np.float32

    def test_unknown_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown shape"):
            create_target_shape("bogus", 7)


class TestCreateTargetMask:
    """``create_target_mask`` on a (7, 7) grid (center (3.5, 3.5))."""

    def test_square(self) -> None:
        # half = 1.5 -> x, y in {2, 3, 4, 5}
        mask = create_target_mask("square", (7, 7), 3)
        assert mask.shape == (7, 7)
        assert mask.dtype == np.float64
        assert mask.sum() == 16.0
        assert set(np.unique(mask)) == {0.0, 1.0}

    def test_circle(self) -> None:
        # radius = 1.5 -> only the 4 center pixels (d^2 = 0.5 <= 2.25)
        mask = create_target_mask("circle", (7, 7), 3)
        assert mask.shape == (7, 7)
        assert mask.dtype == np.float64
        assert mask.sum() == 4.0

    def test_spot_same_as_circle(self) -> None:
        circle = create_target_mask("circle", (7, 7), 3)
        spot = create_target_mask("spot", (7, 7), 3)
        assert np.array_equal(circle, spot)

    def test_gaussian_default_sigma(self) -> None:
        # sigma defaults to size/6 = 0.5; peak at the 4 center pixels (d^2=0.5)
        mask = create_target_mask("gaussian", (7, 7), 3)
        assert mask.shape == (7, 7)
        assert mask.dtype == np.float64
        assert mask.max() == 1.0  # normalized by peak
        # at (2,3): d^2 = 2.5 -> exp(-2.5/0.5) / exp(-0.5/0.5) = exp(-4)
        assert np.isclose(mask[2, 3], np.exp(-4.0))

    def test_gaussian_sigma_override(self) -> None:
        mask = create_target_mask("gaussian", (7, 7), 3, sigma=1.0)
        assert mask.max() == 1.0
        # at (2,3): exp(-1.25) / exp(-0.25) = exp(-1)
        assert np.isclose(mask[2, 3], np.exp(-1.0))

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid shape"):
            create_target_mask("triangle", (7, 7), 3)

    def test_invalid_grid_size_raises(self) -> None:
        with pytest.raises(ValueError, match="grid_size"):
            create_target_mask("square", (7,), 3)


class TestLoadTargetImage:
    """``load_target_image``: .npy branch and image-file branch."""

    def test_npy_returns_array_unchanged(self, tmp_path) -> None:
        arr = np.array([[0.0, 0.25, 0.5], [0.75, 1.0, 0.5]], dtype=np.float64)
        path = tmp_path / "target.npy"
        np.save(path, arr)
        out = load_target_image(path)
        assert out.dtype == np.float32
        assert out.shape == arr.shape
        # max == 1 -> normalization is a no-op, values preserved
        assert np.array_equal(out, arr.astype(np.float32))

    def test_npy_normalizes_by_max(self, tmp_path) -> None:
        arr = np.array([[0.0, 2.0], [4.0, 8.0]], dtype=np.float64)
        path = tmp_path / "target.npy"
        np.save(path, arr)
        out = load_target_image(path)
        assert out.dtype == np.float32
        assert np.allclose(out, arr / 8.0)

    def test_npy_multichannel_takes_first_channel(self, tmp_path) -> None:
        arr = np.zeros((2, 2, 3), dtype=np.float64)
        arr[..., 0] = 4.0
        arr[..., 1] = 2.0
        arr[..., 2] = 1.0
        path = tmp_path / "target.npy"
        np.save(path, arr)
        out = load_target_image(path)
        assert out.shape == (2, 2)
        assert np.allclose(out, 1.0)  # first channel 4.0 / max 4.0

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="Target image not found"):
            load_target_image(tmp_path / "nope.npy")

    def test_image_branch_requires_skimage(self, tmp_path) -> None:
        # The image-file branch lazily imports scikit-image, which is not
        # installed in this environment -> the exact current behavior is an
        # ImportError with the install guidance message.
        from PIL import Image

        img = Image.fromarray(np.array([[0, 64], [255, 128]], dtype=np.uint8))
        path = tmp_path / "tiny.png"
        img.save(path)
        with pytest.raises(ImportError, match="scikit-image"):
            load_target_image(path)


class TestComputeSquareSide:
    """``compute_square_side`` = round(factor * spot * p_cam / d_slm)."""

    def test_default_pitches(self) -> None:
        assert compute_square_side(10) == 15  # 1.5 * 10 * (8e-6 / 8e-6)
        assert compute_square_side(8) == 12

    def test_factor(self) -> None:
        assert compute_square_side(10, factor=2.0) == 20

    def test_pitch_ratio(self) -> None:
        assert compute_square_side(10, factor=2.0, p_cam=4e-6, d_slm=8e-6) == 10

    def test_returns_int(self) -> None:
        assert isinstance(compute_square_side(10), int)


class TestBuildSquareTargetAmplitude:
    """``build_square_target_amplitude``: centered 1/0 square, float64."""

    def test_even_side(self) -> None:
        out = build_square_target_amplitude(10, 10, 4)
        assert out.shape == (10, 10)
        assert out.dtype == np.float64
        assert out.sum() == 16.0
        assert out[3:7, 3:7].sum() == 16.0
        assert out[2, 2] == 0.0

    def test_odd_side(self) -> None:
        out = build_square_target_amplitude(10, 10, 5)
        assert out.sum() == 25.0

    def test_side_larger_than_grid(self) -> None:
        out = build_square_target_amplitude(6, 6, 10)
        assert out.sum() == 36.0  # clipped to the full grid

    def test_side_one(self) -> None:
        out = build_square_target_amplitude(10, 10, 1)
        assert out.sum() == 1.0
        assert out[5, 5] == 1.0

    def test_rectangular_grid(self) -> None:
        out = build_square_target_amplitude(6, 8, 4)
        assert out.shape == (6, 8)
        assert out.sum() == 16.0


class TestCropResizeToGrid:
    """``crop_resize_to_grid``: argmax-centered crop + bilinear resize."""

    def test_resize_output(self) -> None:
        frame = np.zeros((10, 12), dtype=np.float32)
        frame[5, 6] = 1.0
        out = crop_resize_to_grid(frame, 8, 8)
        assert out.shape == (8, 8)
        assert out.dtype == np.float32
        # 10x10 crop with the source pixel at (5,5); output (4,4) interpolates
        # it with weight (1-0.125)^2 = 0.875^2
        assert np.isclose(out[4, 4], 0.875 * 0.875)
        assert out.sum() == pytest.approx(0.875 * 0.875)

    def test_same_size_returns_copy(self) -> None:
        frame = np.arange(64, dtype=np.float32).reshape(8, 8)
        out = crop_resize_to_grid(frame, 8, 8)
        assert out.shape == (8, 8)
        assert out.dtype == np.float32
        # argmax at (7,7) -> crop is frame[3:8, 3:8] zero-padded bottom/right
        assert out[0, 0] == 27.0
        assert out[0, 4] == 31.0
        assert out[0, 5] == 0.0
        assert out[4, 4] == 63.0
        assert out[5, 0] == 0.0
        assert out[7, 7] == 0.0

    def test_edge_argmax_pads(self) -> None:
        frame = np.zeros((10, 8), dtype=np.float32)
        frame[9, 7] = 1.0
        out = crop_resize_to_grid(frame, 8, 8)
        assert out.shape == (8, 8)
        # crop frame[5:10, 3:8] (5x5) padded to 8x8; source pixel at (4,4)
        assert out[4, 4] == 1.0
        assert out[5, 4] == 0.0
        assert out[4, 5] == 0.0
        assert out[7, 7] == 0.0

    def test_nan_replaced_by_zero(self) -> None:
        frame = np.zeros((10, 12), dtype=np.float32)
        frame[5, 6] = np.nan
        out = crop_resize_to_grid(frame, 8, 8)
        assert out.sum() == 0.0


class TestSquareTargetFromMeasurement:
    """``square_target_from_measurement``: CCD frame -> normalized grid target."""

    def test_basic(self) -> None:
        frame = np.zeros((10, 12), dtype=np.float32)
        frame[5, 6] = 100.0
        target, info = square_target_from_measurement(frame, 4, 8, 8)
        assert target.shape == (8, 8)
        assert target.dtype == np.float32
        assert target.max() == 1.0
        assert target.min() == 0.0
        assert info["side_cam_px"] == 4.0
        assert info["n_pixels"] == 16
        assert info["max_brightness"] == 100.0
        assert info["background"] == 0.0
        assert info["total_intensity"] == 100.0
        assert info["centroid"] == [5, 6]
        assert info["target_ccd"].shape == (10, 12)
        assert info["target_ccd"].dtype == np.float32
        assert np.isclose(info["target_ccd"].sum(), 1.0)  # 16 * 1/16
        assert info["side_grid_bins"] == [3.2, 3.2]  # 4 * 8 / 10

    def test_side_px_nonpositive_raises(self) -> None:
        frame = np.zeros((10, 12), dtype=np.float32)
        frame[5, 6] = 1.0
        with pytest.raises(ValueError, match="side_px"):
            square_target_from_measurement(frame, 0, 8, 8)

    def test_no_signal_raises(self) -> None:
        frame = np.zeros((10, 12), dtype=np.float32)
        with pytest.raises(ValueError, match="无有效信号"):
            square_target_from_measurement(frame, 4, 8, 8)

    def test_uniform_background_raises(self) -> None:
        # 10th percentile == 10 -> signal is fully subtracted -> no signal
        frame = np.full((10, 12), 10.0, dtype=np.float32)
        with pytest.raises(ValueError, match="无有效信号"):
            square_target_from_measurement(frame, 4, 8, 8)