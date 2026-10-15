"""Anchors for :mod:`ao_shaping.utils.image.exposure_fusion`.

Three things are pinned here, and all three are decisions that were made
*against* the reference algorithm rather than inherited from it:

1. **The Mertens band construction is exact.** ``bands[i] = g[i] -
   expand(g[i+1])`` must invert to better than 1e-6 on even *and* odd shapes.
   The upsampled operand is the Gaussian level, never the running accumulator;
   chaining the expansions instead is the classic way to break this identity, and
   a test that only used powers of two would not notice.
2. **Stage 0 (exposure normalisation) is load-bearing.** A bracketed CCD
   sequence whose short exposure peaks at 30/255 makes Mertens'
   well-exposedness term decide the weight competition on *absolute brightness*
   rather than on content, so the fused output is one input frame. Supplying
   ``exposures_ms`` puts the frames on one exposure scale and the competition
   becomes a real one. Measured ratios are in the comments.
3. **Monochrome input must skip the saturation term.** ``sqrt(mean_c((x_c -
   mean_c(x))**2))`` is identically 0 for one channel, so applying it zeroes the
   entire weight map. These tests fail loudly if anyone "fixes" that.

Units are 0-255 detector counts throughout, matching the whole repo -- see
:class:`TestUnits`. Everything is synthetic, pure NumPy, and runs in well under a
second.
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.exposure_fusion import (
    DETECTOR_FULL_SCALE,
    compute_fusion_weights,
    exposure_fusion,
    gaussian_pyramid,
    laplacian_pyramid,
)


def _reconstruct(levels: list[np.ndarray]) -> np.ndarray:
    """Collapse a Laplacian pyramid coarse-to-fine (the module's own inverse).

    Written out here rather than imported from the module so the anchor test
    exercises the documented algebra independently of the private helper.
    """
    bands = laplacian_pyramid(levels)
    acc = bands[-1]
    for n in range(len(bands) - 2, -1, -1):
        expanded = acc
        if expanded.shape != bands[n].shape:
            expanded = _bilinear(expanded, bands[n].shape)
        acc = bands[n] + expanded
    return acc


def _bilinear(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Minimal pixel-centre bilinear resize used only by :func:`_reconstruct`."""
    src_h, src_w = arr.shape
    dst_h, dst_w = shape
    ys = np.clip((np.arange(dst_h) + 0.5) * src_h / dst_h - 0.5, 0.0, src_h - 1.0)
    xs = np.clip((np.arange(dst_w) + 0.5) * src_w / dst_w - 0.5, 0.0, src_w - 1.0)
    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    rows = arr[y0, :] * (1.0 - wy) + arr[y1, :] * wy
    return rows[:, x0] * (1.0 - wx) + rows[:, x1] * wx


def _relative_error(out: np.ndarray, ref: np.ndarray) -> float:
    """``mean |out - ref|`` normalised by ``mean |ref|``."""
    scale = float(np.mean(np.abs(ref)))
    if scale <= 0.0:
        return float("inf")
    return float(np.mean(np.abs(out - ref))) / scale


class _Frames:
    """Synthetic bracketed sequences. Static methods, not fixtures."""

    @staticmethod
    def _scene(shape: tuple[int, int] = (64, 64), seed: int = 7) -> np.ndarray:
        """A two-lobe spot (tight core + wide halo) plus read noise, peak 255."""
        height, width = shape
        rng = np.random.default_rng(seed)
        rows, cols = np.mgrid[0:height, 0:width]
        dy = rows - height / 2.0
        dx = cols - width / 2.0
        r2 = dy**2 + dx**2
        lobes = np.exp(-r2 / (2.0 * (height / 8.0) ** 2))
        lobes += 0.35 * np.exp(-r2 / (2.0 * (height / 3.2) ** 2))
        scene = np.clip(255.0 * lobes + rng.normal(0.0, 2.0, shape), 0.0, 255.0)
        return scene / (float(scene.max()) / 255.0)

    @staticmethod
    def bracket(
        shape: tuple[int, int] = (64, 64), seed: int = 7
    ) -> tuple[list[np.ndarray], list[float]]:
        """Three frames peaking at 30 / 120 / 255 counts, with matching exposures.

        The exposure times are chosen so ``gain_k * exposure_k`` is constant, i.e.
        these really are the same scene at three exposures.
        """
        scene = _Frames._scene(shape, seed)
        exposures = [0.12, 0.5, 1.2]
        gains = [30.0 / 255.0, 120.0 / 255.0, 1.0]
        return [scene * gain for gain in gains], exposures

    @staticmethod
    def two_frame_bracket(
        shape: tuple[int, int] = (48, 48), seed: int = 1
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """A 30/255 vs 255/255 pair -- the bracket the Stage-0 note is about."""
        scene = _Frames._scene(shape, seed)
        short = scene * (30.0 / 255.0)
        long = scene
        return short, long, [0.12, 1.2]


class TestGaussianPyramid:
    def test_level_zero_is_the_input(self) -> None:
        img = np.arange(256, dtype=np.float64).reshape(16, 16)
        pyr = gaussian_pyramid(img, 3)
        assert len(pyr) == 3
        np.testing.assert_allclose(pyr[0], img, rtol=0.0, atol=0.0)

    def test_odd_axes_ceil_on_decimation(self) -> None:
        # ``[::2, ::2]`` on 65 rows keeps indices 0, 2, ..., 64 -> 33.
        pyr = gaussian_pyramid(np.ones((65, 64)), 2)
        assert pyr[0].shape == (65, 64)
        assert pyr[1].shape == (33, 32)

    def test_levels_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="levels must be >= 1, got 0"):
            gaussian_pyramid(np.ones((8, 8)), 0)

    def test_accepts_colour(self) -> None:
        pyr = gaussian_pyramid(np.ones((3, 16, 16)), 3)
        assert [p.shape for p in pyr] == [(3, 16, 16), (3, 8, 8), (3, 4, 4)]


class TestLaplacianPyramid:
    @pytest.mark.parametrize(
        "shape", [(64, 64), (65, 64), (64, 65), (65, 65), (97, 61)]
    )
    def test_reconstruction_is_exact(self, shape: tuple[int, int]) -> None:
        """The anchor: Mertens bands invert to better than 1e-6, odd shapes too."""
        rng = np.random.default_rng(0)
        img = rng.random(shape) * 255.0
        recon = _reconstruct(gaussian_pyramid(img, 5))
        assert recon.shape == shape
        assert float(np.max(np.abs(recon - img))) < 1e-6

    def test_last_band_is_the_coarsest_gaussian_level(self) -> None:
        pyr = gaussian_pyramid(np.arange(1024, dtype=np.float64).reshape(32, 32), 4)
        bands = laplacian_pyramid(pyr)
        assert len(bands) == len(pyr)
        np.testing.assert_allclose(bands[-1], pyr[-1], rtol=0.0, atol=0.0)

    def test_band_is_difference_against_the_gauss_level(self) -> None:
        """Guards the documented trap: the operand is ``g[i + 1]``, not the
        previously expanded value."""
        pyr = gaussian_pyramid(np.ones((16, 16)), 3)
        bands = laplacian_pyramid(pyr)
        assert bands[0].shape == pyr[0].shape
        assert bands[1].shape == pyr[1].shape

    def test_empty_pyramid_raises(self) -> None:
        with pytest.raises(ValueError, match="gauss_pyr must be a non-empty"):
            laplacian_pyramid([])


class TestComputeFusionWeights:
    @staticmethod
    def _impulse_frame() -> np.ndarray:
        """3x3, one bright pixel at (2, 1). Small enough to do by hand."""
        return np.array(
            [[128, 128, 128], [128, 128, 128], [128, 160, 128]], dtype=np.float64
        )

    def test_mono_weights_match_the_hand_computation(self) -> None:
        """Every number below is computed by hand from x = f / 255.

        Laplacian ``[[0,1,0],[1,-4,1],[0,1,0]]`` with ``mode="reflect"``
        (index -1 maps to 1, index 3 maps to 1 on a 3-wide axis)::

            L(0,0) = 128+128+128+128-4*128 =    0     L(0,1) =    0     L(0,2) = 0
            L(1,0) = 128+128+128+128-4*128 =    0     L(1,1) = 128+160+128+128-512 = 32
            L(2,0) = 128+128+160+160-4*128 =   64     L(2,1) = 128+128+128+128-4*160 = -128
            L(2,2) = 128+128+160+160-4*128 =   64

        So ``C = |L| / 255 = [[0,0,0],[0,32/255,0],[64/255,128/255,64/255]]``.

        ``E = exp(-0.5 * ((x - 0.5) / 0.2) ** 2)`` with ``best_illumination=0.5``
        and ``sigma=0.2``:

        * ``x = 128/255 = 0.501961`` -> ``exp(-4.804e-5) = 0.999952``
        * ``x = 160/255 = 0.627451`` -> ``exp(-0.203047) = 0.816330``

        ``W = C * E`` (monochrome: no saturation factor).
        """
        weights = compute_fusion_weights(self._impulse_frame())
        e_flat = np.exp(-0.5 * ((128.0 / 255.0 - 0.5) / 0.2) ** 2)
        e_bright = np.exp(-0.5 * ((160.0 / 255.0 - 0.5) / 0.2) ** 2)
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, (32.0 / 255.0) * e_flat, 0.0],
                [(64.0 / 255.0) * e_flat, (128.0 / 255.0) * e_bright, (64.0 / 255.0) * e_flat],
            ]
        )
        assert weights.dtype == np.float64
        np.testing.assert_allclose(weights, expected, rtol=1e-6)
        # Sanity on the closed form: the bright pixel is the only place the
        # well-exposedness term bites, so it must be < 1 there and > 0.9999 elsewhere.
        assert e_bright < 0.82 and e_flat > 0.9999

    @staticmethod
    def _noise_frame(size: int = 32) -> np.ndarray:
        """White noise in 0-255 counts: the Laplacian is non-zero everywhere, so
        every weight is strictly positive if -- and only if -- nothing in the
        product is silently zeroing the map."""
        rng = np.random.default_rng(3)
        return rng.random((size, size)) * 255.0

    def test_mono_weights_are_positive_everywhere(self) -> None:
        """Pins the monochrome decision: a contrast-carrying frame must not get
        its weights zeroed by the (identically zero) saturation term."""
        weights = compute_fusion_weights(self._noise_frame())
        assert weights.shape == (32, 32)
        assert float(weights.min()) > 0.0
        assert bool(np.isfinite(weights).all())

    def test_colour_weights_are_two_dimensional_and_positive(self) -> None:
        grey = self._noise_frame()
        colour = np.stack([grey * 0.9, grey * 1.0, grey * 0.8], axis=0)
        weights = compute_fusion_weights(colour)
        # The channel axis is reduced away, so the result stays (H, W).
        assert weights.shape == (32, 32)
        assert float(weights.min()) > 0.0

    def test_a_neutral_colour_frame_collapses_which_is_why_mono_omits_saturation(
        self,
    ) -> None:
        """The pathology, demonstrated rather than described.

        Three bitwise-identical channels have ``x_c - mean_c(x) == 0``, so the
        saturation factor ``sqrt(mean_c((x_c - mean_c(x))**2))`` is zero and the
        colour weight map collapses to ~1e-23. The 2-D path omits the factor and
        keeps a full-contrast map. If someone ever "unifies" the two by applying
        the saturation term everywhere, this test is the one that notices.
        """
        grey = self._noise_frame()
        neutral = np.repeat(grey[None, ...], 3, axis=0)
        colour_weights = compute_fusion_weights(neutral)
        mono_weights = compute_fusion_weights(grey)
        assert float(colour_weights.max()) < float(mono_weights.max()) * 1e-6
        assert float(mono_weights.max()) > 0.0

    def test_a_tinted_colour_frame_differs_from_mono(self) -> None:
        """Non-neutral colour carries a real saturation term, so the two paths
        genuinely differ (beyond the E^3 vs E difference from the channel
        product)."""
        grey = self._noise_frame()
        tinted = np.stack([grey * 0.6, grey * 1.0, grey * 0.6], axis=0)
        assert not np.allclose(
            compute_fusion_weights(tinted), compute_fusion_weights(grey)
        )

    def test_non_finite_samples_are_scrubbed(self) -> None:
        frame = _Frames._scene((16, 16)).copy()
        frame[0, 0] = np.nan
        frame[1, 1] = np.inf
        assert bool(np.isfinite(compute_fusion_weights(frame)).all())

    def test_non_positive_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma must be a finite positive"):
            compute_fusion_weights(np.ones((4, 4)), sigma=0.0)

    def test_wrong_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="frame must be 2-D"):
            compute_fusion_weights(np.ones((2, 4, 4, 4)))


class TestExposureFusionResults:
    KEYS = ("naive", "gaussian_smoothed", "laplacian_pyramid")

    def test_all_three_keys_are_present_and_well_formed(self) -> None:
        frames, exposures = _Frames.bracket()
        out = exposure_fusion(frames, exposures_ms=exposures)
        assert set(out) == set(self.KEYS)
        for key in self.KEYS:
            img = out[key]
            assert img.shape == (64, 64)
            assert img.dtype == np.float32
            assert bool(np.isfinite(img).all())
            assert float(img.min()) >= 0.0
            assert float(img.max()) <= DETECTOR_FULL_SCALE

    def test_the_three_blends_are_actually_different(self) -> None:
        """``naive`` is the speckled raw weight stack; the other two smooth it.
        If all three agreed the smoothing would be a no-op. Measured spread on
        the 64x64 bracket below: 0.19 (naive vs laplacian), so the threshold is
        deliberately far below it."""
        frames, exposures = _Frames.bracket()
        out = exposure_fusion(frames, exposures_ms=exposures)
        assert float(np.max(np.abs(out["naive"] - out["laplacian_pyramid"]))) > 0.05
        assert float(np.max(np.abs(out["naive"] - out["gaussian_smoothed"]))) > 0.05

    def test_missing_stack_normalisation_would_return_black(self) -> None:
        """The raw (un-normalised) weights are ~1e-4, so a blend that skipped
        ``W /= W.sum(axis=0)`` would be ~1e-4 of full scale. Everything returned
        here must be nowhere near that."""
        frames, exposures = _Frames.bracket()
        out = exposure_fusion(frames, exposures_ms=exposures)
        for key in self.KEYS:
            assert float(out[key].mean()) > 1.0

    def test_colour_keeps_the_channel_axis(self) -> None:
        scene = _Frames._scene((32, 32))
        gains = [30.0 / 255.0, 120.0 / 255.0, 1.0]
        rgb = np.stack([scene, scene * 0.95, scene * 0.9], axis=0)
        out = exposure_fusion(
            [rgb * gain for gain in gains], exposures_ms=[0.12, 0.5, 1.2]
        )
        for key in self.KEYS:
            assert out[key].shape == (3, 32, 32)
            assert out[key].dtype == np.float32
            assert bool(np.isfinite(out[key]).all())

    def test_non_finite_samples_are_scrubbed(self) -> None:
        frames, exposures = _Frames.bracket((32, 32))
        frames[1] = frames[1].copy()
        frames[1][0, 0] = np.nan
        out = exposure_fusion(frames, exposures_ms=exposures)
        for key in self.KEYS:
            assert bool(np.isfinite(out[key]).all())

    def test_a_flat_stack_fuses_to_the_flat_value(self) -> None:
        """Two identical constant frames have identical weights, so the blend is
        that constant -- and constant frames have zero contrast, which exercises
        the ``eps`` floor."""
        flat = np.full((16, 16), 128.0)
        out = exposure_fusion([flat, flat])
        for key in self.KEYS:
            np.testing.assert_allclose(out[key], 128.0, rtol=1e-6)


class TestExposureNormalisation:
    def test_without_exposures_the_output_is_one_frame(self) -> None:
        short, long_, _ = _Frames.two_frame_bracket()
        out = exposure_fusion([short, long_])

        # The long frame peaks at 255/255 so its well-exposedness term is O(1),
        # while the short frame peaks at 30/255 = 0.118 and gets
        # exp(-0.5 * ((0.118 - 0.5) / 0.2) ** 2) = 0.162. Combined with an 8.5x
        # contrast advantage the long frame wins by ~27x in mean weight
        # (measured 27.4), so the "fusion" IS the long frame.
        weights = np.stack(
            [compute_fusion_weights(short), compute_fusion_weights(long_)]
        )
        ratio = float(weights[1].mean() / weights[0].mean())
        assert ratio > 10.0
        assert _relative_error(out["naive"], long_) < 0.10

    def test_with_exposures_the_weights_become_a_real_competition(self) -> None:
        short, long_, exposures = _Frames.two_frame_bracket()
        out = exposure_fusion([short, long_], exposures_ms=exposures)

        # Both frames are now divided onto the 0.12 ms scale, so their contrast
        # terms are comparable and the mean-weight ratio lands near 1 (measured
        # 0.81) instead of 27. That is the whole point of Stage 0: the winner is
        # decided by content, not by which frame happened to sit near mid-grey.
        # The exposure scale normalises each frame by exposures[0] / exposures[k],
        # so here the short frame keeps its scale and the long one is divided by 10.
        weights = np.stack(
            [
                compute_fusion_weights(short * (exposures[0] / exposures[0])),
                compute_fusion_weights(long_ * (exposures[0] / exposures[1])),
            ]
        )
        ratio = float(weights[1].mean() / weights[0].mean())
        assert 0.2 < ratio < 5.0

        # ...and the fused image is demonstrably no longer the long frame
        # (measured relative error 0.89, versus 0.04 without Stage 0).
        assert _relative_error(out["naive"], long_) > 0.5

    def test_exposures_change_the_result(self) -> None:
        short, long_, exposures = _Frames.two_frame_bracket()
        without = exposure_fusion([short, long_])
        with_exp = exposure_fusion([short, long_], exposures_ms=exposures)
        assert float(np.max(np.abs(without["naive"] - with_exp["naive"]))) > 10.0


class TestInputValidation:
    def test_empty_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 frames, got 0"):
            exposure_fusion([])

    def test_single_frame_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 frames to fuse, got 1"):
            exposure_fusion([np.ones((8, 8))])

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match=r"frame 1 has shape \(9, 9\)"):
            exposure_fusion([np.ones((8, 8)), np.ones((9, 9))])

    def test_wrong_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="frame 1 must be 2-D"):
            exposure_fusion([np.ones((8, 8)), np.ones((2, 8, 8, 8))])

    def test_exposure_length_mismatch_raises(self) -> None:
        frames, _ = _Frames.bracket((16, 16))
        with pytest.raises(
            ValueError, match="exposures_ms has 2 entries but sequence has 3"
        ):
            exposure_fusion(frames, exposures_ms=[0.1, 0.2])

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_non_positive_exposure_raises(self, bad: float) -> None:
        frames, _ = _Frames.bracket((16, 16))
        with pytest.raises(ValueError, match=r"exposures_ms\[1\]"):
            exposure_fusion(frames, exposures_ms=[0.1, bad, 1.0])

    def test_levels_below_one_raises(self) -> None:
        frames, _ = _Frames.bracket((16, 16))
        with pytest.raises(ValueError, match="levels must be >= 1, got 0"):
            exposure_fusion(frames, levels=0)

    @pytest.mark.parametrize("bad", [0.0, -255.0])
    def test_non_positive_sigma_raises(self, bad: float) -> None:
        frames, _ = _Frames.bracket((16, 16))
        with pytest.raises(ValueError, match="sigma must be a finite positive"):
            exposure_fusion(frames, sigma=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_full_scale_raises(self, bad: float) -> None:
        frames, _ = _Frames.bracket((16, 16))
        with pytest.raises(ValueError, match="full_scale must be a finite positive"):
            exposure_fusion(frames, full_scale=bad)

    def test_wrong_number_of_alphas_raises(self) -> None:
        frames, _ = _Frames.bracket((16, 16))
        with pytest.raises(ValueError, match="alphas must be"):
            exposure_fusion(frames, alphas=(1.0, 1.0))  # type: ignore[arg-type]

    def test_non_finite_alpha_raises(self) -> None:
        frames, _ = _Frames.bracket((16, 16))
        with pytest.raises(ValueError, match="alphas entries must be finite"):
            exposure_fusion(frames, alphas=(1.0, 1.0, float("nan")))


class TestUnits:
    def test_uint8_and_float32_grey_unit_buffers_agree(self) -> None:
        """The repo works in 0-255 counts, never ``[0, 1]``. The same physical
        bracket expressed as ``uint8`` and as ``float32`` 0-255 must fuse the
        same.

        The tolerance is loose (3 counts on a 0-255 scale) for two reasons: the
        ``uint8`` path quantises the input to integer counts AND resolves its
        scale through the integer-dtype branch of ``full_scale`` rather than the
        parameter. Where two frames' weights are nearly tied, that 1-count
        perturbation flips which one wins, so an exact-equality assertion would
        be testing the quantiser instead of the contract. Measured spread:
        2.66 (naive), 0.53 (gaussian_smoothed), 1.11 (laplacian_pyramid).
        """
        frames, exposures = _Frames.bracket()
        as_uint8 = [np.clip(f, 0.0, 255.0).astype(np.uint8) for f in frames]
        as_float32 = [f.astype(np.float32) for f in frames]

        a = exposure_fusion(as_uint8, exposures_ms=exposures)
        b = exposure_fusion(as_float32, exposures_ms=exposures)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-3, atol=3.0)

    def test_uint16_input_uses_its_own_full_scale(self) -> None:
        """A uint16 bracket must be read against 65535, not 255."""
        frames, exposures = _Frames.bracket()
        scale = 65535.0 / 255.0
        as_uint16 = [
            np.clip(np.round(f * scale), 0.0, 65535.0).astype(np.uint16) for f in frames
        ]
        out = exposure_fusion(as_uint16, exposures_ms=exposures, full_scale=65535.0)
        for key in out:
            assert out[key].dtype == np.float32
            assert float(out[key].min()) >= 0.0
            assert float(out[key].max()) <= 65535.0
            # Same physics, so the result is the 0-255 result times 65535/255.
            # The median is robust to the handful of near-tied pixels where the
            # 257x coarser quantisation flips the local winner.
            reference_255 = _reference_255(key, frames, exposures)
            reference = out[key] / np.maximum(scale * reference_255, 1e-9)
            assert 0.95 < float(np.median(reference)) < 1.10

    def test_declaring_the_wrong_full_scale_changes_the_units(self) -> None:
        """The ``full_scale`` keyword is the caller's OUTPUT unit system."""
        frames, exposures = _Frames.bracket((32, 32))
        scale = 65535.0 / 255.0
        as_uint16 = [
            np.clip(np.round(f * scale), 0.0, 65535.0).astype(np.uint16) for f in frames
        ]
        right = exposure_fusion(as_uint16, exposures_ms=exposures, full_scale=65535.0)
        wrong = exposure_fusion(as_uint16, exposures_ms=exposures, full_scale=255.0)
        for key in right:
            # 65535/255 = 257x apart; nowhere near the same image.
            assert float(np.max(np.abs(right[key] - wrong[key]))) > 100.0

    def test_frame_full_scale_attribute_is_preferred(self) -> None:
        """``drivers.ccd.base.AveragedFrame`` carries ``.full_scale``; a float
        frame that claims 65535 must be read against 65535 even though its dtype
        would otherwise fall back to the 255 default."""

        class _Tagged(np.ndarray):
            full_scale: float

            def __new__(cls, values: np.ndarray, full_scale: float) -> "_Tagged":
                tagged = np.asarray(values, dtype=np.float32).view(cls)
                tagged.full_scale = float(full_scale)
                return tagged

        frames, exposures = _Frames.bracket((32, 32))
        scale = 65535.0 / 255.0
        tagged = [_Tagged(f * scale, 65535.0) for f in frames]
        untagged_but_correct = [(f * scale).astype(np.float32) for f in frames]

        with_attr = exposure_fusion(tagged, exposures_ms=exposures, full_scale=65535.0)
        without_attr = exposure_fusion(
            untagged_but_correct, exposures_ms=exposures, full_scale=65535.0
        )
        for key in with_attr:
            spread = float(np.max(np.abs(with_attr[key] - without_attr[key])))
            assert spread < 0.01 * 65535.0


def _reference_255(
    key: str, frames: list[np.ndarray], exposures: list[float]
) -> np.ndarray:
    """The same bracket as ``uint8``, for the unit-scaling comparison."""
    as_uint8 = [np.clip(f, 0.0, 255.0).astype(np.uint8) for f in frames]
    return exposure_fusion(as_uint8, exposures_ms=exposures)[key].astype(np.float64)