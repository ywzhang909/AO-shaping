import matplotlib.pyplot as plt
import numpy as np
import pytest

from ao_shaping.utils.spots_calc import centroid

gxipy = pytest.importorskip("gxipy")
from ao_shaping.drivers import DahengCamera


def test_cam_list():
    cam_list = DahengCamera.get_cam_list()
    for cam in cam_list:
        print(cam)


def test_cam(cam_id=0):
    with DahengCamera(cam_id=cam_id) as cam:
        img = cam.get_numpy_image()
        assert np.sum(img) > 0


def test_cam_8bit_mode(cam_id=0):
    """Test camera initialization and capture in 8-bit mode."""
    with DahengCamera(cam_id=cam_id, bit_depth=8) as cam:
        img = cam.get_numpy_image(n_sample=1, skip_first=False)
        assert img is not None
        assert img.size > 0
        assert img.dtype == np.uint8
        assert cam._bit_depth == 8


def test_cam_14bit_mode(cam_id=0):
    """Test camera initialization and capture in 14-bit mode."""
    with DahengCamera(cam_id=cam_id, bit_depth=14) as cam:
        img = cam.get_numpy_image(n_sample=1, skip_first=False)
        assert img is not None
        assert img.size > 0
        assert img.dtype == np.uint16
        assert cam._bit_depth == 14


def test_cam_default_bit_depth(cam_id=0):
    """Test that default bit_depth is 8."""
    with DahengCamera(cam_id=cam_id) as cam:
        assert cam._bit_depth == 8
        img = cam.get_numpy_image(n_sample=1, skip_first=False)
        assert img.dtype == np.uint8


def test_cam_bit_depth_initialization(cam_id=0):
    """Test that bit_depth is properly set during initialization."""
    for depth in [8, 14]:
        with DahengCamera(cam_id=cam_id, bit_depth=depth) as cam:
            assert cam._bit_depth == depth
            if depth == 8:
                assert cam._pixel_format == "MONO8"
            else:
                assert cam._pixel_format == "MONO14"


def test_cam_8bit_14bit_intensity_comparison(cam_id=0):
    """Test that 14-bit mode captures higher bit depth data than 8-bit mode."""
    with DahengCamera(cam_id=cam_id, exposure_time_ms=100, bit_depth=8) as cam:
        img_8bit = cam.get_numpy_image(n_sample=1, skip_first=False)
        assert img_8bit.dtype == np.uint8
        max_8bit = np.max(img_8bit)

    with DahengCamera(cam_id=cam_id, exposure_time_ms=100, bit_depth=14) as cam:
        img_14bit = cam.get_numpy_image(n_sample=1, skip_first=False)
        assert img_14bit.dtype == np.uint16
        max_14bit = np.max(img_14bit)

    assert img_8bit.shape == img_14bit.shape
    if max_8bit < 255:
        assert max_14bit > max_8bit, (
            f"14-bit mode should capture higher values: 8bit max={max_8bit}, 14bit max={max_14bit}"
        )


def test_exposure_time_difference(cam_id=0):
    """
    Test that different exposure times produce different images.

    This test captures images at 50ms and 500ms exposure and verifies that:
    1. Both images can be captured successfully
    2. The images have significantly different average intensity
    3. The 500ms exposure image should be brighter (higher average intensity)

    Requirements:
    - Hardware: Daheng CCD camera
    - Expected: 500ms exposure should produce ~10x brighter image than 50ms
    """
    with DahengCamera(cam_id=cam_id, exposure_time_ms=50) as cam:
        img_50ms = cam.get_numpy_image(n_sample=1, skip_first=False)
        assert img_50ms is not None
        assert img_50ms.size > 0
        mean_50ms = np.mean(img_50ms)

    with DahengCamera(cam_id=cam_id, exposure_time_ms=500) as cam:
        img_500ms = cam.get_numpy_image(n_sample=1, skip_first=False)
        assert img_500ms is not None
        assert img_500ms.size > 0
        mean_500ms = np.mean(img_500ms)

    assert mean_500ms > mean_50ms * 2, (
        f"500ms exposure ({mean_500ms:.2f}) should be at least 2x brighter than 50ms ({mean_50ms:.2f})"
    )
    assert mean_500ms < mean_50ms * 20, (
        f"500ms exposure ({mean_500ms:.2f}) should not be more than 20x brighter than 50ms ({mean_50ms:.2f}) - possible saturation"
    )


def test_exposure_time_difference_manual(cam_id=0):
    """
    Manual test for exposure time difference - run this to verify camera behavior.

    This is a template for manual testing. Run with:
    pytest tests/ao_shaping/drivers/test_ccd.py::test_exposure_time_difference_manual -v -s
    """
    with DahengCamera(cam_id=cam_id, exposure_time_ms=50) as cam:
        img_50ms = cam.get_numpy_image(n_sample=1, skip_first=False)
        mean_50ms = np.mean(img_50ms)
        print(f"\n50ms exposure:  mean={mean_50ms:.2f}, shape={img_50ms.shape}")

    with DahengCamera(cam_id=cam_id, exposure_time_ms=500) as cam:
        img_500ms = cam.get_numpy_image(n_sample=1, skip_first=False)
        mean_500ms = np.mean(img_500ms)
        print(f"500ms exposure: mean={mean_500ms:.2f}, shape={img_500ms.shape}")

    ratio = mean_500ms / mean_50ms if mean_50ms > 0 else float("inf")
    print(f"Brightness ratio: {ratio:.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_50ms, cmap="gray")
    axes[0].set_title(f"50ms exposure (mean={mean_50ms:.1f})")
    axes[1].imshow(img_500ms, cmap="gray")
    axes[1].set_title(f"500ms exposure (mean={mean_500ms:.1f})")
    plt.tight_layout()
    plt.show()


def run_exposure_comparison():
    """
    Run this function manually to compare 50ms vs 500ms exposure.

    Usage:
    1. Connect CCD camera
    2. Run: python -c "from tests.ao_shaping.drivers.test_ccd import run_exposure_comparison; run_exposure_comparison()"
    """
    cam_id = 0

    # Capture at 50ms exposure
    with DahengCamera(cam_id=cam_id, exposure_time_ms=50) as cam:
        img_50ms = cam.get_numpy_image(n_sample=1, skip_first=False)

    # Capture at 500ms exposure
    with DahengCamera(cam_id=cam_id, exposure_time_ms=500) as cam:
        img_500ms = cam.get_numpy_image(n_sample=1, skip_first=False)

    # Calculate statistics
    mean_50ms = np.mean(img_50ms)
    mean_500ms = np.mean(img_500ms)
    std_50ms = np.std(img_50ms)
    std_500ms = np.std(img_500ms)

    print(
        f"50ms exposure:  mean={mean_50ms:.2f}, std={std_50ms:.2f}, shape={img_50ms.shape}"
    )
    print(
        f"500ms exposure: mean={mean_500ms:.2f}, std={std_500ms:.2f}, shape={img_500ms.shape}"
    )
    print(f"Brightness ratio: {mean_500ms / mean_50ms:.2f}x")

    # Verify that 500ms is significantly brighter
    assert mean_500ms > mean_50ms * 2, "500ms should be at least 2x brighter than 50ms"
    assert mean_500ms < mean_50ms * 20, (
        "500ms should not be more than 20x brighter (saturation)"
    )

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_50ms, cmap="gray")
    axes[0].set_title(f"50ms exposure (mean={mean_50ms:.1f})")
    axes[1].imshow(img_500ms, cmap="gray")
    axes[1].set_title(f"500ms exposure (mean={mean_500ms:.1f})")
    plt.tight_layout()
    plt.show()

    print("Test passed! Exposure times produce different images as expected.")


def test_reset_exposure_time_brightness(cam_id=0):
    """
    Verify reset_exposure_time correctness by comparing image brightness.

    Strategy (within 3 ms, large difference):
    1. Open the camera and query the hardware exposure range (ms).
    2. Pick a short exposure (>= e_min, >= 0.5 ms) and a target exposure
       (<= 3 ms) that differ by at least 2x.
    3. Reset to the short exposure, capture, and record max brightness.
       Skip if already saturated (max >= 255) — no room to grow.
    4. Reset to the target exposure, capture, and record max brightness.
       It must be strictly brighter.
    5. Reset back to the short exposure, capture again, and verify
       brightness drops again (round-trip consistency).

    The SDK quantizes exposure to integer microseconds, so returned values
    are checked with pytest.approx(abs=0.01) (10 us resolution).
    """
    if not DahengCamera.get_cam_list():
        pytest.skip("No Daheng camera connected")

    with DahengCamera(cam_id=cam_id) as cam:
        e_min, e_max = cam.get_exposure_range()
        target_ms = min(3.0, float(e_max))
        short_ms = max(float(e_min), 0.5)

        if target_ms <= short_ms or target_ms / short_ms < 2.0:
            pytest.skip(
                f"Exposure range {e_min:.3f}-{e_max:.3f} ms "
                "cannot produce a <=3ms >=2x comparison"
            )

        actual_short = cam.reset_exposure_time(short_ms)
        img_short = cam.get_numpy_image(n_sample=8, skip_first=True)
        # Mean is the robust metric: a single hot pixel or the global max
        # of a dim, noisy scene is too unstable (observed 48→60→119 across
        # three captures). The image mean averages over all pixels and
        # tracks exposure linearly in the unsaturated regime.
        b_short = float(np.mean(img_short))
        print(
            f"\n[{short_ms} ms] mean brightness = {b_short:.2f}, "
            f"max = {float(np.max(img_short))}"
        )

        if float(np.max(img_short)) >= 255:
            pytest.skip(
                f"Short exposure {short_ms} ms already saturated; "
                "cannot observe brightness increase"
            )

        actual_target = cam.reset_exposure_time(target_ms)
        img_target = cam.get_numpy_image(n_sample=8, skip_first=True)
        b_target = float(np.mean(img_target))
        print(
            f"[{target_ms} ms] mean brightness = {b_target:.2f}, "
            f"max = {float(np.max(img_target))}"
        )

        actual_short2 = cam.reset_exposure_time(short_ms)
        img_short2 = cam.get_numpy_image(n_sample=8, skip_first=True)
        b_short2 = float(np.mean(img_short2))
        print(
            f"[{short_ms} ms again] mean brightness = {b_short2:.2f}, "
            f"max = {float(np.max(img_short2))}"
        )

    assert actual_short == pytest.approx(short_ms, abs=0.01)
    assert actual_target == pytest.approx(target_ms, abs=0.01)
    assert actual_short2 == pytest.approx(short_ms, abs=0.01)
    assert b_target > b_short, (
        f"Longer exposure ({target_ms} ms, mean={b_target:.2f}) should be "
        f"brighter than shorter exposure ({short_ms} ms, mean={b_short:.2f})"
    )
    # With a >=2x exposure ratio the mean must grow by at least 1.5x
    # (linear sensor response; 1.5x leaves margin for noise/drift).
    # Guards against a degenerate pass on a near-black frame where both
    # means sit at ~0 and any single hot pixel flips the comparison.
    assert b_target >= 1.5 * b_short, (
        f"Mean should grow by >=1.5x for a >=2x exposure ratio; "
        f"got {b_short:.2f} -> {b_target:.2f}"
    )
    assert b_short2 < b_target, (
        f"Resetting back to {short_ms} ms should dim again "
        f"(mean={b_short2:.2f} vs {b_target:.2f})"
    )


def test_exposure_sweep_x2_max_brightness(cam_id=0):
    """
    Verify exposure -> max-brightness monotonicity with a doubling sweep.

    Strategy:
    1. Open the camera and query the hardware exposure range (ms).
    2. Prime the camera: reset to the minimum exposure and discard one
       full acquisition, so the sweep is not contaminated by frames
       captured at the camera's previous (unknown) exposure.
    3. Starting from the minimum exposure, double it at every step
       (via reset_exposure_time) up to the hardware maximum.
    4. At every step capture an 8-frame averaged image (first frame
       skipped, as reset_exposure_time needs a few frames to settle)
       and record the peak brightness (np.max).
    5. Stop the sweep as soon as the peak reaches the 8-bit saturation
       ceiling (255).
    6. Asserts:
       - each requested exposure was applied within the SDK's 10 us
         quantization;
       - from the first step whose peak is above the noise floor
         (NOISE_FLOOR grey levels), the peak never drops when the
         exposure grows;
       - the sweep showed a real >=2x peak increase (linear sensor
         response: each doubling roughly doubles the peak until
         saturation).

    Requires a connected Daheng camera (skips otherwise).
    """
    if not DahengCamera.get_cam_list():
        pytest.skip("No Daheng camera connected")

    NOISE_FLOOR = 8.0  # peaks below this are dominated by sensor noise, not signal

    records: list[tuple[float, float, float]] = []  # (requested ms, actual ms, peak)

    with DahengCamera(cam_id=cam_id) as cam:
        e_min, e_max = cam.get_exposure_range()
        assert e_max > e_min > 0, f"Bad exposure range: {e_min}-{e_max} ms"

        # Prime: flush frames taken at the camera's previous exposure.
        cam.reset_exposure_time(e_min)
        cam.get_numpy_image(n_sample=8)

        e = float(e_min)
        while True:
            actual = cam.reset_exposure_time(e)
            img = cam.get_numpy_image(n_sample=8, skip_first=True)
            peak = float(np.max(img))
            print(
                f"\n[exposure {e:.4g} ms, actual {actual:.4g}] "
                f"max brightness = {peak:.0f}"
            )
            records.append((e, actual, peak))
            if peak >= 255.0:
                print("Peak saturated at 255 - stopping the sweep.")
                break
            if e >= e_max:
                break  # reached the hardware maximum without saturating
            e = min(e * 2.0, e_max)

    if len(records) == 1 and records[0][2] >= 255.0:
        pytest.skip("Peak already saturated at the minimum exposure; cannot observe growth")
    assert len(records) >= 2, "Sweep produced only one step; cannot verify brightness growth"

    # 1) The SDK quantizes exposure to integer microseconds (10 us steps).
    for requested, actual, _ in records:
        assert actual == pytest.approx(requested, abs=0.01), (
            f"Requested {requested} ms but camera reports {actual} ms"
        )

    peaks = [r[2] for r in records]
    # 2) Peak brightness must not drop once the signal is measurable
    #    (below NOISE_FLOOR the peaks are random noise spikes).
    above_floor = [i for i, p in enumerate(peaks) if p >= NOISE_FLOOR]
    assert above_floor, (
        f"Peak never reached the noise floor ({NOISE_FLOOR:.0f} grey) up to "
        f"{records[-1][0]:.4g} ms; scene too dim to verify growth. "
        f"Peaks: {[f'{p:.0f}' for p in peaks]}"
    )
    for i in range(1, len(above_floor)):
        a, b = above_floor[i - 1], above_floor[i]
        assert peaks[b] >= peaks[a] - 1, (
            f"Max brightness should not drop when exposure grows: "
            f"{records[a][0]:.4g} ms -> {records[b][0]:.4g} ms, "
            f"peak {peaks[a]:.0f} -> {peaks[b]:.0f}"
        )

    # 3) The sweep must show a real increase (linear sensor response:
    #    every doubling roughly doubles the peak until saturation).
    first, last = peaks[above_floor[0]], peaks[-1]
    assert last > first, (
        f"Max brightness should increase with exposure: {first:.0f} -> {last:.0f}"
    )
    assert last >= 2 * first, (
        f"Expect >=2x max-brightness growth across the doubling sweep: "
        f"{first:.0f} -> {last:.0f}"
    )
