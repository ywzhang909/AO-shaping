import numpy as np
import matplotlib.pyplot as plt
import pytest

from ao_shaping.utils.spots_calc import centroid
from ao_shaping.drivers import CameraStreamManager


def test_cam_list():
    cam_list = CameraStreamManager.get_cam_list()
    for cam in cam_list:
        print(cam)

def test_cam(cam_id=0):
    with CameraStreamManager(id=cam_id) as cam:
        assert cam.get_numpy_image()

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
    pytest.skip("Requires CCD camera hardware")


def test_exposure_time_difference_manual(cam_id=0):
    """
    Manual test for exposure time difference - run this to verify camera behavior.

    This is a template for manual testing. Run with:
    pytest tests/ao_shaping/drivers/test_ccd.py::test_exposure_time_difference_manual -v -s
    """
    pytest.skip("Requires CCD camera hardware - run manually with hardware connected")


def run_exposure_comparison():
    """
    Run this function manually to compare 50ms vs 500ms exposure.

    Usage:
    1. Connect CCD camera
    2. Run: python -c "from tests.ao_shaping.drivers.test_ccd import run_exposure_comparison; run_exposure_comparison()"
    """
    cam_id = 0

    # Capture at 50ms exposure
    with CameraStreamManager(cam_id=cam_id, exposure_time_ms=50) as cam:
        img_50ms = cam.get_numpy_image(n_sample=1, skip_first=False)

    # Capture at 500ms exposure
    with CameraStreamManager(cam_id=cam_id, exposure_time_ms=500) as cam:
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
