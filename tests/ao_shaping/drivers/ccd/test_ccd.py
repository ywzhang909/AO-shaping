import os

import numpy as np
import matplotlib.pyplot as plt
import pytest

from ao_shaping.utils.image.spots_calc import centroid

# 硬件门控: 与 tests/ao_shaping/tools/slm/test_slm_diagnose_hardware.py 同款
# (README v0.12.0 约定)。未设 AO_RUN_HARDWARE=1 时整模块跳过 —— 无设备时
# Daheng/MII SDK open 失败抛 ConnectionAbortedError, 且避免误触相机。
if os.environ.get("AO_RUN_HARDWARE", "").strip().lower() not in {"1", "true", "yes"}:
    pytest.skip("set AO_RUN_HARDWARE=1 to run CCD hardware tests", allow_module_level=True)

gxipy = pytest.importorskip("gxipy")
from ao_shaping.drivers.ccd.daheng import DahengCamera
from ao_shaping.drivers.ccd.miicam import MIICamera


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
    with MIICamera(cam_id=cam_id, bit_depth=8) as cam:
        img = cam.get_numpy_image(n_sample=1, skip_first=False)
        assert img is not None
        assert img.size > 0
        assert img.dtype == np.uint8
        assert cam._bit_depth == 8


def test_cam_14bit_mode(cam_id=0):
    """Test camera initialization and capture in 14-bit mode."""
    with MIICamera(cam_id=cam_id, bit_depth=14) as cam:
        img = cam.get_numpy_image(n_sample=1, skip_first=False)
        assert img is not None
        assert img.size > 0
        assert img.dtype == np.uint16
        assert cam._bit_depth == 14


def test_cam_default_bit_depth(cam_id=0):
    """Test that default bit_depth is 8."""
    with MIICamera(cam_id=cam_id) as cam:
        assert cam._bit_depth == 8
        img = cam.get_numpy_image(n_sample=1, skip_first=False)
        assert img.dtype == np.uint8


def test_cam_bit_depth_initialization(cam_id=0):
    """Test that bit_depth is properly set during initialization."""
    for depth in [8, 14]:
        with MIICamera(cam_id=cam_id, bit_depth=depth) as cam:
            assert cam._bit_depth == depth
            if depth == 8:
                assert cam._pixel_format == "MONO8"
            else:
                # 14-bit 数据装在 16-bit 容器里, 故像素格式名是 MONO16 而非 MONO14
                # —— 本机硬件实测即报 MONO16 (见 drivers/ccd/AGENTS.md「已知坑」)。
                # 生效位深由 `_bit_depth` (设备回读 14) 承载, 不是容器名。
                assert cam._pixel_format == "MONO16"


def test_cam_8bit_14bit_intensity_comparison(cam_id=0):
    """Test that 14-bit mode captures higher bit depth data than 8-bit mode."""
    with MIICamera(cam_id=cam_id, exposure_time_ms=100, bit_depth=8) as cam:
        img_8bit = cam.get_numpy_image(n_sample=1, skip_first=False)
        assert img_8bit.dtype == np.uint8
        max_8bit = np.max(img_8bit)

    with MIICamera(cam_id=cam_id, exposure_time_ms=100, bit_depth=14) as cam:
        img_14bit = cam.get_numpy_image(n_sample=1, skip_first=False)
        assert img_14bit.dtype == np.uint16
        max_14bit = np.max(img_14bit)

    assert img_8bit.shape == img_14bit.shape
    if max_8bit < 255:
        assert max_14bit > max_8bit, (
            f"14-bit mode should capture higher values: 8bit max={max_8bit}, 14bit max={max_14bit}"
        )


def _saturated_fraction(img):
    """Fraction of pixels at the integer dtype's full scale (clipped)."""
    full = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 1.0
    return float(np.mean(img >= full))


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

    # 饱和免疫: 光斑核心在 50/500 ms 下可能削顶 (8-bit 满量程 255), 削顶区的
    # 均值不再线性。故: 方向断言必做; 比例断言只在两帧都未削顶时执行; 削顶比例显式记录。
    sat_50 = _saturated_fraction(img_50ms)
    sat_500 = _saturated_fraction(img_500ms)
    print(f"\nsaturated fraction: 50ms={sat_50:.4%}, 500ms={sat_500:.4%}")
    assert mean_500ms > mean_50ms, (
        f"500ms exposure ({mean_500ms:.2f}) should be brighter than 50ms ({mean_50ms:.2f})"
    )
    if sat_50 == 0 and sat_500 == 0:
        assert mean_500ms > mean_50ms * 2, (
            f"500ms exposure ({mean_500ms:.2f}) should be at least 2x brighter than 50ms ({mean_50ms:.2f})"
        )
        assert mean_500ms < mean_50ms * 20, (
            f"500ms exposure ({mean_500ms:.2f}) should not be more than 20x brighter than 50ms ({mean_50ms:.2f})"
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

    # 削顶时 imshow 只画白块: 打印削顶比例, 并让两帧共用按 50 ms 帧 p99.5 定的
    # 显示窗, 结构才能在同一量纲下比较 (核心若仍是平的, 那是真实削顶, 不是显示问题).
    sat_50 = _saturated_fraction(img_50ms)
    sat_500 = _saturated_fraction(img_500ms)
    print(f"Saturated fraction: 50ms={sat_50:.4%}, 500ms={sat_500:.4%}")
    display_max = max(1.0, float(np.percentile(img_50ms, 99.5)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_50ms, cmap="gray", vmin=0, vmax=display_max)
    axes[0].set_title(f"50ms exposure (mean={mean_50ms:.1f}, sat={sat_50:.2%})")
    axes[1].imshow(img_500ms, cmap="gray", vmin=0, vmax=display_max)
    axes[1].set_title(f"500ms exposure (mean={mean_500ms:.1f}, sat={sat_500:.2%})")
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
