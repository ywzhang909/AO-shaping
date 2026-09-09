"""MIICAM CCD camera hardware test with visual report generation.

Tests MIICAM camera driver functionality and generates a markdown report with plots
saved to docs/miicam/miicam_report.md

Run with:
    pytest tests/ao_shaping/drivers/ccd/test_miicam_report.py -v --hardware -s
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tests.ao_shaping.utils.test_report import TestReport, TestWithReport

# Import MIICAM module
try:
    from ao_shaping.drivers.ccd.miicam import driver as miicam_driver

    MIICAM_AVAILABLE = True
except ImportError:
    MIICAM_AVAILABLE = False
    miicam_driver = None


class TestMIICAMReport:
    """MIICAM camera hardware tests with visual report."""

    pytestmark = pytest.mark.hardware

    @pytest.fixture
    def camera_module(self):
        """Import and return MIICAM module."""
        if not MIICAM_AVAILABLE:
            pytest.skip("MIICAM module not available")
        return miicam_driver

    @pytest.fixture
    def CameraStreamManager(self, camera_module):
        """Return CameraStreamManager class."""
        return camera_module.CameraStreamManager

    @pytest.fixture
    def camera(self, CameraStreamManager):
        """Create and return camera instance."""
        cam = CameraStreamManager(cam_id=0, exposure_time_ms=20)
        cam.open()
        yield cam
        cam.close()

    def test_miicam_basic_report(self, CameraStreamManager, camera):
        """Test MIICAM basic functionality and generate report."""
        with TestReport("miicam") as report:
            report.add_section("MIICAM 4100 Series Camera Test Report", 2)
            report.add_text("Testing MIICAM camera driver functionality with hardware.")

            # Test 1: Camera enumeration
            with TestWithReport(report, "Camera Enumeration"):
                cam_list = CameraStreamManager.get_cam_list()
                report.add_key_value("Cameras Found", str(len(cam_list)))
                for i, cam_info in enumerate(cam_list):
                    report.add_key_value(f"Camera {i}", str(cam_info))

            # Test 2: Device info
            with TestWithReport(report, "Device Information"):
                info = {
                    "Camera ID": camera.cam_id,
                    "Width": camera.cam_width,
                    "Height": camera.cam_height,
                    "Bit Depth": camera._bit_depth,
                    "Serial Number": getattr(camera, "_sn", "Unknown"),
                    "Exposure Time": f"{camera.exposure_time_ms} ms",
                    "Max Exposure": f"{camera.max_exposure_ms} ms",
                    "Min Exposure": f"{camera.min_exposure_ms} ms",
                }
                report.add_table(
                    ["Parameter", "Value"],
                    [[k, v] for k, v in info.items()],
                )

            # Test 3: Exposure time control
            with TestWithReport(report, "Exposure Time Control"):
                test_exposures = [0.5, 1, 5, 10, 20, 50, 100, 500, 1000, 5000]
                exp_results = []
                for exp in test_exposures:
                    start = time.perf_counter()
                    actual = camera.reset_exposure_time(exp)
                    elapsed = time.perf_counter() - start
                    exp_results.append(
                        [f"{exp} ms", f"{actual:.3f} ms", f"{elapsed * 1000:.1f} ms"]
                    )

                report.add_table(
                    ["Requested", "Actual", "Set Time"],
                    exp_results,
                )

                # Plot exposure linearity
                fig, ax = plt.subplots(figsize=(8, 4))
                requested = [r[0] for r in exp_results]
                actual_vals = [float(r[1].split()[0]) for r in exp_results]
                ax.plot(requested, actual_vals, "o-")
                ax.plot(requested, requested, "k--", label="Ideal")
                ax.set_xlabel("Requested Exposure (ms)")
                ax.set_ylabel("Actual Exposure (ms)")
                ax.set_title("Exposure Time Linearity")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                report.add_plot(
                    fig, "exposure_linearity", "Exposure time linearity test"
                )

            # Test 4: Image capture
            with TestWithReport(report, "Image Capture"):
                # Single frame
                img = camera.get_numpy_image(n_sample=1, skip_first=False)
                report.add_key_value("Single Frame Shape", str(img.shape))
                report.add_key_value("Single Frame Dtype", str(img.dtype))
                report.add_key_value("Min/Max", f"{img.min()} / {img.max()}")
                report.add_key_value("Mean/Std", f"{img.mean():.1f} / {img.std():.1f}")

                # Averaged frames
                img_avg = camera.get_numpy_image(n_sample=10, skip_first=True)
                report.add_key_value("Averaged (10) Shape", str(img_avg.shape))
                report.add_key_value(
                    "Averaged Min/Max", f"{img_avg.min()} / {img_avg.max()}"
                )
                report.add_key_value(
                    "Averaged Mean/Std", f"{img_avg.mean():.1f} / {img_avg.std():.1f}"
                )

                # Display images
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(
                    img,
                    cmap="gray",
                    vmin=0,
                    vmax=255 if img.dtype == np.uint8 else 65535,
                )
                axes[0].set_title("Single Frame")
                axes[0].axis("off")

                axes[1].imshow(
                    img_avg,
                    cmap="gray",
                    vmin=0,
                    vmax=255 if img_avg.dtype == np.uint8 else 65535,
                )
                axes[1].set_title("Averaged (10 frames)")
                axes[1].axis("off")

                # Histogram
                axes[2].hist(img.flatten(), bins=50, alpha=0.7, label="Single")
                axes[2].hist(img_avg.flatten(), bins=50, alpha=0.7, label="Averaged")
                axes[2].set_title("Intensity Histogram")
                axes[2].set_xlabel("Pixel Value")
                axes[2].set_ylabel("Count")
                axes[2].legend()
                axes[2].grid(True)
                plt.tight_layout()
                report.add_plot(fig, "image_capture", "Image capture comparison")

            # Test 5: ROI/Window control
            with TestWithReport(report, "ROI / Window Control"):
                full_w, full_h = camera.cam_width, camera.cam_height

                rois = [
                    ((0, 0), (0, 0), "Full Frame"),
                    (
                        (full_w // 2, full_h // 2),
                        (full_w // 2, full_h // 2),
                        "Center Half",
                    ),
                    (
                        (full_w // 2, full_h // 2),
                        (full_w // 4, full_h // 4),
                        "Center Quarter",
                    ),
                ]

                roi_results = []
                fig, axes = plt.subplots(1, len(rois), figsize=(4 * len(rois), 4))
                if len(rois) == 1:
                    axes = [axes]

                for idx, (center, size, label) in enumerate(rois):
                    cam_size, cam_center = camera.reset_window(center=center, size=size)
                    time.sleep(0.1)
                    img = camera.get_numpy_image(n_sample=1, skip_first=False)
                    roi_results.append(
                        [label, str(cam_size), str(cam_center), str(img.shape)]
                    )

                    axes[idx].imshow(
                        img,
                        cmap="gray",
                        vmin=0,
                        vmax=255 if img.dtype == np.uint8 else 65535,
                    )
                    axes[idx].set_title(f"{label}\n{img.shape}")
                    axes[idx].axis("off")

                plt.tight_layout()
                report.add_plot(fig, "roi_test", "ROI window test")

                report.add_table(
                    ["ROI", "Size", "Center", "Image Shape"],
                    roi_results,
                )

                # Reset to full frame
                camera.reset_window(center=(0, 0), size=(0, 0))

            # Test 6: Auto exposure
            with TestWithReport(report, "Auto Exposure"):
                if hasattr(camera, "enable_auto_exposure"):
                    # Test enable/disable
                    camera.enable_auto_exposure(True)
                    time.sleep(0.5)
                    state = camera.get_auto_exposure_state()
                    report.add_key_value(
                        "Auto Exp Enabled", str(state.get("enabled", "Unknown"))
                    )
                    report.add_key_value("Mode", str(state.get("mode", "Unknown")))

                    # Capture with auto exposure
                    img_auto = camera.get_numpy_image(n_sample=5, skip_first=True)
                    report.add_key_value("Auto Exp Mean", f"{img_auto.mean():.1f}")
                    report.add_key_value("Auto Exp Max", f"{img_auto.max()}")

                    # Test target setting
                    for target in [100, 150, 200]:
                        actual = camera.set_auto_exposure_target(target)
                        report.add_key_value(f"Target {target}", f"Actual {actual}")

                    # Disable
                    camera.enable_auto_exposure(False)
                    state = camera.get_auto_exposure_state()
                    report.add_key_value(
                        "Auto Exp Disabled", str(state.get("enabled", "Unknown"))
                    )

            # Test 7: Bit depth modes
            with TestWithReport(report, "Bit Depth Modes"):
                for bit_depth in [8, 16]:
                    if bit_depth == 8:
                        cam_test = CameraStreamManager(
                            cam_id=0, exposure_time_ms=20, bit_depth=8
                        )
                    else:
                        cam_test = CameraStreamManager(
                            cam_id=0, exposure_time_ms=20, bit_depth=16
                        )

                    cam_test.open()
                    img = cam_test.get_numpy_image(n_sample=1, skip_first=False)
                    report.add_key_value(f"{bit_depth}-bit Shape", str(img.shape))
                    report.add_key_value(f"{bit_depth}-bit Dtype", str(img.dtype))
                    report.add_key_value(
                        f"{bit_depth}-bit Range", f"{img.min()} - {img.max()}"
                    )
                    cam_test.close()

            # Test 8: Frame rate / timing
            with TestWithReport(report, "Frame Timing"):
                n_frames = 30
                start = time.perf_counter()
                frames = []
                for _ in range(n_frames):
                    frame_start = time.perf_counter()
                    img = camera.get_numpy_image(n_sample=1, skip_first=False)
                    frame_end = time.perf_counter()
                    frames.append(frame_end - frame_start)
                total_time = time.perf_counter() - start

                fps = n_frames / total_time
                avg_frame_time = np.mean(frames) * 1000
                min_frame_time = np.min(frames) * 1000
                max_frame_time = np.max(frames) * 1000

                report.add_key_value("Frames Captured", str(n_frames))
                report.add_key_value("Total Time", f"{total_time:.2f} s")
                report.add_key_value("Effective FPS", f"{fps:.1f}")
                report.add_key_value("Avg Frame Time", f"{avg_frame_time:.1f} ms")
                report.add_key_value("Min Frame Time", f"{min_frame_time:.1f} ms")
                report.add_key_value("Max Frame Time", f"{max_frame_time:.1f} ms")

                # Plot frame timing
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(np.arange(n_frames), np.array(frames) * 1000, "o-")
                ax.axhline(
                    float(avg_frame_time),
                    color="r",
                    linestyle="--",
                    label=f"Mean: {avg_frame_time:.1f} ms",
                )
                ax.set_xlabel("Frame Number")
                ax.set_ylabel("Frame Time (ms)")
                ax.set_title("Frame Capture Timing")
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                report.add_plot(fig, "frame_timing", "Frame capture timing analysis")

            # Test 9: Noise analysis
            with TestWithReport(report, "Noise Analysis"):
                # Capture dark frames (with lens cap)
                dark_frames = []
                for _ in range(20):
                    img = camera.get_numpy_image(n_sample=1, skip_first=False)
                    dark_frames.append(img.astype(np.float32))

                dark_stack = np.stack(dark_frames)
                temporal_mean = np.mean(dark_stack, axis=0)
                temporal_std = np.std(dark_stack, axis=0)

                report.add_key_value(
                    "Temporal Mean (dark)", f"{np.mean(temporal_mean):.2f}"
                )
                report.add_key_value(
                    "Temporal Std (dark)", f"{np.mean(temporal_std):.2f}"
                )
                report.add_key_value("Max Temporal Std", f"{np.max(temporal_std):.2f}")

                # Spatial noise
                spatial_mean = np.mean(img.astype(np.float32))
                spatial_std = np.std(img.astype(np.float32))
                report.add_key_value("Spatial Mean (lit)", f"{spatial_mean:.2f}")
                report.add_key_value("Spatial Std (lit)", f"{spatial_std:.2f}")

                # Noise visualization
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                im0 = axes[0].imshow(temporal_mean, cmap="hot")
                axes[0].set_title("Temporal Mean")
                axes[0].axis("off")
                plt.colorbar(im0, ax=axes[0], fraction=0.046)

                im1 = axes[1].imshow(temporal_std, cmap="hot")
                axes[1].set_title("Temporal Std (Noise)")
                axes[1].axis("off")
                plt.colorbar(im1, ax=axes[1], fraction=0.046)

                axes[2].hist(temporal_std.flatten(), bins=50, edgecolor="black")
                axes[2].set_title("Temporal Noise Distribution")
                axes[2].set_xlabel("Std Dev")
                axes[2].set_ylabel("Count")
                axes[2].grid(True)
                plt.tight_layout()
                report.add_plot(fig, "noise_analysis", "Camera noise analysis")

            # Summary
            report.add_section("Summary", 2)
            report.add_text("MIICAM camera hardware tests completed.")
            report.write_summary_table()
            report.save()
            print(f"\nReport saved to: {report.report_path}")

    def test_miicam_slm_interaction_report(self, CameraStreamManager):
        """Test SLM + Camera interaction (requires SLM)."""
        pytest.skip("Requires both SLM and camera hardware")

        # This would test:
        # 1. SLM pattern display + camera capture
        # 2. Diffraction order measurement
        # 3. Beam shaping verification
        pass


def test_miicam_standalone_report():
    """Standalone test that can run without pytest (simulation mode)."""
    with TestReport("miicam") as report:
        report.add_section("MIICAM Simulation Test", 2)
        report.add_text("This test runs without hardware using simulated data.")

        # Simulate camera images
        with TestWithReport(report, "Simulated Image Generation"):
            # Create synthetic images
            height, width = 1024, 1280

            # Gaussian beam profile
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2
            sigma = 100
            beam = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma**2))
            beam_8bit = (beam * 255).astype(np.uint8)
            beam_16bit = (beam * 65535).astype(np.uint16)

            # Add noise
            noise_8bit = beam_8bit + np.random.normal(0, 5, beam_8bit.shape).astype(
                np.uint8
            )
            noise_16bit = beam_16bit + np.random.normal(0, 50, beam_16bit.shape).astype(
                np.uint16
            )

            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes[0, 0].imshow(beam_8bit, cmap="gray")
            axes[0, 0].set_title("Ideal Beam (8-bit)")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(noise_8bit, cmap="gray")
            axes[0, 1].set_title("With Noise (8-bit)")
            axes[0, 1].axis("off")

            axes[0, 2].hist(noise_8bit.flatten(), bins=50)
            axes[0, 2].set_title("8-bit Histogram")
            axes[0, 2].grid(True)

            axes[1, 0].imshow(beam_16bit, cmap="gray")
            axes[1, 0].set_title("Ideal Beam (16-bit)")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(noise_16bit, cmap="gray")
            axes[1, 1].set_title("With Noise (16-bit)")
            axes[1, 1].axis("off")

            axes[1, 2].hist(noise_16bit.flatten(), bins=50)
            axes[1, 2].set_title("16-bit Histogram")
            axes[1, 2].grid(True)

            plt.tight_layout()
            report.add_plot(fig, "simulated_images", "Simulated camera images")

        report.save()
        print(f"Report saved to: {report.report_path}")


if __name__ == "__main__":
    test_miicam_standalone_report()
