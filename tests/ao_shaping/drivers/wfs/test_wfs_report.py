"""Thorlabs WFS hardware test with visual report generation.

Tests Thorlabs Wavefront Sensor driver functionality and generates a markdown report
with plots saved to docs/wfs/wfs_report.md

Run with:
    pytest tests/ao_shaping/drivers/wfs/test_wfs_report.py -v --hardware -s
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tests.ao_shaping.utils.test_report import TestReport, TestWithReport

# Import WFS module
try:
    from ao_shaping.drivers.wfs.thorlab import driver as wfs_driver

    WFS_AVAILABLE = True
except ImportError:
    WFS_AVAILABLE = False
    wfs_driver = None

pytestmark = pytest.mark.hardware


class TestWFSReport:
    """Thorlabs WFS hardware tests with visual report."""

    @pytest.fixture
    def wfs_module(self):
        """Import and return WFS module."""
        if not WFS_AVAILABLE:
            pytest.skip("WFS module not available")
        return wfs_driver

    @pytest.fixture
    def ThorlabWFS(self, wfs_module):
        """Return ThorlabWFS class."""
        return wfs_module.ThorlabWFS

    @pytest.fixture
    def wfs(self, ThorlabWFS):
        """Create and return WFS instance."""
        # "512" or "512H" for high speed
        wfs = ThorlabWFS("512", high_speed=False)
        wfs.open()
        yield wfs
        wfs.close()

    def test_wfs_basic_report(self, ThorlabWFS, wfs):
        """Test WFS basic functionality and generate report."""
        with TestReport("wfs") as report:
            report.add_section("Thorlabs WFS Test Report", 2)
            report.add_text(
                "Testing Thorlabs Wavefront Sensor driver functionality with hardware."
            )

            # Test 1: Device info
            with TestWithReport(report, "Device Information"):
                info = {
                    "Camera Type": getattr(wfs, "cam_type", "Unknown"),
                    "High Speed": getattr(wfs, "high_speed", False),
                    "Resolution": f"{getattr(wfs, 'width', '?')}x{getattr(wfs, 'height', '?')}",
                    "Pixel Size": f"{getattr(wfs, 'pixel_size', '?')} µm",
                    "Lenslet Pitch": f"{getattr(wfs, 'lenslet_pitch', '?')} µm",
                    "Focal Length": f"{getattr(wfs, 'focal_length', '?')} mm",
                    "NA": getattr(wfs, "NA", "?"),
                }
                report.add_table(
                    ["Parameter", "Value"],
                    [[k, v] for k, v in info.items()],
                )

            # Test 2: Exposure control
            with TestWithReport(report, "Exposure Control"):
                if hasattr(wfs, "get_exposure_time_range"):
                    exp_range = wfs.get_exposure_time_range()
                    report.add_key_value(
                        "Exposure Range", f"{exp_range[0]} - {exp_range[1]} ms"
                    )

                test_exposures = [0.1, 0.5, 1, 2, 5, 10, 20]
                exp_results = []
                for exp in test_exposures:
                    try:
                        wfs.set_exposure_time(exp)
                        time.sleep(0.1)
                        actual = wfs.get_exposure_time()
                        exp_results.append([f"{exp} ms", f"{actual:.3f} ms", "OK"])
                    except Exception as e:
                        exp_results.append([f"{exp} ms", "FAIL", str(e)])

                report.add_table(
                    ["Requested", "Actual", "Status"],
                    exp_results,
                )

            # Test 3: Spotfield image
            with TestWithReport(report, "Spotfield Image"):
                img = wfs.get_spotfiled_image()
                report.add_key_value("Shape", str(img.shape))
                report.add_key_value("Dtype", str(img.dtype))
                report.add_key_value("Min/Max", f"{img.min()} / {img.max()}")
                report.add_key_value("Mean/Std", f"{img.mean():.1f} / {img.std():.1f}")

                # Display spotfield
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].imshow(img, cmap="gray")
                axes[0].set_title("Spotfield Image")
                axes[0].axis("off")

                axes[1].hist(img.flatten(), bins=50, edgecolor="black")
                axes[1].set_title("Intensity Histogram")
                axes[1].set_xlabel("Pixel Value")
                axes[1].set_ylabel("Count")
                axes[1].grid(True)
                plt.tight_layout()
                report.add_plot(fig, "spotfield", "Spotfield image and histogram")

            # Test 4: Spot detection and centroids
            with TestWithReport(report, "Spot Detection & Centroids"):
                spots = wfs.get_spots_statics()
                report.add_key_value(
                    "Number of Spots", str(len(spots)) if spots is not None else "0"
                )

                if spots is not None and len(spots) > 0:
                    # Expect spots to have x, y, intensity columns
                    if hasattr(spots, "shape"):
                        report.add_key_value("Spots Array Shape", str(spots.shape))

                    # Visualize spot positions
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

                    # Spotfield with centroids overlaid
                    axes[0].imshow(img, cmap="gray", alpha=0.5)
                    if spots.shape[1] >= 2:
                        axes[0].scatter(
                            spots[:, 0], spots[:, 1], c="r", s=10, alpha=0.7
                        )
                    axes[0].set_title("Detected Spots")
                    axes[0].axis("off")

                    # Spot intensity distribution
                    if spots.shape[1] >= 3:
                        axes[1].hist(spots[:, 2], bins=30, edgecolor="black")
                        axes[1].set_title("Spot Intensity Distribution")
                        axes[1].set_xlabel("Intensity")
                        axes[1].set_ylabel("Count")
                        axes[1].grid(True)
                    plt.tight_layout()
                    report.add_plot(fig, "spot_detection", "Spot detection results")

            # Test 5: Zernike measurement
            with TestWithReport(report, "Zernike Measurement"):
                # Get reference
                wfs.save_user_ref()
                wfs.set_ref_plane(custom=True)
                time.sleep(0.2)

                # Measure Zernike
                zernike = wfs.get_zernike()
                report.add_key_value(
                    "Zernike Modes", str(len(zernike)) if zernike is not None else "0"
                )

                if zernike is not None and len(zernike) > 0:
                    # First few modes are typically piston, tip, tilt - skip them
                    modes_to_plot = min(20, len(zernike) - 3)
                    if modes_to_plot > 0:
                        zernike_vals = zernike[3 : 3 + modes_to_plot]

                        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                        axes[0].bar(range(3, 3 + modes_to_plot), zernike_vals)
                        axes[0].set_xlabel("Zernike Index (Noll)")
                        axes[0].set_ylabel("Coefficient (waves)")
                        axes[0].set_title("Zernike Coefficients (modes 3+)")
                        axes[0].grid(True)

                        # RMS
                        rms = np.sqrt(np.mean(zernike_vals**2))
                        axes[1].text(
                            0.5,
                            0.5,
                            f"RMS = {rms:.4f} waves",
                            fontsize=24,
                            ha="center",
                            va="center",
                        )
                        axes[1].axis("off")
                        axes[1].set_title("Wavefront RMS")
                        plt.tight_layout()
                        report.add_plot(
                            fig, "zernike_measurement", "Zernike wavefront measurement"
                        )

                        report.add_key_value("RMS (modes 3+)", f"{rms:.4f} waves")

            # Test 6: Reference plane comparison
            with TestWithReport(report, "Reference Plane Comparison"):
                # Custom reference
                wfs.save_user_ref()
                wfs.set_ref_plane(custom=True)
                time.sleep(0.1)
                zernike_custom = wfs.get_zernike()

                # Factory reference
                wfs.set_ref_plane(custom=False)
                time.sleep(0.1)
                zernike_factory = wfs.get_zernike()

                if zernike_custom is not None and zernike_factory is not None:
                    diff = zernike_custom - zernike_factory

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].bar(range(len(zernike_custom)), zernike_custom)
                    axes[0].set_title("Custom Reference")
                    axes[0].set_xlabel("Mode")
                    axes[0].set_ylabel("Coefficient")
                    axes[0].grid(True)

                    axes[1].bar(range(len(zernike_factory)), zernike_factory)
                    axes[1].set_title("Factory Reference")
                    axes[1].set_xlabel("Mode")
                    axes[1].grid(True)

                    axes[2].bar(range(len(diff)), diff)
                    axes[2].set_title("Difference (Custom - Factory)")
                    axes[2].set_xlabel("Mode")
                    axes[2].grid(True)
                    plt.tight_layout()
                    report.add_plot(fig, "ref_comparison", "Reference plane comparison")

            # Test 7: Spot deviation measurement
            with TestWithReport(report, "Spot Deviation Measurement"):
                if hasattr(wfs, "get_spot_deviation"):
                    wfs.save_user_ref()
                    wfs.set_ref_plane(custom=True)
                    time.sleep(0.1)

                    dx, dy = wfs.get_spot_deviation(cancel_tile=True)
                    report.add_key_value("Deviation Shape", f"{dx.shape} / {dy.shape}")
                    report.add_key_value("DX Range", f"{dx.min():.2f} - {dx.max():.2f}")
                    report.add_key_value("DY Range", f"{dy.min():.2f} - {dy.max():.2f}")
                    report.add_key_value("DX RMS", f"{np.sqrt(np.mean(dx**2)):.4f}")
                    report.add_key_value("DY RMS", f"{np.sqrt(np.mean(dy**2)):.4f}")

                    # Vector field plot
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    # Magnitude
                    mag = np.sqrt(dx**2 + dy**2)
                    im0 = axes[0].imshow(mag, cmap="hot")
                    axes[0].set_title("Deviation Magnitude")
                    axes[0].axis("off")
                    plt.colorbar(im0, ax=axes[0], fraction=0.046)

                    # Quiver plot (subsampled)
                    step = max(1, dx.shape[0] // 20)
                    y_idx, x_idx = np.meshgrid(
                        np.arange(0, dy.shape[0], step),
                        np.arange(0, dx.shape[1], step),
                        indexing="ij",
                    )
                    axes[1].quiver(
                        x_idx, y_idx, dx[y_idx, x_idx], dy[y_idx, x_idx], color="b"
                    )
                    axes[1].set_title("Deviation Vector Field")
                    axes[1].set_aspect("equal")
                    axes[1].invert_yaxis()
                    plt.tight_layout()
                    report.add_plot(
                        fig, "spot_deviation", "Spot deviation vector field"
                    )

            # Test 8: Pupil optimization
            with TestWithReport(report, "Pupil Optimization"):
                if hasattr(wfs, "optimize_pupil"):
                    result = wfs.optimize_pupil()
                    report.add_key_value("Optimization Result", str(result))

            # Test 9: MLA info
            with TestWithReport(report, "MLA Information"):
                if hasattr(wfs, "get_mla_name"):
                    mla_name = wfs.get_mla_name()
                    report.add_key_value("MLA Name", str(mla_name))

            # Test 10: Frame rate / timing
            with TestWithReport(report, "Frame Timing"):
                n_frames = 50
                times = []
                for _ in range(n_frames):
                    start = time.perf_counter()
                    _ = wfs.get_spotfiled_image()
                    times.append(time.perf_counter() - start)

                fps = n_frames / sum(times)
                avg_ms = np.mean(times) * 1000

                report.add_key_value("Frames", str(n_frames))
                report.add_key_value("Effective FPS", f"{fps:.1f}")
                report.add_key_value("Avg Frame Time", f"{avg_ms:.1f} ms")

                # Timing plot
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(np.arange(n_frames), np.array(times) * 1000, "o-", markersize=3)
                ax.axhline(
                    float(avg_ms),
                    color="r",
                    linestyle="--",
                    label=f"Mean: {avg_ms:.1f} ms",
                )
                ax.set_xlabel("Frame")
                ax.set_ylabel("Time (ms)")
                ax.set_title("Frame Acquisition Timing")
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                report.add_plot(fig, "frame_timing", "Frame acquisition timing")

            # Summary
            report.add_section("Summary", 2)
            report.add_text("Thorlabs WFS hardware tests completed.")
            report.write_summary_table()
            report.save()
            print(f"\nReport saved to: {report.report_path}")

    def test_wfs_calibration_report(self, ThorlabWFS):
        """WFS calibration test with SLM interaction (requires SLM)."""
        pytest.skip("Requires SLM hardware for full calibration")


def test_wfs_standalone_report():
    """Standalone test that can run without pytest (simulation mode)."""
    with TestReport("wfs") as report:
        report.add_section("WFS Simulation Test", 2)
        report.add_text("This test runs without hardware using simulated data.")

        with TestWithReport(report, "Simulated Spotfield"):
            # Create synthetic spotfield
            height, width = 512, 512
            spotfield = np.zeros((height, width), dtype=np.uint16)

            # Grid of spots
            n_spots_x, n_spots_y = 16, 16
            spot_spacing = 32
            spot_radius = 4

            for i in range(n_spots_y):
                for j in range(n_spots_x):
                    cy = i * spot_spacing + spot_spacing // 2
                    cx = j * spot_spacing + spot_spacing // 2
                    # Gaussian spot
                    y, x = np.ogrid[
                        cy - spot_radius : cy + spot_radius + 1,
                        cx - spot_radius : cx + spot_radius + 1,
                    ]
                    gaussian = np.exp(-(x**2 + y**2) / (2 * 1.5**2))
                    spotfield[
                        cy - spot_radius : cy + spot_radius + 1,
                        cx - spot_radius : cx + spot_radius + 1,
                    ] += (gaussian * 1000).astype(np.uint16)

            # Add noise
            spotfield = spotfield + np.random.normal(0, 20, spotfield.shape).astype(
                np.uint16
            )
            spotfield = np.clip(spotfield, 0, 65535)

            # Simulated Zernike
            n_modes = 30
            zernike_true = np.zeros(n_modes)
            zernike_true[3] = 0.1  # Defocus
            zernike_true[4] = 0.05  # Astigmatism
            zernike_true[5] = -0.03  # Astigmatism
            zernike_true[6] = 0.02  # Coma
            zernike_true[7] = 0.01  # Coma

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes[0, 0].imshow(spotfield, cmap="gray")
            axes[0, 0].set_title("Simulated Spotfield")
            axes[0, 0].axis("off")

            axes[0, 1].hist(spotfield.flatten(), bins=50, edgecolor="black")
            axes[0, 1].set_title("Intensity Histogram")
            axes[0, 1].grid(True)

            axes[1, 0].bar(range(n_modes), zernike_true)
            axes[1, 0].set_title("True Zernike Coefficients")
            axes[1, 0].set_xlabel("Noll Index")
            axes[1, 0].set_ylabel("Coefficient (waves)")
            axes[1, 0].grid(True)

            # Simulated measured (with noise)
            zernike_measured = zernike_true + np.random.normal(0, 0.005, n_modes)
            axes[1, 1].bar(
                range(n_modes), zernike_measured, alpha=0.7, label="Measured"
            )
            axes[1, 1].bar(
                range(n_modes), zernike_true, alpha=0.5, label="True", width=0.5
            )
            axes[1, 1].set_title("Measured vs True")
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            plt.tight_layout()
            report.add_plot(fig, "simulated_wfs", "Simulated WFS data")

        report.save()
        print(f"Report saved to: {report.report_path}")


if __name__ == "__main__":
    test_wfs_standalone_report()
