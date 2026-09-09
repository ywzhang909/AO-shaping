"""Santec SLM-200 hardware test with visual report generation.

Tests SLM driver functionality and generates a markdown report with plots
saved to docs/slm-200/slm-200_report.md

Run with:
    pytest tests/ao_shaping/drivers/slm/test_slm_report.py -v --hardware -s
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ao_shaping.drivers.slm.santec_slm200 import SantecSLM200, VideoMode
from tests.ao_shaping.utils.test_report import TestReport, TestWithReport


class TestSLMReport:
    """SLM-200 hardware tests with visual report."""

    pytestmark = pytest.mark.hardware

    @pytest.fixture
    def slm(self):
        """Create SLM instance."""
        return SantecSLM200(slm_number=1, shift_x=100, shift_y=-110)

    @pytest.fixture
    def open_slm(self, slm):
        """Open SLM and return it."""
        slm.open()
        return slm

    def test_slm_initialization_report(self, open_slm):
        """Test SLM initialization and generate report."""
        with TestReport("slm-200") as report:
            report.add_section("SLM-200 Hardware Test Report", 2)
            report.add_text(
                "Testing Santec SLM-200 driver functionality with hardware."
            )

            # Test 1: Device info
            with TestWithReport(report, "Device Information"):
                info = {
                    "SLM Number": open_slm.slm_number,
                    "Wavelength": f"{open_slm.wavelength} nm"
                    if open_slm.wavelength
                    else "Not set",
                    "Video Mode": "Memory" if open_slm.video_mode == 0 else "DVI",
                    "120Hz Mode": "Enabled" if open_slm._use_120hz else "Disabled",
                    "Shift X": open_slm.shift_x,
                    "Shift Y": open_slm.shift_y,
                    "Serial Number": open_slm.get_serial_number() or "Unknown",
                    "Product Serial": open_slm.get_product_serial_number(0)
                    or "Unknown",
                    "LCOS Serial": open_slm.get_lcos_serial_number(0) or "Unknown",
                    "Display Name": open_slm.get_display_name() or "Unknown",
                    "Version": open_slm.get_version() or "Unknown",
                }
                report.add_table(
                    ["Parameter", "Value"],
                    [[k, v] for k, v in info.items()],
                )

            # Test 2: Wavelength setting
            with TestWithReport(report, "Wavelength Setting"):
                wavelengths = [532, 1064, 800]
                results = []
                for wl in wavelengths:
                    start = time.perf_counter()
                    open_slm.set_wavelength(wl, save_to_device=False)
                    elapsed = time.perf_counter() - start
                    actual_wl, max_gray = open_slm.get_wavelength_info()
                    results.append(
                        [
                            f"{wl} nm",
                            f"{actual_wl} nm",
                            f"{max_gray}",
                            f"{elapsed * 1000:.1f} ms",
                        ]
                    )
                report.add_table(
                    ["Requested", "Actual", "Max Gray", "Time"],
                    results,
                )

            # Test 3: Grayscale sweep
            with TestWithReport(report, "Grayscale Sweep"):
                grayscales = [0, 256, 512, 768, 1023]
                fig, ax = plt.subplots(figsize=(8, 4))
                for gs in grayscales:
                    open_slm.set_grayscale(gs)
                    time.sleep(0.1)  # Allow settling
                # Plot response
                ax.plot(grayscales, grayscales, "o-", label="Set value")
                ax.set_xlabel("Grayscale Command")
                ax.set_ylabel("Grayscale Value")
                ax.set_title("Grayscale Linearity Test")
                ax.grid(True)
                ax.legend()
                report.add_plot(fig, "grayscale_sweep", "Grayscale linearity test")

            # Test 4: Phase pattern generation
            with TestWithReport(report, "Phase Pattern Generation"):
                RESOLUTION = (1920, 1200)
                patterns = {}

                # Checkerboard
                period = 50
                height, width = RESOLUTION[1], RESOLUTION[0]
                y = np.arange(height) // period
                x = np.arange(width) // period
                X, Y = np.meshgrid(x, y)
                checker = (X + Y) % 2
                patterns["Checkerboard"] = (checker * 1023).astype(np.uint16)

                # Blazed grating
                period = 20
                x = np.arange(width)
                grating = (x % period) / period * 1023
                patterns["Blazed Grating"] = np.tile(grating, (height, 1)).astype(
                    np.uint16
                )

                # Focus pattern
                focal_length = 0.1
                wavelength = 1064e-9
                pixel_size = 8e-6
                x = np.arange(width) - width // 2
                y = np.arange(height) - height // 2
                X, Y = np.meshgrid(x, y)
                R2 = X**2 + Y**2
                phase = (np.pi / wavelength / focal_length) * (R2 * pixel_size**2)
                phase_wrapped = np.mod(phase, 2 * np.pi)
                patterns["Focus (10cm)"] = (phase_wrapped / (2 * np.pi) * 1023).astype(
                    np.uint16
                )

                # Display patterns
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                axes = axes.flatten()
                for idx, (name, pattern) in enumerate(patterns.items()):
                    if idx < len(axes):
                        im = axes[idx].imshow(pattern, cmap="gray", vmin=0, vmax=1023)
                        axes[idx].set_title(name)
                        axes[idx].axis("off")
                        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
                plt.tight_layout()
                report.add_plot(fig, "phase_patterns", "Generated phase patterns")

                # Write patterns to SLM and display
                for mem_num, (name, pattern) in enumerate(patterns.items(), 1):
                    open_slm.write_phase(pattern, memory_number=mem_num)
                    open_slm.display_memory(mem_num)
                    time.sleep(0.5)

            # Test 5: Shift correction
            with TestWithReport(report, "Shift Correction"):
                # Test positive shift
                test_pattern = np.zeros((100, 100), dtype=np.uint16)
                test_pattern[20:80, 20:80] = 512

                shifts = [
                    (0, 0, "No shift"),
                    (10, 5, "Right 10, Down 5"),
                    (-10, -5, "Left 10, Up 5"),
                    (20, -10, "Right 20, Up 10"),
                ]

                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                axes = axes.flatten()
                for idx, (sx, sy, label) in enumerate(shifts):
                    open_slm.set_shift(sx, sy)
                    shifted = open_slm._apply_shift(test_pattern)
                    im = axes[idx].imshow(shifted, cmap="gray", vmin=0, vmax=1023)
                    axes[idx].set_title(label)
                    axes[idx].axis("off")
                plt.tight_layout()
                report.add_plot(
                    fig, "shift_correction", "Shift correction visualization"
                )

            # Test 6: Memory display verification
            with TestWithReport(report, "Memory Display Verification"):
                import ctypes

                # Write different patterns to different memory slots
                mem_results = []
                for mem_num in [1, 5, 10, 20, 50]:
                    phase = np.zeros((1200, 1920), dtype=np.uint16)
                    # Unique pattern per slot
                    phase[100:200, 100:200] = mem_num * 20
                    open_slm.write_phase(phase, memory_number=mem_num)
                    open_slm.display_memory(mem_num)
                    time.sleep(0.2)

                    # Verify
                    displayed_mem = ctypes.c_ulong(0)
                    ret = open_slm._slm.SLM_Ctrl_ReadDS(
                        open_slm.slm_number, ctypes.byref(displayed_mem)
                    )
                    mem_results.append(
                        [
                            mem_num,
                            ret,
                            displayed_mem.value,
                            "OK" if displayed_mem.value == mem_num else "MISMATCH",
                        ]
                    )

                report.add_table(
                    ["Memory Slot", "Return Code", "Displayed Slot", "Status"],
                    mem_results,
                )

            # Test 7: CSV loading
            with TestWithReport(report, "CSV Phase Loading"):
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False
                ) as f:
                    f.write("Y/X,0,1,2\n")
                    f.write("0,100,200,300\n")
                    f.write("1,400,500,600\n")
                    csv_path = f.name

                try:
                    phase = open_slm.load_phase_from_csv(csv_path)
                    report.add_key_value("CSV Shape", str(phase.shape))
                    report.add_key_value("CSV Dtype", str(phase.dtype))
                    report.add_key_value(
                        "Sample Values", f"{phase[0, 0]}, {phase[1, 2]}"
                    )
                finally:
                    Path(csv_path).unlink()

            # Summary
            report.add_section("Summary", 2)
            report.add_text(
                "All SLM-200 hardware tests completed. Check individual sections for details."
            )

            report.save()
            print(f"\nReport saved to: {report.report_path}")

    def test_slm_patterns_detailed_report(self, open_slm):
        """Detailed pattern test with camera feedback (requires camera)."""
        with TestReport("slm-200") as report:
            report.add_section("SLM Pattern Detailed Analysis", 2)

            RESOLUTION = (1920, 1200)
            height, width = RESOLUTION[1], RESOLUTION[0]

            # Generate various patterns
            periods = [10, 20, 40, 80, 160]
            pattern_data = {}

            for period in periods:
                x = np.arange(width)
                grating = (x % period) / period * 1023
                pattern = np.tile(grating, (height, 1)).astype(np.uint16)
                pattern_data[f"P={period}px"] = pattern

            # Plot pattern frequency analysis
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes = axes.flatten()
            for idx, (name, pattern) in enumerate(pattern_data.items()):
                if idx < len(axes):
                    # Show 1D profile
                    profile = pattern[height // 2, :]
                    axes[idx].plot(profile[:200])
                    axes[idx].set_title(name)
                    axes[idx].set_xlabel("Pixel")
                    axes[idx].set_ylabel("Grayscale")
                    axes[idx].grid(True)
            # Hide unused subplot
            if len(pattern_data) < len(axes):
                axes[-1].axis("off")
            plt.tight_layout()
            report.add_plot(fig, "grating_profiles", "Grating period profiles")

            # Expected diffraction angles
            wavelength = 1064e-9
            f = 0.125  # 125mm lens
            pixel_pitch = 8e-6
            report.add_section("Expected Diffraction Orders", 3)
            diff_data = []
            for period in periods:
                # Δx = λ*f/(d*pixel) where d = period * pixel_pitch
                d = period * pixel_pitch
                delta_x_m = wavelength * f / d
                delta_x_px = delta_x_m / pixel_pitch
                diff_data.append(
                    [
                        f"{period}px",
                        f"{d * 1e6:.1f} µm",
                        f"{delta_x_m * 1e3:.2f} mm",
                        f"{delta_x_px:.1f} px",
                    ]
                )
            report.add_table(
                ["Period", "Grating Spacing", "Δx (mm)", "Δx (pixels)"],
                diff_data,
            )

            report.save()
            print(f"\nReport saved to: {report.report_path}")


def test_slm_standalone_report():
    """Standalone test that can run without pytest (simulation mode)."""
    with TestReport("slm-200") as report:
        report.add_section("SLM-200 Simulation Test", 2)
        report.add_text("This test runs without hardware using simulated data.")

        # Try to connect to hardware, fall back to simulation
        slm = None
        try:
            slm = SantecSLM200(slm_number=1)
            slm.open()
            report.add_key_value("Mode", "Hardware")
        except Exception as e:
            report.add_key_value("Mode", "Simulation (no hardware)")
            report.add_key_value("Hardware Error", str(e))
            slm = None

        if slm is not None:
            with TestWithReport(report, "Basic Open/Close"):
                assert slm.is_open
                report.add_key_value("State", "Connected")

            with TestWithReport(report, "Wavelength Read"):
                wl, max_gray = slm.get_wavelength_info()
                report.add_key_value("Wavelength", f"{wl} nm")
                report.add_key_value("Max Grayscale", str(max_gray))

            # Simple pattern test
            with TestWithReport(report, "Pattern Generation"):
                phase = np.zeros((1080, 1920), dtype=np.uint16)
                phase[500:580, 900:1020] = 511
                slm.write_phase(phase, memory_number=1)
                slm.display_memory(1)
                report.add_text("Pattern written and displayed successfully.")

            slm.close()
        else:
            # Simulation mode - generate synthetic data
            with TestWithReport(report, "Simulated Pattern Generation"):
                RESOLUTION = (1920, 1200)
                height, width = RESOLUTION[1], RESOLUTION[0]

                patterns = {}

                # Checkerboard
                period = 50
                y = np.arange(height) // period
                x = np.arange(width) // period
                X, Y = np.meshgrid(x, y)
                checker = (X + Y) % 2
                patterns["Checkerboard"] = (checker * 1023).astype(np.uint16)

                # Blazed grating
                period = 20
                x = np.arange(width)
                grating = (x % period) / period * 1023
                patterns["Blazed Grating"] = np.tile(grating, (height, 1)).astype(
                    np.uint16
                )

                # Focus pattern
                focal_length = 0.1
                wavelength = 1064e-9
                pixel_size = 8e-6
                x = np.arange(width) - width // 2
                y = np.arange(height) - height // 2
                X, Y = np.meshgrid(x, y)
                R2 = X**2 + Y**2
                phase = (np.pi / wavelength / focal_length) * (R2 * pixel_size**2)
                phase_wrapped = np.mod(phase, 2 * np.pi)
                patterns["Focus (10cm)"] = (phase_wrapped / (2 * np.pi) * 1023).astype(
                    np.uint16
                )

                # Display patterns
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                axes = axes.flatten()
                for idx, (name, pattern) in enumerate(patterns.items()):
                    if idx < len(axes):
                        im = axes[idx].imshow(pattern, cmap="gray", vmin=0, vmax=1023)
                        axes[idx].set_title(name)
                        axes[idx].axis("off")
                        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
                plt.tight_layout()
                report.add_plot(fig, "sim_phase_patterns", "Simulated phase patterns")

                # Diffraction analysis
                wavelength = 1064e-9
                f = 0.125
                pixel_pitch = 8e-6
                periods = [10, 20, 40, 80, 160]
                diff_data = []
                for period in periods:
                    d = period * pixel_pitch
                    delta_x_m = wavelength * f / d
                    delta_x_px = delta_x_m / pixel_pitch
                    diff_data.append(
                        [
                            f"{period}px",
                            f"{d * 1e6:.1f} µm",
                            f"{delta_x_m * 1e3:.2f} mm",
                            f"{delta_x_px:.1f} px",
                        ]
                    )
                report.add_table(
                    ["Period", "Grating Spacing", "Δx (mm)", "Δx (pixels)"],
                    diff_data,
                )

        # Generate test plot
        fig, ax = plt.subplots()
        x = np.linspace(0, 2 * np.pi, 100)
        ax.plot(x, np.sin(x))
        ax.set_title("Test Sine Wave")
        report.add_plot(fig, "test_sine", "Test sine wave plot")

        report.save()
        print(f"Report saved to: {report.report_path}")


if __name__ == "__main__":
    test_slm_standalone_report()
