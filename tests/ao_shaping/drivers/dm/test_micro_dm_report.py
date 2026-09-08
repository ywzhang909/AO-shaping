"""Micro DM hardware test with visual report generation.

Tests Micro DM driver functionality and generates a markdown report with plots
saved to docs/micro-dm/micro-dm_report.md

Run with:
    pytest tests/ao_shaping/drivers/dm/test_micro_dm_report.py -v --hardware -s
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ao_shaping.drivers.dm.MicroDM import (
    MAX_CHANNELS,
    VOLTAGE_MAX,
    VOLTAGE_MIN,
    MicroDM,
    voltages_to_payload,
)
from ao_shaping.drivers.sim.dm import SimMicroDM
from tests.ao_shaping.utils.test_report import TestReport, TestWithReport

pytestmark = pytest.mark.hardware


class TestMicroDMReport:
    """Micro DM hardware tests with visual report."""

    @pytest.fixture
    def dm(self):
        """Create DM instance (hardware or simulation)."""
        # Try hardware first, fall back to simulation
        try:
            dm = MicroDM()
            dm.open()
            if dm.is_connected():
                return dm
        except Exception:
            pass
        # Fallback to simulation
        return SimMicroDM()

    @pytest.fixture
    def sim_dm(self):
        """Create simulation DM."""
        return SimMicroDM()

    def test_micro_dm_basic_report(self, dm):
        """Test Micro DM basic functionality and generate report."""
        with TestReport("micro-dm") as report:
            report.add_section("Micro DM Hardware Test Report", 2)
            report.add_text("Testing Micro Deformable Mirror driver functionality.")

            # Test 1: Device info
            with TestWithReport(report, "Device Information"):
                info = {
                    "Channels (DM_Num)": dm.DM_Num,
                    "Voltage Min": f"{dm.V_Min} V",
                    "Voltage Max": f"{dm.V_Max} V",
                    "Safety Mode": "Enabled"
                    if getattr(dm, "_safety_mode", False)
                    else "Disabled",
                    "Max Neighbor Diff": getattr(dm, "max_neibor_diff", "N/A"),
                    "Type": type(dm).__name__,
                }
                report.add_table(
                    ["Parameter", "Value"],
                    [[k, v] for k, v in info.items()],
                )

            # Test 2: Voltage conversion formula
            with TestWithReport(report, "Voltage Conversion Formula"):
                test_voltages = [-20, -10, 0, 10, 20, 30, 50, 70, 100, 120]
                conv_data = []
                for v in test_voltages:
                    payload = voltages_to_payload(v)
                    conv_data.append(
                        [
                            f"{v} V",
                            payload[0],
                            payload[1],
                            f"0x{payload[0]:02X}{payload[1]:02X}",
                        ]
                    )
                report.add_table(
                    ["Input Voltage", "High Byte", "Low Byte", "Hex Payload"],
                    conv_data,
                )

                # Plot conversion curve
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                voltages = np.linspace(VOLTAGE_MIN, VOLTAGE_MAX, 100)
                high_bytes = []
                low_bytes = []
                for v in voltages:
                    hb, lb = voltages_to_payload(v)
                    high_bytes.append(hb)
                    low_bytes.append(lb)
                ax1.plot(voltages, high_bytes, label="High Byte")
                ax1.plot(voltages, low_bytes, label="Low Byte")
                ax1.set_xlabel("Voltage (V)")
                ax1.set_ylabel("Byte Value")
                ax1.set_title("Voltage to Payload Bytes")
                ax1.legend()
                ax1.grid(True)

                # Combined 16-bit value
                combined = [h * 256 + l for h, l in zip(high_bytes, low_bytes)]
                ax2.plot(voltages, combined)
                ax2.set_xlabel("Voltage (V)")
                ax2.set_ylabel("16-bit Value")
                ax2.set_title("Combined 16-bit Payload")
                ax2.grid(True)
                plt.tight_layout()
                report.add_plot(
                    fig, "voltage_conversion", "Voltage to payload conversion curve"
                )

            # Test 3: Actuator response
            with TestWithReport(report, "Actuator Response"):
                n_channels = dm.DM_Num
                print(f"Testing {n_channels} channels...")

                # Single actuator test
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                axes = axes.flatten()

                # Test individual channels
                test_channels = [
                    0,
                    n_channels // 4,
                    n_channels // 2,
                    3 * n_channels // 4,
                ]
                for idx, ch in enumerate(test_channels):
                    dm.reset_all()
                    dm.set_channel_voltage(ch, 50.0)
                    time.sleep(0.05)
                    positions = dm.get_actuator_positions()
                    axes[idx].plot(positions)
                    axes[idx].axvline(
                        ch, color="r", linestyle="--", label=f"Channel {ch}"
                    )
                    axes[idx].set_title(f"Channel {ch} at 50V")
                    axes[idx].set_xlabel("Channel")
                    axes[idx].set_ylabel("Voltage (V)")
                    axes[idx].legend()
                    axes[idx].grid(True)
                plt.tight_layout()
                report.add_plot(fig, "single_actuator", "Single actuator response")

            # Test 4: Full array response
            with TestWithReport(report, "Full Array Response"):
                # Test pattern: linear gradient
                gradient = np.linspace(VOLTAGE_MIN, VOLTAGE_MAX, n_channels)
                dm.send_voltages(gradient)
                time.sleep(0.1)
                positions = dm.get_actuator_positions()

                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                # Commanded vs actual
                axes[0, 0].plot(gradient, label="Commanded")
                axes[0, 0].plot(positions, label="Actual", alpha=0.7)
                axes[0, 0].set_title("Linear Gradient Response")
                axes[0, 0].set_xlabel("Channel")
                axes[0, 0].set_ylabel("Voltage (V)")
                axes[0, 0].legend()
                axes[0, 0].grid(True)

                # Error
                axes[0, 1].plot(positions - gradient)
                axes[0, 1].set_title("Error (Actual - Commanded)")
                axes[0, 1].set_xlabel("Channel")
                axes[0, 1].set_ylabel("Error (V)")
                axes[0, 1].grid(True)

                # Histogram of errors
                errors = positions - gradient
                axes[1, 0].hist(errors, bins=30, edgecolor="black")
                axes[1, 0].set_title("Error Distribution")
                axes[1, 0].set_xlabel("Error (V)")
                axes[1, 0].set_ylabel("Count")

                # Zernike-like patterns
                from ao_shaping.utils.pattern_helper import PatternHelper

                helper = PatternHelper(
                    (int(np.sqrt(n_channels)), int(np.sqrt(n_channels)))
                )
                zernike_4 = helper.generate_zernike(n=4, m=0, amplitude=20)  # Defocus
                zernike_4 = zernike_4.flatten()
                if len(zernike_4) == n_channels:
                    dm.send_voltages(zernike_4 + 50)  # Offset to mid-range
                    time.sleep(0.1)
                    z_pos = dm.get_actuator_positions()
                    axes[1, 1].plot(zernike_4 + 50, label="Commanded")
                    axes[1, 1].plot(z_pos, label="Actual", alpha=0.7)
                    axes[1, 1].set_title("Zernike Defocus Pattern")
                    axes[1, 1].legend()
                    axes[1, 1].grid(True)

                plt.tight_layout()
                report.add_plot(
                    fig, "full_array_response", "Full array voltage response"
                )

            # Test 5: Safety mode ramping
            if hasattr(dm, "_safety_mode") and dm._safety_mode:
                with TestWithReport(report, "Safety Mode Ramping"):
                    dm.reset_all()
                    time.sleep(0.1)

                    # Large voltage jump
                    target = np.full(n_channels, 80.0)
                    start = time.perf_counter()
                    dm.send_voltages(target)
                    elapsed = time.perf_counter() - start

                    # Check intermediate steps if accessible
                    if hasattr(dm, "apply_log") and dm.apply_log:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        for i, step in enumerate(dm.apply_log):
                            ax.plot(
                                step, alpha=0.5, label=f"Step {i + 1}" if i < 5 else ""
                            )
                        ax.set_title("Safety Ramping Steps")
                        ax.set_xlabel("Channel")
                        ax.set_ylabel("Voltage (V)")
                        ax.grid(True)
                        if len(dm.apply_log) <= 10:
                            ax.legend()
                        plt.tight_layout()
                        report.add_plot(
                            fig, "safety_ramping", "Safety mode voltage ramping steps"
                        )

                    report.add_key_value("Ramp Time", f"{elapsed * 1000:.1f} ms")
                    report.add_key_value(
                        "Steps", str(len(getattr(dm, "apply_log", [])))
                    )

            # Test 6: Neighbor gradient check
            with TestWithReport(report, "Neighbor Gradient Check"):
                if hasattr(dm, "check_dm_unit_grad_safe"):
                    # Safe pattern
                    safe_pattern = np.full(n_channels, 30.0)
                    safe = dm.check_dm_unit_grad_safe(safe_pattern)
                    report.add_key_value("Uniform 30V Safe", str(safe))

                    # Unsafe pattern - alternating high/low
                    unsafe_pattern = np.zeros(n_channels)
                    unsafe_pattern[::2] = VOLTAGE_MAX
                    unsafe_pattern[1::2] = VOLTAGE_MIN
                    unsafe = dm.check_dm_unit_grad_safe(unsafe_pattern)
                    report.add_key_value("Alternating Max/Min Safe", str(unsafe))

                    # Visualize unsafe pattern
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].plot(safe_pattern)
                    axes[0].set_title("Safe Pattern (Uniform)")
                    axes[0].set_ylim(VOLTAGE_MIN - 10, VOLTAGE_MAX + 10)
                    axes[0].grid(True)

                    axes[1].plot(unsafe_pattern)
                    axes[1].set_title("Unsafe Pattern (Alternating)")
                    axes[1].set_ylim(VOLTAGE_MIN - 10, VOLTAGE_MAX + 10)
                    axes[1].grid(True)
                    plt.tight_layout()
                    report.add_plot(
                        fig, "gradient_check", "Gradient safety check patterns"
                    )

            # Test 7: Relay control
            with TestWithReport(report, "Relay Control"):
                if hasattr(dm, "set_relay_state"):
                    dm.set_relay_state(True)
                    time.sleep(0.1)
                    info = dm.get_hardware_info()
                    report.add_key_value("Relay ON", info.get("relay_state", "Unknown"))

                    dm.set_relay_state(False)
                    time.sleep(0.1)
                    info = dm.get_hardware_info()
                    report.add_key_value(
                        "Relay OFF", info.get("relay_state", "Unknown")
                    )

            # Test 8: Reset behavior
            with TestWithReport(report, "Reset Behavior"):
                # Set random voltages
                random_v = np.random.uniform(VOLTAGE_MIN, VOLTAGE_MAX, n_channels)
                dm.send_voltages(random_v)
                time.sleep(0.1)
                before_reset = dm.get_actuator_positions()

                dm.reset_all()
                time.sleep(0.1)
                after_reset = dm.get_actuator_positions()

                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].plot(before_reset)
                axes[0].set_title("Before Reset")
                axes[0].set_ylim(VOLTAGE_MIN - 10, VOLTAGE_MAX + 10)
                axes[0].grid(True)

                axes[1].plot(after_reset)
                axes[1].set_title("After Reset")
                axes[1].set_ylim(VOLTAGE_MIN - 10, VOLTAGE_MAX + 10)
                axes[1].grid(True)
                plt.tight_layout()
                report.add_plot(fig, "reset_behavior", "Reset behavior")

                max_residual = np.max(np.abs(after_reset))
                report.add_key_value("Max Residual Voltage", f"{max_residual:.2f} V")

            # Summary
            report.add_section("Summary", 2)
            report.add_text("Micro DM hardware tests completed.")
            report.write_summary_table()
            report.save()
            print(f"\nReport saved to: {report.report_path}")

    def test_micro_dm_voltage_calibration_report(self, dm):
        """Voltage calibration test with detailed measurements."""
        with TestReport("micro-dm") as report:
            report.add_section("Micro DM Voltage Calibration", 2)

            n_channels = dm.DM_Num

            # Multi-point calibration
            test_voltages = np.linspace(VOLTAGE_MIN, VOLTAGE_MAX, 11)
            calibration_data = []

            fig, ax = plt.subplots(figsize=(10, 6))

            for ch in [0, n_channels // 2, n_channels - 1]:
                measured = []
                for v_cmd in test_voltages:
                    dm.reset_all()
                    dm.set_channel_voltage(ch, v_cmd)
                    time.sleep(0.05)
                    pos = dm.get_actuator_positions()
                    measured.append(pos[ch])
                calibration_data.append([ch] + measured)
                ax.plot(test_voltages, measured, "o-", label=f"Channel {ch}")

            ax.plot(test_voltages, test_voltages, "k--", label="Ideal")
            ax.set_xlabel("Commanded Voltage (V)")
            ax.set_ylabel("Measured Voltage (V)")
            ax.set_title("Voltage Calibration Curve")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            report.add_plot(
                fig, "voltage_calibration", "Per-channel voltage calibration"
            )

            # Linearity error
            fig, ax = plt.subplots(figsize=(10, 4))
            for ch_idx, ch in enumerate([0, n_channels // 2, n_channels - 1]):
                measured = calibration_data[ch_idx][1:]
                error = np.array(measured) - test_voltages
                ax.plot(test_voltages, error, "o-", label=f"Channel {ch}")
            ax.axhline(0, color="k", linestyle="--")
            ax.set_xlabel("Commanded Voltage (V)")
            ax.set_ylabel("Error (V)")
            ax.set_title("Linearity Error")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            report.add_plot(fig, "linearity_error", "Voltage linearity error")

            report.save()
            print(f"\nReport saved to: {report.report_path}")


def test_micro_dm_simulation_report():
    """Standalone simulation test that runs without hardware."""
    with TestReport("micro-dm") as report:
        report.add_section("Micro DM Simulation Test", 2)

        dm = SimMicroDM()
        n_channels = dm.DM_Num

        with TestWithReport(report, "Simulation Initialization"):
            dm.open()
            report.add_key_value("Channels", str(n_channels))
            report.add_key_value("Voltage Range", f"{dm.V_Min} to {dm.V_Max} V")

        with TestWithReport(report, "Transform Function"):
            # Test transform from [-1, 1] to voltage range
            cmds = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            voltages = dm.transform(cmds)
            expected = np.array([-20.0, 15.0, 50.0, 85.0, 120.0])
            report.add_table(
                ["Command", "Expected (V)", "Actual (V)", "Error (V)"],
                [
                    [f"{c:.1f}", f"{e:.1f}", f"{v:.1f}", f"{abs(v - e):.2f}"]
                    for c, e, v in zip(cmds, expected, voltages)
                ],
            )

        with TestWithReport(report, "Pattern Tests"):
            patterns = {
                "Zero": np.zeros(n_channels),
                "Mid": np.full(n_channels, 50.0),
                "Max": np.full(n_channels, 120.0),
                "Gradient": np.linspace(-20, 120, n_channels),
            }

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()
            for idx, (name, pattern) in enumerate(patterns.items()):
                dm.send_voltages(pattern)
                result = dm.get_actuator_positions()
                axes[idx].plot(pattern, label="Command", linestyle="--")
                axes[idx].plot(result, label="Result", alpha=0.7)
                axes[idx].set_title(name)
                axes[idx].legend()
                axes[idx].grid(True)
            plt.tight_layout()
            report.add_plot(fig, "sim_patterns", "Simulation pattern tests")

        with TestWithReport(report, "Safety Ramping"):
            dm.max_neibor_diff = 10.0
            target = np.full(n_channels, 100.0)
            dm.send_voltages(target)
            if hasattr(dm, "apply_log") and dm.apply_log:
                fig, ax = plt.subplots(figsize=(10, 4))
                for i, step in enumerate(dm.apply_log):
                    ax.plot(step, alpha=0.5)
                ax.set_title("Safety Ramping Steps (Simulation)")
                ax.grid(True)
                plt.tight_layout()
                report.add_plot(fig, "sim_ramping", "Simulation safety ramping")

        report.save()
        print(f"Report saved to: {report.report_path}")


if __name__ == "__main__":
    test_micro_dm_simulation_report()
