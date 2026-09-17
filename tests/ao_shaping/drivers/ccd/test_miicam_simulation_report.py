"""MIICAM CCD camera simulation test report.

Generates a markdown report with simulated camera data for testing without hardware.
Output: docs/miicam/miicam_simulation_report.md

Run with:
    python tests/ao_shaping/drivers/ccd/test_miicam_simulation_report.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tests.ao_shaping.utils.test_report import TestReport, TestWithReport


def test_miicam_simulation_report() -> None:
    """Standalone simulation test that runs without hardware."""
    with TestReport("miicam", device_dir="docs/miicam_simulation") as report:
        report.add_section("MIICAM Simulation Test Report", 2)
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
    test_miicam_simulation_report()
