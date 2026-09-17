"""Test report generator utility for hardware device testing.

Generates markdown reports with embedded images for device-specific tests.
Reports are saved to docs/<device>/<device>_report.md
"""

from __future__ import annotations

import datetime
import os
import shutil
from pathlib import Path
from typing import Any


class TestReport:
    """Generates markdown test reports with images for hardware devices."""

    def __init__(self, device_name: str, device_dir: str | None = None):
        """Initialize test report for a device.

        Args:
            device_name: Name of the device (e.g., 'slm-200', 'micro-dm', 'miicam', 'wfs')
            device_dir: Optional custom docs directory. Defaults to docs/<device_name>/
        """
        self.device_name = device_name
        self.project_root = Path(__file__).parents[3]  # AO-shaping root
        if device_dir:
            self.docs_dir = self.project_root / device_dir
        else:
            self.docs_dir = self.project_root / "docs" / device_name

        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.docs_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.report_path = self.docs_dir / f"{device_name}_report.md"
        self.sections: list[str] = []
        self.image_counter = 0

        # Report header
        self._add_header()

    def _add_header(self) -> None:
        """Add report header with timestamp and device info."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.sections.append(f"# {self.device_name.upper()} Hardware Test Report\n")
        self.sections.append(f"**Generated:** {now}\n")
        self.sections.append(f"**Device:** {self.device_name}\n")
        self.sections.append("---\n")

    def add_section(self, title: str, level: int = 2) -> None:
        """Add a section header.

        Args:
            title: Section title
            level: Header level (2=##, 3=###, etc.)
        """
        prefix = "#" * level
        self.sections.append(f"\n{prefix} {title}\n")

    def add_text(self, text: str) -> None:
        """Add plain text content."""
        self.sections.append(f"{text}\n")

    def add_table(self, headers: list[str], rows: list[list[Any]]) -> None:
        """Add a markdown table.

        Args:
            headers: Column headers
            rows: Table rows (list of lists)
        """
        header_row = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_rows = [
            "| " + " | ".join(str(cell) for cell in row) + " |" for row in rows
        ]
        self.sections.append(
            "\n" + header_row + "\n" + separator + "\n" + "\n".join(data_rows) + "\n"
        )

    def add_image(
        self,
        src_path: str | Path,
        caption: str = "",
        alt_text: str | None = None,
        copy_to_images: bool = True,
    ) -> str:
        """Add an image to the report.

        Args:
            src_path: Path to source image
            caption: Image caption
            alt_text: Alt text for markdown (defaults to caption)
            copy_to_images: Whether to copy image to docs/<device>/images/

        Returns:
            Relative path to image in report
        """
        src = Path(src_path)
        if not src.exists():
            self.sections.append(f"\n**Warning: Image not found: {src}**\n")
            return ""

        self.image_counter += 1

        if copy_to_images:
            # Copy to images directory with counter prefix
            ext = src.suffix
            dst_name = f"{self.image_counter:03d}_{src.stem}{ext}"
            dst_path = self.images_dir / dst_name
            shutil.copy2(src, dst_path)
            rel_path = f"images/{dst_name}"
        else:
            rel_path = str(src)

        alt = alt_text or caption or f"Image {self.image_counter}"
        self.sections.append(f"\n![{alt}]({rel_path})\n")
        if caption:
            self.sections.append(f"*{caption}*\n")

        return rel_path

    def add_plot(
        self,
        fig,
        filename: str,
        caption: str = "",
        dpi: int = 150,
        close_fig: bool = True,
    ) -> str:
        """Save a matplotlib figure and add to report.

        Args:
            fig: Matplotlib figure
            filename: Output filename (without extension)
            caption: Image caption
            dpi: Image DPI
            close_fig: Whether to close figure after saving

        Returns:
            Relative path to image in report
        """
        self.image_counter += 1
        fname = f"{self.image_counter:03d}_{filename}.png"
        img_path = self.images_dir / fname
        fig.savefig(img_path, dpi=dpi, bbox_inches="tight")
        if close_fig:
            import matplotlib.pyplot as plt

            plt.close(fig)

        rel_path = f"images/{fname}"
        alt = caption or filename
        self.sections.append(f"\n![{alt}]({rel_path})\n")
        if caption:
            self.sections.append(f"*{caption}*\n")

        return rel_path

    def add_code_block(self, code: str, language: str = "python") -> None:
        """Add a code block."""
        self.sections.append(f"\n```{language}\n{code}\n```\n")

    def add_list(self, items: list[str], ordered: bool = False) -> None:
        """Add a bullet or numbered list."""
        for i, item in enumerate(items):
            prefix = f"{i + 1}. " if ordered else "- "
            self.sections.append(f"{prefix}{item}\n")

    def add_key_value(self, key: str, value: Any) -> None:
        """Add a key-value pair (bold key)."""
        self.sections.append(f"\n**{key}:** {value}\n")

    def add_divider(self) -> None:
        """Add horizontal rule."""
        self.sections.append("\n---\n")

    def add_summary_table(
        self,
        test_name: str,
        status: str,
        duration: float | None = None,
        details: str = "",
    ) -> None:
        """Add a test summary row (for summary table)."""
        if not hasattr(self, "_summary_rows"):
            self._summary_rows = []
        self._summary_rows.append(
            [test_name, status, f"{duration:.2f}s" if duration else "N/A", details]
        )

    def write_summary_table(self) -> None:
        """Write the summary table if any rows were added."""
        if hasattr(self, "_summary_rows") and self._summary_rows:
            self.add_section("Test Summary", 2)
            self.add_table(
                ["Test", "Status", "Duration", "Details"],
                self._summary_rows,
            )

    def save(self) -> Path:
        """Save the report to markdown file."""
        content = "".join(self.sections)
        self.report_path.write_text(content, encoding="utf-8")
        return self.report_path

    def __enter__(self) -> "TestReport":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.save()


def create_device_report(device_name: str) -> TestReport:
    """Factory function to create a test report for a device.

    Args:
        device_name: Device name (e.g., 'slm-200', 'micro-dm', 'miicam', 'wfs')

    Returns:
        TestReport instance
    """
    return TestReport(device_name)


# Convenience function for quick plot + report
def save_plot_to_report(report: TestReport, fig, name: str, caption: str = "") -> str:
    """Save a matplotlib plot to the report's images directory and add reference.

    Args:
        report: TestReport instance
        fig: Matplotlib figure
        name: Base filename
        caption: Caption

    Returns:
        Relative image path
    """
    return report.add_plot(fig, name, caption)


# Context manager for test with automatic report
class TestWithReport:
    """Context manager for a test that generates a report section."""

    def __init__(self, report: TestReport, test_name: str):
        self.report = report
        self.test_name = test_name
        self.start_time: float | None = None

    def __enter__(self) -> "TestWithReport":
        import time

        self.start_time = time.perf_counter()
        self.report.add_section(self.test_name, 3)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import time

        duration = time.perf_counter() - self.start_time if self.start_time else 0
        status = "✅ PASS" if exc_type is None else "❌ FAIL"
        details = str(exc_val) if exc_val else ""
        self.report.add_key_value("Status", status)
        self.report.add_key_value("Duration", f"{duration:.2f}s")
        if details:
            self.report.add_key_value("Error", details)
        self.report.add_divider()


def pytest_html_report_hook(report: TestReport, item, call):
    """Pytest hook to automatically add test results to report.

    Usage in conftest.py:
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(item, call):
            outcome = yield
            rep = outcome.get_result()
            if hasattr(item, 'test_report'):
                pytest_html_report_hook(item.test_report, item, call)
    """
    if call.when == "call":
        status = "✅ PASS" if call.excinfo is None else "❌ FAIL"
        duration = call.duration if hasattr(call, "duration") else 0
        details = str(call.excinfo.value) if call.excinfo else ""
        report.add_summary_table(item.name, status, duration, details)


def generate_test_script_template(device_name: str, test_functions: list[str]) -> str:
    """Generate a test script template for a device.

    Args:
        device_name: Device name
        test_functions: List of test function names to include

    Returns:
        Python script as string
    """
    template = f'''"""Test script for {device_name} - generates visual report."""

import matplotlib.pyplot as plt
import numpy as np

from tests.ao_shaping.utils.test_report import TestReport, TestWithReport


def test_{device_name}_basic():
    """Basic functionality test."""
    with TestReport("{device_name}") as report:
        with TestWithReport(report, "Device Initialization"):
            # TODO: Add initialization test
            pass

        with TestWithReport(report, "Parameter Configuration"):
            # TODO: Add parameter test
            pass

        # Add plots
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Example Plot")
        report.add_plot(fig, "example_plot", "Example test plot")

        report.add_summary_table()
        report.save()


if __name__ == "__main__":
    test_{device_name}_basic()
'''
    return template
