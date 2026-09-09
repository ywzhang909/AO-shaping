#!/usr/bin/env python
"""Test runner script to generate all device test reports.

This script runs all device-specific test reports and generates markdown files
with embedded images in the docs directory.

Usage:
    python tests/ao_shaping/utils/run_all_reports.py [--device DEVICE] [--simulate]

Options:
    --device DEVICE    Run only specific device (slm-200, micro-dm, miicam, wfs)
    --simulate         Run in simulation mode (no hardware required)
    --help             Show this help message
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def run_test_module(module_path: str, device_name: str, simulate: bool = False) -> bool:
    """Run a test module and return success status."""
    print(f"\n{'=' * 60}")
    print(f"Running {device_name} test report...")
    print(f"{'=' * 60}")

    try:
        # Import and run the standalone test function
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        if spec is None or spec.loader is None:
            print(f"✗ Could not load module: {module_path}")
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Look for standalone test function
        if hasattr(module, f"test_{device_name.replace('-', '_')}_standalone_report"):
            func = getattr(
                module, f"test_{device_name.replace('-', '_')}_standalone_report"
            )
            func()
            print(f"✓ {device_name} report generated successfully")
            return True
        elif hasattr(module, "test_slm_standalone_report") and device_name == "slm-200":
            module.test_slm_standalone_report()
            print(f"✓ {device_name} report generated successfully")
            return True
        elif (
            hasattr(module, "test_micro_dm_simulation_report")
            and device_name == "micro-dm"
        ):
            module.test_micro_dm_simulation_report()
            print(f"✓ {device_name} report generated successfully")
            return True
        elif (
            hasattr(module, "test_miicam_standalone_report") and device_name == "miicam"
        ):
            module.test_miicam_standalone_report()
            print(f"✓ {device_name} report generated successfully")
            return True
        elif hasattr(module, "test_wfs_standalone_report") and device_name == "wfs":
            module.test_wfs_standalone_report()
            print(f"✓ {device_name} report generated successfully")
            return True
        else:
            print(f"✗ No standalone test function found for {device_name}")
            return False

    except Exception as e:
        print(f"✗ {device_name} test failed: {e}")
        if not simulate:
            import traceback

            traceback.print_exc()
        return False


def run_pytest_test(module_path: str, device_name: str) -> bool:
    """Run tests via pytest with hardware marker."""
    print(f"\n{'=' * 60}")
    print(f"Running {device_name} hardware tests via pytest...")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                module_path,
                "-v",
                "--hardware",
                "-s",
                "--tb=short",
            ],
            capture_output=False,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"✗ {device_name} pytest failed: {e}")
        return False


def check_docs_dir(device_name: str) -> Path:
    """Check and return docs directory for device."""
    project_root = Path(__file__).parents[3]
    docs_dir = project_root / "docs" / device_name
    docs_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir


def list_generated_reports() -> None:
    """List all generated report files."""
    project_root = Path(__file__).parents[3]
    docs_dir = project_root / "docs"

    print(f"\n{'=' * 60}")
    print("GENERATED REPORTS")
    print(f"{'=' * 60}")

    for device_dir in sorted(docs_dir.iterdir()):
        if device_dir.is_dir():
            report_files = list(device_dir.glob("*_report.md"))
            image_dirs = list(device_dir.glob("images"))
            if report_files:
                for rf in report_files:
                    size = rf.stat().st_size
                    print(f"  📄 {device_dir.name}/{rf.name} ({size:,} bytes)")
            if image_dirs:
                for img_dir in image_dirs:
                    img_count = len(list(img_dir.glob("*.png")))
                    print(f"  🖼️  {device_dir.name}/images/ ({img_count} images)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate device test reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_all_reports.py                    # Run all simulation reports
    python run_all_reports.py --device slm-200   # Run only SLM report
    python run_all_reports.py --simulate         # Force simulation mode
    python run_all_reports.py --pytest           # Run hardware tests via pytest
        """,
    )
    parser.add_argument(
        "--device",
        choices=["slm-200", "micro-dm", "miicam", "wfs"],
        help="Run only specific device test",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in simulation mode (no hardware required)",
    )
    parser.add_argument(
        "--pytest",
        action="store_true",
        help="Run hardware tests via pytest (requires --hardware marker)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List generated reports and exit"
    )

    args = parser.parse_args()

    project_root = Path(__file__).parents[3]

    # Test modules mapping
    test_modules = {
        "slm-200": "tests/ao_shaping/drivers/slm/test_slm_report.py",
        "micro-dm": "tests/ao_shaping/drivers/dm/test_micro_dm_report.py",
        "miicam": "tests/ao_shaping/drivers/ccd/test_miicam_report.py",
        "wfs": "tests/ao_shaping/drivers/wfs/test_wfs_report.py",
    }

    if args.list:
        list_generated_reports()
        return

    devices_to_run = [args.device] if args.device else list(test_modules.keys())

    results = {}
    for device in devices_to_run:
        module_path = project_root / test_modules[device]

        if not module_path.exists():
            print(f"✗ Module not found: {module_path}")
            results[device] = False
            continue

        # Ensure docs directory exists
        check_docs_dir(device)

        if args.pytest:
            success = run_pytest_test(str(module_path), device)
        else:
            success = run_test_module(str(module_path), device, args.simulate)

        results[device] = success

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for device, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {device}: {status}")

    # List generated reports
    list_generated_reports()

    # Exit code
    if all(results.values()):
        print("\n✓ All reports generated successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some reports failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
