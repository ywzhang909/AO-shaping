"""MIICAM SDK path finding and setup utilities."""

import os
import sys
import ctypes

from loguru import logger

from ao_shaping.utils.file import ROOT_DIR


def _find_miicam_sdk_path() -> str | None:
    """Find the MIICAM SDK path by checking multiple possible locations.

    Checks in order:
    1. Environment variable MIICAM_SDK_PATH (user-configurable)
    2. Bundled in project (src/ao_shaping/drivers/ccd/_miicam_sdk)
    3. External libs directory (libs/miicamsdk.20240728/python)

    Returns:
        str | None: Path to SDK if found, None otherwise.
    """
    # Option 0: Environment variable (highest priority, user-configurable)
    env_sdk_path = os.environ.get("MIICAM_SDK_PATH")
    if env_sdk_path and os.path.isdir(env_sdk_path):
        return env_sdk_path

    _project_root = str(ROOT_DIR)

    # Deduplicate paths while preserving order
    seen: set[str] = set()
    _MII_SDK_PATHS: list[str] = []
    for path in [
        # Option 1: Bundled in project (for development)
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_miicam_sdk"),
        # Option 2: External libs directory
        os.path.join(_project_root, "libs", "miicamsdk.20240728", "python"),
    ]:
        if path not in seen:
            seen.add(path)
            _MII_SDK_PATHS.append(path)

    for path in _MII_SDK_PATHS:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "miicam.py")):
            return path
    return None


def _setup_miicam_sdk() -> bool:
    """Set up the MIICAM SDK by adding its path to sys.path and DLL search path.

    Returns:
        bool: True if SDK was found and set up successfully, False otherwise.
    """
    sdk_path = _find_miicam_sdk_path()
    if sdk_path is None:
        logger.error("MIICAM SDK not found")
        return False

    if sdk_path not in sys.path:
        sys.path.append(sdk_path)

    # Add SDK path to DLL search path (Windows)
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(sdk_path)
        except Exception as e:
            logger.warning(f"Failed to add DLL directory: {e}")

    # Pre-load the SDK DLL if available
    dll_path = os.path.join(sdk_path, "MIIUSB.dll")
    if os.path.exists(dll_path):
        try:
            ctypes.CDLL(dll_path)
        except Exception as e:
            logger.warning(f"Failed to pre-load MIIUSB.dll: {e}")

    logger.debug(f"MIICAM SDK set up successfully from: {sdk_path}")
    return True
